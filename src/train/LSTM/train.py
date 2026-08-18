import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import numpy as np

from utils import haversine_np
from dataset import preprocess_data, create_sequences, StormDataset
from model import StormLSTM

class HaversineLoss(nn.Module):
    def __init__(self):
        super(HaversineLoss, self).__init__()

    def forward(self, pred_deltas, target_deltas, base_lat_lon):
        batch_size = pred_deltas.shape[0]
        horizons = pred_deltas.shape[1] // 2
        
        base_lat = base_lat_lon[:, 0:1]
        base_lon = base_lat_lon[:, 1:2]
        
        loss = 0.0
        
        for i in range(horizons):
            idx_lat = i * 2
            idx_lon = i * 2 + 1
            
            true_lat = base_lat + target_deltas[:, idx_lat:idx_lat+1]
            true_lon = base_lon + target_deltas[:, idx_lon:idx_lon+1]
            
            pred_lat = base_lat + pred_deltas[:, idx_lat:idx_lat+1]
            pred_lon = base_lon + pred_deltas[:, idx_lon:idx_lon+1]
            
            true_lat_rad = torch.deg2rad(true_lat)
            true_lon_rad = torch.deg2rad(true_lon)
            pred_lat_rad = torch.deg2rad(pred_lat)
            pred_lon_rad = torch.deg2rad(pred_lon)
            
            dlon = pred_lon_rad - true_lon_rad
            dlat = pred_lat_rad - true_lat_rad
            
            a = torch.sin(dlat / 2.0)**2 + \
                torch.cos(true_lat_rad) * torch.cos(pred_lat_rad) * torch.sin(dlon / 2.0)**2
            a = torch.clamp(a, min=1e-7, max=1.0)
            
            c = 2 * torch.asin(torch.sqrt(a))
            dist_km = 6371.0 * c
            
            # Using Mean Absolute Error for robustness to outliers
            loss += torch.mean(dist_km)
            
        return loss / horizons

def evaluate(model, dataloader, last_lat_lon_test, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y, _ in dataloader:
            x = x.to(device)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    res_str = ""
    res_str += "--- Evaluation Results ---\n"
    
    horizons_name = ['6h', '12h', '18h', '24h']
    for i, h in enumerate(horizons_name):
        idx_lat = i * 2
        idx_lon = i * 2 + 1
        
        y_test_lat = all_targets[:, idx_lat]
        y_test_lon = all_targets[:, idx_lon]
        
        pred_lat = all_preds[:, idx_lat]
        pred_lon = all_preds[:, idx_lon]
        
        y_true_vec = np.column_stack((y_test_lat, y_test_lon))
        y_pred_vec = np.column_stack((pred_lat, pred_lon))
        
        dot_product = np.sum(y_true_vec * y_pred_vec, axis=1)
        norm_true = np.linalg.norm(y_true_vec, axis=1)
        norm_pred = np.linalg.norm(y_pred_vec, axis=1)
        
        denom = norm_true * norm_pred
        similarity = np.zeros_like(dot_product)
        valid_idx = denom > 0
        similarity[valid_idx] = dot_product[valid_idx] / denom[valid_idx]
        
        avg_cosine_sim = np.mean(similarity)
        
        current_lat = last_lat_lon_test[:, 0]
        current_lon = last_lat_lon_test[:, 1]
        
        true_final_lat = current_lat + y_test_lat
        true_final_lon = current_lon + y_test_lon
        
        pred_final_lat = current_lat + pred_lat
        pred_final_lon = current_lon + pred_lon
        
        distance_error_km = haversine_np(true_final_lon, true_final_lat, pred_final_lon, pred_final_lat)
        avg_distance_error_km = np.mean(distance_error_km)
        min_distance_error_km = np.min(distance_error_km)
        max_distance_error_km = np.max(distance_error_km)
        
        res = (
            f"Average Cosine Similarity for {h}: {avg_cosine_sim:.4f} | "
            f"Distance Error: Mean = {avg_distance_error_km:.2f} km, "
            f"Min = {min_distance_error_km:.2f} km, "
            f"Max = {max_distance_error_km:.2f} km\n"
        )
        print(res, end="")
        res_str += res
        
    return res_str

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TRAIN_DATA_PATH = os.path.join(BASE_DIR, '..', 'Datasets', 'Storm_data', 'train.csv')
    TEST_DATA_PATH = os.path.join(BASE_DIR, '..', 'Datasets', 'Storm_data', 'test.csv')
    
    print("Loading train and test datasets...")
    df_train = pd.read_csv(TRAIN_DATA_PATH, low_memory=False)
    df_test = pd.read_csv(TEST_DATA_PATH, low_memory=False)
    
    print("Preprocessing data...")
    df_train, feature_cols = preprocess_data(df_train)
    df_test, _ = preprocess_data(df_test)
    
    seq_len = 4
    horizons = 4
    
    print("Creating sequences...")
    X_train, Y_train, last_lat_lon_train = create_sequences(df_train, feature_cols, seq_len, horizons)
    X_test, Y_test, last_lat_lon_test = create_sequences(df_test, feature_cols, seq_len, horizons)
    
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    
    if len(X_train) == 0:
        print("Error: No training sequences found. Check seq_len and horizons.")
        return
    
    # Scale features
    num_features = len(feature_cols)
    scaler = StandardScaler()
    
    # Flatten X_train to 2D for scaling
    X_train_flat = X_train.reshape(-1, num_features)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    
    X_test_flat = X_test.reshape(-1, num_features)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)
    
    train_dataset = StormDataset(X_train_scaled, Y_train, last_lat_lon_train)
    test_dataset = StormDataset(X_test_scaled, Y_test, last_lat_lon_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = StormLSTM(input_dim=num_features, hidden_dim=64, num_layers=2, output_dim=horizons*2).to(device)
    criterion = HaversineLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 50
    print("Training model...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for x, y, base_lat_lon in train_loader:
            x, y, base_lat_lon = x.to(device), y.to(device), base_lat_lon.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y, base_lat_lon)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
    print("Evaluating model...")
    res_str = evaluate(model, test_loader, last_lat_lon_test, device)
    
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_path = os.path.join(MODEL_DIR, 'lstm_model.pth')
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    
    RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_result.txt")
    with open(RESULT_PATH, 'w') as f:
        f.write(res_str)
        
    print("Results and model saved.")

if __name__ == '__main__':
    main()
