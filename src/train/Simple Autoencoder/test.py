import os
import glob
import re
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.decomposition import PCA

# 1. Model Architecture (Must match simple_autoencoder.py)
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(SimpleAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

# 2. PyTorch Dataset
class StormDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.values.astype(np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])

# 3. Masked Loss
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, pred, target):
        mask = (target >= 0.0).float()
        loss = self.mse(pred, target)
        masked_loss = loss * mask
        return masked_loss.sum() / (mask.sum() + 1e-8)

def find_latest_checkpoint(checkpoint_dir):
    """Finds the checkpoint file with the highest epoch number."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    def extract_epoch(fname):
        match = re.search(r'checkpoint_(\d+)\.pth', fname)
        return int(match.group(1)) if match else -1

    latest_file = max(checkpoint_files, key=extract_epoch)
    return latest_file

def evaluate_test_set(checkpoint_path=None, test_csv_path=None, scaler_path=None):
    # Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
    checkpoint_dir = os.path.join(current_dir, 'checkpoints')
    
    if test_csv_path is None:
        test_csv_path = os.path.join(project_root, 'Datasets', 'Storm_data', 'test.csv')
    if scaler_path is None:
        scaler_path = os.path.join(checkpoint_dir, 'scaler.pkl')
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        
    print("=" * 70)
    print("           STORM AUTOENCODER EVALUATION ON TEST SET")
    print("=" * 70)
    print(f"Test Data Path     : {test_csv_path}")
    print(f"Scaler Path        : {scaler_path}")
    print(f"Checkpoint Path    : {checkpoint_path}")

    # Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation Device  : {device}")
    
    # 1. Load Scaler
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Loaded MinMaxScaler successfully.")

    # 2. Load & Prepare Test Data
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found at: {test_csv_path}")
    df_raw = pd.read_csv(test_csv_path)
    print(f"Loaded {len(df_raw):,} records from test.csv.")

    base_features = ['time', 'grade', 'lat', 'lon', 'pressure_hpa']
    wind_features = ['max_wind_kt', 'dir_50kt', 'rad_50kt_long_nm', 'rad_50kt_short_nm', 
                     'dir_30kt', 'rad_30kt_long_nm', 'rad_30kt_short_nm']
    selected_cols = base_features + wind_features
    df_test = df_raw[selected_cols].copy()

    # Time feature engineering
    df_test['time'] = pd.to_datetime(df_test['time'])
    df_test['year'] = df_test['time'].dt.year
    df_test['month'] = df_test['time'].dt.month
    df_test['day'] = df_test['time'].dt.day
    df_test['hour'] = df_test['time'].dt.hour
    df_test.drop('time', axis=1, inplace=True)

    feature_names = list(df_test.columns)

    # Transform with the saved MinMaxScaler
    scaled_values = scaler.transform(df_test)
    scaled_df = pd.DataFrame(scaled_values, columns=feature_names)
    scaled_df.fillna(-1, inplace=True)

    # 3. Load Model Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    input_size = len(feature_names)
    latent_dim = checkpoint.get('latent_dim', 64)
    epoch_trained = checkpoint.get('epoch', 'N/A')
    
    model = SimpleAutoencoder(input_dim=input_size, latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded Autoencoder checkpoint from Epoch {epoch_trained} (Latent Size: {latent_dim}).\n")

    # 4. Run Test Inference
    test_dataset = StormDataset(scaled_df)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    criterion = MaskedMSELoss().to(device)

    all_preds = []
    all_targets = []
    all_latents = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            latents = model.encode(batch)
            preds = model(batch)
            loss = criterion(preds, batch)
            
            total_loss += loss.item() * len(batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch.cpu().numpy())
            all_latents.append(latents.cpu().numpy())

    avg_test_loss = total_loss / len(test_dataset)
    preds_arr = np.vstack(all_preds)
    targets_arr = np.vstack(all_targets)
    latents_arr = np.vstack(all_latents)

    loss_display = f"{avg_test_loss:.6f}" if avg_test_loss >= 1e-4 else f"{avg_test_loss:.4e}"
    print(f"--> Overall Masked MSE Loss on Test Set: {loss_display}")

    # 5. Calculate Per-Feature Physical Error Matrix
    print("\n" + "=" * 70)
    print("         PER-FEATURE RECONSTRUCTION MATRIX & ERROR METRICS")
    print("=" * 70)
    
    metrics_list = []
    
    for i, col in enumerate(feature_names):
        # Filter out missing (-1) values
        mask = targets_arr[:, i] >= 0.0
        if mask.sum() == 0:
            continue
            
        y_true_scaled = targets_arr[mask, i]
        y_pred_scaled = preds_arr[mask, i]
        
        # Invert scaling back to original physical units
        data_min = scaler.data_min_[i]
        data_range = scaler.data_range_[i]
        
        y_true_orig = y_true_scaled * data_range + data_min
        y_pred_orig = y_pred_scaled * data_range + data_min
        
        mae = mean_absolute_error(y_true_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
        r2 = r2_score(y_true_scaled, y_pred_scaled)
        
        # Unit formatting
        unit = ""
        if col == 'pressure_hpa': unit = " hPa"
        elif 'kt' in col and not col.startswith('dir'): unit = " kt"
        elif 'nm' in col: unit = " nm"
        elif col in ['lat', 'lon']: unit = " °"
        
        metrics_list.append({
            'Feature': col,
            'Physical MAE': f"{mae:.3f}{unit}",
            'Physical RMSE': f"{rmse:.3f}{unit}",
            'R² Score': f"{r2:.4f}",
            'Valid Records': f"{int(mask.sum()):,} ({mask.sum()/len(df_test)*100:.1f}%)"
        })

    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df.to_string(index=False))

    # 6. Categorical Grade Evaluation
    if 'grade' in feature_names:
        grade_idx = feature_names.index('grade')
        g_min = scaler.data_min_[grade_idx]
        g_range = scaler.data_range_[grade_idx]
        
        true_grade = np.round(targets_arr[:, grade_idx] * g_range + g_min).astype(int)
        pred_grade = np.round(preds_arr[:, grade_idx] * g_range + g_min).astype(int)
        grade_acc = accuracy_score(true_grade, pred_grade) * 100
        
        print("\n" + "=" * 70)
        print(f" Grade Classification Exact Match Accuracy: {grade_acc:.2f}%")
        print("=" * 70)

    # 7. Latent Space Representation Analysis (PCA)
    print("\n" + "=" * 70)
    print("             64-D LATENT SPACE VARIANCE ANALYSIS")
    print("=" * 70)
    pca = PCA(n_components=min(10, latent_dim))
    pca.fit(latents_arr)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100
    print(f"Top 3 Principal Components explain : {cumulative_variance[2]:.2f}% of latent variance")
    print(f"Top 5 Principal Components explain : {cumulative_variance[4]:.2f}% of latent variance")
    print(f"Top 10 Principal Components explain: {cumulative_variance[-1]:.2f}% of latent variance")
    print("=" * 70)
    print("Test evaluation finished successfully!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Storm Autoencoder on test.csv")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file (default: latest checkpoint)")
    parser.add_argument("--test-data", type=str, default=None, help="Path to test.csv")
    parser.add_argument("--scaler", type=str, default=None, help="Path to scaler.pkl")
    args = parser.parse_args()

    evaluate_test_set(checkpoint_path=args.checkpoint, test_csv_path=args.test_data, scaler_path=args.scaler)
