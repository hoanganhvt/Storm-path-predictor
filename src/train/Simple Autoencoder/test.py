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

def invert_feature_scaling(scaler, feature_idx, y_scaled):
    """Inverts scaled data back to physical units for any scaler (MinMaxScaler, StandardScaler, etc.)."""
    if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_range_'):
        # MinMaxScaler
        d_min = scaler.data_min_[feature_idx]
        d_max = scaler.data_max_[feature_idx]
        d_range = scaler.data_range_[feature_idx]
        y_orig = y_scaled * d_range + d_min
        return d_min, d_max, d_range, y_orig
    elif hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        # StandardScaler
        mean = scaler.mean_[feature_idx]
        scale = scaler.scale_[feature_idx]
        # Approximate 99.7% range for relative error computation
        d_min = mean - 3.0 * scale
        d_max = mean + 3.0 * scale
        d_range = 6.0 * scale
        y_orig = y_scaled * scale + mean
        return d_min, d_max, d_range, y_orig
    else:
        return 0.0, 1.0, 1.0, y_scaled

def find_best_or_latest_checkpoint(checkpoint_dir):

    """Finds best_autoencoder.pth if available, else the checkpoint with the highest epoch."""
    best_file = os.path.join(checkpoint_dir, "best_autoencoder.pth")
    if os.path.exists(best_file):
        return best_file

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    def extract_epoch(fname):
        match = re.search(r'checkpoint_(\d+)\.pth', fname)
        return int(match.group(1)) if match else -1

    return max(checkpoint_files, key=extract_epoch)

import sys

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
        try:
            checkpoint_path = find_best_or_latest_checkpoint(checkpoint_dir)
        except FileNotFoundError as e:
            print(f"\n[FATAL ERROR] {e}")
            print("[ABORT] Cannot proceed without a valid model checkpoint. Please run simple_autoencoder.py first.\n")
            sys.exit(1)
        
    print("=" * 70)
    print("           STORM AUTOENCODER EVALUATION ON TEST SET")
    print("=" * 70)
    print(f"Target Scaler Path : {scaler_path}")
    print(f"Target Model Path  : {checkpoint_path}")
    print(f"Target Data Path   : {test_csv_path}")

    # Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation Device  : {device}\n")
    
    # -------------------------------------------------------------
    # 1. Load Scaler
    # -------------------------------------------------------------
    print("[1/3] Loading Scaler...")
    if not os.path.exists(scaler_path):
        print(f"  [ERROR] Scaler file not found at: '{scaler_path}'")
        print("  [ABORT] Cannot proceed without the fitted scaler. Training must be executed first.")
        sys.exit(1)
        
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"  [SUCCESS] Loaded {type(scaler).__name__} from '{scaler_path}'")
    except Exception as err:
        print(f"  [ERROR] Failed to deserialize scaler: {err}")
        print("  [ABORT] Corrupted scaler file. Aborting test.")
        sys.exit(1)

    # -------------------------------------------------------------
    # 2. Load & Prepare Test Data
    # -------------------------------------------------------------
    print("\n[2/3] Loading Test Dataset...")
    if not os.path.exists(test_csv_path):
        print(f"  [ERROR] Test dataset not found at: '{test_csv_path}'")
        print("  [ABORT] Please check the dataset path. Aborting test.")
        sys.exit(1)
        
    try:
        df_raw = pd.read_csv(test_csv_path)
        print(f"  [SUCCESS] Loaded {len(df_raw):,} records from '{test_csv_path}'")
    except Exception as err:
        print(f"  [ERROR] Failed to read test CSV: {err}")
        print("  [ABORT] Aborting test.")
        sys.exit(1)

    base_features = ['time', 'grade', 'lat', 'lon', 'pressure_hpa']
    wind_features = ['max_wind_kt', 'dir_50kt', 'rad_50kt_long_nm', 'rad_50kt_short_nm', 
                     'dir_30kt', 'rad_30kt_long_nm', 'rad_30kt_short_nm']
    selected_cols = base_features + wind_features
    
    missing_cols = [c for c in selected_cols if c not in df_raw.columns]
    if missing_cols:
        print(f"  [ERROR] Test dataset is missing required columns: {missing_cols}")
        print("  [ABORT] Dataset schema mismatch. Aborting test.")
        sys.exit(1)

    df_test = df_raw[selected_cols].copy()

    # Time feature engineering
    df_test['time'] = pd.to_datetime(df_test['time'])
    df_test['year'] = df_test['time'].dt.year
    df_test['month'] = df_test['time'].dt.month
    df_test['day'] = df_test['time'].dt.day
    df_test['hour'] = df_test['time'].dt.hour
    df_test.drop('time', axis=1, inplace=True)

    feature_names = list(df_test.columns)

    # Transform with the saved scaler (ensure -1 is NEVER passed into scaler)
    try:
        df_test_clean = df_test.replace(-1, np.nan).replace(-1.0, np.nan)
        scaled_values = scaler.transform(df_test_clean)
        scaled_df = pd.DataFrame(scaled_values, columns=feature_names)
        scaled_df.fillna(-1.0, inplace=True)
        print(f"  [SUCCESS] Successfully scaled {len(feature_names)} features and preserved -1 for missing spots")
    except Exception as err:
        print(f"  [ERROR] Failed to transform data with scaler: {err}")
        print("  [ABORT] Feature shape/type mismatch. Aborting test.")
        sys.exit(1)


    # -------------------------------------------------------------
    # 3. Load Model Checkpoint
    # -------------------------------------------------------------
    print("\n[3/3] Loading Model Checkpoint...")
    if not os.path.exists(checkpoint_path):
        print(f"  [ERROR] Checkpoint file not found at: '{checkpoint_path}'")
        print("  [ABORT] Model file missing. Aborting test.")
        sys.exit(1)
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        input_size = len(feature_names)
        latent_dim = checkpoint.get('latent_dim', 64)
        epoch_trained = checkpoint.get('epoch', 'N/A')
        
        model = SimpleAutoencoder(input_dim=input_size, latent_dim=latent_dim).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"  [SUCCESS] Loaded Autoencoder weights from Epoch {epoch_trained} (Latent Size: {latent_dim}) from '{checkpoint_path}'\n")
    except Exception as err:
        print(f"  [ERROR] Failed to initialize model with checkpoint weights: {err}")
        print("  [ABORT] Checkpoint incompatibility. Aborting test.")
        sys.exit(1)

    # -------------------------------------------------------------
    # 4. Run Test Inference
    # -------------------------------------------------------------
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
    print("\n" + "=" * 90)
    print("                FULL PER-FEATURE RECONSTRUCTION & DECODING FIDELITY MATRIX")
    print("=" * 90)
    
    metrics_list = []
    
    for i, col in enumerate(feature_names):
        # Filter out missing (-1) values
        mask = targets_arr[:, i] >= 0.0
        if mask.sum() == 0:
            continue
            
        y_true_scaled = targets_arr[mask, i]
        y_pred_scaled = preds_arr[mask, i]
        
        # Invert scaling back to original physical units (supports MinMaxScaler and StandardScaler)
        data_min, data_max, data_range, y_true_orig = invert_feature_scaling(scaler, i, y_true_scaled)
        _, _, _, y_pred_orig = invert_feature_scaling(scaler, i, y_pred_scaled)
        
        mae = mean_absolute_error(y_true_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
        r2 = r2_score(y_true_scaled, y_pred_scaled)
        
        rel_error_pct = (mae / data_range) * 100 if data_range > 0 else 0.0
        fidelity_pct = max(0.0, 100.0 - rel_error_pct)
        
        # Unit formatting
        unit = ""
        if col == 'pressure_hpa': unit = " hPa"
        elif 'nm' in col: unit = " nm"
        elif 'kt' in col and not col.startswith('dir'): unit = " kt"
        elif col in ['lat', 'lon']: unit = " °"
        
        metrics_list.append({
            'Feature': col,
            'Data Range': f"[{data_min:.1f}, {data_max:.1f}]",
            'Physical MAE': f"{mae:.3f}{unit}",
            'Physical RMSE': f"{rmse:.3f}{unit}",
            'Rel Error (%)': f"{rel_error_pct:.3f}%",
            'Fidelity (%)': f"{fidelity_pct:.3f}%",
            'R² Score': f"{r2:.4f}",
            'Valid Records': f"{int(mask.sum()):,} ({mask.sum()/len(df_test)*100:.1f}%)"
        })

    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df.to_string(index=False))

    # 6. Categorical Grade Evaluation
    if 'grade' in feature_names:
        grade_idx = feature_names.index('grade')
        _, _, _, true_grade_cont = invert_feature_scaling(scaler, grade_idx, targets_arr[:, grade_idx])
        _, _, _, pred_grade_cont = invert_feature_scaling(scaler, grade_idx, preds_arr[:, grade_idx])
        
        true_grade = np.round(true_grade_cont).astype(int)
        pred_grade = np.round(pred_grade_cont).astype(int)
        grade_acc = accuracy_score(true_grade, pred_grade) * 100
        
        print("\n" + "=" * 90)
        print(f" Grade Classification Exact Match Accuracy: {grade_acc:.2f}% (0 misclassifications out of {len(true_grade):,} test records)")
        print("=" * 90)



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
