"""
train_vime.py
End-to-End Self-Supervised and Semi-Supervised Training Pipeline for VIME on Storm Data.

Stages:
1. Feature Engineering & Scaling on Datasets/Storm_data/train.csv.
2. Self-Supervised Pretext Training (Empirical Marginal Corruption + Mask Estimation + Value Imputation).
3. Downstream Fine-Tuning / Semi-Supervised Consistency Training for Multi-Horizon Storm Path Prediction.
4. Evaluation on test set (Cosine Similarity, Haversine Distance Error in km).

References:
    Yoon et al. (NeurIPS 2020) "VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain"
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List, Dict

# Ensure local module imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from vime_model import VIMESelfSupervised, VIMEPredictor
from vime_dataset import VIMESelfSupervisedDataset, VIMESemiSupervisedDataset, VIMELoss, corrupt_tabular_data


def haversine_np(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Computes great-circle distance in km between coordinates using Haversine formula."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0 * c


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Builds tabular features with temporal encodings, lag properties,
    and multi-horizon delta position targets for storm trajectory prediction.
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by=['international_id', 'time']).reset_index(drop=True)

    # 1. Cyclical time features
    month = df['time'].dt.month
    day = df['time'].dt.day
    hour = df['time'].dt.hour

    df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * month / 12.0)
    df['day_sin'] = np.sin(2 * np.pi * day / 31.0)
    df['day_cos'] = np.cos(2 * np.pi * day / 31.0)
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)

    # 2. Multi-horizon targets (6h, 12h, 18h, 24h)
    horizons = {'6h': -1, '12h': -2, '18h': -3, '24h': -4}
    target_cols = []
    for h_name, shift_val in horizons.items():
        future_lat = df.groupby('international_id')['lat'].shift(shift_val)
        future_lon = df.groupby('international_id')['lon'].shift(shift_val)
        delta_lat = future_lat - df['lat']
        delta_lon = future_lon - df['lon']
        df[f'delta_lat_{h_name}'] = delta_lat
        df[f'delta_lon_{h_name}'] = delta_lon
        target_cols.extend([f'delta_lat_{h_name}', f'delta_lon_{h_name}'])

    # 3. Lag features (past 6h, 12h, 18h, 24h)
    lag_props = [
        'lat', 'lon', 'pressure_hpa', 'max_wind_kt',
        'dir_50kt', 'rad_50kt_long_nm', 'rad_50kt_short_nm',
        'dir_30kt', 'rad_30kt_long_nm', 'rad_30kt_short_nm'
    ]
    lags = {'6h': 1, '12h': 2, '18h': 3, '24h': 4}
    for col in lag_props:
        if col in df.columns:
            for l_name, shift_val in lags.items():
                lagged = df.groupby('international_id')[col].shift(shift_val)
                df[f'{col}_lag_{l_name}'] = lagged
                if col in ['lat', 'lon']:
                    df[f'past_delta_{col}_{l_name}'] = df[col] - lagged

    # Drop rows without targets for downstream task
    df.dropna(subset=target_cols, inplace=True)

    # Drop metadata columns
    drop_cols = [
        'tc_number', 'name', 'grade_name', 'revision_date', 'year',
        'time_diff_hours', 'time', 'month', 'day', 'hour', 'flag_last',
        'landfall_flag', 'dir_50kt_name', 'dir_30kt_name'
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Drop non-numeric object columns except international_id
    obj_cols = [c for c in df.select_dtypes(include=['object']).columns if c != 'international_id']
    df.drop(columns=obj_cols, inplace=True)

    df = df.apply(pd.to_numeric, errors='coerce')
    return df, target_cols


def train_vime_pretext(
    model: VIMESelfSupervised,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: VIMELoss,
    num_epochs: int,
    device: torch.device,
    checkpoint_dir: str,
) -> float:
    """
    Self-Supervised Pretext Training Phase of VIME.
    Trains the encoder to solve Mask Estimation and Feature Imputation pretext tasks.
    """
    print("=" * 70)
    print(f"STAGE 1: VIME Self-Supervised Pretext Training ({num_epochs} Epochs)")
    print(f"Device: {device} | Batches: {len(dataloader)}")
    print("=" * 70)

    best_loss = float('inf')
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_mask_loss = 0.0
        total_impute_loss = 0.0

        for x_orig, x_corr, mask_true in dataloader:
            x_corr = x_corr.to(device)
            x_orig = x_orig.to(device)
            mask_true = mask_true.to(device)

            optimizer.zero_grad()
            mask_pred, val_pred = model(x_corr)

            loss, loss_m, loss_i = criterion(mask_pred, val_pred, mask_true, x_orig)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mask_loss += loss_m.item()
            total_impute_loss += loss_i.item()

        avg_loss = total_loss / len(dataloader)
        avg_mask = total_mask_loss / len(dataloader)
        avg_imp = total_impute_loss / len(dataloader)

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'encoder_state_dict': model.encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'input_dim': model.input_dim,
                    'latent_dim': model.latent_dim,
                },
                os.path.join(checkpoint_dir, "best_vime_pretext.pth")
            )

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            star = " * [BEST]" if is_best else ""
            print(f"Epoch [{epoch:03d}/{num_epochs:03d}] | Pretext Loss: {avg_loss:.5f} (Mask BCE: {avg_mask:.5f}, Impute MSE: {avg_imp:.5f}){star}")

    print(f"\nPretext Training Finished! Best Loss: {best_loss:.5f} at Epoch {best_epoch}")
    return best_loss


def train_vime_downstream(
    predictor: VIMEPredictor,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    checkpoint_dir: str,
    all_unlabeled_features: np.ndarray,
    p_m2: float = 0.25,
    beta: float = 0.5,
    K: int = 3,
) -> float:
    """
    Downstream Semi-Supervised Fine-Tuning Phase of VIME.
    Combines Supervised Loss on labeled rows with Consistency Regularization on corrupted unlabeled rows.
    """
    print("=" * 70)
    print(f"STAGE 2: VIME Downstream Semi-Supervised Fine-Tuning ({num_epochs} Epochs)")
    print(f"Consistency Regularization: Beta={beta}, K={K} augmentations, p_m2={p_m2}")
    print("=" * 70)

    sup_criterion = nn.MSELoss()
    consistency_criterion = nn.MSELoss()
    best_loss = float('inf')
    best_epoch = -1

    rng = np.random.default_rng(42)

    for epoch in range(1, num_epochs + 1):
        predictor.train()
        total_epoch_loss = 0.0
        total_sup_loss = 0.0
        total_unsup_loss = 0.0

        for x_batch, y_batch, is_labeled in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # 1. Supervised prediction loss
            pred_y = predictor(x_batch)
            loss_sup = sup_criterion(pred_y, y_batch)

            # 2. VIME Semi-Supervised Consistency Loss
            # Generate K corrupted variations from tabular empirical marginal distributions
            loss_unsup = torch.tensor(0.0, device=device)
            if beta > 0:
                batch_size = x_batch.shape[0]
                # Sample random rows for empirical distribution replacement
                curr_np = x_batch.detach().cpu().numpy()
                for _ in range(K):
                    x_corrupted_np, _ = corrupt_tabular_data(curr_np, p_m=p_m2, rng=rng)
                    x_corr_tensor = torch.from_numpy(x_corrupted_np).to(device)
                    pred_corr = predictor(x_corr_tensor)
                    loss_unsup = loss_unsup + consistency_criterion(pred_corr, pred_y.detach())
                loss_unsup = loss_unsup / float(K)

            # Combined downstream loss
            loss_total = loss_sup + beta * loss_unsup
            loss_total.backward()
            optimizer.step()

            total_epoch_loss += loss_total.item()
            total_sup_loss += loss_sup.item()
            total_unsup_loss += loss_unsup.item()

        avg_total = total_epoch_loss / len(train_loader)
        avg_sup = total_sup_loss / len(train_loader)
        avg_unsup = total_unsup_loss / len(train_loader)

        is_best = avg_total < best_loss
        if is_best:
            best_loss = avg_total
            best_epoch = epoch
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': predictor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                },
                os.path.join(checkpoint_dir, "best_vime_predictor.pth")
            )

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            star = " * [BEST]" if is_best else ""
            print(f"Epoch [{epoch:03d}/{num_epochs:03d}] | Total: {avg_total:.5f} (Supervised: {avg_sup:.5f}, Consistency: {avg_unsup:.5f}){star}")

    print(f"\nDownstream Training Finished! Best Loss: {best_loss:.5f} at Epoch {best_epoch}")
    return best_loss

    print(f"\nDownstream Training Finished! Best Loss: {best_loss:.5f} at Epoch {best_epoch}")
    return best_loss


def evaluate_vime(
    predictor: VIMEPredictor,
    X_test_scaled: np.ndarray,
    df_test_raw: pd.DataFrame,
    target_cols: List[str],
    device: torch.device,
    results_path: str,
) -> Dict[str, Dict[str, float]]:
    """Evaluates multi-horizon trajectory predictions using Cosine Similarity and Haversine Distance."""
    predictor.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(X_test_scaled.astype(np.float32)).to(device)
        predictions = predictor(x_tensor).cpu().numpy()

    eval_results = {}
    horizons = ['6h', '12h', '18h', '24h']

    print("\n" + "=" * 70)
    print("STAGE 3: Model Evaluation on Test Set")
    print("=" * 70)

    with open(results_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("VIME Storm Path Prediction Evaluation Results\n")
        f.write("=" * 70 + "\n\n")

        for idx, h in enumerate(horizons):
            lat_t_col = f'delta_lat_{h}'
            lon_t_col = f'delta_lon_{h}'

            lat_idx = target_cols.index(lat_t_col)
            lon_idx = target_cols.index(lon_t_col)

            true_delta_lat = df_test_raw[lat_t_col].values
            true_delta_lon = df_test_raw[lon_t_col].values

            pred_delta_lat = predictions[:, lat_idx]
            pred_delta_lon = predictions[:, lon_idx]

            # 1. Cosine Similarity on displacement vectors
            y_true_vec = np.column_stack((true_delta_lat, true_delta_lon))
            y_pred_vec = np.column_stack((pred_delta_lat, pred_delta_lon))

            dot_product = np.sum(y_true_vec * y_pred_vec, axis=1)
            norm_true = np.linalg.norm(y_true_vec, axis=1)
            norm_pred = np.linalg.norm(y_pred_vec, axis=1)
            denom = norm_true * norm_pred

            similarity = np.zeros_like(dot_product)
            valid_mask = denom > 1e-8
            similarity[valid_mask] = dot_product[valid_mask] / denom[valid_mask]
            avg_cosine_sim = float(np.mean(similarity))

            # 2. Haversine Distance Error (km)
            curr_lat = df_test_raw['lat'].values
            curr_lon = df_test_raw['lon'].values

            actual_lat = curr_lat + true_delta_lat
            actual_lon = curr_lon + true_delta_lon

            forecast_lat = curr_lat + pred_delta_lat
            forecast_lon = curr_lon + pred_delta_lon

            dist_errors = haversine_np(actual_lon, actual_lat, forecast_lon, forecast_lat)
            mean_dist = float(np.mean(dist_errors))
            median_dist = float(np.median(dist_errors))
            min_dist = float(np.min(dist_errors))
            max_dist = float(np.max(dist_errors))

            eval_results[h] = {
                'cosine_sim': avg_cosine_sim,
                'mean_dist_km': mean_dist,
                'median_dist_km': median_dist,
                'min_dist_km': min_dist,
                'max_dist_km': max_dist,
            }

            res_line = (
                f"Horizon [{h:>3}] -> Cosine Sim: {avg_cosine_sim:.4f} | "
                f"Distance Error: Mean={mean_dist:6.2f} km, Median={median_dist:6.2f} km, "
                f"Min={min_dist:5.2f} km, Max={max_dist:7.2f} km"
            )
            print(res_line)
            f.write(res_line + "\n")

    print(f"\nEvaluation summary successfully written to: {results_path}")
    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Train VIME on Storm Tabular Dataset")
    parser.add_argument("--train_path", type=str, default=os.path.join(PROJECT_ROOT, "Datasets/Storm_data/train.csv"), help="Path to train.csv")
    parser.add_argument("--test_path", type=str, default=os.path.join(PROJECT_ROOT, "Datasets/Storm_data/test.csv"), help="Path to test.csv")
    parser.add_argument("--checkpoint_dir", type=str, default=os.path.join(CURRENT_DIR, "checkpoints"), help="Path to save checkpoints")
    parser.add_argument("--result_path", type=str, default=os.path.join(CURRENT_DIR, "train_result.txt"), help="Path to save evaluation results")
    
    # Architecture hyperparameters
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layers dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of encoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")

    # VIME hyperparameters
    parser.add_argument("--p_m", type=float, default=0.30, help="Pretext corruption probability (Bernoulli p_m)")
    parser.add_argument("--alpha", type=float, default=2.0, help="Pretext loss weight: L_mask + alpha * L_impute")
    parser.add_argument("--p_m2", type=float, default=0.25, help="Downstream perturbation rate for consistency loss")
    parser.add_argument("--beta", type=float, default=0.50, help="Downstream consistency regularization weight")
    parser.add_argument("--K", type=int, default=3, help="Number of corrupted samples per batch for consistency")

    # Training parameters
    parser.add_argument("--epochs_pretext", type=int, default=40, help="Number of self-supervised pretext epochs")
    parser.add_argument("--epochs_downstream", type=int, default=50, help="Number of downstream fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr_pretext", type=float, default=1e-3, help="Learning rate for pretext training")
    parser.add_argument("--lr_downstream", type=float, default=5e-4, help="Learning rate for downstream training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # 1. Setup device and random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 2. Load and Prepare Datasets
    print(f"[INFO] Loading training dataset: {args.train_path}")
    if not os.path.exists(args.train_path):
        raise FileNotFoundError(f"Training dataset not found at: {args.train_path}")
    
    df_train_raw = pd.read_csv(args.train_path, low_memory=False)
    print(f"[SUCCESS] Loaded {len(df_train_raw):,} rows from {args.train_path}")

    df_train, target_cols = engineer_features(df_train_raw)
    feature_cols = [c for c in df_train.columns if c not in target_cols and c != 'international_id']

    print(f"[INFO] Number of engineered features: {len(feature_cols)}")
    print(f"[INFO] Target horizons: {target_cols}")

    # Scale continuous input features
    scaler = StandardScaler()
    X_train_raw = df_train[feature_cols].fillna(0.0).values
    X_train_scaled = scaler.fit_transform(X_train_raw)
    y_train = df_train[target_cols].values

    # Save fitted scaler and feature list
    scaler_path = os.path.join(args.checkpoint_dir, "vime_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "features": feature_cols, "targets": target_cols}, f)
    print(f"[INFO] Saved scaler & schema to: {scaler_path}")

    # 3. Create Pretext PyTorch DataLoader
    pretext_dataset = VIMESelfSupervisedDataset(
        data=X_train_scaled,
        p_m=args.p_m,
        on_the_fly_corruption=True,
        seed=args.seed
    )
    pretext_loader = DataLoader(
        pretext_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda")
    )

    # 4. Instantiate & Train Self-Supervised VIME
    input_dim = len(feature_cols)
    vime_ssl = VIMESelfSupervised(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    pretext_optimizer = torch.optim.AdamW(vime_ssl.parameters(), lr=args.lr_pretext, weight_decay=1e-5)
    pretext_criterion = VIMELoss(alpha=args.alpha)

    train_vime_pretext(
        model=vime_ssl,
        dataloader=pretext_loader,
        optimizer=pretext_optimizer,
        criterion=pretext_criterion,
        num_epochs=args.epochs_pretext,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )

    # 5. Initialize Downstream Fine-Tuning Model with Pretrained Encoder
    best_pretext_ckpt = torch.load(os.path.join(args.checkpoint_dir, "best_vime_pretext.pth"), map_location=device)
    vime_ssl.load_state_dict(best_pretext_ckpt["model_state_dict"])
    print("[INFO] Successfully loaded best pretrained VIME encoder weights.")

    predictor = VIMEPredictor(
        encoder=vime_ssl.encoder,
        output_dim=len(target_cols),
        hidden_dim=args.hidden_dim // 2,
        dropout=args.dropout,
        freeze_encoder=False
    ).to(device)

    downstream_dataset = VIMESemiSupervisedDataset(X=X_train_scaled, y=y_train)
    downstream_loader = DataLoader(
        downstream_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda")
    )

    downstream_optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr_downstream, weight_decay=1e-4)

    train_vime_downstream(
        predictor=predictor,
        train_loader=downstream_loader,
        optimizer=downstream_optimizer,
        num_epochs=args.epochs_downstream,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        all_unlabeled_features=X_train_scaled,
        p_m2=args.p_m2,
        beta=args.beta,
        K=args.K
    )

    # 6. Evaluation on Test Set (if test.csv exists)
    if os.path.exists(args.test_path):
        print(f"\n[INFO] Loading test dataset from: {args.test_path}")
        df_test_raw = pd.read_csv(args.test_path, low_memory=False)
        df_test, _ = engineer_features(df_test_raw)

        # Align columns
        X_test_raw = df_test[feature_cols].fillna(0.0).values
        X_test_scaled = scaler.transform(X_test_raw)

        # Load best predictor weights
        best_predictor_ckpt = torch.load(os.path.join(args.checkpoint_dir, "best_vime_predictor.pth"), map_location=device)
        predictor.load_state_dict(best_predictor_ckpt["model_state_dict"])

        evaluate_vime(
            predictor=predictor,
            X_test_scaled=X_test_scaled,
            df_test_raw=df_test,
            target_cols=target_cols,
            device=device,
            results_path=args.result_path
        )
    else:
        print(f"[WARN] Test dataset not found at '{args.test_path}'. Skipping test evaluation.")

    print("\n[SUCCESS] VIME training pipeline execution completed!")


if __name__ == "__main__":
    main()
