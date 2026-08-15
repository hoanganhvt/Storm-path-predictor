import os
import sys
import pickle
import argparse
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from model import StormPathCfCLiquidUNet
    from datasets import VariableStormDataset, left_pad_collate_fn
    from utils import (
        prepare_storm_dataframe,
        scale_storm_features,
        build_storm_trajectory_sequences,
    )
except ImportError:
    from .model import StormPathCfCLiquidUNet
    from .datasets import VariableStormDataset, left_pad_collate_fn
    from .utils import (
        prepare_storm_dataframe,
        scale_storm_features,
        build_storm_trajectory_sequences,
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    
    for batch_x, batch_y, _ in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        # Forward pass: encoded first via SimpleAutoencoderEncoder -> CfC Liquid UNet -> 4 FC outputs of 2 nodes
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        total_loss += loss.item() * len(batch_x)
        
    return total_loss / len(dataloader.dataset)


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y, _ in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            total_loss += loss.item() * len(batch_x)
    return total_loss / len(dataloader.dataset)


def train_storm_model(
    train_csv: str,
    autoencoder_weights: Optional[str] = None,
    scaler_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    min_history_steps: int = 2,
    num_future_steps: int = 4
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = os.path.join(current_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("      STORM PATH PREDICTOR TRAINING (LNN + SIMPLE AUTOENCODER)")
    print("=" * 80)
    print(f"Device               : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Train Dataset Path   : {train_csv}")
    print(f"Checkpoints Output   : {output_dir}")
    print(f"Min History Data     : {min_history_steps * 6} hours ({min_history_steps} states)")
    print(f"Forecast Horizons    : {num_future_steps} steps ({num_future_steps * 6} hours ahead: +6h, +12h, +18h, +24h)")
    print(f"Epochs / Batch Size  : {epochs} epochs / {batch_size} batch size")
    print(f"Learning Rate        : {lr}")
    
    # -------------------------------------------------------------
    # 1. Load Training Data and Feature Engineering
    # -------------------------------------------------------------
    print("\n[1/4] Loading training data and performing feature engineering...")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train dataset not found at '{train_csv}'")
        
    df_train_raw = pd.read_csv(train_csv)
    df_train, feature_cols = prepare_storm_dataframe(df_train_raw)
    print(f"  --> {len(feature_cols)} Features: {feature_cols}")
    
    # -------------------------------------------------------------
    # 2. Load or Fit Scaler (StandardScaler / MinMaxScaler)
    # -------------------------------------------------------------
    print("\n[2/4] Loading/Preparing feature scaler...")
    if scaler_path is None:
        default_scaler_path = os.path.abspath(
            os.path.join(current_dir, "../Simple Autoencoder/checkpoints/scaler.pkl")
        )
        if os.path.exists(default_scaler_path):
            scaler_path = default_scaler_path
            
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        print(f"  --> Loading saved scaler from: '{scaler_path}'")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"  --> [SUCCESS] Loaded {type(scaler).__name__} from '{scaler_path}'")
        df_train_scaled, scaler = scale_storm_features(df_train, feature_cols, scaler=scaler, fit_scaler=False)
    else:
        print(f"  --> [NOTE] Fitting new StandardScaler on training dataset...")
        df_train_scaled, scaler = scale_storm_features(df_train, feature_cols, fit_scaler=True, use_standard_scaler=True)
        
    out_scaler_path = os.path.join(output_dir, "storm_scaler.pkl")
    with open(out_scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  --> Saved scaler to: '{out_scaler_path}'")
    
    # -------------------------------------------------------------
    # 3. Build Storm Trajectory Sequences (Expanding History >= 12h)
    # -------------------------------------------------------------
    print("\n[3/4] Building variable-length storm trajectory sequences (>= 12h history)...")
    X_train, Y_train, coords_train, ids_train = build_storm_trajectory_sequences(
        df_train, df_train_scaled, feature_cols,
        min_history_steps=min_history_steps,
        num_future_steps=num_future_steps
    )
    print(f"  --> Prepared {len(X_train):,} sequences across {len(np.unique(ids_train))} unique storms")
    
    # Validation split by storm ID (10% storms for validation)
    unique_storms = np.unique(ids_train)
    rng = np.random.default_rng(42)
    val_storms = set(rng.choice(unique_storms, size=max(1, int(len(unique_storms) * 0.1)), replace=False))
    
    train_idx = [i for i, sid in enumerate(ids_train) if sid not in val_storms]
    val_idx = [i for i, sid in enumerate(ids_train) if sid in val_storms]
    
    X_tr = [X_train[i] for i in train_idx]
    Y_tr = Y_train[train_idx]
    coords_tr = coords_train[train_idx]
    ids_tr = ids_train[train_idx]
    
    X_val = [X_train[i] for i in val_idx]
    Y_val = Y_train[val_idx]
    coords_val = coords_train[val_idx]
    ids_val = ids_train[val_idx]
    
    train_dataset = VariableStormDataset(X_tr, Y_tr, coords_tr, ids_tr)
    val_dataset = VariableStormDataset(X_val, Y_val, coords_val, ids_val)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=left_pad_collate_fn, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        collate_fn=left_pad_collate_fn, pin_memory=(device.type == 'cuda')
    )
    
    # -------------------------------------------------------------
    # 4. Initialize Model and Load Pretrained Autoencoder Weights
    # -------------------------------------------------------------
    print("\n[4/4] Initializing StormPathCfCLiquidUNet and loading Autoencoder weights...")
    input_dim = len(feature_cols)
    latent_dim = 64
    
    model = StormPathCfCLiquidUNet(
        input_dim=input_dim,
        latent_dim=latent_dim,
        unet_hidden_dims=(64, 128, 256),
        fc_hidden_dim=128,
        num_horizons=num_future_steps,
        nodes_per_horizon=2,
        dropout=0.1
    ).to(device)
    
    if autoencoder_weights is None:
        default_ae_path = os.path.abspath(
            os.path.join(current_dir, "../Simple Autoencoder/checkpoints/best_autoencoder.pth")
        )
        if os.path.exists(default_ae_path):
            autoencoder_weights = default_ae_path
            
    if autoencoder_weights and os.path.exists(autoencoder_weights):
        print(f"  --> Loading pretrained Autoencoder weights from: '{autoencoder_weights}'")
        model.encoder.load_pretrained_weights(autoencoder_weights)
        # CRITICAL REQUIREMENT: Do NOT freeze the encoder; keep it trainable!
        model.encoder.freeze(freeze=False)
        print("  --> [CONFIRMED] Pretrained Autoencoder Encoder is UNFROZEN and trainable.")
    else:
        print(f"  --> [NOTE] No pretrained weights specified (training encoder from scratch).")
        model.encoder.freeze(freeze=False)
        
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    print(f"\nStarting training loop for {epochs} epochs...")
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, "best_storm_model.pth")
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'input_dim': input_dim,
                'latent_dim': latent_dim,
                'min_history_steps': min_history_steps,
                'num_future_steps': num_future_steps,
                'feature_cols': feature_cols,
                'scaler_path': out_scaler_path
            }, best_model_path)
            
        cur_lr = optimizer.param_groups[0]['lr']
        best_marker = " * [BEST SAVED]" if is_best else ""
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {cur_lr:.1e}{best_marker}")
        
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Saved best model checkpoint to: '{best_model_path}'")
    print(f"Saved feature scaler to: '{out_scaler_path}'")


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    default_train = os.path.join(project_root, "Datasets", "Storm_data", "train.csv")
    default_ae_weights = os.path.join(project_root, "src", "train", "Simple Autoencoder", "checkpoints", "best_autoencoder.pth")
    default_scaler = os.path.join(project_root, "src", "train", "Simple Autoencoder", "checkpoints", "scaler.pkl")
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    
    parser = argparse.ArgumentParser(description="Train Storm Path Predictor (LNN + Simple Autoencoder)")
    parser.add_argument("--train-csv", type=str, default=default_train, help="Path to train.csv")
    parser.add_argument("--autoencoder-weights", type=str, default=default_ae_weights, help="Path to pretrained SimpleAutoencoder checkpoint")
    parser.add_argument("--scaler", type=str, default=default_scaler, help="Path to fitted scaler (.pkl)")
    parser.add_argument("--output-dir", type=str, default=default_output, help="Directory to save checkpoints and scaler")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--min-history", type=int, default=2, help="Minimum history steps (e.g. 2 = 12h)")
    parser.add_argument("--future-steps", type=int, default=4, help="Future prediction steps (e.g. 4 = 24h)")
    
    args = parser.parse_args()
    
    train_storm_model(
        train_csv=args.train_csv,
        autoencoder_weights=args.autoencoder_weights,
        scaler_path=args.scaler,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_history_steps=args.min_history,
        num_future_steps=args.future_steps
    )


if __name__ == "__main__":
    main()
