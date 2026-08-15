import os
import sys
import pickle
import argparse
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from model import StormPathCfCLiquidUNet
    from datasets import VariableStormDataset, left_pad_collate_fn
    from utils import (
        prepare_storm_dataframe,
        scale_storm_features,
        build_storm_trajectory_sequences,
        generate_evaluation_report,
    )
except ImportError:
    from .model import StormPathCfCLiquidUNet
    from .datasets import VariableStormDataset, left_pad_collate_fn
    from .utils import (
        prepare_storm_dataframe,
        scale_storm_features,
        build_storm_trajectory_sequences,
        generate_evaluation_report,
    )


def evaluate_test_set(
    checkpoint_path: str,
    test_csv_path: str,
    scaler_path: Optional[str] = None,
    output_report_path: Optional[str] = None,
    batch_size: int = 64
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("         STORM PATH PREDICTOR (LNN + SIMPLE AUTOENCODER) EVALUATION")
    print("=" * 80)
    print(f"Device              : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Model Checkpoint    : {checkpoint_path}")
    print(f"Test Dataset        : {test_csv_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: '{checkpoint_path}'")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test dataset not found at: '{test_csv_path}'")
        
    # -------------------------------------------------------------
    # 1. Load Checkpoint
    # -------------------------------------------------------------
    print("\n[1/4] Loading model checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    input_dim = checkpoint.get('input_dim', 15)
    latent_dim = checkpoint.get('latent_dim', 64)
    min_history_steps = checkpoint.get('min_history_steps', 2)
    num_future_steps = checkpoint.get('num_future_steps', 4)
    feature_cols = checkpoint.get('feature_cols', None)
    
    model = StormPathCfCLiquidUNet(
        input_dim=input_dim,
        latent_dim=latent_dim,
        unet_hidden_dims=(64, 128, 256),
        fc_hidden_dim=128,
        num_horizons=num_future_steps,
        nodes_per_horizon=2,
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  --> Successfully loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
    
    # -------------------------------------------------------------
    # 2. Load Scaler
    # -------------------------------------------------------------
    print("\n[2/4] Loading fitted scaler...")
    if scaler_path is None:
        scaler_path = checkpoint.get('scaler_path', os.path.join(current_dir, 'checkpoints', 'storm_scaler.pkl'))
        
    if not os.path.exists(scaler_path):
        alt_scaler = os.path.abspath(os.path.join(current_dir, '../Simple Autoencoder/checkpoints/scaler.pkl'))
        if os.path.exists(alt_scaler):
            scaler_path = alt_scaler
            
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at '{scaler_path}'. Please run train.py first.")
        
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"  --> Successfully loaded scaler from '{scaler_path}'")
    
    # -------------------------------------------------------------
    # 3. Load & Process Test Data
    # -------------------------------------------------------------
    print("\n[3/4] Preparing test dataset...")
    df_raw = pd.read_csv(test_csv_path)
    df_test, prepared_feats = prepare_storm_dataframe(df_raw)
    
    if feature_cols is None:
        feature_cols = prepared_feats
        
    df_test_scaled, _ = scale_storm_features(df_test, feature_cols, scaler=scaler, fit_scaler=False)
    X_test, Y_test, coords_test, ids_test = build_storm_trajectory_sequences(
        df_test, df_test_scaled, feature_cols,
        min_history_steps=min_history_steps,
        num_future_steps=num_future_steps
    )
    print(f"  --> Prepared {len(X_test):,} test sequences across {len(np.unique(ids_test))} unique storms")
    
    test_dataset = VariableStormDataset(X_test, Y_test, coords_test, ids_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=left_pad_collate_fn)
    
    # -------------------------------------------------------------
    # 4. Run Test Inference
    # -------------------------------------------------------------
    print("\n[4/4] Running model inference on test dataset...")
    all_preds, all_targets, all_coords = [], [], []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_coords in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())
            all_coords.append(batch_coords.numpy())
            
    preds_arr = np.concatenate(all_preds, axis=0)      # (N, 4, 2)
    targets_arr = np.concatenate(all_targets, axis=0)  # (N, 4, 2)
    coords_arr = np.concatenate(all_coords, axis=0)    # (N, 2)
    
    # -------------------------------------------------------------
    # 5. Format & Output Metrics Report
    # -------------------------------------------------------------
    if output_report_path is None:
        output_report_path = os.path.join(current_dir, "checkpoints", "test_result.txt")
        
    generate_evaluation_report(
        preds_arr=preds_arr,
        targets_arr=targets_arr,
        coords_arr=coords_arr,
        num_future_steps=num_future_steps,
        output_report_path=output_report_path
    )


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    default_test = os.path.join(project_root, "Datasets", "Storm_data", "test.csv")
    default_checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "best_storm_model.pth")
    default_scaler = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "storm_scaler.pkl")
    default_report = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "test_result.txt")
    
    parser = argparse.ArgumentParser(description="Evaluate Storm Path Predictor (LNN + Simple Autoencoder)")
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint, help="Path to best_storm_model.pth")
    parser.add_argument("--test-csv", type=str, default=default_test, help="Path to test.csv")
    parser.add_argument("--scaler", type=str, default=default_scaler, help="Path to storm_scaler.pkl")
    parser.add_argument("--report", type=str, default=default_report, help="Path to save evaluation report")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    
    args = parser.parse_args()
    
    evaluate_test_set(
        checkpoint_path=args.checkpoint,
        test_csv_path=args.test_csv,
        scaler_path=args.scaler,
        output_report_path=args.report,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
