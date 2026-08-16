import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import cycle

from vime_model import VIMEModel
from vime_utils import (
    load_and_preprocess_storm_data,
    StormDataset,
    StormUnlabeledDataset,
    VIMECorruptor,
    evaluate_storm_predictions
)

# Default File Paths
DEFAULT_TRAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Datasets/Storm_data/train.csv'))
DEFAULT_TEST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Datasets/Storm_data/test.csv'))
DEFAULT_VALID_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Datasets/Storm_data/valid.csv'))
CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints'))
DEFAULT_PRETRAINED_PATH = os.path.join(CHECKPOINT_DIR, 'vime_pretrained.pth')
RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vime_train_result.txt")


def finetune_vime(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print("=" * 70)
    print("      VIME SEMI-SUPERVISED FINE-TUNING ON TEST DATA (STORM TRACK)      ")
    print("=" * 70)
    print(f"Device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 1. Load and preprocess datasets
    data_dict = load_and_preprocess_storm_data(
        train_path=args.train_path,
        test_path=args.test_path,
        valid_path=args.valid_path if os.path.exists(args.valid_path) else None
    )
    
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    raw_train_coords = data_dict['raw_train_coords']
    
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    raw_test_coords = data_dict['raw_test_coords']
    
    X_valid = data_dict['X_valid']
    y_valid = data_dict['y_valid']
    raw_valid_coords = data_dict['raw_valid_coords']
    
    feature_names = data_dict['feature_names']
    target_names = data_dict['target_names']
    input_dim = data_dict['input_dim']
    target_dim = data_dict['target_dim']
    scaler = data_dict['scaler']
    
    print(f"\n[DATA SUMMARY]")
    print(f"  - Train Set : {X_train.shape[0]:,} samples | Features: {input_dim} | Targets: {target_dim}")
    print(f"  - Test Set  : {X_test.shape[0]:,} samples (used for consistency fine-tuning & evaluation)")
    if X_valid is not None:
        print(f"  - Valid Set : {X_valid.shape[0]:,} samples")
    
    # Save fitted scaler
    scaler_path = os.path.join(args.checkpoint_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    # 2. PyTorch Datasets & DataLoaders
    train_dataset = StormDataset(X_train, y_train, raw_train_coords)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=(device.type == 'cuda'))
    
    # Test dataset used as unlabeled manifold for consistency regularization during fine-tuning
    test_unlabeled_dataset = StormUnlabeledDataset(X_test)
    test_unlabeled_loader = DataLoader(test_unlabeled_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=(device.type == 'cuda'))
    
    # Evaluation test loader
    test_eval_dataset = StormDataset(X_test, y_test, raw_test_coords)
    test_eval_loader = DataLoader(test_eval_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Initialize VIME Model
    model = VIMEModel(
        input_dim=input_dim,
        target_dim=target_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    # Load pre-trained weights if available
    pretrained_loaded = False
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"\n[PRETRAIN] Loading pretrained VIME weights from '{args.pretrained_path}'...")
        try:
            ckpt = torch.load(args.pretrained_path, map_location=device)
            if 'encoder_state_dict' in ckpt:
                model.encoder.load_state_dict(ckpt['encoder_state_dict'])
                print("  --> Pretrained Encoder weights loaded successfully.")
            elif 'model_state_dict' in ckpt:
                # Load compatible weights
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items() if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print(f"  --> Loaded {len(pretrained_dict)} matching layers from pretrained model.")
            pretrained_loaded = True
        except Exception as e:
            print(f"[WARNING] Could not load pretrained weights: {e}. Training from scratch.")
    else:
        print(f"\n[INFO] No pretrained checkpoint found at '{args.pretrained_path}'. Initializing VIME weights from scratch.")
        
    # 4. Empirical Marginal Corruptor (using background features from train + test data)
    background_data = np.vstack([X_train, X_test])
    corruptor = VIMECorruptor(background_data=background_data, p_m=args.p_m)
    
    # 5. Loss functions, Optimizer, and LR Scheduler
    criterion_supervised = nn.MSELoss()
    criterion_consistency = nn.MSELoss()
    criterion_recon = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    best_eval_loss = float('inf')
    best_epoch = -1
    
    print("\n" + "=" * 70)
    print(f"Starting VIME Fine-tuning for {args.epochs} epochs...")
    print(f"  - Supervised Task Loss (MSE)")
    print(f"  - Test Data Consistency Regularization (K={args.K}, weight beta={args.beta}, p_m={args.p_m})")
    print(f"  - Test Feature Reconstruction Adaptation (weight gamma={args.gamma})")
    print("=" * 70 + "\n")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        train_sup_loss_accum = 0.0
        train_cons_loss_accum = 0.0
        train_recon_loss_accum = 0.0
        train_total_loss_accum = 0.0
        
        # Cycle through test unlabeled data loader to match labeled training batches
        test_iter = cycle(test_unlabeled_loader)
        num_batches = len(train_loader)
        
        for batch_idx, (batch_x_train, batch_y_train) in enumerate(train_loader):
            batch_x_train = batch_x_train.to(device)
            batch_y_train = batch_y_train.to(device)
            
            batch_x_test = next(test_iter).to(device)
            
            optimizer.zero_grad()
            
            # --- 1. Supervised Task Loss on Training Data ---
            pred_y_train = model(batch_x_train)
            loss_sup = criterion_supervised(pred_y_train, batch_y_train)
            
            # --- 2. Consistency Regularization on Test Data ---
            # Generate K perturbed versions of the test batch using marginal corruption
            # batch_x_test: (B, D)
            pred_y_test, recon_x_test = model.forward_with_recon(batch_x_test)
            
            loss_consistency = torch.tensor(0.0, device=device)
            if args.beta > 0.0 and args.K > 0:
                corrupted_test_k = corruptor.generate_k_corruptions(batch_x_test, K=args.K, device=device)
                for k in range(args.K):
                    corrupted_sample = corrupted_test_k[k]  # (B, D)
                    pred_y_corrupted = model(corrupted_sample)
                    loss_consistency = loss_consistency + criterion_consistency(pred_y_corrupted, pred_y_test.detach())
                loss_consistency = loss_consistency / args.K
                
            # --- 3. Test Domain Reconstruction Adaptation ---
            loss_recon = torch.tensor(0.0, device=device)
            if args.gamma > 0.0:
                loss_recon = criterion_recon(recon_x_test, batch_x_test)
                
            # --- Combined Total Objective ---
            total_loss = loss_sup + args.beta * loss_consistency + args.gamma * loss_recon
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_sup_loss_accum += loss_sup.item()
            train_cons_loss_accum += loss_consistency.item()
            train_recon_loss_accum += loss_recon.item()
            train_total_loss_accum += total_loss.item()
            
        scheduler.step()
        
        avg_sup_loss = train_sup_loss_accum / num_batches
        avg_cons_loss = train_cons_loss_accum / num_batches
        avg_recon_loss = train_recon_loss_accum / num_batches
        avg_total_loss = train_total_loss_accum / num_batches
        
        # --- Validation & Evaluation on Test Set ---
        model.eval()
        test_sup_loss_accum = 0.0
        all_test_preds = []
        all_test_targets = []
        
        with torch.no_grad():
            for bx, by in test_eval_loader:
                bx = bx.to(device)
                by = by.to(device)
                py = model(bx)
                l = criterion_supervised(py, by)
                test_sup_loss_accum += l.item()
                all_test_preds.append(py.cpu().numpy())
                all_test_targets.append(by.cpu().numpy())
                
        avg_test_eval_loss = test_sup_loss_accum / len(test_eval_loader)
        is_best = avg_test_eval_loss < best_eval_loss
        
        if is_best:
            best_eval_loss = avg_test_eval_loss
            best_epoch = epoch
            best_ckpt_path = os.path.join(args.checkpoint_dir, 'best_vime_finetuned.pth')
            torch.save({
                'epoch': epoch,
                'input_dim': input_dim,
                'target_dim': target_dim,
                'latent_dim': args.latent_dim,
                'hidden_dim': args.hidden_dim,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_eval_loss': avg_test_eval_loss,
                'feature_names': feature_names,
                'target_names': target_names,
                'scaler_path': scaler_path
            }, best_ckpt_path)
            
        if epoch % args.print_freq == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Total: {avg_total_loss:.5f} | "
                  f"Sup MSE: {avg_sup_loss:.5f} | Consist: {avg_cons_loss:.5f} | "
                  f"Test MSE: {avg_test_eval_loss:.5f} {'*' if is_best else ''}")
            
    print("\n" + "=" * 70)
    print(f"Fine-tuning Complete! Best Test MSE: {best_eval_loss:.6f} at epoch {best_epoch}")
    print("=" * 70)
    
    # 6. Final Comprehensive Evaluation using Best Checkpoint
    best_ckpt_path = os.path.join(args.checkpoint_dir, 'best_vime_finetuned.pth')
    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best fine-tuned model from epoch {checkpoint['epoch']} for final evaluation.")

    model.eval()
    test_preds_list = []
    with torch.no_grad():
        for bx, _ in test_eval_loader:
            bx = bx.to(device)
            py = model(bx)
            test_preds_list.append(py.cpu().numpy())
            
    y_test_pred = np.vstack(test_preds_list)
    
    # Compute Haversine distance errors and Cosine Similarities across all horizons
    eval_metrics = evaluate_storm_predictions(
        y_true=y_test,
        y_pred=y_test_pred,
        raw_coords=raw_test_coords,
        target_names=target_names
    )
    
    print("\n" + "=" * 70)
    print("                      FINAL EVALUATION RESULTS ON TEST DATA           ")
    print("=" * 70)
    
    result_lines = ["--- VIME Fine-tuning Evaluation Results ---\n"]
    for h in ['6h', '12h', '18h', '24h']:
        m = eval_metrics[h]
        line = (
            f"Average Cosine Similarity for {h}: {m['cosine_similarity']:.4f} | "
            f"Distance Error: Mean = {m['mean_distance_km']:.2f} km, "
            f"Median = {m['median_distance_km']:.2f} km, "
            f"Min = {m['min_distance_km']:.2f} km, "
            f"Max = {m['max_distance_km']:.2f} km"
        )
        print(line)
        result_lines.append(line + "\n")
        
    # Save evaluation summary
    with open(args.result_path, 'w') as f:
        f.writelines(result_lines)
    print(f"\n[OUTPUT] Evaluation results saved to: '{args.result_path}'")
    print(f"[OUTPUT] Best model checkpoint saved to: '{best_ckpt_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VIME Fine-Tuning on Test Data")
    parser.add_argument('--train_path', type=str, default=DEFAULT_TRAIN_PATH, help="Path to train.csv")
    parser.add_argument('--test_path', type=str, default=DEFAULT_TEST_PATH, help="Path to test.csv")
    parser.add_argument('--valid_path', type=str, default=DEFAULT_VALID_PATH, help="Path to valid.csv")
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR, help="Directory to save checkpoints")
    parser.add_argument('--pretrained_path', type=str, default=DEFAULT_PRETRAINED_PATH, help="Path to pretrained weights")
    parser.add_argument('--result_path', type=str, default=RESULT_PATH, help="Path to save evaluation text")
    parser.add_argument('--epochs', type=int, default=50, help="Number of fine-tuning epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--beta', type=float, default=1.0, help="Weight for test consistency regularization")
    parser.add_argument('--gamma', type=float, default=0.2, help="Weight for test feature reconstruction adaptation")
    parser.add_argument('--K', type=int, default=3, help="Number of perturbed test samples for consistency loss")
    parser.add_argument('--p_m', type=float, default=0.3, help="Corruption probability for consistency generation")
    parser.add_argument('--latent_dim', type=int, default=128, help="Latent representation dimension")
    parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dimension in layers")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate")
    parser.add_argument('--print_freq', type=int, default=5, help="Epoch print frequency")
    parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA")
    
    args = parser.parse_args()
    finetune_vime(args)
