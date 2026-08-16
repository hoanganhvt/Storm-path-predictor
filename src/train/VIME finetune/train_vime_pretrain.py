import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vime_model import VIMEModel
from vime_utils import load_and_preprocess_storm_data, StormUnlabeledDataset, VIMECorruptor

# Paths
DEFAULT_TRAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Datasets/Storm_data/train.csv'))
DEFAULT_TEST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Datasets/Storm_data/test.csv'))
DEFAULT_VALID_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Datasets/Storm_data/valid.csv'))
CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints'))


def train_vime_pretrain(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"=== VIME Self-Supervised Pretraining ===")
    print(f"Using device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load and preprocess data
    data_dict = load_and_preprocess_storm_data(
        train_path=args.train_path,
        test_path=args.test_path,
        valid_path=args.valid_path if os.path.exists(args.valid_path) else None
    )
    
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    input_dim = data_dict['input_dim']
    target_dim = data_dict['target_dim']
    scaler = data_dict['scaler']
    
    # Pre-training can utilize all available tabular feature vectors (train + test unlabeled features) or train features only
    if not args.train_only:
        X_pretrain = np.vstack([X_train, X_test])
        print(f"Pretraining dataset: Combined Train + Test features ({len(X_pretrain):,} samples, {input_dim} features)")
    else:
        X_pretrain = X_train
        print(f"Pretraining dataset: Train features only ({len(X_pretrain):,} samples, {input_dim} features)")
        
    # Save fitted scaler
    scaler_path = os.path.join(args.checkpoint_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Exported fitted StandardScaler to: {scaler_path}")

    # Create dataset & corruptor
    pretrain_dataset = StormUnlabeledDataset(X_pretrain)
    dataloader = DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == 'cuda'))
    
    corruptor = VIMECorruptor(background_data=X_pretrain, p_m=args.p_m)
    
    # Initialize Model
    model = VIMEModel(
        input_dim=input_dim,
        target_dim=target_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    # Loss functions & Optimizer
    criterion_mask = nn.BCELoss()
    criterion_recon = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    best_loss = float('inf')
    best_epoch = -1
    
    print(f"\nStarting Self-Supervised Pretraining for {args.epochs} epochs...")
    print(f"Hyperparameters: p_m={args.p_m}, alpha={args.alpha}, batch_size={args.batch_size}, lr={args.lr}\n")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_mask_loss = 0.0
        total_recon_loss = 0.0
        
        for batch_x in dataloader:
            batch_x = batch_x.to(device)
            
            # Generate corruption and binary mask
            corrupted_x, mask = corruptor.corrupt(batch_x, device=device)
            
            optimizer.zero_grad()
            mask_pred, recon_x = model.forward_pretrain(corrupted_x)
            
            loss_mask = criterion_mask(mask_pred, mask)
            loss_recon = criterion_recon(recon_x, batch_x)
            loss = loss_mask + args.alpha * loss_recon
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_mask_loss += loss_mask.item()
            total_recon_loss += loss_recon.item()
            
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        avg_mask_loss = total_mask_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            best_epoch = epoch
            # Save best pre-trained checkpoint
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'vime_pretrained.pth')
            torch.save({
                'epoch': epoch,
                'input_dim': input_dim,
                'latent_dim': args.latent_dim,
                'hidden_dim': args.hidden_dim,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'feature_names': data_dict['feature_names'],
                'target_names': data_dict['target_names'],
                'scaler_path': scaler_path
            }, best_checkpoint_path)
            
        if epoch % args.print_freq == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Total Loss: {avg_loss:.6f} | "
                  f"Mask BCE: {avg_mask_loss:.6f} | Recon MSE: {avg_recon_loss:.6f} {'*' if is_best else ''}")
            
    print(f"\n[COMPLETE] Pretraining finished! Best Loss: {best_loss:.6f} at epoch {best_epoch}")
    print(f"Pretrained model saved to: {os.path.join(args.checkpoint_dir, 'vime_pretrained.pth')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VIME Self-Supervised Pre-Training")
    parser.add_argument('--train_path', type=str, default=DEFAULT_TRAIN_PATH, help="Path to train.csv")
    parser.add_argument('--test_path', type=str, default=DEFAULT_TEST_PATH, help="Path to test.csv")
    parser.add_argument('--valid_path', type=str, default=DEFAULT_VALID_PATH, help="Path to valid.csv")
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR, help="Directory to save checkpoints")
    parser.add_argument('--epochs', type=int, default=50, help="Number of pretraining epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--p_m', type=float, default=0.3, help="Corruption probability (mask rate)")
    parser.add_argument('--alpha', type=float, default=2.0, help="Weight for reconstruction loss")
    parser.add_argument('--latent_dim', type=int, default=128, help="Latent representation dimension")
    parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dimension in layers")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate")
    parser.add_argument('--train_only', action='store_true', help="Pretrain on train dataset only (exclude test features)")
    parser.add_argument('--print_freq', type=int, default=5, help="Epoch print frequency")
    parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA")
    
    args = parser.parse_args()
    train_vime_pretrain(args)
