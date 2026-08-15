import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

import sys

# 1. Import train.csv into a dataframe
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Datasets/Storm_data/train.csv'))
if not os.path.exists(data_path):
    print(f"[ERROR] Training dataset not found at: '{data_path}'")
    print("[ABORT] Cannot proceed with training. Aborting.")
    sys.exit(1)

try:
    df = pd.read_csv(data_path)
    print(f"[SUCCESS] Loaded training dataset with {len(df):,} records from: '{data_path}'")
except Exception as err:
    print(f"[ERROR] Failed to read train.csv: {err}")
    print("[ABORT] Aborting.")
    sys.exit(1)

# 2. Select specified features
base_features = ['time', 'grade', 'lat', 'lon', 'pressure_hpa']
wind_features = ['max_wind_kt', 'dir_50kt', 'rad_50kt_long_nm', 'rad_50kt_short_nm', 
                 'dir_30kt', 'rad_30kt_long_nm', 'rad_30kt_short_nm']
selected_cols = base_features + wind_features

missing_cols = [c for c in selected_cols if c not in df.columns]
if missing_cols:
    print(f"[ERROR] Dataset is missing required columns: {missing_cols}")
    print("[ABORT] Schema mismatch. Aborting.")
    sys.exit(1)

df = df[selected_cols]


# 3. Feature engineering the time features
df['time'] = pd.to_datetime(df['time'])
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.hour
df.drop('time', axis=1, inplace=True)

# 4. Ensure missing values (-1 or NaN) are treated as NaN so that -1 is NEVER scaled
# Scikit-learn ignores NaNs during fit and preserves NaNs during transform
#df_clean = df.replace(-1, np.nan).replace(-1.0, np.nan)


scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_values, columns=df.columns)

# 5. Fill all empty/missing features with -1.0 AFTER scaling (strictly outside [0, 1])
scaled_df.fillna(-1.0, inplace=True)

# Define PyTorch Dataset
class StormDataset(Dataset):

    def __init__(self, dataframe):
        self.data = dataframe.values.astype(np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])


# 5. Write a simple Autoencoder in PyTorch
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

# 5. Device configuration (CUDA if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Prepare DataLoader
dataset = StormDataset(scaled_df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=(device.type == 'cuda'))

# Initialize Model, Loss Function, and Optimizer
input_size = len(scaled_df.columns)
model = SimpleAutoencoder(input_dim=input_size, latent_dim=64).to(device)

# 6. Write a "good" loss function
# Custom Masked MSE Loss:
# Since we filled missing values with -1, standard MSE will force the network to reconstruct -1,
# which skews the learning. A masked loss ignores the -1 values so the model only learns from actual data.
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none') # Compute loss element-wise
        
    def forward(self, pred, target):
        # Create a mask for valid non-missing elements (valid data is in [0, 1], missing is -1)
        mask = (target >= 0.0).float()
        
        # Calculate element-wise MSE
        loss = self.mse(pred, target)
        
        # Apply mask
        masked_loss = loss * mask
        
        # Average the loss only over the valid elements
        return masked_loss.sum() / (mask.sum() + 1e-8)

criterion = MaskedMSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 7. Checkpointing Directory and Scaler Export
CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints'))
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Export the fitted MinMaxScaler alongside model files
SCALER_PATH = os.path.join(CHECKPOINT_DIR, 'scaler.pkl')
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Exported MinMaxScaler to: {SCALER_PATH}")

def save_checkpoint(state, filename, checkpoint_dir=CHECKPOINT_DIR):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"  --> Saved: {filename}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint '{checkpoint_path}' (Epoch {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A'):.6f})")
    return checkpoint

# Training Loop with Checkpointing
def train_autoencoder(num_epochs=50, checkpoint_dir=CHECKPOINT_DIR):
    model.train()
    best_loss = float('inf')
    best_epoch = -1
    
    print(f"\nStarting training for {num_epochs} epochs on {device}...")
    print(f"Checkpoints will be saved to: {checkpoint_dir}\n")
    
    for epoch in range(num_epochs):
        epoch_num = epoch + 1
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            best_epoch = epoch_num
            
        # Display loss with high precision (e.g. 6 decimal places / scientific)
        loss_str = f"{avg_loss:.6f}" if avg_loss >= 1e-4 else f"{avg_loss:.4e}"
        print(f"Epoch [{epoch_num:02d}/{num_epochs:02d}] - Loss: {loss_str} {'*' if is_best else ''}")
        
        # Prepare checkpoint payload
        checkpoint_state = {
            'epoch': epoch_num,
            'input_dim': input_size,
            'latent_dim': 64,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
            'columns': list(scaled_df.columns),
            'scaler_path': SCALER_PATH
        }
        
        # 1. Export this epoch's checkpoint after every single epoch
        save_checkpoint(checkpoint_state, filename=f"checkpoint_{epoch_num}.pth", checkpoint_dir=checkpoint_dir)
        
        # 2. If this epoch is better than previous best, also export best_autoencoder.pth
        if is_best:
            save_checkpoint(checkpoint_state, filename="best_autoencoder.pth", checkpoint_dir=checkpoint_dir)
            
    best_loss_str = f"{best_loss:.6f}" if best_loss >= 1e-4 else f"{best_loss:.4e}"
    print(f"\nTraining completed!")
    print(f"Best loss achieved: {best_loss_str} at epoch {best_epoch}")
    print(f"Best model saved to: {os.path.join(checkpoint_dir, 'best_autoencoder.pth')}")


if __name__ == "__main__":
    train_autoencoder(num_epochs=50)





