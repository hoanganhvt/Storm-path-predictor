import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def preprocess_data(df):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    num_cols = ['lat', 'lon', 'pressure_hpa', 'max_wind_kt', 'dir_50kt', 'rad_50kt_long_nm', 'rad_50kt_short_nm', 'dir_30kt', 'rad_30kt_long_nm', 'rad_30kt_short_nm']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(-1.0, inplace=True)
    
    feature_cols = ['month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos'] + num_cols
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    return df, feature_cols

def create_sequences(df, feature_cols, seq_len=4, horizons=4):
    X = []
    Y = []
    last_lat_lon = []
    
    lat_idx = feature_cols.index('lat')
    lon_idx = feature_cols.index('lon')
    
    for storm_id, group in df.groupby('international_id', sort=False):
        group = group.sort_values('time')
        features = group[feature_cols].values.astype(np.float32)
        
        n_steps = len(features)
        for i in range(n_steps - seq_len - horizons + 1):
            x = features[i : i + seq_len]
            base_lat = features[i + seq_len - 1, lat_idx]
            base_lon = features[i + seq_len - 1, lon_idx]
            
            y = []
            for h in range(horizons):
                future_lat = features[i + seq_len + h, lat_idx]
                future_lon = features[i + seq_len + h, lon_idx]
                y.extend([future_lat - base_lat, future_lon - base_lon])
                
            X.append(x)
            Y.append(y)
            last_lat_lon.append([base_lat, base_lon])
            
    return np.array(X), np.array(Y, dtype=np.float32), np.array(last_lat_lon, dtype=np.float32)

class StormDataset(Dataset):
    def __init__(self, X, Y, last_lat_lon):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.last_lat_lon = torch.tensor(last_lat_lon, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.last_lat_lon[idx]
