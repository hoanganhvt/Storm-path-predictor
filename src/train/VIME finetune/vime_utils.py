import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Optional, Union


def haversine_np(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) using NumPy.
    Returns distance in kilometers.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def compute_cosine_similarity(y_true_lat: np.ndarray, y_true_lon: np.ndarray,
                              y_pred_lat: np.ndarray, y_pred_lon: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes directional cosine similarity between true and predicted displacement vectors.
    """
    y_true_vec = np.column_stack((y_true_lat, y_true_lon))
    y_pred_vec = np.column_stack((y_pred_lat, y_pred_lon))
    
    dot_product = np.sum(y_true_vec * y_pred_vec, axis=1)
    norm_true = np.linalg.norm(y_true_vec, axis=1)
    norm_pred = np.linalg.norm(y_pred_vec, axis=1)
    
    denom = norm_true * norm_pred
    similarity = np.zeros_like(dot_product)
    valid_idx = denom > 1e-8
    similarity[valid_idx] = dot_product[valid_idx] / denom[valid_idx]
    
    avg_cosine_sim = float(np.mean(similarity))
    return avg_cosine_sim, similarity


def feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Performs domain-specific feature engineering for storm path prediction:
    - Cyclical month, day, hour sin/cos transformations
    - Multi-horizon displacement targets (6h, 12h, 18h, 24h)
    - Past lag features (6h, 12h, 18h, 24h) for position and wind properties
    - Past displacement velocity vectors
    """
    df = df.copy()
    
    # 1. Cyclical time features
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31.0)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    # 2. Multi-horizon targets (Assuming 6-hourly intervals: shifts -1, -2, -3, -4)
    horizons = {'6h': -1, '12h': -2, '18h': -3, '24h': -4}
    for h_name, shift_val in horizons.items():
        df[f'future_lat_{h_name}'] = df.groupby('international_id')['lat'].shift(shift_val)
        df[f'future_lon_{h_name}'] = df.groupby('international_id')['lon'].shift(shift_val)
        df[f'delta_lat_{h_name}'] = df[f'future_lat_{h_name}'] - df['lat']
        df[f'delta_lon_{h_name}'] = df[f'future_lon_{h_name}'] - df['lon']
        df.drop(columns=[f'future_lat_{h_name}', f'future_lon_{h_name}'], inplace=True)
        
    # 3. Lag features for position, pressure, and wind properties
    lag_props = ['lat', 'lon', 'pressure_hpa', 'max_wind_kt', 'dir_50kt', 'rad_50kt_long_nm', 'rad_50kt_short_nm', 
                 'dir_30kt', 'rad_30kt_long_nm', 'rad_30kt_short_nm']
    
    lags = {'6h': 1, '12h': 2, '18h': 3, '24h': 4}
    for col in lag_props:
        if col in df.columns:
            for l_name, shift_val in lags.items():
                df[f'{col}_lag_{l_name}'] = df.groupby('international_id')[col].shift(shift_val)
                if col in ['lat', 'lon']:
                    df[f'past_delta_{col}_{l_name}'] = df[col] - df[f'{col}_lag_{l_name}']
                    
    target_cols = []
    for h in ['6h', '12h', '18h', '24h']:
        target_cols.extend([f'delta_lat_{h}', f'delta_lon_{h}'])
    
    # Remove records where forward displacement targets are unavailable
    df.dropna(subset=target_cols, inplace=True)
    
    # Drop non-feature and administrative columns
    cols_to_drop = ['tc_number', 'name', 'grade_name', 'revision_date', 'year', 
                    'time_diff_hours', 'time', 'month', 'day', 'hour', 'flag_last', 
                    'dir_50kt_name', 'dir_30kt_name', 'landfall_flag']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    object_cols = df.select_dtypes(include=['object']).columns
    object_cols = [c for c in object_cols if c != 'international_id']
    df.drop(columns=object_cols, inplace=True)
    
    df = df.apply(pd.to_numeric, errors='coerce')
    return df, target_cols


class StormDataset(Dataset):
    """PyTorch Dataset for supervised labeled storm tabular data."""
    def __init__(self, X: np.ndarray, y: np.ndarray, raw_coords: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.raw_coords = raw_coords  # Array of (lat, lon) for distance evaluation

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class StormUnlabeledDataset(Dataset):
    """PyTorch Dataset for unlabeled / test storm tabular features (for VIME consistency & pretraining)."""
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]


class VIMECorruptor:
    """
    VIME Empirical Marginal Distribution Corruption Generator:
    1. Samples a binary mask m ~ Bernoulli(p_m).
    2. For features where m=1, replaces value with a random sample drawn
       from the empirical marginal distribution of that feature across the dataset.
    3. Produces corrupted tensor x_tilde and mask m.
    """
    def __init__(self, background_data: Union[np.ndarray, torch.Tensor], p_m: float = 0.3):
        """
        background_data: Array of shape (N, D) containing the empirical distribution of features.
        p_m: Probability of corrupting each feature.
        """
        if isinstance(background_data, np.ndarray):
            self.background = torch.tensor(background_data, dtype=torch.float32)
        else:
            self.background = background_data.clone().detach().to(dtype=torch.float32)
        
        self.p_m = p_m
        self.n_samples, self.dim = self.background.shape

    def corrupt(self, x: torch.Tensor, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Corrupts a batch x of shape (B, D).
        Returns:
            corrupted_x: torch.Tensor of shape (B, D)
            mask: torch.Tensor of shape (B, D) with binary entries (1 = corrupted, 0 = original)
        """
        batch_size, dim = x.shape
        if device is None:
            device = x.device

        # Generate binary mask from Bernoulli(p_m)
        mask = (torch.rand(batch_size, dim, device=device) < self.p_m).float()

        # Sample replacement values from empirical marginal distribution
        rand_indices = torch.randint(0, self.n_samples, (batch_size, dim), device=device)
        bg = self.background.to(device)
        
        # Gather random marginal samples per column
        marginal_samples = torch.gather(bg, 0, rand_indices)

        # Corrupted input: x * (1 - mask) + marginal_samples * mask
        corrupted_x = x * (1.0 - mask) + marginal_samples * mask
        return corrupted_x, mask

    def generate_k_corruptions(self, x: torch.Tensor, K: int = 3, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generates K independently corrupted variants of batch x for consistency regularization.
        Returns:
            corrupted_k: torch.Tensor of shape (K, B, D)
        """
        batch_size, dim = x.shape
        if device is None:
            device = x.device
        
        corrupted_list = []
        for _ in range(K):
            corrupted_x, _ = self.corrupt(x, device=device)
            corrupted_list.append(corrupted_x.unsqueeze(0))
        
        return torch.cat(corrupted_list, dim=0)


def load_and_preprocess_storm_data(
    train_path: str,
    test_path: str,
    valid_path: Optional[str] = None
) -> Dict:
    """
    Loads raw storm datasets, applies feature engineering, handles missing values,
    and applies standard normalization fitted on training data.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train dataset not found at: '{train_path}'")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at: '{test_path}'")

    print(f"[DATA] Loading train data from {train_path}...")
    df_train_raw = pd.read_csv(train_path, low_memory=False)
    df_train_raw['time'] = pd.to_datetime(df_train_raw['time'])
    df_train_raw = df_train_raw.sort_values(by=['international_id', 'time']).copy()
    
    print(f"[DATA] Loading test data from {test_path}...")
    df_test_raw = pd.read_csv(test_path, low_memory=False)
    df_test_raw['time'] = pd.to_datetime(df_test_raw['time'])
    df_test_raw = df_test_raw.sort_values(by=['international_id', 'time']).copy()

    df_train, target_cols = feature_engineering(df_train_raw)
    df_test, _ = feature_engineering(df_test_raw)
    
    df_valid = None
    if valid_path and os.path.exists(valid_path):
        print(f"[DATA] Loading valid data from {valid_path}...")
        df_valid_raw = pd.read_csv(valid_path, low_memory=False)
        df_valid_raw['time'] = pd.to_datetime(df_valid_raw['time'])
        df_valid_raw = df_valid_raw.sort_values(by=['international_id', 'time']).copy()
        df_valid, _ = feature_engineering(df_valid_raw)

    # Separate features and targets
    non_feature_cols = target_cols + ['international_id']
    feature_cols = [c for c in df_train.columns if c not in non_feature_cols]

    # Store raw test coordinates (lat, lon) for physical distance verification
    raw_test_coords = df_test[['lat', 'lon']].values
    raw_train_coords = df_train[['lat', 'lon']].values

    X_train_df = df_train[feature_cols].copy()
    y_train_df = df_train[target_cols].copy()
    
    X_test_df = df_test[feature_cols].copy()
    y_test_df = df_test[target_cols].copy()

    # Impute missing values (e.g. wind radii NaNs) using train median
    median_imputer = X_train_df.median()
    X_train_df = X_train_df.fillna(median_imputer).fillna(0.0)
    X_test_df = X_test_df.fillna(median_imputer).fillna(0.0)

    # Standard Scaler fitted on training features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df.values)
    X_test_scaled = scaler.transform(X_test_df.values)
    
    y_train_arr = y_train_df.values
    y_test_arr = y_test_df.values

    X_valid_scaled = None
    y_valid_arr = None
    raw_valid_coords = None
    if df_valid is not None:
        X_valid_df = df_valid[feature_cols].copy().fillna(median_imputer).fillna(0.0)
        y_valid_df = df_valid[target_cols].copy()
        X_valid_scaled = scaler.transform(X_valid_df.values)
        y_valid_arr = y_valid_df.values
        raw_valid_coords = df_valid[['lat', 'lon']].values

    return {
        'X_train': X_train_scaled,
        'y_train': y_train_arr,
        'raw_train_coords': raw_train_coords,
        'X_test': X_test_scaled,
        'y_test': y_test_arr,
        'raw_test_coords': raw_test_coords,
        'X_valid': X_valid_scaled,
        'y_valid': y_valid_arr,
        'raw_valid_coords': raw_valid_coords,
        'feature_names': feature_cols,
        'target_names': target_cols,
        'scaler': scaler,
        'input_dim': len(feature_cols),
        'target_dim': len(target_cols)
    }


def evaluate_storm_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    raw_coords: np.ndarray,
    target_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates predictions against ground truth for all horizons (6h, 12h, 18h, 24h).
    Computes:
    - Directional Cosine Similarity
    - Haversine Distance Errors in km (Mean, Median, Min, Max)
    """
    results = {}
    current_lat = raw_coords[:, 0]
    current_lon = raw_coords[:, 1]
    
    horizons = ['6h', '12h', '18h', '24h']
    
    for h in horizons:
        lat_idx = target_names.index(f'delta_lat_{h}')
        lon_idx = target_names.index(f'delta_lon_{h}')
        
        y_true_lat = y_true[:, lat_idx]
        y_true_lon = y_true[:, lon_idx]
        y_pred_lat = y_pred[:, lat_idx]
        y_pred_lon = y_pred[:, lon_idx]
        
        # Cosine Similarity
        avg_cos_sim, _ = compute_cosine_similarity(y_true_lat, y_true_lon, y_pred_lat, y_pred_lon)
        
        # Physical coordinates
        true_dest_lat = current_lat + y_true_lat
        true_dest_lon = current_lon + y_true_lon
        pred_dest_lat = current_lat + y_pred_lat
        pred_dest_lon = current_lon + y_pred_lon
        
        # Haversine Distance Error
        dist_errors = haversine_np(true_dest_lon, true_dest_lat, pred_dest_lon, pred_dest_lat)
        
        mean_dist = float(np.mean(dist_errors))
        median_dist = float(np.median(dist_errors))
        min_dist = float(np.min(dist_errors))
        max_dist = float(np.max(dist_errors))
        
        results[h] = {
            'cosine_similarity': avg_cos_sim,
            'mean_distance_km': mean_dist,
            'median_distance_km': median_dist,
            'min_distance_km': min_dist,
            'max_distance_km': max_dist
        }
        
    return results
