import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def haversine_distance_km(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Computes Great-Circle (Haversine) distance in kilometers between two GPS coordinates.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0 * c  # Earth mean radius in km


def prepare_storm_dataframe(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Performs feature selection and time feature engineering matching SimpleAutoencoder:
    1. Select base features: ['time', 'grade', 'lat', 'lon', 'pressure_hpa']
    2. Select wind features: ['max_wind_kt', 'dir_50kt', 'rad_50kt_long_nm', 'rad_50kt_short_nm', 'dir_30kt', 'rad_30kt_long_nm', 'rad_30kt_short_nm']
    3. Feature engineer 'time' into 'year', 'month', 'day', 'hour', then drop 'time'.
    
    Returns
    -------
    df : pd.DataFrame
        Processed DataFrame sorted by storm ID and time.
    feature_cols : List[str]
        List of 15 engineered feature column names in standard order.
    """
    df = df_raw.copy()
    
    base_features = ['time', 'grade', 'lat', 'lon', 'pressure_hpa']
    wind_features = [
        'max_wind_kt', 'dir_50kt', 'rad_50kt_long_nm', 'rad_50kt_short_nm',
        'dir_30kt', 'rad_30kt_long_nm', 'rad_30kt_short_nm'
    ]
    required_cols = ['international_id'] + base_features + wind_features
    
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Dataset missing required columns: {missing_cols}")
        
    df = df[required_cols].copy()
    
    # Time feature engineering
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    
    # Order of features matching pretrained autoencoder weights
    feature_cols = [
        'grade', 'lat', 'lon', 'pressure_hpa',
        'max_wind_kt', 'dir_50kt', 'rad_50kt_long_nm', 'rad_50kt_short_nm',
        'dir_30kt', 'rad_30kt_long_nm', 'rad_30kt_short_nm',
        'year', 'month', 'day', 'hour'
    ]
    
    # Sort chronologically per storm
    df = df.sort_values(by=['international_id', 'time']).reset_index(drop=True)
    df.drop('time', axis=1, inplace=True)
    
    return df, feature_cols


def scale_storm_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler = None,
    fit_scaler: bool = False,
    use_standard_scaler: bool = False
) -> Tuple[pd.DataFrame, Union[StandardScaler, MinMaxScaler]]:
    """
    Scales tabular storm features with fitted scaler (StandardScaler or MinMaxScaler),
    preserving -1.0 for missing spots.
    """
    df_feat = df[feature_cols].copy()
    df_clean = df_feat.replace(-1, np.nan).replace(-1.0, np.nan)
    
    if fit_scaler or scaler is None:
        scaler = StandardScaler() if use_standard_scaler else MinMaxScaler()
        scaled_vals = scaler.fit_transform(df_clean)
    else:
        scaled_vals = scaler.transform(df_clean)
        
    scaled_df = pd.DataFrame(scaled_vals, columns=feature_cols, index=df.index)
    scaled_df.fillna(-1.0, inplace=True)
    
    return scaled_df, scaler


def build_storm_trajectory_sequences(
    df_meta: pd.DataFrame,
    df_scaled: pd.DataFrame,
    feature_cols: List[str],
    min_history_steps: int = 2,
    num_future_steps: int = 4
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares training/testing sequences split by storm ID:
    Each state is 6 hours apart.
    With min_history_steps = 2 (at least 12 hours of past data) and num_future_steps = 4:
    For a storm with states [1, 2, 3, 4, 5, 6, 7, 8, 9]:
      - x: [1, 2],          y: [3, 4, 5, 6]
      - x: [1, 2, 3],       y: [4, 5, 6, 7]
      - x: [1, 2, 3, 4],    y: [5, 6, 7, 8]
      - x: [1, 2, 3, 4, 5], y: [6, 7, 8, 9]
      
    Returns
    -------
    X_list : List[np.ndarray]
        List of historical feature arrays of varying sequence lengths (t, num_features).
    Y_arr : np.ndarray
        Array of target coordinate displacements of shape (N, num_future_steps, 2).
    coords_arr : np.ndarray
        Array of current storm coordinates [lat, lon] at prediction point, shape (N, 2).
    ids_arr : np.ndarray
        Array of storm IDs.
    """
    X_list = []
    Y_list = []
    coords_list = []
    storm_ids = []
    
    for storm_id, group in df_meta.groupby('international_id', sort=False):
        group_idx = group.index
        feat_matrix = df_scaled.loc[group_idx, feature_cols].values
        coords_matrix = group[['lat', 'lon']].values
        n_states = len(group)
        
        # Needs at least (min_history_steps + num_future_steps) states in the storm
        if n_states < min_history_steps + num_future_steps:
            continue
            
        for t in range(min_history_steps, n_states - num_future_steps + 1):
            # Input history sequence: states from start up to t-1
            x_seq = feat_matrix[0 : t]
            
            # Current storm coordinate at the prediction point (state t-1)
            curr_lat, curr_lon = coords_matrix[t - 1]
            
            # Future states coordinates (e.g. states t, t+1, t+2, t+3)
            future_coords = coords_matrix[t : t + num_future_steps]  # (4, 2)
            
            # Compute coordinate shifts (delta lat, delta lon) for the future steps
            y_deltas = future_coords - np.array([curr_lat, curr_lon])  # (4, 2)
            
            X_list.append(x_seq)
            Y_list.append(y_deltas)
            coords_list.append(np.array([curr_lat, curr_lon], dtype=np.float32))
            storm_ids.append(storm_id)
            
    Y_arr = np.array(Y_list, dtype=np.float32)
    coords_arr = np.array(coords_list, dtype=np.float32)
    ids_arr = np.array(storm_ids)
    
    return X_list, Y_arr, coords_arr, ids_arr


def generate_evaluation_report(
    preds_arr: np.ndarray,
    targets_arr: np.ndarray,
    coords_arr: np.ndarray,
    num_future_steps: int = 4,
    output_report_path: Optional[str] = None
) -> str:
    """
    Generates and saves comprehensive forecast metrics:
    - MAE/RMSE in degrees for Latitude & Longitude
    - Vector Cosine Similarity
    - Great-Circle Haversine Distance Error in Kilometers (Mean, Median, 90th percentile, Max)
    - Sample trajectory inspection
    """
    horizons = ['6h', '12h', '18h', '24h'][:num_future_steps]
    report_lines = []
    
    header = "=" * 90 + "\n"
    header += "       STORM PATH PREDICTOR (LNN + AUTOENCODER) EVALUATION REPORT\n"
    header += "=" * 90
    report_lines.append(header)
    
    summary_rows = []
    
    for h_idx, h_name in enumerate(horizons):
        true_dlat = targets_arr[:, h_idx, 0]
        true_dlon = targets_arr[:, h_idx, 1]
        pred_dlat = preds_arr[:, h_idx, 0]
        pred_dlon = preds_arr[:, h_idx, 1]
        
        # Latitude & Longitude Errors in Degrees
        mae_lat = mean_absolute_error(true_dlat, pred_dlat)
        mae_lon = mean_absolute_error(true_dlon, pred_dlon)
        rmse_lat = np.sqrt(mean_squared_error(true_dlat, pred_dlat))
        rmse_lon = np.sqrt(mean_squared_error(true_dlon, pred_dlon))
        
        # Vector Cosine Similarity
        true_vecs = np.column_stack([true_dlat, true_dlon])
        pred_vecs = np.column_stack([pred_dlat, pred_dlon])
        dot_prods = np.sum(true_vecs * pred_vecs, axis=1)
        norms = np.linalg.norm(true_vecs, axis=1) * np.linalg.norm(pred_vecs, axis=1)
        valid_mask = norms > 1e-6
        cos_sim = np.zeros_like(dot_prods)
        cos_sim[valid_mask] = dot_prods[valid_mask] / norms[valid_mask]
        avg_cos_sim = np.mean(cos_sim)
        
        # Great-Circle Haversine Distance Error (km)
        current_lat = coords_arr[:, 0]
        current_lon = coords_arr[:, 1]
        
        actual_dest_lat = current_lat + true_dlat
        actual_dest_lon = current_lon + true_dlon
        pred_dest_lat = current_lat + pred_dlat
        pred_dest_lon = current_lon + pred_dlon
        
        dist_errors_km = haversine_distance_km(actual_dest_lat, actual_dest_lon, pred_dest_lat, pred_dest_lon)
        mean_km = np.mean(dist_errors_km)
        median_km = np.median(dist_errors_km)
        p90_km = np.percentile(dist_errors_km, 90)
        max_km = np.max(dist_errors_km)
        
        summary_rows.append({
            'Horizon': h_name,
            'Lat MAE (°)': f"{mae_lat:.3f}°",
            'Lon MAE (°)': f"{mae_lon:.3f}°",
            'Lat RMSE (°)': f"{rmse_lat:.3f}°",
            'Lon RMSE (°)': f"{rmse_lon:.3f}°",
            'Cosine Sim': f"{avg_cos_sim:.4f}",
            'Mean Dist Err (km)': f"{mean_km:.2f} km",
            'Median Dist (km)': f"{median_km:.2f} km",
            '90th % Dist (km)': f"{p90_km:.2f} km",
            'Max Dist (km)': f"{max_km:.2f} km"
        })
        
    df_metrics = pd.DataFrame(summary_rows)
    report_lines.append(df_metrics.to_string(index=False))
    report_lines.append("\n" + "=" * 90)
    report_lines.append("                 SAMPLE PREDICTION INSPECTION (FIRST 3 TEST TRACKS)")
    report_lines.append("=" * 90)
    
    for sample_i in range(min(3, len(preds_arr))):
        c_lat, c_lon = coords_arr[sample_i]
        report_lines.append(f"\n--- Storm Sample #{sample_i+1} [Current Position: Lat {c_lat:.2f}°, Lon {c_lon:.2f}°] ---")
        sample_rows = []
        for h_idx, h_name in enumerate(horizons):
            t_dlat = targets_arr[sample_i, h_idx, 0]
            t_dlon = targets_arr[sample_i, h_idx, 1]
            p_dlat = preds_arr[sample_i, h_idx, 0]
            p_dlon = preds_arr[sample_i, h_idx, 1]
            
            act_lat = c_lat + t_dlat
            act_lon = c_lon + t_dlon
            prd_lat = c_lat + p_dlat
            prd_lon = c_lon + p_dlon
            
            err_km = haversine_distance_km(np.array([act_lat]), np.array([act_lon]), np.array([prd_lat]), np.array([prd_lon]))[0]
            
            sample_rows.append({
                'Horizon': h_name,
                'Actual Destination': f"({act_lat:.2f}°, {act_lon:.2f}°)",
                'Predicted Destination': f"({prd_lat:.2f}°, {prd_lon:.2f}°)",
                'Delta Actual': f"({t_dlat:+.2f}°, {t_dlon:+.2f}°)",
                'Delta Pred': f"({p_dlat:+.2f}°, {p_dlon:+.2f}°)",
                'Distance Error': f"{err_km:.2f} km"
            })
        report_lines.append(pd.DataFrame(sample_rows).to_string(index=False))
        
    report_lines.append("\n" + "=" * 90)
    full_report = "\n".join(report_lines)
    print("\n" + full_report)
    
    if output_report_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_report_path)), exist_ok=True)
        with open(output_report_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
        print(f"\n[SUCCESS] Test evaluation report written to: '{output_report_path}'")
        
    return full_report
