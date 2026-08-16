import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

# Configuration
TRAIN_DATA_PATH = r"c:\Users\ADMIN\Documents\Storm path predictor\Datasets\Storm_data\train.csv"
TEST_DATA_PATH = r"c:\Users\ADMIN\Documents\Storm path predictor\Datasets\Storm_data\test.csv"
MODEL_DIR = r"c:\Users\ADMIN\Documents\Storm path predictor\src\train\Xgboost\models"
RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_result.txt")
os.makedirs(MODEL_DIR, exist_ok=True)

def feature_engineering(df):
    df = df.copy()
    # 2. Cyclical features for time
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    
    # sin/cos encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # We will use periods 1, 2, 3, 4 for 6h, 12h, 18h, 24h assuming data is 6-hourly
    horizons = {'6h': -1, '12h': -2, '18h': -3, '24h': -4}
    for h_name, shift_val in horizons.items():
        df[f'future_lat_{h_name}'] = df.groupby('international_id')['lat'].shift(shift_val)
        df[f'future_lon_{h_name}'] = df.groupby('international_id')['lon'].shift(shift_val)
        df[f'delta_lat_{h_name}'] = df[f'future_lat_{h_name}'] - df['lat']
        df[f'delta_lon_{h_name}'] = df[f'future_lon_{h_name}'] - df['lon']
        df.drop(columns=[f'future_lat_{h_name}', f'future_lon_{h_name}'], inplace=True)
        
    # Lag features for position and wind properties (Past values)
    lag_props = ['lat', 'lon', 'max_wind_kt', 'dir_50kt', 'rad_50kt_long_nm', 'rad_50kt_short_nm', 
                 'dir_30kt', 'rad_30kt_long_nm', 'rad_30kt_short_nm']
    
    lags = {'6h': 1, '12h': 2, '18h': 3, '24h': 4}
    for col in lag_props:
        if col in df.columns:
            for l_name, shift_val in lags.items():
                df[f'{col}_lag_{l_name}'] = df.groupby('international_id')[col].shift(shift_val)
                # Past displacement (velocity vector over past horizons)
                if col in ['lat', 'lon']:
                    df[f'past_delta_{col}_{l_name}'] = df[col] - df[f'{col}_lag_{l_name}']
                
    target_cols = []
    for h in ['6h', '12h', '18h', '24h']:
        target_cols.extend([f'delta_lat_{h}', f'delta_lon_{h}'])
    
    df.dropna(subset=target_cols, inplace=True)
    
    cols_to_drop = ['tc_number', 'name', 'grade_name', 'revision_date', 'year', 'time_diff_hours', 'time', 'month', 'day', 'hour','flag_last']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    object_cols = df.select_dtypes(include=['object']).columns
    object_cols = [c for c in object_cols if c != 'international_id']
    df.drop(columns=object_cols, inplace=True)
    
    df = df.apply(pd.to_numeric, errors='coerce')
    return df, target_cols

def main():
    print("Loading train and test datasets from Storm_data...")
    df_train = pd.read_csv(TRAIN_DATA_PATH, low_memory=False)
    df_test = pd.read_csv(TEST_DATA_PATH, low_memory=False)
    
    # 1. Convert time and ensure chronological order per storm
    df_train['time'] = pd.to_datetime(df_train['time'])
    df_train = df_train.sort_values(by=['international_id', 'time']).copy()
    
    df_test['time'] = pd.to_datetime(df_test['time'])
    df_test = df_test.sort_values(by=['international_id', 'time']).copy()
    
    print("Feature Engineering...")
    df_train, target_cols = feature_engineering(df_train)
    df_test, _ = feature_engineering(df_test)
    
    print(f"Train Shape: {df_train.shape}, Test Shape: {df_test.shape}")
    
    X_train = df_train.drop(columns=target_cols + ['international_id'])
    y_train = df_train[target_cols]
    
    X_test = df_test.drop(columns=target_cols + ['international_id'])
    y_test = df_test[target_cols]
    
    print("Training models...")
    models = {}
    
    for h in ['6h', '12h', '18h', '24h']:
        for coord in ['lat', 'lon']:
            target_name = f'delta_{coord}_{h}'
            print(f"Training for {target_name}...")
            
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            model.fit(X_train, y_train[target_name])
            models[target_name] = model
            
            model_path = os.path.join(MODEL_DIR, f'xgboost_{target_name}.joblib')
            joblib.dump(model, model_path)
    
    print("Evaluating models (Cosine Similarity)...")
    with open(RESULT_PATH, 'w') as f:
        f.write("--- Evaluation Results ---\n")
        for h in ['6h', '12h', '18h', '24h']:
            lat_target = f'delta_lat_{h}'
            lon_target = f'delta_lon_{h}'
            
            y_test_lat = y_test[lat_target].values
            y_test_lon = y_test[lon_target].values
            
            pred_lat = models[lat_target].predict(X_test)
            pred_lon = models[lon_target].predict(X_test)
            
            y_true_vec = np.column_stack((y_test_lat, y_test_lon))
            y_pred_vec = np.column_stack((pred_lat, pred_lon))
            
            dot_product = np.sum(y_true_vec * y_pred_vec, axis=1)
            norm_true = np.linalg.norm(y_true_vec, axis=1)
            norm_pred = np.linalg.norm(y_pred_vec, axis=1)
            
            denom = norm_true * norm_pred
            
            similarity = np.zeros_like(dot_product)
            valid_idx = denom > 0
            similarity[valid_idx] = dot_product[valid_idx] / denom[valid_idx]
            
            avg_cosine_sim = np.mean(similarity)
            
            # Calculate Magnitude/Distance Error using Haversine formula in km
            current_lat = X_test['lat'].values
            current_lon = X_test['lon'].values
            
            true_final_lat = current_lat + y_test_lat
            true_final_lon = current_lon + y_test_lon
            
            pred_final_lat = current_lat + pred_lat
            pred_final_lon = current_lon + pred_lon
            
            distance_error_km = haversine_np(true_final_lon, true_final_lat, pred_final_lon, pred_final_lat)
            avg_distance_error_km = np.mean(distance_error_km)
            min_distance_error_km = np.min(distance_error_km)
            max_distance_error_km = np.max(distance_error_km)
            
            res_str = (
                f"Average Cosine Similarity for {h}: {avg_cosine_sim:.4f} | "
                f"Distance Error: Mean = {avg_distance_error_km:.2f} km, "
                f"Min = {min_distance_error_km:.2f} km, "
                f"Max = {max_distance_error_km:.2f} km"
            )
            print(res_str)
            f.write(res_str + "\n")

    print("\nCalculating Feature Importances...")
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names})
    
    for target_name, model in models.items():
        importance_df[target_name] = model.feature_importances_
        
    importance_df['Mean_Importance'] = importance_df.drop(columns=['Feature']).mean(axis=1)
    
    with open(RESULT_PATH, 'a') as f:
        f.write("\n--- Feature Importances ---\n")
        # Print Average
        avg_df = importance_df.sort_values(by='Mean_Importance', ascending=False)
        f.write("Top 15 Most Important Features (Averaged across all models):\n")
        f.write(avg_df[['Feature', 'Mean_Importance']].head(15).to_string(index=False))
        f.write("\n\n")
        
        # Print for each individual model
        for target_name in models.keys():
            f.write(f"--- Top 15 Features for {target_name} ---\n")
            model_df = importance_df[['Feature', target_name]].sort_values(by=target_name, ascending=False)
            f.write(model_df.head(15).to_string(index=False))
            f.write("\n\n")

    print("Results and feature importances saved to 'train_result.txt'.")
    print("\nAll done!")

if __name__ == '__main__':
    main()
