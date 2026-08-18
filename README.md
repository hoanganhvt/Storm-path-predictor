# Storm Trajectory Prediction Framework

## Project Overview
This repository contains a predictive modeling framework designed to forecast the trajectory of tropical storms up to 24 hours in advance. Utilizing historical storm tracking data from 1980 to the present, the project implements and compares machine learning (XGBoost) and deep learning (Long Short-Term Memory networks) methodologies.

## Data Processing Architecture
The dataset is constructed from historical storm paths spanning over four decades. The data pipeline encompasses:
- Temporal Normalization: Standardization of storm observation data into regular 6-hour progressive intervals.
- Strict Partitioning: A rigorous division into training (80%), testing (15%), and validation (5%) sets. The split is performed at the individual storm level to ensure zero data leakage across sets.
- Feature Engineering: Extraction of geospatial dynamics including past latitude and longitude deltas, wind speeds (`max_wind_kt`), and directional wind vectors (`dir_50kt`, `dir_30kt`).

## Model Architectures
The repository houses multiple predictive models targeting the displacement variables (`delta_lat` and `delta_lon`) at 6-hour, 12-hour, 18-hour, and 24-hour lead times.

### Gradient Boosting (XGBoost)
A robust tree-based ensemble method leveraging extracted temporal lag features. Feature importance analysis validates that historical longitudinal and latitudinal changes (e.g., `past_delta_lon_6h`, `past_delta_lat_6h`) are the strongest predictors of future coordinates.

### Sequence Modeling (LSTM)
A recurrent neural network architecture designed to capture complex temporal dependencies within a storm's lifespan by processing the sequential coordinate and atmospheric feature data.

### Representation Learning (Autoencoder)
A custom PyTorch Autoencoder architecture featuring a specialized masked Mean Squared Error (MSE) loss function. It is utilized to construct latent representations and handle missing feature inputs.

## Performance Metrics and Results
Models are evaluated on Cosine Similarity (measuring directional accuracy) and Mean Distance Error (measuring spatial displacement accuracy in kilometers). 

**XGBoost Evaluation:**
- 6-Hour Forecast: Cosine Similarity 0.9095 | Mean Error 41.84 km
- 12-Hour Forecast: Cosine Similarity 0.8969 | Mean Error 86.54 km
- 18-Hour Forecast: Cosine Similarity 0.8830 | Mean Error 138.65 km
- 24-Hour Forecast: Cosine Similarity 0.8703 | Mean Error 194.87 km

**LSTM Evaluation:**
- 6-Hour Forecast: Cosine Similarity 0.8792 | Mean Error 49.29 km
- 12-Hour Forecast: Cosine Similarity 0.8735 | Mean Error 98.38 km
- 18-Hour Forecast: Cosine Similarity 0.8673 | Mean Error 152.89 km
- 24-Hour Forecast: Cosine Similarity 0.8590 | Mean Error 212.13 km

*Conclusion: The XGBoost implementation currently demonstrates superior performance across all temporal forecast intervals, minimizing spatial distance errors and maintaining higher directional accuracy.*

## Repository Structure
- `Datasets/`: Contains raw aggregated historical data alongside the partitioned datasets (`train.csv`, `test.csv`, `valid.csv`).
- `src/data/`: Modules for data acquisition, temporal filtering (`prepare_data.py`), and exploratory data analysis.
- `src/train/`: Model training scripts, checkpoint management, and detailed evaluation records categorized by architecture.
