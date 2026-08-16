#!/usr/bin/env python3
"""
prepare_data.py
Prepares storm track data for storms occurring from 1980 to present
by splitting them into train, test, and validation sets.

Steps:
1. Load Datasets/data.csv into a pandas DataFrame.
2. Filter for storms that started occurring from 1980 to present.
3. Group records by storm ID (international_id) into a storm array.
4. Filter storms using the 6-hour progressive count algorithm (pinpoint start time t0, match 0h, 6h, 12h..., skip storm if step gap exceeds 6h).
5. Randomly partition the storm array into train (80%), test (15%), and valid (5%).
6. Convert partitioned storm arrays back to DataFrames.
7. Export to train.csv, test.csv, and valid.csv inside the Datasets/Storm_data folder.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Default Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "Datasets" / "data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Datasets" / "Storm_data"


def filter_storms_by_regular_6h_intervals(
    storms: Union[List[pd.DataFrame], np.ndarray],
    step_hours: float = 6.0,
) -> Union[List[pd.DataFrame], np.ndarray]:
    """
    Filters storms based on the 6-hour progressive count algorithm:
    1. For each storm, pinpoint the start time (t0).
    2. Count starts at 0 to get the first state.
    3. Count advances to 6. If a state's time difference with t0 is 6, advance to 12...
    4. If not equal (e.g. intermediate 3h state), continue to the next state.
    5. If a state's time difference exceeds the expected count (gap > 6h), skip and discard that storm.

    Parameters
    ----------
    storms : Union[List[pd.DataFrame], np.ndarray]
        Array or list of DataFrames, where each element represents one storm.
    step_hours : float
        Expected interval step in hours (default: 6.0).

    Returns
    -------
    Union[List[pd.DataFrame], np.ndarray]
        Filtered array or list of storm DataFrames with regular 6-hour state sequences.
    """
    filtered_storms = []

    for storm in storms:
        if len(storm) == 0:
            continue

        storm_sorted = storm.sort_values("time").reset_index(drop=True)
        t0 = storm_sorted["time"].iloc[0]

        expected_hours = 0.0
        matched_indices = []
        is_valid = True

        for i in range(len(storm_sorted)):
            elapsed_hours = (storm_sorted["time"].iloc[i] - t0).total_seconds() / 3600.0

            if elapsed_hours == expected_hours:
                matched_indices.append(i)
                expected_hours += step_hours
            elif elapsed_hours < expected_hours:
                # Intermediate state with elapsed time < expected count, continue to next state
                continue
            else:
                # Elapsed time exceeds expected count (gap > 6h), skip this storm
                is_valid = False
                break

        if is_valid and len(matched_indices) > 0:
            filtered_storms.append(storm_sorted.iloc[matched_indices].copy())

    if isinstance(storms, np.ndarray):
        return np.array(filtered_storms, dtype=object)
    return filtered_storms


def split_storm_data(
    input_path: Path = DEFAULT_INPUT_FILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    min_year: int = 1980,
    train_ratio: float = 0.80,
    test_ratio: float = 0.15,
    valid_ratio: float = 0.05,
    random_seed: Optional[int] = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads data.csv, filters for storms starting from min_year to present,
    splits unique storms into train, test, and valid sets,
    and saves the resulting DataFrames as CSV files in output_dir.

    Parameters
    ----------
    input_path : Path
        Path to the input data.csv file.
    output_dir : Path
        Directory where train.csv, test.csv, and valid.csv will be saved.
    min_year : int
        Minimum start year of storms to include (default: 1980).
    train_ratio : float
        Proportion of storms allocated to training set (default: 0.80).
    test_ratio : float
        Proportion of storms allocated to testing set (default: 0.15).
    valid_ratio : float
        Proportion of storms allocated to validation set (default: 0.05).
    random_seed : Optional[int]
        Random seed for reproducibility (default: 42). Set to None for non-deterministic split.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tuple of (train_df, test_df, valid_df).
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    # 1. Import and turn data file into DataFrame
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded DataFrame with {len(df):,} total rows and {len(df.columns)} columns.")

    if "international_id" not in df.columns:
        raise KeyError("Column 'international_id' not found in dataset.")
    if "time" not in df.columns:
        raise KeyError("Column 'time' not found in dataset.")

    # Parse datetime to find storm start years
    df["time"] = pd.to_datetime(df["time"])

    # 2. Filter for storms that start occurring from min_year (1980) to present
    storm_start_years = df.groupby("international_id")["time"].min().dt.year
    filtered_storm_ids = storm_start_years[storm_start_years >= min_year].index

    df_filtered = df[df["international_id"].isin(filtered_storm_ids)].copy()
    total_storms_raw = len(filtered_storm_ids)

    print(f"Filtered for storms starting from {min_year} to present:")
    print(f" - Retained {total_storms_raw:,} unique storms (out of {df['international_id'].nunique():,} total storms).")
    print(f" - Retained {len(df_filtered):,} records (out of {len(df):,} total records).")

    # 3. Create a storm array holding all different storms based on storm ID
    # Grouping by international_id to create an array of storm sub-dataframes
    storm_groups = [group for _, group in df_filtered.groupby("international_id", sort=False)]
    storm_array = np.array(storm_groups, dtype=object)

    # 4. Filter storms with 6-hour progressive count algorithm
    storm_array = filter_storms_by_regular_6h_intervals(storm_array, step_hours=6.0)
    total_storms = len(storm_array)
    records_retained = sum(len(s) for s in storm_array)

    print(f"Filtered storms using 6-hour progressive count algorithm:")
    print(f" - Retained {total_storms:,} unique storms (removed {total_storms_raw - total_storms:,} storms).")
    print(f" - Retained {records_retained:,} records (removed {len(df_filtered) - records_retained:,} non-6h/gap records).")

    # 5. Randomly create 3 separate storm arrays: train, test, valid
    # (80% train, 15% test, 5% valid)
    rng = np.random.default_rng(seed=random_seed)
    shuffled_indices = rng.permutation(total_storms)
    shuffled_storms = storm_array[shuffled_indices]

    n_train = int(round(total_storms * train_ratio))
    n_test = int(round(total_storms * test_ratio))
    # Assign remaining storms to valid to guarantee sum equals total_storms
    n_valid = total_storms - n_train - n_test

    train_storms = shuffled_storms[:n_train]
    test_storms = shuffled_storms[n_train : n_train + n_test]
    valid_storms = shuffled_storms[n_train + n_test :]

    # 5. Turn storm arrays back into DataFrames
    train_df = pd.concat(train_storms.tolist(), ignore_index=True)
    test_df = pd.concat(test_storms.tolist(), ignore_index=True)
    valid_df = pd.concat(valid_storms.tolist(), ignore_index=True)

    # Sanity checks
    train_id_set = set(train_df["international_id"].unique())
    test_id_set = set(test_df["international_id"].unique())
    valid_id_set = set(valid_df["international_id"].unique())

    assert len(train_id_set.intersection(test_id_set)) == 0, "Leakage between train and test sets!"
    assert len(train_id_set.intersection(valid_id_set)) == 0, "Leakage between train and valid sets!"
    assert len(test_id_set.intersection(valid_id_set)) == 0, "Leakage between test and valid sets!"

    print("\n--- Split Summary (1980-Present) ---")
    print(f"Train : {len(train_storms):>4} storms ({len(train_storms)/total_storms*100:5.1f}%) | {len(train_df):>6} records")
    print(f"Test  : {len(test_storms):>4} storms ({len(test_storms)/total_storms*100:5.1f}%) | {len(test_df):>6} records")
    print(f"Valid : {len(valid_storms):>4} storms ({len(valid_storms)/total_storms*100:5.1f}%) | {len(valid_df):>6} records")
    print(f"Total : {total_storms:>4} storms (100.0%) | {records_retained:>6} records")

    # 6. Save DataFrames to train.csv, test.csv, valid.csv inside output_dir (Storm_data)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    valid_path = output_dir / "valid.csv"

    print(f"\nSaving output CSV files to: {output_dir}")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    valid_df.to_csv(valid_path, index=False)

    print(f" Saved: {train_path.name} ({train_path.stat().st_size / 1024:.1f} KB)")
    print(f" Saved: {test_path.name} ({test_path.stat().st_size / 1024:.1f} KB)")
    print(f" Saved: {valid_path.name} ({valid_path.stat().st_size / 1024:.1f} KB)")
    print("Data preparation complete!")

    return train_df, test_df, valid_df


def main():
    parser = argparse.ArgumentParser(description="Prepare and split storm data (1980-present) into train, test, and valid sets.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE, help="Path to input data.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Path to output Storm_data directory")
    parser.add_argument("--min-year", type=int, default=1980, help="Minimum storm start year (default: 1980)")
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Ratio for training set (default: 0.80)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Ratio for test set (default: 0.15)")
    parser.add_argument("--valid-ratio", type=float, default=0.05, help="Ratio for validation set (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    split_storm_data(
        input_path=args.input,
        output_dir=args.output_dir,
        min_year=args.min_year,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        valid_ratio=args.valid_ratio,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
