import numpy as np
import rasterio
from typing import Tuple

def load_and_preprocess_geotiff(filepath: str) -> np.ndarray:
    """
    Loads a multi-band GeoTIFF, fills missing values, and returns a preprocessed NumPy array.
    """
    with rasterio.open(filepath) as src:
        arr = src.read()

    # Step 1: Fill missing values for the first day using its mean
    first_day = arr[0, :, :]
    mean_value = np.nanmean(first_day)
    missing_pixels = np.isnan(first_day)
    first_day[missing_pixels] = mean_value

    # Step 2: Fill missing values for subsequent days using the previous day's data
    for day in range(1, arr.shape[0]):
        current_day_data = arr[day, :, :]
        previous_day_data = arr[day - 1, :, :]
        missing_pixels_mask = np.isnan(current_day_data)
        current_day_data[missing_pixels_mask] = previous_day_data[missing_pixels_mask]
    
    return arr

def z_score_scale(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Performs Z-score (Standard) scaling on the data.
    Returns:
        scaled_data: (data - mean) / std
        mean: mean of original data
        std: standard deviation of original data
    """
    mean_val = np.mean(data)
    std_val = np.std(data)

    # avoid divide-by-zero if std=0
    if std_val == 0:
        scaled_data = data - mean_val
    else:
        scaled_data = (data - mean_val) / std_val

    return scaled_data, mean_val, std_val

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and scale GeoTIFF satellite data.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input GeoTIFF file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the processed .npy file.")

    args = parser.parse_args()

    print("Loading and preprocessing data...")
    processed_data = load_and_preprocess_geotiff(args.input)
    print("Missing values filled successfully.")

    print("Scaling the data...")
    scaled_data, mean_val, std_val = z_score_scale(processed_data)
    print("Data scaled successfully.")

    # Save both raw processed and scaled if you want
    np.save(args.output, scaled_data)
    print(f"Final data shape: {scaled_data.shape}")
    print(f"Scaling factors: Mean = {mean_val}, Std = {std_val}")