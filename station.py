#!/usr/bin/env python3
"""
Preprocess station + satellite ERA5 data:
 - Load CSVs
 - Clean + feature engineering
 - KNN imputation
 - Standardization + PCA
 - Sequence generation
 - Save features + labels as .npy
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

keep_columns = [
    "Timestamp", "PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO (µg/m³)",
    "NO2 (µg/m³)", "NOx (ppb)", "NH3 (µg/m³)", "SO2 (µg/m³)", 
    "CO (mg/m³)", "Ozone (µg/m³)", "Benzene (µg/m³)", "Toluene (µg/m³)", 
    "Xylene (µg/m³)", "AT (°C)", "RH (%)", "WS (m/s)", "WD (deg)", 
    "TOT-RF (mm)", "SR (W/mt2)", "BP (mmHg)", "VWS (m/s)"
]

def load_and_merge(station_path: str, era5_path: str) -> pd.DataFrame:
    station = pd.read_csv(station_path)
    era5 = pd.read_csv(era5_path)

    station = station[keep_columns]
    # Merge
    df = station.join(pd.DataFrame(era5))

    # Drop unneeded
    drop_cols = ["system:index", ".geo", "time", "TOT-RF (mm)"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Precipitation must be non-negative
    if "total_precipitation" in df.columns:
        df["total_precipitation"] = np.maximum(0, df["total_precipitation"])

    # Wind components
    if "WS (m/s)" in df.columns and "WD (deg)" in df.columns:
        df["u_comp_wind"] = df["WS (m/s)"] * np.cos(np.radians(df["WD (deg)"]))
        df["v_comp_wind"] = df["WS (m/s)"] * np.sin(np.radians(df["WD (deg)"]))

    # Temporal features
    df_time = pd.to_datetime(df["Timestamp"])
    df["hour"] = df_time.dt.hour
    df["day-of-week"] = df_time.dt.dayofweek
    df["day-of-year"] = df_time.dt.dayofyear
    df["month"] = df_time.dt.month
    df["quarter"] = df_time.dt.quarter
    df["is_weekend"] = df["day-of-week"].isin([5, 6]).astype(int)

    # Dewpoint (°C)
    if "AT (°C)" in df.columns and "RH (%)" in df.columns:
        df["dewpoint"] = df["AT (°C)"] - ((100 - df["RH (%)"]) / 5)

    # Drop redundant cols
    df = df.drop(columns=["Timestamp", "RH (%)", "WS (m/s)", "WD (deg)"], errors="ignore")

    # Drop heavy-missing-value columns (manual decision)
    if "Xylene (µg/m³)" in df.columns:
        df = df.drop(columns=["Xylene (µg/m³)"])

    return df


def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    imputer = KNNImputer(
        n_neighbors=4, weights="distance", metric="nan_euclidean", missing_values=np.nan
    )
    imputed = imputer.fit_transform(df)
    return pd.DataFrame(imputed, columns=df.columns)


def scaler_z(X: np.ndarray):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def apply_pca(X: np.ndarray, components: int=15):

    pca = PCA(n_components=components, random_state=RANDOM_SEED)
    X_reduced = pca.fit_transform(X)

    return X_reduced, pca

def main(args):
    print("Loading and merging data...")
    df = load_and_merge(args.station_csv, args.era5_csv)

    print("Imputing missing values...")
    df = impute_data(df)


    labels = df[["Ozone (µg/m³)", "NO2 (µg/m³)"]].values
    feature_cols = [c for c in df.columns if c not in ["Ozone (µg/m³)", "NO2 (µg/m³)"]]
    X = df[feature_cols].values
    X,input_scaler= scaler_z(X)

    print("Applying PCA...")
    X_reduced, pca = apply_pca(X, components=args.components)

    print(f"X shape: {X_reduced.shape}, y shape: {labels.shape}")

    np.save(args.output_features, X_reduced)
    np.save(args.output_labels, labels)
    print(f"Saved features to {args.output_features}")
    print(f"Saved labels to {args.output_labels}")
    print("done...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess station + ERA5 data.")
    parser.add_argument("--station_csv", type=str, required=True, help="Path to station CSV")
    parser.add_argument("--era5_csv", type=str, required=True, help="Path to ERA5 CSV")
    parser.add_argument("--output_features", type=str, default="./station_data_features.npy")
    parser.add_argument("--output_labels", type=str, default="./data_labels.npy")
    parser.add_argument("--seq_len", type=int, default=12, help="Sequence length (hours)")
    parser.add_argument("--components", type=int, default=15, help="components to retain in PCA")
    args = parser.parse_args()
    main(args)
