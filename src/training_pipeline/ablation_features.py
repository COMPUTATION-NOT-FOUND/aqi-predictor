"""
Feature group ablation study — self-maintained, runs nightly as part of training pipeline.

For each feature group in FEATURE_GROUPS, temporarily disable it, train a fixed RF,
log the RMSE delta to MLflow. Results auto-populate the dashboard's Ablation tab.
"""
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import MLFLOW_TRACKING_URI, FEATURE_GROUPS, FORECAST_HOURS

EXPERIMENT_NAME = "feature_ablation"
TARGET_COL      = "aqi_24h"
N_SPLITS        = 3


def _cv_rmse(X: np.ndarray, y: np.ndarray) -> float:
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rmses = []
    for tr, val in tscv.split(X):
        rf.fit(X[tr], y[tr])
        rmses.append(np.sqrt(mean_squared_error(y[val], rf.predict(X[val]))))
    return float(np.mean(rmses))


def _feature_cols_for_group(group: str, all_cols: list[str]) -> list[str]:
    """Return column names that belong to a given feature group."""
    mapping = {
        "time_basic":      ["hour", "month", "season", "is_weekend"] + [f"dow_{i}" for i in range(7)],
        "time_fourier":    ["hour_sin", "hour_cos", "month_sin", "month_cos"],
        "lag_features":    [c for c in all_cols if "lag" in c],
        "rolling_stats":   ["rolling_mean_3h", "rolling_mean_6h", "rolling_mean_24h", "rolling_std_24h"],
        "physics_derived": ["wind_dir_sin", "wind_dir_cos", "mixing_height_proxy",
                            "wind_transport_effect", "temp_humidity_interaction", "pm_ratio"],
        "cross_feature":   ["pressure_anomaly", "aqi_change_rate"],
        "stl_decomp":      ["aqi_trend", "aqi_seasonal", "aqi_residual"],
        "raw_pollutants":  ["pm25", "pm10", "o3", "no2", "so2", "co"],
        "meteorology":     ["temperature", "humidity", "pressure", "wind_speed",
                            "visibility", "cloud_cover", "precipitation_1h"],
    }
    return [c for c in mapping.get(group, []) if c in all_cols]


def run_ablation(df: pd.DataFrame):
    """
    Run feature ablation on a prepared feature DataFrame.
    df must contain all feature columns and TARGET_COL.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    drop_meta = ["timestamp", "city"] + [f"aqi_{h}h" for h in FORECAST_HOURS]
    feature_cols = [c for c in df.columns if c not in drop_meta and df[c].dtype != object]
    df_clean = df[feature_cols + [TARGET_COL]].dropna()

    if len(df_clean) < 100:
        print("[ablation_features] Not enough data for ablation — skipping")
        return

    # Sanitize once here; all X slices below come from this clean array
    X_full = df_clean[feature_cols].values.astype(np.float64)
    X_full = np.where(np.isfinite(X_full), X_full, 0.0)
    y      = df_clean[TARGET_COL].values

    # Build a column-index map so X_reduced slices the sanitized X_full
    col_idx = {col: i for i, col in enumerate(feature_cols)}

    # Baseline with all features
    baseline_rmse = _cv_rmse(X_full, y)
    print(f"[ablation_features] Baseline RMSE (all features): {baseline_rmse:.2f}")

    with mlflow.start_run(run_name="baseline_all_features"):
        mlflow.log_param("dropped_group", "none")
        mlflow.log_metric("rmse", baseline_rmse)
        mlflow.log_metric("rmse_delta", 0.0)

    results = {"baseline": baseline_rmse}

    for group, enabled in FEATURE_GROUPS.items():
        if not enabled:
            continue   # already disabled globally — skip
        if group == "raw_pollutants":
            continue   # never drop raw pollutants (they're the primary signal)

        cols_to_drop = _feature_cols_for_group(group, feature_cols)
        if not cols_to_drop:
            continue

        reduced_cols = [c for c in feature_cols if c not in cols_to_drop]
        if len(reduced_cols) == 0:
            continue

        X_reduced = X_full[:, [col_idx[c] for c in reduced_cols]]
        group_rmse = _cv_rmse(X_reduced, y)
        delta = group_rmse - baseline_rmse

        with mlflow.start_run(run_name=f"drop_{group}"):
            mlflow.log_param("dropped_group", group)
            mlflow.log_param("n_dropped_features", len(cols_to_drop))
            mlflow.log_metric("rmse", group_rmse)
            mlflow.log_metric("rmse_delta", delta)

        results[group] = delta
        print(f"[ablation_features] drop '{group}': RMSE={group_rmse:.2f} (Δ={delta:+.2f})")

    return results  # {group: delta} — negative delta means group is hurting performance
