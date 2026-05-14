"""
Compute all engineered features from raw data, apply normalization,
and compute AQI targets from raw PM2.5 using the US EPA formula.

Normalization:
  - PM2.5, PM10 → log(x+1) → RobustScaler
  - Other pollutants + weather → RobustScaler
  - Cyclic features (Fourier) → no scaling
  - Binary flags → no scaling
  - Targets → NOT scaled (kept in raw AQI units)

Scaler is fit on training data only and saved as scaler_bundle.pkl.
Hourly pipeline loads the pre-fit scaler; only daily training re-fits it.
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import RobustScaler

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import FEATURE_GROUPS, FORECAST_HOURS

SCALER_PATH = Path(__file__).parent.parent.parent / "scaler_bundle.pkl"

# ─── US EPA AQI Formula ───────────────────────────────────────────────────────

_EPA_BREAKPOINTS = [
    (0.0,   12.0,   0,   50),
    (12.1,  35.4,  51,  100),
    (35.5,  55.4, 101,  150),
    (55.5, 150.4, 151,  200),
    (150.5,250.4, 201,  300),
    (250.5,500.4, 301,  500),
]

def pm25_to_aqi(pm25: float) -> int:
    """Convert PM2.5 concentration (µg/m³) to AQI using US EPA formula."""
    pm25 = max(0.0, float(pm25))
    for c_lo, c_hi, i_lo, i_hi in _EPA_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            return round((i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo)
    return 500


# ─── Feature Computation ──────────────────────────────────────────────────────

def _fourier(val: float, period: float):
    angle = 2 * np.pi * val / period
    return np.sin(angle), np.cos(angle)


def compute_time_features(ts: datetime) -> dict:
    hour_sin, hour_cos   = _fourier(ts.hour, 24)
    month_sin, month_cos = _fourier(ts.month, 12)
    return {
        "hour":         ts.hour,
        "day_of_week":  ts.weekday(),
        "month":        ts.month,
        "season":       (ts.month % 12) // 3,   # 0=winter 1=spring 2=summer 3=monsoon
        "is_weekend":   int(ts.weekday() >= 5),
        "hour_sin":     hour_sin,
        "hour_cos":     hour_cos,
        "month_sin":    month_sin,
        "month_cos":    month_cos,
    }


def compute_physics_features(raw: dict) -> dict:
    wind_rad = np.deg2rad(raw.get("wind_deg", 0))
    return {
        "wind_dir_sin":           np.sin(wind_rad),
        "wind_dir_cos":           np.cos(wind_rad),
        "mixing_height_proxy":    raw.get("temperature", 0) / max(raw.get("pressure", 1013), 1) * 1000,
        "wind_transport_effect":  raw.get("wind_speed", 0) * raw.get("pm25", 0),
        "temp_humidity_interaction": raw.get("temperature", 0) * raw.get("humidity", 0) / 100,
        "pm_ratio":               raw.get("pm10", 0) / max(raw.get("pm25", 0.01), 0.01),
        "pressure_anomaly":       0.0,  # filled in after rolling computation
        "aqi_change_rate":        0.0,  # filled in after lag computation
    }


def compute_lag_features(history: pd.DataFrame) -> dict:
    """Compute lag and rolling features from a history DataFrame (sorted asc by time)."""
    aqi_series = history["aqi"].values if "aqi" in history.columns else np.zeros(25)
    pm25_series = history["pm25"].values if "pm25" in history.columns else np.zeros(25)
    pressure_series = history["pressure"].values if "pressure" in history.columns else np.zeros(25)

    def safe_get(arr, idx):
        return float(arr[idx]) if abs(idx) <= len(arr) else 0.0

    lags = {}
    for h in [1, 3, 6, 12, 24]:
        lags[f"aqi_lag_{h}h"]  = safe_get(aqi_series, -h - 1)
        lags[f"pm25_lag_{h}h"] = safe_get(pm25_series, -h - 1)

    # Extended lags: 48h, 72h, 168h (7-day) — weekly pollution patterns
    for h in [48, 72, 168]:
        lags[f"aqi_lag_{h}h"] = safe_get(aqi_series, -h - 1)

    n = len(aqi_series)
    lags["rolling_mean_3h"]  = float(np.mean(aqi_series[-min(3, n):])) if n > 0 else 0.0
    lags["rolling_mean_6h"]  = float(np.mean(aqi_series[-min(6, n):])) if n > 0 else 0.0
    lags["rolling_mean_24h"] = float(np.mean(aqi_series[-min(24, n):])) if n > 0 else 0.0
    lags["rolling_std_24h"]  = float(np.std(aqi_series[-min(24, n):])) if n > 0 else 0.0
    # Range features: explicit volatility bounds the model can learn from
    window_24 = aqi_series[-min(24, n):]
    lags["rolling_min_24h"]  = float(np.min(window_24)) if n > 0 else 0.0
    lags["rolling_max_24h"]  = float(np.max(window_24)) if n > 0 else 0.0
    # Percentage change: day-over-day trend direction
    prev_24h = safe_get(aqi_series, -25)
    curr     = safe_get(aqi_series, -1)
    lags["aqi_pct_change_24h"] = (curr - prev_24h) / max(abs(prev_24h), 1.0)

    lags["aqi_change_rate"]   = safe_get(aqi_series, -1) - safe_get(aqi_series, -2)
    lags["pressure_anomaly"]  = (
        safe_get(pressure_series, -1) - float(np.mean(pressure_series[-min(24, n):]))
        if n > 0 else 0.0
    )
    return lags


def compute_stl_features(history: pd.DataFrame) -> dict:
    """Extract trend, seasonal, residual from AQI history via STL decomposition."""
    aqi = history["aqi"].values if "aqi" in history.columns else np.zeros(48)
    if len(aqi) < 48:
        return {"aqi_trend": 0.0, "aqi_seasonal": 0.0, "aqi_residual": 0.0}
    try:
        stl = STL(pd.Series(aqi), period=24, robust=True)
        result = stl.fit()
        return {
            "aqi_trend":    float(result.trend[-1]),
            "aqi_seasonal": float(result.seasonal[-1]),
            "aqi_residual": float(result.resid[-1]),
        }
    except Exception:
        return {"aqi_trend": 0.0, "aqi_seasonal": 0.0, "aqi_residual": 0.0}


def one_hot_day(day_of_week: int) -> dict:
    """One-hot encode day_of_week (0=Mon … 6=Sun)."""
    return {f"dow_{i}": int(day_of_week == i) for i in range(7)}


def build_feature_row(raw: dict, history: pd.DataFrame, ts: datetime) -> dict:
    """
    Combine all feature groups into a single flat dict.
    Respects FEATURE_GROUPS toggles from config.
    """
    row = {"timestamp": ts.isoformat(), "city": raw.get("city", "karachi")}

    # Use the pre-computed AQI from AQICN API when available (no double conversion).
    # Fall back to pm25_to_aqi() only for historical backfill rows that were built
    # before fetch_current() started providing aqi_from_api.
    if "aqi_from_api" in raw and raw["aqi_from_api"] > 0:
        aqi_current = int(raw["aqi_from_api"])
    else:
        aqi_current = pm25_to_aqi(raw.get("pm25", 0))
    row["aqi"] = aqi_current  # always include current AQI


    if FEATURE_GROUPS["raw_pollutants"]:
        for col in ["pm25", "pm10", "o3", "no2", "so2", "co"]:
            row[col] = float(raw.get(col, 0))

    if FEATURE_GROUPS["meteorology"]:
        for col in ["temperature", "humidity", "pressure", "wind_speed",
                    "visibility", "cloud_cover", "precipitation_1h"]:
            row[col] = float(raw.get(col, 0))

    if FEATURE_GROUPS["time_basic"] or FEATURE_GROUPS["time_fourier"]:
        tf = compute_time_features(ts)
        if FEATURE_GROUPS["time_basic"]:
            for k in ["hour", "month", "season", "is_weekend"]:
                row[k] = tf[k]
            row.update(one_hot_day(tf["day_of_week"]))
        if FEATURE_GROUPS["time_fourier"]:
            for k in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
                row[k] = tf[k]

    if FEATURE_GROUPS["physics_derived"] or FEATURE_GROUPS["cross_feature"]:
        pf = compute_physics_features(raw)
        if FEATURE_GROUPS["physics_derived"]:
            for k in ["wind_dir_sin", "wind_dir_cos", "mixing_height_proxy",
                      "wind_transport_effect", "temp_humidity_interaction", "pm_ratio"]:
                row[k] = pf[k]
        if FEATURE_GROUPS["cross_feature"]:
            for k in ["pressure_anomaly", "aqi_change_rate"]:
                row[k] = pf[k]

    if FEATURE_GROUPS["lag_features"] or FEATURE_GROUPS["rolling_stats"] or FEATURE_GROUPS["cross_feature"]:
        lf = compute_lag_features(history)
        if FEATURE_GROUPS["lag_features"]:
            for h in [1, 3, 6, 12, 24]:
                row[f"aqi_lag_{h}h"]  = lf[f"aqi_lag_{h}h"]
                row[f"pm25_lag_{h}h"] = lf[f"pm25_lag_{h}h"]
            for h in [48, 72, 168]:
                row[f"aqi_lag_{h}h"] = lf[f"aqi_lag_{h}h"]
        if FEATURE_GROUPS["rolling_stats"]:
            for k in ["rolling_mean_3h", "rolling_mean_6h", "rolling_mean_24h", "rolling_std_24h",
                      "rolling_min_24h", "rolling_max_24h", "aqi_pct_change_24h"]:
                row[k] = lf[k]
        if FEATURE_GROUPS["cross_feature"]:
            row["aqi_change_rate"]  = lf["aqi_change_rate"]
            row["pressure_anomaly"] = lf["pressure_anomaly"]

    if FEATURE_GROUPS["stl_decomp"]:
        sf = compute_stl_features(history)
        row.update(sf)

    return row


def add_targets(row: dict, future_aqi: dict) -> dict:
    """Attach target AQI values for 24h/48h/72h horizons."""
    for h in FORECAST_HOURS:
        row[f"aqi_{h}h"] = future_aqi.get(h, None)
    return row


# ─── Normalization ────────────────────────────────────────────────────────────

LOG_COLS    = ["pm25", "pm10"] + [f"pm25_lag_{h}h" for h in [1, 3, 6, 12, 24]]
TARGET_COLS = [f"aqi_{h}h" for h in FORECAST_HOURS]
SKIP_COLS   = TARGET_COLS + ["timestamp", "city"]


def _log_transform(df: pd.DataFrame) -> pd.DataFrame:
    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def fit_scaler(df: pd.DataFrame) -> RobustScaler:
    df = _log_transform(df.copy())
    num_cols = [c for c in df.select_dtypes(include=np.number).columns
                if c not in SKIP_COLS]
    # Guard against inf/NaN from ratios or pct_change on zero values
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    scaler = RobustScaler()
    scaler.fit(df[num_cols])

    scaler.feature_names_ = num_cols
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    return scaler


def load_scaler() -> Optional[RobustScaler]:
    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            return pickle.load(f)
    return None


def apply_scaler(df: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    """Apply a pre-fit scaler to df, tolerating feature set mismatches.

    When the scaler was fit on a different (older) feature set, some columns
    it expects may be absent from df (e.g. after adding new lag features).
    Strategy: build a full-width zero array matching the scaler's expected
    columns, fill in the columns that DO exist, transform the whole array,
    then write back only the columns present in df.  Missing-column slots
    remain zero throughout and are discarded — they never affect df values.
    """
    df = _log_transform(df.copy())
    scaler_cols = scaler.feature_names_           # columns scaler was fit on
    present     = [c for c in scaler_cols if c in df.columns]

    if not present:
        return df  # nothing to scale

    # Build a (1 × len(scaler_cols)) zero array; fill present columns
    arr = np.zeros((len(df), len(scaler_cols)))
    for i, col in enumerate(scaler_cols):
        if col in df.columns:
            arr[:, i] = df[col].values

    scaled = scaler.transform(arr)

    # Write back only the columns that actually exist in df
    for i, col in enumerate(scaler_cols):
        if col in df.columns:
            df[col] = scaled[:, i]

    return df
