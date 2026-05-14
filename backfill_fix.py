"""
Vectorized historical backfill — restores labeled training data to Hopsworks.

Targets are computed by shifting the AQI series forward:
  aqi_24h[t] = aqi[t+24]  (what AQI will be 24 hours from now)
  aqi_48h[t] = aqi[t+48]
  aqi_72h[t] = aqi[t+72]

Only rows where all three targets are known (not NaN) are kept.
This restores a clean, fully-labeled training dataset to Hopsworks.
"""
import sys, os
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import datetime

from src.config import (
    DEFAULT_CITY, DEFAULT_LAT, DEFAULT_LON,
    FEATURE_GROUPS, FORECAST_HOURS,
)
from src.feature_pipeline.fetch_data import (
    fetch_historical_openweather,
    fetch_historical_weather_openmeteo,
)
from src.feature_pipeline.feature_engineering import (
    pm25_to_aqi, compute_time_features, one_hot_day, fit_scaler,
)
from src.feature_pipeline.store_features import get_feature_store, get_or_create_feature_group

# ── 1. Fetch historical data ───────────────────────────────────────────────────
DAYS = 160
print(f"[backfill] Fetching {DAYS} days of historical data...")

end_time   = datetime.datetime.now(datetime.timezone.utc)
start_time = end_time - datetime.timedelta(days=DAYS)

print("  OpenWeather air pollution (µg/m³ concentrations)...")
ow_rows = fetch_historical_openweather(
    DEFAULT_LAT, DEFAULT_LON,
    int(start_time.timestamp()),
    int(end_time.timestamp()),
)
df_ow = pd.DataFrame(ow_rows)
df_ow["timestamp"] = pd.to_datetime(df_ow["timestamp"]).dt.tz_localize(None)
df_ow = df_ow.set_index("timestamp").sort_index()
print(f"  Got {len(df_ow)} pollutant rows")

print("  OpenMeteo weather (temperature, wind, pressure...)...")
df_om = fetch_historical_weather_openmeteo(
    DEFAULT_LAT, DEFAULT_LON,
    start_time.strftime("%Y-%m-%d"),
    end_time.strftime("%Y-%m-%d"),
)
df_om = df_om.sort_index()
df_om.index = df_om.index.tz_localize(None)
print(f"  Got {len(df_om)} weather rows")

# ── 2. Merge on hourly timestamp (nearest within 30min) ───────────────────────
print("[backfill] Merging datasets...")
df_ow_r = df_ow.reset_index()
df_om_r = df_om.reset_index()

df = pd.merge_asof(
    df_om_r.sort_values("timestamp"),
    df_ow_r.sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("30min"),
)
df = df.set_index("timestamp").sort_index()
df["city"] = DEFAULT_CITY
print(f"  Merged → {len(df)} rows")

# ── 3. Compute AQI from real PM2.5 µg/m³ ─────────────────────────────────────
print("[backfill] Computing AQI from PM2.5 µg/m³...")
df["aqi"] = df["pm25"].apply(pm25_to_aqi).astype(int)
print(f"  AQI range: {df['aqi'].min()}–{df['aqi'].max()}")

# ── 4. Vectorized feature engineering ────────────────────────────────────────
print("[backfill] Engineering features (vectorized)...")

# 4a. Time features
ts = df.index
df["hour"]        = ts.hour
df["month"]       = ts.month
df["season"]      = (ts.month % 12) // 3
df["is_weekend"]  = (ts.dayofweek >= 5).astype(int)
df["hour_sin"]    = np.sin(2 * np.pi * ts.hour / 24)
df["hour_cos"]    = np.cos(2 * np.pi * ts.hour / 24)
df["month_sin"]   = np.sin(2 * np.pi * ts.month / 12)
df["month_cos"]   = np.cos(2 * np.pi * ts.month / 12)
for d in range(7):
    df[f"dow_{d}"] = (ts.dayofweek == d).astype(int)

# 4b. Physics features
wind_rad = np.deg2rad(df.get("wind_deg", pd.Series(0, index=df.index)).fillna(0))
df["wind_dir_sin"]           = np.sin(wind_rad)
df["wind_dir_cos"]           = np.cos(wind_rad)
df["mixing_height_proxy"]    = df["temperature"] / df["pressure"].clip(lower=1) * 1000
df["wind_transport_effect"]  = df["wind_speed"] * df["pm25"]
df["temp_humidity_interaction"] = df["temperature"] * df["humidity"] / 100
df["pm_ratio"]               = df["pm10"] / df["pm25"].clip(lower=0.01)

# Replace inf from divisions by zero in feature columns only (targets not yet computed)
df = df.replace([np.inf, -np.inf], 0.0)


# 4c. Lag features (vectorized with shift)
for h in [1, 3, 6, 12, 24]:
    df[f"aqi_lag_{h}h"]  = df["aqi"].shift(h)
    df[f"pm25_lag_{h}h"] = df["pm25"].shift(h)
for h in [48, 72, 168]:
    df[f"aqi_lag_{h}h"]  = df["aqi"].shift(h)

# 4d. Rolling features
df["rolling_mean_3h"]    = df["aqi"].rolling(3,  min_periods=1).mean()
df["rolling_mean_6h"]    = df["aqi"].rolling(6,  min_periods=1).mean()
df["rolling_mean_24h"]   = df["aqi"].rolling(24, min_periods=1).mean()
df["rolling_std_24h"]    = df["aqi"].rolling(24, min_periods=1).std().fillna(0)
df["rolling_min_24h"]    = df["aqi"].rolling(24, min_periods=1).min()
df["rolling_max_24h"]    = df["aqi"].rolling(24, min_periods=1).max()
df["aqi_pct_change_24h"] = df["aqi"].pct_change(24).fillna(0)

# 4e. Cross features
df["aqi_change_rate"]  = df["aqi"].diff(1).fillna(0)
df["pressure_anomaly"] = df["pressure"] - df["pressure"].rolling(24, min_periods=1).mean()

# 4f. STL placeholders (set to 0 — can't do STL vectorized easily)
df["aqi_trend"]    = 0.0
df["aqi_seasonal"] = 0.0
df["aqi_residual"] = 0.0

# ── 5. Compute targets (future AQI values) ────────────────────────────────────
print("[backfill] Computing targets (aqi_24h, aqi_48h, aqi_72h)...")
for h in FORECAST_HOURS:
    df[f"aqi_{h}h"] = df["aqi"].shift(-h)   # shift BACK → future value

# Only keep rows where ALL targets are known
before = len(df)
df = df.dropna(subset=[f"aqi_{h}h" for h in FORECAST_HOURS])
# Fill any remaining NaNs in lag features (first N rows of series have no history)
df = df.fillna(0.0)
print(f"  Kept {len(df)} / {before} rows (dropped last 72 rows without future targets)")


# ── 6. Finalize schema ────────────────────────────────────────────────────────
df = df.reset_index()
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Ensure all expected columns exist with correct types
for col in ["pm25", "pm10", "o3", "no2", "so2", "co",
            "temperature", "humidity", "pressure", "wind_speed",
            "visibility", "cloud_cover", "precipitation_1h"]:
    if col not in df.columns:
        df[col] = 0.0
    df[col] = df[col].fillna(0.0).astype(float)

# Integer targets
for h in FORECAST_HOURS:
    df[f"aqi_{h}h"] = df[f"aqi_{h}h"].astype(int)

print(f"\n[backfill] Final dataset: {len(df)} rows, {len(df.columns)} columns")
print(f"  AQI range: {df['aqi'].min()}–{df['aqi'].max()}")
print(f"  Target range (24h): {df['aqi_24h'].min()}–{df['aqi_24h'].max()}")
print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

# ── Fix types to match Hopsworks feature group schema ──────────────────────────
# bigint columns
for col in ["hour", "month", "season", "is_weekend", "aqi"] + [f"dow_{i}" for i in range(7)]:
    if col in df.columns:
        df[col] = df[col].astype("int64")

# targets must be float (schema is 'double')
for h in FORECAST_HOURS:
    df[f"aqi_{h}h"] = df[f"aqi_{h}h"].astype("float64")

# float32 → float64 for all float columns
for col in df.select_dtypes(include="float32").columns:
    df[col] = df[col].astype("float64")

# Drop columns not in the feature group schema
DROP_COLS = ["wind_deg", "nh3"]
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])



# ── 7. Fit scaler on this data ────────────────────────────────────────────────
print("\n[backfill] Fitting scaler on training split (first 70%)...")
n_train = int(len(df) * 0.70)
df_train = df.iloc[:n_train].copy()
fit_scaler(df_train)
print("  Scaler saved to scaler_bundle.pkl")

# ── 8. Upload to Hopsworks ────────────────────────────────────────────────────
print("\n[backfill] Uploading to Hopsworks...")
fs = get_feature_store()
fg = get_or_create_feature_group(fs)
fg.insert(df, write_options={"wait_for_job": True})
print(f"[backfill] Done! {len(df)} labeled rows uploaded.")
print("\nYou can now re-trigger the training pipeline on GitHub Actions.")
