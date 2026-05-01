"""
Main feature pipeline entry point — runs every hour via GitHub Actions.

Steps:
1. Fetch raw data from AQICN + OpenWeather
2. Build feature row (engineering + encoding)
3. Apply pre-fit scaler (loaded from disk)
4. Push to Hopsworks Feature Store
5. Run PSI drift check; trigger alert if needed
"""
import pandas as pd
from datetime import datetime, timezone
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.config import DEFAULT_CITY, DEFAULT_LAT, DEFAULT_LON
from src.feature_pipeline.fetch_data import fetch_current
from src.feature_pipeline.feature_engineering import (
    build_feature_row, pm25_to_aqi, load_scaler, apply_scaler,
)
from src.feature_pipeline.store_features import insert_features, fetch_training_data
from src.feature_pipeline.drift_monitor import check_drift


def run_pipeline(city: str = DEFAULT_CITY, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON):
    print(f"[pipeline] Running feature pipeline for '{city}' at {datetime.now(timezone.utc).isoformat()}")

    # 1. Fetch raw data
    raw = fetch_current(city=city, lat=lat, lon=lon)

    # 2. Fetch recent history for lag/rolling features (last 72 rows)
    try:
        history = fetch_training_data()
        history = history.tail(72)
    except Exception:
        history = pd.DataFrame()

    # 3. Build feature row
    ts  = datetime.now(timezone.utc)
    row = build_feature_row(raw, history, ts)
    row["aqi"] = pm25_to_aqi(raw.get("pm25", 0))

    df = pd.DataFrame([row])

    # 4. Apply pre-fit scaler (if available)
    scaler = load_scaler()
    if scaler is not None:
        df = apply_scaler(df, scaler)
    else:
        print("[pipeline] No scaler found — storing raw (unscaled) features for now")

    # 5. Push to Hopsworks
    insert_features(df)

    # 6. Drift check
    _, alert = check_drift(df)
    if alert:
        print("[pipeline] DRIFT ALERT — check email / GitHub Actions notification")

    print(f"[pipeline] Done. AQI={row['aqi']}")
    return row


if __name__ == "__main__":
    run_pipeline()
