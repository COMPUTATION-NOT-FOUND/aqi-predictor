"""
Main feature pipeline entry point — runs every hour via GitHub Actions.

Steps:
1. Fetch raw data from AQICN + OpenWeather
2. Build feature row (engineering + encoding)
3. Push raw (unscaled) features to MongoDB Feature Store
4. Run PSI drift check; trigger alert if needed
"""
import pandas as pd
from datetime import datetime, timezone, timedelta
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.config import DEFAULT_CITY, DEFAULT_LAT, DEFAULT_LON, FORECAST_HOURS
from src.feature_pipeline.fetch_data import fetch_current
from src.feature_pipeline.feature_engineering import (
    build_feature_row, pm25_to_aqi,
)
from src.feature_pipeline.store_features import insert_features, fetch_recent
from src.feature_pipeline.drift_monitor import check_drift


def _fill_past_targets(city: str, current_ts: datetime, current_aqi: float) -> None:
    """When the pipeline runs at time t with AQICN AQI = A, write A as the target
    for rows stored at t-24h, t-48h, and t-72h — promoting live AQICN readings
    into training targets without waiting for a manual backfill."""
    from src.db import get_db
    col = get_db()["aqi_features"]
    for h in FORECAST_HOURS:
        target_ts    = current_ts - timedelta(hours=h)
        window_start = target_ts - timedelta(minutes=30)
        window_end   = target_ts + timedelta(minutes=30)
        col.update_many(
            {
                "city":          city,
                "timestamp":     {"$gte": window_start, "$lte": window_end},
                f"aqi_{h}h":     None,
            },
            {"$set": {f"aqi_{h}h": current_aqi}},
        )


def run_pipeline(city: str = DEFAULT_CITY, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON):
    print(f"[pipeline] Running feature pipeline for '{city}' at {datetime.now(timezone.utc).isoformat()}")

    # 1. Fetch raw data
    raw = fetch_current(city=city, lat=lat, lon=lon)

    # 2. Fetch recent history for lag/rolling features (last 72 rows)
    try:
        history = fetch_recent(city, n=72)
    except Exception:
        history = pd.DataFrame()

    # 3. Build feature row (includes aqi via build_feature_row → pm25_to_aqi / aqi_from_api)
    ts  = datetime.now(timezone.utc)
    row = build_feature_row(raw, history, ts)

    df = pd.DataFrame([row])

    # Target columns are unknown at inference time; insert nulls to match schema
    for col in ("aqi_24h", "aqi_48h", "aqi_72h"):
        if col not in df.columns:
            df[col] = float("nan")

    # 4. Push raw (unscaled) features to MongoDB.
    # Scaling is applied at training time (fit on raw training data) and at
    # inference time (apply_scaler in inference.py).  Storing raw values here
    # keeps the two paths consistent and prevents the scaler from being
    # re-fit on already-scaled data during daily training runs.
    insert_features(df)

    # 4b. Promote current pm25-derived AQI as training target for rows 24/48/72h ago.
    # row["aqi"] is now always pm25_to_aqi(openweather_pm25) — consistent with backfill.
    _fill_past_targets(city, ts, float(row["aqi"]))

    # 5. Drift check
    _, alert = check_drift(df)
    if alert:
        print("[pipeline] DRIFT ALERT — check email / GitHub Actions notification")

    print(f"[pipeline] Done. AQI={row['aqi']}")
    return row


if __name__ == "__main__":
    run_pipeline()
