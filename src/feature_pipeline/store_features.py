"""
Push engineered feature rows to the Hopsworks Feature Store.
Uses point-in-time correct inserts to prevent future data leakage.
"""
import pandas as pd
import hopsworks
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import (
    HOPSWORKS_API_KEY, HOPSWORKS_PROJECT,
    HOPSWORKS_FG_NAME, FEATURE_VERSION,
)


def get_feature_store():
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,  # FILL IN: HOPSWORKS_API_KEY in .env
        project=HOPSWORKS_PROJECT,
    )
    return project.get_feature_store()


def get_or_create_feature_group(fs):
    """Get or create the AQI feature group in Hopsworks."""
    return fs.get_or_create_feature_group(
        name=HOPSWORKS_FG_NAME,
        version=FEATURE_VERSION,
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        description="Hourly AQI features for Karachi (and other cities)",
    )


def insert_features(df: pd.DataFrame):
    """Insert a DataFrame of feature rows into the Hopsworks Feature Group."""
    if df.empty:
        return
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"[store_features] Inserted {len(df)} rows into '{HOPSWORKS_FG_NAME}'")


def fetch_training_data(start_time: str = None, end_time: str = None) -> pd.DataFrame:
    """Pull all historical features from Hopsworks for model training."""
    fs = get_feature_store()
    fg = fs.get_feature_group(name=HOPSWORKS_FG_NAME, version=FEATURE_VERSION)
    fv_name = "aqi_feature_view"

    try:
        fv = fs.get_feature_view(name=fv_name, version=FEATURE_VERSION)
    except Exception:
        fv = None

    if fv is None:
        fv = fs.create_feature_view(
            name=fv_name,
            version=FEATURE_VERSION,
            query=fg.select_all(),
        )

    df = fv.get_batch_data(start_time=start_time, end_time=end_time)
    return df.sort_values("timestamp").reset_index(drop=True)
