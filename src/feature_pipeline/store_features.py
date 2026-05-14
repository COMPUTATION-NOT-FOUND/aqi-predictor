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
    """Pull all historical features from Hopsworks for model training.

    Tries the Feature View first (preferred). Falls back to direct Feature Group
    read if the Feature View returns empty — this happens on Hopsworks free tier
    when offline materialization Spark jobs fail.
    """
    fs = get_feature_store()
    fg = fs.get_feature_group(name=HOPSWORKS_FG_NAME, version=FEATURE_VERSION)
    fv_name = "aqi_feature_view"

    # Try feature view first
    try:
        fv = fs.get_feature_view(name=fv_name, version=FEATURE_VERSION)
        df = fv.get_batch_data(start_time=start_time, end_time=end_time)
        df = df.sort_values("timestamp").reset_index(drop=True)
        if len(df) > 10:
            return df
        print(f"[store_features] Feature view returned {len(df)} rows — falling back to direct FG read")
    except Exception as e:
        print(f"[store_features] Feature view read failed ({e}) — falling back to direct FG read")

    # Fallback: read directly from the feature group
    try:
        df = fg.read()
        df = df.sort_values("timestamp").reset_index(drop=True)
        print(f"[store_features] Direct FG read returned {len(df)} rows")
        return df
    except Exception as e2:
        print(f"[store_features] Direct FG read also failed: {e2}")
        return pd.DataFrame()


