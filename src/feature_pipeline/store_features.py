"""
Push and fetch engineered feature rows from MongoDB.
Primary key: (city, timestamp) — enforced via unique compound index.
"""
import pandas as pd
from pymongo import UpdateOne

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.db import get_db


def insert_features(df: pd.DataFrame) -> None:
    """Upsert a DataFrame of feature rows into the 'aqi_features' collection.

    Safe to call repeatedly — existing rows are updated, new rows are inserted.
    Requires a unique compound index on (city, timestamp) in MongoDB Atlas.
    """
    if df.empty:
        return
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    col = get_db()["aqi_features"]
    docs = df.to_dict(orient="records")
    # BSON requires Python datetime, not pandas Timestamp
    for doc in docs:
        if hasattr(doc["timestamp"], "to_pydatetime"):
            doc["timestamp"] = doc["timestamp"].to_pydatetime()
    ops = [
        UpdateOne(
            {"city": d["city"], "timestamp": d["timestamp"]},
            {"$set": d},
            upsert=True,
        )
        for d in docs
    ]
    col.bulk_write(ops, ordered=False)
    print(f"[store_features] Upserted {len(docs)} rows into 'aqi_features'")


def fetch_training_data(start_time: str = None, end_time: str = None) -> pd.DataFrame:
    """Pull all historical features from MongoDB for model training."""
    col = get_db()["aqi_features"]
    query: dict = {}
    if start_time or end_time:
        query["timestamp"] = {}
        if start_time:
            query["timestamp"]["$gte"] = pd.to_datetime(start_time, utc=True).to_pydatetime()
        if end_time:
            query["timestamp"]["$lte"] = pd.to_datetime(end_time, utc=True).to_pydatetime()
    cursor = col.find(query, {"_id": 0}).sort("timestamp", 1)
    df = pd.DataFrame(list(cursor))
    if df.empty:
        print("[store_features] fetch_training_data: no rows found")
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    print(f"[store_features] fetch_training_data: returned {len(df)} rows")
    return df.reset_index(drop=True)


def fetch_oof_predictions(city: str = "karachi") -> pd.DataFrame:
    """Return saved OOF predictions for the champion model (used by the hindcast chart)."""
    col = get_db()["oof_predictions"]
    cursor = col.find({"city": city}, {"_id": 0}).sort("timestamp", 1)
    rows = list(cursor)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def fetch_recent(city: str, n: int = 72) -> pd.DataFrame:
    """Fetch the most recent n feature rows for a city.

    Used by the feature pipeline and dashboard to build lag features
    without pulling the entire training set.
    """
    col = get_db()["aqi_features"]
    cursor = col.find({"city": city}, {"_id": 0}).sort("timestamp", -1).limit(n)
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)
