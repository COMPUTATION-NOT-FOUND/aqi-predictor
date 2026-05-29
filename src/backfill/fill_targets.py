"""
Two jobs run by the daily training GitHub Action (before train_model.py):

1. fix_aqi_column() — one-time migration: corrects stored `aqi` values for
   live pipeline rows that were written using AQICN's composite AQI (~161)
   instead of pm25_to_aqi(openweather_pm25) (~61). Uses the stored `pm25`
   field which was always sourced from OpenWeather and is reliable.

2. fill_null_targets() — ongoing: fills null aqi_24h/48h/72h for rows whose
   future is now known. Uses pm25_to_aqi(future_row["pm25"]) — not
   future_row["aqi"] — so it stays correct regardless of what the aqi column
   contained before the migration.

Run automatically by the daily training GitHub Action (before train_model.py).
Can also be run manually: python src/backfill/fill_targets.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from datetime import timedelta
from pymongo import UpdateOne

from src.db import get_db
from src.config import DEFAULT_CITY, FORECAST_HOURS
from src.feature_pipeline.feature_engineering import pm25_to_aqi


def fix_aqi_column(city: str = DEFAULT_CITY) -> int:
    """Recompute the `aqi` field from stored `pm25` for every row.

    Idempotent — safe to run repeatedly. Corrects live pipeline rows that
    stored AQICN's composite AQI instead of pm25_to_aqi(pm25).
    Returns the number of documents updated.
    """
    col = get_db()["aqi_features"]
    updates = []
    for row in col.find({"city": city, "pm25": {"$exists": True, "$gt": 0}},
                        {"_id": 1, "pm25": 1, "aqi": 1}):
        correct_aqi = float(pm25_to_aqi(row.get("pm25", 0)))
        if abs(row.get("aqi", 0) - correct_aqi) > 1:
            updates.append(UpdateOne({"_id": row["_id"]}, {"$set": {"aqi": correct_aqi}}))
        if len(updates) >= 500:
            col.bulk_write(updates, ordered=False)
            updates.clear()
    if updates:
        col.bulk_write(updates, ordered=False)
    return len(updates)


def fill_null_targets(city: str = DEFAULT_CITY) -> int:
    """Fill null aqi_{h}h targets using pm25_to_aqi(future_row["pm25"]).

    For each row at timestamp t with a missing target, looks up the stored
    pm25 value from the row nearest t+h hours and converts via pm25_to_aqi.
    Using pm25 directly (not the aqi field) makes this robust to any prior
    aqi column contamination.
    Returns the number of documents updated.
    """
    col = get_db()["aqi_features"]

    null_filter = {
        "city": city,
        "$or": [{f"aqi_{h}h": None} for h in FORECAST_HOURS],
    }

    updates = []
    for row in col.find(null_filter, {"_id": 1, "timestamp": 1,
                                      **{f"aqi_{h}h": 1 for h in FORECAST_HOURS}}):
        ts    = row["timestamp"]
        patch = {}

        for h in FORECAST_HOURS:
            if row.get(f"aqi_{h}h") is not None:
                continue

            future = col.find_one(
                {
                    "city":      city,
                    "timestamp": {
                        "$gte": ts + timedelta(hours=h) - timedelta(minutes=30),
                        "$lte": ts + timedelta(hours=h) + timedelta(minutes=30),
                    },
                    "pm25": {"$exists": True, "$gt": 0},
                },
                {"pm25": 1},
                sort=[("timestamp", 1)],
            )
            if future:
                patch[f"aqi_{h}h"] = float(pm25_to_aqi(future["pm25"]))

        if patch:
            updates.append(UpdateOne({"_id": row["_id"]}, {"$set": patch}))

        if len(updates) >= 500:
            col.bulk_write(updates, ordered=False)
            updates.clear()

    if updates:
        col.bulk_write(updates, ordered=False)

    return len(updates)


if __name__ == "__main__":
    print("[fill_targets] Step 1: correcting aqi column from pm25...")
    n_fixed = fix_aqi_column()
    print(f"[fill_targets] Corrected aqi column in {n_fixed} rows")

    print("[fill_targets] Step 2: filling null training targets...")
    n_filled = fill_null_targets()
    print(f"[fill_targets] Filled targets for {n_filled} rows")
