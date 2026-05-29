"""
Retroactively fill null aqi_24h/48h/72h targets for all stored feature rows.

For each row at timestamp t with a missing target, look up the stored `aqi`
value from the row closest to t+24h (or +48h / +72h) within a ±30-minute
window. Uses only AQICN-sourced AQI values already in MongoDB — no new API
calls, no OpenWeather PM2.5 conversion.

Run automatically by the daily training GitHub Action (before train_model.py).
Can also be run manually: python src/backfill/fill_targets.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from datetime import timedelta
from pymongo import UpdateOne

from src.db import get_db
from src.config import DEFAULT_CITY, FORECAST_HOURS


def fill_null_targets(city: str = DEFAULT_CITY) -> int:
    """Fill null aqi_{h}h targets from stored rows h hours later.

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
                    "aqi": {"$exists": True, "$ne": None, "$gt": 0},
                },
                {"aqi": 1},
                sort=[("timestamp", 1)],
            )
            if future:
                patch[f"aqi_{h}h"] = float(future["aqi"])

        if patch:
            updates.append(UpdateOne({"_id": row["_id"]}, {"$set": patch}))

        # Flush in batches to avoid building a huge list in memory
        if len(updates) >= 500:
            col.bulk_write(updates, ordered=False)
            updates.clear()

    if updates:
        col.bulk_write(updates, ordered=False)

    return len(updates)


if __name__ == "__main__":
    n = fill_null_targets()
    print(f"[fill_targets] Updated targets for {n} rows")
