"""
migrate_v1_to_v2.py — One-shot Feature Store migration: v1 → v2

Why this exists
---------------
When 6 new columns were added to feature_engineering.py
(aqi_lag_48h, aqi_lag_72h, aqi_lag_168h, rolling_min_24h,
rolling_max_24h, aqi_pct_change_24h) the live Hopsworks Feature Group
was still on schema version 1.  The hourly pipeline crashed because
Hopsworks rejected inserts that contained columns unknown to the old
schema.

The fix is to bump FEATURE_VERSION to 2 in config.py (already done).
This script copies every row from v1 into v2, back-filling the 6
missing columns with sensible placeholder values so the training
pipeline can use the full history immediately instead of waiting days.

Back-fill strategy for the 6 columns
--------------------------------------
* aqi_lag_48h   → same as aqi_lag_24h  (best available proxy)
* aqi_lag_72h   → same as aqi_lag_24h
* aqi_lag_168h  → same as aqi_lag_24h  (no 7-day history exists in v1)
* rolling_min_24h → aqi_lag_1h (rough lower bound)
* rolling_max_24h → aqi_lag_1h (rough upper bound — will be equal to min,
                    i.e. zero variance; acceptable for historical rows)
* aqi_pct_change_24h → 0.0  (unknown, neutral placeholder)

These placeholders are clearly imperfect but far better than
discarding 180 days of training data.  A note is logged for each
so the situation is auditable.

Usage
-----
    python src/backfill/migrate_v1_to_v2.py

Run ONCE after bumping FEATURE_VERSION to 2 and before the next
training pipeline run.
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.config import (
    HOPSWORKS_API_KEY, HOPSWORKS_PROJECT,
    HOPSWORKS_FG_NAME,
)

V1 = 1
V2 = 2  # must match the new FEATURE_VERSION in config.py

# New columns added in the v2 schema (did not exist in v1 feature group)
NEW_COLUMNS_V2 = [
    "aqi_lag_48h",
    "aqi_lag_72h",
    "aqi_lag_168h",
    "rolling_min_24h",
    "rolling_max_24h",
    "aqi_pct_change_24h",
]


def _back_fill_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 6 v2-only columns using conservative placeholders."""
    for col in ["aqi_lag_48h", "aqi_lag_72h", "aqi_lag_168h"]:
        if col not in df.columns:
            src = "aqi_lag_24h" if "aqi_lag_24h" in df.columns else "aqi"
            df[col] = df[src]
            print(f"  [backfill] {col} ← copied from '{src}'")

    for col in ["rolling_min_24h", "rolling_max_24h"]:
        if col not in df.columns:
            src = "aqi_lag_1h" if "aqi_lag_1h" in df.columns else "aqi"
            df[col] = df[src]
            print(f"  [backfill] {col} ← copied from '{src}' (rough proxy)")

    if "aqi_pct_change_24h" not in df.columns:
        df["aqi_pct_change_24h"] = 0.0
        print("  [backfill] aqi_pct_change_24h ← 0.0 (neutral placeholder)")

    return df


def migrate():
    import hopsworks

    print("[migrate] Logging into Hopsworks...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()

    # ── Read from v1 ──────────────────────────────────────────────────────────
    print(f"[migrate] Reading all rows from '{HOPSWORKS_FG_NAME}' v{V1}...")
    print("[migrate] NOTE: Using fg.read() (direct Hive path) — skipping Arrow Flight")
    print("[migrate]       This avoids the gRPC Socket-closed error on the free tier.")
    fg_v1 = fs.get_feature_group(name=HOPSWORKS_FG_NAME, version=V1)

    # Use fg.read() directly — much more reliable than the Arrow Flight / Feature View
    # path which frequently drops connections on large reads over the free-tier quota.
    df = fg_v1.read()

    print(f"[migrate] Read {len(df):,} rows from v{V1}.")
    if df.empty:
        print("[migrate] v1 is empty — nothing to migrate. Exiting.")
        return

    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Back-fill new columns ─────────────────────────────────────────────────
    print("[migrate] Back-filling missing v2 columns...")
    df = _back_fill_missing_columns(df)

    missing_still = [c for c in NEW_COLUMNS_V2 if c not in df.columns]
    if missing_still:
        raise RuntimeError(f"Could not back-fill columns: {missing_still}")

    # ── Write to v2 ───────────────────────────────────────────────────────────
    print(f"[migrate] Writing {len(df):,} rows into '{HOPSWORKS_FG_NAME}' v{V2}...")
    fg_v2 = fs.get_or_create_feature_group(
        name=HOPSWORKS_FG_NAME,
        version=V2,
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        description=(
            "Hourly AQI features for Karachi (v2: adds extended lag + rolling bounds). "
            "Migrated from v1 on first run of migrate_v1_to_v2.py."
        ),
    )
    # ensure timestamp dtype is correct
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    fg_v2.insert(df, write_options={"wait_for_job": True})
    print(f"[migrate] ✓ Migration complete — {len(df):,} rows now in v{V2}.")
    print("[migrate] You can now run the hourly pipeline and the training pipeline against v2.")


if __name__ == "__main__":
    migrate()
