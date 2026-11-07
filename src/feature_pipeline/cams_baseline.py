"""
CAMS baseline: the forecast Copernicus (CAMS) already publishes, scored the same
way as our models.

Every hourly run stores the CAMS PM2.5 forecast for +24/48/72 h, converted to AQI
with the same 24-hour-mean rule used for our targets. Training then joins those
stored forecasts to the AQI that was later observed and puts a "CAMS" row on the
leaderboard. If our models can't beat a free public forecast, that is worth
knowing.

Only forecasts issued at the time are used, so there is no look-ahead.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import FORECAST_HOURS
from src.feature_pipeline.feature_engineering import pm25_to_aqi

COLLECTION = "cams_forecasts"


def log_cams_forecast(city: str, issued_at: datetime, forecast: dict) -> None:
    """Store one row per run: the CAMS AQI forecast for each horizon."""
    from src.db import get_db
    doc = {"city": city, "issued_at": issued_at}
    for h in FORECAST_HOURS:
        pm = (forecast.get(h) or {}).get("pm25_fc")
        doc[f"aqi_{h}h"] = float(pm25_to_aqi(pm)) if pm and pm > 0 else None
    if all(doc[f"aqi_{h}h"] is None for h in FORECAST_HOURS):
        print("[cams] No CAMS forecast this run; nothing logged")
        return
    try:
        get_db()[COLLECTION].insert_one(doc)
    except Exception as e:
        print(f"[cams] log failed (non-fatal): {e}")


def load_cams_forecasts(city: str, since: datetime) -> pd.DataFrame:
    from src.db import get_db
    rows = list(get_db()[COLLECTION].find(
        {"city": city, "issued_at": {"$gte": since}}, {"_id": 0, "city": 0},
    ))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["issued_at"] = pd.to_datetime(df["issued_at"], utc=True)
    return df.sort_values("issued_at").reset_index(drop=True)


def align_to_rows(timestamps, cams: pd.DataFrame, tolerance_min: int = 45) -> np.ndarray:
    """
    For each row timestamp, the CAMS forecast issued in the same run.
    Returns (n_rows, n_horizons) with NaN where no forecast was logged.
    """
    n = len(timestamps)
    out = np.full((n, len(FORECAST_HOURS)), np.nan)
    if cams.empty or n == 0:
        return out
    rows = pd.DataFrame({"ts": pd.to_datetime(pd.Series(timestamps), utc=True),
                         "pos": np.arange(n)}).sort_values("ts")
    merged = pd.merge_asof(rows, cams, left_on="ts", right_on="issued_at",
                           direction="nearest",
                           tolerance=pd.Timedelta(minutes=tolerance_min))
    for j, h in enumerate(FORECAST_HOURS):
        col = f"aqi_{h}h"
        if col in merged:
            out[merged["pos"].to_numpy(), j] = pd.to_numeric(merged[col], errors="coerce")
    return out


MIN_ROWS = 48   # below this the score is noise; skip the leaderboard row


def evaluate_cams(city, test_timestamps, Y_test, y_pers):
    """
    Leaderboard row for CAMS, scored on the test rows that have a logged CAMS
    forecast. Returns None until enough forecasts have been logged.
    """
    from src.training_pipeline.evaluate_model import evaluate_all
    try:
        ts = pd.to_datetime(pd.Series(test_timestamps), utc=True)
        cams = load_cams_forecasts(city, since=ts.min() - pd.Timedelta(hours=1))
        fc = align_to_rows(ts, cams)
    except Exception as e:
        print(f"[cams] Skipped: {e}")
        return None

    mask = ~np.isnan(fc).any(axis=1)
    n = int(mask.sum())
    if n < MIN_ROWS:
        print(f"[cams] Only {n} test rows have a CAMS forecast (need {MIN_ROWS}). Skipped.")
        return None

    pers = np.asarray(y_pers).reshape(len(Y_test), -1)[mask]
    metrics = evaluate_all(np.asarray(Y_test)[mask], fc[mask], pers)
    # CAMS is scored on a subset of the test set, so note how many rows.
    metrics["n_eval_rows"] = n
    metrics["oof_rmse"] = metrics["rmse"]
    metrics["train_rmse"] = metrics["rmse"]
    metrics["overfit_flag"] = 0
    print(f"[cams] CAMS on {n} test rows: RMSE={metrics['rmse']:.2f} | "
          f"Skill={metrics.get('skill_score', 0):.3f}")
    return {"name": "CAMS", "model": None, "metrics": metrics}
