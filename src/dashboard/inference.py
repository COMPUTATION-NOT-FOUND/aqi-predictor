"""
Dashboard data-access layer (framework-agnostic).

The dashboard NEVER runs model inference live. Forecasts are precomputed hourly by
src/feature_pipeline/predict.py and written to the `predictions` collection; this
module just reads them (plus feature history, the forecast track record, SHAP values,
and the leaderboard). The Streamlit app wraps these with @st.cache_data for caching
and invalidation — so there is no module-level cache machinery here.
"""
import io
import pickle
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import DEFAULT_CITY
from src.db import get_db, get_gridfs
from src.feature_pipeline.store_features import fetch_recent

SHAP_PATH = Path(__file__).parent.parent.parent / "shap_values.pkl"

# A precomputed forecast older than this is flagged stale in the UI.
_PREDICTION_STALE_MINUTES = 180


# ─── Feature history (Panel A / Panel C context) ──────────────────────────────

def get_recent_features(city: str = DEFAULT_CITY, n: int = 200) -> pd.DataFrame:
    """Most recent n feature rows for the city (observed AQI history)."""
    try:
        return fetch_recent(city, n=n)
    except Exception as e:
        print(f"[inference] get_recent_features failed: {e}")
        return pd.DataFrame()


# ─── Latest precomputed forecast (Panel C + gauges) ────────────────────────────

def load_latest_prediction(city: str = DEFAULT_CITY) -> dict:
    """Read the most recent precomputed forecast document.

    Always returns a dict. When no forecast exists yet, `error` is set and the
    horizon values are 0 — the app shows an explicit "forecast pending" state
    rather than silent persistence/fallback values.
    """
    base = {
        "current_aqi": 0.0, "aqi_24h": 0.0, "aqi_48h": 0.0, "aqi_72h": 0.0,
        "lower_24h": 0.0, "upper_24h": 0.0,
        "predicted_at": None, "model_version": None,
        "stale": False, "age_minutes": None, "error": None,
    }
    try:
        doc = get_db()["predictions"].find_one({"city": city}, sort=[("predicted_at", -1)])
    except Exception as e:
        base["error"] = f"Could not reach the database: {e}"
        return base

    if not doc:
        base["error"] = "No forecast yet — the prediction pipeline has not run."
        return base

    by_h = {f["horizon"]: f for f in doc.get("forecasts", [])}
    predicted_at = doc.get("predicted_at")
    age_minutes = None
    stale = False
    if isinstance(predicted_at, datetime):
        if predicted_at.tzinfo is None:
            predicted_at = predicted_at.replace(tzinfo=timezone.utc)
        age_minutes = (datetime.now(timezone.utc) - predicted_at).total_seconds() / 60.0
        stale = age_minutes > _PREDICTION_STALE_MINUTES

    return {
        "current_aqi":   float(doc.get("current_aqi", 0) or 0),
        "aqi_24h":       float(by_h.get(24, {}).get("aqi", 0) or 0),
        "aqi_48h":       float(by_h.get(48, {}).get("aqi", 0) or 0),
        "aqi_72h":       float(by_h.get(72, {}).get("aqi", 0) or 0),
        "lower_24h":     float(by_h.get(24, {}).get("lower") or 0),
        "upper_24h":     float(by_h.get(24, {}).get("upper") or 0),
        "predicted_at":  predicted_at,
        "model_version": doc.get("model_version"),
        "stale":         stale,
        "age_minutes":   age_minutes,
        "error":         None,
    }


# ─── Forecast track record (Panel B — real predicted vs realized) ──────────────

def load_forecast_trackrecord(city: str = DEFAULT_CITY, days: int = 14) -> dict:
    """Join past logged forecasts to the AQI that was actually observed at their
    target time.

    Returns {"observed": DataFrame[timestamp, aqi], "pairs": DataFrame[target_ts,
    horizon, predicted, observed]}. The pairs power the predicted-vs-observed
    comparison and its honest RMSE/IoA/Skill (computed on realized outcomes).
    """
    empty = {"observed": pd.DataFrame(), "pairs": pd.DataFrame()}
    try:
        db = get_db()
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days)
        cutoff_epoch = int(cutoff.timestamp())

        logs = list(db["prediction_log"].find(
            {"city": city, "ts_epoch": {"$gte": cutoff_epoch}},
            {"_id": 0},
        ).sort("ts_epoch", 1))

        feats = list(db["aqi_features"].find(
            {"city": city, "timestamp": {"$gte": cutoff}},
            {"_id": 0, "timestamp": 1, "aqi": 1},
        ).sort("timestamp", 1))
    except Exception as e:
        print(f"[inference] load_forecast_trackrecord failed: {e}")
        return empty

    if not feats:
        return empty
    obs_df = pd.DataFrame(feats)
    obs_df["timestamp"] = pd.to_datetime(obs_df["timestamp"], utc=True)
    obs_df["aqi"] = pd.to_numeric(obs_df["aqi"], errors="coerce")
    obs_df = obs_df.dropna(subset=["aqi"]).sort_values("timestamp").reset_index(drop=True)

    if not logs or obs_df.empty:
        return {"observed": obs_df, "pairs": pd.DataFrame()}

    # Build (target_ts, horizon, predicted) rows from each logged forecast.
    rows = []
    for log in logs:
        made_at = pd.to_datetime(log.get("logged_at"), utc=True, errors="coerce")
        if pd.isna(made_at):
            made_at = pd.to_datetime(log.get("ts_epoch"), unit="s", utc=True, errors="coerce")
        if pd.isna(made_at):
            continue
        for h in (24, 48, 72):
            val = log.get(f"predicted_{h}h")
            if val is None:
                continue
            rows.append({"target_ts": made_at + pd.Timedelta(hours=h),
                         "horizon": h, "predicted": float(val)})
    if not rows:
        return {"observed": obs_df, "pairs": pd.DataFrame()}

    pred_df = pd.DataFrame(rows).sort_values("target_ts").reset_index(drop=True)

    # Match each forecast to the observed AQI nearest its target time (±90 min).
    pairs = pd.merge_asof(
        pred_df, obs_df.rename(columns={"timestamp": "target_ts", "aqi": "observed"}),
        on="target_ts", direction="nearest", tolerance=pd.Timedelta(minutes=90),
    ).dropna(subset=["observed"]).reset_index(drop=True)

    return {"observed": obs_df, "pairs": pairs}


def load_oof_predictions(city: str = DEFAULT_CITY) -> pd.DataFrame:
    """Out-of-fold predictions of the current champion across historical dates.

    Written by the training pipeline (`_save_champion_oof_predictions`), refreshed on
    every champion promotion. Leakage-free: the champion never saw these rows during
    the fold it was scored on. Columns: timestamp, predicted, observed.
    """
    try:
        from src.feature_pipeline.store_features import fetch_oof_predictions
        return fetch_oof_predictions(city)
    except Exception as e:
        print(f"[inference] load_oof_predictions failed: {e}")
        return pd.DataFrame()


def load_prediction_log(city: str = DEFAULT_CITY, days: int = 7) -> pd.DataFrame:
    """Raw prediction_log rows (kept for diagnostics / downstream use)."""
    try:
        cutoff = int(datetime.now(timezone.utc).timestamp()) - days * 86400
        cursor = get_db()["prediction_log"].find(
            {"city": city, "ts_epoch": {"$gte": cutoff}},
            {"_id": 0, "logged_at": 0},
        ).sort("ts_epoch", 1)
        df = pd.DataFrame(list(cursor))
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["ts_epoch"], unit="s", utc=True)
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"[inference] load_prediction_log failed: {e}")
        return pd.DataFrame()


# ─── SHAP values (Feature Importance tab) ──────────────────────────────────────

def load_shap_data() -> dict:
    """Load precomputed SHAP values — local file first, then champion GridFS bundle."""
    if SHAP_PATH.exists():
        try:
            with open(SHAP_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[inference] Local SHAP load failed: {e}")

    try:
        meta = get_db()["model_metadata"].find_one({"model_name": "aqi_champion"})
        if meta and meta.get("artifacts_gridfs_id"):
            zip_bytes = get_gridfs().get(meta["artifacts_gridfs_id"]).read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                if "shap_values.pkl" in zf.namelist():
                    return pickle.loads(zf.read("shap_values.pkl"))
    except Exception as e:
        print(f"[inference] Could not load SHAP from MongoDB: {e}")
    return {}


# ─── Leaderboard metadata ──────────────────────────────────────────────────────

def load_all_models_metadata() -> list[dict]:
    """Metrics for every registered model (for the leaderboard). Single query."""
    try:
        cursor = get_db()["model_metadata"].find({}, {"_id": 0, "artifacts_gridfs_id": 0})
        results = []
        for doc in cursor:
            if doc.get("model_name") == "aqi_champion":
                continue  # shown via promoted_from on the originating model row
            results.append({
                "name":         doc.get("model_name"),
                "version":      doc.get("trained_at", "-"),
                "rmse":         doc.get("rmse"),
                "rmse_d1":      doc.get("rmse_d1"),
                "rmse_d2":      doc.get("rmse_d2"),
                "rmse_d3":      doc.get("rmse_d3"),
                "ioa":          doc.get("ioa"),
                "skill_score":  doc.get("skill_score"),
                "oof_rmse":     doc.get("oof_rmse"),
                "topsis_score": doc.get("topsis_score"),
                "overfit":      bool(doc.get("overfit_flag", 0)),
                "is_champion":  bool(doc.get("is_champion", False)),
            })

        champ = get_db()["model_metadata"].find_one({"model_name": "aqi_champion"})
        if champ and champ.get("promoted_from"):
            for r in results:
                if r["name"] == champ["promoted_from"]:
                    r["is_champion"] = True

        return results or [{"name": "No models registered yet", "version": "-"}]
    except Exception as e:
        print(f"[inference] load_all_models_metadata failed: {e}")
        return [{"name": f"MongoDB unavailable: {e}", "version": "-"}]
