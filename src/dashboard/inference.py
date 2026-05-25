"""
Load champion model from MongoDB and run inference for the dashboard.
Returns 3-day AQI predictions with conformal prediction intervals.

Caching strategy
----------------
- _MODEL_CACHE    : champion model + scaler — never expires
- _FEATURES_CACHE : recent feature history — TTL 3600 s
- _METADATA_CACHE : leaderboard model metadata — TTL 1800 s
- _SHAP_CACHE     : SHAP values — TTL 3600 s

All caches survive across Dash callbacks within one container lifecycle.
A background thread pre-warms _METADATA_CACHE and _SHAP_CACHE at module import
time so the first tab switch feels instant.
"""
import io
import pickle
import zipfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import DEFAULT_CITY, DEFAULT_LAT, DEFAULT_LON, FORECAST_HOURS
from src.db import get_db, get_gridfs
from src.feature_pipeline.store_features import fetch_recent
from src.feature_pipeline.feature_engineering import load_scaler, apply_scaler, pm25_to_aqi
from src.feature_pipeline.fetch_data import fetch_current
from src.feature_pipeline.feature_engineering import build_feature_row
from src.training_pipeline.conformal import load_calibrator, predict_with_intervals

SHAP_PATH = Path(__file__).parent.parent.parent / "shap_values.pkl"

# ─── Module-level caches ──────────────────────────────────────────────────────
_MODEL_CACHE: dict = {"model": None, "scaler": None}

_FEATURES_CACHE: dict = {"data": None, "city": None, "ts": None}
_FEATURES_CACHE_TTL = 3600

_METADATA_CACHE: dict = {"data": None, "ts": None}
_METADATA_CACHE_TTL = 1800

_SHAP_CACHE: dict = {"data": None, "ts": None}
_SHAP_CACHE_TTL = 3600

_cache_lock = threading.Lock()

_PREDICTION_LOG_TTL_DAYS = 30


# ─── Champion model ───────────────────────────────────────────────────────────

def load_champion_model():
    """Download and cache the champion model from MongoDB GridFS.

    The champion ZIP contains: model.pkl/model.keras, model_type.txt,
    scaler_bundle.pkl, shap_values.pkl, conformal_calibrator.pkl.
    """
    with _cache_lock:
        if _MODEL_CACHE["model"] is not None:
            print("[inference] Returning cached champion model")
            return _MODEL_CACHE["model"], _MODEL_CACHE["scaler"]

    import traceback
    try:
        meta = get_db()["model_metadata"].find_one(
            {"model_name": "aqi_champion"}, {"_id": 0}
        )
        if not meta or not meta.get("artifacts_gridfs_id"):
            print("[inference] No champion model artifacts found in MongoDB")
            return None, None

        print("[inference] Downloading champion artifacts from GridFS...")
        zip_bytes = get_gridfs().get(meta["artifacts_gridfs_id"]).read()
        buf = io.BytesIO(zip_bytes)
        model = None
        scaler = None

        with zipfile.ZipFile(buf, "r") as zf:
            model_type = "sklearn"
            if "model_type.txt" in zf.namelist():
                model_type = zf.read("model_type.txt").decode().strip()

            if model_type == "keras":
                try:
                    import tempfile
                    import tensorflow as tf
                    with tempfile.TemporaryDirectory() as td:
                        zf.extract("model.keras", td)
                        model = tf.keras.models.load_model(os.path.join(td, "model.keras"))
                    print("[inference] Champion Keras model loaded")
                except ImportError:
                    print("[inference] Champion is Keras but tensorflow not installed — "
                          "re-run training so a sklearn model wins the champion slot")
                    return None, None
            else:
                model = pickle.loads(zf.read("model.pkl"))
                print("[inference] Champion sklearn model loaded")

            if "scaler_bundle.pkl" in zf.namelist():
                scaler = pickle.loads(zf.read("scaler_bundle.pkl"))

        with _cache_lock:
            _MODEL_CACHE["model"]  = model
            _MODEL_CACHE["scaler"] = scaler
        return model, scaler

    except Exception as e:
        print(f"[inference] CHAMPION LOAD FAILED: {e}\n{traceback.format_exc()}")
        return None, None


# ─── Feature history ──────────────────────────────────────────────────────────

def get_recent_features(city: str = DEFAULT_CITY, n: int = 48) -> pd.DataFrame:
    """Fetch the most recent n feature rows for the given city (cached)."""
    now = datetime.now(timezone.utc)
    with _cache_lock:
        cached = _FEATURES_CACHE
        if (
            cached["data"] is not None
            and cached["city"] == city
            and cached["ts"] is not None
            and (now - cached["ts"]).total_seconds() < _FEATURES_CACHE_TTL
        ):
            print("[inference] Returning cached feature history")
            return cached["data"]

    print("[inference] Fetching feature history from MongoDB...")
    result = fetch_recent(city, n=n)
    with _cache_lock:
        _FEATURES_CACHE["data"] = result
        _FEATURES_CACHE["city"] = city
        _FEATURES_CACHE["ts"]   = now
    return result


# ─── 3-day prediction ─────────────────────────────────────────────────────────

def predict_3day(
    city: str = DEFAULT_CITY,
    lat:  float = DEFAULT_LAT,
    lon:  float = DEFAULT_LON,
) -> dict:
    """
    Run end-to-end inference:
    1. Fetch latest raw data
    2. Build feature row
    3. Apply scaler
    4. Predict with champion model
    5. Add conformal intervals for aqi_24h

    Always returns a valid dict — never raises.
    """
    current_aqi = 0
    raw = {}
    try:
        raw = fetch_current(city=city, lat=lat, lon=lon)
        # Use AQICN's already-computed AQI directly; fall back to PM2.5 conversion
        # only if the API didn't return one. pm25_to_aqi(pm25_from_openweather)
        # causes a slight mismatch because OpenWeather and AQICN use different sensors.
        current_aqi = float(raw.get("aqi_from_api") or 0) or pm25_to_aqi(raw.get("pm25", 0))
    except Exception as e:
        print(f"[inference] fetch_current failed: {e}")
        return {
            "current_aqi": 0,
            "aqi_24h": 0, "aqi_48h": 0, "aqi_72h": 0,
            "lower_24h": 0, "upper_24h": 0,
            "error": f"Could not fetch live data: {e}",
        }

    try:
        history = get_recent_features(city=city)
    except Exception as e:
        print(f"[inference] get_recent_features failed: {e}")
        history = pd.DataFrame()

    try:
        ts  = datetime.now(timezone.utc)
        row = build_feature_row(raw, history, ts)
        df_row = pd.DataFrame([row])
    except Exception as e:
        print(f"[inference] build_feature_row failed: {e}")
        return {
            "current_aqi": current_aqi,
            "aqi_24h": current_aqi, "aqi_48h": current_aqi, "aqi_72h": current_aqi,
            "lower_24h": current_aqi * 0.85, "upper_24h": current_aqi * 1.15,
            "error": f"Feature engineering failed: {e}",
        }

    model, scaler = load_champion_model()
    if model is None:
        return {
            "current_aqi": current_aqi,
            "aqi_24h": current_aqi, "aqi_48h": current_aqi, "aqi_72h": current_aqi,
            "lower_24h": current_aqi * 0.85, "upper_24h": current_aqi * 1.15,
            "error": "No champion model yet — run the training pipeline first.",
        }

    try:
        if scaler is not None:
            df_row = apply_scaler(df_row, scaler)

        drop_cols = ["timestamp", "city", "aqi_24h", "aqi_48h", "aqi_72h"]
        feat_cols = [c for c in df_row.columns if c not in drop_cols and df_row[c].dtype != object]
        X = df_row[feat_cols].values

        try:
            import tensorflow as tf
            if isinstance(model, tf.keras.Model):
                expected_features = model.input_shape[-1]
                if X.shape[1] != expected_features:
                    return {
                        "current_aqi": current_aqi,
                        "aqi_24h": current_aqi, "aqi_48h": current_aqi, "aqi_72h": current_aqi,
                        "lower_24h": current_aqi * 0.85, "upper_24h": current_aqi * 1.15,
                        "error": (
                            f"Champion (LSTM) trained on {expected_features} features, "
                            f"current set has {X.shape[1]}. "
                            "Training pipeline will fix this on next run."
                        ),
                    }
                X = X.reshape(1, 1, expected_features)
        except ImportError:
            pass

        preds = model.predict(X)[0]
        if preds.ndim == 0:
            preds = np.array([preds, preds, preds])

        aqi_24h = float(np.clip(preds[0], 0, 500))
        aqi_48h = float(np.clip(preds[1] if len(preds) > 1 else preds[0], 0, 500))
        aqi_72h = float(np.clip(preds[2] if len(preds) > 2 else preds[0], 0, 500))

    except Exception as e:
        print(f"[inference] predict failed: {e}")
        return {
            "current_aqi": current_aqi,
            "aqi_24h": current_aqi, "aqi_48h": current_aqi, "aqi_72h": current_aqi,
            "lower_24h": current_aqi * 0.85, "upper_24h": current_aqi * 1.15,
            "error": f"Prediction failed: {e}",
        }

    mapie = load_calibrator()
    lower_24h, upper_24h = aqi_24h * 0.85, aqi_24h * 1.15
    if mapie is not None:
        try:
            _, lo, hi = predict_with_intervals(mapie, X)
            lower_24h, upper_24h = float(lo[0]), float(hi[0])
        except Exception:
            pass

    result = {
        "current_aqi": current_aqi,
        "aqi_24h":     aqi_24h,
        "aqi_48h":     aqi_48h,
        "aqi_72h":     aqi_72h,
        "lower_24h":   max(0, lower_24h),
        "upper_24h":   min(500, upper_24h),
        "feature_row": row,
    }

    # Resolve model version from MongoDB metadata
    try:
        champ_meta = get_db()["model_metadata"].find_one(
            {"model_name": "aqi_champion"}, {"trained_at": 1}
        )
        model_ver = (champ_meta or {}).get("trained_at", "unknown")
    except Exception:
        model_ver = "unknown"

    threading.Thread(
        target=_log_prediction_async,
        args=(city, current_aqi, aqi_24h, aqi_48h, aqi_72h, model_ver),
        daemon=True,
        name="prediction-logger",
    ).start()

    return result


# ─── Live prediction logging ──────────────────────────────────────────────────

def _log_prediction_async(
    city: str,
    current_aqi: float,
    aqi_24h: float,
    aqi_48h: float,
    aqi_72h: float,
    model_version: str = "unknown",
) -> None:
    """Insert one prediction row into prediction_log.

    The TTL index on logged_at (expireAfterSeconds=2592000) handles
    automatic 30-day rolling-window cleanup — no application pruning needed.
    """
    try:
        now_utc = datetime.now(timezone.utc)
        doc = {
            "city":          city,
            "ts_epoch":      int(now_utc.timestamp()),
            "logged_at":     now_utc,
            "predicted_24h": round(float(aqi_24h),     2),
            "predicted_48h": round(float(aqi_48h),     2),
            "predicted_72h": round(float(aqi_72h),     2),
            "observed_aqi":  round(float(current_aqi), 2),
            "model_version": str(model_version),
        }
        get_db()["prediction_log"].insert_one(doc)
        print(f"[inference] Logged prediction city={city} AQI={current_aqi:.0f} "
              f"→ 24h={aqi_24h:.0f} 48h={aqi_48h:.0f} 72h={aqi_72h:.0f}")
    except Exception as e:
        print(f"[inference] Prediction logging failed (non-fatal): {e}")


def load_prediction_log(city: str = DEFAULT_CITY, days: int = 7) -> pd.DataFrame:
    """Fetch recent prediction log rows for the hindcast chart."""
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
        print(f"[inference] load_prediction_log failed (non-fatal): {e}")
        return pd.DataFrame()


# ─── SHAP data ────────────────────────────────────────────────────────────────

def load_shap_data() -> dict:
    """Load pre-computed SHAP values (cached).

    Tries local file first (dev/training run). Falls back to extracting
    shap_values.pkl from the champion's GridFS artifact ZIP (Render).
    """
    now = datetime.now(timezone.utc)
    with _cache_lock:
        if (
            _SHAP_CACHE["data"] is not None
            and _SHAP_CACHE["ts"] is not None
            and (now - _SHAP_CACHE["ts"]).total_seconds() < _SHAP_CACHE_TTL
        ):
            return _SHAP_CACHE["data"]

    data = {}
    if SHAP_PATH.exists():
        with open(SHAP_PATH, "rb") as f:
            data = pickle.load(f)
    else:
        try:
            meta = get_db()["model_metadata"].find_one({"model_name": "aqi_champion"})
            if meta and meta.get("artifacts_gridfs_id"):
                zip_bytes = get_gridfs().get(meta["artifacts_gridfs_id"]).read()
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                    if "shap_values.pkl" in zf.namelist():
                        data = pickle.loads(zf.read("shap_values.pkl"))
        except Exception as e:
            print(f"[inference] Could not load SHAP from MongoDB: {e}")

    with _cache_lock:
        _SHAP_CACHE["data"] = data
        _SHAP_CACHE["ts"]   = now
    return data


# ─── Leaderboard metadata ─────────────────────────────────────────────────────

def load_all_models_metadata() -> list[dict]:
    """Fetch metrics for all registered models (for the leaderboard).

    Single MongoDB query — no per-model downloads, no KNOWN_MODEL_NAMES loop.
    """
    now = datetime.now(timezone.utc)
    with _cache_lock:
        if (
            _METADATA_CACHE["data"] is not None
            and _METADATA_CACHE["ts"] is not None
            and (now - _METADATA_CACHE["ts"]).total_seconds() < _METADATA_CACHE_TTL
        ):
            print("[inference] Returning cached leaderboard metadata")
            return _METADATA_CACHE["data"]

    try:
        cursor = get_db()["model_metadata"].find(
            {},
            {"_id": 0, "artifacts_gridfs_id": 0},
        )
        results = []
        for doc in cursor:
            if doc.get("model_name") == "aqi_champion":
                continue  # champion is shown via promoted_from in the originating model row
            results.append({
                "name":        doc.get("model_name"),
                "version":     doc.get("trained_at", "-"),
                "rmse":        doc.get("rmse"),
                "ioa":         doc.get("ioa"),
                "skill_score": doc.get("skill_score"),
                "oof_rmse":    doc.get("oof_rmse"),
                "topsis_score":doc.get("topsis_score"),
                "overfit":     bool(doc.get("overfit_flag", 0)),
                "is_champion": bool(doc.get("is_champion", False)),
            })

        # Mark the promoted_from model as champion
        champ = get_db()["model_metadata"].find_one({"model_name": "aqi_champion"})
        if champ and champ.get("promoted_from"):
            for r in results:
                if r["name"] == champ["promoted_from"]:
                    r["is_champion"] = True

        data = results or [{"name": "No models registered yet", "version": "-"}]
    except Exception as e:
        print(f"[inference] load_all_models_metadata failed: {e}")
        return [{"name": f"MongoDB unavailable: {e}", "version": "-"}]

    with _cache_lock:
        _METADATA_CACHE["data"] = data
        _METADATA_CACHE["ts"]   = now
    return data


# ─── Background cache warm-up ─────────────────────────────────────────────────

def _warm_cache():
    try:
        print("[inference] Background warm-up: loading leaderboard metadata...")
        load_all_models_metadata()
        print("[inference] Background warm-up: loading SHAP data...")
        load_shap_data()
        print("[inference] Background warm-up complete.")
    except Exception as e:
        print(f"[inference] Background warm-up error (non-fatal): {e}")


threading.Thread(target=_warm_cache, daemon=True, name="inference-warmup").start()
