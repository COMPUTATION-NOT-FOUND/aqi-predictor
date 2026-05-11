"""
Load champion model + scaler from Hopsworks and run inference for the dashboard.
Returns 3-day AQI predictions with conformal prediction intervals.

Caching strategy
----------------
- _MODEL_CACHE    : champion model + scaler — never expires (model only changes on re-deploy)
- _FEATURES_CACHE : recent feature history — TTL 3600 s (hourly pipeline cadence)
- _METADATA_CACHE : leaderboard model metadata — TTL 1800 s (enough for a session)
- _SHAP_CACHE     : SHAP values — TTL 3600 s (only changes after training pipeline runs)

All caches survive across Dash callbacks within one container lifecycle.
A background thread pre-warms _METADATA_CACHE and _SHAP_CACHE at module import
time so the first tab switch feels instant instead of blocking 30–60 s on Hopsworks.
"""
import numpy as np
import pandas as pd
import pickle
import threading
from datetime import datetime, timezone
from pathlib import Path
import hopsworks

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import (
    HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, DEFAULT_CITY, DEFAULT_LAT, DEFAULT_LON,
    FORECAST_HOURS,
)
from src.feature_pipeline.store_features import fetch_training_data
from src.feature_pipeline.feature_engineering import load_scaler, apply_scaler, pm25_to_aqi
from src.feature_pipeline.fetch_data import fetch_current
from src.feature_pipeline.feature_engineering import build_feature_row
from src.training_pipeline.conformal import load_calibrator, predict_with_intervals

SHAP_PATH = Path(__file__).parent.parent.parent / "shap_values.pkl"

# ─── Module-level caches ──────────────────────────────────────────────────────
_MODEL_CACHE: dict = {"model": None, "scaler": None}

_FEATURES_CACHE: dict = {"data": None, "city": None, "ts": None}
_FEATURES_CACHE_TTL = 3600   # 1 hour — matches hourly pipeline cadence

_METADATA_CACHE: dict = {"data": None, "ts": None}
_METADATA_CACHE_TTL = 1800   # 30 min — leaderboard data doesn't change that fast

_SHAP_CACHE: dict = {"data": None, "ts": None}
_SHAP_CACHE_TTL = 3600       # 1 hour — SHAP only changes after training pipeline runs

_cache_lock = threading.Lock()


# ─── Champion model ───────────────────────────────────────────────────────────

def load_champion_model():
    """Download and load the champion model from Hopsworks Model Registry.

    Supports two formats written by register_model._save_model_artifact:
      - model.pkl   → sklearn/XGBoost/CatBoost (no keras needed)
      - model.keras → Keras MLP student (requires tensorflow)
    A model_type.txt marker file tells us which format to expect.
    """
    with _cache_lock:
        if _MODEL_CACHE["model"] is not None:
            print("[inference] Returning cached champion model (skipping Hopsworks download)")
            return _MODEL_CACHE["model"], _MODEL_CACHE["scaler"]

    import traceback
    print(f"[inference] Connecting to Hopsworks project='{HOPSWORKS_PROJECT}' key={'SET' if HOPSWORKS_API_KEY else 'MISSING'}")
    try:
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT,
        )
    except Exception as e:
        print(f"[inference] HOPSWORKS LOGIN FAILED: {e}\n{traceback.format_exc()}")
        return None, None

    mr = project.get_model_registry()
    try:
        models = mr.get_models(name="aqi_champion")
        if not models:
            print("[inference] No 'aqi_champion' model found in registry")
            return None, None
        best = models[-1]
        print(f"[inference] Found aqi_champion v{best.version} — downloading...")
        model_dir = best.download()

        # Detect model format via marker written at training time
        type_file = Path(model_dir) / "model_type.txt"
        model_type = type_file.read_text().strip() if type_file.exists() else "sklearn"

        if model_type == "keras":
            # Keras MLP distilled student
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(f"{model_dir}/model.keras")
                print("[inference] Champion Keras model loaded successfully")
            except ImportError:
                print("[inference] CHAMPION LOAD FAILED: champion is a Keras model but tensorflow "
                      "is not installed on Render. Add tensorflow to requirements-dashboard.txt, "
                      "or re-run training so a sklearn/XGBoost model wins the champion slot.")
                return None, None
        else:
            # sklearn / XGBoost / CatBoost — safe to unpickle without keras
            try:
                with open(f"{model_dir}/model.pkl", "rb") as f:
                    model = pickle.load(f)
                print("[inference] Champion sklearn model loaded successfully")
            except (ImportError, ModuleNotFoundError) as e:
                print(f"[inference] CHAMPION LOAD FAILED: pickle references a missing package ({e}). "
                      "Re-run training so a pure sklearn/XGBoost/LightGBM/CatBoost model wins "
                      "the champion slot, then deploy the updated code.")
                return None, None

        scaler_path = Path(model_dir) / "scaler_bundle.pkl"
        scaler = None
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        with _cache_lock:
            _MODEL_CACHE["model"]  = model
            _MODEL_CACHE["scaler"] = scaler
        return model, scaler
    except Exception as e:
        print(f"[inference] CHAMPION LOAD FAILED: {e}\n{traceback.format_exc()}")
        return None, None


# ─── Feature history ──────────────────────────────────────────────────────────

def get_recent_features(city: str = DEFAULT_CITY, n: int = 48) -> pd.DataFrame:
    """Fetch the most recent n feature rows for the given city.

    Results are cached for _FEATURES_CACHE_TTL seconds so repeated
    Dash callbacks (tab switches, 5-min refresh) don't re-query Hopsworks.
    n reduced from 72 → 48: STL needs 48 rows minimum; smaller fetch = faster.
    """
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

    print("[inference] Fetching feature history from Hopsworks...")
    df = fetch_training_data()
    if "city" in df.columns:
        df = df[df["city"] == city]
    result = df.tail(n)
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

    Returns dict with keys: current_aqi, aqi_24h, aqi_48h, aqi_72h,
                             lower_24h, upper_24h, feature_row
    """
    # Fetch + feature engineer latest data
    raw = fetch_current(city=city, lat=lat, lon=lon)
    history = get_recent_features(city=city)
    ts  = datetime.now(timezone.utc)
    row = build_feature_row(raw, history, ts)
    current_aqi = pm25_to_aqi(raw.get("pm25", 0))

    df_row = pd.DataFrame([row])

    # Load champion model + scaler
    model, scaler = load_champion_model()
    if model is None:
        return {
            "current_aqi": current_aqi,
            "aqi_24h": current_aqi, "aqi_48h": current_aqi, "aqi_72h": current_aqi,
            "lower_24h": current_aqi, "upper_24h": current_aqi,
            "error": "No champion model available yet. Run training pipeline first.",
        }

    # Scale
    if scaler is not None:
        df_row = apply_scaler(df_row, scaler)

    drop_cols = ["timestamp", "city", "aqi_24h", "aqi_48h", "aqi_72h"]
    feat_cols  = [c for c in df_row.columns if c not in drop_cols and df_row[c].dtype != object]
    X = df_row[feat_cols].values

    # Predict — handle Keras LSTM models that expect 3D input
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            # LSTM/Sequential expects (batch, timesteps, features)
            # We only have 1 timestep here so reshape accordingly.
            # If the model was trained on a different feature count, it will
            # still fail — the next training run on v2 data will produce a
            # sklearn/XGBoost champion that works without this reshape.
            expected_features = model.input_shape[-1]
            if X.shape[1] != expected_features:
                return {
                    "current_aqi": current_aqi,
                    "aqi_24h": current_aqi, "aqi_48h": current_aqi, "aqi_72h": current_aqi,
                    "lower_24h": current_aqi * 0.85, "upper_24h": current_aqi * 1.15,
                    "error": (
                        f"Champion model (LSTM) was trained on {expected_features} features but "
                        f"current feature set has {X.shape[1]}. "
                        "Run the training pipeline to promote a new champion trained on v2 data."
                    ),
                }
            X = X.reshape(1, 1, expected_features)
    except ImportError:
        pass

    preds = model.predict(X)[0]   # shape (3,) for 3 horizons
    if preds.ndim == 0:
        preds = np.array([preds, preds, preds])

    aqi_24h = float(np.clip(preds[0], 0, 500))
    aqi_48h = float(np.clip(preds[1] if len(preds) > 1 else preds[0], 0, 500))
    aqi_72h = float(np.clip(preds[2] if len(preds) > 2 else preds[0], 0, 500))

    # Conformal intervals for 24h prediction
    mapie = load_calibrator()
    lower_24h, upper_24h = aqi_24h * 0.85, aqi_24h * 1.15  # fallback ±15%
    if mapie is not None:
        try:
            _, lo, hi = predict_with_intervals(mapie, X)
            lower_24h, upper_24h = float(lo[0]), float(hi[0])
        except Exception:
            pass

    return {
        "current_aqi": current_aqi,
        "aqi_24h":     aqi_24h,
        "aqi_48h":     aqi_48h,
        "aqi_72h":     aqi_72h,
        "lower_24h":   max(0, lower_24h),
        "upper_24h":   min(500, upper_24h),
        "feature_row": row,
    }


# ─── SHAP data ────────────────────────────────────────────────────────────────

def load_shap_data() -> dict:
    """Load pre-computed SHAP values (cached for _SHAP_CACHE_TTL seconds).

    Tries local file first (fast, works in dev). Falls back to Hopsworks
    model artifact so Render (ephemeral filesystem) can access it.
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
        # Fallback: download shap_values.pkl bundled alongside champion model artifact
        try:
            project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
            mr = project.get_model_registry()
            models = mr.get_models(name="aqi_champion")
            if models:
                model_dir = models[-1].download()
                shap_remote = Path(model_dir) / "shap_values.pkl"
                if shap_remote.exists():
                    with open(shap_remote, "rb") as f:
                        data = pickle.load(f)
        except Exception as e:
            print(f"[inference] Could not load SHAP from Hopsworks: {e}")

    with _cache_lock:
        _SHAP_CACHE["data"] = data
        _SHAP_CACHE["ts"]   = now
    return data


# ─── Leaderboard metadata ─────────────────────────────────────────────────────

def load_all_models_metadata() -> list[dict]:
    """Fetch metadata for all registered models (for the leaderboard).

    Results are cached for _METADATA_CACHE_TTL seconds so repeated tab
    switches don't trigger a full Hopsworks login on every click.

    NOTE: Hopsworks SDK 4.7 changed get_models() to require a 'name' argument.
    We iterate over all known model names instead of calling get_models() bare.
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

    # All names that the training pipeline ever registers
    KNOWN_MODEL_NAMES = [
        "aqi_champion",
        "Ridge", "Lasso", "ElasticNet",
        "RandomForest", "GradientBoosting",
        "XGBoost", "CatBoost",
        "LSTM", "distilled_student",
    ]
    try:
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
        mr = project.get_model_registry()
        results = []
        for model_name in KNOWN_MODEL_NAMES:
            try:
                models = mr.get_models(name=model_name)  # 4.7 requires name kwarg
                for m in (models or []):
                    tm = m.training_metrics or {}
                    results.append({
                        "name":        m.name,
                        "version":     m.version,
                        "rmse":        tm.get("rmse", None),
                        "ioa":         tm.get("ioa",  None),
                        "skill_score": tm.get("skill_score", None),
                        "oof_rmse":    tm.get("oof_rmse", None),
                        "overfit":     bool(tm.get("overfit_flag", 0)),
                        "is_champion": m.name == "aqi_champion",
                    })
            except Exception:
                pass  # model name simply doesn't exist in registry yet — skip silently

        data = results if results else [{"name": "No models registered yet", "version": "-", "status": "challenger"}]
    except Exception as e:
        data = [{"name": f"Error loading models: {e}", "version": "-"}]

    with _cache_lock:
        _METADATA_CACHE["data"] = data
        _METADATA_CACHE["ts"]   = now
    return data


# ─── Background cache warm-up ─────────────────────────────────────────────────

def _warm_cache():
    """Pre-warm metadata + SHAP caches in the background.

    Called once at module import time so the first tab click never
    blocks on a cold Hopsworks request.  Errors are silently swallowed
    because the cache functions have their own error handling.
    """
    try:
        print("[inference] Background warm-up: loading leaderboard metadata...")
        load_all_models_metadata()
        print("[inference] Background warm-up: loading SHAP data...")
        load_shap_data()
        print("[inference] Background warm-up complete.")
    except Exception as e:
        print(f"[inference] Background warm-up error (non-fatal): {e}")


# Kick off warm-up in a daemon thread so it doesn't block gunicorn startup
threading.Thread(target=_warm_cache, daemon=True, name="inference-warmup").start()
