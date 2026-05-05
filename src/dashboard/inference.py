"""
Load champion model + scaler from Hopsworks and run inference for the dashboard.
Returns 3-day AQI predictions with conformal prediction intervals.
"""
import numpy as np
import pandas as pd
import pickle
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


def load_champion_model():
    """Download and load the champion model from Hopsworks Model Registry.

    Supports two formats written by register_model._save_model_artifact:
      - model.pkl   → sklearn/XGBoost/CatBoost (no keras needed)
      - model.keras → Keras MLP student (requires tensorflow)
    A model_type.txt marker file tells us which format to expect.
    """
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
            with open(f"{model_dir}/model.pkl", "rb") as f:
                model = pickle.load(f)
            print("[inference] Champion sklearn model loaded successfully")

        scaler_path = Path(model_dir) / "scaler_bundle.pkl"
        scaler = None
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        print(f"[inference] CHAMPION LOAD FAILED: {e}\n{traceback.format_exc()}")
        return None, None


def get_recent_features(city: str = DEFAULT_CITY, n: int = 72) -> pd.DataFrame:
    """Fetch the most recent n feature rows for the given city."""
    df = fetch_training_data()
    if "city" in df.columns:
        df = df[df["city"] == city]
    return df.tail(n)


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
    from datetime import datetime, timezone
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

    # Predict
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


def load_shap_data() -> dict:
    """Load pre-computed SHAP values.

    Tries local file first (fast, works in dev). Falls back to Hopsworks
    model artifact so Render (ephemeral filesystem) can access it.
    """
    if SHAP_PATH.exists():
        with open(SHAP_PATH, "rb") as f:
            return pickle.load(f)

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
                    return pickle.load(f)
    except Exception as e:
        print(f"[inference] Could not load SHAP from Hopsworks: {e}")
    return {}


def load_all_models_metadata() -> list[dict]:
    """Fetch metadata for all registered models (for the leaderboard).

    NOTE: Hopsworks SDK 4.7 changed get_models() to require a 'name' argument.
    We iterate over all known model names instead of calling get_models() bare.
    """
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
                    results.append({
                        "name":        m.name,
                        "version":     m.version,
                        "rmse":        m.training_metrics.get("rmse", None) if m.training_metrics else None,
                        "ioa":         m.training_metrics.get("ioa",  None) if m.training_metrics else None,
                        "skill_score": m.training_metrics.get("skill_score", None) if m.training_metrics else None,
                        "oof_rmse":    m.training_metrics.get("oof_rmse", None) if m.training_metrics else None,
                        "overfit":     bool(m.training_metrics.get("overfit_flag", 0)) if m.training_metrics else False,
                        "is_champion": m.name == "aqi_champion",
                    })
            except Exception:
                # Model name simply doesn't exist in registry yet — skip silently
                pass
        return results if results else [{"name": "No models registered yet", "version": "-", "status": "challenger"}]
    except Exception as e:
        return [{"name": f"Error loading models: {e}", "version": "-"}]
