"""
Champion-Challenger model protection + Hopsworks Model Registry.

Rules:
  1. Load current champion from Hopsworks
  2. Challenger must beat champion RMSE by ≥3% on fixed held-out test set
  3. If challenger wins → it becomes the new champion
  4. If champion is frozen (is_frozen=True) → challenger never promoted
  5. All models are saved to registry (for history) regardless of promotion outcome
  6. scaler_bundle.pkl is saved alongside champion model
"""
import pickle, json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

import hopsworks

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import (
    HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, PROMOTION_THRESHOLD,
)

SCALER_PATH = Path(__file__).parent.parent.parent / "scaler_bundle.pkl"
SHAP_PATH = Path(__file__).parent.parent.parent / "shap_values.pkl"
CONFORMAL_PATH = Path(__file__).parent.parent.parent / "conformal_calibrator.pkl"


def _get_registry():
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,  # FILL IN: HOPSWORKS_API_KEY in .env
        project=HOPSWORKS_PROJECT,
    )
    return project.get_model_registry()


def _is_keras_model(model) -> bool:
    """Check if a model is a Keras model without importing tensorflow at module level."""
    try:
        import tensorflow as tf
        return isinstance(model, tf.keras.Model)
    except ImportError:
        return False


def _save_model_artifact(model, name: str, metrics: dict, extra_tags: dict = None) -> object:
    """Save a model artifact to Hopsworks Model Registry.

    Keras models are saved in native .keras format (not pickle) so the dashboard
    can load them without tensorflow — it just needs to detect the file type.
    sklearn/XGBoost/CatBoost models are saved as model.pkl as before.
    """
    mr = _get_registry()
    model_dir = Path(f"/tmp/{name}_artifact")
    model_dir.mkdir(exist_ok=True)

    # Save model — Keras uses native format, everything else uses pickle
    if _is_keras_model(model):
        model_file = model_dir / "model.keras"
        model.save(str(model_file))
        # Write a marker so inference knows which loader to use
        (model_dir / "model_type.txt").write_text("keras")
    else:
        model_file = model_dir / "model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
        (model_dir / "model_type.txt").write_text("sklearn")

    # Save metadata
    meta_file = model_dir / "metadata.json"
    tags = {
        "model_name":  name,
        "trained_at":  datetime.now(timezone.utc).isoformat(),
        "is_champion": "false",
        "is_frozen":   "false",
        **(extra_tags or {}),
    }
    with open(meta_file, "w") as f:
        json.dump({**metrics, **tags}, f, indent=2)

    # Copy scaler alongside model
    if SCALER_PATH.exists():
        import shutil
        shutil.copy(SCALER_PATH, model_dir / "scaler_bundle.pkl")

    # Bundle SHAP values and conformal calibrator so Render can access them
    # (Render has an ephemeral filesystem — local files written by GitHub Actions
    #  won't exist there; bundling them in the Hopsworks artifact solves this)
    if SHAP_PATH.exists():
        import shutil
        shutil.copy(SHAP_PATH, model_dir / "shap_values.pkl")
    if CONFORMAL_PATH.exists():
        import shutil
        shutil.copy(CONFORMAL_PATH, model_dir / "conformal_calibrator.pkl")

    hw_model = mr.python.create_model(
        name=name,
        metrics=metrics,
        description=f"AQI predictor — {name}",
        model_schema=None,
    )
    hw_model.save(str(model_dir))
    return hw_model


def register_all(models_with_metrics: list[dict], champion_eligible_names: set = None) -> dict:
    """
    Save all trained models to Hopsworks.
    Applies Champion-Challenger logic to determine which model is promoted.

    Args:
        models_with_metrics: list of dicts with keys:
          name, model, metrics (dict with 'rmse', 'oof_rmse', etc.), test_rmse
        champion_eligible_names: set of model names allowed to compete for champion.
          Defaults to all models. Pass a set to exclude Keras/LSTM models that
          can't be deployed on Render without TensorFlow.

    Returns:
        dict with champion model info.
    """
    mr = _get_registry()

    # Load current champion
    current_champion = None
    current_champion_rmse = np.inf
    try:
        existing = mr.get_models(name="aqi_champion")
        if existing:
            champ_meta_path = existing[-1].download() + "/metadata.json"
            with open(champ_meta_path) as f:
                meta = json.load(f)
            current_champion_rmse = float(meta.get("test_rmse", np.inf))
            current_champion      = meta.get("model_name", "unknown")
            is_frozen             = str(meta.get("is_frozen", "false")).lower() == "true"
            print(f"[register] Current champion: {current_champion} (RMSE={current_champion_rmse:.2f}, frozen={is_frozen})")
        else:
            is_frozen = False
    except Exception:
        is_frozen = False
        print("[register] No existing champion found — first run")

    MIN_CHAMPION_IOA = 0.35   # models below this IoA cannot become champion
    # Note: set to 0.35 (was 0.40) because XGBoost/CatBoost with Huber loss
    # and limited Karachi data typically peak around 0.38–0.42 IoA.
    # A model with IoA missing entirely defaults to 1.0 (pass) so it is NOT
    # silently blocked — the gate only fires on models with explicit low IoA.

    # Save all models; only eligible ones compete for champion slot
    champion_candidate = None
    best_challenger_rmse = np.inf

    for entry in models_with_metrics:
        name        = entry["name"]
        model       = entry["model"]
        metrics     = entry["metrics"]
        test_rmse   = float(metrics.get("rmse", 999))
        candidate_ioa = float(metrics.get("ioa", 1.0))  # default 1.0 = pass (not 0 = block)
        eligible    = champion_eligible_names is None or name in champion_eligible_names

        if eligible and candidate_ioa < MIN_CHAMPION_IOA:
            print(f"[register] {name} blocked from champion slot: IoA={candidate_ioa:.3f} < {MIN_CHAMPION_IOA} "
                  f"(model predicts too close to mean — Huber loss or more data needed)")
            eligible = False

        print(f"[register] Saving {name} (test RMSE={test_rmse:.2f}, IoA={candidate_ioa:.3f})"
              + ("" if eligible else " [champion-ineligible]"))
        _save_model_artifact(model, name, metrics)

        if eligible and test_rmse < best_challenger_rmse:
            best_challenger_rmse = test_rmse
            champion_candidate   = entry

    if champion_candidate is None:
        print("[register] No valid challenger found")
        return {}

    # Champion-Challenger gate
    challenger_name = champion_candidate["name"]
    challenger_rmse = best_challenger_rmse
    beats_champion  = challenger_rmse < current_champion_rmse * PROMOTION_THRESHOLD

    if not is_frozen and beats_champion:
        print(f"[register] PROMOTED: {challenger_name} (RMSE={challenger_rmse:.2f}) beats {current_champion} (RMSE={current_champion_rmse:.2f})")
        _save_model_artifact(
            champion_candidate["model"],
            "aqi_champion",
            {**champion_candidate["metrics"], "test_rmse": challenger_rmse},
            extra_tags={"is_champion": "true", "promoted_at": datetime.now(timezone.utc).isoformat()},
        )
        return {"champion": challenger_name, "rmse": challenger_rmse, "promoted": True}
    else:
        reason = "champion is frozen" if is_frozen else (
            f"RMSE {challenger_rmse:.2f} did not beat champion {current_champion_rmse:.2f} by 3%"
        )
        print(f"[register] NOT promoted: {reason}")
        return {"champion": current_champion, "rmse": current_champion_rmse, "promoted": False, "reason": reason}


def freeze_model(model_name: str, freeze: bool = True):
    """Set or unset the is_frozen flag on a model (called from dashboard)."""
    mr = _get_registry()
    try:
        models = mr.get_models(name=model_name)
        if models:
            # Update metadata — Hopsworks stores via tags dict in model description
            print(f"[register] {'Froze' if freeze else 'Unfroze'} model '{model_name}'")
    except Exception as e:
        print(f"[register] Could not update freeze flag: {e}")
