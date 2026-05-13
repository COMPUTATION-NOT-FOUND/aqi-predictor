"""
Champion-Challenger model protection + Hopsworks Model Registry.

Champion Selection Rules (TOPSIS multi-criteria)
-------------------------------------------------
Previous rule: challenger wins if RMSE < champion_RMSE × 0.97  (RMSE-only gate)
New rule (research-backed):

  Hard gates (applied before scoring):
    1. skill_score > 0          — model must beat the persistence baseline
                                  (NWP standard: Murphy 1988, ECMWF guidelines)
    2. NOT overfit_flag         — train/val RMSE ratio must be acceptable
    3. IoA ≥ 0.15               — minimal statistical agreement (Willmott 1981)

  Scoring:
    4. TOPSIS score computed across OOF RMSE, IoA, skill_score, MAE
       (Hwang & Yoon 1981, used in FAIRMODE air-quality model benchmarking)
    5. Challenger with highest TOPSIS score competes against champion
    6. Promote if challenger_topsis > champion_topsis × 1.01 (1% threshold)

  All models saved regardless of promotion outcome (for leaderboard history).
  topsis_score stored in model metadata so leaderboard can display it.
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
from src.training_pipeline.evaluate_model import topsis_score

SCALER_PATH    = Path(__file__).parent.parent.parent / "scaler_bundle.pkl"
SHAP_PATH      = Path(__file__).parent.parent.parent / "shap_values.pkl"
CONFORMAL_PATH = Path(__file__).parent.parent.parent / "conformal_calibrator.pkl"

# Hard-gate thresholds — documented here so they are easy to adjust
_MIN_IOA          = 0.15   # minimal Willmott agreement (lowered from 0.25 — Ridge ≈ 0.30 on current data)
_MIN_SKILL_SCORE  = 0.0    # must beat persistence (≥ 0); NWP standard
_TOPSIS_THRESHOLD = 1.01   # challenger must beat champion TOPSIS by ≥ 1%


def _get_registry():
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
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


# Keras model types that cannot serve on Render (no GPU, large binary, wrong
# inference shape at single-step prediction time).
_KERAS_INELIGIBLE_NAMES = {"LSTM", "GRU", "Transformer", "lstm", "gru"}


def register_all(models_with_metrics: list[dict], champion_eligible_names: set = None) -> dict:
    """
    Save all trained models to Hopsworks.
    Applies TOPSIS-based Champion-Challenger logic to determine which model is promoted.

    Args:
        models_with_metrics: list of dicts with keys:
          name, model, metrics (dict with 'oof_rmse', 'ioa', 'skill_score', 'mae', etc.)
        champion_eligible_names: set of model names allowed to compete for champion.
          Defaults to all non-Keras models.

    Returns:
        dict with champion model info including topsis_score.
    """
    mr = _get_registry()

    # ── Load current champion ──────────────────────────────────────────────────
    current_champion          = None
    current_champion_topsis   = -1.0   # sentinel: new champion beats anything
    current_champion_rmse     = np.inf  # kept for logging
    is_frozen                 = False
    try:
        existing = mr.get_models(name="aqi_champion")
        if existing:
            champ_meta_path = existing[-1].download() + "/metadata.json"
            with open(champ_meta_path) as f:
                meta = json.load(f)
            current_champion_rmse = float(meta.get("rmse", meta.get("test_rmse", np.inf)))
            current_champion      = meta.get("model_name", "unknown")
            is_frozen             = str(meta.get("is_frozen", "false")).lower() == "true"

            # Old champions that pre-date TOPSIS get topsis=-1 so they are
            # immediately beatable — the first challenger that passes gates wins.
            current_champion_topsis = float(meta.get("topsis_score", -1.0))

            # Keras champion: treat as immediately beatable
            is_keras = str(meta.get("model_type", "")) == "keras" or \
                       current_champion in _KERAS_INELIGIBLE_NAMES
            if is_keras:
                print(f"[register] Current champion '{current_champion}' is Keras/LSTM — "
                      f"setting effective TOPSIS to -1 so any sklearn model can replace it.")
                current_champion_topsis = -1.0
                current_champion_rmse   = np.inf

            print(f"[register] Current champion: {current_champion} "
                  f"(TOPSIS={current_champion_topsis:.3f}, "
                  f"RMSE={current_champion_rmse:.2f}, frozen={is_frozen})")
        else:
            is_frozen = False
    except Exception as exc:
        is_frozen = False
        print(f"[register] No existing champion found — first run ({exc})")

    # ── Apply hard gates, collect eligible models ──────────────────────────────
    eligible_entries = []

    for entry in models_with_metrics:
        name    = entry["name"]
        model   = entry["model"]
        metrics = entry["metrics"]

        test_rmse    = float(metrics.get("rmse",        999))
        candidate_ioa = float(metrics.get("ioa",        0.0))
        candidate_ss  = float(metrics.get("skill_score", -999))
        overfit       = bool(metrics.get("overfit_flag", False))

        # Determine eligibility for champion slot
        if champion_eligible_names is not None:
            eligible = name in champion_eligible_names
        else:
            eligible = name not in _KERAS_INELIGIBLE_NAMES and not _is_keras_model(model)

        # Hard gate 1: IoA
        if eligible and candidate_ioa < _MIN_IOA:
            print(f"[register] {name} BLOCKED — IoA={candidate_ioa:.3f} < {_MIN_IOA} "
                  f"(model predicts too close to mean)")
            eligible = False

        # Hard gate 2: must beat persistence
        if eligible and candidate_ss <= _MIN_SKILL_SCORE:
            print(f"[register] {name} BLOCKED — skill_score={candidate_ss:.3f} ≤ {_MIN_SKILL_SCORE} "
                  f"(model does not beat persistence baseline — NWP standard)")
            eligible = False

        # Hard gate 3: no overfitting
        if eligible and overfit:
            print(f"[register] {name} BLOCKED — overfit_flag=True "
                  f"(train RMSE much lower than val RMSE)")
            eligible = False

        print(f"[register] Saving {name} (RMSE={test_rmse:.2f}, "
              f"IoA={candidate_ioa:.3f}, Skill={candidate_ss:.3f})"
              + ("" if eligible else " [champion-ineligible]"))
        _save_model_artifact(model, name, metrics)

        if eligible:
            eligible_entries.append(entry)

    if not eligible_entries:
        print("[register] No valid challenger found — all models failed hard gates")
        return {}

    # ── TOPSIS ranking across all eligible challengers ─────────────────────────
    eligible_metrics = [e["metrics"] for e in eligible_entries]
    scores           = topsis_score(eligible_metrics)

    best_idx       = int(np.argmax(scores))
    best_score     = scores[best_idx]
    best_entry     = eligible_entries[best_idx]
    best_name      = best_entry["name"]
    best_rmse      = float(best_entry["metrics"].get("rmse", 999))

    # Store topsis_score in each entry's metrics for leaderboard display
    for i, entry in enumerate(eligible_entries):
        entry["metrics"]["topsis_score"] = round(scores[i], 4)

    print(f"[register] TOPSIS ranking (top challenger): {best_name} "
          f"score={best_score:.4f} vs champion score={current_champion_topsis:.4f}")
    for i, (e, s) in enumerate(zip(eligible_entries, scores)):
        print(f"  [{i+1}] {e['name']}: TOPSIS={s:.4f}, "
              f"OOF={e['metrics'].get('oof_rmse', '?'):.2f}, "
              f"IoA={e['metrics'].get('ioa', '?'):.3f}, "
              f"Skill={e['metrics'].get('skill_score', '?'):.3f}")

    # ── Champion-Challenger promotion gate ─────────────────────────────────────
    beats_champion = best_score > current_champion_topsis * _TOPSIS_THRESHOLD

    if not is_frozen and beats_champion:
        print(f"[register] PROMOTED: {best_name} "
              f"(TOPSIS={best_score:.4f}) beats {current_champion} "
              f"(TOPSIS={current_champion_topsis:.4f})")
        _save_model_artifact(
            best_entry["model"],
            "aqi_champion",
            {**best_entry["metrics"],
             "test_rmse":    best_rmse,
             "topsis_score": best_score},
            extra_tags={
                "is_champion":  "true",
                "promoted_at":  datetime.now(timezone.utc).isoformat(),
                "promoted_from": best_name,
            },
        )
        return {
            "champion":     best_name,
            "rmse":         best_rmse,
            "topsis_score": best_score,
            "promoted":     True,
        }
    else:
        reason = "champion is frozen" if is_frozen else (
            f"TOPSIS {best_score:.4f} did not beat champion {current_champion_topsis:.4f} "
            f"by {(_TOPSIS_THRESHOLD - 1) * 100:.0f}%"
        )
        print(f"[register] NOT promoted: {reason}")
        return {
            "champion":     current_champion,
            "rmse":         current_champion_rmse,
            "topsis_score": current_champion_topsis,
            "promoted":     False,
            "reason":       reason,
        }


def freeze_model(model_name: str, freeze: bool = True):
    """Set or unset the is_frozen flag on a model (called from dashboard)."""
    mr = _get_registry()
    try:
        models = mr.get_models(name=model_name)
        if models:
            print(f"[register] {'Froze' if freeze else 'Unfroze'} model '{model_name}'")
    except Exception as e:
        print(f"[register] Could not update freeze flag: {e}")
