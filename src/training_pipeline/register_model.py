"""
Champion-Challenger model protection + MongoDB Model Registry.

Champion Selection Rules (TOPSIS multi-criteria)
-------------------------------------------------
  Hard gates (applied before scoring):
    1. skill_score > 0          — model must beat the persistence baseline
    2. NOT overfit_flag         — train/val RMSE ratio must be acceptable
    3. IoA ≥ 0.15               — minimal statistical agreement (Willmott 1981)

  Scoring:
    4. TOPSIS score computed across OOF RMSE, IoA, skill_score, MAE
    5. Challenger with highest TOPSIS score competes against champion
    6. Promote if challenger_topsis > champion_topsis × 1.01 (1% threshold)

  All models save metadata regardless of promotion outcome (for leaderboard).
  Only the champion's binary artifacts are stored in GridFS (storage budget).
"""
import io
import json
import pickle
import zipfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.db import get_db, get_gridfs
from src.training_pipeline.evaluate_model import topsis_score

SCALER_PATH    = Path(__file__).parent.parent.parent / "scaler_bundle.pkl"
SHAP_PATH      = Path(__file__).parent.parent.parent / "shap_values.pkl"
CONFORMAL_PATH = Path(__file__).parent.parent.parent / "conformal_calibrator.pkl"

_MIN_IOA          = 0.15
_MIN_SKILL_SCORE  = 0.0
_TOPSIS_THRESHOLD = 1.01

_KERAS_INELIGIBLE_NAMES = {"LSTM", "GRU", "Transformer", "lstm", "gru"}


def _is_keras_model(model) -> bool:
    try:
        import tensorflow as tf
        return isinstance(model, tf.keras.Model)
    except ImportError:
        return False


def _save_metadata(name: str, metrics: dict, tags: dict) -> None:
    """Upsert a model's metrics and tags into the model_metadata collection."""
    doc = {**metrics, **tags, "model_name": name}
    get_db()["model_metadata"].update_one(
        {"model_name": name},
        {"$set": doc},
        upsert=True,
    )


def _save_champion_artifacts(model, metrics: dict, tags: dict,
                             feature_cols: list[str] | None = None) -> None:
    """Bundle the champion artifacts into a ZIP and upload to GridFS.

    Deletes the previous champion ZIP first to reclaim storage space.
    Metadata is then upserted with the new artifacts_gridfs_id.

    `feature_cols` is the exact ordered feature list the model was trained on. It
    is written both into the ZIP (feature_cols.json) and onto the metadata doc, so
    inference can reindex the live feature row to this exact order — eliminating the
    train/serve column-ordering skew that previously corrupted predictions.
    """
    # Delete previous champion artifact to reclaim GridFS space
    prev = get_db()["model_metadata"].find_one(
        {"model_name": "aqi_champion"}, {"artifacts_gridfs_id": 1}
    )
    if prev and prev.get("artifacts_gridfs_id"):
        try:
            get_gridfs().delete(prev["artifacts_gridfs_id"])
        except Exception as e:
            print(f"[register] Could not delete old champion artifact: {e}")

    # Build ZIP in memory — no temp directory needed for pkl models
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if _is_keras_model(model):
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                model_path = os.path.join(td, "model.keras")
                model.save(model_path)
                zf.write(model_path, "model.keras")
            zf.writestr("model_type.txt", "keras")
        else:
            zf.writestr("model.pkl", pickle.dumps(model))
            zf.writestr("model_type.txt", "sklearn")

        if feature_cols is not None:
            zf.writestr("feature_cols.json", json.dumps(list(feature_cols)))
        if SCALER_PATH.exists():
            zf.write(str(SCALER_PATH), "scaler_bundle.pkl")
        if SHAP_PATH.exists():
            zf.write(str(SHAP_PATH), "shap_values.pkl")
        if CONFORMAL_PATH.exists():
            zf.write(str(CONFORMAL_PATH), "conformal_calibrator.pkl")

    buf.seek(0)
    file_id = get_gridfs().put(
        buf,
        filename="aqi_champion_artifacts.zip",
        content_type="application/zip",
    )

    doc = {**metrics, **tags, "model_name": "aqi_champion", "artifacts_gridfs_id": file_id}
    if feature_cols is not None:
        doc["feature_cols"] = list(feature_cols)
    get_db()["model_metadata"].update_one(
        {"model_name": "aqi_champion"},
        {"$set": doc},
        upsert=True,
    )
    print(f"[register] Champion artifacts uploaded to GridFS (id={file_id})")


def register_all(models_with_metrics: list[dict], champion_eligible_names: set = None,
                 feature_cols: list[str] | None = None) -> dict:
    """
    Save all trained models to MongoDB.
    Applies TOPSIS-based Champion-Challenger logic to determine which model is promoted.

    Args:
        models_with_metrics: list of dicts with keys:
          name, model, metrics (dict with 'oof_rmse', 'ioa', 'skill_score', 'mae', etc.)
        champion_eligible_names: set of model names allowed to compete for champion.
          Defaults to all non-Keras models.

    Returns:
        dict with champion model info including topsis_score.
    """
    # ── Load current champion from MongoDB ────────────────────────────────────
    current_champion        = None
    current_champion_topsis = -1.0
    current_champion_rmse   = np.inf
    is_frozen               = False

    try:
        meta = get_db()["model_metadata"].find_one({"model_name": "aqi_champion"})
        if meta:
            current_champion_rmse   = float(meta.get("rmse", meta.get("test_rmse", np.inf)))
            current_champion        = meta.get("promoted_from", "unknown")
            is_frozen               = bool(meta.get("is_frozen", False))
            current_champion_topsis = float(meta.get("topsis_score", -1.0))

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
            print("[register] No existing champion found — first run")
    except Exception as exc:
        print(f"[register] Could not read current champion ({exc})")

    # ── Apply hard gates, save all models, collect eligible challengers ────────
    eligible_entries = []

    for entry in models_with_metrics:
        name    = entry["name"]
        model   = entry["model"]
        metrics = entry["metrics"]

        test_rmse     = float(metrics.get("rmse",        999))
        candidate_ioa = float(metrics.get("ioa",        0.0))
        candidate_ss  = float(metrics.get("skill_score", -999))
        overfit       = bool(metrics.get("overfit_flag", False))

        if champion_eligible_names is not None:
            eligible = name in champion_eligible_names
        else:
            eligible = name not in _KERAS_INELIGIBLE_NAMES and not _is_keras_model(model)

        if eligible and candidate_ioa < _MIN_IOA:
            print(f"[register] {name} BLOCKED — IoA={candidate_ioa:.3f} < {_MIN_IOA}")
            eligible = False

        if eligible and candidate_ss <= _MIN_SKILL_SCORE:
            print(f"[register] {name} BLOCKED — skill_score={candidate_ss:.3f} ≤ {_MIN_SKILL_SCORE}")
            eligible = False

        if eligible and overfit:
            print(f"[register] {name} BLOCKED — overfit_flag=True")
            eligible = False

        print(f"[register] Saving {name} (RMSE={test_rmse:.2f}, "
              f"IoA={candidate_ioa:.3f}, Skill={candidate_ss:.3f})"
              + ("" if eligible else " [champion-ineligible]"))

        _save_metadata(name, metrics, {
            "model_name":  name,
            "trained_at":  datetime.now(timezone.utc).isoformat(),
            "is_champion": False,
            "is_frozen":   False,
            "model_type":  "keras" if _is_keras_model(model) else "sklearn",
            "artifacts_gridfs_id": None,
        })

        if eligible:
            eligible_entries.append(entry)

    if not eligible_entries:
        print("[register] No valid challenger found — all models failed hard gates")
        return {}

    # ── TOPSIS ranking across all eligible challengers ─────────────────────────
    eligible_metrics = [e["metrics"] for e in eligible_entries]
    scores           = topsis_score(eligible_metrics)

    best_idx   = int(np.argmax(scores))
    best_score = scores[best_idx]
    best_entry = eligible_entries[best_idx]
    best_name  = best_entry["name"]
    best_rmse  = float(best_entry["metrics"].get("rmse", 999))

    for i, entry in enumerate(eligible_entries):
        entry["metrics"]["topsis_score"] = round(scores[i], 4)
        _save_metadata(entry["name"], entry["metrics"], {
            "model_name": entry["name"],
            "topsis_score": round(scores[i], 4),
        })

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
        _save_champion_artifacts(
            best_entry["model"],
            {**best_entry["metrics"],
             "test_rmse":    best_rmse,
             "topsis_score": best_score},
            {
                "is_champion":   True,
                "is_frozen":     False,
                "promoted_at":   datetime.now(timezone.utc).isoformat(),
                "promoted_from": best_name,
                "model_type":    "keras" if _is_keras_model(best_entry["model"]) else "sklearn",
            },
            feature_cols=feature_cols,
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


def freeze_model(model_name: str, freeze: bool = True) -> None:
    """Set or unset the is_frozen flag on a model."""
    get_db()["model_metadata"].update_one(
        {"model_name": model_name},
        {"$set": {"is_frozen": freeze}},
    )
    print(f"[register] {'Froze' if freeze else 'Unfroze'} model '{model_name}'")
