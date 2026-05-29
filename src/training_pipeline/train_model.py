"""
Main training pipeline — runs daily via GitHub Actions.

Steps:
1. Fetch historical features from MongoDB
2. Temporal train/val/test split (70/10/20)
3. Fit + re-save RobustScaler on training data
4. Train 12 classical + deep learning models with OOF + TimeSeriesSplit CV
5. Build stacking and voting ensembles
6. Distill best ensemble → lightweight student MLP
7. Calibrate conformal prediction intervals
8. Run feature ablation study
9. Register all models with Champion-Challenger gate
"""
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
warnings.filterwarnings("ignore")

import logging
for _logger in ["mlflow", "mlflow.models.model", "mlflow.sklearn", "mlflow.utils",
                "tensorflow", "absl"]:
    logging.getLogger(_logger).setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import mlflow
import optuna
import shap
import pickle
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import MLFLOW_TRACKING_URI, FORECAST_HOURS
from src.feature_pipeline.store_features import fetch_training_data
from src.feature_pipeline.feature_engineering import fit_scaler, apply_scaler, FEATURE_GROUP_COLS
from src.feature_pipeline.drift_monitor import save_baseline
from src.training_pipeline.models import (
    CLASSICAL_MODELS, OPTUNA_MODELS, build_lstm, lstm_suggest,
)
from sklearn.ensemble import StackingRegressor
from src.training_pipeline.evaluate_model import evaluate_all, overfitting_flag
from src.training_pipeline.distillation import distill
from src.training_pipeline.conformal import calibrate
from src.training_pipeline.ablation_features import run_ablation
from src.training_pipeline.register_model import register_all
from src.training_pipeline.wrappers import FirstOutputWrapper as _FirstOutputWrapper

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Re-apply after imports since mlflow resets its loggers on import
for _logger in ["mlflow", "mlflow.models.model", "mlflow.sklearn", "mlflow.utils",
                "mlflow.tensorflow", "tensorflow", "absl"]:
    logging.getLogger(_logger).setLevel(logging.ERROR)

TARGET_COLS  = [f"aqi_{h}h" for h in FORECAST_HOURS]
N_SPLITS_OOF = 3
SHAP_PATH    = Path(__file__).parent.parent.parent / "shap_values.pkl"


class WeightedVoter:
    def __init__(self, models, weights):
        self.models  = models
        self.weights = weights
    def fit(self, X, Y):
        for m in self.models:
            m.fit(X, Y)
        return self
    def predict(self, X):
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        return np.einsum("i,ijk->jk", self.weights, preds)


class KerasWrapper:
    def __init__(self, model): self.model = model
    def predict(self, X): return self.model.predict(X, verbose=0)



# ─── Data Preparation ─────────────────────────────────────────────────────────

def load_and_split():
    print("[train] Loading features from MongoDB...")
    df = fetch_training_data()
    df = df.dropna(subset=TARGET_COLS)

    from src.config import FEATURE_GROUPS
    drop_meta = ["timestamp", "city"] + TARGET_COLS

    # Exclude columns that belong to currently-disabled feature groups so the
    # model's feature set stays consistent with what build_feature_row produces
    # at inference time (which also respects FEATURE_GROUPS).
    disabled_cols: set[str] = set()
    for group, enabled in FEATURE_GROUPS.items():
        if not enabled:
            disabled_cols.update(FEATURE_GROUP_COLS.get(group, []))

    feature_cols = sorted(
        c for c in df.columns
        if c not in drop_meta and df[c].dtype != object and c not in disabled_cols
    )

    n = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.10)

    # Fit scaler on raw training data only — no leakage into val/test
    df_raw_train = df.iloc[:n_train].copy()  # kept raw for drift baseline
    scaler = fit_scaler(df_raw_train)
    print("[train] Scaler fitted on raw training data")

    # Scale the full dataset; TARGET_COLS and timestamp/city are in SKIP_COLS
    # so they are left in raw AQI units throughout
    df = apply_scaler(df, scaler)

    X = df[feature_cols].values.astype(np.float64)
    X = np.where(np.isfinite(X), X, 0.0)
    Y = df[TARGET_COLS].values       # shape (n, 3) — raw AQI units

    X_train, Y_train = X[:n_train],               Y[:n_train]
    X_val,   Y_val   = X[n_train:n_train+n_val],  Y[n_train:n_train+n_val]
    X_test,  Y_test  = X[n_train+n_val:],         Y[n_train+n_val:]

    print(f"[train] Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, feature_cols, df, df_raw_train


def persistence_baseline(Y_train, Y_test, n_ahead: int = 0) -> np.ndarray:
    """Persistence: predict today's AQI for all horizons."""
    last_train_aqi = Y_train[-1, 0]
    return np.full(Y_test.size, last_train_aqi)


# ─── Training Helpers ─────────────────────────────────────────────────────────

def _oof_rmse(estimator, X, Y, n_splits=N_SPLITS_OOF) -> float:
    """Out-of-fold RMSE using TimeSeriesSplit (primary model selection metric)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds, oof_true = [], []
    for tr, val in tscv.split(X):
        estimator.fit(X[tr], Y[tr])
        oof_preds.append(estimator.predict(X[val]))
        oof_true.append(Y[val])
    y_pred = np.vstack(oof_preds).ravel()
    y_true = np.vstack(oof_true).ravel()
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _save_champion_oof_predictions(
    model, X_train, Y_train, df_train,
    X_val=None, Y_val=None, df_val=None,
    X_test=None, Y_test=None, df_test=None,
) -> None:
    """Save per-sample OOF predictions for the champion model to MongoDB.

    These are read by the dashboard's hindcast chart (Panel B) to overlay
    model predictions on observed AQI without any data leakage.
    Val and test predictions are appended after OOF (model trained on X_train
    only, so these are also leakage-free).
    """
    from src.db import get_db
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_OOF)
    val_indices, preds_24h, obs_24h = [], [], []
    for tr, val in tscv.split(X_train):
        model.fit(X_train[tr], Y_train[tr])
        p = np.asarray(model.predict(X_train[val]))
        val_indices.extend(val.tolist())
        preds_24h.extend((p[:, 0] if p.ndim == 2 else p).tolist())
        obs_24h.extend(Y_train[val, 0].tolist())

    city = str(df_train["city"].iloc[0]) if "city" in df_train.columns else "karachi"
    docs = []
    for idx, pred, obs in zip(val_indices, preds_24h, obs_24h):
        ts = df_train["timestamp"].iloc[idx]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        docs.append({"timestamp": ts, "city": city,
                     "predicted": float(pred), "observed": float(obs)})

    # Refit on full training set, then predict val and test (leakage-free)
    model.fit(X_train, Y_train)
    for X_extra, Y_extra, df_extra in (
        (X_val, Y_val, df_val),
        (X_test, Y_test, df_test),
    ):
        if X_extra is None or df_extra is None:
            continue
        p = np.asarray(model.predict(X_extra))
        preds = (p[:, 0] if p.ndim == 2 else p).tolist()
        for i, (pred, obs) in enumerate(zip(preds, Y_extra[:, 0].tolist())):
            ts = df_extra["timestamp"].iloc[i]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            docs.append({"timestamp": ts, "city": city,
                         "predicted": float(pred), "observed": float(obs)})

    if docs:
        col = get_db()["oof_predictions"]
        col.delete_many({})
        col.insert_many(docs)
        print(f"[train] Saved {len(docs)} OOF+val+test prediction rows to MongoDB")


def train_classical(X_train, Y_train, X_test, Y_test, y_persistence) -> list[dict]:
    results = []
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("aqi_model_training")

    for spec in CLASSICAL_MODELS:
        name = spec["name"]
        print(f"\n[train] === {name} ===")
        with mlflow.start_run(run_name=name):
            est = spec["estimator"]

            # Tune hyperparams
            search = RandomizedSearchCV(
                est, spec["param_dist"],
                n_iter=15, cv=TimeSeriesSplit(n_splits=3),
                scoring="neg_root_mean_squared_error",
                n_jobs=-1, random_state=42, verbose=0,
            )
            search.fit(X_train, Y_train)
            best_est = search.best_estimator_
            best_params = search.best_params_

            # OOF RMSE (leakage-proof)
            oof = _oof_rmse(best_est, X_train, Y_train)
            best_est.fit(X_train, Y_train)   # refit on full train after OOF

            train_preds = best_est.predict(X_train).ravel()
            test_preds  = best_est.predict(X_test).ravel()

            train_rmse = float(np.sqrt(mean_squared_error(Y_train.ravel(), train_preds)))
            metrics    = evaluate_all(Y_test.ravel(), test_preds, y_persistence)
            metrics["oof_rmse"]    = oof
            metrics["train_rmse"]  = train_rmse
            metrics["overfit_flag"] = int(overfitting_flag(train_rmse, metrics["rmse"]))

            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_est, name="model")

            print(f"[train] {name}: RMSE={metrics['rmse']:.2f} | IoA={metrics['ioa']:.3f} | "
                  f"Skill={metrics.get('skill_score', 0):.3f} | OOF={oof:.2f}")

            results.append({"name": name, "model": best_est, "metrics": metrics})

    return results


def train_optuna(X_train, Y_train, X_val, Y_val, X_test, Y_test, y_persistence) -> list[dict]:
    results = []
    mlflow.set_experiment("aqi_model_training")

    for spec in OPTUNA_MODELS:
        name = spec["name"]
        print(f"\n[train] === {name} (Optuna) ===")
        with mlflow.start_run(run_name=name):

            def objective(trial):
                params = spec["suggest"](trial)
                model  = spec["factory"](params)
                tscv   = TimeSeriesSplit(n_splits=3)
                rmses  = []
                for tr, val in tscv.split(X_train):
                    model.fit(X_train[tr], Y_train[tr])
                    preds = model.predict(X_train[val]).ravel()
                    rmses.append(np.sqrt(mean_squared_error(Y_train[val].ravel(), preds)))
                return float(np.mean(rmses))

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=spec["n_trials"], show_progress_bar=False)

            best_params = study.best_params
            best_model  = spec["factory"](best_params)
            oof         = _oof_rmse(best_model, X_train, Y_train)
            best_model.fit(X_train, Y_train)

            train_preds = best_model.predict(X_train).ravel()
            test_preds  = best_model.predict(X_test).ravel()
            train_rmse  = float(np.sqrt(mean_squared_error(Y_train.ravel(), train_preds)))
            metrics     = evaluate_all(Y_test.ravel(), test_preds, y_persistence)
            metrics["oof_rmse"]    = oof
            metrics["train_rmse"]  = train_rmse
            metrics["overfit_flag"] = int(overfitting_flag(train_rmse, metrics["rmse"]))

            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_model, name="model")

            print(f"[train] {name}: RMSE={metrics['rmse']:.2f} | OOF={oof:.2f}")
            results.append({"name": name, "model": best_model, "metrics": metrics})

    return results


def train_lstm(X_train, Y_train, X_val, Y_val, X_test, Y_test, y_persistence) -> dict:
    import optuna, tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    mlflow.set_experiment("aqi_model_training")
    n_outputs = Y_train.shape[1]

    def objective(trial):
        params  = lstm_suggest(trial)
        seq_len = params["sequence_length"]
        n_feat  = X_train.shape[1]

        # Reshape to sequences
        def to_seq(X, Y, seq):
            xs = np.array([X[i:i+seq] for i in range(len(X)-seq)])
            ys = Y[seq:]
            return xs, ys

        Xtr_s, Ytr_s = to_seq(X_train, Y_train, seq_len)
        Xvl_s, Yvl_s = to_seq(X_val,   Y_val,   seq_len)
        if len(Xtr_s) < 50:
            return 999.0

        model = build_lstm(
            units=params["units"], dropout=params["dropout"],
            learning_rate=params["learning_rate"],
            sequence_length=seq_len, n_features=n_feat, n_outputs=n_outputs,
        )
        model.fit(
            Xtr_s, Ytr_s, validation_data=(Xvl_s, Yvl_s),
            epochs=50, batch_size=64, verbose=0,
            callbacks=[EarlyStopping(patience=7, restore_best_weights=True)],
        )
        preds = model.predict(Xvl_s, verbose=0).ravel()
        return float(np.sqrt(mean_squared_error(Yvl_s.ravel(), preds)))

    with mlflow.start_run(run_name="LSTM"):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=15, show_progress_bar=False)
        best = study.best_params
        seq_len = best["sequence_length"]

        def to_seq(X, Y, seq):
            return (np.array([X[i:i+seq] for i in range(len(X)-seq)]), Y[seq:])

        Xtr_s, Ytr_s = to_seq(X_train, Y_train, seq_len)
        Xte_s, Yte_s = to_seq(X_test,  Y_test,  seq_len)
        final_model   = build_lstm(**best, n_features=X_train.shape[1], n_outputs=n_outputs)
        final_model.fit(
            Xtr_s, Ytr_s, epochs=100, batch_size=64, verbose=0,
            callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
        )
        test_preds  = final_model.predict(Xte_s, verbose=0).ravel()
        train_preds = final_model.predict(Xtr_s, verbose=0).ravel()
        train_rmse  = float(np.sqrt(mean_squared_error(Ytr_s.ravel(), train_preds)))
        metrics     = evaluate_all(Yte_s.ravel(), test_preds)
        metrics["oof_rmse"]    = metrics["rmse"]   # LSTM uses val set for OOF proxy
        metrics["train_rmse"]  = train_rmse
        metrics["overfit_flag"] = int(overfitting_flag(train_rmse, metrics["rmse"]))

        mlflow.log_params(best)
        mlflow.log_metrics(metrics)
        lstm_path = "/tmp/lstm_model.keras"
        final_model.save(lstm_path)
        mlflow.tensorflow.log_model(final_model, name="model")

        print(f"[train] LSTM: RMSE={metrics['rmse']:.2f}")
        return {"name": "LSTM", "model": final_model, "metrics": metrics}


def build_ensembles(results: list[dict], X_train, Y_train, X_test, Y_test, y_persistence) -> list[dict]:
    """Build stacking + voting ensembles from the top classical/boosting models."""
    mlflow.set_experiment("aqi_model_training")
    sorted_r = sorted(results, key=lambda r: r["metrics"]["rmse"])
    top3     = sorted_r[:3]

    # Voting ensemble (1/RMSE weights)
    weights = [1.0 / max(r["metrics"]["rmse"], 1e-6) for r in top3]
    w_norm  = [w / sum(weights) for w in weights]

    voter = WeightedVoter([r["model"] for r in top3], w_norm)
    voter.fit(X_train, Y_train)
    test_preds = voter.predict(X_test).ravel()
    v_metrics  = evaluate_all(Y_test.ravel(), test_preds, y_persistence)
    v_metrics["oof_rmse"] = _oof_rmse(voter, X_train, Y_train)
    train_preds_v = voter.predict(X_train).ravel()
    v_metrics["train_rmse"]  = float(np.sqrt(mean_squared_error(Y_train.ravel(), train_preds_v)))
    v_metrics["overfit_flag"] = int(overfitting_flag(v_metrics["train_rmse"], v_metrics["rmse"]))
    ensemble_results = [{"name": "VotingEnsemble", "model": voter, "metrics": v_metrics}]
    print(f"[train] VotingEnsemble: RMSE={v_metrics['rmse']:.2f} | OOF={v_metrics['oof_rmse']:.2f} | IoA={v_metrics.get('ioa', 0):.3f}")

    # Stacking ensemble — MultiOutputRegressor wrapping a Ridge-meta stacker
    try:
        from sklearn.linear_model import Ridge as _Ridge
        estimators = [(r["name"], _FirstOutputWrapper(r["model"])) for r in top3]
        stacker = MultiOutputRegressor(
            StackingRegressor(
                estimators=estimators,
                final_estimator=_Ridge(alpha=1.0),
                cv=3,
            ),
            n_jobs=1,
        )
        stacker.fit(X_train, Y_train)
        stack_preds = stacker.predict(X_test).ravel()
        st_metrics  = evaluate_all(Y_test.ravel(), stack_preds, y_persistence)
        st_metrics["oof_rmse"] = _oof_rmse(stacker, X_train, Y_train)
        train_preds_s = stacker.predict(X_train).ravel()
        st_metrics["train_rmse"]  = float(np.sqrt(mean_squared_error(Y_train.ravel(), train_preds_s)))
        st_metrics["overfit_flag"] = int(overfitting_flag(st_metrics["train_rmse"], st_metrics["rmse"]))
        ensemble_results.append({"name": "StackingEnsemble", "model": stacker, "metrics": st_metrics})
        print(f"[train] StackingEnsemble: RMSE={st_metrics['rmse']:.2f} | OOF={st_metrics['oof_rmse']:.2f} | IoA={st_metrics.get('ioa', 0):.3f}")
    except Exception as e:
        print(f"[train] StackingEnsemble skipped: {e}")

    return ensemble_results


def compute_shap(best_model, X_test, feature_cols):
    """Compute SHAP values for the best tree-based model."""
    try:
        inner = best_model.estimators_[0] if hasattr(best_model, "estimators_") else best_model
        explainer = shap.TreeExplainer(inner)
        sv = explainer.shap_values(X_test[:200])
        with open(SHAP_PATH, "wb") as f:
            pickle.dump({"shap_values": sv, "feature_names": feature_cols}, f)
        print(f"[train] SHAP values saved to {SHAP_PATH}")
    except Exception as e:
        print(f"[train] SHAP skipped: {e}")


# ─── Ablation Feedback ────────────────────────────────────────────────────────

ABLATION_DISABLE_THRESHOLD = -0.5   # drop a group if removing it improves RMSE by >0.5
ABLATION_REENABLE_THRESHOLD = 0.0   # re-enable a group if it's no longer harmful

_OVERRIDE_PATH = Path(__file__).parent.parent.parent / "feature_groups_override.json"

def _persist_ablation_overrides(deltas: dict):
    from src.config import FEATURE_GROUPS
    from src.db import get_db

    # Load current overrides from MongoDB
    db = get_db()
    doc = db["feature_overrides"].find_one({"_id": "current"}) or {}
    overrides = {k: v for k, v in doc.items() if k != "_id"}

    changed = []
    for group, delta in deltas.items():
        if delta < ABLATION_DISABLE_THRESHOLD and FEATURE_GROUPS.get(group, True):
            overrides[group] = False
            changed.append(f"disabled '{group}' (Δ={delta:+.2f})")
        elif delta >= ABLATION_REENABLE_THRESHOLD and overrides.get(group) is False:
            overrides[group] = True
            changed.append(f"re-enabled '{group}' (Δ={delta:+.2f})")

    db["feature_overrides"].update_one(
        {"_id": "current"},
        {"$set": {**overrides, "updated_at": datetime.now(timezone.utc).isoformat()}},
        upsert=True,
    )

    if changed:
        print(f"[ablation] Feature groups updated for next run: {', '.join(changed)}")
    else:
        print("[ablation] No feature group changes — all groups performing as expected")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"[train] Starting training pipeline at {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")

    # 1. Load data — scaler is fit inside load_and_split on raw training data
    X_train, Y_train, X_val, Y_val, X_test, Y_test, feature_cols, df, df_raw_train = load_and_split()

    # 2. Save drift baseline from RAW (unscaled) training distribution so it
    # matches the unscaled rows written by the hourly feature pipeline.
    save_baseline(df_raw_train)

    # 3. Persistence baseline
    y_pers = persistence_baseline(Y_train, Y_test)

    # 4. Train all models
    classical_results = train_classical(X_train, Y_train, X_test, Y_test, y_pers)
    optuna_results    = train_optuna(X_train, Y_train, X_val, Y_val, X_test, Y_test, y_pers)
    lstm_result       = train_lstm(X_train, Y_train, X_val, Y_val, X_test, Y_test, y_pers)
    all_results       = classical_results + optuna_results + [lstm_result]

    # 5. Ensembles (exclude LSTM from voting — shape mismatch)
    ensemble_results  = build_ensembles(
        [r for r in all_results if r["name"] != "LSTM"],
        X_train, Y_train, X_test, Y_test, y_pers,
    )
    all_results += ensemble_results

    # WeightedVoter.fit inside _oof_rmse mutates the constituent model objects
    # (Ridge, XGBoost, etc.) in-place, leaving them fit on an OOF subset instead
    # of the full training set.  Refit every sklearn model here so the state
    # saved to GridFS by register_all is always fit on 100% of X_train.
    for r in classical_results + optuna_results:
        if hasattr(r["model"], "fit"):
            r["model"].fit(X_train, Y_train)

    # 6. Knowledge distillation — prefer LightGBM over CatBoost when available
    _DISTILL_CANDIDATES = ("RandomForest", "XGBoost", "LightGBM", "CatBoost")
    top3 = sorted(
        [r for r in all_results if r["name"] in _DISTILL_CANDIDATES],
        key=lambda r: r["metrics"]["rmse"],
    )[:3]
    if len(top3) == 3:
        student = distill(
            teachers=[r["model"] for r in top3],
            teacher_rmses=[r["metrics"]["rmse"] for r in top3],
            X_train=X_train, y_train=Y_train,
            X_val=X_val, y_val=Y_val,
        )

        wrapped_student = KerasWrapper(student)
        test_preds_s    = wrapped_student.predict(X_test).ravel()
        s_metrics       = evaluate_all(Y_test.ravel(), test_preds_s, y_pers)
        s_metrics["oof_rmse"] = s_metrics["rmse"]
        all_results.append({"name": "DistilledMLP", "model": wrapped_student, "metrics": s_metrics})
        print(f"[train] DistilledMLP: RMSE={s_metrics['rmse']:.2f}")

    # 7. Conformal prediction calibration (on primary target aqi_24h)
    best_non_lstm = sorted(
        [r for r in all_results if r["name"] not in ("LSTM", "DistilledMLP")],
        key=lambda r: r["metrics"]["rmse"],
    )[0]

    try:
        calibrate(_FirstOutputWrapper(best_non_lstm["model"]), X_val, Y_val[:, 0])
    except Exception as e:
        print(f"[train] Conformal calibration skipped: {e}")

    # 8. SHAP — requires a tree-based model
    _tree_names = ("RandomForest", "XGBoost", "CatBoost", "GradientBoosting")
    tree_results = [r for r in all_results if r["name"] in _tree_names]
    if tree_results:
        best_tree = sorted(tree_results, key=lambda r: r["metrics"]["rmse"])[0]
        compute_shap(best_tree["model"], X_test, feature_cols)

    # 9. Feature ablation — results feed back into next run's FEATURE_GROUPS
    ablation_deltas = run_ablation(df)
    if ablation_deltas:
        _persist_ablation_overrides(ablation_deltas)

    # 10. Register all + Champion-Challenger
    # Keras-based models (LSTM, DistilledMLP) are registered for leaderboard
    # comparison but excluded from champion promotion — the dashboard runs on
    # Render without TensorFlow, so the champion must be sklearn-compatible.
    KERAS_MODELS = {"LSTM", "DistilledMLP"}
    sklearn_results = [r for r in all_results if r["name"] not in KERAS_MODELS]
    keras_results   = [r for r in all_results if r["name"] in KERAS_MODELS]

    # Register Keras models with a flag so they appear in leaderboard
    for r in keras_results:
        r["metrics"]["champion_eligible"] = 0  # informational only
    register_all(sklearn_results + keras_results, champion_eligible_names={r["name"] for r in sklearn_results})

    # 11. Save champion OOF predictions for hindcast chart
    try:
        from src.db import get_db as _get_db
        _champ_meta = _get_db()["model_metadata"].find_one(
            {"model_name": "aqi_champion"}, {"promoted_from": 1}
        )
        _champ_name = (_champ_meta or {}).get("promoted_from")
        _champ_result = next((r for r in sklearn_results if r["name"] == _champ_name), None)
        if _champ_result:
            n_train = len(X_train)
            n_val   = len(X_val)
            _save_champion_oof_predictions(
                _champ_result["model"], X_train, Y_train, df.iloc[:n_train],
                X_val=X_val, Y_val=Y_val, df_val=df.iloc[n_train:n_train + n_val].reset_index(drop=True),
                X_test=X_test, Y_test=Y_test, df_test=df.iloc[n_train + n_val:].reset_index(drop=True),
            )
        else:
            print(f"[train] OOF save skipped — champion '{_champ_name}' not found in results")
    except Exception as e:
        print(f"[train] OOF save failed (non-fatal): {e}")

    # Summary
    print(f"\n{'='*60}")
    print("[train] Model comparison (sorted by test RMSE):")
    for r in sorted(all_results, key=lambda r: r["metrics"]["rmse"]):
        print(f"  {r['name']:20s}  RMSE={r['metrics']['rmse']:.2f}  IoA={r['metrics'].get('ioa', 0):.3f}  "
              f"OOF={r['metrics'].get('oof_rmse', 0):.2f}  Overfit={bool(r['metrics'].get('overfit_flag', 0))}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
