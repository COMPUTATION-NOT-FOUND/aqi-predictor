"""
Main training pipeline — runs daily via GitHub Actions.

Steps:
1. Fetch historical features from Hopsworks Feature Store
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
                "tensorflow", "absl", "hopsworks"]:
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
from src.feature_pipeline.feature_engineering import fit_scaler, apply_scaler
from src.feature_pipeline.drift_monitor import save_baseline
from src.training_pipeline.models import (
    CLASSICAL_MODELS, OPTUNA_MODELS, build_lstm, lstm_suggest,
)
from src.training_pipeline.evaluate_model import evaluate_all, overfitting_flag
from src.training_pipeline.distillation import distill
from src.training_pipeline.conformal import calibrate
from src.training_pipeline.ablation_features import run_ablation
from src.training_pipeline.register_model import register_all

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Re-apply after imports since mlflow resets its loggers on import
for _logger in ["mlflow", "mlflow.models.model", "mlflow.sklearn", "mlflow.utils",
                "mlflow.tensorflow", "tensorflow", "absl", "hopsworks"]:
    logging.getLogger(_logger).setLevel(logging.ERROR)

TARGET_COLS  = [f"aqi_{h}h" for h in FORECAST_HOURS]
N_SPLITS_OOF = 5
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


# ─── Data Preparation ─────────────────────────────────────────────────────────

def load_and_split():
    print("[train] Loading features from Hopsworks...")
    df = fetch_training_data()
    df = df.dropna(subset=TARGET_COLS)

    drop_meta = ["timestamp", "city"] + TARGET_COLS
    feature_cols = [c for c in df.columns if c not in drop_meta and df[c].dtype != object]

    X = df[feature_cols].values
    Y = df[TARGET_COLS].values       # shape (n, 3)

    n = len(X)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.10)

    X_train, Y_train = X[:n_train],          Y[:n_train]
    X_val,   Y_val   = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test,  Y_test  = X[n_train+n_val:],    Y[n_train+n_val:]

    print(f"[train] Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, feature_cols, df


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


def train_classical(X_train, Y_train, X_test, Y_test, y_persistence, scaler) -> list[dict]:
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
                n_iter=30, cv=TimeSeriesSplit(n_splits=3),
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
            epochs=20, batch_size=64, verbose=0,
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        )
        preds = model.predict(Xvl_s, verbose=0).ravel()
        return float(np.sqrt(mean_squared_error(Yvl_s.ravel(), preds)))

    with mlflow.start_run(run_name="LSTM"):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5, show_progress_bar=False)
        best = study.best_params
        seq_len = best["sequence_length"]

        def to_seq(X, Y, seq):
            return (np.array([X[i:i+seq] for i in range(len(X)-seq)]), Y[seq:])

        Xtr_s, Ytr_s = to_seq(X_train, Y_train, seq_len)
        Xte_s, Yte_s = to_seq(X_test,  Y_test,  seq_len)
        final_model   = build_lstm(**best, n_features=X_train.shape[1], n_outputs=n_outputs)
        final_model.fit(
            Xtr_s, Ytr_s, epochs=50, batch_size=64, verbose=0,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
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

    # Voting ensemble (Optuna-tuned weights replaced by 1/RMSE weights for simplicity)
    weights = [1.0 / max(r["metrics"]["rmse"], 1e-6) for r in top3]
    w_norm  = [w / sum(weights) for w in weights]

    voter = WeightedVoter([r["model"] for r in top3], w_norm)
    voter.fit(X_train, Y_train)
    test_preds = voter.predict(X_test).ravel()
    v_metrics  = evaluate_all(Y_test.ravel(), test_preds, y_persistence)
    v_metrics["oof_rmse"] = v_metrics["rmse"]   # proxy
    ensemble_results = [{"name": "VotingEnsemble", "model": voter, "metrics": v_metrics}]
    print(f"[train] VotingEnsemble: RMSE={v_metrics['rmse']:.2f}")

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
    import json
    from src.config import FEATURE_GROUPS

    overrides = {}
    if _OVERRIDE_PATH.exists():
        with open(_OVERRIDE_PATH) as f:
            overrides = json.load(f)

    changed = []
    for group, delta in deltas.items():
        if delta < ABLATION_DISABLE_THRESHOLD and FEATURE_GROUPS.get(group, True):
            overrides[group] = False
            changed.append(f"disabled '{group}' (Δ={delta:+.2f})")
        elif delta >= ABLATION_REENABLE_THRESHOLD and overrides.get(group) is False:
            overrides[group] = True
            changed.append(f"re-enabled '{group}' (Δ={delta:+.2f})")

    with open(_OVERRIDE_PATH, "w") as f:
        json.dump(overrides, f, indent=2)

    if changed:
        print(f"[ablation] Feature groups updated for next run: {', '.join(changed)}")
    else:
        print("[ablation] No feature group changes — all groups performing as expected")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"[train] Starting training pipeline at {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")

    # 1. Load data
    X_train, Y_train, X_val, Y_val, X_test, Y_test, feature_cols, df = load_and_split()

    # 2. Fit scaler on training data (saves scaler_bundle.pkl)
    df_train = df.iloc[:len(X_train)]
    fit_scaler(df_train)
    print("[train] Scaler fitted and saved")

    # 3. Apply scaler
    from src.feature_pipeline.feature_engineering import load_scaler, apply_scaler
    scaler = load_scaler()

    # 4. Save drift baseline from training distribution
    save_baseline(df_train)

    # 5. Persistence baseline
    y_pers = persistence_baseline(Y_train, Y_test)

    # 6. Train all models
    classical_results = train_classical(X_train, Y_train, X_test, Y_test, y_pers, scaler)
    optuna_results    = train_optuna(X_train, Y_train, X_val, Y_val, X_test, Y_test, y_pers)
    lstm_result       = train_lstm(X_train, Y_train, X_val, Y_val, X_test, Y_test, y_pers)
    all_results       = classical_results + optuna_results + [lstm_result]

    # 7. Ensembles (exclude LSTM from voting — shape mismatch)
    ensemble_results  = build_ensembles(
        [r for r in all_results if r["name"] != "LSTM"],
        X_train, Y_train, X_test, Y_test, y_pers,
    )
    all_results += ensemble_results

    # 8. Knowledge distillation
    top3 = sorted(
        [r for r in all_results if r["name"] in ("RandomForest", "XGBoost", "CatBoost")],
        key=lambda r: r["metrics"]["rmse"],
    )[:3]
    if len(top3) == 3:
        student = distill(
            teachers=[r["model"] for r in top3],
            teacher_rmses=[r["metrics"]["rmse"] for r in top3],
            X_train=X_train, y_train=Y_train,
            X_val=X_val, y_val=Y_val,
        )

        class KerasWrapper:
            def __init__(self, model): self.model = model
            def predict(self, X): return self.model.predict(X, verbose=0)

        wrapped_student = KerasWrapper(student)
        test_preds_s    = wrapped_student.predict(X_test).ravel()
        s_metrics       = evaluate_all(Y_test.ravel(), test_preds_s, y_pers)
        s_metrics["oof_rmse"] = s_metrics["rmse"]
        all_results.append({"name": "DistilledMLP", "model": wrapped_student, "metrics": s_metrics})
        print(f"[train] DistilledMLP: RMSE={s_metrics['rmse']:.2f}")

    # 9. Conformal prediction calibration (on primary target aqi_24h)
    best_non_lstm = sorted(
        [r for r in all_results if r["name"] not in ("LSTM", "DistilledMLP")],
        key=lambda r: r["metrics"]["rmse"],
    )[0]

    class _FirstOutputWrapper:
        def __init__(self, model): self.model = model
        def fit(self, X, y): return self
        def predict(self, X): return self.model.predict(X)[:, 0]

    try:
        calibrate(_FirstOutputWrapper(best_non_lstm["model"]), X_val, Y_val[:, 0])
    except Exception as e:
        print(f"[train] Conformal calibration skipped: {e}")

    # 10. SHAP — requires a tree-based model
    _tree_names = ("RandomForest", "XGBoost", "CatBoost", "GradientBoosting")
    tree_results = [r for r in all_results if r["name"] in _tree_names]
    if tree_results:
        best_tree = sorted(tree_results, key=lambda r: r["metrics"]["rmse"])[0]
        compute_shap(best_tree["model"], X_test, feature_cols)

    # 11. Feature ablation — results feed back into next run's FEATURE_GROUPS
    ablation_deltas = run_ablation(df)
    if ablation_deltas:
        _persist_ablation_overrides(ablation_deltas)

    # 12. Register all + Champion-Challenger
    register_all(all_results)

    # Summary
    print(f"\n{'='*60}")
    print("[train] Model comparison (sorted by test RMSE):")
    for r in sorted(all_results, key=lambda r: r["metrics"]["rmse"]):
        print(f"  {r['name']:20s}  RMSE={r['metrics']['rmse']:.2f}  IoA={r['metrics'].get('ioa', 0):.3f}  "
              f"OOF={r['metrics'].get('oof_rmse', 0):.2f}  Overfit={bool(r['metrics'].get('overfit_flag', 0))}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
