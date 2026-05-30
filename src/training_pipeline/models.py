"""
All model definitions with their Optuna / RandomizedSearchCV search spaces.
Models are organized as (name, estimator, search_space, tuner_type).
"""
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── Model Registry ───────────────────────────────────────────────────────────

def _wrap_multi(estimator, n_jobs=-1):
    """Wrap single-output estimators for multi-output (3 targets)."""
    return MultiOutputRegressor(estimator, n_jobs=n_jobs)


CLASSICAL_MODELS = [
    {
        "name": "Ridge",
        "estimator": _wrap_multi(Ridge(), n_jobs=1),
        "tuner": "random",
        "param_dist": {"estimator__alpha": [0.001, 0.01, 0.1, 1, 10, 100]},
    },
    {
        "name": "Lasso",
        "estimator": _wrap_multi(Lasso(max_iter=5000), n_jobs=1),
        "tuner": "random",
        "param_dist": {"estimator__alpha": [0.001, 0.01, 0.1, 1, 10, 100]},
    },
    {
        "name": "ElasticNet",
        "estimator": _wrap_multi(ElasticNet(max_iter=5000), n_jobs=1),
        "tuner": "random",
        "param_dist": {
            "estimator__alpha":    [0.001, 0.01, 0.1, 1, 10],
            "estimator__l1_ratio": np.linspace(0.1, 0.9, 9).tolist(),
        },
    },
    {
        "name": "RandomForest",
        # n_jobs=1 here: RandomizedSearchCV(n_jobs=-1) already parallelises folds;
        # setting both to -1 oversubscribes the 2-core free runner and slows things down.
        "estimator": RandomForestRegressor(random_state=42, n_jobs=1),
        "tuner": "random",
        "param_dist": {
            "n_estimators":      [50, 100, 150],
            "max_depth":         [8, 12, 16, None],
            "max_features":      ["sqrt", "log2"],
            "min_samples_leaf":  [3, 5, 10],
            "min_samples_split": [5, 10, 20],
        },
    },
    {
        "name": "GradientBoosting",
        "estimator": _wrap_multi(GradientBoostingRegressor(random_state=42), n_jobs=1),
        "tuner": "random",
        "param_dist": {
            "estimator__n_estimators":    [50, 100, 150],
            "estimator__max_depth":       [3, 5, 7],
            "estimator__learning_rate":   [0.05, 0.1, 0.2],
            "estimator__min_samples_leaf":[3, 5, 10],
        },
    },
]


def xgb_suggest(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 400),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
    }


def cat_suggest(trial: optuna.Trial) -> dict:
    return {
        "iterations":          trial.suggest_int("iterations", 100, 400),
        "depth":               trial.suggest_int("depth", 3, 8),
        "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "min_data_in_leaf":    trial.suggest_int("min_data_in_leaf", 1, 50),
    }


def lgbm_suggest(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 200),
        "max_depth":         trial.suggest_int("max_depth", -1, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }


OPTUNA_MODELS = [
    {
        "name":    "XGBoost",
        "factory": lambda params: MultiOutputRegressor(XGBRegressor(
            objective="reg:squarederror", random_state=42,
            verbosity=0, **params,
        ), n_jobs=1),
        "suggest": xgb_suggest,
        "n_trials": 8,
    },
    {
        "name":    "CatBoost",
        "factory": lambda params: MultiOutputRegressor(CatBoostRegressor(
            loss_function="Huber:delta=1", random_seed=42,
            verbose=0, **params,
        ), n_jobs=1),
        "suggest": cat_suggest,
        "n_trials": 5,
    },
    {
        "name":    "LightGBM",
        "factory": lambda params: MultiOutputRegressor(LGBMRegressor(
            objective="huber", alpha=0.9, random_state=42,
            verbosity=-1, **params,
        ), n_jobs=1),
        "suggest": lgbm_suggest,
        "n_trials": 8,
    },
]


# ─── Keras Models (leaderboard-only; champion-ineligible) ─────────────────────

def build_lstm(params: dict, n_features: int, n_outputs: int):
    """Cheap cuDNN-compatible LSTM (no recurrent_dropout, which forces slow non-cuDNN path)."""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(params["sequence_length"], n_features)),
        tf.keras.layers.LSTM(params["units"], return_sequences=False),
        tf.keras.layers.Dropout(params["dropout"]),
        tf.keras.layers.Dense(n_outputs),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
        loss="mse",
    )
    return model


def lstm_suggest(trial) -> dict:
    return {
        "units":           trial.suggest_categorical("units", [32, 64]),
        "sequence_length": 24,
        "dropout":         trial.suggest_float("dropout", 0.1, 0.3),
        "learning_rate":   1e-3,
    }
