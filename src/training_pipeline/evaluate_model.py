"""
Evaluation metrics for AQI regression models.

Metrics:
  - RMSE  : Root Mean Squared Error (primary ranking metric)
  - MAE   : Mean Absolute Error
  - R²    : Coefficient of Determination
  - IoA   : Index of Agreement (domain standard for environmental forecasting)
  - Skill : Skill Score vs persistence baseline (must be > 0)
  - OOF   : Out-of-fold RMSE (leakage-proof, primary model selection criterion)
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def index_of_agreement(y_true, y_pred) -> float:
    """
    Index of Agreement (Willmott 1981).
    IoA = 1 − Σ(obs−pred)² / Σ(|pred−mean| + |obs−mean|)²
    Range: 0 (no agreement) to 1 (perfect agreement)
    """
    obs  = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    obs_mean = obs.mean()
    numerator   = np.sum((obs - pred) ** 2)
    denominator = np.sum((np.abs(pred - obs_mean) + np.abs(obs - obs_mean)) ** 2)
    if denominator < 1e-10:
        return 1.0
    return float(1.0 - numerator / denominator)


def skill_score(y_true, y_pred, y_persistence) -> float:
    """
    Skill Score = 1 − (RMSE_model / RMSE_persistence).
    > 0 means model beats persistence baseline.
    = 1 means perfect forecast.
    < 0 means worse than just repeating today's value.
    """
    rmse_model = rmse(y_true, y_pred)
    rmse_pers  = rmse(y_true, y_persistence)
    if rmse_pers < 1e-10:
        return 0.0
    return float(1.0 - rmse_model / rmse_pers)


def evaluate_all(y_true, y_pred, y_persistence=None, prefix: str = "") -> dict:
    """Compute all metrics. y_* can be 1D or 2D (multi-output averaged)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    metrics = {
        f"{prefix}rmse": rmse(y_true, y_pred),
        f"{prefix}mae":  mae(y_true, y_pred),
        f"{prefix}r2":   r2(y_true, y_pred),
        f"{prefix}ioa":  index_of_agreement(y_true, y_pred),
    }
    if y_persistence is not None:
        y_pers = np.asarray(y_persistence, dtype=float).ravel()
        metrics[f"{prefix}skill_score"] = skill_score(y_true, y_pred, y_pers)
    return metrics


def overfitting_flag(train_rmse: float, val_rmse: float) -> bool:
    """Return True if train RMSE is suspiciously much lower than val RMSE."""
    if val_rmse < 1e-10:
        return False
    return (train_rmse / val_rmse) < 0.7
