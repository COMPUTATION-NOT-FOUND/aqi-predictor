"""
Evaluation metrics for AQI regression models.

Metrics
-------
RMSE       : Root Mean Squared Error — primary accuracy metric
MAE        : Mean Absolute Error — robust to outliers
R²         : Coefficient of Determination
IoA        : Index of Agreement (Willmott 1981) — domain standard for
             environmental / air-quality forecasting; range 0–1
Skill Score: 1 − RMSE_model/RMSE_persistence — > 0 means beats persistence
OOF RMSE   : Out-of-fold RMSE (leakage-proof) — primary model-selection criterion

Champion Selection — TOPSIS
---------------------------
TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) is
the standard MCDM method used in air-quality model evaluation literature
(FAIRMODE benchmarks, academic studies on ML-based AQI forecasting).

It ranks models by their geometric distance to the ideal-best point vs the
ideal-worst point across all metrics, without requiring hand-tuned weights.

References
----------
Willmott (1981). On the validation of models. Physical Geography 2(2).
FAIRMODE (2014). Forum for Air quality Modelling in Europe — MQI guidance.
Hwang & Yoon (1981). Multiple Attribute Decision Making. Springer.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ─── Point metrics ────────────────────────────────────────────────────────────

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

    Range: 0 (no agreement) to 1 (perfect agreement).
    Advantages over R²: sensitive to proportional and additive differences;
    does not treat additive and proportional errors as equivalent.
    """
    obs  = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    obs_mean    = obs.mean()
    numerator   = np.sum((obs - pred) ** 2)
    denominator = np.sum((np.abs(pred - obs_mean) + np.abs(obs - obs_mean)) ** 2)
    if denominator < 1e-10:
        return 1.0
    return float(1.0 - numerator / denominator)


def skill_score(y_true, y_pred, y_persistence) -> float:
    """
    Skill Score = 1 − (RMSE_model / RMSE_persistence).

    Standard meteorological / NWP verification metric (Murphy 1988).
    > 0  → model beats persistence baseline  ← minimum bar for any forecast model
    = 1  → perfect forecast
    < 0  → worse than just repeating today's value (model should not be champion)
    """
    rmse_model = rmse(y_true, y_pred)
    rmse_pers  = rmse(y_true, y_persistence)
    if rmse_pers < 1e-10:
        return 0.0
    return float(1.0 - rmse_model / rmse_pers)


def overfitting_flag(train_rmse: float, val_rmse: float) -> bool:
    """Return True if train RMSE is suspiciously much lower than val RMSE."""
    if val_rmse < 1e-10:
        return False
    return (train_rmse / val_rmse) < 0.7


def evaluate_all(y_true, y_pred, y_persistence=None, prefix: str = "") -> dict:
    """Compute all metrics.

    y_* may be 1D or 2D (n_samples, n_horizons). Pooled metrics flatten across all
    horizons; when 2D inputs are given, per-horizon metrics (rmse_d1, mae_d1, r2_d1…)
    are also emitted so the leaderboard can show how error grows with lead time.
    """
    yt2 = np.asarray(y_true, dtype=float)
    yp2 = np.asarray(y_pred, dtype=float)

    y_true_flat = yt2.ravel()
    y_pred_flat = yp2.ravel()

    metrics = {
        f"{prefix}rmse": rmse(y_true_flat, y_pred_flat),
        f"{prefix}mae":  mae(y_true_flat, y_pred_flat),
        f"{prefix}r2":   r2(y_true_flat, y_pred_flat),
        f"{prefix}ioa":  index_of_agreement(y_true_flat, y_pred_flat),
    }

    # Per-horizon breakdown (only meaningful for aligned 2D multi-output inputs)
    if yt2.ndim == 2 and yp2.ndim == 2 and yt2.shape == yp2.shape and yt2.shape[1] > 1:
        for i in range(yt2.shape[1]):
            metrics[f"{prefix}rmse_d{i+1}"] = rmse(yt2[:, i], yp2[:, i])
            metrics[f"{prefix}mae_d{i+1}"]  = mae(yt2[:, i], yp2[:, i])
            metrics[f"{prefix}r2_d{i+1}"]   = r2(yt2[:, i], yp2[:, i])

    if y_persistence is not None:
        y_pers = np.asarray(y_persistence, dtype=float).ravel()
        metrics[f"{prefix}skill_score"] = skill_score(y_true_flat, y_pred_flat, y_pers)
    return metrics


# ─── TOPSIS champion selection ────────────────────────────────────────────────

def topsis_score(models_metrics: list[dict]) -> list[float]:
    """
    Compute TOPSIS score for each model in a head-to-head comparison.

    TOPSIS (Hwang & Yoon 1981) is the standard MCDM approach used in
    air-quality model selection literature (FAIRMODE, academic AQI papers).
    It needs no hand-tuned weights — models are ranked by geometric distance
    to the ideal-best vs ideal-worst across all metrics.

    Parameters
    ----------
    models_metrics : list of dicts, each containing at minimum:
        oof_rmse   (float) — out-of-fold RMSE (lower = better)
        ioa        (float) — Index of Agreement 0–1 (higher = better)
        skill_score(float) — vs persistence 0–1 (higher = better)
        mae        (float) — mean absolute error (lower = better)

    Returns
    -------
    list of float, same length as models_metrics.
    Each value ∈ [0, 1]; higher = better champion candidate.
    Returns equal scores [0.5, …] if only one model is passed.

    Algorithm
    ---------
    1. Build matrix X  (rows=models, cols=metrics)
    2. Normalise each column to unit vector: r_ij = x_ij / ||x_j||
    3. Apply equal weights: v_ij = (1/n_metrics) × r_ij
    4. Ideal best  A+_j = max or min depending on direction
       Ideal worst A−_j = opposite
    5. Distance to ideal: S+_i = ||v_i − A+||, S−_i = ||v_i − A−||
    6. TOPSIS score: C_i = S−_i / (S+_i + S−_i)
    """
    n = len(models_metrics)
    if n == 0:
        return []
    if n == 1:
        return [0.5]  # can't rank a single model against itself

    # Metrics and their direction (True = higher is better)
    METRICS = [
        ("oof_rmse",    False),  # lower is better
        ("ioa",         True),   # higher is better
        ("skill_score", True),   # higher is better
        ("mae",         False),  # lower is better
    ]

    # Build decision matrix (fill missing values with conservative defaults)
    X = np.zeros((n, len(METRICS)))
    for i, m in enumerate(models_metrics):
        for j, (key, higher_better) in enumerate(METRICS):
            val = m.get(key, None)
            if val is None or not np.isfinite(float(val)):
                # Conservative default: worst possible value for this direction
                X[i, j] = 0.0 if higher_better else 999.0
            else:
                X[i, j] = float(val)

    # Step 1 — unit-vector normalisation per column
    col_norms = np.sqrt((X ** 2).sum(axis=0))
    col_norms[col_norms < 1e-10] = 1.0   # avoid div-by-zero
    R = X / col_norms

    # Step 2 — equal weights (no subjective weighting)
    W = np.ones(len(METRICS)) / len(METRICS)
    V = R * W

    # Step 3 — ideal best and worst for each metric
    A_best  = np.zeros(len(METRICS))
    A_worst = np.zeros(len(METRICS))
    for j, (_, higher_better) in enumerate(METRICS):
        if higher_better:
            A_best[j]  = V[:, j].max()
            A_worst[j] = V[:, j].min()
        else:
            A_best[j]  = V[:, j].min()
            A_worst[j] = V[:, j].max()

    # Step 4 — Euclidean distance to ideal best and worst
    S_best  = np.sqrt(((V - A_best)  ** 2).sum(axis=1))
    S_worst = np.sqrt(((V - A_worst) ** 2).sum(axis=1))

    # Step 5 — TOPSIS score
    denom = S_best + S_worst
    denom[denom < 1e-10] = 1.0   # avoid div-by-zero (all models identical)
    scores = S_worst / denom

    return [float(s) for s in scores]


def topsis_score_single(model_metrics: dict, leaderboard_context: list[dict]) -> float:
    """
    Compute TOPSIS score for a single model given the current leaderboard.

    Used by register_model to score the champion against all challengers.
    The champion is appended to leaderboard_context, scored, then its score
    is extracted.

    Returns float in [0, 1].
    """
    combined = list(leaderboard_context) + [model_metrics]
    scores   = topsis_score(combined)
    return scores[-1] if scores else 0.5
