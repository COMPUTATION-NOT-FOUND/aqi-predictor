"""
Conformal prediction intervals using MAPIE.

Provides distribution-free prediction intervals at a guaranteed coverage level
(default 90%). The intervals are computed for the distilled student model.

Why conformal prediction over simple error bars?
  - No distributional assumptions (works for any model)
  - Coverage guarantee: 90% of true future AQI values fall within the band
  - Width adapts to input difficulty (harder predictions → wider intervals)
"""
import numpy as np
import pickle
from pathlib import Path
from mapie.regression import SplitConformalRegressor
from mapie.metrics.regression import regression_coverage_score

CONFORMAL_PATH = Path(__file__).parent.parent.parent / "conformal_calibrator.pkl"
COVERAGE_LEVEL = 0.90   # 90% coverage guarantee


def calibrate(model, X_cal: np.ndarray, y_cal: np.ndarray) -> SplitConformalRegressor:
    """
    Fit a MAPIE conformal calibrator on a held-out calibration set.
    Works with any pre-fitted sklearn-compatible model.

    Args:
        model  : any fitted sklearn-compatible regressor
        X_cal  : calibration features (held-out — never seen during training)
        y_cal  : true 24h AQI values for calibration (use primary target only)
    """
    mapie = SplitConformalRegressor(estimator=model, confidence_level=COVERAGE_LEVEL, prefit=True)
    mapie.conformalize(X_cal, y_cal)

    with open(CONFORMAL_PATH, "wb") as f:
        pickle.dump(mapie, f)

    # Verify empirical coverage
    _, intervals = mapie.predict_interval(X_cal)
    lower = intervals[:, 0, 0]
    upper = intervals[:, 1, 0]
    coverage = regression_coverage_score(y_cal, intervals[:, :, 0])
    print(f"[conformal] Calibration coverage: {coverage:.3f} (target: {COVERAGE_LEVEL})")
    return mapie


def load_calibrator() -> SplitConformalRegressor:
    if not CONFORMAL_PATH.exists():
        return None
    with open(CONFORMAL_PATH, "rb") as f:
        return pickle.load(f)


def predict_with_intervals(mapie: SplitConformalRegressor, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        point_preds  : shape (n_samples,)
        lower_bounds : shape (n_samples,)
        upper_bounds : shape (n_samples,)
    """
    preds, intervals = mapie.predict_interval(X)
    lower = intervals[:, 0, 0]
    upper = intervals[:, 1, 0]
    return preds, lower, upper
