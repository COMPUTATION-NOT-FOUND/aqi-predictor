"""
Precompute the 3-day AQI forecast in the pipeline and store it for the dashboard.

Runs hourly (after the feature pipeline) in GitHub Actions. It:
  1. Loads the champion model + its EXACT trained feature_cols from MongoDB GridFS.
  2. Fetches current data + the Open-Meteo/CAMS forecast and builds the live feature
     row (with forecast_leads), using the same 24h-mean AQI definition as training.
  3. Reindexes the row to the stored feature_cols (no train/serve column skew),
     predicts 24/48/72h, and computes a 90% conformal interval for the 24h horizon.
  4. Writes one document to `predictions` (read by the dashboard) and one row to
     `prediction_log` (the real track record charted in Panel B).

The dashboard NEVER runs inference live — it only reads these documents. If this
module fails, the dashboard shows an explicit "forecast pending" state rather than
silent fallback values.
"""
import io
import json
import pickle
import zipfile
import sys, os
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import DEFAULT_CITY, DEFAULT_LAT, DEFAULT_LON, FORECAST_HOURS
from src.db import get_db, get_gridfs
from src.feature_pipeline.fetch_data import fetch_current, fetch_forecast
from src.feature_pipeline.feature_engineering import (
    build_feature_row, apply_scaler, pm25_to_aqi, pm25_mean_to_aqi,
)
from src.feature_pipeline.store_features import fetch_recent
from src.training_pipeline.wrappers import FirstOutputWrapper  # noqa: F401 — for calibrator unpickling

_HISTORY_ROWS = 200   # cover the 168h lag window plus headroom


# ─── Champion loading ─────────────────────────────────────────────────────────

def load_champion():
    """Return (model, scaler, feature_cols, calibrator, version) from GridFS, or Nones."""
    meta = get_db()["model_metadata"].find_one({"model_name": "aqi_champion"})
    if not meta or not meta.get("artifacts_gridfs_id"):
        print("[predict] No champion artifacts in MongoDB.")
        return None, None, None, None, None

    version = meta.get("trained_at", "unknown")
    feature_cols = meta.get("feature_cols")   # may also live in the ZIP
    model = scaler = calibrator = None

    zip_bytes = get_gridfs().get(meta["artifacts_gridfs_id"]).read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        model_type = zf.read("model_type.txt").decode().strip() if "model_type.txt" in names else "sklearn"
        if model_type == "keras":
            # Champion should never be Keras (excluded from eligibility), but guard anyway.
            print("[predict] Champion is Keras — not supported in the precompute step.")
            return None, None, None, None, None
        model = pickle.loads(zf.read("model.pkl"))
        if "scaler_bundle.pkl" in names:
            scaler = pickle.loads(zf.read("scaler_bundle.pkl"))
        if feature_cols is None and "feature_cols.json" in names:
            feature_cols = json.loads(zf.read("feature_cols.json").decode())
        if "conformal_calibrator.pkl" in names:
            try:
                calibrator = pickle.loads(zf.read("conformal_calibrator.pkl"))
            except Exception as e:
                print(f"[predict] Could not load conformal calibrator: {e}")

    return model, scaler, feature_cols, calibrator, version


# ─── Prediction ───────────────────────────────────────────────────────────────

def run(city: str = DEFAULT_CITY, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> dict | None:
    model, scaler, feature_cols, calibrator, version = load_champion()
    if model is None or not feature_cols:
        print("[predict] Aborting — no usable champion model / feature_cols.")
        return None

    now = datetime.now(timezone.utc)

    # 1. Inputs: history (for lags + 24h-mean AQI), current observation, forecast.
    history = fetch_recent(city, n=_HISTORY_ROWS)
    raw = fetch_current(city=city, lat=lat, lon=lon)
    forecast = fetch_forecast(lat=lat, lon=lon, horizons=tuple(FORECAST_HOURS), now=now)

    # Current AQI = 24h trailing-mean PM2.5 → AQI (matches training definition).
    pm25_window = history["pm25"].tail(23).tolist() if (not history.empty and "pm25" in history) else []
    pm25_window.append(raw.get("pm25", 0))
    current_aqi = float(pm25_mean_to_aqi(pm25_window))
    raw["aqi_from_api"] = current_aqi

    # 2. Build the feature row and align it to the champion's exact training columns.
    row = build_feature_row(raw, history, now, forecast=forecast)
    df_row = pd.DataFrame([row])
    if scaler is not None:
        df_row = apply_scaler(df_row, scaler)

    X = df_row.reindex(columns=feature_cols, fill_value=0.0).to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)

    # 3. Predict (model is multi-output: 24/48/72h).
    preds = np.asarray(model.predict(X))[0]
    preds = np.atleast_1d(preds)
    horizon_aqi = [
        float(np.clip(preds[i] if i < len(preds) else preds[-1], 0, 500))
        for i in range(len(FORECAST_HOURS))
    ]

    # 4. Conformal interval for the 24h horizon (90% coverage), if a calibrator exists.
    lower_24h, upper_24h = horizon_aqi[0] * 0.85, horizon_aqi[0] * 1.15
    if calibrator is not None:
        try:
            from src.training_pipeline.conformal import predict_with_intervals
            _, lo, hi = predict_with_intervals(calibrator, X)
            lower_24h, upper_24h = float(lo[0]), float(hi[0])
        except Exception as e:
            print(f"[predict] Conformal interval failed (non-fatal): {e}")
    lower_24h = max(0.0, lower_24h)
    upper_24h = min(500.0, upper_24h)

    # 5. Persist the forecast document for the dashboard.
    forecasts = []
    for i, h in enumerate(FORECAST_HOURS):
        forecasts.append({
            "horizon":   h,
            "target_ts": now + timedelta(hours=h),
            "aqi":       round(horizon_aqi[i], 2),
            "lower":     round(lower_24h, 2) if h == 24 else None,
            "upper":     round(upper_24h, 2) if h == 24 else None,
        })

    doc = {
        "predicted_at":  now,
        "anchor_ts":     now,
        "city":          city,
        "current_aqi":   round(current_aqi, 2),
        "forecasts":     forecasts,
        "model_version": str(version),
    }
    get_db()["predictions"].insert_one(dict(doc))

    # 6. Track-record log row (Panel B): predicted vs later-observed.
    _log_prediction(city, now, current_aqi, horizon_aqi, str(version))

    print(f"[predict] city={city} AQI={current_aqi:.0f} → "
          f"24h={horizon_aqi[0]:.0f} 48h={horizon_aqi[1]:.0f} 72h={horizon_aqi[2]:.0f} "
          f"(model {version})")
    return doc


def _log_prediction(city, now, current_aqi, horizon_aqi, version) -> None:
    """Insert one prediction_log row. TTL index on logged_at handles 30-day cleanup."""
    try:
        get_db()["prediction_log"].insert_one({
            "city":          city,
            "ts_epoch":      int(now.timestamp()),
            "logged_at":     now,
            "predicted_24h": round(float(horizon_aqi[0]), 2),
            "predicted_48h": round(float(horizon_aqi[1]), 2),
            "predicted_72h": round(float(horizon_aqi[2]), 2),
            "observed_aqi":  round(float(current_aqi), 2),
            "model_version": str(version),
        })
    except Exception as e:
        print(f"[predict] prediction_log insert failed (non-fatal): {e}")


if __name__ == "__main__":
    run()
