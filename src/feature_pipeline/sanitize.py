"""Shared input validation/clamping used by BOTH the backfill seeder and the
hourly pipeline so they handle missing/erroneous data identically."""
import numpy as np

# Plausible physical ranges. Out-of-range -> clamped; missing/None -> treated as missing.
_RANGES = {
    "pm25":            (0, 1000),
    "pm10":            (0, 1200),
    "o3":              (0, 2000),
    "no2":             (0, 2000),
    "so2":             (0, 2000),
    "co":              (0, 50000),
    "nh3":             (0, 2000),
    "temperature":     (-60, 60),
    "humidity":        (0, 100),
    "pressure":        (800, 1100),
    "wind_speed":      (0, 120),
    "wind_deg":        (0, 360),
    "cloud_cover":     (0, 100),
    "precipitation_1h":(0, 500),
    "visibility":      (0, 100000),
}


def clean_raw_inputs(d: dict) -> dict:
    """Clamp raw fields to physical ranges.

    Non-finite or un-parseable values are dropped so downstream .get(k, 0)
    fills them consistently in both the backfill and hourly paths.
    """
    out = dict(d)
    for k, (lo, hi) in _RANGES.items():
        if k not in out or out[k] is None:
            continue
        try:
            v = float(out[k])
        except (TypeError, ValueError):
            out.pop(k, None)
            continue
        if not np.isfinite(v):
            out.pop(k, None)
        else:
            out[k] = float(np.clip(v, lo, hi))
    return out


def has_valid_pm25(d: dict) -> bool:
    """A real ambient PM2.5 reading is > 0. Treat <=0 / missing as an outage."""
    try:
        return float(d.get("pm25", 0)) > 0
    except (TypeError, ValueError):
        return False


def finalize_feature_row(row: dict) -> dict:
    """Final guard before insert: replace any non-finite numeric value with 0.0.

    Mirrors the np.where(isfinite, X, 0) guard applied at training time so
    the stored rows are consistent with what the model was trained on.
    Target columns (aqi_24h/48h/72h) are left as-is — None stays None.
    """
    targets = {"aqi_24h", "aqi_48h", "aqi_72h"}
    for k, v in row.items():
        if k in targets or isinstance(v, str):
            continue
        if isinstance(v, (int, float)) and not np.isfinite(v):
            row[k] = 0.0
    return row
