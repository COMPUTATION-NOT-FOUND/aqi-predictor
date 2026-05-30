import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ─────────────────────────────────────────────────────────────────
AQICN_API_KEY       = os.getenv("AQICN_API_KEY")        # FILL IN: https://aqicn.org/api/
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # FILL IN: https://openweathermap.org/api

# ─── MongoDB ───────────────────────────────────────────────────────────────────
MONGODB_URI     = os.getenv("MONGODB_URI")               # FILL IN: Atlas SRV connection string
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "aqi_db")

# ─── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")  # FILL IN: set to remote MLflow URL if needed

# ─── City Config ──────────────────────────────────────────────────────────────
DEFAULT_CITY    = os.getenv("DEFAULT_CITY", "karachi")  # FILL IN: change to your target city
DEFAULT_LAT     = float(os.getenv("DEFAULT_LAT", "24.8607"))   # Karachi latitude
DEFAULT_LON     = float(os.getenv("DEFAULT_LON", "67.0011"))   # Karachi longitude

# ─── Pipeline Config ──────────────────────────────────────────────────────────
BACKFILL_DAYS        = int(os.getenv("BACKFILL_DAYS", "365"))   # one seasonal cycle; set BACKFILL_DAYS=730 env var for one-time 2yr re-seed
FORECAST_HOURS       = [24, 48, 72]                             # 3-day forecast targets

# ─── Champion-Challenger ──────────────────────────────────────────────────────
AQI_ALERT_THRESHOLD  = 150   # AQI level that triggers hazardous alert

# ─── Feature Group Toggles ────────────────────────────────────────────────────
# Set any group to False to drop it — used by ablation study and manual experiments
_OVERRIDE_PATH = Path(__file__).parent.parent / "feature_groups_override.json"

FEATURE_GROUPS = {
    "time_basic":      True,   # hour, day_of_week, month, season, is_weekend
    "time_fourier":    True,   # hour_sin, hour_cos, month_sin, month_cos
    "lag_features":    True,   # aqi_lag_1h … aqi_lag_24h, pm25_lag_*
    "rolling_stats":   True,   # rolling_mean_3h/6h/24h, rolling_std_24h
    "physics_derived": True,   # mixing_height_proxy, wind_transport, temp_humidity
    "cross_feature":   True,   # pressure_anomaly, pm_ratio
    "stl_decomp":      True,   # aqi_trend, aqi_seasonal, aqi_residual
    "raw_pollutants":  True,   # PM2.5, PM10, O3, NO2, SO2, CO (keep True)
    "meteorology":     True,   # temperature, humidity, wind_speed, pressure, etc.
    "forecast_leads":  True,   # forecasted weather + PM2.5 at t+24/48/72h (Open-Meteo/CAMS)
}

# Apply persisted ablation overrides — MongoDB is the source of truth so the
# selection survives across GitHub Actions runs.  Fall back to the local JSON
# file for offline development when MONGODB_URI is not set.
def _load_feature_overrides() -> dict:
    try:
        from src.db import get_db
        doc = get_db()["feature_overrides"].find_one({"_id": "current"}) or {}
        return {k: v for k, v in doc.items() if k not in ("_id", "updated_at")}
    except Exception:
        pass
    if _OVERRIDE_PATH.exists():
        with open(_OVERRIDE_PATH) as _f:
            return json.load(_f)
    return {}

FEATURE_GROUPS.update(_load_feature_overrides())

# ─── Drift Thresholds ─────────────────────────────────────────────────────────
PSI_WARN_THRESHOLD  = 0.1   # moderate drift
PSI_ALERT_THRESHOLD = 0.2   # trigger retraining alert

# ─── AQI Color Zones (US EPA) ─────────────────────────────────────────────────
AQI_ZONES = [
    (0,   50,  "Good",                        "#00e400"),
    (51,  100, "Moderate",                    "#ffff00"),
    (101, 150, "Unhealthy for Sensitive",      "#ff7e00"),
    (151, 200, "Unhealthy",                    "#ff0000"),
    (201, 300, "Very Unhealthy",               "#8f3f97"),
    (301, 500, "Hazardous",                    "#7e0023"),
]

def aqi_color(aqi: float) -> str:
    for lo, hi, _, color in AQI_ZONES:
        if lo <= aqi <= hi:
            return color
    return "#7e0023"

def aqi_label(aqi: float) -> str:
    for lo, hi, label, _ in AQI_ZONES:
        if lo <= aqi <= hi:
            return label
    return "Hazardous"
