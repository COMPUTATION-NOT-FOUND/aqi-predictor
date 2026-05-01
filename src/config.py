import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ─────────────────────────────────────────────────────────────────
AQICN_API_KEY       = os.getenv("AQICN_API_KEY")        # FILL IN: https://aqicn.org/api/
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # FILL IN: https://openweathermap.org/api
HOPSWORKS_API_KEY   = os.getenv("HOPSWORKS_API_KEY")    # FILL IN: Hopsworks > Account Settings > API Key
HOPSWORKS_PROJECT   = os.getenv("HOPSWORKS_PROJECT", "aqi_predictor")  # FILL IN: your Hopsworks project name

# ─── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")  # FILL IN: set to remote MLflow URL if needed

# ─── City Config ──────────────────────────────────────────────────────────────
DEFAULT_CITY    = os.getenv("DEFAULT_CITY", "karachi")  # FILL IN: change to your target city
DEFAULT_LAT     = float(os.getenv("DEFAULT_LAT", "24.8607"))   # Karachi latitude
DEFAULT_LON     = float(os.getenv("DEFAULT_LON", "67.0011"))   # Karachi longitude

# ─── Pipeline Config ──────────────────────────────────────────────────────────
BACKFILL_DAYS        = int(os.getenv("BACKFILL_DAYS", "180"))   # FILL IN: history depth in days
FORECAST_HOURS       = [24, 48, 72]                             # 3-day forecast targets
FEATURE_VERSION      = 1
HOPSWORKS_FG_NAME    = "aqi_features"
HOPSWORKS_FV_NAME    = "aqi_feature_view"

# ─── Champion-Challenger ──────────────────────────────────────────────────────
PROMOTION_THRESHOLD  = 0.97  # challenger must beat champion RMSE by ≥3%
AQI_ALERT_THRESHOLD  = 150   # AQI level that triggers hazardous alert

# ─── Feature Group Toggles ────────────────────────────────────────────────────
# Set any group to False to drop it — used by ablation study and manual experiments
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
}

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
