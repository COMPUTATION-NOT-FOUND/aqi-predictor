"""
Quick connection test — run once to verify all API keys and dependencies work.
Usage: python verify_setup.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results = []

# ── 1. AQICN API ──────────────────────────────────────────────────────────────
print("\n── 1. AQICN API ──────────────────────────────────────────────────────")
try:
    import requests
    key = os.getenv("AQICN_API_KEY")
    r = requests.get(f"https://api.waqi.info/feed/karachi/?token={key}", timeout=10)
    data = r.json()
    if data.get("status") == "ok":
        aqi = data["data"]["aqi"]
        print(f"{PASS} AQICN connected — current Karachi AQI: {aqi}")
        results.append(("AQICN", True))
    else:
        print(f"{FAIL} AQICN error: {data}")
        results.append(("AQICN", False))
except Exception as e:
    print(f"{FAIL} AQICN exception: {e}")
    results.append(("AQICN", False))

# ── 2. OpenWeather API ────────────────────────────────────────────────────────
print("\n── 2. OpenWeather API ───────────────────────────────────────────────")
try:
    key = os.getenv("OPENWEATHER_API_KEY")
    lat, lon = os.getenv("DEFAULT_LAT", "24.8607"), os.getenv("DEFAULT_LON", "67.0011")
    r = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric",
        timeout=10
    )
    data = r.json()
    if r.status_code == 200:
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        print(f"{PASS} OpenWeather connected — Karachi: {temp}°C, humidity {humidity}%")
        results.append(("OpenWeather", True))
    else:
        print(f"{FAIL} OpenWeather error {r.status_code}: {data.get('message')}")
        results.append(("OpenWeather", False))
except Exception as e:
    print(f"{FAIL} OpenWeather exception: {e}")
    results.append(("OpenWeather", False))

# ── 3. Hopsworks ──────────────────────────────────────────────────────────────
print("\n── 3. Hopsworks Feature Store ───────────────────────────────────────")
try:
    import hopsworks
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT", "aqi_predictor"),
    )
    fs = project.get_feature_store()
    print(f"{PASS} Hopsworks connected — project: {project.name}")
    results.append(("Hopsworks", True))
except Exception as e:
    print(f"{FAIL} Hopsworks exception: {e}")
    results.append(("Hopsworks", False))

# ── 4. Key Python packages ────────────────────────────────────────────────────
print("\n── 4. Key Python Packages ───────────────────────────────────────────")
packages = [
    ("pandas", "pd"),
    ("numpy", "np"),
    ("sklearn", "sklearn"),
    ("xgboost", "xgb"),
    ("lightgbm", "lgb"),
    ("catboost", "cb"),
    ("tensorflow", "tf"),
    ("shap", "shap"),
    ("mapie", "mapie"),
    ("mlflow", "mlflow"),
    ("dash", "dash"),
    ("plotly", "plotly"),
    ("mrmr", "mrmr"),
    ("statsmodels", "sm"),
    ("optuna", "optuna"),
]
all_pkg_ok = True
for pkg, alias in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        print(f"  {PASS} {pkg} {ver}")
    except ImportError:
        print(f"  {FAIL} {pkg} not installed")
        all_pkg_ok = False
results.append(("Packages", all_pkg_ok))

# ── 5. Feature engineering smoke test ────────────────────────────────────────
print("\n── 5. Feature Engineering Smoke Test ───────────────────────────────")
try:
    from src.feature_pipeline.feature_engineering import pm25_to_aqi, build_feature_row
    aqi_val = pm25_to_aqi(35.4)
    print(f"{PASS} pm25_to_aqi(35.4) = {aqi_val}  (expected ~100)")

    # Minimal dummy row
    dummy = {
        "pm25": 35.4, "pm10": 70.0, "o3": 40.0, "no2": 20.0, "so2": 5.0, "co": 0.5,
        "temperature": 30.0, "humidity": 60.0, "wind_speed": 3.0, "wind_deg": 180.0,
        "pressure": 1010.0, "visibility": 10000, "cloud_cover": 20, "precipitation_1h": 0.0,
        "timestamp": "2026-05-01T12:00:00",
    }
    row = build_feature_row(dummy)
    print(f"{PASS} build_feature_row returned {len(row)} features")
    results.append(("FeatureEng", True))
except Exception as e:
    print(f"{FAIL} Feature engineering error: {e}")
    results.append(("FeatureEng", False))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("SUMMARY")
print("═" * 60)
all_ok = True
for name, ok in results:
    status = PASS if ok else FAIL
    print(f"  {status} {name}")
    if not ok:
        all_ok = False

if all_ok:
    print(f"\n{PASS} All checks passed — ready to run the backfill!")
    print(f"{INFO} Next step: python src/backfill/backfill_features.py --strategy hybrid")
else:
    print(f"\n{FAIL} Some checks failed — fix the above before running the pipeline.")
print()
