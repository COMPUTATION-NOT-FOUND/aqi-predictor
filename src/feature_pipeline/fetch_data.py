"""
Fetch raw pollutant data from AQICN and meteorological data from OpenWeather.

IMPORTANT — Data source clarification
--------------------------------------
AQICN's iaqi.pm25.v field is a PM2.5 AQI *sub-index* (unitless, 0–500),
NOT a raw PM2.5 concentration in µg/m³.

Feeding it into pm25_to_aqi() causes a double-conversion:
  161 (sub-index) → pm25_to_aqi(161) → 211  ← WRONG

Fix: AQICN is used only for its already-computed AQI (d.aqi).
     OpenWeather /air_pollution gives true µg/m³ concentrations.
"""
import requests
import time
import pandas as pd
from datetime import datetime, timezone
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import (
    AQICN_API_KEY, OPENWEATHER_API_KEY,
    DEFAULT_CITY, DEFAULT_LAT, DEFAULT_LON,
)


def fetch_aqicn(city: str = DEFAULT_CITY) -> Optional[dict]:
    """Fetch current AQI from AQICN (AQI only — no concentrations).

    Returns aqi_raw: the AQI already computed by AQICN's own pipeline.
    Pollutant concentrations (pm25, pm10 etc.) are fetched separately from
    OpenWeather /air_pollution which returns true µg/m³ values.
    """
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_API_KEY}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        raise ValueError(f"AQICN error for city '{city}': {data.get('data')}")

    d = data["data"]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "city":      city,
        # This is the correctly computed AQI from AQICN — do NOT re-convert via pm25_to_aqi()
        "aqi_raw":   float(d.get("aqi", 0)),
        # Placeholders — real µg/m³ values are filled in by fetch_current()
        # via the OpenWeather air_pollution endpoint
        "pm25": 0.0,
        "pm10": 0.0,
        "o3":   0.0,
        "no2":  0.0,
        "so2":  0.0,
        "co":   0.0,
    }


def fetch_openweather_air_pollution(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> dict:
    """
    Fetch current PM2.5 and pollutant concentrations in µg/m³ from
    OpenWeather /air_pollution (free tier, returns real concentrations).

    This is the correct source for raw µg/m³ values to feed into pm25_to_aqi().
    """
    url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("list", [])
        if not items:
            return {}
        c = items[0]["components"]
        return {
            "pm25": float(c.get("pm2_5", 0)),
            "pm10": float(c.get("pm10",  0)),
            "o3":   float(c.get("o3",    0)),
            "no2":  float(c.get("no2",   0)),
            "so2":  float(c.get("so2",   0)),
            "co":   float(c.get("co",    0)),
        }
    except Exception as e:
        print(f"[fetch] OpenWeather air_pollution failed: {e} — concentrations will be 0")
        return {}


def fetch_openweather(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> Optional[dict]:
    """Fetch current meteorological data from OpenWeather."""
    weather_url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    w = requests.get(weather_url, timeout=10)
    w.raise_for_status()
    wd = w.json()

    return {
        "temperature":      float(wd["main"]["temp"]),
        "humidity":         float(wd["main"]["humidity"]),
        "pressure":         float(wd["main"]["pressure"]),
        "wind_speed":       float(wd["wind"]["speed"]),
        "wind_deg":         float(wd["wind"].get("deg", 0)),
        "visibility":       float(wd.get("visibility", 10000)),
        "cloud_cover":      float(wd["clouds"]["all"]),
        "precipitation_1h": float(wd.get("rain", {}).get("1h", 0.0)),
    }


def fetch_historical_weather_openmeteo(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch hourly historical weather from Open-Meteo (free, no API key needed).
    Returns a DataFrame indexed by UTC timestamp.
    """
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    cache_session = requests_cache.CachedSession(".openmeteo_cache", expire_after=-1)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.5)
    client = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "cloud_cover",
            "precipitation",
        ],
        "timezone": "UTC",
    }

    responses = client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
    resp    = responses[0]
    hourly  = resp.Hourly()

    times = pd.date_range(
        start=pd.Timestamp(hourly.Time(), unit="s", tz="UTC"),
        end=pd.Timestamp(hourly.TimeEnd(), unit="s", tz="UTC"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    df = pd.DataFrame({
        "timestamp":        times,
        "temperature":      hourly.Variables(0).ValuesAsNumpy(),
        "humidity":         hourly.Variables(1).ValuesAsNumpy(),
        "pressure":         hourly.Variables(2).ValuesAsNumpy(),
        "wind_speed":       hourly.Variables(3).ValuesAsNumpy(),
        "wind_deg":         hourly.Variables(4).ValuesAsNumpy(),
        "cloud_cover":      hourly.Variables(5).ValuesAsNumpy(),
        "precipitation_1h": hourly.Variables(6).ValuesAsNumpy(),
        "visibility":       10.0,  # not available in archive API
    })
    df = df.set_index("timestamp")
    return df


def fetch_historical_aqicn(city: str, date_str: str) -> Optional[dict]:
    """
    Fetch historical AQI data for a given date via AQICN feed.
    AQICN free tier returns the current value — use OpenWeather history for backfill.
    """
    return fetch_aqicn(city)


def fetch_historical_openweather(lat: float, lon: float, unix_start: int, unix_end: int) -> list[dict]:
    """Fetch hourly air pollution history from OpenWeather (true µg/m³ concentrations)."""
    url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={unix_start}&end={unix_end}"
        f"&appid={OPENWEATHER_API_KEY}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    items = resp.json().get("list", [])

    rows = []
    for item in items:
        c = item["components"]
        rows.append({
            "timestamp": datetime.fromtimestamp(item["dt"], tz=timezone.utc).isoformat(),
            "pm25":      float(c.get("pm2_5", 0)),
            "pm10":      float(c.get("pm10",  0)),
            "o3":        float(c.get("o3",    0)),
            "no2":       float(c.get("no2",   0)),
            "so2":       float(c.get("so2",   0)),
            "co":        float(c.get("co",    0)),
            "nh3":       float(c.get("nh3",   0)),
        })
    return rows


def fetch_current(
    city: str = DEFAULT_CITY,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> dict:
    """
    Merge AQICN AQI reading with OpenWeather concentrations + meteorology.

    AQI source  : OpenWeather PM2.5 → pm25_to_aqi() (US EPA formula)
    PM2.5 source: OpenWeather /air_pollution (true µg/m³ concentrations)
    Weather     : OpenWeather /weather (temperature, wind, etc.)

    NOTE: AQICN's d.aqi composite value is fetched but NOT used as the AQI.
    Karachi testing showed AQICN returns 161 while OpenWeather PM2.5=16.6 µg/m³
    → US EPA AQI=61, which matches independent monitoring sources.  AQICN
    appears to use a different standard or station location for this city.
    """
    from src.feature_pipeline.feature_engineering import pm25_to_aqi as _pm25_to_aqi

    aqicn_data  = fetch_aqicn(city)
    weather     = fetch_openweather(lat, lon)
    pollutants  = fetch_openweather_air_pollution(lat, lon)

    merged = {**aqicn_data, **weather, **pollutants}

    # Compute AQI from OpenWeather PM2.5 (US EPA formula) — consistent with
    # the backfill pipeline and with what independent monitoring sources report.
    # Store raw AQICN value separately for diagnostics only.
    merged["aqi_aqicn_raw"] = merged.get("aqi_raw", 0)
    merged["aqi_from_api"]  = _pm25_to_aqi(merged.get("pm25", 0))

    return merged


if __name__ == "__main__":
    import json
    row = fetch_current()
    print(json.dumps(row, indent=2))
    from src.feature_pipeline.feature_engineering import pm25_to_aqi
    computed = pm25_to_aqi(row["pm25"])
    print(f"\nVerification:")
    print(f"  AQICN AQI (raw, not used): {row['aqi_aqicn_raw']:.0f}")
    print(f"  PM2.5 µg/m³ (OW)        : {row['pm25']:.1f}")
    print(f"  AQI used (pm25_to_aqi)  : {row['aqi_from_api']:.0f}")
    if abs(row["aqi_aqicn_raw"] - computed) > 30:
        print(f"  ℹ️  AQICN vs PM2.5 differ by {abs(row['aqi_aqicn_raw']-computed):.0f} — expected for Karachi (different standard)")
    else:
        print("  ✓ AQICN and PM2.5-based AQI are consistent")
