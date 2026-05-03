"""
Fetch raw pollutant data from AQICN and meteorological data from OpenWeather.
Both APIs return µg/m³ concentrations — no unit conversion needed.
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
    """Fetch current AQI + pollutant concentrations (µg/m³) from AQICN."""
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_API_KEY}"  # FILL IN: AQICN_API_KEY in .env
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        raise ValueError(f"AQICN error for city '{city}': {data.get('data')}")

    d = data["data"]
    iaqi = d.get("iaqi", {})

    return {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "city":       city,
        # AQI — we use this as a verification value; target is computed from PM2.5
        "aqi_raw":    float(d.get("aqi", 0)),
        # Raw concentrations in µg/m³ (discard AQICN's pre-computed sub-indices)
        "pm25":       float(iaqi.get("pm25", {}).get("v", 0)),
        "pm10":       float(iaqi.get("pm10", {}).get("v", 0)),
        "o3":         float(iaqi.get("o3",   {}).get("v", 0)),
        "no2":        float(iaqi.get("no2",  {}).get("v", 0)),
        "so2":        float(iaqi.get("so2",  {}).get("v", 0)),
        "co":         float(iaqi.get("co",   {}).get("v", 0)),
    }


def fetch_openweather(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> Optional[dict]:
    """Fetch current meteorological data from OpenWeather (µg/m³ for air quality)."""
    # Current weather
    weather_url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"  # FILL IN: OPENWEATHER_API_KEY in .env
    )
    w = requests.get(weather_url, timeout=10)
    w.raise_for_status()
    wd = w.json()

    return {
        "temperature":    float(wd["main"]["temp"]),
        "humidity":       float(wd["main"]["humidity"]),
        "pressure":       float(wd["main"]["pressure"]),
        "wind_speed":     float(wd["wind"]["speed"]),
        "wind_deg":       float(wd["wind"].get("deg", 0)),
        "visibility":     float(wd.get("visibility", 10000)),
        "cloud_cover":    float(wd["clouds"]["all"]),
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
    Returns a DataFrame indexed by UTC timestamp with columns matching
    the fields expected by build_feature_row().
    """
    import pandas as pd
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    cache_session = requests_cache.CachedSession(".openmeteo_cache", expire_after=-1)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.5)
    client = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      start_date,
        "end_date":        end_date,
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
    resp = responses[0]
    hourly = resp.Hourly()

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
        "visibility":       10.0,  # not available in archive API — use clear-sky default
    })
    df = df.set_index("timestamp")
    return df


def fetch_historical_aqicn(city: str, date_str: str) -> Optional[dict]:
    """
    Fetch historical AQI data for a given date via AQICN feed.
    Note: AQICN free tier returns the current value; for history use the daily
    archive endpoint if available, otherwise OpenWeather history is used in backfill.
    """
    return fetch_aqicn(city)


def fetch_historical_openweather(lat: float, lon: float, unix_start: int, unix_end: int) -> list[dict]:
    """Fetch hourly air pollution history from OpenWeather (returns list of hourly readings)."""
    url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={unix_start}&end={unix_end}"
        f"&appid={OPENWEATHER_API_KEY}"  # FILL IN: OPENWEATHER_API_KEY in .env
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    items = resp.json().get("list", [])

    rows = []
    for item in items:
        c = item["components"]
        rows.append({
            "timestamp":  datetime.fromtimestamp(item["dt"], tz=timezone.utc).isoformat(),
            "pm25":       float(c.get("pm2_5", 0)),
            "pm10":       float(c.get("pm10",  0)),
            "o3":         float(c.get("o3",    0)),
            "no2":        float(c.get("no2",   0)),
            "so2":        float(c.get("so2",   0)),
            "co":         float(c.get("co",    0)),
            "nh3":        float(c.get("nh3",   0)),
        })
    return rows


def fetch_current(city: str = DEFAULT_CITY, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> dict:
    """Merge AQICN pollutant data with OpenWeather meteorology into one raw row."""
    pollutants = fetch_aqicn(city)
    weather    = fetch_openweather(lat, lon)
    return {**pollutants, **weather}


if __name__ == "__main__":
    import json
    row = fetch_current()
    print(json.dumps(row, indent=2))
