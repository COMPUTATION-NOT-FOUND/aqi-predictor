"""
Historical backfill — seeds the Hopsworks Feature Store with past data.
Run ONCE manually before the first automated training run.

Usage:
  python src/backfill/backfill_features.py
  python src/backfill/backfill_features.py --start 2025-11-01 --end 2026-04-30 --strategy hybrid

Imputation strategies (for missing sensor readings):
  forward_fill | linear | seasonal | knn | hybrid (default)
"""
import argparse
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from sklearn.impute import KNNImputer
from statsmodels.tsa.seasonal import STL

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Force all feature groups on for backfill — Hopsworks schema must stay complete.
# Ablation overrides apply to training only, not to what gets stored.
import src.config as _cfg
for _k in _cfg.FEATURE_GROUPS:
    _cfg.FEATURE_GROUPS[_k] = True

from src.config import (
    DEFAULT_CITY, DEFAULT_LAT, DEFAULT_LON, BACKFILL_DAYS, FORECAST_HOURS,
)
from src.feature_pipeline.fetch_data import fetch_historical_openweather, fetch_historical_weather_openmeteo
from src.feature_pipeline.feature_engineering import build_feature_row, pm25_to_aqi
from src.feature_pipeline.store_features import insert_features


# ─── Imputation Strategies ────────────────────────────────────────────────────

def impute_forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    return df.ffill()


def impute_linear(df: pd.DataFrame) -> pd.DataFrame:
    return df.interpolate(method="linear", limit_direction="both")


def impute_seasonal(df: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps using the STL seasonal component."""
    numeric = df.select_dtypes(include=np.number).columns
    for col in numeric:
        series = df[col]
        if series.isna().sum() == 0:
            continue
        try:
            filled = series.interpolate(method="linear")
            if len(filled) >= 48:
                stl = STL(filled, period=24, robust=True)
                res = stl.fit()
                # Replace NaN positions with seasonal + trend
                mask = series.isna()
                df.loc[mask, col] = (res.seasonal + res.trend)[mask]
        except Exception:
            df[col] = series.interpolate(method="linear", limit_direction="both")
    return df


def impute_knn(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number).columns
    imputer = KNNImputer(n_neighbors=5)
    df[numeric] = imputer.fit_transform(df[numeric])
    return df


def impute_hybrid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hybrid: linear for gaps < 3h, seasonal for gaps 3–24h, KNN for gaps > 24h.
    Operates column-by-column.
    """
    numeric = df.select_dtypes(include=np.number).columns
    for col in numeric:
        series = df[col].copy()
        if series.isna().sum() == 0:
            continue
        # Identify contiguous NaN runs
        na_mask = series.isna()
        run_lengths = na_mask.astype(int).groupby(
            (~na_mask).cumsum()
        ).transform("sum")
        # Small gaps: linear
        small = na_mask & (run_lengths <= 3)
        df.loc[small, col] = series.interpolate(method="linear").loc[small]
        # Medium gaps: seasonal
        medium = na_mask & (run_lengths > 3) & (run_lengths <= 24)
        if medium.any():
            try:
                filled = series.interpolate(method="linear")
                if len(filled.dropna()) >= 48:
                    stl = STL(filled.dropna(), period=24, robust=True)
                    res = stl.fit()
                    idx = filled.dropna().index
                    trend_s   = pd.Series(res.trend,    index=idx)
                    seasonal_s = pd.Series(res.seasonal, index=idx)
                    combined  = trend_s + seasonal_s
                    df.loc[medium & df.index.isin(idx), col] = combined[medium & df.index.isin(idx)]
            except Exception:
                df.loc[medium, col] = series.interpolate(method="linear").loc[medium]
        # Large gaps: KNN on remaining NaNs.
        # If the entire column is NaN, KNNImputer returns shape (n, 0) and the
        # assignment crashes — fill with 0 instead (sensor fully offline).
        still_na = df[col].isna()
        if still_na.any():
            if still_na.all():
                df[col] = 0.0
            else:
                knn = KNNImputer(n_neighbors=5)
                df[[col]] = knn.fit_transform(df[[col]])
    return df


STRATEGIES = {
    "forward_fill": impute_forward_fill,
    "linear":       impute_linear,
    "seasonal":     impute_seasonal,
    "knn":          impute_knn,
    "hybrid":       impute_hybrid,
}


# ─── Main Backfill ────────────────────────────────────────────────────────────

def backfill(
    start_date: str,
    end_date:   str,
    strategy:   str = "hybrid",
    city:       str = DEFAULT_CITY,
    lat:        float = DEFAULT_LAT,
    lon:        float = DEFAULT_LON,
    dry_run:    bool = False,
) -> pd.DataFrame:
    print(f"[backfill] Fetching history from {start_date} to {end_date} using strategy='{strategy}'")

    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end_dt   = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)

    # Fetch pollutant history from OpenWeather Air Pollution History API in 180-day
    # chunks to avoid truncation or rate-limiting on long date ranges.
    def _fetch_chunked(lat, lon, unix_start, unix_end, chunk_days=180):
        rows, step = [], chunk_days * 86400
        s = unix_start
        while s < unix_end:
            e = min(s + step, unix_end)
            rows.extend(fetch_historical_openweather(lat, lon, s, e))
            time.sleep(1)
            s = e
        return rows

    raw_rows = _fetch_chunked(lat, lon, int(start_dt.timestamp()), int(end_dt.timestamp()))
    if not raw_rows:
        print("[backfill] No data returned from OpenWeather history API")
        return pd.DataFrame()

    df_raw = pd.DataFrame(raw_rows)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw = df_raw.set_index("timestamp").sort_index()
    df_raw = df_raw[~df_raw.index.duplicated(keep="last")]

    # Fetch historical weather from Open-Meteo (free, no API key needed)
    print("[backfill] Fetching historical weather from Open-Meteo...")
    try:
        df_weather = fetch_historical_weather_openmeteo(lat, lon, start_date, end_date)
        df_raw = df_raw.join(df_weather, how="left")
        print(f"[backfill] Weather joined — {df_weather.columns.tolist()}")
    except Exception as e:
        print(f"[backfill] Open-Meteo fetch failed ({e}) — weather fields will be zero")

    # Apply chosen imputation strategy
    impute_fn = STRATEGIES.get(strategy, impute_hybrid)
    df_imputed = impute_fn(df_raw.copy())
    df_imputed["city"] = city

    # 24h trailing-mean PM2.5 → AQI (EPA-style averaging) used for both the `aqi`
    # feature and the aqi_{h}h targets, so train and serve share one definition.
    df_imputed["pm25_24h"] = df_imputed["pm25"].rolling(24, min_periods=1).mean()

    # Compute AQI from PM2.5 and build feature rows
    feature_rows = []
    history_so_far = pd.DataFrame()

    # Map from the forecast-lead raw key → the column name in df_imputed it reads from.
    # Mirrors fetch_forecast()'s output so training (forward-shifted actuals) and serving
    # (live forecast API) produce the same forecast_leads columns.
    _LEAD_SOURCE = {
        "temperature": "temperature", "humidity": "humidity", "pressure": "pressure",
        "wind_speed": "wind_speed", "cloud_cover": "cloud_cover",
        "precipitation_1h": "precipitation_1h", "pm25_fc": "pm25_24h",
    }

    from src.feature_pipeline.sanitize import clean_raw_inputs, finalize_feature_row

    for ts, raw in df_imputed.iterrows():
        raw_dict = clean_raw_inputs(raw.to_dict())
        raw_dict["city"] = city
        aqi = pm25_to_aqi(raw_dict.get("pm25_24h", raw_dict.get("pm25", 0)))
        raw_dict["aqi_raw"] = aqi

        # Forecast leads for training rows = the actual future values (forward shift).
        # At inference these same columns come from the Open-Meteo/CAMS forecast API.
        forecast: dict = {}
        for h in FORECAST_HOURS:
            future_ts = ts + pd.Timedelta(hours=h)
            if future_ts in df_imputed.index:
                fr = df_imputed.loc[future_ts]
                forecast[h] = {k: float(fr.get(src, 0) or 0) for k, src in _LEAD_SOURCE.items()}
            else:
                forecast[h] = {}

        # Build feature row with whatever history we have so far
        row = build_feature_row(raw_dict, history_so_far, ts.to_pydatetime(), forecast=forecast)
        row["aqi"] = aqi
        row = finalize_feature_row(row)

        # Add targets: 24h-mean AQI at the future timestamp (matches the `aqi` feature).
        for h in FORECAST_HOURS:
            future_ts = ts + pd.Timedelta(hours=h)
            if future_ts in df_imputed.index:
                row[f"aqi_{h}h"] = pm25_to_aqi(df_imputed.loc[future_ts, "pm25_24h"])
            else:
                row[f"aqi_{h}h"] = None

        feature_rows.append(row)

        # Accumulate history — keep enough rows to cover the 168h lag window.
        history_row = pd.DataFrame([{"timestamp": ts, "aqi": aqi, "pm25": raw_dict.get("pm25", 0),
                                     "pressure": raw_dict.get("pressure", 1013)}])
        history_so_far = pd.concat([history_so_far, history_row], ignore_index=True).tail(200)

    df_features = pd.DataFrame(feature_rows)
    # Drop rows with any null targets (can't train on those)
    target_cols = [f"aqi_{h}h" for h in FORECAST_HOURS]
    df_features = df_features.dropna(subset=target_cols)

    print(f"[backfill] Built {len(df_features)} feature rows")

    if not dry_run:
        insert_features(df_features)
        print("[backfill] Inserted into Hopsworks Feature Store")

    return df_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",    default=(datetime.now(timezone.utc) - timedelta(days=BACKFILL_DAYS)).strftime("%Y-%m-%d"))
    parser.add_argument("--end",      default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--strategy", default="hybrid", choices=list(STRATEGIES.keys()))
    parser.add_argument("--city",     default=DEFAULT_CITY)
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    backfill(
        start_date=args.start,
        end_date=args.end,
        strategy=args.strategy,
        city=args.city,
        dry_run=args.dry_run,
    )
