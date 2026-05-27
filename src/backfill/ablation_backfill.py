"""
Backfill strategy ablation study.

For each of the 5 imputation strategies, backfill the same date range,
train a fixed Random Forest, record RMSE, log to MLflow, and persist results
to MongoDB so the dashboard's Ablation tab works on Render (ephemeral filesystem).

Runs automatically as part of the daily training pipeline.
"""
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import MLFLOW_TRACKING_URI, FORECAST_HOURS
from src.backfill.backfill_features import backfill, STRATEGIES

ABLATION_START = "2025-11-01"
ABLATION_END   = "2026-01-31"  # fixed 3-month window for fair comparison
EXPERIMENT_NAME = "backfill_ablation"
TARGET_COL = "aqi_24h"   # primary target for comparison


def run_ablation():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = {}

    for strategy_name in STRATEGIES:
        print(f"\n[ablation_backfill] Testing strategy: {strategy_name}")
        with mlflow.start_run(run_name=f"backfill_{strategy_name}"):
            mlflow.log_param("strategy", strategy_name)
            mlflow.log_param("target", TARGET_COL)

            # Backfill with this strategy (dry run — don't push to Hopsworks)
            df = backfill(
                start_date=ABLATION_START,
                end_date=ABLATION_END,
                strategy=strategy_name,
                dry_run=True,
            )

            if df.empty or TARGET_COL not in df.columns:
                print(f"[ablation_backfill] No data for strategy '{strategy_name}' — skipping")
                continue

            drop_cols = ["timestamp", "city"] + [f"aqi_{h}h" for h in FORECAST_HOURS]
            feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype != object]
            df_clean = df[feature_cols + [TARGET_COL]].dropna()

            X = df_clean[feature_cols].values
            y = df_clean[TARGET_COL].values

            # Fixed RF (no tuning — comparing strategies, not models)
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            tscv = TimeSeriesSplit(n_splits=3)
            rmses = []
            for train_idx, val_idx in tscv.split(X):
                rf.fit(X[train_idx], y[train_idx])
                preds = rf.predict(X[val_idx])
                rmses.append(np.sqrt(mean_squared_error(y[val_idx], preds)))

            rmse = float(np.mean(rmses))
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("n_rows", len(df_clean))
            results[strategy_name] = rmse
            print(f"[ablation_backfill] {strategy_name}: RMSE={rmse:.2f}")

    best = min(results, key=results.get) if results else "hybrid"
    print(f"\n[ablation_backfill] Best strategy: {best} (RMSE={results.get(best, '?'):.2f})")
    print(f"[ablation_backfill] All results: {results}")
    _persist_to_mongodb(results)
    return results


def _persist_to_mongodb(strategy_rmses: dict):
    """Upsert backfill ablation results into MongoDB so the dashboard works on Render."""
    try:
        from src.db import get_db
        strategies = [
            {"strategy": s, "rmse": r}
            for s, r in strategy_rmses.items()
        ]
        get_db()["ablation_results"].update_one(
            {"_id": "backfill_ablation"},
            {"$set": {
                "strategies": strategies,
                "run_at": datetime.now(timezone.utc).isoformat(),
            }},
            upsert=True,
        )
        print(f"[ablation_backfill] Results persisted to MongoDB ({len(strategies)} strategies)")
    except Exception as e:
        print(f"[ablation_backfill] MongoDB persist failed (non-fatal): {e}")


if __name__ == "__main__":
    run_ablation()
