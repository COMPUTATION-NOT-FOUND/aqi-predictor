"""
Chronos-Bolt zero-shot baseline.

Chronos-Bolt is a pretrained time-series model from Amazon. It is not trained
here: the weights are downloaded from Hugging Face on each run and used as-is,
so it shows how far a general forecaster gets with nothing but the AQI history.

It appears on the leaderboard only. It can never become champion, because the
hourly prediction step would have to download torch and the weights every hour.

Turn it off with TRAIN_CHRONOS=0. Pick the size with CHRONOS_MODEL
(default amazon/chronos-bolt-tiny, ~9M parameters, runs on CPU).
"""
import os

import numpy as np

from src.config import FORECAST_HOURS
from src.training_pipeline.evaluate_model import evaluate_all

CHRONOS_MODEL = os.getenv("CHRONOS_MODEL", "amazon/chronos-bolt-tiny")
CONTEXT_LEN = 2048      # Chronos-Bolt's maximum context
BATCH_SIZE = 64


def evaluate_chronos(aqi_all, test_start, Y_test, y_pers):
    """
    Forecast every test row from the AQI history up to and including that row.

    aqi_all    current AQI for every row (train + val + test), in time order
    test_start index of the first test row in aqi_all
    Y_test     (n_test, n_horizons) observed AQI at each horizon
    y_pers     flattened persistence forecast, as passed to evaluate_all

    Returns a result dict for the leaderboard, or None if Chronos is not
    installed or fails for any reason.
    """
    try:
        import torch
        from chronos import BaseChronosPipeline
    except ImportError:
        print("[chronos] chronos-forecasting not installed. Skipped.")
        return None

    try:
        pipe = BaseChronosPipeline.from_pretrained(
            CHRONOS_MODEL, device_map="cpu", torch_dtype=torch.float32,
        )
        horizon = max(FORECAST_HOURS)
        steps = [h - 1 for h in FORECAST_HOURS]   # step 0 is one hour ahead
        aqi_all = np.asarray(aqi_all, dtype=np.float32)
        n_test = len(Y_test)

        preds = []
        for start in range(0, n_test, BATCH_SIZE):
            context = []
            for i in range(start, min(start + BATCH_SIZE, n_test)):
                end = test_start + i + 1
                context.append(torch.tensor(aqi_all[max(0, end - CONTEXT_LEN):end]))
            quantiles, _mean = pipe.predict_quantiles(
                context, prediction_length=horizon, quantile_levels=[0.5],
            )
            median = quantiles[..., 0].numpy()       # (batch, horizon)
            preds.append(median[:, steps])
        y_pred = np.clip(np.concatenate(preds), 0, 500)

        metrics = evaluate_all(Y_test, y_pred, y_pers)
        # Nothing is fitted, so there is no out-of-fold or train score.
        # Test RMSE stands in for both and the model can never be "overfit".
        metrics["oof_rmse"] = metrics["rmse"]
        metrics["train_rmse"] = metrics["rmse"]
        metrics["overfit_flag"] = 0
        print(f"[chronos] {CHRONOS_MODEL}: RMSE={metrics['rmse']:.2f} | "
              f"IoA={metrics.get('ioa', 0):.3f} | Skill={metrics.get('skill_score', 0):.3f}")
        return {"name": "ChronosBolt", "model": None, "metrics": metrics}
    except Exception as e:
        print(f"[chronos] Skipped: {e}")
        return None
