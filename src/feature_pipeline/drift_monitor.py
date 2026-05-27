"""
PSI (Population Stability Index) based data drift monitor.

PSI < 0.1  → no drift (green)
PSI 0.1–0.2 → moderate drift (yellow, warn)
PSI > 0.2  → significant drift (red, triggers alert)

Baseline distribution is computed from the training window and stored in
MongoDB so it persists across GitHub Actions runs and is accessible on Render.
Each hourly pipeline run computes PSI for all numeric features and logs to
the drift_log collection (capped at 720 entries ~ 30 days of hourly checks).
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime, timezone

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import PSI_WARN_THRESHOLD, PSI_ALERT_THRESHOLD

BASELINE_PATH = Path(__file__).parent.parent.parent / "drift_baseline.pkl"

_BINS = 10


def _psi_for_feature(baseline_vals: np.ndarray, current_vals: np.ndarray) -> float:
    """Compute PSI between baseline and current distributions."""
    eps = 1e-8
    bins = np.histogram_bin_edges(baseline_vals, bins=_BINS)
    base_counts, _ = np.histogram(baseline_vals, bins=bins)
    curr_counts, _ = np.histogram(current_vals,  bins=bins)
    base_pct = base_counts / (base_counts.sum() + eps)
    curr_pct = curr_counts / (curr_counts.sum() + eps)
    base_pct = np.clip(base_pct, eps, None)
    curr_pct = np.clip(curr_pct, eps, None)
    return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))


def save_baseline(df: pd.DataFrame):
    """Save training-set feature distributions as the drift baseline to MongoDB."""
    from src.db import get_db
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    baseline = {col: df[col].dropna().values.tolist() for col in num_cols}
    get_db()["drift_baseline"].update_one(
        {"_id": "current"},
        {"$set": {"features": baseline, "saved_at": datetime.now(timezone.utc).isoformat()}},
        upsert=True,
    )
    print(f"[drift_monitor] Baseline saved to MongoDB ({len(num_cols)} features)")


def load_baseline() -> dict:
    """Load baseline from MongoDB; fall back to local pickle for offline dev."""
    try:
        from src.db import get_db
        doc = get_db()["drift_baseline"].find_one({"_id": "current"})
        if doc and "features" in doc:
            return {col: np.array(vals) for col, vals in doc["features"].items()}
    except Exception:
        pass
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def compute_psi(current_df: pd.DataFrame) -> dict[str, float]:
    """Compute PSI for each numeric feature against the saved baseline."""
    baseline = load_baseline()
    if not baseline:
        return {}
    psi_scores = {}
    for col, base_vals in baseline.items():
        if col in current_df.columns:
            curr_vals = current_df[col].dropna().values
            if len(curr_vals) >= 5:
                psi_scores[col] = _psi_for_feature(base_vals, curr_vals)
    return psi_scores


def log_psi(psi_scores: dict[str, float]):
    """Insert PSI results into MongoDB drift_log collection."""
    from src.db import get_db
    entry = {
        "timestamp": datetime.now(timezone.utc),
        "psi": psi_scores,
        "drifted_features": [k for k, v in psi_scores.items() if v > PSI_ALERT_THRESHOLD],
    }
    col = get_db()["drift_log"]
    col.insert_one(entry)
    # Keep collection lean — drop anything older than 720 entries
    total = col.count_documents({})
    if total > 720:
        oldest = col.find({}, {"_id": 1}).sort("timestamp", 1).limit(total - 720)
        col.delete_many({"_id": {"$in": [d["_id"] for d in oldest]}})
    entry["timestamp"] = entry["timestamp"].isoformat()
    return entry


def load_drift_log(last_n: int = 720) -> list[dict]:
    """Load the last N drift log entries from MongoDB."""
    try:
        from src.db import get_db
        docs = list(
            get_db()["drift_log"]
            .find({}, {"_id": 0})
            .sort("timestamp", -1)
            .limit(last_n)
        )
        for d in docs:
            if hasattr(d.get("timestamp"), "isoformat"):
                d["timestamp"] = d["timestamp"].isoformat()
        return list(reversed(docs))
    except Exception:
        return []


def check_drift(current_df: pd.DataFrame) -> tuple[dict, bool]:
    """
    Run drift check. Returns (psi_scores, alert_triggered).
    Logs results and prints warnings.
    """
    psi_scores = compute_psi(current_df)
    if not psi_scores:
        print("[drift_monitor] No baseline found — skipping drift check")
        return {}, False

    alert_triggered = False
    for feat, psi in psi_scores.items():
        if psi > PSI_ALERT_THRESHOLD:
            print(f"[drift_monitor] ALERT: PSI={psi:.3f} for '{feat}' (threshold {PSI_ALERT_THRESHOLD})")
            alert_triggered = True
        elif psi > PSI_WARN_THRESHOLD:
            print(f"[drift_monitor] WARN:  PSI={psi:.3f} for '{feat}'")

    log_psi(psi_scores)
    return psi_scores, alert_triggered
