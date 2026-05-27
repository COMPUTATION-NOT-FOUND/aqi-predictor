"""
Self-maintained ablation study dashboard.

Reads results from MongoDB ablation_results collection (written by
ablation_features.run_ablation and ablation_backfill.run_ablation).
Falls back to MLflow for local development when MongoDB is unavailable.
Auto-updates whenever the training pipeline runs.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.config import MLFLOW_TRACKING_URI


def _load_feature_ablation_from_mongo() -> tuple[float, list[str], list[float]]:
    """Return (baseline_rmse, groups, deltas) from MongoDB, or empty on failure."""
    try:
        from src.db import get_db
        doc = get_db()["ablation_results"].find_one({"_id": "feature_ablation"})
        if not doc or "groups" not in doc:
            return 0.0, [], []
        baseline = float(doc.get("baseline_rmse", 0.0))
        groups  = [g["group"]      for g in doc["groups"]]
        deltas  = [float(g["rmse_delta"]) for g in doc["groups"]]
        return baseline, groups, deltas
    except Exception:
        return 0.0, [], []


def _load_backfill_ablation_from_mongo() -> tuple[list[str], list[float]]:
    """Return (strategies, rmses) from MongoDB, or empty on failure."""
    try:
        from src.db import get_db
        doc = get_db()["ablation_results"].find_one({"_id": "backfill_ablation"})
        if not doc or "strategies" not in doc:
            return [], []
        strategies = [s["strategy"] for s in doc["strategies"]]
        rmses      = [float(s["rmse"]) for s in doc["strategies"]]
        return strategies, rmses
    except Exception:
        return [], []


def _load_feature_ablation_from_mlflow() -> tuple[float, list[str], list[float]]:
    """Fallback: load feature ablation from local MLflow (dev only)."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        exp = mlflow.get_experiment_by_name("feature_ablation")
        if exp is None:
            return 0.0, [], []
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        if runs.empty or "params.dropped_group" not in runs.columns:
            return 0.0, [], []
        baseline_rows = runs[runs["params.dropped_group"] == "none"]
        ablation_rows = runs[runs["params.dropped_group"] != "none"]
        baseline_rmse = float(baseline_rows["metrics.rmse"].iloc[0]) if not baseline_rows.empty else 0.0
        groups = ablation_rows["params.dropped_group"].tolist()
        deltas = (ablation_rows["metrics.rmse"] - baseline_rmse).tolist()
        return baseline_rmse, groups, deltas
    except Exception:
        return 0.0, [], []


def _load_backfill_ablation_from_mlflow() -> tuple[list[str], list[float]]:
    """Fallback: load backfill ablation from local MLflow (dev only)."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        exp = mlflow.get_experiment_by_name("backfill_ablation")
        if exp is None:
            return [], []
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        if runs.empty or "params.strategy" not in runs.columns:
            return [], []
        return runs["params.strategy"].tolist(), runs["metrics.rmse"].tolist()
    except Exception:
        return [], []


def build_ablation_charts() -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Feature Group Ablation (RMSE Δ)", "Backfill Strategy Comparison (RMSE)"],
        horizontal_spacing=0.14,
    )

    # ── Feature ablation — MongoDB first, MLflow fallback ─────────────────────
    baseline_rmse, feat_groups, feat_deltas = _load_feature_ablation_from_mongo()
    if not feat_groups:
        baseline_rmse, feat_groups, feat_deltas = _load_feature_ablation_from_mlflow()

    if feat_groups:
        colors = ["#f38ba8" if d > 2 else "#fab387" if d > 0 else "#a6e3a1" for d in feat_deltas]
        fig.add_trace(go.Bar(
            x=feat_groups, y=feat_deltas,
            marker_color=colors,
            name="RMSE increase when group dropped",
            hovertemplate="%{x}: %{y:+.2f} RMSE<extra></extra>",
        ), row=1, col=1)
        fig.add_hline(y=0, line_color="white", line_dash="dash", row=1, col=1)
    else:
        fig.add_trace(go.Bar(
            x=["No data yet"], y=[0],
            marker_color=["#313244"],
            showlegend=False,
        ), row=1, col=1)
        fig.add_annotation(
            text="Run training pipeline to see ablation data",
            xref="paper", yref="paper",
            x=0.22, y=0.5,
            font=dict(color="#a6adc8", size=13),
            showarrow=False,
        )

    # ── Backfill ablation — MongoDB first, MLflow fallback ────────────────────
    back_strategies, back_rmses = _load_backfill_ablation_from_mongo()
    if not back_strategies:
        back_strategies, back_rmses = _load_backfill_ablation_from_mlflow()

    if back_strategies:
        min_rmse = min(back_rmses)
        colors_b = ["#f5c842" if r == min_rmse else "#4a90d9" for r in back_rmses]
        fig.add_trace(go.Bar(
            x=back_strategies, y=back_rmses,
            marker_color=colors_b,
            name="RMSE by backfill strategy",
            hovertemplate="%{x}: RMSE=%{y:.2f}<extra></extra>",
        ), row=1, col=2)
    else:
        fig.add_trace(go.Bar(
            x=["No data yet"], y=[0],
            marker_color=["#313244"],
            showlegend=False,
        ), row=1, col=2)
        fig.add_annotation(
            text="Backfill ablation not run yet — only feature pipeline<br>data exists. Right panel is intentionally empty.",
            xref="paper", yref="paper",
            x=0.78, y=0.5,
            font=dict(color="#a6adc8", size=12),
            align="center",
            showarrow=False,
        )

    fig.update_layout(
        title="Self-Maintained Ablation Study (auto-updates nightly)",
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=60),
        height=420,
    )
    fig.update_yaxes(title_text="RMSE Δ vs Baseline", row=1, col=1,
                     gridcolor="#313244", zerolinecolor="#45475a")
    fig.update_yaxes(title_text="RMSE", row=1, col=2,
                     gridcolor="#313244")
    fig.update_xaxes(tickangle=-30)
    return fig
