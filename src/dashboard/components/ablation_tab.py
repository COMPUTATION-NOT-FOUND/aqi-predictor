"""
Self-maintained ablation study dashboard.
Reads results from MLflow experiments and renders comparison charts.
Auto-updates whenever the training pipeline runs.
"""
import mlflow
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.config import MLFLOW_TRACKING_URI


def _load_experiment_runs(experiment_name: str) -> pd.DataFrame:
    """Load all runs from an MLflow experiment into a DataFrame."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            return pd.DataFrame()
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        return runs
    except Exception:
        return pd.DataFrame()


def build_ablation_charts() -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Feature Group Ablation (RMSE Δ)", "Backfill Strategy Comparison (RMSE)"],
        horizontal_spacing=0.12,
    )

    # ── Feature ablation ──────────────────────────────────────────────────────
    feat_runs = _load_experiment_runs("feature_ablation")
    if not feat_runs.empty and "params.dropped_group" in feat_runs.columns:
        baseline_rows = feat_runs[feat_runs["params.dropped_group"] == "none"]
        ablation_rows = feat_runs[feat_runs["params.dropped_group"] != "none"]

        baseline_rmse = float(baseline_rows["metrics.rmse"].iloc[0]) if not baseline_rows.empty else 0.0

        groups = ablation_rows["params.dropped_group"].tolist()
        deltas = (ablation_rows["metrics.rmse"] - baseline_rmse).tolist()

        colors = ["#f38ba8" if d > 2 else "#fab387" if d > 0 else "#a6e3a1" for d in deltas]

        fig.add_trace(go.Bar(
            x=groups, y=deltas,
            marker_color=colors,
            name="RMSE increase when group dropped",
            hovertemplate="%{x}: +%{y:.2f} RMSE<extra></extra>",
        ), row=1, col=1)
        fig.add_hline(y=0, line_color="white", line_dash="dash", row=1, col=1)
    else:
        fig.add_annotation(text="Run training pipeline to populate ablation data",
                           xref="x", yref="y", row=1, col=1,
                           font=dict(color="#a6adc8"), showarrow=False)

    # ── Backfill ablation ─────────────────────────────────────────────────────
    back_runs = _load_experiment_runs("backfill_ablation")
    if not back_runs.empty and "params.strategy" in back_runs.columns:
        strategies = back_runs["params.strategy"].tolist()
        rmses      = back_runs["metrics.rmse"].tolist()
        min_rmse   = min(rmses) if rmses else 0
        colors_b   = ["#f5c842" if r == min_rmse else "#4a90d9" for r in rmses]

        fig.add_trace(go.Bar(
            x=strategies, y=rmses,
            marker_color=colors_b,
            name="RMSE by backfill strategy",
            hovertemplate="%{x}: RMSE=%{y:.2f}<extra></extra>",
        ), row=1, col=2)
    else:
        fig.add_annotation(text="Run backfill ablation to populate data",
                           xref="x2", yref="y2",
                           font=dict(color="#a6adc8"), showarrow=False)

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
