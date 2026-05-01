import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.feature_pipeline.drift_monitor import load_drift_log
from src.config import PSI_WARN_THRESHOLD, PSI_ALERT_THRESHOLD


def build_drift_heatmap() -> go.Figure:
    entries = load_drift_log(last_n=720)
    if not entries:
        return go.Figure().update_layout(
            title="No drift data yet (run feature pipeline first)",
            paper_bgcolor="#1e1e2e", font=dict(color="#cdd6f4"),
        )

    records = []
    for e in entries:
        ts = e["timestamp"]
        for feat, psi in e.get("psi", {}).items():
            records.append({"timestamp": ts, "feature": feat, "psi": psi})

    df = pd.DataFrame(records)
    pivot = df.pivot_table(index="feature", columns="timestamp", values="psi", aggfunc="mean")

    # Keep only top 20 most volatile features
    feature_volatility = pivot.std(axis=1).nlargest(20).index
    pivot = pivot.loc[feature_volatility]

    # Color: green < 0.1, yellow 0.1–0.2, red > 0.2
    z = pivot.fillna(0).values
    colorscale = [
        [0.0,  "#a6e3a1"],
        [PSI_WARN_THRESHOLD / 0.5, "#f9e2af"],
        [PSI_ALERT_THRESHOLD / 0.5, "#f38ba8"],
        [1.0,  "#7e0023"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(c)[:16] for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=colorscale,
        zmin=0, zmax=0.5,
        colorbar=dict(title="PSI", tickvals=[0, 0.1, 0.2, 0.5],
                      ticktext=["0 (ok)", "0.1 (warn)", "0.2 (alert)", "0.5+"]),
        hovertemplate="Feature: %{y}<br>Time: %{x}<br>PSI: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Data Drift Monitor — PSI per Feature (last 30 days)",
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4", size=11),
        margin=dict(l=20, r=20, t=50, b=80),
        xaxis=dict(tickangle=-45),
    )
    return fig
