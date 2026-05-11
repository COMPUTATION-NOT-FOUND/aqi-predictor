import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.feature_pipeline.drift_monitor import load_drift_log
from src.config import PSI_WARN_THRESHOLD, PSI_ALERT_THRESHOLD


def build_drift_heatmap() -> go.Figure:
    entries = load_drift_log(last_n=720)

    _no_data_layout = dict(
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        margin=dict(l=20, r=20, t=60, b=40),
    )

    if not entries:
        # Return a clear, styled "no drift yet" message instead of a blank axes
        fig = go.Figure()
        fig.add_annotation(
            text=(
                "✅  No drift detected yet<br><br>"
                "<span style='font-size:13px;color:#a6adc8'>"
                "The drift monitor compares each hourly batch to the training baseline.<br>"
                "PSI scores will appear here once the feature pipeline has run several times.<br><br>"
                "PSI &lt; 0.1 → no drift &nbsp;|&nbsp; "
                "0.1–0.2 → moderate &nbsp;|&nbsp; "
                "&gt; 0.2 → alert (email sent)"
                "</span>"
            ),
            xref="paper", yref="paper",
            x=0.5, y=0.55,
            font=dict(size=18, color="#a6e3a1"),
            align="center",
            showarrow=False,
        )
        fig.update_layout(
            title="Data Drift Monitor",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            **_no_data_layout,
        )
        return fig

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

    z = pivot.fillna(0).values

    # If everything is effectively zero → show "no drift" message
    if z.max() < 0.01:
        fig = go.Figure()
        fig.add_annotation(
            text=(
                "✅  No significant drift detected<br><br>"
                "<span style='font-size:13px;color:#a6adc8'>"
                f"All feature PSI scores &lt; 0.01 — data distribution matches training baseline."
                "</span>"
            ),
            xref="paper", yref="paper",
            x=0.5, y=0.55,
            font=dict(size=18, color="#a6e3a1"),
            align="center",
            showarrow=False,
        )
        fig.update_layout(title="Data Drift Monitor — No Drift Detected",
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          **_no_data_layout)
        return fig

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
