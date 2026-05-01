import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta, timezone


def build_forecast_chart(current_aqi: float, aqi_24h: float, aqi_48h: float, aqi_72h: float,
                          lower_24h: float, upper_24h: float,
                          history_df: pd.DataFrame = None) -> go.Figure:
    now   = datetime.now(timezone.utc)
    times = [now + timedelta(hours=h) for h in [24, 48, 72]]
    preds = [aqi_24h, aqi_48h, aqi_72h]

    fig = go.Figure()

    # Historical trend (last 7 days)
    if history_df is not None and not history_df.empty and "aqi" in history_df.columns:
        hist = history_df.tail(168)
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(hist.get("timestamp", pd.Series())),
            y=hist["aqi"], mode="lines",
            name="Historical AQI",
            line=dict(color="#4a90d9", width=1.5),
        ))

    # Current AQI marker
    fig.add_trace(go.Scatter(
        x=[now], y=[current_aqi], mode="markers",
        name="Now",
        marker=dict(size=10, color="#ffffff", line=dict(color="#4a90d9", width=2)),
    ))

    # Conformal band for 24h (shaded)
    fig.add_trace(go.Scatter(
        x=[times[0], times[0]], y=[lower_24h, upper_24h],
        mode="lines", line=dict(width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[times[0]], y=[(lower_24h + upper_24h) / 2],
        mode="none", fill="tonexty",
        fillcolor="rgba(255,126,0,0.2)",
        name="90% Confidence",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=times, y=preds, mode="lines+markers",
        name="Forecast",
        line=dict(color="#ff7e00", width=2, dash="dash"),
        marker=dict(size=8),
    ))

    # AQI zone background bands
    zone_colors = [
        (0, 50, "rgba(0,228,0,0.06)"),
        (51, 100, "rgba(255,255,0,0.06)"),
        (101, 150, "rgba(255,126,0,0.06)"),
        (151, 200, "rgba(255,0,0,0.06)"),
        (201, 500, "rgba(126,0,35,0.06)"),
    ]
    all_x = [now - timedelta(days=7)] + times
    for lo, hi, color in zone_colors:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, line_width=0)

    fig.update_layout(
        title="AQI Forecast — Next 72 Hours",
        xaxis_title="Time (UTC)",
        yaxis_title="AQI",
        yaxis=dict(range=[0, max(500, max(preds) * 1.2)]),
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig
