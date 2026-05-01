import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.config import AQI_ZONES, aqi_color, aqi_label


def build_gauge(aqi: float, title: str = "Current AQI") -> go.Figure:
    color = aqi_color(aqi)
    label = aqi_label(aqi)

    steps = []
    for lo, hi, _, c in AQI_ZONES:
        steps.append(dict(range=[lo, hi], color=c, thickness=0.8))

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi,
        number=dict(font=dict(size=48, color=color)),
        title=dict(text=f"{title}<br><span style='font-size:14px;color:{color}'>{label}</span>",
                   font=dict(size=16, color="#cdd6f4")),
        gauge=dict(
            axis=dict(range=[0, 500], tickwidth=1, tickcolor="#cdd6f4",
                      tickvals=[0, 50, 100, 150, 200, 300, 500]),
            bar=dict(color=color, thickness=0.3),
            bgcolor="#1e1e2e",
            borderwidth=0,
            steps=steps,
            threshold=dict(
                line=dict(color="white", width=2),
                thickness=0.75,
                value=aqi,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        margin=dict(l=20, r=20, t=60, b=20),
        height=280,
    )
    return fig
