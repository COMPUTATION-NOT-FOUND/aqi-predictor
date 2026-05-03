from dash import html
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.config import AQI_ALERT_THRESHOLD, aqi_label, aqi_color


def build_alert_banner(aqi: float) -> html.Div:
    if aqi <= AQI_ALERT_THRESHOLD:
        return html.Div()  # no alert

    label = aqi_label(aqi)
    color = aqi_color(aqi)

    return html.Div(
        children=[
            html.I(className="fas fa-exclamation-triangle me-2"),
            html.Strong(f"HAZARDOUS AIR QUALITY ALERT — {label} (AQI {aqi:.0f})"),
            html.Br(),
            html.Span("Avoid outdoor activities. Sensitive groups should remain indoors.",
                      style={"fontSize": "0.9rem"}),
        ],
        style={
            "backgroundColor": color,
            "color": "#ffffff",
            "padding": "14px 20px",
            "borderRadius": "8px",
            "marginBottom": "16px",
            "animation": "pulse 1.5s infinite",
            "fontWeight": "bold",
            "textAlign": "center",
        },
        className="aqi-alert-banner",
    )


ALERT_CSS = """
@keyframes pulse {
    0%   { opacity: 1.0; }
    50%  { opacity: 0.6; }
    100% { opacity: 1.0; }
}
"""
