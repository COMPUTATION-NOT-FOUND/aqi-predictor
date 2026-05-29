"""
Hazardous-AQI alert helper (framework-agnostic).

Returns the alert payload; the Streamlit app renders it with st.error/st.warning.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.config import AQI_ALERT_THRESHOLD, aqi_label


def alert_for(aqi: float) -> dict | None:
    """Return {'label', 'aqi', 'message'} when AQI exceeds the alert threshold, else None."""
    if aqi is None or aqi <= AQI_ALERT_THRESHOLD:
        return None
    return {
        "label":   aqi_label(aqi),
        "aqi":     float(aqi),
        "message": "Avoid outdoor activities. Sensitive groups should remain indoors.",
    }
