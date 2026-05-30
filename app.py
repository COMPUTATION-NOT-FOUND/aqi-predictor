"""
Pearls AQI Predictor — Streamlit dashboard (Streamlit Community Cloud entry point).

Reads everything from MongoDB; never runs model inference live. The 3-day forecast is
precomputed hourly by src/feature_pipeline/predict.py and read here (cached).

Tabs: Live Forecast · Leaderboard · Ablation Study · Feature Importance.

Local run:  streamlit run app.py
Deploy:     Streamlit Community Cloud — main file app.py, secrets set in the UI.
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.config import DEFAULT_CITY, AQI_ZONES
from src.dashboard.inference import (
    load_latest_prediction,
    get_recent_features,
    load_forecast_trackrecord,
    load_oof_predictions,
    load_all_models_metadata,
    load_shap_data,
)
from src.dashboard.components.alerts import alert_for
from src.dashboard.components.gauge import build_gauge
from src.dashboard.components.forecast_chart import (
    build_ground_truth_chart,
    build_trackrecord_chart,
    build_hindcast_chart,
    build_live_forecast_chart,
)
from src.dashboard.components.leaderboard import (
    leaderboard_dataframe, build_radar_chart, METRIC_GLOSSARY,
)
from src.dashboard.components.shap_plot import build_shap_bar
from src.dashboard.components.ablation_tab import build_ablation_charts

st.set_page_config(page_title="AQI Predictor", page_icon=":material/air:", layout="wide")

_PLOTLY = {"width": "stretch", "config": {"displayModeBar": False}}


def _fmt(value, decimals: int = 3) -> str:
    """Format a metric value, tolerating None / non-numeric."""
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _style_rows(row: pd.Series) -> list:
    """Row styler for the leaderboard: gold champion row, red negative skill."""
    is_champ = "CHAMPION" in str(row.get("Status", ""))
    styles = []
    for col in row.index:
        css = ""
        if is_champ:
            css = "background-color:#2a2000;color:#f5c842;font-weight:bold"
        if col == "Skill":
            try:
                if float(row[col]) < 0:
                    css = (css + ";" if css else "") + "color:#f38ba8"
            except (TypeError, ValueError):
                pass
        styles.append(css)
    return styles


# ─── Cached data loaders (TTL 10 min; sidebar Refresh clears all) ──────────────

@st.cache_data(ttl=600, show_spinner=False)
def c_prediction(city: str) -> dict:
    return load_latest_prediction(city)


@st.cache_data(ttl=600, show_spinner=False)
def c_history(city: str) -> pd.DataFrame:
    return get_recent_features(city, n=200)


@st.cache_data(ttl=600, show_spinner=False)
def c_trackrecord(city: str) -> dict:
    return load_forecast_trackrecord(city)


@st.cache_data(ttl=600, show_spinner=False)
def c_oof(city: str) -> pd.DataFrame:
    return load_oof_predictions(city)


@st.cache_data(ttl=600, show_spinner=False)
def c_metadata() -> list:
    return load_all_models_metadata()


@st.cache_data(ttl=600, show_spinner=False)
def c_shap() -> dict:
    return load_shap_data()


@st.cache_data(ttl=600, show_spinner=False)
def c_ablation_fig():
    return build_ablation_charts()


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title(":material/air: AQI Predictor")
    st.caption(f"City: **{DEFAULT_CITY.title()}** · forecasts refresh hourly")
    if st.button("Refresh data", icon=":material/refresh:", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.caption("EPA AQI zones")
    for lo, hi, label, color in AQI_ZONES:
        st.markdown(
            f"<span style='background:{color};color:#000;padding:1px 6px;"
            f"border-radius:4px;font-size:0.72rem'>{label}: {lo}–{hi}</span>",
            unsafe_allow_html=True,
        )

city = DEFAULT_CITY

tab_forecast, tab_board, tab_ablation, tab_shap = st.tabs([
    ":material/insights: Live Forecast",
    ":material/leaderboard: Leaderboard",
    ":material/science: Ablation Study",
    ":material/query_stats: Feature Importance",
])


# ─── Tab 1 — Live Forecast ─────────────────────────────────────────────────────

with tab_forecast:
    pred = c_prediction(city)
    history = c_history(city)
    track = c_trackrecord(city)

    current = pred.get("current_aqi", 0)
    a24, a48, a72 = pred.get("aqi_24h", 0), pred.get("aqi_48h", 0), pred.get("aqi_72h", 0)
    lo24, hi24 = pred.get("lower_24h", 0), pred.get("upper_24h", 0)

    if pred.get("error"):
        st.warning(pred["error"], icon=":material/hourglass_empty:")
    elif pred.get("stale"):
        age_h = (pred.get("age_minutes") or 0) / 60.0
        st.info(f"Latest forecast is {age_h:.1f}h old — the hourly pipeline may be delayed.",
                icon=":material/schedule:")

    alert = alert_for(max(current, a24, a48, a72))
    if alert:
        st.error(f"**HAZARDOUS AIR QUALITY — {alert['label']} (AQI {alert['aqi']:.0f})**  \n{alert['message']}",
                 icon=":material/warning:")

    g1, g2, g3, g4 = st.columns(4)
    for col, val, label in [(g1, current, "Current AQI"), (g2, a24, "24h Forecast"),
                            (g3, a48, "48h Forecast"), (g4, a72, "72h Forecast")]:
        with col:
            st.plotly_chart(build_gauge(val, label), **_PLOTLY)

    with st.expander("How to read these charts", icon=":material/info:"):
        st.markdown(
            "- **Observed AQI** — PM2.5-derived ground truth (24h-mean, EPA scale).\n"
            "- **Forecast Track Record** — past forecasts the system actually made, scored "
            "against the AQI observed at each target time (honest, leakage-free).\n"
            "- **Live Forecast** — the next 72h, with a 90% conformal band on the 24h horizon."
        )

    st.subheader("Observed AQI — last 7 days")
    st.plotly_chart(build_ground_truth_chart(history), **_PLOTLY)

    st.subheader("Forecast track record — predicted vs observed")
    st.plotly_chart(build_trackrecord_chart(track), **_PLOTLY)

    st.subheader("Champion hindcast — model predictions across history")
    st.plotly_chart(build_hindcast_chart(history, c_oof(city)), **_PLOTLY)

    st.subheader("Live forecast — next 72 hours")
    st.plotly_chart(
        build_live_forecast_chart(current, a24, a48, a72, lo24, hi24, history), **_PLOTLY
    )

    if pred.get("predicted_at") is not None:
        st.caption(f"Model: {pred.get('model_version', '—')} · "
                   f"forecast computed at {pred['predicted_at']:%Y-%m-%d %H:%M UTC}")


# ─── Tab 2 — Leaderboard ────────────────────────────────────────────────────────

with tab_board:
    st.header("Model Registry Leaderboard")
    st.caption("Champion selected via TOPSIS (OOF RMSE, IoA, Skill, MAE). "
               "Hard gates: Skill > 0, IoA ≥ 0.15, not overfit.")

    metadata = c_metadata()
    df, champion = leaderboard_dataframe(metadata)

    if champion:
        st.markdown(f"#### :material/workspace_premium: Champion — {champion.get('name', '—')}")
        m = st.columns(5)
        m[0].metric("TOPSIS", _fmt(champion.get("topsis_score")))
        m[1].metric("RMSE", _fmt(champion.get("rmse"), 1))
        m[2].metric("OOF RMSE", _fmt(champion.get("oof_rmse"), 1))
        m[3].metric("IoA", _fmt(champion.get("ioa")))
        m[4].metric("Skill", _fmt(champion.get("skill_score")))

    with st.expander("Metric definitions & standards", icon=":material/menu_book:"):
        st.dataframe(pd.DataFrame(METRIC_GLOSSARY), width="stretch", hide_index=True)

    radar = build_radar_chart(metadata)
    if radar is not None:
        st.plotly_chart(radar, **_PLOTLY)

    if not df.empty:
        st.dataframe(df.style.apply(_style_rows, axis=1), width="stretch", hide_index=True)
    else:
        st.info("No models registered yet — run the training pipeline.")


# ─── Tab 3 — Ablation Study ─────────────────────────────────────────────────────

with tab_ablation:
    st.header("Self-Maintained Ablation Study")
    st.caption("Auto-updates nightly after the training pipeline runs. Left: RMSE increase "
               "when each feature group is removed (taller = more important). "
               "Right: backfill imputation strategy comparison.")
    st.plotly_chart(c_ablation_fig(), **_PLOTLY)


# ─── Tab 4 — Feature Importance (SHAP) ──────────────────────────────────────────

with tab_shap:
    st.header("SHAP Feature Importance")
    st.caption("Mean |SHAP| contribution of each feature on the best tree-based model. "
               "Longer bar = more influence on the prediction.")
    st.plotly_chart(build_shap_bar(c_shap(), top_n=15), **_PLOTLY)
