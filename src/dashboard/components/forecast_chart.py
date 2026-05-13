"""
Forecast chart components — three separate panels for the Live Forecast tab.

Panel A — build_ground_truth_chart()
    Observed AQI for the last 7 days.  This is the "ground truth" baseline
    with AQI zone colour bands.  Source: AQICN / OpenMeteo PM2.5 readings.

Panel B — build_hindcast_chart()
    OOF (out-of-fold) model predictions overlaid on observed values from the
    last training run.  This is the model-verification panel — you can see
    directly whether and where the model tracks reality.
    Metric annotations (RMSE, IoA, Skill Score) are drawn on the chart itself.

    Standard used
    -------------
    IoA  : Willmott (1981) Index of Agreement — environmental-model standard.
           Range 0–1. > 0.3 is acceptable for urban AQI with limited training data.
    Skill: 1 − RMSE_model/RMSE_persistence (Murphy 1988).
           > 0 means the model beats simply repeating yesterday's value.
    RMSE : Root Mean Square Error in AQI units.

Panel C — build_live_forecast_chart()
    Forward-looking 72-hour forecast starting from "Now".
    Includes 90 % conformal prediction interval for the 24-hour horizon
    (computed by MAPIE on a held-out calibration set — not a Gaussian band).
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone


# ─── AQI zone colour bands (US-EPA standard) ──────────────────────────────────
_AQI_ZONES = [
    (0,   50,  "Good",                    "rgba(0,228,0,0.07)"),
    (51,  100, "Moderate",                "rgba(255,255,0,0.07)"),
    (101, 150, "Unhealthy for Sensitive", "rgba(255,126,0,0.07)"),
    (151, 200, "Unhealthy",               "rgba(255,0,0,0.07)"),
    (201, 300, "Very Unhealthy",          "rgba(143,63,151,0.07)"),
    (301, 500, "Hazardous",               "rgba(126,0,35,0.07)"),
]

_DARK_BG   = "#1e1e2e"
_FONT      = {"color": "#cdd6f4"}
_GRID      = {"gridcolor": "#313244", "zerolinecolor": "#313244"}
_LEGEND    = {"bgcolor": "rgba(0,0,0,0)", "font": {"size": 11, "color": "#cdd6f4"}}


def _add_aqi_zone_bands(fig: go.Figure):
    """Add US-EPA AQI colour bands as horizontal rectangles."""
    for lo, hi, label, color in _AQI_ZONES:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, line_width=0,
                      annotation_text=label if lo <= 100 else "",
                      annotation_position="right",
                      annotation_font={"size": 9, "color": "#6c7086"})


def _base_layout(title: str, subtitle: str) -> dict:
    return dict(
        title=dict(
            text=f"<b>{title}</b><br><sup style='color:#6c7086'>{subtitle}</sup>",
            font={"size": 14, "color": "#cdd6f4"},
        ),
        xaxis=dict(title="Time (UTC)", **_GRID),
        yaxis=dict(title="AQI", range=[0, 500], **_GRID),
        plot_bgcolor=_DARK_BG,
        paper_bgcolor=_DARK_BG,
        font=_FONT,
        legend=_LEGEND,
        margin=dict(l=55, r=20, t=65, b=40),
        hovermode="x unified",
    )


# ─── Panel A — Historical Ground Truth ────────────────────────────────────────

def build_ground_truth_chart(history_df: pd.DataFrame | None) -> go.Figure:
    """
    Panel A: Observed AQI for the last 7 days.

    Shows what actually happened — the reference baseline before you look at
    what the model predicted.  Source: AQICN + OpenMeteo PM2.5 sensor data.
    """
    fig = go.Figure()
    _add_aqi_zone_bands(fig)

    if history_df is not None and not history_df.empty and "aqi" in history_df.columns:
        hist = history_df.tail(168).copy()
        ts   = pd.to_datetime(hist.get("timestamp", pd.Series()), utc=True, errors="coerce")
        aqi  = pd.to_numeric(hist["aqi"], errors="coerce")

        fig.add_trace(go.Scatter(
            x=ts, y=aqi, mode="lines",
            name="Observed AQI",
            line=dict(color="#89b4fa", width=2),
            hovertemplate="%{y:.0f} AQI<extra>Observed</extra>",
        ))
        # Mark today
        now = datetime.now(timezone.utc)
        fig.add_vline(x=now.timestamp() * 1000, line_dash="dot",
                      line_color="#6c7086",
                      annotation_text="Now", annotation_position="top right",
                      annotation_font_color="#6c7086")
    else:
        fig.add_annotation(text="No historical data available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font={"color": "#6c7086", "size": 14})

    fig.update_layout(**_base_layout(
        "Observed AQI — Last 7 Days",
        "Source: AQICN + OpenMeteo PM2.5 sensor readings (US-EPA AQI scale)",
    ))
    return fig


# ─── Panel B — Model Hindcast vs Ground Truth ─────────────────────────────────

def build_hindcast_chart(
    history_df: pd.DataFrame | None,
    oof_df: pd.DataFrame | None,
) -> go.Figure:
    """
    Panel B: OOF model predictions overlaid on observed values.

    This is the model-verification panel.  OOF (out-of-fold) predictions
    are made on held-out validation splits — the model never saw these
    samples during training, so this comparison is leakage-free.

    Metric annotations are rendered directly on the chart.

    Metric standards
    ----------------
    RMSE  : Root Mean Square Error (AQI units) — lower is better
    IoA   : Willmott (1981) Index of Agreement; range 0–1
            > 0.3 is acceptable for urban AQI forecasting with limited data
    Skill : Murphy (1988); = 1 − RMSE_model/RMSE_persistence
            > 0 means model beats the naive "repeat yesterday" baseline
    """
    fig = go.Figure()
    _add_aqi_zone_bands(fig)

    has_obs  = (history_df is not None and not history_df.empty
                and "aqi" in history_df.columns)
    has_oof  = (oof_df is not None and not oof_df.empty)

    if has_obs:
        hist = history_df.tail(168).copy()
        ts   = pd.to_datetime(hist.get("timestamp", pd.Series()), utc=True, errors="coerce")
        aqi  = pd.to_numeric(hist["aqi"], errors="coerce")
        fig.add_trace(go.Scatter(
            x=ts, y=aqi, mode="lines",
            name="Observed (ground truth)",
            line=dict(color="#89b4fa", width=2),
            hovertemplate="%{y:.0f}<extra>Observed</extra>",
        ))

    if has_oof:
        oof = oof_df.copy()
        ts_oof  = pd.to_datetime(oof.get("timestamp", pd.Series()), utc=True, errors="coerce")
        pred    = pd.to_numeric(oof.get("predicted", oof.get("aqi_pred", None)), errors="coerce")
        if pred is not None:
            fig.add_trace(go.Scatter(
                x=ts_oof, y=pred, mode="lines+markers",
                name="OOF Predictions (model)",
                line=dict(color="#f38ba8", width=1.5, dash="dash"),
                marker=dict(size=4, color="#f38ba8"),
                hovertemplate="%{y:.0f}<extra>OOF Predicted</extra>",
            ))

            # Compute metrics for annotation
            obs_aligned = pd.to_numeric(
                oof.get("observed", oof.get("aqi", None)), errors="coerce"
            )
            if obs_aligned is not None:
                mask = obs_aligned.notna() & pred.notna()
                if mask.sum() > 2:
                    from src.training_pipeline.evaluate_model import (
                        rmse as _rmse, index_of_agreement, skill_score as _ss
                    )
                    y_true = obs_aligned[mask].values
                    y_pred = pred[mask].values
                    # Persistence baseline = shift by 24 samples (≈1 day at hourly cadence)
                    y_pers = np.roll(y_true, 24)

                    val_rmse  = _rmse(y_true, y_pred)
                    val_ioa   = index_of_agreement(y_true, y_pred)
                    val_skill = _ss(y_true, y_pred, y_pers)

                    _add_metric_annotation(fig, val_rmse, val_ioa, val_skill)

    if not has_obs and not has_oof:
        fig.add_annotation(
            text="Run the training pipeline first to populate OOF predictions",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"color": "#6c7086", "size": 13},
        )

    fig.update_layout(**_base_layout(
        "Model Verification — OOF Predictions vs Observed",
        "OOF = out-of-fold (held-out) — leakage-free · "
        "IoA: Willmott (1981) · Skill: Murphy (1988) 1 − RMSE_model/RMSE_persistence",
    ))
    return fig


def _add_metric_annotation(fig: go.Figure, val_rmse: float, val_ioa: float, val_skill: float):
    """Render RMSE / IoA / Skill Score as a metric badge box on the figure."""
    ioa_color   = "#a6e3a1" if val_ioa   >= 0.3  else ("#f9e2af" if val_ioa   >= 0.2 else "#f38ba8")
    skill_color = "#a6e3a1" if val_skill > 0.0   else "#f38ba8"
    rmse_color  = "#cdd6f4"

    text = (
        f"<b>RMSE</b> <span style='color:{rmse_color}'>{val_rmse:.1f}</span>  "
        f"<b>IoA</b> <span style='color:{ioa_color}'>{val_ioa:.3f}</span>  "
        f"<b>Skill</b> <span style='color:{skill_color}'>{val_skill:+.3f}</span>"
    )
    fig.add_annotation(
        text=text,
        xref="paper", yref="paper",
        x=0.01, y=0.97,
        showarrow=False,
        align="left",
        bgcolor="rgba(30,30,46,0.85)",
        bordercolor="#45475a",
        borderwidth=1,
        borderpad=8,
        font={"size": 12, "color": "#cdd6f4"},
    )


# ─── Panel C — Live 72-Hour Forecast ──────────────────────────────────────────

def build_live_forecast_chart(
    current_aqi: float,
    aqi_24h: float,
    aqi_48h: float,
    aqi_72h: float,
    lower_24h: float,
    upper_24h: float,
    history_df: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Panel C: Live 72-hour forward forecast starting from "Now".

    The shaded band around the 24-hour prediction is a 90 % conformal
    prediction interval — computed with MAPIE on a held-out calibration
    set, not a Gaussian approximation.
    """
    now   = datetime.now(timezone.utc)
    times = [now + timedelta(hours=h) for h in [24, 48, 72]]
    preds = [aqi_24h, aqi_48h, aqi_72h]

    fig = go.Figure()
    _add_aqi_zone_bands(fig)

    # Last 24 h of history for context
    if history_df is not None and not history_df.empty and "aqi" in history_df.columns:
        hist = history_df.tail(24).copy()
        ts   = pd.to_datetime(hist.get("timestamp", pd.Series()), utc=True, errors="coerce")
        aqi  = pd.to_numeric(hist["aqi"], errors="coerce")
        fig.add_trace(go.Scatter(
            x=ts, y=aqi, mode="lines",
            name="Recent observed",
            line=dict(color="#89b4fa", width=1.5),
            hovertemplate="%{y:.0f}<extra>Observed</extra>",
        ))

    # "Now" marker
    fig.add_trace(go.Scatter(
        x=[now], y=[current_aqi], mode="markers",
        name="Now",
        marker=dict(size=12, color="#ffffff", symbol="circle",
                    line=dict(color="#89b4fa", width=2.5)),
        hovertemplate=f"Current: {current_aqi:.0f} AQI<extra>Now</extra>",
    ))

    # 90% conformal band for 24 h
    fig.add_trace(go.Scatter(
        x=[times[0], times[0]], y=[lower_24h, upper_24h],
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[times[0]], y=[(lower_24h + upper_24h) / 2],
        mode="none", fill="tonexty",
        fillcolor="rgba(250,179,135,0.25)",
        name="90% Conformal Interval (24 h)",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=times, y=preds, mode="lines+markers",
        name="Forecast (24 h / 48 h / 72 h)",
        line=dict(color="#fab387", width=2.5, dash="dash"),
        marker=dict(size=9, color="#fab387",
                    line=dict(color="#1e1e2e", width=1.5)),
        hovertemplate="%{y:.0f} AQI<extra>Forecast</extra>",
    ))

    # "Now" vertical separator
    fig.add_vline(x=now.timestamp() * 1000, line_dash="dot",
                  line_color="#6c7086",
                  annotation_text="Now → forecast",
                  annotation_position="top right",
                  annotation_font_color="#6c7086",
                  annotation_font_size=10)

    fig.update_layout(**_base_layout(
        "Real-Time Forecast — Next 72 Hours",
        "90 % interval = MAPIE conformal band (calibrated on held-out set, not Gaussian)",
    ))
    fig.update_layout(yaxis=dict(range=[0, max(500, max(preds) * 1.25)]))
    return fig


# ─── Legacy wrapper (used by older code paths) ────────────────────────────────

def build_forecast_chart(
    current_aqi: float,
    aqi_24h: float,
    aqi_48h: float,
    aqi_72h: float,
    lower_24h: float,
    upper_24h: float,
    history_df: pd.DataFrame | None = None,
) -> go.Figure:
    """Backwards-compatibility shim — delegates to Panel C."""
    return build_live_forecast_chart(
        current_aqi, aqi_24h, aqi_48h, aqi_72h,
        lower_24h, upper_24h, history_df,
    )
