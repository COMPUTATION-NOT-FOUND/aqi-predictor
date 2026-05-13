"""
Pearls AQI Predictor — Plotly Dash Dashboard
5 tabs: Live Forecast | Model Leaderboard | Ablation Study | SHAP | Data Drift

Run locally:  ./run_local.sh
Deploy:       Render free tier (see README.md Step 9)
"""
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.config import DEFAULT_CITY, AQI_ZONES, aqi_color, aqi_label
from src.dashboard.components.alerts import build_alert_banner
from src.dashboard.components.gauge import build_gauge
from src.dashboard.components.forecast_chart import (
    build_ground_truth_chart,
    build_hindcast_chart,
    build_live_forecast_chart,
    build_forecast_chart,   # backwards-compat shim
)
from src.dashboard.components.leaderboard import build_leaderboard
from src.dashboard.components.shap_plot import build_shap_bar
from src.dashboard.components.drift_tab import build_drift_heatmap
from src.dashboard.components.ablation_tab import build_ablation_charts

# ─── App Init ─────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME],
    title="AQI Predictor",
    suppress_callback_exceptions=True,
)
server = app.server   # expose Flask server for Render/Gunicorn

# ─── Shared loading spinner ───────────────────────────────────────────────────
_SPINNER = dbc.Spinner(
    color="primary",
    spinner_style={"width": "3rem", "height": "3rem"},
)

_LOADING_DIV = html.Div(
    [_SPINNER,
     html.P("Loading data from Hopsworks…",
            style={"color": "#a6adc8", "marginTop": "16px", "fontSize": "0.9rem"})],
    style={"display": "flex", "flexDirection": "column",
           "alignItems": "center", "justifyContent": "center",
           "minHeight": "40vh"},
)

# ─── Layout ───────────────────────────────────────────────────────────────────
app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": "#1e1e2e", "minHeight": "100vh", "padding": "0"},
    children=[
        # Header
        dbc.Row(dbc.Col(html.Div([
            html.H2("AQI Predictor", className="mb-0",
                    style={"color": "#cdd6f4", "fontWeight": "700"}),
            html.Small(f"City: {DEFAULT_CITY.title()} • Updates every hour",
                       style={"color": "#a6adc8"}),
        ]), width=12), style={"backgroundColor": "#181825", "padding": "18px 28px", "marginBottom": "0"}),

        # Refresh interval (every 5 min)
        dcc.Interval(id="interval-refresh", interval=5 * 60 * 1000, n_intervals=0),

        # Tabs
        dbc.Tabs(
            id="main-tabs",
            active_tab="tab-forecast",
            style={"backgroundColor": "#181825", "borderBottom": "1px solid #313244"},
            children=[
                dbc.Tab(label="Live Forecast",      tab_id="tab-forecast"),
                dbc.Tab(label="Leaderboard",        tab_id="tab-leaderboard"),
                dbc.Tab(label="Ablation Study",     tab_id="tab-ablation"),
                dbc.Tab(label="Feature Importance", tab_id="tab-shap"),
                dbc.Tab(label="Data Drift",         tab_id="tab-drift"),
            ],
        ),

        # Tab skeleton — rendered instantly; content filled by callback below
        html.Div(id="tab-content", style={"padding": "24px 28px"}),

        # Hidden trigger: fires once immediately after page load to populate forecast
        dcc.Store(id="tab-trigger"),
    ],
)

# ─── Tab skeleton (instant render) ────────────────────────────────────────────

@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab"),
    Input("interval-refresh", "n_intervals"),
)
def render_tab_skeleton(active_tab, _):
    """Render the tab shell immediately with a loading spinner.

    Each tab that requires Hopsworks data is split into:
      1. This callback → returns the shell + dcc.Loading wrapper instantly
      2. A second callback (below) → fills in the real content asynchronously
    """
    if active_tab == "tab-forecast":
        return html.Div([
            dcc.Loading(
                id="forecast-loading",
                type="circle",
                color="#89b4fa",
                children=html.Div(id="forecast-content"),
            ),
        ])
    elif active_tab == "tab-leaderboard":
        return html.Div([
            dcc.Loading(
                id="leaderboard-loading",
                type="circle",
                color="#89b4fa",
                children=html.Div(id="leaderboard-content"),
            ),
        ])
    elif active_tab == "tab-ablation":
        # Ablation reads only MLflow (local files) — no loading state needed
        return _render_ablation()
    elif active_tab == "tab-shap":
        return html.Div([
            dcc.Loading(
                id="shap-loading",
                type="circle",
                color="#89b4fa",
                children=html.Div(id="shap-content"),
            ),
        ])
    elif active_tab == "tab-drift":
        # Drift reads local log file — fast, no loading state needed
        return _render_drift()
    return html.Div("Unknown tab")


# ─── Deferred content callbacks (these do the heavy Hopsworks work) ───────────

@app.callback(
    Output("forecast-content", "children"),
    Input("forecast-content", "id"),   # fires once when element mounts
)
def fill_forecast(_):
    return _render_forecast()


@app.callback(
    Output("leaderboard-content", "children"),
    Input("leaderboard-content", "id"),
)
def fill_leaderboard(_):
    return _render_leaderboard()


@app.callback(
    Output("shap-content", "children"),
    Input("shap-content", "id"),
)
def fill_shap(_):
    return _render_shap()


# ─── Render helpers ───────────────────────────────────────────────────────────

def _render_forecast():
    from src.dashboard.inference import predict_3day, get_recent_features
    try:
        result = predict_3day()
    except Exception as e:
        result = {"error": str(e)}

    current = result.get("current_aqi", 0)
    a24     = result.get("aqi_24h", current)
    a48     = result.get("aqi_48h", current)
    a72     = result.get("aqi_72h", current)
    lo24    = result.get("lower_24h", a24 * 0.85)
    hi24    = result.get("upper_24h", a24 * 1.15)
    error   = result.get("error")

    try:
        history = get_recent_features()
    except Exception:
        history = None

    # ── "How to read" explanation card ────────────────────────────────────────
    how_to_read = html.Details([
        html.Summary("ℹ️  How to read these charts", style={
            "color": "#89b4fa", "fontWeight": "600",
            "fontSize": "0.88rem", "cursor": "pointer",
        }),
        html.Div([
            html.Ul([
                html.Li([
                    html.B("Panel A — Observed AQI: "),
                    "Raw PM2.5-derived AQI readings from AQICN + OpenMeteo sensors. "
                    "This is what actually happened — the ground truth.",
                ], style={"marginBottom": "6px"}),
                html.Li([
                    html.B("Panel B — Model Verification: "),
                    "OOF (out-of-fold) predictions overlaid on observed data. "
                    "The model never saw these samples during training — "
                    "this is a leakage-free check of how well it tracks reality. "
                    "Metric badges: RMSE (error magnitude), IoA (Willmott 1981 agreement, 0–1), "
                    "Skill (1 − RMSE_model/RMSE_persistence; > 0 = beats naive baseline).",
                ], style={"marginBottom": "6px"}),
                html.Li([
                    html.B("Panel C — Live Forecast: "),
                    "Forward-looking 72-hour prediction starting from Now. "
                    "Shaded band = 90% conformal prediction interval "
                    "(MAPIE-calibrated on held-out data, not a Gaussian approximation).",
                ]),
            ], style={"color": "#a6adc8", "fontSize": "0.83rem",
                      "lineHeight": "1.6", "paddingLeft": "18px"}),
        ], style={"padding": "10px 14px",
                  "backgroundColor": "#181825",
                  "borderRadius": "6px",
                  "marginTop": "8px"}),
    ], open=False, style={
        "backgroundColor": "#1e1e2e",
        "border": "1px solid #313244",
        "borderRadius": "8px",
        "padding": "10px 14px",
        "marginBottom": "16px",
    })

    # ── AQI zone legend ───────────────────────────────────────────────────────
    zone_legend = dbc.Row(dbc.Col(html.Div([
        html.Span(f"  {label}: {lo}–{hi}  ",
                  style={"backgroundColor": color, "color": "#000", "padding": "2px 8px",
                         "borderRadius": "4px", "marginRight": "6px", "fontSize": "0.75rem",
                         "fontWeight": "bold"})
        for lo, hi, label, color in AQI_ZONES
    ]), width=12), className="mb-3")

    return dbc.Container(fluid=True, children=[
        build_alert_banner(max(current, a24, a48, a72)),

        # Gauges
        dbc.Row([
            dbc.Col(dcc.Graph(figure=build_gauge(current, "Current AQI"),
                              config={"displayModeBar": False}), md=3),
            dbc.Col(dcc.Graph(figure=build_gauge(a24, "24h Forecast"),
                              config={"displayModeBar": False}), md=3),
            dbc.Col(dcc.Graph(figure=build_gauge(a48, "48h Forecast"),
                              config={"displayModeBar": False}), md=3),
            dbc.Col(dcc.Graph(figure=build_gauge(a72, "72h Forecast"),
                              config={"displayModeBar": False}), md=3),
        ], className="mb-3"),

        how_to_read,
        zone_legend,

        # Panel A — Observed ground truth
        dbc.Row(dbc.Col(
            dcc.Graph(
                id="chart-ground-truth",
                figure=build_ground_truth_chart(history),
                config={"displayModeBar": True},
                style={"height": "300px"},
            ), width=12,
        ), className="mb-3"),

        # Panel B — Model hindcast (OOF predictions vs observed)
        dbc.Row(dbc.Col(
            dcc.Graph(
                id="chart-hindcast",
                figure=build_hindcast_chart(history, None),  # OOF df populated from MLflow if available
                config={"displayModeBar": True},
                style={"height": "300px"},
            ), width=12,
        ), className="mb-3"),

        # Panel C — Live 72-hour forecast
        dbc.Row(dbc.Col(
            dcc.Graph(
                id="chart-live-forecast",
                figure=build_live_forecast_chart(current, a24, a48, a72, lo24, hi24, history),
                config={"displayModeBar": True},
                style={"height": "320px"},
            ), width=12,
        ), className="mb-3"),

        html.Div(f"⚠ {error}",
                 style={"color": "#f38ba8", "marginTop": "8px", "fontSize": "0.85rem"})
        if error else html.Div(),
    ])


def _render_leaderboard():
    from src.dashboard.inference import load_all_models_metadata
    try:
        metadata = load_all_models_metadata()
    except Exception as e:
        metadata = [{"name": f"Error: {e}", "version": "-"}]
    return dbc.Container(fluid=True, children=[build_leaderboard(metadata)])


def _render_ablation():
    return dbc.Container(fluid=True, children=[
        html.H4("Ablation Study Results", style={"color": "#cdd6f4"}),
        html.P("Results auto-update every night after the training pipeline runs.",
               style={"color": "#a6adc8", "fontSize": "0.85rem", "marginBottom": "16px"}),
        dcc.Graph(figure=build_ablation_charts(), style={"height": "450px"}),
        dbc.Row([
            dbc.Col(html.Div([
                html.H6("Feature Ablation", style={"color": "#cdd6f4"}),
                html.P("Each bar shows how much RMSE increases when that feature group is removed. "
                       "Taller bar = feature group is more important.",
                       style={"color": "#a6adc8", "fontSize": "0.82rem"}),
            ]), md=6),
            dbc.Col(html.Div([
                html.H6("Backfill Strategy", style={"color": "#cdd6f4"}),
                html.P(
                    "Compares 5 imputation methods for missing sensor data. "
                    "Gold bar = best strategy (lowest RMSE). "
                    "This panel populates after running the backfill ablation experiment.",
                    style={"color": "#a6adc8", "fontSize": "0.82rem"},
                ),
            ]), md=6),
        ]),
    ])


def _render_shap():
    from src.dashboard.inference import load_shap_data
    try:
        shap_data = load_shap_data()
    except Exception:
        shap_data = {}
    return dbc.Container(fluid=True, children=[
        html.H4("SHAP Feature Importance", style={"color": "#cdd6f4"}),
        html.P("Why did the model predict what it predicted? SHAP shows the contribution of each feature.",
               style={"color": "#a6adc8", "fontSize": "0.85rem", "marginBottom": "16px"}),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=build_shap_bar(shap_data, top_n=15),
                              style={"height": "450px"}), md=8),
            dbc.Col(html.Div([
                html.H6("How to read this", style={"color": "#cdd6f4"}),
                html.Ul([
                    html.Li("Longer bar = feature has more influence on predictions",
                            style={"color": "#a6adc8", "fontSize": "0.82rem"}),
                    html.Li("Red bars = top-importance features",
                            style={"color": "#a6adc8", "fontSize": "0.82rem"}),
                    html.Li("SHAP values computed on the best non-LSTM model (tree explainer)",
                            style={"color": "#a6adc8", "fontSize": "0.82rem"}),
                ]),
            ]), md=4),
        ]),
    ])


def _render_drift():
    return dbc.Container(fluid=True, children=[
        html.H4("Data Drift Monitor", style={"color": "#cdd6f4"}),
        html.P([
            "PSI < 0.1: no drift  |  PSI 0.1–0.2: moderate (yellow)  |  PSI > 0.2: alert (red — email sent)"
        ], style={"color": "#a6adc8", "fontSize": "0.85rem", "marginBottom": "16px"}),
        dcc.Graph(figure=build_drift_heatmap(), style={"height": "500px"}),
    ])


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
