"""
Leaderboard tab — Model Registry display.

Sections
--------
1. Champion badge  — highlights the currently deployed champion model
2. Metric glossary — defines every metric with formula and interpretation
3. Radar chart     — multi-metric comparison of top-5 models at a glance
4. Rankings table  — full model registry, sortable, with TOPSIS score column

Metric standards used
---------------------
RMSE       : Root Mean Square Error (AQI units); lower = better
OOF RMSE   : Same but on held-out cross-validation folds (leakage-proof)
IoA        : Index of Agreement — Willmott (1981), range 0–1
             Used as the standard agreement metric in environmental modeling
Skill Score: 1 − RMSE_model / RMSE_persistence — Murphy (1988)
             > 0 = beats persistence; < 0 = worse than naive repeat
Overfit    : True if train_RMSE / val_RMSE < 0.7 (model memorised training data)
TOPSIS     : Multi-criteria champion score — Hwang & Yoon (1981)
             Euclidean distance to ideal-best across OOF RMSE, IoA, Skill, MAE
             Range 0–1; used in FAIRMODE air-quality model benchmarking
"""
from dash import html, dash_table, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# ─── Metric glossary definitions ──────────────────────────────────────────────
_METRIC_GLOSSARY = [
    {
        "Metric":      "RMSE",
        "Formula":     "√mean((obs − pred)²)",
        "Range":       "0 → ∞  (lower = better)",
        "Interpretation": "Average prediction error in AQI units. Penalises large errors more than MAE.",
    },
    {
        "Metric":      "OOF RMSE",
        "Formula":     "RMSE on held-out CV folds",
        "Range":       "0 → ∞  (lower = better)",
        "Interpretation": "Leakage-proof error. Primary model-selection metric — the model "
                          "never sees these samples during training.",
    },
    {
        "Metric":      "IoA",
        "Formula":     "1 − Σ(obs−pred)² / Σ(|pred−ō| + |obs−ō|)²",
        "Range":       "0 – 1  (higher = better)",
        "Interpretation": "Willmott (1981) Index of Agreement. Measures both bias and scatter. "
                          "> 0.3 is acceptable for urban AQI with limited training data.",
    },
    {
        "Metric":      "Skill Score",
        "Formula":     "1 − RMSE_model / RMSE_persistence",
        "Range":       "−∞ – 1  (> 0 required)",
        "Interpretation": "Murphy (1988). Positive = beats the 'repeat yesterday' baseline. "
                          "Negative = worse than doing nothing. Gate for champion eligibility.",
    },
    {
        "Metric":      "Overfit",
        "Formula":     "train_RMSE / val_RMSE < 0.7",
        "Range":       "True / False",
        "Interpretation": "True if the model memorised training data and will generalise poorly. "
                          "Overfit models are ineligible for the champion slot.",
    },
    {
        "Metric":      "TOPSIS Score",
        "Formula":     "S⁻ / (S⁺ + S⁻)  where S = Euclidean dist to ideal",
        "Range":       "0 – 1  (higher = better)",
        "Interpretation": "Hwang & Yoon (1981) MCDM score across OOF RMSE, IoA, Skill, MAE. "
                          "Used in FAIRMODE air-quality benchmarking. Champion = highest scorer "
                          "that passes all hard gates.",
    },
]


def _champion_badge(df: pd.DataFrame) -> html.Div:
    """Render a prominent card for the current champion model."""
    champ = df[df["is_champion"] == True]  # noqa: E712
    if champ.empty:
        return html.Div(
            "No champion model deployed yet.",
            style={"color": "#6c7086", "fontSize": "0.9rem", "marginBottom": "16px"},
        )
    row = champ.iloc[0]

    def _val(key, fmt=".3f"):
        v = row.get(key)
        try:
            return f"{float(v):{fmt}}"
        except Exception:
            return "—"

    topsis_val = _val("topsis_score")
    ioa_val    = _val("ioa")
    skill_val  = _val("skill_score")
    rmse_val   = _val("rmse", ".1f")
    oof_val    = _val("oof_rmse", ".1f")

    return html.Div([
        html.Div("👑  CURRENT CHAMPION", style={
            "fontSize": "0.7rem", "fontWeight": "700",
            "letterSpacing": "0.12em", "color": "#f5c842",
            "marginBottom": "6px",
        }),
        html.Div([
            html.Span(f"{row['name']}", style={
                "fontSize": "1.5rem", "fontWeight": "800", "color": "#f5c842",
                "marginRight": "12px",
            }),
            html.Span(f"v{row['version']}", style={
                "fontSize": "0.85rem", "color": "#a6adc8",
                "border": "1px solid #45475a", "borderRadius": "4px",
                "padding": "2px 8px",
            }),
        ]),
        html.Div([
            _badge("TOPSIS", topsis_val, "#cba6f7"),
            _badge("IoA",    ioa_val,    "#89dceb"),
            _badge("Skill",  skill_val,  "#a6e3a1"),
            _badge("RMSE",   rmse_val,   "#89b4fa"),
            _badge("OOF",    oof_val,    "#89b4fa"),
        ], style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginTop": "10px"}),
    ], style={
        "backgroundColor": "#2a2000",
        "border": "1px solid #f5c842",
        "borderRadius": "8px",
        "padding": "16px 20px",
        "marginBottom": "20px",
    })


def _badge(label: str, value: str, color: str) -> html.Div:
    return html.Div([
        html.Div(label, style={"fontSize": "0.65rem", "color": "#6c7086",
                                "letterSpacing": "0.08em", "fontWeight": "600"}),
        html.Div(value, style={"fontSize": "1rem", "fontWeight": "700", "color": color}),
    ], style={
        "backgroundColor": "#181825",
        "border": "1px solid #313244",
        "borderRadius": "6px",
        "padding": "6px 12px",
        "minWidth": "72px",
        "textAlign": "center",
    })


def _metric_glossary_card() -> html.Div:
    """Render the metric definition table."""
    glossary_df = pd.DataFrame(_METRIC_GLOSSARY)
    table = dash_table.DataTable(
        data=glossary_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in glossary_df.columns],
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#181825", "color": "#cba6f7",
            "fontWeight": "700", "border": "1px solid #313244",
            "fontSize": "0.8rem", "letterSpacing": "0.05em",
        },
        style_data={
            "backgroundColor": "#1e1e2e", "color": "#cdd6f4",
            "border": "1px solid #313244", "whiteSpace": "normal",
            "height": "auto", "fontSize": "0.82rem",
        },
        style_cell={"padding": "8px 12px", "fontFamily": "monospace",
                    "textAlign": "left", "verticalAlign": "top"},
        style_cell_conditional=[
            {"if": {"column_id": "Metric"},  "width": "110px", "fontWeight": "700", "color": "#89b4fa"},
            {"if": {"column_id": "Formula"}, "width": "200px", "fontFamily": "monospace"},
            {"if": {"column_id": "Range"},   "width": "150px"},
        ],
    )
    return html.Details([
        html.Summary("📖  Metric Definitions & Standards", style={
            "color": "#cba6f7", "fontWeight": "600", "fontSize": "0.9rem",
            "cursor": "pointer", "marginBottom": "8px",
        }),
        table,
    ], open=False, style={"marginBottom": "20px"})


def _radar_chart(df: pd.DataFrame) -> go.Figure:
    """
    Multi-metric radar chart for the top-5 models.

    Axes: OOF RMSE (inverted), IoA, Skill Score, TOPSIS score.
    All axes normalised to [0, 1] so the chart is visually comparable.
    """
    numeric_cols = ["oof_rmse", "ioa", "skill_score", "topsis_score"]
    plot_df = df.copy()

    # Fill missing topsis_score with 0
    for col in numeric_cols:
        if col not in plot_df.columns:
            plot_df[col] = np.nan
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce").fillna(0)

    # Invert OOF RMSE so "higher = better" on all axes
    max_oof = plot_df["oof_rmse"].replace(0, np.nan).max()
    if max_oof and max_oof > 0:
        plot_df["oof_rmse_inv"] = 1.0 - (plot_df["oof_rmse"] / max_oof)
    else:
        plot_df["oof_rmse_inv"] = 0.5

    # Normalise each axis to [0, 1]
    axes_raw = ["oof_rmse_inv", "ioa", "skill_score", "topsis_score"]
    axes_labels = ["OOF RMSE↑\n(inverted)", "IoA", "Skill Score", "TOPSIS"]

    for col in axes_raw:
        lo, hi = plot_df[col].min(), plot_df[col].max()
        rng = hi - lo if hi - lo > 1e-9 else 1.0
        plot_df[col + "_n"] = (plot_df[col] - lo) / rng

    axes_norm = [c + "_n" for c in axes_raw]

    # Top-5 by TOPSIS score (or all if fewer)
    top5 = (
        plot_df.sort_values("topsis_score", ascending=False)
               .drop_duplicates(subset=["name"])
               .head(5)
    )

    _COLORS = ["#cba6f7", "#89b4fa", "#a6e3a1", "#fab387", "#f38ba8"]
    fig = go.Figure()

    for i, (_, row) in enumerate(top5.iterrows()):
        vals = [float(row[c]) for c in axes_norm]
        vals += [vals[0]]   # close the polygon
        label = f"{row['name']} v{row['version']}"
        if row.get("is_champion"):
            label = "👑 " + label

        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=axes_labels + [axes_labels[0]],
            fill="toself",
            name=label,
            line=dict(color=_COLORS[i % len(_COLORS)], width=2),
            fillcolor=_COLORS[i % len(_COLORS)].replace(")", ",0.08)").replace("rgb", "rgba"),
            opacity=0.9,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=_DARK_BG,
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor="#313244", linecolor="#45475a",
                tickfont={"size": 9, "color": "#6c7086"},
            ),
            angularaxis=dict(gridcolor="#313244", linecolor="#45475a",
                             tickfont={"size": 10, "color": "#a6adc8"}),
        ),
        paper_bgcolor=_DARK_BG,
        plot_bgcolor=_DARK_BG,
        font={"color": "#cdd6f4"},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"size": 10, "color": "#cdd6f4"}},
        title=dict(
            text="<b>Multi-Metric Comparison — Top 5 Models</b>"
                 "<br><sup style='color:#6c7086'>All axes normalised to [0,1] · "
                 "Larger area = better overall performance</sup>",
            font={"size": 13, "color": "#cdd6f4"},
        ),
        margin=dict(l=60, r=60, t=70, b=40),
        height=420,
    )
    return fig


_DARK_BG = "#1e1e2e"


def build_leaderboard(models_metadata: list[dict]) -> html.Div:
    if not models_metadata:
        return html.Div("No models registered yet.", style={"color": "#a6adc8"})

    df = pd.DataFrame(models_metadata)

    # Normalise numeric columns
    for col in ["rmse", "ioa", "skill_score", "oof_rmse", "topsis_score", "mae"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    df["is_champion"] = df.apply(lambda r: r.get("is_champion", False), axis=1)
    df["status"]      = df["is_champion"].apply(lambda x: "CHAMPION" if x else "challenger")
    df["overfit"]     = df.get("overfit", False) if "overfit" in df.columns else False

    # Display columns — include topsis_score if present
    base_cols    = ["name", "version", "status", "topsis_score", "rmse", "oof_rmse",
                    "ioa", "skill_score", "overfit"]
    display_cols = [c for c in base_cols if c in df.columns]

    # Column rename map for readability
    col_rename = {
        "oof_rmse":    "OOF RMSE",
        "ioa":         "IoA",
        "skill_score": "Skill Score",
        "topsis_score":"TOPSIS ↑",
        "rmse":        "RMSE",
        "overfit":     "Overfit?",
        "status":      "Status",
        "name":        "Name",
        "version":     "Ver.",
    }

    style_data_conditional = [
        {
            "if": {"filter_query": '{status} = "CHAMPION"'},
            "backgroundColor": "#2a2000",
            "color": "#f5c842",
            "fontWeight": "bold",
        },
        {
            "if": {"filter_query": "{overfit} = True or {overfit} = true"},
            "color": "#f38ba8",
        },
        {
            "if": {"filter_query": "{skill_score} < 0"},
            "color": "#f38ba8",
        },
    ]

    table = dash_table.DataTable(
        id="leaderboard-table",
        data=df[display_cols].to_dict("records"),
        columns=[{
            "name": col_rename.get(c, c.replace("_", " ").title()),
            "id": c
        } for c in display_cols],
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#313244",
            "color": "#cdd6f4",
            "fontWeight": "bold",
            "border": "1px solid #45475a",
            "fontSize": "0.82rem",
        },
        style_data={
            "backgroundColor": "#1e1e2e",
            "color": "#cdd6f4",
            "border": "1px solid #313244",
            "fontSize": "0.82rem",
        },
        style_data_conditional=style_data_conditional,
        style_cell={"padding": "8px 12px", "fontFamily": "monospace"},
        tooltip_header={
            "topsis_score": "TOPSIS: multi-criteria champion score (higher = better). "
                            "Computed via Hwang & Yoon (1981) MCDM across OOF RMSE, IoA, Skill, MAE.",
            "ioa":          "Index of Agreement (Willmott 1981). Range 0–1. "
                            "> 0.3 acceptable for urban AQI. Hard gate: ≥ 0.15.",
            "skill_score":  "Murphy (1988). > 0 = beats persistence. Hard gate: must be > 0.",
            "oof_rmse":     "Out-of-fold RMSE: leakage-proof cross-validation error. "
                            "Primary selection metric.",
            "overfit":      "True if train_RMSE/val_RMSE < 0.7 — model memorised training data. "
                            "Overfit models are ineligible for champion.",
        },
        tooltip_delay=0,
        tooltip_duration=None,
        page_size=20,
    )

    # Radar chart (only if we have > 1 model)
    radar_section = []
    if len(df) > 1:
        radar_section = [
            dcc.Graph(
                figure=_radar_chart(df),
                config={"displayModeBar": False},
                style={"marginBottom": "20px"},
            )
        ]

    return html.Div([
        html.H4("Model Registry Leaderboard",
                style={"color": "#cdd6f4", "marginBottom": "4px",
                       "fontWeight": "700", "fontSize": "1.1rem"}),
        html.P(
            "Champion selected via TOPSIS multi-criteria scoring (OOF RMSE, IoA, Skill Score, MAE). "
            "Hard gates: Skill > 0, IoA ≥ 0.15, not overfit. "
            "Red text = skill score negative (worse than persistence).",
            style={"color": "#6c7086", "fontSize": "0.8rem", "marginBottom": "16px"},
        ),

        # Champion badge
        _champion_badge(df),

        # Metric glossary (collapsed by default)
        _metric_glossary_card(),

        # Radar chart
        *radar_section,

        # Rankings table
        html.H6("Full Model Registry", style={"color": "#a6adc8",
                "fontWeight": "600", "marginBottom": "8px"}),
        table,
    ])
