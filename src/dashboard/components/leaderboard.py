"""
Leaderboard helpers — Model Registry display (framework-agnostic).

Exposes:
- METRIC_GLOSSARY      : list[dict] defining every metric (for an expander/table)
- leaderboard_dataframe: clean a metadata list into (DataFrame, champion_row|None)
- build_radar_chart    : multi-metric radar (go.Figure) for the top-5 models

Rendering (tables, badges) is done by the Streamlit app; this module returns data
and Plotly figures only — no Dash/Streamlit imports.

Metric standards
----------------
RMSE / OOF RMSE : Root Mean Square Error (AQI units); OOF on held-out CV folds.
IoA             : Index of Agreement — Willmott (1981), range 0–1.
Skill Score     : 1 − RMSE_model / RMSE_persistence — Murphy (1988); > 0 required.
TOPSIS          : Multi-criteria champion score — Hwang & Yoon (1981).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

_DARK_BG = "#1e1e2e"

METRIC_GLOSSARY = [
    {"Metric": "RMSE",        "Formula": "√mean((obs − pred)²)",
     "Range": "0 → ∞ (lower better)",
     "Interpretation": "Average prediction error in AQI units. Penalises large errors."},
    {"Metric": "OOF RMSE",    "Formula": "RMSE on held-out CV folds",
     "Range": "0 → ∞ (lower better)",
     "Interpretation": "Leakage-proof error — the primary model-selection metric."},
    {"Metric": "RMSE d1/d2/d3", "Formula": "RMSE per horizon (24/48/72h)",
     "Range": "0 → ∞ (lower better)",
     "Interpretation": "How error grows with lead time; d1 (24h) is usually lowest."},
    {"Metric": "IoA",         "Formula": "1 − Σ(obs−pred)² / Σ(|pred−ō| + |obs−ō|)²",
     "Range": "0 – 1 (higher better)",
     "Interpretation": "Willmott (1981). > 0.3 acceptable for urban AQI. Hard gate ≥ 0.15."},
    {"Metric": "Skill Score", "Formula": "1 − RMSE_model / RMSE_persistence",
     "Range": "−∞ – 1 (> 0 required)",
     "Interpretation": "Murphy (1988). > 0 beats the per-sample 'tomorrow = today' baseline."},
    {"Metric": "Overfit",     "Formula": "train_RMSE / val_RMSE < 0.7",
     "Range": "True / False",
     "Interpretation": "True ⇒ memorised training data; ineligible for champion."},
    {"Metric": "TOPSIS",      "Formula": "S⁻ / (S⁺ + S⁻)",
     "Range": "0 – 1 (higher better)",
     "Interpretation": "Hwang & Yoon (1981) MCDM across OOF RMSE, IoA, Skill, MAE."},
]

# Columns shown in the rankings table, in order, with display names.
_DISPLAY_ORDER = [
    ("name", "Name"), ("version", "Ver."), ("status", "Status"),
    ("topsis_score", "TOPSIS ↑"), ("rmse", "RMSE"), ("oof_rmse", "OOF RMSE"),
    ("rmse_d1", "RMSE d1"), ("rmse_d2", "RMSE d2"), ("rmse_d3", "RMSE d3"),
    ("ioa", "IoA"), ("skill_score", "Skill"), ("overfit", "Overfit?"),
]


def leaderboard_dataframe(models_metadata: list[dict]):
    """Return (display_df, champion_row_dict|None) from a raw metadata list."""
    if not models_metadata:
        return pd.DataFrame(), None

    df = pd.DataFrame(models_metadata)
    for col in ["rmse", "ioa", "skill_score", "oof_rmse", "topsis_score", "mae",
                "rmse_d1", "rmse_d2", "rmse_d3"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(3)

    if "is_champion" not in df.columns:
        df["is_champion"] = False
    df["is_champion"] = df["is_champion"].fillna(False).astype(bool)
    df["status"] = df["is_champion"].map(lambda x: "🏆 CHAMPION" if x else "challenger")
    if "overfit" not in df.columns:
        df["overfit"] = False

    # Sort: champion first, then best TOPSIS, then lowest RMSE.
    sort_cols, ascending = [], []
    if "topsis_score" in df.columns:
        sort_cols.append("topsis_score"); ascending.append(False)
    if "rmse" in df.columns:
        sort_cols.append("rmse"); ascending.append(True)
    if sort_cols:
        df = df.sort_values(by=["is_champion"] + sort_cols, ascending=[False] + ascending)

    champ = df[df["is_champion"]]
    champion_row = champ.iloc[0].to_dict() if not champ.empty else None

    present = [(c, label) for c, label in _DISPLAY_ORDER if c in df.columns]
    display_df = df[[c for c, _ in present]].rename(columns=dict(present)).reset_index(drop=True)
    return display_df, champion_row


def build_radar_chart(models_metadata: list[dict]) -> go.Figure | None:
    """Multi-metric radar for the top-5 models (None if <2 models)."""
    df = pd.DataFrame(models_metadata)
    if len(df) < 2 or "name" not in df.columns:
        return None

    numeric_cols = ["oof_rmse", "ioa", "skill_score", "topsis_score"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "is_champion" not in df.columns:
        df["is_champion"] = False
    if "version" not in df.columns:
        df["version"] = "-"

    max_oof = df["oof_rmse"].replace(0, np.nan).max()
    df["oof_rmse_inv"] = (1.0 - df["oof_rmse"] / max_oof) if (max_oof and max_oof > 0) else 0.5

    axes_raw = ["oof_rmse_inv", "ioa", "skill_score", "topsis_score"]
    axes_labels = ["OOF RMSE↑", "IoA", "Skill", "TOPSIS"]
    for col in axes_raw:
        lo, hi = df[col].min(), df[col].max()
        rng = hi - lo if hi - lo > 1e-9 else 1.0
        df[col + "_n"] = (df[col] - lo) / rng
    axes_norm = [c + "_n" for c in axes_raw]

    top5 = df.sort_values("topsis_score", ascending=False).drop_duplicates("name").head(5)
    colors = ["#cba6f7", "#89b4fa", "#a6e3a1", "#fab387", "#f38ba8"]

    fig = go.Figure()
    for i, (_, row) in enumerate(top5.iterrows()):
        vals = [float(row[c]) for c in axes_norm]
        vals += [vals[0]]
        label = ("🏆 " if row.get("is_champion") else "") + f"{row['name']} v{row['version']}"
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=axes_labels + [axes_labels[0]], fill="toself", name=label,
            line=dict(color=colors[i % len(colors)], width=2), opacity=0.85,
        ))

    fig.update_layout(
        polar=dict(bgcolor=_DARK_BG,
                   radialaxis=dict(visible=True, range=[0, 1], gridcolor="#313244",
                                   tickfont={"size": 9, "color": "#6c7086"}),
                   angularaxis=dict(gridcolor="#313244",
                                    tickfont={"size": 10, "color": "#a6adc8"})),
        paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_BG, font={"color": "#cdd6f4"},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"size": 10, "color": "#cdd6f4"}},
        title=dict(text="<b>Multi-Metric Comparison — Top 5</b>"
                        "<br><sup style='color:#6c7086'>Axes normalised to [0,1] · larger = better</sup>",
                   font={"size": 13, "color": "#cdd6f4"}),
        margin=dict(l=60, r=60, t=70, b=40), height=420,
    )
    return fig
