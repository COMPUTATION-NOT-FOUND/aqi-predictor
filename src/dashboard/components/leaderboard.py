from dash import html, dash_table
import pandas as pd


def build_leaderboard(models_metadata: list[dict]) -> html.Div:
    if not models_metadata:
        return html.Div("No models registered yet.", style={"color": "#a6adc8"})

    df = pd.DataFrame(models_metadata)

    # Round numeric columns
    for col in ["rmse", "ioa", "skill_score", "oof_rmse"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(3)

    df["status"] = df.apply(lambda r: (
        "CHAMPION" if r.get("is_champion") else "challenger"
    ), axis=1)

    display_cols = ["name", "version", "rmse", "oof_rmse", "ioa", "skill_score", "overfit", "status"]
    display_cols = [c for c in display_cols if c in df.columns]

    style_data_conditional = [
        {
            "if": {"filter_query": '{status} = "CHAMPION"'},
            "backgroundColor": "#2a2000",
            "color": "#f5c842",
            "fontWeight": "bold",
        },
        {
            "if": {"filter_query": "{overfit} = True"},
            "color": "#f38ba8",
        },
    ]

    table = dash_table.DataTable(
        data=df[display_cols].to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in display_cols],
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#313244",
            "color": "#cdd6f4",
            "fontWeight": "bold",
            "border": "1px solid #45475a",
        },
        style_data={
            "backgroundColor": "#1e1e2e",
            "color": "#cdd6f4",
            "border": "1px solid #313244",
        },
        style_data_conditional=style_data_conditional,
        style_cell={"padding": "8px 12px", "fontFamily": "monospace"},
        page_size=20,
        tooltip_data=[
            {c: {"value": f"Overfit detected: train RMSE much lower than val RMSE", "type": "markdown"}
             for c in display_cols if c == "overfit" and row.get("overfit")}
            for row in df[display_cols].to_dict("records")
        ],
    )

    return html.Div([
        html.H4("Model Registry Leaderboard", style={"color": "#cdd6f4", "marginBottom": "8px"}),
        html.P("CHAMPION row in gold. Red text = overfitting detected (train RMSE << val RMSE).",
               style={"color": "#a6adc8", "fontSize": "0.85rem", "marginBottom": "12px"}),
        table,
    ])
