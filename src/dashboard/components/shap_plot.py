import numpy as np
import plotly.graph_objects as go


def build_shap_bar(shap_data: dict, top_n: int = 10) -> go.Figure:
    if not shap_data or "shap_values" not in shap_data:
        return go.Figure().update_layout(
            title="SHAP values not available (run training pipeline first)",
            paper_bgcolor="#1e1e2e", font=dict(color="#cdd6f4"),
        )

    sv    = np.asarray(shap_data["shap_values"])
    names = shap_data.get("feature_names", [f"f{i}" for i in range(sv.shape[-1])])

    # Mean absolute SHAP across samples (and targets if multi-output)
    if sv.ndim == 3:
        mean_shap = np.mean(np.abs(sv), axis=(0, 1))
    else:
        mean_shap = np.mean(np.abs(sv), axis=0)

    if len(mean_shap) != len(names):
        names = [f"f{i}" for i in range(len(mean_shap))]

    idx  = np.argsort(mean_shap)[-top_n:]
    vals = mean_shap[idx]
    feat = [names[i] for i in idx]

    colors = [
        "#f38ba8" if v > np.percentile(vals, 70) else
        "#fab387" if v > np.percentile(vals, 40) else
        "#a6e3a1"
        for v in vals
    ]

    fig = go.Figure(go.Bar(
        x=vals, y=feat, orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"SHAP Feature Importance (Top {top_n})",
        xaxis_title="Mean |SHAP value|",
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        margin=dict(l=20, r=20, t=50, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return fig
