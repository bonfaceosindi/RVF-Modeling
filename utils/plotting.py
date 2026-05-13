"""All Plotly visualisation functions for the RVF Streamlit app."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

C = dict(
    S="#2196F3", E="#FF9800", I="#F44336", R="#4CAF50",
    Sv="#64B5F6", Ev="#FFB74D", Iv="#E57373",
    rain="#81D4FA", new_inf="#AB47BC", obs="#7E57C2",
    total_v="#26C6DA", foi="#FF7043",
    human_S="#42A5F5", human_E="#FFA726", human_I="#EF5350", human_R="#66BB6A",
    baseline="#1565C0", intervention="#C62828",
)


def _days(n: int) -> np.ndarray:
    return np.arange(1, n + 1)


def _band(fig, days, lo, hi, color_rgb, row=1, col=1):
    """Add a shaded confidence band to an existing figure."""
    days_list = days.tolist()
    fig.add_trace(go.Scatter(
        x=days_list + days_list[::-1],
        y=np.concatenate([hi, lo[::-1]]).tolist(),
        fill="toself", fillcolor=f"rgba({color_rgb},0.18)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip", showlegend=False,
    ), row=row, col=col)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-RUN PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_overview(out: Dict[str, np.ndarray], raw_rain: np.ndarray) -> go.Figure:
    T = len(out["Sh"])
    days = _days(T)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.22, 0.42, 0.36], vertical_spacing=0.04,
                        subplot_titles=["Rainfall (mm/day)", "Host SEIR Dynamics", "Vector Dynamics"])
    fig.add_trace(go.Bar(x=days, y=raw_rain[:T], name="Rainfall",
                         marker_color=C["rain"], opacity=0.8), row=1, col=1)
    for key, label in [("Sh", "Susceptible"), ("Eh", "Exposed"), ("Ih", "Infectious"), ("Rh", "Recovered")]:
        fig.add_trace(go.Scatter(x=days, y=out[key], name=label,
                                 line=dict(color=C[key[0]], width=2),
                                 hovertemplate="%{y:,.0f}<extra>" + label + "</extra>"), row=2, col=1)
    for key, label in [("Sv", "Sv"), ("Ev", "Ev"), ("Iv", "Iv")]:
        fig.add_trace(go.Scatter(x=days, y=out[key], name=label,
                                 line=dict(color=C[key], width=2),
                                 hovertemplate="%{y:,.0f}<extra>" + label + "</extra>"), row=3, col=1)
    fig.update_xaxes(title_text="Day", row=3, col=1)
    fig.update_yaxes(title_text="mm", row=1, col=1)
    fig.update_yaxes(title_text="Animals", row=2, col=1)
    fig.update_yaxes(title_text="Mosquitoes", row=3, col=1)
    fig.update_layout(height=650, template="plotly_white", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1))
    return fig


def plot_host_dynamics(out: Dict[str, np.ndarray]) -> go.Figure:
    T = len(out["Sh"])
    days = _days(T)
    Nh = out["Sh"][0] + out["Eh"][0] + out["Ih"][0] + out["Rh"][0]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["SEIR Compartments (stacked area)", "Daily New Infections",
                                        "Cumulative Infections", "Attack Rate Over Time (%)"],
                        vertical_spacing=0.14, horizontal_spacing=0.10)
    for key, label in [("Sh", "Susceptible"), ("Rh", "Recovered"), ("Eh", "Exposed"), ("Ih", "Infectious")]:
        fig.add_trace(go.Scatter(x=days, y=out[key], name=label,
                                 stackgroup="seir", line=dict(width=0.5, color=C[key[0]]),
                                 fillcolor=C[key[0]], opacity=0.75,
                                 hovertemplate="%{y:,.0f}<extra>" + label + "</extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=days, y=out["new_inf_h"], name="New infections",
                         marker_color=C["new_inf"], opacity=0.85,
                         hovertemplate="Day %{x}: %{y:,.0f}<extra></extra>"), row=1, col=2)
    cumulative = np.cumsum(out["new_inf_h"])
    fig.add_trace(go.Scatter(x=days, y=cumulative, name="Cumulative",
                             line=dict(color=C["new_inf"], width=2.5),
                             fill="tozeroy", fillcolor="rgba(171,71,188,0.15)",
                             hovertemplate="Day %{x}: %{y:,.0f}<extra></extra>"), row=2, col=1)
    attack_rate = cumulative / max(Nh, 1) * 100
    fig.add_trace(go.Scatter(x=days, y=attack_rate, name="Attack rate",
                             line=dict(color=C["R"], width=2.5),
                             hovertemplate="Day %{x}: %{y:.2f}%<extra></extra>"), row=2, col=2)
    fig.add_annotation(x=days[-1], y=float(attack_rate[-1]),
                       text=f"  Final: {attack_rate[-1]:.1f}%",
                       showarrow=False, font=dict(color=C["R"], size=11),
                       xanchor="left", row=2, col=2)
    fig.update_xaxes(title_text="Day")
    fig.update_yaxes(title_text="Animals", row=1, col=1)
    fig.update_yaxes(title_text="New infections", row=1, col=2)
    fig.update_yaxes(title_text="Total infections", row=2, col=1)
    fig.update_yaxes(title_text="%", row=2, col=2)
    fig.update_layout(height=600, template="plotly_white", showlegend=False)
    return fig


def plot_vector_dynamics(out: Dict[str, np.ndarray]) -> go.Figure:
    T = len(out["Sv"])
    days = _days(T)
    total_v = out["Sv"] + out["Ev"] + out["Iv"]
    inf_frac = np.where(total_v > 0, out["Iv"] / total_v * 100, 0.0)
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Vector Compartments (Sv / Ev / Iv)",
                                        "Total Mosquito Population", "Infectious Fraction (%)"],
                        horizontal_spacing=0.10)
    for key, label in [("Sv", "Sv"), ("Ev", "Ev"), ("Iv", "Iv")]:
        fig.add_trace(go.Scatter(x=days, y=out[key], name=label,
                                 line=dict(color=C[key], width=2),
                                 hovertemplate="%{y:,.0f}<extra>" + label + "</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=total_v, name="Total vectors",
                             line=dict(color=C["total_v"], width=2.5),
                             fill="tozeroy", fillcolor="rgba(38,198,218,0.15)",
                             hovertemplate="Day %{x}: %{y:,.0f}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=days, y=inf_frac, name="Infectious %",
                             line=dict(color=C["Iv"], width=2.5),
                             fill="tozeroy", fillcolor="rgba(229,115,115,0.2)",
                             hovertemplate="Day %{x}: %{y:.3f}%<extra></extra>"), row=1, col=3)
    fig.update_xaxes(title_text="Day")
    fig.update_yaxes(title_text="Mosquitoes", row=1, col=1)
    fig.update_yaxes(title_text="Mosquitoes", row=1, col=2)
    fig.update_yaxes(title_text="% infectious", row=1, col=3)
    fig.update_layout(height=380, template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0))
    return fig


def plot_transmission(out: Dict[str, np.ndarray]) -> go.Figure:
    T = len(out["Sh"])
    days = _days(T)
    Nh = out["Sh"][0] + out["Eh"][0] + out["Ih"][0] + out["Rh"][0]
    foi = out["Iv"] / max(Nh, 1)
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["True vs Reported Cases",
                                        "Force of Infection (proxy)", "Phase Portrait  Ih vs Sh"],
                        horizontal_spacing=0.11)
    fig.add_trace(go.Bar(x=days, y=out["new_inf_h"], name="True infections",
                         marker_color=C["new_inf"], opacity=0.6,
                         hovertemplate="Day %{x}: %{y:,.0f}<extra>True</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=out["obs_cases"], name="Reported cases",
                             line=dict(color=C["obs"], width=2),
                             hovertemplate="Day %{x}: %{y:,.0f}<extra>Reported</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=foi, name="FOI proxy",
                             line=dict(color=C["foi"], width=2),
                             fill="tozeroy", fillcolor="rgba(255,112,67,0.15)",
                             hovertemplate="Day %{x}: %{y:.5f}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=out["Sh"], y=out["Ih"], mode="lines+markers",
                             line=dict(color="#5C6BC0", width=1.5),
                             marker=dict(size=3, opacity=0.6, color=np.arange(T),
                                         colorscale="Viridis",
                                         colorbar=dict(title="Day", len=0.5, x=1.01, thickness=10)),
                             name="Trajectory",
                             hovertemplate="Sh=%{x:,.0f}  Ih=%{y:,.0f}<extra></extra>"), row=1, col=3)
    fig.add_trace(go.Scatter(x=[out["Sh"][0]], y=[out["Ih"][0]], mode="markers",
                             marker=dict(color="green", size=10, symbol="circle"),
                             name="Start"), row=1, col=3)
    fig.add_trace(go.Scatter(x=[out["Sh"][-1]], y=[out["Ih"][-1]], mode="markers",
                             marker=dict(color="red", size=10, symbol="square"),
                             name="End"), row=1, col=3)
    fig.update_xaxes(title_text="Day", col=1)
    fig.update_xaxes(title_text="Day", col=2)
    fig.update_xaxes(title_text="Susceptible hosts (Sh)", col=3)
    fig.update_yaxes(title_text="Cases", row=1, col=1)
    fig.update_yaxes(title_text="Iv / Nh", row=1, col=2)
    fig.update_yaxes(title_text="Infectious hosts (Ih)", row=1, col=3)
    fig.update_layout(height=400, template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0))
    return fig


def plot_rainfall_analysis(out: Dict[str, np.ndarray], raw_rain: np.ndarray,
                            rain_lag: int) -> go.Figure:
    T = len(out["Sv"])
    days = _days(T)
    smoothed = out["rain"]
    lagged = np.zeros(T)
    if rain_lag < T:
        lagged[rain_lag:] = raw_rain[:T - rain_lag]
    total_v = out["Sv"] + out["Ev"] + out["Iv"]
    emergence_proxy = np.clip(np.diff(total_v, prepend=total_v[0]), 0, None)
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Raw vs Smoothed Rainfall",
                                        f"Lagged Rainfall (lag={rain_lag}d)",
                                        "Lagged Rainfall vs Daily Emergence"],
                        horizontal_spacing=0.11)
    fig.add_trace(go.Bar(x=days, y=raw_rain[:T], name="Raw rainfall",
                         marker_color=C["rain"], opacity=0.5,
                         hovertemplate="Day %{x}: %{y:.1f} mm<extra>Raw</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=smoothed, name="Smoothed",
                             line=dict(color="#0288D1", width=2),
                             hovertemplate="Day %{x}: %{y:.2f} mm<extra>Smoothed</extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=days, y=lagged, name="Lagged rainfall",
                         marker_color="#4DB6AC", opacity=0.7,
                         hovertemplate="Day %{x}: %{y:.1f} mm<extra>Lagged</extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=lagged, y=emergence_proxy, mode="markers",
                             marker=dict(color=days, colorscale="Viridis", size=5, opacity=0.7,
                                         colorbar=dict(title="Day", len=0.5, x=1.01)),
                             name="Rainfall vs emergence",
                             hovertemplate="Rain=%{x:.1f} mm  Emergence=%{y:,.0f}<extra></extra>"),
                 row=1, col=3)
    fig.update_xaxes(title_text="Day", col=1)
    fig.update_xaxes(title_text="Day", col=2)
    fig.update_xaxes(title_text="Lagged rainfall (mm)", col=3)
    fig.update_yaxes(title_text="mm/day", row=1, col=1)
    fig.update_yaxes(title_text="mm/day", row=1, col=2)
    fig.update_yaxes(title_text="Δ Total vectors", row=1, col=3)
    fig.update_layout(height=400, template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ensemble_overview(ens: Dict[str, np.ndarray], raw_rain: np.ndarray) -> go.Figure:
    """3-panel overview with median + 90% CI bands across all runs."""
    T = ens["rain"].shape[0]
    days = _days(T)

    def med(k):   return np.median(ens[k], axis=0)
    def p05(k):   return np.percentile(ens[k], 5, axis=0)
    def p95(k):   return np.percentile(ens[k], 95, axis=0)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.22, 0.42, 0.36], vertical_spacing=0.04,
                        subplot_titles=["Rainfall (mm/day)",
                                        "Host SEIR — Median ± 90% CI",
                                        "Vectors — Median ± 90% CI"])
    fig.add_trace(go.Bar(x=days, y=raw_rain[:T], name="Rainfall",
                         marker_color=C["rain"], opacity=0.8), row=1, col=1)

    seir_rgb = {"Sh": "33,150,243", "Eh": "255,152,0", "Ih": "244,67,54", "Rh": "76,175,80"}
    for key, label in [("Sh", "Susceptible"), ("Eh", "Exposed"), ("Ih", "Infectious"), ("Rh", "Recovered")]:
        _band(fig, days, p05(key), p95(key), seir_rgb[key], row=2, col=1)
        fig.add_trace(go.Scatter(x=days, y=med(key), name=label,
                                 line=dict(color=C[key[0]], width=2),
                                 hovertemplate=f"Median %{{y:,.0f}}<extra>{label}</extra>"), row=2, col=1)

    vec_rgb = {"Sv": "100,181,246", "Ev": "255,183,77", "Iv": "229,115,115"}
    for key, label in [("Sv", "Sv"), ("Ev", "Ev"), ("Iv", "Iv")]:
        _band(fig, days, p05(key), p95(key), vec_rgb[key], row=3, col=1)
        fig.add_trace(go.Scatter(x=days, y=med(key), name=label,
                                 line=dict(color=C[key], width=2),
                                 hovertemplate=f"Median %{{y:,.0f}}<extra>{label}</extra>"), row=3, col=1)

    fig.update_xaxes(title_text="Day", row=3, col=1)
    fig.update_yaxes(title_text="mm", row=1, col=1)
    fig.update_yaxes(title_text="Animals", row=2, col=1)
    fig.update_yaxes(title_text="Mosquitoes", row=3, col=1)
    fig.update_layout(height=650, template="plotly_white", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1))
    return fig


def plot_ensemble_incidence(ens: Dict[str, np.ndarray]) -> go.Figure:
    """Daily incidence across all runs: individual traces + median + 90% CI."""
    n_runs, T = ens["new_inf_h"].shape
    days = _days(T)
    median = np.median(ens["new_inf_h"], axis=0)
    p05 = np.percentile(ens["new_inf_h"], 5, axis=0)
    p95 = np.percentile(ens["new_inf_h"], 95, axis=0)
    cum_median = np.median(np.cumsum(ens["new_inf_h"], axis=1), axis=0)
    cum_p05 = np.percentile(np.cumsum(ens["new_inf_h"], axis=1), 5, axis=0)
    cum_p95 = np.percentile(np.cumsum(ens["new_inf_h"], axis=1), 95, axis=0)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f"Daily New Infections ({n_runs} runs)",
                                        "Cumulative Infections"],
                        horizontal_spacing=0.10)

    # Individual runs (light)
    for i in range(n_runs):
        fig.add_trace(go.Scatter(x=days, y=ens["new_inf_h"][i],
                                 line=dict(color="rgba(171,71,188,0.12)", width=1),
                                 showlegend=False, hoverinfo="skip"), row=1, col=1)
    _band(fig, days, p05, p95, "171,71,188", row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=median, name="Median",
                             line=dict(color=C["new_inf"], width=2.5),
                             hovertemplate="Day %{x}: %{y:,.0f}<extra>Median</extra>"), row=1, col=1)

    for i in range(n_runs):
        fig.add_trace(go.Scatter(x=days, y=np.cumsum(ens["new_inf_h"][i]),
                                 line=dict(color="rgba(171,71,188,0.10)", width=1),
                                 showlegend=False, hoverinfo="skip"), row=1, col=2)
    _band(fig, days, cum_p05, cum_p95, "171,71,188", row=1, col=2)
    fig.add_trace(go.Scatter(x=days, y=cum_median, name="Cumulative median",
                             line=dict(color=C["new_inf"], width=2.5),
                             fill="tozeroy", fillcolor="rgba(171,71,188,0.15)",
                             hovertemplate="Day %{x}: %{y:,.0f}<extra>Cum. median</extra>"), row=1, col=2)

    fig.update_xaxes(title_text="Day")
    fig.update_yaxes(title_text="New infections / day", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative infections", row=1, col=2)
    fig.update_layout(height=420, template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# HUMAN SPILLOVER PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_human_dynamics(out: Dict[str, np.ndarray]) -> go.Figure:
    """4-panel human SEIR view."""
    T = len(out["Sh_hum"])
    days = _days(T)
    Nh = float(out["Sh_hum"][0] + out["Eh_hum"][0] + out["Ih_hum"][0] + out["Rh_hum"][0])
    cum_hum = np.cumsum(out["new_inf_hum"])
    attack = cum_hum / max(Nh, 1) * 100

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Human SEIR (stacked area)", "Daily New Human Infections",
                                        "Cumulative Human Infections", "Human Attack Rate (%)"],
                        vertical_spacing=0.14, horizontal_spacing=0.10)

    for key, label, col_key in [
        ("Sh_hum", "Susceptible", "human_S"),
        ("Rh_hum", "Recovered", "human_R"),
        ("Eh_hum", "Exposed", "human_E"),
        ("Ih_hum", "Infectious", "human_I"),
    ]:
        fig.add_trace(go.Scatter(x=days, y=out[key], name=label,
                                 stackgroup="hum", line=dict(width=0.5, color=C[col_key]),
                                 fillcolor=C[col_key], opacity=0.75,
                                 hovertemplate="%{y:,.1f}<extra>" + label + "</extra>"), row=1, col=1)

    fig.add_trace(go.Bar(x=days, y=out["new_inf_hum"], name="New human inf.",
                         marker_color=C["human_I"], opacity=0.85,
                         hovertemplate="Day %{x}: %{y:.2f}<extra></extra>"), row=1, col=2)

    fig.add_trace(go.Scatter(x=days, y=cum_hum, name="Cumulative",
                             line=dict(color=C["human_I"], width=2.5),
                             fill="tozeroy", fillcolor="rgba(66,165,245,0.15)",
                             hovertemplate="Day %{x}: %{y:.1f}<extra></extra>"), row=2, col=1)

    fig.add_trace(go.Scatter(x=days, y=attack, name="Attack rate",
                             line=dict(color=C["human_R"], width=2.5),
                             hovertemplate="Day %{x}: %{y:.3f}%<extra></extra>"), row=2, col=2)

    fig.add_trace(go.Scatter(x=days, y=out["obs_hum"], name="Reported cases",
                             mode="markers",
                             marker=dict(color=C["obs"], size=4, opacity=0.7),
                             hovertemplate="Day %{x}: %{y:,.0f}<extra>Reported</extra>"), row=1, col=2)

    fig.update_xaxes(title_text="Day")
    fig.update_yaxes(title_text="People", row=1, col=1)
    fig.update_yaxes(title_text="Infections / day", row=1, col=2)
    fig.update_yaxes(title_text="Total infections", row=2, col=1)
    fig.update_yaxes(title_text="%", row=2, col=2)
    fig.update_layout(height=600, template="plotly_white", showlegend=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# INTERVENTION COMPARISON PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_intervention_comparison(out_base: Dict[str, np.ndarray],
                                  out_int: Dict[str, np.ndarray],
                                  label: str = "Intervention") -> go.Figure:
    """Overlay baseline vs intervention run across 4 key metrics."""
    T = len(out_base["Sh"])
    days = _days(T)

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Daily New Infections", "Infectious Hosts (Ih)",
                                        "Infectious Vectors (Iv)", "Cumulative Infections"],
                        vertical_spacing=0.14, horizontal_spacing=0.10)

    pairs = [
        (out_base["new_inf_h"],            out_int["new_inf_h"],            1, 1, "New inf."),
        (out_base["Ih"].astype(float),     out_int["Ih"].astype(float),     1, 2, "Ih"),
        (out_base["Iv"],                   out_int["Iv"],                   2, 1, "Iv"),
        (np.cumsum(out_base["new_inf_h"]), np.cumsum(out_int["new_inf_h"]), 2, 2, "Cumulative"),
    ]
    for base_y, int_y, row, col, yname in pairs:
        fig.add_trace(go.Scatter(x=days, y=base_y, name="Baseline",
                                 line=dict(color=C["baseline"], width=2),
                                 showlegend=(row == 1 and col == 1),
                                 hovertemplate=f"Day %{{x}}  Baseline %{{y:,.1f}}<extra>{yname}</extra>"),
                      row=row, col=col)
        fig.add_trace(go.Scatter(x=days, y=int_y, name=label,
                                 line=dict(color=C["intervention"], width=2, dash="dash"),
                                 showlegend=(row == 1 and col == 1),
                                 hovertemplate=f"Day %{{x}}  {label} %{{y:,.1f}}<extra>{yname}</extra>"),
                      row=row, col=col)

    # Reduction annotation on cumulative panel
    base_tot = int(out_base["new_inf_h"].sum())
    int_tot = int(out_int["new_inf_h"].sum())
    pct = (base_tot - int_tot) / max(base_tot, 1) * 100
    fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.06,
                       text=f"<b>Reduction: {pct:.1f}%</b><br>({base_tot:,} → {int_tot:,})",
                       showarrow=False, bgcolor="white", bordercolor="#C62828",
                       font=dict(color="#C62828", size=12))

    fig.update_xaxes(title_text="Day")
    fig.update_layout(height=560, template="plotly_white", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# R₀ DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def plot_r0_gauge(r0_info: dict) -> go.Figure:
    """Gauge + component bar for R₀."""
    R0 = r0_info["R0"]
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "indicator"}, {"type": "bar"}]],
                        subplot_titles=["Basic Reproduction Number R₀", "R₀ Component Breakdown"])

    color = "#4CAF50" if R0 < 1 else ("#FF9800" if R0 < 2 else "#F44336")
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=R0,
        delta={"reference": 1.0, "valueformat": ".3f"},
        gauge={
            "axis": {"range": [0, max(5, R0 * 1.5)], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [{"range": [0, 1], "color": "rgba(76,175,80,0.2)"},
                      {"range": [1, 2], "color": "rgba(255,152,0,0.2)"},
                      {"range": [2, max(5, R0 * 1.5)], "color": "rgba(244,67,54,0.15)"}],
            "threshold": {"line": {"color": "black", "width": 3}, "value": 1},
        },
        title={"text": "R₀"},
        number={"valueformat": ".3f"},
    ), row=1, col=1)

    components = ["R₀_hv\n(host→vector)", "R₀_vh\n(vector→host)", "m\n(vec/host ratio)"]
    values = [r0_info["R0_hv"], r0_info["R0_vh"], r0_info["m"]]
    bar_colors = ["#FF9800", "#2196F3", "#9C27B0"]
    fig.add_trace(go.Bar(x=components, y=values,
                         marker_color=bar_colors, opacity=0.85,
                         text=[f"{v:.3f}" for v in values], textposition="outside",
                         hovertemplate="%{x}: %{y:.4f}<extra></extra>"), row=1, col=2)

    status = "Epidemic likely (R₀ > 1)" if R0 > 1 else "Epidemic unlikely (R₀ < 1)"
    fig.add_annotation(xref="paper", yref="paper", x=0.25, y=-0.12,
                       text=f"<b>{status}</b>", showarrow=False,
                       font=dict(color=color, size=13))
    fig.update_layout(height=380, template="plotly_white", showlegend=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY ANALYSIS (PRCC TORNADO)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sensitivity(param_names: List[str], prcc: np.ndarray,
                     p_vals: np.ndarray) -> go.Figure:
    """Horizontal tornado chart of PRCC values."""
    labels = {
        "latent_days": "Latent period (host)",
        "infectious_days": "Infectious period (host)",
        "p_vh": "Vector→Host prob. (p_vh)",
        "mu": "Mosquito mortality (μ)",
        "sigma": "Vector E→I rate (σ)",
        "b": "Biting rate (b)",
        "p_hv": "Host→Vector prob. (p_hv)",
        "alpha": "Emergence scaling (α)",
    }
    nice_names = [labels.get(p, p) for p in param_names]
    order = np.argsort(np.abs(prcc))
    sorted_names = [nice_names[i] for i in order]
    sorted_prcc = prcc[order]
    sorted_p = p_vals[order]

    colors = ["#F44336" if v > 0 else "#2196F3" for v in sorted_prcc]
    sig_marks = ["★" if p < 0.05 else "" for p in sorted_p]
    text_labels = [f"{v:.3f} {m}" for v, m in zip(sorted_prcc, sig_marks)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_prcc, y=sorted_names,
        orientation="h",
        marker_color=colors, opacity=0.85,
        text=text_labels, textposition="outside",
        hovertemplate="%{y}: PRCC=%{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.add_vline(x=0.5, line_dash="dot", line_color="#4CAF50", line_width=1,
                  annotation_text="Strong +", annotation_position="top right")
    fig.add_vline(x=-0.5, line_dash="dot", line_color="#F44336", line_width=1,
                  annotation_text="Strong −", annotation_position="top left")
    fig.update_layout(
        height=420, template="plotly_white",
        xaxis_title="Partial Rank Correlation Coefficient (PRCC)",
        xaxis_range=[-1.05, 1.05],
        title="Parameter Sensitivity — ★ = statistically significant (p < 0.05)",
        margin=dict(l=200),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVED DATA OVERLAY
# ═══════════════════════════════════════════════════════════════════════════════

def plot_observed_overlay(out: Dict[str, np.ndarray],
                           df_obs,
                           ens: Optional[Dict[str, np.ndarray]] = None) -> go.Figure:
    """Model-predicted reported cases vs uploaded observed case data."""
    import pandas as pd
    T = len(out["obs_cases"])
    days = _days(T)

    fig = go.Figure()

    # Ensemble band (optional) — plain Figure, so build band trace directly
    if ens is not None and "obs_cases" in ens:
        p05 = np.percentile(ens["obs_cases"], 5, axis=0)
        p95 = np.percentile(ens["obs_cases"], 95, axis=0)
        dl = days.tolist()
        fig.add_trace(go.Scatter(
            x=dl + dl[::-1],
            y=np.concatenate([p95, p05[::-1]]).tolist(),
            fill="toself", fillcolor="rgba(126,87,194,0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(x=days,
                                 y=np.median(ens["obs_cases"], axis=0),
                                 name="Model median (reported)", line=dict(color=C["obs"], width=2),
                                 hovertemplate="Day %{x}: %{y:,.0f}<extra>Model median</extra>"))
    else:
        fig.add_trace(go.Scatter(x=days, y=out["obs_cases"],
                                 name="Model reported cases",
                                 line=dict(color=C["obs"], width=2),
                                 hovertemplate="Day %{x}: %{y:,.0f}<extra>Model</extra>"))

    # Observed data
    obs_days = df_obs.iloc[:, 0].to_numpy(dtype=float)
    obs_cases = df_obs.iloc[:, 1].to_numpy(dtype=float)
    fig.add_trace(go.Scatter(x=obs_days, y=obs_cases, mode="markers+lines",
                             name="Observed (uploaded)",
                             marker=dict(color="#FF5722", size=7, symbol="circle"),
                             line=dict(color="#FF5722", width=1.5, dash="dot"),
                             hovertemplate="Day %{x}: %{y:,.0f}<extra>Observed</extra>"))

    fig.update_layout(height=400, template="plotly_white",
                      xaxis_title="Day", yaxis_title="Reported cases",
                      title="Model Predicted vs Observed Cases",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      hovermode="x unified")
    return fig
