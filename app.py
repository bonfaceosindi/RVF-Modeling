"""
Rift Valley Fever Transmission Model — Streamlit Web App
"""

import sys
import pathlib

import numpy as np
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model import (
    HostParams, VectorParams, ObsParams, SimParams,
    HumanParams, InterventionParams,
    simulate, run_ensemble, compute_r0, run_sensitivity,
)
from utils.plotting import (
    plot_overview, plot_host_dynamics, plot_vector_dynamics,
    plot_transmission, plot_rainfall_analysis,
    plot_ensemble_overview, plot_ensemble_incidence,
    plot_human_dynamics, plot_intervention_comparison,
    plot_r0_gauge, plot_sensitivity, plot_observed_overlay,
)
from utils.rainfall_api import fetch_nasa_power, fetch_chirps, OUTBREAK_HIERARCHY

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RVF Transmission Model",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🦟 Rift Valley Fever Transmission Model")
st.caption(
    "Individual-based livestock host model coupled to a rainfall-forced mosquito ODE, "
    "with optional human spillover, interventions, ensemble uncertainty, and sensitivity analysis."
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Rainfall ──────────────────────────────────────────────────────────────
    st.header("📁 Rainfall Input")
    rain_source = st.radio(
        "Rainfall source",
        ["Upload CSV", "NASA POWER", "CHIRPS"],
        horizontal=True,
        help=(
            "Upload CSV: your own data.  "
            "NASA POWER: ~50 km, instant.  "
            "CHIRPS: ~5 km, async (~30–60 s)."
        ),
    )

    sample_path = ROOT / "data" / "sample_rainfall.csv"
    rain_array: "np.ndarray | None" = None

    # ── CSV branch ────────────────────────────────────────────────────────────
    if rain_source == "Upload CSV":
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                st.download_button("⬇ Download sample CSV", f,
                                   "sample_rainfall.csv", "text/csv")
        st.markdown("**Format:** `day,rainfall_mm` (two columns) or single column `rainfall_mm`.")
        uploaded = st.file_uploader("Upload rainfall CSV", type=["csv"])
        if uploaded is not None:
            try:
                df_rain = pd.read_csv(uploaded)
                col = "rainfall_mm" if "rainfall_mm" in df_rain.columns else df_rain.columns[-1]
                rain_array = df_rain[col].to_numpy(dtype=float)
                st.success(f"Loaded {len(rain_array)} days.")
            except Exception as e:
                st.error(f"Parse error: {e}")
        if rain_array is None and sample_path.exists():
            rain_array = pd.read_csv(sample_path)["rainfall_mm"].to_numpy(dtype=float)
            st.info("Using sample rainfall (365 days).")

    # ── API branches (NASA POWER or CHIRPS) ───────────────────────────────────
    else:
        # Cascading location presets — shared by both APIs
        region  = st.selectbox("Region",         list(OUTBREAK_HIERARCHY.keys()))
        country = st.selectbox("Country / Area", list(OUTBREAK_HIERARCHY[region].keys()))
        event   = st.selectbox("Outbreak event", list(OUTBREAK_HIERARCHY[region][country].keys()))
        preset  = OUTBREAK_HIERARCHY[region][country][event]

        if preset.get("note"):
            st.caption(f"📌 {preset['note']}")

        api_lat   = st.number_input("Latitude",  -90.0,  90.0,   float(preset["lat"]), 0.001, format="%.3f")
        api_lon   = st.number_input("Longitude", -180.0, 180.0,  float(preset["lon"]), 0.001, format="%.3f")
        api_start = st.date_input("Start date", value=preset["start"])
        api_end   = st.date_input("End date",   value=preset["end"])

        # Source-specific config
        if rain_source == "NASA POWER":
            _src_key  = "nasa"
            _fetch_fn = fetch_nasa_power
            _label    = "🌐 Fetch NASA POWER Data"
            _spinner  = "Fetching from NASA POWER (~5 s)…"
            st.caption("Source: NASA POWER PRECTOTCORR · ~0.5° resolution · 1981–present")
        else:  # CHIRPS
            _src_key  = "chirps"
            _fetch_fn = fetch_chirps
            _label    = "🌐 Fetch CHIRPS Data"
            _spinner  = "Fetching from ClimateSERV / CHIRPS — async job, may take 30–60 s…"
            st.caption("Source: CHIRPS v2.0 · ~0.05° (~5 km) resolution · 1981–present")

        _rain_key = f"{_src_key}_rain"
        _meta_key = f"{_src_key}_meta"
        if _rain_key not in st.session_state:
            st.session_state[_rain_key] = None
            st.session_state[_meta_key] = None

        fetch_btn = st.button(_label, use_container_width=True)
        if fetch_btn:
            if api_end < api_start:
                st.error("End date must be after start date.")
            else:
                with st.spinner(_spinner):
                    try:
                        arr, meta = _fetch_fn(api_lat, api_lon, api_start, api_end)
                        st.session_state[_rain_key] = arr
                        st.session_state[_meta_key] = meta
                        st.success(
                            f"Fetched {meta['n_days']} days · "
                            f"Total {meta['total_mm']:.1f} mm · "
                            f"Peak {meta['peak_mm']:.1f} mm (day {meta['peak_day']})"
                        )
                    except Exception as e:
                        st.error(f"Fetch failed: {e}")

        if st.session_state[_rain_key] is not None:
            rain_array = st.session_state[_rain_key]
            meta       = st.session_state[_meta_key]
            import plotly.graph_objects as _go
            _fig = _go.Figure()
            _fig.add_bar(
                x=list(range(1, len(rain_array) + 1)),
                y=rain_array.tolist(),
                marker_color="#4C9BE8" if rain_source == "NASA POWER" else "#2E7D32",
                name="mm/day",
            )
            _fig.update_layout(
                height=160, margin=dict(l=0, r=0, t=24, b=0),
                title_text=(
                    f"{meta['source']} · {meta['location']}  "
                    f"({meta['start']} → {meta['end']})"
                ),
                title_font_size=11,
                xaxis_title="Day", yaxis_title="mm",
                showlegend=False,
            )
            st.plotly_chart(_fig, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── Simulation setup ──────────────────────────────────────────────────────
    st.header("⚙️ Simulation Setup")
    with st.expander("Core Setup", expanded=True):
        n_hosts     = st.slider("Livestock population", 100, 5000, 1000, 100)
        init_host_I = st.slider("Initially infectious animals", 0, 50, 1)
        Sv0         = st.number_input("Initial susceptible vectors (Sv₀)", 100, 200_000, 5000, 500)
        seed        = st.number_input("Random seed", 0, 9999, 123, 1)
        n_runs      = st.slider("Ensemble runs (1 = single run)", 1, 200, 1, 1,
                                help="Set >1 to compute uncertainty bands across stochastic realisations.")

    # ── Host parameters ────────────────────────────────────────────────────────
    st.header("🐄 Host Parameters")
    with st.expander("Livestock Disease History"):
        latent_days    = st.slider("Latent period (days)", 1, 14, 3)
        infectious_days = st.slider("Infectious period (days)", 1, 21, 5)
        p_vh           = st.slider("Vector→Host prob. (p_vh)", 0.01, 1.0, 0.20, 0.01)

    # ── Vector parameters ──────────────────────────────────────────────────────
    st.header("🦟 Vector Parameters")
    with st.expander("Mosquito ODE"):
        mu    = st.slider("Daily adult mortality (μ)", 0.01, 0.50, 0.10, 0.01)
        sigma = st.slider("E→I progression rate (σ)", 0.01, 0.50, 0.20, 0.01)
        b     = st.slider("Bites per mosquito per day (b)", 0.05, 1.0, 0.30, 0.05)
        p_hv  = st.slider("Host→Vector prob. (p_hv)", 0.01, 1.0, 0.15, 0.01)

    # ── Rainfall forcing ───────────────────────────────────────────────────────
    st.header("🌧️ Rainfall Forcing")
    with st.expander("Emergence & Carrying Capacity"):
        alpha            = st.slider("Emergence scaling (α)", 1, 300, 50, 5)
        rain_lag_days    = st.slider("Rainfall lag (days)", 0, 30, 14)
        rain_smooth_days = st.slider("Smoothing window (days)", 1, 21, 7)
        K0     = st.number_input("Baseline carrying capacity (K₀)", 1000, 500_000, 10_000, 1000)
        K_rain = st.number_input("Rain-driven K increase (K_rain)", 1, 500, 50, 5)

    # ── Observation model ──────────────────────────────────────────────────────
    st.header("📋 Observation Model")
    with st.expander("Reporting"):
        rho        = st.slider("Livestock reporting fraction (ρ)", 0.01, 1.0, 0.20, 0.01)
        overdisp_k = st.slider("NegBinom overdispersion (k)", 1, 100, 10)

    # ── Human spillover ────────────────────────────────────────────────────────
    st.header("👥 Human Spillover")
    with st.expander("Human Compartment (optional)"):
        include_humans   = st.checkbox("Include human spillover", value=False)
        n_humans         = st.number_input("Human population", 1000, 10_000_000, 10_000, 1000)
        latent_hum       = st.slider("Human latent period (days)", 1, 10, 3)
        infectious_hum   = st.slider("Human infectious period (days)", 1, 21, 6)
        p_vh_hum         = st.slider("Vector→Human prob.", 0.01, 0.50, 0.10, 0.01)
        rho_hum          = st.slider("Human reporting fraction", 0.01, 1.0, 0.30, 0.01)

    # ── Interventions ──────────────────────────────────────────────────────────
    st.header("💉 Interventions")
    with st.expander("Vaccination & Control"):
        use_vacc      = st.checkbox("Vaccination campaign")
        vacc_day      = st.slider("Vaccination day", 1, 365, 60) if use_vacc else 9999
        vacc_cov      = st.slider("Coverage", 0.10, 1.0, 0.70, 0.05) if use_vacc else 0.70
        vacc_eff      = st.slider("Efficacy", 0.10, 1.0, 0.95, 0.05) if use_vacc else 0.95

        use_vc        = st.checkbox("Vector control")
        vc_day        = st.slider("Vector control start day", 1, 365, 60) if use_vc else 9999
        vc_dur        = st.slider("Duration (days)", 7, 120, 30) if use_vc else 30
        vc_mult       = st.slider("Mortality multiplier", 1.5, 10.0, 3.0, 0.5) if use_vc else 3.0

        use_restrict  = st.checkbox("Movement restriction")
        rest_day      = st.slider("Restriction start day", 1, 365, 60) if use_restrict else 9999
        rest_dur      = st.slider("Restriction duration (days)", 7, 120, 30) if use_restrict else 30
        rest_b_mult   = st.slider("Biting rate multiplier", 0.10, 0.90, 0.50, 0.05) if use_restrict else 0.50

    st.divider()
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)


# ── Guard: need rainfall ───────────────────────────────────────────────────────
if rain_array is None:
    st.warning("Upload a rainfall CSV to continue.")
    st.stop()

# ── Build parameter objects ────────────────────────────────────────────────────
def _build_params():
    hp = HostParams(latent_days=latent_days, infectious_days=infectious_days, p_vh=p_vh)
    vp = VectorParams(mu=mu, sigma=sigma, b=b, p_hv=p_hv,
                      alpha=alpha, rain_lag_days=rain_lag_days,
                      rain_smooth_days=rain_smooth_days, K0=float(K0), K_rain=float(K_rain))
    op = ObsParams(rho=rho, overdisp_k=overdisp_k)
    sp = SimParams(n_hosts=n_hosts, init_host_I=init_host_I,
                   Sv0=float(Sv0), seed=int(seed))
    hp_h = HumanParams(n_humans=int(n_humans), latent_days=latent_hum,
                       infectious_days=infectious_hum,
                       p_vh_human=p_vh_hum, rho_human=rho_hum) if include_humans else None
    ip = InterventionParams(
        vacc_day=vacc_day if use_vacc else 9999,
        vacc_coverage=vacc_cov, vacc_efficacy=vacc_eff,
        vc_day=vc_day if use_vc else 9999,
        vc_duration=vc_dur, vc_mortality_mult=vc_mult,
        restrict_day=rest_day if use_restrict else 9999,
        restrict_duration=rest_dur, restrict_b_mult=rest_b_mult,
    ) if (use_vacc or use_vc or use_restrict) else None
    return hp, vp, op, sp, hp_h, ip


# ── Session state init ─────────────────────────────────────────────────────────
for key in ("results", "ensemble", "raw_rain", "r0_info", "sensitivity"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Run ────────────────────────────────────────────────────────────────────────
if run_btn:
    hp, vp, op, sp, hp_h, ip = _build_params()
    with st.spinner(f"Running {'ensemble (' + str(n_runs) + ' runs)' if n_runs > 1 else 'simulation'}…"):
        try:
            if n_runs > 1:
                ens = run_ensemble(rain_array, sp, hp, vp, op, hp_h, ip, n_runs=n_runs)
                # Use the first run as the "representative" single run for tables/exports
                first = simulate(rain_array, sp, hp, vp, op, hp_h, ip)
                st.session_state.results  = first
                st.session_state.ensemble = ens
            else:
                out = simulate(rain_array, sp, hp, vp, op, hp_h, ip)
                st.session_state.results  = out
                st.session_state.ensemble = None
            st.session_state.raw_rain = rain_array.copy()
            st.session_state.r0_info  = compute_r0(sp, hp, vp)
            st.session_state.sensitivity = None   # reset when params change
            st.success("Done.")
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.stop()

out      = st.session_state.results
ens      = st.session_state.ensemble
raw_rain = st.session_state.raw_rain
r0_info  = st.session_state.r0_info

if out is None:
    st.info("Set parameters in the sidebar and click **▶ Run Simulation**.")
    st.stop()


# ── KPI cards ──────────────────────────────────────────────────────────────────
T         = len(out["Sh"])
Nh_init   = int(out["Sh"][0] + out["Eh"][0] + out["Ih"][0] + out["Rh"][0])
total_inf = int(out["new_inf_h"].sum())
if ens is not None:
    total_inf_med = int(np.median(ens["new_inf_h"].sum(axis=1)))
    total_inf_ci  = (int(np.percentile(ens["new_inf_h"].sum(axis=1), 5)),
                     int(np.percentile(ens["new_inf_h"].sum(axis=1), 95)))
    inf_display   = f"{total_inf_med:,}"
    inf_delta     = f"90% CI: {total_inf_ci[0]:,}–{total_inf_ci[1]:,}"
else:
    inf_display = f"{total_inf:,}"
    inf_delta   = None

attack_rate       = total_inf / max(Nh_init, 1) * 100
peak_day          = int(np.argmax(out["new_inf_h"])) + 1
peak_inf          = int(out["new_inf_h"].max())
total_reported    = int(out["obs_cases"].sum())
total_vec_peak    = int((out["Sv"] + out["Ev"] + out["Iv"]).max())
r0_val            = r0_info["R0"] if r0_info else None

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Infections",     inf_display,     inf_delta)
c2.metric("Attack Rate",          f"{attack_rate:.1f}%")
c3.metric("Peak Day",             f"Day {peak_day}", f"{peak_inf:,} cases")
c4.metric("Total Reported Cases", f"{total_reported:,}")
c5.metric("Peak Vector Pop.",     f"{total_vec_peak:,}")
c6.metric("R₀",                   f"{r0_val:.3f}" if r0_val else "—",
          "Epidemic" if (r0_val and r0_val > 1) else "Fade-out")

st.divider()


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Overview",
    "🐄 Host Dynamics",
    "🦟 Vector Dynamics",
    "🔗 Transmission",
    "👥 Human Spillover",
    "💉 Interventions",
    "🎯 R₀ & Sensitivity",
    "🌧️ Rainfall",
    "📥 Data & Export",
])

# ── Tab 1: Overview ────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Simulation Overview")
    if ens is not None:
        st.caption(f"Showing median ± 90% CI across {ens['new_inf_h'].shape[0]} stochastic runs.")
        st.plotly_chart(plot_ensemble_overview(ens, raw_rain), use_container_width=True)
    else:
        st.plotly_chart(plot_overview(out, raw_rain), use_container_width=True)

# ── Tab 2: Host Dynamics ───────────────────────────────────────────────────────
with tab2:
    st.subheader("Livestock SEIR Dynamics")
    if ens is not None:
        st.caption("Stochastic uncertainty bands shown.")
        st.plotly_chart(plot_ensemble_incidence(ens), use_container_width=True)
    st.plotly_chart(plot_host_dynamics(out), use_container_width=True)

# ── Tab 3: Vector Dynamics ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Mosquito Population Dynamics")
    st.plotly_chart(plot_vector_dynamics(out), use_container_width=True)

# ── Tab 4: Transmission ────────────────────────────────────────────────────────
with tab4:
    st.subheader("Transmission Dynamics")
    st.plotly_chart(plot_transmission(out), use_container_width=True)

# ── Tab 5: Human Spillover ─────────────────────────────────────────────────────
with tab5:
    st.subheader("Human Spillover Compartment")
    if "Sh_hum" not in out:
        st.info("Enable **Include human spillover** in the sidebar, then re-run.")
    else:
        Nh_hum = float(out["Sh_hum"][0] + out["Eh_hum"][0] + out["Ih_hum"][0] + out["Rh_hum"][0])
        total_hum = float(out["new_inf_hum"].sum())
        hum_ar    = total_hum / max(Nh_hum, 1) * 100
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric("Total Human Infections", f"{total_hum:,.1f}")
        hc2.metric("Human Attack Rate",       f"{hum_ar:.3f}%")
        hc3.metric("Total Reported (humans)", f"{int(out['obs_hum'].sum()):,}")
        st.plotly_chart(plot_human_dynamics(out), use_container_width=True)

# ── Tab 6: Interventions ───────────────────────────────────────────────────────
with tab6:
    st.subheader("Intervention Comparison")
    any_active = use_vacc or use_vc or use_restrict
    if not any_active:
        st.info("Enable at least one intervention in the sidebar (vaccination, vector control, or "
                "movement restriction), then re-run.")
    else:
        # Build no-intervention baseline
        hp_b, vp_b, op_b, sp_b, hp_h_b, _ = _build_params()
        with st.spinner("Running baseline (no interventions)…"):
            out_base = simulate(rain_array, sp_b, hp_b, vp_b, op_b, hp_h_b, None)

        label_parts = []
        if use_vacc:     label_parts.append(f"Vacc. day {vacc_day}")
        if use_vc:       label_parts.append(f"Vec. control day {vc_day}")
        if use_restrict: label_parts.append(f"Restriction day {rest_day}")
        label = " + ".join(label_parts)

        st.plotly_chart(
            plot_intervention_comparison(out_base, out, label),
            use_container_width=True,
        )

# ── Tab 7: R₀ & Sensitivity ────────────────────────────────────────────────────
with tab7:
    st.subheader("R₀ Calculator")
    if r0_info:
        st.plotly_chart(plot_r0_gauge(r0_info), use_container_width=True)
        with st.expander("Formula & interpretation"):
            st.markdown(r"""
**R₀** for a vector-borne SEIR-SEI model (Next-Generation Matrix):

$$R_0 = b \cdot \sqrt{\frac{p_{vh} \cdot p_{hv} \cdot m \cdot \sigma}{(\sigma + \mu) \cdot \mu \cdot \gamma_h}}$$

where $m = S_{v0}/N_h$ is the vector-to-host ratio at the disease-free equilibrium and
$\gamma_h = 1/\text{infectious\_days}$.

- **R₀ < 1** → epidemic cannot establish; any seeded infections die out
- **R₀ > 1** → epidemic can grow; outbreak size depends on initial conditions
- The threshold is sensitive to *biting rate b* and *vector-to-host ratio m* (the two most actionable parameters for vector control and culling)
            """)

    st.divider()
    st.subheader("Sensitivity Analysis (PRCC)")
    st.markdown(
        "Runs the model ~100 times across Latin-Hypercube-sampled parameter ranges and computes "
        "**Partial Rank Correlation Coefficients** — which parameters drive total infections most."
    )
    n_sens = st.slider("Number of LHS samples", 50, 300, 100, 10)
    sens_btn = st.button("▶ Run Sensitivity Analysis", key="sens_btn")
    if sens_btn:
        hp_s, vp_s, op_s, sp_s, *_ = _build_params()
        with st.spinner(f"Running {n_sens} LHS simulations (n_hosts=200 for speed)…"):
            try:
                names, prcc, pvals = run_sensitivity(
                    rain_array, sp_s, hp_s, vp_s, op_s, n_samples=n_sens
                )
                st.session_state.sensitivity = (names, prcc, pvals)
            except Exception as e:
                st.error(f"Sensitivity analysis failed: {e}")

    if st.session_state.sensitivity is not None:
        names, prcc, pvals = st.session_state.sensitivity
        st.plotly_chart(plot_sensitivity(names, prcc, pvals), use_container_width=True)
        df_sens = pd.DataFrame({
            "Parameter": names,
            "PRCC": np.round(prcc, 4),
            "p-value": np.round(pvals, 4),
            "Significant (p<0.05)": pvals < 0.05,
        }).sort_values("PRCC", key=np.abs, ascending=False)
        st.dataframe(df_sens, use_container_width=True, hide_index=True)

# ── Tab 8: Rainfall ────────────────────────────────────────────────────────────
with tab8:
    st.subheader("Rainfall Analysis")
    st.plotly_chart(plot_rainfall_analysis(out, raw_rain, rain_lag_days), use_container_width=True)

# ── Tab 9: Data & Export ───────────────────────────────────────────────────────
with tab9:
    st.subheader("Observed Data Overlay")
    st.markdown("Upload real reported case data to compare against model predictions.")
    obs_upload = st.file_uploader("Upload observed cases CSV  (`day,cases` columns)",
                                   type=["csv"], key="obs_upload")
    if obs_upload is not None:
        try:
            df_obs = pd.read_csv(obs_upload)
            st.plotly_chart(
                plot_observed_overlay(out, df_obs, ens),
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Could not parse observed data: {e}")
    else:
        st.info("Upload a CSV with columns `day` and `cases` to compare model vs reality.")

    st.divider()
    st.subheader("Full Results Table")
    days_col = np.arange(1, T + 1)
    df_out = pd.DataFrame({
        "day":          days_col,
        "rainfall_mm":  raw_rain[:T],
        "smoothed_rain": out["rain"],
        "Sh": out["Sh"].astype(int),
        "Eh": out["Eh"].astype(int),
        "Ih": out["Ih"].astype(int),
        "Rh": out["Rh"].astype(int),
        "new_inf_h":    out["new_inf_h"].astype(int),
        "obs_cases":    out["obs_cases"].astype(int),
        "Sv": out["Sv"].round(1),
        "Ev": out["Ev"].round(1),
        "Iv": out["Iv"].round(1),
    })
    if "Sh_hum" in out:
        df_out["Sh_hum"] = out["Sh_hum"].round(1)
        df_out["Ih_hum"] = out["Ih_hum"].round(1)
        df_out["new_inf_hum"] = out["new_inf_hum"].round(2)
        df_out["obs_hum"]     = out["obs_hum"].astype(int)

    if ens is not None:
        df_out["new_inf_h_p5"]  = np.percentile(ens["new_inf_h"], 5,  axis=0).astype(int)
        df_out["new_inf_h_med"] = np.median(ens["new_inf_h"], axis=0).astype(int)
        df_out["new_inf_h_p95"] = np.percentile(ens["new_inf_h"], 95, axis=0).astype(int)

    st.dataframe(df_out, use_container_width=True, height=400)
    st.download_button("⬇ Download Results CSV",
                       df_out.to_csv(index=False).encode(),
                       "rvf_results.csv", "text/csv")

    st.divider()
    st.subheader("Parameter Summary")
    params_df = pd.DataFrame([
        ("n_hosts", n_hosts), ("init_host_I", init_host_I), ("Sv0", Sv0), ("seed", seed),
        ("n_runs", n_runs),
        ("latent_days", latent_days), ("infectious_days", infectious_days), ("p_vh", p_vh),
        ("mu", mu), ("sigma", sigma), ("b", b), ("p_hv", p_hv),
        ("alpha", alpha), ("rain_lag_days", rain_lag_days), ("rain_smooth_days", rain_smooth_days),
        ("K0", K0), ("K_rain", K_rain), ("rho", rho), ("overdisp_k", overdisp_k),
        ("include_humans", include_humans),
        ("use_vacc", use_vacc), ("use_vc", use_vc), ("use_restrict", use_restrict),
    ], columns=["Parameter", "Value"])
    st.dataframe(params_df, use_container_width=True, hide_index=True)
