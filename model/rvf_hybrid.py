"""
RVF Hybrid Model
- Hosts  : individual-based model (IBM / agent-based)
- Vectors : ODE model with rainfall-forced emergence
- Humans  : ODE-based spillover compartment (optional)
- Coupling : daily time-step loop
- Observation : syndromic reporting model (NegBinom)
- Interventions : vaccination, vector control, movement restriction (optional)
"""

from __future__ import annotations
from dataclasses import dataclass, replace as _dc_replace
from typing import Dict, List, Tuple
import numpy as np

try:
    from scipy.stats import rankdata, pearsonr
    SCIPY_STATS_OK = True
except Exception:
    SCIPY_STATS_OK = False


# =============================================================================
# 1) CONFIGURATION / PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class HostParams:
    """Livestock host disease natural history."""
    latent_days: float = 3.0
    infectious_days: float = 5.0
    p_vh: float = 0.2            # vector→host transmission probability per bite


@dataclass(frozen=True)
class VectorParams:
    """Vector (mosquito) ODE parameters."""
    mu: float = 0.10             # adult daily mortality rate
    sigma: float = 0.20          # E→I progression rate
    b: float = 0.3               # bites per vector per day
    p_hv: float = 0.15           # host→vector transmission probability per bite

    alpha: float = 50.0          # emergence scaling factor
    rain_lag_days: int = 14      # rainfall lag (days)
    rain_smooth_days: int = 7    # smoothing window (days)

    K0: float = 10_000.0         # baseline carrying capacity
    K_rain: float = 50.0         # rainfall-driven K increase


@dataclass(frozen=True)
class ObsParams:
    """Syndromic reporting model."""
    rho: float = 0.2             # reporting fraction
    overdisp_k: float = 10.0     # NegBinom overdispersion


@dataclass(frozen=True)
class SimParams:
    dt: float = 1.0
    seed: int = 123
    n_hosts: int = 1000
    init_host_state: str = "S"
    init_host_I: int = 1
    Sv0: float = 5000.0
    Ev0: float = 0.0
    Iv0: float = 0.0


@dataclass(frozen=True)
class HumanParams:
    """Human host parameters — ODE-based spillover compartment.

    Humans are a dead-end host; they receive infections from vectors but do not
    transmit back (standard assumption for RVF modelling).
    """
    n_humans: int = 10_000
    latent_days: float = 3.0
    infectious_days: float = 6.0
    p_vh_human: float = 0.10     # vector→human transmission probability per bite
    rho_human: float = 0.30      # human case reporting fraction
    init_human_I: int = 0


@dataclass(frozen=True)
class InterventionParams:
    """Optional intervention parameters applied during the simulation."""
    # Livestock vaccination — one-time mass campaign
    vacc_day: int = 9999         # day of campaign (9999 = disabled)
    vacc_coverage: float = 0.70  # fraction of susceptibles targeted
    vacc_efficacy: float = 0.95  # fraction that become protected (→ R)

    # Vector control — increased adult mosquito mortality
    vc_day: int = 9999
    vc_duration: int = 30
    vc_mortality_mult: float = 3.0  # mu multiplier during control period

    # Movement restriction — reduced vector–host contact rate
    restrict_day: int = 9999
    restrict_duration: int = 30
    restrict_b_mult: float = 0.50   # b multiplier during restriction


# =============================================================================
# 2) HOST IBM
# =============================================================================

@dataclass
class Host:
    id: int
    state: str       # "S", "E", "I", "R"
    days_in_state: int = 0


def init_hosts(n: int, init_state: str = "S", init_I: int = 0,
               rng: np.random.Generator | None = None) -> List[Host]:
    if rng is None:
        rng = np.random.default_rng()
    hosts = [Host(id=i, state=init_state) for i in range(n)]
    if init_I > 0:
        idx = rng.choice(n, size=min(init_I, n), replace=False)
        for i in idx:
            hosts[i].state = "I"
    return hosts


def update_hosts_progression(hosts: List[Host], hp: HostParams) -> None:
    latent = int(round(hp.latent_days))
    infect = int(round(hp.infectious_days))
    for h in hosts:
        if h.state in ("E", "I"):
            h.days_in_state += 1
        if h.state == "E" and h.days_in_state >= latent:
            h.state = "I"
            h.days_in_state = 0
        elif h.state == "I" and h.days_in_state >= infect:
            h.state = "R"
            h.days_in_state = 0


def apply_new_infections(hosts: List[Host], lambda_h: float,
                         rng: np.random.Generator) -> int:
    if lambda_h <= 0:
        return 0
    p = 1.0 - np.exp(-lambda_h)
    s_indices = [i for i, h in enumerate(hosts) if h.state == "S"]
    if not s_indices:
        return 0
    infected = rng.random(len(s_indices)) < p
    count = 0
    for k, i in enumerate(s_indices):
        if infected[k]:
            hosts[i].state = "E"
            hosts[i].days_in_state = 0
            count += 1
    return count


def host_counts(hosts: List[Host]) -> Dict[str, int]:
    out = {"S": 0, "E": 0, "I": 0, "R": 0}
    for h in hosts:
        out[h.state] += 1
    return out


# =============================================================================
# 3) VECTOR ODE (rainfall-forced)
# =============================================================================

def smooth_series(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    return np.convolve(x, np.ones(win) / win, mode="same")


def rainfall_forcing(rain: np.ndarray, t_index: int,
                     vp: VectorParams) -> Tuple[float, float]:
    idx = t_index - vp.rain_lag_days
    r = float(rain[idx]) if idx >= 0 else 0.0
    emergence = vp.alpha * (r / (1.0 + r))
    Kt = vp.K0 + vp.K_rain * r
    return emergence, Kt


def vector_rhs(_t: float, y: np.ndarray, t_index: int, rain: np.ndarray,
               Ih: float, Nh: float, vp: VectorParams) -> np.ndarray:
    Sv, Ev, Iv = y
    Nv = max(Sv + Ev + Iv, 1e-9)
    emergence, Kt = rainfall_forcing(rain, t_index, vp)
    lambda_v = vp.b * vp.p_hv * (Ih / Nh) if Nh > 0 else 0.0
    density_factor = max(0.0, 1.0 - (Nv / max(Kt, 1e-9)))
    dSv = emergence * density_factor - lambda_v * Sv - vp.mu * Sv
    dEv = lambda_v * Sv - vp.sigma * Ev - vp.mu * Ev
    dIv = vp.sigma * Ev - vp.mu * Iv
    return np.array([dSv, dEv, dIv], dtype=float)


def step_vectors_euler(y: np.ndarray, t_index: int, rain: np.ndarray,
                       Ih: float, Nh: float, vp: VectorParams,
                       dt: float = 1.0) -> np.ndarray:
    dy = vector_rhs(0.0, y, t_index, rain, Ih, Nh, vp)  # noqa: _t unused by design
    return np.maximum(y + dt * dy, 0.0)


# =============================================================================
# 4) COUPLING
# =============================================================================

def lambda_host_from_vectors(_Sv: float, _Ev: float, Iv: float, Nh: int,
                              vp: VectorParams, hp: HostParams) -> float:
    if Nh <= 0:
        return 0.0
    return max((vp.b / Nh) * hp.p_vh * Iv, 0.0)


# =============================================================================
# 5) OBSERVATION MODEL
# =============================================================================

def negbinom_rng(mean: float, k: float, rng: np.random.Generator) -> int:
    mean = max(mean, 0.0)
    if mean == 0:
        return 0
    if k <= 0:
        return int(rng.poisson(mean))
    lam = rng.gamma(shape=k, scale=mean / k)
    return int(rng.poisson(lam))


def observe_syndromic(new_infections: float, op: ObsParams,
                      rng: np.random.Generator) -> int:
    return negbinom_rng(op.rho * new_infections, op.overdisp_k, rng)


# =============================================================================
# 6) SIMULATION DRIVER
# =============================================================================

def simulate(rain: np.ndarray,
             sp: SimParams = SimParams(),
             hp: HostParams = HostParams(),
             vp: VectorParams = VectorParams(),
             op: ObsParams = ObsParams(),
             hp_human: "HumanParams | None" = None,
             ip: "InterventionParams | None" = None) -> Dict[str, np.ndarray]:
    """
    Run coupled host-vector model for len(rain) days.

    Optional:
        hp_human : include human spillover ODE compartment
        ip       : apply vaccination / vector-control / movement interventions
    """
    rng = np.random.default_rng(sp.seed)
    T = len(rain)
    rain_sm = smooth_series(rain, vp.rain_smooth_days)

    hosts = init_hosts(sp.n_hosts, sp.init_host_state, sp.init_host_I, rng)
    yv = np.array([sp.Sv0, sp.Ev0, sp.Iv0], dtype=float)

    # Livestock output arrays
    Sh = np.zeros(T, dtype=int)
    Eh = np.zeros(T, dtype=int)
    Ih = np.zeros(T, dtype=int)
    Rh = np.zeros(T, dtype=int)
    Sv_arr = np.zeros(T)
    Ev_arr = np.zeros(T)
    Iv_arr = np.zeros(T)
    new_inf_h = np.zeros(T, dtype=int)
    obs_cases = np.zeros(T, dtype=int)

    # Human output arrays (only allocated if needed)
    if hp_human is not None:
        y_hum = np.array([
            float(hp_human.n_humans - hp_human.init_human_I),
            0.0,
            float(hp_human.init_human_I),
            0.0,
        ])
        Sh_hum = np.zeros(T)
        Eh_hum = np.zeros(T)
        Ih_hum = np.zeros(T)
        Rh_hum = np.zeros(T)
        new_inf_hum = np.zeros(T)
        obs_hum = np.zeros(T, dtype=int)

    for t in range(T):
        # ── 0. interventions ──────────────────────────────────────────────────
        effective_vp = vp
        if ip is not None:
            # One-time vaccination of susceptible livestock
            if t == ip.vacc_day:
                s_list = [h for h in hosts if h.state == "S"]
                n_vacc = int(len(s_list) * ip.vacc_coverage * ip.vacc_efficacy)
                if n_vacc > 0:
                    chosen = rng.choice(len(s_list),
                                        size=min(n_vacc, len(s_list)),
                                        replace=False)
                    for idx in chosen:
                        s_list[idx].state = "R"

            # Effective parameters modified by ongoing interventions
            eff_mu = vp.mu
            eff_b = vp.b
            if ip.vc_day <= t < ip.vc_day + ip.vc_duration:
                eff_mu = vp.mu * ip.vc_mortality_mult
            if ip.restrict_day <= t < ip.restrict_day + ip.restrict_duration:
                eff_b = vp.b * ip.restrict_b_mult
            if eff_mu != vp.mu or eff_b != vp.b:
                effective_vp = _dc_replace(vp, mu=eff_mu, b=eff_b)

        # ── 1. host counts for vector coupling ────────────────────────────────
        hc = host_counts(hosts)
        Nh = len(hosts)
        Ih_t = hc["I"]

        # ── 2. step vectors ───────────────────────────────────────────────────
        yv = step_vectors_euler(yv, t, rain_sm, float(Ih_t), float(Nh),
                                effective_vp, sp.dt)

        # ── 3. host force of infection ────────────────────────────────────────
        lam_h = lambda_host_from_vectors(yv[0], yv[1], yv[2], Nh,  # type: ignore[arg-type]
                                         effective_vp, hp)

        # ── 4. infect & progress hosts ────────────────────────────────────────
        new_inf = apply_new_infections(hosts, lam_h, rng)
        update_hosts_progression(hosts, hp)

        # ── 5. livestock observation model ────────────────────────────────────
        obs = observe_syndromic(new_inf, op, rng)

        # ── 6. human spillover ODE ────────────────────────────────────────────
        if hp_human is not None:
            Nh_hum = float(hp_human.n_humans)
            lam_human = (effective_vp.b / max(Nh_hum, 1)) * hp_human.p_vh_human * yv[2]
            new_hum = lam_human * y_hum[0]
            dSh_h = -new_hum
            dEh_h = new_hum - (1.0 / hp_human.latent_days) * y_hum[1]
            dIh_h = (1.0 / hp_human.latent_days) * y_hum[1] - (1.0 / hp_human.infectious_days) * y_hum[2]
            dRh_h = (1.0 / hp_human.infectious_days) * y_hum[2]
            y_hum = np.maximum(y_hum + np.array([dSh_h, dEh_h, dIh_h, dRh_h]), 0.0)
            obs_h = negbinom_rng(hp_human.rho_human * new_hum, op.overdisp_k, rng)
            Sh_hum[t] = y_hum[0]
            Eh_hum[t] = y_hum[1]
            Ih_hum[t] = y_hum[2]
            Rh_hum[t] = y_hum[3]
            new_inf_hum[t] = new_hum
            obs_hum[t] = obs_h

        # ── 7. record ─────────────────────────────────────────────────────────
        hc2 = host_counts(hosts)
        Sh[t], Eh[t], Ih[t], Rh[t] = hc2["S"], hc2["E"], hc2["I"], hc2["R"]
        Sv_arr[t], Ev_arr[t], Iv_arr[t] = yv[0], yv[1], yv[2]
        new_inf_h[t] = new_inf
        obs_cases[t] = obs

    result: Dict[str, np.ndarray] = {
        "Sh": Sh, "Eh": Eh, "Ih": Ih, "Rh": Rh,
        "Sv": Sv_arr, "Ev": Ev_arr, "Iv": Iv_arr,
        "new_inf_h": new_inf_h,
        "obs_cases": obs_cases,
        "rain": rain_sm,
    }
    if hp_human is not None:
        result.update({
            "Sh_hum": Sh_hum, "Eh_hum": Eh_hum,
            "Ih_hum": Ih_hum, "Rh_hum": Rh_hum,
            "new_inf_hum": new_inf_hum,
            "obs_hum": obs_hum,
        })
    return result


# =============================================================================
# 7) ENSEMBLE RUNNER
# =============================================================================

def run_ensemble(rain: np.ndarray,
                 sp: SimParams = SimParams(),
                 hp: HostParams = HostParams(),
                 vp: VectorParams = VectorParams(),
                 op: ObsParams = ObsParams(),
                 hp_human: "HumanParams | None" = None,
                 ip: "InterventionParams | None" = None,
                 n_runs: int = 50) -> Dict[str, np.ndarray]:
    """
    Run simulate() n_runs times with sequential seeds.

    Returns a dict where each key maps to a 2-D array (n_runs, T).
    The "rain" key is 1-D (deterministic, same for all runs).
    """
    runs = []
    for i in range(n_runs):
        sp_i = _dc_replace(sp, seed=sp.seed + i)
        runs.append(simulate(rain, sp_i, hp, vp, op, hp_human, ip))

    keys = list(runs[0].keys())
    ens: Dict[str, np.ndarray] = {"rain": runs[0]["rain"]}
    for k in keys:
        if k == "rain":
            continue
        ens[k] = np.stack([r[k].astype(float) for r in runs], axis=0)  # (n_runs, T)
    return ens


# =============================================================================
# 8) R₀ CALCULATOR
# =============================================================================

def compute_r0(sp: SimParams, hp: HostParams,
               vp: VectorParams) -> Dict[str, float]:
    """
    Basic reproduction number at the disease-free equilibrium.

    Derived from the Next-Generation Matrix for the SEIR-SEI system:
        R0 = b * sqrt( p_vh * p_hv * m * σ / ((σ+μ) * μ * γ_h) )

    where m = Sv0/Nh  (vector-to-host ratio at DFE)
          γ_h = 1/infectious_days  (host recovery rate)
    """
    m = sp.Sv0 / max(sp.n_hosts, 1)
    gamma_h = 1.0 / hp.infectious_days

    # Component from host → vector (infectious vectors produced per infectious host)
    R0_hv = (vp.b * vp.p_hv * m / gamma_h) * (vp.sigma / (vp.sigma + vp.mu))

    # Component from vector → host (host infections caused per infectious vector)
    R0_vh = vp.b * hp.p_vh / vp.mu

    R0 = float(np.sqrt(max(R0_hv * R0_vh, 0.0)))
    return {"R0": R0, "R0_hv": R0_hv, "R0_vh": R0_vh, "m": m}


# =============================================================================
# 9) SENSITIVITY ANALYSIS (PRCC)
# =============================================================================

def run_sensitivity(rain: np.ndarray,
                    sp_base: SimParams,
                    hp_base: HostParams,
                    vp_base: VectorParams,
                    op_base: ObsParams,
                    n_samples: int = 100) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Latin-Hypercube-based PRCC sensitivity analysis.

    Varies 8 biological parameters over plausible ranges and measures their
    partial rank correlation with total livestock infections.

    Returns (param_names, prcc_values, p_values).
    """
    if not SCIPY_STATS_OK:
        raise RuntimeError("scipy.stats is required for sensitivity analysis.")

    param_ranges: Dict[str, Tuple[float, float]] = {
        "latent_days":     (1.0, 10.0),
        "infectious_days": (2.0, 15.0),
        "p_vh":            (0.05, 0.50),
        "mu":              (0.03, 0.30),
        "sigma":           (0.05, 0.50),
        "b":               (0.10, 0.80),
        "p_hv":            (0.05, 0.40),
        "alpha":           (10.0, 200.0),
    }
    param_names = list(param_ranges.keys())
    n_params = len(param_names)

    # Latin Hypercube Sampling
    lhs_rng = np.random.default_rng(42)
    X = np.zeros((n_samples, n_params))
    for j, (lo, hi) in enumerate(param_ranges.values()):
        strata = (lhs_rng.random(n_samples) + np.arange(n_samples)) / n_samples
        lhs_rng.shuffle(strata)
        X[:, j] = lo + strata * (hi - lo)

    # Use a smaller n_hosts for speed
    sp_fast = _dc_replace(sp_base, n_hosts=200, init_host_I=max(sp_base.init_host_I, 2))
    outcomes = np.zeros(n_samples)

    for i in range(n_samples):
        vals = {k: X[i, j] for j, k in enumerate(param_names)}
        hp_i = _dc_replace(hp_base,
                           latent_days=vals["latent_days"],
                           infectious_days=vals["infectious_days"],
                           p_vh=vals["p_vh"])
        vp_i = _dc_replace(vp_base,
                           mu=vals["mu"],
                           sigma=vals["sigma"],
                           b=vals["b"],
                           p_hv=vals["p_hv"],
                           alpha=vals["alpha"])
        sp_i = _dc_replace(sp_fast, seed=sp_base.seed + i)
        out_i = simulate(rain, sp_i, hp_i, vp_i, op_base)
        outcomes[i] = float(out_i["new_inf_h"].sum())

    # PRCC via rank-transform + OLS residuals
    Xr = np.column_stack([rankdata(X[:, j]) for j in range(n_params)])
    yr = rankdata(outcomes)
    prcc = np.zeros(n_params)
    pvals = np.zeros(n_params)

    for j in range(n_params):
        others = [k for k in range(n_params) if k != j]
        Z = np.column_stack([np.ones(n_samples), Xr[:, others]])

        coef_x, *_ = np.linalg.lstsq(Z, Xr[:, j], rcond=None)
        res_x = Xr[:, j] - Z @ coef_x

        coef_y, *_ = np.linalg.lstsq(Z, yr, rcond=None)
        res_y = yr - Z @ coef_y

        r, p = pearsonr(res_x, res_y)
        prcc[j] = r
        pvals[j] = p

    return param_names, prcc, pvals


# =============================================================================
# 10) QUICK-START EXAMPLE
# =============================================================================

if __name__ == "__main__":
    days = 365
    rain = np.zeros(days)
    rain[60:120] = 10.0
    rain[120:160] = 3.0

    out = simulate(rain=rain)
    print("Host counts (final):", {k: int(out[k][-1]) for k in ["Sh", "Eh", "Ih", "Rh"]})
    print("Total obs cases:", int(out["obs_cases"].sum()))

    r0_info = compute_r0(SimParams(), HostParams(), VectorParams())
    print(f"R0 = {r0_info['R0']:.3f}  (R0_hv={r0_info['R0_hv']:.3f}, R0_vh={r0_info['R0_vh']:.3f})")
