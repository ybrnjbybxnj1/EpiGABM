from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

_STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.fancybox": False,
    "legend.edgecolor": "0.8",
    "xtick.direction": "out",
    "ytick.direction": "out",
}
plt.rcParams.update(_STYLE)

_COLORS = {
    "history_matching": "#1f77b4",
    "rejection": "#2ca02c",
    "annealing": "#ff7f0e",
    "smc": "#9467bd",
}
_LABELS = {
    "history_matching": "History Matching",
    "rejection": "ABC Rejection",
    "annealing": "ABC Annealing",
    "smc": "ABC-SMC",
}
_OBSERVED_COLOR = "#2b2b2b"
_GABM_COLOR = "#d62728"

from experiments.helpers import (
    load_config,
    make_llama_gabm,
    make_small_population,
    save_experiment_json,
    save_figure,
)
from src.calibration.abc import ABCSampler
from src.calibration.history_matching import HistoryMatcher
from src.models.abm import ABMSimulation
from src.models.data_structures import SEIRState
from src.models.seir import SEIRModel
from src.regime.threshold import ThresholdDetector

def _extract_I(result, strain: str = "H1N1") -> np.ndarray:
    return np.array([dr.I.get(strain, 0) for dr in result.days])

def _to_weekly(daily: np.ndarray, agg: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    n = len(daily)
    n_weeks = n // 7
    if n_weeks == 0:
        return np.array([0]), np.array([daily.mean() if agg == "mean" else daily.sum()])
    truncated = daily[: n_weeks * 7].reshape(n_weeks, 7)
    values = truncated.mean(axis=1) if agg == "mean" else truncated.sum(axis=1)
    centers = np.arange(n_weeks) * 7 + 3.5
    return centers, values

def _seir_initial_from_gabm(
    gabm_result, switch_day: int, n_agents: int, strain: str = "H1N1",
) -> SEIRState:
    idx = min(switch_day - 1, len(gabm_result.days) - 1)
    dr = gabm_result.days[idx]
    return SEIRState(
        S=dr.S.get(strain, 0),
        E=dr.E.get(strain, 0),
        I=dr.I.get(strain, 0),
        R=dr.R.get(strain, 0),
        N=n_agents,
    )

def _build_seir_forecaster(
    config: dict, initial: SEIRState, forecast_days: int,
):
    seir = SEIRModel(config)
    default_gamma = seir.gamma

    def run_model(params: dict) -> pd.Series:
        beta = float(params.get("beta", 0.3))
        rho = float(params.get("rho", 0.0))
        gamma_override = params.get("gamma", None)

        adj_i = int(initial.I * (1.0 + rho))
        adj_s = max(initial.S - (adj_i - initial.I), 0)
        state = SEIRState(
            S=adj_s, E=initial.E, I=max(adj_i, 1),
            R=initial.R, N=initial.N,
        )
        if gamma_override is not None:
            seir.gamma = float(gamma_override)
        try:
            states = seir.run(
                days=range(forecast_days + 1), initial=state,
                beta=beta, stochastic=False,
            )
        finally:
            seir.gamma = default_gamma

        return pd.Series([s.I for s in states[1:]])

    return run_model

def run_experiment(
    config_path: str = "config/default.yaml",
    seed: int = 42,
    n_agents: int = 3000,
    n_initial_samples: int = 200,
    n_hm_accept: int = 60,
    n_particles: int = 100,
) -> dict:
    config = load_config(config_path)
    data, households = make_small_population(n=n_agents, seed=seed)
    days = range(1, config["model"]["days"][1])
    strain = "H1N1"

    logger.info("exp8: running full gabm (ground truth)")
    gabm = make_llama_gabm(config, data.copy(), households.copy())
    gabm_result = gabm.run(days=days, seed=seed)
    gabm_I = _extract_I(gabm_result, strain)

    logger.info("exp8: running rule-based abm reference (not calibrated against)")
    obs_config = deepcopy(config)
    obs_config["model"]["lmbd"] = 0.4
    obs_config["model"]["alpha"] = {"H1N1": 0.78, "H3N2": 0.0, "B": 0.0}
    obs_config["model"]["infected_init"] = {"H1N1": 10, "H3N2": 0, "B": 0}
    obs_abm = ABMSimulation(config=obs_config, data=data.copy(), households=households.copy())
    obs_result = obs_abm.run(days=days, seed=seed)
    abm_reference_I = _extract_I(obs_result, strain)

    observed_I = gabm_I

    thresh_cfg = config.get("regime", {}).get("threshold", {})
    detector = ThresholdDetector(thresh_cfg, population_size=n_agents)

    switch_day = None
    prev_inf = 0
    from src.models.data_structures import EpidemicContext
    for dr in gabm_result.days:
        total = dr.I.get(strain, 0) + dr.E.get(strain, 0)
        gr = (total - prev_inf) / max(prev_inf, 1) if prev_inf > 0 else 0.0
        ctx = EpidemicContext(
            day=dr.day, total_infected=total,
            total_susceptible=dr.S.get(strain, 0),
            total_recovered=dr.R.get(strain, 0),
            growth_rate=gr,
            new_infections_today=dr.new_infections.get(strain, 0),
            phase=None,
        )
        if detector.should_switch(ctx):
            switch_day = dr.day
            break
        prev_inf = total

    if switch_day is None:
        switch_day = int(np.argmax(gabm_I)) + 1
        logger.warning("exp8: detector did not trigger, falling back to peak day {}", switch_day)

    logger.info("exp8: switch_day={}", switch_day)

    forecast_len = len(days) - switch_day

    active_idx = np.where(gabm_I > 0)[0]
    active_end = int(active_idx.max()) + 1 if len(active_idx) else len(gabm_I)
    active_end_rel = max(active_end - (switch_day - 1), 1)
    logger.info(
        "exp8: active window for calibration+coverage = days {}..{} ({} relative steps)",
        switch_day, active_end, active_end_rel,
    )

    target = pd.Series(
        gabm_I[switch_day - 1: switch_day - 1 + active_end_rel]
    )

    initial_state = _seir_initial_from_gabm(gabm_result, switch_day, n_agents, strain)
    run_model = _build_seir_forecaster(config, initial_state, forecast_len)

    gamma = float(config["model"]["seir"]["gamma"])
    beta_scale = gamma / n_agents
    param_ranges = {
        "beta": (0.3 * beta_scale, 5.0 * beta_scale),
        "rho": (-0.3, 0.3),
        "gamma": (gamma, gamma * 4.0),
    }
    logger.info(
        "exp8: prior beta in [{:.2e}, {:.2e}], gamma in [{:.3f}, {:.3f}]",
        param_ranges["beta"][0], param_ranges["beta"][1],
        param_ranges["gamma"][0], param_ranges["gamma"][1],
    )

    hm = HistoryMatcher()
    abc = ABCSampler()

    logger.info("exp8: history matching")
    prior_samples = hm.generate_samples(param_ranges, n_initial_samples)
    hm_distances = []
    for _, row in prior_samples.iterrows():
        sim = run_model(row.to_dict())
        hm_distances.append(abc._mse(sim, target))
    prior_samples["_hm_distance"] = hm_distances
    hm_threshold = float(prior_samples["_hm_distance"].nsmallest(n_hm_accept).max())
    hm_accepted = prior_samples[prior_samples["_hm_distance"] < hm_threshold].drop(
        columns=["_hm_distance"],
    )
    hm_best = float(prior_samples["_hm_distance"].min())
    logger.info("exp8: hm kept {} samples (best mse={:.1f})", len(hm_accepted), hm_best)

    logger.info("exp8: abc rejection")
    rej_threshold = float(prior_samples["_hm_distance"].quantile(0.3))
    rej_result = abc.rejection(
        run_model, target, hm_accepted, n_samples=n_particles, threshold=rej_threshold,
    )

    logger.info("exp8: abc annealing")
    ann_result = abc.annealing(
        run_model, target, hm_accepted, n_samples=n_particles,
    )

    logger.info("exp8: abc smc")
    smc_result = abc.smc(
        run_model, target, hm_accepted, n_particles=n_particles,
    )

    default_gamma = float(config["model"]["seir"]["gamma"])

    def posterior_summary(post_samples: pd.DataFrame) -> dict:
        if len(post_samples) == 0:
            return {
                "beta": gamma / n_agents,
                "rho": 0.0,
                "gamma": default_gamma,
            }
        summary = {
            "beta": float(post_samples["beta"].median()),
            "rho": float(post_samples["rho"].median()),
        }
        if "gamma" in post_samples.columns:
            summary["gamma"] = float(post_samples["gamma"].median())
        else:
            summary["gamma"] = default_gamma
        return summary

    methods = {
        "history_matching": hm_accepted,
        "rejection": rej_result.posterior_samples,
        "annealing": ann_result.posterior_samples,
        "smc": smc_result.posterior_samples,
    }

    forecasts: dict[str, np.ndarray] = {}
    best_distances: dict[str, float] = {}
    for name, samples in methods.items():
        summary = posterior_summary(samples)
        sim = run_model(summary).values
        forecasts[name] = sim
        best_distances[name] = float(abc._mse(pd.Series(sim), target))

    smc_trajectories = []
    if len(smc_result.posterior_samples) > 0:
        n_ci = min(200, len(smc_result.posterior_samples))
        sub = smc_result.posterior_samples.sample(
            n=n_ci, replace=n_ci > len(smc_result.posterior_samples),
        )
        for _, row in sub.iterrows():
            smc_trajectories.append(run_model(row.to_dict()).values)
    if smc_trajectories:
        trajectories_arr = np.array(smc_trajectories)
        param_lower = np.percentile(trajectories_arr, 2.5, axis=0)
        param_upper = np.percentile(trajectories_arr, 97.5, axis=0)
        param_median = np.median(trajectories_arr, axis=0)
        obs_sigma = np.sqrt(np.maximum(param_median, 1.0))
        ci_lower = np.maximum(param_lower - 1.96 * obs_sigma, 0.0)
        ci_upper = param_upper + 1.96 * obs_sigma
    else:
        ci_lower = ci_upper = np.zeros(forecast_len)

    colors = _COLORS
    labels = _LABELS

    fig_a, ax_a = plt.subplots(figsize=(12, 6.5))
    x_obs = np.arange(1, len(observed_I) + 1)
    x_fc = np.arange(switch_day, switch_day + forecast_len)

    ax_a.axvspan(1, switch_day, color="#e8eef5", alpha=0.55, zorder=0)
    ax_a.axvspan(switch_day, len(observed_I) + 1, color="#fdf3e7", alpha=0.4, zorder=0)

    ax_a.fill_between(x_fc, ci_lower, ci_upper, color=colors["smc"],
                      alpha=0.15, zorder=1, label="SMC 95% CI")

    ax_a.plot(x_obs, abm_reference_I, color="#b0b0b0", linewidth=1.2,
              linestyle=":", label="rule-based ABM (no behavior)", zorder=2)

    ax_a.plot(x_obs, observed_I, color=_OBSERVED_COLOR, linewidth=2.3,
              label="GABM (ground truth)", zorder=4)
    ax_a.plot(np.arange(1, switch_day + 1), gabm_I[:switch_day],
              color=_GABM_COLOR, linewidth=2.2, label="GABM (pre-switch)", zorder=5)

    for name, fc in forecasts.items():
        ax_a.plot(x_fc, fc, color=colors[name], linewidth=1.6,
                  linestyle="--", label=f"SEIR - {labels[name]}", zorder=3)

    ax_a.axvline(switch_day, color="#333333", linestyle="-.",
                 linewidth=1.2, alpha=0.8, zorder=2)
    ax_a.annotate(
        f"switch day {switch_day}", xy=(switch_day, ax_a.get_ylim()[1] * 0.92),
        xytext=(switch_day + 5, ax_a.get_ylim()[1] * 0.92),
        fontsize=9, color="#333333",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8),
    )

    ax_a.set_xlabel("simulation day")
    ax_a.set_ylabel("infectious population I(t)")
    ax_a.set_title("Hybrid GABM -> SEIR forecast (SEIR calibrated on GABM)")
    ax_a.set_xlim(1, len(observed_I) + 1)
    ax_a.legend(fontsize=9, loc="upper right", ncol=2,
                title="model", title_fontsize=9)
    plt.tight_layout()
    save_figure(fig_a, "exp8_forecast_full", config)
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(10, 5.5))
    zoom_start = max(switch_day - 10, 1)
    zoom_end = min(switch_day + 20, len(observed_I))
    xs_zoom = np.arange(zoom_start, zoom_end + 1)

    ax_b.axvspan(zoom_start, switch_day, color="#e8eef5", alpha=0.55)
    ax_b.axvspan(switch_day, zoom_end + 1, color="#fdf3e7", alpha=0.4)

    mask_zoom = (x_fc >= zoom_start) & (x_fc <= zoom_end)
    ax_b.fill_between(
        x_fc[mask_zoom], ci_lower[mask_zoom], ci_upper[mask_zoom],
        color=colors["smc"], alpha=0.22, label="SMC 95% CI",
    )

    ax_b.plot(xs_zoom, observed_I[zoom_start - 1: zoom_end],
              color=_OBSERVED_COLOR, linewidth=2.3,
              marker="o", markersize=4, label="GABM (ground truth)")
    ax_b.plot(xs_zoom[:switch_day - zoom_start + 1],
              gabm_I[zoom_start - 1: switch_day],
              color=_GABM_COLOR, linewidth=2.2,
              marker="s", markersize=4, label="GABM")
    ax_b.plot(x_fc[mask_zoom], forecasts["smc"][mask_zoom],
              color=colors["smc"], linewidth=1.8, linestyle="--",
              marker="d", markersize=4, label="SEIR (SMC median)")

    ax_b.axvline(switch_day, color="#333333", linestyle="-.", linewidth=1.2)
    ax_b.text(switch_day, ax_b.get_ylim()[1] * 0.97,
              f" switch day {switch_day}", fontsize=9, color="#333333",
              va="top")

    ax_b.set_xlabel("simulation day")
    ax_b.set_ylabel("infectious population I(t)")
    ax_b.set_title(f"Switch region: day {zoom_start} -> {zoom_end}")
    ax_b.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    save_figure(fig_b, "exp8_switch_zoom", config)
    plt.close(fig_b)

    fig_c, ax_c = plt.subplots(figsize=(9, 5.5))
    for name, samples in methods.items():
        if len(samples) == 0 or "beta" not in samples.columns:
            continue
        vals = samples["beta"].values
        ax_c.hist(vals, bins=18, alpha=0.45,
                  label=labels[name], color=colors[name], density=True,
                  edgecolor="white", linewidth=0.5)
        med = float(np.median(vals))
        ax_c.axvline(med, color=colors[name], linestyle="--",
                     linewidth=1.4, alpha=0.85)

    ax_c.set_xlabel(r"$\beta$ (transmission rate)")
    ax_c.set_ylabel("posterior density")
    ax_c.set_title(r"Posterior distribution of $\beta$ across calibration methods")
    ax_c.legend(fontsize=9, title="method", title_fontsize=9)
    plt.tight_layout()
    save_figure(fig_c, "exp8_posterior_beta", config)
    plt.close(fig_c)

    fig_d, ax_d = plt.subplots(figsize=(9, 5.5))
    for name, samples in methods.items():
        if len(samples) == 0 or "rho" not in samples.columns:
            continue
        vals = samples["rho"].values
        ax_d.hist(vals, bins=18, alpha=0.45,
                  label=labels[name], color=colors[name], density=True,
                  edgecolor="white", linewidth=0.5)
        med = float(np.median(vals))
        ax_d.axvline(med, color=colors[name], linestyle="--",
                     linewidth=1.4, alpha=0.85)

    ax_d.axvline(0, color="#888888", linestyle=":", linewidth=0.8)
    ax_d.set_xlabel(r"$\rho$ (initial-$I$ perturbation)")
    ax_d.set_ylabel("posterior density")
    ax_d.set_title(r"Posterior distribution of $\rho$ across calibration methods")
    ax_d.legend(fontsize=9, title="method", title_fontsize=9)
    plt.tight_layout()
    save_figure(fig_d, "exp8_posterior_rho", config)
    plt.close(fig_d)

    sorted_items = sorted(best_distances.items(), key=lambda kv: kv[1])
    best_name = sorted_items[0][0]
    fig_e, ax_e = plt.subplots(figsize=(8, 5.5))
    bar_colors = [colors[m] for m, _ in sorted_items]
    bar_alphas = [0.95 if m == best_name else 0.7 for m, _ in sorted_items]
    bars = ax_e.bar(
        [labels[m] for m, _ in sorted_items],
        [v for _, v in sorted_items],
        color=bar_colors,
        edgecolor="white", linewidth=1.2,
    )
    for bar, alpha in zip(bars, bar_alphas):
        bar.set_alpha(alpha)
    for i, (name, val) in enumerate(sorted_items):
        marker = "  *" if name == best_name else ""
        ax_e.text(i, val, f"{val:.0f}{marker}",
                  ha="center", va="bottom", fontsize=10,
                  fontweight="bold" if name == best_name else "normal")

    ax_e.set_ylabel("MSE vs GABM ground truth (lower is better)")
    ax_e.set_title("Best distance by calibration method")
    plt.setp(ax_e.get_xticklabels(), rotation=10, ha="right")
    ax_e.margins(y=0.15)
    plt.tight_layout()
    save_figure(fig_e, "exp8_method_comparison", config)
    plt.close(fig_e)

    fig_f, ax_f = plt.subplots(figsize=(11, 5.5))
    ax_f.axvspan(0, switch_day - 1, color="#e8eef5", alpha=0.55)
    ax_f.axvspan(switch_day - 1, len(observed_I), color="#fdf3e7", alpha=0.4)

    ax_f.plot(observed_I, color="#6c757d", linewidth=1.7,
              label="rule-based ABM (synthetic)")
    ax_f.plot(gabm_I, color="#1a6bb0", linewidth=2.0, label="full GABM")
    hybrid_curve = np.concatenate([gabm_I[:switch_day], forecasts["smc"]])
    ax_f.plot(hybrid_curve[:len(observed_I)],
              color=colors["smc"], linewidth=1.9, linestyle="--",
              label="hybrid GABM -> SEIR (SMC)")
    ax_f.axvline(switch_day - 1, color="#333333", linestyle="-.",
                 linewidth=1.2, alpha=0.8)
    ax_f.text(switch_day - 1, ax_f.get_ylim()[1] * 0.97,
              f" switch day {switch_day}", fontsize=9, color="#333333", va="top")

    ax_f.set_xlabel("simulation day")
    ax_f.set_ylabel("infectious population I(t)")
    ax_f.set_title("Curve decomposition: ABM vs GABM vs hybrid")
    ax_f.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    save_figure(fig_f, "exp8_decomposition", config)
    plt.close(fig_f)

    fig_g, ax_g = plt.subplots(figsize=(11, 4.5))
    x_fc = np.arange(switch_day - 1, switch_day - 1 + forecast_len)
    obs_slice = observed_I[switch_day - 1: switch_day - 1 + forecast_len]
    active_mask = np.arange(len(obs_slice)) < active_end_rel
    in_ci_full = (obs_slice >= ci_lower) & (obs_slice <= ci_upper)
    in_ci = in_ci_full & active_mask
    out_ci = (~in_ci_full) & active_mask
    ax_g.fill_between(x_fc, ci_lower, ci_upper, color=colors["smc"],
                      alpha=0.22, label="SMC 95% CI")
    ax_g.plot(x_fc, obs_slice, color=_OBSERVED_COLOR, linewidth=1.6,
              label="GABM (ground truth)", zorder=2)
    ax_g.scatter(x_fc[in_ci], obs_slice[in_ci], color="#2ca02c",
                 s=36, zorder=4, edgecolor="white", linewidth=0.7,
                 label=f"in CI ({int(in_ci.sum())})")
    ax_g.scatter(x_fc[out_ci], obs_slice[out_ci], color="#d62728",
                 s=36, zorder=4, edgecolor="white", linewidth=0.7,
                 label=f"out of CI ({int(out_ci.sum())})")
    ax_g.axvline(switch_day - 1 + active_end_rel - 1, color="#888",
                 linestyle=":", linewidth=1, alpha=0.6)
    ax_g.text(switch_day - 1 + active_end_rel - 1,
              ax_g.get_ylim()[1] * 0.95, " active end",
              fontsize=8, color="#666", va="top")
    n_active = int(active_mask.sum())
    coverage = float(in_ci.sum() / n_active) if n_active else 0.0
    ax_g.set_xlabel("simulation day")
    ax_g.set_ylabel("infectious I(t)")
    ax_g.set_title(
        f"Forecast coverage on active window (days {switch_day}..{switch_day+active_end_rel-1}): "
        f"{coverage:.1%}"
    )
    ax_g.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    save_figure(fig_g, "exp8_coverage", config)
    plt.close(fig_g)

    obs_w_x, obs_w_y = _to_weekly(observed_I, "mean")
    gabm_w_x, gabm_w_y = _to_weekly(gabm_I, "mean")
    pre_switch_weeks_mask = gabm_w_x < switch_day
    switch_week = switch_day / 7

    fig_aw, ax_aw = plt.subplots(figsize=(12, 6.5))
    max_week = len(observed_I) / 7
    ax_aw.axvspan(0, switch_week, color="#e8eef5", alpha=0.55)
    ax_aw.axvspan(switch_week, max_week, color="#fdf3e7", alpha=0.4)

    ax_aw.plot(obs_w_x / 7, obs_w_y, color=_OBSERVED_COLOR, linewidth=2.2,
               marker="o", markersize=6, label="GABM ground truth (weekly)",
               zorder=4)
    ax_aw.plot(gabm_w_x[pre_switch_weeks_mask] / 7,
               gabm_w_y[pre_switch_weeks_mask],
               color=_GABM_COLOR, linewidth=2.0, marker="s", markersize=5,
               label="GABM (pre-switch)", zorder=5)

    for name, fc in forecasts.items():
        full_trajectory = np.concatenate([gabm_I[:switch_day], fc])[:len(observed_I)]
        _, fc_weekly = _to_weekly(full_trajectory, "mean")
        fc_week_centers = np.arange(len(fc_weekly)) * 7 + 3.5
        post_mask = fc_week_centers >= switch_day
        ax_aw.plot(fc_week_centers[post_mask] / 7, fc_weekly[post_mask],
                   color=colors[name], linewidth=1.5, linestyle="--",
                   marker="d", markersize=4,
                   label=f"SEIR - {labels[name]}", zorder=3)

    ax_aw.axvline(switch_week, color="#333333", linestyle="-.",
                  linewidth=1.2, alpha=0.8)
    ax_aw.text(switch_week, ax_aw.get_ylim()[1] * 0.97,
               f" switch week {switch_week:.1f}", fontsize=9,
               color="#333333", va="top")

    ax_aw.set_xlabel("week of simulation")
    ax_aw.set_ylabel("mean infectious I(t) per week")
    ax_aw.set_title("Forecast at weekly resolution (matches real surveillance)")
    ax_aw.legend(fontsize=9, loc="upper right", ncol=2,
                 title="model", title_fontsize=9)
    plt.tight_layout()
    save_figure(fig_aw, "exp8_forecast_full_weekly", config)
    plt.close(fig_aw)

    smc_full = np.concatenate([gabm_I[:switch_day], forecasts["smc"]])[:len(observed_I)]
    _, smc_w = _to_weekly(smc_full, "mean")
    smc_w_centers = np.arange(len(smc_w)) * 7 + 3.5
    post_mask_smc = smc_w_centers >= switch_day

    fig_bw, ax_bw = plt.subplots(figsize=(10, 5.5))
    zoom_start_w = max((switch_day - 14) / 7, 0)
    zoom_end_w = min((switch_day + 28) / 7, max_week)
    ax_bw.axvspan(zoom_start_w, switch_week, color="#e8eef5", alpha=0.55)
    ax_bw.axvspan(switch_week, zoom_end_w, color="#fdf3e7", alpha=0.4)

    ax_bw.plot(obs_w_x / 7, obs_w_y, color=_OBSERVED_COLOR, linewidth=2.2,
               marker="o", markersize=6, label="GABM (ground truth)")
    ax_bw.plot(gabm_w_x[pre_switch_weeks_mask] / 7,
               gabm_w_y[pre_switch_weeks_mask],
               color=_GABM_COLOR, linewidth=2.0, marker="s", markersize=5,
               label="GABM")
    ax_bw.plot(smc_w_centers[post_mask_smc] / 7, smc_w[post_mask_smc],
               color=colors["smc"], linewidth=1.8, linestyle="--",
               marker="d", markersize=5, label="SEIR (SMC median)")
    ax_bw.axvline(switch_week, color="#333333", linestyle="-.", linewidth=1.2)
    ax_bw.set_xlim(zoom_start_w, zoom_end_w)
    ax_bw.set_xlabel("week")
    ax_bw.set_ylabel("mean infectious I(t) per week")
    ax_bw.set_title(f"Switch region zoom (weekly, week {zoom_start_w:.1f} -> {zoom_end_w:.1f})")
    ax_bw.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    save_figure(fig_bw, "exp8_switch_zoom_weekly", config)
    plt.close(fig_bw)

    fig_fw, ax_fw = plt.subplots(figsize=(11, 5.5))
    ax_fw.axvspan(0, switch_week, color="#e8eef5", alpha=0.55)
    ax_fw.axvspan(switch_week, max_week, color="#fdf3e7", alpha=0.4)

    ax_fw.plot(obs_w_x / 7, obs_w_y, color="#6c757d", linewidth=1.8,
               marker="o", markersize=5, label="rule-based ABM (synthetic)")
    ax_fw.plot(gabm_w_x / 7, gabm_w_y, color="#1a6bb0", linewidth=2.0,
               marker="s", markersize=5, label="full GABM")
    ax_fw.plot(smc_w_centers / 7, smc_w, color=colors["smc"], linewidth=1.9,
               linestyle="--", marker="d", markersize=5,
               label="hybrid GABM -> SEIR (SMC)")
    ax_fw.axvline(switch_week, color="#333333", linestyle="-.", linewidth=1.2)
    ax_fw.text(switch_week, ax_fw.get_ylim()[1] * 0.97,
               f" switch week {switch_week:.1f}", fontsize=9,
               color="#333333", va="top")
    ax_fw.set_xlabel("week")
    ax_fw.set_ylabel("mean infectious I(t) per week")
    ax_fw.set_title("Curve decomposition (weekly)")
    ax_fw.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    save_figure(fig_fw, "exp8_decomposition_weekly", config)
    plt.close(fig_fw)

    fig_comp, axs = plt.subplots(2, 2, figsize=(14, 9.5))
    gs = axs[0, 0].get_gridspec()
    for ax in axs[0]:
        ax.remove()
    ax_top = fig_comp.add_subplot(gs[0, :])

    ax_top.axvspan(0, switch_day - 1, color="#e8eef5", alpha=0.55, zorder=0)
    ax_top.axvspan(switch_day - 1, len(observed_I), color="#fdf3e7", alpha=0.4, zorder=0)
    ax_top.fill_between(
        np.arange(switch_day - 1, switch_day - 1 + forecast_len),
        ci_lower, ci_upper, color=colors["smc"], alpha=0.15, label="SMC 95% CI",
    )
    ax_top.plot(observed_I, color=_OBSERVED_COLOR, linewidth=2.2,
                label="GABM (ground truth)")
    ax_top.plot(np.arange(switch_day), gabm_I[:switch_day],
                color=_GABM_COLOR, linewidth=2.0, label="GABM (pre-switch)")
    for name, fc in forecasts.items():
        x = np.arange(switch_day - 1, switch_day - 1 + len(fc))
        ax_top.plot(x, fc, color=colors[name], linewidth=1.5,
                    linestyle="--", label=f"SEIR - {labels[name]}")
    ax_top.axvline(switch_day - 1, color="#333333", linestyle="-.",
                   linewidth=1.2, alpha=0.8)
    ax_top.set_xlabel("simulation day")
    ax_top.set_ylabel("infectious I(t)")
    ax_top.set_title("(A) Forecast - observed vs GABM vs SEIR", loc="left")
    ax_top.legend(fontsize=8, loc="upper right", ncol=2)

    ax_bl = axs[1, 0]
    for name, samples in methods.items():
        if len(samples) == 0 or "beta" not in samples.columns:
            continue
        ax_bl.hist(samples["beta"].values, bins=15, alpha=0.45,
                   label=labels[name], color=colors[name], density=True,
                   edgecolor="white", linewidth=0.5)
        ax_bl.axvline(float(np.median(samples["beta"].values)),
                      color=colors[name], linestyle="--", linewidth=1.3, alpha=0.85)
    ax_bl.set_xlabel(r"$\beta$")
    ax_bl.set_ylabel("density")
    ax_bl.set_title(r"(C) Posterior distribution of $\beta$", loc="left")
    ax_bl.legend(fontsize=8)

    ax_br = axs[1, 1]
    sorted_items_c = sorted(best_distances.items(), key=lambda kv: kv[1])
    best_c = sorted_items_c[0][0]
    bars_c = ax_br.bar(
        [labels[m] for m, _ in sorted_items_c],
        [v for _, v in sorted_items_c],
        color=[colors[m] for m, _ in sorted_items_c],
        edgecolor="white", linewidth=1.2,
    )
    for bar, (m, _) in zip(bars_c, sorted_items_c):
        bar.set_alpha(0.95 if m == best_c else 0.7)
    for i, (name, val) in enumerate(sorted_items_c):
        marker = "  *" if name == best_c else ""
        ax_br.text(i, val, f"{val:.0f}{marker}",
                   ha="center", va="bottom", fontsize=9,
                   fontweight="bold" if name == best_c else "normal")
    ax_br.set_ylabel("MSE vs observed")
    ax_br.set_title("(D) Best distance by method", loc="left")
    plt.setp(ax_br.get_xticklabels(), rotation=10, ha="right", fontsize=9)
    ax_br.margins(y=0.15)

    fig_comp.suptitle("Calibration and forecast summary - exp8",
                      fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    save_figure(fig_comp, "exp8_composite", config)
    plt.close(fig_comp)

    fig_x, ax_x = plt.subplots(3, 2, figsize=(14.5, 14))
    mask_zoom_x = (x_fc >= zoom_start - 1) & (x_fc < zoom_end)

    ax = ax_x[0, 0]
    ax.axvspan(0, switch_day - 1, color="#e8eef5", alpha=0.55)
    ax.axvspan(switch_day - 1, len(observed_I), color="#fdf3e7", alpha=0.4)
    ax.fill_between(
        np.arange(switch_day - 1, switch_day - 1 + forecast_len),
        ci_lower, ci_upper, color=colors["smc"], alpha=0.15,
    )
    ax.plot(observed_I, color=_OBSERVED_COLOR, linewidth=1.8, label="GABM (gt)")
    ax.plot(np.arange(switch_day), gabm_I[:switch_day],
            color=_GABM_COLOR, linewidth=1.6, label="GABM")
    for name, fc in forecasts.items():
        x = np.arange(switch_day - 1, switch_day - 1 + len(fc))
        ax.plot(x, fc, color=colors[name], linewidth=1.1, linestyle="--",
                label=labels[name])
    ax.axvline(switch_day - 1, color="#333", linestyle="-.", linewidth=1.0)
    ax.set_xlabel("day")
    ax.set_ylabel("I(t)")
    ax.set_title("(A) Full forecast", loc="left", fontsize=11)
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    ax = ax_x[0, 1]
    ax.axvspan(zoom_start - 1, switch_day - 1, color="#e8eef5", alpha=0.55)
    ax.axvspan(switch_day - 1, zoom_end, color="#fdf3e7", alpha=0.4)
    ax.fill_between(x_fc[mask_zoom_x], ci_lower[mask_zoom_x],
                    ci_upper[mask_zoom_x], color=colors["smc"], alpha=0.22,
                    label="SMC 95% CI")
    ax.plot(xs_zoom, observed_I[zoom_start - 1: zoom_end],
            color=_OBSERVED_COLOR, linewidth=1.8, marker="o", markersize=3,
            label="GABM (gt)")
    ax.plot(xs_zoom[:switch_day - zoom_start + 1],
            gabm_I[zoom_start - 1: switch_day],
            color=_GABM_COLOR, linewidth=1.6, marker="s", markersize=3,
            label="GABM")
    ax.plot(x_fc[mask_zoom_x], forecasts["smc"][mask_zoom_x],
            color=colors["smc"], linewidth=1.5, linestyle="--",
            marker="d", markersize=3, label="SEIR (SMC)")
    ax.axvline(switch_day - 1, color="#333", linestyle="-.", linewidth=1.0)
    ax.set_xlabel("day")
    ax.set_ylabel("I(t)")
    ax.set_title("(B) Switch region zoom", loc="left", fontsize=11)
    ax.legend(fontsize=7)

    ax = ax_x[1, 0]
    for name, samples in methods.items():
        if len(samples) == 0 or "beta" not in samples.columns:
            continue
        ax.hist(samples["beta"].values, bins=15, alpha=0.45,
                label=labels[name], color=colors[name], density=True,
                edgecolor="white", linewidth=0.5)
        ax.axvline(float(np.median(samples["beta"].values)),
                   color=colors[name], linestyle="--", linewidth=1.2, alpha=0.85)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("density")
    ax.set_title(r"(C) Posterior of $\beta$", loc="left", fontsize=11)
    ax.legend(fontsize=7)

    ax = ax_x[1, 1]
    sorted_items_x = sorted(best_distances.items(), key=lambda kv: kv[1])
    best_x = sorted_items_x[0][0]
    bars_x = ax.bar(
        [labels[m] for m, _ in sorted_items_x],
        [v for _, v in sorted_items_x],
        color=[colors[m] for m, _ in sorted_items_x],
        edgecolor="white", linewidth=1.2,
    )
    for bar, (m, _) in zip(bars_x, sorted_items_x):
        bar.set_alpha(0.95 if m == best_x else 0.7)
    for i, (name, val) in enumerate(sorted_items_x):
        marker = "  *" if name == best_x else ""
        ax.text(i, val, f"{val:.0f}{marker}",
                ha="center", va="bottom", fontsize=8,
                fontweight="bold" if name == best_x else "normal")
    ax.set_ylabel("MSE vs observed")
    ax.set_title("(D) Best distance by method", loc="left", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=10, ha="right", fontsize=8)
    ax.margins(y=0.15)

    ax = ax_x[2, 0]
    ax.axvspan(0, switch_day - 1, color="#e8eef5", alpha=0.55)
    ax.axvspan(switch_day - 1, len(observed_I), color="#fdf3e7", alpha=0.4)
    ax.plot(observed_I, color="#6c757d", linewidth=1.5, label="rule-based ABM")
    ax.plot(gabm_I, color="#1a6bb0", linewidth=1.7, label="full GABM")
    hybrid_curve_x = np.concatenate([gabm_I[:switch_day], forecasts["smc"]])
    ax.plot(hybrid_curve_x[:len(observed_I)],
            color=colors["smc"], linewidth=1.5, linestyle="--",
            label="hybrid (SMC)")
    ax.axvline(switch_day - 1, color="#333", linestyle="-.", linewidth=1.0)
    ax.set_xlabel("day")
    ax.set_ylabel("I(t)")
    ax.set_title("(E) Curve decomposition", loc="left", fontsize=11)
    ax.legend(fontsize=7)

    ax = ax_x[2, 1]
    for name, samples in methods.items():
        if len(samples) == 0 or "rho" not in samples.columns:
            continue
        ax.hist(samples["rho"].values, bins=15, alpha=0.45,
                label=labels[name], color=colors[name], density=True,
                edgecolor="white", linewidth=0.5)
        ax.axvline(float(np.median(samples["rho"].values)),
                   color=colors[name], linestyle="--", linewidth=1.2, alpha=0.85)
    ax.axvline(0, color="#888", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("density")
    ax.set_title(r"(F) Posterior of $\rho$", loc="left", fontsize=11)
    ax.legend(fontsize=7)

    fig_x.suptitle("Full calibration and forecast overview - exp8",
                   fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    save_figure(fig_x, "exp8_composite_extended", config)
    plt.close(fig_x)

    variants: dict = {}
    for name, samples in methods.items():
        summary = posterior_summary(samples)
        variants[name] = {
            "n_accepted": int(len(samples)),
            "beta_median": summary["beta"],
            "rho_median": summary["rho"],
            "best_mse": best_distances[name],
        }

    comparison = {
        "switch_day": int(switch_day),
        "forecast_coverage_smc": round(coverage, 4),
        "best_method": min(best_distances, key=best_distances.get),
        "best_mse_overall": round(min(best_distances.values()), 2),
        "gabm_peak_day": int(np.argmax(gabm_I) + 1),
        "gabm_peak_magnitude": int(gabm_I.max()),
        "abm_reference_peak_day": int(np.argmax(abm_reference_I) + 1),
        "abm_reference_peak_magnitude": int(abm_reference_I.max()),
        "calibration_target": "GABM self-consistent (pre/post-switch)",
    }

    save_experiment_json("exp8_calibration_forecast", seed, config, variants, comparison)

    logger.info(
        "exp8 done: switch={}, coverage={:.1%}, best={} (mse={:.1f})",
        switch_day, coverage, comparison["best_method"], comparison["best_mse_overall"],
    )
    return {"variants": variants, "comparison": comparison}

if __name__ == "__main__":
    run_experiment()
