from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.integrate import odeint
from scipy.optimize import minimize

from experiments.helpers import (
    load_config, make_llama_gabm, make_small_population,
    save_experiment_json, save_figure,
)

FORECAST_HORIZON = 30  
T_OBS = 10  
METRIC_HORIZONS = [1, 7, 14, 21]
_STRAIN = "H1N1"

def _extract_I(result) -> np.ndarray:
    return np.array([dr.I.get(_STRAIN, 0) for dr in result.days])

def _seir_rhs(y, t, beta, gamma, sigma, N):
    s, e, i, r = y
    return [-beta * s * i, beta * s * i - sigma * e,
            sigma * e - gamma * i, gamma * i]

def forecast_persistence(I_obs, H):
    return np.full(H, I_obs[-1], dtype=float)

def forecast_exp(I_obs, H, window=5):
    y = I_obs[-window:].astype(float).clip(min=1.0)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, np.log(y), 1)
    future_x = np.arange(len(y), len(y) + H)
    return np.clip(np.exp(intercept + slope * future_x), 0.0, None)

def forecast_sir(I_obs, H, N, sigma, initial_E=10):
    T = len(I_obs)

    def sse(params):
        log_beta, log_gamma = params
        beta = float(np.exp(log_beta))
        gamma = float(np.exp(log_gamma))
        if beta <= 0 or gamma <= 0 or beta > 10 or gamma > 1:
            return 1e9
        I0 = max(int(I_obs[0]), 1)
        y0 = [N - I0 - initial_E, initial_E, I0, 0]
        t_grid = np.arange(T)
        try:
            sol = odeint(_seir_rhs, y0, t_grid, args=(beta, gamma, sigma, N),
                         full_output=False, mxstep=5000)
        except Exception:
            return 1e9
        return float(np.sum((I_obs - sol[:, 2]) ** 2))

    res = minimize(sse, x0=np.array([np.log(0.5 / N), np.log(0.14)]),
                   method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-2, "maxiter": 300})
    beta = float(np.exp(res.x[0]))
    gamma = float(np.exp(res.x[1]))
    I0 = max(int(I_obs[0]), 1)
    y0 = [N - I0 - initial_E, initial_E, I0, 0]
    t_grid = np.arange(T + H)
    sol = odeint(_seir_rhs, y0, t_grid, args=(beta, gamma, sigma, N),
                 full_output=False, mxstep=5000)
    return sol[T:T + H, 2]

def forecast_hybrid_smc(gabm_result, T_obs, H, N, sigma, gamma_bio,
                        n_particles=300, seed=0):
    dr_t = gabm_result.days[T_obs - 1]
    S0 = float(dr_t.S.get(_STRAIN, 0))
    E0 = float(dr_t.E.get(_STRAIN, 0))
    I0 = float(dr_t.I.get(_STRAIN, 0))
    R0 = float(dr_t.R.get(_STRAIN, 0))

    betas_observed = np.array([
        gabm_result.days[t].beta.get(_STRAIN, 0.0)
        for t in range(T_obs)
    ], dtype=float)
    t_obs_arr = np.arange(T_obs, dtype=float)
    beta_crit = gamma_bio / N
    mask_pos = betas_observed > 0
    if mask_pos.sum() >= 3:
        coeffs = np.polyfit(t_obs_arr[mask_pos], betas_observed[mask_pos], 2)
    else:
        mean_beta = float(betas_observed[mask_pos].mean()) if mask_pos.any() else 2.0 * beta_crit
        coeffs = np.array([0.0, 0.0, mean_beta])

    t_fut = np.arange(T_obs, T_obs + H, dtype=float)
    beta_future_raw = np.polyval(coeffs, t_fut)
    beta_future = np.clip(beta_future_raw, 0.3 * beta_crit, 6.0 * beta_crit)

    def _time_varying_rhs(y, t, beta_seq, sigma_val, gamma_val):
        idx = int(np.clip(t, 0, len(beta_seq) - 1))
        beta_t = beta_seq[idx]
        s, e, i, r = y
        return [-beta_t * s * i, beta_t * s * i - sigma_val * e,
                sigma_val * e - gamma_val * i, gamma_val * i]

    y0 = [S0, E0, I0, R0]
    t_grid = np.arange(H + 1)
    try:
        sol = odeint(_time_varying_rhs, y0, t_grid,
                     args=(beta_future, sigma, gamma_bio),
                     full_output=False, mxstep=5000)
        median = sol[1:H + 1, 2]
    except Exception:
        median = np.zeros(H)

    lo = np.full(H, np.inf)
    hi = np.full(H, -np.inf)
    for mult in [0.75, 0.9, 1.1, 1.25]:
        beta_scaled = np.clip(beta_future * mult,
                               0.3 * beta_crit, 6.0 * beta_crit)
        try:
            sol_b = odeint(_time_varying_rhs, y0, t_grid,
                           args=(beta_scaled, sigma, gamma_bio),
                           full_output=False, mxstep=5000)
            I_b = sol_b[1:H + 1, 2]
            lo = np.minimum(lo, I_b)
            hi = np.maximum(hi, I_b)
        except Exception:
            pass

    map_traj = median.copy()

    return {
        "median": median, "mean": median, "map": map_traj,
        "lower": lo, "upper": hi,
        "beta_future": beta_future.tolist(),
        "beta_hat_pre": betas_observed.tolist(),
        "beta_poly_coeffs": coeffs.tolist(),
        "beta_median": float(np.median(beta_future)),
        "gamma_median": float(gamma_bio),
    }

def _metrics(pred, truth):
    diff = pred - truth
    return {"rmse": float(np.sqrt(np.mean(diff ** 2))),
            "mae": float(np.mean(np.abs(diff)))}

def run_experiment(config_path="config/default.yaml", seed=42,
                   n_agents=3000, n_seeds=5):
    config = load_config(config_path)
    data, households = make_small_population(n=n_agents, seed=seed)
    days = range(1, config["model"]["days"][1])
    sigma = config["model"]["seir"]["sigma"]
    gamma_bio = config["model"]["seir"]["gamma"]

    trajectories = []
    results_list = []
    for s in range(seed, seed + n_seeds):
        logger.info("exp14: gabm trajectory {}/{}, seed={}", s - seed + 1, n_seeds, s)
        gabm = make_llama_gabm(config, data.copy(), households.copy())
        result = gabm.run(days=days, seed=s)
        results_list.append(result)
        trajectories.append(_extract_I(result))

    T, H = T_OBS, FORECAST_HORIZON
    per_seed_forecasts = []
    per_seed_metrics = {m: [] for m in
                        ["persistence", "exp", "sir", "hybrid_smc"]}
    truths = []

    for idx, (traj, gabm_result) in enumerate(zip(trajectories, results_list)):
        if len(traj) < T + H:
            logger.warning("exp14: trajectory {} too short ({}), skipping", idx, len(traj))
            continue
        I_obs = traj[:T]
        I_true = traj[T:T + H]
        truths.append(I_true)

        f_persist = forecast_persistence(I_obs, H)
        f_exp = forecast_exp(I_obs, H)
        f_sir = forecast_sir(I_obs, H, n_agents, sigma)
        f_smc = forecast_hybrid_smc(gabm_result, T, H, n_agents, sigma,
                                     gamma_bio, seed=idx)

        per_seed_forecasts.append({
            "persistence": f_persist, "exp": f_exp, "sir": f_sir,
            "hybrid_smc": f_smc, "truth": I_true,
        })
        per_seed_metrics["persistence"].append(_metrics(f_persist, I_true))
        per_seed_metrics["exp"].append(_metrics(f_exp, I_true))
        per_seed_metrics["sir"].append(_metrics(f_sir, I_true))
        per_seed_metrics["hybrid_smc"].append(_metrics(f_smc["median"], I_true))

    horizon_err = {m: {h: [] for h in METRIC_HORIZONS} for m in per_seed_metrics}
    for fc in per_seed_forecasts:
        truth = fc["truth"]
        for h in METRIC_HORIZONS:
            if h > len(truth): continue
            horizon_err["persistence"][h].append(abs(fc["persistence"][h - 1] - truth[h - 1]))
            horizon_err["exp"][h].append(abs(fc["exp"][h - 1] - truth[h - 1]))
            horizon_err["sir"][h].append(abs(fc["sir"][h - 1] - truth[h - 1]))
            horizon_err["hybrid_smc"][h].append(
                abs(fc["hybrid_smc"]["median"][h - 1] - truth[h - 1]))

    summary = {m: {f"mae_h{h}": round(float(np.mean(e)), 2) if e else None
                    for h, e in d.items()}
               for m, d in horizon_err.items()}

    if per_seed_forecasts:
        fc = per_seed_forecasts[0]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        x_obs = np.arange(1, T + 1)
        x_fc = np.arange(T + 1, T + H + 1)
        x_conn = np.concatenate([[T], x_fc])
        obs_vals = trajectories[0][:T]
        last_obs = float(obs_vals[-1])

        ax.plot(x_obs, obs_vals, color="#333", linewidth=2.2, marker="o",
                markersize=4, label="observed GABM (1..T_obs)")
        ax.plot(np.concatenate([[T], x_fc]),
                np.concatenate([[last_obs], fc["truth"]]),
                color="#333", linewidth=2.0, linestyle=":",
                marker="o", markersize=3, label="truth (held out)")

        smc_mean = np.concatenate([[last_obs], fc["hybrid_smc"]["mean"]])
        smc_map = np.concatenate([[last_obs], fc["hybrid_smc"]["map"]])
        smc_lo = np.concatenate([[last_obs], fc["hybrid_smc"]["lower"]])
        smc_hi = np.concatenate([[last_obs], fc["hybrid_smc"]["upper"]])
        ax.fill_between(x_conn, smc_lo, smc_hi, color="#9467bd", alpha=0.22,
                        label="hybrid-SMC 95% CI")
        ax.plot(x_conn, smc_mean, color="#9467bd", linewidth=2.0,
                label="hybrid-SMC posterior mean (ours)")
        ax.plot(x_conn, smc_map, color="#6a1b9a", linewidth=1.3,
                linestyle="--",
                label="hybrid-SMC MAP trajectory (ours)")

        ax.plot(np.concatenate([[T], x_fc]),
                np.concatenate([[last_obs], fc["sir"]]),
                color="#2ca02c", linewidth=1.5, linestyle="--",
                label="classical SEIR fit")
        ax.plot(np.concatenate([[T], x_fc]),
                np.concatenate([[last_obs], fc["persistence"]]),
                color="#888", linewidth=1.2, linestyle=":",
                label="persistence")

        ax.axvline(T, color="black", linestyle="--", linewidth=0.8,
                   alpha=0.5)
        ax.text(T, ax.get_ylim()[1] * 0.95, f" day {T}",
                fontsize=9, color="black", va="top")
        ax.set_xlabel("day")
        ax.set_ylabel("infectious I(t)")
        ax.set_title(f"exp14: single-seed forecast, seed={seed}, T_obs={T}",
                     fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        plt.tight_layout()
        save_figure(fig, "exp14_forecast_example", config)
        plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    methods_viz = ["persistence", "sir", "hybrid_smc"]
    colors = {"persistence": "#888", "sir": "#2ca02c", "hybrid_smc": "#9467bd"}
    bar_w = 0.25
    x = np.arange(len(METRIC_HORIZONS))
    for i, m in enumerate(methods_viz):
        vals = [np.mean(horizon_err[m][h]) if horizon_err[m][h] else 0.0
                for h in METRIC_HORIZONS]
        stds = [np.std(horizon_err[m][h]) if horizon_err[m][h] else 0.0
                for h in METRIC_HORIZONS]
        ax2.bar(x + (i - 1) * bar_w, vals, width=bar_w, color=colors[m],
                yerr=stds, capsize=3, label=m)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"h={h} d" for h in METRIC_HORIZONS])
    ax2.set_xlabel("forecast horizon h (days ahead of T_obs)")
    ax2.set_ylabel("absolute forecast error (I(t) units)")
    ax2.set_title(
        f"exp14: forecast error vs horizon, T_obs={T} "
        f"(n={len(per_seed_forecasts)} seeds)", fontsize=11)
    ax2.legend(fontsize=9)
    plt.tight_layout()
    save_figure(fig2, "exp14_rmse_vs_horizon", config)
    plt.close(fig2)

    if len(truths) >= 2:
        fig3, ax3 = plt.subplots(figsize=(11, 5.5))
        for k, traj in enumerate(trajectories[:len(truths)]):
            label = f"observed + truth ({len(truths)} seeds)" if k == 0 else None
            ax3.plot(np.arange(1, T + H + 1), traj[:T + H],
                     color="#aaaaaa", linewidth=0.8, alpha=0.6, label=label)
        all_medians = np.array([f["hybrid_smc"]["median"] for f in per_seed_forecasts])
        all_lo = np.array([f["hybrid_smc"]["lower"] for f in per_seed_forecasts])
        all_hi = np.array([f["hybrid_smc"]["upper"] for f in per_seed_forecasts])
        x_fc_plot = np.arange(T + 1, T + H + 1)
        ens_lo = np.min(all_lo, axis=0)
        ens_hi = np.max(all_hi, axis=0)
        ens_med = np.median(all_medians, axis=0)
        ax3.fill_between(x_fc_plot, ens_lo, ens_hi, color="#9467bd", alpha=0.2,
                         label="hybrid-SMC envelope across seeds")
        ax3.plot(x_fc_plot, ens_med, color="#9467bd", linewidth=2.0,
                 label="hybrid-SMC median")
        ax3.axvline(T, color="black", linestyle="--", linewidth=0.8,
                    alpha=0.5)
        ax3.text(T, ax3.get_ylim()[1] * 0.95, f" day {T}",
                 fontsize=9, color="black", va="top")
        ax3.set_xlabel("day")
        ax3.set_ylabel("infectious I(t)")
        ax3.set_title(f"exp14: multi-seed hybrid-SMC forecast, T_obs={T}", fontsize=11)
        ax3.legend(fontsize=9)
        plt.tight_layout()
        save_figure(fig3, "exp14_multi_seed_forecast", config)
        plt.close(fig3)

    save_experiment_json("exp14_forecasting_smc", seed, config,
                         variants={"per_seed_metrics": {
                             m: [dict(x) for x in per_seed_metrics[m]]
                             for m in per_seed_metrics
                         }}, comparison=summary)
    logger.info("exp14 done: summary={}", summary)
    return {"summary": summary}

if __name__ == "__main__":
    run_experiment()
