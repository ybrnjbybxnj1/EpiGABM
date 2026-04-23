from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.integrate import odeint
from scipy.optimize import minimize

from experiments.helpers import (
    load_config, make_llama_gabm, make_small_population,
    save_experiment_json, save_figure,
)

FORECAST_HORIZON = 24
T_OBS = 10
METRIC_HORIZONS = [1, 7, 14, 21]
_STRAIN = "H1N1"

def _extract_I(result) -> np.ndarray:
    return np.array([dr.I.get(_STRAIN, 0) for dr in result.days])

def _seir_rhs(y: np.ndarray, t: float, beta: float, gamma: float,
              sigma: float, N: int) -> list[float]:
    s, e, i, r = y
    d_s = -beta * s * i
    d_e = beta * s * i - sigma * e
    d_i = sigma * e - gamma * i
    d_r = gamma * i
    return [d_s, d_e, d_i, d_r]

def forecast_persistence(I_obs: np.ndarray, H: int) -> np.ndarray:
    return np.full(H, I_obs[-1], dtype=float)

# log-linear extrap on last window days
def forecast_exp(I_obs: np.ndarray, H: int, window: int = 5) -> np.ndarray:
    y = I_obs[-window:].astype(float).clip(min=1.0)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, np.log(y), 1)
    future_x = np.arange(len(y), len(y) + H)
    forecast = np.exp(intercept + slope * future_x)
    return np.clip(forecast, 0.0, None)

# classical seir MLE fit then integrate forward
def forecast_sir(I_obs: np.ndarray, H: int, N: int, sigma: float,
                 initial_E: int = 10) -> np.ndarray:
    T = len(I_obs)

    def neg_log_lik(params: np.ndarray) -> float:
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
        I_pred = sol[:, 2]
        resid = I_obs - I_pred
        return float(np.sum(resid ** 2))

    res = minimize(
        neg_log_lik, x0=np.array([np.log(0.5 / N), np.log(0.14)]),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-2, "maxiter": 300},
    )
    beta = float(np.exp(res.x[0]))
    gamma = float(np.exp(res.x[1]))
    I0 = max(int(I_obs[0]), 1)
    y0 = [N - I0 - initial_E, initial_E, I0, 0]
    t_grid = np.arange(T + H)
    sol = odeint(_seir_rhs, y0, t_grid, args=(beta, gamma, sigma, N),
                 full_output=False, mxstep=5000)
    return sol[T:T + H, 2]

def forecast_hybrid(gabm_result, T_obs: int, H: int, N: int, sigma: float,
                    gamma_bio: float) -> dict:
    dr_t = gabm_result.days[T_obs - 1]
    S0 = float(dr_t.S.get(_STRAIN, 0))
    E0 = float(dr_t.E.get(_STRAIN, 0))
    I0 = float(dr_t.I.get(_STRAIN, 0))
    R0 = float(dr_t.R.get(_STRAIN, 0))

    # empirical beta: mean of last 5 gabm days
    beta_window_start = max(T_obs - 5, 1)
    betas = [
        gabm_result.days[d - 1].beta.get(_STRAIN, 0.0)
        for d in range(beta_window_start, T_obs + 1)
    ]
    betas_pos = [b for b in betas if b > 0]
    beta_hat = float(np.mean(betas_pos)) if betas_pos else gamma_bio * 2.0 / N

    y0 = [S0, E0, I0, R0]
    t_grid = np.arange(H + 1)
    sol = odeint(_seir_rhs, y0, t_grid, args=(beta_hat, gamma_bio, sigma, N),
                 full_output=False, mxstep=5000)
    median = sol[1:H + 1, 2]

    # uncertainty band: +/-25% on beta, take per-day envelope
    lo_trajectory = np.full(H, np.inf)
    hi_trajectory = np.full(H, -np.inf)
    for beta_mult in [0.75, 0.9, 1.1, 1.25]:
        sol_b = odeint(
            _seir_rhs, y0, t_grid,
            args=(beta_hat * beta_mult, gamma_bio, sigma, N),
            full_output=False, mxstep=5000,
        )
        I_b = sol_b[1:H + 1, 2]
        lo_trajectory = np.minimum(lo_trajectory, I_b)
        hi_trajectory = np.maximum(hi_trajectory, I_b)

    return {"median": median, "lower": lo_trajectory, "upper": hi_trajectory,
            "beta_hat": beta_hat, "initial_state": (S0, E0, I0, R0)}

def _metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    diff = pred - truth
    return {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
    }

def run_experiment(
    config_path: str = "config/default.yaml",
    seed: int = 42,
    n_agents: int = 3000,
    n_seeds: int = 5,
) -> dict:
    config = load_config(config_path)
    data, households = make_small_population(n=n_agents, seed=seed)
    days = range(1, config["model"]["days"][1])
    sigma = config["model"]["seir"]["sigma"]
    gamma_bio = config["model"]["seir"]["gamma"]

    results_list = []
    trajectories: list[np.ndarray] = []
    for s in range(seed, seed + n_seeds):
        logger.info("exp13: gabm trajectory {}/{}, seed={}", s - seed + 1, n_seeds, s)
        gabm = make_llama_gabm(config, data.copy(), households.copy())
        result = gabm.run(days=days, seed=s)
        results_list.append(result)
        trajectories.append(_extract_I(result))

    T = T_OBS
    H = FORECAST_HORIZON

    per_seed_forecasts = []
    per_seed_metrics = {
        "persistence": [], "exp": [], "sir": [], "hybrid": [],
    }
    truths = []

    for idx, (traj, gabm_result) in enumerate(zip(trajectories, results_list)):
        if len(traj) < T + H:
            logger.warning("exp13: trajectory {} too short ({}), skipping", idx, len(traj))
            continue
        I_obs = traj[:T]
        I_true = traj[T:T + H]
        truths.append(I_true)

        f_persist = forecast_persistence(I_obs, H)
        f_exp = forecast_exp(I_obs, H)
        f_sir = forecast_sir(I_obs, H, n_agents, sigma)
        f_hybrid = forecast_hybrid(gabm_result, T, H, n_agents, sigma, gamma_bio)

        per_seed_forecasts.append({
            "persistence": f_persist,
            "exp": f_exp,
            "sir": f_sir,
            "hybrid": f_hybrid,
            "truth": I_true,
        })
        per_seed_metrics["persistence"].append(_metrics(f_persist, I_true))
        per_seed_metrics["exp"].append(_metrics(f_exp, I_true))
        per_seed_metrics["sir"].append(_metrics(f_sir, I_true))
        per_seed_metrics["hybrid"].append(_metrics(f_hybrid["median"], I_true))

    horizon_rmse = {m: {h: [] for h in METRIC_HORIZONS} for m in per_seed_metrics}
    for idx, fc_set in enumerate(per_seed_forecasts):
        truth = fc_set["truth"]
        for h in METRIC_HORIZONS:
            if h > len(truth):
                continue
            horizon_rmse["persistence"][h].append(
                abs(fc_set["persistence"][h - 1] - truth[h - 1]))
            horizon_rmse["exp"][h].append(abs(fc_set["exp"][h - 1] - truth[h - 1]))
            horizon_rmse["sir"][h].append(abs(fc_set["sir"][h - 1] - truth[h - 1]))
            horizon_rmse["hybrid"][h].append(
                abs(fc_set["hybrid"]["median"][h - 1] - truth[h - 1]))

    summary = {}
    for method, h_dict in horizon_rmse.items():
        summary[method] = {}
        for h, errs in h_dict.items():
            if errs:
                summary[method][f"mae_h{h}"] = round(float(np.mean(errs)), 2)
                summary[method][f"std_h{h}"] = round(float(np.std(errs)), 2)

    if per_seed_forecasts:
        fc = per_seed_forecasts[0]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        x_obs = np.arange(1, T + 1)
        x_fc = np.arange(T + 1, T + H + 1)
        obs_vals = trajectories[0][:T]
        ax.plot(x_obs, obs_vals, color="#333", linewidth=2.2,
                marker="o", markersize=4, label="observed GABM (1..T_obs)")
        ax.plot(x_fc, fc["truth"], color="#333", linewidth=2.0, linestyle=":",
                marker="o", markersize=3, label="truth (held out)")
        ax.fill_between(x_fc, fc["hybrid"]["lower"], fc["hybrid"]["upper"],
                        color="#9467bd", alpha=0.25,
                        label="hybrid GABM->SEIR +/-25% beta band")
        ax.plot(x_fc, fc["hybrid"]["median"], color="#9467bd", linewidth=1.8,
                label="hybrid GABM->SEIR (ours)")
        ax.plot(x_fc, fc["sir"], color="#2ca02c", linewidth=1.5, linestyle="--",
                label="classical SEIR fit")
        ax.plot(x_fc, fc["persistence"], color="#888", linewidth=1.2,
                linestyle=":", label="persistence")
        ax.axvline(T + 0.5, color="black", linestyle="--", linewidth=0.8,
                   alpha=0.5)
        ax.text(T + 0.3, ax.get_ylim()[1] * 0.95, "observation cutoff",
                fontsize=8, color="black", va="top")
        # horizon markers at h=1,7,14,21 relative to T_obs
        for h in METRIC_HORIZONS:
            ax.axvline(T + h, color="#444", linestyle=":", linewidth=0.6,
                       alpha=0.4)
            ax.text(T + h, ax.get_ylim()[1] * 0.05, f"h={h}",
                    fontsize=8, ha="center", color="#444")
        ax.set_xlabel("day")
        ax.set_ylabel("infectious I(t)")
        ax.set_title(
            f"exp13: single-seed forecast, seed={seed}, T_obs={T}, "
            f"h = forecast horizon (days ahead of T_obs): {METRIC_HORIZONS}",
            fontsize=11,
        )
        ax.legend(fontsize=9, loc="upper right")
        plt.tight_layout()
        save_figure(fig, "exp13_forecast_example", config)
        plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    methods_viz = ["persistence", "sir", "hybrid"]
    colors = {"persistence": "#888", "sir": "#2ca02c", "hybrid": "#9467bd"}
    n_methods = len(methods_viz)
    bar_w = 0.25
    x = np.arange(len(METRIC_HORIZONS))
    for i, m in enumerate(methods_viz):
        vals = [np.mean(horizon_rmse[m][h]) if horizon_rmse[m][h] else 0.0
                for h in METRIC_HORIZONS]
        stds = [np.std(horizon_rmse[m][h]) if horizon_rmse[m][h] else 0.0
                for h in METRIC_HORIZONS]
        ax2.bar(x + (i - n_methods / 2 + 0.5) * bar_w, vals,
                width=bar_w, color=colors[m], yerr=stds,
                capsize=3, label=m)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"h={h} d" for h in METRIC_HORIZONS])
    ax2.set_xlabel("forecast horizon h (days ahead of T_obs)")
    ax2.set_ylabel("absolute forecast error at horizon h (I(t) units)")
    ax2.set_title(
        f"exp13: forecast error vs horizon "
        f"(mean +/- std over {len(per_seed_forecasts)} seeds; "
        f"exp extrapolation omitted from viz - it diverges)",
        fontsize=10,
    )
    ax2.legend(fontsize=9)
    plt.tight_layout()
    save_figure(fig2, "exp13_rmse_vs_horizon", config)
    plt.close(fig2)

    if len(truths) >= 2:
        fig3, ax3 = plt.subplots(figsize=(11, 5.5))
        for k, traj in enumerate(trajectories[:len(truths)]):
            label = f"observed + truth ({len(truths)} seeds)" if k == 0 else None
            ax3.plot(np.arange(1, T + H + 1), traj[:T + H],
                     color="#aaaaaa", linewidth=0.8, alpha=0.6, label=label)
        all_medians = np.array([fc["hybrid"]["median"]
                                 for fc in per_seed_forecasts])
        all_lo = np.array([fc["hybrid"]["lower"] for fc in per_seed_forecasts])
        all_hi = np.array([fc["hybrid"]["upper"] for fc in per_seed_forecasts])
        ens_median = np.median(all_medians, axis=0)
        ens_lo = np.min(all_lo, axis=0)
        ens_hi = np.max(all_hi, axis=0)
        x_fc_plot = np.arange(T + 1, T + H + 1)
        ax3.fill_between(x_fc_plot, ens_lo, ens_hi, color="#9467bd", alpha=0.2,
                         label="hybrid envelope (across seeds)")
        ax3.plot(x_fc_plot, ens_median, color="#9467bd", linewidth=2.0,
                 label="hybrid GABM->SEIR (median)")
        ax3.axvline(T + 0.5, color="black", linestyle="--", linewidth=0.8,
                    alpha=0.5)
        ax3.set_xlabel("day")
        ax3.set_ylabel("infectious I(t)")
        ax3.set_title(
            f"exp13: multi-seed forecast - hybrid GABM->SEIR on {len(truths)} trajectories",
            fontsize=11,
        )
        ax3.legend(fontsize=9)
        plt.tight_layout()
        save_figure(fig3, "exp13_multi_seed_forecast", config)
        plt.close(fig3)

    save_experiment_json(
        "exp13_forecasting_benchmark", seed, config,
        variants={"per_seed_metrics": {
            m: [dict(x) for x in per_seed_metrics[m]] for m in per_seed_metrics
        }},
        comparison=summary,
    )

    logger.info("exp13 done: summary = {}", summary)
    return {"summary": summary}

if __name__ == "__main__":
    run_experiment()
