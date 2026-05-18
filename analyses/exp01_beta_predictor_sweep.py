# exp01: koshkareva-2025 beta-predictor sweep on cached GABM trajectories (RMSE/MAE per t_obs x method)

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.integrate import odeint

from analyses.helpers import (
    extract_beta_series,
    extract_compartment_series,
    load_config,
    load_trajectories,
    run_gabm_seeds_parallel,
    run_gabm_seeds_resumable,
    save_figure,
    save_json,
    save_trajectories,
)
from src.calibration.beta_prediction import BetaPredictor
from src.regime.threshold import _cpoint_roll_var_npeople

SIMPLE_METHODS = [
    "last_value",
    "rolling_mean",
    "expanding_mean",
    "biexponential",
    "median_beta",
    "regression_day",
    "regression_day_seir_prev_i",
    "incremental_last_value",
    "incremental_rolling_mean",
]
LSTM_METHOD = "lstm_day_e_prev_i"
MLP_METHOD = "mlp_window_beta_prev_i"

T_OBS_GRID = [6, 8, 10, 12, 14]
FORECAST_HORIZON = 30
METRIC_HORIZONS = [1, 7, 14, 21]

AUTO_SWITCH_THRESH = 0.10
AUTO_SWITCH_MIN_INFECTED_FRAC = 0.01

def _trajectories_payload(results: list, config: dict) -> dict:
    payload = {
        "config_strain": next(
            (k for k, v in config.get("model", {}).get("infected_init", {}).items() if v > 0),
            "H1N1",
        ),
        "n_agents": config["model"]["population_size"],
        "seeds": [r.seed for r in results],
        "betas": [extract_beta_series(r) for r in results],
        "compartments": [extract_compartment_series(r) for r in results],
    }
    return payload

def _seir_with_time_varying_beta(
    initial_state: tuple[float, float, float, float],
    beta_seq: np.ndarray,
    sigma: float,
    gamma: float,
    horizon: int,
) -> np.ndarray:
    def rhs(y, t, betas, sig, gam):
        idx = int(np.clip(t, 0, len(betas) - 1))
        b = betas[idx]
        s, e, i, r = y
        return [-b * s * i, b * s * i - sig * e, sig * e - gam * i, gam * i]

    y0 = list(initial_state)
    t_grid = np.arange(horizon + 1, dtype=float)
    try:
        sol = odeint(rhs, y0, t_grid, args=(beta_seq, sigma, gamma),
                     full_output=False, mxstep=5000)
    except Exception as exc:
        logger.warning("seir integration failed: {}", exc)
        return np.full(horizon, np.nan)
    return sol[1:horizon + 1, 2]

def _auto_switch_day(
    beta_traj: np.ndarray,
    i_traj: np.ndarray,
    n_pop: int,
    fallback: int = 10,
) -> int:
    n_min = max(int(n_pop * AUTO_SWITCH_MIN_INFECTED_FRAC), 5)
    if len(beta_traj) < 16:
        return fallback
    beta_s = pd.Series(beta_traj)
    inf_s = pd.Series(i_traj)
    cpoint = _cpoint_roll_var_npeople(
        beta_s, inf_s, thresh=AUTO_SWITCH_THRESH, n_people=n_min,
    )
    if cpoint >= len(beta_traj) - 1:
        return fallback
    return int(np.clip(cpoint, 6, len(beta_traj) - FORECAST_HORIZON - 1))

def _predict_beta(
    method: str,
    predictor: BetaPredictor,
    beta_obs: np.ndarray,
    prev_i: np.ndarray,
    horizon: int,
) -> np.ndarray:
    series = pd.Series(beta_obs, index=range(len(beta_obs)))
    prev_i_series = pd.Series(prev_i, index=range(len(prev_i)))
    kwargs: dict = {}
    if method in ("regression_day_seir_prev_i", "lstm_day_e_prev_i"):
        kwargs["prev_i"] = prev_i_series
    pred = predictor.predict(method, series, n_ahead=horizon, **kwargs)
    return np.asarray(pred.values, dtype=float)


def _evaluate_method(
    method: str,
    predictor: BetaPredictor,
    beta_traj: np.ndarray,
    comp: dict[str, np.ndarray],
    t_obs: int,
    horizon: int,
    sigma: float,
    gamma: float,
    gamma_scale: float = 1.0,
    skip_day0_sentinel: bool = False,
    adaptive_clip: bool = False,
) -> dict:
    if skip_day0_sentinel:
        beta_obs = beta_traj[1:t_obs]
        prev_i = comp["I"][1:t_obs]
    else:
        beta_obs = beta_traj[:t_obs]
        prev_i = comp["I"][:t_obs]
    try:
        beta_future = _predict_beta(method, predictor, beta_obs, prev_i, horizon)
    except Exception as exc:
        logger.warning("method {} failed at t_obs={}: {}", method, t_obs, exc)
        beta_future = np.full(horizon, np.nan)

    n_pop = comp["S"][0] + comp["E"][0] + comp["I"][0] + comp["R"][0]
    n_pop = max(float(n_pop), 1.0)
    beta_crit = gamma / n_pop

    beta_pred_raw = np.nan_to_num(beta_future, nan=0.0)
    if adaptive_clip:
        beta_observed_window = beta_traj[max(1, t_obs - 7):t_obs]
        beta_max_obs = (
            float(np.nanmax(beta_observed_window))
            if len(beta_observed_window) > 0 else 0.0
        )
        ceiling = max(beta_max_obs * 1.5, 6.0 * beta_crit)
    else:
        beta_max_obs = float("nan")
        ceiling = 6.0 * beta_crit

    n_total = len(beta_pred_raw)
    n_clipped_low = int((beta_pred_raw < 0).sum()) if n_total > 0 else 0
    n_clipped_high = int((beta_pred_raw > ceiling).sum()) if n_total > 0 else 0
    clip_stats = {
        "fraction_clipped_low": n_clipped_low / n_total if n_total > 0 else 0.0,
        "fraction_clipped_high": n_clipped_high / n_total if n_total > 0 else 0.0,
        "ceiling_used": float(ceiling),
        "beta_max_observed": float(beta_max_obs),
        "beta_pred_pre_clip_mean": float(np.mean(beta_pred_raw)) if n_total > 0 else 0.0,
        "beta_pred_pre_clip_max": float(np.max(beta_pred_raw)) if n_total > 0 else 0.0,
    }

    beta_future_clipped = np.clip(beta_pred_raw, 0.0, ceiling)

    initial_state = (
        float(comp["S"][t_obs - 1]),
        float(comp["E"][t_obs - 1]),
        float(comp["I"][t_obs - 1]),
        float(comp["R"][t_obs - 1]),
    )
    gamma_eff = gamma * gamma_scale
    i_pred = _seir_with_time_varying_beta(
        initial_state, beta_future_clipped, sigma, gamma_eff, horizon,
    )
    return {
        "beta_future": beta_future_clipped,
        "i_pred": i_pred,
        "clip_stats": clip_stats,
    }


def _score(i_pred: np.ndarray, i_true: np.ndarray) -> dict:
    h_max = min(len(i_pred), len(i_true))
    if h_max == 0:
        return {"rmse_overall": None, "mae_overall": None,
                **{f"rmse_h{h}": None for h in METRIC_HORIZONS},
                **{f"mae_h{h}": None for h in METRIC_HORIZONS}}
    pred = np.asarray(i_pred[:h_max], dtype=float)
    true = np.asarray(i_true[:h_max], dtype=float)
    diff = pred - true
    out: dict = {
        "rmse_overall": float(np.sqrt(np.nanmean(diff ** 2))),
        "mae_overall": float(np.nanmean(np.abs(diff))),
    }
    for h in METRIC_HORIZONS:
        if h <= h_max:
            window = diff[:h]
            if not np.all(np.isnan(window)):
                out[f"rmse_h{h}"] = float(np.sqrt(np.nanmean(window ** 2)))
                out[f"mae_h{h}"] = float(np.nanmean(np.abs(window)))
            else:
                out[f"rmse_h{h}"] = None
                out[f"mae_h{h}"] = None
        else:
            out[f"rmse_h{h}"] = None
            out[f"mae_h{h}"] = None
    return out

def _plot_beta_diagnostic(
    beta_traj: np.ndarray, t_obs: int, horizon: int, n_pop: float,
    method_to_pred: dict[str, np.ndarray],
    save_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    n_total = len(beta_traj)
    x_obs = np.arange(1, t_obs + 1)
    x_fut = np.arange(t_obs + 1, t_obs + horizon + 1)
    x_truth = np.arange(1, n_total + 1)
    truth_scaled = beta_traj * n_pop
    ax.plot(x_truth, truth_scaled, color="#222", linewidth=2.2, marker="o",
            markersize=3, label="true GABM beta(t) * N")
    ax.axvline(t_obs + 0.5, color="black", linestyle="--", linewidth=0.8,
               alpha=0.6)
    ax.text(t_obs + 0.7, ax.get_ylim()[1] * 0.92,
            f"observation cutoff t_obs={t_obs}", fontsize=9)
    cmap = plt.cm.tab10
    for k, (m, beta_pred) in enumerate(method_to_pred.items()):
        ax.plot(x_fut, beta_pred * n_pop, linewidth=1.4, alpha=0.85,
                color=cmap(k % 10), label=m)
    ax.plot(x_obs, beta_traj[:t_obs] * n_pop, color="#222", linewidth=2.6,
            marker="o", markersize=4)
    ax.set_xlabel("day")
    ax.set_ylabel(r"$\beta(t) \cdot N$  (per-capita transmission rate)")
    ax.set_title(
        f"beta(t) trajectory and predictions (one seed, t_obs={t_obs})",
        fontsize=11,
    )
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.close(fig)


def _plot_i_forecast_per_method(
    beta_traj: np.ndarray, comp: dict[str, np.ndarray], t_obs: int,
    horizon: int, method_to_ipred: dict[str, np.ndarray],
    save_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    n_total = min(len(comp["I"]), t_obs + horizon)
    x_truth = np.arange(1, n_total + 1)
    ax.plot(x_truth, comp["I"][:n_total], color="#222", linewidth=2.0,
            marker="o", markersize=3, label="true GABM I(t)")
    ax.axvline(t_obs + 0.5, color="black", linestyle="--", linewidth=0.8,
               alpha=0.6)
    cmap = plt.cm.tab10
    x_fut = np.arange(t_obs + 1, t_obs + horizon + 1)
    last_obs = float(comp["I"][t_obs - 1])
    for k, (m, i_pred) in enumerate(method_to_ipred.items()):
        x_conn = np.concatenate([[t_obs], x_fut])
        y_conn = np.concatenate([[last_obs], i_pred])
        ax.plot(x_conn, y_conn, linewidth=1.5, alpha=0.85,
                color=cmap(k % 10), label=m)
    ax.set_xlabel("day")
    ax.set_ylabel("infectious I(t)")
    ax.set_title(
        f"per-method I(t) forecast vs ground truth (one seed, t_obs={t_obs})",
        fontsize=11,
    )
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.close(fig)


def _plot_metric_heatmap(
    summary: pd.DataFrame, metric_label: str, save_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.35 * len(summary)) + 0.5))
    data = summary.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(summary.shape[1]))
    ax.set_xticklabels(summary.columns)
    ax.set_yticks(np.arange(summary.shape[0]))
    ax.set_yticklabels(summary.index)
    ax.set_xlabel("t_obs")
    ax.set_ylabel("method")
    ax.set_title(f"mean {metric_label} across seeds", fontsize=11)
    for i in range(summary.shape[0]):
        for j in range(summary.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v > np.nanmedian(data) else "black",
                        fontsize=8)
    plt.colorbar(im, ax=ax, label=metric_label)
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.close(fig)


def _plot_metric_vs_horizon(
    per_method: dict[str, dict[int, list[float]]],
    t_obs: int, metric_label: str, save_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.tab10
    bar_w = 0.8 / max(len(per_method), 1)
    x = np.arange(len(METRIC_HORIZONS))
    for k, (m, h_dict) in enumerate(per_method.items()):
        means = [np.nanmean(h_dict[h]) if h_dict[h] else np.nan
                 for h in METRIC_HORIZONS]
        stds = [np.nanstd(h_dict[h]) if h_dict[h] else 0.0
                for h in METRIC_HORIZONS]
        ax.bar(x + (k - len(per_method) / 2 + 0.5) * bar_w, means,
               width=bar_w, color=cmap(k % 10), yerr=stds, capsize=2,
               label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}d" for h in METRIC_HORIZONS])
    ax.set_xlabel("forecast horizon (days ahead of t_obs)")
    ax.set_ylabel(metric_label)
    ax.set_title(
        f"{metric_label} vs horizon, all methods, t_obs={t_obs}", fontsize=11,
    )
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.close(fig)


def _plot_best_method_bar(
    summary_h7: pd.DataFrame, metric_label: str, save_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    t_obs_list = list(summary_h7.columns)
    best_methods = []
    best_values = []
    for t in t_obs_list:
        col = summary_h7[t]
        best = col.idxmin() if col.notna().any() else "n/a"
        best_methods.append(best)
        best_values.append(float(col.min()) if col.notna().any() else np.nan)
    x = np.arange(len(t_obs_list))
    bars = ax.bar(x, best_values, color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t_obs={t}" for t in t_obs_list])
    ax.set_ylabel(f"{metric_label} (best method)")
    ax.set_title(
        f"best koshkareva method per t_obs ({metric_label}, lower is better)",
        fontsize=11,
    )
    valid = [v for v in best_values if np.isfinite(v)]
    headroom = 0.02 * max(valid) if valid else 1
    for bar, m, v in zip(bars, best_methods, best_values):
        if not np.isfinite(v):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + headroom,
                m, ha="center", va="bottom", fontsize=9, rotation=0)
    plt.tight_layout()
    save_figure(fig, save_name)
    plt.close(fig)

def run_experiment(
    config_path: str = "config/default.yaml",
    seed: int = 42,
    n_agents: int = 3000,
    n_seeds: int = 10,
    use_cache: bool = True,
    with_lstm: bool = False,
    with_mlp: bool = False,
    cache_name: str = "exp01_gabm_trajectories",
    backend_key: str = "llama",
    n_workers: int = 1,
    auto_switch: bool = False,
    gamma_scale: float = 1.0,
    fig_suffix: str | None = None,
    clipfix: bool = False,
    out_subdir: str | None = None,
    t_obs_grid_override: list[int] | None = None,
) -> dict:
    if out_subdir is not None:
        import analyses.helpers as _dh
        new_root = _dh.ANALYSES_ROOT / out_subdir
        _dh.ANALYSES_METRICS = new_root / "metrics"
        _dh.ANALYSES_FIGURES = new_root / "figures"
        _dh.ANALYSES_METRICS.mkdir(parents=True, exist_ok=True)
        _dh.ANALYSES_FIGURES.mkdir(parents=True, exist_ok=True)
        logger.info(
            "exp01: out_subdir='{}' active. metrics->{} figures->{}",
            out_subdir, _dh.ANALYSES_METRICS, _dh.ANALYSES_FIGURES,
        )

    config = load_config(config_path)
    sigma = float(config["model"]["seir"]["sigma"])
    gamma = float(config["model"]["seir"]["gamma"])
    if fig_suffix is None:
        parts = [backend_key]
        if auto_switch:
            parts.append("auto")
        if gamma_scale != 1.0:
            parts.append(f"gs{str(gamma_scale).replace('.', 'p')}")
        if clipfix:
            parts.append("clipfix")
        fig_suffix = "_".join(parts)

    if not use_cache:
        for ext in ("pkl", "json"):
            old = Path("results/analyses/trajectories") / f"{cache_name}.{ext}"
            if old.exists():
                logger.info("exp01: --no-cache, removing {}", old)
                old.unlink()

    payload = load_trajectories(cache_name)
    if payload is None or len(payload.get("seeds", [])) < n_seeds:
        if payload is not None:
            logger.info(
                "exp01: partial cache for {} has {}/{} seeds, resuming",
                cache_name, len(payload.get("seeds", [])), n_seeds,
            )
        else:
            logger.info(
                "exp01: no cached trajectories, running gabm fresh "
                "(backend={})", backend_key,
            )
        run_gabm_seeds_resumable(
            config=config,
            n_agents=n_agents,
            base_seed=seed,
            n_seeds=n_seeds,
            backend_key=backend_key,
            cache_name=cache_name,
        )
        payload = load_trajectories(cache_name)
    else:
        logger.info("exp01: using cached trajectories ({} seeds)",
                    len(payload["seeds"]))

    methods = list(SIMPLE_METHODS)
    if with_lstm:
        lstm_path = Path("trained_models/lstm_day_E_prev_I_for_seir.keras")
        if not lstm_path.exists():
            logger.warning(
                "lstm model not at {}. run train_lstm_beta.py first; "
                "skipping lstm method.", lstm_path,
            )
        else:
            methods.append(LSTM_METHOD)
    if with_mlp:
        mlp_path = Path("trained_models/mlp_window_beta_prev_i.joblib")
        if not mlp_path.exists():
            logger.warning(
                "mlp model not at {}. run train_mlp_beta.py first; "
                "skipping mlp method.", mlp_path,
            )
        else:
            methods.append(MLP_METHOD)

    predictor = BetaPredictor(trained_models_dir="trained_models")
    H = FORECAST_HORIZON

    auto_per_seed: list[int] = []
    if auto_switch:
        for s_idx, _ in enumerate(payload["seeds"][:n_seeds]):
            beta_traj = np.asarray(payload["betas"][s_idx], dtype=float)
            i_traj = np.asarray(payload["compartments"][s_idx]["I"], dtype=float)
            n_pop = (payload["compartments"][s_idx]["S"][0]
                     + payload["compartments"][s_idx]["E"][0]
                     + payload["compartments"][s_idx]["I"][0]
                     + payload["compartments"][s_idx]["R"][0])
            t_obs_seed = _auto_switch_day(beta_traj, i_traj, int(n_pop))
            auto_per_seed.append(t_obs_seed)
        logger.info("auto-switch t_obs per seed: {}", auto_per_seed)
        t_obs_columns = ["auto"]
    else:
        t_obs_columns = t_obs_grid_override if t_obs_grid_override else T_OBS_GRID

    per_method_per_col: dict = {
        col: {m: {"rmse": {h: [] for h in METRIC_HORIZONS},
                  "mae":  {h: [] for h in METRIC_HORIZONS}} for m in methods}
        for col in t_obs_columns
    }
    overall: dict = {
        col: {m: {"rmse_overall": [], "mae_overall": []} for m in methods}
        for col in t_obs_columns
    }
    clip_stats_per_col: dict = {
        col: {m: [] for m in methods} for col in t_obs_columns
    }
    diagnostic_payload: dict = {}

    for col in t_obs_columns:
        per_seed_method_to_pred = {}
        per_seed_method_to_ipred = {}
        for s_idx, seed_used in enumerate(payload["seeds"][:n_seeds]):
            t_obs = auto_per_seed[s_idx] if col == "auto" else col
            beta_traj = np.asarray(payload["betas"][s_idx], dtype=float)
            comp = {k: np.asarray(v, dtype=float)
                    for k, v in payload["compartments"][s_idx].items()}
            if len(beta_traj) < t_obs + H:
                logger.warning(
                    "seed {} too short ({} days) for t_obs={}, H={} - skip",
                    seed_used, len(beta_traj), t_obs, H,
                )
                continue
            i_true = comp["I"][t_obs:t_obs + H]
            for m in methods:
                ev = _evaluate_method(
                    m, predictor, beta_traj, comp, t_obs, H, sigma, gamma,
                    gamma_scale=gamma_scale,
                    skip_day0_sentinel=clipfix,
                    adaptive_clip=clipfix,
                )
                scores = _score(ev["i_pred"], i_true)
                for h in METRIC_HORIZONS:
                    if scores[f"rmse_h{h}"] is not None:
                        per_method_per_col[col][m]["rmse"][h].append(scores[f"rmse_h{h}"])
                    if scores[f"mae_h{h}"] is not None:
                        per_method_per_col[col][m]["mae"][h].append(scores[f"mae_h{h}"])
                if scores["rmse_overall"] is not None:
                    overall[col][m]["rmse_overall"].append(scores["rmse_overall"])
                if scores["mae_overall"] is not None:
                    overall[col][m]["mae_overall"].append(scores["mae_overall"])
                clip_stats_per_col[col][m].append({
                    "seed": int(seed_used),
                    "t_obs": int(t_obs),
                    **ev["clip_stats"],
                })
                if s_idx == 0:
                    per_seed_method_to_pred[m] = ev["beta_future"]
                    per_seed_method_to_ipred[m] = ev["i_pred"]

        if per_seed_method_to_pred:
            beta_traj_0 = np.asarray(payload["betas"][0], dtype=float)
            comp_0 = {k: np.asarray(v, dtype=float)
                      for k, v in payload["compartments"][0].items()}
            t_obs_for_plot = auto_per_seed[0] if col == "auto" else col
            n_pop_0 = float(comp_0["S"][0] + comp_0["E"][0]
                            + comp_0["I"][0] + comp_0["R"][0])
            col_label = f"auto{auto_per_seed[0]}" if col == "auto" else col
            _plot_beta_diagnostic(
                beta_traj_0, t_obs_for_plot, H, n_pop_0, per_seed_method_to_pred,
                save_name=f"exp01_{fig_suffix}_beta_predictions_t_obs{col_label}",
            )
            _plot_i_forecast_per_method(
                beta_traj_0, comp_0, t_obs_for_plot, H, per_seed_method_to_ipred,
                save_name=f"exp01_{fig_suffix}_i_forecast_t_obs{col_label}",
            )
            _plot_metric_vs_horizon(
                {m: per_method_per_col[col][m]["rmse"] for m in methods},
                t_obs_for_plot, "RMSE",
                save_name=f"exp01_{fig_suffix}_rmse_vs_horizon_t_obs{col_label}",
            )
            _plot_metric_vs_horizon(
                {m: per_method_per_col[col][m]["mae"] for m in methods},
                t_obs_for_plot, "MAE",
                save_name=f"exp01_{fig_suffix}_mae_vs_horizon_t_obs{col_label}",
            )
            diagnostic_payload[f"t_obs_{col_label}"] = {
                "beta_predictions_seed0": {
                    m: v.tolist() for m, v in per_seed_method_to_pred.items()
                },
                "i_predictions_seed0": {
                    m: v.tolist() for m, v in per_seed_method_to_ipred.items()
                },
            }

    summary_dfs: dict[tuple[str, int], pd.DataFrame] = {}
    for metric in ("rmse", "mae"):
        for h in METRIC_HORIZONS:
            rows = []
            for m in methods:
                row = []
                for col in t_obs_columns:
                    vals = per_method_per_col[col][m][metric][h]
                    row.append(float(np.nanmean(vals)) if vals else np.nan)
                rows.append(row)
            df = pd.DataFrame(
                rows, index=methods,
                columns=[f"t_obs={c}" for c in t_obs_columns],
            )
            summary_dfs[(metric, h)] = df
            _plot_metric_heatmap(
                df, f"{metric.upper()}@h={h}d",
                save_name=f"exp01_{fig_suffix}_{metric}_heatmap_h{h}",
            )

    _plot_best_method_bar(
        summary_dfs[("rmse", 7)], "RMSE@h=7",
        save_name=f"exp01_{fig_suffix}_best_method_rmse_h7",
    )
    _plot_best_method_bar(
        summary_dfs[("mae", 7)], "MAE@h=7",
        save_name=f"exp01_{fig_suffix}_best_method_mae_h7",
    )

    clip_stats_summary: dict = {}
    for col in t_obs_columns:
        col_key = f"t_obs={col}"
        clip_stats_summary[col_key] = {}
        for m in methods:
            entries = clip_stats_per_col[col][m]
            if not entries:
                clip_stats_summary[col_key][m] = None
                continue
            clip_stats_summary[col_key][m] = {
                "fraction_clipped_low_mean":
                    float(np.mean([e["fraction_clipped_low"] for e in entries])),
                "fraction_clipped_high_mean":
                    float(np.mean([e["fraction_clipped_high"] for e in entries])),
                "ceiling_used_mean":
                    float(np.mean([e["ceiling_used"] for e in entries])),
                "beta_max_observed_mean":
                    float(np.nanmean([e["beta_max_observed"] for e in entries])),
                "beta_pred_pre_clip_mean":
                    float(np.mean([e["beta_pred_pre_clip_mean"] for e in entries])),
                "beta_pred_pre_clip_max":
                    float(np.max([e["beta_pred_pre_clip_max"] for e in entries])),
                "n_seeds": len(entries),
            }

    payload_to_save = {
        "params": {
            "backend_key": backend_key,
            "fig_suffix": fig_suffix,
            "n_agents": n_agents,
            "n_seeds": n_seeds,
            "t_obs_columns": list(t_obs_columns),
            "auto_per_seed": auto_per_seed if auto_switch else None,
            "auto_switch": auto_switch,
            "gamma_scale": gamma_scale,
            "horizon": H,
            "metric_horizons": METRIC_HORIZONS,
            "methods": methods,
            "with_lstm": with_lstm,
            "clipfix": clipfix,
            "out_subdir": out_subdir,
        },
        "clip_stats_summary": clip_stats_summary,
        "clip_stats_per_seed": {
            f"t_obs={col}": {m: clip_stats_per_col[col][m] for m in methods}
            for col in t_obs_columns
        },
        "summary_rmse_per_horizon": {
            f"h={h}": summary_dfs[("rmse", h)].round(2).to_dict(orient="index")
            for h in METRIC_HORIZONS
        },
        "summary_mae_per_horizon": {
            f"h={h}": summary_dfs[("mae", h)].round(2).to_dict(orient="index")
            for h in METRIC_HORIZONS
        },
        "per_seed_raw": {
            f"t_obs={col}": {
                m: {
                    "rmse": per_method_per_col[col][m]["rmse"],
                    "mae":  per_method_per_col[col][m]["mae"],
                }
                for m in methods
            }
            for col in t_obs_columns
        },
        "overall": {
            f"t_obs={col}": {
                m: {
                    "rmse_overall_mean":
                        float(np.nanmean(overall[col][m]["rmse_overall"]))
                        if overall[col][m]["rmse_overall"] else None,
                    "mae_overall_mean":
                        float(np.nanmean(overall[col][m]["mae_overall"]))
                        if overall[col][m]["mae_overall"] else None,
                }
                for m in methods
            }
            for col in t_obs_columns
        },
        "diagnostics_seed0": diagnostic_payload,
    }
    save_json(
        f"exp01_beta_predictor_sweep_{fig_suffix}",
        seed, config, payload_to_save,
    )
    for h in METRIC_HORIZONS:
        logger.info("RMSE@h={}d (mean across seeds)\n{}",
                    h, summary_dfs[("rmse", h)].round(1).to_string())
    for h in METRIC_HORIZONS:
        logger.info("MAE@h={}d  (mean across seeds)\n{}",
                    h, summary_dfs[("mae", h)].round(1).to_string())
    return payload_to_save


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=3000)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument(
        "--backend", type=str, default="llama",
        help="config.agents.backends.<key>: llama, gemma, qwen, "
             "or_llama70b, or_gemma3_27b, or_gpt_oss_20b, or_gpt_oss_120b, "
             "or_glm45_air, or_nemotron9b, or_hermes405b, mock",
    )
    parser.add_argument(
        "--n-workers", type=int, default=1,
        help="parallel gabm seeds (ollama: 1-2, openrouter: 4-8 - "
             "rate-limited free tier may 429)",
    )
    parser.add_argument(
        "--cache-name", type=str, default="exp01_gabm_trajectories",
        help="pickle filename inside results/analyses/trajectories/. use "
             "different names per backend so caches don't clobber",
    )
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore cached trajectories and rerun gabm")
    parser.add_argument("--with-lstm", action="store_true",
                        help="include lstm_day_e_prev_i (requires trained model)")
    parser.add_argument("--with-mlp", action="store_true",
                        help="include mlp_window_beta_prev_i (sklearn MLP "
                             "trained by train_mlp_beta.py; TF-free, "
                             "works on py3.14)")
    parser.add_argument(
        "--auto-switch", action="store_true",
        help="koshkareva-style variance switch detector picks one t_obs per "
             "seed instead of the manual T_OBS_GRID sweep",
    )
    parser.add_argument(
        "--gamma-scale", type=float, default=1.0,
        help="multiply post-switch SEIR gamma by this factor to compensate "
             "for behavioural acceleration (NIR4 audit point #4). 1.0 = "
             "biological gamma, >1.0 brings the SEIR peak earlier.",
    )
    parser.add_argument(
        "--fig-suffix", type=str, default=None,
        help="suffix for figure filenames; defaults to --backend value to "
             "keep per-backend figures separate.",
    )
    parser.add_argument(
        "--clipfix", action="store_true",
        help="apply day-0 sentinel skip + adaptive clip ceiling fix. "
             "use to re-run beta-prediction sweep without recomputing GABM.",
    )
    parser.add_argument(
        "--out-subdir", type=str, default=None,
        help="redirect outputs to results/analyses/<out_subdir>/{metrics,figures}/. "
             "use with --clipfix to keep new sweep results separate from "
             "the original koshkareva-2025 reproduction artifacts.",
    )
    parser.add_argument(
        "--t-obs-grid", type=int, nargs="+", default=None,
        help="override the default T_OBS_GRID [6,8,10,12,14] with custom "
             "values (e.g. --t-obs-grid 7 for smoke tests). ignored when "
             "--auto-switch is on.",
    )
    args = parser.parse_args()
    run_experiment(
        config_path=args.config,
        seed=args.seed,
        n_agents=args.n_agents,
        n_seeds=args.n_seeds,
        use_cache=not args.no_cache,
        with_lstm=args.with_lstm,
        with_mlp=args.with_mlp,
        cache_name=args.cache_name,
        backend_key=args.backend,
        n_workers=args.n_workers,
        auto_switch=args.auto_switch,
        gamma_scale=args.gamma_scale,
        fig_suffix=args.fig_suffix,
        clipfix=args.clipfix,
        out_subdir=args.out_subdir,
        t_obs_grid_override=args.t_obs_grid,
    )


if __name__ == "__main__":
    main()
