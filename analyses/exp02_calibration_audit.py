
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.integrate import odeint

from analyses.helpers import (
    ANALYSES_FIGURES,
    ANALYSES_METRICS,
    detect_switch_day_threshold,
    load_config,
    load_trajectories,
    save_figure,
)
from src.calibration.abc import ABCSampler
from src.calibration.history_matching import HistoryMatcher
import matplotlib.pyplot as plt

GAMMA_BIO = 0.14
SIGMA_FIXED = 0.2
HM_DESIGN_POINTS = 500
ABC_N_PARTICLES = 200
PRE_PEAK_MARGIN = 0

DET_VAR_THRESH = 0.10
DET_MIN_INFECTED_FRAC = 0.01
DET_MIN_DAY = 8


def _seir_forward(
    initial_state: tuple[float, float, float, float],
    beta: float,
    sigma: float,
    gamma: float,
    n_days: int,
) -> np.ndarray:
    def rhs(y, _t, b, sig, gam):
        s, e, i, r = y
        return [-b * s * i, b * s * i - sig * e, sig * e - gam * i, gam * i]

    y0 = list(initial_state)
    t_grid = np.arange(n_days + 1, dtype=float)
    try:
        sol = odeint(rhs, y0, t_grid, args=(beta, sigma, gamma),
                     full_output=False, mxstep=5000)
    except Exception as exc:
        logger.warning("seir integration failed: {}", exc)
        return np.full(n_days, np.nan)
    return sol[1:n_days + 1, 2]


def _make_run_model(
    initial_state: tuple[float, float, float, float],
    n_days: int,
    sigma: float,
    gamma: float,
):
    def run(params: dict) -> pd.Series:
        beta = float(params["beta"])
        I_pred = _seir_forward(initial_state, beta, sigma, gamma, n_days)
        return pd.Series(I_pred, index=range(n_days))
    return run


def _calibrate_one_seed(
    payload: dict,
    seed_idx: int,
) -> dict:
    """Run HM + 3 ABC samplers on one (backend, seed). Returns full per-seed
    record including detector status, posterior samples, and coverage."""
    seed = payload["seeds"][seed_idx]
    comp = payload["compartments"][seed_idx]
    beta_traj = np.asarray(payload["betas"][seed_idx], dtype=float)
    I_truth = np.asarray(comp["I"], dtype=float)
    S_truth = np.asarray(comp["S"], dtype=float)
    E_truth = np.asarray(comp["E"], dtype=float)
    R_truth = np.asarray(comp["R"], dtype=float)
    n_pop = float(S_truth[0] + E_truth[0] + I_truth[0] + R_truth[0])

    detection = detect_switch_day_threshold(
        beta_traj, I_truth, int(n_pop),
        var_thresh=DET_VAR_THRESH,
        min_infected_frac=DET_MIN_INFECTED_FRAC,
        min_day=DET_MIN_DAY,
        pre_peak_margin=PRE_PEAK_MARGIN,
    )
    record: dict = {
        "seed": int(seed),
        "n_pop": int(n_pop),
        "peak_day": detection["peak_day"],
        "t_switch": detection["t_switch"],
        "gate": detection["gate"],
        "pre_peak": detection["pre_peak"],
    }
    if detection["t_switch"] is None:
        record["status"] = f"detector_failed_{detection['gate']}"
        return record

    t_s = int(detection["t_switch"])
    initial_state = (
        float(S_truth[t_s - 1]),
        float(E_truth[t_s - 1]),
        float(I_truth[t_s - 1]),
        float(R_truth[t_s - 1]),
    )
    active_idx = np.where(I_truth > 0)[0]
    last_active_day = int(active_idx[-1]) + 1
    n_days_target = last_active_day - t_s
    if n_days_target < 5:
        record["status"] = "active_window_too_short"
        return record
    target = pd.Series(I_truth[t_s:t_s + n_days_target], index=range(n_days_target))

    beta_crit = GAMMA_BIO / max(n_pop, 1.0)
    beta_lo, beta_hi = 0.3 * beta_crit, 5.0 * beta_crit
    prior_ranges = {"beta": (beta_lo, beta_hi)}
    record["beta_crit"] = float(beta_crit)
    record["prior_lo"] = float(beta_lo)
    record["prior_hi"] = float(beta_hi)
    record["n_days_target"] = int(n_days_target)

    run_model = _make_run_model(
        initial_state, n_days_target, SIGMA_FIXED, GAMMA_BIO,
    )

    hm = HistoryMatcher()
    design = hm.generate_samples(prior_ranges, HM_DESIGN_POINTS)
    distances = []
    for _, row in design.iterrows():
        sim = run_model(row.to_dict())
        distances.append(hm._mse(sim, target))
    design = design.copy()
    design["_d"] = distances
    hm_threshold = float(np.median(distances))
    surviving = design[design["_d"] < hm_threshold].drop(columns=["_d"])
    record["hm"] = {
        "n_design": HM_DESIGN_POINTS,
        "threshold": hm_threshold,
        "n_survivors": int(len(surviving)),
    }

    abc = ABCSampler()
    method_results: dict[str, dict] = {}
    eff_prior = surviving if len(surviving) >= 20 else design.drop(columns=["_d"])
    if len(surviving) < 20:
        record["hm"]["fallback"] = "too_few_survivors_using_full_prior"

    eps_rejection = float(np.percentile(distances, 25))
    rej = abc.rejection(run_model, target, eff_prior,
                        n_samples=ABC_N_PARTICLES, threshold=eps_rejection)
    method_results["rejection"] = _summarise(rej, target, run_model)

    ann = abc.annealing(run_model, target, eff_prior,
                        n_samples=ABC_N_PARTICLES, cooling_steps=3)
    method_results["annealing"] = _summarise(ann, target, run_model)

    smc = abc.smc(run_model, target, eff_prior,
                  n_particles=ABC_N_PARTICLES, n_steps=4)
    method_results["smc"] = _summarise(smc, target, run_model)

    record["status"] = "ok"
    record["calibration"] = method_results
    record["target_active_window"] = target.values.tolist()
    return record


def _summarise(posterior_result, target: pd.Series,
               run_model) -> dict:
    """Posterior-predictive interval + coverage on calibration window."""
    samples = posterior_result.posterior_samples
    if len(samples) == 0:
        return {
            "n_runs": int(posterior_result.n_model_runs),
            "runtime_s": float(posterior_result.runtime_seconds),
            "n_accepted": 0,
            "beta_median": None,
            "coverage_active_window": None,
            "predictive_lo": None,
            "predictive_hi": None,
            "predictive_median": None,
            "rmse_median_predictive": None,
        }
    n_for_pred = min(len(samples), 100)
    sub = samples.sample(n=n_for_pred, replace=False) if len(samples) > n_for_pred else samples
    sims = []
    for _, row in sub.iterrows():
        sims.append(run_model(row.to_dict()).values)
    sims = np.array(sims)
    lo = np.percentile(sims, 2.5, axis=0)
    hi = np.percentile(sims, 97.5, axis=0)
    med = np.percentile(sims, 50, axis=0)
    truth = target.values
    in_ci = (truth >= lo) & (truth <= hi)
    coverage = float(in_ci.mean())
    rmse_med = float(np.sqrt(np.mean((med - truth) ** 2)))
    return {
        "n_runs": int(posterior_result.n_model_runs),
        "runtime_s": float(posterior_result.runtime_seconds),
        "n_accepted": int(len(samples)),
        "beta_median": float(samples["beta"].median()),
        "beta_iqr_lo": float(samples["beta"].quantile(0.25)),
        "beta_iqr_hi": float(samples["beta"].quantile(0.75)),
        "coverage_active_window": coverage,
        "predictive_lo": lo.tolist(),
        "predictive_hi": hi.tolist(),
        "predictive_median": med.tolist(),
        "rmse_median_predictive": rmse_med,
    }


def _fig_per_seed_forecast(record: dict, backend_key: str, fig_dir: Path) -> None:
    if record.get("status") != "ok":
        return
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    truth = np.asarray(record["target_active_window"])
    x = np.arange(len(truth))
    methods = ["rejection", "annealing", "smc"]
    for ax, m in zip(axs, methods):
        s = record["calibration"][m]
        if s["predictive_median"] is None:
            ax.text(0.5, 0.5, f"{m}: 0 accepted", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{m}: NO POSTERIOR")
            continue
        lo = np.asarray(s["predictive_lo"])
        hi = np.asarray(s["predictive_hi"])
        med = np.asarray(s["predictive_median"])
        ax.fill_between(x, lo, hi, alpha=0.25, color="#1f77b4",
                        label="95% predictive")
        ax.plot(x, med, color="#1f77b4", linewidth=2.0, label="median")
        ax.plot(x, truth, color="black", linewidth=1.6, linestyle="--",
                marker="o", markersize=3, label="truth")
        ax.set_xlabel(f"days post-switch (t_switch={record['t_switch']})")
        ax.set_ylabel("infectious I(t)")
        ax.set_title(
            f"{m}  cov={s['coverage_active_window']*100:.0f}%  "
            f"β_med={s['beta_median']:.2e}",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"{backend_key} seed={record['seed']}: peak={record['peak_day']}, "
        f"switch={record['t_switch']} (threshold-var, "
        f"{'pre-peak' if record['pre_peak'] else 'POST-peak'})",
        fontsize=11,
    )
    plt.tight_layout()
    save_figure(
        fig, f"exp02_{backend_key}_seed{record['seed']}_forecast",
        dpi=120,
    )


def _fig_coverage_summary(records: list[dict], backend_key: str) -> None:
    methods = ["rejection", "annealing", "smc"]
    rows = []
    for r in records:
        if r.get("status") != "ok":
            continue
        for m in methods:
            cov = r["calibration"][m].get("coverage_active_window")
            if cov is not None:
                rows.append({"seed": r["seed"], "method": m,
                             "coverage": cov * 100,
                             "pre_peak": r["pre_peak"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(methods))
    box_data = [df[df["method"] == m]["coverage"].values for m in methods]
    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                    patch_artist=True, tick_labels=methods)
    for patch, c in zip(bp["boxes"], ["#1f77b4", "#2ca02c", "#d62728"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        for _, row in sub.iterrows():
            color = "black" if row["pre_peak"] else "red"
            jitter = (np.random.rand() - 0.5) * 0.25
            ax.scatter([positions[i] + jitter], [row["coverage"]],
                       color=color, s=35, alpha=0.85, zorder=3)
    ax.axhline(95, color="grey", linestyle=":", linewidth=1.0,
               label="95% target coverage")
    ax.set_ylabel("coverage on active window (%)")
    ax.set_title(
        f"{backend_key}: per-seed posterior-predictive coverage\n"
        f"(black dots: pre-peak switch; red: post-peak)",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save_figure(fig, f"exp02_{backend_key}_coverage_summary")


def run_audit(
    backend_key: str,
    cache_name: str,
    config_path: str = "config/default.yaml",
    n_seeds: int = 10,
    plot_per_seed: bool = True,
) -> dict:
    config = load_config(config_path)

    payload = load_trajectories(cache_name)
    if payload is None:
        raise FileNotFoundError(f"trajectory cache not found: {cache_name}")

    n_seeds = min(n_seeds, len(payload["seeds"]))
    records = []
    for s_idx in range(n_seeds):
        logger.info("audit: backend={} seed_idx={}/{}",
                    backend_key, s_idx + 1, n_seeds)
        rec = _calibrate_one_seed(payload, s_idx)
        records.append(rec)

    n_ok = sum(1 for r in records if r.get("status") == "ok")
    n_pre_peak = sum(1 for r in records if r.get("pre_peak"))
    gates: dict[str, int] = {}
    for r in records:
        g = r.get("gate", "unknown")
        gates[g] = gates.get(g, 0) + 1
    summary = {
        "backend":          backend_key,
        "cache_name":       cache_name,
        "n_seeds":          n_seeds,
        "n_calibrated_ok":  n_ok,
        "detector": {
            "name":    "threshold_rolling_var_npeople",
            "var_thresh": DET_VAR_THRESH,
            "min_infected_frac": DET_MIN_INFECTED_FRAC,
            "min_day": DET_MIN_DAY,
        },
        "detector_stats": {
            "gate_counts":     gates,
            "pre_peak_count":  n_pre_peak,
            "post_peak_count": n_ok - n_pre_peak,
        },
        "config": {
            "gamma_bio":     GAMMA_BIO,
            "sigma_fixed":   SIGMA_FIXED,
            "hm_design_pts": HM_DESIGN_POINTS,
            "abc_particles": ABC_N_PARTICLES,
            "pre_peak_margin": PRE_PEAK_MARGIN,
        },
    }
    methods = ["rejection", "annealing", "smc"]
    coverage_means = {m: [] for m in methods}
    coverage_pre_peak = {m: [] for m in methods}
    rmse_means = {m: [] for m in methods}
    for r in records:
        if r.get("status") != "ok":
            continue
        for m in methods:
            cov = r["calibration"][m].get("coverage_active_window")
            rmse = r["calibration"][m].get("rmse_median_predictive")
            if cov is not None:
                coverage_means[m].append(cov)
                if r["pre_peak"]:
                    coverage_pre_peak[m].append(cov)
            if rmse is not None:
                rmse_means[m].append(rmse)
    summary["coverage"] = {
        m: {
            "mean_all":      float(np.mean(coverage_means[m])) if coverage_means[m] else None,
            "mean_pre_peak": float(np.mean(coverage_pre_peak[m])) if coverage_pre_peak[m] else None,
            "rmse_median":   float(np.median(rmse_means[m])) if rmse_means[m] else None,
        }
        for m in methods
    }

    ANALYSES_METRICS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    out_path = ANALYSES_METRICS / f"exp02_audit_{backend_key}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    logger.info("audit summary saved to {}", out_path)

    if plot_per_seed:
        for r in records:
            _fig_per_seed_forecast(r, backend_key, ANALYSES_FIGURES)
    _fig_coverage_summary(records, backend_key)

    print(f"\nexp02 audit summary: {backend_key}")
    print(f"  seeds calibrated OK:       {n_ok}/{n_seeds}")
    print(f"  detector gates:            {gates}")
    print(f"  switch  pre-peak:          {n_pre_peak} / post-peak: {n_ok - n_pre_peak}")
    for m in methods:
        cv = summary["coverage"][m]
        cv_all = f"{cv['mean_all']*100:.0f}%" if cv['mean_all'] is not None else "n/a"
        cv_pp = f"{cv['mean_pre_peak']*100:.0f}%" if cv['mean_pre_peak'] is not None else "n/a"
        rm = f"{cv['rmse_median']:.0f}" if cv['rmse_median'] is not None else "n/a"
        print(f"  {m:<12}  cov_all={cv_all:<6}  cov_pre_peak={cv_pp:<6}  RMSE_med={rm}")
    return {"summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", type=str, required=True,
        help="backend slot key (matches cached trajectory in "
             "results/analyses/trajectories/exp01_<key>.pkl)",
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--no-per-seed-figs", action="store_true",
                        help="skip per-seed forecast figures")
    args = parser.parse_args()

    cache_name = f"exp01_{args.backend}"
    run_audit(
        backend_key=args.backend,
        cache_name=cache_name,
        config_path=args.config,
        n_seeds=args.n_seeds,
        plot_per_seed=not args.no_per_seed_figs,
    )


if __name__ == "__main__":
    main()
