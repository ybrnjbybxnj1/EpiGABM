from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from experiments.helpers import (
    compute_metrics,
    load_config,
    make_llama_gabm,
    make_small_population,
    save_experiment_json,
    save_figure,
)

_TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
_COLORS = {
    0.0: "#1f77b4",
    0.3: "#2ca02c",
    0.7: "#ff7f0e",
    1.0: "#d62728",
}

def run_experiment(
    config_path: str = "config/default.yaml",
    seed: int = 42,
    n_agents: int = 3000,
    n_runs_per_temp: int = 5,
    temperatures: list[float] | None = None,
) -> dict:
    temperatures = temperatures or _TEMPERATURES
    config = load_config(config_path)
    data, households = make_small_population(n=n_agents, seed=seed)
    days = range(1, config["model"]["days"][1])
    strain = "H1N1"

    variants: dict = {}
    curves_by_temp: dict[float, list[list[int]]] = {}

    for temp in temperatures:
        variant_config = deepcopy(config)
        variant_config["agents"]["backends"]["llama"]["temperature"] = float(temp)

        peaks = []
        peak_days = []
        totals = []
        curves_by_temp[temp] = []

        for run_idx in range(n_runs_per_temp):
            logger.info(
                "exp10: T={}, run {}/{}",
                temp, run_idx + 1, n_runs_per_temp,
            )
            gabm = make_llama_gabm(
                variant_config, data.copy(), households.copy(),
            )
            result = gabm.run(days=days, seed=seed + run_idx)

            curve = [dr.prevalence.get(strain, 0) for dr in result.days]
            curves_by_temp[temp].append(curve)

            m = compute_metrics(result, strain)
            peaks.append(m["peak_magnitude"])
            peak_days.append(m["peak_day"])
            totals.append(m["total_infected"])

        peaks_arr = np.array(peaks)
        peak_days_arr = np.array(peak_days)
        totals_arr = np.array(totals)

        def _cv(arr: np.ndarray) -> float:
            mu = float(arr.mean())
            return float(arr.std() / mu) if mu != 0 else 0.0

        variants[f"T_{temp}"] = {
            "temperature": float(temp),
            "n_runs": n_runs_per_temp,
            "peak_magnitude_mean": float(peaks_arr.mean()),
            "peak_magnitude_std": float(peaks_arr.std()),
            "peak_magnitude_cv": round(_cv(peaks_arr), 4),
            "peak_day_mean": float(peak_days_arr.mean()),
            "peak_day_std": float(peak_days_arr.std()),
            "peak_day_cv": round(_cv(peak_days_arr), 4),
            "total_infected_mean": float(totals_arr.mean()),
            "total_infected_std": float(totals_arr.std()),
            "total_infected_cv": round(_cv(totals_arr), 4),
        }

    fig, ax = plt.subplots(figsize=(12, 6))
    for temp in temperatures:
        curves = np.array([c[: min(len(c) for c in curves_by_temp[temp])]
                           for c in curves_by_temp[temp]])
        min_len = curves.shape[1]
        mean_c = curves.mean(axis=0)
        std_c = curves.std(axis=0)
        x = np.arange(min_len)
        ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                        color=_COLORS[temp], alpha=0.15)
        ax.plot(x, mean_c, color=_COLORS[temp], linewidth=2,
                label=f"T = {temp}")
    ax.set_xlabel("simulation day")
    ax.set_ylabel("prevalence")
    ax.set_title("GABM prevalence under varying LLM temperature (mean +/- 1 std)")
    ax.legend(fontsize=10, title="temperature", title_fontsize=10)
    plt.tight_layout()
    save_figure(fig, "exp10_curves_by_temp", config)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    peak_cvs = [variants[f"T_{t}"]["peak_magnitude_cv"] for t in temperatures]
    day_cvs = [variants[f"T_{t}"]["peak_day_cv"] for t in temperatures]
    total_cvs = [variants[f"T_{t}"]["total_infected_cv"] for t in temperatures]
    ax2.plot(temperatures, peak_cvs, "o-", label="peak magnitude", linewidth=2)
    ax2.plot(temperatures, day_cvs, "s-", label="peak day", linewidth=2)
    ax2.plot(temperatures, total_cvs, "d-", label="total infected", linewidth=2)
    ax2.set_xlabel("LLM temperature")
    ax2.set_ylabel("coefficient of variation")
    ax2.set_title("result variability scales with temperature")
    ax2.legend(fontsize=10)
    plt.tight_layout()
    save_figure(fig2, "exp10_cv_vs_temp", config)
    plt.close(fig2)

    comparison = {
        "temperatures_tested": temperatures,
        "n_runs_per_temp": n_runs_per_temp,
        "peak_cv_range": [min(peak_cvs), max(peak_cvs)],
        "interpretation": (
            "If CV grows with T, bounded-stochasticity claims at T=0.3 "
            "are partly by construction. If CV is flat, behavior dominates "
            "over sampling noise."
        ),
    }

    save_experiment_json("exp10_temperature_sweep", seed, config, variants, comparison)

    logger.info(
        "exp10 done: peak cv at T=0.0 -> {:.3f}, at T=1.0 -> {:.3f}",
        peak_cvs[0], peak_cvs[-1],
    )
    return {"variants": variants, "comparison": comparison}

if __name__ == "__main__":
    run_experiment()
