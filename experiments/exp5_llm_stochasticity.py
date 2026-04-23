from __future__ import annotations

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
from src.uncertainty.bootstrap import BootstrapUQ

def run_experiment(
    config_path: str = "config/default.yaml",
    seed: int = 42,
    n_agents: int = 200,
    n_runs: int = 10,
) -> dict:
    config = load_config(config_path)
    data, households = make_small_population(n=n_agents, seed=seed)
    days = range(1, config["model"]["days"][1])

    all_results = []

    def run_fn(run_seed: int):
        logger.info("exp5: llama run {}/{}", run_seed + 1, n_runs)
        gabm = make_llama_gabm(config, data.copy(), households.copy())
        result = gabm.run(days=days, seed=run_seed)
        all_results.append(result)
        return result

    uq = BootstrapUQ(config.get("uncertainty", {}))
    ci = uq.llm_stochasticity_uq(run_fn, n_runs=n_runs)

    peak_days = []
    peak_mags = []
    total_infs = []
    for result in all_results:
        metrics = compute_metrics(result)
        peak_days.append(metrics["peak_day"])
        peak_mags.append(metrics["peak_magnitude"])
        total_infs.append(metrics["total_infected"])

    mean_prevalence = ci.median.mean()
    ci_width = float((ci.upper - ci.lower).mean())
    ci_relative_width = ci_width / max(mean_prevalence, 1)

    variants = {
        "bootstrap": {
            "n_runs": n_runs,
            "ci_level": ci.level,
            "ci_mean_width": round(ci_width, 2),
            "ci_relative_width": round(ci_relative_width, 4),
            "peak_day_mean": round(float(np.mean(peak_days)), 1),
            "peak_day_std": round(float(np.std(peak_days)), 2),
            "peak_magnitude_mean": round(float(np.mean(peak_mags)), 1),
            "peak_magnitude_std": round(float(np.std(peak_mags)), 2),
            "total_infected_mean": round(float(np.mean(total_infs)), 1),
            "total_infected_std": round(float(np.std(total_infs)), 2),
        }
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ci.median))
    ax.fill_between(
        x, ci.lower.values, ci.upper.values,
        alpha=0.3, color="tab:blue", label=f"{ci.level:.0%} CI",
    )
    ax.plot(x, ci.median.values, color="tab:blue", linewidth=2, label="median")

    ax.set_xlabel("day")
    ax.set_ylabel("prevalence")
    ax.set_title(f"exp5: LLM stochasticity ({n_runs} Llama runs)", fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_figure(fig, "exp5_bootstrap", config)
    plt.close(fig)

    save_experiment_json("exp5_llm_stochasticity", seed, config, variants)

    logger.info(
        "exp5 done: {} runs, ci width={:.2f}, relative={:.4f}",
        n_runs, ci_width, ci_relative_width,
    )
    return {"variants": variants}

if __name__ == "__main__":
    run_experiment()
