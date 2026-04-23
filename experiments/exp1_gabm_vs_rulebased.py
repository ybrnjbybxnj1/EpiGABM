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
from src.models.abm import ABMSimulation

def run_experiment(
    config_path: str = "config/default.yaml",
    seed: int = 42,
    n_agents: int = 200,
) -> dict:
    config = load_config(config_path)
    data, households = make_small_population(n=n_agents, seed=seed)
    days = range(1, config["model"]["days"][1])
    strain = "H1N1"

    logger.info("exp1: running rule-based abm")
    abm = ABMSimulation(config=config, data=data.copy(), households=households.copy())
    abm_result = abm.run(days=days, seed=seed)

    logger.info("exp1: running gabm (llama 3.1 8b)")
    gabm = make_llama_gabm(config, data.copy(), households.copy())
    gabm_result = gabm.run(days=days, seed=seed)

    abm_metrics = compute_metrics(abm_result, strain)
    gabm_metrics = compute_metrics(gabm_result, strain)

    comparison = {
        "peak_delay_days": gabm_metrics["peak_day"] - abm_metrics["peak_day"],
        "infection_reduction_pct": round(
            (1 - gabm_metrics["total_infected"] / max(abm_metrics["total_infected"], 1)) * 100,
            1,
        ),
        "cost_per_run_usd": gabm_metrics["llm_cost_usd"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("exp1: GABM (Llama 3.1) vs rule-based ABM", fontsize=12)

    for ax_idx, (comp, label) in enumerate(
        [("S", "susceptible"), ("E", "exposed"), ("I", "infectious"), ("R", "recovered")]
    ):
        ax = axes[ax_idx // 2, ax_idx % 2]
        abm_vals = [getattr(dr, comp).get(strain, 0) for dr in abm_result.days]
        gabm_vals = [getattr(dr, comp).get(strain, 0) for dr in gabm_result.days]
        ax.plot(abm_vals, label="rule-based", linewidth=1.5)
        ax.plot(gabm_vals, label="gabm (llama)", linewidth=1.5, linestyle="--")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel("day")

    plt.tight_layout()
    save_figure(fig, "exp1_curves", config)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    metrics_to_plot = ["peak_day", "peak_magnitude", "total_infected"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    abm_vals = [abm_metrics[m] for m in metrics_to_plot]
    gabm_vals = [gabm_metrics[m] for m in metrics_to_plot]
    ax2.bar(x - width / 2, abm_vals, width, label="rule-based")
    ax2.bar(x + width / 2, gabm_vals, width, label="gabm (llama)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_to_plot, fontsize=9)
    ax2.legend()
    ax2.set_title("exp1: peak comparison", fontsize=11)
    plt.tight_layout()
    save_figure(fig2, "exp1_peak_comparison", config)
    plt.close(fig2)

    variants = {"rulebased": abm_metrics, "gabm_llama": gabm_metrics}
    save_experiment_json("exp1_gabm_vs_rulebased", seed, config, variants, comparison)

    logger.info(
        "exp1 done: peak delay={} days, infection reduction={:.1f}%",
        comparison["peak_delay_days"], comparison["infection_reduction_pct"],
    )
    return {"variants": variants, "comparison": comparison}

if __name__ == "__main__":
    run_experiment()
