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
from src.uncertainty.bootstrap import BootstrapUQ

def run_experiment(
    config_path: str = "config/default.yaml",
    seed: int = 42,
    n_agents: int = 200,
) -> dict:
    config = load_config(config_path)
    data, households = make_small_population(n=n_agents, seed=seed)
    days = range(1, config["model"]["days"][1])
    strain = "H1N1"

    logger.info("exp3: running rule-based abm")
    abm = ABMSimulation(config=config, data=data.copy(), households=households.copy())
    result_abm = abm.run(days=days, seed=seed)

    logger.info("exp3: running gabm (llama)")
    gabm = make_llama_gabm(config, data.copy(), households.copy())
    result_gabm = gabm.run(days=days, seed=seed)

    abm_metrics = compute_metrics(result_abm, strain)
    gabm_metrics = compute_metrics(result_gabm, strain)

    uq = BootstrapUQ(config.get("uncertainty", {}))
    divergence = uq.model_disagreement(result_abm, result_gabm)
    mean_rel_diff = float(divergence["rel_diff"].mean())

    comparison = {
        "mean_relative_divergence": round(mean_rel_diff, 4),
        "peak_day_diff": abs(abm_metrics["peak_day"] - gabm_metrics["peak_day"]),
        "total_infected_diff": abs(
            abm_metrics["total_infected"] - gabm_metrics["total_infected"]
        ),
        "infection_reduction_pct": round(
            (1 - gabm_metrics["total_infected"] / max(abm_metrics["total_infected"], 1)) * 100,
            1,
        ),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    abm_prev = [dr.prevalence.get(strain, 0) for dr in result_abm.days]
    gabm_prev = [dr.prevalence.get(strain, 0) for dr in result_gabm.days]
    ax1.plot(abm_prev, label="rule-based abm", linewidth=1.5)
    ax1.plot(gabm_prev, label="gabm (llama)", linewidth=1.5, linestyle="--")
    ax1.set_title("prevalence comparison", fontsize=10)
    ax1.set_xlabel("day")
    ax1.set_ylabel("prevalence")
    ax1.legend(fontsize=8)

    ax2.plot(divergence["day"], divergence["rel_diff"], color="tab:orange")
    ax2.set_title("per-day relative divergence", fontsize=10)
    ax2.set_xlabel("day")
    ax2.set_ylabel("relative difference")

    fig.suptitle("exp3: GABM (Llama) vs rule-based ABM", fontsize=11)
    plt.tight_layout()
    save_figure(fig, "exp3_gabm_vs_abm", config)
    plt.close(fig)

    fig2, ax3 = plt.subplots(figsize=(10, 5))
    isolating = [dr.n_isolating for dr in result_gabm.days]
    masked = [dr.n_masked for dr in result_gabm.days]
    reducing = [dr.n_reducing_contacts for dr in result_gabm.days]
    ax3.plot(isolating, label="isolating", linewidth=1.5)
    ax3.plot(masked, label="masked", linewidth=1.5, linestyle="--")
    ax3.plot(reducing, label="reducing contacts", linewidth=1.5, linestyle=":")
    ax3.set_xlabel("day")
    ax3.set_ylabel("number of agents")
    ax3.set_title("exp3: GABM behavioral dynamics", fontsize=10)
    ax3.legend(fontsize=8)
    plt.tight_layout()
    save_figure(fig2, "exp3_behavior", config)
    plt.close(fig2)

    variants = {"rulebased_abm": abm_metrics, "gabm_llama": gabm_metrics}
    save_experiment_json("exp3_gabm_vs_abm", seed, config, variants, comparison)

    logger.info(
        "exp3 done: divergence={:.4f}, infection reduction={:.1f}%",
        mean_rel_diff, comparison["infection_reduction_pct"],
    )
    return {"variants": variants, "comparison": comparison}

if __name__ == "__main__":
    run_experiment()
