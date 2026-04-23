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

def run_experiment(
    config_path: str = "config/default.yaml",
    seed: int = 42,
    n_agents: int = 200,
    archetype_counts: list[int] | None = None,
) -> dict:
    if archetype_counts is None:
        archetype_counts = [2, 3, 4, 5, 6]

    config = load_config(config_path)
    data, households = make_small_population(n=n_agents, seed=seed)
    days = range(1, config["model"]["days"][1])
    strain = "H1N1"

    # baseline = gabm with all archetypes; measure how well smaller sets match it
    max_archetypes = max(archetype_counts)
    logger.info("exp2: running baseline gabm with {} archetypes (llama)", max_archetypes)
    baseline_gabm = make_llama_gabm(
        config, data.copy(), households.copy(), n_archetypes=max_archetypes,
    )
    baseline_result = baseline_gabm.run(days=days, seed=seed)
    baseline_prevalence = np.array([
        dr.prevalence.get(strain, 0) for dr in baseline_result.days
    ])
    baseline_metrics = compute_metrics(baseline_result, strain)

    variants = {f"baseline_gabm_{max_archetypes}_archetypes": baseline_metrics}
    rmse_list = []
    cost_list = []

    for n_arch in archetype_counts:
        logger.info("exp2: running gabm with {} archetypes (llama)", n_arch)
        gabm = make_llama_gabm(
            config, data.copy(), households.copy(), n_archetypes=n_arch,
        )
        result = gabm.run(days=days, seed=seed)

        gabm_prev = np.array([dr.prevalence.get(strain, 0) for dr in result.days])
        min_len = min(len(baseline_prevalence), len(gabm_prev))
        rmse = float(np.sqrt(np.mean(
            (baseline_prevalence[:min_len] - gabm_prev[:min_len]) ** 2
        )))

        metrics = compute_metrics(result, strain)
        metrics["rmse_vs_baseline"] = round(rmse, 4)
        metrics["n_archetypes"] = n_arch

        variants[f"gabm_{n_arch}_archetypes"] = metrics
        rmse_list.append(rmse)
        cost_list.append(metrics["llm_calls"])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "tab:blue"
    ax1.set_xlabel("number of archetypes")
    ax1.set_ylabel("RMSE vs baseline", color=color1)
    ax1.plot(archetype_counts, rmse_list, "o-", color=color1, label="RMSE")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("LLM calls", color=color2)
    ax2.plot(archetype_counts, cost_list, "s--", color=color2, label="LLM calls")
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("exp2: archetype scaling (Llama 3.1)", fontsize=11)
    plt.tight_layout()
    save_figure(fig, "exp2_scaling", config)
    plt.close(fig)

    save_experiment_json("exp2_archetype_scaling", seed, config, variants)

    logger.info("exp2 done: tested {} archetype counts", len(archetype_counts))
    return {"variants": variants}

if __name__ == "__main__":
    run_experiment()
