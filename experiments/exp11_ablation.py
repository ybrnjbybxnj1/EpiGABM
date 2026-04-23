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
from src.models.abm import ABMSimulation

_VARIANTS_SPEC = [
    "full",
    "no_local_awareness",
    "no_health_stats",
    "no_compliance_caps",
    "single_archetype",
]

def _apply_ablation(config: dict, variant: str) -> dict:
    cfg = deepcopy(config)

    if variant == "full":
        return cfg

    if variant == "no_local_awareness":
        cfg.setdefault("agents", {}).setdefault("behavior_activation", {})
        cfg["agents"]["behavior_activation"]["include_local_awareness"] = False
        return cfg

    if variant == "no_health_stats":
        cfg.setdefault("agents", {})
        cfg["agents"]["disable_archetype_health"] = True
        return cfg

    if variant == "no_compliance_caps":
        cfg.setdefault("agents", {}).setdefault("compliance_rate", {})
        # caps=1.0 means llm output passes through untruncated
        cfg["agents"]["compliance_rate"] = {
            "isolate_healthy": 1.0,
            "mask": 1.0,
            "reduce_contacts": 1.0,
            "see_doctor": 1.0,
        }
        return cfg

    if variant == "single_archetype":
        # only one archetype queried; others fall through
        cfg.setdefault("agents", {})
        cfg["agents"]["ablation_single_archetype"] = "young_active"
        return cfg

    raise ValueError(f"unknown ablation variant: {variant}")

def run_experiment(
    config_path: str = "config/default.yaml",
    seed: int = 42,
    n_agents: int = 3000,
) -> dict:
    config = load_config(config_path)
    data, households = make_small_population(n=n_agents, seed=seed)
    days = range(1, config["model"]["days"][1])
    strain = "H1N1"

    logger.info("exp11: running rule-based abm baseline")
    abm = ABMSimulation(config=config, data=data.copy(), households=households.copy())
    abm_result = abm.run(days=days, seed=seed)
    abm_metrics = compute_metrics(abm_result, strain)
    abm_peak = abm_metrics["peak_magnitude"]

    variants: dict = {"rulebased_abm": abm_metrics}
    curves: dict[str, list[int]] = {
        "rulebased_abm": [dr.prevalence.get(strain, 0) for dr in abm_result.days],
    }

    for variant in _VARIANTS_SPEC:
        logger.info("exp11: ablation '{}'", variant)
        variant_config = _apply_ablation(config, variant)

        n_arch = 1 if variant == "single_archetype" else None
        gabm = make_llama_gabm(
            variant_config, data.copy(), households.copy(), n_archetypes=n_arch,
        )
        result = gabm.run(days=days, seed=seed)

        m = compute_metrics(result, strain)
        peak_reduction_pct = round(
            (1 - m["peak_magnitude"] / max(abm_peak, 1)) * 100, 2,
        )
        m["peak_reduction_pct_vs_abm"] = peak_reduction_pct
        m["variant"] = variant
        variants[variant] = m
        curves[variant] = [dr.prevalence.get(strain, 0) for dr in result.days]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {
        "rulebased_abm": "#6c757d",
        "full": "#1a6bb0",
        "no_local_awareness": "#2ca02c",
        "no_health_stats": "#ff7f0e",
        "no_compliance_caps": "#9467bd",
        "single_archetype": "#d62728",
    }
    for name, curve in curves.items():
        ax.plot(curve, color=colors.get(name, "grey"), linewidth=1.8,
                label=name)
    ax.set_xlabel("simulation day")
    ax.set_ylabel("prevalence I(t)")
    ax.set_title("exp11: GABM ablation - which mechanism drives the effect?")
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    save_figure(fig, "exp11_ablation_curves", config)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    labels = [v for v in _VARIANTS_SPEC]
    reductions = [variants[v]["peak_reduction_pct_vs_abm"] for v in labels]
    bars = ax2.bar(labels, reductions,
                   color=[colors[v] for v in labels], alpha=0.85,
                   edgecolor="white", linewidth=1.2)
    full_reduction = variants["full"]["peak_reduction_pct_vs_abm"]
    ax2.axhline(full_reduction, color="#333", linestyle="--", linewidth=1,
                label=f"full GABM: -{full_reduction:.1f}%")
    for bar, val in zip(bars, reductions):
        ax2.text(bar.get_x() + bar.get_width() / 2, val,
                 f"-{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("peak reduction vs rule-based ABM (%)")
    ax2.set_title("exp11: contribution of each GABM mechanism")
    ax2.legend(fontsize=9)
    plt.setp(ax2.get_xticklabels(), rotation=15, ha="right", fontsize=9)
    plt.tight_layout()
    save_figure(fig2, "exp11_ablation_contribution", config)
    plt.close(fig2)

    comparison = {
        "full_gabm_peak_reduction_pct": full_reduction,
        "ranked_by_lost_effect": sorted(
            [(v, full_reduction - variants[v]["peak_reduction_pct_vs_abm"])
             for v in _VARIANTS_SPEC if v != "full"],
            key=lambda kv: -kv[1],
        ),
    }
    save_experiment_json("exp11_ablation", seed, config, variants, comparison)

    logger.info(
        "exp11 done: full gabm reduces peak by {:.1f}% vs abm. "
        "most impactful ablation: {}",
        full_reduction, comparison["ranked_by_lost_effect"][0][0],
    )
    return {"variants": variants, "comparison": comparison}

if __name__ == "__main__":
    run_experiment()
