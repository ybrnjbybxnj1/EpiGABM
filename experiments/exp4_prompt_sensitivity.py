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

_FRAMINGS = {
    "neutral": "",
    "alarmist": (
        "The local news reports record-breaking flu deaths this week. "
        "Hospitals are overwhelmed and running out of beds."
    ),
    "dismissive": (
        "Your friends say the flu season is no worse than usual. "
        "Most people around you are going about their daily lives."
    ),
}

def run_experiment(
    config_path: str = "config/default.yaml",
    seed: int = 42,
    n_agents: int = 200,
) -> dict:
    config = load_config(config_path)
    data, households = make_small_population(n=n_agents, seed=seed)
    days = range(1, config["model"]["days"][1])
    strain = "H1N1"

    all_curves: list[list[int]] = []
    variants: dict = {}
    peak_days: list[int] = []
    peak_mags: list[int] = []

    for name, extra_prompt in _FRAMINGS.items():
        logger.info("exp4: running prompt variant '{}' (llama)", name)

        variant_config = deepcopy(config)
        if extra_prompt:
            variant_config.setdefault("model", {})
            variant_config["model"]["official_measures"] = [extra_prompt]

        gabm = make_llama_gabm(
            variant_config, data.copy(), households.copy(),
        )
        result = gabm.run(days=days, seed=seed)

        prev = [dr.prevalence.get(strain, 0) for dr in result.days]
        all_curves.append(prev)

        metrics = compute_metrics(result, strain)
        variants[f"variant_{name}"] = metrics
        peak_days.append(metrics["peak_day"])
        peak_mags.append(metrics["peak_magnitude"])

    cv_peak_day = float(np.std(peak_days) / max(np.mean(peak_days), 1))
    cv_peak_mag = float(np.std(peak_mags) / max(np.mean(peak_mags), 1))

    comparison = {
        "cv_peak_day": round(cv_peak_day, 4),
        "cv_peak_magnitude": round(cv_peak_mag, 4),
        "n_variants": len(_FRAMINGS),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    matrix = np.array(all_curves)
    mean_curve = matrix.mean(axis=0)
    min_curve = matrix.min(axis=0)
    max_curve = matrix.max(axis=0)

    x = np.arange(len(mean_curve))
    ax.fill_between(x, min_curve, max_curve, alpha=0.3, color="tab:blue", label="range")
    ax.plot(x, mean_curve, color="tab:blue", linewidth=2, label="mean")
    for i, (name, _) in enumerate(_FRAMINGS.items()):
        ax.plot(all_curves[i], alpha=0.5, linewidth=0.8, linestyle="--", label=name)

    ax.set_xlabel("day")
    ax.set_ylabel("prevalence (H1N1)")
    ax.set_title("exp4: prompt sensitivity (Llama 3.1)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    save_figure(fig, "exp4_prompt_sensitivity", config)
    plt.close(fig)

    save_experiment_json("exp4_prompt_sensitivity", seed, config, variants, comparison)

    logger.info("exp4 done: cv_peak_day={:.4f}, cv_peak_mag={:.4f}", cv_peak_day, cv_peak_mag)
    return {"variants": variants, "comparison": comparison}

if __name__ == "__main__":
    run_experiment()
