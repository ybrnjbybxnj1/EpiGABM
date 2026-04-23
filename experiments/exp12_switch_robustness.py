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
from src.models.data_structures import EpidemicContext
from src.regime.threshold import ThresholdDetector

def _detect_switch_from_curve(
    gabm_result, n_agents: int, thresh_cfg: dict, strain: str = "H1N1",
) -> int | None:
    detector = ThresholdDetector(thresh_cfg, population_size=n_agents)
    prev_inf = 0
    for dr in gabm_result.days:
        total = dr.I.get(strain, 0) + dr.E.get(strain, 0)
        gr = (total - prev_inf) / max(prev_inf, 1) if prev_inf > 0 else 0.0
        ctx = EpidemicContext(
            day=dr.day,
            total_infected=total,
            total_susceptible=dr.S.get(strain, 0),
            total_recovered=dr.R.get(strain, 0),
            growth_rate=gr,
            new_infections_today=dr.new_infections.get(strain, 0),
            phase=None,
        )
        if detector.should_switch(ctx):
            return dr.day
        prev_inf = total
    return None

def run_experiment(
    config_path: str = "config/default.yaml",
    seed: int = 42,
    n_agents: int = 3000,
    n_seeds: int = 5,
    use_abm_for_speed: bool = True,
) -> dict:
    config = load_config(config_path)
    data, households = make_small_population(n=n_agents, seed=seed)
    days = range(1, config["model"]["days"][1])
    strain = "H1N1"
    thresh_cfg = config.get("regime", {}).get("threshold", {})

    switch_days: list[int | None] = []
    peak_days: list[int] = []
    peak_mags: list[float] = []

    for i in range(n_seeds):
        s = seed + i
        if use_abm_for_speed:
            logger.info("exp12: trajectory {}/{} (abm, seed={})", i + 1, n_seeds, s)
            sim = ABMSimulation(
                config=config, data=data.copy(), households=households.copy(),
            )
        else:
            logger.info("exp12: trajectory {}/{} (gabm, seed={})", i + 1, n_seeds, s)
            sim = make_llama_gabm(config, data.copy(), households.copy())
        result = sim.run(days=days, seed=s)

        sd = _detect_switch_from_curve(result, n_agents, thresh_cfg, strain)
        switch_days.append(sd)
        m = compute_metrics(result, strain)
        peak_days.append(m["peak_day"])
        peak_mags.append(m["peak_magnitude"])
        logger.info(
            "  seed={}: switch_day={}, peak_day={}, peak_mag={}",
            s, sd, m["peak_day"], m["peak_magnitude"],
        )

    valid_sds = [s for s in switch_days if s is not None]
    trigger_rate = len(valid_sds) / n_seeds if n_seeds else 0.0

    if valid_sds:
        sd_arr = np.array(valid_sds)
        switch_stats = {
            "triggered_rate": round(trigger_rate, 4),
            "switch_day_mean": float(sd_arr.mean()),
            "switch_day_std": float(sd_arr.std()),
            "switch_day_min": int(sd_arr.min()),
            "switch_day_max": int(sd_arr.max()),
        }
    else:
        switch_stats = {"triggered_rate": 0.0}

    peak_arr = np.array(peak_days)
    mag_arr = np.array(peak_mags)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    seeds_axis = list(range(n_seeds))
    ax1.scatter(seeds_axis, [s if s is not None else -1 for s in switch_days],
                color="#9467bd", s=90, zorder=3,
                label="switch day (None -> -1)")
    ax1.scatter(seeds_axis, peak_days, color="#d62728", s=70,
                marker="s", zorder=2, label="peak day")
    if valid_sds:
        ax1.axhline(float(np.mean(valid_sds)), color="#9467bd",
                    linestyle="--", linewidth=1, label="switch mean")
    ax1.set_xlabel("seed index")
    ax1.set_ylabel("day of simulation")
    ax1.set_title("switch_day and peak_day across seeds")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.bar(seeds_axis, peak_mags, color="#1a6bb0", alpha=0.85,
            edgecolor="white", linewidth=1.2)
    ax2.axhline(float(mag_arr.mean()), color="#333", linestyle="--",
                linewidth=1, label=f"mean {mag_arr.mean():.0f}")
    ax2.set_xlabel("seed index")
    ax2.set_ylabel("peak magnitude")
    ax2.set_title("peak magnitude variance across seeds")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    save_figure(fig, "exp12_switch_robustness", config)
    plt.close(fig)

    variants = {
        "trajectories": {
            f"seed_{seed+i}": {
                "switch_day": switch_days[i],
                "peak_day": int(peak_days[i]),
                "peak_magnitude": int(peak_mags[i]),
            }
            for i in range(n_seeds)
        }
    }

    comparison = {
        "n_seeds": n_seeds,
        "population_size": n_agents,
        "use_abm_for_speed": use_abm_for_speed,
        **switch_stats,
        "peak_day_mean": float(peak_arr.mean()),
        "peak_day_std": float(peak_arr.std()),
        "peak_magnitude_mean": float(mag_arr.mean()),
        "peak_magnitude_std": float(mag_arr.std()),
        "interpretation": (
            "switch_day_std measures how tightly the proportional threshold "
            "detector anchors around the same change point across stochastic "
            "trajectories. Low std => stable detector, not curve-fit artefact."
        ),
    }
    save_experiment_json("exp12_switch_robustness", seed, config, variants, comparison)

    logger.info(
        "exp12 done: trigger rate {:.0%}, switch_day mean={} std={:.1f}",
        trigger_rate,
        switch_stats.get("switch_day_mean", "n/a"),
        switch_stats.get("switch_day_std", 0.0),
    )
    return {"variants": variants, "comparison": comparison}

if __name__ == "__main__":
    run_experiment()
