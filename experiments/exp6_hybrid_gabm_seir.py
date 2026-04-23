from __future__ import annotations

import time

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
from src.models.data_structures import SEIRState
from src.models.hybrid import HybridModel
from src.models.seir import SEIRModel
from src.regime.threshold import ThresholdDetector

def _extract_I(result, strain: str = "H1N1") -> np.ndarray:
    return np.array([dr.I.get(strain, 0) for dr in result.days])

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))

def run_experiment(
    config_path: str = "config/default.yaml",
    seed: int = 42,
    n_agents: int = 200,
) -> dict:
    config = load_config(config_path)
    data, households = make_small_population(n=n_agents, seed=seed)
    days = range(1, config["model"]["days"][1])
    strain = "H1N1"

    logger.info("exp6: running abm_full")
    abm = ABMSimulation(config=config, data=data.copy(), households=households.copy())
    abm_result = abm.run(days=days, seed=seed)
    abm_metrics = compute_metrics(abm_result, strain)

    logger.info("exp6: running gabm_full (llama)")
    t0 = time.time()
    gabm = make_llama_gabm(config, data.copy(), households.copy())
    gabm_result = gabm.run(days=days, seed=seed)
    gabm_time = time.time() - t0
    gabm_metrics = compute_metrics(gabm_result, strain)

    logger.info("exp6: running hybrid_gabm_seir (llama)")
    hybrid_gabm = make_llama_gabm(config, data.copy(), households.copy())
    seir = SEIRModel(config)
    thresh_cfg = config.get("regime", {}).get("threshold", {})
    detector = ThresholdDetector(thresh_cfg, population_size=n_agents)
    hybrid = HybridModel(abm=hybrid_gabm, seir=seir, detector=detector)
    t0 = time.time()
    hybrid_result = hybrid.run(days=days, seed=seed)
    hybrid_time = time.time() - t0
    hybrid_metrics = compute_metrics(hybrid_result, strain)
    switch_day = hybrid_result.config.get("switch_day")

    # pure seir, beta lifted from the gabm run's mean
    logger.info("exp6: running seir_only")
    betas = [dr.beta.get(strain, 0.3) for dr in gabm_result.days]
    pos_betas = [b for b in betas if b > 0]
    avg_beta = float(np.mean(pos_betas)) if pos_betas else 0.3

    day1 = gabm_result.days[0]
    initial = SEIRState(
        S=day1.S.get(strain, n_agents - 10), E=day1.E.get(strain, 0),
        I=day1.I.get(strain, 10), R=day1.R.get(strain, 0), N=n_agents,
    )
    t0 = time.time()
    seir_states = SEIRModel(config).run(
        days=range(len(list(days)) + 1), initial=initial,
        beta=avg_beta, stochastic=False,
    )
    seir_time = time.time() - t0
    seir_I = np.array([s.I for s in seir_states[1:]])
    i_vals = [s.I for s in seir_states]
    peak_idx = int(np.argmax(i_vals))
    seir_only_metrics = {
        "peak_day": peak_idx,
        "peak_magnitude": i_vals[peak_idx],
        "total_infected": int(max(s.R for s in seir_states)),
        "rmse_vs_observed": None, "r_eff_mean": 0.0, "wave_count": 0,
        "n_behavior_updates": 0, "runtime_seconds": round(seir_time, 4),
        "llm_calls": 0, "llm_cost_usd": 0.0,
    }

    gabm_I = _extract_I(gabm_result, strain)
    abm_I = _extract_I(abm_result, strain)
    hybrid_I = _extract_I(hybrid_result, strain)

    comparison = {
        "rmse_hybrid_vs_gabm": round(_rmse(hybrid_I, gabm_I), 2),
        "rmse_seir_vs_gabm": round(_rmse(seir_I, gabm_I), 2),
        "rmse_abm_vs_gabm": round(_rmse(abm_I, gabm_I), 2),
        "switch_day": switch_day,
        "peak_day_diff_hybrid_gabm": hybrid_metrics["peak_day"] - gabm_metrics["peak_day"],
        "runtime_gabm_full": round(gabm_time, 1),
        "runtime_hybrid": round(hybrid_time, 1),
        "runtime_ratio_hybrid_vs_gabm": round(
            hybrid_time / max(gabm_time, 0.01), 2,
        ),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    x_abm = np.arange(1, len(abm_I) + 1)
    x_gabm = np.arange(1, len(gabm_I) + 1)
    x_hybrid = np.arange(1, len(hybrid_I) + 1)
    ax.plot(x_abm, abm_I, color="grey", linewidth=1.2, label="abm (rule-based)")
    ax.plot(x_gabm, gabm_I, color="blue", linewidth=1.5, label="gabm (llama, full)")
    ax.plot(x_hybrid, hybrid_I, color="red", linewidth=1.5, linestyle="--",
            label="hybrid gabm->seir")
    n_plot = min(len(seir_I), len(gabm_I))
    ax.plot(np.arange(1, n_plot + 1), seir_I[:n_plot], color="green", linewidth=1.2,
            linestyle=":", label="seir only")
    ax.set_xlabel("day")
    ax.set_ylabel("infectious (I)")
    ax.set_title("exp6: hybrid GABM->SEIR (Llama 3.1)", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_figure(fig, "exp6_curves", config)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(x_gabm, gabm_I, color="lightblue", linewidth=1.2, label="gabm (reference)")
    ax2.plot(x_hybrid, hybrid_I, color="red", linewidth=1.5, label="hybrid gabm->seir")
    if switch_day is not None:
        ax2.axvline(switch_day, color="black", linestyle="--", linewidth=1,
                    label=f"switch day={switch_day}")
        ax2.axvspan(1, switch_day, alpha=0.08, color="blue")
        ax2.axvspan(switch_day, len(hybrid_I) + 1, alpha=0.08, color="orange")
    ax2.set_xlabel("day")
    ax2.set_ylabel("infectious (I)")
    ax2.set_title("exp6: hybrid switching point", fontsize=11)
    ax2.legend(fontsize=9)
    plt.tight_layout()
    save_figure(fig2, "exp6_switching", config)
    plt.close(fig2)

    variants = {
        "abm_full": abm_metrics,
        "gabm_llama_full": gabm_metrics,
        "hybrid_gabm_seir": hybrid_metrics,
        "seir_only": seir_only_metrics,
    }
    save_experiment_json("exp6_hybrid_gabm_seir", seed, config, variants, comparison)

    logger.info(
        "exp6 done: switch_day={}, rmse hybrid={}, rmse seir={}",
        switch_day, comparison["rmse_hybrid_vs_gabm"], comparison["rmse_seir_vs_gabm"],
    )
    return {"variants": variants, "comparison": comparison}

if __name__ == "__main__":
    run_experiment()
