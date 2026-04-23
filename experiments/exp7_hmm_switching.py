from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from experiments.helpers import (
    compute_metrics,
    load_config,
    make_llama_gabm,
    make_small_population,
    save_experiment_json,
    save_figure,
)
from src.agents.backends.mock import MockBackend 
from src.models.abm import ABMSimulation
from src.models.data_structures import EpidemicContext, Phase
from src.models.hybrid import HybridModel
from src.models.seir import SEIRModel
from src.regime.relative_rt_detector import RelativeRtDetector
from src.regime.hmm_detector import HMMDetector
from src.regime.threshold import ThresholdDetector

def _extract_I(result, strain: str = "H1N1") -> np.ndarray:
    return np.array([dr.I.get(strain, 0) for dr in result.days])

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))

def _label_ground_truth(result, strain: str = "H1N1") -> list[Phase]:
    i_vals = [dr.I.get(strain, 0) for dr in result.days]
    peak = max(i_vals) if i_vals else 1
    baseline_thresh = peak * 0.05

    phases = []
    for t in range(len(i_vals)):
        if t == 0:
            phases.append(Phase.BASELINE)
            continue
        prev = max(i_vals[t - 1], 1)
        rate = (i_vals[t] - i_vals[t - 1]) / prev

        if i_vals[t] < baseline_thresh:
            phases.append(Phase.BASELINE)
        elif rate > 0.05:
            phases.append(Phase.GROWTH)
        elif rate < -0.05:
            phases.append(Phase.DECLINE)
        else:
            phases.append(Phase.PEAK)
    return phases

def _build_training_data(
    config: dict,
    data: pd.DataFrame,
    households: pd.DataFrame,
    seeds: list[int],
    strain: str = "H1N1",
) -> pd.DataFrame:
    frames = []
    days = range(1, config["model"]["days"][1])
    for s in seeds:
        abm = ABMSimulation(config=config, data=data.copy(), households=households.copy())
        result = abm.run(days=days, seed=s)
        incidence = [dr.new_infections.get(strain, 0) for dr in result.days]
        i_vals = [dr.I.get(strain, 0) for dr in result.days]
        growth = [0.0] + [
            (i_vals[t] - i_vals[t - 1]) / max(i_vals[t - 1], 1)
            for t in range(1, len(i_vals))
        ]
        frames.append(pd.DataFrame({"incidence": incidence, "growth_rate": growth}))
    return pd.concat(frames, ignore_index=True)

def _replay_detector(detector, result, strain: str = "H1N1") -> list[Phase]:
    phases = []
    prev_inf = 0
    for dr in result.days:
        total_inf = dr.I.get(strain, 0)
        gr = (total_inf - prev_inf) / max(prev_inf, 1) if prev_inf > 0 else 0.0
        ctx = EpidemicContext(
            day=dr.day,
            total_infected=total_inf,
            total_susceptible=dr.S.get(strain, 0),
            total_recovered=dr.R.get(strain, 0),
            growth_rate=gr,
            new_infections_today=dr.new_infections.get(strain, 0),
            phase=None,
        )
        phases.append(detector.detect(ctx))
        prev_inf = total_inf
    return phases

def _f1_score_safe(true_labels: list, pred_labels: list) -> float:
    classes = set(true_labels) | set(pred_labels)
    f1s = []
    for cls in classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0

_PHASE_COLORS = {
    Phase.BASELINE: "lightgrey",
    Phase.GROWTH: "lightgreen",
    Phase.PEAK: "salmon",
    Phase.DECLINE: "moccasin",
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

    logger.info("exp7: running gabm_full (ground truth)")
    gabm = make_llama_gabm(config, data.copy(), households.copy())
    gabm_result = gabm.run(days=days, seed=seed)
    gabm_metrics = compute_metrics(gabm_result, strain)
    true_phases = _label_ground_truth(gabm_result, strain)

    logger.info("exp7: training hmm on 3 abm trajectories")
    train_data = _build_training_data(
        config, data, households,
        seeds=[seed + 100, seed + 200, seed + 300],
        strain=strain,
    )
    hmm = HMMDetector(n_states=4)
    hmm.fit(train_data)

    logger.info("exp7: running hybrid_hmm")
    hmm_gabm = make_llama_gabm(config, data.copy(), households.copy())
    hmm_hybrid = HybridModel(
        abm=hmm_gabm,
        seir=SEIRModel(config),
        detector=HMMDetector(n_states=4),
    )
    hmm_hybrid.detector.fit(train_data)
    hmm_result = hmm_hybrid.run(days=days, seed=seed)
    hmm_metrics = compute_metrics(hmm_result, strain)
    hmm_switch = hmm_result.config.get("switch_day")

    logger.info("exp7: running hybrid_threshold")
    thresh_gabm = make_llama_gabm(config, data.copy(), households.copy())
    thresh_cfg = config.get("regime", {}).get("threshold", {})
    thresh_hybrid = HybridModel(
        abm=thresh_gabm,
        seir=SEIRModel(config),
        detector=ThresholdDetector(thresh_cfg, population_size=n_agents),
    )
    thresh_result = thresh_hybrid.run(days=days, seed=seed)
    thresh_metrics = compute_metrics(thresh_result, strain)
    thresh_switch = thresh_result.config.get("switch_day")

    logger.info("exp7: running hybrid_relative_rt")
    rel_gabm = make_llama_gabm(config, data.copy(), households.copy())
    rel_cfg = config.get("regime", {}).get("relative_rt", {})
    rel_hybrid = HybridModel(
        abm=rel_gabm,
        seir=SEIRModel(config),
        detector=RelativeRtDetector(population_size=n_agents, config=rel_cfg),
    )
    rel_result = rel_hybrid.run(days=days, seed=seed)
    rel_metrics = compute_metrics(rel_result, strain)
    rel_switch = rel_result.config.get("switch_day")

    logger.info("exp7: replaying detectors for phase accuracy")
    hmm_replay = HMMDetector(n_states=4)
    hmm_replay.fit(train_data)
    hmm_phases = _replay_detector(hmm_replay, gabm_result, strain)

    thresh_replay = ThresholdDetector(thresh_cfg, population_size=n_agents)
    thresh_phases = _replay_detector(thresh_replay, gabm_result, strain)

    rel_replay = RelativeRtDetector(population_size=n_agents, config=rel_cfg)
    rel_phases = _replay_detector(rel_replay, gabm_result, strain)

    true_labels = [p.value for p in true_phases]
    hmm_labels = [p.value for p in hmm_phases]
    thresh_labels = [p.value for p in thresh_phases]
    rel_labels = [p.value for p in rel_phases]

    hmm_f1 = _f1_score_safe(true_labels, hmm_labels)
    thresh_f1 = _f1_score_safe(true_labels, thresh_labels)
    rel_f1 = _f1_score_safe(true_labels, rel_labels)

    gabm_I = _extract_I(gabm_result, strain)
    hmm_I = _extract_I(hmm_result, strain)
    thresh_I = _extract_I(thresh_result, strain)
    rel_I = _extract_I(rel_result, strain)

    comparison = {
        "hmm_f1_macro": round(hmm_f1, 4),
        "threshold_f1_macro": round(thresh_f1, 4),
        "relative_rt_f1_macro": round(rel_f1, 4),
        "hmm_switch_day": hmm_switch,
        "threshold_switch_day": thresh_switch,
        "relative_rt_switch_day": rel_switch,
        "rmse_hmm_vs_gabm": round(_rmse(hmm_I, gabm_I), 2),
        "rmse_threshold_vs_gabm": round(_rmse(thresh_I, gabm_I), 2),
        "rmse_relative_rt_vs_gabm": round(_rmse(rel_I, gabm_I), 2),
        "runtime_ratio_hmm_vs_threshold": round(
            hmm_metrics["runtime_seconds"] / max(thresh_metrics["runtime_seconds"], 0.01), 2,
        ),
    }

    fig, axes = plt.subplots(4, 1, figsize=(12, 5.5), sharex=True)
    phase_data = [
        ("ground truth", true_phases),
        ("relative R_t (ours)", rel_phases),
        ("HMM detector", hmm_phases),
        ("threshold detector", thresh_phases),
    ]
    for ax, (label, phases) in zip(axes, phase_data):
        for t, phase in enumerate(phases):
            ax.axvspan(t, t + 1, color=_PHASE_COLORS.get(phase, "white"), alpha=0.8)
        ax.set_ylabel(label, fontsize=8, rotation=0, labelpad=80, va="center")
        ax.set_yticks([])
        ax.set_xlim(0, len(phases))

    axes[-1].set_xlabel("day")
    fig.suptitle("exp7: phase detection comparison", fontsize=11)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=p.value) for p, c in _PHASE_COLORS.items()
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=7, ncol=4)

    plt.tight_layout()
    save_figure(fig, "exp7_phase_detection", config)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    xs = np.arange(1, len(gabm_I) + 1)
    ax2.plot(xs, gabm_I, color="#2b2b2b", linewidth=1.8, label="GABM (ground truth)")
    ax2.plot(np.arange(1, len(rel_I) + 1), rel_I, color="#9467bd",
             linewidth=1.6, linestyle="-", label="hybrid relative R_t (ours)")
    ax2.plot(np.arange(1, len(hmm_I) + 1), hmm_I, color="#d62728",
             linewidth=1.5, linestyle="--", label="hybrid HMM")
    ax2.plot(np.arange(1, len(thresh_I) + 1), thresh_I, color="#2ca02c",
             linewidth=1.5, linestyle=":", label="hybrid threshold")
    if rel_switch is not None:
        ax2.axvline(rel_switch, color="#9467bd", linewidth=0.9,
                    linestyle="-", alpha=0.6)
        ax2.text(rel_switch, ax2.get_ylim()[1] * 0.92,
                 f" rel switch d{rel_switch}", fontsize=8, color="#9467bd")
    if hmm_switch is not None:
        ax2.axvline(hmm_switch, color="#d62728", linewidth=0.8,
                    linestyle="--", alpha=0.5)
    if thresh_switch is not None:
        ax2.axvline(thresh_switch, color="#2ca02c", linewidth=0.8,
                    linestyle=":", alpha=0.5)
    ax2.set_xlabel("day")
    ax2.set_ylabel("infectious I(t)")
    ax2.set_title("exp7: hybrid forecast across three phase detectors")
    ax2.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    save_figure(fig2, "exp7_forecast", config)
    plt.close(fig2)

    rt_series = rel_replay.rt_history()
    if rt_series:
        fig3, ax3 = plt.subplots(figsize=(11, 4))
        alpha = rel_cfg.get("alpha", 0.7)
        ax3.axhline(1.0, color="#888", linestyle=":", linewidth=0.9,
                    label="R_t = 1 (classical criterion)")
        if rel_replay._plateau is not None:
            plateau = rel_replay._plateau
            ax3.axhline(plateau, color="#c2a5cf", linestyle="--", linewidth=0.9,
                        label=f"plateau R_t = {plateau:.2f}")
            ax3.axhline(alpha * plateau, color="#9467bd", linestyle="--",
                        linewidth=0.9,
                        label=f"alpha*plateau = {alpha * plateau:.2f} (trigger)")
        ax3.plot(rt_series, color="#9467bd", linewidth=1.7, label="R_t")
        if rel_switch is not None:
            ax3.axvline(rel_switch, color="#333", linestyle="-.",
                        linewidth=1.2, alpha=0.8)
            ax3.text(rel_switch, ax3.get_ylim()[1] * 0.95,
                     f" switch d{rel_switch}", fontsize=9, color="#333",
                     va="top")
        ax3.set_xlabel("day")
        ax3.set_ylabel("R_t")
        ax3.set_title("exp7: relative R_t trajectory")
        ax3.legend(fontsize=9, loc="upper right")
        plt.tight_layout()
        save_figure(fig3, "exp7_relative_rt_trajectory", config)
        plt.close(fig3)

    variants = {
        "gabm_full": gabm_metrics,
        "hybrid_relative_rt": {
            **rel_metrics, "switch_day": rel_switch, "f1_macro": round(rel_f1, 4),
        },
        "hybrid_hmm": {
            **hmm_metrics, "switch_day": hmm_switch, "f1_macro": round(hmm_f1, 4),
        },
        "hybrid_threshold": {
            **thresh_metrics, "switch_day": thresh_switch, "f1_macro": round(thresh_f1, 4),
        },
    }
    save_experiment_json("exp7_hmm_switching", seed, config, variants, comparison)

    logger.info(
        "exp7 done: relative f1={:.3f} switch={}, hmm f1={:.3f} switch={}, "
        "threshold f1={:.3f} switch={}",
        rel_f1, rel_switch, hmm_f1, hmm_switch, thresh_f1, thresh_switch,
    )
    return {"variants": variants, "comparison": comparison}

if __name__ == "__main__":
    run_experiment()
