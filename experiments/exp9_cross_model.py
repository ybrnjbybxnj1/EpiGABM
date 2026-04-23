from __future__ import annotations

from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from experiments.helpers import (
    compute_metrics,
    load_config,
    make_ollama_gabm,
    make_small_population,
    save_experiment_json,
    save_figure,
)
from src.models.abm import ABMSimulation

_MODELS = ["llama", "gemma", "qwen"]
_COLORS = {
    "llama": "#d62728",
    "gemma": "#2ca02c",
    "qwen": "#9467bd",
    "rule_based_abm": "#6c757d",
}

def _parse_jsonl_calls(logs_dir: Path, backend_name: str) -> list[dict]:
    pattern = f"*{backend_name}*.jsonl"
    files = sorted(logs_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return []
    calls = []
    with open(files[0], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "archetype_id" in entry:
                calls.append(entry)
    return calls

def _decision_agreement(
    calls_by_model: dict[str, list[dict]],
) -> dict:
    actions = ["isolate", "mask", "reduce_contacts", "see_doctor"]

    def _key(c):
        return (c.get("archetype_id"), c.get("phase"))

    common_keys = None
    for model, calls in calls_by_model.items():
        keys = {_key(c) for c in calls}
        common_keys = keys if common_keys is None else common_keys & keys

    if not common_keys:
        return {a: 0.0 for a in actions}

    lookup = {}
    for model, calls in calls_by_model.items():
        for c in calls:
            k = _key(c)
            if k in common_keys:
                lookup[(k, model)] = c.get("parsed_decision") or {}

    results = {}
    for action in actions:
        agree_count = 0
        for key in common_keys:
            vals = []
            for m in calls_by_model:
                dec = lookup.get((key, m), {})
                vals.append(bool(dec.get(action, False)))
            if all(v == vals[0] for v in vals):
                agree_count += 1
        results[action] = round(agree_count / len(common_keys), 4)
    results["n_common_decisions"] = len(common_keys)
    return results

def run_experiment(
    config_path: str = "config/default.yaml",
    seed: int = 42,
    n_agents: int = 3000,
    models: list[str] | None = None,
) -> dict:
    models = models or _MODELS
    config = load_config(config_path)
    data, households = make_small_population(n=n_agents, seed=seed)
    days = range(1, config["model"]["days"][1])
    strain = "H1N1"

    logger.info("exp9: running rule-based abm reference")
    abm = ABMSimulation(config=config, data=data.copy(), households=households.copy())
    abm_result = abm.run(days=days, seed=seed)
    abm_metrics = compute_metrics(abm_result, strain)
    abm_curve = [dr.prevalence.get(strain, 0) for dr in abm_result.days]

    variants: dict = {"rule_based_abm": abm_metrics}
    curves: dict[str, list[int]] = {"rule_based_abm": abm_curve}
    calls_by_model: dict[str, list[dict]] = {}

    logs_dir = Path(config.get("output", {}).get("logs_dir", "results/logs/"))

    for model_key in models:
        logger.info("exp9: running gabm with backend={}", model_key)
        gabm = make_ollama_gabm(
            config, data.copy(), households.copy(), backend_key=model_key,
        )
        result = gabm.run(days=days, seed=seed)

        m = compute_metrics(result, strain)
        m["peak_reduction_pct_vs_abm"] = round(
            (1 - m["peak_magnitude"] / max(abm_metrics["peak_magnitude"], 1)) * 100, 2,
        )
        variants[model_key] = m
        curves[model_key] = [dr.prevalence.get(strain, 0) for dr in result.days]

        model_name = config["agents"]["backends"][model_key]["model"]
        backend_log_tag = model_name.replace(":", "_").replace(".", "_")
        calls_by_model[model_key] = _parse_jsonl_calls(logs_dir, backend_log_tag)

    agreement = _decision_agreement(calls_by_model)

    parse_rates = {}
    for model_key, calls in calls_by_model.items():
        if not calls:
            parse_rates[model_key] = None
            continue
        ok = sum(1 for c in calls if c.get("parse_success"))
        parse_rates[model_key] = round(ok / len(calls), 4)
        variants[model_key]["parse_success_rate"] = parse_rates[model_key]
        variants[model_key]["n_llm_calls"] = len(calls)
        lats = [c.get("latency_ms", 0) for c in calls if c.get("latency_ms")]
        if lats:
            variants[model_key]["mean_latency_ms"] = int(np.mean(lats))

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, curve in curves.items():
        ax.plot(curve, color=_COLORS.get(name, "grey"),
                linewidth=2.0, label=name)
    ax.set_xlabel("simulation day")
    ax.set_ylabel("prevalence I(t)")
    ax.set_title("exp9: cross-model GABM comparison (Llama / Gemma / Qwen)")
    ax.legend(fontsize=10, title="backend", title_fontsize=10)
    plt.tight_layout()
    save_figure(fig, "exp9_cross_model_curves", config)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5.5))
    reductions = {k: variants[k]["peak_reduction_pct_vs_abm"] for k in models}
    bars = ax2.bar(list(reductions.keys()), list(reductions.values()),
                   color=[_COLORS[k] for k in reductions],
                   edgecolor="white", linewidth=1.2, alpha=0.9)
    for bar, val in zip(bars, reductions.values()):
        ax2.text(bar.get_x() + bar.get_width() / 2, val,
                 f"-{val:.1f}%", ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("peak reduction vs rule-based ABM (%)")
    ax2.set_title("exp9: peak-reduction consistency across LLM backends")
    plt.tight_layout()
    save_figure(fig2, "exp9_peak_reduction", config)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    actions = ["isolate", "mask", "reduce_contacts", "see_doctor"]
    values = [agreement.get(a, 0.0) * 100 for a in actions]
    ax3.bar(actions, values, color="#1a6bb0", alpha=0.85,
            edgecolor="white", linewidth=1.2)
    for i, v in enumerate(values):
        ax3.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=10)
    ax3.set_ylim(0, 110)
    ax3.set_ylabel("agreement rate (%)")
    ax3.set_title(
        f"exp9: cross-model decision agreement "
        f"({agreement.get('n_common_decisions', 0)} common decisions)"
    )
    plt.tight_layout()
    save_figure(fig3, "exp9_decision_agreement", config)
    plt.close(fig3)

    comparison = {
        "peak_reductions": {
            k: variants[k]["peak_reduction_pct_vs_abm"] for k in models
        },
        "peak_reduction_range_pct": round(
            max(reductions.values()) - min(reductions.values()), 2,
        ),
        "parse_success_rates": parse_rates,
        "decision_agreement": agreement,
        "interpretation": (
            "Small range of peak reductions across models and high decision "
            "agreement imply the behavioral effect is a property of the GABM "
            "approach, not of a specific LLM."
        ),
    }
    save_experiment_json("exp9_cross_model", seed, config, variants, comparison)

    logger.info(
        "exp9 done: peak-reduction range {:.1f}%, decision agreement "
        "isolate={:.0%} mask={:.0%}",
        comparison["peak_reduction_range_pct"],
        agreement.get("isolate", 0), agreement.get("mask", 0),
    )
    return {"variants": variants, "comparison": comparison}

if __name__ == "__main__":
    run_experiment()
