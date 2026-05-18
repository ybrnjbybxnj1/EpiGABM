
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from analyses.helpers import (
    ANALYSES_FIGURES,
    ANALYSES_METRICS,
    ANALYSES_TRAJECTORIES,
    load_trajectories,
    save_figure,
)
import matplotlib.pyplot as plt

BACKEND_DISPLAY: dict[str, tuple[str, str]] = {
    "llama":            ("Llama-3.1-8B (local)",            "#1f77b4"),
    "gemma":            ("Gemma-3-12B (local)",             "#17becf"),
    "qwen":             ("Qwen-3-8B (local)",               "#8c564b"),
    "or_llama70b":      ("Llama-3.3-70B (OpenRouter)",      "#ff7f0e"),
    "or_gemma3":        ("Gemma-3-12B (OpenRouter)",        "#bcbd22"),
    "or_gemma3_27b":    ("Gemma-3-27B (OpenRouter)",        "#9467bd"),
    "or_gemma12b":      ("Gemma-3-12B (paid)",              "#bcbd22"),
    "or_gpt_oss_20b":   ("gpt-oss-20B (OpenRouter)",        "#2ca02c"),
    "or_gpt_oss_120b":  ("gpt-oss-120B (OpenRouter)",       "#d62728"),
    "or_glm45_air":     ("GLM-4.5-Air (OpenRouter)",        "#e377c2"),
    "or_glm32b":        ("GLM-4-32B (paid)",                "#e377c2"),
    "or_nemotron9b":    ("Nemotron-Nano-9B (OpenRouter)",   "#7f7f7f"),
    "or_nemotron30b":   ("Nemotron-Nano-30B (paid)",        "#393b79"),
    "or_nemotron30b_long": ("Nemotron-30B-reasoning (mt=1500)", "#393b79"),
    "or_hermes405b":    ("Hermes-3-405B (OpenRouter)",      "#393b79"),
    "or_deepseek_v32":  ("DeepSeek-V3.2 (paid)",            "#3182bd"),
    "or_deepseek_r1d":  ("DeepSeek-R1-distill-Qwen-32B",    "#31a354"),
    "or_deepseek_r1d_long": ("DeepSeek-R1d-reasoning (mt=1500)", "#31a354"),
    "or_qwen235b":      ("Qwen-3-235B-MoE (paid)",          "#756bb1"),
    "or_mistral24b":    ("Mistral-Small-24B (paid)",        "#e6550d"),
    "or_claude_haiku":  ("Claude-3-Haiku (Anthropic)",      "#a55194"),
}

_FALLBACK_PALETTE = plt.cm.tab20.colors

METRIC_CARDS = [
    ("peak_height", "peak height (infectious agents)"),
    ("peak_day",    "peak day (since outbreak start)"),
    ("total_inf",   "total infections (R+I at end)"),
    ("attack_rate", "attack rate (fraction infected)"),
]


def _label_for(key: str) -> str:
    if key in BACKEND_DISPLAY:
        return BACKEND_DISPLAY[key][0]
    return key


def _color_for(key: str, fallback_idx: int) -> str:
    if key in BACKEND_DISPLAY:
        return BACKEND_DISPLAY[key][1]
    return _FALLBACK_PALETTE[fallback_idx % len(_FALLBACK_PALETTE)]


def _short_label(label: str) -> str:
    return label.split(" (")[0]


def _aggregate(trajectories_payload: dict) -> dict:
    comps = trajectories_payload["compartments"]
    peak_days, peak_heights, total_infs, attack_rates = [], [], [], []
    n_pop = comps[0]["S"][0] + comps[0]["E"][0] + comps[0]["I"][0] + comps[0]["R"][0]

    for c in comps:
        I = np.asarray(c["I"], dtype=float)
        R = np.asarray(c["R"], dtype=float)
        active = np.where(I > 0)[0]
        if len(active) == 0:
            continue
        peak_d = int(active[np.argmax(I[active])])
        peak_h = float(I[peak_d])
        total_inf = float(R[-1] + I[-1])
        attack_rate = total_inf / n_pop
        peak_days.append(peak_d + 1)
        peak_heights.append(peak_h)
        total_infs.append(total_inf)
        attack_rates.append(attack_rate)

    return {
        "n_seeds":          len(peak_days),
        "n_pop":            int(n_pop),
        "peak_day_mean":    float(np.mean(peak_days)),
        "peak_day_std":     float(np.std(peak_days)),
        "peak_height_mean": float(np.mean(peak_heights)),
        "peak_height_std":  float(np.std(peak_heights)),
        "peak_height_cv":   float(np.std(peak_heights) / np.mean(peak_heights)),
        "total_inf_mean":   float(np.mean(total_infs)),
        "total_inf_std":    float(np.std(total_infs)),
        "attack_rate_mean": float(np.mean(attack_rates)),
        "attack_rate_std":  float(np.std(attack_rates)),
        "peak_days":        peak_days,
        "peak_heights":     peak_heights,
        "total_infs":       total_infs,
        "attack_rates":     attack_rates,
    }


def _stack_curves(payload: dict, key: str, max_days: int = 60) -> np.ndarray:
    series = []
    for c in payload["compartments"]:
        v = np.asarray(c[key], dtype=float)
        if len(v) >= max_days:
            series.append(v[:max_days])
        else:
            pad = np.full(max_days - len(v), v[-1] if len(v) else 0.0)
            series.append(np.concatenate([v, pad]))
    return np.stack(series, axis=0)


def _stack_betas(payload: dict, max_days: int = 60) -> np.ndarray:
    series = []
    for b in payload["betas"]:
        v = np.asarray(b, dtype=float)
        if len(v) >= max_days:
            series.append(v[:max_days])
        else:
            pad = np.full(max_days - len(v), v[-1] if len(v) else 0.0)
            series.append(np.concatenate([v, pad]))
    return np.stack(series, axis=0)


def _discover_backends(cache_prefix: str) -> list[str]:
    ANALYSES_TRAJECTORIES.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(cache_prefix)}(.+)\.pkl$")
    found: list[str] = []
    for p in sorted(ANALYSES_TRAJECTORIES.glob(f"{cache_prefix}*.pkl")):
        m = pattern.match(p.name)
        if m:
            found.append(m.group(1))
    return found


def _fig_i_curves(payloads, aggregates, keys, colors, labels,
                  fig_suffix, max_days=50, reference=None) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    n_seeds_str = ", ".join(
        str(aggregates[k]["n_seeds"]) for k in keys
    )
    for k, color, label in zip(keys, colors, labels):
        I = _stack_curves(payloads[k], "I", max_days)
        x = np.arange(1, max_days + 1)
        med = np.median(I, axis=0)
        lo = np.percentile(I, 25, axis=0)
        hi = np.percentile(I, 75, axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        ax.plot(x, med, color=color, linewidth=2.0, label=label)
    if reference is not None:
        ref_I = _stack_curves(reference["payload"], "I", max_days)
        ref_med = np.median(ref_I, axis=0)
        x = np.arange(1, max_days + 1)
        ax.plot(x, ref_med, color="black", linewidth=2.5, linestyle="--",
                label=reference["label"], zorder=10)
    ax.set_xlabel("day")
    ax.set_ylabel("infectious I(t) (median across seeds; IQR shaded)")
    ax.set_title(
        f"GABM I(t) by LLM backend ({len(keys)} backends, "
        f"3000 agents, seeds: {n_seeds_str})",
        fontsize=10,
    )
    ax.legend(fontsize=8, ncol=2 if len(keys) > 4 else 1, loc="upper right")
    plt.tight_layout()
    save_figure(fig, f"compare_backends_I_curves{fig_suffix}")


def _fig_beta_curves(payloads, aggregates, keys, colors, labels,
                     fig_suffix, max_days=50, reference=None) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for k, color, label in zip(keys, colors, labels):
        b = _stack_betas(payloads[k], max_days)
        n_pop = aggregates[k]["n_pop"]
        x = np.arange(1, max_days + 1)
        med = np.median(b * n_pop, axis=0)
        lo = np.percentile(b * n_pop, 25, axis=0)
        hi = np.percentile(b * n_pop, 75, axis=0)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        ax.plot(x, med, color=color, linewidth=2.0, label=label)
    if reference is not None:
        ref_b = _stack_betas(reference["payload"], max_days)
        ref_n_pop = reference["n_pop"]
        ref_med = np.median(ref_b * ref_n_pop, axis=0)
        x = np.arange(1, max_days + 1)
        ax.plot(x, ref_med, color="black", linewidth=2.5, linestyle="--",
                label=reference["label"], zorder=10)
    ax.set_xlabel("day")
    ax.set_ylabel(r"$\beta(t) \cdot N$  (median across seeds; IQR shaded)")
    ax.set_title("GABM empirical $\\beta(t)$ by LLM backend", fontsize=11)
    ax.legend(fontsize=8, ncol=2 if len(keys) > 4 else 1, loc="upper right")
    plt.tight_layout()
    save_figure(fig, f"compare_backends_beta_curves{fig_suffix}")


def _fig_metric_bars(aggregates, keys, colors, short_labels, fig_suffix) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(max(14, 1.4 * len(keys) + 8), 4.8))
    for ax, (mtrc, ylabel) in zip(
        axs,
        [("peak_height", "peak height (infectious agents)"),
         ("peak_day",    "peak day (since outbreak start)"),
         ("attack_rate", "attack rate (fraction infected)")],
    ):
        means_l = [aggregates[k][f"{mtrc}_mean"] for k in keys]
        stds_l = [aggregates[k][f"{mtrc}_std"] for k in keys]
        ax.bar(np.arange(len(keys)), means_l, yerr=stds_l, capsize=4,
               color=colors, alpha=0.85)
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{mtrc} (mean +/- std)", fontsize=10)
    plt.tight_layout()
    save_figure(fig, f"compare_backends_metrics{fig_suffix}")


def _fig_peak_box(aggregates, keys, colors, short_labels, fig_suffix,
                  cv_cross: float) -> None:
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(keys) + 4), 5.4))
    box_data = [aggregates[k]["peak_heights"] for k in keys]
    means = [aggregates[k]["peak_height_mean"] for k in keys]
    bp = ax.boxplot(box_data, patch_artist=True, tick_labels=short_labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("peak height (infectious agents)")
    ax.set_title(
        f"peak height per LLM backend ({len(keys)}): "
        f"cross-LLM CV {cv_cross * 100:.1f}%, range {max(means) - min(means):.0f} agents",
        fontsize=11,
    )
    plt.xticks(rotation=25, ha="right", fontsize=8)
    plt.tight_layout()
    save_figure(fig, f"compare_backends_peak_box{fig_suffix}")


def _fig_heatmap_zscore(aggregates, keys, short_labels, fig_suffix) -> None:
    metric_keys = [m[0] for m in METRIC_CARDS]
    matrix = np.array([
        [aggregates[k][f"{m}_mean"] for k in keys]
        for m in metric_keys
    ], dtype=float)
    mu = matrix.mean(axis=1, keepdims=True)
    sd = matrix.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    z = (matrix - mu) / sd

    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(keys) + 3), 0.7 * len(metric_keys) + 1.8))
    vmax = max(abs(z.min()), abs(z.max()), 1.0)
    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(metric_keys)))
    ax.set_yticklabels([m[1] for m in METRIC_CARDS], fontsize=9)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i, j]:+.1f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(z[i, j]) > 1.0 else "black")
    ax.set_title("metric deviation from cross-LLM mean (z-score)", fontsize=11)
    plt.colorbar(im, ax=ax, label="z-score (sigmas from cross-LLM mean)")
    plt.tight_layout()
    save_figure(fig, f"compare_backends_heatmap_zscore{fig_suffix}")


def _fig_scatter(aggregates, keys, colors, labels, fig_suffix,
                 reference=None) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for k, color, label in zip(keys, colors, labels):
        a = aggregates[k]
        ax.scatter(a["peak_days"], a["peak_heights"], color=color, s=70,
                   edgecolor="white", linewidth=0.7, alpha=0.85,
                   label=label, zorder=3)
        ax.scatter([a["peak_day_mean"]], [a["peak_height_mean"]],
                   color=color, s=260, marker="X",
                   edgecolor="black", linewidth=1.2, zorder=4)
    if reference is not None:
        ra = reference["aggregate"]
        ax.scatter(ra["peak_days"], ra["peak_heights"], color="black",
                   marker="o", facecolor="none", edgecolor="black",
                   s=90, linewidth=1.4, label=reference["label"], zorder=5)
        ax.scatter([ra["peak_day_mean"]], [ra["peak_height_mean"]],
                   color="black", s=320, marker="*",
                   edgecolor="white", linewidth=1.4, zorder=6)
        ax.axhline(ra["peak_height_mean"], color="black",
                   linestyle="--", linewidth=0.8, alpha=0.5, zorder=2)
        ax.axvline(ra["peak_day_mean"], color="black",
                   linestyle="--", linewidth=0.8, alpha=0.5, zorder=2)
    ax.set_xlabel("peak day")
    ax.set_ylabel("peak height (infectious agents)")
    ax.set_title(
        "per-seed peak position by LLM backend (dots: seeds, X: centroid)",
        fontsize=11,
    )
    ax.legend(fontsize=8, ncol=2 if len(keys) > 4 else 1, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, f"compare_backends_scatter{fig_suffix}")


def _fig_violins(aggregates, keys, colors, short_labels, fig_suffix) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(max(12, 1.3 * len(keys) + 6), 5.0))
    for ax, mtrc, ylabel, raw_key in [
        (axs[0], "peak_height", "peak height (infectious agents)", "peak_heights"),
        (axs[1], "attack_rate", "attack rate (fraction infected)", "attack_rates"),
    ]:
        data = [aggregates[k][raw_key] for k in keys]
        parts = ax.violinplot(data, positions=np.arange(len(keys)),
                              showmeans=True, showmedians=True, widths=0.8)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
            pc.set_edgecolor("black")
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{mtrc} distribution per backend", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save_figure(fig, f"compare_backends_violins{fig_suffix}")


def _fig_cv_bars(aggregates, keys, fig_suffix) -> None:
    metric_keys = [m[0] for m in METRIC_CARDS]
    cross_cv = []
    within_cv_median = []
    for mk in metric_keys:
        means = np.array([aggregates[k][f"{mk}_mean"] for k in keys])
        stds = np.array([aggregates[k][f"{mk}_std"] for k in keys])
        cross_cv.append(float(np.std(means) / np.mean(means))
                        if np.mean(means) > 0 else 0.0)
        within = stds / np.where(means > 0, means, 1.0)
        within_cv_median.append(float(np.median(within)))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(metric_keys))
    bw = 0.38
    ax.bar(x - bw / 2, np.array(cross_cv) * 100, width=bw,
           color="#d62728", alpha=0.85, label="between-LLM CV (LLM choice effect)")
    ax.bar(x + bw / 2, np.array(within_cv_median) * 100, width=bw,
           color="#1f77b4", alpha=0.85, label="within-LLM CV (median across backends)")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1].split(" (")[0] for m in METRIC_CARDS],
                       fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("coefficient of variation (%)")
    ax.set_title(
        "between- vs within-LLM variability per metric (blue >= red: LLM choice < seed noise)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    for i, (a_cv, w_cv) in enumerate(zip(cross_cv, within_cv_median)):
        ax.text(i - bw / 2, a_cv * 100 + 0.5, f"{a_cv*100:.1f}%",
                ha="center", fontsize=8)
        ax.text(i + bw / 2, w_cv * 100 + 0.5, f"{w_cv*100:.1f}%",
                ha="center", fontsize=8)
    plt.tight_layout()
    save_figure(fig, f"compare_backends_cv_bars{fig_suffix}")


def _fig_table(aggregates, keys, labels, fig_suffix, cv_cross_height: float,
               reference=None) -> None:
    rows = []
    header = ["backend", "n", "peak day",
              "peak height", "total infections", "attack rate"]
    if reference is not None:
        ra = reference["aggregate"]
        rows.append([
            f"{reference['label']} (REFERENCE)",
            f"{ra['n_seeds']}",
            f"{ra['peak_day_mean']:.1f} ± {ra['peak_day_std']:.1f}",
            f"{ra['peak_height_mean']:.0f} ± {ra['peak_height_std']:.0f}",
            f"{ra['total_inf_mean']:.0f} ± {ra['total_inf_std']:.0f}",
            f"{ra['attack_rate_mean']:.3f} ± {ra['attack_rate_std']:.3f}",
        ])
    for k, lab in zip(keys, labels):
        a = aggregates[k]
        rows.append([
            _short_label(lab),
            f"{a['n_seeds']}",
            f"{a['peak_day_mean']:.1f} ± {a['peak_day_std']:.1f}",
            f"{a['peak_height_mean']:.0f} ± {a['peak_height_std']:.0f}",
            f"{a['total_inf_mean']:.0f} ± {a['total_inf_std']:.0f}",
            f"{a['attack_rate_mean']:.3f} ± {a['attack_rate_std']:.3f}",
        ])
    rows.append([
        "cross-LLM CV (peak height)",
        "",
        "",
        f"{cv_cross_height * 100:.1f}%",
        "",
        "",
    ])
    fig, ax = plt.subplots(
        figsize=(max(11, 1.4 * len(header)), 0.45 * (len(rows) + 1) + 1.0),
    )
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=header,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    for j in range(len(header)):
        cell = table[0, j]
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("#404040")
    for j in range(len(header)):
        cell = table[len(rows), j]
        cell.set_facecolor("#fff2cc")
        cell.set_text_props(weight="bold")
    if reference is not None:
        for j in range(len(header)):
            cell = table[1, j]
            cell.set_facecolor("#e0e0e0")
            cell.set_text_props(weight="bold")
    ax.set_title(
        f"GABM cross-LLM comparison: {len(keys)} backends",
        fontsize=11, pad=10,
    )
    save_figure(fig, f"compare_backends_table{fig_suffix}")


def run_comparison(
    backends: list[str],
    cache_prefix: str,
    fig_suffix: str,
    reference_key: str | None = None,
    reference_label: str = "rule-based ABM (no LLM)",
) -> dict | None:
    ANALYSES_FIGURES.mkdir(parents=True, exist_ok=True)
    ANALYSES_METRICS.mkdir(parents=True, exist_ok=True)

    aggregates: dict[str, dict] = {}
    payloads: dict[str, dict] = {}
    backends_filtered = [b for b in backends if b != reference_key]
    for k in backends_filtered:
        payload = load_trajectories(f"{cache_prefix}{k}")
        if payload is None:
            print(f"  WARN: cache for '{k}' not found ({cache_prefix}{k}.pkl), skipping")
            continue
        payloads[k] = payload
        aggregates[k] = _aggregate(payload)

    if not aggregates:
        print("no caches found; nothing to compare")
        return None

    reference: dict | None = None
    if reference_key is not None:
        ref_payload = load_trajectories(f"{cache_prefix}{reference_key}")
        if ref_payload is None:
            print(f"  WARN: reference cache '{reference_key}' not found, "
                  f"falling back to no reference")
        else:
            ref_agg = _aggregate(ref_payload)
            ref_n_pop = ref_payload["compartments"][0]["S"][0] + ref_payload["compartments"][0]["E"][0] \
                        + ref_payload["compartments"][0]["I"][0] + ref_payload["compartments"][0]["R"][0]
            reference = {
                "key": reference_key,
                "label": reference_label,
                "payload": ref_payload,
                "aggregate": ref_agg,
                "n_pop": int(ref_n_pop),
            }

    keys = list(aggregates)
    labels = [_label_for(k) for k in keys]
    colors = [_color_for(k, i) for i, k in enumerate(keys)]
    short_labels = [_short_label(lab) for lab in labels]

    print("\nGABM cross-LLM comparison (direct, from trajectory caches)\n")
    cols = ["peak_day", "peak_height", "total_inf", "attack_rate"]
    col_w = 30
    print(f"{'metric':<20}", end="")
    for lab in labels:
        print(f"{lab[:col_w - 2]:<{col_w}}", end="")
    print()
    print("-" * (20 + col_w * len(labels)))
    for mtrc in cols:
        print(f"{mtrc:<20}", end="")
        for k in keys:
            a = aggregates[k]
            mean = a[f"{mtrc}_mean"]
            std = a[f"{mtrc}_std"]
            print(f"{mean:>8.2f} +- {std:<6.2f}            ", end="")
        print()

    means = [aggregates[k]["peak_height_mean"] for k in keys]
    cv_cross = float(np.std(means) / np.mean(means))
    print()
    print(f"cross-LLM CV on peak height: {cv_cross * 100:.1f}%")
    print(f"cross-LLM peak height range: {min(means):.0f} ... {max(means):.0f}  "
          f"(spread {max(means) - min(means):.0f})")

    n_seeds_per_backend = {k: aggregates[k]["n_seeds"] for k in keys}

    _fig_i_curves(payloads, aggregates, keys, colors, labels, fig_suffix,
                  reference=reference)
    _fig_beta_curves(payloads, aggregates, keys, colors, labels, fig_suffix,
                     reference=reference)
    _fig_metric_bars(aggregates, keys, colors, short_labels, fig_suffix)
    _fig_peak_box(aggregates, keys, colors, short_labels, fig_suffix, cv_cross)
    _fig_heatmap_zscore(aggregates, keys, short_labels, fig_suffix)
    _fig_scatter(aggregates, keys, colors, labels, fig_suffix,
                 reference=reference)
    _fig_violins(aggregates, keys, colors, short_labels, fig_suffix)
    _fig_cv_bars(aggregates, keys, fig_suffix)
    _fig_table(aggregates, keys, labels, fig_suffix, cv_cross,
               reference=reference)

    max_days_curves = 50
    curves: dict = {}
    for k in keys:
        I_arr = _stack_curves(payloads[k], "I", max_days_curves)
        E_arr = _stack_curves(payloads[k], "E", max_days_curves)
        S_arr = _stack_curves(payloads[k], "S", max_days_curves)
        R_arr = _stack_curves(payloads[k], "R", max_days_curves)
        b_arr = _stack_betas(payloads[k], max_days_curves)
        curves[k] = {
            "max_days": max_days_curves,
            "I_median":  np.median(I_arr, axis=0).tolist(),
            "I_iqr_lo":  np.percentile(I_arr, 25, axis=0).tolist(),
            "I_iqr_hi":  np.percentile(I_arr, 75, axis=0).tolist(),
            "E_median":  np.median(E_arr, axis=0).tolist(),
            "S_median":  np.median(S_arr, axis=0).tolist(),
            "R_median":  np.median(R_arr, axis=0).tolist(),
            "beta_median": np.median(b_arr, axis=0).tolist(),
            "beta_iqr_lo": np.percentile(b_arr, 25, axis=0).tolist(),
            "beta_iqr_hi": np.percentile(b_arr, 75, axis=0).tolist(),
        }

    out = {
        "experiment": "compare_backends",
        "backends":   keys,
        "n_pop":      aggregates[keys[0]]["n_pop"],
        "n_seeds_per_backend":         n_seeds_per_backend,
        "cross_llm_cv_peak_height":    cv_cross,
        "cross_llm_peak_height_range": float(max(means) - min(means)),
        "per_backend": {
            k: {
                kk: vv for kk, vv in aggregates[k].items()
                if kk not in ("peak_days", "peak_heights",
                              "total_infs", "attack_rates")
            }
            for k in keys
        },
        "per_backend_raw": {
            k: {
                "peak_days":    aggregates[k]["peak_days"],
                "peak_heights": aggregates[k]["peak_heights"],
                "total_infs":   aggregates[k]["total_infs"],
                "attack_rates": aggregates[k]["attack_rates"],
            }
            for k in keys
        },
        "curves": curves,
    }
    if reference is not None:
        ref_I = _stack_curves(reference["payload"], "I", max_days_curves)
        ref_b = _stack_betas(reference["payload"], max_days_curves)
        ref_n_pop = reference["n_pop"]
        out["reference"] = {
            "key":         reference["key"],
            "label":       reference["label"],
            "n_seeds":     reference["aggregate"]["n_seeds"],
            "n_pop":       ref_n_pop,
            "aggregate":   {
                kk: vv for kk, vv in reference["aggregate"].items()
                if kk not in ("peak_days", "peak_heights",
                              "total_infs", "attack_rates")
            },
            "raw":         {
                "peak_days":    reference["aggregate"]["peak_days"],
                "peak_heights": reference["aggregate"]["peak_heights"],
                "total_infs":   reference["aggregate"]["total_infs"],
                "attack_rates": reference["aggregate"]["attack_rates"],
            },
            "curves": {
                "max_days":    max_days_curves,
                "I_median":    np.median(ref_I, axis=0).tolist(),
                "I_iqr_lo":    np.percentile(ref_I, 25, axis=0).tolist(),
                "I_iqr_hi":    np.percentile(ref_I, 75, axis=0).tolist(),
                "beta_median": np.median(ref_b * ref_n_pop, axis=0).tolist(),
            },
        }
    out_path = ANALYSES_METRICS / f"compare_backends{fig_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\nsaved metrics to {out_path}")
    print(f"saved 9 figures to {ANALYSES_FIGURES}/  (suffix: '{fig_suffix}')")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backends", type=str, default=None,
        help="comma-separated list of backend keys to compare. "
             "default: auto-discover all caches matching --cache-prefix",
    )
    parser.add_argument(
        "--cache-prefix", type=str, default="exp01_",
        help="prefix of pickle filenames in results/analyses/trajectories/",
    )
    parser.add_argument(
        "--fig-suffix", type=str, default="",
        help="appended to figure / json filenames so multiple comparison "
             "runs don't clobber each other (e.g. '_3backends', '_full')",
    )
    parser.add_argument(
        "--reference", type=str, default="jocs_baseline",
        help="cache key (relative to --cache-prefix) of the trajectory used "
             "as the black dashed reference overlay. set to '' to disable. "
             "default: 'jocs_baseline' "
             "(loads results/analyses/trajectories/exp01_jocs_baseline.pkl)",
    )
    parser.add_argument(
        "--reference-label", type=str, default="rule-based ABM (no LLM)",
        help="legend label for the reference overlay",
    )
    args = parser.parse_args()

    if args.backends:
        backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    else:
        backends = _discover_backends(args.cache_prefix)
        if not backends:
            print(f"no caches found under {ANALYSES_TRAJECTORIES} "
                  f"matching '{args.cache_prefix}*.pkl'")
            return
        print(f"auto-discovered backends: {backends}")

    reference_key = args.reference if args.reference else None
    run_comparison(backends, args.cache_prefix, args.fig_suffix,
                   reference_key=reference_key,
                   reference_label=args.reference_label)


if __name__ == "__main__":
    main()
