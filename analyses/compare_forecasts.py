
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from analyses.compare_backends import (
    BACKEND_DISPLAY,
    _color_for,
    _label_for,
    _short_label,
)
from analyses.helpers import (
    ANALYSES_FIGURES,
    ANALYSES_METRICS,
    ANALYSES_TRAJECTORIES,
    load_trajectories,
    save_figure,
)
import matplotlib.pyplot as plt

RAW_GABM_PEAK_HEIGHT_CV = 0.068


def _find_latest_metric_per_backend(
    metrics_dir: Path,
) -> dict[str, Path]:
    pattern = re.compile(
        r"^exp01_beta_predictor_sweep_(.+)_(\d{4}-\d{2}-\d{2}T\d+)\.json$"
    )
    latest: dict[str, tuple[str, Path]] = {}
    for p in sorted(metrics_dir.glob("exp01_beta_predictor_sweep_*.json")):
        m = pattern.match(p.name)
        if not m:
            continue
        backend = m.group(1)
        ts = m.group(2)
        if backend not in latest or ts > latest[backend][0]:
            latest[backend] = (ts, p)
    return {k: v[1] for k, v in latest.items()}


def _load_metric(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _rmse_per_method_per_t_obs(metric: dict, h: int) -> dict[str, dict[int, float]]:
    """Pull RMSE@h=H for each (method, t_obs) cell. None -> NaN."""
    summary = metric["summary_rmse_per_horizon"][f"h={h}"]
    out: dict[str, dict[int, float]] = {}
    for method, by_t_obs in summary.items():
        out[method] = {}
        for col_label, val in by_t_obs.items():
            t_obs = int(col_label.replace("t_obs=", ""))
            out[method][t_obs] = float(val) if val is not None else float("nan")
    return out


def _per_seed_rmse_h7(metric: dict, t_obs: int, method: str) -> list[float]:
    raw = metric.get("per_seed_raw", {})
    by_t_obs = raw.get(f"t_obs={t_obs}", {})
    by_method = by_t_obs.get(method, {})
    rmse = by_method.get("rmse", {})
    vals = rmse.get("7") or rmse.get(7) or []
    return [float(v) for v in vals if v is not None]


def _best_method_per_t_obs(
    metric: dict, h: int = 7,
) -> dict[int, tuple[str, float]]:
    """For each t_obs, return (best_method, its RMSE@h)."""
    rmse = _rmse_per_method_per_t_obs(metric, h)
    t_obs_set: set[int] = set()
    for by_t in rmse.values():
        t_obs_set.update(by_t.keys())
    out = {}
    for t in sorted(t_obs_set):
        best_m, best_v = None, float("inf")
        for m, by_t in rmse.items():
            v = by_t.get(t, float("nan"))
            if np.isfinite(v) and v < best_v:
                best_v = v
                best_m = m
        if best_m is not None:
            out[t] = (best_m, best_v)
    return out


def _fig_forecast_overlay(
    metrics: dict[str, dict],
    payloads: dict[str, dict],
    t_obs: int,
    method: str,
    fig_suffix: str,
    reference: dict | None,
    horizon: int = 30,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    keys = list(metrics)
    for k in keys:
        comps0 = payloads[k]["compartments"][0]
        I_truth = np.asarray(comps0["I"], dtype=float)
        n_show = min(len(I_truth), t_obs + horizon)
        x = np.arange(1, n_show + 1)
        ax.plot(x, I_truth[:n_show], color=_color_for(k, keys.index(k)),
                linewidth=1.0, alpha=0.4)

    for i, k in enumerate(keys):
        diag = metrics[k].get("diagnostics_seed0", {}).get(f"t_obs_{t_obs}", {})
        ipred = diag.get("i_predictions_seed0", {}).get(method)
        if ipred is None:
            continue
        ipred = np.asarray(ipred, dtype=float)
        x_fut = np.arange(t_obs + 1, t_obs + 1 + len(ipred))
        color = _color_for(k, i)
        ax.plot(x_fut, ipred, color=color, linewidth=2.0,
                label=_label_for(k))

    ax.axvline(t_obs + 0.5, color="black", linestyle=":",
               linewidth=1.2, alpha=0.7)
    y_top = ax.get_ylim()[1]
    ax.text(t_obs + 0.7, y_top * 0.92,
            f"day = {t_obs}",
            fontsize=10, va="top", weight="bold")

    if reference is not None:
        ref_comps0 = reference["payload"]["compartments"][0]
        I_ref = np.asarray(ref_comps0["I"], dtype=float)
        n_show_ref = min(len(I_ref), t_obs + horizon)
        x_ref = np.arange(1, n_show_ref + 1)
        ax.plot(x_ref, I_ref[:n_show_ref], color="black", linewidth=2.5,
                linestyle="--", label=reference["label"], zorder=10)

    ax.set_xlabel("day")
    ax.set_ylabel("infectious I(t)")
    ax.set_title(
        f"30-day I(t) forecast across {len(keys)} LLM backends "
        f"(seed 0, t_obs={t_obs}, method='{method}'); "
        f"thick: forecast, thin: GABM truth, black dashed: ABM",
        fontsize=10,
    )
    ax.legend(fontsize=8, ncol=2 if len(keys) > 4 else 1, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, f"compare_forecasts_overlay{fig_suffix}")


def _fig_rmse_at_h7_bar(
    metrics: dict[str, dict],
    t_obs: int,
    fig_suffix: str,
) -> dict[str, tuple[str, float]]:
    """Best-method RMSE@h=7 per backend, fixed t_obs."""
    keys = list(metrics)
    best_per_backend: dict[str, tuple[str, float]] = {}
    for k in keys:
        bests = _best_method_per_t_obs(metrics[k], h=7)
        best_per_backend[k] = bests.get(t_obs, ("?", float("nan")))

    fig, ax = plt.subplots(figsize=(max(10, 1.0 * len(keys) + 4), 5.0))
    short = [_short_label(_label_for(k)) for k in keys]
    colors = [_color_for(k, i) for i, k in enumerate(keys)]
    values = [best_per_backend[k][1] for k in keys]
    methods = [best_per_backend[k][0] for k in keys]
    bars = ax.bar(np.arange(len(keys)), values, color=colors, alpha=0.85)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(short, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("RMSE@h=7 (best method per backend)")
    finite_vals = [v for v in values if np.isfinite(v)]
    if finite_vals:
        cv_forecast = float(np.std(finite_vals) / np.mean(finite_vals))
        spread = float(max(finite_vals) - min(finite_vals))
        title = (f"7-day forecast accuracy per LLM backend (t_obs={t_obs}); "
                 f"cross-LLM CV: {cv_forecast * 100:.1f}%, "
                 f"range {min(finite_vals):.0f}..{max(finite_vals):.0f} "
                 f"(spread {spread:.0f})")
    else:
        title = f"7-day forecast accuracy per LLM backend (t_obs={t_obs})"
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    headroom = 0.04 * max(finite_vals) if finite_vals else 1.0
    for bar, m, v in zip(bars, methods, values):
        if not np.isfinite(v):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + headroom,
                m, ha="center", va="bottom", fontsize=7, rotation=15)
    plt.tight_layout()
    save_figure(fig, f"compare_forecasts_rmse_at_h7_bar{fig_suffix}")
    return best_per_backend


def _fig_best_method_grid(
    metrics: dict[str, dict],
    fig_suffix: str,
    h: int = 7,
) -> None:
    """Heatmap: backend x t_obs, cell = winning method, color = RMSE@h=7."""
    keys = list(metrics)
    short = [_short_label(_label_for(k)) for k in keys]
    t_obs_set: set[int] = set()
    for k in keys:
        for t in _best_method_per_t_obs(metrics[k], h=h).keys():
            t_obs_set.add(t)
    t_obs_grid = sorted(t_obs_set)

    matrix = np.full((len(keys), len(t_obs_grid)), np.nan)
    method_grid: list[list[str]] = [["" for _ in t_obs_grid] for _ in keys]
    for i, k in enumerate(keys):
        bests = _best_method_per_t_obs(metrics[k], h=h)
        for j, t in enumerate(t_obs_grid):
            best = bests.get(t)
            if best is None:
                continue
            method_grid[i][j] = best[0]
            matrix[i, j] = best[1]

    fig, ax = plt.subplots(
        figsize=(max(8, 1.6 * len(t_obs_grid) + 3),
                 max(4, 0.55 * len(keys) + 1.5)),
    )
    im = ax.imshow(matrix, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(t_obs_grid)))
    ax.set_xticklabels([f"t_obs={t}" for t in t_obs_grid], fontsize=9)
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels(short, fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v):
                m = method_grid[i][j]
                label = f"{m}\nRMSE={v:.0f}"
                ax.text(j, i, label, ha="center", va="center",
                        color="white" if v > np.nanmedian(matrix) else "black",
                        fontsize=7)
    ax.set_title(
        f"winning beta-prediction method per (backend, t_obs); color = RMSE@h={h} (lower better)",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, label=f"RMSE@h={h}")
    plt.tight_layout()
    save_figure(fig, f"compare_forecasts_best_method_grid{fig_suffix}")


def _fig_forecast_cv(
    metrics: dict[str, dict],
    fig_suffix: str,
    h: int = 7,
) -> None:
    """For each method, compute cross-LLM CV of RMSE@h=7 averaged over t_obs.
    Compare to raw-GABM CV constant."""
    keys = list(metrics)
    methods_set: set[str] = set()
    for k in keys:
        methods_set.update(metrics[k]["summary_rmse_per_horizon"][f"h={h}"].keys())
    methods = sorted(methods_set)

    cv_per_method: dict[str, float] = {}
    range_per_method: dict[str, tuple[float, float]] = {}
    for m in methods:
        per_backend_means: list[float] = []
        for k in keys:
            row = metrics[k]["summary_rmse_per_horizon"][f"h={h}"].get(m, {})
            vals = [v for v in row.values() if v is not None]
            if not vals:
                continue
            per_backend_means.append(float(np.mean(vals)))
        if not per_backend_means:
            cv_per_method[m] = float("nan")
            range_per_method[m] = (float("nan"), float("nan"))
            continue
        mu = float(np.mean(per_backend_means))
        sd = float(np.std(per_backend_means))
        cv_per_method[m] = sd / mu if mu > 0 else float("nan")
        range_per_method[m] = (min(per_backend_means), max(per_backend_means))

    order = sorted(methods, key=lambda m: cv_per_method.get(m, float("inf")))

    fig, ax = plt.subplots(figsize=(11, 5.4))
    x = np.arange(len(order))
    cvs = [cv_per_method[m] * 100 for m in order]
    bars = ax.bar(x, cvs, color="#9467bd", alpha=0.85,
                  label="cross-LLM CV of RMSE@h=7 (across 9 backends)")
    ax.axhline(RAW_GABM_PEAK_HEIGHT_CV * 100, color="#d62728",
               linestyle="--", linewidth=1.6,
               label=f"raw-GABM peak-height CV "
                     f"({RAW_GABM_PEAK_HEIGHT_CV * 100:.1f}%, baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("cross-LLM CV (%)")
    ax.set_title(
        "LLM-choice perturbation of forecast accuracy per beta-method; "
        "red dashed = raw-GABM peak-height CV baseline",
        fontsize=10,
    )
    headroom = 0.04 * max(cvs) if cvs else 1.0
    for bar, m, v in zip(bars, order, cvs):
        if not np.isfinite(v):
            continue
        rng = range_per_method[m]
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + headroom,
                f"{v:.1f}%\n[{rng[0]:.0f},{rng[1]:.0f}]",
                ha="center", va="bottom", fontsize=7)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save_figure(fig, f"compare_forecasts_cv{fig_suffix}")


def run_comparison(
    backends: list[str] | None,
    cache_prefix: str,
    fig_suffix: str,
    forecast_t_obs: int,
    forecast_method: str,
    reference_key: str | None,
    reference_label: str,
) -> dict | None:
    ANALYSES_FIGURES.mkdir(parents=True, exist_ok=True)
    ANALYSES_METRICS.mkdir(parents=True, exist_ok=True)

    latest = _find_latest_metric_per_backend(ANALYSES_METRICS)
    if backends:
        latest = {k: v for k, v in latest.items() if k in backends}
    if not latest:
        print("no exp01 metric files found")
        return None

    metrics: dict[str, dict] = {}
    payloads: dict[str, dict] = {}
    for k, p in latest.items():
        metrics[k] = _load_metric(p)
        traj = load_trajectories(f"{cache_prefix}{k}")
        if traj is None:
            print(f"  WARN: trajectory cache for '{k}' missing, dropping")
            metrics.pop(k)
            continue
        payloads[k] = traj

    if not metrics:
        print("no usable backends; aborting")
        return None

    keys = list(metrics)
    print(f"\nbackends in forecast comparison ({len(keys)}):")
    for k in keys:
        print(f"  - {_label_for(k):<40}  metric: {latest[k].name}")

    reference = None
    if reference_key:
        ref_payload = load_trajectories(f"{cache_prefix}{reference_key}")
        if ref_payload is not None:
            reference = {"key": reference_key, "label": reference_label,
                         "payload": ref_payload}
        else:
            print(f"  WARN: reference '{reference_key}' missing, no overlay")

    _fig_forecast_overlay(metrics, payloads, forecast_t_obs, forecast_method,
                          fig_suffix, reference)

    best_per_backend = _fig_rmse_at_h7_bar(metrics, forecast_t_obs, fig_suffix)

    _fig_best_method_grid(metrics, fig_suffix, h=7)

    _fig_forecast_cv(metrics, fig_suffix, h=7)

    print(f"\nbest beta-method per backend at t_obs={forecast_t_obs}, h=7")
    for k in keys:
        m, v = best_per_backend[k]
        print(f"  {_short_label(_label_for(k)):<26}  "
              f"method={m:<28}  RMSE@h=7={v:.1f}")

    votes: Counter = Counter()
    for k in keys:
        for t, (m, _) in _best_method_per_t_obs(metrics[k], h=7).items():
            votes[m] += 1
    print(f"\nmethod-of-the-day vote across (backend, t_obs) pairs")
    for m, c in votes.most_common():
        print(f"  {m:<32}  {c} wins")

    out = {
        "experiment":     "compare_forecasts",
        "forecast_t_obs": forecast_t_obs,
        "forecast_method": forecast_method,
        "backends":       keys,
        "metric_files": {k: latest[k].name for k in keys},
        "best_per_backend_at_t_obs": {
            k: {"method": m, "rmse_h7": v}
            for k, (m, v) in best_per_backend.items()
        },
        "method_popularity_votes": dict(votes),
    }
    out_path = ANALYSES_METRICS / f"compare_forecasts{fig_suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\nsaved metrics to {out_path}")
    print(f"saved 4 figures to {ANALYSES_FIGURES}/  (suffix: '{fig_suffix}')")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backends", type=str, default=None,
        help="comma-separated backend keys to include; default: all discovered",
    )
    parser.add_argument("--cache-prefix", type=str, default="exp01_")
    parser.add_argument("--fig-suffix", type=str, default="")
    parser.add_argument(
        "--forecast-t-obs", type=int, default=12,
        help="t_obs for the forecast-overlay figure (default 12, post-peak)",
    )
    parser.add_argument(
        "--forecast-method", type=str, default="expanding_mean",
        help="beta-prediction method for the forecast-overlay figure "
             "(default: expanding_mean  -  the Koshkareva long-horizon winner)",
    )
    parser.add_argument(
        "--reference", type=str, default="jocs_baseline",
        help="trajectory cache key used as black-dashed reference; "
             "set to '' to disable",
    )
    parser.add_argument(
        "--reference-label", type=str, default="rule-based ABM (no LLM)",
    )
    args = parser.parse_args()

    backends = None
    if args.backends:
        backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    reference_key = args.reference if args.reference else None
    run_comparison(
        backends=backends,
        cache_prefix=args.cache_prefix,
        fig_suffix=args.fig_suffix,
        forecast_t_obs=args.forecast_t_obs,
        forecast_method=args.forecast_method,
        reference_key=reference_key,
        reference_label=args.reference_label,
    )


if __name__ == "__main__":
    main()
