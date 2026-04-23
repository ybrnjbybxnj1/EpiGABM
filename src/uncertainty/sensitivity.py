from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from src.models.abm import ABMSimulation
    from src.models.gabm import GABMSimulation

class SensitivityAnalyzer:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def sobol_indices(
        self,
        model_factory: Callable,
        param_ranges: dict[str, tuple[float, float]],
        n_samples: int = 256,
        metric_fn: Callable | None = None,
    ) -> pd.DataFrame:
        from SALib.analyze import sobol
        from SALib.sample import sobol as sobol_sample

        if metric_fn is None:
            metric_fn = _default_metric

        problem = {
            "num_vars": len(param_ranges),
            "names": list(param_ranges.keys()),
            "bounds": list(param_ranges.values()),
        }

        np.random.seed(self.seed)
        X = sobol_sample.sample(problem, n_samples)
        logger.info("sobol: evaluating {} samples", len(X))

        Y = np.array([
            metric_fn(model_factory(dict(zip(problem["names"], x)), self.seed + i))
            for i, x in enumerate(X)
        ])

        result = sobol.analyze(problem, Y)

        rows = []
        for j, name in enumerate(problem["names"]):
            rows.append({
                "parameter": name,
                "S1": float(result["S1"][j]),
                "ST": float(result["ST"][j]),
                "S1_conf": float(result["S1_conf"][j]),
                "ST_conf": float(result["ST_conf"][j]),
            })

        df = pd.DataFrame(rows).sort_values("S1", ascending=False, ignore_index=True)
        logger.info("sobol analysis complete, {} parameters", len(param_ranges))
        return df

    def prompt_sensitivity(
        self,
        model_factory: Callable,
        prompt_variants: list[str],
        n_runs_each: int = 5,
        metric_fn: Callable | None = None,
    ) -> pd.DataFrame:
        if metric_fn is None:
            metric_fn = _default_metrics_dict

        rows = []
        for variant in prompt_variants:
            metrics_list: list[dict] = []
            for run_idx in range(n_runs_each):
                result = model_factory(variant, self.seed + run_idx)
                metrics_list.append(metric_fn(result))

            all_keys = metrics_list[0].keys() if metrics_list else []
            for key in all_keys:
                vals = [m[key] for m in metrics_list]
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                cv = std / mean if mean != 0 else 0.0
                rows.append({
                    "prompt_variant": variant[:80],
                    "metric": key,
                    "mean": mean,
                    "std": std,
                    "cv": cv,
                })

        df = pd.DataFrame(rows)
        logger.info(
            "prompt sensitivity: {} variants x {} runs", len(prompt_variants), n_runs_each,
        )
        return df

def _default_metric(result) -> float:
    return max(sum(dr.prevalence.values()) for dr in result.days)

def _default_metrics_dict(result) -> dict[str, float]:
    prevalences = [sum(dr.prevalence.values()) for dr in result.days]
    total_inf = sum(sum(dr.new_infections.values()) for dr in result.days)
    peak_day = int(np.argmax(prevalences))
    return {
        "peak_prevalence": float(max(prevalences)),
        "peak_day": float(peak_day),
        "total_infections": float(total_inf),
    }
