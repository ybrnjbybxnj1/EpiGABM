from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from src.models.data_structures import ConfidenceIntervals, SimulationResult

class BootstrapUQ:
    def __init__(self, config: dict) -> None:
        self.n_runs: int = config.get("bootstrap_runs", 100)
        self.confidence_level: float = config.get("confidence_level", 0.95)

    def run(
        self,
        run_fn: Callable,
        n_runs: int | None = None,
    ) -> ConfidenceIntervals:
        n = n_runs or self.n_runs
        all_curves: list[pd.Series] = []

        for seed in range(n):
            result = run_fn(seed)
            curve = pd.Series([
                sum(dr.prevalence.values()) for dr in result.days
            ])
            all_curves.append(curve)

        return self._compute_ci(all_curves)

    def llm_stochasticity_uq(
        self,
        run_fn: Callable,
        n_runs: int | None = None,
    ) -> ConfidenceIntervals:
        return self.run(run_fn, n_runs)

    def model_disagreement(
        self,
        results_a: SimulationResult,
        results_b: SimulationResult,
    ) -> pd.DataFrame:
        rows = []
        n_days = min(len(results_a.days), len(results_b.days))

        for i in range(n_days):
            da, db = results_a.days[i], results_b.days[i]
            prev_a = sum(da.prevalence.values())
            prev_b = sum(db.prevalence.values())
            abs_diff = abs(prev_a - prev_b)
            rel_diff = abs_diff / max(prev_a, prev_b, 1)

            rows.append({
                "day": da.day,
                "prevalence_a": prev_a,
                "prevalence_b": prev_b,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            })

        return pd.DataFrame(rows)

    def _compute_ci(self, curves: list[pd.Series]) -> ConfidenceIntervals:
        if not curves:
            return ConfidenceIntervals(
                lower=pd.Series(dtype=float),
                upper=pd.Series(dtype=float),
                median=pd.Series(dtype=float),
                level=self.confidence_level,
            )

        min_len = min(len(c) for c in curves)
        matrix = np.array([c.values[:min_len] for c in curves])

        alpha = 1 - self.confidence_level
        z = stats.norm.ppf(1 - alpha / 2)

        means = matrix.mean(axis=0)
        stds = matrix.std(axis=0)
        medians = np.median(matrix, axis=0)

        lower = means - z * stds
        upper = means + z * stds

        return ConfidenceIntervals(
            lower=pd.Series(lower),
            upper=pd.Series(upper),
            median=pd.Series(medians),
            level=self.confidence_level,
        )
