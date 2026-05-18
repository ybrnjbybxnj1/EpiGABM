# non-parametric bootstrap-percentile CIs on epidemic trajectories (Efron & Tibshirani 1993)
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.models.data_structures import ConfidenceIntervals, SimulationResult

DEFAULT_N_BOOT_RESAMPLES = 1000


class BootstrapUQ:
    def __init__(self, config: dict) -> None:
        self.n_runs: int = config.get("bootstrap_runs", 100)
        self.n_boot_resamples: int = config.get(
            "bootstrap_resamples", DEFAULT_N_BOOT_RESAMPLES,
        )
        self.confidence_level: float = config.get("confidence_level", 0.95)
        self.random_state: int | None = config.get("bootstrap_seed", 0)

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
        matrix = np.array([c.values[:min_len] for c in curves], dtype=float)
        n_curves = matrix.shape[0]

        rng = np.random.default_rng(self.random_state)
        boot_means = np.empty(
            (self.n_boot_resamples, min_len), dtype=float,
        )
        for b in range(self.n_boot_resamples):
            idx = rng.integers(0, n_curves, size=n_curves)
            boot_means[b, :] = matrix[idx, :].mean(axis=0)

        alpha = 1 - self.confidence_level
        lower = np.percentile(boot_means, 100 * (alpha / 2), axis=0)
        upper = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)
        medians = np.median(matrix, axis=0)

        return ConfidenceIntervals(
            lower=pd.Series(lower),
            upper=pd.Series(upper),
            median=pd.Series(medians),
            level=self.confidence_level,
        )
