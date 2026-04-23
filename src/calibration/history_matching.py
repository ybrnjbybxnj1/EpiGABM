from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import uniform

class HistoryMatcher:
    def __init__(self, distance_fn: Callable | None = None) -> None:
        self._distance_fn = distance_fn or self._mse

    @staticmethod
    def _mse(sim: pd.Series, target: pd.Series) -> float:
        min_len = min(len(sim), len(target))
        return float(np.mean((sim.values[:min_len] - target.values[:min_len]) ** 2))

    def wave_reduce(
        self,
        run_model: Callable,
        param_space: pd.DataFrame,
        target: pd.Series,
        threshold: float,
    ) -> pd.DataFrame:
        distances = []
        for _, row in param_space.iterrows():
            sim = run_model(row.to_dict())
            d = self._distance_fn(sim, target)
            distances.append(d)

        param_space = param_space.copy()
        param_space["_hm_distance"] = distances
        accepted = param_space[param_space["_hm_distance"] < threshold]
        accepted = accepted.drop(columns=["_hm_distance"])

        logger.info(
            "hm wave: {}/{} accepted (threshold={:.4f})",
            len(accepted), len(param_space), threshold,
        )
        return accepted

    def generate_samples(
        self,
        param_ranges: dict[str, tuple[float, float]],
        n_samples: int,
    ) -> pd.DataFrame:
        data = {}
        for name, (lo, hi) in param_ranges.items():
            data[name] = uniform.rvs(loc=lo, scale=hi - lo, size=n_samples)
        return pd.DataFrame(data)
