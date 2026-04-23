from __future__ import annotations

import time
from collections.abc import Callable

import pandas as pd
from loguru import logger

from src.calibration.abc import ABCSampler
from src.calibration.history_matching import HistoryMatcher
from src.models.data_structures import PosteriorResult

class CalibrationPipeline:
    def __init__(self, config: dict) -> None:
        self.hm_waves: int = config.get("hm_waves", 3)
        self.abc_method: str = config.get("abc_method", "smc")
        self.n_particles: int = config.get("abc_n_particles", 1000)
        self._hm = HistoryMatcher()
        self._abc = ABCSampler()

    def run(
        self,
        run_model: Callable,
        target_data: pd.Series,
        param_ranges: dict[str, tuple[float, float]],
        n_initial_samples: int = 500,
    ) -> PosteriorResult:
        start = time.perf_counter()
        logger.info(
            "starting calibration: {} hm waves, {} abc",
            self.hm_waves, self.abc_method,
        )
        samples = self._hm.generate_samples(param_ranges, n_initial_samples)
        for wave in range(self.hm_waves):
            if len(samples) == 0:
                logger.warning("hm wave {} produced empty space, stopping", wave)
                break
            threshold = float(samples.apply(
                lambda row: self._quick_distance(run_model, row, target_data),
                axis=1,
            ).quantile(0.3))
            samples = self._hm.wave_reduce(run_model, samples, target_data, threshold)
            logger.info("after hm wave {}: {} samples remain", wave, len(samples))

        if len(samples) == 0:
            logger.warning("no samples survived history matching")
            return PosteriorResult(
                parameters={},
                posterior_samples=pd.DataFrame(),
                metrics={"hm_waves": self.hm_waves, "survived": 0},
                n_model_runs=0,
                runtime_seconds=time.perf_counter() - start,
            )

        result = self._abc.smc(
            run_model=run_model,
            target=target_data,
            prior=samples,
            n_particles=min(self.n_particles, len(samples) * 2),
        )
        result.runtime_seconds = time.perf_counter() - start
        return result

    def _quick_distance(
        self, run_model: Callable, row: pd.Series, target: pd.Series,
    ) -> float:
        sim = run_model(row.to_dict())
        return self._abc._mse(sim, target)
