from __future__ import annotations

import numpy as np
from loguru import logger
from scipy.stats import gamma as _gamma

from src.models.data_structures import EpidemicContext, Phase
from src.regime.base import BaseDetector

def _discretise_gamma_serial_interval(
    mean: float, sd: float, max_lag: int = 14,
) -> np.ndarray:
    if mean <= 0 or sd <= 0:
        raise ValueError("mean and sd must be positive")
    shape = (mean / sd) ** 2
    scale = sd * sd / mean
    w = np.zeros(max_lag)
    for s in range(1, max_lag + 1):
        lo = max(s - 0.5, 0.0)
        hi = s + 0.5
        w[s - 1] = _gamma.cdf(hi, a=shape, scale=scale) - _gamma.cdf(lo, a=shape, scale=scale)
    total = w.sum()
    if total > 0:
        w = w / total
    return w

class RelativeRtDetector(BaseDetector):
    def __init__(
        self,
        population_size: int | None = None,
        config: dict | None = None,
    ) -> None:
        cfg = config or {}
        self.si_mean: float = cfg.get("si_mean", 2.6)
        self.si_sd: float = cfg.get("si_sd", 1.5)
        self.si_max_lag: int = cfg.get("si_max_lag", 14)
        self.tau: int = int(cfg.get("tau", 5))
        if self.tau < 1:
            self.tau = 1
        self.alpha: float = cfg.get("alpha", 0.7)
        self.warmup_days: int = int(cfg.get("warmup_days", 5))
        self.confirm_days: int = int(cfg.get("confirm_days", 2))
        self.min_prevalence: float = cfg.get("min_prevalence", 0.02)
        self.absolute_fallback: float = cfg.get("absolute_fallback", 1.0)

        self.population_size = population_size
        self._w = _discretise_gamma_serial_interval(
            self.si_mean, self.si_sd, self.si_max_lag,
        )

        self._incidence: list[float] = []
        self._rt_history: list[float] = []
        self._warmup_rt: list[float] = []
        self._plateau: float | None = None
        self._below_streak = 0
        self._switch_day: int | None = None
        self._saw_epidemic = False
        self._switch_rt_history: list[float] = []
        self._switch_ratio_history: list[float] = []
        self._switch_streak_history: list[int] = []

    def feed_incidence(self, new_cases: float) -> None:
        self._incidence.append(float(max(new_cases, 0.0)))

    def _lambda_at(self, s: int) -> float:
        lam = 0.0
        max_k = min(self.si_max_lag, s)
        for k in range(1, max_k + 1):
            lam += self._incidence[s - k] * self._w[k - 1]
        return lam

    def _current_rt(self) -> float:
        if not self._incidence:
            return float("nan")

        t = len(self._incidence) - 1
        win_start = max(t - self.tau + 1, 0)

        numerator = float(sum(self._incidence[win_start:t + 1]))
        denom = 0.0
        for s in range(win_start, t + 1):
            denom += self._lambda_at(s)

        if denom <= 1e-9 or numerator <= 0:
            return float("nan")
        return float(numerator / denom)

    def detect(self, context: EpidemicContext) -> Phase:
        if len(self._incidence) < context.day:
            self.feed_incidence(context.new_infections_today)

        pop = self.population_size or max(context.total_infected * 20, 1)
        prevalence = context.total_infected / pop if pop > 0 else 0.0

        if not self._saw_epidemic and prevalence < self.min_prevalence:
            return Phase.BASELINE

        self._saw_epidemic = True
        rt = self._current_rt()
        self._rt_history.append(rt)

        if np.isnan(rt):
            return Phase.GROWTH if prevalence > 0 else Phase.BASELINE
        if rt > 1.2:
            return Phase.GROWTH
        if rt >= 0.95:
            return Phase.PEAK
        return Phase.DECLINE

    def should_switch(self, context: EpidemicContext) -> bool:
        if self._switch_day is not None:
            return context.day >= self._switch_day

        if len(self._incidence) < context.day:
            self.feed_incidence(context.new_infections_today)

        pop = self.population_size or max(context.total_infected * 20, 1)
        prevalence = context.total_infected / pop if pop > 0 else 0.0

        if prevalence < self.min_prevalence:
            self._below_streak = 0
            self._switch_rt_history.append(float("nan"))
            self._switch_ratio_history.append(float("nan"))
            self._switch_streak_history.append(self._below_streak)
            return False

        rt = self._current_rt()
        self._switch_rt_history.append(rt)
        if np.isnan(rt):
            self._below_streak = 0
            self._switch_ratio_history.append(float("nan"))
            self._switch_streak_history.append(self._below_streak)
            return False

        if self._plateau is None:
            self._warmup_rt.append(rt)
            self._switch_ratio_history.append(float("nan"))
            self._switch_streak_history.append(0)
            if len(self._warmup_rt) >= self.warmup_days:
                self._plateau = float(max(self._warmup_rt))
                logger.info(
                    "relative R_t plateau established at {:.3f} (day {}, warmup R_t = {})",
                    self._plateau, context.day,
                    ["{:.2f}".format(r) for r in self._warmup_rt],
                )
            return False

        ratio = rt / self._plateau
        self._switch_ratio_history.append(ratio)

        triggered = (ratio < self.alpha) or (rt < self.absolute_fallback)
        if triggered:
            self._below_streak += 1
        else:
            self._below_streak = 0
        self._switch_streak_history.append(self._below_streak)

        if self._below_streak >= self.confirm_days:
            self._switch_day = context.day
            logger.info(
                "relative R_t switch at day {} (R_t={:.3f}, plateau={:.3f}, ratio={:.2f})",
                context.day, rt, self._plateau, ratio,
            )
            return True
        return False

    def rt_history(self) -> list[float]:
        return list(self._rt_history)
