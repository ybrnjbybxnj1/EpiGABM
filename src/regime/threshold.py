from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

from src.models.data_structures import EpidemicContext, Phase
from src.regime.base import BaseDetector

def _cpoint_roll_var(beta_series: pd.Series, thresh: float = 0.05) -> int:
    scaler = MinMaxScaler()
    var_vals = beta_series.rolling(7).var()
    scaled = scaler.fit_transform(var_vals.values.reshape(-1, 1))
    candidates = np.where(scaled < thresh)[0]
    if len(candidates) == 0:
        return len(beta_series) - 1
    cpoint = int(np.nanmin(candidates))
    return max(cpoint, 14)

def _cpoint_roll_var_seq(beta_series: pd.Series, thresh: float = 0.05) -> int:
    scaler = MinMaxScaler()
    var_vals = beta_series.rolling(7).var()
    scaled = scaler.fit_transform(var_vals.values.reshape(-1, 1))
    ids = np.where(scaled < thresh)[0]
    if len(ids) == 0:
        return len(beta_series) - 1
    splits = np.split(ids, np.where(np.diff(ids) != 1)[0] + 1)
    lengths = np.array([len(s) for s in splits])
    long_enough = np.where(lengths >= 2)[0]
    if len(long_enough) == 0:
        return len(beta_series) - 1
    cpoint = int(splits[long_enough[0]][0])
    return max(cpoint, 14)

def _cpoint_roll_var_npeople(
    beta_series: pd.Series,
    infected_series: pd.Series,
    thresh: float = 0.1,
    n_people: int = 100,
) -> int:
    scaler = MinMaxScaler()
    var_vals = beta_series.rolling(7).var()
    scaled = scaler.fit_transform(var_vals.values.reshape(-1, 1))

    above_n = infected_series[infected_series >= n_people]
    if len(above_n) == 0:
        return len(beta_series) - 1
    day_with_npeople = above_n.index[0]

    candidates = np.where(scaled[day_with_npeople:] < thresh)[0]
    if len(candidates) == 0:
        return len(beta_series) - 1
    cpoint = int(np.nanmin(candidates)) + day_with_npeople
    return max(cpoint, 14)

class ThresholdDetector(BaseDetector):
    def __init__(self, config: dict, population_size: int | None = None) -> None:
        self.method: str = config.get("method", "roll_var_npeople")
        min_infected_frac: float = config.get("min_infected_frac", 0.05)

        if population_size is not None:
            self.min_infected = max(5, int(population_size * min_infected_frac))
        else:
            self.min_infected = config.get("min_infected", 50)

        self._beta_history: list[float] = []
        self._infected_history: list[int] = []
        self._switch_day: int | None = None
        self._saw_epidemic: bool = False

    def detect(self, context: EpidemicContext) -> Phase:
        if context.total_infected < self.min_infected:
            if self._saw_epidemic and context.total_infected > 0:
                return Phase.DECLINE
            return Phase.BASELINE

        self._saw_epidemic = True

        if context.growth_rate > 0.1:
            return Phase.GROWTH
        elif context.growth_rate < -0.05:
            return Phase.DECLINE
        elif context.total_infected > 0 and abs(context.growth_rate) <= 0.1:
            return Phase.PEAK
        return Phase.BASELINE

    def feed_beta(self, beta: float) -> None:
        self._beta_history.append(beta)

    def should_switch(self, context: EpidemicContext) -> bool:
        if len(self._beta_history) <= len(self._infected_history):
            self._beta_history.append(context.growth_rate)
        self._infected_history.append(context.total_infected)

        if self._switch_day is not None:
            return context.day >= self._switch_day

        if len(self._beta_history) < 15:
            return False

        beta_s = pd.Series(self._beta_history)
        inf_s = pd.Series(self._infected_history)

        if self.method == "roll_var":
            cpoint = _cpoint_roll_var(beta_s)
        elif self.method == "roll_var_seq":
            cpoint = _cpoint_roll_var_seq(beta_s)
        elif self.method == "roll_var_npeople":
            cpoint = _cpoint_roll_var_npeople(
                beta_s, inf_s, n_people=self.min_infected,
            )
        else:
            return False

        if cpoint < len(self._beta_history) - 1:
            self._switch_day = context.day
            logger.info("threshold switch triggered at day {}", context.day)
            return True
        return False
