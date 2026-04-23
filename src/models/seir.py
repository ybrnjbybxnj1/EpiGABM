
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.models.data_structures import SEIRState

def _seir_one_day_det(state: SEIRState, beta: float, sigma: float, gamma: float) -> SEIRState:
    s, e, i = state.S, state.E, state.I
    n = state.N
    ds = -beta * s * i
    de = beta * s * i - sigma * e
    di = sigma * e - gamma * i
    new_s = max(int(round(s + ds)), 0)
    new_e = max(int(round(e + de)), 0)
    new_i = max(int(round(i + di)), 0)
    new_r = n - new_s - new_e - new_i
    return SEIRState(S=new_s, E=new_e, I=new_i, R=new_r, N=n)

def _seir_one_day_stoch(state: SEIRState, beta: float, sigma: float, gamma: float) -> SEIRState:
    s, e, i, r = state.S, state.E, state.I, state.R
    n = state.N
    inf_prob = min(abs(beta * i), 1.0)
    new_infections = np.random.binomial(s, inf_prob) if s > 0 else 0
    new_infectious = np.random.binomial(e, sigma) if e > 0 else 0
    new_recovered = np.random.binomial(i, gamma) if i > 0 else 0
    new_s = s - new_infections
    new_e = e + new_infections - new_infectious
    new_i = i + new_infectious - new_recovered
    new_r = n - new_s - new_e - new_i
    return SEIRState(S=new_s, E=new_e, I=new_i, R=new_r, N=n)

class SEIRModel:
    def __init__(self, config: dict) -> None:
        seir_cfg = config["model"]["seir"]
        self.sigma: float = seir_cfg["sigma"]
        self.gamma: float = seir_cfg["gamma"]

    def run(
        self,
        days: range,
        initial: SEIRState,
        beta: float | np.ndarray | None = None,
        stochastic: bool = False,
    ) -> list[SEIRState]:
        if beta is None:
            beta = 0.3
        results = [initial]
        step_fn = _seir_one_day_stoch if stochastic else _seir_one_day_det
        beta_is_array = hasattr(beta, "__len__")
        for t_idx, _ in enumerate(days):
            if t_idx == 0:
                continue
            beta_t = beta[t_idx - 1] if beta_is_array else beta
            state = step_fn(results[-1], beta_t, self.sigma, self.gamma)
            results.append(state)
        logger.debug("seir run: {} days, stochastic={}", len(results), stochastic)
        return results

    def get_curve(self, states: list[SEIRState]) -> pd.DataFrame:
        rows = [{"S": s.S, "E": s.E, "I": s.I, "R": s.R, "N": s.N} for s in states]
        return pd.DataFrame(rows)
