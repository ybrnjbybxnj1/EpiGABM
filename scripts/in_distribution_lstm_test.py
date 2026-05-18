"""in-distribution baseline test for LSTM and MLP beta-predictors.

generates 10 fresh ABM no-behaviour trajectories on seeds 9000-9009
(disjoint from training seeds 1000-1099 and headline-sweep seeds 42-51).
runs LSTM and MLP at t_obs=12, h=14 on each trajectory, computes RMSE
of predicted beta against the ground-truth beta(t_obs+1 .. t_obs+H).

result tells whether LSTM/MLP works on data DRAWN FROM THE SAME
DISTRIBUTION as their training set. compared with the GABM RMSE numbers
(LSTM 179.6, MLP 171.6 median across 9 backends), this isolates the
distribution-shift hypothesis from architectural / sample-size hypotheses.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).parent.parent))

from analyses.helpers import (
    extract_beta_series, extract_compartment_series,
    load_config, make_small_population,
)
from src.calibration.beta_prediction import BetaPredictor
from src.models.abm import ABMSimulation


T_OBS = 12
H = 14


def _abm_trajectory(config: dict, seed: int, n_agents: int = 3000) -> dict:
    data, households = make_small_population(n=n_agents, seed=seed)
    abm = ABMSimulation(
        config=config, data=data.copy(), households=households.copy(),
    )
    days = range(1, config["model"]["days"][1])
    np.random.seed(seed)
    import random; random.seed(seed)
    result = abm.run(days=days, seed=seed)
    return {
        "beta": np.asarray(extract_beta_series(result), dtype=float),
        "I": np.asarray(extract_compartment_series(result)["I"], dtype=float),
    }


def _post_switch_seir_rmse(beta_pred: np.ndarray, beta_true: np.ndarray) -> float:
    """RMSE of predicted beta against ground-truth beta over horizon h."""
    n = min(len(beta_pred), len(beta_true))
    if n == 0:
        return float("nan")
    diff = beta_pred[:n] - beta_true[:n]
    n_pop = 3000.0
    return float(np.sqrt(np.mean((diff * n_pop) ** 2)))


def main() -> None:
    config = load_config("config/default.yaml")
    bp = BetaPredictor()

    seeds = list(range(9000, 9010))
    print(f"\nIn-distribution LSTM/MLP test, {len(seeds)} fresh ABM no-behaviour seeds\n")
    print(f"{'seed':>5}  {'len(beta)':>10}  {'LSTM_RMSE':>10}  {'MLP_RMSE':>10}")

    lstm_rmses, mlp_rmses = [], []
    for s in seeds:
        try:
            tr = _abm_trajectory(config, seed=s)
        except Exception as exc:
            print(f"{s:>5}  failed: {exc}")
            continue
        beta = tr["beta"]
        i_arr = tr["I"]
        if len(beta) < T_OBS + H:
            print(f"{s:>5}  trajectory too short ({len(beta)} days)")
            continue
        beta_obs = pd.Series(beta[:T_OBS])
        i_obs = pd.Series(i_arr[:T_OBS])
        beta_true_future = beta[T_OBS:T_OBS + H]

        lstm_pred = bp.predict("lstm_day_e_prev_i", beta_obs, n_ahead=H, prev_i=i_obs).values
        mlp_pred  = bp.predict("mlp_window_beta_prev_i", beta_obs, n_ahead=H, prev_i=i_obs).values

        lstm_rmse = _post_switch_seir_rmse(lstm_pred, beta_true_future)
        mlp_rmse  = _post_switch_seir_rmse(mlp_pred,  beta_true_future)
        lstm_rmses.append(lstm_rmse)
        mlp_rmses.append(mlp_rmse)
        print(f"{s:>5}  {len(beta):>10}  {lstm_rmse:>10.2f}  {mlp_rmse:>10.2f}")

    if lstm_rmses:
        print(f"  in-dist LSTM (seeds 9000-9009)  median: {np.median(lstm_rmses):.2f}  mean: {np.mean(lstm_rmses):.2f}")
        print(f"  in-dist MLP  (seeds 9000-9009)  median: {np.median(mlp_rmses):.2f}  mean: {np.mean(mlp_rmses):.2f}")
        print()
        print("   (compare with GABM: LSTM median 179.6, MLP median 171.6 across 9 backends)")
        print()


if __name__ == "__main__":
    main()
