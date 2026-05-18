"""diagnostic step 2: replicate exact SEIR integration from exp01 and
observe where the 7-method 'constant-beta' cluster forms.

protocol:
1. per method: raw prediction -> SEIR clip np.clip(., 0, 6*beta_crit) ->
   actual beta_used by integrator
2. per method: full I(t) trajectory over 14 days
3. pairwise diff in I(t) -- which methods give literally identical curves
4. grep src/ for non-trivial clipping/abs operations on beta or I

reproduces the exact pipeline of `exp01_beta_predictor_sweep._evaluate_method`
so the output reflects what really happens in the headline sweep.
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import odeint

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration.beta_prediction import BetaPredictor

PKL = "results/analyses/trajectories/exp01_no_ceil_or_gpt_oss_120b.pkl"
SEED = 42
T_SWITCH = 7
H = 14
SIGMA = 0.2
GAMMA = 0.14
BETA_CRIT_FACTOR = 6.0
METHODS = [
    "last_value", "rolling_mean", "expanding_mean", "biexponential",
    "median_beta", "regression_day", "regression_day_seir_prev_i",
    "incremental_last_value", "incremental_rolling_mean",
    "lstm_day_e_prev_i", "mlp_window_beta_prev_i",
]


def _seir_forward(initial: tuple, beta_seq: np.ndarray, horizon: int) -> np.ndarray:
    def rhs(y, t, betas):
        idx = int(np.clip(t, 0, len(betas) - 1))
        b = betas[idx]
        s, e, i, r = y
        return [-b * s * i, b * s * i - SIGMA * e, SIGMA * e - GAMMA * i, GAMMA * i]
    t_grid = np.arange(horizon + 1, dtype=float)
    sol = odeint(rhs, list(initial), t_grid, args=(beta_seq,), full_output=False, mxstep=5000)
    return sol[1:horizon + 1, 2]


def main() -> None:
    with open(PKL, "rb") as f:
        d = pickle.load(f)
    idx = d["seeds"].index(SEED)
    beta_full = np.asarray(d["betas"][idx], dtype=float)
    comp = d["compartments"][idx]
    i_arr = np.asarray(comp["I"], dtype=float)
    s_arr = np.asarray(comp["S"], dtype=float)
    e_arr = np.asarray(comp["E"], dtype=float)
    r_arr = np.asarray(comp["R"], dtype=float)
    n_pop = int(d["n_agents"])
    beta_crit = GAMMA / n_pop
    beta_max = BETA_CRIT_FACTOR * beta_crit
    print(f"n_pop={n_pop}, gamma={GAMMA}, beta_crit={beta_crit:.6e}, "
          f"beta_max={beta_max:.6e}")

    beta_obs = pd.Series(beta_full[:T_SWITCH])
    i_obs = pd.Series(i_arr[:T_SWITCH])
    initial = (
        float(s_arr[T_SWITCH - 1]), float(e_arr[T_SWITCH - 1]),
        float(i_arr[T_SWITCH - 1]), float(r_arr[T_SWITCH - 1]),
    )

    predictor = BetaPredictor(trained_models_dir="trained_models")

    raw_preds: dict = {}
    clipped_preds: dict = {}
    i_trajectories: dict = {}

    print(f"\nseed {SEED}, t_switch={T_SWITCH}, horizon={H}")
    print(f"{'method':30s}  {'raw mean':>12s}  {'raw [min,max]':>22s}  "
          f"{'clipped mean':>14s}  clipping outcome")

    for m in METHODS:
        try:
            kwargs = {}
            if m in ("regression_day_seir_prev_i", "lstm_day_e_prev_i"):
                kwargs["prev_i"] = i_obs
            raw = np.asarray(predictor.predict(m, beta_obs, n_ahead=H, **kwargs).values,
                             dtype=float)
            clipped = np.clip(np.nan_to_num(raw, nan=0.0), 0.0, beta_max)
            raw_preds[m] = raw
            clipped_preds[m] = clipped

            n_clipped_high = int((raw > beta_max).sum())
            n_clipped_low = int((raw < 0).sum())
            outcome = ""
            if n_clipped_high > 0 and n_clipped_low > 0:
                outcome = f"BOTH ends clipped: {n_clipped_low} low, {n_clipped_high} high"
            elif n_clipped_high > 0:
                outcome = f"clipped to UPPER ({n_clipped_high}/{H})"
            elif n_clipped_low > 0:
                outcome = f"clipped to ZERO ({n_clipped_low}/{H})"
            else:
                outcome = "passes through"
            print(f"{m:30s}  {raw.mean():>12.6f}  "
                  f"[{raw.min():>9.4f},{raw.max():>9.4f}]  "
                  f"{clipped.mean():>14.6e}  {outcome}")

            i_trajectories[m] = _seir_forward(initial, clipped, H)
        except Exception as exc:
            print(f"{m:30s}  FAILED: {exc}")

    print(f"\nidentical-I(t) equivalence classes (max abs diff < 1e-6)")
    used = set()
    classes = []
    for m1 in METHODS:
        if m1 in used or m1 not in i_trajectories:
            continue
        cluster = [m1]; used.add(m1)
        for m2 in METHODS:
            if m2 in used or m2 not in i_trajectories:
                continue
            if np.max(np.abs(i_trajectories[m1] - i_trajectories[m2])) < 1e-6:
                cluster.append(m2); used.add(m2)
        classes.append(cluster)
    for c in classes:
        i_vec = i_trajectories[c[0]]
        print(f"  [{len(c)}]  {' == '.join(c)}")
        print(f"        I(t) = {[f'{x:.1f}' for x in i_vec[:14]]}")

    print(f"\nsrc/-wide grep: clipping/abs/min-max operations")
    src_dirs = ["src/calibration", "src/models", "src/regime", "analyses"]
    patterns = [
        (r"np\.clip\([^)]*beta", "np.clip on beta"),
        (r"np\.maximum\([^)]*0[^)]*,\s*beta", "np.maximum(0, beta)"),
        (r"np\.minimum\([^)]*beta", "np.minimum on beta"),
        (r"\babs\([^)]*beta", "abs(beta)"),
        (r"np\.abs\([^)]*beta", "np.abs(beta)"),
        (r"max\(\s*0[^,]*,\s*beta", "max(0, beta)"),
        (r"if\s+beta\s*<", "if beta <"),
        (r"if\s+\w+\s*<\s*0", "if x < 0 (potential beta filter)"),
    ]
    for d in src_dirs:
        if not Path(d).exists():
            continue
        for f in Path(d).rglob("*.py"):
            text = f.read_text(encoding="utf-8", errors="ignore")
            for pat, label in patterns:
                hits = list(re.finditer(pat, text, flags=re.IGNORECASE))
                for h in hits:
                    line_no = text.count("\n", 0, h.start()) + 1
                    line = text.split("\n")[line_no - 1].strip()
                    print(f"  {f}:{line_no}  [{label}]  {line[:120]}")


if __name__ == "__main__":
    main()
