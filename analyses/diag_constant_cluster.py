"""diagnostic: verify whether the constant-beta method cluster (last_value,
rolling_mean, biexponential, median_beta, regression_day, regression_day_seir,
incremental_rolling_mean) collapses to identical predictions because of:

A) bug   -- universal clipping or shared output path that forces equality;
B) feature -- mathematical degeneration on near-constant pre-switch beta
              (each method is a different functional that happens to map a
              flat input to roughly the same constant).

protocol:
1. print pre-switch beta range / std for one (backend, seed)
2. for each method, print predicted beta first 5 values, mean, std
3. cross-seed check: do same methods collapse on a different seed?
4. spot-check beta_prediction.py for global clipping
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration.beta_prediction import BetaPredictor

PKL = "results/analyses/trajectories/exp01_no_ceil_or_gpt_oss_120b.pkl"
SEEDS_TO_CHECK = [42, 43, 44]
T_SWITCH = 7
H = 30
METHODS = [
    "last_value", "rolling_mean", "expanding_mean", "biexponential",
    "median_beta", "regression_day", "regression_day_seir_prev_i",
    "incremental_last_value", "incremental_rolling_mean",
    "lstm_day_e_prev_i", "mlp_window_beta_prev_i",
]


def main() -> None:
    with open(PKL, "rb") as f:
        d = pickle.load(f)
    predictor = BetaPredictor(trained_models_dir="trained_models")

    for seed in SEEDS_TO_CHECK:
        if seed not in d["seeds"]:
            print(f"seed {seed} not in pkl, skipping")
            continue
        idx = d["seeds"].index(seed)
        beta_full = np.asarray(d["betas"][idx], dtype=float)
        i_full = np.asarray(d["compartments"][idx]["I"], dtype=float)

        print(f"\nseed {seed}, t_switch={T_SWITCH} (pre-switch beta only)")
        beta_obs_arr = beta_full[:T_SWITCH]
        i_obs_arr = i_full[:T_SWITCH]
        print(f"  beta_obs (days 0..{T_SWITCH-1}): {beta_obs_arr}")
        print(f"  beta_obs range: [{beta_obs_arr.min():.6f}, {beta_obs_arr.max():.6f}]  "
              f"std: {beta_obs_arr.std():.6e}")
        if T_SWITCH > 1:
            no_sentinel = beta_obs_arr[1:]
            print(f"  beta_obs (drop day-0 sentinel): {no_sentinel}")
            print(f"  range: [{no_sentinel.min():.6f}, {no_sentinel.max():.6f}]  "
                  f"std: {no_sentinel.std():.6e}")

        print(f"\n  per-method predicted beta (first 5 values, mean, std):")
        per_method = {}
        for m in METHODS:
            try:
                kwargs = {}
                if m in ("regression_day_seir_prev_i", "lstm_day_e_prev_i"):
                    kwargs["prev_i"] = pd.Series(i_obs_arr)
                pred = predictor.predict(m, pd.Series(beta_obs_arr), n_ahead=H, **kwargs)
                v = np.asarray(pred.values, dtype=float)
                per_method[m] = v
                first5 = " ".join(f"{x:.6f}" for x in v[:5])
                print(f"    {m:30s}  first5=[{first5}]  mean={v.mean():.6f}  std={v.std():.6e}")
            except Exception as exc:
                print(f"    {m:30s}  FAILED: {exc}")
                per_method[m] = None

        print(f"\n  pairwise method coincidence (max abs diff in first {H} predicted beta):")
        ms = [m for m in METHODS if per_method.get(m) is not None]
        groups: list[list[str]] = []
        used = set()
        for m1 in ms:
            if m1 in used:
                continue
            cluster = [m1]
            used.add(m1)
            for m2 in ms:
                if m2 in used:
                    continue
                diff = float(np.max(np.abs(per_method[m1] - per_method[m2])))
                if diff < 1e-9:
                    cluster.append(m2)
                    used.add(m2)
            groups.append(cluster)
        for g in groups:
            label = " == ".join(g) if len(g) > 1 else g[0]
            v = per_method[g[0]]
            print(f"    [{len(g)} method(s)] {label}    mean={v.mean():.6f}  std={v.std():.6e}")

    print(f"\nsource-level scan: clipping in src/calibration/beta_prediction.py")
    src = Path("src/calibration/beta_prediction.py").read_text(encoding="utf-8")
    for kw in ("np.clip(", "np.maximum(", "np.minimum(", " max(0", " max(0.0",
               "if pred", "if val"):
        cnt = src.count(kw)
        if cnt:
            print(f"  '{kw}' appears {cnt} time(s)")


if __name__ == "__main__":
    main()
