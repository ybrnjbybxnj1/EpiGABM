"""bootstrap percentile CI on per-seed NRMSE at h=14, t_obs=12.

addresses Step 4 of the methodological-hole plan: every entry in the
in-dist vs GABM NRMSE comparison table needs a 95% percentile CI from
the per-seed RMSE distribution before any "method X beats Y" claim
can be honest.

NRMSE = RMSE / peak_I  (per seed: divide by that seed's peak infectious count
in the corresponding trajectory). this normalizes away the ~1.7x difference in
peak height between in-distribution (JoCS no-behaviour, median peak ~874)
and GABM headline (median ~525 across 9 backends).

bootstrap protocol: B=1000 resamples with replacement from the per-seed
NRMSE pool; statistic = median; CI = 2.5/97.5 percentile.

inputs:
- in-dist: results/analyses/metrics/exp01_beta_predictor_sweep_in_dist_jocs_2026-05-07T132101.json
           results/analyses/trajectories/exp01_jocs_baseline.pkl
- GABM: 9 backend metric JSONs (timestamps 2026-05-07T122623..130528)
        9 trajectory pkls (exp01_<backend>.pkl)

output: prints markdown table + saves to docs/bootstrap_nrmse_ci.md
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np

H = "14"
T_OBS = "t_obs=12"
B = 1000
SEED = 42

BACKENDS = [
    "llama", "or_mistral24b", "or_qwen235b", "or_claude_haiku",
    "or_deepseek_v32", "or_gemma12b", "or_glm32b", "or_gpt_oss_20b",
    "or_gpt_oss_120b",
]

GABM_JSONS = {
    "llama":           "results/analyses/metrics/exp01_beta_predictor_sweep_llama_2026-05-07T122623.json",
    "or_mistral24b":   "results/analyses/metrics/exp01_beta_predictor_sweep_or_mistral24b_2026-05-07T123855.json",
    "or_qwen235b":     "results/analyses/metrics/exp01_beta_predictor_sweep_or_qwen235b_2026-05-07T124336.json",
    "or_claude_haiku": "results/analyses/metrics/exp01_beta_predictor_sweep_or_claude_haiku_2026-05-07T124738.json",
    "or_deepseek_v32": "results/analyses/metrics/exp01_beta_predictor_sweep_or_deepseek_v32_2026-05-07T125117.json",
    "or_gemma12b":     "results/analyses/metrics/exp01_beta_predictor_sweep_or_gemma12b_2026-05-07T125453.json",
    "or_glm32b":       "results/analyses/metrics/exp01_beta_predictor_sweep_or_glm32b_2026-05-07T125830.json",
    "or_gpt_oss_20b":  "results/analyses/metrics/exp01_beta_predictor_sweep_or_gpt_oss_20b_2026-05-07T130202.json",
    "or_gpt_oss_120b": "results/analyses/metrics/exp01_beta_predictor_sweep_or_gpt_oss_120b_2026-05-07T130528.json",
}

INDIST_JSON = "results/analyses/metrics/exp01_beta_predictor_sweep_in_dist_jocs_2026-05-07T132101.json"
INDIST_PKL = "results/analyses/trajectories/exp01_jocs_baseline.pkl"

METHODS = [
    "expanding_mean",
    "incremental_rolling_mean",
    "lstm_day_e_prev_i",
    "mlp_window_beta_prev_i",
    "last_value",
    "rolling_mean",
]


def _peaks_from_pkl(path: str) -> list[float]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return [float(np.max(c["I"])) for c in d["compartments"]]


def _per_seed_rmse(json_path: str, method: str) -> list[float]:
    with open(json_path, "r") as f:
        d = json.load(f)
    return list(map(float, d["per_seed_raw"][T_OBS][method]["rmse"][H]))


def _bootstrap_ci(values: np.ndarray, statistic, b: int = B, seed: int = SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(values))
    samples = np.empty(b, dtype=float)
    for k in range(b):
        idx = rng.integers(0, n, size=n)
        samples[k] = statistic(values[idx])
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> None:
    indist_peaks = _peaks_from_pkl(INDIST_PKL)
    gabm_peaks = {b: _peaks_from_pkl(f"results/analyses/trajectories/exp01_{b}.pkl") for b in BACKENDS}

    rows = []
    for method in METHODS:
        rmse_id = np.asarray(_per_seed_rmse(INDIST_JSON, method))
        nrmse_id = rmse_id / np.asarray(indist_peaks)
        med_id, lo_id, hi_id = _bootstrap_ci(nrmse_id, np.median)

        nrmse_g_pool = []
        for bk in BACKENDS:
            rmse_g = _per_seed_rmse(GABM_JSONS[bk], method)
            peaks = gabm_peaks[bk]
            nrmse_g_pool.extend(np.asarray(rmse_g) / np.asarray(peaks))
        nrmse_g_pool = np.asarray(nrmse_g_pool)
        med_g, lo_g, hi_g = _bootstrap_ci(nrmse_g_pool, np.median)

        rel_delta = (med_g - med_id) / med_id * 100.0

        rows.append((method, med_id, lo_id, hi_id, med_g, lo_g, hi_g, rel_delta))

    hdr = (f"\nbootstrap NRMSE 95% percentile CI (B={B}), h={H} day, t_obs=12\n"
           f"in-dist: 10 seeds (JoCS no-behaviour); GABM: 90 (=9 backends x 10 seeds)\n"
           f"NRMSE = per-seed RMSE / per-seed peak_I\n")
    print(hdr)
    print(f"{'method':28s}  {'in-dist NRMSE [95% CI]':30s}  {'GABM NRMSE [95% CI]':30s}  rel delta")
    for method, med_id, lo_id, hi_id, med_g, lo_g, hi_g, rel in rows:
        a = f"{med_id:.3f} [{lo_id:.3f}, {hi_id:.3f}]"
        b_str = f"{med_g:.3f} [{lo_g:.3f}, {hi_g:.3f}]"
        print(f"{method:28s}  {a:30s}  {b_str:30s}  {rel:+.0f}%")
    print()

    out_path = Path("docs/bootstrap_nrmse_ci.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# bootstrap NRMSE 95% CI  -  in-dist vs GABM\n\n")
        f.write(f"- bootstrap iterations B = {B}; statistic = median; CI = 2.5/97.5 percentile\n")
        f.write(f"- horizon h = {H} day; observation window t_obs = 12\n")
        f.write("- NRMSE = per-seed RMSE / per-seed peak infectious count\n")
        f.write("- in-dist pool size: 10 seeds (JoCS no-behaviour, exp01_jocs_baseline.pkl)\n")
        f.write("- GABM pool size: 90 = 9 backends x 10 seeds (headline sweep 2026-05-07)\n\n")
        f.write("| method | in-dist NRMSE [95% CI] | GABM NRMSE [95% CI] | rel delta (median) |\n")
        f.write("|---|---|---|---|\n")
        for method, med_id, lo_id, hi_id, med_g, lo_g, hi_g, rel in rows:
            f.write(f"| `{method}` | {med_id:.3f} [{lo_id:.3f}, {hi_id:.3f}] | "
                    f"{med_g:.3f} [{lo_g:.3f}, {hi_g:.3f}] | {rel:+.0f}% |\n")
        f.write("\n## interpretation\n\n")
        f.write("a CI overlap between in-dist and GABM means the apparent RMSE difference is "
                "not statistically distinguishable at the seed level after normalization. "
                "non-overlapping CIs license a directional claim (method X is genuinely "
                "worse/better on GABM than on its training distribution).\n")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
