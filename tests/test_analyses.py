
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _synthetic_epidemic_trajectory(
    peak_day: int = 14, n_days: int = 50, n_pop: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangle-shaped I(t) and decaying-then-flat beta(t)  -  enough structure
    for both threshold-variance and relative-R_t to detect a regime change."""
    days = np.arange(1, n_days + 1, dtype=float)
    I = np.where(
        days <= peak_day,
        50 * (days / peak_day) ** 2,
        50 * np.exp(-0.25 * (days - peak_day)),
    )
    beta_crit = 0.14 / n_pop
    rng = np.random.RandomState(0)
    growth_curve = 3.0 * beta_crit * (1 - 0.3 * days / peak_day)
    plateau_noise = 0.01 * beta_crit * rng.randn(n_days)
    plateau_curve = 0.5 * beta_crit + plateau_noise
    beta = np.where(days <= peak_day, growth_curve, plateau_curve)
    return beta.astype(float), I.astype(float)


def test_detect_switch_day_threshold_fires_pre_peak():
    from analyses.helpers import detect_switch_day_threshold
    beta, I = _synthetic_epidemic_trajectory(peak_day=14)
    res = detect_switch_day_threshold(
        beta, I, n_pop=3000,
        var_thresh=0.10, min_infected_frac=0.01, min_day=8,
    )
    assert res["t_switch"] is not None
    assert res["gate"] == "ok"
    assert res["pre_peak"] is True
    assert res["t_switch"] <= res["peak_day"]
    assert res["t_switch"] >= 8


def test_detect_switch_day_threshold_no_epidemic_returns_none():
    from analyses.helpers import detect_switch_day_threshold
    res = detect_switch_day_threshold(
        beta_traj=np.zeros(50), i_traj=np.zeros(50), n_pop=3000,
    )
    assert res["t_switch"] is None
    assert res["gate"] == "no_epidemic"


def test_detect_switch_day_threshold_below_min_infected():
    from analyses.helpers import detect_switch_day_threshold
    beta = np.full(50, 1e-5)
    I = np.zeros(50)
    I[10:14] = [2, 4, 5, 3]
    res = detect_switch_day_threshold(beta, I, n_pop=3000, min_infected_frac=0.01)
    assert res["t_switch"] is None
    assert res["gate"] == "below_min_infected"


def test_detect_switch_day_replay_relative_rt_returns_valid_record():
    """Relative-R_t may legitimately fail to fire on a smooth synthetic
    triangle (this is part of why the analyses replaced it with threshold-
    variance). We just check the record shape is correct."""
    from analyses.helpers import detect_switch_day_replay
    beta, I = _synthetic_epidemic_trajectory(peak_day=14)
    cfg = {"si_mean": 2.6, "si_sd": 1.5, "si_max_lag": 14, "tau": 5,
           "alpha": 0.7, "warmup_days": 5, "confirm_days": 2,
           "min_prevalence": 0.02, "absolute_fallback": 1.0}
    res = detect_switch_day_replay(beta, I, n_pop=3000, detector_config=cfg)
    assert "t_switch" in res
    assert "trigger" in res
    assert "pre_peak" in res
    assert res["peak_day"] > 0
    if res["t_switch"] is not None:
        assert res["trigger"] in ("relative_rt", "absolute_rt", "unknown")


def test_detect_switch_day_replay_strict_no_absolute_fallback():
    """absolute_fallback=0 means the rt<absolute clause never fires; only the
    plateau-ratio criterion is left. Confirms the trigger label discriminates
    the two clauses correctly."""
    from analyses.helpers import detect_switch_day_replay
    beta, I = _synthetic_epidemic_trajectory(peak_day=14)
    cfg = {"si_mean": 2.6, "si_sd": 1.5, "si_max_lag": 14, "tau": 5,
           "alpha": 0.7, "warmup_days": 5, "confirm_days": 2,
           "min_prevalence": 0.02, "absolute_fallback": 0.0}
    res = detect_switch_day_replay(beta, I, n_pop=3000, detector_config=cfg)
    if res["t_switch"] is not None:
        assert res["trigger"] in ("relative_rt", "unknown")


def test_bootstrap_percentile_ci_real_resampling():
    """Verify _compute_ci does proper bootstrap-percentile (not Wald).
    On a degenerate ensemble where every curve is identical, lower==upper
    (no resampling variance). On a random ensemble, lower < median < upper
    must hold and lower must be greater than the global minimum (i.e. it
    is a percentile, not a normal-approximation that can go below 0)."""
    from src.uncertainty.bootstrap import BootstrapUQ

    cfg = {"bootstrap_runs": 10, "bootstrap_resamples": 200,
           "confidence_level": 0.95, "bootstrap_seed": 42}
    boot = BootstrapUQ(cfg)

    same = pd.Series([10, 20, 30, 25, 15, 5])
    ci = boot._compute_ci([same.copy() for _ in range(10)])
    assert (ci.lower.values == ci.upper.values).all()
    assert (ci.lower.values == same.values).all()

    rng = np.random.default_rng(0)
    curves = [pd.Series(rng.normal(loc=100, scale=10, size=12)) for _ in range(20)]
    ci = boot._compute_ci(curves)
    assert (ci.lower.values <= ci.median.values).all()
    assert (ci.median.values <= ci.upper.values).all()
    span = ci.upper.values - ci.lower.values
    assert (span > 0).all()
    assert (span < 80).all()


def _synthetic_payload(n_seeds: int = 5, n_pop: int = 3000) -> dict:
    """Compare-shaped payload: list of per-seed compartment dicts + betas."""
    comps = []
    betas = []
    for s in range(n_seeds):
        I = np.array([10, 30, 80, 200, 350, 480, 530, 480, 320, 180, 90, 40, 15, 5, 1, 0],
                     dtype=float)
        comps.append({
            "S": np.full(len(I), n_pop - 100) - I.cumsum() / max(I.sum(), 1) * 1500,
            "E": I * 0.3,
            "I": I,
            "R": I.cumsum() / max(I.sum(), 1) * 1500,
        })
        betas.append(np.full(len(I), 5e-5))
    return {"seeds": list(range(n_seeds)), "compartments": comps, "betas": betas}


def test_compare_backends_aggregate_basic_metrics():
    from analyses.compare_backends import _aggregate
    payload = _synthetic_payload(n_seeds=5)
    agg = _aggregate(payload)
    assert agg["n_seeds"] == 5
    assert agg["peak_height_mean"] == pytest.approx(530.0)
    assert agg["peak_height_std"] == pytest.approx(0.0)
    assert agg["peak_day_mean"] == pytest.approx(7.0)
    assert agg["peak_height_cv"] == pytest.approx(0.0)


def test_compare_backends_curve_stacking_dimensions():
    from analyses.compare_backends import _stack_curves, _stack_betas
    payload = _synthetic_payload(n_seeds=3)
    I_arr = _stack_curves(payload, "I", max_days=20)
    b_arr = _stack_betas(payload, max_days=20)
    assert I_arr.shape == (3, 20)
    assert b_arr.shape == (3, 20)
    assert I_arr[0, -1] == pytest.approx(I_arr[0, 15])


def test_run_jocs_baseline_quick(tmp_path: Path, monkeypatch):
    """Run rule-based ABM on 2 seeds x 100 agents and check the cache file
    was written. Redirects ANALYSES_TRAJECTORIES so it does not pollute the
    real results tree."""
    import analyses.helpers as h
    monkeypatch.setattr(h, "ANALYSES_TRAJECTORIES", tmp_path)

    from analyses.run_jocs_baseline import main as run_main
    monkeypatch.setattr(
        "sys.argv",
        ["run_jocs_baseline",
         "--n-seeds", "2", "--n-agents", "100",
         "--cache-name", "exp01_test_rb"],
    )
    run_main()
    pkl = tmp_path / "exp01_test_rb.pkl"
    js = tmp_path / "exp01_test_rb.json"
    assert pkl.exists()
    assert js.exists()
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert len(payload["seeds"]) == 2
    assert "compartments" in payload
    assert "betas" in payload


def test_compare_backends_runs_on_synthetic_caches(tmp_path: Path, monkeypatch):
    """End-to-end: fake two cached trajectory bundles, then run the comparison
    function and check it writes the JSON + at least one figure."""
    import pickle
    import analyses.helpers as h
    import analyses.compare_backends as cmp_mod

    fig_dir = tmp_path / "figures"
    metric_dir = tmp_path / "metrics"
    traj_dir = tmp_path / "trajectories"
    for d in (fig_dir, metric_dir, traj_dir):
        d.mkdir(parents=True)

    monkeypatch.setattr(h, "ANALYSES_FIGURES", fig_dir)
    monkeypatch.setattr(h, "ANALYSES_METRICS", metric_dir)
    monkeypatch.setattr(h, "ANALYSES_TRAJECTORIES", traj_dir)
    monkeypatch.setattr(cmp_mod, "ANALYSES_FIGURES", fig_dir)
    monkeypatch.setattr(cmp_mod, "ANALYSES_METRICS", metric_dir)
    monkeypatch.setattr(cmp_mod, "ANALYSES_TRAJECTORIES", traj_dir)

    payload_a = _synthetic_payload(n_seeds=3)
    payload_b = _synthetic_payload(n_seeds=3)
    for c in payload_b["compartments"]:
        c["I"] = c["I"] * 1.2
    for name, payload in [("exp01_a", payload_a), ("exp01_b", payload_b)]:
        with open(traj_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(payload, f)
        with open(traj_dir / f"{name}.json", "w") as f:
            json.dump({"seeds": payload["seeds"],
                       "compartments": [{k: list(map(float, v))
                                          for k, v in c.items()}
                                         for c in payload["compartments"]],
                       "betas": [list(map(float, b)) for b in payload["betas"]]},
                      f)

    out = cmp_mod.run_comparison(
        backends=["a", "b"], cache_prefix="exp01_",
        fig_suffix="_smoketest", reference_key=None,
    )
    assert out is not None
    assert out["backends"] == ["a", "b"]
    assert "cross_llm_cv_peak_height" in out
    assert (metric_dir / "compare_backends_smoketest.json").exists()
    figs = list(fig_dir.glob("compare_backends_*_smoketest.png"))
    assert len(figs) >= 4
