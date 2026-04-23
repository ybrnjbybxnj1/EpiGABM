from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from src.uncertainty.sensitivity import SensitivityAnalyzer

@pytest.fixture
def analyzer():
    return SensitivityAnalyzer(seed=42)

def test_sobol_x_dominates_y(analyzer):
    def model_fn(params: dict, seed: int = 0):
        x, y = params["x"], params["y"]
        val = 5 * x**2 + y
        return val
    def metric_fn(result):
        return result
    df = analyzer.sobol_indices(
        model_factory=model_fn,
        param_ranges={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        n_samples=256,
        metric_fn=metric_fn,
    )
    assert list(df.columns) == ["parameter", "S1", "ST", "S1_conf", "ST_conf"]
    s1_x = float(df.loc[df["parameter"] == "x", "S1"].iloc[0])
    s1_y = float(df.loc[df["parameter"] == "y", "S1"].iloc[0])
    st_x = float(df.loc[df["parameter"] == "x", "ST"].iloc[0])
    st_y = float(df.loc[df["parameter"] == "y", "ST"].iloc[0])
    assert s1_x > s1_y, f"S1(x)={s1_x:.3f} should be > S1(y)={s1_y:.3f}"
    assert st_x > st_y, f"ST(x)={st_x:.3f} should be > ST(y)={st_y:.3f}"

def _make_mock_result(peak: float = 10.0, peak_day: int = 5, total_inf: float = 50.0):
    days = []
    for d in range(10):
        prev = {
            "H1N1": peak if d == peak_day else 1.0,
        }
        new_inf = {"H1N1": total_inf / 10}
        days.append(SimpleNamespace(
            day=d, prevalence=prev, new_infections=new_inf,
        ))
    return SimpleNamespace(days=days)

def test_prompt_sensitivity_columns(analyzer):
    def factory(variant: str, seed: int):
        return _make_mock_result()
    df = analyzer.prompt_sensitivity(
        model_factory=factory,
        prompt_variants=["prompt_a", "prompt_b"],
        n_runs_each=3,
    )
    assert "prompt_variant" in df.columns
    assert "metric" in df.columns
    assert "mean" in df.columns
    assert "std" in df.columns
    assert "cv" in df.columns
    assert "metric_name" not in df.columns

def test_prompt_sensitivity_cv_zero_mean(analyzer):
    def factory(variant: str, seed: int):
        days = [SimpleNamespace(
            day=0, prevalence={"H1N1": 0.0}, new_infections={"H1N1": 0.0},
        )]
        return SimpleNamespace(days=days)
    df = analyzer.prompt_sensitivity(
        model_factory=factory,
        prompt_variants=["zero_prompt"],
        n_runs_each=2,
    )
    for _, row in df.iterrows():
        if row["mean"] == 0.0:
            assert row["cv"] == 0.0
