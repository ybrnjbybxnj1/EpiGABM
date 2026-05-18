from __future__ import annotations
import numpy as np
import pytest
from src.models.data_structures import EpidemicContext, Phase
from src.regime.relative_rt_detector import (
    RelativeRtDetector,
    _discretise_gamma_serial_interval,
)

def test_serial_interval_normalized():
    w = _discretise_gamma_serial_interval(mean=2.6, sd=1.5, max_lag=14)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= 0).all()

def test_serial_interval_peaks_near_mean():
    w = _discretise_gamma_serial_interval(mean=2.6, sd=1.5, max_lag=14)
    assert int(np.argmax(w)) + 1 <= 3

def test_serial_interval_rejects_invalid():
    with pytest.raises(ValueError):
        _discretise_gamma_serial_interval(mean=0.0, sd=1.5)
    with pytest.raises(ValueError):
        _discretise_gamma_serial_interval(mean=2.6, sd=-1.0)

def test_detector_stays_baseline_below_threshold():
    det = RelativeRtDetector(population_size=1000)
    for day in range(1, 10):
        ctx = EpidemicContext(
            day=day, total_infected=5,
            total_susceptible=995, total_recovered=0,
            growth_rate=0.0, new_infections_today=1,
            phase=None,
        )
        assert det.detect(ctx) == Phase.BASELINE
        assert det.should_switch(ctx) is False

def test_detector_switches_on_decay_after_warmup():
    det = RelativeRtDetector(
        population_size=1000,
        config={
            "min_prevalence": 0.05,
            "alpha": 0.7,
            "warmup_days": 5,
            "confirm_days": 2,
            "tau": 5,
        },
    )
    growth = [int(2 * 1.6 ** (d / 2)) for d in range(12)]
    decay = [max(1, int(growth[-1] * (0.6 ** (d / 3)))) for d in range(1, 20)]
    incidence = growth + decay
    total = 0
    switch_day = None
    for day, inc in enumerate(incidence, start=1):
        total += inc
        ctx = EpidemicContext(
            day=day, total_infected=total,
            total_susceptible=1000 - total, total_recovered=0,
            growth_rate=0.0, new_infections_today=inc,
            phase=None,
        )
        if det.should_switch(ctx):
            switch_day = day
            break
    assert switch_day is not None, "detector must eventually switch on decaying curve"
    assert switch_day > len(growth), "switch must land in the decay phase"

def test_detector_does_not_switch_on_noise():
    det = RelativeRtDetector(
        population_size=1000,
        config={"min_prevalence": 0.05},
    )
    rng = np.random.RandomState(0)
    for day in range(1, 30):
        inc = int(abs(rng.normal(1, 1)))
        ctx = EpidemicContext(
            day=day, total_infected=inc * 2,
            total_susceptible=1000, total_recovered=0,
            growth_rate=0.0, new_infections_today=inc,
            phase=None,
        )
        assert det.should_switch(ctx) is False

def test_detector_reports_phases():
    det = RelativeRtDetector(
        population_size=1000,
        config={"min_prevalence": 0.01},
    )
    for day in range(1, 8):
        inc = 2 ** (day // 2 + 1)
        total = sum(2 ** (d // 2 + 1) for d in range(1, day + 1))
        ctx = EpidemicContext(
            day=day, total_infected=total,
            total_susceptible=1000 - total, total_recovered=0,
            growth_rate=0.0, new_infections_today=inc,
            phase=None,
        )
        phase = det.detect(ctx)
        assert phase in {Phase.BASELINE, Phase.GROWTH, Phase.PEAK}

def test_absolute_fallback_fires_on_subcritical():
    det = RelativeRtDetector(
        population_size=1000,
        config={
            "min_prevalence": 0.01,
            "alpha": 0.999,
            "warmup_days": 3,
            "confirm_days": 1,
            "tau": 5,
            "absolute_fallback": 1.0,
        },
    )
    for day in range(1, 25):
        ctx = EpidemicContext(
            day=day, total_infected=50,
            total_susceptible=950, total_recovered=0,
            growth_rate=0.0, new_infections_today=2,
            phase=None,
        )
        det.should_switch(ctx)
    assert det._switch_day is not None

def test_switch_day_is_sticky():
    det = RelativeRtDetector(
        population_size=1000,
        config={
            "min_prevalence": 0.01,
            "alpha": 0.7,
            "warmup_days": 3,
            "confirm_days": 1,
            "tau": 3,
        },
    )
    growth = [10, 20, 40, 60, 80]
    decay = [70, 40, 20, 10, 5]
    total = 0
    for day, inc in enumerate(growth + decay, start=1):
        total += inc
        ctx = EpidemicContext(
            day=day, total_infected=total,
            total_susceptible=1000 - total, total_recovered=0,
            growth_rate=0.0, new_infections_today=inc,
            phase=None,
        )
        det.should_switch(ctx)
    assert det._switch_day is not None
    first_switch = det._switch_day
    for day in range(first_switch, first_switch + 4):
        ctx = EpidemicContext(
            day=day, total_infected=50,
            total_susceptible=950, total_recovered=0,
            growth_rate=0.0, new_infections_today=0,
            phase=None,
        )
        assert det.should_switch(ctx) is True
    assert det._switch_day == first_switch

def test_rt_history_grows_with_calls():
    det = RelativeRtDetector(
        population_size=1000, config={"min_prevalence": 0.01},
    )
    for day in range(1, 8):
        ctx = EpidemicContext(
            day=day, total_infected=100,
            total_susceptible=900, total_recovered=0,
            growth_rate=0.0, new_infections_today=10,
            phase=None,
        )
        det.detect(ctx)
    assert len(det.rt_history()) >= 1
