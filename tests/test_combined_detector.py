from __future__ import annotations
import pytest
from src.models.data_structures import EpidemicContext, Phase
from src.regime.base import BaseDetector
from src.regime.combined_detector import CombinedDetector

class _ManualDetector(BaseDetector):
    def __init__(self, fire_day: int | None = None, phase: Phase = Phase.GROWTH) -> None:
        self.fire_day = fire_day
        self.phase = phase
        self._fired = False
    def detect(self, context: EpidemicContext) -> Phase:
        return self.phase
    def should_switch(self, context: EpidemicContext) -> bool:
        if self.fire_day is None:
            return False
        if context.day >= self.fire_day:
            self._fired = True
            return True
        return False

def _ctx(day: int, total: int = 100) -> EpidemicContext:
    return EpidemicContext(
        day=day, total_infected=total, total_susceptible=0,
        total_recovered=0, growth_rate=0.0,
        new_infections_today=0, phase=None,
    )

def test_empty_list_rejected():
    with pytest.raises(ValueError):
        CombinedDetector([])

def test_primary_detector_owns_detect():
    primary = _ManualDetector(phase=Phase.GROWTH)
    secondary = _ManualDetector(phase=Phase.DECLINE)
    combined = CombinedDetector([primary, secondary])
    assert combined.detect(_ctx(1)) == Phase.GROWTH

def test_earlier_detector_wins():
    early = _ManualDetector(fire_day=10)
    late = _ManualDetector(fire_day=20)
    combined = CombinedDetector([early, late])
    for day in range(1, 15):
        combined.should_switch(_ctx(day))
    assert combined._switch_day == 10
    assert combined.winner == "_ManualDetector"

def test_order_matters_for_primary_only():
    early = _ManualDetector(fire_day=5, phase=Phase.PEAK)
    late = _ManualDetector(fire_day=15, phase=Phase.GROWTH)
    combined = CombinedDetector([late, early])
    assert combined.detect(_ctx(1)) == Phase.GROWTH
    for day in range(1, 10):
        combined.should_switch(_ctx(day))
    assert combined._switch_day == 5

def test_sticky_after_switch():
    det = _ManualDetector(fire_day=3)
    combined = CombinedDetector([det])
    for day in range(1, 4):
        combined.should_switch(_ctx(day))
    first = combined._switch_day
    assert first == 3
    for day in range(4, 10):
        assert combined.should_switch(_ctx(day)) is True
    assert combined._switch_day == first

def test_no_detector_fires():
    a = _ManualDetector(fire_day=None)
    b = _ManualDetector(fire_day=None)
    combined = CombinedDetector([a, b])
    for day in range(1, 30):
        assert combined.should_switch(_ctx(day)) is False
    assert combined._switch_day is None
    assert combined.winner is None
