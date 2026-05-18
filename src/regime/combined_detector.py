from __future__ import annotations

from loguru import logger

from src.models.data_structures import EpidemicContext, Phase
from src.regime.base import BaseDetector

class CombinedDetector(BaseDetector):
    def __init__(self, detectors: list[BaseDetector]) -> None:
        if not detectors:
            raise ValueError("CombinedDetector requires at least one detector")
        self.detectors = detectors
        self._switch_day: int | None = None
        self._winner: str | None = None

    def detect(self, context: EpidemicContext) -> Phase:
        primary_phase = self.detectors[0].detect(context)
        for det in self.detectors[1:]:
            det.detect(context)
        return primary_phase

    def should_switch(self, context: EpidemicContext) -> bool:
        if self._switch_day is not None:
            return context.day >= self._switch_day

        triggered = False
        for det in self.detectors:
            if det.should_switch(context) and not triggered:
                triggered = True
                self._winner = type(det).__name__
                self._switch_day = context.day
                logger.info(
                    "combined detector: switch at day {} (winner={})",
                    context.day, self._winner,
                )
        return triggered

    @property
    def winner(self) -> str | None:
        return self._winner
