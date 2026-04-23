from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.data_structures import EpidemicContext, Phase

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, context: EpidemicContext) -> Phase:
        ...

    @abstractmethod
    def should_switch(self, context: EpidemicContext) -> bool:
        ...

def make_detector(
    regime_cfg: dict,
    population_size: int | None = None,
) -> "BaseDetector":
    method = regime_cfg.get("method", "threshold")

    if method == "relative_rt":
        from src.regime.relative_rt_detector import RelativeRtDetector
        return RelativeRtDetector(
            population_size=population_size,
            config=regime_cfg.get("relative_rt", {}),
        )
    if method == "combined":
        from src.regime.combined_detector import CombinedDetector
        from src.regime.relative_rt_detector import RelativeRtDetector
        from src.regime.threshold import ThresholdDetector
        rel = RelativeRtDetector(
            population_size=population_size,
            config=regime_cfg.get("relative_rt", {}),
        )
        thr = ThresholdDetector(
            regime_cfg.get("threshold", {}),
            population_size=population_size,
        )
        return CombinedDetector([rel, thr])
    if method == "hmm":
        from src.regime.hmm_detector import HMMDetector
        hmm_cfg = regime_cfg.get("hmm", {})
        return HMMDetector(n_states=hmm_cfg.get("n_states", 4))
    from src.regime.threshold import ThresholdDetector
    return ThresholdDetector(
        regime_cfg.get("threshold", {}),
        population_size=population_size,
    )
