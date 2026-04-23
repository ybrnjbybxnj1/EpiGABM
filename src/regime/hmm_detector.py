from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

from src.models.data_structures import EpidemicContext, Phase
from src.regime.base import BaseDetector

try:
    from hmmlearn.hmm import GaussianHMM
    _HAS_HMMLEARN = True
except ImportError:
    from sklearn.mixture import GaussianMixture
    _HAS_HMMLEARN = False
    logger.debug("hmmlearn not available, using GaussianMixture fallback")

class HMMDetector(BaseDetector):
    def __init__(
        self,
        n_states: int = 4,
        features: list[str] | None = None,
    ) -> None:
        self.n_states = n_states
        self.features = features or ["incidence", "growth_rate"]

        if _HAS_HMMLEARN:
            self._model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42,
            )
        else:
            self._model = GaussianMixture(
                n_components=n_states,
                covariance_type="full",
                max_iter=100,
                random_state=42,
            )

        self.scaler = StandardScaler()
        self._state_to_phase: dict[int, Phase] = {}
        self.fitted = False
        self._obs_buffer: list[list[float]] = []

    def fit(self, historical_data: pd.DataFrame) -> None:
        data = historical_data[self.features].dropna()
        if len(data) < self.n_states * 5:
            logger.warning("insufficient data for fit: {} rows", len(data))
            return

        X = self.scaler.fit_transform(data.values)
        self._model.fit(X)
        self._assign_phases()
        self.fitted = True
        logger.info("detector fitted on {} observations, {} states", len(X), self.n_states)

    def _assign_phases(self) -> None:
        if _HAS_HMMLEARN:
            means = self._model.means_[:, 0]
        else:
            means = self._model.means_[:, 0]

        sorted_indices = np.argsort(means)

        if self.n_states == 4:
            self._state_to_phase = {
                int(sorted_indices[0]): Phase.BASELINE,
                int(sorted_indices[1]): Phase.GROWTH,
                int(sorted_indices[3]): Phase.PEAK,
                int(sorted_indices[2]): Phase.DECLINE,
            }
        else:
            phases = [Phase.BASELINE, Phase.GROWTH, Phase.PEAK, Phase.DECLINE]
            for i, idx in enumerate(sorted_indices):
                self._state_to_phase[int(idx)] = phases[min(i, len(phases) - 1)]

    def detect(self, context: EpidemicContext) -> Phase:
        if not self.fitted:
            return Phase.BASELINE

        obs = [context.new_infections_today, context.growth_rate]
        self._obs_buffer.append(obs)

        X = self.scaler.transform(np.array(self._obs_buffer))

        if _HAS_HMMLEARN:
            states = self._model.predict(X)
            current_state = int(states[-1])
        else:
            states = self._model.predict(X)
            window = min(5, len(states))
            recent = states[-window:]
            counts: dict[int, int] = {}
            for s in recent:
                counts[int(s)] = counts.get(int(s), 0) + 1
            current_state = max(counts, key=counts.get)

        return self._state_to_phase.get(current_state, Phase.BASELINE)

    def should_switch(self, context: EpidemicContext) -> bool:
        phase = self.detect(context)
        return phase in (Phase.PEAK, Phase.DECLINE)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            "model": self._model,
            "scaler": self.scaler,
            "state_to_phase": self._state_to_phase,
            "n_states": self.n_states,
            "features": self.features,
            "fitted": self.fitted,
            "has_hmmlearn": _HAS_HMMLEARN,
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        logger.info("detector model saved to {}", path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        self._model = model_data["model"]
        self.scaler = model_data["scaler"]
        self._state_to_phase = model_data["state_to_phase"]
        self.n_states = model_data["n_states"]
        self.features = model_data["features"]
        self.fitted = model_data["fitted"]
        logger.info("detector model loaded from {}", path)
