from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from src.agents.archetypes import ArchetypeManager, load_archetypes
from src.agents.base import BaseLLMBackend
from src.logging.agent_log import AgentLogger
from src.models.abm import ABMSimulation
from src.models.data_structures import SimulationResult
from src.models.gabm import GABMSimulation
from src.models.hybrid import HybridModel
from src.models.seir import SEIRModel
from src.regime.base import BaseDetector
from src.regime.hmm_detector import HMMDetector
from src.regime.relative_rt_detector import RelativeRtDetector
from src.regime.threshold import ThresholdDetector
from src.utils.data_loader import load_data

class RunOrchestrator:
    def __init__(self, config_path: str | Path = "config/default.yaml") -> None:
        config_path = Path(config_path)
        with open(config_path) as f:
            self.config: dict = yaml.safe_load(f)
        self._data: pd.DataFrame | None = None
        self._households: pd.DataFrame | None = None
        self._school_pools: dict | None = None

    def load_population(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        pop_dir = Path(self.config["data"]["population_dir"])
        self._data, self._households, self._school_pools = load_data(pop_dir)
        logger.info("loaded population: {} agents", len(self._data))
        return self._data, self._households

    def create_detector(self) -> BaseDetector:
        regime_cfg = self.config.get("regime", {})
        method = regime_cfg.get("method", "threshold")
        pop_size = len(self._data) if self._data is not None else None
        if method == "relative_rt":
            detector = RelativeRtDetector(
                population_size=pop_size,
                config=regime_cfg.get("relative_rt", {}),
            )
            logger.info("created relative R_t detector, pop={}", pop_size)
            return detector
        if method == "combined":
            from src.regime.base import make_detector
            detector = make_detector(regime_cfg, population_size=pop_size)
            logger.info("created combined detector, pop={}", pop_size)
            return detector
        if method == "hmm":
            hmm_cfg = regime_cfg.get("hmm", {})
            n_states = hmm_cfg.get("n_states", 4)
            detector = HMMDetector(n_states=n_states)
            logger.info("created HMM detector with {} states", n_states)
            return detector
        thresh_cfg = regime_cfg.get("threshold", {})
        detector = ThresholdDetector(thresh_cfg, population_size=pop_size)
        logger.info("created threshold detector, method={}, min_infected={}",
                    thresh_cfg.get("method"), detector.min_infected)
        return detector

    def run_abm(self, seed: int = 42) -> SimulationResult:
        if self._data is None:
            self.load_population()
        days_cfg = self.config["model"]["days"]
        days = range(days_cfg[0], days_cfg[1])
        sim = ABMSimulation(self.config, self._data, self._households)
        result = sim.run(days=days, seed=seed)
        self._save_result(result, "rulebased")
        return result

    def run_gabm(
        self,
        backend: BaseLLMBackend,
        seed: int = 42,
        compare_backend: BaseLLMBackend | None = None,
    ) -> SimulationResult:
        if self._data is None:
            self.load_population()
        days_cfg = self.config["model"]["days"]
        days = range(days_cfg[0], days_cfg[1])
        agents_cfg = self.config.get("agents", {})
        archetypes_path = self.config.get("data", {}).get(
            "archetypes_file", "config/archetypes.yaml",
        )
        archetypes = load_archetypes(archetypes_path)
        backend_name = getattr(backend, "model", type(backend).__name__)
        log_dir = self.config.get("output", {}).get("logs_dir", "results/logs")
        agent_logger = AgentLogger(
            run_id=f"orch_{seed}", backend_name=backend_name, output_dir=log_dir,
        )
        manager = ArchetypeManager(
            archetypes=archetypes,
            agent_logger=agent_logger,
            agents_config=agents_cfg,
            compare_backend=compare_backend,
        )
        detector = self.create_detector()
        sim = GABMSimulation(
            config=self.config,
            data=self._data,
            households=self._households,
            llm_backend=backend,
            archetype_manager=manager,
            detector=detector,
        )
        result = sim.run(days=days, seed=seed)
        self._save_result(result, backend_name)
        return result

    def run_hybrid(
        self,
        seed: int = 42,
        abm: ABMSimulation | None = None,
    ) -> SimulationResult:
        if self._data is None:
            self.load_population()
        days_cfg = self.config["model"]["days"]
        days = range(days_cfg[0], days_cfg[1])
        if abm is None:
            abm = ABMSimulation(self.config, self._data, self._households)
        seir = SEIRModel(self.config)
        detector = self.create_detector()
        hybrid = HybridModel(abm=abm, seir=seir, detector=detector)
        result = hybrid.run(days=days, seed=seed)
        self._save_result(result, "hybrid")
        return result

    def _save_result(self, result: SimulationResult, label: str) -> None:
        out_dir = Path(self.config.get("output", {}).get("metrics_dir", "results/metrics"))
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_label = label.replace(":", "_")
        path = out_dir / f"run_{result.seed}_{safe_label}.json"
        result.to_json(path)
        logger.info("saved result to {}", path)
