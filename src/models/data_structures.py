from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd


class Phase(str, Enum):
    BASELINE = "BASELINE"
    GROWTH = "GROWTH"
    PEAK = "PEAK"
    DECLINE = "DECLINE"


@dataclass
class SEIRState:
    S: int
    E: int
    I: int
    R: int
    N: int


@dataclass
class PosteriorResult:
    parameters: dict[str, float]
    posterior_samples: pd.DataFrame
    metrics: dict[str, float]
    n_model_runs: int
    runtime_seconds: float


@dataclass
class ConfidenceIntervals:
    lower: pd.Series
    upper: pd.Series
    median: pd.Series
    level: float


@dataclass
class EpidemicContext:
    day: int
    total_infected: int
    total_susceptible: int
    total_recovered: int
    growth_rate: float
    new_infections_today: int
    phase: str | None
    official_measures: list[str] = field(default_factory=list)


@dataclass
class DayResult:
    day: int
    S: dict[str, int]
    E: dict[str, int]
    I: dict[str, int]
    R: dict[str, int]
    beta: dict[str, float]
    new_infections: dict[str, int]
    prevalence: dict[str, int]
    phase: str | None = None
    n_isolating: int | None = None
    n_masked: int | None = None
    n_reducing_contacts: int | None = None
    n_seeing_doctor: int | None = None
    archetype_rules_updated: bool = False


@dataclass
class SimulationResult:
    days: list[DayResult]
    config: dict
    seed: int
    backend: str | None = None
    run_id: str | None = None
    total_llm_calls: int = 0
    total_llm_cost_usd: float = 0.0
    runtime_seconds: float = 0.0

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for dr in self.days:
            row = {"day": dr.day}
            for comp in ("S", "E", "I", "R", "beta", "new_infections", "prevalence"):
                vals = getattr(dr, comp)
                for strain, val in vals.items():
                    row[f"{comp}_{strain}"] = val
            row["phase"] = dr.phase
            row["n_isolating"] = dr.n_isolating
            row["n_masked"] = dr.n_masked
            row["n_reducing_contacts"] = dr.n_reducing_contacts
            row["n_seeing_doctor"] = dr.n_seeing_doctor
            rows.append(row)
        return pd.DataFrame(rows)


@dataclass
class AgentSnapshot:
    sp_id: int
    sp_hh_id: int
    archetype: str
    state: str
    illness_day: int
    will_isolate: bool
    wears_mask: bool
    reduces_contacts: bool
    sees_doctor: bool
    work_id: str = "X"
    narrative: str = ""


@dataclass
class DaySnapshot:
    day: int
    phase: str
    agents: list[AgentSnapshot]
