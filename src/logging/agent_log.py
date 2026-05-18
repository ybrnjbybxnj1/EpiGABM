from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.agents.base import BehaviorDecision

@dataclass
class LLMCallLog:
    timestamp: str
    sim_day: int
    phase: str
    archetype_id: str
    backend: str
    prompt: str
    raw_response: str
    parsed_decision: BehaviorDecision | None
    parse_success: bool
    tokens_input: int
    tokens_output: int
    latency_ms: int
    cost_usd: float

class AgentLogger:
    def __init__(self, run_id: str, backend_name: str, output_dir: str = "results/logs") -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        safe_name = backend_name.replace(":", "_")
        self._path = self._dir / f"run_{run_id}_{safe_name}.jsonl"
        self._backend_name = backend_name
        self._count = 0
        self._total_cost: float = 0.0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def count(self) -> int:
        return self._count

    def log(
        self,
        sim_day: int,
        phase: str,
        archetype_id: str,
        prompt: str,
        raw_response: str,
        parsed_decision: BehaviorDecision | None,
        parse_success: bool,
        tokens_input: int = 0,
        tokens_output: int = 0,
        latency_ms: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        entry = LLMCallLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sim_day=sim_day,
            phase=phase,
            archetype_id=archetype_id,
            backend=self._backend_name,
            prompt=prompt,
            raw_response=raw_response,
            parsed_decision=parsed_decision,
            parse_success=parse_success,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )
        record = asdict(entry)
        if entry.parsed_decision is not None:
            record["parsed_decision"] = entry.parsed_decision.model_dump()
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._count += 1
        self._total_cost += cost_usd
        logger.trace(
            "logged call: day={}, archetype={}, success={}",
            sim_day, archetype_id, parse_success,
        )

    def log_phase_change(self, day: int, old_phase: str, new_phase: str) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sim_day": day,
            "event": "phase_change",
            "old_phase": old_phase,
            "new_phase": new_phase,
        }
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.trace("phase change logged: day={}, {} -> {}", day, old_phase, new_phase)

    def get_total_cost(self) -> float:
        return self._total_cost

    def get_total_calls(self) -> int:
        return self._count
