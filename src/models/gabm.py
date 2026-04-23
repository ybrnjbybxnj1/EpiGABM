from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from src.agents.archetypes import ArchetypeManager
from src.agents.base import BaseLLMBackend
from src.models.abm import ABMSimulation
from src.models.data_structures import (
    AgentSnapshot,
    DayResult,
    DaySnapshot,
    EpidemicContext,
    SimulationResult,
)
from src.regime.base import BaseDetector


class GABMSimulation(ABMSimulation):
    def __init__(
        self,
        config: dict,
        data: pd.DataFrame,
        households: pd.DataFrame,
        llm_backend: BaseLLMBackend,
        archetype_manager: ArchetypeManager,
        detector: BaseDetector | None = None,
    ) -> None:
        super().__init__(config, data, households)
        self.llm_backend = llm_backend
        self.archetype_manager = archetype_manager
        self.detector = detector if detector is not None else self._create_detector(
            config, population_size=len(data),
        )
        self._agents_config = config.get("agents", {})
        self._mask_factor: float = self._agents_config.get("mask_reduction_factor", 0.5)
        self._contact_reduction: float = self._agents_config.get("contact_reduction_factor", 0.7)
        self._doctor_day: int = self._agents_config.get("doctor_recovery_day", 6)
        self._snapshot_every: int = config.get(
            "visualization", {},
        ).get("snapshot_every_n_days", 1)
        self._snapshots: list[DaySnapshot] = []
        self._run_id: str = ""
        self.data["archetype"] = self.archetype_manager.assign_archetypes(self.data)
        self.data["will_isolate"] = False
        self.data["wears_mask"] = False
        self.data["reduces_contacts"] = False
        self.data["sees_doctor"] = False
        self.archetype_manager.set_population_data(self.data)
        logger.info(
            "gabm columns added, {} unique archetypes assigned",
            self.data["archetype"].nunique(),
        )

    @staticmethod
    def _create_detector(config: dict, population_size: int | None = None) -> BaseDetector | None:
        regime_cfg = config.get("regime")
        if regime_cfg is None:
            return None
        method = regime_cfg.get("method", "threshold")
        if method == "relative_rt":
            from src.regime.relative_rt_detector import RelativeRtDetector
            return RelativeRtDetector(
                population_size=population_size,
                config=regime_cfg.get("relative_rt", {}),
            )
        if method == "combined":
            from src.regime.base import make_detector
            return make_detector(regime_cfg, population_size=population_size)
        if method == "hmm":
            from src.regime.hmm_detector import HMMDetector
            hmm_cfg = regime_cfg.get("hmm", {})
            return HMMDetector(n_states=hmm_cfg.get("n_states", 4))
        from src.regime.threshold import ThresholdDetector
        thresh_cfg = regime_cfg.get("threshold", {})
        return ThresholdDetector(thresh_cfg, population_size=population_size)

    def set_initial_conditions(self) -> None:
        super().set_initial_conditions()

    def behavior_step(self, day: int, context: EpidemicContext) -> None:
        self._rules_updated_this_day = False
        if self.detector is not None:
            detected = self.detector.detect(context)
            context.phase = detected.value if hasattr(detected, "value") else str(detected)
        old_phase = self.archetype_manager.last_phase
        phase_str = context.phase or "BASELINE"
        if old_phase is not None and phase_str != old_phase:
            self.archetype_manager.agent_logger.log_phase_change(day, old_phase, phase_str)
        activation = self._agents_config.get("behavior_activation", {})
        min_prevalence = activation.get("min_prevalence", 0.05)
        pop_size = len(self.data)
        prevalence = context.total_infected / pop_size if pop_size > 0 else 0.0
        behavior_active = getattr(self, "_behavior_active", False)
        if not behavior_active:
            if prevalence >= min_prevalence:
                self._behavior_active = True
                behavior_active = True
                self.archetype_manager.last_phase = None
        elif context.total_infected == 0:
            self._behavior_active = False
            behavior_active = False
        if not behavior_active:
            self.data["will_isolate"] = False
            self.data["wears_mask"] = False
            self.data["reduces_contacts"] = False
            self.data["sees_doctor"] = False
            self.archetype_manager.last_phase = phase_str
            prev_isolating = getattr(self, "_isolating_set", set())
            if prev_isolating:
                self._restore_agents_to_pools(prev_isolating)
                self._isolating_set = set()
            return
        if self.archetype_manager.should_update(context, day):
            include_local = activation.get("include_local_awareness", True)
            local_awareness = self._compute_local_awareness() if include_local else None
            archetype_health = self._compute_archetype_health()
            self.archetype_manager.update_rules(
                context, self.llm_backend,
                local_awareness=local_awareness,
                archetype_health=archetype_health,
            )
            self._rules_updated_this_day = True
        data = self.data
        self.archetype_manager.apply_decisions_vectorized(data)
        currently_isolating = set(data[data.will_isolate].index)
        prev_isolating = getattr(self, "_isolating_set", set())
        stopped_isolating = prev_isolating - currently_isolating
        self._restore_agents_to_pools(stopped_isolating)
        newly_isolating = currently_isolating - prev_isolating
        self._remove_agents_from_pools(newly_isolating)
        self._isolating_set = currently_isolating

    def _remove_agents_from_pools(self, agent_indices: set[int]) -> None:
        data = self.data
        for idx in agent_indices:
            row = data.loc[idx]
            wid = row.work_id
            if wid == "X":
                continue
            for key in self.strains:
                if row.age > 17:
                    wid_int = int(wid)
                    if (
                        wid_int in self.dict_work_id.get(key, {})
                        and idx in self.dict_work_id[key][wid_int]
                    ):
                        self.dict_work_id[key][wid_int].remove(idx)
                else:
                    wid_str = str(wid)
                    if (
                        wid_str in self.dict_school_id.get(key, {})
                        and idx in self.dict_school_id[key][wid_str]
                    ):
                        self.dict_school_id[key][wid_str].remove(idx)

    def _restore_agents_to_pools(self, agent_indices: set[int]) -> None:
        data = self.data
        for idx in agent_indices:
            row = data.loc[idx]
            wid = row.work_id
            if wid == "X":
                continue
            for key in self.strains:
                if row["susceptible_" + key] == 0:
                    continue
                if row.age > 17:
                    wid_int = int(wid)
                    pool = self.dict_work_id.get(key, {}).get(wid_int, None)
                    if pool is not None and idx not in pool:
                        pool.append(idx)
                else:
                    wid_str = str(wid)
                    pool = self.dict_school_id.get(key, {}).get(wid_str, None)
                    if pool is not None and idx not in pool:
                        pool.append(idx)

    def _compute_local_awareness(self) -> dict[str, dict]:
        data = self.data
        result = {}
        for archetype in self.archetype_manager.archetypes:
            if not archetype.has_llm_agency:
                continue
            arch_agents = data[data["archetype"] == archetype.id]
            if arch_agents.empty:
                result[archetype.id] = {"prevalence": 0.0, "cross_group": False}
                continue
            work_ids = arch_agents[arch_agents["work_id"] != "X"]["work_id"].unique()
            wp_prev = 0.0
            if len(work_ids) > 0:
                coworkers = data[(data["work_id"].isin(work_ids)) & (data["work_id"] != "X")]
                if len(coworkers) > 0:
                    wp_prev = float((coworkers["infected"] > 0).sum()) / len(coworkers)
            hh_ids = arch_agents["sp_hh_id"].unique()
            hh_members = data[data["sp_hh_id"].isin(hh_ids)]
            hh_prev = 0.0
            if len(hh_members) > 0:
                hh_prev = float((hh_members["infected"] > 0).sum()) / len(hh_members)
            cross_group = wp_prev > 0 and hh_prev > 0
            combined = max(wp_prev, hh_prev)
            if cross_group:
                combined = (wp_prev + hh_prev) / 2
            result[archetype.id] = {
                "prevalence": combined,
                "cross_group": cross_group,
            }
        return result

    def _compute_archetype_health(self) -> dict[str, dict]:
        if self._agents_config.get("disable_archetype_health", False):
            return None
        data = self.data
        result = {}
        for archetype in self.archetype_manager.archetypes:
            if not archetype.has_llm_agency:
                continue
            arch_agents = data[data["archetype"] == archetype.id]
            n = len(arch_agents)
            if n == 0:
                result[archetype.id] = {"sick_pct": 0.0, "recovered_pct": 0.0}
                continue
            sick = int((arch_agents["infected"] > 0).sum())
            recovered_mask = (arch_agents["infected"] == 0)
            for key in self.strains:
                recovered_mask &= (arch_agents["susceptible_" + key] == 0)
            recovered = int(recovered_mask.sum())
            result[archetype.id] = {
                "sick_pct": sick / n,
                "recovered_pct": recovered / n,
            }
        return result

    def get_effective_lmbd(self, agent_idx: int, context: str = "household") -> float:
        rate = self.lmbd
        row = self.data.loc[agent_idx]
        if context != "household" and row["wears_mask"]:
            rate *= self._mask_factor
        if context != "household" and row["reduces_contacts"]:
            rate *= self._contact_reduction
        return rate

    def recover(self) -> None:
        data = self.data
        data.loc[data.infected > 0, "illness_day"] += 1
        doctor_recovered = (
            (data.illness_day >= self._doctor_day)
            & (data.sees_doctor)
            & (data.infected > 0)
        )
        standard_recovered = data.illness_day > 8
        recovered_mask = doctor_recovered | standard_recovered
        for key in self.strains:
            data.loc[recovered_mask, "susceptible_" + key] = 0
        data.loc[recovered_mask, ["infected", "illness_day"]] = 0

    def _compute_agent_state(self, idx: int) -> str:
        row = self.data.loc[idx]
        if row.infected > 0:
            if row.illness_day <= 2:
                return "E"
            return "I"
        if all(row["susceptible_" + key] == 0 for key in self.strains):
            return "R"
        return "S"

    def _build_snapshot(self, day: int, phase: str) -> DaySnapshot:
        data = self.data
        agents = []
        for idx in data.index:
            row = data.loc[idx]
            state = self._compute_agent_state(idx)
            illness_day = int(row.illness_day)
            narrative = self._generate_narrative(
                state, illness_day, idx, day, row,
            )
            agents.append(AgentSnapshot(
                sp_id=int(row.sp_id),
                sp_hh_id=int(row.sp_hh_id),
                archetype=str(row.archetype),
                state=state,
                illness_day=illness_day,
                will_isolate=bool(row.will_isolate),
                wears_mask=bool(row.wears_mask),
                reduces_contacts=bool(row.reduces_contacts),
                sees_doctor=bool(row.sees_doctor),
                work_id=str(row.work_id),
                narrative=narrative,
            ))
        return DaySnapshot(day=day, phase=phase or "BASELINE", agents=agents)

    def _generate_narrative(
        self, state: str, illness_day: int, idx: int, day: int, row,
    ) -> str:
        prev_state = None
        if self._snapshots:
            prev_snap = self._snapshots[-1]
            for a in prev_snap.agents:
                if a.sp_id == int(row.sp_id):
                    prev_state = a.state
                    break
        archetype = str(row.archetype)
        parts = []
        days_sick = min(illness_day, day)
        if prev_state == "S" and state == "E":
            parts.append(
                "Something feels off today. I might have been exposed - "
                "someone at work was coughing all day yesterday."
            )
        elif prev_state == "E" and state == "I":
            parts.append(
                "Woke up with a terrible headache and chills. "
                "Definitely coming down with the flu."
            )
        elif prev_state in ("I", "E") and state == "R":
            parts.append(
                "Finally feeling like myself again. "
                "The fever broke overnight and I can breathe normally."
            )
        elif state == "E":
            parts.append("Feeling a bit off. Might be coming down with something.")
        elif state == "I":
            if days_sick <= 1:
                parts.append(
                    "Woke up feeling terrible. Fever, chills, body aches."
                )
            elif days_sick <= 3:
                parts.append(
                    f"Day {days_sick} sick. Still rough but the fever is lower. "
                    "Managed to eat something today."
                )
            else:
                parts.append(
                    f"Day {days_sick} sick. Getting better slowly. "
                    "Hoping to be back on my feet soon."
                )
        elif state == "S":
            n_infected = int((self.data["infected"] > 0).sum())
            pop_size = len(self.data)
            prevalence = n_infected / pop_size if pop_size > 0 else 0
            if prevalence >= 0.05:
                parts.append(
                    "Heard about the flu outbreak in the news. "
                    "A lot of people seem to be getting sick."
                )
            elif prevalence >= 0.01:
                parts.append("Noticed a few people coughing at work. Hopefully nothing serious.")
            else:
                parts.append("Normal day. Nothing unusual.")
        elif state == "R":
            if prev_state in ("I", "E"):
                parts.append("Finally feeling like myself again. Back to my routine.")
            elif prev_state == "R" or prev_state is None:
                parts.append("Normal day. Nothing unusual.")
            else:
                parts.append("Feeling fine. Going about my day.")
        behaviors = []
        if row.will_isolate:
            behaviors.append("staying home")
        if row.wears_mask:
            behaviors.append("wearing a mask")
        if row.reduces_contacts:
            behaviors.append("keeping my distance from people")
        if row.sees_doctor:
            behaviors.append("going to see a doctor")
        if behaviors:
            parts.append("Decided to: " + ", ".join(behaviors) + ".")
        return " ".join(parts)

    def _save_snapshot(self, snapshot: DaySnapshot) -> None:
        out_dir = Path(self.config.get("output", {}).get(
            "snapshots_dir", "results/snapshots/",
        ))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"run_{self._run_id}_day{snapshot.day}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(snapshot), f, ensure_ascii=False)

    def run_day(self, day: int) -> DayResult:
        day_result = super().run_day(day)
        day_result.phase = self.archetype_manager.last_phase or "BASELINE"
        data = self.data
        day_result.n_isolating = int(data.will_isolate.sum())
        day_result.n_masked = int(data.wears_mask.sum())
        day_result.n_reducing_contacts = int(data.reduces_contacts.sum())
        day_result.n_seeing_doctor = int(data.sees_doctor.sum())
        day_result.archetype_rules_updated = getattr(
            self, "_rules_updated_this_day", False,
        )
        if day % self._snapshot_every == 0:
            phase = self.archetype_manager.last_phase or "BASELINE"
            snapshot = self._build_snapshot(day, phase)
            self._snapshots.append(snapshot)
            if self._run_id:
                self._save_snapshot(snapshot)
        return day_result

    def run(self, days: range, seed: int) -> SimulationResult:
        self._run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._snapshots = []
        result = super().run(days, seed)
        result.backend = getattr(self.llm_backend, "model", type(self.llm_backend).__name__)
        result.run_id = self._run_id
        result.total_llm_calls = self.archetype_manager.agent_logger.get_total_calls()
        result.total_llm_cost_usd = self.archetype_manager.agent_logger.get_total_cost()
        return result
