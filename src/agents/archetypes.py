from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.agents.base import (
    BaseLLMBackend,
    BehaviorDecision,
    query_with_fallback,
)
from src.agents.prompts import ArchetypeConfig, build_prompt
from src.logging.agent_log import AgentLogger
from src.models.data_structures import EpidemicContext

def load_archetypes(path: str | Path) -> list[ArchetypeConfig]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    archetypes = []
    for entry in raw["archetypes"]:
        archetypes.append(ArchetypeConfig(
            id=entry["id"],
            name=entry["name"],
            age_range=tuple(entry["age_range"]),
            occupation=entry["occupation"],
            health_literacy=entry["health_literacy"],
            risk_tolerance=entry["risk_tolerance"],
            family=entry["family"],
            prompt_additions=entry.get("prompt_additions", ""),
            has_llm_agency=entry.get("has_llm_agency", True),
        ))
    return archetypes

class ArchetypeManager:
    def __init__(
        self,
        archetypes: list[ArchetypeConfig],
        agent_logger: AgentLogger,
        agents_config: dict,
        compare_backend: BaseLLMBackend | None = None,
    ) -> None:
        self.archetypes = archetypes
        self.agent_logger = agent_logger
        self.agents_config = agents_config
        self.compare_backend = compare_backend
        self.cached_rules: dict[str, BehaviorDecision] = {}
        self.last_phase: str | None = None
        self._epsilon: float = agents_config.get("stochastic_noise", 0.1)
        self._archetype_map = {a.id: a for a in archetypes}
        self._no_agency_ids = frozenset(a.id for a in archetypes if not a.has_llm_agency)
        self._hh_parent_idx: dict[int, int] = {}
        self._population_size: int = 0

    def set_population_data(self, data: pd.DataFrame) -> None:
        self._population_size = len(data)
        adults = data[~data["archetype"].isin(self._no_agency_ids)]
        if adults.empty:
            return
        oldest = adults.loc[adults.groupby("sp_hh_id")["age"].idxmax()]
        self._hh_parent_idx = dict(zip(oldest["sp_hh_id"], oldest.index))

    def update_rules(
        self,
        context: EpidemicContext,
        backend: BaseLLMBackend,
        local_awareness: dict[str, dict] | None = None,
        archetype_health: dict[str, dict] | None = None,
    ) -> None:
        phase_str = context.phase or "BASELINE"
        for archetype in self.archetypes:
            if not archetype.has_llm_agency:
                continue
            arch_awareness = (
                local_awareness.get(archetype.id) if local_awareness else None
            )
            arch_health = (
                archetype_health.get(archetype.id) if archetype_health else None
            )
            prompt = build_prompt(
                archetype, context, "healthy",
                local_awareness=arch_awareness,
                archetype_health=arch_health,
            )
            decision, raw, success = query_with_fallback(backend, prompt)
            self.agent_logger.log(
                sim_day=context.day,
                phase=phase_str,
                archetype_id=archetype.id,
                prompt=prompt,
                raw_response=raw,
                parsed_decision=decision if success else None,
                parse_success=success,
                tokens_input=getattr(backend, "last_tokens_input", 0),
                tokens_output=getattr(backend, "last_tokens_output", 0),
                latency_ms=getattr(backend, "last_latency_ms", 0),
                cost_usd=backend.estimate_cost() if hasattr(backend, "estimate_cost") else 0.0,
            )
            self.cached_rules[archetype.id] = decision
            if self.compare_backend is not None:
                cmp_decision, cmp_raw, cmp_success = query_with_fallback(
                    self.compare_backend, prompt,
                )
                cmp = self.compare_backend
                self.agent_logger.log(
                    sim_day=context.day,
                    phase=phase_str,
                    archetype_id=archetype.id,
                    prompt=prompt,
                    raw_response=cmp_raw,
                    parsed_decision=cmp_decision if cmp_success else None,
                    parse_success=cmp_success,
                    tokens_input=getattr(cmp, "last_tokens_input", 0),
                    tokens_output=getattr(cmp, "last_tokens_output", 0),
                    latency_ms=getattr(cmp, "last_latency_ms", 0),
                    cost_usd=cmp.estimate_cost() if hasattr(cmp, "estimate_cost") else 0.0,
                )
        self.last_phase = phase_str
        n_queried = sum(1 for a in self.archetypes if a.has_llm_agency)
        logger.info(
            "updated rules for {} archetypes (skipped {} without agency), phase={}",
            n_queried, len(self.archetypes) - n_queried, phase_str,
        )

    def get_decision(
        self,
        agent_row: pd.Series,
        population: pd.DataFrame | None = None,
    ) -> BehaviorDecision:
        archetype_id = agent_row["archetype"]
        if archetype_id in self._no_agency_ids:
            return self._inherit_from_parent(agent_row, population)
        if archetype_id not in self.cached_rules:
            return BehaviorDecision(
                isolate=False, isolate_confidence=0.5,
                mask=False, mask_confidence=0.5,
                reduce_contacts=False, reduce_contacts_confidence=0.5,
                see_doctor=False, see_doctor_confidence=0.5,
                reasoning="no cached rules",
            )
        base = self.cached_rules[archetype_id]
        if self._epsilon == 0.0:
            decision = base
        else:
            decision = self._apply_noise(base)
        is_sick = agent_row.get("illness_day", 0) > 0
        if not is_sick:
            decision = BehaviorDecision(
                isolate=decision.isolate,
                isolate_confidence=decision.isolate_confidence,
                mask=decision.mask,
                mask_confidence=decision.mask_confidence,
                reduce_contacts=decision.reduce_contacts,
                reduce_contacts_confidence=decision.reduce_contacts_confidence,
                see_doctor=False,
                see_doctor_confidence=0.1,
                reasoning=decision.reasoning,
            )
        return decision

    def _inherit_from_parent(
        self,
        child_row: pd.Series,
        population: pd.DataFrame | None,
    ) -> BehaviorDecision:
        hh_id = child_row["sp_hh_id"]
        parent_idx = self._hh_parent_idx.get(hh_id)
        if parent_idx is not None and population is not None:
            parent_row = population.loc[parent_idx]
            return self.get_decision(parent_row, population)
        return BehaviorDecision(
            isolate=False, isolate_confidence=0.5,
            mask=False, mask_confidence=0.5,
            reduce_contacts=False, reduce_contacts_confidence=0.5,
            see_doctor=False, see_doctor_confidence=0.5,
            reasoning="child: no parent found in household",
        )

    def apply_decisions_vectorized(self, data: pd.DataFrame) -> None:
        data["will_isolate"] = False
        data["wears_mask"] = False
        data["reduces_contacts"] = False
        data["sees_doctor"] = False
        for arch_id, decision in self.cached_rules.items():
            if arch_id in self._no_agency_ids:
                continue
            mask = data["archetype"] == arch_id
            if not mask.any():
                continue
            if self._epsilon == 0.0:
                data.loc[mask, "will_isolate"] = decision.isolate
                data.loc[mask, "wears_mask"] = decision.mask
                data.loc[mask, "reduces_contacts"] = decision.reduce_contacts
                data.loc[mask, "sees_doctor"] = decision.see_doctor
            else:
                n = mask.sum()
                for action in ("isolate", "mask", "reduce_contacts", "see_doctor"):
                    col = {
                        "isolate": "will_isolate", "mask": "wears_mask",
                        "reduce_contacts": "reduces_contacts", "see_doctor": "sees_doctor",
                    }[action]
                    base_bool = getattr(decision, action)
                    base_conf = getattr(decision, f"{action}_confidence")
                    noisy_conf = np.clip(
                        base_conf + np.random.uniform(-self._epsilon, self._epsilon, n),
                        0.0, 1.0,
                    )
                    flip = np.random.random(n) > noisy_conf
                    values = np.where(flip, not base_bool, base_bool)
                    data.loc[mask, col] = values
        compliance = self.agents_config.get("compliance_rate", {})
        sick = data["illness_day"] > 0
        healthy = ~sick
        iso_healthy_cap = compliance.get("isolate_healthy", 0.25)
        healthy_isolating = healthy & (data["will_isolate"] == True)
        if healthy_isolating.any():
            n_healthy_iso = healthy_isolating.sum()
            max_healthy_iso = int(healthy.sum() * iso_healthy_cap)
            if n_healthy_iso > max_healthy_iso:
                excess = n_healthy_iso - max_healthy_iso
                to_remove = np.random.choice(
                    data.index[healthy_isolating], excess, replace=False,
                )
                data.loc[to_remove, "will_isolate"] = False
        isolating = data["will_isolate"] == True
        data.loc[isolating, "wears_mask"] = False
        data.loc[isolating, "reduces_contacts"] = False
        mask_cap = compliance.get("mask", 0.30)
        not_isolating = ~isolating
        masked_out = not_isolating & (data["wears_mask"] == True)
        if masked_out.any():
            n_masked = masked_out.sum()
            max_masked = int(not_isolating.sum() * mask_cap)
            if n_masked > max_masked:
                excess = n_masked - max_masked
                to_remove = np.random.choice(
                    data.index[masked_out], excess, replace=False,
                )
                data.loc[to_remove, "wears_mask"] = False
        contact_cap = compliance.get("reduce_contacts", 0.40)
        reducing = not_isolating & (data["reduces_contacts"] == True)
        if reducing.any():
            n_reducing = reducing.sum()
            max_reducing = int(not_isolating.sum() * contact_cap)
            if n_reducing > max_reducing:
                excess = n_reducing - max_reducing
                to_remove = np.random.choice(
                    data.index[reducing], excess, replace=False,
                )
                data.loc[to_remove, "reduces_contacts"] = False
        data.loc[healthy, "sees_doctor"] = False
        child_mask = data["archetype"].isin(self._no_agency_ids)
        if child_mask.any():
            for hh_id in data.loc[child_mask, "sp_hh_id"].unique():
                parent_idx = self._hh_parent_idx.get(hh_id)
                if parent_idx is not None and parent_idx in data.index:
                    hh_children = child_mask & (data["sp_hh_id"] == hh_id)
                    for col in ("will_isolate", "wears_mask", "reduces_contacts", "sees_doctor"):
                        data.loc[hh_children, col] = data.at[parent_idx, col]

    def _apply_noise(self, base: BehaviorDecision) -> BehaviorDecision:
        eps = self._epsilon
        fields: dict = {}
        for action in ("isolate", "mask", "reduce_contacts", "see_doctor"):
            conf_key = f"{action}_confidence"
            base_conf = getattr(base, conf_key)
            base_bool = getattr(base, action)
            noisy_conf = float(np.clip(base_conf + np.random.uniform(-eps, eps), 0.0, 1.0))
            flip_prob = 1.0 - noisy_conf
            if np.random.random() < flip_prob:
                noisy_bool = not base_bool
            else:
                noisy_bool = base_bool
            fields[action] = noisy_bool
            fields[conf_key] = noisy_conf
        fields["reasoning"] = base.reasoning
        return BehaviorDecision(**fields)

    def assign_archetypes(self, data: pd.DataFrame) -> pd.Series:
        single_arch = self.agents_config.get("ablation_single_archetype")
        if single_arch is not None:
            return pd.Series(single_arch, index=data.index, dtype=str)
        ages = data["age"]
        work_ids = data["work_id"].astype(str)
        result = pd.Series("young_active", index=data.index, dtype=str)
        result[ages < 18] = "child"
        result[ages >= 60] = "elderly"
        working = (ages >= 18) & (ages < 60)
        result[working & (ages < 36)] = "young_active"
        result[working & (ages >= 36)] = "middle_family"
        has_work = working & (work_ids != "X")
        if has_work.any():
            wid_num = pd.to_numeric(
                work_ids[has_work], errors="coerce",
            ).fillna(0).astype(int)
            mod = wid_num % 20
            hc_idx = mod[mod == 0].index
            result.loc[hc_idx] = "healthcare"
            sw_idx = mod[mod.isin([1, 2])].index
            result.loc[sw_idx] = "service_worker"
        return result

    def should_update(self, context: EpidemicContext, day: int) -> bool:
        trigger = self.agents_config.get("update_trigger", "phase_change")
        phase_str = context.phase or "BASELINE"
        if trigger == "phase_change":
            return phase_str != self.last_phase
        elif trigger == "every_n_steps":
            n = self.agents_config.get("update_every_n", 7)
            return day % n == 1 or self.last_phase is None
        return False
