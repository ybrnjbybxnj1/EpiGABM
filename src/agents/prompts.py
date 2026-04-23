from __future__ import annotations

from dataclasses import dataclass

from src.models.data_structures import EpidemicContext

@dataclass
class ArchetypeConfig:
    id: str
    name: str
    age_range: tuple[int, int]
    occupation: str
    health_literacy: int
    risk_tolerance: int
    family: str
    prompt_additions: str
    has_llm_agency: bool = True

_DECISION_SCHEMA = """\
Reply ONLY with a valid JSON object with these fields:
{
    "isolate": true/false,
    "isolate_confidence": 0.0-1.0,
    "mask": true/false,
    "mask_confidence": 0.0-1.0,
    "reduce_contacts": true/false,
    "reduce_contacts_confidence": 0.0-1.0,
    "see_doctor": true/false,
    "see_doctor_confidence": 0.0-1.0,
    "reasoning": "your thoughts in 2-3 sentences, as a first-person diary entry describing your day and why you made these decisions"
}
No text outside JSON."""

_EPI_BACKGROUND = """\
Background: masks reduce virus transmission by roughly 50%. \
Self-isolation prevents infecting colleagues and neighbors. \
Seeing a doctor speeds up recovery (by ~2 days). \
Reducing contacts lowers both the chance of catching and spreading the virus."""

def _build_persona_block(archetype: ArchetypeConfig) -> str:
    return (
        f"You are {archetype.name}, age {archetype.age_range[0]}-{archetype.age_range[1]}, "
        f"{archetype.occupation}. {archetype.family}.\n"
        f"Health literacy: {archetype.health_literacy}/10. "
        f"Risk tolerance: {archetype.risk_tolerance}/10.\n"
        f"{archetype.prompt_additions}"
    )

def _build_context_block(
    context: EpidemicContext,
    local_awareness: dict | None = None,
) -> str:
    phase_str = context.phase or "unknown"
    measures = ", ".join(context.official_measures) if context.official_measures else "none"
    lines = [
        f"It is day {context.day} of a flu epidemic in Saint Petersburg.",
        f"Total infected: {context.total_infected}. "
        f"New cases today: {context.new_infections_today}. "
        f"Growth rate: {context.growth_rate:+.1%}.",
        f"Epidemic phase: {phase_str}.",
        f"Official measures: {measures}.",
    ]
    if local_awareness is not None:
        prev = local_awareness.get("prevalence", 0.0)
        cross = local_awareness.get("cross_group", False)
        pct = prev * 100
        if cross and pct >= 20:
            lines.append(
                f"Some of your coworkers AND neighbors are sick ({pct:.0f}% "
                "of people you interact with). The flu seems to be spreading everywhere."
            )
        elif pct >= 30:
            lines.append(
                f"Many of your coworkers are sick ({pct:.0f}% of people at work). "
                "People are talking about the epidemic."
            )
        elif pct >= 10:
            lines.append(
                f"A few of your coworkers have called in sick ({pct:.0f}%)."
            )
    return "\n".join(lines)

def _build_agent_state_block(agent_state: str, archetype_health: dict | None = None) -> str:
    state_map = {
        "здоров": "healthy",
        "заражён": "infected",
        "выздоровел": "recovered",
    }
    english_state = state_map.get(agent_state, agent_state)
    lines = [f"Your current health: {english_state}."]
    if archetype_health is not None:
        sick_pct = archetype_health.get("sick_pct", 0.0) * 100
        rec_pct = archetype_health.get("recovered_pct", 0.0) * 100
        if sick_pct >= 1.0 or rec_pct >= 1.0:
            lines.append(
                f"Among people like you: {sick_pct:.0f}% currently sick, "
                f"{rec_pct:.0f}% already recovered."
            )
    return "\n".join(lines)

def build_prompt(
    archetype: ArchetypeConfig,
    context: EpidemicContext,
    agent_state: str,
    local_awareness: dict | None = None,
    archetype_health: dict | None = None,
) -> str:
    parts = [
        _build_persona_block(archetype),
        "",
        _EPI_BACKGROUND,
        "",
        _build_context_block(context, local_awareness=local_awareness),
        "",
        _build_agent_state_block(agent_state, archetype_health=archetype_health),
        "",
        "What will you do? Will you self-isolate, wear a mask, "
        "reduce contacts, see a doctor? Think about your specific life "
        "situation: your job, family, finances, health risks.",
        "",
        _DECISION_SCHEMA,
    ]
    return "\n".join(parts)
