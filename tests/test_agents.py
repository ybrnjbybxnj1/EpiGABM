import json
import pytest
from src.agents.backends.mock import (
    AllIsolateMockBackend,
    AlwaysInvalidMockBackend,
    InvalidJsonMockBackend,
    MockBackend,
)
from src.agents.base import (
    FALLBACK_DECISION,
    BehaviorDecision,
    parse_decision,
    query_with_fallback,
)
from src.agents.prompts import ArchetypeConfig, build_prompt
from src.logging.agent_log import AgentLogger
from src.models.data_structures import EpidemicContext

def test_mock_backend_returns_valid_json():
    backend = MockBackend()
    raw = backend.query("test prompt")
    decision = BehaviorDecision.model_validate_json(raw)
    assert decision.isolate is False
    assert decision.mask is False
    assert decision.reasoning == "mock response"

def test_mock_backend_tracks_calls():
    backend = MockBackend()
    backend.query("prompt 1")
    backend.query("prompt 2")
    assert backend.call_count == 2
    assert backend.call_log == ["prompt 1", "prompt 2"]

def test_all_isolate_mock():
    backend = AllIsolateMockBackend()
    raw = backend.query("any prompt")
    decision = BehaviorDecision.model_validate_json(raw)
    assert decision.isolate is True
    assert decision.mask is True
    assert decision.reduce_contacts is True

def test_parse_decision_valid():
    raw = json.dumps({
        "isolate": True,
        "isolate_confidence": 0.8,
        "mask": False,
        "mask_confidence": 0.3,
        "reduce_contacts": True,
        "reduce_contacts_confidence": 0.7,
        "see_doctor": False,
        "see_doctor_confidence": 0.2,
        "reasoning": "test reasoning",
    })
    decision = parse_decision(raw)
    assert decision.isolate is True
    assert decision.isolate_confidence == 0.8

def test_parse_decision_invalid_json():
    with pytest.raises(ValueError, match="invalid json"):
        parse_decision("this is not json")

def test_parse_decision_missing_field():
    raw = json.dumps({"isolate": True})
    with pytest.raises(ValueError, match="schema validation failed"):
        parse_decision(raw)

def test_query_with_fallback_valid_response():
    backend = MockBackend()
    decision, raw, success = query_with_fallback(backend, "test")
    assert success is True
    assert decision.reasoning == "mock response"
    assert backend.call_count == 1

def test_query_with_fallback_retry_recovers():
    backend = InvalidJsonMockBackend()
    decision, raw, success = query_with_fallback(backend, "test")
    assert success is True
    assert decision.reasoning == "recovered after retry"
    assert backend.call_count == 2

def test_query_with_fallback_total_failure():
    backend = AlwaysInvalidMockBackend()
    decision, raw, success = query_with_fallback(backend, "test")
    assert success is False
    assert decision == FALLBACK_DECISION
    assert decision.reasoning == "fallback: LLM unavailable"
    assert backend.call_count == 2

def test_build_prompt_structure():
    archetype = ArchetypeConfig(
        id="young_active",
        name="Young Working Adult",
        age_range=(18, 35),
        occupation="office worker, courier, freelancer, or student",
        health_literacy=4,
        risk_tolerance=7,
        family="lives alone, in a flatshare, or with parents",
        prompt_additions="Sick days cost you money or missed deadlines.",
    )
    context = EpidemicContext(
        day=10,
        total_infected=150,
        total_susceptible=8000,
        total_recovered=200,
        growth_rate=0.15,
        new_infections_today=30,
        phase="GROWTH",
    )
    prompt = build_prompt(archetype, context, "healthy")
    assert "Young Working Adult" in prompt
    assert "day 10" in prompt
    assert "isolate" in prompt
    assert "JSON" in prompt
    assert "50%" in prompt

def test_agent_logger_writes_jsonl(tmp_path):
    agent_logger = AgentLogger(
        run_id="test_001",
        backend_name="mock",
        output_dir=str(tmp_path),
    )
    decision = BehaviorDecision(
        isolate=True,
        isolate_confidence=0.9,
        mask=True,
        mask_confidence=0.8,
        reduce_contacts=False,
        reduce_contacts_confidence=0.3,
        see_doctor=False,
        see_doctor_confidence=0.2,
        reasoning="test reasoning",
    )
    agent_logger.log(
        sim_day=5,
        phase="GROWTH",
        archetype_id="young_worker",
        prompt="test prompt text",
        raw_response='{"isolate": true}',
        parsed_decision=decision,
        parse_success=True,
        tokens_input=100,
        tokens_output=50,
        latency_ms=200,
        cost_usd=0.001,
    )
    agent_logger.log(
        sim_day=5,
        phase="GROWTH",
        archetype_id="elderly_chronic",
        prompt="another prompt",
        raw_response="not json",
        parsed_decision=None,
        parse_success=False,
    )
    assert agent_logger.count == 2
    lines = agent_logger.path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    entry1 = json.loads(lines[0])
    assert entry1["archetype_id"] == "young_worker"
    assert entry1["parse_success"] is True
    assert entry1["parsed_decision"]["isolate"] is True
    assert entry1["tokens_input"] == 100
    entry2 = json.loads(lines[1])
    assert entry2["archetype_id"] == "elderly_chronic"
    assert entry2["parse_success"] is False
    assert entry2["parsed_decision"] is None

def test_agent_logger_file_path(tmp_path):
    agent_logger = AgentLogger(
        run_id="20240101_120000",
        backend_name="gpt4",
        output_dir=str(tmp_path),
    )
    assert agent_logger.path.name == "run_20240101_120000_gpt4.jsonl"

def test_llm_fallback(tmp_path):
    backend = AlwaysInvalidMockBackend()
    agent_logger = AgentLogger(
        run_id="fallback_test",
        backend_name="mock",
        output_dir=str(tmp_path),
    )
    decision, raw, success = query_with_fallback(backend, "test prompt")
    agent_logger.log(
        sim_day=1,
        phase="BASELINE",
        archetype_id="young_worker",
        prompt="test prompt",
        raw_response=raw,
        parsed_decision=decision if success else None,
        parse_success=success,
    )
    assert success is False
    assert decision == FALLBACK_DECISION
    lines = agent_logger.path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["parse_success"] is False
    assert entry["parsed_decision"] is None

def test_json_logging(tmp_path):
    backend = MockBackend()
    agent_logger = AgentLogger(
        run_id="logging_test",
        backend_name="mock",
        output_dir=str(tmp_path),
    )
    n_calls = 5
    for i in range(n_calls):
        raw = backend.query(f"prompt {i}")
        decision = BehaviorDecision.model_validate_json(raw)
        agent_logger.log(
            sim_day=i,
            phase="GROWTH",
            archetype_id=f"archetype_{i}",
            prompt=f"prompt {i}",
            raw_response=raw,
            parsed_decision=decision,
            parse_success=True,
        )
    assert backend.call_count == n_calls
    lines = agent_logger.path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == backend.call_count
