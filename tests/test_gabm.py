from src.agents.archetypes import ArchetypeManager, load_archetypes
from src.agents.backends.mock import AllIsolateMockBackend, MockBackend
from src.logging.agent_log import AgentLogger
from src.models.abm import ABMSimulation
from src.models.data_structures import EpidemicContext
from src.models.gabm import GABMSimulation

def _make_gabm(config, data, households, backend=None, tmp_path=None):
    if backend is None:
        backend = MockBackend()
    archetypes = load_archetypes("config/archetypes.yaml")
    log_dir = str(tmp_path) if tmp_path else "results/logs"
    agent_logger = AgentLogger(
        run_id="test", backend_name="mock", output_dir=log_dir,
    )
    test_config = dict(config)
    test_agents = dict(config.get("agents", {}))
    test_agents["stochastic_noise"] = 0.0
    test_agents["compare_backends"] = False
    test_agents["behavior_activation"] = {
        "min_phase": "BASELINE",
        "min_prevalence": 0.0,
    }
    test_config["agents"] = test_agents
    manager = ArchetypeManager(
        archetypes=archetypes,
        agent_logger=agent_logger,
        agents_config=test_agents,
    )
    return GABMSimulation(
        config=test_config,
        data=data,
        households=households,
        llm_backend=backend,
        archetype_manager=manager,
    )

def test_gabm_inherits_abm(default_config, small_population, tmp_path):
    data, households = small_population
    days = range(1, 30)
    seed = 42
    abm = ABMSimulation(config=default_config, data=data, households=households)
    abm_result = abm.run(days=days, seed=seed)
    gabm = _make_gabm(default_config, data, households, tmp_path=tmp_path)
    gabm_result = gabm.run(days=days, seed=seed)
    for abm_day, gabm_day in zip(abm_result.days, gabm_result.days):
        assert abm_day.S == gabm_day.S, f"S mismatch on day {abm_day.day}"
        assert abm_day.E == gabm_day.E, f"E mismatch on day {abm_day.day}"
        assert abm_day.I == gabm_day.I, f"I mismatch on day {abm_day.day}"
        assert abm_day.R == gabm_day.R, f"R mismatch on day {abm_day.day}"
    for dr in gabm_result.days:
        assert dr.n_isolating == 0, f"n_isolating should be 0 on day {dr.day}"
        assert dr.n_masked == 0, f"n_masked should be 0 on day {dr.day}"

def test_isolation_reduces_spread(default_config, small_population, tmp_path):
    data, households = small_population
    days = range(1, 50)
    seed = 42
    abm = ABMSimulation(config=default_config, data=data, households=households)
    abm_result = abm.run(days=days, seed=seed)
    backend = AllIsolateMockBackend()
    test_config = dict(default_config)
    test_agents = dict(test_config.get("agents", {}))
    test_agents["behavior_activation"] = {
        "min_phase": "BASELINE",
        "min_prevalence": 0.0,
    }
    test_agents["compliance_rate"] = {
        "isolate": 1.0, "mask": 1.0, "reduce_contacts": 1.0, "see_doctor": 1.0,
    }
    test_config["agents"] = test_agents
    gabm = _make_gabm(test_config, data, households, backend=backend, tmp_path=tmp_path)
    gabm_result = gabm.run(days=days, seed=seed)
    abm_total = sum(
        sum(dr.new_infections.values()) for dr in abm_result.days
    )
    gabm_total = sum(
        sum(dr.new_infections.values()) for dr in gabm_result.days
    )
    assert gabm_total < abm_total, (
        f"isolation should reduce spread: gabm={gabm_total} vs abm={abm_total}"
    )

def test_archetype_caching(default_config, small_population, tmp_path):
    data, households = small_population
    backend = MockBackend()
    gabm = _make_gabm(default_config, data, households, backend=backend, tmp_path=tmp_path)
    gabm.run(days=range(1, 10), seed=42)
    n_with_agency = sum(
        1 for a in gabm.archetype_manager.archetypes if a.has_llm_agency
    )
    assert backend.call_count > 0, "at least one LLM call should have been made"
    assert backend.call_count % n_with_agency == 0, (
        f"call_count={backend.call_count} should be multiple of {n_with_agency}"
    )
    calls_before = backend.call_count
    for idx in gabm.data.index[:20]:
        gabm.archetype_manager.get_decision(gabm.data.loc[idx], gabm.data)
    assert backend.call_count == calls_before, (
        "get_decision should not trigger additional LLM calls"
    )

def test_phase_change_triggers_update(default_config, small_population, tmp_path):
    data, households = small_population
    backend = MockBackend()
    gabm = _make_gabm(default_config, data, households, backend=backend, tmp_path=tmp_path)
    gabm.run(days=range(1, 5), seed=42)
    n_with_agency = sum(
        1 for a in gabm.archetype_manager.archetypes if a.has_llm_agency
    )
    calls_after_first = backend.call_count
    assert calls_after_first >= n_with_agency
    gabm.archetype_manager.last_phase = "DIFFERENT_PHASE"
    context = EpidemicContext(
        day=10, total_infected=50, total_susceptible=8000,
        total_recovered=100, growth_rate=0.1,
        new_infections_today=10, phase="GROWTH",
    )
    gabm.archetype_manager.update_rules(context, backend)
    assert backend.call_count == calls_after_first + n_with_agency

def test_mask_reduces_lambda(default_config, small_population, tmp_path):
    data, households = small_population
    gabm = _make_gabm(default_config, data, households, tmp_path=tmp_path)
    gabm.set_initial_conditions()
    idx = gabm.data.index[0]
    gabm.data.at[idx, "wears_mask"] = True
    expected = gabm.lmbd * default_config["agents"]["mask_reduction_factor"]
    assert gabm.get_effective_lmbd(idx, "workplace") == expected
    assert gabm.get_effective_lmbd(idx, "school") == expected
    assert gabm.get_effective_lmbd(idx, "household") == gabm.lmbd
    idx2 = gabm.data.index[1]
    gabm.data.at[idx2, "wears_mask"] = False
    assert gabm.get_effective_lmbd(idx2, "workplace") == gabm.lmbd

def test_reduce_contacts_reduces_lambda(default_config, small_population, tmp_path):
    data, households = small_population
    gabm = _make_gabm(default_config, data, households, tmp_path=tmp_path)
    gabm.set_initial_conditions()
    idx = gabm.data.index[0]
    gabm.data.at[idx, "reduces_contacts"] = True
    gabm.data.at[idx, "wears_mask"] = False
    contact_factor = default_config["agents"].get("contact_reduction_factor", 0.7)
    expected = gabm.lmbd * contact_factor
    assert gabm.get_effective_lmbd(idx, "workplace") == expected
    assert gabm.get_effective_lmbd(idx, "school") == expected
    assert gabm.get_effective_lmbd(idx, "household") == gabm.lmbd
    gabm.data.at[idx, "wears_mask"] = True
    mask_factor = default_config["agents"]["mask_reduction_factor"]
    expected_both = gabm.lmbd * mask_factor * contact_factor
    assert abs(gabm.get_effective_lmbd(idx, "workplace") - expected_both) < 1e-10

def test_gabm_phase_detection_integration(default_config, small_population, tmp_path):
    data, households = small_population
    gabm = _make_gabm(default_config, data, households, tmp_path=tmp_path)
    assert gabm.detector is not None, "GABM should auto-create detector from config"
    gabm.run(days=range(1, 10), seed=42)
    assert gabm.archetype_manager.last_phase is not None

def test_archetype_assignment(default_config, small_population, tmp_path):
    data, households = small_population
    gabm = _make_gabm(default_config, data, households, tmp_path=tmp_path)
    gabm.set_initial_conditions()
    valid_ids = {a.id for a in gabm.archetype_manager.archetypes}
    assert gabm.data["archetype"].isin(valid_ids).all()
