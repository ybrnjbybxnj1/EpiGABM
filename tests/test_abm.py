import numpy as np
from src.models.abm import ABMSimulation

def test_abm_reproducibility(default_config, small_population):
    data, households = small_population
    days = range(1, 30)
    seed = 123
    sim1 = ABMSimulation(config=default_config, data=data, households=households)
    result1 = sim1.run(days=days, seed=seed)
    sim2 = ABMSimulation(config=default_config, data=data, households=households)
    result2 = sim2.run(days=days, seed=seed)
    assert len(result1.days) == len(result2.days)
    for d1, d2 in zip(result1.days, result2.days):
        assert d1.day == d2.day
        assert d1.S == d2.S, f"S mismatch on day {d1.day}"
        assert d1.E == d2.E, f"E mismatch on day {d1.day}"
        assert d1.I == d2.I, f"I mismatch on day {d1.day}"
        assert d1.R == d2.R, f"R mismatch on day {d1.day}"
        assert d1.new_infections == d2.new_infections, f"new_infections mismatch on day {d1.day}"
        assert d1.prevalence == d2.prevalence, f"prevalence mismatch on day {d1.day}"
        assert d1.beta == d2.beta, f"beta mismatch on day {d1.day}"

def test_abm_infection_spreads(default_config, small_population):
    data, households = small_population
    days = range(1, 50)
    sim = ABMSimulation(config=default_config, data=data, households=households)
    result = sim.run(days=days, seed=42)
    max_prevalence = max(
        sum(dr.prevalence.values()) for dr in result.days
    )
    assert max_prevalence > default_config["model"]["infected_init"]["H1N1"], (
        "infection should spread beyond initial seed"
    )

def test_abm_seir_output_structure(default_config, small_population):
    data, households = small_population
    sim = ABMSimulation(config=default_config, data=data, households=households)
    result = sim.run(days=range(1, 10), seed=1)
    strains = default_config["model"]["strains"]
    for dr in result.days:
        for strain in strains:
            assert strain in dr.S
            assert strain in dr.E
            assert strain in dr.I
            assert strain in dr.R
            assert strain in dr.beta

def test_abm_result_serialization(default_config, small_population, tmp_path):
    data, households = small_population
    sim = ABMSimulation(config=default_config, data=data, households=households)
    result = sim.run(days=range(1, 10), seed=1)
    out_path = tmp_path / "test_run.json"
    result.to_json(out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 100

def test_abm_matches_original(default_config, small_population):
    data, households = small_population
    days = range(1, 80)
    seed = 99
    sim1 = ABMSimulation(config=default_config, data=data, households=households)
    result1 = sim1.run(days=days, seed=seed)
    sim2 = ABMSimulation(config=default_config, data=data, households=households)
    result2 = sim2.run(days=days, seed=seed)
    prev1 = [sum(dr.prevalence.values()) for dr in result1.days]
    prev2 = [sum(dr.prevalence.values()) for dr in result2.days]
    assert prev1 == prev2, "same seed must produce identical prevalence curves"
    n_initial = default_config["model"]["infected_init"]["H1N1"]
    peak_prev = max(prev1)
    assert peak_prev > n_initial, "infection must spread beyond initial seed"
    peak_day_idx = int(np.argmax(prev1))
    assert peak_day_idx > 0, "peak should not be on day 1"
    assert prev1[-1] <= peak_prev, "prevalence should not exceed peak at end"
    n_pop = len(data)
    total_ever_infected = sum(sum(dr.new_infections.values()) for dr in result1.days)
    attack_rate = total_ever_infected / n_pop
    assert 0.05 < attack_rate < 0.90, f"attack rate {attack_rate:.2f} outside plausible range"
    for dr in result1.days:
        for strain in default_config["model"]["strains"]:
            assert dr.S[strain] >= 0
            assert dr.E[strain] >= 0
            assert dr.I[strain] >= 0
            assert dr.R[strain] >= 0
            total = dr.S[strain] + dr.E[strain] + dr.I[strain] + dr.R[strain]
            assert total <= n_pop, (
                f"S+E+I+R > N on day {dr.day}, strain {strain}: {total} > {n_pop}"
            )

def test_abm_to_dataframe(default_config, small_population):
    data, households = small_population
    sim = ABMSimulation(config=default_config, data=data, households=households)
    result = sim.run(days=range(1, 10), seed=1)
    df = result.to_dataframe()
    assert len(df) == 9
    assert "day" in df.columns
    assert "S_H1N1" in df.columns
