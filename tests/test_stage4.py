import numpy as np
import pandas as pd
from src.models.abm import ABMSimulation
from src.models.data_structures import EpidemicContext, Phase, SEIRState
from src.models.hybrid import HybridModel
from src.models.seir import SEIRModel
from src.regime.hmm_detector import HMMDetector
from src.regime.threshold import ThresholdDetector

def test_seir_conservation(default_config):
    seir = SEIRModel(default_config)
    initial = SEIRState(S=9900, E=50, I=40, R=10, N=10000)
    states = seir.run(days=range(100), initial=initial, beta=0.00003, stochastic=False)
    for i, s in enumerate(states):
        total = s.S + s.E + s.I + s.R
        assert total == s.N, f"day {i}: S+E+I+R={total} != N={s.N}"

def test_seir_conservation_stochastic(default_config):
    seir = SEIRModel(default_config)
    initial = SEIRState(S=990, E=5, I=4, R=1, N=1000)
    np.random.seed(42)
    states = seir.run(days=range(50), initial=initial, beta=0.0003, stochastic=True)
    for i, s in enumerate(states):
        total = s.S + s.E + s.I + s.R
        assert total == s.N, f"day {i}: S+E+I+R={total} != N={s.N}"

def test_seir_epidemic_dynamics(default_config):
    seir = SEIRModel(default_config)
    initial = SEIRState(S=9900, E=50, I=50, R=0, N=10000)
    states = seir.run(days=range(200), initial=initial, beta=0.00004, stochastic=False)
    i_values = [s.I for s in states]
    assert max(i_values) > initial.I, "infection should spread"
    assert i_values[-1] < max(i_values), "epidemic should wane"

def test_seir_get_curve(default_config):
    seir = SEIRModel(default_config)
    initial = SEIRState(S=990, E=5, I=4, R=1, N=1000)
    states = seir.run(days=range(20), initial=initial, beta=0.0003)
    df = seir.get_curve(states)
    assert len(df) == len(states)
    for col in ("S", "E", "I", "R", "N"):
        assert col in df.columns

def test_hybrid_switching(default_config, small_population):
    data, households = small_population
    abm = ABMSimulation(config=default_config, data=data, households=households)
    seir = SEIRModel(default_config)
    class EarlySwitchDetector:
        def detect(self, context):
            return Phase.GROWTH
        def should_switch(self, context):
            return context.day >= 5
    detector = EarlySwitchDetector()
    hybrid = HybridModel(abm=abm, seir=seir, detector=detector)
    result = hybrid.run(days=range(1, 30), seed=42)
    assert "switch_day" in result.config
    assert result.config["switch_day"] is not None
    assert result.config["switch_day"] >= 5
    assert len(result.days) == 29

def test_hybrid_no_switch(default_config, small_population):
    data, households = small_population
    abm = ABMSimulation(config=default_config, data=data, households=households)
    seir = SEIRModel(default_config)
    class NeverSwitchDetector:
        def detect(self, context):
            return Phase.BASELINE
        def should_switch(self, context):
            return False
    detector = NeverSwitchDetector()
    hybrid = HybridModel(abm=abm, seir=seir, detector=detector)
    result = hybrid.run(days=range(1, 20), seed=42)
    assert result.config["switch_day"] is None
    assert len(result.days) == 19

def test_hmm_fit_and_detect():
    np.random.seed(42)
    n_days = 200
    incidence = np.concatenate([
        np.random.poisson(5, 50),
        np.arange(5, 55) + np.random.normal(0, 3, 50),
        np.random.poisson(55, 50),
        np.arange(55, 5, -1) + np.random.normal(0, 3, 50),
    ])
    growth_rate = np.diff(incidence, prepend=incidence[0]) / np.maximum(incidence, 1)
    hist_data = pd.DataFrame({
        "incidence": incidence[:n_days],
        "growth_rate": growth_rate[:n_days],
    })
    detector = HMMDetector(n_states=4)
    detector.fit(hist_data)
    assert detector.fitted
    context = EpidemicContext(
        day=70, total_infected=100, total_susceptible=9000,
        total_recovered=50, growth_rate=0.15,
        new_infections_today=40, phase=None,
    )
    phase = detector.detect(context)
    assert isinstance(phase, Phase)

def test_hmm_save_load(tmp_path):
    np.random.seed(42)
    hist_data = pd.DataFrame({
        "incidence": np.random.poisson(20, 100),
        "growth_rate": np.random.normal(0, 0.1, 100),
    })
    detector = HMMDetector(n_states=4)
    detector.fit(hist_data)
    path = tmp_path / "test_hmm.pkl"
    detector.save(str(path))
    loaded = HMMDetector(n_states=4)
    loaded.load(str(path))
    assert loaded.fitted
    assert loaded.n_states == 4

def test_threshold_detector_phases():
    config = {"method": "roll_var_npeople", "min_infected": 50}
    det = ThresholdDetector(config)
    ctx = EpidemicContext(
        day=5, total_infected=10, total_susceptible=9900,
        total_recovered=0, growth_rate=0.0,
        new_infections_today=2, phase=None,
    )
    assert det.detect(ctx) == Phase.BASELINE
    ctx = EpidemicContext(
        day=20, total_infected=200, total_susceptible=9000,
        total_recovered=100, growth_rate=0.2,
        new_infections_today=30, phase=None,
    )
    assert det.detect(ctx) == Phase.GROWTH
    ctx = EpidemicContext(
        day=80, total_infected=100, total_susceptible=5000,
        total_recovered=4000, growth_rate=-0.1,
        new_infections_today=5, phase=None,
    )
    assert det.detect(ctx) == Phase.DECLINE
