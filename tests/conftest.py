import numpy as np
import pandas as pd
import pytest
import yaml

@pytest.fixture
def default_config():
    with open("config/default.yaml") as f:
        return yaml.safe_load(f)

@pytest.fixture
def small_population():
    rng = np.random.RandomState(42)
    n = 100
    ages = rng.randint(5, 80, n)
    work_ids = []
    for i, age in enumerate(ages):
        if age > 17:
            work_ids.append(str(i // 10))
        else:
            if age >= 7:
                work_ids.append(str(100 + age // 3))
            else:
                work_ids.append("X")
    data = pd.DataFrame({
        "sp_id": range(1000, 1000 + n),
        "sp_hh_id": [i // 4 for i in range(n)],
        "age": ages,
        "sex": rng.choice(["M", "F"], n),
        "work_id": work_ids,
    })
    households = pd.DataFrame({
        "sp_id": list(range(n // 4)),
        "latitude": rng.uniform(59.93, 59.95, n // 4),
        "longitude": rng.uniform(30.25, 30.30, n // 4),
    })
    return data, households
