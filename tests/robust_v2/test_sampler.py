import numpy as np

from robust_airfoil.modelling.samplers import UniformAirfoilPointSampler


def test_sampler_is_deterministic_by_epoch_and_balances_groups():
    ids = np.array(["large"] * 100 + ["small"])
    first = list(UniformAirfoilPointSampler(ids, num_samples=200, seed=4))
    second = list(UniformAirfoilPointSampler(ids, num_samples=200, seed=4))
    assert first == second
    sampled_names = ids[first]
    assert 60 < int((sampled_names == "small").sum()) < 140
