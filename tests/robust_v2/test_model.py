import torch

from robust_airfoil.modelling.models import ConditionedPolarMLP


def test_conditioned_model_has_three_point_heads():
    model = ConditionedPolarMLP()
    output = model(torch.zeros(7, 13))
    assert set(output) == {"cl", "log_cd", "cm"}
    assert all(value.shape == (7,) for value in output.values())
