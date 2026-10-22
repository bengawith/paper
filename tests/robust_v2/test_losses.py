import pytest
import torch

from robust_airfoil.modelling.losses import masked_macro_multitask_loss


def test_masked_loss_ignores_missing_targets():
    predictions = {"cl": torch.tensor([1.0, 99.0]), "log_cd": torch.tensor([0.0, 0.0]), "cm": torch.tensor([0.0, 0.0])}
    targets = {name: torch.zeros(2) for name in predictions}
    masks = {"cl": torch.tensor([True, False]), "log_cd": torch.tensor([True, True]), "cm": torch.tensor([False, False])}
    loss, parts = masked_macro_multitask_loss(predictions, targets, masks, torch.ones(2), "mse", 1.0)
    assert parts["cl"].item() == pytest.approx(1.0)
    assert parts["cm"].item() == 0.0
    assert torch.isfinite(loss)
