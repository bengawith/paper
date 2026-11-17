import copy

import pytest
from pydantic import ValidationError

from robust_airfoil.config import TuningConfig, ViabilityConfig, load_all_configs, load_yaml
from robust_airfoil.constants import CONFIG_ROOT


def test_all_checked_in_configs_are_strictly_validated():
    configs = load_all_configs()
    assert set(configs) == {"sources", "viability", "model_baseline", "tuning", "uncertainty", "optimisation"}


def test_unknown_config_key_is_rejected():
    payload = copy.deepcopy(load_yaml(CONFIG_ROOT / "viability.yaml"))
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        ViabilityConfig.model_validate(payload)


def test_invalid_tuning_range_is_rejected():
    payload = copy.deepcopy(load_yaml(CONFIG_ROOT / "tuning.yaml"))
    payload["search"]["dropout"] = [0.5, 0.1]
    with pytest.raises(ValidationError):
        TuningConfig.model_validate(payload)
