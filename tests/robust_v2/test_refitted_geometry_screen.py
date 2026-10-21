"""A physically valid displaced surface does not certify its lossy CST refit."""
from types import SimpleNamespace

import numpy as np

from robust_airfoil.config import UncertaintyConfig, load_yaml
from robust_airfoil.constants import ROOT
from robust_airfoil.optimisation import candidate_uncertainty as module


def test_crossing_cst_refit_is_rejected_before_surrogate_prediction(monkeypatch):
    config = UncertaintyConfig.model_validate(load_yaml(ROOT/'configs/robust_v2/uncertainty.yaml'))
    nominal = np.array([-.13,-.13,-.1,-.08,-.05,.15,.17,.16,.1,.07,0,0])
    crossed = nominal.copy()
    crossed[:5] = .2
    crossed[5:10] = -.2
    monkeypatch.setattr(module,'fit_cst',lambda *args:SimpleNamespace(parameters=crossed,max_error=0.0))
    result = module.candidate_surface_scenarios(nominal,np.zeros((2,16)),config)
    assert result.invalid.all()
    assert all(reason.startswith('invalid_refitted_geometry') for reason in result.invalid_reasons)
    np.testing.assert_array_equal(result.parameters,np.tile(nominal,(2,1)))
