"""The saved calibration decisions must use exactly the released screening rule."""
import numpy as np

from robust_airfoil.modelling.calibrate import apply_trust_model, calibrate_trust_model
from robust_airfoil.modelling.dataset import TARGET_COLUMNS, fit_scaling


def test_calibration_reported_flags_and_fraction_match_runtime_rule(model_frame,tmp_path):
    development = model_frame.copy()
    calibration = model_frame.copy()
    calibration['lower_weight_0'] = np.linspace(-.2,.6,len(calibration))
    for target in TARGET_COLUMNS:
        calibration['prediction_'+target] = calibration[target] + .1
        calibration['ensemble_std_'+target] = .01
    scaling = fit_scaling(development)
    rule = calibrate_trust_model(development,calibration,scaling,tmp_path/'rule.json')
    applied = apply_trust_model(development,calibration,scaling,rule)
    np.testing.assert_array_equal(calibration['trusted_domain'],applied['trusted_domain'])
    assert rule['trusted_calibration_fraction'] == float(applied['trusted_domain'].mean())
