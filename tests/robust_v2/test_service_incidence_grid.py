"""Required-lift evaluation resolution is explicit, not borrowed from a baseline."""
from types import SimpleNamespace

import numpy as np
import pytest

from robust_airfoil.config import OptimisationConfig, load_yaml
from robust_airfoil.constants import ROOT
from robust_airfoil.modelling.dataset import TARGET_COLUMNS
from robust_airfoil.optimisation.study import RobustCandidateEvaluator, _frame


def configuration(grid):
    source=load_yaml(ROOT/'configs/robust_v2/optimisation.yaml')
    source['service_alpha_deg']=grid
    return OptimisationConfig.model_validate(source)


def test_actual_evaluator_uses_explicit_service_resolution_without_changing_baseline():
    grid=np.arange(0.,8.0001,.25).tolist()
    opt=configuration(grid)
    def predict(frame):
        result=frame.copy()
        result['prediction_cl']=.2+.1*result.alpha_deg
        result['prediction_log_cd']=np.log(.01)
        result['prediction_cm']=-.03
        for target in TARGET_COLUMNS:
            result['ensemble_std_'+target]=0.
        return result
    ensemble=SimpleNamespace(first_scaling=SimpleNamespace(feature_center=np.zeros(13),feature_scale=np.ones(13)),predict=predict)
    reference=np.array([-.13,-.13,-.1,-.08,-.05,.15,.17,.16,.1,.07,0.,.002])
    development=_frame(np.tile(reference,(5,1)),np.asarray(grid))
    rule={'support_distance_quantiles':{'trusted_threshold_from_residual_tolerance':1e6},
          'ensemble_disagreement_quantiles':{target:{'trusted_threshold_from_residual_tolerance':1.} for target in TARGET_COLUMNS},
          'alpha_coverage_deg':[-12.,16.]}
    evaluator=RobustCandidateEvaluator(reference,np.zeros((2,12)),np.zeros(2,bool),development,ensemble,rule,opt)
    np.testing.assert_array_equal(evaluator.alpha,grid)
    assert opt.ava_baseline.alpha_deg==[0.,2.,4.,6.,8.]


@pytest.mark.parametrize('grid', [[0.],[0.,0.],[1.,0.],[0.,float('nan')]])
def test_invalid_service_grid_is_rejected(grid):
    with pytest.raises(ValueError):
        configuration(grid)
