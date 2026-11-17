"""Invalid nominal placeholders are not observed perturbation outcomes."""
from types import SimpleNamespace

import numpy as np
import pytest

from robust_airfoil.optimisation import study
from robust_airfoil.optimisation.candidate_uncertainty import CandidateScenarios


def test_diagnostic_moments_exclude_invalid_placeholders_and_report_selection():
    summarise = getattr(study,'summarise_valid_aerodynamics',None)
    assert summarise is not None, 'Explicit failure-aware aerodynamic diagnostics are absent'
    result = summarise(np.array([[2.,3.],[999.,999.],[4.,5.]]),np.array([[.1,.2],[999.,999.],[.3,.4]]),np.array([True,False,True]))
    assert result['conditional_aerodynamic_sample_count'] == 2
    np.testing.assert_allclose(result['mean_lift_by_alpha'],[3.,4.])
    np.testing.assert_allclose(result['mean_cm_by_alpha'],[.2,.3])
    assert 'conditional' in result['aerodynamic_moment_scope']
    empty = summarise(np.ones((2,3)),np.ones((2,3)),np.zeros(2,bool))
    assert empty['mean_lift_by_alpha'] == [None,None,None]


def test_ava_baseline_uses_candidate_scenarios_not_zero_placeholder_deltas():
    evaluator = study.RobustCandidateEvaluator.__new__(study.RobustCandidateEvaluator)
    evaluator.alpha = np.array([0.,1.])
    evaluator.deltas = np.zeros((2,12))
    evaluator.candidate_latent = np.ones((2,16))
    variations = np.zeros((2,12))
    variations[:,0] = [-.1,.1]
    evaluator.prepared_scenarios = SimpleNamespace(evaluate=lambda candidate:CandidateScenarios(variations,np.zeros(2,bool),('', '')))
    def prediction(frame):
        result = frame.copy()
        result['prediction_cl'] = result['lower_weight_0']+result['alpha_deg']
        return result
    evaluator.ensemble = SimpleNamespace(predict=prediction)
    evaluator._nominal_constraints = lambda value:(np.zeros((1,6)),[{}])
    evaluator._support_and_disagreement = lambda frame,shape:np.ones(shape,bool)
    difference = evaluator.ava_objective(np.zeros(12),1.)-evaluator.ava_objective(np.zeros(12),0.)
    assert difference == pytest.approx(np.sqrt(.02))
