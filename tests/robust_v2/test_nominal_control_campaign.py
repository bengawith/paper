"""Real optimisation software on synthetic fixtures; never aerodynamic evidence."""
import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from robust_airfoil.optimisation.problems import DesignBounds


def module():
    name='robust_airfoil.optimisation.nominal_control'
    assert importlib.util.find_spec(name) is not None, 'The matched-budget nominal control implementation is absent'
    return importlib.import_module(name)


class FixtureEvaluator:
    def __init__(self):
        self.counters={'candidate_evaluations':0}

    def _nominal_constraints(self,candidates):
        self.counters['candidate_evaluations']+=len(candidates)
        constraints=np.full((len(candidates),6),-1.)
        constraints[:,0]=candidates[:,0]-.7
        diagnostics=[{'nominal_weighted_drag':float(.01+np.sum((row-.2)**2)),
                      'nominal_required_lift_feasible':True} for row in candidates]
        return constraints,diagnostics


def test_problem_uses_actual_nominal_required_lift_drag_and_failure_is_infeasible():
    implementation=module()
    bounds=DesignBounds(np.zeros(2),np.ones(2))
    problem=implementation.NominalControlProblem(FixtureEvaluator(),bounds)
    output={}
    problem._evaluate(np.array([[.2,.2],[.8,.2]]),output)
    np.testing.assert_allclose(output['F'].ravel(),[.01,.37])
    assert output['G'][0].max()<=0 and output['G'][1].max()>0
    missing=SimpleNamespace(_nominal_constraints=lambda values:(np.full((len(values),6),-1.),[{'nominal_weighted_drag':None,'nominal_required_lift_feasible':False} for _ in values]))
    output={}
    implementation.NominalControlProblem(missing,bounds)._evaluate(np.array([[.2,.2]]),output)
    assert output['G'].max()>0


def test_initial_population_retains_reference_and_same_feasibility_ranked_observed_seeding():
    implementation=module()
    bounds=DesignBounds(np.zeros(2),np.ones(2))
    reference=np.array([.3,.3])
    observed=np.array([[.2,.2],[.8,.8],[.4,.4]])
    population=implementation.initial_nominal_population(FixtureEvaluator(),bounds,reference,observed,8,17)
    again=implementation.initial_nominal_population(FixtureEvaluator(),bounds,reference,observed,8,17)
    np.testing.assert_array_equal(population,again)
    np.testing.assert_array_equal(population[0],reference)
    assert np.all((population>=0)&(population<=1))
    np.testing.assert_array_equal(population[1:4],[[.2,.2],[.4,.4],[.8,.8]])


def test_checkpointed_nominal_profile_completes_declared_seeds_and_refuses_context_mixing(tmp_path):
    implementation=module()
    args=dict(evaluator=FixtureEvaluator(),bounds=DesignBounds(np.zeros(2),np.ones(2)),
              reference=np.array([.3,.3]),observed=np.array([[.2,.2],[.4,.4]]),
              seeds=[61,62],population=8,generations=3,output=tmp_path,
              scientific_context={'model_identity':'synthetic-software-fixture'})
    result=implementation.run_nominal_control_profile(**args)
    assert len(result['runs'])==2
    assert all(len(run['history'])==3 for run in result['runs'])
    assert all(run['feasible_count']>0 for run in result['runs'])
    before=[(tmp_path/f'nominal_seed_{seed}.json').read_bytes() for seed in args['seeds']]
    recovered=implementation.run_nominal_control_profile(**args)
    assert recovered['runs']==result['runs']
    assert before==[(tmp_path/f'nominal_seed_{seed}.json').read_bytes() for seed in args['seeds']]
    args['scientific_context']={'model_identity':'different-scientific-model'}
    with pytest.raises(ValueError,match='context'):
        implementation.run_nominal_control_profile(**args)


def test_reference_duplicate_does_not_shrink_declared_initial_population():
    implementation=module()
    bounds=DesignBounds(np.zeros(2),np.ones(2))
    reference=np.array([.3,.3])
    observed=np.array([[.3,.3],[.3,.3],[.2,.2],[.4,.4]])
    population=implementation.initial_nominal_population(FixtureEvaluator(),bounds,reference,observed,8,19)
    assert population.shape==(8,2)
    assert len(np.unique(population,axis=0))==8
    assert int(np.all(population==reference,axis=1).sum())==1
