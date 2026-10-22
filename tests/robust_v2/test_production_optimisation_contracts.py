"""Prospective scale and checkpoint contracts: synthetic optimisation only."""
import numpy as np
import pytest
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem

from robust_airfoil.optimisation import study
from robust_airfoil.optimisation.problems import DesignBounds


def test_predeclared_scales_enable_the_unchanged_agreement_criterion():
    runs = [{'solutions':[[0.,0.],[1.,1.]],'objectives':[[1.,2.],[2.,1.]],'constraint_values':[[-1.],[-1.]]}]*10
    result = study._agreement(runs,DesignBounds(np.zeros(2),np.ones(2)),objective_scales=np.array([1.,2.]))
    assert result['agreement_status'] == 'pass'
    assert result['objective_agreement']['passed']
    assert result['objective_agreement']['threshold'] == .20
    assert result['objective_agreement']['minimum_feasible_fraction'] == .80
    assert result['objective_scale_context'] == 'predeclared_reference_objectives'


def test_checkpoint_resume_matches_uninterrupted_population(tmp_path):
    save = getattr(study,'save_algorithm_checkpoint',None)
    load = getattr(study,'load_algorithm_checkpoint',None)
    assert save is not None and load is not None, 'Production checkpoint functions must be connected to study'
    context = {'seed':199,'generations':6,'population':12,'synthetic_fixture':True}
    problem = get_problem('zdt1',n_var=4)
    algorithm = NSGA2(pop_size=12)
    algorithm.setup(problem,termination=('n_gen',6),seed=199)
    algorithm.next()
    algorithm.next()
    save(algorithm,tmp_path,context)
    while algorithm.has_next():
        algorithm.next()
    expected = algorithm.pop.get('X').copy()
    recovered = load(tmp_path,context,problem=problem)
    while recovered.has_next():
        recovered.next()
    np.testing.assert_array_equal(recovered.pop.get('X'),expected)
    np.testing.assert_array_equal(recovered.pop.get('F'),algorithm.pop.get('F'))


def test_checkpoint_refuses_changed_context_and_bytes(tmp_path):
    save = getattr(study,'save_algorithm_checkpoint',None)
    load = getattr(study,'load_algorithm_checkpoint',None)
    assert save is not None and load is not None, 'Production checkpoint functions must be connected to study'
    problem = get_problem('zdt1',n_var=4)
    algorithm = NSGA2(pop_size=8)
    algorithm.setup(problem,termination=('n_gen',3),seed=201)
    algorithm.next()
    save(algorithm,tmp_path,{'seed':201})
    with pytest.raises(ValueError,match='context'):
        load(tmp_path,{'seed':202},problem=problem)
    target = tmp_path/'algorithm_checkpoint.pkl'
    target.write_bytes(target.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash'):
        load(tmp_path,{'seed':201},problem=problem)


def test_serialised_objectives_and_constraints_come_from_the_same_final_evaluation():
    from types import SimpleNamespace

    result = SimpleNamespace(X=np.zeros((1,12)),F=np.array([[.01,.02]]),pop=None)
    evaluator = SimpleNamespace(evaluate=lambda x:(np.array([[.011,.022]]),np.full((1,6),-1.),[{}]),counters={})
    callback = SimpleNamespace(history=[])
    stored = study._serialise_nsga_result(result,evaluator,callback,7,64,'fixture-context')
    np.testing.assert_array_equal(stored['objectives'],[[.011,.022]])
    assert stored['final_recheck_max_objective_difference'] == pytest.approx(.002)
