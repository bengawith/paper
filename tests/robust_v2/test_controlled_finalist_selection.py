"""Fixed deterministic selection cannot promote an infeasible or dominated design."""
import importlib
import importlib.util

import numpy as np
import pytest


def module():
    name='robust_airfoil.optimisation.controlled_selection'
    assert importlib.util.find_spec(name) is not None, 'Predeclared deterministic finalist selection is absent'
    return importlib.import_module(name)


def fixture():
    x=np.zeros((5,12))
    x[:,0]=np.arange(5)
    f=np.array([[.8,1.2],[1.,1.],[1.2,.8],[.5,.5],[2.,2.]])
    g=np.full((5,6),-1.)
    g[3,-1]=1.
    return x,f,g


def test_infeasible_best_point_and_dominated_points_cannot_be_selected():
    functions=module()
    x,f,g=fixture()
    result=functions.select_controlled_risk_finalists(x,f,g,np.ones(2))
    assert result['feasible_count']==4 and result['nondominated_count']==3
    np.testing.assert_array_equal(result['selected']['risk_knee']['parameters'],x[1])
    np.testing.assert_array_equal(result['selected']['minimum_mean']['parameters'],x[0])
    np.testing.assert_array_equal(result['selected']['minimum_tail']['parameters'],x[2])


def test_order_does_not_change_tie_breaking_or_selected_geometries():
    functions=module()
    x,f,g=fixture()
    first=functions.select_controlled_risk_finalists(x,f,g,np.ones(2))
    order=np.array([4,1,3,2,0])
    second=functions.select_controlled_risk_finalists(x[order],f[order],g[order],np.ones(2))
    assert first['selected']==second['selected']


def test_nominal_control_requires_feasibility_and_does_not_consult_robust_or_solver_results():
    functions=module()
    x,f,g=fixture()
    nominal=np.array([.04,.03,.02,.001,.05])
    result=functions.select_controlled_nominal_finalist(x,nominal,g)
    np.testing.assert_array_equal(result['parameters'],x[2])
    assert result['nominal_weighted_cd']==pytest.approx(.02)


def test_empty_feasible_set_is_a_reportable_failure_not_an_optimum():
    functions=module()
    x,f,g=fixture()
    with pytest.raises(ValueError,match='feasible'):
        functions.select_controlled_risk_finalists(x,f,np.ones_like(g),np.ones(2))
    with pytest.raises(ValueError,match='feasible'):
        functions.select_controlled_nominal_finalist(x,np.ones(5),np.ones_like(g))


def test_nonfinite_results_are_excluded_and_scales_cannot_be_zero():
    functions=module()
    x,f,g=fixture()
    f[1,0]=np.nan
    result=functions.select_controlled_risk_finalists(x,f,g,np.ones(2))
    assert result['feasible_count']==3
    with pytest.raises(ValueError,match='scale'):
        functions.select_controlled_risk_finalists(x,f,g,np.array([0.,1.]))
