"""Batch acceleration must retain all scalar rising-branch and failure semantics."""
import numpy as np
import pytest

from robust_airfoil.optimisation import objectives


def compare(alpha,cl,cd,required=None,weights=None):
    required = np.array([.4,.7,1.]) if required is None else required
    weights = np.array([.25,.5,.25]) if weights is None else weights
    function=getattr(objectives,'batched_weighted_required_lift_drag',None)
    assert function is not None, 'Exact batched required-lift evaluation is missing'
    actual=function(alpha,cl,cd,required,weights)
    expected=np.asarray([objectives.weighted_required_lift_drag(alpha,a,b,required,weights) for a,b in zip(cl,cd,strict=True)])
    np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)


def test_batch_matches_scalar_for_rising_stalled_and_rerising_curves():
    rng=np.random.default_rng(420)
    alpha=np.array([0.,2.,4.,6.,8.])
    cl=np.cumsum(rng.uniform(-.3,.5,size=(200,5)),axis=1)
    cd=rng.uniform(.004,.03,size=cl.shape)
    compare(alpha,cl,cd)


def test_duplicate_target_knots_and_tiny_plateaux_do_not_create_false_roots():
    alpha=np.arange(5.)
    cl=np.array([[.1,.4,.7,1.,1.1],[.1,.7,.7,.8,1.1],[.1,.7,.7+1e-13,.8,1.1],[.4,.7,1.,1.3,1.4]])
    compare(alpha,cl,np.full_like(cl,.01))


def test_nan_negative_drag_and_unsorted_duplicate_alpha_follow_scalar_contract():
    cl=np.array([[.1,.4,.7,1.,1.2],[.1,np.nan,.7,1.,1.2]])
    cd=np.full_like(cl,.01)
    compare(np.arange(5.),cl,cd)
    cd[0,1]=-.01
    compare(np.arange(5.),cl,cd)
    cl=np.array([[.1,.4,.4,.7,1.1],[.1,.4,.6,.7,1.1]])
    cd=np.full_like(cl,.01)
    compare(np.array([0.,1.,1.,2.,3.]),cl,cd)
    compare(np.array([0.,2.,1.,3.,4.]),cl,cd)


def test_batch_rejects_invalid_weight_definition():
    function=getattr(objectives,'batched_weighted_required_lift_drag',None)
    assert function is not None, 'Exact batched required-lift evaluation is missing'
    with pytest.raises(ValueError):
        function(np.arange(3.),np.ones((2,3)),np.ones((2,3)),np.array([.5]),np.array([-1.]))
