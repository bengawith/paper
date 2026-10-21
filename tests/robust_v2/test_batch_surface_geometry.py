"""Exact polyline semantics for vectorised production geometry operations."""
import numpy as np
import pytest

from robust_airfoil.geometry.validity import validate_geometry
from robust_airfoil.optimisation import candidate_uncertainty as module


def operations():
    interpolate = getattr(module,'interpolate_surface_batches',None)
    validate = getattr(module,'validate_surface_batches',None)
    assert interpolate is not None and validate is not None, 'Vectorised polyline operations are absent'
    return interpolate,validate


def test_batch_interpolation_matches_individual_numpy_interpolation():
    interpolate,_ = operations()
    rng=np.random.default_rng(321)
    xs=np.c_[np.zeros(19),np.sort(rng.uniform(size=(19,79)),axis=1),np.ones(19)]
    ys=rng.uniform(-.1,.1,size=xs.shape)
    surface=np.stack([xs,ys],axis=-1)
    queries=np.sort(rng.uniform(-.01,1.01,size=(19,2001)),axis=1)
    expected=np.asarray([np.interp(q,x,y) for q,x,y in zip(queries,xs,ys,strict=True)])
    np.testing.assert_allclose(interpolate(queries,surface),expected,atol=3e-12,rtol=3e-12)


def test_batch_validity_matches_union_knot_scalar_validation():
    _,validate=operations()
    rng=np.random.default_rng(322)
    x=np.tile(np.linspace(0.,1.,51),(60,1))
    y=.1*np.sqrt(x)*(1-x)
    upper=np.stack([x,y],axis=-1)
    lower=np.stack([x,-y],axis=-1)
    lower[:,1:-1,0]+=rng.uniform(-.009,.009,size=(60,49))
    lower[::3,1:-1,1]+=.15
    upper[1,3,0]=upper[1,2,0]
    lower[2,0,0]=.01
    upper[4,4,1]=np.nan
    lower[5,5,1]=np.inf
    expected=np.asarray([validate_geometry(a,b).valid for a,b in zip(upper,lower,strict=True)])
    np.testing.assert_array_equal(validate(upper,lower),expected)


def test_crossing_at_unmatched_knots_is_not_missed():
    _,validate=operations()
    upper=np.array([[[0.,0.],[.2,.11],[.8,.04],[1.,0.]]])
    lower=np.array([[[0.,0.],[.7,.10],[.9,.03],[1.,0.]]])
    assert not validate(upper,lower)[0]


def test_interpolation_refuses_unordered_or_nonfinite_input():
    interpolate,_=operations()
    with pytest.raises(ValueError):
        interpolate(np.linspace(0,1,9),np.array([[[0.,0.],[.5,1.],[.4,2.],[1.,0.]]]))
