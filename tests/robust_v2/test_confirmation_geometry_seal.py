"""Geometry-only confirmation preparation; synthetic fixtures, never solver evidence.

Owned by the active review/confirmation session; see REVIEW_COORDINATION.md.
Expected RED precedes this task's implementation. Do not repair concurrently.
"""
import importlib
import importlib.util

import numpy as np
import pytest

from robust_airfoil.geometry.cst import reconstruct_cst


def builder():
    name='robust_airfoil.confirmation_geometry'
    assert importlib.util.find_spec(name) is not None, 'The sealed geometry-only confirmation builder is not implemented'
    return importlib.import_module(name).build_confirmation_geometry


def fixture():
    base=np.array([-.13,-.13,-.1,-.08,-.05,.15,.17,.16,.1,.07,0.,.002])
    values=np.tile(base,(7,1))
    values[:,6]+=np.arange(7)*.02
    values[:,2]-=np.arange(7)*.006
    return values


def test_only_distinct_development_clusters_generate_deterministic_confirmation_geometries():
    build=builder()
    values=fixture()
    clusters=np.array([0,1,2,3,4,5,6])
    args=dict(parameters=values,development_indices=np.array([0,1,2,3]),cluster_ids=clusters,
              count=8,seed=441,minimum_rms_separation=1e-6,mixing_limits=(.2,.8))
    first=build(**args)
    second=build(**args)
    assert first==second
    assert len(first['candidates'])==8
    x=(1-np.cos(np.linspace(0,np.pi,201)))/2
    original=np.asarray([np.r_[reconstruct_cst(p,x)[0],reconstruct_cst(p,x)[1]] for p in values])
    accepted=[]
    for candidate in first['candidates']:
        a,b=candidate['parent_indices']
        weight=candidate['first_parent_weight']
        assert a in [0,1,2,3] and b in [0,1,2,3] and clusters[a]!=clusters[b]
        assert .2<=weight<=.8
        np.testing.assert_allclose(candidate['parameters'],weight*values[a]+(1-weight)*values[b],rtol=0,atol=0)
        p=np.asarray(candidate['parameters'])
        upper,lower=reconstruct_cst(p,x)
        vector=np.r_[upper,lower]
        comparisons=np.vstack([original,*[v[None,:] for v in accepted]]) if accepted else original
        separation=float(np.sqrt(np.mean((comparisons-vector)**2,axis=1)).min())
        assert separation>1e-6
        assert candidate['minimum_rms_to_prior_shapes']==pytest.approx(separation,abs=1e-14)
        accepted.append(vector)


def test_impossible_separation_is_reported_not_relaxed():
    build=builder()
    with pytest.raises(RuntimeError,match='separation|admission'):
        build(fixture(),np.array([0,1,2]),np.arange(7),count=2,seed=442,
              minimum_rms_separation=1.,mixing_limits=(.2,.8),maximum_attempts=5)


@pytest.mark.parametrize('development,clusters',[(np.array([0]),np.arange(7)),(np.array([0,1]),np.zeros(7,int))])
def test_at_least_two_distinct_development_clusters_are_required(development,clusters):
    build=builder()
    with pytest.raises(ValueError):
        build(fixture(),development,clusters,count=1,seed=443,minimum_rms_separation=1e-6)
