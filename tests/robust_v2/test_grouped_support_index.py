"""The accelerator must return exact 13-D nearest-state distances, not a heuristic."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from robust_airfoil.modelling import calibrate


def index(features):
    implementation = getattr(calibrate,'GroupedSupportIndex',None)
    assert implementation is not None, 'Exact grouped-geometry support index is not implemented'
    return implementation(features)


def test_grouped_queries_match_full_state_tree_with_missing_incidence_ranges():
    rng = np.random.default_rng(2109)
    pieces = []
    for geometry in rng.normal(size=(21,12)):
        alpha = rng.uniform(-4,5,size=int(rng.integers(1,55)))
        pieces.append(np.c_[np.tile(geometry,(len(alpha),1)),alpha])
    source = np.vstack(pieces)
    queries = rng.normal(size=(127,13))
    actual, ids = index(source).query(queries,k=5)
    expected = cKDTree(source).query(queries,k=5)[0]
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(source[ids]-queries[:,None,:],axis=2),actual,rtol=1e-12,atol=1e-12)


def test_search_expands_when_nearest_geometries_have_no_nearby_incidence():
    source = np.vstack([np.c_[np.repeat(np.arange(8),5),np.tile(np.arange(100,105),8)],np.c_[np.full(5,9),np.linspace(-.1,.1,5)]])
    query = np.array([[0.,0.]])
    np.testing.assert_allclose(index(source).query(query,k=5)[0],cKDTree(source).query(query,k=5)[0],rtol=1e-13,atol=1e-13)


def test_duplicate_states_and_fewer_than_five_rows_follow_ckdtree_semantics():
    for source in (np.zeros((7,3)),np.array([[0.,0.,1.],[1.,2.,4.]])):
        queries = np.zeros((2,3))
        np.testing.assert_allclose(index(source).query(queries,k=5)[0],cKDTree(source).query(queries,k=5)[0],rtol=1e-13,atol=1e-13)


def test_nonfinite_query_is_not_silently_accepted():
    with pytest.raises(ValueError):
        index(np.zeros((7,3))).query(np.array([[0.,np.nan,0.]]),k=5)
