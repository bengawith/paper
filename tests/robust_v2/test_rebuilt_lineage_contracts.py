"""Predeclared contracts for the genuinely rebuilt geometry/model lineage."""
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import lsq_linear
from torch.utils.data._utils.collate import default_collate

from robust_airfoil.data import splits
from robust_airfoil.geometry.cst import fit_cst, reconstruct_cst
from robust_airfoil.modelling.dataset import PolarPointDataset, fit_scaling


def test_valid_closed_surface_cannot_fit_negative_trailing_edge():
    x = (1 - np.cos(np.linspace(0, np.pi, 201))) / 2
    thickness = .1 * np.sqrt(x) * (1-x)**1.25
    upper, lower = np.column_stack([x,thickness]), np.column_stack([x,-thickness])
    fitted = fit_cst(upper,lower)
    assert fitted.parameters[-1] >= 0
    # Independently solve the same convex one-bound problem: do not merely
    # clip the old negative TE, which fails to re-optimise other coefficients.
    design = np.column_stack([np.concatenate(reconstruct_cst(unit,x)[::-1]) for unit in np.eye(12)])
    expected = lsq_linear(design,np.r_[-thickness,thickness],bounds=(np.r_[np.full(11,-np.inf),0],np.full(12,np.inf)),method='bvls',tol=1e-12)
    np.testing.assert_allclose(fitted.parameters,expected.x,atol=1e-9,rtol=1e-9)


def test_reclustered_shapes_never_move_protected_geometry_into_development():
    old = pd.DataFrame({'airfoil_id':['a','b','c','d'], 'split':['development','locked_test','development','calibration']})
    shapes = pd.DataFrame({'airfoil_id':['a','b','c','d'], 'cluster_id':[0,0,1,1]})
    result = splits.protect_rebuilt_splits(shapes,old)
    assert result.set_index('airfoil_id')['split'].to_dict() == {'a':'locked_test','b':'locked_test','c':'calibration','d':'calibration'}
    assert result.groupby('cluster_id')['split'].nunique().max() == 1


def test_rebuilt_partitioning_refuses_unknown_source_identity():
    old = pd.DataFrame({'airfoil_id':['a'],'split':['development']})
    with pytest.raises(ValueError,match='identity'):
        splits.protect_rebuilt_splits(pd.DataFrame({'airfoil_id':['unknown'],'cluster_id':[0]}),old)


def test_resident_tensor_batches_preserve_real_dataset_masks_and_values(model_frame):
    frame = model_frame.copy()
    frame.loc[3,'cl'] = np.nan
    frame.loc[3,'mask_cl'] = False
    dataset = PolarPointDataset(frame,fit_scaling(frame))
    indices = np.array([3,11,2,3,0])
    expected = default_collate([dataset[int(i)] for i in indices])
    tensors = dataset.tensor_dict(torch.device('cpu'))
    for name,values in expected.items():
        torch.testing.assert_close(tensors[name][indices],values,rtol=0,atol=0)
