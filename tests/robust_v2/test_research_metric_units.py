"""Point and geometry aggregation, target transforms and bootstrap units are explicit."""
import importlib
import importlib.util

import numpy as np
import pandas as pd
import pytest


def metric_module():
    name='robust_airfoil.research_metrics'
    assert importlib.util.find_spec(name) is not None, 'Shared research metric accounting is absent'
    return importlib.import_module(name)


def fixture():
    frame=pd.DataFrame({'airfoil_id':['a','b','b','b'],'cl':[1.,1.,1.,1.],
                        'prediction_cl':[2.,1.,1.,1.],'mask_cl':[True]*4,
                        'log_cd':np.log([.01,.02,.02,.02]),'prediction_log_cd':np.log([.02,.02,.02,.02]),'mask_log_cd':[True]*4,
                        'cm':[-.03]*4,'prediction_cm':[-.03]*4,'mask_cm':[True]*4,
                        'trusted_domain':[False,True,True,True]})
    rule={'residual_absolute_quantiles':{'cl':{'0.95':.5},'log_cd':{'0.95':.1},'cm':{'0.95':.01}}}
    return frame,rule


def test_micro_macro_linear_drag_and_full_observed_curve_coverage_are_not_conflated():
    functions=metric_module()
    frame,rule=fixture()
    rows=functions.prediction_metrics(frame,rule)
    lookup={(row['subset'],row['target']):row for row in rows}
    lift=lookup['all','cl']
    assert lift['states']==4 and lift['airfoils']==2
    assert lift['mae']==pytest.approx(.25)
    assert lift['geometry_macro_mae']==pytest.approx(.5)
    assert lift['state_interval_coverage']==pytest.approx(.75)
    assert lift['whole_observed_curve_coverage']==pytest.approx(.5)
    assert lookup['all','cd']['mae']==pytest.approx(.0025)
    assert lookup['all','log_cd']['mae']==pytest.approx(np.log(2)/4)
    assert lookup['accepted','cl']['states']==3
    assert lookup['rejected','cl']['states']==1


def test_cluster_bootstrap_resamples_whole_geometries_and_preserves_fixed_model():
    functions=metric_module()
    frame,rule=fixture()
    before=frame.copy(deep=True)
    result=functions.geometry_bootstrap_metrics(frame,rule,replicates=1000,seed=8)
    pd.testing.assert_frame_equal(frame,before)
    assert result['resampling_units']==2
    assert result['replicates']==1000
    assert result['targets']['cl']['geometry_macro_mae_point']==pytest.approx(.5)
    assert result['targets']['cl']['geometry_macro_mae_interval']==[0.,1.]
    assert 'fixed' in result['scope'].lower()


def test_missing_or_nonfinite_observations_are_accounted_not_filled():
    functions=metric_module()
    frame,rule=fixture()
    frame.loc[0,'mask_cl']=False
    frame.loc[1,'prediction_cm']=np.nan
    rows=functions.prediction_metrics(frame,rule)
    lookup={(row['subset'],row['target']):row for row in rows}
    assert lookup['all','cl']['states']==3
    assert lookup['all','cm']['states']==3
    assert lookup['rejected','cl']['states']==0
    assert lookup['rejected','cl']['mae'] is None
