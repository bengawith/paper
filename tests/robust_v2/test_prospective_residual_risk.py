"""Frozen calibration-only residual-risk ranking for the sealed confirmation cohort.

Owned by the review/confirmation session; deliberate RED is part of this task.
Synthetic fixtures below are software checks, not aerodynamic evidence.
"""
import importlib
import importlib.util

import numpy as np
import pandas as pd
import pytest

TARGETS=['cl','log_cd','cm']


def module():
    name='robust_airfoil.residual_risk'
    assert importlib.util.find_spec(name) is not None, 'Frozen prospective residual-risk implementation is absent'
    return importlib.import_module(name)


def calibration():
    rng=np.random.default_rng(512)
    frame=pd.DataFrame({'split':['calibration']*80,'airfoil_id':np.repeat(['a','b','c','d'],20),
                        'support_distance':rng.uniform(.1,2.,80)})
    for index,target in enumerate(TARGETS):
        frame[target]=rng.normal(size=80)
        frame['prediction_'+target]=frame[target]+rng.normal(scale=.05*(index+1),size=80)
        frame['ensemble_std_'+target]=rng.uniform(.001,.2,80)
        frame['mask_'+target]=True
    return frame


def test_fitted_artifact_is_json_serialisable_and_uses_only_calibration_rows():
    import json
    functions=module()
    artifact=functions.fit_residual_risk(calibration(),ridge_alpha=10.)
    json.dumps(artifact,allow_nan=False)
    assert artifact['training_rows']==80
    assert artifact['models']['support_only']['features']==['support_distance']
    assert artifact['ridge_alpha']==10.
    assert artifact['training_partition']=='calibration'
    changed=calibration()
    changed.loc[0,'split']='locked_test'
    with pytest.raises(ValueError,match='calibration'):
        functions.fit_residual_risk(changed,ridge_alpha=10.)


def test_prediction_does_not_read_confirmation_labels_or_mutate_frozen_artifact():
    import copy
    functions=module()
    artifact=functions.fit_residual_risk(calibration(),ridge_alpha=10.)
    before=copy.deepcopy(artifact)
    query=calibration().drop(columns=[*TARGETS,*['prediction_'+target for target in TARGETS],*['mask_'+target for target in TARGETS]])
    prediction=functions.predict_residual_risk(query,artifact)
    augmented=query.copy()
    for target in TARGETS:
        augmented[target]=1e20
        augmented['prediction_'+target]=-1e20
    same=functions.predict_residual_risk(augmented,artifact)
    for name in prediction:
        np.testing.assert_array_equal(prediction[name],same[name])
    assert before==artifact


def test_frozen_high_error_definition_uses_recorded_training_scales_not_query_quantiles():
    functions=module()
    frame=calibration()
    artifact=functions.fit_residual_risk(frame,ridge_alpha=10.)
    actual=functions.label_residual_risk(frame,artifact)
    extreme=frame.copy()
    extreme.loc[0,'prediction_cl']+=1000.
    modified=functions.label_residual_risk(extreme,artifact)
    np.testing.assert_array_equal(actual['high_error'][1:],modified['high_error'][1:])
    np.testing.assert_allclose(actual['scaled_error'][1:],modified['scaled_error'][1:],rtol=0,atol=0)
    assert modified['high_error'][0]


def test_nonfinite_scoring_features_are_refused_not_replaced_by_plausible_risk():
    functions=module()
    artifact=functions.fit_residual_risk(calibration(),ridge_alpha=10.)
    bad=calibration()
    bad.loc[1,'support_distance']=np.nan
    with pytest.raises(ValueError,match='finite'):
        functions.predict_residual_risk(bad,artifact)
