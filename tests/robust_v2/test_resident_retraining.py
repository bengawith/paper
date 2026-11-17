"""Small real-model tests for the rebuild execution path, not paper evidence."""
import json

import numpy as np
import pytest
import torch

from robust_airfoil.modelling import ensemble
from robust_airfoil.modelling.train import train_model


def test_resident_training_stores_completed_epochs_and_resumes_without_refitting(model_frame, tmp_path):
    options = dict(hidden_width=16, hidden_layers=1, dropout=0.0, batch_size=16,
                   maximum_epochs=2, patience=5, seed=887, resident_batches=True,
                   checkpoint_each_epoch=True)
    result = train_model(model_frame,model_frame,tmp_path,**options)
    history = json.loads((tmp_path/'history.json').read_text())
    assert result['epochs'] == len(history) == 2
    assert (tmp_path/'training_state.pt').is_file()
    before = torch.load(tmp_path/'model.pt',map_location='cpu',weights_only=True)
    resumed = train_model(model_frame,model_frame,tmp_path,**options,resume=True)
    after = torch.load(tmp_path/'model.pt',map_location='cpu',weights_only=True)
    assert resumed['epochs'] == 2
    for key in before:
        torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)


def test_out_of_bag_stopping_rows_share_no_training_cluster(model_frame):
    frame = model_frame.copy()
    frame['cluster_id'] = np.repeat(np.arange(4),12)
    bootstrap, validation, metadata = ensemble.cluster_bootstrap_with_oob(frame,29)
    assert len(bootstrap) > 0 and len(validation) > 0
    assert set(bootstrap.cluster_id).isdisjoint(set(validation.cluster_id))
    assert set(bootstrap.cluster_id) | set(validation.cluster_id) == set(frame.cluster_id)
    assert metadata['sampled_cluster_ids']


def test_resident_resume_refuses_modified_training_values(model_frame,tmp_path):
    options = dict(hidden_width=16, hidden_layers=1, dropout=0.0, batch_size=16,
                   maximum_epochs=1, patience=5, seed=889, resident_batches=True,
                   checkpoint_each_epoch=True)
    train_model(model_frame,model_frame,tmp_path,**options)
    changed = model_frame.copy()
    changed.loc[0,'cl'] += .1
    with pytest.raises(ValueError,match='context'):
        train_model(changed,model_frame,tmp_path,**options,resume=True)
