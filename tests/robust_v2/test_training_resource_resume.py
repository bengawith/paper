"""Resource pressure pauses after an intact checkpoint, never loses training work."""
import json

import pytest

from robust_airfoil.modelling import train


def test_training_resource_pause_leaves_checkpoint_and_resumes_same_context(model_frame,tmp_path,monkeypatch):
    guard=getattr(train,'assert_campaign_resources',None)
    assert guard is not None, 'Actual trainer has no cooperative resource-floor checkpoint gate'
    calls=[]
    def pause_after_first_checkpoint(directory):
        calls.append(str(directory))
        if (tmp_path/'training_state.pt').exists():
            raise RuntimeError('memory resource floor fixture')
        return {'available_memory_bytes':10*1024**3,'disk_free_bytes':100*1024**3}
    monkeypatch.setattr(train,'assert_campaign_resources',pause_after_first_checkpoint)
    arguments=dict(hidden_width=16,hidden_layers=1,dropout=0.,batch_size=16,
                   maximum_epochs=2,patience=5,seed=930,resident_batches=True,
                   checkpoint_each_epoch=True,enforce_resource_floors=True)
    with pytest.raises(RuntimeError,match='memory resource floor'):
        train.train_model(model_frame,model_frame,tmp_path,**arguments)
    assert (tmp_path/'training_state.pt').exists()
    assert len(json.loads((tmp_path/'history.json').read_text()))==1
    monkeypatch.setattr(train,'assert_campaign_resources',lambda directory:{'available_memory_bytes':10*1024**3,'disk_free_bytes':100*1024**3})
    summary=train.train_model(model_frame,model_frame,tmp_path,**arguments,resume=True)
    assert summary['epochs']==2
    assert summary['resumed_from_epoch']==1
    assert len(json.loads((tmp_path/'history.json').read_text()))==2
