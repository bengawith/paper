"""Low resource conditions must stop a campaign, not reduce the experiment."""
from types import SimpleNamespace

import pytest

from robust_airfoil.optimisation import checkpointing


def test_resource_floor_refuses_low_memory_and_disk(monkeypatch,tmp_path):
    guard=getattr(checkpointing,'assert_campaign_resources',None)
    assert guard is not None, 'Production resource guard is missing'
    import shutil

    import psutil
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=2*1024**3))
    monkeypatch.setattr(shutil,'disk_usage',lambda path:SimpleNamespace(free=100*1024**3))
    with pytest.raises(RuntimeError,match='memory'):
        guard(tmp_path)
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=8*1024**3))
    monkeypatch.setattr(shutil,'disk_usage',lambda path:SimpleNamespace(free=10*1024**3))
    with pytest.raises(RuntimeError,match='disk'):
        guard(tmp_path)
    monkeypatch.setattr(shutil,'disk_usage',lambda path:SimpleNamespace(free=100*1024**3))
    assert guard(tmp_path)['available_memory_bytes'] == 8*1024**3
