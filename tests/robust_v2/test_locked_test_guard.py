import pytest

from robust_airfoil.data.splits import assert_locked_test_access


def test_locked_test_requires_frozen_manifest(tmp_path):
    manifest = tmp_path / "FROZEN_MODEL_MANIFEST.json"
    with pytest.raises(PermissionError):
        assert_locked_test_access(manifest, "locked_test")
    assert_locked_test_access(manifest, "calibration")
    manifest.write_text('{"schema_version": 2, "status": "frozen"}', encoding="utf-8")
    assert_locked_test_access(manifest, "locked_test")
