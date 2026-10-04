import json

import numpy as np

from robust_airfoil.hashing import hash_object, sha256_bytes, sha256_file
from robust_airfoil.provenance import write_json_atomic


def test_hashes_are_stable_and_content_sensitive(tmp_path):
    path = tmp_path / "value.txt"
    path.write_bytes(b"abc")
    assert sha256_file(path) == sha256_bytes(b"abc")
    assert hash_object({"b": 2, "a": 1}) == hash_object({"a": 1, "b": 2})
    assert hash_object({"a": 1}) != hash_object({"a": 2})


def test_atomic_json_supports_numpy_and_paths(tmp_path):
    output = tmp_path / "record.json"
    write_json_atomic(output, {"array": np.array([1, 2]), "scalar": np.float64(3.5), "path": tmp_path})
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["array"] == [1, 2]
    assert payload["scalar"] == 3.5
    assert payload["path"] == tmp_path.as_posix()
