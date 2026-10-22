import json

import numpy as np

from robust_airfoil.logging_utils import append_jsonl


def test_jsonl_logging_serialises_numpy_and_paths(tmp_path):
    append_jsonl(
        tmp_path,
        "09-16",
        "finished",
        array=np.asarray([1.0, 2.0]),
        scalar=np.float64(3.0),
        path=tmp_path / "artifact.json",
    )
    row = json.loads((tmp_path / "logs/09-16.jsonl").read_text(encoding="utf-8"))
    assert row["array"] == [1.0, 2.0]
    assert row["scalar"] == 3.0
    assert row["path"].endswith("artifact.json")
