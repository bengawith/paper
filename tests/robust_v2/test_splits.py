import pandas as pd

from robust_airfoil.data.splits import make_cluster_splits


def test_geometry_cluster_never_leaks_between_splits():
    frame = pd.DataFrame({"airfoil_id": [f"a{i}" for i in range(12)], "cluster_id": [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
    splits = make_cluster_splits(frame)
    assert splits.groupby("cluster_id")["split"].nunique().max() == 1
    assert {"development", "calibration", "locked_test"}.issubset(set(splits["split"]))
