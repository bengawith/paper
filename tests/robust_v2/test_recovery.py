import pickle

from robust_airfoil.recovery import discover_candidates, safe_pickle_globals


def test_recovery_redacts_paths_and_never_unpickles(tmp_path):
    csv_path = tmp_path / "airfoil_polar.csv"
    csv_path.write_text("alpha,cl\n0,0\n", encoding="utf-8")
    pickle_path = tmp_path / "airfoil_cache.pkl"
    pickle_path.write_bytes(pickle.dumps({"safe": [1, 2]}))
    candidates = discover_candidates([tmp_path])
    assert {candidate.suffix for candidate in candidates} == {".csv", ".pkl"}
    assert all(str(tmp_path) not in candidate.path_redacted for candidate in candidates)
    assert isinstance(safe_pickle_globals(pickle_path), tuple)
