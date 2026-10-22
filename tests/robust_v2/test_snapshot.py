import subprocess

from robust_airfoil.sources.snapshot import acquire_pinned_snapshot


def test_pinned_local_snapshot_is_verified(tmp_path):
    source = tmp_path / "source"
    (source / "dat/case-dat").mkdir(parents=True)
    (source / "dat/aerofoil-dat").mkdir(parents=True)
    (source / "dat/case-dat/demo-il-1000000.csv").write_text("polar", encoding="utf-8")
    (source / "dat/aerofoil-dat/demo.dat").write_text("coords", encoding="utf-8")
    (source / "LICENSE").write_text("test license", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=source, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "snapshot"], cwd=source, check=True, capture_output=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True).stdout.strip()
    payload = acquire_pinned_snapshot(str(source), commit, tmp_path / "clone", tmp_path / "manifest.json")
    assert payload["actual_commit"] == commit
    assert payload["case_count"] == 1
    assert payload["geometry_count"] == 1
