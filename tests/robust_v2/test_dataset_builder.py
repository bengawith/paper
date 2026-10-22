import pandas as pd

from robust_airfoil.data.builder import build_long_form_dataset


def test_builder_writes_only_genuine_source_rows(tmp_path, coordinate_text, polar_text):
    geometry = tmp_path / "demo.dat"
    polar = tmp_path / "demo-il-1000000.csv"
    geometry.write_text(coordinate_text, encoding="utf-8")
    polar.write_text(polar_text, encoding="utf-8")
    output = tmp_path / "processed"
    summary = build_long_form_dataset([polar], [geometry], output, "test", cosine_points=41)
    points = pd.read_parquet(output / "model_points.parquet")
    assert summary["mapped_airfoils"] == 1
    assert len(points) == 4
    assert points["source_row"].tolist() == [0, 1, 2, 3]
    assert bool(points["mask_log_cd"].all())
