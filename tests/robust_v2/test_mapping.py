from robust_airfoil.data.mapping import build_name_index, canonical_name, map_by_strict_name


def test_exact_mapping_and_no_fuzzy_fallback(tmp_path):
    geometry = tmp_path / "NACA-2412.dat"
    geometry.write_text("same", encoding="utf-8")
    index = build_name_index([geometry])
    assert canonical_name("naca-2412-il-1000000.csv") == "naca2412"
    assert map_by_strict_name("naca2412-il-1000000.csv", index).geometry_path == geometry
    assert map_by_strict_name("naca2413-il-1000000.csv", index).method == "unmapped"
