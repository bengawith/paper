from robust_airfoil.data.polar_parser import parse_polar_text


def test_parser_preserves_rows_metadata_and_gaps(polar_text):
    parsed = parse_polar_text(polar_text)
    assert parsed.metadata == {"source_name": "memory", "reynolds_number": 1e6, "mach": 0.0, "ncrit": 9.0}
    assert parsed.points["alpha_deg"].tolist() == [-2.0, 0.0, 2.0, 4.0]
    assert len(parsed.points) == 4
    assert not parsed.points["duplicate_alpha"].any()


def test_parser_flags_conflicting_duplicate_alpha(polar_text):
    parsed = parse_polar_text(polar_text + "2.0 0.30 0.012 0.008 -0.04 0.6 0.5\n")
    assert parsed.points.loc[parsed.points["alpha_deg"] == 2.0, "conflicting_alpha"].all()
