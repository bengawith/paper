import zipfile

import pytest

from robust_airfoil.geometry.parser import parse_coordinate_text
from robust_airfoil.sources.uiuc import extract_zip_safely


def test_coordinate_parser_ignores_title_and_lednicer_counts(coordinate_text):
    parsed = parse_coordinate_text("61 61\n" + coordinate_text)
    assert parsed.shape == (9, 2)
    assert parsed[:, 0].max() == 1.0


def test_zip_slip_is_rejected(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.dat", "bad")
    with pytest.raises(ValueError):
        extract_zip_safely(archive, tmp_path / "out")


def test_root_directory_member_is_ignored(tmp_path):
    archive = tmp_path / "root-directory.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("/", "")
        handle.writestr("safe/file.dat", "coordinates")
    extracted = extract_zip_safely(archive, tmp_path / "out")
    assert [path.name for path in extracted] == ["file.dat"]
