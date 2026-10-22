"""Real ordered DAT input must be represented by its *declared CST reconstruction*."""

import numpy as np
import pytest

from robust_airfoil.cli import build_parser
from robust_airfoil.geometry.cst import reconstruct_cst
from robust_airfoil.predict import cst_from_airfoil_file, predict_polar


def test_cli_accepts_one_and_only_one_airfoil_input(tmp_path):
    parser = build_parser()
    args = parser.parse_args(["predict", "--airfoil-file", str(tmp_path / "shape.dat")])
    assert args.airfoil_file == tmp_path / "shape.dat"
    assert args.cst is None
    with pytest.raises(SystemExit):
        parser.parse_args(["predict", "--cst=0,0", "--airfoil-file", str(tmp_path / "shape.dat")])


def test_selig_input_is_fitted_and_source_is_identifiable(tmp_path):
    values = np.asarray([-0.1] * 5 + [0.1] * 5 + [0.0, 0.0])
    x = (1 - np.cos(np.linspace(0, np.pi, 201))) / 2
    upper, lower = reconstruct_cst(values, x)
    coords = np.vstack(
        [np.column_stack([x[::-1], upper[::-1]]), np.column_stack([x[1:], lower[1:]])]
    )
    path = tmp_path / "ordered_airfoil.dat"
    path.write_text(
        "Synthetic symmetric CST\n" + "\n".join(f"{xp:.12f} {yp:.12f}" for xp, yp in coords) + "\n"
    )
    fitted, report = cst_from_airfoil_file(path)
    assert fitted.shape == (12,)
    assert np.isfinite(fitted).all()
    assert report["input_type"] == "normalised_source_file_CST_fit"
    assert report["cst_max_absolute_fit_error_chord"] < 0.005
    assert len(report["source_sha256"]) == 64


def test_invalid_dat_file_fails_closed(tmp_path):
    path = tmp_path / "bad.dat"
    path.write_text("bad header\n0.0 0.0\n1.0 0.0\n")
    with pytest.raises(ValueError):
        cst_from_airfoil_file(path)


def test_physical_geometry_rejected_before_model_loading():
    with pytest.raises(ValueError, match="physically inadmissible"):
        predict_polar("robust-v2-controlled-20261020", np.zeros(12), np.array([2.0]))


def test_invalid_incidence_rejected_before_model_loading():
    params = np.asarray([-0.1] * 5 + [0.1] * 5 + [0.0, 0.0])
    with pytest.raises(ValueError, match="finite incidence"):
        predict_polar("robust-v2-controlled-20261020", params, np.array([np.nan]))
