"""The released controlled predictor must never use older numerical labels."""

import json

import numpy as np
import pandas as pd
import pytest

from robust_airfoil.cli import build_parser
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS
from robust_airfoil.predict import load_lineage_development


def _points(n):
    frame = pd.DataFrame(np.zeros((n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    frame["airfoil_id"] = ["a"] * n
    return frame


def test_prediction_defaults_to_reviewed_controlled_lineage():
    args = build_parser().parse_args(["predict", "--cst=" + ",".join(["0"] * 12)])
    assert args.lineage_id == "robust-v2-controlled-20261020"


def test_controlled_support_uses_only_lineage_development(tmp_path):
    lineage = tmp_path / "lineage"
    partition = lineage / "partition_inputs" / "development.parquet"
    partition.parent.mkdir(parents=True)
    _points(3).to_parquet(partition)
    legacy = tmp_path / "data/robust_v2/processed/full_exact/model_points.parquet"
    legacy.parent.mkdir(parents=True)
    _points(20).to_parquet(legacy)
    result, source = load_lineage_development(tmp_path, "robust-v2-controlled-20261020", lineage)
    assert len(result) == 3 and source == partition


def test_repaired_lineage_fails_closed_if_partition_missing(tmp_path):
    for lineage in ("robust-v2-controlled-20261020", "robust-v2-rebuilt-20261014"):
        with pytest.raises(FileNotFoundError, match="own frozen development"):
            load_lineage_development(tmp_path, lineage, tmp_path / "missing")


def test_legacy_lineage_uses_its_original_partition(tmp_path):
    lineage = tmp_path / "lineage"
    (lineage / "splits").mkdir(parents=True)
    (lineage / "splits/full.json").write_text(
        json.dumps(
            {
                "records": [
                    {"airfoil_id": "a", "split": "development"},
                    {"airfoil_id": "b", "split": "locked_test"},
                ]
            }
        )
    )
    legacy = tmp_path / "data/robust_v2/processed/full_exact/model_points.parquet"
    legacy.parent.mkdir(parents=True)
    frame = _points(2)
    frame.loc[1, "airfoil_id"] = "b"
    frame.to_parquet(legacy)
    result, source = load_lineage_development(tmp_path, "robust-v2-current-20260908", lineage)
    assert len(result) == 1 and source == legacy
    assert result.airfoil_id.tolist() == ["a"]


def test_support_source_without_features_is_not_valid(tmp_path):
    folder = tmp_path / "partition_inputs"
    folder.mkdir()
    pd.DataFrame({"airfoil_id": ["a"]}).to_parquet(folder / "development.parquet")
    with pytest.raises(ValueError, match="input features"):
        load_lineage_development(tmp_path, "robust-v2-controlled-20261020", tmp_path)
