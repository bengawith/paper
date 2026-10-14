from __future__ import annotations

from pathlib import Path

import pytest

import robust_airfoil.pipeline as pipeline
import robust_airfoil.study as study
from robust_airfoil.cli import build_parser


def test_lineage_paths_are_isolated_and_validate_id(tmp_path: Path) -> None:
    paths = study.study_paths("current-20260908", tmp_path)
    assert paths.results_root == tmp_path / "results/robust_v2/lineages/current-20260908"
    assert paths.reports_root == tmp_path / "reports/robust_v2/lineages/current-20260908"
    with pytest.raises(ValueError, match="lowercase"):
        study.study_paths("Invalid", tmp_path)


def test_pipeline_lineage_context_restores_legacy_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    legacy_results = pipeline.RESULTS_ROOT
    legacy_reports = pipeline.REPORTS_ROOT
    seen: dict[str, object] = {}

    def fake_run(resume: bool) -> int:
        seen["resume"] = resume
        seen["results"] = pipeline.RESULTS_ROOT
        seen["reports"] = pipeline.REPORTS_ROOT
        seen["manifest"] = pipeline.FROZEN_MANIFEST_PATH
        seen["metadata"] = pipeline._ACTIVE_LINEAGE_METADATA
        return 0

    monkeypatch.setattr(pipeline, "_run_viability", fake_run)
    assert pipeline.run_viability(
        resume=True,
        results_root=tmp_path / "results",
        reports_root=tmp_path / "reports",
        lineage_metadata={"lineage_id": "current-20260908"},
    ) == 0
    assert seen["resume"] is True
    assert seen["results"] == tmp_path / "results"
    assert seen["reports"] == tmp_path / "reports"
    assert seen["manifest"] == tmp_path / "results/FROZEN_MODEL_MANIFEST.json"
    assert seen["metadata"] == {"lineage_id": "current-20260908"}
    assert legacy_results == pipeline.RESULTS_ROOT
    assert legacy_reports == pipeline.REPORTS_ROOT


def test_study_run_creates_immutable_context_and_passes_lineage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data/robust_v2/processed/full_exact"
    data_root.mkdir(parents=True)
    (data_root / "airfoils.parquet").write_bytes(b"airfoils")
    (data_root / "model_points.parquet").write_bytes(b"points")
    protocol = tmp_path / "study.yaml"
    protocol.write_text(
        "\n".join([
            "schema_version: robust-v2-study-v1",
            "status: working",
            "lineage_id: current-20260908",
            "parent_lineage_id: legacy-20260826",
            "engineering_seed: 20260824",
            "evidence_seed: 20260908",
        ]),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_viability(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "run_viability", fake_run_viability)
    assert study.run_study(protocol, "current-20260908", resume=False) == 0
    paths = study.study_paths("current-20260908", tmp_path)
    assert paths.context_path.is_file()
    assert captured["results_root"] == paths.results_root
    assert captured["reports_root"] == paths.reports_root
    metadata = captured["lineage_metadata"]
    assert isinstance(metadata, dict)
    assert metadata["lineage_id"] == "current-20260908"
    assert metadata["parent_lineage_id"] == "legacy-20260826"
    assert metadata["lineage_protocol_hash"]
    assert set(metadata["lineage_dataset_hashes"]) == {"airfoils.parquet", "model_points.parquet"}
    assert metadata["engineering_seed"] == 20260824
    assert metadata["evidence_seed"] == 20260908
    with pytest.raises(FileExistsError, match="New lineage"):
        study.run_study(protocol, "current-20260908", resume=False)


def test_study_cli_parses_full_range() -> None:
    arguments = build_parser().parse_args([
        "study", "run", "--protocol", "configs/robust_v2/study_final.yaml", "--lineage-id", "current-20260908",
    ])
    assert arguments.from_stage == "preflight"
    assert arguments.to_stage == "report"


def test_study_run_passes_requested_partial_phase_range(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data/robust_v2/processed/full_exact"
    data_root.mkdir(parents=True)
    (data_root / "airfoils.parquet").write_bytes(b"airfoils")
    (data_root / "model_points.parquet").write_bytes(b"points")
    protocol = tmp_path / "study.yaml"
    protocol.write_text(
        "\n".join(
            [
                "schema_version: robust-v2-study-v1",
                "status: working",
                "lineage_id: current-20260909",
                "parent_lineage_id: legacy-20260826",
                "engineering_seed: 20260824",
                "evidence_seed: 20260908",
            ]
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run_viability(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "run_viability", fake_run_viability)
    assert study.run_study(
        protocol,
        "current-20260909",
        resume=False,
        from_stage="model",
        to_stage="report",
    ) == 0
    assert captured["phase_ids"] == {"09-16", "17-21", "22"}


def test_study_resume_rejects_changed_processed_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data/robust_v2/processed/full_exact"
    data_root.mkdir(parents=True)
    airfoils = data_root / "airfoils.parquet"
    airfoils.write_bytes(b"airfoils")
    (data_root / "model_points.parquet").write_bytes(b"points")
    protocol = tmp_path / "study.yaml"
    protocol.write_text(
        "\n".join(
            [
                "schema_version: robust-v2-study-v1",
                "status: working",
                "lineage_id: current-20260910",
                "parent_lineage_id: legacy-20260826",
                "engineering_seed: 20260824",
                "evidence_seed: 20260908",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "run_viability", lambda **kwargs: 0)
    assert study.run_study(protocol, "current-20260910", resume=False) == 0
    airfoils.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed processed-data"):
        study.run_study(protocol, "current-20260910", resume=True)
