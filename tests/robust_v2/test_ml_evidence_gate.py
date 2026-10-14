from __future__ import annotations

import json

import robust_airfoil.pipeline as pipeline


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "robust-v2-ml-evidence-v1",
        "status": "passed",
        "ml_gate_passed": True,
        "scientific_go_granted": False,
        "model_id": "frozen-ensemble",
        "model_sha256": "a" * 64,
        "protocol_hash": "b" * 64,
        "input_sha256": "c" * 64,
        "target_metrics": {
            "cl": {
                "model_macro_error": 0.08,
                "dummy_macro_error": 0.12,
                "macro_error_improvement_vs_dummy": 1 / 3,
            },
            "log_cd": {
                "model_macro_error": 0.10,
                "dummy_macro_error": 0.15,
                "macro_error_improvement_vs_dummy": 1 / 3,
            },
        },
    }
    payload.update(overrides)
    return payload


def _pin_evidence_context(monkeypatch, tmp_path) -> None:
    manifest = tmp_path / "results/FROZEN_MODEL_MANIFEST.json"
    model_points = tmp_path / "data/robust_v2/processed/full_exact/model_points.parquet"
    manifest.parent.mkdir(parents=True)
    model_points.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    model_points.write_bytes(b"points")
    monkeypatch.setattr(pipeline, "FROZEN_MANIFEST_PATH", manifest)
    monkeypatch.setattr(
        pipeline,
        "_ACTIVE_LINEAGE_METADATA",
        {"lineage_protocol_hash": "b" * 64},
    )
    monkeypatch.setattr(
        pipeline,
        "sha256_file",
        lambda path: "a" * 64 if path == manifest else "c" * 64,
    )


def test_ml_evidence_gate_is_fail_closed_and_accepts_complete_artifact(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    _pin_evidence_context(monkeypatch, tmp_path)
    artifact = tmp_path / "reports/modelling/ml_evidence.json"
    assert pipeline._ml_evidence_gate(artifact)["status"] == "missing"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps(_payload()), encoding="utf-8")
    assert pipeline._ml_evidence_gate(artifact)["status"] == "passed"


def test_ml_evidence_gate_never_accepts_an_audit_or_insufficient_signal(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    _pin_evidence_context(monkeypatch, tmp_path)
    artifact = tmp_path / "reports/modelling/ml_evidence.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps(_payload(schema_version="ml-prediction-audit-v1")), encoding="utf-8")
    assert pipeline._ml_evidence_gate(artifact)["status"] == "invalid"
    payload = _payload()
    metrics = payload["target_metrics"]
    assert isinstance(metrics, dict)
    metrics["cl"] = {**metrics["cl"], "macro_error_improvement_vs_dummy": 0.19}
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert pipeline._ml_evidence_gate(artifact)["status"] == "invalid"


def test_report_does_not_infer_critical_results_from_invalid_upstream_evidence(
    monkeypatch, tmp_path
) -> None:
    reports_root = tmp_path / "reports"
    results_root = tmp_path / "results"
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "REPORTS_ROOT", reports_root)
    monkeypatch.setattr(pipeline, "RESULTS_ROOT", results_root)
    monkeypatch.setattr(
        pipeline, "_ml_evidence_gate", lambda _: {"status": "passed", "reason": "passed"}
    )
    calibration = results_root / "calibration/trust_calibration.json"
    calibration.parent.mkdir(parents=True)
    calibration.write_text("{}", encoding="utf-8")
    outcome, _, _ = pipeline._phase22(
        {
            "legacy_status": "passed",
            "data": {},
            "source": {},
            "model": {},
            "model_status": "failed",
            "advanced": {},
            "advanced_status": "running",
        }
    )
    blocker_codes = {blocker["code"] for blocker in outcome["blockers"]}
    assert {"model_evidence_not_current", "advanced_evidence_not_current"} <= blocker_codes
    assert "expanded_learning_signal" not in blocker_codes
    assert "no_feasible_robust_optimum" not in blocker_codes


def test_report_requires_explicit_passed_upstream_evidence(monkeypatch, tmp_path) -> None:
    reports_root = tmp_path / "reports"
    results_root = tmp_path / "results"
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "REPORTS_ROOT", reports_root)
    monkeypatch.setattr(pipeline, "RESULTS_ROOT", results_root)
    monkeypatch.setattr(
        pipeline, "_ml_evidence_gate", lambda _: {"status": "passed", "reason": "passed"}
    )
    calibration = results_root / "calibration/trust_calibration.json"
    calibration.parent.mkdir(parents=True)
    calibration.write_text("{}", encoding="utf-8")
    base_results = {
        "legacy_status": "passed",
        "data": {},
        "source": {},
        "model": {},
        "advanced": {},
    }
    for statuses in ({"model_status": "held", "advanced_status": "held"}, {}):
        outcome, _, _ = pipeline._phase22({**base_results, **statuses})
        blocker_codes = {blocker["code"] for blocker in outcome["blockers"]}
        assert {"model_evidence_not_current", "advanced_evidence_not_current"} <= blocker_codes
        assert "expanded_learning_signal" not in blocker_codes
        assert "no_feasible_robust_optimum" not in blocker_codes
