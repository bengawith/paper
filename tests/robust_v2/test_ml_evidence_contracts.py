"""Numeric software tests; these fixtures do not create aerodynamic labels."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from robust_airfoil.bounded_surface_field import FieldConfig, displacement_fields, latent_samples
from robust_airfoil.front_agreement import front_agreement
from robust_airfoil.ml_evidence import (
    finite_pool_regret,
    paired_group_bootstrap,
    selective_regression_report,
)
from robust_airfoil.ml_prediction_audit import run


def test_selective_diagnostics_preserve_missing_reference_mass_and_ties() -> None:
    report = selective_regression_report(
        reference=[0.0, np.nan, 4.0, 6.0],
        prediction=[0.1, 2.2, 4.4, 6.8],
        uncertainty=[0.1, 0.1, 0.3, 0.3],
        group_ids=["a", "b", "c", "d"],
        accepted_by_policy=[True, True, True, True],
        reference_complete=[True, False, True, True],
        tolerance=0.3,
        requested_coverages=[0.0, 0.5, 1.0],
    )
    assert report["frozen_policy"]["accepted_reference_unresolved_mass"] == pytest.approx(0.25)
    assert [row["accepted_mass"] for row in report["risk_coverage"]] == [0.0, 0.5, 1.0]
    assert report["frozen_policy"]["risk_status"] == "unresolved_references_present"
    json.dumps(report, allow_nan=False)


def test_group_bootstrap_and_finite_pool_regret_fail_closed() -> None:
    bootstrap = paired_group_bootstrap(
        [2.0, 4.0, 4.0, 4.0], [1.0, 1.0, 1.0, 1.0], ["a", "b", "b", "b"], repetitions=200
    )
    assert bootstrap["mean_difference"] == pytest.approx(2.0)
    unresolved = finite_pool_regret(["a", "b"], [1.0, 0.0], [1.0, np.nan], [True, False])
    assert unresolved["status"] == "unresolved_reference_pool"
    assert unresolved["finite_pool_regret"] is None


def test_front_agreement_retains_missing_runs_in_feasibility_fraction() -> None:
    result = front_agreement(
        [np.array([[0.0, 1.0]]), np.array([[0.0, 1.0]]), np.empty((0, 2))], np.ones(2)
    )
    assert result["feasible_fraction"] == pytest.approx(2 / 3)
    assert result["passed"] is False


def test_bounded_surface_field_has_fixed_bound_and_closed_endpoints() -> None:
    x = (1.0 - np.cos(np.linspace(0.0, np.pi, 33))) / 2.0
    config = FieldConfig(upper_lower_correlation=0.5)
    upper, lower = displacement_fields(x, latent_samples(config, 16, 11), 0.0025, config)
    assert max(np.max(np.abs(upper)), np.max(np.abs(lower))) <= 0.0025 + 1e-12
    assert np.all(upper[:, [0, -1]] == 0.0)
    assert np.all(lower[:, [0, -1]] == 0.0)


def test_prediction_audit_never_grants_scientific_go(tmp_path) -> None:
    source = tmp_path / "predictions.csv"
    output = tmp_path / "audit.json"
    row = {
        "case_id": "software-fixture-1",
        "cluster_id": "cluster-1",
        "model_id": "fixture-model",
        "model_sha256": "0" * 64,
        "protocol_hash": "1" * 64,
        "pool_role": "diagnostic",
        "target": "cl",
        "reference_value": "1.0",
        "prediction_value": "1.1",
        "uncertainty_score": "0.2",
        "accepted_by_trust": "true",
        "reference_complete": "true",
        "solver_status": "xfoil_complete",
    }
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    record = run(source, output, "cl", 0.3)
    assert record["scientific_go_granted"] is False
    assert json.loads(output.read_text())["input_sha256"]
