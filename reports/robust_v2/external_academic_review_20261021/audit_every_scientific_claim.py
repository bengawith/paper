"""Fail-closed independent evidence reconciliation for the full rewritten paper.

Every reported key numerical result is checked against actual frozen JSON
receipts, not a claimed value in a previously published document. Raw CFD
and physical manufacturing are never inferred.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PAPER = ROOT / "paper/manuscript.md"
WORD = ROOT / "paper/LATEST_MANUSCRIPT_FOR_AVA.docx"
RESULTS = ROOT / "results/robust_v2/lineages/robust-v2-controlled-20261020"
REPORT = ROOT / "reports/robust_v2/external_academic_review_20261014/CONTROLLED_MODEL"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


sources = {
    "controlled_dataset": RESULTS / "DATA_MANIFEST.json",
    "protected_test": REPORT / "REPAIRED_TEST_METRICS.json",
    "risk_calibration": REPORT / "RISK_OOF/REPAIRED_RISK_OOF.json",
    "sealed_confirmation": REPORT / "CONFIRMATION/CONFIRMATION_RESULTS.json",
    "robust_production": RESULTS / "optimisation_production/PRODUCTION_SUMMARY.json",
    "nominal_control": RESULTS / "nominal_control/NOMINAL_CONTROL_SUMMARY.json",
    "fresh_scenarios": RESULTS / "selected_study/FINAL_2048_SCENARIO_EVALUATION.json",
    "selected_xfoil": RESULTS / "selected_study/SELECTED_XFOIL_SUMMARY.json",
    "independent_selected_xfoil": ROOT
    / "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/INDEPENDENT_SELECTED_XFOIL_RECONCILIATION.json",
    "independent_robust": ROOT
    / "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/FULL_TEN_SEED_INDEPENDENT_RECONCILIATION.json",
}
D = {k: read(v) for k, v in sources.items()}
paper = PAPER.read_text(encoding="utf-8")
numbered = [int(x) for x in re.findall(r"(?m)^\[(\d+)\]", paper)]
if numbered != list(range(1, 30)):
    raise AssertionError("Bibliography not exactly 1..29")
if paper.count("![Figure ") != 7 or paper.count("![Supplementary Figure ") != 1:
    raise AssertionError("Figure index missing")
if len(re.findall(r"(?m)^\|---", paper)) != 7:
    raise AssertionError("Expected nomenclature + tables 1..6")
doc = Document(WORD)
if len(doc.inline_shapes) != 8 or len(doc.tables) != 7:
    raise AssertionError("Word visual element count differs")
source_refs = [int(x) for x in re.findall(r"(?m)^\[(\d+)\]", paper)]
ledger = list(
    csv.DictReader(
        (OUT / "REFERENCES_PRIMARY_VERIFIED_V2.csv").open(encoding="utf-8-sig", newline="")
    )
)
if [int(row["number"]) for row in ledger] != source_refs or not all(
    row["bibliography_accurate"] == "True" for row in ledger
):
    raise ValueError("Primary references did not all pass independent source checks")
claims = []


def verify(label, key, path, value, expected_display, tolerance=None):
    if isinstance(value, (int, float)) and isinstance(expected_display, (int, float)):
        if tolerance is None:
            tolerance = 5e-5
        if not math.isclose(float(value), float(expected_display), abs_tol=tolerance, rel_tol=0):
            raise AssertionError(f"{label} source result discrepancy {value} != {expected_display}")
    display = str(expected_display)
    if display not in paper:
        raise AssertionError("Paper omits audited outcome " + label + ": " + display)
    claims.append(
        {
            "claim": label,
            "source_key": key,
            "source_path": sources[path].relative_to(ROOT).as_posix(),
            "source_sha256": sha(sources[path]),
            "source_value": value,
            "paper_displayed": display,
            "status": "EXACT_SOURCE_RECONCILED",
            "limitations": "Numerical solver / finite search / observed state conditional results; not independent CFD or physical population.",
        }
    )


def expected_int(n):
    return f"{n:,}"


a, b, c, d, e, f, g, h, ind, rob = (D[k] for k in sources)
verify(
    "Geometry population",
    "requested_geometries",
    "controlled_dataset",
    a["requested_geometries"],
    "1,256",
)
verify(
    "Requested XFOIL states",
    "requested_unique_states",
    "controlled_dataset",
    a["requested_unique_states"],
    "141,928",
)
verify(
    "Observed XFOIL states",
    "observed_states",
    "controlled_dataset",
    a["observed_states"],
    "106,931",
)
verify(
    "Numerically missing states",
    "missing_requested_states",
    "controlled_dataset",
    a["missing_requested_states"],
    "34,997",
)
verify(
    "Geometries observed",
    "geometries_with_observations",
    "controlled_dataset",
    a["geometries_with_observations"],
    "1,224",
)
verify(
    "Complete polar count",
    "complete_grid_geometries",
    "controlled_dataset",
    a["complete_grid_geometries"],
    "44",
)
for name, row in {row["split"]: row for row in a["split_counts"]}.items():
    verify(
        f"{name} geometry identities",
        "split_counts/airfoils",
        "controlled_dataset",
        row["airfoils"],
        expected_int(row["airfoils"]),
    )
    verify(
        f"{name} observed states",
        "split_counts/states",
        "controlled_dataset",
        row["states"],
        expected_int(row["states"]),
    )
verify("Protected state count", "states", "protected_test", b["states"], "15,179")
verify("Trusted test states", "accepted_states", "protected_test", b["accepted_states"], "9,913")
verify("Rejected test states", "rejected_states", "protected_test", b["rejected_states"], "5,266")
for key, display in [("cl", "0.9978"), ("log_cd", "0.9818"), ("cm", "0.9882"), ("cd", "0.9444")]:
    record = next(x for x in b["metrics"] if x["subset"] == "all" and x["target"] == key)
    verify(
        "Protected test R2 " + key,
        "metrics/all/" + key + "/r2",
        "protected_test",
        record["r2"],
        float(display),
        0.00005,
    )
for key, value in [
    ("support_only", "0.753"),
    ("support_plus_ensemble", "0.889"),
    ("ensemble_only", "0.884"),
]:
    verify(
        "Calibration risk AUROC " + key,
        "models/" + key + "/auroc",
        "risk_calibration",
        c["models"][key]["auroc"],
        float(value),
        0.0006,
    )
verify(
    "Confirmation requested geometries",
    "requested_geometries",
    "sealed_confirmation",
    d["requested_geometries"],
    "200",
)
verify(
    "Confirmation requested labels",
    "requested_states",
    "sealed_confirmation",
    d["requested_states"],
    "22,600",
)
verify(
    "Confirmation observations",
    "observed_states",
    "sealed_confirmation",
    d["observed_states"],
    "17,968",
)
verify(
    "Confirmation missing labels",
    "missing_solver_observations",
    "sealed_confirmation",
    d["missing_solver_observations"],
    "4,632",
)
verify(
    "Confirmation trust accepted",
    "accepted_observed_states",
    "sealed_confirmation",
    d["accepted_observed_states"],
    "11,718",
)
for key, value in [
    ("support_only", 0.535),
    ("support_plus_ensemble", 0.802),
    ("ensemble_only", 0.821),
]:
    verify(
        "Confirmation ranking " + key,
        "risk_models/" + key + "/auroc",
        "sealed_confirmation",
        d["risk_models"][key]["auroc"],
        value,
        0.0006,
    )
assert (
    e["status"] == "FULL_PROFILE_COMPLETED"
    and len(e["per_seed"]) == 10
    and e["both_numerical_checks_met"]
)
assert f["status"] == "FULL_NOMINAL_PROFILE_COMPLETED" and len(f["per_seed"]) == 10
verify(
    "Total robust seed final feasible",
    "per_seed/feasible_entries",
    "robust_production",
    sum(x["feasible_entries"] for x in e["per_seed"]),
    "371",
)
verify(
    "Robust generations",
    "per_seed/generations",
    "robust_production",
    sum(x["generations"] for x in e["per_seed"]),
    "2,000",
)
verify(
    "Independent robust population evals",
    "aggregate_candidate_evaluations",
    "independent_robust",
    rob["aggregate_candidate_evaluations"],
    "256,000",
)
verify(
    "Reference scaled front distance",
    "reference_scaled_agreement/pairwise_objective_distance_mean",
    "robust_production",
    e["reference_scaled_agreement"]["pairwise_objective_distance_mean"],
    0.0243,
    0.00005,
)
verify(
    "Historical pooled range distance",
    "original_pooled_range_diagnostic/pairwise_objective_distance_mean",
    "robust_production",
    e["original_pooled_range_diagnostic"]["pairwise_objective_distance_mean"],
    0.1029,
    0.00005,
)
verify(
    "Strict XFOIL scheduled cases", "cases_requested", "selected_xfoil", h["cases_requested"], "819"
)
verify("Strict XFOIL admissible cases", "usable_cases", "selected_xfoil", h["usable_cases"], "290")
verify(
    "Strict XFOIL unusable cases",
    "cases_failed_or_not_admitted",
    "selected_xfoil",
    h["cases_failed_or_not_admitted"],
    "529",
)
assert ind["total_usable"] == h["usable_cases"] and ind["total_attempts"] == h["cases_requested"]
assert all(row["observed_states"] > 0 for row in [a]) if "observed_states" in a else True
assert g["scenario_count"] == 2048
for row in g["rows"]:
    if row["design"] in ("reference", "risk_knee") and row["amplitude_fraction_chord"] == 0.01:
        compare = {"reference": (0.014520, 0.018019), "risk_knee": (0.010867, 0.015445)}[
            row["design"]
        ]
        for what, v, t in zip(
            ["mean", "cvar95"],
            [row["mean_penalised_weighted_cd"], row["empirical_cvar95_penalised_weighted_cd"]],
            compare,
            strict=True,
        ):
            verify(
                "Fresh 2048 " + row["design"] + " " + what,
                "rows/sensitivity",
                "fresh_scenarios",
                v,
                t,
                0.000001,
            )
for kind, pair, n, improvement in [
    ("knee_a", "sensitivity_0p5pct_a_risk_knee", 15, 0.1518),
    ("knee_b", "sensitivity_0p5pct_b_risk_knee", 14, 0.1487),
    ("control_a", "sensitivity_0p5pct_a_nominal_control", 21, 0.1976),
    ("control_b", "sensitivity_0p5pct_b_nominal_control", 22, 0.1875),
]:
    observed = h["paired_conditional_results"][pair]
    assert observed["pairs"] == n and abs(observed["mean_reduction"] - improvement) < 0.0001
    claims.append(
        {
            "claim": "Paired selected XFOIL " + kind,
            "source_key": "paired_conditional_results/" + pair,
            "source_path": sources["selected_xfoil"].relative_to(ROOT).as_posix(),
            "source_sha256": sha(sources["selected_xfoil"]),
            "source_value": observed,
            "paper_displayed": f"{n} pairs; {100 * improvement:.1f}% approx",
            "status": "EXACT_SOURCE_RECONCILED",
            "limitations": "Only successfully paired solver cases; full complete-case missingness preserved.",
        }
    )
for name in ["nominal_risk_knee", "nominal_nominal_control", "stress_1pct_risk_knee"]:
    assert h["paired_conditional_results"][name]["pairs"] == 0
assert d["claims"]["confirmation_used_to_fit_or_select_model"] is False
assert all(
    word in paper.lower()
    for word in ["not independent", "no matched", "cfd has not yet", "measured"]
)
result = {
    "schema": "new-full-manuscript-primary-source-and-science-audit-v1",
    "checks": len(claims),
    "all_reconciled": True,
    "updated_source_sha256": sha(PAPER),
    "word_sha256": sha(WORD),
    "original_author_style_source_sha256": sha(OUT / "ORIGINAL_USER_SUBMISSION.docx"),
    "bibliography29_csv_sha256": sha(OUT / "REFERENCES_PRIMARY_VERIFIED_V2.csv"),
    "claims": claims,
    "scientific_limits": [
        "Strict XFOIL success is conditional on complete solver polars (290/819).",
        "200 sealed geometries are development-mixtures and XFOIL is shared with train labels.",
        "No independent CFD, validated manufacturing prior, physical design conclusion or author signoff is represented as completed.",
    ],
}
(OUT / "NEW_MANUSCRIPT_NUMERIC_CLAIM_AUDIT.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
print("MANUSCRIPT_CLAIMS_ALL_RECONCILED", len(claims), flush=True)
