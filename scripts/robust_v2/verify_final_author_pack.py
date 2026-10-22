"""Fail-closed acceptance of the actual controlled scientific-review handoff.

Review completeness is separate from journal/physical acceptance. Missing PDF
or uninspected Word pages are not silently replaced by document XML checks.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from docx import Document

from robust_airfoil.modelling.freeze import verify_frozen_artifacts

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/robust_v2/external_academic_review_20261014"
CONTROL = BASE / "CONTROLLED_MODEL"
LINEAGE = ROOT / "results/robust_v2/lineages/robust-v2-controlled-20261020"
PAPER = ROOT / "paper"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def require(path):
    if not path.is_file():
        raise FileNotFoundError("Missing author-review evidence: " + str(path))
    return path


def verify():
    # One authoritative acceptance for the focused paper and current CFD handoff.
    focused = ROOT / "reports/robust_v2/final_release_20261022/FOCUSED_AUTHOR_REVIEW_ACCEPTANCE.json"
    if focused.exists():
        from runpy import run_path

        new_gate = ROOT / "scripts/robust_v2/verify_focused_review.py"
        return run_path(str(new_gate))["verify"]()
    # Current 22-October author revision supersedes earlier manuscript hashes.
    # Reuse its authoritative read-only verifier; never mutate older receipts.
    current = ROOT / "reports/robust_v2/final_release_20261022/FINAL_REVIEW_ACCEPTANCE.json"
    if current.is_file():
        from runpy import run_path

        runner = ROOT / "scripts/robust_v2/verify_coauthor_draft.py"
        run_path(str(runner))["main"]()
        return json.loads(current.read_text(encoding="utf-8"))
    # The 14-October evidence hashes certify its historical manuscript, not
    # the replacement 21-October original-author-voice paper. Dispatch to
    # the latest source/Word/29-reference/46-claim acceptance without ever
    # rewriting or relabelling the frozen historical review receipts.
    latest_dir = ROOT / "reports/robust_v2/external_academic_review_20261021"
    latest = latest_dir / "FINAL_AUTHOR_REWRITE_ACCEPTANCE.json"
    if latest.is_file():
        from runpy import run_path

        run_path(str(latest_dir / "verify_full_rewrite_acceptance.py"))["main"]()
        model_path = LINEAGE / "FROZEN_MODEL_MANIFEST.json"
        frozen = json.loads(require(model_path).read_text(encoding="utf-8"))
        verify_frozen_artifacts(
            model_path, {name: Path(path) for name, path in frozen["artifact_paths"].items()}
        )
        controller = json.loads(
            require(BASE / "CONTINUATION_20261020/CONTROLLED_PIPELINE_STATUS_R4.json").read_text(encoding="utf-8")
        )
        if controller["status"] != "ALL_EXECUTED_GATES_PASS_NOT_INDEPENDENT_CFD_OR_AUTHOR_SUBMISSION":
            raise ValueError("Full controlled numerical pipeline has no terminal passing receipt")
        return json.loads(latest.read_text(encoding="utf-8"))
    source = require(PAPER / "manuscript.md").read_text(encoding="utf-8")
    doc = Document(require(PAPER / "LATEST_MANUSCRIPT_FOR_AVA.docx"))
    word = "\n".join(p.text for p in doc.paragraphs)
    if len(doc.tables) != 7 or len(doc.inline_shapes) != 8:
        raise ValueError("Standalone paper must contain seven tables and eight embedded figures")
    if source.splitlines()[0].removeprefix("# ") not in word:
        raise ValueError("Standalone Word title differs from canonical manuscript")
    if "1,001" not in word or "CFD" not in word or "XFOIL" not in word:
        raise ValueError("Material scientific limits missing from Word")
    if re.search(r"\{[a-z][a-z0-9_]*\}", source):
        raise ValueError("Unresolved scientific-data template placeholder in manuscript")
    if "prior_final_draft" in word or "review instructions" in word.lower():
        raise ValueError("Standalone Word includes historical or internal material")
    refs = re.findall(r"(?m)^\[(\d+)\] ", source)
    if [int(x) for x in refs] != list(range(1, 30)):
        raise ValueError("Expected 29 individually reviewed primary reference entries")
    with require(BASE / "REFERENCES_VERIFIED.csv").open(encoding="utf-8-sig", newline="") as f:
        bibliography = list(csv.DictReader(f))
    if len(bibliography) != 29 or not all(
        row["verification_status"] and row["support_assessment"] for row in bibliography
    ):
        raise ValueError("Primary literature ledger incomplete")
    with require(BASE / "CLAIM_EVIDENCE_MATRIX.csv").open(encoding="utf-8-sig", newline="") as f:
        claims = list(csv.DictReader(f))
    if len(claims) < 40:
        raise ValueError("Original audit and all controlled source claims must be traced")
    for item in claims:
        path = (ROOT / item["source_path"]).resolve()
        if (
            not path.is_relative_to(ROOT.resolve())
            or not path.is_file()
            or sha(path) != item["source_sha256"]
        ):
            raise ValueError(
                "Missing or changed scientific claim source: " + str(item["source_path"])
            )
    evidence = json.loads(
        require(BASE / "EVIDENCE_SHA256_MANIFEST.json").read_text(encoding="utf-8")
    )
    for item in evidence["files"]:
        path = (ROOT / item["relative_path"]).resolve()
        if (
            not path.is_file()
            or sha(path) != item["sha256"]
            or path.stat().st_size != item["bytes"]
        ):
            raise ValueError(
                "Invalid scientific source evidence manifest item: " + item["relative_path"]
            )
    cohort = json.loads(require(LINEAGE / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    if cohort["requested_geometries"] != 1256 or cohort["observed_states"] != 106931:
        raise ValueError("Controlled corpus changed from accepted source-bound numerics")
    manifest_path = LINEAGE / "FROZEN_MODEL_MANIFEST.json"
    frozen = json.loads(require(manifest_path).read_text(encoding="utf-8"))
    verify_frozen_artifacts(
        manifest_path, {name: Path(path) for name, path in frozen["artifact_paths"].items()}
    )
    test = json.loads(require(CONTROL / "REPAIRED_TEST_METRICS.json").read_text(encoding="utf-8"))
    confirmation = json.loads(
        require(CONTROL / "CONFIRMATION/CONFIRMATION_RESULTS.json").read_text(encoding="utf-8")
    )
    production = json.loads(
        require(LINEAGE / "optimisation_production/PRODUCTION_SUMMARY.json").read_text(
            encoding="utf-8"
        )
    )
    nominal = json.loads(
        require(LINEAGE / "nominal_control/NOMINAL_CONTROL_SUMMARY.json").read_text(
            encoding="utf-8"
        )
    )
    selected = json.loads(
        require(LINEAGE / "selected_study/SELECTED_XFOIL_SUMMARY.json").read_text(encoding="utf-8")
    )
    if not test["model_frozen_before_test"] or test["states"] <= 1000:
        raise ValueError("Actual controlled ensemble evaluation missing")
    if (
        confirmation["requested_geometries"] != 200
        or confirmation["claims"]["confirmation_used_to_fit_or_select_model"]
    ):
        raise ValueError("Sealed confirmation was not prospective with respect to fitting")
    if (
        production["status"] != "FULL_PROFILE_COMPLETED"
        or len(production["per_seed"]) != 10
        or any(p["generations"] != 200 for p in production["per_seed"])
    ):
        raise ValueError("Full robust 128x200x64x10 scientific gate incomplete")
    if nominal["status"] != "FULL_NOMINAL_PROFILE_COMPLETED" or len(nominal["per_seed"]) != 10:
        raise ValueError("Full nominal 128x200x10 control incomplete")
    if (
        selected["cases_requested"]
        != selected["usable_cases"] + selected["cases_failed_or_not_admitted"]
    ):
        raise ValueError("Selected XFOIL failed-case accounting incomplete")
    controller = json.loads(
        require(BASE / "CONTINUATION_20261020/CONTROLLED_PIPELINE_STATUS_R4.json").read_text(
            encoding="utf-8"
        )
    )
    if controller["status"] != "ALL_EXECUTED_GATES_PASS_NOT_INDEPENDENT_CFD_OR_AUTHOR_SUBMISSION":
        raise ValueError("Some scientific or full software tests are not terminal/passing")
    for name in [
        "ruff",
        "mypy",
        "full_tests",
        "evaluation",
        "risk_oof",
        "confirmation_freeze",
        "confirmation_solver",
        "confirmation_analysis",
        "production",
        "nominal_control",
    ]:
        if controller["stages"].get(name, {}).get("status") != "PASSED":
            raise ValueError("Missing executed gate: " + name)
    visual = json.loads(require(BASE / "VISUAL_QA.json").read_text(encoding="utf-8"))
    if not visual.get("all_pages_visually_inspected") or visual.get(
        "manuscript_docx_sha256"
    ) != sha(PAPER / "LATEST_MANUSCRIPT_FOR_AVA.docx"):
        raise ValueError("Word layout not visually verified on the exact released document")
    return {
        "status": "REVIEW_DELIVERABLES_ACCEPTED_MAJOR_REVISIONS_BEFORE_SUBMISSION",
        "review_deliverables_complete": True,
        "software_checks": "PASS",
        "controlled_model_training": "COMPLETE",
        "controlled_holdout": "COMPLETE",
        "sealed_confirmation": "COMPLETE",
        "full_robust_128_200_64_10": "COMPLETE",
        "full_nominal_128_200_10": "COMPLETE",
        "source_claims_traced": len(claims),
        "bibliography_entries_reviewed": len(bibliography),
        "embedded_figures": len(doc.inline_shapes),
        "tables": len(doc.tables),
        "selected_xfoil_requested": selected["cases_requested"],
        "selected_xfoil_usable": selected["usable_cases"],
        "numerical_agreement_both_checks_met": production["both_numerical_checks_met"],
        "independent_cfd": "NOT_PERFORMED",
        "journal_submission_ready": False,
        "manufacturing_tolerances": "PRESCRIBED_NOT_MEASURED",
        "coauthor_approval": "NOT_CLAIMED",
        "canonical_manuscript_sha256": sha(PAPER / "manuscript.md"),
        "standalone_docx_sha256": sha(PAPER / "LATEST_MANUSCRIPT_FOR_AVA.docx"),
    }


if __name__ == "__main__":
    result = verify()
    # Never rewrite the 14-October sealed historical author receipt with a
    # subsequent manuscript revision's checksums or scientific conclusions.
    if not (ROOT / "reports/robust_v2/external_academic_review_20261021/FINAL_AUTHOR_REWRITE_ACCEPTANCE.json").exists():
        (BASE / "FINAL_ACCEPTANCE.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2))
