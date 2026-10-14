"""Fail-closed evidence-to-manuscript and standalone Word acceptance checks."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "reports/robust_v2/submission_review_20261014"
PAPER = ROOT / "paper"
TEXT = (PAPER / "manuscript.md").read_text(encoding="utf-8")
NUM = json.loads((PAPER / "data/paper_data.json").read_text(encoding="utf-8"))
TRUST = json.loads(
    (REVIEW / "TRUST_INCREMENTAL/TRUST_INCREMENTAL_CV.json").read_text(encoding="utf-8")
)
CORRECTED = json.loads(
    (REVIEW / "CORRECTED_OPTIMISATION/CORRECTED_SELECTED_DESIGNS.json").read_text(encoding="utf-8")
)
SMOKE = json.loads(
    (REVIEW / "CORRECTED_OPTIMISATION/CORRECTED_SMOKE_FRONT.json").read_text(encoding="utf-8")
)
BIB = json.loads(
    (REVIEW / "REFERENCES/PRIMARY_SOURCE_REFERENCE_LEDGER.json").read_text(encoding="utf-8")
)
DOC = PAPER / "LATEST_MANUSCRIPT_FOR_AVA.docx"
assert DOC.is_file()
D = Document(DOC)
assert len(D.inline_shapes) == 8, f"figures {len(D.inline_shapes)}"
assert len(D.tables) == 7, f"tables {len(D.tables)}"
main = "\n".join(x.text for x in D.paragraphs)
assert "Trust-Calibrated Machine-Learning Surrogates" in main
assert "Independent CFD" in main or "independent CFD" in main
assert "0.767" in main and "0.866" in main
assert "0.149" in main
assert "92 and 96" in main
assert "0.283" in main
assert "prior_final_draft" not in main
assert "scientific review report" not in main.lower()
assert "response to reviewers" not in main.lower()
assert "MANUSCRIPT_AUTHOR_REVIEW" not in main
assert len(re.findall(r"(?m)^\[\d+\] ", TEXT)) == 28
assert len(BIB["entries"]) == 28
assert TRUST["schema"] == "trust-calibration-group-oof-all-fit-is-train-only-v2"
assert TRUST["n_geometries"] == 234
assert len(TRUST["folds"]) == 5
assert 0.069 < TRUST["cluster_bootstrap_95pct_interval"][0] < 0.071
assert 0.138 < TRUST["cluster_bootstrap_95pct_interval"][1] < 0.140
assert abs(TRUST["incremental_auroc"] - 0.09916) < 0.001
assert SMOKE["agreement"]["agreement_status"] == "pass"
assert SMOKE["agreement"]["feasible_run_count"] == 10
assert SMOKE["protocol"]["generations"] == 10
assert SMOKE["protocol"]["population"] == 32
assert SMOKE["protocol"]["uncertainty_samples"] == 16
assert len(CORRECTED["results"]) == 5
assert "historic" in CORRECTED["note"].lower()
assert NUM["design_verification"]["designs"]["within_trust_knee"]["shared_converged"] == 13
assert NUM["design_verification"]["designs"]["reference_naca2412"]["shared_converged"] == 13
assert "confidence" in TEXT.lower()
assert (PAPER / "figures/fig_pareto_front.png").stat().st_size > 50_000
manifest = {
    "status": "PASSED_AUTHOR_REVIEW_ACCEPTANCE_NOT_CFD_OR_FULL_PRODUCTION",
    "schema": "robust-v2-author-acceptance-v1",
    "canonical_manuscript_md_sha256": hashlib.sha256(
        (PAPER / "manuscript.md").read_bytes()
    ).hexdigest(),
    "ava_docx_sha256": hashlib.sha256(DOC.read_bytes()).hexdigest(),
    "docx_tables": len(D.tables),
    "docx_embedded_figures": len(D.inline_shapes),
    "bibliography_entries": len(BIB["entries"]),
    "trust_oof_delta_auroc": TRUST["incremental_auroc"],
    "trust_oof_delta_cluster_interval": TRUST["cluster_bootstrap_95pct_interval"],
    "corrected_smoke_front_distance": SMOKE["agreement"]["pairwise_objective_distance_mean"],
    "corrected_production_optimisation_completed": False,
    "cfd_completed": False,
    "limitations": "No accepted full-budget Pareto front, no CFD/experiment, manufacture prior unmeasured, XFOIL failed cases excluded.",
}
(REVIEW / "FINAL_AUTHOR_ACCEPTANCE.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2))
