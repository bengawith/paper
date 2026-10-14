"""Hash the exact final author-review sources, figures, evidence and CFD case manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/robust_v2/submission_review_20261014"
PAPER = ROOT / "paper"
FILES = [
    PAPER / "manuscript.md",
    PAPER / "response_to_reviewers.md",
    PAPER / "data/paper_data.json",
    OUT / "REVIEW_REPORT.md",
    OUT / "CFD_HANDOFF/CFD_CASE_MANIFEST.json",
    OUT / "CFD_HANDOFF/OPTIONAL_0P5_PERCENT/OPTIONAL_CFD_MANIFEST.json",
    OUT / "ML_DIAGNOSTICS/ml_selective_audit.json",
    OUT / "ML_DIAGNOSTICS/XFOIL_EPS005_TWO_SCRAMBLE_AUDIT.json",
    ROOT
    / "results/robust_v2/submission_review_20261014/xfoil_strict/design_verification_summary.json",
    ROOT
    / "results/robust_v2/submission_review_20261014/xfoil_eps005_n128/design_verification_summary.json",
    ROOT
    / "results/robust_v2/submission_review_20261014/xfoil_eps005_n128_b/design_verification_summary.json",
    OUT / "MANUSCRIPT_AUTHOR_REVIEW.docx",
    OUT / "SCIENTIFIC_REVIEW_REPORT.docx",
    *sorted((PAPER / "figures").glob("fig_*.png")),
]
entries = []
for p in FILES:
    b = p.read_bytes()
    entries.append(
        {
            "path": p.relative_to(ROOT).as_posix(),
            "bytes": len(b),
            "sha256": hashlib.sha256(b).hexdigest(),
        }
    )
manifest = {
    "schema": "robust-v2-final-author-review-evidence-manifest-v1",
    "date": "2026-10-14",
    "readiness": "author-review and Ava CFD handoff; independent CFD NOT complete",
    "frozen_original_lineage": "robust-v2-current-20260908",
    "recomputed_sensitivity": "two 128-scenario XFOIL scrambles at 0.5% chord",
    "modified_science_claims": "restricted to held-out historical and solver-converged selected comparisons",
    "artifacts": entries,
}
(OUT / "EDIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print("Hashed", len(entries), "final review files")
