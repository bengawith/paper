"""Create one verified author-review and CFD handoff package on SamsPC.

Does not overwrite or remove historical archived review package contents.
The new dated subfolders are generated only when absent or when explicitly
setting REPLACE_REVIEW_PACKAGE=1. All source evidence is preserved.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "reports/robust_v2/submission_review_20261014"
PACKAGE_NAME = "READY_FOR_REVIEW"
DESTS = [Path(r"C:\Users\Ben\OneDrive\paper") / PACKAGE_NAME]
FILES = [
    ("README.md", ROOT / "paper/README.md"),
    (
        "HISTORICAL/prior_final_draft.docx",
        Path(r"C:\Users\Ben\OneDrive\paper\prior_final_draft.docx"),
    ),
    ("HISTORICAL/previous_source_snapshot.zip", Path(r"C:\Users\Ben\OneDrive\paper\source_snapshot.zip")),
    ("HISTORICAL/previous_repository_history.bundle", Path(r"C:\Users\Ben\OneDrive\paper\repository_history.bundle")),
    ("HISTORICAL/previous_local_evidence_and_models.zip", Path(r"C:\Users\Ben\OneDrive\paper\local_evidence_and_models.zip")),
    ("HISTORICAL/previous_evidence_inventory.json", Path(r"C:\Users\Ben\OneDrive\paper\evidence_inventory.json")),
    ("READ_THIS_FIRST.md", REVIEW / "READ_THIS_FIRST.md"),
    ("EMAIL_TO_AVA_DRAFT.txt", REVIEW / "EMAIL_TO_AVA_DRAFT.txt"),
    ("COPY_TO_LAPTOP_OFFLINE.ps1", REVIEW / "COPY_TO_LAPTOP_OFFLINE.ps1"),
    ("LATEST_MANUSCRIPT_FOR_AVA.docx", ROOT / "paper/LATEST_MANUSCRIPT_FOR_AVA.docx"),
    ("SCIENTIFIC_REVIEW_REPORT.docx", REVIEW / "SCIENTIFIC_REVIEW_REPORT.docx"),
    ("paper/manuscript.md", ROOT / "paper/manuscript.md"),
    ("paper/response_to_reviewers.md", ROOT / "paper/response_to_reviewers.md"),
    ("paper/data/paper_data.json", ROOT / "paper/data/paper_data.json"),
    ("EVIDENCE/REVIEW_REPORT.md", REVIEW / "REVIEW_REPORT.md"),
    ("EVIDENCE/FINAL_RESEARCH_RESOLUTION.md", REVIEW / "FINAL_RESEARCH_RESOLUTION.md"),
    ("EVIDENCE/FINAL_AUTHOR_ACCEPTANCE.json", REVIEW / "FINAL_AUTHOR_ACCEPTANCE.json"),
    (
        "EVIDENCE/FINAL_SCIENTIFIC_RECONCILIATION.json",
        REVIEW / "FINAL_SCIENTIFIC_RECONCILIATION.json",
    ),
    ("EVIDENCE/TRUST_INCREMENTAL_CV.json", REVIEW / "TRUST_INCREMENTAL/TRUST_INCREMENTAL_CV.json"),
    (
        "EVIDENCE/CORRECTED_SELECTED_DESIGNS.json",
        REVIEW / "CORRECTED_OPTIMISATION/CORRECTED_SELECTED_DESIGNS.json",
    ),
    (
        "EVIDENCE/CORRECTED_SMOKE_FRONT.json",
        REVIEW / "CORRECTED_OPTIMISATION/CORRECTED_SMOKE_FRONT.json",
    ),
    (
        "EVIDENCE/CORRECTED_TEN_SEED_SMOKE_SUMMARY.json",
        REVIEW / "CORRECTED_OPTIMISATION/TEN_SEED_SMOKE_SUMMARY.json",
    ),
    (
        "EVIDENCE/PRIMARY_SOURCE_REFERENCE_LEDGER.json",
        REVIEW / "REFERENCES/PRIMARY_SOURCE_REFERENCE_LEDGER.json",
    ),
    ("EVIDENCE/BIBLIOGRAPHY_AUDIT.json", REVIEW / "REFERENCES/BIBLIOGRAPHY_AUDIT.json"),
    ("EVIDENCE/TONE_REVIEW_NOTE.md", REVIEW / "TONE_REVIEW_NOTE.md"),
    ("EVIDENCE/TONE_POLISH_MANIFEST.json", REVIEW / "TONE_POLISH_MANIFEST.json"),
    ("EVIDENCE/NUMERICAL_RECONCILIATION.json", REVIEW / "NUMERICAL_RECONCILIATION.json"),
    ("EVIDENCE/XFOIL_ARCHIVE_COVERAGE_AUDIT.json", REVIEW / "XFOIL_ARCHIVE_COVERAGE_AUDIT.json"),
    (
        "EVIDENCE/STRICT_XFOIL_DESIGN_SUMMARY.json",
        ROOT
        / "results/robust_v2/submission_review_20261014/xfoil_strict/design_verification_summary.json",
    ),
    ("EVIDENCE/BASELINE_MANUSCRIPT_PRE_REVIEW.md", REVIEW / "BASELINE_manuscript.md"),
    ("EVIDENCE/BASELINE_PAPER_DATA_PRE_REVIEW.json", REVIEW / "BASELINE_paper_data.json"),
    ("EVIDENCE/EDIT_MANIFEST.json", REVIEW / "EDIT_MANIFEST.json"),
    ("EVIDENCE/ML_SELECTIVE_AUDIT.json", REVIEW / "ML_DIAGNOSTICS/ml_selective_audit.json"),
    ("EVIDENCE/ML_SELECTIVE_CURVE.csv", REVIEW / "ML_DIAGNOSTICS/selective_coverage.csv"),
    (
        "EVIDENCE/XFOIL_005_TWO_SCRAMBLE.json",
        REVIEW / "ML_DIAGNOSTICS/XFOIL_EPS005_TWO_SCRAMBLE_AUDIT.json",
    ),
]
for f in (ROOT / "paper/figures").glob("fig_*.png"):
    FILES.append((f"paper/figures/{f.name}", f))
for seed_file in sorted((REVIEW / "CORRECTED_OPTIMISATION/smoke_seeds").glob("*.json")):
    FILES.append((f"EVIDENCE/CORRECTED_SEEDS/{seed_file.name}", seed_file))
CFD = REVIEW / "CFD_HANDOFF"
for file in CFD.rglob("*"):
    if file.is_file():
        FILES.append((f"CFD_HANDOFF/{file.relative_to(CFD).as_posix()}", file))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make(target: Path) -> None:
    if target.exists():
        if os.getenv("REPLACE_REVIEW_PACKAGE", "") != "1":
            raise FileExistsError(f"Review package already exists; refusing overwrite: {target}")
        # OneDrive may hold file handles; replace listed files in place and
        # re-hash rather than deleting the entire review directory.
        pass
    target.mkdir(parents=True, exist_ok=True)
    # On in-place refresh remove *only* files from earlier manifests that
    # are not part of this package. Never delete unknown user files.
    old_manifest = target / "SHA256_MANIFEST.json"
    if old_manifest.exists() and os.getenv("REPLACE_REVIEW_PACKAGE") == "1":
        previous = json.loads(old_manifest.read_text(encoding="utf-8"))
        wanted = {rel for rel, _ in FILES} | {"CFD_HANDOFF_ONLY.zip"}
        for previous_file in previous.get("files", []):
            rel = str(previous_file["path"]).replace("\\", "/")
            if rel not in wanted:
                extra = target / rel
                if extra.is_file():
                    extra.unlink()
    for rel, source in FILES:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = target / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    # Separate Ava-only archive of CFD inputs and protocol.
    cfd_zip = target / "CFD_HANDOFF_ONLY.zip"
    with zipfile.ZipFile(cfd_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in sorted((target / "CFD_HANDOFF").rglob("*")):
            if f.is_file():
                zf.write(f, arcname=f.relative_to(target).as_posix())
    members = [
        {"path": f.relative_to(target).as_posix(), "bytes": f.stat().st_size, "sha256": sha(f)}
        for f in sorted(target.rglob("*"))
        if f.is_file() and f.name != "SHA256_MANIFEST.json"
    ]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "schema": "robust-v2-author-review-package-20261014-v1",
        "date": "2026-10-14",
        "source_git_commit": commit,
        "scientific_status": "author_review_pre_CFD_not_journal_submission_ready",
        "files": members,
    }
    (target / "SHA256_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with zipfile.ZipFile(target / "CFD_HANDOFF_ONLY.zip") as zf:
        assert zf.testzip() is None
    for item in members:
        file = target / item["path"]
        assert file.stat().st_size == item["bytes"] and sha(file) == item["sha256"]
    print(target, "files", len(members), "verified")
    print("  CFD zip bytes:", cfd_zip.stat().st_size)


if __name__ == "__main__":
    for dest in DESTS:
        make(dest)
    assert (DESTS[0] / "SHA256_MANIFEST.json").is_file()
    print("One canonical review package:", DESTS[0])
