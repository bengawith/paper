"""Manifest every author/style, manuscript, model and solver handoff component."""

import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE / "FINAL_NEW_EVIDENCE_SHA256_MANIFEST.json"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for buf in iter(lambda: f.read(1048576), b""):
            h.update(buf)
    return h.hexdigest()


sources = [
    "README.md",
    "paper/README.md",
    "paper/REVIEW_STATUS.md",
    "paper/response_to_reviewers.md",
    "paper/manuscript.md",
    "paper/LATEST_MANUSCRIPT_FOR_AVA.docx",
    "paper/MODEL_CARD.md",
    "reports/robust_v2/external_academic_review_20261021/ORIGINAL_USER_SUBMISSION.docx",
    "reports/robust_v2/external_academic_review_20261021/ORIGINAL_AUTHOR_STYLE_EVIDENCE.json",
    "reports/robust_v2/external_academic_review_20261021/PRE_REWRITE_CONTROLLED_MANUSCRIPT.md",
    "reports/robust_v2/external_academic_review_20261021/PRE_REWRITE_CONTROLLED_WORD.docx",
    "reports/robust_v2/external_academic_review_20261021/FINAL_AUTHOR_REWRITE_ACCEPTANCE.json",
    "reports/robust_v2/external_academic_review_20261021/NEW_MANUSCRIPT_NUMERIC_CLAIM_AUDIT.json",
    "reports/robust_v2/external_academic_review_20261021/REFERENCES_PRIMARY_VERIFIED_V2.csv",
    "reports/robust_v2/external_academic_review_20261021/REFERENCES_PRIMARY_VERIFIED_V2.json",
    "reports/robust_v2/external_academic_review_20261021/FINAL_MANUSCRIPT_INTEGRITY.md",
    "reports/robust_v2/external_academic_review_20261021/VISUAL_QA_FULL_AUTHOR_STYLE.json",
    "reports/robust_v2/external_academic_review_20261021/SCIENTIFIC_REVIEW_REPORT.docx",
    "reports/robust_v2/external_academic_review_20261021/RENDERED_V2/STANDALONE_AUTHOR_VOICE_FINAL.browser_preview.pdf",
    "results/robust_v2/lineages/robust-v2-controlled-20261020/DATA_MANIFEST.json",
    "results/robust_v2/lineages/robust-v2-controlled-20261020/FROZEN_MODEL_MANIFEST.json",
    "results/robust_v2/lineages/robust-v2-controlled-20261020/optimisation_production/PRODUCTION_SUMMARY.json",
    "results/robust_v2/lineages/robust-v2-controlled-20261020/nominal_control/NOMINAL_CONTROL_SUMMARY.json",
    "results/robust_v2/lineages/robust-v2-controlled-20261020/selected_study/SELECTED_XFOIL_SUMMARY.json",
    "reports/robust_v2/external_academic_review_20261014/RAW_XFOIL_EVIDENCE.zip",
]
for p in sorted((HERE / "RENDERED_V2").glob("PAGE_*.png")):
    sources.append(p.relative_to(ROOT).as_posix())
for p in sorted((HERE / "RENDERED_V2").glob("CONTACT_SHEET_*.png")):
    sources.append(p.relative_to(ROOT).as_posix())
if OUT.exists():
    raise FileExistsError("Prior evidence manifest exists")
assert len(sources) == 50, len(sources)
records = []
for name in sources:
    p = ROOT / name
    if not p.is_file():
        raise FileNotFoundError(name)
    records.append({"path": name, "bytes": p.stat().st_size, "sha256": sha(p)})
record = {
    "schema": "20261021-full-author-style-primary-reference-source-manifest-v1",
    "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "source_files": len(records),
    "full_science_reproduced": True,
    "independent_cfd_results_present": False,
    "browser_visual_preview_not_native_word": True,
    "files": records,
}
OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print("FINAL_AUTHOR_EVIDENCE_MANIFEST_FILES", len(records), flush=True)
