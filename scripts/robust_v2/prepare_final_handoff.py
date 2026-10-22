"""Build one small author-facing note and the complete new raw comparator archive.

The older training/XFOIL archive remains a separate, unchanged artifact.
"""

from __future__ import annotations

import hashlib
import json
import runpy
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/robust_v2/final_release_20261022"
RAW = BASE / "NEW_NUMERICAL_COMPARISON_RAW.zip"
INDEX = BASE / "FINAL_HANDOFF_EVIDENCE_INDEX.json"
REPORT = BASE / "CURRENT_SCIENTIFIC_REVIEW_REPORT.docx"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1048576), b""):
            h.update(data)
    return h.hexdigest()


def main() -> None:
    if any(path.exists() for path in (RAW, INDEX, REPORT)):
        raise FileExistsError("Current evidence release was already assembled")
    exporter = runpy.run_path(str(ROOT / "scripts/robust_v2/export_review_docx_20261014.py"))[
        "export"
    ]
    exporter(ROOT / "paper/COAUTHOR_NOTES.md", REPORT)
    items = []
    for pattern in (
        "EXTERNAL_NACA_FAMILY/geometry/*.dat",
        "EXTERNAL_NACA_FAMILY/solver/**/*",
        "EXTERNAL_NACA_FAMILY/before_label_model_predictions.parquet",
        "NEURALFOIL_FULL_CONFIRMATION/*.jsonl",
        "NEURALFOIL_COMPARISON/*.jsonl",
    ):
        items.extend(path for path in BASE.glob(pattern) if path.is_file())
    items = sorted(set(items))
    if len(items) < 440:
        raise ValueError("One or more original numerical comparison raw files are missing")
    with zipfile.ZipFile(RAW, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        for path in items:
            archive.write(path, arcname=path.relative_to(BASE).as_posix())
    with zipfile.ZipFile(RAW) as archive:
        if archive.testzip() is not None:
            raise ValueError("New numerical archive CRC failed")
        if len(archive.namelist()) != len(items):
            raise ValueError("An original comparator file was lost in packing")
    files = [
        ROOT / "paper/manuscript.md",
        ROOT / "paper/LATEST_MANUSCRIPT_FOR_AVA.docx",
        ROOT / "paper/SUPPLEMENTARY_NUMERICAL_COMPARISON.md",
        ROOT / "paper/SUPPLEMENTARY_NUMERICAL_COMPARISON.docx",
        ROOT / "paper/COAUTHOR_NOTES.md",
        ROOT / "paper/README.md",
        ROOT / "paper/REVIEW_STATUS.md",
        REPORT,
        ROOT
        / "reports/robust_v2/external_academic_review_20261021/REFERENCES_PRIMARY_VERIFIED_V2.csv",
        ROOT
        / "reports/robust_v2/external_academic_review_20261021/FINAL_AUTHOR_REWRITE_ACCEPTANCE.json",
        ROOT / "reports/robust_v2/external_academic_review_20261014/RAW_XFOIL_EVIDENCE.zip",
        BASE / "FINAL_REVIEW_ACCEPTANCE.json",
        BASE / "LATEST_WORD_VISUAL_RECEIPT.json",
        BASE / "EXTERNAL_NACA_FAMILY/NEW_FAMILY_RESULTS.json",
        BASE / "EXTERNAL_NACA_FAMILY/INDEPENDENT_RECEIPT_AND_TRUST_AUDIT.json",
        BASE / "NEURALFOIL_FULL_CONFIRMATION/RESULTS.json",
        BASE / "NEURALFOIL_COMPARISON/BENCHMARK_RESULT.json",
        RAW,
    ]
    records = []
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(str(path))
        records.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha(path),
            }
        )
    index = {
        "schema": "robust-v2-20261022-minimal-source-and-review-evidence-v1",
        "status": "COAUTHOR_REVIEW_READY_CFD_PENDING",
        "only_current_manuscript": "paper/LATEST_MANUSCRIPT_FOR_AVA.docx",
        "only_current_comparison_supplement": "paper/SUPPLEMENTARY_NUMERICAL_COMPARISON.docx",
        "raw_new_numerical_comparison_cases": len(items),
        "new_raw_comparison_archive_sha256": sha(RAW),
        "native_word_render_claim": False,
        "files": records,
    }
    INDEX.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("FINAL_COMPACT_INDEX", len(records), "new raw archive files", len(items), flush=True)


if __name__ == "__main__":
    main()
