"""Create a standalone, V1-toned current paper to attach directly to Ava.

Input is ONLY the current paper/manuscript.md with its eight current figures.
No historical DOCX, review report or prior paper sections are merged.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "reports/robust_v2/submission_review_20261014"
SOURCE = REVIEW / "MANUSCRIPT_AUTHOR_REVIEW.docx"
TARGET = ROOT / "paper/LATEST_MANUSCRIPT_FOR_AVA.docx"
COPY = REVIEW / "LATEST_MANUSCRIPT_FOR_AVA.docx"
assert SOURCE.is_file(), "Regenerate the Word source from current manuscript first"
document = Document(SOURCE)
assert len(document.inline_shapes) == 8, "Expected seven main and one supplementary figure"
assert len(document.tables) >= 7, "Expected all research tables"
for section in document.sections:
    for p in section.header.paragraphs:
        p.text = "Gawith, Shahrokhi & Navaie | Aerofoil ML research"
    for p in section.footer.paragraphs:
        text = p.text
        p.text = "Current coauthor manuscript draft | October 2026"
all_text = "\n".join(p.text for p in document.paragraphs)
assert all(k in all_text for k in ["Abstract", "Conclusion", "References"])
assert "prior_final_draft" not in all_text
assert "SCIENTIFIC_REVIEW_REPORT" not in all_text
assert "author-review" not in all_text.lower()
document.save(TARGET)
shutil.copy2(TARGET, COPY)
assert hashlib.sha256(TARGET.read_bytes()).digest() == hashlib.sha256(COPY.read_bytes()).digest()
print(
    "Standalone Ava draft",
    TARGET,
    "bytes",
    TARGET.stat().st_size,
    "tables",
    len(document.tables),
    "figures",
    len(document.inline_shapes),
)
