"""Terminal fail-closed full scientific writing/reference/Word acceptance."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "paper/manuscript.md"
WORD = ROOT / "paper/LATEST_MANUSCRIPT_FOR_AVA.docx"
OUT = HERE / "FINAL_AUTHOR_REWRITE_ACCEPTANCE.json"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for x in iter(lambda: f.read(1048576), b""):
            h.update(x)
    return h.hexdigest()


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def references(body):
    refs = set()
    for group in re.findall(r"\[([0-9,\s\u2013-]+)\]", body):
        for part in group.split(","):
            part = part.strip().replace("-", "–")
            if not part:
                continue
            if "–" in part:
                ends = [int(x) for x in part.split("–")]
                if len(ends) != 2 or ends[0] > ends[1]:
                    raise ValueError("Invalid citation range " + part)
                refs.update(range(ends[0], ends[1] + 1))
            else:
                refs.add(int(part))
    return refs


def main():
    # Idempotent verification: keep the original receipt immutable, and
    # compare the recomputed full acceptance against its exact contents.
    prior_receipt = load(OUT) if OUT.exists() else None
    text = SOURCE.read_text(encoding="utf-8")
    if len(text.split()) < 9000:
        raise AssertionError("Full author-style rewrite missing")
    before = text.split("## References\n")
    if len(before) != 2:
        raise AssertionError("Reference section should appear exactly once")
    numbers = [int(v) for v in re.findall(r"(?m)^\[(\d+)\] ", before[1])]
    if numbers != list(range(1, 30)):
        raise AssertionError("One or more of the 29 real publications missing")
    citations = references(before[0])
    if citations != set(numbers):
        raise AssertionError(f"Missing or extraneous in-text citation {set(numbers) ^ citations}")
    primary = list(
        csv.DictReader(
            (HERE / "REFERENCES_PRIMARY_VERIFIED_V2.csv").open(encoding="utf-8-sig", newline="")
        )
    )
    if len(primary) != 29 or any(row["bibliography_accurate"] != "True" for row in primary):
        raise AssertionError("An original primary reference has not been verified")
    if primary[8]["verification_state"] != "AUTHORS_ORIGINAL_PRIMARY_PDF_VERIFIED":
        raise AssertionError(
            "Kulfan title must be sourced to original not misspelled Crossref metadata"
        )
    numeric = load(HERE / "NEW_MANUSCRIPT_NUMERIC_CLAIM_AUDIT.json")
    if (
        not numeric["all_reconciled"]
        or numeric["checks"] < 46
        or numeric["updated_source_sha256"] != sha(SOURCE)
    ):
        raise AssertionError("The exact numerical claim ledger no longer matches this rewrite")
    if numeric["word_sha256"] != sha(WORD):
        raise AssertionError("Word layout revision did not propagate into science ledger")
    qa = load(HERE / "VISUAL_QA_FULL_AUTHOR_STYLE.json")
    if qa["word_sha256"] != sha(WORD) or qa["manuscript_sha256"] != sha(SOURCE):
        raise AssertionError("Page evidence does not match final source and Word")
    if (
        not qa["all_21_pages_visually_inspected"]
        or qa["pages"] != 21
        or not qa["captions_grouped_with_images"]
    ):
        raise AssertionError("Visual review not complete")
    if qa["native_microsoft_word_pagination_verified"]:
        raise AssertionError("Native Word pagination was not verified")
    doc = Document(WORD)
    if (
        len(doc.tables) != 7
        or len(doc.inline_shapes) != 8
        or len(re.findall(r"(?m)^\|---", text)) != 7
    ):
        raise AssertionError("Word text, figures or table identity changed")
    if text.count("![Figure ") != 7 or text.count("![Supplementary Figure") != 1:
        raise AssertionError("Required eight source figures missing")
    for p in doc.paragraphs:
        if (
            p.text.startswith(("Figure ", "Supplementary Figure "))
            and not p.paragraph_format.keep_with_next
        ):
            raise AssertionError("Figure caption can be separated from its image")
    style = load(HERE / "ORIGINAL_AUTHOR_STYLE_EVIDENCE.json")
    original = HERE / "ORIGINAL_USER_SUBMISSION.docx"
    if style["source_sha256"] != sha(original):
        raise AssertionError("Style source bytes differ from original submission")
    if style["previous_controlled_source_sha256"] != sha(
        HERE / "PRE_REWRITE_CONTROLLED_MANUSCRIPT.md"
    ):
        raise AssertionError("Replaced manuscript baseline was modified")
    raw = ROOT / "reports/robust_v2/external_academic_review_20261014/RAW_XFOIL_EVIDENCE.zip"
    if not raw.is_file() or raw.stat().st_size < 100_000_000:
        raise AssertionError("Raw evidence archive missing")
    required_sections = [
        "## 1. Introduction",
        "## 2. Aerofoil representation",
        "## 3. Data-driven",
        "## 4. Uncertainty",
        "## 5. Results",
        "## 6. Discussion",
        "## 7. Independent CFD verification protocol",
        "## 8. Numerical findings",
    ]
    if any(section not in text for section in required_sections):
        raise AssertionError("Paper contains unresolved content gap")
    if "CFD has not yet been performed" not in text or "no matched" not in text.lower():
        raise AssertionError("Paper conceals missing independent solver or nominal strict result")
    evidence = [
        SOURCE,
        WORD,
        HERE / "REFERENCES_PRIMARY_VERIFIED_V2.csv",
        HERE / "NEW_MANUSCRIPT_NUMERIC_CLAIM_AUDIT.json",
        HERE / "VISUAL_QA_FULL_AUTHOR_STYLE.json",
        HERE / "FINAL_MANUSCRIPT_INTEGRITY.md",
        HERE / "ORIGINAL_AUTHOR_STYLE_EVIDENCE.json",
        ROOT / "paper/README.md",
        ROOT / "paper/REVIEW_STATUS.md",
    ]
    receipt = {
        "schema": "new-20261021-full-rewrite-terminal-acceptance-v1",
        "status": "COMPLETE_NUMERICAL_PAPER_READY_FOR_COAUTHOR_REVIEW",
        "writing_style": "Independently matched to the users original 340-paragraph Word submission",
        "original_author_docx_sha256": sha(original),
        "total_words": len(text.split()),
        "verified_numbered_references": 29,
        "all_29_references_cited": True,
        "primary_literature_crosschecked": True,
        "numeric_claims_reconciled": numeric["checks"],
        "numerical_solver_failures_unchanged": True,
        "physical_independent_cfd_results": "NOT_AVAILABLE",
        "engineering_final_superiority_claim": "WITHHELD_PENDING_CFD",
        "coauthor_approvals": "NOT_CLAIMED",
        "tables": len(doc.tables),
        "figures": len(doc.inline_shapes),
        "browser_preview_pages_inspected": qa["pages"],
        "browser_only_not_native_word_visual_qa": True,
        "next_scientific_step": "Run independent CFD fixed-shape matched-lift study and then write actual CFD physical conclusion",
        "files_sha256": {
            p.relative_to(ROOT).as_posix(): sha(p) for p in evidence if p.is_relative_to(ROOT)
        },
        "limitations": [
            "The 819 selected XFOIL observations include 529 strict unusable cases; no nominal matched pairs.",
            "Numerical evidence alone does not validate manufacturing tolerances or independent fluid physics.",
            "Authors must approve contribution, funding/conflicts and journal submission statements.",
        ],
    }
    if prior_receipt is not None:
        if receipt != prior_receipt:
            raise ValueError("Recomputed author-review result differs from its frozen receipt")
    else:
        OUT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "FULL_AUTHOR_STYLE_REWRITE_ALL_SCIENCE_AND_REFERENCES_ACCEPTED",
        receipt["total_words"],
        flush=True,
    )


if __name__ == "__main__":
    main()
