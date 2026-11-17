"""One canonical, source-bound focused coauthor-review gate.

Numerical research complete does not mean physical CFD completed, journal accepted,
manufacturing statistics validated, or human declarations approved.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

from docx import Document

from robust_airfoil.modelling.freeze import verify_frozen_artifacts

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "paper"
AUDIT = ROOT / "reports/robust_v2/final_release_20261022"
OLD = ROOT / "reports/robust_v2/external_academic_review_20261021"
DATA = ROOT / "results/robust_v2/lineages/robust-v2-controlled-20261020"
DEST = AUDIT / "FOCUSED_AUTHOR_REVIEW_ACCEPTANCE.json"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for x in iter(lambda: f.read(1048576), b""):
            h.update(x)
    return h.hexdigest()


def js(p):
    return json.loads(p.read_text(encoding="utf-8"))


def references(body):
    names = set()
    for group in re.findall(r"\[([0-9,\s–-]+)\]", body):
        for x in group.split(","):
            x = x.strip().replace("-", "–")
            if not x:
                continue
            if "–" in x:
                start, end = map(int, x.split("–"))
                if start > end:
                    raise ValueError("Backwards citation range")
                names.update(range(start, end + 1))
            else:
                names.add(int(x))
    return names


def verify():
    manuscript = P / "manuscript.md"
    word = P / "LATEST_MANUSCRIPT_FOR_AVA.docx"
    cfd = P / "CFD_HANDOFF_CURRENT.zip"
    text = manuscript.read_text(encoding="utf-8")
    body, bibliography = text.split("## References\n")
    refs = [int(n) for n in re.findall(r"(?m)^\[(\d+)\]", bibliography)]
    if refs != list(range(1, 30)) or references(body) != set(refs):
        raise AssertionError("Not all real cited publication identities and in-text numbers match")
    source_audit = js(AUDIT / "FOCUSED_BIBLIOGRAPHY_AUDIT.json")
    if (
        len(source_audit["records"]) != 29
        or source_audit["status"] != "ALL_REFERENCES_PRIMARY_METADATA_RECONCILED"
    ):
        raise ValueError("One or more primary publication identities not verified")
    # The source part was revised, while the 29 actual references have not
    # been.
    current_references = [
        entry.strip() for _, entry in re.findall(r"(?m)^\[(\d+)\]\s+(.+)$", bibliography)
    ]
    if current_references != [entry["citation"] for entry in source_audit["records"]]:
        raise ValueError("The 29 live primary-source-verified citation texts changed")
    if body.count("NeuralFoil") != 1 or "NeuralFoil [29]" not in body:
        raise ValueError("NeuralFoil should be cited as prior work, not an experimental topic")
    if "Post hoc benchmark against an established aerodynamic surrogate" in body:
        raise ValueError("Secondary comparator still dominates the study")
    if body.count("### 5.2.1 Geometry-distinct numerical generalisation challenge") != 1:
        raise ValueError("The source-distinct numerical sensitivity evidence is missing")
    if not 180 <= len(text.split("## Abstract\n", 1)[1].split("## Nomenclature")[0].split()) <= 250:
        raise ValueError("Abstract too long for common applied journals")
    if (
        len(text.split()) < 9000
        or text.count("![Figure ") != 7
        or text.count("![Supplementary Figure") != 1
    ):
        raise ValueError("Incomplete standalone numerical manuscript")
    doc = Document(word)
    if len(doc.tables) != 7 or len(doc.inline_shapes) != 8:
        raise ValueError("Word lost figures/tables")
    wsrc = js(AUDIT / "FOCUSED_WORD_SOURCE.json")
    if sha(word) != wsrc["docx_sha256"] or sha(manuscript) != wsrc["source_sha256"]:
        raise ValueError("Word/source changed since exact export")
    render = js(AUDIT / "FOCUSED_WORD_VISUAL_RECEIPT.json")["main"]
    if render["word_sha256"] != sha(word) or render["figure_caption_image_groups"] != 8:
        raise ValueError("This exact Word has no complete figure/caption QA render")
    for record in render["pages"]:
        if sha(ROOT / record["image"]) != record["sha256"]:
            raise ValueError("Rendered page changed")
    for record in render["contact_sheets"]:
        if sha(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("Contact sheet changed")
    for f in ("COAUTHOR_NOTES.md", "CFD_BRIEF_FOR_AVA.md", "README.md", "REVIEW_STATUS.md"):
        if not (P / f).is_file():
            raise FileNotFoundError(f)
    original = js(DATA / "DATA_MANIFEST.json")
    if (original["requested_unique_states"], original["observed_states"]) != (141928, 106931):
        raise ValueError("Full exact-CST numerical data changed")
    sealed = js(DATA / "confirmation/CONFIRMATION_RESULTS.json")
    if (sealed["requested_states"], sealed["observed_states"]) != (22600, 17968):
        raise ValueError("Original sealed prospective confirmation changed")
    robust = js(DATA / "optimisation_production/PRODUCTION_SUMMARY.json")
    nominal = js(DATA / "nominal_control/NOMINAL_CONTROL_SUMMARY.json")
    selected = js(DATA / "selected_study/SELECTED_XFOIL_SUMMARY.json")
    if (
        not robust["both_numerical_checks_met"]
        or len(robust["per_seed"]) != 10
        or len(nominal["per_seed"]) != 10
    ):
        raise ValueError("Full robust vs matched nominal study incomplete")
    if (
        selected["cases_requested"],
        selected["usable_cases"],
        selected["cases_failed_or_not_admitted"],
    ) != (819, 290, 529):
        raise ValueError("Strict selected XFOIL failures were not retained")
    frozen = DATA / "FROZEN_MODEL_MANIFEST.json"
    f = js(frozen)
    verify_frozen_artifacts(frozen, {key: Path(path) for key, path in f["artifact_paths"].items()})
    new = js(AUDIT / "EXTERNAL_NACA_FAMILY/INDEPENDENT_RECEIPT_AND_TRUST_AUDIT.json")
    if (
        new["requested_geometries"],
        new["solver_observed_states"],
        new["solver_missing_states"],
    ) != (40, 4331, 189):
        raise ValueError("Separate source-distinct numerical challenge not verified")
    prior = js(OLD / "NEW_MANUSCRIPT_NUMERIC_CLAIM_AUDIT.json")
    if prior["checks"] != 46:
        raise ValueError("Original 46 numerical source audit receipts missing")
    for item in prior["claims"]:
        if sha(ROOT / item["source_path"]) != item["source_sha256"]:
            raise ValueError("Frozen numerical source was altered: " + item["claim"])
    with zipfile.ZipFile(cfd) as z:
        if z.testzip() is not None:
            raise ValueError("Current finalist CFD zip damaged")
        meta = json.loads(z.read("CURRENT_CASE_MANIFEST.json"))
        names = set(z.namelist())
        if (
            len(meta["files"]) != 9
            or not meta["historical_14October_CFD_not_applicable_to_this_finalist_study"]
        ):
            raise ValueError("CFD handoff still uses historical candidate set")
        if not all(
            "inputs/nominal/" + v + ".dat" in names
            for v in ("reference", "risk_knee", "nominal_control")
        ):
            raise ValueError("Actual three nominal CFD finalists missing")
        reference = DATA / "selected_study/FROZEN_SELECTED_DESIGNS.json"
        if meta["original_frozen_selected_designs_sha256"] != sha(reference):
            raise ValueError("CFD package does not match selected study")
        for name, record in meta["files"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != record["sha256"]:
                raise ValueError("CFD geometry was modified: " + name)
    diagnostic = AUDIT / "TECHNICAL_BENCHMARK_ONLY/SUPPLEMENTARY_NUMERICAL_COMPARISON.md"
    if not diagnostic.is_file() or (P / "SUPPLEMENTARY_NUMERICAL_COMPARISON.md").exists():
        raise ValueError("Separate comparative diagnostic is missing or mislabelled as core study")
    return {
        "schema": "final-primary-focus-numerical-research-and-CFD-readiness-v1",
        "status": "COAUTHOR_WORKING_DRAFT_ACCEPTED_INDEPENDENT_CFD_PENDING",
        "journal_submission_ready": False,
        "independent_CFD_performed": False,
        "all_29_publication_records_checked": True,
        "source_claims_independently_reconciled": 46,
        "frozen_five_member_model_verified": True,
        "ten_robust_and_ten_nominal_completed": True,
        "selected_strict_xfoil_unusable": 529,
        "new_geometry_distinct_states_admitted": 4331,
        "original_confirmed_states_admitted": 17968,
        "manuscript_words": len(text.split()),
        "abstract_words": len(
            text.split("## Abstract\n", 1)[1].split("## Nomenclature")[0].split()
        ),
        "main_figures": len(doc.inline_shapes),
        "main_tables": len(doc.tables),
        "browser_render_pages": render["page_count"],
        "native_Microsoft_Word_pagination_verified": False,
        "current_three_finalist_CFD_handoff_valid": True,
        "neuralfoil_context": "Cited accurately as prior research; separate adverse posthoc diagnostic retained in archived engineering evidence, not main hypothesis",
        "source_paths_sha256": {
            x.relative_to(ROOT).as_posix(): sha(x)
            for x in [
                manuscript,
                word,
                cfd,
                AUDIT / "FOCUSED_BIBLIOGRAPHY_AUDIT.json",
                AUDIT / "FOCUSED_WORD_VISUAL_RECEIPT.json",
            ]
        },
        "next_required_work": "Ava independent CFD, physical-design conclusions, author declarations and native Word journal formatting",
    }


def main():
    result = verify()
    if DEST.exists():
        if js(DEST) != result:
            raise ValueError("Final review receipt diverges")
    else:
        DEST.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "source_paths_sha256"}, indent=2))


if __name__ == "__main__":
    main()
