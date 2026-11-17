"""Read-only current Robust V2 author/CFD handoff integrity gate.

Historical October audit receipts remain immutable. This checks the live paper
against its original frozen numerical lineage, not the older Word export hashes.
Passing does not establish independent CFD or journal acceptance.
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
PAPER = ROOT / "paper"
LINEAGE = ROOT / "results/robust_v2/lineages/robust-v2-controlled-20261020"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def citation_numbers(body: str) -> set[int]:
    found = set()
    for group in re.findall(r"\[([0-9,\s–-]+)\]", body):
        for term in group.split(","):
            term = term.strip().replace("-", "–")
            if "–" in term:
                start, end = map(int, term.split("–"))
                if start > end:
                    raise AssertionError("Backwards citation range")
                found.update(range(start, end + 1))
            elif term:
                found.add(int(term))
    return found


def verify() -> dict:
    source = (PAPER / "manuscript.md").read_text(encoding="utf-8")
    body, refs_text = source.split("## References\n", 1)
    refs = [int(n) for n in re.findall(r"(?m)^\[(\d+)\]", refs_text)]
    assert refs == list(range(1, 31)) and citation_numbers(body) == set(refs)
    assert "NeuralFoil" in body and "Ju and Zhang [30]" in body
    assert "33 service incidences" in source and "pointwise" in source
    assert "CFD has not yet been performed" in source

    doc_path = PAPER / "LATEST_MANUSCRIPT_FOR_AVA.docx"
    doc = Document(doc_path)
    assert len(doc.tables) == 8 and len(doc.inline_shapes) == 8
    assert source.splitlines()[0][2:] in "\n".join(p.text for p in doc.paragraphs)
    with zipfile.ZipFile(doc_path) as z:
        assert z.testzip() is None
        assert z.read("word/document.xml").count(b"<m:oMath>") >= 40

    corpus = read_json(LINEAGE / "DATA_MANIFEST.json")
    assert (corpus["requested_unique_states"], corpus["observed_states"]) == (141928, 106931)
    robust = read_json(LINEAGE / "optimisation_production/PRODUCTION_SUMMARY.json")
    nominal = read_json(LINEAGE / "nominal_control/NOMINAL_CONTROL_SUMMARY.json")
    assert robust["both_numerical_checks_met"]
    assert len(robust["per_seed"]) == len(nominal["per_seed"]) == 10
    selected = LINEAGE / "selected_study"
    xfoil = read_json(selected / "SELECTED_XFOIL_SUMMARY.json")
    assert (
        xfoil["cases_requested"],
        xfoil["usable_cases"],
        xfoil["cases_failed_or_not_admitted"],
    ) == (819, 290, 529)
    fresh = read_json(selected / "FINAL_2048_SCENARIO_EVALUATION.json")
    rows = [r for r in fresh["rows"]
            if r["design"] in ("reference", "risk_knee", "nominal_control")]
    assert fresh["scenario_count"] == 2048 and len(rows) == 12
    for row in rows:
        assert format(row["mean_penalised_weighted_cd"], ".6f") in source
        assert format(row["empirical_cvar95_penalised_weighted_cd"], ".6f") in source
    freeze = LINEAGE / "FROZEN_MODEL_MANIFEST.json"
    model = read_json(freeze)
    verify_frozen_artifacts(
        freeze, {key: Path(value) for key, value in model["artifact_paths"].items()}
    )

    package = PAPER / "CFD_HANDOFF_CURRENT.zip"
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None and len(z.namelist()) == 14
        assert z.read("READ_ME_FIRST.md") == (PAPER / "CFD_BRIEF_FOR_AVA.md").read_bytes()
        manifest = json.loads(z.read("CURRENT_CASE_MANIFEST.json"))
        assert len(manifest["files"]) == 9
        for name, item in manifest["files"].items():
            digest = sha(z.read(name))
            assert digest == item["sha256"], name
            assert digest == sha((ROOT / item["source"]).read_bytes()), name
        assert sha(z.read("FROZEN_SELECTED_DESIGNS.json")) == manifest[
            "original_frozen_selected_designs_sha256"
        ]
        assert sha(z.read("XFOIL_CASE_SELECTION_PROTOCOL.json")) == manifest[
            "original_xfoil_case_selection_protocol_sha256"
        ]

    return {
        "status": "NUMERICAL_EVIDENCE_AND_CURRENT_CFD_HANDOFF_VERIFIED",
        "independent_cfd_performed": False,
        "journal_submission_ready": False,
        "references": len(refs),
        "word_tables": len(doc.tables),
        "word_figures": len(doc.inline_shapes),
        "frozen_robust_seeds": len(robust["per_seed"]),
        "frozen_nominal_seeds": len(nominal["per_seed"]),
        "strict_xfoil_usable": 290,
        "strict_xfoil_unusable": 529,
        "fresh_sensitivity_rows": len(rows),
        "handoff_geometry_files": len(manifest["files"]),
        "docx_sha256": sha(doc_path.read_bytes()),
        "cfd_package_sha256": sha(package.read_bytes()),
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
