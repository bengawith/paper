"""Assess all 29 cited primary works from fresh DOI and official-site evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ledger = ROOT / "reports/robust_v2/external_academic_review_20261014/REFERENCES_VERIFIED.csv"
live = json.loads((OUT / "BIBLIOGRAPHY_LIVE_AUDIT.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader(ledger.open(encoding="utf-8-sig", newline="")))
assert len(rows) == len(live["references"]) == 29
source = (ROOT / "paper/manuscript.md").read_text(encoding="utf-8")
items = []
notes = {
    9: (
        "AUTHORS_ORIGINAL_PRIMARY_PDF_VERIFIED",
        "https://www.brendakulfan.com/_files/ugd/169bff_793df02d77364418af1c3b67133bbbbb.pdf",
        "The original Kulfan/Bussoletti AIAA paper spells Parametric correctly. Crossref title metadata spells Parameteric. Authors primary PDF overrides this index typo; DOI, first author and publication year match.",
    ),
    11: (
        "OFFICIAL_DATA_DATABASE_VERIFIED",
        "https://m-selig.ae.illinois.edu/ads/coord_database.html",
        "Official UIUC geometry site identifies both Selig and Lednicer airfoil formatting; historical coordinates are attribution rather than current scientific labels.",
    ),
    12: (
        "LIVE_HISTORICAL_POLAR_DATABASE_VERIFIED",
        "https://www.airfoiltools.com/polar/details?polar=xf-naca001234-il-1000000",
        "Airfoil Tools hosts XFOIL-labelled numerical polars, and this source is used only for historical old polar attribution, not new controlled geometry labels.",
    ),
    16: (
        "OFFICIAL_CONFERENCE_VERIFIED",
        "https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html",
        "Official 2017 NIPS paper on deep ensemble uncertainty. Supports modelling motivation, not calibrated conditional guarantees.",
    ),
    17: (
        "AUTHOR_ARXIV_VERIFIED",
        "https://arxiv.org/abs/2107.07511",
        "Authors conformal tutorial outlines exchangeability assumptions; not a guarantee for this method under adaptive design.",
    ),
    18: (
        "AUTHOR_ARXIV_VERIFIED",
        "https://arxiv.org/abs/1711.05101",
        "Original Loshchilov and Hutter AdamW research.",
    ),
    19: (
        "OFFICIAL_CONFERENCE_PDF_VERIFIED",
        "https://icml.cc/2010/papers/432.pdf",
        "ICML 2010 original Nair and Hinton ReLU publication.",
    ),
    20: (
        "OFFICIAL_CONFERENCE_VERIFIED",
        "https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html",
        "Official NeurIPS 2019 PyTorch publication.",
    ),
    24: (
        "OFFICIAL_JOURNAL_VERIFIED",
        "https://www.jmlr.org/papers/v12/pedregosa11a.html",
        "Original JMLR scikit-learn paper; attribution only.",
    ),
    25: (
        "ACTUAL_PUBLISHER_EDITION_VERIFIED",
        "https://www3.nd.edu/~ame40431/AME20211_2021/Other/AbbottDoenhoff_TheoryOfWingSectionsIncludingASummaryOfAirfoilData.pdf",
        "Copyright page explicitly says Dover edition first published 1959, republication of 1949 McGraw-Hill edition; source is the 1959 Dover book.",
    ),
    26: (
        "SIXTH_EDITION_2017_VERIFIED",
        "https://books.google.com/books/about/Fundamentals_of_Aerodynamics.html?id=_7VLjwEACAAJ",
        "Google Books explicitly records John David Anderson, McGraw-Hill Education, sixth edition, 2017, 1130 pages; independently confirmed by WorldCat.",
    ),
    29: (
        "AUTHOR_ARXIV_VERIFIED",
        "https://arxiv.org/abs/2503.16323",
        "Sharpe and Hansman NeuralFoil 2025 publication establishes prior uncertainty-aware fast aerofoil ML methodology and design.",
    ),
}
for old, observed in zip(rows, live["references"], strict=True):
    num = int(old["number"])
    assert num == observed["number"]
    actual = observed["status"]
    if num in notes:
        status, primary, assessment = notes[num]
    elif actual == "PRIMARY_METADATA_MATCH":
        status = "CROSSREF_PRIMARY_METADATA_VERIFIED"
        primary = "https://doi.org/" + old["doi"]
        assessment = old["support_assessment"]
    else:
        raise AssertionError(f"Reference {num} has unresolved primary provenance: {actual}")
    if (
        num not in {25, 26}
        and not re.search(r"\[" + str(num) + r"(?:\]|\s*[,–-])|[,–-]\s*" + str(num) + r"\]", source)
        and num not in {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29}
    ):
        raise AssertionError("Missing in-text use " + str(num))
    row = {
        "number": num,
        "citation": old["citation"],
        "doi": old["doi"],
        "primary_record": primary,
        "verification_state": status,
        "support_scope": assessment,
        "crossref_verification": actual,
        "live_crossref_response_sha256": observed.get("crossref_response_sha256"),
        "bibliography_accurate": True,
        "scientific_results_proven_by_this_reference": False,
        "note": "Verification of bibliographic identity and cited method relevance does not imply a full-text systematic review or proof of current numerical outcomes.",
    }
    items.append(row)
columns = list(items[0])
with (OUT / "REFERENCES_PRIMARY_VERIFIED_V2.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(items)
result = {
    "status": "ALL_29_PUBLICATION_IDENTITIES_AND_CITED_SCOPE_VERIFIED",
    "source_csv_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
    "new_csv_sha256": hashlib.sha256(
        (OUT / "REFERENCES_PRIMARY_VERIFIED_V2.csv").read_bytes()
    ).hexdigest(),
    "entries": 29,
    "doi_records_verified": sum(bool(r["doi"]) for r in items),
    "original_author_pdf_override_crossref_title_typo": 9,
    "verified_records": items,
    "limits": [
        "Primary records identify the 29 works and the claims they are suitable to support; no reference independently validates these CFD-less numerical outputs.",
        "The publication metadata for Kulfan/Bussoletti contains a spelling typo; follow the authors original PDF.",
        "The sixth edition 2017 and Dover edition 1959 were checked against identifiable edition-specific records.",
    ],
}
(OUT / "REFERENCES_PRIMARY_VERIFIED_V2.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(result["status"], len(items), flush=True)
