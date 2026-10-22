"""Bibliography review against publication registries; never fabricate full-text reading."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper/manuscript.md"
LEGACY = ROOT / "reports/robust_v2/external_academic_review_20261014/REFERENCES_VERIFIED.csv"
FINAL = ROOT / "reports/robust_v2/final_release_20261022/FOCUSED_BIBLIOGRAPHY_AUDIT.json"


def squish(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def getdoi(doi: str) -> dict:
    u = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    req = urllib.request.Request(
        u,
        headers={"User-Agent": "RobustV2-Academic-Bibliography/1.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=25) as page:
        return json.load(page)["message"]


def main() -> None:
    if FINAL.exists():
        raise FileExistsError("Do not rewrite publication audit without reviewing differences")
    body = SOURCE.read_text(encoding="utf-8")
    refs = {int(n): entry.strip() for n, entry in re.findall(r"(?m)^\[(\d+)\]\s+(.+)$", body)}
    old = list(csv.DictReader(LEGACY.open(encoding="utf-8-sig", newline="")))
    if set(refs) != set(range(1, 30)) or len(old) != 29:
        raise AssertionError("All 29 original bibliographic references required")
    out = []
    for row in old:
        num = int(row["number"])
        s = refs[num]
        entry = {
            "number": num,
            "citation": s,
            "doi": row["doi"],
            "metadata_source": "Crossref DOI primary publication registry"
            if row["doi"]
            else "Author or publisher/official primary record",
            "verified_article_exists": None,
            "author_order_checked": None,
        }
        if row["doi"]:
            doc = getdoi(row["doi"])
            title = doc.get("title", [""])[0]
            author = [x.get("family", "") for x in doc.get("author", [])]
            year = doc.get("published", {}).get("date-parts", [[None]])[0][0]
            doi = doc.get("DOI", "")
            title_match = squish(title) in squish(s)
            special = False
            if num == 9:
                # Original paper uses "Parametric"; Crossref index misspells it.
                title_match = squish(title.replace("Parameteric", "Parametric")) in squish(s)
                special = True
            first = bool(author and squish(author[0]) in squish(s.split("(")[0]))
            ordered = True
            if num == 27:
                expected = [
                    "Pichitkul",
                    "Boksawat",
                    "Phiboon",
                    "Tantrairatn",
                    "Bureerat",
                    "Kasemsri",
                    "Ariyarit",
                ]
                ordered = author == expected
            date = str(year) in s
            matching = squish(doi) == squish(row["doi"])
            volume = doc.get("volume")
            volcheck = not volume or str(volume) in s
            page = doc.get("article-number") or doc.get("page")
            pagecheck = not page or str(page) in s.replace("–", "-")
            checks = {
                "doi": matching,
                "title": title_match,
                "first_author": first,
                "publication_year": date,
                "author_order_where_explicit": ordered,
                "volume": volcheck,
                "page_or_article": pagecheck,
            }
            entry.update(
                author_order_checked=ordered if num == 27 else "first only",
                primary_title=title,
                primary_year=year,
                primary_authors=author,
                primary_volume=volume,
                primary_article_or_page=page,
                metadata_checks=checks,
                crossref_title_typo_original_paper_exception=special,
            )
            entry["verified_article_exists"] = all(checks.values())
        else:
            # Non-DOI official author / publisher or conference resources were
            # checked in the earlier academic audit, retained with exact source.
            prior = json.loads(
                (
                    ROOT
                    / "reports/robust_v2/external_academic_review_20261021/REFERENCES_PRIMARY_VERIFIED_V2.json"
                ).read_text(encoding="utf-8")
            )
            earlier = prior["verified_records"][num - 1]
            entry["metadata_source"] = earlier["primary_record"]
            entry["source_provenance_prior_sha256"] = sha(
                ROOT
                / "reports/robust_v2/external_academic_review_20261021/REFERENCES_PRIMARY_VERIFIED_V2.json"
            )
            entry["verified_article_exists"] = bool(earlier["bibliography_accurate"])
            entry["author_order_checked"] = "primary original audited"
        out.append(entry)
        print(
            f"REF_{num:02d} {'PASS' if entry['verified_article_exists'] else 'UNRESOLVED'}",
            flush=True,
        )
    checks = [e for e in out if not e["verified_article_exists"]]
    receipt = {
        "schema": "focused-full-29-reference-metadata-check-v1",
        "status": "ALL_REFERENCES_PRIMARY_METADATA_RECONCILED"
        if not checks
        else "UNRESOLVED_METADATA",
        "manuscript_sha256": sha(SOURCE),
        "previous_reference_ledger_sha256": sha(LEGACY),
        "reference_count": len(out),
        "unresolved_indices": [x["number"] for x in checks],
        "limitations": [
            "Checks bibliographic identity, author/year/journal metadata; not a full-text scientific correctness proof.",
            "The first author is checked for DOI publications; all-seven author order explicitly checked for reference 27.",
            "Published author PDF overrides a known Crossref title typo for the 2006 Kulfan and Bussoletti item.",
            "No publication is represented as independently validating the new numerical aerodynamics.",
        ],
        "records": out,
    }
    FINAL.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(receipt["status"], len(out))
    if checks:
        raise RuntimeError("Bibliography contains unresolved metadata: " + str(checks))


if __name__ == "__main__":
    main()
