"""Bibliography audit against public Crossref metadata, without inventing citations.

Each entry is verified or flagged for manual review. The script never edits
the paper or upgrades an uncertain fuzzy match into a confirmed source.
"""

from __future__ import annotations

import difflib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper/manuscript.md"
OUT = ROOT / "reports/robust_v2/submission_review_20261014/REFERENCES"
OUT.mkdir(parents=True, exist_ok=True)
refs = []
for line in PAPER.read_text(encoding="utf-8").splitlines():
    m = re.match(r"^\[(\d+)\]\s+(.*)$", line)
    if m:
        refs.append((int(m.group(1)), m.group(2)))
audit = []
for number, reference in refs:
    title = reference
    split = re.search(r"\(\d{4}\)\.\s*", reference)
    if split:
        title = reference[split.end() :].split(". ", 1)[0].strip()
    exact_doi = re.search(r"https?://doi\.org/([^\s]+)", reference)
    raw = {"number": number, "original_reference": reference, "query_title": title}
    if reference.startswith("Airfoil Tools") or number == 11 or number in (25, 26):
        raw.update(
            {
                "status": "publisher_or_web_catalog_check_required",
                "reason": "Website or book, not necessarily Crossref work",
            }
        )
        audit.append(raw)
        continue
    if number in (17, 18):
        raw.update(
            {
                "status": "known_arxiv_identifier_requires_primary_arxiv_check",
                "reason": "DOI metadata not needed for arXiv identifier",
            }
        )
        audit.append(raw)
        continue
    try:
        if exact_doi:
            url = "https://api.crossref.org/works/" + urllib.parse.quote(
                exact_doi.group(1), safe="/"
            )
        else:
            url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
                {
                    "query.bibliographic": reference,
                    "rows": 3,
                    "select": "DOI,title,author,published,container-title",
                }
            )
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "AerofoilResearchBibliographyAudit/1.0 (academic verification)"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.loads(response.read())
        candidates = [data["message"]] if exact_doi else data["message"]["items"]
        scores = []
        for entry in candidates:
            title_cross = entry.get("title", [""])[0]
            # Avoid false positives from abbreviated titles.
            score = difflib.SequenceMatcher(None, title.lower(), title_cross.lower()).ratio()
            scores.append((score, entry))
        score, best = max(scores, key=lambda x: x[0])
        raw.update(
            {
                "status": "exact_title_candidate"
                if score >= 0.85
                else "manual_bibliography_review",
                "title_similarity": round(score, 4),
                "crossref_title": best.get("title", [""])[0],
                "doi": best.get("DOI"),
                "publisher_link": "https://doi.org/" + str(best.get("DOI")),
                "container_title": best.get("container-title", [""])[0]
                if best.get("container-title")
                else None,
            }
        )
    except Exception as e:
        raw.update({"status": "lookup_unavailable", "failure": str(e)[:120]})
    audit.append(raw)
    time.sleep(0.22)
report = {
    "schema": "bibliography-crossref-audit-v1",
    "caution": "Crossref search similarity is not author/pagination verification; manual checks required for flagged references.",
    "count": len(audit),
    "items": audit,
}
(OUT / "BIBLIOGRAPHY_AUDIT.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print("Bibliography:", len(audit))
for i in audit:
    print(f"[{i['number']:02d}] {i['status']} {i.get('title_similarity', '')} {i.get('doi', '')}")
