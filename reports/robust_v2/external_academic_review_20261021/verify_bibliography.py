"""Independently reconcile all published DOI citation fields with Crossref."""

import csv
import datetime
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
prior = ROOT / "reports/robust_v2/external_academic_review_20261014/REFERENCES_VERIFIED.csv"
output = Path(__file__).with_name("BIBLIOGRAPHY_LIVE_AUDIT.json")
if output.exists():
    raise FileExistsError(output)


def simple(t):
    return re.sub("[^a-z0-9]", "", t.casefold())


results = []
for row in csv.DictReader(prior.open(encoding="utf-8-sig", newline="")):
    number = int(row["number"])
    result = {"number": number, "citation": row["citation"], "doi": row["doi"]}
    try:
        if row["doi"]:
            url = "https://api.crossref.org/works/" + urllib.parse.quote(row["doi"], safe="")
            request = urllib.request.Request(
                url, headers={"User-Agent": "RobustV2ResearchAudit/1.0"}
            )
            with urllib.request.urlopen(request, timeout=22) as f:
                raw = f.read()
            work = json.loads(raw)["message"]
            year = work.get("published", {}).get("date-parts", [[None]])[0][0]
            title = work.get("title", [""])[0]
            authors = [a.get("family", "") for a in work.get("author", [])]
            tests = {
                "title": simple(title) in simple(row["citation"]),
                "first_author": bool(authors and simple(authors[0]) in simple(row["citation"])),
                "year": str(year) in row["citation"],
                "doi": simple(work["DOI"]) == simple(row["doi"]),
            }
            result.update(
                status="PRIMARY_METADATA_MATCH" if all(tests.values()) else "DISCREPANCY",
                tests=tests,
                primary={
                    "title": title,
                    "year": year,
                    "authors": authors,
                    "journal": work.get("container-title"),
                    "volume": work.get("volume"),
                    "issue": work.get("issue"),
                    "pages": work.get("page"),
                    "article_number": work.get("article-number"),
                    "doi": work["DOI"],
                },
                crossref_response_sha256=hashlib.sha256(raw).hexdigest(),
            )
        else:
            result.update(
                status="OFFICIAL_IDENTITY_NEEDS_SOURCE_CHECK", primary_url=row["primary_url"]
            )
    except Exception as exc:
        result.update(status="RETRIEVAL_FAILURE", error=repr(exc))
    results.append(result)
    print(number, result["status"], flush=True)
summary = {
    "utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "references": results,
    "all_doi_identity_match": all(
        x["status"] == "PRIMARY_METADATA_MATCH" for x in results if x["doi"]
    ),
    "unverified": [
        {"number": x["number"], "status": x["status"]}
        for x in results
        if x["status"] != "PRIMARY_METADATA_MATCH"
    ],
}
output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("BIBLIOGRAPHY_RECONCILIATION", len(results), summary["all_doi_identity_match"], flush=True)
