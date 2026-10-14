"""Attach primary publisher/proceedings links to actually identified references.

No generated reference is created; original bibliography numbering and entries
are retained. Books and the AirfoilTools historical site are explicitly flagged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper/manuscript.md"
OUT = ROOT / "reports/robust_v2/submission_review_20261014/REFERENCES"
OUT.mkdir(parents=True, exist_ok=True)
PRIMARY = {
    1: "https://doi.org/10.2514/3.9789",
    2: "https://doi.org/10.1038/s43588-022-00264-7",
    3: "https://doi.org/10.1146/annurev-fluid-010719-060214",
    4: "https://doi.org/10.2514/1.J057129",
    5: "https://doi.org/10.1016/j.paerosci.2022.100849",
    6: "https://doi.org/10.1080/03052150903305468",
    7: "https://doi.org/10.1016/j.compfluid.2011.02.010",
    8: "https://doi.org/10.2514/1.29958",
    9: "https://doi.org/10.2514/6.2006-6948",
    10: "https://doi.org/10.1007/978-3-642-84010-4_1",
    11: "https://m-selig.ae.illinois.edu/ads/coord_database.html",
    12: "http://airfoiltools.com/",
    13: "https://doi.org/10.1109/4235.996017",
    14: "https://doi.org/10.1109/ACCESS.2020.2990567",
    15: "https://doi.org/10.21314/JOR.2000.038",
    16: "https://papers.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles",
    17: "https://arxiv.org/abs/2107.07511",
    18: "https://arxiv.org/abs/1711.05101",
    19: "https://icml.cc/2010/papers/432.pdf",
    20: "https://papers.neurips.cc/paper/9015-pytorch-an-imperative-stylehigh-performance-deep-learning-library.pdf",
    21: "https://doi.org/10.1145/3292500.3330701",
    22: "https://doi.org/10.1038/s41586-020-2649-2",
    23: "https://doi.org/10.1038/s41592-019-0686-2",
    24: "https://www.jmlr.org/papers/v12/pedregosa11a.html",
    27: "https://doi.org/10.1016/j.apples.2026.100310",
    28: "https://doi.org/10.1073/pnas.2204569119",
}
text = PAPER.read_text(encoding="utf-8")
text = text.replace(
    "Neural Information Processing Systems 32 (NeurIPS 2019), 8024–8035.",
    "Neural Information Processing Systems 32 (NeurIPS 2019), 8026–8037.",
)
lines = text.splitlines()
ledger = []
for index, line in enumerate(lines):
    match = re.match(r"^\[(\d+)\]\s+", line)
    if not match:
        continue
    number = int(match.group(1))
    url = PRIMARY.get(number)
    status = "identified_publisher_or_proceedings" if url else "book_manual_catalog_check"
    if number == 12:
        status = "historical_web_source_current_access_not_guaranteed"
    elif number == 11:
        status = "official_university_database"
    elif number in (17, 18):
        status = "primary_arxiv"
    if url and url.lower() not in line.lower():
        line += " " + url
        lines[index] = line
    ledger.append(
        {"number": number, "source_url": url, "status": status, "reference": lines[index]}
    )
assert len(ledger) == 28
text = "\n".join(lines) + "\n"
PAPER.write_text(text, encoding="utf-8", newline="\n")
(OUT / "PRIMARY_SOURCE_REFERENCE_LEDGER.json").write_text(
    json.dumps(
        {
            "schema": "primary-reference-ledger-v1",
            "basis": "Crossref metadata and verified publisher/proceedings pages; books/web resources not claimed verified via Crossref",
            "entries": ledger,
        },
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)
print("Reference entries preserved:", len(ledger), "linked:", len(PRIMARY))
