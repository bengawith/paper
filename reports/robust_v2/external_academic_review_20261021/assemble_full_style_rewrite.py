"""Assemble full user-voice rewrite with separately crosschecked references."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
baseline = HERE / "PRE_REWRITE_CONTROLLED_MANUSCRIPT.md"
current = ROOT / "paper/manuscript.md"
if current.read_bytes() != baseline.read_bytes():
    raise ValueError("Existing source changed since backup")
refs = baseline.read_text(encoding="utf-8").split("\n## References\n")
if len(refs) != 2:
    raise ValueError("No exact reference boundary")
text = "\n## References\n" + refs[1]
text = text.replace(
    "Airfoil Tools (n.d.). Airfoil Tools. Historical source of archived XFOIL polars; current site access was not verified during this review. http://airfoiltools.com/",
    "Airfoil Tools (n.d.). Airfoil Tools: numerical aerofoil polar database. Historical source of archived polars. Accessed 21 October 2026. https://www.airfoiltools.com/",
)
text = text.replace(
    "Anderson, J. D. (2017). Fundamentals of Aerodynamics. 6th edn. McGraw-Hill Education.",
    "Anderson, J. D. (2017). Fundamentals of Aerodynamics, 6th edn. McGraw-Hill Education. https://books.google.com/books/about/Fundamentals_of_Aerodynamics.html?id=_7VLjwEACAAJ",
)
text = text.replace(
    "https://store.doverpublications.com/products/9780486605869",
    "https://books.google.com/books/about/Theory_of_Wing_Sections.html?id=GvpTAAAAMAAJ",
)
full = (
    (HERE / "MANUSCRIPT_STYLE_REWRITE_DRAFT.md").read_text(encoding="utf-8").rstrip()
    + text.rstrip()
    + "\n"
)
if [int(v) for v in re.findall(r"(?m)^\[(\d+)\] ", full)] != list(range(1, 30)):
    raise ValueError("Incorrect references")
if full.count("![Figure ") != 7 or full.count("![Supplementary Figure") != 1:
    raise ValueError("Figure mismatch")
if len(re.findall(r"(?m)^\|---", full)) != 7:
    raise ValueError("Expected nomenclature plus six results tables")
if len(full.split()) < 7500:
    raise ValueError("Not a full manuscript rewrite")
for value in ["106,931", "141,928", "17,968", "22,600", "819", "290", "529", "371", "0.821"]:
    if value not in full:
        raise ValueError("Missing verified result " + value)
current.write_text(full, encoding="utf-8")
print("COMPLETE_REWRITE", len(full.split()), "words, 29 references, 7 tables, 8 images")
