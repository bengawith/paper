"""Improve real DOCX pagination without modifying audited manuscript text."""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "paper/LATEST_MANUSCRIPT_FOR_AVA.docx"
ARCHIVE = HERE / "PRE_LAYOUT_AUTHOR_VOICE_WORD.docx"
if ARCHIVE.exists():
    raise FileExistsError("Existing Word layout backup")
ARCHIVE.write_bytes(SOURCE.read_bytes())
doc = Document(SOURCE)
count = 0
for paragraph in doc.paragraphs:
    if paragraph.text.strip() == "Figures":
        paragraph.paragraph_format.page_break_before = True
        paragraph.paragraph_format.keep_with_next = True
    if paragraph.text.startswith(("Figure ", "Supplementary Figure ")):
        paragraph.paragraph_format.keep_with_next = True
        paragraph.paragraph_format.keep_together = True
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        count += 1
    if paragraph._p.xpath(".//a:blip"):
        paragraph.paragraph_format.keep_together = True
        paragraph.paragraph_format.keep_with_next = False
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
for section in doc.sections:
    if section.header.paragraphs:
        header = section.header.paragraphs[0]
        header.text = "ROBUST V2 | AUTHOR REVIEW | 21 OCTOBER 2026"
        for run in header.runs:
            run.font.size = Pt(8)
doc.core_properties.title = "Data-Driven Aerodynamic Prediction and Robust Aerofoil Selection"
doc.core_properties.author = "Benjamin Ian George Gawith"
doc.core_properties.subject = (
    "Geometry-consistent CST surrogate study with source-bound XFOIL validation"
)
doc.save(SOURCE)
doc = Document(SOURCE)
if count != 8 or len(doc.tables) != 7 or len(doc.inline_shapes) != 8:
    raise AssertionError("Layout pass changed the complete paper")
print("AUTHOR_STYLE_WORD_LAYOUT_REVISED", count, "figure groups", len(doc.tables), "tables")
