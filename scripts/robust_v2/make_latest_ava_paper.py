"""Build the current coauthor Word manuscript from its single Markdown source.

The document is regenerated, never hand-patched; frozen study data and figures
are inputs. Word equations are editable Office Math objects (OMML).
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper" / "manuscript.md"
OUTPUT = ROOT / "paper" / "LATEST_MANUSCRIPT_FOR_AVA.docx"

MATH_SYMBOLS = re.compile(
    r"(?<![\w])(?:C̄_M|C_[LDM]|A_[uUlL](?:,[a-z])?|A_LE|t_TE|"
    r"y_[ul]|B_i|d_s|J_s|L_s|e_k|y_k|ŷ_k)(?![\w])"
)
DISPLAY = re.compile(r"^(?:C\(x\) =|y_[ul]\(x\) =|C̄_M\(p\) ≥|J =|P\(p,S\) =|L_s\(p\) =)")
MARKUP = re.compile(r"(\*\*[^*]+\*\*|(?<!\*)\*[^*]+\*(?!\*))")
MATH_TOKEN = re.compile(r"(C̄|∑|[A-Za-z][A-Za-z0-9]*)(?:_([A-Za-z0-9,]+|\([^)]+\)))?(\^\([^)]+\)|\^[A-Za-z0-9]+)?")


def math_run(value: str):
    run = OxmlElement("m:r")
    text = OxmlElement("m:t")
    text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text.text = value
    run.append(text)
    return run


def math_expression(value: str):
    """Turn explicit subscripts/superscripts into native Word Math elements."""
    if "(1 − x)^(4−i)" in value:
        left, right = value.split("(1 − x)^(4−i)", 1)
        parent = OxmlElement("m:oMath")
        for element in math_expression(left):
            parent.append(element)
        power = OxmlElement("m:sSup")
        base = OxmlElement("m:e")
        base.append(math_run("(1 − x)"))
        power.append(base)
        exponent = OxmlElement("m:sup")
        exponent.append(math_run("4−i"))
        power.append(exponent)
        parent.append(power)
        for element in math_expression(right):
            parent.append(element)
        return parent
    parent = OxmlElement("m:oMath")
    cursor = 0
    for found in MATH_TOKEN.finditer(value):
        if found.start() > cursor:
            parent.append(math_run(value[cursor:found.start()]))
        name, sub, sup = found.groups()
        if sub or sup:
            structure = OxmlElement("m:sSubSup" if sub and sup else "m:sSub" if sub else "m:sSup")
            el = OxmlElement("m:e")
            el.append(math_run(name))
            structure.append(el)
            if sub:
                el = OxmlElement("m:sub")
                el.append(math_run(sub[1:-1] if sub.startswith("(") else sub))
                structure.append(el)
            if sup:
                el = OxmlElement("m:sup")
                el.append(math_run(sup[2:-1] if sup.startswith("^(") else sup[1:]))
                structure.append(el)
            parent.append(structure)
        else:
            parent.append(math_run(name))
        cursor = found.end()
    if cursor < len(value):
        parent.append(math_run(value[cursor:]))
    return parent


def add_inline(paragraph, value: str, *, bold=False, italic=False):
    """Preserve simple Markdown emphasis and add native math for named symbols."""
    for part in MARKUP.split(value):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            add_inline(paragraph, part[2:-2], bold=True, italic=italic)
            continue
        if part.startswith("*") and part.endswith("*"):
            add_inline(paragraph, part[1:-1], bold=bold, italic=True)
            continue
        cursor = 0
        for found in MATH_SYMBOLS.finditer(part):
            if found.start() > cursor:
                run = paragraph.add_run(part[cursor:found.start()])
                run.bold = bold
                run.italic = italic
            paragraph._p.append(math_expression(found.group()))
            cursor = found.end()
        if cursor < len(part):
            run = paragraph.add_run(part[cursor:])
            run.bold = bold
            run.italic = italic


def add_table(document, lines: list[str]):
    rows = [[x.strip() for x in line.strip().strip("|").split("|")] for line in lines]
    rows = [row for row in rows if not all(re.fullmatch(r"[-:\s]*", cell) for cell in row)]
    ncol = max(len(row) for row in rows)
    table = document.add_table(rows=len(rows), cols=ncol)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = True
    for i, row in enumerate(rows):
        for j, content in enumerate(row):
            cell = table.cell(i, j)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(1.5)
            p.paragraph_format.space_before = Pt(1.5)
            add_inline(p, content)
            for run in p.runs:
                run.font.size = Pt(7.7 if ncol >= 5 else 8.3)
                if i == 0:
                    run.bold = True
            if i == 0:
                tcpr = cell._tc.get_or_add_tcPr()
                shade = OxmlElement("w:shd")
                shade.set(qn("w:fill"), "EBEFF2")
                tcpr.append(shade)
    header = table.rows[0]._tr.get_or_add_trPr()
    element = OxmlElement("w:tblHeader")
    element.set(qn("w:val"), "true")
    header.append(element)
    document.add_paragraph().paragraph_format.space_after = Pt(1)


def add_figure(document, path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Figure absent: {path}")
    with Image.open(path) as im:
        width, height = im.size
    w = min(16.7, 18.7 * width / height)
    h = w * height / width
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_together = True
    p.add_run().add_picture(str(path), width=Cm(w), height=Cm(h))


def add_equation(document, value: str):
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_together = True
    para = OxmlElement("m:oMathPara")
    para.append(math_expression(value))
    p._p.append(para)


def make():
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21)
    sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(1.9)
    sec.left_margin, sec.right_margin = Cm(2.15), Cm(2.15)
    sec.header_distance, sec.footer_distance = Cm(1.0), Cm(0.9)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor(24, 28, 33)
    normal.paragraph_format.line_spacing = 1.12
    normal.paragraph_format.space_after = Pt(5.5)
    for name, size, before, after in [
        ("Title", 16.5, 0, 10),
        ("Heading 1", 12.5, 15, 6),
        ("Heading 2", 11.3, 12, 5),
        ("Heading 3", 10.5, 9, 4),
    ]:
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        ppr = style._element.get_or_add_pPr()
        for border in ppr.findall(qn("w:pBdr")):
            ppr.remove(border)

    caption = doc.styles["Caption"]
    caption.font.name = "Times New Roman"
    caption.font.size = Pt(9.3)
    caption.font.color.rgb = RGBColor(22, 30, 39)
    caption.paragraph_format.space_before = Pt(8)
    caption.paragraph_format.space_after = Pt(4)
    caption.paragraph_format.keep_with_next = True
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footrun = footer.add_run("Coauthor manuscript  |  ")
    footrun.font.size = Pt(8)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)

    doc.core_properties.title = "Data-Driven Aerodynamic Prediction and Robust Aerofoil Selection"
    doc.core_properties.author = "Benjamin Ian George Gawith, Ava Shahrokhi, Keivan Navaie"

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    i = 0
    prior = ""
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "---":
            i += 1
            continue
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            add_table(doc, block)
            continue
        figure = re.fullmatch(r"!\[[^]]*]\(([^)]+)\)", line)
        if figure:
            add_figure(doc, (SOURCE.parent / figure.group(1)).resolve())
            i += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2)
            style = "Title" if level == 1 else "Heading 1" if level == 2 else "Heading 2"
            if level >= 3:
                style = "Heading 3"
            if level == 2 and title in ("Authors", "Affiliations", "Corresponding Author",
                                         "Key Words", "Highlights", "Abstract"):
                style = "Heading 3"
            p = doc.add_paragraph(style=style)
            if title == "Nomenclature":
                p.paragraph_format.page_break_before = True
            add_inline(p, title)
            prior = title
            i += 1
            continue
        if re.match(r"^\*\*(?:Table \d+\.|Figure \d+\.|Supplementary Figure S\d+\.)", line):
            add_inline(doc.add_paragraph(style="Caption"), line)
            i += 1
            continue
        if DISPLAY.match(line):
            add_equation(doc, line)
            i += 1
            continue
        numbered = re.match(r"^(\d+)\.\s+(.*)", line)
        if numbered:
            add_inline(doc.add_paragraph(style="List Number"), numbered.group(2))
            i += 1
            continue
        if line.startswith("- "):
            add_inline(doc.add_paragraph(style="List Bullet"), line[2:])
            i += 1
            continue
        if line.startswith("[") and re.match(r"^\[\d+]", line):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.65)
            p.paragraph_format.first_line_indent = Cm(-0.65)
            p.paragraph_format.space_after = Pt(3.5)
            add_inline(p, line)
            i += 1
            continue
        block = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
            r"^(?:#{1,6}\s|\||-\s|!\[|\*\*(?:Table|Figure|Supplementary Figure)|\d+\.\s)", lines[i]
        ):
            block.append(lines[i].strip())
            i += 1
        p = doc.add_paragraph()
        if prior == "Authors":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_inline(p, " ".join(block))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(f"{OUTPUT}: {len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables")
    from zipfile import ZipFile
    with ZipFile(OUTPUT) as archive:
        document_xml = archive.read("word/document.xml")
        print("Word math objects:", document_xml.count(b"<m:oMath>"),
              "pictures:", document_xml.count(b"<pic:pic>"))


if __name__ == "__main__":
    make()
