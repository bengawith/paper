"""Render genuine Word contents for visual review, not native Word pagination.

Uses the actual DOCX paragraph/table/embedded-image bytes. Browser page QA
and native Microsoft Word pagination are explicitly distinguished.
"""

from __future__ import annotations

import hashlib
import json
import re
import runpy
import subprocess
import tempfile
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "reports/robust_v2/final_release_20261022"
CHROME = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
HTML_SOURCE = (
    ROOT
    / "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/render_author_docx_browser.py"
)
RECORD = EVIDENCE / "FOCUSED_WORD_VISUAL_RECEIPT.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(name: str, word: Path, group_figures: bool) -> dict:
    folder = EVIDENCE / "VISUAL_FOCUSED" / name
    if folder.exists():
        raise FileExistsError("Prior rendered pages require inspection: " + str(folder))
    folder.mkdir(parents=True)
    html_doc = runpy.run_path(str(HTML_SOURCE))["html_doc"]
    markup, figures, tables, paragraphs = html_doc(word)
    grouped = 0
    if group_figures:
        pattern = re.compile(
            r"(<p><strong>(?:Figure [1-7]\.|Supplementary Figure S1\.)</strong>[^<]*</p>)"
            r'\s*(<p><img class="figure".*?</p>)',
            re.DOTALL,
        )
        markup, grouped = pattern.subn(
            lambda m: '<div class="figure-group">' + m.group(1) + m.group(2) + "</div>",
            markup,
        )
        if grouped != 8:
            raise ValueError("Eight figure captions must pair to exactly eight actual image bytes")
        markup = markup.replace(
            "</style>",
            """h2.figures-section{break-before:page}
.figure-group{break-inside:avoid;page-break-inside:avoid;margin:8pt 0 13pt}
.figure-group p{margin:0 0 4pt;break-inside:avoid}
.figure-group img.figure{max-height:205mm;break-inside:avoid}
</style>""",
        ).replace("<h2>Figures</h2>", '<h2 class="figures-section">Figures</h2>')
    preview = folder / f"{name}_browser_word_content.pdf"
    with tempfile.TemporaryDirectory(prefix="robust-v2-final-review-browser-") as scratch:
        tmp = Path(scratch)
        html = tmp / "index.html"
        html.write_text(markup, encoding="utf-8")
        args = [
            str(CHROME),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--disable-extensions",
            "--no-default-browser-check",
            "--no-pdf-header-footer",
            "--user-data-dir=" + str(tmp / "chrome_profile"),
            "--print-to-pdf=" + str(preview),
            html.as_uri(),
        ]
        proc = subprocess.run(args, capture_output=True, text=True, timeout=120, check=False)
        if proc.returncode or not preview.is_file() or preview.stat().st_size < 1000:
            raise RuntimeError("Browser preview failed: " + proc.stderr[-500:])
    pdf = pdfium.PdfDocument(str(preview))
    thumbnails = []
    recorded_pages = []
    for i, page in enumerate(pdf, 1):
        im = page.render(scale=1.25).to_pil().convert("RGB")
        out = folder / f"PAGE_{i:02d}.png"
        im.save(out, optimize=True)
        mini = im.copy()
        mini.thumbnail((500, 710))
        thumbnails.append(mini)
        recorded_pages.append(
            {"page": i, "sha256": sha(out), "image": out.relative_to(ROOT).as_posix()}
        )
    contacts = []
    for offset in range(0, len(thumbnails), 6):
        canvas = Image.new("RGB", (1540, 1510), "#e4e9ef")
        draw = ImageDraw.Draw(canvas)
        for j, img in enumerate(thumbnails[offset : offset + 6]):
            col, row = j % 3, j // 3
            x = 10 + col * 510 + (500 - img.width) // 2
            y = 30 + row * 745
            canvas.paste(img, (x, y))
            draw.text(
                (18 + col * 510, 8 + row * 745), f"{name}: page {offset + j + 1}", fill="#17202a"
            )
        file = folder / f"CONTACT_{offset // 6 + 1:02d}.png"
        canvas.save(file, optimize=True)
        contacts.append({"path": file.relative_to(ROOT).as_posix(), "sha256": sha(file)})
    return {
        "word_source_path": word.relative_to(ROOT).as_posix(),
        "word_sha256": sha(word),
        "browser_pdf_sha256": sha(preview),
        "browser_pdf_path": preview.relative_to(ROOT).as_posix(),
        "page_count": len(pdf),
        "source_figures": figures,
        "source_tables": tables,
        "source_paragraphs": paragraphs,
        "figure_caption_image_groups": grouped,
        "browser_derived_not_native_microsoft_word": True,
        "all_pages_visually_inspected": False,
        "pages": recorded_pages,
        "contact_sheets": contacts,
    }


def main() -> None:
    if RECORD.exists():
        raise FileExistsError("Existing Word browser evidence must be preserved")
    manuscript = render("manuscript", ROOT / "paper/LATEST_MANUSCRIPT_FOR_AVA.docx", True)
    result = {
        "schema": "focused-author-working-Word-content-browser-preview-v1",
        "main": manuscript,
        "technical_comparator": "Archived separately; not part of the working journal manuscript",
        "claim": "Ready for visual inspection, not yet evidence of an inspection",
        "native_microsoft_word_page_layout_verified": False,
    }
    RECORD.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "FOCUSED_WORKING_DRAFT_RENDERED",
        manuscript["page_count"],
        "pages, caption groups",
        manuscript["figure_caption_image_groups"],
        flush=True,
    )


if __name__ == "__main__":
    main()
