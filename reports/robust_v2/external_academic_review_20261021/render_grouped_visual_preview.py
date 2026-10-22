"""Render final DOCX with figure captions and images kept together in browser.

This is DOCX-content derived Chromium pagination; never claim native Word pages.
"""

import json
import re
import runpy
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORD = ROOT / "paper/LATEST_MANUSCRIPT_FOR_AVA.docx"
PDF = HERE / "RENDERED_V2/STANDALONE_AUTHOR_VOICE_FINAL.browser_preview.pdf"
SOURCE = (
    ROOT
    / "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/render_author_docx_browser.py"
)
CHROME = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
if PDF.exists():
    raise FileExistsError("A distinct PDF revision already exists")
PDF.parent.mkdir(parents=True, exist_ok=True)
html_doc = runpy.run_path(str(SOURCE))["html_doc"]
markup, images, tables, paragraphs = html_doc(WORD)
markup = markup.replace("<h2>Figures</h2>", '<h2 class="figures-section">Figures</h2>')
pattern = re.compile(
    r'(<p><strong>(?:Figure [1-7]\.|Supplementary Figure S1\.)</strong>[^<]*</p>)\s*(<p><img class="figure".*?</p>)',
    re.DOTALL,
)
markup, changed = pattern.subn(
    lambda m: '<div class="figure-group">' + m.group(1) + m.group(2) + "</div>", markup
)
if changed != 8:
    raise ValueError(
        "Could not reliably pair all 8 figure captions with their actual image paragraphs: "
        + str(changed)
    )
markup = markup.replace(
    "</style>",
    """h2.figures-section{break-before:page;page-break-before:always}
.figure-group{break-inside:avoid;page-break-inside:avoid;margin:8pt 0 13pt;}
.figure-group p{margin:0 0 4pt;break-inside:avoid}
.figure-group img.figure{max-height:205mm;break-inside:avoid}
</style>""",
)
with tempfile.TemporaryDirectory(prefix="robust-v2-final-page-") as directory:
    folder = Path(directory)
    entry = folder / "document.html"
    entry.write_text(markup, encoding="utf-8")
    args = [
        str(CHROME),
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--disable-extensions",
        "--no-default-browser-check",
        "--no-pdf-header-footer",
        "--user-data-dir=" + str(folder / "chromeprofile"),
        "--print-to-pdf=" + str(PDF),
        entry.as_uri(),
    ]
    process = subprocess.run(args, capture_output=True, text=True, timeout=120, check=False)
    if process.returncode or not PDF.is_file() or PDF.stat().st_size < 1000:
        raise RuntimeError("Browser print failed: " + str(process.stderr[-300:]))
print(
    json.dumps(
        {
            "status": "FINAL_DOCX_DERIVED_BROWSER_PREVIEW",
            "paired_figures": changed,
            "figures": images,
            "tables": tables,
            "word_paragraphs": paragraphs,
            "pdf": str(PDF),
            "native_word_pagination_verified": False,
        },
        indent=2,
    )
)
