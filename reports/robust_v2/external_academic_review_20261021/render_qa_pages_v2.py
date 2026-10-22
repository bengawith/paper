"""Produce page-by-page visual inspection surfaces for the 21-page rewrite."""

from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent / "RENDERED_V2"
doc = HERE / "STANDALONE_AUTHOR_VOICE_FINAL.browser_preview.pdf"
pdf = pdfium.PdfDocument(str(doc))
count = len(pdf)
pages = []
for n, page in enumerate(pdf, 1):
    im = page.render(scale=1.5).to_pil().convert("RGB")
    output = HERE / f"PAGE_{n:02d}.png"
    if output.exists():
        raise FileExistsError("Previous page render must be preserved " + str(output))
    im.save(output, optimize=True)
    thumb = im.copy()
    thumb.thumbnail((540, 760))
    pages.append(thumb)
for offset in range(0, count, 6):
    canvas = Image.new("RGB", (1660, 1610), "#e4e8ed")
    pen = ImageDraw.Draw(canvas)
    for index, im in enumerate(pages[offset : offset + 6]):
        col, row = index % 3, index // 3
        x = 10 + col * 550 + (540 - im.width) // 2
        y = 35 + row * 800
        canvas.paste(im, (x, y))
        pen.text((20 + col * 550, 10 + row * 800), f"Paper page {offset + index + 1}", fill="black")
    p = HERE / f"CONTACT_SHEET_{offset // 6 + 1:02d}.png"
    canvas.save(p, optimize=True)
    print("CONTACT_SHEET", p.name, flush=True)
print("ALL_PAGE_PREVIEWS_CREATED", count, flush=True)
