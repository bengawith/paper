import runpy
from pathlib import Path

f = runpy.run_path(
    "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/render_author_docx_browser.py"
)["html_doc"]
markup, *_ = f(Path("paper/LATEST_MANUSCRIPT_FOR_AVA.docx"))
a = markup.find("Figure 1.")
b = markup.find('<img class="figure"')
print("F1_CONTEXT", repr(markup[a - 110 : a + 360]))
print("IMAGE_CONTEXT", repr(markup[b - 90 : b + 50]))
print("COUNT", markup.count("Figure 1."), markup.count('<img class="figure"'))
