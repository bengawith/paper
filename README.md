# Robust V2 — geometry-consistent aerodynamic selection study

The canonical paper is [paper/manuscript.md](paper/manuscript.md), with a reproducibly generated coauthor Word copy at [paper/LATEST_MANUSCRIPT_FOR_AVA.docx](paper/LATEST_MANUSCRIPT_FOR_AVA.docx). The frozen numerical study is complete; **independent CFD has not been performed**.

**For Ava:** [paper/CFD_BRIEF_FOR_AVA.md](paper/CFD_BRIEF_FOR_AVA.md) and [paper/CFD_HANDOFF_CURRENT.zip](paper/CFD_HANDOFF_CURRENT.zip). This **20 October 2026** handoff contains the frozen reference, robust risk-knee and constrained nominal control, with original design identities and SHA-256 case manifest. The 14 October package selected different shapes and is historical only. [paper/COAUTHOR_NOTES.md](paper/COAUTHOR_NOTES.md) identifies the remaining author decisions.

## Reproducibility map

- Current controlled lineage: results/robust_v2/lineages/robust-v2-controlled-20261020/
- Fixed selected experiments, solver ledger, fresh 2,048-scenario sensitivity and SHA-256 receipts: its selected_study/ subdirectory.
- Controlled configurations: configs/robust_v2/; implementation: src/robust_airfoil/; study and reconciliation utilities: scripts/robust_v2/.
- Archived model and raw numerical files are retained with their hashes; not every large third-party/source artifact is ordinary small Git text. Check the relevant frozen manifest before attempting a reconstruction.
- The mean pitching-moment constraint actually executed is over 33 service-incidence points (0°–8°, step 0.25°), **not** a pointwise guarantee. The nominal control fails the alternative earlier five-incidence pointwise allowance at 6° and 8°. Fresh high-amplitude CVaR95 values include the stated candidate-specific trust penalties. Missing numerical solver observations have not been imputed.

## Working environment and targeted checks

Use Python **3.11–3.13** with [uv](https://docs.astral.sh/uv/) and the locked project environment:

~~~console
uv sync --extra dev --extra benchmarks --extra docs
uv run pytest tests/robust_v2 -m "not network" -q
uv run ruff check src/robust_airfoil tests/robust_v2 scripts/robust_v2
uv run mypy src/robust_airfoil
uv run python -B scripts/robust_v2/verify_final_author_pack.py
~~~

XFOIL-dependent checks require the configured solver executable; skip or mark them explicitly when not available. The historical requirements.txt is not the authoritative locked Robust V2 setup. Do not rerun full training, 10-seed searches or the 200-shape solver cohort to review the frozen evidence.

After a source edit, generate the two selected-design figures from the frozen JSON and rebuild Word:

~~~console
uv run python scripts/robust_v2/render_current_selected_figures.py
uv run python scripts/robust_v2/make_latest_ava_paper.py
~~~

Earlier dated figure/Word exporters are archival and may use earlier lineage definitions; **do not regenerate the current paper with those scripts**. Software checks and XFOIL repeatability are not independent viscous-CFD validation. Claims about physical rankings, manufacturing probabilities and journal acceptance remain conditional on Ava's study and coauthor review.
