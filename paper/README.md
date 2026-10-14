# Manuscript package — corrected author-review version (14 October 2026)

**Status: prepared for Ben's scientific review and Ava's selected CFD verification. NOT yet journal-submission-ready.** No journal acceptance is guaranteed.

## Start here

1. Read `reports/robust_v2/submission_review_20261014/REVIEW_REPORT.md` (from repository root).
2. Review the exact current `paper/manuscript.md` and the exported `MANUSCRIPT_AUTHOR_REVIEW.docx`.
3. Inspect all seven regenerated figures, especially the strictly rerun design-verification Figure 5.
4. Inspect `paper/response_to_reviewers.md` and `CFD_HANDOFF/CFD_PROTOCOL_FOR_AVA.md`.
5. Have Ava run selected CFD before claiming independent aerodynamic/physical verification.

## Contents

| File | Purpose |
|---|---|
| `manuscript.md` | Canonical revised manuscript; all headline assertions now evidence-limited |
| `response_to_reviewers.md` | Historical review response rewritten conservatively, not a response to a future journal |
| `data/paper_data.json` | Consolidated numeric data: frozen ML/uncertainty/optimisation lineage plus fresh strict-design XFOIL results |
| `figures/` | Seven regenerated, 300-dpi figures |
| `REVIEW_STATUS.md` | Superseded 3 October note, preserved as history |

Additional review documents and CFD inputs: `reports/robust_v2/submission_review_20261014/`.

## Current validated scientific numbers

- XFOIL-derived held-out grouped states: 26,692 from 233 aerofoils.
- R²: CL 0.989, **log CD** 0.922, CM 0.918. A further frozen-model selective-error diagnostic (Supplementary Figure S1) reports calibration-scaled score versus held-out error (Spearman 0.476/0.452/0.524 for CL/logCD/CM); it does not select a new model.
- Accepted-by-trust fraction: 66.1%. Empirical accepted-set band coverage (CL/logCD/CM): 97.8/96.5/99.2%. These are not guaranteed conditional coverage on optimizer-selected shapes.
- Direct local XFOIL, selected designs: NACA2412 nominal weighted CD 0.007446 versus selected within-screen 0.006051 (18.7% reduction). The deliberately outside-screen design has nominal weighted CD 0.013960 (87% above reference).
- Strict 1%-chord perturbation runs: 13/16 completed/target-lift-feasible scenarios for reference; 13/16 for within-screen; 14/16 outside. Marginal means 0.008349 versus 0.007615 (8.79% lower on successful-case subsets), with **10** common valid within-screen/reference pairs.
- The empirical 95% CVaR from 13 cases reduces to one maximum and is **not a reliable population tail estimate**. The within-screen maximum is higher than the reference.
- Surrogate Pareto-front seed agreement still fails (0.283 > 0.20).
- Independent RANS CFD and experiments remain outstanding.

## Provenance and reproduction

The historical modelling evidence is from frozen lineage `robust-v2-current-20260908`. The **direct design-verification portion** was replaced by a new stricter XFOIL re-evaluation, stored separately at `results/robust_v2/submission_review_20261014/xfoil_strict/` and bound by hash in `paper/data/paper_data.json`.

To regenerate figures:

```powershell
uv run python scripts/robust_v2/make_paper_figures.py
```

To check the revised workflow:

```powershell
uv run ruff check src/robust_airfoil tests/robust_v2 scripts/robust_v2
uv run mypy src/robust_airfoil
uv run pytest tests/robust_v2 -m "not network and not xfoil and not slow" -q
```

The original package and earlier XFOIL evidence are retained. Do not silently rewrite them; use new revisioned evidence for CFD and subsequent experiments.


## 14 October: Additional ML and XFOIL evidence

Two separately scrambled 128-scenario 0.5%-chord XFOIL studies give 92 and 96 strictly completed matched cases and paired mean drag reductions of 15.4% / 15.8%. These results are conditional on solver completion; no independent CFD evidence or unconditional tail guarantee. A frozen-model audit also finds historical acceptance was effectively support-distance-driven, not independently gated by disagreement. See ML_DIAGNOSTICS and the supplement in manuscript.
