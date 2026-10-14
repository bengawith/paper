# Trust-screened ML for risk-aware aerofoil design

This repository accompanies the revised research manuscript by Benjamin Gawith, Ava Shahrokhi and Keivan Navaie.

**Current status (14 October 2026): author-review draft, ready for selected RANS CFD work with Ava. Not journal-submission-ready.** The previous 3 October status note is preserved for history, but superseded by the new scientific review.

## What the study demonstrates

A compact CST-and-angle-conditioned five-member MLP ensemble predicts lift, log drag and pitching moment from XFOIL-labelled data. On 26,692 grouped held-out aerodynamic states (233 aerofoils), R² is approximately 0.989 for lift, 0.922 for log drag and 0.918 for moment. An empirical support/disagreement screen accepts 66.1% of the states with lower observed error than rejected states.

A prescribed smooth, bounded surface-normal deviation model is used for expected-drag and 95% upper-tail surrogate objectives. It is **not** a measured manufacturing-tolerance distribution.

A fresh, strict local XFOIL re-evaluation of **three selected** aerofoils shows:

| Selected design | Nominal weighted drag | Full-sweep/required-lift-feasible perturbed cases | Mean drag, accepted cases |
|---|---:|---:|---:|
| NACA 2412 reference | 0.007446 | 13/16 | 0.008349 |
| Within-screen design | 0.006051 | 13/16 | 0.007615 |
| Outside-screen diagnostic optimum | 0.013960 | 14/16 | 0.014137 |

The selected within-screen design improves nominal weighted drag by **18.7%** against NACA 2412. Marginal means from independently successful cases differ by **8.8%**, with ten fully valid paired scenarios. The empirical 95%-tail statistics are just the largest observations from 13–14 samples, **not trustworthy manufacturing-population CVaR estimates**; the within-screen observed maximum is higher than the reference. The ten-seed robust-front agreement criterion also fails (0.283 > 0.20).

**No independent viscous RANS CFD or experimental verification is yet included.** The local XFOIL is a separate numerical implementation/execution of the same solver family as the training labels. Trust-screening performance is measured on held-out historical states, not guaranteed under adaptive optimisation distribution shift.

## Read, review and reproduce

- Canonical manuscript: [paper/manuscript.md](paper/manuscript.md)
- Paper-specific overview: [paper/README.md](paper/README.md)
- Reviewer response (historical, revised): [paper/response_to_reviewers.md](paper/response_to_reviewers.md)
- Full scientific review: [reports/robust_v2/submission_review_20261014/REVIEW_REPORT.md](reports/robust_v2/submission_review_20261014/REVIEW_REPORT.md)
- Coauthor CFD protocol and geometry inputs: [reports/robust_v2/submission_review_20261014/CFD_HANDOFF/CFD_PROTOCOL_FOR_AVA.md](reports/robust_v2/submission_review_20261014/CFD_HANDOFF/CFD_PROTOCOL_FOR_AVA.md)
- Historical frozen source lineage: `robust-v2-current-20260908`
- Fresh strict verification: `results/robust_v2/submission_review_20261014/xfoil_strict/`, with summary/digest incorporated into `paper/data/paper_data.json`.

Install with `uv sync --extra dev --extra benchmarks`, then validate:

```powershell
uv run ruff check src/robust_airfoil tests/robust_v2 scripts/robust_v2
uv run mypy src/robust_airfoil
uv run pytest tests/robust_v2 -m "not network and not xfoil and not slow" -q
uv run python scripts/robust_v2/make_paper_figures.py
```

For strict XFOIL rerun, use the local official `tools/xfoil/xfoil.exe` and an independently named output folder:

```powershell
$env:ROBUST_V2_XFOIL_OUT="results/robust_v2/validation_rerun_YYYYMMDD/xfoil_strict"
uv run python scripts/robust_v2/verify_designs_xfoil.py
```

Run the submitted CFD protocol separately. The next research decision belongs to the authors and depends on matched-lift CFD results, numerical convergence, actual tolerance evidence (if available), journal requirements and prior resubmission restrictions.

Large source caches, frozen weights and per-case solver logs are local/handoff evidence rather than guaranteed public Git artefacts. No claim of physical safety, universal conformal conditional coverage or guaranteed publication acceptance is made.


## 14 October: Additional ML and XFOIL evidence

Two separately scrambled 128-scenario 0.5%-chord XFOIL studies give 92 and 96 strictly completed matched cases and paired mean drag reductions of 15.4% / 15.8%. These results are conditional on solver completion; no independent CFD evidence or unconditional tail guarantee. A frozen-model audit also finds historical acceptance was effectively support-distance-driven, not independently gated by disagreement. See ML_DIAGNOSTICS and the supplement in manuscript.
