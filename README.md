# Trust-aware machine-learning surrogate for aerofoil design

This repository contains the latest research code, evidence and manuscript by Benjamin I. G. Gawith, Ava Shahrokhi and Keivan Navaie. The next stage is independent CFD verification and a full corrected-physics optimisation acceptance run.

**Status: complete author-review draft; not yet accepted for journal submission.**

## Scientifically supported results

The fixed five-member point-conditioned MLP predicts lift, logarithmic drag and pitching moment from a compact 12-parameter CST geometry plus angle of attack. On 26,692 held-out, geometry-grouped XFOIL-labelled states from 233 aerofoils, its R² is 0.989, 0.922 and 0.918 respectively.

The frozen empirical screening rule accepts 66.1% of held-out states with lower error and higher observed interval coverage. A component audit shows its decisions were effectively support-distance-driven, **not independently improved by its original ensemble spread cutoffs**. A separate five-fold airfoil-group cross-validation on 26,933 calibration rows shows that disagreement adds predictive information: AUROC for identifying high-residual cases improves from 0.767 (support alone) to 0.866 (combined), +0.099 [0.069,0.139] under 300 cluster-bootstrap replicates. This result has not been prospectively validated on an untouched optimiser-selection cohort and does not amend the frozen deployed trust rule.

A selected in-screen design has 18.7% lower nominal local-XFOIL required-lift weighted drag than NACA 2412. Two separate 128-scenario XFOIL comparisons at 0.5%-chord prescribed perturbations show paired mean reductions of 15.4% and 15.8% among the 92 and 96 **successfully completed matched cases**, respectively. Because unsuccessful solver cases are excluded, these are conditional descriptive outcomes, not unconditional manufacturing-risk probabilities. At 1%-chord stress, the small sample maximum is worse for the selected design.

## Important optimisation correction

The historic ten-seed Pareto study applied CST coefficient increments fitted to the reference aerofoil to other candidates. That implementation does not represent physically equal surface-normal variation. The source now generates the same latent uncertainty draw separately on each candidate's actual geometry, performs validity checks, refits CST, and rejects stale optimisation caches. The five historic selected shapes were re-evaluated; **no production robust Pareto optimality is claimed** from that fixed-design test.

Ten independent **low-budget corrected** NSGA-II runs (population 32, generations 10, scenarios 16) all returned feasible solutions, with mean symmetric front-distance 0.149 below the 0.20 criterion. The full production design (population 128, generations 200, scenarios 64, ten seeds) has **not** been rerun under the corrected physics; its agreement and candidate choice remain open.

The uncertainty field uses correlated, continuous but **piecewise-linear** control interpolation and per-draw scaling, with leading/trailing-edge taper. It is a prescribed geometry perturbation model, not a distribution measured from manufacturing.

## What is ready

- Latest standalone Word manuscript: paper/LATEST_MANUSCRIPT_FOR_AVA.docx
- Canonical Markdown manuscript: paper/manuscript.md
- Author scientific review: reports/robust_v2/submission_review_20261014/FINAL_RESEARCH_RESOLUTION.md
- Trust incremental test: reports/robust_v2/submission_review_20261014/TRUST_INCREMENTAL/TRUST_INCREMENTAL_CV.json
- Corrected fixed-design objective audit: reports/robust_v2/submission_review_20261014/CORRECTED_OPTIMISATION/CORRECTED_SELECTED_DESIGNS.json
- Corrected low-budget front: reports/robust_v2/submission_review_20261014/CORRECTED_OPTIMISATION/CORRECTED_SMOKE_FRONT.json
- Ava's proposed CFD inputs and instructions: reports/robust_v2/submission_review_20261014/CFD_HANDOFF/

Run unit validation:

    uv sync --extra dev --extra benchmarks --extra docs
    uv run ruff check src/robust_airfoil tests/robust_v2 scripts/robust_v2
    uv run mypy src/robust_airfoil
    uv run pytest tests/robust_v2 -m "not network and not xfoil and not slow" -q

A sample user-tool command:

    uv run robust-airfoil predict --cst="-0.172,-0.065,-0.125,-0.055,-0.083,0.208,0.190,0.223,0.180,0.218,-0.041,0.002" --alpha-start 0 --alpha-end 10 --alpha-step 2

The original frozen source/data lineage remains preserved separately. Its historical objectives must not be confused with the corrected source's physics. The local OneDrive READY_FOR_REVIEW folder contains the independent author review and CFD handoff; the full model/data archives remain local and hashed.

The study is valuable as ML prediction plus empirically evaluated support/disagreement and selected design-verification evidence. It is **not independent physical validation**: both training labels and direct checks use XFOIL-class models. Ava's selected, mesh-verified CFD will determine whether the aerodynamic ranking survives at higher fidelity. The final paper should only be submitted after coauthor review, a valid full-budget optimisation (or explicit journal-approved narrowing of that claim), and the target journal's required checks.
