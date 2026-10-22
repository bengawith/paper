# Robust V2 — final scientific and implementation review
Date: 14 October 2026

**Decision: READY FOR AUTHOR REVIEW AND AVA'S CFD STUDY; NOT YET A FULLY VALIDATED ROBUST OPTIMUM OR INDEPENDENT PHYSICAL VALIDATION.**

## 1. Ensemble disagreement: measured incremental value without test leakage

The frozen acceptance rule rejected 9,050 of 26,692 held-out XFOIL-labelled states, all of which failed its support-distance criterion. No further states were rejected independently by the stored ensemble-disagreement thresholds.

A new experiment used only the calibration partition (26,933 rows, 234 aerofoils), with five group-separated folds, fitting training-fold-only ridge residual-risk models. On out-of-fold predictions the AUROC for detecting the largest-decile scaled multi-output errors was:

| Risk inputs | AUROC | Average precision |
|---|---:|---:|
| Support distance only | 0.7665 | 0.3728 |
| Ensemble spreads only | 0.8609 | 0.4718 |
| Support plus spreads | 0.8657 | 0.4958 |

The combined AUROC gains 0.0992 over support alone (300 aerofoil-group bootstrap replicates: interval [0.0694, 0.1386]). This demonstrates incremental error-ranking information within the calibration distribution. It does NOT establish conditional conformal coverage or prospectively demonstrate a new operational gate on optimiser-selected shapes. No locked-test-based tuning was performed.

Evidence: TRUST_INCREMENTAL/TRUST_INCREMENTAL_CV.json and ML_DIAGNOSTICS/ml_selective_audit.json.

## 2. Optimisation: real bug identified and corrected in source

The original ten-seed robust NSGA-II code applied reference-geometry-derived CST increments to candidate geometries. That does not represent a common surface-normal deviation for arbitrary designs. It is a substantive scientific operator defect; the historical Pareto front and its failed agreement measure (0.283 compared with 0.20) cannot establish a candidate-specific robust optimum.

The corrected source applies shared latent scrambled-Sobol draws separately to every candidate's own upper and lower surface normals, validates the displaced coordinates, refits CST and records invalid realisations. Historical objective caches (schema v2/v3) are invalidated, and the candidate-specific objective is schema v4. The independent coefficient-perturbation baseline remains a separate, explicitly named study rather than silently substituting for the physical method.

Corrected fixed-design re-evaluation with 64 latent draws gives:

| Previously chosen design (held fixed) | Penalised mean | Penalised CVaR95 | Trust exits |
|---|---:|---:|---:|
| NACA 2412 reference | 0.009421 | 0.019184 | 8/64 |
| Previously selected within-screen knee | 0.009040 | 0.015157 | 18/64 |
| Historical minimum-tail candidate | 0.008938 | 0.015485 | 14/64 |

These objectives include trust-exit penalties and must not be read as pure aerodynamic drag. The selected candidates are historical shapes, not the newly optimised Pareto front. The previous deterministic and simple-AVA designs become infeasible under the corrected evaluation.

Evidence: CORRECTED_OPTIMISATION/CORRECTED_SELECTED_DESIGNS.json, source candidate_uncertainty.py and the linked unit tests.

A ten-seed low-budget corrected-physics probe has been undertaken separately to test execution and approximate repeatability. It is not an accepted replacement for the production population-128, generation-200, ten-seed study. That full corrected campaign remains an acceptance gate unless newly completed and independently verified.

## 3. Direct XFOIL: legitimate case-level advantage, not universal optimality

The selected within-screen aerofoil has 18.7% lower nominal required-lift weighted drag than NACA 2412 in the strict local XFOIL comparison. The outside-screen selected comparator is much worse. This is a genuine selected-case solver result.

At prescribed 1%-chord stress, only 13/16 candidate and reference sweeps are strictly complete and ten scenarios are matched. The empirical 95% CVaR from this set reduces to an extreme sample; the within-screen maximum is higher than the reference.

At 0.5%-chord stress, two independently scrambled 128-scenario campaigns yield 92 and 96 matched complete-sweep cases. Their paired mean drag reductions are 15.4% and 15.8%, conditional on solver convergence. Matched-case empirical upper-tail results also favour the candidate in these two samples, but solver failures make unconditional manufacturing tail-risk statements unjustified.

The training labels and direct numerical check share XFOIL-class modelling assumptions. Independent viscous RANS CFD and eventually experimental evidence are still needed.

Evidence: XFOIL strict summaries and ML_DIAGNOSTICS/XFOIL_EPS005_TWO_SCRAMBLE_AUDIT.json.

## 4. Geometric uncertainty method: description corrected

The implemented random-field controls are spatially correlated and piecewise-linearly interpolated. A separate per-sample maximum normalization and leading/trailing-edge tapers provide bounded continuous displacements. The paper no longer calls this a cubic B-spline generator or a measured manufacturing distribution.

Replacing this field with a new spline or measured uncertainty prior would change the experimental model and require new labels/solver validation. Historic direct-XFOIL outcomes must not be reassigned to such a different model.

## 5. Bibliography and provenance

The paper contains 28 original numbered references. Crossref was queried against primary metadata for scholarly works and 26 entries were augmented with traceable DOI, publisher, conference or author-hosted archive links. Books and the historical AirfoilTools site remain clearly categorised instead of receiving invented DOIs. The PyTorch proceedings pagination was corrected to the published form.

No surrogate parameters or aerodynamic training labels were changed. The manuscript, exported standalone Word draft, numerical audit and selected CFD geometries are backed by SHA-256 manifests.

## 6. Remaining gates before journal submission

1. Full, corrected, ten-seed production multi-objective optimisation with explicit feasibility and unchanged agreement threshold. The current historical Pareto front must remain labelled diagnostic.
2. Optional prospectively evaluated combined support-plus-ensemble screening policy if a stronger operational gate is desired; the frozen deployed historical trust rule remains support dominated.
3. Ava's independent selected CFD: matched lift, near-wall and mesh/domain convergence, transition/turbulence assumptions, equal design constraints and explicit handling of numerical failures.
4. Physical manufacturing tolerance measurements if claims about actual manufacturing variability are required; otherwise retain prescribed geometric stress.
5. Coauthor approval and chosen journal policy, formatting, copyright/third-party data terms, original-submission restrictions and disclosure.

**Author review outcome:** The ML trust signal and direct XFOIL selected-design evidence are useful, reproducible and more informative than a pure predictive-accuracy paper. A publication claim of independently verified robust Pareto optimality remains out of reach until the two major validation gates are completed. This is not concealed by favourable percentages.


## 7. Corrected ten-seed low-budget stability probe

The candidate-specific evaluator completed ten small-budget NSGA-II seeds (32 population, ten generations, 16 shared perturbations each). All ten returned feasible solutions (44 aggregate entries), with a mean normalised symmetric distance 0.149, below the unchanged 0.20 acceptance threshold *for this smoke profile*. The full 128-by-200-by-64 production study has not been completed under the corrected physics; its legacy Pareto result must remain labelled historical.
