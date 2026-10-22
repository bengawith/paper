# Historical Reviewer Responses — Proposed Controlled-Study Revision (Author Working Draft)

**Status: prepared for coauthor consideration, not a response submitted to a journal.** This document addresses the earlier EAAI-26-2180 reviews. It does not claim that the earlier editor has invited or accepted a new submission. The current proposed manuscript is **“Geometric Provenance and Empirical Reliability in Machine-Learning Aerofoil Optimisation: A Controlled XFOIL Study.”** Every declaration and substantive response requires author approval before sending.

## Earlier Reviewer 2 — Practical usefulness, adversarial tests, and selected-design verification

We separated the original historical examples from the final controlled study rather than promoting old numerical improvements into a corrected model result. The original historical same-XFOIL calculations gave an 18.7% nominal weighted-drag decrease for one selected example and favourable completed-case perturbation means; those figures belong only to the original imperfect model/geometry lineage.

In the new exact-CST controlled study, 1,256 source geometries were attempted with genuine local-XFOIL labels. The 10-seed risk-aware and independent 10-seed constrained nominal-only search were both completed with 128 members and 200 generations. Finite risk-front agreement passes both retained numerical checks, but is not proof of a global optimum.

The newly selected strict XFOIL verification **attempted 819 nominal and displaced cases; only 290 satisfy the full 45-angle, required-lift and clean-exit conditions**. The new nominal reference, risk-knee and nominal-control polars all miss required angles and **have zero usable nominal matched pairs**. A nominal percentage improvement cannot be reported under the preregistered protocol. The risk-knee shows favourable **15.2% and 14.9%** mean reductions on the two 0.5%-chord completed matched subsets (n=15 and 14, respectively). The 1%-chord risk-knee has no complete matched subset. These differences cannot be extrapolated to missing solver cases, manufacturing populations, independent RANS, experiments or a tail probability. Every failed case is retained. An explicitly labelled post-hoc equal-control higher-iteration check did not recover any of the three full nominal sweeps; it does **not** replace the original 200-iteration criterion.

## Earlier Reviewer 3 — Source data, geometry validity, architecture and solver fidelity

The original audit found incorrect interpretation of two-block coordinate inputs, changing 1,001 of 1,560 historical CST encodings. After correcting that, fresh numerical comparisons further demonstrated that rotating/resampling a geometry can change XFOIL coefficients at the same recorded incidence. The original archived geometry–label map therefore cannot establish the physical performance of the repaired representation.

Instead we generated new, source-hashed XFOIL observations **on the exact CST reconstructions supplied to the predictor**: 106,931 eligible states from 1,224 observed geometries, against 141,928 requested states. The other 34,997 are missing or unusable, not filled. Five fixed-architecture neural members were genuinely retrained on development geometry clusters, using development-only out-of-bag stopping. Protected historical-identity test performance has R² 0.99778 for lift, 0.98176 for log drag and 0.98817 for moment, but these scores are conditional on solver-observed states and previously inspected geometry identities.

A separate 200-geometry numerical confirmation was selected and sealed before training, then its numerical predictions and risk scores frozen before new XFOIL outcomes. Of 22,600 requested states, 17,968 were eligible solver observations. This is a meaningful prospective *numerical* assessment, **not independent CFD** or an unseen-airfoil-family experiment. Independent converged RANS at equal achieved lift, with mesh/domain/transition sensitivity and moment/constraints accounted for, remains necessary.

## Earlier Reviewer 6 — Figures, stall/high incidence, and numerical robustness

The updated paper contains seven numbered figures and one supplement, seven tables, and no claims of prediction quality in unlabelled/non-converged regimes. Failure at XFOIL angles, post-stall duplicate lift roots, nonfinite/negative drag and untrusted candidate scenarios remains explicit. Strict selected-case partial polars are not completed by interpolation. The archived finite adversarial search is retained as historical evidence rather than presented as the current worst-case certificate.

All 14 pages of the *Chromium-derived Word-content preview* and the separate one-page referee report preview were visually inspected; original layout defects were corrected and all rendered pages preserved. **Native Microsoft Word pagination parity has not been established** and must be reviewed by a coauthor on their installation before journal submission.

## Earlier Reviewer 7 — Clarity and manuscript independence

The previous V1 draft remains separate and is not concatenated with the new paper. The current study uses a methods-first engineering account of where geometry provenance, support screening, numerical solver missingness, candidate-specific prescribed uncertainty and finite-search validation matter. Historical findings, revised results, adverse nominal comparisons, technical acceptance and independent physical validation are differentiated rather than merged.

## Earlier Reviewer 8 — Novelty, baselines, model choice and trust utility

The new fixed-architecture surrogate is compared with ridge, histogram gradient boosting and a weighted mean under the same admitted development/test data. Its favourable numerical accuracy is not described as a controlled superiority claim across unrelated aerodynamic domains.

Existing CST representations, NSGA-II, CVaR, deep ensembles, empirical/conformal prediction and confidence-guided airfoil tools—including NeuralFoil—are cited; no first-ever integration claim is made. Within the corrected calibration-only near-duplicate-geometry separated study, high-error AUROC rises from 0.7532 (support-only) to 0.8893 (combined). On the separately sealed and numerically observed confirmation, support-only AUROC is 0.5346, combined 0.8019, while **disagreement alone performs better still (0.8210)**. These are ranking diagnostics, not a new deployment threshold or a distribution-free adaptive-design guarantee. The manufacturing field is a prescribed sensitivity test, not measured production variation.

## Remaining response and author decisions

The software, complete finite search, fixed numerical confirmation and complete raw-case accounting now have executable receipts, current source/artefact hashes and a separate major-revision referee report. They do **not** resolve the demands for independent RANS/experimental comparison, unconditional manufacturing-risk inference, absent solver labels, comprehensive unseen-family transfer or proof of global optimality. Coauthors must verify authorship/affiliations, funding and conflict declarations, AI-use statements, journal policy and whether this controlled numerical study is a suitable contribution.

**Publication recommendation: major scientific revisions before submission.** The earlier reviews are historical; this is an author working response, not evidence that a reviewer or editor has accepted those answers.
