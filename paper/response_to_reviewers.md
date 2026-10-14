# Response to Reviewers

> Read alongside [REVIEW_STATUS.md](REVIEW_STATUS.md), which documents how each internally-flagged issue was corrected. A selected-design claim is checked numerically by local XFOIL; independent CFD and larger tail campaigns are outstanding. Journal choice and the prior venue's one-year resubmission restriction remain author decisions.

**Manuscript (previous):** "Data-Driven Aerodynamic Prediction: High-Fidelity Lift Curve Estimation for Aerofoils Using Class-Shape Transformation and Artificial Intelligence" (EAAI-26-2180).

**Current revised manuscript:** "Trust-Calibrated Machine-Learning Surrogates for Risk-Aware Aerofoil Design under Manufacturing Uncertainty, with Direct XFOIL Verification."

We thank the reviewers for their constructive assessment. The work has been substantially reframed from a lift-curve prediction study into a robust-design and surrogate-trust framework, which directly addresses the central concern shared across reviews—that predictive accuracy alone does not establish engineering value. Every response below is backed by a concrete, reproducible result in the revised manuscript. All quantities are drawn from durable evidence artefacts (study lineage `robust-v2-current-20260908`, Re = 1×10⁶, M = 0, N_crit = 9).

---

## Reviewer #2

> *"The authors fail to demonstrate the actual usefulness of the approach in a pre-design exploration application… it would be useful to implement an optimization task aimed at finding the airfoil that maximizes the difference between the results of their ML approximator and those obtained with XFOIL using the same CST parameterization."*

This is now the centrepiece of the paper and is implemented as a finite targeted search (not a global proof) (Section 3.5, Section 4.5, Figures 6–7). An adversarial search over the CST design box, using XFOIL as the oracle and the identical CST parameterisation, locates the aerofoil maximising surrogate–solver lift disagreement. The worst case (ΔC_L = 0.67) occurs entirely outside the surrogate's calibrated trust domain, whereas the sole wholly trusted adversarial candidate has a mean disagreement of 0.043; the in-screen adversarial sample is too small for generalisation. We go further and make the design consequence explicit (Section 4.4): a surrogate-optimised aerofoil that sits outside the trust domain is verified by XFOIL to be 87% worse than a NACA 2412 reference, while a design constrained to the trust domain is verified to be 18.7% better at nominal conditions and 8.8% better in expectation under manufacturing uncertainty. Usefulness is thus demonstrated, and one observed high-error example is flagged by the trust screen; a universal worst-case bound is not established.

> *"the work is characterized by many repetitions and redundancies, which make it difficult to read."*

The manuscript has been rewritten from scratch around three clearly delineated contributions with no repeated derivations; redundant restatements of the loss and preprocessing have been removed.

## Reviewer #3

> *"clarify… what the main contribution of this work is and what has not been explored… compared with existing surrogate modelling approaches."*

The Introduction presents three linked contributions: a multi-output surrogate with measured accepted-set error/coverage, a prescribed geometric-uncertainty optimisation, and a selected direct-XFOIL numerical check plus finite adversarial probing. No CFD or experimental results are yet claimed. The specific contribution is an empirical acceptance screen applied to a selected numerical design comparison. Related robust optimisation and design-shift literature is explicitly discussed; calibration or robust optimisation alone is not claimed as novel.

> *"The training data are generated using XFoil rather than experimental data or high-fidelity CFD… the model essentially learns to reproduce XFoil results."*

This is acknowledged explicitly (Section 2, Section 5) and structurally addressed: the framework is prepared for a higher-fidelity extension, but presently has only XFOIL-class numerical evidence, using local XFOIL as a numerical-consistency cross-check rather than only as a label source, and is designed to accept selective CFD/wind-tunnel augmentation. The trust calibration and verification apparatus are label-source-agnostic.

> *"XFoil is known to be less reliable in separated-flow regimes… discuss how this limitation may affect predicted stall characteristics."*

Section 4.5 records 18 fully converged sweeps among 41 evaluated adversarial candidates and explicitly reports failed XFOIL cases. The largest *observed* surrogate disagreement is outside the empirical trust screen, but solver failures cannot be treated as measured aerodynamic discrepancies, and the in-screen adversarial sample is too small for a general guarantee.

> *"edge padding… may introduce artificial patterns."*

Eliminated. The surrogate is now *point-conditioned*: it predicts a coefficient at a single queried angle, so only genuinely measured states are ever used as supervision. Fixed-grid padding and edge extrapolation—the source of the artefacts the reviewer identified—are no longer present.

> *"All data correspond to a single Reynolds number and Mach number… applicability is limited."*

The single condition is now stated explicitly as a deliberate scoping choice (Section 2) that isolates the robustness and trust contributions, with a clear extension path (Section 5): the point-conditioned architecture and data-driven trust model extend to a Re/Mach-parameterised envelope by adding a conditioning input and labelled data, leaving the methodology unchanged.

> *"All figures should have better quality and clarity."*

All figures are regenerated at 150 dpi with 14-point base fonts, explicit subplot labels and descriptive captions (Figures 1–6).

## Reviewer #6

> *"Add (a), (b),… for the subplots… and provide brief descriptions."*

Done for all multi-panel figures (Figures 1 and 5), with per-panel descriptions in the captions and text.

> *"Address why the predicted and reference curves are not correlated well for the high angles."*

The revised framework answers this quantitatively rather than qualitatively: high-angle discrepancy is now shown to be a trust-domain phenomenon (Section 4.1, Table 1; Section 4.5). Within the calibrated trust domain the lift R² is 0.994; errors are higher on rejected states and high-incidence solver difficulties remain and coincides with XFOIL's own convergence difficulties.

> *"Use larger fonts in all the figures."*

Done (14-point base, 15-point titles).

## Reviewer #7

> *"The paper is well written."*

We thank the reviewer.

## Reviewer #8

> *"insufficient novelty… What else is gained apart from computational time?"*

Beyond speed, the paper examines empirical trust screening, prescribed bounded geometry perturbations and a three-design XFOIL case comparison. These offer a practical way to flag extrapolation risk and assess selected design decisions. The CLI labels predicted outputs with trust status, but neither arbitrary geometries nor physical manufacturing outcomes are universally guaranteed.

> *"the developed model… is doubtful… which model?"*

We now justify the model choice quantitatively rather than assume it (Section 4.1, Table 2): under the identical grouped protocol the multilayer perceptron is compared against histogram gradient boosting, ridge regression and a Dummy baseline. We deliberately avoid over-claiming—gradient boosting is marginally stronger on the drag subset—and instead motivate the perceptron by its competitive accuracy, smooth differentiable multi-output map, and the ensemble-disagreement signal that powers the trust calibration. Consistent with the co-author guidance, we did not undertake a further large architecture search.

> *"The ranges of applicability/operating conditions (Re and M) should be clearly stated… limitations should be clearly listed."*

Operating conditions are stated in the Abstract, Section 2 and throughout (Re = 1×10⁶, M = 0, N_crit = 9). Section 5 lists three explicit limitations: the prescribed (not measured) manufacturing model, panel-method labels, and the moderate seed reproducibility of the robust Pareto front.

> *"What happens with the drag and pitch moment curves, why weren't they also considered?"*

Both are now first-class predicted quantities (Table 1: C_D R² = 0.922 in log-space, C_M R² = 0.918) and enter the optimisation as objectives and constraints.

> *"Physics-aware validation score is never explained… validity of the starting dataset is doubtful… which three neural architectures…"*

The previous ad-hoc "Physics Score" and multi-architecture search have been removed in line with the guidance to focus on application rather than architecture. Model selection is now a transparent, grouped, leakage-controlled protocol with a calibrated trust criterion. Dataset construction uses grouped cluster splits with explicit provenance and no padded/extrapolated targets.

> *"Is the developed model truly satisfactory? … Is this truly scientifically significant?"*

The revised contribution is not a lift-curve fit but a robust-design framework with a measured reliability envelope, re-evaluated using a separate local XFOIL run from the same solver family and stress-tested adversarially. Its significance lies in making surrogate-driven aerodynamic design *defensible*: it states, and empirically validates, the boundary of surrogate trust.

> *"Newer, more relevant references should be cited; discussion should be much improved; passive form should be employed more."*

The literature framing (Section 2) and Discussion (Section 5) are expanded, and the register has been shifted toward an impersonal, methods-first style.

---

We believe the revised manuscript resolves the substantive concerns—demonstrated usefulness, drag/moment coverage, honest treatment of solver limitations, and clearly stated scope—while introducing a reproducible integration of empirical screening and selected design checks, pending CFD and larger uncertainty verification.


## Additional verification and trust-component audit (14 October 2026)

The strict re-evaluation now contains two independently seeded 128-scenario XFOIL campaigns at 0.5%-chord prescribed geometric deviation. Their full-sweep and target-lift-feasible matched-case counts are 92 and 96, and the selected within-screen candidate has paired mean weighted-drag reductions of 15.4% and 15.8%, respectively, among those successful pairs. The corresponding empirical upper-tail summaries also favour the candidate on the completed subset, but nonrandom solver failures prevent an unconditional manufacturing-risk or population-CVaR claim. At the 1%-chord stress level the earlier small-sample tail remains unfavourable, and no CFD validation is yet available.

A component audit finds that all 9,050 historical rejected held-out states failed the calibrated support-distance cutoff; ensemble-disagreement thresholds did not independently change acceptance. We therefore attribute the observed error-versus-coverage effect to the **frozen support-dominated screening rule**, not to a separately proven ensemble-disagreement gate. A new calibration-scaled disagreement/support score is evaluated descriptively, without modifying the frozen model or choosing a threshold using the test data. A stronger operational trust model would require prospective validation on a fresh confirmation cohort.

## Coauthor and CFD handoff

These responses describe the evidence available before an independent CFD/RANS addition. The 1%-chord campaign contains 10 matched pairs; two separate 0.5%-chord campaigns contain 92 and 96 matched completed cases. Solver failures still prevent unconditional population-tail conclusions. A new venue's reviewer response must be adapted to its actual review history, and the journal's resubmission policy must be checked by the authors. Selected CFD and a larger independent solver campaign should be incorporated before asserting physically verified manufacturing robustness.
