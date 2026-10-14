# Response to Reviewers

> Read alongside [REVIEW_STATUS.md](REVIEW_STATUS.md), which documents how each internally-flagged issue was corrected. The central claim is settled by direct XFOIL verification. Journal choice and the prior venue's one-year resubmission restriction remain author decisions.

**Manuscript (previous):** "Data-Driven Aerodynamic Prediction: High-Fidelity Lift Curve Estimation for Aerofoils Using Class-Shape Transformation and Artificial Intelligence" (EAAI-26-2180).

**Resubmission (revised, retitled):** "Risk-Aware Aerofoil Design under Manufacturing Uncertainty using a Trust-Calibrated Machine-Learning Surrogate and Multi-Fidelity Verification."

We thank the reviewers for their constructive assessment. The work has been substantially reframed from a lift-curve prediction study into a robust-design and surrogate-trust framework, which directly addresses the central concern shared across reviews—that predictive accuracy alone does not establish engineering value. Every response below is backed by a concrete, reproducible result in the revised manuscript. All quantities are drawn from durable evidence artefacts (study lineage `robust-v2-current-20260908`, Re = 1×10⁶, M = 0, N_crit = 9).

---

## Reviewer #2

> *"The authors fail to demonstrate the actual usefulness of the approach in a pre-design exploration application… it would be useful to implement an optimization task aimed at finding the airfoil that maximizes the difference between the results of their ML approximator and those obtained with XFOIL using the same CST parameterization."*

This is now the centrepiece of the paper and is implemented exactly as requested (Section 3.5, Section 4.5, Figures 6–7). An adversarial search over the CST design box, using XFOIL as the oracle and the identical CST parameterisation, locates the aerofoil maximising surrogate–solver lift disagreement. The worst case (ΔC_L = 0.67) occurs entirely outside the surrogate's calibrated trust domain, whereas within the trust domain the disagreement collapses to 0.043. We go further and make the design consequence explicit (Section 4.4): a surrogate-optimised aerofoil that sits outside the trust domain is verified by XFOIL to be 87% worse than a NACA 2412 reference, while a design constrained to the trust domain is verified to be 18.7% better at nominal conditions and 11.7% better in expectation under manufacturing uncertainty. Usefulness is thus demonstrated, and the surrogate's worst case is shown to be bounded by the trust flag.

> *"the work is characterized by many repetitions and redundancies, which make it difficult to read."*

The manuscript has been rewritten from scratch around three clearly delineated contributions with no repeated derivations; redundant restatements of the loss and preprocessing have been removed.

## Reviewer #3

> *"clarify… what the main contribution of this work is and what has not been explored… compared with existing surrogate modelling approaches."*

The Introduction now states three specific, novel contributions: a *trust-calibrated* multi-output surrogate with an empirically measured domain of reliability; a physically admissible manufacturing-uncertainty model driving risk-aware optimisation; and a multi-fidelity verification/trust framework including adversarial probing. The distinguishing novelty—an operational, calibrated criterion for *when to believe the surrogate*—is absent from the surrogate-modelling works surveyed in Section 2.

> *"The training data are generated using XFoil rather than experimental data or high-fidelity CFD… the model essentially learns to reproduce XFoil results."*

This is acknowledged explicitly (Section 2, Section 5) and structurally addressed: the framework is now multi-fidelity, using XFOIL as an independent verification oracle rather than only as a label source, and is designed to accept selective CFD/wind-tunnel augmentation. The trust calibration and verification apparatus are label-source-agnostic.

> *"XFoil is known to be less reliable in separated-flow regimes… discuss how this limitation may affect predicted stall characteristics."*

Section 4.5 quantifies this: only 18 of 41 adversarial candidates yielded a fully converged XFOIL sweep, and the worst surrogate divergences arise precisely where the solver itself fails to converge. Section 5 discusses that surrogate-unreliable and solver-unreliable regions coincide near stall, and that the calibrated trust domain is the mechanism that keeps optimisation inside the region where both are dependable.

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

The revised framework answers this quantitatively rather than qualitatively: high-angle discrepancy is now shown to be a trust-domain phenomenon (Section 4.1, Table 1; Section 4.5). Within the calibrated trust domain the lift R² is 0.994; the degradation at high incidence is confined to the untrusted region and coincides with XFOIL's own convergence difficulties.

> *"Use larger fonts in all the figures."*

Done (14-point base, 15-point titles).

## Reviewer #7

> *"The paper is well written."*

We thank the reviewer.

## Reviewer #8

> *"insufficient novelty… What else is gained apart from computational time?"*

Beyond speed, the work now contributes an operational surrogate-trust calibration, a physically admissible manufacturing-uncertainty model, and a demonstrated deterministic-versus-robust design capability with solver verification. The gain is *governed* reliability: the ability to state where a fast surrogate may be trusted and to design robustly against manufacturing variation—capabilities a timing improvement alone cannot provide. The framework is also delivered as a usable command-line tool that returns a trust-annotated polar for any CST aerofoil (Section 4.6).

> *"the developed model… is doubtful… which model?"*

We now justify the model choice quantitatively rather than assume it (Section 4.1, Table 2): under the identical grouped protocol the multilayer perceptron is compared against histogram gradient boosting, ridge regression and a Dummy baseline. We deliberately avoid over-claiming—gradient boosting is marginally stronger on the drag subset—and instead motivate the perceptron by its competitive accuracy, smooth differentiable multi-output map, and the ensemble-disagreement signal that powers the trust calibration. Consistent with the co-author guidance, we did not undertake a further large architecture search.

> *"The ranges of applicability/operating conditions (Re and M) should be clearly stated… limitations should be clearly listed."*

Operating conditions are stated in the Abstract, Section 2 and throughout (Re = 1×10⁶, M = 0, N_crit = 9). Section 5 lists three explicit limitations: the prescribed (not measured) manufacturing model, panel-method labels, and the moderate seed reproducibility of the robust Pareto front.

> *"What happens with the drag and pitch moment curves, why weren't they also considered?"*

Both are now first-class predicted quantities (Table 1: C_D R² = 0.922 in log-space, C_M R² = 0.918) and enter the optimisation as objectives and constraints.

> *"Physics-aware validation score is never explained… validity of the starting dataset is doubtful… which three neural architectures…"*

The previous ad-hoc "Physics Score" and multi-architecture search have been removed in line with the guidance to focus on application rather than architecture. Model selection is now a transparent, grouped, leakage-controlled protocol with a calibrated trust criterion. Dataset construction uses grouped cluster splits with explicit provenance and no padded/extrapolated targets.

> *"Is the developed model truly satisfactory? … Is this truly scientifically significant?"*

The revised contribution is not a lift-curve fit but a robust-design framework with a measured reliability envelope, verified against an independent solver and stress-tested adversarially. Its significance lies in making surrogate-driven aerodynamic design *defensible*: it states, and empirically validates, the boundary of surrogate trust.

> *"Newer, more relevant references should be cited; discussion should be much improved; passive form should be employed more."*

The literature framing (Section 2) and Discussion (Section 5) are expanded, and the register has been shifted toward an impersonal, methods-first style.

---

We believe the revised manuscript resolves the substantive concerns—demonstrated usefulness, drag/moment coverage, honest treatment of solver limitations, and clearly stated scope—while introducing a genuinely novel and reproducible surrogate-trust and robust-design contribution.
