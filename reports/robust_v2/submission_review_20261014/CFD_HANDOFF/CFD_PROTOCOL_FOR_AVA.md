# CFD verification handoff for Ava — 14 October 2026

**Status:** pre-CFD handoff, with no CFD or wind-tunnel outcomes claimed. This protocol is for coauthor review and execution, not evidence that RANS has already verified the design.

## Objective and supplied cases

The current ML ensemble predicts CL, CD and CM from a 12-parameter CST geometry and angle of attack. Its training data are XFOIL-derived; a fresh local XFOIL check tests three selected designs. A higher-fidelity CFD stage is needed to test whether their ranking survives a different physical/numerical model.

The sibling **CFD_CASE_MANIFEST.json** and **inputs/** folder contain seven provenance-checked geometries (geometry.dat), polar CSVs, strict solver-run statuses and XFOIL command files. Every file has a SHA-256 digest.

Nominal designs:
1. NACA 2412 reference (inside screen)
2. Within-screen selected risk-aware candidate
3. Outside-screen unrestricted surrogate optimum (counterexample)

Perturbed cases: a representative paired successful scenario and the largest candidate drag among the strictly completed, target-lift-feasible paired scenarios, each supplied for both reference and within-screen candidate. Neither represents a statistically estimated manufacturing tail percentile.

## Physical conditions

- Chord Reynolds number Re_c = 1,000,000; low-Mach/incompressible, nominal Mach 0.
- Reference XFOIL transition setting Ncrit 9; **not a directly transferable CFD turbulence setting**.
- Prescribed required lift CL = 0.4, 0.7, 1.0 with weights 0.25, 0.50, 0.25.
- Consistent chord normalisation and reference quarter-chord pitching-moment convention.
- Geometric perturbation bound 1% of chord, smooth and correlated, **not derived from measured manufacturing tolerances**.

The manifest includes XFOIL angles for each required lift as **initial CFD search guesses**, not CFD predictions or accepted CFD results. The comparison must be performed at the same achieved lift, not necessarily the same angle of attack.

## Solver and mesh plan

A validated viscous finite-volume RANS solver is required (SU2, OpenFOAM, or institutional equivalent). Document exactly which solver/version, equations, turbulence model, transition treatment, inlet turbulence level, boundary conditions, reference length/speed/viscosity, mesh generation, far-field location, wall resolution, numerical schemes and force/moment integrations were used.

As an initial numerical setup, consider ≥20 chord lengths of clearance around the aerofoil and ≥30 chords downstream, wall-resolved near-wall y+ around 1 for SST k-omega where appropriate; these are **initial targets to test**, not universal sufficient conditions.

XFOIL's e^N transition model differs from fully turbulent RANS. Use a validated transition model if available, or explicitly quantify sensitivity to turbulence/transition choices; do not attribute all CFD–XFOIL differences to the ML.

## Required execution

**Phase A — commission and converge:** Inspect each supplied geometry for orientation, chord, TE gap and self-intersection; regenerate only derived mesh files. Run NACA 2412 at an ordinary alpha first, establish integrated-force convergence and mass conservation, and keep residual and force histories.

**Phase B — grid/domain study:** Run at least three meshes for NACA 2412 and the within-screen candidate at one representative service CL, with consistent solver setup. Report cells, wall y+, residual/force stability, CL/CD/CM, wall time and the variation of drag with mesh and far-field distance. A solver run with a residual plot only is not accepted.

**Phase C — equal-lift nominal comparison:** For reference and selected within-screen candidate, iteratively bracket and converge alpha at each requested CL 0.4, 0.7 and 1.0; compute weighted drag only when all conditions converge. Check the outside-screen candidate at least at the representative service condition, preferably all three.

**Phase D — selected perturbations:** Test the two supplied common perturbation pairs on the reference and within-screen candidate. At minimum run the dominant service condition, expanding to all target lifts if practical. A few perturbation CFD cases are diagnostic, **not a 95% tail-risk estimate**.

**Phase E — conclusions:** Compare rank, nominal weighted drag, moment, constraint compliance and sensitivity with XFOIL. If ranking reverses, report it and amend the paper. Investigate effect of transition/turbulence assumptions before attributing disagreements to surrogate error.

## Deliverable schema and acceptance

Create a machine-readable case table with design, scenario, geometry hash, mesh hash, CFD executable/config hash, Re, Mach, alpha, target_CL, achieved_CL, CL/CD/CM, y+ diagnostics, cell count, residual status, force stability, iteration and wall-clock costs, failure reason and reviewer notes.

Keep the meshes or mesh-generating scripts, solver decks, logs, residual/force histories, mesh-convergence figure, nominal/perturbed polar results, and a concise CFD_VALIDATION_REPORT.md. Record all missing and failed cases without numerical imputation.

Do not claim independent physical validation until the CFD evidence is complete; do not claim population CVaR from this selected CFD set.

## Author decisions

Ava and Ben should agree on (a) transition-capable versus fully turbulent RANS and associated sensitivity cases, (b) whether actual manufacturing tolerance measurements exist, and (c) target journal and resubmission restrictions.

## Technical references

MIT XFOIL: https://web.mit.edu/drela/Public/web/xfoil/

SU2 incompressible turbulent NACA 0012 tutorial: https://su2code.github.io/tutorials/Inc_Turbulent_NACA0012/

Feedback-shift coverage caution: Fannjiang et al., PNAS 2022, https://doi.org/10.1073/pnas.2204569119

The current evidence is a numerical XFOIL comparison, not completed CFD. This handoff is the final independent validation phase.

## Optional 0.5%-chord sensitivity cases added in October 2026

The folder `OPTIONAL_0P5_PERCENT/` contains **four additional geometry/polar/status/command bundles**: NACA 2412 and the within-screen selected design under a representative completed scenario and under the largest candidate drag among 92 full-sweep and target-lift-feasible pairs from the first independent 128-scenario scramble. Its manifest hashes each source file. These are *supplemental*, not an instruction to double the initial CFD workload.

Two independently scrambled 0.5%-chord XFOIL campaigns (128 requested scenarios each) recorded 92 and 96 complete matched design/reference pairs. Paired mean drag reductions were 15.4% and 15.8%, respectively, but solver failures were excluded and these do not constitute measured manufacturing probabilities or an independently validated population CVaR. The optional CFD cases permit cross-fidelity sensitivity comparison at a smaller deviation than the 1%-chord stress set.

## Important updated optimisation status for coauthor review (14 October 2026)

A later code audit found that the historical robust NSGA-II campaign transferred reference-fitted CST coefficient increments to other candidate shapes. The current source now applies common latent perturbations to each candidate's **own surface-normal coordinates**, rejects invalid geometry and invalidates the historical objective caches. The selected within-screen geometry provided here is therefore **a historically selected, numerically promising candidate**, not a currently proven robust Pareto optimum under the corrected uncertainty model. A corrected low-budget ten-seed smoke check passed (mean front distance 0.149, threshold 0.20), but the full production ten-seed campaign remains outstanding.

**Purpose of Ava's CFD:** verify aerodynamic rank/coefficients of the supplied *fixed geometries* against a higher-fidelity viscous model, not certify that a new optimiser has found a unique global optimum. No physical CVaR95 or measured manufacturing reliability can be inferred from these selected CFD cases alone. Preserve all adverse, failed and divergent outcomes.
