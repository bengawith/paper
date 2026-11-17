# Independent CFD brief for Ava — Robust V2 frozen finalists

Use this brief with CFD_HANDOFF_CURRENT.zip and the current manuscript. This is the frozen **20 October 2026** selection, not the earlier 14 October package. No independent CFD results are represented as complete.

## Cases and identity

The three **required** nominal sections in inputs/nominal are:

1. reference.dat — SHA-256 9493f06e84a8c975718ba14a6decf76041f3f7988157614237d3a394a4c9deb2
2. risk_knee.dat — SHA-256 870836bca0f9f844e8f0d439b893aa53163ada5262d608d0c0c464b9029bde83
3. nominal_control.dat — SHA-256 80ba1ff59d761042493f4659399509df00b63c93c8d5330a0ebf6356176ffe72

CURRENT_CASE_MANIFEST.json contains all nine geometry hashes, source paths and scenario identities. FROZEN_SELECTED_DESIGNS.json and XFOIL_CASE_SELECTION_PROTOCOL.json preserve the original decision and solver protocol. Do not change the nominal designs following CFD results or substitute a historical risk-knee/nominal comparator.

## Conditions and comparisons

- Independent viscous CFD at chord Reynolds number **1,000,000** and consistent low-Mach assumptions. Supply the actual dimensional chord, fluid properties, reference velocity and Mach number; the original XFOIL model used M = 0 and Ncrit = 9. Ncrit is **not** a RANS turbulence/transition setting.
- Define the coordinate frame, angle convention, force directions, reference chord/area and pitching-moment sign/reference point (**quarter chord**) before calculation. Confirm the trailing-edge gap, surface orientation and dimensional scaling from each exact DAT.
- The engineering mission is **matched achieved lift** at C_L = **0.4, 0.7 and 1.0** for **each** section, with target weights **0.25, 0.50 and 0.25**. Locate each lift target on its initial rising branch at an incidence within **0°–8°**, using a numerical lift tolerance selected and recorded **before** observing the comparisons. Report requested and achieved C_L, incidence, C_D and C_M for each target, plus failure or multiple-crossing cases. Do not extrapolate a missing root.
- Compute each section's nominal weighted drag J only if all three achieved-lift cases are admissible. Give absolute coefficients, matched-lift differences and percentages with the reference denominator stated. Check against the nominal-only control as well as the robust risk-knee; a lower numerical surrogate loss does not predetermine the CFD ranking.
- **Important moment scope:** the completed optimisation constrained the arithmetic mean C_M at **33 incidences, 0°–8° in 0.25° steps**, not a pointwise 0/2/4/6/8° requirement. Frozen reference mean C_M = −0.051967 and lower bound = −0.061967; the risk-knee mean is −0.060804 and nominal-control mean −0.061965 under the numerical model. The nominal control violates the original *alternative* five-incidence pointwise bound at 6° (by 0.004877) and 8° (by 0.002805). These are XFOIL/surrogate study facts, **not CFD results**. Report CFD moment versus incidence independently, assess the mean rule on a declared consistent grid, and disclose any pointwise behaviour; do not call the nominal control pointwise feasible.

## Independent numerical quality

1. Document solver and version, mesh generator, exact geometry and mesh/deck hashes, governing equations, near-wall treatment and y⁺, turbulence and transition models, discretisation, pressure far field, outlet/inlet boundaries, chord and domain size. If transitional results depend materially on the model, state this.
2. For **all three** primary geometries, give a **minimum three-level mesh study** using consistent model/geometry settings. Report grid cells, wall resolution and integral C_L/C_D/C_M changes at representative mission states; assess domain sensitivity and transition/turbulence-model sensitivity on the same three-design comparison. Do not substitute a converged residual alone for force/moment and mesh convergence.
3. Record lift/drag/moment histories, residuals, continuation or incidence-search method, steady/unsteady behaviour, stopping criteria, achieved convergence tolerances, solver nonconvergence, ambiguous lift roots and every excluded case. If any mission target cannot be established reliably, report the nominal metric as unavailable rather than inventing an estimate.
4. Compare with a suitable published or established reference benchmark if one is available for the chosen geometry/flow model. Explain differing model assumptions rather than tuning to the desired design ranking.

## Optional sensitivity cases and deliverables

Only after the nominal comparison, the six **optional** DATs in inputs/optional_0p5pct offer two fixed 0.5%-chord perturbation indices (A/000 and B/007) for each design. The indices were selected by recipe, not favourable aerodynamic results. Six cases cannot estimate a physical manufacturing distribution, CVaR95 or a robust superiority probability.

Return the completed CFD_RESULTS_TEMPLATE.csv, solver input decks, meshing scripts/files and hashes, convergence and residual/force histories, grid/domain/model studies for all three designs, pressure/flow figures where useful, and an **independent CFD_VALIDATION_REPORT.md** explaining numerical quality, absolute and matched-lift comparisons, moment assessment, failures and limitations. Keep all initially requested cases in the ledger, including failures and ranking reversals. The original finalist selection must remain frozen.

The manuscript will acquire independent physical conclusions **only when this evidence has been checked**.