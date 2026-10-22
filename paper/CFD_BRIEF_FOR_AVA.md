# Ava - final frozen-design CFD validation (22 October 2026)

IMPORTANT: Use this handoff, not the older 14 October CFD handoff. That earlier package compared an outside-screen deterministic candidate and an earlier risk-knee, whereas this paper compares the frozen final reference, robust risk-knee and matched nominal-control designs.

## Primary CFD cases
Run all THREE geometries from inputs/nominal: reference, risk_knee, nominal_control. Their SHA-256 and original case identities are in CURRENT_CASE_MANIFEST.json, derived directly from the frozen 20 October selected-study provenance.

## Physical model and conditions
Use independent viscous CFD, Re_c=1,000,000 and consistent low-Mach conditions; record actual solver version, case decks, mesh hashes, domain, wall y+, transition/turbulence assumptions, boundary conditions, chord and pitching-moment convention. XFOIL's Ncrit=9 is not an interchangeable RANS turbulence setting.

## Minimum accepted engineering comparison
1. Run validated source geometry and correct orientation, TE gap, surface continuity, and quarter-chord moment.
2. Verify integral-force and moment convergence; use at least three mesh densities and domain/transition-model sensitivity on reference and risk-knee. Record the CFD scheme, residuals, wall time and all failures.
3. For EACH of the three fixed geometries, locate incidences at achieved C_L=0.4, 0.7 and 1.0 within documented tolerance. Report achieved lift, C_D, C_M, convergence and mesh sensitivity. Weighted drag uses 0.25,0.50,0.25 and is only estimable if all three matched-lift cases converge.
4. Only after the nominal cases, consider six optional geometric sensitivity cases from inputs/optional_0p5pct. Two fixed scenario indices (A/000 and B/007) are present for each design. These were chosen deterministically by index, not solver success or promising drag outcomes. They cannot estimate a manufacturing tail probability or physical CVaR95.
5. Return a CFD_RESULTS_TEMPLATE.csv filled with ACTUAL observations, solver input/mesh files, residual and force histories, mesh studies, pressure/flow figures where appropriate, complete failure ledger, and an evidence-based CFD_VALIDATION_REPORT.md.

The final manuscript already contains the full numerical research and its supported conclusions. After the CFD evidence is checked, add the results and an honest physical-design conclusion, including any reversed rankings or inconclusive comparisons. No CFD output is invented in this handoff.

Original frozen experimental selection: FROZEN_SELECTED_DESIGNS.json. Exact accepted numerical-case identities: XFOIL_CASE_SELECTION_PROTOCOL.json. Neither is altered in this package.
