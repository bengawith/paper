# XFOIL integrity and CFD readiness

## Independent raw-artifact recomputation

`audit_raw_xfoil_and_fronts.py` reads 825 stored cases, including nominal cases, across the 1%-chord 16-draw campaign and two 0.5%-chord 128-draw campaigns. Every coordinate array matches the reconstruction from its fixed candidate and latent draw within 1e-11. All 825 pass the current direct-coordinate validity checks. Solver usability is separate: clean exit, no timeout, actual twelve-angle coverage, finite positive drag, operating commands and required-lift roots are checked from the raw files. Usable totals including nominal cases are 43, 336 and 344; failures are retained, not imputed.

Nominal weighted drag reproduces as reference 0.0074464020141, selected candidate 0.0060511314705, outside-screen comparator 0.0139600173795. This is −18.7% and +87.5% versus reference for the two selected comparators. Selection is deliberate; it is not a population screen-quality estimate.

At 1% chord, each reference and selected design has 13 usable perturbations but only ten matched pairs. Marginal means differ by about 8.8%; paired means differ by 12.2%. These are different subsets/estimands. The candidate's marginal maximum is worse than reference, about 0.00989 versus 0.00909. An apparent favourable matched-subset maximum must not replace that adverse observation. Small-sample CVaR is just an extreme-sample diagnostic here.

At 0.5% chord, campaign A has reference/candidate usable counts 105/110 and 92 matched pairs; B has 115/107 and 96 pairs. Mean paired drag reductions reproduce as 15.3996% and 15.7803%. Matched empirical CVaR changes are −12.4043% and −11.6701%. All matched pairs are geometrically admissible. Because 36 and 32 pairs fail joint usability, these are conditional completed-case distributions, not unconditional manufacturing-risk estimates.

Evidence: `RAW_XFOIL_RECOMPUTATION/all_825_cases.csv`, `XFOIL_RECALCULATED.json` and all raw-source hashes. This audit recalculates existing polars; it does not claim 825 new solver executions.

## Supplied CFD cases

All 28 files in the seven main geometry/polar/status/command bundles match the main manifest. All sixteen files in the four optional 0.5%-chord bundles also match. The seven main geometries comprise three nominal designs and two selected reference/candidate perturbation pairs. The optional cases are a completed median-like scenario and the largest completed candidate-drag scenario in one existing sample, not estimated population quantiles.

The fixed-geometry files remain useful for exploratory cross-fidelity checks. Their historical trust labels and optimiser names must not be interpreted as repaired-model acceptance. The current geometry-corpus finding strengthens the reason to treat them as fixed examples rather than robust optima.

## CFD commissioning and acceptance checklist

Record executable/version, mesh-generator/config hashes, Re/Mach/reference dimensions, force/moment conventions, turbulence and transition models, boundary conditions, numerical schemes and complete residual/force histories. Match achieved CL 0.4/0.7/1.0, not merely the XFOIL angle guesses. Ncrit 9 is not a CFD turbulence input.

Inspect geometry ordering and trailing-edge gap before meshing. Establish three systematically refined meshes and domain sensitivity on the reference and candidate at representative lift. Wall-resolved y+ around one and farfield clearances suggested in the supplied protocol are initial targets requiring verification, not universal sufficient conditions. Test force stability and conservation, not residuals alone. Document equal geometry/moment constraints; retain ranking reversals and failed/divergent cases. A selected RANS result is a cross-fidelity numerical check, not experimental truth or population CVaR.

**CFD outcomes are absent.** Solver/model choice, convergence budgets and scope require Ava's technical approval. No cloud, CFD or external laboratory job was launched in this review.

Primary method/protocol references: https://web.mit.edu/drela/Public/web/xfoil/ ; https://su2code.github.io/tutorials/Inc_Turbulent_NACA0012/ ; https://doi.org/10.1073/pnas.2204569119 .
