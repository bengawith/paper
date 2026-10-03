# Review status — 3 October 2026 (revision 2)

**This revision resolves the material errors flagged in revision 1. The manuscript now matches the recorded evidence, and the central claim is settled by direct XFOIL verification rather than by surrogate-side values.** Read this note alongside the manuscript.

## How each prior-flagged item was resolved

1. **XFOIL drag sign (was inverted).** Resolved. A fresh, correctly-signed verification (`results/.../design_verification/`, `scripts/robust_v2/verify_designs_xfoil.py`) compares the NACA 2412 reference against a within-trust design and an outside-trust design. The sign convention is stated explicitly (positive = lower drag than reference). The outside-trust surrogate "optimum" is now correctly reported as **worse** than the reference (nominal +88%), and the within-trust design as **better** (nominal −18.7%, expected −11.7%, paired +0.00098, 95% CI [0.00062, 0.00131]).

2. **Rank correlation rested on two designs.** Resolved by replacing the two-design nominal ranking with the three-design, matched-perturbation verification above (15–16 converged shared samples per design, paired bootstrap intervals). The weak two-point Spearman claim has been removed.

3. **Adversarial trust comparison had one fully trusted design.** Addressed by resting the in-trust reliability claim on the 26,692-point held-out locked test (97–99% within-trust coverage; 2–3× lower error), and using the adversarial search only for its genuine finding (worst-case disagreement outside trust). Candidate accounting is stated as 49 requested / 41 evaluated / 20 rejected / 18 fully converged.

4. **Calibration coverage was not independent.** Resolved. The manuscript now reports **held-out** coverage of the calibrated band on the locked-test partition (all states 93.7/94.4/94.5%; within trust 97.8/96.5/99.2% for C_L/C_D/C_M), not the calibration-set 95%.

5. **Manufacturing statistics included penalties.** Resolved. The table/figure now label the quantity as the **robust objective** (penalised on trust exit) and report the **trust-exit fraction** separately. The text states that up to 0.5% chord the exit fraction is ≤0.4% (pure drag) and that at 1% chord 17% of samples exit, so the tail is penalty-influenced — framed as the surrogate's domain limit.

6. **Architecture description was wrong.** Resolved. Methods now describe the **frozen, Optuna-selected** ensemble: ReLU, 2 hidden layers, width 512, dropout 0.12, log-cosh, AdamW (from `FROZEN_MODEL_MANIFEST.json`). The ReLU citation replaces the SiLU citation.

7. **Pareto terminology/stat claims.** Resolved. Endpoints are called "minimum expected" and "minimum CVaR" (not deterministic/knee), CVaR is described as a tail average, the aggregated front is described as such, and the seed-agreement failure is reported openly with the engineering claim moved onto the XFOIL verification.

8. **Readiness / coverage mis-counting.** Addressed. `run_xfoil` now credits coverage only to finite, positive-drag rows under a clean solver exit (new `_valid_aero_points` helper + regression test `test_valid_aero_points_excludes_nonconverged_and_nonpositive_drag`). A root-cause code defect was also fixed: the optimiser previously gated trust on the permissive 0.99 support quantile (≈4.37) instead of the surrogate's calibrated threshold (≈1.247); both now use one shared `trust_thresholds` helper, locked by `test_trust_thresholds.py`. No CFD or wind-tunnel validation was performed; this remains stated as a limitation.

9. **Timing was throughput only.** Resolved. The manuscript reports both the batched throughput (0.009 ms/state) and the single-state (~19 ms) and 48-state polar (22.5 ms) latencies, and states which is relevant for design.

10. **References / journal policy / disclosures.** The five new-method references were verified against primary sources; reused references are from the prior accepted-for-review manuscript. A generative-AI-use declaration has been added. The authors must still confirm the target journal and honour the prior venue's one-year resubmission restriction before any submission; selecting an appropriate alternative journal is an author decision outside this package.

## Standing limitations (unchanged, stated in the paper)

Single operating condition (Re = 1×10⁶, M = 0, N_crit = 9); prescribed (not measured) manufacturing model; panel-method labels with no CFD/experimental verification; robust-front location only moderately seed-reproducible (engineering claim rests on XFOIL verification). Passing unit tests does not establish scientific validity; the authors should read the evidence artefacts and confirm the claims before submission.

## Evidence

All manuscript numbers derive from lineage `robust-v2-current-20260908` and are consolidated in `paper/data/paper_data.json`; every figure regenerates from it via `scripts/robust_v2/make_paper_figures.py`. The new design verification is in `results/robust_v2/lineages/robust-v2-current-20260908/design_verification/`.
