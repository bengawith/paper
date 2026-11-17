# Rebuilt scientific lineage: protocol fixed before new model outcomes

14 October 2026. This extends the in-place academic review under Ben's explicit instruction to perform full model training, results and write-up, not merely recommend a rebuild. Base: 156f07f77e92f77df34a6cc6f6a1726cc0e1638a. Existing dirty work is this review and is preserved. Google directive 18JVmcpDJsspJ_1vzPCNhtUDXFG9KPq11AqcES7BoEAM was read again at the start of this continuation; its revision is unchanged.

## Design and boundaries

Use a new lineage `robust-v2-rebuilt-20261014`; never rewrite `robust-v2-current-20260908`, archived raw coordinates, raw polars, model weights or frozen acceptance files. Existing favourable and adverse findings remain historical evidence. Work stays in the authorised branch and repository. No reset, clean, force push, main merge, message sending, paid provisioning or AirfoilTools scraping.

Rebuild the existing CST/point-conditioned five-member MLP pipeline, not a new architecture zoo. New source is reached by real training/evaluation consumers. Changes to model/geometry/trust/physical objectives invalidate all dependent caches. Original uncertainty amplitude, correlation, service lift and engineering constraints remain unchanged; no favourable threshold is selected after results.

## Geometry admission

Parse supported Selig contours and two-block Lednicer orders explicitly. Reject ambiguous/invalid source shapes with named reasons. Retain twelve CST features. A newly identified subdefect is negative fitted trailing-edge thickness: 652/656 repaired crossing-fit failures have a negative TE coefficient. Constrain this one physical coefficient to nonnegative in the least-squares fit, solving the remaining coefficients again (not clipping an unconstrained fit). Keep subsequent full reconstructed-surface validity checks. This is a convex one-bound least-squares correction, not a new manufacturing law.

Use the existing 0.005-chord maximum CST error cap consistently for corpus and uncertainty refits. Check reconstruction on a denser grid as well as the fit grid. Record all exclusions and both unconstrained historical and repaired outcomes; do not exclude by label or prediction error. Recompute physical geometry clusters using the existing RMS threshold 1e-4. Preserve historical test/calibration membership: when a new physical cluster spans old partitions, assign the whole cluster to the most protected partition (test > calibration > development). Never move any old test/calibration geometry into training. Report changed counts and historical-test inspection honestly; do not call the historical test pristine.

## Training and evaluation

Keep the recorded width512, two residual blocks, ReLU, dropout0.12181152899002505, AdamW learning rate0.0016830217255088852, weight decay1.0205288760315097e-5, batch1024, log-cosh loss, per-aerofoil weighting and five cluster-bootstrap members. Seeds 2026101400 through2026101404. For each member, stop using only out-of-bag DEVELOPMENT geometry clusters, maximum300 epochs, patience25, minimum improvement1e-7. Calibration and test cannot affect stopping or architecture. Train-only robust scalers are stored and verified. Fixed loss and masks are retained; efficient resident batches must match the existing batch-loss/gradient semantics in tests.

Retain per-epoch histories and restartable checkpoints. Fit existing empirical trust construction on repaired calibration predictions only. Recompute grouped risk diagnostics with per-fold train-only scalers, residual thresholds and ridge alpha10. Do not promote an unconfirmed combined risk policy. Freeze weights/scalers/config/data/split/trust hashes before evaluating the repaired historical test. Evaluate micro/macro errors, linear/log drag, coverage and accepted/rejected counts with all denominators explicit. Repeat modest baselines under identical repaired partitions. New selected-geometry solver checks are an external-usefulness check, not untouched natural-population test data.

## Production uncertainty and optimisation

Profile and repair computational bottlenecks only with numerical-equivalence tests: common-x interpolation/CST linear algebra reuse, repeated-geometry support queries, redundant evaluations. Do not reduce the required production profile: NSGA-II population128, generations200,64 common scenarios,ten seeds. Candidate-specific actual normals, upper/lower coupling, leading/trailing-edge behaviour, positive finite drag, required-lift roots, invalid/refit penalties and nominal feasibility remain binding. Cache complete scientific/model/source context and preserve checkpoints.

Freeze objective-distance scales before production from the two positive reference objectives under the new frozen model and the same64 scenarios. They are fixed reference magnitudes, not pooled observed front ranges. Keep the0.20 mean symmetric-distance threshold and at least80% feasible runs. Report the original post-hoc-scale smoke separately; do not claim its number establishes prospective convergence. All ten seeds count, including empty/failed fronts. Distinguish finite algorithmic agreement from global optimality.

Select final within-screen candidates from the pooled feasible non-dominated front using a recorded deterministic rule without XFOIL outcomes. Evaluate finalists under2048 fresh common scenarios and report failure/refit/trust penalties and sensitivity, not pure drag when penalties apply. Perform strict actual-coordinate XFOIL checks at equal required lift, preserving every non-converged case and using independent scrambles for selected perturbation comparisons.

## Editorial and release gates

Update the actual paper and standalone Word only from new completed evidence, retaining historical lessons and negative results without presenting historical figures as new results. Expand the claim/citation ledger as needed, primary-source-check current relevant literature, and complete visual page inspection. Complete Ruff/mypy/all applicable tests, evidence hashes, ZIP checks, normal commits/push and one post-push canonical OneDrive package. The package manifest must carry `source_git_commit` equal to verified remote HEAD. Local OneDrive existence does not establish cloud sync.

Independent viscous CFD is not fabricated. Inspect available authorised solver capability; any actual run must have a justified deck, mesh/domain checks, force histories and explicit turbulence/transition scope. Otherwise the unchanged directive's Ava CFD handoff remains explicitly outstanding, never converted to acceptance by software tests.

## Execution ledger

Task1: active — physical fitting, corpus rebuild and no-leakage split contracts.
Task2: pending — complete five-member training, calibration, freeze and evaluation.
Task3: pending — exact evaluator acceleration, full ten-seed production and final2048-scenario/XFOIL results.
Task4: pending — references, full updated write-up, visuals, final software/manifest/package/push gates.

Ruling: execute the owner-approved in-place rebuild directly rather than ask for another scope confirmation. Routine method choices above are explicit and frozen before new results. Costs: narrower representable-geometry domain; held-out historical data remain retrospective; model settings are a controlled baseline rather than a claimed optimal architecture.

## Production execution clarification — before launching any production seed

All five rebuilt members completed at 37/35/33/65/31 epochs; stopping epochs were chosen only from out-of-bag development clusters. Calibration, model freeze, repaired historical-test evaluation and the fixed-baseline comparisons are complete. The protected test is explicitly retrospective, not relabelled pristine.

The full production campaign will use the unchanged YAML full profile: population128, generations200,64 shared scenarios,ten seeds. Profile index2 gives seeds20263824 through20263833. Latent uncertainty seed remains20262824 (uncertainty seed+2000). Workstation floors are3GiB available RAM and20GiB free disk, checked between saved generations; CPU numerical threads remain2. Exceeding a floor pauses at a preserved checkpoint, never shrinks the experiment.

Two agreement summaries are mandatory: (a) the prospective reference-objective normalisation fixed before production and (b) the original pooled-range numerical diagnostic with exactly the original0.20/80% rule. A favourable reference-scaled value must not rescue a failed original pooled-range comparison. A claim that the full original numerical criterion passed requires its own computed result. Neither agreement summary certifies a global optimum or independent CFD.

The fit, geometry and root accelerators have explicit equivalence tests. A final recheck stores objectives and constraints from the same evaluation and records any difference from optimiser-held objectives. Infeasible fallback populations remain labelled as fallbacks. The simple lift-sensitivity baseline now uses actual candidate-specific scenarios; it refuses failed scenarios rather than treating zero-delta placeholders as measured sensitivity. Conditional aerodynamic diagnostics explicitly omit invalid placeholders and disclose their retained count.
