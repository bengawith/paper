# Continuation ledger — full scientific closure, 20 October 2026

Directive Google Doc18JVmcpDJsspJ_1vzPCNhtUDXFG9KPq11AqcES7BoEAM checked live: same revision as the full text previously read. Branch robust-v2-viability; HEAD and fetched remote156f07f77e92f77df34a6cc6f6a1726cc0e1638a. Dirty tree consists of the existing review/rebuild and is preserved. No production scientific job was active at resumption. Initial available RAM about4.5GiB and disk118.5GB; retain3GiB/20GiB resource floors and bounded numerical threads.

## Completed inherited work

The corrected-order/nonnegative-TE admitted corpus contains1,256 aerofoils and141,383 archived states. All five rebuilt ensemble members genuinely completed training (37/35/33/65/31 epochs), OOB-development stopping, calibration, freeze and protected retrospective evaluation. Their output artifacts remain immutable. The production integration preflight is complete; the full ten-seed production campaign has not yet run. Nine offline-copy safety cases pass. The manuscript/package remains an unfinished earlier draft and must not be called final.

## Ruling: close the coordinate-to-label frame mismatch before production

Evidence: FRAME_CHECK/FRAME_CHECK.json plus raw commands/polars. The archived ag35 polar agrees closely with fresh raw-coordinate XFOIL at the recorded alpha (CL RMSE0.001274 over6 points) but not the rotated/resampled geometry (RMSE0.172927 over6 points). A18 likewise differs after normalisation, with incomplete fresh sweeps retained. Even NACA2412 resampling changes the numerical solution. This independently confirms that a geometry parser repair alone does not certify the archived geometry/alpha/force-reference pairing. Do not infer undocumented generator normalisation or analytically relabel coefficients at a different effective Reynolds number.

Action: create a separate controlled-coordinate lineage robust-v2-controlled-20261020. Use every one of the1,256 already admitted CST vectors as a defined numerical geometry, not as an assertion of identity to its source foil. Generate fresh labels on its exact unit-chord CST reconstruction, Re1e6/Mach0/Ncrit9, fixed200 iterations, two directed sweeps0..16 and0..-12 degrees at0.25-degree spacing. All113 distinct requested angles are accounted for. Use at most two solver workers; each segment has a30-second limit. No scraping or external compute is needed.

Admit only actual finite positive-drag CL/CD/CM rows from clean zero-exit, non-timeout solver processes, within the requested grid. A partially converged polar supplies observed rows, never filled or extrapolated labels, and is not declared a fully verified case. Consistent duplicate alpha0 is counted once; conflicting duplicates are excluded and recorded. Every absent/failed observation and source/command/executable hash is retained. Geometry admission is not changed by predictive or aerodynamic outcomes.

Recluster the actual reconstructed geometry at the same1e-4 RMS threshold and apply the most-protected historical partition rule before looking at any new aerodynamic labels. This prevents repaired aliases crossing splits. Train the same fixed five-member512-wide/two-residual-block baseline with the same OOB-development stopping protocol; no architecture or test-driven search. The historical shapes have been inspected, so the new cohort is not described as pristine prospective sampling. This last data repair costs a full controlled solver-label campaign and another training run, but closes the physical input/output contract rather than qualifying an avoidable defect.

## Remaining execution contracts

1. Controlled corpus: all1,256 requested geometries and113 angles accounted for, exact geometry-source identity, no imputed rows, immutable raw receipts and hashes, protected physical clusters.
2. Actual five-member training, calibration, freeze, grouped micro/macro/coverage evaluation, modest fixed baselines and calibration-only risk OOF on the controlled lineage.
3. Unreduced128×200×64×10 production, with exact source/model context and generation checkpoints, original0.20/80% criterion reported separately from prospective reference scaling. No source change while a production campaign runs. All seeds count.
4. Deterministic finalist selection before external solver outcomes,2,048 fresh surrogate scenarios and strict fresh XFOIL selected-design validation with all failures/adverse results. Independent viscous CFD remains a distinct stage; do not manufacture it.
5. Complete literature/claim review, current paper/figures/standalone Word and visual QA, all command/manifest gates, coherent commits and normal push, one post-push hash-verified OneDrive package. Original raw evidence and previous model generations remain segregated historical evidence.

Status: task1 active. No task is marked complete from process launch alone.
