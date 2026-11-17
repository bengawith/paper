# ML methods, data integrity and leakage audit

## Scope and evidence

Inputs: immutable `robust-v2-current-20260908` model, split, tuning, calibration and prediction records; `data/robust_v2/processed/full_exact`; active preprocessing, training, ensemble and evaluation source. Reproduction: `recompute_ml.py`, `audit_geometry_corpus.py`; outputs `ML_RECOMPUTATION/ML_AUDIT.json`, metrics CSV and `GEOMETRY_REPAIR/GEOMETRY_AUDIT.json` plus per-aerofoil records. All relevant source hashes are linked by the claim/evidence matrix and evidence manifest.

## Verified executable properties

The stored 179,780-state table uses numerical source rows rather than padded polar targets. Lift has one unavailable label; drag and moment have all rows. The grouped split contains development 1,093 aerofoils/1,088 clusters; calibration 234/233; historical test 233/233. Recorded cluster identifiers do not overlap. The audited scaling and tuning inputs do not include non-development identities. Recorded cluster-bootstrap membership and member scalers reproduce from their saved inputs. Re-inference matches the saved test predictions exactly in the available CUDA environment.

The frozen architecture is an input projection of 13 to 512 features, two residual blocks containing two affine layers each, and three scalar heads: five hidden affine transformations and 1,064,451 parameters per member. ReLU makes it piecewise differentiable, not globally smooth. Five members, grouped bootstrap, dropout 0.12, layer normalisation, masked log-cosh loss and AdamW are reflected in the archived configuration and source. A raw member spread is an empirical diagnostic, not independently calibrated epistemic uncertainty.

## Critical integrity defect

The coordinate-order audit changes 1,001 of 1,560 stored feature vectors. It compares 1,557 parseable records, rejecting three ambiguous files. Historical versus repaired maximum-fit errors above 0.005 chord are 931 versus 289 within those comparable records; the all-1,560 historical count is 933, a different denominator. Repaired reconstructed fits also expose 656 crossing/admission failures. The active parser now treats Lednicer blocks and Selig contours separately, and training-data admission validates the reconstructed fit.

This means an apparent absence of identifier overlap does not prove physical near-duplicate separation. Clusters and support must be recomputed after the accepted geometry policy is frozen. Corrected feature vectors cannot be evaluated using the old weights and relabelled as new test evidence. A rebuild must preserve a newly declared untouched confirmation set; the historical test has already been inspected repeatedly and is not pristine for future modelling choices.

## Metric reconciliation

Historical all-state R²: CL 0.98869049; log-CD 0.92184771; CM 0.91801080. MAE: 0.05713828, 0.14340075, 0.00959476. Aerofoil-macro MAE: 0.05602204, 0.14839180, 0.00954105. Linear-drag MAE is 0.00454658 and R² 0.85282766; these must not be confused with the log-drag values. Accepted/rejected metrics, masks and polar-wise coverage are retained separately. Dataset-family claims are not upgraded to leave-one-family-out: no authoritative family taxonomy was established.

The historical gradient-boosting/ridge/mean comparison is retained as an archived grouped-development comparison, not a new training run. A further architecture zoo is not justified by this audit. Repeat the same modest baselines after geometry repair; comparisons must use equal feature acceptance, split, masks and cost accounting.

## Runtime and integrity

The archived predictor-only GTX 1660 measurements distinguish about 19 ms/single state, 22.5 ms/48-state polar and 55.5 ms/6,144-state batch. CUDA synchronization brackets timing; batch amortisation is not per-query latency or full-workflow speedup. The fresh five-candidate, 64-scenario evaluation took about 3.6 s and is explicitly a different workload. Checkpoint and scaling hashes are now checked before deserialisation, preventing altered weights from silently satisfying a manifest-only check.

## Acceptance boundary

Code repair and historical-number reproduction are complete review outcomes. Rebuilt training, reclustering, retraining, fresh confirmation and physical-geometry generalisation are not complete. No historical split or coefficient is silently replaced.
