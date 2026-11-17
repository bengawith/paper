# Uncertainty, objectives and optimisation audit

## Physical operator and representation

The active operator uses Gaussian-correlated controls, piecewise-linear spatial interpolation, per-realisation/per-surface maximum normalisation and leading/trailing-edge tapering. It is bounded and continuous, not a cubic B-spline manufacturing process. The separate spline module is not evidence that the active historical campaign used splines. Amplitudes are prescribed chord fractions, not empirical tolerance distributions. Normalisation also changes the interpretation of latent amplitude/correlation parameters.

The original full campaign transported reference-fitted CST increments to candidate shapes. The current code instead applies shared latent draws to each candidate's normals, validates displaced coordinates, resamples, refits and validates the reconstructed fit. Invalid rows retain nominal parameters only as a matrix-shape placeholder and are explicitly penalised; they are not observed nominal outcomes. Tests reproduce and close crossing-geometry, mismatched-x and invalid-refit defects.

## Objective and feasibility

Required CL values are 0.4, 0.7 and 1.0, with weights 0.25/0.50/0.25. All must have an admissible unique root on the initial rising polar branch. Moment, thickness/geometry, angle and trust checks are separate from the drag value. Mean and empirical CVaR95 apply to a penalised objective; do not label them pure aerodynamic drag when trust or refit penalties occur. The implemented empirical upper-tail mass is fractional when n×0.05 is not integral; at n=13 it reduces to the sample maximum.

## What changed in the selected-design diagnostic

The final-source rerun on five historical designs and 64 latent draws gives:

| Fixed design | Mean penalty-inclusive objective | CVaR95 objective | Trust exits | Invalid refits |
|---|---:|---:|---:|---:|
| Reference | 0.00942056 | 0.01918442 | 8 | 0 |
| Historical minimum-tail design | 0.00893805 | 0.01548513 | 14 | 0 |
| Historical selected knee | 0.00958516 | 0.01515743 | 15 | 7 |

The deterministic and simple sensitivity-baseline designs are infeasible. The knee's mean is worse than the reference's under the current audit, superseding the earlier pre-refit-screen mean of about 0.009040. Its lower tail objective does not conceal this result. Evidence: `COMPUTE_PROFILE/CURRENT_SELECTED_DESIGNS.json`. These values use the historical trained model and are not a repaired-model production result.

## Fronts, seeds and acceptance

The original 128-population, 200-generation, 64-scenario, ten-seed front is invalid for candidate-specific surface uncertainty and has a reported distance 0.283 above 0.20. Ten later schema-4 smoke files reproduce 44 feasible returned entries and distance 0.1492924385 for 32×10×16. Their objective scales are pooled observed ranges, chosen after the outcomes. The active caller now records a descriptive threshold comparison but refuses scientific acceptance under such post-hoc normalisation. No threshold is relaxed and no new favourable scale is selected.

Current caches bind actual checkpoint/scaler hashes, complete scientific source hashes, dependency/Python/device identity, geometry, latent draws, calibration, objective configuration and seed. A mismatch raises a request for a fresh output revision, preserving historical evidence. Historical feasibility/front plots do not become current merely because code is fixed.

## Production disposition

Do not run or approve the full study using a knowingly compromised geometric model lineage. First freeze the geometry-admission and split policy, rebuild and retrain, then freeze objective scales, evaluation draws, candidate-selection protocol and acceptance conditions before the production results are observed. The full 128×200×64×10 experiment is still open. The fresh approximately 3.6-second fixed-design profile is evidence of one workload, not a completion estimate or production-speed claim.
