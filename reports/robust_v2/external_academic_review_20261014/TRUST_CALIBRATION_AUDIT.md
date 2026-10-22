# Trust calibration and selective prediction

## Frozen operational rule

The 26,692-state historical test has 17,642 accepted and 9,050 rejected states. All 9,050 rejected states fail support distance. The stored spread cutoffs add zero independent rejections; six moment-spread failures and the recorded angle failures already fail support. Thus the empirical accepted-set benefit is attributed to a support-dominated rule, not to proven independent operational value of ensemble spread.

The bands are target-wise, residual-quantile constructions. State observations within a polar are correlated. All-state coverage is 0.936685 for lift, 0.943916 for log-drag and 0.944890 for moment; accepted coverage is 0.977497, 0.965310 and 0.991781. Accepted lift coverage rounds to 97.7% at one decimal place. Complete observed-polar coverage is only about 51.1%, 44.2% and 66.5%; the nominal state-wise band is not a simultaneous curve band. Conditional-on-acceptance or optimiser-shift guarantees are not inherited from ordinary marginal conformal arguments.

## Incremental disagreement study

The old audit grouped aerofoil names and allowed one near-duplicate geometry cluster to cross two folds. The canonical script now resolves every calibration row against the frozen geometry-cluster map and refuses missing/non-calibration identities. Corrected GroupKFold(5) contains no shared stored geometry clusters. Residual normalisation, high-error thresholds and input scaling are training-fold-only; ridge alpha remains fixed at 10.

The rerun `CANONICAL_TRUST_INCREMENTAL/TRUST_INCREMENTAL_CV.json` reproduces the independent audit exactly: support AUROC 0.7679624719; spreads 0.8585365516; combined 0.8667371583; difference 0.0987746864. Combined average precision is 0.4848707689 versus support 0.3609338589. A 300-geometry-cluster bootstrap gives [0.0666791433, 0.1402019524] for the difference. Resampling holds the OOF predictions fixed and does not include the whole refitting procedure's variability. The cluster-balanced and per-fold diagnostics are in `TRUST_RECOMPUTATION`.

This is positive residual-ranking information in the historical calibration distribution. It is not a deployed policy, an untouched confirmation result, a model retraining improvement or proof of reliability on adaptive geometries. The geometry-encoding defect also limits the physical meaning of the source clusters. The frozen trust JSON and model weights remain unchanged.

## Decision

Retain the historical accepted-set metrics as descriptive; replace obsolete ID-grouped incremental numbers in the manuscript; require repaired geometric support and a prospective confirmation cohort before changing any operational gate. Do not tune on the historical test or call retrospective selective-coverage curves pre-registered without dated evidence.

Primary theory: Angelopoulos and Bates, https://arxiv.org/abs/2107.07511; Fannjiang et al., https://doi.org/10.1073/pnas.2204569119. These sources motivate the assumptions and cautions, not acceptance of this implementation.
