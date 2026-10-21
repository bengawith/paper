# Historical reviewer-response working draft — not for submission unchanged

This document concerns the earlier EAAI-26-2180 reviews, not a current review invitation. The proposed revised manuscript is "Empirical Reliability Screening for Machine-Learning Aerofoil Prediction, with Selected XFOIL Comparisons". Coauthors must approve the scope and verify the original venue's restrictions before any submission.

## Usefulness and adversarial design (earlier Reviewer 2)

A finite search and three selected fixed-geometry local-XFOIL comparisons are supplied. The selected candidate has18.7% lower nominal required-lift weighted drag; two0.5%-chord scrambles have92/96 completed matched pairs with15.4%/15.8% lower means. These are conditional, same-solver examples, not a universal trust bound or unconditional manufacturing expectation. The overall discrepancy and the selected converged plotting subset are distinguished. Failed solves and the adverse1%-chord marginal maximum are retained.

## Model scope, data, geometry and solver fidelity (earlier Reviewer 3)

The point-conditioned representation eliminates padded supervised targets and includes CL, log-CD and CM at one explicit operating condition. However, the new audit found a coordinate-order defect affecting1,001 stored feature vectors. Source repair is not a retrained model: corpus acceptance, grouping, training/calibration and fresh confirmation must be completed before the original physical-map accuracy claim can stand. XFOIL-on-XFOIL checks do not answer the need for independent CFD or experiment.

## Figures and high-incidence interpretation (earlier Reviewer 6)

Figure captions, panels and labels are revised, including an obscured panel letter, literal escaped newline and overlapping axes. High-incidence failures are not asserted to occur only outside trust. One accepted adversarial probe cannot establish a domain-wide maximum. Final Word layout is checked separately from numerical reproduction.

## Positive writing feedback (earlier Reviewer 7)

The methods-first style of the earlier paper is retained while redundant assurances and unsupported physical claims are removed. The current manuscript remains one standalone paper, with V1 and the referee report kept separately.

## Novelty, model choice and engineering contribution (earlier Reviewer 8)

The historical grouped-development MLP/gradient-boosting/ridge/mean comparison is reported without architecture-superiority claims. Existing robust optimisation and confidence-aware aerodynamic tools, including NeuralFoil, are discussed explicitly. The original deployed screen is support dominated; a corrected calibration-only geometry-cluster audit shows incremental spread ranking information, not a newly validated operational trust gate. The architecture, uncertainty generator and CVaR/penalty meanings are corrected.

## Unresolved substantive response

The scientific response cannot be described as complete before a repaired geometric model lineage and prospective evaluation exist. Full candidate-specific production optimisation with fixed acceptance normalisation and the agreed CFD study also remain open. This working draft does not claim that all reviewer concerns have been resolved by software tests. See the current referee report, claim matrix and open author decisions for exact evidence and remaining scope.
