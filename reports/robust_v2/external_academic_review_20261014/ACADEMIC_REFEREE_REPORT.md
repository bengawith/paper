# Independent computational academic-integrity review — final controlled study

21 October 2026. Editorial recommendation: **MAJOR REVISIONS BEFORE SUBMISSION**.

This is a separate computational/source-grounded self-review and not a fabricated independent human peer referee or coauthor endorsement. This manuscript is an author-review draft, not a journal acceptance.

## Source defects, scope and positive evidence

The earlier coordinate-order defect affected 1,001/1,560 encoded aerofoils. The subsequent exact-geometry numerical corpus contains **106,931 actual local-XFOIL numerical labels** from 1,224 CST geometries, out of 141,928 requested states. Missing or unusable cases total 34,997; only 44 shape sweeps are complete. The omitted states are not estimated, filled or treated as random.

Five fixed-architecture neural members were retrained on genuine local solver labels, stopped using only out-of-bag development clusters, and hash-frozen before protected test evaluation. The held-out historical-identity numerical test contains 15,179 observed states from 175 aerofoils. These identities were previously inspected and are not a pristine independent family sample.

On the controlled calibration data, support-only AUROC is 0.7532; combined support-plus-disagreement AUROC is 0.8893. The combined score contains ranking information, not a newly deployed guaranteed coverage policy.

A separately sealed 200-geometry confirmation cohort, chosen before training and predicted before its labels, yielded 17,968/22,600 clean numerical outcomes. The frozen incremental high-error ranking AUROC is 0.2673. The ensemble-disagreement-only AUROC is 0.8210, compared with 0.8019 for the combined model: the combined signal is not necessarily the strongest risk ranking. The cohort consists of source-development geometry mixtures, not independent aircraft families. All solver failures remain in the denominator of requested cases.

The full candidate-specific NSGA-II profile ran ten seeds at population128/generations200/scenarios64; a matched constrained nominal-only ten-seed control also completed at 128×200. The predeclared reporting verdict is that both fixed multi-seed numerical comparisons pass. Reference-scaled distance is 0.0243; historical pooled-range distance is 0.1029. A finite seeded search, successful or unsuccessful, cannot establish global optimality.

Fresh strict selected-design XFOIL was attempted for 819 cases. 290 had all fixed required-lift and solver-completion conditions; 529 did not. The final paper keeps paired-success denominators, adverse tails and any unfavorable ranking reversals.

## Independent research readiness decision

The software reconstruction and numerical evidence are auditable. **Independent viscous RANS CFD and physical experiments have not been performed.** A measured manufacturing distribution is unavailable; the field remains prescribed sensitivity. Ncrit is not itself a RANS transition/turbulence specification. No journal submission, coauthor CRediT confirmation, funding/conflict disclosure approval or probability of acceptance is claimed.

Ava and the authors must confirm the physical mission and boundary conditions, run converged mesh/domain/transition-sensitive independent CFD at matched lift, retain full force and moment histories and failed cases, examine any changed ranking, approve declarations and choose the target journal. This remains a major-revision scientific gate, not something a code/test/packaging pass can waive.

## Traceability and artefact acceptance

Source fixes, training hashes, frozen cohort identity, before-label risk scores, the exact numerical loss/failure accounting, ten seeded robust and nominal records, 2,048-scenario sensitivity and fresh selected-case receipts are retained in the repository/evidence archives. The current paper and report are separate standalone Word files; source/claim matrices and the SHA256 manifest are the authority. The authors should inspect both in native Word, including every page, before distributing.
