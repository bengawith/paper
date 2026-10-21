# Geometric Provenance and Empirical Reliability in Machine-Learning Aerofoil Optimisation: A Controlled XFOIL Study

## Authors

Benjamin Ian George Gawith; Ava Shahrokhi; Keivan Navaie

## Affiliations

Benjamin Ian George Gawith and Ava Shahrokhi: Liverpool John Moores University, Liverpool, UK. Keivan Navaie: Lancaster University, Lancaster, UK. The authors must verify their exact current institutional affiliations and designate the corresponding author before submission.

## Keywords

Aerofoil; surrogate modelling; numerical reproducibility; Class–Shape Transformation; neural ensemble; reliability screening; uncertainty quantification; multi-objective optimisation

## Highlights

- An independent audit found that 1,001 of 1,560 original aerofoil encodings changed after correcting coordinate-order interpretation.
- A controlled numerical dataset was generated on the exact CST geometry inputs, retaining all solver failures and missing observations.
- A new five-member ensemble was trained without using the calibration or protected historical test partitions for early stopping.
- A separately sealed 200-geometry numerical confirmation cohort was selected before model training and labelled only after predictions were frozen.
- Robust and nominal search outcomes, including any instability or adverse results, are reported without treating same-solver XFOIL checks as independent CFD.

## Abstract

Aerofoil surrogate models can interpolate numerical flow labels accurately without necessarily supporting reliable geometry selection. We audit an earlier five-member neural surrogate and discover a coordinate-order defect that changes 1,001 of 1,560 stored geometry encodings. Additional solver comparisons show that reorienting a source aerofoil can change the aerodynamic response at the same recorded angle of attack. Instead of reusing those uncertain geometry–label pairings, we assemble a controlled, fixed-operating-condition dataset using local XFOIL calculations performed on the exact twelve-parameter Class–Shape Transformation reconstructions supplied to the model. The programme requests 141,928 states across 1,256 geometries and retains 106,931 complete numerical observations; 34,997 requested states are not available as clean, finite, positive-drag labels. A five-member point-conditioned ensemble is trained using only development geometry clusters and evaluated on 15,179 observed states from 175 protected historical identities, achieving R² 0.9978, 0.9818 and 0.9882 for lift, logarithmic drag and moment, respectively. On a separately selected and prospectively frozen 200-geometry numerical confirmation cohort, 17,968 of 22,600 requested states are observed. On the sealed cohort with observable solver labels, the calibration-trained frozen combined support/disagreement score raises high-error ranking AUROC by 0.267 over support alone; the disagreement-only score performs slightly better still (0.821 versus 0.802). The predeclared 128 × 200 × 64 × 10 robust-design study met both specified numerical agreement diagnostics. Fresh selected-design local XFOIL comparisons attempted 819 strictly defined local solver cases, with 290 usable outcomes and 529 failures or rejected cases. Because the labels and verification share a numerical solver family, and the prescribed manufacturing field is not measured, these outcomes are conditional numerical evidence, not independent physical validation or proof of global aerodynamic optimality.

## Nomenclature

| Symbol | Definition | Units |
|---|---|---|
| α | Angle of attack in the fixed source/solver coordinate frame | degrees |
| C_L, C_D, C_M | Lift, drag, and pitching-moment coefficients | dimensionless |
| A_U,i, A_L,i | Upper and lower CST Bernstein coefficients | dimensionless |
| Re, M, Ncrit | Reynolds number, Mach number, XFOIL transition parameter | dimensionless |
| ε | Prescribed surface-normal deviation scale | chord fraction |
| C_D,w | Weighted drag at three prescribed lift targets | dimensionless |
| CVaR95 | Upper-five-percent tail mean of the penalised objective | objective units |
| d_s | Standardised nearest-training-state support distance | dimensionless |
| OOB | Out-of-bag development geometry-cluster validation | — |

## 1. Introduction

Machine-learning surrogates are widely used to reduce the repeated expense of aerodynamic analysis during preliminary optimisation [1–7]. CST parameterisation allows a compact geometry description [8,9], and XFOIL permits economical numerical exploration under prescribed operating conditions [10]. Efficient numerical prediction nevertheless differs from demonstrating a superior or robust selected design: data provenance, geometry interpretation, residual dependence, optimisation-induced distribution shift, solver convergence and constraints all affect the conclusion.

Confidence-aware aerodynamic tools and robust surrogate-assisted optimisation already exist [4,5,27,29]. Deep ensembles can provide useful predictive disagreement information [16], but spread is not itself a calibrated epistemic guarantee. Likewise, nominal marginal conformal coverage [17] does not automatically hold after adaptive design selection [28]. This study does not claim to originate any of these components.

The motivation is a reproducible implementation audit of a previously reported framework. The first audit identified incorrect treatment of two-block coordinate files, a reference-shape increment in the manufacturing operator, inappropriate post-hoc front normalisation and inconsistent admission of reconstructed geometry. We retain the earlier results as historical evidence, then construct a separate numerical dataset whose aerodynamic labels are explicitly paired with the actual reconstructed geometry inputs. The resulting model and design comparisons are evaluated under an auditable, preregistered local protocol.

The research questions are whether a geometrically consistent surrogate achieves useful grouped numerical prediction accuracy; whether support and ensemble disagreement add residual-risk information on new, sealed geometries; whether a corrected candidate-specific robustness search meets its unchanged multi-seed numerical agreement criteria; and whether deliberately selected fixed designs retain advantages in a strict solver verification protocol that preserves every failed case.

## 2. Numerical data and geometric representation

### 2.1 Original corpus audit

The original source used 1,560 aerofoils and 179,780 numerical states obtained from UIUC geometry files and an archived Airfoil Tools/XFOIL-labelled polar collection [10–12]. Recomputing geometry features with a coordinate-order-aware parser altered 1,001 stored aerofoil vectors. A fitted CST trailing-edge coefficient could also be negative even for an intended closed source shape. These faults could affect both the physical support definition and model input–output consistency. The historical 26,692-state metrics are retained for reproducibility comparison, not represented as validation of the repaired physical-geometry pipeline.

Fresh six-point local solver comparisons of raw and normalised coordinates demonstrated that numerical polars cannot generally be assumed invariant to geometry normalisation at fixed recorded incidence. For one archived geometry, the raw-coordinate lift root-mean-square disagreement with the stored polar was approximately 0.00127, while the rotated/resampled-coordinate comparison was approximately 0.173, under the specific local checks. This observation motivated a separate exact-geometry label campaign rather than retrospective angle or coefficient relabelling.

### 2.2 Exact-coordinate controlled labels

The controlled experiment uses 1,256 geometry-only admitted twelve-parameter CST reconstructions, selected without aerodynamic outcome-based screening. The original format, ambiguous input handling, fit accuracy and geometric admission use an explicit source policy, including a nonnegative fitted trailing-edge thickness and a maximum reconstruction deviation of 0.005 chord. Physical near-duplicate clusters are recomputed, and their partitions inherit the most protected old designation (test ahead of calibration ahead of development). No old protected geometry is introduced into development through reclustering.

For each exact CST shape, the local XFOIL 6.99 executable is provided that shape's saved coordinates, Re = 10⁶, M = 0, Ncrit = 9 and a fixed 200-iteration limit. Directed sweeps from 0° to 16° and 0° to −12° use 0.25° increments, requesting 113 distinct incidences. A zero process return code, no timeout, finite lift/drag/moment and strictly positive drag are required before any printed row is admitted. Consistent duplicate zero-incidence rows are merged; contradictory duplicates are rejected. Missing states are not filled, and a partially observed polar is not described as a complete numerical verification.

**Table 1. Controlled numerical data coverage and protected partitions.**

| Group | Aerofoils with observed labels | Geometry clusters | Usable numerical states |
|---|---:|---:|---:|
| Development | 858 | 850 | 74,946 |
| Calibration | 191 | 189 | 16,806 |
| Protected historical test | 175 | 173 | 15,179 |
| Total observed | 1,224 | 1,212 | 106,931 |
| Requested states | 1,256 source geometries | — | 141,928 |

Of the 141,928 states, 34,997 are unobserved or ineligible. Only 44 source geometries have all 113 requested states. This large solver-selection effect is a limitation of the label distribution: evaluation metrics condition on the observed numerical rows and do not estimate predictions for missing or non-converged regimes.

### 2.3 CST parameterisation and conditions

Five lower and five upper Bernstein weights, a leading-edge modifier, and a trailing-edge thickness give twelve parameters, followed by incidence as the thirteenth neural input [8,9]. Geometry is reconstructed, checked for positive cross-sectional area and non-crossing surfaces, and represented on common cosine-spaced chordwise knots. The representation is an approximation to the original file, not proof of identity to that file. The controlled numerical experiment avoids this ambiguity by evaluating aerodynamic labels on the reconstruction itself.

**Table 2. Geometry features and fixed aerodynamic mission definition.**

| Description | Number or value | Interpretation |
|---|---|---|
| Lower and upper CST weights | 5 + 5 | Fourth-degree Bernstein surfaces |
| Leading-edge modifier | 1 | Shared CST feature |
| Trailing-edge thickness | 1 | Nonnegative fitted coefficient |
| Neural incidence feature | 1 | α in degrees |
| Source solver conditions | Re 10⁶; M 0; Ncrit 9 | Fixed numerical domain |
| Required lift targets | 0.4, 0.7, 1.0 | On the initial strictly rising branch |
| Required-lift weights | 0.25, 0.50, 0.25 | Mean weighted drag |
| Risk measure | Mean and empirical CVaR95 | Penalty-inclusive rather than always physical drag |

## 3. Predictive and reliability methodology

### 3.1 Neural model and stopping

The point-conditioned surrogate predicts C_L, log(C_D) and C_M jointly. Five residual MLPs each have a thirteen-input projection, two residual blocks of two affine transformations, and three output heads. The fixed inherited architecture has width 512, ReLU activation, dropout approximately 0.122 and 1,064,451 trainable parameters per member. Earlier development-only hyperparameter search selected this configuration [18–21]; there is no fresh architecture search on the new protected data.

Member training resamples development geometry clusters. Only out-of-bag development clusters select the stopping epoch, with a 300-epoch maximum and patience of 25; calibration and test observations do not enter early stopping, scaling or architecture selection. Masks retain actual observed target availability. Training histories and checkpoint/scaler SHA-256 hashes are preserved for all five members. HistGradientBoosting, ridge regression and a weighted-mean baseline are fitted to the same development records without retuning on the test.

### 3.2 Empirical reliability screen and prediction bands

The inherited empirical rule uses standardised development-support distance, output-specific ensemble spread thresholds and incidence bounds. Its thresholds and residual-based marginal bands are refitted from the controlled calibration partition only. The accepted fraction is not a claim of conditional, simultaneous polar-wise or adaptive-selection coverage. Rejection is distinguished from lack of a numerical ground-truth solver row.

A separate five-fold grouped risk study uses geometry-cluster-disjoint folds within calibration only. Each training fold independently defines error scales, upper-decile error labels and feature standardisation. Ridge models compare support only, spread only and support plus spread. Their held-out out-of-fold (OOF) rankings are evaluated without promoting a new operational trust cutoff. An explicitly labelled geometry-cluster bootstrap conditions on the fitted OOF scores rather than reproducing the whole training procedure.

### 3.3 Prospective sealed numerical confirmation

Before controlled-model training, a cohort of 200 geometry-only candidates was selected by fixed-seed convex combinations of development CST shapes and screened for physical validity and distance from known source geometries. This is genuinely out of the model-fitting and risk-calibration data, but it is not an independent unseen aircraft family or a manufacturing population sample. After model and calibration-risk fitting, predictions, risk scores, feature scaling and threshold definitions were frozen and hashed before any corresponding XFOIL labels were generated.

Every confirmation shape was submitted to the same real numerical conditions and clean-process admission rule. Metrics are computed only for observed matching labels. Missing solver outcomes are enumerated, not predicted or imputed as truth. Bootstrap summaries resample geometry identities with frozen predictions and do not create a mathematical guarantee under optimizer feedback.

### 3.4 Candidate-specific prescribed uncertainty and constraints

The geometrical uncertainty is a prescribed spatially correlated, piecewise-linearly interpolated surface-normal field. Its sampled controls are coupled between surfaces according to a specified latent correlation, normalised by individual realisation and tapered near leading and trailing edges. Amplitudes of 0.1%, 0.25%, 0.5% and 1% chord are sensitivity settings, not measured manufacturing tolerances or asserted failure probabilities.

The corrected operator displaces *each candidate's own physical surface*, validates displaced coordinates, refits CST for surrogate use and validates the reconstructed fit. Invalid geometry, missing required lift roots and empirical trust exits receive a declared objective penalty; none is evaluated as an observed nominal aerodynamic outcome. Required lift intersections are constrained to a unique crossing on the initial rising polar branch within the declared service domain. Thickness, area, leading-edge-radius proxy and pitching moment remain explicit nominal constraints.

### 3.5 Predeclared multi-seed optimisation and solver verification

NSGA-II [13,14] is applied to the mean and sample CVaR95 penalty-inclusive weighted-drag objectives with 128 individuals, 200 generations, 64 common latent scenarios and 10 distinct seeds. Each generation is checkpointed and all failures/empty feasible fronts are counted. An acceptance condition of at least 80% feasible seeds and a mean cross-seed normalised front distance no larger than 0.20 is retained. Normalisation by fixed pre-run reference objectives is reported separately from the historically used pooled-range normalisation, which is retrospective and cannot establish prospective acceptance.

A matched-budget nominal-only control uses ten other independent seeds, 128 individuals and 200 generations under the same geometry, moment, service-lift and empirical screen constraints. Finite feasible pooled fronts are searched by a deterministic, predeclared ranking independent of verification outcomes. Selected candidates are then checked on 2,048 fresh common latent scenarios and, where relevant, on actual displaced geometries using strict local XFOIL. The latter requires all 45 requested incidence points from 0° to 11° (step 0.25°), clean process completion, positive finite drag and all three service-root intersections in 0° to 8°. Incomplete cases are retained but not assigned weighted drag. A rerun of the same XFOIL family is neither independent RANS CFD nor an experiment.

## 4. Results

### 4.1 Controlled model reproduction and external numerical error

The controlled labelled corpus and its complete source hashes reconcile to 106,931 usable states on 1,224 geometries. Model freeze preceded evaluation on 175 protected historical identities. The controlled test includes 15,179 observed states, of which 9,913 are accepted and 5,266 are rejected by the recalibrated inherited screen.

**Table 3. Controlled held-out numerical error and baseline comparison on the same repaired geometry representation.**

| Target | Ensemble R² | Ensemble MAE | Geometry-macro MAE | Ridge R² | HistGradientBoosting R² |
|---|---:|---:|---:|---:|---:|
| C_L | 0.9978 | 0.02205 | 0.02185 | 0.9656 | 0.9933 |
| log(C_D) | 0.9818 | 0.04737 | 0.05123 | 0.1628 | 0.9518 |
| C_M | 0.9882 | 0.00331 | 0.00320 | 0.8966 | 0.9587 |
| Linear C_D | 0.9444 | 0.00132 | 0.00135 | — | — |

The controlled input and label distributions are not the same as the original archived model, so improvements in R² or MAE must not be inferred solely from a numerical comparison of the two published runs. Every accuracy statistic refers to clean solver-observed states.

### 4.2 Empirical screen, held-out coverage and residual risk

Table 4 distinguishes the inherited operational screen from a newly evaluated residual-ranking diagnostic. Test state-wise residual-band coverage is 96.8%, 95.9% and 97.7% for lift, logarithmic drag and moment, respectively; whole-observed-polar coverage is smaller and does not represent a simultaneous polar guarantee.

**Table 4. Controlled reliability calibration and prospective numerical confirmation.**

| Endpoint | Support only | Support plus disagreement | Difference / interpretation |
|---|---:|---:|---|
| Calibration-cluster OOF high-error AUROC | 0.753 | 0.889 | 0.136 |
| Calibration ensemble-only AUROC | — | 0.884 | Separate disagreement-only ranking model |
| New sealed-shape, observed-label AUROC | 0.535 | 0.802 | 0.267 |
| Sealed confirmation ensemble-only AUROC | — | 0.821 | higher versus combined |
| Confirmation observation coverage | — | — | 17,968/22,600 numerical states |
| Confirmation empirical screen accepted | — | — | 11,718/17,968 observed states |

The calibration-cluster fixed-OOF bootstrap difference interval is ['0.091', '0.200']; the sealed-cohort frozen-model geometry-bootstrap difference interval is ['0.200', '0.340']. Their interpretations differ: the former describes retrospective calibration scores, whereas the latter checks prelabel frozen scores on different numerical geometries. Neither accounts for uncertain missing solver outcomes or grants optimizer-induced conditional coverage. Notably, the disagreement-only model has 0.821 AUROC on the sealed cohort and higher performance than the combined score (0.802); consequently the combined model is not asserted to be the best reliability ranking mechanism. Figure 3 displays all three risk rankings, and Supplementary Figure S1 shows the empirical retained-state error relationship.

### 4.3 Corrected search feasibility and agreement

The nominal reference's fixed pre-run penalty-inclusive objectives are ['0.01209', '0.01638']. The robust ten-seed programme returned 10 seeds with at least one feasible final design and 371 feasible final entries in total. Its cross-seed distance is 0.024 on prospective reference scales and 0.103 under the original pooled-range numerical comparator; the unchanged 0.20/80% numerical acceptance met both specified numerical agreement diagnostics. All 2,000 completed generations across ten seeds are recorded, including infeasible or unstable outcomes. Numerical front agreement is not evidence of global Pareto optimality.

**Table 5. Full-seed optimization and matched-budget nominal control.**

| Property | Corrected robust NSGA-II | Nominal-only constrained control |
|---|---|---|
| Independent seeds | 10 | 10 |
| Population × generations | 128 × 200 | 128 × 200 |
| Shared manufacturing scenarios per robust evaluation | 64 | Not applicable to nominal objective |
| Feasible seeds | 10 | 10 |
| Distinct numerical acceptance | met both specified numerical agreement diagnostics | Feasible constrained nominal search |
| Minimum recorded feasible objective | 0.00855 (penalised mean) | 0.00574 (nominal weighted drag) |

The candidate-specific original reference-increment defect is not repeated. A comparative nominal objective is not a substitute for the robust objective: the matched search budget intentionally evaluates a different response. The finite search and its constraints do not establish a physically optimum aerofoil.

### 4.4 Fresh scenario sensitivity and selected XFOIL cases

The selected risk-knee, nominal control and reference are chosen and hashed before external verification. Each is evaluated under fresh 2,048-scenario sensitivity sets. The penalty-inclusive mean/CVaR95 at 1% chord for the reference are 0.014520/0.018019, while the selected risk knee gives 0.010867/0.015445; their invalid-refit, root-failure and untrusted-scenario counts remain part of those objectives.

Strict local-XFOIL checks at fixed required lift yield the conditional paired numerical comparisons in Table 6. Results preserve solver failure, partial polars, missing required-lift roots, adverse maxima and any reversal of the predicted ranking.

**Table 6. Selected actual-coordinate XFOIL comparison against the same-solver reference. Positive reductions favour the chosen design, conditional on matched usable cases.**

| Scenario and comparator | Matched usable pairs | Mean drag reduction | Reference / candidate maximum |
|---|---:|---:|---|
| Nominal, risk-knee | 0 | not estimable | not estimable |
| 1% chord, risk-knee | 0 | not estimable | not estimable |
| 0.5% chord scramble A, risk-knee | 15 | 15.2% | 0.00860 / 0.00721 |
| 0.5% chord scramble B, risk-knee | 14 | 14.9% | 0.00814 / 0.00695 |
| Nominal, nominal-control comparator | 0 | not estimable | not estimable |

The fixed strict XFOIL nominal grid yielded no complete matched reference–candidate pair for at least one selected comparison, so a new nominal drag-reduction percentage cannot be estimated for that comparator; historical favourable nominal results must not be substituted. A maximum among selected, successfully converged cases is not an unconditional CVaR95 estimate. The missing verification cases comprise difficult flow and geometry conditions rather than a validated random missingness mechanism. The numerical selection rule never used those outcomes to choose the final candidates.

### 4.5 Computational and evidence accounting

The historical GTX 1660 predictor-only benchmark separated single-state, polar and batched evaluation latencies. It does not measure an end-to-end optimisation speedup over XFOIL. The new training member logs, exact source/data/scaler hashes, evaluation receipts, numerical solver executable hash, per-seed optimization checkpoints and manuscript claim matrix form the reproducibility record. Readiness is assessed independently for software, training and calibration, prospective numerical confirmation, robust search, selected same-solver checks and independent CFD.

## 5. Discussion

Three distinguishable sources of uncertainty are visible. Predictive residual error is measured conditional on available numerical solver labels. The epistemic/empirical screen describes distance from historical feature support and member disagreement but has no automatic probability or accepted-domain guarantee. Geometric manufacturing uncertainty is simulated with a declared field and amplitude, not measured from produced aerofoils. Penalty-inclusive robust objectives mix these mechanisms, so their physical interpretation cannot be reduced to 'drag'.

The new exact-coordinate dataset resolves an important ambiguity: each training geometry is exactly the one supplied to its numerical solver. The large number of missing states, however, means it is a *convergence-selected* numerical sample, especially at challenging incidence. Simple baselines, macro errors and the confirmation cohort reduce some overinterpretation, but neither replaces independent flow physics or experiments.

The separately sealed confirmation is useful because geometry selection precedes the new model training, and its risk predictions precede all confirmation labels. This blocks a common direct adaptive leak. It does not create an independent new family distribution: the test shapes are mixtures of older development geometries and can remain correlated with them. Resampling frozen scores is not a full uncertainty analysis over model retraining or manufactured geometry.

The ten-seed study measures repeatability within a finite optimization protocol [13,14]. Failure to meet the fixed 0.20/80% rule must be reported directly, not rescued by changing normalisation after examining results. Passing it would indicate bounded algorithmic consistency, not global optimality. Strict lift-root feasibility and equal moment/geometry constraints remain necessary for engineering comparisons.

Existing confidence-guided airfoil modelling and robust optimisation literature [4,5,27,29] limit any claim of architectural novelty. The useful contribution is an executable demonstration of why geometry–label provenance, controlled data admission, prospective ranking checks, common physical manufacturing scenarios and solver-failure accounting jointly matter to selected-design evidence.

Independent viscous RANS CFD and experimental validation are not part of these XFOIL runs. A subsequent coauthor-approved comparison should document Reynolds/transition modelling, grid and domain convergence, wall resolution, equal achieved lift, complete force/moment histories, uncertainty in ranking and all rejected designs. Ncrit 9 cannot be treated as a directly transferable RANS turbulence setting.

## 6. Conclusion

A source-grounded review found and repaired consequential geometry, refit-admission, trust-grouping and optimization-provenance defects. The replacement controlled experiment generated 106,931 genuine numerical labels on exact CST geometries, retrained five ensemble members, evaluated protected and sealed confirmation cohorts and executed the declared multi-seed design protocol. Its numerical prediction, risk-ranking, solver-convergence and optimization results are retained whether favourable or adverse. Any conclusions about selected aerofoils remain conditional on the same solver family and prescribed geometric disturbances. Without independent CFD, measured manufacturing variability and coauthor-approved research claims, the work is a controlled numerical research study rather than a certified robust aerofoil design or a submission-ready physical-validation claim.

## CRediT authorship contribution statement

Proposed, subject to explicit coauthor approval: Benjamin Ian George Gawith — Conceptualisation, Methodology, Software, Data curation, Formal analysis, Writing – original draft; Ava Shahrokhi — Supervision, Conceptualisation, Methodology, Writing – review & editing; Keivan Navaie — Supervision, Methodology, Writing – review & editing.

## Declaration of competing interest

The authors must verify and approve their disclosure declarations before submission. No coauthor declaration is inferred by this computational audit.

## Declaration of generative AI use

AI-assisted tools were used to inspect implementation, reproduce numerical calculations and prepare scientific drafting. All retained quantitative claims must be linked to executable source and evidence receipts. Responsibility for the final manuscript and author disclosures remains with the authors.

## Data and code availability

The repository contains the source and compact audit evidence; the larger labelled-coordinate, checkpoint and solver-case archives are retained as separately hashed local artifacts. The controlled XFOIL labels, immutable geometry selections and model weights are distinguished from the historical dataset, which has not been erased or relabelled. Access to third-party coordinate source files remains subject to their respective terms. Local OneDrive packaging does not itself prove a cloud-synchronised or public research deposit.

## Funding

The authors must verify this statement before publication. The earlier manuscript reported no specific grant funding, which has not been independently reconfirmed.

## Figures

**Figure 1.** Held-out controlled-geometry numerical surrogate accuracy for C_L, log C_D and C_M. All points are observed clean-process XFOIL labels, not CFD or experimental observations.

![Figure 1](figures/fig_surrogate_accuracy.png)

**Figure 2.** Exact-coordinate label admission across development, calibration and the protected historical test; observed versus requested state counts and geometric support, with solver failures retained.

![Figure 2](figures/fig_perturbation_validity.png)

**Figure 3.** Calibration-cluster out-of-fold and sealed numerical-confirmation residual-risk ranking, keeping prospective and retrospective calculations separate.

![Figure 3](figures/fig_surrogate_trust_stress.png)

**Figure 4.** Corrected ten-seed robust objective fronts and finite agreement under reference-scaled normalisation. The plot is not proof of global Pareto optimality.

![Figure 4](figures/fig_pareto_front.png)

**Figure 5.** Conditional strict selected-design XFOIL matched-case comparison, including unsuccessful counts and adverse sampled outcomes.

![Figure 5](figures/fig_design_verification.png)

**Figure 6.** Ten-seed convergence trajectories and feasible-population shares. Any failed numerical acceptance is displayed unchanged.

![Figure 6](figures/fig_worstcase_polar.png)

**Figure 7.** Fresh 2,048-scenario prescribed-amplitude penalty-inclusive objective sensitivity for the fixed reference and risk-knee candidates.

![Figure 7](figures/fig_manufacturing_risk.png)

**Supplementary Figure S1.** Frozen-model prediction error and empirical selective retention on the observed confirmation label subset; rejected/missing states are not assigned a made-up true error.

![Supplementary Figure S1](figures/fig_ml_selective_coverage.png)


## References

[1] Drela, M. & Giles, M. B. (1987). Viscous–inviscid analysis of transonic and low Reynolds number airfoils. AIAA Journal, 25(10), 1347–1355. https://doi.org/10.2514/3.9789

[2] Vinuesa, R. & Brunton, S. L. (2022). Enhancing computational fluid dynamics with machine learning. Nature Computational Science, 2(6), 358–366. https://doi.org/10.1038/s43588-022-00264-7

[3] Brunton, S. L., Noack, B. R. & Koumoutsakos, P. (2020). Machine learning for fluid mechanics. Annual Review of Fluid Mechanics, 52, 477–508. https://doi.org/10.1146/annurev-fluid-010719-060214

[4] Li, J., Bouhlel, M. A. & Martins, J. R. R. A. (2019). Data-based approach for fast airfoil analysis and optimization. AIAA Journal, 57(2), 581–596. https://doi.org/10.2514/1.J057129

[5] Li, J., Du, X. & Martins, J. R. R. A. (2022). Machine learning in aerodynamic shape optimization. Progress in Aerospace Sciences, 134, 100849. https://doi.org/10.1016/j.paerosci.2022.100849

[6] Shahrokhi, A. & Jahangirian, A. (2010). A surrogate assisted evolutionary optimization method with application to the transonic airfoil design. Engineering Optimization, 42(6), 497–515. https://doi.org/10.1080/03052150903305468

[7] Jahangirian, A. & Shahrokhi, A. (2011). Aerodynamic shape optimization using efficient evolutionary algorithms and unstructured CFD solver. Computers & Fluids, 46(1), 270–276. https://doi.org/10.1016/j.compfluid.2011.02.010

[8] Kulfan, B. M. (2008). Universal parametric geometry representation method. Journal of Aircraft, 45(1), 142–158. https://doi.org/10.2514/1.29958

[9] Kulfan, B. M. & Bussoletti, J. E. (2006). "Fundamental" parametric geometry representations for aircraft component shapes. In 11th AIAA/ISSMO Multidisciplinary Analysis and Optimization Conference, AIAA 2006-6948. https://doi.org/10.2514/6.2006-6948

[10] Drela, M. (1989). XFOIL: An analysis and design system for low Reynolds number airfoils. In Low Reynolds Number Aerodynamics, Lecture Notes in Engineering, vol. 54, Springer, 1–12. https://doi.org/10.1007/978-3-642-84010-4_1

[11] Selig, M. S. (n.d.). UIUC Airfoil Data Site. University of Illinois Urbana–Champaign. Accessed 14 October 2026. https://m-selig.ae.illinois.edu/ads/coord_database.html

[12] Airfoil Tools (n.d.). Airfoil Tools. Historical source of archived XFOIL polars; current site access was not verified during this review. http://airfoiltools.com/

[13] Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197. https://doi.org/10.1109/4235.996017

[14] Blank, J. & Deb, K. (2020). pymoo: Multi-objective optimization in Python. IEEE Access, 8, 89497–89509. https://doi.org/10.1109/ACCESS.2020.2990567

[15] Rockafellar, R. T. & Uryasev, S. (2000). Optimization of conditional value-at-risk. Journal of Risk, 2(3), 21–41. https://doi.org/10.21314/JOR.2000.038

[16] Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems 30 (NIPS 2017), 6402–6413. https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html

[17] Angelopoulos, A. N. & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511. https://arxiv.org/abs/2107.07511

[18] Loshchilov, I. & Hutter, F. (2019). Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR 2019). arXiv:1711.05101. https://arxiv.org/abs/1711.05101

[19] Nair, V. & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. In Proceedings of the 27th International Conference on Machine Learning (ICML 2010), 807–814. https://icml.cc/2010/papers/432.pdf

[20] Paszke, A., Gross, S., Massa, F. et al. (2019). PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32 (NeurIPS 2019), 8026–8037. https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html

[21] Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2623–2631. https://doi.org/10.1145/3292500.3330701

[22] Harris, C. R., Millman, K. J., van der Walt, S. J. et al. (2020). Array programming with NumPy. Nature, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2

[23] Virtanen, P., Gommers, R., Oliphant, T. E. et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature Methods, 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2

[24] Pedregosa, F., Varoquaux, G., Gramfort, A. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830. https://www.jmlr.org/papers/v12/pedregosa11a.html

[25] Abbott, I. H. & von Doenhoff, A. E. (1959). Theory of Wing Sections, Including a Summary of Airfoil Data. Dover Publications. https://store.doverpublications.com/products/9780486605869

[26] Anderson, J. D. (2017). Fundamentals of Aerodynamics. 6th edn. McGraw-Hill Education.

[27] Pichitkul, A., Boksawat, A., Phiboon, T., Tantrairatn, S., Bureerat, S., Kasemsri, R. & Ariyarit, A. (2026). An intelligent multi-objective robust design optimization framework based on PathLine Transformation and surrogate-assisted sampling for uncertainty-aware airfoil design. Applications in Engineering Science, 25, 100310. https://doi.org/10.1016/j.apples.2026.100310

[28] Fannjiang, C., Bates, S., Angelopoulos, A. N., Listgarten, J. & Jordan, M. I. (2022). Conformal prediction under feedback covariate shift for biomolecular design. Proceedings of the National Academy of Sciences, 119(43), e2204569119. https://doi.org/10.1073/pnas.2204569119

[29] Sharpe, P. & Hansman, R. J. (2025). NeuralFoil: An Airfoil Aerodynamics Analysis Tool Using Physics-Informed Machine Learning. arXiv:2503.16323v1. https://arxiv.org/abs/2503.16323

