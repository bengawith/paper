# Empirical Reliability Screening for Machine-Learning Aerofoil Prediction, with Selected XFOIL Comparisons

## Authors

Benjamin Ian George Gawith ᵃ, *Ava Shahrokhi ᵇ, Keivan Navaie ᶜ

## Affiliations

ᵃ School of Engineering, Liverpool John Moores University, Liverpool, UK

ᵇ Department of Maritime and Mechanical Engineering, Liverpool John Moores University, Liverpool, UK

ᶜ School of Computing and Communications, Lancaster University, Lancaster, UK

## Corresponding Author

Ava Shahrokhi, A.Shahrokhi@ljmu.ac.uk

## Keywords

Aerodynamic surrogate; Empirical reliability screening; Class–Shape Transformation; Geometry preprocessing; Ensemble disagreement; Numerical verification

## Highlights

- Historical held-out prediction metrics reproduce, but a coordinate-order defect limits their geometric interpretation.
- Correct handling of coordinate blocks changes 1,001 of the 1,560 stored aerofoil feature vectors.
- Calibration-only, geometry-cluster-separated analysis identifies additional error-ranking information in ensemble disagreement.
- A fixed selected geometry has 18.7% lower nominal weighted drag than the reference in local XFOIL.
- Neither a repaired-model robust Pareto optimum nor independent CFD validation has been established.

## Abstract

We examine the reliability of a point-conditioned machine-learning aerofoil surrogate and distinguish reproducible numerical results from evidence sufficient for aerodynamic design. A five-member multilayer-perceptron ensemble predicts lift, logarithmic drag and pitching moment from twelve Class–Shape Transformation parameters and angle of attack at Reynolds number 10⁶, Mach number zero and Ncrit 9. Its historical held-out R² values of 0.989, 0.922 and 0.918 reproduce on 26,692 states. However, an implementation audit identifies incorrect interpretation of two-block coordinate files: correcting the geometry parser changes 1,001 of 1,560 stored feature vectors. The historical metrics therefore do not validate the repaired physical-geometry pipeline. The frozen empirical screen accepts 66.1% of held-out states and is effectively governed by support distance. In a separate five-fold analysis within the historical calibration partition, with near-duplicate geometry clusters kept together, adding ensemble disagreement increases the AUROC for detecting large errors from 0.768 to 0.867. This remains a retrospective diagnostic, not a new validated screening policy. Independent recomputation of stored local-XFOIL polars confirms 18.7% lower nominal required-lift weighted drag for one selected geometry and mean reductions of 15.4% and 15.8% among 92 and 96 completed matched perturbation cases. Source-level geometry, refit-admission and optimisation-cache defects are corrected, but the trained lineage and full robust optimisation have not been regenerated. The results support a limited fixed-geometry numerical comparison and demonstrate why geometric provenance, selection effects and separate solver verification must accompany surrogate accuracy claims.

## Nomenclature

| Symbol | Description | Units |
|---|---|---|
| α | Angle of attack | deg |
| C_L, C_D, C_M | Lift, drag and pitching-moment coefficients | Dimensionless |
| A_L,i, A_U,i | Lower and upper CST coefficients | Dimensionless |
| Δz_LE, Δ_TE | Leading-edge modifier and trailing-edge thickness | Chord-normalised |
| ε | Prescribed surface-displacement scale | Fraction of chord |
| C_D,w | Required-lift weighted drag | Dimensionless |
| CVaR95 | Mean of the upper 5% probability mass of an objective | Objective units |
| d_s | Standardised development-support distance | Dimensionless |
| Re | Chord Reynolds number | Dimensionless |

## 1. Introduction

The design and optimisation of aerodynamic surfaces require repeated estimates of lift, drag and moment. Numerical flow analysis and experiments provide the basis for engineering decisions, but their cost can constrain early design-space exploration. Surrogate models are consequently used to approximate aerodynamic responses and accelerate optimisation [1–5]. Earlier work by the present research group applied surrogate-assisted evolutionary methods to aerodynamic shape optimisation [6,7].

A compact geometric representation can make this approximation convenient. Class–Shape Transformation (CST) describes each surface through a small set of coefficients [8,9], while a point-conditioned model can estimate the aerodynamic coefficients at a requested incidence without padding a polar to a fixed grid. Accurate interpolation on a held-out dataset is nevertheless not sufficient to establish the reliability of a design selected by an optimiser. Errors in coordinate interpretation, geometry reconstruction, uncertainty propagation or the selection rule can affect the engineering conclusion even when the prediction metrics are reproducible.

This study examines those distinctions using an archived neural surrogate, empirical reliability screening and selected local-XFOIL calculations. The contribution is not the invention of robust aerofoil optimisation or uncertainty-aware neural prediction. Related robust surrogate-assisted optimisation has already been reported [27], and NeuralFoil combines geometric parameterisation, learned aerodynamic responses, confidence information and design optimisation over a broader operating envelope [29]. Prediction intervals also require particular care when a model influences the distribution of subsequently selected inputs [28].

Our aim is to establish which conclusions survive a reproducibility and implementation audit. We report the historical prediction results, a corrected calibration-group analysis and fixed-geometry XFOIL comparisons, while identifying the preprocessing and optimisation work required before a stronger design claim is justified. The distinction between reproducing a stored calculation and validating a physical-geometry workflow is central throughout.

## 2. Aerofoil representation and data

### 2.1 CST representation

Each aerofoil is represented by five Bernstein coefficients for each surface, one leading-edge modification weight and a trailing-edge thickness, giving twelve inputs before incidence is added (Table 1). The representation follows the CST approach [8,9]. Coordinate normalisation, surface identification, CST fitting and reconstruction are separate operations; a small fitting residual alone does not establish a valid reconstructed aerofoil.

**Table 1. The twelve geometry parameters supplied to the surrogate.**

| Parameters | Description | Count |
|---|---|---:|
| A_L,0 to A_L,4 | Lower-surface Bernstein coefficients | 5 |
| A_U,0 to A_U,4 | Upper-surface Bernstein coefficients | 5 |
| Δz_LE | Leading-edge modification weight | 1 |
| Δ_TE | Trailing-edge thickness | 1 |
| Total | Geometry features | 12 |

### 2.2 Sources and operating condition

The historical dataset contains 1,560 encoded aerofoils and 179,780 aerodynamic states. Coordinates originate from the UIUC Airfoil Data Site [11]; polar labels originate from Airfoil Tools [12] and use XFOIL-class calculations [10] at Re = 1,000,000, M = 0 and Ncrit = 9. There are 179,779 usable lift labels and 179,780 usable log-drag and moment labels. Only available source solver rows are used; the point-conditioned model does not require edge padding or extrapolated supervised targets. These are numerical labels, not experimental measurements.

The stored split assigns 1,093 aerofoils to development, 234 to calibration and 233 to the historical held-out partition. Their respective stored geometry-cluster counts are 1,088, 233 and 233. The archived cluster identifiers do not overlap across these partitions, and the recorded tuning and scaling procedures use development data. This verifies the recorded split mechanism, not the correctness of the underlying physical shapes. No authoritative family-held-out protocol is claimed.

### 2.3 Coordinate-order defect and its consequences

UIUC coordinate files do not all use one surface ordering. In the archived implementation, a fallback based on ordinate ordering mishandled files containing two leading-edge-to-trailing-edge blocks. Correcting the parser changes the feature vectors of 1,001 of the 1,560 stored aerofoils. Of 1,557 geometries that can be compared under the repaired parser, 931 historical fits exceed a maximum error of 0.005 chord, compared with 289 after repair. Three records are rejected as ambiguous rather than guessed. The audit also finds 656 reconstructed fits failing the implemented crossing-surface validity check, demonstrating that representation admission needs to be evaluated after refitting as well as before it.

The parser and reconstructed-geometry checks are corrected in the source. The frozen training data, checkpoints and original split are preserved rather than silently replaced. Rebuilding the accepted corpus, reclustering it, retraining the ensemble, recalibrating its screen and obtaining a fresh confirmation cohort remain necessary. Accordingly, all historical ML metrics below refer to the archived encoded-feature distribution; they do not establish the accuracy of a repaired physical-geometry model. The NACA 2412 feature vector is unchanged by the parser correction. The separate fixed-coordinate XFOIL comparisons are assessed directly against their stored geometry files in Section 4.4.

## 3. Methods

### 3.1 Point-conditioned ensemble

The surrogate receives thirteen inputs: the twelve geometry parameters and α. Separate output heads estimate C_L, log C_D and C_M. The retained configuration was selected from 50 Optuna trials using three-fold grouped development evaluation [21]. Its implementation contains an input projection followed by two residual blocks, each containing two affine transformations: five hidden affine layers of width 512, not simply two hidden layers. Each member has 1,064,451 trainable parameters, ReLU activations [19], layer normalisation and dropout of 0.12. Training uses AdamW [18] and a masked log-cosh objective in PyTorch [20]. The map is piecewise differentiable; ReLU does not make it globally smooth.

Five members are fitted through resampling development geometry clusters. Their mean is the predictor and their standard deviation is a disagreement signal. Deep ensembles motivate this diagnostic [16], but raw spread is not automatically calibrated epistemic uncertainty. The archived member checkpoints and scaling bundles are checked against their manifest before loading. Reproduction of their predictions does not remove the dataset limitation identified in Section 2.3.

### 3.2 Empirical screening and interval interpretation

The frozen rule combines development-support distance, target-specific ensemble-spread limits and incidence bounds. Calibration residuals provide target-wise, split-conformal-style bands [17]. The observations within an aerofoil polar are dependent, and reported state-wise coverage must not be interpreted as simultaneous coverage of a complete polar. Nor does marginal calibration establish coverage conditional on acceptance or after adaptive optimisation [28]. We therefore use the term empirical screen rather than a reliability certificate.

For an incremental disagreement diagnostic, a fixed ridge risk model is evaluated in five folds within the historical calibration partition only. Near-duplicate geometry clusters, not just aerofoil names, are kept together. Residual scaling, the high-error threshold and input standardisation are fitted on each training fold. The target marks the upper decile of the training-fold-scaled multi-output error. Three input sets are compared: support distance, ensemble spreads, and their combination. The uncertainty summary resamples 233 geometry clusters 300 times while holding the fitted out-of-fold predictions fixed; it does not include all variability from refitting the analysis. No operational screening threshold or frozen aerodynamic model is changed.

### 3.3 Prescribed geometric uncertainty

The uncertainty model applies bounded surface-normal displacements to each candidate's own coordinates. Gaussian-correlated control coefficients are interpolated piecewise linearly in chordwise position, normalised separately for each realisation and surface, and tapered near the leading edge and to zero at the trailing edge. The displacement is continuous but not a verified continuously differentiable manufacturing profile. The amplitudes 0.1%, 0.25%, 0.5% and 1% chord are prescribed sensitivity settings, not tolerances inferred from production measurements.

Common latent scenarios permit paired comparisons between fixed designs. Each displaced geometry is checked, resampled and CST-refitted for surrogate evaluation; the reconstructed fit is then checked again. Invalid scenarios are explicitly counted and penalised, not replaced by nominal outcomes. Independent coefficient and coordinate perturbations are alternative numerical representations, not measured manufacturing processes. The implemented normalisation also means a latent correlation setting should not be read as a measured correlation of finished surfaces.

### 3.4 Required-lift objective and optimisation status

Weighted drag is evaluated at C_L = 0.4, 0.7 and 1.0 with weights 0.25, 0.50 and 0.25. All three roots must exist on the admissible initial rising polar branch; failure is not bypassed by selecting a later post-stall crossing. The optimisation objectives are the mean and empirical CVaR95 of the penalised weighted-drag quantity [15], subject to geometry, moment, required-lift and screening constraints. Trust exits and invalid geometry therefore affect an objective that is not always physical drag.

The historical NSGA-II campaign [13,14] used population 128, 200 generations, 64 common scenarios and ten seeds. It transferred CST increments fitted to the reference aerofoil to other candidates instead of displacing each candidate's own surface. Its front is consequently not accepted as a candidate-specific robust result. The active implementation corrects this operator, checks refitted geometry and rejects caches whose source, model, dependency or evaluation context differs. The full corrected campaign has not been completed.

A later historical smoke profile used 32 candidates, ten generations and sixteen scenarios over ten seeds. Its front distances are normalised using the pooled observed objective ranges. Although that computation is reproducible, these are post-hoc scales, not a prospective acceptance definition. Current code reports such results as diagnostic and does not self-issue scientific acceptance. A future production protocol must freeze its evaluation and normalisation choices before observing the results.

### 3.5 Local XFOIL checks

Three intentionally selected fixed geometries are compared using local XFOIL 6.99 at the same nominal operating condition as the labels. Acceptance requires clean solver exit, all twelve requested incidences from 0° to 11°, finite positive drag and all required-lift roots. Partial polars are not filled or promoted to successful cases. Stored coordinates, commands, statuses and polars are audited separately; a successful process exit alone is insufficient.

The 1%-chord campaign requests sixteen shared scenarios per design. Two additional, independently seeded scrambled-Sobol campaigns request 128 scenarios per design at 0.5% chord. Their cases are paired within each campaign, but solver failure can select a nonrepresentative subset. These calculations share the XFOIL model family with the training labels and are not independent CFD or experimental validation. A separate finite adversarial search examines observed surrogate–XFOIL disagreement without claiming a global extremum.

## 4. Results

### 4.1 Historical prediction accuracy and model comparison

The frozen ensemble reproduces its stored predictions on all 26,692 held-out states. Historical R² values are 0.98869 for C_L, 0.92185 for log C_D and 0.91801 for C_M; corresponding MAEs are 0.05714, 0.14340 and 0.00959 (Fig. 1). Aerofoil-macro MAEs are 0.05602, 0.14839 and 0.00954, respectively. These values are not newly trained repaired-geometry results.

Table 2 retains the archived grouped-development baseline comparison. Gradient boosting is slightly stronger on the reported log-drag subset, while the MLP is stronger on moment. This supports a pragmatic choice of model, not architectural superiority. Because the shared input lineage is affected by the geometry defect, the comparison must also be repeated after corpus repair before it is used to justify physical design generalisation.

**Table 2. Historical grouped-development R² values. This is a comparison on the archived feature representation.**

| Model | C_L | log C_D | C_M |
|---|---:|---:|---:|
| MLP ensemble | 0.984 | 0.904 | 0.875 |
| Histogram gradient boosting | 0.983 | 0.915 | 0.760 |
| Ridge regression | 0.948 | 0.083 | 0.741 |
| Mean predictor | −0.001 | −0.009 | −0.000 |

The frozen screen accepts 17,642 states (66.1%) and rejects 9,050. Every rejection fails the support-distance limit; spread or angle restrictions reject no additional states independently. Accepted predictions have lower observed errors, but the original ensemble-spread cutoffs do not establish a separate selection benefit. Table 3 reports the reproduced state-wise coverage; accepted lift coverage is 97.7% when rounded to one decimal place.

**Table 3. Historical held-out error, R² and state-wise interval coverage. Drag error and R² are in linear coefficient units. Conditional accepted-set coverage is descriptive.**

| Target | MAE accepted / rejected | R² accepted / rejected | Coverage all / accepted |
|---|---|---|---|
| C_L | 0.0401 / 0.0903 | 0.9943 / 0.9783 | 0.9367 / 0.9775 |
| C_D | 0.00350 / 0.00659 | 0.8873 / 0.7935 | 0.9439 / 0.9653 |
| C_M | 0.00605 / 0.01650 | 0.9569 / 0.8728 | 0.9449 / 0.9918 |

The fractions of observed full polars whose every available state is covered are approximately 51.1%, 44.2% and 66.5% for lift, log-drag and moment. These are diagnostics on the historical polar sample, not simultaneous prediction guarantees. A calibration-scaled support/disagreement ranking also shows lower macro error as fewer historical test states are retained (Supplementary Fig. S1); this is retrospective description rather than an untouched confirmation of a new rule.

In the corrected calibration-only geometry-cluster analysis, support-only, spread-only and combined AUROCs are 0.7680, 0.8585 and 0.8667. The combined-minus-support difference is 0.0988, with a fixed-out-of-fold cluster-bootstrap interval of [0.0667, 0.1402]. The earlier grouping by aerofoil name allowed one near-duplicate cluster to cross folds; the revised grouping removes that overlap. Disagreement therefore contains incremental ranking information within these historical calibration observations, but deployment value on new optimiser-selected geometries remains untested.

### 4.2 Historical uncertainty summaries

Table 4 and Figures 2–3 retain the original prescribed-uncertainty summaries so that the earlier claims can be examined rather than silently discarded. Each amplitude used 2,048 scenarios. Trust exits occur in about 0.4% of the 0.5%-chord sample and 17% of the 1%-chord sample. The upper-tail objective includes penalties and is not an estimate of pure aerodynamic drag or physical manufacturing failure probability.

**Table 4. Historical surrogate objective summaries under the original geometry/refit admission. These results have not been regenerated under the repaired corpus and current refit checks.**

| ε (% chord) | Mean penalised objective | Empirical CVaR95 objective | Trust exits |
|---|---:|---:|---:|
| 0.1 | 0.008007 | 0.008150 | 0.0% |
| 0.25 | 0.008081 | 0.008477 | 0.0% |
| 0.5 | 0.008311 | 0.009594 | 0.4% |
| 1.0 | 0.009850 | 0.015836 | 17.0% |

Under the original admission rule at 1% chord, 2,048/2,048 correlated-field and coefficient-perturbation cases passed, compared with 6/2,048 independent-coordinate-noise cases. This demonstrates compatibility with that numerical representation and threshold, not physical impossibility of other surfaces. The newly added reconstructed-CST checks make it inappropriate to label the historical 100% acceptance as a current physical-admissibility result.

### 4.3 Optimisation diagnostics after the corrections

The original full-budget optimisation reported 188 aggregated non-dominated entries and a mean normalised front distance of 0.283, exceeding the stated 0.20 threshold. Its reference-increment uncertainty operator is additionally invalid for the intended candidate-specific calculation. The later schema-4 smoke profile has 44 feasible returned entries across ten seeds and reproduces a distance of 0.1493 using its post-hoc pooled scales (Fig. 4). Neither result establishes a currently accepted robust Pareto front.

Five historical designs were then held fixed and evaluated with the repaired source and 64 common latent scenarios. The reference has mean/CVaR95 penalised objectives 0.009421/0.019184, eight trust exits and no invalid refits. The previously selected knee has 0.009585/0.015157, fifteen trust exits and seven invalid refits. Its mean objective is therefore slightly worse than the reference under this current diagnostic, despite its lower tail objective. The historical minimum-tail candidate has 0.008938/0.015485 and fourteen trust exits. The deterministic and simple sensitivity-baseline candidates are infeasible. These are fixed-design checks using the historical learned model, not a newly optimised or independently validated result.

### 4.4 Selected fixed-geometry XFOIL comparisons

The audit reconstructs and matches all 825 stored nominal and perturbation coordinate records across the three campaigns. All satisfy the implemented direct-coordinate validity checks. Separate status, command, polar and required-lift checks reproduce the selected-design summaries below. This supports the integrity of those fixed-coordinate calculations independently of whether the original ML feature corpus was correctly constructed.

The selected candidate has nominal weighted drag 0.006051 compared with 0.007446 for the NACA 2412 reference, a reduction of 18.7%. The outside-screen comparator has 0.013960, an increase of 87.5%. The historical support distances 0.60, 0.90 and 3.72 describe how these examples were selected, not a repaired-model certificate or proof that accepted candidates are generally better.

**Table 5. Strict local-XFOIL 1%-chord comparison. Sixteen perturbation scenarios were requested per design. Marginal summaries use different completed subsets; paired differences use matched successful cases. Bootstrap brackets are descriptive resampling ranges, not guaranteed population confidence intervals.**

| Fixed design | Historical d_s | Nominal C_D,w | Completed / requested | Marginal mean C_D,w | Marginal maximum | Paired mean improvement |
|---|---:|---:|---:|---:|---:|---|
| NACA 2412 reference | 0.60 | 0.00745 | 13 / 16 | 0.00835 | 0.00909 | — |
| Selected candidate | 0.90 | 0.00605 | 13 / 16 | 0.00761 | 0.00989 | +0.00103 [0.00058, 0.00142]; 10 pairs |
| Outside-screen comparator | 3.72 | 0.01396 | 14 / 16 | 0.01414 | 0.01475 | −0.00568 [−0.00602, −0.00537]; 11 pairs |

The approximately 8.8% difference between the two marginal completed-case means does not estimate unconditional expected performance. On the ten matched reference/candidate cases, mean drag is 0.008416/0.007388, giving a 12.2% paired reduction; this is a different estimand and subset. At this sample size empirical CVaR95 reduces to a sample maximum. The selected candidate's marginal maximum is approximately 8.9% worse than the reference's, and that adverse observation is retained rather than replaced by a favourable matched-subset tail comparison.

Two 0.5%-chord campaigns each request 128 scenarios per design. Their independent scrambles yield 92 and 96 matched usable reference/candidate cases, with mean reductions of 15.4% and 15.8% (Table S1). All matched cases pass the current direct-coordinate checks. Their empirical matched-subset tail summaries also favour the candidate, but 36 and 32 requested pairs are not jointly usable. Failure is not plausibly random with respect to geometric or flow difficulty, so unconditional tail-risk claims remain unsupported.

**Table S1. Independent 0.5%-chord scrambled-Sobol campaigns, conditional on completed matched solver cases.**

| Scramble | Requested / reference usable / candidate usable | Matched usable | Paired mean reduction | Matched empirical CVaR95 change |
|---|---|---:|---:|---:|
| A | 128 / 105 / 110 | 92 | 15.4% | −12.4% |
| B | 128 / 115 / 107 | 96 | 15.8% | −11.7% |

### 4.5 Finite adversarial probing

The historical adversarial search records 49 requested candidates, 41 evaluated records, 20 geometry rejections and 18 fully converged sweeps. These are overlapping workflow counts, not a disjoint partition. The largest recorded discrepancy statistic is approximately 0.67 in lift outside the historical screen. The recorded mean outside-screen gap is approximately 0.27, compared with 0.043 for only one wholly accepted probe (Fig. 6).

Figure 7 is a different selection: the largest mean gap among the eligible converged records used by that plotting routine, approximately 0.52 with a maximum gap of 0.61. It is not the overall 0.67 record. Partial solver results, sparse accepted probes and the historical geometry representation prevent a global worst-case conclusion or a reliable estimate of accepted-domain extremes.

### 4.6 Implementation and computational scope

The command-line implementation exposes prediction, screening, uncertainty propagation, optimisation and solver comparison. The archived GTX 1660 benchmark reports approximately 19 ms for one state, 22.5 ms for a 48-state polar and 55.5 ms for a 6,144-state batch. The last value corresponds to about 0.009 ms per state only when amortised over that batch. Timing boundaries in the benchmark synchronise CUDA; training, acquisition, geometry processing and verification are not included. These figures are not a measured end-to-end speedup over direct XFOIL.

The current source audit additionally times five fixed candidates with 64 perturbation scenarios at about 3.6 seconds in the available environment. This small workload is not a benchmark of full production optimisation. The implementation uses NumPy, SciPy and scikit-learn [22–24], with explicit hashes for the model and evaluation context. Unit tests and reproducible source repair do not substitute for a regenerated scientific lineage.

## 5. Discussion and limitations

The positive and negative results address different questions. Fixed stored predictions reproduce, and ensemble disagreement ranks residual risk within historical calibration observations. Separately, selected actual coordinate files produce favourable nominal and completed-case mean XFOIL results. Neither finding repairs the invalid interpretation of a substantial fraction of the training geometry. The feature corpus and its geometric support must be regenerated before the surrogate can be evaluated as the intended physical map.

The present evidence is therefore narrower than a validated robust-design framework. It does not establish universal accepted-domain reliability, global optimisation, unconditional manufacturing-risk reduction or agreement with an independent flow model. Related CST-based surrogate and confidence-guided design methods already exist [4,5,27,29]. The potentially useful contribution is a transparent separation of prediction error, screening decisions, geometric provenance and selected solver evidence; its eventual strength depends on a corrected, prospectively evaluated study rather than on favourable percentages alone.

A defensible next scientific study requires a fixed geometry-admission and fitting policy, a rebuilt dataset and grouping, retraining and recalibration, and an untouched confirmation cohort. The production optimisation must use candidate-specific perturbations, a frozen agreement normalisation and unchanged acceptance thresholds. Manufacturing variability must either be measured or consistently described as prescribed geometric sensitivity.

Independent selected CFD remains unperformed. A useful comparison must match achieved lift, document transition and turbulence assumptions, check wall resolution and mesh/domain convergence, preserve adverse and failed cases, and assess moment and geometry constraints alongside drag. Ncrit is not a directly transferable CFD turbulence parameter. A small set of selected CFD calculations can test the numerical ranking but cannot establish a population CVaR95. Experiments would provide an additional level of physical evidence; neither the historical XFOIL labels nor a second XFOIL execution supplies it.

## 6. Conclusion

The historical ensemble's stored prediction metrics and selected local-XFOIL comparisons are reproducible, but a coordinate-order defect affects 1,001 stored aerofoil encodings and prevents those metrics from validating the repaired physical-geometry workflow. Source checks now address coordinate ordering, common-x thickness, reconstructed-CST validity, model integrity and stale optimisation evidence. Correct geometry-cluster separation preserves an incremental disagreement-ranking result within historical calibration data, with AUROC increasing from 0.768 to 0.867. The selected fixed candidate retains 18.7% lower nominal XFOIL weighted drag and lower completed-case means in two 0.5%-chord campaigns, while adverse small-sample tail behaviour and a worse current penalised mean objective are explicitly retained. A rebuilt and freshly evaluated ML lineage, a valid production optimisation and independent CFD are required before stronger aerodynamic-design claims are made.

## CRediT authorship contribution statement

Benjamin Ian George Gawith: Conceptualization, Methodology, Software, Data curation, Formal analysis, Writing – original draft. Ava Shahrokhi: Supervision, Conceptualization, Methodology, Writing – review & editing. Keivan Navaie: Supervision, Methodology, Writing – review & editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Declaration of generative AI use

AI-assisted tools were used for software engineering, reproducibility checks and manuscript drafting. Numerical findings are linked to executable code and archived evidence. Responsibility for scientific interpretation, final content and submission declarations rests with the authors; this revision does not imply completed coauthor approval.

## Data and code availability

Source code and compact review evidence are versioned separately from the large local model, source-data and solver archives. The frozen lineage is retained to make historical calculations inspectable and is not presented as repaired training data. The corrected parser, audit scripts and SHA-256 records permit reconstruction of the stated checks when the documented local archives are available. Third-party data redistribution remains subject to the original source terms. Public availability of every large training or solver artifact is not claimed.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Figures

**Figure 1.** Historical encoded-feature held-out prediction results for (a) lift, (b) log-drag and (c) moment. The geometry-preprocessing defect limits physical interpretation; these are not repaired-model validation results.

![Figure 1](figures/fig_surrogate_accuracy.png)

**Figure 2.** Historical penalised surrogate objective and screening exits against prescribed amplitude. Neither tail values nor exits are measured manufacturing probabilities.

![Figure 2](figures/fig_manufacturing_risk.png)

**Figure 3.** Historical admission under three perturbation generators at 1% chord. The original refit rule differs from the current reconstructed-geometry validation.

![Figure 3](figures/fig_perturbation_validity.png)

**Figure 4.** Historical schema-4, ten-seed low-budget front. Its distance of 0.149 uses post-hoc pooled scales; it is not current production acceptance or proof of Pareto optimality.

![Figure 4](figures/fig_pareto_front.png)

**Figure 5.** Fixed-geometry XFOIL comparison using each design's own completed 1%-chord scenarios. The selected candidate's adverse marginal maximum is retained; no precise population-tail estimate is implied.

![Figure 5](figures/fig_design_verification.png)

**Figure 6.** Historical finite adversarial search. Support and disagreement statistics are descriptive, workflow counts overlap and only one probe is wholly accepted.

![Figure 6](figures/fig_surrogate_trust_stress.png)

**Figure 7.** Selected eligible converged adversarial record with the largest mean discrepancy in that subset. Its mean/maximum gap is approximately 0.52/0.61, not the overall 0.67 discrepancy record.

![Figure 7](figures/fig_worstcase_polar.png)

**Supplementary Figure S1.** Retrospective per-aerofoil macro error against retained historical test states using calibration-scaled support and disagreement. No threshold is fitted to the test set; the plot is not prospective confirmation of a new screening policy.

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
