# Trust-Calibrated Machine-Learning Surrogates for Risk-Aware Aerofoil Design under Manufacturing Uncertainty, with Direct XFOIL Verification

## Authors

Benjamin Ian George Gawith ᵃ, *Ava Shahrokhi ᵇ, Keivan Navaie ᶜ

## Affiliations

ᵃ School of Engineering, Liverpool John Moores University, Liverpool, UK
ᵇ Department of Maritime and Mechanical Engineering, Liverpool John Moores University, Liverpool, UK
ᶜ School of Computing and Communications, Lancaster University, Lancaster, UK

## Corresponding Author

Ava Shahrokhi, A.Shahrokhi@ljmu.ac.uk

## Keywords

Surrogate trust calibration; Robust aerodynamic design; Manufacturing uncertainty; Conditional value-at-risk; Class–Shape Transformation; Direct XFOIL numerical comparison

## Highlights

- A compact point-conditioned ML ensemble estimates lift, log-drag and pitching moment from CST geometry with an empirically assessed reliability screen.
- On 26,692 held-out XFOIL-labelled states, accepted predictions show higher empirical interval coverage and lower error than rejected predictions; 66.1% of states are accepted.
- A selected outside-screen surrogate optimum underperforms NACA 2412 in local XFOIL, whereas a selected within-screen design improves nominal and sampled mean drag.
- The within-screen design lowers nominal XFOIL weighted drag 18.7% and sampled mean drag about 8.8% over 13–14 completed sweeps; tail-risk improvement is not established.
- A finite adversarial search finds its largest observed ML–XFOIL disagreement outside the screen, although in-screen adversarial evidence is sparse.

## Abstract

Machine-learning surrogates accelerate aerodynamic design but may be exploited outside their training support. We develop a five-member, point-conditioned multilayer-perceptron ensemble predicting C_L, log C_D and C_M from 12 CST geometry parameters and angle of attack at Re=1×10^6, M=0 and Ncrit=9. On 26,692 held-out XFOIL-derived states from 233 grouped aerofoils, R² is 0.989 for C_L, 0.922 for log C_D and 0.918 for C_M. An empirical screening rule accepts 66.1% of states, where observed 95%-band coverage is 97.8%, 96.5% and 99.2%, respectively, and errors are smaller than outside the screen. We use a prescribed smooth, bounded, correlated surface-normal perturbation model to study mean and adverse-tail weighted drag at fixed lift. In separate local XFOIL re-evaluation, a selected within-screen design has 18.7% lower nominal drag and about 8.8% lower sample-mean drag than NACA 2412, while a selected outside-screen candidate is substantially worse. The 1%-chord verification includes only 13–14 fully completed perturbation sweeps per design; its 95% CVaR and resampling intervals are exploratory rather than reliable population tail estimates. A finite adversarial search also finds large disagreement outside the screen. These results support the practical value of explicit empirical surrogate-domain screening, while optimisation-seed variability, unmeasured manufacturing deviations, XFOIL-only validation and the single operating condition constrain the claim. Selected CFD remains to be performed.

## Nomenclature

| Symbol | Description | Units |
|---|---|---|
| α | Angle of attack | deg |
| C_L, C_D, C_M | Lift, drag, pitching-moment coefficients | – |
| A_{U,i}, A_{L,i} | Upper and lower Bernstein (CST) coefficients | – |
| Δz_LE | Leading-edge modification weight | – |
| ζ_TE | Trailing-edge thickness | – |
| ε | Manufacturing perturbation amplitude | fraction of chord |
| C̄_D^w | Required-lift weighted drag coefficient | – |
| CVaR₉₅ | 95% conditional value-at-risk (mean of the worst 5%) | – |
| d_s | Support distance from the development manifold | – |
| Re | Chord Reynolds number | – |

## 1. Introduction

The design and optimisation of aerodynamic surfaces underpins efficiency and performance across aerospace, automotive and renewable-energy applications, and the accurate prediction of aerodynamic forces governs the iterative design cycle for wings, blades and rotor sections. High-fidelity Computational Fluid Dynamics (CFD) and wind-tunnel testing remain the authoritative tools, but they are resource intensive: long turnaround times and substantial computational cost constrain the rapid exploration of large design spaces, especially in early-stage studies [1,2]. Even a fast panel method such as XFOIL [10] becomes a bottleneck once it is embedded inside a stochastic optimisation loop that demands thousands of evaluations for every candidate.

To relieve this constraint the community has increasingly adopted data-driven surrogates, in which a machine-learning model learns a direct map from a compact geometric description to the aerodynamic coefficients and thereafter predicts almost instantaneously [3,4,5]. In earlier work we used a multilayer perceptron surrogate to replace CFD evaluations inside an evolutionary aerodynamic optimisation pipeline [6,7], and a broad literature now applies neural networks, random forests and physics-informed models to aerofoil analysis and aerodynamic shape optimisation [5]. In a precursor to the present study we demonstrated that a lightweight multilayer perceptron trained on CST geometries could reproduce the lift curve of an aerofoil with high pointwise accuracy.

Assessment of the precursor showed that aggregate predictive accuracy alone cannot establish engineering value. Here we test whether an empirical reliability screen helps avoid extrapolation failures in selected design decisions. Additional drag/moment outputs and local XFOIL cross-checks respond to previous concerns without claiming independent physical validation.

We investigate accuracy with an empirical acceptance screen and test its effect on selected design outcomes. The contributions are:
- A grouped, point-conditioned multi-output surrogate with measured held-out error and acceptance coverage; this is not a formal validity certificate for adaptively selected new designs.
- A prescribed bounded correlated surface-deviation model with mean/upper-tail objective optimisation, including trust-exit penalties and explicit limitations.
- A local XFOIL numerical cross-check of three selected designs and a finite adversarial disagreement search; CFD and experimental validation are not yet available.

Recent surrogate-assisted robust airfoil optimisation [27] and work on feedback-driven distribution shift in conformal prediction [28] mean that neither robust surrogate optimisation nor calibration alone is novel. Our specific emphasis is the operational use and limits of an empirical acceptance rule in a verified design example.

- A trust-calibrated multi-output surrogate: an ensemble of point-conditioned multilayer perceptrons that predicts C_L, C_D and C_M from a 12-parameter CST vector, with a split-conformal trust model whose coverage we validate on genuinely held-out data.
- A physically admissible manufacturing-uncertainty model and a risk-aware optimisation: a smooth, bounded, spatially correlated surface-normal displacement field, and a genetic-algorithm optimisation that minimises the expected and tail-risk weighted drag at required lift subject to aerodynamic and geometric constraints.
- A selected-design local XFOIL comparison showing how empirical trust screening can avoid a concrete surrogate extrapolation failure, alongside a finite adversarial search that identifies a large outside-screen discrepancy; neither finding constitutes universal certification.

Throughout, every reported quantity is produced from durable, hash-pinned evidence artefacts by a reproducible pipeline, and we report limiting findings — such as the reduced seed-to-seed reproducibility of the robust front and the convergence limits of XFOIL at high incidence — openly rather than concealing them.

## 2. Aerofoil representation and dataset

### 2.1 CST parameterisation

The representation of aerofoil geometry must be compact enough for efficient learning and expressive enough to capture real-world variation. We adopt the Class–Shape Transformation (CST) method [8,9], which encodes each surface as a class function multiplied by a Bernstein-polynomial shape function and represents the aerofoil as a fixed-length coefficient vector. Following the dimensionality study of our precursor work, we describe each aerofoil with five Bernstein coefficients per surface, a leading-edge modification weight Δz_LE and a trailing-edge thickness ζ_TE, giving a 12-parameter vector (Table 1). This encoding is compact, differentiable and, importantly for the present study, used by the surrogate and reconstructed candidate geometries; the original source polars came from coordinate aerofoils, so CST fitting/reconstruction error remains a separate source of discrepancy.

**Table 1. The 12 CST parameters used as the surrogate input features.**

| Parameter(s) | Description | Count |
|---|---|---|
| A_{L,0} … A_{L,4} | Lower-surface Bernstein coefficients | 5 |
| A_{U,0} … A_{U,4} | Upper-surface Bernstein coefficients | 5 |
| Δz_LE | Leading-edge modification weight | 1 |
| ζ_TE | Trailing-edge thickness | 1 |
| | **Total** | **12** |

### 2.2 Data provenance and operating condition

The geometries and aerodynamic labels are drawn from public databases: aerofoil coordinates from the UIUC Airfoil Data Site [11] and the corresponding XFOIL polars from Airfoil Tools [12], computed at a chord Reynolds number Re = 1×10⁶, Mach number M = 0 and transition parameter N_crit = 9. The set spans conventional families such as NACA, Wortmann, Selig, Eppler and Clark Y as well as unconventional and bio-inspired shapes. We deliberately restrict the study to this single operating condition so that the manufacturing-robustness and trust-calibration contributions can be isolated without conflation with Reynolds- or Mach-scaling effects; this is a scoping decision, not an oversight, and Section 5 sets out the extension path. In contrast to the precursor study, drag and pitching moment are treated here as first-class predicted quantities rather than being discarded.

### 2.3 Grouped splitting and target construction

A recurring risk in aerofoil datasets is leakage between near-duplicate shapes. To guard against it we perform every split at the level of geometric clusters of aerofoils rather than individual polars, so that an entire cluster is held out together. We reserve a *locked-test* partition of 233 aerofoils (26,692 aerodynamic states) that is untouched during training, tuning and calibration and is used only for final reporting, and a separate calibration partition that parameterises the trust model of Section 3.2. Because the surrogate is point-conditioned — it predicts a coefficient at a single queried angle of attack, as described in Section 3.1 — we do not pad or extrapolate curves onto a fixed grid, and only genuine source solver-computed aerodynamic rows are used as supervision. This removes the edge-padding artefacts that complicated the precursor model at its most fragile design step.

## 3. Methodology

### 3.1 Point-conditioned surrogate and ensemble

The surrogate maps the concatenation of the 12 CST parameters and a single angle of attack — thirteen inputs in total — to the three aerodynamic coefficients (C_L, log C_D, C_M) at that state. We learn drag in logarithmic space because it spans an order of magnitude across the polar. The architecture and its training hyperparameters were selected by an Optuna search [21] over 50 trials using grouped three-fold cross-validation, so that the configuration is chosen on held-out geometry groups rather than on leaked near-duplicates. The selected network has two hidden layers of width 512 with rectified-linear activations [19], residual connections and layer normalisation, dropout of 0.12, and is trained with the AdamW optimiser [18] and a log-cosh objective. Consistent with the guidance we received at review, we did not pursue an extensive new architecture search; the reviewers did not identify the network itself as the limiting factor, and, as Section 4.1 shows, the compact multilayer perceptron is already competitive with a strong gradient-boosting baseline while also furnishing ensemble-disagreement diagnostics for a prospective, better-balanced screening method.

To examine model disagreement we train five ensemble members with a grouped cluster-bootstrap, resampling geometry clusters rather than individual states [16]. Their mean is the predictor; their standard deviation is a disagreement signal, not independently calibrated epistemic uncertainty. The saved 2026 screening rule also includes disagreement limits, but the component audit below finds those limits nonbinding beyond the support-distance restriction on the held-out set.

### 3.2 Trust calibration

We build target-wise absolute-residual bands on a separate calibration partition with a split-conformal-style quantile construction [17]. The frozen acceptance rule evaluates development-support distance, ensemble-disagreement limits and angle range using the same code path for prediction and optimisation. However, a direct decomposition of the locked-test decisions found that **all 9,050 rejected states failed the support-distance criterion**: 0 states were additionally rejected solely by disagreement or angle limits (CL and log-CD disagreement thresholds reject no states; moment spread exceeds its threshold in only six already unsupported states). The current acceptance performance is therefore primarily a **support-distance screen**, not evidence that ensemble disagreement made an independent selection contribution. The frozen model was not retuned using the evaluated test set. Ordinary conformal marginal coverage assumptions include exchangeability, and the conditional-on-acceptance and optimisation-induced shifts do not inherit formal 95% guarantees [28]. We use 'empirical screen', not 'certified reliability domain'.

### 3.3 Manufacturing-uncertainty model

We prescribe surface-normal variability bounded by ε times nominal chord. A smooth correlated field with modelled correlation length 0.15 chord tapers near the leading edge and vanishes at the trailing edge. These are modelling assumptions, not measured manufacturing tolerances. Amplitudes 0.1%, 0.25%, 0.5% and 1% of chord are considered, with 1% a stress case. Actual perturbed coordinates undergo geometry checks and CST refitting; invalid geometry, refit failure and trust exits must be accounted separately. Independent CST-coefficient and uncorrelated coordinate perturbations provide sensitivity comparisons, not validated alternative manufacturing processes.

### 3.4 Risk-aware optimisation

We conduct a two-objective NSGA-II study [13,14] (population 128, 200 generations, 10 seeds) minimising the mean and empirical CVaR95 of required-lift weighted drag at C_L=0.4/0.7/1.0, with weights 0.25/0.50/0.25. Relative constraints include thickness, section area, leading-edge radius, pitching moment and attainable operating angles. The same empirical trust screen used for reporting restricts candidate evaluation; this reduces unsupported extrapolation without guaranteeing accuracy. Each candidate is evaluated with 64 shared-latent scrambled Sobol scenarios. When a perturbed case leaves trust, the reported optimisation quantity includes penalties and must be distinguished from physical drag. A 64-scenario tail estimate serves ranking, not definitive 95%-tail inference. A lift-versus-sensitivity scalar baseline is retained.

### 3.5 Direct XFOIL numerical cross-check and finite adversarial probing

We re-evaluate selected designs using local XFOIL 6.99 at Re=1×10^6, M=0, Ncrit=9, accepting only finite positive-drag rows with clean solver exit and required-angle coverage. This is a separate numerical execution of an XFOIL-class method, not independent experimental or RANS evidence. NACA 2412 and two selected designs are assessed nominally and under shared perturbations; failed solves are not filled in. The 13–14 fully completed scenarios per design permit an exploratory paired mean comparison but cannot resolve a 95% tail distribution. A finite adversarial CST search with local refinement finds large observed CL disagreement on α=0–12 degrees. It does not prove a global maximum or bound within-screen error, especially because only one fully trusted adversarial candidate was recorded.

## 4. Results and discussion

### 4.1 Surrogate accuracy, model justification and the value of trust

On the 26,692 grouped held-out XFOIL-derived states, R² is 0.989 for C_L, 0.922 for log C_D and 0.918 for C_M; the corresponding MAEs are 0.057, 0.143 in log-drag and 0.0096 in moment (Fig. 1). These test outcomes quantify prediction fidelity on the sampled data distribution, not on arbitrary optimiser-generated geometries.

To justify the choice of model rather than assume it, we compared the multilayer perceptron against strong classical baselines under the same grouped protocol (Table 2). The perceptron is competitive with histogram-based gradient boosting on lift and drag and is clearly superior on the moment coefficient, while ridge regression collapses on drag; the Dummy predictor confirms that the task is non-trivial. We do not claim architectural superiority — indeed gradient boosting is marginally stronger on this particular drag subset — but the perceptron combines competitive accuracy with a smooth, differentiable multi-output map and, crucially, provides both a compact differentiable map and disagreement diagnostics, although the active frozen screening decision was predominantly determined by geometry/operating-point support.

**Table 2. Coefficient of determination (R²) by model on the grouped development evaluation. Higher is better.**

| Model | C_L | log C_D | C_M |
|---|---|---|---|
| MLP ensemble (this work) | 0.984 | 0.904 | 0.875 |
| Histogram gradient boosting | 0.983 | 0.915 | 0.760 |
| Ridge regression | 0.948 | 0.083 | 0.741 |
| Dummy (mean predictor) | −0.001 | −0.009 | −0.000 |

The empirical trust screen accepts 17,642 of 26,692 held-out states (66.1%) and rejects 9,050. Target-wise observed interval coverage is 93.7%, 94.4% and 94.5% across all states, versus 97.8%, 96.5% and 99.2% on the selected accepted subset. Accepted errors are lower by factors around 2.2, 1.9 and 2.7 for C_L, linear C_D and C_M (Table 3). These are descriptive conditional-on-selection observations, not a formal exchangeability-independent guarantee for arbitrary design optimisation.

**Table 3. Locked-test accuracy and held-out trust-band coverage, inside and outside the calibrated trust domain. Drag errors are in linear coefficient units.**

| Target | MAE within / outside | R² within / outside | Held-out 95%-band coverage, all / within trust |
|---|---|---|---|
| C_L | 0.040 / 0.090 | 0.994 / 0.978 | 0.937 / 0.978 |
| C_D | 0.0035 / 0.0066 | 0.887 / 0.793 | 0.944 / 0.965 |
| C_M | 0.0061 / 0.0165 | 0.957 / 0.873 | 0.945 / 0.992 |

To test the **utility of the existing uncertainty signals without altering the frozen model or tuning on the test set**, we also computed a pre-specified descriptive selective-prediction diagnostic (Supplementary Fig. S1). The risk score is the maximum of development-support distance and three ensemble-disagreement measures, each scaled by its 95th percentile on the separate calibration partition. On the locked-test observations, this score correlates positively with absolute prediction error (Spearman ρ = 0.476 for C_L, 0.452 for log C_D and 0.524 for C_M). Group-macro error versus retained coverage is shown in Supplementary Fig. S1. This is evidence that the frozen ensemble's support/disagreement signals have ranking value on the observed distribution; it is **not** a calibrated optimizer-shift guarantee, new post-test model-selection rule or proof that the surrogate has been improved by retraining.

On the recorded GTX 1660 benchmark, a 6,144-state batch requires 55.5 ms, equivalent to 0.009 ms per state when amortised across the batch. Measured single-state and 48-state-polar latencies are about 19 ms and 22.5 ms, respectively. These are distinct workload definitions; the batch metric is not a per-query latency or a measured full-workflow advantage over direct XFOIL including training, geometry and validation overhead.

### 4.2 Manufacturing risk and the necessity of a physical perturbation model

At each prescribed amplitude the surrogate evaluates 2,048 scenarios (Table 4). The tabulated quantities are penalised robust-objective values, not invariably pure physical drag. At 0.5% chord about 0.4% of cases leave the trust screen; at the 1%-chord stress case 17% exit, so the upper tail is materially influenced by penalties. These exits identify a support limitation, not a measured probability of physical manufacturing failure.

**Table 4. Robust weighted-drag objective and trust-domain exit versus manufacturing amplitude (2,048 samples per level, smooth surface-normal model).**

| Amplitude ε (% chord) | Expected C̄_D^w | CVaR₉₅ C̄_D^w | Samples leaving trust domain |
|---|---|---|---|
| 0.1 | 0.008007 | 0.008150 | 0.0% |
| 0.25 | 0.008081 | 0.008477 | 0.0% |
| 0.5 | 0.008311 | 0.009594 | 0.4% |
| 1.0 | 0.009850 | 0.015836 | 17.0% |

At 1%-chord amplitude, all 2,048 scenarios from the smooth correlated and multiplicative-CST generators pass the implemented geometry/refit admissibility check. Uncorrelated coordinate noise passes in 6/2,048 cases. This supports use of a smooth field compatible with the CST representation, but failure of a CST-refit threshold alone does not prove a real manufactured surface is physically impossible or that the chosen correlation model is representative of a specific process.

### 4.3 The deterministic–robust front and its seed reproducibility

The reported aggregate surrogate-side front has 188 feasible non-dominated entries, but its seed agreement does not satisfy the predeclared criterion: mean normalised symmetric distance 0.283 exceeds threshold 0.20, with 2–47 feasible solutions per seed (Fig. 4). The reason for the variability is not independently isolated, and global optimiser repeatability is not established. Consequently, the Pareto plot is exploratory; the following three-design XFOIL comparison is a separate, limited numerical check rather than validation of the entire front.

### 4.4 Direct XFOIL check of three selected designs

We compare three intentionally selected designs under local XFOIL (Fig. 5, Table 5): NACA 2412 (support distance 0.60), a within-screen risk-aware candidate (0.90), and an unrestricted outside-screen surrogate candidate (3.72); the support threshold is approximately 1.25. This study of selected examples is not a population estimate of the quality of all accepted or rejected designs.

The within-screen candidate has 18.7% lower nominal weighted drag than NACA 2412 (0.00605 versus 0.00745). For the prescribed 1%-chord perturbations, its marginal mean from 13 completed scenarios is 0.00761 versus 0.00835 for the reference's 13; these means use different successful scenario subsets (even though both have n=13) and show an approximate 8.8% reduction. The reported 10-pair mean improvement is +0.00103. Its bootstrap range [0.00058, 0.00142] is **descriptive**, not a guaranteed population confidence interval: scrambled Sobol outcomes are not IID and three reference and three within-screen perturbation sweeps failed. The empirical CVaR95 values (0.00989 for the candidate and 0.00909 for reference) each reduce to one observed maximum at this sample size; the candidate's value is about 8.9% higher, with tail uncertainty unresolved. The outside-screen candidate is substantially worse in this example, with 0.01396 nominal drag (about 87% above reference). Together these cases motivate the acceptance screen but do not prove universal surrogate reliability or physical manufacturing robustness.

**Table 5. Selected-design XFOIL numerical results (Re = 1×10⁶): 16 perturbation sweeps requested per design; 13 fully completed for the reference, 13 for the within-screen candidate, and 14 for the outside-screen diagnostic. Means use each design's completed cases; paired differences use 10 and 11 matched successful scenarios, respectively. Empirical CVaR95 is a single sample-maximum diagnostic at this size. Brackets are descriptive bootstrap resampling ranges, not guaranteed confidence intervals.**

| Design | Support d_s | Within trust | Nominal C̄_D^w | Expected C̄_D^w (mfg) | CVaR₉₅ (mfg) | Paired improvement vs reference |
|---|---|---|---|---|---|---|
| NACA 2412 reference | 0.60 | yes | 0.00745 | 0.00835 | 0.00909 | — |
| Robust design (within screen) | 0.90 | yes | 0.00605 | 0.00761 | 0.00989 | +0.00103 [0.00058, 0.00142] (10 pairs) |
| Surrogate optimum (outside screen) | 3.72 | no | 0.01396 | 0.01414 | 0.01475 | −0.00568 [−0.00602, −0.00537] (11 pairs) |

**Supplementary two-scramble XFOIL sensitivity at 0.5%-chord prescribed deviation.** To assess whether the direction of the selected-design comparison depends on one 16-point stress sample, we ran two separate 128-scenario scrambled-Sobol campaigns at 0.5% chord, retaining only full XFOIL sweeps with all required-lift roots available. Each uses the same latent scenarios across compared designs and the two scrambles are independently seeded. The reference/inside-screen comparison has 92 and 96 valid pairs; the paired mean weighted-drag reductions are 15.4% and 15.8%, respectively. Both samples also show lower empirical upper-tail drag for the selected candidate among **completed matched cases** (Table S1). However, incomplete XFOIL solutions are excluded—not imputed—and the complete-case selection is neither random nor independent of flow/geometric difficulty. These results strengthen the numerical same-solver evidence at 0.5% chord but do **not** establish unconditional manufacturing-risk probabilities or independent CFD accuracy.

**Table S1. Independent 0.5%-chord XFOIL scrambles; all means and empirical tail values are conditional on full solver completion and attainable required lift.**

| Independent scramble | Requested/reference usable/selected usable | Matched usable | Paired mean drag reduction | Candidate empirical CVaR95 change on matched cases |
|---|---:|---:|---:|---:|
| A | 128 / 105 / 110 | 92 | 15.4% | -12.4% |
| B | 128 / 115 / 107 | 96 | 15.8% | -11.7% |

The selected comparison illustrates one case in which screening helps avoid a surrogate-extrapolation failure. Because candidate selection is deliberate, numerical XFOIL checks share modelling assumptions with training labels, and CFD/experimental validation is absent, the result supports a case-specific design advantage, not proof that accepted candidates are always reliable or superior.

### 4.5 Adversarial surrogate-trust probing

The finite adversarial campaign reports overlapping workflow counts (49 requested, 41 evaluated, 20 geometry rejections and 18 fully converged sweeps); these categories are not an exhaustive disjoint partition. The largest observed lift discrepancy is ΔC_L=0.67, for a wholly outside-screen candidate. The observed average outside-screen gap is 0.27 versus 0.043 for only **one** fully trusted adversarial probe. This comparison is too small to claim that all worst errors are outside trust or to estimate within-screen worst-case risk. Many demanding shapes also cause XFOIL convergence problems, limiting what either surrogate or solver can establish near separation.

### 4.6 A usable design tool

The code provides a CLI-oriented pipeline for trust-annotated prediction, uncertainty studies, optimisation and XFOIL checks. The measured isolated single-state/polar latencies are about 19/22.5 ms and batched amortised throughput is 0.009 ms per state. The tool is a reproducible research implementation, not an independently CFD-validated aerodynamic design product or formal reliability certificate.

## 5. Discussion

The work demonstrates an observed error-versus-coverage trade-off from empirical screening, plus a limited three-design XFOIL illustration where one accepted candidate improves nominal and sampled mean drag and one rejected candidate performs poorly. This does not establish that all accepted designs are superior, nor that all XFOIL-separated-flow solutions are physically accurate. Related robust surrogate-assisted aerofoil optimisation already exists [27], so the contribution is the specific integration of screening, a transparent risk objective and design counterexample, not invention of robust optimisation.

The transferable lesson is that empirical surrogate screening can expose extrapolation risk and influence selected design decisions, but its conditional accepted-set accuracy cannot be elevated to a universal guarantee for optimiser-selected distributions [28]. This case motivates wider independent solver verification and reporting of accepted coverage, failures and computational cost.

Five limitations bound interpretation: (1) geometric deviations are a prescribed random-field model, not measured manufacturing tolerances; (2) both training labels and validation belong to an XFOIL-class solver family, with no independent CFD/wind-tunnel validation; (3) the 13–14-case perturbation comparison cannot establish a 95% tail distribution, and bootstrap intervals from scrambled Sobol samples are descriptive; (4) robust-front seed agreement fails (0.283 > 0.20); and (5) only one operating condition is studied, with solver and surrogate weakness near difficult separated-flow regimes. Selected CFD and larger independently scrambled perturbation experiments are required before stronger physical conclusions.

## 6. Conclusion

We present a compact multi-output ML aerodynamic surrogate with a measured held-out acceptance/error trade-off, coupled to an exploratory bounded geometric-uncertainty optimisation. On 26,692 grouped XFOIL-derived states, R² is 0.989 for C_L, 0.922 for log C_D and 0.918 for C_M. In a selected local XFOIL comparison, a within-screen candidate has 18.7% lower nominal drag and approximately 8.8% lower successful-case mean drag than NACA 2412, while an outside-screen candidate is substantially worse. This is useful evidence for screening against surrogate extrapolation, not a proof of universal correctness, verified 95%-tail superiority or independent physical robustness. The optimiser also fails its seed-repeatability threshold. The study provides a reproducible ML-and-engineering case and a clear path to CFD, larger perturbation campaigns and measured manufacturing uncertainty.

## CRediT authorship contribution statement

Benjamin Ian George Gawith: Conceptualization, Methodology, Software, Data curation, Formal analysis, Writing – original draft. Ava Shahrokhi: Supervision, Conceptualization, Methodology, Writing – review & editing. Keivan Navaie: Supervision, Methodology, Writing – review & editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Declaration of generative AI use

During the preparation of this work the authors used AI-assisted tooling to support software engineering, reproducibility checks and manuscript drafting. All scientific results derive from the authors' own executable pipeline, and the authors reviewed and verified all content and take full responsibility for it.

## Data and code availability

The repository and archived package contain source, compact evidence and a separate checksum-indexed local evidence/model archive; large training and solver artefacts are not all publicly committed. Reproduction requires those archived files and their documented environment. Public redistribution of third-party source data is subject to source terms and attribution.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Figures

**Figure 1.** Surrogate accuracy on the held-out grouped locked-test aerofoils for (a) lift, (b) log-drag and (c) pitching-moment coefficients; points are coloured by trust-domain membership and the dashed line is the identity.

![Figure 1](figures/fig_surrogate_accuracy.png)

**Figure 2.** Penalised robust-objective summary and trust exits versus prescribed amplitude; 0.5% includes about 0.4% trust exits and 1% about 17%, so upper-tail values are not pure aerodynamic drag.

![Figure 2](figures/fig_manufacturing_risk.png)

**Figure 3.** Geometry/CST-refit acceptance under three prescribed perturbation generators at 1% chord (2,048 samples). Uncorrelated coordinate noise fails this specific acceptance test in 99.7% of cases; this is not independent proof of physical impossibility.

![Figure 3](figures/fig_perturbation_validity.png)

**Figure 4.** Aggregated surrogate robust front (expected weighted drag versus 95% CVaR) over ten seeds; the annotation reports the measured seed-to-seed front disagreement, which exceeds the pre-registered threshold and is why the engineering claim rests on the XFOIL verification of Fig. 5 rather than on these surrogate-side values.

![Figure 4](figures/fig_pareto_front.png)

**Figure 5.** Selected-design local XFOIL comparison. Perturbation means use 13–14 completed sweeps and CVaR95 is effectively the largest sample, not a precise population-tail estimate.

![Figure 5](figures/fig_design_verification.png)

**Figure 6.** Finite adversarial discrepancy search by support distance and trust membership; only one fully trusted adversarial design was available, limiting the inside/outside comparison.

![Figure 6](figures/fig_surrogate_trust_stress.png)

**Figure 7.** Largest observed discrepancy in the finite tested search, outside the acceptance screen; no mathematically global worst-case claim is made.

![Figure 7](figures/fig_worstcase_polar.png)

**Supplementary Figure S1.** Descriptive accuracy-versus-retained-coverage curves for the frozen MLP ensemble on the historical locked-test partition. The risk score uses only calibration-partition quantiles for scaling, and the held-out subset is used for evaluation, not fitting. Each panel reports per-aerofoil macro MAE for C_L, log C_D and C_M.

![Supplementary Figure S1](figures/fig_ml_selective_coverage.png)

## References

[1] Drela, M. & Giles, M. B. (1987). Viscous–inviscid analysis of transonic and low Reynolds number airfoils. AIAA Journal, 25(10), 1347–1355.

[2] Vinuesa, R. & Brunton, S. L. (2022). Enhancing computational fluid dynamics with machine learning. Nature Computational Science, 2(6), 358–366.

[3] Brunton, S. L., Noack, B. R. & Koumoutsakos, P. (2020). Machine learning for fluid mechanics. Annual Review of Fluid Mechanics, 52, 477–508.

[4] Li, J., Bouhlel, M. A. & Martins, J. R. R. A. (2019). Data-based approach for fast airfoil analysis and optimization. AIAA Journal, 57(2), 581–596.

[5] Li, J., Du, X. & Martins, J. R. R. A. (2022). Machine learning in aerodynamic shape optimization. Progress in Aerospace Sciences, 134, 100849.

[6] Shahrokhi, A. & Jahangirian, A. (2010). A surrogate assisted evolutionary optimization method with application to the transonic airfoil design. Engineering Optimization, 42(6), 497–515.

[7] Jahangirian, A. & Shahrokhi, A. (2011). Aerodynamic shape optimization using efficient evolutionary algorithms and unstructured CFD solver. Computers & Fluids, 46(1), 270–276.

[8] Kulfan, B. M. (2008). Universal parametric geometry representation method. Journal of Aircraft, 45(1), 142–158.

[9] Kulfan, B. M. & Bussoletti, J. E. (2006). "Fundamental" parametric geometry representations for aircraft component shapes. In 11th AIAA/ISSMO Multidisciplinary Analysis and Optimization Conference, AIAA 2006-6948.

[10] Drela, M. (1989). XFOIL: An analysis and design system for low Reynolds number airfoils. In Low Reynolds Number Aerodynamics, Lecture Notes in Engineering, vol. 54, Springer, 1–12.

[11] Selig, M. S. (2010). UIUC Airfoil Data Site. University of Illinois at Urbana–Champaign. https://m-selig.ae.illinois.edu/ads/coord_database.html

[12] Airfoil Tools (2019). Airfoil Tools. http://airfoiltools.com/

[13] Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197.

[14] Blank, J. & Deb, K. (2020). pymoo: Multi-objective optimization in Python. IEEE Access, 8, 89497–89509.

[15] Rockafellar, R. T. & Uryasev, S. (2000). Optimization of conditional value-at-risk. Journal of Risk, 2(3), 21–41.

[16] Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems 30 (NIPS 2017), 6402–6413.

[17] Angelopoulos, A. N. & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511.

[18] Loshchilov, I. & Hutter, F. (2019). Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR 2019). arXiv:1711.05101.

[19] Nair, V. & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. In Proceedings of the 27th International Conference on Machine Learning (ICML 2010), 807–814.

[20] Paszke, A., Gross, S., Massa, F. et al. (2019). PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32 (NeurIPS 2019), 8024–8035.

[21] Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2623–2631.

[22] Harris, C. R., Millman, K. J., van der Walt, S. J. et al. (2020). Array programming with NumPy. Nature, 585, 357–362.

[23] Virtanen, P., Gommers, R., Oliphant, T. E. et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature Methods, 17, 261–272.

[24] Pedregosa, F., Varoquaux, G., Gramfort, A. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

[25] Abbott, I. H. & von Doenhoff, A. E. (1959). Theory of Wing Sections, Including a Summary of Airfoil Data. Dover Publications.

[26] Anderson, J. D. (2017). Fundamentals of Aerodynamics. 6th edn. McGraw-Hill Education.

[27] Pichitkul, A., Boksawat, A., Phiboon, T., Tantrairatn, S., Bureerat, S., Kasemsri, R. & Ariyarit, A. (2026). An intelligent multi-objective robust design optimization framework based on PathLine Transformation and surrogate-assisted sampling for uncertainty-aware airfoil design. Applications in Engineering Science, 25, 100310. https://doi.org/10.1016/j.apples.2026.100310

[28] Fannjiang, C., Bates, S., Angelopoulos, A. N., Listgarten, J. & Jordan, M. I. (2022). Conformal prediction under feedback covariate shift for biomolecular design. Proceedings of the National Academy of Sciences, 119(43), e2204569119. https://doi.org/10.1073/pnas.2204569119

---

*All quantitative results in Sections 4–6 are drawn from the durable evidence artefacts of study lineage `robust-v2-current-20260908` at Re = 1×10⁶, M = 0, N_crit = 9.*
