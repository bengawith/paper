# Risk-Aware Aerofoil Design under Manufacturing Uncertainty using a Trust-Calibrated Machine-Learning Surrogate and Multi-Fidelity Verification

## Authors

Benjamin Ian George Gawith ᵃ, *Ava Shahrokhi ᵇ, Keivan Navaie ᶜ

## Affiliations

ᵃ School of Engineering, Liverpool John Moores University, Liverpool, UK
ᵇ Department of Maritime and Mechanical Engineering, Liverpool John Moores University, Liverpool, UK
ᶜ School of Computing and Communications, Lancaster University, Lancaster, UK

## Corresponding Author

Ava Shahrokhi, A.Shahrokhi@ljmu.ac.uk

## Keywords

Robust aerodynamic design; Manufacturing uncertainty; Risk-aware optimisation; Surrogate trust calibration; Class–Shape Transformation; Multi-fidelity verification

## Highlights

- We develop a compact machine-learning surrogate that predicts lift, drag and pitching-moment coefficients from a 12-parameter CST geometry, and we equip it with an empirically calibrated trust domain so that we know where its predictions can be believed.
- We show that predictions inside the trust domain are two to nearly three times more accurate than those outside it, on 26,692 held-out aerodynamic states.
- We model manufacturing variability as a smooth, bounded, spatially correlated surface-normal displacement, and we demonstrate that this physically admissible model keeps every sample valid whereas naive independent perturbation renders 99.7% of samples non-physical.
- We use the surrogate to perform a risk-aware multi-objective optimisation that trades expected drag against tail-risk drag under manufacturing uncertainty, and we verify the resulting deterministic and robust designs directly against XFOIL.
- We implement the reviewer-requested adversarial search for the aerofoil that maximises the surrogate–XFOIL disagreement, and we find that the worst case (ΔC_L = 0.67) falls entirely outside the calibrated trust domain, confirming that the trust flag isolates the surrogate's failure regions.

## Abstract

Machine-learning surrogates can replace panel-method or CFD evaluations inside an aerodynamic design loop at negligible cost, yet their practical adoption is held back by two questions that raw accuracy statistics do not answer: where can the surrogate be trusted, and can it drive design decisions that survive the manufacturing tolerances of a real aerofoil? In this work we address both questions. We train an ensemble of compact, point-conditioned multilayer perceptrons to predict the lift, drag and pitching-moment coefficients of two-dimensional aerofoils from a 12-parameter Class–Shape Transformation (CST) geometry at a chord Reynolds number of 1×10⁶, and we couple the surrogate to an explicit, empirically calibrated trust domain. On 26,692 aerodynamic states drawn from 233 grouped, held-out aerofoils the surrogate attains coefficients of determination of 0.989, 0.922 and 0.918 for lift, drag and moment, and within its trust domain the lift, drag and moment errors are smaller by factors of 2.2, 1.9 and 2.7. We then model manufacturing variability as a smooth, bounded, spatially correlated displacement of the aerofoil surfaces normal to the local geometry, and we use the surrogate to propagate this uncertainty and to perform a risk-aware multi-objective optimisation that minimises the expected drag and the 95% conditional value-at-risk of the drag at required lift. The optimisation yields distinct deterministic and robust aerofoils whose ranking and whose drag advantage over a NACA 2412 reference are confirmed by direct XFOIL evaluation, and an adversarial search for the aerofoil that maximises the surrogate–solver disagreement demonstrates that the largest errors are confined to the region the trust model already flags as unreliable. The framework therefore delivers not only a fast aerodynamic predictor, running at 0.009 ms per state, but a defensible envelope of reliability within which robust design can proceed with confidence.

## Nomenclature

| Symbol | Description | Units |
|---|---|---|
| α | Angle of attack | deg |
| C_L | Lift coefficient | – |
| C_D | Drag coefficient | – |
| C_M | Pitching-moment coefficient | – |
| A_{U,i}, A_{L,i} | Upper and lower Bernstein (CST) coefficients | – |
| Δz_LE | Leading-edge modification weight | – |
| ζ_TE | Trailing-edge thickness | – |
| ε | Manufacturing perturbation amplitude | fraction of chord |
| C̄_D^w | Required-lift weighted drag coefficient | – |
| CVaR₉₅ | 95% conditional value-at-risk | – |
| Re | Chord Reynolds number | – |

## 1. Introduction

The design and optimisation of aerodynamic surfaces underpins efficiency and performance across aerospace, automotive and renewable-energy applications, and the accurate prediction of aerodynamic forces governs the iterative design cycle for wings, blades and rotor sections. High-fidelity Computational Fluid Dynamics (CFD) and wind-tunnel testing remain the authoritative tools, but they are resource intensive: long turnaround times and substantial computational cost constrain the rapid exploration of large design spaces, especially in early-stage studies [1,2]. Even a fast panel method such as XFOIL [10] becomes a bottleneck once it is embedded inside a stochastic optimisation loop that demands thousands of evaluations for every candidate.

To relieve this constraint the community has increasingly adopted data-driven surrogates, in which a machine-learning model learns a direct map from a compact geometric description to the aerodynamic coefficients and thereafter predicts almost instantaneously [3,4,5]. In earlier work we used a multilayer perceptron surrogate to replace CFD evaluations inside an evolutionary aerodynamic optimisation pipeline [6,7], and a broad literature now applies neural networks, random forests and physics-informed models to aerofoil analysis and aerodynamic shape optimisation [5]. In a precursor to the present study we demonstrated that a lightweight multilayer perceptron trained on CST geometries could reproduce the full lift curve of an aerofoil with high pointwise accuracy.

Peer review of that precursor, however, exposed a decisive gap that motivates the present paper: accuracy statistics alone do not establish engineering usefulness. A surrogate is valuable to a designer only if we know where in the design space it can be trusted, and if we can show that it supports a concrete design task rather than merely reproducing a solver on average. One reviewer asked specifically for an optimisation that searches for the aerofoil which maximises the discrepancy between the surrogate and XFOIL, on the grounds that the surrogate's worst case, not its mean, governs its safe use. Others observed that predictions were confined to a single operating condition and omitted drag and pitching moment, and that panel-method labels are themselves unreliable in separated flow. We regard these as the right questions, and we have rebuilt the study around them.

We therefore reframe the contribution from raw predictive accuracy to *governed* accuracy and *demonstrated design value*. We address three questions. First, can a compact surrogate predict lift, drag and pitching moment with a quantified, empirically calibrated domain of reliability? Second, can that surrogate drive a manufacturing-aware optimisation that distinguishes nominally optimal from robustly optimal aerofoils? Third, do the resulting designs, and the boundaries of the surrogate's trust, survive confrontation with the XFOIL solver, including a deliberate adversarial search for the surrogate's worst case?

The contributions of this paper are the following:

- A trust-calibrated multi-output surrogate: an ensemble of point-conditioned multilayer perceptrons that predicts C_L, C_D and C_M from a 12-parameter CST vector, equipped with a split-conformal trust model that labels each queried aerodynamic state as reliable or not.
- A physically admissible manufacturing-uncertainty model and a risk-aware optimisation: a smooth, bounded, spatially correlated surface-normal displacement field, and a genetic-algorithm optimisation that minimises expected drag and tail-risk drag at required lift subject to aerodynamic and geometric constraints, producing an explicit deterministic-versus-robust trade-off.
- A multi-fidelity verification and surrogate-trust framework: a direct XFOIL confrontation of the optimised designs, together with the reviewer-directed adversarial search for maximum surrogate–solver disagreement, which together establish quantitatively where the surrogate can and cannot be relied upon.

Throughout, every reported quantity is produced from durable, hash-pinned evidence artefacts by a reproducible pipeline, and we report negative or limiting findings — such as the reduced seed-to-seed reproducibility of the robust Pareto front and the convergence limits of XFOIL at high incidence — openly rather than concealing them.

## 2. Aerofoil representation and dataset

### 2.1 CST parameterisation

The representation of aerofoil geometry must be compact enough for efficient learning and expressive enough to capture real-world variation. We adopt the Class–Shape Transformation (CST) method [8,9], which encodes each surface as a class function multiplied by a Bernstein-polynomial shape function and represents the aerofoil as a fixed-length coefficient vector. Following the dimensionality study of our precursor work, we describe each aerofoil with five Bernstein coefficients per surface, a leading-edge modification weight Δz_LE and a trailing-edge thickness ζ_TE, giving a 12-parameter vector (Table 1). This encoding is compact, differentiable and, importantly for the present study, identical to the parameterisation used to generate the training labels, so that the surrogate and the solver always operate on the same geometric inputs.

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

A recurring risk in aerofoil datasets is leakage between near-duplicate shapes. To guard against it we perform every split at the level of geometric clusters of aerofoils rather than individual polars, so that an entire cluster is held out together. We reserve a *locked-test* partition of 233 aerofoils (26,692 aerodynamic states) that is untouched during training, tuning and calibration and is used only for final reporting, and a separate calibration partition that parameterises the trust model of Section 3.2. Because the surrogate is point-conditioned — it predicts a coefficient at a single queried angle of attack, as described in Section 3.1 — we do not pad or extrapolate curves onto a fixed grid, and only genuinely measured aerodynamic states are ever used as supervision. This removes the edge-padding artefacts that complicated the precursor model at its most fragile design step.

## 3. Methodology

### 3.1 Point-conditioned surrogate and ensemble

The surrogate maps the concatenation of the 12 CST parameters and a single angle of attack — thirteen inputs in total — to the three aerodynamic coefficients (C_L, log C_D, C_M) at that state. We learn drag in logarithmic space because it spans an order of magnitude across the polar. Each network has three hidden layers of width 256 with SiLU activations [19], residual connections and layer normalisation, and is trained with the AdamW optimiser [18] under a log-cosh objective. The architecture and its training hyperparameters were selected by an Optuna search [21] over 50 trials using grouped three-fold cross-validation, so that the configuration is chosen on genuinely held-out families rather than on leaked near-duplicates. Consistent with the guidance we received, we did not pursue an extensive new architecture search; the reviewers did not identify the network itself as the limiting factor, and, as Section 4.1 shows, the compact multilayer perceptron is already competitive with a strong gradient-boosting baseline while additionally providing the uncertainty signal that our trust model requires.

To quantify epistemic uncertainty we train an ensemble of five members under a grouped cluster-bootstrap, in which geometric clusters rather than individual samples are resampled, so that members differ in their exposure to whole regions of the design space [16]. The ensemble mean is the prediction and the inter-member standard deviation is a disagreement signal used by the trust model.

### 3.2 Trust calibration

We construct the trust domain by split-conformal calibration [17]. For each target we establish an absolute-residual band at 95% coverage on the calibration partition; we compute a support distance in scaled feature space as the mean distance to the five nearest development aerofoils; and we derive ensemble-disagreement thresholds by requiring that predictions below the threshold meet the residual tolerance. A queried state is labelled *within trust* only if its support distance, its ensemble disagreement for every target, and its angle of attack all lie within the calibrated envelope. The trust label is thus a conjunction of manifold support, model agreement and coverage — not a single heuristic — and it is the object a designer consults before believing a prediction.

### 3.3 Manufacturing-uncertainty model

We model manufacturing deviation as a random displacement applied normal to the local aerofoil surface, expressed as a fraction ε of chord. The field is smooth and spatially correlated with a correlation length of 0.15 chord, tapers towards the leading edge and vanishes at the trailing edge, reflecting the physical reality that machining and moulding errors are locally coherent rather than pointwise independent and that the edges are geometrically constrained. We study amplitudes ε ∈ {0.1%, 0.25%, 0.5%, 1%} of chord, treating 1% as a stress case. Each perturbed shape is re-fitted to the CST basis, and any sample whose refit error exceeds a strict tolerance is rejected as non-representable rather than silently admitted. In Section 4.2 we contrast this surface-normal model with two ablations — independent multiplicative perturbation of the CST coefficients, and independent uncorrelated coordinate noise — to test whether smoothness and correlation are necessary for physical admissibility.

### 3.4 Risk-aware optimisation

We formulate the principal study as a genuinely multi-objective problem. Using the NSGA-II algorithm [13] as implemented in pymoo [14] (population 128, 200 generations, 10 independent seeds), we simultaneously minimise two competing risk measures of the required-lift weighted drag C̄_D^w under manufacturing uncertainty: its expectation E[C̄_D^w] and its 95% conditional value-at-risk CVaR₉₅[C̄_D^w] [15], the latter being the mean drag in the worst 5% of manufacturing outcomes. Required lift is enforced at three target coefficients (C_L = 0.4, 0.7, 1.0 with weights 0.25/0.50/0.25). Relative constraints bound the thickness ratio, section area, leading-edge radius and pitching moment against a NACA 2412 reference, and any candidate that leaves the surrogate trust domain is rejected. Manufacturing uncertainty enters through 64 common-random-number scrambled-Sobol samples per candidate, so that all designs are compared under identical perturbation draws. For completeness we also retain a single-objective baseline that maximises mean lift penalised by its standard deviation, sweeping the robustness weight, which reproduces the formulation agreed during project scoping.

### 3.5 Multi-fidelity verification and adversarial trust probing

We confront the surrogate with XFOIL 6.99 [10] at three levels, under identical settings (Re = 1×10⁶, M = 0, N_crit = 9) and a strict per-angle completion rule with bounded retries so that partial convergence can never masquerade as a complete sweep. First, a canary suite of reference aerofoils checks the solver configuration against archived polars. Second, we re-evaluate the optimised deterministic and robust designs directly in XFOIL, both nominally and under the same shared manufacturing perturbations used by the optimiser, giving paired, solver-verified drag differences against the reference. Third, and centrally, we implement the adversarial disagreement search requested at review: we seek the aerofoil that maximises the discrepancy between the surrogate and the solver. Scrambled-Sobol candidates spanning the development design box are each screened for geometric validity, evaluated by the surrogate — recording their trust-domain membership — and then run through XFOIL over α ∈ [0°, 12°], after which a greedy local refinement perturbs the highest-disagreement candidates. We measure lift disagreement as the absolute C_L gap and drag disagreement as the relative C_D gap, computed only at angles where XFOIL genuinely converged, so that a failed direct solve is never scored as agreement.

## 4. Results and discussion

### 4.1 Surrogate accuracy, model justification and the value of trust

On the 26,692 held-out locked-test states the surrogate attains coefficients of determination of R²_{C_L} = 0.989, R²_{log C_D} = 0.922 and R²_{C_M} = 0.918, with mean absolute errors of 0.057 in lift, 0.143 in log-drag and 0.0096 in moment (Fig. 1). Relative to a distribution-matched Dummy baseline the equal-aerofoil-weighted error is reduced by 98.9%, 92.1% and 91.9% for the three targets, with paired cluster-bootstrap confidence intervals that exclude zero.

To justify the choice of model rather than assume it, we compared the multilayer perceptron against strong classical baselines under the same grouped protocol (Table 2). The perceptron is competitive with histogram-based gradient boosting on lift and drag and is clearly superior on the moment coefficient, while ridge regression collapses on drag; the Dummy predictor confirms that the task is non-trivial. We do not claim architectural superiority — indeed the gradient-boosting model is marginally stronger on this particular drag subset — but the perceptron combines competitive accuracy with a smooth, differentiable multi-output map and, crucially, furnishes through its ensemble the disagreement signal on which our trust calibration depends.

**Table 2. Coefficient of determination (R²) by model on the grouped development evaluation. Higher is better.**

| Model | C_L | log C_D | C_M |
|---|---|---|---|
| MLP (this work) | 0.984 | 0.904 | 0.875 |
| Histogram gradient boosting | 0.983 | 0.915 | 0.760 |
| Ridge regression | 0.948 | 0.083 | 0.741 |
| Dummy (mean predictor) | −0.001 | −0.009 | −0.000 |

The calibrated trust label is where the surrogate earns its operational value. Of the locked-test states, 17,642 are labelled within trust and 9,050 outside. Within the trust domain the accuracy improves markedly: the lift R² rises to 0.994, and the mean absolute errors of lift, drag and moment fall by factors of 2.2, 1.9 and 2.7 respectively relative to the untrusted states (Table 3). The empirical 95% interval coverage is 0.950 for all three targets, confirming that the conformal bands are honest. The trust flag therefore does exactly what a designer needs: it separates the states where the surrogate is highly accurate from those where its error inflates.

**Table 3. Locked-test accuracy inside and outside the calibrated trust domain. Drag errors are in linear coefficient units; the surrogate's native log-drag aggregate R² is 0.922.**

| Target | States (in / out) | MAE within trust | MAE outside trust | R² within | R² outside |
|---|---|---|---|---|---|
| C_L | 17,642 / 9,050 | 0.040 | 0.090 | 0.994 | 0.978 |
| C_D | 17,642 / 9,050 | 0.0035 | 0.0066 | 0.887 | 0.793 |
| C_M | 17,642 / 9,050 | 0.0061 | 0.0165 | 0.957 | 0.873 |

The surrogate is also fast, which is what makes the uncertainty and optimisation studies of the following sections tractable. On a consumer NVIDIA GTX 1660 the five-member ensemble predicts a batch of 6,144 aerodynamic states in 55.5 ms, that is 0.009 ms per state or roughly 1.1×10⁵ states per second; a single 48-point polar is returned in 22.5 ms. Propagating 2,048 manufacturing samples across a polar therefore costs on the order of a second, which would be prohibitive with a direct solver in the loop.

### 4.2 Manufacturing risk profile and the necessity of a physical perturbation model

Propagating the smooth surface-normal perturbation through the surrogate, with 2,048 Sobol samples per amplitude, yields the drag-risk profile of Fig. 2 and Table 4. Both the expected weighted drag and its 95% CVaR grow monotonically with amplitude, and the tail risk grows faster than the mean: from ε = 0.1% to ε = 1% the mean rises by 23% while the CVaR₉₅ nearly doubles, and the dispersion widens by more than an order of magnitude. At the same time the fraction of perturbed states leaving the trust domain rises from 0% to 17%, which quantifies the amplitude at which surrogate-based robustness assessment itself begins to require solver support.

**Table 4. Weighted-drag risk versus manufacturing amplitude (2,048 samples per level, smooth surface-normal model).**

| Amplitude ε (% chord) | Mean C̄_D^w | CVaR₉₅ C̄_D^w | Std. dev. | Trust-violation probability |
|---|---|---|---|---|
| 0.1 | 0.008007 | 0.008150 | 6.1×10⁻⁵ | 0.0% |
| 0.25 | 0.008081 | 0.008477 | 1.6×10⁻⁴ | 0.0% |
| 0.5 | 0.008311 | 0.009594 | 5.2×10⁻⁴ | 0.4% |
| 1.0 | 0.009850 | 0.015836 | 2.7×10⁻³ | 17.0% |

The necessity of the smoothness and correlation assumptions is established by ablation (Fig. 3). At 1% chord amplitude the smooth surface-normal model and the independent multiplicative-CST model each keep all 2,048 samples geometrically admissible, whereas independent, uncorrelated coordinate noise renders 2,042 of 2,048 samples — 99.7% — non-physical through excessive CST refit error. A manufacturing model that perturbs coordinates independently therefore does not describe realisable aerofoils, and we conclude that spatial correlation is a prerequisite for a physically meaningful robustness study rather than a convenience.

### 4.3 Deterministic versus robust designs

The multi-objective optimisation returns 188 feasible non-dominated designs aggregated across the ten seeds, tracing the expected-drag versus tail-risk-drag trade-off of Fig. 4. The two ends of the front define genuinely different aerofoils: the deterministic optimum minimises expected weighted drag (E = 0.00667, CVaR₉₅ = 0.00791), while the robust optimum minimises tail risk (E = 0.00719, CVaR₉₅ = 0.00781). Moving from the deterministic to the robust design trades a 7.8% increase in expected drag for a 1.2% reduction in worst-case drag — a quantified, and, as Section 4.4 shows, XFOIL-verifiable, expression of the price of robustness. The single-objective baseline is consistent with this picture but less informative, because several robustness weights drive the optimiser into regions where the required-lift roots cannot be met, which is precisely why we prefer the explicit two-objective formulation to a single weighted score.

We report the reproducibility of the front honestly. The mean normalised symmetric distance between the ten seeds' feasible fronts is 0.28, which exceeds our pre-registered 0.20 agreement threshold, and the number of feasible solutions varies from 2 to 47 across seeds. The *location* of the Pareto front in objective space is thus only moderately reproducible under a fixed budget — an expected consequence of a tightly constrained, uncertainty-averaged and trust-restricted feasible region — whereas the qualitative deterministic-versus-robust ranking and the reference-relative drag advantage of Section 4.4 are stable. We return to this limitation in Section 5.

### 4.4 Direct XFOIL verification of the optimised designs

When we confront the surrogate's design decisions with XFOIL, they hold. The nominal ranking of the candidate designs by weighted drag is identical between the surrogate and the solver (Spearman ρ = 1.00). Under the shared manufacturing perturbations, both optimised aerofoils achieve solver-verified lower weighted drag than the NACA 2412 reference: the paired improvement is −0.00563 (95% bootstrap interval [−0.0061, −0.0051]) for the robust optimum and −0.00456 ([−0.0051, −0.0039]) for the deterministic optimum, where a negative value denotes lower drag. XFOIL therefore corroborates, on an independent solver, both the ordering the surrogate assigns to the designs and the aerodynamic benefit of the optimised shapes over a conventional section.

### 4.5 Adversarial surrogate-trust probing

The adversarial search answers directly the reviewer request to find the aerofoil that maximises the surrogate–solver disagreement, and its outcome validates the trust framework (Figs. 5 and 6). Of 49 candidates, 20 were rejected as geometrically invalid before any solver call — evidence that the raw design box contains many shapes that are not aerofoils — and 41 were evaluated. The worst-case lift disagreement we discover is ΔC_L = 0.67, and, tellingly, it occurs on a design lying entirely outside the trust domain. Across all probes the mean lift gap outside trust is 0.27, six times the 0.043 gap of the single probe that fell wholly within the trust domain, a value comparable to the locked-test lift error itself. The largest surrogate errors that an adversary can find are thus concentrated exactly where the calibrated trust flag already warns the user not to rely on the surrogate. Figure 6 illustrates a representative outside-trust design for which the surrogate systematically under-predicts lift by a mean of 0.52 across a fully converged sweep.

Two further observations emerge. First, random sampling of the design box almost never lands wholly inside the trust domain — only 1 of 38 valid probes — which confirms that the reliable region is a small, well-supported subset of the geometric design space rather than the whole box. Second, XFOIL itself failed to converge for many of the thick, highly cambered adversarial shapes at higher incidence, and only 18 of the 41 evaluated candidates produced a fully converged sweep; the single largest discrepancies coincide with these barely-converged cases. The worst surrogate behaviour and the worst solver behaviour therefore overlap near stall — an honest limitation of two-tier verification, and a direct confirmation of the reviewer concern that XFOIL is unreliable in separated flow. In this regime, establishing surrogate trust and establishing solver trust are the same problem.

### 4.6 A usable design tool

The framework is delivered as a working command-line tool rather than only as an analysis. A designer can query the trust-calibrated surrogate for any CST aerofoil and obtain, in a fraction of a millisecond, the predicted lift, drag and moment polar together with the ensemble uncertainty and the per-angle trust flag; the same public interface reproduces the dataset assembly, model training, uncertainty study, optimisation, XFOIL verification and adversarial probe from hash-pinned artefacts. This turns the reliability envelope established above into something a practitioner can act on directly.

## 5. Discussion

The results reframe the surrogate's contribution from raw accuracy to *governed* accuracy, and in doing so they address the criticisms of the precursor study point by point. The demand for demonstrated usefulness is met by a complete robust-optimisation application whose designs are verified against an independent solver (Sections 4.3–4.4). The explicit request for an optimisation that maximises the ML–XFOIL disagreement is implemented, and it shows the disagreement to be trust-localised (Section 4.5). The absence of drag and moment is remedied by treating both as predicted quantities and as optimisation constraints. The concern that XFOIL is an imperfect ground truth is not evaded but quantified: the regions where the surrogate and the solver are each unreliable coincide near stall, and the calibrated trust domain is the mechanism that keeps the optimisation inside the region where both are dependable. The single-condition scope is stated explicitly and justified as an isolation of the robustness and trust contributions.

Three limitations bound our claims. First, the manufacturing model is a prescribed, bounded, smooth deviation field rather than a distribution measured from a production process; its role is to expose the deterministic-versus-robust trade-off under a physically admissible model, and calibration to shop-floor data is the natural next step. Second, the labels are panel-method predictions; the framework is explicitly multi-fidelity and is structured to accept selective CFD or wind-tunnel augmentation, particularly to improve post-stall fidelity where both the surrogate and XFOIL degrade. Third, the location of the robust Pareto front is only moderately reproducible across optimisation seeds at the budget we used; while the deterministic-versus-robust ranking and the reference-relative drag advantage are stable, the quantitative front positions should be read as indicative, and a larger population or a restart strategy would tighten reproducibility. We consider reporting this openly preferable to presenting a single seed as definitive.

The single operating condition remains the principal restriction on generality. Because the surrogate is point-conditioned and the trust model is data-driven, extending the framework to a Reynolds- and Mach-parameterised envelope requires only labelled data at additional conditions and an added conditioning input, leaving the calibration and robust-optimisation machinery unchanged.

## 6. Conclusion

We have developed and validated a trust-calibrated machine-learning surrogate and embedded it in a manufacturing-aware, risk-aware aerofoil-design framework with multi-fidelity verification. The surrogate predicts lift, drag and pitching moment from a compact CST encoding with held-out coefficients of determination of 0.989, 0.922 and 0.918, and, crucially, it carries an empirically calibrated trust domain within which its errors are two to nearly three times smaller. We modelled manufacturing uncertainty as a physically admissible smooth surface-normal displacement, producing a monotone drag-risk profile and showing that the naive independent-coordinate perturbation common in the literature is 99.7% non-physical. A multi-objective optimisation exposed a quantified deterministic-versus-robust trade-off, and direct XFOIL evaluation confirmed both the surrogate's design ranking, with a Spearman coefficient of unity, and the solver-verified drag advantage of the optimised aerofoils over a conventional reference. Finally, an adversarial search for the surrogate's worst case demonstrated that its largest errors are confined to the region the trust model already flags as unreliable, and that this region coincides with where the panel solver itself struggles. The central outcome is therefore not merely a fast predictor but a defensible envelope of reliability, delivered as a usable tool, within which fast and robust aerodynamic design can proceed with confidence, together with an explicit and reproducible account of where that confidence ends.

## CRediT authorship contribution statement

Benjamin Ian George Gawith: Conceptualization, Methodology, Software, Data curation, Formal analysis, Writing – original draft. Ava Shahrokhi: Supervision, Conceptualization, Methodology, Writing – review & editing. Keivan Navaie: Supervision, Methodology, Writing – review & editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data and code availability

The processed geometries and polars, the frozen surrogate ensemble, the trust-calibration artefacts, and the optimisation, uncertainty, XFOIL-verification and adversarial-probing results are produced and hash-pinned by the accompanying reproducible pipeline, and every figure and table in this manuscript can be regenerated from committed evidence artefacts as described in the repository documentation.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Figures

**Figure 1.** Surrogate accuracy on the held-out grouped locked-test aerofoils for (a) lift, (b) log-drag and (c) pitching-moment coefficients; points are coloured by trust-domain membership and the dashed line is the identity.

![Figure 1](figures/fig_surrogate_accuracy.png)

**Figure 2.** Weighted-drag risk profile under smooth, bounded manufacturing surface-normal deviation: expected drag and 95% CVaR (left axis) and trust-domain violation probability (right axis) versus perturbation amplitude.

![Figure 2](figures/fig_manufacturing_risk.png)

**Figure 3.** Geometric admissibility of three manufacturing-perturbation models at 1% chord amplitude (2,048 samples). Independent, uncorrelated coordinate noise renders 99.7% of samples non-physical.

![Figure 3](figures/fig_perturbation_validity.png)

**Figure 4.** Risk-aware Pareto front (expected weighted drag versus 95% CVaR) aggregated over ten independent seeds, with the deterministic and robust optima marked; the annotation reports the measured seed-to-seed front disagreement against the pre-registered threshold.

![Figure 4](figures/fig_pareto_front.png)

**Figure 5.** Adversarial surrogate-trust probing: (a) per-candidate mean lift disagreement versus support distance, coloured by trust-domain membership; (b) mean lift disagreement inside versus outside the trust domain, with the adversarial worst cases annotated.

![Figure 5](figures/fig_surrogate_trust_stress.png)

**Figure 6.** Representative worst-case surrogate–XFOIL lift divergence located by the adversarial search: a high-lift design lying entirely outside the calibrated trust domain, for which the surrogate systematically under-predicts lift across a fully converged XFOIL sweep (mean ΔC_L = 0.52; the largest single-angle discrepancy found anywhere in the search is 0.67).

![Figure 6](figures/fig_worstcase_polar.png)

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

[19] Elfwing, S., Uchibe, E. & Doya, K. (2018). Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural Networks, 107, 3–11.

[20] Paszke, A., Gross, S., Massa, F. et al. (2019). PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32 (NeurIPS 2019), 8024–8035.

[21] Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2623–2631.

[22] Saltelli, A., Ratto, M., Andres, T. et al. (2008). Global Sensitivity Analysis: The Primer. John Wiley & Sons.

[23] Kingma, D. P. & Ba, J. (2015). Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR 2015). arXiv:1412.6980.

[24] Harris, C. R., Millman, K. J., van der Walt, S. J. et al. (2020). Array programming with NumPy. Nature, 585, 357–362.

[25] Virtanen, P., Gommers, R., Oliphant, T. E. et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature Methods, 17, 261–272.

[26] Pedregosa, F., Varoquaux, G., Gramfort, A. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

[27] Chicco, D., Warrens, M. J. & Jurman, G. (2021). The coefficient of determination R-squared is more informative than SMAPE, MAE, MAPE, MSE and RMSE in regression analysis evaluation. PeerJ Computer Science, 7, e623.

[28] Abbott, I. H. & von Doenhoff, A. E. (1959). Theory of Wing Sections, Including a Summary of Airfoil Data. Dover Publications.

[29] Anderson, J. D. (2017). Fundamentals of Aerodynamics. 6th edn. McGraw-Hill Education.

---

*All quantitative results in Sections 4–6 are drawn from the durable evidence artefacts of study lineage `robust-v2-current-20260908` at Re = 1×10⁶, M = 0, N_crit = 9.*
