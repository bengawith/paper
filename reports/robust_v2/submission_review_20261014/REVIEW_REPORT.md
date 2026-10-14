# Robust V2 — independent manuscript and science review
## Author review edition, 14 October 2026

**Decision: READY FOR BEN'S SCIENTIFIC REVIEW AND AVA'S SELECTED CFD STAGE; NOT YET JOURNAL-SUBMISSION-READY.**

This is a corrective research review, not an acceptance guarantee. Its emphasis is to distinguish **verified model performance and selected solver comparisons** from general physical reliability. A negative CFD result, a changed performance ranking or a failure to match the mathematical target is a valid outcome that must be reported.

### 1. What was reviewed

- Primary repository: bengawith/paper, robust-v2-viability.
- Starting archived commit: 8b14165ce74396e0c504b1f165dd6eb248d51cae; clean initial local worktree and matching remote were observed.
- Initial OneDrive package: manuscript Markdown, seven figures, reviewer response, paper_data.json, source snapshot, frozen model/data evidence ZIP, Git bundle and SHA256 manifest.
- Both OneDrive package copies: 21/21 manifest file hashes and sizes matched; both ZIP CRC scans passed (2,798 source and 11,916 evidence entries).
- Historical summary in paper/REVIEW_STATUS.md was read and preserved, but current manuscript wording supersedes its more categorical assertions.
- Source scripts for design verification, strict XFOIL runner and publication-figure generation; 26,692-state aggregate evidence and updated direct-design summaries.
- New source literature checked on conformal prediction under optimisation-induced shift and a 2026 multi-objective surrogate-assisted robust aerofoil optimisation example.

### 2. Strengths worth retaining

1. The learned model predicts all three relevant targets from compact CST geometry/alpha, avoiding artificial fixed-grid padded V2 labels.
2. The held-out XFOIL-derived evaluation reports R² 0.989 for CL, 0.922 for log CD and 0.918 for CM across 26,692 states on grouped 233 aerofoils.
3. The empirical screen accepts 17,642/26,692 states (66.1%) and separates lower-error from higher-error cases. Held-out coverage on the accepted subset is 97.8%, 96.5% and 99.2% for CL/logCD/CM, but this is not a universal conditional-coverage guarantee.
4. The numerical design illustration contains an outside-screen surrogate failure and a within-screen candidate with lower nominal drag than NACA 2412 in a separate XFOIL execution.
5. The prescribed smooth geometric perturbation model is a useful engineering test case, when clearly differentiated from observed manufacturing tolerances.

### 3. Corrected strict-XFOIL verification

The previous verification script calculated design weighted drag from partial polar rows even when the full requested 0°–11° sweep did not satisfy strict solver completion. That was a **material evidence/engineering issue**, not merely editorial.

Fixed implementation:
- require XFOIL run.status == ok, return code zero and no timeout **before** accepting the weighted drag result;
- persist per-run strict run_status.json with valid-point coverage, endpoint span, clean exit, failed segments/retries and error reason;
- rerun all three nominal designs and 16 prescribed perturbations per design without overwriting the earlier frozen lineage;
- keep every failed scenario excluded and visible, never replace it with a nominal or fabricated value;
- include a regression test for a partial but target-lift-usable polar.

**Fresh strict results, local XFOIL 6.99, Re=1e6:**

| Design | Nominal weighted CD | Completed perturbations / requested | Successful-scenario mean CD | Largest observed CD |
|---|---:|---:|---:|---:|
| NACA 2412 reference | 0.007446 | 13 / 16 | 0.008349 | 0.009090 |
| Within-screen design | 0.006051 | 13 / 16 | 0.007615 | 0.009895 |
| Outside-screen diagnostic design | 0.013960 | 14 / 16 | 0.014137 | 0.014751 |

The nominal inside-screen improvement is **18.74%**. The separately computed successful-scenario means differ by **8.79%**, but they are conditioned on solver success. Only **10 matched reference/within-screen scenarios** yield a valid paired comparison; the paired mean reduction is **+0.001028** in weighted CD and the reported bootstrap resampling interval is approximately [+0.000580,+0.001417].

This interval is **descriptive**, not a finite-sample formal 95% confidence guarantee: Sobol draws are not IID and numerical-solver failures select which cases contribute. The empirical CVaR95 from 13 or 14 observations is effectively the **largest observed drag**. This maximum is 8.85% larger for the inside-screen candidate than for the reference. A beneficial nominal and selected-scenario mean **does not** establish beneficial worst-5% drag.

The old permissive 15–16-success results were preserved in the baseline paper-data file. The new canonical direct-design component is the strict rerun under results/robust_v2/submission_review_20261014/xfoil_strict/.

### 4. Scientific interpretation and manuscript edits

**Changes already made:**
- removed language claiming that screening mathematically certifies arbitrary optimiser-generated geometries;
- labelled empirical accepted-set coverage as descriptive, with dependence/exchangeability/feedback-shift caveats;
- differentiated a separate local execution of XFOIL from independent RANS or experiment;
- corrected XFOIL-derived rows from "measured" to "solver-computed";
- differentiated CST source geometry approximation from the original coordinate geometry used in source XFOIL polars;
- differentiated modelled smooth perturbation from measured production tolerances;
- labelled trust exits as **penalised robust objective** rather than always pure CD;
- differentiated failure of CST refit from proof that a manufactured surface is impossible;
- reported the optimiser seed-agreement failure (0.283 > declared 0.20);
- retained the small adversarial finding without extrapolating from its sole wholly trusted design to global safety;
- corrected 0.009 ms/state **batched throughput** versus ~19 ms single-state and ~22.5 ms 48-state polar latency;
- harmonised reviewer-letter title with the manuscript title and added explicit pre-CFD warnings;
- updated design-verification table/abstract/conclusion and regenerated seven figures at 300 dpi output with conservative labels;
- added 2026 prior-art and 2022 feedback covariate-shift references.

**Novelty position:** Related multi-objective robust optimisation using aerodynamic surrogates exists. This paper's defensible contribution is a measured trust acceptance/error trade-off that changes selected ML-assisted design decisions, plus an auditable solver counterexample—not the invention of multi-objective robustness, conformal prediction or XFOIL-driven optimisation.

**Limits of the trust statement:** Typical conformal marginal coverage assumes exchangeability. The accepted-set coverage from an additional trust filter does not automatically carry a finite-sample conditional guarantee; optimiser-generated designs may induce feedback covariate shift. "Certified", "necessary" and "proved reliable for any CST geometry" are not supportable by the current data.

### 5. Remaining blockers and severity

| Blocker | Severity | Exact resolution |
|---|---|---|
| No selected RANS CFD/experimental evidence | Critical for Ava's proposed final study | Run the separate CFD_CASE_MANIFEST cases through a properly checked viscous RANS solver, with reference/candidate mesh convergence and same-lift comparison. |
| Sparse tail-risk XFOIL campaign, failure-selected results | Critical for definitive manufacturing-risk claims | Increase independent scrambled replicate counts and direct solver coverage, or retain strictly exploratory phrasing; never claim population CVaR95 from 13 extrema. |
| Optimiser front seed-agreement failure (0.283 > 0.20) | Major | Diagnose optimisation/caching/feasibility variation, rerun under fixed inputs, or keep front exploratory rather than asserting global optimum. |
| Manufacturing tolerance distribution not measured | Major for manufacturing-specific relevance | Obtain tolerance metrology or explicitly retain "prescribed geometric variability" and present epsilon/correlation sensitivity without physical probability claims. |
| No externally independent aerodynamic truth | Major | Selected CFD now; matched experiment later if resources permit. |
| Trust conditional guarantees under optimizer shift unsupported | Interpretive limit | Report held-out empirical accepted coverage; do not claim certified design-domain coverage. |
| Journal fit/resubmission restriction unresolved | Editorial | Ava/Ben to choose journal and verify policies, AI-use/disclosure and figure formatting. |
| Confirmatory model test provenance, older initial-run environment | Documentation | Keep all frozen hashes/lineage; do not silently replace held-out provenance or claim complete historical retraining. |

### 6. What to examine during Ben's review

Read the revised abstract, Section 3.2 (what trust means), Section 3.3 (uncertainty model), Section 4.3 (repeatability), Section 4.4 and Table 5 (strict XFOIL results), Section 4.5 (adversarial counts), Discussion, Limitations and Conclusions.

Check that the scientific emphasis reflects the intended **ML + engineering** balance. The paper should not bury model performance or imply the numerical optimisation is a proof of physical design dominance.

Check Figure 5 in particular. Its upper-tail bars are **finite-sample maxima**, not a stable design CVaR comparison. The numerical values are valid within the strict accepted-case subset but cannot be sold as measured manufacturing reliability.

### 7. Hand-off to Ava

Ready-to-run CFD inputs and protocol are at:
reports/robust_v2/submission_review_20261014/CFD_HANDOFF/

Each exported geometry is accompanied by exact source-run status and hash, XFOIL polar, and preliminary alpha guesses for required lift. The seven-case selection comprises three nominal designs plus two **shared** perturbation scenarios for reference/inside-screen comparison.

Ava should first review the physical model/transition assumptions, then undertake a 3-mesh NACA/candidate comparison and same-lift CFD at CL 0.4/0.7/1.0. The final manuscript conclusion must be revised to reflect CFD results, including any ranking reversal.

### 8. Reproducibility and source commands

From the existing Windows repository:

    uv sync --extra dev --extra benchmarks
    uv run python scripts/robust_v2/review_claims_20261014.py
    uv run python scripts/robust_v2/audit_design_xfoil_coverage.py
    uv run python scripts/robust_v2/make_paper_figures.py
    uv run python scripts/robust_v2/prepare_cfd_handoff_20261014.py
    uv run ruff check src/robust_airfoil tests/robust_v2 scripts/robust_v2
    uv run mypy src/robust_airfoil
    uv run pytest tests/robust_v2 -m "not network and not xfoil and not slow" -q

The actual 14 October strict XFOIL rerun used a separate output path (the runner supports ROBUST_V2_XFOIL_OUT), retaining the original historical lineage. All new figures are regenerated from the consolidated paper data, whose direct-design component was updated to the strict run and points to its SHA. The original pre-promotion paper data remains in this review folder.

### 9. Claim-evidence status

- **Supported on held-out XFOIL-labelled data:** multi-output surrogate accuracy and empirical screening association with error.
- **Supported for three selected nominal XFOIL cases:** numerical nominal drag ordering; within-screen -18.74% versus NACA 2412; outside-screen substantially worse.
- **Exploratory, not generalised:** 13–14 successful 1%-chord perturbations, paired n=10, resampled intervals and sample-max tails; finite adversarial candidates, one wholly trusted case.
- **Not supported yet:** independent CFD/experimental aerodynamic agreement; strong physical manufacturing claims; robust Pareto-front repeatability under declared gate; formal 95% conditional coverage for optimiser-selected new designs.
- **Submission status:** author review-ready, not publication-ready until coauthor agreement and at least appropriately scoped CFD or corresponding narrower venue/claim decision.

### 10. Author approval and acceptance

Only Ben and the coauthors can approve the final research conclusions, authorship/disclosure statements and journal choice. There is no way to guarantee journal acceptance. The appropriate goal is a reproducible, accurately scoped paper whose strongest claims withstand an independent methodological review.

## Useful external literature

- Pichitkul et al., Applications in Engineering Science 25 (2026), 100310, https://doi.org/10.1016/j.apples.2026.100310 (related recent robust surrogate airfoil optimisation).
- Fannjiang et al., PNAS 119 (2022), e2204569119, https://doi.org/10.1073/pnas.2204569119 (feedback covariate shift in learned design).
- Angelopoulos & Bates, Foundations and Trends in Machine Learning (2023), DOI 10.1561/2200000101 (marginal conformal guarantees and limitations).

### 11. Additional 0.5% perturbation and frozen-trust audit

Independent 128-scenario, 0.5%-chord XFOIL checks completed on two scrambles (50000 and 51000). Matched completed cases: 92, 96. Paired mean drag reductions of 15.4% and 15.8% (success-conditioned). Matched-case empirical CVaR95 changes of -12.4% and -11.7%. No formal tail or reliability guarantee follows because both designs' failed XFOIL scenarios are nonrandomly excluded. These are genuine direct-solver checks from two new independently scrambled input designs; no new aerodynamic labels were used to retrain the frozen MLP.

**Material trust-calibration finding:** decomposition of the historical frozen acceptance rule on the held-out set finds 9050 states rejected by support distance, equal to all 9050 rejected states, and 0 states rejected independently by ensemble spread. Thus the active historical screen is effectively support-only. Ensemble disagreement has descriptive error-ranking value (Spearman 0.476/0.452/0.524 for CL/logCD/CM) but the published acceptance improvement cannot be attributed to a separate ensemble-disagreement cutoff. Do not retune the historical model or change test thresholds post hoc; any newly developed gate requires a preregistered calibration protocol and a fresh final confirmation cohort.
