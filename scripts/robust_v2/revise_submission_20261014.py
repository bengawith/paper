"""Conservative manuscript reconciliation; all original frozen evidence is left intact."""

import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root / "reports/robust_v2/submission_review_20261014"
out.mkdir(parents=True, exist_ok=True)
file = root / "paper/manuscript.md"
response_file = root / "paper/response_to_reviewers.md"
status_file = root / "paper/REVIEW_STATUS.md"
for p in (file, response_file, status_file):
    dest = out / ("BASELINE_" + p.name)
    if not dest.exists():
        dest.write_bytes(p.read_bytes())
text = file.read_text(encoding="utf-8")
changes = []


def bullet(prefix, new):
    global text
    pat = re.compile(r"(?m)^" + re.escape(prefix) + r"[^\n]*$")
    matches = list(pat.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"bullet preimage {prefix!r} count {len(matches)}")
    text = pat.sub(lambda m: new, text, count=1)
    changes.append(prefix)


def paragraph(prefix, new):
    global text
    chunks = text.split("\n\n")
    idx = [i for i, c in enumerate(chunks) if c.startswith(prefix)]
    if len(idx) != 1:
        raise ValueError(f"preimage {prefix!r} count {len(idx)}")
    chunks[idx[0]] = new
    text = "\n\n".join(chunks)
    changes.append(prefix)


def swap(old, new):
    global text
    if text.count(old) != 1:
        raise ValueError(f"swap preimage {old[:65]!r}, count={text.count(old)}")
    text = text.replace(old, new)
    changes.append(old[:70])


bullet(
    "- We build a compact",
    "- A compact point-conditioned ML ensemble estimates lift, log-drag and pitching moment from CST geometry with an empirically assessed reliability screen.",
)
bullet(
    "- We show on 26,692",
    "- On 26,692 held-out XFOIL-labelled states, accepted predictions show higher empirical interval coverage and lower error than rejected predictions; 66.1% of states are accepted.",
)
bullet(
    "- We prove why",
    "- A selected outside-screen surrogate optimum underperforms NACA 2412 in local XFOIL, whereas a selected within-screen design improves nominal and sampled mean drag.",
)
bullet(
    "- We deliver, and XFOIL",
    "- The within-screen design lowers nominal XFOIL weighted drag 18.7% and sampled mean drag about 11.7% over 15–16 converged scenarios; tail-risk improvement is not established.",
)
bullet(
    "- We implement the reviewer",
    "- A finite adversarial search finds its largest observed ML–XFOIL disagreement outside the screen, although in-screen adversarial evidence is sparse.",
)
paragraph(
    "Machine-learning surrogates can replace",
    """Machine-learning surrogates accelerate aerodynamic design but may be exploited outside their training support. We develop a five-member, point-conditioned multilayer-perceptron ensemble predicting C_L, log C_D and C_M from 12 CST geometry parameters and angle of attack at Re=1×10^6, M=0 and Ncrit=9. On 26,692 held-out XFOIL-derived states from 233 grouped aerofoils, R² is 0.989 for C_L, 0.922 for log C_D and 0.918 for C_M. An empirical screening rule accepts 66.1% of states, where observed 95%-band coverage is 97.8%, 96.5% and 99.2%, respectively, and errors are smaller than outside the screen. We use a prescribed smooth, bounded, correlated surface-normal perturbation model to study mean and adverse-tail weighted drag at fixed lift. In separate local XFOIL re-evaluation, a selected within-screen design has 18.7% lower nominal drag and about 11.7% lower sample-mean drag than NACA 2412, while a selected outside-screen candidate is substantially worse. The 1%-chord verification includes only 15–16 converged perturbations per design; its 95% CVaR and resampling intervals are exploratory rather than reliable population tail estimates. A finite adversarial search also finds large disagreement outside the screen. These results support the practical value of explicit empirical surrogate-domain screening, while optimisation-seed variability, unmeasured manufacturing deviations, XFOIL-only validation and the single operating condition constrain the claim. Selected CFD remains to be performed.""",
)

paragraph(
    "Peer review of that precursor",
    "Assessment of the precursor showed that aggregate predictive accuracy alone cannot establish engineering value. Here we test whether an empirical reliability screen helps avoid extrapolation failures in selected design decisions. Additional drag/moment outputs and local XFOIL cross-checks respond to previous concerns without claiming independent physical validation.",
)
paragraph(
    "We therefore reframe the contribution",
    """We investigate accuracy with an empirical acceptance screen and test its effect on selected design outcomes. The contributions are:
- A grouped, point-conditioned multi-output surrogate with measured held-out error and acceptance coverage; this is not a formal validity certificate for adaptively selected new designs.
- A prescribed bounded correlated surface-deviation model with mean/upper-tail objective optimisation, including trust-exit penalties and explicit limitations.
- A local XFOIL numerical cross-check of three selected designs and a finite adversarial disagreement search; CFD and experimental validation are not yet available.

Recent surrogate-assisted robust airfoil optimisation [27] and work on feedback-driven distribution shift in conformal prediction [28] mean that neither robust surrogate optimisation nor calibration alone is novel. Our specific emphasis is the operational use and limits of an empirical acceptance rule in a verified design example.""",
)

swap(
    "identical to the parameterisation used to generate the training labels, so that the surrogate and the solver always operate on the same geometric inputs",
    "used by the surrogate and reconstructed candidate geometries; the original source polars came from coordinate aerofoils, so CST fitting/reconstruction error remains a separate source of discrepancy",
)
swap(
    "only genuinely measured aerodynamic states are ever used as supervision",
    "only genuine source solver-computed aerodynamic rows are used as supervision",
)
swap(
    "genuinely held-out families rather than on leaked near-duplicates",
    "held-out geometry groups rather than on leaked near-duplicates",
)
paragraph(
    "We construct the trust domain by",
    """We build target-wise absolute-residual bands using a separate calibration partition and a split-conformal-style construction [17]. Development-support distance, ensemble disagreement and angle limits define an additional acceptance screen, applied consistently to evaluation and design. Observed held-out coverage is reported in Section 4.1. Ordinary conformal marginal coverage relies on exchangeability: this screen does not automatically guarantee target-wise conditional 95% coverage or accuracy under optimiser-induced distribution shift [28]. We call the result an empirical trust screen, not a certified design domain or physical safety guarantee.""",
)
paragraph(
    "We model manufacturing deviation as",
    """We prescribe surface-normal variability bounded by ε times nominal chord. A smooth correlated field with modelled correlation length 0.15 chord tapers near the leading edge and vanishes at the trailing edge. These are modelling assumptions, not measured manufacturing tolerances. Amplitudes 0.1%, 0.25%, 0.5% and 1% of chord are considered, with 1% a stress case. Actual perturbed coordinates undergo geometry checks and CST refitting; invalid geometry, refit failure and trust exits must be accounted separately. Independent CST-coefficient and uncorrelated coordinate perturbations provide sensitivity comparisons, not validated alternative manufacturing processes.""",
)
paragraph(
    "We formulate the principal study as",
    """We conduct a two-objective NSGA-II study [13,14] (population 128, 200 generations, 10 seeds) minimising the mean and empirical CVaR95 of required-lift weighted drag at C_L=0.4/0.7/1.0, with weights 0.25/0.50/0.25. Relative constraints include thickness, section area, leading-edge radius, pitching moment and attainable operating angles. The same empirical trust screen used for reporting restricts candidate evaluation; this reduces unsupported extrapolation without guaranteeing accuracy. Each candidate is evaluated with 64 shared-latent scrambled Sobol scenarios. When a perturbed case leaves trust, the reported optimisation quantity includes penalties and must be distinguished from physical drag. A 64-scenario tail estimate serves ranking, not definitive 95%-tail inference. A lift-versus-sensitivity scalar baseline is retained.""",
)
swap(
    "### 3.5 Multi-fidelity verification and adversarial trust probing",
    "### 3.5 Direct XFOIL numerical cross-check and finite adversarial probing",
)
paragraph(
    "We confront the surrogate with XFOIL 6.99",
    """We re-evaluate selected designs using local XFOIL 6.99 at Re=1×10^6, M=0, Ncrit=9, accepting only finite positive-drag rows with clean solver exit and required-angle coverage. This is a separate numerical execution of an XFOIL-class method, not independent experimental or RANS evidence. NACA 2412 and two selected designs are assessed nominally and under shared perturbations; failed solves are not filled in. The 15–16 successful scenarios per design permit an exploratory paired mean comparison but cannot resolve a 95% tail distribution. A finite adversarial CST search with local refinement finds large observed CL disagreement on α=0–12 degrees. It does not prove a global maximum or bound within-screen error, especially because only one fully trusted adversarial candidate was recorded.""",
)
paragraph(
    "On the 26,692 held-out locked-test",
    "On the 26,692 grouped held-out XFOIL-derived states, R² is 0.989 for C_L, 0.922 for log C_D and 0.918 for C_M; the corresponding MAEs are 0.057, 0.143 in log-drag and 0.0096 in moment (Fig. 1). These test outcomes quantify prediction fidelity on the sampled data distribution, not on arbitrary optimiser-generated geometries.",
)
paragraph(
    "The calibrated trust label is",
    """The empirical trust screen accepts 17,642 of 26,692 held-out states (66.1%) and rejects 9,050. Target-wise observed interval coverage is 93.7%, 94.4% and 94.5% across all states, versus 97.8%, 96.5% and 99.2% on the selected accepted subset. Accepted errors are lower by factors around 2.2, 1.9 and 2.7 for C_L, linear C_D and C_M (Table 3). These are descriptive conditional-on-selection observations, not a formal exchangeability-independent guarantee for arbitrary design optimisation.""",
)
paragraph(
    "The surrogate is fast,",
    "On the recorded GTX 1660 benchmark, a 6,144-state batch requires 55.5 ms, equivalent to 0.009 ms per state when amortised across the batch. Measured single-state and 48-state-polar latencies are about 19 ms and 22.5 ms, respectively. These are distinct workload definitions; the batch metric is not a per-query latency or a measured full-workflow advantage over direct XFOIL including training, geometry and validation overhead.",
)
paragraph(
    "Propagating the smooth surface-normal",
    """At each prescribed amplitude the surrogate evaluates 2,048 scenarios (Table 4). The tabulated quantities are penalised robust-objective values, not invariably pure physical drag. At 0.5% chord about 0.4% of cases leave the trust screen; at the 1%-chord stress case 17% exit, so the upper tail is materially influenced by penalties. These exits identify a support limitation, not a measured probability of physical manufacturing failure.""",
)
paragraph(
    "The necessity of the smoothness",
    """At 1%-chord amplitude, all 2,048 scenarios from the smooth correlated and multiplicative-CST generators pass the implemented geometry/refit admissibility check. Uncorrelated coordinate noise passes in 6/2,048 cases. This supports use of a smooth field compatible with the CST representation, but failure of a CST-refit threshold alone does not prove a real manufactured surface is physically impossible or that the chosen correlation model is representative of a specific process.""",
)
paragraph(
    "The multi-objective optimisation returns",
    """The reported aggregate surrogate-side front has 188 feasible non-dominated entries, but its seed agreement does not satisfy the predeclared criterion: mean normalised symmetric distance 0.283 exceeds threshold 0.20, with 2–47 feasible solutions per seed (Fig. 4). The reason for the variability is not independently isolated, and global optimiser repeatability is not established. Consequently, the Pareto plot is exploratory; the following three-design XFOIL comparison is a separate, limited numerical check rather than validation of the entire front.""",
)
swap(
    "### 4.4 Direct XFOIL verification: trust membership governs real performance",
    "### 4.4 Direct XFOIL check of three selected designs",
)
paragraph(
    "The central result of the paper is",
    """We compare three intentionally selected designs under local XFOIL (Fig. 5, Table 5): NACA 2412 (support distance 0.60), a within-screen risk-aware candidate (0.90), and an unrestricted outside-screen surrogate candidate (3.72); the support threshold is approximately 1.25. This study of selected examples is not a population estimate of the quality of all accepted or rejected designs.""",
)
paragraph(
    "The verdict is unambiguous.",
    """The within-screen candidate has 18.7% lower nominal weighted drag than NACA 2412 (0.00605 versus 0.00745). For the prescribed 1%-chord perturbations, its marginal mean from 15 successful scenarios is 0.00751 versus 0.00850 for the reference's 16; these unequally conditioned means give an approximate 11.7% reduction. The reported 15-pair mean improvement is +0.00098. Its bootstrap range [0.00062, 0.00131] is **descriptive**, not a guaranteed population confidence interval: scrambled Sobol outcomes are not IID and one candidate solve failed. The empirical CVaR95 values (0.00990 for the candidate and 0.00977 for reference) each reduce to one observed maximum at this sample size; the candidate's value is about 1.3% higher, with tail uncertainty unresolved. The outside-screen candidate is substantially worse in this example, with 0.01396 nominal drag (about 87% above reference). Together these cases motivate the acceptance screen but do not prove universal surrogate reliability or physical manufacturing robustness.""",
)
swap(
    "**Table 5. XFOIL-verified required-lift weighted drag (Re = 1×10⁶). A positive paired improvement denotes lower drag than the NACA 2412 reference.**",
    "**Table 5. Selected-design XFOIL numerical results (Re = 1×10⁶): 16 perturbations requested per design, 15 successful within-screen and 16 for reference/outside-screen. Marginal means use available successful cases; paired differences use matched cases. CVaR95 is a maximum-observation diagnostic for this small sample. Square brackets indicate descriptive paired resampling ranges, not guaranteed confidence intervals.**",
)
paragraph(
    "This is the engineering justification for trust calibration.",
    """The selected comparison illustrates one case in which screening helps avoid a surrogate-extrapolation failure. Because candidate selection is deliberate, numerical XFOIL checks share modelling assumptions with training labels, and CFD/experimental validation is absent, the result supports a case-specific design advantage, not proof that accepted candidates are always reliable or superior.""",
)
paragraph(
    "The adversarial search answers",
    """The finite adversarial campaign reports overlapping workflow counts (49 requested, 41 evaluated, 20 geometry rejections and 18 fully converged sweeps); these categories are not an exhaustive disjoint partition. The largest observed lift discrepancy is ΔC_L=0.67, for a wholly outside-screen candidate. The observed average outside-screen gap is 0.27 versus 0.043 for only **one** fully trusted adversarial probe. This comparison is too small to claim that all worst errors are outside trust or to estimate within-screen worst-case risk. Many demanding shapes also cause XFOIL convergence problems, limiting what either surrogate or solver can establish near separation.""",
)
paragraph(
    "The framework is delivered as",
    "The code provides a CLI-oriented pipeline for trust-annotated prediction, uncertainty studies, optimisation and XFOIL checks. The measured isolated single-state/polar latencies are about 19/22.5 ms and batched amortised throughput is 0.009 ms per state. The tool is a reproducible research implementation, not an independently CFD-validated aerodynamic design product or formal reliability certificate.",
)
paragraph(
    "The results reframe",
    """The work demonstrates an observed error-versus-coverage trade-off from empirical screening, plus a limited three-design XFOIL illustration where one accepted candidate improves nominal and sampled mean drag and one rejected candidate performs poorly. This does not establish that all accepted designs are superior, nor that all XFOIL-separated-flow solutions are physically accurate. Related robust surrogate-assisted aerofoil optimisation already exists [27], so the contribution is the specific integration of screening, a transparent risk objective and design counterexample, not invention of robust optimisation.""",
)
paragraph(
    "The novel and transferable lesson",
    """The transferable lesson is that empirical surrogate screening can expose extrapolation risk and influence selected design decisions, but its conditional accepted-set accuracy cannot be elevated to a universal guarantee for optimiser-selected distributions [28]. This case motivates wider independent solver verification and reporting of accepted coverage, failures and computational cost.""",
)
paragraph(
    "Three limitations bound our claims.",
    """Five limitations bound interpretation: (1) geometric deviations are a prescribed random-field model, not measured manufacturing tolerances; (2) both training labels and validation belong to an XFOIL-class solver family, with no independent CFD/wind-tunnel validation; (3) the 15–16-case perturbation comparison cannot establish a 95% tail distribution, and bootstrap intervals from scrambled Sobol samples are descriptive; (4) robust-front seed agreement fails (0.283 > 0.20); and (5) only one operating condition is studied, with solver and surrogate weakness near difficult separated-flow regimes. Selected CFD and larger independently scrambled perturbation experiments are required before stronger physical conclusions.""",
)
paragraph(
    "We have developed a trust-calibrated",
    """We present a compact multi-output ML aerodynamic surrogate with a measured held-out acceptance/error trade-off, coupled to an exploratory bounded geometric-uncertainty optimisation. On 26,692 grouped XFOIL-derived states, R² is 0.989 for C_L, 0.922 for log C_D and 0.918 for C_M. In a selected local XFOIL comparison, a within-screen candidate has 18.7% lower nominal drag and approximately 11.7% lower successful-case mean drag than NACA 2412, while an outside-screen candidate is substantially worse. This is useful evidence for screening against surrogate extrapolation, not a proof of universal correctness, verified 95%-tail superiority or independent physical robustness. The optimiser also fails its seed-repeatability threshold. The study provides a reproducible ML-and-engineering case and a clear path to CFD, larger perturbation campaigns and measured manufacturing uncertainty.""",
)
paragraph(
    "The processed geometries and polars,",
    "The repository and archived package contain source, compact evidence and a separate checksum-indexed local evidence/model archive; large training and solver artefacts are not all publicly committed. Reproduction requires those archived files and their documented environment. Public redistribution of third-party source data is subject to source terms and attribution.",
)
swap(
    "**Figure 3.** Geometric admissibility of three manufacturing-perturbation models at 1% chord amplitude (2,048 samples). Independent, uncorrelated coordinate noise renders 99.7% of samples non-physical.",
    "**Figure 3.** Geometry/CST-refit acceptance under three prescribed perturbation generators at 1% chord (2,048 samples). Uncorrelated coordinate noise fails this specific acceptance test in 99.7% of cases; this is not independent proof of physical impossibility.",
)
swap(
    "**Figure 5.** Direct XFOIL verification. Required-lift weighted drag (nominal, expected under manufacturing uncertainty, and 95% CVaR) for the NACA 2412 reference, a within-trust robust design, and the outside-trust surrogate optimum. Trust-domain membership, not the surrogate's predicted optimality, governs real performance.",
    "**Figure 5.** Selected-design local XFOIL comparison. Perturbation means use 15–16 converged scenarios and CVaR95 is effectively the largest sample, not a precise population-tail estimate.",
)
swap(
    "**Figure 6.** Adversarial surrogate-trust probing: (a) per-candidate mean lift disagreement versus support distance, coloured by trust-domain membership; (b) mean lift disagreement inside versus outside the trust domain, with the adversarial worst case annotated.",
    "**Figure 6.** Finite adversarial discrepancy search by support distance and trust membership; only one fully trusted adversarial design was available, limiting the inside/outside comparison.",
)
swap(
    "**Figure 7.** Representative worst-case surrogate–XFOIL lift divergence located by the adversarial search: a high-lift design lying entirely outside the calibrated trust domain, for which the surrogate systematically under-predicts lift across a fully converged XFOIL sweep.",
    "**Figure 7.** Largest observed discrepancy in the finite tested search, outside the acceptance screen; no mathematically global worst-case claim is made.",
)
swap(
    "\n---\n\n*All quantitative results",
    "\n[27] Pichitkul, A., Boksawat, A., Phiboon, T., Tantrairatn, S., Bureerat, S., Kasemsri, R. & Ariyarit, A. (2026). An intelligent multi-objective robust design optimization framework based on PathLine Transformation and surrogate-assisted sampling for uncertainty-aware airfoil design. Applications in Engineering Science, 25, 100310. https://doi.org/10.1016/j.apples.2026.100310\n\n[28] Fannjiang, C., Bates, S., Angelopoulos, A. N., Listgarten, J. & Jordan, M. I. (2022). Conformal prediction under feedback covariate shift for biomolecular design. Proceedings of the National Academy of Sciences, 119(43), e2204569119. https://doi.org/10.1073/pnas.2204569119\n\n---\n\n*All quantitative results",
)
# Figure 2 caption preserves original table/plot but clarifies the penalties.
text = text.replace(
    "**Figure 2.** Robust weighted-drag objective (expected and 95% CVaR) and the fraction of samples leaving the trust domain, versus manufacturing amplitude. Up to 0.5% chord the objective is a pure drag estimate; at 1% chord 17% of samples exit the trust domain.",
    "**Figure 2.** Penalised robust-objective summary and trust exits versus prescribed amplitude; 0.5% includes about 0.4% trust exits and 1% about 17%, so upper-tail values are not pure aerodynamic drag.",
)
text = text.replace(
    "a usable tool, in which the surrogate is trusted only where it has been certified",
    "a usable research tool, in which unverified extrapolation is explicitly screened",
)
assert "Selected CFD remains to be performed" in text
file.write_text(text, encoding="utf-8", newline="\n")
r = response_file.read_text(encoding="utf-8")
for old, new in [
    (
        "The central claim is settled by direct XFOIL verification.",
        "A selected-design claim is checked numerically by local XFOIL; independent CFD and larger tail campaigns are outstanding.",
    ),
    (
        "implemented exactly as requested",
        "implemented as a finite targeted search (not a global proof)",
    ),
    ("the worst case (ΔC_L = 0.67)", "the largest observed candidate (ΔC_L = 0.67)"),
    (
        "the surrogate's worst case is shown to be bounded by the trust flag",
        "one observed high-error example is flagged by the trust screen; a universal worst-case bound is not established",
    ),
    (
        "verified against an independent solver",
        "re-evaluated using a separate local XFOIL run from the same solver family",
    ),
    (
        "the framework is now multi-fidelity",
        "the framework is prepared for a higher-fidelity extension, but presently has only XFOIL-class numerical evidence",
    ),
    (
        "using XFOIL as an independent verification oracle",
        "using local XFOIL as a numerical-consistency cross-check",
    ),
    (
        "the trust calibration and verification apparatus are label-source-agnostic",
        "the implemented screening rule would require new calibration and validation for another label source",
    ),
    (
        "the surrogate's largest errors are confined outside the domain",
        "the largest error observed in this finite search lies outside the acceptance screen",
    ),
    (
        "whereas within the trust domain the disagreement collapses to 0.043",
        "whereas the sole wholly trusted adversarial candidate has a mean disagreement of 0.043; the in-screen adversarial sample is too small for generalisation",
    ),
    (
        "the calibrated trust domain is the mechanism that keeps optimisation inside the region where both are dependable",
        "the trust screen helps limit extrapolation, without certifying the physics or the selected design",
    ),
    (
        "genuinely novel and reproducible surrogate-trust and robust-design contribution",
        "a reproducible integration of empirical screening and selected design checks, pending CFD and larger uncertainty verification",
    ),
    (
        "The distinguishing novelty—an operational, calibrated criterion for *when to believe the surrogate*—is absent from the surrogate-modelling works surveyed in Section 2.",
        "The specific contribution is an empirical acceptance screen applied to a selected numerical design comparison. Related robust optimisation and design-shift literature is explicitly discussed; calibration or robust optimisation alone is not claimed as novel.",
    ),
    (
        "the degradation at high incidence is confined to the untrusted region",
        "errors are higher on rejected states and high-incidence solver difficulties remain",
    ),
    (
        "confirmed to be 18.7% better at nominal conditions and 11.7% better in expectation under manufacturing uncertainty",
        "found to have 18.7% lower nominal drag and approximately 11.7% lower successful-scenario mean under a small 15–16-case numerical study; physical 95%-tail robustness is not established",
    ),
    (
        "the framework is also delivered as a usable command-line tool that returns a trust-annotated polar for any CST aerofoil",
        "the framework includes a research CLI with empirical trust flags; outputs outside support are not validated",
    ),
]:
    if old in r:
        r = r.replace(old, new)
r = r.replace(
    '**Resubmission (revised, retitled):** "Risk-Aware Aerofoil Design under Manufacturing Uncertainty using a Trust-Calibrated Machine-Learning Surrogate and Multi-Fidelity Verification."',
    '**Current revised manuscript:** "Trust-Calibrated Machine-Learning Surrogates for Risk-Aware Aerofoil Design under Manufacturing Uncertainty, with Direct XFOIL Verification."',
)
r += "\n\n## Coauthor and CFD handoff\n\nThese responses describe the evidence available before an independent CFD/RANS addition. The 15–16 matched XFOIL perturbations do not support a definitive 95% tail probability statement. A new venue's reviewer response must be adapted to its actual review history, and the journal's resubmission policy must be checked by the authors. Selected CFD and a larger independent solver campaign should be incorporated before asserting physically verified manufacturing robustness.\n"
response_file.write_text(r, encoding="utf-8", newline="\n")
st = status_file.read_text(encoding="utf-8")
status_file.write_text(
    "> Historic review note (3 October 2026), superseded by reports/robust_v2/submission_review_20261014/REVIEW_REPORT.md for current claim wording. Frozen results unchanged.\n\n"
    + st,
    encoding="utf-8",
    newline="\n",
)
manifest = {
    "date": "2026-10-14",
    "count": len(changes),
    "changed_paragraphs": changes,
    "files": {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (file, response_file, status_file)
    },
}
(out / "EDIT_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print("Changed manuscript", len(changes), "paragraphs/fragments; source evidence untouched")
