"""Conservative V1-style manuscript polish, with exact preimage guards.

Applies the V1 paper's methods-first British-English tone, removes duplicate
contributions and trims promotional/informal phrases. Does not change data,
scientific equations, source labels, figure values, tables or references.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper/manuscript.md"
OUT = ROOT / "reports/robust_v2/submission_review_20261014"
OUT.mkdir(parents=True, exist_ok=True)
BEFORE = OUT / "MANUSCRIPT_PRE_TONE_POLISH.md"
if not BEFORE.exists():
    BEFORE.write_bytes(PAPER.read_bytes())
text = PAPER.read_text(encoding="utf-8")
records = []


def replace_once(old: str, new: str, label: str):
    global text
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {n}")
    text = text.replace(old, new)
    records.append({"section": label, "before_chars": len(old), "after_chars": len(new)})


abstract_old = (
    "Machine-learning surrogates accelerate aerodynamic design but may be exploited outside their training support. "
    "We develop a five-member, point-conditioned multilayer-perceptron ensemble predicting C_L, log C_D and C_M "
)
start = text.index(abstract_old)
stop = text.index("\n\n## Nomenclature", start)
abstract_before = text[start:stop]
abstract_new = (
    "We study the use of a compact machine-learning surrogate in risk-aware aerofoil design and examine how "
    "prediction reliability affects the resulting design decisions. A five-member ensemble of point-conditioned "
    "multilayer perceptrons predicts lift, log-drag and pitching moment from twelve Class–Shape Transformation "
    "(CST) parameters and angle of attack at Re = 10⁶, M = 0 and Ncrit = 9. On 26,692 grouped, held-out "
    "XFOIL-derived states from 233 aerofoils, the model achieves R² values of 0.989, 0.922 and 0.918 for "
    "these respective targets. An empirical support-based screen retains 66.1% of states, on which "
    "the observed 95%-band coverage is 97.8%, 96.5% and 99.2%, with lower errors than on excluded states. "
    "We apply smooth, bounded surface-normal perturbations to examine the expected and adverse-tail "
    "aerodynamic objectives of selected designs. A separate local XFOIL comparison finds 18.7% lower "
    "nominal required-lift weighted drag for a selected within-screen design than for NACA 2412. "
    "Two independently scrambled 0.5%-chord studies also yield lower mean weighted drag on "
    "92 and 96 converged, matched design/reference scenarios. An unrestricted outside-screen candidate "
    "instead performs worse under direct XFOIL. These results demonstrate the practical value of "
    "empirically screening surrogate predictions while quantifying its limitations. The stochastic "
    "manufacturing model is prescribed rather than measured, optimisation repeatability remains limited, "
    "and independent CFD or experimental verification is required before broader physical conclusions."
)
replace_once(abstract_before, abstract_new, "abstract")

duplicate = (
    "- A trust-calibrated multi-output surrogate: an ensemble of point-conditioned multilayer perceptrons "
    "that predicts C_L, C_D and C_M from a 12-parameter CST vector, with a split-conformal trust model "
    "whose coverage we validate on genuinely held-out data.\n"
    "- A physically admissible manufacturing-uncertainty model and a risk-aware optimisation: a smooth, bounded, "
    "spatially correlated surface-normal displacement field, and a genetic-algorithm optimisation that minimises "
    "the expected and tail-risk weighted drag at required lift subject to aerodynamic and geometric constraints.\n"
    "- A selected-design local XFOIL comparison showing how empirical trust screening can avoid a concrete "
    "surrogate extrapolation failure, alongside a finite adversarial search that identifies a large "
    "outside-screen discrepancy; neither finding constitutes universal certification.\n\n"
)
replace_once(duplicate, "", "remove duplicated contributions")

replace_once(
    "The design and optimisation of aerodynamic surfaces underpins efficiency",
    "The design and optimisation of aerodynamic surfaces underpin efficiency",
    "subject-verb agreement",
)
replace_once(
    "Assessment of the precursor showed that aggregate predictive accuracy alone cannot establish engineering value. "
    "Here we test whether an empirical reliability screen helps avoid extrapolation failures in selected design "
    "decisions. Additional drag/moment outputs and local XFOIL cross-checks respond to previous concerns without "
    "claiming independent physical validation.",
    "The precursor demonstrated the accuracy and inference speed of the CST-based lift-curve model, but did not "
    "establish its behaviour when used to select new geometries. In the present study we extend the predicted "
    "outputs to drag and pitching moment, quantify prediction reliability across the available design space, "
    "and evaluate selected optimisation results using separate local XFOIL calculations.",
    "motivation closer to V1",
)
replace_once(
    "We investigate accuracy with an empirical acceptance screen and test its effect on selected design outcomes. "
    "The contributions are:",
    "We investigate three related aspects of surrogate-assisted aerofoil design:",
    "contribution introduction",
)
replace_once(
    "Recent surrogate-assisted robust airfoil optimisation [27] and work on feedback-driven distribution shift "
    "in conformal prediction [28] mean that neither robust surrogate optimisation nor calibration alone is novel. "
    "Our specific emphasis is the operational use and limits of an empirical acceptance rule in a verified "
    "design example.",
    "Surrogate-assisted robust aerofoil optimisation has been investigated previously [27]. "
    "Prediction uncertainty can also change when a trained model is used to select subsequent inputs [28]. "
    "The emphasis here is therefore on the measured limitations of an empirical acceptance rule and its "
    "effect on selected aerodynamic design outcomes.",
    "V1-style literature gap",
)
replace_once(
    "Throughout, every reported quantity is produced from durable, hash-pinned evidence artefacts by a "
    "reproducible pipeline, and we report limiting findings — such as the reduced seed-to-seed "
    "reproducibility of the robust front and the convergence limits of XFOIL at high incidence — "
    "openly rather than concealing them.",
    "The numerical results are accompanied by archived source, configuration and evaluation records. "
    "The limitations of the surrogate, optimisation and XFOIL verification are considered alongside "
    "the reported design results.",
    "replace promotional reproducibility wording",
)
replace_once(
    "Consistent with the guidance we received at review, we did not pursue an extensive new architecture "
    "search; the reviewers did not identify the network itself as the limiting factor, and, as Section 4.1 "
    "shows, the compact multilayer perceptron is already competitive with a strong gradient-boosting "
    "baseline while also furnishing ensemble-disagreement diagnostics for a prospective, better-balanced "
    "screening method.",
    "We retain the compact MLP architecture as the principal learned model. Section 4.1 compares "
    "its predictive accuracy with linear, mean and gradient-boosting baselines, while the ensemble "
    "provides additional disagreement diagnostics.",
    "remove meta peer-review narration",
)
replace_once(
    "To justify the choice of model rather than assume it, we compared the multilayer perceptron "
    "against strong classical baselines under the same grouped protocol (Table 2).",
    "We compare the multilayer perceptron with established regression baselines under the same "
    "grouped evaluation protocol (Table 2).",
    "model-results tone",
)
replace_once(
    "The selected comparison illustrates one case in which screening helps avoid a surrogate-extrapolation failure.",
    "The selected comparison illustrates how screening may reduce surrogate-extrapolation errors.",
    "cautious interpretation",
)
replace_once(
    "This is useful evidence for screening against surrogate extrapolation, not a proof of universal "
    "correctness, verified 95%-tail superiority or independent physical robustness.",
    "These findings support the use of an empirical screening rule, while leaving population tail-risk "
    "and independent aerodynamic accuracy unresolved.",
    "conclusion V1 tone",
)
replace_once(
    "The code provides a CLI-oriented pipeline for trust-annotated prediction, uncertainty studies, "
    "optimisation and XFOIL checks.",
    "The implementation provides command-line access to trust-annotated predictions, uncertainty "
    "propagation, optimisation and XFOIL comparisons.",
    "tool language",
)
# Avoid numerical wording errors: rename Table 5 'expected' for outcome-conditioned means.
replace_once(
    "| Design | Support d_s | Within trust | Nominal C̄_D^w | Expected C̄_D^w (mfg) | CVaR₉₅ (mfg) | Paired improvement vs reference |",
    "| Design | Support d_s | Within screen | Nominal C̄_D^w | Mean C̄_D^w (completed cases) | Empirical upper-tail maximum | Paired improvement vs reference |",
    "Table 5 scientifically accurate headings",
)
replace_once(
    "these means use different successful scenario subsets (even though both have n=13) and show an approximate 8.8% reduction.",
    "The subsets differ because of solver failures, so the approximately 8.8% difference is a conditional descriptive comparison.",
    "paired cases clarity",
)
# Explicitly retain forward scientific limitations from V1 style in the conclusion.
PAPER.write_text(text, encoding="utf-8", newline="\n")
summary = {
    "schema": "robust-v2-v1-tone-polish-v1",
    "date": "2026-10-14",
    "reference": "prior_final_draft.docx (V1): measured methods-first prose; no scientific numbers borrowed",
    "new_sha256": hashlib.sha256(PAPER.read_bytes()).hexdigest(),
    "n_changes": len(records),
    "changes": records,
}
(OUT / "TONE_POLISH_MANIFEST.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print("Tone changes:", len(records), "abstract words", len(abstract_new.split()))
