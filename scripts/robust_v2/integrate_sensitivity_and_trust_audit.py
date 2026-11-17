"""Add the independent 0.5%-chord solver sensitivity and trust-component audit to author draft.

Idempotent; updates commentary only from current machine-readable evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "reports/robust_v2/submission_review_20261014"
EVID = REVIEW / "ML_DIAGNOSTICS"
SENS = EVID / "XFOIL_EPS005_TWO_SCRAMBLE_AUDIT.json"
ML = EVID / "ml_selective_audit.json"
s = json.loads(SENS.read_text(encoding="utf-8"))
d = json.loads(ML.read_text(encoding="utf-8"))
assert s["independent_scrambles_finished"] == 2
assert all(r["paired_improvement_mean"] > 0 for r in s["runs"])
first, second = s["runs"]
assert first["scramble_seed_offset"] != second["scramble_seed_offset"]
screen = d["frozen_rule_component_audit"]
assert screen["failed_disagreement_without_support_failure"] == 0
paper_json = ROOT / "paper/data/paper_data.json"
bundle = json.loads(paper_json.read_text(encoding="utf-8"))
bundle["additional_xfoil_eps005_two_scrambles"] = {
    "results": s,
    "source_sha256": hashlib.sha256(SENS.read_bytes()).hexdigest(),
    "role": "supplementary descriptive validation at 0.5%-chord prescribed perturbation",
}
bundle["trust_screen_component_audit"] = {
    "results": screen,
    "source_sha256": hashlib.sha256(ML.read_bytes()).hexdigest(),
}
paper_json.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

manuscript = ROOT / "paper/manuscript.md"
body = manuscript.read_text(encoding="utf-8")
start_marker = "The selected comparison illustrates one case in which screening helps avoid"
if body.count(start_marker) != 1:
    raise RuntimeError("Unexpected manuscript preimage")
paragraph = (
    "**Supplementary two-scramble XFOIL sensitivity at 0.5%-chord prescribed deviation.** "
    "To assess whether the direction of the selected-design comparison depends on one 16-point stress sample, "
    "we ran two separate 128-scenario scrambled-Sobol campaigns at 0.5% chord, retaining only full XFOIL "
    "sweeps with all required-lift roots available. Each uses the same latent scenarios across compared designs "
    "and the two scrambles are independently seeded. The reference/inside-screen comparison has "
    f"{first['paired_usable']} and {second['paired_usable']} valid pairs; "
    f"the paired mean weighted-drag reductions are {first['paired_improvement_fraction'] * 100:.1f}% and "
    f"{second['paired_improvement_fraction'] * 100:.1f}%, respectively. "
    "Both samples also show lower empirical upper-tail drag for the selected candidate among **completed matched cases** "
    "(Table S1). However, incomplete XFOIL solutions are excluded—not imputed—and the complete-case selection "
    "is neither random nor independent of flow/geometric difficulty. These results strengthen the numerical "
    "same-solver evidence at 0.5% chord but do **not** establish unconditional manufacturing-risk "
    "probabilities or independent CFD accuracy.\n\n"
    "**Table S1. Independent 0.5%-chord XFOIL scrambles; all means and empirical tail values are conditional "
    "on full solver completion and attainable required lift.**\n\n"
    "| Independent scramble | Requested/reference usable/selected usable | Matched usable | "
    "Paired mean drag reduction | Candidate empirical CVaR95 change on matched cases |\n"
    "|---|---:|---:|---:|---:|\n"
    f"| A | 128 / {first['reference_usable']} / {first['candidate_usable']} | {first['paired_usable']} | "
    f"{first['paired_improvement_fraction'] * 100:.1f}% | {first['paired_empirical_cvar95_candidate_change_fraction'] * 100:.1f}% |\n"
    f"| B | 128 / {second['reference_usable']} / {second['candidate_usable']} | {second['paired_usable']} | "
    f"{second['paired_improvement_fraction'] * 100:.1f}% | {second['paired_empirical_cvar95_candidate_change_fraction'] * 100:.1f}% |\n\n"
)
if "**Supplementary two-scramble XFOIL sensitivity" in body:
    before, after = body.split("**Supplementary two-scramble XFOIL sensitivity", 1)
    suffix = after.split(start_marker, 1)[1]
    body = before + paragraph + start_marker + suffix
else:
    body = body.replace(start_marker, paragraph + start_marker)
manuscript.write_text(body, encoding="utf-8", newline="\n")

review = REVIEW / "REVIEW_REPORT.md"
txt = review.read_text(encoding="utf-8")
marker = "### 11. Additional 0.5% perturbation and frozen-trust audit"
extra = (
    f"\n\n{marker}\n\n"
    f"Independent 128-scenario, 0.5%-chord XFOIL checks completed on two scrambles "
    f"({first['scramble_seed_offset']} and {second['scramble_seed_offset']}). "
    f"Matched completed cases: {first['paired_usable']}, {second['paired_usable']}. "
    f"Paired mean drag reductions of {first['paired_improvement_fraction'] * 100:.1f}% and "
    f"{second['paired_improvement_fraction'] * 100:.1f}% (success-conditioned). "
    f"Matched-case empirical CVaR95 changes of {first['paired_empirical_cvar95_candidate_change_fraction'] * 100:.1f}% and "
    f"{second['paired_empirical_cvar95_candidate_change_fraction'] * 100:.1f}%. "
    "No formal tail or reliability guarantee follows because both designs' failed XFOIL scenarios are nonrandomly excluded. "
    "These are genuine direct-solver checks from two new independently scrambled input designs; no new aerodynamic labels were used to retrain the frozen MLP.\n\n"
    "**Material trust-calibration finding:** decomposition of the historical frozen acceptance rule on the "
    f"held-out set finds {screen['failed_support_distance']} states rejected by support distance, "
    f"equal to all {screen['untrusted_states_total']} rejected states, and "
    f"{screen['failed_disagreement_without_support_failure']} states rejected independently by ensemble spread. "
    "Thus the active historical screen is effectively support-only. Ensemble disagreement has descriptive error-ranking "
    "value (Spearman 0.476/0.452/0.524 for CL/logCD/CM) but the published acceptance improvement cannot be attributed "
    "to a separate ensemble-disagreement cutoff. Do not retune the historical model or change test thresholds post hoc; "
    "any newly developed gate requires a preregistered calibration protocol and a fresh final confirmation cohort.\n"
)
if marker in txt:
    txt = txt.split(marker, 1)[0].rstrip()
review.write_text(txt.rstrip() + extra, encoding="utf-8", newline="\n")

for target in (ROOT / "README.md", ROOT / "paper/README.md", REVIEW / "READ_THIS_FIRST.md"):
    text = target.read_text(encoding="utf-8")
    marker2 = "## 14 October: Additional ML and XFOIL evidence"
    if marker2 in text:
        text = text.split(marker2, 1)[0].rstrip()
    text += (
        "\n\n" + marker2 + "\n\n"
        f"Two separately scrambled 128-scenario 0.5%-chord XFOIL studies give {first['paired_usable']} "
        f"and {second['paired_usable']} strictly completed matched cases and paired mean drag reductions "
        f"of {first['paired_improvement_fraction'] * 100:.1f}% / {second['paired_improvement_fraction'] * 100:.1f}%. "
        "These results are conditional on solver completion; no independent CFD evidence or unconditional tail guarantee. "
        "A frozen-model audit also finds historical acceptance was effectively support-distance-driven, "
        "not independently gated by disagreement. See ML_DIAGNOSTICS and the supplement in manuscript.\n"
    )
    target.write_text(text, encoding="utf-8", newline="\n")
print("Integrated authentic paired-solver and frozen-model evidence into manuscript and review.")
