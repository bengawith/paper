"""Promote strict 2026-10-14 XFOIL re-evaluation to current paper evidence.

Does not delete historical evidence; stores original consolidated paper_data.
Changes only direct-design verification quantities, associated manuscript wording,
and figure labels, preserving the other frozen lineage metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/robust_v2/submission_review_20261014"
OLD = ROOT / "paper/data/paper_data.json"
NEW = (
    ROOT
    / "results/robust_v2/submission_review_20261014/xfoil_strict/design_verification_summary.json"
)
DOC = ROOT / "paper/manuscript.md"
REV = ROOT / "paper/response_to_reviewers.md"
FIG = ROOT / "scripts/robust_v2/make_paper_figures.py"
original = json.loads(OLD.read_text(encoding="utf-8"))
fresh = json.loads(NEW.read_text(encoding="utf-8"))
backup = OUT / "BASELINE_paper_data.json"
if not backup.exists():
    backup.write_bytes(OLD.read_bytes())
for name, record in fresh["designs"].items():
    if record["shared_requested"] != 16:
        raise AssertionError(name)
    if record["shared_converged"] != len(record["shared_weighted_cd_samples"]):
        raise AssertionError(name)
    for suffix in ["nominal"] + [f"shared_{j:02d}" for j in range(16)]:
        status = (
            ROOT
            / "results/robust_v2/submission_review_20261014/xfoil_strict"
            / name
            / suffix
            / "run_status.json"
        )
        if status.is_file():
            state = json.loads(status.read_text(encoding="utf-8"))
            if not state.get("clean_exit", False) and state.get("status") == "ok":
                raise AssertionError(status)
        else:
            raise AssertionError(f"Missing strict-run status: {status}")
nomref = fresh["designs"]["reference_naca2412"]["nominal_weighted_cd"]
nomcand = fresh["designs"]["within_trust_knee"]["nominal_weighted_cd"]
meanref = fresh["designs"]["reference_naca2412"]["shared_mean_weighted_cd"]
meancand = fresh["designs"]["within_trust_knee"]["shared_mean_weighted_cd"]
nomgain = (nomref - nomcand) / nomref * 100
meangain = (meanref - meancand) / meanref * 100
cvarref = fresh["designs"]["reference_naca2412"]["shared_cvar95_weighted_cd"]
cvarcand = fresh["designs"]["within_trust_knee"]["shared_cvar95_weighted_cd"]
tailchange = (cvarcand / cvarref - 1) * 100
nref = fresh["designs"]["reference_naca2412"]["shared_converged"]
ncand = fresh["designs"]["within_trust_knee"]["shared_converged"]
nout = fresh["designs"]["outside_trust_deterministic"]["shared_converged"]
pairs = fresh["paired_improvement_vs_reference"]["within_trust_knee"]["paired_samples"]
original["design_verification"] = fresh
original["strict_direct_xfoil_review"] = {
    "executed_utc_date": "2026-10-14",
    "source": "results/robust_v2/submission_review_20261014/xfoil_strict/design_verification_summary.json",
    "source_sha256": hashlib.sha256(NEW.read_bytes()).hexdigest(),
    "strict_status_json_per_case": True,
    "meaning": "full-span, positive-drag and clean-exit gate",
    "historical_permissive_data_backup": "reports/robust_v2/submission_review_20261014/BASELINE_paper_data.json",
    "nominal_improvement_pct": nomgain,
    "marginal_success_only_mean_improvement_pct": meangain,
    "upper_tail_sample_max_change_pct": tailchange,
    "note": "95% CVaR with 13-14 scenarios equals sample maximum; no physical tail inference.",
}
OLD.write_text(json.dumps(original, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
draft = DOC.read_text(encoding="utf-8")
# In revised manuscript existing sentences include deliberately repeated old values.
draft = draft.replace(
    "15–16 converged perturbations per design",
    "13–14 fully completed perturbation sweeps per design",
)
draft = draft.replace("15–16 converged scenarios", "13–14 completed sweeps")
draft = draft.replace(
    "15–16 successful scenarios per design", "13–14 fully completed scenarios per design"
)
draft = draft.replace("15–16 successful scenarios", "13–14 fully completed scenarios")
draft = draft.replace("15–16 accepted perturbations", "13–14 strictly completed perturbations")
draft = draft.replace("15–16-case", "13–14-case")
draft = draft.replace("15–16-cases", "13–14-cases")
draft = draft.replace("15–16", "13–14")
draft = draft.replace("15 successful scenarios", "13 completed scenarios")
draft = draft.replace("reference's 16", "reference's 13")
draft = draft.replace(
    "16 requested perturbation scenarios per design",
    "16 requested perturbation scenarios per design",
)
draft = draft.replace(
    "16 requested perturbations per design, 15 successful within-screen and 16 for reference/outside-screen",
    "16 perturbations requested per design; 13 full sweeps for within-screen/reference and 14 for outside-screen",
)
draft = draft.replace("The reported 15-pair mean", "The reported 10-pair mean")
draft = draft.replace("0.00751", f"{meancand:.5f}")
draft = draft.replace("0.00850", f"{meanref:.5f}")
draft = draft.replace("0.00977", f"{cvarref:.5f}")
draft = draft.replace("0.00990", f"{cvarcand:.5f}")
draft = draft.replace("11.7%", f"{meangain:.1f}%")
draft = draft.replace(
    "+0.00098",
    f"+{fresh['paired_improvement_vs_reference']['within_trust_knee']['mean_improvement_vs_reference']:.5f}",
)
draft = draft.replace("[0.00062, 0.00131]", "[0.00058, 0.00142]")
draft = draft.replace(
    "0.01412", f"{fresh['designs']['outside_trust_deterministic']['shared_mean_weighted_cd']:.5f}"
)
draft = draft.replace(
    "0.01475", f"{fresh['designs']['outside_trust_deterministic']['shared_cvar95_weighted_cd']:.5f}"
)
draft = draft.replace(
    "−0.00561",
    f"−{abs(fresh['paired_improvement_vs_reference']['outside_trust_deterministic']['mean_improvement_vs_reference']):.5f}",
)
draft = draft.replace("[−0.00591, −0.00530]", "[−0.00602, −0.00537]")
draft = draft.replace("about 1.3% higher", f"about {tailchange:.1f}% higher")
draft = draft.replace(
    "one candidate solve failed",
    "three reference and three within-screen perturbation sweeps failed",
)
draft = draft.replace(
    "13 completed scenarios, 13 completed reference cases",
    "13 completed scenarios for both candidate and reference",
)
# Rewrite direct-design Table 5 rows exactly (do not silently leave obsolete values).
lines = draft.splitlines()
for i, line in enumerate(lines):
    if line.startswith("| NACA 2412 reference |"):
        lines[i] = (
            "| NACA 2412 reference | 0.60 | yes | 0.00745 | "
            + f"{meanref:.5f} | {cvarref:.5f} | — |"
        )
    elif line.startswith("| Robust design (within trust) |"):
        lines[i] = (
            "| Robust design (within screen) | 0.90 | yes | 0.00605 | "
            + f"{meancand:.5f} | {cvarcand:.5f} | +0.00103 [0.00058, 0.00142] (10 pairs) |"
        )
    elif line.startswith("| Surrogate optimum (outside trust) |"):
        bad = fresh["designs"]["outside_trust_deterministic"]
        lines[i] = (
            f"| Surrogate optimum (outside screen) | 3.72 | no | 0.01396 | {bad['shared_mean_weighted_cd']:.5f} | {bad['shared_cvar95_weighted_cd']:.5f} | −0.00568 [−0.00602, −0.00537] (11 pairs) |"
        )
draft = "\n".join(lines) + "\n"
# Mismatched denominator and failures explained in the table caption.
draft = draft.replace(
    "The means use each design's converged cases (within-screen n=15, reference/outside n=16)",
    "The means use strictly complete cases (within-screen/reference n=13, outside n=14)",
)
draft = draft.replace(
    "while paired differences use only matched successful scenarios",
    "while paired differences use only matched successful scenarios (10 and 11 respectively)",
)
DOC.write_text(draft, encoding="utf-8", newline="\n")
r = REV.read_text(encoding="utf-8")
r = r.replace("11.7%", f"{meangain:.1f}%")
r = r.replace("15–16", "13–14")
r = r.replace("15 successful", "13 completed")
REV.write_text(r, encoding="utf-8", newline="\n")
f = FIG.read_text(encoding="utf-8")
f = f.replace("15–16", "13–14")
f = f.replace("within-trust\\nrobust design", "within-screen\\nselected design")
FIG.write_text(f, encoding="utf-8", newline="\n")
print(f"Strict: n_ref={nref} n_candidate={ncand} n_outside={nout}, matched={pairs}")
print(f"Nominal {nomgain:.3f}%  sample-mean {meangain:.3f}%  observed-max change {tailchange:.3f}%")
