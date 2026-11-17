"""Reproduce numerical claims in the paper's frozen compact evidence, without editing it.

Usage: python scripts/robust_v2/review_claims_20261014.py
This audit is deterministic, stdlib-only, and does not generate aerodynamic labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper" / "data" / "paper_data.json"
DEST = ROOT / "reports" / "robust_v2" / "submission_review_20261014"
DEST.mkdir(parents=True, exist_ok=True)
d = json.loads(DATA.read_text(encoding="utf-8"))
v = d["design_verification"]
designs = v["designs"]
ref = designs["reference_naca2412"]
good = designs["within_trust_knee"]
bad = designs["outside_trust_deterministic"]
u = d["nsga2_full"]


def pct_improvement(reference, candidate):
    return 100.0 * (reference - candidate) / reference


# No resampling of missing solver outputs. Candidate's missing realisation stays missing.
ref_samples = ref["shared_weighted_cd_samples"]
good_samples = good["shared_weighted_cd_samples"]
bad_samples = bad["shared_weighted_cd_samples"]
known = {}
known["nominal_drag_improvement_pct"] = pct_improvement(
    ref["nominal_weighted_cd"], good["nominal_weighted_cd"]
)
known["marginal_mean_drag_improvement_pct"] = pct_improvement(
    ref["shared_mean_weighted_cd"], good["shared_mean_weighted_cd"]
)
known["nominal_outside_worse_pct"] = -pct_improvement(
    ref["nominal_weighted_cd"], bad["nominal_weighted_cd"]
)
known["within_trust_reported_paired_improvement"] = v["paired_improvement_vs_reference"][
    "within_trust_knee"
]["mean_improvement_vs_reference"]
known["within_trust_reported_paired_n"] = v["paired_improvement_vs_reference"]["within_trust_knee"][
    "paired_samples"
]
known["outside_reported_paired_n"] = v["paired_improvement_vs_reference"][
    "outside_trust_deterministic"
]["paired_samples"]
known["reference_sample_n"] = len(ref_samples)
known["within_trust_sample_n"] = len(good_samples)
known["outside_trust_sample_n"] = len(bad_samples)
known["estimated_tail_effective_count_reference"] = 0.05 * len(ref_samples)
known["estimated_tail_effective_count_within"] = 0.05 * len(good_samples)
known["xfoil_cvar_is_single_max_reference"] = math.isclose(
    ref["shared_cvar95_weighted_cd"], max(ref_samples), abs_tol=1e-12
)
known["xfoil_cvar_is_single_max_within"] = math.isclose(
    good["shared_cvar95_weighted_cd"], max(good_samples), abs_tol=1e-12
)
known["relative_cvar_increase_within_pct"] = 100 * (
    good["shared_cvar95_weighted_cd"] / ref["shared_cvar95_weighted_cd"] - 1
)
known["optimisation_mean_symmetric_distance"] = u["mean_symmetric_distance"]
known["optimisation_agreement_threshold"] = u["agreement_threshold"]
known["optimisation_agreement_pass"] = u["mean_symmetric_distance"] <= u["agreement_threshold"]
known["trust_inside_fraction"] = d["surrogate_trusted_fraction"]
known["heldout_coverage_all"] = {
    target: d["held_out_trust_band_coverage"][f"{target}_all"] for target in ("cl", "log_cd", "cm")
}
known["heldout_coverage_within"] = {
    target: d["held_out_trust_band_coverage"][f"{target}_trusted"]
    for target in ("cl", "log_cd", "cm")
}
known["source_sha256"] = hashlib.sha256(DATA.read_bytes()).hexdigest()

notes = [
    "XFOIL uses a 16-scenario exploratory campaign; the within-trust candidate has only 15 successful outcomes.",
    "The reported marginal means use different success sets (reference n=16, candidate n=15). Prefer paired-complete-case differences, report solver failure, and do not treat missing outcomes as favourable.",
    "Empirical 95% CVaR from n=15/16 corresponds to approximately one maximum observation, not a stable tail-risk estimate.",
    "The recorded paired bootstrap interval resamples scrambled-Sobol scenarios, which are not an IID random sample. Call this a descriptive sensitivity interval, not guaranteed population coverage.",
    "Locked-test band coverage is empirical for the evaluated distribution. It does not confer conditional coverage or certification under optimizer-induced distribution shift.",
    "A selected outside-trust counterexample and selected inside-trust success do not prove a universal or necessary relation across all candidate geometries.",
    "The 0.01-chord risk figure includes an out-of-trust penalty for some scenarios; it is not pure physical drag.",
    "The reported optimisation seed-distance threshold is not met. Keep repeatability limitation visible.",
    "AirfoilTools/XFOIL training labels and local XFOIL checks share solver physics; no independent RANS/experimental verification exists.",
    "Adversarial requested/evaluated/rejected figures require explicit denominators and stage-by-stage accounting.",
]
output = {
    "schema": "robust-v2-manuscript-reconciliation-20261014-v1",
    "source": DATA.relative_to(ROOT).as_posix(),
    "numeric_recalculation": known,
    "interpretive_limits": notes,
}
dest = DEST / "NUMERICAL_RECONCILIATION.json"
dest.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(known, indent=2))
print("Wrote", dest)
