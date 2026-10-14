"""Audit frozen selected designs under corrected candidate-specific geometric uncertainty.

This does not perform an optimisation search and DOES NOT replace the old Pareto
claim. It independently evaluates historic selected geometries using a
candidate-specific, shared-latent, surface-normal perturbation physics contract.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from robust_airfoil.config import OptimisationConfig, UncertaintyConfig, load_yaml
from robust_airfoil.constants import ROOT
from robust_airfoil.modelling.ensemble import LoadedEnsemble
from robust_airfoil.optimisation.study import RobustCandidateEvaluator
from robust_airfoil.uncertainty.sampling import sobol_normal_samples

OUT = ROOT / "reports/robust_v2/submission_review_20261014/CORRECTED_OPTIMISATION"
OUT.mkdir(parents=True, exist_ok=True)
LIN = ROOT / "results/robust_v2/lineages/robust-v2-current-20260908"
configs = (
    OptimisationConfig.model_validate(load_yaml(ROOT / "configs/robust_v2/optimisation.yaml")),
    UncertaintyConfig.model_validate(load_yaml(ROOT / "configs/robust_v2/uncertainty.yaml")),
)
opt, unc = configs
points = pd.read_parquet(ROOT / "data/robust_v2/processed/full_exact/model_points.parquet")
split = json.loads((LIN / "splits/full.json").read_text(encoding="utf-8"))
dev_ids = {str(r["airfoil_id"]) for r in split["records"] if r["split"] == "development"}
dev = points.loc[points.airfoil_id.astype(str).isin(dev_ids)].copy()
calibration = json.loads((LIN / "calibration/trust_calibration.json").read_text(encoding="utf-8"))
selected = json.loads((LIN / "optimisation/selected_designs.json").read_text(encoding="utf-8"))
reference = next(
    np.asarray(d["parameters"], float) for d in selected if d["design_id"] == "verified_reference"
)
ensemble = LoadedEnsemble(LIN / "full_ensemble")
count = 64
latent = sobol_normal_samples(
    count, 2 * unc.basis_control_points_per_surface, unc.seed + 2_000, antithetic=True
)
evaluator = RobustCandidateEvaluator(
    reference,
    np.zeros((count, 12)),
    np.zeros(count, bool),
    dev,
    ensemble,
    calibration,
    opt,
    candidate_latent=latent,
    uncertainty=unc,
)
rows = []
for entry in selected:
    parameters = np.asarray(entry["parameters"], float)
    objectives, constraints, diagnostics = evaluator.evaluate(parameters[None, :])
    result = {
        "design_id": entry["design_id"],
        "historic_expected_penalised_objective": entry["expected_weighted_cd"],
        "historic_cvar95_penalised_objective": entry["cvar_95_weighted_cd"],
        "corrected_expected_penalised_objective": float(objectives[0, 0]),
        "corrected_cvar95_penalised_objective": float(objectives[0, 1]),
        "corrected_nominal_feasible": bool(np.all(constraints[0] <= 0)),
        "corrected_geometry_invalid_scenarios": int(
            diagnostics[0]["manufacturing_invalid_samples"]
        ),
        "corrected_trust_exit_scenarios": int(diagnostics[0]["uncertainty_trust_violations"]),
        "corrected_missing_lift_scenarios": int(diagnostics[0]["missing_lift_roots"]),
    }
    rows.append(result)
    print(
        result["design_id"],
        result["corrected_expected_penalised_objective"],
        result["corrected_cvar95_penalised_objective"],
        flush=True,
    )
report = {
    "schema": "corrected-candidate-specific-evaluation-of-historical-designs-v1",
    "note": "Historic candidates held fixed; physical uncertainty model corrected; NOT a newly optimised Pareto front.",
    "physical_protocol": "each latent Sobol draw applied to each candidate's OWN surface normal, followed by geometry/CST refit",
    "sample_count": count,
    "latent_sha256": hashlib.sha256(np.asarray(latent, dtype="<f8").tobytes()).hexdigest(),
    "calibration_source_sha256": hashlib.sha256(
        (LIN / "calibration/trust_calibration.json").read_bytes()
    ).hexdigest(),
    "selected_source_sha256": hashlib.sha256(
        (LIN / "optimisation/selected_designs.json").read_bytes()
    ).hexdigest(),
    "results": rows,
}
(OUT / "CORRECTED_SELECTED_DESIGNS.json").write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
