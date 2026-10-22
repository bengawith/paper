"""Ten-seed LOW-BUDGET diagnostic for corrected candidate-normal optimisation.

Not the 128 population / 200 generation production study. Does not approve
the old Pareto front. Executes real corrected objectives on fixed geometry,
ensemble, calibration, latent and bounds; retains every seed file as evidence.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from robust_airfoil.config import OptimisationConfig, UncertaintyConfig, load_yaml
from robust_airfoil.constants import ROOT
from robust_airfoil.modelling.ensemble import LoadedEnsemble
from robust_airfoil.optimisation.problems import DesignBounds
from robust_airfoil.optimisation.study import _run_nsga_profile
from robust_airfoil.uncertainty.sampling import sobol_normal_samples

LIN = ROOT / "results/robust_v2/lineages/robust-v2-current-20260908"
OUT = ROOT / "reports/robust_v2/submission_review_20261014/CORRECTED_OPTIMISATION"
OUT.mkdir(parents=True, exist_ok=True)
opt = OptimisationConfig.model_validate(load_yaml(ROOT / "configs/robust_v2/optimisation.yaml"))
unc = UncertaintyConfig.model_validate(load_yaml(ROOT / "configs/robust_v2/uncertainty.yaml"))
profile = opt.full_nsga2.model_copy(
    update={
        "population": 32,
        "generations": 10,
        "uncertainty_samples": 16,
        "independent_seeds": 10,
    }
)
selected = json.loads((LIN / "optimisation/selected_designs.json").read_text(encoding="utf-8"))
ref = next(
    np.asarray(r["parameters"], float) for r in selected if r["design_id"] == "verified_reference"
)
old_summary = json.loads(
    (LIN / "optimisation/optimisation_summary.json").read_text(encoding="utf-8")
)
bounds = DesignBounds(
    np.array(old_summary["bounds"]["lower"], float), np.array(old_summary["bounds"]["upper"], float)
)
points = pd.read_parquet(ROOT / "data/robust_v2/processed/full_exact/model_points.parquet")
split = json.loads((LIN / "splits/full.json").read_text(encoding="utf-8"))
dev_ids = {str(r["airfoil_id"]) for r in split["records"] if r["split"] == "development"}
dev = points.loc[points.airfoil_id.astype(str).isin(dev_ids)].copy()
cal = json.loads((LIN / "calibration/trust_calibration.json").read_text(encoding="utf-8"))
latent = sobol_normal_samples(
    16, unc.basis_control_points_per_surface * 2, unc.seed + 2000, antithetic=True
)
ensemble = LoadedEnsemble(LIN / "full_ensemble")
manifest = LIN / "full_ensemble/ensemble_manifest.json"
(OUT / "smoke_seeds").mkdir(parents=True, exist_ok=True)
report = _run_nsga_profile(
    name="corrected-smoke",
    profile=profile,
    optimisation=opt,
    reference_parameters=ref,
    master_deltas=latent,
    master_invalid=np.zeros(len(latent), bool),
    development_points=dev,
    ensemble=ensemble,
    calibration=cal,
    bounds=bounds,
    output_root=OUT / "smoke_seeds",
    profile_index=7,
    ensemble_manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    uncertainty=unc,
)
compact = {
    "schema": "corrected-optimisation-ten-seed-smoke-v1",
    "status": "diagnostic_not_full_200_generation_acceptance",
    "profile": profile.model_dump(mode="json"),
    "nominal_reference": ref.tolist(),
    "manufacturing": "candidate-specific normal field; same latent scenarios per design",
    "agreement": report["agreement"],
    "per_run": [
        {
            "seed": r["seed"],
            "solution_count": r["solution_count"],
            "fallback": r["returned_population_fallback"],
            "context_hash": r["cache_context_hash"],
        }
        for r in report["runs"]
    ],
}
(OUT / "TEN_SEED_SMOKE_SUMMARY.json").write_text(
    json.dumps(compact, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
print(
    json.dumps(
        {
            "agreement": compact["agreement"]["agreement_status"],
            "distance": compact["agreement"]["pairwise_objective_distance_mean"],
            "feasible_runs": compact["agreement"]["feasible_run_count"],
        },
        indent=2,
    )
)
