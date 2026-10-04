"""Clean, correctly-signed XFOIL verification of surrogate-optimised designs.

Confirms the central thesis directly: a design inside the surrogate's calibrated
trust domain is verified by XFOIL as robust against the NACA 2412 reference,
whereas a surrogate-optimised design that sits outside the trust domain is not.
A positive paired difference denotes LOWER drag than the reference (improvement).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from robust_airfoil.config import OptimisationConfig, UncertaintyConfig, load_yaml
from robust_airfoil.constants import ROOT
from robust_airfoil.geometry.cst import fit_cst, reconstruct_cst
from robust_airfoil.geometry.normalise import resample_surfaces_to_common_x
from robust_airfoil.geometry.validity import validate_geometry
from robust_airfoil.modelling.calibrate import apply_trust_model, trust_thresholds
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS
from robust_airfoil.modelling.ensemble import LoadedEnsemble
from robust_airfoil.modelling.evaluate import cvar
from robust_airfoil.optimisation.objectives import weighted_required_lift_drag
from robust_airfoil.sources.xfoil import locate_xfoil
from robust_airfoil.uncertainty.fields import smooth_normal_perturbation
from robust_airfoil.uncertainty.sampling import sobol_normal_samples
from robust_airfoil.validation.xfoil_runner import XFoilCase, run_xfoil

PARAM = FEATURE_COLUMNS[:-1]
LIN = "robust-v2-current-20260908"
BASE = ROOT / "results/robust_v2/lineages" / LIN
OUT = BASE / "design_verification"
N_SHARED = 16
ALPHA = (0.0, 11.0, 1.0)


def _write_geometry(path: Path, name: str, upper: np.ndarray, lower: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = np.vstack([upper[::-1], lower[1:]])
    path.write_text("\n".join([f"_{name}", *(f"{x:.12f} {y:.12f}" for x, y in coords)]) + "\n", encoding="utf-8")


def _weighted_cd(points: pd.DataFrame, opt: OptimisationConfig) -> float | None:
    if not {"alpha_deg", "cl", "cd"}.issubset(points.columns):
        return None
    valid = points.dropna(subset=["alpha_deg", "cl", "cd"]).sort_values("alpha_deg")
    if not len(valid):
        return None
    value = weighted_required_lift_drag(
        valid["alpha_deg"].to_numpy(float), valid["cl"].to_numpy(float), valid["cd"].to_numpy(float),
        np.asarray(opt.service_targets.cl_required, float), np.asarray(opt.service_targets.weights, float),
    )
    return float(value) if np.isfinite(value) else None


def _support(parameters: np.ndarray, ensemble: LoadedEnsemble, dev: pd.DataFrame,
             calibration: dict, alpha: np.ndarray) -> float:
    frame = pd.DataFrame(np.repeat(parameters[None, :], len(alpha), axis=0), columns=pd.Index(PARAM))
    frame["alpha_deg"] = alpha
    frame["airfoil_id"] = "q"
    trusted = apply_trust_model(dev, ensemble.predict(frame), ensemble.first_scaling, calibration)
    return float(trusted["support_distance"].max())


def _run(parameters: np.ndarray, upper: np.ndarray, lower: np.ndarray, name: str, exe: Path,
         opt: OptimisationConfig, unc: UncertaintyConfig) -> float | None:
    cosx = (1 - np.cos(np.linspace(0, np.pi, len(upper)))) / 2
    fu, fl = resample_surfaces_to_common_x(upper, lower, cosx)
    if not validate_geometry(upper, lower).valid or fit_cst(fu, fl).max_error > unc.maximum_cst_refit_error_fraction_chord:
        return None
    geom = OUT / name / "geometry.dat"
    _write_geometry(geom, name, upper, lower)
    case = XFoilCase(geom, 1_000_000, 0.0, 9.0, ALPHA[0], ALPHA[1], ALPHA[2], 100, "positive")
    result = run_xfoil(exe, case, OUT / name, 150)
    return _weighted_cd(result.parsed_points, opt)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data_root = ROOT / "data/robust_v2/processed/full_exact"
    airfoils = pd.read_parquet(data_root / "airfoils.parquet")
    points = pd.read_parquet(data_root / "model_points.parquet")
    split = json.loads((BASE / "splits/full.json").read_text(encoding="utf-8"))
    dev_ids = {str(r["airfoil_id"]) for r in split["records"] if r["split"] == "development"}
    dev = points.loc[points["airfoil_id"].astype(str).isin(sorted(dev_ids))].copy()
    cols = [f"lower_weight_{i}" for i in range(5)] + [f"upper_weight_{i}" for i in range(5)] + ["leading_edge_weight", "TE_thickness"]
    rrow = airfoils.loc[airfoils["airfoil_id"] == "naca2412"].iloc[0]
    reference = np.asarray([rrow[c] for c in cols], float)
    calibration = json.loads((BASE / "calibration/trust_calibration.json").read_text(encoding="utf-8"))
    opt = OptimisationConfig.model_validate(load_yaml(ROOT / "configs/robust_v2/optimisation.yaml"))
    unc = UncertaintyConfig.model_validate(load_yaml(ROOT / "configs/robust_v2/uncertainty.yaml"))
    support_threshold, _, _ = trust_thresholds(calibration)
    exe = locate_xfoil(ROOT)
    assert exe is not None
    ensemble = LoadedEnsemble(BASE / "full_ensemble")
    alpha_support = np.asarray(opt.ava_baseline.alpha_deg, float)

    selected = json.loads((BASE / "optimisation/selected_designs.json").read_text(encoding="utf-8"))
    by_id = {d["design_id"]: np.asarray(d["parameters"], float) for d in selected}
    designs = {
        "reference_naca2412": reference,
        "within_trust_knee": by_id["trust_constrained_pareto_knee"],
        "outside_trust_deterministic": by_id["deterministic_optimum"],
    }

    controls = unc.basis_control_points_per_surface
    amplitude = max(unc.amplitudes_fraction_chord)
    latent = sobol_normal_samples(N_SHARED, controls * 2, unc.seed + 50_000, antithetic=True)

    records: dict[str, dict] = {}
    nominal_cd: dict[str, float | None] = {}
    sample_cd: dict[str, list[float | None]] = {}
    cosx = (1 - np.cos(np.linspace(0, np.pi, unc.cosine_points_per_surface))) / 2
    for name, params in designs.items():
        uy, ly = reconstruct_cst(params, cosx)
        upper, lower = np.column_stack([cosx, uy]), np.column_stack([cosx, ly])
        support = _support(params, ensemble, dev, calibration, alpha_support)
        nominal_cd[name] = _run(params, upper, lower, f"{name}/nominal", exe, opt, unc)
        samples: list[float | None] = []
        for j, row in enumerate(latent):
            pert = smooth_normal_perturbation(
                upper, lower, row[:controls], row[controls:], amplitude,
                unc.correlation_length_chord, unc.leading_edge_taper, unc.trailing_edge_zero_displacement,
            )
            if not pert.valid:
                samples.append(None)
                continue
            samples.append(_run(params, pert.upper, pert.lower, f"{name}/shared_{j:02d}", exe, opt, unc))
        sample_cd[name] = samples
        converged = [v for v in samples if v is not None]
        records[name] = {
            "support_distance": support,
            "within_trust_domain": bool(support <= support_threshold),
            "nominal_weighted_cd": nominal_cd[name],
            "shared_converged": len(converged),
            "shared_requested": len(samples),
            "shared_weighted_cd_samples": converged,
            "shared_mean_weighted_cd": float(np.mean(converged)) if converged else None,
            "shared_cvar95_weighted_cd": (
                cvar(np.asarray(converged, float), 0.95) if len(converged) >= 2 else None
            ),
        }
        print(
            f"{name} support={support:.3f} trust={records[name]['within_trust_domain']} "
            f"nominal={nominal_cd[name]} conv={len(converged)}/{len(samples)}"
        )

    ref_samples = sample_cd["reference_naca2412"]
    rng = np.random.default_rng(unc.seed + 7)
    paired = {}
    for name in ("within_trust_knee", "outside_trust_deterministic"):
        diffs = np.asarray([
            ref_samples[j] - sample_cd[name][j]
            for j in range(N_SHARED)
            if ref_samples[j] is not None and sample_cd[name][j] is not None
        ], float)
        if len(diffs) >= 2:
            boot = rng.choice(diffs, size=(10000, len(diffs)), replace=True).mean(axis=1)
            paired[name] = {
                "paired_samples": int(len(diffs)),
                "mean_improvement_vs_reference": float(diffs.mean()),
                "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                "sign_convention": "positive means design has LOWER weighted drag than the reference",
            }
        else:
            paired[name] = {"paired_samples": int(len(diffs)), "mean_improvement_vs_reference": None}
        print("paired", name, paired[name])

    summary = {
        "schema_version": "robust-v2-design-verification-v1",
        "reynolds_number": 1_000_000, "mach": 0.0, "ncrit": 9.0,
        "alpha_grid_deg": list(ALPHA), "shared_perturbations": N_SHARED,
        "manufacturing_amplitude_fraction_chord": amplitude,
        "strict_support_threshold": support_threshold,
        "designs": records,
        "paired_improvement_vs_reference": paired,
    }
    (OUT / "design_verification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", OUT / "design_verification_summary.json")


if __name__ == "__main__":
    main()
