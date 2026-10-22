"""Select genuine fixed finalists before running fresh common-scenario/XFOIL checks.

No solver outcome may enter selection. Immutable evidence is preserved at every
stage. Missing cases remain missing, not nominal/mean-imputed observations.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from robust_airfoil.config import OptimisationConfig, UncertaintyConfig, load_yaml
from robust_airfoil.geometry.cst import reconstruct_cst
from robust_airfoil.geometry.validity import validate_geometry
from robust_airfoil.hashing import hash_object, sha256_file
from robust_airfoil.modelling.ensemble import LoadedEnsemble
from robust_airfoil.modelling.freeze import verify_frozen_artifacts
from robust_airfoil.optimisation.checkpointing import assert_campaign_resources
from robust_airfoil.optimisation.controlled_selection import (
    select_controlled_nominal_finalist,
    select_controlled_risk_finalists,
)
from robust_airfoil.optimisation.study import RobustCandidateEvaluator
from robust_airfoil.provenance import utc_now, write_json_atomic
from robust_airfoil.research_lineage import resolve_lineage
from robust_airfoil.uncertainty.fields import smooth_normal_perturbation
from robust_airfoil.uncertainty.sampling import sobol_normal_samples
from robust_airfoil.validation.controlled_selected import run_selected_xfoil

ROOT = Path(__file__).resolve().parents[2]
PATHS = resolve_lineage(ROOT, "controlled")
LINEAGE = PATHS["results"]
REPORTS = PATHS["reports"] / "SELECTED_STUDY"
OUT = LINEAGE / "selected_study"
STUDY = ROOT / "configs/robust_v2/controlled_study.yaml"
EXE = ROOT / "tools/xfoil/xfoil.exe"


def sha_record(path):
    return {"relative_path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}


def write_geometry(parameters, path):
    x = (1 - np.cos(np.linspace(0, np.pi, 201))) / 2
    upper, lower = reconstruct_cst(np.asarray(parameters, float), x)
    top = np.c_[x, upper]
    bottom = np.c_[x, lower]
    if not validate_geometry(top, bottom).valid:
        raise ValueError("Selected nominal geometry is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Do not replace existing selected solver geometry")
    coordinates = np.vstack([top[::-1], bottom[1:]])
    path.write_text(
        "_controlled_selected\n"
        + "\n".join(f"{xp:.12f} {yp:.12f}" for xp, yp in coordinates)
        + "\n",
        encoding="utf-8",
    )
    return top, bottom


def select():
    selection_path = OUT / "FROZEN_SELECTED_DESIGNS.json"
    if selection_path.exists():
        return json.loads(selection_path.read_text(encoding="utf-8"))
    production_root = LINEAGE / "optimisation_production"
    nominal_root = LINEAGE / "nominal_control"
    protocol_path = production_root / "PRODUCTION_PROTOCOL.json"
    full_path = production_root / "PRODUCTION_SUMMARY.json"
    nominal_summary_path = nominal_root / "NOMINAL_CONTROL_SUMMARY.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    summary = json.loads(full_path.read_text(encoding="utf-8"))
    nominal = json.loads(nominal_summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "FULL_PROFILE_COMPLETED" or len(summary.get("per_seed", [])) != 10:
        raise ValueError("Cannot select without all ten completed robust seeds")
    if (
        nominal.get("status") != "FULL_NOMINAL_PROFILE_COMPLETED"
        or len(nominal.get("per_seed", [])) != 10
    ):
        raise ValueError("Cannot select without the complete matched-budget nominal control")
    config = load_yaml(STUDY)
    if config["selection"]["final_solver_outcomes_used_for_selection"]:
        raise ValueError("Post-solver selection forbidden")
    if any((LINEAGE / "selected_study").glob("xfoil/*")):
        raise RuntimeError("Fresh selected solver outcomes already exist before frozen selection")
    x = []
    f = []
    g = []
    source = []
    for record in summary["per_seed"]:
        filename = production_root / record["file"]
        if sha256_file(filename) != record["sha256"]:
            raise ValueError("Robust seed evidence hash changed")
        case = json.loads(filename.read_text(encoding="utf-8"))
        if len(case["history"]) != 200:
            raise ValueError("Incomplete robust seed")
        for geom, obj, constraint in zip(
            case["solutions"], case["objectives"], case["constraint_values"], strict=True
        ):
            x.append(geom)
            f.append(obj)
            g.append(constraint)
            source.append(record["seed"])
    robust = select_controlled_risk_finalists(
        np.asarray(x),
        np.asarray(f),
        np.asarray(g),
        np.asarray(protocol["context"]["reference_objective_scales"], float),
    )
    xn = []
    fn = []
    gn = []
    for record in nominal["per_seed"]:
        filename = nominal_root / record["file"]
        if sha256_file(filename) != record["sha256"]:
            raise ValueError("Nominal seed evidence hash changed")
        case = json.loads(filename.read_text(encoding="utf-8"))
        if len(case["history"]) != 200:
            raise ValueError("Incomplete nominal seed")
        xn.extend(case["solutions"])
        fn.extend(case["objectives"])
        gn.extend(case["constraint_values"])
    nominal_selected = select_controlled_nominal_finalist(
        np.asarray(xn), np.asarray(fn), np.asarray(gn)
    )
    chosen = {
        "reference": protocol["context"]["reference_parameters"],
        "nominal_control": nominal_selected["parameters"],
        "risk_knee": robust["selected"]["risk_knee"]["parameters"],
        "risk_mean": robust["selected"]["minimum_mean"]["parameters"],
        "risk_tail": robust["selected"]["minimum_tail"]["parameters"],
    }
    records = {
        "schema": "controlled-pre-verification-selected-designs-v1",
        "created_utc": utc_now(),
        "selection_inputs": {
            "robust_protocol": sha_record(protocol_path),
            "robust_summary": sha_record(full_path),
            "nominal_summary": sha_record(nominal_summary_path),
            "study_protocol": sha_record(STUDY),
            "entrypoint": sha_record(Path(__file__)),
        },
        "risk_selection": robust,
        "nominal_control_selection": nominal_selected,
        "fixed_design_parameters": chosen,
        "selection_complete_before_selected_solver": True,
        "solver_outcomes_used_for_selection": False,
        "claims": "Finite ten-seed algorithmic selection; no global optimum or independent CFD implied",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(selection_path, records)
    write_json_atomic(REPORTS / "FROZEN_SELECTED_DESIGNS.json", records)
    return records


def final_surrogate(sel):
    target = OUT / "FINAL_2048_SCENARIO_EVALUATION.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    manifest_path = LINEAGE / "FROZEN_MODEL_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_frozen_artifacts(
        manifest_path, {name: Path(path) for name, path in manifest["artifact_paths"].items()}
    )
    summary = json.loads(
        (LINEAGE / "optimisation_production/PRODUCTION_PROTOCOL.json").read_text(encoding="utf-8")
    )
    context = summary["context"]
    opt = OptimisationConfig.model_validate(context["optimisation"])
    unc = UncertaintyConfig.model_validate(context["uncertainty"])
    development = pd.read_parquet(LINEAGE / "partition_inputs/development.parquet")
    calibration = json.loads(
        (LINEAGE / "calibration/trust_calibration.json").read_text(encoding="utf-8")
    )
    ensemble = LoadedEnsemble(LINEAGE / "full_ensemble")
    latent = sobol_normal_samples(
        2048, 2 * unc.basis_control_points_per_surface, unc.seed + 90000, antithetic=True
    )
    evidence = []
    for amplitude in load_yaml(STUDY)["final_evaluation"]["amplitudes_fraction_chord"]:
        prepared = unc.model_copy(update={"amplitudes_fraction_chord": [float(amplitude)]})
        for name, parameters in sel["fixed_design_parameters"].items():
            assert_campaign_resources(OUT)
            evaluator = RobustCandidateEvaluator(
                np.asarray(sel["fixed_design_parameters"]["reference"], float),
                np.zeros((len(latent), 12)),
                np.zeros(len(latent), bool),
                development,
                ensemble,
                calibration,
                opt,
                candidate_latent=latent,
                uncertainty=prepared,
            )
            began = time.monotonic()
            scores, constraints, diag = evaluator.evaluate(np.asarray(parameters, float)[None, :])
            evidence.append(
                {
                    "design": name,
                    "amplitude_fraction_chord": amplitude,
                    "mean_penalised_weighted_cd": float(scores[0, 0]),
                    "empirical_cvar95_penalised_weighted_cd": float(scores[0, 1]),
                    "nominal_feasible": bool(np.all(constraints[0] <= 0)),
                    "constraint_values": constraints[0].tolist(),
                    "manufacturing_invalid_samples": diag[0]["manufacturing_invalid_samples"],
                    "uncertainty_trust_violations": diag[0]["uncertainty_trust_violations"],
                    "missing_lift_roots": diag[0]["missing_lift_roots"],
                    "elapsed_seconds": time.monotonic() - began,
                }
            )
            print(
                "FINAL_2048",
                name,
                amplitude,
                evidence[-1]["mean_penalised_weighted_cd"],
                flush=True,
            )
    output = {
        "schema": "controlled-fixed-finalists-fresh-2048-scenarios-v1",
        "created_utc": utc_now(),
        "selected_sha256": sha256_file(OUT / "FROZEN_SELECTED_DESIGNS.json"),
        "frozen_model_sha256": sha256_file(manifest_path),
        "latent_scramble_seed": unc.seed + 90000,
        "scenario_count": 2048,
        "latent_sha256": hash_object(latent.tolist()),
        "rows": evidence,
        "interpretation": "Penalised surrogate numerical-sensitivity distributions for fixed candidates, not actual manufactured populations, empirical tolerances or independent CFD.",
    }
    write_json_atomic(target, output)
    write_json_atomic(REPORTS / "FINAL_2048_SCENARIO_EVALUATION.json", output)
    return output


def perturbation_pairs(sel):
    summary = OUT / "SELECTED_XFOIL_SUMMARY.json"
    if summary.exists():
        return json.loads(summary.read_text(encoding="utf-8"))
    select_sha = sha256_file(OUT / "FROZEN_SELECTED_DESIGNS.json")
    strict = load_yaml(STUDY)["selected_xfoil"]
    unc = UncertaintyConfig.model_validate(load_yaml(ROOT / "configs/robust_v2/uncertainty.yaml"))
    designs = ["reference", "risk_knee", "nominal_control"]
    x = (1 - np.cos(np.linspace(0, np.pi, unc.cosine_points_per_surface))) / 2
    controls = unc.basis_control_points_per_surface
    parameters = {key: np.asarray(sel["fixed_design_parameters"][key], float) for key in designs}
    nominal_coordinates = {}
    cases = []
    for name in designs:
        geom = OUT / "xfoil" / name / "nominal/geometry.dat"
        if not geom.is_file():
            nominal_coordinates[name] = write_geometry(parameters[name], geom)
        else:
            uy, ly = reconstruct_cst(parameters[name], x)
            nominal_coordinates[name] = (np.c_[x, uy], np.c_[x, ly])
        cases.append(
            {"design": name, "campaign": "nominal", "sample_index": None, "geometry": geom}
        )
    for campaign in strict["campaigns"]:
        name = campaign["name"]
        n = int(campaign["scenarios"])
        amplitude = float(campaign["amplitude"])
        latent = sobol_normal_samples(
            n, controls * 2, unc.seed + int(campaign["scramble_offset"]), antithetic=True
        )
        for design in designs:
            upper, lower = nominal_coordinates[design]
            for index, row in enumerate(latent):
                rho = float(unc.upper_lower_correlation)
                lower_modes = (
                    rho * row[:controls] + np.sqrt(max(0.0, 1 - rho * rho)) * row[controls:]
                )
                displaced = smooth_normal_perturbation(
                    upper,
                    lower,
                    row[:controls],
                    lower_modes,
                    amplitude,
                    unc.correlation_length_chord,
                    unc.leading_edge_taper,
                    unc.trailing_edge_zero_displacement,
                )
                path = OUT / "xfoil" / design / name / f"sample_{index:03d}/geometry.dat"
                if not displaced.valid:
                    cases.append(
                        {
                            "design": design,
                            "campaign": name,
                            "sample_index": index,
                            "geometry": None,
                            "failure": "invalid_geometry_before_xfoil",
                            "reasons": list(displaced.invalid_reasons),
                        }
                    )
                    continue
                if path.exists():
                    expected = path.read_text(encoding="utf-8")
                    if not expected.startswith("_selected_displacement"):
                        raise ValueError("Unexpected existing geometry")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    coords = np.vstack([displaced.upper[::-1], displaced.lower[1:]])
                    path.write_text(
                        "_selected_displacement\n"
                        + "\n".join(f"{xp:.12f} {yp:.12f}" for xp, yp in coords)
                        + "\n",
                        encoding="utf-8",
                    )
                cases.append(
                    {"design": design, "campaign": name, "sample_index": index, "geometry": path}
                )
    manifest = OUT / "XFOIL_CASE_SELECTION_PROTOCOL.json"
    protocol = {
        "selected_designs_sha256": select_sha,
        "study_config_sha256": sha256_file(STUDY),
        "uncertainty_config_sha256": sha256_file(ROOT / "configs/robust_v2/uncertainty.yaml"),
        "executable_sha256": sha256_file(EXE),
        "requested_cases": len(cases),
        "case_identity": [
            {
                k: (v.relative_to(ROOT).as_posix() if isinstance(v, Path) else v)
                for k, v in item.items()
            }
            for item in cases
        ],
    }
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Solver selection protocol changed")
    else:
        write_json_atomic(manifest, protocol)
    records = []

    def run(item):
        if item["geometry"] is None:
            return {k: v for k, v in item.items() if k != "geometry"} | {
                "usable": False,
                "weighted_cd": None,
            }
        assert_campaign_resources(OUT)
        output = item["geometry"].parent / "run"
        receipt = run_selected_xfoil(
            EXE, item["geometry"], output, panel_count=160, timeout_seconds=30
        )
        return {k: v for k, v in item.items() if k != "geometry"} | {
            "usable": bool(receipt["assessment"]["usable"]),
            "weighted_cd": receipt["assessment"]["weighted_cd"],
            "geometry_sha256": sha256_file(item["geometry"]),
            "receipt_sha256": sha256_file(output / "CASE_RECEIPT.json"),
            "run_folder": output.relative_to(ROOT).as_posix(),
            "assessment": receipt["assessment"],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run, item): item for item in cases}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            records.append(row)
            if len(records) % 20 == 0:
                print(
                    "SELECTED_XFOIL",
                    len(records),
                    "/",
                    len(cases),
                    "usable",
                    sum(r["usable"] for r in records),
                    flush=True,
                )
                write_json_atomic(
                    REPORTS / "SELECTED_XFOIL_PROGRESS.json",
                    {
                        "completed": len(records),
                        "requested": len(cases),
                        "usable": sum(r["usable"] for r in records),
                        "utc": utc_now(),
                    },
                )
    records.sort(
        key=lambda r: (
            r["campaign"],
            r["design"],
            r["sample_index"] if r["sample_index"] is not None else -1,
        )
    )
    paired = {}
    for campaign in ["nominal", *[row["name"] for row in strict["campaigns"]]]:
        ref = {
            r["sample_index"]: r
            for r in records
            if r["campaign"] == campaign and r["design"] == "reference" and r["usable"]
        }
        for name in designs[1:]:
            other = {
                r["sample_index"]: r
                for r in records
                if r["campaign"] == campaign and r["design"] == name and r["usable"]
            }
            intersection = sorted(set(ref) & set(other), key=lambda n: -1 if n is None else n)
            if not intersection:
                paired[campaign + "_" + name] = {
                    "pairs": 0,
                    "mean_reduction": None,
                    "mean_difference": None,
                }
                continue
            a = np.asarray([ref[k]["weighted_cd"] for k in intersection], float)
            b = np.asarray([other[k]["weighted_cd"] for k in intersection], float)
            paired[campaign + "_" + name] = {
                "pairs": len(intersection),
                "reference_mean": float(a.mean()),
                "candidate_mean": float(b.mean()),
                "mean_difference": float((a - b).mean()),
                "mean_reduction": float((a - b).mean() / a.mean()),
                "reference_max": float(a.max()),
                "candidate_max": float(b.max()),
            }
    output = {
        "schema": "full-controlled-selected-XFOIL-v1",
        "created_utc": utc_now(),
        "preverification_design_selection_sha256": select_sha,
        "preverification_solver_protocol_sha256": sha256_file(manifest),
        "cases_requested": len(cases),
        "usable_cases": sum(r["usable"] for r in records),
        "cases_failed_or_not_admitted": sum(not r["usable"] for r in records),
        "cases": records,
        "paired_conditional_results": paired,
        "interpretation": "Fresh same-family XFOIL numerical checks at fixed controls. Solver failures remain missing; no unconditional manufacturing CVaR or independent CFD assertion.",
    }
    write_json_atomic(summary, output)
    write_json_atomic(REPORTS / "SELECTED_XFOIL_SUMMARY.json", output)
    return output


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    selected = select()
    final_surrogate(selected)
    perturbation_pairs(selected)
    print("CONTROLLED_2048_AND_SELECTED_XFOIL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
