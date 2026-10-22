"""Frozen-before-label 40-section analytic NACA4-family numerical challenge.

No source-family observations from the controlled corpus are reused for labels.
The family is constructed by an analytic source recipe and physical separation
from all 1,256 existing CST shapes. The 5-member model/calibration never retrain.
This remains same-family XFOIL numerical evidence, not independent CFD.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from robust_airfoil.benchmarks.neuralfoil import benchmark_neuralfoil_from_coordinates
from robust_airfoil.data.builder import admit_local_polar_rows, merge_local_polar_rows
from robust_airfoil.geometry.cst import fit_cst, reconstruct_cst
from robust_airfoil.geometry.validity import validate_geometry
from robust_airfoil.modelling.calibrate import apply_trust_model
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS
from robust_airfoil.modelling.ensemble import LoadedEnsemble
from robust_airfoil.modelling.freeze import verify_frozen_artifacts
from robust_airfoil.provenance import write_json_atomic
from robust_airfoil.validation.xfoil_runner import XFoilCase, _run_xfoil_segment

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/robust_v2/processed/controlled_20261020/airfoils.parquet"
LINEAGE = ROOT / "results/robust_v2/lineages/robust-v2-controlled-20261020"
OUT = ROOT / "reports/robust_v2/final_release_20261022/EXTERNAL_NACA_FAMILY"
EXE = ROOT / "tools/xfoil/xfoil.exe"
X = (1 - np.cos(np.linspace(0, np.pi, 201))) / 2
ALPHA = np.arange(-12.0, 16.0001, 0.25)
SEGMENTS = (("positive", 0.0, 16.0, 0.25), ("negative", 0.0, -12.0, -0.25))
RANDOM_SEED = 20261022
EXPECTED_N = 40
MIN_SOURCE_GEOM_RMS = 0.002
MIN_PAIRWISE_GEOM_RMS = 0.002
PARAMS = FEATURE_COLUMNS[:-1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def analytic_naca4(m: float, p: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    # Conventional closed-trailing-edge NACA4 camber and thickness formula.
    yt = 5 * t * (0.2969 * np.sqrt(X) - 0.1260 * X - 0.3516 * X**2 + 0.2843 * X**3 - 0.1036 * X**4)
    yc = np.where(
        p > X, m / p**2 * (2 * p * X - X**2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * X - X**2)
    )
    slope = np.where(p > X, 2 * m / p**2 * (p - X), 2 * m / (1 - p) ** 2 * (p - X))
    theta = np.arctan(slope)
    xu, yu = X - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = X + yt * np.sin(theta), yc - yt * np.cos(theta)
    return (np.column_stack((X, np.interp(X, xu, yu))), np.column_stack((X, np.interp(X, xl, yl))))


def physical_signature(params: np.ndarray) -> np.ndarray:
    up, low = reconstruct_cst(params, X)
    return np.concatenate((up, low))


def geometry_seal() -> dict:
    folder = OUT / "geometry"
    path = OUT / "GEOMETRY_SELECTION_BEFORE_LABELS.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError("Unsealed preexisting external-family geometry")
    source = pd.read_parquet(DATA)
    if len(source) != 1256:
        raise ValueError("Original 1256 CST shapes changed")
    archived = np.stack([physical_signature(row) for row in source[PARAMS].to_numpy(float)])
    rng = np.random.default_rng(RANDOM_SEED)
    selected = []
    shapes = []
    proposals = 0
    while len(selected) < EXPECTED_N and proposals < 3000:
        proposals += 1
        m = float(rng.uniform(0.005, 0.065))
        p = float(rng.uniform(0.2, 0.75))
        t = float(rng.uniform(0.075, 0.195))
        upper, lower = analytic_naca4(m, p, t)
        fitted = fit_cst(upper, lower)
        if fitted.max_error > 0.005:
            continue
        validity = validate_geometry(fitted.reconstructed_upper, fitted.reconstructed_lower)
        if not validity.valid:
            continue
        signature = physical_signature(fitted.parameters)
        diff = np.sqrt(np.mean((archived - signature) ** 2, axis=1))
        mindist = float(diff.min())
        if mindist < MIN_SOURCE_GEOM_RMS:
            continue
        if (
            shapes
            and float(np.sqrt(np.mean((np.stack(shapes) - signature) ** 2, axis=1)).min())
            < MIN_PAIRWISE_GEOM_RMS
        ):
            continue
        uid = f"external_naca4_{len(selected):03d}"
        file = folder / f"{uid}.dat"
        file.parent.mkdir(parents=True, exist_ok=True)
        upper_y, lower_y = reconstruct_cst(fitted.parameters, X)
        coords = np.vstack(
            [np.column_stack((X[::-1], upper_y[::-1])), np.column_stack((X[1:], lower_y[1:]))]
        )
        file.write_text(
            uid + "\n" + "\n".join(f"{xp:.12f} {yp:.12f}" for xp, yp in coords) + "\n",
            encoding="utf-8",
        )
        selected.append(
            {
                "id": uid,
                "source_m": m,
                "source_p": p,
                "source_t": t,
                "feature_cst": fitted.parameters.tolist(),
                "fit_max_error_chord": float(fitted.max_error),
                "min_rms_to_all_existing_1256": mindist,
                "geometry_path": file.relative_to(ROOT).as_posix(),
                "geometry_sha256": sha(file),
            }
        )
        shapes.append(signature)
    if len(selected) != EXPECTED_N:
        raise RuntimeError(f"Failed to find {EXPECTED_N} physically distinct NACA4-family shapes")
    record = {
        "schema": "analytic-naca4-external-family-geometry-source-seal-v1",
        "seed": RANDOM_SEED,
        "proposals": proposals,
        "accepted": len(selected),
        "min_archive_rms_chord": MIN_SOURCE_GEOM_RMS,
        "min_internal_rms_chord": MIN_PAIRWISE_GEOM_RMS,
        "original_1256_source_sha256": sha(DATA),
        "no_solver_labels_used_for_selection": True,
        "source_geometry_family": "Analytically generated four-digit NACA camber and thickness curves, CST-reconstructed for exact solver labels",
        "not_necessarily_unseen_by_neuralfoil_internal_training": True,
        "cases": selected,
    }
    write_json_atomic(path, record)
    print("FROZEN_EXTERNAL_GEOMETRIES", len(selected), "proposals", proposals, flush=True)
    return record


def before_labels(selection: dict) -> dict:
    path = OUT / "PREDICTIONS_BEFORE_XFOIL_LABELS.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    manifest = LINEAGE / "FROZEN_MODEL_MANIFEST.json"
    frozen = json.loads(manifest.read_text(encoding="utf-8"))
    verify_frozen_artifacts(manifest, {k: Path(v) for k, v in frozen["artifact_paths"].items()})
    ensemble = LoadedEnsemble(LINEAGE / "full_ensemble")
    development = pd.read_parquet(LINEAGE / "partition_inputs/development.parquet")
    calibration = json.loads(
        (LINEAGE / "calibration/trust_calibration.json").read_text(encoding="utf-8")
    )
    frame = pd.DataFrame(
        [
            dict(
                zip(PARAMS, case["feature_cst"], strict=True),
                alpha_deg=float(alpha),
                airfoil_id=case["id"],
            )
            for case in selection["cases"]
            for alpha in ALPHA
        ]
    )
    our = apply_trust_model(
        development, ensemble.predict(frame), ensemble.first_scaling, calibration
    )
    native = {}
    for case in selection["cases"]:
        parameters = np.asarray(case["feature_cst"], float)
        upper_y, lower_y = reconstruct_cst(parameters, X)
        coordinates = np.vstack(
            [np.column_stack((X[::-1], upper_y[::-1])), np.column_stack((X[1:], lower_y[1:]))]
        )
        nf = benchmark_neuralfoil_from_coordinates(coordinates, ALPHA)
        native[case["id"]] = {"status": nf.status, "reason": nf.reason}
        if nf.status == "ok":
            for key in ("cl", "cd", "cm", "analysis_confidence"):
                val = getattr(nf, key)
                native[case["id"]][key] = np.asarray(val).tolist() if val is not None else None
    predictions = OUT / "before_label_model_predictions.parquet"
    our.to_parquet(predictions, index=False)
    record = {
        "schema": "external-NACA-prediction-freeze-before-any-XFOIL-label-v1",
        "geometry_selection_sha256": sha(OUT / "GEOMETRY_SELECTION_BEFORE_LABELS.json"),
        "frozen_ensemble_manifest_sha256": sha(manifest),
        "calibration_sha256": sha(LINEAGE / "calibration/trust_calibration.json"),
        "development_partition_sha256": sha(LINEAGE / "partition_inputs/development.parquet"),
        "prediction_file_sha256": sha(predictions),
        "comparison_neuralfoil": native,
        "prior_xfoil_solver_labels_generated": False,
        "requested_states": len(our),
        "not_used_to_refit_models": True,
    }
    write_json_atomic(path, record)
    print("FROZEN_EXTERNAL_PREDICTIONS_BEFORE_LABELS", len(our), flush=True)
    return record


def one_case(case: dict, context: str) -> dict:
    uid = case["id"]
    out = OUT / "solver" / uid
    receipt = out / "CASE_RECEIPT.json"
    if receipt.exists():
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if record["context_sha256"] != context:
            raise ValueError("Different old solver case context")
        for name, expected in record["files_sha256"].items():
            if sha(out / name) != expected:
                raise ValueError("Old solver case source changed")
        return record
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Partial solver case needs inspection")
    out.mkdir(parents=True, exist_ok=True)
    geometry = ROOT / case["geometry_path"]
    if sha(geometry) != case["geometry_sha256"]:
        raise ValueError("Frozen geometry changed")
    pieces = []
    segments = []
    for label, start, end, step in SEGMENTS:
        request = start + np.arange(round((end - start) / step) + 1) * step
        xfoil = XFoilCase(geometry, 1000000, 0.0, 9.0, start, end, step, 200, label)
        started = time.perf_counter()
        result = _run_xfoil_segment(EXE, xfoil, out / label, 30)
        frame = result.parsed_points.copy()
        frame["source_segment"] = label
        frame["source_path"] = (
            result.polar_path.relative_to(ROOT).as_posix() if result.polar_path else ""
        )
        frame["source_sha256"] = sha(result.polar_path) if result.polar_path else ""
        pieces.append(
            admit_local_polar_rows(
                frame, request, return_code=result.return_code, timed_out=result.timed_out
            )
        )
        segments.append(
            {
                "name": label,
                "status": result.status,
                "return_code": result.return_code,
                "timed_out": result.timed_out,
                "admitted_states": len(pieces[-1]),
                "seconds": time.perf_counter() - started,
            }
        )
    observed, audit = merge_local_polar_rows(pieces)
    observed["airfoil_id"] = uid
    observed["geometry_sha256"] = case["geometry_sha256"]
    observed.to_parquet(out / "observed.parquet", index=False)
    result = {
        "id": uid,
        "context_sha256": context,
        "requested": len(ALPHA),
        "observed": len(observed),
        "segments": segments,
        "merge_audit": audit,
        "files_sha256": {
            p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()
        },
    }
    write_json_atomic(receipt, result)
    return result


def numerical_solver(selection: dict, pred: dict) -> list[dict]:
    receipt = OUT / "ALL_40_SOLVER_RESULTS.json"
    if receipt.exists():
        return json.loads(receipt.read_text(encoding="utf-8"))["cases"]
    if pred["prior_xfoil_solver_labels_generated"]:
        raise ValueError("The source protocol was not before-label frozen")
    if sha(OUT / "GEOMETRY_SELECTION_BEFORE_LABELS.json") != pred["geometry_selection_sha256"]:
        raise ValueError("Source geometry revision changed")
    if sha(OUT / "before_label_model_predictions.parquet") != pred["prediction_file_sha256"]:
        raise ValueError("Frozen model predictions changed")
    context = hashlib.sha256(
        json.dumps(
            {
                "seal": pred["geometry_selection_sha256"],
                "predictions": pred["prediction_file_sha256"],
                "xfoil": sha(EXE),
                "requested_alpha": ALPHA.tolist(),
                "iterations": 200,
                "source_code": sha(Path(__file__)),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    records = []
    for idx, case in enumerate(selection["cases"], 1):
        records.append(one_case(case, context))
        if idx % 5 == 0:
            print(
                "NEW_FAMILY_XFOIL",
                idx,
                "/",
                len(selection["cases"]),
                "observed",
                sum(x["observed"] for x in records),
                flush=True,
            )
    result = {
        "schema": "new-analytic-source-40-shape-real-xfoil-v1",
        "requested_states": len(selection["cases"]) * len(ALPHA),
        "observed_states": sum(r["observed"] for r in records),
        "context_sha256": context,
        "source_polar_not_imputed": True,
        "cases": records,
    }
    write_json_atomic(receipt, result)
    return records


def evaluate(selection: dict, prediction: dict, cases: list[dict]) -> None:
    result = OUT / "NEW_FAMILY_RESULTS.json"
    if result.exists():
        raise FileExistsError("Previous completed family results exist")
    expected = pd.read_parquet(OUT / "before_label_model_predictions.parquet")
    gathered = [
        pd.read_parquet(OUT / "solver" / case["id"] / "observed.parquet")
        for case in cases
        if case["observed"]
    ]
    available = pd.concat(gathered, ignore_index=True)
    merged = expected.merge(
        available[["airfoil_id", "alpha_deg", "cl", "cd", "cm"]],
        on=["airfoil_id", "alpha_deg"],
        how="left",
        validate="one_to_one",
    )
    valid = merged.dropna(subset=["cl", "cd", "cm"]).copy()
    if len(valid) != len(available):
        raise ValueError("Numerical missing cases were altered")
    reported = {}
    for model in ("ours", "neuralfoil"):
        for target in ("cl", "cd", "cm"):
            if model == "ours":
                pred = (
                    valid["prediction_" + target].to_numpy(float)
                    if target != "cd"
                    else np.exp(valid["prediction_log_cd"].to_numpy(float))
                )
            else:
                column = target
                baseline = {
                    cid: np.asarray(row[column], float)
                    for cid, row in prediction["comparison_neuralfoil"].items()
                    if row["status"] == "ok"
                }
                byid = {
                    cid: dict(zip(ALPHA, values, strict=True)) for cid, values in baseline.items()
                }
                pred = np.asarray(
                    [
                        byid[str(uid)][float(alpha)]
                        for uid, alpha in zip(valid.airfoil_id, valid.alpha_deg, strict=True)
                    ]
                )
            obs = valid[target].to_numpy(float)
            reported[f"{model}_{target}_mae"] = float(np.abs(pred - obs).mean())
    report = {
        "schema": "geometrically-separated-analytic-family-prospective-numerical-challenge-v1",
        "requested_geometries": len(selection["cases"]),
        "geometry_source_selection_sha256": sha(OUT / "GEOMETRY_SELECTION_BEFORE_LABELS.json"),
        "before_label_predictions_sha256": sha(OUT / "PREDICTIONS_BEFORE_XFOIL_LABELS.json"),
        "requested_states": len(expected),
        "observed_states": len(valid),
        "missing_states": len(expected) - len(valid),
        "metrics": reported,
        "family": "Analytic NACA4 four-digit parameterised shapes physically screened against all 1256 original CST shapes",
        "limits": [
            "The new family was selected without solver labels and before its model scores were evaluated; no model was retrained.",
            "The selection occurred after development/previous confirmation research; not a preregistered external publication benchmark.",
            "NeuralFoil training-geometry overlap is unknown.",
            "XFOIL is the same numerical solver family as training and does not establish independent CFD truth.",
        ],
    }
    write_json_atomic(result, report)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selection = geometry_seal()
    if (OUT / "solver").exists() and not (OUT / "PREDICTIONS_BEFORE_XFOIL_LABELS.json").exists():
        raise RuntimeError("Solver cases existed before prediction freeze")
    before = before_labels(selection)
    records = numerical_solver(selection, before)
    evaluate(selection, before, records)


if __name__ == "__main__":
    main()
