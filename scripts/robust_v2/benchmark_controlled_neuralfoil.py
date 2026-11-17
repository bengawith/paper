"""Full-cohort, same-label NeuralFoil comparison: diagnostic, not new truth.

All geometries with observed alpha=2,4,6 on both the locked historical test
and sealed confirmation are preselected, before NeuralFoil inference. Uses
saved prelabel controlled predictions; never fits to benchmark outcomes.
Retains every nonconvergent/invalid benchmark case, not just successes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from robust_airfoil.benchmarks.neuralfoil import benchmark_neuralfoil_from_coordinates
from robust_airfoil.geometry.cst import reconstruct_cst
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS
from robust_airfoil.provenance import write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
LINEAGE = ROOT / "results/robust_v2/lineages/robust-v2-controlled-20261020"
ANGLES = (2.0, 4.0, 6.0)
INPUTS = {
    "protected_historical_test": LINEAGE / "locked_test/predictions.parquet",
    "sealed_development_mixtures": LINEAGE / "confirmation/joined_observed_predictions.parquet",
}


def filehash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def select() -> pd.DataFrame:
    selected = []
    for cohort, source in INPUTS.items():
        points = pd.read_parquet(source)
        valid = points["alpha_deg"].isin(ANGLES)
        valid &= np.isfinite(points[["cl", "cd", "cm"]]).all(axis=1)
        valid &= points["cd"] > 0
        frame = points.loc[valid].copy()
        frame["cohort"] = cohort
        unique = frame[["airfoil_id", "alpha_deg"]].duplicated()
        if unique.any():
            raise ValueError(f"Duplicate numerical geometry-incidence identities in {cohort}")
        selected.append(frame)
    all_points = pd.concat(selected, ignore_index=True)
    return all_points.sort_values(["cohort", "airfoil_id", "alpha_deg"]).reset_index(drop=True)


def metrics(df: pd.DataFrame, prefix: str) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {"observed_points": len(df)}
    for target in ("cl", "log_cd", "cm", "cd"):
        actual = df[target].to_numpy(float)
        prediction = df[f"{prefix}_{target}"].to_numpy(float)
        error = prediction - actual
        result[f"{target}_mae"] = float(np.abs(error).mean()) if len(df) else None
        result[f"{target}_rmse"] = float(np.sqrt(np.mean(error**2))) if len(df) else None
        sum_squares = np.sum((actual - actual.mean()) ** 2) if len(df) else 0
        result[f"{target}_r2"] = (
            float(1.0 - np.sum(error**2) / sum_squares) if sum_squares > 0 else None
        )
        group = df.assign(_err=np.abs(error)).groupby("airfoil_id")["_err"].mean()
        result[f"{target}_geometry_macro_mae"] = float(group.mean()) if len(group) else None
    return result


def main() -> None:
    import aerosandbox
    import neuralfoil

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "BENCHMARK_PRESELECTION.json"
    rows_path = out / "NEURALFOIL_CASES.jsonl"
    result_path = out / "BENCHMARK_RESULT.json"
    if result_path.exists():
        raise FileExistsError("A prior completed benchmark must not be replaced")
    points = select()
    fixed = {
        "schema": "all-available-predeclared-three-incidence-neuralfoil-comparison-v1",
        "angles_deg": list(ANGLES),
        "source_inputs": {
            k: {"path": v.relative_to(ROOT).as_posix(), "sha256": filehash(v)}
            for k, v in INPUTS.items()
        },
        "selection": [
            {
                "cohort": str(r.cohort),
                "airfoil_id": str(r.airfoil_id),
                "alpha_deg": float(r.alpha_deg),
                "geometry_sha256": str(
                    r.get("geometry_sha256_prediction", r.get("geometry_sha256", ""))
                ),
            }
            for _, r in points.iterrows()
        ],
        "limitations": [
            "All available solver-observed states at three preset incidences. This does not test XFOIL failures.",
            "Historical test geometries are previously inspected; sealed mixtures come from development parents.",
            "NeuralFoil's original training identity overlap is unknown; no independent family-blind claim.",
            "NeuralFoil receives a refitted 18-parameter Kulfan shape from the same exact CST coordinates.",
        ],
    }
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != fixed:
            raise ValueError("Benchmark selected cases or source bytes changed")
    else:
        write_json_atomic(protocol_path, fixed)
    pairs = points.groupby(["cohort", "airfoil_id"], sort=True)
    groups = list(pairs)
    if rows_path.exists():
        prior = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
        if len(prior) > len(groups) or any(
            (row["cohort"], row["airfoil_id"]) != key
            for row, (key, _) in zip(prior, groups, strict=False)
        ):
            raise ValueError("Checkpoint ordering or identity mismatch")
    else:
        prior = []
    print(
        "SELECTED",
        len(points),
        "states across",
        len(groups),
        "geometries",
        "RESUME",
        len(prior),
        flush=True,
    )
    x = (1 - np.cos(np.linspace(0, np.pi, 201))) / 2
    t0 = time.perf_counter()
    with rows_path.open("a", encoding="utf-8") as log:
        for n, ((cohort, airfoil), rows) in enumerate(groups[len(prior) :], len(prior)):
            params = rows.iloc[0][FEATURE_COLUMNS[:-1]].to_numpy(float)
            upper, lower = reconstruct_cst(params, x)
            coordinates = np.vstack(
                [np.column_stack([x[::-1], upper[::-1]]), np.column_stack([x[1:], lower[1:]])]
            )
            before = time.perf_counter()
            test = benchmark_neuralfoil_from_coordinates(
                coordinates, rows["alpha_deg"].to_numpy(float)
            )
            elapsed = time.perf_counter() - before
            record = {
                "cohort": str(cohort),
                "airfoil_id": str(airfoil),
                "status": test.status,
                "reason": test.reason,
                "points": len(rows),
                "neuralfoil_elapsed_seconds_including_geometry_refit": elapsed,
            }
            if test.status == "ok":
                for field in ("cl", "cd", "cm", "analysis_confidence"):
                    value = getattr(test, field)
                    record[field] = np.asarray(value).tolist() if value is not None else None
                if not all(np.isfinite(record[k]).all() for k in ("cl", "cd", "cm")):
                    record["status"] = "nonfinite"
                elif not all(value > 0 for value in record["cd"]):
                    record["status"] = "nonpositive_drag"
                elif any(len(record[k]) != len(rows) for k in ("cl", "cd", "cm")):
                    record["status"] = "shape_mismatch"
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            if (n + 1) % 25 == 0:
                print(
                    "NEURALFOIL",
                    n + 1,
                    "/",
                    len(groups),
                    "elapsed",
                    round(time.perf_counter() - t0, 1),
                    flush=True,
                )
    readback = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    if len(readback) != len(groups):
        raise AssertionError("Every selected geometry must have a retained case receipt")
    common = []
    for receipt, ((cohort, airfoil), frame) in zip(readback, groups, strict=True):
        if (receipt["cohort"], receipt["airfoil_id"]) != (cohort, airfoil):
            raise ValueError("Case source order corrupted")
        if receipt["status"] != "ok":
            continue
        part = frame.copy()
        for key in ("cl", "cd", "cm"):
            part["neuralfoil_" + key] = receipt[key]
        part["neuralfoil_log_cd"] = np.log(part["neuralfoil_cd"])
        part["controlled_cl"] = part["prediction_cl"]
        part["controlled_cd"] = np.exp(part["prediction_log_cd"])
        part["controlled_log_cd"] = part["prediction_log_cd"]
        part["controlled_cm"] = part["prediction_cm"]
        common.append(part)
    paired = pd.concat(common, ignore_index=True) if common else points.iloc[:0].copy()
    summary = {
        "schema": "matched-observed-XFOIL-labels-neuralfoil-comparison-v1",
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "neuralfoil": getattr(neuralfoil, "__version__", "installed"),
            "aerosandbox": getattr(aerosandbox, "__version__", "installed"),
        },
        "preselection_sha256": filehash(protocol_path),
        "case_records_sha256": filehash(rows_path),
        "case_geometries_attempted": len(groups),
        "cases_ok": sum(x["status"] == "ok" for x in readback),
        "cases_failed": sum(x["status"] != "ok" for x in readback),
        "requested_observed_states": len(points),
        "paired_observed_states": len(paired),
        "metrics_by_cohort": {},
        "per_case_status_counts": {
            status: sum(x["status"] == status for x in readback)
            for status in sorted(set(x["status"] for x in readback))
        },
        "inference_timing": {
            "neuralfoil_total_seconds_including_18_parameter_geometry_refit": float(
                sum(x["neuralfoil_elapsed_seconds_including_geometry_refit"] for x in readback)
            ),
            "not_a_comparable_model_only_speed_benchmark": True,
        },
        "limits": fixed["limitations"]
        + [
            "Both methods compared against the SAME numerical XFOIL labels, not independent CFD.",
            "Conditional errors use only paired successfully evaluated states.",
            "The baseline is not retrained; its internal training data may overlap these airfoils.",
            "No evidence of superiority is asserted unless actual paired metrics support it.",
        ],
    }
    for cohort in INPUTS:
        sub = paired[paired["cohort"] == cohort]
        requested = points[points["cohort"] == cohort]
        summary["metrics_by_cohort"][cohort] = {
            "requested_states": len(requested),
            "paired_states": len(sub),
            "requested_geometries": requested["airfoil_id"].nunique(),
            "paired_geometries": sub["airfoil_id"].nunique(),
            "controlled_on_paired": metrics(sub, "controlled"),
            "neuralfoil_on_paired": metrics(sub, "neuralfoil"),
        }
    write_json_atomic(result_path, summary)
    print(
        "DONE",
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("metrics_by_cohort", "limits")}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
