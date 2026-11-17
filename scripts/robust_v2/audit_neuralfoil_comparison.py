"""Recompute the NeuralFoil comparison and paired clustered bootstrap from raw cases.

The benchmark's alpha 2/4/6 fixed selection was frozen before neuralfoil was
run. No outcome selection and no retuning of the controlled five-member model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from benchmark_controlled_neuralfoil import INPUTS, ROOT, select

from robust_airfoil.provenance import write_json_atomic

HERE = ROOT / "reports/robust_v2/final_release_20261022/NEURALFOIL_COMPARISON"
SEED = 20261022
BOOTSTRAP = 3000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    outcome = HERE / "INDEPENDENT_PAIRED_BENCHMARK_AUDIT.json"
    if outcome.exists():
        raise FileExistsError("Previous independent benchmark audit must not be replaced")
    protocol_path = HERE / "BENCHMARK_PRESELECTION.json"
    cases_path = HERE / "NEURALFOIL_CASES.jsonl"
    summary_path = HERE / "BENCHMARK_RESULT.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    selected = select()
    actual_preselection = [
        (str(row.cohort), str(row.airfoil_id), float(row.alpha_deg))
        for _, row in selected.iterrows()
    ]
    frozen_preselection = [
        (str(row["cohort"]), str(row["airfoil_id"]), float(row["alpha_deg"]))
        for row in protocol["selection"]
    ]
    if actual_preselection != frozen_preselection:
        raise ValueError("Preselection changed since NeuralFoil inference")
    for name, path in INPUTS.items():
        if protocol["source_inputs"][name]["sha256"] != sha(path):
            raise ValueError("Primary XFOIL labelled source has changed")
    if summary["preselection_sha256"] != sha(protocol_path) or summary[
        "case_records_sha256"
    ] != sha(cases_path):
        raise ValueError("Primary comparator hash receipts changed")
    groups = list(selected.groupby(["cohort", "airfoil_id"], sort=True))
    if len(groups) != len(rows):
        raise ValueError("Benchmark geometry count does not match source")
    all_rows = []
    for (cohort, airfoil), part in groups:
        case = rows[len(all_rows)]
        if (case["cohort"], case["airfoil_id"]) != (cohort, airfoil):
            raise ValueError("Benchmark geometry ordering corrupted")
        if case["status"] != "ok":
            raise ValueError("Original benchmark contains a failure, cannot ignore it")
        for key in ("cl", "cd", "cm"):
            if len(case[key]) != len(part):
                raise ValueError("Benchmark output length mismatch")
        all_rows.append(case)
    if len(rows) != 357 or sum(len(x) for _, x in groups) != 1013:
        raise ValueError("Unexpected benchmark observation count")
    tests = {}
    rng = np.random.default_rng(SEED)
    for cohort in INPUTS:
        cohort_means = {"cl": [], "log_cd": [], "cm": [], "cd": []}
        frame = selected[selected["cohort"] == cohort]
        ids = sorted(frame["airfoil_id"].unique())
        for identifier in ids:
            part = frame[frame["airfoil_id"] == identifier].sort_values("alpha_deg")
            case = next(r for r in rows if r["cohort"] == cohort and r["airfoil_id"] == identifier)
            for target in cohort_means:
                exact = part[target].to_numpy(float)
                if target == "cd":
                    other = np.asarray(case["cd"], float)
                    ours = np.exp(part["prediction_log_cd"].to_numpy(float))
                elif target == "log_cd":
                    other = np.log(np.asarray(case["cd"], float))
                    ours = part["prediction_log_cd"].to_numpy(float)
                else:
                    other = np.asarray(case[target], float)
                    ours = part[f"prediction_{target}"].to_numpy(float)
                cohort_means[target].append(
                    float((np.abs(other - exact) - np.abs(ours - exact)).mean())
                )
        result = {
            "requested_states": len(frame),
            "geometry_count": len(ids),
            "improvement_sign": "Negative means NeuralFoil lower absolute error than five-member controlled surrogate",
            "bootstrap_seed": SEED,
            "bootstrap_replicates": BOOTSTRAP,
            "per_target": {},
        }
        for target, individual in cohort_means.items():
            values = np.asarray(individual)
            samples = rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))
            boot = values[samples].mean(axis=1)
            result["per_target"][target] = {
                "neuralfoil_minus_controlled_geometry_macro_mae": float(values.mean()),
                "geometry_cluster_fixed_model_bootstrap_95pct": np.quantile(
                    boot, [0.025, 0.975]
                ).tolist(),
                "neuralfoil_lower_error_fraction_of_geometries": float(np.mean(values < 0)),
                "num_geometry_clusters": len(values),
            }
        tests[cohort] = result
    audited = {
        "schema": "independent-source-reconciled-competitor-diagnostic-v1",
        "comparison_source_sha256": sha(summary_path),
        "case_record_sha256": sha(cases_path),
        "selection_protocol_sha256": sha(protocol_path),
        "all_case_labels_matched": True,
        "comparator_cohorts": tests,
        "interpretation": (
            "A matched XFOIL label comparison, not independent CFD. "
            "NeuralFoil internal training-airfoil overlap is unknown; "
            "the difference is not evidence of out-of-training-family generality. "
            "The geometry bootstrap resamples frozen predictions, not retraining."
        ),
        "no_model_weights_or_original_scientific_receipts_changed": True,
    }
    write_json_atomic(outcome, audited)
    print(json.dumps(tests, indent=2))


if __name__ == "__main__":
    main()
