"""Author a compact, source-bound comparator supplement, without invented claims."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/robust_v2/final_release_20261022"
FULL = BASE / "NEURALFOIL_FULL_CONFIRMATION/RESULTS.json"
THREE = BASE / "NEURALFOIL_COMPARISON/BENCHMARK_RESULT.json"
AUDIT = BASE / "NEURALFOIL_COMPARISON/INDEPENDENT_PAIRED_BENCHMARK_AUDIT.json"
OUTPUT = ROOT / "paper/SUPPLEMENTARY_NUMERICAL_COMPARISON.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Never silently overwrite the reviewed supplement")
    full = json.loads(FULL.read_text(encoding="utf-8"))
    short = json.loads(THREE.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if (
        full["geometries_completed_by_neuralfoil"] != 200
        or full["states_requested"] != 22600
        or full["same_solver_observed_paired_states"] != 17968
        or short["paired_observed_states"] != 1013
        or short["case_geometries_attempted"] != 357
        or not audit["all_case_labels_matched"]
    ):
        raise ValueError("Incomplete or inconsistent comparator source evidence")
    lines = [
        "# Supplement: competitor and tool qualification",
        "",
        "## S1. Why the study does not claim a better general-purpose predictor",
        "",
        "We compare the exact frozen five-member numerical surrogate to NeuralFoil "
        "0.3.3 xlarge via AeroSandbox 4.2.10. Both predictions are checked against "
        "the SAME available XFOIL labels at Re=1e6, Ncrit=9, M=0. The geometry "
        "supplied to NeuralFoil is the exact twelve-feature CST contour but is "
        "refitted to its native eighteen-parameter Kulfan encoding. Both are "
        "pretrained and are not tuned to this comparison.",
        "",
        "The internal training-airfoil overlap of NeuralFoil is unknown, so these "
        "figures are NOT independent held-out-family performance for that model. "
        "Our protected historical shapes were inspected in the earlier audit; "
        "the sealed confirmation contains development-shape mixtures.",
        "",
        "### Table S1. Entire sealed cohort: 200 shapes, 17,968 paired solver-observed states",
        "",
        "| Output | Our MAE | NeuralFoil MAE | 95% geometry-bootstrap interval, NeuralFoil − ours |",
        "|---|---:|---:|---|",
    ]
    for k, label in (("cl", "C_L"), ("cd", "C_D"), ("log_cd", "log(C_D)"), ("cm", "C_M")):
        result = full["per_target_observed_label_error"][k]
        low, high = result["fixed_model_geometry_bootstrap_95pct"]
        lines.append(
            f"| {label} | {result['controlled_mae']:.6f} | "
            f"{result['neuralfoil_mae']:.6f} | [{low:.6f}, {high:.6f}] |"
        )
    lines.extend(
        [
            "",
            "All four numerical errors favour NeuralFoil on the observed cases. "
            "The uncertainty intervals resample the already-fitted geometry results, "
            "not model retraining, actual CFD, or physically independent experiments.",
            "",
            "### Table S2. All actually observed cases at three preselected incidence angles",
            "",
            "| Cohort | Shapes | States | Our lift MAE | NF lift MAE | Our drag MAE | NF drag MAE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, label in (
        ("protected_historical_test", "Protected historical test"),
        ("sealed_development_mixtures", "Sealed mixture confirmation"),
    ):
        item = short["metrics_by_cohort"][name]
        ours = item["controlled_on_paired"]
        nf = item["neuralfoil_on_paired"]
        lines.append(
            f"| {label} | {item['paired_geometries']} | {item['paired_states']} | "
            f"{ours['cl_mae']:.6f} | {nf['cl_mae']:.6f} | "
            f"{ours['cd_mae']:.6f} | {nf['cd_mae']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The incidence values 2°, 4° and 6° were fixed before competitor inference; "
            "all 357 geometries with genuinely observed labels at those incidences "
            "were evaluated. There was no favourable-outcome filtering of neural results.",
            "",
            "## S2. Predicting numerical XFOIL missingness is not predicting real aerodynamic error",
            "",
            "Every one of the 22,600 pre-requested confirmation states is included; "
            "4,632 have no usable local XFOIL label. Risk scores from our model were "
            "frozen before those labels; the external pretrained competitor was later "
            "run on the same fixed input identities.",
            "",
            "| Model's fixed score for missing XFOIL observation | AUROC | Requests scored |",
            "|---|---:|---:|",
        ]
    )
    for k, label in (
        ("neuralfoil_minus_confidence", "NeuralFoil negative analysis_confidence"),
        ("our_frozen_ensemble_disagreement_risk", "Our frozen ensemble disagreement"),
        ("our_frozen_combined_risk", "Our frozen support + disagreement"),
        ("our_frozen_support_risk", "Our frozen support alone"),
    ):
        result = full["predicting_xfoil_unavailability_auroc"][k]
        lines.append(
            f"| {label} | {result['auroc']:.4f} | {result['eligible_requested_states']:,} |"
        )
    lines.extend(
        [
            "",
            "NeuralFoil confidence predicts this numerical-solver missingness more strongly "
            "than our model's risk features. This is NOT a physical validity probability "
            "or an independent CFD error measure. The numerical solver and convergence "
            "criteria define the outcome; no missing state was filled with pseudo truth.",
            "",
            "## S3. What this tool contributes",
            "",
            "Our model is NOT a general-purpose accuracy or runtime improvement over "
            "NeuralFoil. The tool's useful and testable contribution is the explicitly "
            "source-bound, geometry-consistent prediction and trust workflow, coupled "
            "to fail-closed geometry admission and to the fully recorded controlled "
            "optimisation experiment. The default CLI now chooses the controlled model "
            "and checks its own 74,946 development observations plus frozen artifact hashes. "
            "A prior path mistakenly reused historical support data and has been corrected.",
            "",
            "The research value therefore rests on failure-aware numerical evidence, "
            "provenance, prospective score/label separation, and robust versus nominal "
            "search comparison—not an unsubstantiated state-of-the-art predictor.",
            "",
            "## S4. Source-bound reproducibility",
            "",
            "Benchmark scripts: scripts/robust_v2/benchmark_controlled_neuralfoil.py, "
            "scripts/robust_v2/benchmark_full_confirmation_neuralfoil.py, and "
            "scripts/robust_v2/audit_neuralfoil_comparison.py. They retain every attempted "
            "case and source input SHA-256. This is a post hoc competitor comparison, "
            "not a redesign or retuning of the earlier sealed confirmation.",
            "",
            f"Full confirmation result SHA-256: {sha(FULL)}.",
            f"Three-incidence benchmark result SHA-256: {sha(THREE)}.",
            f"Independent paired bootstrap audit SHA-256: {sha(AUDIT)}.",
            "",
            "Independent viscous CFD, actual manufacturing distribution, and broad "
            "new-family performance remain outside the current numerical evidence.",
            "",
        ]
    )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("SUPPLEMENT_CREATED", len(lines), OUTPUT)


if __name__ == "__main__":
    main()
