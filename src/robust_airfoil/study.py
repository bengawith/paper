from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robust_airfoil.config import load_yaml
from robust_airfoil.constants import ROOT
from robust_airfoil.hashing import hash_paths, sha256_file
from robust_airfoil.model_evidence import build_ml_evidence
from robust_airfoil.pipeline import run_viability
from robust_airfoil.provenance import git_output, utc_now, write_json_atomic

STAGES = (
    "preflight",
    "legacy_audit",
    "recovery",
    "source_acquisition",
    "dataset",
    "model",
    "advanced",
    "report",
)
STAGE_PHASES = {
    "preflight": "00",
    "legacy_audit": "02",
    "recovery": "03",
    "source_acquisition": "04",
    "dataset": "05-08",
    "model": "09-16",
    "advanced": "17-21",
    "report": "22",
}
_LINEAGE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,79}$")


@dataclass(frozen=True)
class StudyPaths:
    lineage_id: str
    results_root: Path
    reports_root: Path

    @property
    def context_path(self) -> Path:
        return self.reports_root / "STUDY_CONTEXT.json"

    @property
    def status_path(self) -> Path:
        return self.reports_root / "STUDY_STATUS.json"


def study_paths(lineage_id: str, root: Path | None = None) -> StudyPaths:
    if not _LINEAGE_ID.fullmatch(lineage_id):
        raise ValueError("lineage_id must use lowercase letters, digits, '_' or '-'")
    root = root or ROOT
    return StudyPaths(
        lineage_id=lineage_id,
        results_root=root / "results/robust_v2/lineages" / lineage_id,
        reports_root=root / "reports/robust_v2/lineages" / lineage_id,
    )


def _protocol(path: Path) -> dict[str, Any]:
    payload = load_yaml(path)
    if payload.get("schema_version") != "robust-v2-study-v1":
        raise ValueError("Study protocol must declare schema_version robust-v2-study-v1")
    if payload.get("status") not in {"working", "approved"}:
        raise ValueError("Study protocol status must be working or approved")
    required_strings = ("lineage_id", "parent_lineage_id")
    missing = [name for name in required_strings if not isinstance(payload.get(name), str)]
    if missing:
        raise ValueError(f"Study protocol requires string fields: {', '.join(missing)}")
    for name in required_strings:
        if not _LINEAGE_ID.fullmatch(payload[name]):
            raise ValueError(f"Study protocol {name} is not a valid lineage identifier")
    for name in ("engineering_seed", "evidence_seed"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Study protocol {name} must be a non-negative integer")
    return payload


def _dataset_hashes() -> dict[str, str]:
    paths = [
        ROOT / "data/robust_v2/processed/full_exact/airfoils.parquet",
        ROOT / "data/robust_v2/processed/full_exact/model_points.parquet",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot establish study lineage without processed data: {missing}")
    return {path.name: sha256_file(path) for path in paths}


def _material_evidence_hashes(paths: StudyPaths) -> dict[str, str]:
    candidates = [
        paths.results_root / "FROZEN_MODEL_MANIFEST.json",
        paths.results_root / "splits/full.json",
        paths.results_root / "uncertainty/manufacturing_study.json",
        paths.results_root / "optimisation/optimisation_summary.json",
        paths.results_root / "xfoil_canaries/canary_summary.json",
        paths.results_root / "xfoil_candidates/candidate_validation_summary.json",
        paths.reports_root / "modelling/model_ladder.json",
        paths.reports_root / "viability/advanced_studies.json",
    ]
    return hash_paths([path for path in candidates if path.is_file()], ROOT)


def _write_context(
    paths: StudyPaths,
    protocol_path: Path,
    protocol: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    if paths.context_path.exists():
        if not resume:
            raise FileExistsError("New lineage paths must not already exist; use --resume for the exact lineage")
        existing = json.loads(paths.context_path.read_text(encoding="utf-8"))
        if existing.get("protocol_hash") != sha256_file(protocol_path):
            raise RuntimeError("Refusing to resume a lineage with a different protocol")
        if existing.get("dataset_hashes") != _dataset_hashes():
            raise RuntimeError("Refusing to resume a lineage with changed processed-data hashes")
        return existing
    if paths.results_root.exists() or paths.reports_root.exists():
        raise FileExistsError("New lineage paths must not already exist; use --resume for the exact lineage")
    dataset_hashes = _dataset_hashes()
    context = {
        "schema_version": 1,
        "status": "created",
        "lineage_id": paths.lineage_id,
        "parent_lineage_id": protocol["parent_lineage_id"],
        "protocol_path": protocol_path.as_posix(),
        "protocol_hash": sha256_file(protocol_path),
        "protocol_status": protocol["status"],
        "dataset_hashes": dataset_hashes,
        "git_sha": git_output("rev-parse", "HEAD", cwd=ROOT),
        "created_utc": utc_now(),
    }
    paths.context_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths.context_path, context)
    return context


def run_study(
    protocol_path: Path,
    lineage_id: str,
    *,
    resume: bool,
    from_stage: str = "preflight",
    to_stage: str = "report",
) -> int:
    if from_stage not in STAGES or to_stage not in STAGES:
        raise ValueError(f"Stages must be one of: {', '.join(STAGES)}")
    if STAGES.index(from_stage) > STAGES.index(to_stage):
        raise ValueError("--from must not follow --to")
    protocol_path = Path(protocol_path).resolve()
    protocol = _protocol(protocol_path)
    if protocol["lineage_id"] != lineage_id:
        raise ValueError("The protocol lineage_id must match --lineage-id")
    paths = study_paths(lineage_id)
    if resume and not paths.context_path.is_file():
        raise FileNotFoundError("Cannot resume a lineage without its immutable study context")
    context = _write_context(paths, protocol_path, protocol, resume=resume)
    lineage_metadata = {
        "lineage_id": context["lineage_id"],
        "parent_lineage_id": context["parent_lineage_id"],
        "lineage_protocol_hash": context["protocol_hash"],
        "lineage_dataset_hashes": context["dataset_hashes"],
        "engineering_seed": protocol["engineering_seed"],
        "evidence_seed": protocol["evidence_seed"],
    }
    selected_stages = STAGES[STAGES.index(from_stage) : STAGES.index(to_stage) + 1]
    phase_ids = {STAGE_PHASES[stage] for stage in selected_stages}
    selected_phase_ids = None if selected_stages == STAGES else phase_ids
    exit_code = run_viability(
        resume=resume,
        results_root=paths.results_root,
        reports_root=paths.reports_root,
        lineage_metadata=lineage_metadata,
        phase_ids=selected_phase_ids,
    )
    state_path = paths.reports_root / "run_state.json"
    report_path = paths.reports_root / "viability/viability_report.json"
    ml_evidence_path = paths.reports_root / "modelling/ml_evidence.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    status = {
        "schema_version": 1,
        "lineage_id": paths.lineage_id,
        "execution_exit_code": exit_code,
        "scientific_assessment": report.get("Executive decision", "not_assessed"),
        "stage_range": {"from": from_stage, "to": to_stage},
        "run_state_hashes": hash_paths([state_path], ROOT) if state_path.is_file() else {},
        "report_hashes": hash_paths([report_path], ROOT) if report_path.is_file() else {},
        "ml_evidence_hashes": (
            hash_paths([ml_evidence_path], ROOT) if ml_evidence_path.is_file() else {}
        ),
        "material_evidence_hashes": _material_evidence_hashes(paths),
        "updated_utc": utc_now(),
    }
    write_json_atomic(paths.status_path, status)
    return exit_code


def study_status(lineage_id: str) -> dict[str, Any]:
    paths = study_paths(lineage_id)
    if not paths.context_path.is_file():
        raise FileNotFoundError(f"No study lineage named {lineage_id}")
    context = json.loads(paths.context_path.read_text(encoding="utf-8"))
    status = json.loads(paths.status_path.read_text(encoding="utf-8")) if paths.status_path.is_file() else {}
    protocol_path = Path(context["protocol_path"])
    if not protocol_path.is_file() or sha256_file(protocol_path) != context.get("protocol_hash"):
        raise RuntimeError("Study protocol evidence is missing or changed")
    if context.get("dataset_hashes") != _dataset_hashes():
        raise RuntimeError("Study processed-data evidence is missing or changed")
    state_path = paths.reports_root / "run_state.json"
    current_state_hashes = hash_paths([state_path], ROOT) if state_path.is_file() else {}
    if status.get("run_state_hashes") != current_state_hashes:
        raise RuntimeError("Study run-state evidence is missing or changed")
    report_path = paths.reports_root / "viability/viability_report.json"
    current_report_hashes = hash_paths([report_path], ROOT) if report_path.is_file() else {}
    if status.get("report_hashes") != current_report_hashes:
        raise RuntimeError("Study report evidence is missing or changed")
    ml_evidence_path = paths.reports_root / "modelling/ml_evidence.json"
    current_ml_evidence_hashes = (
        hash_paths([ml_evidence_path], ROOT) if ml_evidence_path.is_file() else {}
    )
    if status.get("ml_evidence_hashes", {}) != current_ml_evidence_hashes:
        raise RuntimeError("Study ML-evidence artifact is missing or changed")
    if status.get("material_evidence_hashes") != _material_evidence_hashes(paths):
        raise RuntimeError("Study material evidence is missing or changed")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    return {
        "context": context,
        "status": {**status, "scientific_assessment": report.get("Executive decision", "not_assessed")},
    }


def build_study_ml_evidence(protocol_path: Path, lineage_id: str) -> dict[str, Any]:
    """Create the narrow, hash-bound ML gate artifact for an existing lineage."""
    protocol_path = Path(protocol_path).resolve()
    protocol = _protocol(protocol_path)
    if protocol["lineage_id"] != lineage_id:
        raise ValueError("The protocol lineage_id must match --lineage-id")
    paths = study_paths(lineage_id)
    if not paths.context_path.is_file():
        raise FileNotFoundError("ML evidence requires an existing immutable study context")
    context = json.loads(paths.context_path.read_text(encoding="utf-8"))
    if context.get("protocol_hash") != sha256_file(protocol_path):
        raise RuntimeError("Refusing to write ML evidence for a different protocol")
    return build_ml_evidence(
        paths.results_root,
        paths.reports_root,
        ROOT / "data/robust_v2/processed/full_exact/model_points.parquet",
        context["protocol_hash"],
    )


def build_study_surrogate_stress(
    lineage_id: str,
    *,
    pool: int = 48,
    refine_top: int = 6,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Run the adversarial surrogate-vs-XFOIL disagreement search for a lineage.

    This realises the reviewer-directed worst-case probe: it searches the
    development CST design box for the aerofoil that maximises the discrepancy
    between the frozen surrogate ensemble and XFOIL, and relates the observed
    disagreement to the calibrated trust domain. XFOIL is a hard external oracle;
    no surrogate proxy is ever substituted for a direct-solver evaluation.
    """
    import numpy as np
    import pandas as pd

    from robust_airfoil.optimisation.problems import development_bounds
    from robust_airfoil.sources.xfoil import locate_xfoil
    from robust_airfoil.surrogate_stress import run_surrogate_stress

    paths = study_paths(lineage_id)
    if not paths.context_path.is_file():
        raise FileNotFoundError("Surrogate stress requires an existing immutable study context")
    ensemble_root = paths.results_root / "full_ensemble"
    if not ensemble_root.is_dir():
        raise FileNotFoundError("Surrogate stress requires the frozen ensemble; run the model phase first")
    executable = locate_xfoil(ROOT)
    if executable is None:
        raise RuntimeError("Surrogate stress requires a locatable XFOIL executable")

    data_root = ROOT / "data/robust_v2/processed/full_exact"
    airfoils = pd.read_parquet(data_root / "airfoils.parquet")
    points = pd.read_parquet(data_root / "model_points.parquet")
    split = json.loads((paths.results_root / "splits/full.json").read_text(encoding="utf-8"))
    development_ids = {str(r["airfoil_id"]) for r in split["records"] if r["split"] == "development"}
    development_points: pd.DataFrame = points.loc[
        points["airfoil_id"].astype(str).isin(sorted(development_ids))
    ].copy()
    param_cols = (
        [f"lower_weight_{i}" for i in range(5)]
        + [f"upper_weight_{i}" for i in range(5)]
        + ["leading_edge_weight", "TE_thickness"]
    )
    bounds = development_bounds(airfoils[param_cols].to_numpy(float))
    reference_name = (
        "naca2412" if "naca2412" in set(airfoils["airfoil_id"]) else str(airfoils.iloc[0]["airfoil_id"])
    )
    row = airfoils.loc[airfoils["airfoil_id"] == reference_name].iloc[0]
    reference_parameters = np.asarray([row[c] for c in param_cols], dtype=float)
    calibration = json.loads(
        (paths.results_root / "calibration/trust_calibration.json").read_text(encoding="utf-8")
    )
    return run_surrogate_stress(
        ensemble_root,
        development_points,
        calibration,
        bounds,
        reference_parameters,
        executable,
        paths.results_root / "surrogate_stress",
        pool=pool,
        refine_top=refine_top,
        timeout_seconds=timeout_seconds,
    )
