from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import traceback
import zipfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import torch

from robust_airfoil.benchmarks.neuralfoil import benchmark_neuralfoil_from_coordinates
from robust_airfoil.config import (
    OptimisationConfig,
    TuningConfig,
    UncertaintyConfig,
    load_all_configs,
    load_sources_config,
    load_yaml,
)
from robust_airfoil.constants import (
    CONFIG_ROOT,
    FROZEN_MANIFEST_PATH,
    REPORTS_ROOT,
    RESULTS_ROOT,
    ROOT,
    RUN_STATE_PATH,
)
from robust_airfoil.data.builder import build_long_form_dataset
from robust_airfoil.data.legacy import write_legacy_audit
from robust_airfoil.data.mapping import build_name_index, canonical_name
from robust_airfoil.data.splits import (
    assert_locked_test_access,
    make_cluster_splits,
    write_split_manifest,
)
from robust_airfoil.geometry.cst import reconstruct_cst
from robust_airfoil.hashing import hash_paths, sha256_file
from robust_airfoil.logging_utils import append_jsonl, configure_phase_logging
from robust_airfoil.modelling.baselines import fit_baselines
from robust_airfoil.modelling.calibrate import apply_trust_model, calibrate_trust_model
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS, TARGET_COLUMNS
from robust_airfoil.modelling.ensemble import load_member, predict_ensemble, train_ensemble
from robust_airfoil.modelling.evaluate import regression_metrics
from robust_airfoil.modelling.freeze import freeze_artifacts, verify_frozen_artifacts
from robust_airfoil.modelling.train import overfit_one_batch, train_model
from robust_airfoil.modelling.tune import tune_model
from robust_airfoil.optimisation.problems import development_bounds
from robust_airfoil.optimisation.study import run_optimisation_studies
from robust_airfoil.provenance import (
    environment_record,
    evidence_as_dict,
    file_evidence,
    git_output,
    utc_now,
    write_json_atomic,
)
from robust_airfoil.recovery import discover_candidates, write_recovery_csv
from robust_airfoil.reports.viability import build_viability_report
from robust_airfoil.run_state import PhaseStatus, RunState
from robust_airfoil.sources.airfoiltools import AirfoilToolsClient
from robust_airfoil.sources.snapshot import acquire_pinned_snapshot
from robust_airfoil.sources.uiuc import acquire_uiuc_zip, extract_zip_safely
from robust_airfoil.sources.xfoil import locate_xfoil
from robust_airfoil.uncertainty.study import run_manufacturing_study
from robust_airfoil.validation.xfoil_runner import run_canary_suite, run_candidate_validation

_ACTIVE_LINEAGE_METADATA: dict[str, Any] = {}


@contextmanager
def _lineage_paths(
    results_root: Path | None,
    reports_root: Path | None,
    lineage_metadata: dict[str, Any] | None,
) -> Any:
    """Temporarily bind pipeline-owned evidence paths to one immutable lineage."""
    if (results_root is None) != (reports_root is None):
        raise ValueError("Lineage results_root and reports_root must be supplied together")
    if results_root is None:
        yield
        return
    assert reports_root is not None
    lock_path = ROOT / ".robust-airfoil-pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            owner_pid = int(lock_path.read_text(encoding="ascii").strip())
            os.kill(owner_pid, 0)
        except (OSError, ValueError):
            lock_path.unlink(missing_ok=True)
            lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            raise RuntimeError("Another isolated robust-airfoil lineage is already running")
    try:
        os.write(lock_descriptor, str(os.getpid()).encode("ascii"))
    except Exception:
        os.close(lock_descriptor)
        lock_path.unlink(missing_ok=True)
        raise
    global RESULTS_ROOT, REPORTS_ROOT, RUN_STATE_PATH, FROZEN_MANIFEST_PATH, _ACTIVE_LINEAGE_METADATA
    previous = RESULTS_ROOT, REPORTS_ROOT, RUN_STATE_PATH, FROZEN_MANIFEST_PATH, _ACTIVE_LINEAGE_METADATA
    RESULTS_ROOT = Path(results_root)
    REPORTS_ROOT = Path(reports_root)
    RUN_STATE_PATH = REPORTS_ROOT / "run_state.json"
    FROZEN_MANIFEST_PATH = RESULTS_ROOT / "FROZEN_MODEL_MANIFEST.json"
    _ACTIVE_LINEAGE_METADATA = dict(lineage_metadata or {})
    try:
        yield
    finally:
        RESULTS_ROOT, REPORTS_ROOT, RUN_STATE_PATH, FROZEN_MANIFEST_PATH, _ACTIVE_LINEAGE_METADATA = previous
        os.close(lock_descriptor)
        lock_path.unlink(missing_ok=True)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> Path:
    write_json_atomic(path, payload)
    return path


def _phase(
    state: RunState,
    phase_id: str,
    name: str,
    command: str,
    function: Callable[[], tuple[dict[str, Any], list[Path], str]],
    *,
    resume: bool,
    input_paths: list[Path] | None = None,
) -> tuple[dict[str, Any], str]:
    logger, log_path = configure_phase_logging(REPORTS_ROOT, phase_id)
    provenance_paths = [
        *sorted(CONFIG_ROOT.glob("*.yaml")),
        *sorted((ROOT / "src/robust_airfoil").rglob("*.py")),
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    config_hashes = hash_paths([path for path in provenance_paths if path.is_file()], ROOT)
    inputs = sorted(path for path in (input_paths or []) if path.is_file())
    input_hashes = hash_paths(inputs, ROOT)
    if resume and state.can_skip(phase_id, config_hashes, input_hashes):
        record = state.phases[phase_id]
        logger.info("skipped: validated hashes for %s", name)
        append_jsonl(REPORTS_ROOT, phase_id, "skipped", reason="validated_config_input_output_hashes")
        return record.summary, record.status
    state.begin(
        phase_id,
        name,
        git_sha=git_output("rev-parse", "HEAD", cwd=ROOT),
        config_hashes=config_hashes,
        input_hashes=input_hashes,
        command=command,
        log_path=str(log_path.relative_to(ROOT)),
    )
    append_jsonl(REPORTS_ROOT, phase_id, "started", name=name)
    try:
        summary, outputs, status = function()
        phase_status: PhaseStatus = status  # type: ignore[assignment]
        state.finish(phase_id, phase_status, summary, outputs)
        logger.info("%s: %s", status, summary)
        append_jsonl(REPORTS_ROOT, phase_id, "finished", status=status, summary=summary)
        return summary, status
    except Exception as exc:
        logger.exception("phase failed")
        append_jsonl(REPORTS_ROOT, phase_id, "failed", error=repr(exc), traceback=traceback.format_exc())
        state.fail(phase_id, "evidence_contract", repr(exc), True)
        return {"error": repr(exc), "traceback": traceback.format_exc()}, "failed"


def _phase00() -> tuple[dict[str, Any], list[Path], str]:
    setup = REPORTS_ROOT / "setup"
    setup.mkdir(parents=True, exist_ok=True)
    targets = [
        ROOT / "data/csv/dataset_12CST_params.csv",
        ROOT / "scripts/train_best_model.py",
        ROOT / "src/models.py",
        ROOT / "src/utils.py",
        *sorted((ROOT / "trained_model").glob("**/*")),
    ]
    targets = [path for path in targets if path.is_file()]
    evidence = [evidence_as_dict(file_evidence(path, ROOT)) for path in targets]
    hashes_path = setup / "historical_hashes.csv"
    with hashes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evidence[0]))
        writer.writeheader()
        writer.writerows(evidence)
    environment = environment_record(ROOT)
    environment_path = _write(setup / "environment.json", environment)
    (setup / "environment.txt").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    freeze_result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    if freeze_result.returncode != 0 or not freeze_result.stdout.strip():
        freeze_result = subprocess.run(
            ["uv", "pip", "freeze", "--python", sys.executable],
            capture_output=True,
            text=True,
            check=False,
        )
    if freeze_result.returncode != 0 or not freeze_result.stdout.strip():
        raise RuntimeError(
            "unable to capture installed-package provenance: "
            f"return_code={freeze_result.returncode}, stderr={freeze_result.stderr.strip()}"
        )
    freeze_path = setup / "package_freeze.txt"
    freeze_path.write_text(freeze_result.stdout, encoding="utf-8")
    return {
        "historical_files_hashed": len(evidence),
        "environment": environment,
        "installed_packages": len(freeze_result.stdout.splitlines()),
    }, [hashes_path, environment_path, freeze_path], "passed"


def _phase02() -> tuple[dict[str, Any], list[Path], str]:
    output = RESULTS_ROOT / "legacy" / "legacy_audit.json"
    audit = write_legacy_audit(ROOT / "data/csv/dataset_12CST_params.csv", output)
    discrepancies = {
        "confirmed": [
            "train_best_model.py unpacks compare_physical_metrics as three values but utility returns one dictionary",
            "physics_aware_tuner.py passes unsupported use_distance_features argument",
            "physics_aware_tuner.py masked evaluation unpacks a three-item batch twice, second time into two values",
            "reported full dataset count 1569 conflicts with committed CSV row count 1568",
            "reported full-dataset StandardScaler conflicts with train-only RobustScaler model artifact",
            "random final split is not geometry-cluster leakage resistant",
        ],
        "exact_reproduction_status": "held",
        "reason": "checked-in historical training entrypoints contain confirmed runtime defects and omit exact split indices/scaler provenance",
    }
    discrepancy_path = _write(REPORTS_ROOT / "modelling" / "v1_discrepancies.json", discrepancies)
    return {"audit": audit, **discrepancies}, [output, discrepancy_path], "held"


def _phase03() -> tuple[dict[str, Any], list[Path], str]:
    home = Path.home()
    roots = [ROOT, ROOT.parent, home / "Documents", home / "Downloads", home / "Desktop", home / "OneDrive"]
    for drive in (Path("D:/"), Path("E:/")):
        if drive.exists():
            roots.append(drive)
    candidates = discover_candidates(roots, maximum_files=60_000)
    output = REPORTS_ROOT / "data" / "local_recovery_candidates.csv"
    write_recovery_csv(output, candidates)
    summary = {"roots_scanned": [str(path) for path in roots], "candidate_count": len(candidates), "accepted_raw_count": 0, "note": "Discovery only; no unsafe pickle deserialisation and no historical source silently substituted."}
    summary_path = _write(REPORTS_ROOT / "data" / "local_recovery_summary.json", summary)
    return summary, [output, summary_path], "passed"


def _download_xfoil(url: str) -> dict[str, Any]:
    target_dir = ROOT / "tools" / "xfoil"
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / "XFOIL6.99.zip"
    try:
        response = httpx.get(url, timeout=90, follow_redirects=True)
        response.raise_for_status()
        archive_path.write_bytes(response.content)
        if zipfile.is_zipfile(archive_path):
            extract_zip_safely(archive_path, target_dir)
        executable = locate_xfoil(ROOT)
        return {"status": "available" if executable else "downloaded_no_executable_found", "path": str(executable) if executable else None, "archive_sha256": sha256_file(archive_path)}
    except Exception as exc:
        return {"status": "unavailable", "reason": repr(exc)}


def _phase04() -> tuple[dict[str, Any], list[Path], str]:
    config = load_sources_config()
    outputs: list[Path] = []
    source_report: dict[str, Any] = {}
    uiuc_archive = ROOT / "data/robust_v2/raw/uiuc_current/coord_seligFmt.zip"
    uiuc_manifest = ROOT / "data/robust_v2/manifests/uiuc_current.json"
    try:
        source_report["uiuc"] = acquire_uiuc_zip(config.uiuc.zip_url, uiuc_archive, uiuc_manifest)
        extracted = extract_zip_safely(uiuc_archive, ROOT / "data/robust_v2/raw/uiuc_current/coordinates")
        source_report["uiuc"]["extracted_files"] = len(extracted)
        outputs.append(uiuc_manifest)
    except Exception as exc:
        source_report["uiuc"] = {"status": "source_unavailable", "reason": repr(exc)}
    snapshot_dir = ROOT / "data/robust_v2/raw/airfoiltools_snapshot/repo"
    snapshot_manifest = ROOT / "data/robust_v2/manifests/airfoiltools_snapshot.json"
    source_report["snapshot"] = acquire_pinned_snapshot(config.snapshot.repo_url, config.snapshot.commit, snapshot_dir, snapshot_manifest)
    outputs.append(snapshot_manifest)
    live_client = AirfoilToolsClient(
        config.airfoiltools_live.csv_url_template,
        config.airfoiltools_live.robots_url,
        config.airfoiltools_live.canary_key,
        config.airfoiltools_live.require_contact_env,
        config.airfoiltools_live.minimum_delay_seconds,
        config.airfoiltools_live.read_timeout_seconds,
        config.airfoiltools_live.stop_on_status,
        config.airfoiltools_live.max_transient_attempts,
        ROOT / "data/robust_v2/raw/airfoiltools_live",
    )
    source_report["live"] = asdict(live_client.probe_canary())
    source_report["xfoil"] = _download_xfoil(config.xfoil.windows_zip_url)
    report_path = _write(REPORTS_ROOT / "data" / "source_acquisition.json", source_report)
    outputs.append(report_path)
    return source_report, outputs, "passed"


def _exact_source_paths() -> tuple[list[Path], list[Path]]:
    config = load_sources_config()
    geometry_paths = sorted((ROOT / config.continuity_geometry_dir).glob("*.dat"))
    polar_root = ROOT / "data/robust_v2/raw/airfoiltools_snapshot/repo/dat/case-dat"
    polar_paths = sorted(polar_root.glob("*-il-1000000.csv"))
    geometry_index = build_name_index(geometry_paths)
    exact_polars = [path for path in polar_paths if canonical_name(path.name) in geometry_index and len(geometry_index[canonical_name(path.name)]) == 1]
    return exact_polars, geometry_paths


def _phase05_08() -> tuple[dict[str, Any], list[Path], str]:
    polars, geometries = _exact_source_paths()
    output = ROOT / "data/robust_v2/processed/full_exact"
    summary = build_long_form_dataset(polars, geometries, output, "third_party_airfoiltools_snapshot")
    summary["candidate_exact_polars"] = len(polars)
    summary_path = _write(REPORTS_ROOT / "data" / "dataset_summary.json", summary)
    outputs = [output / name for name in ("airfoils.parquet", "polar_points.parquet", "model_points.parquet", "mappings.json", "rejected.json")]
    outputs.append(summary_path)
    parse_fraction = (summary["candidate_exact_polars"] - summary["rejected_records"]) / max(summary["candidate_exact_polars"], 1)
    summary["parse_success_fraction"] = parse_fraction
    status = "passed" if summary["mapped_airfoils"] >= 20 and summary["joint_model_points"] >= 500 and parse_fraction >= 0.95 else "held"
    return summary, outputs, status


def _select_airfoils(airfoils: pd.DataFrame, count: int, seed: int) -> list[str]:
    if len(airfoils) <= count:
        return sorted(airfoils["airfoil_id"].astype(str).tolist())
    ordered = airfoils.sort_values(["maximum_thickness", "section_area", "airfoil_id"]).reset_index(drop=True)
    positions = np.linspace(0, len(ordered) - 1, count, dtype=int)
    selected = ordered.iloc[positions]["airfoil_id"].astype(str).tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(selected)
    return selected


def _pilot_splits(airfoils: pd.DataFrame, seed: int) -> pd.DataFrame:
    clusters = np.asarray(sorted(airfoils["cluster_id"].unique()))
    if len(clusters) < 25:
        return make_cluster_splits(airfoils, seed)
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    assignment = {int(cluster): ("development" if i < 15 else "calibration" if i < 20 else "locked_test") for i, cluster in enumerate(clusters)}
    result = airfoils[["airfoil_id", "cluster_id"]].copy()
    result["split"] = result["cluster_id"].apply(lambda value: assignment[int(value)])
    return result


def _train_validation_from_development(points: pd.DataFrame, airfoils: pd.DataFrame, splits: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    development_ids = splits.loc[splits["split"] == "development", "airfoil_id"]
    development_airfoils = airfoils[airfoils["airfoil_id"].isin(development_ids)]
    clusters = np.asarray(sorted(development_airfoils["cluster_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    validation_count = max(1, int(round(0.2 * len(clusters))))
    validation_clusters = set(clusters[:validation_count])
    validation_ids = set(development_airfoils.loc[development_airfoils["cluster_id"].isin(validation_clusters), "airfoil_id"])
    validation = points[points["airfoil_id"].isin(validation_ids)].copy()
    train = points[points["airfoil_id"].isin(set(development_ids) - validation_ids)].copy()
    return train, validation


def _run_model_stage(label: str, count: int, seed: int, maximum_epochs: int = 180) -> dict[str, Any]:
    cached_summary = RESULTS_ROOT / label / "stage_summary.json"
    if cached_summary.is_file() and (RESULTS_ROOT / label / "mlp/checkpoint.pt").is_file() and (RESULTS_ROOT / label / "mlp/scaling.joblib").is_file():
        cached = _json(cached_summary)
        if cached.get("gate_passed"):
            return cached
    data_root = ROOT / "data/robust_v2/processed/full_exact"
    airfoils = pd.read_parquet(data_root / "airfoils.parquet")
    points = pd.read_parquet(data_root / "model_points.parquet")
    selected_ids = _select_airfoils(airfoils, count, seed)
    selected_airfoils = airfoils[airfoils["airfoil_id"].isin(selected_ids)].copy()
    selected_points = points[points["airfoil_id"].isin(selected_ids)].copy()
    splits = _pilot_splits(selected_airfoils, seed) if count == 25 else make_cluster_splits(selected_airfoils, seed)
    split_path = RESULTS_ROOT / f"splits/{label}.json"
    write_split_manifest(splits, split_path, seed)
    train, validation = _train_validation_from_development(selected_points, selected_airfoils, splits, seed)
    output_dir = RESULTS_ROOT / label
    overfit = overfit_one_batch(train, seed, steps=600)
    baselines = fit_baselines(train, validation, seed)
    _write(output_dir / "baselines.json", baselines.metrics)
    model = train_model(train, validation, output_dir / "mlp", seed=seed, maximum_epochs=maximum_epochs)
    dummy = baselines.metrics["dummy"]
    cl_improvement = 1 - float(model["metrics"]["cl"]["macro_mse"]) / max(float(dummy["cl"]["macro_mse"]), 1e-12)
    cd_improvement = 1 - float(model["metrics"]["log_cd"]["macro_mse"]) / max(float(dummy["log_cd"]["macro_mse"]), 1e-12)
    gate = bool(overfit["passed"] and cl_improvement >= 0.20 and cd_improvement >= 0.20)
    summary = {
        "selected_airfoils": len(selected_airfoils), "joint_points": len(selected_points),
        "development_train_airfoils": train["airfoil_id"].nunique(), "development_validation_airfoils": validation["airfoil_id"].nunique(),
        "overfit": overfit, "baselines": baselines.metrics, "model": model,
        "cl_improvement_over_dummy": cl_improvement, "log_cd_improvement_over_dummy": cd_improvement, "gate_passed": gate,
        "split_manifest": str(split_path.relative_to(ROOT)),
    }
    _write(output_dir / "stage_summary.json", summary)
    return summary


def _phase12_16() -> tuple[dict[str, Any], list[Path], str]:
    model_ladder_path = REPORTS_ROOT / "modelling/model_ladder.json"
    cached_outputs = [
        model_ladder_path,
        RESULTS_ROOT / "pilot/stage_summary.json",
        RESULTS_ROOT / "expansion_200/stage_summary.json",
    ]
    if FROZEN_MANIFEST_PATH.is_file() and all(path.is_file() for path in cached_outputs):
        frozen = _json(FROZEN_MANIFEST_PATH)
        frozen_artifacts = {
            name: Path(path)
            for name, path in frozen.get("artifact_paths", {}).items()
        }
        verify_frozen_artifacts(FROZEN_MANIFEST_PATH, frozen_artifacts)
        cached_summary = _json(model_ladder_path)
        if (
            cached_summary.get("expansion_200", {}).get("gate_passed")
            and cached_summary.get("tuning", {}).get("status") == "completed"
            and (RESULTS_ROOT / "locked_test/metrics.json").is_file()
        ):
            return cached_summary, cached_outputs, "passed"
    engineering_seed = int(_ACTIVE_LINEAGE_METADATA.get("engineering_seed", 20260824))
    pilot = _run_model_stage("pilot", 25, engineering_seed, 180)
    expansion = _run_model_stage("expansion_200", 200, engineering_seed + 1, 220)
    full: dict[str, Any] = {"status": "not_started"}
    tuning: dict[str, Any] = {"status": "not_started"}
    calibration: dict[str, Any] = {"status": "not_started"}
    if expansion["gate_passed"]:
        data_root = ROOT / "data/robust_v2/processed/full_exact"
        airfoils = pd.read_parquet(data_root / "airfoils.parquet")
        points = pd.read_parquet(data_root / "model_points.parquet")
        splits = make_cluster_splits(airfoils, engineering_seed)
        split_path = RESULTS_ROOT / "splits/full.json"
        write_split_manifest(splits, split_path, engineering_seed)
        development_ids = splits.loc[splits["split"] == "development", "airfoil_id"].astype(str).tolist()
        calibration_ids = splits.loc[splits["split"] == "calibration", "airfoil_id"].astype(str).tolist()
        locked_ids = splits.loc[splits["split"] == "locked_test", "airfoil_id"].astype(str).tolist()
        development_airfoils = airfoils[airfoils["airfoil_id"].isin(development_ids)].copy()
        development_points = points[points["airfoil_id"].isin(development_ids)].copy()
        development_points_with_clusters = development_points.merge(
            development_airfoils[["airfoil_id", "cluster_id"]],
            on="airfoil_id",
            how="left",
            validate="many_to_one",
        )
        baseline_train, baseline_validation = _train_validation_from_development(points, airfoils, splits, engineering_seed)
        full = train_model(baseline_train, baseline_validation, RESULTS_ROOT / "full_baseline/mlp", maximum_epochs=260)
        tuning_config = TuningConfig.model_validate(load_yaml(CONFIG_ROOT / "tuning.yaml"))
        study = tune_model(development_points, development_airfoils, RESULTS_ROOT / "tuning", tuning_config)
        best_epochs = [int(value) for value in study.best_trial.user_attrs.get("best_epochs", [])]
        selected_epochs = max(5, int(round(float(np.median(best_epochs))))) if best_epochs else 24
        tuning = {
            "status": "completed",
            "trials": len(study.trials),
            "completed_trials": sum(trial.state.name == "COMPLETE" for trial in study.trials),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "selected_epochs": selected_epochs,
            "grouped_folds": tuning_config.grouped_folds,
        }
        tuning_path = _write(RESULTS_ROOT / "tuning/summary.json", tuning)
        ensemble_root = RESULTS_ROOT / "full_ensemble"
        ensemble = train_ensemble(
            development_points_with_clusters,
            ensemble_root,
            study.best_params,
            selected_epochs,
            engineering_seed,
            members=5,
        )
        first_model, first_scaling = load_member(ensemble_root / "member_000")
        del first_model
        calibration_points = points[points["airfoil_id"].isin(calibration_ids)].copy()
        calibration_predictions = predict_ensemble(ensemble_root, calibration_points)
        calibration_path = RESULTS_ROOT / "calibration/trust_calibration.json"
        calibration = calibrate_trust_model(development_points, calibration_predictions, first_scaling, calibration_path)
        calibration_predictions_path = RESULTS_ROOT / "calibration/calibration_predictions.parquet"
        calibration_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_predictions.to_parquet(calibration_predictions_path, index=False)
        artifacts: dict[str, Path] = {
            "airfoils": data_root / "airfoils.parquet",
            "model_points": data_root / "model_points.parquet",
            "split_manifest": split_path,
            "tuning_summary": tuning_path,
            "ensemble_manifest": ensemble_root / "ensemble_manifest.json",
            "trust_calibration": calibration_path,
            "calibration_predictions": calibration_predictions_path,
        }
        for index in range(5):
            artifacts[f"member_{index:03d}_checkpoint"] = ensemble_root / f"member_{index:03d}/checkpoint.pt"
            artifacts[f"member_{index:03d}_scaling"] = ensemble_root / f"member_{index:03d}/scaling.joblib"
        freeze_artifacts(
            FROZEN_MANIFEST_PATH,
            artifacts,
            {
                "git_sha": git_output("rev-parse", "HEAD", cwd=ROOT),
                "dataset_hash": sha256_file(data_root / "model_points.parquet"),
                "split_hash": sha256_file(split_path),
                "tuning_best_value": study.best_value,
                "selected_hyperparameters": study.best_params,
                "training_seeds": [engineering_seed + index for index in range(5)],
                "selected_epochs": selected_epochs,
                "feature_columns": FEATURE_COLUMNS,
                "target_columns": TARGET_COLUMNS,
                **_ACTIVE_LINEAGE_METADATA,
            },
        )
        verify_frozen_artifacts(FROZEN_MANIFEST_PATH, artifacts)
        assert_locked_test_access(FROZEN_MANIFEST_PATH, "locked_test", artifacts)
        locked_points = points[points["airfoil_id"].isin(locked_ids)].copy()
        locked_predictions = predict_ensemble(ensemble_root, locked_points)
        locked_predictions = apply_trust_model(development_points, locked_predictions, first_scaling, calibration)
        locked_dir = RESULTS_ROOT / "locked_test"
        locked_dir.mkdir(parents=True, exist_ok=True)
        locked_predictions_path = locked_dir / "predictions.parquet"
        locked_predictions.to_parquet(locked_predictions_path, index=False)
        full["ensemble"] = {"member_count": ensemble["member_count"], "selected_epochs": selected_epochs}
        full["locked_test_metrics"] = regression_metrics(locked_predictions, {name: f"prediction_{name}" for name in TARGET_COLUMNS})
        full["locked_test_trusted_fraction"] = float(locked_predictions["trusted_domain"].mean())
        full["frozen_manifest_hash"] = sha256_file(FROZEN_MANIFEST_PATH)
        _write(locked_dir / "metrics.json", {
            "metrics": full["locked_test_metrics"],
            "trusted_fraction": full["locked_test_trusted_fraction"],
            "frozen_manifest_hash": full["frozen_manifest_hash"],
            "member_count": 5,
            "evaluated_utc": utc_now(),
        })
    summary = {"pilot": pilot, "expansion_200": expansion, "full_baseline": full, "tuning": tuning, "calibration": calibration}
    output = _write(REPORTS_ROOT / "modelling/model_ladder.json", summary)
    status = "passed" if expansion["gate_passed"] else "held"
    return summary, [output, RESULTS_ROOT / "pilot/stage_summary.json", RESULTS_ROOT / "expansion_200/stage_summary.json"], status


def _phase14_21(model_summary: dict[str, Any]) -> tuple[dict[str, Any], list[Path], str]:
    del model_summary
    if not FROZEN_MANIFEST_PATH.is_file():
        raise RuntimeError("Advanced studies require a verified frozen ensemble")
    frozen = _json(FROZEN_MANIFEST_PATH)
    frozen_artifacts = {
        name: Path(path)
        for name, path in frozen.get("artifact_paths", {}).items()
    }
    verify_frozen_artifacts(FROZEN_MANIFEST_PATH, frozen_artifacts)
    data_root = ROOT / "data/robust_v2/processed/full_exact"
    airfoils = pd.read_parquet(data_root / "airfoils.parquet")
    points = pd.read_parquet(data_root / "model_points.parquet")
    split_payload = _json(RESULTS_ROOT / "splits/full.json")
    splits = pd.DataFrame(split_payload["records"])
    development_ids = splits.loc[splits["split"] == "development", "airfoil_id"].astype(str).tolist()
    development_points = points[points["airfoil_id"].isin(development_ids)].copy()
    development_airfoils = airfoils[airfoils["airfoil_id"].isin(development_ids)].copy()
    reference_name = "naca2412" if "naca2412" in set(airfoils["airfoil_id"]) else str(airfoils.iloc[0]["airfoil_id"])
    row = airfoils.loc[airfoils["airfoil_id"] == reference_name].iloc[0]
    parameters = np.asarray([row[f"lower_weight_{i}"] for i in range(5)] + [row[f"upper_weight_{i}"] for i in range(5)] + [row["leading_edge_weight"], row["TE_thickness"]], dtype=float)
    x = (1 - np.cos(np.linspace(0, np.pi, 201))) / 2
    upper_y, lower_y = reconstruct_cst(parameters, x)
    upper, lower = np.column_stack([x, upper_y]), np.column_stack([x, lower_y])
    ensemble_root = RESULTS_ROOT / "full_ensemble"
    uncertainty_config = UncertaintyConfig.model_validate(load_yaml(CONFIG_ROOT / "uncertainty.yaml"))
    optimisation_config = OptimisationConfig.model_validate(load_yaml(CONFIG_ROOT / "optimisation.yaml"))
    parameter_columns = [f"lower_weight_{i}" for i in range(5)] + [f"upper_weight_{i}" for i in range(5)] + ["leading_edge_weight", "TE_thickness"]
    development_parameter_std = development_airfoils[parameter_columns].to_numpy(float).std(axis=0, ddof=1)
    calibration = _json(RESULTS_ROOT / "calibration/trust_calibration.json")
    uncertainty_path = RESULTS_ROOT / "uncertainty/manufacturing_study.json"
    uncertainty = run_manufacturing_study(
        parameters,
        development_parameter_std,
        upper,
        lower,
        ensemble_root,
        optimisation_config.ava_baseline.alpha_deg,
        uncertainty_config,
        development_points,
        calibration,
        uncertainty_path,
    )
    bounds = development_bounds(airfoils[[f"lower_weight_{i}" for i in range(5)] + [f"upper_weight_{i}" for i in range(5)] + ["leading_edge_weight", "TE_thickness"]].to_numpy(float))
    optimisation = run_optimisation_studies(
        parameters,
        bounds,
        development_points,
        ensemble_root,
        calibration,
        optimisation_config,
        uncertainty_config,
        upper,
        lower,
        RESULTS_ROOT / "optimisation",
    )
    coordinates = np.vstack([upper[::-1], lower[1:]])
    neuralfoil = asdict(
        benchmark_neuralfoil_from_coordinates(
            coordinates, np.asarray([0.0, 2.0, 4.0, 6.0, 8.0])
        )
    )
    neuralfoil = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in neuralfoil.items()}
    xfoil = locate_xfoil(ROOT)
    xfoil_status: dict[str, Any]
    if xfoil is None:
        xfoil_status = {"status": "unavailable", "comparison": "not_run"}
    else:
        xfoil_status = run_canary_suite(
            xfoil,
            ROOT,
            RESULTS_ROOT / "xfoil_canaries",
            load_sources_config().xfoil.timeout_seconds,
        )
        candidate_validation = run_candidate_validation(
            xfoil,
            optimisation["five_design_comparison"],
            ensemble_root,
            development_points,
            calibration,
            uncertainty_config,
            optimisation_config,
            RESULTS_ROOT / "xfoil_candidates",
            load_sources_config().xfoil.timeout_seconds,
        )
        xfoil_status["candidate_validation"] = candidate_validation
        all_xfoil_completed = bool(
            xfoil_status["all_cases_completed"] and candidate_validation["all_cases_completed"]
        )
        xfoil_status["status"] = "completed" if all_xfoil_completed else "partial"
        xfoil_status["comparison"] = "completed" if all_xfoil_completed else "partial"
    summary = {
        "reference": reference_name,
        "uncertainty": uncertainty,
        "optimisation": optimisation,
        "neuralfoil": neuralfoil,
        "xfoil": xfoil_status,
    }
    output = _write(REPORTS_ROOT / "viability/advanced_studies.json", summary)
    bounds_respected = all(level["bound_respected"] for level in uncertainty["levels"].values())
    full_runs = optimisation["nsga2"]["full"]["runs"]
    xfoil_complete = bool(
        xfoil_status.get("all_cases_completed")
        and xfoil_status.get("candidate_validation", {}).get("all_cases_completed")
    )
    status = "passed" if bounds_respected and len(full_runs) == optimisation_config.full_nsga2.independent_seeds and xfoil_complete else "held"
    return summary, [output, uncertainty_path, RESULTS_ROOT / "optimisation/optimisation_summary.json", RESULTS_ROOT / "xfoil_canaries/canary_summary.json", RESULTS_ROOT / "xfoil_candidates/candidate_validation_summary.json"], status


def _ml_evidence_gate(path: Path) -> dict[str, Any]:
    """Validate ML gate evidence without allowing it to make the overall GO decision."""
    result: dict[str, Any] = {"path": str(path.relative_to(ROOT)), "status": "missing"}
    if not path.is_file():
        result["reason"] = "The required ML evidence artifact has not been produced."
        return result
    try:
        payload = _json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result.update(status="invalid", reason=f"ML evidence could not be parsed: {exc}")
        return result
    if not isinstance(payload, dict):
        result.update(status="invalid", reason="ML evidence must be a JSON object.")
        return result
    if payload.get("schema_version") != "robust-v2-ml-evidence-v1":
        result.update(status="invalid", reason="ML evidence has an unsupported schema version.")
        return result
    if payload.get("status") != "passed" or payload.get("ml_gate_passed") is not True:
        result.update(status="invalid", reason="ML evidence does not record a passed ML acceptance gate.")
        return result
    if payload.get("scientific_go_granted") is not False:
        result.update(
            status="invalid",
            reason="ML evidence must not grant the overall scientific decision by itself.",
        )
        return result
    lineage_protocol_hash = _ACTIVE_LINEAGE_METADATA.get("lineage_protocol_hash")
    if not isinstance(lineage_protocol_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", lineage_protocol_hash
    ):
        result.update(
            status="invalid",
            reason="ML evidence can only be accepted inside a lineage with a pinned protocol hash.",
        )
        return result
    if not FROZEN_MANIFEST_PATH.is_file():
        result.update(status="invalid", reason="The frozen-model manifest is missing.")
        return result
    model_points_path = ROOT / "data/robust_v2/processed/full_exact/model_points.parquet"
    if not model_points_path.is_file():
        result.update(status="invalid", reason="The frozen model input dataset is missing.")
        return result
    expected_hashes = {
        "model_sha256": sha256_file(FROZEN_MANIFEST_PATH),
        "protocol_hash": lineage_protocol_hash,
        "input_sha256": sha256_file(model_points_path),
    }
    for field in ("model_sha256", "protocol_hash", "input_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            result.update(status="invalid", reason=f"ML evidence has an invalid {field}.")
            return result
        if value != expected_hashes[field]:
            result.update(
                status="invalid",
                reason=f"ML evidence {field} does not match the pinned lineage evidence.",
            )
            return result
    if not isinstance(payload.get("model_id"), str) or not payload["model_id"].strip():
        result.update(status="invalid", reason="ML evidence has no model identifier.")
        return result
    target_metrics = payload.get("target_metrics")
    if not isinstance(target_metrics, dict):
        result.update(status="invalid", reason="ML evidence has no target metrics object.")
        return result
    for target in ("cl", "log_cd"):
        metrics = target_metrics.get(target)
        if not isinstance(metrics, dict):
            result.update(status="invalid", reason=f"ML evidence is missing {target} target metrics.")
            return result
        try:
            model_error = float(metrics["model_macro_error"])
            dummy_error = float(metrics["dummy_macro_error"])
            improvement = float(metrics["macro_error_improvement_vs_dummy"])
        except (KeyError, TypeError, ValueError):
            result.update(status="invalid", reason=f"ML evidence has incomplete {target} metrics.")
            return result
        expected_improvement = (dummy_error - model_error) / dummy_error if dummy_error > 0 else None
        if (
            not np.isfinite([model_error, dummy_error, improvement]).all()
            or model_error < 0
            or dummy_error <= 0
            or model_error > dummy_error
            or improvement < 0.20
            or expected_improvement is None
            or not np.isclose(improvement, expected_improvement, rtol=1e-9, atol=1e-12)
        ):
            result.update(
                status="invalid",
                reason=(
                    f"ML evidence does not demonstrate the required 20% {target} "
                    "macro-error improvement over Dummy."
                ),
            )
            return result
    return {
        "path": str(path.relative_to(ROOT)),
        "status": "passed",
        "model_id": payload["model_id"],
        "model_sha256": payload["model_sha256"],
        "protocol_hash": payload["protocol_hash"],
        "input_sha256": payload["input_sha256"],
        "target_metrics": {target: target_metrics[target] for target in ("cl", "log_cd")},
        "scientific_go_granted": False,
    }


def _phase22(all_results: dict[str, Any]) -> tuple[dict[str, Any], list[Path], str]:
    model = all_results.get("model", {})
    advanced = all_results.get("advanced", {})
    model_status = all_results.get("model_status", "pending")
    advanced_status = all_results.get("advanced_status", "pending")
    model_evidence_available = model_status == "passed"
    advanced_evidence_available = advanced_status == "passed"
    data = all_results.get("data", {})
    source = all_results.get("source", {})
    uncertainty = advanced.get("uncertainty", {})
    optimisation = advanced.get("optimisation", {})
    xfoil = advanced.get("xfoil", {})
    blockers: list[dict[str, str]] = []
    ml_evidence = _ml_evidence_gate(REPORTS_ROOT / "modelling/ml_evidence.json")

    def add_blocker(code: str, severity: str, detail: str, resolution: str) -> None:
        blockers.append(
            {"code": code, "severity": severity, "detail": detail, "resolution": resolution}
        )

    if all_results.get("legacy_status") != "passed":
        add_blocker(
            "v1_exact_reproduction",
            "high",
            "Committed V1 entrypoints are internally inconsistent and exact split/scaler provenance is absent; exact reproduction is not supportable.",
            "Recover the original V1 split IDs, scaler state, dependency lock, and training command, then rerun the frozen V1 comparison before making V1-to-V2 claim language.",
        )
    if not model_evidence_available:
        add_blocker(
            "model_evidence_not_current",
            "high",
            f"Model phase 09-16 is {model_status}; no current verified model evidence is available for assessment.",
            "Repair the failed or incomplete model phase, verify its frozen artifacts, and rerun phases 09-22 before interpreting model performance.",
        )
    elif not bool(model.get("expansion_200", {}).get("gate_passed", False)):
        add_blocker(
            "expanded_learning_signal",
            "critical",
            "The 200-airfoil model gate did not demonstrate the required CL and log(CD) improvement over Dummy.",
            "Inspect grouped residuals and source strata, repair the data/model defect, and rerun phases 09-22.",
        )
    if ml_evidence["status"] != "passed":
        add_blocker(
            "ml_evidence_missing_or_invalid",
            "high",
            str(ml_evidence["reason"]),
            "Produce a lineage-bound ML evidence artifact for the frozen model, with independent-group macro-error comparisons against Dummy for CL and log(CD), then rerun phase 22. An audit report alone cannot grant this gate.",
        )
    calibration_path = RESULTS_ROOT / "calibration/trust_calibration.json"
    if not calibration_path.exists():
        add_blocker(
            "trust_calibration_missing",
            "critical",
            "Calibration and trusted-domain evidence were not produced.",
            "Complete grouped tuning and ensemble training, calibrate on the untouched calibration split, freeze, and rerun locked-test and advanced phases.",
        )
    level_summaries: dict[str, Any] = {}
    predominantly_invalid = False
    for amplitude, level in uncertainty.get("levels", {}).items():
        samples = max(1, int(level.get("samples", 0)))
        invalid_fraction = float(level.get("invalid_geometry", samples)) / samples
        predominantly_invalid |= invalid_fraction > 0.5
        level_summaries[amplitude] = {
            "samples": level.get("samples"),
            "valid_geometry": level.get("valid_geometry"),
            "invalid_geometry": level.get("invalid_geometry"),
            "trust_domain_violations": level.get("trust_domain_violations"),
            "violation_probability": level.get("violation_probability"),
            "bound_respected": level.get("bound_respected"),
            "weighted_cd_distribution": level.get(
                "weighted_cd_distribution_with_violation_penalty", {}
            ),
            "sample_matrix_sha256": level.get("sample_matrix_sha256"),
        }
    if advanced_evidence_available and predominantly_invalid:
        add_blocker(
            "predominantly_invalid_uncertainty",
            "critical",
            "At least one intended manufacturing level generated predominantly invalid geometry.",
            "Redesign the physical perturbation basis or reduce the unsupported manufacturing amplitude, then rerun uncertainty and optimisation without suppressing invalid samples.",
        )
    full_runs = optimisation.get("nsga2", {}).get("full", {}).get("runs", [])
    feasible_full = sum(
        int(np.all(np.asarray(run.get("constraint_values", []), dtype=float) <= 0, axis=1).sum())
        for run in full_runs
        if len(run.get("constraint_values", []))
    )
    if advanced_evidence_available and (not full_runs or feasible_full == 0):
        add_blocker(
            "no_feasible_robust_optimum",
            "critical",
            "The full robust optimisation did not produce a feasible trusted candidate.",
            "Diagnose constraint margins and trust rejections, repair the optimiser or model-domain bounds, and rerun all ten full seeds.",
        )
    agreement = optimisation.get("nsga2", {}).get("full", {}).get("agreement", {})
    if advanced_evidence_available and full_runs and agreement.get("agreement_status") != "pass":
        add_blocker(
            "optimisation_seed_agreement",
            "high",
            "The ten-seed full NSGA-II fronts did not satisfy the predeclared agreement rule.",
            "Increase convergence budget or repair front instability, then rerun all full seeds with the same frozen model and common random samples.",
        )
    candidate_validation = xfoil.get("candidate_validation", {})
    if not advanced_evidence_available:
        add_blocker(
            "advanced_evidence_not_current",
            "high",
            f"Advanced phase 17-21 is {advanced_status}; no current verified uncertainty, optimisation, or XFOIL evidence is available for assessment.",
            "Repair the failed or incomplete advanced phase and rerun phases 17-22 with the verified frozen model before interpreting optimisation or XFOIL results.",
        )
    elif xfoil.get("comparison") != "completed":
        add_blocker(
            "direct_xfoil_incomplete",
            "high",
            "The six-airfoil canary and/or five-design direct-XFOIL campaign is incomplete.",
            "Resolve every failed XFOIL case and rerun nominal, 64 shared perturbations, and adverse-tail validations for all five designs.",
        )
    robust_effects = [
        candidate_validation.get("effect_sizes", {}).get(design_id, {})
        for design_id in (
            "ava_robust_optimum",
            "smooth_correlated_robust_optimum",
            "trust_constrained_pareto_knee",
        )
    ]
    ranking_reversal = any(
        effect.get("nominal_weighted_cd_improvement_vs_reference") is not None
        and float(effect["nominal_weighted_cd_improvement_vs_reference"]) < 0
        and effect.get("shared_perturbation_paired_improvement", {}).get("mean") is not None
        and float(effect["shared_perturbation_paired_improvement"]["mean"]) < 0
        for effect in robust_effects
    )
    if ranking_reversal:
        add_blocker(
            "xfoil_robust_ranking_reversal",
            "high",
            "Direct XFOIL reversed at least one claimed robust improvement both nominally and under paired shared perturbations.",
            "Diagnose surrogate bias for the reversed design, revise the trusted-domain/model formulation, and repeat optimisation and direct validation before making a robust-ranking claim.",
        )

    fundamental_failure = any(blocker["severity"] == "critical" for blocker in blockers)
    if fundamental_failure:
        decision = "NO-GO"
    elif blockers:
        decision = "HOLD"
    else:
        decision = "GO"

    split_path = RESULTS_ROOT / "splits/full.json"
    leakage: dict[str, Any] = {"status": "not_available"}
    if split_path.is_file():
        split_frame = pd.DataFrame(_json(split_path)["records"])
        split_ids = {
            name: set(split_frame.loc[split_frame["split"] == name, "airfoil_id"].astype(str))
            for name in ("development", "calibration", "locked_test")
        }
        split_clusters = {
            name: set(split_frame.loc[split_frame["split"] == name, "cluster_id"].astype(int))
            for name in ("development", "calibration", "locked_test")
        }
        leakage = {
            "status": "passed"
            if all(
                not (split_ids[left] & split_ids[right])
                and not (split_clusters[left] & split_clusters[right])
                for left, right in (
                    ("development", "calibration"),
                    ("development", "locked_test"),
                    ("calibration", "locked_test"),
                )
            )
            else "failed",
            "airfoil_counts": {name: len(values) for name, values in split_ids.items()},
            "cluster_counts": {name: len(values) for name, values in split_clusters.items()},
            "airfoil_overlaps": {
                f"{left}__{right}": sorted(split_ids[left] & split_ids[right])
                for left, right in (
                    ("development", "calibration"),
                    ("development", "locked_test"),
                    ("calibration", "locked_test"),
                )
            },
            "cluster_overlaps": {
                f"{left}__{right}": sorted(split_clusters[left] & split_clusters[right])
                for left, right in (
                    ("development", "calibration"),
                    ("development", "locked_test"),
                    ("calibration", "locked_test"),
                )
            },
            "manifest_sha256": sha256_file(split_path),
        }

    five_designs: list[dict[str, Any]] = []
    for design in optimisation.get("five_design_comparison", []):
        diagnostic = dict(design.get("diagnostics", {}))
        drag_samples = np.asarray(diagnostic.pop("sample_weighted_drag", []), dtype=float)
        if len(drag_samples):
            diagnostic["weighted_drag_distribution"] = {
                "mean": float(np.mean(drag_samples)),
                "standard_deviation": float(np.std(drag_samples, ddof=1)),
                "q05": float(np.quantile(drag_samples, 0.05)),
                "q50": float(np.quantile(drag_samples, 0.50)),
                "q95": float(np.quantile(drag_samples, 0.95)),
            }
        design_cases = [
            record
            for record in candidate_validation.get("case_records", [])
            if record.get("design_id") == design.get("design_id")
        ]
        five_designs.append(
            {
                **{key: value for key, value in design.items() if key != "diagnostics"},
                "diagnostics": diagnostic,
                "direct_xfoil": {
                    "summary": candidate_validation.get("design_summary", {}).get(
                        design.get("design_id"), {}
                    ),
                    "nominal": next(
                        (case for case in design_cases if case.get("kind") == "nominal"), None
                    ),
                    "shared_weighted_cd": [
                        case.get("xfoil_weighted_cd")
                        for case in design_cases
                        if case.get("kind") == "shared" and case.get("status") == "ok"
                    ],
                    "adverse_tail": [
                        case
                        for case in design_cases
                        if case.get("kind") == "adverse_tail"
                    ],
                },
            }
        )

    optimisation_summary = {
        "device": optimisation.get("device"),
        "manufacturing_common_random_numbers": optimisation.get(
            "manufacturing_common_random_numbers", {}
        ),
        "ava_independent_cst_common_random_numbers": optimisation.get(
            "ava_independent_cst_common_random_numbers", {}
        ),
        "deterministic_drag_optimisation": optimisation.get(
            "deterministic_drag_optimisation", {}
        ),
        "ava_lambda_sweep": {
            key: {
                field: value
                for field, value in result.items()
                if field != "diagnostics"
            }
            for key, result in optimisation.get("ava_lambda_sweep", {}).items()
        },
        "nsga2": {
            name: {
                "population": profile.get("population"),
                "generations": profile.get("generations"),
                "uncertainty_samples": profile.get("uncertainty_samples"),
                "independent_seeds": profile.get("independent_seeds"),
                "completed_runs": len(profile.get("runs", [])),
                "feasible_solutions": sum(
                    int(
                        np.all(
                            np.asarray(run.get("constraint_values", []), dtype=float) <= 0,
                            axis=1,
                        ).sum()
                    )
                    for run in profile.get("runs", [])
                    if len(run.get("constraint_values", []))
                ),
                "agreement": profile.get("agreement", {}),
                "final_generation_diagnostics": [
                    run.get("history", [])[-1] if run.get("history") else None
                    for run in profile.get("runs", [])
                ],
            }
            for name, profile in optimisation.get("nsga2", {}).items()
        },
    }

    report_path = REPORTS_ROOT / "viability/VIABILITY_REPORT.md"
    json_path = REPORTS_ROOT / "viability/viability_report.json"
    handover_path = REPORTS_ROOT / "handover/AGENT_HANDOVER.md"
    excluded_manifest_paths = {report_path.resolve(), json_path.resolve(), handover_path.resolve()}
    artifact_paths: list[Path] = []
    artifact_roots = [
        ROOT / "data/robust_v2/manifests",
        ROOT / "data/robust_v2/splits",
        ROOT / "data/robust_v2/processed",
        RESULTS_ROOT,
        REPORTS_ROOT / "setup",
        REPORTS_ROOT / "data",
        REPORTS_ROOT / "modelling",
    ]
    for artifact_root in artifact_roots:
        if artifact_root.exists():
            artifact_paths.extend(
                path
                for path in artifact_root.rglob("*")
                if path.is_file() and path.resolve() not in excluded_manifest_paths
            )
    artifact_manifest = hash_paths(sorted(set(artifact_paths)), ROOT)
    working_status = git_output("status", "--short", cwd=ROOT)
    changed_files = [line[3:] for line in working_status.splitlines() if len(line) >= 4]
    quality_gate_path = REPORTS_ROOT / "setup/quality_gates.json"
    quality_gates = (
        _json(quality_gate_path)
        if quality_gate_path.is_file()
        else {"status": "pending_final_validation", "claims": []}
    )
    local_commits = git_output(
        "log", "--oneline", "baseline-pre-robust-v2-20260824..HEAD", cwd=ROOT
    )
    report_path = REPORTS_ROOT / "viability/VIABILITY_REPORT.md"
    json_path = REPORTS_ROOT / "viability/viability_report.json"
    handover_path = REPORTS_ROOT / "handover/AGENT_HANDOVER.md"
    payload = {
        "Executive decision": decision,
        "Branch": git_output("branch", "--show-current", cwd=ROOT),
        "Current SHA": git_output("rev-parse", "HEAD", cwd=ROOT),
        "Working-tree status": working_status,
        "Environment": _json(REPORTS_ROOT / "setup/environment.json") if (REPORTS_ROOT / "setup/environment.json").exists() else {},
        "PyTorch device": "cuda" if torch.cuda.is_available() else "cpu",
        "XFOIL status": {
            "status": xfoil.get("status"),
            "comparison": xfoil.get("comparison"),
            "canary_completed": xfoil.get("completed_cases"),
            "canary_requested": xfoil.get("requested_cases"),
            "candidate_all_cases_completed": candidate_validation.get("all_cases_completed"),
            "candidate_design_summary": candidate_validation.get("design_summary", {}),
        },
        "V1 audit": all_results.get("legacy", {}),
        "Source reachability": source,
        "Local recovery": all_results.get("recovery", {}),
        "UIUC snapshot": source.get("uiuc", {}),
        "Pinned polar snapshot": source.get("snapshot", {}),
        "Live AirfoilTools": source.get("live", {}),
        "UIUC continuity/current differences": {
            "source_summary": source,
            "data_summary": data,
        },
        "Mappings": {"fraction": data.get("mapping_fraction"), "candidate_exact_polars": data.get("candidate_exact_polars")},
        "Accepted airfoils": data.get("mapped_airfoils", 0),
        "Accepted joint points": data.get("joint_model_points", 0),
        "Rejected records": data.get("rejected_records", 0),
        "Geometry and CST results": {
            "data_summary": data,
            "five_design_geometry_metrics": {
                design["design_id"]: design.get("diagnostics", {}).get("geometry_metrics")
                for design in five_designs
            },
        },
        "Leakage tests": leakage,
        "Pilot model": model.get("pilot", {}),
        "200-airfoil model": model.get("expansion_200", {}),
        "Full baseline": model.get("full_baseline", {}),
        "ML evidence": ml_evidence,
        "Baseline comparisons": {
            "pilot": model.get("pilot", {}).get("baseline_comparison", {}),
            "expansion_200": model.get("expansion_200", {}).get("baseline_comparison", {}),
            "locked_test_metrics": model.get("full_baseline", {}).get("locked_test_metrics", {}),
        },
        "Tuning status": model.get("tuning", {}),
        "Calibration status": model.get("calibration", {}),
        "NeuralFoil benchmark": advanced.get("neuralfoil", {}),
        "XFOIL comparison": {
            "canary_ranking": xfoil.get("local_lift_to_drag_ranking", []),
            "candidate_nominal_xfoil_ranking": candidate_validation.get(
                "nominal_xfoil_ranking", []
            ),
            "candidate_nominal_surrogate_ranking": candidate_validation.get(
                "nominal_surrogate_ranking", []
            ),
            "nominal_rank_spearman": candidate_validation.get("nominal_rank_spearman"),
            "effect_sizes": candidate_validation.get("effect_sizes", {}),
            "full_artifact": "results/robust_v2/xfoil_candidates/candidate_validation_summary.json",
        },
        "Uncertainty validity": {
            "interpretation": uncertainty.get("interpretation"),
            "effective_configuration": uncertainty.get("effective_configuration", {}),
            "levels": level_summaries,
            "sobol_convergence": {
                amplitude: study.get("convergence", {})
                for amplitude, study in uncertainty.get("sobol", {}).items()
            },
            "ablations": uncertainty.get("ablations", {}),
            "full_artifact": "results/robust_v2/uncertainty/manufacturing_study.json",
        },
        "Optimisation study": optimisation_summary,
        "Five-design comparison": five_designs,
        "Blockers and severity": blockers,
        "Scientific limitations": [
            "The archived polar corpus is heterogeneous and does not provide experimental truth or uniform solver metadata for every source row.",
            "The manufacturing model is a prescribed bounded smooth-field model, not a measured shop-floor distribution.",
            "The surrogate is valid only inside the calibrated geometry/condition support domain and inherits source-label bias.",
            "Local XFOIL 6.99 is an independent numerical cross-check, not wind-tunnel or flight validation.",
            "V1-to-V2 claim language remains unsupported until exact V1 split, scaler, dependency, and training provenance are recovered.",
        ],
        "Resolution plan": [blocker["resolution"] for blocker in blockers],
        "Exact resume command": "uv run python -m robust_airfoil run --profile viability --resume",
        "Primary report path": str(report_path.relative_to(ROOT)),
        "Handover path": str(handover_path.relative_to(ROOT)),
        "Files created or modified": changed_files,
        "Local commits made": local_commits.splitlines() if local_commits else [],
        "Tests passed failed skipped": quality_gates,
        "Remaining failures": blockers,
        "Output hash manifest": artifact_manifest,
    }
    build_viability_report(payload, report_path, json_path)
    handover_path.parent.mkdir(parents=True, exist_ok=True)
    handover_path.write_text(
        "# Agent Handover\n\n"
        + f"Decision: **{decision}**\n\n"
        + "\n".join(
            f"- [{blocker['severity'].upper()}] {blocker['code']}: {blocker['detail']} Resolution: {blocker['resolution']}"
            for blocker in blockers
        )
        + "\n\nResume:\n```powershell\nuv run python -m robust_airfoil run --profile viability --resume\n```\n",
        encoding="utf-8",
    )
    next_commands = REPORTS_ROOT / "handover/NEXT_COMMANDS.ps1"
    next_commands.write_text("uv run python -m robust_airfoil run --profile viability --resume\n", encoding="utf-8")
    return {"decision": decision, "blockers": blockers, "report": str(report_path)}, [report_path, json_path, handover_path, next_commands], "passed" if decision == "GO" else "held"


def run_viability(
    resume: bool = True,
    *,
    results_root: Path | None = None,
    reports_root: Path | None = None,
    lineage_metadata: dict[str, Any] | None = None,
    phase_ids: set[str] | None = None,
) -> int:
    """Run the viability pipeline, optionally in an isolated evidence lineage."""
    with _lineage_paths(results_root, reports_root, lineage_metadata):
        if phase_ids is None:
            return _run_viability(resume)
        return _run_viability(resume, phase_ids)


def _run_viability(resume: bool = True, phase_ids: set[str] | None = None) -> int:
    valid_phase_ids = {"00", "02", "03", "04", "05-08", "09-16", "17-21", "22"}
    if phase_ids is not None and not phase_ids <= valid_phase_ids:
        raise ValueError("Unknown viability phase requested")

    def requested(phase_id: str) -> bool:
        return phase_ids is None or phase_id in phase_ids

    def previous(phase_id: str) -> tuple[dict[str, Any], str]:
        record = state.phases.get(phase_id)
        if record is None:
            return {}, "pending"
        return record.summary, record.status

    load_all_configs()
    state = RunState(RUN_STATE_PATH)
    results: dict[str, Any] = {}
    historical_inputs = [
        ROOT / "data/csv/dataset_12CST_params.csv",
        ROOT / "scripts/train_best_model.py",
        ROOT / "src/models.py",
        ROOT / "src/utils.py",
        *sorted((ROOT / "trained_model").glob("**/*")),
    ]
    if requested("00"):
        results["setup"], _ = _phase(
            state, "00", "preflight and provenance", "robust-airfoil run", _phase00,
            resume=resume, input_paths=historical_inputs,
        )
    else:
        results["setup"], _ = previous("00")
    if requested("02"):
        results["legacy"], results["legacy_status"] = _phase(
            state, "02", "legacy audit", "audit_legacy", _phase02,
            resume=resume, input_paths=[ROOT / "data/csv/dataset_12CST_params.csv"],
        )
    else:
        results["legacy"], results["legacy_status"] = previous("02")
    if requested("03"):
        results["recovery"], _ = _phase(
            state, "03", "bounded local recovery", "discover_legacy_data --auto-roots", _phase03,
            resume=resume,
        )
    else:
        results["recovery"], _ = previous("03")
    if requested("04"):
        results["source"], source_status = _phase(
            state, "04", "source acquisition", "acquire_sources", _phase04,
            resume=resume,
        )
    else:
        results["source"], source_status = previous("04")
    if requested("05-08") and source_status != "failed":
        polar_inputs, geometry_inputs = _exact_source_paths()
        results["data"], _ = _phase(
            state, "05-08", "parse map and build long-form dataset", "build_dataset", _phase05_08,
            resume=resume,
            input_paths=[
                ROOT / "data/robust_v2/manifests/uiuc_current.json",
                ROOT / "data/robust_v2/manifests/airfoiltools_snapshot.json",
                *polar_inputs,
                *geometry_inputs,
            ],
        )
    elif requested("05-08"):
        results["data"] = {"error": "source acquisition failed"}
    else:
        data_root = ROOT / "data/robust_v2/processed/full_exact"
        airfoils_path = data_root / "airfoils.parquet"
        points_path = data_root / "model_points.parquet"
        dataset_summary_path = REPORTS_ROOT / "data/dataset_summary.json"
        if dataset_summary_path.is_file():
            results["data"] = _json(dataset_summary_path)
        elif airfoils_path.is_file() and points_path.is_file():
            results["data"] = {
                "mapped_airfoils": len(pd.read_parquet(airfoils_path)),
                "status": "adopted_hash_pinned_processed_data",
            }
        else:
            results["data"], _ = previous("05-08")
    if results.get("data", {}).get("mapped_airfoils", 0) >= 20:
        model_inputs = [
            ROOT / "data/robust_v2/processed/full_exact/airfoils.parquet",
            ROOT / "data/robust_v2/processed/full_exact/model_points.parquet",
        ]
        if requested("09-16"):
            results["model"], model_status = _phase(
                state, "09-16", "splits model ladder tuning calibration", "train_baseline", _phase12_16,
                resume=resume, input_paths=model_inputs,
            )
        else:
            model_summary_path = REPORTS_ROOT / "modelling/model_ladder.json"
            previous_model, model_status = previous("09-16")
            results["model"] = (
                _json(model_summary_path)
                if model_status == "passed" and model_summary_path.is_file()
                else previous_model
            )
        results["model_status"] = model_status
        if requested("17-21") and model_status == "passed":
            advanced_inputs = [*model_inputs, FROZEN_MANIFEST_PATH]
            results["advanced"], advanced_status = _phase(
                state, "17-21", "benchmarks uncertainty optimisation validation", "run_uncertainty_and_optimisation",
                lambda: _phase14_21(results["model"]), resume=resume, input_paths=advanced_inputs,
            )
        elif requested("17-21"):
            state.invalidate(
                "17-21",
                f"Model phase 09-16 is {model_status}; advanced evidence is not valid",
            )
            results["advanced"] = {
                "status": "blocked",
                "reason": "Advanced studies require a passed, verified frozen model phase",
                "upstream_phase": "09-16",
                "upstream_status": model_status,
            }
            advanced_status = "blocked"
        else:
            advanced_summary_path = REPORTS_ROOT / "viability/advanced_studies.json"
            previous_advanced, advanced_status = previous("17-21")
            results["advanced"] = (
                _json(advanced_summary_path)
                if advanced_status == "passed" and advanced_summary_path.is_file()
                else previous_advanced
            )
        results["advanced_status"] = advanced_status
    else:
        results["model"], results["advanced"] = {}, {}
        results["model_status"], results["advanced_status"] = "blocked", "blocked"
    report_inputs = [
        REPORTS_ROOT / "setup/environment.json",
        REPORTS_ROOT / "data/source_acquisition.json",
        REPORTS_ROOT / "data/dataset_summary.json",
        REPORTS_ROOT / "modelling/model_ladder.json",
        REPORTS_ROOT / "modelling/ml_evidence.json",
        REPORTS_ROOT / "viability/advanced_studies.json",
        RESULTS_ROOT / "uncertainty/manufacturing_study.json",
        RESULTS_ROOT / "optimisation/optimisation_summary.json",
        RESULTS_ROOT / "xfoil_canaries/canary_summary.json",
        RESULTS_ROOT / "xfoil_candidates/candidate_validation_summary.json",
        REPORTS_ROOT / "setup/quality_gates.json",
    ]
    if requested("22"):
        final, status = _phase(
            state, "22", "viability report and handover", "build_report", lambda: _phase22(results),
            resume=resume, input_paths=report_inputs,
        )
    else:
        final, status = {"status": "not_run", "requested_phases": sorted(phase_ids or [])}, "passed"
    print(json.dumps(final, indent=2))
    return 0 if status in {"passed", "held"} else 1
