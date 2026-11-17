"""Publish one post-push source-bound OneDrive handoff without deleting history.

Require a clean, normally pushed authorised branch and explicit review
acceptance. Stage and hash every file before moving the prior handoff aside.
Local OneDrive bytes do not prove remote cloud synchronisation.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "reports/robust_v2/external_academic_review_20261014"
CONTROL = BASE / "CONTROLLED_MODEL"
LINEAGE = ROOT / "results/robust_v2/lineages/robust-v2-controlled-20261020"
ONEDRIVE = Path("C:/Users/Ben/OneDrive/paper")
LIVE = ONEDRIVE / "READY_FOR_REVIEW"
RELEASE_SCHEMA = "academic-review-release-v2"


def git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def sha(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def checked_source(rel):
    path = ROOT / rel
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def validate_release_preconditions():
    if git("branch", "--show-current") != "robust-v2-viability":
        raise ValueError("Not on authorised review branch")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Uncommitted source: do not publish a staged or dirty branch")
    git("fetch", "origin", "robust-v2-viability")
    head = git("rev-parse", "HEAD")
    if head != git("rev-parse", "origin/robust-v2-viability"):
        raise ValueError("Review commit is not the published branch tip")
    final = json.loads(
        checked_source(
            "reports/robust_v2/external_academic_review_20261014/FINAL_ACCEPTANCE.json"
        ).read_text(encoding="utf-8")
    )
    if not final.get("review_deliverables_complete") or final.get("software_checks") != "PASS":
        raise ValueError("The final author review gate is not accepted")
    for rel in [
        "results/robust_v2/lineages/robust-v2-controlled-20261020/optimisation_production/PRODUCTION_SUMMARY.json",
        "results/robust_v2/lineages/robust-v2-controlled-20261020/confirmation/CONFIRMATION_RESULTS.json",
    ]:
        if not checked_source(rel).is_file():
            raise FileNotFoundError(rel)
    return head, final


def sources():
    direct = [
        ("README.md", "paper/README.md"),
        (
            "READ_THIS_FIRST.md",
            "reports/robust_v2/external_academic_review_20261014/READ_THIS_FIRST.md",
        ),
        ("LATEST_MANUSCRIPT_FOR_AVA.docx", "paper/LATEST_MANUSCRIPT_FOR_AVA.docx"),
        (
            "SCIENTIFIC_REVIEW_REPORT.docx",
            "reports/robust_v2/external_academic_review_20261014/SCIENTIFIC_REVIEW_REPORT.docx",
        ),
        ("paper/manuscript.md", "paper/manuscript.md"),
        ("paper/MODEL_CARD.md", "paper/MODEL_CARD.md"),
        ("paper/response_to_reviewers.md", "paper/response_to_reviewers.md"),
        (
            "COPY_TO_LAPTOP_OFFLINE.ps1",
            "reports/robust_v2/external_academic_review_20261014/COPY_TO_LAPTOP_OFFLINE.ps1",
        ),
        (
            "EVIDENCE/FINAL_ACCEPTANCE.json",
            "reports/robust_v2/external_academic_review_20261014/FINAL_ACCEPTANCE.json",
        ),
        (
            "EVIDENCE/EVIDENCE_SHA256_MANIFEST.json",
            "reports/robust_v2/external_academic_review_20261014/EVIDENCE_SHA256_MANIFEST.json",
        ),
        (
            "EVIDENCE/CLAIM_EVIDENCE_MATRIX.csv",
            "reports/robust_v2/external_academic_review_20261014/CLAIM_EVIDENCE_MATRIX.csv",
        ),
        (
            "EVIDENCE/REFERENCES_VERIFIED.csv",
            "reports/robust_v2/external_academic_review_20261014/REFERENCES_VERIFIED.csv",
        ),
        (
            "EVIDENCE/REBUILD_PROTOCOL.md",
            "reports/robust_v2/external_academic_review_20261014/REBUILD_PROTOCOL.md",
        ),
        (
            "EVIDENCE/EXACT_RUNTIME_SOURCE_BYTES.zip",
            "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/EXACT_RUNTIME_SOURCE_BYTES.zip",
        ),
        (
            "EVIDENCE/EXACT_RUNTIME_SOURCE_MANIFEST.json",
            "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/EXACT_RUNTIME_SOURCE_MANIFEST.json",
        ),
        (
            "EVIDENCE/SCIENTIFIC_REVIEW.md",
            "reports/robust_v2/external_academic_review_20261014/ACADEMIC_REFEREE_REPORT.md",
        ),
        (
            "EVIDENCE/CONTROLLED_DATA_MANIFEST.json",
            "results/robust_v2/lineages/robust-v2-controlled-20261020/DATA_MANIFEST.json",
        ),
        (
            "EVIDENCE/CONTROLLED_TRAINING_MANIFEST.json",
            "results/robust_v2/lineages/robust-v2-controlled-20261020/full_ensemble/ensemble_manifest.json",
        ),
        (
            "EVIDENCE/FROZEN_MODEL_MANIFEST.json",
            "results/robust_v2/lineages/robust-v2-controlled-20261020/FROZEN_MODEL_MANIFEST.json",
        ),
        (
            "EVIDENCE/CONTROLLED_TEST_METRICS.json",
            "reports/robust_v2/external_academic_review_20261014/CONTROLLED_MODEL/REPAIRED_TEST_METRICS.json",
        ),
        (
            "EVIDENCE/CONTROLLED_BASELINES.json",
            "reports/robust_v2/external_academic_review_20261014/CONTROLLED_MODEL/REPAIRED_BASELINES.json",
        ),
        (
            "EVIDENCE/CONTROLLED_RISK_OOF.json",
            "reports/robust_v2/external_academic_review_20261014/CONTROLLED_MODEL/RISK_OOF/REPAIRED_RISK_OOF.json",
        ),
        (
            "EVIDENCE/CONFIRMATION_RESULTS.json",
            "results/robust_v2/lineages/robust-v2-controlled-20261020/confirmation/CONFIRMATION_RESULTS.json",
        ),
        (
            "EVIDENCE/PRODUCTION_SUMMARY.json",
            "results/robust_v2/lineages/robust-v2-controlled-20261020/optimisation_production/PRODUCTION_SUMMARY.json",
        ),
        (
            "EVIDENCE/NOMINAL_CONTROL_SUMMARY.json",
            "results/robust_v2/lineages/robust-v2-controlled-20261020/nominal_control/NOMINAL_CONTROL_SUMMARY.json",
        ),
        (
            "EVIDENCE/SOLVER_CASE_ACCOUNTING.json",
            "reports/robust_v2/external_academic_review_20261014/CONTROLLED_MODEL/SOLVER_CASE_ACCOUNTING.json",
        ),
        (
            "EVIDENCE/CONFIRMATION_CASE_ACCOUNTING.json",
            "reports/robust_v2/external_academic_review_20261014/CONTROLLED_MODEL/CONFIRMATION/SOLVER_CASE_ACCOUNTING.json",
        ),
    ]
    for path in sorted((ROOT / "paper/figures").glob("fig_*.png")):
        direct.append(("paper/figures/" + path.name, path.relative_to(ROOT).as_posix()))
    for path in sorted((LINEAGE / "full_ensemble").glob("member_*/checkpoint.pt")):
        direct.append(
            ("MODELS/" + path.relative_to(LINEAGE).as_posix(), path.relative_to(ROOT).as_posix())
        )
    for path in sorted((LINEAGE / "full_ensemble").glob("member_*/scaling.joblib")):
        direct.append(
            ("MODELS/" + path.relative_to(LINEAGE).as_posix(), path.relative_to(ROOT).as_posix())
        )
    for name in ["development", "calibration", "locked_test"]:
        p = LINEAGE / "partition_inputs" / (name + ".parquet")
        direct.append(("MODEL_DATA/" + name + ".parquet", p.relative_to(ROOT).as_posix()))
    for p in sorted((ROOT / "reports/robust_v2/submission_review_20261014/CFD_HANDOFF").rglob("*")):
        if p.is_file():
            direct.append(
                (
                    "CFD_HANDOFF/"
                    + p.relative_to(
                        ROOT / "reports/robust_v2/submission_review_20261014/CFD_HANDOFF"
                    ).as_posix(),
                    p.relative_to(ROOT).as_posix(),
                )
            )
    # Include the exact browser-derived pages that were inspected, alongside
    # explicit evidence that these are NOT native Microsoft Word rendering.
    direct.append(("EVIDENCE/FINAL_VISUAL_QA.json", "reports/robust_v2/external_academic_review_20261014/VISUAL_QA.json"))
    direct.append(("EVIDENCE/FINAL_VISUAL_QA_V3.json", "reports/robust_v2/external_academic_review_20261014/VISUAL_QA_V3.json"))
    visual = BASE / "RENDERED_V3"
    for image in sorted(visual.rglob("*")):
        if image.is_file() and image.suffix.lower() in {".png", ".pdf"}:
            direct.append(("VISUAL_PREVIEW_NON_NATIVE_WORD/" + image.relative_to(visual).as_posix(), image.relative_to(ROOT).as_posix()))
    if (BASE / "EMAIL_TO_AVA_DRAFT.txt").is_file():
        direct.append(
            (
                "EMAIL_TO_AVA_DRAFT.txt",
                (BASE / "EMAIL_TO_AVA_DRAFT.txt").relative_to(ROOT).as_posix(),
            )
        )
    # Full source/label evidence is separately archived and hash verified.
    direct.extend([
        ("RAW_EVIDENCE/RAW_XFOIL_EVIDENCE.zip", "reports/robust_v2/external_academic_review_20261014/RAW_XFOIL_EVIDENCE.zip"),
        ("RAW_EVIDENCE/RAW_XFOIL_EVIDENCE_SHA256.json", "reports/robust_v2/external_academic_review_20261014/RAW_XFOIL_EVIDENCE_SHA256.json"),
        ("MODEL_DATA/controlled_model_points.parquet", "data/robust_v2/processed/controlled_20261020/model_points.parquet"),
        ("MODEL_DATA/controlled_airfoils.parquet", "data/robust_v2/processed/controlled_20261020/airfoils.parquet"),
        ("EVIDENCE/FULL_TEN_SEED_INDEPENDENT_RECONCILIATION.json", "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/FULL_TEN_SEED_INDEPENDENT_RECONCILIATION.json"),
        ("EVIDENCE/FULL_TEN_SEED_NOMINAL_INDEPENDENT_RECONCILIATION.json", "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/FULL_TEN_SEED_NOMINAL_INDEPENDENT_RECONCILIATION.json"),
        ("EVIDENCE/INDEPENDENT_2048_SENSITIVITY_RECONCILIATION.json", "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/INDEPENDENT_2048_SENSITIVITY_RECONCILIATION.json"),
        ("EVIDENCE/INDEPENDENT_SELECTED_XFOIL_RECONCILIATION.json", "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/INDEPENDENT_SELECTED_XFOIL_RECONCILIATION.json"),
        ("EVIDENCE/PRODUCTION_PROTOCOL.json", "results/robust_v2/lineages/robust-v2-controlled-20261020/optimisation_production/PRODUCTION_PROTOCOL.json"),
        ("EVIDENCE/NOMINAL_CONTROL_PROTOCOL.json", "results/robust_v2/lineages/robust-v2-controlled-20261020/nominal_control/NOMINAL_CONTROL_PROTOCOL.json"),
    ])
    for p in sorted((LINEAGE / "optimisation_production").glob("nsga2_rebuilt-full-production_seed_*.json")):
        direct.append(("RESULTS/ROBUST_SEEDS/" + p.name, p.relative_to(ROOT).as_posix()))
    for p in sorted((LINEAGE / "nominal_control").glob("nominal_seed_*.json")):
        direct.append(("RESULTS/NOMINAL_SEEDS/" + p.name, p.relative_to(ROOT).as_posix()))
    for rel in ["GEOMETRY_SELECTION_FROZEN.json", "FROZEN_CONFIRMATION_PREDICTOR.json", "PREDICTIONS_BEFORE_LABELS_RECEIPT.json", "predictions_before_labels.parquet", "observed_confirmation_points.parquet", "CONFIRMATION_RESULTS.json"]:
        p = LINEAGE / "confirmation" / rel
        direct.append(("RESULTS/CONFIRMATION/" + rel, p.relative_to(ROOT).as_posix()))
    # Independently frozen finalist selection, strict solver accounting and
    # all actually inspected Word-derived pages belong to the coauthor handoff.
    # Browser previews are explicitly NOT native Microsoft Word page layouts.
    review_evidence = [
        ("EVIDENCE/VISUAL_QA.json", "reports/robust_v2/external_academic_review_20261014/VISUAL_QA.json"),
        ("EVIDENCE/VISUAL_QA_V3.json", "reports/robust_v2/external_academic_review_20261014/VISUAL_QA_V3.json"),
        ("EVIDENCE/CONTROLLED_PIPELINE_STATUS_R3_FAILED.json", "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/CONTROLLED_PIPELINE_STATUS_R3.json"),
        ("EVIDENCE/CONTROLLED_PIPELINE_STATUS_R4_COMPLETED.json", "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/CONTROLLED_PIPELINE_STATUS_R4.json"),
        ("EVIDENCE/NOMINAL_R4_RECOVERY_PROOF.json", "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/NOMINAL_R4_RECOVERY_PROOF.json"),
        ("EVIDENCE/SELECTED_STUDY_EXECUTION.json", "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/SELECTED_STUDY_EXECUTION.json"),
        ("EVIDENCE/AUTHOR_DRAFT_EXECUTION.json", "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/AUTHOR_DRAFT_EXECUTION.json"),
        ("EVIDENCE/RAW_ARCHIVE_EXECUTION.json", "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/RAW_ARCHIVE_EXECUTION.json"),
        ("EVIDENCE/FROZEN_SELECTED_DESIGNS.json", "results/robust_v2/lineages/robust-v2-controlled-20261020/selected_study/FROZEN_SELECTED_DESIGNS.json"),
        ("EVIDENCE/FRESH_2048_SCENARIO_EVALUATION.json", "results/robust_v2/lineages/robust-v2-controlled-20261020/selected_study/FINAL_2048_SCENARIO_EVALUATION.json"),
        ("EVIDENCE/XFOIL_CASE_SELECTION_PROTOCOL.json", "results/robust_v2/lineages/robust-v2-controlled-20261020/selected_study/XFOIL_CASE_SELECTION_PROTOCOL.json"),
        ("EVIDENCE/SELECTED_XFOIL_SUMMARY.json", "results/robust_v2/lineages/robust-v2-controlled-20261020/selected_study/SELECTED_XFOIL_SUMMARY.json"),
    ]
    direct.extend(review_evidence)
    pages = ROOT / "reports/robust_v2/external_academic_review_20261014/RENDERED_V3"
    for item in sorted(pages.iterdir()):
        if item.is_file():
            direct.append(("WORD_DERIVED_BROWSER_PREVIEWS/" + item.name, item.relative_to(ROOT).as_posix()))
    # Reconstructable execution receipts and their original stdout/stderr are
    # review evidence. Preserve failed as well as passing test and solver logs.
    # Logs remain excluded from Git to avoid contaminating the source archive.
    for logfile in sorted(BASE.rglob("*.log")):
        direct.append(("EXECUTION_LOGS/" + logfile.relative_to(BASE).as_posix(), logfile.relative_to(ROOT).as_posix()))
    revision = BASE / "CONTINUATION_20261020/AUTHOR_EOF_R6"
    for proof in sorted(set(revision.glob("PRIOR_*")) | set(revision.glob("*.json"))):
        direct.append(("EVIDENCE/AUTHOR_EOF_R6/" + proof.name, proof.relative_to(ROOT).as_posix()))
    for filename in ["FINAL_SOFTWARE_ACCEPTANCE_GATES.json", "FINAL_CANDIDATE_POST_ACCEPTANCE_STATUS.json"]:
        record = BASE / "CONTINUATION_20261020" / filename
        direct.append(("EVIDENCE/" + filename, record.relative_to(ROOT).as_posix()))
    return direct


def _file_map(directory):
    return {item.relative_to(directory).as_posix(): item for item in directory.rglob('*') if item.is_file()}


def _safe_onedrive_install(staging, live, backup, manifest):
    """Preserve a fully verified prior release before updating cloud-backed files.

    Windows OneDrive reparse directories can refuse rename/replace (WinError5).
    Hydrated files can still be copied directly. The release manifest is copied
    LAST, only after every new file and removal has been source-verified.
    """
    desired = {item['path']: item for item in manifest['files']}
    if len(desired) != len(manifest['files']):
        raise ValueError('Duplicate staged package paths')
    if backup.exists():
        raise FileExistsError('Historical release backup path already exists')
    previous = {}
    if live.exists():
        old_manifest = json.loads((live / 'SHA256_MANIFEST.json').read_text(encoding='utf-8'))
        previous = {item['path']: item for item in old_manifest['files']}
        if len(previous) != len(old_manifest['files']):
            raise ValueError('Prior manifest contains duplicate paths')
        old_files = _file_map(live)
        if set(old_files) != set(previous) | {'SHA256_MANIFEST.json'}:
            raise ValueError('Existing package has unmanifested files; refusing overwrite')
        for relative, item in previous.items():
            old_size = item.get('size', item.get('bytes'))
            if old_files[relative].stat().st_size != old_size or sha(old_files[relative]) != item['sha256']:
                raise ValueError('Historical source no longer matches its manifest: ' + relative)
        shutil.copytree(live, backup)
        copies = _file_map(backup)
        if copies.keys() != old_files.keys() or any(sha(copies[key]) != sha(path) for key, path in old_files.items()):
            raise ValueError('Versioned historical backup is not byte-identical')
    else:
        live.mkdir(parents=True)
    for relative, item in desired.items():
        source = staging / relative
        target = live / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and sha(target) == item['sha256']:
            continue
        if target.exists() and (relative not in previous or sha(target) != previous[relative]['sha256']):
            raise ValueError('Canonical file was unexpectedly modified: ' + relative)
        shutil.copy2(source, target)
        if target.stat().st_size != item['size'] or sha(target) != item['sha256']:
            raise ValueError('OneDrive hydrated-copy checksum mismatch: ' + relative)
    for relative in sorted(set(previous) - set(desired)):
        path = live / relative
        if not path.is_file() or sha(path) != sha(backup / relative):
            raise ValueError('Prior-only file changed before removal: ' + relative)
        path.unlink()
    # OneDrive Files On-Demand virtual directories may refuse rmdir; only
    # physical file membership is part of the released manifest contract.
    if set(_file_map(live)) != set(desired) | {'SHA256_MANIFEST.json'}:
        raise ValueError('Canonical contents do not exactly match the staged file set')
    for relative, item in desired.items():
        if sha(live / relative) != item['sha256']:
            raise ValueError('Final package file hash changed: ' + relative)
    shutil.copy2(staging / 'SHA256_MANIFEST.json', live / 'SHA256_MANIFEST.json')
    if sha(staging / 'SHA256_MANIFEST.json') != sha(live / 'SHA256_MANIFEST.json'):
        raise ValueError('Canonical activation manifest checksum mismatch')
    for name in ['SOURCE_COMMIT.zip', 'CFD_HANDOFF_ONLY.zip', 'RAW_EVIDENCE/RAW_XFOIL_EVIDENCE.zip']:
        with zipfile.ZipFile(live / name) as archive:
            if archive.testzip() is not None:
                raise ValueError('Release ZIP CRC validation failed: ' + name)
    return backup if backup.exists() else None


def main():
    head, final = validate_release_preconditions()
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    staging = ONEDRIVE / f".READY_FOR_REVIEW_STAGING_{head[:10]}"
    if staging.exists():
        raise FileExistsError("Unverified staged package exists: " + str(staging))
    staging.mkdir(parents=True)
    entries = []
    for relative, source in sources():
        origin = checked_source(source)
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, path)
        if sha(origin) != sha(path):
            raise OSError("Copied file differs: " + relative)
        entries.append({"path": relative, "size": path.stat().st_size, "sha256": sha(path)})
    with (staging / "SOURCE_COMMIT.zip").open("wb") as out:
        subprocess.run(["git", "archive", "--format=zip", head], cwd=ROOT, stdout=out, check=True)
    with zipfile.ZipFile(staging / "SOURCE_COMMIT.zip") as archive:
        if archive.testzip() is not None:
            raise OSError("Source ZIP CRC failure")
    sourcezip = staging / "SOURCE_COMMIT.zip"
    entries.append(
        {"path": "SOURCE_COMMIT.zip", "size": sourcezip.stat().st_size, "sha256": sha(sourcezip)}
    )
    with zipfile.ZipFile(
        staging / "CFD_HANDOFF_ONLY.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=8
    ) as archive:
        for path in sorted((staging / "CFD_HANDOFF").rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(staging).as_posix())
    with zipfile.ZipFile(staging / "CFD_HANDOFF_ONLY.zip") as archive:
        if archive.testzip() is not None:
            raise OSError("CFD ZIP CRC failure")
    cfdzip = staging / "CFD_HANDOFF_ONLY.zip"
    entries.append(
        {"path": "CFD_HANDOFF_ONLY.zip", "size": cfdzip.stat().st_size, "sha256": sha(cfdzip)}
    )
    manifest = {
        "schema": RELEASE_SCHEMA,
        "source_branch": "robust-v2-viability",
        "source_commit_sha": head,
        "source_remote_sha": head,
        "created_utc": timestamp,
        "review_status": final.get("status"),
        "cloud_sync_confirmed": False,
        "files": sorted(entries, key=lambda r: r["path"]),
    }
    (staging / "SHA256_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for entry in entries:
        path = staging / entry["path"]
        if path.stat().st_size != entry["size"] or sha(path) != entry["sha256"]:
            raise OSError("Staged integrity check failed")
    expected_count = len(entries) + 1
    if sum(1 for path in staging.rglob("*") if path.is_file()) != expected_count:
        raise OSError("Unmanifested staged files")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Repository changed during packaging")
    if git("rev-parse", "HEAD") != head:
        raise RuntimeError("Repository commit changed during packaging")
    backup = ONEDRIVE / f"HISTORICAL_READY_FOR_REVIEW_{timestamp}"
    if backup.exists():
        raise FileExistsError("Versioned historical destination already exists")
    verified_backup = _safe_onedrive_install(staging, LIVE, backup, manifest)
    print(
        json.dumps(
            {
                "status": "LOCALLY_VERIFIED_ONEDRIVE_HANDOFF",
                "source_commit_sha": head,
                "files": len(entries),
                "live": str(LIVE),
                "historical_directory": str(verified_backup) if verified_backup else None,
                "manifest_sha256": sha(LIVE / "SHA256_MANIFEST.json"),
                "cloud_sync_confirmed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
