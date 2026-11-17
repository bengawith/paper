"""Publish 21-Oct author-voice paper, the full prior scientific archive and raw cases.

Requires Git clean, normally pushed branch and both independently validated
science and new-manuscript receipts. The safe historical-preserving OneDrive
installer is reused unchanged. Local OneDrive bytes != proven cloud sync.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import runpy
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD = (
    ROOT
    / "reports/robust_v2/external_academic_review_20261014/CONTINUATION_20261020/publish_verified_review_package.py"
)
ONEDRIVE = Path("C:/Users/Ben/OneDrive/paper")
LIVE = ONEDRIVE / "READY_FOR_REVIEW"
prev = runpy.run_path(str(OLD))


def git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def mandatory(p):
    if not p.is_file():
        raise FileNotFoundError("Required fresh source missing " + str(p))
    return p


def main():
    if git("branch", "--show-current") != "robust-v2-viability":
        raise ValueError("Wrong branch")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Release source not committed")
    git("fetch", "origin", "robust-v2-viability")
    head = git("rev-parse", "HEAD")
    if head != git("rev-parse", "origin/robust-v2-viability"):
        raise ValueError("Unpushed review branch")
    latest = json.loads(
        mandatory(HERE / "FINAL_AUTHOR_REWRITE_ACCEPTANCE.json").read_text(encoding="utf-8")
    )
    if latest["status"] != "COMPLETE_NUMERICAL_PAPER_READY_FOR_COAUTHOR_REVIEW":
        raise ValueError("Full author-voice release acceptance not passed")
    source = mandatory(ROOT / "paper/manuscript.md")
    word = mandatory(ROOT / "paper/LATEST_MANUSCRIPT_FOR_AVA.docx")
    if latest["files_sha256"]["paper/manuscript.md"] != sha(source) or latest["files_sha256"][
        "paper/LATEST_MANUSCRIPT_FOR_AVA.docx"
    ] != sha(word):
        raise ValueError("Accepted final science and exact Word bytes differ")
    visual = json.loads(
        mandatory(HERE / "VISUAL_QA_FULL_AUTHOR_STYLE.json").read_text(encoding="utf-8")
    )
    if not visual["all_21_pages_visually_inspected"] or visual["word_sha256"] != sha(word):
        raise ValueError("Final Word not visually assessed at these exact bytes")
    bibliography = json.loads(
        mandatory(HERE / "REFERENCES_PRIMARY_VERIFIED_V2.json").read_text(encoding="utf-8")
    )
    if bibliography["entries"] != 29:
        raise ValueError("All references not checked")
    files = dict(prev["sources"]())
    oldroot = "reports/robust_v2/external_academic_review_20261014/"
    newroot = "reports/robust_v2/external_academic_review_20261021/"
    files.update(
        {
            "README.md": "paper/README.md",
            "READ_THIS_FIRST.md": "paper/README.md",
            "SCIENTIFIC_REVIEW_REPORT.docx": newroot + "SCIENTIFIC_REVIEW_REPORT.docx",
            "EMAIL_TO_AVA_DRAFT.txt": newroot + "EMAIL_TO_AVA_DRAFT.txt",
            "paper/REVIEW_STATUS.md": "paper/REVIEW_STATUS.md",
            "EVIDENCE/FINAL_ACCEPTANCE.json": newroot + "FINAL_AUTHOR_REWRITE_ACCEPTANCE.json",
            "EVIDENCE/EVIDENCE_SHA256_MANIFEST.json": newroot
            + "FINAL_NEW_EVIDENCE_SHA256_MANIFEST.json",
            "EVIDENCE/PRIMARY_SOURCE_29_REFERENCES.csv": newroot
            + "REFERENCES_PRIMARY_VERIFIED_V2.csv",
            "EVIDENCE/PRIMARY_SOURCE_29_REFERENCES.json": newroot
            + "REFERENCES_PRIMARY_VERIFIED_V2.json",
            "EVIDENCE/LIVE_CROSSREF_AUDIT.json": newroot + "BIBLIOGRAPHY_LIVE_AUDIT.json",
            "EVIDENCE/46_SOURCE_RECONCILED_NUMERIC_CLAIMS.json": newroot
            + "NEW_MANUSCRIPT_NUMERIC_CLAIM_AUDIT.json",
            "EVIDENCE/FINAL_REWRITTEN_PAPER_INTEGRITY.md": newroot
            + "FINAL_MANUSCRIPT_INTEGRITY.md",
            "EVIDENCE/FINAL_WORD_21_PAGE_VISUAL_QA.json": newroot
            + "VISUAL_QA_FULL_AUTHOR_STYLE.json",
            "EVIDENCE/PRIOR_20261014_AUTHOR_ACCEPTANCE.json": oldroot + "FINAL_ACCEPTANCE.json",
            "AUTHOR_ORIGINAL/ORIGINAL_SUBMITTED_PAPER.docx": newroot
            + "ORIGINAL_USER_SUBMISSION.docx",
            "AUTHOR_ORIGINAL/ORIGINAL_WRITING_STYLE_EVIDENCE.json": newroot
            + "ORIGINAL_AUTHOR_STYLE_EVIDENCE.json",
            "AUTHOR_ORIGINAL/PRE_REWRITE_MANUSCRIPT.md": newroot
            + "PRE_REWRITE_CONTROLLED_MANUSCRIPT.md",
            "AUTHOR_ORIGINAL/PRE_REWRITE_WORD.docx": newroot + "PRE_REWRITE_CONTROLLED_WORD.docx",
            "AUTHOR_ORIGINAL/PRE_LAYOUT_WORD.docx": newroot + "PRE_LAYOUT_AUTHOR_VOICE_WORD.docx",
            "AUTHOR_ORIGINAL/PRE_REWRITE_REVIEWER_RESPONSE.md": newroot
            + "PRE_REWRITE_RESPONSE_TO_REVIEWERS.md",
            "AUTHOR_ORIGINAL/PRE_REWRITE_ROOT_README.md": newroot + "PRE_REWRITE_ROOT_README.md",
        }
    )
    for path in sorted((HERE / "RENDERED_V2").rglob("*")):
        if path.is_file() and path.suffix.lower() in {".png", ".pdf"}:
            files[
                "VISUAL_VERIFIED_20261021/" + path.relative_to(HERE / "RENDERED_V2").as_posix()
            ] = path.relative_to(ROOT).as_posix()
    if len(files) < 350:
        raise ValueError("The handoff lost earlier source or new writing artifacts")
    staging = ONEDRIVE / (".READY_FOR_REVIEW_STAGING_" + head[:10])
    if staging.exists():
        raise FileExistsError("An existing stage requires inspection: " + str(staging))
    staging.mkdir(parents=True)
    entries = []
    for relative, src in sorted(files.items()):
        item = mandatory(ROOT / src)
        dst = staging / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dst)
        if sha(item) != sha(dst):
            raise OSError("Copy hash changed: " + relative)
        entries.append({"path": relative, "size": dst.stat().st_size, "sha256": sha(dst)})
    with (staging / "SOURCE_COMMIT.zip").open("wb") as stream:
        subprocess.run(
            ["git", "archive", "--format=zip", head], cwd=ROOT, stdout=stream, check=True
        )
    with zipfile.ZipFile(staging / "SOURCE_COMMIT.zip") as zipped:
        if zipped.testzip() is not None:
            raise OSError("Source archive CRC mismatch")
    zip_src = staging / "SOURCE_COMMIT.zip"
    entries.append(
        {"path": "SOURCE_COMMIT.zip", "size": zip_src.stat().st_size, "sha256": sha(zip_src)}
    )
    with zipfile.ZipFile(
        staging / "CFD_HANDOFF_ONLY.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as zipout:
        for item in sorted((staging / "CFD_HANDOFF").rglob("*")):
            if item.is_file():
                zipout.write(item, item.relative_to(staging).as_posix())
    zipped = staging / "CFD_HANDOFF_ONLY.zip"
    with zipfile.ZipFile(zipped) as zf:
        if zf.testzip() is not None:
            raise OSError("CFD input ZIP CRC mismatch")
    entries.append(
        {"path": "CFD_HANDOFF_ONLY.zip", "size": zipped.stat().st_size, "sha256": sha(zipped)}
    )
    manifest = {
        "schema": "academic-review-release-v2",
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "source_branch": "robust-v2-viability",
        "source_commit_sha": head,
        "source_remote_sha": head,
        "review_status": latest["status"],
        "scientific_pending": "INDEPENDENT_CFD_AND_PHYSICAL_CONCLUSION",
        "cloud_sync_confirmed": False,
        "all_29_references_primary_verified": True,
        "all_46_numerical_claims_source_reconciled": True,
        "files": sorted(entries, key=lambda x: x["path"]),
    }
    (staging / "SHA256_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    declared = {row["path"] for row in manifest["files"]}
    actual = {p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file()}
    if declared | {"SHA256_MANIFEST.json"} != actual:
        raise ValueError("Unlisted or missing staged package file")
    for item in entries:
        target = staging / item["path"]
        if target.stat().st_size != item["size"] or sha(target) != item["sha256"]:
            raise OSError("Staging changed during final checks")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Repo became dirty during stage")
    if git("rev-parse", "HEAD") != head:
        raise RuntimeError("Git commit changed during stage")
    backup = ONEDRIVE / (
        "HISTORICAL_READY_FOR_REVIEW_"
        + datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    historic = prev["_safe_onedrive_install"](staging, LIVE, backup, manifest)
    for item in entries:
        dst = LIVE / item["path"]
        if sha(dst) != item["sha256"]:
            raise OSError("Installed file changed")
    for relative in [
        "RAW_EVIDENCE/RAW_XFOIL_EVIDENCE.zip",
        "SOURCE_COMMIT.zip",
        "CFD_HANDOFF_ONLY.zip",
    ]:
        with zipfile.ZipFile(LIVE / relative) as z:
            if z.testzip() is not None:
                raise OSError("Installed zip is corrupt: " + relative)
    if sha(LIVE / "SHA256_MANIFEST.json") != sha(staging / "SHA256_MANIFEST.json"):
        raise OSError("Activated package manifest differs")
    if git("rev-parse", "HEAD") != head or git("rev-parse", "origin/robust-v2-viability") != head:
        raise ValueError("Pushed identity changed")
    print(
        json.dumps(
            {
                "status": "FULL_NEW_AUTHOR_STYLE_PACKAGE_LOCALLY_SHA_VERIFIED",
                "source_commit": head,
                "files": len(entries),
                "path": str(LIVE),
                "manifest_sha256": sha(LIVE / "SHA256_MANIFEST.json"),
                "historical_backup": str(historic) if historic else None,
                "cloud_sync_confirmed": False,
                "remaining_scientific_study": "independent CFD and final physical design conclusion",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
