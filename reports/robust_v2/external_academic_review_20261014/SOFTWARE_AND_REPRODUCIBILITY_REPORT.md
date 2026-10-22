# Software and reproducibility report

## Environment and boundary

Work is performed in the authorised existing `robust-v2-viability` branch on SamsPC, with base `156f07f77e92f77df34a6cc6f6a1726cc0e1638a`, not a reset clone. The source/model/data lineage remains preserved. Windows, Python 3.11.15, uv, the existing CUDA-capable PyTorch environment and local XFOIL executable were available. The final command log records actual dependency resolution, versions and exit codes rather than inferring them from a previous green run.

## Reproduced and fixed defects

Tests cover wrong coordinate ordering, reversed-x interpolation and thickness, nonfinite inputs, missing required-lift feasibility, angle rejection, checkpoint/scaler integrity, stale scientific caches, invalid reconstructed candidate and training fits, calibration cluster separation, and post-hoc front-scale acceptance. Each material source change was preceded by a reproduced failure; old and new outputs are retained. The added scientific-admission tests tighten rather than skip assertions. Historical cache mismatch is a stop requiring a new revision, not permission to overwrite evidence.

The available full non-network test invocation includes the local XFOIL option. Its final result, the prescribed non-network/non-slow subset, Ruff, mypy, uv sync, diff hygiene, current author-pack verifier and rendered document checks are in `validation_final/results.json`, `FINAL_ACCEPTANCE.json` and `VISUAL_QA.json`. A software pass does not mean the repaired model was trained or the production study was run.

## Reproduction commands

From the repository root on the documented machine:

    uv sync --extra dev --extra benchmarks --extra docs
    uv run ruff check src/robust_airfoil tests/robust_v2 scripts/robust_v2
    uv run mypy src/robust_airfoil
    uv run pytest tests/robust_v2 -m "not network and not xfoil and not slow" -q
    uv run pytest -m "not network" --run-xfoil -q
    uv run --extra docs python scripts/robust_v2/verify_final_author_pack.py
    uv run python reports/robust_v2/external_academic_review_20261014/recompute_ml.py
    uv run python reports/robust_v2/external_academic_review_20261014/audit_geometry_corpus.py
    uv run python scripts/robust_v2/audit_trust_incremental_calibration.py
    uv run python reports/robust_v2/external_academic_review_20261014/audit_raw_xfoil_and_fronts.py
    uv run python reports/robust_v2/external_academic_review_20261014/profile_candidate.py

Read script output paths before rerunning: analytical outputs belong to this review revision. Frozen source polars, weights and original reports must not be replaced. Large local artifacts are necessary for numerical reproduction and are not falsely represented as fully committed or publicly redistributable.

## Release integrity

The final Git commit is pushed normally to the same branch; no force push or merge is required. The OneDrive package is refreshed only from a clean repository whose HEAD equals the fetched remote branch. Existing package generations are moved into labelled history rather than deleted. Every packaged file is hashed; ZIP archives are CRC-tested. The offline-copy script refuses content/branch/commit mismatches and does not reset or switch the repository. The post-push package manifest carries the final commit, avoiding a self-referential commit-hash claim in pre-commit evidence.

## Review limitation

The final code/package inspection is an explicit self-review in this tool environment, not a fictitious independent agent review. The scientific conclusions remain subject to coauthor judgement and independent validation.
