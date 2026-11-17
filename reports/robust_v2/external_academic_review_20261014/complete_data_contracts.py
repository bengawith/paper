"""Complete the separately reproduced geometry and calibration-group fixes."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TESTS = ['-m', 'pytest', 'tests/robust_v2/test_review_data_contracts.py', '-q']
with (OUT/'DATA_CONTRACTS_RED.log').open('w', encoding='utf-8') as log:
    red = subprocess.run([sys.executable, *TESTS], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
assert red.returncode != 0, 'Expected the independently observed missing safeguards'
changes = []
def edit(rel, old, new):
    path = ROOT/rel
    text = path.read_text(encoding='utf-8')
    assert text.count(old) == 1, (rel, text.count(old))
    after = text.replace(old, new)
    path.write_text(after, encoding='utf-8')
    changes.append({'path': rel, 'before_sha256': hashlib.sha256(text.encode()).hexdigest(), 'after_sha256': hashlib.sha256(after.encode()).hexdigest()})

edit('src/robust_airfoil/data/builder.py',
     'from robust_airfoil.geometry.cst import fit_cst',
     'from robust_airfoil.geometry.cst import fit_cst, reconstruct_cst')
edit('src/robust_airfoil/data/builder.py',
     '                fit = fit_cst(normal.upper, normal.lower)\n                metrics = geometry_metrics(normal.upper, normal.lower)',
     '                fit = fit_cst(normal.upper, normal.lower)\n                refit_upper, refit_lower = reconstruct_cst(fit.parameters, normal.upper[:, 0])\n                refit_validity = validate_geometry(\n                    np.column_stack([normal.upper[:, 0], refit_upper]),\n                    np.column_stack([normal.lower[:, 0], refit_lower]),\n                )\n                if not refit_validity.valid:\n                    raise ValueError("invalid_refitted_geometry:" + ";".join(refit_validity.reasons))\n                metrics = geometry_metrics(normal.upper, normal.lower)')
path = ROOT/'src/robust_airfoil/data/splits.py'
text = path.read_text(encoding='utf-8')
text += '''\n\ndef calibration_geometry_groups(predictions: pd.DataFrame, records: pd.DataFrame) -> np.ndarray:
    """Resolve geometry clusters, rejecting unknown or non-calibration rows.

    Names alone are not independent groups: two names can belong to the same
    near-duplicate geometry cluster. The caller must use these groups for both
    cross-validation and resampling, and retain the source split-manifest hash.
    """
    required = {"airfoil_id", "cluster_id", "split"}
    if not required.issubset(records.columns) or "airfoil_id" not in predictions:
        raise ValueError("Calibration grouping requires complete split records")
    if records["airfoil_id"].duplicated().any():
        raise ValueError("Ambiguous duplicate airfoil IDs in split records")
    indexed = records.set_index("airfoil_id")
    partitions = predictions["airfoil_id"].map(indexed["split"])
    groups = predictions["airfoil_id"].map(indexed["cluster_id"])
    if groups.isna().any() or not partitions.eq("calibration").all():
        raise ValueError("Every risk-audit row must map to a calibration geometry cluster")
    return groups.to_numpy()
'''
path.write_text(text, encoding='utf-8')
edit('scripts/robust_v2/audit_trust_incremental_calibration.py',
     'from sklearn.preprocessing import StandardScaler',
     'from sklearn.preprocessing import StandardScaler\n\nfrom robust_airfoil.data.splits import calibration_geometry_groups')
edit('scripts/robust_v2/audit_trust_incremental_calibration.py',
     'OUT = ROOT / "reports/robust_v2/submission_review_20261014/TRUST_INCREMENTAL"',
     'OUT = ROOT / "reports/robust_v2/external_academic_review_20261014/CANONICAL_TRUST_INCREMENTAL"\nSPLIT = ROOT / "results/robust_v2/lineages/robust-v2-current-20260908/splits/full.json"')
edit('scripts/robust_v2/audit_trust_incremental_calibration.py',
     'groups = d.airfoil_id.astype(str).to_numpy()',
     'split_records = pd.DataFrame(json.loads(SPLIT.read_text(encoding="utf-8"))["records"])\ngroups = calibration_geometry_groups(d, split_records)')
edit('scripts/robust_v2/audit_trust_incremental_calibration.py',
     '    "cv": "GroupKFold(5) with no shared airfoil IDs",',
     '    "cv": "GroupKFold(5) with no shared geometry clusters",\n    "n_airfoils": int(d.airfoil_id.nunique()),\n    "split_manifest_sha256": hashlib.sha256(SPLIT.read_bytes()).hexdigest(),')
edit('scripts/robust_v2/audit_trust_incremental_calibration.py',
     '    "schema": "trust-calibration-group-oof-all-fit-is-train-only-v2",',
     '    "schema": "trust-calibration-geometry-cluster-oof-train-only-v3",')
edit('scripts/robust_v2/audit_trust_incremental_calibration.py',
     '    "limits": "Calibration-distribution CV diagnostic only. Not tested on pristine optimisation-selected/CFD examples. No operational trust threshold changed.",',
     '    "limits": "Retrospective calibration diagnostic using historical geometry clusters. Bootstrap resamples fixed OOF predictions, not refits. Training geometry encodings require repair. No prospective confirmation or operational threshold change.",')
with (OUT/'DATA_CONTRACTS_GREEN.log').open('w', encoding='utf-8') as log:
    green = subprocess.run([sys.executable, *TESTS, 'tests/robust_v2/test_dataset_builder.py'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
print((OUT/'DATA_CONTRACTS_GREEN.log').read_text(encoding='utf-8'), flush=True)
(OUT/'DATA_CONTRACT_EDIT.json').write_text(json.dumps(changes, indent=2), encoding='utf-8')
raise SystemExit(green.returncode)
