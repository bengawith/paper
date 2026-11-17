"""Exact, logged patches for reproduced review defects; never touches frozen data."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PARSER = argparse.ArgumentParser()
PARSER.add_argument('fix', choices=['alpha','required_lift','ensemble_identity','cache'])
args = PARSER.parse_args()
patches = []
if args.fix == 'alpha':
    patches = [('src/robust_airfoil/optimisation/study.py',
'''        trusted = support <= self.support_threshold
        for target, threshold in self.disagreement_thresholds.items():''',
'''        trusted = support <= self.support_threshold
        alpha = predictions["alpha_deg"].to_numpy(float).reshape(shape)
        trusted &= (alpha >= self.alpha_coverage[0]) & (alpha <= self.alpha_coverage[1])
        for target, threshold in self.disagreement_thresholds.items():''')]
elif args.fix == 'required_lift':
    patches = [
('src/robust_airfoil/optimisation/study.py',
'''            candidate_cm = float(prediction["prediction_cm"].mean())
            margins = [''',
'''            candidate_cm = float(prediction["prediction_cm"].mean())
            nominal_drag = weighted_required_lift_drag(
                self.alpha,
                prediction["prediction_cl"].to_numpy(float),
                np.exp(prediction["prediction_log_cd"].to_numpy(float)),
                self.required,
                self.weights,
            )
            lift_feasible = bool(np.isfinite(nominal_drag))
            margins = ['''),
('src/robust_airfoil/optimisation/study.py',
'''                1.0 if validity.valid and trust_ok else -1.0,''',
'''                1.0 if validity.valid and trust_ok and lift_feasible else -1.0,'''),
('src/robust_airfoil/optimisation/study.py',
'''                    "trusted": trust_ok,
                    "maximum_support_distance":''',
'''                    "trusted": trust_ok,
                    "nominal_required_lift_feasible": lift_feasible,
                    "nominal_weighted_drag": float(nominal_drag) if lift_feasible else None,
                    "maximum_support_distance":''')]
elif args.fix == 'ensemble_identity':
    patches = [('src/robust_airfoil/modelling/ensemble.py',
'''        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.members = [load_member(member_root / f"member_{index:03d}", self.device) for index in range(members)]''',
'''        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.member_root = member_root
        manifest_path = member_root / "ensemble_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not 1 <= members <= int(manifest["member_count"]):
            raise ValueError("Requested member count is inconsistent with the ensemble manifest")
        self.artifact_hashes: dict[str, str] = {}
        for index in range(members):
            for label, filename in (("checkpoint", "checkpoint.pt"), ("scaling", "scaling.joblib")):
                key = f"member_{index:03d}_{label}"
                path = member_root / f"member_{index:03d}" / filename
                actual = sha256_file(path)
                if actual != manifest["artifact_hashes"].get(key):
                    raise ValueError(f"Ensemble artifact hash mismatch: {key}")
                self.artifact_hashes[key] = actual
        self.artifact_hashes["ensemble_manifest"] = sha256_file(manifest_path)
        self.members = [load_member(member_root / f"member_{index:03d}", self.device) for index in range(members)]''')]
elif args.fix == 'cache':
    patches = [
('src/robust_airfoil/optimisation/study.py',
'''import json
from itertools import combinations''',
'''import json
import platform
from importlib.metadata import version
from itertools import combinations'''),
('src/robust_airfoil/optimisation/study.py',
'''        "schema_version": 4,
        "cache_context_hash":''',
'''        "schema_version": 5,
        "cache_context_hash":'''),
('src/robust_airfoil/optimisation/study.py',
'''    development_matrix = development_points[FEATURE_COLUMNS].to_numpy(float)
    cache_context_hash = hash_object(''',
'''    development_matrix = development_points[FEATURE_COLUMNS].to_numpy(float)
    implementation_root = Path(__file__).parents[1]
    implementation_hashes = {
        path.relative_to(implementation_root).as_posix(): sha256_file(path)
        for path in sorted(implementation_root.rglob("*.py"))
    }
    cache_context_hash = hash_object('''),
('src/robust_airfoil/optimisation/study.py',
'''            "objective_schema_version": 4,
            "profile_name":''',
'''            "objective_schema_version": 5,
            "scientific_source_sha256": implementation_hashes,
            "actual_ensemble_artifact_sha256": ensemble.artifact_hashes,
            "dependency_versions": {
                name: version(name) for name in ("numpy", "scipy", "pandas", "torch", "pymoo")
            },
            "python_version": platform.python_version(),
            "inference_device": str(ensemble.device),
            "profile_name":'''),
('src/robust_airfoil/optimisation/study.py',
'''                and cached.get("schema_version") == 4''',
'''                and cached.get("schema_version") == 5
                and cached.get("seed") == seed'''),
('src/robust_airfoil/optimisation/study.py',
'''            # All schema-v2/v3 caches used reference-fitted CST increments and
            # are incompatible with candidate-specific physical perturbations.
            # Recompute under schema v4; do not migrate historical objectives.''',
'''            # A context mismatch is evidence from another scientific revision,
            # not permission to overwrite the historical result in place.
            raise RuntimeError(
                f"Incompatible cached evidence at {seed_path}; choose a fresh output revision"
            )''')]

original = {}
updated = {}
for relative, old, new in patches:
    path = ROOT/relative
    if relative not in updated:
        original[relative] = path.read_text(encoding='utf-8')
        updated[relative] = original[relative]
    if updated[relative].count(old) != 1:
        raise RuntimeError(f'Expected one exact patch anchor for {args.fix}: {relative}')
    updated[relative] = updated[relative].replace(old,new,1)
for relative, text in updated.items():
    path = ROOT/relative
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(text,encoding='utf-8',newline='\n')
    record = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fix':args.fix,'path':relative,'before_sha256':before,'after_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    with (OUT/'SOURCE_EDIT_LOG.jsonl').open('a',encoding='utf-8') as log:
        log.write(json.dumps(record)+'\n')
    print(json.dumps(record))
