"""Read-only inventory of the frozen scientific evidence; writes only this review."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
LIN = ROOT / 'results/robust_v2/lineages/robust-v2-current-20260908'

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

frames = {
    'points': ROOT / 'data/robust_v2/processed/full_exact/model_points.parquet',
    'airfoils': ROOT / 'data/robust_v2/processed/full_exact/airfoils.parquet',
    'calibration': LIN / 'calibration/calibration_predictions.parquet',
    'locked_test': LIN / 'locked_test/predictions.parquet',
}
report = {'frames': {}}
for name, path in frames.items():
    frame = pd.read_parquet(path)
    report['frames'][name] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'shape': list(frame.shape), 'columns': frame.columns.tolist(), 'airfoil_ids': int(frame.airfoil_id.nunique())}
    if name == 'airfoils':
        report['cst_fit_errors'] = frame[['cst_rms_error', 'cst_max_error']].describe(percentiles=[.5, .9, .95, .99]).to_dict()
split = pd.DataFrame(read_json(LIN / 'splits/full.json')['records'])
report['split_counts'] = split.groupby('split').agg(airfoils=('airfoil_id','nunique'), clusters=('cluster_id','nunique')).to_dict()
report['cross_split_clusters'] = split.groupby('cluster_id').split.nunique().loc[lambda s:s>1].to_dict()
report['calibration_cluster_sizes'] = split.loc[split.split=='calibration'].groupby('cluster_id').size().value_counts().to_dict()
report['frozen_manifest_paths'] = [p.relative_to(ROOT).as_posix() for p in ROOT.glob('**/FROZEN_MODEL_MANIFEST.json') if '.venv' not in p.parts]
manifest = read_json(LIN/'full_ensemble/ensemble_manifest.json')
report['ensemble_manifest'] = {k:v for k,v in manifest.items() if k!='summaries'}
report['member_summary_keys'] = list(manifest.get('summaries',[{}])[0])
report['member_architectures'] = [row.get('architecture') for row in manifest.get('summaries',[])]
report['member_epochs'] = [{k:row.get(k) for k in ['seed','epochs','best_epoch']} for row in manifest.get('summaries',[])]
report['paper_data_summary'] = {k:v for k,v in read_json(ROOT/'paper/data/paper_data.json').items() if k!='nsga2_full'}
report['october_xfoil_roots'] = {}
for folder in sorted((ROOT/'results/robust_v2/submission_review_20261014').iterdir()):
    if folder.is_dir():
        report['october_xfoil_roots'][folder.name] = [p.relative_to(folder).as_posix() for p in folder.glob('*')][:12]
(OUT/'EVIDENCE_INVENTORY.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['ensemble_manifest','paper_data_summary','member_architectures']}, indent=2))
print('READ_ONLY_EVIDENCE_INVENTORY_COMPLETE')
