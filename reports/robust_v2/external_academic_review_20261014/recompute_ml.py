"""Independent frozen-evidence arithmetic, split/scaler audit and grouped risk reanalysis.

Never trains or changes the aerodynamic surrogate. All outputs are new review artefacts.
The risk model uses calibration labels only, never locked-test labels for fitting/tuning.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS, TARGET_COLUMNS
from robust_airfoil.modelling.ensemble import LoadedEnsemble

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / 'ML_RECOMPUTATION'
OUT.mkdir(exist_ok=True)
LIN = ROOT/'results/robust_v2/lineages/robust-v2-current-20260908'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'scope': 'Frozen numerical evidence verification; retrospective diagnostics, not a fresh confirmation set.'}
frozen = read(LIN/'FROZEN_MODEL_MANIFEST.json')
checks = []
for key, expected in frozen['artifact_hashes'].items():
    path = Path(frozen['artifact_paths'][key])
    actual = sha(path) if path.is_file() else None
    checks.append({'key':key, 'path':str(path), 'expected':expected, 'actual':actual, 'matches':actual==expected})
report['frozen_artifacts'] = checks
if not all(r['matches'] for r in checks):
    raise RuntimeError('A frozen input differs: stop inference and preserve the failed check')
points = pd.read_parquet(ROOT/'data/robust_v2/processed/full_exact/model_points.parquet')
airfoils = pd.read_parquet(ROOT/'data/robust_v2/processed/full_exact/airfoils.parquet')
split = pd.DataFrame(read(LIN/'splits/full.json')['records'])
folds_tuning = pd.DataFrame(read(LIN/'tuning/fold_manifest.json')['records'])
cal = pd.read_parquet(LIN/'calibration/calibration_predictions.parquet')
test = pd.read_parquet(LIN/'locked_test/predictions.parquet')
calibration = read(LIN/'calibration/trust_calibration.json')
assignments = split.set_index('airfoil_id')
report['split_counts'] = split.groupby('split').agg(airfoils=('airfoil_id','nunique'), clusters=('cluster_id','nunique')).reset_index().to_dict('records')
report['split_cluster_overlap'] = int((split.groupby('cluster_id').split.nunique()>1).sum())
report['tuning_cluster_overlap'] = int((folds_tuning.groupby('cluster_id').fold.nunique()>1).sum())
report['tuning_outside_development'] = sorted(set(folds_tuning.airfoil_id)-set(split.loc[split.split=='development','airfoil_id']))
report['conditions'] = points[['reynolds_number','mach','ncrit']].drop_duplicates().to_dict('records')
report['point_count'] = len(points)
report['mask_counts'] = {t:int(points['mask_'+t].sum()) for t in TARGET_COLUMNS}
report['ineligible_rows'] = int((~points.model_eligible).sum())
report['duplicate_airfoil_alpha_rows'] = int(points.duplicated(['airfoil_id','alpha_deg']).sum())
report['calibration_source_sha256'] = sha(LIN/'calibration/calibration_predictions.parquet')
report['locked_test_source_sha256'] = sha(LIN/'locked_test/predictions.parquet')
report['test_id_assignment_errors'] = int((test.airfoil_id.map(assignments.split)!='locked_test').sum())
report['calibration_id_assignment_errors'] = int((cal.airfoil_id.map(assignments.split)!='calibration').sum())
report['geometry_hash_cross_split'] = int((airfoils.merge(split[['airfoil_id','split']],on='airfoil_id').groupby('normalised_geometry_hash').split.nunique()>1).sum())
report['cst_error_statistics'] = airfoils[['cst_rms_error','cst_max_error']].describe(percentiles=[.5,.9,.95,.99]).to_dict()
report['cst_max_error_exceeds_0p5pct'] = int((airfoils.cst_max_error>.005).sum())
report['cst_worst_examples'] = airfoils.nlargest(10,'cst_max_error')[['airfoil_id','cst_rms_error','cst_max_error','geometry_path']].to_dict('records')
report['naca2412_fit'] = airfoils.loc[airfoils.airfoil_id=='naca2412',['airfoil_id','cst_rms_error','cst_max_error']].to_dict('records')
dev = points.loc[points.airfoil_id.isin(split.loc[split.split=='development','airfoil_id'])].merge(airfoils[['airfoil_id','cluster_id']],on='airfoil_id',validate='many_to_one')
manifest = read(LIN/'full_ensemble/ensemble_manifest.json')
scaler_checks = []
for index, member in enumerate(manifest['summaries']):
    cluster_draws = member['cluster_bootstrap']['sampled_cluster_ids']
    bootstrap = pd.concat([dev.loc[dev.cluster_id==c] for c in cluster_draws], ignore_index=True)
    scaling = joblib.load(LIN/f'full_ensemble/member_{index:03d}/scaling.joblib')
    values = bootstrap[FEATURE_COLUMNS].to_numpy(float)
    center = np.nanmedian(values,axis=0)
    spread = np.nanpercentile(values,75,axis=0)-np.nanpercentile(values,25,axis=0)
    spread[spread<1e-9] = 1
    differences = [float(np.max(np.abs(center-scaling.feature_center))),float(np.max(np.abs(spread-scaling.feature_scale)))]
    for target in TARGET_COLUMNS:
        y = bootstrap.loc[bootstrap['mask_'+target],target].to_numpy(float)
        scale = float(np.percentile(y,75)-np.percentile(y,25))
        if scale<1e-9: scale=1
        differences.extend([abs(float(np.median(y))-scaling.target_center[target]),abs(scale-scaling.target_scale[target])])
    scaler_checks.append({'member':index,'bootstrap_draws':len(cluster_draws),'unique_clusters':len(set(cluster_draws)),'outside_development_clusters':len(set(cluster_draws)-set(dev.cluster_id)),'maximum_scaling_difference':max(differences),'epochs':member['epochs'],'best_epoch':member['best_epoch'],'architecture':member['architecture']})
report['train_only_scalers'] = scaler_checks
print('Frozen hashes, splits and five train-only scalers checked',flush=True)
ensemble = LoadedEnsemble(LIN/'full_ensemble')
report['architecture'] = {'member_parameters':sum(p.numel() for p in ensemble.members[0][0].parameters()), 'linear_modules':sum(isinstance(m,torch.nn.Linear) for m in ensemble.members[0][0].modules()),'residual_blocks':len(ensemble.members[0][0].blocks),'interpretation':'One input projection, two two-linear residual blocks, three output heads; five hidden affine layers, not two hidden affine layers.'}
start = time.monotonic()
fresh = ensemble.predict(test)
report['fresh_frozen_prediction'] = {'seconds':time.monotonic()-start,'device':str(ensemble.device),'max_absolute_difference_by_target':{t:float(np.max(np.abs(fresh['prediction_'+t]-test['prediction_'+t]))) for t in TARGET_COLUMNS}}
print('Frozen locked-test inference repeated',flush=True)
sc = ensemble.first_scaling
matrix = (dev[FEATURE_COLUMNS].to_numpy(float)-sc.feature_center)/sc.feature_scale
distances = cKDTree(matrix).query((test[FEATURE_COLUMNS].to_numpy(float)-sc.feature_center)/sc.feature_scale,k=5)[0].mean(axis=1)
support = distances<=calibration['support_distance_quantiles']['trusted_threshold_from_residual_tolerance']
alpha_ok = test.alpha_deg.between(*calibration['alpha_coverage_deg']).to_numpy()
spread_ok = {t:(test['ensemble_std_'+t]<=calibration['ensemble_disagreement_quantiles'][t]['trusted_threshold_from_residual_tolerance']).to_numpy() for t in TARGET_COLUMNS}
trusted = support & alpha_ok
for v in spread_ok.values(): trusted &= v
report['trust'] = {'distance_recompute_max_error':float(np.max(np.abs(distances-test.support_distance))),'accepted':int(trusted.sum()),'rejected':int((~trusted).sum()),'stored_flag_disagreements':int(np.sum(trusted!=test.trusted_domain.to_numpy())),'support_rejected':int((~support).sum()),'angle_rejected':int((~alpha_ok).sum()),'spread_rejected':{t:int((~v).sum()) for t,v in spread_ok.items()},'spread_additional_rejections':int(np.sum(support & alpha_ok & ~np.logical_and.reduce(list(spread_ok.values()))))}
metrics = []
for group, mask in [('all',np.ones(len(test),bool)),('accepted',trusted),('rejected',~trusted)]:
    for target in ['cl','log_cd','cm','cd']:
        base = 'log_cd' if target=='cd' else target
        valid = mask & test['mask_'+base].to_numpy(bool)
        actual = test.loc[valid,target].to_numpy(float)
        predicted = np.exp(test.loc[valid,'prediction_log_cd'].to_numpy(float)) if target=='cd' else test.loc[valid,'prediction_'+target].to_numpy(float)
        by_airfoil = pd.DataFrame({'id':test.loc[valid,'airfoil_id'].to_numpy(),'ae':np.abs(actual-predicted),'se':(actual-predicted)**2}).groupby('id').mean()
        row = {'subset':group,'target':target,'n_rows':int(valid.sum()),'n_airfoils':len(by_airfoil),'mae':mean_absolute_error(actual,predicted),'mse':mean_squared_error(actual,predicted),'r2':r2_score(actual,predicted),'geometry_macro_mae':float(by_airfoil.ae.mean()),'geometry_macro_mse':float(by_airfoil.se.mean())}
        if target!='cd':
            covered = np.abs(actual-predicted)<=calibration['residual_absolute_quantiles'][target]['0.95']
            per_foil = pd.DataFrame({'id':test.loc[valid,'airfoil_id'].to_numpy(),'covered':covered}).groupby('id').covered.agg(['mean','all'])
            row.update(interval_coverage=float(covered.mean()),geometry_macro_interval_coverage=float(per_foil['mean'].mean()),whole_observed_curve_coverage=float(per_foil['all'].mean()))
        metrics.append(row)
report['metrics'] = metrics
pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
by_fit = test.merge(airfoils[['airfoil_id','cst_max_error']],on='airfoil_id',validate='many_to_one')
by_fit['fit_bin'] = pd.cut(by_fit.cst_max_error,[0,.001,.005,.01,.02,np.inf],include_lowest=True).astype(str)
by_fit['absolute_cl_error'] = abs(by_fit.cl-by_fit.prediction_cl)
report['fit_error_strata'] = by_fit.groupby('fit_bin',observed=True).agg(rows=('cl','size'),foils=('airfoil_id','nunique'),mean_absolute_cl_error=('absolute_cl_error','mean'),accepted_fraction=('trusted_domain','mean')).reset_index().to_dict('records')
(OUT/'ML_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'trust':report['trust'],'architecture':report['architecture'],'fresh':report['fresh_frozen_prediction'],'scaler_differences':[r['maximum_scaling_difference'] for r in scaler_checks]},indent=2),flush=True)
