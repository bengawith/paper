"""Reproduce the historical ID-grouped risk diagnostic, then repair its grouping.

Only calibration data are read. The aerodynamic ensemble and operational screen
are never fitted or changed. All fold normalisation/labels/models are train-only.
"""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
LIN = ROOT/'results/robust_v2/lineages/robust-v2-current-20260908'
OUT = Path(__file__).resolve().parent/'TRUST_RECOMPUTATION'
OUT.mkdir(exist_ok=True)
CAL = LIN/'calibration/calibration_predictions.parquet'
SPLIT = LIN/'splits/full.json'
raw = pd.read_parquet(CAL)
split = pd.DataFrame(json.loads(SPLIT.read_text(encoding='utf-8'))['records'])
cluster_map = split.set_index('airfoil_id').cluster_id
partition_map = split.set_index('airfoil_id').split
if not raw.airfoil_id.map(partition_map).eq('calibration').all():
    raise ValueError('Non-calibration rows are forbidden in this risk study')
TARGETS = ['cl','log_cd','cm']
FEATURES = {'support_only':['support_distance'], 'ensemble_only':['ensemble_std_'+t for t in TARGETS], 'support_plus_ensemble':['support_distance']+['ensemble_std_'+t for t in TARGETS]}
errors = np.column_stack([abs(raw[t]-raw['prediction_'+t]) for t in TARGETS])
valid = np.isfinite(errors).all(axis=1) & np.isfinite(raw[FEATURES['support_plus_ensemble']].to_numpy(float)).all(axis=1)
valid &= raw[['mask_'+t for t in TARGETS]].all(axis=1).to_numpy()
d = raw.loc[valid].reset_index(drop=True)
errors = errors[valid]
airfoil = d.airfoil_id.astype(str).to_numpy()
clusters = d.airfoil_id.map(cluster_map).to_numpy()
if pd.isna(clusters).any(): raise ValueError('Every row must map to a frozen geometry cluster')

def run(grouping):
    groups = airfoil if grouping=='airfoil_id' else clusters
    cv = GroupKFold(n_splits=5)
    predictions = {name:np.full(len(d),np.nan) for name in FEATURES}
    risk = np.full(len(d),np.nan)
    binary = np.full(len(d),-1,int)
    fold_id = np.zeros(len(d),int)
    records = []
    for fold,(train,val) in enumerate(cv.split(d,groups=groups),1):
        scales = np.maximum(np.quantile(errors[train],.9,axis=0),1e-8)
        train_score = np.max(errors[train]/scales,axis=1)
        held_score = np.max(errors[val]/scales,axis=1)
        threshold = float(np.quantile(train_score,.9))
        risk[val] = held_score
        binary[val] = (held_score>=threshold).astype(int)
        fold_id[val] = fold
        record = {'fold':fold,'train_rows':len(train),'evaluation_rows':len(val),'train_airfoil_ids':sorted(set(airfoil[train])),'evaluation_airfoil_ids':sorted(set(airfoil[val])),'train_clusters':sorted(int(x) for x in set(clusters[train])),'evaluation_clusters':sorted(int(x) for x in set(clusters[val])),'overlapping_geometry_clusters':sorted(int(x) for x in set(clusters[train])&set(clusters[val])),'train_only_target_p90':scales.tolist(),'train_only_high_error_threshold':threshold,'evaluation_high_error_rate':float(binary[val].mean()),'models':{}}
        for name,cols in FEATURES.items():
            model = make_pipeline(StandardScaler(),Ridge(alpha=10.0))
            model.fit(d.iloc[train][cols].to_numpy(float),np.log1p(train_score))
            predictions[name][val] = model.predict(d.iloc[val][cols].to_numpy(float))
            standardiser,ridge = model.steps[0][1],model.steps[1][1]
            record['models'][name] = {'features':cols,'feature_center':standardiser.mean_.tolist(),'feature_scale':standardiser.scale_.tolist(),'coefficients':ridge.coef_.tolist(),'intercept':float(ridge.intercept_),'fold_auroc':float(roc_auc_score(binary[val],predictions[name][val])),'fold_average_precision':float(average_precision_score(binary[val],predictions[name][val]))}
        records.append(record)
    if grouping=='geometry_cluster' and any(r['overlapping_geometry_clusters'] for r in records):
        raise AssertionError('Cluster-separated OOF construction leaked')
    if any(not np.isfinite(v).all() for v in predictions.values()):
        raise ValueError('Incomplete out-of-fold predictions')
    foil_counts = pd.Series(airfoil).value_counts()
    macro_weights = np.asarray([1/foil_counts[x] for x in airfoil])
    metrics = {name:{'auroc':float(roc_auc_score(binary,pred)),'average_precision':float(average_precision_score(binary,pred)),'geometry_balanced_auroc':float(roc_auc_score(binary,pred,sample_weight=macro_weights)),'mean_fold_auroc':float(np.mean([r['models'][name]['fold_auroc'] for r in records]))} for name,pred in predictions.items()}
    rng = np.random.default_rng(20261014)
    unique,index = np.unique(groups,return_inverse=True)
    bootstrap = []
    for _ in range(300):
        weights = rng.multinomial(len(unique),np.full(len(unique),1/len(unique)))[index]
        if not weights[binary==1].sum() or not weights[binary==0].sum(): continue
        bootstrap.append(float(roc_auc_score(binary,predictions['support_plus_ensemble'],sample_weight=weights)-roc_auc_score(binary,predictions['support_only'],sample_weight=weights)))
    fold_frame = pd.DataFrame({'airfoil_id':airfoil,'cluster_id':clusters,'fold':fold_id,'fold_scaled_error':risk,'train_fold_defined_large_error':binary,**predictions})
    fold_frame.to_csv(OUT/f'{grouping}_oof_rows.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    fold_frame[['airfoil_id','cluster_id','fold']].drop_duplicates().to_csv(OUT/f'{grouping}_fold_assignments.csv',index=False)
    result = {'grouping':grouping,'n_rows':len(d),'n_airfoils':len(set(airfoil)),'n_geometry_clusters':len(set(clusters)),'n_resampling_units':len(unique),'excluded_incomplete_rows':int((~valid).sum()),'models':metrics,'incremental_auroc':metrics['support_plus_ensemble']['auroc']-metrics['support_only']['auroc'],'bootstrap_replicates':len(bootstrap),'fixed_oof_group_bootstrap_95pct_interval':np.quantile(bootstrap,[.025,.975]).tolist(),'folds_with_cluster_overlap':[r['fold'] for r in records if r['overlapping_geometry_clusters']],'folds':records,'interpretation':'Retrospective calibration-distribution error-ranking diagnostic. Interval resamples fixed OOF predictions; it does not include refitting variability, establish conditional coverage, or validate an operational threshold under design shift.'}
    (OUT/f'{grouping}_CV.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['folds','interpretation']},indent=2),flush=True)
    return result

historical = run('airfoil_id')
corrected = run('geometry_cluster')
summary = {'schema':'calibration-only-cluster-separated-risk-review-v1','calibration_sha256':hashlib.sha256(CAL.read_bytes()).hexdigest(),'split_sha256':hashlib.sha256(SPLIT.read_bytes()).hexdigest(),'historical_id_grouping':{k:v for k,v in historical.items() if k!='folds'},'corrected_cluster_grouping':{k:v for k,v in corrected.items() if k!='folds'},'aerodynamic_model_changed':False,'operational_trust_threshold_changed':False,'locked_test_accessed_by_this_script':False}
(OUT/'TRUST_REVIEW_SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
print('CALIBRATION_ONLY_GROUPING_REVIEW_FINISHED',flush=True)
