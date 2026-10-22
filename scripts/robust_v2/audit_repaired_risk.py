"""Calibration-only geometry-cluster OOF risk analysis for the rebuilt model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from robust_airfoil.data.splits import calibration_geometry_groups
from robust_airfoil.hashing import sha256_file
from robust_airfoil.research_lineage import resolve_lineage

ROOT=Path(__file__).resolve().parents[2]
LINEAGE=ROOT/'results/robust_v2/lineages/robust-v2-rebuilt-20261014'
OUT=ROOT/'reports/robust_v2/external_academic_review_20261014/REBUILT_MODEL/RISK_OOF'
TARGETS=['cl','log_cd','cm']
FEATURES={'support_only':['support_distance'],'ensemble_only':['ensemble_std_'+t for t in TARGETS],'support_plus_ensemble':['support_distance']+['ensemble_std_'+t for t in TARGETS]}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--lineage', choices=['rebuilt','controlled'], default='rebuilt', help='Explicit immutable scientific revision; never overwrites the other lineage')
    args=parser.parse_args()
    paths=resolve_lineage(ROOT,args.lineage)
    LINEAGE=paths['results']
    OUT=paths['reports']/'RISK_OOF'
    if (OUT/'REPAIRED_RISK_OOF.json').exists():
        raise FileExistsError('The completed repaired risk analysis is immutable; use a new revision')
    OUT.mkdir(parents=True,exist_ok=True)
    source=LINEAGE/'calibration/calibration_predictions.parquet'
    splits_path=LINEAGE/'splits/full.json'
    d=pd.read_parquet(source)
    errors=np.column_stack([abs(d[t]-d['prediction_'+t]) for t in TARGETS])
    valid=np.isfinite(errors).all(axis=1)&np.isfinite(d[FEATURES['support_plus_ensemble']].to_numpy(float)).all(axis=1)&d[['mask_'+t for t in TARGETS]].all(axis=1).to_numpy()
    excluded=int((~valid).sum())
    d=d.loc[valid].reset_index(drop=True)
    errors=errors[valid]
    groups=calibration_geometry_groups(d,pd.DataFrame(json.loads(splits_path.read_text(encoding='utf-8'))['records']))
    oof={name:np.full(len(d),np.nan) for name in FEATURES}
    high=np.full(len(d),-1,int)
    score=np.full(len(d),np.nan)
    assigned=np.zeros(len(d),int)
    folds=[]
    for fold,(train,val) in enumerate(GroupKFold(5).split(d,groups=groups),1):
        if set(groups[train])&set(groups[val]):
            raise AssertionError('Geometry clusters overlap between fitting and evaluation')
        scales=np.maximum(np.quantile(errors[train],.9,axis=0),1e-8)
        train_score=np.max(errors[train]/scales,axis=1)
        validation_score=np.max(errors[val]/scales,axis=1)
        threshold=float(np.quantile(train_score,.9))
        score[val]=validation_score
        high[val]=(validation_score>=threshold).astype(int)
        assigned[val]=fold
        fitted={}
        for name,columns in FEATURES.items():
            model=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
            model.fit(d.iloc[train][columns].to_numpy(float),np.log1p(train_score))
            oof[name][val]=model.predict(d.iloc[val][columns].to_numpy(float))
            scaler,ridge=model.steps[0][1],model.steps[1][1]
            fitted[name]={'columns':columns,'feature_center':scaler.mean_.tolist(),'feature_scale':scaler.scale_.tolist(),'coefficients':ridge.coef_.tolist(),'intercept':float(ridge.intercept_),'fold_auroc':float(roc_auc_score(high[val],oof[name][val]))}
        folds.append({'fold':fold,'train_clusters':[int(x) for x in sorted(set(groups[train]))],'evaluation_clusters':[int(x) for x in sorted(set(groups[val]))],'train_rows':len(train),'evaluation_rows':len(val),'training_residual_p90':scales.tolist(),'training_high_error_threshold':threshold,'evaluation_high_error_fraction':float(high[val].mean()),'models':fitted})
    if (assigned==0).any() or any(not np.isfinite(v).all() for v in oof.values()):
        raise AssertionError('Incomplete OOF coverage')
    unique,index=np.unique(groups,return_inverse=True)
    cluster_counts=np.bincount(index)
    balanced=1./cluster_counts[index]
    metrics={name:{'auroc':float(roc_auc_score(high,pred)),'average_precision':float(average_precision_score(high,pred)),'cluster_balanced_auroc':float(roc_auc_score(high,pred,sample_weight=balanced)),'mean_fold_auroc':float(np.mean([r['models'][name]['fold_auroc'] for r in folds]))} for name,pred in oof.items()}
    bootstrap=[]
    rng=np.random.default_rng(20261014)
    for _ in range(300):
        weights=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)))[index]
        if not weights[high==0].sum() or not weights[high==1].sum():
            continue
        bootstrap.append(float(roc_auc_score(high,oof['support_plus_ensemble'],sample_weight=weights)-roc_auc_score(high,oof['support_only'],sample_weight=weights)))
    rows=pd.DataFrame({'airfoil_id':d.airfoil_id,'geometry_cluster':groups,'fold':assigned,'fold_train_defined_high_error':high,'scaled_error':score,**oof})
    rows.to_csv(OUT/'oof_rows.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    rows[['airfoil_id','geometry_cluster','fold']].drop_duplicates().to_csv(OUT/'fold_membership.csv',index=False)
    report={'schema':'rebuilt-calibration-only-cluster-risk-v1','states':len(d),'airfoils':int(d.airfoil_id.nunique()),'geometry_clusters':len(unique),'excluded_incomplete_rows':excluded,'models':metrics,'incremental_auroc':metrics['support_plus_ensemble']['auroc']-metrics['support_only']['auroc'],'fixed_oof_cluster_bootstrap_replicates':len(bootstrap),'fixed_oof_cluster_bootstrap_95pct_interval':np.quantile(bootstrap,[.025,.975]).tolist(),'folds':folds,'source_sha256':sha256_file(source),'split_sha256':sha256_file(splits_path),'oof_sha256':sha256_file(OUT/'oof_rows.csv.gz'),'locked_test_read':False,'operational_policy_changed':False,'scope':'Retrospective error ranking within repaired calibration distribution; all per-fold fitting/normalisation/labels are train-fold-only; fixed-OOF bootstrap does not include complete refitting variability or establish optimiser-shift coverage.'}
    (OUT/'REPAIRED_RISK_OOF.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='folds'},indent=2),flush=True)


if __name__ == '__main__':
    main()
