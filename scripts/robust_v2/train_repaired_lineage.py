"""Train five genuinely new members; only development OOB clusters choose stopping."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import pandas as pd
import torch

from robust_airfoil.hashing import sha256_file
from robust_airfoil.modelling.ensemble import cluster_bootstrap_with_oob
from robust_airfoil.modelling.train import train_model
from robust_airfoil.research_lineage import resolve_lineage

ROOT = Path(__file__).resolve().parents[2]
LINEAGE = ROOT/'results/robust_v2/lineages/robust-v2-rebuilt-20261014'
DATA = ROOT/'data/robust_v2/processed/rebuilt_20261014'
REVIEW = ROOT/'reports/robust_v2/external_academic_review_20261014/REBUILT_MODEL'
OLD = ROOT/'results/robust_v2/lineages/robust-v2-current-20260908/full_ensemble/ensemble_manifest.json'


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--lineage', choices=['rebuilt','controlled'], default='rebuilt', help='Explicit immutable scientific revision; never overwrites the other lineage')
    args=parser.parse_args()
    paths=resolve_lineage(ROOT,args.lineage)
    LINEAGE=paths['results']
    DATA=paths['data']
    REVIEW=paths['reports']
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    data_manifest = json.loads((LINEAGE/'DATA_MANIFEST.json').read_text(encoding='utf-8'))
    if sha256_file(DATA/'model_points.parquet') != data_manifest['data_files']['model_points.parquet']:
        raise ValueError('Rebuilt corpus changed after its data manifest')
    # This is a deterministic partition extraction, not model selection or
    # inspection of calibration/test outcomes. Fitting below uses only dev.
    input_root = LINEAGE/'partition_inputs'
    input_root.mkdir(exist_ok=True)
    if not (input_root/'PARTITIONS.json').exists():
        frame = pd.read_parquet(DATA/'model_points.parquet')
        partition_hashes = {}
        for name in ('development','calibration','locked_test'):
            output = input_root/(name+'.parquet')
            frame.loc[frame['split'].eq(name)].to_parquet(output,index=False)
            partition_hashes[name] = sha256_file(output)
        (input_root/'PARTITIONS.json').write_text(json.dumps({'source_sha256':sha256_file(DATA/'model_points.parquet'),'partition_sha256':partition_hashes},indent=2),encoding='utf-8')
        del frame
    partitions = json.loads((input_root/'PARTITIONS.json').read_text(encoding='utf-8'))
    if sha256_file(input_root/'development.parquet') != partitions['partition_sha256']['development']:
        raise ValueError('Development partition hash changed')
    development = pd.read_parquet(input_root/'development.parquet')
    if not development['split'].eq('development').all():
        raise ValueError('Non-development observations in trainer input')
    hyper = json.loads(OLD.read_text(encoding='utf-8'))['hyperparameters']
    members = LINEAGE/'full_ensemble'
    members.mkdir(exist_ok=True)
    REVIEW.mkdir(parents=True,exist_ok=True)
    summaries, hashes = [], {}
    start = time.monotonic()
    for index in range(5):
        seed = 2026101400+index
        bootstrap, stopping, selection = cluster_bootstrap_with_oob(development,seed)
        directory = members/f'member_{index:03d}'
        directory.mkdir(exist_ok=True)
        (directory/'BOOTSTRAP_AND_OOB.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        print(f'START_MEMBER={index} train_rows={len(bootstrap)} stopping_rows={len(stopping)} seed={seed}',flush=True)
        summary = train_model(
            bootstrap,stopping,directory,seed=seed,
            hidden_width=int(hyper['hidden_width']),hidden_layers=int(hyper['hidden_layers']),
            activation=hyper['activation'],dropout=float(hyper['dropout']),
            learning_rate=float(hyper['learning_rate']),weight_decay=float(hyper['weight_decay']),
            batch_size=int(hyper['batch_size']),maximum_epochs=300,patience=25,
            loss_name=hyper['loss'],residual=bool(hyper['residual']),layer_norm=True,
            resident_batches=True,checkpoint_each_epoch=True,
            resume=(directory/'training_state.pt').exists(),
        )
        summary['cluster_bootstrap'] = selection
        summaries.append(summary)
        for label,filename in [('checkpoint','checkpoint.pt'),('scaling','scaling.joblib')]:
            hashes[f'member_{index:03d}_{label}'] = sha256_file(directory/filename)
        progress = {'status':'TRAINING_IN_PROGRESS','completed_members':index+1,'members_required':5,'elapsed_seconds':time.monotonic()-start,'completed_epochs':[int(s['epochs']) for s in summaries],'best_epochs':[int(s['best_epoch'])+1 for s in summaries]}
        (REVIEW/'TRAINING_PROGRESS.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
        print('COMPLETED_MEMBER',index,'epochs',summary['epochs'],'best_epoch',int(summary['best_epoch'])+1,flush=True)
    manifest = {'schema':'repaired-geometry-five-member-oob-stopped-ensemble-v1','lineage':LINEAGE.name,'member_count':5,'base_seed':2026101400,'maximum_epochs':300,'patience':25,'hyperparameters':hyper,'architecture_selection':'inherited fixed baseline settings; no architecture search or calibration/test tuning','stopping':'per-member out-of-bag development geometry clusters only','data_manifest_sha256':sha256_file(LINEAGE/'DATA_MANIFEST.json'),'development_partition_sha256':sha256_file(input_root/'development.parquet'),'artifact_hashes':hashes,'summaries':summaries,'elapsed_seconds':time.monotonic()-start,'device':str(next(iter(torch.cuda.get_device_name() for _ in [0])) if torch.cuda.is_available() else 'cpu')}
    text = json.dumps(manifest,indent=2,default=lambda value:value.tolist() if hasattr(value,'tolist') else str(value))
    (members/'ensemble_manifest.json').write_text(text,encoding='utf-8')
    (REVIEW/'ENSEMBLE_MANIFEST.json').write_text(text,encoding='utf-8')
    (REVIEW/'TRAINING_PROGRESS.json').write_text(json.dumps({'status':'FIVE_MEMBERS_TRAINED','completed_members':5,'epochs':[int(s['epochs']) for s in summaries],'best_epochs':[int(s['best_epoch'])+1 for s in summaries],'elapsed_seconds':time.monotonic()-start},indent=2),encoding='utf-8')
    print('FIVE_MEMBER_REPAIRED_ENSEMBLE_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
