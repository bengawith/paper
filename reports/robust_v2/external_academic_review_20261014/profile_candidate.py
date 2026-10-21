"""Measure current candidate-specific physical uncertainty before allocating a campaign."""
from pathlib import Path
import cProfile
import io
import json
import os
import pstats
import time
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import pandas as pd
import torch
from robust_airfoil.config import OptimisationConfig, UncertaintyConfig, load_yaml
from robust_airfoil.modelling.ensemble import LoadedEnsemble
from robust_airfoil.optimisation.study import RobustCandidateEvaluator
from robust_airfoil.uncertainty.sampling import sobol_normal_samples

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent/'COMPUTE_PROFILE'
OUT.mkdir(exist_ok=True)
LIN = ROOT/'results/robust_v2/lineages/robust-v2-current-20260908'
torch.set_num_threads(2)
opt = OptimisationConfig.model_validate(load_yaml(ROOT/'configs/robust_v2/optimisation.yaml'))
unc = UncertaintyConfig.model_validate(load_yaml(ROOT/'configs/robust_v2/uncertainty.yaml'))
points = pd.read_parquet(ROOT/'data/robust_v2/processed/full_exact/model_points.parquet')
splits = json.loads((LIN/'splits/full.json').read_text())['records']
dev = points[points.airfoil_id.isin([r['airfoil_id'] for r in splits if r['split']=='development'])]
selected = json.loads((LIN/'optimisation/selected_designs.json').read_text())
ref = np.asarray(next(r['parameters'] for r in selected if r['design_id']=='verified_reference'))
ensemble = LoadedEnsemble(LIN/'full_ensemble')
cal = json.loads((LIN/'calibration/trust_calibration.json').read_text())
latent = sobol_normal_samples(64,16,unc.seed+2000,antithetic=True)
evaluator = RobustCandidateEvaluator(ref,np.zeros((64,12)),np.zeros(64,bool),dev,ensemble,cal,opt,candidate_latent=latent,uncertainty=unc)
profile = cProfile.Profile()
start = time.monotonic()
profile.enable()
f,g,d = evaluator.evaluate(np.asarray([r['parameters'] for r in selected]))
profile.disable()
seconds = time.monotonic()-start
buffer = io.StringIO()
pstats.Stats(profile,stream=buffer).strip_dirs().sort_stats('cumulative').print_stats(35)
(OUT/'profile.txt').write_text(buffer.getvalue(),encoding='utf-8')
rows = [{'design_id':r['design_id'],'parameters':r['parameters'],'objectives':f[i].tolist(),'constraints':g[i].tolist(),'diagnostics':d[i]} for i,r in enumerate(selected)]
result = {'seconds_five_candidates_64_scenarios':seconds,'device':str(ensemble.device),'rows':rows,'nominal_feasible':[bool(np.all(row<=0)) for row in g],'scope':'New v5 code evaluates historical fixed designs; this is not a new optimisation.'}
(OUT/'CURRENT_SELECTED_DESIGNS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'seconds':seconds,'nominal_feasible':result['nominal_feasible'],'objectives':f.tolist()},indent=2))
print(buffer.getvalue())
