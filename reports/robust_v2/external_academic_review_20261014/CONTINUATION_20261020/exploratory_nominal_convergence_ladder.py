"""Post-hoc numerical convergence sensitivity for all three fixed nominal designs.

Triggered by original predeclared ITER200 XFOIL missing 1-5 requested angles
despite zero exit. Apply the same ITER400/ITER800 ladder to ALL three design
geometries, unchanged panel count (160), 45-angle/required-lift and warmup
rules. Preserve original invalid statuses. Secondary exploratory results must
NOT replace the preregistered strict 200-iteration acceptance numbers or be
described as independent CFD or physical model improvement.
"""
from __future__ import annotations
import json
import tempfile
import shutil
import hashlib
import subprocess
import datetime
from pathlib import Path
import pandas as pd
from robust_airfoil.data.polar_parser import parse_polar_file
from robust_airfoil.hashing import sha256_file
from robust_airfoil.validation.controlled_selected import selected_xfoil_commands,assess_selected_polar

ROOT=Path(__file__).resolve().parents[4]
LINEAGE=ROOT/'results/robust_v2/lineages/robust-v2-controlled-20261020'
BASE=LINEAGE/'selected_study/xfoil'
OUT=ROOT/'reports/robust_v2/external_academic_review_20261014/EXPLORATORY_NOMINAL_CONVERGENCE'
EXE=ROOT/'tools/xfoil/xfoil.exe'
ITERATIONS=(400,800)
DESIGNS=('reference','nominal_control','risk_knee')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return sha256_file(path)
def run(design,iterations):
 source=BASE/design/'nominal/geometry.dat'
 original=BASE/design/'nominal/run/CASE_RECEIPT.json'
 previous=json.loads(original.read_text(encoding='utf-8'))
 if previous['assessment']['usable']:
  print('Original 200-iteration result was usable, but the declared equal-control convergence ladder is still run',design,flush=True)
 if sha(source)!=previous['identity']['geometry_sha256']:
  raise ValueError('Frozen nominal geometry differs from original selected solver case')
 folder=OUT/design/f'iter_{iterations}'
 if folder.exists():raise FileExistsError('Preserve prior exploratory solver outcome: '+str(folder))
 folder.mkdir(parents=True)
 shutil.copy2(source,folder/'geometry.dat')
 command=selected_xfoil_commands(160)
 if command.count('ITER 200')!=1:raise ValueError('Original strict command not identifiable')
 command=command.replace('ITER 200',f'ITER {iterations}')
 (folder/'command.txt').write_text(command,encoding='utf-8')
 with tempfile.TemporaryDirectory(prefix='xfoil-exploratory-nominal-') as tmp:
  d=Path(tmp)
  shutil.copy2(source,d/'geometry.dat')
  try:
   output=subprocess.run([str(EXE.resolve())],input=command,cwd=d,
     capture_output=True,encoding='utf-8',errors='replace',
     text=True,timeout=90,check=False)
   code=output.returncode
   timeout=False
   stdout,stderr=output.stdout,output.stderr
  except subprocess.TimeoutExpired as exc:
   code=None;timeout=True
   stdout=exc.stdout.decode('utf-8',errors='replace') if isinstance(exc.stdout,bytes) else exc.stdout or ''
   stderr=exc.stderr.decode('utf-8',errors='replace') if isinstance(exc.stderr,bytes) else exc.stderr or ''
  (folder/'stdout.txt').write_text(stdout,encoding='utf-8')
  (folder/'stderr.txt').write_text(stderr,encoding='utf-8')
  points=pd.DataFrame()
  if (d/'polar.txt').exists():
   shutil.copy2(d/'polar.txt',folder/'polar.txt')
   try:points=parse_polar_file(folder/'polar.txt').points
   except (ValueError,IndexError,RuntimeError) as error:
    print('POLAR_PARSE_FAILURE',design,iterations,str(error),flush=True)
  points.to_csv(folder/'parsed_points.csv',index=False)
 result=assess_selected_polar(points,return_code=code,timed_out=timeout)
 fields={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
 record={'schema':'exploratory-nominal-iteration-sensitivity-v1',
  'design':design,'iterations':iterations,'computed_utc':utc(),
  'original_200_iteration_receipt_sha256':sha(original),
  'original_200_iteration_usable':previous['assessment']['usable'],
  'source_nominal_geometry_sha256':sha(source),
  'panel_count':160,'solver_version':'local-XFOIL-6.99',
  'executable_sha256':sha(EXE),'timeout_seconds':90,'assessment':result,
  'source_files_sha256':fields,
  'interpretation':'Post-hoc equal-control convergence sensitivity only; DOES NOT supersede frozen 200-iteration strict acceptance, and is NOT independent CFD.'}
 (folder/'CASE_RECEIPT.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
 print('CONVERGENCE_SENSITIVITY',design,iterations,'usable',result['usable'],'missing',len(result['missing_alpha_deg']),flush=True)
 return record
def main():
 selected=LINEAGE/'selected_study/SELECTED_XFOIL_SUMMARY.json'
 if not selected.is_file():raise FileNotFoundError('The original strict 819-case study must be terminal before a post-hoc convergence ladder')
 if OUT.exists():raise FileExistsError('Prior exploratory ladder revision exists')
 OUT.mkdir(parents=True)
 sources={'strict_selected_sha256':sha(selected),'selection_sha256':sha(LINEAGE/'selected_study/FROZEN_SELECTED_DESIGNS.json'),
   'executable_sha256':sha(EXE),'panel_count':160,'iterations':[400,800],
   'designs':list(DESIGNS),'study_order':'All three designs at both iteration counts, irrespective of nominal original solver success',
   'no_posthoc_original_acceptance_replacement':True,'exploratory_only':True}
 (OUT/'PROTOCOL.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
 results=[]
 for n in ITERATIONS:
  for name in DESIGNS:results.append(run(name,n))
 summary={'schema':'posthoc-all-designs-nominal-xfoil-iteration-sensitivity-v1',
   'protocol_sha256':sha(OUT/'PROTOCOL.json'),
   'original_frozen_strict_sha256':sha(selected),
   'six_cases_attempted':len(results),'original_200_usable':sum(r['original_200_iteration_usable'] for r in results)//2,
   'iter_400_usable':sum(r['assessment']['usable'] for r in results if r['iterations']==400),
   'iter_800_usable':sum(r['assessment']['usable'] for r in results if r['iterations']==800),
   'results':[{'design':r['design'],'iterations':r['iterations'],
      'usable':r['assessment']['usable'],'weighted_cd':r['assessment']['weighted_cd'],
      'missing_alpha_deg':r['assessment']['missing_alpha_deg']} for r in results],
   'scope':'Exploratory convergence sensitivity due original VISCAL nonconvergence, not the predeclared strict verification or independent flow physics.'}
 (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print('ALL_EXPLORATORY_NOMINAL_LADDER_COMPLETE',summary['iter_400_usable'],summary['iter_800_usable'],flush=True)
if __name__=='__main__':main()
