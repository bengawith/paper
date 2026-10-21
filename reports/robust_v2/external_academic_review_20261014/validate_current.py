"""Record current checks in a uniquely named, immutable review validation folder."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('phase')
parser.add_argument('--author-pack',action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
base = Path(__file__).resolve().parent
out = base / ('validation_' + args.phase)
if out.exists(): raise RuntimeError('Choose a fresh validation phase; existing logs are preserved')
out.mkdir()
commands = [
 ('diff_check',['git','diff','--check']),
 ('ruff',['uv','run','ruff','check','src/robust_airfoil','tests/robust_v2','scripts/robust_v2']),
 ('mypy',['uv','run','mypy','src/robust_airfoil']),
 ('tests',['uv','run','pytest','tests/robust_v2','-m','not network','--run-xfoil','-q']),
]
if args.author_pack: commands.append(('author_pack',['uv','run','--extra','docs','python','scripts/robust_v2/verify_final_author_pack.py']))
env = dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
results = []
for name,command in commands:
 started = time.monotonic()
 with (out/(name+'.log')).open('w',encoding='utf-8') as log:
  log.write(json.dumps({'command':command,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})+'\n')
  log.flush()
  code = subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=False).returncode
 row = {'check':name,'returncode':code,'seconds':time.monotonic()-started,'command':command}
 results.append(row)
 (out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
 print(json.dumps(row),flush=True)
(out/'git_status.txt').write_text(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True),encoding='utf-8')
(out/'source_hashes.json').write_text(json.dumps({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src/robust_airfoil','scripts/robust_v2','tests/robust_v2'] for p in (root/folder).rglob('*.py')},indent=2),encoding='utf-8')
raise SystemExit(0 if all(r['returncode']==0 for r in results) else 1)
