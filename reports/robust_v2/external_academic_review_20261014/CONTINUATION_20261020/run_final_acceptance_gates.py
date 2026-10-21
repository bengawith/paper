"""Rerun actual repository release gates without touching preserved science."""
from __future__ import annotations
import datetime,json,os,subprocess,time,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
RESULT=HERE/'FINAL_SOFTWARE_ACCEPTANCE_GATES.json'
ENV=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
COMMANDS=[
 ('uv_sync',['uv','sync','--extra','dev','--extra','benchmarks','--extra','docs']),
 ('ruff',['uv','run','ruff','check','src/robust_airfoil','tests/robust_v2','scripts/robust_v2']),
 ('mypy',['uv','run','mypy','src/robust_airfoil']),
 ('prescribed_unit',['uv','run','pytest','tests/robust_v2','-m','not network and not xfoil and not slow','-q']),
 ('full_nonnetwork_xfoil',['uv','run','pytest','-m','not network','--run-xfoil','-q']),
 ('author_acceptance',['uv','run','--extra','docs','python','scripts/robust_v2/verify_final_author_pack.py']),
 ('git_diff_check',['git','diff','--check']),
]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for part in iter(lambda:f.read(1048576),b''):h.update(part)
 return h.hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 if RESULT.exists():raise FileExistsError('Final release QA already run in this revision')
 records=[]
 for name,cmd in COMMANDS:
  log=HERE/('FINAL_ACCEPTANCE_'+name+'.log')
  if log.exists():raise FileExistsError('Original final QA log exists: '+str(log))
  start=time.monotonic()
  with log.open('w',encoding='utf-8') as output:
   output.write(json.dumps({'command':cmd,'utc':utc(),'cwd':str(ROOT)})+'\n');output.flush()
   result=subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=output,stderr=subprocess.STDOUT,check=False)
  item={'name':name,'command':cmd,'returncode':result.returncode,'duration_seconds':round(time.monotonic()-start,3),
        'log':log.relative_to(ROOT).as_posix(),'log_sha256':sha(log),'finished_utc':utc()}
  records.append(item)
  RESULT.write_text(json.dumps({'schema':'controlled-scientific-final-actual-commands-v1','all_passed':False,
                    'completed_commands':records,'pending_commands':[n for n,_ in COMMANDS[len(records):]]},indent=2),encoding='utf-8')
  print('TERMINAL_RELEASE_CHECK',name,result.returncode,item['duration_seconds'],flush=True)
  if result.returncode:
   print('FAILED_COMMAND_LOG',log,flush=True)
   raise SystemExit(result.returncode)
 RESULT.write_text(json.dumps({'schema':'controlled-scientific-final-actual-commands-v1','all_passed':True,
        'completed_commands':records,'pending_commands':[],'finished_utc':utc(),
        'scope':'Software/document review gate only; not a substitute for independent CFD or scientific coauthor approval.'},indent=2),encoding='utf-8')
 print('ALL_SEVEN_FINAL_ACCEPTANCE_COMMANDS_PASS',flush=True)
if __name__=='__main__':main()
