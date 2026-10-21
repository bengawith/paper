"""Independent final static, typing and complete nonnetwork acceptance receipt.

Never duplicate a currently running gate; refuse to overwrite any prior log.
The scientific implementation and all study thresholds remain untouched.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
OUT=HERE/"FINAL_CANDIDATE_POST_ACCEPTANCE_STATUS.json"
STAGES=[
 ("ruff",["uv","run","ruff","check","src/robust_airfoil","tests/robust_v2","scripts/robust_v2"]),
 ("mypy",["uv","run","mypy","src/robust_airfoil"]),
 ("pytest",[sys.executable,"-m","pytest","-m","not network","--run-xfoil","-q"]),
]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def main():
 if OUT.exists():raise FileExistsError("Previous post-acceptance gate needs review")
 for name,_ in STAGES:
  if (HERE/f"FINAL_CANDIDATE_POST_ACCEPTANCE_{name.upper()}.log").exists():
   raise FileExistsError("Existing gate log requires inspection: "+name)
 env=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",PYTHONIOENCODING="utf-8")
 records=[]
 for name,cmd in STAGES:
  dest=HERE/f"FINAL_CANDIDATE_POST_ACCEPTANCE_{name.upper()}.log"
  began=time.monotonic()
  print("START_FINAL_GATE",name,utc(),flush=True)
  with dest.open("w",encoding="utf-8") as output:
   output.write("START "+utc()+"\nCOMMAND "+repr(cmd)+"\n");output.flush()
   proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,check=False)
  records.append(dict(gate=name,exit=proc.returncode,elapsed_seconds=time.monotonic()-began,
    completed_utc=utc(),log=dest.relative_to(ROOT).as_posix(),log_sha256=sha(dest)))
  print("END_FINAL_GATE",name,"EXIT",proc.returncode,"SHA",sha(dest),flush=True)
  if proc.returncode:break
 state=dict(status="PASS" if len(records)==3 and all(r["exit"]==0 for r in records) else "FAILED",
  completed_utc=utc(),head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
  branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
  records=records)
 OUT.write_text(json.dumps(state,indent=2),encoding="utf-8")
 print(json.dumps(state,indent=2),flush=True)
 if state["status"]!="PASS":raise SystemExit(1)
if __name__=="__main__":main()
