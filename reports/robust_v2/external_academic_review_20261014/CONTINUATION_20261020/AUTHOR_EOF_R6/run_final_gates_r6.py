"""Post-whitespace exact terminal gates; preserve previous failed evidence."""
import datetime, hashlib, json, os, subprocess, time
from pathlib import Path
root = Path(__file__).resolve().parents[5]
out = Path(__file__).resolve().parent
gates = [
    ("ruff", ["uv", "run", "ruff", "check", "src/robust_airfoil", "tests/robust_v2", "scripts/robust_v2"]),
    ("mypy", ["uv", "run", "mypy", "src/robust_airfoil"]),
    ("prescribed_unit", ["uv", "run", "pytest", "tests/robust_v2", "-m", "not network and not xfoil and not slow", "-q"]),
    ("full_nonnetwork_xfoil", ["uv", "run", "pytest", "-m", "not network", "--run-xfoil", "-q"]),
    ("author_pack", ["uv", "run", "--extra", "docs", "python", "scripts/robust_v2/verify_final_author_pack.py"]),
    ("git_diff_check", ["git", "diff", "--check"]),
]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    receipt = out / "FINAL_R6_GATES.json"
    if receipt.exists():raise FileExistsError("Prior R6 test receipt exists; do not overwrite")
    rows = []
    for name,cmd in gates:
        log = out / ("R6_" + name + ".log")
        if log.exists():raise FileExistsError(str(log))
        t = time.monotonic()
        p = subprocess.run(cmd,cwd=root,capture_output=True,text=True,encoding="utf-8",errors="replace",env=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2"),check=False)
        log.write_text(p.stdout+p.stderr,encoding="utf-8")
        row = dict(gate=name,exit=p.returncode,seconds=time.monotonic()-t,sha256=sha(log),log=str(log.relative_to(root).as_posix()))
        rows.append(row)
        print(name,p.returncode,round(row["seconds"],2),flush=True)
        if p.returncode:break
    receipt.write_text(json.dumps(dict(status="PASS" if len(rows)==len(gates) and all(x["exit"]==0 for x in rows) else "FAILED",completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows),indent=2)+"\n",encoding="utf-8")
    if len(rows)!=len(gates) or any(x["exit"] for x in rows):raise SystemExit(1)
if __name__=="__main__":main()
