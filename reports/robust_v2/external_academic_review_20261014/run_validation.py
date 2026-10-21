"""Capture the directive's initial non-network validation without rewriting evidence."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / 'validation_baseline'
OUT.mkdir(exist_ok=True)
ENV = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONIOENCODING='utf-8')
COMMANDS = [
    ('diff_check', ['git', 'diff', '--check']),
    ('uv_sync', ['uv', 'sync', '--extra', 'dev', '--extra', 'benchmarks', '--extra', 'docs']),
    ('ruff', ['uv', 'run', 'ruff', 'check', 'src/robust_airfoil', 'tests/robust_v2', 'scripts/robust_v2']),
    ('mypy', ['uv', 'run', 'mypy', 'src/robust_airfoil']),
    ('unit', ['uv', 'run', 'pytest', 'tests/robust_v2', '-m', 'not network and not xfoil and not slow', '-q']),
    ('author_pack', ['uv', 'run', '--extra', 'docs', 'python', 'scripts/robust_v2/verify_final_author_pack.py']),
]
rows = []
for name, command in COMMANDS:
    started = time.monotonic()
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (OUT / f'{name}.log').open('w', encoding='utf-8') as log:
        log.write(json.dumps({'command': command, 'cwd': str(ROOT), 'utc': stamp}) + '\n')
        log.flush()
        result = subprocess.run(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=False)
    row = {'name': name, 'command': command, 'utc': stamp, 'returncode': result.returncode, 'seconds': round(time.monotonic()-started,3)}
    rows.append(row)
    (OUT/'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(row), flush=True)
    if name == 'uv_sync' and result.returncode:
        print('Dependency sync failed; remaining commands still report the available environment honestly.', flush=True)
print('VALIDATION_SEQUENCE_FINISHED', flush=True)
