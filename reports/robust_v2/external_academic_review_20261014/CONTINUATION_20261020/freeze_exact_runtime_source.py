"""Preserve exact executed runtime bytes independent of Git autocrlf checkout.

Production protocol source hashes bind working-file bytes. A later Git commit
may normalise line endings and therefore cannot alone reproduce those digests.
This archive retains exact bytes and source hashes without changing live code.
"""
from __future__ import annotations
import datetime
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
PROTOCOL=ROOT/'results/robust_v2/lineages/robust-v2-controlled-20261020/optimisation_production/PRODUCTION_PROTOCOL.json'
MANIFEST=HERE/'EXACT_RUNTIME_SOURCE_MANIFEST.json'
ZIP=HERE/'EXACT_RUNTIME_SOURCE_BYTES.zip'
EXTRA=['scripts/robust_v2/run_repaired_production.py',
       'scripts/robust_v2/run_controlled_nominal.py',
       'scripts/robust_v2/train_repaired_lineage.py',
       'scripts/robust_v2/evaluate_repaired_lineage.py',
       'scripts/robust_v2/freeze_confirmation_predictor.py',
       'scripts/robust_v2/run_confirmation_solver.py',
       'scripts/robust_v2/analyse_controlled_confirmation.py',
       'scripts/robust_v2/audit_repaired_risk.py',
       'scripts/robust_v2/complete_controlled_selected_study.py',
       'configs/robust_v2/optimisation.yaml',
       'configs/robust_v2/uncertainty.yaml',
       'configs/robust_v2/controlled_study.yaml']
def sha(b):return hashlib.sha256(b).hexdigest()
def old_train(current):
 newline='\r\n' if bytes([13,10]) in current else '\n'
 text=current.decode('utf-8').replace('\r\n','\n')
 old=text
 remove='from robust_airfoil.optimisation.checkpointing import assert_campaign_resources\n'
 assert old.count(remove)==1
 old=old.replace(remove,'')
 sig='    enforce_resource_floors: bool = False,\n'
 assert old.count(sig)==1
 old=old.replace(sig,'')
 clause='                   layer_norm, resident_batches, enforce_resource_floors],'
 assert old.count(clause)==1
 old=old.replace(clause,'                   layer_norm, resident_batches],')
 gate='        if enforce_resource_floors:\n            assert_campaign_resources(output_dir)\n'
 assert old.count(gate)==1
 old=old.replace(gate,'')
 return old.replace('\n',newline).encode('utf-8')
def main():
 if ZIP.exists() or MANIFEST.exists():
  raise FileExistsError('The executed source snapshot already exists')
 protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
 recorded=protocol['context']['source_sha256']
 snapshots={}
 for relative,expected in sorted(recorded.items()):
  path=ROOT/'src/robust_airfoil'/relative
  raw=path.read_bytes()
  if sha(raw)!=expected:raise ValueError('Scientific source changed after production context freeze: '+relative)
  snapshots['src/robust_airfoil/'+relative]=raw
 for relative in EXTRA:
  path=ROOT/relative
  snapshots[relative]=path.read_bytes()
 historical=json.loads((HERE/'POST_TRAIN_RESOURCE_GUARD_SOURCE_CHANGE.json').read_text(encoding='utf-8'))
 trained=old_train(snapshots['src/robust_airfoil/modelling/train.py'])
 if sha(trained)!=historical['training_source_sha256']:
  raise ValueError('Cannot recreate original controlled-model training source from the guarded source revision')
 snapshots['historical_controlled_training_source/robust_airfoil/modelling/train.py']=trained
 compressed=io.BytesIO()
 with zipfile.ZipFile(compressed,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
  for relative,raw in sorted(snapshots.items()):
   item=zipfile.ZipInfo(relative,date_time=(2026,10,21,0,0,0))
   item.compress_type=zipfile.ZIP_DEFLATED
   item.external_attr=(0o644<<16)
   archive.writestr(item,raw,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
 ZIP.write_bytes(compressed.getvalue())
 with zipfile.ZipFile(ZIP) as archive:
  if archive.testzip() is not None:raise IOError('Exact-runtime source archive CRC mismatch')
 output={'schema':'exact-runtime-code-bytes-reconstruction-v1',
         'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'production_protocol_context_hash':protocol['context_hash'],
         'recorded_source_git_commit':protocol['source_git_commit'],
         'context_lib_files_hash_verified':len(recorded),
         'pre_resource_guard_training_source_sha256':sha(trained),
         'note':'Source files in the running study are not modified. This byte-identical ZIP allows reproduction even if git autocrlf changes working checkout line endings.',
         'archive_sha256':sha(ZIP.read_bytes()),
         'archived_files':[{ 'path':p,'sha256':sha(b),'bytes':len(b)} for p,b in sorted(snapshots.items())]}
 MANIFEST.write_text(json.dumps(output,indent=2),encoding='utf-8')
 print('EXACT_RUNTIME_SNAPSHOT_FROZEN',len(snapshots),'files,',len(ZIP.read_bytes()),'bytes',flush=True)
if __name__=='__main__':main()
