"""Add opt-in training resource guard, preserving existing default model semantics.

The five completed controlled-model weights are immutable. This guard only
executes on *new* training invocations that explicitly request resource floors;
the original run did not request this option.
"""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
p=ROOT/'src/robust_airfoil/modelling/train.py'
before=p.read_text(encoding='utf-8')
def change(a,b):
 global before
 assert before.count(a)==1,repr(a)
 before=before.replace(a,b)
old_sha=hashlib.sha256(p.read_bytes()).hexdigest()
change('from robust_airfoil.modelling.samplers import UniformAirfoilPointSampler',
       'from robust_airfoil.modelling.samplers import UniformAirfoilPointSampler\nfrom robust_airfoil.optimisation.checkpointing import assert_campaign_resources')
change('    checkpoint_each_epoch: bool = False,\n    resume: bool = False,',
       '    checkpoint_each_epoch: bool = False,\n    resume: bool = False,\n    enforce_resource_floors: bool = False,')
change('                   layer_norm, resident_batches],',
       '                   layer_norm, resident_batches, enforce_resource_floors],')
change('    for epoch in range(first_epoch, maximum_epochs):\n        if epochs_without_improvement >= patience:',
       '    for epoch in range(first_epoch, maximum_epochs):\n        if enforce_resource_floors:\n            assert_campaign_resources(output_dir)\n        if epochs_without_improvement >= patience:')
p.write_text(before,encoding='utf-8')
record={'schema':'posttraining-opt-in-resource-guard-revision-v1',
        'source_path':p.relative_to(ROOT).as_posix(),
        'training_source_sha256':old_sha,
        'new_source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'original_controlled_five_member_weights_retrained':False,
        'original_checkpoint_receipts_modified':False,
        'enabled_in_original_member_training':False,
        'change_scope':'Adds opt-in guard before each new epoch; existing default False remains unchanged',
        'source_provenance_boundary':'Already produced weights remain bound to original prior source hash. Resume of an existing checkpoint under the revised source must refuse via its context hash.'}
(HERE/'POST_TRAIN_RESOURCE_GUARD_SOURCE_CHANGE.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
