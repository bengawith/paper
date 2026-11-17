"""Read-only interim explanation of strict selected XFOIL failures.

This is an incomplete-in-progress snapshot, not a final scientific report.
"""
from pathlib import Path
from collections import Counter,defaultdict
import json
ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'results/robust_v2/lineages/robust-v2-controlled-20261020/selected_study/xfoil'
paths=list(BASE.rglob('CASE_RECEIPT.json'))
groups=defaultdict(Counter)
missing=Counter()
for path in paths:
 receipt=json.loads(path.read_text(encoding='utf-8'))
 assessment=receipt['assessment']
 parts=path.relative_to(BASE).parts
 key=(parts[0],parts[1])
 groups[key]['attempted']+=1
 groups[key]['usable' if assessment['usable'] else 'unusable']+=1
 if not assessment['usable']:
  groups[key]['reason_'+str(assessment.get('failure_reason'))]+=1
  if assessment.get('missing_alpha_deg'):
   missing[len(assessment['missing_alpha_deg'])]+=1
for key in sorted(groups):
 print(key,dict(groups[key]))
print('complete_receipts_so_far',len(paths))
print('missing_alpha_point_count_distribution',dict(missing))
