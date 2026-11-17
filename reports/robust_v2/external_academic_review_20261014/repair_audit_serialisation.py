"""Repair review-output null encoding and an incorrectly indexed test fixture."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
p = ROOT/'reports/robust_v2/external_academic_review_20261014/audit_geometry_corpus.py'
s = p.read_text(encoding='utf-8')
s = s.replace("changed.nlargest(10,'maximum_parameter_change').to_dict('records')", "json.loads(changed.nlargest(10,'maximum_parameter_change').to_json(orient='records'))")
s = s.replace("valid.loc[valid.airfoil_id=='naca2412'].to_dict('records')", "json.loads(valid.loc[valid.airfoil_id=='naca2412'].to_json(orient='records'))")
p.write_text(s,encoding='utf-8')
p = ROOT/'tests/robust_v2/test_geometry_coordinate_order.py'
s = p.read_text(encoding='utf-8')
assert 'points[[3,14]] = points[[14,3]]' in s
s = s.replace('points[[3,14]] = points[[14,3]]','points[[3,17]] = points[[17,3]]')
p.write_text(s,encoding='utf-8')
(ROOT/'reports/robust_v2/external_academic_review_20261014/TEST_FIXTURE_CORRECTION.md').write_text('# Coordinate-order fixture correction\n\nThe first unordered-input fixture swapped entries 3 and 14, which both had x=0.05. That changed surface y membership but not coordinate order, so an ordering-only guard could not be required to reject it. Corrected the fixture to indices 3 and 17 (different x positions); the assertion and fail-closed requirement are unchanged. The separate Selig/Lednicer equivalence and endpoint tests already demonstrate the real source defect. Also repaired JSON serialization of absent error-column values as null rather than nonstandard NaN; the full corpus CSV and numerical outputs had already been produced. No frozen scientific evidence changed.\n',encoding='utf-8')
print('Audit serialization and unequal-x fixture corrected')
