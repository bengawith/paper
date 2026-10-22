"""Correct audited figure labels/layout without changing historical numerical data."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
backup=OUT/'BEFORE_REVIEW/figures'
if not backup.exists():
    shutil.copytree(ROOT/'paper/figures',backup)
rows=[]
def edit(path, replacements):
    p=ROOT/path; before=p.read_text(encoding='utf-8'); text=before
    for old,new in replacements:
        if old not in text:raise ValueError((path,old))
        text=text.replace(old,new)
    p.write_text(text,encoding='utf-8')
    rows.append({'path':path,'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(text.encode()).hexdigest()})

edit('scripts/robust_v2/make_paper_figures.py',[
('axes[1].legend(markerscale=4, loc="upper left")','axes[1].legend(markerscale=4, loc="lower right")'),
('Surrogate accuracy on held-out (grouped) locked-test aerofoils','Historical encoded-feature test predictions (geometry lineage requires repair)'),
('Surrogate objective including trust-exit penalties','Historical surrogate objective including trust-exit penalties'),
('Acceptance of three prescribed perturbation generators (1% chord, 2048 draws)','Historical geometry/refit admission (1% chord, 2048 draws)'),
('"Candidate-specific corrected NSGA-II: low-budget diagnostic\\\\n"','"Historical schema-4 smoke front\\n"'),
('f"10 seeds; 32 population × 10 generations × 16 scenarios; "','f"10 seeds; 32 × 10 × 16; post-hoc scaled "'),
('"NOT a production Pareto optimum"','"Historical diagnostic: not current acceptance"'),
('"Within-trust\\nrobust design"','"Historically selected\\ncandidate"'),
('"Outside-trust\\nsurrogate optimum"','"Outside-screen\\ncomparator"'),
('Selected-design XFOIL numerical check (13–14 valid scenarios)','Fixed-coordinate XFOIL check (13–14 completed scenarios)'),
('"Largest observed surrogate–XFOIL divergence\\n"','"Largest mean gap in the eligible converged subset\\n"'),
('reports/robust_v2/submission_review_20261014/ML_DIAGNOSTICS/fig_ml_selective_coverage.png','reports/robust_v2/external_academic_review_20261014/ML_SELECTIVE_CURRENT/fig_ml_selective_coverage.png'),
])
edit('scripts/robust_v2/audit_ml_selective_20261014.py',[
('reports/robust_v2/submission_review_20261014/ML_DIAGNOSTICS','reports/robust_v2/external_academic_review_20261014/ML_SELECTIVE_CURRENT'),
('xlabel="States retained by calibration-scaled risk score (%)"','xlabel="Retained states (%)"'),
('title=f"{truth} | Spearman={values[\'error_risk_spearman\']:.2f}"','title=f"({\'abc\'[list(results).index(truth)]}) {truth} | Spearman={values[\'error_risk_spearman\']:.2f}"'),
('Descriptive selective prediction | frozen ensemble, no test-set fitting','Historical selective prediction | retrospective, no test-set fitting'),
])
(OUT/'FIGURE_EDIT_LOG.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print('FIGURE_SOURCE_PRESENTATION_REPAIRED')
