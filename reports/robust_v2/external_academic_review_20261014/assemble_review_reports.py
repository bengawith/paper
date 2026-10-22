"""Assemble the in-place author review from the separately reproduced evidence.

This writes review documents, not aerodynamic labels or frozen model evidence.
Final validation and the post-push OneDrive release manifest are separate gates.
"""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OLD = ROOT/'reports/robust_v2/submission_review_20261014'
PAPER = ROOT/'paper'
LIN = ROOT/'results/robust_v2/lineages/robust-v2-current-20260908'
BASE = '156f07f77e92f77df34a6cc6f6a1726cc0e1638a'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def write(name, text):
    (OUT/name).write_text(text.strip()+'\n', encoding='utf-8')

backup = OUT/'BEFORE_REVIEW'
backup.mkdir(exist_ok=True)
for rel in ['manuscript.md','LATEST_MANUSCRIPT_FOR_AVA.docx','response_to_reviewers.md','README.md']:
    p = PAPER/rel
    target = backup/rel
    if p.is_file() and not target.exists():
        shutil.copy2(p,target)
old_text = (backup/'manuscript.md').read_text(encoding='utf-8')
refs = re.findall(r'(?m)^\[(\d+)\] (.+)$', old_text)
assert len(refs)==28
entries = {int(n):v for n,v in refs}
entries[11] = 'Selig, M. S. (n.d.). UIUC Airfoil Data Site. University of Illinois Urbana–Champaign. Accessed 14 October 2026. https://m-selig.ae.illinois.edu/ads/coord_database.html'
entries[12] = 'Airfoil Tools (n.d.). Airfoil Tools. Historical source of archived XFOIL polars; current site access was not verified during this review. http://airfoiltools.com/'
entries[16] = re.sub(r'https?://\S+', 'https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html', entries[16])
entries[20] = re.sub(r'https?://\S+', 'https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html', entries[20])
entries[25] += ' https://store.doverpublications.com/products/9780486605869'
entries[29] = 'Sharpe, P. & Hansman, R. J. (2025). NeuralFoil: An Airfoil Aerodynamics Analysis Tool Using Physics-Informed Machine Learning. arXiv:2503.16323v1. https://arxiv.org/abs/2503.16323'
body = (OUT/'MANUSCRIPT_REVISED_BODY.md').read_text(encoding='utf-8').strip()
(PAPER/'manuscript.md').write_text(body+'\n\n## References\n\n'+'\n\n'.join(f'[{n}] {entries[n]}' for n in sorted(entries))+'\n', encoding='utf-8')

primary_urls = {
11:'https://m-selig.ae.illinois.edu/ads/coord_database.html',12:'http://airfoiltools.com/',
16:'https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html',
17:'https://arxiv.org/abs/2107.07511',18:'https://arxiv.org/abs/1711.05101',
19:'https://icml.cc/Conferences/2010/abstracts.html',
20:'https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html',
24:'https://jmlr.org/papers/v12/pedregosa11a.html',
25:'https://store.doverpublications.com/products/9780486605869',
26:'https://www.mheducation.com/highered/product/fundamentals-of-aerodynamics-anderson.html',
29:'https://arxiv.org/html/2503.16323v1'}
support = {
1:'Viscous/inviscid aerodynamic-method background; not independent CFD evidence for this study.',
2:'ML/CFD context; not proof of this surrogate accuracy or speedup.',
3:'Fluid-mechanics ML background; not a novelty certificate.',
4:'Established data-based airfoil analysis/optimisation; excludes a first-ever surrogate-design claim.',
5:'Aerodynamic shape-optimisation literature context; model comparisons here still require common evaluation.',
6:'Prior surrogate-assisted evolutionary aerodynamic optimisation; not validation of the current geometry.',
7:'Prior evolutionary CFD shape optimisation; not current CFD results.',
8:'CST representation foundation; does not certify the implementation or fitting tolerance.',
9:'Parametric geometry representation foundation; exact metadata retains its historical conference identity.',
10:'XFOIL numerical-method provenance; same-solver repetition is not independent physical validation.',
11:'Coordinate database and format description; accessed page, no unsupported 2010 publication date.',
12:'Archived label-source attribution only; present availability and redistribution terms unverified.',
13:'NSGA-II algorithm attribution; not convergence or optimality of this run.',
14:'pymoo software attribution; not a pass for the study acceptance gate.',
15:'CVaR definition; finite sample and penalised-objective caveats remain.',
16:'Ensemble-disagreement motivation; raw standard deviation is not a coverage guarantee.',
17:'Conformal-prediction framework and assumptions; no conditional/adaptive-design guarantee here.',
18:'AdamW attribution; exact optimiser implementation is checked in repository code.',
19:'ReLU origin; not a claim of global smoothness.',
20:'PyTorch attribution; not evidence of this application performance.',
21:'Optuna attribution; does not itself rule out data leakage.',
22:'NumPy software attribution.',23:'SciPy software attribution.',24:'scikit-learn software attribution.',
25:'General wing-section background only; not used to support any numerical result.',
26:'General aerodynamics background only; sixth-edition/year confirmation still requires the authors copy.',
27:'Existing surrogate-assisted robust aerofoil optimisation; current audit accessed metadata/abstract, not all methods.',
28:'Feedback covariate-shift caution; ordinary marginal coverage is insufficient for adaptively selected designs.',
29:'Existing confidence-aware ML aerofoil analysis and design; the present integration is not first-of-kind.'}
ledger = json.loads((OUT/'REFERENCES_PRIMARY/CROSSREF_LEDGER.json').read_text(encoding='utf-8'))
by_n = {int(r['number']):r for r in ledger}
reference_rows=[]
for n in range(1,30):
    r=by_n.get(n,{})
    doi=r.get('doi','') or ''
    if doi:
        status='PRIMARY_CROSSREF_METADATA_VERIFIED'
        access='Crossref bibliographic metadata; no claim that full text was reviewed'
        url=r.get('primary_url','https://doi.org/'+doi)
        h=r.get('metadata_sha256','')
    else:
        url=primary_urls[n]; h=''; status='PRIMARY_PAGE_IDENTITY_CHECKED'
        access='Official author/publisher/conference page or abstract; not full text'
    if n==12:
        status='ARCHIVED_ATTRIBUTION_CURRENT_ACCESS_UNVERIFIED'; access='Live retrieval failed; archived local provenance retained'
    elif n==25:
        status='PRIMARY_TITLE_AUTHOR_CHECKED_EDITION_YEAR_UNCONFIRMED'; access='Dover title/author page; original 1959 edition not independently confirmed from the accessible page'
    elif n==26:
        status='EXACT_CITED_EDITION_UNVERIFIED'; access='Current publisher page is seventh edition, not evidence for the cited sixth edition (2017)'
    elif n==27:
        access='Crossref metadata plus publisher abstract; detailed full-text comparison not claimed'
    elif n==28:
        access='Primary article text and metadata; feedback-shift assumptions examined'
    elif n==29:
        access='Author arXiv metadata and HTML methodology; preprint status retained'
    occurrence = re.findall(r'\[([^\]]+)\]', body)
    active=False
    for x in occurrence:
        if any(str(n)==z.strip() for z in x.split(',')):
            active=True
        for a,b in re.findall(r'(\d+)[–-](\d+)',x):
            if int(a)<=n<=int(b):active=True
    reference_rows.append({'number':n,'citation':entries[n],'doi':doi,'primary_url':url,'checked_date':'2026-10-14','verification_status':status,'access_scope':access,'support_assessment':support[n],'cited_in_body':active,'change':'Added directly relevant NeuralFoil literature' if n==29 else 'Removed unsupported web publication year' if n in (11,12) else 'Corrected primary link' if n in (16,20,25) else 'Retained; scope and any unresolved bibliographic fields explicitly recorded','metadata_sha256':h})
with (OUT/'REFERENCES_VERIFIED.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(reference_rows[0]));w.writeheader();w.writerows(reference_rows)

reports={
'ACADEMIC_REFEREE_REPORT.md':r'''
# Independent academic review — Robust V2
14 October 2026 | Reviewed base: `156f07f77e92f77df34a6cc6f6a1726cc0e1638a`

## Recommendation

**Major scientific revisions before journal submission. The review and source-repair deliverable are distinct from a submission-ready research result.** This was a separate source/evidence audit followed by corrections and a final self-review; it is not an external human referee endorsement or completed coauthor approval.

The starting manuscript was materially too optimistic about the geometric integrity of its ML pipeline. A reproducible coordinate-order defect changes 1,001 of 1,560 stored aerofoil feature vectors. That finding has consequences for training inputs, physical grouping, support distance, reconstruction and design optimisation. It cannot be resolved by polishing claims or merely rerunning the old full optimisation. A new accepted geometry corpus and model lineage are required.

The positive results are not discarded. Stored held-out predictions reproduce, calibration-only ensemble disagreement retains incremental residual-ranking information after correcting a near-duplicate fold overlap, and direct recomputation of all 825 stored selected-design solver cases preserves the nominal and completed-case XFOIL advantages. Those support limited historical-feature and fixed-coordinate statements, not a repaired-model robust-design claim.

## Major findings and disposition

**F01 — Coordinate preprocessing, critical.** Two-block leading-edge-to-trailing-edge input was mishandled by a median-ordinate fallback. The parser now recognises supported coordinate orders and refuses ambiguous cases. Source correction is complete; regeneration of training data, clusters, model and screening is not. The geometry audit lists every affected record, three ambiguous rejections and all reconstruction diagnostics. No frozen evidence was rewritten.

**F02 — Geometric admission, critical.** Thickness cannot be checked by subtracting ordinates at different x positions. Source checks now compare interpolated surfaces on shared knots, reject malformed/nonfinite inputs and validate reconstructed CST after refitting, both in dataset admission and candidate uncertainty. This is not cosmetic: the historically selected knee now has seven invalid refits among 64 draws. Its current penalised mean is 0.009585, worse than the reference's 0.009421, while its tail objective remains lower. Both outcomes are reported.

**F03 — Calibration grouping, important.** The original incremental audit grouped 234 aerofoil names although they represent 233 stored geometry clusters. One near-duplicate cluster crossed evaluation folds. Canonical analysis now uses the geometry-cluster mapping with fail-closed unknown/non-calibration checks. AUROC changes from 0.767962 for support alone to 0.866737 for support plus spreads; the fixed-OOF cluster-bootstrap difference interval is [0.066679, 0.140202]. This is still retrospective and still uses the historical feature lineage.

**F04 — Optimisation acceptance, important.** The original reference-increment operator is unsuitable for candidate-specific uncertainty. The later smoke front is reproducible but uses post-hoc pooled objective ranges. Current code prevents those diagnostics from self-issuing prospective acceptance. Neither the original 0.283 front-distance result nor the later 0.1493 smoke establishes a current full-budget Pareto result. Actual model/scaler hashes, source, dependency and evaluation context now protect caches; mismatched historical files are refused, not overwritten.

**F05 — Evidence interpretation, important.** Accepted lift interval coverage rounds to 97.7%, not 97.8%. The frozen screen's extra spread cutoffs reject no states independently of support. ReLU is piecewise differentiable, and the model has an input affine projection plus four residual affine transforms, not just two hidden layers. The 1%-chord sample maximum remains adverse. The selected converged adversarial plot is not the overall largest-gap record. These corrections are integrated into text and figures, not left as a detached addendum.

**F06 — Reproducibility and handoff, important.** Historical author-pack acceptance checks hardcoded superseded claims. The current verifier must validate the revised evidence and explicit research blockers, not certify journal readiness. The offline copy must refuse dirty or mismatched repositories and must never force-reset a branch. The refreshed OneDrive package is bound to the actual pushed commit with file hashes and archive checks. Software and packaging success are reported separately from scientific acceptance.

## What the evidence currently supports

The archived predictor reproduces its 26,692-state metrics exactly in the available environment. There is no detected overlap among the recorded development, calibration and test cluster identifiers, and checked scaling/tuning uses development data; this does not rehabilitate the physical-geometry representation. Disagreement contains additional ranking information within the historical calibration distribution. One fixed selected geometry has 18.7% lower nominal local-XFOIL weighted drag. Two independent 0.5%-chord scrambles show lower completed-case paired mean drag by 15.4% and 15.8% across 92 and 96 pairs. Every stored selected-design coordinate record matches its reconstructed scenario, and all 44 supplied CFD-input file hashes match their manifests.

## Why this is not yet submission-ready

A repaired geometry/model lineage and fresh evaluation are absent. Full corrected optimisation with a prospective normalisation protocol is absent. Independent CFD and measured manufacturing tolerances are absent. Solver failures prevent unconditional manufacturing-tail inference. Exact editions of two general background books and the current Airfoil Tools access/terms remain incompletely verified. Coauthors must approve the revised title, scientific scope, contribution statement, disclosures and destination journal.

The appropriate next scientific choice is not another architecture search. It is a controlled rebuild of the geometry-to-polar map, followed by prospective selection and a matched-constraint solver comparison. Alternatively, the authors may deliberately choose a narrower numerical case-study or reproducibility contribution, but should not present that as completion of the original robust-optimality claim.

## Evidence and review coverage

The companion claim matrix gives source files, JSON/CSV selectors, hashes, reproduction methods and disposition. Separate reports cover all bibliography entries, ML provenance and leakage, calibration, uncertainty/optimisation, raw XFOIL and CFD readiness, all figures/tables, manuscript style, software validation and owner decisions. `FINAL_ACCEPTANCE.json` records deliverable versus scientific status. There is no concealed conversion of an open research gate into a pass.
''',
'ML_METHODS_AND_LEAKAGE_AUDIT.md':r'''
# ML methods, data integrity and leakage audit

## Scope and evidence

Inputs: immutable `robust-v2-current-20260908` model, split, tuning, calibration and prediction records; `data/robust_v2/processed/full_exact`; active preprocessing, training, ensemble and evaluation source. Reproduction: `recompute_ml.py`, `audit_geometry_corpus.py`; outputs `ML_RECOMPUTATION/ML_AUDIT.json`, metrics CSV and `GEOMETRY_REPAIR/GEOMETRY_AUDIT.json` plus per-aerofoil records. All relevant source hashes are linked by the claim/evidence matrix and evidence manifest.

## Verified executable properties

The stored 179,780-state table uses numerical source rows rather than padded polar targets. Lift has one unavailable label; drag and moment have all rows. The grouped split contains development 1,093 aerofoils/1,088 clusters; calibration 234/233; historical test 233/233. Recorded cluster identifiers do not overlap. The audited scaling and tuning inputs do not include non-development identities. Recorded cluster-bootstrap membership and member scalers reproduce from their saved inputs. Re-inference matches the saved test predictions exactly in the available CUDA environment.

The frozen architecture is an input projection of 13 to 512 features, two residual blocks containing two affine layers each, and three scalar heads: five hidden affine transformations and 1,064,451 parameters per member. ReLU makes it piecewise differentiable, not globally smooth. Five members, grouped bootstrap, dropout 0.12, layer normalisation, masked log-cosh loss and AdamW are reflected in the archived configuration and source. A raw member spread is an empirical diagnostic, not independently calibrated epistemic uncertainty.

## Critical integrity defect

The coordinate-order audit changes 1,001 of 1,560 stored feature vectors. It compares 1,557 parseable records, rejecting three ambiguous files. Historical versus repaired maximum-fit errors above 0.005 chord are 931 versus 289 within those comparable records; the all-1,560 historical count is 933, a different denominator. Repaired reconstructed fits also expose 656 crossing/admission failures. The active parser now treats Lednicer blocks and Selig contours separately, and training-data admission validates the reconstructed fit.

This means an apparent absence of identifier overlap does not prove physical near-duplicate separation. Clusters and support must be recomputed after the accepted geometry policy is frozen. Corrected feature vectors cannot be evaluated using the old weights and relabelled as new test evidence. A rebuild must preserve a newly declared untouched confirmation set; the historical test has already been inspected repeatedly and is not pristine for future modelling choices.

## Metric reconciliation

Historical all-state R²: CL 0.98869049; log-CD 0.92184771; CM 0.91801080. MAE: 0.05713828, 0.14340075, 0.00959476. Aerofoil-macro MAE: 0.05602204, 0.14839180, 0.00954105. Linear-drag MAE is 0.00454658 and R² 0.85282766; these must not be confused with the log-drag values. Accepted/rejected metrics, masks and polar-wise coverage are retained separately. Dataset-family claims are not upgraded to leave-one-family-out: no authoritative family taxonomy was established.

The historical gradient-boosting/ridge/mean comparison is retained as an archived grouped-development comparison, not a new training run. A further architecture zoo is not justified by this audit. Repeat the same modest baselines after geometry repair; comparisons must use equal feature acceptance, split, masks and cost accounting.

## Runtime and integrity

The archived predictor-only GTX 1660 measurements distinguish about 19 ms/single state, 22.5 ms/48-state polar and 55.5 ms/6,144-state batch. CUDA synchronization brackets timing; batch amortisation is not per-query latency or full-workflow speedup. The fresh five-candidate, 64-scenario evaluation took about 3.6 s and is explicitly a different workload. Checkpoint and scaling hashes are now checked before deserialisation, preventing altered weights from silently satisfying a manifest-only check.

## Acceptance boundary

Code repair and historical-number reproduction are complete review outcomes. Rebuilt training, reclustering, retraining, fresh confirmation and physical-geometry generalisation are not complete. No historical split or coefficient is silently replaced.
''',
'TRUST_CALIBRATION_AUDIT.md':r'''
# Trust calibration and selective prediction

## Frozen operational rule

The 26,692-state historical test has 17,642 accepted and 9,050 rejected states. All 9,050 rejected states fail support distance. The stored spread cutoffs add zero independent rejections; six moment-spread failures and the recorded angle failures already fail support. Thus the empirical accepted-set benefit is attributed to a support-dominated rule, not to proven independent operational value of ensemble spread.

The bands are target-wise, residual-quantile constructions. State observations within a polar are correlated. All-state coverage is 0.936685 for lift, 0.943916 for log-drag and 0.944890 for moment; accepted coverage is 0.977497, 0.965310 and 0.991781. Accepted lift coverage rounds to 97.7% at one decimal place. Complete observed-polar coverage is only about 51.1%, 44.2% and 66.5%; the nominal state-wise band is not a simultaneous curve band. Conditional-on-acceptance or optimiser-shift guarantees are not inherited from ordinary marginal conformal arguments.

## Incremental disagreement study

The old audit grouped aerofoil names and allowed one near-duplicate geometry cluster to cross two folds. The canonical script now resolves every calibration row against the frozen geometry-cluster map and refuses missing/non-calibration identities. Corrected GroupKFold(5) contains no shared stored geometry clusters. Residual normalisation, high-error thresholds and input scaling are training-fold-only; ridge alpha remains fixed at 10.

The rerun `CANONICAL_TRUST_INCREMENTAL/TRUST_INCREMENTAL_CV.json` reproduces the independent audit exactly: support AUROC 0.7679624719; spreads 0.8585365516; combined 0.8667371583; difference 0.0987746864. Combined average precision is 0.4848707689 versus support 0.3609338589. A 300-geometry-cluster bootstrap gives [0.0666791433, 0.1402019524] for the difference. Resampling holds the OOF predictions fixed and does not include the whole refitting procedure's variability. The cluster-balanced and per-fold diagnostics are in `TRUST_RECOMPUTATION`.

This is positive residual-ranking information in the historical calibration distribution. It is not a deployed policy, an untouched confirmation result, a model retraining improvement or proof of reliability on adaptive geometries. The geometry-encoding defect also limits the physical meaning of the source clusters. The frozen trust JSON and model weights remain unchanged.

## Decision

Retain the historical accepted-set metrics as descriptive; replace obsolete ID-grouped incremental numbers in the manuscript; require repaired geometric support and a prospective confirmation cohort before changing any operational gate. Do not tune on the historical test or call retrospective selective-coverage curves pre-registered without dated evidence.

Primary theory: Angelopoulos and Bates, https://arxiv.org/abs/2107.07511; Fannjiang et al., https://doi.org/10.1073/pnas.2204569119. These sources motivate the assumptions and cautions, not acceptance of this implementation.
''',
'UNCERTAINTY_AND_OPTIMISATION_AUDIT.md':r'''
# Uncertainty, objectives and optimisation audit

## Physical operator and representation

The active operator uses Gaussian-correlated controls, piecewise-linear spatial interpolation, per-realisation/per-surface maximum normalisation and leading/trailing-edge tapering. It is bounded and continuous, not a cubic B-spline manufacturing process. The separate spline module is not evidence that the active historical campaign used splines. Amplitudes are prescribed chord fractions, not empirical tolerance distributions. Normalisation also changes the interpretation of latent amplitude/correlation parameters.

The original full campaign transported reference-fitted CST increments to candidate shapes. The current code instead applies shared latent draws to each candidate's normals, validates displaced coordinates, resamples, refits and validates the reconstructed fit. Invalid rows retain nominal parameters only as a matrix-shape placeholder and are explicitly penalised; they are not observed nominal outcomes. Tests reproduce and close crossing-geometry, mismatched-x and invalid-refit defects.

## Objective and feasibility

Required CL values are 0.4, 0.7 and 1.0, with weights 0.25/0.50/0.25. All must have an admissible unique root on the initial rising polar branch. Moment, thickness/geometry, angle and trust checks are separate from the drag value. Mean and empirical CVaR95 apply to a penalised objective; do not label them pure aerodynamic drag when trust or refit penalties occur. The implemented empirical upper-tail mass is fractional when n×0.05 is not integral; at n=13 it reduces to the sample maximum.

## What changed in the selected-design diagnostic

The final-source rerun on five historical designs and 64 latent draws gives:

| Fixed design | Mean penalty-inclusive objective | CVaR95 objective | Trust exits | Invalid refits |
|---|---:|---:|---:|---:|
| Reference | 0.00942056 | 0.01918442 | 8 | 0 |
| Historical minimum-tail design | 0.00893805 | 0.01548513 | 14 | 0 |
| Historical selected knee | 0.00958516 | 0.01515743 | 15 | 7 |

The deterministic and simple sensitivity-baseline designs are infeasible. The knee's mean is worse than the reference's under the current audit, superseding the earlier pre-refit-screen mean of about 0.009040. Its lower tail objective does not conceal this result. Evidence: `COMPUTE_PROFILE/CURRENT_SELECTED_DESIGNS.json`. These values use the historical trained model and are not a repaired-model production result.

## Fronts, seeds and acceptance

The original 128-population, 200-generation, 64-scenario, ten-seed front is invalid for candidate-specific surface uncertainty and has a reported distance 0.283 above 0.20. Ten later schema-4 smoke files reproduce 44 feasible returned entries and distance 0.1492924385 for 32×10×16. Their objective scales are pooled observed ranges, chosen after the outcomes. The active caller now records a descriptive threshold comparison but refuses scientific acceptance under such post-hoc normalisation. No threshold is relaxed and no new favourable scale is selected.

Current caches bind actual checkpoint/scaler hashes, complete scientific source hashes, dependency/Python/device identity, geometry, latent draws, calibration, objective configuration and seed. A mismatch raises a request for a fresh output revision, preserving historical evidence. Historical feasibility/front plots do not become current merely because code is fixed.

## Production disposition

Do not run or approve the full study using a knowingly compromised geometric model lineage. First freeze the geometry-admission and split policy, rebuild and retrain, then freeze objective scales, evaluation draws, candidate-selection protocol and acceptance conditions before the production results are observed. The full 128×200×64×10 experiment is still open. The fresh approximately 3.6-second fixed-design profile is evidence of one workload, not a completion estimate or production-speed claim.
''',
'XFOIL_AND_CFD_READINESS.md':r'''
# XFOIL integrity and CFD readiness

## Independent raw-artifact recomputation

`audit_raw_xfoil_and_fronts.py` reads 825 stored cases, including nominal cases, across the 1%-chord 16-draw campaign and two 0.5%-chord 128-draw campaigns. Every coordinate array matches the reconstruction from its fixed candidate and latent draw within 1e-11. All 825 pass the current direct-coordinate validity checks. Solver usability is separate: clean exit, no timeout, actual twelve-angle coverage, finite positive drag, operating commands and required-lift roots are checked from the raw files. Usable totals including nominal cases are 43, 336 and 344; failures are retained, not imputed.

Nominal weighted drag reproduces as reference 0.0074464020141, selected candidate 0.0060511314705, outside-screen comparator 0.0139600173795. This is −18.7% and +87.5% versus reference for the two selected comparators. Selection is deliberate; it is not a population screen-quality estimate.

At 1% chord, each reference and selected design has 13 usable perturbations but only ten matched pairs. Marginal means differ by about 8.8%; paired means differ by 12.2%. These are different subsets/estimands. The candidate's marginal maximum is worse than reference, about 0.00989 versus 0.00909. An apparent favourable matched-subset maximum must not replace that adverse observation. Small-sample CVaR is just an extreme-sample diagnostic here.

At 0.5% chord, campaign A has reference/candidate usable counts 105/110 and 92 matched pairs; B has 115/107 and 96 pairs. Mean paired drag reductions reproduce as 15.3996% and 15.7803%. Matched empirical CVaR changes are −12.4043% and −11.6701%. All matched pairs are geometrically admissible. Because 36 and 32 pairs fail joint usability, these are conditional completed-case distributions, not unconditional manufacturing-risk estimates.

Evidence: `RAW_XFOIL_RECOMPUTATION/all_825_cases.csv`, `XFOIL_RECALCULATED.json` and all raw-source hashes. This audit recalculates existing polars; it does not claim 825 new solver executions.

## Supplied CFD cases

All 28 files in the seven main geometry/polar/status/command bundles match the main manifest. All sixteen files in the four optional 0.5%-chord bundles also match. The seven main geometries comprise three nominal designs and two selected reference/candidate perturbation pairs. The optional cases are a completed median-like scenario and the largest completed candidate-drag scenario in one existing sample, not estimated population quantiles.

The fixed-geometry files remain useful for exploratory cross-fidelity checks. Their historical trust labels and optimiser names must not be interpreted as repaired-model acceptance. The current geometry-corpus finding strengthens the reason to treat them as fixed examples rather than robust optima.

## CFD commissioning and acceptance checklist

Record executable/version, mesh-generator/config hashes, Re/Mach/reference dimensions, force/moment conventions, turbulence and transition models, boundary conditions, numerical schemes and complete residual/force histories. Match achieved CL 0.4/0.7/1.0, not merely the XFOIL angle guesses. Ncrit 9 is not a CFD turbulence input.

Inspect geometry ordering and trailing-edge gap before meshing. Establish three systematically refined meshes and domain sensitivity on the reference and candidate at representative lift. Wall-resolved y+ around one and farfield clearances suggested in the supplied protocol are initial targets requiring verification, not universal sufficient conditions. Test force stability and conservation, not residuals alone. Document equal geometry/moment constraints; retain ranking reversals and failed/divergent cases. A selected RANS result is a cross-fidelity numerical check, not experimental truth or population CVaR.

**CFD outcomes are absent.** Solver/model choice, convergence budgets and scope require Ava's technical approval. No cloud, CFD or external laboratory job was launched in this review.

Primary method/protocol references: https://web.mit.edu/drela/Public/web/xfoil/ ; https://su2code.github.io/tutorials/Inc_Turbulent_NACA0012/ ; https://doi.org/10.1073/pnas.2204569119 .
''',
'FIGURES_TABLES_AND_STATISTICS_AUDIT.md':r'''
# Figures, tables and statistical interpretation

## Figure-by-figure review

1. Historical test scatter: numerical values reproduce; identify encoded-feature limitation. Move the legend away from panel (b). Do not imply repaired physical-model validity.
2. Amplitude/objective plot: retain penalty-inclusive labels; mark historical operator/admission, not measured manufacturing risk.
3. Perturbation acceptance bars: 100%, 100% and 6/2048 reproduce the original admission. Label that admission as historical; it is not the current post-refit geometry screen.
4. Smoke front: replace the literal backslash-n in the title, shorten the title and explicitly identify schema-4/post-hoc scales. The 44 entries and 0.149292 distance reproduce; this is not production acceptance.
5. Selected XFOIL bars: replace 'robust design' and 'surrogate optimum' labels with historical fixed-candidate descriptions. Retain the adverse marginal maximum and distinguish successful-case means from population expectation.
6. Adversarial search: keep accepted sample count one and overlapping workflow-count caveat. The mean-gap scatter and maximum-gap reference line are different statistics.
7. Converged adversarial example: caption the selected eligible converged subset. Its mean/maximum gap 0.52/0.61 is not the overall 0.67 record.
S1. Selective prediction: shorten overlapping x-axis labels to retained states (%), add clear panel labels and retain historical/retrospective scope. No newly deployed threshold is implied.

## Tables and denominators

The standalone manuscript contains seven tables: nomenclature, CST features, historical baseline comparison, accepted/rejected metrics, historical uncertainty objectives, strict 1% XFOIL outcomes and the two 0.5% scrambles. Log-CD R² cannot be substituted for linear-CD R². Point-wise and aerofoil-macro summaries are separately labelled. Accepted CL coverage 0.97749688 rounds to 97.7%, although a four-decimal table shows 0.9775. Never infer a 95% simultaneous polar band from a state-wise residual quantile.

The 1% means are marginal completed subsets (13/13), whereas paired mean differences use ten matched cases. The adverse marginal maximum is retained. At n=13, empirical CVaR95 is one sample maximum, not a resolved population tail. The two larger scrambles have 92 and 96 completed matched pairs; missingness is flow/geometry dependent, and independent scramble seeds do not turn cases within a scramble into IID observations.

The calibration bootstrap resamples fixed OOF predictions at the geometry-cluster level. Its interval omits full model-refitting variability and is not an optimiser-shift guarantee. The smoke scale is post-hoc. Table values and curves are not combined across these different experimental generations.

## Visual and structural gate

All eight starting figures were opened and inspected. Corrected figures and the final Word export require a new render and page-by-page inspection; the completed results are recorded in `VISUAL_QA.json`. Word structure must contain eight embedded figures and seven tables, valid headers, no V1 manuscript or referee-report concatenation, no missing figure paths and no unresolved citation tokens. Structural assertions alone do not establish layout quality.
''',
'NOVELTY_AND_LITERATURE_POSITION.md':r'''
# Novelty and literature position

## Defensible comparison

CST geometric representations, learned aerodynamic surrogates, evolutionary aerodynamic optimisation, ensembles and CVaR all predate this implementation. Li, Bouhlel and Martins (2019), Li, Du and Martins (2022), and the group's earlier surrogate/CFD work establish the relevant background. Pichitkul et al. (2026), DOI 10.1016/j.apples.2026.100310, is directly relevant existing robust aerofoil work; the current review checked its metadata and publisher abstract, not every full-text method.

NeuralFoil (Sharpe and Hansman, 2025, arXiv:2503.16323v1) was a material missing comparison. Its author-hosted paper combines parametric geometry, learned aerodynamics, confidence information and constrained design. Thus combining ML, geometry and a confidence-aware design loop is not by itself novel here. It is added as reference 29 without claiming a controlled accuracy/speed superiority comparison.

Fannjiang et al. (2022), DOI 10.1073/pnas.2204569119, explains why prediction after model-driven selection requires special coverage treatment. The present frozen support rule and retrospective residual ranking do not supply that treatment. Ordinary conformal-style bands should not be sold as an adaptive-design certificate.

## Possible contribution and current weakness

The useful present material is the explicit distinction among historical prediction reproduction, empirical rejection decisions, candidate-specific uncertainty, coordinate integrity and selected solver evidence. The geometry defect and changed fixed-design objectives are scientifically informative negative findings; they are not signs that the old stronger claim has now become valid. A successful repaired-lineage study could quantify whether a prospectively selected support-plus-disagreement rule improves equal-budget design decisions and failure detection. This has not yet been demonstrated.

A limited fixed-geometry XFOIL case study may be publishable in an appropriate context, but three deliberately selected designs and same-solver labels do not establish general engineering value over existing tools. The authors must choose whether to rebuild the originally intended study or explicitly adopt a narrower contribution. This review does not choose a journal or claim its acceptance criteria are met.

## Bibliography audit boundary

`REFERENCES_VERIFIED.csv` covers all 28 inherited entries and the added reference 29, with status, primary source, support scope and access limitation for each. Eighteen inherited DOI entries have stored Crossref primary metadata. Official author/conference/publisher pages were used for non-DOI entries where available. Airfoil Tools access failed; the exact cited editions/years of two general background books remain partly unconfirmed. Neither unresolved book supports a numerical result, and they are identified as background rather than implied evidence. A publisher page for a seventh edition is not verification of a cited sixth edition. No fabricated DOI or page-level full-text claim is supplied.

Primary links: https://doi.org/10.2514/1.J057129 ; https://doi.org/10.1016/j.paerosci.2022.100849 ; https://doi.org/10.1016/j.apples.2026.100310 ; https://arxiv.org/html/2503.16323v1 ; https://doi.org/10.1073/pnas.2204569119 .
''',
'MANUSCRIPT_STYLE_AND_EDIT_LOG.md':r'''
# Manuscript style and edit record

## Style reference

The separate `prior_final_draft.docx` was read for tone and structure: methods-first paragraphs, practical aerodynamic motivation, compact parameter tables and direct explanation of modelling choices. Its obsolete Physics Score, fixed-grid padding, family-held-out claims and high-fidelity wording were not copied into the revision. V1 remains a separate historical document, never merged into the current paper.

## Integrated scientific edits

The draft title is narrowed to empirical reliability screening and selected XFOIL comparisons. This is a proposed coauthor revision, not an accepted journal title. Abstract, highlights, data methods, results, discussion and conclusion now disclose the 1,001-feature-vector coordinate defect and the absence of a rebuilt model. Historical versus current source/evidence are distinguished throughout.

The architecture is corrected to an input projection plus two two-affine residual blocks, with piecewise-differentiable ReLU. Lift coverage rounding and geometry-cluster OOF values are corrected. The manufacturing field is described as prescribed, correlated and piecewise linearly interpolated, not measured or spline-based. The current seven invalid knee refits and its worse mean objective are integrated, rather than retaining a superseded favourable value. Post-hoc smoke normalisation no longer supports acceptance language.

The fixed-coordinate XFOIL positives remain, with matched/marginal denominators, failed-solver selection, the adverse marginal maximum and same-solver limitation. The converged adversarial figure is separated from the overall largest discrepancy. Existing reviewer responses are treated as a historical-response working draft, not a new venue endorsement. Web dates and broken primary links are corrected; NeuralFoil is added as relevant prior work. Exact unresolved book editions are recorded, not invented.

## Presentation edits

Fix the figure legend over panel (b), literal escaped newline, overclaiming candidate labels and overlapping selective-coverage labels. The Word export retains one standalone paper, eight figures, seven tables and its own references. Review instructions, claim matrix, source manifests, V1 and CFD workflow remain separate. Final rendered-page findings are recorded in `VISUAL_QA.json`.

## Preservation and responsibility

The pre-review manuscript, Word document, reviewer response and paper README are retained under `BEFORE_REVIEW`; the base Git commit provides the original source. Source edits have RED/GREEN evidence in the review logs and the final Git diff. The final source-to-document hash map records exact outputs. The authors must approve the proposed title, wording, authorship, funding/conflict/AI disclosures and any submission; this review cannot attest personal declarations on their behalf.
''',
'OPEN_SCIENTIFIC_DECISIONS_FOR_BEN_AVA.md':r'''
# Open scientific decisions for Ben and Ava

## Decisions that require author judgement

**Geometry acceptance and rebuilt lineage.** Choose and freeze supported coordinate formats, ambiguous-record disposition, maximum fit-error/representation criteria, reconstructed-geometry validity and physical near-duplicate grouping. Rebuild data and splits, retrain and recalibrate, and reserve a genuinely uninspected confirmation cohort. Source fixes do not permit silently substituting new inputs under old weights.

**Contribution and title.** Decide whether to complete the intended robust-design research or submit a narrower reproducibility/fixed-geometry numerical contribution. The revised title is a proposed safe description. Do not claim a full robust optimum by dropping the outstanding production gate without making the changed research question explicit.

**Prospective optimisation acceptance.** Fix objective scales and convergence/agreement criteria before seeing the repaired campaign. Keep the declared 0.20 threshold or explicitly justify a new protocol before execution; do not tune a pooled range to make a front pass. Select final geometries using a rule frozen before solver validation.

**CFD scope and transition physics.** Ava should approve the solver, turbulence/transition treatment, equal-lift constraints, mesh/domain studies and selected cases. Existing fixed geometry bundles are intact but are not newly accepted robust optima. Keep failures, ranking reversals and the uncertainty of same-solver comparisons visible.

**Manufacturing interpretation.** Supply real tolerance/correlation measurements or retain prescribed geometric sensitivity wording. Do not substitute a new smooth field and reattribute old solver outcomes to it.

**Publication and declarations.** Confirm the two background-book editions if retained, third-party data/model redistribution terms, exact prior-venue resubmission restrictions, destination journal requirements and all coauthor/CRediT/conflict/funding/AI declarations. No current journal approval, submission invitation or one-year-policy verification is claimed here.

## Not decisions delegated back unnecessarily

The coordinate parser, common-x thickness, reconstructed-refit admission, calibration-group integrity, model-artifact hashing, stale-cache protection, post-hoc acceptance label, manuscript arithmetic and visible figure defects were repairable within the authorised scope. Those were actioned rather than left as suggestions. The reports and package provide exact evidence and a reconstruction path for the remaining scientific work.
''',
'SOFTWARE_AND_REPRODUCIBILITY_REPORT.md':r'''
# Software and reproducibility report

## Environment and boundary

Work is performed in the authorised existing `robust-v2-viability` branch on SamsPC, with base `156f07f77e92f77df34a6cc6f6a1726cc0e1638a`, not a reset clone. The source/model/data lineage remains preserved. Windows, Python 3.11.15, uv, the existing CUDA-capable PyTorch environment and local XFOIL executable were available. The final command log records actual dependency resolution, versions and exit codes rather than inferring them from a previous green run.

## Reproduced and fixed defects

Tests cover wrong coordinate ordering, reversed-x interpolation and thickness, nonfinite inputs, missing required-lift feasibility, angle rejection, checkpoint/scaler integrity, stale scientific caches, invalid reconstructed candidate and training fits, calibration cluster separation, and post-hoc front-scale acceptance. Each material source change was preceded by a reproduced failure; old and new outputs are retained. The added scientific-admission tests tighten rather than skip assertions. Historical cache mismatch is a stop requiring a new revision, not permission to overwrite evidence.

The available full non-network test invocation includes the local XFOIL option. Its final result, the prescribed non-network/non-slow subset, Ruff, mypy, uv sync, diff hygiene, current author-pack verifier and rendered document checks are in `validation_final/results.json`, `FINAL_ACCEPTANCE.json` and `VISUAL_QA.json`. A software pass does not mean the repaired model was trained or the production study was run.

## Reproduction commands

From the repository root on the documented machine:

    uv sync --extra dev --extra benchmarks --extra docs
    uv run ruff check src/robust_airfoil tests/robust_v2 scripts/robust_v2
    uv run mypy src/robust_airfoil
    uv run pytest tests/robust_v2 -m "not network and not xfoil and not slow" -q
    uv run pytest -m "not network" --run-xfoil -q
    uv run --extra docs python scripts/robust_v2/verify_final_author_pack.py
    uv run python reports/robust_v2/external_academic_review_20261014/recompute_ml.py
    uv run python reports/robust_v2/external_academic_review_20261014/audit_geometry_corpus.py
    uv run python scripts/robust_v2/audit_trust_incremental_calibration.py
    uv run python reports/robust_v2/external_academic_review_20261014/audit_raw_xfoil_and_fronts.py
    uv run python reports/robust_v2/external_academic_review_20261014/profile_candidate.py

Read script output paths before rerunning: analytical outputs belong to this review revision. Frozen source polars, weights and original reports must not be replaced. Large local artifacts are necessary for numerical reproduction and are not falsely represented as fully committed or publicly redistributable.

## Release integrity

The final Git commit is pushed normally to the same branch; no force push or merge is required. The OneDrive package is refreshed only from a clean repository whose HEAD equals the fetched remote branch. Existing package generations are moved into labelled history rather than deleted. Every packaged file is hashed; ZIP archives are CRC-tested. The offline-copy script refuses content/branch/commit mismatches and does not reset or switch the repository. The post-push package manifest carries the final commit, avoiding a self-referential commit-hash claim in pre-commit evidence.

## Review limitation

The final code/package inspection is an explicit self-review in this tool environment, not a fictitious independent agent review. The scientific conclusions remain subject to coauthor judgement and independent validation.
''',
'TAKEOVER_AND_REPRODUCTION.md':r'''
# Takeover and reproduction

## Start here

Read `ACADEMIC_REFEREE_REPORT.md`, `FINAL_ACCEPTANCE.json`, `CLAIM_EVIDENCE_MATRIX.csv` and `OPEN_SCIENTIFIC_DECISIONS_FOR_BEN_AVA.md`. The finished review outcome is major scientific revision, not a submission-ready robust optimum. Do not repeat the old smoke campaign and present it as closure.

Canonical repository: `C:\Users\Ben\workspaces\github.com\bengawith\paper`, branch `robust-v2-viability`. Canonical author package: `C:\Users\Ben\OneDrive\paper\READY_FOR_REVIEW`. The package `SHA256_MANIFEST.json` identifies the actually pushed release commit and all packaged file hashes. Cloud synchronisation to another device is not established by local file existence.

## Scientific sequence for a new approved lineage

Freeze repaired geometry admission and fitting rules; rebuild source-to-polar mappings and physical clusters; create a fresh confirmation design; retrain and recalibrate without inspecting that confirmation cohort; repeat equal-protocol simple baselines; prospectively compare screening policies; run candidate-specific production optimisation with fixed objective normalisation; select designs without using their subsequent validation outcomes; execute strict solver and agreed independent CFD checks; reconcile every claim and adverse result before submission.

The current training corpus and weights are immutable historical evidence. Do not reuse their old cache namespace for new shapes or silently redefine an old test set. The active cache refuses mismatched evidence. Actual model/scaler artifacts, source hashes and configurations must accompany any new result.

## Existing reproducible audits

`GEOMETRY_REPAIR` gives every coordinate/fit finding. `ML_RECOMPUTATION` gives historical inference and split/scaler/bootstrap checks. `TRUST_RECOMPUTATION` retains the initial independent grouping comparison, while `CANONICAL_TRUST_INCREMENTAL` is the corrected canonical script's result. `RAW_XFOIL_RECOMPUTATION` covers all 825 stored cases, all 44 CFD-input file hashes and the historical ten-seed distance recalculation. `COMPUTE_PROFILE` holds the final-source five-design diagnostic, including seven invalid knee refits and its worse mean objective.

Run the commands in the software report from the repository root using the locked environment and available local evidence archives. The pre-review text/Word files are preserved in `BEFORE_REVIEW` and the original base is retained in Git history. No automated task or unobserved research process is promised to continue after this review.

## Offline handoff

Use the packaged `COPY_TO_LAPTOP_OFFLINE.ps1` only after reading its parameters. It verifies the source manifest, requires the matching clean repository branch and commit, refuses an existing destination and verifies the copied bytes. It never uses `git switch -C`, reset, clean or forced update. Failure is a reportable mismatch, not permission to discard local work. The copied package is a review snapshot, not a claim that a newly trained model or CFD exists.
'''
}
for name,text in reports.items():write(name,text)

# High-value claims are linked to exact source hashes and selectors, not prose-only assurances.
claims=[]
def claim(identifier,location,statement,source,selector,value,status,limits):
    p=ROOT/source
    assert p.is_file(),source
    claims.append({'claim_id':identifier,'manuscript_location':location,'claim':statement,'source_path':source,'source_sha256':sha(p),'selector':selector,'recomputed_value':value,'status':status,'limit_or_action':limits})
R='reports/robust_v2/external_academic_review_20261014/'
claim('C01','Abstract;2.3','Coordinate repair changes stored feature vectors',R+'GEOMETRY_REPAIR/GEOMETRY_AUDIT.json','changed parameter encodings','1001/1560','CRITICAL_LINEAGE_BLOCKER','Source repaired; rebuild model and physical grouping before generalisation claims')
claim('C02','2.3','Ambiguous geometries rejected, not guessed',R+'GEOMETRY_REPAIR/GEOMETRY_AUDIT.json','rejected inputs','3','REPRODUCED','Three rejected records are not part of the 1557 comparable fit statistics')
claim('C03','2.3','Comparable maximum-fit errors above 0.005 chord',R+'GEOMETRY_REPAIR/GEOMETRY_AUDIT.json','comparable old/new counts','931/289 of1557','REPRODUCED','All1560 historical old count933 has a different denominator')
claim('C04','2.3','Reconstructed repaired fits failing validity',R+'GEOMETRY_REPAIR/GEOMETRY_AUDIT.json','refitted geometry invalid','656','REPRODUCED','A repaired parser does not guarantee a valid or sufficiently accurate CST fit')
claim('C05','2.2','Stored split has disjoint identifiers',R+'ML_RECOMPUTATION/ML_AUDIT.json','partition/overlap audits','1093/234/233 airfoils;1088/233/233 clusters','HISTORICAL_ONLY','Misspecified geometry can undermine physical grouping despite disjoint recorded identifiers')
claim('C06','3.1','Model architecture and member integrity',R+'ML_RECOMPUTATION/ML_AUDIT.json','architecture;artifact validation','5 hidden affine transforms;1064451 parameters/member','REPRODUCED','Two residual blocks are not two hidden affine layers')
for n,target,value in [('07','CL','0.9886904895'),('08','logCD','0.9218477075'),('09','CM','0.9180108026')]:
    claim('C'+n,'4.1;Fig1','Historical R2 '+target,R+'ML_RECOMPUTATION/ML_AUDIT.json','target metrics',value,'HISTORICAL_REPRODUCED','Not repaired-geometry model accuracy')
claim('C10','4.1;Table3','Accepted coverage lift',R+'ML_RECOMPUTATION/ML_AUDIT.json','accepted CL coverage','0.9774968824','CORRECTED_ROUNDING','97.7 percent at one decimal; no simultaneous/adaptive guarantee')
claim('C11','4.1','Frozen screen adds no independent spread rejection',R+'ML_RECOMPUTATION/ML_AUDIT.json','trust component audit','9050 support failures;0 extra spread rejection','REPRODUCED','Attribution to support-dominated original screen')
claim('C12','3.2;4.1','Geometry-cluster-separated OOF AUROC',R+'CANONICAL_TRUST_INCREMENTAL/TRUST_INCREMENTAL_CV.json','models.support_only / support_plus_ensemble','0.7679624719 / 0.8667371583','CORRECTED_AND_REPRODUCED','233 stored geometry clusters; no operational policy change')
claim('C13','4.1','Fixed-OOF group bootstrap difference interval',R+'CANONICAL_TRUST_INCREMENTAL/TRUST_INCREMENTAL_CV.json','cluster_bootstrap_95pct_interval','0.0666791433 to0.1402019524','DESCRIPTIVE_REPRODUCED','300 cluster resamples; no whole-procedure refit uncertainty')
claim('C14','3.3','Active field interpolation/taper/normalisation','src/robust_airfoil/uncertainty/fields.py','correlated_control_field;smooth_normal_perturbation','piecewise-linear;per-draw max;LE/TE taper','SOURCE_VERIFIED','Prescribed sensitivity model, not measured manufacturing tolerances or cubic splines')
claim('C15','3.4','Required-lift feasibility and penalty objective','src/robust_airfoil/optimisation/objectives.py','required_lift_drag;weighted_required_lift_drag','CL0.4/0.7/1.0 weights0.25/0.5/0.25','SOURCE_VERIFIED','Initial rising branch; invalid/missing targets do not become physical drag')
claim('C16','4.3','Historical schema4 smoke front distance',R+'RAW_XFOIL_RECOMPUTATION/HISTORICAL_FRONT_RECALCULATED.json','mean_symmetric_distance;seeds','0.1492924385;44 feasible entries/10 seeds','HISTORICAL_DIAGNOSTIC','Post-hoc pooled scales and superseded refit checks; not production acceptance')
claim('C17','4.3','Current fixed-design reference objective',R+'COMPUTE_PROFILE/CURRENT_SELECTED_DESIGNS.json','rows[verified_reference]','0.0094205632 /0.0191844201;8 exits;0 invalid','REPRODUCED_CURRENT_SOURCE_OLD_MODEL','Not a newly trained or optimised result')
claim('C18','4.3','Current fixed-design knee objective',R+'COMPUTE_PROFILE/CURRENT_SELECTED_DESIGNS.json','rows[trust_constrained_pareto_knee]','0.0095851571 /0.0151574330;15 exits;7 invalid','ADVERSE_RESULT_RETAINED','Mean worse than reference; supersedes old favourable0.009040 mean')
claim('C19','4.4','All stored selected coordinates reconstruct',R+'RAW_XFOIL_RECOMPUTATION/all_825_cases.csv','coordinate_identity;physical_admissible','825/825 both','REPRODUCED','Direct-coordinate validity separate from ML feature validity and solver convergence')
claim('C20','4.4;Table5','Nominal selected XFOIL drag',R+'RAW_XFOIL_RECOMPUTATION/XFOIL_RECALCULATED.json','campaigns[0].designs.*.nominal','reference0.0074464020;candidate0.0060511315','REPRODUCED','18.7 percent reduction for deliberately selected fixed example only')
claim('C21','4.4;Table5','Nominal outside-screen comparison',R+'RAW_XFOIL_RECOMPUTATION/XFOIL_RECALCULATED.json','campaigns[0].designs.outside_trust_deterministic.nominal','0.0139600174','REPRODUCED','87.5 percent higher; not a population estimate of screen utility')
claim('C22','Table5;Fig5','1 percent marginal completed outcomes',R+'RAW_XFOIL_RECOMPUTATION/XFOIL_RECALCULATED.json','campaigns[0].designs','13/13/14 usable;candidate maximum worse','ADVERSE_RESULT_RETAINED','Marginal subsets differ; 95 percent empirical tail at this size is maximum')
claim('C23','4.4','1 percent paired mean comparison',R+'RAW_XFOIL_RECOMPUTATION/XFOIL_RECALCULATED.json','campaigns[0].paired','10pairs;difference0.0010281660;12.2171percent','CONDITIONAL_REPRODUCED','Not the marginal8.8percent estimand')
claim('C24','4.4;TableS1','0.5 percent scrambleA paired mean',R+'RAW_XFOIL_RECOMPUTATION/XFOIL_RECALCULATED.json','campaigns[1].paired','92pairs;15.3996percent;CVaRchange-12.4043percent','CONDITIONAL_REPRODUCED','36 requested pairs not jointly usable; no imputation')
claim('C25','4.4;TableS1','0.5 percent scrambleB paired mean',R+'RAW_XFOIL_RECOMPUTATION/XFOIL_RECALCULATED.json','campaigns[2].paired','96pairs;15.7803percent;CVaRchange-11.6701percent','CONDITIONAL_REPRODUCED','32 requested pairs not jointly usable; no imputation')
claim('C26','5','Main CFD handoff file integrity',R+'RAW_XFOIL_RECOMPUTATION/CFD_MAIN_INPUT_HASH_AUDIT.json','all matches','28/28','HASH_VERIFIED','Input readiness only; no CFD output exists')
claim('C27','5','Optional CFD handoff file integrity',R+'RAW_XFOIL_RECOMPUTATION/CFD_OPTIONAL_INPUT_HASH_AUDIT.json','all matches','16/16','HASH_VERIFIED','Selected cases are not population tail quantiles')
claim('C28','4.5;Fig6/7','Finite adversarial evidence','results/robust_v2/lineages/robust-v2-current-20260908/surrogate_stress/surrogate_stress_summary.json','records;worst_case_overall','overall about0.67;eligible converged subset mean0.52 maximum0.61','HISTORICAL_SCOPED','Overlapping workflow counts and only1 fully accepted probe; not global bound')
claim('C29','4.6','Predictor-only latency scope','scripts/robust_v2/benchmark_inference.py','CUDA timing boundaries','single/polar/batch are distinct workloads','SOURCE_VERIFIED_HISTORICAL_TIMINGS','No full-workflow speedup; no current raw benchmark rerun claimed')
claim('C30','1;5','Novelty comparison includes existing confidence-guided design',R+'REFERENCES_VERIFIED.csv','references4,5,27,28,29','Established components;limited specific integration','PRIMARY_LITERATURE_SCOPED','No first-ever or superiority claim; record fulltext access limits')
claim('C31','Table2','Baseline development comparison','paper/data/paper_data.json','baseline metrics','MLP0.984/0.904/0.875;HGB0.983/0.915/0.760','ARCHIVED_REPORT_ONLY','Not fresh baseline training; geometry repair invalidates stronger physical interpretation')
claim('C32','Table4;Fig2/3','Historical uncertainty sample summaries','paper/data/paper_data.json','manufacturing_levels;perturbation_ablations','2048draws;0/0/0.4/17percent exits','HISTORICAL_ONLY','Original admission not current post-refit checks')
with (OUT/'CLAIM_EVIDENCE_MATRIX.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(claims[0]));w.writeheader();w.writerows(claims)

status='''# Robust V2 — completed academic review, major scientific revision required

The canonical review is `reports/robust_v2/external_academic_review_20261014/ACADEMIC_REFEREE_REPORT.md`. The current standalone manuscript is `paper/LATEST_MANUSCRIPT_FOR_AVA.docx`; Markdown source is `paper/manuscript.md`.

The review identified a coordinate-order defect changing 1,001 of 1,560 stored feature vectors. Parser, geometry/refit admission, calibration grouping, model integrity and cache/acceptance safeguards are corrected in source, but the training corpus, physical grouping, model and full optimisation have not been regenerated. Do not describe the historical R² values as repaired-model validation.

Positive fixed-geometry local-XFOIL results reproduce: 18.7% lower nominal drag and 15.4%/15.8% lower means among 92/96 completed matched 0.5%-chord cases. These remain same-solver, selected, conditional comparisons. Current refit admission also reveals seven invalid knee scenarios and a worse penalised mean objective; no robust-optimality claim is accepted.

The completed review package, claim matrix, reference audit and CFD-input checks separate software/review completion from open scientific gates. `FINAL_ACCEPTANCE.json` is authoritative for this revision. The old `submission_review_20261014` directory is historical evidence, not the current release decision. Coauthor approval, repaired-lineage evaluation, prospective production optimisation and independent CFD remain open.

Validate with `uv sync --extra dev --extra benchmarks --extra docs`, the documented Ruff/mypy/test commands, and `uv run --extra docs python scripts/robust_v2/verify_final_author_pack.py`. Large local model/data archives are required for numerical reproduction; not all are publicly committed.
'''
(ROOT/'README.md').write_text(status,encoding='utf-8')
(PAPER/'README.md').write_text(status.replace('`paper/LATEST_MANUSCRIPT_FOR_AVA.docx`','`LATEST_MANUSCRIPT_FOR_AVA.docx`').replace('`paper/manuscript.md`','`manuscript.md`'),encoding='utf-8')
(PAPER/'MODEL_CARD.md').write_text('''# Frozen model card — historical encoded-feature lineage

**Not accepted for repaired-geometry design use.** Lineage: `robust-v2-current-20260908`.

Five point-conditioned residual MLP members map 12 CST parameters plus incidence to CL, log-CD and CM at Re1e6, Mach0, Ncrit9. Each member has five hidden affine transformations of width512 and 1,064,451 trainable parameters. Recorded inference and metric reproduction are historical encoded-feature checks only.

A coordinate-order defect changes1,001/1,560 stored feature vectors under repair. The parser and admission code are corrected, but weights and support/calibration were not retrained. Unknown/new physical geometry must not be accepted merely because the old empirical screen returns true. The original gate is support dominated; spread cutoffs add no independent held-out rejection.

Known limitations: corrupted/misinterpreted training features; incomplete CST representability; correlated polar observations; retrospective inspected test data; one Reynolds/Mach/transition condition; same-XFOIL-family labels; no independent CFD/experiment; failed-case selection; unmeasured manufacturing prior; no completed corrected production optimum. Source integrity hashes prevent silently changing model/scalers; they do not certify scientific fitness.

Use only for reproducibility/diagnostics until a new accepted corpus, split, training, calibration and prospective confirmation lineage is produced. Read the current academic referee report and FINAL_ACCEPTANCE.json.
''',encoding='utf-8')
(PAPER/'response_to_reviewers.md').write_text('''# Historical reviewer-response working draft — not for submission unchanged

This document concerns the earlier EAAI-26-2180 reviews, not a current review invitation. The proposed revised manuscript is "Empirical Reliability Screening for Machine-Learning Aerofoil Prediction, with Selected XFOIL Comparisons". Coauthors must approve the scope and verify the original venue's restrictions before any submission.

## Usefulness and adversarial design (earlier Reviewer 2)

A finite search and three selected fixed-geometry local-XFOIL comparisons are supplied. The selected candidate has18.7% lower nominal required-lift weighted drag; two0.5%-chord scrambles have92/96 completed matched pairs with15.4%/15.8% lower means. These are conditional, same-solver examples, not a universal trust bound or unconditional manufacturing expectation. The overall discrepancy and the selected converged plotting subset are distinguished. Failed solves and the adverse1%-chord marginal maximum are retained.

## Model scope, data, geometry and solver fidelity (earlier Reviewer 3)

The point-conditioned representation eliminates padded supervised targets and includes CL, log-CD and CM at one explicit operating condition. However, the new audit found a coordinate-order defect affecting1,001 stored feature vectors. Source repair is not a retrained model: corpus acceptance, grouping, training/calibration and fresh confirmation must be completed before the original physical-map accuracy claim can stand. XFOIL-on-XFOIL checks do not answer the need for independent CFD or experiment.

## Figures and high-incidence interpretation (earlier Reviewer 6)

Figure captions, panels and labels are revised, including an obscured panel letter, literal escaped newline and overlapping axes. High-incidence failures are not asserted to occur only outside trust. One accepted adversarial probe cannot establish a domain-wide maximum. Final Word layout is checked separately from numerical reproduction.

## Positive writing feedback (earlier Reviewer 7)

The methods-first style of the earlier paper is retained while redundant assurances and unsupported physical claims are removed. The current manuscript remains one standalone paper, with V1 and the referee report kept separately.

## Novelty, model choice and engineering contribution (earlier Reviewer 8)

The historical grouped-development MLP/gradient-boosting/ridge/mean comparison is reported without architecture-superiority claims. Existing robust optimisation and confidence-aware aerodynamic tools, including NeuralFoil, are discussed explicitly. The original deployed screen is support dominated; a corrected calibration-only geometry-cluster audit shows incremental spread ranking information, not a newly validated operational trust gate. The architecture, uncertainty generator and CVaR/penalty meanings are corrected.

## Unresolved substantive response

The scientific response cannot be described as complete before a repaired geometric model lineage and prospective evaluation exist. Full candidate-specific production optimisation with fixed acceptance normalisation and the agreed CFD study also remain open. This working draft does not claim that all reviewer concerns have been resolved by software tests. See the current referee report, claim matrix and open author decisions for exact evidence and remaining scope.
''',encoding='utf-8')
write('READ_THIS_FIRST.md',status+'\nThe OneDrive copy is rebuilt after the final normal push; verify its root SHA256_MANIFEST.json before relying on an offline copy. No email has been sent and no journal submission has been made.\n')
write('PROGRESS.md', '# Academic review progress\n\nBase '+BASE+'; authorised in-place branch.\n\nCompleted: full manuscript/evidence read; raw ML/geometry/calibration/XFOIL/CFD-input/front recomputations; source fixes with reproduced failures; integrated manuscript and reports. Final visual validation, final command gate and post-push package record are separate recorded steps.\n\nRuling: no new production run on the compromised frozen geometric lineage; no manufactured CFD, independent reviewer or coauthor approval. Historical outcomes preserved and labelled; proposed narrower title requires coauthor approval.\n')
print('DOCUMENTS_ASSEMBLED',len(reports),'reports;',len(claims),'claims;',len(reference_rows),'references;',len(body.split()),'body words')
