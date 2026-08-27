# End-to-end execution guide

## Scientific destination

Build a point-conditioned surrogate

\[
(\mathbf p,\alpha)\rightarrow(C_L,\log C_D,C_M)
\]

at the fixed source condition \(Re=10^6\), \(M=0\), \(N_{crit}=9\), then use it
inside a trust-calibrated, risk-aware aerofoil optimisation under bounded,
correlated surface uncertainty.

The old 12-to-48 \(C_L\) network is retained as the historical baseline. It must not
define the new data model because its wide fixed grid required padded targets.

## Non-negotiable rules

1. Preserve the rejected-paper state with a Git tag and hashes.
2. Never modify raw coordinates or raw polar responses.
3. Do not fill failed/missing source angles for training.
4. Split by nominal geometry cluster, never by point row.
5. Fit scalers and choose hyperparameters without the locked test set.
6. Treat direct XFOIL as source-method verification, not independent high fidelity.
7. Do not call synthetic displacement a measured manufacturing distribution.
8. Do not maximise lift without drag, moment, thickness, and trust constraints.
9. Do not run bulk source requests until the one-, 25-, and 200-record pilots pass.
10. Every table and figure must be reproducible from a committed config, split,
    dataset hash, code SHA, model SHA, and seed.

## Stage A — repository freeze

In a clean parent directory:

```powershell
git clone https://github.com/bengawith/paper.git
git clone https://github.com/bengawith/curve_gen.git
cd paper
git fetch --all --tags --prune
git status --short
git rev-parse HEAD
git tag --list submitted-v1
git tag -a submitted-v1 -m "Exact repository state used by the rejected paper"
git switch -c robust-v2
New-Item -ItemType Directory -Force reports\setup, logs, data\legacy | Out-Null
git rev-parse HEAD | Out-File reports\setup\starting_sha.txt
git ls-files | Out-File reports\setup\tracked_files.txt
Get-FileHash data\csv\dataset_12CST_params.csv -Algorithm SHA256 |
  Format-List | Out-File reports\setup\legacy_dataset_sha256.txt
```

If `submitted-v1` or `robust-v2` already exists, inspect it rather than overwriting it.

Copy only reusable interface/geometry ideas from `curve_gen`; its recurrent 14-input,
96-output model is not the scientific base.

## Stage B — environment

Use Python 3.11 for the new branch. Keep any working historical environment separate
as `.venv-legacy`.

```powershell
winget install --id=astral-sh.uv -e
uv python install 3.11
uv venv --python 3.11
uv sync --extra dev
.venv\Scripts\Activate.ps1
python scripts\00_environment_report.py
nvidia-smi
```

Install PyTorch using the current official selector after reading the GPU/driver
output. Verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Commit `pyproject.toml`, `.python-version`, and the resolved `uv.lock`.

## Stage C — legacy baseline

Before changing the model:

1. Hash the CSV, saved model, scalers, config, and evaluation results.
2. Audit the wide dataset's exact columns and shape.
3. Reconstruct the original random split with seed 42 in a new reproduction script.
4. Load the saved MLP and scalers.
5. Recompute pointwise valid-region metrics.
6. Compare every number against `trained_model/evaluation_results.json`.
7. Record discrepancies rather than tuning until they disappear or are explained.

The current repository has inconsistent angle defaults and metric return signatures.
Repair those in a versioned reproduction module; do not pretend the old script runs
unchanged.

**Gate:** a clean command creates `reports/reproduction_v1.json` and identifies the
exact legacy metric state.

## Stage D — recover historical source data

Search the local PC, OneDrive, backups, and old working directories for:

- raw AirfoilTools CSV/TXT files;
- pickles or JSON caches;
- the pre-paper generator script;
- intermediate datasets containing `Cd`, `Cdp`, `Cm`, `Top_Xtr`, or `Bot_Xtr`;
- old environments and `pip freeze` outputs.

Decision:

- **A — raw polar files found:** parse and hash them; no rescrape for those keys.
- **B — trusted cache found:** export to immutable text/Parquet and preserve the
  original cache hash.
- **C — only lift-only processed CSV found:** perform controlled re-ingestion.
- **D — source unavailable:** run a small local-XFOIL feasibility dataset and do not
  silently substitute another neural model as training truth.

## Stage E — source mapping

Build `airfoil_mapping.csv` with:

- `uiuc_filename`
- `uiuc_stem`
- `legacy_dataset_name`
- `canonical_name`
- `airfoiltools_slug`
- `polar_key`
- `raw_geometry_sha256`
- `normalised_geometry_sha256`
- `mapping_method`
- `mapping_status`
- `review_note`

Mapping order:

1. exact historical key;
2. exact filename/stem;
3. explicit alias table;
4. geometry comparison;
5. manual review.

Never auto-accept a fuzzy name match. Quarantine ambiguous and unmatched items.

**Gate:** the mapping report gives counts for exact, alias, ambiguous, unmatched,
invalid-geometry, and missing-polar cases.

## Stage F — raw polar ingestion

Preserve the complete response. Parse the header dynamically. Validate each response's
Reynolds number, Mach, Ncrit, XFOIL version, airfoil name, and point count.

Keep all genuine source rows and all columns:

- alpha
- CL
- CD
- CDp
- CM
- upper transition location
- lower transition location

Missing angles remain missing. Do not select 48 rows by index, interpolate a common
range, or pad endpoints.

Run in this sequence:

1. parser fixture;
2. one verified polar;
3. 25 mapped aerofoils;
4. 200 mapped aerofoils;
5. full accepted mapping.

After each stage, inspect source-condition mismatches, parse failures, HTTP outcomes,
duplicate alpha rows, field completeness, and raw hashes.

## Stage G — geometry processing

For every UIUC coordinate file:

1. detect Selig versus Lednicer ordering;
2. remove only exact duplicate points;
3. translate/rotate/scale to unit chord;
4. orient TE-upper → LE → TE-lower;
5. split upper/lower surfaces;
6. interpolate each onto a 201-point cosine-spaced x grid;
7. test crossing and local thickness;
8. fit the 12-parameter CST representation;
9. reconstruct coordinates;
10. record RMSE and maximum error;
11. calculate thickness, camber, area, leading-edge radius, and TE thickness;
12. retain the raw and processed geometry hashes.

Do not choose a CST fit-error cutoff before plotting the entire error distribution and
manually inspecting the worst cases.

## Stage H — long-form tables

Write immutable, schema-checked Parquet tables:

- `airfoils.parquet`
- `polars.parquet`
- `polar_points.parquet`
- `model_points.parquet`

The final modelling view contains one genuine source operating point per row. It uses
`log_cd = log(cd)` only for finite positive drag values. Invalid zero/negative drag is
quarantined, not repaired with an arbitrary epsilon.

Give every nominal aerofoil equal aggregate sample weight:

\[
w_{i,k}=1/n_i
\]

where \(n_i\) is the accepted point count for aerofoil \(i\).

## Stage I — audits before modelling

Produce:

- geometry parse report;
- coordinate/CST round-trip report;
- name/polar join report;
- condition-consistency report;
- alpha-coverage heat map;
- target-completeness report;
- duplicate-alpha report;
- exact/near-duplicate geometry report;
- family/cluster coverage report;
- source-version report;
- accepted/rejected row ledger.

Lock the primary operating interval only after these plots exist. A pre-stall interval
is expected, but it must be selected from data coverage and solver reliability, not
inherited uncritically.

## Stage J — leakage-resistant splitting

1. Hash exact normalised geometries.
2. Represent each geometry by upper/lower ordinates on the common x grid.
3. calculate nearest-neighbour shape distances.
4. inspect nearest pairs manually.
5. freeze a documented near-duplicate threshold.
6. form connected geometry clusters.
7. assign whole clusters to 70/15/15 train/validation/test partitions.
8. keep all operating points and future perturbations of a nominal airfoil together.
9. use family holdouts only as secondary stress tests.

Publish the manifests before model tuning.

## Stage K — baseline model

Primary formulation:

\[
[12\ {\rm CST},\alpha]\rightarrow[C_L,\log C_D,C_M]
\]

Start with:

- shared MLP trunk;
- three output heads;
- three hidden layers, width 256;
- SiLU activation;
- dropout 0.10;
- AdamW;
- learning rate \(10^{-3}\);
- weight decay \(10^{-5}\);
- batch size 1024, reduced if memory requires;
- output standardisation fitted on training only;
- Huber or log-cosh loss in standardised target units;
- equal initial target weights;
- equal aggregate nominal-airfoil weight.

Run:

1. one-batch shape test;
2. one-batch overfit test;
3. 1,000-row smoke training;
4. 25-airfoil training;
5. 200-airfoil training;
6. full training.

A failed one-batch overfit test means the data/model/loss pipeline is wrong; tuning
must not begin.

Report CL, log-CD, CD, and CM metrics separately. Never restore the mixed-unit Physics
Score as the primary selector.

## Stage L — tuning

Use grouped inner folds only. Keep the locked test closed.

Initial Optuna search:

- hidden layers: 2–6;
- width: 128–512;
- residual blocks: off/on;
- activation: ReLU/GELU/SiLU;
- dropout: 0–0.30;
- learning rate: \(10^{-5}\)–\(3\times10^{-3}\), log scale;
- weight decay: \(10^{-8}\)–\(10^{-3}\), log scale;
- batch size: 512–8192 subject to memory;
- Huber delta or log-cosh choice;
- optional learned task weighting only as a later ablation.

Objective:

\[
\frac{1}{3}\left(
{\rm NMAE}_{CL}+{\rm NMAE}_{\log CD}+{\rm NMAE}_{CM}
\right)
\]

averaged across grouped folds, with each aerofoil weighted equally.

Run the first reproducibility study sequentially with a fixed sampler seed. Save the
Optuna SQLite database and every trial's config, fold result, epoch, seed, runtime,
and failure reason. Parallel studies can be used later for speed but are not expected
to reproduce an identical suggestion sequence.

Select a stable region of the search space, not merely the numerically best noisy
trial. Refit the top configurations with several seeds.

## Stage M — final evaluation and trust calibration

After architecture, filters, splits, loss, and metrics are frozen:

1. train five independently seeded/group-resampled ensemble members;
2. evaluate once on the locked test;
3. calculate error versus alpha, family, shape distance, and ensemble disagreement;
4. calibrate residual quantiles inside the supported domain;
5. define a trust policy based on observed held-out error.

Call ensemble variance an empirical model-disagreement estimate, not an exact
epistemic distribution.

Benchmark NeuralFoil on the identical held-out geometries/conditions, but do not use
its outputs as labels for your model.

## Stage N — uncertainty model

Generate smooth upper/lower normal displacement fields with B-spline or
Karhunen–Loève modes. Study amplitude and correlation length separately.

Initial stress-test amplitudes:

- 0.1% chord;
- 0.25% chord;
- 0.5% chord;
- 1.0% chord.

Compare:

1. independent percentage perturbation of CST coefficients;
2. independent point noise;
3. smooth correlated surface-normal displacement.

Refit perturbed coordinates to CST for surrogate inference, but run direct solvers on
the actual displaced coordinates. Quantify the aerodynamic effect of CST refitting.

## Stage O — optimisation

Preferred engineering problem:

- choose required lift conditions;
- solve for alpha that attains each required CL;
- minimise expected weighted CD;
- minimise adverse-tail CD CVaR;
- constrain pitching moment;
- constrain maximum thickness, minimum local thickness, area, LE radius, TE thickness;
- constrain geometry validity and calibrated surrogate trust.

Compare:

1. reference geometry;
2. deterministic optimum;
3. naïve CST-perturbation robust optimum;
4. smooth-field robust optimum;
5. trust-calibrated robust Pareto knee.

Debug a scalar problem with differential evolution. Use NSGA-II for the Pareto study.
Reuse common Sobol uncertainty samples across candidates and repeat optimiser seeds.

## Stage P — direct validation

Direct XFOIL:

- first compare ten ordinary source aerofoils with the archived AirfoilTools values;
- document XFOIL version and panel differences;
- rerun reference, deterministic, and robust selected designs;
- use independent perturbation samples;
- compare rankings, means, tails, and constraint violations.

Higher fidelity:

- selected two-dimensional RANS cases with mesh/domain/convergence studies; or
- matched experimental evidence where operating conditions permit.

UIUC wind-tunnel datasets are primarily low-Reynolds-number evidence and should not be
merged indiscriminately into the \(Re=10^6\) training labels.

**Publication gate:** the robust improvement must exceed calibrated surrogate and
sampling uncertainty and remain in the same direction under direct solver evidence.

## Checkpoint prompts

### After setup

```
I am on branch robust-v2 at Git SHA [SHA]. Attached are environment_report.json,
git status, the legacy CSV SHA-256, Python/PyTorch/CUDA versions, and pytest output.
Audit only the setup and give me the next commands. Do not begin tuning.
```

### After legacy recovery

```
Attached are legacy_search.csv, legacy_dataset_audit.json, and inventories of any
trusted raw polar/cache files. Classify this as recovery Path A, B, C, or D. Produce
an exact field mapping and identify what must be downloaded again.
```

### After the 25-airfoil pilot

```
Attached are the 25 raw polar files, ingestion JSONL manifest, mapping report, parser
failures, source metadata summary, and target completeness report. Audit provenance,
source-condition consistency, and parser correctness before authorising 200 records.
```

### After the full data audit

```
Attached are geometry, CST-fit, join, coverage, duplicate, target-completeness, and
metadata reports. Determine the defensible primary operating interval and exact
inclusion/exclusion rules. Do not use the test data to make these decisions.
```

### After baseline training

```
Attached are the frozen split manifest, baseline config, learning curves, per-target
and per-angle metrics, one-batch overfit result, and worst 20 nominal geometries.
Diagnose pipeline/model errors before proposing any hyperparameter search.
```

### After tuning

```
Attached are the Optuna SQLite study/export, grouped-fold results, repeated-seed
results, hardware report, and configs. Select a stable final configuration without
opening the locked test set.
```

### Before optimisation

```
Attached are locked-test metrics, calibration plots, support-distance residual plots,
ensemble disagreement results, trust thresholds, geometry constraint tests, and
uncertainty convergence tests. Audit whether the surrogate is safe enough to enter
the optimiser and identify any remaining go/no-go failure.
```

### After optimisation

```
Attached are repeated-seed Pareto fronts, selected geometries, uncertainty samples,
surrogate risk distributions, trust/constraint reports, and direct XFOIL validation.
Determine which claims are supported, which are not, and the exact RANS cases needed.
```
