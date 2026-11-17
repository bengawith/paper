# MASTER IMPLEMENTATION MANDATE
## Robust Multi-Output Aerofoil Surrogate, Manufacturing-Uncertainty and Optimisation Viability Programme

**Give this entire document to the implementation agent. Do not shorten it.**

---

## 0. ROLE, OPERATING MODE AND NON-NEGOTIABLE INSTRUCTION

You are the principal implementation agent working locally on Ben Gawith's Windows PC in the existing Git repository named `paper`.

You are not being asked merely to propose a plan. You must inspect the repository, preserve its historical state, implement the complete new pipeline, run the viability programme, collect evidence, diagnose failures, apply safe fixes, resume automatically, and leave the repository in a clear, reproducible state that another engineer can continue without oral history.

Operate autonomously. Do not repeatedly ask Ben what to do. Resolve ordinary implementation choices from this mandate, repository evidence, source metadata, tests and measured results. Ask only where a secret, credential, legal permission, or genuinely irreducible research choice is required. If a source or tool fails, follow the fallback hierarchy in this mandate rather than stopping or improvising an undocumented substitute.

You must be exact, conservative and evidence-driven:

1. Do not claim a phase passed until its tests and output contracts pass.
2. Do not modify, delete, overwrite or reinterpret historical raw data or trained artefacts.
3. Do not silently change the scientific question because a convenient implementation is easier.
4. Do not silently substitute NeuralFoil, another neural model or synthetic data for AirfoilTools/XFOIL labels.
5. Do not train on padded or invented aerodynamic values in the primary V2 model.
6. Do not split operating-point rows from the same nominal geometry across train, calibration and test.
7. Do not inspect the locked test set while selecting models, filters, calibration rules, uncertainty models or optimisation settings.
8. Do not hammer AirfoilTools or evade access restrictions.
9. Do not use fuzzy name matching as an automatic geometry identity decision.
10. Do not force-push, rewrite history, delete branches, commit secrets, or push remotely unless Ben separately instructs you to push.
11. Make every phase idempotent and resumable. Interrupted work must continue from recorded state rather than start again.
12. Use atomic writes for manifests and important JSON files.
13. Hash every source file, processed dataset, split, configuration and model artefact used in a reported result.
14. Preserve failed inputs and failure reasons. Never make failure disappear by dropping rows without a ledger.
15. Prefer a correct HOLD decision over a fabricated GO decision.

---

## 1. PROJECT CONTEXT YOU MUST UNDERSTAND

The existing research predicts aerofoil lift curves from a compact Class–Shape Transformation representation.

The submitted V1 study used:

- approximately 1,568 UIUC aerofoil geometries;
- a 12-parameter CST/Kulfan geometry vector:
  - five lower-surface weights;
  - five upper-surface weights;
  - one leading-edge modification parameter;
  - one trailing-edge thickness parameter;
- AirfoilTools XFOIL-derived aerodynamic labels at:
  - Reynolds number `1_000_000`;
  - Mach number `0.0`;
  - `Ncrit = 9`;
- a fixed 48-point angle-of-attack target;
- interpolation within available source coverage;
- edge padding outside source coverage;
- distance-based validity weights;
- an MLP selected through model comparison and physics-oriented evaluation.

The new work is not a cosmetic rerun. It must test whether the existing geometry-to-aerodynamics framework can be extended into a scientifically useful robust-design pipeline by:

1. preserving and auditing the historical V1 result;
2. retaining the full genuine source polar data rather than only 48 sampled lift values;
3. adding drag and moment outputs:
   - `CL`;
   - `CD`;
   - `CDp`;
   - `CM`;
   - upper and lower transition positions where available;
4. using a point-conditioned model:
   \[
   (\text{12 CST parameters}, \alpha)
   \rightarrow
   (C_L,\log C_D,C_M);
   \]
5. estimating empirical model trust and uncertainty;
6. perturbing actual aerofoil surfaces using smooth, correlated manufacturing deviations;
7. comparing:
   - the reference aerofoil;
   - a deterministic optimum;
   - Ava's simple independent-CST-perturbation robust baseline;
   - a smooth correlated-surface robust optimum;
   - a trust-constrained robust optimum;
8. testing both:
   - Ava's original lift-versus-sensitivity objective;
   - a more engineering-useful drag-risk objective at required lift;
9. validating selected results with direct local XFOIL runs;
10. producing an explicit `GO`, `HOLD`, or `NO-GO` viability decision and a complete handover.

The V1 pipeline and V2 pipeline must coexist. Do not edit historical scripts in place merely to make them agree with the paper. Put all new implementation in a clearly isolated V2 namespace and document discrepancies.

---

## 2. THE RESEARCH QUESTION AND VIABILITY QUESTION

### 2.1 Research question

Can a compact 12-parameter CST representation support an angle-conditioned, multi-output surrogate that predicts `CL`, `CD` and `CM` accurately enough within a calibrated pre-stall domain to drive robust aerofoil optimisation under bounded manufacturing geometry uncertainty?

### 2.2 Viability question

Can the complete data, geometry, model, uncertainty, optimisation and direct-solver chain be executed reproducibly on this PC using available public and local evidence, with enough mapped real polar data and enough predictive signal to justify a full study?

### 2.3 Meaning of a GO decision

A viability `GO` means:

- the complete pipeline runs end to end;
- provenance is intact;
- a sufficient number of exact or geometry-confirmed source polars exists;
- the point-conditioned model learns real signal for `CL` and `CD`;
- uncertainty generation produces valid geometry;
- optimisation completes without exploiting unsupported geometry;
- direct XFOIL cross-checks do not overturn the basic candidate ranking;
- no unresolved data leakage or fatal methodological flaw exists.

It does **not** mean the paper's final scientific claim is already proven.

---

## 3. AUTHORITATIVE SOURCE HIERARCHY AND EXACT LINKS

Use sources in this order. Record `source_tier` for every aerodynamic polar and every geometry.

### Tier 0 — Ben's own local historical evidence

Search for:

- original AirfoilTools CSV files;
- cached HTTP responses;
- CSV, Parquet or JSON exports;
- trusted project-created pickle files;
- old repository clones;
- deleted-but-reachable Git objects;
- ZIP archives;
- intermediate datasets;
- logs containing exact polar keys.

Never load an unknown pickle directly. Inspect it with `pickletools` first. If it cannot be safely recovered, quarantine it and continue using the next source tier.

### Tier 1 — Current committed project evidence

Primary repository:

- `https://github.com/bengawith/paper.git`

Historical generator repository:

- `https://github.com/bengawith/curve_gen.git`

Use the committed `paper/data/aerofoil_data` geometries as the **continuity geometry set** for V2. Do not silently replace them with a newer UIUC snapshot.

Inspect at minimum:

- `data/aerofoil_data/`
- `data/csv/dataset_12CST_params.csv`
- `trained_model/`
- `scripts/train_best_model.py`
- `scripts/physics_aware_tuner.py`
- `src/models.py`
- `src/utils.py`
- the complete Git history and reflogs;
- `curve_gen/scripts/full_set_gen.py`.

Known historical issue to verify, not blindly assume: the old full-set generator downloaded a rich AirfoilTools polar but retained only selected `CL` and alpha values, sampling 48 source rows by row index and discarding `CD`, `CDp`, `CM`, `Top_Xtr` and `Bot_Xtr`.

### Tier 2 — Official UIUC geometry source

Landing page:

- `https://m-selig.ae.illinois.edu/ads.html`

Coordinate database:

- `https://m-selig.ae.illinois.edu/ads/coord_database.html`

Current standard-format ZIP archive:

- `https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip`

Individual standard-format directory:

- `https://m-selig.ae.illinois.edu/ads/coord_seligFmt/`

UIUC FAQ:

- `https://m-selig.ae.illinois.edu/ads_faq.html`

Use the current ZIP only as a separately versioned comparison/supplement. Preserve the download SHA-256, HTTP headers and retrieval timestamp. Do not overwrite the continuity set.

The parser must support:

- Selig order: upper trailing edge → leading edge → lower trailing edge;
- Lednicer-style two-block inputs if encountered locally;
- comment lines beginning with `#`;
- finite trailing-edge thickness;
- title lines;
- XFOIL's problem with names beginning with `T` or `F`.

### Tier 3 — Pinned public AirfoilTools-format forensic snapshot

Repository:

- `https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model.git`

Pin this exact observed commit for the initial reproduction:

- `6c0d35b44a33337afae0578402964df314b07698`

Relevant paths:

- `dat/case-dat/`
- `dat/aerofoil-dat/`
- `dat-saved/cases-df.csv`
- `resources/cases.py`
- `LICENSE.md`
- `CITATION.cff`
- `README.md`

This repository contains AirfoilTools-format raw files with metadata and columns such as:

- `Alpha`
- `Cl`
- `Cd`
- `Cdp`
- `Cm`
- `Top_Xtr`
- `Bot_Xtr`

It is a **third-party snapshot**, not the authoritative origin. Use it only with:

- the pinned commit;
- its licence and citation retained;
- exact source metadata parsed;
- geometry confirmation against UIUC/committed shapes;
- deterministic cross-checks against local XFOIL and live AirfoilTools when available;
- `source_tier = "third_party_airfoiltools_snapshot"`.

Never relabel these rows as direct live downloads.

### Tier 4 — Live AirfoilTools probe and limited recovery

Landing page:

- `https://airfoiltools.com/`

Robots file to archive before any retrieval:

- `https://airfoiltools.com/robots.txt`

Canonical polar CSV template:

- `https://airfoiltools.com/polar/csv?polar={POLAR_KEY}`

HTTP historical fallback template:

- `http://airfoiltools.com/polar/csv?polar={POLAR_KEY}`

Polar details template:

- `https://airfoiltools.com/polar/details?polar={POLAR_KEY}`

Canary key:

- `xf-naca2412-il-1000000`

Canary CSV:

- `https://airfoiltools.com/polar/csv?polar=xf-naca2412-il-1000000`

Live access is not guaranteed. One failed canary must not halt the programme. Do not perform bulk live retrieval unless:

- `robots.txt` and any visible terms allow it;
- a canary succeeds;
- content validation succeeds;
- `AIRFOILTOOLS_CONTACT` is set to a real contact address;
- one-request concurrency and the rate policy below are active.

### Tier 5 — Official local XFOIL cross-check and limited fallback

Official XFOIL page:

- `https://web.mit.edu/drela/Public/web/xfoil/`

Official Windows XFOIL 6.99 ZIP:

- `https://web.mit.edu/drela/Public/web/xfoil/XFOIL6.99.zip`

Official current Unix source listed by MIT:

- `https://web.mit.edu/drela/Public/web/xfoil/xfoil6.996.tgz`

Official user guide:

- `https://web.mit.edu/drela/Public/web/xfoil/xfoil_doc.txt`

Official sample sessions:

- `https://web.mit.edu/drela/Public/web/xfoil/sessions.txt`

Use XFOIL for:

- canary validation;
- ordinary-airfoil source cross-checks;
- selected candidate validation;
- limited fallback if no adequate polar archive exists.

Do not regenerate all 1,568 aerofoils with local XFOIL during the viability phase.

### Tier 6 — Benchmark only

NeuralFoil:

- `https://github.com/peterdsharpe/NeuralFoil`
- `https://pypi.org/project/neuralfoil/`

AeroSandbox:

- `https://github.com/peterdsharpe/AeroSandbox`
- `https://aerosandbox.readthedocs.io/en/master/autoapi/aerosandbox/geometry/airfoil/airfoil_families/`

NeuralFoil may be used only as:

- an external comparator;
- a ranking benchmark;
- a confidence/trust comparison;
- a diagnostic for candidate behaviour.

Never use NeuralFoil outputs as V2 training labels.

### Tooling and library documentation

`uv`:

- `https://docs.astral.sh/uv/`
- `https://docs.astral.sh/uv/getting-started/installation/`

PyTorch selector:

- `https://pytorch.org/get-started/locally/`

Optuna:

- `https://optuna.readthedocs.io/en/stable/`
- `https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html`
- `https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html`

SciPy Sobol QMC:

- `https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html`

SALib:

- `https://salib.readthedocs.io/en/latest/`

pymoo NSGA-II:

- `https://pymoo.org/algorithms/moo/nsga2.html`

---

## 4. REPOSITORY SAFETY AND BRANCHING

Begin inside the existing `paper` directory.

### 4.1 Establish repository root

Run and record:

```powershell
git rev-parse --show-toplevel
git remote -v
git branch --show-current
git rev-parse HEAD
git status --porcelain=v1
git log --oneline --decorate -15
```

Abort only if this is not a Git repository or is clearly not `bengawith/paper`.

### 4.2 Preserve current work

Create:

```text
reports/robust_v2/setup/
```

Write:

- `starting_git_sha.txt`
- `starting_branch.txt`
- `starting_status.txt`
- `starting_log.txt`
- `starting_remotes.txt`
- `starting_diff.patch`
- `starting_untracked_files.txt`

If the working tree is dirty:

- do not discard, reset or overwrite the user's changes;
- create the V2 branch from the current HEAD;
- constrain your changes to the new V2 namespace;
- if a required shared file already has uncommitted changes, preserve it and make the minimum non-destructive merge;
- record any overlap in `reports/robust_v2/setup/preexisting_change_conflicts.md`.

Create or switch to:

```text
robust-v2-viability
```

Use:

```powershell
git switch -c robust-v2-viability
```

or, if it exists:

```powershell
git switch robust-v2-viability
```

Create an annotated baseline tag only if it does not already exist:

```text
baseline-pre-robust-v2-20260824
```

Do **not** call it the exact submitted state unless the repository evidence proves that.

### 4.3 Hash historical artefacts

Hash at minimum:

- `data/csv/dataset_12CST_params.csv`
- every file in `trained_model/`
- `scripts/train_best_model.py`
- `src/models.py`
- `src/utils.py`

Write:

```text
reports/robust_v2/setup/historical_hashes.csv
```

Fields:

```text
relative_path
size_bytes
sha256
git_tracked
git_blob_sha
modified_utc
```

### 4.4 Commit policy

Make local milestone commits after phases pass. Suggested messages:

```text
robust-v2: add reproducible project scaffold
robust-v2: add provenance-safe source ingestion
robust-v2: add geometry and CST processing
robust-v2: add grouped data splits and guards
robust-v2: add conditioned multi-output baseline
robust-v2: add uncertainty and optimisation smoke tests
robust-v2: record viability evidence
```

Do not push automatically.

---

## 5. REQUIRED V2 REPOSITORY LAYOUT

Create this layout without moving or renaming historical V1 files:

```text
paper/
├── pyproject.toml
├── uv.lock
├── .env.example
├── .github/
│   └── workflows/
│       └── robust-v2-ci.yml
│
├── configs/
│   └── robust_v2/
│       ├── sources.yaml
│       ├── viability.yaml
│       ├── model_baseline.yaml
│       ├── tuning.yaml
│       ├── uncertainty.yaml
│       └── optimisation.yaml
│
├── src/
│   └── robust_airfoil/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── constants.py
│       ├── hashing.py
│       ├── logging_utils.py
│       ├── provenance.py
│       ├── run_state.py
│       ├── recovery.py
│       │
│       ├── sources/
│       │   ├── __init__.py
│       │   ├── uiuc.py
│       │   ├── airfoiltools.py
│       │   ├── snapshot.py
│       │   └── xfoil.py
│       │
│       ├── geometry/
│       │   ├── __init__.py
│       │   ├── parser.py
│       │   ├── normalise.py
│       │   ├── cst.py
│       │   ├── metrics.py
│       │   ├── validity.py
│       │   └── clustering.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── legacy.py
│       │   ├── mapping.py
│       │   ├── polar_parser.py
│       │   ├── quality.py
│       │   ├── builder.py
│       │   ├── schemas.py
│       │   └── splits.py
│       │
│       ├── modelling/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── samplers.py
│       │   ├── models.py
│       │   ├── losses.py
│       │   ├── baselines.py
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   ├── tune.py
│       │   └── calibrate.py
│       │
│       ├── benchmarks/
│       │   ├── __init__.py
│       │   └── neuralfoil.py
│       │
│       ├── uncertainty/
│       │   ├── __init__.py
│       │   ├── fields.py
│       │   ├── sampling.py
│       │   └── propagation.py
│       │
│       ├── optimisation/
│       │   ├── __init__.py
│       │   ├── objectives.py
│       │   ├── constraints.py
│       │   ├── problems.py
│       │   └── runner.py
│       │
│       ├── validation/
│       │   ├── __init__.py
│       │   ├── xfoil_runner.py
│       │   └── ranking.py
│       │
│       └── reports/
│           ├── __init__.py
│           ├── figures.py
│           └── viability.py
│
├── scripts/
│   └── robust_v2/
│       ├── bootstrap.ps1
│       ├── run_viability.ps1
│       ├── run_viability.py
│       ├── audit_legacy.py
│       ├── discover_legacy_data.py
│       ├── acquire_sources.py
│       ├── build_dataset.py
│       ├── train_baseline.py
│       ├── tune_model.py
│       ├── benchmark_neuralfoil.py
│       ├── validate_xfoil.py
│       ├── run_uncertainty.py
│       ├── run_optimisation.py
│       └── build_report.py
│
├── data/
│   └── robust_v2/
│       ├── raw/
│       │   ├── recovered/
│       │   ├── uiuc_current/
│       │   ├── airfoiltools_live/
│       │   ├── airfoiltools_snapshot/
│       │   └── xfoil_validation/
│       ├── interim/
│       │   ├── normalised_coordinates/
│       │   ├── cst_fits/
│       │   └── mappings/
│       ├── processed/
│       ├── manifests/
│       ├── rejected/
│       └── splits/
│
├── results/
│   └── robust_v2/
│       ├── legacy/
│       ├── pilot/
│       ├── expansion_200/
│       ├── full_baseline/
│       ├── tuning/
│       ├── calibration/
│       ├── uncertainty/
│       ├── optimisation/
│       └── validation/
│
├── reports/
│   └── robust_v2/
│       ├── setup/
│       ├── data/
│       ├── modelling/
│       ├── viability/
│       └── handover/
│
├── tests/
│   └── robust_v2/
│       ├── fixtures/
│       ├── test_hashing.py
│       ├── test_run_state.py
│       ├── test_recovery.py
│       ├── test_airfoiltools_parser.py
│       ├── test_airfoiltools_client.py
│       ├── test_snapshot.py
│       ├── test_uiuc_parser.py
│       ├── test_geometry_normalise.py
│       ├── test_cst_roundtrip.py
│       ├── test_geometry_validity.py
│       ├── test_mapping.py
│       ├── test_dataset_builder.py
│       ├── test_splits.py
│       ├── test_model.py
│       ├── test_losses.py
│       ├── test_sampler.py
│       ├── test_training_smoke.py
│       ├── test_locked_test_guard.py
│       ├── test_neuralfoil_adapter.py
│       ├── test_xfoil_parser.py
│       ├── test_uncertainty_fields.py
│       ├── test_risk_metrics.py
│       ├── test_optimisation_smoke.py
│       └── test_viability_report.py
│
└── tools/
    └── xfoil/
```

Add generated/raw/model-heavy paths to `.gitignore`, while committing:

- configuration;
- source code;
- tests;
- small fixtures;
- manifests;
- reports;
- compact metrics;
- environment lockfile;
- source licences/citations.

Do not commit third-party nested repositories or large raw downloads.

---

## 6. ENVIRONMENT SETUP

### 6.1 Use Python 3.11

Use Python 3.11 for V2 compatibility. Do not mutate a historical V1 virtual environment.

Install `uv` if absent:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then:

```powershell
uv python install 3.11
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
```

### 6.2 Project dependencies

Create or safely extend `pyproject.toml`. Do not discard existing metadata.

Runtime dependencies:

```text
numpy
pandas
pyarrow
scipy
scikit-learn
matplotlib
pydantic
pydantic-settings
pyyaml
httpx
tenacity
beautifulsoup4
lxml
aerosandbox
neuralfoil
optuna
sqlalchemy
pymoo
SALib
joblib
tqdm
rich
typer
platformdirs
psutil
shapely
```

Development dependencies:

```text
pytest
pytest-cov
hypothesis
ruff
mypy
pandas-stubs
types-PyYAML
respx
```

Use `uv add` and `uv add --dev`, then commit `uv.lock`.

### 6.3 PyTorch

First inspect:

```powershell
nvidia-smi
```

Use the current official selector at:

```text
https://pytorch.org/get-started/locally/
```

For the known GTX 1660 SUPER environment, CUDA 11.8 is a conservative compatible option if it remains supported. The candidate command is:

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```

But verify the current official selector before using it. If GPU installation fails, use the official CPU wheel and continue:

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The viability programme must run on CPU if necessary. Record whether GPU acceleration is available; do not falsify it.

Verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 6.4 Environment report

Implement an environment-report function that writes:

```text
reports/robust_v2/setup/environment.json
reports/robust_v2/setup/environment.txt
reports/robust_v2/setup/package_freeze.txt
```

Include:

- OS and build;
- CPU;
- RAM;
- GPU and driver;
- Python;
- `uv`;
- PyTorch;
- CUDA;
- all dependency versions;
- Git;
- repository SHA;
- active branch;
- hostname;
- UTC timestamp.

Do not store usernames, tokens or unnecessary personal paths in committed reports; redact home-directory prefixes.

---

## 7. CONFIGURATION CONTRACTS

Create the following YAML files and validate them with Pydantic models. Unknown keys must raise an error.

### 7.1 `configs/robust_v2/sources.yaml`

```yaml
project_seed: 20260824

conditions:
  reynolds_number: 1000000
  mach: 0.0
  ncrit: 9.0

continuity_geometry_dir: data/aerofoil_data
legacy_dataset: data/csv/dataset_12CST_params.csv

uiuc:
  landing_url: https://m-selig.ae.illinois.edu/ads.html
  database_url: https://m-selig.ae.illinois.edu/ads/coord_database.html
  zip_url: https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip
  individual_dir_url: https://m-selig.ae.illinois.edu/ads/coord_seligFmt/
  use_current_as_replacement: false

snapshot:
  repo_url: https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model.git
  commit: 6c0d35b44a33337afae0578402964df314b07698
  case_glob: dat/case-dat/*-il-1000000.csv
  geometry_dir: dat/aerofoil-dat
  source_tier: third_party_airfoiltools_snapshot

airfoiltools_live:
  enabled: true
  robots_url: https://airfoiltools.com/robots.txt
  csv_url_template: https://airfoiltools.com/polar/csv?polar={polar_key}
  http_fallback_template: http://airfoiltools.com/polar/csv?polar={polar_key}
  details_url_template: https://airfoiltools.com/polar/details?polar={polar_key}
  canary_key: xf-naca2412-il-1000000
  concurrency: 1
  minimum_delay_seconds: 2.0
  connect_timeout_seconds: 15
  read_timeout_seconds: 30
  max_transient_attempts: 3
  max_live_pilot_records: 25
  stop_on_status: [401, 403, 429]
  require_contact_env: AIRFOILTOOLS_CONTACT
  permit_bulk_without_successful_canary: false

xfoil:
  official_page: https://web.mit.edu/drela/Public/web/xfoil/
  windows_zip_url: https://web.mit.edu/drela/Public/web/xfoil/XFOIL6.99.zip
  documentation_url: https://web.mit.edu/drela/Public/web/xfoil/xfoil_doc.txt
  sessions_url: https://web.mit.edu/drela/Public/web/xfoil/sessions.txt
  executable_env: XFOIL_EXE
  timeout_seconds: 240
  iterations: 70
```

### 7.2 `configs/robust_v2/viability.yaml`

```yaml
seed: 20260824

pilot:
  target_airfoils: 25
  minimum_mapped_airfoils: 20
  minimum_joint_points: 500
  minimum_parse_success_fraction: 0.95
  minimum_mapping_fraction: 0.80
  split_counts:
    development: 15
    calibration: 5
    locked_test: 5

expansion:
  target_airfoils: 200
  start_automatically_after_go: true

full:
  build_all_available_exact_or_geometry_confirmed: true
  start_baseline_after_expansion_pass: true
  start_tuning_trials_after_baseline_pass: 50

geometry:
  cosine_points_per_surface: 201
  exact_hash_decimals: 8
  pilot_near_duplicate_rms_threshold: 0.0001
  minimum_local_thickness: 0.0

model:
  require_one_batch_overfit: true
  one_batch_relative_loss_reduction: 0.99
  require_cl_better_than_dummy_fraction: 0.20
  require_log_cd_better_than_dummy_fraction: 0.20
  cm_improvement_is_desirable_not_initially_fatal: true

live_network_tests: false
xfoil_tests: false
slow_tests: false
```

### 7.3 `configs/robust_v2/model_baseline.yaml`

```yaml
seed: 20260824

features:
  geometry:
    - lower_weight_0
    - lower_weight_1
    - lower_weight_2
    - lower_weight_3
    - lower_weight_4
    - upper_weight_0
    - upper_weight_1
    - upper_weight_2
    - upper_weight_3
    - upper_weight_4
    - leading_edge_weight
    - TE_thickness
  operating:
    - alpha_deg

targets:
  - cl
  - log_cd
  - cm

architecture:
  type: conditioned_multi_head_mlp
  hidden_width: 256
  hidden_layers: 3
  activation: silu
  dropout: 0.10
  layer_norm: true
  residual: true

training:
  optimiser: adamw
  learning_rate: 0.001
  weight_decay: 0.00001
  batch_size: 1024
  maximum_epochs: 300
  early_stopping_patience: 25
  gradient_clip_norm: 1.0
  loss: huber
  huber_delta_standardised: 1.0
  deterministic_warn_only: true
  mixed_precision: false
  equalise_nominal_airfoil_weight: true
```

### 7.4 `configs/robust_v2/tuning.yaml`

```yaml
seed: 20260824
storage: sqlite:///results/robust_v2/tuning/conditioned_polar_v2.sqlite3
study_name: conditioned_polar_v2
direction: minimize
n_jobs: 1
grouped_folds: 3
pilot_trials: 20
automatic_initial_full_trials: 50
pruner:
  type: median
  startup_trials: 10
  warmup_epochs: 20
  interval_epochs: 5

search:
  hidden_layers: [2, 6]
  hidden_width_choices: [128, 192, 256, 384, 512]
  activation_choices: [relu, gelu, silu]
  dropout: [0.0, 0.30]
  learning_rate_log: [0.00001, 0.003]
  weight_decay_log: [0.00000001, 0.001]
  batch_size_choices: [512, 1024, 2048, 4096]
  loss_choices: [huber, logcosh]
  residual_choices: [false, true]
```

### 7.5 `configs/robust_v2/uncertainty.yaml`

```yaml
seed: 20260824
representation: smooth_correlated_surface_normal
cosine_points_per_surface: 201
basis_control_points_per_surface: 8
correlation_length_chord: 0.15
upper_lower_correlation: 0.0
leading_edge_taper: true
trailing_edge_zero_displacement: true
amplitudes_fraction_chord: [0.001, 0.0025, 0.005, 0.01]

optimisation_samples:
  method: scrambled_sobol
  count: 64
  common_random_numbers: true
  antithetic: true

final_evaluation_samples:
  method: scrambled_sobol
  count: 2048

convergence_counts: [16, 32, 64, 128, 256, 512, 1024, 2048]

ablations:
  - independent_multiplicative_cst
  - independent_coordinate_noise
  - smooth_correlated_surface_normal
```

### 7.6 `configs/robust_v2/optimisation.yaml`

```yaml
seed: 20260824
reference_preference: naca2412

conditions:
  reynolds_number: 1000000
  mach: 0.0
  ncrit: 9.0

service_targets:
  cl_required: [0.4, 0.7, 1.0]
  weights: [0.25, 0.50, 0.25]

ava_baseline:
  alpha_deg: [0.0, 2.0, 4.0, 6.0, 8.0]
  uncertainty_fraction: 0.01
  lambda_sensitivity_values: [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

risk_objectives:
  - expected_weighted_cd
  - cvar_95_weighted_cd

relative_constraints:
  thickness_ratio_lower: 0.98
  thickness_ratio_upper: 1.02
  section_area_lower: 0.98
  leading_edge_radius_lower: 0.80
  cm_allowance_below_reference: 0.01

trust:
  require_calibrated_domain: true
  reject_untrusted_candidates: true

debug_nsga2:
  population: 32
  generations: 10
  uncertainty_samples: 16

pilot_nsga2:
  population: 64
  generations: 50
  uncertainty_samples: 64

full_nsga2:
  population: 128
  generations: 200
  uncertainty_samples: 64
  independent_seeds: 10
```

---

## 8. RUN STATE, LOGGING AND RESUMABILITY

Implement `RunState` in `src/robust_airfoil/run_state.py`.

State file:

```text
reports/robust_v2/run_state.json
```

Each phase entry must contain:

```text
phase_id
name
status: pending | running | passed | held | failed
started_utc
finished_utc
git_sha
config_hashes
input_hashes
output_hashes
command
log_path
summary
failure_type
failure_message
retryable
```

Rules:

- write state atomically via temp file + replace;
- a phase may be skipped only if:
  - status is `passed`;
  - all declared outputs exist;
  - all output hashes still match;
  - relevant configuration and input hashes still match;
- otherwise rerun it;
- append structured JSON logs to:
  ```text
  reports/robust_v2/logs/{phase_id}.jsonl
  ```
- also write readable `.log` files;
- never suppress stack traces from unexpected failures;
- classify failures as:
  - environment;
  - source_unavailable;
  - source_forbidden;
  - parse;
  - mapping;
  - geometry;
  - model;
  - solver;
  - optimisation;
  - evidence_contract.

Implement the single resumable entry point:

```powershell
uv run python -m robust_airfoil run --profile viability --resume
```

and the PowerShell wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/robust_v2/run_viability.ps1
```

---

## 9. PHASED EXECUTION STATE MACHINE

Implement and execute these phases in order.

### Phase 00 — Preflight and repository snapshot

Outputs:

```text
reports/robust_v2/setup/*
reports/robust_v2/run_state.json
```

Pass criteria:

- repository root confirmed;
- current changes preserved;
- V2 branch active;
- historical hashes recorded.

### Phase 01 — Environment and scaffold

Actions:

- create V2 tree;
- install/lock dependencies;
- detect PyTorch device;
- create configs;
- create test markers;
- run import smoke tests.

Pass criteria:

```powershell
uv run python -c "import robust_airfoil"
uv run pytest tests/robust_v2/test_hashing.py tests/robust_v2/test_run_state.py -q
uv run ruff check src/robust_airfoil tests/robust_v2
```

### Phase 02 — Historical V1 audit and best-effort reproduction

Do not modify V1 scripts or artefacts.

Implement:

- `scripts/robust_v2/audit_legacy.py`
- `src/robust_airfoil/data/legacy.py`

Audit:

- exact dataset shape;
- exact feature columns;
- exact CL columns;
- exact validity/distance columns;
- row count;
- duplicate count;
- missing values;
- angle-grid evidence;
- scaler contents;
- model config;
- saved metric files;
- random split logic;
- LOFO claims versus code;
- early-stopping implementation;
- utility/model signature inconsistencies;
- model input/output shape;
- historical environment evidence.

Attempt to load and evaluate the committed model. Reconstruct the historical 70/15/15 seed-42 split exactly where supported by code. Calculate:

- full pointwise MSE and R²;
- valid-only MSE and R²;
- per-airfoil metrics;
- physical metrics with corrected, separately reported units;
- discrepancy versus stored metrics.

Write:

```text
results/robust_v2/legacy/legacy_audit.json
results/robust_v2/legacy/reproduction_metrics.json
reports/robust_v2/modelling/V1_REPRODUCTION_REPORT.md
reports/robust_v2/modelling/V1_CODE_PAPER_DISCREPANCIES.md
```

A reproduction discrepancy does not block V2 if it is fully traced and documented. Unexplained silent divergence blocks any claim of exact V1 reproduction.

### Phase 03 — Local historical data discovery

Implement:

```powershell
uv run python scripts/robust_v2/discover_legacy_data.py --auto-roots
```

Search, in a bounded way:

- repository and parent;
- `Documents`;
- `Downloads`;
- `Desktop`;
- `OneDrive`;
- existing `D:\` and `E:\` roots if present;
- old Git worktrees/clones.

Extensions:

```text
.csv .txt .json .jsonl .parquet .feather .pkl .pickle .joblib .zip .7z
```

Content signatures:

```text
Polar key
Reynolds number
Ncrit
Max Cl/Cd
Alpha,Cl,Cd,Cdp,Cm,Top_Xtr,Bot_Xtr
airfoiltools.com/polar
xf-
Top_Xtr
Bot_Xtr
```

For every candidate, record:

```text
path_redacted
size
sha256
extension
signature_matches
likely_source
safe_to_parse
priority
```

Do not unpickle automatically. Use `pickletools.genops()` to list referenced globals. If globals are not from a strict whitelist, quarantine.

Outputs:

```text
data/robust_v2/manifests/recovery_candidates.csv
reports/robust_v2/data/LOCAL_RECOVERY_REPORT.md
```

Copy accepted raw files into `data/robust_v2/raw/recovered/` without altering originals. Use content-addressed names and manifests.

### Phase 04 — Source acquisition and inventory

#### 04A. UIUC current snapshot

Download:

```text
https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip
```

Save raw ZIP, headers and SHA-256. Extract safely:

- reject absolute paths;
- reject `..` path traversal;
- retain all `.dat` files;
- do not replace `data/aerofoil_data`.

Produce a comparison:

```text
committed_only
current_only
same_name_same_hash
same_name_changed_hash
```

#### 04B. Pinned AirfoilTools-format snapshot

Clone to ignored cache:

```powershell
git clone https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model.git data/robust_v2/raw/airfoiltools_snapshot/repo
git -C data/robust_v2/raw/airfoiltools_snapshot/repo checkout 6c0d35b44a33337afae0578402964df314b07698
```

Verify exact HEAD. Copy licence/citation into the source-evidence report. Inventory:

```text
dat/case-dat/*-il-1000000.csv
```

Do not parse other Reynolds numbers into the primary V2 dataset.

#### 04C. Live canary only

Before any live CSV:

1. fetch and archive `robots.txt`;
2. require `AIRFOILTOOLS_CONTACT`;
3. request only the NACA 2412 canary;
4. validate body before saving as a polar.

If `AIRFOILTOOLS_CONTACT` is absent, skip live access and record `not_attempted_missing_contact`; do not stop.

If the canary returns:

- `401`, `403`, or `429`: stop all live access;
- `404`: mark unavailable;
- `5xx` or network timeout: retry at most three times with exponential delays, then use fallback;
- HTML, CAPTCHA or a non-polar body: stop;
- valid CSV: retain it and allow at most 25 pilot requests.

Do not launch a full live scrape in this programme.

#### 04D. XFOIL acquisition

Locate in order:

1. `XFOIL_EXE`;
2. `tools/xfoil/**/xfoil.exe`;
3. `PATH`;
4. official MIT Windows ZIP.

If downloading, save ZIP SHA-256 and source manifest. Extract to an ignored versioned directory.

Run `xfoil.exe` with a trivial NACA 0012 session to prove it launches before calling the source available.

Outputs:

```text
data/robust_v2/manifests/source_files.jsonl
data/robust_v2/manifests/source_inventory.json
reports/robust_v2/data/SOURCE_ACQUISITION_REPORT.md
```

### Phase 05 — AirfoilTools polar parser and immutable raw layer

Implement a robust parser that:

- detects the `Alpha` header dynamically;
- accepts commas with surrounding whitespace;
- parses metadata by key rather than fixed line numbers;
- handles UTF-8 with BOM and common Windows encodings;
- treats blank, `*`, `nan`, `NaN` and malformed numeric fields as missing;
- never interpolates;
- never pads;
- preserves source row order;
- validates expected condition metadata;
- records unknown XFOIL version as null, not guessed;
- records every rejection reason.

Canonical aerodynamic columns:

```text
alpha_deg
cl
cd
cdp
cm
top_xtr
bot_xtr
```

Canonical metadata:

```text
polar_key
source_airfoil_name
reynolds_number
mach
ncrit
max_cl_cd
max_cl_cd_alpha
original_url
xfoil_version
```

Expected conditions for primary inclusion:

```text
reynolds_number == 1_000_000
mach == 0.0
ncrit == 9.0
```

Use numeric tolerances only for floating representation, not to accept another physical condition.

Duplicate-angle rules:

- exact duplicate numeric rows: retain one eligible row and flag duplicate provenance;
- conflicting rows at the same alpha: retain all raw rows, mark the alpha unresolved, exclude it from primary modelling until resolved;
- never average conflicting source rows silently.

Create a synthetic fixture and a small attributed format fixture. Do not copy an entire third-party polar into tests unnecessarily.

### Phase 06 — Geometry parsing, normalisation and CST

#### 06.1 Coordinate parsing

Support:

- title line;
- comments beginning `#`;
- arbitrary whitespace;
- commas only where safely detectable;
- Selig single-loop order;
- Lednicer two-block order;
- repeated LE or TE points;
- finite-thickness TE;
- reversed loops;
- non-unit chord.

#### 06.2 Normalisation

For each geometry:

1. remove only exact consecutive duplicate coordinates initially;
2. identify LE and upper/lower TE points;
3. derive chord line;
4. translate LE to `(0,0)`;
5. rotate chord line to the x-axis;
6. scale mean TE x-coordinate to 1;
7. orient as upper TE → LE → lower TE;
8. split surfaces;
9. interpolate each surface independently on 201 cosine-spaced x/c values;
10. preserve an immutable copy of original coordinates;
11. hash both raw and normalised geometry.

#### 06.3 Validity checks

Calculate:

- finite coordinates;
- monotonic x on each split surface after orientation;
- self-intersection;
- surface crossing;
- minimum local thickness;
- maximum thickness and location;
- camber and location;
- section area;
- trailing-edge thickness;
- approximate leading-edge radius;
- curvature anomalies;
- point-spacing anomalies.

Do not automatically "smooth away" defects. Quarantine invalid shapes with reasons.

#### 06.4 CST extraction

Use five weights per surface plus:

- `leading_edge_weight`;
- `TE_thickness`.

Canonical feature order is fixed and must be stored with each model.

Use AeroSandbox's maintained Kulfan extraction where compatible. Wrap it so library API changes are detected by tests. Reconstruct coordinates from fitted parameters and calculate:

- RMS ordinate error;
- maximum absolute ordinate error;
- thickness error;
- camber error;
- area error.

Do not set a permanent CST rejection threshold before generating and inspecting the full error distribution. For pilot eligibility, require finite parameters and no geometry-invalid flag; report reconstruction outliers separately.

Important historical check: verify whether old code used six weights per side while the final dataset uses five. Do not inherit a historical parameter-count inconsistency into V2.

Outputs:

```text
data/robust_v2/processed/airfoils.parquet
data/robust_v2/interim/normalised_coordinates/
data/robust_v2/interim/cst_fits/
reports/robust_v2/data/GEOMETRY_AND_CST_REPORT.md
```

### Phase 07 — Geometry-to-polar mapping

Create:

```text
data/robust_v2/manifests/airfoil_mapping.csv
```

Fields:

```text
airfoil_id
continuity_filename
continuity_stem
legacy_dataset_name
canonical_name
snapshot_polar_filename
source_airfoil_name
polar_key
snapshot_geometry_filename
mapping_method
mapping_status
name_rule
geometry_rms
raw_geometry_sha256
normalised_geometry_sha256
notes
```

Mapping order:

1. recovered exact historical polar key;
2. exact canonical name;
3. explicit deterministic alias rule;
4. exact normalised geometry hash;
5. geometry RMS confirmation;
6. manual/ambiguous quarantine.

Allowed automatic statuses:

```text
exact_historical_key
exact_name
explicit_alias_verified
geometry_exact
geometry_confirmed
```

Non-eligible statuses:

```text
ambiguous
unmatched
geometry_mismatch
missing_geometry
missing_polar
condition_mismatch
invalid_geometry
```

Explicit aliases may include exact deterministic patterns such as:

- UIUC `n0012` ↔ source `naca0012`;
- punctuation, spaces and underscore normalisation;
- exact regex-confirmed NACA family names.

A fuzzy string similarity score may rank candidates for review but must never by itself produce an accepted mapping.

For a geometry confirmation:

- normalise both committed/UIUC and snapshot geometry to the same grid;
- require matching orientation;
- store RMS and max error;
- use a strict threshold;
- produce comparison plots for borderline cases.

### Phase 08 — Build long-form datasets

Create four connected Parquet tables.

#### `airfoils.parquet`

One row per nominal geometry:

```text
airfoil_id
canonical_name
continuity_filename
family_label
geometry_cluster_id
raw_geometry_path
raw_geometry_sha256
normalised_geometry_sha256
geometry_valid
geometry_failure_reason
lower_weight_0 ... lower_weight_4
upper_weight_0 ... upper_weight_4
leading_edge_weight
TE_thickness
cst_roundtrip_rmse
cst_roundtrip_max_abs_error
thickness_ratio
max_thickness_x
max_camber_ratio
max_camber_x
section_area
leading_edge_radius
trailing_edge_thickness
```

#### `polars.parquet`

One row per source polar:

```text
polar_id
airfoil_id
source_tier
source_repository
source_commit
polar_key
original_url
raw_path
raw_sha256
retrieved_utc
reynolds_number
mach
ncrit
xfoil_version
point_count
alpha_min_deg
alpha_max_deg
metadata_match
parse_status
```

#### `polar_points.parquet`

One row per genuine source row:

```text
polar_id
airfoil_id
source_row_index
alpha_deg
cl
cd
cdp
cm
top_xtr
bot_xtr
finite_cl
finite_cd
finite_cm
duplicate_alpha
conflicting_duplicate
quality_flags
joint_model_eligible
```

#### `model_points.parquet`

One row per model input/target point:

```text
airfoil_id
geometry_cluster_id
family_label
split
polar_id
lower_weight_0 ... lower_weight_4
upper_weight_0 ... upper_weight_4
leading_edge_weight
TE_thickness
alpha_deg
cl
log_cd
cm
cl_available
cd_available
cm_available
point_weight
source_tier
dataset_version
```

Rules:

- `log_cd = log(cd)` only where `cd > 0`;
- do not add an epsilon to invalid drag;
- preserve partial targets with masks;
- create `model_points_joint.parquet` for rows with all three primary targets;
- create a strict contiguous-curve evaluation subset separately;
- do not create a padded 48-point V2 training target;
- do not use row count as sample weight.

Equalise nominal-airfoil contribution:

\[
w_{i,k} = 1/n_i
\]

within each target-availability set, or use an equivalent two-stage sampler.

Generate:

```text
reports/robust_v2/data/alpha_coverage.csv
reports/robust_v2/data/target_completeness.csv
reports/robust_v2/data/accepted_rejected_ledger.csv
reports/robust_v2/data/DATASET_AUDIT.md
```

### Phase 09 — Geometry clustering and leakage-resistant splits

Build fixed geometry vectors from 201 upper and 201 lower ordinates.

1. Hash coordinates rounded to eight decimals for exact duplicates.
2. Use nearest-neighbour RMS to find near duplicates.
3. Use union-find to create connected geometry clusters.
4. For the pilot, use the configured conservative threshold and report sensitivity.
5. For full data, inspect the nearest-distance distribution and at least the 100 closest non-identical pairs before freezing the threshold.
6. Assign complete clusters to one split.
7. Every future perturbed realisation inherits the nominal geometry split.

Pilot split:

```text
15 development clusters
5 calibration clusters
5 locked-test clusters
```

If exactly 25 cannot be obtained due clustering, preserve the 60/20/20 intent while keeping at least four locked-test clusters.

Full split target:

```text
70% development
15% calibration
15% locked test
```

Balance approximately across:

- family;
- thickness bins;
- camber bins;
- source coverage bins.

Write:

```text
data/robust_v2/splits/development.csv
data/robust_v2/splits/calibration.csv
data/robust_v2/splits/locked_test.csv
data/robust_v2/splits/split_manifest.json
```

Implement a locked-test guard:

- training and tuning code must refuse to load `locked_test`;
- evaluation code must refuse to unlock it until:
  ```text
  results/robust_v2/FROZEN_MODEL_MANIFEST.json
  ```
  exists;
- the manifest must hash:
  - dataset;
  - splits;
  - model config;
  - scalers;
  - calibration rule;
  - trust rule;
  - selected epoch;
  - Git SHA.

### Phase 10 — Deterministic 25-airfoil pilot selection

Select from the exact/verified intersection, not by first 25 filenames.

Use seed `20260824`.

Target diversity:

- NACA and near-symmetric profiles;
- Selig/AG low-Re families;
- Eppler;
- Wortmann/FX;
- Göttingen/historical;
- thick and thin extremes;
- camber extremes;
- median conventional shapes.

Selection algorithm:

1. identify eligible mapped airfoils;
2. reserve family-diverse candidates;
3. reserve geometry extremes based on thickness/camber/area;
4. fill remaining slots with seeded random selection;
5. prevent near-duplicate clusters;
6. write the rationale for each selected aerofoil.

Pass source gate when:

- at least 20 mapped valid airfoils exist;
- parse success is at least 95% of selected source files;
- mapping acceptance is at least 80%;
- at least 500 joint `CL/CD/CM` points exist;
- 100% of accepted rows match the declared condition metadata.

If the pilot is too small, expand deterministic selection up to 50 available airfoils before issuing HOLD.

### Phase 11 — Conditioned multi-output model

Implement `ConditionedPolarMLP`.

Input:

```text
12 CST features + alpha_deg = 13 features
```

Output heads:

```text
cl
log_cd
cm
```

Architecture:

- shared trunk;
- three hidden layers;
- width 256;
- SiLU;
- LayerNorm;
- dropout 0.10;
- residual connections where dimensions match;
- separate linear heads.

Preprocessing:

- fit feature scalers only on development rows;
- fit target scalers only on development rows;
- store exact feature order;
- store target transform metadata;
- standardise alpha as an ordinary continuous input;
- do not fit anything on calibration or locked test.

Loss:

- target-masked Huber or log-cosh in standardised units;
- macro/equal-airfoil point weighting;
- average target losses so scale does not allow one output to dominate.

Training:

- AdamW;
- learning rate `1e-3`;
- weight decay `1e-5`;
- batch size up to 1024, reduced if required;
- maximum 300 epochs;
- early-stopping patience 25;
- gradient clipping 1.0;
- seed all libraries;
- deterministic algorithms with `warn_only=True`;
- no mixed precision in initial viability;
- best checkpoint selected on a development-validation group split, not calibration or test.

Checkpoint contract:

```text
model_state
optimiser_state
epoch
training_history
model_config
feature_columns
target_columns
feature_scaler
target_scaler
dataset_hash
split_hash
config_hash
git_sha
seed
device
```

### Phase 12 — Mandatory training ladder

Do not jump to a full training run.

#### 12A. Forward-shape test

Assert:

```text
input:  (batch, 13)
output: dict or tensor equivalent to (batch, 3)
```

#### 12B. One-batch overfit

Use a small fixed batch and train until:

- relative loss reduction is at least 99%; or
- a documented stricter absolute standardised loss is reached.

If it fails, debug alignment, scaling, masks, sampler, loss and model before proceeding.

#### 12C. 1,000-row smoke

Verify:

- finite loss;
- finite gradients;
- checkpoint write/read;
- deterministic split;
- inverse transforms;
- CPU path;
- CUDA path if available.

#### 12D. 25-airfoil pilot

Train:

- Dummy mean baseline;
- Ridge baseline;
- HistGradientBoosting baseline per target;
- conditioned MLP.

Evaluate on calibration for development decisions. Keep locked test closed.

Required initial signal:

- MLP `CL` macro error improves over Dummy by at least 20%;
- MLP `log(CD)` macro error improves over Dummy by at least 20%;
- no NaNs or collapsed constant output;
- `CM` improvement is desirable but a weak `CM` pilot is not alone fatal;
- errors and plots are physically interpretable.

If the 25-airfoil result is statistically inconclusive but the pipeline is sound, automatically continue to the 200-airfoil expansion before deciding HOLD.

### Phase 13 — Evaluation metrics

Report both micro and per-airfoil macro metrics.

For `CL`:

- MAE;
- RMSE;
- R²;
- error versus alpha;
- lift-curve slope error in supported linear ranges;
- maximum-lift error only on strict curve subsets.

For `CD`:

- `log(CD)` MAE and RMSE;
- `CD` MAE and RMSE;
- drag-count error:
  \[
  10^4 |\Delta C_D|;
  \]
- error versus alpha;
- error versus `CL`;
- drag at required lift.

For `CM`:

- MAE;
- RMSE;
- R²;
- error versus alpha and lift.

Design metrics:

- target-lift root availability;
- error in alpha required for target lift;
- error in drag at target lift;
- ranking correlation;
- top-k candidate overlap;
- constraint-classification accuracy.

Breakdowns:

- family;
- geometry cluster support distance;
- alpha;
- source coverage;
- thickness;
- camber;
- transition movement;
- source tier.

Use bootstrap confidence intervals by nominal aerofoil, not by individual row.

### Phase 14 — NeuralFoil benchmark

Install and inspect the current NeuralFoil API rather than guessing it.

For the same held-out nominal geometries and operating points:

- `Re = 1_000_000`;
- same alpha;
- same coordinate geometry;
- collect:
  - `CL`;
  - `CD`;
  - `CM`;
  - transition outputs;
  - `analysis_confidence`.

Compare:

- V2 versus source labels;
- NeuralFoil versus source labels;
- V2 versus NeuralFoil;
- error versus NeuralFoil confidence;
- candidate ranking.

Do not fit V2 to NeuralFoil outputs.

Write:

```text
results/robust_v2/pilot/neuralfoil_metrics.json
reports/robust_v2/modelling/NEURALFOIL_BENCHMARK.md
```

### Phase 15 — Local XFOIL wrapper and validation

#### 15.1 General safety

Run every XFOIL case:

- in a unique temporary directory with no spaces in filenames;
- with a copied, sanitised coordinate file;
- with graphics disabled where supported;
- under a subprocess timeout;
- with stdout/stderr captured;
- with the exact command script preserved;
- with process termination on timeout;
- with no shell interpolation of untrusted filenames.

Sanitise title lines beginning with `T` or `F` by prefixing `_`.

#### 15.2 Do not falsely claim exact AirfoilTools reproduction

The local official Windows binary is XFOIL 6.99, while the historical web service may have used another build and undisclosed details. Label local runs:

```text
local_xfoil_6_99_crosscheck
```

unless exact equivalence is proven.

#### 15.3 Base command template

Generate a command file equivalent to:

```text
PLOP
G F

LOAD foil.dat
PANE
OPER
VISC 1000000
MACH 0
VPAR
N 9

ITER 70
PACC
polar.txt

INIT
ALFA 0
ASEQ 0 15 0.25
PACC

QUIT
```

The blank lines are meaningful menu exits/prompts. Validate this template against the official session examples and actual transcript.

For a negative sweep, use a separate fresh process:

```text
INIT
ALFA 0
ASEQ 0 -10 -0.25
```

Merge converged unique rows after parsing. Do not infer missing rows.

Do not issue `FILT` merely because the paper mentions it unless you have verified the correct menu and effect. If testing a smoothing compatibility mode, make it a separately labelled ablation and retain the command transcript.

#### 15.4 Canary validation set

Run at least:

- NACA 0012;
- NACA 2412;
- Clark Y;
- Eppler E387;
- one Selig/AG profile;
- one FX/Wortmann profile if valid.

Compare local XFOIL to snapshot/live source on overlapping alpha:

- convergence;
- `CL`;
- `CD`;
- `CM`;
- transition;
- ranking.

Differences are expected. Characterise them; do not silently merge source tiers.

### Phase 16 — Model trust and empirical uncertainty calibration

After architecture selection is frozen:

1. train five ensemble members;
2. use different seeds;
3. resample nominal geometry clusters, not point rows;
4. compute ensemble disagreement;
5. calculate development-support distance in scaled CST space;
6. use untouched calibration geometries to model residual quantiles as a function of:
   - k-nearest-neighbour distance;
   - ensemble disagreement;
   - alpha;
   - target;
   - source coverage;
7. define a trusted domain from measured residual tolerance.

Use split-conformal or quantile calibration where valid, while respecting grouped nominal geometries.

Do not call ensemble standard deviation a formally exact epistemic variance. Use:

```text
ensemble disagreement
empirical predictive uncertainty
calibrated residual band
```

Store:

```text
results/robust_v2/calibration/trust_model.*
reports/robust_v2/modelling/TRUST_CALIBRATION_REPORT.md
```

### Phase 17 — Manufacturing geometry uncertainty

#### 17.1 Ava baseline

Implement independent multiplicative perturbation:

\[
p_j' = p_j(1+u_j), \quad u_j \sim U[-0.01,0.01].
\]

For parameters very near zero, do not silently add an arbitrary scale. Record that multiplicative perturbation gives near-zero absolute variation and include an explicit sensitivity ablation using 1% of the development-set parameter standard deviation.

This is the simple baseline, not the primary physical model.

#### 17.2 Independent coordinate-noise ablation

Perturb coordinates independently, then demonstrate why it produces roughness/invalidity.

#### 17.3 Primary smooth correlated surface-normal field

On a common surface parameterisation:

\[
\mathbf r'(s)=\mathbf r(s)+\delta n(s)\hat{\mathbf n}(s).
\]

Construct a smooth basis using cubic B-splines or a covariance eigenbasis.

Requirements:

- bounded maximum normal displacement;
- zero or controlled TE displacement;
- LE taper/control;
- configurable upper/lower correlation;
- common random numbers across candidates;
- deterministic seed;
- actual coordinate geometry generated and checked;
- CST refit used only for surrogate input;
- actual displaced coordinates retained for XFOIL validation.

For bounded Sobol coefficients \(\xi_j \in [-1,1]\), normalise the basis so:

\[
\max_s |\delta n(s)| \le \epsilon c.
\]

Test:

```text
0.1%, 0.25%, 0.5%, 1.0% chord
```

Unless measurement data exists, call these prescribed bounded uncertainty levels, not measured manufacturing distributions.

Reject and count:

- self-intersection;
- negative local thickness;
- TE failure;
- LE failure;
- excessive CST refit error;
- trust-domain violation.

### Phase 18 — Sobol sampling and convergence

Use scrambled Sobol QMC with `random_base2` and powers of two.

Do not skip/thin the sequence.

Use:

- 64 common samples during optimisation;
- 2048 independent final evaluation samples;
- antithetic pairing where valid.

Convergence schedule:

```text
16, 32, 64, 128, 256, 512, 1024, 2048
```

Monitor:

- mean;
- standard deviation;
- 5th and 95th percentiles;
- violation probability;
- CVaR 95%.

Store the exact sample matrix hash.

### Phase 19 — Optimisation formulations

#### 19.1 Ava's original viability objective

For service angles \(\alpha_k\), perturb each candidate and calculate:

\[
\overline{C_L}
=
\frac{1}{K}\sum_k E[C_L(\alpha_k)]
\]

and:

\[
\overline{\sigma_{C_L}}
=
\frac{1}{K}\sum_k \operatorname{Std}[C_L(\alpha_k)].
\]

Use a normalised minimisation objective:

\[
J_{\text{Ava}}
=
-\frac{\overline{C_L}-\overline{C_{L,\mathrm{ref}}}}{s_{CL}}
+
\lambda
\frac{\overline{\sigma_{C_L}}}{s_{\sigma}}.
\]

Run:

```text
lambda = 0, 0.25, 0.5, 1, 2, 4
```

This must be reported as the faithful simple baseline. It is not sufficient alone because unconstrained lift maximisation can reward extreme camber and poor drag/moment.

#### 19.2 Recommended drag-risk formulation

At each required lift \(C_{L,\mathrm{req},k}\), solve within the trusted alpha interval:

\[
C_L(p,\xi_s,\alpha_{k,s})=C_{L,\mathrm{req},k}.
\]

If no root exists, mark a constraint violation.

For realisation \(s\):

\[
D_s(p)=\sum_k w_k C_D(p,\xi_s,\alpha_{k,s}).
\]

Objectives:

\[
\min E[D_s]
\]

and:

\[
\min \operatorname{CVaR}_{0.95}(D_s).
\]

#### 19.3 Constraints

Relative to the verified reference aerofoil:

\[
0.98\,t_{\mathrm{ref}}\le t/c\le1.02\,t_{\mathrm{ref}}
\]

\[
A(p)\ge0.98A_{\mathrm{ref}}
\]

\[
r_{LE}(p)\ge0.8r_{LE,\mathrm{ref}}
\]

\[
C_M(p,\alpha_k)\ge C_{M,\mathrm{ref}}(\alpha_k)-0.01.
\]

Also enforce:

- no crossing;
- no self-intersection;
- minimum local thickness;
- TE limit;
- CST validity;
- training-support/trust limit;
- target-lift availability;
- uncertainty-realisation feasibility rate.

#### 19.4 Design bounds

Derive bounds from development data only:

- robust 1st–99th percentiles;
- reference-centred limits;
- trust/support constraints.

Never let the optimiser explore arbitrary enormous CST values.

#### 19.5 Optimisers

First run a scalar SciPy differential-evolution smoke.

Then use pymoo NSGA-II.

Debug:

```text
population 32
generations 10
uncertainty samples 16
```

Pilot:

```text
population 64
generations 50
uncertainty samples 64
```

Full:

```text
population 128
generations 200
uncertainty samples 64
10 independent seeds
```

Track:

- hypervolume;
- feasible fraction;
- front movement;
- duplicate candidates;
- trust rejections;
- invalid geometry;
- repeated-seed agreement.

### Phase 20 — Designs to compare

Always compare:

1. verified reference aerofoil;
2. deterministic optimum;
3. Ava independent-CST robust optimum;
4. smooth correlated-surface robust optimum;
5. trust-constrained Pareto knee.

For each, report:

- CST parameters;
- actual coordinates;
- geometry metrics;
- nominal predictions;
- uncertainty distributions;
- expected drag;
- CVaR;
- mean lift;
- lift variability;
- moment;
- trust;
- constraint margins;
- direct XFOIL results.

### Phase 21 — Direct validation of selected designs

For viability smoke:

- reference;
- deterministic candidate;
- one robust candidate;
- nominal plus at least four shared perturbations.

For the complete study:

- all five designs;
- nominal;
- at least 50–100 independent perturbations initially;
- selected adverse-tail realisations.

Compare:

- surrogate and XFOIL means;
- tails;
- constraint violations;
- candidate ranking;
- effect size;
- confidence intervals.

If XFOIL reverses the claimed robust ranking, issue HOLD and diagnose before proceeding.

### Phase 22 — Viability report

Generate:

```text
reports/robust_v2/viability/VIABILITY_REPORT.md
reports/robust_v2/viability/viability_report.json
reports/robust_v2/handover/AGENT_HANDOVER.md
reports/robust_v2/handover/NEXT_COMMANDS.ps1
```

The report must include:

1. executive decision: `GO`, `HOLD`, or `NO-GO`;
2. exact Git branch and SHA;
3. environment;
4. V1 reproduction status;
5. source reachability;
6. local recovery findings;
7. pinned snapshot inventory;
8. UIUC continuity/current differences;
9. mapping counts by status;
10. accepted/rejected data counts;
11. geometry/CST results;
12. leakage tests;
13. pilot model metrics;
14. baseline comparisons;
15. NeuralFoil benchmark;
16. XFOIL cross-check;
17. uncertainty validity;
18. optimisation smoke;
19. blockers and their severity;
20. scientific limitations;
21. exact continuation commands;
22. complete changed-file list;
23. output/hash manifest.

---

## 10. AUTOMATIC CONTINUATION AFTER VIABILITY

Do not stop after a promising 25-airfoil smoke.

### If the pilot is GO or inconclusive-but-pipeline-sound

Automatically:

1. build the deterministic 200-airfoil expansion;
2. rerun source, mapping, geometry and split audits;
3. train baseline and compare to simple baselines;
4. run the 200-airfoil XFOIL/NeuralFoil diagnostic sample;
5. update the viability decision.

### If the 200-airfoil expansion passes

Automatically:

1. parse all exact or geometry-confirmed available `Re=1e6` polars from local recovery and the pinned snapshot;
2. build the full long-form dataset;
3. freeze development/calibration/locked-test splits;
4. train the full baseline on development;
5. run 50 sequential persistent Optuna trials;
6. retrain the best stable configurations across multiple seeds;
7. select and freeze the architecture;
8. calibrate uncertainty/trust on calibration;
9. write `FROZEN_MODEL_MANIFEST.json`;
10. evaluate the locked test once;
11. run uncertainty and optimisation pilots;
12. update the handover.

### If live AirfoilTools is unavailable

Continue using local recovery and the pinned snapshot. Do not wait and do not repeatedly retry the site.

### If the pinned snapshot has insufficient exact matches

Use local XFOIL only to expand a controlled pilot/validation set. Do not silently turn a small local-XFOIL set into a claim equivalent to the original AirfoilTools corpus.

---

## 11. LIVE AIRFOILTOOLS CLIENT — EXACT DESIGN CONTRACT

Implement `AirfoilToolsClient` with:

```python
class AirfoilToolsClient:
    def probe_canary(self) -> ProbeResult: ...
    def fetch_polar(self, polar_key: str) -> RawFetchResult: ...
    def fetch_many(self, polar_keys: Sequence[str], limit: int) -> FetchSummary: ...
```

Use `httpx.Client`:

- `follow_redirects=True`;
- explicit connect/read timeouts;
- one connection;
- descriptive User-Agent containing the contact;
- no cookies persisted beyond the client session;
- TLS verification enabled;
- no proxy bypass;
- no CAPTCHA handling;
- no browser automation.

User-Agent format:

```text
Ben-Gawith-LJMU-Airfoil-Research/2.0 (contact: <AIRFOILTOOLS_CONTACT>)
```

Response validation before treating a body as data:

1. status 200;
2. body length plausible;
3. body not beginning with HTML;
4. no CAPTCHA/access-denied markers;
5. dynamically detected aerodynamic header;
6. metadata parseable;
7. condition metadata exact;
8. source key matches requested key.

Raw naming:

```text
{polar_key}__{sha256_prefix}.csv
```

Manifest entry:

```text
request_url
requested_key
status_code
retrieved_utc
elapsed_seconds
content_type
content_length
etag
last_modified
sha256
raw_path
validation_status
failure_reason
```

Retry policy:

- retry only network errors and `500`, `502`, `503`, `504`;
- backoff approximately 2, 5 and 10 seconds;
- do not retry `401`, `403`, `404`;
- for `429`, honour one `Retry-After`, then stop the live run;
- maintain the minimum delay even after successful requests;
- never use parallel workers.

Every live-network test must be marked `network` and disabled by default.

---

## 12. XFOIL WRAPPER — EXACT DESIGN CONTRACT

Implement:

```python
@dataclass(frozen=True)
class XFoilCase:
    geometry_path: Path
    reynolds_number: float
    mach: float
    ncrit: float
    alpha_start: float
    alpha_end: float
    alpha_step: float
    iterations: int
    mode: str

@dataclass
class XFoilResult:
    status: str
    return_code: int | None
    timed_out: bool
    command_text: str
    stdout_path: Path
    stderr_path: Path
    polar_path: Path | None
    parsed_points: pd.DataFrame
    failure_reason: str | None
```

Requirements:

- use `subprocess.Popen` without `shell=True`;
- send command text via stdin;
- run in temporary directory;
- copy/sanitise geometry;
- timeout and kill process tree;
- preserve transcript;
- parse only written polar rows;
- count non-convergence by missing requested alpha;
- run positive and negative sweeps separately;
- never fill missing alphas;
- merge only unique converged rows;
- record executable SHA-256 and version source.

Unit-test the polar parser using fixtures. Integration tests require `--run-xfoil`.

---

## 13. PRIMARY MODEL — IMPLEMENTATION DETAILS

Suggested implementation:

```python
class ResidualMLPBlock(nn.Module):
    ...

class ConditionedPolarMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 13,
        hidden_width: int = 256,
        hidden_layers: int = 3,
        dropout: float = 0.10,
    ):
        ...
        self.cl_head = nn.Linear(hidden_width, 1)
        self.log_cd_head = nn.Linear(hidden_width, 1)
        self.cm_head = nn.Linear(hidden_width, 1)

    def forward(self, x):
        ...
        return {
            "cl": ...,
            "log_cd": ...,
            "cm": ...,
        }
```

Loss API:

```python
def masked_macro_multitask_loss(
    predictions: dict[str, Tensor],
    targets: dict[str, Tensor],
    masks: dict[str, Tensor],
    airfoil_weights: Tensor,
    loss_name: str,
    huber_delta: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    ...
```

The loss must:

- ignore unavailable targets;
- normalise each head by its available weighted count;
- average active head losses equally;
- fail clearly if a batch has no active target;
- be stable for large residuals;
- never silently return NaN.

Sampler API:

```python
class UniformAirfoilPointSampler(Sampler[int]):
    ...
```

It must sample nominal airfoils uniformly, then points within an airfoil, or achieve mathematically equivalent aggregate weighting.

---

## 14. TUNING CONTRACT

Use Optuna with persistent SQLite and `load_if_exists=True`.

Example:

```python
study = optuna.create_study(
    study_name=config.study_name,
    storage=config.storage,
    load_if_exists=True,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=config.seed),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=20,
        interval_steps=5,
    ),
)
```

Initial tuning must be sequential (`n_jobs=1`) for reproducibility.

Objective:

1. grouped cross-validation by geometry cluster inside development;
2. calculate standardised absolute errors per nominal airfoil;
3. macro-average airfoils;
4. average active `CL`, `log(CD)`, and `CM` objectives;
5. report intermediate epoch objective;
6. prune only after warmup;
7. write every trial's full evidence.

Do not use the locked test.

After tuning:

- retrain top 5 configurations over at least 3 seeds;
- compare mean and spread;
- choose simplest configuration statistically indistinguishable from best;
- record the selection rule before opening test.

---

## 15. TEST SUITE — REQUIRED TESTS AND ACCEPTANCE BEHAVIOUR

### 15.1 Default local test command

```powershell
uv run pytest tests/robust_v2 -m "not network and not xfoil and not slow" -q
```

### 15.2 Full unit coverage command

```powershell
uv run pytest tests/robust_v2 -m "not network and not xfoil" --cov=robust_airfoil --cov-report=html --cov-report=term-missing
```

### 15.3 Test categories

#### Repository/provenance

- deterministic SHA-256;
- atomic JSON writes;
- source manifest append;
- raw files immutable;
- redacted paths;
- run-state resume;
- changed input invalidates a passed phase.

#### AirfoilTools parser

- dynamic header;
- metadata order variation;
- BOM;
- whitespace;
- blank fields;
- `*`;
- HTML rejection;
- wrong Re rejection;
- wrong Mach rejection;
- wrong Ncrit rejection;
- duplicate alpha;
- conflicting duplicate;
- all seven aerodynamic columns;
- unknown XFOIL version remains null.

#### Live client

Using `respx` only:

- success;
- 301/302 redirect;
- 403 stop;
- 429 stop;
- transient 502 retry;
- HTML response rejection;
- cache reuse;
- no overwrite;
- rate-delay call;
- limit enforced;
- contact required.

#### UIUC/geometry

- Selig order;
- Lednicer order;
- comments;
- title;
- name beginning T/F sanitisation;
- repeated LE;
- finite TE;
- reversed input;
- non-unit chord;
- orientation;
- cosine interpolation;
- surface crossing;
- self-intersection;
- negative thickness;
- geometry metrics;
- deterministic hash.

#### CST

- five weights per side;
- feature order;
- extraction;
- reconstruction;
- roundtrip;
- symmetric profile;
- cambered profile;
- finite TE;
- API-change failure message.

#### Mapping

- exact name;
- NACA alias;
- punctuation normalisation;
- geometry exact;
- geometry confirmed;
- ambiguous rejection;
- fuzzy score cannot accept;
- mismatched geometry rejection.

#### Dataset

- no padded rows;
- `cd > 0` required for log;
- target masks;
- point weights aggregate equally by airfoil;
- source metadata retained;
- rejected ledger complete;
- schema roundtrip;
- stable dataset hash.

#### Splits

- no airfoil across splits;
- no cluster across splits;
- perturbation inherits split;
- deterministic seed;
- locked-test guard;
- scaler uses development only.

#### Model/training

- forward shape;
- masked loss;
- no-target failure;
- one-batch overfit;
- checkpoint roundtrip;
- inverse target transform;
- deterministic smoke;
- CPU;
- optional CUDA;
- no test access during tune;
- sampler balance.

#### NeuralFoil

- adapter output schema;
- confidence retained;
- no use as training source.

#### XFOIL

- command generation;
- blank-line placement;
- filename sanitisation;
- polar parsing;
- timeout;
- missing-alpha handling;
- positive/negative merge;
- no interpolation.

#### Uncertainty

- max displacement bound;
- smoothness;
- common random numbers;
- reproducibility;
- TE taper;
- invalid geometry rejection;
- CST refit recorded;
- antithetic samples;
- Sobol power-of-two rule.

#### Risk/optimisation

- mean;
- standard deviation;
- quantile;
- CVaR against hand calculation;
- target-lift root;
- no-root constraint;
- thickness/area/moment constraints;
- trust rejection;
- NSGA-II smoke;
- deterministic seed;
- result serialisation.

#### Reporting

- all required sections;
- decision enum;
- hashes included;
- failed phase visible;
- exact resume command.

### 15.4 CI

Create a CPU-only GitHub Actions workflow that runs:

```text
ruff
mypy on src/robust_airfoil
unit tests excluding network/xfoil/slow
```

Do not run live scraping or XFOIL in CI.

---

## 16. DECISION RULES

### GO

Issue GO when all are true:

- end-to-end pipeline passes;
- provenance and raw immutability pass;
- pilot/expanded mapping and parse gates pass;
- no leakage;
- one-batch overfit passes;
- `CL` and `log(CD)` learn clear signal over Dummy;
- uncertainty produces mostly valid geometry under intended levels;
- optimisation smoke returns feasible candidates;
- trust constraints prevent obvious extrapolation exploitation;
- local XFOIL diagnostics do not reverse the core ranking;
- no critical unresolved blocker.

### HOLD

Issue HOLD when the code path works but one or more of these remains unresolved:

- insufficient exact/geometry-confirmed polars;
- source version mismatch not characterised;
- model signal inadequate for design;
- drag errors larger than candidate improvement;
- uncertainty assumptions dominate the result;
- XFOIL reverses ranking;
- mapping ambiguity material;
- calibration data too small;
- test set was accidentally exposed;
- a paper/code discrepancy affects claims.

HOLD must include an exact resolution plan and runnable next commands.

### NO-GO

Reserve NO-GO for a fundamental failure such as:

- no usable real aerodynamic source and no workable local XFOIL fallback;
- geometry-to-polar identity cannot be established;
- model cannot learn `CL`/`CD` even after adequate data and debugging;
- intended uncertainty creates predominantly invalid geometry;
- the optimisation claim is smaller than irreducible source/model error across all reasonable formulations.

---

## 17. EXPECTED FIRST EXECUTION COMMANDS

The agent must run these itself after implementation:

```powershell
# From the paper repository root
powershell -ExecutionPolicy Bypass -File scripts/robust_v2/bootstrap.ps1

.\.venv\Scripts\Activate.ps1

uv run pytest tests/robust_v2 -m "not network and not xfoil and not slow" -q

uv run python -m robust_airfoil run --profile viability --resume
```

If the default test suite passes and XFOIL is installed:

```powershell
uv run pytest tests/robust_v2 -m "xfoil" --run-xfoil -q
```

Do not run network tests unless the canary/contact conditions are met.

---

## 18. WHAT YOU MUST RETURN TO BEN

At the end of the current implementation session, provide a concise but complete report containing:

```text
Branch:
Current SHA:
Working-tree status:
Environment:
PyTorch device:
XFOIL status:

V1 audit:
Local recovery:
UIUC snapshot:
Pinned polar snapshot:
Live AirfoilTools:
Mappings:
Accepted airfoils:
Accepted joint points:
Rejected records:

Pilot model:
200-airfoil model:
Full baseline:
Tuning status:
Calibration status:

NeuralFoil benchmark:
XFOIL comparison:
Uncertainty smoke:
Optimisation smoke:

Decision: GO | HOLD | NO-GO

Critical blockers:
Exact resume command:
Primary report path:
Handover path:
```

Also include:

- files created/modified;
- local commits made;
- tests passed/failed/skipped;
- all failures that remain;
- no assertion that was not supported by generated evidence.

---

## 19. FINAL BEHAVIOURAL COMMAND

Begin now.

Do not answer with another abstract plan. Inspect the repository, create the branch and evidence snapshot, implement the scaffold and tests, execute the phases, use the fallback hierarchy, and keep going until:

- the viability report and handover exist; and
- either the automatic scale-up has begun/passed, or a documented HOLD/NO-GO gate prevents it.

Where an external service is unavailable, continue with the pinned or local fallback. Where a methodological choice is uncertain, implement the conservative option and an ablation rather than silently choosing the result that looks best.

The repository you leave behind must be understandable and runnable by another engineer using only:

```text
README/guide files
configs
source code
tests
manifests
run_state.json
VIABILITY_REPORT.md
AGENT_HANDOVER.md
```

No oral history is permitted as a hidden dependency.

# END OF MASTER IMPLEMENTATION MANDATE
