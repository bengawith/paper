# Data-Driven Aerodynamic Prediction

Data-Driven Aerodynamic Prediction: High-Fidelity Lift Curve Estimation for Aerofoils Using Class-Shape Transformation and Artificial Intelligence

**Short summary:** This repository accompanies the research paper above. It contains code, datasets (small subsets and metadata), and scripts used to train, evaluate and analyze machine learning models that predict high-fidelity lift curves from airfoil Class-Shape Transformation (CST) parameters and operating conditions.

**Contents**
- **Code:** Core library is inside the `src/` package. Top-level scripts are in `scripts/`.
- **Data:** Sample and processed datasets are in `data/csv/` and raw `data/aerofoil_data/`.
- **Models:** Trained model artifacts are in `trained_model/`.

**Quick Links**
- **Main package:** [src](src)
- **Scripts:** [scripts](scripts)

## Robust V2 viability study

The provenance-first Robust V2 study is implemented in `src/robust_airfoil/` and is governed by `DIRECTIVE/MASTER_IMPLEMENTATION_PROMPT_AIRFOIL_ROBUST_V2.md`.

As of **August 27, 2026**, the completed viability run is **HOLD**, not GO. The model-quality, calibration, uncertainty-sampling, and optimisation-feasibility stages completed, but the contract remains blocked by:

1. unsupported exact V1 reproduction because the historical split/scaler/dependency provenance is incomplete;
2. failed ten-seed NSGA-II front-agreement threshold; and
3. incomplete strict direct-XFOIL sweep coverage.

Canonical evidence:

- `reports/robust_v2/viability/VIABILITY_REPORT.md`
- `reports/robust_v2/viability/viability_report.json`
- `reports/robust_v2/handover/AGENT_HANDOVER.md`
- `results/robust_v2/uncertainty/manufacturing_study.json`
- `results/robust_v2/optimisation/optimisation_summary.json`
- `results/robust_v2/xfoil_canaries/canary_summary.json`
- `results/robust_v2/xfoil_candidates/candidate_validation_summary.json`

Run and validate from PowerShell:

```powershell
uv sync --extra dev
uv run ruff check src/robust_airfoil tests/robust_v2
uv run mypy src/robust_airfoil
uv run pytest tests/robust_v2 -m "not network and not xfoil and not slow" -q
uv run python -m robust_airfoil run --profile viability --resume
```

Large raw datasets, model checkpoints, XFOIL executables, per-segment console logs, and per-case polar files are intentionally excluded from Git. Their hashes and compact summaries remain in the committed provenance manifests and reports.

**Installation**
- Create and activate a virtual environment (recommended Python 3.10+):

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# or POSIX
source .venv/bin/activate
```

- Install runtime + (optional) dev requirements:

```bash
pip install -r requirements.txt         # core runtime dependencies
```


**Primary Scripts & Usage**
- `scripts/train_best_model.py` — Train or re-train models using dataset parameters defined in `data/csv/` and model configs in `physics_aware/model_details/`.
- `scripts/physics_aware_tuner.py` — Hyperparameter sweep and tuning wrapper for physics-aware losses and architectures.
- `scripts/physics_aware_sensitivity_analysis.py` — Runs sensitivity analysis (Morris / Sobol) on model inputs.
- `scripts/gather_model_details.py` — Collates model metadata for `physics_aware/model_details/`.
- `scripts/analyze_physics_results.py` — Post-hoc analysis and figure generation.

**Core library highlights**
- `src/utils.py` — Utility functions including CST parameter handling, physics-metrics (`calculate_physical_metrics`, `calculate_physics_score`), small helper model `QuickMLP`, and training/evaluation helpers.
- `src/models.py` / `src/model_components.py` — Model definitions (MLP, GRU, LSTM, CNN flavours used in the experiments).

**Data format**
- Predictands: lift coefficient curves are arranged as columns `alpha_0..alpha_47` for a fixed alpha grid, or packed arrays in the processed CSV files under `data/csv/`.
- Inputs: CST parameters (e.g., `lower_weight_*`, `upper_weight_*`, `TE_thickness`, `leading_edge_weight`), metadata columns and operating conditions.

Refer to `data/csv/dataset_12CST_params.csv` for an example.

**Reproducibility**
- The repository includes utilities to set random seeds (`src.utils.set_seed`) and helper methods to evaluate physics-aware metrics.
