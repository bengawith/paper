# Risk-Aware Aerofoil Design under Manufacturing Uncertainty

**Trust-calibrated machine-learning surrogate, risk-aware optimisation, and multi-fidelity (XFOIL) verification for two-dimensional aerofoils.**

This repository accompanies the manuscript
*"Risk-Aware Aerofoil Design under Manufacturing Uncertainty using a Trust-Calibrated
Machine-Learning Surrogate and Multi-Fidelity Verification"* (Gawith, Shahrokhi, Navaie).
The manuscript and its point-by-point response to the previous round of peer review are in
[`paper/`](paper/).

## What this project does

A compact surrogate predicts lift, drag and pitching-moment coefficients of an aerofoil
directly from a 12-parameter Class–Shape Transformation (CST) geometry, and is used to
perform **manufacturing-aware robust design** with every conclusion **verified against the
XFOIL panel solver**. The three contributions are:

1. **A trust-calibrated multi-output surrogate.** An ensemble of point-conditioned MLPs
   predicts `C_L`, `C_D`, `C_M`, with a split-conformal *trust domain* that labels each
   query as reliable or not. Errors inside the trust domain are 2–2.7× smaller than outside.
2. **Risk-aware optimisation under manufacturing uncertainty.** Manufacturing deviation is a
   smooth, bounded, spatially correlated surface-normal displacement (0.1–1% chord). NSGA-II
   minimises expected drag and 95% CVaR tail-risk drag at required lift, giving an explicit
   deterministic-versus-robust trade-off.
3. **Multi-fidelity verification and adversarial trust probing.** XFOIL confirms the
   surrogate's design ranking (Spearman ρ = 1.00) and the drag advantage of the optimised
   designs; an adversarial search (per Reviewer #2) locates the worst-case surrogate–solver
   disagreement and shows it is confined to the untrusted region.

## Headline results (Re = 1×10⁶, M = 0, N_crit = 9)

| Result | Value |
|---|---|
| Surrogate accuracy, held-out (26,692 states, 233 aerofoils) | `C_L` R²=0.989, `log C_D` R²=0.922, `C_M` R²=0.918 |
| Accuracy inside vs outside trust domain (`C_L` MAE) | 0.040 vs 0.090 |
| Manufacturing drag risk, 0.1% → 1% chord | mean 0.0080 → 0.0099; CVaR₉₅ 0.0082 → 0.0158 |
| Physical admissibility of perturbation models (1% chord) | smooth-normal 100% vs independent-coordinate 0.3% |
| Deterministic vs robust optimum | E `C_D` 0.00667 / 0.00719; CVaR₉₅ 0.00791 / 0.00781 |
| Surrogate vs XFOIL design ranking | Spearman ρ = 1.00 |
| Adversarial worst-case ΔC_L (outside trust) | 0.67, vs 0.04 inside trust |
| Inference speed (5-member ensemble, GTX 1660) | 0.009 ms/state (≈1.1×10⁵ states/s) |

All numbers are drawn from the durable evidence artefacts of study lineage
`robust-v2-current-20260908` and consolidated in
[`paper/data/paper_data.json`](paper/data/paper_data.json).

## Repository layout

| Path | Contents |
|---|---|
| `src/robust_airfoil/` | The provenance-first pipeline (surrogate, trust calibration, uncertainty, optimisation, XFOIL verification, adversarial probing). |
| `src/robust_airfoil/surrogate_stress.py` | Reviewer-directed adversarial surrogate-vs-XFOIL disagreement search. |
| `paper/manuscript.md` | The full manuscript. |
| `paper/response_to_reviewers.md` | Point-by-point response to the prior EAAI reviews. |
| `paper/figures/` | Publication figures (regenerable from committed data). |
| `paper/data/paper_data.json` | Consolidated, single-source numbers behind every figure and table. |
| `scripts/robust_v2/` | Reproduction helpers (adversarial search, figure generation). |
| `results/robust_v2/lineages/` | Hash-pinned result artefacts per study lineage. |
| `reports/robust_v2/lineages/` | Study context, run state, and evidence reports. |
| `src/utils.py`, `src/models.py` | **Legacy V1 code** for the previous submission, preserved unchanged. |

## Reproduce

```powershell
uv sync --extra dev --extra benchmarks

# fast checks
uv run pytest tests/robust_v2 -q
uv run ruff check src/robust_airfoil tests/robust_v2
uv run mypy src/robust_airfoil

# use the trust-calibrated surrogate as a tool: predict a polar for one CST aerofoil
uv run robust-airfoil predict --cst="-0.172,-0.065,-0.125,-0.055,-0.083,0.208,0.190,0.223,0.180,0.218,-0.041,0.002" --alpha-start 0 --alpha-end 10 --alpha-step 2

# regenerate the ML acceptance evidence for the frozen model
uv run robust-airfoil model evidence

# reviewer-directed adversarial surrogate-vs-XFOIL search (requires local XFOIL 6.99)
uv run robust-airfoil validate stress --lineage-id robust-v2-current-20260908

# benchmark inference speed and regenerate all manuscript figures from committed data
uv run python scripts/robust_v2/benchmark_inference.py
uv run python scripts/robust_v2/make_paper_figures.py
```

The public interface (`robust-airfoil predict|data|model|optimize|validate|report|reproduce`)
serves both the end-user prediction tool and each reproducible study stage against an
immutable, hash-pinned lineage. XFOIL 6.99 must be present at `tools/xfoil/xfoil.exe` or on
`XFOIL_EXE` for the direct-solver stages. When passing CST values that begin with a minus
sign, use the `--cst=...` form so the shell does not treat them as flags.

## Scope and honest limitations

- The study is deliberately restricted to a single operating condition (Re = 1×10⁶, M = 0,
  N_crit = 9) to isolate the robustness and trust contributions; the point-conditioned
  architecture extends to a Re/Mach-parameterised envelope given additional labelled data.
- The manufacturing model is a *prescribed*, physically admissible perturbation field, not a
  measured production distribution.
- Labels are panel-method (XFOIL) predictions; the framework is multi-fidelity and structured
  to accept selective CFD/wind-tunnel augmentation.
- The robust Pareto front's *location* is only moderately reproducible across optimisation
  seeds at the budget used (mean symmetric distance 0.28 vs a 0.20 threshold); the
  deterministic-versus-robust ranking and the reference-relative drag advantage are stable.
- A separate internal engineering gate (`reports/robust_v2/.../VIABILITY_REPORT.md`) reports
  **HOLD** on production-readiness of the tool (chiefly unrecoverable V1-reproduction
  provenance); this is distinct from, and does not diminish, the scientific results above.

## Environment

Python 3.11–3.13, `uv`, PyTorch 2.13 (CUDA 13.0), NVIDIA GTX 1660 (6 GB), XFOIL 6.99.
Large raw datasets, model checkpoints, XFOIL binaries and per-case console/polar files are
excluded from Git; their hashes and compact summaries remain in the committed provenance
manifests and reports.
