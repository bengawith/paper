# Manuscript package

This directory contains the resubmission manuscript and all material needed to
reproduce its quantitative content.

| File | Purpose |
|---|---|
| `manuscript.md` | Full manuscript (abstract, methodology, results, discussion, figures). |
| `response_to_reviewers.md` | Point-by-point response mapping each prior reviewer comment to a concrete change and result. |
| `data/paper_data.json` | Single consolidated source of every number in the manuscript, extracted from the durable lineage artefacts. |
| `figures/` | Publication figures (150 dpi, 14 pt fonts, labelled subplots). |

## Regenerating the figures

```powershell
uv run python scripts/robust_v2/make_paper_figures.py
```

Figures are produced only from committed lineage artefacts (`results/robust_v2/lineages/robust-v2-current-20260908/`)
via `paper/data/paper_data.json`, so no figure contains a number that is not backed by a durable evidence file.

## Figure index

| Figure | File | Shows |
|---|---|---|
| 1 | `fig_surrogate_accuracy.png` | Held-out surrogate accuracy for `C_L`, `C_D`, `C_M`, coloured by trust membership. |
| 2 | `fig_manufacturing_risk.png` | Robust weighted-drag objective (expected and CVaR₉₅) and trust-exit fraction vs manufacturing amplitude. |
| 3 | `fig_perturbation_validity.png` | Geometric admissibility of three perturbation models. |
| 4 | `fig_pareto_front.png` | Aggregated surrogate robust front with minimum-expected and minimum-CVaR endpoints. |
| 5 | `fig_design_verification.png` | Direct XFOIL verification: within-trust design beats NACA 2412; outside-trust optimum fails. |
| 6 | `fig_surrogate_trust_stress.png` | Adversarial disagreement vs trust domain. |
| 7 | `fig_worstcase_polar.png` | Worst-case surrogate–XFOIL lift divergence (outside trust). |

## Provenance

All results derive from study lineage `robust-v2-current-20260908` at Re = 1×10⁶, M = 0,
N_crit = 9. The surrogate accuracy and trust split come from the frozen ensemble's
locked-test predictions; the manufacturing risk and perturbation ablations from the
uncertainty study; the Pareto front from the ten-seed NSGA-II optimisation; the XFOIL
verification from the candidate-validation artefact; and the adversarial probe from
`results/robust_v2/lineages/robust-v2-current-20260908/surrogate_stress/`.
