# Robust V2 — latest author manuscript and evidence

**14 October 2026: ready for Ben's offline review and Ava's independent CFD planning. Not yet a journal-submission-ready robust-optimisation result.**

## The latest paper — no mixed versions

- `LATEST_MANUSCRIPT_FOR_AVA.docx` is the **standalone current manuscript only** (one paper; eight current figures and seven tables).
- `manuscript.md` is the canonical source for that document.
- `figures/` holds seven main and one supplementary ML diagnostic figure.
- `response_to_reviewers.md` concerns the **earlier reviews**, not a current invitation or endorsement by a new journal.

The original V1 paper belongs **only in the review archive as `prior_final_draft.docx`** for tone/method comparison. It is not merged into the Ava-ready manuscript. The full scientific assessment is `reports/robust_v2/submission_review_20261014/FINAL_RESEARCH_RESOLUTION.md` (from repository root).

## What is backed by evidence

- Five-member point-conditioned MLP: grouped held-out R² **0.989 for CL, 0.922 for log-CD and 0.918 for CM**, across 26,692 XFOIL-computed states from 233 aerofoils.
- Frozen acceptance rule: 66.1% accepted, predominantly on **support distance**. The original ensemble-spread thresholds reject no additional held-out states independently.
- Independent calibration-only five-fold **geometry-group** analysis: adding ensemble disagreement to support distance improves high-error AUROC from **0.767 to 0.866**, difference **0.099**, with a 300-group-bootstrap interval approximately **[0.069, 0.139]**. This is an *incremental ML signal within calibration data*, not a newly certified or deployed trust policy.
- Selected **nominal** direct-XFOIL design comparison: within-screen aerofoil has **18.7% lower** required-lift weighted drag than NACA 2412. Two 128-scenario, 0.5%-chord draws have **92 and 96 complete matched cases**, with lower mean drag by **15.4% and 15.8%** among successfully converged pairs. Failures were excluded, so no unconditional physical CVaR claim is established.
- Original multi-objective optimisation incorrectly transferred reference aerofoil CST perturbation increments to candidate shapes. The **code has been corrected** to apply common latent scenarios to each candidate's *own surface normals*, refit CST and explicitly count/penalise invalid geometry/trust exits. Old robust Pareto optimality is invalidated. A separate **10-seed low-budget smoke** (32 population × 10 generations × 16 perturbations) finds feasible solutions across all ten and objective-front distance **0.149** versus 0.20 threshold. **The full 128 × 200 × 64 × 10 production rerun remains outstanding.** The corrected candidate scores include trust penalties and are not pure drag.

## Boundaries before submission

The manufacturing field is a prescribed, correlated, continuous *piecewise-linearly interpolated* surface-normal deviation model, not measured production tolerances. Local XFOIL shares modelling assumptions with the training-label solver family. Independent viscous CFD or experiments have not yet established aerodynamic agreement. The optimiser production run and coauthor acceptance of final claims are still required before a robust Pareto-optimality statement.

Start at `reports/robust_v2/submission_review_20261014/READ_THIS_FIRST.md`. The canonical packaged review folder on SamsPC is `C:\Users\Ben\OneDrive\paper\READY_FOR_REVIEW`, with SHA-256 manifest, strict XFOIL evidence, code-provenance records, a saved response email, an offline-copy PowerShell script, and selected CFD geometry+protocol.

Run software checks:

```powershell
uv sync --extra dev --extra benchmarks --extra docs
uv run ruff check src/robust_airfoil tests/robust_v2 scripts/robust_v2
uv run mypy src/robust_airfoil
uv run pytest tests/robust_v2 -m "not network and not xfoil and not slow" -q
```

No external journal or coauthor is claimed to have approved or accepted this manuscript.
