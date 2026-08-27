# Agent Handover

Decision: **HOLD**

- [HIGH] v1_exact_reproduction: Committed V1 entrypoints are internally inconsistent and exact split/scaler provenance is absent; exact reproduction is not supportable. Resolution: Recover the original V1 split IDs, scaler state, dependency lock, and training command, then rerun the frozen V1 comparison before making V1-to-V2 claim language.
- [HIGH] optimisation_seed_agreement: The ten-seed full NSGA-II fronts did not satisfy the predeclared agreement rule. Resolution: Increase convergence budget or repair front instability, then rerun all full seeds with the same frozen model and common random samples.
- [HIGH] direct_xfoil_incomplete: The six-airfoil canary and/or five-design direct-XFOIL campaign is incomplete. Resolution: Resolve every failed XFOIL case and rerun nominal, 64 shared perturbations, and adverse-tail validations for all five designs.

Resume:
```powershell
uv run python -m robust_airfoil run --profile viability --resume
```
