# Agent Handover

Decision: **HOLD**

- [HIGH] v1_exact_reproduction: Committed V1 entrypoints are internally inconsistent and exact split/scaler provenance is absent; exact reproduction is not supportable. Resolution: Recover the original V1 split IDs, scaler state, dependency lock, and training command, then rerun the frozen V1 comparison before making V1-to-V2 claim language.
- [HIGH] advanced_evidence_not_current: Advanced phase 17-21 is held; no current verified uncertainty, optimisation, or XFOIL evidence is available for assessment. Resolution: Repair the failed or incomplete advanced phase and rerun phases 17-22 with the verified frozen model before interpreting optimisation or XFOIL results.

Resume:
```powershell
uv run python -m robust_airfoil run --profile viability --resume
```
