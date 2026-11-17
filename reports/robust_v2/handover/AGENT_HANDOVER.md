# Agent Handover

Decision: **HOLD**

- [HIGH] v1_exact_reproduction: Committed V1 entrypoints are internally inconsistent and exact split/scaler provenance is absent; exact reproduction is not supportable. Resolution: Recover the original V1 split IDs, scaler state, dependency lock, and training command, then rerun the frozen V1 comparison before making V1-to-V2 claim language.
- [HIGH] model_evidence_not_current: Model phase 09-16 is failed; no current verified model evidence is available for assessment. Resolution: Repair the failed or incomplete model phase, verify its frozen artifacts, and rerun phases 09-22 before interpreting model performance.
- [HIGH] ml_evidence_missing_or_invalid: The required ML evidence artifact has not been produced. Resolution: Produce a lineage-bound ML evidence artifact for the frozen model, with independent-group macro-error comparisons against Dummy for CL and log(CD), then rerun phase 22. An audit report alone cannot grant this gate.
- [HIGH] advanced_evidence_not_current: Advanced phase 17-21 is running; no current verified uncertainty, optimisation, or XFOIL evidence is available for assessment. Resolution: Repair the failed or incomplete advanced phase and rerun phases 17-22 with the verified frozen model before interpreting optimisation or XFOIL results.

Resume:
```powershell
uv run python -m robust_airfoil run --profile viability --resume
```
