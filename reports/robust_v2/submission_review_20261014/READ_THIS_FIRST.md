# Robust V2 — author review and CFD handoff

**14 October 2026 — This is an author-review manuscript, not a submitted or accepted journal paper.**

Open in this order:

**For the flight:** The package also includes `EMAIL_TO_AVA_DRAFT.txt` (send only after your review) and `COPY_TO_LAPTOP_OFFLINE.ps1` (safe, fail-closed laptop copy; refuses dirty Git working trees). The original V1 DOCX is included unchanged for tone comparison. The independent CFD and optimizer repeatability limitations remain open; this is an author-review package, not a journal-ready or CFD-completed result.

1. **SCIENTIFIC_REVIEW_REPORT.docx** — conclusions, verified numbers, fixes, remaining blockers and author decisions.
2. **MANUSCRIPT_AUTHOR_REVIEW.docx** — current manuscript text, 7 tables and all 7 main figures plus supplementary ML-selectivity figure.
3. **paper/manuscript.md** — canonical plain-text manuscript, then **paper/response_to_reviewers.md**.
4. **CFD_HANDOFF/CFD_PROTOCOL_FOR_AVA.md** and **CFD_HANDOFF/CFD_CASE_MANIFEST.json** — proposed CFD plan for coauthor review and hash-pinned input geometry/polars.
5. **EVIDENCE/** — strict direct-XFOIL summary, baseline comparison, numerical reconciliation and audit.

Key facts to check:
- ML held-out R²: CL 0.989, logCD 0.922, CM 0.918; in-screen acceptance 66.1% and conditional-on-selection empirical coverage, not a formal certification.
- Direct XFOIL at one Re/Mach/Ncrit: 18.7% lower NOMINAL weighted drag for a selected within-screen design versus NACA2412, and 8.8% lower marginal means over differently filtered completed runs; 10 valid paired perturbations.
- Empirical 95% CVaR from 13 cases degenerates to the largest observation; no robust population-tail improvement is established. The within-screen sample maximum is higher than reference.
- Ten-seed NSGA-II agreement 0.283 > 0.20, therefore front not repeatability-certified.
- Manufacturing deviation amplitude is PRESCRIBED, not a measured production distribution.
- CFD/experimental aerodynamic verification HAS NOT BEEN PERFORMED. Ava's independent viscous RANS work remains next.

The package includes **CFD_HANDOFF_ONLY.zip** for selectively sharing just inputs/protocol with Ava; it is not a simulation result.

Do not submit the manuscript before the authors approve the claims and the selected CFD study is integrated (or agree explicitly to a narrower XFOIL-only paper). Verify any prior journal resubmission restrictions, chosen venue's rules, figure formatting and generative-AI declaration.

File integrity: SHA256_MANIFEST.json records size and SHA-256 for every included file. Earlier archival packages remain untouched in the parent OneDrive paper directories. OneDrive local presence does not prove completed cloud sync.


## 14 October: Additional ML and XFOIL evidence

Two separately scrambled 128-scenario 0.5%-chord XFOIL studies give 92 and 96 strictly completed matched cases and paired mean drag reductions of 15.4% / 15.8%. These results are conditional on solver completion; no independent CFD evidence or unconditional tail guarantee. A frozen-model audit also finds historical acceptance was effectively support-distance-driven, not independently gated by disagreement. See ML_DIAGNOSTICS and the supplement in manuscript.
