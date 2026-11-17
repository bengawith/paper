# Robust V2 — final offline author review (14 October 2026)

**Read this first.** This is the current paper and evidence for Ben's review and Ava's selected CFD stage. It is **not** a claim of accepted journal publication, independent CFD results or globally optimal robust design.

## The standalone paper to send Ava

**`LATEST_MANUSCRIPT_FOR_AVA.docx`** is the current manuscript alone. It contains the latest abstract, methods, numerical results, seven main figures, one supplementary ML figure, tables, limitations and 28 references. It does **not** contain the old paper, previous submissions, revision commentary or this review report. Open it and review before emailing Ava.

`HISTORICAL/prior_final_draft.docx` is the separate V1 tone reference. Historical source, Git-history, and frozen-model/data archives are also included under `HISTORICAL/`, explicitly **not** presented as the latest code. The latest code comes from the Git branch whose commit is recorded in this package’s manifest. None of the historical material is merged into the Ava-ready manuscript.

## Recommended reading order

1. `LATEST_MANUSCRIPT_FOR_AVA.docx` — standalone latest draft.
2. `SCIENTIFIC_REVIEW_REPORT.docx` and `EVIDENCE/FINAL_RESEARCH_RESOLUTION.md` — what has been independently checked and what remains scientifically incomplete.
3. `EVIDENCE/TRUST_INCREMENTAL_CV.json` and `EVIDENCE/CORRECTED_SELECTED_DESIGNS.json` — the new ML/optimisation audit.
4. `EVIDENCE/CORRECTED_SMOKE_FRONT.json` — the ten-seed *low-budget* corrected Pareto diagnostic (not production optimality).
5. `CFD_HANDOFF/CFD_PROTOCOL_FOR_AVA.md`, `CFD_HANDOFF/CFD_CASE_MANIFEST.json` and `CFD_HANDOFF_ONLY.zip` — exact input geometry, numerical comparisons and suggested CFD verification.
6. `EMAIL_TO_AVA_DRAFT.txt` — friendly email to review and send **only with your permission**.
7. `COPY_TO_LAPTOP_OFFLINE.ps1` — fail-closed single-command offline copy from inside your laptop's existing Git repository.

## Scientific conclusions that are supportable today

- On 26,692 geometry-grouped held-out **XFOIL-labelled** states, the fixed MLP predicts CL, log-CD, CM with R² approximately 0.989/0.922/0.918. XFOIL is numerical training truth, not experiment.
- The historical trust acceptance rule rejects one-third of states predominantly on geometry support. A separate **calibration-only**, five-fold group cross-validation finds higher high-error discrimination from adding ensemble disagreement: AUROC 0.767 to 0.866, gain 0.099 [0.069, 0.139] from 300 airfoil-group bootstrap resamples. This is not a newly final-test-validated operational gate.
- A selected design has 18.7% lower nominal local-XFOIL weighted drag than NACA 2412. Two 128-scenario XFOIL numerical checks at prescribed 0.5%-chord amplitude yield 92 and 96 fully matched, converged comparisons with about 15.4%/15.8% lower paired mean drag. These comparisons are conditional on solver success; physical manufacturing tail-risk probability and CFD confirmation remain unknown.
- A material optimisation-physics bug was corrected: each candidate now receives its own surface-normal uncertainty field with shared latent scenarios. Prior optimisation objective values and front claims are superseded.
- The corrected ten-seed **smoke** uses population 32, 10 generations and 16 scenarios. All seeds yielded feasible designs, with mean symmetric front distance 0.149 versus the unchanged 0.20 rule. **This is not the required full 128/200/64/10 production acceptance.**
- The geometric deviation field is prescribed and piecewise-linear/continuous after correlated controls. No actual process-metrology tolerance distribution has been demonstrated.

## What Ava will still need to establish

A different viscous solver with a documented mesh/domain study, consistent transition/turbulence assumptions and lift-matched comparisons. Do not claim an independently CFD-verified or robust Pareto-optimal design before this work and the corrected production optimiser are completed.

## Integrity and offline access

The canonical folder is `C:\Users\Ben\OneDrive\paper\READY_FOR_REVIEW` on SamsPC. `SHA256_MANIFEST.json` lists every included file and verifies the package's exact Git source commit. Original frozen research archives are retained separately in the parent OneDrive folder and not silently merged.

The laptop copy script refuses a dirty working tree, verifies the branch against the package, and hashes source/destination files. Sync OneDrive to your laptop before departing. Local file presence on SamsPC does not by itself prove OneDrive cloud sync completed.
