# Independent final manuscript and reference integrity audit
21 October 2026 · Controlled Robust V2 author-review revision

## Audit judgement

**The complete numerical manuscript and standalone Word draft are ready for coauthor review.** The paper is not represented as a finished independent aerodynamic CFD validation or an approved journal submission. Its experimental narrative was rewritten from the original author's own 2025 Word submission, not simply copyedited from the October computational audit.

### Source of writing style

The exact original is `C:\Users\Ben\OneDrive\paper\prior_final_draft.docx`. Its SHA-256 is `cd6807e1b4211761eaa3b055c6461b1ba6353585bd347364d87e89f3f1b98125`. Original structure: 340 paragraphs, 8 tables, 20 inline figures and 51 identified headings. Its recurrent features are engineering-motivation opening, first-person-plural methodological exposition, mathematical geometry/data definitions, explanatory transitions and explicitly numbered contributions. The new paper follows this order and voice while correcting typographic errors, repetitions, unsupported generalisations, old 48-point lift-curve claims and implied CFD equivalence. Preserved originals and both pre-rewrite manuscript/Word versions are in this audit directory; only the new full manuscript is installed under `paper/`.

### Numerical-science verification

**46 source-bound claims independently reconciled against frozen experiment receipts**, with checks on requested and observed states, geometry identity, protected split counts, trust/coverage, ensemble errors, calibration and confirmation high-error ranking, seed-completion, numerical front agreement, fresh scenario results, and strict selected solver attempts. The ledger `NEW_MANUSCRIPT_NUMERIC_CLAIM_AUDIT.json` lists the exact source pathname and SHA-256 supporting each claim.

Critical adverse evidence is stated prominently and unaltered:
- 106,931 of 141,928 controlled XFOIL states admitted, leaving 34,997 missing; only 44 complete requested polars.
- The sealed 200-geometry confirmation has 17,968/22,600 admissible states; the disagreement-only risk AUROC (0.821) exceeds the combined score (0.802), notwithstanding its gain relative to support alone (0.535).
- All ten robust seeds and ten nominal-control seeds complete the original 128 × 200 campaigns; the robust mean–CVaR95 objectives remain distinct from nominal drag and passing the two numerical agreement measures is not global optimality.
- Exactly 290/819 strict selected XFOIL cases are usable and 529 fail or are rejected. Zero complete matched *nominal* pairs exist. The small positive paired reductions under 0.5% chord disturbances are explicitly conditional on their successful-case denominators. The unestimable 1% knee comparison is not filled with the historical model.
- The proposed perturbation amplitudes are not measured manufacturing tolerances. The held-out historical geometry identities have been inspected previously. Both new labels and numerical confirmation use the same local XFOIL family.

### Bibliographic verification — all 29 cited sources

Every bibliography entry maps to an identifiable real publication, publisher record, official conference paper, official source dataset or author repository. The independent sources and claim-appropriate scope are recorded in `REFERENCES_PRIMARY_VERIFIED_V2.csv` and `REFERENCES_PRIMARY_VERIFIED_V2.json`; all 29 also have actual in-text citations (including expanded numeric ranges) and no out-of-range reference numbers. DOI metadata was independently re-queried live rather than treating the earlier audit as proof.

One **true metadata conflict** was found and resolved without inventing a title: Crossref's indexing for Kulfan & Bussoletti (2006), DOI 10.2514/6.2006-6948, writes *Parameteric*. The authors' original AIAA paper itself writes *Parametric*, so the manuscript follows the primary author PDF rather than reproducing the Crossref typographical error. The accessible original is `https://www.brendakulfan.com/_files/ugd/169bff_793df02d77364418af1c3b67133bbbbb.pdf`. The 1959 Dover republication of Abbott and von Doenhoff is verified by its own copyright page, not inferred from a later retailer; the 2017 sixth edition of Anderson is confirmed by edition-specific Google Books/WorldCat catalogue entries. Official publishers/arXiv/conference pages support the non-DOI literature, and Airfoil Tools is used only for historical polar-source attribution. None of these references is misrepresented as independently validating the current numerical outcome.

Novelty assertions are bounded by prior engineering-optimisation and CFD work (Li, Shahrokhi and Jahangirian), Kulfan's CST, Deb's NSGA-II, Rockafellar–Uryasev CVaR, the ensemble/calibration literature and Sharpe–Hansman NeuralFoil. We claim the *verified numerical integration and provenance*, not invented algorithmic priority.

### Paper structure and visual acceptance

The complete new manuscript contains about 9,244 words, abstract, nomenclature, introduction, explicit contributions, CST and source-data methodology, model and risk experiments, uncertainty and nominal/robust optimisation definitions, quantitative tables, numerical results, full discussion and limitations, an executable independent CFD protocol, numerically supported present findings, authorship/disclosure draft sections and all 29 references. The Word document contains **seven tables** (nomenclature and Tables 1–6) and **eight embedded figures** (seven numbered and one supplementary).

All **21 pages** of the latest browser-derived DOCX-content PDF preview were inspected in four contact sheets; the second render explicitly keeps figure captions adjacent to the correct images and begins the figure section cleanly. Content, figure identities, table readability and page flow are confirmed in `VISUAL_QA_FULL_AUTHOR_STYLE.json`. The page preview is a Chromium reconstruction of the actual Word content, *not native Microsoft Word pagination*. The Word file itself has explicit keep-with-next caption and keep-together image formatting, but native-Word print pagination should also be checked by the authors before submission.

### Scientific work deliberately left open

The **only unavailable independent physical/numerical study** is the planned separate viscous CFD programme and the consequent final physical aerodynamic comparison. Section 7 defines fixed shapes, equal achieved lift, moments, domain/grid/transition modelling, convergence, matched geometric disturbances and a failure-aware numerical case ledger. Section 8 contains the final supported **numerical** findings but explicitly withholds a final **physical design superiority** conclusion until independent CFD exists.

Coauthors must personally approve institutional affiliations, roles, conflict/funding declarations, AI-use disclosure and final journal-specific choices. Those non-delegable statements are left as clearly marked author decisions rather than falsely treated as experimental defects or silently approved on their behalf.

### Release safeguards

No source geometry, label dataset, frozen model, selected design, NSGA-II seed or raw XFOIL result was re-run or changed to support the rewritten narrative. The old manuscript, Word file and raw solver/archive SHA-256 evidence remain intact. Final software and reproducibility tests must be re-run on the completed manuscript revision before normal branch push and re-packaging the canonical OneDrive author handoff.
