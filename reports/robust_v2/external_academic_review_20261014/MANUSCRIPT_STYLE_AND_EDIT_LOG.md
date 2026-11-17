# Manuscript style and edit record

## Style reference

The separate `prior_final_draft.docx` was read for tone and structure: methods-first paragraphs, practical aerodynamic motivation, compact parameter tables and direct explanation of modelling choices. Its obsolete Physics Score, fixed-grid padding, family-held-out claims and high-fidelity wording were not copied into the revision. V1 remains a separate historical document, never merged into the current paper.

## Integrated scientific edits

The draft title is narrowed to empirical reliability screening and selected XFOIL comparisons. This is a proposed coauthor revision, not an accepted journal title. Abstract, highlights, data methods, results, discussion and conclusion now disclose the 1,001-feature-vector coordinate defect and the absence of a rebuilt model. Historical versus current source/evidence are distinguished throughout.

The architecture is corrected to an input projection plus two two-affine residual blocks, with piecewise-differentiable ReLU. Lift coverage rounding and geometry-cluster OOF values are corrected. The manufacturing field is described as prescribed, correlated and piecewise linearly interpolated, not measured or spline-based. The current seven invalid knee refits and its worse mean objective are integrated, rather than retaining a superseded favourable value. Post-hoc smoke normalisation no longer supports acceptance language.

The fixed-coordinate XFOIL positives remain, with matched/marginal denominators, failed-solver selection, the adverse marginal maximum and same-solver limitation. The converged adversarial figure is separated from the overall largest discrepancy. Existing reviewer responses are treated as a historical-response working draft, not a new venue endorsement. Web dates and broken primary links are corrected; NeuralFoil is added as relevant prior work. Exact unresolved book editions are recorded, not invented.

## Presentation edits

Fix the figure legend over panel (b), literal escaped newline, overclaiming candidate labels and overlapping selective-coverage labels. The Word export retains one standalone paper, eight figures, seven tables and its own references. Review instructions, claim matrix, source manifests, V1 and CFD workflow remain separate. Final rendered-page findings are recorded in `VISUAL_QA.json`.

## Preservation and responsibility

The pre-review manuscript, Word document, reviewer response and paper README are retained under `BEFORE_REVIEW`; the base Git commit provides the original source. Source edits have RED/GREEN evidence in the review logs and the final Git diff. The final source-to-document hash map records exact outputs. The authors must approve the proposed title, wording, authorship, funding/conflict/AI disclosures and any submission; this review cannot attest personal declarations on their behalf.
