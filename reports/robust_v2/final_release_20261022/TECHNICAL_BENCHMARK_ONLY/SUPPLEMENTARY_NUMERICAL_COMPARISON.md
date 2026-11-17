# Supplement: competitor and tool qualification

## S1. Why the study does not claim a better general-purpose predictor

We compare the exact frozen five-member numerical surrogate to NeuralFoil 0.3.3 xlarge via AeroSandbox 4.2.10. Both predictions are checked against the SAME available XFOIL labels at Re=1e6, Ncrit=9, M=0. The geometry supplied to NeuralFoil is the exact twelve-feature CST contour but is refitted to its native eighteen-parameter Kulfan encoding. Both are pretrained and are not tuned to this comparison.

The internal training-airfoil overlap of NeuralFoil is unknown, so these figures are NOT independent held-out-family performance for that model. Our protected historical shapes were inspected in the earlier audit; the sealed confirmation contains development-shape mixtures.

### Table S1. Entire sealed cohort: 200 shapes, 17,968 paired solver-observed states

| Output | Our MAE | NeuralFoil MAE | 95% geometry-bootstrap interval, NeuralFoil − ours |
|---|---:|---:|---|
| C_L | 0.020610 | 0.009355 | [-0.012404, -0.008826] |
| C_D | 0.001146 | 0.000937 | [-0.000331, -0.000065] |
| log(C_D) | 0.043070 | 0.031413 | [-0.014248, -0.007314] |
| C_M | 0.003019 | 0.001465 | [-0.001700, -0.001165] |

All four numerical errors favour NeuralFoil on the observed cases. The uncertainty intervals resample the already-fitted geometry results, not model retraining, actual CFD, or physically independent experiments.

### Table S2. All actually observed cases at three preselected incidence angles

| Cohort | Shapes | States | Our lift MAE | NF lift MAE | Our drag MAE | NF drag MAE |
|---|---:|---:|---:|---:|---:|---:|
| Protected historical test | 166 | 470 | 0.016705 | 0.004807 | 0.000264 | 0.000161 |
| Sealed mixture confirmation | 191 | 543 | 0.014184 | 0.004259 | 0.000214 | 0.000112 |

The incidence values 2°, 4° and 6° were fixed before competitor inference; all 357 geometries with genuinely observed labels at those incidences were evaluated. There was no favourable-outcome filtering of neural results.

## S2. Predicting numerical XFOIL missingness is not predicting real aerodynamic error

Every one of the 22,600 pre-requested confirmation states is included; 4,632 have no usable local XFOIL label. Risk scores from our model were frozen before those labels; the external pretrained competitor was later run on the same fixed input identities.

| Model's fixed score for missing XFOIL observation | AUROC | Requests scored |
|---|---:|---:|
| NeuralFoil negative analysis_confidence | 0.8874 | 22,600 |
| Our frozen ensemble disagreement | 0.7351 | 22,600 |
| Our frozen support + disagreement | 0.7164 | 22,600 |
| Our frozen support alone | 0.4626 | 22,600 |

NeuralFoil confidence predicts this numerical-solver missingness more strongly than our model's risk features. This is NOT a physical validity probability or an independent CFD error measure. The numerical solver and convergence criteria define the outcome; no missing state was filled with pseudo truth.

## S3. What this tool contributes

Our model is NOT a general-purpose accuracy or runtime improvement over NeuralFoil. The tool's useful and testable contribution is the explicitly source-bound, geometry-consistent prediction and trust workflow, coupled to fail-closed geometry admission and to the fully recorded controlled optimisation experiment. The default CLI now chooses the controlled model and checks its own 74,946 development observations plus frozen artifact hashes. A prior path mistakenly reused historical support data and has been corrected.

The research value therefore rests on failure-aware numerical evidence, provenance, prospective score/label separation, and robust versus nominal search comparison—not an unsubstantiated state-of-the-art predictor.

## S4. Source-bound reproducibility

Benchmark scripts: scripts/robust_v2/benchmark_controlled_neuralfoil.py, scripts/robust_v2/benchmark_full_confirmation_neuralfoil.py, and scripts/robust_v2/audit_neuralfoil_comparison.py. They retain every attempted case and source input SHA-256. This is a post hoc competitor comparison, not a redesign or retuning of the earlier sealed confirmation.

Full confirmation result SHA-256: fe873dbc8e1919816a27ed958b09f2b9827fc7c94c5277715a5dad65c750d75d.
Three-incidence benchmark result SHA-256: ae485aa237e19e59dee91a8ebf5d249bccd25f06ccf239690fdd60beda883770.
Independent paired bootstrap audit SHA-256: c70a98912a24d8a2372337d2a85b6c298b7f0f145c753ed053f6af7304c574bb.

Independent viscous CFD, actual manufacturing distribution, and broad new-family performance remain outside the current numerical evidence.
