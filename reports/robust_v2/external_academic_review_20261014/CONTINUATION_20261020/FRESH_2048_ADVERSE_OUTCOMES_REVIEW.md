# Fresh fixed-design 2,048-scenario sensitivity: independently reviewed adverse outcomes

**Status:** source-bound supplementary computational review, not independent flow validation or measured manufacturing uncertainty. This report is a separate diagnostic, not an optimizer rerun or a change to frozen scientific thresholds.

**Exact source:** `results/robust_v2/lineages/robust-v2-controlled-20261020/selected_study/FINAL_2048_SCENARIO_EVALUATION.json`, SHA-256 `6203883c4a0c723e0a0d757314337650be72f12ba556c9947a5b0df0c2a609b5`. The full 20-row report uses one fresh 2,048-point latent scramble for each prescribed amplitude and the same latent values across the five fixed designs. Values are **penalty-inclusive surrogate objectives**, not observed physical drag or XFOIL verification outcomes.

| Prescribed amplitude / chord | Fixed design | Penalised mean weighted C_D | Empirical penalised CVaR95 | Geometry-invalid / 2048 | Trust violations / 2048 | Missing lift roots / 2048 |
|---|---|---:|---:|---:|---:|---:|
| 0.1% | reference | 0.007349 | 0.007402 | 0 | 0 | 0 |
| 0.1% | nominal_control | **0.006267** | 0.011661 | 0 | 180 | 0 |
| 0.1% | risk_knee | 0.006382 | **0.006455** | 0 | 0 | 0 |
| 0.1% | risk_mean | 0.006504 | 0.006602 | 0 | 0 | 0 |
| 0.1% | risk_tail | **0.010122** | **0.012647** | 0 | 1236 | 0 |
| 0.25% | reference | 0.007379 | 0.007639 | 0 | 2 | 0 |
| 0.25% | nominal_control | 0.008008 | 0.012007 | 0 | 733 | 0 |
| 0.25% | risk_knee | 0.006412 | 0.006760 | 0 | 2 | 0 |
| 0.25% | risk_mean | 0.006526 | 0.006772 | 0 | 0 | 0 |
| 0.25% | risk_tail | **0.011246** | **0.012889** | 0 | 1538 | 0 |
| 0.5% | reference | 0.008540 | 0.015771 | 0 | 272 | 0 |
| 0.5% | nominal_control | 0.009969 | 0.012477 | 0 | 1269 | 0 |
| 0.5% | risk_knee | 0.006946 | 0.014433 | 0 | 119 | 0 |
| 0.5% | risk_mean | **0.006719** | 0.009161 | 0 | 28 | 0 |
| 0.5% | risk_tail | **0.012314** | 0.013076 | 0 | 1817 | 0 |
| 1.0% | reference | 0.014520 | 0.018019 | 0 | 1362 | 0 |
| 1.0% | nominal_control | 0.012607 | **0.013409** | 0 | 1823 | 0 |
| 1.0% | risk_knee | 0.010867 | 0.015445 | 0 | 982 | 0 |
| 1.0% | risk_mean | **0.010369** | 0.015538 | 0 | 840 | 0 |
| 1.0% | risk_tail | 0.013586 | **0.013863** | 0 | **1971** | 0 |

**Critical adverse findings that should not be suppressed.** The nominal-only comparator achieves the lowest observed penalty-inclusive *mean* at 0.1% chord (0.006267, slightly better than the knee's 0.006382), but its tail and trust-exit counts are substantially worse: 0.011661 versus 0.006455 CVaR95 and 180 versus zero trust exits. The nominal comparator has 1,823/2,048 empirical trust violations at 1% chord even while its CVaR95 is below the knee's reported objective. This is not a proof of robust physical drag reduction.

The nominally selected **risk-tail finalist is an especially important negative example**. It has 1,236/2,048 trust exits already at 0.1% chord and 1,971/2,048 at 1% chord (approximately 96%). Its penalty-inclusive mean remains worse than the risk-knee at each of the four evaluated amplitudes. A lower recorded CVaR95 on the 1% sensitivity set is therefore not meaningful as independent aerodynamic superiority; it may reflect the chosen trust/penalty construction. State this directly.

The knee does better than the fixed reference in the *penalty-inclusive fresh mean* at all four tested amplitudes; its 1% mean 0.010867 compares with reference 0.014520. Yet at 0.5% amplitude risk-mean is lower than the knee (0.006719 versus 0.006946), and at 1% risk-mean is lower (0.010369 versus 0.010867); the knee is not the unique best mean finalist. At 1% every fixed design has hundreds or thousands of trust exits. None of these rows measures a physical distribution of manufactured aerofoils, and the same-solver selected XFOIL results must be reported separately with their own paired-success denominators and failures.

**Integrity obligations:** retain the original 20 source rows, all adverse rankings and the case counts, with a clear distinction between surrogate/penalty scenario sensitivity, genuine local XFOIL outcomes, and independently executed CFD (not performed). The specified amplitudes are scenario settings, not statistically observed tolerances; an empirical CVaR over these fixed artificial scenarios is not a physically calibrated failure probability. This note does not alter the selected design rule, model, source code, frozen thresholds, or upstream receipts.
