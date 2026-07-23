# v1.1 causal/uplift zero-base experiment

## Decision

- Retrospective decision: **REJECT_ALL_UPLIFT_HYPOTHESES**
- Production ready: **false**
- Strict scheduled sessions: **88**
- Global familywise reality-check p: **0.9224**

The eight mechanisms estimate an event-versus-no-event counterfactual. They are structurally separate from v10 title-value regression and v11 historical-title analog retrieval. The panel is retrospective, so even a passing mechanism could only become a forward-shadow candidate.

## Locked hypotheses

| Policy | net20 | net40 | net60 | days | slots | max-t L95 | reality p | gate |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `U01_MATCHED_NO_EVENT_K1` | -0.0263% | -0.1376% | -0.2490% | 49 | 49 | -0.4550% | 1.0000 | FAIL |
| `U01_MATCHED_NO_EVENT_K2` | +0.0416% | -0.0368% | -0.1152% | 49 | 69 | -0.2678% | 1.0000 | FAIL |
| `U02_FAMILY_T_LEARNER_K1` | +0.1153% | -0.0529% | -0.2211% | 74 | 74 | -0.5949% | 1.0000 | FAIL |
| `U02_FAMILY_T_LEARNER_K2` | -0.1377% | -0.2877% | -0.4377% | 74 | 132 | -0.6719% | 1.0000 | FAIL |
| `U03_AIPW_CATE_K1` | +0.0214% | -0.1786% | -0.3786% | 88 | 88 | -0.6541% | 1.0000 | FAIL |
| `U03_AIPW_CATE_K2` | +0.0071% | -0.1929% | -0.3929% | 88 | 176 | -0.5700% | 1.0000 | FAIL |
| `U04_DATE_RESIDUAL_UPLIFT_K1` | +0.0689% | -0.0561% | -0.1811% | 55 | 55 | -0.4231% | 1.0000 | FAIL |
| `U04_DATE_RESIDUAL_UPLIFT_K2` | +0.1650% | +0.0706% | -0.0237% | 55 | 83 | -0.1906% | 0.9224 | FAIL |
| `U05_ISSUER_SELF_CONTROL_K1` | -0.2928% | -0.4928% | -0.6928% | 88 | 88 | -1.0841% | 1.0000 | FAIL |
| `U05_ISSUER_SELF_CONTROL_K2` | -0.3299% | -0.5242% | -0.7185% | 88 | 171 | -0.9173% | 1.0000 | FAIL |
| `U06_PROPENSITY_OVERLAP_T_K1` | +0.0000% | +0.0000% | +0.0000% | 0 | 0 | +0.0000% | 1.0000 | FAIL |
| `U06_PROPENSITY_OVERLAP_T_K2` | +0.0000% | +0.0000% | +0.0000% | 0 | 0 | +0.0000% | 1.0000 | FAIL |
| `U07_POSITIVE_TAIL_UPLIFT_K1` | -0.0262% | -0.0330% | -0.0398% | 3 | 3 | -0.1018% | 1.0000 | FAIL |
| `U07_POSITIVE_TAIL_UPLIFT_K2` | -0.0131% | -0.0165% | -0.0199% | 3 | 3 | -0.0509% | 1.0000 | FAIL |
| `U08_HONEST_CROSSFIT_POLICY_TREE_K1` | -0.2141% | -0.3914% | -0.5686% | 78 | 78 | -1.2128% | 1.0000 | FAIL |
| `U08_HONEST_CROSSFIT_POLICY_TREE_K2` | -0.3008% | -0.4667% | -0.6326% | 78 | 146 | -0.9802% | 1.0000 | FAIL |

## Why the point-estimate leader is rejected

`U04_DATE_RESIDUAL_UPLIFT_K2` had the highest net40 point estimate, but failed: `net60_mean_positive`, `familywise_95_lower_net40_mean_positive`, `familywise_reality_check_p_at_most_0_10`, `all_three_temporal_slices_net40_positive`, `best_20_days_removed_net20_positive`, `top_10_positive_pnl_codes_to_cash_net20_positive`.

## Integrity

- Input/protocol/PIT/mutation: **PASS**
- Independent P&L reconstruction: **PASS**
- Byte-identical runner rerun: **PASS**

No threshold was retuned after observing results. No candidate is frozen because no policy passes the preregistered retrospective gate.
