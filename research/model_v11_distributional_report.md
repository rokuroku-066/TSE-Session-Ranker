# model_v11_distributional zero-base screen

## Decision

The unadjusted point winner was D08_SLOTWISE_CASH_STOP_K1, but no policy passed every preregistered retrospective return, familywise, stability, tail, coverage, concentration, mutation, and independent-P&L gate. No forward-shadow finalist was frozen and production remains unchanged.

This panel is retrospective and previously available to the project. Passing the retrospective gate can freeze only one exact forward-shadow candidate; it cannot change production.

## All preregistered candidate policies

| Policy | Net20 | Net40 | Net60 | Months + | Slice min | Top20 removed | ES05 net40 | Exec frac | Days | Codes | FW95 lower uplift | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D08_SLOTWISE_CASH_STOP_K1 | +0.2139% | +0.1184% | +0.0229% | 9/13 | -0.0364% | -0.1031% | -2.5163% | 0.477 | 127 | 67 | -0.1341% | reject |
| D08_SLOTWISE_CASH_STOP_K2 | +0.1453% | +0.0683% | -0.0088% | 8/13 | +0.0109% | -0.0223% | -1.3610% | 0.385 | 167 | 104 | -0.1648% | reject |
| D04_NORMALISED_CONFORMAL_LCB_K1 | +0.0072% | +0.0065% | +0.0057% | 1/13 | +0.0000% | +0.0000% | +0.0000% | 0.004 | 1 | 1 | -0.3186% | reject |
| D04_NORMALISED_CONFORMAL_LCB_K2 | +0.0033% | +0.0026% | +0.0018% | 1/13 | +0.0000% | +0.0000% | +0.0000% | 0.004 | 1 | 2 | -0.2609% | reject |
| D07_REGIME_ABSTAIN_K1 | +0.0000% | +0.0000% | +0.0000% | 0/13 | +0.0000% | +0.0000% | +0.0000% | 0.000 | 0 | 0 | -0.3234% | reject |
| D07_REGIME_ABSTAIN_K2 | +0.0000% | +0.0000% | +0.0000% | 0/13 | +0.0000% | +0.0000% | +0.0000% | 0.000 | 0 | 0 | -0.2641% | reject |
| D05_FOREST_TREE_LCB_K1 | +0.0344% | -0.0257% | -0.0859% | 4/13 | -0.1502% | -0.3002% | -3.6660% | 0.301 | 80 | 73 | -0.3897% | reject |
| D05_FOREST_TREE_LCB_K2 | -0.0110% | -0.0568% | -0.1027% | 4/13 | -0.0920% | -0.2015% | -2.5908% | 0.229 | 80 | 110 | -0.3624% | reject |
| D01_CODE_EB_RESIDUAL_K1 | +0.1400% | -0.0593% | -0.2585% | 6/13 | -0.1897% | -0.4810% | -7.1259% | 0.996 | 265 | 100 | -0.5555% | reject |
| D06_EVT_RESIDUAL_ES_K2 | -0.1671% | -0.1844% | -0.2017% | 1/13 | -0.4023% | -0.2154% | -3.6540% | 0.086 | 30 | 40 | -0.5088% | reject |
| D06_EVT_RESIDUAL_ES_K1 | -0.2260% | -0.2486% | -0.2711% | 0/13 | -0.4240% | -0.3035% | -5.1665% | 0.113 | 30 | 24 | -0.6128% | reject |
| D01_CODE_EB_RESIDUAL_K2 | -0.1215% | -0.3200% | -0.5185% | 1/13 | -0.4300% | -0.5140% | -5.1924% | 0.992 | 266 | 161 | -0.7632% | reject |
| D03_DOWNSIDE_FEASIBLE_MEAN_K2 | -0.2735% | -0.4724% | -0.6712% | 2/13 | -0.8626% | -0.6611% | -5.8551% | 0.994 | 266 | 407 | -0.9143% | reject |
| D02_SURVIVAL_INTEGRAL_K1 | -0.4104% | -0.6089% | -0.8074% | 1/13 | -0.7733% | -0.8478% | -7.7231% | 0.992 | 264 | 181 | -1.1542% | reject |
| D02_SURVIVAL_INTEGRAL_K2 | -0.4156% | -0.6141% | -0.8126% | 0/13 | -0.8832% | -0.7350% | -5.7443% | 0.992 | 266 | 339 | -0.9989% | reject |
| D03_DOWNSIDE_FEASIBLE_MEAN_K1 | -0.4490% | -0.6483% | -0.8475% | 4/13 | -1.1474% | -1.0001% | -8.2713% | 0.996 | 265 | 219 | -1.2353% | reject |

## Capacity-matched controls

| Policy | Net20 | Net40 | Net60 | Months + | Top20 removed | ES05 net40 | Codes |
|---|---:|---:|---:|---:|---:|---:|---:|
| C00_DAILY_RANK_RIDGE_K1 | +0.2049% | +0.0079% | -0.1891% | 7/13 | -0.1370% | -4.0208% | 114 |
| C00_DAILY_RANK_RIDGE_K2 | +0.2553% | +0.0587% | -0.1379% | 8/13 | -0.0054% | -2.6207% | 205 |

## Gate failures by policy

- `D08_SLOTWISE_CASH_STOP_K1`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive, traded_days_at_least_150, unique_codes_at_least_100
- `D08_SLOTWISE_CASH_STOP_K2`: best_20_days_removed_net20_positive, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive
- `D04_NORMALISED_CONFORMAL_LCB_K1`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, executed_slot_fraction_at_least_0_3, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, largest_positive_code_pnl_share_at_most_0_25, net40_greater_than_capacity_matched_control, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive, traded_days_at_least_150, unique_codes_at_least_100
- `D04_NORMALISED_CONFORMAL_LCB_K2`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, executed_slot_fraction_at_least_0_3, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, largest_positive_code_pnl_share_at_most_0_25, net40_greater_than_capacity_matched_control, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive, traded_days_at_least_150, unique_codes_at_least_100
- `D07_REGIME_ABSTAIN_K1`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, executed_slot_fraction_at_least_0_3, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, largest_positive_code_pnl_share_at_most_0_25, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive, traded_days_at_least_150, unique_codes_at_least_100
- `D07_REGIME_ABSTAIN_K2`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, executed_slot_fraction_at_least_0_3, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, largest_positive_code_pnl_share_at_most_0_25, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive, traded_days_at_least_150, unique_codes_at_least_100
- `D05_FOREST_TREE_LCB_K1`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top_10_positive_pnl_codes_to_cash_net20_positive, traded_days_at_least_150, unique_codes_at_least_100
- `D05_FOREST_TREE_LCB_K2`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, executed_slot_fraction_at_least_0_3, familywise_95_lower_net40_uplift_positive, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top_10_positive_pnl_codes_to_cash_net20_positive, traded_days_at_least_150
- `D01_CODE_EB_RESIDUAL_K1`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive
- `D06_EVT_RESIDUAL_ES_K2`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, executed_slot_fraction_at_least_0_3, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive, traded_days_at_least_150, unique_codes_at_least_100
- `D06_EVT_RESIDUAL_ES_K1`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, executed_slot_fraction_at_least_0_3, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, largest_positive_code_pnl_share_at_most_0_25, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive, traded_days_at_least_150, unique_codes_at_least_100
- `D01_CODE_EB_RESIDUAL_K2`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive
- `D03_DOWNSIDE_FEASIBLE_MEAN_K2`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top_10_positive_pnl_codes_to_cash_net20_positive
- `D02_SURVIVAL_INTEGRAL_K1`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top_10_positive_pnl_codes_to_cash_net20_positive
- `D02_SURVIVAL_INTEGRAL_K2`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top_10_positive_pnl_codes_to_cash_net20_positive
- `D03_DOWNSIDE_FEASIBLE_MEAN_K1`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top_10_positive_pnl_codes_to_cash_net20_positive

## Integrity

- Independent panel rejoin and P&L reproduction: exact
- Embedded pick outcomes equal independently joined outcomes: yes
- Target-day outcome mutation invariance: exact
- Monthly expanding folds: 13
- Candidate policies in the familywise test: 16

## Production gate

Even a retrospective passer must remain unchanged for at least 60 new sessions and pass the preregistered forward mean, block-bootstrap, three-slice, tail-removal, and independent-audit checks. No candidate in this report is production-authorised.

## Reproduction

```bash
PYTHONPATH=src:. .venv/bin/python research/model_v11_distributional_runner.py
PYTHONPATH=src:. .venv/bin/python research/model_v11_distributional_audit.py
```

- Panel SHA-256: `6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb`
- Protocol SHA-256: `9e7b98550f3066ee6b6eb8ec0ebdaf8120d211020b7c24be29618786e124e480`
- Runner SHA-256: `6c3c60788353a071493ce1c940c3ea71d6d069ae941b37bebffc09237421f70e`
- Picks SHA-256: `e648d4dd74eabc1d385b277dd78fb41ce23dfbbda0eaa0afd2a7cb94f0ff4f32`
- Audit SHA-256: `984ddcb65a39a4189d675060fb74e297b086689f0e4f531d2f7a1355ef6c8086`
