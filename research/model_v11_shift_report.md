# model_v11_shift zero-base screen

## Decision

The unadjusted point winner was S08_ENVIRONMENT_RESIDUALISED_K1, but no policy passed every preregistered return, familywise, stability, tail, coverage, concentration, mutation, and independent-P&L gate. No forward-shadow finalist was frozen and production remains unchanged.

This is a retrospective mechanism screen on a panel already available to the wider project. A retrospective passer could freeze only one exact forward-shadow specification; it cannot authorise production.

## All preregistered candidate policies

| Policy | Net20 | Net40 | Net60 | Months + | Slice min | Top20 removed | ES05 net40 | Exec frac | Days | Codes | FW95 lower uplift | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S08_ENVIRONMENT_RESIDUALISED_K1 | +0.2728% | +0.0766% | -0.1196% | 8/13 | -0.0779% | -0.0706% | -3.9585% | 0.981 | 261 | 113 | -0.0529% | reject |
| S08_ENVIRONMENT_RESIDUALISED_K2 | +0.2243% | +0.0277% | -0.1689% | 8/13 | -0.0083% | -0.0389% | -2.9622% | 0.983 | 266 | 201 | -0.0837% | reject |
| S05_LEAVE_ONE_STATE_OUT_MEDIAN_COEF_K1 | +0.2215% | +0.0245% | -0.1725% | 6/13 | -0.1512% | -0.1389% | -4.1845% | 0.985 | 262 | 114 | -0.1736% | reject |
| S05_LEAVE_ONE_STATE_OUT_MEDIAN_COEF_K2 | +0.1985% | +0.0018% | -0.1948% | 5/13 | -0.0635% | -0.0693% | -3.1731% | 0.983 | 266 | 203 | -0.1898% | reject |
| S06_UNSUPERVISED_LATENT_EXPERTS_K1 | +0.0000% | +0.0000% | +0.0000% | 0/13 | +0.0000% | +0.0000% | +0.0000% | 0.000 | 0 | 0 | -0.3521% | reject |
| S06_UNSUPERVISED_LATENT_EXPERTS_K2 | +0.0000% | +0.0000% | +0.0000% | 0/13 | +0.0000% | +0.0000% | +0.0000% | 0.000 | 0 | 0 | -0.2828% | reject |
| S04_ERA_INVARIANT_SIGN_K2 | +0.1882% | -0.0084% | -0.2050% | 6/13 | -0.0387% | -0.0832% | -2.9062% | 0.983 | 266 | 205 | -0.2228% | reject |
| S03_WORST_MONTH_GROUP_DRO_K2 | +0.0939% | -0.1001% | -0.2941% | 6/13 | -0.2148% | -0.1663% | -3.9535% | 0.970 | 266 | 178 | -0.4249% | reject |
| S04_ERA_INVARIANT_SIGN_K1 | +0.0587% | -0.1376% | -0.3338% | 6/13 | -0.2357% | -0.2247% | -4.1568% | 0.981 | 261 | 107 | -0.3874% | reject |
| S01_DENSITY_RATIO_RECENT_K1 | +0.0320% | -0.1650% | -0.3620% | 4/13 | -0.3256% | -0.3030% | -4.8766% | 0.985 | 262 | 129 | -0.4235% | reject |
| S01_DENSITY_RATIO_RECENT_K2 | -0.0219% | -0.2193% | -0.4166% | 3/13 | -0.3011% | -0.2442% | -4.1632% | 0.987 | 266 | 230 | -0.5176% | reject |
| S02_COVARIATE_CHANGEPOINT_RESET_K1 | -0.0439% | -0.2213% | -0.3988% | 4/13 | -0.4873% | -0.3710% | -5.2370% | 0.887 | 236 | 123 | -0.6649% | reject |
| S03_WORST_MONTH_GROUP_DRO_K1 | -0.0361% | -0.2286% | -0.4211% | 4/13 | -0.2941% | -0.3962% | -6.0963% | 0.962 | 256 | 103 | -0.6057% | reject |
| S02_COVARIATE_CHANGEPOINT_RESET_K2 | -0.0656% | -0.2445% | -0.4235% | 2/13 | -0.5186% | -0.2989% | -4.1793% | 0.895 | 244 | 227 | -0.5891% | reject |
| S07_RECENT_SUPPORT_CONFORMAL_K2 | -0.0490% | -0.2482% | -0.4475% | 0/13 | -0.4422% | -0.2576% | -3.2120% | 0.996 | 266 | 258 | -0.5690% | reject |
| S07_RECENT_SUPPORT_CONFORMAL_K1 | -0.0940% | -0.2932% | -0.4924% | 2/13 | -0.4614% | -0.3325% | -4.0924% | 0.996 | 265 | 155 | -0.7331% | reject |

## Capacity-matched controls

| Policy | Net20 | Net40 | Net60 | Months + | Top20 removed | ES05 net40 | Codes |
|---|---:|---:|---:|---:|---:|---:|---:|
| C00_DAILY_RANK_RIDGE_K1 | +0.2049% | +0.0079% | -0.1891% | 7/13 | -0.1370% | -4.0208% | 114 |
| C00_DAILY_RANK_RIDGE_K2 | +0.2553% | +0.0587% | -0.1379% | 8/13 | -0.0054% | -2.6207% | 205 |

## Gate failures by policy

- `S08_ENVIRONMENT_RESIDUALISED_K1`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive
- `S08_ENVIRONMENT_RESIDUALISED_K2`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net40_greater_than_capacity_matched_control, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive
- `S05_LEAVE_ONE_STATE_OUT_MEDIAN_COEF_K1`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive
- `S05_LEAVE_ONE_STATE_OUT_MEDIAN_COEF_K2`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net40_greater_than_capacity_matched_control, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive
- `S06_UNSUPERVISED_LATENT_EXPERTS_K1`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, executed_slot_fraction_at_least_0_3, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, largest_positive_code_pnl_share_at_most_0_25, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive, traded_days_at_least_150, unique_codes_at_least_100
- `S06_UNSUPERVISED_LATENT_EXPERTS_K2`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, executed_slot_fraction_at_least_0_3, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, largest_positive_code_pnl_share_at_most_0_25, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive, traded_days_at_least_150, unique_codes_at_least_100
- `S04_ERA_INVARIANT_SIGN_K2`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive
- `S03_WORST_MONTH_GROUP_DRO_K2`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive
- `S04_ERA_INVARIANT_SIGN_K1`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive
- `S01_DENSITY_RATIO_RECENT_K1`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive
- `S01_DENSITY_RATIO_RECENT_K2`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive
- `S02_COVARIATE_CHANGEPOINT_RESET_K1`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive
- `S03_WORST_MONTH_GROUP_DRO_K1`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive
- `S02_COVARIATE_CHANGEPOINT_RESET_K2`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top10_code_weight_share_at_most_0_25, top_10_positive_pnl_codes_to_cash_net20_positive
- `S07_RECENT_SUPPORT_CONFORMAL_K2`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, largest_code_weight_share_at_most_0_05, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top_10_positive_pnl_codes_to_cash_net20_positive
- `S07_RECENT_SUPPORT_CONFORMAL_K1`: all_three_temporal_slices_net40_positive, best_20_days_removed_net20_positive, expected_shortfall05_net40_not_worse_than_control, familywise_95_lower_net40_uplift_positive, net40_greater_than_capacity_matched_control, net40_mean_positive, net60_mean_positive, positive_months_net40_at_least_10, top_10_positive_pnl_codes_to_cash_net20_positive

## Mechanism rationale and limits

- Han, Huang, and Wang study model assessment and selection under [temporal distribution shift](https://proceedings.mlr.press/v235/han24b.html), including synthesis of current and historical epochs. Their adaptive rolling-window method is not reproduced here: window or candidate selection on these outcomes was explicitly prohibited. The paper supports treating non-stationarity as a model-selection problem, not the profitability of any registered policy.
- Fujii and Yasumura report a Japanese-stock prediction design based on [autoencoder change-point detection](https://www.jstage.jst.go.jp/article/jsaisigtwo/2024/SAI-051/2024_04/_article/-char/ja). It motivates an outcome-free structural-break hypothesis in this domain, but does not validate this protocol's multivariate mean-contrast detector or its trading economics.
- Kim et al. introduce continual causal/uplift tasks under [temporal and domain shifts](https://proceedings.mlr.press/v208/kim23a.html). That work motivates evaluating mechanisms across changing environments; its observational uplift setting is materially different from same-day return ranking.
- Zhou et al. show that [Group-DRO can fail when predefined groups do not capture the relevant spurious correlations](https://arxiv.org/abs/2106.07171). Calendar months are therefore a falsifiable environment definition, not an assumed sufficient partition. A positive or negative S03 result cannot establish causal invariance.

These sources were fixed as mechanism context and limitations. They were not used to change candidates, thresholds, gates, or interpretation after seeing this family's returns.

## Integrity

- Independent panel rejoin and P&L reproduction: exact
- Embedded outcomes match independent panel values: yes
- Target-day outcome mutation: exact
- Monthly expanding folds: 13
- Candidate policies in familywise test: 16

## Production gate

A retrospective passer must remain unchanged for at least 60 new sessions and pass the preregistered forward mean, block-bootstrap, three-slice, tail-removal, and independent-audit checks. No candidate in this report is production-authorised.

## Reproduction

```bash
PYTHONPATH=src:. .venv/bin/python research/model_v11_shift_runner.py
PYTHONPATH=src:. .venv/bin/python research/model_v11_shift_audit.py
```

- Panel SHA-256: `6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb`
- Protocol SHA-256: `76d8c4129f57526c6d0a4d826779414c3ee4dcba48e818c72697ba8d3627a0cc`
- Runner SHA-256: `a17ac2b2b8daa31b91dcb345dd95850a94f462ed3c02370c7bf8a0b833b2a80a`
- Evaluation dependency SHA-256: `6c3c60788353a071493ce1c940c3ea71d6d069ae941b37bebffc09237421f70e`
- Picks SHA-256: `f71e7aeeb6f054dd7df58d9d83e809c7441a7fbf458d11f1b6d861f3e062910c`
- Audit SHA-256: `fb24dafb6e04c26f879615125fe25ec7fee94b32bce257d0ac9b0f128bf34316`
