# v1.1 calendar and institutional-date zero-base experiment

## Decision

- Retrospective decision: **REJECT_ALL_CALENDAR_HYPOTHESES**
- Production ready: **false**
- Scheduled sessions: **266**
- Strict TDnet score sessions: **88**
- Global reality-check p: **0.8376**

Every weekday, holiday, turn-of-month, quarter, fiscal, SQ, reporting-window, release-clock and density definition was frozen before execution. No calendar window or posterior strength was altered after reading returns.

## Results

| Policy | net20 | net40 | net60 | days | codes | max-t L95 | reality p | gate |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `C01_WEEKDAY_ISSUER_EB_K1` | -0.1427% | -0.3389% | -0.5352% | 261 | 50 | -0.7494% | 1.0000 | FAIL |
| `C01_WEEKDAY_ISSUER_EB_K2` | -0.1205% | -0.3119% | -0.5033% | 261 | 88 | -0.6162% | 1.0000 | FAIL |
| `C02_POST_HOLIDAY_ISSUER_EB_K1` | +0.0146% | -0.0282% | -0.0711% | 57 | 9 | -0.2505% | 1.0000 | FAIL |
| `C02_POST_HOLIDAY_ISSUER_EB_K2` | +0.0350% | -0.0078% | -0.0507% | 57 | 19 | -0.1473% | 1.0000 | FAIL |
| `C03_PRE_HOLIDAY_ISSUER_EB_K1` | +0.1003% | +0.0582% | +0.0161% | 56 | 9 | -0.2508% | 0.8376 | FAIL |
| `C03_PRE_HOLIDAY_ISSUER_EB_K2` | +0.0462% | +0.0040% | -0.0381% | 56 | 15 | -0.1589% | 0.9976 | FAIL |
| `C04_MONTH_START_ISSUER_EB_K1` | -0.0601% | -0.0804% | -0.1007% | 27 | 3 | -0.2367% | 1.0000 | FAIL |
| `C04_MONTH_START_ISSUER_EB_K2` | -0.0827% | -0.0973% | -0.1120% | 27 | 6 | -0.2222% | 1.0000 | FAIL |
| `C05_MONTH_END_ISSUER_EB_K1` | -0.0204% | -0.0498% | -0.0791% | 39 | 10 | -0.1896% | 1.0000 | FAIL |
| `C05_MONTH_END_ISSUER_EB_K2` | +0.0153% | -0.0141% | -0.0434% | 39 | 16 | -0.1177% | 1.0000 | FAIL |
| `C06_TURN_MONTH_DIFFERENTIAL_EB_K1` | -0.1458% | -0.1842% | -0.2225% | 51 | 8 | -0.3851% | 1.0000 | FAIL |
| `C06_TURN_MONTH_DIFFERENTIAL_EB_K2` | -0.0901% | -0.1232% | -0.1563% | 51 | 12 | -0.2733% | 1.0000 | FAIL |
| `C07_QUARTER_END_ISSUER_EB_K1` | +0.0312% | +0.0162% | +0.0012% | 20 | 2 | -0.0811% | 0.9902 | FAIL |
| `C07_QUARTER_END_ISSUER_EB_K2` | +0.0193% | +0.0043% | -0.0108% | 20 | 6 | -0.0732% | 0.9972 | FAIL |
| `C08_FISCAL_HALF_YEAR_ISSUER_EB_K1` | -0.0166% | -0.0204% | -0.0241% | 5 | 1 | -0.0481% | 1.0000 | FAIL |
| `C08_FISCAL_HALF_YEAR_ISSUER_EB_K2` | -0.0123% | -0.0161% | -0.0198% | 5 | 2 | -0.0413% | 1.0000 | FAIL |
| `C09_SQ_PROXY_ISSUER_EB_K1` | +0.0251% | +0.0153% | +0.0055% | 13 | 6 | -0.1272% | 0.9916 | FAIL |
| `C09_SQ_PROXY_ISSUER_EB_K2` | +0.0167% | +0.0069% | -0.0029% | 13 | 15 | -0.0818% | 0.9958 | FAIL |
| `C10_EARNINGS_SEASON_TDNET_FAMILY_K1` | -0.0001% | -0.0009% | -0.0016% | 1 | 1 | -0.0031% | 1.0000 | FAIL |
| `C10_EARNINGS_SEASON_TDNET_FAMILY_K2` | -0.0001% | -0.0004% | -0.0008% | 1 | 1 | -0.0016% | 1.0000 | FAIL |
| `C11_RELEASE_CLOCK_FAMILY_EB_K1` | +0.0000% | +0.0000% | +0.0000% | 0 | 0 | +0.0000% | 1.0000 | FAIL |
| `C11_RELEASE_CLOCK_FAMILY_EB_K2` | +0.0000% | +0.0000% | +0.0000% | 0 | 0 | +0.0000% | 1.0000 | FAIL |
| `C12_HIGH_DENSITY_FAMILY_EB_K1` | +0.0000% | +0.0000% | +0.0000% | 0 | 0 | +0.0000% | 1.0000 | FAIL |
| `C12_HIGH_DENSITY_FAMILY_EB_K2` | +0.0000% | +0.0000% | +0.0000% | 0 | 0 | +0.0000% | 1.0000 | FAIL |

## Point-estimate leader

`C03_PRE_HOLIDAY_ISSUER_EB_K1` is rejected because it failed: `familywise_95_lower_net40_mean_positive`, `familywise_reality_check_p_at_most_0_10`, `all_three_temporal_slices_net40_positive`, `positive_months_at_least_8`, `best_20_days_removed_net20_positive`, `top_10_positive_pnl_codes_to_cash_net20_positive`, `traded_days_at_least_80`, `executed_slot_fraction_at_least_0_3`, `unique_codes_at_least_100`, `largest_code_weight_share_at_most_0_05`, `top10_code_weight_share_at_most_0_25`, `largest_positive_code_pnl_share_at_most_0_25`.

## Integrity

- Protocol/input/PIT/mutation: **PASS**
- Independent outcome join and P&L: **PASS**
- Byte-identical rerun: **PASS**

## Research scope warning

The cited Japanese calendar studies are old, mixed, and largely evaluate index or close-to-close returns. One pre-holiday re-examination found that most holiday effects did not persist outside narrower Golden Week/afternoon settings, and a later sample reported disappearance of monthly effects. Accordingly this experiment treated every mechanism as falsifiable and did not transfer published index effects directly to individual-stock O-C returns.

- https://doi.org/10.14490/jjss.34.129
- https://doi.org/10.1016/0922-1425(91)90001-S
- https://doi.org/10.1016/0927-538X(94)00026-4
- https://doi.org/10.1016/S0922-1425(01)00067-6

No candidate passes every gate, so no forward-shadow finalist is frozen.
