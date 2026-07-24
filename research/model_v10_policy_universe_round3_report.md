# Round 3: H12 incremental concentration audit

The relevant question is whether H12 adds robust value over L4, not merely whether H12 is positive in isolation.

## Matched uplift

- Mean uplift net20: +0.0212%.
- Median uplift net20: +0.0000%.
- Positive/zero uplift days: 33.8% / 34.6%.
- Positive uplift months: 7/13.
- Four blocks: {"B1": -0.03261681158258058, "B2": 0.040831354535484456, "B3": 0.028944360268239922, "B4": 0.048290974828135706}.
- After top 5/10/20 positive uplift days: -0.0299% / -0.0711% / -0.1336%.
- Top 5/10/20 share of all positive uplift: 19.9% / 35.2% / 56.8%.
- Moving-block one-sided 90% lower: -0.0373%.

## Random-split falsification

- Random binary-split null mean: +0.1644%.
- Observed H12: +0.3044%.
- One-sided null p-value: 0.0010.

## Tail comparison

- H12 ES5 / worst day: -2.6219% / -5.7772%.
- L4 ES5 / worst day: -2.4593% / -4.6310%.

## Fixed support checks

- top20_positive_uplift_days_removed_mean_positive: fail
- at_least_8_of_13_monthly_uplifts_positive: fail
- all_four_blocks_nonnegative: fail
- winsor_1_99_uplift_positive: pass
- ordinary_block_one_sided90_lower_positive: fail
- random_split_one_sided_p_below_0_10: pass
- expected_shortfall_not_worse_than_control: fail

Overall: fail.

Failure does not prove that momentum diversification has no effect. It means the already-viewed panel does not support replacing L4. The idea can remain only as a separately labelled forward research arm.
