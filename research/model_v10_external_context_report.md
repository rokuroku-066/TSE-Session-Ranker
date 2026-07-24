# Model v10 external-context generation 1

## Decision

**Retrospective-only screen; no production promotion.** The score-period
outcomes were already available to upstream research. The result is an
exploratory point-in-time external-data screen, not a fresh holdout.

All eight registered hypotheses (`X01`–`X08`) and the price-core control
(`X00`) were evaluated at top 1 and top 2, giving 18 candidate streams across
266 sessions from 2024-07-01 through 2025-07-31.

## Integrity and timing

- Protocol: `model_v10_external_context_gen1_20260723`
- Protocol SHA-256:
  `e7d28aa8492c7b478f66b9e0f3c67ac6afa46283a48177fb4c1825aef1305b15`
- Runner SHA-256:
  `3c0fafb751650a099ba3c789ca6cb6c7eb23a74589cca6b83ae125a5b3cd4cd3`
- Result SHA-256:
  `7942bdca4ecda77ccc15800b9a5027ec79585f77ab037e623694f85272129baf`
- Frozen panel SHA-256:
  `6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb`
- External series: FRED `SP500`, `NASDAQCOM`, `VIXCLS`, and `DEXJPUS`.
- Mapping rule: the latest source observation must be strictly earlier than
  the TSE score date; same-date observations are forbidden.
- Recorded mapping summary: `strictly_prior_all=true`, maximum carry age 3
  calendar days, latest source date 2025-07-30.
- Every recorded monthly model fold ends strictly before its score month.

## Results

Values are scheduled-day mean percentages. `Tail20` is net20 after removing
the best 20 days; `Code-cash` replaces the ten largest positive
profit-contributing codes with cash. Values below are displayed to six decimal
places; the result JSON retains full precision.

| Candidate | net20 | net40 | net60 | + months net40 | Tail20 | Code-cash |
|---|---:|---:|---:|---:|---:|---:|
| X00 control top1 | +0.204894 | +0.007902 | -0.189091 | 7/13 | -0.136982 | -0.182457 |
| X00 control top2 | +0.255321 | +0.058705 | -0.137912 | 8/13 | -0.005433 | -0.033199 |
| X01 US-market beta top1 | +0.095402 | -0.102343 | -0.300087 | 6/13 | -0.530071 | -0.318138 |
| X01 US-market beta top2 | +0.009807 | -0.188313 | -0.386434 | 4/13 | -0.449856 | -0.311199 |
| X02 growth-spread beta top1 | -0.037984 | -0.237984 | -0.437984 | 3/13 | -0.665269 | -0.509599 |
| X02 growth-spread beta top2 | -0.235901 | -0.434773 | -0.633645 | 2/13 | -0.620271 | -0.540614 |
| X03 USD/JPY beta top1 | -0.129185 | -0.328433 | -0.527681 | 5/13 | -0.966541 | -0.723710 |
| X03 USD/JPY beta top2 | -0.203129 | -0.402753 | -0.602377 | 3/13 | -0.759066 | -0.598378 |
| X04 VIX-change beta top1 | +0.086898 | -0.111598 | -0.310095 | 5/13 | -0.571570 | -0.367715 |
| X04 VIX-change beta top2 | -0.083187 | -0.282435 | -0.481683 | 4/13 | -0.542170 | -0.425807 |
| X05 beta composite top1 | +0.209949 | +0.010701 | -0.188547 | 5/13 | -0.617041 | -0.457016 |
| X05 beta composite top2 | +0.033890 | -0.164230 | -0.362351 | 5/13 | -0.459900 | -0.340751 |
| X06 context Ridge top1 | +0.262279 | +0.064535 | -0.133210 | 6/13 | -0.078043 | -0.064140 |
| X06 context Ridge top2 | +0.160636 | -0.037108 | -0.234853 | 7/13 | -0.099693 | -0.053969 |
| X07 TDnet interaction top1 | +0.163367 | -0.034377 | -0.232121 | 4/13 | -0.180737 | -0.141107 |
| X07 TDnet interaction top2 | +0.155057 | -0.041936 | -0.238928 | 4/13 | -0.101298 | -0.031393 |
| X08 regime experts top1 | +0.223914 | +0.030681 | -0.162552 | 8/13 | -0.115882 | -0.121311 |
| X08 regime experts top2 | +0.058311 | -0.136802 | -0.331915 | 5/13 | -0.200664 | -0.120629 |

The largest new-data point estimate is X06 top1: net40
`+0.06453450109597947%`. Its three net40 slices are
`+0.2141545807505479%`, `-0.17488104203497165%`, and
`+0.19423255509418735%`. At the same top-1 breadth it exceeds X00 by
`+0.05663259624514006` percentage points, but X06 top2 is
`-0.09581311032476225` points below X00 top2. Breadth stability therefore
fails.

The X00 top2 control is the only stream with positive net40 in all three
slices: `+0.12316320101876113%`, `+0.03069519986739586%`, and
`+0.026923757226225066%`. No external-context hypothesis is positive in all
three.

## Robustness and protocol gaps

- All 18 streams are negative at 60 bp.
- All 18 streams are negative after removing the best 20 days at net20.
- All 18 streams are negative when the ten largest positive
  profit-contributing codes are moved to cash at net20.
- X06 top1 is negative in confirmation A, negative at 60 bp, negative on
  `Tail20`, and negative on `Code-cash`.
- The protocol requires full monthly results, but the result retains only each
  stream's positive-month count.
- The result records a strict-prior mapping summary, but not the registered
  source-date mutation audit.
- The registered moving-block family-wise lower bound is absent. Consequently,
  no unadjusted winner is admissible for selection or promotion.

The independent cross-artifact audit reproduces the scheduled-day P&L and
classifies X06 as an incomplete-robustness exploratory estimate. No picks CSV
is added by this integration.

