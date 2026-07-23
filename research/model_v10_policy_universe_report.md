# v1.0 zero-base policy/universe screen

## Status

This is a retrospective mechanism screen on an already-viewed frozen panel. It cannot authorize production use.

The previous rank-2-only and market-breadth switch mechanisms were prohibited. All sixteen preregistered new mechanisms are retained below.

## Ranked results

| Policy | Net20 | Net40 | Net60 | Months + | Blocks >=0 | Top20 removed | Mean names | Zero days | Family 90% lower uplift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H12_MOMENTUM_BUCKET_DIVERSIFICATION | +0.3044% | +0.1082% | -0.0881% | 13/13 | 4/4 | +0.0462% | 2.00 | 0 | -0.1439% |
| C00_L4_TOP2_EQUAL | +0.2832% | +0.0866% | -0.1100% | 11/13 | 4/4 | +0.0135% | 2.00 | 0 | n/a |
| H03_DISAGREEMENT_PENALTY | +0.2669% | +0.0703% | -0.1264% | 11/13 | 4/4 | +0.0020% | 2.00 | 0 | -0.1815% |
| H08_ADVERSE_EVENT_EXCLUSION | +0.2614% | +0.0644% | -0.1326% | 11/13 | 4/4 | -0.0074% | 2.00 | 0 | -0.1870% |
| H05_INVERSE_UNCERTAINTY_WEIGHT | +0.2607% | +0.0637% | -0.1333% | 11/13 | 4/4 | -0.0080% | 2.00 | 0 | -0.1877% |
| H01_ENSEMBLE_MEAN | +0.2606% | +0.0636% | -0.1334% | 11/13 | 4/4 | -0.0082% | 2.00 | 0 | -0.1878% |
| H04_STABLE_TOPSET | +0.2606% | +0.0636% | -0.1334% | 11/13 | 4/4 | -0.0082% | 2.00 | 0 | -0.1878% |
| H02_ENSEMBLE_WORSTCASE | +0.2577% | +0.0611% | -0.1356% | 11/13 | 4/4 | -0.0111% | 2.00 | 0 | -0.1907% |
| H13_LOW_CORRELATION_PAIR | +0.2150% | +0.0184% | -0.1782% | 12/13 | 4/4 | -0.0735% | 2.00 | 0 | -0.2333% |
| H16_EVENT_AWARE_UTILITY | +0.2110% | +0.0137% | -0.1837% | 9/13 | 4/4 | -0.0646% | 2.00 | 0 | -0.2373% |
| H15_SPREAD_CASH_OVERLAY | +0.1997% | +0.0495% | -0.1007% | 11/13 | 4/4 | -0.0166% | 2.00 | 0 | -0.2487% |
| H09_EVENT_BARBBELL | +0.1780% | -0.0190% | -0.2160% | 10/13 | 4/4 | -0.1023% | 2.00 | 0 | -0.2704% |
| H06_GAP_CONFIDENCE_COUNT | +0.1012% | -0.0961% | -0.2935% | 10/13 | 3/4 | -0.1501% | 1.48 | 0 | -0.3471% |
| H07_LIQUIDITY_RELIABILITY_UNIVERSE | +0.0637% | -0.1355% | -0.3348% | 9/13 | 3/4 | -0.1849% | 2.00 | 0 | -0.3846% |
| H14_POSTERIOR_NET_ABSTAIN | +0.0000% | +0.0000% | +0.0000% | 0/13 | 4/4 | +0.0000% | 0.00 | 266 | -0.4483% |
| H10_STYLE_RESIDUAL | -0.1221% | -0.3195% | -0.5169% | 3/13 | 0/4 | -0.3626% | 2.00 | 0 | -0.5705% |
| H11_RISK_UTILITY | -0.1740% | -0.3725% | -0.5710% | 1/13 | 0/4 | -0.2757% | 2.00 | 0 | -0.6223% |

## Preregistered shortlist checks

| Policy | Beat control | Net40 + | Top20 removed + | >=8 positive months | 4 blocks >=0 | Family lower >0 | All |
|---|---:|---:|---:|---:|---:|---:|---:|
| H01_ENSEMBLE_MEAN | no | yes | no | yes | yes | no | no |
| H02_ENSEMBLE_WORSTCASE | no | yes | no | yes | yes | no | no |
| H03_DISAGREEMENT_PENALTY | no | yes | yes | yes | yes | no | no |
| H04_STABLE_TOPSET | no | yes | no | yes | yes | no | no |
| H05_INVERSE_UNCERTAINTY_WEIGHT | no | yes | no | yes | yes | no | no |
| H06_GAP_CONFIDENCE_COUNT | no | no | no | yes | no | no | no |
| H07_LIQUIDITY_RELIABILITY_UNIVERSE | no | no | no | yes | no | no | no |
| H08_ADVERSE_EVENT_EXCLUSION | no | yes | no | yes | yes | no | no |
| H09_EVENT_BARBBELL | no | no | no | yes | yes | no | no |
| H10_STYLE_RESIDUAL | no | no | no | no | no | no | no |
| H11_RISK_UTILITY | no | no | no | no | no | no | no |
| H12_MOMENTUM_BUCKET_DIVERSIFICATION | yes | yes | yes | yes | yes | no | no |
| H13_LOW_CORRELATION_PAIR | no | yes | no | yes | yes | no | no |
| H14_POSTERIOR_NET_ABSTAIN | no | no | no | no | yes | no | no |
| H15_SPREAD_CASH_OVERLAY | no | yes | no | yes | yes | no | no |
| H16_EVENT_AWARE_UTILITY | no | yes | no | yes | yes | no | no |

## Interpretation

The point winner was H12_MOMENTUM_BUCKET_DIVERSIFICATION, but no new mechanism passed all preregistered return, period, tail and multiplicity checks. The zero-base screen therefore does not justify replacing the control.

The point winner is descriptive only. Any new mechanism requires a new, content-addressed forward shadow beginning after registration.

## Reproduction

```bash
PYTHONPATH=src:. python /tmp/v10_policy_universe/runner.py
```

- Panel SHA-256: `6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb`
- Protocol SHA-256: `d2fc192f3d0f576389507be9974319ae9193883c0daa0ff7cd64236ae532d443`
- Runner SHA-256: `2252c6c1fbc1f8da2d7860f839e443f214b1e07291d7389771ae1f087a5e5234`
