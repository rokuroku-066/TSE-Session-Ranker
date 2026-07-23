# Model v10 policy/universe Round 2 integration audit

Status: **`retain_for_forward_shadow` invalidated**

This is a later integration audit. The locked Round 2 protocol, runner, result,
and report remain byte-for-byte unchanged.

## Finding

The locked protocol requires the momentum-20 reference to beat at least four of
five unrelated split variables. The locked runner instantiates and evaluates
only four unrelated split comparators:

- `F03_OVERNIGHT20_HALF`
- `F04_OCMEAN20_HALF`
- `F05_ATR_HALF`
- `F06_CLOSE_LOCATION_HALF`

It consequently reports `specificity_wins_vs_unrelated_splits = 4`,
`specificity_total_unrelated_splits = 4`, and `specificity_supported = true`.
This executes a 4-of-4 check, not the preregistered 4-of-5 check.

The missing fifth comparator means the registered specificity gate was not
completed. Therefore the derived
`decision.retain_for_forward_shadow = true` is invalidated by this integration
audit. This audit does not substitute a corrected retrospective result and does
not alter the locked `production_change = false` conclusion.

## Locked artifact hashes

- Protocol: `6b9d74d5ab2bc6957a2359fbc48e6549e252abbe3f0417553e76b49838aa21e0`
- Runner: `4b2cc114e52e0665e9375dca21fe3cd0fd4917b1605407e3a4cab3eafe0c053a`
- Result: `95b869112fb9462d7c07decae68a41e222c4f3e206a436ac4dc3b5d778068f0a`
- Report: `0ecc17d67701500b8080072af3c2bf4de8758533f1f63ee89a4916cbe403c5b7`
