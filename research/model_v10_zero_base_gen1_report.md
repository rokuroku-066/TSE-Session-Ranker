# Model v10 zero-base generation-one partial screen

## Decision

**Retrospective-only screen; no production promotion and no selected
specification.** All score-period outcomes were previously available to
upstream research, and the registered temporal slices are not untouched
holdouts.

This artifact is explicitly a **6/18 partial generation-one screen**, not
completion of `model_v10_zero_base_gen1_20260723`. It evaluates only `Z01`,
`Z02`, `Z03`, `Z05`, `Z06`, and `Z07`, each at top 1 and top 2. Registered
`Z04` and `Z08`–`Z18` are absent.

## Integrity and scope

- Protocol SHA-256:
  `f8867299501d295770c52ecd6437195802ef1b7728e07107f47c620f4d01ad18`
- Runner SHA-256:
  `6537451cd7d5c498713cafd228d64557258c775fef57c1ad93ede88ed0fde354`
- Result SHA-256:
  `14b09e0bffe4ebc1160299d804931fa9c55eef94985cecf1110f9417122f146a`
- Frozen panel SHA-256:
  `6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb`
- Score period: 2024-07-01 through 2025-07-31, 266 sessions.
- The 39 monthly folds recorded for `Z05`–`Z07` are all strictly prior.
  `Z01`–`Z03` are shifted expanding/rolling rules and do not emit fold rows.

## Results

Values are scheduled-day mean percentages. `Tail20` is net20 after removing
the best 20 days; `Code-cash` replaces the ten largest positive
profit-contributing codes with cash. Values below are displayed to six decimal
places; the result JSON retains full precision.

| Candidate | net20 | net40 | net60 | + months net40 | Tail20 | Code-cash |
|---|---:|---:|---:|---:|---:|---:|
| Z01 weekday EB top1 | -0.070672 | -0.269920 | -0.469168 | 4/13 | -0.621648 | -0.398836 |
| Z01 weekday EB top2 | -0.023114 | -0.221610 | -0.420107 | 6/13 | -0.435044 | -0.288936 |
| Z02 lottery penalty top1 | -0.189641 | -0.388889 | -0.588138 | 1/13 | -0.313748 | -0.255106 |
| Z02 lottery penalty top2 | -0.243456 | -0.443080 | -0.642704 | 0/13 | -0.335049 | -0.274579 |
| Z03 left-tail alpha top1 | -0.043411 | -0.239652 | -0.435893 | 4/13 | -0.622245 | -0.410441 |
| Z03 left-tail alpha top2 | -0.012800 | -0.210168 | -0.407537 | 4/13 | -0.390552 | -0.289632 |
| Z05 daily-z Ridge top1 | +0.150461 | -0.045028 | -0.240517 | 8/13 | -0.186899 | -0.253745 |
| Z05 daily-z Ridge top2 | +0.092994 | -0.102871 | -0.298736 | 5/13 | -0.149354 | -0.202740 |
| Z06 exp-decay HL60 top1 | +0.308507 | +0.113770 | -0.080966 | 5/13 | -0.052113 | -0.067685 |
| Z06 exp-decay HL60 top2 | +0.118101 | -0.078139 | -0.274380 | 4/13 | -0.119491 | -0.145285 |
| Z07 exp-decay HL120 top1 | +0.244491 | +0.048250 | -0.147990 | 7/13 | -0.089352 | -0.128066 |
| Z07 exp-decay HL120 top2 | +0.238581 | +0.042716 | -0.153148 | 7/13 | -0.022415 | -0.058050 |

The largest point estimate is Z06 top1: net40
`+0.11377037369046766%`. Its three net40 slices are
`+0.210564200333014%`, `-0.09622550406508647%`, and
`+0.2619717377627345%`; only 5/13 months are positive. At net20 it becomes
`-0.05211309007107555%` after removing the best 20 days and
`-0.06768520740033153%` after moving the ten leading profit codes to cash.
Its largest single-code share of positive code-level profit is
`0.2866862565751102`.

Z07 top2 is the only reported stream positive in all three net40 slices:
`+0.04440355452749399%`, `+0.06741629037653592%`, and
`+0.01221226369308547%`. It still fails at 60 bp
(`-0.1531484015510181%`), `Tail20`
(`-0.022414512438531264%`), and `Code-cash`
(`-0.05804999384198461%`).

## Incompleteness and related v10 tracks

The other v10 artifacts cover related mechanisms but do not complete the
literal 18-ID umbrella protocol:

- Architecture `M01`/`M02` cover related listwise surrogates, and `M10` covers
  a related point-in-time mixture; these are not the exact registered
  `Z08`–`Z11` specifications.
- The online-ensemble track (`O01`–`O03`) covers related strictly-prior online
  aggregation; it is a separate protocol rather than a literal `Z16` result.
- Market-residual `R01` separately implements the `Z18` factor-residual idea
  and was rejected in its own track.
- The peer-residual `Z17` idea remains unevaluated.

`Z04` and `Z08`–`Z18` therefore remain absent from this result even where a
separate track tests a related mechanism; `Z17` remains unevaluated across
tracks.

## Robustness and protocol gaps

- All 12 reported streams are negative at 60 bp.
- All 12 are negative on both `Tail20` and `Code-cash`.
- The protocol-required family-specific controls are absent.
- The moving-block family-wise lower bound is absent.
- No target-day outcome-mutation result is recorded.
- Because only 6/18 hypotheses were run and the registered control,
  multiplicity, and mutation outputs are missing, Z06 is an unadjusted partial
  screen estimate, not an admissible selected specification.

The independent cross-artifact audit classifies this track as **no selection;
protocol incomplete**. No picks CSV is added by this integration.
