# v1.7 exact-liquidity reliability hypotheses

## Result-driven diagnosis

v1.6 did not show that completed-session liquidity is universally useless. It
rejected three particular mechanisms:

1. a fixed hard activity veto;
2. a global additive Ridge containing all seven exact-liquidity ranks; and
3. a fixed rule that makes close/VWAP reversal stronger as turnover shock rises.

The saved v1.6 picks give a sharper clue. C00 rank 1 averaged about
`-0.1993%` gross per day with a `41.6%` hit rate, while rank 2 averaged about
`+0.5636%` with a `65.8%` hit rate. LQ02 improved its first slot to roughly
`+0.0216%` gross, but displaced the profitable shoulder: its second slot fell
to roughly `-0.0250%`. When LQ02 selected C00 rank 2 in slot 1, those 14 dates
averaged about `+0.6513%` gross with a `78.6%` hit rate. Its new selections
outside C00's top two were negative on average.

That pattern motivates a different family-level hypothesis:

> Exact D-1 activity is more likely to describe the reliability, persistence,
> or regime meaning of an already strong price signal than to be unconditional
> mean alpha or a hard veto.

This is not a rank-2 rule. C00 rank 2 alone remained tail-dependent: its net40
mean was positive, but net60, top-four-profit-days-removed, and
top-five-profit-codes-cash stresses were negative. The new family therefore
keeps the existing strict tail, concentration, cost, and paired-uplift gates.

## Why this is not parameter retuning

No v1.6 veto threshold, Ridge alpha, reversal direction, capacity, or cost is
searched. Every candidate has one fixed slot, Ridge candidates use `alpha=1`,
the pairwise candidates use `C=1`, and the family contains exactly ten
mechanisms. The only deterministic candidate is RPY04. The other mechanisms
change where activity enters the system: top-set comparison, multi-day price
memory, issuer state, market state, or training-label reliability.

## Registered family

| ID | New mechanism | Fixed comparison |
|---|---|---|
| `PAIR01_TOP2_EXACT_LIQ_RERANK` | Freeze C00 top two, then use OOF pairwise exact-activity differences to choose one | `PAIR00_PRICE_ONLY_TOP2_RERANK` |
| `TW02_TURNOVER_WEIGHTED_PRICE_MEMORY` | Compare turnover-weighted 5/20-day OC memory with equal-day memory | `C00_PRICE_RIDGE_TOP1` |
| `VW03_AGGREGATE_COST_BASIS` | Use 20-day aggregate VWAP gap and 5-vs-20-day cost-basis migration | `C00_PRICE_RIDGE_TOP1` |
| `RPY04_RETURN_PER_TURNOVER_REVERSAL` | Reverse large five-day return per yen of turnover | `C00_PRICE_RIDGE_TOP1` |
| `AR05_SECURITY_ACTIVITY_REGIME_EXPERTS` | Route between two price experts by issuer turnover-shock state | `C00_PRICE_RIDGE_TOP1` |
| `PS06_PERSISTENT_VS_ISOLATED_ACTIVITY` | Separate persistent activity from an isolated shock and interact both with close/VWAP state | `C00_PRICE_RIDGE_TOP1` |
| `VP07_VWAP_RANGE_PRESSURE_MEMORY` | Use turnover-weighted 5/20-day memory of close location versus VWAP and range | `C00_PRICE_RIDGE_TOP1` |
| `RW08_EXACT_LOT_RELIABILITY_WEIGHT` | Use traded-lot history only as training-label reliability weight | `C00_PRICE_RIDGE_TOP1` |
| `MR09_TURNOVER_WEIGHTED_MARKET_REGIME` | Interact momentum with market turnover concentration and turnover-weighted VWAP-pressure breadth | `C00_PRICE_RIDGE_TOP1` |
| `AT10_ATTENTION_MIGRATION` | Model a security's migration in cross-sectional turnover rank and its interaction with close/VWAP state | `C00_PRICE_RIDGE_TOP1` |

The exact formulas, missing-data rules, feature transforms, fit schedules, and
winner rule are authoritative in
`research/model_v17_liquidity_reliability_protocol.json`.

## Pairwise leakage barrier

PAIR01 is the most direct response to the observed top-set calibration issue,
so it gets an additional nested time barrier. For every historical pair month,
C00 is fitted only through the preceding month. Its top two for that historical
month are therefore OOF. The pair model for a replay month uses only those OOF
pairs strictly before the replay month. PAIR00 receives the identical pair set
and fitting procedure but only the C00 score difference; PAIR01 adds the seven
exact-liquidity rank differences. Neither policy can select a security outside
the frozen C00 top two.

## Evaluation authority and data boundary

- Training expands monthly from 2025-09-01.
- Pairwise OOF months begin 2025-12-01.
- The candidate-specific replay is 79 official sessions from 2026-04-01
  through 2026-07-27.
- The 79 official JPX PDFs were acquired and SHA-256 locked before their raw
  market outcomes were parsed for this run.
- Project-level outcomes have been viewed by earlier repository work, so this
  is retrospective candidate-specific evidence, not an untouched holdout.
- Costs of 20/40/60 bp are sensitivity haircuts, not measured spread,
  slippage, or fill cost.
- Family size is ten. The paired one-sided bootstrap confidence is `0.99`,
  giving familywise one-sided confidence `0.90` by Bonferroni.
- Passing requires every registered mean, median, cost, half-period, monthly,
  tail, code-cash, paired-uplift, concentration, and execution-coverage gate.
- At most one retrospective research nominee can be named. Production remains
  unchanged and orders remain disabled regardless of the result.

## Interpretation rule

If no candidate passes every gate, the family is rejected. A positive point
estimate or an isolated attractive month is not a model improvement. If one or
more candidates pass, the registered paired-lower-bound rule chooses at most
one nominee for genuinely new forward observation; it still does not authorize
production promotion.
