# v0.6 feature candidate inventory

This inventory was frozen before the v0.6 retrospective diagnostic was run.
The target is the **same-session open-to-close net return**, using only values
known by `08:58:59 Asia/Tokyo` on the target session.  A historical feature for
date `D` may use observations through `D-1`; it may never use the realised open,
open time, volume, high, low, or close of `D`.

The machine-readable source of truth is
[`model_v06_feature_catalog.json`](model_v06_feature_catalog.json).  This file
explains why each block exists and why several attractive-looking candidates
are deliberately deferred.

## Decision classes

| Class | Meaning |
|---|---|
| `historical_screen` | Reconstructable from the sealed JPX monthly PDFs or cached TDnet date indexes. Included as one all-or-none mechanism block. |
| `forward_only` | Useful at 08:58, but no timestamped historical archive exists. Collect append-only; never substitute a later value. |
| `blocked_source` | Conceptually useful, but the current archive lacks point-in-time fields or source bytes. |
| `exclude` | Redundant, economically misleading, or inherently leaky for this target. |

## Historically screenable blocks

`G0` is the common anchor.  Every ablation is `G0 + exactly one block`, so a
model change cannot be mistaken for a feature contribution.

| Block | Mechanism | Features | Main guard |
|---|---|---|---|
| `G0_price_core` | Persistent intraday behaviour, prior gaps, volatility and medium-term price state | 15 existing point-in-time columns | All rolling price sources are shifted at least one session. |
| `G1_short_reversal` | A large recent close-to-close move or range expansion may continue or reverse after the next open | `cc_reversal_1_atr`, `cc_momentum_3_atr`, `range_shock_1_20`, `cc_vol_ratio_5_20` | Returns above 30% are treated as possible corporate-action discontinuities. Range-shock denominator ends at `D-2`. |
| `G2_gap_trait` | Securities differ in how often prior overnight gaps are faded or continued | `overnight_std_20`, `gap_fill_rate_20`, `gap_response_beta_60`, `gap_frequency_20` | These summarize **past** gaps only. The realised gap on `D` is forbidden. |
| `G3_session_dynamics` | Morning/afternoon continuation, midday repricing and whole-session close location | 7 AM/PM and range-history features | Every AM/PM observation is shifted before it reaches `D`. |
| `G4_market_regime` | Breadth, dispersion and market trend alter the ranking of high-beta/high-ATR stocks | 12 market state fields plus 5 stock×market interactions | Cross-sectional market state is formed on `D-1`; common state alone cannot change a linear rank, hence preregistered interactions are included. |
| `G5_liquidity_proxy` | Repeated no-trade/flat sessions identify names where a theoretical rank may not be executable | 20/60-day no-trade and flat/range rates | Explicit no-trade rows are used; missing source rows are unknown, not no-trade. No range-based fake volume is created. |
| `T0_clean_event` | A semantically separated TDnet event vector | 24 corrected title-family flags | `自己株式取得` is excluded from M&A. Employee compensation is separated from external financing. Unknown revision direction remains explicit. |
| `T1_event_structure` | Timing, bundle complexity, stage and recent disclosure intensity | 13 count/timing/stage fields | Only documents published no later than the target cutoff are assigned. Missing date indexes remain missing, never “no disclosure”. |
| `X0_event_context` | The same event may behave differently after a run-up or in a broad market shock | 6 preregistered event×price/market interactions | Screen only after `T0_clean_event`; do not search arbitrary interaction pairs. |

### Why the TDnet taxonomy is rebuilt

The prior broad title rules were not clean enough for feature attribution.
Across 97,006 cached titles, 30.4% of the old M&A matches also matched buyback
because bare `株式取得` matched `自己株式取得`.  More than half of old equity-
financing matches appeared to be stock options, equity compensation, or
director/employee plans.  Also, only about 3% of earnings-revision titles state
an up/down direction.  v0.6 therefore keeps `direction_unknown` and separates
capital supply from compensation instead of forcing a signed event.

## Forward-only candidates

These are higher priority than many historical indicators, but they become
valid only after the exact snapshot and its observation time are saved.

| Block | Candidate fields | Required timestamp/provenance |
|---|---|---|
| `F0_market_0858` | Nikkei 225 futures, TOPIX futures, spread, USDJPY, US-index futures, volatility index | Exact value observed on target date at or before 08:58:59 JST; no daily-close proxy or carry-forward. |
| `F1_indicative_gap` | Indicative gap, gap/ATR, beta-adjusted gap, gap×historical response | Exchange/broker indicative price snapshot at or before cutoff. The realised open must be stored in a different outcome column. |
| `F2_auction_pts` | Market-order imbalance, board depth, special quote, expected open time, indicative volume, PTS return/volume | Append-only board/PTS snapshot with target date and observation timestamp. |
| `F3_exact_liquidity` | Lagged volume/turnover, ADV20, Amihud, lot value, order/ADV, observed slippage | Source with explicit share/JPY units and point-in-time trading unit. |
| `F4_tdnet_quantitative` | Earnings surprise, dividend yield change, buyback pressure, dilution, M&A/contract size | Original PDF/XBRL bytes, request start, SHA-256, parser version, field availability and extraction location. |

For quantitative disclosures, the preferred scale is economic rather than a
raw percentage: profit change/market cap, dividend change/prior close, buyback
amount/market cap and per-day amount/ADV, potential new shares/outstanding
shares, and acquisition price or target profit/market cap.

## Blocked until a dated source exists

| Candidate | Reason |
|---|---|
| Sector residual, sector momentum and sector beta | The current repository has no dated membership history. Applying today’s sector retroactively would introduce survivorship/look-ahead bias. |
| Market cap, shares outstanding and fundamentals | Need point-in-time security master/fundamental vintages; prior share price is not a size proxy. |
| Historical PDF quantitative values | URLs were cached, but original PDF bytes were not. A new download is useful for retrospective discovery only, not a sealed point-in-time test. |
| Historical turnover/volume/lot features | The monthly JPX PDFs used by v0.5 contain AM/PM OHLC but no usable volume, turnover or trading-unit fields. |

## Explicit exclusions

- Actual target-session open/gap, open time, volume, high, low or close.
- 09:00-or-later futures/FX values and end-of-day values presented as 08:58
  context.
- RSI, MACD, generic moving-average crosses and long indicator menus in the
  first screen; they mostly re-express the registered momentum/win/volatility
  state and increase the search surface.
- Share price as a proxy for company size.
- High-low range as a proxy for turnover, order-book depth or transaction cost.
- Current sector/index membership backfilled across history.

## Validation order

1. Freeze this catalog, the JSON catalog and the protocol by SHA-256.
2. Run leakage mutation tests and label-blind availability/distribution checks.
3. Keep the raked logistic anchor fixed while comparing `G0` against
   `G0 + one group` on known retrospective data.  This is a development
   diagnostic, not a holdout.
4. Form at most one preregistered survivor union, then run leave-one-group-out.
5. Display top two every source-complete day.  Keep display and trade decisions
   in separate outputs; a future out-of-fold expected-net gate may leave either
   sleeve in cash and may not replace it with rank 3.
6. Start genuine prospective feature development on the first session after
   the frozen implementation.  After 120 sessions and six calendar months,
   lock features/model/gate and evaluate the next 100 sessions without changes.

