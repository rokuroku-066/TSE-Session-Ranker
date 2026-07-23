# Model v10 Z17 peer-residual report

## Decision

`Z17_peer_residual_model` is rejected as an economic candidate under the frozen retrospective protocol. Both top-1 and top-2 portfolios were negative at every reported cost, every temporal slice was negative, no month was positive at the primary 40 bp cost, and both concentration stress tests remained negative. No production artifact or policy was changed.

This is a preregistered retrospective mechanism test, not an untouched holdout.

## Frozen design

- Protocol SHA-256: `c45d854cf35a6af3920e6f763d726b385b33559d18e981174503609cdfa36dbb`
- Frozen panel SHA-256: `6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb`
- Runner SHA-256: `a6235fa7cb2246c5199c4aa8177c8c3b227bb44eccde1b07de939bc9bd2b691a`
- Score period: 2024-07-01 through 2025-07-31, 266 sessions in 13 monthly expanding folds
- Peer model: `StandardScaler` plus eight-group `KMeans(random_state=20260723, n_init=10)`
- Profile minimum: 20 complete historical rows per code across the six frozen profile features
- Predictive model: median imputation with missing indicators, standardization, and `Ridge(alpha=1)` on the 15 G0 features with date-equal weights
- Target: within-date percentile of stock open-to-close return minus its fold-fixed, leave-one-out same-date peer mean

## Overall results

All values are mean daily net returns in percentage points.

| Portfolio | Net20 | Net40 | Net60 | Positive months at net40 | Best 20 days removed, net20 | Top 10 profit codes to cash, net20 |
|---|---:|---:|---:|---:|---:|---:|
| Top 1 | -0.2964 | -0.4964 | -0.6964 | 0/13 | -0.5826 | -0.4644 |
| Top 2 | -0.2995 | -0.4991 | -0.6988 | 0/13 | -0.5054 | -0.4116 |

Top 1 selected 183 unique codes; its ten most frequently selected codes represented 17.29% of sessions. Top 2 selected 322 unique codes; its ten most frequently selected codes represented 12.78% of its 532 fixed sleeves. One top-2 sleeve had an unobserved outcome and correctly remained cash without cost or replacement.

## Monthly results

Mean daily net40 percentage points:

| Month | Top 1 | Top 2 |
|---|---:|---:|
| 2024-07 | -0.1275 | -0.0158 |
| 2024-08 | -0.7787 | -0.7960 |
| 2024-09 | -0.2549 | -0.6075 |
| 2024-10 | -0.3725 | -0.5070 |
| 2024-11 | -0.5712 | -0.4668 |
| 2024-12 | -1.1423 | -0.9209 |
| 2025-01 | -0.2732 | -0.7153 |
| 2025-02 | -0.1259 | -0.3626 |
| 2025-03 | -0.3325 | -0.2264 |
| 2025-04 | -1.2651 | -0.6937 |
| 2025-05 | -0.4328 | -0.6490 |
| 2025-06 | -0.4823 | -0.4057 |
| 2025-07 | -0.2263 | -0.1649 |

The result JSON contains all 13 monthly means at 20, 40, and 60 bp.

## Temporal slices

| Portfolio | Slice | Net20 | Net40 | Net60 |
|---|---|---:|---:|---:|
| Top 1 | Discovery | -0.1833 | -0.3833 | -0.5833 |
| Top 1 | Confirmation A | -0.3053 | -0.5053 | -0.7053 |
| Top 1 | Confirmation B | -0.3992 | -0.5992 | -0.7992 |
| Top 2 | Discovery | -0.2733 | -0.4733 | -0.6733 |
| Top 2 | Confirmation A | -0.3440 | -0.5440 | -0.7440 |
| Top 2 | Confirmation B | -0.2738 | -0.4726 | -0.6714 |

## Integrity audit

- All eight peer groups were occupied in every fold. Qualified profiles increased from 3,496 in the first fold to 3,793 in the last.
- Low-history or missing-profile codes used the frozen nearest/global profile fallback. No training target required the same-date global outcome fallback because every focal row had at least one observed peer-group member.
- Every fold's training and profile dates ended before its scoring dates, and `feature_source_max_date < date` held for all fitted and scored rows.
- Date-equal training-weight totals differed from one by at most `1.1102230246251565e-16`.
- Deterministically mutating every scoring-month outcome and label produced bitwise-identical scores and identical top-1/top-2 selections in all 13 folds.
- A separate explicit date-and-slot P&L loop exactly reproduced the vectorized daily returns at 20, 40, and 60 bp and the top-10-profit-code cash stress test; maximum absolute difference was `0.0`.
- Selection SHA-256: `76f6b6f02101a2b71b11cf6776b39344bec004ad4cb09986d49043a8609b6126`
- Audit SHA-256: `25a542ec5b3c4f4ea5a0f23f953578595fe4d447f3ed5679d32206f686e1f523`
- Result SHA-256: `3eeba440a914c5cfe941f5460bf94f81e89528f755fdf24a0a60f10601fd6fc2`

## Production status

`production_model_changed` is `false`, and `production_promotion` is `none`. The negative result was recorded without post-result tuning.
