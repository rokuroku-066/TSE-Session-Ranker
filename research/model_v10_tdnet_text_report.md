# TDnet title/text zero-base validation

## Decision

The experiment found one new point-estimate lead but no promotion-ready rule.

`T02_char_value_event_only` (character TF-IDF title bundles, Ridge on clipped
open-to-close return, daily top 1 among overnight TDnet event stocks) produced:

- gross mean: **+0.9191% per scored day**
- net 20 bp: **+0.7191%**
- net 40 bp: **+0.5191%**
- net 60 bp: **+0.3191%**

The same 88-session sample gave L4 top 1 net 40 bp of **+0.0222%**.
Nevertheless, T02 is retained only as a shadow hypothesis because it failed
the preregistered robustness checks:

- positive months: **3/5**
- discovery net 40 bp: **+0.3152%**
- June 2025 confirmation: **+1.5585%**
- July 2025 confirmation: **-0.0190%**
- net 20 bp after removing the 20 best days: **-0.5721%**
- net 20 bp after replacing the 10 best profit-contributing codes with cash:
  **-0.1085%**
- paired delta versus L4 90% moving-block interval:
  **[-0.0213%, +1.0142%]**
- 10-candidate familywise moving-block reality-check p-value: **0.2103**

The candidate's own unadjusted 90% block interval for net 40 bp was
`[+0.0934%, +0.9350%]`, but the paired and familywise evidence is not strong
enough to distinguish it from model-search luck.

## Protocol and point-in-time controls

The protocol was written before candidate outcomes were read:

- protocol: `protocol.json`
- protocol SHA-256:
  `700b0262f805c37659453af72651897d116e8bd1f6329a602989b90b212c5b3a`
- frozen panel SHA-256:
  `6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb`
- runner SHA-256:
  `f49203ef23a6ff15bb609178c6c3aa278b3d052e6383503703a45ec1b3e67744`
- result SHA-256:
  `3a40e3541fb24f89f2b009b48b2af4d3a84afe2e96ce03010ea0a87442800196`

Each disclosure was assigned to the first TSE session whose 08:58:59 JST
cutoff was not earlier than its publication timestamp. A prior-session
intraday disclosure was excluded because it could already have been traded.
Same-morning, post-close, weekend, and holiday disclosures were retained.
The mapping audit found **zero future-publication violations**.

A target session was called source-complete only when every calendar-date
HTML page from the prior TSE session through the target session existed.
Missing pages were never represented as no-event observations.

## Coverage limitation

The local cache contains 218 HTML pages and 50,719 disclosures:

- 2024-04-02 through 2024-07-31: 110 pages
- 2025-03-31 through 2025-07-31: 108 pages
- strict source-complete score sessions: **88**
- frozen score sessions excluded as source-missing: **178**
- scored months: July 2024 and April–July 2025
- qualifying overnight disclosure bundles on scored sessions: **14,770**

Thus only about one third of the frozen 266-session evaluation calendar could
be evaluated. The two cache periods are also discontinuous. More importantly,
the historical HTML files have no observation-time metadata sidecars.
Publication-time PIT can be checked, but archive finality and the exact time
at which the pages were observed cannot be proven. These limitations alone
prevent a production claim.

The return simulation also has no historical 08:58 order book, opening-auction
depth, spread, or rejected-order reconstruction. Fixed 20/40/60 bp costs are
stress scenarios, not proof that the selected small-cap names were executable
at those costs.

## Candidate comparison

All figures below are mean daily return after 40 bp round-trip cost.

| Candidate | Top 1 | Top 2 | Positive months, top 1 |
|---|---:|---:|---:|
| G0 price core | +0.1181% | +0.0211% | 3/5 |
| L4 price control | +0.0222% | +0.1075% | 3/5 |
| L4, no-event stocks only | +0.0261% | +0.1145% | 3/5 |
| T01 char TF-IDF rank target | -0.2303% | -0.3152% | 1/5 |
| **T02 char TF-IDF value target** | **+0.5191%** | -0.0855% | **3/5** |
| T03 rank-text overlay on L4 | -0.5609% | -0.4425% | 0/5 |
| T04 value-text overlay on L4 | -0.6517% | -0.5181% | 0/5 |
| T05 title novelty | -0.1851% | -0.1831% | 2/5 |
| T06 intensity/congestion | -0.4568% | -0.3622% | 1/5 |
| T07 corroboration/contradiction | -0.7169% | -0.3146% | 0/5 |
| T08 all structured event features | -0.4010% | -0.3866% | 2/5 |
| T09 structured overlay on L4 | -0.7197% | -0.4021% | 0/5 |
| T10 value-event/L4 barbell | +0.5191% | -0.0366% | 3/5 |

T10 top 1 was identical to T02 because at least one positive value prediction
was available on every scored day. Its top-2 fallback did not help.

## What the selected titles show

The strongest T02 winners often had recurring, information-bearing title
templates:

- monthly sales速報
- monthly performance reports
- earnings presentation material
- earnings releases

However, the same title families also appeared among large losses. Examples
included monthly sales reports, earnings releases, and company-division
announcements. A title identifies the *kind* of news but often omits its
numeric direction and surprise relative to expectations. The structured
positive/negative lexicons also failed, and a fixed text uplift over L4 made
results materially worse.

This suggests that the next information increment is not another title
keyword. It is the body/PDF content:

- actual-versus-forecast numerical surprise
- revision amount scaled by market capitalization
- monthly sales growth versus the firm's own recent history
- whether a buyback is new, executable in the market, and imminent
- simultaneous positive and adverse disclosures
- management guidance and causal language

## Relation to prior research

The direction is consistent with, but does not replicate, recent Japanese
disclosure research:

- A 2025 JSAI study used roughly 690,000 TDnet titles and reported that
  pretrained Japanese title embeddings improved classification of the lower
  5% of announcement-window CARs. Its target, model, and horizon differ from
  this same-day open-to-close ranking task, so it is motivation rather than
  validation:
  [JSAI title-embedding study](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/FIN-035/2025_68/_article/-char/ja)
- A separate Japanese PEAD study reports useful interaction between numerical
  and textual earnings surprises. That directly supports adding parsed
  numerical/body features rather than relying on title templates alone:
  [JSAI LLM/PEAD study](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/FIN-035/2025_157/_article/-char/ja)
- General learning-to-rank work motivates cross-sectional ranking objectives,
  but here the preregistered value target beat the rank target. This is a
  sample finding, not evidence that ranking objectives are generally inferior:
  [Learning to Rank for Stock Portfolio Selection](https://arxiv.org/abs/2012.07149)

## Recommended next test

Keep T02 frozen as a shadow baseline and collect a continuous, provenance-
tagged TDnet archive. The next preregistered experiment should parse PDFs and
add numerical surprise features to the same monthly walk-forward design.
It should not retune TF-IDF ranges or filters on these 88 sessions.

Minimum evidence before reconsidering promotion:

1. a continuous untouched period of at least 120 sessions;
2. positive net 40 bp in both halves;
3. positive paired lower bound versus L4;
4. positive result after tail-day and code-concentration stress tests;
5. reproducible page-observation provenance;
6. a separately frozen prospective or later-year confirmation set.

## Artifacts

- `protocol.json`
- `runner.py`
- `result.json`
- `picks.csv`
- `bundle_coverage.csv`
- `audit.py`
- `audit.json`

No production model or repository file was changed by this experiment.
