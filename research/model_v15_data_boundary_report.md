# v1.5 real-data boundary: collect or reject

## Decision

The proposed material-strength × pre-open-absorption experiment is not a model
result. A current official TDnet index and linked PDF bodies were acquired with
the final collector, and one deliberately narrow forecast-revision table schema
was parsed and recomputed. This establishes a working acquisition and parsing
sample for arm A. It does not establish a source-complete historical or
prospective cohort, so A is not a model candidate.

The exact 08:58 auction, quote, futures, PTS, and execution inputs required by
arm B were not acquired. B and interaction arm AB are therefore rejected in
full, not retained as blocked or placeholder candidates.

No candidate was scored, no market outcome was inspected by this probe, no
production model changed, and `orders_allowed=false`.

## What was actually acquired

The final live probe started at 2026-07-28 16:51:32 JST. It requested two
official TDnet index pages containing 165 disclosures and downloaded all nine
PDF bodies whose titles matched the strict forecast-revision filter.

The two page responses had concordant `Last-Modified` headers. That header is
not treated as an atomic source watermark. After all nine PDFs were fetched,
both index pages were requested again, starting at 16:52:34 JST. For each URL,
the confirmation response was byte-identical to its initial response and had
the same parsed content, total, and `Last-Modified` header. A changing index
fails the probe as a torn snapshot.

The strict parser accepted one unambiguous full-year table with the expected
five-column header and units. It rejected eight unsupported or ambiguous
documents. Unknown tokens, mixed or reordered rows, missing units, inconsistent
revision amounts, and non-recomputing percentages fail closed. Unsupported
documents were not filled from their titles.

The accepted document was Tessec (6337), published at 15:30:

| Measure (JPY million) | Prior | Revised | Reported amount | Reported change |
|---|---:|---:|---:|---:|
| Sales | 6,300 | 7,000 | +700 | +11.1% |
| Operating profit | 450 | 1,100 | +650 | +144.4% |
| Ordinary profit | 590 | 1,300 | +710 | +120.3% |
| Net profit | 540 | 1,040 | +500 | +92.6% |

The parser independently recomputed all four amounts and percentages. It then
produced the implemented signed operating-profit strength, capped at `+0.30`.
The cap is frozen for any future protocol but was not preregistered before this
source probe.

Every request and extraction occurred after the 08:58:59 cutoff. The extraction
is physically quarantined from `candidate_records`, which remains empty. The
manifest also records `source_complete=false`, no clock-sync evidence, zero
PIT-complete sessions, zero historical validation sessions, and no order
authority.

The complete generated manifest, including both index receipts, both stability
receipts, all nine PDF receipts, the accepted extraction, and all eight
rejection reasons, is
[`model_v15_tdnet_live_probe.json`](model_v15_tdnet_live_probe.json).
Raw external documents are not committed. Their official URLs, response
timestamps, byte sizes, and SHA-256 values are retained in the manifest.
Consequently the manifest is evidence from this network run, not an independent
third-party attestation of the raw bytes.

## Acquisition implementation

`tse-session-ranker` now has two explicit commands:

```bash
tse-session-ranker download-tdnet-official \
  --url https://www.release.tdnet.info/inbs/I_list_001_YYYYMMDD.html \
  --destination var/tdnet-official

tse-session-ranker probe-tdnet-material \
  --index-url https://www.release.tdnet.info/inbs/I_list_001_YYYYMMDD.html \
  --destination var/tdnet-official \
  --manifest var/tdnet-material-probe.json
```

The collector:

- accepts only unmodified HTTPS URLs on the canonical official TDnet host;
- rejects redirects before reading their payloads;
- records request start and completed-response timestamps;
- preserves content type, `Last-Modified`, byte size, and SHA-256;
- distinguishes fresh network responses from cached replays;
- archives the prior raw and receipt before an explicit overwrite;
- follows the official same-date pagination graph to closure;
- requires complete, gap-free, non-overlapping page ranges;
- accepts an official zero-disclosure page only with its exact structural
  no-data state and no disclosure, document, or pagination links;
- re-requests every discovered index page and rejects any page that changes
  during the probe;
- downloads linked PDF bodies instead of treating URLs or titles as content;
- hash-checks PDFs before and after bounded `pdftotext` extraction;
- quarantines every extraction from candidate records; and
- never emits model scores, market outcomes, or order authority.

A repeated probe against an existing destination must use `--overwrite`.
Cached raw plus a locally authored receipt cannot prove a fresh network run.
Prior evidence is preserved under the destination's `history` directory.

## Historical TDnet is browseable, but not validation-ready

JPX documents that Listed Company Search exposes financial and other timely
disclosures for 121 months. This run also exercised the interactive official
search for code 6337 and reached historical PDF and XBRL links. The earlier
claim that public historical bodies were categorically unavailable was
therefore rejected.

That per-issuer browse path does not, by itself, establish a gap-free bulk
export across the eligible universe, contemporaneous point-in-time receipts,
or a complete security-session join. No such cohort was acquired in this run.
Historical material, D+1/D+3, and issuer-hierarchy model approaches are closed
for this protocol rather than left pending.

Official availability references:

- [JPX overview of TDnet and ten-year Listed Company Search](https://www.jpx.co.jp/english/equities/listing/disclosure/tdnet/index.html)
- [JPX Listed Company Search retention guide](https://www.jpx.co.jp/english/listing/co-search/01.html)
- [JPX TDnet paid database service](https://www.jpx.co.jp/english/markets/paid-info-listing/tdnet/02.html)

## Why the other approaches were closed

JPX distributes exact pre-open quote prices and sizes, ten price levels,
accumulated depth, and market-order size through dedicated real-time products.
Historical order-book data is also a dedicated market-data product. No
entitlement or raw snapshot from those products was configured:

- [JPX real-time market data](https://www.jpx.co.jp/english/markets/paid-info-equities/realtime/index.html)
- [JPX historical market data](https://www.jpx.co.jp/english/markets/paid-info-equities/historical/index.html)

Japannext states that historical tick data is provided to members and potential
trading clients. No entitlement or raw PTS snapshot was configured:

- [Japannext PTS technology and historical market data](https://www.japannext.co.jp/en/pts)

Consequently:

- `B_0858_PRICE_ABSORPTION` and `AB_MATERIAL_X_ABSORPTION` are rejected;
- auction-path, execution-first, and post-open sequential approaches are
  rejected;
- D+1/D+3, graph, hierarchical, and hedged variants are rejected because their
  complete point-in-time joins were not acquired; and
- A remains collector-only with no model candidate.

These are input-feasibility decisions for this protocol, not evidence that the
economic hypotheses have zero alpha.
