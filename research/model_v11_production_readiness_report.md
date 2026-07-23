# T02 OOT data-readiness / data-gap verification

## 結論

**Fail closed: T02のexact no-tuning OOT統計gateもexecution gateも現在は実行不能です。**

- Frozen T02 artifact hashes: 4/4 exact
- Joint provenance-complete score sessions: 0/120 minimum
- Required calendar months: 0/6 minimum
- Exact no-tuning statistical gate can run: False
- Execution gate can run: False
- Verifier scope: data_readiness_only_not_production_certification
- Can certify production: False
- Genuinely untouched holdout present: False
- Fresh paper-live/later holdout required: True

## JPX price window

- Expected files: 160 (warmup 1 + score 159)
- Prior parser audit reports source complete: 160/160 files, 159/159 score sessions
- Raw PDFs currently present and hash-exact: 0/160
- Currently reproducible source-complete score sessions: 0/159
- Provenance-complete score sessions: 0/159
- Parser hash exact: True (build/lib/tse_session_ranker/data/jpx.py, src/tse_session_ranker/data/jpx.py)
- Registry-bound acquisition/parse manifest available/contract-exact: False/False

prior監査JSONは160 PDF・622,724 parsed rows・reject 0と記録しています。ただしraw bytesとsealed acquisition manifestが無いため、現在のworkspaceから同じ入力を再構成してその記録を再証明することはできません。

## TDnet window and provenance

- Expected calendar-date pages: 243 (2025-08-01 … 2026-03-31)
- Pages present: 0/243
- Metadata sidecars present: 0/243
- Parser-valid/source-complete/provenance-complete pages: 0/0/0
- Provenance-complete score sessions: 0/159

不足pageをno-eventとして扱うことは禁止されているため、TDnet 0件日は生成していません。

## Execution and context fields

| Field | Explicitly registered and content-validated OOT evidence |
|---|---:|
| Exact 08:58 indicative price | no |
| Executable order simulation/fill | no |
| 08:58 bid/ask spread | no |
| Realized spread | no |
| Slippage | no |
| 08:58 special quote | no |
| Open time / delayed-open handling | no |
| Exact 08:58 futures | no |
| PTS price and volume | no |
| Volume | no |
| Turnover | no |
| Minimum trading unit | no |
| Tick size | no |
| Predicted opening turnover | no |

JPX parser codeにはfinal special quote、volume、turnover、trading unitの出力能力がありますが、現物raw/parsed exportとfield-level coverageが無く、final special quoteも08:58 snapshotではありません。このためgate証拠には昇格させていません。

ファイル名・CSV/TSV headerの探索結果はdiagnostic candidateにすぎません。空のheader-only file、OOT外の行、PIT/provenance/hash未検証fileはfield evidenceとして数えません。positive certificationには明示的に凍結したmanifestとcontent readerが必要です。

execution ledgerは、hash-bound T02 decision artifactと独立replay audit、approved source、policy auditへ結合し、全order decisionをfilled、special-quote cancel、delayed-open cancel、liquidity cancel、unfilledのいずれかで完全被覆する必要があります。08:58秒内観測、tick/lot、spread/slippage、予定注文額の日次合計と参加率は入力値から再計算します。

## Exact blockers

- `JPX_RAW_FILES_MISSING_OR_MISMATCHED` (statistical_input) — Official JPX daily quotation PDFs must be locally present and match the audited per-file SHA-256 values. Observed: 0/160 hash-exact PDFs; missing=160, mismatched=0
- `JPX_REGISTERED_MANIFEST_UNAVAILABLE` (statistical_input) — The registry-bound acquisition/parse manifest must be present and hash-exact for source and parser provenance. Observed: path=None; exists=False; registered_search_file=False; sha256_exact=False; contract_exact=False
- `JPX_PROVENANCE_BELOW_MINIMUM` (statistical_input) — At least 120 score sessions require source URL, byte count, SHA-256, request/receipt timestamps, parser version, and parse completion. Observed: 0 provenance-complete score sessions
- `TDNET_DAILY_PAGES_MISSING` (statistical_input) — Every calendar-date TDnet index from warmup through score_end must be locally present. Observed: 0/243 pages; 243 missing
- `TDNET_REGISTERED_MANIFEST_UNAVAILABLE` (statistical_input) — The registry-bound TDnet page/metadata manifest must be present, hash-exact, unique, and enumerate all 243 dates. Observed: path=None; exists=False; registered_search_file=False; sha256_exact=False; contract_exact=False
- `TDNET_SOURCE_COMPLETENESS_MISSING` (statistical_input) — Every TDnet page must be parser-valid, finalized, and have integrity-valid acquisition metadata; a gap is not a no-event day. Observed: 0/243 source-complete pages
- `TDNET_PROVENANCE_INCOMPLETE` (statistical_input) — Every TDnet page needs source URL, byte count, SHA-256, request start, receipt completion, parser version, and parse completion. Observed: 0/243 provenance-complete pages
- `JOINT_COVERAGE_BELOW_MINIMUM` (statistical_input) — JPX and TDnet must be jointly provenance-complete for at least the registered minimum number of score sessions. Observed: 0/120 joint score sessions
- `JOINT_MONTHS_BELOW_MINIMUM` (statistical_input) — Jointly provenance-complete score sessions must span at least the registered minimum calendar months. Observed: 0/6 months: []
- `FIELD_EXECUTION_PRICE_OR_FILL_MISSING` (execution_input) — Either exact 08:58 indicative price or an executable order simulation is required. Observed: Neither field has an explicitly registered and content-validated OOT artifact.
- `FIELD_BID_ASK_SPREAD_MISSING` (execution_input) — 08:58 bid/ask spread Observed: No explicitly registered and content-validated OOT artifact was supplied.
- `FIELD_REALIZED_SPREAD_MISSING` (execution_input) — realized spread Observed: No explicitly registered and content-validated OOT artifact was supplied.
- `FIELD_SLIPPAGE_MISSING` (execution_input) — realized slippage Observed: No explicitly registered and content-validated OOT artifact was supplied.
- `FIELD_SPECIAL_QUOTE_0858_MISSING` (execution_input) — 08:58 special-quote state Observed: No explicitly registered and content-validated OOT artifact was supplied.
- `FIELD_OPEN_TIME_OR_DELAYED_OPEN_MISSING` (execution_input) — actual/expected open time and delayed-open handling Observed: No explicitly registered and content-validated OOT artifact was supplied.
- `FIELD_TURNOVER_MISSING` (execution_input) — JPY turnover Observed: No explicitly registered and content-validated OOT artifact was supplied.
- `FIELD_MINIMUM_TRADING_UNIT_MISSING` (execution_input) — effective minimum trading unit Observed: No explicitly registered and content-validated OOT artifact was supplied.
- `FIELD_TICK_SIZE_MISSING` (execution_input) — effective tick size Observed: No explicitly registered and content-validated OOT artifact was supplied.
- `FIELD_PREDICTED_OPENING_TURNOVER_MISSING` (execution_input) — predicted opening turnover Observed: No explicitly registered and content-validated OOT artifact was supplied.
- `CAPACITY_AT_INTENDED_CAPITAL_NOT_EVALUABLE` (execution_input) — Capacity must be checked at the intended capital size. Observed: No registry-bound execution ledger passed lot, notional, turnover, predicted-opening-turnover, and intended-capital checks.

blocker件数は重複し得るfailed requirement数であり、独立した欠測dataset数ではありません。

## Production boundary

このverifierはrepository-hash-boundなdata readinessだけを判定し、外部署名を検証しません。production昇格には、別成果物として署名・hash bindingされた統計結果、execution結果、intended-capital capacity結果、独立承認が必要です。
また、登録historical windowはgenuinely untouchedではないため、事前固定したpaper-liveまたはさらに後年のholdoutが別途必要です。

## Reproduction

No network access is used.

```bash
PYTHONPATH=src:. .venv/bin/python research/model_v11_production_readiness.py
PYTHONPATH=src:. .venv/bin/pytest -q tests/test_model_v11_production_readiness.py
```

- Protocol SHA-256: `5eceb8bf396a703a6a3bf217837331f6af95a7b303c91c79e639d132f575024d`
- Parser audit SHA-256: `317d5cde9741429263cc91c11575300660f94f62123725fc8ce2d76cdee02eeb`
- Data-evidence registry SHA-256: `742224f1b1d85bbfa5142d86346594f6b54d2719d39531257c2a3a9b392ba9e3`
- Search roots: .
