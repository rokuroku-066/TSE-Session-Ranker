# v1.8 post-merge hypothesis iteration ledger

## Authority and decision

This ledger records two fixed post-merge diagnostics on the already observed
v1.7 period.  Both are **development-only**: they are neither untouched
forward evidence nor grounds for production promotion.  Both hypotheses are
rejected, the production model is unchanged, and `orders_allowed=false`.

No threshold, score penalty, issuer cap, quota, rank window, cost, or gate may
now be swept or tuned on these outcomes.  In particular, a softer E60
threshold, a larger issuer quota, or a score-dependent diversification penalty
would be a new outcome-informed search and is prohibited on this period.

## E60 fixed historical prototype: rejected

The single registered mechanism was an exceedance classifier:

```text
training universe/folds/weights: frozen C00 G0, monthly expanding,
                                 date-equal weights
label:                          1[open-to-close return > 0.60 percentage point]
model:                          LogisticRegression(C=1, penalty=L2,
                                 solver=lbfgs, max_iter=100,
                                 class_weight=None)
selection:                      probability descending, then code ascending
```

The diagnostic harness first reproduced the saved C00 selection codes on
`79/79` scheduled slots; maximum absolute C00 score difference was
`8.33e-17`.  E60 then produced:

| metric | result |
|---|---:|
| net mean at 40 bp | `-0.012660%` per session |
| net median at 40 bp | `+0.480282%` |
| net mean at 60 bp | `-0.210128%` per session |
| fixed first / second half, net 40 bp | `-0.216119%` / `+0.185712%` |
| positive months | `2/4` |
| top four profit days removed, net 40 bp | `-0.338585%` |
| top five profit codes made cash, net 40 bp | `-0.446063%` |
| paired delta versus C00, net 40 bp | `-0.002500` percentage point |
| moving-block one-sided 90% lower bound, block 5 | `-0.445906` percentage point |
| moving-block one-sided 90% lower bound, block 20 | `-0.301877` percentage point |
| unique selected codes | `58` |
| maximum code share | `7.69%` |
| top-ten code share | `37.18%` |
| executed slots | `78/79` |

A fixed one-selection-per-code diagnostic repaired breadth mechanically
(`78` unique codes, `1.28%` maximum share, `12.82%` top-ten share), but returns
collapsed: net-40 mean `-0.278103%`, net-40 median `-0.400000%`, net-60 mean
`-0.475572%`, fixed halves `-0.668506%` / `+0.102539%`, and paired delta
`-0.267943` percentage point.  E60 therefore does not supply a robust alpha,
and issuer capping is not a missing sufficient condition.

## C00 strict issuer-quota reroute: rejected

### Fixed rule and binding checks

For a registered denominator of `N` slots, use the single strict integer cap

```text
q(N) = ceil(N / 40) - 1.
```

The top-ten constraint is binding: `10q/N < 0.25` is equivalent to
`q < N/40`; it also implies the weaker per-code constraint `q/N < 0.05`.
Thus `N=79` gives `q=1`, without a grid or fitted parameter.  In chronological
order, the rule selects the highest C00-scored eligible code whose
**strictly-prior** selection count is below `q`, breaking score ties by code
ascending.  It scans deeper ranks and never goes to cash merely because a
higher-ranked issuer reached the cap; only an empty registered source universe
can produce source cash.

Before evaluation, the diagnostic verified all three warm-up and all `239`
daily source-byte locks.  The daily source-set SHA-256 was
`203e996f5285412643a45c4135ef535bfeed9e6d0d4988a7156fba017d2e9381`.
It also verified the persisted protocol, input lock, score ledger, and outcome
ledger hashes, plus the raw-derived compact snapshot SHA-256
`b99a7331940fe367da292cb3d0e0111fbdc25bcc8761cbc3ba03cd95a35bc26a`.
The full C00 cross-section was recomputed before rerouting.  Recomputed versus
saved C00 code, name, rank, and IEEE score each matched `79/79`; rejoined label
and open-to-close outcome each matched `79/79`.

### Exact results

The rule signaled `78/79` slots and executed `77/79`.  It rerouted `50` slots,
kept the saved C00 choice on `28`, and selected `78` different codes.  Selected
source rank had mean `2.538462`, median `2`, 90th percentile `5.3`, and maximum
`8`.  Consecutive signaled codes changed on `100%` of transitions.

| metric | result | gate |
|---|---:|:---:|
| gross mean | `-0.092776%` | -- |
| net mean at 20 bp | `-0.287713%` | -- |
| net mean at 40 bp | `-0.482650%` | FAIL |
| net median at 40 bp | `-0.374113%` | FAIL |
| net mean at 60 bp | `-0.677587%` | FAIL |
| compounded net return at 40 bp | `-32.783169%` | -- |
| fixed first / second half, net 40 bp | `-0.650562%` / `-0.318936%` | FAIL |
| positive months | `0/4` | FAIL |
| top four profit days removed, net 40 bp | `-0.667032%` | FAIL |
| top five profit codes made cash, net 40 bp | `-0.662958%` | FAIL |
| paired delta versus saved C00, net 40 bp | `-0.472490` percentage point | -- |
| moving-block lower bound, block 5, 20,000 draws, 90% | `-0.871276` percentage point | -- |
| familywise moving-block lower bound, same draws, 99% | `-1.149544` percentage point | FAIL |
| unique selected codes | `78` | PASS |
| maximum code share | `1.282051%` | PASS |
| top-ten code share | `12.820513%` | PASS |
| executed slots | `77/79` | PASS |
| executed-slot fraction | `97.468354%` | PASS |

All four monthly net-40 means were negative: April `-0.135193%`, May
`-1.251825%`, June `-0.201897%`, and July `-0.461983%`.  Only the five breadth
and execution gates passed; the complete 13-gate conjunction failed.  The
mechanism is structurally unsuitable: it guarantees diversification by
forcing progressively deeper C00 ranks, but destroys rather than broadens the
historical alpha.

For the A2 earliest-start denominator `N=135`, the same untuned formula gives
`q(135)=3`: the registered-denominator bounds are maximum share
`3/135=2.222222%` and top-ten share `30/135=22.222222%`.  If a future protocol instead computes
concentration over actual signaled slots after forced source cash, it must bind
that denominator before activation; `q=3` alone does not guarantee a strict
25% top-ten bound when fewer than `121` signals remain.

## A2 execution-path feasibility: engineering evidence only

The first post-merge rehearsal exposed a non-statistical blocker.  Replaying
the then-current 247-PDF corpus once took `779.193` seconds, while the frozen
daily path replayed it up to three times at a month boundary.  A late D-1 file
could therefore make an otherwise valid pre-open decision operationally
impossible.  This timing is not model performance evidence, but activation on
that path would have converted infrastructure latency into selective missing
days.

The A2 repair keeps every raw PDF as terminal computational authority and adds
only content-addressed derived evidence: one canonical parsed shard per PDF, a
monthly cumulative 12-column model-price snapshot, and a daily target-slice
cache.  A fresh compact-only rehearsal on the locked corpus completed exact
CSV decode, suffix merge, full-prefix G0 construction, projection, monthly fit,
all `2,664` scores, and top-two selection in `67.554` seconds, with peak RSS
`3.815 GiB`; hashing all referenced objects adds about `1.501` seconds.

The acceleration was admitted only after a raw-31-column versus compact-
12-column proof.  The two paths produced a bit- and dtype-exact
`1,214,734 x 22` G0 panel, identical fitted Ridge arrays, identical eligible
code set, identical IEEE score bytes (score-set SHA-256 prefix `49db5958`),
and the same top two codes, `3920` and `6063`.  A truncated 100-session path was
rejected even though it selected the same pair: pandas rolling accumulators
retained prefix-dependent floating-point state and changed exact bytes.  A2
therefore always reconstructs the full model-price prefix; it does not use a
lookback shortcut.  Activation still requires the protocol-bound full/compact
consumer receipt, independent terminal reconstruction, three cold integrated
rehearsals, and the complete audit/test suite.  These timing and equivalence
checks do not alter SH01, its thresholds, or its gates.

## Next admissible hypothesis family

The next test must add genuinely new, point-in-time information and use a fresh
non-overlapping outcome period.  The acquisition and leakage contract remains
the existing [model_v10_preopen_data_report.md](model_v10_preopen_data_report.md):

1. First priority is the 08:58 TSE opening-auction order book together with
   same-time OSE index futures.  Freeze the candidate universe and snapshot
   times before outcomes, retain event/receipt/computation timestamps, and do
   not substitute the 09:00 open or later order events.
2. The more practical lower-cost route is official TDnet structured numeric
   data: forecast revisions, dividends, and share-buyback magnitude, accepted
   only when publication, receipt, and computation all finish by 08:58:59.
3. Feature definitions, missingness, cost, gates, and the one candidate
   mechanism must be frozen before opening the fresh outcomes.  The E60 and
   quota period cannot be reused as confirmation evidence.

### TD01 fixed candidate: proposed, not evaluated

The first concrete member of that family is `TD01_EPS_REVISION_OVERRIDE`.
It is a proposal only; no historical or forward return has been opened for
this rule.  For each C00-eligible issuer with one unambiguous, effective
full-year EPS forecast revision available by the decision cutoff, define

```text
u_i(D) = 100 * (EPS_new - EPS_old) / unadjusted_close_i(D-1).
```

`EPS_new`, `EPS_old`, and the close must be parsed from exact decimal strings
on the same fiscal-period, consolidation, and D-1 share basis.  Positivity is
tested on the exact EPS difference, and cross-issuer ordering uses exact
cross-multiplication; no floating-point rounding, clipping, winsorisation, or
threshold search may change the winner.  Select the largest strictly positive
`u`, breaking an exact tie by normalized code ascending.  If the complete
source contains no positive revision, retain frozen C00 rank 1.  If the source
snapshot is incomplete, ambiguous, or has an unresolved correction/share-
basis chain, the registered experiment integrity-aborts before any candidate
decision.  It is not a candidate-cash option.  A future independently signed
and preregistered negative-availability authority could define a different
fail-closed design, but local omission by an operator cannot do so.

The historical acquisition contract is the blocking condition.  Bytes
downloaded now cannot acquire a fictitious 2021--2026 receipt timestamp.
Retrospective development is permitted only if an official or licensed
provider supplies a gap-free as-of snapshot with an authenticated immutable
document version, publication/sequence authority, correction ancestry, and
finality as of each old cutoff.  Otherwise TD01 remains unevaluated and moves
directly to an outcome-blind, fixed 20-consecutive-session live source/parser
pilot.  That pilot may expose only acquisition and parsing health, never
prices, selections, labels, returns, or gate progress.

The post-merge inventory contains no qualifying new-information corpus.  The
only retained market originals are the 248 JPX daily/monthly price PDFs used
by the C00 lineage.  There are no retained 08:58 TSE order-book, OSE futures,
or PTS snapshots.  The sole TDnet probe manifest records zero candidate
events, an incomplete source, no clock-synchronisation evidence, and receipts
after the decision cutoff; its referenced response bodies are not retained.
Public TDnet pages can support network/parser/stability QA, but without an
atomic snapshot identifier, monotone delivery sequence, cutoff-completion
watermark, and correction/cancellation ancestry they remain `partial`.  Such
an observation fails the fixed 20-consecutive-registered-session pilot; it
does not extend, restart, or roll the pilot until 20 convenient valid days have
accumulated.  “No extracted event” must not be reinterpreted as a complete
no-event day.

The same anti-discretion rule applies to the repaired SH01 forward run.  Until
there is an independently timestamped, mandatory acquisition attempt with a
canonical negative response, a missing, partial, ambiguous, or unparsable
D-1 predictor source is an integrity abort, not a counted cash day.  An
operator-provided empty vector cannot prove non-availability.  Failure to run
the canonical day also leaves the registered denominator incomplete and can
never improve a gate.

The repaired A2 implementation deliberately does not claim independent
official-source authentication.  Its future JPX PDF acquisition is a declared
manual operator-attested boundary: it binds a syntactically canonical JPX URL
label, filename, operator-reported receipt time, byte count, and body SHA-256
before the cutoff, and all subsequent artifacts are tamper-evident against
that seal.  The URL label and receipt time are not server evidence, and A2 does
not prove cryptographically that JPX served those bytes.  Missing, conflicting, or
unparseable input still aborts the experiment.  A stronger source-authenticated
version requires a separately preregistered, permissioned or licensed feed (or
an independent signed acquisition service); it must not silently add mandatory
bulk retrieval to a site whose published usage guidance discourages automated
collection.

Likewise, failure to create the registered monthly fold or the exact two-row
score pair is an integrity abort, not model cash.  Those files are outputs of
deterministic locked code, so their absence only proves that the computation
was not completed.  The only strategy-level cash states retained by SH01 are
an exactly replayed insufficiency across the three closed prior months and an
exact zero state median.  Before either can be used, the prior D-1 outcome and
any month close must be sealed from the distinct outcome raw authority; an
omitted prior outcome cannot manufacture an “insufficient” state.

After a passing pilot, the implementation and source manifest are sealed and
`S0` is the first registered session of the next calendar month.  The terminal
session is the first registered month-end covering at least 120 scheduled
sessions and six complete represented months; coverage, overrides, and
outcomes cannot extend or restart it.  All scheduled sessions are in every
performance denominator, with candidate cash represented by zero.  A missing
or corrupt outcome is an integrity abort rather than zero.

Sequential terminal tests use non-recycled **nominal** alpha spending

```text
alpha_k = 0.05 / 2^k,  k = 1, 2, ...
```

so the arithmetic sum of the nominal levels is at most `0.05`.  Because the
registered moving-block percentile interval is an approximate resampling
procedure, this is not asserted to be a finite-sample FWER theorem.  Increment
`k` when a preregistered hypothesis reaches its activation-C commitment, not
only when it later unseals terminal outcomes.  An activated run that aborts,
is abandoned, or lacks a denominator record permanently consumes that slot
and its fixed window; neither the nominal level nor any session can be
recycled.  A preregistration that never reaches activation C never starts its
window.  Each `k` admits one preregistered mechanism, not an outcome-informed
threshold or penalty sweep.

SH01 is conservatively charged as slot `k=1` if A2 reaches activation C, even
though its older development gate remains the already frozen 90% moving-block
lower bound; that gate is not retroactively relabelled as a 97.5% test.  The
next prospective confirmation actually launched reserves `k=2` and a nominal
one-sided level `0.0125` before its first observation.  Thus a v1.9 SH01
confirmation launched first takes `k=2` and moves TD01 to `k=3`; if TD01 is
launched first, TD01 takes `k=2`.  Slot order is determined by activation C,
never by which result later looks preferable.
