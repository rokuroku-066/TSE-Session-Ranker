# v1.8 lagged shoulder-state forward hypothesis

## Diagnosis from v1.6 and v1.7

The result that needs explaining is not a stable rank-2 premium. It is a
reversal of the ordering at the top of the frozen price-only model.

| Period | C00 rank 1 gross | C00 rank 2 gross | rank 1 minus rank 2 |
|---|---:|---:|---:|
| v1.6, 2025-12-01 through 2026-03-31 | `-0.1993%` | `+0.5636%` | `-0.7629pt/day` |
| v1.7, 2026-04-01 through 2026-07-27 | `+0.374650%` | `-0.087771%` | `+0.462421pt/day` |

The adjacent-period polarity changed by about `1.2253pt/day`. In v1.7 the
price-only pair model selected rank 2 on every complete day and lost
`-0.477644%/day` after the primary 40 bp haircut. The exact-liquidity reranker
partly corrected that error, but its advantage over C00 rank 1 was only a
post-hoc `+0.0166pt/day`, and no v1.7 candidate survived late-period, tail,
issuer-concentration, median, and paired-lower-bound gates.

These observations reject a fixed rank-2 rule. They do not prove that the
reversal is predictable. Two competing explanations remain:

1. the relative calibration of rank 1 and rank 2 changes slowly and its sign
   persists long enough to be measured from strictly prior OOF outcomes; or
2. the observed reversals are unstable order-statistic noise, in which case a
   lagged state rule will not beat rank 1 on genuinely later data.

v1.8 is a forward test of the first explanation and is allowed to reject it.

## Single registered candidate

The only selection-authorised candidate is
`SH01_LAGGED_MONTHLY_SHOULDER_STATE`. It does not add a feature, tune Ridge,
search a window, or refit during a month. The v1.7 C00 price-only Ridge,
`alpha=1`, feature set, universe, monthly expanding fit, and top-two ordering
remain unchanged.

For each completed calendar month `m`, compute a daily complete-pair difference

```text
d[t] = C00_rank1_open_to_close_return_pct[t]
       - C00_rank2_open_to_close_return_pct[t]
```

and set `month_median[m] = median(d[t])` only when the month contains at least
10 complete OOF pairs. For target month `M`, use the three immediately prior
consecutive calendar months, without skipping an unavailable month:

```text
state[M] = median(
    month_median[M-3],
    month_median[M-2],
    month_median[M-1],
)
```

- `state[M] > 0`: select frozen C00 rank 1.
- `state[M] < 0`: select frozen C00 rank 2.
- exact zero or any unavailable required month: cash.

The state is sealed before the first counted session of the target month and is
constant within that month. Target-month outcomes can never alter that month's
state. There is no zero tolerance, alternate window, fallback rank, or lower
rank replacement.

## Initial state is already determined

The input for the first possible target month is bound to the immutable v1.7
picks artifact, not recomputed from a newly selected historical recipe.

| Registered historical seed slice | Complete pairs | Median rank1-minus-rank2 return |
|---|---:|---:|
| 2026-05 | 17 | `+0.4532617412224217pt` |
| 2026-06 | 19 | `+0.5353494177210093pt` |
| 2026-07-01 through 2026-07-27 (partial) | 18 | `+0.09214571919513584pt` |

Their median is `+0.4532617412224217pt`; therefore an August 2026 state sealed
from these inputs selects rank 1. The July row is a transparent, fixed v1.7
historical exception: v1.7 ended on 2026-07-27, so the registered slice has 18
complete pairs and is not a full completed July. It cannot establish a
partial-month precedent for any forward month.

The bound seed summaries remain usable only while they fall inside a target
month's exact three-month lookback: May is usable for August only, June for
August and September, and July for August, September, and October. They then
roll off naturally; an unavailable intervening forward month is never skipped
to retain an older seed. Sessions before activation are never backfilled. Once
forward collection starts, only pre-open-sealed C00 pairs from counted sessions
may enter new forward monthly medians.

## Why this is not gate optimisation

- There is one candidate and no parameter, sign, capacity, cost, or window
  grid.
- `C00_PRICE_RIDGE_TOP1` is the primary paired control fixed before fresh
  outcomes; `C02_C00_RANK2` is a selection-ineligible diagnostic.
- Reducing the family from ten to one is valid only because it is done before
  the new forward period. It does not retroactively change v1.7's result.
- The v1.7 absolute cost, median, chronological-slice, tail-day, issuer-cash,
  issuer-diversity, concentration, and execution principles remain required.
- The state candidate must have a positive paired point difference as well as
  a nonnegative one-sided 90% moving-block lower bound versus rank 1. Merely
  reproducing rank 1 cannot be called an improvement.

## Forward evidence boundary

Activation is deliberately separate from preregistration. A preregistration
commit freezes this hypothesis, the protocol, transitive runtime lock, runner,
independent audit, tests, scheduled-session calendar, and exact GitHub Actions
workflow bytes. A later activation
payload binds that commit and every artifact hash. A third commit records a
receipt that binds the payload commit. Each of the three commits must be the
exact head SHA of the first completed run-attempt-1 `pull_request` run of that
workflow, and that run itself must succeed. Counting begins on the first TSE
session, no earlier than 2026-08-06, whose 08:58:59 JST cutoff is strictly
after the receipt commit's successful workflow server `updated_at`. That
immutable server time alone selects the session. The post-response observation
must finish before the already-selected cutoff or activation fails and requires
a new preregistration; it cannot move the start later. Committer, author, HTTP
Date, retrieval, and local filesystem times cannot move this boundary.

Those observations are themselves immutable evidence, not hand-entered times
or branch tips. The payload contains normalized GitHub commit and branch
observations for the preregistration commit; the receipt contains the same for
the payload commit; and each decision contains them for the receipt commit.
Alongside each identity pair is a normalized successful workflow-run receipt
for the same exact A, B, or C head SHA. The run ID and stable server projection
are refetched at terminal; a later successful rerun cannot replace an earlier
failure.
The locked Python HTTPS client records a stable response projection, HTTP
evidence, exact response-body digest, and retrieval time. Its TLS trust is not
left to host defaults: the runtime lock fixes one absolute CA file by path,
basename, size, and exact bytes, requires `SSL_CERT_FILE` and `SSL_CERT_DIR`
absent, and permits only `ssl.create_default_context(cafile=registered_path)`
with no CA directory, `cadata`, or caller-supplied trust. This preserves the
same three-commit activation while allowing terminal audit to verify the past
receipts and refetch current commit identity and ancestry. Authentication, if
used, is ephemeral and is never stored or hashed into an artifact.

The runtime lock fixes the exact transitive project files, Python executable
and ABI, every distribution actually loaded by the runner/audit path, native
numerical backends, single-thread fit/score context, Asia/Tokyo zoneinfo, and
PDF parser binary. The A2 runtime-closure repair additionally registers the
host's exact `zstandard==0.25.0` tree and its loaded `backend_c` ELF extension;
this is an operational dependency discovered before any counted outcome and
does not alter SH01, its gates, or its calendar. The closure includes the interpreter-startup
`distutils-precedence.pth` and `_distutils_hack`, the pandas dependencies
python-dateutil, pytz, and six, the scikit-learn dependency charset-normalizer,
setuptools, and both charset-normalizer native extensions. Every live module
origin must resolve to an exact hashed standard-library, project, or registered
distribution file; a new startup hook, user-site/customize module, lazy import,
namespace escape, or unowned site-package file aborts. It also binds every
registered ELF root and resolved recursive shared object used by Python
extensions, numerical libraries, `pdftotext`, and Git, plus the dynamic loader,
`ld.so.cache`, the explicit HTTPS CA file, and an unset critical environment.
Operational validation directly hashes those registered absolute paths; it
does not trust an unpinned linkage tool, host CA discovery, or inherited
child-process environment.
The only originless distribution-generated alias is exact `six.moves`, whose
null origin/file, empty search locations, `six._SixMetaPathImporter` loader,
and hashed `six.py` provider are all fixed; no wildcard alias is accepted.
It deliberately excludes the v1.8 runner, audit, and nonauthority rehearsal to
avoid a hash cycle; the activation payload binds those three files directly
while each pins the already-fixed lock bytes. Operational activation, parsing, fitting,
scoring, and terminal reconstruction abort on runtime drift rather than
silently changing numerical or parsing semantics.

This preregistration is the A2 correction to the never-activated A attempt.
The superseded A bytes produced no activation payload, receipt, decision,
outcome, or result. A2 repairs only the zstandard runtime closure and the
operational path needed to meet the pre-open cutoff; it does not change SH01,
C00, C02, Ridge, the feature set, state formula, costs, gates, or registered
calendar. Activation under the superseded protocol/runtime is forbidden. The
fixed post-merge iteration report records that E60, quota variants, and TD01
produced no selection-authorised signal; they remain development-only
rejections and do not enter production or the A2 forward family.
The A2 preregistration is the first commit on the fresh registered A2 branch
created from current `main` after PR #8 merged; it does not append A/B/C or
daily authority to the superseded experimental branch.

Before payload B, A2 builds one preactivation cache anchor outside the daily
cutoff. The anchor binds every registered historical raw object through `H`,
each standalone parser shard, an exact cumulative raw31 snapshot, the compact
12-column model-price snapshot, and one full raw31-versus-compact12 G0
consumer-equivalence receipt. That receipt compares stable date/code rows,
column order, dtypes, null masks, strings, booleans, and IEEE float bytes
including negative zero for the full prefix plus the first synthetic target.
The payload validates that the exact predecessor of the earliest possible
counted session is strictly later than `H`; actual receipt C may move the start
later but can never make it earlier or reuse an impossible window. A slow
full-raw reference build is untimed and fixes the comparison hashes before the
daily timing gate. The payload-bound nonauthority rehearsal then performs
three cold compact runs for both a month-boundary case and the registered
`intramonth_fold_reuse_upper_bound_proxy`; every run must match the reference
target/fold/score/top-two bytes exactly and finish within 300 seconds or 120
seconds respectively. An untimed strict-runtime preflight first validates the
project/module closure and complete absence of canonical authority, then issues
one opaque process-local capability. The separately timed seam accepts only that
exact capability and performs no repository read/write, network request,
calendar lookup, or module-closure scan; an untimed postflight revalidates the
closure and revokes the capability. It uses only fresh `/tmp` roots and creates
no activation, decision, outcome, checkpoint, or result authority.
Reference and boundary calls compute every exact full-prefix proof once and
bind canonical model-price CSV bytes/rows, source/shard identities, model
semantic, full G0 digest, and pre-month training semantic into the private fold
token. The intramonth proxy must freshly decode its compact snapshot, merge and
canonical-encode the complete prefix, and match every byte/identity binding
before it may reuse only those proof digests. It still freshly builds one panel,
round-trips the target cache, recomputes current fold input hashes, validates the
fold/bundle, and scores. Production follows the same causal boundary: exact
retained month-source/snapshot bytes and bindings may reuse their already sealed
model/training semantics without any caller frame/hash, while current fold
input hashes and the target score remain fresh. Boundary creation and terminal
recompute all semantics. This removes measured redundant proof rehashing; it
does not relax either timing gate or the exact reference comparison.

Predictor PDFs are a declared manual trust boundary. The operator attests that
each labeled file is a faithful acquisition of the named official JPX PDF,
without omission, substitution, or alteration before sealing. A2 proves the
retained bytes and every downstream computation, but has no server receipt,
licensed availability feed, independent acquisition-time proof, terminal JPX
refetch, or independent origin/authenticity proof. URL and receipt time are
operator metadata, and the result must say `official_source_verified=false`.
A missing, partial, ambiguous, mislabeled, late, unparsable, conflicting, or
unavailable source aborts the activated experiment; it never creates cash.
Predictor and outcome authority for the same labeled daily PDF live under
physically disjoint roots as separate single-link inodes, yet their filename,
URL label, receipt metadata, byte count, and SHA-256 must match exactly. The
same bytes may be parsed separately by the two authorities; A2 makes no
parse-once claim for that boundary.

Activation time alone is not sufficient proof that each later shadow decision
was fixed pre-open. Each counted session therefore has two fixed public
checkpoint paths: a safety-cash proposal and a primary proposal. The canonical
`prepare-day` command obtains complete D-1 predictor authority, builds the
compact-prefix panel exactly once, prepares a monthly fold when due, seals the
daily target slice, attaches the prior counted outcome from a physically
distinct raw copy, closes a completed prior month, derives state, seals the
exact two-row score shard, and creates both opaque checkpoint proposals in
that order. Source, fold/bundle, or two-row-pair absence aborts. Only exact-zero
state and unavailable state from the three required prior-month qualifications
authorise cash. `finalize-terminal` alone attaches the final counted outcome,
after the outcome-blind terminal predictor/checkpoint gate.

`prepare-day` is the only daily mutation surface. It loads the exact compact
snapshot through the preceding month, merges only the registered current-month
suffix through D-1, builds the full-prefix G0 panel once, and derives both a
month-boundary training slice/fold and the outcome-null D target slice from
that same object. A recent-window approximation, incremental rolling G0
state, caller-supplied panel, split month/source/state/top2/checkpoint command,
or daily cumulative raw reparse is not registered. At terminal, each unique
predictor raw object is reparsed exactly once, each standalone text/report/
shard is compared, and raw31 rebuilds every monthly training fold and daily
target cache before state, outcome, score, or performance is opened.

The local checkpoint pair uses precreated private directories, directory locks,
fsync, staged no-replace publication, fixed intent bytes, and exact replay so a
local crash can be completed without changing a nonce, timestamp, decision, or
byte. Conflicting, aliased, unsafe, or unexplained partial state aborts. The
threat model excludes a same-UID process that deliberately bypasses the
registered cooperative lock; group/world-writable parents are forbidden. A
second noninteractive command uses only the locked Python
HTTPS client and GitHub Git Data API to publish safety first and primary second
as adjacent sole-parent, one-file commits on the registered branch while
keeping the preregistered Actions workflow byte-identical. It creates the exact
blob, derived tree, and sole-parent commit, then advances the fixed ref with
`force=false`; before mutation it verifies the recursive parent tree is
complete, every protected blob is unchanged, and the new path is absent. A
branch race or uncertain mutation response stops all further mutation; the
publisher never retries a mutation. A later process may perform read-only
recovery and continue only if the independently reconstructed immutable remote
commit/tree/blob/path/workflow state is exactly canonical. No
Git remote helper, libcurl, SSH, credential helper, or persisted token enters
the operational boundary. The
primary path, commit, and first attempt-1 workflow run are mandatory, and that
run's GitHub server `created_at` must be strictly before 08:58:59 JST. A
missing, late-created, ambiguous, partial, reordered, or otherwise noncanonical
primary is an outcome-blind integrity abort, never discretionary cash. The
exact remote state is the scientific authority: terminal audit does not claim
to prove that every prescribed pre-mutation read, workflow poll, or post-PATCH
verification call completed, and it does not reject a byte-identical canonical
chain solely because another construction process produced it.

Primary is used unconditionally when its first run object has server
`created_at` strictly before cutoff. Its eventual status, conclusion,
`run_started_at`, and `updated_at` cannot change that role. The process may wait
outcome-blind for stable terminal evidence, but failure to materialize that
evidence before attaching the target outcome aborts the experiment rather than
creating cash. The historical `safety_cash` role is only a mandatory opaque
publication predecessor: its envelope contains an independently nonce-protected
copy of the exact same decision core and is never a resolution. Retrieval time,
later runs, or alternate commits can never upgrade or downgrade the primary.

Publishing the decision itself would break the sealed experiment, so the Git
proposal contains no state, rank, code, score, failure reason, core hash, or
nonce. It binds only a safe external object key, constant byte count, and
SHA-256. That append-only object is a fixed 16,384-byte binary envelope whose
declared canonical-JSON payload contains a fresh hidden 256-bit nonce and the
exact decision core; deterministic zero fill prevents length leakage.
Operational code may use it without disclosing it; humans,
terminal outputs, and Git cannot reveal it before terminal. At terminal the
runner and independent audit securely reopen it through a retained directory
file descriptor, verify the public commitment and exact remote Git Data
blob/tree/commit/compare and run ancestry, enumerate the entire checkpoint
path and workflow history to reject hidden
alternatives, and independently reproduce the decision chain and result input
hashes. This breaks no hash cycle: the opaque core exists before its proposal
commit, while the sealed decision later binds that commit and workflow proof.
The last counted primary commit is the terminal checkpoint tip. A later result,
audit, or documentation commit may make the current branch tip newer, but
remote paginated history must still prove that it descends from the terminal
tip and never changed a proposal, workflow, or preregistered artifact.
Daily decisions are a closed four-value registry: selected rank 1, selected
rank 2, cash because a required prior month is unavailable, or cash because the
exact state is zero. Checkpoint safety adds no decision or failure reason.
Source/model/pair failure terminates the experiment and cannot become a fifth
decision. Exception text or caller prose cannot become a hidden label.

Every scheduled session from that point is retained. Missing source or model
authority is an irreversible integrity abort, not a delayed start, cash, or a
replaceable observation.
The preregistered, hash-bound calendar registry spans 2026-08-05 through
2027-12-30, while A2's fixed not-before rule makes 2026-08-06 the earliest
possible counted session; the unused 2026-08-05 registry row is retained for
calendar identity and helper determinism. A later source failure cannot rewrite
the counted denominator.
The terminal date is the last scheduled session of the first calendar month-end
at which at least 120 scheduled sessions and at least six distinct calendar
months have been counted.
For the earliest permitted start of 2026-08-06, session 120 is 2027-02-03 and
the fixed terminal month-end is 2027-02-26, giving 135 scheduled sessions
across seven represented months, split 67/68, with at least 122 executed days
required.

Each decision, outcome, and completed-month record is first sealed as a
create-once per-key authority shard. Its JSONL is only an exact recoverable
derived view: a torn suffix is rebuilt from shards, while divergence, extras,
aliases, reordered keys, or a hash-chain break abort. Score authority uses the
same per-session pattern. Referenced content-addressed derived objects are
authoritative only through their bound manifests and hashes; unrelated objects
are explicitly nonauthority and no complete-store-enumeration claim is made.

An automated sealed process may ingest completed outcomes to form the next
month's strictly lagged state. The registered May-through-July seed values and
the resulting initial August rank-1 state above are public preregistration
facts. After activation, forward-derived monthly states, selected codes,
realised returns, and aggregate performance remain sealed until the
deterministic terminal closeout. Humans may inspect source-health and
hash-chain telemetry only.

Terminal metrics remain unreachable until the terminal session's outcome is
linked and the terminal calendar month's completed-month record has been
hash-chain and coverage validated. Only then are the three-row-per-session
picks, every input binding, metrics, gates, and canonical result independently
recomputed and compared byte for byte. If an integrity failure occurs instead,
the sole post-C `abort` path irreversibly terminates the activation and may
create one outcome-blind result under the exact canonical schema. It records
only a finite generic reason/stage, strict runtime identity, the non-null C
activation identity, and opaque byte fingerprints of already-existing files.
It cannot parse or expose decisions, outcomes, scores, picks, or performance;
all metrics, gates, and nominee fields are null. Once it exists every semantic
or mutating command is locked out. `evaluate` reads only the final sorted-JSON
status line to recognise the abort and opens no performance authority.

## Interpretation

Passing v1.8 would establish at most one forward research nominee for a
separate sealed confirmation. It would not prove that the mechanism caused the
regime change, measure executable transaction costs, change the production
model, or permit orders. Failing any registered gate rejects SH01. The same
fresh period cannot be reopened to change the state definition or introduce a
runner-up.
