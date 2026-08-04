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
session, no earlier than 2026-08-05, whose 08:58:59 JST cutoff is strictly
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
PDF parser binary. The closure includes the interpreter-startup
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
It deliberately excludes the v1.8 runner and audit to avoid a hash
cycle; the activation payload binds those two files directly while both pin
the already-fixed lock bytes. Operational activation, parsing, fitting,
scoring, and terminal reconstruction abort on runtime drift rather than
silently changing numerical or parsing semantics.

Activation time alone is not sufficient proof that each later shadow decision
was fixed pre-open. Each counted session therefore has two fixed public
checkpoint paths: a safety-cash proposal and a primary proposal. One role-free
command derives and seals both cores and creates both opaque proposals as an
indivisible pair. A second noninteractive command uses only the locked Python
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
Daily nonselection reasons are a closed finite registry: missing versus partial
source, unavailable fold versus unavailable pair, and insufficient state versus
exact-zero state. Checkpoint safety adds no decision or failure reason. Exception
text or caller prose cannot become a hidden experimental label.

Every scheduled session from that point is retained. Missing source or model
state is fail-closed cash, not a delayed start or a replaceable observation.
The scheduled-session denominator is the preregistered, hash-bound TSE calendar
from 2026-08-05 through 2027-12-30; a later source failure cannot rewrite it.
The terminal date is the last scheduled session of the first calendar month-end
at which at least 120 scheduled sessions and at least six distinct calendar
months have been counted.

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
the sole `abort` path may create one outcome-blind result under the exact
canonical schema. It records only a finite generic reason/stage, strict runtime
identity, activation hashes that can be validated without performance access,
and byte hashes of already-existing canonical files; it cannot parse or expose
decisions, outcomes, picks, or performance and cannot overwrite any result.

## Interpretation

Passing v1.8 would establish at most one forward research nominee for a
separate sealed confirmation. It would not prove that the mechanism caused the
regime change, measure executable transaction costs, change the production
model, or permit orders. Failing any registered gate rejects SH01. The same
fresh period cannot be reopened to change the state definition or introduce a
runner-up.
