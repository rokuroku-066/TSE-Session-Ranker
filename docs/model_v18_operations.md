# v1.8 A2 outcome-blind operations

**ops/model_v18_operations.py** is a non-executing planner around the
canonical v1.8 A2 runner. It does not import the runner, score a model, inspect
performance, publish Git data, or enable orders. Its source inventory and plan
files are operator conveniences, never scientific authority.

## Provenance boundary

Every source must be acquired and supplied manually. This helper performs no
network retrieval. A registered source records:

- the exact retained PDF bytes, byte count, and SHA-256;
- its canonical JPX URL label;
- the operator-attested time the exact bytes were first possessed;
- mode "manual_operator_attested_v1";
- "official_source_verified=false"; and
- the fixed provenance caveat ID and text hash.

The label and operator attestation do not independently prove server origin,
availability, acquisition time, or authenticity. The runner proves the bytes
and computation only after sealing. Never invent a receipt time, substitute a
mirror, or describe the label as independently verified.

## Private disjoint roots

Create one operations root and four pairwise-disjoint sibling store roots
outside the repository:

~~~bash
python -B ops/model_v18_operations.py init \
  --ops-root /durable/tse-v18/operations \
  --predictor-raw-store-root /durable/tse-v18/predictor-raw \
  --predictor-derived-store-root /durable/tse-v18/predictor-derived \
  --outcome-raw-store-root /durable/tse-v18/outcome-raw \
  --checkpoint-core-store-root /durable/tse-v18/checkpoint-core
~~~

All five roots must be owner-operated mode 0700, must not contain one another,
and must not alias an inode. The create-once layout records their absolute
paths and inode identities. Do not move, symlink, hard-link, chmod, or reuse
any root for another role.

The operations root contains only manually staged inputs, a source-only
inventory, and non-executing plans. Only the runner writes scientific raw,
derived, outcome, checkpoint, and local canonical authorities.

## Register manual inputs

Register each locked historical PDF before building the anchor:

~~~bash
python -B ops/model_v18_operations.py register-source \
  --ops-root /durable/tse-v18/operations \
  --kind price_warmup \
  --source-file /incoming/202505.pdf \
  --source-url 'https://www.jpx.co.jp/markets/statistics-equities/daily/tvdivq0000001jan-att/202505.pdf' \
  --received-at '2026-08-05T00:15:00Z'
~~~

Register each forward daily file in the same way:

~~~bash
python -B ops/model_v18_operations.py register-source \
  --ops-root /durable/tse-v18/operations \
  --kind daily \
  --source-file /incoming/stq_20260805.pdf \
  --source-url 'https://www.jpx.co.jp/markets/statistics-equities/daily/exacttoken-att/stq_20260805.pdf' \
  --received-at '2026-08-05T07:30:00+09:00'
~~~

The daily label must use exact host "www.jpx.co.jp", the canonical
"/markets/statistics-equities/daily/<lowercase-token>-att/<filename>" path,
and no query, fragment, encoding, dot segment, or host alias. Historical
hash/size/URL bindings are checked where already frozen. An identical
registration is accepted as an exact retry; any changed field is rejected.

Source-only readiness can be checked without opening model or performance
evidence:

~~~bash
python -B ops/model_v18_operations.py source-health \
  --ops-root /durable/tse-v18/operations \
  --phase anchor
~~~

## Freeze the preactivation stores and predictor anchor

After preregistration commit A and its CI are final, create the store-readiness
plan:

~~~bash
python -B ops/model_v18_operations.py plan-stores \
  --ops-root /durable/tse-v18/operations \
  --format shell
~~~

Its only runner sequence is:

1. validate-runtime
2. prepare-operational-stores with all four fixed roots

After all 248 manual anchor files through 2026-08-04 are present, create the
predictor-cache plan:

~~~bash
python -B ops/model_v18_operations.py plan-cache \
  --ops-root /durable/tse-v18/operations \
  --format shell
~~~

The plan supplies the exact ordered anchor registry to
prepare-predictor-cache, fixes "--through 2026-08-04", and supplies only the
predictor raw and predictor derived roots. Missing anchor bytes are a blocker;
there is no partial-anchor path.

Plans are persisted mode 0600 under the operations root. Repeating the same
request accepts only byte-identical plan content. A changed retry is rejected.
Printing shell form still does not execute a command.

## Canonical A/B/C activation

Create the payload plan after commit A is observable and its exact workflow
has succeeded:

~~~bash
python -B ops/model_v18_operations.py plan-activation \
  --ops-root /durable/tse-v18/operations \
  --stage payload \
  --commit-sha AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA \
  --predictor-cache-anchor-manifest-object-key \
    model_v18_shoulder_state/cache-anchor/EXACT_KEY/manifest.json \
  --format shell
~~~

Execute the emitted runner command, commit the canonical payload as commit B,
and wait for B's exact successful workflow. Then plan the receipt:

~~~bash
python -B ops/model_v18_operations.py plan-activation \
  --ops-root /durable/tse-v18/operations \
  --stage receipt \
  --commit-sha BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB \
  --format shell
~~~

Commit the canonical receipt as sole-child commit C, wait for C's exact
successful workflow, and create the local activation context:

~~~bash
python -B ops/model_v18_operations.py plan-activation \
  --ops-root /durable/tse-v18/operations \
  --stage preflight \
  --commit-sha CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC \
  --format shell
~~~

The planner never supplies payload, receipt, output, or activation-context
path overrides. The only activation context is
**research/model_v18_shoulder_state_activation_context.json**. Do not copy or
replace it with a durable external JSON file. Do not activate until a reliable
scheduler and the required GitHub token are available.

## Daily sequence

Manually register every new suffix PDF before the target cutoff. For the first
counted session, the planner supplies exactly every source later than anchor H
through exact D-1. For every subsequent session, it supplies exactly the one
new D-1 file. Historical or prior-forward files are not resubmitted.

Check readiness and create the deterministic plan:

~~~bash
python -B ops/model_v18_operations.py source-health \
  --ops-root /durable/tse-v18/operations \
  --phase daily \
  --session 2026-08-06

python -B ops/model_v18_operations.py plan-day \
  --ops-root /durable/tse-v18/operations \
  --session 2026-08-06 \
  --as-of '2026-08-06T07:45:00+09:00' \
  --format shell
~~~

The only daily runner sequence is:

1. validate-runtime
2. prepare-day
3. publish-checkpoint
4. decide

prepare-day is the sole source/month/fold/state/score/checkpoint preparation
path. On later days it also copies exact D-1 bytes into the physically
distinct outcome authority, attaches the already sealed prior decision
outcome, and closes a month when required. No split mutation command is
authoritative.

A plan timestamp is not cutoff evidence. All source receipt, preparation,
checkpoint publication, and workflow timing rules are enforced again by the
runner. Start with enough margin for the measured cold path and remote
workflow exposure.

Missing, partial, ambiguous, late, conflicting, or unparsable source evidence
is a global experiment integrity failure. Do not publish, decide, omit the
session, or convert that failure to cash. The only legitimate daily cash
decisions are those the complete-data runner derives from exact
state-insufficient or exact-state-zero rules.

prepare-day and decide permit only their registered exact-input recovery.
publish-checkpoint is the remote compare-and-swap boundary: never blindly
retry it after a response that may have crossed the ref update. Continue with
decide only for the runner's fixed read-only reconstruction; a missing,
half-published, late, or noncanonical pair integrity-aborts.

## Terminal sequence

After the terminal decision is sealed, manually register the terminal
session's own daily PDF. Then:

~~~bash
python -B ops/model_v18_operations.py source-health \
  --ops-root /durable/tse-v18/operations \
  --phase terminal

python -B ops/model_v18_operations.py plan-terminal \
  --ops-root /durable/tse-v18/operations \
  --session YYYY-MM-DD \
  --as-of 'AWARE-ISO-8601-TIMESTAMP' \
  --format shell
~~~

The terminal order is fixed:

1. validate-runtime
2. finalize-terminal
3. evaluate

finalize-terminal must finish the last outcome and terminal-month authority
before evaluate may open performance evidence. Both commands use all four
layout roots and canonical local artifact defaults.

## Global integrity abort

After activation, create an outcome-blind terminal abort plan with one
registered stage/reason pair:

~~~bash
python -B ops/model_v18_operations.py plan-abort \
  --ops-root /durable/tse-v18/operations \
  --integrity-stage source_ingestion \
  --failure-reason source_integrity_failure \
  --format shell
~~~

An abort plan contains only the runner's abort command. It has no store,
ledger, outcome, score, pick, evaluation, or custom context argument. Repeating
it is legal only with the identical stage and reason. An activated aborted run
consumes its registered experiment slot; abort is not a cash day.

## Referenced-only monitoring

~~~bash
python -B ops/model_v18_operations.py health \
  --ops-root /durable/tse-v18/operations \
  --session YYYY-MM-DD
~~~

Health validates only manually registered staged sources, stats a short exact
set of canonical paths, and stats the four root directories themselves. It
does not walk an external store, enumerate unreferenced objects or staging
names, read scientific artifact bytes, open outcome paths, or parse
performance semantics.

## Strict launch and token handling

Every planned runner step uses the runtime-lock Python executable and unsets
every environment name whose registered loader value is null, including
PYTHONPATH, all registered LD controls, Python startup controls,
PYTHON_ZSTANDARD_IMPORT_POLICY, SSL_CERT_FILE, and SSL_CERT_DIR.

GitHub-observing and publishing runner commands may receive the task-relevant
GITHUB_TOKEN from the execution environment. The helper never reads, persists,
prints, or embeds its value in a layout or plan.
