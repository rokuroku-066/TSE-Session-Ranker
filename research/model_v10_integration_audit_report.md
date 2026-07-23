# v0.10 cross-artifact integration audit

## Outcome

All recomputed scheduled-day P&L values match their result files to floating-
point tolerance. Referenced protocol, runner, panel, manifest, score-cache, and
picks hashes also match the current files.

The integration decision nevertheless needs correction:

- no v0.10 artifact supports a production replacement;
- policy Round 2's `retain_for_forward_shadow=true` must be withdrawn because
  its locked specificity rule was not implemented;
- root generation 1 and the external-context track are incomplete relative to
  their own protocols;
- T02 TDnet title value is the strongest point estimate, but only on 88
  discontinuous source-complete sessions and it fails tail, code, paired, and
  familywise robustness.

## Material findings

### 1. Policy Round 2 fails its locked decision rule

The protocol says that momentum-20 must beat at least four of **five**
unrelated split variables and that neither adjacent horizon may reverse the
result:

- `/tmp/v10_policy_universe/protocol_round2.json:90-93`

The runner defines only four unrelated variables (`F03` through `F06`), reports
4/4, and sets `specificity_supported = specificity_wins >= 4`. The adjacent
horizons are placed in the output but are not part of that boolean:

- `/tmp/v10_policy_universe/runner_round2.py:296-304`
- `/tmp/v10_policy_universe/runner_round2.py:318-320`
- `/tmp/v10_policy_universe/runner_round2.py:337-350`

The result therefore cannot satisfy the literal registered criterion:

- `/tmp/v10_policy_universe/result_round2.json:1099-1115`

Corrected decision: withdraw `retain_for_forward_shadow=true`. A missing fifth
comparator cannot be repaired after outcomes are known; any retry must be a
newly labelled, locked forward test.

Round 3 independently leaves H12 rejected as a robust replacement: top-20
uplift removal, monthly uplift, all-four-block, moving-block lower-bound, and
expected-shortfall checks fail.

### 2. Root generation 1 is a partial screen, not completion of its protocol

The protocol registers 18 hypotheses, but the result evaluates only six
(`Z01`, `Z02`, `Z03`, `Z05`, `Z06`, `Z07`) at top 1 and top 2. The other 12
registered IDs are absent.

The same protocol requires a control for each family, moving-block familywise
inference, and target-day outcome mutation. Those outputs are also absent:

- `research/model_v10_zero_base_protocol.json:63-193`

The 12 reported candidate streams and their P&L are arithmetically correct.
`Z06_exp_decay_rank_ridge_hl60__top1` is still only an unadjusted partial-screen
point estimate, not an admissible selected specification.

### 3. External-context results omit registered robustness outputs

All eight registered hypotheses are evaluated at top 1 and top 2, plus the
control. FRED source dates are strictly earlier than their TSE score dates,
with a maximum carry age of three days. The stock beta construction is shifted
and the P&L arithmetic is correct.

However, the protocol also requires full monthly results, a source-date
mutation audit, and a moving-block familywise lower bound:

- `research/model_v10_external_context_protocol.json:79-96`

The result supplies only the positive-month count, three slices, tail removal,
code removal, and a mapping summary. It has no monthly vector, mutation result,
or multiplicity calculation. `X06` therefore remains an unadjusted
exploratory point estimate.

### 4. Online mutation evidence is incomplete

The online runner makes selections before updating state, so its code ordering
is point-in-time. Its P&L is independently reproduced here.

The registered mutation test, however, asks for target-day invariance of the
stateful specifications. The runner strips all outcomes and compares only
`O02_equal_vote_control`, whose selections never depend on outcomes:

- `research/model_v10_online_ensemble_protocol.json:61-75`
- `research/run_model_v10_online_ensemble.py:307-324`

This does not empirically test O01/O03. It does not change their rejection—
both materially underperform C00—but protocol compliance should not be called
complete until the stateful mutation check is recorded.

### 5. Exact duplicates and repeated controls are not independent evidence

Independent key comparison found:

- policy H01 and H04 select the same two names on all 266 sessions;
- TDnet T02 and T10 top 1 select the same name on all 88 sessions;
- the G0 top-2 control is identical across model-architecture, external,
  market-residual, and online artifacts on all 266 sessions.

The duplicate streams do not change the reported max-statistic numerically,
but candidate counts overstate the number of distinct selection paths and the
repeated G0 results are not independent confirmations.

## Corrected compact comparison

Returns are scheduled-day means in percent. `Tail20` is net20 after deleting
the 20 best days. `Code-cash` replaces the ten largest positive
profit-contributing codes with cash. Controls use the same candidate count and
sample as their challenger unless noted.

| Track | Candidate | Days | k | net20 | net40 | net60 | + months at net40 | Tail20 | Code-cash | Audit decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Architecture | M02 extreme-gain Ridge | 266 | 2 | +0.1305 | -0.0658 | -0.2620 | 7/13 | -0.1398 | -0.1433 | reject |
| Architecture | C00 G0 control | 266 | 2 | +0.2553 | +0.0587 | -0.1379 | 8/13 | -0.0054 | -0.0332 | comparison only |
| Root gen1 | Z06 exp-decay HL60 | 266 | 1 | +0.3085 | +0.1138 | -0.0810 | 5/13 | -0.0521 | -0.0677 | incomplete protocol |
| Root control* | X00 G0 top1 | 266 | 1 | +0.2049 | +0.0079 | -0.1891 | 7/13 | -0.1370 | -0.1825 | external control |
| External | X06 context Ridge | 266 | 1 | +0.2623 | +0.0645 | -0.1332 | 6/13 | -0.0780 | -0.0641 | incomplete robustness |
| External | X00 G0 top1 | 266 | 1 | +0.2049 | +0.0079 | -0.1891 | 7/13 | -0.1370 | -0.1825 | comparison only |
| Policy R2 | F08 opposite-75 | 266 | 2 | +0.3518 | +0.1559 | -0.0400 | 8/13 | +0.0517 | +0.0767 | locked rule failed |
| Policy R1/R3 | H12 momentum halves | 266 | 2 | +0.3044 | +0.1082 | -0.0881 | 8/13 | +0.0462 | +0.0197 | robust replacement rejected |
| Policy | L4 control | 266 | 2 | +0.2832 | +0.0866 | -0.1100 | 8/13 | +0.0135 | -0.0114 | comparison only |
| Online | O03 follow leader | 266 | 2 | +0.0823 | -0.1135 | -0.3094 | 7/13 | -0.1616 | -0.1214 | reject |
| Online | C00 G0 control | 266 | 2 | +0.2553 | +0.0587 | -0.1379 | 8/13 | -0.0054 | -0.0332 | comparison only |
| Market residual | R01 residual top1 | 266 | 1 | +0.0390 | -0.1595 | -0.3580 | 7/13 | -0.4010 | -0.3713 | reject |
| Market residual | R00 raw-rank top1 | 266 | 1 | +0.2049 | +0.0079 | -0.1891 | 7/13 | -0.1370 | -0.1825 | comparison only |
| TDnet text† | T02 char-value top1 | 88 | 1 | +0.7191 | +0.5191 | +0.3191 | 3/5 | -0.5721 | -0.1085 | exploratory forward hypothesis only |
| TDnet text† | L4 top1 | 88 | 1 | +0.2200 | +0.0222 | -0.1755 | 3/5 | -0.6226 | -0.4915 | comparison only |

\* Root gen1 did not emit its own control. X00 is the exact same G0 top-1
specification from the external artifact and is shown only as a cross-artifact
reference.

† TDnet covers only July 2024 and April–July 2025. Its values must not be placed
in an unconditional league table with the 266-session tracks.

For T02, the paired net40 delta versus L4 is +0.4968%, but its 90% moving-block
interval is `[-0.0213%, +1.0142%]` and the 10-candidate family reality-check
p-value is `0.2103`. Its own positive interval is therefore not enough to
distinguish it from search luck.

For H12, the familywise one-sided 90% lower uplift is `-0.1439%`; Round 3's
ordinary one-sided lower bound is `-0.0373%`. F08 improves the point estimate,
but its familywise 80% interval versus H12 is
`[-0.1650%, +0.2597%]`, before accounting for the protocol defect.

## Arithmetic and artifact integrity

Independent recomputation used cash for unobserved slots and charged
`cost_bps / 100` percentage points only to executed exposure.

| Track | Recomputed streams | Maximum absolute P&L difference | Scheduled grid |
|---|---:|---:|---|
| Model architectures | 14 | 1.11e-16 | complete |
| Root gen1 | 12 | 1.11e-16 | complete |
| External context | 18 | 1.11e-16 | complete |
| Policy selected/control | 3 | 6.94e-17 | complete |
| Online ensemble | 4 | 0 | complete |
| Market residual | 4 | 5.55e-17 | complete |
| TDnet text | 26 | 2.22e-16 | complete |

Every hash referenced by a result file matches. Two result files do not bind
their picks:

- root gen1 picks SHA-256:
  `0fa208fbbbcde328f4c3cad795979d9f3665024f68d77592b792ff3b777840ed`
- external picks SHA-256:
  `a16efc0b71afa8a73664798733c5cadb516b46df7d7eb1f1b8b3f792563d101e`

Those values should be recorded if these artifacts are retained.

## Corrected admissible decisions

| Track | Corrected status |
|---|---|
| Model architectures | all 13 new candidates rejected |
| Root gen1 | no selection; protocol incomplete |
| External context | no selection; registered robustness incomplete |
| Policy Round 1 | H12 descriptive only; familywise gate failed |
| Policy Round 2 | forward-shadow retention withdrawn |
| Policy Round 3 | H12 robust replacement rejected |
| Online ensemble | O01–O03 rejected |
| Market residual | R01 rejected |
| TDnet title/text | T02 frozen as exploratory forward hypothesis only |
| Overall | no production change |

The forward research queue can retain two ideas without claiming validation:

1. T02's title/body-value direction, tested next on a continuous,
   provenance-tagged TDnet period with numeric PDF surprises.
2. Momentum diversification, newly registered from scratch; neither the
   invalid Round-2 retain flag nor its viewed 75/25 weight may be treated as
   confirmation.

## Audit artifacts

- Detailed machine-readable audit:
  `research/model_v10_integration_audit.json`
- Independent arithmetic:
  `/tmp/v10_integration_audit/calc.json`
- Audit runner:
  `/tmp/v10_integration_audit/audit_calc.py`

The calculation JSON and runner remain temporary independent-audit artifacts;
they are not repository deliverables because they depend on `/tmp` frozen
inputs.
