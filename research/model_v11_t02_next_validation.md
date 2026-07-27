# T02 次回検証計画 — forward protocol v2

## 結論

次に行う検証は、新しい特徴量・モデル・閾値の探索ではない。凍結した
`v10_t02_char_value_event_top1`を変更せず、次の順序で証拠を積み上げる。

1. forward protocol v2とactivation payloadをhash固定し、独立timestamp receiptでactivationする
2. 登録済みhistorical OOT入力をprovenance付きで復元する
3. exact no-tuning OOT統計gateを実行する
4. activation後のpaper-live shadowを120 scheduled営業日以上収集する
5. execution・capacity gateを独立監査する

いずれかの前段がfailした場合、後段をproduction根拠として実行しない。

## このPRの範囲

このPRは`research/model_v11_t02_forward_protocol_v2.json`を追加し、次を明文化する。

- 評価損益は**target sessionの始値→終値**である
- T02は月初ごとのmonthly expanding walk-forwardで再学習する
- 月中再学習、hyperparameter変更、欠落sourceのno-event化を禁止する
- 凍結runnerと同じ`price_training_eligible`・`price_eligible` maskを使う
- 月初前にpre-score foldをsealし、予測・結果は月末closeoutへ分離する
- activation payloadをdefault branchへ置き、独立timestamp receiptが発行される前のsessionは数えない
- activation後はsource欠落・model失敗日もcash 0として全scheduled-session分母へ残す
- production変更と実注文を引き続き禁止する

既存の`research/model_v10_forward_protocol.json`は、後続artifactからhash参照される
履歴記録なので上書きしない。旧protocolの
`target-session close-to-open return`は記述上の誤りとして残し、v2のerrataで
`target-session open-to-close return`へ訂正する。

既存OOT protocolの「incomplete source means cash」も曖昧なのでerrataを追加する。
historical OOTではsource欠落日をsource-complete分母から除外し、no-eventとは扱わない。
一方、prospective shadowでは運用障害を含む実システム性能を測るため、
activation後の全scheduled sessionを主分母に固定する。

## Phase 0 — protocol activation

### 2段階activation

1段目の`research/model_v11_t02_forward_activation_payload.json`は、次を固定して
default branchへcommitする。

- protocol path・SHA-256
- repository ID・default branch
- 将来の`not_before_session`
- 学習snapshot、source-registry schema、scoring codeのpath・SHA-256
- eligibility codeとuniverse configのpath・SHA-256
- 承認者

payload自身を含むcommit SHAをpayloadへ書くことはできない。file内容が変わると
treeとcommit SHAも変わるため、これは自己参照になるからである。

2段目では独立したtimestamp/attestation serviceがdefault branch上のpayloadを
観測し、`research/model_v11_t02_forward_activation_receipt.json`相当のreceiptを
append-only evidence storeへ発行する。receiptは次を固定する。

- payload SHA-256
- payloadが到達したdefault-branch commit SHA
- 観測時のdefault-branch tip SHAと観測時刻
- receipt発行時刻・issuer・署名
- `not_before_session`
- `earliest_calendar_eligible_session`

Gitのauthor/committer dateはactivation時刻に使わない。receiptのrepo mirrorを
後でcommitしても、そのmirror commit SHAをreceipt本文へ自己参照させない。

`earliest_calendar_eligible_session`は、`not_before_session`とreceipt発行後に
08:58:59を迎える最初のJPX sessionの遅い方である。
`first_counted_session`は別にpipelineがcutoff前に記録し、必ず
`earliest_calendar_eligible_session`と一致させる。pre-score foldまたはpipelineが
間に合わなければactivationを無効にして新しいpayload/receiptを作る。後の日を
都合よく開始日に置き換えない。一度開始した後は全scheduled sessionへ必ず
1 decision stateを残す。過去sessionはbackfillしない。

### activation前のdry-run

- cutoff後のrecordが`LeakageError`になる
- 欠落TDnet pageがno-eventにならない
- source欠落・model失敗がcash 0としてscheduled-session分母へ残る
- 同一protocol/sessionの二重decision、sequence gap、hash-chain切断が拒否される
- 同じraw・同じfold artifactから同じdecision hashが再現される
- eventなし、source欠落、model失敗が別decision stateになる
- payload/receiptに自身を含むcommit SHA fieldが存在しない

## Phase 1 — historical OOT data readiness

対象窓は既存protocolどおり。

```text
warmup: 2025-08-01
score:  2025-08-04 ... 2026-03-31
```

最初にperformanceではなくdata readinessを判定する。

- JPX公式日報PDF 160件
- TDnet calendar-date page 243件
- source URL、byte count、SHA-256
- request開始、receipt完了、parse完了
- parser version・parser SHA-256
- registry-bound acquisition/parse manifest
- JPXとTDnetがjoint provenance-completeな120 score sessions以上
- 6 calendar months以上

raw bytesまたはmanifestが不足する場合、過去audit JSONの自己申告値だけで
statistical gateを実行しない。

historical OOTのsource欠落日は`cash_no_event`ではなく、OOT protocolどおり
source-complete評価集合から除外する。除外日数と理由は全scheduled dateに対して
報告し、performanceを見てsource completenessを変更しない。

## Phase 2 — exact no-tuning OOT

T02の次を変更しない。

```text
char TF-IDF: 2–5 gram
min_df:      3
max_features: 30000
sublinear_tf: true
norm:        l2
value target: training 1/99 percentile後に[-10,+10]%
Ridge:       alpha=20, solver=lsqr
decision:    event-only top1
fallback:    なし
training mask: price_training_eligible == true
scoring mask:  price_eligible == true
```

月ごとに、対象月より前に結果が確定したsource-complete sessionだけで
上記training maskを満たすevent bundleへvectorizerとRidgeを再fitする。
凍結runner、`UniversePolicy` config、eligibility codeのpath・SHA-256を固定する。

最初のscore sessionのcutoff前にpre-score fold manifestをsealし、次を
canonical hashで保存する。

- ordered training row identity
- ordered bundle textとtarget
- source payload
- clip lower/upper
- scoring code SHA-256
- vocabulary、IDF、Ridge係数・intercept
- dtype、shape、byte order
- Python、NumPy、pandas、SciPy、scikit-learn version

`prediction_artifact_sha256`は未来の月内予測を含むためpre-score foldへ入れない。
月末に、sealed fold hash、全scheduled session、decision-ledger head、
prediction・outcome artifactを指すcloseout manifestを別に作る。

統計gateは既存の
`research/model_v11_t02_oot_protocol.json`に従う。どれか一つでも不合格なら、
同じ窓で調整せずT02をproduction候補から棄却する。
全OOT gate合格を示すresultと独立auditのpath・SHA-256は、prospective
promotion reviewの機械可読な必須入力にする。

## Phase 3 — prospective shadow

`first_counted_session`から最低120 scheduled営業日かつ4か月を収集する。
主P&L分母は全activated scheduled sessionで固定する。

```text
selected           → 実際の始値→終値、約定想定cost
cash_no_event      → gross 0、cost 0
fail_closed_source → gross 0、cost 0
fail_closed_model  → gross 0、cost 0
```

source-complete subsetのP&Lは診断として併記するが、主指標へ置き換えない。

途中では次だけを監査し、損益を理由に仕様を変えない。

- source completeness
- candidate generation rate
- PIT compliance
- pre-score foldとmonth closeoutの再現性
- append-only decision ledgerのsession一意性・sequence・hash chain
- execution evidence coverage

n-gram、Ridge alpha、target clip、rank、fallback、閾値を変える場合は、
別protocol IDと別counterを作る。

### live snapshotとfinal archive

候補生成に使えるのは08:58:59までにreceipt・parse・computeが完了したlive
snapshotだけである。target-date pageを「日次final」とは呼ばず、登録済みの
sequence/watermarkまでcompleteとする。raw bytes、URL、request/receipt/parse時刻、
parser hash、record identityを保存する。

大引け後には別のfinal archiveを取得し、live recordがfinal archiveに一致すること、
watermark以前のrecord欠落がないことをprefix auditする。final archiveは当日の
featureへ絶対に使わず、不一致は新しいaudit recordでPIT違反として追記する。
元decisionを書き換えない。

decision payloadは`project_canonical_json_v1`でhash化する。これはPythonの
`json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(",", ":"),
allow_nan=False)`をUTF-8化し、非有限floatを拒否するproject固有仕様である。
RFC 8785準拠とは主張しない。Python versionとcontract IDをartifactへ保存し、
hash field自身は明示指定した場合だけpreimageから除外する。
`decision_id`は`protocol_sha256`、`activation_receipt_sha256`、`session_date`の
project-canonical JSON objectから一意に導出する。ledger envelopeはmonotonic
sequence、`previous_record_sha256`、`record_sha256`で連鎖する。

## Phase 4 — execution・capacity

統計gateの後に、registry-boundなsimulationまたはpaper executionで次を検証する。

- 08:58 bid/askまたはindicative price
- special quoteとdelayed open
- tick・minimum lot丸め
- simulated/actual fill
- realized spread・slippage
- turnover・predicted opening turnover
- intended capitalに対するparticipation
- filled、cancelled、unfilledの完全被覆

最低40 filled rows、30 filled sessionsを満たし、独立P&L auditと明示的人手承認が
完了するまで`orders_allowed=false`を維持する。

## 停止条件

次の場合はfail closedとする。

- source pageまたはprovenanceが欠ける
- live snapshotにregistered sequence/watermarkがない
- 月次fitがcutoff前に完了しない
- pre-score foldを同じordered inputから再現できない
- cutoff後の情報がcandidate生成へ混入する
- activation前sessionをforward counterへ算入する
- 開始後のscheduled sessionにdecision stateがない
- 同じprotocol/sessionの二重decision、sequence gap、hash-chain切断がある
- post-close prefix auditでlive sourceの欠落が判明する
- 同じ評価窓を見た後で仕様を調整する
- execution ledgerが全decisionを被覆しない

この計画はedgeの存在を仮定しない。目的は、現行freezeのまま本番採用を支持する
証拠が得られるかを反証可能な形で判定することである。
