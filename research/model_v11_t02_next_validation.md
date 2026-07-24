# T02 次回検証計画 — forward protocol v2

## 結論

次に行う検証は、新しい特徴量・モデル・閾値の探索ではない。凍結した
`v10_t02_char_value_event_top1`を変更せず、次の順序で証拠を積み上げる。

1. forward protocol v2をhash固定してactivationする
2. 登録済みhistorical OOT入力をprovenance付きで復元する
3. exact no-tuning OOT統計gateを実行する
4. activation後のpaper-live shadowを120 source-complete営業日以上収集する
5. execution・capacity gateを独立監査する

いずれかの前段がfailした場合、後段をproduction根拠として実行しない。

## このPRの範囲

このPRは`research/model_v11_t02_forward_protocol_v2.json`を追加し、次を明文化する。

- 評価損益は**target sessionの始値→終値**である
- T02は月初ごとのmonthly expanding walk-forwardで再学習する
- 月中再学習、hyperparameter変更、欠落sourceのno-event化を禁止する
- foldごとの入力・vectorizer・係数・予測artifactをhash保存する
- activation manifestがdefault branchへcommitされる前のsessionは数えない
- production変更と実注文を引き続き禁止する

既存の`research/model_v10_forward_protocol.json`は、後続artifactからhash参照される
履歴記録なので上書きしない。旧protocolの
`target-session close-to-open return`は記述上の誤りとして残し、v2のerrataで
`target-session open-to-close return`へ訂正する。

## Phase 0 — protocol activation

### 必須成果物

`research/model_v11_t02_forward_activation.json`を別commitで作成し、少なくとも
次を固定する。

- merged protocolのSHA-256
- activation commit SHA
- activation日時
- 最初に数えるJPX session
- 学習snapshotとsource registryのpath・SHA-256
- scoring code SHA-256
- 承認者

最初のcount対象は、activation manifestがdefault branchへcommitされた時刻より
後に08:58:59 JSTを迎えるsource-complete sessionとする。過去sessionはbackfillしない。

### activation前のdry-run

- cutoff後のrecordが`LeakageError`になる
- 欠落TDnet pageがno-eventにならない
- 同一decision IDへの異なるpayload追記が拒否される
- 同じraw・同じfold artifactから同じdecision hashが再現される
- eventなし、source欠落、model失敗が別decision stateになる

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
```

月ごとに、対象月より前に結果が確定したsource-complete sessionだけで
vectorizerとRidgeを再fitする。各foldの学習row identity、vocabulary、IDF、
係数、予測結果をhash保存する。

統計gateは既存の
`research/model_v11_t02_oot_protocol.json`に従う。どれか一つでも不合格なら、
同じ窓で調整せずT02をproduction候補から棄却する。

## Phase 3 — prospective shadow

activation後、最低120 source-complete営業日かつ4か月を収集する。

途中では次だけを監査し、損益を理由に仕様を変えない。

- source completeness
- candidate generation rate
- PIT compliance
- fold artifact再現性
- append-only decision ledger整合性
- execution evidence coverage

n-gram、Ridge alpha、target clip、rank、fallback、閾値を変える場合は、
別protocol IDと別counterを作る。

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
- 月次fitがcutoff前に完了しない
- fold artifactを同じ入力から再現できない
- cutoff後の情報がcandidate生成へ混入する
- activation前sessionをforward counterへ算入する
- 同じ評価窓を見た後で仕様を調整する
- execution ledgerが全decisionを被覆しない

この計画はedgeの存在を仮定しない。目的は、現行freezeのまま本番採用を支持する
証拠が得られるかを反証可能な形で判定することである。
