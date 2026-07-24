# Model v11 — Cross-family integration audit

## 結論

- 統合監査: **PASS**
- 本番候補: **0**
- 7系統の概念仮説: **67**
- 実行可能spec: **59**
- capacity別候補variant: **118**
- family内で経済的に異なるselection sequence: **112**
- comparator variant: **14**
- scored series総数: **132**
- family gate通過: **0/118**
- forward-shadow finalist: **0**

監査PASSは成果物の結合・PIT・独立再計算が整合したという意味であり、収益性や本番適格性のPASSではない。7系統はいずれもretrospectiveで、本番モデルを変更する権限を持たない。

## 7系統

| family | conceptual | variants | unique seq. | sessions | months | canonical decision | audit | passers | production |
|---|---:|---:|---:|---:|---:|---|:---:|---:|:---:|
| `new_data` | 11 | 6 | 6 | 107 | 6 | `independent_audit_and_manifest` | PASS | 0 | false |
| `analog` | 10 | 20 | 20 | 88 | 5 | `independent_audit` | PASS | 0 | false |
| `distributional` | 8 | 16 | 15 | 266 | 13 | `finalized_result_and_independent_audit` | PASS | 0 | false |
| `uplift` | 8 | 16 | 15 | 88 | 5 | `independent_audit` | PASS | 0 | false |
| `graph` | 10 | 20 | 20 | 266 | 13 | `independent_audit` | PASS | 0 | false |
| `shift` | 8 | 16 | 15 | 266 | 13 | `finalized_result_and_independent_audit` | PASS | 0 | false |
| `calendar` | 12 | 24 | 21 | 266 | 13 | `independent_audit` | PASS | 0 | false |

variant数は候補mechanismとcapacityの組を数え、control/comparatorを除く。new_dataは実装可能な2仮説と固定combinationのtop1/top2を6 policyとし、利用不能の9仮説も概念仮説総数には含めた。
Analog、graph、uplift、calendarのrunner resultは`pending_independent_audit`のpre-audit状態であり、最終判断は後発の独立auditをcanonicalとする。new_dataはauditとmanifest、distributional/shiftは独立auditを埋め込んだfinalized resultを使う。

118 variantはすべて経済的に別ではない。family内の実行/cash actionを日付・銘柄・weightでhashすると112 sequenceとなる。`D07_*_K1/K2`、`U06_*_K1/K2`、`S06_*_K1/K2`、calendarの`C11/C12_*_K1/K2`はそれぞれ全cashで重複する。非互換family間はdeduplicateしない。

## new_data blocked countの照合

- 凍結resultのcanonical blocked hypothesisは **9**。`ND01`, `ND02`, `ND03`, `ND06`, `ND07`, `ND08`, `ND09`, `ND10`, `ND11`。
- acquisition routeでまとめると **8** group。`ND01`と`ND02`がstructured forecast numeric feedを共通の前提にするためである。
- したがって「blocked hypothesis=8」へ書き換えない。8は取得group数、9は登録hypothesis数で、数える単位が違う。

## 横断順位付けの禁止

- **cross-family ranking permitted: false**
- 107-session、88-session、266-sessionの評価窓が混在する。
- source-completeness、候補universe、cash denominator、capacity、control、familywise手法が同一ではない。
- したがって各系統のnet40点推定値を並べた「総合1位」は作らない。共通日だけの事後的再ランキングも未登録の別仮説になるため禁止する。

## 多重性と証拠権限

- 全7系統が同じpanel SHA `6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb` を使う。
- 観測済み候補policyは118、family-local gate単位は98。
- 各familyの補正はfamily内だけであり、7系統を見た後の横断選抜を補正していない。窓が非互換なので横断p値も算出しない。
- familyごとの事前登録は機構の反証には有効だが、project-levelで既に見たpanelをfresh OOTへ戻さない。保守的な横断措置は候補を選ばないことである。

## PIT・provenanceの限界

- new_data/analogの履歴TDnet cacheにはlocal receipt timestampがなく、archive finalityは証明されていない。
- uplift/calendarのPIT PASSはpublication timestamp cutoffとsource-completeness filterの検証であり、履歴時点で同じarchive bytesを受領済みだったことの証明ではない。
- calendar featureはfrozen panel session indexからの決定論的導出。official holiday/SQ calendar datasetのprovenanceはなく、SQはproxyである。

## 未解消データblocker（exact）

- `frozen T02 OOT raw data missing`
- `08:58 execution data missing`
- `08:58 futures data missing`
- `08:58 orderbook data missing`
- `08:58 PTS data missing`
- `08:58 liquidity data missing`

凍結T02のscore開始は2025-08-04だが、利用可能なpanel/TDnet rawは2025-07-31まで。OOT protocol自身も`untouched_holdout_claim=false`で、T02 result/auditは存在しない。実始値・日次OHLCを08:58気配、先物、板、PTS、liquidityの代理にすることは禁止する。

## 本番認可rule

本番候補は、(1) 該当familyの全事前登録gate、(2) 横断選抜を含めて凍結したfresh OOT gate、(3) 08:58時点の実行可能性・spread/slippage・lot/capacity gate、(4) 独立監査、の全てがPASSした場合に限る。
現在はその積集合が空なので、本番候補は0、ordersは不許可、production modelは変更しない。

**結論は「現行freeze/dataの下で本番採用を支持する証拠がない」であり、「edgeが存在しない」ではない。** 未検証dataを取得した後も、既存結果の微調整ではなく、新しい事前登録とfresh periodが必要である。
