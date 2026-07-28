# v1.3 symbolic context zero-base report

## 結論

OHLCを連続値の説明変数として回帰する従来方式をやめ、完了済みの値動きを
離散token列へ変換し、可変長context treeで次セッションの損益を逐次推定した。
登録済み4 candidate × top1/top2の8 variantはすべてqualification gateを
通過しなかった。新しいforward shadow、production candidate、注文権限は
いずれも追加しない。

点推定が最も高かった`CT02_RELATIVE_PATH__top1`でも、20 bp後は
`+0.181176%/日`だった一方、主stressの40 bp後は`-0.018824%/日`だった。
上位10日を除くと40 bp後`-0.326248%/日`、利益上位10 codeを現金化すると
`-0.313757%/日`であり、期間別にもdiscoveryとconfirmation Aが負だった。
「低コストなら正」という一点だけでは採用根拠にならない。

## これまでとの違い

今回のfeatureは、各銘柄の完了済みOHLCから次の3種類のsymbol列を作る。

- absolute path: 始値→終値、overnight、range、close locationを固定binへ変換
- relative path: 同じ4量を当日の全銘柄内tertileへ変換
- direction path: 始値→終値とovernightの方向だけを長いtoken列へ変換

target日には1～6セッション前のtokenだけを使う。モデルはRidge、logit、
tree ensemble、近傍タイトル検索ではない。各suffixのdate-equal損益平均を
保持し、行数と異なる日数の両方で親contextへ縮約する
prequential variable-order context treeである。各日の候補をscoreした後にだけ
その日の損益を更新するため、当日終値が当日判断へ入ることはない。

## 事前登録とinput erratum

最初のprotocolとrunnerは、candidate outcomeを計算する前にcommit
`71284929c462ab77f39d1f370aa7bdf4225ed612`へ固定した。

最初に指定した7列pickleは、既存の90%収録率検査で2024年7月の18 sessionを
不完備と判定し、feature/model scoring前に停止した。coverage thresholdを
緩めず、candidate resultを1件も見ないままinput-only erratum v2を登録した。
v2はv0.5ですでにhash固定済みのJPX公式月次PDF 19本とparser v6を再利用する。
feature、bin、depth、backoff、candidate、capacity、cost、gate、score期間は
一切変更していない。

公式PDFの再parse結果は次のとおり。

```text
PDF files:             19
canonical rows:        1,523,928
modeling rows:         1,524,104
codes:                 4,124
sessions:              386
no-trade rows:         57,465
partial-session rows:  34,495
parser rejected rows:  0
source-incomplete:     0
score sessions:        266
```

## 実行結果

主指標は40 bp控除後のscheduled-session平均である。

| variant | net20 | net40 | net60 | 正の月 | paired delta vs C00 | familywise L90 | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| CT01 absolute top1 | -0.106171 | -0.306171 | -0.506171 | 2/13 | -0.314073 | -0.761986 | FAIL |
| CT01 absolute top2 | -0.152299 | -0.352299 | -0.552299 | 2/13 | -0.411003 | -0.725630 | FAIL |
| **CT02 relative top1** | **+0.181176** | **-0.018824** | **-0.218824** | **5/13** | **-0.026726** | **-0.435428** | **FAIL** |
| CT02 relative top2 | -0.009507 | -0.209507 | -0.409507 | 3/13 | -0.268212 | -0.562390 | FAIL |
| CT03 direction top1 | -0.450965 | -0.650213 | -0.849461 | 0/13 | -0.658115 | -1.194361 | FAIL |
| CT03 direction top2 | -0.334574 | -0.534198 | -0.733822 | 0/13 | -0.592902 | -0.944968 | FAIL |
| CT04 consensus top1 | -0.455736 | -0.654984 | -0.854232 | 0/13 | -0.662886 | -1.032674 | FAIL |
| CT04 consensus top2 | -0.362191 | -0.561815 | -0.761439 | 0/13 | -0.620520 | -0.928573 | FAIL |

単位は1 scheduled sessionあたりのpercentage pointである。paired L90は
8 variantのBonferroni補正後、5日moving-block bootstrapによる
candidate minus同capacity C00の片側90%下限である。

対照`C00_DAILY_RANK_RIDGE`は過去artifactを参照せず再学習した。

| control | net20 | net40 | net60 |
|---|---:|---:|---:|
| C00 top1 | +0.204894 | +0.007902 | -0.189091 |
| C00 top2 | +0.255321 | +0.058705 | -0.137912 |

C00 top2 net20はv0.8の既知値`+0.25532118075031435%/日`と
小数点以下まで一致した。したがって今回の負けは、比較pipelineやcalendarが
別物になったためではない。

## 最も近かったCT02の反証

| check | observed | required | result |
|---|---:|---:|---|
| 全期間net40 | -0.018824 | > 0 | FAIL |
| discovery net40 | -0.055737 | > 0 | FAIL |
| confirmation A net40 | -0.136220 | > 0 | FAIL |
| confirmation B net40 | +0.155050 | > 0 | PASS |
| 上位10日除外net40 | -0.326248 | > 0 | FAIL |
| 利益上位10 code現金化net40 | -0.313757 | > 0 | FAIL |
| 正の月 | 5/13 | >= 9/13 | FAIL |
| familywise paired L90 | -0.435428 | >= 0 | FAIL |
| unique code | 246 | >= 50 | PASS |
| 最大code選択比率 | 1.13% | <= 10% | PASS |

銘柄集中は小さいため、失敗原因は単一銘柄への偏りではない。相対pathには
confirmation Bだけ正になる時変性があり、低コスト点推定も上位日・上位codeへ
強く依存した。これは安定した条件付き期待損益ではなく、既知期間の一部でだけ
現れた弱い関係と解釈するのが妥当である。

absolute、direction、consensusは全期間・全sliceで大幅に負だった。
過去数日の離散的な値動きの「文法」だけから当日始値→終値を順位付けする仮説は、
今回の固定表現とcontext treeでは反証された。

## 独立監査

独立auditは次を再計算してすべてPASSした。

- protocol、input erratum、runner、result、picksのhash binding
- 5モデル × 266 session × 2 slot
- labelと始値→終値return符号
- v0.8 G0 controlの完全再現
- 20/40/60 bp損益、3期間、上位10日除外、上位10 code現金化
- 8 variantのfamilywise paired bootstrap
- 全gateとproduction/order境界

```text
gate passers:             0 / 8
forward shadow candidate: none
production candidate:     none
production model changed: false
orders allowed:           false
```

## 成果物

```text
research/model_v13_symbolic_context_protocol.json
research/model_v13_symbolic_context_execution_failure_001.json
research/model_v13_symbolic_context_input_erratum_v2.json
research/model_v13_symbolic_context_runner.py
research/model_v13_symbolic_context_result.json
research/model_v13_symbolic_context_picks.csv
research/model_v13_symbolic_context_audit.py
research/model_v13_symbolic_context_audit.json
research/model_v13_symbolic_context_report.md
```

この結果を見てbin、depth、backoff、consensus比率を調整することは、
事前登録した停止規則に反する。同じ266日上でこのfamilyを微調整せず、
再挑戦する場合は新しい情報源または未観測期間を持つ別protocolとする。
