# v1.7 liquidity-as-reliability model evaluation

## 結論

v1.6で棄却された固定veto・流動性の全体加算・固定VWAP逆張りを微調整せず、
その評価結果から「実流動性は平均alphaではなく、価格signalの信頼性・状態・
rank1/rank2の選択に効く」という新しい機序仮説を立てた。10候補を事前登録し、
2026-04-01～07-27の79 scheduled sessionsへ一回だけ適用した。

結果は`selection_rejected_all_candidates`、gate passerは`0/10`、winnerはない。
したがってv1.7によるモデル改善は確認できず、production modelを変更せず、
`orders_allowed=false`を維持する。

候補の点推定には改善方向のものがあった。AR05はnet40
`+0.192782%/session`、VP07はfamilywise paired下限が候補中もっとも0に近い
`-0.038969pt`、PAIR01は価格だけのtop2 rerankerより`+0.484059pt`だった。
しかし10候補すべてで後半slice、中央値、上位4利益日除外、上位5利益code現金化、
Bonferroni補正済み下限、集中度が不合格だった。平均改善は通常日へ広がらず、
少数日・少数銘柄・前半期間へ集中している。

## v1.6の評価から立てた仮説

v1.6が直接反証したのは次の3機構だけだった。

- 固定閾値によるhard activity veto
- 7つの実流動性rankを価格Ridgeへ一律に加えるglobal reranking
- 高turnover時にclose/VWAP乖離を固定方向へ逆張りするsignal

保存済みv1.6 picksを診断すると、C00 rank1はgross
`-0.1993%/day`、rank2は`+0.5636%/day`だった一方、LQ02はrank1の極端な損失を
弱めてもrank2の利益を消していた。そこでv1.7は「流動性そのものをalphaとして
足す」のではなく、価格候補の相対的な信頼性、issuer状態、時間記憶、学習weight、
market regimeとして使うfamilyに限定した。

この診断はrank2固定ルールを正当化しない。v1.7ではC00 top1、価格だけの
top2 pair reranker、C00 rank2を別々に保存し、rank2 shoulderが新期間にも
続くかを同時に観測した。

## 事前登録した10候補

事前登録コミットは`78743a825d3c214f05c5f92f34206aa9a7a33731`である。
候補数はちょうど10、全候補1 fixed slot、capacity・alpha・閾値のgridはない。

| ID | 新しい機序 | 固定した使い方 |
|---|---|---|
| `PAIR01` | frozen top2内の相対信頼性 | C00 top2を先に固定し、価格score差とD-1実流動性7特徴差からbinary rerank |
| `TW02` | turnover-weighted price memory | 過去returnを実turnoverで重み付けした5日・20日記憶をG0へ追加 |
| `VW03` | aggregate cost basis | 実VWAPと価格の5日・20日basis gap/migrationをG0へ追加 |
| `RPY04` | return per turnover | 価格変化を実turnover shockで割った決定論的reversal。唯一の非教師あり候補 |
| `AR05` | issuer activity state | D-1 activity状態別に価格expertを分け、当日の状態でdispatch |
| `PS06` | persistent/isolated activity | 継続turnoverと単発turnover、およびVWAP位置とのinteractionをG0へ追加 |
| `VP07` | VWAP-range pressure memory | 日中range内のVWAP圧力を実turnover加重で5日・20日集約 |
| `RW08` | label reliability weight | 実出来lot数を学習sampleの信頼性weightとして使用 |
| `MR09` | market liquidity regime | turnover加重breadth/集中度と価格momentum interactionでmarket状態を表現 |
| `AT10` | attention migration | 短期から長期へのactivity移動とVWAP位置interactionをG0へ追加 |

対照は`C00_PRICE_RIDGE_TOP1`、PAIR01用の
`PAIR00_PRICE_ONLY_TOP2_RERANK`、診断用の`C02_C00_RANK2`である。
PAIR01以外はC00とpaired比較した。PAIR01だけは同じfrozen top2・同じpair学習集合の
PAIR00とpaired比較した。

## 入力・PIT・評価契約

公式JPX daily PDF 239本を2025-08-01～2026-07-27まで入力した。旧160本は既存lock、
新79本はoutcome parse前にname・bytes・SHA-256を固定した。
`jpx_daily_text_v6_special_quote_marker`で930,286行を復元し、rejectは0だった。

```text
daily sources:                    239
legacy / extension:               160 / 79
daily date bounds:                2025-08-01 .. 2026-07-27
parsed / rejected rows:           930,286 / 0
panel rows / codes:               1,187,484 / 4,096
selection sessions:               79
selection source-complete:        78 / 79
same-day finality used in score:  false
maximum feature source < target:  true
```

2026-06-30はfrozen C00 top2が完全でないため、登録済みcash ruleどおり全モデルを
cashにした。予定日の分母から除外せず、下位順位へ置換していない。score ledgerは
outcome列なしで意味的hashを固定してから始値→終値returnをjoinした。
48 foldsはすべてscore月より前のlabelだけを使った。

主コストは40bp、感応度は20/60bp。5-session moving-block bootstrapを20,000回、
family size 10に対するBonferroni個別片側confidence 99%で実施した。
40bp平均・中央値、60bp平均、固定前後半、正の月、上位利益日/code stress、
paired下限、銘柄分散、集中度、実行率をすべてANDで要求した。

## 選抜結果

単位は1 scheduled sessionあたりのpercentage pointである。`LB99`は登録対照との差の
Bonferroni補正済み片側99%下限、`late`は固定後半40 sessions、`tail4`は上位4利益日を
除外したnet40平均である。

| candidate | net20 | net40 | median40 | net60 | LB99 | late | tail4 | unique codes | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `PAIR01` | +0.203884% | +0.006415% | -0.316528% | -0.191053% | -0.064551pt | -0.192338% | -0.399748% | 44 | FAIL |
| `TW02` | +0.194632% | +0.002227% | -0.316458% | -0.190178% | -0.157715pt | -0.287655% | -0.352167% | 34 | FAIL |
| `VW03` | +0.338163% | +0.145758% | -0.079487% | -0.046647% | -0.149557pt | -0.185668% | -0.210817% | 34 | FAIL |
| `RPY04` | +0.327041% | +0.134636% | -0.400000% | -0.057769% | -0.648966pt | -0.286746% | -0.446152% | 39 | FAIL |
| `AR05` | +0.385187% | +0.192782% | -0.096970% | +0.000377% | -0.224118pt | -0.025370% | -0.210179% | 35 | FAIL |
| `PS06` | +0.355611% | +0.163206% | -0.079487% | -0.029199% | -0.181725pt | -0.158666% | -0.192438% | 30 | FAIL |
| `VP07` | +0.345347% | +0.152942% | +0.000000% | -0.039463% | -0.038969pt | -0.106835% | -0.195245% | 30 | FAIL |
| `RW08` | +0.000890% | -0.191515% | -0.316458% | -0.383920% | -0.611633pt | -0.178283% | -0.490320% | 32 | FAIL |
| `MR09` | +0.116340% | -0.076065% | -0.223009% | -0.268470% | -0.148198pt | -0.164293% | -0.434635% | 33 | FAIL |
| `AT10` | +0.170980% | -0.023956% | -0.316458% | -0.218893% | -0.471729pt | -0.252633% | -0.455960% | 32 | FAIL |

価格対照C00 top1はgross`+0.374650%`、net20`+0.182245%`、
net40`-0.010160%`、net60`-0.202565%`で、76/79日実行した。
C00 rank2とPAIR00はこのreplayで完全に同じ選択となり、gross`-0.087771%`、
net40`-0.477644%`だった。v1.6で見えたrank2 shoulderは新期間で再現しなかった。

## 共通failureと候補別の読み方

10/10候補が次をすべて落とした。

- 前後半の両方でnet40正
- familywise paired下限が0以上
- net40中央値が正
- 上位4利益日除外後もnet40正
- 上位5利益codeをcashにしてもnet40正
- 最大code share 5%以下、top10 code share 25%以下

一方、10/10候補が実行日数とslot率のgateを通った。棄却理由はcash sparsityではなく、
薄い平均edge、右tail依存、銘柄集中、Jun/Julの状態反転である。

- `AR05`は絶対net40の点推定が最大で、60bpでもわずかに正だった。ただしlate、
  FWER下限、中央値、tail、code分散をすべて落とした。issuer activity routingには
  弱い仮説生成evidenceがあるが、確認ではない。
- `VP07`はFWER下限がもっとも0に近いが、C00から選択が変わったのは7日だけで、
  lateとtail stressが負だった。
- `PAIR01`はPAIR00比で大きく救済したが、net40はほぼ0で、通常日・code stress・
  集中度を通らない。
- `RW08`は前後半とも負で、実lot数をlabel reliability weightにする仮説を支持しない。
- `MR09`、`AT10`、`TW02`も登録した特定表現を支持しない。これはmarket regime一般を
  否定する結果ではない。
- `RPY04`は点推定が正でも極端な利益日への依存が最大で、robustなreversalではない。

## frozen top2の事後診断

以下は新候補の採否に使わない、保存済みpicksだけの事後診断である。

PAIR00はcompleteな78日すべてでC00 rank2を選んだ。PAIR01はrank1を40日、rank2を
38日、cashを1日選び、PAIR00から40日switchした。これによりPAIR00比では
`+0.484059pt/session`改善した。

ただし登録対照ではないC00 top1との事後比較では、PAIR01の全79日改善は
`+0.0166pt/session`にすぎない。PAIR00救済の大半はexact liquidity固有の優位ではなく、
「常時rank2」を選んだprice-only pair baselineの訂正で説明できる。7月はPAIR01が
18日中17日rank2を選び、同月のnet40は`-0.464369%`だった。なぜ係数が反転したかは
保存artifactだけでは特定できない。

## 次の新仮説

v1.6からv1.7への反転がもっとも明確に示すのは、**C00 rank1–rank2のshoulder
polarityは固定的なrank2 premiumではなく、strictly lagged OOFのrank別実現差で
観測できるslow stateとして持続・反転する**という仮説である。

次のv1.8では、v1.7をdevelopment扱いし、v1.7の流動性特徴・係数・windowを
この79日に合わせ直さない。前月までのOOF rank1-minus-rank2実現差だけでslow stateを
作り、frozen C00 top2のrank positionまたはcashを選ぶ、別機構として固定する。
予測される帰結は平均差だけでなく、固定前後半、tail-day、profit-code cash、集中度の
すべてで改善が広がることである。これをfreshなsource-complete期間へ事前登録して
確認できるまでは、exact activityが有効とも、rank2 premiumがあるとも主張しない。

## 独立監査

selection runnerおよびprojectの損益・bootstrap helperをimportしない別監査が、
13 models × 79 dates = 1,027 fixed slotsから全指標とdecisionを再計算した。

```text
checks:                              29 / 29 PASS
discrepancies:                        0
maximum numeric difference:           0.0 (tolerance 1e-12)
score / picks rows:                   1,027 / 1,027
observed outcome slots:               993
all folds strictly prior:             true
gate passers:                         0 / 10
winner / research nominee:            none / none
raw PDF reparse in independent audit: false
feature reconstruction in audit:      false
```

raw PDF再parseと特徴再構築は独立監査の範囲外である。source・parser hash、
score ledgerのoutcome-free契約、PIT mutation testで別に検証した。

## 再現artifact

```text
hypothesis registry SHA-256: 6e52ae3d6362583225496c4685fc1d601d0567753facb3eb24a5430cffa10c12
protocol SHA-256:            f7d2efa30c5f6ca03a95e1f6e84e0fb6de3f68e8f0d520183877e2f0ab4a416f
replay input lock SHA-256:   1d9a391c8e6b8d2003498c18ba09dac904e672e1f17c05516998ffb71e11c475
parser source SHA-256:       1bd2e74acced608eb36c3606b593ea407d2d1e5f54b3283790ef8fd0fb1041f7
runner SHA-256:              6394161d70d8ca76862ee34bdaa3a0aeb95adf57c3dd3e6685fc65d456980807
scores / semantic SHA-256:   8e2e8d0fc4fcdbb716ec1de0cafba2b6300015b93f93020d6309987018e18244
picks SHA-256:               32de442e2012d901963399f9fd91fc69fb3082e93f0cfa7981c279b2b7467273
result SHA-256:              1463ae399c5de8ca33e762303d1ba3322ce21a383d2ad81f202a063bb8c148dd
audit runner SHA-256:        ba31df20eff03a8b4c776e2b1fe7136317651beb8c47341aec4ec7f3d64f7f8c
audit JSON SHA-256:          41a626f96dff850aec581838cd698695d541e6bb4e9882b5f1ce27fc68218dec
tests SHA-256:               6dbc0b61d8ddcfd6311d7ac104d4fad27b1af4d9298a949ae98ad137daa5981c
```

```text
targeted v1.7 tests:       16 passed
full tests:               429 passed
subtests:                 192 passed
gate passers:               0 / 10
production model changed:  false
orders allowed:             false
```
