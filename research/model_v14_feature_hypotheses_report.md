# v1.4 feature contrast hypotheses and confirmation report

## 結論

終値が始値を上回った行と、終値が始値以下だった行を、当日には未知である
終値を説明変数へ入れずに比較した。2024-07-01～2024-10-31の84 scheduled
sessions、244,704行、53特徴のdiscoveryでは21特徴が事前登録した安定条件を
満たした。しかし、その結果から別protocolで固定した4モデル × top1/top2の
8 variantは、Stage Aで未使用だった2024-11-01～2025-07-31のconfirmationで
qualification gateを1件も通過しなかった。

したがって、v1.4の結論は`0/8`、forward shadow候補なし、production候補なし、
production model変更なし、注文権限なしである。discovery上のクラス差は、
取引コストと安定性stressに耐える銘柄選択収益へは変換できなかった。

## これまでの検証から変えた点

v1.1では7機構family、59 spec、capacity別118 variantがすべてgate不合格だった。
v1.3では完了済みOHLCのsymbolic context treeを4候補 × 2 capacityで検証し、
こちらも`0/8`だった。そこでv1.4は、既存のrow-wise回帰や離散contextの
微調整ではなく、まず`close > open`と`close <= open`の特徴分布を日付等重みで
比較し、その観察から異なる数理機構を持つ仮説を独立に固定した。

Stage Aで使った特徴はすべてD-1以前の完了済み公式sessionだけを参照する。
既存G0 15特徴に加え、AM・lunch gap・PMの分解10、32 sessionの周波数特徴20、
符号遷移・entropy・相関などのpath complexity 8を導入した。当日または未来の
OHLCはscoreに使用していない。

## Stage A：`close > open`対`close <= open`

| 項目 | 実測 |
|---|---:|
| discovery期間 | 2024-07-01～2024-10-31 |
| scheduled sessions | 84 |
| 比較行 | 244,704 |
| `close > open` | 105,753（43.2167%） |
| `close <= open` | 138,951（56.7833%） |
| 比較特徴 | 53 |
| stable signal | 21 |

各特徴を日付内percentile rankへ変換し、日ごとの
`positive mean rank - nonpositive mean rank`を84日で平均した。stableの条件は
BH q値`<= 0.05`、絶対rank gap`>= 0.005`、欠損率`<= 10%`、かつ固定した
3 discovery sliceの符号がすべて全期間と一致することだった。

| 特徴群 | stable / 比較数 | 主な観察 |
|---|---:|---|
| 既存G0 | 8 / 15 | 60日OC平均・20日勝率が正、overnight平均が負 |
| session分解 | 6 / 10 | 前日rangeとAM平均が正、lunch平均・前日PMが負 |
| spectrum | 3 / 20 | lunch低周波energy、overnight高周波・dominant frequencyが正 |
| path complexity | 4 / 8 | component entropy、AM/PM相関、OC符号遷移率が負 |

絶対rank gapが大きかった特徴は次のとおりである。正は`close > open`群で
同日内rankが高く、負は低いことを表す。

| feature | daily rank gap | BH q | raw Cohen's d |
|---|---:|---:|---:|
| `oc_mean_60` | +0.022588 | 0.00000668 | +0.032375 |
| `oc_win_20` | +0.022530 | 0.0000000000400 | +0.052149 |
| `overnight_mean_20` | -0.018511 | 0.00000000000501 | -0.089495 |
| `component_sign_entropy_20` | -0.017627 | 0.000204 | -0.054928 |
| `overnight_mean_60` | -0.017074 | 0.00000000000848 | -0.043925 |
| `overnight_last` | -0.016852 | 0.0000206 | +0.003097 |
| `lunch_mean_5` | -0.016427 | 0.000000000188 | -0.055012 |
| `oc_mean_20` | +0.015184 | 0.001615 | -0.017142 |

最大rank gapでも`0.022588`、stable特徴の最大絶対Cohen's dでも`0.089495`で、
効果は小さい。また、次の4特徴は日付内rank gapと全行poolのraw mean差で符号が
逆転した。

| feature | daily rank gap | raw mean差（positive - nonpositive） |
|---|---:|---:|
| `overnight_last` | -0.016852 | +0.004455 |
| `oc_mean_20` | +0.015184 | -0.006495 |
| `xrank_atr14_pct` | +0.014009 | -0.004945 |
| `range_abs_oc_ratio_20` | +0.011465 | -0.002037 |

これは日ごとの市場水準・銘柄構成とpool全体の重みが結論を反転させ得ることを
示す。したがって仮説の方向は、事前登録した日付内rank gapだけに従った。
21信号はdiscovery上の関連であり、因果性や経済的収益性を示さない。
normal近似とBH補正も、相関した53特徴間の依存を消すものではない。

## Stage Aから固定した4仮説

| ID | 仮説 | 登録モデル |
|---|---|---|
| `DMD01` | 完了済み全銘柄OC rank場の低rank線形発展に、row-wise rolling集計にはない翌日方向情報がある | OC cross-section、60 state、exact DMD rank 8、月内operator固定 |
| `DMD02` | AM・lunch gap・PM rank場の共同回転が翌日のOC順序を予測する | 3場stack、32 state、exact DMD rank 12、月内operator固定 |
| `SP01` | stableだったlunch低周波energyとovernight高周波・dominant frequencyが非線形に相互作用する | 3 spectral rank、hidden 6のdeterministic tanh MLP、月次expanding fit |
| `GN01` | stableだったpath-complexity rankはクラス条件付き分布が異なる | entropy・range/OC比・AM/PM相関・符号遷移率のGaussian Naive Bayes、月次expanding fit |

Stage Aの結果を見た後にこの4候補、特徴、方向、lookback、rank、学習器の
parameterをStage B protocolへ固定し、そのprotocol・runner・testをremoteへ
登録してからconfirmationを1回だけ実行した。

## Stage B confirmation設計

- score期間：2024-11-01～2025-07-31、182 scheduled sessions、9か月
- 固定slice：2024-11～2025-01、2025-02～04、2025-05～07
- capacity：top1 / top2、cost：20 / 40 / 60 bp、主判定40 bp
- DMD：各月開始前の連続stateだけでoperatorをfitし、月内は固定
- MLP・GaussianNB・C00：score月より前のlabelだけで月次expanding fit
- 多重性：8 variantにBonferroni補正した片側familywise 90%、
  5日moving-block bootstrap 10,000回
- 主gate：net40・net60正、3 sliceすべてnet40正、上位10日除外後と
  利益上位10 code現金化後もnet40正、正の月6/9以上、C00対比のfamilywise
  下限非負、銘柄分散・実行率条件

同点はsecurity code昇順、未約定slotはcashとし、182日を分母から除かなかった。

## Stage B実行結果

単位は1 scheduled sessionあたりのpercentage pointである。FW L90は
同capacityの`C00_DAILY_RANK_RIDGE`に対するpaired差のBonferroni補正済み
片側familywise 90%下限である。

| variant | net20 | net40 | net60 | 正の月 | FW L90 vs C00 | gate |
|---|---:|---:|---:|---:|---:|:---:|
| `DMD01__top1` | -0.075290% | -0.275290% | -0.475290% | 2/9 | -0.521562pt | FAIL |
| `DMD01__top2` | -0.063783% | -0.263783% | -0.463783% | 1/9 | -0.630936pt | FAIL |
| `DMD02__top1` | -0.371816% | -0.571816% | -0.771816% | 0/9 | -0.940859pt | FAIL |
| `DMD02__top2` | -0.350941% | -0.550941% | -0.750941% | 0/9 | -0.928263pt | FAIL |
| `SP01__top1` | -0.113622% | -0.311424% | -0.509226% | 0/9 | -0.594375pt | FAIL |
| `SP01__top2` | -0.185446% | -0.383797% | -0.582149% | 0/9 | -0.705908pt | FAIL |
| `GN01__top1` | -0.133762% | -0.333762% | -0.533762% | 3/9 | -0.692147pt | FAIL |
| `GN01__top2` | -0.226527% | -0.426527% | -0.626527% | 1/9 | -0.746468pt | FAIL |

8 variantすべてでnet40とnet60が負、3 fixed sliceのnet40もすべて負だった。
正の月は最大でも`3/9`、上位10日除外後net40、利益上位10 code現金化後net40、
FW L90も全件で負だった。さらに`SP01__top1`はunique code 93、
top10 code比率31.32%で分散条件を外し、`GN01__top1`はunique code 84、
最大code比率7.14%、top10比率33.52%でも不合格だった。

同じ期間・capacityの対照C00のnet40はtop1が`-0.124631%/日`、
top2が`+0.028955%/日`だった。全候補のnet40点推定は対応するC00を下回り、
低コスト側にも採用可能な候補はなかった。

## 入力erratumと公式PDF replay

Stage A protocolの`parser_version`表記
`jpx-stock-prices-v6`は、実装済みparser名
`jpx_daily_text_v6_special_quote_marker`へappend-only erratumで訂正した。
また、実行cacheのSHAは生成resultで初めて記録されたため、hash固定済みの
JPX公式月次株価表PDF 19本から独立に再parse・再実行した。

公式PDF replayは244,704行、84 sessions、53特徴、21 stable signalを再現し、
cache実行との`contrast` object exact matchは`PASS`だった。erratumは
日付、特徴、比較法、stable条件、discovery所見、仮説を変更していない。
全入力panelはsource integrity確認と共通upstream構築のためloadされるが、
Stage Aのfeature engineとcontrastは2024-10-31で物理的に切られ、
confirmation行の利用は0だった。

## 独立監査

confirmation runnerをimportしない別監査で、出力CSVから全指標を再計算した。

```text
182 sessions × 5 models × 2 slots:       PASS
label / return sign:                      PASS
v1.3 C00 confirmation 364 rows exact:     PASS
cost / slice / tail-code removal:         PASS
code concentration / execution:           PASS
familywise moving-block bootstrap:         PASS
12 gate checks × 8 variants:              PASS
decision / production / order boundary:   PASS
mismatches:                               0
```

```text
confirmation result SHA-256:
de6b5e27e07def3524361f129a89d2e58da916c19c6c8e7f49219eb27305cba2

confirmation picks SHA-256:
97aaeeb09d5590c8250a5c8989197744aa4e7c4844eeda6eef369b8dcdaf90ee

independent audit SHA-256:
ed302c5369d00ec5c0464fc69a90f4ab6532bf0b134c40c1c6eb99ec54778f27
```

## 最終判断

```text
gate passers:             0 / 8
forward shadow candidate: none
production candidate:     none
production model changed: false
orders allowed:           false
```

Stage Aの21信号は、3 discovery sliceで同符号のクラス差を持った。一方、DMDによる
cross-stock場の発展、spectral MLP、complexityの生成分類という互いに異なる
4仮説は、固定confirmationで一貫した純収益を作れず反証された。同じ期間で
feature方向やmodel parameterを再調整せず、次の検証には新しい未観測期間または
新しい情報源を必要とする。
