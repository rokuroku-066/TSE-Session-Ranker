# v1.6 exact D-1 liquidity model evaluation

## 結論

これまで入力不足で未実行だった、JPX公式日次相場表の実出来高・実売買代金・
実VWAP・実売買単元をD-1以前だけから使う3候補を固定し、top1/top2の
6 variantを評価した。2025-12-01～2026-03-31の80 scheduled sessionsで
事前登録gateを通過したvariantは`0/6`だった。

したがって、hard activity veto、実流動性7特徴のRidge追加、VWAP × flow反転は
すべて棄却する。候補固有の2026-04-01～07-27 replayは開いておらず、
forward shadow候補、production候補ともに0である。既存packageとproduction
modelは変更せず、`orders_allowed=false`を維持する。

## 実データ境界

v1.5の「取得できる実データだけを実装し、取得できない候補は丸ごと棄却する」
契約を引き継いだ。v1.6では、以前
`blocked_local_archive_ohlc_only`だった
`ND11_pit_liquidity_and_unit_cost`について、公式JPX日次相場表に次の実測値が
あることを確認して取得経路を成立させた。

- 出来高（実株数）
- 売買代金（円）
- VWAP（円/株）
- 売買単元（株）

2025-08-01～2026-03-31の公式PDF 160本をrunへ入力し、全ファイルのSHA-256が
既存parser auditと一致した。固定parser
`jpx_daily_text_v6_special_quote_marker`で622,724行を復元し、
rejected rowは0だった。価格warm-upの月次PDF 3本も既存input lockと一致した。
欠測を価格proxyや0で補わず、20営業日連続の正の実測値と一定の売買単元を
満たさない銘柄はfail-closedにした。

公式日次PDFのうち、通常の約3,900行より著しく少なかった
2025-09-29、2025-12-29、2026-03-30は既存のsource coverage判定で
不完備になった。selection期間の既存price-panel source-coverage判定は
78/80 sessions completeであり、80日の予定分母からは除いていない。不完備日
自身のscoreはD-1情報ですでに固定され、target outcomeが欠けるslotだけcashに
なった。直前universeが不完備になる翌sessionはfail-closedで全slotをcashにした。

20/40/60bpは実spreadや実slippageではない。固定した感応度haircutであり、
実約定能力の証拠とは扱わない。

## PITと評価設計

この評価は過去のproject-level outcomeが既知であるため、
`project_level_untouched=false`のretrospective candidate-specific selection
である。結果は候補棄却または別lock後のforward shadow候補指名にだけ使え、
production昇格には使えない。

- liquidity warm-up: 2025-08-01～08-29
- initial fit: 2025-09-01～11-28
- locked selection: 2025-12-01～2026-03-31、80 sessions
- candidate-specific locked replay: 2026-04-01～07-27、79 sessions
- 学習: 月初だけのexpanding fit、学習labelは必ず前月末まで
- 主コスト: 40bp、感応度: 20/60bp
- family: 3候補 × top1/top2 = 6 variant
- 統計: paired moving-block bootstrap、block 5、20,000回
- 多重性: Bonferroni個別confidence 98.3333%

対象日のOHLC、出来高、売買代金、VWAP、売買単元、source finalityはscoreへ
入れていない。score ledgerをoutcomeなしで先に意味的hashへ固定し、その後に
始値→終値returnをjoinした。未約定・欠測・veto枠はgrossもcostも0のcashとし、
下位順位へ置換していない。

## 固定した3候補

| ID | 異なるアプローチ | 固定仕様 |
|---|---|---|
| `LQ01_ACTIVITY_VETO` | 実活動量によるhard veto | C00のtop2を先に固定し、20日中央値で売買代金1億円以上、出来高5万株以上、最低単元比率1%以下だけを残す。置換なし |
| `LQ02_EXACT_LIQUIDITY_RIDGE` | 実測特徴の教師あり増分 | 価格G0 15特徴へ、実流動性7特徴のD-1横断rankを追加。日別return-rank Ridge、`alpha=10` |
| `LQ03_VWAP_FLOW_REVERSAL` | 学習なしの機構signal | `-rank(close/VWAP-1) × (1+rank(turnover shock))/2`へLQ01と同じveto。方向反転版なし |

対照`C00_PRICE_RIDGE`は同じactivity-available universeを使うが、活動量の値を
入力しない価格G0 15特徴のRidgeである。net40はtop1
`-0.584255%/日`、top2`-0.200313%/日`だった。

## 選抜結果

単位は1 scheduled sessionあたりのpercentage pointである。実行率は固定slot
に対するobserved-outcome率で、codeを選択済みでもtarget outcomeが欠ければcash
とする。FWER LBは同capacityのC00に対するpaired差のBonferroni補正済み片側下限
である。

| variant | net40 mean | net40 median | net60 mean | 実行率 | unique code | 正の月 | FWER LB | gate |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `LQ01_ACTIVITY_VETO__top1` | -0.086286% | 0.000000% | -0.106286% | 10.00% | 8 | 0/4 | -0.079858pt | FAIL |
| `LQ01_ACTIVITY_VETO__top2` | -0.031272% | 0.000000% | -0.055022% | 11.88% | 16 | 2/4 | -0.297593pt | FAIL |
| `LQ02_EXACT_LIQUIDITY_RIDGE__top1` | -0.363429% | -0.243151% | -0.555929% | 96.25% | 44 | 2/4 | -0.212709pt | FAIL |
| `LQ02_EXACT_LIQUIDITY_RIDGE__top2` | -0.384194% | -0.403264% | -0.575444% | 95.63% | 88 | 1/4 | -0.496770pt | FAIL |
| `LQ03_VWAP_FLOW_REVERSAL__top1` | -0.040695% | 0.000000% | -0.045695% | 2.50% | 2 | 1/4 | -0.008732pt | FAIL |
| `LQ03_VWAP_FLOW_REVERSAL__top2` | -0.003979% | 0.000000% | -0.012729% | 4.38% | 7 | 1/4 | -0.271114pt | FAIL |

全variantでnet40平均、net60平均、両固定slice、上位4利益日除外後、
familywise下限が不合格だった。LQ01/LQ03はcoverage、銘柄分散、集中度にも
大きく届かなかった。LQ02は95%以上の実行率を持つため、cash化だけでは説明
できない反証であるが、両capacityとも絶対net40とnet60が負だった。

数値上もっとも0に近い`LQ03__top2`は、160 slot中7 slotしか実行していない。
その小さい損失を予測edgeとは解釈しない。情報量が比較的多い
`LQ02__top1`はC00 top1より点推定で+0.220826pt改善したが、絶対net40は
`-0.363429%/日`、FWER LBは`-0.212709pt`、前後半sliceも負であるため
採用条件を満たさない。`LQ02__top2`はC00 top2より点推定でも
`-0.183881pt/日`悪化した。

## 研究判断

今回初めて、proxyではないD-1の実出来高・実売買代金・実VWAP・実売買単元を
履歴評価へ入れられた。しかし、固定した3つの使い方はいずれも、負の価格対照を
安定して正のコスト後損益へ変換できなかった。

同じselection outcomeへveto閾値、Ridge alpha、特徴方向、capacityを合わせ直す
ことは禁止する。LQ01/LQ03の固定仕様と、LQ02の単純な全体再順位付けはここで
閉じる。実流動性は今後、alpha特徴よりも候補の実行可能性・容量診断に使う余地が
あるが、tie-breakやuncertainty weightを試す場合も新しい期間で別protocolとして
固定する必要がある。

次の有力な情報源は、D-1 activityの再加工ではなく、08:58 auction/order-book、
実spread/slippage、またはsource-completeなmaterial cohortである。実データの
取得経路とPIT証拠が成立しない系統は、v1.5契約どおり候補ごと棄却する。

## 独立監査

selection runnerとprojectの損益・bootstrap helperをimportしない別監査で、
640固定slotから次を再計算した。

```text
artifact / parser bindings:                PASS
4 models x 80 dates x 2 slots:             PASS
score-ledger semantic hash:                PASS
label / return sign and veto cash rule:    PASS
strictly-prior monthly folds:              PASS
20 / 40 / 60bp metrics:                    PASS
tail-day / profit-code cash stress:        PASS
concentration / execution:                 PASS
paired moving-block bootstrap:             PASS
13 gates x 6 variants:                     PASS
winner / rejection / no-replay decision:   PASS
production / order boundary:               PASS
checks:                                    25 / 25
discrepancies:                             0
maximum numeric difference:                0.0
```

この監査は保存済みpicksから評価・decisionを独立再計算するものであり、
raw PDFの再parseと特徴再構築は範囲外である。raw/PIT側はrunnerのsource hash
検査とmutation testで別に検証した。

## 再現artifact

```text
protocol SHA-256:
f7b1329959b237323d6aa08d87494c9523bb5bdd4e29ee8c30cd04ece9e4387e

runner SHA-256:
de5b0a6382f53ce4d513a187b4d2243149e60b8b9a7a69043e9b74d12171a537

selection result SHA-256:
7f8aff8b1e86240c00de3b124e9564ff5b3a0277f91a2cb9b27ef859207f4008

selection picks SHA-256:
65fb583501e30658273cb3d98725e2e4d142fa5681d3ed82023da597ddaf4b71

independent audit runner SHA-256:
45e2bf0a696197a1d7a0206def38d2517292ec35740c0a1df5e4c50d1fd26063

independent audit result SHA-256:
60cf81b6cf91fc1820ab5fa7d35587bea6ded91036018139aadb0882c4b6d5ed
```

```text
targeted v1.6 tests:       17 passed
full tests:               413 passed
subtests:                 192 passed
gate passers:              0 / 6
locked replay opened:      false
forward shadow candidate:  none
production candidate:      none
production model changed:  false
orders allowed:            false
```
