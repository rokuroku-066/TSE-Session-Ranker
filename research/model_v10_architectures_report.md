# v1.0 ゼロベース・モデル構造検証レポート

## 結論

新たに事前登録した13個のモデル構造は、いずれも対照の
`C00_daily_rank_ridge`を上回らなかった。事前登録した堅牢性条件を
すべて満たす候補は0件であり、採用候補はない。

今回の結果は、L4/L6、rank2単独、open-to-close breadth切替を使わずに
得られた。全候補は毎日スコア上位2銘柄を50%ずつ保有する同一ポート
フォリオで評価した。

現時点では、損失関数やモデル族をさらに同じパネルへ合わせ込むより、
08:58時点の板・予想始値・先物・PTS・出来高・売買代金・スプレッド・
単元金額など、新しいpoint-in-time情報を得ることが優先される。

## 事前登録

- protocol:
  `/tmp/v10_model_architectures/protocol.json`
- protocol SHA-256:
  `bb5224836397070a1945d0226caf1af3f9f2057b9da1265a97a5b1421c82aa7a`
- 登録時刻:
  `2026-07-23T14:20:44+09:00`
- 登録状態:
  結果実行前、ただし全履歴結果が既にプロジェクトから見えている
  retrospective research
- production promotion:
  不可

事前登録した13仮説は次のとおり。

| ID | 新しい考え方 |
|---|---|
| M01 | topを指数的に重くするlistwise gain Ridge |
| M02 | 上下の極端順位を3乗で強調するlistwise Ridge |
| M03 | コスト中心のbounded utilityを直接回帰 |
| M04 | 上昇tail確率－1.5×下落tail確率 |
| M05 | コスト超過確率×条件付き損益のhurdle model |
| M06 | 損益差で重み付けしたpairwise logistic rank |
| M07 | 上位・下位decileだけのpairwise logistic rank |
| M08 | 35% lower expectileの反復重み付き回帰 |
| M09 | 時系列3分割expertのmedian score |
| M10 | 前日市場dispersionによる2-expert mixture |
| M11 | 平均予測－0.5×予測絶対誤差 |
| M12 | 非線形35% conditional quantile |
| M13 | conditional median－downside spread penalty |

対照はG0の15特徴だけを使う、日次横断面percentile targetの
`Ridge(alpha=1)`である。

## 入力と評価条件

### frozen input

- panel:
  `/tmp/model_v07_corrected_panel.pkl`
- panel SHA-256:
  `6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb`
- 1,524,104行
- 4,124銘柄
- 386営業日
- scoring:
  2024-07-01から2025-07-31、266営業日

### walk-forward

- 13 calendar-month folds
- 各月の学習は当該月より前の日付だけ
- `price_training_eligible == true`かつ損益観測済みの行だけで学習
- `price_eligible == true`だけを当月にscore
- target-day outcomeを99%と-99%へ交互に変更しても、
  14仕様すべてscore差0、top2一致

### 損益

- 毎日2slot、50%ずつ
- 未約定slotは現金で、手数料なし
- round-trip cost:
  20bp、40bp、60bp
- tail:
  各候補の最良5、10、20日を除外
- stability:
  13か月、discovery、confirmation A、confirmation B
- multiplicity:
  13候補を同一家族として、10日circular moving-block bootstrap
  2,000回によるsingle-step familywise max-t下限

## 結果

単位は1営業日あたり%。`tail20`は最良20日を除いたnet20。

| ID | net20 | net40 | net60 | tail20 | net20プラス月 |
|---|---:|---:|---:|---:|---:|
| **C00 control** | **+0.2553** | **+0.0587** | **-0.1379** | **-0.0054** | **12/13** |
| M01 | +0.0058 | -0.1893 | -0.3844 | -0.3737 | 6/13 |
| M02 | +0.1305 | -0.0658 | -0.2620 | -0.1398 | 9/13 |
| M03 | -0.5560 | -0.7552 | -0.9545 | -0.8722 | 0/13 |
| M04 | -0.2706 | -0.4676 | -0.6646 | -0.5345 | 4/13 |
| M05 | -0.5482 | -0.7478 | -0.9474 | -0.9120 | 1/13 |
| M06 | -0.2431 | -0.4401 | -0.6371 | -0.5508 | 3/13 |
| M07 | +0.0348 | -0.1592 | -0.3531 | -0.1647 | 6/13 |
| M08 | -0.2979 | -0.4960 | -0.6942 | -0.5828 | 2/13 |
| M09 | +0.1073 | -0.0885 | -0.2844 | -0.1446 | 10/13 |
| M10 | -0.1019 | -0.2993 | -0.4967 | -0.3532 | 4/13 |
| M11 | -0.2958 | -0.4921 | -0.6883 | -0.5022 | 2/13 |
| M12 | -0.1439 | -0.3428 | -0.5416 | -0.2908 | 2/13 |
| M13 | -0.2189 | -0.4178 | -0.6166 | -0.3095 | 0/13 |

最高の新規候補はM02だったが、controlに対するnet40差は
`-0.1245ポイント`。familywise max-t 80%下限は
`-0.2524ポイント`だった。M09の差は`-0.1472ポイント`、
同80%下限は`-0.2786ポイント`である。

事前登録条件の通過数は0/13。

## 解釈

### 1. magnitudeを直接学習すると悪化した

M03、M05、M08、M11はすべて大幅なマイナスだった。raw returnの
大きさや条件付きpayoffは、現在の15特徴では安定して予測できない。
同じデータでutility係数や閾値を動かす根拠はない。

### 2. ordinal targetだけが比較的残った

新規候補で比較的ましだったM02とM09は、どちらも損益の絶対値では
なく日次順位を中心にした仕様だった。

- M02とcontrolの共通選択:
  1日平均1.56/2銘柄
- M09とcontrolの共通選択:
  1日平均1.45/2銘柄
- daily net20 correlation:
  M02-control 0.785、M09-control 0.650

それでもcontrolを改善していない。日次順位targetの中心的な情報を
保持した候補だけが近い成績になり、変形や時間分割がsignalを薄めた
と解釈するのが自然である。

### 3. nonlinear quantileは別銘柄を選んだがedgeがなかった

M12とM13のcontrolとの共通選択は、それぞれ1日平均0.011銘柄、
0.041銘柄にすぎない。別の領域を発見したのではなく、flatまたは
取引しにくい銘柄を多く上位化し、コスト負けした可能性が高い。
売買代金、spread、出来高、単元金額がない状態でのlower quantile
最適化は続けない。

### 4. controlも実運用に十分なほど堅牢ではない

controlはnet20が+0.2553%だが、最良20日の寄与は1日平均換算
+0.2603ポイントであり、それらを除くと-0.0054%になる。

- net40 median:
  -0.0466%
- net60 mean:
  -0.1379%
- 補足的な10日block bootstrapのnet40下限:
  80% -0.0061%、90% -0.0372%
- 銘柄7043の選択:
  60slot
- 最大1銘柄のnet20 slot利益寄与:
  26.9%

したがってcontrolは「今回の比較では最良」であって、
production昇格候補ではない。

## 再現・独立監査

### runner

- `/tmp/v10_model_architectures/run.py`
- SHA-256:
  `7f51bdd67ba4caafeb2d228676249598c34d7ca5afbab0e55397ac8b1db35c78`
- 初回runtime:
  280.4秒

### 初回成果物

- result:
  `/tmp/v10_model_architectures/result.json`
- result SHA-256:
  `c1ef294cac7457734318e1b106ee9d71e5aadabbefa8b6598274a7a3b840406c`
- picks:
  `/tmp/v10_model_architectures/picks.csv`
- picks SHA-256:
  `f5d2de4ac0daa6b8a6a7e665235ce1f554302bb7f6399395e7b1993983b45ef2`

### 完全再実行

- picks SHA-256は初回と完全一致
- metrics、multiplicity、gates、decision、mutation auditはJSON値で完全一致
- fold情報もruntimeを除いて完全一致

### 独立audit

- `/tmp/v10_model_architectures/audit.py`
- audit script SHA-256:
  `afa57435a1b6dc890f97aeb3d9d0920a5b8021ef64e1555e4e7fa7014b07daf5`
- `/tmp/v10_model_architectures/audit.json`
- audit SHA-256:
  `d4ea3c93ae8c19edb2147af5934985ec6654ddd50fb508f7747edf465d2739dc`
- 独立損益再計算の最大差:
  `1.11e-16`

## provenance上の注意

panel本体のSHA-256はv0.8で使用した値と一致する。一方、現在の
`/tmp/model_v07_corrected_panel.pkl.manifest.json`のSHA-256は
`25e08c...`で、committed v0.8 protocolに記録された
`ab896f...`とは一致しない。

今回のprotocolは実行前に現在のmanifestを固定し、panel rows、
columns、calendar、panel file hashを検証したため、今回の結果内部の
再現性は確保されている。ただし旧v0.8 reproduction runnerを現在の
manifestでそのまま実行するとlock違反で停止する。この差の由来は
別途provenance修復が必要である。

## 次の判断

今回の13モデルからは新仕様を採用しない。同じfrozen panelで、
utility係数、quantile、pair weighting、expert境界を微調整することも
停止する。

次の検証単位はモデルではなく、次の新しい情報または意思決定設計に
置く。

1. 08:58の予想始値、板、spread、特別気配
2. 先物・指数・sectorの寄り前変化
3. PTS価格と出来高
4. TDnet本文から抽出した定量surprise
5. 過去売買代金、出来高、単元金額と実行可能cost
6. 既存price-only controlのtail銘柄を事前に識別できる特徴
7. 2026-07-23以降の未使用forward outcomes

この段階での最も強い結論は、「複雑なモデルが不足している」のでは
なく、「現在のpoint-in-time特徴だけでは、costとtailを超える
conditional payoffを学習できていない」である。
