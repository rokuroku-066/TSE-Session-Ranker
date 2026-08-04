# 検証台帳 — tse-session-ranker

> [!NOTE]
> このファイルは追記専用の検証台帳です。記録は古い順で、**末尾のEntryが最新判断**です。まず各Entry先頭の要約表を読み、必要な場合だけ詳細を展開してください。

## 運用ルール（2026-07-21制定）

- 既存エントリの本文・数値・判断を後から上書きまたは削除しない。
- 新しい検証は、使用バージョン、検証日、検証対象コミット、直前Entryとの関係を明記して末尾へ追加する。
- 過去記録に誤りが見つかった場合も元記録は残し、訂正エントリで対象箇所、理由、影響を明記する。
- 最新エントリの判断が現在の運用判断となる。過去エントリは当時の判断として読む。
- 詳細な機械可読結果が別ファイルにある場合も、主要条件・指標・結論・制約はこの台帳内に残す。

<details>
<summary><strong>追記用テンプレート</strong></summary>

```markdown
---

## Entry NNN — vX.Y.Z：検証テーマ

| 項目 | 内容 |
|---|---|
| 検証日 | YYYY-MM-DD |
| 検証対象コミット | `SHA` |
| 親Entry | Entry NNN |
| 検証対象 | 比較したモデル・特徴・ルール |
| 採用判断 | 採用仕様または変更なし |
| 当時の運用判断 | shadow / production / 棄却 |
| 主指標 | コスト条件を含む主要結果 |
| 機械可読成果物 | パスとSHA-256 |

> **一行結論:** 結果と次の判断を簡潔に記載する。

<details>
<summary><strong>詳細な検証条件・結果・再現手順</strong></summary>

本文

</details>





```

</details>

---


## Entry 001 — tse-session-ranker 0.2.0：モデル手法比較

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-21 |
| 検証対象コミット | `9bd0588076cbd358a15eab03db748a458df8be90` |
| 親Entry | — |
| 検証対象 | 同一walk-forward条件で6モデル手法を比較 |
| 採用判断 | 正則化ロジスティック回帰を固定 |
| 当時の運用判断 | shadow。60～100新規営業日の前向き検証が必要 |
| 主指標 | 選択期top1・往復20bp後 `+0.2278%/日`。3か月中プラス1か月 |
| 機械可読成果物 | `research/profit_model_comparison.json`<br>SHA-256 `97dee1498e818f2774313417261e426d05e13b9722a2386b0c74df5a5e7e90fc` |
| 検証時script | `research/compare_profit_models.py`<br>SHA-256 `a85ba486daa39c619ead3f67a7390de5bb09b9f4d98908d0f76776f8b38adaca` |

> **一行結論:** 6手法では正則化logitが首位だったが期間安定性はなく、モデルだけを固定して特徴量比較へ進んだ。

<details>
<summary><strong>詳細な検証条件・結果・再現手順</strong></summary>

検証日: 2026-07-21

### 結論

主目的を勝率ではなく、次のscheduled-day損益率に固定した。

```text
top1の日次損益 = 始値→終値騰落率 - 約定時のみ往復20bp
主指標 = 評価対象日すべての平均
```

canonical JPXデータと同一の月次walk-forward条件で6手法を比較した結果、選択期間で最も高かったのは現行の正則化ロジスティック回帰だった。したがってproduction estimatorは維持し、採否基準・データ意味論・損益集計・成果物検証をprofit-firstへ更新した。

これは優位性の証明ではない。選択期間は57日しかなく、3か月中プラスは1か月、上位5日除外平均もマイナスである。PR固定後の新規60～100営業日を真の前向き評価にする。

### 再現コマンド

```bash
PYTHONPATH=src python research/compare_profit_models.py \
  --daily /tmp/package_collected.pkl \
  --output research/profit_model_comparison.json

PYTHONPATH=src python -m unittest discover -v
```

モデル仕様、seed、fold、全指標、データhashは`research/profit_model_comparison.json`、日次選定結果は同名の3つのpick CSVに保存した。

### データ

現行JPX parserで2024年1月～2025年7月の元TXTを再収集した。

```text
期間:                  2024-01-04 ～ 2025-07-31
行数:                  1,510,910
銘柄:                  4,124
観測セッション:        386
片場4価格:             34,259
無約定:                54,982
入力ファイルSHA-256:   0dfe401559d34bb4b468e307c2a08146186f7f8e9b391add7cf59760b5208eff
canonical内容SHA-256:  0e1388037ea75ff8ee503bb4bb70493e418dfb16504eba29452bfecd47e815ba
```

内容hashは`date, code, open, high, low, close, traded`を`pandas.util.hash_pandas_object(index=False)`で行hash化し、そのbyte列へSHA-256を適用した値である。

比較時はcanonical入力に観測された386日を明示calendarとして固定した。calendar SHA-256は`966f4a416d0929487d850b16be87c9c1cd69cd9f8c0447a3e76214a5715c489e`。これは独立したJPX営業日マスターではないため、上流で1日分が完全欠落していないことまでは証明しない。実運用では公式営業日calendarを`--calendar`で必ず渡す。

### 評価設計

- 探索診断: 2024-09-02～10-31
- モデル選択: 2025-01-06～03-31
- 既知ベンチマーク: 2025-04-01～07-31
- 各月は月初より前の行だけで再学習
- 対象日前営業日の銘柄集合を、当日結果を見る前に固定
- 当日欠落は順位付け後に未約定・損益0・コスト0
- top2は各50%固定で、未約定枠を事後再配分しない
- モデル間で候補集合、fold、コスト、損益関数を共通化
- 各手法は1つの事前固定仕様だけを比較し、選択期間でgrid探索しない
- 2025年4～7月は過去に閲覧済みなので、選択期間の勝者だけを照合

### 比較した手法

| 固定候補 | 選択期top1 20bp後平均 | 選択期top2 20bp後平均 | top1最大DD |
|---|---:|---:|---:|
| **正則化logit** | **+0.2278%** | -0.0200% | -12.58% |
| Huber-SGD回帰 | -0.0075% | -0.2090% | -8.32% |
| 多閾値期待損益 | -0.2139% | -0.3806% | -20.35% |
| 値幅加重logit | -0.2254% | -0.2787% | -21.12% |
| Ridge損益回帰 | -0.3310% | -0.2951% | -23.58% |
| 日次pairwise ranking | -0.5761% | -0.2740% | -29.14% |

選択規則どおり、top1平均が最大の正則化logitを採用した。これは個別勝率を最大化した判断ではない。学習ラベルは`close > open`だが、モデル自体の採否は実現損益で決めている。出力確率は未校正であり、同日内の順位スコアとしてだけ使う。

採用モデルの固定仕様:

```text
特徴: session_v2の12特徴
前処理: median補完 + 欠測indicator + 標準化
推定器: LogisticRegression
C: 0.08
class_weight: balanced
日付重み: 各日の学習weight合計を同じにする
random_state: 31
```

選択期top1の詳細:

```text
57日 / 57約定
約定時勝率:              57.89%
gross平均:               +0.4278%
20bp後平均:              +0.2278%
20bp後中央値:            +0.2580%
最大DD:                  -12.58%
3か月中プラス:           1か月
上位5日除外平均:         -0.0784%
```

月別は1月`-0.1462%`、2月`+1.0368%`、3月`-0.1451%`。探索診断期には同じlogitが`-0.4755%`だったため、期間安定性は未確認である。

### 既知ベンチマーク

選択後、勝者だけを2025年4～7月へ適用した。完全未使用holdoutではない。

```text
top1:
  84日 / 84約定
  約定時勝率             65.48%
  gross平均              +0.4958%
  20bp後平均             +0.2958%
  20bp後中央値           +0.3564%
  最大DD                 -10.84%
  4か月中プラス          3か月
  上位5日除外平均        +0.0362%

top2等金額:
  168 signals / 167約定
  20bp後平均             +0.2904%
  最大DD                 -6.51%
```

top1月別20bp後平均は4月`-0.2203%`、5月`+0.1365%`、6月`+0.6176%`、7月`+0.6260%`。コスト感応度は0bp`+0.4958%`、40bp`+0.0958%`、60bp`-0.1042%`だった。

### source coverage修正

監査で旧評価に次の問題を発見したため、旧`+0.2216%`などの値は採用判断から撤回した。

- 不完全日を削除して日付を詰め、翌日を古い銘柄集合へ再接続していた
- rolling coverageが長期欠落へ自己適応していた
- 欠落終値を古い終値で補い、翌日の夜間ギャップを計算していた
- 全行欠落日は営業日calendarなしでは検出できなかった
- 最大DDが初期NAVを含まず、初日下落を見落とした

新しい`prior_session_universe_source_mask_v2`では日付を保持し、不完全日とその直後を学習・評価から除外する。coverage基準は過去に合格したセッションだけから作るため、欠落が自分で基準を下げない。未知の直前終値を必要とするovernightは欠測にし、全行欠落日は明示calendarからplaceholderを生成する。calendar prefix hashはartifactへ保存して推論時に照合する。

### 自動テスト

```text
Python 3.12.13
numpy 2.3.5
pandas 2.2.3
scikit-learn 1.8.0
joblib 1.5.3
36 tests passed
```

主要な境界テストには、特徴量の時点制約、学習・推論parity、将来データ不変性、部分欠落・継続欠落・全行欠落、未約定の固定現金配分、初期NAV込みDD、cost検証、artifact checksum/payload/calendar整合性、旧artifact拒否、板snapshot cutoffを含む。

### 昇格条件

このPRの仕様を固定し、新規60～100営業日をshadow記録する。少なくとも次を満たすまでは自動発注の優位性が確立したとは扱わない。

- 40約定以上、30独立取引日以上
- 実測コスト控除後の平均と中央値がプラス
- 4期間中3期間以上がプラス
- 上位5日除外平均がプラス
- 最大1日の利益寄与が総プラス損益の25%未満
- calendarとsource収録率が検証済み

artifact schemaは4、config schemaは2である。0.1.xおよび修正前0.2.0のartifactは再学習が必要。

</details>

---

## Entry 002 — tse-session-ranker 0.3.0：TDnet特徴量比較

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-21 |
| 検証対象コミット | `574fa4c8e9e208727f32628867c23dcd04ef66ff` |
| 親Entry | Entry 001 |
| 検証対象 | 正則化logitを固定し、12個の特徴ブロックを比較 |
| 採用判断 | 価格12特徴＋TDnet明確材料6フラグの`session_v3_tdnet_clear` |
| 当時の運用判断 | **shadow only**。自動発注へ昇格しない |
| 主指標 | top1・往復20bp後 `-0.075784%/日` |
| 価格12特徴との差 | `+0.062514pt/日`、paired 90% CI `[-0.134592, +0.271630]` |
| 機械可読成果物 | `research/feature_set_comparison.json`<br>SHA-256 `6074ad08a22e71c7ec274f3a4e42bd650138cb7d42c3d32ae529f6e640cc0868` |
| 検証時script | `research/compare_feature_sets.py`<br>SHA-256 `9aa766ef6593197a0fb947ebb44bd6fb6787679769ab7512c332928a10688564` |

> **一行結論:** TDnet 6フラグは相対首位だが、絶対損益は負で信頼区間もゼロを跨ぐため、実装は固定してforward shadow検証へ進む。

<details>
<summary><strong>詳細な検証条件・結果・再現手順</strong></summary>

検証日: 2026-07-21

> [!WARNING]
> **0.3.0はshadow運用専用であり、自動発注へ使用しない。** 開発期間では価格特徴だけのモデルよりTDnet特徴を加えたモデルの損益が改善したが、往復20bp控除後のtop1平均は依然として`-0.0758%/日`である。比較差の90% bootstrap区間もゼロを跨ぎ、未使用holdoutは残っていない。したがって、現時点では正の期待損益もTDnet特徴の優位性も確立していない。

### 結論

推定器を正則化ロジスティック回帰へ固定し、12個の事前定義した特徴ブロックを月次walk-forwardで比較した。開発3期間、合計142日のtop1コスト後平均が最も高かったのは、価格12特徴にTDnetの明確な材料6フラグを加えた`baseline_tdnet_clear_flags`だった。この18特徴をproduction仕様`session_v3_tdnet_clear`として採用する。

```text
目的指標 = 全評価対象日のtop1平均損益率
日次損益 = 始値→終値騰落率 - 約定時のみ往復20bp

価格12特徴のみ:       -0.138298%/日
価格12 + TDnet 6特徴: -0.075784%/日
差:                    +0.062514ポイント/日
差の90% bootstrap CI: [-0.134592, +0.271630]
```

改善幅は小さく、統計的にも不確実で、絶対損益は負である。候補は毎日、適格ユニバースから1位を`CORE`、2位を`RESERVE`として表示できるが、これは発注推奨を意味しない。

### 固定モデル

0.2.0の比較で選んだ正則化logitを、この特徴量比較では変更していない。モデル種類やハイパーパラメータを特徴セットごとに調整せず、損益差を特徴設計の差として比較した。

```text
推定器:         LogisticRegression
C:              0.08
class_weight:   balanced
random_state:   31
前処理:         median補完 + 欠測indicator + 標準化
日付重み:       各日の実行可能な学習行のweight合計を同じにする
学習ラベル:     close > open
採否指標:       top1のコスト控除後平均損益率
```

出力確率は未校正であり、「上昇確率XX%」としては使わない。同日内の候補を並べる順位スコアとしてだけ使用する。

### 比較した特徴設計

個別列を結果に合わせて逐次追加する方式は使わず、経済的な意味ごとに次の12ブロックを事前定義した。

| ID | 特徴数 | 内容 |
|---|---:|---|
| `baseline_12` | 12 | 価格ベースライン |
| `baseline_price_shape` | 16 | ベースライン + 値幅・終値位置・ギャップ埋め |
| `baseline_momentum` | 15 | ベースライン + ATR正規化モメンタム |
| `baseline_buyback` | 13 | ベースライン + 自己株取得決定 |
| `baseline_dividend_sign` | 14 | ベースライン + 増配・減配 |
| `baseline_buyback_dividend` | 15 | ベースライン + 自己株取得・配当方向 |
| **`baseline_tdnet_clear_flags`** | **18** | **ベースライン + 明確なTDnet材料6フラグ** |
| `baseline_tdnet_clear_scores` | 16 | ベースライン + TDnet材料の合成スコア |
| `baseline_tdnet_timing` | 18 | ベースライン + 開示件数・時間帯・経過時間 |
| `baseline_tdnet_timing_clear` | 24 | TDnet時間特徴 + 明確な材料6フラグ |
| `baseline_price_shape_tdnet_clear` | 22 | 値動き形状 + 明確な材料6フラグ |
| `baseline_momentum_tdnet_clear` | 21 | モメンタム + 明確な材料6フラグ |

勝者が使う価格12特徴は次のとおり。

```text
oc_last
oc_mean_5, oc_mean_20, oc_mean_60
oc_win_5, oc_win_20, oc_win_60
oc_std_20
overnight_last
overnight_mean_20, overnight_mean_60
night_day_corr_60
```

いずれも対象日より1セッション以上前へshiftする。対象日の始値・高値・安値・終値は特徴へ入らない。

追加したTDnet 6特徴は次の0/1フラグである。

```text
tdnet_has_revision_up
tdnet_has_revision_down
tdnet_has_dividend_up
tdnet_has_dividend_down
tdnet_has_buyback_decision
tdnet_has_equity_financing
```

分類は開示タイトルだけを使う。方向語は親カテゴリと同時に一致した場合だけ有効にするため、「配当予想の上方修正」を業績上方修正として数えない。ToSTNeT、取得状況、取得終了は新規自己株取得決定から、払込完了、行使状況、発行結果は新規エクイティ調達から除外する。

これはPDF本文の業績修正額、配当利回り、買付方法や取得規模をまだ使っていない保守的な実装である。`tdnet_has_buyback_decision=1`でも継続的な市場買付を保証しない。

### データ

#### 日足と営業日calendar

比較に使った日足は2025-03-31以前へ切り詰め、既知ベンチマーク期間を読み込まないようにした。

```text
日足期間:               2024-01-04 ～ 2025-03-31
行数:                   1,178,727
銘柄:                   4,096
明示calendarセッション: 302
日足ファイルSHA-256:    0dfe401559d34bb4b468e307c2a08146186f7f8e9b391add7cf59760b5208eff
日足内容SHA-256:        cee70d867eaa0a7e13484bd7858c35329c86c7fe5e40953b2d1e55a558456f18
calendar内容SHA-256:    c4e07784ebc9e6340c0ca95882ce3de9f8b2846ac47d2bfe53da56008d78a61a
```

全銘柄が欠落した日を単なる休場日と誤認しないため、日足から推測した日付列ではなく明示的な取引所営業日calendarを渡した。収録率不足日は日付を詰めず、その日と直後の依存行を学習・評価対象から除外する。

#### TDnet

```text
日別HTML:               453ファイル
sidecar metadata:       453ファイル
開示:                   95,915件
開示銘柄コード:         4,645
公開時刻範囲:           2024-01-04 08:00 ～ 2025-03-31 19:00 JST
TDnet dataset SHA-256:  1ee87ae068549ec61484c77382dff919080925148fd3a9c20faa89ad8a611aec
```

現行downloaderは第三者の公開TDnet日別ミラーを使う。ミラーの掲載時刻を`published_at`としており、公式配信からの掲載遅延や後日の訂正を排除できない。

422日はHTTPリクエスト開始時刻の意味で観測時刻を保存した旧schema v1 sidecarだが、明示的なprovenanceタグがない。2025-03-01～03-31の31日は、過去に保存したHTMLのファイルmtimeが掲載日より十分後であることを根拠にした`legacy_historical_file_mtime_assumption`で補った。したがって453日すべてを**今回の履歴研究専用cache**として扱う。productionは`network_request_start_recorded`が明示されたsidecarだけを受け入れ、タグなしsidecarとlegacy mtimeをコード上で拒否する。

TDnet export manifest schema 2は、開示、完全日、日別観測時刻、provenance、元source hashをcanonical dataset digestへ結合する。export本体だけでなくmanifestの観測時刻を単独改変した場合も読み込みを拒否する。

週末・祝日にも開示があるため、対象範囲の全カレンダー日を要求する。各開示は、`08:58:59 JST`のcutoffがミラー掲載時刻以後となる最初の取引日へ割り当てる。

### 評価設計

| 区分 | 評価期間 | 日数 | 用途 |
|---|---|---:|---|
| early development | 2024-06-03～08-30 | 44 | 開発比較 |
| exploration | 2024-09-02～10-31 | 41 | 開発比較 |
| selection | 2025-01-06～03-31 | 57 | 開発比較 |
| combined development | 上記3期間 | 142 | 最終選択 |
| known benchmark | 2025-04-01～07-31 | — | 今回はアクセスせず |

選択規則はcombined developmentのtop1コスト後平均が最大、同点ならtop2、次に特徴数が少ないもの、最後にfeature-set IDの順である。各月は月初より前の行だけで再学習した。

2025-04～07月は過去の作業で結果を閲覧済みのため、真のholdoutではない。今回の特徴選択ラウンドではアクセス予算を0とし、結果ファイルにも`not accessed by this feature-selection round`と記録した。つまり、選択後に成績を確認できる未使用期間は残っていない。

候補集合、fold、モデル、コストと損益関数は全特徴セットで共通にした。対象日前営業日の銘柄集合を先に固定し、当日データが欠けていた候補は順位を組み替えず、未約定・損益0・コスト0として扱う。top2は各50%固定で、未約定枠を約定銘柄へ事後再配分しない。

### 結果

#### 全開発期間

| 特徴セット | top1約定 | 勝率 | gross平均 | 20bp後平均 | 20bp後中央値 | 複利損益 | 最大DD | プラス月 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 価格12特徴 | 142/142 | 50.70% | +0.0617% | **-0.1383%** | -0.1007% | -21.32% | -42.50% | 3/8 |
| **価格12 + TDnet 6** | **142/142** | **50.70%** | **+0.1242%** | **-0.0758%** | **-0.1007%** | **-14.80%** | **-40.78%** | **4/8** |

top2等金額の20bp後平均も価格のみ`-0.1548%/日`に対し、TDnet追加は`-0.0733%/日`だった。どちらも負であり、「2件出せば正になる」という結果ではない。

期間別top1は次のとおり。

| 期間 | 価格12特徴 | 価格12 + TDnet 6 | 差 |
|---|---:|---:|---:|
| early development | -0.2984% | -0.3321% | -0.0337pt |
| exploration | -0.4755% | -0.4036% | +0.0719pt |
| selection | +0.2278% | +0.3578% | +0.1301pt |
| **combined** | **-0.1383%** | **-0.0758%** | **+0.0625pt** |

TDnet追加でtop1順位が変わったのは142日のうち19日だけで、その損益差はプラス12日、マイナス7日、同一123日だった。日単位のpaired bootstrap 20,000回による平均差の90%区間は`[-0.134592, +0.271630]`ポイント/日である。この区間は特徴セットの発見にも使った開発期間上の診断値であり、holdoutの信頼区間ではない。

上位5日を除いたwinnerの平均は`-0.3290%/日`で、成績は一部の日に依存している。以上から、相対的な勝者として実装へ固定する判断と、実発注を許可しない判断を同時に採る。

### 時点制約と漏洩防止

- 価格特徴は対象日より前の行だけから計算し、対象日OHLCを参照しない。
- 月次walk-forwardの学習末日は各スコア月の月初より前とする。
- 候補ユニバースは対象日前営業日の情報で固定する。
- TDnet開示は`published_at <= 対象日08:58:59 JST`のときだけ対象日特徴へ入る。
- `training_end`後の日足・開示を追加しても、特徴、補完値、scaler、係数、学習hashが変わらないことをテストする。
- 全カレンダー日のTDnetページを要求し、HTMLとsidecarの日付、byte数、SHA-256、観測時刻を照合する。
- 対象日ページはdecision cutoff以後にリクエストを開始したものだけを受け入れる。
- 明示calendarで全行欠落日を検出し、日付を詰めて古い銘柄集合へ翌日を接続しない。
- 欠落した直前終値を古い値で埋めず、overnight特徴を欠測にする。
- 順位付け後の当日欠落は別銘柄への差し替えをせず、損益・コストとも0にする。
- artifactに特徴名、学習cutoff、calendar prefix、日足・TDnetのcutoff hashを保存し、推論時に照合する。

MarketSpeed IIの08:58板はモデル順位を変更せず、古いsnapshot、過熱した気配、買い特別気配、遅い予想寄りを`DISPLAY_ONLY`にする発注vetoとしてのみ使う。

### 再現方法

外部入力として、同一内容の日足、明示JPX営業日calendar、sidecar付きTDnet日別HTMLが必要である。

```bash
PYTHONPATH=src python research/compare_feature_sets.py \
  --daily /tmp/package_collected.pkl \
  --calendar /tmp/jpx_sessions_2024_2025.csv \
  --tdnet-cache /tmp/tdnet_full_cache \
  --output research/feature_set_comparison.json
```

出力:

```text
research/feature_set_comparison.json
research/feature_set_comparison_early_development_picks.csv
research/feature_set_comparison_exploration_picks.csv
research/feature_set_comparison_selection_picks.csv
```

JSONにはモデル仕様、12特徴セット、fold、全指標、paired bootstrap、入力・実装hash、runtime、TDnet観測provenanceを保存している。

検証コマンド:

```bash
PYTHONPATH=src python -m compileall -q src tests research
PYTHONPATH=src python -m unittest discover -v
python -m pip wheel . --no-deps --no-build-isolation \
  --wheel-dir /tmp/tse-session-ranker-wheel
```

最終確認時の環境はPython 3.12.13、NumPy 2.3.5、pandas 2.2.3、scikit-learn 1.8.0、joblib 1.5.3で、55 testsがソースツリーと隔離インストールしたwheelの双方で成功した。wheelのSHA-256は`221709c077a237158608c38ae5905cbc2ccb67de184a2a10f27586f84f47f3ca`。

テストは、価格・TDnet双方の時点制約、親カテゴリを伴わない方向語の除外、ToSTNeT・取得状況・払込完了の除外、公開時刻と取引日cutoffの対応、可変な当日cache、sidecar hash・provenance、export metadata改変、ミラーlayout変更、明示calendar、将来データ不変性、学習・推論parity、部分・継続・全行欠落、未約定の固定現金配分、初期NAV込みDD、cost検証、固定logit parameter、artifact checksum・payload整合性、旧schema拒否を含む。

### 実発注への昇格条件

0.3.0のモデル・特徴・コスト・候補規則を変更せず、PR作成後の完全新規データをshadow記録する。条件を変更した場合は、その版の成績をゼロから数え直す。

少なくとも次をすべて満たすまでは自動発注へ昇格しない。

- 60～100新規営業日の前向き記録
- 40約定以上、30独立取引日以上
- 実測コスト控除後の平均と中央値がともにプラス
- 日次平均の片側90%下限が0以上
- 4期間中3期間以上がプラス
- 上位5日除外平均がプラス
- 最大1日の利益寄与が総プラス損益の25%未満
- 明示calendar、日足、TDnetの収録率とsidecar provenanceが検証済み
- 推論時刻、候補、約定可否、実測スリッページを改変不能な形で毎日保存

昇格前に係数、閾値、特徴を調整した場合、その時点までのshadow期間は新仕様のholdoutとして再利用しない。

### 互換性

0.3.0はconfig schema **3**、artifact schema **5**である。特徴セットIDは`session_v3_tdnet_clear`、データ意味論は`prior_session_universe_source_mask_tdnet_v3`。0.2以前のartifactはTDnet cutoff、calendar、18特徴の整合性を証明できないため、0.3.0で再学習する必要がある。

</details>

---

## Entry 003 — tse-session-ranker 0.4.0：sealed holdout parser incident

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-21 |
| 検証対象ロック | `7b1d6922dcb591607730e58a98946511c600bf31bbd8ae2a43ee3be50ada83cb` |
| 親Entry | Entry 002 |
| 検証対象 | v0.4固定winnerの2025-08-04～2026-03-31 sealed holdout |
| 採用判断 | **損益評価前に停止。parser-only recoveryを1回だけ許可** |
| 当時の運用判断 | 結果未確定。実発注不可 |
| 主指標 | 未計算（metric evaluation count = 0） |
| 機械可読成果物 | `research/model_v04_holdout_parse_failure.json` |

> **一行結論:** 全PDF解析後の行数検査でfail-closedし、損益を一度も計算せず停止した。モデルを固定したまま、公式の特別気配接頭辞だけを解釈する監査可能なrecoveryへ進む。

<details>
<summary><strong>incidentと回復条件</strong></summary>

正式選定では、価格・市場・TDnetを統合した62特徴、`positive_session`ラベル、`C=0.03`、全履歴、日別・クラス均衡の正則化logitが開発winnerとなった。選定ロックのSHA-256は上記のとおり。ただし開発確認の往復20bp後平均は`-0.022342%/日`で、開発gateは不合格だった。

sealed holdoutは全入力hashとruntimeを照合し、消費receiptを排他的に作成してから160 PDFを解析した。その後のmanifest検査で、普通株として数えた行に未解析行があるため停止した。この時点ではparsed export、特徴パネル、モデル学習、候補、損益指標のいずれも作成していない。

行形式だけを再診断した結果、47ファイル・63行の全件が同一原因だった。JPXの`Final Special Quote`欄に、買い・売りを示す半角カナ`ｶ`または`ｳ`が数値の直前に付いていた。OHLC列のずれ、任意の4価格欠落、正の出来高を伴う価格全欠落は確認されなかった。

回復では次を固定する。

- winner、baseline、特徴、目的変数、C、学習窓、コスト、gate、bootstrapを変更しない。
- 元selection lock、元receipt、failure recordを削除・上書きしない。
- 既存実装の変更をJPX parserだけに限定し、selection runnerとprotocolはbyte単位で維持する。
- 旧parserが受理した全行のcanonical値が新parserでも完全一致することを確認する。
- 開発期間のcanonical価格出力が変わらないことを確認する。
- parser patch、非parser実装、全入力、runtime、固定winnerをrecovery lockへ結合する。
- recovery metric runは1回だけとし、開始前に別の排他的receiptを作る。
- 最終結果にはraw parse回数、metric evaluation回数、parser recovery実施、holdout retuningなしを明記する。

</details>

---

## Entry 004 — 研究プロトコルv0.4：特徴量選定とparser recovery後の非正式診断

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-21 |
| 検証対象コミット | 本PRへ収録。固定対象は下記selection lock・runner・diagnostic lockのSHA-256で識別 |
| 親Entry | Entry 003 |
| 検証対象 | 固定済み62特徴logitとv0.3 controlの2025-08-04～2026-03-31診断 |
| 採用判断 | JPX parser v6を採用。62特徴モデルは負のresearch baselineとして固定し、productionへ昇格しない |
| 当時の運用判断 | **実発注不可。研究・shadow表示のみ** |
| 主指標 | v0.4 top1・往復20bp後 `-0.086199%/日`、top2 `-0.059361%/日` |
| 正式評価状態 | formal metric run `0`。formal recoveryはモデル評価前のOOMで未完了 |
| 非正式診断状態 | `diagnostic_no_demonstrated_edge`。winner/controlを各1回だけ評価 |
| 機械可読成果物 | `research/model_v04_post_failure_diagnostic_result.json`<br>SHA-256 `36a1e7ec557d0549399392224d0acbd89f369012c294290724465605447854ed` |
| 独立再集計 | `research/model_v04_post_failure_diagnostic_result_audit.json`<br>SHA-256 `f81d83ffe85cbe324bc27523c93d619de485a59cf196354debbc35d0b75d628d` |

> **一行結論:** 正則化logitのまま価格・前後場・市場・TDnetを62特徴まで比較したが、固定winnerは非正式な未使用期間診断でも20bp控除後損益が負だった。仕様は再現用baselineとして確定し、自動発注へは採用しない。

<details>
<summary><strong>選定、正式評価事故、非正式診断、最終仕様</strong></summary>

### 目的と探索範囲

最適化対象は勝率ではなく、次のscheduled-day損益率とした。

```text
日次損益 = 始値→終値騰落率 - 約定時のみ往復20bp
主指標   = top1の日次損益平均
表示     = 毎日top1・top2
発注評価 = top1
```

推定器の種類はL2正則化ロジスティック回帰へ固定した。その上で、2021年以降の学習データを使い、時系列を次のように分離した。

| 区分 | 期間 | 用途 |
|---|---|---|
| feature screen | 2022-03-01～12-30 | 6特徴block × 4目的ラベル |
| model design | 2023-01-04～12-29 | 特徴・目的、C、学習窓、class weight |
| development confirmation | 2024-01-04～2025-07-31 | 月次walk-forwardで最終選定 |
| sealed期間 | 2025-08-04～2026-03-31 | 選定後に1回だけ確認する予定だった期間 |
| 既知除外期間 | 2026-04-01～07-21 | 以前の調査で閲覧済みのためholdoutにしない |

比較registryは、6特徴block、4目的ラベル、`C={0.03, 0.08, 0.2}`、504営業日またはexpanding学習窓、class weight有無で事前固定した。段階ごとに候補を減らし、後段から候補を追加していない。protocolとselection lockは次のとおり。

```text
research/model_v04_protocol.json
b7ccbf62de187cd8f70fdb8fc0acca2fb598ce978d948639b7453d0d60ed30c1

research/model_v04_selection_lock.json
7b1d6922dcb591607730e58a98946511c600bf31bbd8ae2a43ee3be50ada83cb
```

### 固定winner

```text
特徴block:       session_market_tdnet（62特徴）
目的ラベル:      close > open
推定器:          L2 LogisticRegression
C:               0.03
class weight:    balanced
日付weight:      各学習日の合計を等しくする
学習窓:          2021-01-04以降のexpanding
最小学習期間:    252 sessions
再学習:          月次
random state:    31
```

62特徴の内訳は次のとおり。

```text
前営業日までの価格・前後場shape: 34
前営業日までの市場context:         5
08:58:59までのTDnet・interaction:  23
```

価格群には、日中・夜間騰落、5/20/60日momentum、ATR固定変換、横断rank、前場・後場の騰落と値幅、昼休みgap、後場終値位置を含む。市場群には前日の等金額市場騰落、60日beta、市場残差とbeta・ATR interactionを含む。

TDnet群には開示有無・件数・発表時刻、決算、業績修正方向、配当方向、自己株取得、ToSTNeT、エクイティ調達、優待、分割、M&A、支配権取引、減損、監査問題、および事前値動き・市場contextとの固定interactionを含む。PDF本文の業績修正額や買付規模は使用していない。

development confirmationのwinnerは386 scheduled sessionsで、往復20bp後top1平均`-0.022342%/日`、プラス月8/19、上位5日除外平均`-0.152758%/日`だった。3 finalistに対するmoving-block reality-checkのp値は`0.674516`。したがってselection lock作成時点から状態は`development_best_no_edge`であり、正の優位性を主張していない。

### JPX parser incidentと回復

Entry 003の初回正式runは、JPX相場表の`Final Special Quote`欄に付く半角カナ`ｶ`・`ｳ`を解析できず、損益計算前にfail-closedした。

parser v6では、この欄の数値直前だけにmarkerを許可した。

```text
対象PDF:                    160
旧parser受理行:             622,661
旧parser拒否行:                  63
新parser受理行:             622,724
新parser拒否行:                   0
marker内訳:                 ｶ 37、ｳ 26
旧parser受理行の同一性:     完全一致
return・metric参照:         なし
```

開発データでも4,788,079行、4,387銘柄、canonical 23列とモデル入力内容が完全一致した。

```text
parser recovery audit
317d5cde9741429263cc91c11575300660f94f62123725fc8ce2d76cdee02eeb

development parser compatibility
ee8bc1c30aa8409d18650ca5eaf30e35d3fcc8969bfa2982eba1900ebef9eb9f
```

正式recovery runは全160 PDF・622,724行を拒否0で解析した。その後、特徴パネル構築中に20GiBのcgroup上限へ達し、exit 137で終了した。

```text
feature panel build:        開始、未完了
winner evaluation count:    0
v0.3 control evaluation:    0
formal metric run count:    0
picks・result:               未作成
```

正式runのlock、消費receipt、failure recordは上書きせず保存した。

```text
formal recovery lock
1f87a81e21175dc52c82f8a42762ab52cfd8ab39a3b18bcdb6061d908ab95554

formal recovery receipt
dfaaa3cf94614e76a5bc826386d63380cbcd0473523744d2571dce98125587e2

formal runtime failure
f34912dbc9c9c787d10caffc9766e132f6527f3ccc1ab3acb66a9145286bb5c3
```

このためformal recoveryは未完了で、formal validationは成立していない。

### メモリ制約下の非正式診断

正式runを再試行せず、同じ特徴をメモリ制約内で生成する診断専用panel builderを別途作成した。開発期間のwinnerとv0.3 controlについて、旧パネルと候補の非return列が完全一致し、return差の最大値は双方`4.44e-16`だった。

```text
diagnostic panel builder
fe2a38b8e67c3387896c43eb096325f6083d1ecf157a7a033de80e556f450f1f

development equivalence audit
7870171ec64788236003337030a6ee6fc456a1d1567932acb88e07cb7c33cb4c
```

ただし、このpanelでは診断前に対象期間のラベルがmaterializeされている。正式runの実行権も既に消費されているため、後続結果をsealed formal holdoutまたはformal validationとは呼ばない。

診断runnerの監査中、同じファイルをhash確認後に別openするTOCTOUと、formal/non-formal状態名の曖昧さを検出した。旧lock `74643bd1da5e5a72565f275304bf7c919e9a358a52b03df784466631706b39dc` は**未消費のまま使用禁止**とした。修正版は同一file descriptorで`hash → deserialize → 再hash`し、状態を分離して記録する。

```text
diagnostic runner
38ce95b092f951194452385b41f9a22de0b36957d771204dd0e77ad1253a3613

authoritative diagnostic lock v2
868a496459ab2276dd2e6cd9a68e8e1e110883e7fed8219c687d317c123abd06

diagnostic receipt
01b62b92f87e1d4feb895c66712d209068f760b3b44e693fd3f88e9fcb978b4a
```

```text
formal metric run count:       0
diagnostic metric run count:   1
winner evaluation count:       1
v0.3 control evaluation count: 1
診断内の再選択・再調整:        なし
```

### 非正式診断結果

評価対象は159 scheduled sessions。153日で候補を表示し、表示率は96.226%だった。

| 仕様 | 20bp後平均 | 中央値 | 40bp後平均 | PF | プラス月 | 上位5日除外平均 |
|---|---:|---:|---:|---:|---:|---:|
| **v0.4 top1** | **-0.086199%** | 0.000000% | -0.278652% | 0.9021 | 2/8 | -0.254070% |
| **v0.4 top2** | **-0.059361%** | 0.000000% | -0.251813% | 0.9119 | 2/8 | -0.193278% |
| v0.3 control top1 | -0.087886% | -0.071877% | -0.280338% | 0.8976 | 3/8 | -0.313019% |
| v0.3 control top2 | +0.111147% | +0.041531% | -0.081306% | 1.1974 | 4/8 | -0.078121% |

v0.4 top1はgross平均`+0.106254%/日`、約定銘柄勝率54.90%だったが、往復20bpを控除すると負になった。勝率が50%を上回っても、今回の損益目的を満たさない。

v0.4 top1のmoving-block bootstrap片側90%下限もすべて負だった。

```text
block 5:   -0.320354%
block 10:  -0.312789%
block 20:  -0.337140%
```

v0.3 controlとの差は`+0.001686pt/日`に過ぎず、paired bootstrap片側下限はblock 5/10/20のすべてで負だった。順位は大きく変わっても、損益改善へ結び付いていない。

v0.3 control top2は20bpで正だったが、事前登録した主目的ではない。4/8か月しかプラスでなく、上位5日除外平均と40bp stressが負である。この期間を見た後にtop2へ切り替えると後付け選択になるため、productionへ昇格しない。

独立監査はpicksから20bp・40bp損益を再集計し、JSONとの差が最大`3.33e-16`であることを確認した。監査時に、両モデルとも候補なしだった6日を`NaN != NaN`として数えたため、result内の`ranking_changed_days=149`は正しくは`143`と判明した。これは表示項目で、gate・損益・bootstrap・statusには使われない。元resultを上書きせず、次の追補監査へ訂正を保存した。

```text
result audit runner
e5f3593023af127d9bea1dc085fd11a692431aaa94c5e675f5b1f411362a2f7f

result audit artifact
f81d83ffe85cbe324bc27523c93d619de485a59cf196354debbc35d0b75d628d

audit checks: 27 / 27 passed
```

### 先物特徴

日経225先物・TOPIX先物の08:58:59以前の騰落率と、個別株beta・ATR・TDnet材料とのinteractionは、有力な前向き仮説として実装した。

```text
market_beta_60 × futures_return
ATR14 × abs(futures_return)
market_beta_60 × Nikkei-TOPIX futures spread
TDnet material flag × futures_return
```

ただし正確な過去時点snapshotがないため、今回選定した62特徴には含めていない。日足終値、9:00以後の値、後から確定した値を代理投入しない。`ingest-market-context`はtimezone付き`observed_at`、同日08:58:59以前、明示的な`return_definition`を必須にし、過去行の変更と欠測日のcarry-forwardを拒否する。

```text
historical status:  未検証
current model:      不使用
next action:        PR後に08:58:59以前のsnapshotを毎日append-only保存
```

### 最終判断

```text
JPX parser v6:             採用
62特徴logit:               負のresearch baselineとして仕様確定
production model昇格:      なし
自動発注:                  不可
CORE / RESERVE表示:        shadow・診断用途のみ
formal validation:         未成立
```

同じ2025-08～2026-03期間で特徴、C、top-kを再選択しない。v0.3 control top2も今回の結果から採用しない。次の仕様変更は新しいprotocolとして登録し、PR後に収集した先物snapshotを含む完全新規期間で評価する。

</details>

---

## Entry 005 — 研究プロトコルv0.5：広域モデル・特徴量比較（r2）

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-22 |
| 検証対象コミット | 本PRへ収録。protocol・実装・入力・成果物のSHA-256で固定 |
| 親Entry | Entry 004 |
| 検証対象 | 16モデルrecipe・6 family group・4特徴block。計47候補評価 |
| authoritative protocol | `model_v05_broad_family_retrospective_r2_20260722`<br>SHA-256 `d2efa76555a5a4b62a1eefa186c8077e0c1891fa27a7bc0f01dacc463a721d3f` |
| 採用判断 | 選定winnerは確認期で再現せず棄却。production変更なし |
| 当時の運用判断 | **retrospective research only。実発注不可** |
| 主指標 | 固定winnerの確認期top2・往復20bp後 `-0.484256%/日`、block-5片側90%下限 `-0.686340%` |
| gate | `retrospective_gate_failed`。7条件中、表示率2条件だけ通過 |
| 機械可読成果物 | `research/model_v05_broad_family_result.json`<br>SHA-256 `80ce66830b3dd2f7dd56260f807f936bbc9945e82184e9f39034b1213a59facd` |
| 独立再集計 | `research/model_v05_broad_family_result_audit.json`<br>SHA-256 `131cccb8923bff43615d8d0a559407553644f23ad41892160d71b21b3f17678b` |

> **一行結論:** logit固定を外して幅広く比較したが、選定期首位の18特徴raked logitは固定後の確認期で大幅なマイナスへ反転した。TDnet 62特徴の診断対照も頑健性条件を満たさず、採用できる損益優位性は確認できなかった。

<details>
<summary><strong>preflight、固定設計、結果、制約、再現情報</strong></summary>

### Preflight failureとr2への改訂

r1はモデル指標を1件も計算する前に停止した。価格履歴60日を必要とするため、当初のfamily screen開始日`2024-06-03`以前に使える学習日は40日しかなく、事前固定した最低80日を満たさなかった。

```text
r1 protocol:
research/model_v05_protocol_preflight_001.json
da1a1498c663678e8044838c6b03ed89d656c4747157d94c44e9fcebd579b2a2

r1 input lock:
research/model_v05_input_lock_preflight_001.json
a765980d9ab989f8bcc55f83b7bc3459ef6106c5bd75dc1d4451ede787180d6d

r1 failure record:
research/model_v05_preflight_failure_001.json
a236c688ef19025d84835e7b670ae628857f40d96199a82e21541ed976c43f71

completed_candidate_metrics: 0
```

損益、順位、候補銘柄を一切見ていない段階だったため、r2では実データの利用可能境界に合わせて次の2条件だけを改訂した。識別子、登録時刻、r1への参照も更新し、入力lockとpanelを作り直した。

```text
family screen開始:       2024-06-03 → 2024-07-01
最低eligible学習日数:   80 → 60
```

最初の2024年7月foldは、2024-04-03～06-28のちょうど60学習日・201,708行で境界を通過する。39個の`session_market`特徴に全欠測列はない。r1の失敗記録とprotocol bytesは上書きせず保存した。

### 目的と固定評価設計

勝率ではなく、毎日1～2件を表示したときのコスト控除後損益を主目的にした。

```text
ラベル:             当日始値→終値騰落率
主指標:             top2を50:50で保有した日次平均
主コスト:           約定時に往復20bp
stress cost:        往復40bp
表示枠:             全営業日2枠
欠員・未約定:       現金0%、他枠へ事後再配分しない
最低表示率:         rank1、rank2とも95%
学習窓:             expanding、月初に再学習
学習cutoff:         各採点月より厳密に前
```

期間を次の順に固定した。

| 区分 | 期間 | 用途 |
|---|---|---|
| family screen | 2024-07-01～08-30 | 16 recipeを比較し、6 familyごとに1件だけ通過 |
| feature design | 2024-09-02～12-30 | 6 family代表 × 4特徴blockと固定controlを比較 |
| development confirmation | 2025-01-06～03-31 | feature designで固定したwinnerを再選択なしで確認 |
| known benchmark reference | 2025-04-01～07-31 | 既知・非sealed。選定にもgateにも使用しない |
| prospective final | 2026-07-23以降100営業日 | 唯一production昇格を判断できる将来期間 |

feature designの上位3件を記録したが、research championは同段階の主指標1位へ固定した。confirmationで別候補の成績が良くても差し替えない。途中のmodel failure、非収束、入力変更、実装変更は実験全体を停止する設計とし、部分完走したregistryからは選ばない。

### 比較したモデルと特徴量

正則化ロジスティック回帰への固定を外し、次の6 family group・16 recipeを事前登録した。

```text
linear classifier:
  legacy / date-class raked / net20 label / return-weighted logit

linear return:
  ridge / elastic-net SGD / Huber SGD

gradient boosting:
  squared / absolute return / top-quintile分類 / 日次return percentile

bagged trees:
  ExtraTrees回帰 / ExtraTrees top-quintile分類 / RandomForest回帰

ranking・ordinal:
  pairwise linear rank / multi-threshold expected return
```

全recipeはcontent-addressed ID、固定parameter whitelist、固定random stateを持つ。学習日は日別等weightとし、該当recipeではクラス周辺もrakingした。tree系は一様なrow bootstrapで日別weightを崩さないよう`bootstrap=False`で固定した。

| 特徴block | 特徴数 | 内容 |
|---|---:|---|
| `legacy_price_12` | 12 | 前営業日までの価格・overnight履歴 |
| `legacy_v03_18` | 18 | 価格12 + 寄り前TDnet 6 |
| `session_market` | 39 | 価格、前後場shape、市場context |
| `session_market_tdnet` | 62 | 39特徴 + TDnet・interaction 23 |

TDnetは開示タイトルの有無、件数、公開時刻、決算、業績・配当修正、自己株取得、ToSTNeT、資金調達、優待、分割、M&A、減損、監査問題などを08:58:59 JST以前だけで生成した。PDF本文の修正額、配当利回り、買付規模、取得倍率は構造化していない。

### データとprovenance

JPX月次相場表PDF 19件をparser v6で解析した。個別URL・ファイルSHA・runtimeはinput lockに保存した。

```text
元JPX行数:             1,523,928
full session:          1,431,968
partial session:          34,495
no-trade:                 57,465
rejected:                      0
銘柄コード:                4,124
営業日:                      386
期間:             2024-01-04～2025-07-31
calendar SHA:     966f4a416d0929487d850b16be87c9c1cd69cd9f8c0447a3e76214a5715c489e
```

元PDFには前場・後場OHLCがある一方、利用可能な出来高、売買代金、売買単位はない。これらを他の列で代用していない。

特徴生成後のpanelは1,524,104行、4,124コード、386営業日。

```text
panel SHA:
e2d8ef255b8988b0b0773a31f18859d236d8c6350850a5a764da6dbbeba669b9

panel manifest:
research/model_v05_panel_manifest.json
94cb0e8802957318edf522b892b67fe351432f15820d6534985609b9acdfc59d

input lock:
research/model_v05_input_lock.json
02370bda9c5fe73b166c557bcdc837d363f5deafbe91baa2d450b33dfcd45272
```

TDnetは第三者の日別公開ミラーから461 HTMLと461 sidecarを取得し、97,006開示を解析した。全sidecarのprovenanceは`network_request_start_recorded`で、観測時刻は2026-07-22 12:10:43～12:17:52 JST。選定・confirmationに必要な日別ページは2025-03-31まで完全だった。第三者ミラーの配信遅延や後日の訂正を完全には排除できない。

```text
TDnet dataset SHA:
fb217159b5b03146cb0868bf87abe8a936e958d69d0f7acfe72a8864824c446e
```

runnerはpanel全列・dtypeのsemantic hash、営業日、入力・protocol・全実装SHA、runtime fingerprintをinput lockへ結合し、モデル評価前後に再照合した。run receiptは排他的に作成した。

### Family screen

7～8月の43営業日で各family groupから1件を選んだ。全候補が両枠を100%表示したが、6代表の主指標はすべてマイナスだった。

| family代表 | 目的 | top2・20bp後 | block-5片側90%下限 | rank2表示率 |
|---|---|---:|---:|---:|
| raked logit `C=0.03` | 上昇分類 | -0.463808% | -0.967333% | 100% |
| ridge `alpha=10` | raw return | -0.543373% | -1.051507% | 100% |
| HGB rank percentile | 日次return順位 | -0.390728% | -0.917068% | 100% |
| ExtraTrees top quintile | 日次上位20%分類 | **-0.009938%** | -0.566123% | 100% |
| pairwise linear rank | 同日pairwise | -0.770386% | -1.302295% | 100% |
| multi-threshold | 閾値積分return | -0.644450% | -0.947327% | 100% |

### Feature designとwinner固定

9～12月で上位3件は次のとおりだった。

| 順位 | model・特徴block | 特徴数 | top2・20bp後 | block-5下限 | 上位5日除外 | top1・20bp後 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | raked logit・`legacy_v03_18` | 18 | **+0.210773%** | -0.044446% | -0.036103% | -0.068103% |
| 2 | raked logit・`legacy_price_12` | 12 | +0.100476% | -0.159399% | -0.149331% | -0.101164% |
| 3 | ridge return・`session_market` | 39 | +0.076696% | -0.566666% | -0.694987% | -0.773118% |

主指標1位の次をresearch championとして、この時点で固定した。

```text
model:             raked LogisticRegression
C:                 0.03
objective:         close > open
weight:            学習日・クラス周辺をraking
features:          価格12 + TDnet 6
candidate ID:      raked_logit_c003__636907555fe518d0__f_legacy_v03_18
```

ただし選定時点でもblock-5下限、上位5日除外、top1は負で、点推定は脆弱だった。同じraked logitでは、TDnet追加により選定期の点推定が12特徴`+0.100476%`から18特徴`+0.210773%`へ改善した。一方、39特徴は`-0.271138%`、TDnetを加えた62特徴も`-0.121333%`で、特徴増加そのものに一貫した優位性はなかった。

### Development confirmation

固定後の2025年1～3月・57営業日で再選択せず評価した。

| 仕様 | top1・20bp後 | top2 gross | top2・20bp後 | 40bp後 | PF | プラス月 | 上位5日除外 | block-5下限 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **固定winner：raked logit・18特徴** | -0.487961% | -0.284256% | **-0.484256%** | -0.684256% | 0.4437 | 0/3 | -0.729231% | -0.686340% |
| 固定price-only reference：raked logit・12特徴 | -0.515772% | -0.311120% | -0.511120% | -0.711120% | 0.4292 | 0/3 | -0.755213% | -0.695925% |
| legacy logit・18特徴control | -0.606472% | -0.315839% | -0.515839% | -0.715839% | 0.4672 | 0/3 | -0.792530% | -0.726790% |
| raked logit・62特徴TDnet診断control | -0.030024% | +0.239661% | **+0.039661%** | -0.160339% | 1.0541 | 1/3 | -0.373447% | -0.318735% |
| 上位3件equal-rank ensemble | -0.370969% | +0.091508% | -0.108492% | -0.308492% | 0.8874 | 1/3 | -0.706907% | -0.588760% |

固定winnerはrank1・rank2を57/57日表示し、top2は114/114枠約定した。それでも20bp後中央値`-0.360154%`、複利損益`-24.7342%`、最大drawdown`-24.7342%`だった。選定期の正の点推定は確認期で再現しなかった。

62特徴TDnet診断controlだけは20bp後の点推定が正だったが、これはfeature designの首位ではなく、confirmationを見てから差し替えることは禁止されている。さらに40bp、上位5日除外、3か月中2か月、bootstrap下限が負で、頑健な優位性を示していない。次期winnerとして採用せず、仮説として将来データへ持ち越すだけとした。

retrospective gateは次のとおり。

| 条件 | 実績 | 判定 |
|---|---:|---|
| top2・20bp後平均 `>= 0` | -0.484256% | 不合格 |
| top2・上位5日除外平均 `>= 0` | -0.729231% | 不合格 |
| top2 PF `>= 1` | 0.4437 | 不合格 |
| プラス月比率 `>= 60%` | 0/3 | 不合格 |
| top2 block-5片側90%下限 `>= 0` | -0.686340% | 不合格 |
| rank1表示率 `>= 95%` | 100% | 合格 |
| rank2表示率 `>= 95%` | 100% | 合格 |

```text
numeric_gate_passed:          false
production_promotion_allowed: false
decision:                     retrospective_gate_failed
```

### Known benchmark reference

2025年4～7月は過去調査ですでに閲覧済みで、真のholdoutではない。選定・gate・候補差し替えには使わなかった。

固定winnerはTDnetを必要とするが、必要な84営業日のうち完全な日別ページは5日、coverageは5.95%だけだった。このため損益0として扱わず、winnerのbenchmark評価そのものを行っていない。欠落79営業日はresult JSONに列挙した。

価格のみの固定referenceは84営業日で評価できた。

```text
top2 gross平均:             -0.092995%/日
top2・20bp後:               -0.291805%/日
top2・40bp後:               -0.490614%/日
block-5片側90%下限:        -0.444794%
PF:                          0.5977
プラス月:                    1/4
約定枠:                    167/168
```

これは参考値であり、retrospective gateへ含めていない。

### 独立監査と再現情報

独立監査runnerはresultに埋め込まれた指標を信用せず、4つのpicks CSVからコスト、日次portfolio、月次値、drawdown、bootstrap、family shortlist、feature finalist、winner固定を再計算した。

```text
audit checks:                         18 / 18 passed
再計算したcandidate metrics:          47
最大絶対差:                           4.62e-14

audit runner:
research/audit_model_v05_broad_family_result.py
71bd914c2220cea3b38bf67ca180970660c1818752a7daa6cc9b42af1099afb9

audit artifact:
research/model_v05_broad_family_result_audit.json
131cccb8923bff43615d8d0a559407553644f23ad41892160d71b21b3f17678b
```

主要成果物は次のとおり。

```text
protocol:
research/model_v05_protocol.json
d2efa76555a5a4b62a1eefa186c8077e0c1891fa27a7bc0f01dacc463a721d3f

result:
research/model_v05_broad_family_result.json
80ce66830b3dd2f7dd56260f807f936bbc9945e82184e9f39034b1213a59facd

result manifest:
research/model_v05_broad_family_result.manifest.json
46c0983660617474d6e6a06bafb065dccc3bf63e638968aaed8468ae8ee00f29

exclusive run receipt:
research/model_v05_broad_family_result.run.json
f9a602db78caacd3105c293903bf21ef8f9aa32d36b903f52e4986a37dd1a882

runner:
research/compare_model_families_v05.py
b32b4f64d00e432a5ae95f1fc723afd0d93b697c4f910914fda2c843fd56539d

model family implementation:
src/tse_session_ranker/research_models.py
0e8dd5bbbaea784e40975d3567163f2e153d05e7cf3ae2cd33bcd09e63467a16
```

stage別picksのSHAは次のとおり。

```text
family_screen:              da9c3f0989afd9d67ef9c621d4fbc37dfc33f0eea7328c8a62037839d288e10c
feature_design:             80f0f0f70f3d302178ba303f5d685d173c875c9ac94ce333e834a87a6695a66f
development_confirmation:  3117e316d7095c785892a6cac1887c3749dde74d9aa6aeef2077b26364ac301b
known_benchmark_reference:  aadbaf42104805b7f1b52e0aed7f459f1d5ad6eb15230fbccfa4a4fec0f345b1
```

実行環境と検証結果。

```text
Python 3.12.13
NumPy 2.3.5
pandas 2.2.3
scikit-learn 1.8.0
joblib 1.5.3
pdftotext 24.02.0
thread limit 1
metric elapsed 1,957.14秒
peak RSS 2,496,752 KiB

PYTHONPATH=src python -m unittest -v
122 tests passed

compileall: passed
artifact SHA cross-check: 30 / 30 passed
wheel build: passed
wheel SHA: dc7983c5a108b801fd7eb95fd6c8b860c1560bd12422e420088022a7b2e1508a
```

### 制約と最終判断

この比較には次の制約がある。

- 2025-07-31までの全期間は既知のretrospective dataで、sealed holdoutではない。
- 16 recipeと4特徴blockを探索しており、段階分離後もmodel-selection biasは残る。
- 正確な過去08:58:59時点の日経225先物・TOPIX先物snapshotがなく、先物特徴は今回使用していない。9:00以後や日足終値で代用していない。
- 元JPX PDFに出来高、売買代金、売買単位がなく、流動性・約定容量を評価できない。
- 20bp・40bpは固定仮定で、実測slippageやmarket impactではない。
- TDnetはタイトル分類で、本文の定量情報を利用していない。
- model scoreは同日順位用で、校正済みの上昇確率ではない。
- production artifact、CLI/API既定モデル、発注可否は変更していない。

今回確定できたのは「モデルを広げても、既知期間で再現可能な損益優位性は見つからなかった」という負の結果である。正の点推定だった62特徴TDnet controlへ後付けで切り替えず、同じ期間で閾値やfeatureを再調整しない。

```text
research champion:          確認期で棄却
retrospective edge:         未実証
production model変更:       なし
自動発注:                   不可
次の正式判断:               2026-07-23以降100新規営業日
```

将来期間では08:58:59以前の先物snapshot、TDnet完全日、実測の板・約定・slippageをappend-onlyで保存する。protocolを事前固定し、100営業日が完了するまで途中の成績でmodel、特徴、top-kを差し替えない。

</details>
---

## Entry 006 — 研究プロトコルv0.6：特徴量候補棚卸し・group ablation

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-22 |
| 検証対象コミット | 本PRへ収録。catalog・protocol・実装・入力・成果物のSHA-256で固定 |
| 親Entry | Entry 005 |
| 検証対象 | 共通価格core + 7単独追加block + 1条件付きinteraction block。固定raked logitで特徴寄与だけを比較 |
| 候補台帳 | `research/model_v06_feature_catalog.json`<br>SHA-256 `bbf87983979d48afbe1eaa51f3368cca713c306f2b7d4cdc8cda54e5fe43414a` |
| authoritative protocol | `model_v06_feature_group_diagnostic_20260722`<br>SHA-256 `20038eb33d87a1fbcd187548f7f14c65dca136d00850566311a6888bd0083021` |
| 採用判断 | 一次通過した前後場shapeは次期間で逆転したため撤回。production変更なし |
| 当時の運用判断 | **retrospective development diagnostic only。実発注不可** |
| 主指標 | 一次首位`G0+G3_session_dynamics` top2・20bp後 `-0.218990%/日`。次期間で価格core比 `-0.593191pt/日` |
| 機械可読成果物 | `research/model_v06_feature_result.json`<br>SHA-256 `bdf91871e05dde03579b2521e4fcd558683b2ed9ab9de750b9535614261be6c1` |
| 独立再集計 | `research/model_v06_feature_result_audit.json`<br>SHA-256 `2e36b3a68cfe60f059f8859aa6e84967b4d9a3523e2636f71074b1151720be90` |

> **一行結論:** 特徴候補を時点・データ可用性・機構ごとに先に固定して再検証したが、既存履歴から安定して残る追加blockは0個だった。次は実始値を代用せず、08:58先物・予想gap・板・PTS・定量TDnetを前向き収集する。

<details>
<summary><strong>候補棚卸し、taxonomy修正、検証結果、再現情報</strong></summary>

### なぜ候補棚卸しからやり直したか

v0.5の12/18/39/62特徴は、同じbaseへ一群ずつ足した比較ではなかった。そのため成績差が価格、前後場、市場、TDnetのどの機構によるものか分離できなかった。v0.6では損益を見る前に候補を次の4区分へ分けた。

```text
historical_screen:
  既存JPX PDF / TDnet HTMLから時点を守って再生成可能

forward_only:
  08:58時点では使えるが、正確な履歴snapshotがない

blocked_source:
  有力だが、当時の原本・単位・構成銘柄履歴がない

exclude:
  target-session leakage、意味の誤ったproxy、または高重複
```

候補一覧、算式、機構、最低履歴、source制約、採否理由は次の2ファイルだけで追える。

```text
human-readable inventory:
research/model_v06_feature_candidates.md
e8907787b9227504eec2939d775f3926b1fde0a126af3dbcb8de0f6716b2f7e3

machine-readable catalog:
research/model_v06_feature_catalog.json
bbf87983979d48afbe1eaa51f3368cca713c306f2b7d4cdc8cda54e5fe43414a
```

### 固定したhistorical block

| Block | 追加特徴数 | 機構 |
|---|---:|---|
| `G0_price_core` | 15 | 既存のOC、overnight、volatility、close momentum/location。全比較の共通base |
| `G1_short_reversal` | 4 | 前日close-to-close shock、3日momentum、range shock、短長vol比 |
| `G2_gap_trait` | 4 | 過去gapの分散、fill率、反応beta、gap頻度 |
| `G3_session_dynamics` | 7 | AM平均・勝率、midday gap、AM/PM相関、close location、range圧縮 |
| `G4_market_regime` | 17 | 市場breadth、dispersion、tail、trend、volとbeta/ATR interaction |
| `G5_liquidity_proxy` | 4 | 無約定率、flat OC率、高安同値率。出来高の偽proxyは不使用 |
| `T0_clean_event` | 24 | 修正版TDnet title taxonomy |
| `T1_event_structure` | 13 | 件数、時刻、bundle、stage、過去60/252日開示強度 |
| `X0_event_context` | 6 | TDnet family×事前run-up、市場tail。`T0`通過時だけ評価 |

前日市場平均のように全銘柄で同値の特徴は、線形モデルの同日順位を単独では変えない。そのため`G4`には`beta×市場trend`、`ATR×dispersion`など、順位へ作用するinteractionを最初から限定登録した。結果を見て任意の掛け合わせを追加していない。

### TDnet taxonomyの修正

既存461日・97,006タイトルを調べ、次の汚染を確認した。

```text
旧ma_alliance 4,288件中、buybackとの重複: 1,303件（30.4%）
原因: 裸の「株式取得」が「自己株式取得」にも一致

旧equity_financing 2,694件中、
株式報酬・SO・役職員向けと見られるタイトル: 1,437件（53.3%）

業績修正4,496件中、タイトルだけで上方/下方を判定可能: 136件（3.0%）
```

研究用`T0`では、自己株買いをM&Aから除外し、外部資金調達と株式報酬を分け、業績・配当の方向不明を明示した。ToSTNeTも「取得決定を消す」のではなくstageとmethodを別列にした。production 6特徴の意味はこのEntryでは変更していない。

### 時点制約と自動テスト

価格候補はすべて`D-1`以前で計算する。対象日以降のOHLC・前後場OHLCを一括で2.75倍に変更して再構築し、対象日までの36個の追加価格特徴がbit-exactで不変であることを確認した。

TDnetは各文書を、公開時刻以後で最初に到来する`08:58:59 JST` cutoffへ割り当てる。次を自動テストした。

- `自己株式取得`がM&Aにならない
- 譲渡制限付株式報酬が外部増資にならない
- 方向語のない業績修正は`direction_unknown=1`
- 08:59開示は当日08:58 cutoffへ入らない
- TDnet source欠落はNaNのまま、完全日に開示なしだけ0
- 登録外interactionを生成しない
- catalogの列順と実装定数が一致

```text
PYTHONPATH=src:. python -m unittest discover -v
128 tests passed
```

### 入力とpanel

catalog、protocol、runner実装、JPX PDF 19件、TDnet HTML/sidecar 461日をmetric前にinput lockへ結合した。

```text
input lock:
research/model_v06_feature_input_lock.json
06d0fbf6f4395a0a760f73b8a35f23a20505df3c34658bb4e5bb6d96bfd31d86

panel:
1,524,104行 / 4,124コード / 386営業日
calendar SHA:
966f4a416d0929487d850b16be87c9c1cd69cd9f8c0447a3e76214a5715c489e

panel file SHA:
f2494efe6e66c02307b736b231a6759895ba8dc8471b4060ebc34dbd87e82cfd
```

label-blind QAでは全price sourceがtargetより前であることを確認した。group availabilityは`G2_gap_trait`だけ94.999%で98%基準未達、その他追加groupは99.03～100%。絶対相関0.95以上は4組で、主に`revision`と`revision_direction_unknown`、TDnet event有無とevent時だけ非ゼロになる構造値だった。

### 比較設計

特徴寄与を分離する間だけモデルを固定した。

```text
model:           raked LogisticRegression
C:               0.03
label:           close > open
weight:          日別合計同一 + class margin同一
retrain:         月次expanding
primary metric:  毎日top2・50:50、往復20bp後平均
stress:          往復40bp
display:         source-complete日はrank1/2を常に保存
trade:           OOF expected-net gate未確定のため別CSVを空で保存
```

期間は次の順に固定した。すべて既知期間なので、名称にかかわらずproductionを昇格できるholdoutではない。

| 段階 | 期間 | 営業日 | 用途 |
|---|---|---:|---|
| group screen | 2024-07-01～10-31 | 84 | `G0`対`G0+1 block` |
| union prune | 2024-11-01～12-30 | 41 | 通過block unionと1回だけのleave-one-group-out |
| stability report | 2025-01-06～03-31 | 57 | 固定recipeを再選択なしで報告 |

単独block通過には、availability 98%以上、G0比uplift正、プラス月50%以上、top5除外uplift非負、共通5日block resampleによるmax-T調整片側80%下限非負をすべて要求した。

### group screen結果

すべてtop2等金額・往復20bp後。

| 追加block | 特徴数合計 | 平均 | G0比uplift | max-T 80%下限 | プラス月 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| `G0_price_core` | 15 | -0.593657% | — | — | 1/4 | base |
| `G1_short_reversal` | 19 | -0.421963% | +0.171695pt | -0.152613pt | 0/4 | 不通過 |
| `G2_gap_trait` | 19 | -0.723076% | -0.129418pt | -0.453726pt | 0/4 | 不通過 |
| **`G3_session_dynamics`** | 22 | **-0.218990%** | **+0.374668pt** | **+0.050360pt** | **2/4** | 一次通過 |
| `G4_market_regime` | 32 | -0.238768% | +0.354890pt | +0.030582pt | 1/4 | 月次条件不通過 |
| `G5_liquidity_proxy` | 19 | -0.658416% | -0.064759pt | -0.389067pt | 1/4 | 不通過 |
| `T0_clean_event` | 39 | -0.581658% | +0.011999pt | -0.312309pt | 1/4 | 不通過 |
| `T1_event_structure` | 28 | -0.591744% | +0.001913pt | -0.322395pt | 1/4 | 不通過 |

一次通過した`G3`も20bp後の絶対損益は負、40bp後は`-0.416609%/日`、block-5片側90%下限は`-0.551913%`だった。相対改善だけで実発注候補にはしない。

### union pruneとstability

次の2024年11～12月では結果が逆転した。

```text
G0 + G3:
  top2 net20  -0.174880%/日
  プラス月    0/2

G0 only:
  top2 net20  +0.418311%/日

G3の期間外寄与:
  -0.593191pt/日
```

事前規則どおり`G3`を外し、retrospective locked recipeは`G0_price_core`へ戻した。2025年1～3月のstability reportは次のとおり。

```text
top1 net20:            -0.543763%/日
top2 net20:            -0.421878%/日
top2 net40:            -0.621878%/日
top2 median net20:     -0.254140%/日
top2 profit factor:     0.4892
プラス月:              0/3
block-5片側90%下限:   -0.615878%
```

したがって、今回のhistorical blockから固定できる追加特徴は0個。screenの点推定首位へ後付けで切り替えず、production artifact、CLI既定モデル、発注可否を変更しない。

### 独立再集計

独立audit runnerはresult内の集計値を信用せず、3つのdisplay-picks CSVから12 recipeの損益、bootstrap、max-T uplift、qualification、survivor、leave-one-out、stability差を再計算した。

```text
audit checks:                     15 / 15 passed
再計算recipe:                     12
比較した数値:                     1,926
最大絶対metric差:                 2.84e-14
最大絶対max-T差:                  5.55e-17

audit runner:
research/audit_model_v06_feature_result.py
5b538a6b27da64fbab1713a31fc0ecc1491a0b241c62fc32a454d9375e23c54a

audit artifact:
research/model_v06_feature_result_audit.json
2e36b3a68cfe60f059f8859aa6e84967b4d9a3523e2636f71074b1151720be90
```

displayとtradeは別成果物にした。trade gateは未校正なので、display結果を自動的にtradeと呼び替えていない。

```text
group screen display picks:
0b784c8d472aee2ad64def2098ffaaeb7a4e162b6b294ddb07692592f16cbbca

union prune display picks:
aea2c793a256ef4b759166855d593a2eb5239108adf491ff8d40c5bacb31cd82

stability display picks:
202a79f00202e2092da354f3353a6cb5449692d47948254cb0e4b899457b74de

empty trade-picks marker:
1c2531a02ced07b36e9f63d729aff862d60c70c7f74a9b03efedebc18011a94a
```

### 次に検証する特徴

履歴に正確な08:58 snapshotがないため、実始値や日足終値で代用せず、次を前向き収集する。

```text
F0_market_0858:
  Nikkei/TOPIX先物、spread、USDJPY、米株先物、volatility

F1_indicative_gap:
  08:58予想gap、gap/ATR、市場調整後gap、過去gap反応とのinteraction

F2_auction_pts:
  成行imbalance、板depth、特別気配、予想寄り時刻、PTS価格/出来高

F3_exact_liquidity:
  売買代金、ADV20、Amihud、単元金額、注文額/ADV、実測slippage

F4_tdnet_quantitative:
  本業利益増加/時価総額、増配利回り、買付規模・日次圧力、
  希薄化率、M&A・受注規模
```

前向きfeature-developmentは120営業日かつ6暦月。そこでfeature、model、0～2件のOOF trade gateを固定し、その次の100営業日をsealed confirmationにする。途中損益で変更した場合は、それまでをburnしてconfirmationを最初からやり直す。

</details>

---

## Entry 007 — 研究プロトコルv0.6.1：v0.6独立監査と訂正

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-22 |
| 検証対象コミット | 本PRへ収録。Entry 006のresultとauditを独立再計算 |
| 親Entry | Entry 006 |
| 検証対象 | 12 recipeの損益、shared block resample、選抜判定、966個のprovenance hash、TDnet実効イベント露出 |
| 採用判断 | 追加特徴block 0個、production変更なしを維持 |
| 当時の運用判断 | retrospective development diagnostic only。実発注不可 |
| 主指標 | 最大計算差 `2.84e-14`、hash 966/966一致、漏洩違反0件 |
| 機械可読成果物 | `research/model_v06_feature_result_audit_supplement.json`<br>SHA-256 `43226162dbb194868ecbeb164843709bfe366ea020debbcae404569d71139a57` |

> **一行結論:** Entry 006の数値と最終判断は再現したが、`max-T`の呼称とTDnetの`event_days`の意味を訂正する。訂正後も生き残る追加blockは0個で変わらない。

<details>
<summary><strong>訂正内容と影響</strong></summary>

Entry 006の各値をdisplay-picks CSVから再計算し、12 recipe・1,926数値の最大差は`2.84e-14`、shared resampleの最大差は`4.44e-16`だった。`G3_session_dynamics`の一次通過とunion-pruneでの削除も再現した。

### 統計量名

Entry 006の`max-T`は標準誤差でstudentizeしていない。正確には`shared moving-block max-mean adjustment`である。これは呼称の訂正で、臨界値、下限、選抜結果に影響はない。今後、studentizeしない限り`max-T`とは呼ばない。

### TDnetの実効標本数

Entry 006の`event_days = 84`は市場全体にclean eventがあった日数で、選択銄柄の実効露出ではなかった。再集計では市場全体に84日・14,003 security-daysのeventがあった一方、top2でのevent露出は次のとおり。

| 指標 | `T0_clean_event` | `T1_event_structure` |
|---|---:|---:|
| top2表示枠 | 168 | 168 |
| event保有表示枠 | 20 | 23 |
| event露出があった独立日 | 18 | 17 |

「top2でevent露出が30独立日以上」を正しい疎イベントgateとするとT0/T1はどちらも不通過。元々return-uplift系の基準でも不通過なので、survivorと最終recipeは変わらない。今後は、event保有security-days、event保有独立日、top-k内のevent保有枠数、top-k内のevent露出独立日を併記する。

```text
historical追加特徴block:  0個
locked retrospective recipe:     G0_price_core
G0自体の損益優位性:        未実証
production変更:                なし
次の対象:                       F0～F4の前向きsnapshot
```

</details>


---

## Entry 008 — 研究プロトコルv0.7：個別誤差分析とOOF meta-gate

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-22 |
| 検証対象コミット | 本PRへ収録 |
| 親Entry | Entry 006・007 |
| 検証対象 | G0の日次top2誤差、4系統の寄り前meta-gate、各50%枠の発注／現金化 |
| 採用判断 | 4 gateすべて不採用。`locked_gate = null` |
| 当時の運用判断 | 毎日top2の表示は維持するが、自動発注は0件 |
| 主指標 | 選択期forced top2は往復20bp後 `-0.274360%/日`。各gateの調整片側80%下限はすべて負 |
| 機械可読成果物 | `research/model_v07_error_result.json`<br>SHA-256 `e81b2abcc65c1aca958a3219951d1b1c419f0588878edcd41a2de09148714943`<br>`research/model_v07_error_result_audit.json`<br>SHA-256 `9fa5c9fcfcf7ef98cf77655ee6824cd1f5f061368ab750e33d98d7d4aa66053b` |

> **一行結論:** 大損失・大利益を個別に調べ、スコア形状、モデル間合意、反復選定、downside予測を試したが、コスト後のプラス期待値は証明できなかった。

<details>
<summary><strong>誤差の個別調査、固定gate、未使用期間での確認</strong></summary>

### 誤差の実例

2024年7～10月のG0 top2から50行を抽出した。分類は重複を認める。

```text
catastrophic loss:       9
large win:               5
high-confidence loss:   27
repeat loss:             14
```

大きな外れは、2024-08-05 TDSE `-10.76%`、2024-07-25 サンケン電気 `-7.91%`、2024-10-17 リックソフト `-7.69%`、反対側は2024-10-31 アツギ `+9.39%`、2024-08-05 サトー商会 `+8.00%`だった。8月5～6日を除いても既存recipeはプラスにならず、急落日だけが原因ではない。

G0 scoreと実現始値→終値のSpearman相関は期間ごとに`-0.094`、`+0.013`、`-0.166`、`+0.007`で、ほぼ単調性がない。また複数期間でrank 2がrank 1を上回った。後から「rank 1を飛ばす」のではなく、表示top2を固定して発注gateを独立検証する設計に改めた。

### OOF meta-gate

各対象月のgateは前月末までのデータだけで学習。G0 top2の各50%枠を独立に発注または現金化し、rank 3へは置換しない。

| gate | 発注日 / 枠 | net20 | top 5日除外 | 共通block調整80%下限 | 判定 |
|---|---:|---:|---:|---:|---|
| forced top2 | 41 / 82 | -0.274360% | -0.825284% | — | 対照 |
| score geometry | 2 / 2 | +0.008372% | -0.004472% | -0.024327pt | 不合格 |
| family agreement | 23 / 25 | -0.120293% | -0.276633% | -0.152992pt | 不合格 |
| stability / repeat | 24 / 27 | -0.158191% | -0.306185% | -0.190890pt | 不合格 |
| downside utility | 12 / 12 | -0.012558% | -0.083226% | -0.045257pt | 不合格 |

geometryのプラスは2日・2枠だけで、最小標本と上位日除外に失敗した。よって`locked_gate = null`とし、登録済みの正式replayは実行していない。監査用の後付けcounterfactual replayも、各gateの全replay期間合算net20は`-0.0895～-0.1801%/日`であり、判断を変えない。

### 独立監査

```text
audit checks:                    23 / 23 passed
独立再学習・比較予測:       1,784行
最大予測差:                    0
最大metric差:                  2.22e-14
対象月outcome変更不変:         11 / 11月
```

制約として、runnerは1,784行の全gate予測値を個別artifactに保存せず、空のslice CSVにheaderがない。監査で再計算しているが、次回からは予測値もhash-bound artifactにする。

```text
protocol SHA-256: 31cb5512b197ce59f418bc4021af164d362a5be6c6ad06454f4508f2eec48ebd
result SHA-256:   e81b2abcc65c1aca958a3219951d1b1c419f0588878edcd41a2de09148714943
manifest SHA-256: 9b8d46fcd5b506d6603d5574383915e3c71b5345b7e98d8e2b3b5029ea13557f
```

</details>

---

## Entry 009 — 研究プロトコルv0.7.1：posthoc tail cash-vetoと非線形E0

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-22 |
| 検証対象コミット | 本PRへ収録 |
| 親Entry | Entry 008 |
| 検証対象 | 個別外れ値から固定した5 tail条件、単独veto、1/2/3条件veto、非線形特徴を自由係数で学習するE0 |
| 採用判断 | 自動発注・shadow markerとも採用0。`veto_any_1_of_5`は損失軽減の観察に限定 |
| 当時の運用判断 | 過去結果を見て作ったposthoc診断。production昇格はprotocol上禁止 |
| 主指標 | 182日replayで1条件以上vetoはnet20 `+0.005473%/日`だが、net40 `-0.084087%`、絶対80%下限 `-0.048318%` |
| 機械可読成果物 | `research/model_v07_tail_risk_result.json`<br>SHA-256 `8323febc5bf1e5fd737e619f8b9932d5b30088da9b309fa5d0aa2376ec74f6bc`<br>`research/model_v07_tail_risk_result.audit.json`<br>SHA-256 `1702045d33e3752e948f0ce9bf76365049496c5518b0a0da173a76b40a27d0b2` |

> **一行結論:** tailの単調回避は自由係数の特徴追加より有望だが、コスト・上位日依存・期間安定性を通過せず、新規データで確認する仮説に留まった。

<details>
<summary><strong>tail条件、cash-veto、E0の失敗機構</strong></summary>

次の条件はすべて対象日前までの日足だけで算出できる。

```text
session_range_ratio_5_20 >= 1.5
cc_vol_ratio_5_20          >= 1.7
xrank_close_momentum_60    <= -0.6
flat_oc_rate_20            >= 0.10
overnight_last             >= +1.7%
```

G0 top2の各50%枠を保持し、条件に触れた枠だけを現金化した。欠測時は仮のvetoを生成せず、rank 3で穴埋めしない。

| recipe | replay net20 | forced比 | 判定 |
|---|---:|---:|---|
| forced top2 | -0.279700% | — | 対照 |
| range expansion veto | -0.125516% | +0.154183pt | 絶対損益が負 |
| deep 60d loser veto | -0.064014% | +0.215686pt | 絶対損益が負 |
| any 1 of 5 | **+0.005473%** | **+0.285173pt** | 不合格 |
| any 2 of 5 | -0.006461% | +0.273238pt | 不合格 |

`any 1 of 5`は163/364枠、126/182日だけ発注し、対照より大きく損失を減らした。しかし往復40bpで`-0.084087%`、上位5日除外で`-0.073142%`、9か月中プラス3か月、独立3 replay中プラス1期間、replay Aでforced比`-0.024761pt`だった。そのため「期待値があるrule」ではなく`loss-mitigation observation`と記録する。

### E0が改善にならなかった理由

同じ5条件の超過量と件数を7個の非線形特徴としてG0 logitへ自由係数で追加した。

```text
E0 net20:                          -0.242410%
G0比:                              +0.037290pt
paired 80%下限:                  -0.196280pt
G0と同じ銘柄枠:                    33.24%
tail条件数平均 G0 / E0:             1.057 / 2.113
2条件以上の比率 G0 / E0:          30.24% / 73.73%
```

モデルはtail条件を「下値リスク」ではなく「上値機会」としても学習し、むしろ複数tailの銘柄を多く選ぶようになった。実際、大損失と大利益の両方が同じtail領域にある。今後試す場合は、自由符号の特徴追加ではなく、実行スコアと分離した単調downside headまたはcash-vetoにする。

独立監査は24/24項目に合格した。ただしprotocolのartifact一覧に`condition_slice`がないこと、E0のtail露出比率の分母がcomplete-caseであることはminor注記とする。

```text
protocol SHA-256: 3b76e0ec8824953146e50392f0a406b4b95238e23992bb4d0951b16daf049dc8
result SHA-256:   8323febc5bf1e5fd737e619f8b9932d5b30088da9b309fa5d0aa2376ec74f6bc
manifest SHA-256: 1b15515410128f6542e0cf5530ec95bb39d987872d93c33eb44e60571c0c9beb
```

</details>

---

## Entry 010 — 研究プロトコルv0.6.2：T1整列訂正とTDnet v0.7意味契約

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-22 |
| 検証対象コミット | 本PRへ収録 |
| 親Entry | Entry 006・007 |
| 検証対象 | T1の2列整列バグの訂正診断、TDnetのfresh/follow-up・経済family意味の再設計 |
| 採用判断 | T1追加は不採用を維持。新規`tdnet_v07_*` 26列は特徴候補に登録するが損益未検証 |
| 当時の運用判断 | production変更なし。既定v0.3もshadow表示のみ |
| 主指標 | 訂正T1 top2 net20 `-0.565582%/日`、G0比 `+0.028075pt`、調整80%下限 `-0.088563pt` |
| 機械可読成果物 | `research/model_v06_t1_correction_result.json`<br>SHA-256 `e6af141fbd5566a3b77ac603e0cb88920205ea8f37aa351e4849e270c6105595`<br>`research/model_v06_t1_correction_result_audit.json`<br>SHA-256 `695bb7f92af8c653f330d61c8c08feea300505eaf598cfdf5598fd9189acfde6`<br>`research/model_v07_tdnet_semantics_audit.json`<br>SHA-256 `1cd257170114cf200ed7e49182dba7c24454edf57a8f2b52761414315e805bbb` |

> **一行結論:** Entry 006・007のT1単独値は訂正するが、改善量は小さく絶対損益も下限も負で、「追加特徴なし・自動発注なし」の最終判断は変わらない。

<details>
<summary><strong>訂正範囲、再計算、TDnet v0.7候補の意味</strong></summary>

### Entry 006・007への訂正

bundleを作成後、`code / bundle_start`でsortする前の行indexでfamily countを保持し、sort後の別行へ付与していた。影響列は次の2つ。

```text
tdnet_clean_family_count_log1p
tdnet_clean_single_family
```

全panelでそれぞれ37,433行、28,405行、group screenのevent行でそれぞれ9,106/14,003行、7,033/14,003行が変わった。G0、T0、他のT1 11列、Entry 008・009のG0ベース診断に影響はない。

| recipe | top2 net20 | G0比 | プラス月 |
|---|---:|---:|---:|
| G0 | -0.593657% | — | 1/4 |
| Entry 006の旧T1 | -0.591744% | +0.001913pt | 1/4 |
| **訂正T1** | **-0.565582%** | **+0.028075pt** | **1/4** |

```text
訂正T1 - 旧T1:                    +0.026162pt
paired 80%下限:                     -0.034182pt
訂正T1 - G0 調整80%下限:          -0.088563pt
選定が変わった枠:                   13 / 168、9日
event保有枠 / 独立日:               23 / 18
```

Entry 007の監査は、凍結されたバグ入りpanelとresultを正確に再現したが、特徴生成前のfamily countと行keyの対応までは監査していなかった。本訂正はG0と旧T1の168枠・score・outcome・指標を差0で再現し、独立監査19/19項目に合格した。489数値の最大差は`2.13e-14`、block再計算差は`1.11e-16`だった。

overlayは66,702 unique `date/code`を持ち、旧event key 34,491/34,491を収録する。completeでkeyなしだけ0、incompleteはNaNとする。ただし登録protocolはoverlay自体のhashは結合する一方、生成元TDnet、builder、各開示時刻を結合していない。したがって数値監査は合格だが、過去08:58時点の生データPITを完全に証明する成果物ではない。

### TDnet v0.7の研究用意味契約

旧`tdnet_clean_*`は凍結成果物の再現のため列名と意味を変えず、新しい26列を`tdnet_v07_*`として追加した。

- 「開示文書を観測」と「freshな分類済み経済family」を分離
- 初回予想と予想修正を分離
- 株主配当、受取配当、グループ内配当、子会社配当を分離
- 自己株買い、自己株消却、エクイティ、M&Aをfresh / follow-upに分離
- M&Aを取得、売却、組織再編、内部再編に分離
- 経済family数はraw、`log1p`、singleを保持し、follow-up family数と分離
- completeな開示元でイベントなしなら0、incompleteまたは完全性不明はNaNに固定

登録した26列は次のとおり。

```text
tdnet_v07_observed_any
tdnet_v07_fresh_classified_economic_any
tdnet_v07_has_forecast_initial
tdnet_v07_has_forecast_revision
tdnet_v07_has_shareholder_dividend
tdnet_v07_has_received_dividend
tdnet_v07_has_intercompany_dividend
tdnet_v07_has_subsidiary_dividend
tdnet_v07_has_progress_stage
tdnet_v07_has_fresh_buyback
tdnet_v07_has_followup_buyback
tdnet_v07_has_fresh_equity
tdnet_v07_has_followup_equity
tdnet_v07_has_fresh_share_cancellation
tdnet_v07_has_followup_share_cancellation
tdnet_v07_has_fresh_ma
tdnet_v07_has_followup_ma
tdnet_v07_has_ma_acquisition
tdnet_v07_has_ma_divestiture
tdnet_v07_has_ma_reorganization
tdnet_v07_has_ma_internal_reorganization
tdnet_v07_economic_family_count
tdnet_v07_economic_family_count_log1p
tdnet_v07_single_economic_family
tdnet_v07_followup_family_count
tdnet_v07_followup_family_count_log1p
```

経過・結果タイトルが新規材料に入らないようfamily別にstage gateし、M&Aの自己株・買収防衛・固定資産・政策保有株などの誤検知を回帰fixtureにした。この26列は意味とPIT実装を整えた「候補」であり、損益の改善を証明した採用済み特徴ではない。

独立監査は42/42項目に合格した。対象TDnet sourceは2024-01-04～2025-04-09の461日、97,006文書で、386 target sessionへ割り当てた66,702 bundleを監査した。自己株公開買付けへの応募、子会社主体の自己株取得、市場買付結果、期間延長、補足説明、調査委員会などを個別に調べ、issuerの新規決定とfollow-upを分離した。

```text
bundle buyback fresh / follow-up:       1,712 / 6,090
fresh経済familyありbundle:             29,282
follow-up familyありbundle:             10,342
bundle出力のPIT違反 / 候補セル欠測:       0 / 0
fresh-follow-up重複・訂正economic・主体誤分類: 0
M&A subtype重複・既知false positive:           0
全63候補列dtype（旧37 + 新26）:           float32
input shuffle差:                           0
v0.7分類器を全反転しても旧37列:         bit-exact
```

complete no-eventは63列すべて0、`False`とnullable `NA`のpartial sourceは63列すべてNaNを再現した。incomplete行でもsource-max timestampはprovenanceとして残るが、候補63列には含まれず、モデル値として漏れない。

本監査は保存済みHTML・sidecar・実装をhash拘束し、公開時刻から08:58:59 cutoffへの遡及割当を検証する。ただし、各ページが当時のセッション前に同時取得されていたことまでは証明しない。実行にはgit管理外の`research/.cache/model_v05_tdnet`にある461 HTML + 461 sidecarが必要で、全922ファイルのhashはaudit JSONに収録した。

```bash
PYTHONPATH=src:. python research/audit_model_v07_tdnet_semantics.py --verify
```

```text
protocol SHA-256: 939fe5a330cb973f3a43bbf5d2302ed9495093075acbb8ed98624e85a0e2e4d6
result SHA-256:   e6af141fbd5566a3b77ac603e0cb88920205ea8f37aa351e4849e270c6105595
manifest SHA-256: 5d689a4cfad747dde4f9d0c9ab6d667ef220fc18bd74fd784fa9b242bc91ca70
picks SHA-256:    0c1f72492bb5fb653b4c1a7a689060133d5c6d46037dc3c282ff084dee10aea3

TDnet audit runner SHA-256:
455add37ea6efd5e109bbae158a5533f3a1935fae232a7ca8b6ebf866d38bc2d

TDnet audit result SHA-256:
1cd257170114cf200ed7e49182dba7c24454edf57a8f2b52761414315e805bbb

TDnet audit manifest SHA-256:
ce4e872e07c10ee305bcd8b2b2004d1af3d6921e9d3b79fb9c8fca8e581ee1e9
```

</details>

---

## Entry 011 — 研究プロトコルv0.8：目的変数のゼロベース再設計と売買可能性仮説

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-22 |
| 検証対象コミット | 本PRへ収録 |
| 親Entry | Entry 008～010 |
| 目的 | 毎日1～2件を寄り前に順位付けし、勝率ではなく始値→終値のコスト控除後損益率を最大化する |
| 検証対象 | target/model 17案、感度10案、売買可能性15案、固定役9案。うち新しい特徴仮説はFZ 6案とL2～L14の13案 |
| 採用判断 | `Ridge(alpha=1)`と同日リターン順位targetを固定shadowへ。平均役L4、頑健役L6。本番昇格なし |
| 主指標 | L4 net20 `+0.283192%/日`、net40 `+0.086575%/日`。L6 net20 `+0.259376%/日`、net40 `+0.068023%/日` |
| 機械可読成果物 | `research/model_v08_target_result.json`<br>SHA-256 `6ccef56a4bf15ae091914194a0952f157db6cb5771d359b69afad2b174bf4ebc`<br>`research/model_v08_target_result.manifest.json`<br>SHA-256 `cb1442899d39318dc01b4dd60d15c25a794644a07bc59a0e89b1e632763378cc` |

> **一行結論:** 上昇確率や生リターンより同日横断順位のRidgeが良かったが、追加特徴の改善は多重比較後に未証明であり、既知期間の探索を停止して2仕様だけを前向きshadowへ固定した。

<details>
<summary><strong>全target/model案、FZ仮説、売買可能性仮説、反証結果</strong></summary>

### 共通の時系列契約

```text
対象:              2024-07-01～2025-07-31、266営業日
学習:              各対象月より前だけ、月次expanding
表示:              score上位2件、50%ずつ
未約定:            現金、損益0
主コスト:          往復20bp
stress:            往復40bp、60bp
target候補の評価:   top2日次平均
```

同日順位targetは学習日ごとに次を計算する。対象月の実現値は学習にも特徴にも入れない。

```text
y(i, d) = 2 * percentile_rank_d(open_to_close_return_i) - 1
```

この式は有限の横断銘柄数`n`では最小値が`2/n - 1`、tieがなければ日次平均が`1/n`となり、厳密なゼロ中心ではない。既に評価した仕様との同一性を保つため式を変更せず固定した。

Ridgeのlossでは各学習日のweight合計を1にし、学習期間中央値＋欠測indicator、標準化、`Ridge`の順でfitした。再現対象のimputerとscaler自体は行等重みでfitされるため、pipeline全体が日付等重みという意味ではない。このtargetは市場全体が同方向へ動く日の共通成分を落とし、毎日の銘柄間順位へ学習目的を合わせる。

### 第1段階：17 target/model案

すべて同じ266日にtop2を出し、同じ20/40bpを控除した。

| candidate | net20 | net40 | best 5日除外net20 | 判断 |
|---|---:|---:|---:|---|
| expanding raked logit | -0.378844% | -0.578092% | -0.472190% | 棄却 |
| FZ1 OC信頼度t値 | -0.391641% | -0.590889% | -0.485232% | 棄却 |
| FZ2 直近OC surprise | -0.334027% | -0.533276% | -0.433458% | 棄却 |
| FZ3 OC符号合意 | -0.396137% | -0.595385% | -0.488341% | 棄却 |
| FZ4 horizon分散 | -0.524639% | -0.723135% | -0.629943% | 棄却 |
| FZ5 gap-day coupling | -0.403932% | -0.603557% | -0.493946% | 棄却 |
| FZ6 momentum符号合意 | -0.400441% | -0.599689% | -0.494201% | 棄却 |
| rolling-120 raked logit | -0.187181% | -0.384925% | -0.303879% | 棄却 |
| 日次中心化Ridge | -0.169960% | -0.366577% | -0.274121% | 棄却 |
| 日次中心化Huber | -0.064491% | -0.255844% | -0.185916% | 棄却 |
| **同日順位Ridge** | **+0.200939%** | **+0.003570%** | **+0.092685%** | 感度検証へ |
| 当日top 10% logit | -0.643271% | -0.842895% | -0.956868% | 棄却 |
| 当日top 5% logit | -0.657449% | -0.857449% | -0.972327% | 棄却 |
| top-bottom decile spread | +0.118617% | -0.075368% | -0.015005% | stress不合格 |
| 2-part期待リターン | -0.535504% | -0.734377% | -0.662936% | 棄却 |
| downside-penalized Ridge | -0.118518% | -0.315511% | -0.186480% | 棄却 |
| relevance-weighted Ridge | -0.093827% | -0.290068% | -0.178672% | 棄却 |

FZ1～FZ6はすべて検証済みで採用0。FZ2にはlogit対照比の正の点推定があったが、片側80%下限が負で、4 block中プラス2 blockだけだった。

### 第2段階：順位targetの感度10案

| candidate | net20 | 要点 |
|---|---:|---|
| expanding alpha 1 | **+0.255321%** | 首位、12/13月プラス |
| expanding alpha 10 | +0.200939% | 初期対照 |
| expanding alpha 100 | +0.105420% | 過剰縮小 |
| rolling 120 alpha 10 | +0.237649% | top 20日除外後`+0.007398%` |
| rolling 252 alpha 10 | +0.201886% | alpha 1未満 |
| fixed pre-July alpha 10 | -0.000682% | 時間適応を失い棄却 |
| winsorized rank alpha 10 | +0.208379% | 改善小 |
| rank Huber | +0.025572% | 棄却 |
| decile spread control | +0.118617% | 40bpで負 |
| rank/spread ensemble | +0.126592% | 単独alpha 1未満 |

alpha 1のG0は好成績だったが、銘柄7043だけで総net slot損益の26.87%を占め、best 20日除外では`-0.005433%`へ落ちた。この集中を理由に特徴・screenを追加検証した。

### 第3段階：売買可能性15案

L0・L1は対照、L2～L14がこの段階の新しい13仮説。G5の4 raw列はv0.6でも候補に含まれており、「順位targetとの組合せ」が新しいことを明示する。実出来高・売買代金・spread・単元の履歴はないため、名称にかかわらず実際の流動性を証明する列ではない。

| ID / 仮説 | net20 | net40 | best 20日除外 | shadow gate |
|---|---:|---:|---:|---|
| L0 alpha1 G0対照 | +0.255321% | +0.058705% | -0.005433% | 不合格 |
| L1 rolling120対照 | +0.237649% | +0.041408% | +0.007398% | 不合格 |
| L2 `no_trade_rate_20`追加 | +0.216457% | +0.023224% | -0.058340% | 不合格 |
| L3 `no_trade_rate_60`追加 | +0.181506% | -0.009096% | -0.037215% | 不合格 |
| **L4 `flat_oc_rate_20`追加** | **+0.283192%** | **+0.086575%** | **+0.013510%** | 平均役 |
| L5 `zero_range_rate_20`追加 | +0.260235% | +0.064370% | -0.006753% | 点検通過 |
| **L6 G5 4列追加** | **+0.259376%** | **+0.068023%** | **+0.017913%** | 頑健役 |
| L7 illiquidity max | +0.265601% | +0.068985% | -0.005215% | 不合格 |
| L8 no-trade × low-ATR | +0.238502% | +0.041885% | -0.023619% | 不合格 |
| L9 G0欠測率 | +0.255321% | +0.058705% | -0.005433% | no-op、棄却 |
| L10 no-trade上位10%除外 | +0.138367% | -0.059753% | -0.113008% | 棄却 |
| L11 flat上位25%除外 | +0.054522% | -0.143598% | -0.186488% | 棄却 |
| L12 zero-range上位10%除外 | +0.091552% | -0.106945% | -0.140620% | 棄却 |
| L13 no-trade/flat複合 | +0.085484% | -0.105118% | -0.151340% | 棄却 |
| L14 rolling120 + G5 | +0.243898% | +0.054048% | -0.007331% | 点検通過 |

hard screenは候補数を絞っただけでなく損益も大きく落としたため、日足proxyで「流動性の悪そうな銘柄を一律除外する」案は棄却した。

### 固定役の反証

確認前にL4を平均収益役、L6を頑健性役へ固定し、役割の再選択はしなかった。

```text
L4: net20 +0.283192%、net40 +0.086575%、net60 -0.110041%
    勝率 57.17%、中央値net20 +0.180468%、11/13月プラス
L6: net20 +0.259376%、net40 +0.068023%、net60 -0.123331%
    勝率 56.78%、中央値net20 +0.196112%、12/13月プラス
```

15候補familyのmax-statistic補正では、G0比の改善は次のとおり。

```text
L4 point uplift:                 +0.027871pt
L4 調整片側80%下限:             -0.094078pt
L6 point uplift:                 +0.004055pt
L6 調整片側80%下限:             -0.117894pt
```

よって「`flat_oc_rate_20`が真に改善した」ことも、「G5全体が改善した」ことも証明できない。L4の係数符号は13月中11月で正、L6のG5係数も概ね正だった。これは低流動性を避けるpenaltyではなく、低流動性らしい銘柄をむしろ選ぶ可能性がある。

追加のcash fragilityでも20bpはプラスを保ったが、40bpでは上位4銘柄cash後に負へ落ちた。

| stress | L4 net20 / net40 | L6 net20 / net40 |
|---|---:|---:|
| 7043を現金化 | +0.213735% / +0.038547% | +0.202125% / +0.027689% |
| 上位4銘柄を現金化 | +0.112715% / -0.049691% | +0.122052% / -0.036219% |
| `no_trade_rate_20 > 0`を現金化 | +0.197947% / +0.046067% | +0.136851% / +0.018806% |

誤差CSVはG0・L4・L6ごとにbest day、worst day、高score損失を各20行、計180行保存した。L4でもKYCOM `-8.06%`、EM-Net Japan `-8.12%`、Qualtec `-6.80%`などの大損失が残り、単一のproxyで損失理由を分離できなかった。

### 再現・停止判断

```text
calendar-month folds:                        13
G0/L4/L6のfit:                               39
各仕様のscheduled slots:                    532
strictly-prior source違反:                    0
対象月outcome変更時の最大score差:             0
対象月outcome変更後のtop2 key一致:          true
独立PnL再計算差:                              0
```

```bash
PYTHONPATH=src:. python research/evaluate_model_v08_target.py \
  --panel /tmp/model_v07_corrected_panel.pkl \
  --output /tmp/model_v08_target_reproduction.json \
  --picks-output /tmp/model_v08_target_final_picks.csv \
  --error-output /tmp/model_v08_target_error_cases.csv
```

全ての過去outcomeは既に見ており、ここから同じ266日へ別の式を追加しても選択バイアスだけが増える。L4・L6を2026-07-23以降へ固定し、実出来高・売買代金・単元・spread・08:58先物・板・PTSの前向き保存が得られるまで、同じpanelでのtarget/model調整を停止する。

```text
protocol SHA-256:    14f660e59d43818a2a0150df6614c6665d017c5fc82ba4b9b0f037d08f47b12d
runner SHA-256:      d12ffd46e2ddbb6934f7e25e24617f24094133d3ee46bdd9eb2f626c96509a7e
result SHA-256:      6ccef56a4bf15ae091914194a0952f157db6cb5771d359b69afad2b174bf4ebc
manifest SHA-256:    cb1442899d39318dc01b4dd60d15c25a794644a07bc59a0e89b1e632763378cc
error cases SHA-256: 044fd3786ded321af52693eecb2369c1a1e1b584bd7f4cf09899a05c061025d0
```

</details>

---

## Entry 012 — 研究プロトコルv0.8：価格・市場状態12仮説とユニバース5仮説

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-22 |
| 検証対象コミット | 本PRへ収録 |
| 親Entry | Entry 011 |
| 目的 | 既存G0価格coreへ、寄り前に確定する価格履歴・市場状態・履歴品質を追加したとき、始値→終値のコスト後損益が改善するかを検証する |
| 新規仮説 | 特徴量12群（H01～H12）、候補集合5群（U01～U05） |
| 共通仕様 | 月次expanding walk-forward、同日順位Ridge、毎日top2、各50%、未約定は現金 |
| 採用判断 | 追加特徴0群、ユニバース変更0群。H07のalpha依存を確認して棄却。本番変更なし |
| 機械可読成果物 | `research/model_v08_feature_result.json`、`research/model_v08_feature_result.manifest.json` |

> **一行結論:** alpha 10ではH07だけが全期間の点検条件を満たしたが、固定したalpha 1へ移すとG0を下回った。12特徴群・5ユニバースのどれも多重比較調整後の事前gateを通過せず、追加採用は0とした。

<details>
<summary><strong>12特徴群、5ユニバース、alpha移送、PIT監査の全結果</strong></summary>

### 検証契約

```text
対象:                2024-07-01～2025-07-31、266営業日
学習:                対象月より前だけ、月次expanding
モデル:              Ridge(alpha=10)、同日リターン順位target
表示:                score上位2件、50%ずつ
主指標:              往復20bp控除後の日次top2平均
stress:              往復40bp、best 5日除外
discovery:           2024-07-01～2024-10-31
confirmation A:      2024-11-01～2025-03-31
confirmation B:      2025-04-01～2025-07-31
多重比較:            共通5日block max-statistic、片側80%下限
```

候補ごとにG0へ1群だけ追加し、同じfold・同じtop2枠で比較した。ユニバース仮説はG0モデル自体を変えず、順位付け前の候補集合だけを寄り前条件で制限した。`max-T下限`はdiscovery期間のG0比改善に対する多重比較調整後の片側80%下限で、全17案が負だった。

### H01～H12：新しい価格・市場状態仮説

| ID | 寄り前に計算する仮説 | net20 | net40 | best 5日除外 | max-T下限 | 判断 |
|---|---|---:|---:|---:|---:|---|
| H01 | G0 raw値の日次横断順位 | -0.018604% | -0.216349% | -0.124316% | -0.535099pt | 棄却 |
| H02 | 銘柄別の拡張履歴alphaを60観測へshrink | +0.070668% | -0.126325% | -0.025172% | -0.559008pt | 棄却 |
| H03 | 60日downside semideviation・2% tail非対称 | +0.130297% | -0.067072% | +0.027677% | -0.305155pt | stress不合格 |
| H04 | 1・5・20・60日の加速／減速 | +0.156987% | -0.038877% | +0.065232% | -0.208950pt | stress不合格 |
| H05 | 直前market breadth・tail・dispersion・volとのinteraction | +0.194286% | -0.003082% | +0.089411% | -0.126229pt | stress不合格 |
| H06 | 曜日・月末と銘柄状態、同曜日の縮小alpha | +0.133741% | -0.063251% | +0.030229% | -0.425651pt | stress不合格 |
| H07 | no-trade・flat・zero-rangeから作る連続activity品質 | **+0.239232%** | **+0.046375%** | **+0.145797%** | -0.191025pt | alpha移送へ |
| H08 | 前日終値をlot価格・microstructureのproxyにする | +0.072447% | -0.124921% | -0.003802% | -0.351058pt | 棄却 |
| H09 | 直前shock・range拡大・close位置による継続／枯渇 | +0.079376% | -0.115360% | -0.045137% | -0.413973pt | 棄却 |
| H10 | 過去gapの大きさと銘柄固有のgap反応・fill傾向 | +0.194330% | -0.003038% | +0.085731% | -0.192455pt | stress不合格 |
| H11 | momentum・ATRの横断crowdingとのinteraction | +0.175201% | -0.021040% | +0.075633% | -0.339790pt | stress不合格 |
| H12 | 履歴長・G0欠測率・feature ageによるsource品質 | +0.198508% | +0.001140% | +0.090208% | -0.263677pt | 改善未証明 |
| G0対照 | 価格core 15列 | +0.200939% | +0.003570% | +0.092685% | — | 対照 |

H07だけはalpha 10で全期間net20、net40、best 5日除外が正だった。しかしdiscoveryの調整済み下限は負で、特徴効果を証明していない。ほか11群はG0と同等以下か40bpで負だった。

### U01～U05：候補集合を変える仮説

| ID | 寄り前screen | net20 | net40 | best 5日除外 | max-T下限 | 判断 |
|---|---|---:|---:|---:|---:|---|
| U01 | flat20≦10%、no-trade60≦5%、zero-range20≦5% | +0.149701% | -0.048795% | +0.034423% | -0.226275pt | 棄却 |
| U02 | ATR横断順位を中央帯`[-0.8,+0.8]`へ限定 | +0.174507% | -0.020982% | +0.069041% | -0.263413pt | 棄却 |
| U03 | 前日終値200～5,000円 | +0.131598% | -0.066898% | +0.021311% | -0.410988pt | 棄却 |
| U04 | 始値→終値の先行履歴180観測以上 | +0.149015% | -0.004368% | +0.054336% | -0.372253pt | 棄却 |
| U05 | 60日内の−2%以下発生率5%以下 | +0.111953% | -0.085415% | -0.007114% | -0.203223pt | 棄却 |

5つすべてG0より悪化した。候補数を減らすhard screenは、見かけ上の低流動性やtail riskを避けても、同時に収益候補を落としていた。

### H07を固定alpha 1へ一度だけ移送

Entry 011でtarget/model側から固定した`Ridge(alpha=1)`へ、alpha 10で唯一残ったH07だけを移した。二つ目の特徴群は後から足していない。

| 期間 | alpha 1 G0 | alpha 1 G0+H07 | 改善 |
|---|---:|---:|---:|
| discovery | +0.318401% | +0.242339% | **-0.076062pt** |
| confirmation A | +0.229675% | +0.265109% | +0.035435pt |
| confirmation B | +0.222162% | +0.177679% | **-0.044483pt** |
| 全266日 | +0.255321% | +0.230309% | **-0.025012pt** |

全期間paired 5日block bootstrapの片側80%下限は`-0.084153pt`。3期間中2期間と全期間で悪化したので、H07は「汎用的な改善」ではなくalpha 10固有の順位変化と判断して棄却した。

### 実装監査で直した点

最初の研究実装ではH02・H03・H06が入力行順へ依存し得たため、正規の`code/date`順へ安定sortして計算し、整数位置で元の行順へ戻すよう修正した。非RangeIndexを含むshuffled入力との完全一致、将来outcome変更が過去特徴へ影響しないこと、重複`date/code`のfail-closed、前日終値joinの位置不変をテストした。安全版の再実行で上表の数値が変わらないことを確認した。

正確な出来高、売買代金、spread、売買単元、08:58板、PTS、先物snapshotはこの履歴にない。前日終値や日足の無約定率を、それらの代用品であるかのように解釈しない。先物特徴は2026-07-23以降に08:58:59以前の値を前向き保存してから、別protocolで検証する。

```text
protocol SHA-256:       0620d86113bfe6672104e6bfb4d68bc1eaeaf9de0f3e8b6408156fcd83a70a7f
runner SHA-256:         e1d2a0409b74de33202a7c41113abe9e5e3b3b869e51f6ed2e0bf7022e1ba6d5
result SHA-256:         824e414fe8cee3e1032cf85f7aad4a7153d203765df4140b5a230e144e0bc402
manifest SHA-256:       4eacc25080f4bfffc657322282a5c1a40a7f2b2714c93d331c84fecd1d37f641
safe metrics SHA-256:   987c43562d8205d90eb7cf7af3d58cdfa5aa8aef493d56aefee4734b624b4548
```

</details>

---

## Entry 013 — 研究プロトコルv0.8：TDnet 12仮説とゼロベース探索の停止判断

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-22 |
| 検証対象コミット | 本PRへ収録 |
| 親Entry | Entry 011～012 |
| 目的 | TDnetの材料family・fresh/follow-up・到来時刻・開示密度・発表前run-upを使い、価格coreに対する追加損益を検証する |
| 新規仮説 | TDnet 12群（H01～H12） |
| 共通仕様 | source-complete日のみ、月次expanding、同日順位Ridge、毎日top2、20/40bp |
| 採用判断 | alpha 10で0/12群、alpha 1の固定3 recipeで0/3。title-only探索を停止。本番変更なし |
| データ制約 | 完全取得187/266営業日（70.30%）、最終完全日2025-04-07 |
| 機械可読成果物 | `research/model_v08_tdnet_result.json`、`research/model_v08_tdnet_result.manifest.json` |

> **一行結論:** fresh/follow-up、到来時刻、開示bundle、発表前run-upを含む12群は、G0を安定して上回らなかった。TDnetを捨てるのではなく、タイトルだけの追加探索を止め、完全な前向き収集とPDF本文の金額・修正率へ次の仮説空間を移す。

<details>
<summary><strong>TDnet 12群、alpha 1再確認、全43仮説の最終判断</strong></summary>

### データと比較可能範囲

```text
要求した評価日:           266日（2024-07-01～2025-07-31）
TDnet source-complete:    187日（70.3008%）
比較可能期間:             2024-07-01～2025-04-07
文書:                     97,006件
不完全日:                 eventなしの0へ変換せず、比較から除外
```

取得完全性98%を昇格条件にしたため、損益が良くてもこのデータだけでは本番へ昇格できない。全recipeは共通の187日だけで比較した。

各世代で結果を見る前に固定したscratch仕様も、消える`/tmp`参照だけにせずbyte-exactで保存した。

| 登録 | 安定保存先 | SHA-256 |
|---|---|---|
| generation 1 executable contract | `research/model_v08_tdnet_registration_gen1.py` | `719f39a78a5b8bce4530ad49b05aba2189c9c4b1c2e6e9caef1bde598a7ede1c` |
| generation 2 protocol | `research/model_v08_tdnet_registration_gen2.json` | `de7850005b1ddfc1f98a6a3cb3e69eef1242b39f0589f1a1bf2fad9d99a45b39` |
| generation 3 protocol | `research/model_v08_tdnet_registration_gen3.json` | `8071e5f574028ad30853dbca4e986875b9d9958fa65fe20556f3b0f4c86b9b1e` |
| generation 4 protocol | `research/model_v08_tdnet_registration_gen4.json` | `3e97c5795c43ace38adb99054b4b3300bc8104c4f8bf3f948b214436ae25c4ac` |

### alpha 10で検証した12仮説

| ID | TDnet仮説 | net20 | net40 | best 5日除外 | G0比 | 判断 |
|---|---|---:|---:|---:|---:|---|
| H01 | fresh経済材料、follow-up数、進捗stageを分離 | +0.083593% | -0.114802% | -0.048394% | -0.050607pt | 棄却 |
| H02 | 初回予想と予想修正を分離 | +0.113338% | -0.085058% | -0.025681% | -0.020863pt | 棄却 |
| H03 | 株主配当・受取配当・グループ内配当を分離 | +0.127989% | -0.070407% | -0.014925% | -0.006212pt | 棄却 |
| H04 | 自己株・equity・消却のfresh/follow-up | +0.112833% | -0.085562% | -0.030496% | -0.021367pt | 棄却 |
| H05 | M&Aの取得・売却・再編方向とstage | +0.108427% | -0.089969% | -0.026070% | -0.025773pt | 棄却 |
| H06 | 同日経済family数と単独／複合材料 | +0.089011% | -0.109385% | -0.044799% | -0.045190pt | 棄却 |
| H07 | freshだけ・follow-upだけ・同時発生 | +0.103009% | -0.095386% | -0.034747% | -0.031191pt | 棄却 |
| H08 | 寄り前・場中・引け後、材料age、週末経過 | **+0.144128%** | -0.054268% | -0.000460% | **+0.009928pt** | stress不合格 |
| H09 | 文書数・bundle幅・過去60/252日開示頻度 | +0.077972% | -0.120423% | -0.056141% | -0.056228pt | 棄却 |
| H10 | fresh/follow-upと発表前20/60日momentum | +0.141531% | -0.056330% | -0.002225% | +0.007331pt | stress不合格 |
| H11 | fresh材料と発表前20日上昇／下落hinge | +0.098124% | -0.100272% | -0.035435% | -0.036076pt | 棄却 |
| H12 | stageと寄り前／引け後／ageのinteraction | +0.118368% | -0.080027% | -0.024809% | -0.015832pt | 棄却 |
| G0対照 | 価格core 15列 | +0.134200% | -0.064195% | -0.008542% | — | 対照 |

H08とH10だけはnet20の点推定がG0より僅かに良かったが、40bpとbest 5日除外で負だった。登録した12群のmax-statistic調整済み片側80%下限もすべて負で、正式合格は0だった。

### alpha 1へ固定して一度だけ再確認

H08単独、H10単独、H08+H10だけを事前固定し、Entry 011の`Ridge(alpha=1)`へ移した。

| recipe | net20 | net40 | best 5日除外 | alpha 1 G0比 | 判断 |
|---|---:|---:|---:|---:|---|
| alpha 1 G0 | +0.219063% | +0.021737% | +0.065287% | — | 対照 |
| G0+H08 | +0.150839% | -0.047022% | +0.006435% | -0.068224pt | 棄却 |
| G0+H10 | +0.213440% | +0.016114% | +0.058981% | -0.005623pt | 棄却 |
| G0+H08+H10 | +0.192753% | -0.005108% | +0.043716% | -0.026310pt | 棄却 |

alpha 10で見えた小さな改善はalpha 1へ移らなかった。閾値や組合せを追加探索せず、登録した停止規則を発動した。

### 最低10仮説という依頼に対する最終棚卸し

| 仮説family | 新規仮説数 | 全件結果を保存 | 採用 |
|---|---:|---|---:|
| 非線形価格変換FZ1～FZ6 | 6 | Entry 011・target result | 0 |
| 売買可能性L2～L14 | 13 | Entry 011・target result | L4/L6をshadow役のみ |
| 価格・市場状態H01～H12 | 12 | Entry 012・feature result | 0 |
| TDnet H01～H12 | 12 | 本Entry・TDnet result | 0 |
| **新しい特徴量仮説合計** | **43** | **43/43** | **本番0** |
| 候補ユニバースU01～U05 | 5 | Entry 012・feature result | 0 |

今回の履歴で最も高い暫定仕様はEntry 011のL4（net20 `+0.283192%/日`）で、L6は集中度が低い頑健性対照として残す。ただしL4/L6のG0比改善は多重比較調整後に未証明で、60bpでは両方マイナスである。よって毎日1～2件のshadow候補には使用しても、発注承認には使わずproductionも変更しない。

### これ以上同じ履歴を回さない理由と次の検証

12価格群、5ユニバース、12 TDnet群まで同じ266日の情報を使い切り、追加群の優位性は残らなかった。同じ期間へさらに閾値やinteractionを合わせることは改善ではなく選択バイアスになるため、ゼロベース探索をここで停止する。

次の改善余地は、未取得情報を前向きに作ることに限る。

```text
・2026-07-23以降の08:58:59以前の先物騰落率
・予想寄り、板imbalance、spread、PTS価格・出来高
・売買代金、出来高、単元金額と実約定slippage
・TDnet PDF本文の旧予想、新予想、増減額、時価総額比
・開示ごとのfresh decision、実施開始日、取得方法
```

これらは過去値を終値や9:00以後の値で代用せず、内容hashと観測時刻を保存し、outcomeを見る前に次protocolへ固定する。

```bash
PYTHONPATH=src:. python research/audit_model_v08_zero_base.py --verify
```

```text
TDnet protocol SHA-256:  6cca576b054de7abd2684137e11b2c9b01d489e1b62cb7ce780375868d0ce4a8
TDnet runner SHA-256:    03ee9ccede56ac8c8722e5d19f47bea9fc43af201ab6283f7f2040fdb64e5a63
TDnet result SHA-256:    a49fb6c14bd02855340b2c0eebc3a52dcec8d5ec7a0f23d4cfeef6eac18ebd8e
TDnet manifest SHA-256:  a10d75244825a8f185c8274c7a4b59b01f82e5cb539f01e0658e739d770dad9b
aggregate auditor SHA-256: 0b7315a8466c94422448dfdc217248c1069475e217914de123d24034a90ee71c
aggregate audit SHA-256: 0b7674a4681d44e1e4cbd315b17b04bbddf43421fd56459eb8c3b2ac52005c50
```

</details>

---

## Entry 014 — 研究プロトコルv0.9：個別誤差、順位深度、市場breadthの反証ループ

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-23 |
| 親Entry | Entry 011～013 |
| 目的 | 現行L4/L6の実行結果を個別に分析し、寄り前に確定する規則へ変換して、始値→終値のコスト後損益を再検証する |
| 入力期間 | 2024-07-01～2025-07-31、266営業日 |
| frozen panel | 1,524,104行、4,124銘柄、SHA-256 `6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb` |
| 今回の台帳 | 個別誤差/meta 20件、target・portfolio 33件、breadth反証4件、計57件 |
| 主コスト | 往復20bp。40bp必須stress、60bp追加stress |
| 暫定首位 | 前営業日の有効な始値→終値breadthでL4/L6のrank 2を切り替える1銘柄shadow |
| 採用判断 | production変更なし。全結果は既知期間のretrospective / posthoc診断 |
| 機械可読成果物 | `research/model_v09_protocol_ledger.json`、`research/model_v09_result.json`、`research/model_v09_manifest.json` |

> **一行結論:** 目的変数・正則化・ensembleより、L4/L6の2位と前営業日の始値→終値breadthの組合せが高かった。ただし2位優位は3位へ連続せず、利益の大半を少数の上昇日に依存し、breadthによるL4 rank 2への増分も未証明なので、仕様を固定した前向きshadowへ移す。

<details>
<summary><strong>57仮説、反証結果、暫定仕様、停止判断</strong></summary>

### 権限と検証順序

上流のv0.8ですでに266日のoutcomeを参照している。したがって、以下の`discovery`、`confirmation A/B`はsub-analysis内の時系列規律を表すだけで、未閲覧holdoutではない。

```text
discovery:       2024-07-01～2024-10-31、84日
confirmation A:  2024-11-01～2025-03-31、98日
confirmation B:  2025-04-01～2025-07-31、84日
```

検証は次の順で行った。

1. discoveryの個別誤差からH01～H19を固定し、A/Bへ変更なしで適用。
2. breadth switchとrank 2を見た後、統合規則を個別誤差系H20として別protocolへ固定。
3. target、前処理、配分のH01～H16を実行。
4. rank 1の弱さを見た後、rank 2/3と配分のH17～H21を固定。
5. 選択頻度の影響H22～H26、rank 2の仕様感度H27～H33を順次固定。
6. 最後にbreadthの分母とreturn horizonをF1/F1b/F2/F3で反証。

後の段階ほどadaptive / posthocである。名前が重なる二つのH20は、機械可読台帳では次のようにnamespaceを分離した。

```text
error_meta.H20_breadth_rank2_one
target_portfolio.H20_L4_rank3_only
```

### 現行対照と主要結果

すべてscheduled-dayの平均リターン。1銘柄規則は表示銘柄へstrategy sleeveの100%、2銘柄規則は明記した比率を配分する。1銘柄の結果を「2枠の片方50%、残り現金」と読み替えない。

| 規則 | 銘柄/日 | 配分 | net20 | net40 | net60 | best 20日除外net20 | 正の月 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v0.8 L4 rank 1+2 | 2 | 50/50 | +0.283192% | +0.086575% | -0.110041% | +0.013510% | 11/13 |
| v0.8 L6 rank 1+2 | 2 | 50/50 | +0.259376% | +0.068023% | -0.123331% | +0.017913% | 12/13 |
| L4 rank 1 | 1 | 100% | +0.141095% | -0.055897% | -0.252890% | -0.195468% | 10/13 |
| **L4 rank 2** | 1 | 100% | **+0.425289%** | **+0.229048%** | **+0.032808%** | +0.006616% | 11/13 |
| L4 rank 3 | 1 | 100% | -0.128573% | -0.325565% | -0.522558% | -0.383063% | 4/13 |
| L4 rank 1+2 | 2 | 25/75 | +0.354240% | +0.157812% | -0.038617% | **+0.025392%** | **13/13** |
| H03 OC breadth switch | 2 | 50/50 | +0.343128% | +0.147263% | -0.048602% | +0.062756% | 12/13 |
| posthoc H20・raw OC breadth rank 2 | 1 | 100% | +0.482449% | +0.285457% | +0.088464% | +0.050425% | 11/13 |
| **H20分母修正版・valid OC breadth rank 2** | 1 | 100% | **+0.487187%** | **+0.290195%** | **+0.093202%** | — | 11/13 |
| canonical CC breadth rank 2 | 1 | 100% | +0.310433% | +0.113441% | -0.083552% | — | 9/13 |

H20分母修正版の未使用ではない`confirmation A+B`では、net20 `+0.474899%`、net40 `+0.277097%`、net60 `+0.079295%`、約定銘柄の勝率60.56%、8/9か月プラスだった。best 20日除外後は`+0.012029%`、上位利益10コードを現金化すると`+0.025963%`まで薄くなる。

### target・前処理を変えても改善しなかった

H01～H09で次を検証した。

```text
厳密ゼロ中心の同日順位target
日次5/95% winsor後の順位
事前20日volでrisk-adjustした順位
boundedな実現値幅を混ぜた順位
日付等重みのimputer/scaler
alpha 1/10/100のrank ensemble
L4/L6のmodel-rank ensemble
```

厳密ゼロ中心と日付等重みpreprocessは、L4/L6とも532/532枠が既存仕様と完全一致した。winsor版L4の改善は`+0.000903pt/日`だけ。risk-adjust版はnet20 `+0.148613%`、実現値幅版は`-0.169830%`、alpha ensembleは`+0.190351%`へ悪化した。

したがって、有限銘柄数による小さなtarget offsetや前処理の行重みは現在の律速ではない。実現値幅を強めれば損益が上がるという仮説も棄却した。

### rank 2は滑らかなscore overshootではない

全額配分のnet20は、

```text
rank 1  +0.141095%
rank 2  +0.425289%
rank 3  -0.128573%
```

となった。rank 2とrank 3がともにrank 1を上回るなら「最大score付近だけが過熱」という説明が可能だったが、rank 3で直ちに損失へ反転した。よってrank 2は広い中位plateauではなく、同じ履歴に固有の局所的な順位反転として扱う。

rank 2は3つの等日数期間すべてでプラスだった一方、best 20日が累積net20利益の98.56%を占めた。20日除外後はnet20 `+0.006616%`、net40 `-0.189319%`、net60 `-0.385254%`。全H01～H33を一つのadaptive familyとして扱ったL4 top2比upliftの片側80%下限は`-0.071372pt`だった。

一方、rank 1/2を25/75にする規則は13/13か月プラスで、best 20日除外後もnet20 `+0.025392%`。平均はrank 2単独より低いが、分散shadowとして残す。

### 選択頻度ではrank 2を説明できない

直前20回で3回以上選ばれたrank 1をrank 2へ置換、前日連続選定の置換、20/60日novelty、rank 1/2の頻度tiltをH22～H26で検証した。単一銘柄のnet20は`+0.122761%`～`+0.244592%`で、rank 2単独を再現しなかった。

さらにwinsor target、alpha ensemble、L4/L6 ensemble、L6、alpha 10/100、rolling 120学習窓のrank 2をH27～H33で検証した。

```text
自身のtop2対照を上回る:       5/7
net40がプラス:                6/7
best 20日除外後net20がプラス: 0/7
3条件すべて合格:              0/7
```

rank 2の方向は複数仕様へ部分的に移ったが、tail-day依存は解消しなかった。

### breadthの意味を分解した

元H20のbreadthは、raw日足に存在した全行について`open_to_close > 0`の比率を計算していた。これにはflat行が非上昇として含まれ、2024年7月にはraw分母が一時的に約3,700から約3,100へ減る日もあった。

そこで結果を見る前に次の三定義を固定した。

| 定義 | return horizon | 分母 |
|---|---|---|
| raw OC | 始値→終値 | rawに存在する全行 |
| **valid OC** | 始値→終値 | `traded & outcome_observed & source_complete` |
| canonical CC | 前日終値→当日終値 | `traded & outcome_observed & source_complete` |

ルールはすべて同じ。

```text
前営業日のbreadth < 0.50  → L4 rank 2
前営業日のbreadth >= 0.50 → L6 rank 2
```

raw OCとvalid OCの相関は0.9994、状態が変わったのは5/266日で、`confirmation A+B`のnet20は`+0.4680%`から`+0.4749%`へほぼ不変だった。無効・非取引行を除く修正は、成績を見て選ぶ改善ではなく定義のhardeningとして採用する。

canonical CCとの相関は0.7055で、状態は63/266日、実際の選択銘柄は33/266日変わった。`confirmation A+B`のnet20は`+0.296389%`、net60は`-0.099216%`へ悪化した。したがって効いている可能性があるのは「市場breadth一般」ではなく、前営業日の**始値→終値の買い持続幅**である。

`.45/.50/.55`の感度は反証用途だけに固定した。raw OCは三つとも`confirmation A+B`のnet60がプラスだったが、同じ履歴の点推定で閾値を選び直さず`.50`を維持する。

### 独立監査で見つかった制約

- H03/H20の保存損益はpicksから誤差0で再計算でき、breadth sourceは全266日で対象日の直前営業日だった。
- locked picksと別系統raw日足を照合すると901行は最大誤差`1.78e-15`、131行はraw側にdate×code自体がなくprovenanceを照合できなかった。誤ラベルとは断定しないが、入力系統を一本化する。
- 個別銘柄のrolling特徴は直前営業日一致84.96%、最大28暦日staleだった。これは20営業日ではなくlast-20-observed-rowsになる場合がある。
- 保存された`max_t_*`はstudentized max-tではなく、未標準化のmax-mean統計量だった。再計算値は正しいが名称を訂正する。studentized感度の片側90%下限はH03 `-0.024531pt`、posthoc H20のL4 top2比で`-0.048538pt`。
- H20の適切な増分対照はL4 top2ではなくL4 rank 2。raw OC H20の増分は`+0.042881pt/日`、通常5日block bootstrapの片側90%下限は`-0.033398pt`で、breadth追加価値は未証明。

### 暫定shadow仕様

点推定首位を、閾値を再調整せず次のIDで固定する。

```text
v09_valid_oc_breadth_rank2

prior_market_oc_breadth =
    前営業日の有効・取引銘柄について
    mean(open_to_close_return > 0)

if prior_market_oc_breadth < 0.50:
    L4のscore順位2位を1銘柄
else:
    L6のscore順位2位を1銘柄

表示銘柄へstrategy sleeveの100%
候補数は毎日1件
```

並走する対照は三つ。

```text
C0: L4 rank 1+2、50/50
C1: L4 rank 2、100%
C2: L4 rank 1+2、25/75
```

この仕様は「最も高かったretrospective shadow」であり、実発注承認ではない。正確な出来高、売買代金、単元、spread、08:58板、PTS、先物snapshotが履歴にないため、0.49%/日の点推定を約定可能収益として扱わない。

### 前向き昇格条件と停止判断

2026-07-23以降を新しいforward counterとし、途中で特徴、rank、breadth閾値を変えない。変更した場合は別IDでゼロから数える。

```text
最低120 source-complete営業日、4か月
候補生成遵守率98%以上
実spread・slippage込み40bp stress後がプラス
非重複20日blockの4/5以上がプラス
best 20日除外後と上位利益10コード除外後がプラス
L4 rank 2に対するpaired差の調整済み下限が0以上
```

同じ266日へ新しいtarget、interaction、閾値を追加しても、真の改善と選択バイアスを区別できない。今回の57件で、既存日足だけを使う同一panel探索を停止する。次の改善余地は、08:58:59以前の先物、板、PTS、実流動性と、TDnet PDF本文の定量値を前向きに保存し、outcomeを見る前にprotocolへ固定することに限定する。

```bash
PYTHONPATH=src:. python research/audit_model_v09.py --root research
pytest -q tests/test_model_v09_artifacts.py tests/test_research_regimes.py
```

```text
protocol ledger SHA-256: 177f077e2ecb9130c4e95fe77f3614afe1ba8e79f985171089ccc8755f25c26d
result SHA-256:          faad02135eaf2a3fd3b7d0250919bbfd8abe5e11385fd67d0bf4c752be00c80d
manifest chain:          483511ed362a62516b632a4f78bab6a06809f7913a51baade8435311a1b49a81
```

</details>

---

## Entry 015 — 研究プロトコルv1.0：先行研究からのゼロベース再設計とTDnet text暫定首位

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-23 |
| 親Entry | Entry 011～014 |
| 目的 | 08:58:59 JSTまでの情報だけで、当日の始値→終値のコスト控除後平均損益を最大化する1～2銘柄を選ぶ |
| 入力期間 | 2024-07-01～2025-07-31、266 score営業日 |
| frozen panel | 1,524,104行、4,124銘柄、SHA-256 `6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb` |
| 今回の探索 | 一次資料19本・文献仮説12件、モデル構造13件、方策・universe 16件、TDnet text 10件、外部市場8件、online 3件、market/peer residual |
| 評価 | 月次expanding walk-forward、top1/top2、未約定枠は現金、往復20/40/60bp |
| 点推定首位 | `T02_char_value_event_only`、TDnetタイトルchar TF-IDF + value Ridge、event-only top1 |
| 最終判断 | production変更なし。T02を2026-07-24以降の固定exploratory shadowに限定 |
| 機械可読成果物 | `research/model_v10_summary.json`、`research/model_v10_integration_audit.json`、`research/model_v10_forward_protocol.json` |

> **一行結論:** 既存日足上でモデルを複雑化した案と、先行研究から移植した価格・注意力・順位・残差仮説は頑健性を改善しなかった。TDnetタイトルから値幅を直接予測するT02が88日で点推定首位になったが、右裾依存と多重探索を通過していないため、実発注せず仕様を固定した前向きshadowへ移す。

<details>
<summary><strong>先行研究、全track比較、監査訂正、前向き仕様</strong></summary>

### 権限と検証規律

上流のEntry 011～014ですでに同じ266日のoutcomeを参照している。今回の
月次walk-forwardは各fold内の未来混入を防ぐが、研究全体として未閲覧の
holdoutを作るものではない。したがって、

```text
・retrospective結果からproductionへ昇格しない
・各runnerの実行前にprotocolと仮説を保存する
・結果が良い案だけでなく失敗案も同じ成果物へ残す
・20/40/60bp、月別、時系列slice、tail、code集中を同時評価する
・同じ履歴でT02を再調整せず、次の観測をforward counterへ送る
```

を権限境界とした。

### 先行研究から移植した仮説

一次資料19本を調査し、寄り付き反転、注意力、発表混雑、発表曜日、
Learning-to-Rank、数値とテキストの組合せを12仮説へ変換した。主な
出発点は次のとおり。

| 先行研究の示唆 | 寄り前に確定する実装 | 結果 |
|---|---|---|
| TSEの寄り付き価格誤差は日中に修正され得る | H01 negative/positive gap hinge | top2 net20 `+0.1132%`、対照差`-0.1700pt` |
| 日本株の大幅下落後に反発パターンがある | H01/H02 negative shockと5日OC反転 | 両方とも対照未満 |
| overnightとintradayの投資家層には綱引きがある | H03 joint-sign rate | top2 net20 `+0.1498%`、対照差`-0.1334pt` |
| attentionは寄り付き過大反応を生み得る | H04 overnight×ATR×activity | top1のみ改善したがtail/code/FW不合格 |
| 同時発表の多さは情報処理を遅らせ得る | H06 TDnet市場混雑×方向 | TDnet対照より`-0.0527pt` |
| 金曜発表は注意を得にくい | H07 disclosure age/weekday×方向 | TDnet対照より`-0.0911pt` |
| 複数の同方向材料は単一材料を裏付ける | H08 directional corroboration | `+0.0129pt`改善、頑健性不合格 |
| 発表順序・announcement waveが反応を変える | H09 prior completed wave | 過去waveが0件でfeasibility failure |
| daily top選定にはLearning-to-Rankが適する | M01/M02/M06/M07 | 最良M02もnet40 `-0.0658%` |
| 日本語開示テキストは短期反応を補足する | T01～T10 | rank/overlayは失敗、T02 valueだけ点推定首位 |

文献の効果をそのまま仮定せず、今回の日本株標本で再現しなければ棄却した。
引用、識別子、仮説への対応、取得日は
`research/model_v10_literature_review.md`、機械可読な12仮説は
`research/model_v10_literature_hypotheses.json`に保存した。

### 全trackの比較

数値はscheduled dayの日次平均%。TDnet trackはstrict source-completeな
88日だけなので、266日trackと同一母集団の順位比較には使わない。

| Track | 代表候補 | 日数 | k | net20 | net40 | net60 | 判断 |
|---|---|---:|---:|---:|---:|---:|---|
| G0対照 | daily-rank Ridge | 266 | 2 | +0.2553 | +0.0587 | -0.1379 | best20除外`-0.0054` |
| 13モデル構造 | M02 extreme-gain Ridge | 266 | 2 | +0.1305 | -0.0658 | -0.2620 | 0/13合格 |
| 文献価格block | H04 attention | 266 | 2 | +0.1810 | -0.0163 | -0.2137 | 対照差負 |
| 方策・universe | H12 momentum分散 | 266 | 2 | +0.3044 | +0.1082 | -0.0881 | Round 2判断を監査で撤回 |
| gen1部分実行 | Z06 exp-decay top1 | 266 | 1 | +0.3085 | +0.1138 | -0.0810 | 18登録中6実行、選定不可 |
| 外部市場 | X06 context Ridge top1 | 266 | 1 | +0.2623 | +0.0645 | -0.1332 | 必須監査出力不足 |
| online expert | O03 follow-leader | 266 | 2 | +0.0823 | -0.1135 | -0.3094 | 棄却 |
| market residual | R01 | 266 | 1 | +0.0390 | -0.1595 | -0.3580 | 棄却 |
| peer residual | Z17 | 266 | 1 | -0.2964 | -0.4964 | -0.6964 | 0/13 net40月、棄却 |
| **TDnet text** | **T02 char-value event-only** | **88** | **1** | **+0.7191** | **+0.5191** | **+0.3191** | exploratory forwardのみ |

この比較から、rank loss、pairwise、quantile、mixture、utility/hurdle、
online experts、market/peer中立化を追加しても既存対照を頑健に上回らない
ことを確認した。peer residualは8 peer群を各foldの過去情報だけで再構成
したが、top1/top2ともnet20から負で、13か月すべてnet40が負だった。

### T02の仕様と結果

T02は勝敗分類ではなく値幅の条件付き平均を予測し、right tailを含む
平均損益を目的にした。

```text
source:
    08:58:59 JSTまでに公開され、
    strict source-completeと判定できるTDnet開示

bundle:
    同一銘柄の対象タイトルを公開時刻順にseparator付きで連結

representation:
    TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=3,
        max_features=30000,
        sublinear_tf=True,
        norm="l2",
    )

target:
    学習期間の1/99 percentileでclipし、
    さらに[-10,+10]%へclipした始値→終値リターン

model:
    Ridge(alpha=20)

decision:
    event銘柄を予測値降順、同点はコード昇順
    top1を1件、eventなしは現金
    価格モデルfallbackなし
```

88 source-complete営業日の実測は次のとおり。

| 指標 | T02 top1 |
|---|---:|
| net20 | `+0.719059%/日` |
| net40 | `+0.519059%/日` |
| net60 | `+0.319059%/日` |
| net40勝率 | `47.73%` |
| net40中央値 | `-0.188301%` |
| 正の月 | `3/5` |
| best 20勝ち日除外net20 | `-0.572088%/日` |
| 上位利益10code現金化net20 | `-0.108484%/日` |
| L4比paired 90%区間 | `[-0.0213pt, +1.0142pt]` |
| candidate-family reality-check | `p=0.2103` |

勝率が50%未満でも平均が正なのは、今回の目的が勝率ではなく平均損益で
あり、大きな上昇を少数捉えたためである。同時に、tail/code除外後が負に
なるため、同じ事実が脆弱性も示す。T10 top1はT02と完全に同じ選択であり、
独立した再現例として数えない。

### 独立監査による訂正

保存された全trackの損益を独立再計算し、最大誤差は`2.22e-16`だった。
損益式とは別に、次の手順上の問題を発見し、元の判断より監査判断を優先
した。

1. policy Round 2は「無関係な5比較中4勝」を要求したが、runnerは4比較
   しか実装せず4/4を合格にした。H12の
   `retain_for_forward_shadow=true`を撤回する。
2. zero-base gen1は18登録仮説のうち6件だけを実行した。Z06の数値から
   winnerを選べない。
3. external contextは月次vector、source-date mutation、familywise下限が
   未出力。X06はexploratory点推定以上に扱わない。
4. policy H01/H04、TDnet T02/T10 top1は完全重複し、独立試行ではない。
5. TDnet 88日は不連続な5か月で、historical HTMLに実観測時刻sidecarが
   ない。公開時刻PITは確認できてもarchive finalityは証明できない。

訂正後の権威成果物は
`research/model_v10_integration_audit_report.md`であり、
production置換は0件である。

### 08:58 point-in-time基盤

次の改善は同じ日足へのモデル追加ではなく、新しい寄り前情報を正しい
時刻で収集することとした。

```text
1. OSE先物の08:58騰落率、basis、08:45以降13分の方向・出来高
2. TSE寄り板の予想約定値、1/3/10本imbalance、成行差、spread
3. PTS価格・出来高・売買代金
4. 20日売買代金、出来高、単元金額、tick、実slippage
5. TDnet PDF本文の旧予想、新予想、増減額、時価総額比
6. 月次売上trend、決算数値とタイトル文脈の不一致
```

`src/tse_session_ranker/data/preopen_pit.py`は、取引所の
`source_event_at`、ローカルの`received_at`、特徴の`computed_at`を分離し、

```text
available_at = max(received_at, computed_at)
available_at <= target session 08:58:59 JST
```

を強制する。欠測、真の0、対象外、source不完全を別状態で保存し、同じ
identityに異なる値を追記することも拒否する。

### 固定した前向きshadow

`research/model_v10_forward_protocol.json`に、結果確認後の変更を禁止した
仕様を保存した。

```text
ID:                 v10_t02_char_value_event_top1
開始:               2026-07-24
候補:               原則TDnet event top1、eventなし/source欠落は0件
用途:               exploratory shadowのみ
実発注:             禁止
最低観測:           120 source-complete営業日、4か月
仕様変更:           forward counterをゼロへ戻し、別IDにする
```

昇格判断には、net40が前半・後半とも正、L4比paired片側90%下限が0以上、
best 20日除外後と上位利益10code現金化後のnet20が正、候補生成/PIT遵守率
98%以上、実spread・slippage・最低単元・売買代金の合格をすべて要求する。
forward中にn-gram、Ridge alpha、clip、fallback、閾値を変更しない。

### 再現と成果物

```bash
PYTHONPATH=src:. python research/audit_model_v09.py --root research
PYTHONPATH=src:. python research/model_v10_architectures_audit.py
python -m compileall -q src research tests
python -m pytest -q
```

主要成果物：

```text
research/model_v10_literature_review.md
research/model_v10_literature_hypotheses.json
research/model_v10_literature_protocol.json
research/model_v10_literature_result.json
research/model_v10_literature_validation_report.md
research/model_v10_tdnet_text_protocol.json
research/model_v10_tdnet_text_result.json
research/model_v10_tdnet_text_audit.json
research/model_v10_peer_residual_protocol.json
research/model_v10_peer_residual_result.json
research/model_v10_peer_residual_audit.json
research/model_v10_integration_audit.json
research/model_v10_summary.json
research/model_v10_forward_protocol.json
```

### 最終判断

```text
production変更:                 なし
retrospective点推定首位:       T02 char-value event-only top1
用途:                           仕様固定の前向きexploratory shadow
毎日の候補数:                   TDnet eventがあれば1件、なければ0件
同じ88日での再調整:            停止
次の改善単位:                   08:58板・先物・実流動性・TDnet本文
```

これは改善を放棄する判断ではない。同じ結果を見ながら閾値を変えるループを
止め、情報量を増やした未使用期間で、事前固定した仕様同士を比較できる状態へ
移した判断である。

</details>

---

## Entry 016 — 研究プロトコルv1.1：7系統ゼロベース反証と本番readinessのfail-closed統合

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-23 |
| 親Entry | Entry 015 |
| 目的 | T02の微調整ではなく、情報源・推定対象・意思決定構造が異なる7系統をゼロベースで反証し、本番採用の証拠権限を統合監査する |
| 7系統 | new data、historical analog、distributional decision、uplift、cross-stock graph、distribution shift、calendar/institution |
| 探索規模 | 概念仮説67件、実行可能spec 59件、capacity別候補variant 118件 |
| 選択の独立性 | family内の経済的に異なるselection sequenceは112件。comparator 14件を含むscored seriesは132件 |
| 評価窓 | strict-source 88日、new-data 107日、price-panel 266日が混在 |
| family gate | 通過0/118、forward-shadow finalist 0 |
| 最終判断 | 現行freeze/dataの下で本番採用を支持する証拠は0件。production変更なし、orders不許可 |
| 機械可読成果物 | `research/model_v11_integration_audit.json`、`research/model_v11_production_readiness.json`、各`model_v11_*_{protocol,result,audit}.json` |

> **一行結論:** 67の概念仮説を7つの異なる機構familyへ分け、59 spec・118 candidate variantを事前固定して検証したが、family固有の全gateを通過したvariantは0件だった。窓・universe・cash denominator・controlが異なるためfamily間の点推定順位は作らず、共有済み履歴panel上のfamily-local多重性補正も横断選抜の権限には使わない。凍結T02のfresh OOT rawと08:58実行データも未充足なので、本番候補は0のままとする。

<details>
<summary><strong>7系統の結果、統合監査、fresh OOT/readiness</strong></summary>

### 権限と数え方

7系統はいずれも各runnerの実行前にprotocolを固定し、月次expanding
walk-forward、対象月outcome mutation、独立P&L再計算を行った。ただし
全系統がprojectで既に参照した同一panel
`6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb`
を再利用している。family内の未来混入を防いでも、project-levelのfresh
holdoutには戻らない。

```text
概念仮説:                         67
実行可能な概念仮説:               58
固定combination spec:              1
実行可能spec合計:                 59
capacity別candidate variant:      118
family内unique selection sequence: 112
comparator variant:                14
scored series合計:                132
family gate通過:                 0/118
forward-shadow finalist:            0
production candidate:               0
```

118 variantはすべて経済的に別ではない。日付・銘柄・実行weightの
selection sequenceをfamily内でhashすると112件になる。
`D07_REGIME_ABSTAIN_K1/K2`、`U06_PROPENSITY_OVERLAP_T_K1/K2`、
`S06_UNSUPERVISED_LATENT_EXPERTS_K1/K2`、calendarの
`C11_RELEASE_CLOCK_FAMILY_EB_K1/K2`と
`C12_HIGH_DENSITY_FAMILY_EB_K1/K2`は、それぞれ全cashのため重複する。
非互換なfamily間は同じcash pathでも同一試行としてdeduplicateしない。

new-dataの凍結resultが列挙するblocked hypothesisは9件
（ND01、ND02、ND03、ND06、ND07、ND08、ND09、ND10、ND11）である。
取得routeではND01/ND02がstructured forecast numeric feedを共有するため
8 groupにまとめられるが、登録仮説数を8へ書き換えない。

### family間順位を作らない

数値は各family内の代表的な点推定であり、横断ランキングではない。

| family | representative | scheduled sessions | net40 | familywise evidence | 結論 |
|---|---|---:|---:|---|---|
| new data | `ND05_release_clock_and_fiscal_horizon__top1` | 107 | `-0.230428%` | L4差`-0.294670pt`、Holm p=`0.2974` | promotion gate自体を無効化、0件 |
| historical analog | `A06_dual_tail_neighbor_utility` top1 | 88 | `-0.048478%` | T02差`-0.567536pt`、simultaneous L90=`-1.220501pt` | 0/10 |
| distributional | `D08_SLOTWISE_CASH_STOP_K1` | 266 | `+0.118420%` | matched control差`+0.110518pt`、FW L95=`-0.134116pt` | tail・集中・sliceを含む全gate不合格 |
| uplift | `U04_DATE_RESIDUAL_UPLIFT_K2` | 88 | `+0.070633%` | FW L95=`-0.190559%`、global p=`0.9224` | 0/16 |
| graph | `G10_graph_disagreement_cash` top1 | 266 | `-0.176527%` | C00差`-0.184429pt`、simultaneous L90=`-0.486103pt` | 0/10 |
| distribution shift | `S08_ENVIRONMENT_RESIDUALISED_K1` | 266 | `+0.076608%` | matched control差`+0.068707pt`、FW L95=`-0.052917pt` | slice・tail・集中を含む全gate不合格 |
| calendar/institution | `C03_PRE_HOLIDAY_ISSUER_EB_K1` | 266 | `+0.058234%` | FW L95=`-0.250769%`、global p=`0.8376` | 0/24 |

calendarのC03はnet20/net40/net60が
`+0.100339/+0.058234/+0.016129%`だったが、取引は56/266日、unique codeは
9だけだった。confirmation Aはnet40 `-0.164160%`、best 20日除外net20は
`-0.211241%`、上位利益10 code現金化net20は`-0.128115%`である。
calendar全24 policyのfamilywise現実性検定もglobal p=`0.837632`で、
forward finalistはない。

88日、107日、266日は同じ母集団ではない。さらにsource-completeness、
候補universe、cashを含むscheduled-day denominator、capacity、control、
familywise手法が異なる。したがって、

```text
cross-family ranking permitted: false
common-window posthoc reranking: false
global 118-variant multiplicity correction: 未実施
selection authority from family-local correction: なし
```

とする。各familyの補正はfamily内の反証には使えるが、7 familyを見た後の
winner選択を補正しない。窓が非互換なまま横断p値を作る代わりに、
retrospective panelから候補を選ばないことを保守的な措置とした。

### canonical decisionの優先順位

analog、graph、uplift、calendarのrunner resultは、独立監査前の
`pending_independent_audit`を意図的に保持している。最終判断には後発の
auditを使う。

| family | canonical decision source |
|---|---|
| new data | independent audit + hash-binding manifest |
| analog | independent audit |
| distributional | independent auditを埋め込んだfinalized result + standalone audit |
| uplift | independent audit |
| graph | independent audit |
| shift | independent auditを埋め込んだfinalized result + standalone audit |
| calendar | independent audit |

runner resultの`pending`表示だけを読んで、auditの0 passersを上書きしては
ならない。7 familyすべてでprotocol hash、result binding、target mutation、
独立P&L、production falseを統合auditが再確認した。

### PIT・archive finality・calendar provenance

new-data/analogの履歴TDnet cacheはpublication timestampを持つが、当時の
local receipt timestampとarchive finalityを証明するsidecarを持たない。
uplift/calendarのPIT PASSも、publication-time cutoffと
source-completeness filterを検証したという限定された意味であり、履歴時点で
同じarchive bytesを受領済みだったことの証明ではない。

calendar featureはfrozen panelのsession indexから決定論的に生成した。
official holiday/SQ calendar datasetはbindされておらず、C09のSQはproxyで
ある。これは実装の未来混入がないことと、制度calendarの公式provenanceが
あることを区別するための制限である。

### 凍結T02のfresh OOT

`research/model_v11_t02_oot_protocol.json`は、v1.0 T02のvectorizer、target、
Ridge alpha、universe、rank、fallback、cash ruleを変更せず、次の期間だけを
評価する。

```text
warmup:                        2025-08-01
score:                         2025-08-04 ... 2026-03-31
expected score sessions:       159
minimum source-complete:        120
minimum calendar months:          6
untouched_holdout_claim:       false
T02 result/audit:              なし
```

`untouched_holdout_claim=false`なのは、同じ日付のaggregate returnが無関係な
v0.4分析で既に参照されたためである。一方、T02仕様自体は2025-07-31までの
データで固定され、bound artifact 4/4のhashは一致した。

現在のworkspaceでは対象期間のJPX raw PDFはhash-exact `0/160`、TDnet日別
pageは`0/243`、joint provenance-complete score sessionは`0/120` minimumで
ある。したがってexact no-tuning OOT統計gateもexecution gateも実行できない。

統合auditのexact data blockerは次の6件。

```text
frozen T02 OOT raw data missing
08:58 execution data missing
08:58 futures data missing
08:58 orderbook data missing
08:58 PTS data missing
08:58 liquidity data missing
```

data-readiness verifier v3はGit管理rootとevidence registryへ明示登録した
source rootだけを探索し、protocolが要求するraw、provenance、実行/context
fieldが揃うかをfail-closedで点検した。過去auditに残る絶対pathから`/tmp`を
推測して走査しないため、別test runの一時fileで結果は変わらない。
20件のblocking requirementと、未取得のoptional research-context 3 fieldを
分離して記録する。blocker件数は重複し得るfailed requirement数であり、
独立した欠測dataset数ではない。

ファイル名・CSV/TSV headerの探索結果はdiagnostic candidateに限定し、
header-only fileはgate evidenceにならない。positive certificationには、
明示的に凍結したmanifest/content reader、非空のOOT行、PIT timestamp、
型・非欠測率、provenance/hash、joint-session coverageが必要である。
verifier自身はdata readinessだけを判定し、本番を認可しない。
`model_v11_data_evidence_registry.json`には、T02 protocol hash、parser-audit
hash、160取引日のdate-set digest、JPX/TDnet parser SHA、許可source host/path
を固定し、registry自体のSHAもverifierへ固定した。
TDnetの空pageは日付見出しだけでなくtable headerと前日・翌日linkを要求し、
page/meta双方を243日manifestへhash bindする。JPXは公式host、raw byte count、
parser version/SHAを照合したうえでcanonical parserにより各PDFを再parseし、
日付・必須列・row数・reject数をauditと照合する。runtime importのpath/SHAも
canonical parserへ一致させる。execution evidenceには正方向のregistry-bound
CSV validatorを実装し、T02 decision artifactと独立replay audit、approved
source、policy/simulator hashへ結合する。全order decisionを`filled`、
`cancelled_special_quote`、`cancelled_delayed_open`、
`cancelled_liquidity`、`unfilled`のいずれかで過不足なく被覆し、cash decisionは
ledgerへ混入させない。40約定・30約定日、08:58:00～08:58:59 PIT、bid/ask、
tick/lot、価格からのspread/slippage、予定注文額、実売買代金0.5%、
予想寄付売買代金5%、予定額と実約定額双方の日次合計intended capitalを
再計算する。自己申告されたreplay/policy auditは内部整合性までしか通さず、
dated security master・JPX履歴からのtick/lot/20日売買代金再導出と
provider-authenticated originがない限りexecution evidenceの`valid`をfalseにする。
このregistryはrepository hashで固定されるが外部署名ではないため、verifierは
引き続き本番を認可しない。

### 本番認可rule

本番候補には次をすべて要求する。

1. 該当familyの事前登録retrospective gateを全て通過
2. 横断選抜も事前固定した、genuinely laterなfresh OOT gateを通過
3. source receipt/finalityとPIT complianceを独立検証
4. 08:58 indicative/bid/askまたはorder simulation、realized spread/slippage、
   special quote、delayed open、turnover、tick/lot、予想寄付turnover、
   outcome完全被覆、日次capacityを検証
5. 独立P&L auditと明示的人手承認

現在は1～4が未充足なので、その積集合は空である。
先物、PTS、volumeは追加研究contextであり、現行execution gateの必須項目とは
数えない。bid/askとtickはspread/slippage・指値・丸めを再計算するため必須である。
また登録済みhistorical windowはgenuinely untouchedではないため、これを通過しても
事前固定したpaper-liveまたはさらに後年のholdoutを通過するまで本番認可しない。

```text
production candidate:    0
production model changed: false
orders allowed:          false
```

これは「edgeが存在しない」という結論ではない。正確な結論は
**「現行freezeと利用可能dataの下で、本番採用を支持する証拠がない」**である。
不足dataを取得した後も、既存結果へ閾値を合わせるのではなく、新しいprotocolと
fresh periodを登録して検証する。

### 再現と成果物

```bash
python research/model_v11_integration_audit.py
PYTHONPATH=src:. python research/model_v11_production_readiness.py
python -m unittest tests.test_model_v11_integration -v
python -m pytest -q
```

最終回帰結果：

```text
307 passed
166 subtests passed
```

主要成果物：

```text
research/model_v11_new_data_{protocol,result,audit}.json
research/model_v11_analog_{protocol,result,audit}.json
research/model_v11_distributional_{protocol,result,audit}.json
research/model_v11_uplift_{protocol,result,audit}.json
research/model_v11_graph_{protocol,result,audit}.json
research/model_v11_shift_{protocol,result,audit}.json
research/model_v11_calendar_{protocol,result,audit}.json
research/model_v11_integration_audit.json
research/model_v11_integration_report.md
research/model_v11_t02_oot_protocol.json
research/model_v11_data_evidence_registry.json
research/model_v11_production_readiness.json
research/model_v11_production_readiness_report.md
```

</details>

---

## Entry 017 — 研究プロトコルv1.2：T02 forward契約の再監査とPhase 0 runtime

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-27 |
| 親Entry | Entry 016 |
| 対象 | draft PR #4 `agent/harden-t02-forward-validation`、base `998e2c0a3f93fe751e85dddfb168344f7c32298e` |
| registration | `v12_t02_forward_protocol_v2_pr4_20260727` |
| registration SHA-256 | `544732bdcff81840453a4b336e55467f5d84555b61326af903ac59c4fdfb84f8` |
| 候補 | 凍結済み`v10_t02_char_value_event_top1`。特徴・Ridge・順位規則は変更なし |
| Phase 0 | synthetic fixture runtime 18 tests PASS。実市場損益は0観測 |
| activation | payloadなし、receiptなし、first counted sessionなし、counter 0 |
| 最終判断 | protocol/runtimeの内部契約はPhase 0 PASS。本番根拠は増えておらず、production変更なし・orders不許可 |

> **一行結論:** 前向き検証の実行可能性をゼロから再監査し、activation SHAの自己参照、TDnet live completenessと翌日finalityの混同、月初foldと月末prediction hashの循環、価格適格性mask漏れ、source欠落日を主分母から落とす選択バイアスを訂正した。PIT、月次fit、4 decision state、canonical hash、ledger chainを合成fixtureで実行できたが、activationも実市場観測も開始していないため、本番候補は引き続き0件である。

<details>
<summary><strong>再監査、protocol v2、Phase 0、残るblocker</strong></summary>

### 追記専用履歴

Entry 016までの先頭150,920 bytesは変更していない。

```text
Entry 016 prefix SHA-256:
33a3e107942dc5b0f128b16aee13cfd318a895059b9600067cf3db7c573ce4a8
```

回帰testはこのprefixをbyte単位で検証する。今後はEntry 017を書き換えず、
Entry 018以降へ追記する。

### clean-checkoutで発見した差

最初のclean cloneでは全test中1件だけ失敗した。判定値ではなく、
Entry 016のimmutable readiness artifactに、生成環境固有の
`build/lib/tse_session_ranker/data/jpx.py`がcanonical source pathと重複して
記録されていたためである。

```text
fresh matching_paths:
  src/tse_session_ranker/data/jpx.py

stored matching_paths:
  build/lib/tse_session_ranker/data/jpx.py
  src/tse_session_ranker/data/jpx.py
```

過去artifactは上書きしていない。環境依存の探索path一覧だけをcore比較から分離し、
canonical source pathがfresh/stored双方に存在すること、それ以外のJPX readiness、
blocker、raw件数、parser SHAが完全一致することを検証する。

### protocol v2の主要訂正

#### activation

同じfileへ、そのfileを含むcommit SHAを書くのは自己参照になるため廃止した。

```text
activation payload:
  protocol / code / config / training snapshot / source schema
  not_before_session / payload_sha256
  containing commit SHAは書かない

independent timestamp receipt:
  payload SHA
  payloadがdefault branchへ到達したcommit SHA
  ref観測時刻・tip SHA
  issuer・署名・earliest_calendar_eligible_session
```

Git author/committer dateはactivation時刻に使わない。
`first_counted_session`は`earliest_calendar_eligible_session`と一致させる。
foldまたはpipelineが間に合わなければactivationを無効にし、新しい
payload/receiptを作る。後の日を結果確認後に開始日へ置き換えない。

#### TDnet PIT

08:58:59までの候補生成には、cutoff前にrequest、receipt、parse、computeが完了し、
登録済みwatermarkを持つlive snapshotだけを使う。翌日以後のfinal archiveは
prefix、raw hash、parser hash、record identityの監査専用とする。不一致は
PIT violationとして追記し、元decisionやcounterを遡及変更しない。

#### monthly fold

未来の月内prediction hashを月初foldへ入れない。

```text
pre-score fold:
  ordered training identity / title / target / source payload
  price_training_eligible mask
  clip lower / upper
  source / parser / scoring / config / eligibility code SHA
  vocabulary / IDF / coefficient / intercept / environment versions

month closeout:
  sealed fold SHA / scheduled session set
  decision-ledger head / prediction SHA / outcome SHA
```

凍結runnerと同じく、学習は`price_training_eligible == true`、推論は
`price_eligible == true`に限定する。maskまたはcode/config hashを変える場合は
別candidate・別counterとする。

#### prospective評価分母

historical OOTのsource欠落日はsource-complete集合から除外する。一方、
prospective shadowは`first_counted_session`以後の全scheduled JPX sessionを
主分母に固定する。

```text
selected           実際の始値→終値、登録cost
cash_no_event      gross 0、cost 0
fail_closed_source gross 0、cost 0
fail_closed_model  gross 0、cost 0
```

source-complete subsetは診断だけに使う。未知の終値はpre-open decisionへ入れず、
大引け後のoutcome logをdecision SHAへlinkする。

### Phase 0 runtime

`src/tse_session_ranker/t02_forward.py`へ、外部取得・注文機能を持たないpureな
forward primitiveを追加した。

```text
cutoff:
  strict JST、request_started <= received <= computed <= 08:58:59

training:
  monthly expanding、target monthとfit時点未確定outcomeを除外
  char 2-5 TF-IDF、Ridge(alpha=20, solver=lsqr)

scoring:
  price_eligible event-codeのみ
  predicted value降順、同点code昇順、top1

decision:
  selected / cash_no_event / fail_closed_source / fail_closed_model
  protocol / activation payload / receipt / sessionをhash bind

ledger:
  decision ID一意、zero-based sequence
  previous_record_sha256 / record_sha256
  exact retryはidempotent、異payloadと時系列逆行はreject
```

hashは`project_canonical_json_v1`を使う。これはPython
`json.dumps(ensure_ascii=False, sort_keys=True, separators=(",", ":"),
allow_nan=False)`のUTF-8 bytesであり、RFC 8785準拠とは主張しない。

合成fixtureで、post-cutoff拒否、live/final prefix差、provenance mutation、
target-month outcome mutation不変、train/score mask、4状態、同点code昇順、
fold/decision再現、activation hash binding、ledger conflictを確認した。
Phase 0はedge・外部真正性・実行可能性の証拠ではなく、in-memory ledgerも
durable external append-only storeの代替ではない。

### 登録成果物とblocker

```text
research/model_v12_t02_forward_registration.json
SHA-256:
544732bdcff81840453a4b336e55467f5d84555b61326af903ac59c4fdfb84f8
```

registrationはprotocol、plan、runtime、tests、README、CI workflowをhash bindし、
synthetic-only、market observation 0、activation未開始を明示する。

```text
historical OOT:
  JPX official raw PDFs         0 / 160
  TDnet calendar pages          0 / 243
  joint provenance sessions     0 / 120 minimum

prospective:
  approved live cutoff source   なし
  activation payload / receipt  なし
  first counted session         なし

execution:
  registered 08:58 board/fill/capacity evidence なし
```

したがってhistorical OOT、prospective損益、execution/capacityは未評価である。

### 再現と最終状態

```bash
python -m json.tool research/model_v11_t02_forward_protocol_v2.json
python -m json.tool research/model_v12_t02_forward_registration.json
python -m compileall -q src research tests
python -m pytest -q
```

```text
342 passed
192 subtests passed

Phase 0 synthetic runtime:      PASS
historical OOT:                 BLOCKED_INPUT_MISSING
activation payload:             absent
activation receipt:             absent
first counted session:          none
forward counter:                 0
production candidate:           false
production model changed:       false
orders allowed:                  false
```

</details>


---

## Entry 018 — 研究プロトコルv1.3：OHLC symbolic context treeのゼロベース反証

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-27 |
| 親Entry | Entry 017 |
| 目的 | 連続値のtabular回帰をやめ、完了済みOHLCのsymbol列を逐次学習する可変長context treeを反証する |
| base protocol | `model_v13_symbolic_context_zero_base_20260727` |
| input revision | outcome未計算のpreflight failure後、公式JPX PDF 19本へinput-only v2 erratum |
| 候補 | absolute path、relative path、direction path、3者固定平均 |
| variant | 4 candidate × top1/top2 = 8 |
| score期間 | 2024-07-01～2025-07-31、266 scheduled session |
| 対照 | v0.8 G0 15特徴、同日順位`Ridge(alpha=1)`、同一universe・calendar・cost |
| 最良点推定 | `CT02_RELATIVE_PATH__top1`、net20 `+0.181176%/日`、net40 `-0.018824%/日` |
| gate | `0/8` |
| 最終判断 | 全件棄却。forward shadow追加なし、production変更なし、orders不許可 |

> **一行結論:** OHLCをabsolute/relative/direction token列へ変換し、1～6 sessionのsuffix別損益をscore後に日次更新する非回帰型context treeを事前登録して検証した。最良CT02 top1も40 bp後は負で、3期間、上位10日除外、利益上位10 code除外、familywise対照比較を通過せず、8 variantすべてを棄却した。

<details>
<summary><strong>事前登録、input erratum、結果、独立監査</strong></summary>

### 従来方式との非重複

今回のfeatureはrolling meanやmomentumの列を増やすものではない。
完了した各sessionを次のsymbolへ変換し、target日にはshift済みtokenだけを使う。

```text
absolute shape:
  open-to-close / overnight / range / close-locationを固定bin化
  maximum suffix depth = 3

relative shape:
  同じ4量を当日のtraded銘柄内tertileへ変換
  maximum suffix depth = 3

direction:
  open-to-close / overnightのcoarse direction
  maximum suffix depth = 6
```

モデルはRidge、logit、pairwise rank、tree ensemble、TDnetタイトル近傍検索ではない。
各contextのclipped returnをdate-equalで集計し、次の固定reliabilityで親suffixへ
backoffするprequential reward tableである。

```text
reliability =
  min(
    rows / (rows + 200),
    distinct_dates / (distinct_dates + 20)
  )

prediction =
  reliability * context_mean
  + (1 - reliability) * parent_prediction
```

各sessionは候補をscoreした後にだけupdateする。同日outcomeが同日predictionへ
入ることはなく、target/future OHLC mutation testもPASSした。

### 事前登録とpreflight failure

candidate outcomeを計算する前に、protocol、runner、合成fixture testをcommit
`71284929c462ab77f39d1f370aa7bdf4225ed612`へ固定した。

最初の7列pickleは既存の90%収録率検査で2024年7月の18 sessionを不完備と判定し、
feature/model scoring前に停止した。

```text
candidate scores computed:     0
candidate outcomes inspected:  false
coverage threshold relaxed:    false
```

failureを`model_v13_symbolic_context_execution_failure_001.json`へ保存し、
inputだけをv0.5でhash固定済みのJPX公式月次PDF 19本・parser v6へ変更した。
feature、bin、depth、backoff、candidate、capacity、cost、gate、score期間は
変更していない。

```text
base protocol SHA-256:
7762221b33781a9d976487e50c1f5483bfc7ec6cea0e2b615d19ed36cf0b9703

input erratum v2 SHA-256:
991ef4dfd20171d9be371d1b8d69d4074fe6536339bb7ff264f9708758c67853
```

### 公式JPX input

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
calendar SHA-256:
966f4a416d0929487d850b16be87c9c1cd69cd9f8c0447a3e76214a5715c489e
```

### 全variant結果

主判定costは40 bpである。paired L90は、8 variantのBonferroni補正後、
5日moving-block bootstrapによる同capacity C00との差の片側90%下限。

| variant | net20 | net40 | net60 | 正の月 | paired delta | familywise L90 | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| CT01 absolute top1 | -0.106171 | -0.306171 | -0.506171 | 2/13 | -0.314073 | -0.761986 | FAIL |
| CT01 absolute top2 | -0.152299 | -0.352299 | -0.552299 | 2/13 | -0.411003 | -0.725630 | FAIL |
| **CT02 relative top1** | **+0.181176** | **-0.018824** | **-0.218824** | **5/13** | **-0.026726** | **-0.435428** | **FAIL** |
| CT02 relative top2 | -0.009507 | -0.209507 | -0.409507 | 3/13 | -0.268212 | -0.562390 | FAIL |
| CT03 direction top1 | -0.450965 | -0.650213 | -0.849461 | 0/13 | -0.658115 | -1.194361 | FAIL |
| CT03 direction top2 | -0.334574 | -0.534198 | -0.733822 | 0/13 | -0.592902 | -0.944968 | FAIL |
| CT04 consensus top1 | -0.455736 | -0.654984 | -0.854232 | 0/13 | -0.662886 | -1.032674 | FAIL |
| CT04 consensus top2 | -0.362191 | -0.561815 | -0.761439 | 0/13 | -0.620520 | -0.928573 | FAIL |

同じinputから再学習したC00は次のとおり。

| control | net20 | net40 | net60 |
|---|---:|---:|---:|
| C00 top1 | +0.204894 | +0.007902 | -0.189091 |
| C00 top2 | +0.255321 | +0.058705 | -0.137912 |

C00 top2 net20はEntry 011の既知値
`+0.25532118075031435%/日`と小数点以下まで一致した。

### 最良CT02 top1のgate

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

集中度は低いため、単一銘柄への依存だけが失敗原因ではない。
confirmation Bだけが正で、低cost点推定も上位日・上位codeを除くと崩れた。
固定したsymbolic pathとcontext treeでは、安定した条件付き期待損益を示せなかった。

### 独立監査

runnerとは別のauditで次を再計算し、すべてPASSした。

```text
artifact binding:                       PASS
5 models x 266 sessions x 2 slots:      PASS
label / return sign:                    PASS
v0.8 G0 exact reproduction:             PASS
cost / subperiod / tail-code removal:   PASS
familywise bootstrap:                   PASS
gate decision:                          PASS
production / order boundary:            PASS
```

```text
result SHA-256:
1f1a47ce04c4f72e5c581195120741c45d3a9cf013801627ecd2c1a356fa33de

picks SHA-256:
3d47891da9a18469bce7a38ee4b74952485a6092f25a54e326817b88f8cb8715

audit SHA-256:
dc6cb31b1661cb398277b058629007a3635efd941044258bde9ebbc3d28cbb23
```

### 再現と最終状態

```bash
PYTHONPATH=src:. python research/model_v13_symbolic_context_runner.py \
  --jpx-directory research/.cache/model_v05_jpx

PYTHONPATH=src:. python research/model_v13_symbolic_context_audit.py

python -m pytest -q
```

```text
registered variants:       8
gate passers:              0
forward shadow candidate:  none
production candidate:      none
production model changed:  false
orders allowed:            false
```

同じ266日を見てbin、depth、backoff、consensus比率を再調整しない。
新しい試行は、別protocolと未観測期間または異なる情報源を必要とする。

</details>


---

## Entry 019 — 研究プロトコルv1.4：陽線・非陽線の特徴差から作る異機構モデルの反証

| 項目 | 内容 |
|---|---|
| 検証日 | 2026-07-27 |
| 親Entry | Entry 018 |
| Stage A | `close > open`と`close <= open`を、D-1以前の53特徴で比較 |
| discovery | 2024-07-01～2024-10-31、84 scheduled sessions、244,704行 |
| stable signal | 21 / 53 |
| Stage B候補 | OC DMD、session-component DMD、spectral MLP、complexity GaussianNB |
| variant | 4 candidate × top1/top2 = 8 |
| confirmation | 2024-11-01～2025-07-31、182 scheduled sessions |
| 対照 | v1.3と同一の`C00_DAILY_RANK_RIDGE` |
| 最良点推定 | `DMD01__top2`、net20 `-0.063783%/日`、net40 `-0.263783%/日` |
| gate | `0/8` |
| 最終判断 | 全件棄却。forward shadow追加なし、production変更なし、orders不許可 |

> **一行結論:** 53特徴を陽線・非陽線で日付等重み比較すると21特徴に小さいが安定した差が見えた。しかし、その差から事前固定した低rank動学、周波数MLP、複雑度生成分類の4仮説は、未使用confirmationの全8 variantで20 bp後から負となり、3期間・tail・銘柄・familywise対照の全条件を通過しなかった。

<details>
<summary><strong>特徴比較、仮説固定、confirmation、独立監査</strong></summary>

### 従来検証から変えた点

v1.1の7機構family・118 variant、v1.3のsymbolic context tree・8 variantは
いずれもgateを通過しなかった。今回は既存rankerのfeature追加やcontext treeの
parameter変更を行わず、最初に目的変数の2群を記述比較した。

```text
positive:     close > open
nonpositive:  close <= open

feature cutoff:
  target sessionのD-1以前に完了した公式sessionのみ

new representations:
  AM / lunch gap / PM decomposition       10
  32-session frequency-domain summaries   20
  path complexity summaries                8
  frozen G0                                15
  total                                    53
```

各featureをその日のeligible銘柄内percentile rankへ変換し、
`mean(rank | positive) - mean(rank | nonpositive)`を日ごとに計算した。
84日の等重み平均を主effectとし、53特徴全体のBenjamini-Hochberg q値、
欠損率、3固定sliceの符号を事前登録した。

```text
stable:
  q <= 0.05
  absolute mean daily rank gap >= 0.005
  missing rate <= 10%
  all 3 discovery slices have the aggregate sign
```

### Stage Aの比較結果

```text
rows:                 244,704
scheduled sessions:          84
close > open:           105,753  (43.2167%)
close <= open:          138,951  (56.7833%)
stable signals:              21 / 53
```

| group | stable / total | 主な方向 |
|---|---:|---|
| frozen G0 | 8 / 15 | OC 20～60日の継続は正、overnight 20～60日は負 |
| session decomposition | 6 / 10 | AM・rangeは正、lunch・前日PMは負 |
| spectrum | 3 / 20 | lunch低周波、overnight高周波・dominant frequencyは正 |
| path complexity | 4 / 8 | component entropy、AM/PM相関、OC符号遷移率は負 |

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

最大rank gapは約2.26 percentile point、最大絶対Cohen's dも約0.09であり、
効果は大きくない。`overnight_last`、`oc_mean_20`、`xrank_atr14_pct`、
`range_abs_oc_ratio_20`は、日付内rank gapと全行pooled raw mean差の符号が
逆転した。仮説方向は事前登録した日付内rank gapへ限定した。

### Stage A input erratumと公式PDF replay

Stage A protocolのparser名に表記ずれがあり、実行cache SHAも元protocolには
含まれていなかった。元protocol/resultを変更せず、input metadata限定の
append-only erratumを追加した。

```text
incorrect parser label:
  jpx-stock-prices-v6

canonical parser:
  jpx_daily_text_v6_special_quote_marker

panel cache SHA-256:
  abcc6de28217f721358278c17039a60e4542361a60e2b1ac929325ab1a97516f

input erratum SHA-256:
  656be1249bc8d4f9fa8ce2f22bb6ab3c14450a16cf15c7c86f3e6492cf77c90f
```

hash固定済みJPX公式PDF 19本から再parse・再構築したreplayは、
244,704行、84日、53特徴、21 stable signalを再現した。

```text
cache contrast canonical SHA-256:
  b09156d385da821fd450269d57e513d6c8911d9528ad048e4f2c4a4d18025ec5

official-PDF contrast canonical SHA-256:
  b09156d385da821fd450269d57e513d6c8911d9528ad048e4f2c4a4d18025ec5

exact contrast match: PASS
confirmation rows used by Stage A feature/contrast: 0
```

### Stage Bで固定した4仮説

| candidate | 仮説 | 固定実装 |
|---|---|---|
| `DMD01` | 全銘柄OC rank場の低rank発展に翌日情報がある | 直前60 state、exact DMD rank 8、月内operator固定 |
| `DMD02` | AM・lunch・PM rank場の共同回転が翌日OC順序を予測する | 3場stack、直前32 state、exact DMD rank 12 |
| `SP01` | stableなlunch/overnight周波数特徴が非線形に相互作用する | 3 rank入力、tanh MLP hidden 6、月次expanding |
| `GN01` | path complexityのクラス条件付き分布が異なる | 4 rank入力、GaussianNB、方向固定、月次expanding |

Stage A結果を見た後、feature、方向、DMD lookback/rank、classifier parameter、
4 candidate × top1/top2、cost、期間、12 gateを別protocolへ固定した。
protocol、runner、合成testをGitHub remote commit
`f03134b0ccd9b686259dcd0d09941e1349bd8370`へ登録した後にだけ、
confirmationを1回実行した。

```text
Stage B protocol SHA-256:
ddc635c986dcb56072beda4889bfbf319019db31dfe3f74eedec24e51d8276af

Stage B runner SHA-256:
affd37391f4e399a173f2180e52127e25c78a1a33c508141ec5c3654321f25fa
```

### Confirmation設計

```text
score:                  2024-11-01 .. 2025-07-31
scheduled sessions:     182
months:                   9
slices:                   3
candidate variants:       8
costs:                20 / 40 / 60 bp
primary cost:             40 bp
bootstrap:                 5-session moving block
resamples:            10,000
familywise confidence:    one-sided 90%, Bonferroni over 8
```

全条件を必須とした。

```text
net40 > 0
net60 > 0
3 confirmation slicesのnet40がすべて > 0
上位10日除外後net40 > 0
利益上位10 code現金化後net40 > 0
正の月 >= 6 / 9
same-capacity C00差のfamilywise L90 >= 0
unique code >= 100
最大code比率 <= 5%
top10 code比率 <= 25%
traded days >= 150
executed slot fraction >= 80%
```

### 全variant結果

| variant | net20 | net40 | net60 | 正の月 | familywise L90 vs C00 | gate |
|---|---:|---:|---:|---:|---:|---|
| `DMD01__top1` | -0.075290 | -0.275290 | -0.475290 | 2/9 | -0.521562 | FAIL |
| **`DMD01__top2`** | **-0.063783** | **-0.263783** | **-0.463783** | **1/9** | **-0.630936** | **FAIL** |
| `DMD02__top1` | -0.371816 | -0.571816 | -0.771816 | 0/9 | -0.940859 | FAIL |
| `DMD02__top2` | -0.350941 | -0.550941 | -0.750941 | 0/9 | -0.928263 | FAIL |
| `SP01__top1` | -0.113622 | -0.311424 | -0.509226 | 0/9 | -0.594375 | FAIL |
| `SP01__top2` | -0.185446 | -0.383797 | -0.582149 | 0/9 | -0.705908 | FAIL |
| `GN01__top1` | -0.133762 | -0.333762 | -0.533762 | 3/9 | -0.692147 | FAIL |
| `GN01__top2` | -0.226527 | -0.426527 | -0.626527 | 1/9 | -0.746468 | FAIL |

8 variantすべてが20 bp後から負で、3固定sliceのnet40も全件・全sliceで負だった。
上位10日除外、利益上位10 code現金化、familywise C00比較も全件負である。
DMDは全182日を約定し、数値rank failureは0だったため、fail-closed cashが
成績を下げたわけではない。

`SP01__top1`はunique code 93、top10 code比率31.32%、
`GN01__top1`はunique code 84、最大code比率7.14%、top10比率33.52%であり、
経済成績に加えて集中条件も外した。

### 対照C00

| control | net20 | net40 | net60 |
|---|---:|---:|---:|
| C00 top1 | +0.073172 | -0.124631 | -0.322433 |
| C00 top2 | +0.226207 | +0.028955 | -0.168298 |

v1.4 C00のconfirmation 364行・全8列は、v1.3 picksの同期間と完全一致した。
全候補のnet40点推定は同capacity C00を下回った。

### 独立監査

confirmation runnerをimportしない別scriptで、picksからすべて再計算した。

```text
artifact binding:                         PASS
5 models x 182 sessions x 2 slots:        PASS
label / return sign:                      PASS
v1.3 C00 364 rows / 8 columns exact:      PASS
cost / slices / tail-code removal:        PASS
concentration / execution:                PASS
familywise bootstrap:                     PASS
12 gates x 8 variants:                    PASS
decision / production / order boundary:   PASS
mismatch:                                    0
```

```text
confirmation result SHA-256:
de6b5e27e07def3524361f129a89d2e58da916c19c6c8e7f49219eb27305cba2

confirmation picks SHA-256:
97aaeeb09d5590c8250a5c8989197744aa4e7c4844eeda6eef369b8dcdaf90ee

audit runner SHA-256:
33d514678ba6b100dd074bfd07bc0b291b8cb8ddbadf3fb6445319e1bc33321d

audit result SHA-256:
ed302c5369d00ec5c0464fc69a90f4ab6532bf0b134c40c1c6eb99ec54778f27
```

### 再現と最終状態

```bash
PYTHONPATH=src:. python research/model_v14_feature_contrast_runner.py \
  --panel-cache /tmp/tse_v14_panel.joblib

PYTHONPATH=src:. python research/model_v14_confirmation_runner.py \
  --panel-cache /tmp/tse_v14_panel.joblib

PYTHONPATH=src:. python research/model_v14_confirmation_audit.py

python -m pytest -q
```

```text
373 passed
192 subtests passed

registered variants:       8
gate passers:              0
forward shadow candidate:  none
production candidate:      none
production model changed:  false
orders allowed:            false
```

同じconfirmation期間でfeature方向、DMD rank/lookback、MLP/GNB parameterを
再調整しない。次の試行には新しい未観測期間または異なる情報源を必要とする。

</details>

---

## Entry 020 — 研究プロトコルv1.5：実データ境界と取得不能候補の棄却

| 項目 | 内容 |
|---|---|
| 実施日 | 2026-07-28 |
| 親Entry | Entry 019 |
| 基準コミット | `22c1cb9a9880e6e6af2562911a8d484b0effb3e4` |
| 目的 | 異機構案を「実データ取得を実証する」か「候補ごと棄却する」かに二分 |
| 実取得 | TDnet公式index 2ページ、165開示、業績予想修正PDF 9本、全2ページ安定性再取得 |
| 厳格抽出 | 1本成功、unsupported/ambiguous 8本はfail-closed |
| 市場評価 | 未実施。score 0、outcome参照0、gate未実行 |
| 最終判断 | Aはcollectorだけ実装。B・ABと他の入力不足案は棄却。新規model候補0 |
| production | 変更なし |
| orders | `false` |

> **一行結論:** 公式TDnet本文のfresh実取得と数値再計算は1つの対応table schemaで
> 成功したが、同日cutoff後のsource probeに過ぎない。08:58板・PTS・約定証拠は取得経路を実証できず、
> 代替値を作らずB/ABを候補ごと閉じた。性能がないという反証ではなく、実行可能な
> 入力がないという棄却である。

<details>
<summary><strong>実取得、抽出、候補境界</strong></summary>

### 「collect or reject」契約

v1.1ではTDnet本文/XBRL、08:58先物・板・PTS、execution証拠をblockedとして
登録していた。v1.5ではblockedのまま数式・synthetic testだけを追加する案を
採用しなかった。

```text
real raw + receipt + strict parse:
  collector implementation may proceed

missing entitlement / raw / receipt:
  whole approach = REJECTED_INPUT_UNAVAILABLE

forbidden:
  title or URL as body
  same-day open as indicative-open proxy
  daily futures OHLC as 08:58 proxy
  missing source as zero/no-event
```

### 公式TDnet live probe

新しい`probe-tdnet-material`を実際に公式current indexへ接続した。

```text
index page 1:
  requested  2026-07-28T16:51:32.330007+09:00
  received   2026-07-28T16:51:37.257144+09:00
  HTTP       200
  bytes      63,941
  SHA-256    7f6ecc8d464182defe3a2884234bcd73df800b07eb9c14b487042b518a66dbd3

index page 2:
  requested  2026-07-28T16:51:37.271596+09:00
  received   2026-07-28T16:51:42.428360+09:00
  HTTP       200
  bytes      43,979
  SHA-256    1050dcae2eaa2f6f9a96fe58f84bda055a74a3eeb9199a5a1fe5cd426756706e

Last-Modified header concordance:
  Tue, 28 Jul 2026 07:44:00 GMT

all-page stability re-request:
  page 001 requested  2026-07-28T16:52:34.676202+09:00
           received   2026-07-28T16:52:39.858895+09:00
           bytes      63,941
           SHA-256    7f6ecc8d464182defe3a2884234bcd73df800b07eb9c14b487042b518a66dbd3
  page 002 requested  2026-07-28T16:52:39.859655+09:00
           received   2026-07-28T16:52:47.533544+09:00
           bytes      43,979
           SHA-256    1050dcae2eaa2f6f9a96fe58f84bda055a74a3eeb9199a5a1fe5cd426756706e
  各URLでinitial responseとbyte/Last-Modified/parsed contentが一致
```

165開示のうちタイトルfilterに一致した9 PDFをbodyとして取得した。strict v1
parserは単一の通期・期待5列・単位/増減額/増減率を再計算可能なtable schemaだけを受理し、1本を抽出した。
sector固有、複数期間、表header不足など8本はタイトル値で補わず拒否した。

受理したテセック（6337）のPDF receipt:

```text
published  2026-07-28T15:30:00+09:00
requested  2026-07-28T16:51:47.965250+09:00
received   2026-07-28T16:51:54.715386+09:00
HTTP       200 application/pdf
bytes      77,140
SHA-256    b06e03d459f54da7fb84236cc0daa245de208fb53de583149528cfb8598cd772
```

| 百万円 | 前回 | 今回 | 公表増減率 |
|---|---:|---:|---:|
| 売上高 | 6,300 | 7,000 | +11.1% |
| 営業利益 | 450 | 1,100 | +144.4% |
| 経常利益 | 590 | 1,300 | +120.3% |
| 純利益 | 540 | 1,040 | +92.6% |

4増減額と4比率を金額から再計算し、比率は差0.11 percentage point以内を確認した。
Aの実装値は営業利益growthを符号付き30%でcapして`+0.30`となる。このcapは将来
protocol用に固定したが、source probe前の事前登録値ではない。

ただし公開・取得・計算はすべて2026-07-28の08:58:59後である。同日特徴には
不適格で、次営業日についてもsealed source-complete snapshotと営業日calendarを
まだ証明していない。抽出は`candidate_records`から物理的に隔離し、
scoreもoutcomeも作っていない。

JPX Listed Company Searchについても63370をlive確認し、過去PDF/XBRL linkへ
到達できた。公式仕様上の掲載期間は121か月であり、「公開履歴本文が一律取得不能」
という前提は撤回した。ただし全issuerのgap-free bulk export、当時cutoff receipt、
完全なsecurity-session joinは未取得であるため、履歴を使うmodel案は候補に残さない。

### armと他アプローチの決定

| arm | source | decision |
|---|---|---|
| C0 frozen T02 | 既存 | controlのみ、再実行なし |
| A quantitative material | current bodyを実取得 | collector実装、model候補なし |
| B 08:58 price absorption | 契約/rawなし | `REJECTED_INPUT_UNAVAILABLE` |
| AB interaction | Bが欠落 | `REJECTED_INPUT_UNAVAILABLE` |

D+1/D+3、auction path、execution-first、09:05/09:15 sequential、
exposure graph、event/issuer hierarchy、hedged portfolioも、各々に必要な
PIT raw joinをこのrunで実証できないため候補として棄却した。

### 再現

```bash
tse-session-ranker probe-tdnet-material \
  --index-url https://www.release.tdnet.info/inbs/I_list_001_YYYYMMDD.html \
  --destination var/tdnet-official \
  --manifest var/tdnet-material-probe.json

python -m pytest -q tests/test_tdnet_material.py \
  tests/test_model_v15_data_boundary_artifacts.py
```

```text
targeted tests:              23 passed
full tests:                 396 passed
subtests:                   192 passed
model scores:                0
market outcomes inspected:   0
new shadow candidates:       0
production changed:          false
orders allowed:              false
```

Decision artifact SHA-256:

```text
751f2e7c1828c80debd69d231fde4c55a89e2b21ce05addd89576a38a69b3538
```

Full live-probe manifest SHA-256:

```text
25faff2a287d761492c7af336e6eb30bd53e373f35d66436bcc862380cd6a870
```

</details>

---

## Entry 021 — 研究プロトコルv1.6：JPX実流動性3機構のretrospective反証

| 項目 | 内容 |
|---|---|
| 実施日 | 2026-07-28 |
| 親Entry | Entry 020 |
| 事前登録コミット | `917a82352c27996ad4853c226a05f8eccab90d8e` |
| authority | `retrospective_candidate_specific_selection`、`project_level_untouched=false` |
| 実入力 | 公式JPX daily PDF 160本、2025-08-01～2026-03-31 |
| parser | 622,724行、rejected row 0 |
| 実測channel | volume、turnover、VWAP、trading unit |
| selection | 2025-12-01～2026-03-31、80 scheduled sessions、panel source-coverage 78/80 |
| candidates | LQ01 activity veto、LQ02 exact-liquidity Ridge、LQ03 VWAP-flow reversal |
| variants | 3候補 × top1/top2 = 6 |
| gate | `0/6` |
| 最終判断 | 全件棄却、winnerなし、この実験によるcandidate-specific replay未開封 |
| production | 変更なし |
| orders | `false` |

> **一行結論:** 実JPX D-1 activity channelは取得・hash固定・PIT化できたが、
> 6 variantすべてでnet40平均とfamilywise対照下限が負だった。低損失に見える
> veto/reversalも2～17/80日の低稼働によるcash中心であり、forward候補にしない。

<details>
<summary><strong>実データ、固定候補、評価結果</strong></summary>

### 入力可用性の解消

Entry 020の`collect or reject`契約に従い、以前
`blocked_local_archive_ohlc_only`だった
`ND11_pit_liquidity_and_unit_cost`を再確認した。公式JPX日次相場表から、
出来高を実株数、売買代金を円、VWAPを円/株、売買単元を株数として取得できた。

```text
official daily PDFs:       160
date bounds:               2025-08-01 .. 2026-03-31
raw hashes vs parser audit: 160 / 160 exact
parsed rows:               622,724
rejected rows:             0
parser:                    jpx_daily_text_v6_special_quote_marker
price warm-up PDFs:        3 / 3 input-lock exact
panel:                     878,736 rows / 4,060 codes
liquidity-ready rows:      427,721
```

`actual_volume=true`などの記録は実測channelが存在することを表し、全行が完全という
意味ではない。銘柄ごとにD-1まで20営業日連続で正のvolume、turnover、VWAP、
trading unit、closeを要求し、期間中の売買単元変更もfail-closedにした。
欠測を0や価格proxyで補っていない。

通常約3,900行のところ、2025-09-29は1,563行、2025-12-29は3,161行、
2026-03-30は1,489行だったため、既存source coverage判定で不完備になった。
selection内の既存price-panel source-coverage判定は78/80 completeだが、
80予定日を損益分母から除いていない。不完備日自身はD-1情報でscoreを固定した。
欠けたtarget outcomeのslotだけをcash、gross 0、cost 0とし、直前universeが
不完備になる翌sessionはfail-closedで全slotをcashにした。

### 事前固定した3機構

| ID | 機構 | 固定仕様 |
|---|---|---|
| `LQ01_ACTIVITY_VETO` | hard activity veto | C00 top2を先に固定し、20日売買代金中央値1億円以上、出来高中央値5万株以上、最低単元比率1%以下だけを残す。下位置換なし |
| `LQ02_EXACT_LIQUIDITY_RIDGE` | supervised additive | G0価格15特徴へD-1実流動性7特徴の横断rankを追加し、return-rank Ridge `alpha=10` |
| `LQ03_VWAP_FLOW_REVERSAL` | deterministic mechanism | `-rank(close/VWAP-1) * (1+rank(turnover shock))/2`へLQ01と同じveto。学習なし、逆符号版なし |

すべて対象日のOHLC、volume、turnover、VWAP、trading unit、source finalityを
禁止した。月次expanding fitはscore月の前月末までのlabelだけを使い、
月中再学習はしていない。outcome列なしのscore ledgerを先に意味的hashへ固定し、
その後にdate/codeで始値→終値returnをjoinした。固定slotがveto・欠測ならcashで、
rank 3以降へ置換していない。

### selection gate

主判定は40bp、感応度は20/60bpとした。これは実spread/slippageではなく固定
haircutである。3候補 × top1/top2の6 variantを同一familyとし、
5-session moving-block bootstrap 20,000回、Bonferroni個別confidence
98.3333%で同capacityの価格C00とpaired比較した。

全条件ANDで、net40平均・中央値、net60平均、前後半固定slice、正の月3/4、
上位4利益日除外、利益上位5 code現金化、familywise対照下限、unique code、
単一・top10 code集中、実行日数・slot率を要求した。

### selection結果

単位は1 scheduled sessionあたりのpercentage pointである。L90はC00との差の
Bonferroni補正済み片側下限、実行日は非cash outcomeが1つ以上ある日数である。

| variant | net20 | net40 | net60 | 実行日 | L90 vs C00 | gate |
|---|---:|---:|---:|---:|---:|:---:|
| `LQ01_ACTIVITY_VETO__top1` | -0.066286% | -0.086286% | -0.106286% | 8/80 | -0.079858pt | FAIL |
| `LQ01_ACTIVITY_VETO__top2` | -0.007522% | -0.031272% | -0.055022% | 17/80 | -0.297593pt | FAIL |
| `LQ02_EXACT_LIQUIDITY_RIDGE__top1` | -0.170929% | -0.363429% | -0.555929% | 77/80 | -0.212709pt | FAIL |
| `LQ02_EXACT_LIQUIDITY_RIDGE__top2` | -0.192944% | -0.384194% | -0.575444% | 78/80 | -0.496770pt | FAIL |
| `LQ03_VWAP_FLOW_REVERSAL__top1` | -0.035695% | -0.040695% | -0.045695% | 2/80 | -0.008732pt | FAIL |
| `LQ03_VWAP_FLOW_REVERSAL__top2` | +0.004771% | -0.003979% | -0.012729% | 7/80 | -0.271114pt | FAIL |

capacity別C00は次のとおりだった。

| control | net20 | net40 | net60 |
|---|---:|---:|---:|
| `C00_PRICE_RIDGE__top1` | -0.391755% | -0.584255% | -0.776755% |
| `C00_PRICE_RIDGE__top2` | -0.009063% | -0.200313% | -0.391563% |

LQ01/LQ03はnet40中央値が0だが、予測の安定性ではなく大半がcashだからである。
`LQ03__top2`のnet20微益も7/80日しか実行しておらず、coverage・分散・集中・
tail・familywise条件を通らない。LQ02 top1はC00 top1より点推定で
`+0.220826pt/日`だが、絶対net40、両固定slice、familywise下限が負だった。
LQ02 top2は実行率95.625%、88 codeとcoverageがある一方、C00 top2より点推定でも
`-0.183881pt/日`悪化した。

### replay停止とauthority

事前登録winner ruleはpasserが0なら候補全件を棄却し、2026-04-01～07-27の
candidate-specific replay PDFを開かないと定めた。今回のgate passerは0なので、
別winner lockを作らずreplay inputをこの実験では取得・閲覧していない。

ただしEntry 004によりproject outcomesは以前に閲覧済みであり、これは
genuinely untouchedなholdoutではない。`locked_replay_input_opened=false`は
この実験の入力境界だけを表し、project全体の未見性を意味しない。仮にpasserが
あってもproduction昇格権限はなかった。

### artifactと再現

```text
protocol SHA-256:
f7b1329959b237323d6aa08d87494c9523bb5bdd4e29ee8c30cd04ece9e4387e

runner SHA-256:
de5b0a6382f53ce4d513a187b4d2243149e60b8b9a7a69043e9b74d12171a537

selection result SHA-256:
7f8aff8b1e86240c00de3b124e9564ff5b3a0277f91a2cb9b27ef859207f4008

selection picks SHA-256:
65fb583501e30658273cb3d98725e2e4d142fa5681d3ed82023da597ddaf4b71

daily source-set SHA-256:
c20e57b1eb56013d7c92977dc187e1746123c7d5d7995d2a8d6c6cd3864bd94e

score-ledger semantic SHA-256:
32c0e9a363000f7b37fd445541177d2add8d039d6beb26234b478977663cba20

pdftotext:
24.02.0

pdftotext executable SHA-256:
0fb98ea179e19154a90202608c164f2a319b79f16576fa6534b2d601033565e7

independent audit runner SHA-256:
45e2bf0a696197a1d7a0206def38d2517292ec35740c0a1df5e4c50d1fd26063

independent audit result SHA-256:
60cf81b6cf91fc1820ab5fa7d35587bea6ded91036018139aadb0882c4b6d5ed
```

selection runnerとprojectの損益・bootstrap helperをimportしない別監査は、
640固定slotから20/40/60bp損益、tail・code cash stress、集中・実行率、
paired moving-block bootstrap、6 × 13 gate、winner・authorityを再計算した。
25/25 checkがPASS、discrepancy 0、最大数値差0.0だった。raw PDF再parseと特徴再構築は
監査範囲外とし、source hash検査とmutation testから分離している。

```bash
PYTHONPATH=src:. python research/model_v16_liquidity_runner.py \
  --price-warmup-directory /path/to/locked-monthly-pdfs \
  --daily-directory /path/to/locked-daily-pdfs

python -m pytest -q tests/test_model_v16_liquidity.py \
  tests/test_model_v16_liquidity_artifacts.py
python -m pytest -q
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

同じselection outcomeへveto閾値、Ridge alpha、特徴方向、capacityを合わせ直さない。
次の試行には新しい未観測期間、または08:58 auction/order-book、実spread/slippage、
source-complete material cohortのような異なる実情報を必要とする。取得経路とPIT
証拠を成立させられない系統は、Entry 020どおり候補ごと棄却する。

</details>

## Entry 022 — 研究プロトコルv1.7：実流動性を信頼性・状態として使う10機構の反証

| 項目 | 内容 |
|---|---|
| 実施日 | 2026-08-04 |
| 親Entry | Entry 021 |
| 事前登録コミット | `78743a825d3c214f05c5f92f34206aa9a7a33731` |
| authority | `retrospective_candidate_specific_selection`、`project_level_untouched=false` |
| 実入力 | 公式JPX daily PDF 239本、2025-08-01～2026-07-27 |
| parser | 930,286行、rejected row 0 |
| selection | 2026-04-01～07-27、79 scheduled sessions、source-complete 78/79 |
| family | liquidity-as-reliability/state/regime 10候補、1 fixed slot、gridなし |
| controls | C00 top1、PAIR00 price-only top2 reranker、C02 C00 rank2 diagnostic |
| gate | `0/10` |
| 最終判断 | 全候補棄却、winner・research nomineeなし |
| production | 変更なし |
| orders | `false` |

> **一行結論:** v1.6の閾値やalphaを調整せず、流動性を相対信頼性・issuer状態・
> 時間記憶・学習weight・market regimeとして使う10機構へ組み替えたが、全候補が
> late slice、familywise下限、中央値、tail、code集中を落とした。点推定の改善は
> 少数日・少数銘柄・前半へ集中し、production改善を確認できない。

<details>
<summary><strong>新仮説、one-shot評価、次のshoulder-state仮説</strong></summary>

### 評価結果からの仮説生成

Entry 021が反証したのは、hard veto、実流動性7特徴のglobal additive Ridge、
固定VWAP-flow reversalの3機構である。保存済みv1.6 picksではC00 rank1がgross
`-0.1993%/day`、rank2が`+0.5636%/day`で、LQ02はrank1を弱めてもrank2の利益を
消していた。そこで実流動性を平均alphaとして足さず、価格signalの信頼性・
状態・tail・学習weightとして使う次の10候補を事前固定した。

```text
PAIR01  frozen C00 top2内のexact-liquidity binary rerank
TW02    turnover-weighted price memory
VW03    aggregate VWAP cost basis
RPY04   return-per-turnover deterministic reversal
AR05    issuer activity-regime price experts
PS06    persistent vs isolated activity
VP07    VWAP-range pressure memory
RW08    exact traded-lot reliability weight
MR09    turnover-weighted market regime
AT10    attention migration
```

全候補1 fixed slotで、capacity・alpha・閾値gridはない。PAIR01は同じfrozen top2と
pair学習集合のPAIR00、残り9候補はC00 top1とpaired比較した。主コスト40bp、
感応度20/60bp、moving-block bootstrap 20,000回、block 5、family size 10に対する
個別片側confidence 99%を固定した。

### inputとPIT

新79 daily PDFはoutcome parse前にname・bytes・SHA-256を固定した。旧160本と合わせ、
公式239本を固定parserで930,286行へ復元し、rejectは0だった。

```text
daily date bounds:                2025-08-01 .. 2026-07-27
legacy / extension daily:         160 / 79
daily parsed / rejected rows:     930,286 / 0
panel rows / codes:               1,187,484 / 4,096
selection source complete:        78 / 79
score / picks fixed rows:         1,027 / 1,027
observed outcome slots:           993
same-day finality used in score:  false
maximum liquidity source < target:true
strictly-prior registered folds:  48 / 48
```

2026-06-30はfrozen C00 top2が完全でないため、全モデルを登録済みcash ruleで処理した。
予定日の分母から除外せず、下位順位へ置換していない。outcome列なしのscore ledgerを
先に意味的hashへ固定してからreturnをjoinした。

### one-shot結果

単位は1 scheduled sessionあたりのpercentage pointである。`LB99`は登録対照との差の
Bonferroni補正済み片側99%下限、`late`は固定後半40 sessionsである。

| candidate | net20 | net40 | net60 | LB99 | late | tail4 | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `PAIR01` | +0.203884% | +0.006415% | -0.191053% | -0.064551pt | -0.192338% | -0.399748% | FAIL |
| `TW02` | +0.194632% | +0.002227% | -0.190178% | -0.157715pt | -0.287655% | -0.352167% | FAIL |
| `VW03` | +0.338163% | +0.145758% | -0.046647% | -0.149557pt | -0.185668% | -0.210817% | FAIL |
| `RPY04` | +0.327041% | +0.134636% | -0.057769% | -0.648966pt | -0.286746% | -0.446152% | FAIL |
| `AR05` | +0.385187% | +0.192782% | +0.000377% | -0.224118pt | -0.025370% | -0.210179% | FAIL |
| `PS06` | +0.355611% | +0.163206% | -0.029199% | -0.181725pt | -0.158666% | -0.192438% | FAIL |
| `VP07` | +0.345347% | +0.152942% | -0.039463% | -0.038969pt | -0.106835% | -0.195245% | FAIL |
| `RW08` | +0.000890% | -0.191515% | -0.383920% | -0.611633pt | -0.178283% | -0.490320% | FAIL |
| `MR09` | +0.116340% | -0.076065% | -0.268470% | -0.148198pt | -0.164293% | -0.434635% | FAIL |
| `AT10` | +0.170980% | -0.023956% | -0.218893% | -0.471729pt | -0.252633% | -0.455960% | FAIL |

C00 top1はgross`+0.374650%`、net40`-0.010160%`だった。C02 rank2とPAIR00は
79日すべて同じfixed slotでnet40`-0.477644%`となり、Entry 021で観測したrank2
shoulder優位は再現しなかった。

全10候補が両slice、FWER下限、net40中央値、上位4利益日除外、上位5利益code cash、
最大code share、top10 code shareを不合格とした。一方で10候補すべてが実行日数・
slot率を通過しており、failureはcash sparsityではない。AR05の絶対net40とVP07の
FWER下限は候補中最良でも、late・tail・diversityが負で採用できない。

### pair診断と次の仮説

保存済みpicksの事後診断では、PAIR00はcompleteな78日すべてでrank2を選んだ。
PAIR01はrank1を40日、rank2を38日、cashを1日選び、PAIR00比では
`+0.484059pt/session`改善した。しかし登録対照でないC00 top1との事後差は全79日で
`+0.0166pt/session`にすぎず、exact liquidity固有の優位は未証明である。

Entry 021のrank2優位からEntry 022のrank2 net40`-0.477644%`への反転に基づき、
次の中心仮説を次のように限定する。

> **rank1–rank2 shoulder polarityは固定premiumではなく、strictly lagged OOFの
> rank別実現差で観測できるslow stateとして持続・反転する。**

v1.8ではv1.7をdevelopment扱いし、v1.7特徴・係数・windowをこの79日に合わせ直さない。
前月までのOOF rank1-minus-rank2実現差だけでfrozen top2のrank positionまたはcashを
選ぶ新機序をfresh期間へ事前登録する。tail・code cash・集中度gateは緩めない。

### 独立監査とartifact

selection runnerとproject損益・bootstrap helperをimportしない別監査が、
全metric、20,000-sample paired bootstrap、gate、winner、authorityを再計算した。
29/29 checkがPASS、discrepancy 0、最大数値差0.0だった。raw PDF再parseと特徴再構築は
監査範囲外で、source/parser hashとPIT mutation testから分離している。

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
runtime:                    978.260319 seconds
Python / NumPy:             3.12.13 / 2.3.5
pandas / scikit-learn:      2.2.3 / 1.8.0
targeted v1.7 tests:        16 passed
full tests:                429 passed
subtests:                  192 passed
gate passers:                0 / 10
winner / nominee:            none / none
production model changed:   false
orders allowed:             false
```

</details>

## Entry 023 — 研究プロトコルv1.8：lagged monthly shoulder-stateのfresh-forward事前登録

| 項目 | 内容 |
|---|---|
| 実施日 | 2026-08-04 |
| 親Entry | Entry 022 |
| status | `implementation_complete_activation_pending` |
| authority | `genuinely_later_forward_shadow_development` |
| candidate | `SH01_LAGGED_MONTHLY_SHOULDER_STATE`、family size 1 |
| controls | `C00_PRICE_RIDGE_TOP1`、`C02_C00_RANK2` |
| fresh期間 | 2026-08-05以後、120 scheduled sessions以上、6 calendar months以上、最初の適格月末まで |
| activation | 3 direct commits A/B/C。本Entryを含むcommit Aのみを作り、payload B・receipt Cは未作成 |
| evaluation | 未開封。score、decision、outcome、picks、resultは未生成 |
| production | 変更なし |
| orders | `false` |

> **一行結論:** v1.6で観測したrank2優位がv1.7で反転したため、固定rank2 premiumや
> v1.7流動性特徴の再調整をやめ、前3完了月のstrictly lagged rank1-minus-rank2
> 実現差だけでC00 frozen top2の順位を月次選択する1機構を、未観測のforward期間へ
> 固定した。実装は完了したが、合格・性能改善・production昇格はまだ主張しない。

<details>
<summary><strong>仮説、fresh-forward契約、PIT証拠、合格条件</strong></summary>

### 評価結果から立てた新仮説

Entry 022ではC00 rank2/PAIR00がnet40 `-0.477644%`となり、Entry 021のrank2
shoulder優位が反転した。PAIR01はPAIR00を救済しても、未登録のC00 top1との事後差は
全79日で約`+0.0166pt/session`にすぎず、exact liquidity固有の優位は確認できない。

したがってv1.8の仮説を次の1点へ限定する。

> **C00 rank1–rank2 shoulder polarityは固定premiumではなく、前月までのOOF
> rank別実現差が示すslow stateとして持続・反転する。**

不足していたのは、平均alphaを増やす別特徴や閾値の微調整ではない。合格には、
未観測期間でも状態方向が持続し、利益が少数日・少数codeへ集中せず、40/60bp、
両時系列slice、中央値、tail除去、code-cash stress、familywise対照下限を同時に
通ることが必要である。v1.8はこの不足を検証するが、通過を保証しない。

### SH01の固定式

各counted sessionで、v1.7と同じG0価格Ridge `alpha=1`を前月末までのデータだけで
月次fitし、outcomeなしでC00 top2を固定する。complete pairの日だけ次を作る。

```text
d_t = C00 rank1 open-to-close return pct
    - C00 rank2 open-to-close return pct
```

各完了月はcomplete pairが10日以上なら`d_t`の通常中央値、未満ならunavailableとする。
対象月Mのstateは、飛ばしなしでM-3、M-2、M-1の3月中央値の通常中央値とする。

```text
state > 0 : frozen C00 rank1を1 slot選択
state < 0 : frozen C00 rank2を1 slot選択
state = 0 : cash
必要月unavailable : cash
```

対象月の途中ではstateを更新しない。target-month outcome、partial month aggregate、
target-date price、後日revisionは入れない。rank3以下への置換、window・sign・alpha・
capacity search、v1.7の79日に対する再調整はない。

固定済みseedは次のとおりで、2026-07だけはv1.7が07-27で終了したため、そのbytesへ
事前にbindされた透明なpartial historical exceptionである。forward月には再利用しない。

| completed month | complete pairs | median rank1-rank2 |
|---|---:|---:|
| 2026-05 | 17 | +0.4532617412224217pt |
| 2026-06 | 19 | +0.5353494177210093pt |
| 2026-07（01–27） | 18 | +0.09214571919513584pt |

条件どおり2026-08の初期stateは`+0.4532617412224217pt`、選択順位はrank1である。

### genuinely-later forwardと停止境界

登録calendarはJPX営業日343本、2026-08-05～2027-12-30を固定した。最初のcounted
sessionはreceipt commit Cのsuccessful pull-request workflowが持つGitHub server
`updated_at`だけから、08:58:59 Asia/Tokyo cutoffより前の最初の登録sessionへ機械的に
決める。観測がその既決定cutoffに間に合わなければactivation全体を失敗させ、後ろへ
ずらしたりbackfillしたりしない。

仮にfirst counted sessionが2026-08-05なら、120本目は2027-02-02、terminalは
2027-02-26、N=136、7 calendar months、前後半68/68となる。実際のterminalは
activationで確定したfirst sessionから同じ規則で再計算する。

このcommit Aではactivation payload/receiptを作らず、forward source archive、exact
hash、parser、runtimeを揃えた後に、payloadだけのdirect child B、receiptだけのdirect
child Cを作る。A/B/Cそれぞれの最初のattempt-1 pull-request workflow成功を要求する。
現時点ではcounted session、performance、nomineeは存在しない。

### 日次PIT checkpoint

日次のprimaryとhistorical名`safety_cash`は、同じdecision coreを別nonceの固定
16,384-byte envelopeへ封印する。`safety_cash`は公開順序の証拠だけで、cashやfallbackの
選択権限を持たない。primaryの最初のattempt-1 workflow `created_at < cutoff`だけが
日次の選択権威で、status、conclusion、`run_started_at`、`updated_at`は順位やcashを
変えない。primaryがmissing、late-created、ambiguous、改変ならsession cashではなく
実験全体をintegrity abortする。

公開は固定12-step GitHub Git Data API契約でsafety→primaryのsole-parent chainを作る。
terminal監査はlocal Gitを信頼せず、remote ref、全path history、commit/tree/blob、
compare、Actions全pagination、両coreを独立再構築する。外部predictor/outcome/core証拠は
repo外append-only storeへ置き、dirfd、`O_NOFOLLOW`、single-link、inode uniquenessで
symlink・hardlink・TOCTOUを拒否する。

runtime lockはPython/package tree、startup hook、live module origin、ELF root/shared
object、Git、pdftotext、CA bundle、critical environmentをexact bytesへ固定する。
completed-month ledgerのterminal-inclusive schema/hash/chronologyをoutcome-blindに
検証するまで、runnerと独立auditはoutcome ledgerをopen・parseしない。

### 合格gateと権限

主コスト40bp、感応度20/60bp、paired moving-block bootstrapはblock 20、20,000回、
seed 20260805、family size 1、片側90%下限である。次をすべてANDで要求する。

```text
net40 mean > 0
net40 median > 0
net60 mean > 0
early / late両sliceのnet40 > 0
ceil(75% × represented months)以上の月でnet40 > 0
上位4利益日を除いてnet40 > 0
利益上位5 codeをcashにしてnet40 > 0
paired point delta vs C00 top1 > 0
paired one-sided lower bound >= 0
unique code >= 40
maximum code share <= 0.05
top10 code share <= 0.25
executed days >= ceil(90% × N)
executed slot fraction >= 0.80
```

通過しても変更なしのSH01をv1.9 sealed confirmationへ1件nominateできるだけで、
production promotion、注文、v1.8 outcomeに合わせた係数・window変更は禁止する。

### artifactと事前検証

```text
hypothesis SHA-256:          d92dfea8b02d2e10516e4c98fae7219e8b8cdc91967d57df1c5f546ef55111d6
protocol SHA-256:            c623fabfa8e94381bce27d359cefc6e51a9a80f1c18f62f6098cfdfc8e9f6112
runtime-lock file SHA-256:   95a867e2f8f187528a7ba3d24f4db4f1bf0964531b6e0e2b85d88d0c758f1e32
runtime-lock self SHA-256:   fcb453b3532625e2eefad679972389bd80cb7ee7685a40610d5aa858d7a32c2b
calendar SHA-256:            c5c5908b0e26ebd57eb2e473b9d4ce7f92a7c8b336f6ad70596152de971b77a7
runner SHA-256:              ac9a8311799d18288499004771b99eb4d22734d40641bcf2e1af39b0a97e64e9
independent audit SHA-256:   4fa58c8ef35ca5c784ec93708bb8a0d6458236c023501f7cc9649a28e6dd87ec
tests SHA-256:               4b796da33f953739de91c8ec12657a73045eb7eb2ce10f888970b274948f20be
workflow SHA-256:            7c9811768511bacb889e0f7bfa9508c2d2212079e8857db3b152d75868c232c7
```

```text
py_compile:                  PASS
protocol validation:        PASS
strict runtime validation:  PASS
targeted v1.8 tests:        53 passed (232.25s)
full tests:                 482 passed (246.60s)
subtests:                   192 passed
warnings:                    27
read-only material blockers: 0
activation payload/receipt: absent / absent
outcomes / result:          unopened / absent
production model changed:   false
orders allowed:             false
```

</details>
