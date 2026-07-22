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
