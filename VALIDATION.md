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
