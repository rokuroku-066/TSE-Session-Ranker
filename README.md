# TSE Session Ranker

寄り前時点で東証普通株を毎日順位付けし、**始値→終値のコスト控除後損益率**を最大化する1～2件を選ぶPythonパッケージです。適時開示を必須条件にせず、各銘柄の日中セッション特性を主モデルにします。

0.2.0では、正則化ロジスティック回帰、日次pairwise ranking、多閾値期待損益、二段階回帰などを同一条件で比較し、20bp控除後のtop1平均損益が最も高かった正則化ロジスティック回帰を暫定採用しています。学習ラベルは陽線/陰線ですが、**モデルの採否は勝率ではなく損益率で決めます**。出力は未校正なので個別銘柄の勝率ではなく同日内の順位スコアです。

既定設定では1位を`CORE`、2位を`RESERVE`として表示し、2位は板条件を満たしても発注可にはしません。この仕様はprofit-first探索版であり、新規60～100営業日の前向き検証が必要です。

## できること

- JPX株式相場表のPDFまたは`pdftotext -layout`済みTXTを正規化
- 片場だけ約定した4価格行と無約定行を保持
- 対象日OHLCを一切使わない特徴量生成
- 日ごとの総学習重みを均等化した正則化ロジスティック回帰
- 前営業日の銘柄集合を先に固定し、当日欠落を未約定・損益0として評価
- コスト後平均損益、月別損益、利益集中、最大ドローダウンを主評価
- 月次expanding walk-forwardバックテスト
- 翌営業日の上位1～2銘柄推論
- MarketSpeed IIからCSV保存した8:58板スナップショットの暫定veto
- モデル、設定、学習期間、データhash、checksumの保存

## インストール

リポジトリをcheckoutして開発・検証する場合:

```bash
python -m pip install -e .
```

ビルド済みwheelを配備する場合:

```bash
python -m pip install path/to/tse_session_ranker-0.2.0-py3-none-any.whl
```

wheelには`config/default.json`を含めていないため、下記コマンド例の`--config`を省けば同じ組み込み既定値を使います。Parquetを使う場合だけ、source checkoutで`python -m pip install -e '.[parquet]'`を使います。標準ではCSVとpickleが利用できます。pickleとjoblibはいずれも任意コードを含められる形式なので、信頼できるローカルファイルだけを読み込んでください。

## 1. 日足データ収集

[JPXの月間相場表ページ](https://www.jpx.co.jp/markets/statistics-equities/price/index.html)または日報アーカイブから株式相場表PDFを保存し、月単位のフォルダを渡します。TXTと同名PDFが両方ある場合はTXTを優先します。

```bash
tse-session-ranker collect-jpx \
  --input inputs/jpx/2024 inputs/jpx/2025 \
  --output var/daily.pkl
```

URLが分かっている場合は、明示したURLだけをダウンロードできます。サイト構造を推測する自動スクレイピングはしません。

```bash
tse-session-ranker download-jpx \
  --url 'https://www.jpx.co.jp/.../quotation.pdf' \
  --destination inputs/jpx/2026
```

汎用の日足CSV/pickleも、次の必須列を持てば学習・推論へ直接渡せます。

```text
date, code, open, high, low, close
```

任意列は`name, volume, turnover, traded, partial_session`です。`name`を省くとcodeで補完します。コードは英字入り新コードを壊さないよう文字列で保持します。汎用データを使う場合も、最新日のactive universeを判定できるよう、上場中だが無約定の銘柄を`traded=False`・OHLC欠測の行として含めてください。

学習とバックテストでは対象日前営業日に存在したコードから候補集合を先に作り、その後で対象日の結果をjoinします。対象日行が欠けても候補から消さず、未約定・損益0・コスト0として残します。source coverage基準は、過去に合格した20セッションの銘柄収録数90%分位から作ります。90%未満の日は不完全とし、その日と、その日を候補集合の元にする直後のセッションを学習・評価から除外します。不完全日を削除して古い日へ再接続せず、長期欠落が基準を自動的に引き下げることもありません。実際の市場再編で銘柄数が10%以上恒久的に減る場合は、データ期間または基準を明示的にリセットしてください。

1営業日が全行欠落した場合も検出するため、JPX営業日calendarの指定を強く推奨します。CSVなら`date`列、TXTなら1行1日で用意し、学習・backtest・推論に同じcalendarを渡します。`train_end`と`evaluation_end`にはcalendar上の取引日を指定し、calendar自体もその日以降まで用意してください。artifactには学習終了日までのcalendar hashが入り、別calendarやcalendarなしの推論を拒否します。

```text
date
2026-07-17
2026-07-21
```

収集直後は、末尾の全行欠落も含めて診断できます。

```bash
tse-session-ranker doctor \
  --daily var/daily.pkl \
  --calendar inputs/jpx_sessions.csv \
  --expected-through 2026-07-21
```

## 2. 固定仕様で学習

```bash
tse-session-ranker train \
  --daily var/daily.pkl \
  --train-end 2026-07-17 \
  --artifact var/models/session_v2.joblib \
  --calendar inputs/jpx_sessions.csv \
  --config config/default.json
```

`training_end`より後の行は、特徴量・imputer・scaler・係数・データhashのすべてから除外されます。保存時には`session_v2.joblib.manifest.json`も自動生成され、ロード時のchecksumとpayload整合性検証に必須です。artifactを移動・配備するときは、この2ファイルを必ずセットにしてください。joblibの互換性をfail-closedで扱うため、ロード環境は学習時と同じscikit-learn major/minor版を使います。

0.2.0は候補集合とsource embargoの意味を`prior_session_universe_source_mask_v2`へ変更し、config schemaを2、artifact schemaを4へ更新しました。0.1.xおよび修正前0.2.0のartifactは再学習が必要です。

## 3. 毎朝の推論

```bash
tse-session-ranker predict \
  --daily var/daily.pkl \
  --artifact var/models/session_v2.joblib \
  --target-date 2026-07-21 \
  --expected-history-date 2026-07-17 \
  --calendar inputs/jpx_sessions.csv \
  --top-k 2 \
  --output var/predictions/2026-07-21.csv
```

当日行は不要です。モジュール内で`target_date`用のラベルなし行を作り、前営業日までの履歴だけから特徴量を計算します。`--expected-history-date`と実データの最終日が一致しなければ停止します。さらに、最新完了セッションに存在しないコードは上場廃止・コード変更済みとして除き、最新日の銘柄収録率が直近20日中央値の90%未満、または日足が7暦日超古い場合も推論を停止します。

## 4. MarketSpeed IIの寄り前スナップショット

MarketSpeed II RSS自体はWindows/Excel上で動くため、このパッケージは定時保存したCSVを時刻付きで取り込みます。最低限の列は次のとおりです。

```text
observed_at,code,indicative_price
```

推奨列:

```text
target_date,prior_close,buy_special,sell_special,expected_open_at,
buy_market_qty,sell_market_qty,pts_price,pts_volume
```

`observed_at`は`2026-07-21T08:58:00+09:00`のようにタイムゾーン必須です。

```bash
tse-session-ranker ingest-preopen \
  --input inputs/preopen/2026-07-21.csv \
  --output var/preopen.pkl

tse-session-ranker predict \
  --daily var/daily.pkl \
  --artifact var/models/session_v2.joblib \
  --target-date 2026-07-21 \
  --expected-history-date 2026-07-17 \
  --calendar inputs/jpx_sessions.csv \
  --preopen var/preopen.pkl \
  --as-of 2026-07-21T08:58:00+09:00
```

板オーバーレイは`model_score`と`model_rank`を変えません。次の場合だけ`DISPLAY_ONLY`にします。

- 8:58時点のスナップショットが欠測または3分超古い
- 市場調整前の暫定ギャップがATR14の+2倍以上
- 買い特別気配
- 予想寄り付きが9:05より後
- 2位以下の`RESERVE`

板CSVを渡さない場合は`MODEL_ONLY`と表示します。このvetoはまだ履歴検証済みの予測特徴ではなく、発注可否のshadowルールです。

## 5. Walk-forward検証

```bash
tse-session-ranker backtest \
  --daily var/daily.pkl \
  --start 2025-01-06 \
  --end 2025-07-31 \
  --calendar inputs/jpx_sessions.csv \
  --output-dir var/backtests/session_v2
```

各月のモデルはその月より前のデータだけで学習します。出力は`summary.json`, `folds.csv`, `scores.pkl`, `top1.csv`, `top2.csv`です。条件を変更したら過去成績へ継ぎ足さず、新しいバージョンとしてゼロから数えます。

主指標は`top1.net_mean_pct_at_cost`です。top2は各銘柄へ事前に50%ずつ割り当て、片方が未約定でも約定銘柄へ資金を事後再配分しません。`summary.json`には0/10/20/40/60bpのコスト感応度、コスト後中央値、複利損益、最大ドローダウン、Profit Factor、月別損益、上位5日除外平均も保存します。勝率・AUC・Brierは診断指標です。

## Python API

```python
import pandas as pd

from tse_session_ranker import SessionRanker

jpx_sessions = pd.read_csv("inputs/jpx_sessions.csv")["date"]
ranker = SessionRanker()
ranker.collect_jpx(["inputs/jpx"], output="var/daily.pkl")
ranker.train(
    "var/daily.pkl",
    train_end="2026-07-17",
    artifact_path="var/models/session_v2.joblib",
    expected_sessions=jpx_sessions,
)
result = ranker.predict(
    "var/daily.pkl",
    target_date="2026-07-21",
    expected_history_date="2026-07-17",
    expected_sessions=jpx_sessions,
    top_k=2,
)
print(result.candidates[["model_rank", "code", "name", "model_score"]])
```

## 現行仕様

モデル特徴は、直前の日中騰落、日中騰落の5/20/60日平均・勝率、20日標準偏差、夜間騰落の直前値・20/60日平均、夜間と日中の60日相関です。すべて対象日より前へshiftされています。

表示対象は、履歴60セッション以上、前日終値100～30,000円、ATR14が0.25～5%、直近20セッションの無約定・始終同値率20%以下です。既存の検証仕様を再現するため、モデル学習母集団だけATR上限15%を明示設定し、候補選定時は5%で切ります。

JPXの日報相場表には出来高・売買代金がないため、現段階では流動性を検証済み条件として扱っていません。実運用ではベンダー日足の`volume/turnover`と単元株数マスターを追加し、発注サイズ判定を別policyとして有効化してください。

学習と推論では同じ価格調整方式を使ってください。分割・併合を未調整のまま跨ぐデータはATRとリターンを壊すため、提供元の調整済みOHLCを使うか、該当lookbackを対象外にします。

モデル選択と検証結果の詳細は、リポジトリの[VALIDATION.md](https://github.com/rokuroku-066/TSE-Session-Ranker/blob/main/VALIDATION.md)を参照してください。
