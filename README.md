# TSE Session Ranker

寄り前時点で「当日の終値が始値を上回る」銘柄を、東証普通株の全体から毎日1～2件順位付けするPythonパッケージです。適時開示を必須条件にせず、各銘柄の日中セッション特性を主モデルにします。

現行の`session_v2`は検証版です。ロジスティック回帰の出力は未校正なので、個別銘柄の「勝率」ではなく**同日内の順位スコア**として使います。1位を`CORE`、2位を`RESERVE`として表示し、2位は板条件を満たしても発注可にはしません。

## できること

- JPX株式相場表のPDFまたは`pdftotext -layout`済みTXTを正規化
- 片場だけ約定した4価格行と無約定行を保持
- 対象日OHLCを一切使わない特徴量生成
- 日ごとの総学習重みを均等化した正則化ロジスティック回帰
- 月次expanding walk-forwardバックテスト
- 翌営業日の上位1～2銘柄推論
- MarketSpeed IIからCSV保存した8:58板スナップショットの暫定veto
- モデル、設定、学習期間、データhash、checksumの保存

## インストール

```bash
python -m pip install -e .
```

Parquetを使う場合だけ、`python -m pip install -e '.[parquet]'`を使います。標準ではCSVとpickleが利用できます。pickleとjoblibはいずれも任意コードを含められる形式なので、信頼できるローカルファイルだけを読み込んでください。

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
date, code, name, open, high, low, close
```

任意列は`volume, turnover, traded, partial_session`です。コードは英字入り新コードを壊さないよう文字列で保持します。汎用データを使う場合も、最新日のactive universeを判定できるよう、上場中だが無約定の銘柄を`traded=False`・OHLC欠測の行として含めてください。

## 2. 固定仕様で学習

```bash
tse-session-ranker train \
  --daily var/daily.pkl \
  --train-end 2026-07-17 \
  --artifact var/models/session_v2.joblib \
  --config config/default.json
```

`training_end`より後の行は、特徴量・imputer・scaler・係数・データhashのすべてから除外されます。保存時には`session_v2.joblib.manifest.json`も自動生成され、ロード時のchecksum検証に必須です。artifactを移動・配備するときは、この2ファイルを必ずセットにしてください。

## 3. 毎朝の推論

```bash
tse-session-ranker predict \
  --daily var/daily.pkl \
  --artifact var/models/session_v2.joblib \
  --target-date 2026-07-21 \
  --expected-history-date 2026-07-17 \
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
  --output-dir var/backtests/session_v2
```

各月のモデルはその月より前のデータだけで学習します。出力は`summary.json`, `folds.csv`, `scores.pkl`, `top1.csv`, `top2.csv`です。条件を変更したら過去成績へ継ぎ足さず、新しいバージョンとしてゼロから数えます。

## Python API

```python
from tse_session_ranker import SessionRanker

ranker = SessionRanker()
ranker.collect_jpx(["inputs/jpx"], output="var/daily.pkl")
ranker.train(
    "var/daily.pkl",
    train_end="2026-07-17",
    artifact_path="var/models/session_v2.joblib",
)
result = ranker.predict(
    "var/daily.pkl",
    target_date="2026-07-21",
    expected_history_date="2026-07-17",
    top_k=2,
)
print(result.candidates[["model_rank", "code", "name", "model_score"]])
```

## 現行仕様

モデル特徴は、直前の日中騰落、日中騰落の5/20/60日平均・勝率、20日標準偏差、夜間騰落の直前値・20/60日平均、夜間と日中の60日相関です。すべて対象日より前へshiftされています。

表示対象は、履歴60セッション以上、前日終値100～30,000円、ATR14が0.25～5%、直近20セッションの無約定・始終同値率20%以下です。既存の検証仕様を再現するため、モデル学習母集団だけATR上限15%を明示設定し、候補選定時は5%で切ります。

JPXの日報相場表には出来高・売買代金がないため、現段階では流動性を検証済み条件として扱っていません。実運用ではベンダー日足の`volume/turnover`と単元株数マスターを追加し、発注サイズ判定を別policyとして有効化してください。

学習と推論では同じ価格調整方式を使ってください。分割・併合を未調整のまま跨ぐデータはATRとリターンを壊すため、提供元の調整済みOHLCを使うか、該当lookbackを対象外にします。
