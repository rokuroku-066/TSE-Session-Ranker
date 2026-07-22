# TSE Session Ranker

寄り前時点で東証普通株を順位付けし、当日の**始値→終値のコスト控除後損益率**を最大化する候補を毎日1～2件出すPythonパッケージです。

現行CLI/APIの既定モデルは0.3.0互換の`session_v3_tdnet_clear`です。正則化ロジスティック回帰に、価格履歴12特徴と寄り前までのTDnet適時開示6特徴を入力し、1位を`CORE`、2位を`RESERVE`として表示します。

> [!WARNING]
> **自動発注には使用できません。** v0.6で候補特徴を機構別に固定し、同一モデルでgroup ablationをやり直しました。一次通過した前後場shapeも次期間で価格core比 `-0.593191pt/日`に反転し、追加採用は0群です。既定のv0.3を含む全候補はshadow・研究用途だけです。

## 最新の研究判断

| 仕様 | 特徴・手法 | 用途 | 20bp後結果 |
|---|---|---|---:|
| v0.3既定 | 価格12 + TDnet 6、logit | CLI/APIのshadow表示 | 既知診断top1 `-0.087886%/日` |
| v0.4固定候補 | 価格・前後場・市場・TDnet 62、logit | 負のresearch baseline | 既知診断top1 `-0.086199%/日` |
| v0.5選定winner | 価格12 + TDnet 6、raked logit | retrospectiveで棄却 | 確認期top2 `-0.484256%/日` |
| v0.5 TDnet診断対照 | 価格・前後場・市場・TDnet 62、raked logit | 後付け採用禁止 | 確認期top2 `+0.039661%/日` |
| v0.6一次首位 | 価格core + 前後場shape、raked logit | 次期間で棄却 | screen top2 `-0.218990%/日` |
| v0.6最終lock | 価格coreのみ、raked logit | 損益優位性未実証 | stability top2 `-0.421878%/日` |

v0.6は損益を見る前に、価格反転、過去gap特性、前後場shape、市場レジーム、流動性proxy、整理済みTDnet、開示構造の7追加群を固定しました。特徴寄与を分離するためraked logitは共通です。独立監査で12 recipe・1,926数値と966個のprovenance hashを再現し、追加特徴なしという負の結果を確定しました。候補の定義は[feature candidate inventory](research/model_v06_feature_candidates.md)、経緯と訂正は[VALIDATION.md](VALIDATION.md) Entry 006・007にあります。

## 設計の要点

- production互換の既定推定器は`LogisticRegression(C=0.08, class_weight="balanced")`で固定
- v0.4研究winnerも同じL2正則化logitで、`C=0.03`、日別・クラス均衡、expanding学習窓
- v0.5研究ではモデル固定を外したが、選定winnerが確認期で再現せず、production推定器は変更なし
- v0.6では特徴候補を先に7群へ固定し、共通モデルのgroup ablationで追加寄与を分離。次期間まで残った追加群は0
- 学習ラベルは`close > open`または損益・同日順位。研究仕様の採否は勝率ではなくコスト後損益で決定し、v0.5の主指標はtop2等金額
- 対象日OHLCを特徴量へ入れず、価格特徴はすべて1セッション以上shift
- 適時開示は各文書を「公開時刻以前で最初に到来する08:58:59 JSTの取引日」へ割当
- 上方・下方修正、増配・減配、新規自己株取得決定、新規エクイティ調達をタイトルから保守的に分類
- 月次walk-forward、前営業日の候補集合固定、未約定枠は現金のまま評価
- 明示的なJPX営業日calendar、日足収録率、TDnet日別ページの取得時刻をfail-closedで検証
- MarketSpeed IIの8:58板は順位を変更せず、発注可否のvetoにだけ使用
- 08:58:59以前の先物contextはappend-onlyで保存できるが、履歴snapshot不足のため現行順位には未使用

## インストール

```bash
python -m pip install -e .
```

ビルド済みwheelの場合:

```bash
python -m pip install path/to/tse_session_ranker-0.3.0-py3-none-any.whl
```

wheel単体には`config/default.json`を含めていないため、配備先では`--config`を省略すれば同じ組み込み既定値を使えます。

Parquetを使う場合のみ`python -m pip install -e '.[parquet]'`が必要です。pickle/joblibは信頼できるローカルファイルだけを読み込んでください。

## 1. 日足と営業日calendar

JPX株式相場表のPDFまたは`pdftotext -layout`済みTXTを収集します。

```bash
tse-session-ranker collect-jpx \
  --input inputs/jpx/2024 inputs/jpx/2025 inputs/jpx/2026 \
  --output var/daily.pkl
```

汎用CSV/pickleも次の列があれば利用できます。

```text
date, code, open, high, low, close
```

任意列は`name, volume, turnover, traded, partial_session`です。上場中の無約定銘柄も`traded=False`の行として保持してください。

全銘柄が欠落した取引日を検出するため、JPX営業日をCSVまたは1行1日で用意します。

```text
date
2026-07-17
2026-07-21
```

```bash
tse-session-ranker doctor \
  --daily var/daily.pkl \
  --calendar inputs/jpx_sessions.csv \
  --expected-through 2026-07-21
```

JPX parser v6は、`Final Special Quote`欄の数値直前に付く特別気配marker`ｶ`・`ｳ`だけを受理します。他の価格・騰落欄に同じmarkerが現れた場合は、列ずれを疑って従来どおり拒否します。

## 2. TDnet適時開示

公開日別インデックスをカレンダー日単位で保存します。現行downloaderは第三者の公開TDnet日別ミラーを利用するため、ミラー掲載の遅延や後日の訂正までは保証できません。週末・祝日にも開示があり得るため、前回取得日から対象日まで**全日**を指定します。

```bash
tse-session-ranker download-tdnet \
  --start 2024-01-04 \
  --end 2026-07-21 \
  --destination var/tdnet-html \
  --workers 4
```

各HTMLには、リクエスト開始時刻、取得完了時刻、SHA-256と`network_request_start_recorded` provenanceを記録した`.meta.json`が付きます。次の安全条件を満たさないページは再取得または拒否されます。

- 過去日のページは翌カレンダー日以後にリクエストされている
- 対象日ページは08:58:59以後にリクエストが開始されている
- HTMLとsidecarのサイズ・hash・日付が一致する
- HTMLの見出しと開示カード構造が期待形式に一致する

したがって、当日推論用のダウンロードは08:59以後に開始してください。当日朝に取得した可変ページは翌日も自動的に再取得されます。sidecarがない、provenanceタグがない、または履歴研究用mtimeで補った旧キャッシュはproductionでは利用できません。`--overwrite`で再取得してください。

読み込みを速くしたい場合は、検証済みHTML一式をデータセットへ変換できます。manifestも必ずセットで保持してください。

```bash
tse-session-ranker collect-tdnet \
  --input var/tdnet-html \
  --output var/tdnet.pkl
```

以降の`--tdnet-cache`には、HTMLディレクトリまたはmanifest付きの`var/tdnet.pkl`を指定できます。export manifest schema 2は、開示本体だけでなく完全日、観測時刻、provenanceもdataset checksumへ結合し、単独のmetadata改変を拒否します。

### 使用する適時開示特徴

モデルへ入れるのは次の6個の0/1特徴だけです。

```text
tdnet_has_revision_up
tdnet_has_revision_down
tdnet_has_dividend_up
tdnet_has_dividend_down
tdnet_has_buyback_decision
tdnet_has_equity_financing
```

誤分類を減らすため、方向語は親カテゴリと同時に一致した場合だけ有効です。例えば「配当予想の上方修正」は業績上方修正にしません。また、ToSTNeT・取得状況・取得終了は新規自己株取得決定から、払込完了・行使状況・発行結果は新規エクイティ調達から除外します。

`tdnet_has_buyback_decision`はタイトル上の新規取得決定であり、継続的な市場買付を保証するものではありません。0.3.0はPDF本文の取得方法や業績修正額をまだ構造化していません。

v0.4研究では、開示有無・件数・時刻、決算、ToSTNeT、優待、分割、M&A、減損、監査問題などを含むTDnet 20特徴と3 interactionまで広げました。これは`research/finalize_logit_v04.py`の比較専用registryで、通常の`train`・`predict`が使うproduction 6特徴へは昇格していません。

## 3. 学習

```bash
tse-session-ranker train \
  --daily var/daily.pkl \
  --tdnet-cache var/tdnet.pkl \
  --train-end 2026-07-17 \
  --artifact var/models/session_v3.joblib \
  --calendar inputs/jpx_sessions.csv \
  --config config/default.json
```

`training_end`より後の日足と08:58:59より後の適時開示は、特徴、補完値、scaler、係数、学習hashのすべてから除外されます。保存物は次の2ファイルです。

```text
session_v3.joblib
session_v3.joblib.manifest.json
```

0.3.0はconfig schema 3、artifact schema 5です。0.2以前のartifactは再学習してください。ロード環境は学習時と同じscikit-learn major/minor版を使います。

## 4. 毎朝の推論

対象日までのTDnetページを08:59以後に更新してから実行します。

```bash
tse-session-ranker download-tdnet \
  --start 2026-07-17 \
  --end 2026-07-21 \
  --destination var/tdnet-live

tse-session-ranker predict \
  --daily var/daily.pkl \
  --tdnet-cache var/tdnet-live \
  --artifact var/models/session_v3.joblib \
  --target-date 2026-07-21 \
  --expected-history-date 2026-07-17 \
  --calendar inputs/jpx_sessions.csv \
  --top-k 2 \
  --output var/predictions/2026-07-21.csv
```

当日OHLC行は不要です。対象日のラベルなし行を内部生成し、前営業日までの日足と対象日08:58:59までの開示だけで順位付けします。日足が古い、最新日収録率が90%未満、営業日calendarがartifactと不一致、必要なTDnet日別ページが欠ける、または取得開始がcutoffより早い場合は停止します。

## 5. MarketSpeed II板オーバーレイ

最低列:

```text
observed_at,code,indicative_price
```

推奨列:

```text
target_date,prior_close,buy_special,sell_special,expected_open_at,
buy_market_qty,sell_market_qty,pts_price,pts_volume
```

```bash
tse-session-ranker ingest-preopen \
  --input inputs/preopen/2026-07-21.csv \
  --output var/preopen.pkl

tse-session-ranker predict \
  --daily var/daily.pkl \
  --tdnet-cache var/tdnet-live \
  --artifact var/models/session_v3.joblib \
  --target-date 2026-07-21 \
  --expected-history-date 2026-07-17 \
  --calendar inputs/jpx_sessions.csv \
  --preopen var/preopen.pkl \
  --as-of 2026-07-21T08:58:00+09:00
```

板は`model_score`と`model_rank`を変更しません。欠測・3分超古い、暫定ギャップがATR14の+2倍以上、買い特別気配、予想寄りが9:05より後、または`RESERVE`の場合だけ`DISPLAY_ONLY`にします。

## 6. 先物contextの前向き収集

日経225先物・TOPIX先物の寄り前騰落率は、v0.4の次に検証する候補です。正確な過去08:58 snapshotを復元できないため、現在のモデル順位には入れません。日足終値、9:00以後の値、後から確定した値で代用しないでください。

最低列:

```text
date,observed_at,nikkei_return_pct,topix_return_pct,return_definition
```

例:

```csv
date,observed_at,nikkei_return_pct,topix_return_pct,return_definition,source
2026-07-21,2026-07-21T08:58:00+09:00,-1.20,-0.85,previous_cash_close_to_08:58:59_JST,forward-snapshot
```

```bash
tse-session-ranker ingest-market-context \
  --input inputs/market-context/2026-07-21.csv \
  --existing var/market-context.pkl \
  --output var/market-context.pkl
```

`observed_at`はtimezone付きで、対象日と同じJST日付の08:58:59以前でなければ拒否します。`return_definition`を途中で変更できず、既存日への異なる値の上書き、過去へのbackfill、欠測日のcarry-forwardも拒否します。

研究用に実装済みのinteractionは次のとおりです。

```text
market_beta_60 × Nikkei/TOPIX futures return
ATR14 × abs(futures return)
market_beta_60 × Nikkei-TOPIX futures spread
TDnet material flag × futures return
```

少なくとも60～100新規営業日をappend-onlyで収集してから、新しいprotocolで採否を判断します。それまでは欠測indicatorを含め、通常のv0.3・v0.4スコアへ混ぜません。

## 7. Walk-forward検証

```bash
tse-session-ranker backtest \
  --daily var/daily.pkl \
  --tdnet-cache var/tdnet.pkl \
  --start 2025-01-06 \
  --end 2025-03-31 \
  --calendar inputs/jpx_sessions.csv \
  --output-dir var/backtests/session_v3
```

各月は月初より前のデータだけで学習します。主指標は`top1.net_mean_pct_at_cost`です。top2は各50%固定で、未約定枠を約定銘柄へ事後再配分しません。

## Python API

```python
import pandas as pd

from tse_session_ranker import SessionRanker

sessions = pd.read_csv("inputs/jpx_sessions.csv")["date"]
ranker = SessionRanker()
ranker.download_tdnet(
    "2024-01-04", "2026-07-21", destination="var/tdnet-html"
)
ranker.collect_tdnet("var/tdnet-html", output="var/tdnet.pkl")
ranker.train(
    "var/daily.pkl",
    train_end="2026-07-17",
    tdnet_indexes="var/tdnet.pkl",
    artifact_path="var/models/session_v3.joblib",
    expected_sessions=sessions,
)
result = ranker.predict(
    "var/daily.pkl",
    target_date="2026-07-21",
    tdnet_indexes="var/tdnet-live",
    expected_history_date="2026-07-17",
    expected_sessions=sessions,
    top_k=2,
)
print(result.candidates[["model_rank", "code", "name", "model_score"]])
```

## モデル特徴と候補条件

production互換の価格特徴12個は、直前の日中騰落、日中騰落の5/20/60日平均・勝率、20日標準偏差、夜間騰落の直前値・20/60日平均、夜間と日中の60日相関です。これに上記TDnet特徴6個を加えます。

v0.4の62特徴は`research_features.py`と研究runnerだけで生成します。前場・後場shape、ATR固定変換、横断rank、市場beta・残差、広いTDnetカテゴリと事前値動きinteractionを加えましたが、20bp控除後の優位性が確認できなかったためproductionの`FEATURE_COLUMNS`やartifact schemaは変更していません。

表示対象は、履歴60セッション以上、前日終値100～30,000円、ATR14が0.25～5%、直近20セッションの無約定・始終同値率20%以下です。JPX相場表には出来高・売買代金がないため、0.3.0では注文サイズ用の流動性判定をモデルへ含めていません。

学習と推論では同じ株価調整方式を使ってください。分割・併合を未調整のまま跨ぐとリターンとATRが壊れます。

比較条件、全結果、制約は追記専用の[VALIDATION.md](VALIDATION.md)に記録します。Entryは古い順で、末尾が最新の運用判断です。
