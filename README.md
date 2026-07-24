# TSE Session Ranker

寄り前時点で東証普通株を順位付けし、当日の**始値→終値のコスト控除後損益率**を最大化する候補を毎日1～2件出すPythonパッケージです。

現行CLI/APIの既定モデルは0.3.0互換の`session_v3_tdnet_clear`です。正則化ロジスティック回帰に、価格履歴12特徴と寄り前までのTDnet適時開示6特徴を入力し、1位を`CORE`、2位を`RESERVE`として表示します。

> [!WARNING]
> **自動発注には使用できません。** v1.1では情報源・推定対象・意思決定構造が異なる7系統をゼロベースで検証しましたが、118 variantの本番gate通過は0件でした。凍結T02のOOT rawと08:58実行証拠も未充足です。productionモデルは変更しておらず、CLIの既存`CORE` / `ORDER_ELIGIBLE`も研究上の発注承認を意味しません。

## 最新の研究判断

| 仕様 | 特徴・手法 | 用途 | 20bp後結果 |
|---|---|---|---:|
| v0.3既定 | 価格12 + TDnet 6、logit | CLI/APIのshadow表示 | 既知診断top1 `-0.087886%/日` |
| v0.4固定候補 | 価格・前後場・市場・TDnet 62、logit | 負のresearch baseline | 既知診断top1 `-0.086199%/日` |
| v0.5選定winner | 価格12 + TDnet 6、raked logit | retrospectiveで棄却 | 確認期top2 `-0.484256%/日` |
| v0.5 TDnet診断対照 | 価格・前後場・市場・TDnet 62、raked logit | 後付け採用禁止 | 確認期top2 `+0.039661%/日` |
| v0.6一次首位 | 価格core + 前後場shape、raked logit | 次期間で棄却 | screen top2 `-0.218990%/日` |
| v0.6最終lock | 価格coreのみ、raked logit | 損益優位性未実証 | stability top2 `-0.421878%/日` |
| v0.6.2訂正T1 | 価格core + TDnet開示構造 | 整列訂正後も棄却 | screen top2 `-0.565582%/日` |
| v0.7 meta-gate | G0 top2の各50%枠をOOF発注判定 | 4 gateすべて不合格 | `locked_gate = null` |
| v0.7.1 tail veto | 5 tail条件で各枠を現金化 | 損失軽減の後付け観察のみ | net20 `+0.005473%/日`、net40 `-0.084087%/日`（不合格） |
| v0.8 G0対照 | 価格core 15、同日順位Ridge | retrospective対照 | net20 `+0.255321%/日`、net40 `+0.058705%/日` |
| v0.8 L4平均役 | G0 + `flat_oc_rate_20`、同日順位Ridge | 固定shadow候補 | net20 `+0.283192%/日`、net40 `+0.086575%/日` |
| v0.8 L6頑健役 | G0 + 売買不能・横ばいproxy 4列、同日順位Ridge | 固定shadow候補 | net20 `+0.259376%/日`、net40 `+0.068023%/日` |
| v0.8 TDnet | fresh/follow-up・時刻・bundle等12群 | 追加採用0群 | source-complete `187/266日`、全群不合格 |
| v0.9 L4 rank 2 | L4のscore順位2位へ100% | 1銘柄shadow対照 | net20 `+0.425289%/日`、net40 `+0.229048%/日` |
| v0.9 25/75 | L4 rank 1/2へ25%/75% | 分散shadow対照 | net20 `+0.354240%/日`、13/13か月プラス |
| v0.9 valid-OC breadth | 前営業日の始値→終値breadthでL4/L6 rank 2を切替 | retrospective predecessor | net20 `+0.487187%/日`、net60 `+0.093202%/日` |
| **v1.0 T02 TDnet text** | 開示タイトルchar 2–5gram TF-IDF + value Ridge | **前向きexploratory shadow・実発注不可** | 88日でnet20 `+0.719059%/日`、net40 `+0.519059%/日`、net60 `+0.319059%/日` |
| **v1.1 zero-base audit** | 7機構family、59 spec・118 capacity variant | **全件棄却・本番候補0** | family gate `0/118` |

v0.8では価格・市場状態、売買可能性、TDnet、非線形変換の計43特徴仮説と5ユニバース仮説を評価した。v0.9では有力仕様の誤差から、個別誤差/meta 20件、target・portfolio 33件、breadth反証4件の計57件を追加検証した。v1.0では19本の一次資料から12仮説を事前登録し、13モデル構造、16方策・ユニバース案、TDnetタイトル10案、外部市場8案、online expert、market/peer residualを月次walk-forwardで比較した。v1.1では67の概念仮説をnew data、historical analog、distributional decision、uplift、cross-stock graph、distribution shift、calendar/institutionへ分け、58実装可能仮説と1固定combinationを118 capacity variantとして反証した。

独立監査の結果、v0.9 breadthの追加価値は未証明であり、v1.0のpolicy Round 2も「5比較中4勝」のはずが4比較しか実装されていなかったため昇格判断を撤回した。点推定首位のT02も、好成績20日除外後net20 `-0.572088%/日`、上位利益10コードを現金化すると`-0.108484%/日`、familywise reality-check `p=0.2103`である。v1.1でもfamily gate通過は0件だったため、新しいproduction採用は0件。全経緯は追記専用の[VALIDATION.md](VALIDATION.md)、v1.1の横断判断は[integration report](research/model_v11_integration_report.md)、入力不足は[data-readiness report](research/model_v11_production_readiness_report.md)、条件を変えないT02の次回評価は[OOT protocol](research/model_v11_t02_oot_protocol.json)に保存する。

### v1.1ゼロベース検証

```text
conceptual hypotheses:                 67
executable hypotheses + fixed combo: 58 + 1
capacity variants:                    118
family-local unique selections:       112
family gate passers:                  0
forward finalists:                    0
production candidates:                0
```

7系統は同じ微調整の枝ではなく、異なる情報・目的変数・選択機構から作った。評価窓はstrict-source 88日、new-data 107日、price-panel 266日で、universe、cash denominator、controlも異なるため、点推定だけの横断順位は作っていない。distributional、shift、calendarにはnet40が正の代表仕様もあったが、familywise下限、tail除外、期間slice、銘柄集中のいずれかを通過できなかった。

凍結T02を変えずに評価する2025-08-04～2026-03-31のOOT protocolも登録した。ただし現在のworkspaceで再現可能なrawはJPX `0/160`、TDnet `0/243`、joint provenance-completeはminimum `0/120`である。data-readiness verifierは20件の重複し得るblocking requirementと、先物・PTS・出来高の任意research context 3項目を分離し、ファイル名やheaderだけを証拠にしない。registry・parser audit・source manifestをhash固定し、JPX raw再parse、TDnet page/meta照合、T02 decisionの独立replay、全注文結果の完全被覆、08:58 bid/ask・tick/lot・spread/slippage再計算、予定額と実約定額双方の日次合計capacityまで通らなければ入力readyにしない。自己申告auditは内部整合性までしか通さず、dated security master・価格履歴からの参照値再計算と外部認証がない限りexecution evidenceをvalidにしない。hashは内部整合性を示すだけで外部真正性や本番認可を示さず、登録期間もgenuinely untouchedではない。したがって現時点の判断は`orders_allowed=false`で、既存package/productionモデルは変更しない。

### v1.0暫定shadow仕様

```text
08:58:59 JSTまでに公開・受信・計算が完了し、
source-completeと判定できるTDnet開示だけを使用

同一銘柄の対象タイトルを公開時刻順に連結
char 2–5gram TF-IDF:
    min_df=3, max_features=30000, sublinear_tf=True, norm="l2"
value target:
    学習期間の1/99 percentileでclip後、[-10,+10]%へclip
model:
    Ridge(alpha=20)
decision:
    予測値降順、同点は銘柄コード昇順のtop1
    eventなし・source欠落は現金、価格モデルfallbackなし
```

これは勝率ではなく平均損益を狙うため、88日のnet40勝率は47.7%、中央値は負でも平均が正になった。ただし右裾依存が強い。最低120 source-complete営業日・4か月、前後半net40正、L4比の片側90%下限非負、tail/code除外後も正、PIT遵守率98%以上、実spread・slippage・最低単元の合格をすべて満たすまで実発注へ昇格させない。

### v0.9暫定shadow仕様

```text
prior_market_oc_breadth =
    前営業日の traded & outcome_observed & source_complete 銘柄について
    mean(open_to_close_return > 0)

prior_market_oc_breadth < 0.50:
    L4のscore順位2位を1銘柄
otherwise:
    L6のscore順位2位を1銘柄
```

表示銘柄へstrategy sleeveの100%を配分する評価で、2024-07～2025-07のnet20は`+0.487187%/日`だった。ただし既知期間のposthoc結果であり、best 20日または上位利益10コードを除くと余裕は小さい。並走対照として、L4 rank 1/2の50/50、L4 rank 2単独、L4 rank 1/2の25/75を同時保存する。breadth閾値・rank・配分はforward中に変更しない。

### v0.8固定shadow仕様

学習日`d`内の実現始値→終値リターン順位を、次の目的変数へ変換します。

```text
y(i, d) = 2 * percentile_rank_d(open_to_close_return_i) - 1
```

この既存評価式は有限の横断銘柄数`n`では最小値が`2/n - 1`、tieがなければ日次平均が`1/n`で、厳密なゼロ中心ではありません。過去結果との同一性を保つため式は変えず、今後の仕様比較では別candidateとして扱います。

Ridgeのlossでは各学習日の重み合計を1にそろえ、欠測は学習期間中央値と欠測indicatorで処理し、標準化後に`Ridge(alpha=1)`を月次で再学習します。再現対象のimputerとscaler自体は行等重みでfitされるため、pipeline全体が日付等重みという意味ではありません。対象月の行は学習へ入れず、score降順・銘柄コード昇順で毎日2件を50%ずつ表示します。L4はG0 15列へ`flat_oc_rate_20`だけを追加し、L6はさらに`no_trade_rate_20`、`no_trade_rate_60`、`zero_range_rate_20`を加えます。未約定枠は現金です。

L4・L6とも60bpコストではマイナスで、L4は単一銘柄が総損益の24.53%を占めます。また各proxyの係数は概ね正であり、「流動性の低い銘柄を避けるpenalty」ではありません。正確な出来高・売買代金・単元・spread・08:58板がないため、現段階の用途は毎日1～2件のshadow候補生成だけです。

## 設計の要点

- production互換の既定推定器は`LogisticRegression(C=0.08, class_weight="balanced")`で固定
- v0.4研究winnerも同じL2正則化logitで、`C=0.03`、日別・クラス均衡、expanding学習窓
- v0.5研究ではモデル固定を外したが、選定winnerが確認期で再現せず、production推定器は変更なし
- v0.6では特徴候補を先に7群へ固定し、共通モデルのgroup ablationで追加寄与を分離。次期間まで残った追加群は0
- v0.7では表示top2を固定し、モデル間合意・スコア形状・downsideと、事前日足tailを発注gateとして分離。未合格枠はrank 3で置換せず現金
- v0.8ではモデル固定を外して17 target/model案を比較し、日内の市場共通変動を落とす同日順位Ridgeを固定shadowへ採用。追加特徴のproduction採用は0群
- v0.9では57件を追加反証し、`prior_market_oc_breadth`でL4/L6のrank 2を切り替える1銘柄規則を前向きshadowへ固定。production採用は0
- v1.0では先行研究・モデル構造・方策・外部市場・TDnet text・online/residualをゼロベース比較。T02 text-value top1を前向きexploratory shadowへ固定したが、tail/code集中と多重性のためproduction採用は0
- v1.1では情報源・推定対象・意思決定構造の異なる7 familyを独立protocolで反証。118 variantのgate通過0、非互換窓の横断順位なし、fresh OOTと実行証拠が揃うまでproduction採用0
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

v0.7研究用には、開示観測とfresh材料、初回予想と予想修正、株主配当と受取配当、自己株買い・エクイティ・M&Aのfresh/follow-upを分離する`tdnet_v07_*` 26列を追加しました。旧`tdnet_clean_*`は凍結成果物再現のため意味を変えません。v0.8では新26列と既存の時刻・bundle列を12群へ分けて検証しましたが、source-completeは187/266日で、追加採用は0群でした。production 6特徴は変更していません。

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

板は`model_score`と`model_rank`を変更しません。欠測・3分超古い、暫定ギャップがATR14の+2倍以上、買い特別気配、予想寄りが9:05より後、または`RESERVE`の場合だけ`DISPLAY_ONLY`にします。`ORDER_ELIGIBLE`は既存の板品質・legacy vetoだけの状態名であり、期待損益gateの合格を示さないため、発注には使用できません。

## 6. 先物contextの前向き収集

日経225先物・TOPIX先物の寄り前騰落率は、次の前向き検証候補です。正確な過去08:58 snapshotを復元できないため、v0.8の順位にも入れていません。日足終値、9:00以後の値、後から確定した値で代用しないでください。

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
