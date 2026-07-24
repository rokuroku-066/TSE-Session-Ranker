# v10 pre-open point-in-time data track

調査日: 2026-07-23  
対象意思決定時刻: 各営業日 08:58:59 Asia/Tokyo  
目的: 寄り前に実在した情報だけを保存し、`始値→終値`の収益率を予測する次期検証基盤を設計する

## 結論

次の検証で最優先すべき新規データは、**08:58 の現物オークション板**と**同時刻の OSE 先物**である。どちらも、その日の寄り付き直前に起きている需給と市場全体の価格発見を直接測る。適時開示のタイトル分類や前日までの日足だけを増やすより、現在の欠落を埋める情報量が大きい。

ただし、正確な過去 08:58 データは無償の公開ページからは復元できない。現実的な順番は次のとおり。

1. 既存の日足・売買代金・売買単位を point-in-time 化する。費用は増えず、実装も小さい。
2. 口座条件を満たす場合、kabuステーションAPI等で **OSE先物と事前に絞った最大50銘柄の現物板を前向き保存**する。
3. TDnetを公式のJ-Quants経由へ移し、業績修正・自社株買いを数値化する。
4. ライセンスされたJapannextデータを得られる場合、まず完結済みの夜間セッション、次に08:58時点の日中セッションを保存する。
5. 前向き標本でオークション特徴量に価値が見えた場合だけ、JPXの有料10本板履歴を購入して過去検証を加速する。

無償・無登録で、全銘柄について正確な過去08:58板、同時刻先物、PTS価格・数量を一括取得できる公式ソースは確認できなかった。公開画面のスクレイピングを本番の取得方法にはしない。

## 既存実装の監査

対象リポジトリには、今回の土台になる実装がすでにある。

| モジュール | 現在できること | 次に直す点 |
|---|---|---|
| `data/preopen.py` | `observed_at <= as_of` の最新スナップショット、気配値、市場注文数量、PTS価格・数量 | 取引所時刻、ローカル受信時刻、計算完了時刻を分離する。取得不能・対象外・取引ゼロを分ける。PTSの市場・セッションを明示する |
| `data/market_context.py` | 08:58:59の締切、タイムゾーン、日付、追記専用を厳格に検証 | 先物限月、ロール規則、参照価格、遅延状態、受信時刻を追加する |
| `data/tdnet.py` | 開示時刻、完全取得日、provenance、カテゴリ、取得開始時刻を管理 | 非公式ミラー依存から公式ソースへ移行し、数値 surprise と同時開示 bundle を保存する |
| `data/jpx.py` | JPX日報の午前・午後・日次OHLC、出来高、売買代金、VWAP、売買単位、最終特別気配を単位付きで正規化 | 当日08:58にはD-1以前だけを使うことを機械的に保証し、実スプレッドと推定代理値を区別する |

`preopen.py` が現在持つ `pts_price=None` は、「PTSで取引なし」「対象銘柄でない」「API障害」「まだ取得していない」のどれか判別できない。このまま欠損補完すると、誤ったゼロや選択バイアスが生じる。

## 取得可能性と時刻の意味

### 1. 08:58 OSE指数先物

[OSEの現行取引時間](https://www.jpx.co.jp/english/derivatives/rules/trading-hours/index.html)では、主要指数先物の日中立会は08:45に始まる。したがって08:58のNikkei 225/TOPIX先物は「寄り前予測値」ではなく、**すでに約13分売買された実価格**である。CFDを代用するより、対象市場と同じ日本時間・同じ取引所の先物を優先する。

保存する生データ:

- 限月別の最終値、最終約定時刻、best bid/ask、累積出来高
- セッション始値
- 前営業日の現物終値と先物清算値
- 限月、満期日、残存日数
- `source_event_at`、`source_received_at`、`computed_at`

派生候補:

- `futures_return_from_prior_cash_close`
- `futures_return_from_day_open`
- bid/ask midpoint と spread
- 08:57:00→08:58:55 の momentum と realized range
- front/next spread、限月交代フラグ
- 現物ギャップから先物リターンを引いた市場調整後ギャップ

取得方法:

| 方法 | 費用・制約 | 過去深度 | 用途 |
|---|---|---|---|
| [kabuステーションAPI](https://www.kabu.com/item/kabustation_api/default.html) | 口座・Windowsアプリ・利用プランが必要。条件を満たせば追加費用なし。登録/PUSHは現状最大50銘柄、PUSHは約400ms間隔と[公式FAQ](https://kabucom.github.io/kabusapi/ptal/faq.html)に記載 | 基本的に現在値取得。過去08:58復元には使えない | 最小費用の前向き収集 |
| [OSEリアルタイムデータ](https://www.jpx.co.jp/english/markets/paid-info-derivatives/realtime/index.html)または認定ベンダー | 契約・利用区分別料金。再配信には別許諾 | 契約次第 | 全銘柄・高品質本番 |
| [OSEヒストリカルデータ](https://www.jpx.co.jp/english/markets/paid-info-derivatives/historical/index.html) / J-Quants DataCube | 有料、商品・期間単位 | 商品カタログと契約期間による | tick/1分で過去08:58を再現 |
| J-Quants個人向け通常プラン | 先物OHLCはPremium対象だが日次。2026年の株式tick/1分追加サービスもデリバティブを対象外と[JPXが明記](https://www.jpx.co.jp/english/corporate/news/news-releases/6020/20260119.html) | プランにより最大約20年の日次 | 08:58特徴量には不足 |

漏洩リスク:

- 当日の日次始値・高値・安値・終値、清算値を08:58特徴量へ入れない。
- 限月を後から出来高最大で選ばない。寄り前に固定したロール規則を版管理する。
- 「15分遅延値」をリアルタイム値として扱わない。
- 取引所時刻が08:58以前でも、ローカル受信が08:58:59を過ぎたレコードは不採用にする。

### 2. 現物の寄り付きオークション板

[TSE現物の注文受付は08:00、午前立会開始は09:00](https://www.jpx.co.jp/english/equities/trading/domestic/01.html)。よって08:58の値は約定価格ではなく、09:00の板寄せへ向かう途中の注文状態である。名称は `predicted_open` ではなく **`indicative_equilibrium_price`** とし、09:00まで変わり得ることをデータ辞書へ明記する。

保存する生データ:

- indicative equilibrium price または配信元が定義する事前気配
- top 1/3/10 の累積買い・売り数量
- 成行買い・売り数量
- best bid/ask、over/under数量
- 特別気配方向、更新時刻
- 08:57:00、08:58:00、08:58:30、08:58:55 の複数スナップショット

派生候補:

- `indicative_gap = indicative_equilibrium_price / prior_close - 1`
- `(buy_qty - sell_qty) / (buy_qty + sell_qty)` を1/3/10本と成行で別々に計算
- 30/60/120秒のindicative price変化、imbalance変化
- price stability、quote age
- 先物で市場調整したindicative gap
- 1単元金額、予想約定数量に対する注文金額比

[FLEX Standard](https://www.jpx.co.jp/english/markets/paid-info-equities/realtime/index.html)は10本気配、10本外累積数量、成行数量等を含み、FLEX MBOは個々の注文・変更・取消・約定を含む。直接受信または情報ベンダー契約が必要で、再配信には制約がある。

過去データ:

- [FLEXヒストリカル](https://www.jpx.co.jp/english/markets/paid-info-equities/historical/01.html)は2011-01-11以降が中心で、2021-05-24以降は受信時刻付きpcapを提供する。公開料金表上、内部利用の全期間はFLEX Standard月額30万円、MBO月額49.5万円、セット月額60万円。直近30日、単月スポットもある。
- [10本板再構築データ](https://www.jpx.co.jp/english/markets/paid-info-equities/historical/02.html)は2026-06-29開始、元データは2025-04-01以降。翌営業日09:30頃までにSnowflakeへ配信し、内部単体利用は月額40万円。過去08:58を比較的扱いやすい形で取得できる。
- [15分遅延株価API](https://www.jpx.co.jp/english/markets/paid-info-equities/realtime/06.html)は、08:58のオークション板を配信しないため代替にならない。

漏洩リスク:

- 実始値で `indicative_gap` を置換しない。
- 08:58:59より後の取消・追加注文、09:00の約定数量を使わない。
- ベンダー独自の「予想始値」は計算定義とモデル版を保存し、取引所由来の値と混ぜない。
- 当日の全スナップショットから後知恵で最良時刻を選ばない。取得時刻グリッドを事前固定する。

### 3. PTS価格・数量

[Japannext PTS](https://www.japannext.co.jp/en/pts)は日中08:20–16:30、夜間17:00–06:00で、連続売買・価格時間優先、指値注文のみである。TSEの板寄せとは仕組みが違う。

二つの独立した特徴群として保存する。

1. `JNX/night`: 前夜17:00から当朝06:00までの完結済みセッション
2. `JNX/day`: 当朝08:20から08:58:59までの部分セッション

保存する生データ:

- venue、session、session start/end、対象となるTSE営業日
- last price、last trade time、best bid/ask
- session open/high/low、volume、turnover、trade count
- 完全取得か、部分取得か、未対応か

[Japannextのサポートページ](https://www.japannext.co.jp/en/support)では、ITCH/GLIMPSEの市場データや過去tickを会員・利用希望者へ提供しているが、契約・問い合わせが必要である。公開サイトの市況表示は遅延し得るため、全銘柄の正確な08:58データ源にはしない。kabuステーションAPIの公式仕様はSOR非対応なので、venue別PTSの代替にならない。

Cboe JapanのPTSは2025-08-29で終了すると[Cboeが公表](https://ir.cboe.com/news/news-details/2025/Cboe-Plans-to-Cease-Japanese-Equities-Operations/default.aspx)している。現行設計でCboe Japanを固定的な入力源にしない。ODX等の別市場を追加するときもvenueを分離する。

欠損規則:

- 完全取得したセッションで約定なし: `last_price=null/missing`、`volume=0/zero`
- API障害: `last_price=null/missing`、`volume=null/missing`、coverage=`unavailable`
- 銘柄非対応: coverage=`not_supported`
- 08:20以降だけ取得した途中セッション: coverage=`partial`

価格0を「取引なし」の番兵値にしてはいけない。

漏洩リスク:

- 翌朝の対象日付と前夜セッションの日付を取り違えない。
- 夜間終値を、日中PTSの08:58値で後から上書きしない。
- 当日16:30までのPTS高値・安値・出来高を08:58特徴量にしない。
- 複数PTSを集約したSOR値を単一venueの値として扱わない。

### 4. 売買代金・出来高・スプレッド・売買単位

この群は既存JPXパーサーを活かせるため、最初に実装する。

保存する生データ:

- D-1までのOHLC、出来高（株）、売買代金（円）、VWAP
- 売買単位、終値時点のtick size
- 取引停止・無約定・特別気配フラグ
- corporate action調整係数と、その公表・有効時刻

派生候補:

- 1単元金額
- 5/20/60日中央値売買代金、出来高、turnover volatility
- 発注金額 / 20日中央値売買代金
- zero-volume率
- Amihud `abs(return)/turnover`
- high-low spread proxy
- 価格/tick、tick/価格

実際のbid/ask spreadは板データがなければ分からない。日足から推定した値は必ず `spread_proxy` と名付け、実測spreadと同じ列へ入れない。

[JPXの過去日報ページ](https://www.jpx.co.jp/markets/statistics-equities/daily/03.html)は古い年まで公開する一方、個別銘柄・日付の利用を想定し、自動取得を控えることと再配布制限を明示している。公開されていることをバルク収集許諾と解釈しない。継続取得は[J-Quants API](https://www.jpx.co.jp/english/markets/other-data-services/j-quants-api/index.html)またはDataCubeを優先する。

J-Quants個人向けの[2025年料金・履歴表](https://www.jpx.co.jp/english/corporate/news/news-releases/6020/20250505-01.html)では、Freeは12週遅延・2年、Lightは5年、Standardは10年、Premiumは全期間（最大約20年）という区分である。最新の料金・対象データは契約前に再確認する。

[TSE内国株は2018-10-01以降100株単位へ統一](https://www.jpx.co.jp/english/equities/trading/domestic/03.html)されたが、ETF、REIT、外国株等を含めて定数100と決め打ちせず、銘柄マスタの有効日付き値を使う。

漏洩リスク:

- D当日のvolume/turnover/high/lowをDの08:58へ入れない。
- 分割調整済み過去値を使う場合、当時未公表のcorporate actionを遡及適用しない。
- 上場廃止銘柄を現在の銘柄マスタで落とさない。
- 無約定日の出来高0とファイル欠損を分ける。

### 5. TDnetの定量 surprise

タイトル分類だけでなく、同じ会計期間・同じ連結範囲の旧予想と新予想を明示的に結ぶ。

保存する生データ:

- 開示日・開示時刻・ローカル受信時刻・文書ID・文書URL
- 対象会計期間、連結/単体、会計基準、通貨、単位
- 売上、営業利益、経常利益、純利益、EPSの旧予想・新予想
- 配当の旧予想・新予想
- 自社株買い上限株数・金額、発行済株式数、取得方法、開始日・終了日
- 同時刻前後の全開示文書をbundleとして保持
- PDF/構造化データのparse versionと抽出信頼度

派生候補:

- 各利益の会社予想改定率と絶対増加額
- 増加額 / 前日時価総額
- 新会社予想の前年実績比
- 配当増加額 / 前日終値
- 自社株買い株数 / 自己株を除く発行済株式数
- 買付上限金額 / 前日時価総額
- 買付開始までの営業日数
- 上方修正と減配、増資、下方修正等のconflict flag

[J-Quants個人向けTDnet追加サービス](https://www.jpx.co.jp/english/corporate/news/news-releases/6020/20260518-01.html)は2026年開始、Light以上への追加月額11,000円で、日中の最新開示と最大5年の履歴をAPI/CSVで提供する。個人利用向けであり、法人・研究利用は契約区分を確認する。

法人向けJ-Quants Proには、[決算短信の構造化数値](https://jpx.gitbook.io/j-quants-pro/api-reference/statements)と[自己株式取得](https://jpx.gitbook.io/j-quants-pro/api-reference/share_buyback_tdnet)のAPIがある。Financial Summaryは2008-07-07以降を中心に提供する。TDnet Snowflakeは準リアルタイムだが、決算集中時には数分から数十分遅れる場合があると[仕様](https://jpx.gitbook.io/j-quants-pro/snowflake/timely_disclosure)が説明している。

したがって `DisclosedTime <= 08:58:59` だけでは不十分で、**実際のAPI受信時刻も締切以前**でなければならない。

アナリストconsensus surpriseは、当時のconsensusスナップショットを別途ライセンス・保存できる場合だけ作る。現在値から過去のconsensusを逆算しない。

## 先行研究から得られる検証仮説

以下は「採用済みの勝てる特徴量」ではなく、検証順位を決める根拠である。

1. Xiao and Yamamoto, *Price discovery, order submission, and tick size during preopen period* は、TSEのpre-open注文に価格発見があり、tick sizeが価格発見速度に関係することを報告している（[Pacific-Basin Finance Journal, DOI 10.1016/j.pacfin.2020.101428](https://waseda.elsevierpure.com/en/publications/price-discovery-order-submission-and-tick-size-during-preopen-per)）。  
   検証仮説: imbalanceの水準だけでなく、08:57→08:58:55のindicative price収束速度、tick単位の変化回数を使う。

2. Yamamoto, *Intraday technical analysis of individual stocks on the Tokyo Stock Exchange* は、板・注文フローimbalanceに短期予測力がある一方、非約定やpicking-offを含めると単純売買戦略の利益にならないことを示す（[Journal of Banking & Finance, DOI 10.1016/j.jbankfin.2012.07.006](https://waseda.elsevierpure.com/en/publications/intraday-technical-analysis-of-individual-stocks-on-the-tokyo-sto/)）。  
   検証仮説: 予測損益と約定可能性を分け、最低単元・売買代金・spread・特別気配を必須の実行コスト特徴量にする。

3. Amihud and Mendelson, *Trading mechanisms and stock returns: An empirical investigation* は、TSEの始値形成における一時的な価格変動・反転を分析している（[Journal of Finance/DOI 10.1016/0922-1425(89)90013-3](https://doi.org/10.1016/0922-1425(89)90013-3)）。  
   検証仮説: 大きなindicative gapそのものを買い材料にせず、先物調整後gapと持続的な買いimbalanceの組合せ、または過熱反転を別モデルで検証する。

4. 東証高速化前後の価格発見を扱う研究も、寄り付き周辺の情報反映速度が市場設計に依存することを示している（[SAFE Working Paper 144](https://econpapers.repec.org/paper/zbwsafewp/144.htm)）。  
   検証仮説: 過去の古い市場制度と現在を無条件にpoolせず、制度変更・tick table変更・システム更新でregimeを分ける。

5. TDnetタイトル埋め込みやLLMによるpost-earnings-announcement driftを扱う国内研究が進んでいる（[JSAI FIN-035 タイトル埋め込み](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/FIN-035/2025_68/_article/-char/en)、[同 LLM/PEAD](https://www.jstage.jst.go.jp/article/jsaisigtwo/2025/FIN-035/2025_157/_article/-char/en)）。  
   検証仮説: タイトルカテゴリを捨てず、構造化surprise・文書bundle・本文embeddingを別チャネルで保持する。ただし08:58までに実際に処理完了した文書だけを使う。

研究上の共通注意は、**統計的な方向予測力と、寄り付きで実行できる純損益は別**という点である。本プロジェクトでは、最終目的変数を勝率だけでなくコスト控除後 `open_to_close_return` とし、候補数・非約定・特別気配も同じwalk-forward期間で評価する。

## 推奨するデータ契約と取得順

「無料」は無条件の意味ではなく、口座・プラン・利用規約を満たしたとき追加料金がないものを含む。下表は法律意見ではなく、公式ページから確認できた技術・商用条件の整理である。

| 優先 | データ | 最小の取得経路 | 増分費用の目安 | 歴史検証 | 判断 |
|---:|---|---|---:|---|---|
| 0 | D-1日足・出来高・売買代金・単元・tick | 既存JPXパーサー、継続はJ-Quants | 既存契約内 | 十分 | 今すぐpoint-in-time化 |
| 1 | 08:58 OSE先物 | 条件を満たす証券口座API | 条件付き0円 | なし | 前向き収集開始 |
| 1 | 08:58 TSE板 | 同APIで事前ランキング上位最大50銘柄 | 条件付き0円 | なし | 4時点を保存 |
| 2 | TDnet公式文書/構造化値 | 個人TDnet追加またはPro | 個人追加11,000円/月、Proは見積 | 個人最大5年、Proは長期 | 非公式ミラー置換 |
| 3 | Japannext night/day | Japannextまたは認定ベンダー | 要問い合わせ | 要問い合わせ | 契約できる場合のみ |
| 4 | TSE 10本板履歴 | JPX Snowflake | 40万円/月から | 2025-04-01以降 | 前向き結果を見て購入判断 |
| 5 | FLEX/OSEフルfeed | JPX/ベンダー | 契約・用途別 | 商品別 | 本番規模が必要になってから |

## 固定する取得プロトコル

対象日Dごとに以下を追記し、過去行を更新しない。

1. D-1の市場終了後に、D-1までの日足・単元・tick・corporate actionを確定する。
2. 前夜17:00–当朝06:00のPTS夜間セッションを、終了マーカー受信後に保存する。
3. 08:57:00、08:58:00、08:58:30、08:58:55に、同じ候補集合のOSE先物とTSE板を取得する。
4. 08:58:59に入力ledgerをfreezeする。
5. それ以前に公表されていても、受信・parse完了が08:58:59を過ぎたTDnet文書はDの推論から外す。
6. 09:00以降の実始値・約定・終値は別のlabel tableへ追記し、feature tableを更新しない。

推奨freshness:

| 群 | 推論時の最大age | 補足 |
|---|---:|---|
| OSE先物 | 5秒 | 最終イベント時刻と受信時刻を両方保持 |
| TSEオークション板 | 5秒 | 08:58:55付近のスナップショット |
| PTS day | 15秒 | 部分セッションであることを明示 |
| PTS night | セッション終了確認 | 最終約定が古くても「終了済み完全取得」と区別 |
| 日足流動性 | 直前の有効営業日 | source session dateがD未満であることを検証 |
| TDnet | D-1 15:30以降等の事前固定窓 | published/received/computedの全てが締切以前 |

## 添付スキーマと検証コード

- `research/model_v10_preopen_schema.json`: `preopen-pit-v1` のJSON Schemaと単位付きfield catalog
- `src/tse_session_ranker/data/preopen_pit.py`: タイムゾーン、締切、受信時刻、計算完了時刻、freshness、追記専用identity、missing/zeroを検証
- `tests/test_preopen_pit.py`: 21ケース

主要な不変条件:

```text
available_at = max(source_received_at, computed_at)
available_at <= 08:58:59 Asia/Tokyo

source_event_at <= source_received_at <= computed_at

no_activity を主張できるのは coverage_status == complete の場合だけ

価格:
  取引なし -> null / missing
  0円       -> 不正

出来高:
  完全取得かつ取引なし -> 0 / zero
  API障害              -> null / missing
```

この実装は取得アダプターではなく、今後どのベンダーを選んでも同じpoint-in-time規則へ正規化する最小参照実装である。外部契約、購入、ログイン、公開システムへの書き込みは行っていない。

## 次のPRへ入れる最小範囲

1. このledgerメタデータを既存 `preopen.py` / `market_context.py` の共通型として導入する。
2. 既存JPX日足から、D-1限定の流動性特徴量を作る。
3. `PreopenProvider` インターフェースを作り、fixture providerで時刻・欠損テストを先に通す。
4. 実口座・契約情報をリポジトリへ入れず、ローカルadapterでOSE先物1～2本と候補最大50銘柄を前向き保存する。
5. 20独立取引日または30候補日までは特徴量定義を変更せず、coverage率、遅延、欠損、候補数を先に監査する。
6. 収集品質が90%以上になってから、purged walk-forwardで増分損益を検証する。

この順序なら、高額な板履歴を購入する前に、データが本当に毎朝揃うか、オークション特徴量が既存モデルへ追加価値を持つかを前向きに判定できる。
