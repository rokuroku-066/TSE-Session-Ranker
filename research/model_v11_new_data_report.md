# model_v11_new_data — zero-base information-space result

実行日: 2026-07-23  
判定: **本番採用不可。metadata-only仮説群を棄却し、微調整を停止する。**

## 何をゼロから検証したか

`model_v11_new_data_protocol.json` を候補損益を見る前に固定し、タイトル埋め込みや既存価格特徴の微修正ではなく、11個のデータ生成仮説を登録した。

1. 同一期間・同一連結範囲の旧予想→新予想
2. 利益増分 / D-1時価総額
3. 自社株買い上限株数・金額によるbuyback pressure
4. 同時開示された悪材料
5. 発表時刻・決算四半期・会計期
6. PDF数表の形状・数値密度
7. 本文の否定、据置、撤回、中止、単発要因
8. 08:58 OSE先物
9. 08:58 TSE寄り板
10. PTS夜間・08:20以降の日中セッション
11. D-1売買代金、出来高、単元、tick、実行コスト

ローカルでPITに近い形で実装できたのは、TDnet indexに存在する `published_at`、タイトル、URLから作る「同時悪材料bundle」と「発表時刻・決算期」だけだった。これらは本文・数値そのものではなく、既存タイトル情報の構造化metadataである。新しい情報量が限定的であることを明示した上で、固定Ridge、月次walk-forward、hyperparameter searchなしで検証した。

## データ棚卸し

- 補正済みpanel: 1,524,104行、4,124コード、2024-01-04〜2025-07-31。
- TDnet index cache: 218ページ、50,719開示、50,719一意URL。公開時刻・コード・社名・タイトル・PDF URLはある。
- TDnet PDF本文/XBRL/構造化数値: ローカル0件。URLを本文扱いしていない。
- JPX月次PDF: 2024-01〜2025-07の19か月。ただしAM/PM OHLCのみで、出来高、売買代金、単元、板、先物、PTSはない。
- TDnet strict-source-complete session: 134日。2024-08〜2025-03の159 panel sessionは完全にcache gapで、0件開示として補完していない。
- 早期未使用期間: 2024-05〜06にstrict-complete 33日、6,935 bundle。固定最小学習量を満たしたwalk-forward scoreは2024-06の19日。
- 全score: 107日（2024-06、2024-07、2025-04〜07）、18,054 event bundle。
- publication cutoff違反0、prior-close以前への誤割当0。

完全な棚卸しと必要契約は `model_v11_new_data_inventory.json`、前向き取得仕様は `model_v11_new_data_collector_spec.json` に保存した。

## walk-forward結果

数値は日次平均リターン、単位は%ポイント。`net20/40/60` はそれぞれ往復20/40/60bp控除後である。イベント候補が足りないslotはcashとする固定ルールだが、今回の3候補は全score日でslotが埋まった。

| policy | top | net20 | net40 | net60 |
|---|---:|---:|---:|---:|
| L4 price control | 1 | +0.262 | +0.064 | -0.134 |
| L4 price control | 2 | +0.321 | +0.124 | -0.072 |
| 同時悪材料bundle | 1 | -0.307 | -0.507 | -0.707 |
| 同時悪材料bundle | 2 | -0.228 | -0.428 | -0.628 |
| 発表時刻・決算期 | 1 | -0.030 | -0.230 | -0.430 |
| 発表時刻・決算期 | 2 | -0.196 | -0.396 | -0.596 |
| 固定結合 | 1 | -0.267 | -0.467 | -0.667 |
| 固定結合 | 2 | -0.340 | -0.540 | -0.740 |

早期未使用2024-06のnet40も候補はすべて負だった。

- 同時悪材料bundle: top1 -0.171、top2 -0.187
- 発表時刻・決算期: top1 -0.505、top2 -1.200
- 固定結合: top1 -0.311、top2 -1.089

後続期間でも3候補はtop1/top2とも負で、期間を変えて方向が救済されなかった。

## familywise・tail・code集中

3候補 × top1/top2の6検定を一つのfamilyとし、L4との差を日次paired two-sided test、Holm補正で評価した。

- 有意な上振れは0件。
- top2は3候補すべてL4より悪く、Holm補正後も負方向の差が残った。
  - 同時悪材料bundle: 差 -0.552%pt、補正p=0.0420
  - 発表時刻・決算期: 差 -0.520%pt、補正p=0.0420
  - 固定結合: 差 -0.664%pt、補正p=0.0111
- top1は補正後有意ではないが、差は全候補で負（-0.295〜-0.571%pt）。
- top20勝ち日除外後net20は、全候補・L4とも負。
- 上位利益10コードをcash置換したnet20も、全候補・L4とも負。
- したがって、L4の正の全期間平均もtail/code集中を通過せず、本番根拠にはならない。

`model_v11_new_data_audit.py` はrunnerをimportせず、保存picksからslot gross、約定slotだけのコスト、top1/top2の等ウェイト、cash、月別、期間別、tail、code置換、familywiseを再計算した。408個の数値・scalarチェックと8個の構造groupを `1e-12` で照合し、passした。

## 利用不能データをfail-closedにした理由

| 仮説 | 現状 | 取得に必要なもの |
|---|---|---|
| 旧→新予想、PDF数表、本文否定 | URLのみ。PDF本文・構造化セル・receipt sidecarなし | J-Quants TDnet add-on/Pro、またはimmutable PDF archiveとversioned parser |
| 利益増分/時価総額 | 上記に加えPIT発行済株式・自己株式なし | structured forecast + effective-dated security master + D-1 close |
| buyback pressure | タイトルflagのみ。株数・金額・方法・期間なし | Pro buyback fieldsまたはPDF抽出 + capital structure |
| 08:58先物 | exact snapshotなし | OSE entitlement付きbroker APIの前向き保存、またはlicensed historical feed |
| 08:58寄り板 | pre-open snapshotなし | broker APIで固定候補を前向き保存、またはJPX 10本板再構築契約 |
| PTS | venue/session別履歴なし | Japannext ITCH/GLIMPSEまたは認定vendor契約 |
| 実流動性・単元・cost | OHLCのみ | J-Quants/DataCube + effective-dated master + 実broker execution ledger |

実始値を08:58気配へ置換、日次先物OHLCを08:58先物へ置換、欠損を取引ゼロへ置換、PDF URLを本文へ置換、といった代理はすべて禁止した。

## 結論と次のゼロベースloop

同時悪材料・発表時刻・決算期だけでは、20/40/60bp、top1/top2、未使用早期期間、後続期間、tail、code集中、familywiseのどこでも本番候補にならなかった。このlineは追加の閾値調整やRidge alpha調整をせず終了する。

次の検証は同じmetadataの微調整ではなく、`model_v11_new_data_collector_spec.json` に従って実際に新しいraw informationを作る段階である。

1. 20 sessionのdata-quality pilotをlabels sealedで実行する。
2. TDnet official numeric/body、D-1 liquidity/master、08:58先物、08:58板、PTSをappend-onlyで保存する。
3. cutoff-valid coverage 98%以上、post-cutoff違反0、raw hash/replay 100%を確認する。
4. その後に別protocolで少なくとも120 session・6か月のforward testを固定する。
5. 独立P&L、実spread/slippage、top-day、top-code、familywise、operational replayをすべて通るまでproduction promotionを禁止する。

現時点で完成したのは「本番モデル」ではなく、負の仮説検証結果と、次の本当に新しい情報空間を作るための取得仕様である。
