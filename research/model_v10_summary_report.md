# v1.0 ゼロベース検証・先行研究統合レポート

## 結論

目的は、08:58:59 JSTまでに利用可能な情報だけで、当日の
始値→終値のコスト控除後損益が高い銘柄を毎日1～2件選ぶことである。

先行研究からの12仮説、13種のモデル構造、16種の方策・ユニバース、
TDnetタイトル10仮説、外部市場8仮説、オンラインexpert、
market/peer residualを、月次walk-forwardと20/40/60bpのコストで
検証した。**productionを置き換えられる新仕様は0件**だった。

点推定首位は、TDnet開示タイトルの文字2～5gram TF-IDFから
始値→終値の値幅をRidgeで予測し、開示銘柄だけから毎日1件を選ぶ
`T02_char_value_event_only`である。88 source-complete営業日の成績は
net20 `+0.7191%`、net40 `+0.5191%`、net60 `+0.3191%`だった。
ただし5か月中プラスは3か月、好成績20日除外後のnet20は
`-0.5721%`、上位利益10コードを現金化すると`-0.1085%`である。
L4比のpaired 90%区間は`[-0.0213%, +1.0142%]`、候補familyの
reality-check p値は`0.2103`だった。

したがってT02は、**現在の暫定1位だが、実発注仕様ではなく、
条件を変えない前向きexploratory shadow**としてだけ固定する。
同じ88日でn-gram、閾値、Ridge係数を再調整しない。

## 固定入力と評価

```text
panel:             1,524,104行、4,124銘柄、386営業日
panel SHA-256:     6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb
score期間:         2024-07-01～2025-07-31、266営業日
walk-forward:      月次expanding、対象月より前だけで学習
候補数:            毎日top1またはtop2
損益:              始値→終値、未約定枠は現金
コスト:            往復20/40/60bp
fragility:         月別、3期間、好成績20日除外、利益上位10code現金化
権限:              retrospective research only、production昇格不可
```

## 先行研究から立てた仮説

| 研究上のメカニズム | 今回の実装 | 実測判断 |
|---|---|---|
| TSEの寄り付き価格誤差は日中に修正される | H01・大幅安/高の非対称hinge | top2 net20 `+0.1132%`、対照差`-0.1700pt`で棄却 |
| 日本株の大幅安後反発、短期日中反転 | H01/H02・negative shock、5日OC反転×activity | どちらも対照より悪化 |
| overnightとintradayの投資家層の綱引き | H03・joint sign rate | top2 net20 `+0.1498%`、対照差`-0.1334pt` |
| 個人投資家attentionによる寄り付き過大評価 | H04・overnight×ATR×activity | top2は悪化。top1だけ`+0.2185%`だがtail/code/FW不合格 |
| 同日発表が多いと情報処理が遅れる | H06・TDnet市場混雑×方向 | TDnet対照より`-0.0527pt` |
| 金曜・休日越し発表は反応が遅れる | H07・開示曜日/age×方向 | TDnet対照より`-0.0911pt` |
| 利益と配当など同方向情報の裏付け | H08・directional corroboration | `+0.0129pt`だけ改善したがFW下限・集中耐性不合格 |
| 決算発表順序・announcement wave | H09・10営業日gapの波 | 完了済み過去waveが0件でfeasibility failure |
| daily topを直接扱うLearning-to-Rank | M01/M02/M06/M07 | 最良M02でもnet40 `-0.0658%`、全13モデル中合格0 |
| signal decayと取引費用を直接扱う | utility、hurdle、expectile、online expert | magnitude/utility学習とonline O01～O03は全て対照未満 |
| 日本語TDnetテキストは短期反応を補足する | char TF-IDF rank/value、構造化タイトル | rank/overlayは失敗。value event-only T02だけ点推定首位 |
| 数値surpriseと文脈情報の組合せ | 現履歴ではタイトルのみ。PDF本文は未取得 | 次のデータ仮説として採用、今回の有効性主張はしない |

先行研究は仮説の出発点であり、日本株の今回の標本での有効性を保証する
ものではない。一次資料と12仮説の定義は
`model_v10_literature_review.md`と
`model_v10_literature_hypotheses.json`に保存した。

## ゼロベース検証の比較

数値は日次平均%。異なる日数のTDnetを266日trackと直接順位比較しない。

| Track | 代表候補 | 日数 | k | net20 | net40 | net60 | tail/監査判断 |
|---|---|---:|---:|---:|---:|---:|---|
| G0対照 | daily-rank Ridge | 266 | 2 | +0.2553 | +0.0587 | -0.1379 | best20除外`-0.0054` |
| モデル構造13種 | M02 extreme-gain Ridge | 266 | 2 | +0.1305 | -0.0658 | -0.2620 | 0/13合格 |
| 文献価格block | H04 attention | 266 | 2 | +0.1810 | -0.0163 | -0.2137 | 対照差負 |
| 方策・universe | H12 momentum分散 | 266 | 2 | +0.3044 | +0.1082 | -0.0881 | FW下限負、Round 2判定撤回 |
| gen1部分実行 | Z06 exp-decay top1 | 266 | 1 | +0.3085 | +0.1138 | -0.0810 | 18登録中6実行のため選定不可 |
| 外部市場 | X06 context Ridge top1 | 266 | 1 | +0.2623 | +0.0645 | -0.1332 | robustness出力不足、tail負 |
| online expert | O03 follow-leader | 266 | 2 | +0.0823 | -0.1135 | -0.3094 | 棄却 |
| market residual | R01 top1 | 266 | 1 | +0.0390 | -0.1595 | -0.3580 | 棄却 |
| peer residual | Z17 top1 | 266 | 1 | -0.2964 | -0.4964 | -0.6964 | 0/13 net40月、棄却 |
| **TDnet text** | **T02 char-value top1** | **88** | **1** | **+0.7191** | **+0.5191** | **+0.3191** | exploratory forwardのみ |

peer residualは8 peer群を毎foldの過去情報だけで再構成し、同日peer平均を
引いたtargetを学習した。target-day outcome mutation、全13foldのPIT、
独立損益再計算は合格したが、top1/top2ともnet20から負であり明確に棄却
した。

## 独立監査で訂正したこと

全trackの保存損益を独立再計算し、最大誤差は`2.22e-16`だった。一方、
次の手順上の問題を検出した。

1. Policy Round 2は「無関係な5比較中4勝」を要求したのに4比較しか
   実装せず、4/4を合格にした。`retain_for_forward_shadow=true`は撤回。
2. root gen1は18登録中6件だけの部分実行。Z06を選定仕様にできない。
3. external contextは月次vector、source-date mutation、familywise下限が
   未出力。X06は点推定以上に扱わない。
4. policy H01/H04、TDnet T02/T10-top1は完全に同じ選択であり、独立した
   再現例として数えない。
5. TDnet 88日は不連続な5か月で、過去HTMLの実観測時刻sidecarがない。

訂正後も、production置換は0件である。詳細は
`model_v10_integration_audit_report.md`を正とする。

## 暫定shadow仕様

```text
candidate_id:
    v10_t02_char_value_event_top1

source:
    08:58:59 JSTまでの公開時刻を持ち、
    source-completeと判定できるTDnet開示だけ

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
    さらに[-10,+10]へclipした始値→終値%

model:
    Ridge(alpha=20)

decision:
    value予測降順、同点は銘柄コード昇順
    event銘柄のtop1を1件
    候補がなければ現金。価格モデルによる後付けfallbackなし
```

これは予測勝率ではなく、right tailを含む平均損益を狙う仕様である。
実測勝率は47.7%、中央値は負でも平均損益は正だった。ただし右裾依存が
強いため、自動発注せず次の条件を満たすまでshadow表示だけにする。

```text
・連続したsource-complete 120営業日以上、4か月以上
・net40が前半・後半ともプラス
・L4比paired片側90%下限が0以上
・best20日除外後と上位利益10code現金化後のnet20がプラス
・候補生成/PIT遵守率98%以上
・08:58板、実spread、売買代金、単元で発注可能性を別判定
```

## 次に増やす情報

既存日足へモデルや閾値を追加する探索は停止する。次の改善単位は新しい
point-in-time情報である。

1. 08:58のOSE先物騰落率、basis、直近13分の方向・出来高
2. TSE寄り板の予想約定値、1/3/10本imbalance、市場注文差、spread
3. PTS価格・出来高・売買代金
4. 過去20日売買代金、出来高、単元金額、tick、実slippage
5. TDnet PDF本文の旧予想・新予想・増減額・時価総額比
6. 月次売上の前年比・会社自身の直近trend、決算数値と文脈の不一致

`preopen_pit.py`と`model_v10_preopen_schema.json`は、取引所時刻と
ローカル受信時刻・計算完了時刻を分離し、
`available_at=max(received_at, computed_at) <= 08:58:59`を強制する。
欠測、真の0、対象外、source不完全も別状態で保存する。

## 最終判断

```text
production変更:                 なし
暫定retrospective首位:         T02 char-value event-only top1
用途:                           固定した前向きexploratory shadow
毎日の候補数:                   原則1件、eventなしは0件
同じ履歴での追加チューニング:   停止
次の検証:                       08:58データとTDnet本文の前向き収集
```

今回確定したのは「完成した売買モデル」ではなく、**同じ日足だけでは
モデル複雑化に改善余地がなく、TDnet本文と寄り前microstructureへ
情報空間を移すべきこと**である。
