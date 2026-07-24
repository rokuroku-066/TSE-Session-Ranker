# Model v10 ゼロベース仮説のための先行研究調査

## 1. 目的と結論

目的は、**寄り前 08:58:59 JST までに既知の情報だけを使い、当日の始値→終値リターン（OC）を最大化する1～2銘柄を選ぶこと**である。

既存の L4/L6 の順位や breadth 閾値を調整するのではなく、一次資料から別の経済メカニズムを起点にした。現在のリポジトリで後ろ向き検証できる新規候補を12件登録する。

優先順位は次のとおり。

1. **H01～H05：価格形成メカニズム**  
   東京市場固有の寄り付き価格誤差、短期反転、夜間と日中の投資家層の違いを直接扱う。
2. **H10～H12：目的関数とモデル構造**  
   全銘柄の符号や平均二乗誤差ではなく、「毎日の上位1～2件のコスト控除後利益」を直接学習する。
3. **H06～H09：TDnetの注意配分と情報構造**  
   単純な材料カテゴリではなく、混雑、曜日、裏付けの一致、決算発表順序を使う。

これは先行研究から導いた**事前仮説**であり、日本株の現在の標本で有効だと確認した結果ではない。特に海外市場の結果を東京市場へ移植できるとは限らない。

## 2. 現在のデータで可能なこと

凍結済みパネル `/tmp/model_v07_corrected_panel.pkl` は1,524,104行、110列で、次を保持している。

- 厳密に前営業日以前の OC、overnight、close momentum、ATR、AM/PM、market state
- no-trade、flat OC、zero range の流動性proxy
- 08:58:59までに観測可能なTDnetタイトル分類、件数、時刻、bundle構造
- `source_complete`、`tdnet_source_complete`、各特徴の最大ソース日時

価格履歴には出来高・売買代金がない。そのため、出来高ショック、Amihud、実スプレッドを必要とする仮説は今回の12件に含めない。TDnetページは既知の完全期間だけを使い、欠落をゼロ件とみなさない。

## 3. 一次資料から得た主要な示唆

### 東京市場の寄り付きと短期反転

- Amihud and Mendelson は、TSEの寄り付き価格には過剰反応があり、その価格誤差が日中に修正されると報告した。別研究でも、朝の板寄せは昼の板寄せよりノイジーで非効率だった。  
  [Market microstructure and price discovery on the Tokyo Stock Exchange](https://doi.org/10.1016/0922-1425(89)90013-3)、[Volatility, Efficiency, and Trading](https://doi.org/10.1111/j.1540-6261.1991.tb04643.x)
- Bremer, Hiraki, and Sweeney は、TSEで大幅下落後には有意な反発がある一方、大幅上昇後には同じ対称的パターンが乏しいと報告した。  
  [Predictable Patterns after Large Stock Price Changes on the Tokyo Stock Exchange](https://doi.org/10.2307/2331204)
- Miwa は、日本株の短期反転が過去の**日中リターン**に由来し、流動性が低い銘柄・市場変動が高い局面ほど強いと報告した。  
  [Short-Term Return Reversals and Intraday Transactions](https://doi.org/10.1142/S2010139219500022)

### overnight と OC は同じリターンとして扱えない

- Lou, Polk, and Skouras は、銘柄レベルでovernight成分とintraday成分がそれぞれ持続する一方、成分をまたぐと反転する「tug of war」を報告した。  
  [A tug of war: Overnight versus intraday expected returns](https://doi.org/10.1016/j.jfineco.2019.03.011)
- Berkman et al. は、個人投資家の注目が強い銘柄ほど寄り付きが押し上げられ、その後の日中に反転しやすく、評価困難・裁定困難な銘柄ほど強いと報告した。  
  [Paying Attention: Overnight Returns and the Hidden Cost of Buying at the Open](https://doi.org/10.1017/S0022109012000270)
- Aboody et al. はovernightリターンの短期持続性と、評価困難銘柄での強い持続性を示し、overnightを銘柄固有センチメントのproxyとして支持した。  
  [Overnight Returns and Firm-Specific Investor Sentiment](https://doi.org/10.1017/S0022109017000989)
- 2026年のRFS論文は、長めの過去**日中**リターンにはmomentumがある一方、overnight成分には同じmomentumが見られないと報告している。  
  [What Drives Momentum and Reversal? Evidence from Day and Night Signals](https://doi.org/10.1093/rfs/hhag036)

### 開示反応は材料カテゴリだけで決まらない

- 同日に他社の発表が多いほど決算への即時反応が弱く、後続driftが強い。  
  [Driven to Distraction](https://doi.org/10.1111/j.1540-6261.2009.01501.x)
- 金曜日の決算は即時反応が15%弱く、遅延反応が70%強い。  
  [Investor Inattention and Friday Earnings Announcements](https://doi.org/10.1111/j.1540-6261.2009.01447.x)
- 利益と配当の同時発表には相互の裏付け効果があり、同方向の発表ほど各情報の信頼性が高く評価される。  
  [Earnings and Dividend Announcements: Is There a Corroboration Effect?](https://www.nber.org/papers/w1248)
- 発表企業にはannouncement premiumがあり、発表順序が早い企業ほど高く、遅い企業ほど低いと報告されている。  
  [Earnings Announcements and Systematic Risk](https://doi.org/10.1111/jofi.12361)
- TDnetは公平・迅速・広範な公表のための公式システムで、開示プロセスと掲載時点が明確である。  
  [JPX: TDnetの概要](https://www.jpx.co.jp/equities/listing/disclosure/tdnet/index.html)

### 毎日1～2件なら、全銘柄の回帰誤差より上位順位が重要

- Poh et al. は、通常の回帰・分類後のソートより、pairwise/listwise構造を直接学ぶLearning-to-Rankがcross-sectional戦略に適すると示した。  
  [Building Cross-Sectional Systematic Strategies By Learning to Rank](https://arxiv.org/abs/2012.07149)
- Zhang, Wu, and Chen は、リスト上下端を重視するlistwise lossを提示した。今回のlong-onlyでは上端だけに限定して使う。  
  [Constructing long-short stock portfolio with a new listwise learn-to-rank algorithm](https://doi.org/10.1080/14697688.2021.1939117)
- Howard は、銘柄群の異質性を無視した単一モデルより、group-specific modelや相対リターン予測が改善につながると報告した。  
  [Choices Matter When Training Machine Learning Models for Return Prediction](https://doi.org/10.1080/0015198X.2024.2388024)
- Gârleanu and Pedersen は、予測リターンの強さだけでなく、signal decayと取引費用を意思決定へ明示的に入れる必要を示した。  
  [Dynamic Trading with Predictable Returns and Transaction Costs](https://doi.org/10.1111/jofi.12080)

## 4. 登録する12仮説

### H01：TSE非対称・大幅下落反発

**着想**  
TSEの大幅下落後反発は、大幅上昇後の反落と対称ではない。既存の線形 `cc_reversal_1_atr` だけではこの非対称性を表現できない。

**特徴**

```text
neg_shock_1 = min(max(-cc_reversal_1_atr, 0), 5)
pos_shock_1 = min(max( cc_reversal_1_atr, 0), 5)
neg_shock_sq = neg_shock_1 ** 2
pos_shock_sq = pos_shock_1 ** 2
neg_shock_x_range = neg_shock_1 * range_shock_1_20
neg_shock_x_market_nonnegative =
    neg_shock_1 * 1(prior_market_cc_return_pct >= 0)
```

**予測符号**  
`neg_shock_1` は当日OCに正で、二乗項は追加的に正。市場全体が下落していない日の個別negative shockでも残る。`pos_shock_1`の絶対効果は小さい。

**反証**  
未使用ブロックでnegative hingeの上位群が、対称線形ベースラインよりnet20を改善しない、またはpositive shockと同程度しか反転しない。

**PIT**  
前営業日のcloseまで。target日のopen、high、low、closeは使用禁止。

---

### H02：週次intraday在庫反転 × 流動性・市場変動

**着想**  
Miwaの日本株結果に合わせ、close-to-closeではなく過去5日間のOCだけを反転signalにする。流動性proxyと市場volatilityで効果を変える。

**特徴**

```text
oc5_reversal = -oc_mean_5
illiquidity_proxy =
    mean(no_trade_rate_20, flat_oc_rate_20, zero_range_rate_20)
market_vol_state = market_cc_vol_ratio_5_20
oc5_reversal_x_illiquidity
oc5_reversal_x_market_vol
```

**予測符号**  
過去5日OCが低いほど当日OCは高い。効果はilliquidityとmarket volatilityが高いほど強い。

**反証**  
符号が逆、相互作用が再現しない、または上位1～2件のtail-removed net20が改善しない。

**PIT**  
すべて前営業日までにshift。no-tradeを欠測の代用にしない。

---

### H03：joint overnight–day tug-of-war状態

**着想**  
平均overnightと平均OCを別々に投入するだけでなく、同一日に「positive overnight → negative OC」が何回起きたかを直接数える。これはopening price pressureが日中に修正される銘柄固有traitである。

**特徴**

```text
posnight_negday_rate_20
negnight_posday_rate_20
tug_balance_20 =
    negnight_posday_rate_20 - posnight_negday_rate_20
tug_persistence_60
```

**予測符号**  
`tug_balance_20` が高い銘柄ほど当日OCが高い。positive-night/negative-day型が多いほど当日OCは低い。

**反証**  
joint sign rateが単なる `oc_mean_20` と `overnight_mean_20` を超えない、または期間別に符号が反転する。

**PIT**  
各pairは両方の結果が確定した過去営業日のみ。rolling前に必ず1営業日shiftし、no-trade日は分母から除く。

---

### H04：retail-attention opening overpricing proxy

**着想**  
高overnightセンチメントが、評価・裁定困難proxyの高い銘柄で寄り付き過大評価になり、OCで反転するという仮説。

**特徴**

```text
positive_overnight_rate_20
positive_overnight_streak
attention_pressure =
    positive_overnight_rate_20
    * max(xrank_atr14_pct, 0)
    * activity_friction
attention_pressure_x_overnight_last_positive
```

`activity_friction` はno-trade/flat/zero-rangeから作るが、実流動性とは呼ばない。

**予測符号**  
attention pressureが高いほど当日OCは負。特に直近overnightが正のとき強い。

**反証**  
高pressure群がOCで反転しない、低ATR・高activity群にも同じ効果が出る、または単純overnight平均に負ける。

**PIT**  
current target gapは使用禁止。`overnight_last`は前営業日に確定したgapだけ。

---

### H05：day-only horizon-separated reversal / momentum

**着想**  
先行研究の窓に合わせ、直近約1か月の日中成分は反転、12か月から直近1か月を除いた日中成分はmomentumとして分離する。overnight momentumをplaceboとして同じ窓で比較する。履歴不足日はこの仮説の評価対象外にし、結果0で埋めない。

**特徴・モデル**

```text
short_day_reversal_20 = -sum(OC[t-20:t-1])
day_momentum_252_skip20 = sum(OC[t-252:t-21])
night_momentum_252_skip20 = sum(overnight[t-252:t-21])
day_only model removes night_momentum; the night arm is a fixed placebo
```

**予測符号**  
`short_day_reversal_20 > 0` と `day_momentum_252_skip20 > 0` は当日OCに正。`night_momentum_252_skip20`には同じmomentumがない。

**反証**  
day-onlyモデルが同自由度のday+nightモデルを上回らない、day momentumの符号が安定しない、またはnight placeboと差がない。

**PIT**  
前営業日までのOC/overnight系列だけ。20、252、skip20を固定し、252営業日の完全履歴がない行は欠測にする。

---

### H06：TDnet marketwide distraction

**着想**  
各社の文書数ではなく、同じ寄り前判断日に市場全体で処理すべき開示が何件あるかを注意混雑度にする。好材料が混雑日に過小反応し、寄り後も上がるかを調べる。

**特徴**

```text
market_tdnet_code_count
market_tdnet_document_count
market_earnings_count
market_unrelated_family_count
supportive_event_x_log1p(market_tdnet_code_count)
adverse_event_x_log1p(market_tdnet_code_count)
```

**予測符号**  
revision-up、dividend-up、fresh buybackなどのsupportive eventでは混雑との相互作用が正。悪材料では負方向の継続を想定するためlong候補から除外。

**反証**  
混雑日に好材料の寄り後継続が強くならない、または混雑単体だけで説明される。

**PIT**  
当日08:58:59までに到着し、`tdnet_source_complete=True` の行だけで日次集計。欠測日を0件にしない。現在の完全期間に限定。

---

### H07：Friday-disclosure / long-holiday attention lag

**着想**  
金曜日に開示された材料と長期休暇をまたぐ材料は処理が遅れる可能性がある。既存のtarget日曜日sin/cosではなく、**実際の開示曜日・経過時間と方向付きTDnet材料との相互作用**を検証する。

**特徴**

```text
source_event_is_friday
calendar_gap_days_since_prior_session
supportive_event_x_source_event_is_friday
supportive_event_x_weekend_age
supportive_event_x_post_long_holiday
latest_age_hours_x_attention_day
```

**予測符号**  
好材料のOC継続は金曜日開示または長期休暇をまたいだ場合に強い。単なるtarget日の全銘柄Friday効果ではない。

**反証**  
非開示銘柄でも同じ効果、好悪材料で符号差がない、または標本外で消える。

**PIT**  
取引カレンダーとcutoff前TDnetだけ。発表後の出来高・記事数は禁止。

---

### H08：corroborated directional bundle

**着想**  
「文書数が多い」ではなく、利益・業績予想・配当が同じ方向を向くbundleを裏付けありと定義する。相反bundleは別にする。

**特徴**

```text
positive_corroboration =
    revision-up AND dividend-up
    OR forecast-up AND fresh buyback
    OR dividend-up AND fresh buyback
negative_corroboration =
    revision-down AND dividend-down
directional_conflict =
    supportive flag AND adverse flag
corroboration_count
```

**予測符号**  
positive corroborationは正、directional conflictは負または不確実性増大。単独の好材料より同方向bundleが強い。

**反証**  
同方向bundleが単独材料を上回らない、文書数を統制すると消える、またはconflictとの差がない。

**PIT**  
同じcutoff前bundle内だけ。タイトルで方向不明の項目を勝手にpositiveへ分類しない。

---

### H09：earnings-wave sequence

**着想**  
決算シーズンの早い発表と遅い発表では、systematic informationと市場の注意状態が異なる。固定した月日ではなく、当日までの市場発表累積で順序を定義する。

**特徴**

```text
prior_20_session_market_earnings_count
current_wave_percentile =
    prior cumulative earnings count / expanding historical wave-size estimate
days_since_market_earnings_wave_start
earnings_event_x_early_wave
earnings_event_x_late_wave
```

**予測符号**  
early-wave earningsはlate-waveよりOCが高い。late-waveでは既にpeer情報が価格へ入っているため弱い。

**反証**  
early/late差がない、単なる月効果で消える、発表方向を統制すると逆転する。

**PIT**  
wave-sizeの分母に将来の最終発表件数を使わない。過去年の確定wave-sizeまたはexpanding estimateのみ。

---

### H10：top-heavy listwise net-gain ranker

**着想**  
既存のpairwise rankerは全pairをほぼ同等に扱う。しかし実際に価値があるのは毎日の1～2位だけである。

**学習仕様**

```text
query = target date
net_utility_i = clip(OC_i, -5, +5) - 0.20
nonnegative_relevance_i = net_utility_i + 5.20
objective = ListMLE or LambdaMART-style NDCG@2
discount = [1.0, 0.5]
```

最低限、top-vs-restだけを重くする線形listwise surrogateを同じ特徴数で比較する。

**予測**  
全pairの順位精度が同じでも、top1/top2 net20が改善する。

**反証**  
L4、pairwise linear、daily-rank Ridgeに対し、固定未使用期間のtop2 net20・tail-removed・月次安定性のいずれも改善しない。

**PIT**  
query内のtarget returnは学習期間だけ。日付単位split、scoring日より前だけでfit。前処理もfold内。

---

### H11：opening-noise group-specific experts

**着想**  
単一のglobal modelへ交互作用を足すのではなく、寄り付き価格誤差の強い銘柄群と通常群で別モデルを学習する。

**事前group**

```text
opening-noise group:
    high historical posnight-negday rate
    OR high overnight_std_20 and high activity_friction

ordinary group:
    otherwise
```

閾値はtraining fold内の50%分位とし、score日に再計算する。両groupとも同じ基本特徴・同じ正則化候補1個を使う。

**予測**  
opening-noise群ではovernight/attention反転、ordinary群ではday momentumの係数が異なり、global modelよりtop2 net20が高い。

**反証**  
group別係数の符号差が再現しない、groupモデルがglobal modelを上回らない、少数groupだけのoutlier利益に依存する。

**PIT**  
group判定は過去データだけ。全標本分位を使わずfold内分位を保存する。

---

### H12：distributional net-value decision

**着想**  
勝率、平均return、rank percentileを別々に最適化するのではなく、正負の確率と条件付き金額を分け、コスト控除後期待値を作る。これは既存の4閾値ordinal modelとは異なるtwo-part distribution modelである。

**モデル**

```text
p_pos = P(OC > 0.20%)
mu_pos = E[min(OC, +5%) | OC > 0.20%]
mu_neg = E[max(OC, -5%) | OC <= 0.20%]
net_value = p_pos * mu_pos + (1-p_pos) * mu_neg - 0.20%
uncertainty_penalty = expanding residual MAD by predicted-value decile
score = net_value - 0.5 * uncertainty_penalty
```

毎日score上位2件を表示する。`net_value <= 0`ならshadow上はcashも併記し、「毎日1～2件」と「取引すべきか」を分離する。

**予測**  
符号logitやdaily-rank Ridgeより、top1/top2のnet20とprofit factorが高い。

**反証**  
conditional magnitudeが不安定、score calibrationが期間間で崩れる、または20/40bp双方で既存モデルを上回らない。

**PIT**  
条件付き平均、MAD、calibrationはtraining fold内のみ。scoring日のcross-sectional outcomesは一切使用しない。

## 5. 検証順序

同じ標本で組合せを探すと再び停滞するため、次の単方向手順を推奨する。

1. 12仮説の定義・式・符号・反証条件をhash固定する。
2. 価格仮説H01～H05、TDnet仮説H06～H09、モデル仮説H10～H12を**別family**として評価する。
3. 各仮説は「追加した完全版」と「同自由度control」を1対1で比較する。
4. primaryはscheduled-day top2 equal-weight net20、secondaryはtop1 net20、net40、中央値、profit factor。
5. top20 winning days除外、top10 profit codes cash、月別・固定block別を必須にする。
6. 12仮説をまとめたblock bootstrap / multiple-testing adjustmentを行う。
7. retrospective winnerはshadow候補に留める。2026-07-23以降のsource-completeデータを変更せず蓄積する。
8. family内で勝者がなければ、既存L4/L6の再微調整へ戻らず、そのfamilyを棄却する。

## 6. 最初に実行すべき三つ

計算量と識別力から、最初は次の三つがよい。

1. **H01 非対称・大幅下落反発**  
   日本市場固有の一次資料があり、既存線形featureとの差が明確。
2. **H03 joint tug-of-war**  
   現在のraw価格だけで作れ、opening errorという目的に直結する。
3. **H10 top-heavy listwise**  
   「全銘柄をそこそこ当てる」から「毎日の1～2位を当てる」へ目的関数そのものを変えられる。

この三つが全て失敗した場合でも、H06～H09のTDnet注意モデルは独立した情報源なので、別familyとして検証価値が残る。

## 7. 解釈上の注意

- 文献の多くは米国市場であり、日本への外挿は仮説にすぎない。
- TSE固有研究でも標本期間・売買制度は現在と異なる。制度変更後にも再現するかが反証点になる。
- overnightと当日OCの関係を検証するとき、**target日の実始値から作ったovernightを特徴にすると完全なlook-ahead**になる。使用できるのは過去overnightか、今後append-onlyで保存する08:58気配だけ。
- TDnetの「開示なし」と「ソース欠落」を区別する。
- 1～2銘柄/日の目的では、平均的な予測精度、AUC、全pair相関だけを合格根拠にしない。
