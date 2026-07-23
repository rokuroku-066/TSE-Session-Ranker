# Model v11 — Cross-stock graph and information propagation

## 結論

- 判定: **reject_family**
- 本番昇格: **なし**（production gate: false）
- familywise比較対照: `C00_daily_rank_ridge`（top1 net40 0.0079%）
- PIT対象: 266 sessions / 13 calendar months

## 事前登録と分離

- Protocol SHA-256: `2f2382716ba9f7b151988470334b4da93bee5f68e30f7264beedcf16bf790244`
- 10 graph hypotheses、共通k=8、trailing 120 sessions、node capacity 1536。
- 閾値・alpha gridなし。Z17の同日peer平均残差targetは再実装していない。
- 外部economic causal-chain、日米sector、news/Twitter sentimentは利用不能のためfail-closed。価格相関を代理の因果関係とは解釈しない。

## 結果

| candidate | top1 net40 | top2 net40 | Δ vs C00 | Δ vs C01 | best20除去 net20 | top10 code cash net20 | +months | fill days | simultaneous LB | adj. p | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `G01_positive_oc_corr_message` | -0.2373% | -0.4476% | -0.2452% | -0.1814% | -0.5176% | -0.3661% | 3/13 | 266/266 | -0.5468% | 0.9994 | FAIL |
| `G02_signed_oc_corr_message` | -0.2320% | -0.4496% | -0.2399% | -0.1761% | -0.5119% | -0.3633% | 4/13 | 266/266 | -0.5416% | 0.9994 | FAIL |
| `G03_directed_lead_lag_message` | -0.4685% | -0.5228% | -0.4764% | -0.4126% | -0.5810% | -0.4599% | 2/13 | 266/266 | -0.7781% | 1.0000 | FAIL |
| `G04_overnight_corr_message` | -0.3993% | -0.4365% | -0.4072% | -0.3434% | -0.5669% | -0.4151% | 3/13 | 266/266 | -0.7089% | 1.0000 | FAIL |
| `G05_cojump_neighbor_pressure` | -0.3521% | -0.4779% | -0.3600% | -0.2962% | -0.5761% | -0.4138% | 3/13 | 266/266 | -0.6617% | 1.0000 | FAIL |
| `G06_tdnet_event_neighbor_spillover` | -0.4362% | -0.4151% | -0.4441% | -0.3803% | -0.6618% | -0.4886% | 1/13 | 187/266 | -0.7458% | 1.0000 | FAIL |
| `G07_neighbor_conditioned_empirical_uplift` | -0.3050% | -0.4450% | -0.3129% | -0.2491% | -0.6513% | -0.4169% | 3/13 | 266/266 | -0.6146% | 1.0000 | FAIL |
| `G08_dynamic_community_archetype` | -0.2447% | -0.3291% | -0.2526% | -0.1888% | -0.5113% | -0.3169% | 3/13 | 266/266 | -0.5542% | 0.9996 | FAIL |
| `G09_two_hop_oc_diffusion` | -0.4267% | -0.4747% | -0.4346% | -0.3708% | -0.5586% | -0.4325% | 3/13 | 266/266 | -0.7362% | 1.0000 | FAIL |
| `G10_graph_disagreement_cash` | -0.1765% | -0.2772% | -0.1844% | -0.1206% | -0.3878% | -0.2453% | 6/13 | 266/266 | -0.4861% | 0.9980 | FAIL |

### 対照

| comparator | top1 net20 | top1 net40 | top1 net60 | top2 net40 |
|---|---:|---:|---:|---:|
| `C00_daily_rank_ridge` | 0.2049% | 0.0079% | -0.1891% | 0.0587% |
| `C01_L4_price_control` | 0.1411% | -0.0559% | -0.2529% | 0.0866% |
| `Z17_peer_residual_reference` | — | -0.4964% | — | -0.4991% |

## 独立gate

点推定最大は `G10_graph_disagreement_cash`（top1 net40 -0.1765%）。

未達条件: `top1_net40_positive`, `top2_net40_positive`, `top1_uplift_vs_C00_positive`, `top1_uplift_vs_C01_positive`, `familywise_lower_bound_positive`, `familywise_p_at_most_0_10`, `best20_removed_net20_positive`, `top10_codes_cash_net20_positive`, `all_three_slices_net40_positive`, `at_least_nine_of_thirteen_months_net40_positive`

## 監査

- PIT/source/mutation: **PASS**
- 独立P&L再計算: **PASS**
- v10 C00 top2同値再現: **PASS**
- byte-identical picks rerun: **PASS**
- metrics/folds rerun: **PASS**

## 一次資料と限界

- [経済因果チェーンを用いたリードラグ効果の実証分析](https://www.jstage.jst.go.jp/article/jsaisigtwo/2020/FIN-024/2020_171/_article/-char/ja/) — Motivates directed shock propagation, but the paper's text-extracted economic causal chain is unavailable here; price-derived edges are not treated as causal substitutes.
- [部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略](https://www.jstage.jst.go.jp/article/jsaisigtwo/2026/FIN-036/2026_76/_article/-char/ja/) — Motivates low-dimensional propagation and lead-lag structure; US ETF and sector data are unavailable, so no Japanese/US proxy is constructed.
- [ニュースセンチメントのGranger因果関係に基づく銘柄間ネットワークを通した日本株式市場の情報伝播構造の分析](https://www.jstage.jst.go.jp/article/jsaisigtwo/2026/BI-028/2026_19/_article/-char/ja/) — Motivates directed information networks and supplies an important falsification prior: its major centrality measures had no significant future-return predictive power. Centrality alone is therefore not a candidate.
- [株式市場に埋め込まれたグループ相関構造―日米比較](https://www.jstage.jst.go.jp/article/trafst/7/2/7_92/_article/-char/ja/) — Motivates signed correlation networks and a fixed four-community archetype, without claiming that the historical four-group finding must persist.

本検証のedgeは内生的な価格・overnight・TDnet関係であり、供給網、業種、sentiment、因果関係の観測ではない。retrospective panelからの判断はforward-shadow候補までで、本番採用には登録後120件以上のprospective gateと明示的人手承認が必要。
