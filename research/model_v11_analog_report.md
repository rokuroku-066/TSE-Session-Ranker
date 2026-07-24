# Model v11 — Historical analog retrieval

## 結論

- 判定: **reject_family**
- 本番昇格: **なし**（production gate: false）
- familywise比較対照: `T02_char_value_event_only`（top1 net40 0.5191%）
- PIT対象: 88 sessions / 2024-07, 2025-04, 2025-05, 2025-06, 2025-07
- TDnet履歴HTMLには観測時刻sidecarがなく、このrun単独では本番採用を認めない。

## 事前登録

- Protocol: `v11_historical_analog_zero_base_20260723`
- SHA-256: `823012afa89fe9ec982a7ca82c78c3f24697b6969d1d3fcaa73f5ee7d3b82e86`
- 構造的仮説: 10件
- 共通k=31。k/閾値grid、結果確認後の候補変更、analog候補の係数学習はいずれもなし。

## 結果

| candidate | top1 net40 | top2 net40 | Δ vs L4 | Δ vs T02 | best20除去 net20 | top10 code cash net20 | +months | fill days | simultaneous LB | adj. p | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `A01_global_similarity_mean` | -0.2695% | -0.4605% | -0.2917% | -0.7886% | -1.2053% | -0.7207% | 1/5 | 88/88 | -1.4415% | 1.0000 | FAIL |
| `A02_issuer_excluded_mean` | -0.6533% | -0.5760% | -0.6755% | -1.1724% | -1.3432% | -0.9130% | 0/5 | 88/88 | -1.8253% | 1.0000 | FAIL |
| `A03_time_decay_mean` | -0.2162% | -0.3961% | -0.2384% | -0.7352% | -1.1197% | -0.6625% | 1/5 | 88/88 | -1.3882% | 1.0000 | FAIL |
| `A04_cross_code_one_per_issuer` | -0.4852% | -0.4919% | -0.5075% | -1.0043% | -1.2906% | -0.8241% | 1/5 | 88/88 | -1.6573% | 1.0000 | FAIL |
| `A05_neighbor_lower_quantile` | -0.1687% | -0.0032% | -0.1909% | -0.6877% | -0.4044% | -0.2803% | 2/5 | 33/88 | -1.3407% | 1.0000 | FAIL |
| `A06_dual_tail_neighbor_utility` | -0.0485% | -0.2079% | -0.0707% | -0.5675% | -0.7965% | -0.4456% | 2/5 | 88/88 | -1.2205% | 1.0000 | FAIL |
| `A07_event_family_prototype` | -0.2826% | -0.3677% | -0.3049% | -0.8017% | -0.7835% | -0.4539% | 1/5 | 87/88 | -1.4547% | 1.0000 | FAIL |
| `A08_same_issuer_memory` | -0.4305% | -0.3884% | -0.4528% | -0.9496% | -1.1406% | -0.7110% | 2/5 | 88/88 | -1.6026% | 1.0000 | FAIL |
| `A09_robust_neighbor_consensus` | -0.3217% | -0.3326% | -0.3440% | -0.8408% | -1.0720% | -0.6507% | 1/5 | 88/88 | -1.4938% | 1.0000 | FAIL |
| `A10_family_then_text_cross_code` | -0.2149% | -0.2044% | -0.2371% | -0.7339% | -1.1010% | -0.6463% | 1/5 | 88/88 | -1.3869% | 1.0000 | FAIL |

### 対照

| comparator | top1 net20 | top1 net40 | top1 net60 | top2 net40 |
|---|---:|---:|---:|---:|
| `L4_price_control` | 0.2200% | 0.0222% | -0.1755% | 0.1075% |
| `T02_char_value_event_only` | 0.7191% | 0.5191% | 0.3191% | -0.0855% |

## 独立gate

点推定最大は `A06_dual_tail_neighbor_utility`（top1 net40 -0.0485%）。

未達条件: `top1_net40_positive`, `top2_net40_positive`, `top1_uplift_vs_L4_positive`, `top1_uplift_vs_T02_positive`, `familywise_lower_bound_positive`, `familywise_p_at_most_0_10`, `best20_removed_net20_positive`, `top10_codes_cash_net20_positive`, `all_three_slices_net40_positive`, `at_least_four_of_five_months_net40_positive`

全候補について月次・3 slice・best20除去・top10 code cash・top2・familywise補正を同時に要求した。通過候補がなければwinnerを選ばない。

## 監査

- Input/PIT/mutation: **PASS**
- P&L再計算: **PASS**
- v10 L4/T02同値再現: **PASS**
- byte-identical picks rerun: **PASS**
- candidate metrics rerun: **PASS**

## 本番判断

このretrospective runから本番モデルへの変更は行っていない。事前登録された全retrospective条件に加え、観測時刻provenanceと登録後120件以上の新規strict-source-complete sessionでprospective gateを満たし、明示的人手承認を得るまでは本番採用不可。
