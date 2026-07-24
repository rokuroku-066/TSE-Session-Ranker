# Round 2: momentum-diversification falsification

This round was registered after the H12 point result was visible. It is a posthoc falsification study, not confirmation.

| Variant | Net20 | Net40 | Net60 | Months + | Top20 removed | ES5 | Worst day |
|---|---:|---:|---:|---:|---:|---:|---:|
| F08_MOMENTUM20_OPPOSITE75 | +0.3518% | +0.1559% | -0.0400% | 12/13 | +0.0517% | -2.7780% | -7.0452% |
| F10_MOMENTUM20_SIGN | +0.3186% | +0.1220% | -0.0746% | 12/13 | +0.0468% | -2.7314% | -5.7772% |
| F00_H12_EXACT | +0.3044% | +0.1082% | -0.0881% | 13/13 | +0.0462% | -2.6219% | -5.7772% |
| F11_MOMENTUM20_TOP10_OPPOSITE | +0.2957% | +0.0991% | -0.0976% | 12/13 | +0.0369% | -2.6980% | -5.7772% |
| F06_CLOSE_LOCATION_HALF | +0.2880% | +0.0925% | -0.1030% | 12/13 | +0.0507% | -2.6862% | -5.7772% |
| F07_MOMENTUM20_LEADER75 | +0.2571% | +0.0605% | -0.1362% | 12/13 | -0.0231% | -3.1464% | -6.4465% |
| F13_MOMENTUM20_EXTREMES_ONLY | +0.2281% | +0.0315% | -0.1651% | 10/13 | -0.0341% | -3.3173% | -8.0241% |
| F09_MOMENTUM20_EXTREME_TERCILE | +0.2220% | +0.0254% | -0.1712% | 10/13 | -0.0293% | -3.3074% | -8.0241% |
| F02_MOMENTUM60_HALF | +0.2056% | +0.0090% | -0.1876% | 11/13 | -0.0474% | -2.9816% | -5.7772% |
| F12_MOMENTUM20_TOP20_NEUTRAL | +0.1300% | -0.0666% | -0.2632% | 9/13 | -0.0914% | -2.8567% | -6.2040% |
| F01_MOMENTUM5_HALF | +0.1218% | -0.0741% | -0.2700% | 9/13 | -0.0724% | -2.1377% | -4.3419% |
| F05_ATR_HALF | +0.0973% | -0.0989% | -0.2952% | 8/13 | -0.1289% | -2.7629% | -5.3030% |
| F04_OCMEAN20_HALF | +0.0966% | -0.1000% | -0.2966% | 8/13 | -0.1593% | -3.2838% | -8.0241% |
| F03_OVERNIGHT20_HALF | +0.0278% | -0.1695% | -0.3669% | 7/13 | -0.2102% | -2.9538% | -5.3030% |
| F14_MOMENTUM20_RESIDUAL_ONLY | -0.0163% | -0.2137% | -0.4110% | 6/13 | -0.2979% | -3.5225% | -8.1820% |

## Mechanism checks

- Specificity wins versus unrelated splits: 4/4.
- Adjacent horizons net20: {"F01_MOMENTUM5_HALF": 0.12175014011509544, "F02_MOMENTUM60_HALF": 0.20563002572137346}.
- Smooth alternatives beating L4 with positive top-20-removed return: F08_MOMENTUM20_OPPOSITE75, F10_MOMENTUM20_SIGN, F11_MOMENTUM20_TOP10_OPPOSITE.
- Retain for forward shadow: yes.

The reference can be retained only when both specificity and smoothness checks pass. No retrospective result changes production.
