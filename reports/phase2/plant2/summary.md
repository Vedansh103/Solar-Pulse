# Phase 2 results, Plant 2

Train 15 May to 7 June (576 h), settings chosen on 8 to 10 June (72 h), scored once on the test week 11 to 17 June (168 h). RMSE and MAE in kW per reporting inverter. Skill = 1 - RMSE / RMSE of seasonal naive at the same horizon (above 0 means better than the baseline).

SARIMAX uses measured irradiation and module temperature at the forecast hour, so its scores are an upper bound; in operation these inputs would come from a weather forecast.

## Test metrics

| model | horizon | rmse_all | mae_all | rmse_daylight | mae_daylight | skill_vs_seasonal_naive |
|---|---|---|---|---|---|---|
| Persistence (t-1) | 1h | 112.032 | 70.926 | 151.817 | 128.021 | -0.041 |
| Seasonal naive (t-24) | 1h | 107.658 | 60.025 | 146.279 | 110.815 | 0.0 |
| Seasonal naive (t-24) | 24h | 107.658 | 60.025 | 146.279 | 110.815 | 0.0 |
| Holt-Winters {'trend': 'add', 'damped_trend': True} | 1h | 89.553 | 50.26 | 121.596 | 89.291 | 0.168 |
| Holt-Winters {'trend': 'add', 'damped_trend': True} | 24h | 138.672 | 87.666 | 187.671 | 147.734 | -0.288 |
| SARIMA ((1, 0, 1), (1, 1, 1)) | 1h | 82.918 | 43.322 | 112.652 | 79.626 | 0.23 |
| SARIMA ((1, 0, 1), (1, 1, 1)) | 24h | 93.393 | 50.389 | 126.896 | 93.026 | 0.133 |
| SARIMAX (irradiation, module temp) ((1, 0, 1), (0, 1, 1)) | 1h | 69.649 | 34.33 | 94.631 | 63.059 | 0.353 |
| SARIMAX (irradiation, module temp) ((1, 0, 1), (0, 1, 1)) | 24h | 77.682 | 36.505 | 105.547 | 67.131 | 0.278 |

## Stationarity (training period)

| series | adf_stat | adf_p | kpss_stat | kpss_p | verdict |
|---|---|---|---|---|---|
| raw | -7.6022 | 0.0 | 0.0583 | 0.1 | stationary |
| seasonal diff (lag 24) | -11.0912 | 0.0 | 0.0478 | 0.1 | stationary |

## Settings chosen on validation

- Holt-Winters: {'trend': 'add', 'damped_trend': True}
- SARIMA: ((1, 0, 1), (1, 1, 1))
- SARIMAX (irradiation, module temp): ((1, 0, 1), (0, 1, 1))

| model | setting | val_rmse_1h | val_rmse_24h | val_rmse_mean | seconds |
|---|---|---|---|---|---|
| Holt-Winters | {'trend': None, 'damped_trend': False} | 91.852 | 103.691 | 97.772 | 0.1 |
| Holt-Winters | {'trend': 'add', 'damped_trend': True} | 94.077 | 98.206 | 96.142 | 0.2 |
| SARIMA | ((1, 0, 1), (0, 1, 1)) | 85.514 | 95.461 | 90.488 | 3.9 |
| SARIMA | ((2, 0, 1), (0, 1, 1)) | 86.565 | 95.476 | 91.02 | 11.0 |
| SARIMA | ((1, 0, 1), (1, 1, 1)) | 85.496 | 95.468 | 90.482 | 4.8 |
| SARIMA | ((2, 0, 0), (0, 1, 1)) | 85.959 | 95.461 | 90.71 | 3.9 |
| SARIMAX (irradiation, module temp) | ((1, 0, 1), (0, 1, 1)) | 69.152 | 62.532 | 65.842 | 6.0 |
| SARIMAX (irradiation, module temp) | ((2, 0, 1), (0, 1, 1)) | 70.186 | 62.608 | 66.397 | 16.3 |
| SARIMAX (irradiation, module temp) | ((1, 0, 1), (1, 1, 1)) | 69.152 | 62.569 | 65.86 | 9.3 |
| SARIMAX (irradiation, module temp) | ((2, 0, 0), (0, 1, 1)) | 69.508 | 62.569 | 66.038 | 6.4 |

## Ljung-Box on in-sample residuals (lags 24 and 48)

| model | lag | lb_stat | lb_p | white_noise |
|---|---|---|---|---|
| Holt-Winters | 24 | 75.6401 | 0.0 | False |
| Holt-Winters | 48 | 130.6958 | 0.0 | False |
| SARIMA | 24 | 35.0831 | 0.0672 | True |
| SARIMA | 48 | 67.7109 | 0.0319 | False |
| SARIMAX (irradiation, module temp) | 24 | 30.5387 | 0.1675 | True |
| SARIMAX (irradiation, module temp) | 48 | 43.1616 | 0.671 | True |
