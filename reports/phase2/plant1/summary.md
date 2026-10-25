# Phase 2 results, Plant 1

Train 15 May to 7 June (576 h), settings chosen on 8 to 10 June (72 h), scored once on the test week 11 to 17 June (168 h). RMSE and MAE in kW per reporting inverter. Skill = 1 - RMSE / RMSE of seasonal naive at the same horizon (above 0 means better than the baseline).

SARIMAX uses measured irradiation and module temperature at the forecast hour, so its scores are an upper bound; in operation these inputs would come from a weather forecast.

## Test metrics

| model | horizon | rmse_all | mae_all | rmse_daylight | mae_daylight | skill_vs_seasonal_naive |
|---|---|---|---|---|---|---|
| Persistence (t-1) | 1h | 143.053 | 92.025 | 194.256 | 168.072 | -0.079 |
| Seasonal naive (t-24) | 1h | 132.616 | 67.87 | 180.189 | 125.298 | 0.0 |
| Seasonal naive (t-24) | 24h | 132.616 | 67.87 | 180.189 | 125.298 | 0.0 |
| Holt-Winters {'trend': 'add', 'damped_trend': True} | 1h | 95.323 | 50.834 | 129.519 | 93.79 | 0.281 |
| Holt-Winters {'trend': 'add', 'damped_trend': True} | 24h | 114.37 | 63.293 | 155.398 | 116.798 | 0.138 |
| SARIMA ((1, 0, 1), (1, 1, 1)) | 1h | 87.52 | 48.122 | 118.913 | 88.688 | 0.34 |
| SARIMA ((1, 0, 1), (1, 1, 1)) | 24h | 112.294 | 62.289 | 152.578 | 114.994 | 0.153 |
| SARIMAX (irradiation, module temp) ((2, 0, 0), (0, 1, 1)) | 1h | 15.893 | 8.31 | 21.581 | 14.924 | 0.88 |
| SARIMAX (irradiation, module temp) ((2, 0, 0), (0, 1, 1)) | 24h | 17.26 | 9.152 | 23.436 | 16.439 | 0.87 |

## Stationarity (training period)

| series | adf_stat | adf_p | kpss_stat | kpss_p | verdict |
|---|---|---|---|---|---|
| raw | -6.3678 | 0.0 | 0.0187 | 0.1 | stationary |
| seasonal diff (lag 24) | -13.5328 | 0.0 | 0.0418 | 0.1 | stationary |

## Settings chosen on validation

- Holt-Winters: {'trend': 'add', 'damped_trend': True}
- SARIMA: ((1, 0, 1), (1, 1, 1))
- SARIMAX (irradiation, module temp): ((2, 0, 0), (0, 1, 1))

| model | setting | val_rmse_1h | val_rmse_24h | val_rmse_mean | seconds |
|---|---|---|---|---|---|
| Holt-Winters | {'trend': None, 'damped_trend': False} | 96.593 | 93.196 | 94.894 | 0.5 |
| Holt-Winters | {'trend': 'add', 'damped_trend': True} | 96.513 | 92.868 | 94.69 | 0.6 |
| SARIMA | ((1, 0, 1), (0, 1, 1)) | 87.73 | 94.423 | 91.077 | 2.7 |
| SARIMA | ((2, 0, 1), (0, 1, 1)) | 87.929 | 94.426 | 91.178 | 6.8 |
| SARIMA | ((1, 0, 1), (1, 1, 1)) | 87.718 | 94.25 | 90.984 | 3.4 |
| SARIMA | ((2, 0, 0), (0, 1, 1)) | 87.821 | 94.424 | 91.122 | 2.6 |
| SARIMAX (irradiation, module temp) | ((1, 0, 1), (0, 1, 1)) | 14.612 | 14.298 | 14.455 | 5.9 |
| SARIMAX (irradiation, module temp) | ((2, 0, 1), (0, 1, 1)) | 14.823 | 14.372 | 14.598 | 7.3 |
| SARIMAX (irradiation, module temp) | ((1, 0, 1), (1, 1, 1)) | 14.605 | 14.292 | 14.448 | 9.9 |
| SARIMAX (irradiation, module temp) | ((2, 0, 0), (0, 1, 1)) | 14.516 | 14.183 | 14.349 | 5.1 |

## Ljung-Box on in-sample residuals (lags 24 and 48)

| model | lag | lb_stat | lb_p | white_noise |
|---|---|---|---|---|
| Holt-Winters | 24 | 56.864 | 0.0002 | False |
| Holt-Winters | 48 | 80.9285 | 0.0021 | False |
| SARIMA | 24 | 12.3217 | 0.976 | True |
| SARIMA | 48 | 31.9485 | 0.9639 | True |
| SARIMAX (irradiation, module temp) | 24 | 24.7808 | 0.4178 | True |
| SARIMAX (irradiation, module temp) | 48 | 58.0229 | 0.1524 | True |
