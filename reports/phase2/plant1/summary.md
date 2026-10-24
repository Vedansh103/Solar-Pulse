# Phase 2 results, Plant 1

Train 15 May to 7 June (576 h), settings chosen on 8 to 10 June (72 h), scored once on the test week 11 to 17 June (168 h). RMSE and MAE in kW per reporting inverter. Skill = 1 - RMSE / RMSE of seasonal naive at the same horizon (above 0 means better than the baseline).

SARIMAX uses measured irradiation and module temperature at the forecast hour, so its scores are an upper bound; in operation these inputs would come from a weather forecast.

## Test metrics

| model | horizon | rmse_all | mae_all | rmse_daylight | mae_daylight | skill_vs_seasonal_naive |
|---|---|---|---|---|---|---|
| Persistence (t-1) | 1h | 143.053 | 92.025 | 194.256 | 168.072 | -0.079 |
| Seasonal naive (t-24) | 1h | 132.616 | 67.87 | 180.189 | 125.298 | 0.0 |
| Seasonal naive (t-24) | 24h | 132.616 | 67.87 | 180.189 | 125.298 | 0.0 |
| Holt-Winters {'trend': None, 'damped_trend': False} | 1h | 93.383 | 51.961 | 126.84 | 93.944 | 0.296 |
| Holt-Winters {'trend': None, 'damped_trend': False} | 24h | 110.705 | 63.107 | 150.389 | 114.713 | 0.165 |
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

- Holt-Winters: {'trend': None, 'damped_trend': False}
- SARIMA: ((1, 0, 1), (1, 1, 1))
- SARIMAX (irradiation, module temp): ((2, 0, 0), (0, 1, 1))

| model | setting | val_rmse_1h | val_rmse_24h | val_rmse_mean | seconds |
|---|---|---|---|---|---|
| Holt-Winters | {'trend': None, 'damped_trend': False} | 99.996 | 97.272 | 98.634 | 0.1 |
| Holt-Winters | {'trend': 'add', 'damped_trend': True} | 101.507 | 109.226 | 105.366 | 0.2 |
| SARIMA | ((1, 0, 1), (0, 1, 1)) | 87.73 | 94.423 | 91.077 | 4.4 |
| SARIMA | ((2, 0, 1), (0, 1, 1)) | 87.929 | 94.426 | 91.178 | 11.4 |
| SARIMA | ((1, 0, 1), (1, 1, 1)) | 87.718 | 94.25 | 90.984 | 5.3 |
| SARIMA | ((2, 0, 0), (0, 1, 1)) | 87.821 | 94.424 | 91.122 | 4.4 |
| SARIMAX (irradiation, module temp) | ((1, 0, 1), (0, 1, 1)) | 14.612 | 14.298 | 14.455 | 9.9 |
| SARIMAX (irradiation, module temp) | ((2, 0, 1), (0, 1, 1)) | 14.823 | 14.372 | 14.598 | 12.1 |
| SARIMAX (irradiation, module temp) | ((1, 0, 1), (1, 1, 1)) | 14.605 | 14.292 | 14.448 | 16.5 |
| SARIMAX (irradiation, module temp) | ((2, 0, 0), (0, 1, 1)) | 14.516 | 14.183 | 14.349 | 8.9 |

## Ljung-Box on in-sample residuals (lags 24 and 48)

| model | lag | lb_stat | lb_p | white_noise |
|---|---|---|---|---|
| Holt-Winters | 24 | 47.5704 | 0.0029 | False |
| Holt-Winters | 48 | 75.1472 | 0.0074 | False |
| SARIMA | 24 | 12.3217 | 0.976 | True |
| SARIMA | 48 | 31.9485 | 0.9639 | True |
| SARIMAX (irradiation, module temp) | 24 | 24.7808 | 0.4177 | True |
| SARIMAX (irradiation, module temp) | 48 | 58.023 | 0.1524 | True |
