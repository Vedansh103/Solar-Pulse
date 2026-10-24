# Phase 2 results, Plant 1

Train 15 May to 7 June (576 h), settings chosen on 8 to 10 June (72 h), scored once on the test week 11 to 17 June (168 h). RMSE and MAE in kW per reporting inverter. Skill = 1 - RMSE / RMSE of seasonal naive at the same horizon (above 0 means better than the baseline).

SARIMAX uses measured irradiation and module temperature at the forecast hour, so its scores are an upper bound; in operation these inputs would come from a weather forecast.

## Test metrics

| model | horizon | rmse_all | mae_all | rmse_daylight | mae_daylight | skill_vs_seasonal_naive |
|---|---|---|---|---|---|---|
| Persistence (t-1) | 1h | 143.053 | 92.025 | 194.256 | 168.072 | -0.079 |
| Seasonal naive (t-24) | 1h | 132.616 | 67.87 | 180.189 | 125.298 | 0.0 |
| Seasonal naive (t-24) | 24h | 132.616 | 67.87 | 180.189 | 125.298 | 0.0 |
| Holt-Winters {'trend': None, 'damped_trend': False} | 1h | 93.537 | 51.181 | 127.085 | 93.71 | 0.295 |
| Holt-Winters {'trend': None, 'damped_trend': False} | 24h | 111.363 | 62.495 | 151.311 | 115.042 | 0.16 |
| SARIMA ((2, 0, 1), (0, 1, 1)) | 1h | 87.956 | 49.67 | 119.324 | 87.809 | 0.337 |
| SARIMA ((2, 0, 1), (0, 1, 1)) | 24h | 112.921 | 63.274 | 153.413 | 115.513 | 0.149 |
| SARIMAX (irradiation, module temp) ((2, 0, 0), (0, 1, 1)) | 1h | 15.892 | 8.311 | 21.579 | 14.921 | 0.88 |
| SARIMAX (irradiation, module temp) ((2, 0, 0), (0, 1, 1)) | 24h | 17.259 | 9.155 | 23.435 | 16.439 | 0.87 |

## Stationarity (training period)

| series | adf_stat | adf_p | kpss_stat | kpss_p | verdict |
|---|---|---|---|---|---|
| raw | -6.3679 | 0.0 | 0.0187 | 0.1 | stationary |
| seasonal diff (lag 24) | -13.5328 | 0.0 | 0.0418 | 0.1 | stationary |

## Settings chosen on validation

- Holt-Winters: {'trend': None, 'damped_trend': False}
- SARIMA: ((2, 0, 1), (0, 1, 1))
- SARIMAX (irradiation, module temp): ((2, 0, 0), (0, 1, 1))

| model | setting | val_rmse_1h | val_rmse_24h | val_rmse_mean | seconds |
|---|---|---|---|---|---|
| Holt-Winters | {'trend': None, 'damped_trend': False} | 99.502 | 96.271 | 97.886 | 0.1 |
| Holt-Winters | {'trend': 'add', 'damped_trend': True} | 101.471 | 109.0 | 105.236 | 0.2 |
| SARIMA | ((1, 0, 1), (0, 1, 1)) | 87.73 | 94.423 | 91.077 | 4.4 |
| SARIMA | ((2, 0, 1), (0, 1, 1)) | 86.833 | 94.126 | 90.48 | 11.3 |
| SARIMA | ((1, 0, 1), (1, 1, 1)) | 87.718 | 94.25 | 90.984 | 5.5 |
| SARIMA | ((2, 0, 0), (0, 1, 1)) | 87.821 | 94.424 | 91.122 | 4.4 |
| SARIMAX (irradiation, module temp) | ((1, 0, 1), (0, 1, 1)) | 14.611 | 14.298 | 14.454 | 10.6 |
| SARIMAX (irradiation, module temp) | ((2, 0, 1), (0, 1, 1)) | 14.819 | 14.371 | 14.595 | 14.9 |
| SARIMAX (irradiation, module temp) | ((1, 0, 1), (1, 1, 1)) | 14.604 | 14.292 | 14.448 | 17.2 |
| SARIMAX (irradiation, module temp) | ((2, 0, 0), (0, 1, 1)) | 14.518 | 14.186 | 14.352 | 9.3 |

## Ljung-Box on in-sample residuals (lags 24 and 48)

| model | lag | lb_stat | lb_p | white_noise |
|---|---|---|---|---|
| Holt-Winters | 24 | 42.989 | 0.01 | False |
| Holt-Winters | 48 | 66.006 | 0.0432 | False |
| SARIMA | 24 | 11.4223 | 0.9857 | True |
| SARIMA | 48 | 30.2618 | 0.9787 | True |
| SARIMAX (irradiation, module temp) | 24 | 24.7241 | 0.4209 | True |
| SARIMAX (irradiation, module temp) | 48 | 57.9335 | 0.1543 | True |
