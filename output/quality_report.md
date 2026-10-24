# Quality report (Phase 1)

Phase 1 flags only: no values filled, altered or removed. Index is continuous, 816 hourly rows per plant (15 May 00:00 to 17 Jun 23:00 IST).

| Check | Plant 1 | Plant 2 |
|---|---|---|
| Raw generation rows | 68,778 | 67,698 |
| Raw weather rows | 3,182 | 3,259 |
| Duplicate rows (gen + weather) | 0 | 0 |
| Null values (raw files) | 0 | 0 |
| Distinct generation slots present (of 3,264) | 3,158 | 3,259 |
| Distinct weather slots present (of 3,264) | 3,182 | 3,259 |
| Hours with no generation data | 20 | 0 |
| Hours with no weather data | 15 | 0 |
| Hours with partial slots (1-3 of 4) | 13 | 5 |
| Hours with some inverter not reporting (flag_low_coverage=1) | 40 | 227 |
| Fewest / most rows per inverter | 3,104 / 3,155 | 2,355 / 3,259 |
| Hourly AC vs irradiation correlation | 0.998 | 0.940 |
| Median AC per inverter, 11:00-13:59 (kW) | 955 | 644 |
| Night zeros kept (daylight=0, AC=0) | 326 | 303 |
| IQR outliers, all hours (fences) | 0 (-937, 1562) | 0 (-788, 1313) |
| IQR outliers, daylight hours only (fences) | 0 of 470 (-857, 1839) | 0 of 513 (-807, 1512) |
| IQR outliers, daylight, per hour of day | 10 | 9 |

## Notes

- AC is the mean per reporting inverter; `inverters_reporting` is the hourly mean count, `flag_low_coverage` is 1 if any slot in the hour had fewer than 22 inverters (blank when no generation).
- Outliers use the 1.5 x IQR rule and are flagged only. The all-hours rule is distorted by night zeros (bimodal data), so daylight-only and per-hour counts are also given.
- DC_POWER, yields, PLANT_ID and sensor/raw-time columns are dropped from the master files; inverter SOURCE_KEY is kept only in `inverter_hourly_plantX.csv`.

## Cleaning plan (Phase 2)

- Interpolate short gaps (a few hours); flag long gaps (e.g. multi-hour overnight and 20 May daytime gap in Plant 1) rather than fill.
- Treat Plant 2 low-coverage hours via the per-reporting-inverter mean already used here.
