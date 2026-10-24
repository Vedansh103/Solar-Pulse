# What was changed from the raw Kaggle files

## Summary

- Plant 1: 68,778 generation rows and 3,182 weather rows became 816 hourly rows.
- Plant 2: 67,698 generation rows and 3,259 weather rows became 816 hourly rows.
- No value was filled, interpolated, clipped or edited. No row was deleted for being zero or an outlier.
- Gaps are blank cells in the output, with flags showing where they are.

## Changes, in order

| # | Change | Why |
| --- | --- | --- |
| 1 | Timestamps parsed with an explicit format. Plant 1 generation is day first (dd-mm-yyyy hh:mm), the other three files are ISO. Times are treated as local time (IST). | The formats differ between files |
| 2 | Columns dropped from generation files: PLANT_ID, DC_POWER, DAILY_YIELD, TOTAL_YIELD. | PLANT_ID is constant. Plant 1 DC power is about 10 times AC power (unit or reporting issue). Yield columns are unreliable (Plant 2 TOTAL_YIELD ranges from 0 to 2.2 billion, DAILY_YIELD is non zero at midnight). |
| 3 | Columns dropped from weather files: PLANT_ID, SOURCE_KEY. | Constant, one sensor per plant |
| 4 | Raw DATE_TIME text column replaced by a parsed timestamp index. | Proper time index |
| 5 | The 22 inverters were aggregated: per 15 minute slot, the mean AC_POWER over inverters that reported. | A raw sum would fall whenever an inverter stops reporting |
| 6 | 15 minute slots averaged to hourly (generation and weather). | The project works at hourly resolution |
| 7 | A continuous hourly index (816 rows) was built first, then generation and weather were left joined on it. | No missing or duplicate timestamps by construction. Gaps appear as blanks |
| 8 | Columns added: inverters_reporting, slots_present, hour, is_daylight, flag_low_coverage, has_generation, has_weather. | Coverage and context, so gaps and weak hours are visible |
| 9 | Inverter keys removed from the master files and saved separately as inverter_hourly_plantX.csv (22 inverters by 816 hours). | Needed later for per inverter anomaly detection |
| 10 | Plant 1 and Plant 2 were kept as two separate files, never stacked. | Different plants, inverter behaviour and sensors |
| 11 | is_daylight uses irradiation above 0.005, not above zero. | Plant 2 irradiation shows noise at night, so a zero threshold would mark night hours as daylight |

## Master file columns

| Column | Unit | Description |
| --- | --- | --- |
| timestamp (index) | IST, hourly | Continuous, 816 rows |
| ac_power_kw | kW | Target. Mean AC power per reporting inverter. Blank where no generation data |
| inverters_reporting | count | Mean number of inverters reporting across the slots of the hour (maximum 22) |
| slots_present | count | 15 minute generation slots present in the hour (maximum 4, 0 means none) |
| irradiation | as in source file | Hourly mean irradiation |
| module_temperature | degrees C | Hourly mean panel temperature |
| ambient_temperature | degrees C | Hourly mean air temperature |
| hour | 0 to 23 | Hour of day |
| is_daylight | flag | 1 when irradiation is above 0.005 (the sensor shows noise up to about 0.002 at night), blank where weather is missing |
| flag_low_coverage | flag | 1 when some slot in the hour had fewer than 22 inverters reporting, blank where no generation data |
| has_generation, has_weather | flag | 1 when the hour has data from that source |

## Files produced

| File | Content |
| --- | --- |
| master_hourly_plant1.csv | 816 rows, Plant 1 (training, tuning and test source) |
| master_hourly_plant2.csv | 816 rows, Plant 2 (external validation source) |
| inverter_hourly_plant1.csv, inverter_hourly_plant2.csv | 816 rows by 22 inverters, hourly mean AC per inverter |
| snapshot_10_rows.csv | 10 consecutive daytime hours from Plant 1 for the slides |
| quality_report.md | Gaps, coverage, outliers, cleaning plan, planned split |
