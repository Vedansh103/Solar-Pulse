# Solar-Pulse

Short-term forecasting and anomaly detection of hourly solar power generation at two Indian PV plants,
using irradiation and temperature data.

- **Data:** Kaggle "Solar Power Generation Data" (anikannal), two plants, 15 May to 17 June 2020,
  resampled to 816 hourly rows per plant.
- **Target:** `ac_power_kw`, hourly AC power in kW, as the mean per reporting inverter (22 inverters per plant).
- **Horizons:** 1 hour ahead (operations) and 24 hours ahead (day-ahead scheduling).
- **Metrics:** RMSE and MAE in kW, over all hours and daylight hours only. No MAPE, because night output is zero.
  Every model must beat the naive baselines.

## Data split (time ordered, never shuffled)

| Role | Plant | Period | Hours |
|---|---|---|---|
| Train | 1 | 15 May to 7 June | 576 |
| Validation (tuning) | 1 | 8 to 10 June | 72 |
| Test (used once) | 1 | 11 to 17 June | 168 |
| External validation (transfer) | 2 | 11 to 17 June | 168 |

## Folder layout

```
data/
  raw/          4 Kaggle CSVs (never edited)
  processed/    master_hourly_plantX.csv, inverter_hourly_plantX.csv, quality_report.md, snapshot
  features/     features_plantX.csv (Phase 1 output, model input)
  anomalies/    alert tables (Phase 5)
src/
  preprocess/   raw -> processed (P0)
  eda/          EDA figures (P0)
  features/     processed -> features (P1)
  models/       classical, tree, deep learning models (P2-P4)
  anomaly/      plant and inverter anomaly detection (P5)
  evaluation/   transfer test, leaderboard (P6-P7)
  rag/          knowledge base ingestion and retrieval (P8)
  llm/          LLM assistant and tools (P9-P10)
  api/, app/    backend and dashboard (P11)
configs/        plant location and other settings
reports/        metrics tables and written results per phase
figures/        generated plots (not tracked, rebuilt by scripts)
docs/           project reference PDFs and change log
slides/         presentation decks
kb/             RAG knowledge base store
tests/
```

## Run the pipeline

```bash
pip install -r requirements.txt
python src/preprocess/preprocess.py        # P0: raw -> data/processed
python src/eda/eda.py                      # P0: EDA figures -> figures/eda
python src/features/build.py               # P1: data/processed -> data/features
python src/models/classical.py --plant 1   # P2: baselines & classical models -> reports/phase2
python src/models/classical.py --plant 2
python src/models/tree_models.py --plant 1 # P3: gradient-boosted trees & ensemble -> reports/phase3
python src/models/shap_analysis.py --plant 1 # P3: SHAP explainability analysis -> figures/phase3
python src/models/tree_models.py --plant 2
python src/models/shap_analysis.py --plant 2
pytest -q                                  # Run automated test suite (22 tests)
```

## Phases

| Phase | Work | Status |
|---|---|---|
| P0 | Data foundation and EDA | Done |
| P1 | Gap imputation and feature engineering | Done |
| P2 | Baselines and classical models (persistence, seasonal naive, Holt-Winters, SARIMA, SARIMAX) | Done |
| P3 | Gradient boosting (XGBoost, LightGBM, CatBoost), Ensembles, and SHAP | Done |
| P4 | Deep learning (LSTM/GRU, TCN) and P10-P90 forecast bands | Planned |
| P5 | Anomaly detection (STL residuals, inverter peer ratio, Isolation Forest) | Planned |
| P6 | Cross-plant transfer test on Plant 2 | Planned |
| P7 | Leaderboard and model registry | Planned |
| P8 | RAG knowledge base | Planned |
| P9 | LLM assistant with data tools | Planned |
| P10 | LLM evaluation and safety | Planned |
| P11 | Dashboard, API and deployment | Planned |

## Caveats

- Only 34 days of data: no weekly or seasonal effects, no long-horizon forecasts.
- Models with irradiation as an input use measured irradiation, so their accuracy is an upper bound
  (in real use it would come from a weather forecast).
- Both plants share the same dates and weather, so Plant 2 is a transfer check, not an independent test.
- The plant locations are not given in the source data; the clear-sky features use an approximate location
  (`configs/plant_location.yaml`).
