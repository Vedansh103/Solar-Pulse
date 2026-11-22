# Phase 5: Anomaly Detection Report — Plant 1

## 1. Executive Summary
- **Total Inverters Monitored:** 22 inverters over 816 continuous hours (34 days).
- **Total Inverter Disconnect Events:** 14 hours.
- **Total Soiling / Underperformance Events:** 53 hours.
- **Most Reliable Inverter:** `ZoEaEvLYb1n2sOq` (100.0% availability health).
- **Worst Performing Inverter:** `1BY6WEcLGh8j5v7` (13 fault hours, ~6036.2 kWh lost).

## 2. Key Findings & Plant Diagnosis
- **Cohort Normalization:** Comparing each inverter against the contemporaneous cohort median reliably separates weather effects from equipment faults. When all inverters drop simultaneously, it is a passing cloud; when a single inverter drops to zero while peers generate >400 kW, it is an isolated electrical or string failure.
- **Plant 1 vs. Plant 2 Comparison:**
  - Plant 1 exhibits high stability with very few isolated inverter trips.
  - Plant 2 experiences chronic recurring inverter disconnects across multiple inverters (`LYwnQax7tkwH5Cb`, `Et9kgGMDl729KT4`), directly explaining the higher forecasting variance and the necessity of tree/LSTM models over linear SARIMAX on Plant 2.

## 3. Generated Figures
- Inverter Health Ranking: `figures/phase5/plant1/plant1_inverter_fault_ranking.png`
- Inverter Availability Matrix Heatmap: `figures/phase5/plant1/plant1_inverter_heatmap.png`
- Anomaly Scatter (Power vs Irradiance): `figures/phase5/plant1/plant1_anomaly_scatter.png`
