# Phase 5: Anomaly Detection Report — Plant 2

## 1. Executive Summary
- **Total Inverters Monitored:** 22 inverters over 816 continuous hours (34 days).
- **Total Inverter Disconnect Events:** 711 hours.
- **Total Soiling / Underperformance Events:** 596 hours.
- **Most Reliable Inverter:** `Mx2yZCDsyf6DPfv` (91.3% availability health).
- **Worst Performing Inverter:** `Quc1TzYxW2pYoWX` (192 fault hours, ~86533.3 kWh lost).

## 2. Key Findings & Plant Diagnosis
- **Cohort Normalization:** Comparing each inverter against the contemporaneous cohort median reliably separates weather effects from equipment faults. When all inverters drop simultaneously, it is a passing cloud; when a single inverter drops to zero while peers generate >400 kW, it is an isolated electrical or string failure.
- **Plant 1 vs. Plant 2 Comparison:**
  - Plant 1 exhibits high stability with very few isolated inverter trips.
  - Plant 2 experiences chronic recurring inverter disconnects across multiple inverters (`LYwnQax7tkwH5Cb`, `Et9kgGMDl729KT4`), directly explaining the higher forecasting variance and the necessity of tree/LSTM models over linear SARIMAX on Plant 2.

## 3. Generated Figures
- Inverter Health Ranking: `figures/phase5/plant2/plant2_inverter_fault_ranking.png`
- Inverter Availability Matrix Heatmap: `figures/phase5/plant2/plant2_inverter_heatmap.png`
- Anomaly Scatter (Power vs Irradiance): `figures/phase5/plant2/plant2_anomaly_scatter.png`
