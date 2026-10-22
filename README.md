# Fault Detection & Anomaly Diagnosis on the Tennessee Eastman Process (TEP)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end unsupervised and semi-supervised machine learning framework designed for fault detection, sensor anomaly isolation, and dimensionality reduction on the realistic industrial benchmark: the **Tennessee Eastman Process (TEP)**.

---

## 📌 Project Overview

The Tennessee Eastman Process (TEP) represents a complex, non-linear, multi-unit chemical process widely utilized as an industry-standard benchmark for process control, monitoring, and fault diagnosis.

Industrial plants generate dense multivariate telemetry with strong cross-correlations, temporal dynamics, and sensor noise. This project builds robust anomaly detection pipelines by:
1. **Dimensionality Reduction & Statistical Metrics**: Projecting continuous and sampled process streams into orthogonal latent spaces using **Principal Component Analysis (PCA)** and tracking anomaly thresholds via **Squared Prediction Error (SPE)**.
2. **Feature Reconstruction via Gradient Boosting**: Training an ensemble of **LightGBM Regressors** in an all-for-one reconstruction configuration to detect faults when collective residual errors deviate from steady-state thresholds.
3. **Temporal Dynamics & Denoising**: Generating dynamic lag variables to capture transient delay states and applying moving-average window filters to attenuate sensor noise.

---

## ⚙️ Process & Data Characteristics

The simulated plant comprises 5 main operation units: **Reactor**, **Condenser**, **Compressor**, **Separator**, and **Stripper**.

The pipeline monitors **52 process telemetry variables** across **20 pre-defined operational disturbances** (`IDV(1)` to `IDV(20)`):

| Stream Category | Count | Sampling Rate / Description |
| :--- | :---: | :--- |
| **Manipulated Variables (`XMV`)** | 12 | Valves, coolant flow, reactor agitation |
| **Continuous Measurements (`XMEAS 1–22`)** | 22 | Temperatures, pressures, levels, flow rates |
| **Sampled Reactor Composition (`XMEAS 23–28`)** | 6 | Sampled every 0.1 hr (Components A–F) |
| **Sampled Purge Gas Composition (`XMEAS 29–36`)** | 8 | Sampled every 0.1 hr (Components A–H) |
| **Sampled Product Composition (`XMEAS 37–41`)** | 5 | Sampled every 0.25 hr (Components D–H) |
| **Total Monitored Channels** | **52** | Full process integration across all units[cite: 1] |

---

## 🏗️ Methodology & Pipeline

```text
Raw .RData Files (Fault-Free / Faulty)
         │
         ▼
[ Data Ingestion & Preprocessing ] (pyreadr, pandas)
         │
         ├──> [ Signal Filtering ] ───> Rolling Window Moving Average (Noise Attenuation)
         │
         └──> [ Temporal Dynamics ] ──> Dynamic Lag Matrix Generation (statsmodels)
                                                    │
                   ┌────────────────────────────────┴──────────────────────────────┐
                   ▼                                                               ▼
        [ PCA Subspace Modeling ]                                   [ LightGBM Ensemble Reconstruction ]
  - Variance Retention: 90%                                               - All-for-one regression per feature
  - SPE Residual Thresholding                                             - Collective Mean Squared Error limit
