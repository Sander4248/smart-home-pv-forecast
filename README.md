# SmartHome PV Forecast (Prototype)

This repository contains the realistic forecasting pipeline developed as part of a bachelor's thesis for predicting PV yields for a single-family home. It covers day-ahead and hourly models based on forecast-supported weather data (e.g., Open-Meteo) and derived solar geometric characteristics (pvlib). Random Forest and XGBoost are implemented, as well as periodized training windows (month/season/year) and forecast horizons (D+1...D+7). The pipeline includes feature selection (expert-based sweep, forward selection), transparent baselines (persistence, 7-day average, weekday average), and evaluation with MAE, RMSE, R², sMAPE, nRMSE, bias, and energy balance deviation. The goal is to embed it in smart home energy management (e.g., charging/operating windows for EVs, storage, hot water). The code is **a prototype** from a **bachelor's thesis** and is provided without warranty.

# SPDX-License-Identifier: MIT

# Copyright (c) 2025 Alexander Sturm
