# Leather Goods Forecast

A forecasting tool that combines classical time series methods (Prophet, SARIMA) with a machine learning model (Random Forest) to forecast future demand for leather goods. This repository provides an end-to-end example, including data preparation, model training, validation, and predictions.

## Features
- Uses Prophet for baseline seasonal-trend forecasting.
- Incorporates a Random Forest model for feature-based forecasting.
- Adds a SARIMA model for traditional time series modeling.
- Ensembles predictions from all three models.
- Integrates holiday information and fourier terms for seasonality.
- Includes a validation step for historical error computation and uncertainty estimation.

## Requirements
Install the dependencies using:
```bash
pip install -r requirements.txt
