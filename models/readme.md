# Model Artifacts

This directory contains trained model artifacts used in the deployed forecasting system.

Due to GitHub file size limitations, large model files (.h5, .pkl) are not included directly in this repository.

## Accessing Models

All trained models are available through the deployed Hugging Face Space:

👉 https://huggingface.co/spaces/codedevrosh/QuantXVision/tree/main/models

The deployment environment includes:

- LSTM models (short-term forecasting)
- Prophet models (long-term forecasting)
- Random Forest models
- ARIMA model (Reliance)

## Reproducing Models

To retrain the models locally:

1. Ensure processed dataset is available in `data/processed/`
2. Run training scripts from the `src/` directory:
   - `train_lstm_model.py`
   - `train_prophet_model.py`
   - `train_rf_model.py`
   - `arima_model.py`

This repository provides the full reproducible training pipeline.
