import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT/ "models" / "lstm"
LOOKBACK = 60

def predict_lstm(stock_df, stock, future_days):

    name = stock.replace(".NS","")

    model = load_model(MODEL_DIR / f"{name}.keras")
    scaler = joblib.load(MODEL_DIR / f"{name}_scaler.pkl")

    series = stock_df["Close"].values.reshape(-1,1)
    scaled = scaler.transform(series)

    window = scaled[-LOOKBACK:].reshape(1,LOOKBACK,1)

    preds = []

    for _ in range(future_days):
        p = model.predict(window, verbose=0)[0][0]
        preds.append(p)
        window = np.append(window[:,1:,:], [[[p]]], axis=1)

    preds = scaler.inverse_transform(
        np.array(preds).reshape(-1,1)
    )

    dates = pd.bdate_range(
        start=stock_df["Date"].iloc[-1],
        periods=future_days+1
    )[1:]

    return dates, preds
