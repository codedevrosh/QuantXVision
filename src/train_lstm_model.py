import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT/ "models" / "lstm"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# PARAMETERS
# =========================

LOOKBACK = 60
EPOCHS = 15
BATCH_SIZE = 32

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PROCESSED/ "nifty50_processed.csv")

stocks = sorted(df["Stock"].unique())

print("Total stocks:", len(stocks))

# =========================
# SEQUENCE BUILDER
# =========================

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# =========================
# TRAIN LOOP
# =========================

for stock in stocks:

    print(f"\nTraining {stock}")

    stock_df = df[df["Stock"] == stock].sort_values("Date")
    series = stock_df["Close"].values.reshape(-1,1)

    if len(series) < 200:
        print("Skipping — not enough data")
        continue

    # -------- scale ----------
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    # -------- sequences -------
    X, y = create_sequences(scaled, LOOKBACK)

    # -------- model -----------
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # -------- save model -------
    name = stock.replace(".NS","")

    model.save(MODEL_DIR / f"{name}.keras")
    joblib.dump(scaler, MODEL_DIR / f"{name}_scaler.pkl")

    print("Saved:", name)

print("\nALL STOCK MODELS TRAINED")
