import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nifty50_technical.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "rf"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# CONFIGURATION
# =====================================================

FEATURES = [
    "SMA_20",
    "SMA_50",
    "RSI_14",
    "Volatility_20",
    "Returns"
]

MIN_ROWS_REQUIRED = 200

LOG = "[ML TRAIN]"

# =====================================================
# TRAINING FUNCTION FOR ONE STOCK
# =====================================================

def train_single_stock(stock_df):

    stock_df = stock_df.sort_values("Date").copy()

    # Predict next day close
    stock_df["Target"] = stock_df["Close"].shift(-1)
    stock_df = stock_df.dropna()

    if len(stock_df) < MIN_ROWS_REQUIRED:
        return None

    X = stock_df[FEATURES]
    y = stock_df["Target"]

    split = int(len(X) * 0.8)

    X_train = X[:split]
    y_train = y[:split]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model

# =====================================================
# MAIN TRAINING PIPELINE
# =====================================================

def run_training():

    print("\n==============================")
    print("ML BATCH TRAINING STARTED")
    print("==============================")

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    stocks = sorted(df["Stock"].unique())

    trained = 0
    skipped = 0
    failed = 0

    for stock in stocks:

        try:
            print(f"\n{LOG} Processing {stock}")

            model_path = MODEL_DIR / f"{stock}.pkl"

            # ---------------------------
            # Skip existing model
            # ---------------------------
            if model_path.exists():
                print(f"{LOG} Skipped (already trained)")
                skipped += 1
                continue

            stock_df = df[df["Stock"] == stock].copy()

            model = train_single_stock(stock_df)

            if model is None:
                print(f"{LOG} Skipped (insufficient data)")
                skipped += 1
                continue

            joblib.dump(model, model_path)
            print(f"{LOG} Saved → {model_path.name}")

            trained += 1

        except Exception as e:
            print(f"{LOG} FAILED for {stock}")
            print("Error:", e)
            failed += 1

    print("\n==============================")
    print("ML TRAINING COMPLETE")
    print("==============================")
    print("Trained :", trained)
    print("Skipped :", skipped)
    print("Failed  :", failed)
    print("Saved in:", MODEL_DIR)
    print("==============================\n")

# =====================================================
# SCRIPT ENTRY
# =====================================================

if __name__ == "__main__":
    run_training()
