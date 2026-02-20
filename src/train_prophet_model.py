import pandas as pd
import joblib
from pathlib import Path
from prophet import Prophet

# =========================================================
# PROJECT PATHS (robust resolution)
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT/ "models" / "prophet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# TRAINING CONFIGURATION
# =========================================================

MIN_DATA_POINTS = 300        # minimum history required
LOG_PREFIX = "[PROPHET TRAIN]"

# =========================================================
# LOAD DATA
# =========================================================

print(f"{LOG_PREFIX} Loading dataset...")

df = pd.read_csv(DATA_PATH/"nifty50_processed.csv")

stocks = sorted(df["Stock"].unique())

print(f"{LOG_PREFIX} Total stocks found: {len(stocks)}")

# =========================================================
# TRAINING LOOP
# =========================================================

trained = 0
skipped = 0
failed = 0

for stock in stocks:

    try:
        print(f"\n{LOG_PREFIX} Processing {stock}")

        model_path = MODEL_DIR / f"{stock}.pkl"

        # -------------------------------------------------
        # Skip already trained
        # -------------------------------------------------
        if model_path.exists():
            print(f"{LOG_PREFIX} Skipped (already trained)")
            skipped += 1
            continue

        stock_df = df[df["Stock"] == stock].copy()
        stock_df = stock_df.sort_values("Date")

        # -------------------------------------------------
        # Check minimum data
        # -------------------------------------------------
        if len(stock_df) < MIN_DATA_POINTS:
            print(f"{LOG_PREFIX} Skipped (insufficient data: {len(stock_df)})")
            skipped += 1
            continue

        # -------------------------------------------------
        # Prophet format
        # -------------------------------------------------
        prophet_df = stock_df[["Date", "Close"]]
        prophet_df.columns = ["ds", "y"]

        # -------------------------------------------------
        # LONG TERM OPTIMIZED MODEL
        # -------------------------------------------------
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_mode="multiplicative"
        )

        model.fit(prophet_df)

        # -------------------------------------------------
        # Save model
        # -------------------------------------------------
        joblib.dump(model, model_path)

        print(f"{LOG_PREFIX} Saved → {model_path.name}")
        trained += 1

    except Exception as e:
        print(f"{LOG_PREFIX} FAILED for {stock}")
        print("Error:", e)
        failed += 1


# =========================================================
# SUMMARY
# =========================================================

print("\n==========================================")
print(" PROPHET TRAINING COMPLETE")
print("==========================================")
print("New models trained :", trained)
print("Skipped            :", skipped)
print("Failed             :", failed)
print("Saved in           :", MODEL_DIR)
print("==========================================")
