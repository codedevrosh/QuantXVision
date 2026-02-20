import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import joblib
import matplotlib.pyplot as plt


# =====================================================
# PROJECT PATHS
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# LOAD ONE STOCK
# =====================================================
def load_stock(stock_name="RELIANCE.NS"):
    df = pd.read_csv(DATA_PROCESSED / "nifty50_processed.csv")

    stock_df = df[df["Stock"] == stock_name].sort_values("Date")
    stock_df = stock_df.set_index("Date")
    stock_df = stock_df.asfreq("B").ffill()

    stock_df["Close"] = stock_df["Close"].ffill()
    stock_df = stock_df.dropna()
    return stock_df["Close"]


# =====================================================
# CHECK STATIONARITY (ADF TEST)
# =====================================================
def check_stationarity(series):
    series = series.dropna()

    if len(series) < 10:
        print("Series too short for ADF test")
        return True  # skip stationarity enforcement

    result = adfuller(series)

    print("ADF Statistic:", result[0])
    print("p-value:", result[1])

    if result[1] < 0.05:
        print("Series is stationary ✅")
        return True
    else:
        print("Series is NOT stationary ❌")
        return False


# =====================================================
# DIFFERENCE UNTIL STATIONARY
# =====================================================
def make_stationary(series):

    d = 0
    temp_series = series.copy()

    while True:

        if len(temp_series.dropna()) < 10:
            print("Series too small — stopping differencing")
            break

        if check_stationarity(temp_series):
            break

        temp_series = temp_series.diff().dropna()
        d += 1
        print("Differencing applied:", d)

    return temp_series, d


# =====================================================
# TRAIN ARIMA
# =====================================================
def train_arima(series, order):

    model = ARIMA(series, order=order)
    fitted_model = model.fit()

    return fitted_model


# =====================================================
# TRAIN TEST SPLIT
# =====================================================
def time_split(series, split_ratio=0.8):

    split_index = int(len(series) * split_ratio)

    train = series[:split_index]
    test = series[split_index:]

    return train, test


# =====================================================
# FORECAST
# =====================================================
def forecast(model, steps):
    return model.forecast(steps=steps)


# =====================================================
# MAIN PIPELINE
# =====================================================
def run_arima(stock_name="RELIANCE.NS"):

    print("Loading stock data...")
    series = load_stock(stock_name)

    print("Checking stationarity...")
    stationary_series, d = make_stationary(series)

    print("Splitting data...")
    train, test = time_split(stationary_series)

    print("Training ARIMA model...")
    order = (5, d, 0)   # simple baseline order
    model = train_arima(train, order)

    print(model.summary())

    print("Forecasting...")
    predictions = forecast(model, len(test))

    print("Saving model...")
    joblib.dump(model, MODEL_DIR / f"arima_{stock_name}.pkl")

    print("Plotting results...")
    plt.figure(figsize=(12,6))
    plt.plot(test.index, test, label="Actual")
    plt.plot(test.index, predictions, label="Forecast")
    plt.legend()
    plt.title(f"ARIMA Forecast — {stock_name}")
    plt.show()

    print("ARIMA pipeline completed ✅")


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    run_arima("RELIANCE.NS")
