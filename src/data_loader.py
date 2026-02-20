import pandas as pd
from pathlib import Path
import yfinance as yf
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


NIFTY50_SYMBOLS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS",
    "TITAN.NS","ULTRACEMCO.NS","NESTLEIND.NS","BAJFINANCE.NS","WIPRO.NS",
    "HCLTECH.NS","POWERGRID.NS","NTPC.NS","ONGC.NS","JSWSTEEL.NS",
    "TATASTEEL.NS","COALINDIA.NS","BAJAJFINSV.NS","GRASIM.NS","HDFCLIFE.NS",
    "SBILIFE.NS","DRREDDY.NS","CIPLA.NS","EICHERMOT.NS","HEROMOTOCO.NS",
    "APOLLOHOSP.NS","DIVISLAB.NS","BRITANNIA.NS","ADANIPORTS.NS","INDUSINDBK.NS",
    "TECHM.NS","TATAMOTORS.NS","BAJAJ-AUTO.NS","UPL.NS","HINDALCO.NS",
    "BPCL.NS","SHREECEM.NS","M&M.NS","ADANIENT.NS","IOC.NS"
]


def download_all_stock_data():

    all_data = []

    for symbol in NIFTY50_SYMBOLS:

        print("Downloading:", symbol)

        df = yf.download(
            symbol,
            start="2010-01-01",
            end="2025-01-01",
            progress=False,
            threads=False
        )

        if df.empty:
            print("Skipped:", symbol)
            continue

        # ======================================
        # CRITICAL STEP — FLATTEN MULTIINDEX
        # ======================================
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # reset index
        df = df.reset_index()

        # add stock identifier
        df["Stock"] = symbol

        # remove column name hierarchy label
        df.columns.name = None

        # append vertically
        all_data.append(df)

        time.sleep(1)

    final_df = pd.concat(all_data, axis=0, ignore_index=True)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(DATA_RAW / "nifty50_all_stock_data.csv")

    print("All stock data downloaded successfully!")


if __name__ == "__main__":
    download_all_stock_data()