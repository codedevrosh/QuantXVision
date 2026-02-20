import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

def load_data():
    df = pd.read_csv(DATA_RAW / "nifty50_all_stock_data.csv")

    # remove index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # select OHLCV structure
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Stock']]

    # correct data types
    df["Date"] = pd.to_datetime(df["Date"])

    num_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # enforce time ordering
    df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)

    # validation
    print("Rows:", len(df))
    print("Stocks:", df["Stock"].nunique())
    print("Missing:", df.isnull().sum().sum())
    print("Duplicates:", df.duplicated().sum())

    return df

def run_preprocessing():

    print("Loading data...")
    final_df = load_data()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(DATA_PROCESSED / "nifty50_processed.csv")

    print("Data cleaned successfully!")

if __name__ == "__main__":
    run_preprocessing()