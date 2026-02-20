import pandas as pd
from pathlib import Path

# =====================================================
# PROJECT PATHS (robust)
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nifty50_processed.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "nifty50_technical.csv"

# =====================================================
# TECHNICAL INDICATORS FOR ONE STOCK
# =====================================================

def compute_indicators(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators for a single stock dataframe
    """

    stock_df = stock_df.sort_values("Date").copy()

    # -----------------------------
    # Moving Averages (Trend)
    # -----------------------------
    stock_df["SMA_20"] = stock_df["Close"].rolling(20).mean()
    stock_df["SMA_50"] = stock_df["Close"].rolling(50).mean()

    # -----------------------------
    # Returns (Momentum)
    # -----------------------------
    stock_df["Returns"] = stock_df["Close"].pct_change()

    # -----------------------------
    # Volatility (Risk)
    # -----------------------------
    stock_df["Volatility_20"] = stock_df["Returns"].rolling(20).std()

    # -----------------------------
    # RSI (Momentum Strength)
    # -----------------------------
    window = 14
    delta = stock_df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    stock_df["RSI_14"] = 100 - (100 / (1 + rs))

    return stock_df


# =====================================================
# BUILD INDICATORS FOR ALL STOCKS
# =====================================================

def build_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply indicator computation to every stock
    """

    processed_list = []
    stocks = df["Stock"].unique()

    for stock in stocks:
        print("Processing:", stock)
        stock_df = df[df["Stock"] == stock].copy()
        stock_df = compute_indicators(stock_df)
        processed_list.append(stock_df)

    df_final = pd.concat(processed_list)
    df_final = df_final.dropna()

    return df_final


# =====================================================
# MAIN PIPELINE
# =====================================================

def run_feature_pipeline():
    """
    Main execution pipeline
    """

    print("\n========== FEATURE ENGINE START ==========")

    print("Loading base dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    print("Computing indicators...")
    df_technical = build_all_indicators(df)

    print("Saving technical dataset...")
    df_technical.to_csv(OUTPUT_PATH, index=False)

    print("Saved to:", OUTPUT_PATH)
    print("Final rows:", len(df_technical))
    print("========== FEATURE ENGINE COMPLETE ==========\n")


# =====================================================
# SCRIPT EXECUTION
# =====================================================

if __name__ == "__main__":
    run_feature_pipeline()
