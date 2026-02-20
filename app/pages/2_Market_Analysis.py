import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Market Analysis",
    page_icon="📈",
    layout="wide"
)
st.title(" Market Technical Analysis")

# =====================================================
# LOAD DATA
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nifty50_technical.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# =====================================================
# STOCK SELECTION (CLEAN DISPLAY)
# =====================================================

stocks = sorted(df["Stock"].unique())
display_map = {s.replace(".NS", ""): s for s in stocks}

selected_display = st.selectbox("Select Stock", list(display_map.keys()))
selected_stock = display_map[selected_display]

stock_df = df[df["Stock"] == selected_stock].sort_values("Date")

# =====================================================
# DATA DATE INFO
# =====================================================

data_start = stock_df["Date"].min().date()
data_end = stock_df["Date"].max().date()

st.caption(f" Data available from {data_start} to {data_end}")
st.caption(f" Latest market date: {data_end}")


# =====================================================
# DATE RANGE FILTER
# =====================================================

st.subheader("Select Analysis Period")

min_date = stock_df["Date"].min().date()
max_date = stock_df["Date"].max().date()

selected_range = st.date_input(
    "Choose date range",
    value=(max_date - pd.Timedelta(days=365), max_date),
    min_value=min_date,
    max_value=max_date
)

# Ensure user selected both dates
if len(selected_range) == 2:
    start_date, end_date = selected_range

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered_df = stock_df[
        (stock_df["Date"] >= start_date) &
        (stock_df["Date"] <= end_date)
    ].copy()

    # Handle empty result
    if filtered_df.empty:
        st.warning("No data available for selected date range.")
        st.stop()

    stock_df = filtered_df

    st.caption(f"Showing data from {start_date.date()} to {end_date.date()}")

else:
    st.warning("Please select both start and end date.")
    st.stop()


# =====================================================
# LATEST VALUES (FROM FILTERED DATA)
# =====================================================

latest = stock_df.iloc[-1]

price = latest["Close"]
rsi = latest["RSI_14"]
sma20 = latest["SMA_20"]
sma50 = latest["SMA_50"]
volatility = latest["Volatility_20"]

# =====================================================
# SIGNAL ENGINE
# =====================================================

def get_trend(price, sma20, sma50):
    if price > sma20 > sma50:
        return " Uptrend "
    elif price < sma20 < sma50:
        return " Downtrend "
    else:
        return "Sideways"

def get_rsi_signal(rsi):
    if rsi > 70:
        return "Overbought "
    elif rsi < 30:
        return "Oversold "
    else:
        return "Neutral"

trend = get_trend(price, sma20, sma50)
rsi_signal = get_rsi_signal(rsi)

# =====================================================
# METRIC CARDS
# =====================================================

st.subheader("Current Market Snapshot")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Current Price", f"{price:.2f}")
col2.metric("RSI", f"{rsi:.2f}")
col3.metric("Trend", trend)
col4.metric("Volatility", f"{volatility:.4f}")

st.divider()

# =====================================================
# PRICE + MOVING AVERAGE CHART
# =====================================================

st.subheader("Price Trend & Moving Averages")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=stock_df["Date"],
    y=stock_df["Close"],
    name="Close",
    line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=stock_df["Date"],
    y=stock_df["SMA_20"],
    name="SMA 20",
    line=dict(dash="dot")
))

fig.add_trace(go.Scatter(
    x=stock_df["Date"],
    y=stock_df["SMA_50"],
    name="SMA 50",
    line=dict(dash="dash")
))

fig.update_layout(height=450)
st.plotly_chart(fig, use_container_width=True)

# =====================================================
# RSI CHART
# =====================================================

st.subheader("RSI Indicator")

fig_rsi = go.Figure()

fig_rsi.add_trace(go.Scatter(
    x=stock_df["Date"],
    y=stock_df["RSI_14"],
    name="RSI"
))

fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")

fig_rsi.update_layout(height=300)
st.plotly_chart(fig_rsi, use_container_width=True)

# =====================================================
# VOLATILITY CHART
# =====================================================

st.subheader("Volatility (Risk Level)")

fig_vol = go.Figure()

fig_vol.add_trace(go.Scatter(
    x=stock_df["Date"],
    y=stock_df["Volatility_20"],
    name="Volatility"
))

fig_vol.update_layout(height=300)
st.plotly_chart(fig_vol, use_container_width=True)

# =====================================================
# INTERPRETATION PANEL
# =====================================================

st.divider()
st.subheader(" Market Interpretation")

col1, col2 = st.columns(2)

with col1:
    st.info(f"Trend Status: {trend}")

with col2:
    st.info(f"RSI Condition: {rsi_signal}")

# =====================================================
# TRADING SIGNAL
# =====================================================

def trading_signal(price, sma20, rsi):
    if price > sma20 and rsi < 70:
        return "BUY "
    elif price < sma20 and rsi > 30:
        return "SELL "
    else:
        return "HOLD "

signal = trading_signal(price, sma20, rsi)

st.success(f"Suggested Action: {signal}")

