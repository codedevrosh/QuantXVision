import streamlit as st
import pandas as pd
import plotly.express as px
from utils.stock_utils import clean_name, model_name

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(page_title="DashBoard",
    page_icon="📈",
    layout="wide")

st.title(" Market Intelligence Dashboard")
st.markdown("Real-time overview of NIFTY50 stock data and model coverage")

# ----------------------------------
# LOAD DATA
# ----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/nifty50_processed.csv")

df = load_data()

# ----------------------------------
# TOP METRICS ROW
# ----------------------------------
col1, col2, col3, col4 = st.columns(4)

total_stocks = df["Stock"].nunique()
total_records = len(df)
latest_date = df["Date"].max()

col1.metric("Tracked Stocks", total_stocks)
col2.metric("Total Records", f"{total_records:,}")
col3.metric("Latest Data", latest_date)
col4.metric("Models Active", total_stocks)  # assuming all trained

st.divider()

# ----------------------------------
# STOCK PRICE TREND
# ----------------------------------
st.subheader(" Stock Price Trends")

display_names = sorted(df["Stock"].apply(clean_name).unique())
selected_display = st.selectbox("Stock", display_names)
selected_stock = model_name(selected_display)

stock_df = df[df["Stock"] == selected_stock]

fig = px.line(
    stock_df,
    x="Date",
    y="Close",
    title=f"{selected_display} Price Movement",
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------
# MODEL COVERAGE
# ----------------------------------
st.subheader(" Model Coverage")

coverage_data = pd.DataFrame({
    "Model": ["LSTM", "Prophet", "ARIMA", "ML Regression"],
    "Coverage": [49, 49, 25, 49]  # adjust if needed
})

fig2 = px.bar(
    coverage_data,
    x="Model",
    y="Coverage",
    color="Model",
    title="Models Available Per Stock"
)

st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------
# SECTOR DISTRIBUTION (optional if available)
# ----------------------------------
if "Sector" in df.columns:
    st.subheader("🏢 Sector Distribution")

    sector_counts = df[["Stock", "Sector"]].drop_duplicates()["Sector"].value_counts()

    fig3 = px.pie(
        values=sector_counts.values,
        names=sector_counts.index,
        title="Market Sector Composition"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------
# QUICK INSIGHTS PANEL
# ----------------------------------
st.subheader(" Market Insights")

colA, colB = st.columns(2)

with colA:
    top_gainer = (
        df.groupby("Stock")["Close"]
        .apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
        .idxmax()
    )
    st.success(f"Top Performing Stock: {top_gainer}")

with colB:
    most_volatile = (
        df.groupby("Stock")["Close"]
        .std()
        .idxmax()
    )
    st.warning(f"Most Volatile Stock: {most_volatile}")

st.divider()

st.info("Dashboard shows historical performance and model coverage. Use Forecasting page for predictions.")

