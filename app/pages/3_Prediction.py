import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from data.loader import load_data
from models.lstm_service import predict_lstm
from models.prophet_service import predict_prophet

st.set_page_config(
    page_title="Prediction",
    page_icon="📈",
    layout="wide"
)
# =====================================================
# HEADER
# =====================================================

st.header(" Forecasting Engine")

# =====================================================
# LOAD DATA
# =====================================================

df = load_data()

# clean display (remove .NS)
display_map = {s.replace(".NS", ""): s for s in sorted(df["Stock"].unique())}

selected_display = st.selectbox("Stock", list(display_map.keys()))
stock = display_map[selected_display]

sdf = df[df["Stock"] == stock].sort_values("Date")

# =====================================================
# FORECAST MODE
# =====================================================

mode = st.radio("Forecast Horizon", ["Short Term", "Long Term"], horizontal=True)

if mode == "Short Term":
    days = st.slider("Prediction Window (Days)", 5, 90, 30)
else:
    years = st.slider("Prediction Horizon (Years)", 1, 10, 3)

st.divider()

# =====================================================
# RUN FORECAST
# =====================================================

if st.button("Predict", use_container_width=True):

    with st.spinner("Running model..."):

        try:

            # -----------------------------------------
            # LSTM
            # -----------------------------------------
            if mode == "Short Term":

                dates, preds = predict_lstm(sdf, stock, days)
                lower = None
                upper = None

            # -----------------------------------------
            # PROPHET
            # -----------------------------------------
            else:
                dates, preds = predict_prophet(stock, years)
                lower = None
                upper = None

            dates = np.array(dates).flatten()
            preds = np.array(preds).flatten()
            # =====================================================
            # INTERACTIVE PLOT
            # =====================================================

            fig = go.Figure()

            # historical
            fig.add_trace(go.Scatter(
                x=sdf["Date"].iloc[-200:],
                y=sdf["Close"].iloc[-200:],
                mode="lines",
                name="Historical"
            ))

            # forecast
            fig.add_trace(go.Scatter(
                x=dates,
                y=preds,
                mode="lines",
                name="Forecast",
                line=dict(color="red", width=3)
            ))

            # confidence interval (prophet)
            if lower is not None:
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=upper,
                    line=dict(width=0),
                    showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=dates,
                    y=lower,
                    fill="tonexty",
                    name="Confidence Interval",
                    line=dict(width=0),
                    opacity=0.3
                ))

            fig.update_layout(
                title=f"{selected_display} Forecast",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x unified",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # =====================================================
            # SHOW FORECAST VALUES TABLE
            # =====================================================

            st.subheader("Predicted Values")

            result_df = pd.DataFrame({
                "Date": dates,
                "Predicted Price": preds
            })

            if lower is not None:
                result_df["Lower Bound"] = lower
                result_df["Upper Bound"] = upper

            st.dataframe(result_df.reset_index(drop=True))

        except Exception as e:
            st.error("Prediction failed")
            st.exception(e)

