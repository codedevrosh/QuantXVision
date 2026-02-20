import streamlit as st

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="QuantXVision",
    page_icon="📈",
    layout="wide"
)

# -----------------------------------------------------
# CUSTOM CSS (CURVED CARDS + MODERN UI)
# -----------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 48px;
    font-weight: 700;
    text-align: center;
    margin-top: 20px;
}

.subtitle {
    font-size: 20px;
    text-align: center;
    color: gray;
    margin-bottom: 40px;
}

.card {
    padding: 25px;
    border-radius: 18px;
    background: #111827;
    box-shadow: 0 4px 18px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

.feature-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 8px;
}

.feature-desc {
    color: #9ca3af;
}

.section-title {
    font-size: 28px;
    font-weight: 600;
    margin-top: 40px;
    margin-bottom: 15px;
}

.center-text {
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# HEADER
# -----------------------------------------------------

st.markdown('<div class="main-title">📈 QuantXVision</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Hybrid Multi-Model Stock Prediction System for Intelligent Market Forecasting</div>',
    unsafe_allow_html=True
)

st.divider()

# -----------------------------------------------------
# ABOUT APP
# -----------------------------------------------------

st.markdown('<div class="section-title"> About This Platform</div>', unsafe_allow_html=True)

st.write("""
This platform is an stock market forecasting system designed to predict price movements
for NIFTY-50 companies using multiple advanced time series and machine learning models.

It integrates classical statistical models, deep learning architectures, and probabilistic trend modeling
to provide both short-term and long-term forecasts.

The system enables traders, analysts, and researchers to explore market behaviour, compare forecasting models,
and generate predictive insights from historical stock data.
""")

# -----------------------------------------------------
# FEATURE CARDS
# -----------------------------------------------------

st.markdown('<div class="section-title"> Core Features</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="feature-title">📊 Multi-Model Forecasting</div>
        <div class="feature-desc">
        Combines ARIMA, Facebook Prophet, LSTM Deep Learning,
        and Machine Learning regression models.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="feature-title">⏱ Short & Long Term Prediction</div>
        <div class="feature-desc">
        LSTM predicts short-term price movement,
        Prophet predicts long-term market trends.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="feature-title">📈 Confidence Intervals</div>
        <div class="feature-desc">
        Long-term forecasts include probabilistic confidence
        bands to quantify uncertainty.
        </div>
    </div>
    """, unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div class="card">
        <div class="feature-title">🔬 Model Comparison Lab</div>
        <div class="feature-desc">
        Compare forecasting accuracy across statistical,
        ML, and deep learning models.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="card">
        <div class="feature-title">🏢 Enterprise Architecture</div>
        <div class="feature-desc">
        Production-ready pipeline with model registry,
        batch training, and modular services.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div class="card">
        <div class="feature-title">📊 Interactive Analytics</div>
        <div class="feature-desc">
        Visual dashboards for market exploration,
        prediction, and performance analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------
# MODEL ARCHITECTURE
# -----------------------------------------------------

st.markdown('<div class="section-title"> Forecasting Architecture</div>', unsafe_allow_html=True)

st.write("""
**Short-Term Forecasting**
- Deep Learning LSTM neural networks
- Sequence-based temporal modeling
- Captures recent market momentum

**Long-Term Forecasting**
- Facebook Prophet probabilistic trend modeling
- Seasonal and macro-trend decomposition
- Multi-year projection capability

""")

# -----------------------------------------------------
# HOW TO USE
# -----------------------------------------------------

st.markdown('<div class="section-title"> How To Use</div>', unsafe_allow_html=True)

st.write("""
1️⃣ Navigate to **Prediction** page  
2️⃣ Select a stock from NIFTY-50  
3️⃣ Choose prediction horizon  
4️⃣ Generate forecast  
5️⃣ Explore results and confidence bands  

Use the **Market Analysis** section for historical exploration.
""")

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------

st.divider()

st.markdown(
    '<div class="center-text">Built with Machine Learning • Time Series Modeling • Deep Learning</div>',
    unsafe_allow_html=True
)

