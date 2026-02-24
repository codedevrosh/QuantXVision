#  Multi-Stock Price Prediction & Forecasting Platform

An end-to-end machine learning system for predicting short-term and long-term price movements of Nifty 50 stocks using deep learning and time-series forecasting techniques.

 **Live Demo:** _Add Hugging Face Link Here_  
 **Models Used:** LSTM • Prophet • Random Forest • ARIMA  
 **Deployment:** Streamlit + Hugging Face Spaces  

---

##  Project Description

This project builds a comprehensive multi-stock forecasting framework designed to analyze and predict stock price behavior using historical market data from 2010 to 2025.

The system integrates:

- **LSTM** for short-term next-day prediction  
- **Prophet** for long-term trend forecasting  
- **Random Forest** for feature-based regression  
- **ARIMA** as a statistical benchmark  

Unlike single-model stock prediction projects, this system performs comparative model analysis and deploys the most effective approaches in a real-time web application.

The final platform enables:

- Short-term price forecasting  
- Long-term trend projection  
- Model performance comparison  
- Interactive cloud deployment  

---

## Project Structure

The project follows a modular architecture to ensure scalability, maintainability, and separation between training and deployment.

---

## 🛠 Technologies Used

### Programming & Data Handling
- Python
- Pandas
- NumPy

### Machine Learning & Deep Learning
- TensorFlow / Keras (LSTM)
- Scikit-learn (Random Forest)
- Statsmodels (ARIMA)
- Prophet (Time-Series Forecasting)

### Data Collection
- yfinance

### Visualization
- Matplotlib

### Deployment
- Streamlit
- Hugging Face Spaces

---

## How It Works

### Data Collection
Historical stock price data is collected using the `yfinance` library for Nifty 50 stocks (2010–2025).

### Data Preprocessing
- Cleaning & validation  
- Datetime conversion  
- Sorting by stock & date  
- Missing value handling  

### Feature Engineering
Technical indicators generated:

- SMA (20 & 50 day)
- RSI (14 day)
- Rolling Volatility (20 day)
- Daily Returns

### Model Training

**LSTM (Short-Term Forecasting)**
- 60-day lookback window
- Stacked LSTM layers
- MinMax scaling
- Best performing model

**Prophet (Long-Term Forecasting)**
- Yearly seasonality
- Trend decomposition
- Multiplicative seasonality mode

**Random Forest**
- Ensemble regression model
- Uses engineered technical indicators

**ARIMA**
- Stationarity enforced via ADF test
- Linear statistical baseline

---

## Deployment

- Models are serialized after training
- Loaded dynamically in Streamlit app
- User selects stock and forecast type
- Predictions generated in real time
- Hosted on Hugging Face Spaces

This demonstrates a complete machine learning lifecycle from data engineering to production deployment.

---

## Limitations

- Uses historical price data only
- No sentiment or macroeconomic signals
- No real-time streaming integration
- Not an automated trading system

---

## Future Enhancements

- Integrate financial news sentiment analysis
- Add macroeconomic indicators
- Implement portfolio optimization
- Enable real-time data streaming
- Explore reinforcement learning trading models

---

##  Author

Arockia Roshan  
Machine Learning & Data Science Enthusiast


