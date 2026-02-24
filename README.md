#  Multi-Stock Price Prediction and Analysis Using Machine Learning and Time-Series Forecasting

An end-to-end machine learning system for predicting short-term and long-term price movements of Nifty 50 stocks using deep learning and time-series forecasting techniques.

 **Live Demo:** _https://huggingface.co/spaces/codedevrosh/QuantXVision_  
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

```
multi-stock-prediction/
│
├── app/ # Streamlit web application 
│ ├── analytics/
│ │ └── indicators.py # Technical indicator logic for dashboard analytics
│ │
│ ├── components/
│ │ └── metric_card.py # Reusable UI metric display component
│ │
│ ├── core/
│ │ ├── config.py # App-level configuration settings
│ │ └── styles.py # Custom UI styling definitions
│ │
│ ├── data/
│ │ └── loader.py # Loads processed data for app usage
│ │
│ ├── models/
│ │ ├── lstm_service.py # LSTM inference service for short-term prediction
│ │ └── prophet_service.py # Prophet inference service for long-term forecasting
│ │
│ ├── pages/
│ │ ├── 1_Dashboard.py # Main overview dashboard
│ │ ├── 2_Market_Analysis.py # Market trend & analytics page
│ │ └── 3_Prediction.py # Stock prediction interface
│ │
│ ├── utils/
│ │ └── stock_utils.py # Helper functions for stock processing
│ │
│ └── Home.py # Application landing page
│
├── data/
│ ├── raw/
│ │ └── nifty50_all_stock_data.csv # Raw downloaded stock dataset
│ │
│ └── processed/
│ ├── nifty50_processed.csv # Cleaned OHLCV dataset
│ └── nifty50_technical.csv # Feature-engineered dataset with indicators
│
├── models/
│ ├── lstm/ # Saved trained LSTM models + scalers
│ ├── prophet/ # Saved Prophet models
│ ├── rf/ # Saved Random Forest models
│ └── arima_RELIANCE.NS.pkl # Trained ARIMA model for Reliance
│
├── notebooks/ # Research & experimentation notebooks
│ ├── arima_model.ipynb # ARIMA modeling notebook
│ ├── data_preprocessing.ipynb # Data cleaning & preprocessing notebook
│ ├── lstm_model.ipynb # LSTM training notebook
│ ├── prophet_model.ipynb # Prophet training notebook
│ ├── rf_model.ipynb # Random Forest modeling notebook
│ └── technical_features.ipynb # Feature engineering notebook
│
├── src/ # Core training & pipeline logic
│ ├── models/
│ │ └── classical/
│ │ └── arima_model.py # ARIMA statistical model implementation
│ │
│ ├── init.py # Package initializer
│ ├── data_loader.py # Centralized data loading utility
│ ├── features.py # Technical feature computation logic
│ ├── preprocess_data.py # Data cleaning & validation script
│ ├── train_lstm_model.py # LSTM training pipeline
│ ├── train_prophet_model.py # Prophet training pipeline
│ └── train_rf_model.py # Random Forest training pipeline
│
├── Project_Report.docx # Full project documentation
├── README.md # Project overview documentation
└── requirements.txt # Python dependencies
```
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
## Conclusion

This project demonstrates a complete end-to-end stock forecasting system combining deep learning and classical time-series models. The comparative analysis confirms that LSTM significantly outperforms traditional statistical approaches for short-term prediction, while Prophet provides stable long-term trend forecasting. 

By integrating modeling, evaluation, and cloud deployment, this platform showcases a practical and production-ready application of machine learning in financial time-series analysis.

---

##  Author

Arockia Roshan A  
Machine Learning & Data Science Enthusiast










