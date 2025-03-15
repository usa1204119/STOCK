import streamlit as st
import pandas as pd
import yfinance as yf
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

# 🚀 Streamlit App Title
st.title('📈 Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App!')
st.sidebar.info("Created by [Utkarsh Sharma](https://www.linkedin.com/in/utkarsh-sharma-143b1720a/)")

# 🚀 Sidebar: Stock Symbol & Date Inputs
option = st.sidebar.text_input('Enter Stock Symbol', value='SPY').upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter Duration (days)', value=3000)
start_date = st.sidebar.date_input('Start Date', value=today - datetime.timedelta(days=duration))
end_date = st.sidebar.date_input('End Date', today)

# 🚀 Download Stock Data
@st.cache_resource
def download_data(symbol, start_date, end_date):
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if df.empty:
            st.error("⚠️ No data found for this stock symbol. Please try another one.")
        return df
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

data = download_data(option, start_date, end_date)
scaler = StandardScaler()

# 🚀 Sidebar: Button to Fetch Data
if st.sidebar.button('Fetch Data'):
    if start_date < end_date:
        st.sidebar.success(f'Start Date: `{start_date}`\nEnd Date: `{end_date}`')
    else:
        st.sidebar.error('⚠️ Error: End date must be after start date')

# 🚀 Main Navigation
def main():
    page = st.sidebar.selectbox('Choose an Option', ['📊 Visualize', '📅 Recent Data', '🔮 Predict'])
    if page == '📊 Visualize':
        tech_indicators()
    elif page == '📅 Recent Data':
        dataframe()
    else:
        predict()

# 🚀 Technical Indicators
def tech_indicators():
    """ Display technical indicators with TA library """
    st.header('📊 Technical Indicators')
    option = st.radio('Select Indicator', ['Close', 'MACD', 'RSI', 'SMA', 'EMA'])

    if data.empty:
        st.error("⚠️ No stock data available. Please check the stock symbol or date range.")
        return

    # Ensure 'Close' is a Series
    close_series = data[['Close']].squeeze()  # Fix shape issue

   

    # Compute Other Indicators
    data['MACD'] = MACD(close_series).macd()
    data['RSI'] = RSIIndicator(close_series).rsi()
    data['SMA'] = SMAIndicator(close_series, window=14).sma_indicator()
    data['EMA'] = EMAIndicator(close_series).ema_indicator()

    # Display Selected Indicator
    if option == 'Close':
        st.line_chart(data['Close'])
    elif option == 'MACD':
        st.line_chart(data['MACD'])
    elif option == 'RSI':
        st.line_chart(data['RSI'])
    elif option == 'SMA':
        st.line_chart(data['SMA'])
    else:
        st.line_chart(data['EMA'])

# 🚀 Show Recent Data
def dataframe():
    st.header('📅 Recent Stock Data')
    if data.empty:
        st.error("⚠️ No data available.")
    else:
        st.dataframe(data.tail(10))

# 🚀 Prediction Models
def predict():
    """ Predict stock prices using different ML models """
    st.header('🔮 Stock Price Prediction')
    model_choice = st.radio('Choose Model', [
        'Linear Regression', 'Random Forest', 'Extra Trees', 'KNN', 'XGBoost'
    ])
    days = int(st.number_input('Forecast Days', value=5))

    if st.button('Predict'):
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'Extra Trees': ExtraTreesRegressor(),
            'KNN': KNeighborsRegressor(),
            'XGBoost': XGBRegressor()
        }
        model = models[model_choice]
        model_engine(model, days)

# 🚀 Train & Predict Stock Prices
def model_engine(model, days):
    if data.empty:
        st.error("⚠️ No stock data available. Cannot perform prediction.")
        return

    df = data[['Close']].copy()
    df['Future'] = df['Close'].shift(-days)

    # Prepare Data
    X = df.drop(['Future'], axis=1).values
    y = df['Future'].values[:-days]  # Avoid NaN values in labels
    X_forecast = X[-days:]  # Future prediction set

    # Scale Data
    X = scaler.fit_transform(X[:-days])
    X_forecast = scaler.transform(X_forecast)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Display Metrics
    st.text(f'📊 Model Performance:\nR² Score: {r2_score(y_test, preds):.4f}\nMAE: {mean_absolute_error(y_test, preds):.4f}')

    # Predict Future Prices
    forecast_pred = model.predict(X_forecast)
    st.header('📆 Predicted Prices')
    for i, price in enumerate(forecast_pred, start=1):
        st.text(f'Day {i}: ${price:.2f}')

# 🚀 Run App
if __name__ == '__main__':
    main()
