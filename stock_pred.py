import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np 


# Streamlit app title
st.title("Stock Prediction Application")

# Input for stock ticker
ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", value="AAPL")

# Fetch historical data
@st.cache_data
def fetch_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period='5y')[['Close']]

if ticker:
    data = fetch_data(ticker)

    # Display historical data
    st.subheader("Historical Data")
    st.line_chart(data['Close'])

    # Add moving averages
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # Linear Regression for Next-Day Prediction
    data['Target'] = data['Close'].shift(-1)
    data = data.dropna()

    X = data[['Close']]
    y = data['Target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict for the next 5 days
    predictions = []
    last_price = data['Close'].iloc[-1]
    
    for _ in range(5):  # Predict for the next 5 days
        next_day_prediction = model.predict([[last_price]])[0]
        predictions.append(next_day_prediction)
        last_price = next_day_prediction  # Update the last price for the next prediction

    # Prepare dates for the next 5 trading days
    prediction_dates = []
    current_date = data.index[-1]
    
    for _ in range(5):
        # Find the next trading day (skip weekends/holidays)
        current_date += pd.Timedelta(days=1)
        while current_date.weekday() >= 5:  # If Saturday or Sunday, skip
            current_date += pd.Timedelta(days=1)
        prediction_dates.append(current_date)

    prediction_df = pd.DataFrame({'Date': prediction_dates, 'Prediction': predictions})
    prediction_df.set_index('Date', inplace=True)

    # Combine historical data, moving averages, and predictions for visualization
    data_for_plot = data.copy()
    
    # Ensure today's data is included for the plot (i.e., the last actual data point)
    data_for_plot['Prediction'] = np.nan  # Initialize the prediction column
    for date, prediction in zip(prediction_dates, predictions):
        data_for_plot.loc[date, 'Prediction'] = prediction

    st.subheader(f"{ticker} Stock Price and Predictions")
    st.line_chart(data_for_plot[['Close', 'SMA_10', 'SMA_50', 'Prediction']])

    # Display predictions for the next week
    st.subheader("Next Week's Predictions")
    for date, prediction in zip(prediction_dates, predictions):
        st.write(f"Predicted Closing Price for **{date.strftime('%Y-%m-%d')}**: **${prediction:.2f}**")
