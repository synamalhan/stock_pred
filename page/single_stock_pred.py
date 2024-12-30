import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from utils import calculate_rsi, train_rf_model_with_graphs, train_lstm_model_with_graphs
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def single_stock_page():
    st.header("Single Stock Prediction")

    # Stock Ticker Input
    ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", value="AAPL")

    if ticker:
        # Fetch stock data from Yahoo Finance
        stock = yf.Ticker(ticker)
        data = stock.history(period="5y")
        today_price = stock.history(period="1d")['Close'][-1]
        stock_info = stock.info

        # Display stock information
        st.subheader("Stock Information")
        st.write(f"**Name**: {stock_info.get('longName', 'N/A')}")
        st.write(f"**Symbol**: {ticker}")
        st.write(f"**Today's Price**: ${today_price:.2f}")

        pred = st.container()
        # Plotting Stock Prices
        st.subheader(f"Stock Prices for {ticker}")
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'])])
        fig.update_layout(title=f'{ticker} Stock Prices', xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

        # Add RSI Calculation
        st.subheader("Relative Strength Index (RSI)")
        data['RSI'] = calculate_rsi(data)
        st.line_chart(data['RSI'])

        # Train and Display Random Forest Model
        st.sidebar.subheader("Random Forest Model")
        st.sidebar.write("Training the model...")
        model_rf, accuracy, confusion_matrix_fig, precision_recall_fig = train_rf_model_with_graphs(data)
        st.sidebar.write(f"Accuracy: {accuracy:.2f}")

        # Random Forest Graphs (Confusion Matrix and Precision-Recall Curve)
        st.sidebar.subheader("Random Forest Performance Graphs")
        with st.sidebar.expander("Confusion Matrix"):
            st.plotly_chart(confusion_matrix_fig, use_container_width=True)
        with st.sidebar.expander("Precision-Recall Curve"):
            st.plotly_chart(precision_recall_fig, use_container_width=True)

        # Train and Display LSTM Model
        st.sidebar.subheader("LSTM Model")
        st.sidebar.write("Training the model...")
        model_lstm, history, loss_curve_fig = train_lstm_model_with_graphs(data)

        # Display Training Loss Curve
        st.sidebar.subheader("LSTM Performance Graphs")
        with st.sidebar.expander("Training Loss Curve"):
            st.plotly_chart(loss_curve_fig, use_container_width=True)

        # Predict next week's prices and behavior
        pred.subheader("Upcoming Week Predictions")
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)
        predictions = []
        behavior = []

        # Use LSTM for prediction (example placeholder, replace with actual LSTM predictions)
        last_closes = data['Close'][-50:].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_closes = scaler.fit_transform(last_closes)

        for i in range(7):
            input_seq = scaled_closes[-50:].reshape(1, 50, 1)  # LSTM expects input in 3D shape
            predicted = model_lstm.predict(input_seq)
            predicted_price = scaler.inverse_transform(predicted)[0][0]
            predictions.append(predicted_price)
            scaled_closes = np.append(scaled_closes, scaler.transform([[predicted_price]]))[-50:]
            behavior.append("Increase" if predicted_price > last_closes[-1] else "Decrease")
            last_closes = np.append(last_closes, [[predicted_price]], axis=0)

        # Create DataFrame for predictions
        predictions_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Price": [f"${pred:.2f}" for pred in predictions],
            "Behavior": behavior
        })
        pred.write(predictions_df)

