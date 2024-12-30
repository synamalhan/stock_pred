import streamlit as st
import yfinance as yf
import pandas as pd
from utils import track_portfolio, calculate_rsi, train_rf_model_with_graphs, train_lstm_model_with_graphs, sharpe_ratio, sortino_ratio
import plotly.graph_objects as go
import numpy as np

def portfolio_pred_page():
    st.header("Portfolio Prediction")

    # Portfolio Tickers Input
    portfolio_tickers = st.text_input("Enter stock tickers separated by commas (e.g., AAPL, GOOGL, MSFT):", value="AAPL,GOOGL,MSFT")
    tickers_list = [ticker.strip() for ticker in portfolio_tickers.split(",")]

    if tickers_list:
        # Fetch portfolio data
        portfolio_df = track_portfolio(tickers_list)

        # # Debug: Display portfolio structure
        # st.write("Portfolio DataFrame Structure")
        # st.write(portfolio_df.head())

        # Check if the columns match the tickers and handle missing 'Close' column
        if not all(ticker in portfolio_df.columns for ticker in tickers_list):
            st.error("Some of the specified tickers are missing in the portfolio data.")
            return

        # Add 'Close' column for compatibility
        portfolio_df['Close'] = portfolio_df[tickers_list[0]]  # Assuming first ticker's prices represent 'Close'

        # Portfolio Performance (Line chart for each ticker)
        st.subheader("Portfolio Performance")
        st.line_chart(portfolio_df)

        # Risk/Return Analysis
        st.subheader("Risk/Return Analysis")
        returns = portfolio_df.pct_change().dropna()
        for ticker in tickers_list:
            sharpe = sharpe_ratio(returns[ticker])
            sortino = sortino_ratio(returns[ticker])
            st.write(f"{ticker} - Sharpe Ratio: {sharpe:.2f}, Sortino Ratio: {sortino:.2f}")

        # Train and Display Random Forest Model
        st.sidebar.divider()

        model_detail = st.sidebar.container(border=True)
        model_detail.subheader("Random Forest Model")
        model_detail.caption("Training the model...")
        model_rf, accuracy, confusion_matrix_fig, precision_recall_fig = train_rf_model_with_graphs(portfolio_df)
        model_detail.write(f"Accuracy: {accuracy:.2f}")

        # Random Forest Graphs (Confusion Matrix and Precision-Recall Curve)
        model_detail.subheader("Random Forest Performance Graphs")
        with model_detail.popover("Confusion Matrix"):
            st.plotly_chart(confusion_matrix_fig, use_container_width=True)
        with model_detail.popover("Precision-Recall Curve"):
            st.plotly_chart(precision_recall_fig, use_container_width=True)

        # Train and Display LSTM Model
        model_detail.subheader("LSTM Model")
        model_detail.caption("Training the model...")
        model_lstm, history, loss_curve_fig = train_lstm_model_with_graphs(portfolio_df)

        # Display Training Loss Curve
        model_detail.subheader("LSTM Performance Graphs")
        with model_detail.popover("Training Loss Curve"):
            st.plotly_chart(loss_curve_fig, use_container_width=True)
