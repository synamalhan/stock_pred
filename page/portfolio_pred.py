import streamlit as st
import yfinance as yf
from utils import track_portfolio, sharpe_ratio, sortino_ratio

def portfolio_page():
    st.header("Portfolio Analysis")

    # Portfolio Tickers Input
    portfolio_tickers = st.text_input("Enter stock tickers separated by commas (e.g., AAPL, GOOGL, MSFT):", value="AAPL,GOOGL,MSFT")
    tickers_list = [ticker.strip() for ticker in portfolio_tickers.split(",")]

    if tickers_list:
        # Fetch portfolio data
        portfolio_df = track_portfolio(tickers_list)

        # Plot Portfolio Performance
        st.subheader("Portfolio Performance")
        st.line_chart(portfolio_df)

        # Risk/Return Analysis
        st.subheader("Risk/Return Analysis")
        returns = portfolio_df.pct_change().dropna()
        for ticker in tickers_list:
            sharpe = sharpe_ratio(returns[ticker])
            sortino = sortino_ratio(returns[ticker])
            st.write(f"{ticker} - Sharpe Ratio: {sharpe:.2f}, Sortino Ratio: {sortino:.2f}")
