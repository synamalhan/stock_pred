import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import plotly.graph_objects as go
import numpy as np

# Streamlit app title
st.title("Enhanced Stock Prediction App with Random Forest")
st.sidebar.title("About the Model")
# with st.expander(" Moving Averages"):
#     st.write("**SMA_10** and **SMA_50** are the 10-day and 50-day simple moving averages, which smooth stock price trends.")

# Input for stock ticker
ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", value="AAPL")

# Fetch historical data and stock details
@st.cache_data
def fetch_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period="5y")[['Close']], stock.history(period="1d")['Close'].iloc[-1], stock.info

if ticker:
    data, current_price, stock_info = fetch_data(ticker)

    # Display stock details
    st.subheader(f"Details for {ticker}")
    st.write(f"**Name**: {stock_info.get('longName', 'Unknown')}")
    st.write(f"**Current Price**: ${current_price:.2f}")

    # Feature Engineering
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['Return'] = data['Close'].pct_change()  # Daily returns
    data['Lag_1'] = data['Close'].shift(1)  # Lagged features
    data['Lag_2'] = data['Close'].shift(2)
    data['Lag_3'] = data['Close'].shift(3)

    # Target Variable (1: Price Up, 0: Price Down)
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    # Drop NaN values
    data = data.dropna()

    # Input Features and Target Variable
    features = ['SMA_10', 'SMA_50', 'Return', 'Lag_1', 'Lag_2', 'Lag_3']
    X = data[features]
    y = data['Target']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    cm = confusion_matrix(y_test, y_pred)

    # Sidebar: Display model performance
    with st.sidebar.expander("Model Performance Metrics", expanded=True):
        # st.subheader("Accuracy")
        with st.popover("Accuracy"):
            st.write("**Accuracy** measures the percentage of correctly predicted outcomes compared to the total predictions.")

        st.write(f"Accuracy: **{accuracy:.2f}**")
        # st.subheader("AUC-ROC")
        with st.popover("AUC-ROC"):
            st.write("**AUC-ROC** measures a model's ability to distinguish between classes.")

        st.write(f"AUC-ROC: **{roc_auc:.2f}**")

    # Confusion Matrix
    # st.sidebar.subheader("Confusion Matrix")
    with st.sidebar.popover("Confusion Matrix"):
        st.write("The **Confusion Matrix** shows the breakdown of actual vs. predicted values, helping visualize true positives, false positives, etc.")

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: Down', 'Predicted: Up'],
        y=['Actual: Down', 'Actual: Up'],
        colorscale='viridis',
        showscale=True))
    fig_cm.update_layout(title_text="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    st.sidebar.plotly_chart(fig_cm)

    # ROC Curve
    # st.sidebar.subheader("")
    with st.sidebar.popover("ROC Curve"):
        st.write("The **ROC Curve** plots the True Positive Rate vs. False Positive Rate at various thresholds, evaluating model performance.")

    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name="ROC Curve", line=dict(color='royalblue')))  # Custom color
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name="Random Model"))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.sidebar.plotly_chart(fig_roc)

    # Predictions for Next 5 Days
    st.subheader("Predictions for the Next 5 Days")
    predictions = []
    last_row = X.iloc[-1].values.reshape(1, -1)

    for _ in range(5):  # Predict for the next 5 days
        prediction = model.predict(last_row)[0]
        confidence = model.predict_proba(last_row)[0][prediction]
        predictions.append((prediction, confidence))

        # Simulate next day's input features
        last_close = last_row[0][3]  # 'Lag_1' is at index 3
        if prediction == 1:  # Price goes up
            last_close *= 1 + confidence * 0.02  # Adjust by confidence
        else:  # Price goes down
            last_close *= 1 - confidence * 0.02  # Adjust by confidence

        # Update lagged features
        new_row = np.array([
            last_close,  # Update 'Lag_1'
            last_row[0][3],  # Update 'Lag_2' from previous 'Lag_1'
            last_row[0][4],  # Update 'Lag_3' from previous 'Lag_2'
        ] + [0, 0, 0])  # SMA_10, SMA_50, Return are 0 (can enhance later)
        last_row = new_row.reshape(1, -1)

    # Prepare Dates for Next Predictions
    prediction_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=5, freq='B')
    prediction_df = pd.DataFrame({
        'Date': prediction_dates,
        'Prediction': [("Up" if pred[0] == 1 else "Down") for pred in predictions],
        'Confidence (%)': [f"{pred[1] * 100:.2f}" for pred in predictions]
    })
    st.table(prediction_df)

    # Visualization of Stock Prices and Moving Averages
    st.subheader(f"{ticker} Stock Prices and Moving Averages")
    fig_prices = go.Figure()
    fig_prices.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name="Closing Price", line=dict(color='green')))  
    fig_prices.add_trace(go.Scatter(x=data.index, y=data['SMA_10'], mode='lines', name="SMA_10", line=dict(color='orange')))  
    fig_prices.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name="SMA_50", line=dict(color='red')))  
    fig_prices.update_layout(title="Stock Prices and Moving Averages", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_prices)

    