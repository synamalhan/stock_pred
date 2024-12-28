# Enhanced Stock Prediction App with Random Forest

## Overview

This Streamlit app uses a Random Forest Classifier to predict stock prices for the next 5 days based on historical stock data. The app fetches stock data from Yahoo Finance (```yfinance```) and uses various machine learning techniques for predictions, including feature engineering and model evaluation with performance metrics.

## Features

- **Stock Ticker Input**: Enter a stock ticker (e.g., AAPL) to fetch historical data.
- **Predictions**: Displays predictions for the next 5 days with confidence.
- **Model Metrics**: Shows performance metrics like Accuracy, AUC-ROC, Confusion Matrix, and ROC Curve.
- **Visualizations**: Interactive visualizations for stock prices, moving averages, and model performance.

## Requirements

The project requires the following Python libraries:

- ```streamlit```
- ```yfinance```
- ```pandas```
- ```scikit-learn```
- ```plotly```
- ```numpy```

You can install the dependencies using the following command:

```
pip install -r requirements.txt
```

## How to Use

1. Clone the repository to your local machine:
   ```
   git clone https://github.com/synamalhan/stock_pred.git
   ```

2. Navigate to the project directory:
   ```
   cd enhanced-stock-prediction
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

5. Enter a stock ticker (e.g., AAPL) in the input box to see the stock prediction results.

## Model Evaluation Metrics

### Accuracy
The **Accuracy** metric measures the percentage of correctly predicted outcomes compared to the total predictions.

### AUC-ROC
The **AUC-ROC** metric evaluates the model’s ability to distinguish between classes. A higher AUC value indicates better performance.

### Confusion Matrix
The **Confusion Matrix** provides insights into the true positives, false positives, true negatives, and false negatives for the model's predictions.

### ROC Curve
The **ROC Curve** plots the True Positive Rate vs. False Positive Rate at various thresholds, helping visualize the model's performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
