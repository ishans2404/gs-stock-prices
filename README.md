# Goldman Sachs Stock Price Time Series Analysis

This repository contains the code and analysis for performing time series analysis on Goldman Sachs (GS) stock price data. The analysis includes data visualization, preprocessing, model building using LSTM neural networks, and evaluation metrics.

## Table of Contents

1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Data Collection](#data-collection)
4. [Data Visualization](#data-visualization)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Building](#model-building)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Conclusion](#conclusion)
10. [Future Work](#future-work)

## Introduction

This project aims to predict the future closing prices of Goldman Sachs (GS) stock using a Long Short-Term Memory (LSTM) neural network. LSTM models are particularly suitable for sequence prediction tasks like stock price forecasting due to their ability to capture temporal dependencies in data.

## Dependencies

Ensure you have the following Python libraries installed:
- ta
- visualkeras
- pandas
- numpy
- tensorflow
- yfinance
- plotly
- matplotlib
- seaborn
- scikit-learn

Install these libraries using pip:
```bash
pip install ta visualkeras pandas numpy tensorflow yfinance plotly matplotlib seaborn scikit-learn
```

## Data Collection

The historical data for Goldman Sachs (ticker symbol: GS) is fetched using the Yahoo Finance API through the `yfinance` library. Data spans from January 1, 2000, to January 1, 2024.

## Data Visualization

Visualize the historical stock data including closing prices and trading volumes using `plotly` and `matplotlib`. Visualizations include:
- Line plots of closing prices over time
- Line plots of trading volumes over time
- Waterfall chart showing daily price differences

## Data Preprocessing

Prepare the data by:
- Scaling using `MinMaxScaler` to normalize the closing price data
- Creating sequences of historical data for LSTM input
- Splitting the data into training and testing sets

## Model Building

Build a sequential LSTM model using Keras with TensorFlow backend. The model architecture consists of multiple LSTM layers with dropout regularization to prevent overfitting. The model is compiled using the Adam optimizer and Mean Squared Error (MSE) loss function.

## Evaluation

Evaluate the model performance using metrics such as:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R2) score

Model training progress and validation metrics are visualized using `plotly` and `matplotlib`.

## Results

Visualize and analyze the predicted vs actual closing prices of Goldman Sachs stock using `plotly`. Discuss insights gained from the model predictions and compare them with actual data trends.

## Conclusion

Summarize the findings from the analysis, including model performance, insights into stock price movements, and the effectiveness of LSTM for this forecasting task. Discuss any limitations encountered and potential areas for improvement.

## Future Work

Discuss potential future improvements or extensions to the project, such as:
- Incorporating additional technical indicators or external factors
- Experimenting with different neural network architectures or hyperparameters
- Enhancing model interpretability and robustness

---
