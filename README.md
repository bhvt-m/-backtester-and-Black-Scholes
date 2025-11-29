# HDFC Bank Option Pricing & Trading Analysis

This project implements a comprehensive framework for analyzing and trading HDFC Bank options using the Black-Scholes model and Machine Learning.

## Project Structure

-   **`blackscholes.py`**: The core implementation of the Black-Scholes pricing model and Greeks calculation.
-   **`train.py`**: Integrates ML (Linear Regression, Random Forest) to predict future volatility and uses it to price options more accurately.
-   **`direction.py`**: A standalone script to predict the daily directional movement (Up/Down) of the stock using Random Forest.
-   **`multiyear.py`**: A robust 5-year backtesting engine that simulates an option selling strategy based on ML signals and Black-Scholes pricing.
-   **`theory/`**: Contains detailed theoretical explanations (`theory.md`).
-   **`images/`**: Stores generated P&L charts and analysis plots.

## Key Features

1.  **Black-Scholes Model**:
    -   Calculates Call/Put prices.
    -   Computes Greeks (Delta, Gamma, Vega, Theta, Rho).
    -   Simulates P&L for short positions.

2.  **Machine Learning**:
    -   **Volatility Forecasting**: Predicts realized volatility to improve pricing accuracy.
    -   **Directional Classification**: Predicts market direction (Bullish/Bearish) with ~56% win rate (in backtests).

3.  **Backtesting**:
    -   Walk-forward validation over 5 years of data.
    -   Simulates Short Call / Short Put strategies.
    -   Metrics: Sharpe Ratio, Max Drawdown, Win Rate, Profit Distribution.

## Setup & Usage

1.  **Install Dependencies**:
    ```bash
    pip install yfinance pandas numpy scikit-learn matplotlib py_vollib
    ```

2.  **Run Scripts**:
    -   **Basic Pricing**: `python blackscholes.py`
    -   **ML Volatility**: `python shortcall.py`
    -   **Direction Prediction**: `python direction.py`
    -   **Full Backtest**: `python multiyear.py`

## Results Summary

-   The **Directional Model** achieved a win rate of approximately **56%** in the 5-year backtest.
-   The **Option Selling Strategy** (unhedged) showed a negative total P&L due to large drawdowns during volatile periods, highlighting the importance of risk management (hedging) in real-world application.
-   **Volatility Prediction** using Random Forest provided a more dynamic input for the pricing model compared to static historical averages.
