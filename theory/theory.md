# Theory: Black-Scholes & Machine Learning in Options Trading

This document outlines the theoretical foundations and practical implementation of the models used in this project.

## 1. The Black-Scholes Model

The Black-Scholes-Merton model is a mathematical model for pricing an options contract. It estimates the theoretical value of European-style options based on five key variables.

### 1.1 Inputs
1.  **Spot Price ($S$)**: The current market price of the underlying asset (HDFC Bank).
2.  **Strike Price ($K$)**: The price at which the option can be exercised.
3.  **Time to Expiry ($T$)**: Time remaining until the option contract expires (in years).
4.  **Risk-Free Rate ($r$)**: The theoretical return of an investment with zero risk (e.g., Treasury bond yield).
5.  **Volatility ($\sigma$)**: A measure of how much the stock price fluctuates. This is the only unobservable parameter and must be estimated.

### 1.2 The Formulas
For a **Call Option ($C$)**:
$$ C = S N(d_1) - K e^{-rT} N(d_2) $$

For a **Put Option ($P$)**:
$$ P = K e^{-rT} N(-d_2) - S N(-d_1) $$

Where:
$$ d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}} $$
$$ d_2 = d_1 - \sigma\sqrt{T} $$
* $N(x)$ is the cumulative distribution function of the standard normal distribution.

### 1.3 The Greeks
The "Greeks" measure the sensitivity of the option price to changes in input parameters:
-   **Delta ($\Delta$)**: Sensitivity to stock price ($S$). $\Delta \approx 0.5$ for ATM options.
-   **Gamma ($\Gamma$)**: Sensitivity of Delta to stock price. Highest for ATM options near expiry.
-   **Vega ($\nu$)**: Sensitivity to volatility ($\sigma$). Higher volatility = Higher option prices.
-   **Theta ($\Theta$)**: Sensitivity to time ($T$). Options lose value as time passes (Time Decay).
-   **Rho ($\rho$)**: Sensitivity to interest rates ($r$).

---

## 2. Machine Learning Integration

While Black-Scholes provides a pricing framework, it assumes constant volatility and log-normal returns. We enhance this by using Machine Learning to predict inputs and direction.

### 2.1 Volatility Forecasting (`shortcall.py`)
Instead of using a simple historical average for $\sigma$, we use ML models to predict future volatility.
-   **Features**: Past realized volatility (1-day, 3-day, 7-day, 14-day).
-   **Models**: Linear Regression, Random Forest.
-   **Output**: Predicted Annualized Volatility ($\sigma_{pred}$).
-   **Usage**: Plug $\sigma_{pred}$ into the Black-Scholes formula to get a "fairer" price based on market dynamics.

### 2.2 Directional Prediction (`direction.py`)
We predict whether the stock will close higher or lower the next day.
-   **Features**: Log returns, Rolling Volatility, Momentum.
-   **Model**: Random Forest Classifier.
-   **Output**: Probability of UP move ($P_{up}$).

---

## 3. Trading Strategy (`multiyear.py`)

We combine Black-Scholes pricing with ML signals to execute an **Option Selling Strategy**.

### 3.1 The Logic
Selling options (Short Volatility) benefits from Time Decay (Theta) but carries risk if the market moves sharply against the position. We use ML to bias our trades:

-   **Bullish Signal** ($P_{up} > 0.55$):
    -   **Action**: Sell Put Option (Short Put).
    -   **Rationale**: If stock goes up or stays flat, the Put expires worthless (or loses value), and we keep the premium.
-   **Bearish Signal** ($P_{up} < 0.45$):
    -   **Action**: Sell Call Option (Short Call).
    -   **Rationale**: If stock goes down or stays flat, the Call loses value, and we profit.
-   **Neutral**:
    -   **Action**: Stay in Cash (No Trade).

### 3.2 Backtesting
-   **Walk-Forward Validation**: The model is retrained on a rolling window (e.g., past 200 days) to predict the next day, avoiding look-ahead bias.
-   **Pricing**: Options are priced daily using Black-Scholes with dynamic volatility.
-   **P&L**: Calculated as $Price_{entry} - Price_{exit}$.

### 3.3 Performance Metrics
-   **Sharpe Ratio**: Risk-adjusted return.
-   **Max Drawdown**: Largest peak-to-trough decline.
-   **Win Rate**: Percentage of profitable trades.
