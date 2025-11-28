import yfinance as yf
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Data Fetching
def get_hdfc_spot_and_history(period: str = "1y"):
    ticker = yf.Ticker("HDFCBANK.NS")  # HDFC Bank on NSE
    hist = ticker.history(period=period)
    if hist.empty:
        raise ValueError("No data returned for HDFCBANK.NS – check internet / ticker")
    spot = float(hist["Close"].iloc[-1])
    return spot, hist

S, hdfc_hist = get_hdfc_spot_and_history(period="1y")
print(f"Live-ish HDFC spot from Yahoo (Close) = {S:.2f}")

# 2. Feature Engineering & ML Volatility Prediction
print("\n--- ML Volatility Prediction ---")
df = hdfc_hist.copy()
df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
df["vol_1d"] = df["log_ret"].abs()               # 1-day realized vol
df["vol_3d"] = df["vol_1d"].rolling(3).mean()
df["vol_7d"] = df["vol_1d"].rolling(7).mean()
df["vol_14d"] = df["vol_1d"].rolling(14).mean()

# Tomorrow's realized volatility → prediction target
df["target_vol"] = df["vol_1d"].shift(-1)

# Drop NaN
df.dropna(inplace=True)

features = ["vol_1d", "vol_3d", "vol_7d", "vol_14d"]
X = df[features].values
y = df["target_vol"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lr = lin_reg.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f"Linear Regression MSE: {mse_lr:.6f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest MSE: {mse_rf:.6f}")

# Predict Next Day Volatility
latest_features = df[features].iloc[-1:].values
predicted_tomorrow_vol = rf.predict(latest_features)[0]
print(f"Predicted tomorrow's volatility (RandomForest): {predicted_tomorrow_vol:.6f}")

# Annualized Sigma for Black-Scholes
predicted_sigma_annual = predicted_tomorrow_vol * np.sqrt(252)
print(f"Predicted annualized sigma ≈ {predicted_sigma_annual:.4f}")

# Use this predicted sigma
sigma = predicted_sigma_annual

# 3. Black-Scholes Setup
K = 1000.0
t = 30
T = t/365 
r = 0.03
flag = 'c'  # call

print(f"\n--- Black-Scholes Model (using predicted sigma={sigma:.4f}) ---")

# Math functions
def norm_cdf(x: float) -> float:
     return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
 
def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return call

def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    call = bs_call_price(S, K, T, r, sigma)
    put = call - S + K * math.exp(-r * T)
    return put

# Manual Calculation
call_price_manual = bs_call_price(S, K, T, r, sigma)
put_price_manual = bs_put_price(S, K, T, r, sigma)

print("Manual Black-Scholes:")
print(f"Call price  (C) = {call_price_manual:.4f}")
print(f"Put price   (P) = {put_price_manual:.4f}")

# Library Calculation
call_price_lib = black_scholes(flag, S, K, T, r, sigma)
put_price_lib  = black_scholes('p', S, K, T, r, sigma)

print("\npy_vollib Black-Scholes:")
print(f"Call price  (C) = {call_price_lib:.4f}")
print(f"Put price   (P) = {put_price_lib:.4f}")

# Greeks
call_delta = delta('c', S, K, T, r, sigma)
call_gamma = gamma('c', S, K, T, r, sigma)
call_vega  = vega('c', S, K, T, r, sigma)
call_theta = theta('c', S, K, T, r, sigma)
call_rho   = rho('c', S, K, T, r, sigma)

print("\nCall Greeks (py_vollib):")
print(f"Delta = {call_delta:.4f}")
print(f"Gamma = {call_gamma:.6f}")
print(f"Vega  = {call_vega:.4f}")
print(f"Theta = {call_theta:.4f}")
print(f"Rho   = {call_rho:.4f}")

# 4. P&L Plotting
premium = call_price_lib
S_T = np.linspace(S * 0.6, S * 1.4, 200)
long_call_payoff = np.maximum(S_T - K, 0.0)
short_call_payoff = -long_call_payoff
short_call_profit = premium + short_call_payoff

plt.figure()
plt.axhline(0, linestyle='--')
plt.axvline(K, linestyle='--', label='Strike K')
plt.plot(S_T, short_call_profit, label='Short Call P&L at Expiry')
plt.title(f"Short Call P&L (HDFC, 30d, Sigma={sigma:.2f})")
plt.xlabel("HDFC Price at Expiry (S_T)")
plt.ylabel("Profit / Loss")
plt.legend()
plt.grid(True)
plt.show()

print(f"\nIf you SHORT this call, you receive premium ≈ {premium:.2f} today.")
