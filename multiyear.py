import yfinance as yf
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# 1. Black-Scholes Functions
# ---------------------------------------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes price of a European call option.
    S: spot price
    K: strike
    T: time to expiry in years
    r: risk-free rate (annual)
    sigma: annual volatility
    """
    if T <= 0:
        # At expiry, it's just intrinsic value
        return max(S - K, 0.0)

    # Avoid division by zero if sigma is too small
    if sigma < 1e-6:
        return max(S - K, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return call


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes price of a European put option (via put-call parity).
    """
    call = bs_call_price(S, K, T, r, sigma)
    put = call - S + K * math.exp(-r * T)
    return put


# ---------------------------------------------------------
# 2. Data Fetching & Feature Engineering
# ---------------------------------------------------------

def fetch_data(ticker="HDFCBANK.NS", index_ticker="^NSEI", period="5y"):
    """
    Fetch HDFC Bank and NSE index data and align by date.
    """
    print(f"Fetching data for {ticker} and {index_ticker} over {period}...")
    df = yf.Ticker(ticker).history(period=period)
    df_idx = yf.Ticker(index_ticker).history(period=period)

    # Align dates by joining index close
    df = df.join(df_idx["Close"].rename("Index_Close"), how="inner")

    return df


def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def calculate_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    bandwidth = (upper - lower) / sma
    return bandwidth


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical + volatility features and a next-day direction target.
    """
    df = df.copy()

    # 1. Returns & Direction
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["target"] = (df["log_ret"].shift(-1) > 0).astype(int)  # 1 if Next Day Up

    # 2. RSI
    df["rsi"] = calculate_rsi(df["Close"])

    # 3. MACD
    df["macd"], df["macd_signal"] = calculate_macd(df["Close"])
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 4. Bollinger Band Width
    df["bb_width"] = calculate_bollinger_bands(df["Close"])

    # 5. Volume % Change
    df["vol_pct_chg"] = df["Volume"].pct_change()

    # 6. Market Index Moves
    df["index_ret"] = np.log(df["Index_Close"] / df["Index_Close"].shift(1))

    # 7. Overnight Gaps (Open - Prev Close)
    df["gap"] = df["Open"] - df["Close"].shift(1)

    # 8. Volatility Clusters (Rolling Std Dev) – annualized for BS
    df["volatility_20d"] = df["log_ret"].rolling(20).std() * np.sqrt(252)
    df["volatility_5d"] = df["log_ret"].rolling(5).std() * np.sqrt(252)

    # 9. Moving Averages
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["sma_200"] = df["Close"].rolling(200).mean()
    df["dist_sma_50"] = (df["Close"] - df["sma_50"]) / df["sma_50"]

    # Clean up NaNs/Infs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


# ---------------------------------------------------------
# 3. Backtesting Engine (Walk-Forward with Options)
# ---------------------------------------------------------

def backtest_strategy(df: pd.DataFrame, start_idx=200, train_window=200) -> pd.DataFrame:
    """
    Walk-forward backtest simulating 1-day holding of 30DTE option sells.

    Strategy:
      - Train RF classifier on past `train_window` days to predict next-day direction.
      - At day t, use RF to get Prob(Up | features_t).
        - If Prob(Up) > 0.55: sell ATM put (bullish).
        - If Prob(Up) < 0.45: sell ATM call (bearish).
        - Else: hold no position.
      - Reprice the option at day t+1 with updated S and sigma and close the position.
    """

    features = [
        "rsi", "macd", "macd_hist", "bb_width", "vol_pct_chg",
        "index_ret", "gap", "volatility_5d", "volatility_20d",
        "dist_sma_50"
    ]

    dates = []
    pnl_curve = [0.0]  # cumulative P&L (currency units)
    daily_pnl = []
    positions = []

    print(f"Starting backtest from index {start_idx} to {len(df)-1}...")

    total_steps = len(df) - 1 - start_idx

    # Risk-free rate (annual)
    r = 0.05

    for i in range(start_idx, len(df) - 1):
        # 1. Train Window
        train_data = df.iloc[i - train_window: i]
        X_train = train_data[features].values
        y_train = train_data["target"].values

        # 2. Train Model
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 3. Predict for current day 'i'
        current_features = df.iloc[i:i + 1][features].values
        prob_up = model.predict_proba(current_features)[0, 1]

        # 4. Trading Logic (Option Selling)
        threshold_up = 0.55
        threshold_down = 0.45

        S_t = df.iloc[i]["Close"]
        sigma_t = df.iloc[i]["volatility_20d"]
        if sigma_t == 0 or np.isnan(sigma_t):
            sigma_t = 0.20  # fallback

        K = S_t              # ATM
        T_entry = 30 / 365.0 # 30 DTE at entry

        position_type = "NONE"
        entry_price = 0.0

        if prob_up > threshold_up:
            # Bullish view -> sell ATM put
            position_type = "SHORT_PUT"
            entry_price = bs_put_price(S_t, K, T_entry, r, sigma_t)

        elif prob_up < threshold_down:
            # Bearish view -> sell ATM call
            position_type = "SHORT_CALL"
            entry_price = bs_call_price(S_t, K, T_entry, r, sigma_t)

        # 5. Calculate P&L at t+1 (close/mark-to-market)
        S_next = df.iloc[i + 1]["Close"]
        sigma_next = df.iloc[i + 1]["volatility_20d"]
        if sigma_next == 0 or np.isnan(sigma_next):
            sigma_next = sigma_t

        T_exit = 29 / 365.0  # 1 day later, 29 days to expiry

        exit_price = 0.0
        pnl = 0.0

        if position_type == "SHORT_PUT":
            exit_price = bs_put_price(S_next, K, T_exit, r, sigma_next)
            # For a short option, profit if option value falls
            pnl = entry_price - exit_price

        elif position_type == "SHORT_CALL":
            exit_price = bs_call_price(S_next, K, T_exit, r, sigma_next)
            # For a short option, profit if option value falls
            pnl = entry_price - exit_price

        # Update P&L (per 1 contract / share equivalent)
        pnl_curve.append(pnl_curve[-1] + pnl)
        daily_pnl.append(pnl)
        dates.append(df.index[i + 1])
        positions.append(position_type)

        day_num = i - start_idx + 1
        if day_num % 50 == 0:
            print(f"Processed Day {day_num}/{total_steps}")

    return pd.DataFrame({
        "Date": dates,
        "Daily_PnL": daily_pnl,
        "Cum_PnL": pnl_curve[1:],  # drop initial 0
        "Position": positions,
    })


# ---------------------------------------------------------
# 4. Metrics & Evaluation
# ---------------------------------------------------------

def evaluate_metrics(results: pd.DataFrame):
    print("\n--- Performance Metrics (Option Strategy) ---")

    # 1. Sharpe-like Ratio (Annualized) on raw PnL
    mean_pnl = np.mean(results["Daily_PnL"])
    std_pnl = np.std(results["Daily_PnL"])
    sharpe = (mean_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0.0

    # 2. Max Drawdown in absolute currency terms
    cum_pnl = np.array(results["Cum_PnL"])
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    max_dd = np.min(drawdown)

    # 3. Win Rate on active trades
    active_trades = results[results["Position"] != "NONE"]
    wins = active_trades[active_trades["Daily_PnL"] > 0]
    win_rate = len(wins) / len(active_trades) if len(active_trades) > 0 else 0.0

    print(f"Sharpe Ratio (PnL based): {sharpe:.4f}")
    print(f"Max Drawdown (Abs):       {max_dd:.2f}")
    print(f"Win Rate:                 {win_rate:.2%}")
    print(f"Total Trades:             {len(active_trades)}")
    print(f"Final Cumulative P&L:     {results['Cum_PnL'].iloc[-1]:.2f}")


# ---------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------

if __name__ == "__main__":
    # 1. Fetch
    df = fetch_data(period="5y")
    print(f"Raw data shape: {df.shape}")

    # 2. Features
    df = add_features(df)
    print(f"Feature data shape: {df.shape}")

    # 3. Backtest
    results = backtest_strategy(df, start_idx=200, train_window=200)

    # 4. Metrics
    evaluate_metrics(results)
