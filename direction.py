
import yfinance as yf
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# -----------------------------
# 1. Data Fetching
# -----------------------------

def get_hdfc_history(period: str = "1y") -> pd.DataFrame:
    """
    Fetch historical OHLC data for HDFC Bank from Yahoo Finance.
    """
    ticker = yf.Ticker("HDFCBANK.NS")
    hist = ticker.history(period=period)

    if hist.empty:
        raise ValueError("No data returned for HDFCBANK.NS – check internet / ticker")

    return hist


# -----------------------------
# 2. Feature Engineering
# -----------------------------

def build_features_for_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features and target for UP/DOWN prediction.
    - Features:
        log_ret    : today's log return
        vol_3d     : 3-day avg abs return
        vol_7d     : 7-day avg abs return
        vol_14d    : 14-day avg abs return
    - Target:
        target_up  : 1 if tomorrow's return > 0, else 0
    """
    df = df.copy()

    # Log returns
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))

    # Realized volatility windows
    df["vol_1d"] = df["log_ret"].abs()
    df["vol_3d"] = df["vol_1d"].rolling(3).mean()
    df["vol_7d"] = df["vol_1d"].rolling(7).mean()
    df["vol_14d"] = df["vol_1d"].rolling(14).mean()

    # Future return (tomorrow)
    df["future_return"] = df["log_ret"].shift(-1)

    # Binary direction label: 1 = up, 0 = down/flat
    df["target_up"] = (df["future_return"] > 0).astype(int)

    # Drop rows with NaN from rolling / shifting
    df.dropna(inplace=True)

    return df


# -----------------------------
# 3. Train/Test Split & Model
# -----------------------------

def train_direction_model(df_features: pd.DataFrame):
    """
    Train a RandomForestClassifier on the features and target_up.
    Returns:
        model, X_train, X_test, y_train, y_test
    """
    feature_cols = ["log_ret", "vol_3d", "vol_7d", "vol_14d"]

    X = df_features[feature_cols].values
    y = df_features["target_up"].values

    # Time-ordered split (no shuffle to avoid look-ahead bias)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, feature_cols


# -----------------------------
# 4. Evaluation & Latest Signal
# -----------------------------

def evaluate_and_signal(model, X_test, y_test, df_features, feature_cols):
    """
    Print accuracy + basic report and compute the latest
    probability that HDFC goes UP tomorrow.
    """
    # Test accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy (direction UP/DOWN): {acc:.3f}")

    print("\nClassification report (on test data):")
    print(classification_report(y_test, y_pred, digits=3))

    # Latest row (most recent date with valid features)
    latest_row = df_features.iloc[-1:]
    latest_features = latest_row[feature_cols].values

    prob_up = model.predict_proba(latest_features)[0, 1]
    prob_down = 1.0 - prob_up

    print("--------------------------------------------------")
    print(f"Model probability HDFC goes UP tomorrow   : {prob_up:.3f}")
    print(f"Model probability HDFC goes DOWN tomorrow : {prob_down:.3f}")

    # Simple interpretation for short call strategy
    if prob_up < 0.4:
        print("\nInterpretation: Model expects SIDEWAYS/DOWN more than UP.")
        print("For a short CALL position, this is a relatively safer environment.")
    elif prob_up > 0.6:
        print("\nInterpretation: Model expects a strong chance of UP move.")
        print("Short CALL becomes risky (stock rally could hurt you).")
    else:
        print("\nInterpretation: Model is quite neutral / uncertain.")
        print("Short CALL is a mixed bet – neither clearly safe nor clearly dangerous.")


# -----------------------------
# 5. Main
# -----------------------------

if __name__ == "__main__":
    print("Fetching HDFC Bank data...")
    hist = get_hdfc_history(period="1y")
    print(f"Data loaded: {len(hist)} rows")

    print("\nBuilding features for direction prediction...")
    df_feat = build_features_for_direction(hist)
    print(f"Feature dataset size after dropping NaNs: {len(df_feat)} rows")

    print("\nTraining RandomForest direction model...")
    model, X_train, X_test, y_train, y_test, feature_cols = train_direction_model(df_feat)

    evaluate_and_signal(model, X_test, y_test, df_feat, feature_cols)
