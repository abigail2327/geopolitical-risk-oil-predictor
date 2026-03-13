import numpy as np
import pandas as pd


def calculate_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 5-day rolling volatility using log returns.
    """
    df = df.copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df[df["Close"] > 0]  # log(0) and log(negative) are undefined
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility"] = df["log_returns"].rolling(window=5).std()
    df["volatility"] = df["volatility"].clip(upper=df["volatility"].quantile(0.99))
    return df