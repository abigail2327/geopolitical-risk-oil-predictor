import os
import yfinance as yf
import pandas as pd


def download_oil_prices(start_date: str = "2020-01-01") -> pd.DataFrame:
    """
    Download WTI crude oil futures (CL=F) from Yahoo Finance.

    Returns:
        DataFrame with flat OHLCV columns and a DatetimeIndex named 'Date'.
    """
    print("Downloading oil price data...")
    oil_data = yf.download("CL=F", start=start_date, auto_adjust=True)

    # yfinance >=0.2.18 returns MultiIndex columns like ("Close", "CL=F").
    # Flatten to plain single-level names.
    if isinstance(oil_data.columns, pd.MultiIndex):
        oil_data.columns = oil_data.columns.get_level_values(0)

    oil_data.index.name = "Date"
    return oil_data


def save_data(df: pd.DataFrame, filename: str) -> None:
    """
    Save a DataFrame to CSV inside the local data/ directory.
    The index (Date) is written as a regular column named 'Date'.
    """
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", filename)
    df.reset_index().to_csv(path, index=False)
    print(f"Saved → {path}")