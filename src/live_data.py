import yfinance as yf
import pandas as pd


def get_live_oil() -> pd.DataFrame:
    ticker = yf.Ticker("CL=F")
    data = ticker.history(period="5d", interval="1h")
    data.reset_index(inplace=True)
    return data