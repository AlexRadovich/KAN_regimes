import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = ["SPY", "QQQ", "GLD", "TLT"]

def fetch_data(start="2010-01-01", end="2024-12-31"):
    raw = yf.download(TICKERS, start=start, end=end)["Close"]
    return raw.dropna()

def compute_features(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    rets = prices.pct_change()
    features = pd.DataFrame(index=prices.index)

    for col in prices.columns:
        features[f"{col}_ret"]     = rets[col]
        features[f"{col}_vol"]     = rets[col].rolling(window).std()
        features[f"{col}_mom"]     = prices[col].pct_change(window)
        delta = rets[col]
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss + 1e-9)
        features[f"{col}_rsi"]    = 100 - (100 / (1 + rs))

    return features.dropna()