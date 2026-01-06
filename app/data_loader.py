import pandas as pd

def load_historical_data() -> pd.DataFrame:
    df = pd.read_csv("ml_trading_signals.csv")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df.set_index("Date", inplace=True)

    df = df.sort_index().dropna()

    # Expected columns:
    # Open, High, Low, Close, Volume, Signal (1 buy, -1 sell, 0 hold)

    return df
