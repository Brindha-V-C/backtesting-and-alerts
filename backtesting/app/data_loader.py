import requests
import pandas as pd

ML_BASE_URL = "http://127.0.0.1:8001"
HISTORICAL_ENDPOINT = "/api/v1/ml/signal/historical"


def load_historical_data(ticker: str) -> pd.DataFrame:
    """
    Fetches 5 years OHLCV + ML signals from ML service
    and returns it as a DataFrame.
    """

    payload = {
        "ticker": ticker
    }

    response = requests.post(
        f"{ML_BASE_URL}{HISTORICAL_ENDPOINT}",
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"ML API failed: {response.status_code} - {response.text}"
        )

    data = response.json()

    if "rows" not in data:
        raise ValueError("Invalid response from ML API")

    df = pd.DataFrame(data["rows"])

    # Parse date & set index
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    df = df.sort_index()

    # Rename columns to match engine expectations
    df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "signal": "Signal"
    }, inplace=True)

    return df
