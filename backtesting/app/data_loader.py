import requests
import pandas as pd
import logging

# -------------------------------------------------
# LOGGER SETUP
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("DataLoader")

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
ML_BASE_URL = "http://127.0.0.1:8001"
HISTORICAL_ENDPOINT = "/api/v1/ml/signal/historical"


def load_historical_data(ticker: str) -> pd.DataFrame:
    """
    Fetches 5 years OHLCV + ML signals from ML service
    and returns it as a DataFrame.
    """

    logger.info(f"📡 Fetching historical ML data for ticker: {ticker}")

    payload = {"ticker": ticker}

    try:
        response = requests.post(
            f"{ML_BASE_URL}{HISTORICAL_ENDPOINT}",
            json=payload,
            timeout=120
        )
    except Exception as e:
        logger.error(f"❌ Connection to ML API failed: {e}")
        raise

    logger.info(f"📥 ML API Status Code: {response.status_code}")

    if response.status_code != 200:
        logger.error(f"❌ ML API Error: {response.text}")
        raise RuntimeError(
            f"ML API failed: {response.status_code} - {response.text}"
        )

    data = response.json()

    if "rows" not in data:
        logger.error(f"❌ Invalid response format: {data}")
        raise ValueError("Invalid response from ML API")

    logger.info(f"✅ Rows received from ML API: {len(data['rows'])}")

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

    logger.info(
        f"📊 Final DataFrame shape: {df.shape} | Columns: {list(df.columns)}"
    )

    return df
