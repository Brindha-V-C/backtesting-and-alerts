import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="ML Signal Service")

# =================================================
# LOAD MODELS
# =================================================
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

FEATURES = ["Daily_Return", "Volatility", "SMA_ratio", "EMA_ratio", "MACD"]

# =================================================
# FEATURE ENGINEERING
# =================================================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix yfinance multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]

    df["Daily_Return"] = close.pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(14).std()

    df["SMA20"] = close.rolling(20).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()

    df["SMA_ratio"] = close / df["SMA20"]
    df["EMA_ratio"] = close / df["EMA20"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    df.dropna(inplace=True)
    return df

# =================================================
# REQUEST SCHEMA
# =================================================
class TickerRequest(BaseModel):
    ticker: str

# =================================================
# 1️⃣ LIVE SIGNAL API (Dashboard)
# =================================================
@app.post("/api/v1/ml/signal/live")
def get_live_signal(request: TickerRequest):
    """
    Used by Dashboard → Predict Signal button
    Returns only today's signal
    """
    ticker = request.ticker.upper()

    df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
    if df.empty:
        return {"error": "No data found"}

    df = create_features(df)
    X = df[FEATURES].tail(1)

    rf_pred = rf_model.predict(X)[0]
    xgb_pred = xgb_model.predict(X)[0]
    avg_pred = (rf_pred + xgb_pred) / 2

    signal = "BUY" if avg_pred > 0 else "SELL"

    return {
        "ticker": ticker,
        "signal": signal,
        "expected_return": float(avg_pred),
        "current_price": float(df["Close"].iloc[-1])
    }

# =================================================
# 2️⃣ HISTORICAL SIGNALS API (Backtesting)
# =================================================
@app.post("/api/v1/ml/signal/historical")
def get_historical_signals(request: TickerRequest):
    """
    Used by Backtesting Engine
    Returns 5 years OHLCV + ML signals
    """
    ticker = request.ticker.upper()

    df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        return {"error": "No historical data"}

    df = create_features(df)

    X = df[FEATURES]
    rf_preds = rf_model.predict(X)
    xgb_preds = xgb_model.predict(X)
    avg_preds = (rf_preds + xgb_preds) / 2

    df["Signal"] = np.where(avg_preds > 0, 1, -1)


    # Convert to JSON-safe structure
    records = []
    for date, row in df.iterrows():
        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"]),
            "signal": int(row["Signal"])
        })

    return {
        "ticker": ticker,
        "rows": records
    }

# =================================================
# RUN
# =================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
