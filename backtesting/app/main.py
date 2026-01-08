from fastapi import FastAPI
from pydantic import BaseModel
from app.data_loader import load_historical_data
from app.engine import BacktestEngine

app = FastAPI(title="Backtesting Service")

# -------------------------------------------------
# REQUEST SCHEMA
# -------------------------------------------------
class BacktestRequest(BaseModel):
    ticker: str


# -------------------------------------------------
# RUN BACKTEST
# -------------------------------------------------
@app.post("/api/v1/backtest/run")
def run_backtest(request: BacktestRequest):
    """
    Triggered by Dashboard → Run Backtest button
    """

    ticker = request.ticker.upper()

    # 1️⃣ Fetch historical data + ML signals
    df = load_historical_data(ticker)

    # 2️⃣ Run backtesting engine
    engine = BacktestEngine(df)

    market = engine.run_market()
    ml = engine.run_ml()

    equity_curve, pnl_graph, trade_visual = engine.build_graphs(
        market, ml
    )

    # 3️⃣ API Response
    return {
        "ml_metrics": ml["ml_metrics"],
        "market_metrics": market["metrics"],
        "trading_metrics": ml["trading_metrics"],
        "equity_curve": equity_curve,
        "pnl_graph": pnl_graph,
        "trade_visualization": trade_visual
    }


