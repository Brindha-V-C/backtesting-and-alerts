from fastapi import FastAPI
from app.data_loader import load_historical_data
from app.engine import BacktestEngine
from app.schemas import BacktestResponse

app = FastAPI(title="Backtesting Service")

@app.post("/api/v1/backtest/run")
def run_backtest():
    df = load_historical_data()

    engine = BacktestEngine(df)

    market = engine.run_market()
    ml = engine.run_ml()

    equity_curve, pnl_graph, trade_visual = engine.build_graphs(market, ml)

    return {
        "ml_metrics": ml["ml_metrics"],
        "market_metrics": market["metrics"],
        "trading_metrics": ml["trading_metrics"],
        "equity_curve": equity_curve,
        "pnl_graph": pnl_graph,
        "trade_visualization": trade_visual
    }
