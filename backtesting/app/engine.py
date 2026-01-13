import numpy as np
import pandas as pd
import vectorbt as vbt
import logging

# ---------------- BASIC LOGGER ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)
logger = logging.getLogger("BacktestEngine")


class BacktestEngine:
    INITIAL_CAPITAL = 1_000_000
    FEES = 0.002
    TRADING_DAYS = 252

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        logger.info("Backtest engine started")

    @staticmethod
    def to_py(val):
        if hasattr(val, "item"):
            return val.item()
        return val

    # ---------------- MARKET ----------------
    def run_market(self):
        try:
            logger.info("Running market backtest...")

            returns = self.df["Close"].pct_change().dropna()
            equity = (1 + returns).cumprod() * self.INITIAL_CAPITAL

            n_years = len(returns) / self.TRADING_DAYS
            drawdown = (equity - equity.cummax()) / equity.cummax()

            start_equity = equity.iloc[0]
            end_equity = equity.iloc[-1]

            return {
                "metrics": {
                    "total_return_pct": self.to_py((end_equity / start_equity - 1) * 100),
                    "cagr_pct": self.to_py(((end_equity / start_equity) ** (1 / n_years) - 1) * 100),
                    "volatility_pct": self.to_py(returns.std() * np.sqrt(self.TRADING_DAYS) * 100),
                    "sharpe_ratio": self.to_py((returns.mean() / returns.std()) * np.sqrt(self.TRADING_DAYS)),
                    "max_drawdown_pct": self.to_py(abs(drawdown.min()) * 100),
                },
                "equity": equity
            }

        except Exception as e:
            logger.error(f"Market backtest failed: {e}")
            raise

    # ---------------- ML STRATEGY ----------------
    def run_ml(self):
        try:
            logger.info("Running ML backtest...")

            entries = self.df["Signal"] == 1
            exits = self.df["Signal"] == -1

            pf = vbt.Portfolio.from_signals(
                close=self.df["Close"],
                entries=entries,
                exits=exits,
                init_cash=self.INITIAL_CAPITAL,
                fees=self.FEES,
                freq="1D"
            )

            stats = pf.stats()
            equity = pf.value()
            returns = pf.returns()

            n_years = len(returns.dropna()) / self.TRADING_DAYS
            trades = pf.trades.records_readable.copy()

            winning_df = trades[trades["PnL"] > 0]
            losing_df = trades[trades["PnL"] < 0]

            avg_win = (
                (winning_df["Avg Exit Price"] - winning_df["Avg Entry Price"]).mean()
                if not winning_df.empty else 0.0
            )

            avg_loss = (
                (losing_df["Avg Exit Price"] - losing_df["Avg Entry Price"]).mean()
                if not losing_df.empty else 0.0
            )

            return {
                "ml_metrics": {
                    "total_return_pct": self.to_py(stats["Total Return [%]"]),
                    "cagr_pct": self.to_py(((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) * 100),
                    "volatility_pct": self.to_py(returns.std() * np.sqrt(self.TRADING_DAYS) * 100),
                    "sharpe_ratio": self.to_py(stats["Sharpe Ratio"]),
                    "max_drawdown_pct": self.to_py(abs(stats["Max Drawdown [%]"])),
                    "total_equity_value": self.to_py(equity.iloc[-1]),
                    "total_trades": int(self.to_py(stats["Total Trades"])),
                    "win_rate_pct": self.to_py(stats["Win Rate [%]"]),
                },
                "trading_metrics": {
                    "winning_trades": len(winning_df),
                    "losing_trades": len(losing_df),
                    "profit_factor": self.to_py(stats["Profit Factor"]),
                    "avg_win_price": float(self.to_py(avg_win)),
                    "avg_loss_price": float(self.to_py(avg_loss)),
                },
                "equity": equity,
                "trades": trades
            }

        except Exception as e:
            logger.error(f"ML backtest failed: {e}")
            raise

    # ---------------- CONFIDENCE ----------------
    @staticmethod
    def calculate_confidence(ml_metrics, market_metrics):
        try:
            raw = (
                (ml_metrics["sharpe_ratio"] / market_metrics["sharpe_ratio"]) *
                (ml_metrics["cagr_pct"] / market_metrics["cagr_pct"])
            )
            normalized = min(max((raw / 2) * 100, 0), 100)
            return round(normalized, 2)
        except Exception:
            return 0.0

    # ---------------- GRAPHS ----------------
    def build_graphs(self, market, ml):
        try:
            logger.info("Building graphs...")

            equity_curve = [
                {
                    "date": str(d),
                    "market": market["equity"].iloc[i] / market["equity"].iloc[0],
                    "ml": ml["equity"].iloc[i] / ml["equity"].iloc[0],
                }
                for i, d in enumerate(market["equity"].index)
            ]

            trades_df = ml["trades"]
            pnl_points = []

            if not trades_df.empty:
                for _, row in trades_df.iterrows():
                    pnl_points.append({
                        "entry_date": str(row["Entry Timestamp"]),
                        "exit_date": str(row["Exit Timestamp"]),
                        "pnl": float(row["PnL"]),
                        "is_profit": row["PnL"] > 0
                    })

            trade_marks = {
                "dates": self.df.index.astype(str).tolist(),
                "close": self.df["Close"].tolist(),
                "buy_dates": self.df.index[self.df["Signal"] == 1].astype(str).tolist(),
                "sell_dates": self.df.index[self.df["Signal"] == -1].astype(str).tolist()
            }

            return equity_curve, pnl_points, trade_marks

        except Exception as e:
            logger.error(f"Graph build failed: {e}")
            raise
