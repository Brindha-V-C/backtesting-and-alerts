import numpy as np
import pandas as pd
import vectorbt as vbt


class BacktestEngine:
    INITIAL_CAPITAL = 1_000_000
    FEES = 0.002
    TRADING_DAYS = 252

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @staticmethod
    def to_py(val):
        if hasattr(val, "item"):   # numpy scalar
            return val.item()
        return val


    # ---------------- MARKET ----------------
    def run_market(self):
        returns = self.df["Close"].pct_change().dropna()

        equity = (1 + returns).cumprod()
        equity = equity * self.INITIAL_CAPITAL

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


    # ---------------- ML STRATEGY ----------------
    def run_ml(self):
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
        trades = pf.trades.records_readable

        n_years = len(returns.dropna()) / self.TRADING_DAYS

        print("Trade columns:", trades.columns.tolist())


        return {
            "metrics": {
                "total_return_pct": self.to_py(stats["Total Return [%]"]),
                "cagr_pct": self.to_py(((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) * 100),
                "volatility_pct": self.to_py(returns.std() * np.sqrt(self.TRADING_DAYS) * 100),
                "sharpe_ratio": self.to_py(stats["Sharpe Ratio"]),
                "max_drawdown_pct": self.to_py(abs(stats["Max Drawdown [%]"])),
                "total_trades": int(self.to_py(stats["Total Trades"])),
                "win_rate_pct": self.to_py(stats["Win Rate [%]"]),
                "profit_factor": self.to_py(stats["Profit Factor"]),
            },
            "equity": equity,
            "trades": trades
        }

    # ---------------- GRAPH DATA ----------------
    def build_graphs(self, market, ml):
        # ---------------- Equity Curve ----------------
        equity_curve = [
            {
                "date": str(d),
                "market": market["equity"].iloc[i] / market["equity"].iloc[0],
                "ml": ml["equity"].iloc[i] / ml["equity"].iloc[0],
            }
            for i, d in enumerate(market["equity"].index)
        ]

        trades_df = ml["trades"]

        # ---------------- Trade PnL Scatter ----------------
        pnl_points = []
        if not trades_df.empty:
            for _, row in trades_df.iterrows():
                pnl_points.append({
                    "entry_date": str(row["Entry Timestamp"]),
                    "exit_date": str(row["Exit Timestamp"]),
                    "entry_price": float(row["Avg Entry Price"]),
                    "exit_price": float(row["Avg Exit Price"]),
                    "pnl": float(row["PnL"]),
                    "return_pct": float(row["Return"] * 100),
                    "direction": row["Direction"],
                    "is_profit": row["PnL"] > 0
                })

        # ---------------- Trade Visualization ----------------
        trade_marks = {
            "dates": self.df.index.astype(str).tolist(),
            "close": self.df["Close"].tolist(),
            "buy_dates": self.df.index[self.df["Signal"] == 1].astype(str).tolist(),
            "sell_dates": self.df.index[self.df["Signal"] == -1].astype(str).tolist()
        }

        return equity_curve, pnl_points, trade_marks
