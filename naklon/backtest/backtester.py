"""
Backtesting Engine

Бэктестинг стратегии торговли по наклонкам:
- Симуляция торговли на исторических данных
- Расчёт метрик производительности
- Генерация отчётов
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from naklon.strategy.engine import StrategyEngine
from naklon.strategy.risk import PositionStatus

logger = logging.getLogger("naklon.backtest")


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    total_bars: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    total_pnl: float
    total_pnl_pct: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown_pct: float
    initial_capital: float
    final_equity: float
    equity_curve: list[float]
    trades: list[dict]

    def summary(self) -> str:
        """Return formatted summary string."""
        return (
            f"\n{'='*60}\n"
            f"  BACKTEST RESULTS — {self.symbol} ({self.timeframe})\n"
            f"  Стратегия: Наклонки (Trendline Breakout)\n"
            f"{'='*60}\n"
            f"  Period:           {self.start_date} → {self.end_date}\n"
            f"  Total bars:       {self.total_bars}\n"
            f"{'─'*60}\n"
            f"  Initial Capital:  ${self.initial_capital:,.2f}\n"
            f"  Final Equity:     ${self.final_equity:,.2f}\n"
            f"  Total P&L:        ${self.total_pnl:,.2f} ({self.total_pnl_pct:+.2f}%)\n"
            f"{'─'*60}\n"
            f"  Total Trades:     {self.total_trades}\n"
            f"  Wins / Losses:    {self.winning_trades} / {self.losing_trades}\n"
            f"  Win Rate:         {self.win_rate_pct:.1f}%\n"
            f"  Avg Win:          ${self.avg_win:,.2f}\n"
            f"  Avg Loss:         ${self.avg_loss:,.2f}\n"
            f"  Profit Factor:    {self.profit_factor:.2f}\n"
            f"  Max Drawdown:     {self.max_drawdown_pct:.2f}%\n"
            f"{'='*60}\n"
        )


class Backtester:
    """Backtests the Naklon trendline trading strategy.

    Процесс:
    1. Загрузка исторических данных
    2. Итерация по каждой свече
    3. Генерация сигналов и симуляция сделок
    4. Расчёт метрик и equity curve
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str | None = None,
    ) -> BacktestResult:
        """Run backtest on historical data.

        Args:
            df: OHLCV DataFrame.
            symbol: Trading pair symbol.
            timeframe: Candle timeframe.

        Returns:
            BacktestResult with all metrics and trade details.
        """
        if timeframe is None:
            timeframe = self.config.get("timeframes", {}).get("primary", "15m")

        engine = StrategyEngine(self.config)

        # Determine warmup period
        lookback = self.config.get("trendline", {}).get("lookback_bars", 120)
        warmup = lookback + 50  # Extra bars for indicator warmup

        if len(df) < warmup + 10:
            logger.warning(
                "Insufficient data for backtest: %d bars (need at least %d)",
                len(df), warmup + 10,
            )
            return self._empty_result(symbol, timeframe, df)

        logger.info(
            "Starting backtest: %s (%s) | %d bars | Capital: $%s",
            symbol, timeframe, len(df), self.config["capital"]["initial"],
        )

        equity_curve = [engine.risk_manager.equity]
        current_day = None

        # Iterate through each bar after warmup
        for i in range(warmup, len(df)):
            bar_data = df.iloc[: i + 1]

            # Get timestamp
            if "timestamp" in df.columns:
                ts = df["timestamp"].iloc[i]
                if isinstance(ts, str):
                    current_time = datetime.fromisoformat(ts)
                else:
                    current_time = ts.to_pydatetime()
            else:
                current_time = datetime(2025, 1, 1)

            # Reset daily P&L at day boundary
            if current_day is not None and current_time.date() != current_day:
                engine.risk_manager.reset_daily()
            current_day = current_time.date()

            # Run strategy on this bar
            engine.on_new_candle(symbol, bar_data, current_time)

            # Record equity
            current_price = df["close"].iloc[i]
            equity = engine.risk_manager.equity
            # Add unrealized P&L
            for pos in engine.risk_manager.positions:
                if pos.status != PositionStatus.CLOSED and pos.symbol == symbol:
                    equity += pos.unrealized_pnl(current_price)
            equity_curve.append(equity)

        # Close any remaining positions at last price
        last_price = df["close"].iloc[-1]
        last_time = datetime.now()
        if "timestamp" in df.columns:
            ts = df["timestamp"].iloc[-1]
            if isinstance(ts, str):
                last_time = datetime.fromisoformat(ts)
            elif hasattr(ts, "to_pydatetime"):
                last_time = ts.to_pydatetime()

        for pos in engine.risk_manager.positions:
            if pos.status != PositionStatus.CLOSED:
                engine.risk_manager.force_close_position(
                    pos, last_price, last_time, "backtest_end",
                )

        # Build results
        perf = engine.get_performance_summary()

        start_date = ""
        end_date = ""
        if "timestamp" in df.columns:
            start_date = str(df["timestamp"].iloc[warmup])[:19]
            end_date = str(df["timestamp"].iloc[-1])[:19]

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            total_bars=len(df) - warmup,
            total_trades=perf.get("total_trades", 0),
            winning_trades=perf.get("winning_trades", 0),
            losing_trades=perf.get("losing_trades", 0),
            win_rate_pct=perf.get("win_rate_pct", 0),
            total_pnl=perf.get("total_pnl", 0),
            total_pnl_pct=perf.get("total_pnl_pct", 0),
            avg_win=perf.get("avg_win", 0),
            avg_loss=perf.get("avg_loss", 0),
            profit_factor=perf.get("profit_factor", 0),
            max_drawdown_pct=perf.get("max_drawdown_pct", 0),
            initial_capital=perf.get("initial_capital", 10000),
            final_equity=perf.get("final_equity", 10000),
            equity_curve=equity_curve,
            trades=engine.get_trade_log(),
        )

    def _empty_result(
        self, symbol: str, timeframe: str, df: pd.DataFrame,
    ) -> BacktestResult:
        """Return an empty backtest result."""
        capital = self.config.get("capital", {}).get("initial", 10000)
        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            start_date="",
            end_date="",
            total_bars=len(df),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0,
            total_pnl=0,
            total_pnl_pct=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            max_drawdown_pct=0,
            initial_capital=capital,
            final_equity=capital,
            equity_curve=[capital],
            trades=[],
        )

    def run_multi_symbol(
        self,
        data: dict[str, pd.DataFrame],
        timeframe: str | None = None,
    ) -> dict[str, BacktestResult]:
        """Run backtest across multiple symbols.

        Args:
            data: Dict mapping symbol -> OHLCV DataFrame.
            timeframe: Candle timeframe.

        Returns:
            Dict mapping symbol -> BacktestResult.
        """
        results = {}
        for symbol, df in data.items():
            results[symbol] = self.run(df, symbol, timeframe)
        return results
