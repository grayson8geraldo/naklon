"""
Strategy Engine v2.0 — Core orchestrator

Рабочий цикл:
1. Проверка режима рынка (не торгуем в squeeze)
2. Генерация сигналов по наклонкам с 6 подтверждениями
3. Smart стоп-лосс + 3-уровневый тейк-профит
4. Адаптивное плечо + градуированные лимиты
"""

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from naklon.strategy.risk import PositionSide, RiskManager
from naklon.strategy.signals import SignalGenerator, SignalType, TradeSignal

logger = logging.getLogger("naklon.engine")


class StrategyEngine:
    """Main strategy engine with regime filtering and adaptive risk."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.signal_generator = SignalGenerator(config)
        self.risk_manager = RiskManager(config)

        exit_cfg = config.get("exit_rules", {})
        self.exit_on_opposite = exit_cfg.get("exit_on_opposite_signal", True)
        self.tp1_rr = exit_cfg.get("tp1_at_rr", 0.75)
        self.tp2_rr = exit_cfg.get("tp2_at_rr", 1.5)
        self.tp3_rr = exit_cfg.get("tp3_at_rr", 3.0)

        self.symbols = config.get("symbols", ["BTC/USDT"])
        self.primary_tf = config.get("timeframes", {}).get("primary", "5m")

        self._last_signals: dict[str, TradeSignal] = {}
        self._trade_log: list[dict] = []
        self._bar_counter = 0

    def on_new_candle(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_time: datetime,
    ) -> list[TradeSignal]:
        """Process a new candle."""
        self._bar_counter += 1
        self.risk_manager.set_current_bar(self._bar_counter)

        current_price = df["close"].iloc[-1]
        self.risk_manager.update_positions({symbol: current_price}, current_time)

        metrics = self.risk_manager.get_metrics()
        if metrics.max_daily_loss_reached:
            logger.warning(
                "HARD DAILY LIMIT (%.2f%%). Trading stopped.",
                metrics.daily_pnl_pct,
            )
            return []

        if metrics.soft_limit_reached:
            logger.info(
                "SOFT DAILY LIMIT (%.2f%%). Position size halved.",
                metrics.daily_pnl_pct,
            )

        signals = self.signal_generator.generate_signals(
            df, symbol, self.primary_tf,
        )

        executed_signals = []
        for signal in signals:
            executed = self._process_signal(signal, current_time)
            if executed:
                executed_signals.append(signal)

        return executed_signals

    def _process_signal(self, signal: TradeSignal, current_time: datetime) -> bool:
        symbol = signal.symbol

        if self.exit_on_opposite:
            self._close_opposite_positions(signal, current_time)

        existing = self._get_open_positions(symbol)
        for pos in existing:
            if (
                (signal.type == SignalType.LONG and pos.side == PositionSide.LONG)
                or (signal.type == SignalType.SHORT and pos.side == PositionSide.SHORT)
            ):
                logger.info(
                    "Already have %s position for %s, skipping.", pos.side.value, symbol,
                )
                return False

        quantity = self.risk_manager.calculate_position_size(
            signal.entry_price, signal.stop_loss, symbol,
        )
        if quantity is None:
            logger.info(
                "Position rejected by risk manager for %s (score: %.1f).",
                symbol, signal.score,
            )
            return False

        side = (
            PositionSide.LONG if signal.type == SignalType.LONG else PositionSide.SHORT
        )

        # Calculate 3-level TP
        risk = abs(signal.entry_price - signal.stop_loss)
        if signal.type == SignalType.LONG:
            tp1 = signal.entry_price + risk * self.tp1_rr
            tp2 = signal.entry_price + risk * self.tp2_rr
            tp3 = signal.entry_price + risk * self.tp3_rr
        else:
            tp1 = signal.entry_price - risk * self.tp1_rr
            tp2 = signal.entry_price - risk * self.tp2_rr
            tp3 = signal.entry_price - risk * self.tp3_rr

        leverage = self.risk_manager.leverage
        margin = (quantity * signal.entry_price) / leverage
        notional = quantity * signal.entry_price

        pos = self.risk_manager.open_position(
            symbol=symbol,
            side=side,
            entry_price=signal.entry_price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=tp1,
            entry_time=current_time,
            take_profit_2=tp2,
            take_profit_3=tp3,
        )

        risk_amount = abs(signal.entry_price - signal.stop_loss) * quantity
        trade_info = {
            "time": current_time.isoformat(),
            "position_id": pos.id,
            "symbol": symbol,
            "side": side.value,
            "entry_price": signal.entry_price,
            "quantity": round(quantity, 8),
            "notional_value": round(notional, 2),
            "margin": round(margin, 2),
            "leverage": leverage,
            "stop_loss": signal.stop_loss,
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "tp3": round(tp3, 2),
            "risk_amount": round(risk_amount, 2),
            "risk_reward": signal.risk_reward,
            "signal_score": signal.score,
            "signal_strength": signal.strength.value,
            "market_regime": signal.market_regime,
            "trendline_type": signal.breakout.trendline.type.value,
            "trendline_touches": signal.breakout.trendline.touches,
            "trendline_angle": round(signal.breakout.trendline.angle_deg, 1),
            "break_pct": round(signal.breakout.break_pct, 3),
            "confirmations": signal.confirmations,
        }
        self._trade_log.append(trade_info)

        logger.info(
            "OPEN %s %s x%d | Entry: %.2f | SL: %.2f | TP1: %.2f | TP2: %.2f | TP3: %.2f | "
            "Margin: $%.2f | Risk: $%.2f | Score: %.1f | Regime: %s",
            side.value.upper(),
            symbol,
            leverage,
            signal.entry_price,
            signal.stop_loss,
            tp1, tp2, tp3,
            margin,
            risk_amount,
            signal.score,
            signal.market_regime,
        )

        self._last_signals[symbol] = signal
        return True

    def _close_opposite_positions(
        self, signal: TradeSignal, current_time: datetime,
    ):
        for pos in self._get_open_positions(signal.symbol):
            is_opposite = (
                (signal.type == SignalType.LONG and pos.side == PositionSide.SHORT)
                or (signal.type == SignalType.SHORT and pos.side == PositionSide.LONG)
            )
            if is_opposite:
                self.risk_manager.force_close_position(
                    pos, signal.entry_price, current_time, "opposite_signal",
                )
                logger.info(
                    "CLOSE %s %s (opposite signal) | PnL: $%.2f",
                    pos.side.value.upper(),
                    pos.symbol,
                    pos.realized_pnl,
                )

    def _get_open_positions(self, symbol: str) -> list:
        from naklon.strategy.risk import PositionStatus
        return [
            p for p in self.risk_manager.positions
            if p.symbol == symbol and p.status != PositionStatus.CLOSED
        ]

    def get_trade_log(self) -> list[dict]:
        return self._trade_log

    def get_performance_summary(self) -> dict:
        closed = self.risk_manager.closed_positions
        if not closed:
            return {
                "total_trades": 0,
                "equity": self.risk_manager.equity,
                "pnl": 0,
                "pnl_pct": 0,
            }

        wins = [p for p in closed if p.realized_pnl > 0]
        losses = [p for p in closed if p.realized_pnl <= 0]
        total_pnl = sum(p.realized_pnl for p in closed)

        avg_win = sum(p.realized_pnl for p in wins) / len(wins) if wins else 0
        avg_loss = sum(p.realized_pnl for p in losses) / len(losses) if losses else 0

        gross_profit = sum(p.realized_pnl for p in wins)
        gross_loss = abs(sum(p.realized_pnl for p in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown
        equity_curve = []
        running_equity = self.risk_manager.initial_capital
        for p in closed:
            running_equity += p.realized_pnl
            equity_curve.append(running_equity)

        max_dd = 0
        peak = equity_curve[0] if equity_curve else self.risk_manager.initial_capital
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return {
            "total_trades": len(closed),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(len(wins) / len(closed) * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(
                total_pnl / self.risk_manager.initial_capital * 100, 2,
            ),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "initial_capital": self.risk_manager.initial_capital,
            "final_equity": round(self.risk_manager.equity, 2),
        }
