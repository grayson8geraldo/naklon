"""
Risk Management Module — Разгон депозита

Управление рисками для малого депозита $100-200 с плечом:
- Размер позиции с учётом кредитного плеча
- Расчёт маржи
- Агрессивный размер позиции (3% на сделку)
- Трейлинг-стоп
- Частичное закрытие позиции
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"


@dataclass
class Position:
    """Represents an open trading position."""
    id: str
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    take_profit_2: float       # TP2 для полного закрытия
    entry_time: datetime
    leverage: int = 10
    margin_used: float = 0.0   # Маржа = notional / leverage
    status: PositionStatus = PositionStatus.OPEN
    trailing_stop: float | None = None
    trailing_activated: bool = False
    partial_close_done: bool = False
    realized_pnl: float = 0.0
    close_price: float | None = None
    close_time: datetime | None = None
    close_reason: str = ""

    @property
    def risk_amount(self) -> float:
        """Dollar risk for this position."""
        return abs(self.entry_price - self.stop_loss) * self.quantity

    @property
    def notional_value(self) -> float:
        """Position notional value at entry."""
        return self.entry_price * self.quantity

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at current price."""
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Unrealized P&L as percentage of margin used."""
        if self.margin_used <= 0:
            return 0
        return self.unrealized_pnl(current_price) / self.margin_used * 100

    def liquidation_price(self) -> float:
        """Approximate liquidation price (simplified)."""
        liq_distance = self.entry_price / self.leverage * 0.95  # 95% of margin
        if self.side == PositionSide.LONG:
            return self.entry_price - liq_distance
        else:
            return self.entry_price + liq_distance


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot."""
    total_equity: float
    available_margin: float
    open_positions: int
    daily_pnl: float
    daily_pnl_pct: float
    max_daily_loss_reached: bool
    total_risk_exposure: float
    total_risk_pct: float
    leverage: int


class RiskManager:
    """Manages position sizing and risk controls for small deposit acceleration.

    Разгон депозита $100-200 с плечом x10:
    1. 3% депозита на сделку = $4.5 риска при $150
    2. С плечом x10: маржа $4.5, позиция $45
    3. Не более 2 позиций одновременно
    4. Дневной лимит убытков 10%
    5. Трейлинг-стоп после +1%
    6. Частичное закрытие 50% при R:R 1:1
    """

    def __init__(self, config: dict[str, Any]):
        cap_cfg = config.get("capital", {})
        self.initial_capital = cap_cfg.get("initial", 150)
        self.currency = cap_cfg.get("currency", "USDT")

        risk_cfg = config.get("risk_management", {})
        self.max_risk_pct = risk_cfg.get("max_risk_per_trade_pct", 3.0) / 100
        self.max_positions = risk_cfg.get("max_open_positions", 2)
        self.default_rr = risk_cfg.get("default_rr_ratio", 2.0)
        self.max_daily_loss_pct = risk_cfg.get("max_daily_loss_pct", 10.0) / 100
        self.trailing_activation_pct = risk_cfg.get(
            "trailing_stop_activation_pct", 1.0,
        ) / 100
        self.trailing_distance_pct = risk_cfg.get(
            "trailing_stop_distance_pct", 0.4,
        ) / 100
        self.leverage = risk_cfg.get("leverage", 10)

        exit_cfg = config.get("exit_rules", {})
        self.partial_tp_pct = exit_cfg.get("partial_take_profit_pct", 50) / 100
        self.partial_tp_rr = exit_cfg.get("partial_tp_at_rr", 1.0)
        self.use_trailing = exit_cfg.get("use_trailing_stop", True)

        self.equity = self.initial_capital
        self.positions: list[Position] = []
        self.closed_positions: list[Position] = []
        self.daily_pnl = 0.0
        self._position_counter = 0

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        symbol: str,
    ) -> float | None:
        """Calculate position size based on risk per trade WITH LEVERAGE.

        Формула для разгона депозита:
          risk_amount = equity * max_risk_pct         # $4.5 при $150 * 3%
          quantity = risk_amount / |entry - sl|         # сколько монет
          margin_needed = quantity * entry / leverage   # маржа для позиции

        Пример BTC/USDT с $150, плечо x10:
          - Риск 3% = $4.50
          - Entry = $60,000, SL = $59,700 (0.5% = 1 ATR на 5м)
          - risk_per_unit = $300
          - quantity = $4.50 / $300 = 0.015 BTC
          - Notional = 0.015 * $60,000 = $900
          - Margin = $900 / 10 = $90

        Returns:
            Position size in base currency, or None if trade rejected.
        """
        if self._daily_loss_exceeded():
            return None

        open_positions = [p for p in self.positions if p.status != PositionStatus.CLOSED]
        if len(open_positions) >= self.max_positions:
            return None

        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            return None

        # Max risk amount in dollars
        max_risk_amount = self.equity * self.max_risk_pct

        # Position size based on risk
        quantity = max_risk_amount / risk_per_unit

        # Check margin requirement
        margin_needed = (quantity * entry_price) / self.leverage
        available = self._available_margin()

        if margin_needed > available:
            # Scale down to fit available margin
            quantity = (available * self.leverage) / entry_price

        if quantity <= 0:
            return None

        return quantity

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        entry_time: datetime,
        take_profit_2: float | None = None,
    ) -> Position:
        """Open a new position with leverage."""
        self._position_counter += 1
        margin = (quantity * entry_price) / self.leverage

        if take_profit_2 is None:
            take_profit_2 = take_profit

        pos = Position(
            id=f"NAK-{self._position_counter:04d}",
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            take_profit_2=take_profit_2,
            entry_time=entry_time,
            leverage=self.leverage,
            margin_used=margin,
        )
        self.positions.append(pos)
        return pos

    def update_positions(self, current_prices: dict[str, float], current_time: datetime):
        """Update all open positions with current prices."""
        for pos in self.positions:
            if pos.status == PositionStatus.CLOSED:
                continue

            price = current_prices.get(pos.symbol)
            if price is None:
                continue

            if self._check_stop_loss(pos, price, current_time):
                continue
            if self._check_take_profit(pos, price, current_time):
                continue

            self._check_partial_tp(pos, price, current_time)

            if self.use_trailing:
                self._update_trailing_stop(pos, price)

    def _check_stop_loss(
        self, pos: Position, price: float, current_time: datetime,
    ) -> bool:
        active_sl = pos.trailing_stop if pos.trailing_activated else pos.stop_loss

        hit = False
        if pos.side == PositionSide.LONG and price <= active_sl:
            hit = True
        elif pos.side == PositionSide.SHORT and price >= active_sl:
            hit = True

        if hit:
            self._close_position(
                pos, price, current_time,
                "trailing_stop" if pos.trailing_activated else "stop_loss",
            )
            return True
        return False

    def _check_take_profit(
        self, pos: Position, price: float, current_time: datetime,
    ) -> bool:
        # Check TP2 (full close)
        tp = pos.take_profit_2 if pos.partial_close_done else pos.take_profit
        hit = False
        if pos.side == PositionSide.LONG and price >= tp:
            hit = True
        elif pos.side == PositionSide.SHORT and price <= tp:
            hit = True

        if hit:
            self._close_position(pos, price, current_time, "take_profit")
            return True
        return False

    def _check_partial_tp(
        self, pos: Position, price: float, current_time: datetime,
    ):
        if pos.partial_close_done:
            return

        risk_per_unit = abs(pos.entry_price - pos.stop_loss)
        partial_tp_price = (
            pos.entry_price + risk_per_unit * self.partial_tp_rr
            if pos.side == PositionSide.LONG
            else pos.entry_price - risk_per_unit * self.partial_tp_rr
        )

        should_partial = False
        if pos.side == PositionSide.LONG and price >= partial_tp_price:
            should_partial = True
        elif pos.side == PositionSide.SHORT and price <= partial_tp_price:
            should_partial = True

        if should_partial:
            close_qty = pos.quantity * self.partial_tp_pct
            pnl = pos.unrealized_pnl(price) * self.partial_tp_pct
            pos.quantity -= close_qty
            pos.realized_pnl += pnl
            pos.partial_close_done = True
            pos.status = PositionStatus.PARTIALLY_CLOSED
            pos.margin_used *= (1 - self.partial_tp_pct)
            self.daily_pnl += pnl
            self.equity += pnl

            # Move SL to breakeven after partial TP
            pos.stop_loss = pos.entry_price

    def _update_trailing_stop(self, pos: Position, price: float):
        activation_distance = pos.entry_price * self.trailing_activation_pct

        if pos.side == PositionSide.LONG:
            if price >= pos.entry_price + activation_distance:
                new_trail = price * (1 - self.trailing_distance_pct)
                if not pos.trailing_activated or new_trail > pos.trailing_stop:
                    pos.trailing_stop = new_trail
                    pos.trailing_activated = True

        elif pos.side == PositionSide.SHORT:
            if price <= pos.entry_price - activation_distance:
                new_trail = price * (1 + self.trailing_distance_pct)
                if not pos.trailing_activated or new_trail < pos.trailing_stop:
                    pos.trailing_stop = new_trail
                    pos.trailing_activated = True

    def _close_position(
        self, pos: Position, price: float, current_time: datetime, reason: str,
    ):
        pnl = pos.unrealized_pnl(price)
        pos.realized_pnl += pnl
        pos.close_price = price
        pos.close_time = current_time
        pos.close_reason = reason
        pos.status = PositionStatus.CLOSED

        self.daily_pnl += pnl
        self.equity += pnl
        self.closed_positions.append(pos)

    def force_close_position(
        self, pos: Position, price: float, current_time: datetime, reason: str = "manual",
    ):
        self._close_position(pos, price, current_time, reason)

    def get_metrics(self) -> RiskMetrics:
        open_pos = [p for p in self.positions if p.status != PositionStatus.CLOSED]
        total_risk = sum(p.risk_amount for p in open_pos)

        return RiskMetrics(
            total_equity=self.equity,
            available_margin=self._available_margin(),
            open_positions=len(open_pos),
            daily_pnl=self.daily_pnl,
            daily_pnl_pct=self.daily_pnl / self.initial_capital * 100,
            max_daily_loss_reached=self._daily_loss_exceeded(),
            total_risk_exposure=total_risk,
            total_risk_pct=total_risk / self.equity * 100 if self.equity > 0 else 0,
            leverage=self.leverage,
        )

    def reset_daily(self):
        self.daily_pnl = 0.0

    def _available_margin(self) -> float:
        """Available margin = equity - used margin."""
        used = sum(
            p.margin_used
            for p in self.positions
            if p.status != PositionStatus.CLOSED
        )
        return max(0, self.equity - used)

    def _daily_loss_exceeded(self) -> bool:
        return self.daily_pnl <= -(self.equity * self.max_daily_loss_pct)
