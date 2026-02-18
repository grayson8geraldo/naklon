"""
Risk Management Module v2.0 — Разгон депозита

- 3-уровневая система тейк-профитов (TP1/TP2/TP3)
- Адаптивное плечо (уменьшается при серии убытков)
- Градуированные дневные лимиты (мягкий + жёсткий)
- Слиппаж и комиссии
- Маржинальный расчёт позиции
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
    take_profit: float         # TP1
    take_profit_2: float       # TP2
    take_profit_3: float       # TP3
    entry_time: datetime
    leverage: int = 10
    margin_used: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    trailing_stop: float | None = None
    trailing_activated: bool = False
    tp1_done: bool = False
    tp2_done: bool = False
    realized_pnl: float = 0.0
    close_price: float | None = None
    close_time: datetime | None = None
    close_reason: str = ""
    entry_bar: int = 0        # For max hold time tracking

    @property
    def risk_amount(self) -> float:
        return abs(self.entry_price - self.stop_loss) * self.quantity

    @property
    def notional_value(self) -> float:
        return self.entry_price * self.quantity

    def unrealized_pnl(self, current_price: float) -> float:
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.margin_used <= 0:
            return 0
        return self.unrealized_pnl(current_price) / self.margin_used * 100

    def liquidation_price(self) -> float:
        """Approximate liquidation price (Binance-style)."""
        # Maintenance margin rate ~5% for 10x
        maintenance_rate = 0.05
        liq_distance = self.entry_price * maintenance_rate
        if self.side == PositionSide.LONG:
            return self.entry_price - (self.entry_price / self.leverage) + liq_distance
        else:
            return self.entry_price + (self.entry_price / self.leverage) - liq_distance


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot."""
    total_equity: float
    available_margin: float
    open_positions: int
    daily_pnl: float
    daily_pnl_pct: float
    max_daily_loss_reached: bool
    soft_limit_reached: bool
    total_risk_exposure: float
    total_risk_pct: float
    leverage: int
    effective_leverage: int
    losing_streak: int


class RiskManager:
    """Risk management for deposit acceleration with adaptive leverage.

    Разгон депозита $100-200 с адаптивным плечом:
    1. 3% на сделку, мягкий лимит 5%: уменьшить размер, жёсткий 10%: стоп
    2. 3 убытка подряд → плечо x5 вместо x10
    3. TP1 (40% позиции) → TP2 (35%) → TP3 (25%)
    4. Слиппаж 0.05% + комиссия 0.04% учитываются
    """

    def __init__(self, config: dict[str, Any]):
        cap_cfg = config.get("capital", {})
        self.initial_capital = cap_cfg.get("initial", 150)

        risk_cfg = config.get("risk_management", {})
        self.max_risk_pct = risk_cfg.get("max_risk_per_trade_pct", 3.0) / 100
        self.max_positions = risk_cfg.get("max_open_positions", 2)
        self.default_rr = risk_cfg.get("default_rr_ratio", 2.0)
        self.max_daily_loss_pct = risk_cfg.get("max_daily_loss_pct", 10.0) / 100
        self.soft_daily_loss_pct = risk_cfg.get("soft_daily_loss_pct", 5.0) / 100
        self.trailing_activation_pct = risk_cfg.get(
            "trailing_stop_activation_pct", 0.8,
        ) / 100
        self.trailing_distance_pct = risk_cfg.get(
            "trailing_stop_distance_pct", 0.3,
        ) / 100
        self.base_leverage = risk_cfg.get("leverage", 10)
        self.adaptive_leverage = risk_cfg.get("adaptive_leverage", True)
        self.losing_streak_cut = risk_cfg.get("losing_streak_leverage_cut", 3)
        self.slippage_pct = risk_cfg.get("slippage_pct", 0.05) / 100
        self.commission_pct = risk_cfg.get("commission_pct", 0.04) / 100

        exit_cfg = config.get("exit_rules", {})
        self.tp1_rr = exit_cfg.get("tp1_at_rr", 0.75)
        self.tp1_pct = exit_cfg.get("tp1_close_pct", 40) / 100
        self.tp2_rr = exit_cfg.get("tp2_at_rr", 1.5)
        self.tp2_pct = exit_cfg.get("tp2_close_pct", 35) / 100
        self.tp3_rr = exit_cfg.get("tp3_at_rr", 3.0)
        self.use_trailing = exit_cfg.get("use_trailing_stop", True)
        self.max_hold_bars = exit_cfg.get("max_hold_bars", 60)

        self.equity = self.initial_capital
        self.positions: list[Position] = []
        self.closed_positions: list[Position] = []
        self.daily_pnl = 0.0
        self._position_counter = 0
        self._losing_streak = 0
        self._current_bar = 0

    @property
    def leverage(self) -> int:
        """Current effective leverage (adapts to losing streak)."""
        if self.adaptive_leverage and self._losing_streak >= self.losing_streak_cut:
            return max(3, self.base_leverage // 2)
        return self.base_leverage

    def set_current_bar(self, bar_idx: int):
        """Track current bar for max hold time."""
        self._current_bar = bar_idx

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        symbol: str,
    ) -> float | None:
        """Calculate position size with adaptive leverage and margin check."""
        if self._daily_loss_hard():
            return None

        open_positions = [p for p in self.positions if p.status != PositionStatus.CLOSED]
        if len(open_positions) >= self.max_positions:
            return None

        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            return None

        # Effective risk % (halved if soft daily limit reached)
        effective_risk_pct = self.max_risk_pct
        if self._daily_loss_soft():
            effective_risk_pct = self.max_risk_pct / 2

        max_risk_amount = self.equity * effective_risk_pct
        quantity = max_risk_amount / risk_per_unit

        # Apply slippage to entry
        slippage_cost = entry_price * self.slippage_pct
        adjusted_risk = risk_per_unit + slippage_cost
        quantity = max_risk_amount / adjusted_risk

        # Check margin with current leverage
        lev = self.leverage
        margin_needed = (quantity * entry_price) / lev
        available = self._available_margin()

        if margin_needed > available:
            quantity = (available * lev) / entry_price

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
        take_profit_3: float | None = None,
    ) -> Position:
        """Open a new position with 3-level TP and commission deduction."""
        self._position_counter += 1
        lev = self.leverage
        margin = (quantity * entry_price) / lev

        # Deduct entry commission
        commission = quantity * entry_price * self.commission_pct
        self.equity -= commission

        if take_profit_2 is None:
            take_profit_2 = take_profit
        if take_profit_3 is None:
            take_profit_3 = take_profit_2

        pos = Position(
            id=f"NAK-{self._position_counter:04d}",
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            entry_time=entry_time,
            leverage=lev,
            margin_used=margin,
            entry_bar=self._current_bar,
        )
        self.positions.append(pos)
        return pos

    def update_positions(self, current_prices: dict[str, float], current_time: datetime):
        """Update all open positions."""
        for pos in self.positions:
            if pos.status == PositionStatus.CLOSED:
                continue

            price = current_prices.get(pos.symbol)
            if price is None:
                continue

            # Check max hold time
            if self.max_hold_bars > 0:
                bars_held = self._current_bar - pos.entry_bar
                if bars_held >= self.max_hold_bars:
                    pnl = pos.unrealized_pnl(price)
                    if pnl >= 0:
                        # Close at breakeven or profit if held too long
                        self._close_position(pos, price, current_time, "max_hold_time")
                        continue

            if self._check_stop_loss(pos, price, current_time):
                continue
            if self._check_tp3(pos, price, current_time):
                continue

            self._check_tp1(pos, price, current_time)
            self._check_tp2(pos, price, current_time)

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
            reason = "trailing_stop" if pos.trailing_activated else "stop_loss"
            self._close_position(pos, price, current_time, reason)
            return True
        return False

    def _check_tp1(self, pos: Position, price: float, current_time: datetime):
        """TP1: close 40% of position."""
        if pos.tp1_done:
            return

        risk = abs(pos.entry_price - pos.stop_loss)
        tp1_price = (
            pos.entry_price + risk * self.tp1_rr
            if pos.side == PositionSide.LONG
            else pos.entry_price - risk * self.tp1_rr
        )

        should_tp = False
        if pos.side == PositionSide.LONG and price >= tp1_price:
            should_tp = True
        elif pos.side == PositionSide.SHORT and price <= tp1_price:
            should_tp = True

        if should_tp:
            close_qty = pos.quantity * self.tp1_pct
            pnl = pos.unrealized_pnl(price) * self.tp1_pct
            # Commission on partial close
            commission = close_qty * price * self.commission_pct
            pnl -= commission

            pos.quantity -= close_qty
            pos.realized_pnl += pnl
            pos.tp1_done = True
            pos.status = PositionStatus.PARTIALLY_CLOSED
            pos.margin_used *= (1 - self.tp1_pct)
            self.daily_pnl += pnl
            self.equity += pnl

            # Move SL to breakeven after TP1
            pos.stop_loss = pos.entry_price

    def _check_tp2(self, pos: Position, price: float, current_time: datetime):
        """TP2: close 35% of original (next chunk)."""
        if pos.tp2_done or not pos.tp1_done:
            return

        risk = abs(pos.entry_price - pos.stop_loss) if pos.stop_loss != pos.entry_price else (
            abs(pos.entry_price - pos.take_profit) / self.tp1_rr
        )
        # Use original risk
        original_risk = abs(pos.entry_price - pos.take_profit) / self.tp1_rr if self.tp1_rr > 0 else risk

        tp2_price = (
            pos.entry_price + original_risk * self.tp2_rr
            if pos.side == PositionSide.LONG
            else pos.entry_price - original_risk * self.tp2_rr
        )

        should_tp = False
        if pos.side == PositionSide.LONG and price >= tp2_price:
            should_tp = True
        elif pos.side == PositionSide.SHORT and price <= tp2_price:
            should_tp = True

        if should_tp:
            # Close proportional to remaining
            remaining_after_tp1 = 1 - self.tp1_pct
            tp2_fraction = self.tp2_pct / remaining_after_tp1 if remaining_after_tp1 > 0 else 0.5
            tp2_fraction = min(1.0, tp2_fraction)

            close_qty = pos.quantity * tp2_fraction
            pnl = pos.unrealized_pnl(price) * tp2_fraction
            commission = close_qty * price * self.commission_pct
            pnl -= commission

            pos.quantity -= close_qty
            pos.realized_pnl += pnl
            pos.tp2_done = True
            pos.margin_used *= (1 - tp2_fraction)
            self.daily_pnl += pnl
            self.equity += pnl

            # Move SL to TP1 level after TP2
            original_risk_2 = abs(pos.entry_price - pos.take_profit) / self.tp1_rr if self.tp1_rr > 0 else 0
            if pos.side == PositionSide.LONG:
                pos.stop_loss = pos.entry_price + original_risk_2 * self.tp1_rr * 0.5
            else:
                pos.stop_loss = pos.entry_price - original_risk_2 * self.tp1_rr * 0.5

    def _check_tp3(
        self, pos: Position, price: float, current_time: datetime,
    ) -> bool:
        """TP3: close remaining position (full exit)."""
        if not pos.tp1_done:
            # TP3 only triggers after TP1
            return False

        tp = pos.take_profit_3

        hit = False
        if pos.side == PositionSide.LONG and price >= tp:
            hit = True
        elif pos.side == PositionSide.SHORT and price <= tp:
            hit = True

        if hit:
            self._close_position(pos, price, current_time, "take_profit_3")
            return True
        return False

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
        # Apply slippage against us
        if pos.side == PositionSide.LONG:
            exit_price = price * (1 - self.slippage_pct)
        else:
            exit_price = price * (1 + self.slippage_pct)

        pnl = pos.unrealized_pnl(exit_price)

        # Commission on close
        commission = pos.quantity * exit_price * self.commission_pct
        pnl -= commission

        pos.realized_pnl += pnl
        pos.close_price = exit_price
        pos.close_time = current_time
        pos.close_reason = reason
        pos.status = PositionStatus.CLOSED

        self.daily_pnl += pnl
        self.equity += pnl
        self.closed_positions.append(pos)

        # Track losing streak for adaptive leverage
        if pos.realized_pnl < 0:
            self._losing_streak += 1
        else:
            self._losing_streak = 0

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
            max_daily_loss_reached=self._daily_loss_hard(),
            soft_limit_reached=self._daily_loss_soft(),
            total_risk_exposure=total_risk,
            total_risk_pct=total_risk / self.equity * 100 if self.equity > 0 else 0,
            leverage=self.base_leverage,
            effective_leverage=self.leverage,
            losing_streak=self._losing_streak,
        )

    def reset_daily(self):
        self.daily_pnl = 0.0

    def _available_margin(self) -> float:
        used = sum(
            p.margin_used
            for p in self.positions
            if p.status != PositionStatus.CLOSED
        )
        return max(0, self.equity - used)

    def _daily_loss_soft(self) -> bool:
        """Soft limit: reduce position size."""
        return self.daily_pnl <= -(self.equity * self.soft_daily_loss_pct)

    def _daily_loss_hard(self) -> bool:
        """Hard limit: stop trading."""
        return self.daily_pnl <= -(self.equity * self.max_daily_loss_pct)
