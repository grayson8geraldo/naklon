"""
Signal Generator Module

Генерация торговых сигналов на основе:
1. Пробоя наклонок (основной сигнал)
2. Подтверждение техническими индикаторами
3. Фильтрация по тренду и объёму
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from naklon.indicators.technical import TechnicalIndicators
from naklon.indicators.trendline import (
    BreakoutDirection,
    BreakoutSignal,
    TrendlineDetector,
)


class SignalType(str, Enum):
    LONG = "long"
    SHORT = "short"


class SignalStrength(str, Enum):
    STRONG = "strong"      # Все подтверждения совпадают
    MODERATE = "moderate"  # Большинство подтверждений
    WEAK = "weak"          # Только пробой наклонки


@dataclass
class TradeSignal:
    """A validated trade signal."""
    type: SignalType
    strength: SignalStrength
    symbol: str
    timeframe: str
    bar_idx: int
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    breakout: BreakoutSignal
    confirmations: dict[str, bool]
    score: float  # 0-100 score

    @property
    def risk_pct(self) -> float:
        """Risk as percentage of entry price."""
        return abs(self.entry_price - self.stop_loss) / self.entry_price * 100


class SignalGenerator:
    """Generates and validates trading signals based on trendline breakouts.

    Процесс генерации сигнала:
    1. Обнаружение пробоя наклонки (TrendlineDetector)
    2. Проверка направления тренда (EMA filter)
    3. Проверка RSI (не в зоне перекупленности/перепроданности)
    4. Подтверждение MACD
    5. Подтверждение объёмом
    6. Расчёт стоп-лосса (ATR-based) и тейк-профита (R:R)
    7. Скоринг сигнала
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.trendline_detector = TrendlineDetector(config)
        self.indicators = TechnicalIndicators(config)

        entry_cfg = config.get("entry_rules", {})
        self.require_volume = entry_cfg.get("require_volume_confirmation", True)
        self.require_retest = entry_cfg.get("require_retest", False)
        self.min_rsi_long = entry_cfg.get("min_rsi_for_long", 35)
        self.max_rsi_long = entry_cfg.get("max_rsi_for_long", 65)
        self.min_rsi_short = entry_cfg.get("min_rsi_for_short", 35)
        self.max_rsi_short = entry_cfg.get("max_rsi_for_short", 65)
        self.macd_confirm = entry_cfg.get("macd_confirmation", True)
        self.ema_filter = entry_cfg.get("ema_trend_filter", True)
        self.wait_candles = entry_cfg.get("wait_candles_after_break", 1)

        exit_cfg = config.get("exit_rules", {})
        self.tp_rr = exit_cfg.get("take_profit_at_rr", 2.0)

        risk_cfg = config.get("risk_management", {})
        self.default_rr = risk_cfg.get("default_rr_ratio", 2.0)

        self.atr_period = config.get("indicators", {}).get("atr", {}).get("period", 14)

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> list[TradeSignal]:
        """Generate trade signals for the given data.

        Args:
            df: OHLCV DataFrame with at least lookback + indicator warmup bars.
            symbol: Trading pair symbol (e.g., "BTC/USDT").
            timeframe: Candle timeframe (e.g., "15m").

        Returns:
            List of validated trade signals.
        """
        # Calculate indicators
        df = self.indicators.calculate_all(df)

        # Detect trendlines
        trendlines = self.trendline_detector.detect_trendlines(df)

        if not trendlines:
            return []

        # Detect breakouts at the last bar
        breakouts = self.trendline_detector.detect_breakout(df, trendlines)

        signals: list[TradeSignal] = []
        for breakout in breakouts:
            signal = self._validate_and_build_signal(
                df, breakout, symbol, timeframe,
            )
            if signal is not None:
                signals.append(signal)

        # Sort by score descending
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

    def scan_historical(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> list[TradeSignal]:
        """Scan historical data for signals (used in backtesting).

        Instead of only checking the last bar, scans all bars after warmup.

        Args:
            df: Full OHLCV DataFrame.
            symbol: Trading pair symbol.
            timeframe: Candle timeframe.

        Returns:
            List of all historical signals found.
        """
        df = self.indicators.calculate_all(df)

        warmup = max(
            self.trendline_detector.lookback,
            self.indicators.ema_trend + 10,
            self.indicators.bb_period + 10,
        )

        all_signals: list[TradeSignal] = []

        # Detect trendlines on sliding window
        for i in range(warmup, len(df)):
            window = df.iloc[: i + 1]
            trendlines = self.trendline_detector.detect_trendlines(window)
            if not trendlines:
                continue

            breakouts = self.trendline_detector.detect_breakout(
                window, trendlines, bar_idx=len(window) - 1,
            )

            for breakout in breakouts:
                signal = self._validate_and_build_signal(
                    window, breakout, symbol, timeframe,
                )
                if signal is not None:
                    # Remap bar_idx to the original df index
                    signal.bar_idx = i
                    all_signals.append(signal)

        return all_signals

    def _validate_and_build_signal(
        self,
        df: pd.DataFrame,
        breakout: BreakoutSignal,
        symbol: str,
        timeframe: str,
    ) -> TradeSignal | None:
        """Validate breakout and build a trade signal with confirmations.

        Args:
            df: OHLCV DataFrame with indicators calculated.
            breakout: Detected breakout signal.
            symbol: Trading pair.
            timeframe: Timeframe.

        Returns:
            TradeSignal if valid, None otherwise.
        """
        idx = breakout.bar_idx
        if idx >= len(df):
            return None

        row = df.iloc[idx]
        close = row["close"]
        atr = row.get("atr", close * 0.02)  # Fallback: 2% of price

        if pd.isna(atr) or atr <= 0:
            atr = close * 0.02

        # Determine signal type
        if breakout.direction == BreakoutDirection.BULLISH:
            signal_type = SignalType.LONG
        else:
            signal_type = SignalType.SHORT

        # Run confirmations
        confirmations = self._check_confirmations(df, idx, signal_type)

        # Volume filter (hard requirement if configured)
        if self.require_volume and not breakout.volume_confirmed:
            return None

        # Retest filter
        if self.require_retest:
            breakout = self.trendline_detector.check_retest(df, breakout)
            if not breakout.retest_occurred:
                return None

        # Calculate stop-loss and take-profit
        if signal_type == SignalType.LONG:
            stop_loss = close - 1.5 * atr
            risk = close - stop_loss
            take_profit = close + risk * self.default_rr
        else:
            stop_loss = close + 1.5 * atr
            risk = stop_loss - close
            take_profit = close - risk * self.default_rr

        if risk <= 0:
            return None

        risk_reward = abs(close - take_profit) / risk

        # Calculate signal score
        score = self._calculate_score(breakout, confirmations, risk_reward)

        # Determine strength
        confirmed_count = sum(1 for v in confirmations.values() if v)
        total = len(confirmations)
        if confirmed_count >= total - 1:
            strength = SignalStrength.STRONG
        elif confirmed_count >= total // 2:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        # Filter out weak signals
        if score < 40:
            return None

        return TradeSignal(
            type=signal_type,
            strength=strength,
            symbol=symbol,
            timeframe=timeframe,
            bar_idx=idx,
            entry_price=close,
            stop_loss=round(stop_loss, 8),
            take_profit=round(take_profit, 8),
            risk_reward=round(risk_reward, 2),
            breakout=breakout,
            confirmations=confirmations,
            score=round(score, 1),
        )

    def _check_confirmations(
        self,
        df: pd.DataFrame,
        idx: int,
        signal_type: SignalType,
    ) -> dict[str, bool]:
        """Check all confirmation indicators.

        Returns:
            Dictionary of confirmation name -> passed boolean.
        """
        row = df.iloc[idx]
        confirmations = {}

        # RSI confirmation
        rsi = row.get("rsi", 50)
        if signal_type == SignalType.LONG:
            confirmations["rsi_ok"] = self.min_rsi_long <= rsi <= self.max_rsi_long
        else:
            confirmations["rsi_ok"] = self.min_rsi_short <= rsi <= self.max_rsi_short

        # MACD confirmation
        if self.macd_confirm:
            macd_hist = row.get("macd_histogram", 0)
            if signal_type == SignalType.LONG:
                confirmations["macd_ok"] = macd_hist > 0 or row.get(
                    "macd_bullish_cross", False,
                )
            else:
                confirmations["macd_ok"] = macd_hist < 0 or row.get(
                    "macd_bearish_cross", False,
                )

        # EMA trend filter
        if self.ema_filter:
            if signal_type == SignalType.LONG:
                confirmations["ema_trend_ok"] = bool(row.get("ema_uptrend", False)) or (
                    row.get("close", 0) > row.get("ema_trend", 0)
                )
            else:
                confirmations["ema_trend_ok"] = bool(row.get("ema_downtrend", False)) or (
                    row.get("close", 0) < row.get("ema_trend", 0)
                )

        # Volume spike
        confirmations["volume_spike"] = bool(row.get("volume_spike", False))

        return confirmations

    def _calculate_score(
        self,
        breakout: BreakoutSignal,
        confirmations: dict[str, bool],
        risk_reward: float,
    ) -> float:
        """Calculate signal quality score (0-100).

        Scoring:
        - Trendline strength:  0-25 points
        - Break size:          0-15 points
        - Confirmations:       0-40 points
        - Risk/Reward:         0-20 points
        """
        score = 0.0

        # Trendline strength (max 25)
        score += breakout.trendline.strength * 25

        # Break size (max 15) — bigger break = stronger signal, up to 2%
        score += min(15, breakout.break_pct / 2 * 15)

        # Confirmations (max 40)
        if confirmations:
            confirmed = sum(1 for v in confirmations.values() if v)
            score += (confirmed / len(confirmations)) * 40

        # Risk/Reward (max 20)
        score += min(20, risk_reward / 3 * 20)

        return min(100, score)
