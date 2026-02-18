"""
Signal Generator Module v2.0

Генерация сигналов для разгона депозита:
1. Фильтр режима рынка (ADX + BB squeeze + ATR)
2. Пробой наклонки с подтверждением объёмом
3. 6 подтверждений: RSI, MACD, EMA, объём, сила свечи, режим рынка
4. Smart стоп-лосс: trendline + swing low + ATR
5. Скоринг 0-100 с жёстким порогом 50+
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
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


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
    score: float
    market_regime: str = "unknown"

    @property
    def risk_pct(self) -> float:
        return abs(self.entry_price - self.stop_loss) / self.entry_price * 100


class SignalGenerator:
    """Generates signals with market regime filter and smart stop-loss."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.trendline_detector = TrendlineDetector(config)
        self.indicators = TechnicalIndicators(config)

        entry_cfg = config.get("entry_rules", {})
        self.require_volume = entry_cfg.get("require_volume_confirmation", True)
        self.require_retest = entry_cfg.get("require_retest", False)
        self.min_rsi_long = entry_cfg.get("min_rsi_for_long", 30)
        self.max_rsi_long = entry_cfg.get("max_rsi_for_long", 70)
        self.min_rsi_short = entry_cfg.get("min_rsi_for_short", 30)
        self.max_rsi_short = entry_cfg.get("max_rsi_for_short", 70)
        self.macd_confirm = entry_cfg.get("macd_confirmation", True)
        self.ema_filter = entry_cfg.get("ema_trend_filter", True)
        self.wait_candles = entry_cfg.get("wait_candles_after_break", 0)
        self.min_score = entry_cfg.get("min_signal_score", 50)
        self.require_candle_str = entry_cfg.get("require_candle_strength", True)
        self.candle_str_threshold = entry_cfg.get("candle_strength_threshold", 0.4)

        exit_cfg = config.get("exit_rules", {})
        self.tp_rr = exit_cfg.get("tp3_at_rr", exit_cfg.get("take_profit_at_rr", 2.0))

        risk_cfg = config.get("risk_management", {})
        self.default_rr = risk_cfg.get("default_rr_ratio", 2.0)

        self.atr_period = config.get("indicators", {}).get("atr", {}).get("period", 14)

        # Market regime config
        regime_cfg = config.get("market_regime", {})
        self.require_trend = regime_cfg.get("require_trend", True)

    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> list[TradeSignal]:
        """Generate trade signals with all filters applied."""
        # Calculate indicators
        df = self.indicators.calculate_all(df)

        # === MARKET REGIME FILTER ===
        last_row = df.iloc[-1]
        regime = last_row.get("market_regime", "ranging")
        tradeable = bool(last_row.get("regime_tradeable", True))

        if self.require_trend and not tradeable:
            return []

        # Detect trendlines and breakouts
        trendlines = self.trendline_detector.detect_trendlines(df)
        if not trendlines:
            return []

        breakouts = self.trendline_detector.detect_breakout(df, trendlines)

        signals: list[TradeSignal] = []
        for breakout in breakouts:
            signal = self._validate_and_build_signal(
                df, breakout, symbol, timeframe, regime,
            )
            if signal is not None:
                signals.append(signal)

        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

    def scan_historical(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> list[TradeSignal]:
        """Scan historical data for signals (backtesting)."""
        df = self.indicators.calculate_all(df)

        warmup = max(
            self.trendline_detector.lookback,
            self.indicators.ema_trend + 10,
            self.indicators.bb_period + 10,
        )

        all_signals: list[TradeSignal] = []

        for i in range(warmup, len(df)):
            window = df.iloc[: i + 1]
            trendlines = self.trendline_detector.detect_trendlines(window)
            if not trendlines:
                continue

            breakouts = self.trendline_detector.detect_breakout(
                window, trendlines, bar_idx=len(window) - 1,
            )

            for breakout in breakouts:
                regime = window.iloc[-1].get("market_regime", "ranging")
                tradeable = bool(window.iloc[-1].get("regime_tradeable", True))

                # Apply regime filter in backtest too
                if self.require_trend and not tradeable:
                    continue

                signal = self._validate_and_build_signal(
                    window, breakout, symbol, timeframe, regime,
                )
                if signal is not None:
                    signal.bar_idx = i
                    all_signals.append(signal)

        return all_signals

    def _validate_and_build_signal(
        self,
        df: pd.DataFrame,
        breakout: BreakoutSignal,
        symbol: str,
        timeframe: str,
        regime: str = "unknown",
    ) -> TradeSignal | None:
        """Validate breakout and build signal with smart SL and all confirmations."""
        idx = breakout.bar_idx
        if idx >= len(df):
            return None

        row = df.iloc[idx]
        close = row["close"]
        atr = row.get("atr", close * 0.02)

        if pd.isna(atr) or atr <= 0:
            atr = close * 0.02

        # Determine signal type
        if breakout.direction == BreakoutDirection.BULLISH:
            signal_type = SignalType.LONG
        else:
            signal_type = SignalType.SHORT

        # Run all confirmations (6 checks)
        confirmations = self._check_confirmations(df, idx, signal_type, breakout)

        # Volume: hard requirement if configured
        if self.require_volume and not breakout.volume_confirmed:
            return None

        # Retest filter
        if self.require_retest:
            breakout = self.trendline_detector.check_retest(df, breakout)
            if not breakout.retest_occurred:
                return None

        # === SMART STOP-LOSS ===
        stop_loss = self._calculate_smart_sl(df, idx, signal_type, breakout, atr)

        risk = abs(close - stop_loss)
        if risk <= 0:
            return None

        # Cap SL distance at 2.5% of entry
        max_sl_distance = close * 0.025
        if risk > max_sl_distance:
            if signal_type == SignalType.LONG:
                stop_loss = close - max_sl_distance
            else:
                stop_loss = close + max_sl_distance
            risk = max_sl_distance

        take_profit = (
            close + risk * self.default_rr if signal_type == SignalType.LONG
            else close - risk * self.default_rr
        )

        risk_reward = abs(close - take_profit) / risk if risk > 0 else 0

        # Calculate score
        score = self._calculate_score(breakout, confirmations, risk_reward, regime)

        # Determine strength
        confirmed_count = sum(1 for v in confirmations.values() if v)
        total = len(confirmations)
        if confirmed_count >= total - 1:
            strength = SignalStrength.STRONG
        elif confirmed_count >= total // 2 + 1:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        # Filter: minimum score
        if score < self.min_score:
            return None

        # Filter: reject WEAK signals entirely
        if strength == SignalStrength.WEAK:
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
            market_regime=regime,
        )

    def _calculate_smart_sl(
        self,
        df: pd.DataFrame,
        idx: int,
        signal_type: SignalType,
        breakout: BreakoutSignal,
        atr: float,
    ) -> float:
        """Smart stop-loss: picks best of trendline-based, swing-based, and ATR-based.

        For LONG:
          1. Trendline SL: trendline price - 0.3 ATR (наклонка как поддержка)
          2. Swing SL: recent swing low - 0.3 ATR
          3. ATR SL: entry - 1.5 ATR (fallback)
          → Pick the TIGHTEST SL that gives min 0.3% risk

        For SHORT: mirror logic.
        """
        close = df["close"].iloc[idx]
        tl_price = breakout.trendline.price_at(idx)
        buffer = atr * 0.3

        if signal_type == SignalType.LONG:
            # Option 1: Trendline as support
            sl_trendline = tl_price - buffer
            # Option 2: Recent swing low
            sl_swing = self.trendline_detector.find_recent_swing_low(df, idx, lookback=20) - buffer
            # Option 3: ATR-based
            sl_atr = close - 1.5 * atr

            # Pick the highest (tightest for long) that still gives min risk
            min_sl = close * 0.997  # Minimum 0.3% risk
            candidates = [sl_trendline, sl_swing, sl_atr]
            valid = [s for s in candidates if s < min_sl]

            if valid:
                return max(valid)  # Tightest valid SL
            return sl_atr  # Fallback

        else:
            sl_trendline = tl_price + buffer
            sl_swing = self.trendline_detector.find_recent_swing_high(df, idx, lookback=20) + buffer
            sl_atr = close + 1.5 * atr

            min_sl = close * 1.003  # Minimum 0.3% risk
            candidates = [sl_trendline, sl_swing, sl_atr]
            valid = [s for s in candidates if s > min_sl]

            if valid:
                return min(valid)  # Tightest valid SL
            return sl_atr

    def _check_confirmations(
        self,
        df: pd.DataFrame,
        idx: int,
        signal_type: SignalType,
        breakout: BreakoutSignal,
    ) -> dict[str, bool]:
        """Check all 6 confirmation indicators."""
        row = df.iloc[idx]
        confirmations = {}

        # 1. RSI confirmation
        rsi = row.get("rsi", 50)
        if signal_type == SignalType.LONG:
            confirmations["rsi_ok"] = self.min_rsi_long <= rsi <= self.max_rsi_long
        else:
            confirmations["rsi_ok"] = self.min_rsi_short <= rsi <= self.max_rsi_short

        # 2. MACD confirmation (histogram + growing)
        if self.macd_confirm:
            macd_hist = row.get("macd_histogram", 0)
            macd_growing = row.get("macd_hist_growing", False)
            if signal_type == SignalType.LONG:
                confirmations["macd_ok"] = (macd_hist > 0) or (
                    bool(row.get("macd_bullish_cross", False))
                ) or (macd_growing and macd_hist > -abs(macd_hist) * 0.5)
            else:
                confirmations["macd_ok"] = (macd_hist < 0) or (
                    bool(row.get("macd_bearish_cross", False))
                ) or (not macd_growing and macd_hist < abs(macd_hist) * 0.5)

        # 3. EMA trend filter
        if self.ema_filter:
            if signal_type == SignalType.LONG:
                confirmations["ema_trend_ok"] = bool(row.get("ema_uptrend", False)) or (
                    row.get("close", 0) > row.get("ema_trend", 0)
                )
            else:
                confirmations["ema_trend_ok"] = bool(row.get("ema_downtrend", False)) or (
                    row.get("close", 0) < row.get("ema_trend", 0)
                )

        # 4. Volume spike
        confirmations["volume_ok"] = bool(row.get("volume_spike", False))

        # 5. Candle strength
        candle_str = breakout.candle_strength
        if self.require_candle_str:
            if signal_type == SignalType.LONG:
                confirmations["candle_ok"] = candle_str >= (1 - self.candle_str_threshold)
            else:
                confirmations["candle_ok"] = candle_str >= (1 - self.candle_str_threshold)
        else:
            confirmations["candle_ok"] = True

        # 6. ADX trend strength
        adx = row.get("adx", 0)
        confirmations["adx_ok"] = adx > 18  # Slightly below threshold

        return confirmations

    def _calculate_score(
        self,
        breakout: BreakoutSignal,
        confirmations: dict[str, bool],
        risk_reward: float,
        regime: str,
    ) -> float:
        """Calculate signal quality score (0-100).

        Scoring v2.0:
        - Trendline strength:    0-30 points
        - Break size:            0-20 points
        - Confirmations:         0-35 points
        - Risk/Reward:           0-10 points
        - Regime bonus:          0-5 points
        """
        score = 0.0

        # Trendline strength (max 30)
        score += breakout.trendline.strength * 30

        # Break size (max 20) — bigger break = stronger
        score += min(20, breakout.break_pct / 1.5 * 20)

        # Confirmations (max 35)
        if confirmations:
            confirmed = sum(1 for v in confirmations.values() if v)
            score += (confirmed / len(confirmations)) * 35

        # Risk/Reward (max 10)
        if risk_reward >= 2.0:
            score += 10
        elif risk_reward >= 1.5:
            score += 7
        elif risk_reward >= 1.0:
            score += 4
        else:
            score -= 5  # Penalize bad R:R

        # Regime bonus (max 5)
        if regime == "trending":
            score += 5
        elif regime == "ranging":
            score += 0
        # squeeze = already filtered out

        return min(100, max(0, score))
