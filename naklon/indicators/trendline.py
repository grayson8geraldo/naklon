"""
Trendline (Наклонка) Detection Module

Алгоритм автоматического обнаружения наклонных линий тренда (наклонок):
1. Определение пивотных точек (локальных максимумов и минимумов)
2. Перебор комбинаций пивотных точек для построения линий
3. Валидация наклонок по количеству касаний и углу наклона
4. Обнаружение пробоев наклонок с подтверждением
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class TrendlineType(str, Enum):
    """Type of trendline."""
    SUPPORT = "support"        # Восходящая наклонка по минимумам
    RESISTANCE = "resistance"  # Нисходящая наклонка по максимумам


class BreakoutDirection(str, Enum):
    """Direction of trendline breakout."""
    BULLISH = "bullish"    # Пробой сопротивления вверх
    BEARISH = "bearish"    # Пробой поддержки вниз


@dataclass
class Trendline:
    """Represents a detected trendline (наклонка)."""
    type: TrendlineType
    start_idx: int             # Индекс начальной точки
    end_idx: int               # Индекс конечной точки
    start_price: float         # Цена в начальной точке
    end_price: float           # Цена в конечной точке
    slope: float               # Наклон (цена/бар)
    intercept: float           # Пересечение с осью Y
    angle_deg: float           # Угол наклона в градусах
    touches: int               # Количество касаний
    touch_indices: list[int] = field(default_factory=list)
    strength: float = 0.0      # Сила наклонки (0-1)
    offset: int = 0            # Offset to map window indices to original df indices

    def price_at(self, bar_idx: int) -> float:
        """Get trendline price at given bar index (in original df coordinates)."""
        local_idx = bar_idx - self.offset
        return self.slope * local_idx + self.intercept

    def is_valid(self, current_idx: int) -> bool:
        """Check if trendline is still valid (not too old)."""
        local_current = current_idx - self.offset
        return (local_current - self.end_idx) < 200


@dataclass
class BreakoutSignal:
    """Represents a trendline breakout signal."""
    trendline: Trendline
    direction: BreakoutDirection
    bar_idx: int
    break_price: float
    trendline_price: float
    break_pct: float           # Размер пробоя в %
    volume_confirmed: bool
    retest_occurred: bool = False
    retest_idx: int | None = None


class TrendlineDetector:
    """Detects trendlines (наклонки) and their breakouts.

    Алгоритм:
    - Находит пивотные точки (свинг-хай/свинг-лоу)
    - Строит кандидатные наклонки через пары пивотов
    - Считает касания и валидирует по параметрам
    - Детектирует пробои с подтверждением объёмом
    """

    def __init__(self, config: dict[str, Any]):
        tl_cfg = config.get("trendline", {})
        self.min_touches = tl_cfg.get("min_touches", 3)
        self.lookback = tl_cfg.get("lookback_bars", 120)
        self.touch_tolerance = tl_cfg.get("touch_tolerance_pct", 0.15) / 100
        self.break_threshold = tl_cfg.get("break_threshold_pct", 0.3) / 100
        self.min_angle = tl_cfg.get("min_slope_angle_deg", 10)
        self.max_angle = tl_cfg.get("max_slope_angle_deg", 80)
        self.pivot_left = tl_cfg.get("pivot_left_bars", 5)
        self.pivot_right = tl_cfg.get("pivot_right_bars", 3)

        vol_cfg = config.get("indicators", {}).get("volume", {})
        self.volume_ma_period = vol_cfg.get("ma_period", 20)
        self.volume_spike = vol_cfg.get("spike_multiplier", 1.5)

    def find_pivots(self, df: pd.DataFrame) -> tuple[list[int], list[int]]:
        """Find pivot highs and pivot lows in price data.

        Пивот-хай: бар, чей максимум выше максимумов N баров слева и справа.
        Пивот-лоу: бар, чей минимум ниже минимумов N баров слева и справа.

        Returns:
            Tuple of (pivot_high_indices, pivot_low_indices).
        """
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)

        pivot_highs = []
        pivot_lows = []

        for i in range(self.pivot_left, n - self.pivot_right):
            # Check pivot high
            left_highs = highs[i - self.pivot_left : i]
            right_highs = highs[i + 1 : i + 1 + self.pivot_right]
            if len(left_highs) > 0 and len(right_highs) > 0:
                if highs[i] > np.max(left_highs) and highs[i] > np.max(right_highs):
                    pivot_highs.append(i)

            # Check pivot low
            left_lows = lows[i - self.pivot_left : i]
            right_lows = lows[i + 1 : i + 1 + self.pivot_right]
            if len(left_lows) > 0 and len(right_lows) > 0:
                if lows[i] < np.min(left_lows) and lows[i] < np.min(right_lows):
                    pivot_lows.append(i)

        return pivot_highs, pivot_lows

    def _build_trendline(
        self,
        idx1: int,
        price1: float,
        idx2: int,
        price2: float,
        prices: np.ndarray,
        tl_type: TrendlineType,
        avg_price: float,
    ) -> Trendline | None:
        """Build and validate a single trendline candidate between two pivots.

        Args:
            idx1, price1: First pivot point.
            idx2, price2: Second pivot point.
            prices: Array of prices to check touches against (highs or lows).
            tl_type: Type of trendline.
            avg_price: Average price for angle normalization.

        Returns:
            Trendline if valid, None otherwise.
        """
        if idx2 <= idx1:
            return None

        # Calculate slope and intercept
        slope = (price2 - price1) / (idx2 - idx1)
        intercept = price1 - slope * idx1

        # Calculate angle (normalize slope by average price for meaningful angle)
        normalized_slope = slope / avg_price * 100  # slope as % per bar
        angle_deg = abs(np.degrees(np.arctan(normalized_slope)))

        if angle_deg < self.min_angle or angle_deg > self.max_angle:
            return None

        # Count touches
        tolerance = avg_price * self.touch_tolerance
        touch_indices = []

        for i in range(idx1, len(prices)):
            tl_price = slope * i + intercept
            if abs(prices[i] - tl_price) <= tolerance:
                touch_indices.append(i)

        if len(touch_indices) < self.min_touches:
            return None

        # Calculate strength: more touches + wider span = stronger
        span = touch_indices[-1] - touch_indices[0] if len(touch_indices) > 1 else 1
        strength = min(1.0, (len(touch_indices) / 5) * 0.5 + (span / self.lookback) * 0.5)

        return Trendline(
            type=tl_type,
            start_idx=idx1,
            end_idx=touch_indices[-1],
            start_price=price1,
            end_price=prices[touch_indices[-1]],
            slope=slope,
            intercept=intercept,
            angle_deg=angle_deg,
            touches=len(touch_indices),
            touch_indices=touch_indices,
            strength=strength,
        )

    def detect_trendlines(self, df: pd.DataFrame) -> list[Trendline]:
        """Detect all valid trendlines in the given OHLCV data.

        Алгоритм:
        1. Находим пивотные точки
        2. Для линий поддержки: перебираем пары pivot_lows
        3. Для линий сопротивления: перебираем пары pivot_highs
        4. Валидируем по касаниям и углу

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume.

        Returns:
            List of detected valid trendlines.
        """
        # Use lookback window EXCLUDING the last bar (so breakout can be detected on it)
        # Trendlines are built from historical bars; current bar is checked for breakout
        history = df.iloc[:-1] if len(df) > self.lookback else df
        window_start = max(0, len(history) - self.lookback)
        work_df = history.tail(self.lookback).reset_index(drop=True)
        if len(work_df) < self.pivot_left + self.pivot_right + 2:
            return []

        pivot_highs, pivot_lows = self.find_pivots(work_df)
        highs = work_df["high"].values
        lows = work_df["low"].values
        avg_price = work_df["close"].mean()

        trendlines: list[Trendline] = []

        # Support trendlines (through pivot lows, ascending)
        for i in range(len(pivot_lows)):
            for j in range(i + 1, len(pivot_lows)):
                idx1, idx2 = pivot_lows[i], pivot_lows[j]
                tl = self._build_trendline(
                    idx1, lows[idx1], idx2, lows[idx2],
                    lows, TrendlineType.SUPPORT, avg_price,
                )
                if tl is not None:
                    tl.offset = window_start
                    trendlines.append(tl)

        # Resistance trendlines (through pivot highs, descending)
        for i in range(len(pivot_highs)):
            for j in range(i + 1, len(pivot_highs)):
                idx1, idx2 = pivot_highs[i], pivot_highs[j]
                tl = self._build_trendline(
                    idx1, highs[idx1], idx2, highs[idx2],
                    highs, TrendlineType.RESISTANCE, avg_price,
                )
                if tl is not None:
                    tl.offset = window_start
                    trendlines.append(tl)

        # Remove duplicates: keep strongest trendline in similar slope/intercept groups
        trendlines = self._deduplicate(trendlines, avg_price)

        # Sort by strength descending
        trendlines.sort(key=lambda t: t.strength, reverse=True)

        return trendlines

    def _deduplicate(
        self, trendlines: list[Trendline], avg_price: float,
    ) -> list[Trendline]:
        """Remove near-duplicate trendlines, keeping the strongest."""
        if not trendlines:
            return []

        slope_tol = avg_price * 0.001  # Tolerance for slope similarity
        intercept_tol = avg_price * 0.02  # Tolerance for intercept similarity

        kept: list[Trendline] = []
        for tl in sorted(trendlines, key=lambda t: t.strength, reverse=True):
            is_dup = False
            for existing in kept:
                if (
                    tl.type == existing.type
                    and abs(tl.slope - existing.slope) < slope_tol
                    and abs(tl.intercept - existing.intercept) < intercept_tol
                ):
                    is_dup = True
                    break
            if not is_dup:
                kept.append(tl)

        return kept

    def detect_breakout(
        self,
        df: pd.DataFrame,
        trendlines: list[Trendline],
        bar_idx: int | None = None,
        lookback_bars: int = 5,
    ) -> list[BreakoutSignal]:
        """Detect trendline breakouts at the current or specified bar.

        Пробой наклонки определяется когда:
        - Цена недавно (в последние N баров) была вблизи наклонки
        - Текущая цена закрытия пробила наклонку с превышением порога
        - Опционально: подтверждение объёмом (спайк)

        Args:
            df: OHLCV DataFrame.
            trendlines: Active trendlines to check.
            bar_idx: Bar index to check. Defaults to last bar.
            lookback_bars: Number of recent bars to check for proximity.

        Returns:
            List of breakout signals detected.
        """
        if bar_idx is None:
            bar_idx = len(df) - 1

        if bar_idx < lookback_bars or bar_idx >= len(df):
            return []

        close = df["close"].iloc[bar_idx]
        volume = df["volume"].iloc[bar_idx]

        # Volume confirmation
        vol_start = max(0, bar_idx - self.volume_ma_period)
        vol_ma = df["volume"].iloc[vol_start : bar_idx].mean()
        volume_confirmed = volume > vol_ma * self.volume_spike if vol_ma > 0 else False

        breakouts: list[BreakoutSignal] = []

        for tl in trendlines:
            if not tl.is_valid(bar_idx):
                continue

            tl_price = tl.price_at(bar_idx)
            if tl_price <= 0:
                continue

            break_pct = (close - tl_price) / tl_price

            # Check if price was recently near or on the other side of the trendline
            was_near_or_other_side = False
            for k in range(max(1, bar_idx - lookback_bars), bar_idx):
                past_close = df["close"].iloc[k]
                past_tl = tl.price_at(k)
                proximity = abs(past_close - past_tl) / past_tl
                if proximity < self.touch_tolerance * 3:
                    was_near_or_other_side = True
                    break
                # Check if price was on the other side
                if tl.type == TrendlineType.RESISTANCE and past_close <= past_tl:
                    was_near_or_other_side = True
                    break
                elif tl.type == TrendlineType.SUPPORT and past_close >= past_tl:
                    was_near_or_other_side = True
                    break

            if not was_near_or_other_side:
                continue

            if tl.type == TrendlineType.RESISTANCE:
                # Bullish breakout: price breaks above resistance
                if close > tl_price and break_pct > self.break_threshold:
                    breakouts.append(BreakoutSignal(
                        trendline=tl,
                        direction=BreakoutDirection.BULLISH,
                        bar_idx=bar_idx,
                        break_price=close,
                        trendline_price=tl_price,
                        break_pct=abs(break_pct) * 100,
                        volume_confirmed=volume_confirmed,
                    ))

            elif tl.type == TrendlineType.SUPPORT:
                # Bearish breakout: price breaks below support
                if close < tl_price and abs(break_pct) > self.break_threshold:
                    breakouts.append(BreakoutSignal(
                        trendline=tl,
                        direction=BreakoutDirection.BEARISH,
                        bar_idx=bar_idx,
                        break_price=close,
                        trendline_price=tl_price,
                        break_pct=abs(break_pct) * 100,
                        volume_confirmed=volume_confirmed,
                    ))

        return breakouts

    def check_retest(
        self,
        df: pd.DataFrame,
        breakout: BreakoutSignal,
        max_bars: int = 10,
    ) -> BreakoutSignal:
        """Check if price retests the broken trendline.

        Ретест — когда цена после пробоя возвращается к наклонке,
        касается её и отскакивает в направлении пробоя.

        Args:
            df: OHLCV DataFrame.
            breakout: The breakout signal to check.
            max_bars: Maximum bars to wait for retest.

        Returns:
            Updated breakout signal with retest info.
        """
        tl = breakout.trendline
        start = breakout.bar_idx + 1
        end = min(start + max_bars, len(df))

        for i in range(start, end):
            tl_price = tl.price_at(i)
            tolerance = tl_price * self.touch_tolerance

            if breakout.direction == BreakoutDirection.BULLISH:
                # Price should come back down near trendline (now support)
                if abs(df["low"].iloc[i] - tl_price) <= tolerance:
                    # And then close above it
                    if df["close"].iloc[i] > tl_price:
                        breakout.retest_occurred = True
                        breakout.retest_idx = i
                        return breakout

            elif breakout.direction == BreakoutDirection.BEARISH:
                # Price should come back up near trendline (now resistance)
                if abs(df["high"].iloc[i] - tl_price) <= tolerance:
                    # And then close below it
                    if df["close"].iloc[i] < tl_price:
                        breakout.retest_occurred = True
                        breakout.retest_idx = i
                        return breakout

        return breakout
