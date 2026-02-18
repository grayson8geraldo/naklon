"""
Trendline (Наклонка) Detection Module v2.0

Улучшения:
- ATR-based tolerance (масштабируется с волатильностью)
- Recency weighting (свежие касания весят больше)
- Надёжные пивоты (5 баров слева, 3 справа)
- Angle filter 8-60 градусов (отсекает шум)
- Smart breakout: порог 0.3% + volume + candle strength
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class TrendlineType(str, Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"


class BreakoutDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class Trendline:
    """Represents a detected trendline (наклонка)."""
    type: TrendlineType
    start_idx: int
    end_idx: int
    start_price: float
    end_price: float
    slope: float
    intercept: float
    angle_deg: float
    touches: int
    touch_indices: list[int] = field(default_factory=list)
    strength: float = 0.0
    offset: int = 0

    def price_at(self, bar_idx: int) -> float:
        """Get trendline price at given bar index."""
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
    break_pct: float
    volume_confirmed: bool
    candle_strength: float = 0.5   # 0=bearish, 1=bullish
    retest_occurred: bool = False
    retest_idx: int | None = None


class TrendlineDetector:
    """Detects trendlines with ATR-based tolerance and recency weighting."""

    def __init__(self, config: dict[str, Any]):
        tl_cfg = config.get("trendline", {})
        self.min_touches = tl_cfg.get("min_touches", 2)
        self.lookback = tl_cfg.get("lookback_bars", 80)
        self.touch_tolerance_atr = tl_cfg.get("touch_tolerance_atr", 1.0)
        # Fallback for old config format
        self.touch_tolerance_pct = tl_cfg.get("touch_tolerance_pct", 0.2) / 100
        self.break_threshold = tl_cfg.get("break_threshold_pct", 0.3) / 100
        self.min_angle = tl_cfg.get("min_slope_angle_deg", 8)
        self.max_angle = tl_cfg.get("max_slope_angle_deg", 60)
        self.pivot_left = tl_cfg.get("pivot_left_bars", 5)
        self.pivot_right = tl_cfg.get("pivot_right_bars", 3)
        self.recency_weight = tl_cfg.get("recency_weight", 2.0)
        self.max_age = tl_cfg.get("max_age_bars", 150)

        vol_cfg = config.get("indicators", {}).get("volume", {})
        self.volume_ma_period = vol_cfg.get("ma_period", 20)
        self.volume_spike = vol_cfg.get("spike_multiplier", 1.5)

    def find_pivots(self, df: pd.DataFrame) -> tuple[list[int], list[int]]:
        """Find pivot highs and lows with stricter validation."""
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)

        pivot_highs = []
        pivot_lows = []

        for i in range(self.pivot_left, n - self.pivot_right):
            left_highs = highs[i - self.pivot_left : i]
            right_highs = highs[i + 1 : i + 1 + self.pivot_right]
            if len(left_highs) > 0 and len(right_highs) > 0:
                if highs[i] > np.max(left_highs) and highs[i] > np.max(right_highs):
                    pivot_highs.append(i)

            left_lows = lows[i - self.pivot_left : i]
            right_lows = lows[i + 1 : i + 1 + self.pivot_right]
            if len(left_lows) > 0 and len(right_lows) > 0:
                if lows[i] < np.min(left_lows) and lows[i] < np.min(right_lows):
                    pivot_lows.append(i)

        return pivot_highs, pivot_lows

    def _get_tolerance(self, df: pd.DataFrame, idx: int, avg_price: float) -> float:
        """Get ATR-based tolerance at a given bar index.

        Uses ATR if available, falls back to percentage-based tolerance.
        """
        if "atr" in df.columns and idx < len(df):
            atr_val = df["atr"].iloc[idx]
            if not pd.isna(atr_val) and atr_val > 0:
                return atr_val * self.touch_tolerance_atr
        return avg_price * self.touch_tolerance_pct

    def _build_trendline(
        self,
        idx1: int,
        price1: float,
        idx2: int,
        price2: float,
        prices: np.ndarray,
        tl_type: TrendlineType,
        avg_price: float,
        df: pd.DataFrame,
    ) -> Trendline | None:
        """Build and validate a trendline with ATR-based tolerance and recency weighting."""
        if idx2 <= idx1:
            return None

        slope = (price2 - price1) / (idx2 - idx1)
        intercept = price1 - slope * idx1

        # Angle calculation
        normalized_slope = slope / avg_price * 100
        angle_deg = abs(np.degrees(np.arctan(normalized_slope)))

        if angle_deg < self.min_angle or angle_deg > self.max_angle:
            return None

        # Count touches with ATR-based tolerance
        touch_indices = []
        n = len(prices)

        for i in range(idx1, n):
            tl_price = slope * i + intercept
            tolerance = self._get_tolerance(df, i, avg_price)
            if abs(prices[i] - tl_price) <= tolerance:
                touch_indices.append(i)

        if len(touch_indices) < self.min_touches:
            return None

        # Calculate strength with recency weighting
        span = touch_indices[-1] - touch_indices[0] if len(touch_indices) > 1 else 1
        total_bars = n

        # Recency score: touches in last 30% of bars count double
        recency_cutoff = int(total_bars * 0.7)
        base_touches = sum(1 for t in touch_indices if t < recency_cutoff)
        recent_touches = sum(1 for t in touch_indices if t >= recency_cutoff)
        weighted_touches = base_touches + recent_touches * self.recency_weight

        touch_score = min(1.0, weighted_touches / 5)
        span_score = min(1.0, span / max(self.lookback, 1))
        # Bonus: age of last touch (fresher = stronger)
        last_touch_age = total_bars - touch_indices[-1] - 1
        freshness = max(0.0, 1.0 - last_touch_age / 30)  # 0 if last touch >30 bars ago

        strength = touch_score * 0.4 + span_score * 0.3 + freshness * 0.3
        strength = min(1.0, strength)

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
        """Detect all valid trendlines in OHLCV data."""
        history = df.iloc[:-1] if len(df) > self.lookback else df
        window_start = max(0, len(history) - self.lookback)
        work_df = history.tail(self.lookback).reset_index(drop=True)
        if len(work_df) < self.pivot_left + self.pivot_right + 2:
            return []

        # Calculate ATR on work_df for tolerance
        if "atr" not in work_df.columns:
            prev_close = work_df["close"].shift(1)
            tr1 = work_df["high"] - work_df["low"]
            tr2 = (work_df["high"] - prev_close).abs()
            tr3 = (work_df["low"] - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            work_df["atr"] = tr.ewm(span=14, adjust=False).mean()

        pivot_highs, pivot_lows = self.find_pivots(work_df)
        highs = work_df["high"].values
        lows = work_df["low"].values
        avg_price = work_df["close"].mean()

        trendlines: list[Trendline] = []

        # Support trendlines (through pivot lows)
        for i in range(len(pivot_lows)):
            for j in range(i + 1, len(pivot_lows)):
                idx1, idx2 = pivot_lows[i], pivot_lows[j]
                tl = self._build_trendline(
                    idx1, lows[idx1], idx2, lows[idx2],
                    lows, TrendlineType.SUPPORT, avg_price, work_df,
                )
                if tl is not None:
                    tl.offset = window_start
                    trendlines.append(tl)

        # Resistance trendlines (through pivot highs)
        for i in range(len(pivot_highs)):
            for j in range(i + 1, len(pivot_highs)):
                idx1, idx2 = pivot_highs[i], pivot_highs[j]
                tl = self._build_trendline(
                    idx1, highs[idx1], idx2, highs[idx2],
                    highs, TrendlineType.RESISTANCE, avg_price, work_df,
                )
                if tl is not None:
                    tl.offset = window_start
                    trendlines.append(tl)

        trendlines = self._deduplicate(trendlines, avg_price)
        trendlines.sort(key=lambda t: t.strength, reverse=True)
        return trendlines

    def _deduplicate(
        self, trendlines: list[Trendline], avg_price: float,
    ) -> list[Trendline]:
        """Remove near-duplicate trendlines, keeping the strongest."""
        if not trendlines:
            return []

        slope_tol = avg_price * 0.001
        intercept_tol = avg_price * 0.02

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
        """Detect trendline breakouts with candle strength analysis."""
        if bar_idx is None:
            bar_idx = len(df) - 1

        if bar_idx < lookback_bars or bar_idx >= len(df):
            return []

        close = df["close"].iloc[bar_idx]
        high = df["high"].iloc[bar_idx]
        low = df["low"].iloc[bar_idx]
        volume = df["volume"].iloc[bar_idx]

        # Volume confirmation
        vol_start = max(0, bar_idx - self.volume_ma_period)
        vol_ma = df["volume"].iloc[vol_start : bar_idx].mean()
        volume_confirmed = volume > vol_ma * self.volume_spike if vol_ma > 0 else False

        # Candle strength
        candle_range = high - low
        candle_str = (close - low) / candle_range if candle_range > 0 else 0.5

        breakouts: list[BreakoutSignal] = []

        for tl in trendlines:
            if not tl.is_valid(bar_idx):
                continue

            tl_price = tl.price_at(bar_idx)
            if tl_price <= 0:
                continue

            break_pct = (close - tl_price) / tl_price

            # Check proximity in recent bars
            was_near_or_other_side = False
            for k in range(max(1, bar_idx - lookback_bars), bar_idx):
                past_close = df["close"].iloc[k]
                past_tl = tl.price_at(k)
                proximity = abs(past_close - past_tl) / past_tl
                if proximity < self.break_threshold * 3:
                    was_near_or_other_side = True
                    break
                if tl.type == TrendlineType.RESISTANCE and past_close <= past_tl:
                    was_near_or_other_side = True
                    break
                elif tl.type == TrendlineType.SUPPORT and past_close >= past_tl:
                    was_near_or_other_side = True
                    break

            if not was_near_or_other_side:
                continue

            if tl.type == TrendlineType.RESISTANCE:
                # Bullish breakout: above resistance + threshold
                if close > tl_price and break_pct > self.break_threshold:
                    breakouts.append(BreakoutSignal(
                        trendline=tl,
                        direction=BreakoutDirection.BULLISH,
                        bar_idx=bar_idx,
                        break_price=close,
                        trendline_price=tl_price,
                        break_pct=abs(break_pct) * 100,
                        volume_confirmed=volume_confirmed,
                        candle_strength=candle_str,
                    ))

            elif tl.type == TrendlineType.SUPPORT:
                # Bearish breakout: below support + threshold
                if close < tl_price and abs(break_pct) > self.break_threshold:
                    breakouts.append(BreakoutSignal(
                        trendline=tl,
                        direction=BreakoutDirection.BEARISH,
                        bar_idx=bar_idx,
                        break_price=close,
                        trendline_price=tl_price,
                        break_pct=abs(break_pct) * 100,
                        volume_confirmed=volume_confirmed,
                        candle_strength=1.0 - candle_str,  # Invert for short
                    ))

        return breakouts

    def check_retest(
        self,
        df: pd.DataFrame,
        breakout: BreakoutSignal,
        max_bars: int = 10,
    ) -> BreakoutSignal:
        """Check if price retests the broken trendline."""
        tl = breakout.trendline
        start = breakout.bar_idx + 1
        end = min(start + max_bars, len(df))
        tolerance_pct = self.break_threshold

        for i in range(start, end):
            tl_price = tl.price_at(i)
            tolerance = tl_price * tolerance_pct

            if breakout.direction == BreakoutDirection.BULLISH:
                if abs(df["low"].iloc[i] - tl_price) <= tolerance:
                    if df["close"].iloc[i] > tl_price:
                        breakout.retest_occurred = True
                        breakout.retest_idx = i
                        return breakout

            elif breakout.direction == BreakoutDirection.BEARISH:
                if abs(df["high"].iloc[i] - tl_price) <= tolerance:
                    if df["close"].iloc[i] < tl_price:
                        breakout.retest_occurred = True
                        breakout.retest_idx = i
                        return breakout

        return breakout

    def find_recent_swing_low(self, df: pd.DataFrame, bar_idx: int, lookback: int = 20) -> float:
        """Find the most recent swing low for smart SL placement."""
        start = max(0, bar_idx - lookback)
        lows = df["low"].iloc[start:bar_idx + 1]
        return lows.min() if len(lows) > 0 else df["close"].iloc[bar_idx]

    def find_recent_swing_high(self, df: pd.DataFrame, bar_idx: int, lookback: int = 20) -> float:
        """Find the most recent swing high for smart SL placement."""
        start = max(0, bar_idx - lookback)
        highs = df["high"].iloc[start:bar_idx + 1]
        return highs.max() if len(highs) > 0 else df["close"].iloc[bar_idx]
