"""
Technical Indicators Module

Расчёт технических индикаторов для подтверждения сигналов наклонок:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- EMA (Exponential Moving Average)
- Bollinger Bands
- ATR (Average True Range)
- Volume analysis
"""

from typing import Any

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Calculates and manages technical indicators for the strategy.

    Все индикаторы рассчитываются на основе библиотеки `ta` или вручную
    через pandas/numpy для полного контроля над параметрами.
    """

    def __init__(self, config: dict[str, Any]):
        ind_cfg = config.get("indicators", {})

        self.rsi_period = ind_cfg.get("rsi", {}).get("period", 14)
        self.rsi_overbought = ind_cfg.get("rsi", {}).get("overbought", 70)
        self.rsi_oversold = ind_cfg.get("rsi", {}).get("oversold", 30)

        macd_cfg = ind_cfg.get("macd", {})
        self.macd_fast = macd_cfg.get("fast", 12)
        self.macd_slow = macd_cfg.get("slow", 26)
        self.macd_signal = macd_cfg.get("signal", 9)

        ema_cfg = ind_cfg.get("ema", {})
        self.ema_fast = ema_cfg.get("fast", 9)
        self.ema_slow = ema_cfg.get("slow", 21)
        self.ema_trend = ema_cfg.get("trend", 50)

        bb_cfg = ind_cfg.get("bollinger", {})
        self.bb_period = bb_cfg.get("period", 20)
        self.bb_std = bb_cfg.get("std_dev", 2.0)

        self.atr_period = ind_cfg.get("atr", {}).get("period", 14)

        vol_cfg = ind_cfg.get("volume", {})
        self.vol_ma_period = vol_cfg.get("ma_period", 20)
        self.vol_spike_mult = vol_cfg.get("spike_multiplier", 1.5)

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators and add them as columns.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume.

        Returns:
            DataFrame with added indicator columns.
        """
        df = df.copy()
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_ema(df)
        df = self.add_bollinger(df)
        df = self.add_atr(df)
        df = self.add_volume_analysis(df)
        return df

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI (Relative Strength Index).

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(span=self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(span=self.rsi_period, min_periods=self.rsi_period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_overbought"] = df["rsi"] > self.rsi_overbought
        df["rsi_oversold"] = df["rsi"] < self.rsi_oversold
        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD (Moving Average Convergence Divergence).

        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD Line, signal_period)
        Histogram = MACD Line - Signal Line
        """
        ema_fast = df["close"].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.macd_slow, adjust=False).mean()

        df["macd_line"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd_line"].ewm(
            span=self.macd_signal, adjust=False,
        ).mean()
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

        # MACD crossover signals
        df["macd_bullish_cross"] = (
            (df["macd_line"] > df["macd_signal"])
            & (df["macd_line"].shift(1) <= df["macd_signal"].shift(1))
        )
        df["macd_bearish_cross"] = (
            (df["macd_line"] < df["macd_signal"])
            & (df["macd_line"].shift(1) >= df["macd_signal"].shift(1))
        )
        return df

    def add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Exponential Moving Averages."""
        df["ema_fast"] = df["close"].ewm(span=self.ema_fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.ema_slow, adjust=False).mean()
        df["ema_trend"] = df["close"].ewm(span=self.ema_trend, adjust=False).mean()

        # Trend direction
        df["ema_uptrend"] = (
            (df["ema_fast"] > df["ema_slow"]) & (df["ema_slow"] > df["ema_trend"])
        )
        df["ema_downtrend"] = (
            (df["ema_fast"] < df["ema_slow"]) & (df["ema_slow"] < df["ema_trend"])
        )
        return df

    def add_bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands.

        Middle = SMA(period)
        Upper = Middle + std_dev * STD(period)
        Lower = Middle - std_dev * STD(period)
        """
        df["bb_middle"] = df["close"].rolling(window=self.bb_period).mean()
        rolling_std = df["close"].rolling(window=self.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + self.bb_std * rolling_std
        df["bb_lower"] = df["bb_middle"] - self.bb_std * rolling_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # Position within bands (0 = lower, 1 = upper)
        band_range = df["bb_upper"] - df["bb_lower"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / band_range.replace(0, np.nan)
        return df

    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ATR (Average True Range) for stop-loss calculation.

        True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR = EMA(True Range, period)
        """
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        df["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = df["true_range"].ewm(span=self.atr_period, adjust=False).mean()
        df["atr_pct"] = df["atr"] / df["close"] * 100  # ATR as % of price
        return df

    def add_volume_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume analysis indicators.

        - Volume MA: скользящая средняя объёма
        - Volume ratio: текущий объём / средний объём
        - Volume spike: объём выше порога
        """
        df["volume_ma"] = df["volume"].rolling(window=self.vol_ma_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, np.nan)
        df["volume_spike"] = df["volume_ratio"] > self.vol_spike_mult

        # On-Balance Volume (OBV) simplified direction
        df["obv_direction"] = np.where(
            df["close"] > df["close"].shift(1), 1,
            np.where(df["close"] < df["close"].shift(1), -1, 0),
        )
        return df

    def get_trend_bias(self, df: pd.DataFrame, idx: int = -1) -> str:
        """Get overall trend bias at a specific bar.

        Returns:
            "bullish", "bearish", or "neutral".
        """
        row = df.iloc[idx]
        bullish_score = 0
        bearish_score = 0

        # EMA alignment
        if row.get("ema_uptrend", False):
            bullish_score += 2
        elif row.get("ema_downtrend", False):
            bearish_score += 2

        # MACD
        if row.get("macd_histogram", 0) > 0:
            bullish_score += 1
        elif row.get("macd_histogram", 0) < 0:
            bearish_score += 1

        # RSI
        rsi = row.get("rsi", 50)
        if rsi > 55:
            bullish_score += 1
        elif rsi < 45:
            bearish_score += 1

        # Price vs Bollinger
        bb_pos = row.get("bb_position", 0.5)
        if bb_pos > 0.6:
            bullish_score += 1
        elif bb_pos < 0.4:
            bearish_score += 1

        if bullish_score > bearish_score + 1:
            return "bullish"
        elif bearish_score > bullish_score + 1:
            return "bearish"
        return "neutral"
