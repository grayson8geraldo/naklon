"""
Technical Indicators Module v2.0

Расчёт технических индикаторов:
- RSI, MACD, EMA, Bollinger Bands, ATR
- ADX (фильтр тренда)
- Анализ объёма
- Фильтр режима рынка (trending / ranging / squeeze)
- Сила свечи (candle strength)
"""

from typing import Any

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Calculates technical indicators with market regime analysis."""

    def __init__(self, config: dict[str, Any]):
        ind_cfg = config.get("indicators", {})

        self.rsi_period = ind_cfg.get("rsi", {}).get("period", 14)
        self.rsi_overbought = ind_cfg.get("rsi", {}).get("overbought", 70)
        self.rsi_oversold = ind_cfg.get("rsi", {}).get("oversold", 30)

        macd_cfg = ind_cfg.get("macd", {})
        self.macd_fast = macd_cfg.get("fast", 8)
        self.macd_slow = macd_cfg.get("slow", 21)
        self.macd_signal = macd_cfg.get("signal", 5)

        ema_cfg = ind_cfg.get("ema", {})
        self.ema_fast = ema_cfg.get("fast", 8)
        self.ema_slow = ema_cfg.get("slow", 21)
        self.ema_trend = ema_cfg.get("trend", 50)

        bb_cfg = ind_cfg.get("bollinger", {})
        self.bb_period = bb_cfg.get("period", 20)
        self.bb_std = bb_cfg.get("std_dev", 2.0)

        self.atr_period = ind_cfg.get("atr", {}).get("period", 14)

        adx_cfg = ind_cfg.get("adx", {})
        self.adx_period = adx_cfg.get("period", 14)
        self.adx_threshold = adx_cfg.get("trend_threshold", 20)

        vol_cfg = ind_cfg.get("volume", {})
        self.vol_ma_period = vol_cfg.get("ma_period", 20)
        self.vol_spike_mult = vol_cfg.get("spike_multiplier", 1.5)

        # Market regime config
        regime_cfg = config.get("market_regime", {})
        self.regime_adx_min = regime_cfg.get("adx_min", 20)
        self.regime_atr_expansion_min = regime_cfg.get("atr_expansion_min", 0.8)
        self.regime_bb_squeeze_skip = regime_cfg.get("bb_squeeze_skip", True)
        self.regime_bb_squeeze_threshold = regime_cfg.get("bb_squeeze_threshold", 0.6)

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        df = df.copy()
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_ema(df)
        df = self.add_bollinger(df)
        df = self.add_atr(df)
        df = self.add_adx(df)
        df = self.add_volume_analysis(df)
        df = self.add_candle_strength(df)
        df = self.add_market_regime(df)
        return df

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI (Relative Strength Index)."""
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
        """Add MACD with crossover detection."""
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

        # MACD histogram acceleration (growing or shrinking)
        df["macd_hist_growing"] = df["macd_histogram"] > df["macd_histogram"].shift(1)
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
        """Add Bollinger Bands with squeeze detection."""
        df["bb_middle"] = df["close"].rolling(window=self.bb_period).mean()
        rolling_std = df["close"].rolling(window=self.bb_period).std()
        df["bb_upper"] = df["bb_middle"] + self.bb_std * rolling_std
        df["bb_lower"] = df["bb_middle"] - self.bb_std * rolling_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # Position within bands (0 = lower, 1 = upper)
        band_range = df["bb_upper"] - df["bb_lower"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / band_range.replace(0, np.nan)

        # BB squeeze detection
        bb_width_ma = df["bb_width"].rolling(window=self.bb_period).mean()
        df["bb_squeeze"] = df["bb_width"] < (bb_width_ma * self.regime_bb_squeeze_threshold)
        return df

    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ATR (Average True Range) with expansion detection."""
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        df["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = df["true_range"].ewm(span=self.atr_period, adjust=False).mean()
        df["atr_pct"] = df["atr"] / df["close"] * 100

        # ATR expansion: current ATR vs average ATR
        df["atr_sma"] = df["atr"].rolling(window=20).mean()
        df["atr_expansion"] = df["atr"] / df["atr_sma"].replace(0, np.nan)
        return df

    def add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ADX (Average Directional Index) for trend strength.

        ADX > 20 = trend exists, ADX > 40 = strong trend.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        # Only keep positive directional movement
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        # True range
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        # Smoothed with EMA
        period = self.adx_period
        atr_smooth = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr_smooth.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_smooth.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df["adx"] = dx.ewm(span=period, adjust=False).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        df["adx_trending"] = df["adx"] > self.adx_threshold
        return df

    def add_volume_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume analysis with acceleration."""
        df["volume_ma"] = df["volume"].rolling(window=self.vol_ma_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, np.nan)
        df["volume_spike"] = df["volume_ratio"] > self.vol_spike_mult

        # Volume acceleration: growing volume over last 3 bars
        df["volume_accel"] = (
            (df["volume"] > df["volume"].shift(1))
            & (df["volume"].shift(1) > df["volume"].shift(2))
        )

        # OBV direction
        df["obv_direction"] = np.where(
            df["close"] > df["close"].shift(1), 1,
            np.where(df["close"] < df["close"].shift(1), -1, 0),
        )
        return df

    def add_candle_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candle strength analysis.

        Candle strength: where close is within the high-low range.
        1.0 = close at high (max bullish)
        0.0 = close at low (max bearish)
        """
        candle_range = df["high"] - df["low"]
        df["candle_strength"] = (df["close"] - df["low"]) / candle_range.replace(0, np.nan)
        df["candle_strength"] = df["candle_strength"].fillna(0.5)

        # Body ratio: how much of candle is body vs wicks
        body = (df["close"] - df["open"]).abs()
        df["body_ratio"] = body / candle_range.replace(0, np.nan)
        df["body_ratio"] = df["body_ratio"].fillna(0)

        # Bullish engulfing pattern
        df["bullish_engulfing"] = (
            (df["close"] > df["open"])
            & (df["open"].shift(1) > df["close"].shift(1))
            & (df["close"] > df["open"].shift(1))
            & (df["open"] < df["close"].shift(1))
        )
        df["bearish_engulfing"] = (
            (df["close"] < df["open"])
            & (df["open"].shift(1) < df["close"].shift(1))
            & (df["close"] < df["open"].shift(1))
            & (df["open"] > df["close"].shift(1))
        )
        return df

    def add_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Determine market regime: trending, ranging, or squeeze.

        Режим рынка:
        - TRENDING: ADX > 20, ATR растёт — торгуем
        - RANGING: ADX < 20, ATR нормальный — осторожно
        - SQUEEZE: BB сжатие, ATR падает — НЕ торгуем
        """
        # Conditions
        is_trending = df.get("adx_trending", pd.Series(False, index=df.index))
        atr_exp = df.get("atr_expansion", pd.Series(1.0, index=df.index))
        is_squeeze = df.get("bb_squeeze", pd.Series(False, index=df.index))

        # Regime classification
        conditions = [
            is_trending & (atr_exp >= self.regime_atr_expansion_min),
            ~is_trending & ~is_squeeze,
            is_squeeze,
        ]
        choices = ["trending", "ranging", "squeeze"]
        df["market_regime"] = np.select(conditions, choices, default="ranging")

        # Tradeable flag
        df["regime_tradeable"] = (
            (df["market_regime"] != "squeeze")
            & (atr_exp >= self.regime_atr_expansion_min * 0.7)  # Softer bound
        )
        return df

    def get_trend_bias(self, df: pd.DataFrame, idx: int = -1) -> str:
        """Get overall trend bias."""
        row = df.iloc[idx]
        bullish_score = 0
        bearish_score = 0

        if row.get("ema_uptrend", False):
            bullish_score += 2
        elif row.get("ema_downtrend", False):
            bearish_score += 2

        if row.get("macd_histogram", 0) > 0:
            bullish_score += 1
        elif row.get("macd_histogram", 0) < 0:
            bearish_score += 1

        rsi = row.get("rsi", 50)
        if rsi > 55:
            bullish_score += 1
        elif rsi < 45:
            bearish_score += 1

        bb_pos = row.get("bb_position", 0.5)
        if bb_pos > 0.6:
            bullish_score += 1
        elif bb_pos < 0.4:
            bearish_score += 1

        # ADX bonus
        if row.get("plus_di", 0) > row.get("minus_di", 0):
            bullish_score += 1
        elif row.get("minus_di", 0) > row.get("plus_di", 0):
            bearish_score += 1

        if bullish_score > bearish_score + 1:
            return "bullish"
        elif bearish_score > bullish_score + 1:
            return "bearish"
        return "neutral"

    def get_regime_info(self, df: pd.DataFrame, idx: int = -1) -> dict:
        """Get market regime info for display."""
        row = df.iloc[idx]
        return {
            "regime": row.get("market_regime", "unknown"),
            "adx": round(row.get("adx", 0), 1),
            "atr_expansion": round(row.get("atr_expansion", 1.0), 2),
            "bb_squeeze": bool(row.get("bb_squeeze", False)),
            "tradeable": bool(row.get("regime_tradeable", True)),
        }
