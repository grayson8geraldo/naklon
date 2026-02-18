"""
Data Fetcher Module

Получение OHLCV данных с криптобирж через ccxt:
- Загрузка исторических данных для бэктестинга
- Получение данных в реальном времени
"""

import logging
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger("naklon.data")


class DataFetcher:
    """Fetches OHLCV data from cryptocurrency exchanges via ccxt.

    Supports:
    - Historical data download for backtesting
    - Multiple symbols and timeframes
    - Automatic pagination for large date ranges
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        exchange_cfg = config.get("exchange", {})
        self.exchange_name = exchange_cfg.get("name", "binance")
        self.testnet = exchange_cfg.get("testnet", True)
        self._exchange = None

    def _get_exchange(self):
        """Lazy-initialize the exchange connection."""
        if self._exchange is None:
            import ccxt
            exchange_class = getattr(ccxt, self.exchange_name)
            self._exchange = exchange_class({
                "enableRateLimit": True,
            })
            if self.testnet and hasattr(self._exchange, "set_sandbox_mode"):
                self._exchange.set_sandbox_mode(True)
            self._exchange.load_markets()
        return self._exchange

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        since: datetime | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV candle data from exchange.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT").
            timeframe: Candle timeframe (e.g., "1m", "5m", "15m", "1h", "4h", "1d").
            since: Start datetime. If None, fetches most recent data.
            limit: Number of candles to fetch (max depends on exchange).

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        exchange = self._get_exchange()

        since_ms = int(since.timestamp() * 1000) if since else None

        logger.info(
            "Fetching %d candles for %s (%s) from %s",
            limit, symbol, timeframe, self.exchange_name,
        )

        all_candles = []
        fetched = 0
        current_since = since_ms

        while fetched < limit:
            batch_limit = min(1000, limit - fetched)
            candles = exchange.fetch_ohlcv(
                symbol, timeframe, since=current_since, limit=batch_limit,
            )
            if not candles:
                break

            all_candles.extend(candles)
            fetched += len(candles)

            if len(candles) < batch_limit:
                break

            # Move since to after last candle
            current_since = candles[-1][0] + 1

        if not all_candles:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        logger.info("Fetched %d candles for %s", len(df), symbol)
        return df

    def fetch_multi_timeframe(
        self,
        symbol: str,
        timeframes: list[str],
        limit: int = 500,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple timeframes.

        Args:
            symbol: Trading pair.
            timeframes: List of timeframes to fetch.
            limit: Candles per timeframe.

        Returns:
            Dict mapping timeframe -> DataFrame.
        """
        result = {}
        for tf in timeframes:
            result[tf] = self.fetch_ohlcv(symbol, tf, limit=limit)
        return result

    @staticmethod
    def generate_sample_data(
        bars: int = 500,
        base_price: float = 60000.0,
        volatility: float = 0.02,
        seed: int | None = 42,
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing and backtesting.

        Creates realistic-looking price action with trends and reversals.

        Args:
            bars: Number of candles to generate.
            base_price: Starting price.
            volatility: Price volatility factor.
            seed: Random seed for reproducibility.

        Returns:
            Synthetic OHLCV DataFrame.
        """
        import numpy as np

        if seed is not None:
            np.random.seed(seed)

        timestamps = pd.date_range(
            start="2025-01-01", periods=bars, freq="15min",
        )

        prices = [base_price]
        # Create realistic crypto price action with trends, breakouts and reversals
        trend = 0.0
        regime_len = 0
        regime_target = np.random.randint(30, 80)
        for i in range(1, bars):
            regime_len += 1
            # Switch trend regime periodically (simulates trend/range/reversal)
            if regime_len >= regime_target:
                trend = np.random.choice([-1, 0, 1]) * np.random.uniform(0.001, 0.004)
                regime_len = 0
                regime_target = np.random.randint(30, 80)

            # Trend persistence + noise
            trend = trend * 0.98 + np.random.randn() * 0.0008
            # Random walk with trend — higher volatility for crypto
            vol_factor = volatility * (1 + 0.5 * abs(np.random.randn()))
            change = np.random.randn() * vol_factor * prices[-1] / 100 + trend * prices[-1]
            # Occasional spikes (simulates breakout candles)
            if np.random.random() < 0.03:
                change *= np.random.uniform(2, 4)
            new_price = prices[-1] + change
            prices.append(max(new_price, prices[-1] * 0.92))

        prices = np.array(prices)
        noise = np.random.uniform(0.997, 1.003, bars)

        opens = prices * noise
        # Wider wicks for crypto candles
        highs = np.maximum(opens, prices) * np.random.uniform(1.001, 1.008, bars)
        lows = np.minimum(opens, prices) * np.random.uniform(0.992, 0.999, bars)
        closes = prices
        # Volume with spikes during breakouts
        base_vol = np.random.lognormal(mean=10, sigma=0.8, size=bars)
        vol_spikes = np.where(np.random.random(bars) < 0.08, np.random.uniform(2, 5, bars), 1)
        volumes = base_vol * vol_spikes

        return pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        })
