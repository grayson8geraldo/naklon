#!/usr/bin/env python3
"""
Naklon — Crypto Intraday Trendline (Наклонки) Trading Strategy

Главный модуль:
  - backtest:  Запуск бэктеста на исторических/синтетических данных
  - live:      Подключение к бирже (ccxt) и живой мониторинг сигналов
  - analyze:   Анализ текущих наклонок на графике

Usage:
  python main.py backtest [--symbol BTC/USDT] [--bars 1000]
  python main.py analyze  [--symbol BTC/USDT]
  python main.py live     [--symbol BTC/USDT]
"""

import argparse
import json
import sys
from pathlib import Path

from naklon.backtest.backtester import Backtester
from naklon.data.fetcher import DataFetcher
from naklon.indicators.technical import TechnicalIndicators
from naklon.indicators.trendline import TrendlineDetector
from naklon.utils.config import load_config
from naklon.utils.logger import setup_logger


def cmd_backtest(config: dict, args: argparse.Namespace):
    """Run backtest on historical or synthetic data."""
    logger = setup_logger("naklon", config["logging"]["level"], config["logging"]["file"])

    symbol = args.symbol or config["symbols"][0]
    bars = args.bars

    logger.info("=" * 60)
    logger.info("  NAKLON — Backtest Mode")
    logger.info("  Стратегия: Торговля по наклонкам (Trendline Breakout)")
    logger.info("  Capital: $%s | Symbol: %s", config["capital"]["initial"], symbol)
    logger.info("=" * 60)

    # Get data
    if args.live_data:
        fetcher = DataFetcher(config)
        timeframe = config["timeframes"]["primary"]
        df = fetcher.fetch_ohlcv(symbol, timeframe, limit=bars)
        if df.empty:
            logger.error("Failed to fetch data from exchange.")
            sys.exit(1)
    else:
        logger.info("Using synthetic data (%d bars, seed=%d)", bars, args.seed)
        df = DataFetcher.generate_sample_data(
            bars=bars,
            base_price=args.base_price,
            volatility=args.volatility,
            seed=args.seed,
        )

    # Run backtest
    backtester = Backtester(config)
    timeframe = config["timeframes"]["primary"]
    result = backtester.run(df, symbol, timeframe)

    # Print results
    print(result.summary())

    # Save trade log
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "symbol": result.symbol,
                    "timeframe": result.timeframe,
                    "start_date": result.start_date,
                    "end_date": result.end_date,
                    "total_bars": result.total_bars,
                    "total_trades": result.total_trades,
                    "win_rate_pct": result.win_rate_pct,
                    "total_pnl": result.total_pnl,
                    "total_pnl_pct": result.total_pnl_pct,
                    "profit_factor": result.profit_factor,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "initial_capital": result.initial_capital,
                    "final_equity": result.final_equity,
                },
                "trades": result.trades,
            }, f, indent=2, default=str)
        logger.info("Results saved to %s", args.output)


def cmd_analyze(config: dict, args: argparse.Namespace):
    """Analyze current trendlines for a symbol."""
    logger = setup_logger("naklon", config["logging"]["level"], config["logging"]["file"])

    symbol = args.symbol or config["symbols"][0]

    logger.info("=" * 60)
    logger.info("  NAKLON — Analyze Mode")
    logger.info("  Analyzing trendlines for %s", symbol)
    logger.info("=" * 60)

    # Get data
    if args.live_data:
        fetcher = DataFetcher(config)
        timeframe = config["timeframes"]["primary"]
        df = fetcher.fetch_ohlcv(symbol, timeframe, limit=args.bars)
    else:
        df = DataFetcher.generate_sample_data(bars=args.bars, seed=args.seed)

    # Calculate indicators
    indicators = TechnicalIndicators(config)
    df = indicators.calculate_all(df)

    # Detect trendlines
    detector = TrendlineDetector(config)
    trendlines = detector.detect_trendlines(df)

    if not trendlines:
        print(f"\nNo valid trendlines found for {symbol}")
        return

    print(f"\n{'='*60}")
    print(f"  Found {len(trendlines)} trendlines (наклонок) for {symbol}")
    print(f"{'='*60}")

    for i, tl in enumerate(trendlines, 1):
        print(f"\n  #{i} [{tl.type.value.upper()}]")
        print(f"  Touches: {tl.touches} | Angle: {tl.angle_deg:.1f}°")
        print(f"  Strength: {tl.strength:.2f}")
        print(f"  Start: bar {tl.start_idx} @ {tl.start_price:.2f}")
        print(f"  End:   bar {tl.end_idx} @ {tl.end_price:.2f}")
        print(f"  Slope: {tl.slope:.4f}/bar")

    # Check for breakouts at last bar
    breakouts = detector.detect_breakout(df, trendlines)
    if breakouts:
        print(f"\n{'─'*60}")
        print(f"  ACTIVE BREAKOUT SIGNALS:")
        for bo in breakouts:
            print(f"  → {bo.direction.value.upper()} breakout!")
            print(f"    Break price: {bo.break_price:.2f}")
            print(f"    Trendline at: {bo.trendline_price:.2f}")
            print(f"    Break size: {bo.break_pct:.3f}%")
            print(f"    Volume confirmed: {'YES' if bo.volume_confirmed else 'NO'}")
    else:
        print(f"\n  No active breakouts at current bar.")

    # Show current indicator snapshot
    trend_bias = indicators.get_trend_bias(df)
    last = df.iloc[-1]
    print(f"\n{'─'*60}")
    print(f"  INDICATOR SNAPSHOT:")
    print(f"  Trend bias: {trend_bias.upper()}")
    print(f"  RSI: {last.get('rsi', 0):.1f}")
    print(f"  MACD Hist: {last.get('macd_histogram', 0):.4f}")
    print(f"  ATR: {last.get('atr', 0):.2f} ({last.get('atr_pct', 0):.2f}%)")
    print(f"  Volume ratio: {last.get('volume_ratio', 0):.2f}x")
    print(f"  BB position: {last.get('bb_position', 0):.2f}")
    print(f"{'='*60}\n")


def cmd_live(config: dict, args: argparse.Namespace):
    """Run live monitoring (signal generation only — no auto-execution)."""
    logger = setup_logger("naklon", config["logging"]["level"], config["logging"]["file"])

    symbol = args.symbol or config["symbols"][0]
    timeframe = config["timeframes"]["primary"]

    logger.info("=" * 60)
    logger.info("  NAKLON — Live Monitor Mode")
    logger.info("  Monitoring %s on %s timeframe", symbol, timeframe)
    logger.info("  Capital: $%s", config["capital"]["initial"])
    logger.info("  NOTE: Signal monitoring only. No auto-execution.")
    logger.info("=" * 60)

    fetcher = DataFetcher(config)
    indicators = TechnicalIndicators(config)
    detector = TrendlineDetector(config)

    import time

    # Determine sleep interval from timeframe
    tf_seconds = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900,
        "30m": 1800, "1h": 3600, "4h": 14400,
    }
    interval = tf_seconds.get(timeframe, 900)

    print(f"\nMonitoring {symbol} ({timeframe}). Press Ctrl+C to stop.\n")

    try:
        while True:
            df = fetcher.fetch_ohlcv(symbol, timeframe, limit=200)
            if df.empty:
                logger.warning("No data received, retrying...")
                time.sleep(30)
                continue

            df = indicators.calculate_all(df)
            trendlines = detector.detect_trendlines(df)
            breakouts = detector.detect_breakout(df, trendlines) if trendlines else []

            last = df.iloc[-1]
            ts = last.get("timestamp", "")
            price = last["close"]

            status = (
                f"[{ts}] {symbol} = {price:.2f} | "
                f"Trendlines: {len(trendlines)} | "
                f"RSI: {last.get('rsi', 0):.1f} | "
                f"MACD: {last.get('macd_histogram', 0):.4f}"
            )

            if breakouts:
                for bo in breakouts:
                    status += (
                        f" | *** {bo.direction.value.upper()} BREAKOUT "
                        f"({bo.break_pct:.2f}%, "
                        f"vol={'YES' if bo.volume_confirmed else 'NO'}) ***"
                    )

            print(status)
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Naklon — Crypto Intraday Trendline Trading Strategy",
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to config.yaml",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="Run backtest")
    bt_parser.add_argument("--symbol", "-s", default=None, help="Trading pair")
    bt_parser.add_argument("--bars", "-b", type=int, default=1000, help="Number of bars")
    bt_parser.add_argument("--live-data", action="store_true", help="Use live exchange data")
    bt_parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data")
    bt_parser.add_argument("--base-price", type=float, default=60000.0, help="Base price for synthetic data")
    bt_parser.add_argument("--volatility", type=float, default=0.02, help="Volatility for synthetic data")
    bt_parser.add_argument("--output", "-o", default=None, help="Output JSON file path")

    # Analyze command
    az_parser = subparsers.add_parser("analyze", help="Analyze trendlines")
    az_parser.add_argument("--symbol", "-s", default=None, help="Trading pair")
    az_parser.add_argument("--bars", "-b", type=int, default=500, help="Number of bars")
    az_parser.add_argument("--live-data", action="store_true", help="Use live exchange data")
    az_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Live command
    lv_parser = subparsers.add_parser("live", help="Live monitoring")
    lv_parser.add_argument("--symbol", "-s", default=None, help="Trading pair")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    config = load_config(args.config)

    commands = {
        "backtest": cmd_backtest,
        "analyze": cmd_analyze,
        "live": cmd_live,
    }

    commands[args.command](config, args)


if __name__ == "__main__":
    main()
