#!/usr/bin/env python3
"""
Naklon — Разгон депозита по наклонкам

Стратегия интрадей-торговли криптовалют на 5м ТФ.
Депозит: $100-200, фьючерсы с плечом x10.

Команды:
  python main.py signal     — Показать текущий сигнал (ВХОД ПРЯМО СЕЙЧАС)
  python main.py scan       — Сканировать несколько пар
  python main.py backtest   — Прогнать на истории
  python main.py monitor    — Следить за сигналами в реальном времени
"""

import argparse
import json
import sys
from pathlib import Path

from naklon.backtest.backtester import Backtester
from naklon.data.fetcher import DataFetcher
from naklon.indicators.technical import TechnicalIndicators
from naklon.indicators.trendline import TrendlineDetector
from naklon.strategy.signals import SignalGenerator, SignalType
from naklon.strategy.risk import RiskManager
from naklon.utils.config import load_config
from naklon.utils.logger import setup_logger


def format_price(price: float) -> str:
    """Format price based on magnitude."""
    if price >= 1000:
        return f"{price:,.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    else:
        return f"{price:.6f}"


def print_signal_card(signal, config, equity=None):
    """Print a clear, actionable signal card with 3-level TP."""
    if equity is None:
        equity = config["capital"]["initial"]

    leverage = config.get("risk_management", {}).get("leverage", 10)
    risk_pct = config.get("risk_management", {}).get("max_risk_per_trade_pct", 3.0)
    exit_cfg = config.get("exit_rules", {})
    tp1_rr = exit_cfg.get("tp1_at_rr", 0.75)
    tp2_rr = exit_cfg.get("tp2_at_rr", 1.5)
    tp3_rr = exit_cfg.get("tp3_at_rr", 3.0)
    tp1_close = exit_cfg.get("tp1_close_pct", 40)
    tp2_close = exit_cfg.get("tp2_close_pct", 35)
    tp3_close = exit_cfg.get("tp3_close_pct", 25)

    entry = signal.entry_price
    sl = signal.stop_loss
    risk = abs(entry - sl)

    if signal.type == SignalType.LONG:
        direction = "LONG"
        arrow = "^"
        tp1 = entry + risk * tp1_rr
        tp2 = entry + risk * tp2_rr
        tp3 = entry + risk * tp3_rr
        sl_pct = (entry - sl) / entry * 100
        tp1_pct = (tp1 - entry) / entry * 100
        tp2_pct = (tp2 - entry) / entry * 100
        tp3_pct = (tp3 - entry) / entry * 100
    else:
        direction = "SHORT"
        arrow = "v"
        tp1 = entry - risk * tp1_rr
        tp2 = entry - risk * tp2_rr
        tp3 = entry - risk * tp3_rr
        sl_pct = (sl - entry) / entry * 100
        tp1_pct = (entry - tp1) / entry * 100
        tp2_pct = (entry - tp2) / entry * 100
        tp3_pct = (entry - tp3) / entry * 100

    # Position sizing
    risk_amount = equity * (risk_pct / 100)
    risk_per_unit = abs(entry - sl)
    quantity = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
    margin = (quantity * entry) / leverage
    notional = quantity * entry

    regime = getattr(signal, 'market_regime', 'unknown')
    regime_label = {"trending": "ТРЕНД", "ranging": "БОКОВИК", "squeeze": "СЖАТИЕ"}.get(
        regime, regime.upper())

    print()
    print(f"  {'='*56}")
    print(f"  {arrow}{arrow}{arrow}  {direction}  {signal.symbol}  x{leverage}  [{regime_label}]  {arrow}{arrow}{arrow}")
    print(f"  {'='*56}")
    print()
    print(f"  ВХОД:               {format_price(entry)}")
    print(f"  СТОП-ЛОСС:          {format_price(sl):>12s}   (-{sl_pct:.2f}% / -{sl_pct*leverage:.1f}% с плечом)")
    print(f"  TP1 ({tp1_close}%, R:{tp1_rr}):  {format_price(tp1):>12s}   (+{tp1_pct:.2f}% / +{tp1_pct*leverage:.1f}% с плечом)")
    print(f"  TP2 ({tp2_close}%, R:{tp2_rr}):  {format_price(tp2):>12s}   (+{tp2_pct:.2f}% / +{tp2_pct*leverage:.1f}% с плечом)")
    print(f"  TP3 ({tp3_close}%, R:{tp3_rr}):  {format_price(tp3):>12s}   (+{tp3_pct:.2f}% / +{tp3_pct*leverage:.1f}% с плечом)")
    print()
    print(f"  {'─'*56}")
    print(f"  Депозит:        ${equity:.2f}")
    print(f"  Маржа:          ${margin:.2f}")
    print(f"  Размер позиции: ${notional:.2f}")
    print(f"  Риск на сделку: ${risk_amount:.2f} ({risk_pct}%)")
    print()
    print(f"  При TP1 (R:{tp1_rr}):  +${risk_amount * tp1_rr:.2f}")
    print(f"  При TP2 (R:{tp2_rr}):  +${risk_amount * tp2_rr:.2f}")
    print(f"  При TP3 (R:{tp3_rr}):  +${risk_amount * tp3_rr:.2f}")
    print(f"  При SL:            -${risk_amount:.2f}")
    print()
    print(f"  {'─'*56}")
    print(f"  Сила:           {signal.strength.value.upper()} ({signal.score:.0f}/100)")
    print(f"  Режим рынка:    {regime_label}")
    print(f"  Наклонка:       {signal.breakout.trendline.type.value} | "
          f"касаний: {signal.breakout.trendline.touches} | "
          f"угол: {signal.breakout.trendline.angle_deg:.1f}")
    print(f"  Пробой:         {signal.breakout.break_pct:.2f}%")

    checks = signal.confirmations
    ok_count = sum(1 for v in checks.values() if v)
    confirms = []
    for name, ok in checks.items():
        label = name.replace("_ok", "").replace("_", " ").upper()
        confirms.append(f"{'[+]' if ok else '[-]'} {label}")
    print(f"  Подтверждения:  {ok_count}/{len(checks)} — {' | '.join(confirms)}")

    print()
    print(f"  ПЛАН ДЕЙСТВИЙ:")
    if signal.type == SignalType.LONG:
        print(f"  1. Открыть LONG {signal.symbol} по рынку")
    else:
        print(f"  1. Открыть SHORT {signal.symbol} по рынку")
    print(f"  2. Стоп-лосс:  {format_price(sl)}")
    print(f"  3. При {format_price(tp1)} — закрыть {tp1_close}%, SL в безубыток")
    print(f"  4. При {format_price(tp2)} — закрыть {tp2_close}%, SL подтянуть")
    print(f"  5. Остаток {tp3_close}% — TP на {format_price(tp3)} или трейлинг-стоп")
    print(f"  {'='*56}")
    print()


def cmd_signal(config: dict, args: argparse.Namespace):
    """Show current signal for a symbol — the main command."""
    logger = setup_logger("naklon", "WARNING")

    symbol = args.symbol or config["symbols"][0]

    if args.live_data:
        fetcher = DataFetcher(config)
        timeframe = config["timeframes"]["primary"]
        df = fetcher.fetch_ohlcv(symbol, timeframe, limit=200)
        if df.empty:
            print(f"\n  Не удалось загрузить данные для {symbol}")
            sys.exit(1)
    else:
        df = DataFetcher.generate_sample_data(
            bars=500, base_price=args.base_price,
            seed=args.seed, volatility=0.04,
        )

    signal_gen = SignalGenerator(config)
    signals = signal_gen.generate_signals(
        df, symbol, config["timeframes"]["primary"],
    )

    if not signals:
        last_close = df["close"].iloc[-1]
        indicators = TechnicalIndicators(config)
        df_ind = indicators.calculate_all(df)
        detector = TrendlineDetector(config)
        trendlines = detector.detect_trendlines(df)

        print()
        print(f"  {'='*52}")
        print(f"  НЕТ СИГНАЛА — {symbol}")
        print(f"  {'='*52}")
        print(f"  Цена:        {format_price(last_close)}")
        print(f"  Наклонок:    {len(trendlines)}")

        last = df_ind.iloc[-1]
        rsi = last.get("rsi", 0)
        trend = indicators.get_trend_bias(df_ind)
        regime_info = indicators.get_regime_info(df_ind)
        regime_label = {"trending": "ТРЕНД", "ranging": "БОКОВИК", "squeeze": "СЖАТИЕ"}.get(
            regime_info["regime"], "?")

        print(f"  RSI:         {rsi:.1f}")
        print(f"  Тренд:       {trend.upper()}")
        print(f"  ADX:         {regime_info['adx']:.1f}")
        print(f"  Режим рынка: {regime_label} {'(торгуем)' if regime_info['tradeable'] else '(ждём)'}")
        print(f"  MACD:        {last.get('macd_histogram', 0):.4f}")

        if trendlines:
            print(f"\n  Ближайшие наклонки:")
            for i, tl in enumerate(trendlines[:3], 1):
                tl_price = tl.price_at(len(df) - 1)
                dist = (last_close - tl_price) / last_close * 100
                print(f"    #{i} [{tl.type.value}] @ {format_price(tl_price)} "
                      f"({dist:+.2f}% от цены) | "
                      f"касаний: {tl.touches}")
            print(f"\n  Ждём пробоя наклонки для входа.")
        else:
            print(f"\n  Наклонки не сформированы. Ждём.")

        print(f"  {'='*52}")
        print()
        return

    # Show the best signal
    best = signals[0]
    equity = config["capital"]["initial"]
    print_signal_card(best, config, equity)

    # If there are more signals
    if len(signals) > 1:
        print(f"  + ещё {len(signals) - 1} сигнал(ов). Используй --symbol для другой пары.")


def cmd_scan(config: dict, args: argparse.Namespace):
    """Scan multiple symbols for signals."""
    logger = setup_logger("naklon", "WARNING")
    symbols = config["symbols"]

    print()
    print(f"  {'='*60}")
    print(f"  СКАНИРОВАНИЕ — {len(symbols)} пар | 5м ТФ | x{config['risk_management'].get('leverage', 10)}")
    print(f"  {'='*60}")

    found_signals = []

    for symbol in symbols:
        if args.live_data:
            fetcher = DataFetcher(config)
            df = fetcher.fetch_ohlcv(symbol, config["timeframes"]["primary"], limit=200)
            if df.empty:
                print(f"  {symbol}: ошибка загрузки")
                continue
        else:
            import hashlib
            seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16) % 10000
            base = {"BTC/USDT": 60000, "ETH/USDT": 3000, "SOL/USDT": 150,
                    "BNB/USDT": 600, "XRP/USDT": 0.6}.get(symbol, 1000)
            df = DataFetcher.generate_sample_data(
                bars=500, base_price=base, seed=seed, volatility=0.04,
            )

        signal_gen = SignalGenerator(config)
        signals = signal_gen.generate_signals(
            df, symbol, config["timeframes"]["primary"],
        )

        last_close = df["close"].iloc[-1]

        if signals:
            s = signals[0]
            direction = "LONG" if s.type == SignalType.LONG else "SHORT"
            found_signals.append((symbol, s))
            print(f"  {symbol:12s}  {format_price(last_close):>12s}  "
                  f">>> {direction:5s} | Score: {s.score:.0f} | "
                  f"{s.strength.value.upper()}")
        else:
            detector = TrendlineDetector(config)
            tls = detector.detect_trendlines(df)
            print(f"  {symbol:12s}  {format_price(last_close):>12s}  "
                  f"    нет сигнала | наклонок: {len(tls)}")

    print(f"  {'='*60}")

    if found_signals:
        print(f"\n  Найдено {len(found_signals)} сигнал(ов)!\n")
        for symbol, signal in found_signals:
            print_signal_card(signal, config, config["capital"]["initial"])
    else:
        print(f"\n  Сигналов нет. Ждём пробоев наклонок.")
    print()


def cmd_backtest(config: dict, args: argparse.Namespace):
    """Run backtest."""
    logger = setup_logger("naklon", config["logging"]["level"], config["logging"]["file"])

    symbol = args.symbol or config["symbols"][0]
    bars = args.bars
    leverage = config.get("risk_management", {}).get("leverage", 10)

    print()
    print(f"  {'='*52}")
    print(f"  БЭКТЕСТ — {symbol} | 5м | x{leverage}")
    print(f"  Депозит: ${config['capital']['initial']}")
    print(f"  {'='*52}")

    if args.live_data:
        fetcher = DataFetcher(config)
        df = fetcher.fetch_ohlcv(symbol, config["timeframes"]["primary"], limit=bars)
        if df.empty:
            print("  Ошибка загрузки данных")
            sys.exit(1)
    else:
        df = DataFetcher.generate_sample_data(
            bars=bars, base_price=args.base_price,
            seed=args.seed, volatility=0.04,
        )

    backtester = Backtester(config)
    result = backtester.run(df, symbol, config["timeframes"]["primary"])

    print(result.summary())

    # Show individual trades
    if result.trades:
        print(f"  Последние сделки:")
        print(f"  {'─'*60}")
        for t in result.trades[-10:]:
            side = t["side"].upper()
            pnl_key = "risk_amount"
            entry = t["entry_price"]
            tp1 = t.get("tp1", t.get("take_profit", 0))
            sl = t["stop_loss"]
            score = t["signal_score"]
            print(f"  {side:5s} @ {format_price(entry)} | "
                  f"SL: {format_price(sl)} | TP1: {format_price(tp1)} | "
                  f"Score: {score:.0f}")
        print()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "symbol": result.symbol,
                    "timeframe": result.timeframe,
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
        print(f"  Результаты сохранены: {args.output}")


def _scan_active_pairs(fetcher, top_n=20, min_vol=50_000_000):
    """Scan exchange for top volume USDT pairs."""
    print(f"\n  Сканирую биржу — ищу активные пары по объёму...")
    pairs = fetcher.fetch_top_volume_symbols(top_n=top_n, min_volume_usd=min_vol)

    if not pairs:
        print(f"  Не удалось получить тикеры с биржи.")
        return []

    print(f"\n  {'='*64}")
    print(f"  ТОП-{len(pairs)} ПАР ПО ОБЪЁМУ (24ч)")
    print(f"  {'='*64}")
    for i, p in enumerate(pairs, 1):
        vol_m = p["volume_usd"] / 1_000_000
        chg = p["change_pct"]
        chg_str = f"+{chg:.1f}%" if chg >= 0 else f"{chg:.1f}%"
        print(f"  {i:2d}. {p['symbol']:14s} {format_price(p['price']):>12s} | "
              f"Vol: ${vol_m:,.0f}M | {chg_str}")
    print(f"  {'='*64}\n")

    return [p["symbol"] for p in pairs]


def cmd_monitor(config: dict, args: argparse.Namespace):
    """Live monitoring mode — monitors active or specified symbols for signals."""
    logger = setup_logger("naklon", "WARNING")

    timeframe = config["timeframes"]["primary"]
    leverage = config.get("risk_management", {}).get("leverage", 10)

    import time

    tf_seconds = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900,
        "30m": 1800, "1h": 3600,
    }
    interval = tf_seconds.get(timeframe, 300)

    # For --top-volume, force live exchange (public data, no keys needed)
    if args.top_volume:
        live_config = {**config, "exchange": {**config.get("exchange", {}), "testnet": False}}
        fetcher = DataFetcher(live_config)
    else:
        fetcher = DataFetcher(config)

    # Determine which symbols to monitor
    if args.symbol:
        symbols = [args.symbol]
        mode = "single"
    elif args.top_volume:
        top_n = args.top_n
        min_vol = args.min_vol * 1_000_000
        symbols = _scan_active_pairs(fetcher, top_n=top_n, min_vol=min_vol)
        if not symbols:
            print("  Нет активных пар. Проверь подключение к бирже.")
            sys.exit(1)
        mode = "volume"
    else:
        symbols = config["symbols"]
        mode = "config"

    print()
    print(f"  {'='*64}")
    if mode == "single":
        print(f"  МОНИТОРИНГ — {symbols[0]} | {timeframe} | x{leverage}")
    elif mode == "volume":
        print(f"  МОНИТОРИНГ — ТОП {len(symbols)} АКТИВНЫХ ПАР | {timeframe} | x{leverage}")
    else:
        print(f"  МОНИТОРИНГ — {len(symbols)} пар | {timeframe} | x{leverage}")
        for sym in symbols:
            print(f"    - {sym}")
    print(f"  Обновление каждые {interval // 60} мин. Ctrl+C для выхода.")
    if mode == "volume":
        rescan_mins = args.rescan
        print(f"  Пересканирование объёмов каждые {rescan_mins} мин.")
    print(f"  {'='*64}")
    print()

    last_signal_bars = {sym: -1 for sym in symbols}
    last_rescan = time.time()
    rescan_interval = args.rescan * 60 if mode == "volume" else float("inf")

    try:
        while True:
            from datetime import datetime
            now = datetime.now().strftime("%H:%M:%S")

            # Rescan for active pairs periodically
            if mode == "volume" and (time.time() - last_rescan) >= rescan_interval:
                print(f"\n  [{now}] Пересканирование активных пар...\n")
                new_symbols = _scan_active_pairs(fetcher, top_n=top_n, min_vol=min_vol)
                if new_symbols:
                    added = set(new_symbols) - set(symbols)
                    removed = set(symbols) - set(new_symbols)
                    if added:
                        print(f"  + Новые: {', '.join(added)}")
                    if removed:
                        print(f"  - Ушли: {', '.join(removed)}")
                    symbols = new_symbols
                    for sym in symbols:
                        if sym not in last_signal_bars:
                            last_signal_bars[sym] = -1
                last_rescan = time.time()

            found_any = False

            for symbol in symbols:
                df = fetcher.fetch_ohlcv(symbol, timeframe, limit=200)
                if df.empty:
                    print(f"  [{now}] {symbol}: нет данных")
                    continue

                signal_gen = SignalGenerator(config)
                signals = signal_gen.generate_signals(df, symbol, timeframe)

                last = df.iloc[-1]
                ts = str(last.get("timestamp", ""))[:19]
                price = last["close"]

                if signals:
                    s = signals[0]
                    if s.bar_idx != last_signal_bars.get(symbol, -1):
                        last_signal_bars[symbol] = s.bar_idx
                        found_any = True
                        print(f"\n  !!! НОВЫЙ СИГНАЛ {symbol} в {ts} !!!\n")
                        print_signal_card(s, config, config["capital"]["initial"])
                else:
                    indicators = TechnicalIndicators(config)
                    df_ind = indicators.calculate_all(df)
                    r = df_ind.iloc[-1]
                    detector = TrendlineDetector(config)
                    tls = detector.detect_trendlines(df)
                    tl_str = f"Наклонок: {len(tls)}" if len(tls) > 0 else "Нет наклонок"
                    print(f"  [{now}] {symbol:14s} = {format_price(price)} | "
                          f"RSI: {r.get('rsi', 0):.0f} | "
                          f"{tl_str} | Ждём...")

            if len(symbols) > 1:
                print(f"  {'─'*64}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n  Мониторинг остановлен.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Naklon — Разгон депозита по наклонкам",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python main.py signal                           Сигнал на синтетике
  python main.py signal --symbol BTC/USDT --live  Сигнал по реальным данным
  python main.py scan --live                      Сканировать все пары
  python main.py backtest --bars 3000             Бэктест
  python main.py monitor                          Мониторинг 5 пар
  python main.py monitor --top-volume             Мониторинг ТОП пар по объёму
  python main.py monitor -V --top-n 10            ТОП-10 самых активных пар
        """,
    )
    parser.add_argument("--config", "-c", default=None, help="Путь к config.yaml")

    subparsers = parser.add_subparsers(dest="command", help="Команда")

    # Signal command
    sig = subparsers.add_parser("signal", help="Показать текущий сигнал")
    sig.add_argument("--symbol", "-s", default=None, help="Торговая пара")
    sig.add_argument("--live", "--live-data", dest="live_data", action="store_true",
                     help="Реальные данные с биржи")
    sig.add_argument("--seed", type=int, default=42, help="Seed для синтетики")
    sig.add_argument("--base-price", type=float, default=60000.0, help="Базовая цена")

    # Scan command
    sc = subparsers.add_parser("scan", help="Сканировать все пары")
    sc.add_argument("--live", "--live-data", dest="live_data", action="store_true",
                    help="Реальные данные с биржи")

    # Backtest command
    bt = subparsers.add_parser("backtest", help="Бэктест стратегии")
    bt.add_argument("--symbol", "-s", default=None, help="Торговая пара")
    bt.add_argument("--bars", "-b", type=int, default=2000, help="Количество баров")
    bt.add_argument("--live", "--live-data", dest="live_data", action="store_true",
                    help="Реальные данные")
    bt.add_argument("--seed", type=int, default=42, help="Seed")
    bt.add_argument("--base-price", type=float, default=60000.0, help="Базовая цена")
    bt.add_argument("--output", "-o", default=None, help="Сохранить в JSON")

    # Monitor command
    mo = subparsers.add_parser("monitor", help="Мониторинг сигналов",
                               formatter_class=argparse.RawDescriptionHelpFormatter,
                               epilog="""
Режимы:
  python main.py monitor                          5 пар из конфига
  python main.py monitor --top-volume             ТОП-20 пар по объёму (живые)
  python main.py monitor --top-volume --top-n 10  ТОП-10 пар по объёму
  python main.py monitor --symbol SOL/USDT        Одна пара
""")
    mo.add_argument("--symbol", "-s", default=None, help="Одна пара")
    mo.add_argument("--all", "-a", action="store_true", default=False,
                    help="Все пары из config.yaml")
    mo.add_argument("--top-volume", "-V", action="store_true", default=False,
                    help="Автопоиск активных пар по объёму на бирже")
    mo.add_argument("--top-n", type=int, default=20,
                    help="Сколько топ пар мониторить (по умолч. 20)")
    mo.add_argument("--min-vol", type=float, default=50,
                    help="Мин. объём в млн $ (по умолч. 50)")
    mo.add_argument("--rescan", type=int, default=30,
                    help="Пересканировать объёмы каждые N мин (по умолч. 30)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\n  Быстрый старт: python main.py signal\n")
        sys.exit(0)

    config = load_config(args.config)

    commands = {
        "signal": cmd_signal,
        "scan": cmd_scan,
        "backtest": cmd_backtest,
        "monitor": cmd_monitor,
    }

    commands[args.command](config, args)


if __name__ == "__main__":
    main()
