"""
Telegram Notifier — отправка сигналов в Telegram.

Использует Telegram Bot API напрямую через urllib (без зависимостей).

Настройка:
  1. Создай бота через @BotFather -> получишь токен
  2. Напиши боту /start
  3. Узнай свой chat_id через @userinfobot или @getmyid_bot
  4. Пропиши в config.yaml:
     telegram:
       bot_token: "123456:ABC-DEF..."
       chat_id: "987654321"
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Any

logger = logging.getLogger("naklon.telegram")


class TelegramNotifier:
    """Sends trade signals to Telegram via Bot API."""

    API_URL = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, config: dict[str, Any]):
        tg_cfg = config.get("telegram", {})
        self.bot_token = tg_cfg.get("bot_token", "")
        self.chat_id = str(tg_cfg.get("chat_id", ""))
        self.enabled = tg_cfg.get("enabled", True) and bool(self.bot_token) and bool(self.chat_id)

        if not self.enabled:
            logger.warning(
                "Telegram disabled — set telegram.bot_token and telegram.chat_id in config.yaml"
            )

    def _api_call(self, method: str, payload: dict) -> dict | None:
        """Make a Telegram Bot API call."""
        url = self.API_URL.format(token=self.bot_token, method=method)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            logger.error("Telegram API error %d: %s", e.code, body)
            return None
        except Exception as e:
            logger.error("Telegram send failed: %s", e)
            return None

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a text message to the configured chat."""
        if not self.enabled:
            return False

        result = self._api_call("sendMessage", {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        })
        if result and result.get("ok"):
            logger.debug("Telegram message sent")
            return True
        return False

    def send_signal(self, signal, config: dict) -> bool:
        """Format and send a TradeSignal to Telegram."""
        if not self.enabled:
            return False

        text = self._format_signal(signal, config)
        return self.send_message(text)

    def send_startup(self, symbols: list[str], mode: str, timeframe: str) -> bool:
        """Send a startup notification."""
        if not self.enabled:
            return False

        if mode == "volume":
            header = f"<b>NAKLON запущен</b> — ТОП {len(symbols)} пар по объёму"
        elif mode == "single":
            header = f"<b>NAKLON запущен</b> — {symbols[0]}"
        else:
            header = f"<b>NAKLON запущен</b> — {len(symbols)} пар"

        lines = [header, f"ТФ: {timeframe}", ""]
        for sym in symbols[:30]:
            lines.append(f"  {sym}")
        lines.append("")
        lines.append("Жду пробоев наклонок...")

        return self.send_message("\n".join(lines))

    def send_no_signals_summary(self, statuses: list[dict]) -> bool:
        """Send periodic summary when no signals found (optional)."""
        if not self.enabled:
            return False

        lines = ["<b>Статус мониторинга</b>", ""]
        for s in statuses[:20]:
            tl_count = s.get("trendlines", 0)
            rsi = s.get("rsi", 0)
            icon = "~" if tl_count > 0 else "-"
            lines.append(f"{icon} {s['symbol']:14s} RSI:{rsi:.0f} | наклонок: {tl_count}")
        lines.append("")
        lines.append("Сигналов нет. Мониторинг продолжается.")

        return self.send_message("\n".join(lines))

    @staticmethod
    def _format_signal(signal, config: dict) -> str:
        """Format a TradeSignal into a Telegram message."""
        from naklon.strategy.signals import SignalType

        leverage = config.get("risk_management", {}).get("leverage", 10)
        equity = config["capital"]["initial"]
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
        else:
            direction = "SHORT"
            arrow = "v"
            tp1 = entry - risk * tp1_rr
            tp2 = entry - risk * tp2_rr
            tp3 = entry - risk * tp3_rr
            sl_pct = (sl - entry) / entry * 100

        risk_amount = equity * (risk_pct / 100)

        regime = getattr(signal, "market_regime", "unknown")
        regime_label = {
            "trending": "ТРЕНД", "ranging": "БОКОВИК", "squeeze": "СЖАТИЕ",
        }.get(regime, regime.upper())

        # Confirmations
        checks = signal.confirmations
        ok_count = sum(1 for v in checks.values() if v)
        confirms_str = " ".join(
            f"{'[+]' if v else '[-]'}{k.replace('_ok', '').upper()}"
            for k, v in checks.items()
        )

        def fp(price):
            if price >= 1000:
                return f"{price:,.2f}"
            elif price >= 1:
                return f"{price:.4f}"
            else:
                return f"{price:.6f}"

        lines = [
            f"{arrow}{arrow}{arrow} <b>{direction} {signal.symbol} x{leverage}</b> [{regime_label}]",
            "",
            f"<b>Вход:</b>    {fp(entry)}",
            f"<b>SL:</b>      {fp(sl)}  (-{sl_pct:.2f}% / -{sl_pct*leverage:.1f}% с плечом)",
            f"<b>TP1</b> ({tp1_close}%, R:{tp1_rr}):  {fp(tp1)}",
            f"<b>TP2</b> ({tp2_close}%, R:{tp2_rr}):  {fp(tp2)}",
            f"<b>TP3</b> ({tp3_close}%, R:{tp3_rr}):  {fp(tp3)}",
            "",
            f"Риск: ${risk_amount:.2f} ({risk_pct}%)",
            f"Score: {signal.score:.0f}/100 | {signal.strength.value.upper()}",
            f"Наклонка: {signal.breakout.trendline.type.value} | "
            f"касаний: {signal.breakout.trendline.touches} | "
            f"пробой: {signal.breakout.break_pct:.2f}%",
            f"Подтверждения: {ok_count}/{len(checks)} {confirms_str}",
            "",
            f"<b>ПЛАН:</b>",
            f"1. {direction} {signal.symbol} по рынку",
            f"2. SL: {fp(sl)}",
            f"3. TP1 {fp(tp1)} -> закрыть {tp1_close}%, SL в б/у",
            f"4. TP2 {fp(tp2)} -> закрыть {tp2_close}%",
            f"5. TP3 {fp(tp3)} -> остаток {tp3_close}%",
        ]

        return "\n".join(lines)
