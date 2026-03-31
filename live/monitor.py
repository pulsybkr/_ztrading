from datetime import datetime
from typing import Optional
import pandas as pd


class Monitor:
    def __init__(self):
        self.trades_log = []
        self.positions_log = []

    def log_trade(self, trade_data: dict):
        self.trades_log.append({
            "time": datetime.now(),
            **trade_data
        })

    def log_position_update(self, position_data: dict):
        self.positions_log.append({
            "time": datetime.now(),
            **position_data
        })

    def get_summary(self) -> dict:
        if not self.trades_log:
            return {"total_trades": 0, "pnl_total": 0.0}

        pnls = [t.get("pnl", 0) for t in self.trades_log]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            "total_trades": len(self.trades_log),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "winrate": len(wins) / len(pnls) * 100 if pnls else 0,
            "pnl_total": sum(pnls),
            "avg_win": sum(wins) / len(wins) if wins else 0,
            "avg_loss": sum(losses) / len(losses) if losses else 0,
        }

    def print_summary(self):
        summary = self.get_summary()
        print("\n=== Trading Summary ===")
        print(f"Total trades: {summary['total_trades']}")
        print(f"Winrate: {summary['winrate']:.1f}%")
        print(f"P/L Total: ${summary['pnl_total']:.2f}")
        if summary['total_trades'] > 0:
            print(f"Avg Win: ${summary['avg_win']:.2f}")
            print(f"Avg Loss: ${summary['avg_loss']:.2f}")


def send_telegram_message(message: str, token: str = None, chat_id: str = None):
    if not token or not chat_id:
        return False

    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {"chat_id": chat_id, "text": message}
        resp = requests.post(url, data=data, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def send_discord_webhook(message: str, webhook_url: str = None):
    if not webhook_url:
        return False

    try:
        import requests
        data = {"content": message}
        resp = requests.post(webhook_url, json=data, timeout=10)
        return resp.status_code in [200, 204]
    except Exception:
        return False


class AlertManager:
    def __init__(self, telegram_token: str = None, telegram_chat_id: str = None,
                 discord_webhook: str = None):
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.discord_webhook = discord_webhook

    def alert_trade(self, direction: str, price: float, sl: float, pnl: float = None):
        msg = f"NovaGold Signal: {direction} @ {price:.2f} | SL: {sl:.2f}"
        if pnl is not None:
            msg += f" | P/L: ${pnl:.2f}"

        self._send_alert(msg)

    def alert_position_closed(self, ticket: int, pnl: float):
        msg = f"Position {ticket} fermee | P/L: ${pnl:.2f}"
        self._send_alert(msg)

    def alert_error(self, error_msg: str):
        msg = f"NovaGold Error: {error_msg}"
        self._send_alert(msg)

    def _send_alert(self, message: str):
        if self.discord_webhook:
            send_discord_webhook(message, self.discord_webhook)
        if self.telegram_token:
            send_telegram_message(message, self.telegram_token, self.telegram_chat_id)
