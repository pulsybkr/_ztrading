import MetaTrader5 as mt5
import pandas as pd
from core.types import Signal, Direction, BacktestConfig
from datetime import datetime
from typing import Optional
from rich.console import Console

console = Console()

MT5_TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


class MT5Bridge:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.connected = False
        self.positions = []

    def connect(self, login: int = None, password: str = None, server: str = None) -> bool:
        # Fall back to .env if credentials not provided
        if not login or not password or not server:
            try:
                from dotenv import load_dotenv
                import os
                load_dotenv()
                login = login or (int(os.getenv("MT5_LOGIN")) if os.getenv("MT5_LOGIN") else None)
                password = password or os.getenv("MT5_PASSWORD")
                server = server or os.getenv("MT5_SERVER")
            except Exception:
                pass

        if not mt5.initialize():
            print(f"MT5 init failed: {mt5.last_error()}")
            return False

        if login and password and server:
            if not mt5.login(login, password=password, server=server):
                print(f"Login failed: {mt5.last_error()}")
                return False

        self.connected = True
        info = mt5.account_info()
        console.print(f"[green]✓ Connecté[/green] {info.server} | {info.name} | Balance: [bold]{info.currency} {info.balance:.2f}[/bold]")
        return True

    def is_connected(self) -> bool:
        return self.connected and mt5.terminal_info() is not None

    def get_current_price(self, symbol: str = None) -> Optional[dict]:
        if symbol is None:
            symbol = self.config.symbol
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {"bid": tick.bid, "ask": tick.ask, "time": tick.time}

    def execute_signal(self, signal: Signal, lot: float = None) -> Optional[int]:
        if not self.connected:
            return None

        if lot is None:
            lot = self.config.lot_size

        symbol = self.config.symbol
        tick = self.get_current_price(symbol)
        if tick is None:
            print(f"Pas de cotation pour {symbol}")
            return None

        if signal.direction == Direction.LONG:
            price = tick["ask"]
            sl = price - self.config.sl_atr_mult * signal.atr
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick["bid"]
            sl = price + self.config.sl_atr_mult * signal.atr
            order_type = mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": round(sl, 2),
            "tp": 0.0,
            "deviation": 10,
            "magic": 240331,
            "comment": f"NovaReborn_{signal.session.value}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Ordre echoue: {result.retcode} - {result.comment}")
            return None

        dir_str = "LONG" if signal.direction == Direction.LONG else "SHORT"
        color = "green" if signal.direction == Direction.LONG else "red"
        console.print(f"[{color}]▶ {dir_str}[/{color}] {lot} lots @ [bold]{price:.2f}[/bold] | SL: {sl:.2f} | Ticket: {result.order}")
        return result.order

    def close_position(self, ticket: int, lot: float = None) -> bool:
        if lot is None:
            lot = self.config.lot_size

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False

        pos = positions[0]
        symbol = pos.symbol
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY

        tick = self.get_current_price(symbol)
        if tick is None:
            return False

        price = tick["bid"] if order_type == mt5.ORDER_TYPE_SELL else tick["ask"]

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 240331,
            "comment": "NovaReborn_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            console.print(f"[red]✗ Close échoué: {result.retcode}[/red]")
            return False

        console.print(f"[cyan]■ Position {ticket} fermée[/cyan]")
        return True

    def get_open_positions(self, symbol: str = None, magic: int = 240331) -> list:
        if symbol is None:
            symbol = self.config.symbol
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return []
        return [p for p in positions if p.magic == magic]

    def _get_current_atr(self, symbol: str) -> float:
        """Compute current ATR from recent candles (same method as backtest engine)."""
        period = self.config.keltner_atr_period
        tf = getattr(self.config, "signal_timeframe", "M5")
        df = self.get_candles(symbol, tf, count=period + 5)
        if df.empty or len(df) < period:
            return 0.0
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if pd.notna(atr) else 0.0

    def update_trailing_stops(self, db=None) -> int:
        updated = 0
        positions = self.get_open_positions()

        for pos in positions:
            tick = self.get_current_price(pos.symbol)
            if tick is None:
                continue

            cur_atr = self._get_current_atr(pos.symbol)
            if cur_atr <= 0:
                continue

            be_dist    = self.config.breakeven_atr_mult * cur_atr
            trail_dist = self.config.trailing_atr_mult  * cur_atr
            old_sl     = pos.sl
            event_type = "trailing_update"

            if pos.type == mt5.POSITION_TYPE_BUY:
                pnl = tick["bid"] - pos.price_open
                if pnl >= be_dist:
                    new_sl = tick["bid"] - trail_dist
                    if new_sl < pos.price_open:
                        new_sl = pos.price_open
                        event_type = "breakeven"
                    if new_sl > old_sl + 0.10 and new_sl < tick["bid"]:
                        if self._modify_sl(pos.ticket, new_sl):
                            unrealized = round(pnl * self.config.lot_size * self.config.contract_size, 2)
                            label = "[yellow]⇡ BE[/yellow]" if event_type == "breakeven" else "[cyan]⇡ Trail[/cyan]"
                            console.print(f"  {label} SL {old_sl:.2f} → [bold]{new_sl:.2f}[/bold] | Prix={tick['bid']:.2f} | ATR={cur_atr:.3f} | PnL=[green]+${unrealized}[/green]")
                            if db:
                                db.log_sl_event(event_type, old_sl, new_sl, tick["bid"], tick["ask"], cur_atr, unrealized)
                            updated += 1

            elif pos.type == mt5.POSITION_TYPE_SELL:
                pnl = pos.price_open - tick["ask"]
                if pnl >= be_dist:
                    new_sl = tick["ask"] + trail_dist
                    if new_sl > pos.price_open or new_sl == 0:
                        new_sl = pos.price_open
                        event_type = "breakeven"
                    if (new_sl < old_sl - 0.10 or old_sl == 0) and new_sl > tick["ask"]:
                        if self._modify_sl(pos.ticket, new_sl):
                            unrealized = round(pnl * self.config.lot_size * self.config.contract_size, 2)
                            label = "[yellow]⇣ BE[/yellow]" if event_type == "breakeven" else "[cyan]⇣ Trail[/cyan]"
                            console.print(f"  {label} SL {old_sl:.2f} → [bold]{new_sl:.2f}[/bold] | Prix={tick['ask']:.2f} | ATR={cur_atr:.3f} | PnL=[green]+${unrealized}[/green]")
                            if db:
                                db.log_sl_event(event_type, old_sl, new_sl, tick["bid"], tick["ask"], cur_atr, unrealized)
                            updated += 1

        return updated

    def _modify_sl(self, ticket: int, new_sl: float) -> bool:
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": round(new_sl, 2),
            "tp": 0.0,
        }
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE

    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("MT5 deconnecte")

    def get_candles(self, symbol: str = None, timeframe: str = "M5", count: int = 100) -> pd.DataFrame:
        """Fetch recent candles directly from MT5 (for live use)."""
        if symbol is None:
            symbol = self.config.symbol
        tf = MT5_TIMEFRAMES.get(timeframe.upper(), mt5.TIMEFRAME_M5)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"tick_volume": "tick_volume"})
        return df

    def get_account_info(self) -> dict:
        info = mt5.account_info()
        return {
            "login":     info.login,
            "server":    info.server,
            "name":      info.name,
            "currency":  info.currency,
            "leverage":  info.leverage,
            "balance":   info.balance,
            "equity":    info.equity,
            "profit":    info.profit,
            "margin":    info.margin,
            "freemargin": info.margin_free,
        }
