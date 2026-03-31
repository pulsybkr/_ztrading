import MetaTrader5 as mt5
from core.types import Signal, Direction, BacktestConfig
from datetime import datetime
from typing import Optional


class MT5Bridge:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.connected = False
        self.positions = []

    def connect(self, login: int = None, password: str = None, server: str = None) -> bool:
        if not mt5.initialize():
            print(f"MT5 init failed: {mt5.last_error()}")
            return False

        if login and password and server:
            if not mt5.login(login, password=password, server=server):
                print(f"Login failed: {mt5.last_error()}")
                return False

        self.connected = True
        info = mt5.account_info()
        print(f"Connecte: {info.server} | Balance: ${info.balance:.2f}")
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
        print(f"Ordre execute: {dir_str} {lot} lots @ {price:.2f} | SL: {sl:.2f}")
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
            print(f"Close echoue: {result.retcode}")
            return False

        print(f"Position {ticket} fermee")
        return True

    def get_open_positions(self, symbol: str = None, magic: int = 240331) -> list:
        if symbol is None:
            symbol = self.config.symbol
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return []
        return [p for p in positions if p.magic == magic]

    def update_trailing_stops(self) -> int:
        updated = 0
        positions = self.get_open_positions()

        for pos in positions:
            tick = self.get_current_price(pos.symbol)
            if tick is None:
                continue

            if pos.type == mt5.POSITION_TYPE_BUY:
                current_pnl = (tick["bid"] - pos.price_open) * pos.volume
                if current_pnl > 0:
                    be_level = pos.price_open + self.config.breakeven_atr_mult * (pos.sl - pos.price_open)
                    new_sl = tick["bid"] - self.config.trailing_atr_mult * (pos.sl - pos.price_open)
                    if tick["bid"] > be_level and new_sl > pos.sl + 0.10:
                        if self._modify_sl(pos.ticket, new_sl):
                            updated += 1

            elif pos.type == mt5.POSITION_TYPE_SELL:
                current_pnl = (pos.price_open - tick["ask"]) * pos.volume
                if current_pnl > 0:
                    be_level = pos.price_open - self.config.breakeven_atr_mult * (pos.price_open - pos.sl)
                    new_sl = tick["ask"] + self.config.trailing_atr_mult * (pos.price_open - pos.sl)
                    if tick["ask"] < be_level and new_sl < pos.sl - 0.10:
                        if self._modify_sl(pos.ticket, new_sl):
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
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  SL mis a jour -> {new_sl:.2f}")
            return True
        return False

    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("MT5 deconnecte")

    def get_account_info(self) -> dict:
        info = mt5.account_info()
        return {
            "balance": info.balance,
            "equity": info.equity,
            "profit": info.profit,
            "margin": info.margin,
            "freemargin": info.margin_free,
        }
