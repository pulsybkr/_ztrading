import numpy as np
import pandas as pd
from numba import njit
from core.types import Signal, Trade, Direction, TradeResult, BacktestConfig
from backtest.costs import CostModel


@njit
def resolve_trade_ticks(
    tick_prices: np.ndarray,
    tick_times: np.ndarray,
    entry_price: float,
    direction: int,
    sl_price: float,
    atr: float,
    breakeven_atr_mult: float,
    trailing_atr_mult: float,
    trailing_step: float,
) -> tuple:
    n = len(tick_prices)
    current_sl = sl_price
    be_activated = False
    n_trailing = 0
    max_favorable = 0.0
    max_adverse = 0.0

    for i in range(n):
        price = tick_prices[i]

        if direction == 1:
            pnl = price - entry_price
        else:
            pnl = entry_price - price

        if pnl > max_favorable:
            max_favorable = pnl
        if pnl < max_adverse:
            max_adverse = pnl

        if direction == 1 and price <= current_sl:
            return current_sl, i, max_favorable, max_adverse, be_activated, n_trailing
        elif direction == -1 and price >= current_sl:
            return current_sl, i, max_favorable, max_adverse, be_activated, n_trailing

        if not be_activated and pnl >= breakeven_atr_mult * atr:
            if direction == 1:
                current_sl = entry_price + 0.05
            else:
                current_sl = entry_price - 0.05
            be_activated = True

        if be_activated:
            if direction == 1:
                new_sl = price - trailing_atr_mult * atr
                if new_sl > current_sl + trailing_step:
                    current_sl = new_sl
                    n_trailing += 1
            else:
                new_sl = price + trailing_atr_mult * atr
                if new_sl < current_sl - trailing_step:
                    current_sl = new_sl
                    n_trailing += 1

    return tick_prices[-1], n - 1, max_favorable, max_adverse, be_activated, n_trailing


def resolve_with_ticks(signal: Signal, ticks: pd.DataFrame,
                       config: BacktestConfig, cost_model: CostModel) -> Trade:
    is_long = signal.direction == Direction.LONG

    if ticks is not None and "ask" in ticks.columns and "bid" in ticks.columns:
        entry_price = ticks.iloc[0]["ask"] if is_long else ticks.iloc[0]["bid"]
        prices = ticks["bid"].values if is_long else ticks["ask"].values
    else:
        entry_price = cost_model.adjust_entry_price(signal.price, is_long)
        prices = ticks["close"].values if ticks is not None else np.array([signal.price])

    if is_long:
        sl_price = entry_price - config.sl_atr_mult * signal.atr
    else:
        sl_price = entry_price + config.sl_atr_mult * signal.atr

    times = ticks["time"].astype(np.int64).values if ticks is not None else np.array([0])

    exit_price, exit_idx, mfe, mae, be_activated, n_trailing = resolve_trade_ticks(
        tick_prices=prices,
        tick_times=times,
        entry_price=entry_price,
        direction=1 if is_long else -1,
        sl_price=sl_price,
        atr=signal.atr,
        breakeven_atr_mult=config.breakeven_atr_mult,
        trailing_atr_mult=config.trailing_atr_mult,
        trailing_step=config.trailing_step_points * config.point_value,
    )

    if is_long:
        price_diff = exit_price - entry_price
    else:
        price_diff = entry_price - exit_price

    gross_pnl = price_diff * config.lot_size * config.contract_size

    costs = cost_model.entry_cost(config.lot_size, is_long)

    if gross_pnl > costs["total"]:
        result = TradeResult.WIN
    elif abs(price_diff) < 0.10:
        result = TradeResult.BREAKEVEN
    else:
        result = TradeResult.LOSS

    exit_time = ticks.iloc[exit_idx]["time"] if ticks is not None else signal.time

    return Trade(
        signal=signal,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_time=signal.time,
        exit_time=exit_time,
        direction=signal.direction,
        lot_size=config.lot_size,
        sl_initial=sl_price,
        result=result,
        gross_pnl=gross_pnl,
        commission=costs["commission"],
        spread_cost=costs["spread_cost"],
        slippage_cost=costs["slippage"],
        max_favorable=mfe * config.lot_size * config.contract_size,
        max_adverse=mae * config.lot_size * config.contract_size,
        breakeven_activated=be_activated,
        n_trailing_updates=n_trailing,
    )


def resolve_with_candles(signal: Signal, candles_m1: pd.DataFrame,
                         config: BacktestConfig, cost_model: CostModel) -> Trade:
    # Apply spread to simulate realistic bid/ask from OHLC candles.
    # LONG: enter at ask (price + spread), monitor sl/trailing against bid (price).
    # SHORT: enter at bid (price - spread), monitor sl/trailing against ask (price).
    sp = cost_model.spread
    pseudo_ticks = []
    for _, candle in candles_m1.iterrows():
        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
        t = candle["time"]
        if signal.direction == Direction.LONG:
            pseudo_ticks.extend([
                {"time": t, "bid": o,     "ask": o + sp},
                {"time": t, "bid": l,     "ask": l + sp},
                {"time": t, "bid": h,     "ask": h + sp},
                {"time": t, "bid": c,     "ask": c + sp},
            ])
        else:
            pseudo_ticks.extend([
                {"time": t, "bid": o - sp, "ask": o},
                {"time": t, "bid": h - sp, "ask": h},
                {"time": t, "bid": l - sp, "ask": l},
                {"time": t, "bid": c - sp, "ask": c},
            ])

    pseudo_df = pd.DataFrame(pseudo_ticks)
    return resolve_with_ticks(signal, pseudo_df, config, cost_model)
