import pandas as pd
import numpy as np
from datetime import timedelta
from core.types import Signal, BacktestConfig
from backtest.resolver import resolve_with_ticks
from backtest.costs import CostModel


def triple_barrier_label(signal: Signal, candles: pd.DataFrame,
                         config: BacktestConfig,
                         ticks: pd.DataFrame = None,
                         max_hours: int = 4) -> dict:
    """
    Apply Triple Barrier labeling with realistic exit logic (trailing + breakeven).

    Simule la MÊME logique de sortie que le backtest :
    1. SL initial à sl_atr_mult × ATR
    2. Breakeven activé à breakeven_atr_mult × ATR de profit
    3. Trailing stop qui suit à trailing_atr_mult × ATR

    Utilise les ticks si disponibles (résolution tick-by-tick), sinon fallback sur bougies.

    Returns:
        dict with:
            - label: 1 (win, net P/L > 0) or 0 (loss or timeout)
            - barrier: 'trailing', 'sl', 'breakeven', or 'timeout'
            - return: normalized return
            - hold_time: hours until exit
            - exit_price: prix de sortie réel
            - gross_pnl: P/L brut avant coûts
            - net_pnl: P/L net après coûts
    """
    is_long = signal.direction.value == 1

    start_time = signal.time
    end_time = start_time + timedelta(hours=max_hours)

    # Entry at open of first candle — realistic next-bar execution
    entry_window = candles[candles["time"] >= start_time]
    if len(entry_window) == 0:
        return {"label": 0, "barrier": "timeout", "return": 0.0, "hold_time": max_hours,
                "exit_price": signal.price, "gross_pnl": 0.0, "net_pnl": 0.0}

    entry_price = entry_window.iloc[0]["open"]

    # Si ticks disponibles, utiliser resolve_with_ticks (logique exacte du backtest)
    if ticks is not None and len(ticks) > 0:
        # Filtrer les ticks sur la fenêtre de temps
        tick_window = ticks[(ticks["time"] >= start_time) & (ticks["time"] <= end_time)]

        if len(tick_window) > 10:
            # Utiliser le même resolver que le backtest
            cost_model = CostModel(spread=config.spread, commission_per_lot=config.commission_per_lot)
            trade = resolve_with_ticks(signal, tick_window, config, cost_model)

            # Label = 1 si le trade est gagnant (net_pnl > 0)
            label = 1 if trade.net_pnl > 0 else 0

            # Déterminer la barrière touchée
            if trade.breakeven_activated and trade.n_trailing_updates > 0:
                barrier = "trailing"
            elif trade.breakeven_activated:
                barrier = "breakeven"
            elif trade.exit_price <= trade.sl_initial + 0.01:  # Proche du SL initial
                barrier = "sl"
            else:
                barrier = "trailing"

            hold_time = (trade.exit_time - start_time).total_seconds() / 3600
            ret = (trade.exit_price - entry_price) / entry_price

            return {
                "label": label,
                "barrier": barrier,
                "return": ret,
                "hold_time": hold_time,
                "exit_price": trade.exit_price,
                "gross_pnl": trade.gross_pnl,
                "net_pnl": trade.net_pnl,
            }

    # Fallback: résolution sur bougies (moins précis mais fonctionne sans ticks)
    window = candles[(candles["time"] >= start_time) & (candles["time"] <= end_time)]

    if len(window) == 0:
        return {"label": 0, "barrier": "timeout", "return": 0.0, "hold_time": max_hours,
                "exit_price": entry_price, "gross_pnl": 0.0, "net_pnl": 0.0}

    # Initialiser les variables de trailing
    atr = signal.atr
    sl_distance = config.sl_atr_mult * atr
    breakeven_dist = config.breakeven_atr_mult * atr
    trailing_dist = config.trailing_atr_mult * atr

    # SL initial
    if is_long:
        current_sl = entry_price - sl_distance
    else:
        current_sl = entry_price + sl_distance

    be_activated = False
    exit_price = None
    exit_candle = None

    for _, candle in window.iterrows():
        high, low, close = candle["high"], candle["low"], candle["close"]

        if is_long:
            # Calculer le P/L actuel
            pnl = close - entry_price

            # Activer breakeven si profit suffisant
            if not be_activated and pnl >= breakeven_dist:
                current_sl = entry_price + 0.05  # SL au breakeven + spread
                be_activated = True

            # Mettre à jour le trailing stop
            if be_activated:
                trailing_sl = close - trailing_dist
                if trailing_sl > current_sl:
                    current_sl = trailing_sl

            # Vérifier si le SL (ou trailing) est touché
            if low <= current_sl:
                exit_price = current_sl
                exit_candle = candle
                break
        else:
            # SHORT
            pnl = entry_price - close

            if not be_activated and pnl >= breakeven_dist:
                current_sl = entry_price - 0.05
                be_activated = True

            if be_activated:
                trailing_sl = close + trailing_dist
                if trailing_sl < current_sl:
                    current_sl = trailing_sl

            if high >= current_sl:
                exit_price = current_sl
                exit_candle = candle
                break

    # Timeout
    if exit_price is None:
        return {"label": 0, "barrier": "timeout", "return": 0.0, "hold_time": max_hours,
                "exit_price": entry_price, "gross_pnl": 0.0, "net_pnl": 0.0}

    # Calculer le P/L
    if is_long:
        gross_pnl = (exit_price - entry_price) * config.lot_size * config.contract_size
    else:
        gross_pnl = (entry_price - exit_price) * config.lot_size * config.contract_size

    # Coûts approximatifs (spread + commission)
    spread_cost = config.spread * config.lot_size * config.contract_size
    commission = config.commission_per_lot * config.lot_size
    net_pnl = gross_pnl - spread_cost - commission

    # Label = 1 si net_pnl > 0
    label = 1 if net_pnl > 0 else 0

    # Déterminer la barrière
    if be_activated and abs(exit_price - entry_price) < 0.1:
        barrier = "breakeven"
    elif be_activated:
        barrier = "trailing"
    else:
        barrier = "sl"

    hold_time = (exit_candle["time"] - start_time).total_seconds() / 3600
    ret = (exit_price - entry_price) / entry_price

    return {
        "label": label,
        "barrier": barrier,
        "return": ret,
        "hold_time": hold_time,
        "exit_price": exit_price,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
    }


def label_signals(signals: list[Signal], candles: pd.DataFrame,
                  config: BacktestConfig = None,
                  ticks: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add labels to signals DataFrame.

    Args:
        signals: Liste des signaux à labelliser
        candles: Bougies pour la résolution (fallback si pas de ticks)
        config: Configuration du backtest
        ticks: Données tick-by-tick pour résolution précise (optionnel)
    """
    if config is None:
        config = BacktestConfig()

    labels = []
    for signal in signals:
        # Filtrer les ticks pour la fenêtre de ce signal
        tick_window = None
        if ticks is not None:
            end_time = signal.time + pd.Timedelta(hours=4)
            tick_window = ticks[(ticks["time"] >= signal.time) & (ticks["time"] <= end_time)]

        label_data = triple_barrier_label(signal, candles, config, tick_window)
        label_data["time"] = signal.time
        label_data["signal_idx"] = signal.candle_index
        labels.append(label_data)

    return pd.DataFrame(labels)
