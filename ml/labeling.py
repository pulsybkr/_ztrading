import pandas as pd
import numpy as np
from datetime import timedelta
from core.types import Signal, BacktestConfig


def triple_barrier_label(signal: Signal, candles: pd.DataFrame,
                         config: BacktestConfig,
                         tp_atr_mult: float = 3.0,
                         max_hours: int = 4) -> dict:
    """
    Apply Triple Barrier labeling to a signal.
    
    Returns:
        dict with:
            - label: 1 (win) or 0 (loss)
            - barrier: 'tp', 'sl', or 'timeout'
            - return: normalized return
            - hold_time: hours until exit
    """
    i = signal.candle_index
    entry_price = signal.price
    is_long = signal.direction.value == 1
    
    atr = signal.atr
    tp_distance = tp_atr_mult * atr
    sl_distance = config.sl_atr_mult * atr
    
    tp_price = entry_price + tp_distance if is_long else entry_price - tp_distance
    sl_price = entry_price - sl_distance if is_long else entry_price + sl_distance
    
    start_time = signal.time
    end_time = start_time + timedelta(hours=max_hours)
    
    window = candles[(candles["time"] > start_time) & (candles["time"] <= end_time)]
    
    if len(window) == 0:
        return {"label": 0, "barrier": "timeout", "return": 0.0, "hold_time": max_hours}
    
    for _, candle in window.iterrows():
        high, low = candle["high"], candle["low"]
        
        if is_long:
            if high >= tp_price:
                ret = (tp_price - entry_price) / entry_price
                hold = (candle["time"] - start_time).total_seconds() / 3600
                return {"label": 1, "barrier": "tp", "return": ret, "hold_time": hold}
            if low <= sl_price:
                ret = (sl_price - entry_price) / entry_price
                hold = (candle["time"] - start_time).total_seconds() / 3600
                return {"label": 0, "barrier": "sl", "return": ret, "hold_time": hold}
        else:
            if low <= tp_price:
                ret = (entry_price - tp_price) / entry_price
                hold = (candle["time"] - start_time).total_seconds() / 3600
                return {"label": 1, "barrier": "tp", "return": ret, "hold_time": hold}
            if high >= sl_price:
                ret = (entry_price - sl_price) / entry_price
                hold = (candle["time"] - start_time).total_seconds() / 3600
                return {"label": 0, "barrier": "sl", "return": ret, "hold_time": hold}
    
    final_price = window.iloc[-1]["close"]
    if is_long:
        ret = (final_price - entry_price) / entry_price
    else:
        ret = (entry_price - final_price) / entry_price
    
    hold = max_hours
    label = 1 if ret > 0 else 0
    
    return {"label": label, "barrier": "timeout", "return": ret, "hold_time": hold}


def label_signals(signals: list[Signal], candles: pd.DataFrame,
                config: BacktestConfig = None) -> pd.DataFrame:
    """
    Add labels to signals DataFrame.
    """
    if config is None:
        config = BacktestConfig()
    
    labels = []
    for signal in signals:
        label_data = triple_barrier_label(signal, candles, config)
        label_data["time"] = signal.time
        label_data["signal_idx"] = signal.candle_index
        labels.append(label_data)
    
    return pd.DataFrame(labels)
