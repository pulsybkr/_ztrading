import pandas as pd
import numpy as np
from core.types import Signal, SessionType


FEATURE_COLUMNS = [
    "tick_volume_ratio",
    "atr_ratio_mtf",
    "keltner_distance",
    "rsi_14",
    "momentum_5",
    "session_id",
    "ema_slope_h1",
    "vol_ratio_20",
]


def build_features(signal: Signal, candles_m5: pd.DataFrame,
                   candles_h1: pd.DataFrame = None) -> dict:
    i = signal.candle_index
    row = candles_m5.iloc[i]

    features = {}

    if "tick_volume" in candles_m5.columns:
        vol_window = candles_m5.iloc[max(0, i-20):i]["tick_volume"]
        if len(vol_window) > 0 and vol_window.mean() > 0:
            features["tick_volume_ratio"] = row["tick_volume"] / vol_window.mean()
        else:
            features["tick_volume_ratio"] = 1.0
    else:
        features["tick_volume_ratio"] = 1.0

    if signal.atr_ratio > 0:
        features["atr_ratio_mtf"] = signal.atr_ratio
    elif candles_h1 is not None and "atr" in candles_h1.columns:
        h1_mask = candles_h1["time"] <= row["time"]
        if h1_mask.sum() > 0:
            atr_h1 = candles_h1.loc[h1_mask].iloc[-1]["atr"]
            features["atr_ratio_mtf"] = row["atr"] / atr_h1 if atr_h1 > 0 else 0
        else:
            features["atr_ratio_mtf"] = 0
    else:
        features["atr_ratio_mtf"] = 0

    if signal.direction.value == 1:
        features["keltner_distance"] = (row["close"] - row["kc_upper"]) / row["atr"]
    else:
        features["keltner_distance"] = (row["kc_lower"] - row["close"]) / row["atr"]

    closes = candles_m5.iloc[max(0, i-14):i+1]["close"]
    if len(closes) >= 14:
        delta = closes.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        features["rsi_14"] = 100 - (100 / (1 + gain / loss)) if loss > 0 else 50
    else:
        features["rsi_14"] = 50

    if i >= 5:
        price_5_ago = candles_m5.iloc[i - 5]["close"]
        features["momentum_5"] = (row["close"] - price_5_ago) / row["atr"]
    else:
        features["momentum_5"] = 0

    session_map = {
        SessionType.SGE_OPEN: 0,
        SessionType.LONDON: 1,
        SessionType.OVERLAP: 2,
        SessionType.OFF_SESSION: 3,
    }
    features["session_id"] = session_map.get(signal.session, 3)

    if candles_h1 is not None and "kc_mid" in candles_h1.columns:
        h1_recent = candles_h1[candles_h1["time"] <= row["time"]].tail(5)
        if len(h1_recent) >= 2:
            ema_values = h1_recent["kc_mid"].values
            slope = (ema_values[-1] - ema_values[0]) / len(ema_values)
            atr_h1 = h1_recent.iloc[-1]["atr"]
            features["ema_slope_h1"] = slope / atr_h1 if atr_h1 > 0 else 0
        else:
            features["ema_slope_h1"] = 0
    else:
        features["ema_slope_h1"] = 0

    if i >= 20:
        window = candles_m5.iloc[i-20:i+1]["close"]
        features["vol_ratio_20"] = window.std() / window.mean() if window.mean() > 0 else 0
    else:
        features["vol_ratio_20"] = 0

    return features


def signals_to_dataframe(signals: list[Signal], candles_m5: pd.DataFrame,
                        candles_h1: pd.DataFrame = None) -> pd.DataFrame:
    rows = []
    for signal in signals:
        features = build_features(signal, candles_m5, candles_h1)
        features["time"] = signal.time
        features["direction"] = signal.direction.value
        features["signal_idx"] = signal.candle_index
        rows.append(features)

    df = pd.DataFrame(rows)
    df = df.fillna(0)
    return df
