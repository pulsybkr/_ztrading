from core.types import Signal, BacktestConfig, SessionType
from core.constants import SESSIONS
import pandas as pd


def apply_regime_filters(signals: list[Signal], config: BacktestConfig) -> list[Signal]:
    filtered = signals

    if config.use_session_filter:
        allowed = [SessionType(s) for s in config.allowed_sessions]
        filtered = [s for s in filtered if s.session in allowed]

    if config.use_atr_ratio_filter:
        # Only filter if atr_ratio was actually computed (H1 data exists)
        filtered = [
            s for s in filtered
            if s.atr_ratio > 0 and s.atr_ratio >= config.atr_ratio_threshold
        ]

    return filtered


def compute_atr_ratio(candles_m5_df: "pd.DataFrame", candles_h1_df: "pd.DataFrame | None", signal: Signal) -> float:
    if candles_h1_df is None or len(candles_h1_df) == 0:
        return 1.0

    m5_time = signal.time
    h1_mask = candles_h1_df["time"] <= m5_time
    if h1_mask.sum() == 0:
        return 1.0

    atr_m5 = signal.atr
    atr_h1 = candles_h1_df.loc[h1_mask].iloc[-1]["atr"]

    if atr_h1 == 0:
        return 0.0

    return atr_m5 / atr_h1


def attach_atr_ratios(signals: list[Signal], candles_m5_df: "pd.DataFrame", 
                      candles_h1_df: "pd.DataFrame | None" = None) -> list[Signal]:
    for signal in signals:
        signal.atr_ratio = compute_atr_ratio(candles_m5_df, candles_h1_df, signal)
    return signals
