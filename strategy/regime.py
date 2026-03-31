from core.types import Signal, BacktestConfig, SessionType
from core.constants import SESSIONS


def apply_regime_filters(signals: list[Signal], config: BacktestConfig) -> list[Signal]:
    filtered = signals

    if config.use_session_filter:
        allowed = [SessionType(s) for s in config.allowed_sessions]
        filtered = [s for s in filtered if s.session in allowed]

    if config.use_atr_ratio_filter:
        filtered = [
            s for s in filtered
            if hasattr(s, "atr_ratio") and s.atr_ratio >= config.atr_ratio_threshold
        ]

    return filtered


def compute_atr_ratio(candles_m5, candles_h1, index: int) -> float:
    if candles_h1 is None or len(candles_h1) == 0:
        return 1.0

    m5_time = candles_m5.iloc[index]["time"]
    h1_mask = candles_h1["time"] <= m5_time
    if h1_mask.sum() == 0:
        return 1.0

    atr_m5 = candles_m5.iloc[index]["atr"]
    atr_h1 = candles_h1.loc[h1_mask].iloc[-1]["atr"]

    if atr_h1 == 0:
        return 0.0

    return atr_m5 / atr_h1
