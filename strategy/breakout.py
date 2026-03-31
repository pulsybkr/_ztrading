import pandas as pd
from core.types import Signal, Direction, SessionType
from core.constants import SESSIONS
from strategy.keltner import compute_keltner


def detect_session(t: pd.Timestamp) -> SessionType:
    current_time = t.time()
    for session_name, (start, end) in SESSIONS.items():
        if start <= current_time <= end:
            return SessionType(session_name)
    return SessionType.OFF_SESSION


def detect_breakouts(df: pd.DataFrame,
                     ema_period: int = 20,
                     atr_period: int = 14,
                     multiplier: float = 2.0) -> list[Signal]:
    df = compute_keltner(df, ema_period, atr_period, multiplier)
    signals = []

    in_breakout = False
    last_direction = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if pd.isna(row["atr"]) or pd.isna(row["kc_upper"]):
            continue

        if row["close"] > row["kc_upper"] and prev["close"] <= prev["kc_upper"]:
            if not in_breakout or last_direction != Direction.LONG:
                signals.append(Signal(
                    time=row["time"],
                    direction=Direction.LONG,
                    price=row["close"],
                    keltner_band=row["kc_upper"],
                    atr=row["atr"],
                    session=detect_session(row["time"]),
                    candle_index=i,
                ))
                in_breakout = True
                last_direction = Direction.LONG

        elif row["close"] < row["kc_lower"] and prev["close"] >= prev["kc_lower"]:
            if not in_breakout or last_direction != Direction.SHORT:
                signals.append(Signal(
                    time=row["time"],
                    direction=Direction.SHORT,
                    price=row["close"],
                    keltner_band=row["kc_lower"],
                    atr=row["atr"],
                    session=detect_session(row["time"]),
                    candle_index=i,
                ))
                in_breakout = True
                last_direction = Direction.SHORT

        elif row["close"] <= row["kc_upper"] and row["close"] >= row["kc_lower"]:
            in_breakout = False
            last_direction = None

    return signals
