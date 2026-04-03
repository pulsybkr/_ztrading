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
                     multiplier: float = 2.0,
                     candle_minutes: int = 5,
                     keltner_precomputed: bool = False) -> list[Signal]:
    """
    Détecte les cassures Keltner sur le DataFrame fourni.

    Args:
        df: DataFrame avec les bougies (OHLC)
        ema_period: Période EMA pour Keltner
        atr_period: Période ATR pour Keltner
        multiplier: Multiplicateur pour les bandes Keltner
        candle_minutes: Durée d'une bougie en minutes (pour décaler le signal)
        keltner_precomputed: Si True, suppose que kc_upper/kc_lower/atr sont déjà dans df

    Si keltner_precomputed=True, la fonction suppose que les colonnes suivantes existent :
    - kc_upper, kc_lower, atr
    """
    if not keltner_precomputed:
        df = compute_keltner(df, ema_period, atr_period, multiplier)
    else:
        # Vérifier que les colonnes requises sont présentes
        required_cols = ["kc_upper", "kc_lower", "atr"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"keltner_precomputed=True mais colonnes manquantes: {missing}")
        df = df.copy()  # Éviter de modifier le DataFrame original
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
                # Entry at NEXT candle open: offset = 1 période pour éviter le look-ahead
                next_time = row["time"] + pd.Timedelta(minutes=candle_minutes)
                signals.append(Signal(
                    time=next_time,
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
                next_time = row["time"] + pd.Timedelta(minutes=candle_minutes)
                signals.append(Signal(
                    time=next_time,
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
