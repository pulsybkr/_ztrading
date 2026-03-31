import pandas as pd
import numpy as np


def compute_keltner(df: pd.DataFrame,
                    ema_period: int = 20,
                    atr_period: int = 14,
                    multiplier: float = 2.0) -> pd.DataFrame:
    df = df.copy()

    df["kc_mid"] = df["close"].ewm(span=ema_period, adjust=False).mean()

    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(span=atr_period, adjust=False).mean()

    df["kc_upper"] = df["kc_mid"] + multiplier * df["atr"]
    df["kc_lower"] = df["kc_mid"] - multiplier * df["atr"]

    return df
