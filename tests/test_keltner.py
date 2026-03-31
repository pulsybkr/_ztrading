import pytest
import pandas as pd
import numpy as np
from strategy.keltner import compute_keltner


@pytest.fixture
def sample_candles():
    dates = pd.date_range("2025-01-01", periods=50, freq="5min")
    np.random.seed(42)
    
    base_price = 2000.0
    data = {
        "time": dates,
        "open": [base_price + np.random.randn() * 0.5 for _ in range(50)],
        "high": [],
        "low": [],
        "close": [],
        "tick_volume": [1000 + np.random.randint(0, 500) for _ in range(50)],
    }
    
    for i in range(50):
        o = data["open"][i]
        c = o + np.random.randn() * 0.5
        h = max(o, c) + abs(np.random.randn()) * 0.3
        l = min(o, c) - abs(np.random.randn()) * 0.3
        data["high"].append(h)
        data["low"].append(l)
        data["close"].append(c)
    
    return pd.DataFrame(data)


def test_keltner_ema_period(sample_candles):
    df = compute_keltner(sample_candles, ema_period=20, atr_period=14, multiplier=2.0)
    
    assert "kc_mid" in df.columns
    assert "kc_upper" in df.columns
    assert "kc_lower" in df.columns
    assert "atr" in df.columns
    
    assert df["kc_upper"].iloc[-1] > df["kc_mid"].iloc[-1]
    assert df["kc_lower"].iloc[-1] < df["kc_mid"].iloc[-1]
    
    assert df["kc_upper"].iloc[-1] - df["kc_lower"].iloc[-1] > 0


def test_keltner_multiplier_effect(sample_candles):
    df_small = compute_keltner(sample_candles, ema_period=20, atr_period=14, multiplier=1.5)
    df_large = compute_keltner(sample_candles, ema_period=20, atr_period=14, multiplier=3.0)
    
    assert (df_large["kc_upper"] - df_large["kc_mid"]).iloc[-1] > (df_small["kc_upper"] - df_small["kc_mid"]).iloc[-1]
    assert (df_large["kc_mid"] - df_large["kc_lower"]).iloc[-1] > (df_small["kc_mid"] - df_small["kc_lower"]).iloc[-1]
