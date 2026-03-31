import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import BacktestConfig
from backtest.engine import run_backtest

if __name__ == "__main__":
    config = BacktestConfig(
        symbol="XAUUSD",
        lot_size=0.01,
        initial_balance=10_000,
        keltner_ema_period=20,
        keltner_multiplier=2.0,
        sl_atr_mult=2.0,
        trailing_atr_mult=1.5,
        breakeven_atr_mult=1.0,
    )

    print("Quick backtest XAUUSD...")
    trades, metrics = run_backtest(
        config,
        start="2025-06-01",
        end="2025-12-31",
        verbose=True,
    )
