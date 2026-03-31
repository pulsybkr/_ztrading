"""
Script de backtest rapide — lance les 3 configurations et compare.

Usage:
    python scripts/quick_backtest.py
    python scripts/quick_backtest.py --start 2025-06-01 --end 2025-12-31
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import BacktestConfig
from backtest.engine import run_backtest, run_comparison


def quick_single(start: str = "2025-01-01", end: str = "2025-12-31"):
    """Run a single baseline backtest."""
    config = BacktestConfig(
        symbol="XAUUSD",
        lot_size=0.01,
        initial_balance=10_000,
        keltner_ema_period=20,
        keltner_atr_period=14,
        keltner_multiplier=2.0,
        sl_atr_mult=2.0,
        breakeven_atr_mult=1.0,
        trailing_atr_mult=1.5,
        spread=0.20,
        commission_per_lot=3.50,
        slippage=0.05,
        use_session_filter=False,
        use_atr_ratio_filter=False,
    )

    trades, metrics = run_backtest(config, start, end, resolution="auto")
    return trades, metrics


def quick_compare(start: str = "2025-01-01", end: str = "2025-12-31"):
    """Run comparison of all configurations."""
    results = run_comparison(start, end, resolution="auto")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick Backtest")
    parser.add_argument("--start", default="2025-01-01", help="Start date")
    parser.add_argument("--end", default="2025-12-31", help="End date")
    parser.add_argument("--compare", action="store_true", help="Run comparison mode")

    args = parser.parse_args()

    if args.compare:
        quick_compare(args.start, args.end)
    else:
        quick_single(args.start, args.end)
