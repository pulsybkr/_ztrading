"""
Generate synthetic XAUUSD M5 data for testing the backtester.
Uses realistic price movements based on typical gold volatility.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def generate_synthetic_m5(start_date: str, end_date: str, output_dir: str = "data/parquet/candles") -> str:
    """
    Generate synthetic M5 candles for XAU/USD.
    
    Args:
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD'
        output_dir: Output directory for parquet files
    
    Returns:
        Path to generated file
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate 5-minute candles
    dates = pd.date_range(start, end, freq="5min")
    # Only keep times between Sunday 22:00 and Friday 22:00 (MT5 trading week)
    dates = [d for d in dates if d.weekday() < 5 and (d.hour >= 22 or d.hour < 22)]
    
    np.random.seed(42)  # Reproducible

    # Starting price ~$2000 (gold price)
    base_price = 2000.0

    # Gold trades with ~100-200 point range daily
    # Need moves > 2x ATR (ATR ~5-10 points) to trigger breakouts
    volatility = 3.0  # Points per 5min candle

    candles = []
    current_price = base_price
    trend = 0.0  # Momentum component

    for i, dt in enumerate(dates):
        # Random walk with occasional trends
        trend = trend * 0.95 + np.random.randn() * 0.3  # Momentum decay
        shock = np.random.randn() * volatility + trend

        open_price = current_price
        close_price = open_price + shock

        # High/Low includes the full range of movement
        high_price = max(open_price, close_price) + abs(np.random.randn()) * volatility
        low_price = min(open_price, close_price) - abs(np.random.randn()) * volatility

        # Tick volume (realistic: 1000-10000 ticks per 5min for gold)
        tick_volume = int(np.random.lognormal(8, 0.5))

        # Spread in points (20-50 points typical for gold)
        spread = int(np.random.uniform(20, 50))

        candles.append({
            "time": dt,
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "tick_volume": tick_volume,
            "spread": spread,
            "real_volume": 0,
        })

        current_price = close_price
    
    df = pd.DataFrame(candles)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / f"XAUUSD_M5_{start_date}_synthetic.parquet"
    
    df.to_parquet(filepath, engine="pyarrow", compression="snappy")
    print(f"Generated {len(df)} M5 candles -> {filepath}")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"  Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    
    return str(filepath)


if __name__ == "__main__":
    import typer
    cli = typer.Typer()
    
    @cli.command()
    def generate(
        start: str = "2025-06-01",
        end: str = "2025-06-30",
        output: str = "data/parquet/candles",
    ):
        """Generate synthetic M5 data for backtesting."""
        generate_synthetic_m5(start, end, output)
    
    cli()
