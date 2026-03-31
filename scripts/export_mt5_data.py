import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.mt5_export import init_mt5, export_ticks, export_candles, shutdown_mt5
from datetime import datetime
import typer

cli = typer.Typer()


@cli.command()
def export_ticks_cmd(
    symbol: str = "XAUUSD",
    year: int = 2025,
    month: int = 1,
    output: str = "data/parquet/ticks",
):
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)

    if not init_mt5():
        return

    export_ticks(symbol, start, end, output)
    shutdown_mt5()


@cli.command()
def export_candles_cmd(
    symbol: str = "XAUUSD",
    timeframe: str = "M5",
    year: int = 2025,
    output: str = "data/parquet/candles",
):
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)

    if not init_mt5():
        return

    export_candles(symbol, timeframe, start, end, output)
    shutdown_mt5()


if __name__ == "__main__":
    cli()
