import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def init_mt5(login: int = None, password: str = None, server: str = None) -> bool:
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return False

    if login and password and server:
        if not mt5.login(login, password=password, server=server):
            print(f"MT5 login failed: {mt5.last_error()}")
            return False

    info = mt5.account_info()
    print(f"Connecte: {info.server} | Compte: {info.login} | Balance: ${info.balance:.2f}")
    return True


def export_ticks(symbol: str, start: datetime, end: datetime, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    current = start.replace(day=1)
    while current < end:
        next_month = (current + timedelta(days=32)).replace(day=1)
        month_end = min(next_month, end)

        filename = f"{symbol}_ticks_{current.strftime('%Y-%m')}.parquet"
        filepath = output_path / filename

        if filepath.exists():
            print(f"{filename} existe deja, skip")
            current = next_month
            continue

        print(f"Export ticks {symbol} {current.strftime('%Y-%m')}...", end=" ", flush=True)

        ticks = mt5.copy_ticks_range(symbol, current, month_end, mt5.COPY_TICKS_ALL)

        if ticks is None or len(ticks) == 0:
            print("Aucun tick")
            current = next_month
            continue

        df = pd.DataFrame(ticks)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[["time", "bid", "ask", "last", "volume", "flags"]]

        df.to_parquet(filepath, engine="pyarrow", compression="snappy")
        print(f"{len(df):,} ticks -> {filepath.stat().st_size / 1024 / 1024:.1f} MB")

        del df, ticks
        current = next_month


def export_candles(symbol: str, timeframe: str, start: datetime, end: datetime, output_dir: str):
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

    if timeframe not in tf_map:
        raise ValueError(f"Timeframe inconnu: {timeframe}. Disponibles: {list(tf_map.keys())}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / f"{symbol}_{timeframe}_{start.year}-{end.year}.parquet"

    print(f"Export {symbol} {timeframe} {start.date()} -> {end.date()}...", end=" ", flush=True)

    rates = mt5.copy_rates_range(symbol, tf_map[timeframe], start, end)

    if rates is None or len(rates) == 0:
        print("Aucune bougie")
        return

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]

    df.to_parquet(filepath, engine="pyarrow", compression="snappy")
    print(f"{len(df):,} bougies -> {filepath.stat().st_size / 1024:.0f} KB")


def shutdown_mt5():
    mt5.shutdown()
    print("MT5 deconnecte")
