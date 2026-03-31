"""
Export des données depuis MetaTrader 5 vers Parquet.
À exécuter sur un PC Windows avec MT5 installé et connecté.

Usage:
    python -m data.mt5_export --help
    python -m data.mt5_export all --start 2025-01-01 --end 2025-12-31
    python -m data.mt5_export ticks --start 2025-06-01 --end 2025-12-31
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os


# ─── MT5 Connection ─────────────────────────────────────────

def init_mt5(login: int = None, password: str = None, server: str = None) -> bool:
    """Initialise la connexion MT5. Lit .env si pas d'arguments."""
    if not mt5.initialize():
        print(f"❌ MT5 init failed: {mt5.last_error()}")
        return False

    # Try loading from .env if not provided
    if not login:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            login = int(os.getenv("MT5_LOGIN", 0))
            password = os.getenv("MT5_PASSWORD", "")
            server = os.getenv("MT5_SERVER", "")
        except ImportError:
            # Load .env manually
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                for line in env_path.read_text().strip().splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip()
                login = int(os.getenv("MT5_LOGIN", 0))
                password = os.getenv("MT5_PASSWORD", "")
                server = os.getenv("MT5_SERVER", "")

    if login and password and server:
        if not mt5.login(login, password=password, server=server):
            print(f"❌ MT5 login failed: {mt5.last_error()}")
            return False

    info = mt5.account_info()
    if info:
        print(f"✅ Connecté: {info.server} | Compte: {info.login} | Balance: ${info.balance:.2f}")
    return True


def shutdown_mt5():
    mt5.shutdown()
    print("🔌 MT5 déconnecté")


# ─── Tick Export ─────────────────────────────────────────────

def export_ticks(symbol: str, start: datetime, end: datetime,
                 output_dir: str = "data/parquet/ticks"):
    """
    Exporte les ticks mois par mois pour éviter la surcharge RAM.
    Chaque mois = 1 fichier parquet.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    current = start.replace(day=1)
    total_ticks = 0

    while current < end:
        next_month = (current + timedelta(days=32)).replace(day=1)
        month_end = min(next_month, end)

        filename = f"{symbol}_ticks_{current.strftime('%Y-%m')}.parquet"
        filepath = output_path / filename

        if filepath.exists():
            existing = pd.read_parquet(filepath)
            print(f"⏭️  {filename} existe ({len(existing):,} ticks), skip")
            total_ticks += len(existing)
            current = next_month
            continue

        print(f"📥 Export ticks {symbol} {current.strftime('%Y-%m')}...", end=" ", flush=True)

        ticks = mt5.copy_ticks_range(symbol, current, month_end, mt5.COPY_TICKS_ALL)

        if ticks is None or len(ticks) == 0:
            print(f"⚠️  Aucun tick pour {current.strftime('%Y-%m')}")
            current = next_month
            continue

        df = pd.DataFrame(ticks)
        df["time"] = pd.to_datetime(df["time"], unit="s")

        # Garder uniquement les colonnes utiles
        cols = [c for c in ["time", "bid", "ask", "last", "volume", "flags"] if c in df.columns]
        df = df[cols]

        df.to_parquet(filepath, engine="pyarrow", compression="snappy")
        size_mb = filepath.stat().st_size / 1024 / 1024
        print(f"✅ {len(df):,} ticks → {size_mb:.1f} MB")
        total_ticks += len(df)

        del df, ticks
        current = next_month

    print(f"\n📊 Total ticks exportés: {total_ticks:,}")


# ─── Candle Export ───────────────────────────────────────────

def export_candles(symbol: str, timeframe: str, start: datetime, end: datetime,
                   output_dir: str = "data/parquet/candles"):
    """
    Exporte les bougies pour un timeframe donné.
    Télécharge mois par mois pour éviter les limites de l'API MT5.
    """
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

    start_year = start.year
    end_year = end.year
    filepath = output_path / f"{symbol}_{timeframe}_{start_year}-{end_year}.parquet"

    print(f"📥 Export {symbol} {timeframe} {start.date()} → {end.date()}...", end=" ", flush=True)

    current = start
    all_dfs = []

    while current < end:
        next_month = (current + timedelta(days=32)).replace(day=1)
        month_end = min(next_month, end)

        rates = mt5.copy_rates_range(symbol, tf_map[timeframe], current, month_end)
        
        if rates is not None and len(rates) > 0:
            all_dfs.append(pd.DataFrame(rates))

        current = next_month

    if not all_dfs:
        print("⚠️  Aucune bougie")
        return None

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["time"])
    
    df["time"] = pd.to_datetime(df["time"], unit="s")

    cols = [c for c in ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
            if c in df.columns]
    df = df[cols]

    df.to_parquet(filepath, engine="pyarrow", compression="snappy")
    size_mb = filepath.stat().st_size / 1024 / 1024
    print(f"✅ {len(df):,} bougies → {size_mb:.2f} MB")

    return str(filepath)


# ─── Export All ──────────────────────────────────────────────

def export_all(symbol: str, start: datetime, end: datetime,
               output_dir: str = "data/parquet",
               include_ticks: bool = True):
    """
    Exporte toutes les données nécessaires au backtesting :
    - Bougies M1, M5, M15, H1
    - Ticks (optionnel, le plus lourd)
    """
    candles_dir = f"{output_dir}/candles"
    ticks_dir = f"{output_dir}/ticks"

    print("=" * 60)
    print(f"  Export complet {symbol}")
    print(f"  Période: {start.date()} → {end.date()}")
    print("=" * 60)

    # Candles (du plus léger au plus lourd)
    for tf in ["H1", "M15", "M5", "M1"]:
        print(f"\n{'─' * 40}")
        export_candles(symbol, tf, start, end, candles_dir)

    # Ticks (le plus lourd)
    if include_ticks:
        print(f"\n{'─' * 40}")
        print("⚠️  Export ticks (peut prendre du temps)...")
        export_ticks(symbol, start, end, ticks_dir)

    print(f"\n{'=' * 60}")
    print(f"  Export terminé !")
    print(f"  Données dans: {output_dir}")
    print(f"{'=' * 60}")


# ─── Data Verification ──────────────────────────────────────

def verify_data(data_dir: str = "data/parquet"):
    """Vérifie l'intégrité et affiche un résumé des données."""
    data_path = Path(data_dir)

    print("\n📊 Données disponibles:")
    print("─" * 70)
    print(f"{'Fichier':<45} {'Rows':>10} {'Size':>10}")
    print("─" * 70)

    for f in sorted(data_path.rglob("*.parquet")):
        try:
            df = pd.read_parquet(f)
            size = f.stat().st_size / 1024 / 1024
            start = df["time"].min()
            end = df["time"].max()
            print(f"  {f.name:<43} {len(df):>10,} {size:>8.1f} MB")
            print(f"    └─ {start} → {end}")
        except Exception as e:
            print(f"  {f.name:<43} ❌ ERREUR: {e}")

    print("─" * 70)


# ─── CLI ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export MT5 → Parquet")
    parser.add_argument("action", choices=["ticks", "candles", "all", "verify"],
                        help="Type d'export")
    parser.add_argument("--symbol", default="XAUUSD", help="Symbole MT5")
    parser.add_argument("--timeframe", default="M5", help="M1, M5, M15, H1, H4, D1")
    parser.add_argument("--start", default="2025-01-01", help="Date début YYYY-MM-DD")
    parser.add_argument("--end", default="2025-12-31", help="Date fin YYYY-MM-DD")
    parser.add_argument("--output", default="data/parquet", help="Répertoire de sortie")
    parser.add_argument("--no-ticks", action="store_true", help="Skip ticks dans 'all'")

    args = parser.parse_args()

    if args.action == "verify":
        verify_data(args.output)
        sys.exit(0)

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    if not init_mt5():
        sys.exit(1)

    try:
        if args.action == "ticks":
            export_ticks(args.symbol, start_dt, end_dt, f"{args.output}/ticks")
        elif args.action == "candles":
            export_candles(args.symbol, args.timeframe, start_dt, end_dt, f"{args.output}/candles")
        elif args.action == "all":
            export_all(args.symbol, start_dt, end_dt, args.output,
                       include_ticks=not args.no_ticks)
    finally:
        shutdown_mt5()
