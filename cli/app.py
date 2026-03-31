import typer
from rich.console import Console
from typing import Optional
from pathlib import Path

app = typer.Typer(
    name="nova",
    help="NovaGold Reborn -- Trading ML pour XAU/USD",
    add_completion=False,
)
console = Console()

data_app = typer.Typer(help="Gestion des donnees MT5 <-> Parquet")
backtest_app = typer.Typer(help="Backtesting")
optimize_app = typer.Typer(help="Optimisation des parametres")
train_app = typer.Typer(help="ML : entrainement et evaluation")
live_app = typer.Typer(help="Trading live MT5")

app.add_typer(data_app, name="data")
app.add_typer(backtest_app, name="backtest")
app.add_typer(optimize_app, name="optimize")
app.add_typer(train_app, name="train")
app.add_typer(live_app, name="live")


@data_app.command("export")
def data_export(
    symbol: str = typer.Option("XAUUSD", help="Symbole MT5"),
    type: str = typer.Option("candles", help="ticks ou candles"),
    timeframe: str = typer.Option("M5", help="M1, M5, M15, H1, H4, D1"),
    start: str = typer.Option(..., help="Date debut YYYY-MM-DD"),
    end: str = typer.Option(..., help="Date fin YYYY-MM-DD"),
    output: str = typer.Option("data/parquet", help="Repertoire de sortie"),
):
    from data.mt5_export import init_mt5, export_ticks, export_candles, shutdown_mt5
    from datetime import datetime

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    if not init_mt5():
        raise typer.Exit(1)

    if type == "ticks":
        export_ticks(symbol, start_dt, end_dt, f"{output}/ticks")
    else:
        export_candles(symbol, timeframe, start_dt, end_dt, f"{output}/candles")

    shutdown_mt5()


@data_app.command("info")
def data_info():
    from data.loader import DataLoader
    loader = DataLoader()
    console.print(loader.info().to_string())


@backtest_app.command("run")
def backtest_run(
    symbol: str = typer.Option("XAUUSD", help="Symbole"),
    start: str = typer.Option(..., help="Date debut YYYY-MM-DD"),
    end: str = typer.Option(..., help="Date fin YYYY-MM-DD"),
    lot: float = typer.Option(0.01, help="Taille de lot"),
    commission: float = typer.Option(3.50, help="Commission $/lot round-turn"),
    spread: float = typer.Option(0.20, help="Spread moyen en $"),
    balance: float = typer.Option(10000, help="Capital initial $"),
    keltner_period: int = typer.Option(20, help="Periode EMA Keltner"),
    keltner_mult: float = typer.Option(2.0, help="Multiplicateur Keltner"),
    sl_atr_mult: float = typer.Option(2.0, help="SL = N x ATR"),
    trailing_atr_mult: float = typer.Option(1.5, help="Trailing = N x ATR"),
    breakeven_atr_mult: float = typer.Option(1.0, help="Breakeven a N x ATR"),
    regime_filter: bool = typer.Option(False, help="Activer filtre de session"),
    sessions: str = typer.Option("sge_open,london,overlap", help="Sessions autorisees"),
    atr_ratio_filter: bool = typer.Option(False, help="Activer filtre ATR ratio"),
    atr_ratio_threshold: float = typer.Option(0.15, help="Seuil ATR ratio M5/H1"),
):
    from core.types import BacktestConfig
    from backtest.engine import run_backtest

    config = BacktestConfig(
        symbol=symbol,
        lot_size=lot,
        initial_balance=balance,
        commission_per_lot=commission,
        spread=spread,
        keltner_ema_period=keltner_period,
        keltner_multiplier=keltner_mult,
        sl_atr_mult=sl_atr_mult,
        trailing_atr_mult=trailing_atr_mult,
        breakeven_atr_mult=breakeven_atr_mult,
        use_session_filter=regime_filter,
        allowed_sessions=sessions.split(","),
        use_atr_ratio_filter=atr_ratio_filter,
        atr_ratio_threshold=atr_ratio_threshold,
    )

    trades, metrics = run_backtest(config, start, end)


@optimize_app.command("grid")
def optimize_grid(
    symbol: str = typer.Option("XAUUSD"),
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    param: list[str] = typer.Option([], help="param:val1,val2,val3"),
):
    from itertools import product
    from core.types import BacktestConfig
    from backtest.engine import run_backtest

    param_grid = {}
    for p in param:
        name, values = p.split(":")
        param_grid[name] = [float(v) for v in values.split(",")]

    keys = list(param_grid.keys())
    all_combos = list(product(*param_grid.values()))
    console.print(f"{len(all_combos)} combinaisons a tester...")

    best_pf = 0
    best_params = {}

    for i, combo in enumerate(all_combos):
        params = dict(zip(keys, combo))
        config = BacktestConfig(symbol=symbol, **{k: v for k, v in params.items()})
        trades, metrics = run_backtest(config, start, end, verbose=False)

        pf = metrics.get("profit_factor", 0)
        if pf > best_pf:
            best_pf = pf
            best_params = params

        console.print(f"  [{i+1}/{len(all_combos)}] {params} -> PF={pf:.2f}")

    console.print(f"\nMeilleur : PF={best_pf:.2f} | {best_params}")


@train_app.command("prepare")
def train_prepare(
    symbol: str = typer.Option("XAUUSD", help="Symbole"),
    start: str = typer.Option(..., help="Date debut YYYY-MM-DD"),
    end: str = typer.Option(..., help="Date fin YYYY-MM-DD"),
    output: str = typer.Option("ml/models/signals_features.parquet", help="Fichier de sortie"),
):
    from data.loader import DataLoader
    from strategy.breakout import detect_breakouts
    from strategy.regime import attach_atr_ratios
    from ml.features import signals_to_dataframe
    from strategy.keltner import compute_keltner
    import pandas as pd

    console.print(f"Preparation des donnees {symbol} {start} -> {end}...")

    loader = DataLoader()
    df_m5 = loader.load_candles(symbol, "M5", start, end)
    df_m5 = compute_keltner(df_m5)

    signals = detect_breakouts(df_m5)

    df_h1 = None
    try:
        df_h1 = loader.load_candles(symbol, "H1", start, end)
        df_h1 = compute_keltner(df_h1)
        signals = attach_atr_ratios(signals, df_m5, df_h1)
        console.print(f"  H1 charge: {len(df_h1)} bougies")
    except Exception:
        pass

    df = signals_to_dataframe(signals, df_m5, df_h1)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, engine="pyarrow", compression="snappy")
    console.print(f"  {len(df)} signaux sauvegardes -> {output}")


@train_app.command("run")
def train_run(
    symbol: str = typer.Option("XAUUSD", help="Symbole"),
    start: str = typer.Option(..., help="Date debut YYYY-MM-DD"),
    end: str = typer.Option(..., help="Date fin YYYY-MM-DD"),
    train_weeks: int = typer.Option(8, help="Semaines d'entrainement"),
    test_weeks: int = typer.Option(2, help="Semaines de test"),
    output: str = typer.Option("ml/models/model.txt", help="Fichier modele"),
):
    from data.loader import DataLoader
    from strategy.breakout import detect_breakouts
    from strategy.regime import attach_atr_ratios
    from strategy.keltner import compute_keltner
    from ml.trainer import walk_forward_train, train_final_model, save_model

    console.print(f"Entrainement walk-forward {symbol}...")

    loader = DataLoader()
    df_m5 = loader.load_candles(symbol, "M5", start, end)
    df_m5 = compute_keltner(df_m5)
    signals = detect_breakouts(df_m5)

    df_h1 = None
    try:
        df_h1 = loader.load_candles(symbol, "H1", start, end)
        df_h1 = compute_keltner(df_h1)
        signals = attach_atr_ratios(signals, df_m5, df_h1)
    except Exception:
        pass

    console.print(f"  {len(signals)} signaux detects")

    fold_results = walk_forward_train(
        signals=signals,
        candles_m5=df_m5,
        candles_h1=df_h1,
        train_weeks=train_weeks,
        test_weeks=test_weeks,
    )

    if fold_results:
        best_fold = max(fold_results, key=lambda x: x["precision"])
        console.print(f"\nMeilleur fold: {best_fold['fold']} avec precision={best_fold['precision']:.3f}")
        save_model(best_fold["model"], output)
    else:
        console.print("[red]Aucun fold valide - donn�es insuffisantes[/red]")


@live_app.command("paper")
def live_paper(
    symbol: str = typer.Option("XAUUSD", help="Symbole"),
    lot: float = typer.Option(0.01, help="Taille de lot"),
    login: int = typer.Option(None, help="MT5 login"),
    password: str = typer.Option(None, help="MT5 password"),
    server: str = typer.Option(None, help="MT5 server"),
):
    from core.types import BacktestConfig
    from live.signal_loop import SignalLoop
    from live.mt5_bridge import MT5Bridge

    config = BacktestConfig(symbol=symbol, lot_size=lot)
    bridge = MT5Bridge(config)

    if not bridge.connect(login, password, server):
        console.print("[red]Connexion MT5 echouee[/red]")
        raise typer.Exit(1)

    loop = SignalLoop(config, bridge)
    console.print("[yellow]Paper trading - signaux simules[/yellow]")
    console.print("[yellow]Ctrl+C pour arreter[/yellow]")

    try:
        loop.start(paper=True, regime_filter=True)
    except KeyboardInterrupt:
        loop.stop()
        bridge.disconnect()


@live_app.command("start")
def live_start(
    symbol: str = typer.Option("XAUUSD", help="Symbole"),
    lot: float = typer.Option(0.01, help="Taille de lot"),
    login: int = typer.Option(..., help="MT5 login"),
    password: str = typer.Option(..., help="MT5 password"),
    server: str = typer.Option(..., help="MT5 server"),
):
    from core.types import BacktestConfig
    from live.signal_loop import SignalLoop
    from live.mt5_bridge import MT5Bridge

    config = BacktestConfig(symbol=symbol, lot_size=lot)
    bridge = MT5Bridge(config)

    if not bridge.connect(login, password, server):
        console.print("[red]Connexion MT5 echouee[/red]")
        raise typer.Exit(1)

    loop = SignalLoop(config, bridge)
    console.print("[red]=== LIVE TRADING ACTIF ===[/red]")
    console.print("[red]Ctrl+C pour arreter[/red]")

    try:
        loop.start(paper=False, regime_filter=True)
    except KeyboardInterrupt:
        loop.stop()
        bridge.disconnect()


@live_app.command("status")
def live_status(
    login: int = typer.Option(None, help="MT5 login"),
    password: str = typer.Option(None, help="MT5 password"),
    server: str = typer.Option(None, help="MT5 server"),
):
    from core.types import BacktestConfig
    from live.mt5_bridge import MT5Bridge

    config = BacktestConfig()
    bridge = MT5Bridge(config)

    if not bridge.connect(login, password, server):
        console.print("[red]Connexion echouee[/red]")
        raise typer.Exit(1)

    info = bridge.get_account_info()
    positions = bridge.get_open_positions()

    console.print(f"\n=== Account ===")
    console.print(f"Balance: ${info['balance']:.2f}")
    console.print(f"Equity:  ${info['equity']:.2f}")
    console.print(f"Profit:  ${info['profit']:.2f}")

    console.print(f"\n=== Positions ouvertes: {len(positions)} ===")
    for pos in positions:
        console.print(f"  {pos.ticket} | {pos.symbol} | {pos.type} | {pos.volume} lots | P/L: ${pos.profit:.2f}")

    bridge.disconnect()


if __name__ == "__main__":
    app()
