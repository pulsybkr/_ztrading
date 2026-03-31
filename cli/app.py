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


# ─── DATA COMMANDS ───────────────────────────────────────────

@data_app.command("export")
def data_export(
    symbol: str = typer.Option("XAUUSD", help="Symbole MT5"),
    type: str = typer.Option("candles", help="ticks, candles, ou all"),
    timeframe: str = typer.Option("M5", help="M1, M5, M15, H1, H4, D1"),
    start: str = typer.Option(..., help="Date debut YYYY-MM-DD"),
    end: str = typer.Option(..., help="Date fin YYYY-MM-DD"),
    output: str = typer.Option("data/parquet", help="Repertoire de sortie"),
    no_ticks: bool = typer.Option(False, help="Skip ticks dans mode 'all'"),
):
    """Exporter les données depuis MT5 vers Parquet."""
    from data.mt5_export import init_mt5, export_ticks, export_candles, export_all, shutdown_mt5
    from datetime import datetime

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    if not init_mt5():
        raise typer.Exit(1)

    try:
        if type == "ticks":
            export_ticks(symbol, start_dt, end_dt, f"{output}/ticks")
        elif type == "all":
            export_all(symbol, start_dt, end_dt, output, include_ticks=not no_ticks)
        else:
            export_candles(symbol, timeframe, start_dt, end_dt, f"{output}/candles")
    finally:
        shutdown_mt5()


@data_app.command("info")
def data_info():
    """Afficher les données disponibles."""
    from data.loader import DataLoader
    loader = DataLoader()
    df = loader.info()
    if len(df) == 0:
        console.print("[yellow]Aucune donnée trouvée dans data/parquet/[/yellow]")
    else:
        console.print(df.to_string(index=False))


@data_app.command("verify")
def data_verify():
    """Vérifier l'intégrité des données."""
    from data.mt5_export import verify_data
    verify_data()


# ─── BACKTEST COMMANDS ───────────────────────────────────────

@backtest_app.command("run")
def backtest_run(
    symbol: str = typer.Option("XAUUSD", help="Symbole"),
    start: str = typer.Option(..., help="Date debut YYYY-MM-DD"),
    end: str = typer.Option(..., help="Date fin YYYY-MM-DD"),
    lot: float = typer.Option(0.01, help="Taille de lot"),
    commission: float = typer.Option(3.50, help="Commission $/lot round-turn"),
    spread: float = typer.Option(0.20, help="Spread moyen en $"),
    balance: float = typer.Option(100, help="Capital initial $ (compte reel ~50-100 EUR)"),
    keltner_period: int = typer.Option(20, help="Periode EMA Keltner"),
    keltner_mult: float = typer.Option(2.0, help="Multiplicateur Keltner"),
    sl_atr_mult: float = typer.Option(1.0, help="SL = N x ATR"),
    trailing_atr_mult: float = typer.Option(0.75, help="Trailing = N x ATR"),
    breakeven_atr_mult: float = typer.Option(0.5, help="Breakeven a N x ATR"),
    regime_filter: bool = typer.Option(False, help="Activer filtre de session"),
    sessions: str = typer.Option("sge_open,london,overlap", help="Sessions autorisees"),
    atr_ratio_filter: bool = typer.Option(False, help="Activer filtre ATR ratio"),
    atr_ratio_threshold: float = typer.Option(0.15, help="Seuil ATR ratio M5/H1"),
    use_ml_filter: bool = typer.Option(False, help="Activer filtre ML (necessite ml/models/model.joblib)"),
    ml_threshold: float = typer.Option(0.55, help="Seuil de confiance ML"),
    timeframe: str = typer.Option("M5", help="Timeframe des signaux: M1, M5, M15, M30, H1"),
    resolution: str = typer.Option("auto", help="Resolution: auto, ticks, m1, m5"),
):
    """Lancer un backtest."""
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
        use_ml_filter=use_ml_filter,
        ml_threshold=ml_threshold,
        signal_timeframe=timeframe.upper(),
    )

    trades, metrics = run_backtest(config, start, end, resolution=resolution)


@backtest_app.command("compare")
def backtest_compare(
    start: str = typer.Option(..., help="Date debut YYYY-MM-DD"),
    end: str = typer.Option(..., help="Date fin YYYY-MM-DD"),
    resolution: str = typer.Option("auto", help="Resolution: auto, ticks, m1, m5"),
):
    """Comparer les configs: baseline vs session vs session+ATR."""
    from backtest.engine import run_comparison
    run_comparison(start, end, resolution=resolution)


# ─── OPTIMIZE COMMANDS ───────────────────────────────────────

@optimize_app.command("grid")
def optimize_grid(
    symbol: str = typer.Option("XAUUSD"),
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    param: list[str] = typer.Option([], help="param:val1,val2,val3"),
):
    """Grid search sur les paramètres."""
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


# ─── TRAIN COMMANDS ──────────────────────────────────────────

@train_app.command("prepare")
def train_prepare(
    symbol: str = typer.Option("XAUUSD", help="Symbole"),
    start: str = typer.Option(..., help="Date debut YYYY-MM-DD"),
    end: str = typer.Option(..., help="Date fin YYYY-MM-DD"),
    output: str = typer.Option("ml/models/signals_features.parquet", help="Fichier de sortie"),
):
    """Préparer les features pour l'entraînement ML."""
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
    console.print(f"  M5: {len(df_m5)} bougies")

    signals = detect_breakouts(df_m5)
    console.print(f"  Signaux détectés: {len(signals)}")

    df_h1 = None
    try:
        df_h1 = loader.load_candles(symbol, "H1", start, end)
        df_h1 = compute_keltner(df_h1)
        signals = attach_atr_ratios(signals, df_m5, df_h1)
        console.print(f"  H1 chargé: {len(df_h1)} bougies")
    except FileNotFoundError:
        console.print("  [yellow]H1 non disponible, atr_ratio sera 0[/yellow]")

    df = signals_to_dataframe(signals, df_m5, df_h1)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, engine="pyarrow", compression="snappy")
    console.print(f"  [green]{len(df)} signaux sauvegardés -> {output}[/green]")


@train_app.command("run")
def train_run(
    symbol: str = typer.Option("XAUUSD", help="Symbole"),
    start: str = typer.Option(..., help="Date debut YYYY-MM-DD"),
    end: str = typer.Option(..., help="Date fin YYYY-MM-DD"),
    train_weeks: int = typer.Option(8, help="Semaines d'entrainement"),
    test_weeks: int = typer.Option(2, help="Semaines de test"),
    output: str = typer.Option("ml/models/model.joblib", help="Fichier modele"),
):
    """Entraîner le modèle LightGBM en walk-forward."""
    from data.loader import DataLoader
    from strategy.breakout import detect_breakouts
    from strategy.regime import attach_atr_ratios
    from strategy.keltner import compute_keltner
    from ml.trainer import walk_forward_train, train_final_model, save_model
    from core.types import BacktestConfig

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
        console.print(f"  H1: {len(df_h1)} bougies")
    except FileNotFoundError:
        console.print("  [yellow]H1 non disponible[/yellow]")

    console.print(f"  {len(signals)} signaux détectés")

    config = BacktestConfig()
    fold_results = walk_forward_train(
        signals=signals,
        candles_m5=df_m5,
        candles_h1=df_h1,
        train_weeks=train_weeks,
        test_weeks=test_weeks,
        config=config,
    )

    if fold_results:
        # Display fold results
        from rich.table import Table
        table = Table(title="Walk-Forward Results")
        table.add_column("Fold", justify="right")
        table.add_column("Train", min_width=20)
        table.add_column("Test", min_width=20)
        table.add_column("Samples", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")

        for f in fold_results:
            table.add_row(
                str(f["fold"]),
                f"{f['train_start']} → {f['train_end']}",
                f"{f['test_start']} → {f['test_end']}",
                f"{f['train_samples']}/{f['test_samples']}",
                f"{f['accuracy']:.3f}",
                f"{f['precision']:.3f}",
                f"{f['recall']:.3f}",
            )
        console.print(table)

        # Save best model by precision
        best_fold = max(fold_results, key=lambda x: x["precision"])
        metadata = {
            "best_fold": best_fold["fold"],
            "precision": best_fold["precision"],
            "recall": best_fold["recall"],
            "accuracy": best_fold["accuracy"],
            "train_period": f"{best_fold['train_start']} → {best_fold['train_end']}",
            "test_period": f"{best_fold['test_start']} → {best_fold['test_end']}",
            "train_weeks": train_weeks,
            "test_weeks": test_weeks,
        }
        save_model(best_fold["model"], output, metadata=metadata)
        console.print(f"\n[green]Meilleur fold: {best_fold['fold']} | "
                      f"precision={best_fold['precision']:.3f}[/green]")
    else:
        console.print("[red]Aucun fold valide — données insuffisantes[/red]")


# ─── LIVE COMMANDS ───────────────────────────────────────────

@live_app.command("paper")
def live_paper(
    symbol: str = typer.Option("XAUUSD", help="Symbole"),
    lot: float = typer.Option(0.01, help="Taille de lot"),
    login: int = typer.Option(None, help="MT5 login"),
    password: str = typer.Option(None, help="MT5 password"),
    server: str = typer.Option(None, help="MT5 server"),
):
    """Paper trading (simulation temps réel sur MT5 demo)."""
    from core.types import BacktestConfig
    from live.signal_loop import SignalLoop
    from live.mt5_bridge import MT5Bridge

    config = BacktestConfig(symbol=symbol, lot_size=lot)
    bridge = MT5Bridge(config)

    if not bridge.connect(login, password, server):
        console.print("[red]Connexion MT5 echouee[/red]")
        raise typer.Exit(1)

    loop = SignalLoop(config, bridge)
    console.print("[yellow]Paper trading — signaux simules[/yellow]")
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
    """LIVE TRADING — utiliser avec prudence."""
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
    """Afficher le status du compte MT5."""
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
