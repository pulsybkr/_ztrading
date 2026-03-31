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

app.add_typer(data_app, name="data")
app.add_typer(backtest_app, name="backtest")
app.add_typer(optimize_app, name="optimize")


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


if __name__ == "__main__":
    app()
