from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from core.types import BacktestConfig

console = Console()


def print_report(trades: list, metrics: dict, config: BacktestConfig):

    if "error" in metrics:
        console.print(f"[red]Erreur: {metrics['error']}[/red]")
        return

    lines = []
    lines.append(f"Periode     : {trades[0].entry_time.date()} -> {trades[-1].exit_time.date()}")
    lines.append(f"Capital     : ${config.initial_balance:,.2f}")
    lines.append(f"Lot size    : {config.lot_size} ({config.lot_size * config.contract_size:.0f}$/point)")
    lines.append(f"Commission  : ${config.commission_per_lot}/lot RT | Spread: ${config.spread}")
    lines.append("")
    lines.append("--- Resultats ---")

    net = metrics["net_profit"]
    net_color = "green" if net > 0 else "red"
    lines.append(f"Trades total     : {metrics['total_trades']}")
    lines.append(f"Trades gagnants  : {metrics['winning_trades']} ({metrics['winrate']:.1f}%)")
    lines.append(f"Trades perdants  : {metrics['losing_trades']}")
    lines.append(f"Gain moyen       : ${metrics['avg_win']:+.2f}")
    lines.append(f"Perte moyenne    : ${metrics['avg_loss']:+.2f}")
    lines.append(f"Profit Factor    : {metrics['profit_factor']:.2f}")
    lines.append(f"Esperance/trade  : ${metrics['expectancy']:+.2f}")
    lines.append(f"Max Drawdown     : ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)")
    lines.append(f"Sharpe Ratio     : {metrics['sharpe_ratio']:.2f}")
    lines.append(f"Commissions payees: ${metrics['total_commissions']:.2f}")

    panel_text = "\n".join(lines)
    console.print(Panel(panel_text, title=f"Backtest {config.symbol}", border_style="cyan"))

    console.print(f"\n  [bold {net_color}]Profit net : ${net:+.2f}[/bold {net_color}]")
    console.print(f"  [bold]Balance finale : ${metrics['final_balance']:,.2f}[/bold]\n")

    if metrics.get("session_stats"):
        table = Table(title="Performance par session")
        table.add_column("Session", style="cyan")
        table.add_column("Trades", justify="right")
        table.add_column("Winrate", justify="right")
        table.add_column("PF", justify="right")
        table.add_column("P/L Net", justify="right")

        for session, stats in metrics["session_stats"].items():
            pnl_color = "green" if stats["net_pnl"] > 0 else "red"
            table.add_row(
                session,
                str(stats["count"]),
                f"{stats['winrate']:.1f}%",
                f"{stats['profit_factor']:.2f}",
                f"[{pnl_color}]${stats['net_pnl']:+.2f}[/{pnl_color}]",
            )
        console.print(table)
