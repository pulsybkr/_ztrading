import pandas as pd
import numpy as np
from core.types import Trade, TradeResult, SessionType


def compute_metrics(trades: list[Trade], initial_balance: float) -> dict:
    if not trades:
        return {"error": "Aucun trade"}

    net_pnls = np.array([t.net_pnl for t in trades])
    gross_pnls = np.array([t.gross_pnl for t in trades])

    wins = net_pnls[net_pnls > 0]
    losses = net_pnls[net_pnls < 0]

    equity = np.cumsum(net_pnls) + initial_balance
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_dd = drawdown.min()
    max_dd_pct = (max_dd / peak[np.argmin(drawdown)]) * 100 if len(peak) > 0 else 0

    total_wins = wins.sum() if len(wins) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 1
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    if len(net_pnls) > 1 and net_pnls.std() > 0:
        sharpe = (net_pnls.mean() / net_pnls.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    total_commissions = sum(t.commission + t.spread_cost + t.slippage_cost for t in trades)

    session_stats = {}
    for session in SessionType:
        session_trades = [t for t in trades if t.signal.session == session]
        if session_trades:
            s_pnls = [t.net_pnl for t in session_trades]
            s_wins = [p for p in s_pnls if p > 0]
            s_losses = [p for p in s_pnls if p < 0]
            session_stats[session.value] = {
                "count": len(session_trades),
                "winrate": len(s_wins) / len(session_trades) * 100,
                "profit_factor": sum(s_wins) / abs(sum(s_losses)) if s_losses else float("inf"),
                "net_pnl": sum(s_pnls),
            }

    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "winrate": len(wins) / len(trades) * 100,
        "avg_win": wins.mean() if len(wins) > 0 else 0,
        "avg_loss": losses.mean() if len(losses) > 0 else 0,
        "expectancy": net_pnls.mean(),
        "profit_factor": profit_factor,
        "net_profit": net_pnls.sum(),
        "gross_profit": gross_pnls.sum(),
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "sharpe_ratio": sharpe,
        "total_commissions": total_commissions,
        "final_balance": initial_balance + net_pnls.sum(),
        "equity_curve": equity.tolist(),
        "session_stats": session_stats,
    }
