"""
Run backtest and save results to a text file.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import BacktestConfig
from backtest.engine import run_backtest
import json
from datetime import datetime


def run_and_save(start="2025-01-01", end="2025-12-31", timeframe="M5"):
    configs = {
        "1_baseline": BacktestConfig(
            signal_timeframe=timeframe,
            use_session_filter=False,
            use_atr_ratio_filter=False,
        ),
        "2_session_filter": BacktestConfig(
            signal_timeframe=timeframe,
            use_session_filter=True,
            use_atr_ratio_filter=False,
            allowed_sessions=["asian", "london", "overlap"],
        ),
        "3_session+atr_ratio": BacktestConfig(
            signal_timeframe=timeframe,
            use_session_filter=True,
            use_atr_ratio_filter=True,
            atr_ratio_threshold=0.15,
            allowed_sessions=["asian", "london", "overlap"],
        ),
    }

    all_results = {}

    for name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"  Running: {name} [{timeframe}]")
        print(f"{'='*60}")
        trades, metrics = run_backtest(config, start, end, verbose=True,
                                       show_trades=False, resolution="auto")

        if "error" in metrics:
            print(f"  ERROR: {metrics['error']}")
            all_results[name] = {"error": metrics["error"]}
            continue

        summary = {
            "total_trades": metrics["total_trades"],
            "winning_trades": metrics["winning_trades"],
            "losing_trades": metrics["losing_trades"],
            "winrate": round(metrics["winrate"], 1),
            "profit_factor": round(metrics["profit_factor"], 2),
            "net_profit": round(metrics["net_profit"], 2),
            "expectancy": round(metrics["expectancy"], 2),
            "avg_win": round(metrics["avg_win"], 2),
            "avg_loss": round(metrics["avg_loss"], 2),
            "max_drawdown": round(metrics["max_drawdown"], 2),
            "max_drawdown_pct": round(metrics["max_drawdown_pct"], 2),
            "sharpe_ratio": round(metrics["sharpe_ratio"], 2),
            "total_commissions": round(metrics["total_commissions"], 2),
            "final_balance": round(metrics["final_balance"], 2),
            "total_signals": metrics.get("total_signals_detected", 0),
            "signals_filtered": metrics.get("signals_filtered", 0),
            "resolution_stats": metrics.get("resolution_stats", {}),
            "session_stats": metrics.get("session_stats", {}),
        }
        all_results[name] = summary

        print(f"  Trades: {summary['total_trades']} | WR: {summary['winrate']}%")
        print(f"  PF: {summary['profit_factor']} | Net: ${summary['net_profit']}")
        print(f"  Max DD: ${summary['max_drawdown']} ({summary['max_drawdown_pct']}%)")
        print(f"  Sharpe: {summary['sharpe_ratio']} | Esperance: ${summary['expectancy']}/trade")

    # Save to file
    output_file = "backtest_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "period": f"{start} -> {end}",
            "results": all_results,
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':<25} {'Trades':>7} {'WR':>7} {'PF':>7} {'Net P/L':>10} {'Sharpe':>7}")
    print(f"  {'-'*63}")
    for name, r in all_results.items():
        if "error" in r:
            print(f"  {name:<25} ERROR")
        else:
            print(f"  {name:<25} {r['total_trades']:>7} {r['winrate']:>6}% {r['profit_factor']:>7} ${r['net_profit']:>9} {r['sharpe_ratio']:>7}")

    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--timeframe", default="M5")
    args = parser.parse_args()
    run_and_save(args.start, args.end, timeframe=args.timeframe.upper())
