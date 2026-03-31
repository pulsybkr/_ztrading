import pandas as pd
from datetime import timedelta
from typing import Optional

from core.types import Trade, Signal, BacktestConfig, Direction
from data.loader import DataLoader
from strategy.breakout import detect_breakouts
from strategy.regime import apply_regime_filters, attach_atr_ratios
from backtest.resolver import resolve_with_ticks, resolve_with_candles
from backtest.costs import CostModel
from backtest.metrics import compute_metrics
from backtest.report import print_report
from strategy.keltner import compute_keltner


def run_backtest(
    config: BacktestConfig,
    start: str,
    end: str,
    data_dir: str = "data/parquet",
    verbose: bool = True,
) -> tuple[list[Trade], dict]:
    loader = DataLoader(data_dir)
    cost_model = CostModel(
        spread=config.spread,
        commission_per_lot=config.commission_per_lot,
        slippage_max=config.slippage,
    )

    all_trades: list[Trade] = []
    total_signals = 0
    filtered_signals = 0

    in_position = False
    position_exit_time = None

    for chunk in loader.iter_months(config.symbol, start, end):
        if verbose:
            print(f"\nTraitement {chunk.month}...")

        # Load H1 candles if ATR ratio filter is needed
        candles_h1 = None
        if config.use_atr_ratio_filter:
            try:
                candles_h1 = loader.load_candles(
                    config.symbol, "H1",
                    str(chunk.candles_m5["time"].min()),
                    str(chunk.candles_m5["time"].max())
                )
                # Compute ATR on H1 for ratio calculation
                if "atr" not in candles_h1.columns:
                    candles_h1 = compute_keltner(candles_h1, config.keltner_ema_period, config.keltner_atr_period, 1.0)
            except Exception:
                candles_h1 = None

        signals = detect_breakouts(
            chunk.candles_m5,
            ema_period=config.keltner_ema_period,
            atr_period=config.keltner_atr_period,
            multiplier=config.keltner_multiplier,
        )
        total_signals += len(signals)

        # Attach ATR ratios if needed
        if candles_h1 is not None:
            signals = attach_atr_ratios(signals, chunk.candles_m5, candles_h1)

        # Apply regime filters
        if config.use_session_filter or config.use_atr_ratio_filter:
            signals_before = len(signals)
            signals = apply_regime_filters(signals, config)
            filtered_signals += signals_before - len(signals)

        if verbose:
            print(f"   Signaux detectes: {len(signals)}")

        for signal in signals:
            if in_position and position_exit_time and signal.time < position_exit_time:
                continue

            tick_start = signal.time
            tick_end = signal.time + timedelta(hours=4)

            ticks = chunk.get_ticks(tick_start, tick_end)

            if ticks is not None and len(ticks) > 10:
                trade = resolve_with_ticks(signal, ticks, config, cost_model)
            elif chunk.candles_m1 is not None and len(chunk.candles_m1) > 0:
                m1_window = chunk.candles_m1[
                    (chunk.candles_m1["time"] >= tick_start) &
                    (chunk.candles_m1["time"] <= tick_end)
                ]
                if len(m1_window) > 0:
                    trade = resolve_with_candles(signal, m1_window, config, cost_model)
                else:
                    continue
            else:
                # Fallback: use M5 candles to create pseudo-ticks
                m5_mask = (chunk.candles_m5["time"] >= tick_start) & (chunk.candles_m5["time"] <= tick_end)
                m5_window = chunk.candles_m5.loc[m5_mask].reset_index(drop=True)
                if len(m5_window) > 0:
                    trade = resolve_with_candles(signal, m5_window, config, cost_model)
                else:
                    continue

            all_trades.append(trade)
            in_position = True
            position_exit_time = trade.exit_time

            if verbose:
                emoji = "green" if trade.net_pnl > 0 else "red"
                dir_str = "LONG" if trade.direction == Direction.LONG else "SHORT"
                print(f"   [{emoji}]{dir_str}[/{emoji}] {trade.entry_time.strftime('%m-%d %H:%M')} "
                      f"-> {trade.exit_time.strftime('%H:%M')} | "
                      f"P/L: ${trade.net_pnl:+.2f}")

    metrics = compute_metrics(all_trades, config.initial_balance)

    if verbose:
        print_report(all_trades, metrics, config)

    return all_trades, metrics
