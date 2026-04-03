"""
Moteur de backtest principal.
Orchestre : chargement données → détection signaux → résolution → rapport.
Supporte: timeframes M1, M5, M15, M30, H1 pour la détection de signaux.
Résolution: ticks > M1 > signal avec fallback automatique.
"""

import pandas as pd
from datetime import timedelta, datetime
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

# Durée en minutes par timeframe
TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60}


def run_backtest(
    config: BacktestConfig,
    start: str,
    end: str,
    data_dir: str = "data/parquet",
    verbose: bool = True,
    resolution: str = "auto",  # "ticks", "m1", "m5", "auto"
    show_trades: bool = True,  # False = n'affiche pas chaque trade (utile pour M1)
) -> tuple[list[Trade], dict]:
    """
    Lance un backtest complet, chunk par chunk.

    Args:
        config: Configuration du backtest
        start: Date début YYYY-MM-DD
        end: Date fin YYYY-MM-DD
        data_dir: Répertoire des données
        verbose: Affichage détaillé
        resolution: Force une résolution ("ticks", "m1", "m5") ou "auto" pour fallback

    Returns:
        (trades, metrics) - Liste des trades et dictionnaire de métriques
    """
    loader = DataLoader(data_dir)
    cost_model = CostModel(
        spread=config.spread,
        commission_per_lot=config.commission_per_lot,
        slippage_max=config.slippage,
    )

    # Durée d'une bougie en minutes (détermine le décalage signal → entrée)
    candle_minutes = TIMEFRAME_MINUTES.get(config.signal_timeframe, 5)

    # Load ML model — cherche d'abord model_{TF}.joblib, puis model.joblib (legacy M5)
    ml_model = None
    ml_feature_names = None
    if config.use_ml_filter:
        tf = config.signal_timeframe
        model_paths = [
            f"ml/models/model_{tf}.joblib",
            "ml/models/model.joblib",  # legacy M5
        ]
        loaded = False
        for model_path in model_paths:
            try:
                from ml.trainer import load_model
                ml_model, metadata = load_model(model_path)
                trained_tf = metadata.get("signal_timeframe", "M5")
                if trained_tf != tf:
                    print(f"  [yellow]Filtre ML ignoré : modèle entraîné sur {trained_tf}, "
                          f"timeframe actuel={tf}[/yellow]")
                    ml_model = None
                    break
                from ml.features import FEATURE_COLUMNS
                ml_feature_names = FEATURE_COLUMNS
                if verbose:
                    print(f"  Filtre ML [{tf}] activé (seuil={config.ml_threshold}, "
                          f"modèle: {model_path})")
                loaded = True
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"  [yellow]Filtre ML ignoré : {e}[/yellow]")
                break
        if not loaded and ml_model is None and config.use_ml_filter:
            print(f"  [yellow]Filtre ML ignoré : aucun modèle trouvé pour {tf} "
                  f"(entraîner avec: train run --timeframe {tf})[/yellow]")

    all_trades: list[Trade] = []
    total_signals = 0
    filtered_signals = 0
    resolution_stats = {"ticks": 0, "m1": 0, "signal": 0, "skipped": 0}

    in_position = False
    position_exit_time = None
    daily_trade_count = {}

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Backtest {config.symbol} | {start} → {end}")
        print(f"  Lot: {config.lot_size} | SL: {config.sl_atr_mult}x ATR")
        print(f"  Timeframe: {config.signal_timeframe} | Résolution: {resolution}")
        print(f"  Keltner: EMA({config.keltner_ema_period}), ATR({config.keltner_atr_period}), mult={config.keltner_multiplier}")
        print(f"  Regime: session={config.use_session_filter}, atr_ratio={config.use_atr_ratio_filter}")
        print(f"{'=' * 60}")

    for chunk in loader.iter_months(config.symbol, start, end, load_h1=True,
                                     signal_timeframe=config.signal_timeframe):
        if verbose:
            tick_status = "✅ ticks" if chunk.has_ticks else "❌ no ticks"
            m1_status = f"✅ M1({len(chunk.candles_m1)})" if chunk.has_m1 else "❌ no M1"
            sig_count = len(chunk.candles_signal)
            h1_status = f"✅ H1({len(chunk.candles_h1)})" if chunk.has_h1 else "❌ no H1"
            print(f"\n📊 {chunk.month} | {tick_status} | "
                  f"✅ {chunk.signal_timeframe}({sig_count}) | {m1_status} | {h1_status}")

        # Compute Keltner on signal timeframe once — used by breakout detection AND ML features
        candles_signal = compute_keltner(
            chunk.candles_signal,
            config.keltner_ema_period,
            config.keltner_atr_period,
            config.keltner_multiplier,
        )

        # Compute Keltner on H1 if needed (ATR ratio filter or ML features)
        candles_h1 = None
        if chunk.has_h1 and (config.use_atr_ratio_filter or config.use_ml_filter):
            candles_h1 = chunk.candles_h1
            if "atr" not in candles_h1.columns:
                candles_h1 = compute_keltner(
                    candles_h1,
                    config.keltner_ema_period,
                    config.keltner_atr_period,
                    1.0
                )

        # 1. Detect breakouts on signal timeframe (Keltner déjà calculé, pas recalculé en interne)
        signals = detect_breakouts(
            candles_signal,
            ema_period=config.keltner_ema_period,
            atr_period=config.keltner_atr_period,
            multiplier=config.keltner_multiplier,
            candle_minutes=candle_minutes,
            keltner_precomputed=True,  # Keltner déjà calculé ligne 124-129
        )
        total_signals += len(signals)

        # 2. Attach ATR ratios if needed
        if candles_h1 is not None and config.use_atr_ratio_filter:
            signals = attach_atr_ratios(signals, candles_signal, candles_h1)

        # 3. Apply regime filters
        signals_before_filter = len(signals)
        if config.use_session_filter or config.use_atr_ratio_filter:
            signals = apply_regime_filters(signals, config)

        # 4. Apply ML filter (M5 only — model trained on M5 features)
        if ml_model is not None and len(signals) > 0:
            from ml.features import build_features
            import pandas as pd
            kept = []
            for sig in signals:
                feats = build_features(sig, candles_signal, candles_h1)
                X = pd.DataFrame([feats])[ml_feature_names]
                proba = ml_model.predict_proba(X)[0][1]
                if proba >= config.ml_threshold:
                    kept.append(sig)
            signals = kept

        filtered_signals += signals_before_filter - len(signals)

        if verbose:
            print(f"   Signaux: {signals_before_filter} détectés → {len(signals)} après filtres (-{signals_before_filter - len(signals)})")

        # 4. Resolve each signal
        for signal in signals:
            # Check if already in position
            if in_position and position_exit_time and signal.time < position_exit_time:
                continue

            # Check daily trade limit
            trade_day = signal.time.date()
            day_key = str(trade_day)
            daily_trade_count.setdefault(day_key, 0)
            if daily_trade_count[day_key] >= config.max_daily_trades:
                continue

            # Resolution window: 4h after signal
            tick_start = signal.time
            tick_end = signal.time + timedelta(hours=4)

            trade = None
            used_resolution = None

            # Try resolution in priority order: ticks > M1 > M5
            if resolution in ("ticks", "auto") and chunk.has_ticks:
                ticks = chunk.get_ticks(tick_start, tick_end)
                if ticks is not None and len(ticks) > 10:
                    trade = resolve_with_ticks(signal, ticks, config, cost_model)
                    used_resolution = "ticks"

            # M1 resolution — disponible seulement si le signal n'est pas déjà M1
            if trade is None and resolution in ("m1", "auto") and chunk.has_m1:
                m1_window = chunk.candles_m1[
                    (chunk.candles_m1["time"] >= tick_start) &
                    (chunk.candles_m1["time"] <= tick_end)
                ]
                if len(m1_window) > 0:
                    trade = resolve_with_candles(signal, m1_window, config, cost_model)
                    used_resolution = "m1"

            # Fallback sur les bougies signal (M1, M5, M15... selon config)
            if trade is None and resolution in ("m5", "auto"):
                sig_mask = (
                    (candles_signal["time"] >= tick_start) &
                    (candles_signal["time"] <= tick_end)
                )
                sig_window = candles_signal.loc[sig_mask].reset_index(drop=True)
                if len(sig_window) > 0:
                    trade = resolve_with_candles(signal, sig_window, config, cost_model)
                    used_resolution = "signal"

            if trade is None:
                resolution_stats["skipped"] += 1
                continue

            resolution_stats[used_resolution if used_resolution in resolution_stats else "signal"] += 1
            all_trades.append(trade)
            in_position = True
            position_exit_time = trade.exit_time
            daily_trade_count[day_key] += 1

            if verbose and show_trades:
                color = "green" if trade.net_pnl > 0 else "red"
                dir_str = "LONG" if trade.direction == Direction.LONG else "SHORT"
                res_tag = f"[{used_resolution}]"
                print(f"   [{color}]{dir_str}[/{color}] {trade.entry_time.strftime('%m-%d %H:%M')} "
                      f"→ {trade.exit_time.strftime('%H:%M')} | "
                      f"P/L: ${trade.net_pnl:+.2f} {res_tag}")

    # Compute metrics
    metrics = compute_metrics(all_trades, config.initial_balance)

    # Add extra metadata to metrics
    metrics["total_signals_detected"] = total_signals
    metrics["signals_filtered"] = filtered_signals
    metrics["resolution_stats"] = resolution_stats
    metrics["config"] = {
        "signal_timeframe": config.signal_timeframe,
        "keltner_ema": config.keltner_ema_period,
        "keltner_mult": config.keltner_multiplier,
        "sl_atr_mult": config.sl_atr_mult,
        "trailing_atr_mult": config.trailing_atr_mult,
        "breakeven_atr_mult": config.breakeven_atr_mult,
        "session_filter": config.use_session_filter,
        "atr_ratio_filter": config.use_atr_ratio_filter,
    }

    if verbose:
        # N'affiche M1 fine-résolution que si le signal n'est pas déjà M1
        m1_part = f"M1={resolution_stats['m1']}, " if config.signal_timeframe != "M1" else ""
        print(f"\n📈 Résolution utilisée: ticks={resolution_stats['ticks']}, "
              f"{m1_part}{config.signal_timeframe}={resolution_stats['signal']}, "
              f"skip={resolution_stats['skipped']}")
        print_report(all_trades, metrics, config)

    return all_trades, metrics


def _backtest_worker(args: tuple):
    """
    Worker picklable pour ProcessPoolExecutor.
    Doit être défini au niveau module (pas de lambda ni fonction imbriquée).
    """
    name, config, start, end, data_dir, resolution = args
    try:
        trades, metrics = run_backtest(
            config, start, end,
            data_dir=data_dir,
            verbose=False,
            resolution=resolution,
        )
        return name, trades, metrics
    except Exception as e:
        return name, [], {"error": str(e)}


def run_parallel_backtests(
    named_configs: dict,
    start: str,
    end: str,
    data_dir: str = "data/parquet",
    resolution: str = "auto",
    max_workers: int = None,
) -> dict:
    """
    Lance plusieurs backtests en parallèle via ProcessPoolExecutor.

    Args:
        named_configs: {"nom": BacktestConfig, ...}
        max_workers: None = nb de cœurs CPU disponibles

    Returns:
        {"nom": {"trades": [...], "metrics": {...}}, ...}
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if max_workers is None:
        max_workers = min(len(named_configs), os.cpu_count() or 4)

    args_list = [
        (name, config, start, end, data_dir, resolution)
        for name, config in named_configs.items()
    ]

    results = {}
    print(f"\n⚡ Lancement de {len(args_list)} backtests en parallèle ({max_workers} workers)...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_backtest_worker, args): args[0] for args in args_list}
        for future in as_completed(futures):
            name, trades, metrics = future.result()
            results[name] = {"trades": trades, "metrics": metrics}
            status = "✅" if "error" not in metrics else "❌"
            n = metrics.get("total_trades", 0)
            pnl = metrics.get("net_profit", 0)
            print(f"  {status} {name} → {n} trades | P/L: ${pnl:+.2f}")

    # Remettre dans l'ordre original
    return {name: results[name] for name in named_configs if name in results}


def run_comparison(
    start: str,
    end: str,
    data_dir: str = "data/parquet",
    resolution: str = "auto",
    signal_timeframe: str = "M5",
    max_workers: int = None,
) -> dict:
    """
    Lance les backtests comparatifs séquentiellement :
    1. Baseline (Keltner seul)
    2. + Filtre session
    3. + Filtre session + ATR ratio

    Note: séquentiel par design — évite les blocages Windows avec Numba+multiprocessing.
    Utiliser run_parallel_backtests() directement pour les grid search.

    Returns:
        dict avec les résultats de chaque configuration
    """
    from rich.console import Console
    from rich.table import Table
    console = Console()

    configs = {
        "1_baseline": BacktestConfig(
            signal_timeframe=signal_timeframe,
            use_session_filter=False,
            use_atr_ratio_filter=False,
        ),
        "2_session": BacktestConfig(
            signal_timeframe=signal_timeframe,
            use_session_filter=True,
            use_atr_ratio_filter=False,
        ),
        "3_session+atr": BacktestConfig(
            signal_timeframe=signal_timeframe,
            use_session_filter=True,
            use_atr_ratio_filter=True,
            atr_ratio_threshold=0.15,
        ),
    }

    results = {}
    total = len(configs)

    for i, (name, config) in enumerate(configs.items(), 1):
        console.print(f"\n[bold cyan][{i}/{total}] {name}[/bold cyan]")
        trades, metrics = run_backtest(
            config, start, end,
            data_dir=data_dir,
            verbose=True,
            show_trades=False,  # Pas de trade par trade dans le comparatif
            resolution=resolution,
        )
        results[name] = {"trades": trades, "metrics": metrics}
        if "error" not in metrics:
            console.print(f"  ✅ {metrics['total_trades']} trades | "
                          f"PF={metrics['profit_factor']:.2f} | "
                          f"P/L=${metrics['net_profit']:+.2f}")

    # Summary table
    console.print(f"\n\n{'=' * 80}")
    console.print("[bold]  COMPARAISON DES CONFIGURATIONS[/bold]")
    console.print(f"{'=' * 80}\n")

    table = Table(title=f"Backtest {start} → {end}")
    table.add_column("Config", style="cyan", min_width=18)
    table.add_column("Trades", justify="right")
    table.add_column("Winrate", justify="right")
    table.add_column("PF", justify="right")
    table.add_column("Net P/L", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Espérance", justify="right")

    for name, data in results.items():
        m = data["metrics"]
        if "error" in m:
            table.add_row(name, "0", "-", "-", "-", "-", "-", "-")
            continue

        pnl = m["net_profit"]
        pnl_color = "green" if pnl > 0 else "red"
        table.add_row(
            name,
            str(m["total_trades"]),
            f"{m['winrate']:.1f}%",
            f"{m['profit_factor']:.2f}",
            f"[{pnl_color}]${pnl:+.2f}[/{pnl_color}]",
            f"${m['max_drawdown']:.2f}",
            f"{m['sharpe_ratio']:.2f}",
            f"${m['expectancy']:+.2f}",
        )

    console.print(table)

    return results
