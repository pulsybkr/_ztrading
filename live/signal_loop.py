import time
import pandas as pd
from datetime import datetime, timezone
from typing import Optional
import threading

from core.types import Signal, Direction, BacktestConfig
from strategy.breakout import detect_breakouts, detect_session
from strategy.keltner import compute_keltner
from strategy.regime import apply_regime_filters, attach_atr_ratios
from live.mt5_bridge import MT5Bridge
from live.db import LiveDB
from rich.console import Console
from rich.table import Table

console = Console()

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


class SignalLoop:
    def __init__(self, config: BacktestConfig, bridge: MT5Bridge):
        self.config = config
        self.bridge = bridge
        self.running = False
        self.in_position = False
        self.paper_mode = False
        self.position_ticket = None
        self.last_signal_time = None
        self.check_interval = 60
        self._trailing_thread: Optional[threading.Thread] = None
        self._ml_model = None
        self._ml_feature_names = None
        self.db: Optional[LiveDB] = None

    def _load_ml_model(self):
        if not self.config.use_ml_filter:
            return
        tf = getattr(self.config, "signal_timeframe", "M5")
        paths = [f"ml/models/model_{tf}.joblib", "ml/models/model.joblib"]
        for path in paths:
            try:
                from ml.trainer import load_model
                model, metadata = load_model(path)
                trained_tf = metadata.get("signal_timeframe", tf)
                if trained_tf != tf:
                    console.print(f"[yellow]⚠ ML: modèle entraîné sur {trained_tf}, signaux en {tf}[/yellow]")
                self._ml_model = model
                self._ml_feature_names = metadata.get("feature_names")
                console.print(f"[green]✓ ML chargé:[/green] {path} (precision={metadata.get('precision', '?')})")
                return
            except Exception:
                continue
        console.print(f"[yellow]⚠ Aucun modèle pour {tf} — filtre ML désactivé[/yellow]")
        self.config.use_ml_filter = False

    def _start_trailing_thread(self):
        def _worker():
            while self.running and self.in_position and not self.paper_mode:
                try:
                    updated = self.bridge.update_trailing_stops(db=self.db)
                    if updated:
                        console.print(f"[dim]{_ts()}[/dim] [cyan]{updated} SL mis à jour[/cyan]")
                except Exception as e:
                    console.print(f"[red]⚠ Trailing error: {e}[/red]")
                time.sleep(5)

        self._trailing_thread = threading.Thread(target=_worker, daemon=True)
        self._trailing_thread.start()

    def _print_session_header(self, mode: str, account: dict):
        tf = getattr(self.config, "signal_timeframe", "M5")

        table = Table(title=f"NovaGold — Session {'PAPER' if mode == 'paper' else '🔴 LIVE'}", show_header=False, border_style="gold1")
        table.add_column("Clé", style="dim", width=22)
        table.add_column("Valeur", style="bold")
        table.add_row("Compte",    f"{account.get('name')} (#{account.get('login')})")
        table.add_row("Serveur",   account.get('server', '?'))
        table.add_row("Balance",   f"{account.get('currency')} {account.get('balance', 0):.2f}")
        table.add_row("Levier",    f"1:{account.get('leverage', '?')}")
        table.add_row("Symbole",   self.config.symbol)
        table.add_row("Timeframe", tf)
        table.add_row("Lot",       str(self.config.lot_size))
        table.add_row("Keltner",   f"EMA{self.config.keltner_ema_period} × {self.config.keltner_multiplier}")
        table.add_row("SL",        f"{self.config.sl_atr_mult}× ATR")
        table.add_row("Breakeven", f"{self.config.breakeven_atr_mult}× ATR")
        table.add_row("Trailing",  f"{self.config.trailing_atr_mult}× ATR")
        table.add_row("Sessions",  ", ".join(self.config.allowed_sessions) if self.config.use_session_filter else "toutes")
        table.add_row("Filtre ML", f"seuil {self.config.ml_threshold}" if self.config.use_ml_filter else "désactivé")
        console.print(table)

    def start(self, paper: bool = True, regime_filter: bool = True):
        self.running = True
        self.paper_mode = paper
        self._load_ml_model()

        # Init DB + session
        self.db = LiveDB()
        try:
            account = self.bridge.get_account_info()
        except Exception:
            account = {}

        mode = "paper" if paper else "live"
        session_id = self.db.open_session(mode, account, self.config)
        self._print_session_header(mode, account)
        console.print(f"[dim]Session #{session_id} enregistrée — DB: live/data/live_sessions.db[/dim]\n")
        console.print(f"[bold]Démarrage boucle (intervalle={self.check_interval}s) — Ctrl+C pour arrêter[/bold]\n")

        try:
            while self.running:
                try:
                    self._check_and_execute(paper, regime_filter)
                    time.sleep(self.check_interval)
                except Exception as e:
                    console.print(f"[red]⚠ Erreur loop: {e}[/red]")
                    time.sleep(10)
        finally:
            self._on_stop()

    def stop(self):
        self.running = False

    def _on_stop(self):
        console.print(f"\n[bold]Arrêt de la session...[/bold]")
        if self.db:
            try:
                info = self.bridge.get_account_info()
                self.db.close_session(info.get("balance", 0), info.get("equity", 0))
            except Exception:
                self.db.close_session(0, 0)
            self.db.close()
            console.print("[dim]Session fermée et sauvegardée.[/dim]")

    def _check_and_execute(self, paper: bool, regime_filter: bool):
        if not self.bridge.is_connected():
            console.print(f"[yellow]{_ts()} MT5 déconnecté, reconnexion...[/yellow]")
            if not self.bridge.connect():
                return

        if self.in_position:
            self._check_position()
            return

        signal, opp_id = self._detect_signal(regime_filter)
        if signal is None:
            return

        color = "green" if signal.direction == Direction.LONG else "red"
        console.print(f"[{color}][{_ts()}] ▶ Signal {signal.direction.name}[/{color}] @ {signal.price:.2f} | ATR={signal.atr:.3f} | Session={signal.session.value}")

        if paper:
            console.print(f"[yellow][PAPER] Ordre simulé: {signal.direction.name} @ {signal.price:.2f}[/yellow]")
            self.in_position = True
            self.last_signal_time = signal.time
            if self.db:
                # Paper: log trade without ticket
                self.db.open_trade(
                    ticket=0,
                    direction=signal.direction.name,
                    entry_price=signal.price,
                    sl=signal.price - self.config.sl_atr_mult * signal.atr if signal.direction == Direction.LONG
                       else signal.price + self.config.sl_atr_mult * signal.atr,
                    lot=self.config.lot_size,
                )
        else:
            ticket = self.bridge.execute_signal(signal)
            if ticket:
                self.in_position = True
                self.position_ticket = ticket
                self.last_signal_time = signal.time
                if self.db:
                    positions = self.bridge.get_open_positions()
                    sl = next((p.sl for p in positions if p.ticket == ticket), 0.0)
                    self.db.open_trade(
                        ticket=ticket,
                        direction=signal.direction.name,
                        entry_price=signal.price,
                        sl=sl,
                        lot=self.config.lot_size,
                    )
                self._start_trailing_thread()

    def _detect_signal(self, regime_filter: bool):
        """Detect and filter signals. Returns (signal, opp_id) or (None, None).
        Logs all opportunities (taken and rejected) to DB."""
        try:
            tf = getattr(self.config, "signal_timeframe", "M5")
            df = self.bridge.get_candles(self.config.symbol, tf, count=100)

            if len(df) < 25:
                return None, None

            df = compute_keltner(
                df,
                ema_period=self.config.keltner_ema_period,
                atr_period=self.config.keltner_atr_period,
                multiplier=self.config.keltner_multiplier,
            )

            signals = detect_breakouts(
                df,
                ema_period=self.config.keltner_ema_period,
                atr_period=self.config.keltner_atr_period,
                multiplier=self.config.keltner_multiplier,
            )

            if not signals:
                return None, None

            latest = signals[-1]

            if self.last_signal_time and latest.time <= self.last_signal_time:
                return None, None

            # Get KC bands for logging
            idx = latest.candle_index
            kc_upper = float(df.iloc[idx].get("kc_upper", 0)) if "kc_upper" in df.columns else None
            kc_lower = float(df.iloc[idx].get("kc_lower", 0)) if "kc_lower" in df.columns else None

            # ── Session filter ────────────────────────────────────────
            if regime_filter and self.config.use_session_filter:
                if latest.session.value not in self.config.allowed_sessions:
                    reason = f"session_filtered ({latest.session.value} ∉ {self.config.allowed_sessions})"
                    console.print(f"[dim]{_ts()}[/dim] [dim]✗ {latest.direction.name} @ {latest.price:.2f} — {reason}[/dim]")
                    if self.db:
                        self.db.log_opportunity(latest, taken=False, rejection_reason=reason,
                                                kc_upper=kc_upper, kc_lower=kc_lower)
                    return None, None

            # ── ML filter ────────────────────────────────────────────
            ml_proba = None
            if self.config.use_ml_filter and self._ml_model is not None:
                try:
                    from ml.features import build_features, FEATURE_COLUMNS
                    import pandas as pd
                    features = build_features(latest, df)
                    X = pd.DataFrame([[features[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
                    ml_proba = float(self._ml_model.predict_proba(X)[0][1])
                    if ml_proba < self.config.ml_threshold:
                        reason = f"ml_filtered (proba={ml_proba:.3f} < {self.config.ml_threshold})"
                        console.print(f"[dim]{_ts()}[/dim] [dim]✗ {latest.direction.name} @ {latest.price:.2f} — {reason}[/dim]")
                        if self.db:
                            self.db.log_opportunity(latest, taken=False, rejection_reason=reason,
                                                    ml_proba=ml_proba, ml_threshold=self.config.ml_threshold,
                                                    kc_upper=kc_upper, kc_lower=kc_lower)
                        return None, None
                    console.print(f"[dim]{_ts()}[/dim] [green]✓ ML proba={ml_proba:.3f}[/green]")
                except Exception as e:
                    console.print(f"[yellow]⚠ ML erreur: {e} — signal accepté[/yellow]")

            # ── Signal accepté ───────────────────────────────────────
            opp_id = None
            if self.db:
                opp_id = self.db.log_opportunity(
                    latest, taken=True,
                    ml_proba=ml_proba, ml_threshold=self.config.ml_threshold if self.config.use_ml_filter else None,
                    kc_upper=kc_upper, kc_lower=kc_lower,
                )

            return latest, opp_id

        except Exception as e:
            console.print(f"[red]⚠ Erreur détection: {e}[/red]")
            return None, None

    def _check_position(self):
        if self.paper_mode:
            console.print(f"[dim]{_ts()} [PAPER] Position simulée active[/dim]")
            return

        try:
            positions = self.bridge.get_open_positions()
            if not positions:
                console.print(f"[{_ts()}] [cyan]■ Position fermée (SL/TP ou fermeture manuelle)[/cyan]")

                # Try to get exit info from deal history
                exit_price = 0.0
                exit_reason = "unknown"
                pnl_net = None
                try:
                    import MetaTrader5 as mt5
                    from datetime import timedelta
                    deals = mt5.history_deals_get(
                        datetime.now(timezone.utc) - timedelta(hours=1),
                        datetime.now(timezone.utc),
                    )
                    if deals:
                        last_deal = sorted(deals, key=lambda d: d.time)[-1]
                        exit_price = last_deal.price
                        pnl_net = round(last_deal.profit - last_deal.commission - last_deal.swap, 2)
                        exit_reason = "sl" if last_deal.reason == mt5.DEAL_REASON_SL else \
                                      "tp" if last_deal.reason == mt5.DEAL_REASON_TP else "manual"
                        color = "green" if pnl_net >= 0 else "red"
                        console.print(f"  Sortie @ {exit_price:.2f} | Raison={exit_reason} | PnL=[{color}]{'+' if pnl_net >= 0 else ''}{pnl_net}[/{color}]")
                except Exception:
                    pass

                if self.db:
                    self.db.close_trade(exit_price, exit_reason, pnl_net=pnl_net)

                self.in_position = False
                self.position_ticket = None
        except Exception as e:
            console.print(f"[red]⚠ Erreur gestion position: {e}[/red]")


def run_signal_loop(config: BacktestConfig, paper: bool = True,
                   regime_filter: bool = True, login: int = None,
                   password: str = None, server: str = None):
    bridge = MT5Bridge(config)

    if not bridge.connect(login, password, server):
        console.print("[red]Impossible de se connecter à MT5[/red]")
        return

    loop = SignalLoop(config, bridge)
    loop.start(paper=paper, regime_filter=regime_filter)
