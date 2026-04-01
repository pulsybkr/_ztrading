import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import threading

from core.types import Signal, Direction, BacktestConfig
from strategy.breakout import detect_breakouts, detect_session
from strategy.keltner import compute_keltner
from strategy.regime import apply_regime_filters, attach_atr_ratios
from live.mt5_bridge import MT5Bridge


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

    def _load_ml_model(self):
        """Load ML model for the configured timeframe if available."""
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
                    print(f"[ML] Attention: modele entraine sur {trained_tf}, signaux en {tf}")
                self._ml_model = model
                self._ml_feature_names = metadata.get("feature_names")
                print(f"[ML] Modele charge: {path} (precision={metadata.get('precision', '?')})")
                return
            except Exception:
                continue
        print(f"[ML] Aucun modele trouve pour {tf} — filtre ML desactive")
        self.config.use_ml_filter = False

    def _start_trailing_thread(self):
        """Start background thread that updates trailing SL every 5 seconds."""
        def _worker():
            while self.running and self.in_position and not self.paper_mode:
                try:
                    updated = self.bridge.update_trailing_stops()
                    if updated:
                        print(f"[Trailing] {updated} SL mis a jour")
                except Exception as e:
                    print(f"[Trailing] Erreur: {e}")
                time.sleep(5)

        self._trailing_thread = threading.Thread(target=_worker, daemon=True)
        self._trailing_thread.start()

    def start(self, paper: bool = True, regime_filter: bool = True):
        print(f"Demarrage du signal loop (paper={paper})...")
        self.running = True
        self.paper_mode = paper
        self._load_ml_model()

        while self.running:
            try:
                self._check_and_execute(paper, regime_filter)
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Erreur loop: {e}")
                time.sleep(10)

    def stop(self):
        print("Arret du signal loop...")
        self.running = False

    def _check_and_execute(self, paper: bool, regime_filter: bool):
        if not self.bridge.is_connected():
            print("MT5 pas connecte, reconnexion...")
            if not self.bridge.connect():
                return

        if self.in_position:
            self._check_position()
            return

        signal = self._detect_signal(regime_filter)
        if signal is None:
            return

        print(f"Signal detecte: {signal.direction.name} @ {signal.price:.2f}")

        if paper:
            print(f"[PAPER] Ordre simule: {signal.direction.name} @ {signal.price:.2f}")
            self.in_position = True
            self.last_signal_time = signal.time
        else:
            ticket = self.bridge.execute_signal(signal)
            if ticket:
                self.in_position = True
                self.position_ticket = ticket
                self.last_signal_time = signal.time
                # Start trailing stop thread
                self._start_trailing_thread()

    def _detect_signal(self, regime_filter: bool) -> Optional[Signal]:
        try:
            tf = getattr(self.config, "signal_timeframe", "M5")
            df = self.bridge.get_candles(self.config.symbol, tf, count=100)

            if len(df) < 25:
                return None

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
                return None

            latest_signal = signals[-1]

            if self.last_signal_time and latest_signal.time <= self.last_signal_time:
                return None

            if regime_filter and self.config.use_session_filter:
                if latest_signal.session.value not in self.config.allowed_sessions:
                    print(f"Signal filtre (session): {latest_signal.session.value}")
                    return None

            # ML filter
            if self.config.use_ml_filter and self._ml_model is not None:
                try:
                    from ml.features import build_features
                    features = build_features(latest_signal, df)
                    proba = self._ml_model.predict_proba([features])[0][1]
                    if proba < self.config.ml_threshold:
                        print(f"Signal filtre (ML): proba={proba:.3f} < {self.config.ml_threshold}")
                        return None
                    print(f"[ML] Signal valide: proba={proba:.3f}")
                except Exception as e:
                    print(f"[ML] Erreur filtre: {e} — signal accepte quand meme")

            return latest_signal

        except Exception as e:
            print(f"Erreur detection: {e}")
            return None

    def _check_position(self):
        # Paper mode: no real MT5 position — keep in_position until manually reset
        if self.paper_mode:
            print("[PAPER] Position simulee active")
            return

        try:
            positions = self.bridge.get_open_positions()
            if not positions:
                print("Position fermee (SL/TP atteint ou fermeture manuelle)")
                self.in_position = False
                self.position_ticket = None
                # trailing thread will stop on its own (checks self.in_position)
        except Exception as e:
            print(f"Erreur gestion position: {e}")


def run_signal_loop(config: BacktestConfig, paper: bool = True,
                   regime_filter: bool = True, login: int = None,
                   password: str = None, server: str = None):
    bridge = MT5Bridge(config)

    if not bridge.connect(login, password, server):
        print("Impossible de se connecter a MT5")
        return

    loop = SignalLoop(config, bridge)
    loop.start(paper=paper, regime_filter=regime_filter)
