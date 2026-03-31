import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import threading

from core.types import Signal, Direction, BacktestConfig
from data.loader import DataLoader
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
        self.position_ticket = None
        self.last_signal_time = None
        self.check_interval = 60

    def start(self, paper: bool = True, regime_filter: bool = True):
        print(f"Demarrage du signal loop (paper={paper})...")
        self.running = True

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

    def _detect_signal(self, regime_filter: bool) -> Optional[Signal]:
        try:
            loader = DataLoader()
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=2)

            df = loader.load_candles(
                self.config.symbol, "M5",
                start_time.strftime("%Y-%m-%d"),
                end_time.strftime("%Y-%m-%d")
            )

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
                allowed_sessions = [s for s in self.config.allowed_sessions]
                if latest_signal.session.value not in allowed_sessions:
                    print(f"Signal filtre (session): {latest_signal.session.value}")
                    return None

            return latest_signal

        except Exception as e:
            print(f"Erreur detection: {e}")
            return None

    def _check_position(self):
        positions = self.bridge.get_open_positions()
        if not positions:
            print("Position fermee")
            self.in_position = False
            self.position_ticket = None
            return

        if self.bridge.config.max_positions == 1:
            self.bridge.update_trailing_stops()


def run_signal_loop(config: BacktestConfig, paper: bool = True,
                   regime_filter: bool = True, login: int = None,
                   password: str = None, server: str = None):
    bridge = MT5Bridge(config)

    if not bridge.connect(login, password, server):
        print("Impossible de se connecter a MT5")
        return

    loop = SignalLoop(config, bridge)
    loop.start(paper=paper, regime_filter=regime_filter)
