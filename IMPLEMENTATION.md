# NovaGold Reborn — Code d'implémentation

> Tous les bouts de code critiques du projet, dans l'ordre d'implémentation.

---

## Table des matières

1. [Types & constantes](#1-types--constantes)
2. [Export MT5 → Parquet](#2-export-mt5--parquet)
3. [Chargement chunk par chunk](#3-chargement-chunk-par-chunk)
4. [Keltner Channel](#4-keltner-channel)
5. [Détection Breakout](#5-détection-breakout)
6. [Modèle de coûts](#6-modèle-de-coûts)
7. [Résolution tick par tick (SL/BE/Trailing)](#7-résolution-tick-par-tick)
8. [Moteur de backtest](#8-moteur-de-backtest)
9. [Rapport de performance](#9-rapport-de-performance)
10. [CLI principal](#10-cli-principal)
11. [Filtres de régime (Phase 2)](#11-filtres-de-régime-phase-2)
12. [Feature engineering (Phase 3)](#12-feature-engineering-phase-3)
13. [Walk-forward LightGBM (Phase 4)](#13-walk-forward-lightgbm-phase-4)
14. [Bridge MT5 live (Phase 5)](#14-bridge-mt5-live-phase-5)

---

## 1. Types & constantes

```python
# core/types.py

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Direction(Enum):
    LONG = 1
    SHORT = -1


class TradeResult(Enum):
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


class SessionType(Enum):
    SGE_OPEN = "sge_open"       # 01:30 - 03:00 UTC
    LONDON = "london"           # 08:00 - 10:30 UTC
    OVERLAP = "overlap"         # 12:00 - 16:30 UTC
    OFF_SESSION = "off_session" # Tout le reste


@dataclass
class Signal:
    """Un signal de breakout détecté."""
    time: datetime
    direction: Direction
    price: float               # Prix de la cassure
    keltner_band: float        # Niveau de la bande cassée
    atr: float                 # ATR(14) au moment du signal
    session: SessionType
    candle_index: int          # Index dans le DataFrame


@dataclass
class Trade:
    """Un trade résolu (après simulation)."""
    signal: Signal
    entry_price: float         # Prix d'entrée réel (avec spread/slippage)
    exit_price: float          # Prix de sortie réel
    entry_time: datetime
    exit_time: datetime
    direction: Direction
    lot_size: float
    sl_initial: float          # Niveau SL initial
    result: TradeResult

    # Financier (tout en dollars)
    gross_pnl: float = 0.0     # P/L brut
    commission: float = 0.0    # Commission payée
    spread_cost: float = 0.0   # Coût du spread à l'entrée
    slippage_cost: float = 0.0
    net_pnl: float = 0.0      # P/L net (gross - costs)

    # Méta
    max_favorable: float = 0.0    # MFE (Max Favorable Excursion)
    max_adverse: float = 0.0      # MAE (Max Adverse Excursion)
    breakeven_activated: bool = False
    n_trailing_updates: int = 0

    def __post_init__(self):
        self.net_pnl = self.gross_pnl - self.commission - self.spread_cost - self.slippage_cost


@dataclass
class BacktestConfig:
    """Configuration complète d'un backtest."""
    symbol: str = "XAUUSD"
    lot_size: float = 0.01
    initial_balance: float = 10_000.0
    point_value: float = 0.01
    contract_size: float = 100  # 1 lot = 100 oz

    # Keltner
    keltner_ema_period: int = 20
    keltner_atr_period: int = 14
    keltner_multiplier: float = 2.0

    # Sorties
    sl_atr_mult: float = 2.0
    breakeven_atr_mult: float = 1.0
    trailing_atr_mult: float = 1.5
    trailing_step_points: int = 10

    # Coûts
    spread: float = 0.20
    commission_per_lot: float = 3.50
    slippage: float = 0.05

    # Filtres
    use_session_filter: bool = False
    use_atr_ratio_filter: bool = False
    atr_ratio_threshold: float = 0.15
    allowed_sessions: list = field(default_factory=lambda: ["sge_open", "london", "overlap"])

    # ML (Phase 4)
    use_ml_filter: bool = False
    ml_threshold: float = 0.62

    @property
    def dollar_per_point(self) -> float:
        """Combien de $ par mouvement de 1 point (0.01) pour la taille de lot."""
        # 0.01 lot × 100 oz × 0.01 = $0.01 par point
        # Simplifié : 0.01 lot = 1$ par mouvement de $1 sur l'or
        return self.lot_size * self.contract_size * self.point_value
```

```python
# core/constants.py

from datetime import time

# Sessions en UTC
SESSIONS = {
    "sge_open": (time(1, 30), time(3, 0)),
    "london":   (time(8, 0),  time(10, 30)),
    "overlap":  (time(12, 0), time(16, 30)),
}

# XAU/USD specs
XAUUSD = {
    "symbol": "XAUUSD",
    "point": 0.01,
    "digits": 2,
    "contract_size": 100,  # 1 lot = 100 oz
    "tick_size": 0.01,
}
```

---

## 2. Export MT5 → Parquet

```python
# data/mt5_export.py
"""
Export des données depuis MetaTrader 5 vers Parquet.
À exécuter sur un PC Windows avec MT5 installé et connecté.
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def init_mt5(login: int = None, password: str = None, server: str = None) -> bool:
    """Initialise la connexion MT5."""
    if not mt5.initialize():
        print(f"❌ MT5 init failed: {mt5.last_error()}")
        return False

    if login and password and server:
        if not mt5.login(login, password=password, server=server):
            print(f"❌ MT5 login failed: {mt5.last_error()}")
            return False

    info = mt5.account_info()
    print(f"✅ Connecté: {info.server} | Compte: {info.login} | Balance: ${info.balance:.2f}")
    return True


def export_ticks(symbol: str, start: datetime, end: datetime, output_dir: str):
    """
    Exporte les ticks mois par mois pour éviter la surcharge RAM.
    Chaque mois = 1 fichier parquet.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    current = start.replace(day=1)
    while current < end:
        # Fin du mois
        next_month = (current + timedelta(days=32)).replace(day=1)
        month_end = min(next_month, end)

        filename = f"{symbol}_ticks_{current.strftime('%Y-%m')}.parquet"
        filepath = output_path / filename

        if filepath.exists():
            print(f"⏭️  {filename} existe déjà, skip")
            current = next_month
            continue

        print(f"📥 Export ticks {symbol} {current.strftime('%Y-%m')}...", end=" ", flush=True)

        ticks = mt5.copy_ticks_range(symbol, current, month_end, mt5.COPY_TICKS_ALL)

        if ticks is None or len(ticks) == 0:
            print(f"⚠️  Aucun tick")
            current = next_month
            continue

        df = pd.DataFrame(ticks)
        # Convertir timestamp ms → datetime
        df["time"] = pd.to_datetime(df["time"], unit="s")
        # Garder uniquement les colonnes utiles
        df = df[["time", "bid", "ask", "last", "volume", "flags"]]

        df.to_parquet(filepath, engine="pyarrow", compression="snappy")
        print(f"✅ {len(df):,} ticks → {filepath.stat().st_size / 1024 / 1024:.1f} MB")

        del df, ticks
        current = next_month


def export_candles(symbol: str, timeframe: str, start: datetime, end: datetime, output_dir: str):
    """
    Exporte les bougies pour un timeframe donné.
    Tout en un seul fichier parquet (les bougies sont légères).
    """
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

    if timeframe not in tf_map:
        raise ValueError(f"Timeframe inconnu: {timeframe}. Disponibles: {list(tf_map.keys())}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / f"{symbol}_{timeframe}_{start.year}-{end.year}.parquet"

    print(f"📥 Export {symbol} {timeframe} {start.date()} → {end.date()}...", end=" ", flush=True)

    rates = mt5.copy_rates_range(symbol, tf_map[timeframe], start, end)

    if rates is None or len(rates) == 0:
        print("⚠️  Aucune bougie")
        return

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]

    df.to_parquet(filepath, engine="pyarrow", compression="snappy")
    print(f"✅ {len(df):,} bougies → {filepath.stat().st_size / 1024:.0f} KB")


def shutdown_mt5():
    mt5.shutdown()
    print("🔌 MT5 déconnecté")
```

---

## 3. Chargement chunk par chunk

```python
# data/loader.py
"""
Chargeur de données RAM-safe.
Charge les parquets mois par mois, jamais tout en mémoire.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Iterator, Optional
import glob


@dataclass
class DataChunk:
    """Un mois de données chargées en mémoire."""
    month: str                              # "2025-06"
    candles_m5: pd.DataFrame                # Bougies M5
    candles_m1: Optional[pd.DataFrame]      # Bougies M1 (si dispo)
    ticks_path: Optional[Path]              # Chemin vers les ticks (lazy load)

    def get_ticks(self, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """Charge uniquement les ticks nécessaires pour résoudre un trade."""
        if self.ticks_path is None or not self.ticks_path.exists():
            return None

        df = pd.read_parquet(self.ticks_path)
        mask = (df["time"] >= start) & (df["time"] <= end)
        return df.loc[mask].reset_index(drop=True)


class DataLoader:
    def __init__(self, data_dir: str = "data/parquet"):
        self.data_dir = Path(data_dir)
        self.ticks_dir = self.data_dir / "ticks"
        self.candles_dir = self.data_dir / "candles"

    def info(self) -> pd.DataFrame:
        """Liste tous les fichiers de données disponibles."""
        rows = []
        for f in sorted(self.data_dir.rglob("*.parquet")):
            size_mb = f.stat().st_size / 1024 / 1024
            rows.append({
                "fichier": f.name,
                "type": "ticks" if "ticks" in f.parts else "candles",
                "taille": f"{size_mb:.1f} MB",
            })
        return pd.DataFrame(rows)

    def load_candles(self, symbol: str, timeframe: str,
                     start: str, end: str) -> pd.DataFrame:
        """Charge toutes les bougies d'un coup (elles sont légères)."""
        pattern = f"{symbol}_{timeframe}_*.parquet"
        files = sorted(self.candles_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(f"Aucun fichier trouvé: {pattern} dans {self.candles_dir}")

        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

        # Filtrer par dates
        mask = (df["time"] >= start) & (df["time"] <= end)
        return df.loc[mask].reset_index(drop=True)

    def iter_months(self, symbol: str, start: str, end: str) -> Iterator[DataChunk]:
        """
        Itère mois par mois. Charge bougies M5 + lien vers ticks.
        Chaque itération = 1 mois en mémoire max.
        """
        # Charger toutes les bougies M5 (légères, ~50 MB pour 1 an)
        all_m5 = self.load_candles(symbol, "M5", start, end)

        # Grouper par mois
        all_m5["month"] = all_m5["time"].dt.to_period("M")

        for month, group in all_m5.groupby("month"):
            month_str = str(month)

            # Chercher le fichier ticks correspondant
            ticks_file = self.ticks_dir / f"{symbol}_ticks_{month_str}.parquet"

            # Chercher bougies M1 si disponibles
            try:
                m1 = self.load_candles(
                    symbol, "M1",
                    str(group["time"].min()),
                    str(group["time"].max())
                )
            except FileNotFoundError:
                m1 = None

            chunk = DataChunk(
                month=month_str,
                candles_m5=group.reset_index(drop=True),
                candles_m1=m1,
                ticks_path=ticks_file if ticks_file.exists() else None,
            )

            yield chunk
            # Le garbage collector libère le chunk précédent
```

---

## 4. Keltner Channel

```python
# strategy/keltner.py
"""
Calcul des Keltner Channels.
Basé sur EMA + ATR (pas écart-type comme Bollinger).
"""

import pandas as pd
import numpy as np


def compute_keltner(df: pd.DataFrame,
                    ema_period: int = 20,
                    atr_period: int = 14,
                    multiplier: float = 2.0) -> pd.DataFrame:
    """
    Ajoute les colonnes Keltner au DataFrame de bougies.

    Colonnes ajoutées :
      - kc_mid   : EMA(close, period)
      - kc_upper : EMA + mult × ATR
      - kc_lower : EMA - mult × ATR
      - atr      : ATR(atr_period)

    Args:
        df: DataFrame avec colonnes [time, open, high, low, close]
        ema_period: Période de l'EMA centrale
        atr_period: Période de l'ATR
        multiplier: Multiplicateur pour les bandes

    Returns:
        DataFrame avec colonnes Keltner ajoutées
    """
    df = df.copy()

    # EMA du close
    df["kc_mid"] = df["close"].ewm(span=ema_period, adjust=False).mean()

    # ATR (True Range moyen)
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(span=atr_period, adjust=False).mean()

    # Bandes
    df["kc_upper"] = df["kc_mid"] + multiplier * df["atr"]
    df["kc_lower"] = df["kc_mid"] - multiplier * df["atr"]

    return df
```

---

## 5. Détection Breakout

```python
# strategy/breakout.py
"""
Détection des breakouts Keltner Channel.
Un breakout = le close d'une bougie franchit une bande.
"""

import pandas as pd
from core.types import Signal, Direction, SessionType
from core.constants import SESSIONS
from strategy.keltner import compute_keltner
from datetime import time as dtime


def detect_session(t: pd.Timestamp) -> SessionType:
    """Détermine la session active pour un timestamp UTC."""
    current_time = t.time()
    for session_name, (start, end) in SESSIONS.items():
        if start <= current_time <= end:
            return SessionType(session_name)
    return SessionType.OFF_SESSION


def detect_breakouts(df: pd.DataFrame,
                     ema_period: int = 20,
                     atr_period: int = 14,
                     multiplier: float = 2.0) -> list[Signal]:
    """
    Détecte les breakouts sur un DataFrame de bougies M5 avec Keltner.

    Règles :
      - LONG  : close > kc_upper ET close précédent <= kc_upper
      - SHORT : close < kc_lower ET close précédent >= kc_lower

    On ne prend que le PREMIER breakout par mouvement
    (pas de re-signal tant que le prix reste hors du canal).

    Returns:
        Liste de Signal détectés
    """
    df = compute_keltner(df, ema_period, atr_period, multiplier)
    signals = []

    # État : est-on déjà en breakout ?
    in_breakout = False
    last_direction = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # Skip si données Keltner pas encore stables
        if pd.isna(row["atr"]) or pd.isna(row["kc_upper"]):
            continue

        # Breakout LONG : close franchit la bande haute
        if row["close"] > row["kc_upper"] and prev["close"] <= prev["kc_upper"]:
            if not in_breakout or last_direction != Direction.LONG:
                signals.append(Signal(
                    time=row["time"],
                    direction=Direction.LONG,
                    price=row["close"],
                    keltner_band=row["kc_upper"],
                    atr=row["atr"],
                    session=detect_session(row["time"]),
                    candle_index=i,
                ))
                in_breakout = True
                last_direction = Direction.LONG

        # Breakout SHORT : close franchit la bande basse
        elif row["close"] < row["kc_lower"] and prev["close"] >= prev["kc_lower"]:
            if not in_breakout or last_direction != Direction.SHORT:
                signals.append(Signal(
                    time=row["time"],
                    direction=Direction.SHORT,
                    price=row["close"],
                    keltner_band=row["kc_lower"],
                    atr=row["atr"],
                    session=detect_session(row["time"]),
                    candle_index=i,
                ))
                in_breakout = True
                last_direction = Direction.SHORT

        # Retour dans le canal → reset
        elif row["close"] <= row["kc_upper"] and row["close"] >= row["kc_lower"]:
            in_breakout = False
            last_direction = None

    return signals
```

---

## 6. Modèle de coûts

```python
# backtest/costs.py
"""
Modèle de coûts réaliste pour matcher MT5.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CostModel:
    spread: float = 0.20           # $ (20 points sur l'or)
    commission_per_lot: float = 3.50  # $ round-turn par lot standard
    slippage_max: float = 0.05     # $ max de slippage (aléatoire)
    swap_long: float = -0.50       # $/lot/jour
    swap_short: float = 0.30       # $/lot/jour

    def entry_cost(self, lot_size: float, direction_long: bool) -> dict:
        """
        Calcule les coûts à l'entrée.

        Returns:
            dict avec spread_cost, commission, slippage, total
        """
        # Spread : payé à l'entrée (buy au ask, sell au bid)
        spread_cost = self.spread * lot_size * 100  # × contract_size

        # Commission : proportionnelle au lot
        commission = self.commission_per_lot * lot_size / 1.0  # par lot standard

        # Slippage : aléatoire entre 0 et max
        slippage = np.random.uniform(0, self.slippage_max) * lot_size * 100

        return {
            "spread_cost": spread_cost,
            "commission": commission,
            "slippage": slippage,
            "total": spread_cost + commission + slippage,
        }

    def adjust_entry_price(self, price: float, direction_long: bool) -> float:
        """Ajuste le prix d'entrée avec le spread."""
        if direction_long:
            return price + self.spread  # Achète au ask
        else:
            return price - self.spread  # Vend au bid (pas d'ajustement ici, le bid est déjà le prix)
        # Note : si les ticks ont bid/ask séparés, utiliser directement le ask/bid

    def swap_cost(self, lot_size: float, direction_long: bool, days: int) -> float:
        """Calcule le swap pour les positions overnight."""
        rate = self.swap_long if direction_long else self.swap_short
        return rate * lot_size * days
```

---

## 7. Résolution tick par tick

C'est le **cœur critique** du backtester. C'est ce qui rend les résultats réalistes.

```python
# backtest/resolver.py
"""
Résolution tick par tick du SL, breakeven et trailing stop.
C'est ici que la magie opère : on sait EXACTEMENT si le SL ou le trailing
a été touché en premier, contrairement à un backtest sur bougies.
"""

import numpy as np
import pandas as pd
from numba import njit
from core.types import Signal, Trade, Direction, TradeResult, BacktestConfig
from backtest.costs import CostModel


@njit
def resolve_trade_ticks(
    tick_prices: np.ndarray,     # Colonne bid ou ask selon direction
    tick_times: np.ndarray,      # Timestamps (int64, nanosecondes)
    entry_price: float,
    direction: int,              # 1 = LONG, -1 = SHORT
    sl_price: float,             # SL initial
    atr: float,                  # ATR au moment de l'entrée
    breakeven_atr_mult: float,   # Seuil breakeven en multiple d'ATR
    trailing_atr_mult: float,    # Distance trailing en multiple d'ATR
    trailing_step: float,        # Pas minimum pour update trailing (en $)
) -> tuple:  # (exit_price, exit_idx, max_favorable, max_adverse, be_activated, n_trailing)
    """
    Simule le trade tick par tick avec :
    1. SL initial fixe
    2. Breakeven quand profit >= breakeven_atr_mult × ATR
    3. Trailing stop à trailing_atr_mult × ATR du prix

    Tout est vectorisé avec Numba pour la performance.
    """
    n = len(tick_prices)
    current_sl = sl_price
    be_activated = False
    n_trailing = 0
    max_favorable = 0.0
    max_adverse = 0.0
    best_price = entry_price

    for i in range(n):
        price = tick_prices[i]

        # Calcul du P/L non réalisé
        if direction == 1:  # LONG
            pnl = price - entry_price
        else:  # SHORT
            pnl = entry_price - price

        # Mise à jour MFE / MAE
        if pnl > max_favorable:
            max_favorable = pnl
        if pnl < max_adverse:
            max_adverse = pnl

        # --- Vérifier SL ---
        if direction == 1 and price <= current_sl:
            return current_sl, i, max_favorable, max_adverse, be_activated, n_trailing
        elif direction == -1 and price >= current_sl:
            return current_sl, i, max_favorable, max_adverse, be_activated, n_trailing

        # --- Breakeven ---
        if not be_activated and pnl >= breakeven_atr_mult * atr:
            if direction == 1:
                current_sl = entry_price + 0.05  # +5 points au-dessus de l'entrée
            else:
                current_sl = entry_price - 0.05
            be_activated = True

        # --- Trailing Stop ---
        if be_activated:
            if direction == 1:
                new_sl = price - trailing_atr_mult * atr
                if new_sl > current_sl + trailing_step:
                    current_sl = new_sl
                    n_trailing += 1
            else:
                new_sl = price + trailing_atr_mult * atr
                if new_sl < current_sl - trailing_step:
                    current_sl = new_sl
                    n_trailing += 1

    # Si on arrive ici, le trade n'est pas fermé
    # Fermeture forcée au dernier tick
    return tick_prices[-1], n - 1, max_favorable, max_adverse, be_activated, n_trailing


def resolve_with_ticks(signal: Signal, ticks: pd.DataFrame,
                       config: BacktestConfig, cost_model: CostModel) -> Trade:
    """
    Résout un trade en utilisant les données tick par tick.
    Wrapper autour de la fonction Numba.
    """
    is_long = signal.direction == Direction.LONG

    # Prix d'entrée ajusté (spread)
    if ticks is not None and "ask" in ticks.columns and "bid" in ticks.columns:
        # Utiliser le vrai bid/ask du premier tick
        entry_price = ticks.iloc[0]["ask"] if is_long else ticks.iloc[0]["bid"]
        # Pour le suivi : LONG regarde le bid (prix de vente), SHORT regarde le ask
        prices = ticks["bid"].values if is_long else ticks["ask"].values
    else:
        entry_price = cost_model.adjust_entry_price(signal.price, is_long)
        prices = ticks["close"].values if ticks is not None else np.array([signal.price])

    # Calcul SL initial
    if is_long:
        sl_price = entry_price - config.sl_atr_mult * signal.atr
    else:
        sl_price = entry_price + config.sl_atr_mult * signal.atr

    # Résolution Numba
    times = ticks["time"].astype(np.int64).values if ticks is not None else np.array([0])

    exit_price, exit_idx, mfe, mae, be_activated, n_trailing = resolve_trade_ticks(
        tick_prices=prices,
        tick_times=times,
        entry_price=entry_price,
        direction=1 if is_long else -1,
        sl_price=sl_price,
        atr=signal.atr,
        breakeven_atr_mult=config.breakeven_atr_mult,
        trailing_atr_mult=config.trailing_atr_mult,
        trailing_step=config.trailing_step_points * config.point_value,
    )

    # Calcul P/L en dollars
    if is_long:
        price_diff = exit_price - entry_price
    else:
        price_diff = entry_price - exit_price

    # Conversion en dollars : mouvement × lot × contract_size
    gross_pnl = price_diff * config.lot_size * config.contract_size

    # Coûts
    costs = cost_model.entry_cost(config.lot_size, is_long)

    # Déterminer le résultat
    if gross_pnl > costs["total"]:
        result = TradeResult.WIN
    elif abs(price_diff) < 0.10:  # ~10 points = breakeven zone
        result = TradeResult.BREAKEVEN
    else:
        result = TradeResult.LOSS

    exit_time = ticks.iloc[exit_idx]["time"] if ticks is not None else signal.time

    return Trade(
        signal=signal,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_time=signal.time,
        exit_time=exit_time,
        direction=signal.direction,
        lot_size=config.lot_size,
        sl_initial=sl_price,
        result=result,
        gross_pnl=gross_pnl,
        commission=costs["commission"],
        spread_cost=costs["spread_cost"],
        slippage_cost=costs["slippage"],
        max_favorable=mfe * config.lot_size * config.contract_size,
        max_adverse=mae * config.lot_size * config.contract_size,
        breakeven_activated=be_activated,
        n_trailing_updates=n_trailing,
    )


def resolve_with_candles(signal: Signal, candles_m1: pd.DataFrame,
                         config: BacktestConfig, cost_model: CostModel) -> Trade:
    """
    Fallback : résolution avec bougies M1 si pas de ticks.
    Simule OHLC : Open → High/Low (selon direction) → Close.
    Moins précis mais mieux que rien.
    """
    # Simuler des "pseudo-ticks" à partir des bougies M1
    pseudo_ticks = []
    for _, candle in candles_m1.iterrows():
        # Ordre OHLC simulé
        if signal.direction == Direction.LONG:
            # Pour un long, le pire arrive au Low d'abord
            pseudo_ticks.extend([
                {"time": candle["time"], "bid": candle["open"], "ask": candle["open"]},
                {"time": candle["time"], "bid": candle["low"],  "ask": candle["low"]},
                {"time": candle["time"], "bid": candle["high"], "ask": candle["high"]},
                {"time": candle["time"], "bid": candle["close"],"ask": candle["close"]},
            ])
        else:
            # Pour un short, le pire arrive au High d'abord
            pseudo_ticks.extend([
                {"time": candle["time"], "bid": candle["open"], "ask": candle["open"]},
                {"time": candle["time"], "bid": candle["high"], "ask": candle["high"]},
                {"time": candle["time"], "bid": candle["low"],  "ask": candle["low"]},
                {"time": candle["time"], "bid": candle["close"],"ask": candle["close"]},
            ])

    pseudo_df = pd.DataFrame(pseudo_ticks)
    return resolve_with_ticks(signal, pseudo_df, config, cost_model)
```

---

## 8. Moteur de backtest

```python
# backtest/engine.py
"""
Moteur de backtest principal.
Orchestre : chargement données → détection signaux → résolution → rapport.
"""

import pandas as pd
from datetime import timedelta
from typing import Optional

from core.types import Trade, Signal, BacktestConfig, Direction
from data.loader import DataLoader
from strategy.breakout import detect_breakouts
from strategy.regime import apply_regime_filters
from backtest.resolver import resolve_with_ticks, resolve_with_candles
from backtest.costs import CostModel
from backtest.metrics import compute_metrics
from backtest.report import print_report


def run_backtest(
    config: BacktestConfig,
    start: str,
    end: str,
    data_dir: str = "data/parquet",
    verbose: bool = True,
) -> tuple[list[Trade], dict]:
    """
    Lance un backtest complet, chunk par chunk.

    Returns:
        (trades, metrics) - Liste des trades et dictionnaire de métriques
    """
    loader = DataLoader(data_dir)
    cost_model = CostModel(
        spread=config.spread,
        commission_per_lot=config.commission_per_lot,
        slippage_max=config.slippage,
    )

    all_trades: list[Trade] = []
    total_signals = 0
    filtered_signals = 0

    # Position tracking : pas de nouveau trade si déjà en position
    in_position = False
    position_exit_time = None

    for chunk in loader.iter_months(config.symbol, start, end):
        if verbose:
            print(f"\n📊 Traitement {chunk.month}...")

        # 1. Détecter les breakouts sur M5
        signals = detect_breakouts(
            chunk.candles_m5,
            ema_period=config.keltner_ema_period,
            atr_period=config.keltner_atr_period,
            multiplier=config.keltner_multiplier,
        )
        total_signals += len(signals)

        # 2. Appliquer les filtres de régime (Phase 2)
        if config.use_session_filter or config.use_atr_ratio_filter:
            signals = apply_regime_filters(signals, config)
            filtered_signals += len(signals)

        if verbose:
            print(f"   Signaux détectés: {len(signals)}")

        # 3. Résoudre chaque signal
        for signal in signals:
            # Vérifier qu'on n'est pas déjà en position
            if in_position and position_exit_time and signal.time < position_exit_time:
                continue

            # Fenêtre de résolution : 4h après le signal
            tick_start = signal.time
            tick_end = signal.time + timedelta(hours=4)

            # Essayer ticks d'abord, sinon fallback M1
            ticks = chunk.get_ticks(tick_start, tick_end)

            if ticks is not None and len(ticks) > 10:
                trade = resolve_with_ticks(signal, ticks, config, cost_model)
            elif chunk.candles_m1 is not None:
                m1_window = chunk.candles_m1[
                    (chunk.candles_m1["time"] >= tick_start) &
                    (chunk.candles_m1["time"] <= tick_end)
                ]
                if len(m1_window) > 0:
                    trade = resolve_with_candles(signal, m1_window, config, cost_model)
                else:
                    continue
            else:
                continue  # Pas de données pour résoudre

            all_trades.append(trade)
            in_position = True
            position_exit_time = trade.exit_time

            if verbose:
                emoji = "🟢" if trade.net_pnl > 0 else "🔴"
                dir_str = "LONG" if trade.direction == Direction.LONG else "SHORT"
                print(f"   {emoji} {dir_str} {trade.entry_time.strftime('%m-%d %H:%M')} "
                      f"→ {trade.exit_time.strftime('%H:%M')} | "
                      f"P/L: ${trade.net_pnl:+.2f}")

    # 4. Calculer les métriques
    metrics = compute_metrics(all_trades, config.initial_balance)

    if verbose:
        print_report(all_trades, metrics, config)

    return all_trades, metrics
```

---

## 9. Rapport de performance

```python
# backtest/metrics.py
"""
Calcul des métriques de performance.
"""

import numpy as np
from core.types import Trade, TradeResult, SessionType


def compute_metrics(trades: list[Trade], initial_balance: float) -> dict:
    """Calcule toutes les métriques de performance."""
    if not trades:
        return {"error": "Aucun trade"}

    net_pnls = np.array([t.net_pnl for t in trades])
    gross_pnls = np.array([t.gross_pnl for t in trades])

    wins = net_pnls[net_pnls > 0]
    losses = net_pnls[net_pnls < 0]

    # Equity curve
    equity = np.cumsum(net_pnls) + initial_balance
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_dd = drawdown.min()
    max_dd_pct = (max_dd / peak[np.argmin(drawdown)]) * 100 if len(peak) > 0 else 0

    # Profit Factor
    total_wins = wins.sum() if len(wins) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 1
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    # Sharpe (annualisé, assumant ~252 jours de trading)
    if len(net_pnls) > 1 and net_pnls.std() > 0:
        sharpe = (net_pnls.mean() / net_pnls.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Commissions totales
    total_commissions = sum(t.commission + t.spread_cost + t.slippage_cost for t in trades)

    # Stats par session
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
```

```python
# backtest/report.py
"""
Affichage Rich du rapport de backtest.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from core.types import BacktestConfig

console = Console()


def print_report(trades: list, metrics: dict, config: BacktestConfig):
    """Affiche un rapport de backtest formaté avec Rich."""

    if "error" in metrics:
        console.print(f"[red]❌ {metrics['error']}[/red]")
        return

    # --- Panel principal ---
    lines = []
    lines.append(f"Période     : {trades[0].entry_time.date()} → {trades[-1].exit_time.date()}")
    lines.append(f"Capital     : ${config.initial_balance:,.2f}")
    lines.append(f"Lot size    : {config.lot_size} ({config.lot_size * config.contract_size:.0f}$/point)")
    lines.append(f"Commission  : ${config.commission_per_lot}/lot RT | Spread: ${config.spread}")
    lines.append("")
    lines.append("─── Résultats ───")

    # Colorer le profit
    net = metrics["net_profit"]
    net_color = "green" if net > 0 else "red"
    lines.append(f"Trades total     : {metrics['total_trades']}")
    lines.append(f"Trades gagnants  : {metrics['winning_trades']} ({metrics['winrate']:.1f}%)")
    lines.append(f"Trades perdants  : {metrics['losing_trades']}")
    lines.append(f"Gain moyen       : ${metrics['avg_win']:+.2f}")
    lines.append(f"Perte moyenne    : ${metrics['avg_loss']:+.2f}")
    lines.append(f"Profit Factor    : {metrics['profit_factor']:.2f}")
    lines.append(f"Espérance/trade  : ${metrics['expectancy']:+.2f}")
    lines.append(f"Max Drawdown     : ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)")
    lines.append(f"Sharpe Ratio     : {metrics['sharpe_ratio']:.2f}")
    lines.append(f"Commissions payées: ${metrics['total_commissions']:.2f}")

    panel_text = "\n".join(lines)
    console.print(Panel(panel_text, title=f"Backtest {config.symbol}", border_style="cyan"))

    # Ligne de profit net en gros
    console.print(f"\n  [bold {net_color}]Profit net : ${net:+.2f}[/bold {net_color}]")
    console.print(f"  [bold]Balance finale : ${metrics['final_balance']:,.2f}[/bold]\n")

    # --- Table par session ---
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
```

---

## 10. CLI principal

```python
# cli/app.py
"""
CLI principal avec Typer + Rich.
Point d'entrée : `nova` ou `python -m cli.app`
"""

import typer
from rich.console import Console
from typing import Optional
from pathlib import Path

app = typer.Typer(
    name="nova",
    help="NovaGold Reborn — Trading ML pour XAU/USD",
    add_completion=False,
)
console = Console()

# Sous-groupes
data_app = typer.Typer(help="Gestion des données MT5 ↔ Parquet")
backtest_app = typer.Typer(help="Backtesting")
optimize_app = typer.Typer(help="Optimisation des paramètres")
train_app = typer.Typer(help="ML : entraînement et évaluation")
live_app = typer.Typer(help="Trading live MT5")

app.add_typer(data_app, name="data")
app.add_typer(backtest_app, name="backtest")
app.add_typer(optimize_app, name="optimize")
app.add_typer(train_app, name="train")
app.add_typer(live_app, name="live")


# ═══════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════

@data_app.command("export")
def data_export(
    symbol: str = typer.Option("XAUUSD", help="Symbole MT5"),
    type: str = typer.Option("candles", help="ticks ou candles"),
    timeframe: str = typer.Option("M5", help="M1, M5, M15, H1, H4, D1"),
    start: str = typer.Option(..., help="Date début YYYY-MM-DD"),
    end: str = typer.Option(..., help="Date fin YYYY-MM-DD"),
    output: str = typer.Option("data/parquet", help="Répertoire de sortie"),
):
    """Exporte les données MT5 vers Parquet."""
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
    """Affiche les données disponibles."""
    from data.loader import DataLoader
    loader = DataLoader()
    console.print(loader.info().to_string())


# ═══════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════

@backtest_app.command("run")
def backtest_run(
    symbol: str = typer.Option("XAUUSD", help="Symbole"),
    start: str = typer.Option(..., help="Date début YYYY-MM-DD"),
    end: str = typer.Option(..., help="Date fin YYYY-MM-DD"),
    lot: float = typer.Option(0.01, help="Taille de lot"),
    commission: float = typer.Option(3.50, help="Commission $/lot round-turn"),
    spread: float = typer.Option(0.20, help="Spread moyen en $"),
    balance: float = typer.Option(10000, help="Capital initial $"),
    keltner_period: int = typer.Option(20, help="Période EMA Keltner"),
    keltner_mult: float = typer.Option(2.0, help="Multiplicateur Keltner"),
    sl_atr_mult: float = typer.Option(2.0, help="SL = N × ATR"),
    trailing_atr_mult: float = typer.Option(1.5, help="Trailing = N × ATR"),
    breakeven_atr_mult: float = typer.Option(1.0, help="Breakeven à N × ATR"),
    regime_filter: bool = typer.Option(False, help="Activer filtre de session"),
    sessions: str = typer.Option("sge_open,london,overlap", help="Sessions autorisées"),
):
    """Lance un backtest avec les paramètres donnés."""
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
    # Le rapport est affiché par run_backtest si verbose=True


# ═══════════════════════════════════════════
# OPTIMIZE
# ═══════════════════════════════════════════

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

    # Parser les paramètres
    param_grid = {}
    for p in param:
        name, values = p.split(":")
        param_grid[name] = [float(v) for v in values.split(",")]

    keys = list(param_grid.keys())
    all_combos = list(product(*param_grid.values()))
    console.print(f"🔍 {len(all_combos)} combinaisons à tester...")

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

        console.print(f"  [{i+1}/{len(all_combos)}] {params} → PF={pf:.2f}")

    console.print(f"\n[green]✅ Meilleur : PF={best_pf:.2f} | {best_params}[/green]")


if __name__ == "__main__":
    app()
```

---

## 11. Filtres de régime (Phase 2)

```python
# strategy/regime.py
"""
Filtres de régime : session temporelle + énergie ATR.
Zéro ML, zéro lag.
"""

from core.types import Signal, BacktestConfig, SessionType
from core.constants import SESSIONS


def apply_regime_filters(signals: list[Signal], config: BacktestConfig) -> list[Signal]:
    """
    Filtre les signaux selon les règles de régime.
    Retourne uniquement les signaux qui passent tous les filtres.
    """
    filtered = signals

    # Filtre 1 : Session temporelle
    if config.use_session_filter:
        allowed = [SessionType(s) for s in config.allowed_sessions]
        filtered = [s for s in filtered if s.session in allowed]

    # Filtre 2 : Énergie ATR (ratio M5/H1)
    # Note : nécessite l'ATR H1 pré-calculé dans le signal ou le DataFrame
    if config.use_atr_ratio_filter:
        filtered = [
            s for s in filtered
            if hasattr(s, "atr_ratio") and s.atr_ratio >= config.atr_ratio_threshold
        ]

    return filtered


def compute_atr_ratio(candles_m5, candles_h1, index: int) -> float:
    """
    Calcule le ratio ATR M5 / ATR H1 pour un index donné.
    Ratio élevé = marché actif, breakout plus crédible.
    Ratio faible = marché mort, ignorer les signaux.
    """
    if candles_h1 is None or len(candles_h1) == 0:
        return 1.0  # Pas de données H1, on laisse passer

    # Trouver la bougie H1 correspondante
    m5_time = candles_m5.iloc[index]["time"]
    h1_mask = candles_h1["time"] <= m5_time
    if h1_mask.sum() == 0:
        return 1.0

    atr_m5 = candles_m5.iloc[index]["atr"]
    atr_h1 = candles_h1.loc[h1_mask].iloc[-1]["atr"]

    if atr_h1 == 0:
        return 0.0

    return atr_m5 / atr_h1
```

---

## 12. Feature engineering (Phase 3)

```python
# ml/features.py
"""
Construction des features pour le LightGBM.
~8-10 features, toutes normalisées/relatives.
"""

import pandas as pd
import numpy as np
from core.types import Signal, SessionType


def build_features(signal: Signal, candles_m5: pd.DataFrame,
                   candles_h1: pd.DataFrame = None) -> dict:
    """
    Construit le vecteur de features pour un signal de breakout.

    Toutes les features sont relatives/normalisées pour être
    robustes aux changements de niveau de prix.
    """
    i = signal.candle_index
    row = candles_m5.iloc[i]

    features = {}

    # 1. Tick volume relatif (proxy participation)
    if "tick_volume" in candles_m5.columns:
        vol_window = candles_m5.iloc[max(0, i-20):i]["tick_volume"]
        if len(vol_window) > 0 and vol_window.mean() > 0:
            features["tick_volume_ratio"] = row["tick_volume"] / vol_window.mean()
        else:
            features["tick_volume_ratio"] = 1.0
    else:
        features["tick_volume_ratio"] = 1.0

    # 2. ATR ratio multi-TF
    if candles_h1 is not None and "atr" in candles_h1.columns:
        h1_mask = candles_h1["time"] <= row["time"]
        if h1_mask.sum() > 0:
            atr_h1 = candles_h1.loc[h1_mask].iloc[-1]["atr"]
            features["atr_ratio_mtf"] = row["atr"] / atr_h1 if atr_h1 > 0 else 0
        else:
            features["atr_ratio_mtf"] = 0
    else:
        features["atr_ratio_mtf"] = 0

    # 3. Distance à la bande Keltner (normalisée par ATR)
    if signal.direction.value == 1:
        features["keltner_distance"] = (row["close"] - row["kc_upper"]) / row["atr"]
    else:
        features["keltner_distance"] = (row["kc_lower"] - row["close"]) / row["atr"]

    # 4. RSI(14)
    closes = candles_m5.iloc[max(0, i-14):i+1]["close"]
    if len(closes) >= 14:
        delta = closes.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        features["rsi_14"] = 100 - (100 / (1 + gain / loss)) if loss > 0 else 50
    else:
        features["rsi_14"] = 50

    # 5. Momentum 5 bougies (rendement normalisé)
    if i >= 5:
        price_5_ago = candles_m5.iloc[i - 5]["close"]
        features["momentum_5"] = (row["close"] - price_5_ago) / row["atr"]
    else:
        features["momentum_5"] = 0

    # 6. Session (encodage ordinal)
    session_map = {
        SessionType.SGE_OPEN: 0,
        SessionType.LONDON: 1,
        SessionType.OVERLAP: 2,
        SessionType.OFF_SESSION: 3,
    }
    features["session_id"] = session_map.get(signal.session, 3)

    # 7. Pente EMA H1 (tendance de fond)
    if candles_h1 is not None and "kc_mid" in candles_h1.columns:
        h1_recent = candles_h1[candles_h1["time"] <= row["time"]].tail(5)
        if len(h1_recent) >= 2:
            ema_values = h1_recent["kc_mid"].values
            # Pente normalisée par ATR H1
            slope = (ema_values[-1] - ema_values[0]) / len(ema_values)
            atr_h1 = h1_recent.iloc[-1]["atr"]
            features["ema_slope_h1"] = slope / atr_h1 if atr_h1 > 0 else 0
        else:
            features["ema_slope_h1"] = 0
    else:
        features["ema_slope_h1"] = 0

    # 8. Largeur Bollinger M15 (proxy volatilité intermédiaire)
    # Simplifié : on utilise le ratio écart-type / moyenne sur 20 bougies M5
    if i >= 20:
        window = candles_m5.iloc[i-20:i+1]["close"]
        features["vol_ratio_20"] = window.std() / window.mean() if window.mean() > 0 else 0
    else:
        features["vol_ratio_20"] = 0

    return features


FEATURE_COLUMNS = [
    "tick_volume_ratio",
    "atr_ratio_mtf",
    "keltner_distance",
    "rsi_14",
    "momentum_5",
    "session_id",
    "ema_slope_h1",
    "vol_ratio_20",
]
```

---

## 13. Walk-forward LightGBM (Phase 4)

```python
# ml/trainer.py
"""
Entraînement LightGBM avec walk-forward sliding window.
Le modèle ne voit JAMAIS les données futures.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Optional

from ml.features import build_features, FEATURE_COLUMNS
from core.types import Signal, BacktestConfig


def walk_forward_train(
    signals_with_labels: pd.DataFrame,
    train_weeks: int = 8,
    test_weeks: int = 2,
    step_weeks: int = 2,
    model_params: dict = None,
) -> list[dict]:
    """
    Walk-forward training avec fenêtre glissante.

    Args:
        signals_with_labels: DataFrame avec features + colonne 'label' (1=GO, 0=NO-GO)
        train_weeks: Taille fenêtre d'entraînement
        test_weeks: Taille fenêtre de test
        step_weeks: Pas d'avancement

    Returns:
        Liste de résultats par fold
    """
    if model_params is None:
        model_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": 400,
            "max_depth": 5,
            "num_leaves": 20,
            "learning_rate": 0.02,
            "min_child_samples": 300,
            "subsample": 0.7,
            "colsample_bytree": 0.6,
            "reg_alpha": 0.5,
            "reg_lambda": 0.5,
            "verbose": -1,
            "n_jobs": -1,
        }

    df = signals_with_labels.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])

    start = df["time"].min()
    end = df["time"].max()

    train_delta = timedelta(weeks=train_weeks)
    test_delta = timedelta(weeks=test_weeks)
    step_delta = timedelta(weeks=step_weeks)

    fold_results = []
    fold = 0
    current = start

    while current + train_delta + test_delta <= end:
        train_end = current + train_delta
        test_end = train_end + test_delta

        # Split
        train_mask = (df["time"] >= current) & (df["time"] < train_end)
        test_mask = (df["time"] >= train_end) & (df["time"] < test_end)

        X_train = df.loc[train_mask, FEATURE_COLUMNS]
        y_train = df.loc[train_mask, "label"]
        X_test = df.loc[test_mask, FEATURE_COLUMNS]
        y_test = df.loc[test_mask, "label"]

        if len(X_train) < 50 or len(X_test) < 10:
            current += step_delta
            continue

        # Entraînement
        model = lgb.LGBMClassifier(**model_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.log_evaluation(0)],
        )

        # Prédiction
        probas = model.predict_proba(X_test)[:, 1]
        preds = (probas >= 0.62).astype(int)

        # Métriques du fold
        accuracy = (preds == y_test.values).mean()
        precision = preds[y_test.values == 1].mean() if (y_test.values == 1).sum() > 0 else 0

        fold_results.append({
            "fold": fold,
            "train_start": current.date(),
            "train_end": train_end.date(),
            "test_start": train_end.date(),
            "test_end": test_end.date(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": accuracy,
            "precision": precision,
            "model": model,
        })

        print(f"  Fold {fold}: train {len(X_train)} | test {len(X_test)} | "
              f"acc={accuracy:.3f} | prec={precision:.3f}")

        current += step_delta
        fold += 1

    return fold_results
```

---

## 14. Bridge MT5 live (Phase 5)

```python
# live/mt5_bridge.py
"""
Bridge Python ↔ MT5 pour l'exécution live.
IMPORTANT : ne tourne que sur Windows (lib MetaTrader5).
"""

import MetaTrader5 as mt5
from core.types import Signal, Direction, BacktestConfig
from datetime import datetime


class MT5Bridge:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.connected = False

    def connect(self, login: int, password: str, server: str) -> bool:
        if not mt5.initialize():
            print(f"❌ MT5 init failed: {mt5.last_error()}")
            return False
        if not mt5.login(login, password=password, server=server):
            print(f"❌ Login failed: {mt5.last_error()}")
            return False
        self.connected = True
        info = mt5.account_info()
        print(f"✅ Connecté: {info.server} | Balance: ${info.balance:.2f}")
        return True

    def execute_signal(self, signal: Signal) -> bool:
        """Exécute un signal de breakout sur MT5."""
        if not self.connected:
            return False

        symbol = self.config.symbol
        lot = self.config.lot_size

        # Récupérer le prix actuel
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"❌ Pas de cotation pour {symbol}")
            return False

        # SL basé sur ATR
        if signal.direction == Direction.LONG:
            price = tick.ask
            sl = price - self.config.sl_atr_mult * signal.atr
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = price + self.config.sl_atr_mult * signal.atr
            order_type = mt5.ORDER_TYPE_SELL

        # Pas de TP fixe (trailing gère la sortie)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": round(sl, 2),
            "tp": 0.0,  # Pas de TP, trailing stop via script
            "deviation": 10,
            "magic": 240331,
            "comment": f"NovaReborn_{signal.session.value}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Ordre échoué: {result.retcode} - {result.comment}")
            return False

        dir_str = "LONG" if signal.direction == Direction.LONG else "SHORT"
        print(f"✅ {dir_str} {lot} lots @ {price:.2f} | SL: {sl:.2f}")
        return True

    def update_trailing(self):
        """
        Met à jour le trailing stop pour les positions ouvertes.
        À appeler à chaque nouveau tick ou nouvelle bougie.
        """
        positions = mt5.positions_get(symbol=self.config.symbol)
        if positions is None or len(positions) == 0:
            return

        for pos in positions:
            if pos.magic != 240331:
                continue

            tick = mt5.symbol_info_tick(self.config.symbol)
            if tick is None:
                continue

            # Calcul du profit actuel en $
            if pos.type == mt5.POSITION_TYPE_BUY:
                current_pnl = tick.bid - pos.price_open
                # Breakeven check
                be_level = pos.price_open + self.config.breakeven_atr_mult * pos.sl  # Simplifié
                if current_pnl > 0 and tick.bid > be_level:
                    new_sl = max(pos.sl, tick.bid - self.config.trailing_atr_mult * current_pnl)
                    if new_sl > pos.sl + 0.10:  # Minimum 10 points de step
                        self._modify_sl(pos.ticket, new_sl)

            elif pos.type == mt5.POSITION_TYPE_SELL:
                current_pnl = pos.price_open - tick.ask
                if current_pnl > 0:
                    new_sl = min(pos.sl, tick.ask + self.config.trailing_atr_mult * current_pnl)
                    if new_sl < pos.sl - 0.10:
                        self._modify_sl(pos.ticket, new_sl)

    def _modify_sl(self, ticket: int, new_sl: float):
        """Modifie le SL d'une position."""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": round(new_sl, 2),
            "tp": 0.0,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  📐 SL mis à jour → {new_sl:.2f}")

    def disconnect(self):
        mt5.shutdown()
        self.connected = False
```

---

## Notes d'implémentation

### Ordre de développement recommandé

1. `core/types.py` + `core/constants.py` — les fondations
2. `data/mt5_export.py` — export des données
3. `data/loader.py` — chargement chunk par chunk
4. `strategy/keltner.py` — calcul Keltner
5. `strategy/breakout.py` — détection des signaux
6. `backtest/costs.py` — modèle de coûts
7. `backtest/resolver.py` — résolution tick par tick (**le plus complexe**)
8. `backtest/metrics.py` + `backtest/report.py` — métriques et affichage
9. `backtest/engine.py` — orchestration
10. `cli/app.py` — interface CLI
11. `strategy/regime.py` — filtres de session (Phase 2)
12. `ml/features.py` — feature engineering (Phase 3)
13. `ml/trainer.py` — walk-forward LightGBM (Phase 4)
14. `live/mt5_bridge.py` — bridge MT5 (Phase 5)

### Points critiques

- Le **resolver tick par tick** (fichier 7) est le code le plus important.
  S'il est faux, tout le backtest est faux. Tester avec des cas simples d'abord.
- Le **CostModel** doit matcher ton broker exact. Lance un trade réel de 0.01 lot,
  note le spread + commission payés, et configure les mêmes valeurs.
- La fonction Numba `resolve_trade_ticks` doit être compilée une première fois
  (lent au premier appel, rapide ensuite). C'est normal.
- Le walk-forward LightGBM (fichier 13) ne s'implémente qu'après avoir validé
  que la stratégie brute a un semblant d'edge (PF > 1.0 sur certaines sessions).
