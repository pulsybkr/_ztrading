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
    ASIAN = "asian"
    LONDON = "london"
    OVERLAP = "overlap"
    OFF_SESSION = "off_session"


@dataclass
class Signal:
    time: datetime
    direction: Direction
    price: float
    keltner_band: float
    atr: float
    session: SessionType
    candle_index: int
    atr_ratio: float = 0.0


@dataclass
class Trade:
    signal: Signal
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    direction: Direction
    lot_size: float
    sl_initial: float
    result: TradeResult
    gross_pnl: float = 0.0
    commission: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    net_pnl: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    breakeven_activated: bool = False
    n_trailing_updates: int = 0

    def __post_init__(self):
        self.net_pnl = self.gross_pnl - self.commission - self.spread_cost - self.slippage_cost


@dataclass
class BacktestConfig:
    symbol: str = "XAUUSD"
    lot_size: float = 0.01
    initial_balance: float = 100.0   # compte réel ~50-100 EUR
    point_value: float = 0.01
    contract_size: float = 100

    signal_timeframe: str = "M5"  # Timeframe des signaux: M1, M5, M15, M30, H1

    keltner_ema_period: int = 20
    keltner_atr_period: int = 14
    keltner_multiplier: float = 2.0

    # SL serré (1x ATR) : ~$3 de risque max par trade sur 0.01 lot
    # Évite les trades qui traînent 4h sans toucher le SL
    sl_atr_mult: float = 1.0
    breakeven_atr_mult: float = 0.5
    trailing_atr_mult: float = 0.75
    trailing_step_points: int = 10

    spread: float = 0.20
    commission_per_lot: float = 3.50
    slippage: float = 0.05

    use_session_filter: bool = False
    use_atr_ratio_filter: bool = False
    atr_ratio_threshold: float = 0.15
    allowed_sessions: list = field(default_factory=lambda: ["asian", "london", "overlap"])
    max_positions: int = 1
    max_daily_trades: int = 10

    use_ml_filter: bool = False
    ml_threshold: float = 0.62

    @property
    def dollar_per_point(self) -> float:
        return self.lot_size * self.contract_size * self.point_value
