from datetime import time

SESSIONS = {
    "sge_open": (time(1, 30), time(3, 0)),
    "london":   (time(8, 0),  time(10, 30)),
    "overlap":  (time(12, 0), time(16, 30)),
}

XAUUSD = {
    "symbol": "XAUUSD",
    "point": 0.01,
    "digits": 2,
    "contract_size": 100,
    "tick_size": 0.01,
}

DEFAULT_KELTNER_EMA_PERIOD = 20
DEFAULT_KELTNER_ATR_PERIOD = 14
DEFAULT_KELTNER_MULTIPLIER = 2.0

DEFAULT_SL_ATR_MULT = 2.0
DEFAULT_BREAKEVEN_ATR_MULT = 1.0
DEFAULT_TRAILING_ATR_MULT = 1.5
DEFAULT_TRAILING_STEP_POINTS = 10

DEFAULT_SPREAD = 0.20
DEFAULT_COMMISSION_PER_LOT = 3.50
DEFAULT_SLIPPAGE_MAX = 0.05

DEFAULT_INITIAL_BALANCE = 10_000.0
DEFAULT_LOT_SIZE = 0.01
