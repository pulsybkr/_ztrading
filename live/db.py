"""
SQLite logger for live/paper trading sessions.
Stores everything: sessions, opportunities (taken & rejected), trades, SL events.
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = "live/data/live_sessions.db"


class LiveDB:
    def __init__(self, db_path: str = DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")  # safe for multi-thread
        self._create_tables()
        self.session_id: int = None
        self.current_trade_id: int = None
        self._current_opportunity_id: int = None

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at     TEXT NOT NULL,
                ended_at       TEXT,
                mode           TEXT NOT NULL,           -- 'paper' | 'live'
                account_login  INTEGER,
                account_server TEXT,
                account_name   TEXT,
                currency       TEXT,
                balance_start  REAL,
                equity_start   REAL,
                balance_end    REAL,
                equity_end     REAL,
                pnl_session    REAL,
                symbol         TEXT,
                timeframe      TEXT,
                params         TEXT                     -- JSON full BacktestConfig
            );

            CREATE TABLE IF NOT EXISTS opportunities (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id       INTEGER NOT NULL,
                detected_at      TEXT NOT NULL,
                direction        TEXT NOT NULL,          -- 'LONG' | 'SHORT'
                price            REAL,
                atr              REAL,
                atr_ratio        REAL,
                session_type     TEXT,
                kc_upper         REAL,
                kc_lower         REAL,
                taken            INTEGER NOT NULL DEFAULT 0,
                rejection_reason TEXT,                   -- NULL if taken
                ml_proba         REAL,
                ml_threshold     REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS trades (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id       INTEGER NOT NULL,
                opportunity_id   INTEGER,
                ticket           INTEGER,
                direction        TEXT NOT NULL,
                entry_price      REAL,
                entry_time       TEXT,
                sl_initial       REAL,
                lot_size         REAL,
                exit_price       REAL,
                exit_time        TEXT,
                exit_reason      TEXT,                   -- 'sl' | 'manual' | 'unknown'
                pnl_gross        REAL,
                pnl_net          REAL,
                n_trailing_updates INTEGER DEFAULT 0,
                breakeven_activated INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (opportunity_id) REFERENCES opportunities(id)
            );

            CREATE TABLE IF NOT EXISTS trade_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id        INTEGER NOT NULL,
                event_at        TEXT NOT NULL,
                event_type      TEXT NOT NULL,           -- 'breakeven' | 'trailing_update' | 'exit'
                old_sl          REAL,
                new_sl          REAL,
                price_bid       REAL,
                price_ask       REAL,
                atr             REAL,
                unrealized_pnl  REAL,
                FOREIGN KEY (trade_id) REFERENCES trades(id)
            );
        """)
        self.conn.commit()

    # ── Session ──────────────────────────────────────────────────────

    def open_session(self, mode: str, account: dict, config) -> int:
        params = {k: v for k, v in config.__dict__.items()}
        cur = self.conn.execute(
            """INSERT INTO sessions
               (started_at, mode, account_login, account_server, account_name,
                currency, balance_start, equity_start, symbol, timeframe, params)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                _now(), mode,
                account.get("login"), account.get("server"), account.get("name"),
                account.get("currency"),
                account.get("balance"), account.get("equity"),
                config.symbol,
                getattr(config, "signal_timeframe", "M5"),
                json.dumps(params, default=str),
            )
        )
        self.conn.commit()
        self.session_id = cur.lastrowid
        return self.session_id

    def close_session(self, balance_end: float, equity_end: float):
        if not self.session_id:
            return
        self.conn.execute(
            """UPDATE sessions
               SET ended_at=?, balance_end=?, equity_end=?, pnl_session=?
               WHERE id=?""",
            (_now(), balance_end, equity_end,
             round(equity_end - (self._get_session_balance_start() or equity_end), 2),
             self.session_id)
        )
        self.conn.commit()

    def _get_session_balance_start(self) -> float:
        row = self.conn.execute(
            "SELECT balance_start FROM sessions WHERE id=?", (self.session_id,)
        ).fetchone()
        return row["balance_start"] if row else None

    # ── Opportunities ────────────────────────────────────────────────

    def log_opportunity(self, signal, taken: bool,
                        rejection_reason: str = None,
                        ml_proba: float = None,
                        ml_threshold: float = None,
                        kc_upper: float = None,
                        kc_lower: float = None) -> int:
        ts = signal.time.isoformat() if hasattr(signal.time, "isoformat") else str(signal.time)
        cur = self.conn.execute(
            """INSERT INTO opportunities
               (session_id, detected_at, direction, price, atr, atr_ratio,
                session_type, kc_upper, kc_lower, taken, rejection_reason,
                ml_proba, ml_threshold)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.session_id, ts,
                signal.direction.name,
                signal.price, signal.atr,
                getattr(signal, "atr_ratio", 0.0),
                signal.session.value,
                kc_upper, kc_lower,
                1 if taken else 0,
                rejection_reason, ml_proba, ml_threshold,
            )
        )
        self.conn.commit()
        self._current_opportunity_id = cur.lastrowid
        return self._current_opportunity_id

    # ── Trades ───────────────────────────────────────────────────────

    def open_trade(self, ticket: int, direction: str,
                   entry_price: float, sl: float, lot: float) -> int:
        cur = self.conn.execute(
            """INSERT INTO trades
               (session_id, opportunity_id, ticket, direction,
                entry_price, entry_time, sl_initial, lot_size)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.session_id, self._current_opportunity_id,
                ticket, direction,
                entry_price, _now(), sl, lot,
            )
        )
        self.conn.commit()
        self.current_trade_id = cur.lastrowid
        return self.current_trade_id

    def close_trade(self, exit_price: float, exit_reason: str,
                    pnl_gross: float = None, pnl_net: float = None):
        if not self.current_trade_id:
            return
        # Count events
        n_trail = self.conn.execute(
            "SELECT COUNT(*) FROM trade_events WHERE trade_id=? AND event_type='trailing_update'",
            (self.current_trade_id,)
        ).fetchone()[0]
        n_be = self.conn.execute(
            "SELECT COUNT(*) FROM trade_events WHERE trade_id=? AND event_type='breakeven'",
            (self.current_trade_id,)
        ).fetchone()[0]
        self.conn.execute(
            """UPDATE trades
               SET exit_price=?, exit_time=?, exit_reason=?,
                   pnl_gross=?, pnl_net=?,
                   n_trailing_updates=?, breakeven_activated=?
               WHERE id=?""",
            (exit_price, _now(), exit_reason,
             pnl_gross, pnl_net,
             n_trail, 1 if n_be > 0 else 0,
             self.current_trade_id)
        )
        self.conn.commit()
        self.current_trade_id = None

    # ── Trade events ─────────────────────────────────────────────────

    def log_sl_event(self, event_type: str, old_sl: float, new_sl: float,
                     price_bid: float, price_ask: float,
                     atr: float, unrealized_pnl: float = None):
        if not self.current_trade_id:
            return
        self.conn.execute(
            """INSERT INTO trade_events
               (trade_id, event_at, event_type, old_sl, new_sl,
                price_bid, price_ask, atr, unrealized_pnl)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.current_trade_id, _now(), event_type,
                old_sl, new_sl, price_bid, price_ask, atr, unrealized_pnl,
            )
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")
