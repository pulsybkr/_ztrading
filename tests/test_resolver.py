import pytest
import pandas as pd
import numpy as np
from core.types import BacktestConfig, Direction
from backtest.costs import CostModel
from backtest.resolver import resolve_with_ticks


@pytest.fixture
def sample_signal():
    from core.types import Signal, SessionType
    from datetime import datetime
    
    return Signal(
        time=datetime(2025, 1, 1, 9, 0),
        direction=Direction.LONG,
        price=2000.0,
        keltner_band=1998.0,
        atr=5.0,
        session=SessionType.LONDON,
        candle_index=0,
    )


@pytest.fixture
def sample_ticks():
    dates = pd.date_range("2025-01-01 09:00", periods=100, freq="1min")
    prices = np.cumsum(np.random.randn(100) * 0.5 + 0.1) + 2000.0
    
    return pd.DataFrame({
        "time": dates,
        "bid": prices,
        "ask": prices + 0.2,
    })


def test_cost_model_entry():
    model = CostModel(spread=0.20, commission_per_lot=3.50, slippage_max=0.05)
    
    costs = model.entry_cost(0.01, True)
    
    assert costs["spread_cost"] == 0.20 * 0.01 * 100
    assert "commission" in costs
    assert "slippage" in costs
    assert costs["total"] > 0


def test_cost_model_adjust_price():
    model = CostModel()
    
    long_price = model.adjust_entry_price(2000.0, True)
    assert long_price == 2000.20
    
    short_price = model.adjust_entry_price(2000.0, False)
    assert short_price == 1999.80


def test_resolve_trade_ticks_long_hit_sl(sample_signal, sample_ticks):
    config = BacktestConfig(
        lot_size=0.01,
        contract_size=100,
        sl_atr_mult=2.0,
        breakeven_atr_mult=1.0,
        trailing_atr_mult=1.5,
        trailing_step_points=10,
        point_value=0.01,
    )
    model = CostModel()
    
    signal = sample_signal
    signal.direction = Direction.LONG
    
    trade = resolve_with_ticks(signal, sample_ticks, config, model)
    
    assert trade.direction == Direction.LONG
    assert trade.entry_price > 0
    assert trade.exit_price > 0
    assert trade.lot_size == 0.01


def test_resolve_trade_ticks_short(sample_signal, sample_ticks):
    config = BacktestConfig(
        lot_size=0.01,
        contract_size=100,
        sl_atr_mult=2.0,
        breakeven_atr_mult=1.0,
        trailing_atr_mult=1.5,
        trailing_step_points=10,
        point_value=0.01,
    )
    model = CostModel()
    
    signal = sample_signal
    signal.direction = Direction.SHORT
    
    trade = resolve_with_ticks(signal, sample_ticks, config, model)
    
    assert trade.direction == Direction.SHORT
