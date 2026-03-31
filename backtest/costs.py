from dataclasses import dataclass, field
import numpy as np


@dataclass
class CostModel:
    spread: float = 0.20
    commission_per_lot: float = 3.50
    slippage_max: float = 0.05
    swap_long: float = -0.50
    swap_short: float = 0.30
    random_seed: int = 42

    def __post_init__(self):
        self._rng = np.random.default_rng(self.random_seed)

    def entry_cost(self, lot_size: float, direction_long: bool) -> dict:
        # Spread is already captured via bid/ask prices — do NOT add it again here.
        # Only commission and slippage are added as explicit costs.
        commission = self.commission_per_lot * lot_size
        slippage = self._rng.uniform(0, self.slippage_max) * lot_size * 100

        return {
            "spread_cost": 0.0,
            "commission": commission,
            "slippage": slippage,
            "total": commission + slippage,
        }

    def adjust_entry_price(self, price: float, direction_long: bool) -> float:
        if direction_long:
            return price + self.spread
        else:
            return price - self.spread

    def spread_from_tick(self, bid: float, ask: float) -> float:
        """Calculate real spread from tick bid/ask."""
        return ask - bid

    def swap_cost(self, lot_size: float, direction_long: bool, days: int) -> float:
        rate = self.swap_long if direction_long else self.swap_short
        return rate * lot_size * days

