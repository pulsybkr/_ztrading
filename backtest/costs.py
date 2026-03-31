from dataclasses import dataclass
import numpy as np


@dataclass
class CostModel:
    spread: float = 0.20
    commission_per_lot: float = 3.50
    slippage_max: float = 0.05
    swap_long: float = -0.50
    swap_short: float = 0.30

    def entry_cost(self, lot_size: float, direction_long: bool) -> dict:
        spread_cost = self.spread * lot_size * 100

        commission = self.commission_per_lot * lot_size / 1.0

        slippage = np.random.uniform(0, self.slippage_max) * lot_size * 100

        return {
            "spread_cost": spread_cost,
            "commission": commission,
            "slippage": slippage,
            "total": spread_cost + commission + slippage,
        }

    def adjust_entry_price(self, price: float, direction_long: bool) -> float:
        if direction_long:
            return price + self.spread
        else:
            return price - self.spread

    def swap_cost(self, lot_size: float, direction_long: bool, days: int) -> float:
        rate = self.swap_long if direction_long else self.swap_short
        return rate * lot_size * days
