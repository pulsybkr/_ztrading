import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class DataChunk:
    month: str
    candles_m5: pd.DataFrame
    candles_m1: Optional[pd.DataFrame]
    ticks_path: Optional[Path]

    def get_ticks(self, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
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
        pattern = f"{symbol}_{timeframe}_*.parquet"
        files = sorted(self.candles_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(f"Aucun fichier trouvé: {pattern} dans {self.candles_dir}")

        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

        mask = (df["time"] >= start) & (df["time"] <= end)
        return df.loc[mask].reset_index(drop=True)

    def iter_months(self, symbol: str, start: str, end: str) -> Iterator[DataChunk]:
        all_m5 = self.load_candles(symbol, "M5", start, end)

        all_m5["month"] = all_m5["time"].dt.to_period("M")

        for month, group in all_m5.groupby("month"):
            month_str = str(month)

            ticks_file = self.ticks_dir / f"{symbol}_ticks_{month_str}.parquet"

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
