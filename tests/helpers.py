from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_prices(
    periods: int = 260, start: str = "2024-01-04", codes: int = 4
) -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range(start, periods=periods)
    for code_index in range(codes):
        close = 700.0 + code_index * 250.0
        for index, date in enumerate(dates):
            overnight = 0.0015 * np.sin(index * 0.31 + code_index)
            intraday = 0.004 * np.cos(index * 0.47 + code_index * 0.8)
            open_price = close * (1.0 + overnight)
            close_price = open_price * (1.0 + intraday)
            spread = 0.004 + 0.001 * abs(np.sin(index + code_index))
            rows.append(
                {
                    "date": date,
                    "code": f"{1001 + code_index}",
                    "name": f"銘柄{code_index + 1}",
                    "open": open_price,
                    "high": max(open_price, close_price) * (1.0 + spread),
                    "low": min(open_price, close_price) * (1.0 - spread),
                    "close": close_price,
                    "volume": 100_000 + index * 10 + code_index,
                }
            )
            close = close_price
    return pd.DataFrame(rows)

