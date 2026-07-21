from __future__ import annotations

import numpy as np
import pandas as pd

from tse_session_ranker.data.tdnet import TDnetDataset, normalize_tdnet_disclosures


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


def synthetic_tdnet(
    prices: pd.DataFrame, through: object | None = None
) -> TDnetDataset:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(prices["date"]).unique()))
    first = dates.min()
    disclosure = normalize_tdnet_disclosures(
        pd.DataFrame(
            {
                "published_at": [
                    pd.Timestamp(f"{first.date()} 08:00:00", tz="Asia/Tokyo")
                ],
                "code": [str(prices.iloc[0]["code"])],
                "name": [str(prices.iloc[0].get("name", prices.iloc[0]["code"]))],
                "title": ["自己株式取得に係る事項の決定に関するお知らせ"],
                "url": ["https://example.invalid/tdnet.pdf"],
            }
        )
    )
    complete_through = (
        pd.Timestamp(through).normalize() if through is not None else dates.max()
    )
    complete = pd.date_range(dates.min(), complete_through, freq="D")
    observed = pd.Series(
        [
            (date + pd.Timedelta(days=1, minutes=1)).tz_localize("Asia/Tokyo")
            for date in complete
        ],
        index=complete,
        name="observed_at",
    )
    return TDnetDataset(
        disclosures=disclosure,
        complete_dates=complete,
        observed_at_by_date=observed,
        provenance_by_date=pd.Series(
            "network_request_start_recorded",
            index=complete,
            name="provenance",
        ),
        source_sha256="synthetic-tdnet-source",
        source_files=len(complete),
    )
