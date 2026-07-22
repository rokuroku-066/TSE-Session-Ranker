"""Leakage-safe feature candidates for the v0.6 research protocol.

This module is deliberately outside the production feature manifest.  It
implements only candidates that can be reconstructed from the historical JPX
monthly OHLC archive and cached TDnet date indexes.  Exact 08:58 futures,
auction, PTS, liquidity and quantitative disclosure fields remain forward-only
and are listed in ``research/model_v06_feature_catalog.json``.
"""

from __future__ import annotations

from datetime import time as clock_time
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .data.common import SESSION_OHLC, normalize_expected_sessions
from .data.tdnet import normalize_tdnet_disclosures
from .exceptions import DataValidationError, LeakageError


G1_SHORT_REVERSAL_COLUMNS: tuple[str, ...] = (
    "cc_reversal_1_atr",
    "cc_momentum_3_atr",
    "range_shock_1_20",
    "cc_vol_ratio_5_20",
)

G2_GAP_TRAIT_COLUMNS: tuple[str, ...] = (
    "overnight_std_20",
    "gap_fill_rate_20",
    "gap_response_beta_60",
    "gap_frequency_20",
)

G3_SESSION_DYNAMICS_COLUMNS: tuple[str, ...] = (
    "am_return_mean_5",
    "am_return_mean_20",
    "am_win_rate_20",
    "midday_gap_mean_20",
    "am_pm_corr_20",
    "prior_session_close_location",
    "session_range_ratio_5_20",
)

G4_MARKET_REGIME_COLUMNS: tuple[str, ...] = (
    "prior_market_cc_return_pct",
    "prior_market_overnight_return_pct",
    "prior_market_am_return_pct",
    "prior_market_pm_return_pct",
    "prior_market_breadth",
    "prior_market_dispersion",
    "prior_market_tail_balance",
    "market_cc_momentum_5",
    "market_cc_momentum_20",
    "market_cc_vol_5",
    "market_cc_vol_20",
    "market_cc_vol_ratio_5_20",
    "market_beta_x_prior_market_cc_return",
    "market_beta_x_market_cc_momentum_5",
    "market_beta_x_market_cc_momentum_20",
    "atr_x_market_dispersion",
    "atr_x_market_cc_vol_20",
)

G5_LIQUIDITY_PROXY_COLUMNS: tuple[str, ...] = (
    "no_trade_rate_20",
    "no_trade_rate_60",
    "flat_oc_rate_20",
    "zero_range_rate_20",
)

HISTORICAL_PRICE_CANDIDATE_COLUMNS: tuple[str, ...] = (
    *G1_SHORT_REVERSAL_COLUMNS,
    *G2_GAP_TRAIT_COLUMNS,
    *G3_SESSION_DYNAMICS_COLUMNS,
    *G4_MARKET_REGIME_COLUMNS,
    *G5_LIQUIDITY_PROXY_COLUMNS,
)

T0_CLEAN_EVENT_COLUMNS: tuple[str, ...] = (
    "tdnet_clean_any",
    "tdnet_clean_has_earnings",
    "tdnet_clean_has_revision",
    "tdnet_clean_revision_up_title",
    "tdnet_clean_revision_down_title",
    "tdnet_clean_revision_direction_unknown",
    "tdnet_clean_has_dividend",
    "tdnet_clean_dividend_up_title",
    "tdnet_clean_dividend_down_title",
    "tdnet_clean_dividend_direction_unknown",
    "tdnet_clean_has_buyback_decision",
    "tdnet_clean_has_buyback_tostnet",
    "tdnet_clean_has_buyback_status",
    "tdnet_clean_has_external_equity_financing",
    "tdnet_clean_has_equity_compensation",
    "tdnet_clean_has_equity_financing_status",
    "tdnet_clean_has_benefit",
    "tdnet_clean_has_split",
    "tdnet_clean_has_ma_transaction",
    "tdnet_clean_has_business_alliance",
    "tdnet_clean_has_control_transaction",
    "tdnet_clean_has_impairment_loss",
    "tdnet_clean_has_audit_problem",
    "tdnet_clean_has_correction",
)

T1_EVENT_STRUCTURE_COLUMNS: tuple[str, ...] = (
    "tdnet_clean_document_count_log1p",
    "tdnet_clean_family_count_log1p",
    "tdnet_clean_premarket_count_log1p",
    "tdnet_clean_intraday_count_log1p",
    "tdnet_clean_postclose_count_log1p",
    "tdnet_clean_latest_age_hours_log1p",
    "tdnet_clean_bundle_width_minutes_log1p",
    "tdnet_clean_has_progress_stage",
    "tdnet_clean_single_family",
    "tdnet_clean_support_adverse_conflict",
    "tdnet_clean_prior_bundle_count_60_log1p",
    "tdnet_clean_prior_bundle_count_252_log1p",
    "tdnet_clean_weekend_age",
)

X0_EVENT_CONTEXT_COLUMNS: tuple[str, ...] = (
    "tdnet_clean_revision_x_xrank_close_momentum_20",
    "tdnet_clean_dividend_x_xrank_close_momentum_20",
    "tdnet_clean_buyback_x_xrank_close_momentum_20",
    "tdnet_clean_external_financing_x_xrank_close_momentum_20",
    "tdnet_clean_ma_x_xrank_close_momentum_20",
    "tdnet_clean_any_x_prior_market_tail_balance",
)

TDNET_CANDIDATE_COLUMNS: tuple[str, ...] = (
    *T0_CLEAN_EVENT_COLUMNS,
    *T1_EVENT_STRUCTURE_COLUMNS,
)

_TDNET_FAMILY_FLAG_NAMES: tuple[str, ...] = (
    "earnings",
    "revision",
    "dividend",
    "buyback_decision",
    "buyback_tostnet",
    "buyback_status",
    "external_equity_financing",
    "equity_compensation",
    "equity_financing_status",
    "benefit",
    "split",
    "ma_transaction",
    "business_alliance",
    "control_transaction",
    "impairment_loss",
    "audit_problem",
    "correction",
)


def _require_columns(
    frame: pd.DataFrame, columns: Sequence[str], source: str
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise DataValidationError(f"{source} is missing columns: {missing}")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe = pd.to_numeric(denominator, errors="coerce").where(
        pd.to_numeric(denominator, errors="coerce").gt(0)
    )
    return pd.to_numeric(numerator, errors="coerce") / safe


def _safe_percent(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return 100.0 * (_safe_divide(numerator, denominator) - 1.0)


def _prior_rolling_values(
    frame: pd.DataFrame,
    values: pd.Series,
    *,
    window: int,
    operation: str,
    min_periods: int,
    shift: int = 1,
) -> pd.Series:
    shifted = values.groupby(frame["code"], sort=False).shift(shift)
    roller = shifted.groupby(frame["code"], sort=False).rolling(
        window, min_periods=min_periods
    )
    if operation == "mean":
        result = roller.mean()
    elif operation == "sum":
        result = roller.sum()
    elif operation == "std":
        result = roller.std()
    elif operation == "median":
        result = roller.median()
    else:
        raise ValueError(f"unsupported rolling operation: {operation}")
    return result.reset_index(level=0, drop=True).sort_index()


def _prior_pair_slope(
    frame: pd.DataFrame,
    x_values: pd.Series,
    y_values: pd.Series,
    *,
    window: int,
    min_pairs: int,
) -> pd.Series:
    valid = x_values.notna() & y_values.notna()
    x = x_values.where(valid, 0.0)
    y = y_values.where(valid, 0.0)
    pieces = {
        "n": valid.astype(float),
        "x": x,
        "y": y,
        "xx": x * x,
        "xy": x * y,
    }
    sums = {
        name: _prior_rolling_values(
            frame,
            values,
            window=window,
            operation="sum",
            min_periods=1,
        )
        for name, values in pieces.items()
    }
    n = sums["n"]
    safe_n = n.where(n.gt(0))
    covariance_numerator = sums["xy"] - sums["x"] * sums["y"] / safe_n
    variance_numerator = sums["xx"] - sums["x"] * sums["x"] / safe_n
    slope = covariance_numerator / variance_numerator.where(
        variance_numerator.gt(1e-12)
    )
    return slope.where(n.ge(min_pairs))


def _prior_pair_corr(
    frame: pd.DataFrame,
    left: pd.Series,
    right: pd.Series,
    *,
    window: int,
    min_pairs: int,
) -> pd.Series:
    valid = left.notna() & right.notna()
    x = left.where(valid, 0.0)
    y = right.where(valid, 0.0)
    values = {
        "n": valid.astype(float),
        "x": x,
        "y": y,
        "xx": x * x,
        "yy": y * y,
        "xy": x * y,
    }
    sums = {
        name: _prior_rolling_values(
            frame,
            series,
            window=window,
            operation="sum",
            min_periods=1,
        )
        for name, series in values.items()
    }
    n = sums["n"]
    safe_n = n.where(n.gt(0))
    cov = sums["xy"] - sums["x"] * sums["y"] / safe_n
    var_x = (sums["xx"] - sums["x"] * sums["x"] / safe_n).clip(lower=0)
    var_y = (sums["yy"] - sums["y"] * sums["y"] / safe_n).clip(lower=0)
    denominator = np.sqrt(var_x * var_y)
    return (cov / denominator.where(denominator.gt(1e-12))).where(
        n.ge(min_pairs)
    )


def _date_rolling(
    values: pd.Series,
    *,
    window: int,
    operation: str,
    min_periods: int,
) -> pd.Series:
    shifted = values.shift(1)
    roller = shifted.rolling(window, min_periods=min_periods)
    if operation == "sum":
        return roller.sum()
    if operation == "std":
        return roller.std()
    raise ValueError(f"unsupported date rolling operation: {operation}")


def add_historical_candidate_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add v0.6 historical candidate blocks using dates strictly before target.

    The input is expected to be the full point-in-time research panel before
    its raw OHLC columns are projected away.  Current rows are source
    observations only; every candidate returned for date ``D`` is shifted or
    mapped from a market aggregate ending at ``D-1``.
    """

    _require_columns(
        panel,
        (
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "traded",
            "prior_close",
            "true_range",
            "atr14_pct",
            "overnight",
            "oc_return_pct",
            "market_beta_60",
            *SESSION_OHLC,
        ),
        "research panel",
    )
    frame = panel.copy().reset_index(drop=True)
    frame["_candidate_original_order"] = np.arange(len(frame))
    parsed_dates = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    if parsed_dates.isna().any():
        raise DataValidationError("research panel contains an invalid date")
    if getattr(parsed_dates.dt, "tz", None) is not None:
        parsed_dates = parsed_dates.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    frame["date"] = parsed_dates.dt.normalize()
    frame["code"] = frame["code"].astype(str).str.strip()
    if frame["code"].eq("").any() or frame.duplicated(["date", "code"]).any():
        raise DataValidationError("research panel has empty or duplicate date/code")
    frame = frame.sort_values(["code", "date"], kind="stable").reset_index(drop=True)

    for column in (
        "open",
        "high",
        "low",
        "close",
        "prior_close",
        "true_range",
        "atr14_pct",
        "overnight",
        "oc_return_pct",
        "market_beta_60",
        *SESSION_OHLC,
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    traded = frame["traded"].fillna(False).astype(bool)
    observed = frame.get(
        "outcome_observed", pd.Series(True, index=frame.index, dtype=bool)
    ).fillna(False).astype(bool)
    source_complete = frame.get(
        "source_complete", pd.Series(True, index=frame.index, dtype=bool)
    ).fillna(False).astype(bool)

    cc_simple = _safe_percent(frame["close"], frame["prior_close"]).where(traded)
    cc_log = (100.0 * np.log1p(cc_simple / 100.0)).where(cc_simple.abs().le(30.0))
    prior_cc = cc_log.groupby(frame["code"], sort=False).shift(1)
    safe_atr = frame["atr14_pct"].where(frame["atr14_pct"].gt(0))
    frame["cc_reversal_1_atr"] = prior_cc / safe_atr
    frame["cc_momentum_3_atr"] = _prior_rolling_values(
        frame, cc_log, window=3, operation="sum", min_periods=3
    ) / safe_atr
    cc_std_5 = _prior_rolling_values(
        frame, cc_log, window=5, operation="std", min_periods=4
    )
    cc_std_20 = _prior_rolling_values(
        frame, cc_log, window=20, operation="std", min_periods=10
    )
    frame["cc_vol_ratio_5_20"] = _safe_divide(cc_std_5, cc_std_20)

    true_range_pct = 100.0 * _safe_divide(
        frame["true_range"], frame["prior_close"]
    )
    prior_true_range = true_range_pct.groupby(frame["code"], sort=False).shift(1)
    range_baseline = _prior_rolling_values(
        frame,
        true_range_pct,
        window=20,
        operation="median",
        min_periods=10,
        shift=2,
    )
    frame["range_shock_1_20"] = _safe_divide(prior_true_range, range_baseline)

    clipped_overnight = frame["overnight"].where(frame["overnight"].abs().le(30.0))
    frame["overnight_std_20"] = _prior_rolling_values(
        frame,
        clipped_overnight,
        window=20,
        operation="std",
        min_periods=10,
    )
    gap_known = (
        clipped_overnight.notna()
        & frame["oc_return_pct"].notna()
        & clipped_overnight.abs().ge(0.25)
    )
    gap_filled = pd.Series(np.nan, index=frame.index, dtype=float)
    gap_filled.loc[gap_known] = (
        clipped_overnight.loc[gap_known]
        * frame.loc[gap_known, "oc_return_pct"]
    ).lt(0.0).astype(float)
    frame["gap_fill_rate_20"] = _prior_rolling_values(
        frame, gap_filled, window=20, operation="mean", min_periods=8
    )
    frame["gap_response_beta_60"] = _prior_pair_slope(
        frame,
        clipped_overnight,
        frame["oc_return_pct"],
        window=60,
        min_pairs=20,
    )
    gap_frequency = (
        clipped_overnight.abs() / safe_atr
    ).ge(0.5).astype(float).where(clipped_overnight.notna() & safe_atr.notna())
    frame["gap_frequency_20"] = _prior_rolling_values(
        frame, gap_frequency, window=20, operation="mean", min_periods=10
    )

    am_return = _safe_percent(frame["am_close"], frame["am_open"])
    pm_return = _safe_percent(frame["pm_close"], frame["pm_open"])
    midday_gap = _safe_percent(frame["pm_open"], frame["am_close"])
    frame["am_return_mean_5"] = _prior_rolling_values(
        frame, am_return, window=5, operation="mean", min_periods=3
    )
    frame["am_return_mean_20"] = _prior_rolling_values(
        frame, am_return, window=20, operation="mean", min_periods=10
    )
    am_win = am_return.gt(0).astype(float).where(am_return.notna())
    frame["am_win_rate_20"] = _prior_rolling_values(
        frame, am_win, window=20, operation="mean", min_periods=10
    )
    frame["midday_gap_mean_20"] = _prior_rolling_values(
        frame, midday_gap, window=20, operation="mean", min_periods=10
    )
    frame["am_pm_corr_20"] = _prior_pair_corr(
        frame, am_return, pm_return, window=20, min_pairs=10
    )
    session_span = (frame["high"] - frame["low"]).where(
        frame["high"].gt(frame["low"])
    )
    close_location = ((frame["close"] - frame["low"]) / session_span).clip(0, 1)
    frame["prior_session_close_location"] = close_location.groupby(
        frame["code"], sort=False
    ).shift(1)
    session_range_pct = 100.0 * _safe_divide(session_span, frame["prior_close"])
    session_range_5 = _prior_rolling_values(
        frame, session_range_pct, window=5, operation="mean", min_periods=3
    )
    session_range_20 = _prior_rolling_values(
        frame, session_range_pct, window=20, operation="mean", min_periods=10
    )
    frame["session_range_ratio_5_20"] = _safe_divide(
        session_range_5, session_range_20
    )

    # Market state is constructed on each source date, then shifted one
    # exchange session before being mapped back to target rows.
    market_mask = traded & observed & source_complete
    market_cc_source = cc_log.where(market_mask)
    market_overnight_source = clipped_overnight.where(market_mask)
    market_am_source = am_return.where(market_mask)
    market_pm_source = pm_return.where(market_mask)
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    date_key = frame["date"]
    market_cc = market_cc_source.groupby(date_key, sort=True).mean().reindex(dates)
    market_overnight = (
        market_overnight_source.groupby(date_key, sort=True).mean().reindex(dates)
    )
    market_am = market_am_source.groupby(date_key, sort=True).mean().reindex(dates)
    market_pm = market_pm_source.groupby(date_key, sort=True).mean().reindex(dates)
    breadth = (
        market_cc_source.gt(0).astype(float).where(market_cc_source.notna())
        .groupby(date_key, sort=True)
        .mean()
        .reindex(dates)
    )
    cc_median = market_cc_source.groupby(date_key, sort=True).median().reindex(dates)
    absolute_deviation = (market_cc_source - date_key.map(cc_median)).abs()
    dispersion = (
        1.4826
        * absolute_deviation.groupby(date_key, sort=True).median().reindex(dates)
    )
    cc_atr = (cc_log / safe_atr).where(market_mask)
    positive_tail = cc_atr.gt(1.0).astype(float).where(cc_atr.notna())
    negative_tail = cc_atr.lt(-1.0).astype(float).where(cc_atr.notna())
    tail_balance = (
        positive_tail.groupby(date_key, sort=True).mean().reindex(dates)
        - negative_tail.groupby(date_key, sort=True).mean().reindex(dates)
    )
    market_state = {
        "prior_market_cc_return_pct": market_cc.shift(1),
        "prior_market_overnight_return_pct": market_overnight.shift(1),
        "prior_market_am_return_pct": market_am.shift(1),
        "prior_market_pm_return_pct": market_pm.shift(1),
        "prior_market_breadth": breadth.shift(1),
        "prior_market_dispersion": dispersion.shift(1),
        "prior_market_tail_balance": tail_balance.shift(1),
        "market_cc_momentum_5": _date_rolling(
            market_cc, window=5, operation="sum", min_periods=5
        ),
        "market_cc_momentum_20": _date_rolling(
            market_cc, window=20, operation="sum", min_periods=10
        ),
        "market_cc_vol_5": _date_rolling(
            market_cc, window=5, operation="std", min_periods=4
        ),
        "market_cc_vol_20": _date_rolling(
            market_cc, window=20, operation="std", min_periods=10
        ),
    }
    for name, values in market_state.items():
        frame[name] = frame["date"].map(values)
    frame["market_cc_vol_ratio_5_20"] = _safe_divide(
        frame["market_cc_vol_5"], frame["market_cc_vol_20"]
    )
    beta = frame["market_beta_60"]
    frame["market_beta_x_prior_market_cc_return"] = (
        beta * frame["prior_market_cc_return_pct"]
    )
    frame["market_beta_x_market_cc_momentum_5"] = (
        beta * frame["market_cc_momentum_5"]
    )
    frame["market_beta_x_market_cc_momentum_20"] = (
        beta * frame["market_cc_momentum_20"]
    )
    frame["atr_x_market_dispersion"] = safe_atr * frame["prior_market_dispersion"]
    frame["atr_x_market_cc_vol_20"] = safe_atr * frame["market_cc_vol_20"]

    explicit_no_trade = (~traded).astype(float).where(observed & source_complete)
    frame["no_trade_rate_20"] = _prior_rolling_values(
        frame, explicit_no_trade, window=20, operation="mean", min_periods=10
    )
    frame["no_trade_rate_60"] = _prior_rolling_values(
        frame, explicit_no_trade, window=60, operation="mean", min_periods=30
    )
    flat_oc = frame["oc_return_pct"].abs().lt(1e-12).astype(float).where(traded)
    frame["flat_oc_rate_20"] = _prior_rolling_values(
        frame, flat_oc, window=20, operation="mean", min_periods=10
    )
    zero_range = frame["high"].eq(frame["low"]).astype(float).where(traded)
    frame["zero_range_rate_20"] = _prior_rolling_values(
        frame, zero_range, window=20, operation="mean", min_periods=10
    )

    prior_source_date = frame.groupby("code", sort=False)["date"].shift(1)
    if (prior_source_date.notna() & prior_source_date.ge(frame["date"])).any():
        raise LeakageError("candidate price features contain a non-prior source")
    frame["candidate_price_source_max_date"] = prior_source_date
    frame[list(HISTORICAL_PRICE_CANDIDATE_COLUMNS)] = frame[
        list(HISTORICAL_PRICE_CANDIDATE_COLUMNS)
    ].astype("float32")
    frame = frame.sort_values("_candidate_original_order", kind="stable").drop(
        columns=["_candidate_original_order"]
    )
    return frame.reset_index(drop=True)


def _contains(title: pd.Series, expression: str) -> pd.Series:
    return title.str.contains(expression, regex=True, na=False)


def _clean_title_flags(title: pd.Series) -> pd.DataFrame:
    """Return mutually interpretable title flags for retrospective research."""

    flags = pd.DataFrame(index=title.index)
    flags["earnings"] = _contains(title, r"決算短信")
    flags["revision"] = _contains(
        title,
        r"業績予想.*(?:修正|差異|未定|撤回|取下|取り下)|通期.*予想.*修正",
    )
    flags["revision_up_title"] = flags["revision"] & _contains(
        title, r"上方修正|上方に修正|増額修正"
    )
    flags["revision_down_title"] = flags["revision"] & _contains(
        title, r"下方修正|下方に修正|減額修正"
    )
    flags["revision_direction_unknown"] = flags["revision"] & ~(
        flags["revision_up_title"] | flags["revision_down_title"]
    )
    flags["dividend"] = _contains(title, r"配当")
    flags["dividend_up_title"] = flags["dividend"] & _contains(
        title, r"増配|復配"
    )
    flags["dividend_down_title"] = flags["dividend"] & _contains(
        title, r"減配|無配"
    )
    flags["dividend_direction_unknown"] = flags["dividend"] & ~(
        flags["dividend_up_title"] | flags["dividend_down_title"]
    )

    self_stock = _contains(title, r"自己株(?:式)?")
    flags["buyback_tostnet"] = self_stock & _contains(
        title, r"ToSTNeT|ＴｏＳＴＮｅＴ|立会外"
    )
    flags["buyback_status"] = self_stock & _contains(
        title, r"取得状況|取得結果|取得終了|取得完了|取得実績"
    )
    flags["buyback_decision"] = self_stock & _contains(
        title,
        r"取得に係る事項|取得の決定|取得枠|取得を行う|自己株式取得.*(?:決定|実施)",
    ) & ~flags["buyback_status"]

    flags["equity_financing_status"] = _contains(
        title,
        r"払込完了|発行結果|行使状況|大量行使|月間行使|行使完了|発行中止|失権|新株予約権.*消却",
    )
    flags["equity_compensation"] = _contains(
        title,
        r"ストック.?オプション|株式報酬|譲渡制限付株式|役員.*新株予約権|従業員.*新株予約権|取締役.*(?:株式|新株予約権)",
    )
    broad_financing = _contains(
        title,
        r"第三者割当|公募増資|新株式.*発行|新株予約権.*発行|転換社債|CB|ライツ.?オファリング|株式の売出し",
    )
    flags["external_equity_financing"] = broad_financing & ~(
        flags["equity_financing_status"] | flags["equity_compensation"]
    )

    flags["benefit"] = _contains(title, r"株主優待")
    flags["split"] = _contains(title, r"株式分割")
    ma_expression = (
        r"子会社化|完全子会社|(?<!自己)株式取得|持分取得|事業譲受|事業譲渡|"
        r"合併|会社分割|株式交換"
    )
    flags["ma_transaction"] = _contains(title, ma_expression) & ~self_stock
    flags["business_alliance"] = _contains(
        title, r"業務提携|資本提携|資本業務提携"
    )
    flags["control_transaction"] = _contains(
        title, r"公開買付|ＴＯＢ|TOB|ＭＢＯ|MBO|株式交換"
    )
    flags["impairment_loss"] = _contains(
        title, r"減損|特別損失|債権放棄"
    )
    flags["audit_problem"] = _contains(
        title,
        r"不適切会計|継続企業の前提|内部統制.*不備|調査報告書|監査法人.*(?:異動|辞任|退任|意見不表明|限定付)|(?:異動|辞任|退任).*監査法人",
    )
    flags["correction"] = _contains(title, r"訂正")
    return flags.astype(float)


def _prior_bundle_counts(
    bundles: pd.DataFrame, *, days: int
) -> pd.Series:
    output = pd.Series(0.0, index=bundles.index, dtype=float)
    horizon = pd.Timedelta(days=days).value
    for positions in bundles.groupby("code", sort=False).indices.values():
        pos = np.asarray(positions, dtype=int)
        times = pd.DatetimeIndex(bundles.loc[pos, "bundle_start"]).asi8
        left = np.searchsorted(times, times - horizon, side="left")
        output.loc[pos] = np.arange(len(pos), dtype=float) - left
    return output


def build_clean_tdnet_candidate_features(
    disclosures: pd.DataFrame,
    expected_sessions: Iterable[object],
    *,
    decision_time: str = "08:58:59",
) -> pd.DataFrame:
    """Aggregate corrected TDnet title/timing features at the pre-open cutoff."""

    events = normalize_tdnet_disclosures(disclosures)
    sessions = normalize_expected_sessions(expected_sessions)
    try:
        cutoff_time = clock_time.fromisoformat(decision_time)
    except ValueError as exc:
        raise ValueError("decision_time must be an ISO local time") from exc
    cutoffs = pd.DatetimeIndex(
        [
            pd.Timestamp.combine(date.date(), cutoff_time).tz_localize("Asia/Tokyo")
            for date in sessions
        ]
    )
    published = pd.DatetimeIndex(events["published_at"])
    target_positions = cutoffs.searchsorted(published, side="left")
    in_range = target_positions < len(cutoffs)
    events = events.loc[in_range].copy()
    target_positions = target_positions[in_range]
    events["date"] = sessions[target_positions].to_numpy()
    target_cutoff = pd.Series(cutoffs[target_positions], index=events.index)
    events["age_hours"] = (
        target_cutoff - events["published_at"]
    ).dt.total_seconds() / 3600.0
    if events["age_hours"].lt(0).any():
        raise LeakageError("TDnet candidate event was assigned before publication")

    event_date = events["published_at"].dt.tz_localize(None).dt.normalize()
    event_minutes = (
        events["published_at"].dt.hour * 60 + events["published_at"].dt.minute
    )
    same_target = event_date.eq(events["date"])
    is_session = event_date.isin(sessions)
    close_minutes = np.where(
        event_date.lt(pd.Timestamp("2024-11-05")), 15 * 60, 15 * 60 + 30
    )
    events["premarket"] = same_target
    events["intraday"] = is_session & ~same_target & event_minutes.lt(close_minutes)
    events["postclose"] = ~(
        events["premarket"] | events["intraday"]
    )
    flags = _clean_title_flags(events["title"])
    events = pd.concat([events, flags], axis=1)
    progress_expression = (
        r"取得状況|取得結果|終了|完了|経過|変更|中止|払込|行使状況|発行結果"
    )
    events["progress_stage"] = _contains(events["title"], progress_expression)
    supportive = events[
        [
            "revision_up_title",
            "dividend_up_title",
            "buyback_decision",
            "benefit",
            "split",
        ]
    ].max(axis=1)
    adverse = events[
        [
            "revision_down_title",
            "dividend_down_title",
            "external_equity_financing",
            "impairment_loss",
            "audit_problem",
        ]
    ].max(axis=1)
    events["supportive"] = supportive
    events["adverse"] = adverse

    keys = ["date", "code"]
    grouped = events.groupby(keys, sort=True)
    bundles = grouped.size().rename("document_count").reset_index()
    bundles["bundle_start"] = grouped["published_at"].min().to_numpy()
    bundles["bundle_end"] = grouped["published_at"].max().to_numpy()
    bundles["latest_age_hours"] = grouped["age_hours"].min().to_numpy()
    for name in ("premarket", "intraday", "postclose"):
        bundles[f"{name}_count"] = grouped[name].sum().to_numpy()
    for name in (
        *_TDNET_FAMILY_FLAG_NAMES,
        "revision_up_title",
        "revision_down_title",
        "revision_direction_unknown",
        "dividend_up_title",
        "dividend_down_title",
        "dividend_direction_unknown",
        "progress_stage",
        "supportive",
        "adverse",
    ):
        bundles[name] = grouped[name].max().to_numpy(dtype=float)
    family_count = bundles[list(_TDNET_FAMILY_FLAG_NAMES)].sum(axis=1)
    bundles["family_count"] = family_count
    bundles = bundles.sort_values(["code", "bundle_start"], kind="stable").reset_index(
        drop=True
    )
    bundles["prior_bundle_count_60"] = _prior_bundle_counts(bundles, days=60)
    bundles["prior_bundle_count_252"] = _prior_bundle_counts(bundles, days=252)

    result = bundles[keys].copy()
    result["tdnet_clean_any"] = 1.0
    direct_map = {
        "tdnet_clean_has_earnings": "earnings",
        "tdnet_clean_has_revision": "revision",
        "tdnet_clean_revision_up_title": "revision_up_title",
        "tdnet_clean_revision_down_title": "revision_down_title",
        "tdnet_clean_revision_direction_unknown": "revision_direction_unknown",
        "tdnet_clean_has_dividend": "dividend",
        "tdnet_clean_dividend_up_title": "dividend_up_title",
        "tdnet_clean_dividend_down_title": "dividend_down_title",
        "tdnet_clean_dividend_direction_unknown": "dividend_direction_unknown",
        "tdnet_clean_has_buyback_decision": "buyback_decision",
        "tdnet_clean_has_buyback_tostnet": "buyback_tostnet",
        "tdnet_clean_has_buyback_status": "buyback_status",
        "tdnet_clean_has_external_equity_financing": "external_equity_financing",
        "tdnet_clean_has_equity_compensation": "equity_compensation",
        "tdnet_clean_has_equity_financing_status": "equity_financing_status",
        "tdnet_clean_has_benefit": "benefit",
        "tdnet_clean_has_split": "split",
        "tdnet_clean_has_ma_transaction": "ma_transaction",
        "tdnet_clean_has_business_alliance": "business_alliance",
        "tdnet_clean_has_control_transaction": "control_transaction",
        "tdnet_clean_has_impairment_loss": "impairment_loss",
        "tdnet_clean_has_audit_problem": "audit_problem",
        "tdnet_clean_has_correction": "correction",
    }
    for output, source in direct_map.items():
        result[output] = bundles[source].astype(float)
    result["tdnet_clean_document_count_log1p"] = np.log1p(
        bundles["document_count"]
    )
    result["tdnet_clean_family_count_log1p"] = np.log1p(family_count)
    for name in ("premarket", "intraday", "postclose"):
        result[f"tdnet_clean_{name}_count_log1p"] = np.log1p(
            bundles[f"{name}_count"]
        )
    result["tdnet_clean_latest_age_hours_log1p"] = np.log1p(
        bundles["latest_age_hours"]
    )
    width_minutes = (
        bundles["bundle_end"] - bundles["bundle_start"]
    ).dt.total_seconds() / 60.0
    result["tdnet_clean_bundle_width_minutes_log1p"] = np.log1p(width_minutes)
    result["tdnet_clean_has_progress_stage"] = bundles["progress_stage"]
    result["tdnet_clean_single_family"] = family_count.eq(1).astype(float)
    result["tdnet_clean_support_adverse_conflict"] = (
        bundles["supportive"].gt(0) & bundles["adverse"].gt(0)
    ).astype(float)
    result["tdnet_clean_prior_bundle_count_60_log1p"] = np.log1p(
        bundles["prior_bundle_count_60"]
    )
    result["tdnet_clean_prior_bundle_count_252_log1p"] = np.log1p(
        bundles["prior_bundle_count_252"]
    )
    result["tdnet_clean_weekend_age"] = bundles["latest_age_hours"].ge(48).astype(
        float
    )
    result["tdnet_clean_feature_source_max_timestamp"] = bundles["bundle_end"]
    result[list(TDNET_CANDIDATE_COLUMNS)] = result[
        list(TDNET_CANDIDATE_COLUMNS)
    ].astype("float32")
    return result[
        [
            "date",
            "code",
            *TDNET_CANDIDATE_COLUMNS,
            "tdnet_clean_feature_source_max_timestamp",
        ]
    ].sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def attach_clean_tdnet_candidate_features(
    panel: pd.DataFrame,
    disclosures: pd.DataFrame,
    expected_sessions: Iterable[object],
    *,
    decision_time: str = "08:58:59",
    completeness_column: str = "tdnet_source_complete",
) -> pd.DataFrame:
    """Attach clean TDnet candidates without treating source failure as zero."""

    _require_columns(panel, ("date", "code", completeness_column), "panel")
    features = build_clean_tdnet_candidate_features(
        disclosures, expected_sessions, decision_time=decision_time
    )
    frame = panel.merge(
        features,
        on=["date", "code"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    complete = frame[completeness_column].eq(True)
    frame.loc[complete, list(TDNET_CANDIDATE_COLUMNS)] = frame.loc[
        complete, list(TDNET_CANDIDATE_COLUMNS)
    ].fillna(0.0)
    frame.loc[~complete, list(TDNET_CANDIDATE_COLUMNS)] = np.nan
    cutoff_time = clock_time.fromisoformat(decision_time)
    cutoffs = pd.Series(
        [
            pd.Timestamp.combine(pd.Timestamp(date).date(), cutoff_time).tz_localize(
                "Asia/Tokyo"
            )
            for date in frame["date"]
        ],
        index=frame.index,
    )
    leaked = frame["tdnet_clean_feature_source_max_timestamp"].notna() & (
        frame["tdnet_clean_feature_source_max_timestamp"] > cutoffs
    )
    if leaked.any():
        raise LeakageError("clean TDnet candidates contain a post-cutoff source")
    return frame


def add_event_context_interactions(panel: pd.DataFrame) -> pd.DataFrame:
    """Add only the six interactions preregistered in the v0.6 catalog."""

    _require_columns(
        panel,
        (
            "xrank_close_momentum_20",
            "prior_market_tail_balance",
            "tdnet_clean_any",
            "tdnet_clean_has_revision",
            "tdnet_clean_has_dividend",
            "tdnet_clean_has_buyback_decision",
            "tdnet_clean_has_external_equity_financing",
            "tdnet_clean_has_ma_transaction",
        ),
        "candidate panel",
    )
    frame = panel.copy()
    runup = pd.to_numeric(frame["xrank_close_momentum_20"], errors="coerce")
    interactions = {
        "tdnet_clean_revision_x_xrank_close_momentum_20": (
            frame["tdnet_clean_has_revision"] * runup
        ),
        "tdnet_clean_dividend_x_xrank_close_momentum_20": (
            frame["tdnet_clean_has_dividend"] * runup
        ),
        "tdnet_clean_buyback_x_xrank_close_momentum_20": (
            frame["tdnet_clean_has_buyback_decision"] * runup
        ),
        "tdnet_clean_external_financing_x_xrank_close_momentum_20": (
            frame["tdnet_clean_has_external_equity_financing"] * runup
        ),
        "tdnet_clean_ma_x_xrank_close_momentum_20": (
            frame["tdnet_clean_has_ma_transaction"] * runup
        ),
        "tdnet_clean_any_x_prior_market_tail_balance": (
            frame["tdnet_clean_any"] * frame["prior_market_tail_balance"]
        ),
    }
    for name, values in interactions.items():
        frame[name] = pd.to_numeric(values, errors="coerce").astype("float32")
    return frame
