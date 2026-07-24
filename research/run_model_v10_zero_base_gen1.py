#!/usr/bin/env python3
"""Independent v0.10 generation-1 screen on the frozen research panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PANEL_SHA256 = "6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb"
PROTOCOL_SHA256 = "f8867299501d295770c52ecd6437195802ef1b7728e07107f47c620f4d01ad18"
TRAIN_START = pd.Timestamp("2024-01-04")
SCORE_START = pd.Timestamp("2024-07-01")
SCORE_END = pd.Timestamp("2025-07-31")
BASE_FEATURES = (
    "oc_last",
    "oc_mean_5",
    "oc_mean_20",
    "oc_mean_60",
    "oc_win_20",
    "oc_std_20",
    "overnight_last",
    "overnight_mean_20",
    "overnight_mean_60",
    "night_day_corr_60",
    "xrank_atr14_pct",
    "xrank_close_momentum_5",
    "xrank_close_momentum_20",
    "xrank_close_momentum_60",
    "xrank_prior_close_location_20",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_panel(panel: pd.DataFrame) -> None:
    required = {
        "date",
        "code",
        "name",
        "oc_return_pct",
        "outcome_observed",
        "source_complete",
        "price_eligible",
        "price_training_eligible",
        "candidate_price_source_max_date",
        *BASE_FEATURES,
    }
    missing = sorted(required - set(panel))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if panel.duplicated(["date", "code"]).any():
        raise ValueError("duplicate date/code rows")
    dates = pd.to_datetime(panel["date"], errors="raise")
    source = pd.to_datetime(panel["candidate_price_source_max_date"], errors="coerce")
    if (source.notna() & source.ge(dates)).any():
        raise ValueError("non-prior feature source")


def desired_slots(sessions: pd.DatetimeIndex, top_k: int) -> pd.DataFrame:
    return pd.MultiIndex.from_product(
        [sessions, range(1, top_k + 1)], names=["date", "model_rank"]
    ).to_frame(index=False)


def rank_scores(
    scoring: pd.DataFrame,
    score: np.ndarray | pd.Series,
    candidate_id: str,
    top_k: int,
) -> pd.DataFrame:
    ranked = scoring.assign(model_score=np.asarray(score, dtype=float))
    ranked = ranked[np.isfinite(ranked["model_score"])].sort_values(
        ["date", "model_score", "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    ranked = ranked.groupby("date", sort=True, as_index=False).head(top_k).copy()
    ranked["model_rank"] = ranked.groupby("date", sort=False).cumcount() + 1
    ranked["candidate_id"] = candidate_id
    return ranked


def complete_picks(
    actual: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    candidate_id: str,
    top_k: int,
) -> pd.DataFrame:
    keep = [
        "date",
        "model_rank",
        "code",
        "name",
        "model_score",
        "oc_return_pct",
    ]
    picks = desired_slots(sessions, top_k).merge(
        actual[keep],
        on=["date", "model_rank"],
        how="left",
        validate="one_to_one",
        sort=True,
    )
    picks["candidate_id"] = candidate_id
    picks["top_k"] = top_k
    return picks


def daily_returns(picks: pd.DataFrame, cost_bps: float) -> pd.Series:
    executed = picks["oc_return_pct"].notna()
    slot = picks["oc_return_pct"].fillna(0.0) - executed.astype(float) * cost_bps / 100.0
    return slot.groupby(picks["date"], sort=True).sum().div(int(picks["top_k"].iloc[0]))


def candidate_metrics(picks: pd.DataFrame) -> dict[str, object]:
    daily = {cost: daily_returns(picks, float(cost)) for cost in (20, 40, 60)}
    net20 = daily[20]
    net40 = daily[40]
    slices = {
        "discovery": ("2024-07-01", "2024-10-31"),
        "confirmation_a": ("2024-11-01", "2025-03-31"),
        "confirmation_b": ("2025-04-01", "2025-07-31"),
    }
    executed = picks["oc_return_pct"].notna()
    slot20 = picks["oc_return_pct"].fillna(0.0) - executed.astype(float) * 0.20
    by_code = (
        picks.assign(net20_slot=slot20)
        .dropna(subset=["code"])
        .groupby("code", sort=False)["net20_slot"]
        .sum()
        .sort_values(ascending=False)
    )
    top_codes = set(by_code.head(10).index)
    without_codes = picks.copy()
    mask = without_codes["code"].isin(top_codes)
    without_codes.loc[mask, "oc_return_pct"] = np.nan
    monthly = net40.groupby(net40.index.to_period("M")).mean()
    return {
        "top_k": int(picks["top_k"].iloc[0]),
        "scheduled_days": int(net20.size),
        "executed_slots": int(executed.sum()),
        "unique_codes": int(picks["code"].nunique()),
        "mean_pct": {
            "net20": float(net20.mean()),
            "net40": float(net40.mean()),
            "net60": float(daily[60].mean()),
        },
        "temporal_slices_net40": {
            name: float(net40.loc[start:end].mean())
            for name, (start, end) in slices.items()
        },
        "monthly_net40": {str(key): float(value) for key, value in monthly.items()},
        "positive_months_net40": int((monthly > 0.0).sum()),
        "best_days_removed_net20": {
            str(count): float(net20.drop(net20.nlargest(count).index).mean())
            for count in (5, 10, 20)
        },
        "worst_days_removed_net20": {
            str(count): float(net20.drop(net20.nsmallest(count).index).mean())
            for count in (5, 10, 20)
        },
        "top_10_profit_codes_to_cash_net20": float(
            daily_returns(without_codes, 20.0).mean()
        ),
        "largest_profit_code_share": (
            float(by_code.iloc[0] / by_code.sum())
            if len(by_code) and by_code.sum() > 0.0
            else None
        ),
    }


def add_literature_features(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel[
        [
            "date",
            "code",
            "name",
            "oc_return_pct",
            "outcome_observed",
            "source_complete",
            "price_eligible",
        ]
    ].copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["code", "date"], kind="stable")
    valid = (
        work["outcome_observed"].eq(True)
        & work["source_complete"].eq(True)
        & work["oc_return_pct"].notna()
    )
    observed = work["oc_return_pct"].where(valid)
    previous = observed.groupby(work["code"], sort=False).shift(1)
    previous_group = previous.groupby(work["code"], sort=False)
    work["z02_score"] = -(
        previous_group.rolling(20, min_periods=10).max().reset_index(level=0, drop=True)
    )
    mean60 = (
        previous_group.rolling(60, min_periods=20).mean().reset_index(level=0, drop=True)
    )
    downside_square = previous.clip(upper=0.0).pow(2)
    downside = (
        downside_square.groupby(work["code"], sort=False)
        .rolling(60, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
        .pow(0.5)
    )
    work["z03_score"] = mean60 - 0.5 * downside

    work["weekday"] = work["date"].dt.weekday
    current = observed.fillna(0.0)
    current_count = observed.notna().astype(int)
    keys = [work["code"], work["weekday"]]
    prior_sum = current.groupby(keys, sort=False).cumsum() - current
    prior_count = current_count.groupby(keys, sort=False).cumsum() - current_count

    daily_market = (
        work.loc[valid, ["date", "oc_return_pct"]]
        .groupby("date", sort=True)["oc_return_pct"]
        .mean()
        .rename("market_oc")
        .to_frame()
    )
    daily_market["weekday"] = daily_market.index.weekday
    daily_market["prior_weekday_sum"] = (
        daily_market.groupby("weekday", sort=False)["market_oc"].cumsum()
        - daily_market["market_oc"]
    )
    daily_market["prior_weekday_count"] = (
        daily_market.groupby("weekday", sort=False).cumcount()
    )
    daily_market["prior_weekday_mean"] = (
        daily_market["prior_weekday_sum"]
        / daily_market["prior_weekday_count"].replace(0, np.nan)
    )
    prior_market = work["date"].map(daily_market["prior_weekday_mean"])
    work["z01_score"] = (
        prior_sum + 20.0 * prior_market
    ) / (prior_count + 20.0)
    return work.sort_values(["date", "code"], kind="stable")


def simple_signal_picks(
    panel: pd.DataFrame, sessions: pd.DatetimeIndex
) -> dict[str, pd.DataFrame]:
    features = add_literature_features(panel)
    scoring = features.loc[
        features["date"].between(SCORE_START, SCORE_END)
        & features["price_eligible"].eq(True)
    ].copy()
    output: dict[str, pd.DataFrame] = {}
    for hypothesis, column in (
        ("Z01_code_weekday_empirical_bayes", "z01_score"),
        ("Z02_max_lottery_20_penalty", "z02_score"),
        ("Z03_left_tail_adjusted_alpha_60", "z03_score"),
    ):
        for top_k in (1, 2):
            candidate_id = f"{hypothesis}__top{top_k}"
            actual = rank_scores(scoring, scoring[column], candidate_id, top_k)
            output[candidate_id] = complete_picks(
                actual, sessions, candidate_id, top_k
            )
    return output


def daily_zscore_target(training: pd.DataFrame) -> np.ndarray:
    returns = training["oc_return_pct"].astype(float)
    grouped = returns.groupby(training["date"], sort=False)
    lower = grouped.transform(lambda values: values.quantile(0.05))
    upper = grouped.transform(lambda values: values.quantile(0.95))
    clipped = returns.clip(lower=lower, upper=upper)
    mean = clipped.groupby(training["date"], sort=False).transform("mean")
    std = clipped.groupby(training["date"], sort=False).transform("std")
    target = (clipped - mean) / std.replace(0.0, np.nan)
    return target.fillna(0.0).clip(-3.0, 3.0).to_numpy(dtype=float)


def daily_rank_target(training: pd.DataFrame) -> np.ndarray:
    return (
        training["oc_return_pct"]
        .groupby(training["date"], sort=False)
        .rank(method="average", pct=True)
        .mul(2.0)
        .sub(1.0)
        .to_numpy(dtype=float)
    )


def model_picks(
    panel: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    all_sessions: pd.DatetimeIndex,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    projection = [
        "date",
        "code",
        "name",
        "oc_return_pct",
        "price_eligible",
        "price_training_eligible",
        *BASE_FEATURES,
    ]
    frame = panel.loc[:, projection]
    periods = pd.period_range(SCORE_START.to_period("M"), SCORE_END.to_period("M"))
    candidate_ids = [
        "Z05_daily_zscore_ridge",
        "Z06_exp_decay_rank_ridge_hl60",
        "Z07_exp_decay_rank_ridge_hl120",
    ]
    parts: dict[str, list[pd.DataFrame]] = {
        f"{candidate}__top{top_k}": []
        for candidate in candidate_ids
        for top_k in (1, 2)
    }
    folds: list[dict[str, object]] = []
    session_number = {date: index for index, date in enumerate(all_sessions)}
    for period in periods:
        score_start = max(period.start_time.normalize(), SCORE_START)
        score_end = min(period.end_time.normalize(), SCORE_END)
        training = frame.loc[
            frame["date"].between(TRAIN_START, score_start - pd.Timedelta(days=1))
            & frame["price_training_eligible"].eq(True)
            & frame["oc_return_pct"].notna()
        ].copy()
        scoring = frame.loc[
            frame["date"].between(score_start, score_end)
            & frame["price_eligible"].eq(True)
        ].copy()
        if training.empty or training["date"].max() >= score_start:
            raise ValueError(f"invalid prior training fold {period}")
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        x_train = imputer.fit_transform(training.loc[:, list(BASE_FEATURES)])
        x_score = imputer.transform(scoring.loc[:, list(BASE_FEATURES)])
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_score = scaler.transform(x_score)
        date_count = training.groupby("date", sort=False)["code"].transform("size")
        equal_date_weight = 1.0 / date_count.to_numpy(dtype=float)
        score_ordinal = session_number[sessions[sessions.to_period("M") == period][0]]
        train_ordinal = training["date"].map(session_number).to_numpy(dtype=float)
        age = score_ordinal - train_ordinal
        if np.any(age <= 0):
            raise ValueError("non-prior age in exponential fold")
        targets_and_weights = {
            "Z05_daily_zscore_ridge": (
                daily_zscore_target(training),
                equal_date_weight,
            ),
            "Z06_exp_decay_rank_ridge_hl60": (
                daily_rank_target(training),
                equal_date_weight * np.exp(-math.log(2.0) * age / 60.0),
            ),
            "Z07_exp_decay_rank_ridge_hl120": (
                daily_rank_target(training),
                equal_date_weight * np.exp(-math.log(2.0) * age / 120.0),
            ),
        }
        for candidate_id, (target, weight) in targets_and_weights.items():
            model = Ridge(alpha=1.0)
            model.fit(x_train, target, sample_weight=weight)
            score = model.predict(x_score)
            for top_k in (1, 2):
                full_id = f"{candidate_id}__top{top_k}"
                parts[full_id].append(
                    rank_scores(scoring, score, full_id, top_k)
                )
            folds.append(
                {
                    "period": str(period),
                    "candidate_id": candidate_id,
                    "train_start": str(training["date"].min().date()),
                    "train_end": str(training["date"].max().date()),
                    "train_rows": int(len(training)),
                    "score_rows": int(len(scoring)),
                    "strictly_prior": True,
                }
            )
        del x_train, x_score, training, scoring
    output: dict[str, pd.DataFrame] = {}
    for candidate_id, candidate_parts in parts.items():
        top_k = int(candidate_id.rsplit("top", 1)[1])
        actual = pd.concat(candidate_parts, ignore_index=True)
        output[candidate_id] = complete_picks(
            actual, sessions, candidate_id, top_k
        )
    return output, folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="/tmp/model_v07_corrected_panel.pkl")
    parser.add_argument(
        "--protocol",
        default="/workspace/scratch/8678b1d14f37/research/model_v10_zero_base_protocol.json",
    )
    parser.add_argument("--output", default="/tmp/v10_root_gen1/result.json")
    parser.add_argument("--picks-output", default="/tmp/v10_root_gen1/picks.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel_path = Path(args.panel)
    protocol_path = Path(args.protocol)
    if sha256_file(panel_path) != PANEL_SHA256:
        raise ValueError("panel SHA-256 changed")
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("protocol SHA-256 changed")
    panel = joblib.load(panel_path, mmap_mode="r")
    validate_panel(panel)
    all_sessions = pd.DatetimeIndex(
        pd.to_datetime(panel["date"]).drop_duplicates().sort_values()
    )
    sessions = all_sessions[
        (all_sessions >= SCORE_START) & (all_sessions <= SCORE_END)
    ]
    if len(sessions) != 266:
        raise ValueError("score session count changed")
    picks_by_candidate = simple_signal_picks(panel, sessions)
    model_output, folds = model_picks(panel, sessions, all_sessions)
    picks_by_candidate.update(model_output)
    all_picks = pd.concat(picks_by_candidate.values(), ignore_index=True)
    output_path = Path(args.output)
    picks_path = Path(args.picks_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_picks.to_csv(picks_path, index=False)
    payload = {
        "schema_version": 1,
        "protocol_id": "model_v10_zero_base_gen1_20260723",
        "authority": "retrospective_method_screen_no_production_promotion",
        "protocol_sha256": PROTOCOL_SHA256,
        "panel_sha256": PANEL_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "score_sessions": int(len(sessions)),
        "candidate_count": int(len(picks_by_candidate)),
        "candidates": {
            candidate_id: candidate_metrics(picks)
            for candidate_id, picks in picks_by_candidate.items()
        },
        "folds": folds,
        "picks_path": str(picks_path),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
