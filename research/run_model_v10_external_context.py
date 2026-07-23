#!/usr/bin/env python3
"""Evaluate strictly prior US-market context for TSE open-to-close ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PANEL_SHA = "6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb"
PROTOCOL_SHA = "e7d28aa8492c7b478f66b9e0f3c67ac6afa46283a48177fb4c1825aef1305b15"
EQUITY_SHA = "42f0a3257a04def43505c64e79e451a2f587fa4cd99fb90ee3b5fe68d121c0aa"
FX_SHA = "f5eb986d66d7099d4a675ef92cd123e5908b413c78dec23a008366e037d92f49"
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
TDNET_FEATURES = (
    "tdnet_clean_any",
    "tdnet_clean_has_earnings",
    "tdnet_clean_revision_up_title",
    "tdnet_clean_revision_down_title",
    "tdnet_clean_dividend_up_title",
    "tdnet_clean_dividend_down_title",
    "tdnet_clean_has_buyback_decision",
    "tdnet_clean_has_external_equity_financing",
)
FACTOR_COLUMNS = (
    "us_sp500_ret",
    "us_growth_spread",
    "us_vix_change",
    "usdjpy_ret",
)
BETA_SCORE_COLUMNS = (
    "x01_us_market_oc_beta60",
    "x02_us_growth_spread_beta60",
    "x04_vix_change_beta60",
    "x03_usdjpy_beta60",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_series(path: Path, column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["observation_date", column])
    frame["source_date"] = pd.to_datetime(frame.pop("observation_date"), errors="raise")
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[column]).sort_values("source_date")
    frame[f"{column}_ret"] = frame[column].pct_change() * 100.0
    return frame.dropna(subset=[f"{column}_ret"])


def map_strict_prior(
    sessions: pd.DatetimeIndex,
    source: pd.DataFrame,
    value_column: str,
    output_column: str,
) -> pd.DataFrame:
    left = pd.DataFrame({"date": sessions.sort_values()})
    right = source[["source_date", value_column]].sort_values("source_date")
    mapped = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="source_date",
        direction="backward",
        allow_exact_matches=False,
    )
    if mapped["source_date"].isna().any():
        raise ValueError(f"missing prior source for {output_column}")
    if mapped["source_date"].ge(mapped["date"]).any():
        raise ValueError(f"non-prior source for {output_column}")
    mapped = mapped.rename(
        columns={
            "source_date": f"{output_column}_source_date",
            value_column: output_column,
        }
    )
    return mapped


def external_context(
    sessions: pd.DatetimeIndex, equity_path: Path, fx_path: Path
) -> pd.DataFrame:
    sp = read_series(equity_path, "SP500")
    nasdaq = read_series(equity_path, "NASDAQCOM")
    vix = read_series(equity_path, "VIXCLS")
    fx = read_series(fx_path, "DEXJPUS")
    context = map_strict_prior(
        sessions, sp, "SP500_ret", "us_sp500_ret"
    )
    for mapped in (
        map_strict_prior(
            sessions, nasdaq, "NASDAQCOM_ret", "us_nasdaq_ret"
        ),
        map_strict_prior(
            sessions, vix, "VIXCLS_ret", "us_vix_change"
        ),
        map_strict_prior(
            sessions, fx, "DEXJPUS_ret", "usdjpy_ret"
        ),
    ):
        context = context.merge(mapped, on="date", how="left", validate="one_to_one")
    context["us_growth_spread"] = (
        context["us_nasdaq_ret"] - context["us_sp500_ret"]
    )
    context["us_growth_spread_source_date"] = context[
        ["us_nasdaq_ret_source_date", "us_sp500_ret_source_date"]
    ].max(axis=1)
    source_columns = [column for column in context if column.endswith("_source_date")]
    for column in source_columns:
        if context[column].ge(context["date"]).any():
            raise ValueError(f"same-day external source in {column}")
    context["max_external_source_date"] = context[source_columns].max(axis=1)
    context["max_external_age_days"] = (
        context["date"] - context["max_external_source_date"]
    ).dt.days
    return context


def rolling_beta_scores(
    panel: pd.DataFrame, context: pd.DataFrame
) -> pd.DataFrame:
    work = panel[
        [
            "date",
            "code",
            "oc_return_pct",
            "outcome_observed",
            "source_complete",
            "price_eligible",
        ]
    ].copy()
    work["_row"] = np.arange(len(work))
    work = work.merge(
        context[["date", *FACTOR_COLUMNS]],
        on="date",
        how="left",
        validate="many_to_one",
    ).sort_values(["code", "date"], kind="stable")
    valid_outcome = (
        work["outcome_observed"].eq(True)
        & work["source_complete"].eq(True)
        & work["oc_return_pct"].notna()
    )
    output = pd.DataFrame(index=work.index)

    def rolling_mean(series: pd.Series) -> pd.Series:
        return (
            series.groupby(work["code"], sort=False)
            .rolling(60, min_periods=20)
            .mean()
            .reset_index(level=0, drop=True)
        )

    for factor, score_column in zip(
        FACTOR_COLUMNS, BETA_SCORE_COLUMNS, strict=True
    ):
        pair = valid_outcome & work[factor].notna()
        x = work[factor].where(pair).groupby(work["code"], sort=False).shift(1)
        y = (
            work["oc_return_pct"]
            .where(pair)
            .groupby(work["code"], sort=False)
            .shift(1)
        )
        mean_x = rolling_mean(x)
        mean_y = rolling_mean(y)
        covariance = rolling_mean(x * y) - mean_x * mean_y
        variance = rolling_mean(x * x) - mean_x * mean_x
        beta = covariance / variance.where(variance.abs() > 1e-10)
        output[score_column] = beta.clip(-5.0, 5.0) * work[factor]

    for column in BETA_SCORE_COLUMNS:
        eligible_score = output[column].where(work["price_eligible"].eq(True))
        mean = eligible_score.groupby(work["date"], sort=False).transform("mean")
        std = eligible_score.groupby(work["date"], sort=False).transform("std")
        output[f"{column}_z"] = (output[column] - mean) / std.replace(0.0, np.nan)
    output["x05_external_beta_composite"] = output[
        [f"{column}_z" for column in BETA_SCORE_COLUMNS]
    ].mean(axis=1)
    output["_row"] = work["_row"].to_numpy()
    output = output.sort_values("_row").set_index("_row")
    output.index.name = None
    if not output.index.equals(pd.RangeIndex(len(panel))):
        raise ValueError("external beta row restoration failed")
    return output


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
    desired = pd.MultiIndex.from_product(
        [sessions, range(1, top_k + 1)], names=["date", "model_rank"]
    ).to_frame(index=False)
    picks = desired.merge(
        actual[
            [
                "date",
                "model_rank",
                "code",
                "name",
                "model_score",
                "oc_return_pct",
            ]
        ],
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


def metrics(picks: pd.DataFrame) -> dict[str, object]:
    daily20 = daily_returns(picks, 20.0)
    daily40 = daily_returns(picks, 40.0)
    daily60 = daily_returns(picks, 60.0)
    monthly = daily40.groupby(daily40.index.to_period("M")).mean()
    slices = {
        "discovery": ("2024-07-01", "2024-10-31"),
        "confirmation_a": ("2024-11-01", "2025-03-31"),
        "confirmation_b": ("2025-04-01", "2025-07-31"),
    }
    executed = picks["oc_return_pct"].notna()
    by_code = (
        picks.assign(
            net20=picks["oc_return_pct"].fillna(0.0)
            - executed.astype(float) * 0.20
        )
        .dropna(subset=["code"])
        .groupby("code")["net20"]
        .sum()
        .sort_values(ascending=False)
    )
    cash = picks.copy()
    cash.loc[cash["code"].isin(by_code.head(10).index), "oc_return_pct"] = np.nan
    return {
        "mean_pct": {
            "net20": float(daily20.mean()),
            "net40": float(daily40.mean()),
            "net60": float(daily60.mean()),
        },
        "temporal_slices_net40": {
            key: float(daily40.loc[start:end].mean())
            for key, (start, end) in slices.items()
        },
        "positive_months_net40": int((monthly > 0.0).sum()),
        "best_20_days_removed_net20": float(
            daily20.drop(daily20.nlargest(20).index).mean()
        ),
        "top_10_profit_codes_to_cash_net20": float(
            daily_returns(cash, 20.0).mean()
        ),
        "unique_codes": int(picks["code"].nunique()),
        "executed_slots": int(executed.sum()),
    }


def rank_target(training: pd.DataFrame) -> np.ndarray:
    return (
        training["oc_return_pct"]
        .groupby(training["date"], sort=False)
        .rank(method="average", pct=True)
        .mul(2.0)
        .sub(1.0)
        .to_numpy(dtype=float)
    )


def fit_score(
    training: pd.DataFrame,
    scoring: pd.DataFrame,
    columns: tuple[str, ...],
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    x_train = imputer.fit_transform(training.loc[:, list(columns)])
    x_score = imputer.transform(scoring.loc[:, list(columns)])
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_score = scaler.transform(x_score)
    count = training.groupby("date", sort=False)["code"].transform("size")
    weight = 1.0 / count.to_numpy(dtype=float)
    model = Ridge(alpha=1.0)
    model.fit(x_train, rank_target(training), sample_weight=weight)
    return model.predict(x_score)


def model_candidates(
    panel: pd.DataFrame,
    extras: pd.DataFrame,
    context: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    frame = panel[
        [
            "date",
            "code",
            "name",
            "oc_return_pct",
            "price_eligible",
            "price_training_eligible",
            "tdnet_source_complete",
            *BASE_FEATURES,
            *TDNET_FEATURES,
        ]
    ].copy()
    for column in [*BETA_SCORE_COLUMNS, "x05_external_beta_composite"]:
        frame[column] = extras[column].to_numpy()
    frame = frame.merge(
        context[["date", *FACTOR_COLUMNS]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    frame["us_x_overnight_mean60"] = (
        frame["us_sp500_ret"] * frame["overnight_mean_60"]
    )
    frame["growth_x_momentum20"] = (
        frame["us_growth_spread"] * frame["xrank_close_momentum_20"]
    )
    frame["vix_x_atr"] = frame["us_vix_change"] * frame["xrank_atr14_pct"]
    frame["fx_x_momentum20"] = (
        frame["usdjpy_ret"] * frame["xrank_close_momentum_20"]
    )
    for factor in FACTOR_COLUMNS:
        frame[f"tdnet_any_x_{factor}"] = frame["tdnet_clean_any"] * frame[factor]
    external_features = (
        *BASE_FEATURES,
        *BETA_SCORE_COLUMNS,
        "x05_external_beta_composite",
        "us_x_overnight_mean60",
        "growth_x_momentum20",
        "vix_x_atr",
        "fx_x_momentum20",
    )
    tdnet_external_features = (
        *external_features,
        *TDNET_FEATURES,
        "tdnet_source_complete",
        *(f"tdnet_any_x_{factor}" for factor in FACTOR_COLUMNS),
    )
    candidates = {
        "X00_price_core_control": BASE_FEATURES,
        "X06_external_context_rank_ridge": external_features,
        "X07_tdnet_external_shock_interaction": tdnet_external_features,
    }
    parts: dict[str, list[pd.DataFrame]] = {
        f"{candidate}__top{top_k}": []
        for candidate in [*candidates, "X08_us_shock_regime_experts"]
        for top_k in (1, 2)
    }
    folds: list[dict[str, object]] = []
    periods = pd.period_range(SCORE_START.to_period("M"), SCORE_END.to_period("M"))
    for period in periods:
        start = max(period.start_time.normalize(), SCORE_START)
        end = min(period.end_time.normalize(), SCORE_END)
        training = frame.loc[
            frame["date"].between(TRAIN_START, start - pd.Timedelta(days=1))
            & frame["price_training_eligible"].eq(True)
            & frame["oc_return_pct"].notna()
        ].copy()
        scoring = frame.loc[
            frame["date"].between(start, end)
            & frame["price_eligible"].eq(True)
        ].copy()
        for candidate_id, columns in candidates.items():
            score = fit_score(training, scoring, tuple(columns))
            for top_k in (1, 2):
                full_id = f"{candidate_id}__top{top_k}"
                parts[full_id].append(
                    rank_scores(scoring, score, full_id, top_k)
                )
            folds.append(
                {
                    "period": str(period),
                    "candidate_id": candidate_id,
                    "train_end": str(training["date"].max().date()),
                    "strictly_prior": True,
                }
            )

        train_regime = pd.cut(
            training["us_sp500_ret"],
            bins=[-np.inf, -0.25, 0.25, np.inf],
            labels=["negative", "neutral", "positive"],
        )
        score_regime = pd.cut(
            scoring["us_sp500_ret"],
            bins=[-np.inf, -0.25, 0.25, np.inf],
            labels=["negative", "neutral", "positive"],
        )
        regime_score = np.full(len(scoring), np.nan, dtype=float)
        for regime in ("negative", "neutral", "positive"):
            train_mask = train_regime.eq(regime).to_numpy()
            score_mask = score_regime.eq(regime).to_numpy()
            if train_mask.sum() < 1000 or not score_mask.any():
                continue
            regime_score[score_mask] = fit_score(
                training.loc[train_mask],
                scoring.loc[score_mask],
                tuple(BASE_FEATURES),
            )
        for top_k in (1, 2):
            full_id = f"X08_us_shock_regime_experts__top{top_k}"
            parts[full_id].append(
                rank_scores(scoring, regime_score, full_id, top_k)
            )
        folds.append(
            {
                "period": str(period),
                "candidate_id": "X08_us_shock_regime_experts",
                "train_end": str(training["date"].max().date()),
                "strictly_prior": True,
            }
        )
    output: dict[str, pd.DataFrame] = {}
    for candidate_id, candidate_parts in parts.items():
        top_k = int(candidate_id.rsplit("top", 1)[1])
        output[candidate_id] = complete_picks(
            pd.concat(candidate_parts, ignore_index=True),
            sessions,
            candidate_id,
            top_k,
        )
    return output, folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="/tmp/model_v07_corrected_panel.pkl")
    parser.add_argument(
        "--protocol",
        default="/workspace/scratch/8678b1d14f37/research/model_v10_external_context_protocol.json",
    )
    parser.add_argument(
        "--equity-context", default="/tmp/v10_fred/daily,_close.csv"
    )
    parser.add_argument("--fx-context", default="/tmp/v10_fred/daily.csv")
    parser.add_argument("--output", default="/tmp/v10_external_context/result.json")
    parser.add_argument("--picks-output", default="/tmp/v10_external_context/picks.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel_path = Path(args.panel)
    protocol_path = Path(args.protocol)
    equity_path = Path(args.equity_context)
    fx_path = Path(args.fx_context)
    expected = (
        (panel_path, PANEL_SHA),
        (protocol_path, PROTOCOL_SHA),
        (equity_path, EQUITY_SHA),
        (fx_path, FX_SHA),
    )
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise ValueError(f"SHA-256 changed: {path}")
    panel = joblib.load(panel_path, mmap_mode="r")
    all_sessions = pd.DatetimeIndex(
        pd.to_datetime(panel["date"]).drop_duplicates().sort_values()
    )
    sessions = all_sessions[
        (all_sessions >= SCORE_START) & (all_sessions <= SCORE_END)
    ]
    if len(sessions) != 266:
        raise ValueError("score session count changed")
    context = external_context(all_sessions, equity_path, fx_path)
    extras = rolling_beta_scores(panel, context)
    scoring = panel.loc[
        panel["date"].between(SCORE_START, SCORE_END)
        & panel["price_eligible"].eq(True),
        ["date", "code", "name", "oc_return_pct"],
    ].copy()
    scoring_extras = extras.loc[scoring.index]
    picks: dict[str, pd.DataFrame] = {}
    for hypothesis, column in (
        ("X01_us_market_oc_beta60", "x01_us_market_oc_beta60"),
        ("X02_us_growth_spread_beta60", "x02_us_growth_spread_beta60"),
        ("X03_usdjpy_beta60", "x03_usdjpy_beta60"),
        ("X04_vix_change_beta60", "x04_vix_change_beta60"),
        ("X05_external_beta_composite", "x05_external_beta_composite"),
    ):
        for top_k in (1, 2):
            candidate_id = f"{hypothesis}__top{top_k}"
            actual = rank_scores(
                scoring,
                scoring_extras[column],
                candidate_id,
                top_k,
            )
            picks[candidate_id] = complete_picks(
                actual, sessions, candidate_id, top_k
            )
    model_output, folds = model_candidates(
        panel, extras, context, sessions
    )
    picks.update(model_output)
    all_picks = pd.concat(picks.values(), ignore_index=True)
    output_path = Path(args.output)
    picks_path = Path(args.picks_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_picks.to_csv(picks_path, index=False)
    result = {
        "schema_version": 1,
        "protocol_id": "model_v10_external_context_gen1_20260723",
        "authority": "retrospective_new_data_source_screen_no_promotion",
        "protocol_sha256": PROTOCOL_SHA,
        "panel_sha256": PANEL_SHA,
        "runner_sha256": sha256_file(Path(__file__)),
        "source_mapping": {
            "strictly_prior_all": bool(
                context["max_external_source_date"].lt(context["date"]).all()
            ),
            "max_age_days": int(context["max_external_age_days"].max()),
            "source_date_max": str(context["max_external_source_date"].max().date()),
        },
        "candidate_count": int(len(picks)),
        "candidates": {
            candidate_id: metrics(candidate_picks)
            for candidate_id, candidate_picks in picks.items()
        },
        "folds": folds,
        "picks_path": str(picks_path),
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
