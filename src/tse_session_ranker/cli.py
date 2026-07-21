from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from .api import SessionRanker
from .config import RankerConfig
from .data.common import normalize_daily_prices
from .exceptions import SessionRankerError
from .io import read_frame, write_frame, write_json


def _ranker(config_path: str | None = None) -> SessionRanker:
    return (
        SessionRanker.from_config(config_path)
        if config_path
        else SessionRanker(RankerConfig())
    )


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tse-session-ranker",
        description="TSE pre-open session ranking: collect, train, backtest, predict",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-jpx", help="download explicit JPX file URLs")
    download.add_argument(
        "--url", action="append", required=True, help="explicit file URL; repeat as needed"
    )
    download.add_argument("--destination", required=True, help="download directory")
    download.add_argument("--overwrite", action="store_true")

    collect = subparsers.add_parser("collect-jpx", help="parse JPX PDF/TXT quotation files")
    collect.add_argument("--input", nargs="+", required=True, help="PDF/TXT files or directories")
    collect.add_argument("--output", required=True, help="canonical .csv/.pkl/.parquet")
    collect.add_argument("--existing")
    collect.add_argument("--manifest")
    collect.add_argument("--config")

    preopen = subparsers.add_parser(
        "ingest-preopen", help="validate and append MarketSpeed-exported snapshots"
    )
    preopen.add_argument("--input", required=True)
    preopen.add_argument("--output", required=True)
    preopen.add_argument("--existing")
    preopen.add_argument("--config")

    train = subparsers.add_parser("train", help="fit the fixed session_v2 ranker")
    train.add_argument("--daily", required=True)
    train.add_argument("--train-start")
    train.add_argument("--train-end", required=True, help="inclusive YYYY-MM-DD cutoff")
    train.add_argument("--artifact", required=True, help="output .joblib path")
    train.add_argument("--config")

    predict = subparsers.add_parser("predict", help="rank the next TSE session")
    predict.add_argument("--daily", required=True)
    predict.add_argument("--artifact", required=True)
    predict.add_argument("--target-date", required=True, help="session date YYYY-MM-DD")
    predict.add_argument(
        "--expected-history-date",
        required=True,
        help="expected latest completed TSE session YYYY-MM-DD",
    )
    predict.add_argument(
        "--top-k", type=int, default=None, help="override artifact display_top_k"
    )
    predict.add_argument("--preopen")
    predict.add_argument("--as-of", help="timezone-aware cutoff, e.g. ...T08:58:00+09:00")
    predict.add_argument("--output")

    backtest = subparsers.add_parser(
        "backtest", help="run fixed-spec expanding monthly walk-forward"
    )
    backtest.add_argument("--daily", required=True)
    backtest.add_argument("--start", required=True)
    backtest.add_argument("--end", required=True)
    backtest.add_argument("--train-start")
    backtest.add_argument("--output-dir", required=True)
    backtest.add_argument("--config")

    doctor = subparsers.add_parser("doctor", help="validate a canonical daily dataset")
    doctor.add_argument("--daily", required=True)
    return parser


def _run(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download-jpx":
        report = SessionRanker().download_jpx(
            args.url, args.destination, overwrite=args.overwrite
        )
        _print_json(report)
        return
    if args.command == "collect-jpx":
        ranker = _ranker(args.config)
        _, report = ranker.collect_jpx(
            args.input,
            output=args.output,
            existing=args.existing,
            manifest_path=args.manifest,
        )
        _print_json(report)
        return
    if args.command == "ingest-preopen":
        ranker = _ranker(args.config)
        frame = ranker.ingest_preopen(
            args.input, output=args.output, existing=args.existing
        )
        _print_json(
            {
                "rows": len(frame),
                "codes": frame["code"].nunique(),
                "min_observed_at": frame["observed_at"].min(),
                "max_observed_at": frame["observed_at"].max(),
            }
        )
        return
    if args.command == "train":
        ranker = _ranker(args.config)
        result = ranker.train(
            args.daily,
            train_start=args.train_start,
            train_end=args.train_end,
            artifact_path=args.artifact,
        )
        _print_json(result.artifact.manifest)
        return
    if args.command == "predict":
        ranker = SessionRanker.from_artifact(args.artifact)
        if args.preopen and not args.as_of:
            raise SystemExit("--as-of is required with --preopen")
        result = ranker.predict(
            args.daily,
            target_date=args.target_date,
            top_k=args.top_k,
            preopen_snapshots=args.preopen,
            as_of=args.as_of,
            expected_history_date=args.expected_history_date,
        )
        if args.output:
            write_frame(result.candidates, args.output)
        columns = [
            column
            for column in (
                "model_rank",
                "selection_role",
                "code",
                "name",
                "model_score",
                "prior_close",
                "atr14_pct",
                "order_status",
                "veto_reasons",
            )
            if column in result.candidates
        ]
        print(result.candidates[columns].to_string(index=False))
        _print_json(
            {
                "target_date": result.target_date.date(),
                "run_id": result.run_id,
                "eligible_universe_size": result.eligible_universe_size,
                "total_universe_size": result.total_universe_size,
                "calibration_status": result.calibration_status,
                "output": args.output,
            }
        )
        return
    if args.command == "backtest":
        ranker = _ranker(args.config)
        result = ranker.backtest(
            args.daily,
            evaluation_start=args.start,
            evaluation_end=args.end,
            train_start=args.train_start,
        )
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_json(result.summary, output / "summary.json")
        write_frame(result.folds, output / "folds.csv")
        write_frame(result.scores, output / "scores.pkl")
        write_frame(result.picks_top1, output / "top1.csv")
        write_frame(result.picks_top2, output / "top2.csv")
        _print_json(result.summary)
        return
    if args.command == "doctor":
        frame = normalize_daily_prices(read_frame(args.daily))
        by_date = frame.groupby("date")["code"].nunique()
        _print_json(
            {
                "rows": len(frame),
                "codes": frame["code"].nunique(),
                "dates": frame["date"].nunique(),
                "min_date": frame["date"].min().date(),
                "max_date": frame["date"].max().date(),
                "median_codes_per_date": float(by_date.median()),
                "no_trade_rows": int((~frame["traded"]).sum()),
                "partial_session_rows": int(frame["partial_session"].sum()),
                "has_volume": bool(frame["volume"].notna().any()),
                "has_turnover": bool(frame["turnover"].notna().any()),
            }
        )


def main(argv: Sequence[str] | None = None) -> None:
    try:
        _run(argv)
    except (SessionRankerError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
