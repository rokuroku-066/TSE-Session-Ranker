from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from .api import SessionRanker
from .config import RankerConfig
from .data.common import normalize_daily_prices, session_coverage_report
from .exceptions import SessionRankerError
from .io import json_dumps, read_frame, write_frame, write_json


def _ranker(config_path: str | None = None) -> SessionRanker:
    return (
        SessionRanker.from_config(config_path)
        if config_path
        else SessionRanker(RankerConfig())
    )


def _calendar(path: str | None) -> object | None:
    if path is None:
        return None
    source = Path(path)
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
        if "date" not in frame:
            raise ValueError("session calendar CSV requires a date column")
        return frame["date"]
    return [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _print_json(payload: object) -> None:
    print(json_dumps(payload))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tse-session-ranker",
        description="Profit-first TSE pre-open ranking: collect, train, backtest, predict",
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

    download_tdnet = subparsers.add_parser(
        "download-tdnet", help="download public TDnet date-index pages"
    )
    download_tdnet.add_argument("--start", required=True)
    download_tdnet.add_argument("--end", required=True)
    download_tdnet.add_argument("--destination", required=True)
    download_tdnet.add_argument("--workers", type=int, default=4)
    download_tdnet.add_argument("--overwrite", action="store_true")

    download_official_tdnet = subparsers.add_parser(
        "download-tdnet-official",
        help="download explicit official TDnet index/PDF URLs with receipts",
    )
    download_official_tdnet.add_argument(
        "--url",
        action="append",
        required=True,
        help="official release.tdnet.info URL; repeat as needed",
    )
    download_official_tdnet.add_argument(
        "--destination",
        required=True,
        help="raw download directory",
    )
    download_official_tdnet.add_argument("--overwrite", action="store_true")

    probe_tdnet_material = subparsers.add_parser(
        "probe-tdnet-material",
        help="acquire and strictly parse current official forecast revisions",
    )
    probe_tdnet_material.add_argument(
        "--index-url",
        required=True,
        help="official page-001 TDnet index URL",
    )
    probe_tdnet_material.add_argument(
        "--destination",
        required=True,
        help="raw download directory",
    )
    probe_tdnet_material.add_argument(
        "--manifest",
        required=True,
        help="collectibility manifest JSON",
    )
    probe_tdnet_material.add_argument("--overwrite", action="store_true")

    collect_tdnet = subparsers.add_parser(
        "collect-tdnet", help="parse cached TDnet date-index pages"
    )
    collect_tdnet.add_argument("--input", nargs="+", required=True)
    collect_tdnet.add_argument("--output", required=True)
    collect_tdnet.add_argument("--manifest")

    preopen = subparsers.add_parser(
        "ingest-preopen", help="validate and append MarketSpeed-exported snapshots"
    )
    preopen.add_argument("--input", required=True)
    preopen.add_argument("--output", required=True)
    preopen.add_argument("--existing")
    preopen.add_argument("--config")

    market_context = subparsers.add_parser(
        "ingest-market-context",
        help="validate and append exact-date pre-open futures snapshots",
    )
    market_context.add_argument("--input", required=True)
    market_context.add_argument("--output", required=True)
    market_context.add_argument("--existing")
    market_context.add_argument("--config")

    train = subparsers.add_parser(
        "train", help="fit the fixed profit-first session_v3 ranker"
    )
    train.add_argument("--daily", required=True)
    train.add_argument("--train-start")
    train.add_argument("--train-end", required=True, help="inclusive YYYY-MM-DD cutoff")
    train.add_argument("--artifact", required=True, help="output .joblib path")
    train.add_argument("--config")
    train.add_argument(
        "--calendar",
        required=True,
        help="CSV with a date column, or one YYYY-MM-DD session per line",
    )
    train.add_argument(
        "--tdnet-cache", required=True, help="TDnet HTML directory or exported dataset"
    )

    predict = subparsers.add_parser("predict", help="rank the next TSE session")
    predict.add_argument("--daily", required=True)
    predict.add_argument("--artifact", required=True)
    predict.add_argument("--target-date", required=True, help="session date YYYY-MM-DD")
    predict.add_argument(
        "--expected-history-date",
        required=True,
        help=(
            "expected latest completed TSE session YYYY-MM-DD"
        ),
    )
    predict.add_argument(
        "--top-k", type=int, default=None, help="override artifact display_top_k"
    )
    predict.add_argument("--preopen")
    predict.add_argument("--as-of", help="timezone-aware cutoff, e.g. ...T08:58:00+09:00")
    predict.add_argument("--output")
    predict.add_argument(
        "--calendar",
        required=True,
        help="same exchange-session calendar used for artifact training",
    )
    predict.add_argument(
        "--tdnet-cache", required=True, help="TDnet HTML directory or exported dataset"
    )

    backtest = subparsers.add_parser(
        "backtest", help="run fixed-spec expanding monthly walk-forward"
    )
    backtest.add_argument("--daily", required=True)
    backtest.add_argument("--start", required=True)
    backtest.add_argument("--end", required=True)
    backtest.add_argument("--train-start")
    backtest.add_argument("--output-dir", required=True)
    backtest.add_argument("--config")
    backtest.add_argument(
        "--calendar",
        required=True,
        help="CSV with a date column, or one YYYY-MM-DD session per line",
    )
    backtest.add_argument(
        "--tdnet-cache", required=True, help="TDnet HTML directory or exported dataset"
    )

    doctor = subparsers.add_parser("doctor", help="validate a canonical daily dataset")
    doctor.add_argument("--daily", required=True)
    doctor.add_argument("--calendar")
    doctor.add_argument(
        "--expected-through",
        help="last session that should be present; requires --calendar",
    )
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
        ranker = SessionRanker()
        _, report = ranker.collect_jpx(
            args.input,
            output=args.output,
            existing=args.existing,
            manifest_path=args.manifest,
        )
        _print_json(report)
        return
    if args.command == "download-tdnet":
        report = SessionRanker().download_tdnet(
            args.start,
            args.end,
            args.destination,
            overwrite=args.overwrite,
            max_workers=args.workers,
        )
        _print_json(
            {
                "dates": len(report),
                "downloaded": sum(row["status"] == "downloaded" for row in report),
                "refreshed": sum(row["status"] == "refreshed" for row in report),
                "cached": sum(row["status"] == "cached" for row in report),
                "finalized": sum(bool(row["finalized"]) for row in report),
                "provisional": sum(not bool(row["finalized"]) for row in report),
                "destination": args.destination,
            }
        )
        return
    if args.command == "download-tdnet-official":
        report = SessionRanker().download_official_tdnet(
            args.url,
            args.destination,
            overwrite=args.overwrite,
        )
        _print_json(
            {
                "files": len(report),
                "bytes": sum(int(row["bytes"]) for row in report),
                "destination": args.destination,
                "receipts": report,
            }
        )
        return
    if args.command == "probe-tdnet-material":
        report = SessionRanker().probe_tdnet_material(
            args.index_url,
            args.destination,
            overwrite=args.overwrite,
        )
        write_json(report, args.manifest)
        _print_json(
            {
                "index_date": report["index_date"],
                "index_disclosures": report["coverage"][
                    "index_disclosures"
                ],
                "forecast_revision_documents": report["coverage"][
                    "forecast_revision_documents"
                ],
                "strict_extractions": report["coverage"][
                    "strict_extractions"
                ],
                "strict_rejections": report["coverage"][
                    "strict_rejections"
                ],
                "historical_validation_ready": report["readiness"][
                    "historical_validation_ready"
                ],
                "orders_allowed": report["integrity"]["orders_allowed"],
                "manifest": args.manifest,
            }
        )
        return
    if args.command == "collect-tdnet":
        dataset = SessionRanker().collect_tdnet(
            args.input,
            output=args.output,
            manifest_path=args.manifest,
        )
        _print_json(
            {
                "rows": len(dataset.disclosures),
                "codes": dataset.disclosures["code"].nunique(),
                "source_files": dataset.source_files,
                "source_sha256": dataset.source_sha256,
                "complete_from": dataset.complete_dates.min(),
                "complete_through": dataset.complete_dates.max(),
                "observed_at_min": dataset.observed_at_by_date.min(),
                "observed_at_max": dataset.observed_at_by_date.max(),
                "output": args.output,
            }
        )
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
    if args.command == "ingest-market-context":
        ranker = _ranker(args.config)
        frame = ranker.ingest_market_context(
            args.input, output=args.output, existing=args.existing
        )
        _print_json(
            {
                "rows": len(frame),
                "min_date": frame["date"].min(),
                "max_date": frame["date"].max(),
                "min_observed_at": frame["observed_at"].min(),
                "max_observed_at": frame["observed_at"].max(),
                "return_definition": (
                    frame.iloc[0]["return_definition"] if len(frame) else None
                ),
                "output": args.output,
            }
        )
        return
    if args.command == "train":
        ranker = _ranker(args.config)
        result = ranker.train(
            args.daily,
            train_start=args.train_start,
            train_end=args.train_end,
            tdnet_indexes=args.tdnet_cache,
            artifact_path=args.artifact,
            expected_sessions=_calendar(args.calendar),
        )
        sidecar = Path(args.artifact).with_suffix(
            Path(args.artifact).suffix + ".manifest.json"
        )
        public_manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        _print_json(
            {
                **result.artifact.manifest,
                "artifact_path": str(Path(args.artifact)),
                "manifest_path": str(sidecar),
                "artifact_schema_version": public_manifest[
                    "artifact_schema_version"
                ],
                "package_version": public_manifest["package_version"],
                "artifact_sha256": public_manifest["artifact_sha256"],
            }
        )
        return
    if args.command == "predict":
        ranker = SessionRanker.from_artifact(args.artifact)
        if args.preopen and not args.as_of:
            raise SystemExit("--as-of is required with --preopen")
        result = ranker.predict(
            args.daily,
            target_date=args.target_date,
            tdnet_indexes=args.tdnet_cache,
            top_k=args.top_k,
            preopen_snapshots=args.preopen,
            as_of=args.as_of,
            expected_history_date=args.expected_history_date,
            expected_sessions=_calendar(args.calendar),
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
                "tdnet_has_revision_up",
                "tdnet_has_dividend_up",
                "tdnet_has_buyback_decision",
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
                "score_semantics": result.score_semantics,
                "selection_objective": ranker.config.selection_objective,
                "session_calendar_mode": ranker.artifact.manifest[
                    "session_calendar_mode"
                ],
                "tdnet_training_source_sha256": ranker.artifact.manifest[
                    "tdnet_source_sha256"
                ],
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
            tdnet_indexes=args.tdnet_cache,
            train_start=args.train_start,
            expected_sessions=_calendar(args.calendar),
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
        coverage = session_coverage_report(
            frame,
            expected_sessions=_calendar(args.calendar),
            expected_through=args.expected_through,
        )
        incomplete = coverage.loc[~coverage["source_complete"], "date"]
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
                "source_incomplete_dates": [
                    str(pd.Timestamp(value).date()) for value in incomplete
                ],
            }
        )


def main(argv: Sequence[str] | None = None) -> None:
    try:
        _run(argv)
    except (SessionRankerError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
