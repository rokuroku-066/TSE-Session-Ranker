#!/usr/bin/env python3
"""Staged, profit-first comparison of broad model families.

The runner is research-only.  It never writes a production artifact and it
retains two cash slots for every scheduled exchange session, including days on
which no candidate can be scored.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research import finalize_logit_v04 as feature_base  # noqa: E402
from tse_session_ranker.config import RankerConfig  # noqa: E402
from tse_session_ranker.data.common import (  # noqa: E402
    normalize_expected_sessions,
    prepare_modeling_prices,
    session_calendar_hash,
)
from tse_session_ranker.data.jpx import (  # noqa: E402
    PARSER_VERSION as JPX_PARSER_VERSION,
    collect_jpx,
)
from tse_session_ranker.data.tdnet import (  # noqa: E402
    TDNET_FEATURE_COLUMNS,
    TDNET_INDEX_URL,
    collect_tdnet_dataset,
    merge_tdnet_features,
    tdnet_target_completeness,
)
from tse_session_ranker.features import PRICE_FEATURE_COLUMNS, build_feature_panel  # noqa: E402
from tse_session_ranker.io import read_frame, write_frame, write_json  # noqa: E402
from tse_session_ranker.profit import daily_portfolio_returns, profit_metrics  # noqa: E402
from tse_session_ranker.research_models import (  # noqa: E402
    ResearchModelSpec,
    fit_research_model,
)
from tse_session_ranker.validation import (  # noqa: E402
    moving_block_bootstrap,
)


PANEL_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
PRIMARY_COST_BPS = 20.0
STRESS_COST_BPS = 40.0
TRAIN_START = pd.Timestamp("2024-01-04")
MIN_TRAIN_SESSIONS = 60
TOP_K = 2
TDNET_REQUIRED_THROUGH = pd.Timestamp("2025-03-31")
EXPECTED_CALENDAR_SHA256 = (
    "966f4a416d0929487d850b16be87c9c1cd69cd9f8c0447a3e76214a5715c489e"
)
MINIMUM_DISPLAY_RATE = 0.95


@dataclass(frozen=True)
class CandidateSpec:
    model: ResearchModelSpec
    feature_block: str

    @property
    def candidate_id(self) -> str:
        return f"{self.model.spec_id}__f_{self.feature_block}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model": self.model.canonical_dict(),
            "model_spec_id": self.model.spec_id,
            "feature_block": self.feature_block,
        }


@dataclass(frozen=True)
class EnsembleSpec:
    members: tuple[CandidateSpec, ...]
    source_policy: str = "member_specific_training_common_scoring_intersection"

    @property
    def candidate_id(self) -> str:
        encoded = (
            self.source_policy
            + "\n"
            + "\n".join(member.candidate_id for member in self.members)
        ).encode()
        return f"equal_rank_ensemble__{hashlib.sha256(encoded).hexdigest()[:16]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family": "equal_within_date_percentile_rank_ensemble",
            "source_policy": self.source_policy,
            "members": [member.to_dict() for member in self.members],
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_file_hashes(directory: Path, pattern: str) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(directory.glob(pattern))
        if path.is_file()
    }


def content_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"content hash lacks columns: {missing}")
    values = pd.util.hash_pandas_object(frame[list(columns)], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def streaming_content_hash(
    frame: pd.DataFrame, columns: Sequence[str], *, chunk_rows: int = 50_000
) -> str:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"streaming content hash lacks columns: {missing}")
    digest = hashlib.sha256()
    digest.update(json.dumps(list(columns), ensure_ascii=False).encode("utf-8"))
    digest.update(
        json.dumps(
            {column: str(frame[column].dtype) for column in columns},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    )
    for start in range(0, len(frame), chunk_rows):
        values = pd.util.hash_pandas_object(
            frame.iloc[start : start + chunk_rows].loc[:, list(columns)],
            index=False,
        ).to_numpy()
        digest.update(values.tobytes())
    return digest.hexdigest()


def feature_blocks() -> dict[str, tuple[str, ...]]:
    frozen = feature_base.feature_blocks()
    return {
        "legacy_price_12": tuple(PRICE_FEATURE_COLUMNS),
        "legacy_v03_18": frozen["legacy_v03"],
        "session_market": frozen["session_market"],
        "session_market_tdnet": frozen["session_market_tdnet"],
    }


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    raw = json.loads(payload.decode("utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported model-v05 protocol schema")
    if raw.get("status") != "frozen_before_model_v05_metrics":
        raise ValueError("model-v05 protocol is not frozen")
    registered = [ResearchModelSpec.from_dict(item) for item in raw["model_registry"]]
    names = [item.name for item in registered]
    if len(names) != len(set(names)):
        raise ValueError("model-v05 protocol has duplicate model names")
    requested_blocks = tuple(raw["feature_blocks"])
    unknown = sorted(set(requested_blocks) - set(feature_blocks()))
    if unknown:
        raise ValueError(f"protocol has unknown feature blocks: {unknown}")
    grouped_names = [
        str(name)
        for values in raw["family_groups"].values()
        for name in values
    ]
    if len(grouped_names) != len(set(grouped_names)) or set(grouped_names) != set(names):
        raise ValueError("every registered model must belong to exactly one family group")
    if (
        float(raw["stage_policy"]["minimum_rank1_and_rank2_display_rate"])
        != MINIMUM_DISPLAY_RATE
    ):
        raise ValueError("protocol and implementation display-rate gates differ")
    return raw, digest


def _implementation_hashes() -> dict[str, str]:
    research_relatives = (
        "research/compare_model_families_v05.py",
        "research/finalize_logit_v04.py",
    )
    source_relatives = tuple(
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "src/tse_session_ranker").rglob("*.py"))
    )
    relatives = (*research_relatives, *source_relatives)
    return {relative: sha256_file(ROOT / relative) for relative in relatives}


def _block_uses_tdnet(feature_block: str) -> bool:
    return any(column.startswith("tdnet_") for column in feature_blocks()[feature_block])


def _panel_manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _raw_input_hashes(
    jpx_directory: Path,
    tdnet_directory: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    return {
        "jpx_pdf_sha256": directory_file_hashes(jpx_directory, "*.pdf"),
        "tdnet_html_sha256": directory_file_hashes(tdnet_directory, "*.html"),
        "tdnet_metadata_sha256": directory_file_hashes(
            tdnet_directory, "*.html.meta.json"
        ),
        "protocol_sha256": sha256_file(protocol_path),
        "implementation_sha256": _implementation_hashes(),
    }


def _pdftotext_version() -> str:
    completed = subprocess.run(
        ["pdftotext", "-v"],
        check=True,
        capture_output=True,
        text=True,
    )
    return (completed.stderr or completed.stdout).splitlines()[0].strip()


def _runtime_fingerprint() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "pdftotext": _pdftotext_version(),
    }


def _validate_cached_panel(
    panel: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    manifest: Mapping[str, Any],
) -> None:
    expected_columns = list(manifest["panel_columns"])
    if list(panel.columns) != expected_columns:
        raise ValueError("model-v05 panel columns changed")
    observed_dtypes = {column: str(panel[column].dtype) for column in panel.columns}
    if observed_dtypes != manifest["panel_dtypes"]:
        raise ValueError("model-v05 panel dtypes changed")
    if len(panel) != int(manifest["rows"]):
        raise ValueError("model-v05 panel row count changed")
    if panel["code"].nunique() != int(manifest["codes"]):
        raise ValueError("model-v05 panel code count changed")
    if panel.duplicated(["date", "code"]).any():
        raise ValueError("model-v05 cached panel has duplicate date/code rows")
    if session_calendar_hash(sessions, through=sessions.max()) != EXPECTED_CALENDAR_SHA256:
        raise ValueError("model-v05 cached calendar changed")
    leaked_price = panel["feature_source_max_date"].notna() & (
        panel["feature_source_max_date"] >= panel["date"]
    )
    if leaked_price.any():
        raise ValueError("model-v05 cached panel has non-prior price features")
    tdnet_source = pd.to_datetime(
        panel["tdnet_feature_source_max_timestamp"], errors="coerce", utc=True
    )
    target_cutoff = (
        pd.to_datetime(panel["date"], errors="raise")
        .dt.tz_localize("Asia/Tokyo")
        + pd.Timedelta(hours=8, minutes=58, seconds=59)
    ).dt.tz_convert("UTC")
    if (tdnet_source.notna() & tdnet_source.gt(target_cutoff)).any():
        raise ValueError("model-v05 cached panel has post-cutoff TDnet features")
    observed_semantic = streaming_content_hash(panel, panel.columns)
    if observed_semantic != manifest["panel_semantic_sha256"]:
        raise ValueError("model-v05 panel semantic checksum mismatch")


def _build_panel(
    jpx_directory: Path,
    tdnet_directory: Path,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, Any]]:
    pdfs = sorted(jpx_directory.glob("*.pdf"))
    if len(pdfs) != 19 or pdfs[0].name != "202401.pdf" or pdfs[-1].name != "202507.pdf":
        raise ValueError("model-v05 requires exactly 202401..202507 JPX monthly PDFs")
    print(f"parse JPX PDFs: {len(pdfs)}", flush=True)
    canonical, jpx_report = collect_jpx(pdfs)
    if jpx_report.get("parser_version") != JPX_PARSER_VERSION:
        raise ValueError("JPX parser report version does not match the imported parser")
    rejected_rows = sum(
        int(item.get("rejected_rows", 0))
        for item in jpx_report.get("inputs", [])
    )
    if rejected_rows != 0:
        raise ValueError("JPX parser rejected ordinary-stock rows")
    sessions = normalize_expected_sessions(canonical["date"].drop_duplicates())
    if sessions.min() != pd.Timestamp("2024-01-04") or sessions.max() != pd.Timestamp(
        "2025-07-31"
    ):
        raise ValueError("JPX calendar has an unexpected boundary")
    if len(sessions) != 386:
        raise ValueError(f"JPX calendar expected 386 sessions, found {len(sessions)}")
    observed_calendar_hash = session_calendar_hash(sessions, through=sessions.max())
    if observed_calendar_hash != EXPECTED_CALENDAR_SHA256:
        raise ValueError(
            "JPX sessions do not match the preregistered exchange calendar hash"
        )

    settings = RankerConfig()
    modeling, coverage = prepare_modeling_prices(
        canonical,
        coverage_lookback=settings.source_coverage_lookback,
        minimum_source_coverage=settings.minimum_source_coverage,
        expected_sessions=sessions,
    )
    incomplete_price = coverage.loc[~coverage["source_complete"], "date"]
    if len(incomplete_price):
        raise ValueError(
            "official JPX reconstruction has incomplete sessions: "
            + ", ".join(str(value.date()) for value in incomplete_price[:5])
        )
    del canonical
    gc.collect()

    print("build prior-price features", flush=True)
    panel = build_feature_panel(modeling, settings)
    del modeling
    gc.collect()
    panel = feature_base.add_bounded_daily_features(panel)
    panel = feature_base.add_session_market_features(panel)
    panel["price_eligible"] = panel["eligible"]
    panel["price_training_eligible"] = panel["training_eligible"]

    print("parse and attach TDnet features", flush=True)
    tdnet_dataset = collect_tdnet_dataset(tdnet_directory)
    tdnet_coverage = tdnet_target_completeness(
        sessions,
        tdnet_dataset.complete_dates,
        tdnet_dataset.observed_at_by_date,
        decision_time=settings.preopen.decision_time,
    )
    required_tdnet_sessions = sessions[
        (sessions > sessions.min()) & (sessions <= TDNET_REQUIRED_THROUGH)
    ]
    incomplete_tdnet = required_tdnet_sessions[
        ~tdnet_coverage.reindex(required_tdnet_sessions, fill_value=False).to_numpy()
    ]
    if len(incomplete_tdnet):
        raise ValueError(
            "TDnet cache is incomplete in model-selection periods: "
            + ", ".join(str(value.date()) for value in incomplete_tdnet[:5])
        )
    panel = merge_tdnet_features(
        panel,
        tdnet_dataset,
        sessions,
        decision_time=settings.preopen.decision_time,
    )
    panel["tdnet_revision_x_xrank_oc_mean_20"] = (
        panel["tdnet_has_revision"] * panel["xrank_oc_mean_20"]
    )
    panel["tdnet_equity_financing_x_xrank_oc_mean_20"] = (
        panel["tdnet_has_equity_financing"] * panel["xrank_oc_mean_20"]
    )
    panel["tdnet_any_x_market_beta_x_prior_market_return"] = (
        panel["tdnet_any"] * panel["market_beta_x_prior_market_return"]
    )
    interaction_columns = (
        *feature_base.TDNET_RUNUP_INTERACTION_COLUMNS,
        *feature_base.TDNET_MARKET_INTERACTION_COLUMNS,
    )
    panel[list(interaction_columns)] = panel[list(interaction_columns)].astype(
        "float32"
    )
    # TDnet completeness is a feature-block-specific condition.  The helper
    # used by v0.4 gates the whole panel because every v0.4 candidate had the
    # same source policy; broad-family research must retain price-only models
    # when the independent disclosure archive is unavailable.
    panel["eligible"] = panel["price_eligible"]
    panel["training_eligible"] = panel["price_training_eligible"]
    required_features = tuple(
        dict.fromkeys(column for values in feature_blocks().values() for column in values)
    )
    missing_features = sorted(set(required_features) - set(panel.columns))
    if missing_features:
        raise ValueError(f"model-v05 panel lacks features: {missing_features}")
    keep = tuple(
        dict.fromkeys(
            (
                "date",
                "code",
                "name",
                "eligible",
                "training_eligible",
                "price_eligible",
                "price_training_eligible",
                "label",
                "oc_return_pct",
                "outcome_observed",
                "source_complete",
                "universe_source_complete",
                "feature_source_max_date",
                "tdnet_source_complete",
                "tdnet_feature_source_max_timestamp",
                *required_features,
            )
        )
    )
    panel = panel.loc[:, list(keep)].copy()
    numeric_features = [column for column in required_features if column in panel]
    panel[numeric_features] = panel[numeric_features].astype("float32")
    panel = panel.sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    if panel.duplicated(["date", "code"]).any():
        raise AssertionError("model-v05 panel has duplicate date/code rows")
    leaked = panel["feature_source_max_date"].notna() & (
        panel["feature_source_max_date"] >= panel["date"]
    )
    if leaked.any():
        raise AssertionError("model-v05 panel contains a non-prior price source")
    provenance = {
        "jpx_parse_report": jpx_report,
        "tdnet_source_sha256": tdnet_dataset.source_sha256,
        "tdnet_source_files": tdnet_dataset.source_files,
        "tdnet_disclosures": int(len(tdnet_dataset.disclosures)),
        "tdnet_provenance_counts": {
            str(key): int(value)
            for key, value in tdnet_dataset.provenance_by_date.value_counts().items()
        },
        "tdnet_observed_at_min": tdnet_dataset.observed_at_by_date.min().isoformat(),
        "tdnet_observed_at_max": tdnet_dataset.observed_at_by_date.max().isoformat(),
        "tdnet_complete_target_sessions_through_2025_03_31": int(
            tdnet_coverage.reindex(required_tdnet_sessions, fill_value=False).sum()
        ),
        "price_coverage": {
            "sessions": int(len(coverage)),
            "incomplete_sessions": 0,
        },
    }
    return panel, sessions, provenance


def load_or_build_panel(
    panel_path: Path,
    jpx_directory: Path,
    tdnet_directory: Path,
    protocol_path: Path,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, Any]]:
    manifest_path = _panel_manifest_path(panel_path)
    input_hashes = _raw_input_hashes(jpx_directory, tdnet_directory, protocol_path)
    if panel_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != PANEL_SCHEMA_VERSION:
            raise ValueError("model-v05 panel manifest schema changed")
        if manifest.get("inputs") != input_hashes:
            raise ValueError("model-v05 panel inputs changed; remove cache explicitly")
        if sha256_file(panel_path) != manifest.get("panel_file_sha256"):
            raise ValueError("model-v05 panel cache checksum mismatch")
        print("load verified model-v05 panel cache", flush=True)
        panel = read_frame(panel_path)
        sessions = normalize_expected_sessions(manifest["sessions"])
        _validate_cached_panel(panel, sessions, manifest)
        return panel, sessions, manifest["provenance"]

    if panel_path.exists() or manifest_path.exists():
        raise FileExistsError("partial model-v05 panel cache exists")
    panel, sessions, provenance = _build_panel(jpx_directory, tdnet_directory)
    if _raw_input_hashes(jpx_directory, tdnet_directory, protocol_path) != input_hashes:
        raise RuntimeError("model-v05 inputs or implementation changed during panel build")
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    written = write_frame(panel, panel_path)
    manifest = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "inputs": input_hashes,
        "panel_path": str(written),
        "panel_file_sha256": sha256_file(written),
        "panel_semantic_sha256": streaming_content_hash(panel, panel.columns),
        "panel_columns": list(panel.columns),
        "panel_dtypes": {column: str(panel[column].dtype) for column in panel.columns},
        "rows": int(len(panel)),
        "codes": int(panel["code"].nunique()),
        "sessions": [str(value.date()) for value in sessions],
        "calendar_sha256": session_calendar_hash(sessions, through=sessions.max()),
        "runtime": _runtime_fingerprint(),
        "features": {key: list(value) for key, value in feature_blocks().items()},
        "provenance": provenance,
    }
    write_json(manifest, manifest_path)
    return panel, sessions, provenance


def _jpx_source_url(filename: str) -> str:
    year = int(filename[:4])
    if year == 2024:
        return f"https://www.jpx.co.jp/markets/statistics-equities/daily/data/{filename}"
    return (
        "https://www.jpx.co.jp/markets/statistics-equities/daily/"
        f"tvdivq0000001jan-att/{filename}"
    )


def ensure_input_lock(
    lock_path: Path,
    *,
    panel_path: Path,
    jpx_directory: Path,
    tdnet_directory: Path,
    protocol_path: Path,
) -> tuple[dict[str, Any], str]:
    panel_manifest_path = _panel_manifest_path(panel_path)
    panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    inputs = panel_manifest["inputs"]
    if _raw_input_hashes(jpx_directory, tdnet_directory, protocol_path) != inputs:
        raise RuntimeError("current inputs differ from the panel before lock creation")
    jpx_sources = []
    for filename, digest in sorted(inputs["jpx_pdf_sha256"].items()):
        source = jpx_directory / filename
        jpx_sources.append(
            {
                "filename": filename,
                "source_url": _jpx_source_url(filename),
                "bytes": source.stat().st_size,
                "sha256": digest,
            }
        )
    tdnet_sources = []
    for filename, digest in sorted(inputs["tdnet_html_sha256"].items()):
        token = filename.removesuffix(".html")
        metadata_name = filename + ".meta.json"
        tdnet_sources.append(
            {
                "date": str(pd.to_datetime(token, format="%Y%m%d").date()),
                "source_url": TDNET_INDEX_URL.format(date=token),
                "html_filename": filename,
                "html_sha256": digest,
                "metadata_filename": metadata_name,
                "metadata_sha256": inputs["tdnet_metadata_sha256"][metadata_name],
            }
        )
    parse_report = panel_manifest["provenance"]["jpx_parse_report"]
    expected = {
        "schema_version": 1,
        "lock_id": "model_v05_retrospective_inputs_20260722",
        "status": "frozen_before_model_v05_metrics",
        "registered_at_utc": panel_manifest["created_at_utc"],
        "protocol_path": _portable_path(protocol_path),
        "protocol_sha256": inputs["protocol_sha256"],
        "calendar": {
            "start": panel_manifest["sessions"][0],
            "end": panel_manifest["sessions"][-1],
            "sessions": len(panel_manifest["sessions"]),
            "sha256": panel_manifest["calendar_sha256"],
        },
        "jpx": {
            "parser_version": parse_report["parser_version"],
            "rows": int(parse_report["rows"]),
            "codes": int(parse_report["codes"]),
            "no_trade_rows": int(parse_report["no_trade_rows"]),
            "partial_session_rows": int(parse_report["partial_session_rows"]),
            "field_availability": {
                "am_pm_ohlc": True,
                "volume": bool(parse_report["has_volume"]),
                "turnover": bool(parse_report["has_turnover"]),
                "trading_unit": False,
            },
            "sources": jpx_sources,
        },
        "tdnet": {
            "selection_complete_through": str(TDNET_REQUIRED_THROUGH.date()),
            "source_files": len(tdnet_sources),
            "sources": tdnet_sources,
        },
        "panel": {
            "manifest_sha256": sha256_file(panel_manifest_path),
            "file_sha256": panel_manifest["panel_file_sha256"],
            "semantic_sha256": panel_manifest["panel_semantic_sha256"],
            "rows": panel_manifest["rows"],
            "codes": panel_manifest["codes"],
        },
        "implementation_sha256": inputs["implementation_sha256"],
        "runtime": panel_manifest["runtime"],
    }
    if lock_path.exists():
        observed = json.loads(lock_path.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError("model-v05 input lock differs from the verified panel inputs")
    else:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(expected, lock_path)
    digest = sha256_file(lock_path)
    return expected, digest


def _periods(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    output: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for period in pd.period_range(start.to_period("M"), end.to_period("M"), freq="M"):
        output.append(
            (
                max(start, period.start_time.normalize()),
                min(end, period.end_time.normalize()),
                str(period),
            )
        )
    return output


def _desired_slots(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.MultiIndex.from_product(
        [sessions, range(1, TOP_K + 1)], names=["date", "model_rank"]
    ).to_frame(index=False)


def _empty_actual_picks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "model_rank": pd.Series(dtype="int64"),
            "candidate_id": pd.Series(dtype="string"),
            "code": pd.Series(dtype="string"),
            "name": pd.Series(dtype="string"),
            "model_score": pd.Series(dtype="float64"),
            "label": pd.Series(dtype="float64"),
            "oc_return_pct": pd.Series(dtype="float64"),
            "outcome_observed": pd.Series(dtype="boolean"),
            "source_complete": pd.Series(dtype="boolean"),
            "universe_source_complete": pd.Series(dtype="boolean"),
        }
    )


def _rank_top_two(scoring: pd.DataFrame, scores: np.ndarray, candidate_id: str) -> pd.DataFrame:
    ranked = scoring.assign(model_score=np.asarray(scores, dtype=float)).sort_values(
        ["date", "model_score", "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    ranked = ranked.groupby("date", sort=True, as_index=False).head(TOP_K).copy()
    ranked["model_rank"] = ranked.groupby("date", sort=False).cumcount() + 1
    ranked["candidate_id"] = candidate_id
    return ranked


def _metrics(
    picks: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int = 31,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    lower_bounds: dict[str, float] = {}
    displayed = (
        picks["code"].notna()
        if "code" in picks
        else picks["label"].notna()
    )
    rank1_display = displayed[picks["model_rank"].eq(1)]
    rank2_display = displayed[picks["model_rank"].eq(2)]
    by_date = picks.assign(_displayed=displayed).pivot(
        index="date", columns="model_rank", values="_displayed"
    )
    output["display"] = {
        "rank1_rate": float(rank1_display.mean()),
        "rank2_rate": float(rank2_display.mean()),
        "both_slots_rate": float(by_date.reindex(columns=[1, 2], fill_value=False).all(axis=1).mean()),
        "rank1_sessions": int(rank1_display.sum()),
        "rank2_sessions": int(rank2_display.sum()),
        "scheduled_sessions": int(picks["date"].nunique()),
    }
    for top_k in (1, 2):
        key = f"top{top_k}"
        metrics20 = profit_metrics(picks, top_k=top_k, cost_bps=PRIMARY_COST_BPS)
        metrics40 = profit_metrics(picks, top_k=top_k, cost_bps=STRESS_COST_BPS)
        daily = daily_portfolio_returns(
            picks, top_k=top_k, cost_bps=PRIMARY_COST_BPS
        ).set_index("date")["net_return_pct"]
        interval = moving_block_bootstrap(
            daily,
            block_length=5,
            samples=bootstrap_samples,
            confidence=0.90,
            random_state=seed + top_k,
        )
        lower_bounds[key] = interval.one_sided_lower_pct
        output[key] = {
            "net20": metrics20,
            "net40": metrics40,
            "block5_bootstrap": asdict(interval),
        }
    rank2 = picks[picks["model_rank"].eq(2)].copy()
    rank2["model_rank"] = 1
    output["rank2_standalone_net20"] = profit_metrics(
        rank2, top_k=1, cost_bps=PRIMARY_COST_BPS
    )
    output["joint_selection_score"] = float(min(lower_bounds.values()))
    return output


def _candidate_order(item: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = item["metrics"]
    return (
        -metrics["top2"]["net20"]["net_mean_pct_at_cost"],
        -metrics["top2"]["block5_bootstrap"]["one_sided_lower_pct"],
        -metrics["top2"]["net20"]["top5_removed_net_mean_pct"],
        -metrics["top1"]["net20"]["net_mean_pct_at_cost"],
        int(item["feature_count"]),
        str(item["candidate_id"]),
    )


def _evaluation_projection(panel: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return list(
        dict.fromkeys(
            [
                "date",
                "code",
                "name",
                "eligible",
                "training_eligible",
                "price_eligible",
                "price_training_eligible",
                "label",
                "oc_return_pct",
                "outcome_observed",
                "source_complete",
                "universe_source_complete",
                "tdnet_source_complete",
                *columns,
            ]
        )
    )


def _source_mask(panel: pd.DataFrame, feature_block: str, *, training: bool) -> pd.Series:
    base = panel["price_training_eligible" if training else "price_eligible"].astype(
        bool
    )
    if _block_uses_tdnet(feature_block):
        return base & panel["tdnet_source_complete"].eq(True)
    return base


def evaluate_candidate(
    panel: pd.DataFrame,
    candidate: CandidateSpec,
    *,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    sessions: pd.DatetimeIndex,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    columns = feature_blocks()[candidate.feature_block]
    projection = _evaluation_projection(panel, columns)
    scheduled = sessions[(sessions >= evaluation_start) & (sessions <= evaluation_end)]
    pick_parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for score_start, score_end, period_name in _periods(
        evaluation_start, evaluation_end
    ):
        training = panel.loc[
            panel["date"].between(TRAIN_START, score_start - pd.Timedelta(days=1))
            & _source_mask(panel, candidate.feature_block, training=True)
            & panel["label"].notna(),
            projection,
        ].copy()
        train_days = int(training["date"].nunique())
        if train_days < MIN_TRAIN_SESSIONS:
            raise ValueError(
                f"{candidate.candidate_id} has {train_days} training sessions before {score_start.date()}"
            )
        scoring = panel.loc[
            panel["date"].between(score_start, score_end)
            & _source_mask(panel, candidate.feature_block, training=False),
            projection,
        ].copy()
        if not scoring.empty and not training["date"].max() < scoring["date"].min():
            raise AssertionError("training and scoring dates overlap")
        fitted = fit_research_model(candidate.model, training, columns)
        if not scoring.empty:
            pick_parts.append(
                _rank_top_two(
                    scoring,
                    fitted.score(scoring),
                    candidate.candidate_id,
                )
            )
        folds.append(
            {
                "period": period_name,
                "train_start": str(training["date"].min().date()),
                "train_end": str(training["date"].max().date()),
                "train_sessions": train_days,
                "train_rows": int(len(training)),
                "score_start": str(score_start.date()),
                "score_end": str(score_end.date()),
                "scheduled_sessions": int(
                    ((scheduled >= score_start) & (scheduled <= score_end)).sum()
                ),
                "scored_sessions": int(scoring["date"].nunique()),
                "score_rows": int(len(scoring)),
            }
        )
        del fitted, training, scoring
        gc.collect()
    actual = (
        pd.concat(pick_parts, ignore_index=True)
        if pick_parts
        else _empty_actual_picks()
    )
    picks = _desired_slots(scheduled).merge(
        actual,
        on=["date", "model_rank"],
        how="left",
        validate="one_to_one",
        sort=True,
    )
    picks["candidate_id"] = candidate.candidate_id
    metrics = _metrics(picks, bootstrap_samples=bootstrap_samples)
    result = {
        **candidate.to_dict(),
        "status": "completed",
        "feature_count": len(columns),
        "features": list(columns),
        "period": {
            "start": str(evaluation_start.date()),
            "end": str(evaluation_end.date()),
            "scheduled_sessions": int(len(scheduled)),
        },
        "metrics": metrics,
        "folds": folds,
    }
    keep = [
        column
        for column in (
            "candidate_id",
            "date",
            "model_rank",
            "code",
            "name",
            "model_score",
            "label",
            "oc_return_pct",
            "outcome_observed",
            "source_complete",
            "universe_source_complete",
        )
        if column in picks
    ]
    return result, picks[keep].copy()


def evaluate_ensemble(
    panel: pd.DataFrame,
    ensemble: EnsembleSpec,
    *,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    sessions: pd.DatetimeIndex,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    scheduled = sessions[(sessions >= evaluation_start) & (sessions <= evaluation_end)]
    union_columns = tuple(
        dict.fromkeys(
            column
            for member in ensemble.members
            for column in feature_blocks()[member.feature_block]
        )
    )
    projection = _evaluation_projection(panel, union_columns)
    ensemble_uses_tdnet = any(
        _block_uses_tdnet(member.feature_block) for member in ensemble.members
    )
    pick_parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for score_start, score_end, period_name in _periods(
        evaluation_start, evaluation_end
    ):
        common_scoring_mask = panel["price_eligible"].astype(bool)
        if ensemble_uses_tdnet:
            common_scoring_mask &= panel["tdnet_source_complete"].eq(True)
        scoring = panel.loc[
            panel["date"].between(score_start, score_end)
            & common_scoring_mask,
            projection,
        ].copy()
        member_ranks: list[pd.Series] = []
        member_training: list[dict[str, Any]] = []
        for member in ensemble.members:
            columns = feature_blocks()[member.feature_block]
            training = panel.loc[
                panel["date"].between(
                    TRAIN_START, score_start - pd.Timedelta(days=1)
                )
                & _source_mask(panel, member.feature_block, training=True)
                & panel["label"].notna(),
                projection,
            ].copy()
            train_days = int(training["date"].nunique())
            if train_days < MIN_TRAIN_SESSIONS:
                raise ValueError(
                    f"ensemble member {member.candidate_id} has insufficient training sessions"
                )
            member_training.append(
                {
                    "candidate_id": member.candidate_id,
                    "train_start": str(training["date"].min().date()),
                    "train_end": str(training["date"].max().date()),
                    "train_sessions": train_days,
                    "train_rows": int(len(training)),
                }
            )
            if not scoring.empty:
                fitted = fit_research_model(member.model, training, columns)
                score = pd.Series(fitted.score(scoring), index=scoring.index)
                percentile = score.groupby(scoring["date"], sort=False).rank(
                    method="average", pct=True
                )
                member_ranks.append(percentile)
                del fitted
            del training
        if not scoring.empty:
            ensemble_score = pd.concat(member_ranks, axis=1).mean(axis=1)
            if (
                len(ensemble_score) != len(scoring)
                or not np.isfinite(ensemble_score.to_numpy(dtype=float)).all()
            ):
                raise ValueError("ensemble produced invalid scores")
            pick_parts.append(
                _rank_top_two(
                    scoring,
                    ensemble_score.to_numpy(),
                    ensemble.candidate_id,
                )
            )
        folds.append(
            {
                "period": period_name,
                "member_training": member_training,
                "score_start": str(score_start.date()),
                "score_end": str(score_end.date()),
                "scheduled_sessions": int(
                    ((scheduled >= score_start) & (scheduled <= score_end)).sum()
                ),
                "scored_sessions": int(scoring["date"].nunique()),
                "score_rows": int(len(scoring)),
            }
        )
        del scoring, member_ranks
        gc.collect()
    actual = (
        pd.concat(pick_parts, ignore_index=True)
        if pick_parts
        else _empty_actual_picks()
    )
    picks = _desired_slots(scheduled).merge(
        actual,
        on=["date", "model_rank"],
        how="left",
        validate="one_to_one",
        sort=True,
    )
    picks["candidate_id"] = ensemble.candidate_id
    result = {
        **ensemble.to_dict(),
        "status": "completed",
        "feature_count": len(union_columns),
        "features": list(union_columns),
        "period": {
            "start": str(evaluation_start.date()),
            "end": str(evaluation_end.date()),
            "scheduled_sessions": int(len(scheduled)),
        },
        "metrics": _metrics(picks, bootstrap_samples=bootstrap_samples),
        "folds": folds,
    }
    keep = [
        column
        for column in (
            "candidate_id",
            "date",
            "model_rank",
            "code",
            "name",
            "model_score",
            "label",
            "oc_return_pct",
            "outcome_observed",
            "source_complete",
            "universe_source_complete",
        )
        if column in picks
    ]
    return result, picks[keep].copy()


def _evaluate(
    panel: pd.DataFrame,
    candidate: CandidateSpec | EnsembleSpec,
    *,
    stage: str,
    period: Mapping[str, Any],
    sessions: pd.DatetimeIndex,
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidate_id = candidate.candidate_id
    print(f"[{stage}] start {candidate_id}", flush=True)
    started = time.monotonic()
    kwargs = {
        "evaluation_start": pd.Timestamp(period["start"]),
        "evaluation_end": pd.Timestamp(period["end"]),
        "sessions": sessions,
        "bootstrap_samples": int(period["bootstrap_samples"]),
    }
    if isinstance(candidate, EnsembleSpec):
        result, picks = evaluate_ensemble(panel, candidate, **kwargs)
    else:
        result, picks = evaluate_candidate(panel, candidate, **kwargs)
    result["elapsed_seconds"] = float(time.monotonic() - started)
    picks.insert(0, "stage", stage)
    print(
        f"[{stage}] done {candidate_id} "
        f"display={result['metrics']['display']['rank2_rate']:.3f} "
        f"top2={result['metrics']['top2']['net20']['net_mean_pct_at_cost']:+.6f}",
        flush=True,
    )
    return result, picks


def _completed(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in results if item.get("status") == "completed"]


def _selectable(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in _completed(results):
        display = item["metrics"]["display"]
        if (
            display["rank1_rate"] >= MINIMUM_DISPLAY_RATE
            and display["rank2_rate"] >= MINIMUM_DISPLAY_RATE
        ):
            output.append(item)
    return output


def _shortlist_family_screen(
    results: Sequence[dict[str, Any]],
    model_by_spec_id: Mapping[str, ResearchModelSpec],
    *,
    family_groups: Mapping[str, Sequence[str]],
) -> list[ResearchModelSpec]:
    chosen: list[ResearchModelSpec] = []
    selectable = _selectable(results)
    for group_name, model_names in family_groups.items():
        allowed = set(model_names)
        matches = [
            result
            for result in selectable
            if model_by_spec_id[result["model_spec_id"]].name in allowed
        ]
        if not matches:
            raise RuntimeError(
                f"family screen produced no selectable candidate for {group_name}"
            )
        winner = min(matches, key=_candidate_order)
        chosen.append(model_by_spec_id[winner["model_spec_id"]])
    return chosen


def _unique_candidates(values: Iterable[CandidateSpec]) -> list[CandidateSpec]:
    output: dict[str, CandidateSpec] = {}
    for value in values:
        output.setdefault(value.candidate_id, value)
    return list(output.values())


def _candidate_uses_tdnet(candidate: CandidateSpec | EnsembleSpec) -> bool:
    if isinstance(candidate, EnsembleSpec):
        return any(_block_uses_tdnet(member.feature_block) for member in candidate.members)
    return _block_uses_tdnet(candidate.feature_block)


def _source_coverage(
    panel: pd.DataFrame,
    candidate: CandidateSpec | EnsembleSpec,
    sessions: pd.DatetimeIndex,
    period: Mapping[str, Any],
) -> dict[str, Any]:
    scheduled = sessions[
        (sessions >= pd.Timestamp(period["start"]))
        & (sessions <= pd.Timestamp(period["end"]))
    ]
    daily = panel.groupby("date", sort=True).agg(
        price_source_complete=("source_complete", "first"),
        universe_source_complete=("universe_source_complete", "first"),
        tdnet_source_complete=("tdnet_source_complete", "first"),
    )
    available = (
        daily["price_source_complete"].eq(True)
        & daily["universe_source_complete"].eq(True)
    ).reindex(scheduled, fill_value=False)
    requires_tdnet = _candidate_uses_tdnet(candidate)
    if requires_tdnet:
        available &= daily["tdnet_source_complete"].eq(True).reindex(
            scheduled, fill_value=False
        )
    missing = scheduled[~available.to_numpy()]
    return {
        "requires_tdnet": requires_tdnet,
        "scheduled_sessions": int(len(scheduled)),
        "available_sessions": int(available.sum()),
        "coverage_rate": float(available.mean()),
        "complete": bool(available.all()),
        "missing_sessions": [str(value.date()) for value in missing],
    }


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _acquire_run_lock(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise FileExistsError(f"model-v05 run lock already exists: {path}") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _retrospective_gate(result: dict[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    gate = protocol["retrospective_gate"]
    metrics = result["metrics"]
    checks = {
        "top2_net_mean_20bp": metrics["top2"]["net20"]["net_mean_pct_at_cost"]
        >= gate["top2_net_mean_20bp_min"],
        "top2_top5_removed": metrics["top2"]["net20"]["top5_removed_net_mean_pct"]
        >= gate["top2_top5_removed_min"],
        "top2_profit_factor": metrics["top2"]["net20"]["profit_factor"]
        >= gate["top2_profit_factor_min"],
        "top2_positive_month_ratio": metrics["top2"]["net20"]["positive_months"]
        / metrics["top2"]["net20"]["months"]
        >= gate["positive_month_ratio_min"],
        "top2_block5_lcb": metrics["top2"]["block5_bootstrap"]["one_sided_lower_pct"]
        >= gate["top2_5day_block_90pct_lcb_min"],
        "rank1_display_rate": metrics["display"]["rank1_rate"]
        >= gate["minimum_display_rate"],
        "rank2_display_rate": metrics["display"]["rank2_rate"]
        >= gate["minimum_display_rate"],
    }
    numeric_passed = all(checks.values())
    return {
        "checks": checks,
        "numeric_gate_passed": numeric_passed,
        "production_promotion_allowed": False,
        "decision": (
            "retrospective_numeric_gate_only; fresh forward validation still required"
            if numeric_passed
            else "retrospective_gate_failed"
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jpx-dir", required=True)
    parser.add_argument("--tdnet-dir", required=True)
    parser.add_argument(
        "--protocol", default=str(ROOT / "research/model_v05_protocol.json")
    )
    parser.add_argument(
        "--panel-cache", default="/tmp/model_v05_broad_family_panel.pkl"
    )
    parser.add_argument(
        "--input-lock",
        default=str(ROOT / "research/model_v05_input_lock.json"),
    )
    parser.add_argument(
        "--output", default=str(ROOT / "research/model_v05_broad_family_result.json")
    )
    parser.add_argument("--build-panel-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.monotonic()
    protocol_path = Path(args.protocol).resolve()
    protocol, protocol_digest = load_protocol(protocol_path)
    output_path = Path(args.output).resolve()
    panel_path = Path(args.panel_cache).resolve()
    jpx_directory = Path(args.jpx_dir).resolve()
    tdnet_directory = Path(args.tdnet_dir).resolve()
    input_lock_path = Path(args.input_lock).resolve()
    panel, sessions, provenance = load_or_build_panel(
        panel_path,
        jpx_directory,
        tdnet_directory,
        protocol_path,
    )
    input_lock, input_lock_digest = ensure_input_lock(
        input_lock_path,
        panel_path=panel_path,
        jpx_directory=jpx_directory,
        tdnet_directory=tdnet_directory,
        protocol_path=protocol_path,
    )
    runtime_fingerprint = _runtime_fingerprint()
    if input_lock["runtime"] != runtime_fingerprint:
        raise RuntimeError("runtime differs from the frozen model-v05 input lock")
    if args.build_panel_only:
        print(f"panel={panel_path}", flush=True)
        print(f"input_lock={input_lock_path}", flush=True)
        print(f"input_lock_sha256={input_lock_digest}", flush=True)
        return
    stage_names = (
        "family_screen",
        "feature_design",
        "development_confirmation",
        "known_benchmark_reference",
    )
    manifest_path = output_path.with_suffix(".manifest.json")
    run_lock_path = output_path.with_suffix(".run.lock")
    run_receipt_path = output_path.with_suffix(".run.json")
    planned_paths = [
        output_path,
        manifest_path,
        run_receipt_path,
        *[
            output_path.with_name(output_path.stem + f"_{stage}_picks.csv")
            for stage in stage_names
        ],
    ]
    existing = [str(path) for path in planned_paths if path.exists()]
    if existing:
        raise FileExistsError("model-v05 outputs already exist: " + ", ".join(existing))
    run_input_snapshot = _raw_input_hashes(
        jpx_directory, tdnet_directory, protocol_path
    )
    panel_manifest = json.loads(
        _panel_manifest_path(panel_path).read_text(encoding="utf-8")
    )
    if run_input_snapshot != panel_manifest["inputs"]:
        raise RuntimeError("current inputs differ from the frozen panel inputs")
    if run_input_snapshot["protocol_sha256"] != protocol_digest:
        raise RuntimeError("protocol changed between byte-load and experiment start")
    panel_sha256 = sha256_file(panel_path)
    panel_manifest_sha256 = sha256_file(_panel_manifest_path(panel_path))
    run_lock_sha256 = _acquire_run_lock(
        run_lock_path,
        {
            "schema_version": 1,
            "status": "model_v05_metric_run_started",
            "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "protocol_sha256": protocol_digest,
            "panel_sha256": panel_sha256,
            "panel_manifest_sha256": panel_manifest_sha256,
            "input_lock_sha256": input_lock_digest,
            "output_path": _portable_path(output_path),
            "planned_pick_paths": [
                _portable_path(path)
                for path in planned_paths
                if path.suffix == ".csv"
            ],
        },
    )
    models = [ResearchModelSpec.from_dict(item) for item in protocol["model_registry"]]
    model_by_name = {model.name: model for model in models}
    model_by_spec_id = {model.spec_id: model for model in models}
    periods = protocol["periods"]
    stage_results: dict[str, list[dict[str, Any]]] = {}
    stage_picks: dict[str, list[pd.DataFrame]] = {}

    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        family_candidates = [
            CandidateSpec(model, periods["family_screen"]["feature_block"])
            for model in models
        ]
        family_results: list[dict[str, Any]] = []
        family_picks: list[pd.DataFrame] = []
        for candidate in family_candidates:
            result, picks = _evaluate(
                panel,
                candidate,
                stage="family_screen",
                period=periods["family_screen"],
                sessions=sessions,
            )
            family_results.append(result)
            if not picks.empty:
                family_picks.append(picks)
        stage_results["family_screen"] = family_results
        stage_picks["family_screen"] = family_picks
        shortlist = _shortlist_family_screen(
            family_results,
            model_by_spec_id,
            family_groups=protocol["family_groups"],
        )

        design_candidates = [
            CandidateSpec(model, block)
            for model in shortlist
            for block in protocol["feature_blocks"]
        ]
        controls = [
            CandidateSpec(
                model_by_name[control["model"]], control["feature_block"]
            )
            for control in protocol["common_window_controls"].values()
        ]
        design_candidates = _unique_candidates([*design_candidates, *controls])
        design_results: list[dict[str, Any]] = []
        design_picks: list[pd.DataFrame] = []
        for candidate in design_candidates:
            result, picks = _evaluate(
                panel,
                candidate,
                stage="feature_design",
                period=periods["feature_design"],
                sessions=sessions,
            )
            design_results.append(result)
            if not picks.empty:
                design_picks.append(picks)
        stage_results["feature_design"] = design_results
        stage_picks["feature_design"] = design_picks
        finalist_count = int(protocol["stage_policy"]["feature_design_finalists"])
        finalists_by_id = {candidate.candidate_id: candidate for candidate in design_candidates}
        finalist_results = sorted(_selectable(design_results), key=_candidate_order)[
            :finalist_count
        ]
        if len(finalist_results) != finalist_count:
            raise RuntimeError("feature design did not produce three finalists")
        finalists = [finalists_by_id[item["candidate_id"]] for item in finalist_results]
        research_champion = finalists[0]
        price_only_results = [
            result
            for result in _selectable(design_results)
            if not _block_uses_tdnet(
                finalists_by_id[result["candidate_id"]].feature_block
            )
        ]
        if not price_only_results:
            raise RuntimeError("feature design produced no selectable price-only reference")
        price_only_result = min(price_only_results, key=_candidate_order)
        price_only_reference = finalists_by_id[price_only_result["candidate_id"]]
        ensemble = EnsembleSpec(tuple(finalists))

        confirmation_candidates: list[CandidateSpec | EnsembleSpec] = [
            *_unique_candidates(
                [research_champion, price_only_reference, *controls]
            ),
            ensemble,
        ]
        confirmation_results: list[dict[str, Any]] = []
        confirmation_picks: list[pd.DataFrame] = []
        for candidate in confirmation_candidates:
            result, picks = _evaluate(
                panel,
                candidate,
                stage="development_confirmation",
                period=periods["development_confirmation"],
                sessions=sessions,
            )
            confirmation_results.append(result)
            if not picks.empty:
                confirmation_picks.append(picks)
        stage_results["development_confirmation"] = confirmation_results
        stage_picks["development_confirmation"] = confirmation_picks
        confirmation_by_id = {
            result["candidate_id"]: result for result in confirmation_results
        }
        champion_confirmation = confirmation_by_id[research_champion.candidate_id]
        price_only_confirmation = confirmation_by_id[price_only_reference.candidate_id]

        benchmark_period = periods["known_benchmark_reference"]
        champion_benchmark_coverage = _source_coverage(
            panel, research_champion, sessions, benchmark_period
        )
        benchmark_candidates = [price_only_reference]
        if champion_benchmark_coverage["complete"]:
            benchmark_candidates.append(research_champion)
        benchmark_candidates = _unique_candidates(benchmark_candidates)
        benchmark_results: list[dict[str, Any]] = []
        benchmark_pick_frames: list[pd.DataFrame] = []
        for candidate in benchmark_candidates:
            result, picks = _evaluate(
                panel,
                candidate,
                stage="known_benchmark_reference",
                period=benchmark_period,
                sessions=sessions,
            )
            benchmark_results.append(result)
            benchmark_pick_frames.append(picks)
        stage_results["known_benchmark_reference"] = benchmark_results
        stage_picks["known_benchmark_reference"] = benchmark_pick_frames

    if _raw_input_hashes(jpx_directory, tdnet_directory, protocol_path) != run_input_snapshot:
        raise RuntimeError("inputs or implementation changed during model evaluation")
    if sha256_file(input_lock_path) != input_lock_digest:
        raise RuntimeError("input lock changed during model evaluation")
    if sha256_file(panel_path) != panel_sha256:
        raise RuntimeError("panel changed during model evaluation")
    if sha256_file(_panel_manifest_path(panel_path)) != panel_manifest_sha256:
        raise RuntimeError("panel manifest changed during model evaluation")
    if any(path.exists() for path in planned_paths):
        raise FileExistsError("a model-v05 output appeared during model evaluation")
    pick_paths: dict[str, str] = {}
    pick_hashes: dict[str, str] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for stage, frames in stage_picks.items():
        if not frames:
            continue
        path = output_path.with_name(output_path.stem + f"_{stage}_picks.csv")
        written = write_frame(pd.concat(frames, ignore_index=True), path)
        pick_paths[stage] = _portable_path(written)
        pick_hashes[stage] = sha256_file(written)

    gate = _retrospective_gate(champion_confirmation, protocol)
    champion_benchmark = next(
        (
            result
            for result in stage_results["known_benchmark_reference"]
            if result["candidate_id"] == research_champion.candidate_id
        ),
        None,
    )
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": _portable_path(protocol_path),
        "protocol_sha256": protocol_digest,
        "status": (
            "retrospective_numeric_gate_passed_no_production_promotion"
            if gate["numeric_gate_passed"]
            else "retrospective_no_demonstrated_edge"
        ),
        "objective": protocol["objective"],
        "stages": stage_results,
        "family_screen_shortlist": [model.canonical_dict() for model in shortlist],
        "feature_design_finalists": [candidate.to_dict() for candidate in finalists],
        "confirmation_ensemble": ensemble.to_dict(),
        "selected_winner": research_champion.to_dict(),
        "selected_winner_stage": "feature_design",
        "confirmation_did_not_reselect": True,
        "selected_winner_confirmation": champion_confirmation,
        "price_only_reference": price_only_reference.to_dict(),
        "price_only_reference_confirmation": price_only_confirmation,
        "known_benchmark_is_true_holdout": False,
        "known_benchmark_affects_selection_or_gate": False,
        "known_benchmark_champion_source_coverage": champion_benchmark_coverage,
        "known_benchmark_champion_result": champion_benchmark,
        "retrospective_confirmation_gate": gate,
        "production_model_changed": False,
        "production_promotion": "forbidden_until_fresh_forward_validation",
        "prospective_final": protocol["periods"]["prospective_final"],
        "pick_paths": pick_paths,
        "pick_sha256": pick_hashes,
        "run_receipt_path": _portable_path(run_receipt_path),
        "run_receipt_sha256": run_lock_sha256,
        "data": {
            "panel_path": str(panel_path),
            "panel_sha256": panel_sha256,
            "panel_manifest_sha256": panel_manifest_sha256,
            "input_lock_path": _portable_path(input_lock_path),
            "input_lock_sha256": input_lock_digest,
            "input_lock": input_lock,
            "panel_rows": int(len(panel)),
            "panel_codes": int(panel["code"].nunique()),
            "sessions": int(len(sessions)),
            "calendar_sha256": session_calendar_hash(sessions, through=sessions.max()),
            "provenance": provenance,
        },
        "implementation_sha256": run_input_snapshot["implementation_sha256"],
        "runtime": {
            **runtime_fingerprint,
            "elapsed_seconds": float(time.monotonic() - started),
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "thread_limit": 1,
        },
    }
    write_json(payload, output_path)
    result_sha = sha256_file(output_path)
    result_manifest = {
        "schema_version": 1,
        "result_path": _portable_path(output_path),
        "result_sha256": result_sha,
        "protocol_sha256": payload["protocol_sha256"],
        "panel_sha256": payload["data"]["panel_sha256"],
        "input_lock_sha256": input_lock_digest,
        "pick_sha256": pick_hashes,
        "run_receipt_sha256": run_lock_sha256,
        "implementation_sha256": payload["implementation_sha256"],
    }
    write_json(result_manifest, manifest_path)
    os.replace(run_lock_path, run_receipt_path)
    print(f"winner={research_champion.candidate_id}", flush=True)
    print(f"status={payload['status']}", flush=True)
    print(f"result={output_path}", flush=True)
    print(f"result_sha256={result_sha}", flush=True)


if __name__ == "__main__":
    main()
