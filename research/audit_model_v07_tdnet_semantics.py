#!/usr/bin/env python3
"""Reproduce and content-bind the full-corpus TDnet v0.7 semantic audit.

The frozen corpus lives in ``research/.cache`` and is intentionally not part
of version control.  This auditor records the SHA-256 and byte size of every
HTML index and metadata sidecar so the result can be reproduced when that
exact cache is restored.  Historical cache provenance is permitted because
this is a retrospective title-semantic audit; point-in-time safety here means
that no disclosure published after the 08:58:59 decision cutoff is assigned
to that session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import tse_session_ranker.research_candidates as candidates  # noqa: E402
from tse_session_ranker.data.common import (  # noqa: E402
    normalize_expected_sessions,
    session_calendar_hash,
)
from tse_session_ranker.data.tdnet import (  # noqa: E402
    collect_tdnet_dataset,
)
from tse_session_ranker.io import write_json  # noqa: E402


AUDIT_ID = "model_v07_tdnet_semantics_full_corpus_v1"
DECISION_TIME = "08:58:59"
SHUFFLE_SEED = 781
DEFAULT_TDNET_CACHE = ROOT / "research/.cache/model_v05_tdnet"
DEFAULT_PANEL_MANIFEST = ROOT / "research/model_v05_panel_manifest.json"
DEFAULT_OUTPUT = ROOT / "research/model_v07_tdnet_semantics_audit.json"
IMPLEMENTATION_PATHS = (
    ROOT / "src/tse_session_ranker/research_candidates.py",
    ROOT / "src/tse_session_ranker/data/tdnet.py",
    ROOT / "src/tse_session_ranker/data/common.py",
)
FROZEN_EXPECTATIONS = {
    "source_files": 461,
    "documents": 97_006,
    "sessions": 386,
    "bundles": 66_702,
    "candidate_columns": 63,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value


class AuditRecorder:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, detail: Any = None) -> None:
        self.items.append(
            {
                "name": name,
                "passed": bool(passed),
                "detail": _json_safe(detail),
            }
        )

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [item for item in self.items if not item["passed"]]

    def summary(self) -> dict[str, Any]:
        return {
            "total": len(self.items),
            "passed": len(self.items) - len(self.failed),
            "failed": len(self.failed),
            "items": self.items,
        }


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash a path-independent, ordered cache inventory."""

    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["inventory_name"])):
        for value in (
            record["inventory_name"],
            str(record["bytes"]),
            record["sha256"],
        ):
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _file_record(
    path: Path, *, inventory_name: str, kind: str
) -> dict[str, Any]:
    return {
        "kind": kind,
        "inventory_name": inventory_name,
        "path": _portable_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def cache_inventory(tdnet_cache: str | Path) -> dict[str, Any]:
    root = Path(tdnet_cache).resolve()
    html_paths = sorted(root.glob("*.html"))
    if not html_paths:
        raise FileNotFoundError(f"no TDnet HTML files found under {root}")
    pairs: list[dict[str, Any]] = []
    flat: list[dict[str, Any]] = []
    expected_metadata: set[Path] = set()
    for html in html_paths:
        metadata = html.with_suffix(html.suffix + ".meta.json")
        if not metadata.is_file():
            raise FileNotFoundError(f"TDnet metadata sidecar is missing: {metadata}")
        expected_metadata.add(metadata)
        html_record = _file_record(
            html, inventory_name=f"html:{html.name}", kind="html"
        )
        metadata_record = _file_record(
            metadata,
            inventory_name=f"metadata:{metadata.name}",
            kind="metadata",
        )
        flat.extend((html_record, metadata_record))
        pairs.append(
            {
                "index_date": html.stem,
                "html": html_record,
                "metadata": metadata_record,
            }
        )
    orphaned = sorted(set(root.glob("*.html.meta.json")) - expected_metadata)
    if orphaned:
        raise ValueError(
            "TDnet cache has metadata without HTML: "
            + ", ".join(path.name for path in orphaned[:5])
        )
    temporary = root == Path("/tmp") or Path("/tmp") in root.parents
    try:
        relative = root.relative_to(ROOT)
    except ValueError:
        relative = None
    gitignored_cache = relative is not None and str(relative).startswith(
        "research/.cache"
    )
    if temporary:
        location_kind = "temporary_directory"
        constraint = (
            "The exact input exists only under /tmp and must be restored from "
            "the recorded per-file hashes before re-execution."
        )
    elif gitignored_cache:
        location_kind = "gitignored_repository_cache"
        constraint = (
            "The TDnet cache is excluded by .gitignore and is not carried by "
            "the PR. Re-execution requires restoring all recorded files."
        )
    else:
        location_kind = "external_or_versioned_directory"
        constraint = "Re-execution requires every recorded file and sidecar."
    return {
        "root": _portable_path(root),
        "currently_available": root.is_dir(),
        "location_kind": location_kind,
        "temporary_only": temporary,
        "version_control_expected": not gitignored_cache and not temporary,
        "reexecution_constraint": constraint,
        "html_files": len(html_paths),
        "metadata_files": len(expected_metadata),
        "total_files": len(flat),
        "total_bytes": sum(int(item["bytes"]) for item in flat),
        "aggregate_sha256": _aggregate_records(flat),
        "pairs": pairs,
    }


def stable_frame_sha256(frame: pd.DataFrame) -> str:
    """Return a dtype-aware semantic hash for a deterministically sorted frame."""

    digest = hashlib.sha256()
    schema = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "rows": len(frame),
    }
    digest.update(
        json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    digest.update(b"\0")
    hashes = pd.util.hash_pandas_object(
        frame.reset_index(drop=True), index=False, categorize=True
    ).to_numpy(dtype="uint64")
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {
        str(float(key)): int(value)
        for key, value in series.value_counts().sort_index().items()
    }


def _violation_detail(
    disclosures: pd.DataFrame, mask: pd.Series, *, limit: int = 5
) -> dict[str, Any]:
    positions = mask[mask].index[:limit]
    samples = []
    for position in positions:
        row = disclosures.loc[position]
        samples.append(
            {
                "index_date": str(pd.Timestamp(row["index_date"]).date()),
                "published_at": row["published_at"].isoformat(),
                "code": str(row["code"]),
                "title": str(row["title"]),
            }
        )
    return {"violation_count": int(mask.sum()), "samples": samples}


def semantic_invariant_report(
    disclosures: pd.DataFrame,
    legacy: pd.DataFrame,
    semantic: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Compute independent full-title intersections that must remain empty."""

    contains = candidates._contains
    economic = list(candidates._V07_ECONOMIC_FAMILY_NAMES)
    classified_buyback = semantic[
        ["v07_fresh_buyback", "v07_followup_buyback"]
    ].max(axis=1).gt(0)
    masks: dict[str, pd.Series] = {
        "fresh_followup_buyback_overlap": semantic["v07_fresh_buyback"].gt(0)
        & semantic["v07_followup_buyback"].gt(0),
        "fresh_followup_equity_overlap": semantic["v07_fresh_equity"].gt(0)
        & semantic["v07_followup_equity"].gt(0),
        "fresh_followup_share_cancellation_overlap": semantic[
            "v07_fresh_share_cancellation"
        ].gt(0)
        & semantic["v07_followup_share_cancellation"].gt(0),
        "fresh_followup_ma_overlap": semantic["v07_fresh_ma"].gt(0)
        & semantic["v07_followup_ma"].gt(0),
        "correction_has_fresh_economic_family": legacy["correction"].gt(0)
        & semantic[economic].max(axis=1).gt(0),
        "correction_has_followup_family": legacy["correction"].gt(0)
        & semantic[list(candidates._V07_FOLLOWUP_FAMILY_NAMES)]
        .max(axis=1)
        .gt(0),
        "ma_acquisition_is_takeover_defense": semantic[
            "v07_ma_acquisition"
        ].gt(0)
        & contains(
            disclosures["title"],
            r"買収防衛|大規模買付.*対応|大量取得行為.*対応",
        ),
        "ma_acquisition_divestiture_overlap": semantic[
            "v07_ma_acquisition"
        ].gt(0)
        & semantic["v07_ma_divestiture"].gt(0),
        "ma_reorganization_other_subtype_overlap": semantic[
            "v07_ma_reorganization"
        ].gt(0)
        & (
            semantic["v07_ma_acquisition"].gt(0)
            | semantic["v07_ma_divestiture"].gt(0)
        ),
        "ma_internal_without_reorganization": semantic[
            "v07_ma_internal_reorganization"
        ].gt(0)
        & ~semantic["v07_ma_reorganization"].gt(0),
        "ma_divestiture_is_asset_disposal": semantic[
            "v07_ma_divestiture"
        ].gt(0)
        & contains(
            disclosures["title"],
            (
                r"固定資産|販売用不動産|不動産|信託受益権|"
                r"準共有持分"
            ),
        ),
        "ma_divestiture_is_policy_security_sale": semantic[
            "v07_ma_divestiture"
        ].gt(0)
        & contains(
            disclosures["title"],
            r"政策保有株式|投資有価証券|保有株式",
        )
        & ~contains(
            disclosures["title"],
            r"子会社|関連会社|関係会社|持分法|特定子会社|事業",
        ),
        "received_and_shareholder_dividend_overlap": semantic[
            "v07_received_dividend"
        ].gt(0)
        & semantic["v07_shareholder_dividend"].gt(0),
        "subsidiary_and_shareholder_dividend_overlap": semantic[
            "v07_subsidiary_dividend"
        ].gt(0)
        & semantic["v07_shareholder_dividend"].gt(0),
        "intercompany_without_received_dividend": semantic[
            "v07_intercompany_dividend"
        ].gt(0)
        & ~semantic["v07_received_dividend"].gt(0),
        "legacy_equity_status_is_fresh_equity": legacy[
            "equity_financing_status"
        ].gt(0)
        & semantic["v07_fresh_equity"].gt(0),
        "nonissuer_buyback_applicant_classified": classified_buyback
        & contains(
            disclosures["title"],
            r"(?:自己|自社)株(?:式)?.{0,30}(?:取得|買付).{0,30}応募",
        ),
        "nonissuer_child_buyback_classified": classified_buyback
        & contains(
            disclosures["title"],
            r"(?:当社)?(?:連結)?(?:子会社|関連会社).{0,60}"
            r"(?:による|における).{0,30}(?:自己|自社)株(?:式)?.{0,20}"
            r"(?:取得|買付)",
        ),
        "clear_buyback_followup_is_fresh": semantic[
            "v07_fresh_buyback"
        ].gt(0)
        & contains(
            disclosures["title"],
            r"事後調整|調整取引|補足説明|Q.?A|Q&amp;A|Ｑ.?Ａ|"
            r"調査委員会|第三者委員会|調査結果|再発防止|"
            r"分配可能額を超えた|(?:取得|買付)(?:の)?.{0,30}"
            r"(?:状況|結果|実績|終了|完了|中止|再開|延長|"
            r"一部変更|方法追加|総数変更)",
        ),
    }
    return {
        name: _violation_detail(disclosures, mask)
        for name, mask in masks.items()
    }


def synthetic_semantics_probe() -> dict[str, Any]:
    titles = [
        (
            "自己株式の取得状況及び通期業績予想の修正に関する"
            "お知らせ"
        ),
        "（訂正）通期業績予想の上方修正に関するお知らせ",
        "連結子会社からの配当金受領に関するお知らせ",
        "完全子会社間の吸収合併に関するお知らせ",
    ]
    disclosures = pd.DataFrame(
        {
            "published_at": [
                pd.Timestamp("2026-07-20 16:00:00+09:00")
                + pd.Timedelta(minutes=position)
                for position in range(len(titles))
            ],
            "code": [str(7001 + position) for position in range(len(titles))],
            "title": titles,
        }
    )
    features = candidates.build_clean_tdnet_candidate_features(
        disclosures, pd.DatetimeIndex(["2026-07-21"])
    ).set_index("code")
    checks = {
        "mixed_followup_buyback": bool(
            features.loc["7001", "tdnet_v07_has_followup_buyback"] == 1.0
        ),
        "mixed_forecast_remains_fresh": bool(
            features.loc["7001", "tdnet_v07_has_forecast_revision"] == 1.0
            and features.loc["7001", "tdnet_v07_economic_family_count"] == 1.0
        ),
        "correction_not_economic": bool(
            features.loc["7002", "tdnet_v07_economic_family_count"] == 0.0
        ),
        "received_is_not_shareholder_dividend": bool(
            features.loc["7003", "tdnet_v07_has_received_dividend"] == 1.0
            and features.loc["7003", "tdnet_v07_has_shareholder_dividend"]
            == 0.0
        ),
        "internal_reorganization_is_reorganization": bool(
            features.loc["7004", "tdnet_v07_has_ma_reorganization"] == 1.0
            and features.loc[
                "7004", "tdnet_v07_has_ma_internal_reorganization"
            ]
            == 1.0
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def empty_and_completeness_probe() -> dict[str, Any]:
    sessions = pd.DatetimeIndex(["2026-07-21"])
    empty = pd.DataFrame(
        {
            "published_at": pd.Series(dtype="datetime64[ns, Asia/Tokyo]"),
            "code": pd.Series(dtype="object"),
            "title": pd.Series(dtype="object"),
        }
    )
    empty_features = candidates.build_clean_tdnet_candidate_features(
        empty, sessions
    )
    expected_columns = [
        "date",
        "code",
        *candidates.TDNET_CANDIDATE_COLUMNS,
        "tdnet_clean_feature_source_max_timestamp",
    ]
    disclosures = pd.DataFrame(
        {
            "published_at": [pd.Timestamp("2026-07-20 16:00:00+09:00")],
            "code": ["7101"],
            "title": ["2026年3月期 決算短信"],
        }
    )
    panel = pd.DataFrame(
        {
            "probe": [
                "complete_event",
                "complete_no_event",
                "incomplete_event",
                "nullable_event",
            ],
            "date": [pd.Timestamp("2026-07-21")] * 4,
            "code": ["7101", "7199", "7101", "7101"],
            "tdnet_source_complete": pd.Series(
                [True, True, False, pd.NA], dtype="boolean"
            ),
        }
    )
    attached = candidates.attach_clean_tdnet_candidate_features(
        panel, disclosures, sessions
    ).set_index("probe")
    columns = list(candidates.TDNET_CANDIDATE_COLUMNS)
    checks = {
        "empty_rows_zero": empty_features.empty,
        "empty_schema_exact": empty_features.columns.tolist() == expected_columns,
        "empty_candidate_dtypes_float32": all(
            str(empty_features[column].dtype) == "float32" for column in columns
        ),
        "empty_source_timestamp_timezone": str(
            empty_features["tdnet_clean_feature_source_max_timestamp"].dtype
        )
        == "datetime64[ns, Asia/Tokyo]",
        "complete_event_observed": bool(
            attached.loc["complete_event", "tdnet_v07_observed_any"] == 1.0
        ),
        "complete_no_event_all_zero": bool(
            attached.loc["complete_no_event", columns].eq(0.0).all()
        ),
        "incomplete_event_all_missing": bool(
            attached.loc["incomplete_event", columns].isna().all()
        ),
        "nullable_event_all_missing": bool(
            attached.loc["nullable_event", columns].isna().all()
        ),
        "attached_candidate_dtypes_float32": all(
            str(attached[column].dtype) == "float32" for column in columns
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "incomplete_source_timestamp_retained_for_provenance": bool(
            pd.notna(
                attached.loc[
                    "incomplete_event",
                    "tdnet_clean_feature_source_max_timestamp",
                ]
            )
        ),
    }


def _bundle_geometry(
    features: pd.DataFrame, recorder: AuditRecorder
) -> dict[str, Any]:
    candidate_columns = list(candidates.TDNET_CANDIDATE_COLUMNS)
    expected_columns = [
        "date",
        "code",
        *candidate_columns,
        "tdnet_clean_feature_source_max_timestamp",
    ]
    candidate_dtypes = {
        column: str(features[column].dtype) for column in candidate_columns
    }
    recorder.check(
        "candidate_schema_exact",
        features.columns.tolist() == expected_columns,
        {"columns": len(candidate_columns)},
    )
    recorder.check(
        "candidate_columns_float32",
        all(dtype == "float32" for dtype in candidate_dtypes.values()),
        pd.Series(candidate_dtypes).value_counts().to_dict(),
    )
    recorder.check(
        "candidate_values_complete",
        not features[candidate_columns].isna().any(axis=None),
        {"missing_cells": int(features[candidate_columns].isna().sum().sum())},
    )
    economic = features["tdnet_v07_economic_family_count"].astype(float)
    followup = features["tdnet_v07_followup_family_count"].astype(float)
    geometry = {
        "fresh_any_matches_positive_count": bool(
            features["tdnet_v07_fresh_classified_economic_any"]
            .eq(economic.gt(0).astype(float))
            .all()
        ),
        "single_matches_count_one": bool(
            features["tdnet_v07_single_economic_family"]
            .eq(economic.eq(1).astype(float))
            .all()
        ),
        "economic_log1p_exact": bool(
            np.allclose(
                features["tdnet_v07_economic_family_count_log1p"],
                np.log1p(economic),
                rtol=0.0,
                atol=1e-6,
            )
        ),
        "followup_log1p_exact": bool(
            np.allclose(
                features["tdnet_v07_followup_family_count_log1p"],
                np.log1p(followup),
                rtol=0.0,
                atol=1e-6,
            )
        ),
        "economic_counts_integral": bool(
            np.equal(economic, np.floor(economic)).all()
        ),
        "followup_counts_integral": bool(
            np.equal(followup, np.floor(followup)).all()
        ),
    }
    for name, passed in geometry.items():
        recorder.check(f"bundle_geometry:{name}", passed)
    sums = {
        column: int(features[column].sum())
        for column in candidates.V07_EVENT_SEMANTIC_COLUMNS
        if column.startswith("tdnet_v07_has_")
        or column in {
            "tdnet_v07_observed_any",
            "tdnet_v07_fresh_classified_economic_any",
        }
    }
    return {
        "geometry": geometry,
        "candidate_dtypes": candidate_dtypes,
        "economic_family_count_distribution": _value_counts(economic),
        "followup_family_count_distribution": _value_counts(followup),
        "semantic_indicator_sums": sums,
    }


def _pit_report(
    features: pd.DataFrame, recorder: AuditRecorder
) -> dict[str, Any]:
    decision = pd.Timestamp(DECISION_TIME).time()
    cutoffs = pd.Series(
        [
            pd.Timestamp.combine(pd.Timestamp(date).date(), decision).tz_localize(
                "Asia/Tokyo"
            )
            for date in features["date"]
        ],
        index=features.index,
    )
    sources = features["tdnet_clean_feature_source_max_timestamp"]
    violations = sources.notna() & sources.gt(cutoffs)
    recorder.check(
        "pit_source_not_after_decision_cutoff",
        not violations.any(),
        {"violation_count": int(violations.sum())},
    )
    age_seconds = (cutoffs - sources).dt.total_seconds()
    return {
        "decision_time": DECISION_TIME,
        "violation_count": int(violations.sum()),
        "minimum_source_age_seconds": float(age_seconds.min()),
        "latest_source_timestamp": sources.max().isoformat(),
    }


def _legacy_perturbation_report(
    disclosures: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    baseline: pd.DataFrame,
    recorder: AuditRecorder,
) -> dict[str, Any]:
    legacy_columns = [
        "date",
        "code",
        *candidates.T0_CLEAN_EVENT_COLUMNS,
        *candidates.T1_EVENT_STRUCTURE_COLUMNS,
        "tdnet_clean_feature_source_max_timestamp",
    ]
    original = candidates._v07_semantic_title_flags

    def inverted(
        title: pd.Series, legacy_flags: pd.DataFrame
    ) -> pd.DataFrame:
        return 1.0 - original(title, legacy_flags)

    candidates._v07_semantic_title_flags = inverted
    try:
        perturbed = candidates.build_clean_tdnet_candidate_features(
            disclosures, sessions, decision_time=DECISION_TIME
        )
    finally:
        candidates._v07_semantic_title_flags = original
    mismatch = None
    try:
        assert_frame_equal(
            baseline[legacy_columns], perturbed[legacy_columns], check_exact=True
        )
        equal = True
    except AssertionError as exc:
        equal = False
        mismatch = str(exc)[:2_000]
    baseline_hash = stable_frame_sha256(baseline[legacy_columns])
    perturbed_hash = stable_frame_sha256(perturbed[legacy_columns])
    recorder.check(
        "legacy_columns_bit_exact_under_v07_perturbation",
        equal and baseline_hash == perturbed_hash,
        {"mismatch": mismatch},
    )
    return {
        "legacy_candidate_columns": len(candidates.T0_CLEAN_EVENT_COLUMNS)
        + len(candidates.T1_EVENT_STRUCTURE_COLUMNS),
        "baseline_sha256": baseline_hash,
        "v07_inverted_sha256": perturbed_hash,
        "bit_exact": equal and baseline_hash == perturbed_hash,
    }


def _implementation_binding() -> dict[str, Any]:
    records = [
        _file_record(
            path,
            inventory_name=f"implementation:{_portable_path(path)}",
            kind="implementation",
        )
        for path in IMPLEMENTATION_PATHS
    ]
    return {
        "aggregate_sha256": _aggregate_records(records),
        "files": records,
    }


def audit(
    *,
    tdnet_cache: str | Path = DEFAULT_TDNET_CACHE,
    panel_manifest_path: str | Path = DEFAULT_PANEL_MANIFEST,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    recorder = AuditRecorder()
    tdnet_cache = Path(tdnet_cache).resolve()
    panel_manifest_path = Path(panel_manifest_path).resolve()
    output_path = Path(output_path).resolve()
    inventory = cache_inventory(tdnet_cache)
    panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    sessions = normalize_expected_sessions(panel_manifest["sessions"])
    dataset = collect_tdnet_dataset(
        tdnet_cache, allow_historical_provenance=True
    )
    disclosures = dataset.disclosures

    computed_html = {
        pair["html"]["path"].rsplit("/", 1)[-1]: pair["html"]["sha256"]
        for pair in inventory["pairs"]
    }
    computed_metadata = {
        pair["metadata"]["path"].rsplit("/", 1)[-1]: pair["metadata"][
            "sha256"
        ]
        for pair in inventory["pairs"]
    }
    expected_html = panel_manifest["inputs"]["tdnet_html_sha256"]
    expected_metadata = panel_manifest["inputs"]["tdnet_metadata_sha256"]
    recorder.check(
        "input_html_inventory_matches_panel_manifest",
        computed_html == expected_html,
        {"computed": len(computed_html), "expected": len(expected_html)},
    )
    recorder.check(
        "input_metadata_inventory_matches_panel_manifest",
        computed_metadata == expected_metadata,
        {
            "computed": len(computed_metadata),
            "expected": len(expected_metadata),
        },
    )
    recorder.check(
        "tdnet_logical_source_hash_matches_panel_manifest",
        dataset.source_sha256
        == panel_manifest["provenance"]["tdnet_source_sha256"],
        {"source_sha256": dataset.source_sha256},
    )
    recorder.check(
        "session_calendar_hash_matches_panel_manifest",
        session_calendar_hash(sessions) == panel_manifest["calendar_sha256"],
        {"calendar_sha256": session_calendar_hash(sessions)},
    )

    for name, actual in {
        "source_files": dataset.source_files,
        "documents": len(disclosures),
        "sessions": len(sessions),
    }.items():
        recorder.check(
            f"frozen_count:{name}",
            actual == FROZEN_EXPECTATIONS[name],
            {"actual": actual, "expected": FROZEN_EXPECTATIONS[name]},
        )

    legacy = candidates._clean_title_flags(disclosures["title"])
    semantic = candidates._v07_semantic_title_flags(
        disclosures["title"], legacy
    )
    invariants = semantic_invariant_report(disclosures, legacy, semantic)
    for name, detail in invariants.items():
        recorder.check(
            f"semantic_invariant:{name}",
            detail["violation_count"] == 0,
            detail,
        )
    raw_semantic_counts = {
        column: int(semantic[column].sum())
        for column in semantic.columns
        if not column.startswith("v07_economic_")
    }
    raw_economic_counts = {
        column: int(semantic[column].sum())
        for column in candidates._V07_ECONOMIC_FAMILY_NAMES
    }
    document_hash_frame = pd.concat(
        [
            disclosures[["index_date", "published_at", "code", "title"]],
            semantic,
        ],
        axis=1,
    )

    features = candidates.build_clean_tdnet_candidate_features(
        disclosures, sessions, decision_time=DECISION_TIME
    )
    recorder.check(
        "frozen_count:bundles",
        len(features) == FROZEN_EXPECTATIONS["bundles"],
        {"actual": len(features), "expected": FROZEN_EXPECTATIONS["bundles"]},
    )
    recorder.check(
        "frozen_count:candidate_columns",
        len(candidates.TDNET_CANDIDATE_COLUMNS)
        == FROZEN_EXPECTATIONS["candidate_columns"],
        {
            "actual": len(candidates.TDNET_CANDIDATE_COLUMNS),
            "expected": FROZEN_EXPECTATIONS["candidate_columns"],
        },
    )
    bundle_report = _bundle_geometry(features, recorder)
    pit = _pit_report(features, recorder)

    shuffled = candidates.build_clean_tdnet_candidate_features(
        disclosures.sample(frac=1.0, random_state=SHUFFLE_SEED),
        sessions,
        decision_time=DECISION_TIME,
    )
    shuffle_mismatch = None
    try:
        assert_frame_equal(features, shuffled, check_exact=True)
        shuffle_exact = True
    except AssertionError as exc:
        shuffle_exact = False
        shuffle_mismatch = str(exc)[:2_000]
    full_hash = stable_frame_sha256(features)
    shuffled_hash = stable_frame_sha256(shuffled)
    recorder.check(
        "input_shuffle_is_bit_exact",
        shuffle_exact and full_hash == shuffled_hash,
        {"seed": SHUFFLE_SEED, "mismatch": shuffle_mismatch},
    )

    legacy_report = _legacy_perturbation_report(
        disclosures, sessions, features, recorder
    )
    completeness = empty_and_completeness_probe()
    recorder.check(
        "empty_and_completeness_probe", completeness["passed"], completeness
    )
    synthetic = synthetic_semantics_probe()
    recorder.check("synthetic_semantics_probe", synthetic["passed"], synthetic)

    implementation = _implementation_binding()
    audit_script = _file_record(
        Path(__file__).resolve(),
        inventory_name=f"audit:{_portable_path(Path(__file__))}",
        kind="audit_script",
    )
    manifest_path = output_path.with_suffix(".manifest.json")
    checks = recorder.summary()
    source_dates = pd.DatetimeIndex(dataset.complete_dates)
    result = {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "scope": {
            "authority": (
                "Retrospective TDnet title-semantic and pre-open assignment "
                "audit; it does not select a trading model."
            ),
            "decision_time": DECISION_TIME,
            "allow_historical_provenance": True,
            "shuffle_seed": SHUFFLE_SEED,
            "wall_clock_timestamp_omitted_for_determinism": True,
        },
        "bindings": {
            "result_path": _portable_path(output_path),
            "result_manifest_path": _portable_path(manifest_path),
            "audit_script": audit_script,
            "implementation": implementation,
            "panel_manifest": {
                "path": _portable_path(panel_manifest_path),
                "bytes": panel_manifest_path.stat().st_size,
                "sha256": sha256_file(panel_manifest_path),
            },
            "tdnet_logical_source_sha256": dataset.source_sha256,
            "cache_inventory_sha256": inventory["aggregate_sha256"],
        },
        "input_inventory": inventory,
        "corpus": {
            "documents": len(disclosures),
            "document_date_min": str(disclosures["index_date"].min().date()),
            "document_date_max": str(disclosures["index_date"].max().date()),
            "source_dates": len(source_dates),
            "source_date_min": str(source_dates.min().date()),
            "source_date_max": str(source_dates.max().date()),
            "target_sessions": len(sessions),
            "target_session_min": str(sessions.min().date()),
            "target_session_max": str(sessions.max().date()),
            "bundles": len(features),
            "bundle_date_min": str(features["date"].min().date()),
            "bundle_date_max": str(features["date"].max().date()),
        },
        "candidate_schema": {
            "candidate_columns": list(candidates.TDNET_CANDIDATE_COLUMNS),
            "candidate_column_count": len(candidates.TDNET_CANDIDATE_COLUMNS),
            "legacy_columns": [
                *candidates.T0_CLEAN_EVENT_COLUMNS,
                *candidates.T1_EVENT_STRUCTURE_COLUMNS,
            ],
            "legacy_column_count": len(candidates.T0_CLEAN_EVENT_COLUMNS)
            + len(candidates.T1_EVENT_STRUCTURE_COLUMNS),
            "v07_columns": list(candidates.V07_EVENT_SEMANTIC_COLUMNS),
            "v07_column_count": len(candidates.V07_EVENT_SEMANTIC_COLUMNS),
            "candidate_dtypes": bundle_report["candidate_dtypes"],
            "source_timestamp_dtype": str(
                features["tdnet_clean_feature_source_max_timestamp"].dtype
            ),
        },
        "distributions": {
            "document_semantic_counts": raw_semantic_counts,
            "document_economic_family_counts": raw_economic_counts,
            "bundle": {
                key: value
                for key, value in bundle_report.items()
                if key != "candidate_dtypes"
            },
        },
        "invariants": invariants,
        "pit": pit,
        "reproducibility": {
            "input_shuffle": {
                "seed": SHUFFLE_SEED,
                "bit_exact": shuffle_exact and full_hash == shuffled_hash,
                "baseline_sha256": full_hash,
                "shuffled_sha256": shuffled_hash,
            },
            "legacy_bit_exact": legacy_report,
            "empty_and_completeness": completeness,
            "synthetic_semantics": synthetic,
        },
        "semantic_hashes": {
            "document_flags_sha256": stable_frame_sha256(document_hash_frame),
            "bundle_all_sha256": full_hash,
            "bundle_legacy_sha256": legacy_report["baseline_sha256"],
            "bundle_v07_sha256": stable_frame_sha256(
                features[
                    ["date", "code", *candidates.V07_EVENT_SEMANTIC_COLUMNS]
                ]
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "limitations": [
            inventory["reexecution_constraint"],
            (
                "Historical cache provenance proves title timestamps and "
                "retrospective cutoff assignment, not contemporaneous capture "
                "of every historical page before its original session."
            ),
            (
                "The semantic classifier uses disclosure titles only; PDF "
                "body quantities and economic magnitudes are outside scope."
            ),
            (
                "Incomplete rows retain the latest observed source timestamp "
                "as provenance while every candidate value fails closed to NaN."
            ),
        ],
        "checks": checks,
        "verdict": {
            "computational_integrity": "pass" if checks["failed"] == 0 else "fail",
            "blocking_findings": [
                item["name"] for item in checks["items"] if not item["passed"]
            ],
        },
    }
    return _json_safe(result)


def _manifest_payload(report: Mapping[str, Any], output_path: Path) -> dict[str, Any]:
    bindings = report["bindings"]
    return {
        "schema_version": 1,
        "result_path": _portable_path(output_path),
        "result_sha256": sha256_file(output_path),
        "audit_script_path": bindings["audit_script"]["path"],
        "audit_script_sha256": bindings["audit_script"]["sha256"],
        "implementation_aggregate_sha256": bindings["implementation"][
            "aggregate_sha256"
        ],
        "panel_manifest_path": bindings["panel_manifest"]["path"],
        "panel_manifest_sha256": bindings["panel_manifest"]["sha256"],
        "tdnet_logical_source_sha256": bindings["tdnet_logical_source_sha256"],
        "cache_inventory_sha256": bindings["cache_inventory_sha256"],
    }


def _first_mismatch(left: Any, right: Any, path: str = "root") -> str | None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return f"{path} key mismatch: {sorted(set(left) ^ set(right))[:10]}"
        for key in left:
            mismatch = _first_mismatch(left[key], right[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path} length mismatch: {len(left)} != {len(right)}"
        for position, (a, b) in enumerate(zip(left, right)):
            mismatch = _first_mismatch(a, b, f"{path}[{position}]")
            if mismatch is not None:
                return mismatch
        return None
    if left != right:
        return f"{path} mismatch: {left!r} != {right!r}"
    return None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tdnet-cache", default=str(DEFAULT_TDNET_CACHE))
    parser.add_argument(
        "--panel-manifest", default=str(DEFAULT_PANEL_MANIFEST)
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--manifest",
        default=None,
        help="Companion manifest path; defaults to OUTPUT with .manifest.json.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Recompute and compare with the stored result and manifest.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing result when generating a new audit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = Path(args.output).resolve()
    manifest = (
        Path(args.manifest).resolve()
        if args.manifest is not None
        else output.with_suffix(".manifest.json")
    )
    if args.verify:
        if not output.is_file() or not manifest.is_file():
            print("stored audit result or manifest is missing", file=sys.stderr)
            return 1
        stored = json.loads(output.read_text(encoding="utf-8"))
        stored_manifest = json.loads(manifest.read_text(encoding="utf-8"))
        if stored_manifest.get("result_sha256") != sha256_file(output):
            print("stored manifest does not bind the result", file=sys.stderr)
            return 1
        recomputed = audit(
            tdnet_cache=args.tdnet_cache,
            panel_manifest_path=args.panel_manifest,
            output_path=output,
        )
        mismatch = _first_mismatch(stored, recomputed)
        if mismatch is not None:
            print(mismatch, file=sys.stderr)
            return 1
        expected_manifest = _manifest_payload(recomputed, output)
        mismatch = _first_mismatch(stored_manifest, expected_manifest)
        if mismatch is not None:
            print(mismatch, file=sys.stderr)
            return 1
        print(
            f"verified {AUDIT_ID}: {recomputed['corpus']['documents']} documents, "
            f"{recomputed['corpus']['bundles']} bundles"
        )
        return 0
    if (output.exists() or manifest.exists()) and not args.overwrite:
        print(
            "audit output already exists; pass --overwrite or use --verify",
            file=sys.stderr,
        )
        return 1
    report = audit(
        tdnet_cache=args.tdnet_cache,
        panel_manifest_path=args.panel_manifest,
        output_path=output,
    )
    write_json(report, output)
    write_json(_manifest_payload(report, output), manifest)
    print(
        f"wrote {output}: {report['checks']['passed']}/"
        f"{report['checks']['total']} checks passed"
    )
    return 0 if report["checks"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
