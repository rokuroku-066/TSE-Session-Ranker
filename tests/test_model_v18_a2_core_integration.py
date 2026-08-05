from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping

import pandas as pd
import pytest

from research import model_v18_shoulder_state_runner as runner


ANCHOR_SESSION = pd.Timestamp("2026-07-31")
NEW_SOURCE_SESSION = pd.Timestamp("2026-08-05")
TARGET_SESSION = pd.Timestamp("2026-08-06")
FIRST_COUNTED_SESSION = TARGET_SESSION
ANCHOR_FILE = "202607.pdf"
NEW_FILE = "stq_20260805.pdf"
ANCHOR_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    f"{ANCHOR_FILE}"
)
NEW_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/daily/synthetic-att/"
    f"{NEW_FILE}"
)


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def _external_path(root: Path, key: str) -> Path:
    return root.joinpath(*key.split("/"))


def _tree_identity(root: Path) -> tuple[tuple[Any, ...], ...]:
    rows: list[tuple[Any, ...]] = []
    for path in sorted([root, *root.rglob("*")], key=lambda item: item.as_posix()):
        metadata = path.lstat()
        rows.append(
            (
                "." if path == root else path.relative_to(root).as_posix(),
                stat.S_IFMT(metadata.st_mode),
                stat.S_IMODE(metadata.st_mode),
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_nlink,
                metadata.st_size,
                path.read_bytes() if stat.S_ISREG(metadata.st_mode) else None,
            )
        )
    return tuple(rows)


def _full31_rows(dates: pd.DatetimeIndex, source_file: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_format = (
        "jpx_stock_quotations_auction_regular_way_domestic_ordinary"
    )
    for code_index, code in enumerate(("1001", "1002", "1003")):
        for session_index, session in enumerate(dates):
            open_price = 1_000.0 + 25.0 * code_index + 0.5 * session_index
            return_fraction = (
                0.003 if (session_index + code_index) % 2 else -0.003
            )
            close = open_price * (1.0 + return_fraction)
            high = max(open_price, close) * 1.005
            low = min(open_price, close) * 0.995
            volume = 100_000.0 + 1_000.0 * code_index + 100.0 * session_index
            vwap = (open_price + close) / 2.0
            rows.append(
                {
                    "date": session,
                    "code": code,
                    "name": f"name-{code}",
                    "raw_name": f"name-{code}",
                    "trading_unit": 100,
                    "final_special_quote": None,
                    "net_change": close - open_price,
                    "vwap": vwap,
                    "volume": volume,
                    "turnover": volume * vwap,
                    "volume_unit": "shares",
                    "turnover_unit": "yen",
                    "source_volume_unit": "shares",
                    "source_turnover_unit": "yen",
                    "source_file": source_file.replace(".pdf", ".txt"),
                    "source_line": code_index + 1,
                    "source_format": source_format,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "am_open": open_price,
                    "am_high": high,
                    "am_low": low,
                    "am_close": vwap,
                    "pm_open": vwap,
                    "pm_high": high,
                    "pm_low": low,
                    "pm_close": close,
                    "traded": True,
                    "partial_session": False,
                }
            )
    return runner._coerce_jsonl_frame(
        pd.DataFrame(rows),
        runner.PARSED_PRICE_COLUMNS,
        label=f"synthetic parsed {source_file}",
    )


def _install_registry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    calendar: pd.DatetimeIndex,
) -> None:
    files_by_latest = {
        ANCHOR_SESSION: [ANCHOR_FILE],
        NEW_SOURCE_SESSION: [ANCHOR_FILE, NEW_FILE],
        pd.Timestamp("2026-08-06"): [
            ANCHOR_FILE,
            NEW_FILE,
            "stq_20260806.pdf",
        ],
        pd.Timestamp("2026-08-07"): [
            ANCHOR_FILE,
            NEW_FILE,
            "stq_20260806.pdf",
            "stq_20260807.pdf",
        ],
    }

    def expected_files(latest: Any) -> tuple[list[str], dict[str, str]]:
        key = pd.Timestamp(latest).normalize()
        names = list(files_by_latest[key])
        return names, {
            name: ("price_warmup" if name == ANCHOR_FILE else "daily")
            for name in names
        }

    def latest_required(target: Any, *args: Any, **kwargs: Any) -> pd.Timestamp:
        value = pd.Timestamp(target).normalize()
        positions = [index for index, item in enumerate(calendar) if item == value]
        if len(positions) != 1 or positions[0] == 0:
            raise runner.V18Error("synthetic target has no D-1")
        return pd.Timestamp(calendar[positions[0] - 1]).normalize()

    monkeypatch.setattr(runner, "load_registered_calendar", lambda *args: calendar)
    monkeypatch.setattr(runner, "_expected_predictor_files", expected_files)
    monkeypatch.setattr(
        runner, "_latest_required_predictor_source_session", latest_required
    )
    monkeypatch.setattr(
        runner,
        "_latest_registered_source_before_month",
        lambda month: ANCHOR_SESSION,
    )
    monkeypatch.setattr(runner, "_historical_predictor_metadata", lambda: {})


def _install_parser(
    monkeypatch: pytest.MonkeyPatch,
    frames: Mapping[str, pd.DataFrame],
) -> None:
    monkeypatch.setattr(
        runner,
        "_pdftotext_cache_contract",
        lambda: {
            "file_sha256": "1" * 64,
            "elf_closure_sha256": "2" * 64,
            "argv_environment_contract_sha256": "3" * 64,
        },
    )

    def collect(paths: list[Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
        if len(paths) != 1:
            raise AssertionError("per-shard parser must receive exactly one raw file")
        name = Path(paths[0]).name
        frame = frames[name].copy(deep=True)
        text = f"synthetic pdftotext bytes for {name}\n".encode()
        digest = hashlib.sha256(text).hexdigest()
        return frame, {
            "inputs": [
                {
                    "path": name.replace(".pdf", ".txt"),
                    "rejected_rows": 0,
                    "text_byte_count": len(text),
                    "text_sha256": digest,
                    "sha256": digest,
                }
            ]
        }

    monkeypatch.setattr(runner, "_collect_jpx_registered", collect)


def _snapshot_binding(
    snapshot: Mapping[str, Any], manifest_key: str, manifest_payload: bytes
) -> dict[str, Any]:
    return {
        "model_price_snapshot_target_month": snapshot["target_month"],
        "model_price_snapshot_latest_source_session": snapshot[
            "latest_source_session"
        ],
        "model_price_snapshot_object_key": snapshot["data_object_key"],
        "model_price_snapshot_byte_count": int(snapshot["data_byte_count"]),
        "model_price_snapshot_file_sha256": snapshot["data_sha256"],
        "model_price_snapshot_semantic_sha256": snapshot[
            "model_price_semantic_sha256"
        ],
        "model_price_snapshot_manifest_object_key": manifest_key,
        "model_price_snapshot_manifest_byte_count": len(manifest_payload),
        "model_price_snapshot_manifest_file_sha256": hashlib.sha256(
            manifest_payload
        ).hexdigest(),
        "model_price_snapshot_manifest_sha256": snapshot[
            "snapshot_manifest_sha256"
        ],
    }


def _score_pair(
    *,
    target: pd.Timestamp,
    source_sha: str,
    fold_sha: str,
    verified: datetime,
    generated: datetime | None = None,
) -> pd.DataFrame:
    generated_at = datetime.now(runner.TOKYO) if generated is None else generated
    frame = pd.DataFrame(
        {
                "session_date": [str(target.date())] * 2,
                "source_rank": [1, 2],
                "code": ["1001", "1002"],
                "name": ["name-1001", "name-1002"],
                "model_score": [0.75, 0.25],
                "feature_source_max_date": [
                    str(NEW_SOURCE_SESSION.date())
                ]
                * 2,
                "score_generated_at": [generated_at.isoformat()] * 2,
                "runtime_lock_sha256": [runner.RUNTIME_LOCK_SHA256] * 2,
                "runtime_lock_verified_at": [verified.isoformat()] * 2,
                "source_manifest_sha256": [source_sha] * 2,
                "c00_fold_manifest_sha256": [fold_sha] * 2,
        }
    )
    # prepare_day deliberately reloads the immutable shard with this dtype.
    frame["code"] = frame["code"].astype("string")
    return runner.validate_score_rows(frame)


def _prepare_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Any]:
    calendar = pd.bdate_range("2026-07-31", "2026-08-10")
    _install_registry(monkeypatch, calendar=calendar)
    anchor_prices = _full31_rows(
        pd.bdate_range("2026-07-01", end=ANCHOR_SESSION), ANCHOR_FILE
    )
    new_prices = _full31_rows(pd.DatetimeIndex([NEW_SOURCE_SESSION]), NEW_FILE)
    _install_parser(
        monkeypatch,
        {ANCHOR_FILE: anchor_prices, NEW_FILE: new_prices},
    )

    predictor_raw = _private_directory(tmp_path / "predictor-raw")
    predictor_derived = _private_directory(tmp_path / "predictor-derived")
    outcome_raw = _private_directory(tmp_path / "outcome-raw")
    checkpoint_core = _private_directory(tmp_path / "checkpoint-core")
    local = _private_directory(tmp_path / "local")
    directories = {
        "SOURCE_MANIFEST_DIR": _private_directory(local / "source"),
        "MONTH_SOURCE_MANIFEST_DIR": _private_directory(local / "month-source"),
        "FOLD_MANIFEST_DIR": _private_directory(local / "fold"),
        "FOLD_MODEL_DIR": _private_directory(local / "bundle"),
        "STATE_MANIFEST_DIR": _private_directory(local / "state"),
        "SCORE_SESSION_DIR": _private_directory(local / "score-sessions"),
        "OUTCOME_MANIFEST_DIR": _private_directory(local / "outcome-manifests"),
    }
    for name, path in directories.items():
        monkeypatch.setattr(runner, name, path)
    monkeypatch.setattr(runner, "DECISION_LEDGER", local / "decisions.jsonl")
    monkeypatch.setattr(runner, "OUTCOME_LEDGER", local / "outcomes.jsonl")
    monkeypatch.setattr(
        runner, "COMPLETED_MONTH_LEDGER", local / "completed-months.jsonl"
    )
    monkeypatch.setattr(runner, "SCORE_OUTPUT", local / "scores.csv")

    now = datetime.now(runner.TOKYO)
    verified = now - timedelta(minutes=10)
    receipt = now - timedelta(minutes=5)
    activation_observed = now - timedelta(minutes=9)
    monkeypatch.setattr(runner, "_STRICT_RUNTIME_ACTIVE", False)

    anchor_pdf = tmp_path / ANCHOR_FILE
    anchor_pdf.write_bytes(b"%PDF-1.7 synthetic anchor bytes\n")
    anchor_key = runner._predictor_object_key(ANCHOR_FILE, "price_warmup")
    anchor_count, anchor_sha = runner._seal_predictor_object(
        anchor_pdf,
        predictor_raw_store_root=predictor_raw,
        object_key=anchor_key,
    )
    anchor_raw_record = {
        "object_key": anchor_key,
        "file": ANCHOR_FILE,
        "url": ANCHOR_URL,
        "byte_count": anchor_count,
        "sha256": anchor_sha,
    }
    anchor_shard, retained_anchor_prices, anchor_binding = (
        runner.ensure_predictor_parsed_shard(
            anchor_raw_record,
            predictor_raw_store_root=predictor_raw,
            predictor_derived_store_root=predictor_derived,
            chronology_class="anchor",
            raw_received_at=None,
            runtime_lock_verified_at=verified,
            created_at=verified + timedelta(minutes=1),
            sealed_at=verified + timedelta(minutes=2),
        )
    )
    snapshot, retained_model_prices, snapshot_manifest_key, snapshot_payload = (
        runner.materialize_model_price_snapshot(
            runner._coerce_model_price_frame(
                retained_anchor_prices, label="synthetic anchor compact prefix"
            ),
            predictor_derived_store_root=predictor_derived,
            target_month="2026-08",
            latest_source_session=ANCHOR_SESSION,
            raw_records=[anchor_raw_record],
            shard_bindings=[anchor_binding],
            previous_snapshot_manifest_sha256=None,
            runtime_lock_verified_at=verified,
        )
    )
    assert retained_model_prices.equals(
        runner._coerce_model_price_frame(
            anchor_prices, label="expected synthetic anchor compact prefix"
        )
    )
    binding = _snapshot_binding(snapshot, snapshot_manifest_key, snapshot_payload)
    anchor_manifest = {
        "raw_sources": [anchor_raw_record],
        "parsed_shards": [anchor_binding],
    }
    anchor_manifest_key = (
        f"{runner.CACHE_ANCHOR_OBJECT_PREFIX}synthetic-anchor.manifest.json"
    )
    anchor_manifest_payload = runner._json_file_bytes(anchor_manifest)
    runner._write_external_bytes_once(
        anchor_manifest_payload,
        store_root=predictor_derived,
        object_key=anchor_manifest_key,
        prefix=runner.CACHE_ANCHOR_OBJECT_PREFIX,
        label="synthetic anchor manifest",
    )
    anchor_summary = {
        "latest_source_session": str(ANCHOR_SESSION.date()),
        "snapshot_manifest_object_key": anchor_manifest_key,
        "snapshot_manifest_file_sha256": hashlib.sha256(
            anchor_manifest_payload
        ).hexdigest(),
        **binding,
    }
    monkeypatch.setattr(
        runner,
        "validate_predictor_cache_anchor_summary",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        runner,
        "validate_predictor_cache_anchor",
        lambda *args, **kwargs: (
            dict(anchor_manifest),
            dict(anchor_summary),
            retained_anchor_prices,
        ),
    )
    monkeypatch.setattr(runner, "_STRICT_RUNTIME_ACTIVE", True)
    monkeypatch.setattr(runner, "_STRICT_RUNTIME_VERIFIED_AT", verified)

    payload_sha = "a" * 64
    receipt_sha = "b" * 64
    context = {
        "activation_payload_sha256": payload_sha,
        "activation_receipt_sha256": receipt_sha,
        "activation_receipt_commit_sha": "c" * 40,
        "activation_receipt_commit_observed_at": activation_observed.isoformat(),
        "first_counted_session": str(FIRST_COUNTED_SESSION.date()),
    }
    monkeypatch.setattr(
        runner, "validate_activation_context", lambda value: dict(value)
    )
    monkeypatch.setattr(
        runner,
        "validate_activation_payload",
        lambda *args, **kwargs: (
            {"predictor_cache_anchor": dict(anchor_summary)},
            payload_sha,
        ),
    )

    fold = {
        "fold_manifest_sha256": "d" * 64,
        "fold_model_bundle_file_sha256": "e" * 64,
    }
    bundle = {"synthetic": True}
    monkeypatch.setattr(
        runner,
        "_load_or_create_month_fold",
        lambda *args, **kwargs: (dict(fold), dict(bundle)),
    )
    monkeypatch.setattr(runner, "pair_history_from_ledgers", lambda *args: [])
    monkeypatch.setattr(
        runner,
        "derive_month_state",
        lambda history, month: {"state": "insufficient_history"},
    )

    def build_state(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "target_month": "2026-08",
            "created_at": kwargs["created_at"],
            "activation_payload_sha256": payload_sha,
            "activation_receipt_sha256": receipt_sha,
            "c00_fold_manifest_sha256": fold["fold_manifest_sha256"],
            "fold_model_bundle_file_sha256": fold[
                "fold_model_bundle_file_sha256"
            ],
            "state_manifest_sha256": "f" * 64,
        }

    monkeypatch.setattr(runner, "build_state_manifest", build_state)
    monkeypatch.setattr(
        runner, "validate_state_manifest", lambda value: dict(value)
    )

    def freeze(
        panel: pd.DataFrame, target: Any, **kwargs: Any
    ) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
        replay_verified = (
            verified
            if kwargs.get("runtime_lock_verified_at") is None
            else runner._timestamp(
                kwargs["runtime_lock_verified_at"], "synthetic score verified"
            )
        )
        replay_generated = (
            None
            if kwargs.get("score_generated_at") is None
            else runner._timestamp(
                kwargs["score_generated_at"], "synthetic score generated"
            )
        )
        pair = _score_pair(
            target=pd.Timestamp(target),
            source_sha=kwargs["source_manifest_sha256"],
            fold_sha=fold["fold_manifest_sha256"],
            verified=replay_verified,
            generated=replay_generated,
        )
        return pair, dict(fold), dict(bundle)

    monkeypatch.setattr(runner, "freeze_c00_top2", freeze)
    monkeypatch.setattr(
        runner,
        "prepare_checkpoint",
        lambda **kwargs: {"checkpoint_batch_id": "synthetic-checkpoint-batch"},
    )

    new_pdf = tmp_path / NEW_FILE
    new_pdf.write_bytes(b"%PDF-1.7 synthetic D-1 bytes\n")
    panel_calls: list[str] = []
    real_panel_builder = runner.build_forward_c00_panel

    def counted_panel(prices: pd.DataFrame, target: Any) -> pd.DataFrame:
        panel_calls.append(str(pd.Timestamp(target).date()))
        return real_panel_builder(prices, target)

    monkeypatch.setattr(runner, "build_forward_c00_panel", counted_panel)

    def prepare() -> dict[str, Any]:
        return runner.prepare_day(
            session_date=TARGET_SESSION,
            new_predictor_pdfs=[new_pdf],
            new_predictor_file_names=[NEW_FILE],
            new_predictor_urls=[NEW_URL],
            new_predictor_received_at=[receipt],
            predictor_raw_store_root=predictor_raw,
            predictor_derived_store_root=predictor_derived,
            outcome_raw_store_root=outcome_raw,
            checkpoint_core_store_root=checkpoint_core,
            activation_context=context,
        )

    return {
        "prepare": prepare,
        "panel_calls": panel_calls,
        "predictor_raw": predictor_raw,
        "predictor_derived": predictor_derived,
        "outcome_raw": outcome_raw,
        "checkpoint_core": checkpoint_core,
        "local": local,
        "directories": directories,
        "anchor_raw_record": anchor_raw_record,
        "anchor_binding": anchor_binding,
        "anchor_shard": anchor_shard,
        "snapshot": snapshot,
        "snapshot_binding": binding,
        "context": context,
        "new_pdf": new_pdf,
        "receipt": receipt,
    }


def test_prepare_day_materializes_the_core_a2_chain_and_builds_one_panel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _prepare_fixture(monkeypatch, tmp_path)

    result = case["prepare"]()

    assert result["target_session"] == str(TARGET_SESSION.date())
    assert result["checkpoint_batch_id"] == "synthetic-checkpoint-batch"
    assert case["panel_calls"] == [str(TARGET_SESSION.date())]
    source_path = case["directories"]["SOURCE_MANIFEST_DIR"] / (
        f"{TARGET_SESSION.date()}.json"
    )
    month_path = case["directories"]["MONTH_SOURCE_MANIFEST_DIR"] / "2026-08.json"
    source = runner.read_json(source_path)
    month_source = runner.read_json(month_path)
    assert source_path.read_bytes() == runner._json_file_bytes(source)
    assert month_path.read_bytes() == runner._json_file_bytes(month_source)
    assert source["source_manifest_sha256"] == result["source_manifest_sha256"]
    assert month_source["month_source_manifest_sha256"] == result[
        "month_source_manifest_sha256"
    ]
    assert source["model_price_snapshot_manifest_sha256"] == case["snapshot"][
        "snapshot_manifest_sha256"
    ]
    assert source["source_files"] == [ANCHOR_FILE, NEW_FILE]
    assert len(source["parsed_shards"]) == 2
    forward_binding = source["parsed_shards"][-1]
    assert forward_binding["shard_manifest_object_key"].startswith(
        runner.PREDICTOR_SHARD_OBJECT_PREFIX
    )
    assert _external_path(
        case["predictor_derived"], forward_binding["shard_object_key"]
    ).read_bytes() == runner.canonical_frame_jsonl_bytes(
        _full31_rows(pd.DatetimeIndex([NEW_SOURCE_SESSION]), NEW_FILE),
        runner.PARSED_PRICE_COLUMNS,
        label="expected forward shard bytes",
    )
    cache_manifest_path = _external_path(
        case["predictor_derived"], source["g0_panel_cache_manifest_object_key"]
    )
    cache_manifest = runner.read_json(cache_manifest_path)
    assert cache_manifest_path.read_bytes() == runner._json_file_bytes(cache_manifest)
    assert cache_manifest["cache_manifest_sha256"] == source[
        "g0_panel_cache_manifest_sha256"
    ]
    assert source["previous_counted_target_session"] is None
    assert source["previous_counted_source_manifest_sha256"] is None


def test_prepare_day_completed_source_retry_is_exact_and_rebuilds_one_panel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _prepare_fixture(monkeypatch, tmp_path)
    first = case["prepare"]()
    before = (
        _tree_identity(case["predictor_raw"]),
        _tree_identity(case["predictor_derived"]),
        _tree_identity(case["outcome_raw"]),
    )
    case["panel_calls"].clear()

    second = case["prepare"]()

    assert second == first
    assert case["panel_calls"] == [str(TARGET_SESSION.date())]
    assert (
        _tree_identity(case["predictor_raw"]),
        _tree_identity(case["predictor_derived"]),
        _tree_identity(case["outcome_raw"]),
    ) == before


def test_immediate_predecessor_hash_fork_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _prepare_fixture(monkeypatch, tmp_path)
    case["prepare"]()
    source_path = case["directories"]["SOURCE_MANIFEST_DIR"] / (
        f"{TARGET_SESSION.date()}.json"
    )
    source = runner.read_json(source_path)
    source["source_manifest_sha256"] = "0" * 64
    source_path.write_bytes(runner._json_file_bytes(source))

    with pytest.raises(runner.V18Error, match="immediate predecessor manifest changed"):
        runner._source_manifest_predecessor_binding(
            "2026-08-07",
            first_counted_session_value=FIRST_COUNTED_SESSION,
            require_decision_binding=False,
        )


def test_catch_up_cannot_skip_a_missing_immediate_counted_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _prepare_fixture(monkeypatch, tmp_path)
    case["prepare"]()

    with pytest.raises(runner.V18Error, match="immediate predecessor manifest is missing"):
        runner._source_manifest_predecessor_binding(
            "2026-08-10",
            first_counted_session_value=FIRST_COUNTED_SESSION,
            require_decision_binding=False,
        )
