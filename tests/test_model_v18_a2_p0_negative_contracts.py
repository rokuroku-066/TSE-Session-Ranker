from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

from research import model_v18_shoulder_state_runner as runner


DAILY_FILE = "stq_20260805.pdf"
DAILY_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/daily/synthetic-att/"
    f"{DAILY_FILE}"
)
MONTHLY_FILE = "202607.pdf"
MONTHLY_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    f"{MONTHLY_FILE}"
)


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def _tree_bytes(root: Path) -> tuple[tuple[str, bytes | None], ...]:
    return tuple(
        (
            "." if path == root else path.relative_to(root).as_posix(),
            path.read_bytes() if path.is_file() and not path.is_symlink() else None,
        )
        for path in sorted([root, *root.rglob("*")], key=lambda item: item.as_posix())
    )


def _parsed_prices(dates: pd.DatetimeIndex, source_file: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_format = (
        "jpx_stock_quotations_auction_regular_way_domestic_ordinary"
    )
    for code_index, code in enumerate(("1001", "1002", "1003")):
        for session_index, session in enumerate(dates):
            open_price = 1_000.0 + 10.0 * code_index + session_index
            close = open_price * 1.002
            high = close * 1.003
            low = open_price * 0.997
            volume = 100_000.0 + 1_000.0 * code_index
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
        label="P0 parsed-price fixture",
    )


def _valid_shard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    file_name: str,
    source_url: str,
    kind: str,
    dates: pd.DatetimeIndex,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    raw_root = _private_directory(tmp_path / "raw")
    derived_root = _private_directory(tmp_path / "derived")
    source = tmp_path / file_name
    source.write_bytes(b"%PDF-1.7 synthetic P0 bytes\n")
    key = runner._predictor_object_key(file_name, kind)
    count, digest = runner._seal_predictor_object(
        source,
        predictor_raw_store_root=raw_root,
        object_key=key,
    )
    raw_record = {
        "object_key": key,
        "file": file_name,
        "url": source_url,
        "byte_count": count,
        "sha256": digest,
    }
    prices = _parsed_prices(dates, file_name)
    text = b"synthetic locked pdftotext bytes\n"
    text_sha = hashlib.sha256(text).hexdigest()
    monkeypatch.setattr(
        runner,
        "_pdftotext_cache_contract",
        lambda: {
            "file_sha256": "1" * 64,
            "elf_closure_sha256": "2" * 64,
            "argv_environment_contract_sha256": "3" * 64,
        },
    )
    monkeypatch.setattr(
        runner,
        "_collect_jpx_registered",
        lambda paths: (
            prices.copy(deep=True),
            {
                "inputs": [
                    {
                        "path": file_name.replace(".pdf", ".txt"),
                        "rejected_rows": 0,
                        "text_byte_count": len(text),
                        "text_sha256": text_sha,
                        "sha256": text_sha,
                    }
                ]
            },
        ),
    )
    now = datetime.now(runner.TOKYO)
    manifest, _, _ = runner.ensure_predictor_parsed_shard(
        raw_record,
        predictor_raw_store_root=raw_root,
        predictor_derived_store_root=derived_root,
        chronology_class="anchor",
        raw_received_at=None,
        runtime_lock_verified_at=now - timedelta(minutes=3),
        created_at=now - timedelta(minutes=2),
        sealed_at=now - timedelta(minutes=1),
    )
    return manifest, raw_record, raw_root, derived_root


def test_daily_parsed_shard_filename_date_swap_is_rejected_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, raw_record, raw_root, derived_root = _valid_shard(
        monkeypatch,
        tmp_path,
        file_name=DAILY_FILE,
        source_url=DAILY_URL,
        kind="daily",
        dates=pd.DatetimeIndex(["2026-08-05"]),
    )
    changed = {**manifest, "min_date": "2026-08-04", "max_date": "2026-08-04"}
    changed["manifest_sha256"] = runner.canonical_json_sha256(
        changed, exclude_fields={"manifest_sha256"}
    )
    before = (_tree_bytes(raw_root), _tree_bytes(derived_root))

    with pytest.raises(
        runner.V18Error, match="daily parsed shard dates differ from its filename"
    ):
        runner._validate_parsed_shard_manifest(
            changed,
            raw_record=raw_record,
            predictor_derived_store_root=derived_root,
            expected_chronology_class="anchor",
            decode_data=False,
        )

    assert (_tree_bytes(raw_root), _tree_bytes(derived_root)) == before


def test_monthly_parsed_shard_outside_month_date_is_rejected_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, raw_record, raw_root, derived_root = _valid_shard(
        monkeypatch,
        tmp_path,
        file_name=MONTHLY_FILE,
        source_url=MONTHLY_URL,
        kind="price_warmup",
        dates=pd.DatetimeIndex(["2026-07-01", "2026-07-31"]),
    )
    changed = {**manifest, "max_date": "2026-08-01"}
    changed["manifest_sha256"] = runner.canonical_json_sha256(
        changed, exclude_fields={"manifest_sha256"}
    )
    before = (_tree_bytes(raw_root), _tree_bytes(derived_root))

    with pytest.raises(
        runner.V18Error, match="monthly parsed shard dates differ from its filename"
    ):
        runner._validate_parsed_shard_manifest(
            changed,
            raw_record=raw_record,
            predictor_derived_store_root=derived_root,
            expected_chronology_class="anchor",
            decode_data=False,
        )

    assert (_tree_bytes(raw_root), _tree_bytes(derived_root)) == before


@pytest.mark.parametrize("source_role", ["anchor", "new"])
@pytest.mark.parametrize(
    "bad_url",
    [
        f"{DAILY_URL}?download=1",
        DAILY_URL.replace("www.jpx.co.jp/", "www.jpx.co.jp:443/"),
        f"https://www.jpx.co.jp/markets/statistics-equities/daily/{DAILY_FILE}",
    ],
)
def test_daily_url_query_port_and_path_reject_before_raw_seal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_role: str,
    bad_url: str,
) -> None:
    raw_root = _private_directory(tmp_path / "raw")
    derived_root = _private_directory(tmp_path / "derived")
    source = tmp_path / DAILY_FILE
    source.write_bytes(b"%PDF-1.7 URL preflight bytes\n")
    calendar = pd.DatetimeIndex(["2026-08-05", "2026-08-06"])
    monkeypatch.setattr(runner, "load_registered_calendar", lambda *args: calendar)
    monkeypatch.setattr(
        runner,
        "_latest_required_predictor_source_session",
        lambda *args, **kwargs: pd.Timestamp("2026-08-05"),
    )
    monkeypatch.setattr(
        runner,
        "_expected_predictor_files",
        lambda latest: ([DAILY_FILE], {DAILY_FILE: "daily"}),
    )
    monkeypatch.setattr(runner, "_historical_predictor_metadata", lambda: {})
    seal_calls: list[str] = []

    def forbidden_seal(*args: Any, **kwargs: Any) -> tuple[int, str]:
        seal_calls.append("seal")
        pytest.fail("invalid daily URL reached irreversible raw sealing")

    monkeypatch.setattr(runner, "_seal_predictor_object", forbidden_seal)
    before = (_tree_bytes(raw_root), _tree_bytes(derived_root))

    with pytest.raises(
        runner.V18Error, match="official JPX HTTPS URL|canonical daily JPX URL label"
    ):
        if source_role == "anchor":
            runner.build_predictor_cache_anchor(
                [source],
                [DAILY_FILE],
                [bad_url],
                through_session="2026-08-05",
                predictor_raw_store_root=raw_root,
                predictor_derived_store_root=derived_root,
            )
        else:
            runner.build_predictor_source_manifest(
                [source],
                [DAILY_FILE],
                [bad_url],
                session_date="2026-08-06",
                source_received_at="2026-08-05T07:00:00+09:00",
                predictor_raw_store_root=raw_root,
                predictor_derived_store_root=derived_root,
                cache_chronology_class="anchor",
            )

    assert seal_calls == []
    assert (_tree_bytes(raw_root), _tree_bytes(derived_root)) == before


def _month_source_stub(*, sealed_at: str) -> dict[str, Any]:
    value = {field: None for field in runner.MONTH_SOURCE_MANIFEST_FIELDS}
    value.update(
        {
            "target_month": "2026-08",
            "seal_session": "2026-08-06",
            "sealed_at": sealed_at,
            "source_set_sha256": "1" * 64,
            "parsed_shard_set_sha256": "2" * 64,
            "g0_training_panel_semantic_sha256": "3" * 64,
        }
    )
    value["month_source_manifest_sha256"] = runner.canonical_json_sha256(
        value, exclude_fields={"month_source_manifest_sha256"}
    )
    return value


def _install_fake_fold_fit(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    training = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-07-31")],
            "code": ["1001"],
            "_daily_rank_target": [0.1],
            **{column: [0.0] for column in runner.G0_FEATURES},
        }
    )
    fake_ridge = SimpleNamespace(
        coef_=np.zeros(len(runner.G0_FEATURES), dtype=float),
        intercept_=np.asarray([0.0]),
    )
    fake_model = SimpleNamespace(named_steps={"model": fake_ridge})
    monkeypatch.setattr(
        runner.v17, "_candidate_training", lambda *args, **kwargs: training
    )
    monkeypatch.setattr(
        runner.v17, "fit_daily_rank_ridge", lambda *args, **kwargs: fake_model
    )
    monkeypatch.setattr(
        runner, "_numeric_execution", lambda **kwargs: nullcontext()
    )
    monkeypatch.setattr(
        runner,
        "export_c00_model_bundle",
        lambda *args, **kwargs: {
            "target_month": "2026-08",
            "created_at": kwargs["created_at"],
            "runtime_lock_sha256": runner.RUNTIME_LOCK_SHA256,
            "runtime_lock_verified_at": kwargs["runtime_lock_verified_at"],
            "transformed_feature_order": list(runner.G0_FEATURES),
            "imputer_statistics": {"sha256": "1" * 64},
            "imputer_indicator_features": {"sha256": "2" * 64},
            "scaler_mean": {"sha256": "3" * 64},
            "scaler_scale": {"sha256": "4" * 64},
            "ridge_coef": {"sha256": "5" * 64},
            "ridge_intercept": 0.0,
            "fold_model_bundle_sha256": "6" * 64,
        },
    )
    monkeypatch.setattr(
        runner,
        "_month_seal_session",
        lambda *args, **kwargs: pd.Timestamp("2026-08-05"),
    )
    return training


def test_real_fold_build_validates_with_exact_month_source_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # The repository's lightweight test venv may be newer than the frozen
    # operational wheel set.  Keep the real pandas/sklearn fitting path while
    # presenting the exact registered version labels to its manifest checks.
    monkeypatch.setattr(runner.np, "__version__", "2.3.5")
    monkeypatch.setattr(runner.pd, "__version__", "2.2.3")
    monkeypatch.setattr(runner.sklearn, "__version__", "1.8.0")
    dates = pd.DatetimeIndex(["2026-07-29", "2026-07-30", "2026-07-31"])
    rows: list[dict[str, Any]] = []
    for date_index, session in enumerate(dates):
        for code_index, code in enumerate(("1001", "1002", "1003")):
            rows.append(
                {
                    "date": session,
                    "code": code,
                    "common_training_eligible": True,
                    "oc_return_pct": float(code_index - 1) + date_index / 10.0,
                    **{
                        feature: float(feature_index + code_index + date_index)
                        for feature_index, feature in enumerate(runner.G0_FEATURES)
                    },
                }
            )
    panel = pd.DataFrame(rows)
    month_source = _month_source_stub(sealed_at="2026-08-06T06:30:00+09:00")
    synthetic_root = tmp_path / "synthetic-root"
    month_source_path = (
        synthetic_root
        / "research/model_v18_shoulder_state_month_source_manifests/2026-08.json"
    )
    month_source_path.parent.mkdir(parents=True)
    month_source_path.write_bytes(runner._json_file_bytes(month_source))
    month_source_path.chmod(0o600)
    monkeypatch.setattr(runner, "ROOT", synthetic_root)

    _, manifest, bundle = runner._build_fold(
        panel,
        pd.Period("2026-08"),
        fit_started_at="2026-08-06T07:00:00+09:00",
        fit_completed_at="2026-08-06T07:05:00+09:00",
        bundle_created_at="2026-08-06T07:06:00+09:00",
        sealed_at="2026-08-06T07:07:00+09:00",
        runtime_lock_verified_at="2026-08-06T06:00:00+09:00",
        first_counted_session_value="2026-08-06",
        month_source_manifest=month_source,
    )

    protocol_fields = tuple(
        runner.read_json(runner.PROTOCOL)["c00_contract"][
            "fold_manifest_required_fields"
        ]
    )
    assert tuple(manifest) == protocol_fields
    assert tuple(manifest)[13:19] == (
        "feature_names",
        "month_source_manifest_path",
        "month_source_manifest_sha256",
        "training_source_set_sha256",
        "training_parsed_shard_set_sha256",
        "training_g0_panel_semantic_sha256",
    )
    assert runner.validate_fold_manifest(manifest, bundle) == manifest[
        "fold_manifest_sha256"
    ]

    for field in (
        "month_source_manifest_sha256",
        "training_source_set_sha256",
        "training_parsed_shard_set_sha256",
        "training_g0_panel_semantic_sha256",
    ):
        tampered = dict(manifest)
        tampered[field] = "f" * 64
        tampered["fold_manifest_sha256"] = runner.canonical_json_sha256(
            tampered, exclude_fields={"fold_manifest_sha256"}
        )
        with pytest.raises(runner.V18Error, match="month-source binding changed"):
            runner.validate_fold_manifest(tampered, bundle)

    missing = dict(manifest)
    del missing["month_source_manifest_path"]
    with pytest.raises(runner.V18Error, match="missing fields"):
        runner.validate_fold_manifest(missing, bundle)

    extra = dict(manifest)
    extra["unregistered_month_source_binding"] = "f" * 64
    with pytest.raises(runner.V18Error, match="fields differ from protocol"):
        runner.validate_fold_manifest(extra, bundle)


def test_nonauthority_rehearsal_seam_uses_real_fold_score_and_one_panel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner.np, "__version__", "2.3.5")
    monkeypatch.setattr(runner.pd, "__version__", "2.2.3")
    monkeypatch.setattr(runner.sklearn, "__version__", "1.8.0")
    monkeypatch.setattr(
        runner, "PROTOCOL_SHA256", runner.sha256_file(runner.PROTOCOL)
    )
    protocol_value = runner.read_json(runner.PROTOCOL)
    runtime_value = runner.read_json(runner.RUNTIME_LOCK)
    monkeypatch.setattr(
        runner,
        "validate_protocol",
        lambda: (protocol_value, runner.PROTOCOL_SHA256),
    )
    monkeypatch.setattr(
        runner,
        "validate_runtime_lock",
        lambda *, strict_environment=False: (
            runtime_value,
            runner.RUNTIME_LOCK_SHA256,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_normalised_native_threadpools",
        lambda *, hash_libraries: [
            {
                key: (1 if key == "num_threads" else child)
                for key, child in item.items()
                if key != "library_sha256"
            }
            for item in runtime_value["runtime"]["native_threadpools"]
        ],
    )
    monkeypatch.setattr(runner, "_STRICT_RUNTIME_ACTIVE", True)
    monkeypatch.setattr(
        runner,
        "_STRICT_RUNTIME_VERIFIED_AT",
        runner._timestamp("2026-08-05T00:00:00+09:00", "test runtime verified"),
    )
    monkeypatch.setattr(
        runner, "_validate_startup_and_module_closure", lambda **kwargs: {}
    )
    for name in (
        "ACTIVATION_PAYLOAD",
        "ACTIVATION_RECEIPT",
        "ACTIVATION_CONTEXT",
        "RESULT_OUTPUT",
    ):
        monkeypatch.setattr(runner, name, tmp_path / f"absent-{name}.json")
    monkeypatch.setattr(runner, "MONTH_SOURCE_MANIFEST_DIR", tmp_path / "month")
    monkeypatch.setattr(runner, "FOLD_MANIFEST_DIR", tmp_path / "fold")
    monkeypatch.setattr(runner, "FOLD_MODEL_DIR", tmp_path / "bundle")
    rehearsal_contract = runner.prepare_a2_nonauthority_rehearsal_contract()

    dates = pd.bdate_range("2025-09-01", "2026-08-04")
    full = _parsed_prices(dates, "synthetic_rehearsal.pdf")
    date_index = pd.factorize(pd.to_datetime(full["date"]))[0]
    code_index = pd.factorize(full["code"].astype(str))[0]
    return_fraction = (((date_index * 2 + code_index) % 7) - 3) * 0.0015
    full["close"] = full["open"] * (1.0 + return_fraction)
    full["high"] = np.maximum(full["open"], full["close"]) * 1.003
    full["low"] = np.minimum(full["open"], full["close"]) * 0.997
    full["am_close"] = (full["open"] + full["close"]) / 2.0
    full["pm_open"] = full["am_close"]
    full["pm_close"] = full["close"]
    full["am_high"] = full["high"]
    full["pm_high"] = full["high"]
    full["am_low"] = full["low"]
    full["pm_low"] = full["low"]
    full["net_change"] = full["close"] - full["open"]
    full["vwap"] = (full["open"] + full["close"]) / 2.0
    full["turnover"] = full["volume"] * full["vwap"]
    full = runner._coerce_jsonl_frame(
        full, runner.PARSED_PRICE_COLUMNS, label="varied synthetic rehearsal prices"
    )
    projected = runner._coerce_model_price_frame(
        full, label="synthetic rehearsal compact projection"
    )
    july_snapshot = runner.canonical_model_price_csv_bytes(
        projected.loc[pd.to_datetime(projected["date"]).le("2026-07-31")],
        label="synthetic July snapshot",
    )
    june_snapshot = runner.canonical_model_price_csv_bytes(
        projected.loc[pd.to_datetime(projected["date"]).le("2026-06-30")],
        label="synthetic June snapshot",
    )
    august_suffix = full.loc[pd.to_datetime(full["date"]).gt("2026-07-31")]
    july_august_suffix = full.loc[pd.to_datetime(full["date"]).gt("2026-06-30")]
    times = {
        "target_session": "2026-08-05",
        "runtime_lock_verified_at": "2026-08-04T20:00:00+09:00",
        "month_source_sealed_at": "2026-08-05T06:30:00+09:00",
        "fit_started_at": "2026-08-05T07:00:00+09:00",
        "fit_completed_at": "2026-08-05T07:05:00+09:00",
        "score_generated_at": "2026-08-05T07:10:00+09:00",
        "source_manifest_sha256": "1" * 64,
        "source_set_sha256": "2" * 64,
        "parsed_shard_set_sha256": "3" * 64,
    }
    panel_calls = 0
    fold_calls = 0
    real_panel = runner.build_forward_c00_panel
    real_fold = runner._build_fold

    def counted_panel(*args: Any, **kwargs: Any) -> pd.DataFrame:
        nonlocal panel_calls
        panel_calls += 1
        return real_panel(*args, **kwargs)

    def counted_fold(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        nonlocal fold_calls
        fold_calls += 1
        return real_fold(*args, **kwargs)

    monkeypatch.setattr(runner, "build_forward_c00_panel", counted_panel)
    monkeypatch.setattr(runner, "_build_fold", counted_fold)

    with monkeypatch.context() as timed:
        timed.setattr(
            runner,
            "read_json",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("timed rehearsal read project JSON")
            ),
        )
        timed.setattr(
            runner,
            "sha256_file",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("timed rehearsal hashed a project file")
            ),
        )
        timed.setattr(
            runner.os.path,
            "lexists",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("timed rehearsal probed canonical authority")
            ),
        )
        timed.setattr(
            runner,
            "_latest_required_predictor_source_session",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("timed rehearsal reopened the calendar")
            ),
        )
        reference, _ = runner.build_a2_nonauthority_rehearsal_day(
            full,
            [],
            rehearsal_contract=rehearsal_contract,
            input_kind="full31_reference",
            **times,
        )
        boundary, boundary_token = runner.build_a2_nonauthority_rehearsal_day(
            july_snapshot,
            [august_suffix],
            rehearsal_contract=rehearsal_contract,
            input_kind="month_boundary_compact",
            **times,
        )
        intramonth, _ = runner.build_a2_nonauthority_rehearsal_day(
            june_snapshot,
            [july_august_suffix],
            rehearsal_contract=rehearsal_contract,
            input_kind="intramonth_fold_reuse_upper_bound_proxy",
            reuse_fold_token=boundary_token,
            **times,
        )
    runner.validate_a2_nonauthority_rehearsal_postflight(rehearsal_contract)

    assert panel_calls == 3
    assert fold_calls == 2
    assert reference["comparison"] == boundary["comparison"] == intramonth[
        "comparison"
    ]
    assert boundary["scope"] == "nonauthority_rehearsal_only"
    assert boundary["production_authority"] is False
    assert boundary["canonical_artifact_written"] is False
    assert boundary["comparison"]["g0_training_row_count"] > 0
    with pytest.raises(runner.V18Error, match="missing fields"):
        runner.validate_fold_manifest(boundary, boundary_token.model_bundle)


def test_terminal_month_authority_directories_are_exact_and_heal_link_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    directories = {
        "MONTH_SOURCE_MANIFEST_DIR": tmp_path / "month-source",
        "FOLD_MANIFEST_DIR": tmp_path / "fold",
        "FOLD_MODEL_DIR": tmp_path / "bundle",
    }
    for name, directory in directories.items():
        directory.mkdir(mode=0o700)
        os.chmod(directory, 0o700)
        monkeypatch.setattr(runner, name, directory)
        for month in ("2026-08", "2026-09"):
            path = directory / f"{month}.json"
            path.write_bytes(b"{}\n")
            os.chmod(path, 0o600)
    staged_final = directories["FOLD_MANIFEST_DIR"] / "2026-08.json"
    staged_alias = staged_final.parent / f".{staged_final.name}.staging"
    os.link(staged_final, staged_alias)

    runner._validate_exact_month_authority_directories(["2026-08", "2026-09"])

    assert staged_final.stat().st_nlink == 1
    assert not staged_alias.exists()
    extra = directories["FOLD_MODEL_DIR"] / "unregistered.json"
    extra.write_bytes(b"{}\n")
    os.chmod(extra, 0o600)
    with pytest.raises(runner.V18Error, match="directory name set changed"):
        runner._validate_exact_month_authority_directories(
            ["2026-08", "2026-09"]
        )


def test_terminal_month_authority_directory_rejects_nonprivate_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in (
        "MONTH_SOURCE_MANIFEST_DIR",
        "FOLD_MANIFEST_DIR",
        "FOLD_MODEL_DIR",
    ):
        directory = tmp_path / name
        directory.mkdir(mode=0o700)
        os.chmod(directory, 0o700)
        monkeypatch.setattr(runner, name, directory)
        path = directory / "2026-08.json"
        path.write_bytes(b"{}\n")
        os.chmod(path, 0o600)
    os.chmod(runner.MONTH_SOURCE_MANIFEST_DIR, 0o755)

    with pytest.raises(runner.V18Error, match="directory metadata changed"):
        runner._validate_exact_month_authority_directories(["2026-08"])


def test_fold_rejects_month_source_sealed_after_fit_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _install_fake_fold_fit(monkeypatch)

    with pytest.raises(
        runner.V18Error, match="fit predates its sealed month-source authority"
    ):
        runner._build_fold(
            panel,
            pd.Period("2026-08"),
            fit_started_at="2026-08-05T07:00:00+09:00",
            fit_completed_at="2026-08-05T07:10:00+09:00",
            bundle_created_at="2026-08-05T07:11:00+09:00",
            sealed_at="2026-08-05T07:12:00+09:00",
            runtime_lock_verified_at="2026-08-05T06:00:00+09:00",
            month_source_manifest=_month_source_stub(
                sealed_at="2026-08-05T07:01:00+09:00"
            ),
        )


def test_fold_rejects_seal_after_monthly_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _install_fake_fold_fit(monkeypatch)

    with pytest.raises(
        runner.V18Error, match="not sealed before first cutoff"
    ):
        runner._build_fold(
            panel,
            pd.Period("2026-08"),
            fit_started_at="2026-08-05T07:00:00+09:00",
            fit_completed_at="2026-08-05T07:10:00+09:00",
            bundle_created_at="2026-08-05T07:11:00+09:00",
            sealed_at="2026-08-05T09:00:00+09:00",
            runtime_lock_verified_at="2026-08-05T06:00:00+09:00",
            first_counted_session_value="2026-08-05",
            month_source_manifest=_month_source_stub(
                sealed_at="2026-08-05T06:30:00+09:00"
            ),
        )


@pytest.mark.parametrize("predecessor", ["2026-08-03", "2026-08-04"])
def test_activation_not_before_predecessor_must_strictly_follow_anchor_h(
    monkeypatch: pytest.MonkeyPatch,
    predecessor: str,
) -> None:
    anchor_summary = {"latest_source_session": "2026-08-04"}
    payload = {
        "not_before_session": "2026-08-05",
        "predictor_cache_anchor": anchor_summary,
        "additional_test_artifacts": [
            {"path": path, "sha256": "a" * 64}
            for path in runner.ADDITIONAL_TEST_ARTIFACT_PATHS
        ],
    }
    protocol = {
        "activation": {
            "preregistration_commit": {
                "additional_test_artifact_paths": list(
                    runner.ADDITIONAL_TEST_ARTIFACT_PATHS
                )
            },
            "payload": {
                "required_fields": list(payload),
                "fixed_values": {},
            }
        }
    }
    monkeypatch.setattr(
        runner,
        "validate_predictor_cache_anchor_summary",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        runner,
        "_validate_additional_test_artifacts",
        lambda value, **kwargs: tuple(dict(item) for item in value),
    )
    monkeypatch.setattr(
        runner,
        "_latest_required_predictor_source_session",
        lambda target: pd.Timestamp(predecessor),
    )

    with pytest.raises(
        runner.V18Error,
        match="not-before predecessor must strictly follow cache anchor H",
    ):
        runner.validate_activation_payload(payload, protocol=protocol)
