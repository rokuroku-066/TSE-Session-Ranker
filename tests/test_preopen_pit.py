from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from tse_session_ranker.data.preopen_pit import (
    ActivityStatus,
    CoverageStatus,
    FeatureGroup,
    FieldStatus,
    PointInTimeRecord,
    SchemaError,
    decision_cutoff,
    latest_by_source,
    safe_imbalance,
    validate_append_batch,
    validate_record,
)
from tse_session_ranker.exceptions import LeakageError


TOKYO = ZoneInfo("Asia/Tokyo")
UTC = ZoneInfo("UTC")
TARGET = date(2026, 7, 23)


def at(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(
        2026,
        7,
        23,
        hour,
        minute,
        second,
        tzinfo=TOKYO,
    )


def pts_record(**overrides: object) -> PointInTimeRecord:
    base: dict[str, object] = {
        "target_date": TARGET,
        "instrument_id": "7203",
        "feature_group": FeatureGroup.PTS,
        "venue": "JNX",
        "session": "night",
        "source": "licensed_vendor",
        "source_record_id": "JNX-night-7203-20260723",
        "source_event_at": at(6, 0),
        "source_received_at": at(6, 0, 1),
        "computed_at": at(6, 0, 2),
        "coverage_status": CoverageStatus.COMPLETE,
        "activity_status": ActivityStatus.NO_ACTIVITY,
        "values": {
            "last_price": None,
            "volume": 0,
            "turnover": 0.0,
            "trade_count": 0,
        },
        "field_status": {
            "last_price": FieldStatus.MISSING,
            "volume": FieldStatus.ZERO,
            "turnover": FieldStatus.ZERO,
            "trade_count": FieldStatus.ZERO,
        },
        "source_version": "feed-v1",
    }
    base.update(overrides)
    return PointInTimeRecord(**base)  # type: ignore[arg-type]


class CutoffTests(unittest.TestCase):
    def test_cutoff_is_tokyo_aware(self) -> None:
        cutoff = decision_cutoff(TARGET)
        self.assertEqual(
            cutoff.isoformat(),
            "2026-07-23T08:58:59+09:00",
        )

    def test_exact_cutoff_is_inclusive(self) -> None:
        record = replace(
            pts_record(),
            source_received_at=at(8, 58, 58),
            computed_at=at(8, 58, 59),
        )
        self.assertIs(validate_record(record), record)

    def test_event_before_cutoff_but_receipt_after_cutoff_is_leakage(
        self,
    ) -> None:
        record = replace(
            pts_record(),
            source_event_at=at(8, 58, 50),
            source_received_at=at(8, 59, 1),
            computed_at=at(8, 59, 2),
        )
        with self.assertRaises(LeakageError):
            validate_record(record)

    def test_computation_after_cutoff_is_leakage(self) -> None:
        record = replace(
            pts_record(),
            source_received_at=at(8, 58, 50),
            computed_at=at(8, 59),
        )
        with self.assertRaises(LeakageError):
            validate_record(record)

    def test_timezone_conversion_does_not_change_availability(self) -> None:
        received_utc = at(8, 58, 50).astimezone(UTC)
        computed_utc = at(8, 58, 51).astimezone(UTC)
        record = replace(
            pts_record(),
            source_received_at=received_utc,
            computed_at=computed_utc,
        )
        self.assertIs(validate_record(record), record)

    def test_naive_timestamp_is_rejected(self) -> None:
        record = replace(
            pts_record(),
            computed_at=datetime(2026, 7, 23, 8, 58, 0),
        )
        with self.assertRaises(SchemaError):
            validate_record(record)

    def test_stale_record_rejected_by_group_specific_window(self) -> None:
        with self.assertRaises(SchemaError):
            validate_record(
                pts_record(),
                max_age=timedelta(minutes=30),
            )

    def test_prior_night_event_can_target_next_cash_session(self) -> None:
        previous_evening = datetime(
            2026,
            7,
            22,
            17,
            30,
            tzinfo=TOKYO,
        )
        record = replace(
            pts_record(),
            source_event_at=previous_evening,
            source_received_at=previous_evening
            + timedelta(milliseconds=20),
            computed_at=previous_evening + timedelta(milliseconds=40),
        )
        self.assertIs(validate_record(record), record)


class MissingZeroTests(unittest.TestCase):
    def test_no_trade_has_missing_price_and_explicit_zero_volume(
        self,
    ) -> None:
        record = pts_record()
        self.assertIs(validate_record(record), record)
        self.assertIsNone(record.values["last_price"])
        self.assertEqual(record.values["volume"], 0)

    def test_zero_price_is_not_a_no_trade_sentinel(self) -> None:
        record = replace(
            pts_record(),
            values={"last_price": 0, "volume": 0},
            field_status={
                "last_price": FieldStatus.ZERO,
                "volume": FieldStatus.ZERO,
            },
        )
        with self.assertRaises(SchemaError):
            validate_record(record)

    def test_numeric_zero_must_be_declared_zero(self) -> None:
        record = replace(
            pts_record(),
            values={"volume": 0},
            field_status={"volume": FieldStatus.OBSERVED},
        )
        with self.assertRaises(SchemaError):
            validate_record(record)

    def test_missing_requires_null(self) -> None:
        record = replace(
            pts_record(),
            values={"volume": 100},
            field_status={"volume": FieldStatus.MISSING},
        )
        with self.assertRaises(SchemaError):
            validate_record(record)

    def test_unavailable_feed_cannot_claim_no_activity(self) -> None:
        record = replace(
            pts_record(),
            coverage_status=CoverageStatus.UNAVAILABLE,
            activity_status=ActivityStatus.NO_ACTIVITY,
        )
        with self.assertRaises(SchemaError):
            validate_record(record)

    def test_unavailable_feed_remains_unknown_not_zero(self) -> None:
        record = replace(
            pts_record(),
            coverage_status=CoverageStatus.UNAVAILABLE,
            activity_status=ActivityStatus.UNKNOWN,
            values={"last_price": None, "volume": None},
            field_status={
                "last_price": FieldStatus.MISSING,
                "volume": FieldStatus.MISSING,
            },
        )
        self.assertIs(validate_record(record), record)


class LedgerTests(unittest.TestCase):
    def test_exact_duplicate_is_idempotent(self) -> None:
        record = pts_record()
        merged = validate_append_batch([record], [record])
        self.assertEqual(merged, (record,))

    def test_conflicting_duplicate_is_rejected(self) -> None:
        record = pts_record()
        changed = replace(
            record,
            values={"last_price": None, "volume": 100},
            field_status={
                "last_price": FieldStatus.MISSING,
                "volume": FieldStatus.OBSERVED,
            },
        )
        with self.assertRaises(SchemaError):
            validate_append_batch([record], [changed])

    def test_latest_snapshot_selected_by_actual_availability(self) -> None:
        early = replace(
            pts_record(),
            source_record_id="early",
            source_received_at=at(8, 55),
            computed_at=at(8, 55, 1),
        )
        late = replace(
            pts_record(),
            source_record_id="late",
            source_event_at=at(8, 57, 59),
            source_received_at=at(8, 58),
            computed_at=at(8, 58, 1),
        )
        selected = latest_by_source(
            [early, late],
            cutoff=decision_cutoff(TARGET),
        )
        self.assertEqual(selected, (late,))

    def test_latest_does_not_silently_drop_post_cutoff_rows(self) -> None:
        late = replace(
            pts_record(),
            source_received_at=at(8, 59),
            computed_at=at(8, 59, 1),
        )
        with self.assertRaises(LeakageError):
            latest_by_source(
                [late],
                cutoff=decision_cutoff(TARGET),
            )


class DerivedFeatureTests(unittest.TestCase):
    def test_imbalance(self) -> None:
        self.assertAlmostEqual(
            safe_imbalance(300, 100) or 0.0,
            0.5,
        )
        self.assertAlmostEqual(
            safe_imbalance(100, 300) or 0.0,
            -0.5,
        )

    def test_zero_total_or_missing_is_undefined(self) -> None:
        self.assertIsNone(safe_imbalance(0, 0))
        self.assertIsNone(safe_imbalance(None, 10))

    def test_negative_quantity_rejected(self) -> None:
        with self.assertRaises(SchemaError):
            safe_imbalance(-1, 10)


if __name__ == "__main__":
    unittest.main()
