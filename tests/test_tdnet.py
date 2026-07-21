from __future__ import annotations

import json
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tse_session_ranker.data.tdnet import (
    TDnetDataset,
    _cache_metadata,
    _write_cache_metadata,
    build_tdnet_features,
    collect_tdnet_dataset,
    merge_tdnet_features,
    normalize_tdnet_disclosures,
    parse_tdnet_index,
    tdnet_target_completeness,
)
from tse_session_ranker.api import SessionRanker, _tdnet
from tse_session_ranker.exceptions import DataValidationError


def _html(rows: list[tuple[str, str, str]]) -> str:
    cards = []
    for published_time, code, title in rows:
        cards.append(
            f"""
            <div class="row card">
              <div class="one column">{published_time}</div>
              <div class="three columns"><a href="/contents/tdnet/{code}">会社（{code}）</a></div>
              <div class="eight columns"><a href="https://example.test/{code}.pdf">{title}</a></div>
            </div>
            """
        )
    return (
        "<html><body><h1>2025年01月06日（月）に提出された適時開示情報</h1>"
        + "".join(cards)
        + "</body></html>"
    )


class TDnetTests(unittest.TestCase):
    def test_optional_name_and_url_columns_are_filled(self) -> None:
        normalized = normalize_tdnet_disclosures(
            pd.DataFrame(
                {
                    "published_at": ["2025-01-06T08:00:00+09:00"],
                    "code": ["1001"],
                    "title": ["業績予想の上方修正に関するお知らせ"],
                }
            )
        )
        self.assertEqual(normalized.loc[0, "name"], "1001")
        self.assertEqual(normalized.loc[0, "url"], "")

    def test_naive_published_time_is_rejected(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "explicit timezone"):
            normalize_tdnet_disclosures(
                pd.DataFrame(
                    {
                        "published_at": ["2025-01-06 08:00:00"],
                        "code": ["1001"],
                        "title": ["業績予想の上方修正に関するお知らせ"],
                    }
                )
            )

    def test_malformed_disclosure_card_is_rejected(self) -> None:
        html = (
            "<html><body><h1>2025年01月06日（月）に提出された適時開示情報</h1>"
            '<div class="row card"><div class="one column">16:00</div>'
            '<div class="eight columns">broken</div></div></body></html>'
        )
        with self.assertRaisesRegex(DataValidationError, "malformed disclosure card"):
            parse_tdnet_index(html, "2025-01-06")

    def test_mirror_advertisement_card_is_skipped_structurally(self) -> None:
        disclosure = _html(
            [("16:00", "1001", "業績予想の上方修正に関するお知らせ")]
        )
        advertisement = (
            '<div class="row card"><div class="u-full-width">'
            '<div class="ad-box"><ins data-revive-zoneid="27"></ins>'
            "</div></div></div>"
        )
        parsed = parse_tdnet_index(
            disclosure.replace("</body>", advertisement + "</body>"),
            "2025-01-06",
        )
        self.assertEqual(parsed["code"].tolist(), ["1001"])

    def test_mirror_layout_change_with_document_links_fails_closed(self) -> None:
        html = (
            "<html><body>"
            "<h1>2025年01月06日（月）に提出された適時開示情報</h1>"
            '<section class="new-disclosure-layout">'
            '<a href="/contents/tdnet/1001">会社（1001）</a>'
            '<a href="https://webapi.yanoshin.jp/rd.php?'
            'https://www.release.tdnet.info/inbs/example.pdf">'
            "業績予想の上方修正</a></section></body></html>"
        )
        with self.assertRaisesRegex(DataValidationError, "unparsed disclosure links"):
            parse_tdnet_index(html, "2025-01-06")

    def test_cache_sidecar_and_export_roundtrip_are_verified(self) -> None:
        index_date = pd.Timestamp("2025-01-06")
        payload = _html(
            [("16:00", "1001", "業績予想の上方修正に関するお知らせ")]
        ).encode("utf-8")
        observed_at = pd.Timestamp("2025-01-07T00:01:00+09:00")
        fetched_at = pd.Timestamp("2025-01-07T00:01:01+09:00")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            html_path = root / "20250106.html"
            html_path.write_bytes(payload)
            with self.assertRaisesRegex(DataValidationError, "metadata is missing"):
                collect_tdnet_dataset(root)
            _write_cache_metadata(
                html_path,
                _cache_metadata(index_date, payload, observed_at, fetched_at),
            )
            dataset = collect_tdnet_dataset(root)
            self.assertEqual(len(dataset.disclosures), 1)
            export = root / "tdnet.pkl"
            custom_manifest = root / "audit" / "tdnet-manifest.json"
            SessionRanker().collect_tdnet(
                root, output=export, manifest_path=custom_manifest
            )
            self.assertTrue(custom_manifest.exists())
            loaded = _tdnet(export)
            self.assertEqual(loaded.source_sha256, dataset.source_sha256)
            pd.testing.assert_frame_equal(loaded.disclosures, dataset.disclosures)
            manifest_path = export.with_suffix(export.suffix + ".manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            original_manifest = json.loads(json.dumps(manifest))
            observed_date = next(iter(manifest["observed_at_by_date"]))
            manifest["observed_at_by_date"][observed_date] = (
                "2025-01-07T08:59:00+09:00"
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(DataValidationError, "dataset checksum"):
                _tdnet(export)
            manifest_path.write_text(
                json.dumps(original_manifest), encoding="utf-8"
            )
            export.write_bytes(export.read_bytes() + b"tampered")
            with self.assertRaisesRegex(DataValidationError, "checksum"):
                _tdnet(export)

    def test_historical_cache_provenance_requires_explicit_research_opt_in(self) -> None:
        index_date = pd.Timestamp("2025-01-06")
        payload = _html(
            [("16:00", "1001", "業績予想の上方修正に関するお知らせ")]
        ).encode("utf-8")
        observed_at = pd.Timestamp("2026-07-21T12:00:00+09:00")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            html_path = root / "20250106.html"
            html_path.write_bytes(payload)
            metadata = _cache_metadata(
                index_date, payload, observed_at, observed_at
            )
            metadata["provenance"] = (
                "legacy_historical_file_mtime_assumption"
            )
            _write_cache_metadata(html_path, metadata)
            with self.assertRaisesRegex(DataValidationError, "provenance"):
                collect_tdnet_dataset(root)
            research = collect_tdnet_dataset(
                root, allow_historical_provenance=True
            )
            self.assertEqual(
                research.provenance_by_date.iloc[0],
                "legacy_historical_file_mtime_assumption",
            )

    def test_parser_and_085859_cutoff_mapping(self) -> None:
        parsed = parse_tdnet_index(
            _html(
                [
                    ("08:58", "1001", "配当予想の修正（増配）に関するお知らせ"),
                    ("08:59", "1002", "業績予想の上方修正に関するお知らせ"),
                    ("15:30", "1003", "自己株式取得に係る事項の決定に関するお知らせ"),
                ]
            ),
            "2025-01-06",
        )
        sessions = pd.to_datetime(["2025-01-06", "2025-01-07"])
        features = build_tdnet_features(parsed, sessions)
        same_day = features[
            features["date"].eq(pd.Timestamp("2025-01-06"))
        ]
        next_day = features[
            features["date"].eq(pd.Timestamp("2025-01-07"))
        ]
        self.assertEqual(same_day["code"].tolist(), ["1001"])
        self.assertEqual(set(next_day["code"]), {"1002", "1003"})
        self.assertEqual(float(same_day["tdnet_has_dividend_up"].iloc[0]), 1.0)

    def test_tostnet_and_completion_are_not_new_decisions(self) -> None:
        events = parse_tdnet_index(
            _html(
                [
                    (
                        "16:00",
                        "1001",
                        "ToSTNeT-3による自己株式取得に係る事項の決定に関するお知らせ",
                    ),
                    (
                        "16:10",
                        "1002",
                        "第三者割当による新株式発行に係る払込完了のお知らせ",
                    ),
                ]
            ),
            "2025-01-06",
        )
        features = build_tdnet_features(
            events, pd.to_datetime(["2025-01-06", "2025-01-07"])
        ).set_index("code")
        self.assertEqual(features.loc["1001", "tdnet_has_buyback_decision"], 0.0)
        self.assertEqual(features.loc["1002", "tdnet_has_equity_financing"], 0.0)

    def test_direction_words_require_the_matching_parent_category(self) -> None:
        events = parse_tdnet_index(
            _html(
                [
                    ("16:00", "1001", "配当予想の上方修正に関するお知らせ"),
                    ("16:10", "1002", "業績予想の上方修正に関するお知らせ"),
                ]
            ),
            "2025-01-06",
        )
        features = build_tdnet_features(
            events, pd.to_datetime(["2025-01-06", "2025-01-07"])
        ).set_index("code")
        self.assertEqual(features.loc["1001", "tdnet_has_revision_up"], 0.0)
        self.assertEqual(features.loc["1002", "tdnet_has_revision_up"], 1.0)

    def test_normal_audit_review_is_not_an_audit_problem(self) -> None:
        events = parse_tdnet_index(
            _html(
                [
                    (
                        "16:00",
                        "1001",
                        "決算短信（監査法人による期中レビューの完了）",
                    ),
                    (
                        "16:10",
                        "1002",
                        "監査法人の辞任及び継続企業の前提に関するお知らせ",
                    ),
                ]
            ),
            "2025-01-06",
        )
        features = build_tdnet_features(
            events, pd.to_datetime(["2025-01-06", "2025-01-07"])
        ).set_index("code")
        self.assertEqual(features.loc["1001", "tdnet_has_audit_problem"], 0.0)
        self.assertEqual(features.loc["1002", "tdnet_has_audit_problem"], 1.0)

    def test_weekend_index_gap_is_not_treated_as_no_disclosure(self) -> None:
        sessions = pd.to_datetime(["2025-01-10", "2025-01-14"])
        complete = pd.to_datetime(
            ["2025-01-10", "2025-01-11", "2025-01-13", "2025-01-14"]
        )
        observed = pd.Series(
            pd.to_datetime(
                [
                    "2025-01-11T00:01:00+09:00",
                    "2025-01-12T00:01:00+09:00",
                    "2025-01-14T00:01:00+09:00",
                    "2025-01-14T09:00:00+09:00",
                ]
            ),
            index=complete,
        )
        coverage = tdnet_target_completeness(sessions, complete, observed)
        self.assertFalse(bool(coverage.loc[pd.Timestamp("2025-01-14")]))

    def test_target_page_must_be_observed_after_preopen_cutoff(self) -> None:
        sessions = pd.to_datetime(["2025-01-06", "2025-01-07"])
        complete = pd.date_range("2025-01-06", "2025-01-07", freq="D")
        stale = pd.Series(
            pd.to_datetime(
                [
                    "2025-01-07T00:01:00+09:00",
                    "2025-01-07T08:58:30+09:00",
                ]
            ),
            index=complete,
        )
        coverage = tdnet_target_completeness(sessions, complete, stale)
        self.assertFalse(bool(coverage.loc[pd.Timestamp("2025-01-07")]))
        fresh = stale.copy()
        fresh.loc[pd.Timestamp("2025-01-07")] = pd.Timestamp(
            "2025-01-07T08:59:00+09:00"
        )
        coverage = tdnet_target_completeness(sessions, complete, fresh)
        self.assertTrue(bool(coverage.loc[pd.Timestamp("2025-01-07")]))
        invalid = fresh.copy()
        invalid.loc[pd.Timestamp("2025-01-07")] = pd.NaT
        coverage = tdnet_target_completeness(sessions, complete, invalid)
        self.assertFalse(bool(coverage.loc[pd.Timestamp("2025-01-07")]))

    def test_nondefault_decision_time_is_used_by_leak_check(self) -> None:
        sessions = pd.to_datetime(["2025-01-06", "2025-01-07"])
        html = _html(
            [("09:00", "1001", "業績予想の上方修正に関するお知らせ")]
        ).replace("2025年01月06日（月）", "2025年01月07日（火）")
        disclosures = parse_tdnet_index(html, "2025-01-07")
        complete = pd.date_range("2025-01-06", "2025-01-07", freq="D")
        dataset = TDnetDataset(
            disclosures=disclosures,
            complete_dates=complete,
            observed_at_by_date=pd.Series(
                pd.to_datetime(
                    [
                        "2025-01-07T00:01:00+09:00",
                        "2025-01-07T09:02:00+09:00",
                    ]
                ),
                index=complete,
            ),
            provenance_by_date=pd.Series(
                "network_request_start_recorded", index=complete
            ),
            source_sha256="nondefault-cutoff",
            source_files=2,
        )
        panel = pd.DataFrame(
            {
                "date": [pd.Timestamp("2025-01-07")],
                "code": ["1001"],
                "eligible": [True],
                "training_eligible": [True],
            }
        )
        merged = merge_tdnet_features(
            panel, dataset, sessions, decision_time="09:01:00"
        )
        self.assertEqual(merged.loc[0, "tdnet_has_revision_up"], 1.0)

    def test_incomplete_tdnet_source_masks_features_and_eligibility(self) -> None:
        sessions = pd.to_datetime(["2025-01-10", "2025-01-14"])
        disclosures = normalize_tdnet_disclosures(
            pd.DataFrame(
                {
                    "published_at": [
                        pd.Timestamp("2025-01-10 16:00", tz="Asia/Tokyo")
                    ],
                    "code": ["1001"],
                    "title": ["配当予想の修正（増配）に関するお知らせ"],
                    "name": ["会社"],
                    "url": ["https://example.test/a.pdf"],
                }
            )
        )
        dataset = TDnetDataset(
            disclosures=disclosures,
            complete_dates=pd.to_datetime(
                ["2025-01-10", "2025-01-11", "2025-01-13", "2025-01-14"]
            ),
            observed_at_by_date=pd.Series(
                pd.to_datetime(
                    [
                        "2025-01-11T00:01:00+09:00",
                        "2025-01-12T00:01:00+09:00",
                        "2025-01-14T00:01:00+09:00",
                        "2025-01-14T09:00:00+09:00",
                    ]
                ),
                index=pd.to_datetime(
                    ["2025-01-10", "2025-01-11", "2025-01-13", "2025-01-14"]
                ),
            ),
            provenance_by_date=pd.Series(
                "network_request_start_recorded",
                index=pd.to_datetime(
                    ["2025-01-10", "2025-01-11", "2025-01-13", "2025-01-14"]
                ),
            ),
            source_sha256="test",
            source_files=4,
        )
        panel = pd.DataFrame(
            {
                "date": [pd.Timestamp("2025-01-14")],
                "code": ["1001"],
                "eligible": [True],
                "training_eligible": [True],
            }
        )
        merged = merge_tdnet_features(panel, dataset, sessions)
        self.assertFalse(bool(merged.loc[0, "tdnet_source_complete"]))
        self.assertFalse(bool(merged.loc[0, "eligible"]))
        self.assertTrue(np.isnan(merged.loc[0, "tdnet_has_dividend_up"]))


if __name__ == "__main__":
    unittest.main()
