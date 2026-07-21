from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tse_session_ranker.exceptions import DataValidationError
from tse_session_ranker.data.jpx import PARSER_VERSION, collect_jpx, parse_jpx_text


class JPXParserTests(unittest.TestCase):
    def test_full_partial_no_trade_and_alphanumeric_rows(self) -> None:
        content = "\n".join(
            [
                "20250701   13010   極洋    普通株式 4600 4635 4520 4525 4530 4550 4515 4530",
                "20250701   13820   ホーブ    普通株式 1980 1980 1970 1970",
                "20250701   29610   日本調理機    普通株式",
                "20250701   130A0   Ｖｅｒｉｔａｓ  Ｉｎ  Ｓｉｌｉｃｏ 普通株式 1090 1090 1020 1041 1041 1065 1035 1051",
                "20250701   13970   ＥＴＦ 受益証券 100 101 99 100 100 101 99 100",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text(content, encoding="utf-8")
            frame, report = parse_jpx_text(path)
        self.assertEqual(len(frame), 4)
        self.assertEqual(report.full_session_rows, 2)
        self.assertEqual(report.partial_session_rows, 1)
        self.assertEqual(report.no_trade_rows, 1)
        self.assertEqual(report.parser_version, PARSER_VERSION)
        kyokuyo = frame.loc[frame["code"].eq("1301")].iloc[0]
        self.assertEqual(kyokuyo["open"], 4600)
        self.assertEqual(kyokuyo["high"], 4635)
        self.assertEqual(kyokuyo["low"], 4515)
        self.assertEqual(kyokuyo["close"], 4530)
        self.assertEqual(
            kyokuyo[
                [
                    "am_open",
                    "am_high",
                    "am_low",
                    "am_close",
                    "pm_open",
                    "pm_high",
                    "pm_low",
                    "pm_close",
                ]
            ].tolist(),
            [4600, 4635, 4520, 4525, 4530, 4550, 4515, 4530],
        )
        hove = frame.loc[frame["code"].eq("1382")].iloc[0]
        self.assertTrue(hove["partial_session"])
        self.assertEqual(hove["open"], 1980)
        self.assertTrue(
            hove[
                [
                    "am_open",
                    "am_high",
                    "am_low",
                    "am_close",
                    "pm_open",
                    "pm_high",
                    "pm_low",
                    "pm_close",
                ]
            ].isna().all()
        )
        no_trade = frame.loc[frame["code"].eq("2961")].iloc[0]
        self.assertFalse(no_trade["traded"])
        self.assertTrue(no_trade[["open", "high", "low", "close"]].isna().all())
        self.assertTrue(
            no_trade[
                [
                    "am_open",
                    "am_high",
                    "am_low",
                    "am_close",
                    "pm_open",
                    "pm_high",
                    "pm_low",
                    "pm_close",
                ]
            ].isna().all()
        )
        self.assertIn("130A", set(frame["code"]))

    def test_collect_accepts_one_path_not_only_an_iterable(self) -> None:
        content = (
            "20250701 13010 極洋 普通株式 "
            "4600 4635 4520 4525 4530 4550 4515 4530\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text(content, encoding="utf-8")
            frame, report = collect_jpx(path)
        self.assertEqual(len(frame), 1)
        self.assertEqual(report["codes"], 1)
        self.assertEqual(frame.iloc[0]["am_close"], 4525)
        self.assertEqual(frame.iloc[0]["pm_open"], 4530)

    def test_daily_report_parses_regular_way_and_normalises_units(self) -> None:
        content = "\n".join(
            [
                "2025年8月1日(金曜日)                                      1- 1",
                "株 式 相 場 表 Stock Quotations",
                "Auction Trades Regular Way",
                (
                    "1301 100 Ｓ ＦＯＯＤＳ "
                    "4,665.00 4,740.00 4,660.00 4,740.00 "
                    "4,735.00 4,745.00 4,705.00 4,725.00 "
                    "－ 40.00 4,717.8889 31.5 148,613.500"
                ),
                (
                    "2961 100 日本調理機 "
                    "－ － － － 3,975.00 3,975.00 3,930.00 3,970.00 "
                    "－ -10.00 3,955.0000 0.5 1,977.500"
                ),
                "293A 100 Ｐ－ＢＡＢＹ ＪＯＢ " + " ".join(["－"] * 13),
                "2025年8月1日(金曜日)                                      1- 2",
                (
                    "130A 100 Ｖｅｒｉｔａｓ "
                    "1,090.00 1,090.00 1,020.00 1,041.00 "
                    "1,041.00 1,065.00 1,035.00 1,051.00 "
                    "－ -9.00 1,050.0000 1.25 1,312.500"
                ),
                "2025年8月1日(金曜日)                                      2- 1",
                (
                    "9999 100 後続表 "
                    "100 101 99 100 100 101 99 100 － 0 100 10 1,000"
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daily-report.txt"
            path.write_text(content, encoding="utf-8")
            frame, report = parse_jpx_text(path)

        self.assertEqual(set(frame["code"]), {"1301", "2961", "293A", "130A"})
        self.assertEqual(report.full_session_rows, 2)
        self.assertEqual(report.partial_session_rows, 1)
        self.assertEqual(report.no_trade_rows, 1)
        self.assertEqual(report.rejected_rows, 0)
        self.assertEqual(
            report.source_format,
            "jpx_stock_quotations_auction_regular_way_domestic_ordinary",
        )
        kyokuyo = frame.loc[frame["code"].eq("1301")].iloc[0]
        self.assertEqual(kyokuyo["name"], "Ｓ ＦＯＯＤＳ")
        self.assertEqual(kyokuyo["trading_unit"], 100)
        self.assertEqual(kyokuyo["volume"], 31_500)
        self.assertEqual(kyokuyo["turnover"], 148_613_500)
        self.assertEqual(kyokuyo["volume_unit"], "shares")
        self.assertEqual(kyokuyo["turnover_unit"], "JPY")
        self.assertEqual(str(kyokuyo["date"].date()), "2025-08-01")
        partial = frame.loc[frame["code"].eq("2961")].iloc[0]
        self.assertTrue(partial["partial_session"])
        self.assertEqual(
            partial[["open", "high", "low", "close"]].tolist(),
            [3975, 3975, 3930, 3970],
        )
        self.assertTrue(partial[["am_open", "pm_open"]].isna().all())
        no_trade = frame.loc[frame["code"].eq("293A")].iloc[0]
        self.assertFalse(no_trade["traded"])
        self.assertTrue(no_trade[["open", "high", "low", "close"]].isna().all())

    def test_daily_report_stops_at_domestic_preferred_stock(self) -> None:
        content = "\n".join(
            [
                "2025年8月1日(金曜日) 1-1",
                "Auction Trades Regular Way",
                "Domestic Stock",
                (
                    "1301 100 極洋 100 101 99 100 100 101 99 100 "
                    "－ 0 100 1 100"
                ),
                "Domestic Preferred Stock",
                # A later investment-certificate row also has a four-digit
                # code.  It must not leak into the ordinary-stock universe.
                (
                    "8421 1 Ｙ－信中金 193000 194000 192000 193000 "
                    "193000 194000 193000 193300 － 300 193328 0.3 58192"
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "universe-boundary.txt"
            path.write_text(content, encoding="utf-8")
            frame, report = parse_jpx_text(path)
        self.assertEqual(frame["code"].tolist(), ["1301"])
        self.assertEqual(report.ordinary_rows, 1)

    def test_daily_report_parses_special_quote_side_markers_only_in_its_field(
        self,
    ) -> None:
        content = "\n".join(
            [
                "2025年8月1日(金曜日) 1-1",
                "Auction Trades Regular Way",
                (
                    "1301 100 極洋 4,665 4,740 4,660 4,740 "
                    "4,735 4,745 4,705 4,725 ｶ4,745.00 40 4,717 1 4,717"
                ),
                (
                    "1332 100 ニッスイ 940 950 935 945 946 955 942 952 "
                    "ｳ2,676.00 7 949 1 949"
                ),
                # The same marker in Net Change is layout drift, not a valid
                # extension of the generic numeric grammar.
                (
                    "1333 100 マルハニチロ 2,900 2,910 2,890 2,905 "
                    "2,906 2,920 2,900 2,915 － ｶ10 2,908 1 2,908"
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "special-quote-marker.txt"
            path.write_text(content, encoding="utf-8")
            frame, report = parse_jpx_text(path)

        self.assertEqual(frame["code"].tolist(), ["1301", "1332"])
        self.assertEqual(
            frame.set_index("code")["final_special_quote"].to_dict(),
            {"1301": 4745.0, "1332": 2676.0},
        )
        self.assertEqual(report.ordinary_rows, 3)
        self.assertEqual(report.parsed_rows, 2)
        self.assertEqual(report.rejected_rows, 1)

    def test_daily_report_rejects_ambiguous_session_subset(self) -> None:
        content = "\n".join(
            [
                "2025年8月1日(金曜日) 1-1",
                "Auction Trades Regular Way",
                "1301 100 極洋 100 101 － － － － － － － 0 100 1 100",
                (
                    "1332 100 ニッスイ 100 101 99 100 100 101 99 100 "
                    "－ 0 100 1 100"
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "damaged.txt"
            path.write_text(content, encoding="utf-8")
            frame, report = parse_jpx_text(path)
        self.assertEqual(frame["code"].tolist(), ["1332"])
        self.assertEqual(report.rejected_rows, 1)

    def test_daily_report_rejects_conflicting_heading_dates(self) -> None:
        content = "\n".join(
            [
                "2025年8月1日(金曜日) 1-1",
                "Auction Trades Regular Way",
                (
                    "1301 100 極洋 100 101 99 100 100 101 99 100 "
                    "－ 0 100 1 100"
                ),
                "2025年8月2日(土曜日) 1-2",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conflict.txt"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(DataValidationError, "conflicting heading dates"):
                parse_jpx_text(path)


if __name__ == "__main__":
    unittest.main()
