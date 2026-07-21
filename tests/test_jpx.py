from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tse_session_ranker.data.jpx import collect_jpx, parse_jpx_text


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
        kyokuyo = frame.loc[frame["code"].eq("1301")].iloc[0]
        self.assertEqual(kyokuyo["open"], 4600)
        self.assertEqual(kyokuyo["high"], 4635)
        self.assertEqual(kyokuyo["low"], 4515)
        self.assertEqual(kyokuyo["close"], 4530)
        hove = frame.loc[frame["code"].eq("1382")].iloc[0]
        self.assertTrue(hove["partial_session"])
        self.assertEqual(hove["open"], 1980)
        no_trade = frame.loc[frame["code"].eq("2961")].iloc[0]
        self.assertFalse(no_trade["traded"])
        self.assertTrue(no_trade[["open", "high", "low", "close"]].isna().all())
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


if __name__ == "__main__":
    unittest.main()
