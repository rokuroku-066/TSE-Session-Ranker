from __future__ import annotations

from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
from zoneinfo import ZoneInfo

import pytest

from tse_session_ranker.cli import build_parser
from tse_session_ranker.data.tdnet_material import (
    MATERIAL_PARSER_VERSION,
    ForecastRevision,
    OfficialDownloadReceipt,
    TDnetMaterialError,
    collect_official_forecast_revisions,
    download_official_tdnet_urls,
    forecast_material_strength,
    parse_forecast_revision_text,
    parse_official_tdnet_index,
)


INDEX_URL = (
    "https://www.release.tdnet.info/inbs/I_list_001_20260728.html"
)


def _official_html() -> str:
    return """
    <!DOCTYPE html>
    <html><body>
      <div id="kaiji-date-1">2026年07月28日</div>
      <div id="kaiji-text-1">に開示された情報</div>
      <div class="kaijiSum">1～2件&nbsp;/&nbsp;全2件</div>
      <div onClick="pagerLink('I_list_001_20260728.html')">1</div>
      <table id="main-list-table">
        <tr>
          <td class="oddnew-L kjTime">08:58</td>
          <td class="oddnew-M kjCode">63370</td>
          <td class="oddnew-M kjName">テセック</td>
          <td class="oddnew-M kjTitle">
            <a href="140120260728500719.pdf">業績予想の修正に関するお知らせ</a>
          </td>
          <td class="oddnew-M kjXbrl"></td>
          <td class="oddnew-M kjPlace">東</td>
          <td class="oddnew-R kjHistroy"></td>
        </tr>
        <tr>
          <td class="evennew-L kjTime">15:30</td>
          <td class="evennew-M kjCode">190A0</td>
          <td class="evennew-M kjName">新コード社</td>
          <td class="evennew-M kjTitle">
            <a href="140120260728500720.pdf">決算短信</a>
          </td>
          <td class="evennew-M kjXbrl">
            <div><a href="081220260728500720.zip">XBRL</a></div>
          </td>
          <td class="evennew-M kjPlace">東</td>
          <td class="evennew-R kjHistroy"></td>
        </tr>
      </table>
      <div class="kaijiSum">1～2件&nbsp;/&nbsp;全2件</div>
    </body></html>
    """


def _forecast_text() -> str:
    return """
    各 位                                                   2026年7月28日
                                  上場会社名  株式会社 テセック
                                  （コード番号 6337）

    業績予想の修正に関するお知らせ

    2027年３月期通期連結業績予想数値の修正(2026年４月１日〜2027年３月31日)
                                                    親会社株主に帰属     1株当たり
                    売上高    営業利益    経常利益
                                                    する当期純利益       当期純利益
                    百万円    百万円      百万円      百万円       円 銭
    前回発表予想(A)   6,300      450        590        540       104.67
    今回修正予想(B)   7,000    1,100      1,300      1,040       203.13
    増減額(B-A)         700      650        710        500
    増減率(%)           11.1    144.4      120.3       92.6
    """


def _revision(**overrides: object) -> ForecastRevision:
    values: dict[str, object] = {
        "company": "株式会社 テセック",
        "code": "6337",
        "period": "2027年3月期通期連結業績予想数値の修正",
        "scope": "consolidated",
        "unit": "JPY_million",
        "prior_sales": 6300.0,
        "prior_operating_profit": 450.0,
        "prior_ordinary_profit": 590.0,
        "prior_net_profit": 540.0,
        "prior_eps": 104.67,
        "current_sales": 7000.0,
        "current_operating_profit": 1100.0,
        "current_ordinary_profit": 1300.0,
        "current_net_profit": 1040.0,
        "current_eps": 203.13,
        "reported_sales_revision_pct": 11.1,
        "reported_operating_revision_pct": 144.4,
        "reported_ordinary_revision_pct": 120.3,
        "reported_net_revision_pct": 92.6,
        "reported_sales_delta": 700.0,
        "reported_operating_delta": 650.0,
        "reported_ordinary_delta": 710.0,
        "reported_net_delta": 500.0,
    }
    values.update(overrides)
    return ForecastRevision(**values)  # type: ignore[arg-type]


def test_official_index_parses_pagination_urls_and_codes() -> None:
    page = parse_official_tdnet_index(
        _official_html(),
        source_url=INDEX_URL,
    )
    assert page.index_date == date(2026, 7, 28)
    assert (page.range_start, page.range_end, page.total_disclosures) == (
        1,
        2,
        2,
    )
    assert [item.code for item in page.disclosures] == ["6337", "190A"]
    assert page.disclosures[0].published_at.isoformat() == (
        "2026-07-28T08:58:00+09:00"
    )
    assert page.disclosures[1].xbrl_url == (
        "https://www.release.tdnet.info/inbs/081220260728500720.zip"
    )
    assert page.page_urls == (INDEX_URL,)
    pager_page = parse_official_tdnet_index(
        _official_html().replace("pagerLink", "pager"),
        source_url=INDEX_URL,
    )
    assert pager_page == page


def test_official_zero_disclosure_page_requires_exact_message() -> None:
    html = """
    <!DOCTYPE html><html><body>
      <div id="kaiji-date-1">2026年07月28日</div>
      <div id="kaiji-text-1">
        に開示された情報はありません。
      </div>
    </body></html>
    """
    page = parse_official_tdnet_index(html, source_url=INDEX_URL)
    assert (page.range_start, page.range_end, page.total_disclosures) == (
        0,
        0,
        0,
    )
    assert page.disclosures == ()
    with pytest.raises(TDnetMaterialError, match="pagination summary"):
        parse_official_tdnet_index(
            html.replace(
                "開示された情報はありません。",
                "現在表示できません。",
            ),
            source_url=INDEX_URL,
        )
    with pytest.raises(TDnetMaterialError, match="pagination summary"):
        parse_official_tdnet_index(
            html.replace(
                '<div id="kaiji-text-1">',
                '<div class="other">1～1件 / 全1件</div>'
                '<div id="kaiji-text-1">',
            ),
            source_url=INDEX_URL,
        )
    for stray_link in (
        "<a href='140120260728500719.pdf'>PDF</a>",
        '<a href="081220260728500720.zip">XBRL</a>',
    ):
        with pytest.raises(TDnetMaterialError, match="pagination summary"):
            parse_official_tdnet_index(
                html.replace("</body>", f"{stray_link}</body>"),
                source_url=INDEX_URL,
            )
    with pytest.raises(TDnetMaterialError, match="date heading"):
        parse_official_tdnet_index(
            html.replace(
                '<div id="kaiji-date-1">',
                '<div id="kaiji-date-1">2026年07月27日</div>'
                '<div id="kaiji-date-1">',
            ),
            source_url=INDEX_URL,
        )
    with pytest.raises(TDnetMaterialError, match="pagination summary"):
        parse_official_tdnet_index(
            html.replace(
                '<div id="kaiji-text-1">',
                "<p>",
            ).replace("</div>\n    </body>", "</p>\n    </body>", 1),
            source_url=INDEX_URL,
        )
    with pytest.raises(TDnetMaterialError, match="pagination summary"):
        parse_official_tdnet_index(
            html.replace(
                '<div id="kaiji-text-1">',
                '<div class="kaijiSum">broken</div>'
                '<div id="kaiji-text-1">',
            ),
            source_url=INDEX_URL,
        )


def test_official_index_fails_closed_on_count_or_origin_drift() -> None:
    with pytest.raises(TDnetMaterialError, match="row count"):
        parse_official_tdnet_index(
            _official_html()
            .replace("1～2件", "1～3件")
            .replace("全2件", "全3件"),
            source_url=INDEX_URL,
        )
    with pytest.raises(TDnetMaterialError, match="official HTTPS"):
        parse_official_tdnet_index(
            _official_html(),
            source_url=INDEX_URL.replace(
                "www.release.tdnet.info",
                "example.invalid",
            ),
        )
    with pytest.raises(TDnetMaterialError, match="XBRL URL"):
        parse_official_tdnet_index(
            _official_html().replace(
                "081220260728500720.zip",
                "081220260728500720.zip?unverified=1",
            ),
            source_url=INDEX_URL,
        )


def test_downloader_rejects_proxy_or_guessed_urls_before_network() -> None:
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(TDnetMaterialError, match="official HTTPS"):
            download_official_tdnet_urls(
                "https://example.invalid/140120260728500719.pdf",
                directory,
            )
        with pytest.raises(TDnetMaterialError, match="unsupported"):
            download_official_tdnet_urls(
                "https://www.release.tdnet.info/inbs/guessed.pdf",
                directory,
            )


def test_downloader_rejects_redirected_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Headers:
        @staticmethod
        def get_content_type() -> str:
            return "text/html"

        @staticmethod
        def get(_name: str) -> None:
            return None

    class _RedirectedResponse:
        status = 200
        headers = _Headers()

        def __enter__(self) -> "_RedirectedResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def geturl() -> str:
            return "https://example.invalid/copied-tdnet.html"

        @staticmethod
        def read() -> bytes:
            raise AssertionError("redirected payload must not be read")

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _RedirectedResponse(),
    )
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(TDnetMaterialError, match="redirected"):
            download_official_tdnet_urls(INDEX_URL, directory)


def test_hash_matching_cache_still_requires_valid_official_payload() -> None:
    payload = b"not an official HTML response"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / Path(INDEX_URL).name
        path.write_bytes(payload)
        sidecar = path.with_suffix(path.suffix + ".receipt.json")
        sidecar.write_text(
            json.dumps(
                {
                    "url": INDEX_URL,
                    "path": str(path.resolve()),
                    "requested_at": "2026-07-28T08:00:00+09:00",
                    "received_at": "2026-07-28T08:00:01+09:00",
                    "http_status": 200,
                    "content_type": "text/html",
                    "last_modified": None,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "provenance": "network_request_start_recorded",
                    "downloaded_in_run": True,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(TDnetMaterialError, match="response is invalid"):
            download_official_tdnet_urls(INDEX_URL, directory)


def test_overwrite_preserves_prior_raw_and_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_payload = _official_html().replace(
        "<html><body>",
        "<html><body>適時開示情報閲覧サービス",
    ).encode()
    second_payload = first_payload.replace(
        "新コード社".encode(),
        "変更後コード社".encode(),
    )
    payloads = iter((first_payload, second_payload))

    class _Headers:
        @staticmethod
        def get_content_type() -> str:
            return "text/html"

        @staticmethod
        def get(name: str) -> str | None:
            return (
                "Tue, 28 Jul 2026 06:44:00 GMT"
                if name == "Last-Modified"
                else None
            )

    class _Response:
        status = 200
        headers = _Headers()

        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def geturl() -> str:
            return INDEX_URL

        def read(self) -> bytes:
            return self.payload

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(next(payloads)),
    )
    with tempfile.TemporaryDirectory() as directory:
        first = download_official_tdnet_urls(INDEX_URL, directory)[0]
        second = download_official_tdnet_urls(
            INDEX_URL,
            directory,
            overwrite=True,
        )[0]
        assert first.sha256 != second.sha256
        archive = (
            Path(directory)
            / "history"
            / Path(INDEX_URL).name
            / first.sha256
        )
        assert (archive / "payload").read_bytes() == first_payload
        archived_receipts = list(archive.glob("*.receipt.json"))
        assert len(archived_receipts) == 1
        assert (
            Path(second.path).read_bytes() == second_payload
        )


def test_cached_fixture_and_self_reported_receipt_cannot_prove_live_source() -> None:
    payload = _official_html().replace(
        "<html><body>",
        "<html><body>適時開示情報閲覧サービス",
    ).encode()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / Path(INDEX_URL).name
        path.write_bytes(payload)
        path.with_suffix(path.suffix + ".receipt.json").write_text(
            json.dumps(
                {
                    "url": INDEX_URL,
                    "path": str(path.resolve()),
                    "requested_at": "2026-07-28T08:00:00+09:00",
                    "received_at": "2026-07-28T08:00:01+09:00",
                    "http_status": 200,
                    "content_type": "text/html",
                    "last_modified": "Tue, 28 Jul 2026 06:44:00 GMT",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "provenance": "network_request_start_recorded",
                    "downloaded_in_run": True,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(TDnetMaterialError, match="fresh index network"):
            collect_official_forecast_revisions(INDEX_URL, directory)


def test_real_table_shape_recomputes_all_percentages_and_strength() -> None:
    revision = parse_forecast_revision_text(_forecast_text())
    assert revision.code == "6337"
    assert revision.company == "株式会社 テセック"
    assert revision.scope == "consolidated"
    assert revision.unit == "JPY_million"
    assert revision.parser_version == MATERIAL_PARSER_VERSION
    assert revision.prior_operating_profit == 450.0
    assert revision.current_operating_profit == 1100.0
    assert revision.reported_operating_revision_pct == 144.4
    assert forecast_material_strength(revision) == 0.30


def test_ambiguous_missing_or_inconsistent_table_is_rejected() -> None:
    with pytest.raises(TDnetMaterialError, match="exactly one current"):
        parse_forecast_revision_text(
            _forecast_text()
            + "\n今回修正予想(B) 7,000 1,100 1,300 1,040 203.13\n"
        )
    with pytest.raises(TDnetMaterialError, match="unavailable numeric"):
        parse_forecast_revision_text(
            _forecast_text().replace("1,100", "-")
        )
    with pytest.raises(TDnetMaterialError, match="does not recompute"):
        parse_forecast_revision_text(
            _forecast_text().replace("144.4", "12.3")
        )
    with pytest.raises(TDnetMaterialError, match="amount does not recompute"):
        parse_forecast_revision_text(
            _forecast_text().replace(
                "700      650",
                "700        1",
            )
        )
    with pytest.raises(TDnetMaterialError, match="unknown token"):
        parse_forecast_revision_text(
            _forecast_text().replace(
                "6,300      450        590        540       104.67",
                "未定 450 590 540 104.67 999",
            )
        )
    with pytest.raises(TDnetMaterialError, match="header and units"):
        parse_forecast_revision_text(
            _forecast_text().replace(
                "百万円    百万円",
                "千円      百万円",
                1,
            )
        )
    with pytest.raises(TDnetMaterialError, match="header and units"):
        parse_forecast_revision_text(
            _forecast_text().replace(
                "百万円       円 銭",
                "百万円       円 銭\n                    売上高",
                1,
            )
        )
    with pytest.raises(TDnetMaterialError, match="header and units"):
        parse_forecast_revision_text(
            _forecast_text().replace(
                "百万円       円 銭",
                "百万円       円 銭       円 銭",
                1,
            )
        )
    with pytest.raises(TDnetMaterialError, match="header and units"):
        parse_forecast_revision_text(
            _forecast_text().replace(
                "売上高    営業利益    経常利益",
                "売上高    営業利益    経常利益    営業利益",
                1,
            )
        )
    with pytest.raises(TDnetMaterialError, match="header and units"):
        parse_forecast_revision_text(
            _forecast_text().replace(
                "親会社株主に帰属     1株当たり",
                "親会社株主に帰属     1株当たり     1株当たり",
                1,
            )
        )
    with pytest.raises(TDnetMaterialError, match="header and units"):
        parse_forecast_revision_text(
            _forecast_text().replace(
                "売上高    営業利益",
                "営業利益  売上高",
                1,
            )
        )


def test_three_plus_page_graph_downloads_each_index_once_per_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_urls = [
        INDEX_URL.replace("_001_", f"_{page:03d}_")
        for page in range(1, 5)
    ]
    requested: list[tuple[str, str]] = []
    cached_stability_urls: set[str] = set()
    base_time = datetime(
        2026,
        7,
        28,
        8,
        0,
        tzinfo=ZoneInfo("Asia/Tokyo"),
    )

    def page_html(page: int) -> str:
        pager = "".join(
            f"""<div onClick="pagerLink('{Path(url).name}')">p</div>"""
            for url in page_urls
        )
        document_id = f"1401202607285{page:05d}"
        return f"""
        <!DOCTYPE html><html><body>
          適時開示情報閲覧サービス
          <div id="kaiji-date-1">2026年07月28日</div>
          <div id="kaiji-text-1">に開示された情報</div>
          <div class="kaijiSum">{page}～{page}件 / 全4件</div>
          {pager}
          <table>
            <tr>
              <td class="kjTime">08:0{page}</td>
              <td class="kjCode">63370</td>
              <td class="kjName">会社{page}</td>
              <td class="kjTitle">
                <a href="{document_id}.pdf">決算短信</a>
              </td>
              <td class="kjXbrl"></td>
              <td class="kjPlace">東</td>
            </tr>
          </table>
          <div class="kaijiSum">{page}～{page}件 / 全4件</div>
        </body></html>
        """

    def fake_download(
        urls: object,
        destination: str | Path,
        *,
        overwrite: bool = False,
        timeout: float = 30.0,
    ) -> list[OfficialDownloadReceipt]:
        del timeout
        values = [urls] if isinstance(urls, str) else list(urls)
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        receipts: list[OfficialDownloadReceipt] = []
        for offset, url in enumerate(values):
            requested.append((str(target.resolve()), str(url)))
            page = page_urls.index(str(url)) + 1
            path = target / Path(str(url)).name
            downloaded = overwrite or not path.exists()
            if target.name == "stability" and str(url) in cached_stability_urls:
                downloaded = False
            payload = page_html(page).encode()
            path.write_bytes(payload)
            timestamp = base_time + timedelta(seconds=len(requested) + offset)
            receipts.append(
                OfficialDownloadReceipt(
                    url=str(url),
                    path=str(path.resolve()),
                    requested_at=timestamp,
                    received_at=timestamp + timedelta(milliseconds=1),
                    http_status=200,
                    content_type="text/html",
                    last_modified="Tue, 28 Jul 2026 07:00:00 GMT",
                    bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    downloaded_in_run=downloaded,
                )
            )
        return receipts

    monkeypatch.setattr(
        "tse_session_ranker.data.tdnet_material."
        "download_official_tdnet_urls",
        fake_download,
    )
    with tempfile.TemporaryDirectory() as directory:
        manifest = collect_official_forecast_revisions(
            INDEX_URL,
            directory,
        )
        root = str(Path(directory).resolve())
        stability = str((Path(directory) / "stability").resolve())
        assert [
            url for destination, url in requested if destination == root
        ] == page_urls
        assert [
            url for destination, url in requested if destination == stability
        ] == page_urls
        assert manifest["coverage"]["index_pages"] == 4
        assert len(manifest["source"]["stability_receipts"]) == 4
    cached_stability_urls.add(page_urls[1])
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(TDnetMaterialError, match="changed during"):
            collect_official_forecast_revisions(
                INDEX_URL,
                directory,
            )


def test_material_strength_rejects_undefined_base_and_caps_both_tails() -> None:
    assert forecast_material_strength(
        _revision(
            current_operating_profit=495.0,
            reported_operating_revision_pct=10.0,
            reported_operating_delta=45.0,
        )
    ) == pytest.approx(0.10)
    assert forecast_material_strength(
        _revision(
            current_operating_profit=0.0,
            reported_operating_revision_pct=-100.0,
            reported_operating_delta=-450.0,
        )
    ) == -0.30
    with pytest.raises(TDnetMaterialError, match="positive prior"):
        forecast_material_strength(
            _revision(prior_operating_profit=0.0)
        )


def test_cli_exposes_receipted_download_and_material_probe() -> None:
    parser = build_parser()
    download = parser.parse_args(
        [
            "download-tdnet-official",
            "--url",
            INDEX_URL,
            "--destination",
            "/tmp/tdnet",
        ]
    )
    assert download.command == "download-tdnet-official"
    probe = parser.parse_args(
        [
            "probe-tdnet-material",
            "--index-url",
            INDEX_URL,
            "--destination",
            "/tmp/tdnet",
            "--manifest",
            "/tmp/probe.json",
        ]
    )
    assert probe.command == "probe-tdnet-material"
