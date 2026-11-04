"""Collect the public egg-catalog series that answered without a new key.

Snapshots record what was retrieved. A catalog row with no series stays missing.
NASS and EIA demo credentials are not written into the snapshot.
"""

import csv
import io
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urljoin
from urllib.request import Request, build_opener

from quant_rsi.core.artifacts import atomic_write, canonical, digest
from quant_rsi.market.news_feed import NoRedirect
from quant_rsi.research.ams_shell_eggs import parse_shell_egg_report, pdf_text
from quant_rsi.research.nass_chickens import _download as nass_download
from quant_rsi.research.nass_chickens import _key as nass_key
from quant_rsi.research.nass_chickens import parse_nass_rows

MAX_BYTES = 8 * 1024 * 1024
MAX_PAGES = 24
MAX_FILES = 240
USER_AGENT = "QuantRSI research (github.com/natureofnature/Quant_RSI)"
EIA_DEMO_KEY = "DEMO_KEY"
DIESEL_SERIES = "EMD_EPD2D_PTE_NUS_DPG"
NY_PUBLICATION = "https://esmis.nal.usda.gov/publication/daily-new-york-shell-egg"
NY_CURRENT = "https://www.ams.usda.gov/mnreports/ams_2734.pdf"
BREAKING_PUBLICATION = "https://esmis.nal.usda.gov/publication/daily-national-breaking-stock"
BREAKING_CURRENT = "https://www.ams.usda.gov/mnreports/ams_3888.pdf"
SEDI_PAGES = (
    "https://esmis.nal.usda.gov/publication/shell-egg-demand-indicator",
    "https://esmis.nal.usda.gov/publication/weekly-shell-egg-demand-indicator",
)
INVENTORY_PAGES = (
    "https://esmis.nal.usda.gov/publication/weekly-shell-egg-inventory",
    "https://esmis.nal.usda.gov/publication/national-shell-egg-inventory",
)
APHIS_FILES = (
    "https://www.aphis.usda.gov/sites/default/files/hpai-commercial-backyard-flocks.csv",
    "https://www.aphis.usda.gov/sites/default/files/hpai-commercial-backyard-flocks.xlsx",
)
NOAA_IOWA = (
    "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/"
    "statewide/time-series/19/tavg/1/7/2024-2026.csv"
)
YAHOO = {
    "corn": "ZC=F",
    "meal": "ZM=F",
    "gas": "NG=F",
    "mxn": "MXN=X",
}
BLS_SERIES = {
    "bls_eggs": "APU0000708111",
    "bls_food": "CUUR0000SAF1",
}
EXTRA_SERIES = (
    {
        "field": "incubators",
        "entity": "US egg-type eggs in incubators",
        "short_desc": "CHICKENS, EGG TYPE - EGGS IN INCUBATORS, MEASURED IN EGGS",
        "domain_desc": "TOTAL",
        "freq_desc": "POINT IN TIME",
        "unit": "eggs",
    },
    {
        "field": "rate_of_lay",
        "entity": "US table-egg layers",
        "short_desc": "CHICKENS, LAYERS, TABLE - RATE OF LAY, MEASURED IN EGGS / 100 LAYER",
        "domain_desc": "TOTAL",
        "unit": "eggs_per_100_layers",
    },
    {
        "field": "molt",
        "entity": "US layers",
        "short_desc": "CHICKENS, LAYERS - BEING MOLTED, MEASURED IN PCT OF INVENTORY",
        "domain_desc": "TOTAL",
        "freq_desc": "POINT IN TIME",
        "unit": "percent_of_inventory",
        "allow_zero": True,
    },
    {
        "field": "molt_completed",
        "entity": "US layers",
        "short_desc": "CHICKENS, LAYERS - MOLT COMPLETED, MEASURED IN PCT OF INVENTORY",
        "domain_desc": "TOTAL",
        "freq_desc": "POINT IN TIME",
        "unit": "percent_of_inventory",
        "allow_zero": True,
    },
    {
        "field": "layer_loss",
        "entity": "US layers",
        "short_desc": "CHICKENS, LAYERS - LOSS, DEATH & RENDERED, MEASURED IN HEAD",
        "domain_desc": "TOTAL",
        "unit": "head",
    },
)
STATE_LAYERS = ("IA", "IN", "OH", "PA", "CA")
ROWS = (
    ("餐桌蛋產蛋雞群規模（table-egg layers）", "結構／產能", "layers", ""),
    ("育成雞／青年雞庫存（pullets）", "結構／產能", "placements", ""),
    ("蛋型種雞／孵化供給", "結構／產能", "", "沒有把孵化用蛋雞群收進這一次"),
    ("換羽比例（% in molt）", "結構／產能", "molt", ""),
    ("Cage-free／有機佔比", "結構／產能", "", "沒有公開的佔比序列"),
    ("Cage-free 轉型產能折損", "結構／產能", "", "沒有公開的產能折損序列"),
    ("區域集中度（IA／IN／OH／PA／CA 等）", "結構／產能", "concentration", ""),
    ("包裝／分級廠產能與利用率", "結構／產能", "", "沒有公開的利用率序列"),
    ("每百隻日產蛋率（rate of lay）", "供需", "rate_of_lay", ""),
    ("強制換羽／休產週期", "供需", "molt_completed", ""),
    ("孵化所蛋入孵／投放", "供需", "incubators", ""),
    ("雛雞安置（chick placements）", "供需", "hatched", ""),
    ("死亡率／淘汰與 disposition", "供需", "layer_loss", ""),
    ("殼蛋冷藏庫存", "供需", "", "ESMIS 沒有週庫存出版頁"),
    ("破殼／加工蛋庫存", "供需", "", "ESMIS 沒有破殼庫存出版頁"),
    ("庫存可售天數（days on hand）", "供需", "", "ESMIS 沒有 SEDI 出版頁"),
    ("Shell Egg Demand Indicator（SEDI）", "供需", "", "ESMIS 沒有 SEDI 出版頁"),
    ("破殼用蛋佔比（% to breakers）", "供需", "", "ESMIS 沒有 SEDI 出版頁"),
    ("HPAI 商業蛋雞確診場數", "疾病", "", "APHIS 頁面沒有可下載的批量檔"),
    ("HPAI 受影響隻數（蛋雞）", "疾病", "", "APHIS 頁面沒有可下載的批量檔"),
    ("HPAI 火雞等非蛋雞商業場", "疾病", "", "APHIS 頁面沒有可下載的批量檔"),
    ("後院／野生鳥確診", "疾病", "", "APHIS 頁面沒有可下載的批量檔"),
    ("州檢疫／移動限制", "疾病", "", "沒有集中的檢疫時間序列"),
    ("撲殺後補欄滯後", "疾病", "", "沒有撲殺隻數，不能減育成雞"),
    ("疫苗政策／授權傳聞", "疾病", "", "不解析新聞"),
    ("候鳥遷徙飛道季節", "疾病", "", "沒有飛道時間序列"),
    ("玉米現貨／CBOT 期貨", "飼料／成本", "corn", ""),
    ("豆粕（soybean meal）", "飼料／成本", "meal", ""),
    ("飼料綜合成本指數（自建）", "飼料／成本", "feed_index", ""),
    ("紙漿／紙盒包裝成本", "飼料／成本", "", "BLS 公開回應沒有序列標題，沒有收下"),
    ("農場勞工／最低工資", "飼料／成本", "", "BLS 公開回應沒有序列標題，沒有收下"),
    ("柴油／公路運輸燃料", "宏觀／能源", "diesel", ""),
    ("天然氣（禽舍加溫）", "宏觀／能源", "gas", ""),
    ("電力成本", "宏觀／能源", "", "沒有區域電價序列"),
    ("短期利率／營運資金成本", "宏觀／能源", "effr", ""),
    ("實質收入／消費信心", "宏觀／能源", "", "Conference Board 不是公開序列"),
    ("CPI 食品／通脹環境", "宏觀／能源", "bls_food", ""),
    ("復活節／聖週需求峰", "季節／天氣", "easter", ""),
    ("感恩節／聖誕烘焙季", "季節／天氣", "baking", ""),
    ("學校學期／餐飲週期", "季節／天氣", "", "這次日曆沒有做學曆"),
    ("夏季熱應激", "季節／天氣", "temperature", ""),
    ("中西部暴風雪／物流中斷", "季節／天氣", "", "不解析氣象新聞"),
    ("玉米帶乾旱（飼料前瞻）", "季節／天氣", "drought", ""),
    ("殼蛋／蛋製品出口量", "貿易", "", "Census 貿易接口要另外的鑰匙"),
    ("進口量（殼蛋／蛋製品）", "貿易", "", "Census 貿易接口要另外的鑰匙"),
    ("關稅／衛生貿易壁壘", "貿易", "", "沒有事件時間序列"),
    ("USD 對主要貿易夥伴匯率（如 MXN）", "貿易", "mxn", ""),
    ("貨櫃／冷鏈運費", "貿易", "", "Freightos 不是公開序列"),
    ("州 cage-free 法（CA Prop 12 等）", "政策／監管", "", "沒有收成法規時間序列"),
    ("USDA 分級／標示規則變更", "政策／監管", "", "沒有規則變更時間序列"),
    ("食品安全召回（沙門氏菌等）", "政策／監管", "", "沒有召回時間序列"),
    ("Farm bill／禽業相關條款", "政策／監管", "", "沒有法案時間序列"),
    ("乙醇／RFS 玉米政策溢出", "政策／監管", "", "沒有 RFS 時間序列"),
    ("零售促銷／feature activity", "替代品／需求", "", "週報概述不抽成數字"),
    ("餐飲／烘焙／食品服務需求", "替代品／需求", "", "沒有公開的餐飲需求序列"),
    ("Breaker 需求（液蛋／蛋粉）", "替代品／需求", "", "只收到破殼價格，沒有需求量"),
    ("膽固醇／健康新聞衝擊", "替代品／需求", "", "不解析新聞"),
    ("植物蛋白／蛋替代品滲透", "替代品／需求", "", "沒有公開滲透率"),
    ("人口／家庭戶數趨勢", "替代品／需求", "", "Census 要另外的鑰匙"),
    ("雞肉批發／零售價", "替代品／需求", "", "這次沒有收禽肉報價"),
    ("火雞價（節日季）", "替代品／需求", "", "這次沒有收火雞報價"),
    ("目標價格序列選擇（FOB／carton／breaking／CPI）", "市場微結構／報價", "target_prices", ""),
    ("AMS 日報報價量／成交量", "市場微結構／報價", "egg_volume", ""),
    ("TE 相對 AMS 發布滯後／對齊", "市場微結構／報價", "", "Trading Economics 不收"),
    ("現貨議價 vs 公式／合約價", "市場微結構／報價", "", "週報概述不抽成數字"),
    ("批發–零售價差", "市場微結構／報價", "spread", ""),
    ("媒體 HPAI 報導／搜尋熱度", "市場微結構／報價", "", "不收搜尋熱度"),
    ("大撲殺後 YoY 基數效應", "市場微結構／報價", "", "沒有撲殺隻數"),
    ("庫存／用量風格比率", "市場微結構／報價", "", "沒有週庫存，不能除產量"),
    ("ERS／USDA 產量與價格展望修訂", "市場微結構／報價", "", "沒有展望修訂序列"),
)


def parse_ny_shell_egg(text):
    """Average delivered price for caged white large, cents per dozen."""
    compact = " ".join(text.split())
    dated = re.search(r"Report for:\s*(\d{2}/\d{2}/\d{4})", compact)
    if not dated:
        raise ValueError("New York shell egg report has no report date")
    report_date = datetime.strptime(dated[1], "%m/%d/%Y").date()
    caged = compact.split("Explanatory", 1)[0]
    match = re.search(r"(?<!Extra )Large ([\d.]+) - ([\d.]+) ([\d.]+)", caged)
    if not match:
        raise ValueError("New York shell egg report has no large average")
    value = float(match[3])
    if not _positive(value):
        raise ValueError("Invalid New York large price")
    return {
        "report_date": report_date.isoformat(),
        "value": value,
        "unit": "cents_per_dozen",
        "row": "ny_caged_white_large_delivered",
    }


def parse_breaking_stock(text):
    """National caged breaking-stock 48 lb average, cents per dozen."""
    compact = " ".join(text.split())
    dated = re.search(r"Report for:\s*(\d{2}/\d{2}/\d{4})", compact)
    if not dated:
        raise ValueError("Breaking stock report has no report date")
    report_date = datetime.strptime(dated[1], "%m/%d/%Y").date()
    match = re.search(r"48 lb\.? ([\d.]+) - ([\d.]+) ([\d.]+)", compact)
    if not match:
        raise ValueError("Breaking stock report has no 48 lb average")
    value = float(match[3])
    if not _positive(value):
        raise ValueError("Invalid breaking stock price")
    return {
        "report_date": report_date.isoformat(),
        "value": value,
        "unit": "cents_per_dozen",
        "row": "national_caged_breaking_48lb",
    }


def esmis_links(raw, page_url, publication_url, suffix):
    """Keep same-host ESMIS PDFs whose name ends with this report."""
    if page_url.split("?", 1)[0] != publication_url:
        raise ValueError("ESMIS publication URL does not match")
    found = []
    for href in re.findall(r'href=["\']([^"\']+)', raw.decode("utf-8", "replace")):
        url = urljoin(page_url, href)
        if not url.startswith("https://esmis.nal.usda.gov/sites/default/release-files/"):
            continue
        if not url.lower().endswith(suffix.lower()):
            continue
        if url not in found:
            found.append(url)
    return found


def parse_yahoo_chart(payload, ticker):
    """Daily closes. Two bars on one New York date keep the later bar."""
    chart = payload.get("chart", {}) if isinstance(payload, dict) else None
    result = chart.get("result") if isinstance(chart, dict) else None
    if chart.get("error") or not isinstance(result, list) or len(result) != 1:
        raise ValueError(f"Unexpected Yahoo chart for {ticker}")
    row = result[0]
    stamps = row.get("timestamp") or []
    closes = ((row.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    if len(stamps) != len(closes):
        raise ValueError(f"Yahoo chart lengths differ for {ticker}")
    currency = (row.get("meta") or {}).get("currency")
    selected = {}
    for stamp, close in zip(stamps, closes, strict=True):
        if not isinstance(stamp, int) or not _positive(close):
            continue
        day = datetime.fromtimestamp(stamp, tz=_eastern()).date().isoformat()
        selected[day] = {"date": day, "value": float(close), "ticker": ticker, "currency": currency}
    if not selected:
        raise ValueError(f"Yahoo chart has no closes for {ticker}")
    return [selected[key] for key in sorted(selected)]


def parse_bls(payload, series_id):
    """Monthly values become visible on the 15th of the following month."""
    series = ((payload.get("Results") or {}).get("series") or [None])[0]
    if not isinstance(series, dict) or series.get("seriesID") != series_id:
        raise ValueError("Unexpected BLS response")
    selected = {}
    for row in series.get("data") or []:
        period = row.get("period")
        year = row.get("year")
        if not isinstance(period, str) or not re.fullmatch(r"M\d{2}", period):
            continue
        if not str(year).isdigit():
            continue
        month = int(period[1:])
        if not 1 <= month <= 12:
            continue
        try:
            value = float(row.get("value"))
        except (TypeError, ValueError):
            continue
        if not _positive(value):
            continue
        visible = _next_month_day(int(year), month, 15)
        selected[(int(year), month)] = {
            "date": f"{int(year):04d}-{month:02d}",
            "year": int(year),
            "month": month,
            "value": value,
            "visible_on": visible.isoformat(),
            "series_id": series_id,
        }
    if not selected:
        raise ValueError(f"BLS returned no {series_id} rows")
    return [selected[key] for key in sorted(selected)]


def parse_effr(text):
    """Keep the effective federal funds rate column from a NY Fed CSV page."""
    rows = csv.DictReader(io.StringIO(text))
    if rows.fieldnames is None or "Rate Type" not in rows.fieldnames:
        raise ValueError("NY Fed file is not the rate CSV")
    selected = {}
    for row in rows:
        if row.get("Rate Type") != "EFFR":
            continue
        try:
            published = datetime.strptime(row["Effective Date"], "%m/%d/%Y").date()
            value = float(row["Rate (%)"])
        except (KeyError, TypeError, ValueError):
            continue
        if not _positive(value):
            continue
        selected[published.isoformat()] = {
            "date": published.isoformat(),
            "value": value,
            "unit": "percent",
        }
    if not selected:
        raise ValueError("NY Fed file has no EFFR rows")
    return [selected[key] for key in sorted(selected)]


def parse_eia_diesel(payload):
    """Weekly U.S. on-highway diesel, dollars per gallon."""
    rows = (payload.get("response") or {}).get("data") or []
    selected = {}
    for row in rows:
        if row.get("series") != DIESEL_SERIES:
            continue
        period = row.get("period")
        try:
            value = float(row.get("value"))
            date.fromisoformat(period)
        except (TypeError, ValueError):
            continue
        if not _positive(value):
            continue
        selected[period] = {"date": period, "value": value, "unit": "dollars_per_gallon"}
    if not selected:
        raise ValueError("EIA returned no diesel rows")
    return [selected[key] for key in sorted(selected)]


def parse_drought(text):
    """Area in moderate drought or worse. Iowa is kept when present, otherwise CONUS."""
    rows = csv.DictReader(io.StringIO(text))
    if rows.fieldnames is None or "MapDate" not in rows.fieldnames:
        raise ValueError("Drought file is not the USDM table")
    selected = {}
    for row in rows:
        area = row.get("AreaOfInterest")
        if area not in {"IA", "ia", "CONUS"}:
            continue
        try:
            published = datetime.strptime(row["MapDate"], "%Y%m%d").date()
            value = sum(float(row[name]) for name in ("D1", "D2", "D3", "D4"))
        except (KeyError, TypeError, ValueError):
            continue
        if value < 0:
            continue
        selected[published.isoformat()] = {
            "date": published.isoformat(),
            "value": value,
            "unit": "percent_area_moderate_or_worse",
            "region": area,
        }
    regions = {row["region"] for row in selected.values()}
    if "IA" in regions or "ia" in regions:
        selected = {key: row for key, row in selected.items() if row["region"] in {"IA", "ia"}}
    if not selected:
        raise ValueError("Drought file has no Iowa or national rows")
    return [selected[key] for key in sorted(selected)]


def align_carried(records, session_dates, *, date_field="date", carry_days=6, hide_on_date=False):
    """Carry a dated value onto sessions. A longer gap stays empty."""
    ordered = []
    for record in records:
        try:
            published = date.fromisoformat(record[date_field])
            value = float(record["value"])
        except (KeyError, TypeError, ValueError):
            continue
        ordered.append((published, value))
    ordered.sort()
    collapsed = []
    for published, value in ordered:
        if collapsed and collapsed[-1][0] == published:
            collapsed[-1] = (published, value)
        else:
            collapsed.append((published, value))
    values = []
    for raw in session_dates:
        session = date.fromisoformat(raw)
        selected = None
        for index, (published, value) in enumerate(collapsed):
            if session < published or (hide_on_date and session == published):
                continue
            successor = collapsed[index + 1][0] if index + 1 < len(collapsed) else None
            if successor is not None and session > successor:
                continue
            if successor is None or (successor - published).days > carry_days:
                if session > published + timedelta(days=carry_days):
                    continue
            selected = value
        values.append(selected)
    return values


def calendar_column(records, session_dates, field):
    """Days until the holiday plus one, so the holiday session stays positive."""
    by_date = {row["date"]: row[field] + 1 for row in records}
    return [by_date.get(day) for day in session_dates]


def load_egg_catalog(directory, session_dates, cutoffs):
    """Read retained catalog snapshots. Missing files leave that field absent."""
    directory = Path(directory)
    from quant_rsi.research.ams_shell_eggs import align_shell_eggs
    from quant_rsi.research.nass_chickens import align_nass

    columns = {}
    yahoo = _read_snapshot(directory / "yahoo" / "snapshot.json")
    if yahoo:
        for field in ("corn", "meal", "gas", "mxn"):
            columns[field] = align_carried(yahoo["series"].get(field) or [], session_dates)
        columns["feed_index"] = align_carried(yahoo.get("feed_index") or [], session_dates)
    for folder, field, carry, hide in (
        ("diesel", "diesel", 10, True),
        ("effr", "effr", 6, True),
        ("drought", "drought", 14, True),
    ):
        snapshot = _read_snapshot(directory / folder / "snapshot.json")
        if snapshot:
            columns[field] = align_carried(
                snapshot["records"], session_dates, carry_days=carry, hide_on_date=hide
            )
    bls = _read_snapshot(directory / "bls" / "snapshot.json")
    if bls:
        for field in ("bls_eggs", "bls_food"):
            rows = [
                {"date": row["visible_on"], "value": row["value"]}
                for row in bls["series"].get(field) or []
            ]
            columns[field] = align_carried(rows, session_dates, carry_days=45)
    calendar = _read_snapshot(directory / "calendar" / "snapshot.json")
    if calendar:
        columns["easter"] = calendar_column(calendar["records"], session_dates, "easter")
        columns["thanksgiving"] = calendar_column(
            calendar["records"], session_dates, "thanksgiving"
        )
    for folder, field in (("ny_shell", "ny_carton"), ("breaking", "breaking")):
        snapshot = _read_snapshot(directory / folder / "snapshot.json")
        if snapshot:
            rows = [
                {"report_date": row["date"], "value": row["value"]} for row in snapshot["records"]
            ]
            columns[field] = align_shell_eggs(rows, session_dates)
    shell = _read_snapshot(directory / "shell_index.json")
    if shell:
        columns["egg_price"] = align_shell_eggs(
            [{"report_date": row["date"], "value": row["value"]} for row in shell["prices"]],
            session_dates,
        )
        columns["egg_volume"] = align_shell_eggs(
            [{"report_date": row["date"], "value": row["value"]} for row in shell["volumes"]],
            session_dates,
        )
    nass_rows = []
    for path in (directory / "nass_prior.json", directory / "nass" / "snapshot.json"):
        snapshot = _read_snapshot(path)
        if snapshot:
            nass_rows.extend(snapshot["records"])
    if nass_rows and cutoffs:
        for field in sorted({row["field"] for row in nass_rows}):
            columns[f"nass_{field}"] = align_nass(nass_rows, cutoffs, field)
    return columns


def _read_snapshot(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def calendar_rows(first, last):
    """Days until the next Easter, Thanksgiving, and Christmas, plus the month."""
    start, end = date.fromisoformat(first), date.fromisoformat(last)
    if start > end or (end - start).days > 4000:
        raise ValueError("Invalid calendar interval")
    rows = []
    day = start
    while day <= end:
        rows.append(
            {
                "date": day.isoformat(),
                "easter": _days_until(day, _easter),
                "thanksgiving": _days_until(day, _thanksgiving),
                "christmas": _days_until(day, lambda year: date(year, 12, 25)),
                "month": day.month,
            }
        )
        day += timedelta(days=1)
    return rows


def coverage_rows(found):
    """One row per catalog line. A missing key uses its written reason."""
    if len(ROWS) != 70:
        raise ValueError("The egg catalog no longer has 70 rows")
    rendered = []
    for label, category, key, gap in ROWS:
        item = found.get(key) if key else None
        if item and item.get("count"):
            rendered.append(
                {
                    "label": label,
                    "category": category,
                    "status": "collected",
                    "count": item["count"],
                    "start": item.get("start", ""),
                    "end": item.get("end", ""),
                    "unit": item.get("unit", ""),
                    "note": item.get("note", ""),
                }
            )
            continue
        reason = (item or {}).get("note") or gap or "這次沒有收到數字"
        rendered.append(
            {
                "label": label,
                "category": category,
                "status": "missing",
                "count": 0,
                "start": "",
                "end": "",
                "unit": "",
                "note": reason,
            }
        )
    return rendered


def render_coverage(rows):
    lines = [
        "# 蛋價目錄覆蓋",
        "",
        "有數的列是這次實際收到的原始序列。缺的列寫原因，沒有填數。",
        "",
        "| 因子 | 類別 | 狀態 | 筆數 | 起 | 迄 | 單位 | 說明 |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        cells = [
            row["label"],
            row["category"],
            "有數" if row["status"] == "collected" else "缺",
            str(row["count"]),
            row["start"],
            row["end"],
            row["unit"],
            row["note"].replace("|", "/"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    collected = sum(row["status"] == "collected" for row in rows)
    lines.extend(["", f"有數 {collected} 列，缺 {len(rows) - collected} 列。", ""])
    return "\n".join(lines)


def collect_catalog(directory, shell_dir, nass_dir, first="2024-01-01", last=None):
    """Write snapshots and coverage.md. Network calls stay inside this function."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    last = last or date.today().isoformat()
    found = {}
    _retain_json(directory / "shell_index.json", _shell_rows(shell_dir, found))
    _retain_json(directory / "nass_prior.json", _prior_nass(nass_dir, found))
    _collect_extra_nass(directory / "nass", found, int(first[:4]))
    _derive_concentration(found)
    _collect_pdfs(
        directory / "ny_shell",
        found,
        "ny_carton",
        NY_PUBLICATION,
        "/ams_2734.pdf",
        NY_CURRENT,
        parse_ny_shell_egg,
        first,
        last,
        "cents_per_dozen",
        "紐約紙盒、籠飼白殼大蛋送到店的均價。報告日當天不可見。",
    )
    _collect_pdfs(
        directory / "breaking",
        found,
        "breaking",
        BREAKING_PUBLICATION,
        "/ams_3888.pdf",
        BREAKING_CURRENT,
        parse_breaking_stock,
        first,
        last,
        "cents_per_dozen",
        "全國籠飼破殼 48 lb 均價，美分／打。報告日當天不可見。",
    )
    _probe_pages(SEDI_PAGES, "ESMIS 沒有 SEDI 出版頁")
    _probe_pages(INVENTORY_PAGES, "ESMIS 沒有週庫存出版頁")
    _collect_yahoo(directory / "yahoo", found, first)
    _collect_bls(directory / "bls", found)
    _derive_spread(found)
    _derive_target_prices(found)
    _collect_effr(directory / "effr", found)
    _collect_diesel(directory / "diesel", found, first)
    _collect_drought(directory / "drought", found, first, last)
    _collect_temperature(directory / "temperature", found)
    _collect_calendar(directory / "calendar", found, first, last)
    _probe_aphis()
    text = render_coverage(coverage_rows(found))
    atomic_write(directory / "coverage.md", text.encode())
    return found


def _shell_rows(shell_dir, found):
    directory = Path(shell_dir)
    snapshot = json.loads((directory / "snapshot.json").read_text())
    prices, volumes = [], []
    for record in snapshot["records"]:
        body = directory / f"{record['resource_sha256']}.body"
        parsed = parse_shell_egg_report(pdf_text(body.read_bytes()))
        prices.append(
            {"date": parsed["report_date"], "value": parsed["value"], "unit": parsed["unit"]}
        )
        volumes.append(
            {
                "date": parsed["report_date"],
                "value": parsed["volume"],
                "unit": parsed["volume_unit"],
            }
        )
    found["_egg_by_date"] = {row["date"]: row["value"] for row in prices}
    found["egg_price"] = _summary(
        prices, "cents_per_dozen", "全國 FOB 籠飼白殼大蛋加權均價。沿用已留下的 ESMIS 檔。"
    )
    found["egg_volume"] = _summary(volumes, "cases_30_dozen", "同一份報告裡白殼大蛋的報價量。")
    return {"prices": prices, "volumes": volumes}


def _prior_nass(nass_dir, found):
    snapshot = json.loads((Path(nass_dir) / "snapshot.json").read_text())
    notes = {
        "inventory": ("layers", "head", "餐桌蛋產蛋雞群。可見時間是 Quick Stats load_time。"),
        "placements": (
            "placements",
            "head",
            "這是育成雞投放，不是育成雞庫存。可見時間是 load_time。",
        ),
        "hatched": ("hatched", "head", "蛋雞雛孵化。可見時間是 load_time。"),
    }
    grouped = {field: [] for field in notes}
    for record in snapshot["records"]:
        if record["field"] not in grouped:
            continue
        grouped[record["field"]].append(
            {"date": f"{record['year']:04d}-{record['month']:02d}", "value": record["value"]}
        )
    for field, (key, unit, note) in notes.items():
        found[key] = _summary(grouped[field], unit, note)
    found["_layer_values"] = {row["date"]: row["value"] for row in grouped.get("inventory", [])}
    found["_egg_rows_ready"] = True
    return {"records": snapshot["records"], "source_sha256": snapshot["sha256"]}


def _collect_extra_nass(directory, found, first_year):
    directory = Path(directory)
    retained = directory / "snapshot.json"
    if retained.exists():
        snapshot = json.loads(retained.read_bytes())
        _check_snapshot(snapshot, "nass_egg_catalog")
        _apply_nass(snapshot["records"], found)
        for failure in snapshot.get("failures") or []:
            found.setdefault(
                failure["field"],
                {"count": 0, "note": f"NASS 這條沒有收下：{failure['reason']}"},
            )
        return snapshot
    directory.mkdir(parents=True, exist_ok=True)
    key = nass_key()
    records, resources, failures = [], [], []
    for series in _state_series():
        query = {
            "short_desc": series["short_desc"],
            "domain_desc": series["domain_desc"],
            "agg_level_desc": series.get("agg_level_desc", "NATIONAL"),
            "freq_desc": series.get("freq_desc", "MONTHLY"),
            "year__GE": str(first_year),
            "format": "JSON",
        }
        if series.get("state_alpha"):
            query["state_alpha"] = series["state_alpha"]
        try:
            raw = nass_download(nass_api(query, key))
            name = digest(raw) + ".body"
            atomic_write(directory / name, raw)
            parsed = parse_nass_rows(json.loads(raw), series)
            records.extend(parsed)
            resources.append({"file": name, "sha256": digest(raw), "field": series["field"]})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append({"field": series["field"], "reason": str(exc)})
    snapshot = {
        "provider": "nass_egg_catalog",
        "first_year": first_year,
        "records": records,
        "failures": failures,
        "sha256": digest(canonical(records)),
        "resources": resources,
        "notice": (
            "Extra NASS monthly and first-of-month series. Visibility is load_time. "
            "A later reload is not a point-in-time revision history. The key is not stored."
        ),
    }
    atomic_write(retained, canonical(snapshot))
    _apply_nass(records, found)
    for failure in failures:
        found.setdefault(
            failure["field"],
            {"count": 0, "note": f"NASS 這條沒有收下：{failure['reason']}"},
        )
    return snapshot


def _state_series():
    series = list(EXTRA_SERIES)
    for state in STATE_LAYERS:
        series.append(
            {
                "field": f"layers_{state.lower()}",
                "entity": f"{state} table-egg layers",
                "short_desc": "CHICKENS, LAYERS, TABLE - INVENTORY, AVG, MEASURED IN HEAD",
                "domain_desc": "INVENTORY OF LAYERS, TABLE",
                "agg_level_desc": "STATE",
                "state_alpha": state,
                "unit": "head",
            }
        )
    return series


def _apply_nass(records, found):
    notes = {
        "incubators": "蛋型種蛋入孵。頻率標成月初，不是 MONTHLY。",
        "rate_of_lay": "餐桌蛋產蛋雞每百隻產蛋數。",
        "molt": "正在換羽的產蛋雞佔存欄比例。",
        "molt_completed": "換羽完成的產蛋雞佔存欄比例。",
        "layer_loss": "產蛋雞死亡、淘汰與化製，單位是隻。",
    }
    grouped = {}
    for record in records:
        grouped.setdefault(record["field"], []).append(
            {
                "date": f"{record['year']:04d}-{record['month']:02d}",
                "value": record["value"],
                "unit": record["unit"],
            }
        )
    for field, note in notes.items():
        rows = grouped.get(field) or []
        if rows:
            found[field] = _summary(rows, rows[0]["unit"], note)
    found["_state_layers"] = grouped


def _derive_concentration(found):
    layers = found.get("_layer_values") or {}
    states = found.get("_state_layers") or {}
    ready = all(f"layers_{state.lower()}" in states for state in STATE_LAYERS)
    if not layers or not ready:
        found["concentration"] = {"count": 0, "note": "五州產蛋雞群沒有到齊"}
        return
    shares = []
    for month, total in sorted(layers.items()):
        parts = []
        for state in STATE_LAYERS:
            match = [row for row in states[f"layers_{state.lower()}"] if row["date"] == month]
            if len(match) != 1:
                parts = None
                break
            parts.append(match[0]["value"])
        if not parts or not _positive(total) or sum(parts) > total:
            continue
        shares.append({"date": month, "value": sum(parts) / total})
    if shares:
        found["concentration"] = _summary(
            shares, "share_of_national", "IA、IN、OH、PA、CA 餐桌蛋產蛋雞群除以全國。"
        )
        return
    found["concentration"] = {"count": 0, "note": "五州和全國對不齊，沒有算集中度"}


def _collect_pdfs(
    directory, found, key, publication, suffix, current, parse, first, last, unit, note
):
    directory = Path(directory)
    retained = directory / "snapshot.json"
    if retained.exists():
        snapshot = json.loads(retained.read_bytes())
        _check_snapshot(snapshot, key)
        found[key] = _summary(snapshot["records"], unit, note)
        return snapshot
    directory.mkdir(parents=True, exist_ok=True)
    urls, failures = [], []
    for page in range(MAX_PAGES):
        page_url = publication if page == 0 else f"{publication}?page={page}"
        try:
            raw = _download(page_url)
        except HTTPError as exc:
            failures.append({"url": page_url, "reason": f"HTTP {exc.code}"})
            break
        except (OSError, ValueError) as exc:
            failures.append({"url": page_url, "reason": str(exc)})
            break
        fresh = [url for url in esmis_links(raw, page_url, publication, suffix) if url not in urls]
        if not fresh:
            break
        urls.extend(fresh)
        if len(urls) > MAX_FILES:
            raise ValueError("ESMIS archive exceeds the file bound")
    urls.append(current)
    records = []
    earliest = date.fromisoformat(first) - timedelta(days=14)
    through = date.fromisoformat(last)
    for url in urls:
        try:
            raw = _download(url)
            parsed = parse(pdf_text(raw))
            published = date.fromisoformat(parsed["report_date"])
            if earliest <= published <= through:
                name = digest(raw) + ".body"
                if not (directory / name).exists():
                    atomic_write(directory / name, raw)
                records.append(
                    {
                        "date": parsed["report_date"],
                        "value": parsed["value"],
                        "unit": parsed["unit"],
                        "file": name,
                    }
                )
        except (OSError, ValueError) as exc:
            failures.append({"url": url, "reason": str(exc)})
    by_date = {}
    for record in records:
        previous = by_date.get(record["date"])
        if previous is not None and previous["value"] != record["value"]:
            raise ValueError(f"Conflicting {key} values for one report date")
        by_date[record["date"]] = record
    records = [by_date[item] for item in sorted(by_date)]
    snapshot = {
        "provider": key,
        "records": records,
        "failures": failures,
        "sha256": digest(canonical(records)),
        "notice": note,
    }
    atomic_write(retained, canonical(snapshot))
    if records:
        found[key] = _summary(records, unit, note)
    else:
        found[key] = {"count": 0, "note": failures[0]["reason"] if failures else "沒有讀到數字"}
    return snapshot


def _collect_yahoo(directory, found, first):
    directory = Path(directory)
    retained = directory / "snapshot.json"
    if retained.exists():
        snapshot = json.loads(retained.read_bytes())
        _check_snapshot(snapshot, "yahoo_feed")
        for key, rows in snapshot["series"].items():
            found[key] = _summary(rows, rows[0]["currency"] if rows else "", snapshot["notes"][key])
        if snapshot.get("feed_index"):
            found["feed_index"] = _summary(
                snapshot["feed_index"], "index", snapshot["notes"]["feed_index"]
            )
        else:
            found["feed_index"] = {"count": 0, "note": "玉米或豆粕沒有到齊"}
        return snapshot
    directory.mkdir(parents=True, exist_ok=True)
    series, notes, failures = {}, {}, []
    for key, ticker in YAHOO.items():
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{quote(ticker, safe='')}?interval=1d&range=5y"
        )
        try:
            rows = [
                row
                for row in parse_yahoo_chart(json.loads(_download(url)), ticker)
                if row["date"] >= first
            ]
            if not rows:
                raise ValueError(f"Yahoo chart has no rows on or after {first}")
            series[key] = rows
            notes[key] = f"Yahoo {ticker} 日線收盤。"
            found[key] = _summary(rows, rows[0]["currency"] if rows else "", notes[key])
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append({"field": key, "reason": str(exc)})
            found[key] = {"count": 0, "note": str(exc)}
    feed = _feed_index(series.get("corn") or [], series.get("meal") or [])
    notes["feed_index"] = "玉米與豆粕各自對第一個重疊日做成 100，再各半相加。"
    if feed:
        found["feed_index"] = _summary(feed, "index", notes["feed_index"])
    else:
        found["feed_index"] = {"count": 0, "note": "玉米或豆粕沒有到齊"}
    snapshot = {
        "provider": "yahoo_feed",
        "series": series,
        "feed_index": feed,
        "failures": failures,
        "notes": notes,
        "sha256": digest(canonical({"series": series, "feed_index": feed})),
    }
    atomic_write(retained, canonical(snapshot))
    return snapshot


def _feed_index(corn, meal):
    corn_map = {row["date"]: row["value"] for row in corn}
    meal_map = {row["date"]: row["value"] for row in meal}
    common = sorted(set(corn_map) & set(meal_map))
    if len(common) < 2:
        return []
    base_corn, base_meal = corn_map[common[0]], meal_map[common[0]]
    if not _positive(base_corn) or not _positive(base_meal):
        return []
    return [
        {
            "date": day,
            "value": 50.0 * (corn_map[day] / base_corn + meal_map[day] / base_meal),
            "unit": "index",
        }
        for day in common
    ]


def _collect_bls(directory, found):
    directory = Path(directory)
    retained = directory / "snapshot.json"
    if retained.exists():
        snapshot = json.loads(retained.read_bytes())
        _check_snapshot(snapshot, "bls_public")
        for key, rows in snapshot["series"].items():
            found[key] = _summary(rows, snapshot["units"][key], snapshot["notes"][key])
        found["_retail_rows"] = snapshot["series"].get("bls_eggs") or []
        return snapshot
    directory.mkdir(parents=True, exist_ok=True)
    series, notes, units, failures = {}, {}, {}, []
    labels = {
        "bls_eggs": ("dollars_per_dozen", "BLS 大蛋零售均價。參考月下一個月 15 日才可見。"),
        "bls_food": ("index", "BLS 城市平均食品 CPI。參考月下一個月 15 日才可見。"),
    }
    for key, series_id in BLS_SERIES.items():
        url = f"https://api.bls.gov/publicAPI/v1/timeseries/data/{series_id}"
        try:
            rows = parse_bls(json.loads(_download(url)), series_id)
            series[key] = rows
            units[key], notes[key] = labels[key]
            found[key] = _summary(rows, units[key], notes[key])
            if key == "bls_eggs":
                found["_retail_rows"] = rows
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append({"field": key, "reason": str(exc)})
            found[key] = {"count": 0, "note": str(exc)}
    snapshot = {
        "provider": "bls_public",
        "series": series,
        "units": units,
        "notes": notes,
        "failures": failures,
        "sha256": digest(canonical(series)),
    }
    atomic_write(retained, canonical(snapshot))
    return snapshot


def _derive_spread(found):
    eggs = found.get("_egg_by_date") or {}
    retail = found.get("_retail_rows") or found.get("bls_eggs", {}).get("_rows") or []
    if not eggs or not retail:
        found["spread"] = {"count": 0, "note": "批發指數或零售蛋價沒有到齊"}
        return
    spreads = []
    for row in retail:
        visible = date.fromisoformat(row["visible_on"])
        prior = [
            day
            for day in eggs
            if date.fromisoformat(day) <= visible and (visible - date.fromisoformat(day)).days <= 6
        ]
        if not prior:
            continue
        wholesale = eggs[max(prior)]
        spreads.append(
            {
                "date": row["visible_on"],
                "value": row["value"] * 100.0 - wholesale,
                "unit": "cents_per_dozen",
            }
        )
    if spreads:
        found["spread"] = _summary(
            spreads,
            "cents_per_dozen",
            "零售美分減去可見日前 6 日內的批發指數。沒有批發價的月份留空。",
        )
        return
    found["spread"] = {"count": 0, "note": "零售可見日附近沒有批發指數"}


def _derive_target_prices(found):
    names = [
        ("egg_price", "AMS_2843"),
        ("ny_carton", "AMS_2734"),
        ("breaking", "AMS_3888"),
        ("bls_eggs", "BLS 零售蛋價"),
    ]
    present = [label for key, label in names if (found.get(key) or {}).get("count")]
    if present:
        ready = [key for key, _label in names if (found.get(key) or {}).get("count")]
        found["target_prices"] = {
            "count": len(present),
            "start": min(found[key]["start"] for key in ready),
            "end": max(found[key]["end"] for key in ready),
            "unit": "series",
            "note": "收到 " + "、".join(present) + "。",
        }


def _collect_effr(directory, found):
    directory = Path(directory)
    retained = directory / "snapshot.json"
    if retained.exists():
        snapshot = json.loads(retained.read_bytes())
        _check_snapshot(snapshot, "ny_fed_effr")
        found["effr"] = _summary(snapshot["records"], "percent", snapshot["notice"])
        return snapshot
    directory.mkdir(parents=True, exist_ok=True)
    selected, start = {}, 0
    while start < 12000:
        url = (
            "https://markets.newyorkfed.org/read?productCode=50&eventCodes=500"
            f"&limit=500&startPosition={start}&sort=postDt:1&format=csv"
        )
        try:
            page = parse_effr(_download(url).decode("utf-8", "replace"))
        except ValueError:
            break
        added = 0
        for row in page:
            if row["date"] not in selected:
                selected[row["date"]] = row
                added += 1
        if added == 0 or len(page) < 500:
            break
        start += 500
    records = [selected[key] for key in sorted(selected) if key >= "2024-01-01"]
    notice = "紐約聯儲有效聯邦基金利率。日期是生效日，公布在下一個工作天。"
    snapshot = {
        "provider": "ny_fed_effr",
        "records": records,
        "sha256": digest(canonical(records)),
        "notice": notice,
    }
    atomic_write(retained, canonical(snapshot))
    found["effr"] = _summary(records, "percent", notice)
    return snapshot


def _collect_diesel(directory, found, first):
    directory = Path(directory)
    retained = directory / "snapshot.json"
    if retained.exists():
        snapshot = json.loads(retained.read_bytes())
        _check_snapshot(snapshot, "eia_diesel")
        found["diesel"] = _summary(snapshot["records"], "dollars_per_gallon", snapshot["notice"])
        return snapshot
    directory.mkdir(parents=True, exist_ok=True)
    url = (
        "https://api.eia.gov/v2/petroleum/pri/gnd/data/?api_key="
        + EIA_DEMO_KEY
        + "&frequency=weekly&data[0]=value&facets[series][]="
        + DIESEL_SERIES
        + f"&start={first}&sort[0][column]=period&sort[0][direction]=asc&offset=0&length=5000"
    )
    try:
        records = parse_eia_diesel(json.loads(_download(url)))
        records = [row for row in records if row["date"] >= first]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        found["diesel"] = {"count": 0, "note": str(exc)}
        return None
    notice = "EIA 每週美國公路二號柴油，公開 DEMO_KEY。period 是價格日，不是重建的發布時刻。"
    snapshot = {
        "provider": "eia_diesel",
        "records": records,
        "sha256": digest(canonical(records)),
        "notice": notice,
    }
    atomic_write(retained, canonical(snapshot))
    found["diesel"] = _summary(records, "dollars_per_gallon", notice)
    return snapshot


def _collect_drought(directory, found, first, last):
    directory = Path(directory)
    retained = directory / "snapshot.json"
    if retained.exists():
        snapshot = json.loads(retained.read_bytes())
        _check_snapshot(snapshot, "usdm_iowa")
        found["drought"] = _summary(
            snapshot["records"], snapshot["records"][0]["unit"], snapshot["notice"]
        )
        return snapshot
    start = datetime.strptime(first, "%Y-%m-%d")
    end = datetime.strptime(last, "%Y-%m-%d")
    url = (
        "https://usdmdataservices.unl.edu/api/USStatistics/"
        "GetDroughtSeverityStatisticsByAreaPercent?aoi=us"
        f"&startdate={start.month}/{start.day}/{start.year}"
        f"&enddate={end.month}/{end.day}/{end.year}&statisticsType=2"
    )
    try:
        records = parse_drought(_download(url).decode("utf-8", "replace"))
    except (OSError, ValueError) as exc:
        found["drought"] = {"count": 0, "note": f"乾旱接口沒有收下：{exc}"}
        return None
    directory.mkdir(parents=True, exist_ok=True)
    region = records[0].get("region", "CONUS")
    notice = (
        "美國本土中度以上乾旱面積比例。" if region == "CONUS" else "愛荷華州中度以上乾旱面積比例。"
    )
    notice += "不是整個玉米帶的加權指數。"
    snapshot = {
        "provider": "usdm_iowa",
        "records": records,
        "sha256": digest(canonical(records)),
        "notice": notice,
    }
    atomic_write(retained, canonical(snapshot))
    found["drought"] = _summary(records, "percent_area_moderate_or_worse", notice)
    return snapshot


def _collect_temperature(directory, found):
    directory = Path(directory)
    retained = directory / "snapshot.json"
    if retained.exists():
        snapshot = json.loads(retained.read_bytes())
        _check_snapshot(snapshot, "noaa_iowa_tavg")
        found["temperature"] = _summary(snapshot["records"], "fahrenheit", snapshot["notice"])
        return snapshot
    try:
        raw = _download(NOAA_IOWA).decode("utf-8", "replace")
    except (OSError, ValueError) as exc:
        found["temperature"] = {"count": 0, "note": f"NOAA 沒有收下：{exc}"}
        return None
    rows = []
    for line in raw.splitlines():
        if not re.match(r"\d{6},", line):
            continue
        period, value = line.split(",", 2)[:2]
        try:
            number = float(value)
            parsed = datetime.strptime(period, "%Y%m")
        except ValueError:
            continue
        rows.append({"date": parsed.date().isoformat(), "value": number, "unit": "fahrenheit"})
    if not rows:
        found["temperature"] = {"count": 0, "note": "NOAA 回應裡沒有愛荷華月均溫"}
        return None
    directory.mkdir(parents=True, exist_ok=True)
    notice = "NOAA 愛荷華州月均溫，華氏。不是產區加權溫度。"
    snapshot = {
        "provider": "noaa_iowa_tavg",
        "records": rows,
        "sha256": digest(canonical(rows)),
        "notice": notice,
    }
    atomic_write(retained, canonical(snapshot))
    found["temperature"] = _summary(rows, "fahrenheit", notice)
    return snapshot


def _collect_calendar(directory, found, first, last):
    rows = calendar_rows(first, last)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = {"provider": "egg_calendar", "records": rows, "sha256": digest(canonical(rows))}
    atomic_write(directory / "snapshot.json", canonical(snapshot))
    found["easter"] = _summary(
        [{"date": row["date"], "value": row["easter"]} for row in rows],
        "days",
        "距離下一個復活節的天數。當天是 0。",
    )
    found["baking"] = _summary(
        [{"date": row["date"], "value": row["thanksgiving"]} for row in rows],
        "days",
        "同一份日曆有感恩節和聖誕距離，以及月份。這裡筆數是感恩節距離。",
    )
    return snapshot


def _probe_pages(urls, reason):
    for url in urls:
        try:
            _download(url)
        except HTTPError as exc:
            if exc.code == 404:
                continue
            raise ValueError(f"{reason}：{url} 回 HTTP {exc.code}") from exc
        except OSError:
            continue
        else:
            raise ValueError(f"{url} 有出版頁，這次還沒有解析")
    return reason


def _probe_aphis():
    for url in APHIS_FILES:
        try:
            raw = _download(url)
        except (OSError, ValueError):
            continue
        if raw.startswith(b"PK") or raw[:40].lower().startswith((b"state", b"date")):
            raise ValueError(f"APHIS bulk file became available at {url}")
    return None


def nass_api(query, key):
    return "https://quickstats.nass.usda.gov/api/api_GET/?" + urlencode(query | {"key": key})


def _download(url):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with build_opener(NoRedirect).open(request, timeout=60) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Response exceeds the collection bound")
    return raw


def _summary(rows, unit, note):
    dates = [row["date"] for row in rows]
    return {
        "count": len(rows),
        "start": min(dates) if dates else "",
        "end": max(dates) if dates else "",
        "unit": unit,
        "note": note,
        "_rows": rows,
    }


def _check_snapshot(snapshot, provider):
    if snapshot.get("provider") != provider:
        raise ValueError("Retained catalog snapshot identity changed")
    if provider == "yahoo_feed":
        payload = {"feed_index": snapshot.get("feed_index"), "series": snapshot.get("series")}
    elif "records" in snapshot:
        payload = snapshot["records"]
    else:
        payload = snapshot.get("series")
    if digest(canonical(payload)) != snapshot["sha256"]:
        raise ValueError("Retained catalog snapshot identity changed")


def _retain_json(path, payload):
    atomic_write(Path(path), canonical(payload))


def _positive(value):
    return isinstance(value, (int, float)) and value > 0 and value == value


def _eastern():
    from zoneinfo import ZoneInfo

    return ZoneInfo("America/New_York")


def _next_month_day(year, month, day):
    if month == 12:
        return date(year + 1, 1, day)
    return date(year, month + 1, day)


def _days_until(day, holiday):
    this_year = holiday(day.year)
    target = this_year if this_year >= day else holiday(day.year + 1)
    return (target - day).days


def _easter(year):
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    g = (b - (b + 8) // 25 + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    month, day = divmod(h + ell - 7 * ((a + 11 * h + 22 * ell) // 451) + 114, 31)
    return date(year, month, day + 1)


def _thanksgiving(year):
    first = date(year, 11, 1)
    return first + timedelta(days=(3 - first.weekday()) % 7 + 21)
