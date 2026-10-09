"""Tests for scripts/research/dukascopy_web_to_node.py (Dukascopy website candles -> dukascopy-node layout).

The website export is EMULATED from dukascopy-node layout bars (timestamp in epoch ms, open, high, low,
close), in every layout the research found plausible; the converter must turn each one back into exactly
the original bars. The end-to-end tests do the same with the dry-run synthetic files (74,444 bars per side)
and then run `python -m propkit zeno-v1 signals` on the converted pair.

Environment (optional):
  DUKA_SYNTH_DIR  folder with SYNTHETIC_xauusd_m15_bid.csv and SYNTHETIC_xauusd_m15_ask.csv (the PC dry
                  run's made-up data); without it the end-to-end tests make the same data themselves
  PROPKIT_REPO    a folder holding the propkit package; default: this repository
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import random
import re
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "scripts" / "research"
sys.path.insert(0, str(HERE))
import dukascopy_web_to_node as conv  # noqa: E402

UTC = dt.timezone.utc
SCRIPT = HERE / "dukascopy_web_to_node.py"
SYNTH_DIR = Path(os.environ["DUKA_SYNTH_DIR"]) if os.environ.get("DUKA_SYNTH_DIR") else None
PROPKIT_REPO = Path(os.environ["PROPKIT_REPO"]) if os.environ.get("PROPKIT_REPO") else ROOT
NEWS_REL = "propkit/data/news_calendar/us_macro_events_2015-01-01_2025-09-27.csv"
HAVE_SYNTH = bool(SYNTH_DIR and (SYNTH_DIR / "SYNTHETIC_xauusd_m15_bid.csv").is_file())


# ----------------------------------------------------------------------------------------------------
# dukascopy-node layout data

def _t(text: str) -> int:
    return int(dt.datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp())


def make_node_rows(start: str, days: int, seed: int = 3, all_week: bool = False, price: float = 1200.0):
    """Small deterministic bid/ask bars in dukascopy-node layout: [(ts_ms, o, h, l, c)] as text."""
    rnd = random.Random(seed)
    t0 = _t(start)
    bid, ask = [], []
    c = price
    for k in range(days * 96):
        ts = t0 + k * 900
        d = dt.datetime.fromtimestamp(ts, UTC)
        if not all_week and (d.weekday() == 5 or (d.weekday() == 4 and d.hour >= 21)
                             or (d.weekday() == 6 and d.hour < 22)):
            continue
        o = round(c, 3)
        c = round(c + rnd.gauss(0, 1.5), 3)
        hi = round(max(o, c) + abs(rnd.gauss(0, 1.0)), 3)
        lo = round(min(o, c) - abs(rnd.gauss(0, 1.0)), 3)
        sp = round(0.25 + abs(rnd.gauss(0, 0.08)), 3)
        b = [o, hi, lo, c]
        a = [round(x + sp, 3) for x in b]
        bid.append((ts * 1000,) + tuple(conv.canon_plain(f"{x:.3f}") for x in b))
        ask.append((ts * 1000,) + tuple(conv.canon_plain(f"{x:.3f}") for x in a))
    return bid, ask


def read_node(path: Path):
    with open(path, newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    return [tuple([int(r[0])] + r[1:5]) for r in rows[1:]]


def read_converted(path: Path):
    with open(path, newline="", encoding="ascii") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["timestamp", "open", "high", "low", "close", "volume"]
    return rows[1:]


def assert_same_bars(converted: Path, original_rows) -> None:
    """The converted file holds exactly the original bars: same times, same decimal prices, and the price
    text is the original's printed as JavaScript prints numbers."""
    got = read_converted(converted)
    assert len(got) == len(original_rows)
    for g, o in zip(got, original_rows):
        assert int(g[0]) == o[0]
        assert [Decimal(x) for x in g[1:5]] == [Decimal(x) for x in o[1:5]]
        assert g[1:5] == [conv.canon_plain(x) for x in o[1:5]]
    raw = converted.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n") and raw.isascii()


# ----------------------------------------------------------------------------------------------------
# the website export, emulated

TIME_FORMATS = {
    "dmy_ms": "%d.%m.%Y %H:%M:%S.000",      # the quoted website rows: 03.10.2016 00:00:00.000
    "dmy_s": "%d.%m.%Y %H:%M:%S",
    "dmy_m": "%d.%m.%Y %H:%M",
    "iso_s": "%Y-%m-%d %H:%M:%S",
    "iso_T_ms": "%Y-%m-%dT%H:%M:%S.000",
    "ymd_dot": "%Y.%m.%d %H:%M",
}


def _last_sunday(year: int, month: int) -> dt.date:
    d = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


def eet_offset(ts: int) -> int:
    """EET/EEST (EU rule): +03:00 from the last Sunday of March 01:00 UTC to the last Sunday of October
    01:00 UTC, else +02:00. Written out so the tests need no time-zone database."""
    y = dt.datetime.fromtimestamp(ts, UTC).year
    a = _t(_last_sunday(y, 3).isoformat() + "T01:00:00")
    b = _t(_last_sunday(y, 10).isoformat() + "T01:00:00")
    return 3 * 3600 if a <= ts < b else 2 * 3600


def offset_text(off: int, style: str = "hhmm") -> str:
    sign = "+" if off >= 0 else "-"
    h, m = divmod(abs(off) // 60, 60)
    if style == "Z":
        assert off == 0
        return "Z"
    if style == "bare":
        return f" {sign}{h:02d}{m:02d}"
    return f" GMT{sign}{h:02d}:{m:02d}" if style == "colon" else f" GMT{sign}{h:02d}{m:02d}"


def time_text(ts: int, v: dict) -> str:
    tz = v.get("tz")
    off = None if tz is None else (eet_offset(ts) if tz == "eet" else int(tz))
    s = dt.datetime.fromtimestamp(ts + (off or 0), UTC).strftime(TIME_FORMATS[v.get("tfmt", "dmy_ms")])
    if off is not None:
        s += offset_text(off, v.get("tzstyle", "hhmm"))
    return s


def price_text(txt: str, v: dict) -> str:
    if v.get("decimals"):
        txt = f"{Decimal(txt):.{v['decimals']}f}"
    if v.get("decimal_comma"):
        txt = txt.replace(".", ",")
    return txt


def volume_text(ts: int, side: str, v: dict) -> str:
    n = (ts // 900) % 9973 + (7 if side == "ask" else 0)
    style = v.get("volume", "repr")
    if style == "repr":                  # binary float noise, as in 12.870000000000003
        txt = repr(0.1 + n * 0.01)
    elif style == "units2":              # 1130000.00
        txt = f"{(n + 1) * 1000:.2f}"
    elif style == "java_exp":            # Java Double.toString below 1e-3
        txt = f"{n % 9 + 1}.{n % 10}E-5"
    elif style == "js_exp":              # JavaScript below 1e-6
        txt = f"{n % 9 + 1}e-7"
    else:
        raise ValueError(style)
    return txt.replace(".", ",") if v.get("decimal_comma") else txt


def website_rows(node_rows, side: str, v: dict):
    """(ts_s, o, h, l, c, volume) as the website would list them; with v['flats'] every missing
    15-minute slot between the first and the last bar becomes a flat (O=H=L=C=previous close, volume 0)."""
    out = []
    if not v.get("flats"):
        for r in node_rows:
            ts = r[0] // 1000
            out.append((ts,) + tuple(r[1:5]) + (volume_text(ts, side, v),))
        return out
    by = {r[0] // 1000: r for r in node_rows}
    first, last = node_rows[0][0] // 1000, node_rows[-1][0] // 1000
    prev_close = None
    for ts in range(first, last + 900, 900):
        r = by.get(ts)
        if r is not None:
            out.append((ts,) + tuple(r[1:5]) + (volume_text(ts, side, v),))
            prev_close = r[4]
        else:
            fv = v.get("flat_volume", "0")
            out.append((ts, prev_close, prev_close, prev_close, prev_close, fv))
    return out


def _dmy(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, UTC).strftime("%d.%m.%Y")


def write_website(folder: Path, side: str, wrows, v: dict, start_s: int, end_s: int, name: str | None = None) -> Path:
    header = [v.get("header", "Gmt time"), "Open", "High", "Low", "Close", "Volume"]
    if v.get("columns"):
        header = v["columns"]
    d = v.get("delim", ",")
    lines = [d.join(header)]
    written = []
    for ts, o, h, lo, c, vol in wrows:
        if start_s <= ts < end_s:
            cells = {"time": time_text(ts, v), "Open": price_text(o, v), "High": price_text(h, v),
                     "Low": price_text(lo, v), "Close": price_text(c, v), "Volume": vol}
            lines.append(d.join(cells["time"] if k == header[0] else cells[k] for k in header))
            written.append(ts)
    nl = v.get("newline", "\n")
    text = nl.join(lines) + nl
    data = text.encode("utf-16") if v.get("utf16") else text.encode("ascii")
    if v.get("bom"):
        data = b"\xef\xbb\xbf" + data
    if name is None:   # the export page's naming: <PAIR>_Candlestick_<period>_<SIDE>_<from>-<to>.csv
        name = f"XAUUSD_Candlestick_15_M_{side.upper()}_{_dmy(written[0])}-{_dmy(written[-1])}.csv"
    p, k = folder / name, 1
    while p.exists():
        p = folder / f"{Path(name).stem} ({k}){Path(name).suffix}"
        k += 1
    p.write_bytes(data)
    return p


def part_windows(first: int, last: int, how: str):
    """[start, end) windows in UTC seconds covering first..last."""
    if how == "one":
        return [(first, last + 900)]
    y0 = dt.datetime.fromtimestamp(first, UTC).year
    y1 = dt.datetime.fromtimestamp(last, UTC).year
    if how == "yearly":                  # the steps' plan: one part per calendar year
        return [(max(first, _t(f"{y}-01-01")), min(last + 900, _t(f"{y + 1}-01-01"))) for y in range(y0, y1 + 1)]
    if how == "yearly_overlap":          # parts overlapping by 3 days on each side
        return [(max(first, _t(f"{y}-01-01") - 3 * 86400), min(last + 900, _t(f"{y + 1}-01-01") + 3 * 86400))
                for y in range(y0, y1 + 1)]
    if how == "weekly_overlap":          # short parts that overlap by a day
        out, s = [], first
        while s <= last:
            out.append((s, min(last + 900, s + 8 * 86400)))
            s += 7 * 86400
        return out
    raise ValueError(how)


def emulate(folder: Path, bid_rows, ask_rows, v: dict) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for side, rows in (("bid", bid_rows), ("ask", ask_rows)):
        wrows = website_rows(rows, side, v)
        for s, e in part_windows(wrows[0][0], wrows[-1][0], v.get("parts", "one")):
            paths.append(write_website(folder, side, wrows, v, s, e))
        if v.get("dup_file"):              # the browser saved one part twice: "... (1).csv"
            src = paths[-1]
            dup = src.with_name(src.stem + " (1).csv")
            dup.write_bytes(src.read_bytes())
            paths.append(dup)
    return paths


def expected_volume(text: str) -> str:
    t = text.replace(",", ".")
    return conv.canon_plain(format(Decimal(t), "f"))


def run_conv(args, capsys):
    rc = conv.main([str(a) for a in args])
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


# ----------------------------------------------------------------------------------------------------
# every plausible layout converts back to the original bars

SMALL = make_node_rows("2014-12-29", 45)                       # includes bars before 2015-01-01
DST_SPRING = make_node_rows("2015-03-27", 4, seed=5, all_week=True)   # 29 Mar 2015 01:00 UTC switch
DST_AUTUMN = make_node_rows("2015-10-23", 4, seed=6, all_week=True)   # 25 Oct 2015: a repeated local hour

VARIANTS = {
    "gmt_ms_lf": {},
    "gmt_s_crlf_bom": {"tfmt": "dmy_s", "newline": "\r\n", "bom": True},
    "GMT_time_no_seconds": {"header": "GMT time", "tfmt": "dmy_m"},
    "time_utc_iso": {"header": "Time (UTC)", "tfmt": "iso_s"},
    "time_gmt+0_header": {"header": "Time (GMT+0)"},
    "GmtTime_one_word": {"header": "GmtTime"},
    "gmt_iso_T": {"tfmt": "iso_T_ms"},
    "gmt_year_first_dots": {"tfmt": "ymd_dot"},
    "gmt_with_gmt+0000": {"tz": 0},
    "local_singapore": {"header": "Local time", "tz": 8 * 3600},
    "local_singapore_colon": {"header": "Local time", "tz": 8 * 3600, "tzstyle": "colon"},
    "local_new_york_winter": {"header": "Local time", "tz": -5 * 3600},
    "local_bare_offset": {"header": "Local time", "tz": 8 * 3600, "tzstyle": "bare"},
    "gmt_iso_Z": {"tfmt": "iso_T_ms", "tz": 0, "tzstyle": "Z"},
    "time_header_with_offsets": {"header": "Time", "tz": 2 * 3600},
    "semicolon_decimal_comma": {"delim": ";", "decimal_comma": True},
    "tab_fixed_5_decimals": {"delim": "\t", "decimals": 5},
    "pipe_fixed_3_decimals": {"delim": "|", "decimals": 3},
    "utf16_tab": {"delim": "\t", "utf16": True},
    "volume_units_2dp": {"volume": "units2"},
    "volume_java_exponent": {"volume": "java_exp"},
    "volume_js_exponent": {"volume": "js_exp"},
    "columns_reordered": {"columns": ["Gmt time", "Volume", "Close", "Open", "High", "Low"]},
    "flats_volume_0": {"flats": True, "flat_volume": "0"},
    "flats_volume_0.00": {"flats": True, "flat_volume": "0.00"},
    "weekly_parts_overlap_dup_file_flats": {"parts": "weekly_overlap", "dup_file": True, "flats": True},
    "local_sgt_weekly_parts_crlf": {"header": "Local time", "tz": 8 * 3600, "parts": "weekly_overlap",
                                    "newline": "\r\n"},
}


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_variant_round_trip(name, tmp_path, capsys):
    v = VARIANTS[name]
    bid, ask = SMALL
    emulate(tmp_path / "web", bid, ask, v)
    rc, out, err = run_conv([tmp_path / "web", "--out", tmp_path / "out"], capsys)
    assert rc == 0, err
    assert_same_bars(tmp_path / "out" / "xauusd_m15_bid.csv", bid)
    assert_same_bars(tmp_path / "out" / "xauusd_m15_ask.csv", ask)
    for side, rows in (("bid", bid), ("ask", ask)):
        vols = [r[5] for r in read_converted(tmp_path / "out" / f"xauusd_m15_{side}.csv")]
        assert vols == [expected_volume(volume_text(r[0] // 1000, side, v)) for r in rows]
    assert "PAIR: bid and ask hold exactly the same" in out
    early = sum(1 for r in bid if r[0] // 1000 < conv.RANGE_START_S)
    assert early > 0 and f"opening before 2015-01-01 00:00 UTC: {early} (kept" in out
    if v.get("flats"):
        assert "flats dropped (volume exactly 0, as dukascopy-node's default ignoreFlats): 0 " not in out
        assert "prices moving: 0)" in out


@pytest.mark.parametrize("rows", [DST_SPRING, DST_AUTUMN], ids=["spring", "autumn"])
@pytest.mark.parametrize("style", ["hhmm", "colon"])
def test_local_eet_with_daylight_saving(rows, style, tmp_path, capsys):
    v = {"header": "Local time", "tz": "eet", "tzstyle": style}
    bid, ask = rows
    paths = emulate(tmp_path / "web", bid, ask, v)
    if rows is DST_AUTUMN:                 # the repeated local hour is really in the file
        local = [ln.split(",")[0].split(" GMT")[0] for ln in paths[0].read_text().splitlines()[1:]]
        assert len(local) != len(set(local))
    rc, out, err = run_conv([tmp_path / "web", "--out", tmp_path / "out"], capsys)
    assert rc == 0, err
    assert_same_bars(tmp_path / "out" / "xauusd_m15_bid.csv", bid)
    assert_same_bars(tmp_path / "out" / "xauusd_m15_ask.csv", ask)


def test_flats_are_dropped_and_counted(tmp_path, capsys):
    bid, ask = SMALL
    v = {"flats": True}
    emulate(tmp_path / "web", bid, ask, v)
    n_flats = len(website_rows(bid, "bid", v)) - len(bid)
    rc, out, err = run_conv([tmp_path / "web", "--out", tmp_path / "out"], capsys)
    assert rc == 0, err
    for side in ("BID", "ASK"):
        assert (f"{side}: flats dropped (volume exactly 0, as dukascopy-node's default ignoreFlats): {n_flats} "
                f"(open=high=low=close: {n_flats}; prices moving: 0)") in out


def test_moving_bar_with_volume_0_is_dropped_with_warning(tmp_path, capsys):
    bid, ask = SMALL
    web = tmp_path / "web"
    web.mkdir()
    wb, wa = website_rows(bid, "bid", {}), website_rows(ask, "ask", {})
    wb[100] = wb[100][:5] + ("0",)
    wa[100] = wa[100][:5] + ("0",)
    end = wb[-1][0] + 900
    write_website(web, "bid", wb, {}, wb[0][0], end)
    write_website(web, "ask", wa, {}, wa[0][0], end)
    rc, out, err = run_conv([web, "--out", tmp_path / "out"], capsys)
    assert rc == 0, err
    assert "prices moving: 1)" in out and "WARNING: BID: 1 bar(s) with volume 0 have moving prices" in out
    assert len(read_converted(tmp_path / "out" / "xauusd_m15_bid.csv")) == len(bid) - 1


def test_overlap_volume_only_difference_is_a_duplicate(tmp_path, capsys):
    bid, ask = SMALL
    web = tmp_path / "web"
    web.mkdir()
    for side, rows in (("bid", bid), ("ask", ask)):
        w = website_rows(rows, side, {})
        mid = w[len(w) // 2][0]
        n = sum(1 for r in w if mid <= r[0] < mid + 3600)
        write_website(web, side, w, {}, w[0][0], mid + 3600)
        w2 = [r[:5] + (r[5] + "1",) if r[0] < mid + 3600 else r for r in w]   # other non-zero volume
        write_website(web, side, w2, {}, mid, w[-1][0] + 900)
    rc, out, err = run_conv([web, "--out", tmp_path / "out"], capsys)
    assert rc == 0, err
    assert n >= 1
    assert f"BID: {n} exact duplicate row(s) dropped (overlapping parts or files); {n} of them differ only" in out
    assert_same_bars(tmp_path / "out" / "xauusd_m15_bid.csv", bid)


def test_one_side_only(tmp_path, capsys):
    bid, ask = SMALL
    emulate(tmp_path / "web", bid, ask, {})
    for p in (tmp_path / "web").glob("*ASK*"):
        p.unlink()
    rc, out, err = run_conv([tmp_path / "web", "--out", tmp_path / "out"], capsys)
    assert rc == 0, err
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["xauusd_m15_bid.csv"]
    assert "NOTE: only BID files were given" in out


def test_bid_ask_flags_with_any_names(tmp_path, capsys):
    bid, ask = SMALL
    web = tmp_path / "web"
    web.mkdir()
    b = write_website(web, "bid", website_rows(bid, "bid", {}), {}, 0, 2 ** 40, name="gold_first.csv")
    a = write_website(web, "ask", website_rows(ask, "ask", {}), {}, 0, 2 ** 40, name="gold_second.csv")
    rc, out, err = run_conv(["--bid", b, "--ask", a, "--out", tmp_path / "out"], capsys)
    assert rc == 0, err
    assert_same_bars(tmp_path / "out" / "xauusd_m15_bid.csv", bid)


def test_check_only_writes_nothing(tmp_path, capsys):
    bid, ask = SMALL
    emulate(tmp_path / "web", bid, ask, {})
    rc, out, err = run_conv([tmp_path / "web", "--check-only"], capsys)
    assert rc == 0, err
    assert "CHECK ONLY: all checks passed; nothing was written" in out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["web"]


# ----------------------------------------------------------------------------------------------------
# the holdout lock

LOCK_ROWS = make_node_rows("2025-09-24", 7, seed=9, price=3700.0, all_week=True)


def test_lock_cuts_rows_and_prints_only_the_count(tmp_path, capsys):
    bid, ask = LOCK_ROWS
    web = tmp_path / "web"
    web.mkdir()
    cut_prices = set()
    for side, rows in (("bid", bid), ("ask", ask)):
        w = website_rows(rows, side, {})
        bad = []
        for r in w:
            if r[0] >= conv.LOCK_S:
                cut_prices.update(r[1:5])
                r = (r[0], "not-a-price", r[2], r[3], r[4], r[5])   # never parsed: cut first
            bad.append(r)
        write_website(web, side, bad, {}, 0, 2 ** 40, name=f"XAUUSD_lock_{side.upper()}.csv")
    n_cut = sum(1 for r in bid if r[0] // 1000 >= conv.LOCK_S)
    assert n_cut > 0
    rc, out, err = run_conv([web, "--out", tmp_path / "out"], capsys)
    assert rc == 0, err
    assert f"BID: holdout lock: {n_cut} row(s) opening at or after 2025-09-28 00:00 UTC were cut" in out
    text = out + err
    for day in ("2025-09-28 00:15", "2025-09-29", "2025-09-30", "28.09.2025", "29.09.2025"):
        assert day not in text
    printed_numbers = set(re.findall(r"\d+\.\d+", text))
    assert not (printed_numbers & cut_prices)
    for side in ("bid", "ask"):
        got = read_converted(tmp_path / "out" / f"xauusd_m15_{side}.csv")
        assert max(int(g[0]) for g in got) // 1000 < conv.LOCK_S
        assert len(got) == sum(1 for r in bid if r[0] // 1000 < conv.LOCK_S)


def test_unreadable_time_after_the_lock_is_not_shown(tmp_path, capsys):
    bid, ask = LOCK_ROWS
    web = tmp_path / "web"
    web.mkdir()
    for side, rows in (("bid", bid), ("ask", ask)):
        p = write_website(web, side, website_rows(rows, side, {}), {}, 0, 2 ** 40, name=f"x_{side.upper()}.csv")
        p.write_text(p.read_text().replace("29.09.2025 10:00:00.000", "29.09.2025 10h00"))
    rc, out, err = run_conv([web, "--out", tmp_path / "out"], capsys)
    assert rc == 3
    assert "value not shown" in err and "29.09.2025" not in err


# ----------------------------------------------------------------------------------------------------
# refusals: data problems (exit 3, nothing written)

def _web_pair(tmp_path, v=None, bid_rows=None, ask_rows=None, edit=None):
    bid, ask = SMALL if bid_rows is None else (bid_rows, ask_rows)
    v = v or {}
    web = tmp_path / "web"
    web.mkdir(exist_ok=True)
    paths = []
    for side, rows in (("bid", bid), ("ask", ask)):
        w = website_rows(rows, side, v)
        if edit:
            w = edit(side, w)
        paths.append(write_website(web, side, w, v, 0, 2 ** 40))
    return web, paths


def _refused(tmp_path, capsys, web, needle, code=3):
    rc, out, err = run_conv([web, "--out", tmp_path / "out"], capsys)
    assert rc == code, (out, err)
    assert needle in err, err
    assert not (tmp_path / "out").exists()
    return out, err


def test_local_time_without_offset_is_refused(tmp_path, capsys):
    web, _ = _web_pair(tmp_path, {"header": "Local time"})
    _refused(tmp_path, capsys, web, "cannot be placed in UTC")


@pytest.mark.parametrize("header", ["Time", "Time (EET)", "Date", "Time (GMT+2)", "EET time"])
def test_zone_unknown_without_offset_is_refused(header, tmp_path, capsys):
    web, _ = _web_pair(tmp_path, {"header": header})
    _refused(tmp_path, capsys, web, "cannot be placed in UTC")


def test_gmt_header_with_nonzero_offset_is_refused(tmp_path, capsys):
    web, _ = _web_pair(tmp_path, {"tz": 8 * 3600})
    _refused(tmp_path, capsys, web, "contradicts itself")


def test_mixed_offsets_and_none_in_local_file_is_refused(tmp_path, capsys):
    web, paths = _web_pair(tmp_path, {"header": "Local time", "tz": 8 * 3600})
    p = paths[0]
    lines = p.read_text().splitlines()
    lines[5] = lines[5].replace(" GMT+0800", "")
    p.write_text("\n".join(lines) + "\n")
    _refused(tmp_path, capsys, web, "line 6: the time column is 'Local time' and the time has no GMT offset")


def test_excel_slash_dates_are_refused(tmp_path, capsys):
    web, paths = _web_pair(tmp_path)
    p = paths[0]
    lines = p.read_text().splitlines()
    d, rest = lines[1].split(" ", 1)
    dd, mm, yy = d.split(".")
    lines[1] = f"{int(yy)}/{int(mm)}/{int(dd)} 0:00," + lines[1].split(",", 1)[1]
    p.write_text("\n".join(lines) + "\n")
    _refused(tmp_path, capsys, web, "Excel")


def test_excel_e_plus_volume_is_refused(tmp_path, capsys):
    def edit(side, w):
        w[3] = w[3][:5] + ("1.13E+06",)
        return w
    web, _ = _web_pair(tmp_path, edit=edit)
    _refused(tmp_path, capsys, web, "Excel")


def test_price_in_scientific_notation_is_refused(tmp_path, capsys):
    def edit(side, w):
        w[3] = (w[3][0], "1.2005E3") + w[3][2:]
        return w
    web, _ = _web_pair(tmp_path, edit=edit)
    _refused(tmp_path, capsys, web, "scientific notation")


@pytest.mark.parametrize("step,words", [(60, "1 minute"), (3600, "1 hour"), (300, "5 minutes")])
def test_not_15_minute_bars_are_refused(step, words, tmp_path, capsys):
    bid, ask = SMALL
    base = bid[0][0] // 1000

    def respace(rows):
        return [(base * 1000 + i * step * 1000,) + r[1:] for i, r in enumerate(rows)]
    web, _ = _web_pair(tmp_path, bid_rows=respace(bid), ask_rows=respace(ask))
    _refused(tmp_path, capsys, web, f"mostly {words} apart, not 15 minutes")


def test_bars_off_the_15_minute_grid_are_refused(tmp_path, capsys):
    bid, ask = SMALL
    shift = lambda rows: [(r[0] + 300_000,) + r[1:] for r in rows]   # noqa: E731
    web, _ = _web_pair(tmp_path, bid_rows=shift(bid), ask_rows=shift(ask))
    _refused(tmp_path, capsys, web, "do not open on a 15-minute boundary")


def test_seconds_in_times_are_refused(tmp_path, capsys):
    web, paths = _web_pair(tmp_path)
    p = paths[0]
    p.write_text(p.read_text().replace(":00.000,", ":07.000,", 1))
    _refused(tmp_path, capsys, web, "not on a whole minute")


def test_not_xauusd_prices_are_refused(tmp_path, capsys):
    bid, ask = make_node_rows("2015-01-05", 10, price=1.1)
    fix = lambda rows: [(r[0],) + tuple(conv.canon_plain(f"{abs(Decimal(x)) + Decimal('0.5'):.5f}") for x in r[1:])
                        for r in rows]   # noqa: E731
    web, _ = _web_pair(tmp_path, bid_rows=fix(bid), ask_rows=fix(ask))
    _refused(tmp_path, capsys, web, "do not look like XAU/USD")


def test_other_instrument_name_is_refused(tmp_path, capsys):
    bid, ask = SMALL
    web = tmp_path / "web"
    web.mkdir()
    write_website(web, "bid", website_rows(bid, "bid", {}), {}, 0, 2 ** 40,
                  name="EURUSD_Candlestick_15_M_BID_01.01.2015-31.12.2015.csv")
    write_website(web, "ask", website_rows(ask, "ask", {}), {}, 0, 2 ** 40)
    _refused(tmp_path, capsys, web, "the name says EURUSD")


def test_swapped_sides_are_refused(tmp_path, capsys):
    bid, ask = SMALL
    web, _ = _web_pair(tmp_path, bid_rows=ask, ask_rows=bid)
    _refused(tmp_path, capsys, web, "look swapped")


def test_same_side_twice_is_refused(tmp_path, capsys):
    bid, _ = SMALL
    web, _ = _web_pair(tmp_path, bid_rows=bid, ask_rows=bid)
    _refused(tmp_path, capsys, web, "same side was downloaded twice")


def test_overlapping_parts_that_disagree_are_refused(tmp_path, capsys):
    bid, ask = SMALL
    web = tmp_path / "web"
    web.mkdir()
    for side, rows in (("bid", bid), ("ask", ask)):
        w = website_rows(rows, side, {})
        mid = w[len(w) // 2][0]
        write_website(web, side, w, {}, w[0][0], mid + 3600)
        shifted = [(r[0] + 3600,) + r[1:] for r in w]          # the second part is one hour off
        write_website(web, side, shifted, {}, mid, w[-1][0] + 900)
    _refused(tmp_path, capsys, web, "appear twice with different values")


def test_bid_ask_time_mismatch_is_refused_and_nothing_written(tmp_path, capsys):
    def edit(side, w):
        if side == "bid":                       # a bid-only flat: volume 0 on one side
            w[50] = (w[50][0], w[50][4], w[50][4], w[50][4], w[50][4], "0")
        return w
    web, _ = _web_pair(tmp_path, edit=edit)
    out, err = _refused(tmp_path, capsys, web, "bid and ask do NOT hold the same bar times")
    assert "1 only in ask" in err and "0 only in bid" in err


def test_ask_open_below_bid_open_is_refused(tmp_path, capsys):
    def edit(side, w):
        if side == "ask":
            r = w[40]
            w[40] = (r[0], "100.5", r[2], "100.5", r[4], r[5])
        return w
    web, _ = _web_pair(tmp_path, edit=edit)
    _refused(tmp_path, capsys, web, "ask open below the bid open")


def test_high_below_close_is_refused(tmp_path, capsys):
    def edit(side, w):
        r = w[30]
        w[30] = (r[0], r[1], r[3], r[3], r[4], r[5]) if side == "bid" else r
        return w
    bid, ask = SMALL
    web, _ = _web_pair(tmp_path, edit=edit)
    _refused(tmp_path, capsys, web, "high below open or close")


def test_missing_volume_column_is_refused(tmp_path, capsys):
    web, paths = _web_pair(tmp_path)
    for p in paths:
        lines = [ln.rsplit(",", 1)[0] for ln in p.read_text().splitlines()]
        p.write_text("\n".join(lines) + "\n")
    _refused(tmp_path, capsys, web, "Volume not found")


def test_tick_file_is_refused(tmp_path, capsys):
    web = tmp_path / "web"
    web.mkdir()
    (web / "XAUUSD_Ticks_BID.csv").write_text("Gmt time,Ask,Bid,AskVolume,BidVolume\n"
                                              "02.01.2015 00:00:01.123,1186.5,1186.2,1.1,2.2\n")
    _refused(tmp_path, capsys, web, "tick file")


def test_dukascopy_node_file_as_input_is_refused(tmp_path, capsys):
    web = tmp_path / "web"
    web.mkdir()
    (web / "xauusd_m15_bid.csv").write_text("timestamp,open,high,low,close,volume\n1420156800000,1,2,0.5,1.5,3\n")
    _refused(tmp_path, capsys, web, "already is a dukascopy-node style file")


def test_empty_file_is_refused(tmp_path, capsys):
    web = tmp_path / "web"
    web.mkdir()
    (web / "XAUUSD_BID.csv").write_bytes(b"")
    _refused(tmp_path, capsys, web, "the file is empty")


def test_cut_off_row_is_refused(tmp_path, capsys):
    web, paths = _web_pair(tmp_path)
    p = paths[1]
    p.write_text(p.read_text() + "05.01.2015 10:00:00.000,1200.1\n")
    _refused(tmp_path, capsys, web, "fields but the first line has 6")


# ----------------------------------------------------------------------------------------------------
# refusals: usage (exit 2)

def test_no_arguments_is_a_usage_error(tmp_path, capsys):
    rc, out, err = run_conv([], capsys)
    assert rc == 2 and "no input files" in err


def test_out_folder_with_files_is_refused(tmp_path, capsys):
    web, _ = _web_pair(tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "keep.txt").write_text("x")
    rc, out, err = run_conv([web, "--out", tmp_path / "out"], capsys)
    assert rc == 2 and "never overwrites" in err
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["keep.txt"]


def test_empty_existing_out_folder_is_used(tmp_path, capsys):
    web, _ = _web_pair(tmp_path)
    (tmp_path / "out").mkdir()
    rc, out, err = run_conv([web, "--out", tmp_path / "out"], capsys)
    assert rc == 0, err


@pytest.mark.parametrize("where", ["same", "inside"])
def test_out_in_input_folder_is_refused(where, tmp_path, capsys):
    web, _ = _web_pair(tmp_path)
    out_dir = web if where == "same" else web / "converted"
    rc, out, err = run_conv([web, "--out", out_dir], capsys)
    assert rc == 2 and "input folder" in err
    assert not (web / "converted").exists()


@pytest.mark.parametrize("which", ["input", "out", "member"])
def test_locked_paths_are_refused(which, tmp_path, capsys):
    web, _ = _web_pair(tmp_path)
    if which == "input":
        locked = tmp_path / "locked_holdout_web"
        web.rename(locked)
        args = [locked, "--out", tmp_path / "out"]
    elif which == "out":
        args = [web, "--out", tmp_path / "x.locked"]
    else:
        (web / "XAUUSD_BID_2025.csv.locked").write_text("never read")
        args = [web, "--out", tmp_path / "out"]
    rc, out, err = run_conv(args, capsys)
    assert rc == 2 and "locked" in err
    assert not (tmp_path / "out").exists() and not (tmp_path / "x.locked").exists()


def test_unfinished_download_is_refused(tmp_path, capsys):
    web, _ = _web_pair(tmp_path)
    (web / "Unconfirmed 123456.crdownload").write_bytes(b"x")
    rc, out, err = run_conv([web, "--out", tmp_path / "out"], capsys)
    assert rc == 2 and "unfinished download" in err


def test_name_without_side_is_refused(tmp_path, capsys):
    web, paths = _web_pair(tmp_path)
    paths[0].rename(web / "XAUUSD_Candlestick_15_M_2015.csv")
    rc, out, err = run_conv([web, "--out", tmp_path / "out"], capsys)
    assert rc == 2 and "cannot tell from the name" in err


def test_flag_contradicting_name_is_refused(tmp_path, capsys):
    web, paths = _web_pair(tmp_path)
    rc, out, err = run_conv(["--bid", paths[1], "--out", tmp_path / "out"], capsys)
    assert rc == 2 and "its name says ASK" in err


def test_script_exit_codes_in_a_subprocess(tmp_path):
    web, _ = _web_pair(tmp_path)
    env = dict(os.environ, PYTHONUTF8="1")
    ok = subprocess.run([sys.executable, str(SCRIPT), str(web), "--out", str(tmp_path / "out")],
                        capture_output=True, text=True, env=env)
    assert ok.returncode == 0, ok.stderr
    assert "DONE: exit code 0" in ok.stdout
    bad_args = subprocess.run([sys.executable, str(SCRIPT), "--nonsense"], capture_output=True, text=True, env=env)
    assert bad_args.returncode == 2 and bad_args.stderr.startswith("ERROR:")
    again = subprocess.run([sys.executable, str(SCRIPT), str(web), "--out", str(tmp_path / "out")],
                           capture_output=True, text=True, env=env)
    assert again.returncode == 2
    (web / "XAUUSD_ASK_extra.csv").write_text("Local time,Open,High,Low,Close,Volume\n"
                                              "05.01.2015 10:00:00.000,1200,1201,1199,1200.5,3\n")
    data = subprocess.run([sys.executable, str(SCRIPT), str(web), "--out", str(tmp_path / "out2")],
                          capture_output=True, text=True, env=env)
    assert data.returncode == 3 and "cannot be placed in UTC" in data.stderr
    assert not (tmp_path / "out2").exists()


# ----------------------------------------------------------------------------------------------------
# review fixes (version 1.1)

def _split_write(web: Path, side: str, wrows, cut: int, second=None, names=("a", "b")):
    """Two parts of one side: rows before `cut`, and rows from `cut` on (taken from `second` if given)."""
    end = 2 ** 40
    write_website(web, side, wrows, {}, 0, cut, name=f"XAUUSD_{names[0]}_{side.upper()}.csv")
    write_website(web, side, second if second is not None else wrows, {}, cut, end,
                  name=f"XAUUSD_{names[1]}_{side.upper()}.csv")


@pytest.mark.parametrize("step,words", [(3600, "1 hour"), (1800, "30 minutes"), (86400, "1 day")])
def test_one_part_with_another_bar_size_is_refused(step, words, tmp_path, capsys):
    # a 1-hour (or 30-minute, daily) part among 15-minute parts: each file is checked on its own
    bid, ask = SMALL
    web = tmp_path / "web"
    web.mkdir()
    cut = _t("2015-01-26")
    for side, rows in (("bid", bid), ("ask", ask)):
        w = website_rows(rows, side, {})
        coarse = [r for r in w if r[0] % step == 0]
        _split_write(web, side, w, cut, second=coarse)
    _refused(tmp_path, capsys, web, f"XAUUSD_b_BID.csv: the bars in this file are mostly {words} apart")


def test_file_name_with_another_candle_size_is_refused(tmp_path, capsys):
    bid, ask = SMALL
    web = tmp_path / "web"
    web.mkdir()
    write_website(web, "bid", website_rows(bid, "bid", {}), {}, 0, 2 ** 40,
                  name="XAUUSD_Candlestick_1_Hour_BID_01.01.2015-31.12.2015.csv")
    write_website(web, "ask", website_rows(ask, "ask", {}), {}, 0, 2 ** 40)
    _refused(tmp_path, capsys, web, "the name says 1 Hour candles, not 15 minutes")


@pytest.mark.parametrize("case,needle", [("ask_part_is_bid", "bid and ask prices are identical on"),
                                         ("both_parts_swapped", "ask close is below the bid close on")])
def test_one_month_of_the_wrong_side_is_refused(case, needle, tmp_path, capsys):
    # e.g. a year downloaded with the side left on BID and then named ASK: the whole-file checks pass
    bid, ask = SMALL
    web = tmp_path / "web"
    web.mkdir()
    cut = _t("2015-02-01")
    wb, wa = website_rows(bid, "bid", {}), website_rows(ask, "ask", {})
    if case == "ask_part_is_bid":
        _split_write(web, "bid", wb, cut)
        _split_write(web, "ask", wa, cut, second=wb)
    else:
        _split_write(web, "bid", wb, cut, second=wa)
        _split_write(web, "ask", wa, cut, second=wb)
    out, err = _refused(tmp_path, capsys, web, needle)
    assert "PAIR: in 2015-02 the " in err and "XAUUSD_b_BID.csv (BID) and XAUUSD_b_ASK.csv (ASK)" in err


def _three_winters():
    """January-February of 2015, 2016 and 2017, each starting on a Sunday 22:00 UTC (a week open)."""
    parts = [make_node_rows(start, 42, seed=k) for k, start in
             enumerate(("2015-01-04T22:00:00", "2016-01-03T22:00:00", "2017-01-01T22:00:00"))]
    return parts


@pytest.mark.parametrize("shift,needle", [
    (0, None),
    (3600, "week opens are at a different time of day than the other files' in the same months (e.g. 23:00 "
           "against 22:00 UTC)"),
    (-3600, "week opens are at a different time of day than the other files' in the same months (e.g. 21:00 "
            "against 22:00 UTC)"),
    (900, "week opens are off the hour or outside 21:00-23:00 UTC (22:15 x"),
    (3 * 3600, "week opens are off the hour or outside 21:00-23:00 UTC (01:00 x"),
])
def test_one_shifted_part_gets_a_warning(shift, needle, tmp_path, capsys):
    web = tmp_path / "web"
    web.mkdir()
    for k, (bid, ask) in enumerate(_three_winters()):
        for side, rows in (("bid", bid), ("ask", ask)):
            if k == 1:
                rows = [(r[0] + shift * 1000,) + r[1:] for r in rows]
            write_website(web, side, website_rows(rows, side, {}), {}, 0, 2 ** 40,
                          name=f"XAUUSD_Candlestick_15_M_{side.upper()}_{2015 + k}.csv")
    rc, out, err = run_conv([web, "--check-only"], capsys)
    assert rc == 0, err
    week_warnings = [ln for ln in out.splitlines() if ln.startswith("WARNING") and "week opens" in ln]
    if needle is None:
        assert week_warnings == []
    else:
        assert len(week_warnings) == 2, week_warnings          # BID and ASK, the 2016 part only
        for side in ("BID", "ASK"):
            assert any(ln.startswith(f"WARNING: {side}: XAUUSD_Candlestick_15_M_{side}_2016.csv: ") and needle in ln
                       for ln in week_warnings), week_warnings


def test_week_open_check_on_a_real_shaped_schedule():
    # [inferred] real schedule: Sunday 18:00 New York (22:00 UTC in US summer time, 23:00 in winter)
    def ny18(d):
        dst = dt.date(d.year, 3, 8 + (6 - dt.date(d.year, 3, 8).weekday()) % 7) <= d < \
            dt.date(d.year, 11, 1 + (6 - dt.date(d.year, 11, 1).weekday()) % 7)
        return _t(d.isoformat()) + (22 if dst else 23) * 3600
    opens, d = [], dt.date(2015, 1, 4)
    while d <= dt.date(2025, 9, 21):
        opens.append(ny18(d))
        d += dt.timedelta(days=7)
    for part_of in (lambda ts: dt.datetime.fromtimestamp(ts, UTC).year,
                    lambda ts: (dt.datetime.fromtimestamp(ts, UTC).year, (dt.datetime.fromtimestamp(ts, UTC).month - 1) // 3)):
        keys = sorted({part_of(ts) for ts in opens})
        index = {k: i for i, k in enumerate(keys)}
        names = {i: str(k) for k, i in index.items()}
        base = [(ts, index[part_of(ts)]) for ts in opens]
        assert conv.week_open_warnings("BID", base, names) == []
        target = len(keys) // 2
        for shift in (3600, -3600):
            moved = [(ts + shift if i == target else ts, i) for ts, i in base]
            w = conv.week_open_warnings("BID", moved, names)
            assert len(w) == 1 and f"BID: {names[target]}: " in w[0], w


def _write_utf8(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


@pytest.mark.parametrize("digit", ["１", "١"])
def test_non_ascii_digit_in_a_price_is_refused_and_nothing_written(digit, tmp_path, capsys):
    web, paths = _web_pair(tmp_path)
    p = paths[1]                                  # the ASK file: the BID file would be written first
    lines = p.read_text().splitlines()
    cells = lines[100].split(",")
    cells[4] = digit + cells[4][1:]
    lines[100] = ",".join(cells)
    _write_utf8(p, "\n".join(lines) + "\n")
    _refused(tmp_path, capsys, web, "is not a plain positive number")


def test_any_failure_while_writing_leaves_no_file(tmp_path, capsys, monkeypatch):
    web, _ = _web_pair(tmp_path)
    real_rename = conv.os.rename
    calls = []

    def rename(a, b):
        calls.append(b)
        if len(calls) == 2:
            raise RuntimeError("simulated failure while writing the second file")
        return real_rename(a, b)
    monkeypatch.setattr(conv.os, "rename", rename)
    with pytest.raises(RuntimeError):
        conv.main([str(web), "--out", str(tmp_path / "out")])
    assert list((tmp_path / "out").iterdir()) == []


def test_render_failure_writes_nothing(tmp_path, capsys, monkeypatch):
    web, _ = _web_pair(tmp_path)
    real_render = conv.render
    n = []

    def render(rows):
        n.append(1)
        if len(n) == 2:
            raise UnicodeEncodeError("ascii", "x", 0, 1, "simulated")
        return real_render(rows)
    monkeypatch.setattr(conv, "render", render)
    with pytest.raises(UnicodeEncodeError):
        conv.main([str(web), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("text,visible", [
    ("01.01.2015 10h00", True), ("2024-12-31 23:45", True), ("28.09.25 00:15", False), ("9/28/25 0:15", False),
    ("29.09.2025 10h00", False), ("1759017600000", False), ("01.01.2015 2026", False), ("", False)])
def test_shown_prints_a_time_only_with_a_year_before_2025(text, visible):
    assert (conv.shown(text) == repr(text)) is visible


@pytest.mark.parametrize("tail", ["9/28/25 0:15,7777.1,7777.2,7777.0,7777.1,1.5",
                                  "28.09.25 00:15,7777.1,7777.2,7777.0,7777.1,1.5"])
def test_unreadable_post_lock_time_with_two_digit_year_is_not_shown(tail, tmp_path, capsys):
    bid, ask = LOCK_ROWS
    web = tmp_path / "web"
    web.mkdir()
    for side, rows in (("bid", bid), ("ask", ask)):
        p = write_website(web, side, website_rows(rows, side, {}), {}, 0, 2 ** 40, name=f"x_{side.upper()}.csv")
        if side == "bid":
            p.write_text(p.read_text() + tail + "\n")
    out, err = _refused(tmp_path, capsys, web, "value not shown")
    assert tail.split(",")[0] not in out + err and "7777" not in out + err and " line " in err


@pytest.mark.parametrize("label", ["Volume (units)", "Tick volume", "Volumes", "VOLUME"])
def test_volume_column_with_a_unit_or_other_spelling_is_read(label, tmp_path, capsys):
    web, paths = _web_pair(tmp_path)
    for p in paths:
        lines = p.read_text().splitlines()
        lines[0] = lines[0].replace("Volume", label)
        p.write_text("\n".join(lines) + "\n")
    rc, out, err = run_conv([web, "--out", tmp_path / "out"], capsys)
    assert rc == 0, err
    assert_same_bars(tmp_path / "out" / "xauusd_m15_bid.csv", SMALL[0])


def test_overlapping_flats_with_different_prices_are_duplicates(tmp_path, capsys):
    # the overlap day of two parts is a closed day: every row is a flat, filled differently by each part
    bid, ask = SMALL
    web = tmp_path / "web"
    web.mkdir()
    a, b = _t("2015-01-10"), _t("2015-01-10T12:00:00")          # a Saturday: flats only
    for side, rows in (("bid", bid), ("ask", ask)):
        w = website_rows(rows, side, {"flats": True})
        assert all(r[5] == "0" for r in w if a <= r[0] < b)
        other = [(r[0],) + tuple(conv.canon_plain(f"{Decimal(x) + Decimal('0.5')}") for x in r[1:5]) + (r[5],)
                 if a <= r[0] < b else r for r in w]
        write_website(web, side, w, {}, 0, b, name=f"XAUUSD_p1_{side.upper()}.csv")
        write_website(web, side, other, {}, a, 2 ** 40, name=f"XAUUSD_p2_{side.upper()}.csv")
    rc, out, err = run_conv([web, "--out", tmp_path / "out"], capsys)
    assert rc == 0, err
    assert "48 of them are volume-0 rows (flats) whose prices differ between the files" in out
    assert_same_bars(tmp_path / "out" / "xauusd_m15_bid.csv", bid)
    assert_same_bars(tmp_path / "out" / "xauusd_m15_ask.csv", ask)


@pytest.mark.parametrize("gap_end,warn", [("2015-04-05T22:00:00", False), ("2015-04-07T22:00:00", True)])
def test_holiday_weekend_gap_is_listed_and_only_a_longer_gap_warns(gap_end, warn, tmp_path, capsys):
    # Good Friday 2015: last bar Thursday 20:45 UTC, next bar Sunday 22:00 UTC (73 h 15 min)
    bid, ask = make_node_rows("2015-03-23", 21, seed=4)
    a, b = _t("2015-04-02T21:00:00") * 1000, _t(gap_end) * 1000
    keep = lambda rows: [r for r in rows if not (a <= r[0] < b)]   # noqa: E731
    web, _ = _web_pair(tmp_path, bid_rows=keep(bid), ask_rows=keep(ask))
    rc, out, err = run_conv([web, "--check-only"], capsys)
    assert rc == 0, err
    assert f"BID: gaps longer than 3 days (holiday weekends are about 3 days): 2015-04-02 20:45 UTC to " in out
    assert ("WARNING: BID: 1 gap(s) longer than 4 days" in out) is warn


def test_link_into_a_locked_path_is_refused(tmp_path, capsys):
    web, _ = _web_pair(tmp_path)
    target = tmp_path / "locked_holdout" / "XAUUSD_2025_BID.csv"     # never created, never opened
    try:
        os.symlink(target, web / "XAUUSD_extra_BID.csv")
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available here")
    rc, out, err = run_conv([web, "--out", tmp_path / "out"], capsys)
    assert rc == 2 and "locked" in err
    assert not (tmp_path / "out").exists() and not (tmp_path / "locked_holdout").exists()


def test_post_lock_rows_are_cut_before_any_other_check(tmp_path, capsys):
    bid, ask = LOCK_ROWS
    web = tmp_path / "web"
    web.mkdir()
    for side, rows in (("bid", bid), ("ask", ask)):
        p = write_website(web, side, website_rows(rows, side, {}), {}, 0, 2 ** 40, name=f"x_{side.upper()}.csv")
        if side == "bid":
            p.write_text(p.read_text() + "03.10.2025 00:00:00.000,1.2\n"                        # cut short
                         + "03.10.2025 00:15:00.000 GMT+0300,3700,3701,3699,3700.5,2\n")      # contradicts header
    n_cut = sum(1 for r in bid if r[0] // 1000 >= conv.LOCK_S)
    rc, out, err = run_conv([web, "--check-only"], capsys)
    assert rc == 0, err
    assert f"BID: holdout lock: {n_cut + 2} row(s) opening at or after" in out
    assert f"ASK: holdout lock: {n_cut} row(s) opening at or after" in out


def test_out_whose_parent_is_missing_is_refused(tmp_path, capsys):
    web, _ = _web_pair(tmp_path)
    rc, out, err = run_conv([web, "--out", tmp_path / "typo" / "deeper" / "out"], capsys)
    assert rc == 2 and "the folder above --out does not exist" in err
    assert not (tmp_path / "typo").exists()


# ----------------------------------------------------------------------------------------------------
# end to end: the dry-run synthetic files -> emulated website export -> converter -> the same bars

E2E_VARIANTS = {
    "gmt_yearly_parts_flats": {"parts": "yearly", "flats": True},
    "gmt_yearly_overlap_dup_crlf_bom": {"parts": "yearly_overlap", "dup_file": True, "flats": True,
                                        "newline": "\r\n", "bom": True},
    "local_singapore_yearly_overlap": {"header": "Local time", "tz": 8 * 3600, "parts": "yearly_overlap"},
    "local_eet_semicolon_decimal_comma": {"header": "Local time", "tz": "eet", "parts": "yearly",
                                          "delim": ";", "decimal_comma": True, "flats": True,
                                          "flat_volume": "0,00"},
    "gmt_fixed_3_decimals_units": {"decimals": 3, "volume": "units2", "parts": "yearly"},
}


def _make_synthetic(out: Path) -> None:
    """The PC dry run's made-up data (PC_DRYRUN_2026-10-09.md step 2): 74,444 M15 bars, 2015-2017."""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(7)
    t = pd.date_range("2015-01-01", "2017-12-31 23:45", freq="15min", tz="UTC")
    dow, h = t.dayofweek, t.hour
    t = t[~((dow == 5) | ((dow == 4) & (h >= 21)) | ((dow == 6) & (h < 22)))]
    n = len(t)
    r = rng.standard_t(4, size=n) * 1.5 + 0.002
    c = 1200 + np.cumsum(r)
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) + np.abs(rng.normal(0, 1.0, n))
    lo = np.minimum(o, c) - np.abs(rng.normal(0, 1.0, n))
    sp = 0.25 + np.abs(rng.normal(0, 0.08, n))
    ms = ((t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")).astype("int64")
    bid = pd.DataFrame({"timestamp": ms, "open": o.round(3), "high": hi.round(3), "low": lo.round(3),
                        "close": c.round(3)})
    ask = bid.copy()
    for k in ("open", "high", "low", "close"):
        ask[k] = (bid[k] + sp).round(3)
    bid.to_csv(out / "SYNTHETIC_xauusd_m15_bid.csv", index=False)
    ask.to_csv(out / "SYNTHETIC_xauusd_m15_ask.csv", index=False)


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    if HAVE_SYNTH:
        return SYNTH_DIR
    out = tmp_path_factory.mktemp("synthetic")
    _make_synthetic(out)
    return out


@pytest.fixture(scope="module")
def synthetic(synth_dir):
    return (read_node(synth_dir / "SYNTHETIC_xauusd_m15_bid.csv"),
            read_node(synth_dir / "SYNTHETIC_xauusd_m15_ask.csv"))


@pytest.mark.parametrize("name", sorted(E2E_VARIANTS))
def test_e2e_synthetic_round_trip(name, synthetic, tmp_path, capsys):
    bid, ask = synthetic
    assert len(bid) == len(ask) == 74444
    emulate(tmp_path / "web", bid, ask, E2E_VARIANTS[name])
    rc, out, err = run_conv([tmp_path / "web", "--out", tmp_path / "out"], capsys)
    assert rc == 0, err
    assert_same_bars(tmp_path / "out" / "xauusd_m15_bid.csv", bid)
    assert_same_bars(tmp_path / "out" / "xauusd_m15_ask.csv", ask)
    assert "PAIR: bid and ask hold exactly the same 74444 bar times" in out
    assert [ln for ln in out.splitlines() if ln.startswith("WARNING") and "data ends" not in ln] == []


def _triggers_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.startswith("Triggers:")]
    assert len(lines) == 1, text
    return lines[0]


@pytest.mark.skipif(not (PROPKIT_REPO / "propkit").is_dir(), reason="no propkit package at PROPKIT_REPO")
def test_e2e_propkit_signals_same_triggers(synthetic, synth_dir, tmp_path, capsys):
    bid, ask = synthetic
    emulate(tmp_path / "web", bid, ask, E2E_VARIANTS["gmt_yearly_overlap_dup_crlf_bom"])
    rc, out, err = run_conv([tmp_path / "web", "--out", tmp_path / "conv"], capsys)
    assert rc == 0, err
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    results = {}
    for label, b, a in (("original", synth_dir / "SYNTHETIC_xauusd_m15_bid.csv",
                         synth_dir / "SYNTHETIC_xauusd_m15_ask.csv"),
                        ("converted", tmp_path / "conv" / "xauusd_m15_bid.csv",
                         tmp_path / "conv" / "xauusd_m15_ask.csv")):
        r = subprocess.run([sys.executable, "-m", "propkit", "zeno-v1", "signals", "--m15-bid", str(b),
                            "--m15-ask", str(a), "--news", NEWS_REL, "--out", str(tmp_path / f"sig_{label}")],
                           cwd=PROPKIT_REPO, capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout + r.stderr
        results[label] = r.stdout
    assert _triggers_line(results["converted"]) == _triggers_line(results["original"])
    assert _triggers_line(results["converted"]).startswith("Triggers: 2304, eligible 205")
    for name in ("signals.csv", "decisions.csv", "g0_sample.csv"):
        assert (tmp_path / "sig_converted" / name).read_bytes() == (tmp_path / "sig_original" / name).read_bytes()
