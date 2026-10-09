"""`python -m propkit zeno-v1 g0-charts` (propkit.zeno_g0_charts, propkit.cli): the G0 charts page.

End to end through the command line on synthetic M15 bid/ask bars (NOT market data): one self-contained page
with 20 sections, the sample and its folder untouched; no leak (no bar time after a row's entry is drawn or
written, and garbage in place of every price after a row's entry-bar open gives the same section byte for
byte); the drawn values equal the sample's, signals.csv's and the engine's; known answers on the canonical
hand-built long and short; the long/short mirror; the refusals; the help and the docs.
Research only: nothing places, prepares or simulates orders.
"""
from __future__ import annotations

import contextlib
import hashlib
import html
import io
import math
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from propkit import calendar as cal
from propkit import cli
from propkit import zeno_g0_charts as g
from propkit import zeno_report as zr
from propkit import zeno_v1 as z
from tests.unit.zeno_v1_testkit import EVAL_10_X1, NEWS_CSV, Scenario, utc

HEADER = "RESEARCH ONLY - not trading advice"
PRICE_COLS = ["open", "high", "low", "close"]
VOID_TAGS = {"meta", "br", "hr", "img", "input", "link", "col", "area", "base", "wbr", "source"}
SECTION_RE = re.compile(r'<section class="row (?:long|short)" id="row-([^"]+)".*?</section>', re.S)


# ---------------------------------------------------------------------------------------
# helpers

def call(args) -> tuple[int, str, str]:
    """cli.main with stdout and stderr captured (both must be ASCII)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main([str(a) for a in args])
    assert out.getvalue().isascii() and err.getvalue().isascii()
    return code, out.getvalue(), err.getvalue()


def console(path) -> str:
    """A path as the command prints it: ASCII, anything else as a backslash escape (cli._say, for a cp936
    console)."""
    return str(path).encode("ascii", "backslashreplace").decode("ascii")


def in_page(path) -> str:
    """A path as the page holds it: HTML-escaped, then ASCII with character references."""
    return html.escape(str(path), quote=True).encode("ascii", "xmlcharrefreplace").decode("ascii")


def write_pair(folder: Path, bid: pd.DataFrame, ask: pd.DataFrame, stem: str = "SYNTH") -> tuple[Path, Path]:
    """Bid and ask CSV files (time, open, high, low, close)."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = (folder / f"{stem}_M15_bid.csv", folder / f"{stem}_M15_ask.csv")
    bid.to_csv(paths[0], index=False)
    ask.to_csv(paths[1], index=False)
    return paths


def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The bid and ask tables (time, open, high, low, close) of a zeno frame."""
    out = []
    for side in ("bid", "ask"):
        df = frame[["time"] + [f"{side}_{c}" for c in PRICE_COLS]].copy()
        df.columns = ["time"] + PRICE_COLS
        out.append(df)
    return out[0], out[1]


def garble(bid: pd.DataFrame, ask: pd.DataFrame, entry_t: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copies of bid and ask in which every price after the open of the bar opening at entry_t is garbage:
    that bar keeps its bid and ask open and gets a new high, low and close; every later bar is new (still
    valid bars, ask >= bid). The bar times are unchanged."""
    rng = np.random.default_rng(seed)
    b, a = bid.copy(), ask.copy()
    t = b["time"].to_numpy()
    later, at = t > entry_t, t == entry_t
    assert at.sum() == 1 and later.any()
    n = int(later.sum())
    o, c = rng.uniform(600.0, 5000.0, n), rng.uniform(600.0, 5000.0, n)
    h, lo = np.maximum(o, c) + rng.uniform(0.0, 40.0, n), np.minimum(o, c) - rng.uniform(0.0, 40.0, n)
    sp = rng.uniform(0.0, 3.0, n)[:, None]
    b.loc[later, PRICE_COLS] = np.c_[o, h, lo, c]
    a.loc[later, PRICE_COLS] = np.c_[o, h, lo, c] + sp
    k = int(np.flatnonzero(at)[0])
    ob, oa = float(b.at[k, "open"]), float(a.at[k, "open"])
    c1 = ob + rng.uniform(-30.0, 30.0)
    s1 = float(rng.uniform(0.0, 3.0))
    b.loc[k, ["high", "low", "close"]] = [max(ob, c1) + 7.0, min(ob, c1) - 9.0, c1]
    a.loc[k, ["high", "low", "close"]] = [max(oa, c1 + s1) + 5.0, min(oa, c1 + s1) - 4.0, c1 + s1]
    return b, a


def sections(page: str) -> dict[str, str]:
    """The per-row sections of a page, by sample_no."""
    found = {m.group(1): m.group(0) for m in SECTION_RE.finditer(page)}
    assert len(found) == page.count("<section ")
    return found


class _Parse(HTMLParser):
    """Every element that carries data-role (tag, attributes) and whether the tags are balanced."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.marks: list[tuple[str, dict]] = []
        self.stack: list[str] = []
        self.bad: list[str] = []

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if "data-role" in d:
            self.marks.append((tag, d))
        if tag not in VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        d = dict(attrs)
        if "data-role" in d:
            self.marks.append((tag, d))

    def handle_endtag(self, tag):
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.bad.append(tag)


def marks(fragment: str) -> dict[str, list[dict]]:
    """data-role -> the attributes of every element with that role, in page order."""
    p = _Parse()
    p.feed(fragment)
    p.close()
    out: dict[str, list[dict]] = {}
    for _, d in p.marks:
        out.setdefault(d["data-role"], []).append(d)
    return out


def one(m: dict[str, list[dict]], role: str) -> dict:
    assert len(m.get(role, [])) == 1, (role, len(m.get(role, [])))
    return m[role][0]


def t_of(text: str) -> int:
    """UTC epoch seconds of stage 1's 'YYYY-MM-DD HH:MM:SS UTC' text."""
    return int(pd.Timestamp(str(text).removesuffix("UTC").strip(), tz="UTC").timestamp())


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def folder_state(folder: Path) -> dict[str, str]:
    return {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}


# ---------------------------------------------------------------------------------------
# the end-to-end fixture: stage 1, then the charts

@pytest.fixture(scope="module")
def g0(tmp_path_factory):
    """~6 months of synthetic M15 bid/ask bars from 2015-01-01 (spread 0.05 USD/oz), `zeno-v1 signals` with the
    packaged news calendar (20 eligible signals sampled), then `zeno-v1 g0-charts` with its defaults."""
    d = tmp_path_factory.mktemp("zeno_g0_charts")
    frame = z.synthetic_m15_bidask(n_bars=16_000, seed=3, spread=0.05)
    bid_t, ask_t = split(frame)
    bid, ask = write_pair(d / "in", bid_t, ask_t)
    sig = d / "g0"
    code, _, err = call(["zeno-v1", "signals", "--m15-bid", bid, "--m15-ask", ask, "--news", g.PACKAGED_NEWS_CSV,
                         "--out", sig])
    assert code == 0, err
    sample = sig / "g0_sample.csv"
    before = folder_state(sig)
    code, out, err = call(["zeno-v1", "g0-charts", "--m15-bid", bid, "--m15-ask", ask, "--sample", sample])
    page_path = sig / g.G0_CHARTS_FILE
    return {"dir": d, "bid": bid, "ask": ask, "bid_t": bid_t, "ask_t": ask_t, "frame": frame, "sig": sig,
            "sample": sample, "before": before, "code": code, "out": out, "err": err, "page_path": page_path,
            "page": page_path.read_text(encoding="ascii") if page_path.is_file() else "",
            "df": pd.read_csv(sample, dtype=str, keep_default_na=False),
            "signals": pd.read_csv(sig / "signals.csv").set_index("signal_no")}


@pytest.fixture(scope="module")
def engine(g0):
    """The engine's own view of the same files: prepare() with the packaged calendar."""
    frame = z.load_m15_bidask(g0["bid"], g0["ask"])
    news = z.read_news_csv(g.PACKAGED_NEWS_CSV)
    return {"prep": z.prepare(frame, news), "news": news}


# ---------------------------------------------------------------------------------------
# end to end

def test_end_to_end_one_self_contained_page_and_the_sample_untouched(g0):
    assert g0["code"] == 0, g0["err"]
    assert g0["err"] == ""
    assert g0["out"].splitlines() == [
        HEADER, f"Wrote {console(g0['page_path'])}",
        "20 of 20 rows of g0_sample.csv drawn, each up to its entry and nothing after it; open the page in a "
        "browser and write y or n in agree_y_n of the sample."]
    # only the page was added beside the sample; every file there (the sample first) is byte-identical
    after = folder_state(g0["sig"])
    assert set(after) == set(g0["before"]) | {g.G0_CHARTS_FILE} and g.G0_CHARTS_FILE not in g0["before"]
    assert {k: after[k] for k in g0["before"]} == g0["before"]
    assert len(g0["df"]) == 20 and (g0["df"]["agree_y_n"] == "").all()
    page = g0["page"]
    assert page.isascii() and page.startswith("<!DOCTYPE html>")
    # self-contained: no network reference of any kind, no script, no form, nothing loaded
    low = page.lower()
    for bad in ("http", "//cdn", "<script", "<link", "<img", "<iframe", "<object", "<embed", "<form", "<input",
                "<button", "<textarea", "contenteditable", "url(", "@import", "@font-face", "src=", "xmlns"):
        assert bad not in low, bad
    assert set(re.findall(r'href="([^"]*)"', page)) == {f"#row-{k}" for k in range(1, 21)}
    p = _Parse()
    p.feed(page)
    p.close()
    assert p.bad == [] and p.stack == [], (p.bad, p.stack)
    # header: title, data files with sha256 and bar count, the sample's sha256, the chart cell, how to answer
    assert f"<title>{g.TITLE}</title>" in page and f"<h1>{g.TITLE}</h1>" in page
    assert g.TITLE == "G0 signal check - RESEARCH ONLY - no results"
    head = page.split("<section ", 1)[0]
    for f in (g0["bid"], g0["ask"], g0["sample"], g.PACKAGED_NEWS_CSV):
        assert sha(f) in head and in_page(f) in head
    assert f"{len(g0['frame'])} bars" in head and "20 rows" in head
    assert z.CHART_CELL.label in head and "evaluation/c10/S1/x1.5" in head
    for words in ("agree_y_n", "<b>y</b> or <b>n</b>", "<b>every</b> row", "at least 18 y of 20",
                  "each chart ends at the entry", "not a form"):
        assert words in head, words
    # 20 sections in sample order, one per row, each with its title times, the TradingView hint, the chart,
    # the 1h panel, the checklist with rule numbers and units, and an empty answer box
    secs = sections(page)
    assert list(secs) == [str(k) for k in range(1, 21)]
    for _, row in g0["df"].iterrows():
        s = secs[row["sample_no"]]
        assert f'data-side="{row["side"]}"' in s
        for col in ("signal_time_utc", "signal_time_sgt", "signal_time_server"):
            assert row[col] in s, col
        assert "XAUUSD, 15 m" in s and "Alt+G" in s and row["trigger_bar_time_sgt"][:16] in s
        assert s.count('<svg class="m15"') == 1 and s.count('<svg class="h1"') == 1
        for rule in ("2 (D3, D4)", "3 (D5)", "3 (D7)", "3 (D6, D8)", "4 (D9)", "4 (D9, D10)", "4 (D11)",
                     "5, 8 (D12)", "<td class=\"rule\">5</td>", "8 (D18)", "10 (D11)", "9 (D19)", "9 (D20)", "D1"):
            assert rule in s, rule
        assert s.count("USD/oz") > 20 and " bar(s) after the " in s and " min " in s
        assert f"Your answer for row {row['sample_no']}:" in s and s.count('<span class="box">') == 2
        assert "g0_sample.csv agrees" in s and "the code says <b>eligible</b>" in s
        # nothing about how a trade ended
        text = s.lower()
        for bad in ("tp1", "tp2", "pnl", "p&amp;l", "profit", "breakeven", "target", "r multiple", "r_multiple",
                    "outcome", "exit"):
            assert bad not in text, bad


def test_news_option_reads_another_calendar_and_says_so(g0, tmp_path):
    news = tmp_path / "news6.csv"
    news.write_text(NEWS_CSV, encoding="ascii")
    out = tmp_path / "other.html"
    code, stdout, err = call(["zeno-v1", "g0-charts", "--m15-bid", g0["bid"], "--m15-ask", g0["ask"], "--sample",
                              g0["sample"], "--news", news, "--out", out])
    assert code == 0, err
    page = out.read_text(encoding="ascii")
    assert sha(news) in page and "6 NFP/CPI/PPI/FOMC events" in page and "Stage 1 read another news calendar" in page
    assert len(sections(page)) == 20
    assert folder_state(g0["sig"]) == {**g0["before"], g.G0_CHARTS_FILE: sha(g0["page_path"])}


# ---------------------------------------------------------------------------------------
# no leak

def test_no_bar_time_after_the_entry_is_drawn_or_written(g0):
    for no, s in sections(g0["page"]).items():
        row = g0["df"].set_index("sample_no").loc[no]
        entry_t, signal_t = t_of(row["entry_time_utc"]), t_of(row["signal_time_utc"])
        trig_t = t_of(row["trigger_bar_time_utc"])
        assert f'data-entry-time="{entry_t}"' in s and f'data-signal-time="{signal_t}"' in s
        m = marks(s)
        times = [int(d[k]) for lst in m.values() for d in lst for k in ("data-t", "data-t0", "data-t1") if k in d]
        assert times and max(times) == entry_t
        at_entry = {d["data-role"] for lst in m.values() for d in lst if d.get("data-t") == str(entry_t)}
        assert at_entry <= {"entry", "entry-level", "stop", "gap"} and {"entry", "stop"} <= at_entry
        candles = [int(d["data-t"]) for d in m["candle"]]
        assert max(candles) == trig_t < entry_t and candles == sorted(candles)
        assert all(int(d["data-close-t"]) <= signal_t for d in m["h1-candle"])
        assert all(int(d["data-t"]) < signal_t for d in m["h1-candle"])
        # times written as text: every UTC and SGT date-time (the calendar's event time aside) is at or before
        # the entry, and the only server time is the signal's
        body = re.sub(r'<span class="news-event"[^>]*>.*?</span>', "", s)
        body = re.sub(r"<[^>]+>", " ", body)
        utcs = [t_of(x) for x in re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})? UTC", body)]
        sgts = [t_of(x[:-4]) - 8 * 3600 for x in re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})? SGT", body)]
        assert utcs and sgts and max(utcs) <= entry_t and max(sgts) <= entry_t
        assert set(re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} server", body)) == {row["signal_time_server"]}


def test_garbage_after_the_latest_entry_changes_no_section(g0, tmp_path):
    # two data sets that differ only after the latest entry bar's open: every section is byte-identical and
    # the page differs only in the data files' names and sha256
    last_entry = max(t_of(x) for x in g0["df"]["entry_time_utc"])
    b, a = garble(g0["bid_t"], g0["ask_t"], last_entry, seed=11)
    bid2, ask2 = write_pair(tmp_path / "garbage", b, a, stem="OTHER")
    assert sha(bid2) != sha(g0["bid"]) and sha(ask2) != sha(g0["ask"])
    out = tmp_path / "g.html"
    code, _, err = call(["zeno-v1", "g0-charts", "--m15-bid", bid2, "--m15-ask", ask2, "--sample", g0["sample"],
                         "--out", out])
    assert code == 0, err
    page2 = out.read_text(encoding="ascii")
    assert sections(page2) == sections(g0["page"])
    note = ('<p class="warn">The sample was drawn from other files (their sha256 differ from these), but each of '
            "its rows is an eligible signal of these files with the same entry and stop.</p>")
    assert note in page2 and note not in g0["page"]
    back = page2.replace(note, "")
    for new, old in ((bid2, g0["bid"]), (ask2, g0["ask"])):
        back = back.replace(in_page(new), in_page(old)).replace(sha(new), sha(old))
    assert back == g0["page"]


def test_each_section_reads_nothing_after_its_own_entry_bar_open(g0, tmp_path):
    # row by row: garbage from the row's own entry bar on (its open kept), a one-row sample; the row's section
    # is byte-identical to the one drawn from the clean files
    full = sections(g0["page"])
    df = g0["df"]
    for r in range(len(df)):
        row = df.iloc[[r]]
        d = tmp_path / f"r{r + 1}"
        b, a = garble(g0["bid_t"], g0["ask_t"], t_of(row["entry_time_utc"].iat[0]), seed=100 + r)
        bid, ask = write_pair(d, b, a)
        row.to_csv(d / "g0_sample.csv", index=False)
        shutil.copyfile(g0["sig"] / "signals_report.json", d / "signals_report.json")
        code, _, err = call(["zeno-v1", "g0-charts", "--m15-bid", bid, "--m15-ask", ask, "--sample",
                             d / "g0_sample.csv"])
        assert code == 0, (r, err)
        got = sections((d / g.G0_CHARTS_FILE).read_text(encoding="ascii"))
        assert list(got) == [row["sample_no"].iat[0]]
        assert got[row["sample_no"].iat[0]] == full[row["sample_no"].iat[0]], f"sample row {r + 1}"


# ---------------------------------------------------------------------------------------
# values

def test_drawn_values_equal_the_sample_signals_csv_and_the_engine(g0, engine):
    prep, news = engine["prep"], engine["news"]
    # the bid file as the engine reads it (load_m15_bidask)
    bid_all = prep.frame[["time"] + [f"bid_{c}" for c in PRICE_COLS]].set_axis(["time"] + PRICE_COLS, axis=1)
    bid = bid_all.set_index("time")
    times = bid_all["time"].to_numpy()
    h1 = prep.h1
    secs = sections(g0["page"])
    sides = set()
    for _, row in g0["df"].iterrows():
        m = marks(secs[row["sample_no"]])
        sg = g0["signals"].loc[int(row["signal_no"])]
        side = 1 if row["side"] == "long" else -1
        sides.add(side)
        ext_t, pl_t = t_of(row["extreme_bar_time_utc"]), t_of(row["pullback_bar_time_utc"])
        trig_t, entry_t = t_of(row["trigger_bar_time_utc"]), t_of(row["entry_time_utc"])
        arm_t = t_of(sg["arm_bar_time_utc"])

        def price(role: str) -> float:
            return float(one(m, role)["data-price"])

        # the sample's values
        assert price("entry") == pytest.approx(float(row["entry_price"]), abs=1e-9)
        assert price("entry-level") == price("entry")
        assert price("stop") == pytest.approx(float(row["stop_level"]), abs=1e-9)
        assert price("h") == pytest.approx(float(row["h_level"]), abs=1e-9)
        assert price("l") == pytest.approx(float(row["l_level"]), abs=1e-9)
        assert int(one(m, "entry")["data-t"]) == entry_t
        assert int(one(m, "h-mark" if side > 0 else "l-mark")["data-t"]) == ext_t
        assert int(one(m, "pullback")["data-t"]) == pl_t and int(one(m, "threshold")["data-t"]) == pl_t
        assert int(one(m, "trigger")["data-t"]) == trig_t
        # signals.csv's values (the same stage-1 run)
        for role, col in (("retrace", "retrace_level"), ("void", "void_level"), ("pullback", "pullback_level"),
                          ("threshold", "trigger_level")):
            assert price(role) == pytest.approx(float(sg[col]), abs=1e-9), role
        assert int(one(m, "armed")["data-t"]) == arm_t
        s = secs[row["sample_no"]]
        assert f"{float(sg['atr_trigger']):.3f} USD/oz vs 2 x {float(sg['atr_median']):.3f}" in s
        assert f"leg {float(sg['leg_usd']):.3f} USD/oz vs 1.5 x {float(sg['atr_arm']):.3f}" in s
        assert f"{float(sg['spread_entry']):.3f} USD/oz vs 10% x" in s
        # the candles are the bid file's rows, consecutive, ending at the trigger bar
        cand = m["candle"]
        ct = np.array([int(d["data-t"]) for d in cand])
        k0, k1 = np.searchsorted(times, ct[0]), np.searchsorted(times, trig_t)
        assert np.array_equal(ct, times[k0:k1 + 1])
        for d in cand:
            o, h, lo, c = bid.loc[int(d["data-t"]), PRICE_COLS]
            assert (float(d["data-o"]), float(d["data-h"]), float(d["data-l"]), float(d["data-c"])) == (o, h, lo, c)

        def n_bars(t0: int, t1: int) -> int:
            return int(((ct >= t0) & (ct <= t1)).sum())

        wt = one(m, "window-trigger")
        assert int(wt["data-t1"]) == trig_t and int(wt["data-t0"]) == times[np.searchsorted(times, pl_t) + 1]
        assert n_bars(int(wt["data-t0"]), trig_t) == int(sg["bars_since_pullback"])
        ext_win, far_win = ("window-h", "window-l") if side > 0 else ("window-l", "window-h")
        assert int(one(m, ext_win)["data-t1"]) == arm_t and n_bars(int(one(m, ext_win)["data-t0"]), arm_t) == 20
        k_ext = int(np.searchsorted(times, ext_t))
        assert int(one(m, far_win)["data-t1"]) == times[k_ext - 1]
        assert n_bars(int(one(m, far_win)["data-t0"]), times[k_ext - 1]) == 20
        # re-derived here from the bid file: the far end's bar (the latest of the 20 bars before the extreme
        # holding L, or H for a short) and the first bar after the extreme whose wick reached the 50% level
        win = bid_all.iloc[k_ext - 20:k_ext]
        col, lvl = ("low", float(row["l_level"])) if side > 0 else ("high", float(row["h_level"]))
        far_t = int(win.loc[win[col] == lvl, "time"].iloc[-1])
        assert int(one(m, "l-mark" if side > 0 else "h-mark")["data-t"]) == far_t
        after = bid_all[(bid_all["time"] > ext_t) & (bid_all["time"] <= arm_t)]
        hit = (after["low"] <= price("retrace") + 1e-9) if side > 0 else (after["high"] >= price("retrace") - 1e-9)
        assert int(one(m, "touch")["data-t"]) == int(after.loc[hit, "time"].iloc[0])
        # the engine's 1h bars and EMA30: the last closed 1h bar at the trigger close and the one 5 bars earlier
        j = int(z.h1_index_at_m15_close(np.array([trig_t]), h1["close_time"].to_numpy())[0])
        hc = m["h1-candle"]
        assert len(hc) == min(g.H1_PANEL_BARS, j + 1)
        assert int(hc[-1]["data-t"]) == int(h1["time"].iat[j])
        signal_t = t_of(row["signal_time_utc"])
        assert int(h1["close_time"].iat[j]) <= signal_t < int(h1["close_time"].iat[j + 1])
        assert float(hc[-1]["data-c"]) == float(h1["close"].iat[j]) == price("h1-close")
        np.testing.assert_array_equal([float(d["data-ema"]) for d in hc], h1["ema"].to_numpy()[j + 1 - len(hc):j + 1])
        assert price("ema-now") == float(h1["ema"].iat[j])
        e5 = one(m, "ema-5-earlier")
        assert int(e5["data-t"]) == int(h1["time"].iat[j - 5])
        assert float(e5["data-price"]) == float(h1["ema"].iat[j - 5])
        # the calendar's nearest event to the signal time
        nt = int(re.search(r'data-news-t="(\d+)"', s).group(1))
        dist = np.abs(news.times.astype(np.int64) - signal_t)
        assert nt == int(news.times[int(np.argmin(dist))])
    assert sides == {1, -1}


# ---------------------------------------------------------------------------------------
# known answers on the canonical hand-built setups (tests/unit/zeno_v1_testkit.py)

def _canonical(side: int, cell: z.ZenoCell = EVAL_10_X1, entry_spread: float | None = None,
               weekend: bool = False, holes: bool = False):
    """The canonical hand-built setup (zeno_v1_testkit.long_setup, mirrored for a short) annotated in `cell`:
    the trigger closes 14:00 UTC = 22:00 SGT; entry_spread: the entry bar's spread (default 0.20 USD/oz);
    weekend: instead the setup opens on Sunday 22:00 UTC right after a weekend gap whose last bar before it
    opens on the hour (Friday 20:00 UTC), so the gap sits beside the far end's bar and two dated hours are
    one bar apart; holes: a 15-minute hole after each of the 12 bars before the setup (more gaps than the
    caption strip has room for)."""
    if weekend:
        sc = Scenario("2024-03-08 14:00")                      # a Friday
        sc.flat_until("2024-03-08 20:15", 2000.0)
        sc.skip_to("2024-03-10 22:00")
    elif holes:
        sc = Scenario("2024-03-05 06:00")
        for _ in range(12):
            sc.flat(1, 2000.0)
            sc.skip_to(cal.utc_str(sc.cursor + 900)[:16])
    else:
        sc = Scenario("2024-03-05 12:00")
    ti = sc.setup(side)
    # the entry bar opens at the trigger close and then moves (its high, low and close must not be used)
    sc.add(*((2006.0, 2009.0, 2005.0, 2008.0) if side > 0 else (1994.0, 1995.0, 1991.0, 1992.0)),
           spread=entry_spread)
    sc.flat(3, sc.bid[-1][3])
    prep = sc.prepare(trend="long" if side > 0 else "short")
    dec = z.screen(prep, z.ZenoConfig(cell, 100_000.0))
    idx = int(np.flatnonzero((dec["event"] == "trigger").to_numpy())[0])
    a = g.annotate(prep, idx, z.chart_prices(prep), dec, cell.label)
    return sc, ti, a


@pytest.mark.parametrize("side", [1, -1])
def test_known_answers_on_the_canonical_long_and_short(side):
    sc, ti, a = _canonical(side)
    b3, b0, b5 = ti - 4, ti - 7, ti - 2
    assert (a["side"], a["ext_bar"], a["arm_bar"], a["touch_bar"], a["pl_bar"], a["trigger_bar"], a["k"]) == \
        (side, b3, b5, b5, b5, ti, 2)
    assert (a["h_bar"], a["l_bar"]) == ((b3, b0) if side > 0 else (b0, b3))
    assert a["h_window"] == (b5 - 19, b5) and a["l_window"] == (b3 - 20, b3 - 1)
    # the long's levels; a short's are 2 x 2000 - these, with H and L swapped
    lvl = {"h": 2010.0, "l": 1995.0, "retrace": 2002.5, "pl_level": 2002.0, "threshold": 2005.5,
           "worst_close": 2003.0, "prior_close": 2004.0, "trigger_close": 2006.0}
    for key, v in lvl.items():
        got = a[key] if side > 0 else a[{"h": "l", "l": "h"}.get(key, key)]
        assert got == (v if side > 0 else 4000.0 - v), key
    assert a["void"] == pytest.approx(1998.21 if side > 0 else 2001.79, abs=1e-9)
    assert (a["leg"], a["atr_arm"], a["atr"], a["atr_median"]) == (15.0, 2.0, 2.0, 2.0)
    assert (a["entry"], a["stop"]) == pytest.approx((2006.20, 2001.50) if side > 0 else (1994.00, 1998.70), abs=1e-9)
    assert a["spread"] == pytest.approx(0.20, abs=1e-9) and a["dist"] == pytest.approx(4.70, abs=1e-9)
    assert (a["worst_bar"], a["prior_bar"]) == (b5, ti - 1)
    assert a["entry_time"] == sc.time(ti + 1) and a["signal_time"] == sc.time(ti) + 900
    assert a["bars"][-1]["t"] == sc.time(ti) and len(a["bars"]) == ti - a["start"] + 1
    assert a["trend_ok"] and a["session_ok"] and not a["news_blocked"] and not a["news_used"]
    assert a["cell"]["status"] == "eligible" and a["cell"]["label"] == "evaluation/c10/S1/x1"
    # the checklist: every number with its unit and the code's verdict, all "yes" here
    rows = g.checklist_rows(a)
    text = " | ".join(v for _, _, v, _ in rows)
    assert [r[0] for r in rows] == ["2 (D3, D4)", "3 (D5)", "3 (D5)", "3 (D7)", "3 (D6, D8)", "3 (D6, D8)", "4 (D9)",
                                    "4 (D9, D10)", "4 (D11)", "5, 8 (D12)", "5", "8 (D18)", "8 (D18)", "10 (D11)",
                                    "9 (D19)", "9 (D20)", "D1"]
    assert all(v.startswith('<span class="chip ok">yes</span>') for _, _, _, v in rows)
    assert "leg 15.000 USD/oz vs 1.5 x 2.000 = 3.000 USD/oz" in text
    assert "4.700 USD/oz vs 3 x 2.000 = 6.000 USD/oz" in text and "2.000 USD/oz vs 2 x 2.000 = 4.000 USD/oz" in text
    assert "0.200 USD/oz vs 10% x 4.700 = 0.470 USD/oz" in text
    assert "2024-03-05 22:00 SGT = 2024-03-05 14:00 UTC" in text and "2 bar(s) after the" in text
    if side > 0:
        assert "pullback low 2002.000 - 0.25 x ATR14 2.000 = 2001.500 USD/oz (a bid level)" in text
        assert "close 2006.000 &gt; 2005.500 USD/oz" in text and "H = 2010.000 USD/oz" in text
        assert g.trend_words(a).startswith("UP: ") and "agrees with a long" in g.trend_words(a)
    else:
        assert "pullback high 1998.000 + 0.25 x ATR14 2.000 + entry spread 0.200 = 1998.700 USD/oz (an ask level)" \
            in text
        assert "close 1994.000 &lt; 1994.500 USD/oz" in text and "L = 1990.000 USD/oz" in text
        assert g.trend_words(a).startswith("DOWN: ") and "agrees with a short" in g.trend_words(a)
    # the chart: the last candle is the trigger bar, the entry marker is at the next open, nothing later
    m = marks(g.m15_svg(a))
    assert int(m["candle"][-1]["data-t"]) == sc.time(ti) and len(m["candle"]) == len(a["bars"])
    assert int(one(m, "entry")["data-t"]) == sc.time(ti + 1)
    assert int(one(m, "trigger")["data-t"]) == sc.time(ti) and int(one(m, "pullback")["data-t"]) == sc.time(b5)
    assert int(one(m, "touch")["data-t"]) == sc.time(b5) and int(one(m, "armed")["data-t"]) == sc.time(b5)
    wt = one(m, "window-trigger")
    assert (int(wt["data-t0"]), int(wt["data-t1"])) == (sc.time(b5 + 1), sc.time(ti))


# ---------------------------------------------------------------------------------------
# mirror

Q = 1.0 / 64.0                      # price grid: every price, spread and level is exact in binary
SPREAD = 0.25
M = 6000.0                          # mirror: bid' = M - ask, ask' = M - bid (prices stay in [2048, 4096))
C = M - SPREAD                      # bid bars mirror around C / 2: bid' = C - bid
SWAP = {"h": "l", "l": "h", "h-mark": "l-mark", "l-mark": "h-mark", "window-h": "window-l", "window-l": "window-h"}
FILL_ROLES = {"entry", "entry-level", "stop"}       # ask / bid fill levels mirror around M / 2


def _walk(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    step = np.round(rng.normal(0.0, 0.9, n) / Q) * Q
    c = 3000.0 + np.cumsum(step)
    o = np.r_[3000.0, c[:-1]]
    h = np.maximum(o, c) + np.round(np.abs(rng.normal(0, 0.5, n)) / Q) * Q
    lo = np.minimum(o, c) - np.round(np.abs(rng.normal(0, 0.5, n)) / Q) * Q
    t = utc("2024-01-01 00:00") + 900 * np.arange(n, dtype=np.int64)
    return pd.DataFrame({"time": t, "open": o, "high": h, "low": lo, "close": c})


def _mirror_preps():
    bid = _walk(12_000, seed=5)
    ask = bid.copy()
    ask[PRICE_COLS] += SPREAD
    mb = pd.DataFrame({"time": bid["time"], "open": M - ask["open"], "high": M - ask["low"], "low": M - ask["high"],
                       "close": M - ask["close"]})
    ma = pd.DataFrame({"time": bid["time"], "open": M - bid["open"], "high": M - bid["low"], "low": M - bid["high"],
                       "close": M - bid["close"]})
    f, fm = z.bidask_frame(bid, ask), z.bidask_frame(mb, ma)
    atr = np.round(z.atr14_m15(f) * 256) / 256                      # the same dyadic ATR on both sides
    ema = np.round(z.ema30_h1(z.h1_from_m15(f)["close"].to_numpy()) / Q) * Q
    return (z.prepare(f, test_indicators={"atr14": atr, "ema30_h1": ema}),
            z.prepare(fm, test_indicators={"atr14": atr, "ema30_h1": C - ema}))


def _same(x: float, y: float) -> bool:
    """Equal, or both NaN (no value)."""
    return x == y or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y))


def test_a_short_is_the_mirror_of_a_long():
    # bid' = M - ask, ask' = M - bid: a long of one data set is a short of the other at the same bars; every
    # annotation of the short is the long's reflected (bid levels around C / 2, fills around M / 2), and the
    # drawn marks and the checklist verdicts are the same
    pa, pb = _mirror_preps()
    cfg = z.ZenoConfig(EVAL_10_X1, 100_000.0)
    da, db = z.screen(pa, cfg), z.screen(pb, cfg)
    ca, cb = z.chart_prices(pa), z.chart_prices(pb)
    trig_b = {(int(r.bar_index), r.side): i for i, r in db[db["event"] == "trigger"].iterrows()}
    flip = {"long": "short", "short": "long"}
    n = {"long": 0, "short": 0}
    for i, r in da[da["event"] == "trigger"].iterrows():
        if int(r.bar_index) + 1 >= pa.n:
            continue
        j = trig_b.pop((int(r.bar_index), flip[r.side]))
        a, b = g.annotate(pa, i, ca, da, "x"), g.annotate(pb, j, cb, db, "x")
        n[r.side] += 1
        assert b["side"] == -a["side"]
        for key in ("start", "trigger_bar", "ext_bar", "arm_bar", "touch_bar", "pl_bar", "k", "worst_bar",
                    "prior_bar", "h_window", "l_window", "entry_time", "signal_time", "h1_last", "h1_prev",
                    "trading_day", "trend_ok", "session_ok", "news_blocked", "leg", "atr_arm", "atr",
                    "atr_median", "spread", "dist"):
            assert _same(b[key], a[key]), key
        assert (b["h_bar"], b["l_bar"]) == (a["l_bar"], a["h_bar"])
        assert (b["h"], b["l"]) == (C - a["l"], C - a["h"])
        for key in ("retrace", "pl_level", "threshold", "worst_close", "prior_close", "trigger_close", "h1_close",
                    "ema_now", "ema_prev"):
            assert _same(b[key], C - a[key]), key
        assert b["void"] == pytest.approx(C - a["void"], abs=1e-9)
        assert (b["entry"], b["stop"]) == (M - a["entry"], M - a["stop"])
        assert (b["entry_bid_open"], b["entry_ask_open"]) == (M - a["entry_ask_open"], M - a["entry_bid_open"])
        assert [(x["t"], x["o"], x["h"], x["l"], x["c"]) for x in b["bars"]] == \
            [(x["t"], C - x["o"], C - x["l"], C - x["h"], C - x["c"]) for x in a["bars"]]
        assert b["cell"]["status"] == a["cell"]["status"] and b["cell"]["reasons"] == a["cell"]["reasons"]
        if sum(n.values()) % 3 and r.status != "eligible":    # drawn: every eligible pair and a third of the rest
            continue
        # the drawn marks: the same roles at the same bars, prices reflected
        for svg in (g.m15_svg, g.h1_svg):
            ma, mb = marks(svg(a)), marks(svg(b))
            assert {SWAP.get(k, k) for k in ma} == set(mb)
            for role, lst in ma.items():
                other = mb[SWAP.get(role, role)]
                assert len(other) == len(lst), role
                for x, y in zip(lst, other):
                    for key in ("data-t", "data-t0", "data-t1", "data-close-t"):
                        assert x.get(key) == y.get(key), (role, key)
                    if "data-price" in x:
                        want = (M if role in FILL_ROLES else C) - float(x["data-price"])
                        assert float(y["data-price"]) == pytest.approx(want, abs=1e-9), role
                    if role == "candle":
                        ox, hx, lx, cx = (float(x[f"data-{k}"]) for k in "ohlc")
                        assert tuple(float(y[f"data-{k}"]) for k in "ohlc") == (C - ox, C - lx, C - hx, C - cx)
                    if role == "h1-candle":
                        assert float(y["data-c"]) == C - float(x["data-c"])
        # the checklist: the same rules and the same verdicts
        va, vb = g.checklist_rows(a), g.checklist_rows(b)
        assert [(r[0], r[3].split("</span>")[0]) for r in va] == [(r[0], r[3].split("</span>")[0]) for r in vb]
    assert n["long"] >= 5 and n["short"] >= 5, n
    assert all(k[0] + 1 >= pb.n for k in trig_b), "every trigger of the mirrored data has its twin"


# ---------------------------------------------------------------------------------------
# rules 8 and 10 in the declared cell (it decided eligibility; its prices differ from the chart's)

def _row(rows, rule: str, what_starts: str):
    hit = [r for r in rows if r[0] == rule and r[1].startswith(what_starts)]
    assert len(hit) == 1, (rule, what_starts)
    return hit[0]


def _chip(verdict: str) -> str:
    return verdict.split("</span>")[0] + "</span>"


YES, NO = '<span class="chip ok">yes</span>', '<span class="chip no">NO</span>'


@pytest.mark.parametrize("side", [1, -1])
def test_rules_8_and_10_show_and_judge_the_declared_cell(side):
    # S1 x1.5 (the stage-1 cell): the cell's R is the chart's + 0.5 x the spread (0.10 USD/oz); both are shown,
    # the cell's first, and the verdicts are the engine's in that cell
    cell = z.ZenoCell("evaluation", 10.0, "S1", 1.5)
    _, _, a = _canonical(side, cell)
    assert a["cell"]["status"] == "eligible"
    rows = g.checklist_rows(a)
    w, sp = _row(rows, "8 (D18)", "stop distance"), _row(rows, "10 (D11)", "entry spread")
    assert w[2] == ("4.800 USD/oz vs 3 x 2.000 = 6.000 USD/oz in the declared cell evaluation/c10/S1/x1.5, which "
                    "decided eligibility; at the chart&#x27;s prices (g0_sample.csv): 4.700 USD/oz vs 3 x 2.000 = "
                    "6.000 USD/oz")
    assert sp[2] == ("0.300 USD/oz vs 10% x 4.800 = 0.480 USD/oz in the declared cell evaluation/c10/S1/x1.5, which "
                     "decided eligibility; at the chart&#x27;s prices (g0_sample.csv): 0.200 USD/oz vs 10% x 4.700 = "
                     "0.470 USD/oz")
    assert w[3] == YES + " not wider in that cell" and sp[3] == YES + " not above in that cell"
    # a wide data spread at the entry (1.60 USD/oz) and the S2 cell (0.18 USD/oz): at the chart's prices R = 6.1
    # > 3 x ATR14 = 6 and the spread is above 10% of R, but the engine compared the cell's R = 4.68 and spread
    # 0.18 and found the signal eligible; the page must say what the engine did
    s2 = z.ZenoCell("evaluation", 10.0, "S2", 1.0)
    _, _, a = _canonical(side, s2, entry_spread=1.6)
    assert a["cell"]["status"] == "eligible" and a["dist"] == pytest.approx(6.1, abs=1e-9)
    rows = g.checklist_rows(a)
    w, sp = _row(rows, "8 (D18)", "stop distance"), _row(rows, "10 (D11)", "entry spread")
    assert w[2].startswith("4.680 USD/oz vs 3 x 2.000 = 6.000 USD/oz in the declared cell evaluation/c10/S2/x1, "
                           "which decided eligibility") and w[2].endswith("6.100 USD/oz vs 3 x 2.000 = 6.000 USD/oz")
    assert sp[2].startswith("0.180 USD/oz vs 10% x 4.680 = 0.468 USD/oz in the declared cell") and \
        sp[2].endswith("1.600 USD/oz vs 10% x 6.100 = 0.610 USD/oz")
    assert _chip(w[3]) == YES and _chip(sp[3]) == YES
    # the same data in the chart's own cell: the engine blocks it on both counts, and the page says so
    _, _, a = _canonical(side, EVAL_10_X1, entry_spread=1.6)
    assert set(a["cell"]["reasons"].split(";")) >= {"stop_wider_than_3_atr", "spread_gt_10pct_of_stop"}
    rows = g.checklist_rows(a)
    w, sp = _row(rows, "8 (D18)", "stop distance"), _row(rows, "10 (D11)", "entry spread")
    assert w[2] == ("6.100 USD/oz vs 3 x 2.000 = 6.000 USD/oz in the declared cell evaluation/c10/S1/x1, which "
                    "decided eligibility; the chart&#x27;s prices are the same")
    assert w[3] == NO + " wider in that cell" and sp[3] == NO + " above in that cell"


def test_rule_8_and_10_lines_show_the_declared_cells_numbers_on_every_row(g0, engine):
    # the e2e page (declared cell evaluation/c10/S1/x1.5): each row's rule-8 and rule-10 lines carry the R and the
    # spread that screen() compared in that cell, and the engine's verdicts
    prep = engine["prep"]
    cell = zr.STAGE1_CELL
    dec = z.screen(prep, z.ZenoConfig(cell, 100_000.0))
    idx = g.trigger_rows(dec, g0["df"])
    secs = sections(g0["page"])
    for r, i in enumerate(idx):
        row = dec.loc[i]
        s = 1 if row["side"] == "long" else -1
        r_cell, atr = s * (row["entry_price"] - row["stop_level"]), row["atr_trigger"]
        sec = secs[g0["df"]["sample_no"].iat[r]]
        where = f" in the declared cell {cell.label}, which decided eligibility"
        assert f"{r_cell:.3f} USD/oz vs 3 x {atr:.3f} = {3 * atr:.3f} USD/oz{where}" in sec, r + 1
        assert f"{row['spread_entry']:.3f} USD/oz vs 10% x {r_cell:.3f} = {0.1 * r_cell:.3f} USD/oz{where}" in sec
        assert sec.count(" in that cell</td>") == 2 and f"{YES} not wider in that cell" in sec


# ---------------------------------------------------------------------------------------
# readability: nothing drawn crosses a label, labels apart, the right day on the time axes. The checks read
# the SVG geometry only (an independent oracle; text widths from Arial's metrics, which Liberation Sans shares)

_EM = {**dict.fromkeys("0123456789", 0.556), " ": 0.278, ".": 0.278, ":": 0.278, "-": 0.333, "/": 0.278,
       "%": 0.889, ">": 0.584, "<": 0.584, "(": 0.333, ")": 0.333, "U": 0.722, "T": 0.611, "C": 0.722, "S": 0.667,
       "G": 0.778, "a": 0.556, "b": 0.556, "d": 0.556, "e": 0.556, "g": 0.556, "h": 0.556, "i": 0.222, "m": 0.833,
       "n": 0.556, "p": 0.556, "o": 0.556, "r": 0.333, "s": 0.5, "t": 0.278, "w": 0.722, "y": 0.5, "k": 0.5}
CSS_PX = {"axis": 11.0, "axis2": 10.0, "small": 11.0}      # the page's CSS font sizes, which win over attributes


def text_w(s: str, px: float, bold: bool = False) -> float:
    return sum(_EM.get(ch, 0.667) for ch in s) * px * (1.1 if bold else 1.0)


class _Svg(HTMLParser):
    """Every drawn element of an SVG fragment in document order: tag, attributes, text, the enclosing <g>s."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.els: list[dict] = []
        self.groups: list[dict] = []
        self.cur = None

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == "g":
            self.groups.append(d)
        elif tag != "svg":
            self.els.append({"tag": tag, "a": d, "text": "", "g": list(self.groups)})
            self.cur = self.els[-1] if tag == "text" else None

    def handle_startendtag(self, tag, attrs):
        if tag not in ("g", "svg"):
            self.els.append({"tag": tag, "a": dict(attrs), "text": "", "g": list(self.groups)})

    def handle_endtag(self, tag):
        if tag == "g" and self.groups:
            self.groups.pop()
        if tag == "text":
            self.cur = None

    def handle_data(self, data):
        if self.cur is not None:
            self.cur["text"] += data


def svg_els(svg: str) -> list[dict]:
    p = _Svg()
    p.feed(svg)
    p.close()
    for k, e in enumerate(p.els):
        e["k"] = k
    return p.els


def _font_px(e: dict) -> float:
    classes = e["a"].get("class", "").split()
    return next((CSS_PX[c] for c in classes if c in CSS_PX), 11.5)


def bbox(e: dict) -> tuple[float, float, float, float] | None:
    """(x0, y0, x1, y1) of what an element paints (stroke included), None for an element that paints nothing."""
    a = e["a"]
    f = lambda k: float(a[k])                                                      # noqa: E731
    half = float(a.get("stroke-width", 1.0)) / 2 if a.get("stroke", "none") != "none" else 0.0
    if e["tag"] == "line":
        return min(f("x1"), f("x2")) - half, min(f("y1"), f("y2")) - half, max(f("x1"), f("x2")) + half, \
            max(f("y1"), f("y2")) + half
    if e["tag"] == "rect":
        return f("x") - half, f("y") - half, f("x") + f("width") + half, f("y") + f("height") + half
    if e["tag"] == "circle":
        r = f("r") + half
        return f("cx") - r, f("cy") - r, f("cx") + r, f("cy") + r
    if e["tag"] in ("polygon", "path"):
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", a.get("points") or a.get("d"))]
        xs_, ys_ = nums[0::2], nums[1::2]
        return min(xs_) - half, min(ys_) - half, max(xs_) + half, max(ys_) + half
    if e["tag"] == "text" and e["text"].strip():
        px = _font_px(e)
        w = text_w(e["text"], px, bold=bool(a.get("font-weight")))
        x, y = f("x"), f("y")
        x0 = {"start": x, "middle": x - w / 2, "end": x - w}[a.get("text-anchor", "start")]
        return x0, y - 0.75 * px, x0 + w, y + 0.22 * px
    return None


def hits(b1, b2, tol: float = 0.5) -> bool:
    return b1[0] < b2[2] - tol and b2[0] < b1[2] - tol and b1[1] < b2[3] - tol and b2[1] < b1[3] - tol


@pytest.fixture(scope="module")
def charts(g0):
    """(name, m15 svg, h1 svg) of every section of the e2e page and of six hand-built setups: the canonical long
    and short, the same right after a weekend gap (a gap caption beside the far end's mark, two dated axis
    hours one bar apart), and the same after a hole behind every bar (captions written under the chart)."""
    out = []
    for no, s in sections(g0["page"]).items():
        out.append((f"row {no}", re.search(r'<svg class="m15".*?</svg>', s, re.S).group(0),
                    re.search(r'<svg class="h1".*?</svg>', s, re.S).group(0)))
    for kind in ("", "weekend", "holes"):
        for side in (1, -1):
            a = _canonical(side, weekend=kind == "weekend", holes=kind == "holes")[2]
            out.append((f"{kind} canonical {'long' if side > 0 else 'short'}".strip(), g.m15_svg(a), g.h1_svg(a)))
    return out


def _frame(els: list[dict]) -> tuple[float, float, float, float]:
    r = next(e for e in els if e["tag"] == "rect")                 # the plot frame is the first rectangle
    return bbox(r)


def test_event_label_connectors_cross_no_other_label(charts):
    # V3: each event label's dotted connector runs from its own box down to its mark and through no other
    # label's box; boxes never overlap; every connector is drawn before every box (a box hides a line)
    for name, svg, _ in charts:
        els = svg_els(svg)
        top = _frame(els)[1]
        boxes = [e for e in els if e["tag"] == "rect" and float(e["a"]["height"]) == 17.0
                 and float(e["a"]["y"]) + 17.0 <= top + 0.1]
        links = [e for e in els if e["tag"] == "line" and e["a"].get("stroke-dasharray") == "1 2"]
        assert boxes and len(links) == len(boxes), name
        bb = [bbox(b) for b in boxes]
        for i in range(len(bb)):
            for j in range(i + 1, len(bb)):
                assert not hits(bb[i], bb[j], tol=0.0), (name, "boxes overlap", i, j)
        for ln in links:
            x, y1, y2 = float(ln["a"]["x1"]), float(ln["a"]["y1"]), float(ln["a"]["y2"])
            own = [k for k, b in enumerate(bb) if b[0] + 2 < x < b[2] - 2 and abs(b[3] - 0.8 - y1) <= 1.5]
            assert len(own) == 1, (name, "connector without its box", x)
            for k, b in enumerate(bb):
                if k != own[0]:
                    assert not (b[0] - 1 < x < b[2] + 1 and y1 < b[3] and b[1] < y2), \
                        (name, "a connector crosses another label", x, b)
        assert max(e["k"] for e in links) < min(e["k"] for e in boxes), (name, "a connector drawn over a box")


def _axis(els: list[dict]) -> list[list[dict]]:
    """The time-axis labels under the plot, one list per line (SGT, then UTC), each in x order."""
    bottom = _frame(els)[3]
    lab = [e for e in els if e["tag"] == "text" and e["a"].get("text-anchor") == "middle"
           and float(e["a"]["y"]) > bottom and {"axis", "axis2"} & set(e["a"].get("class", "").split())]
    lines = {}
    for e in lab:
        lines.setdefault(float(e["a"]["y"]), []).append(e)
    return [sorted(lines[y], key=lambda e: float(e["a"]["x"])) for y in sorted(lines)]


def _tick_times(els: list[dict]) -> dict[str, int]:
    """x (as written) -> the bar open there: the candles' wicks, and the entry diamond's centre."""
    out = {}
    for e in els:
        roles = [gr.get("data-role") for gr in e["g"]]
        if e["tag"] == "line" and ("candle" in roles or "h1-candle" in roles):
            out[e["a"]["x1"]] = int(next(gr["data-t"] for gr in e["g"] if "data-t" in gr))
        if e["tag"] == "polygon" and e["a"].get("data-role") == "entry":
            b = bbox(e)
            out[f"{(b[0] + b[2]) / 2:.1f}"] = int(e["a"]["data-t"])
    return out


def test_time_axis_labels_apart_and_on_the_right_day(charts):
    # V4, V8: on the M15 and the 1h axis, labels on one line never come within 6 px of each other, and every
    # label is read on the right day: an undated SGT time takes the SGT date of the label before it; an undated
    # UTC time takes the date shown above it on the SGT line, else the UTC date of the UTC label before it
    for name, m15, h1 in charts:
        for which, svg in (("M15", m15), ("1h", h1)):
            els = svg_els(svg)
            lines = _axis(els)
            assert len(lines) == 2 and len(lines[0]) == len(lines[1]) >= 2, (name, which)
            for line in lines:
                ext = [bbox(e) for e in line]
                for b0, b1 in zip(ext, ext[1:]):
                    assert b1[0] - b0[2] >= 6.0, (name, which, "labels too close", [e["text"] for e in line])
            when = _tick_times(els)
            s_day = u_day = None
            for es, eu in zip(*lines):
                assert es["a"]["x"] == eu["a"]["x"]
                t = when[es["a"]["x"]]
                sg = str(pd.Timestamp(t + 8 * 3600, unit="s"))[5:16]
                ug = str(pd.Timestamp(t, unit="s"))[5:16]
                st, ut = es["text"], eu["text"].removesuffix(" UTC")
                s_read = st if len(st) == 11 else f"{s_day} {st}"
                u_read = ut if len(ut) == 11 else f"{st[:5] if len(st) == 11 else u_day} {ut}"
                assert (s_read, u_read) == (sg, ug), (name, which, st, ut, sg, ug)
                s_day, u_day = s_read[:5], u_read[:5]


def test_gap_captions_clear_of_every_mark_and_line(charts):
    # V5: a data gap's caption ("gap 2 d 1 h 45 min") touches nothing else drawn: no candle, mark, level, grid
    # or gap line, connector or other text
    n = 0
    for name, svg, _ in charts:
        els = svg_els(svg)
        frame = next(e for e in els if e["tag"] == "rect")
        caps = [e for e in els if e["tag"] == "text" and e["text"].startswith("gap ")]
        assert len(caps) == sum(e["a"].get("data-role") == "gap" for e in els), name     # every gap captioned
        for cap in caps:
            n += 1
            cb = bbox(cap)
            for e in els:
                if e is cap or e is frame or e["a"].get("data-role") == "window-trigger-shade":
                    continue
                b = bbox(e)
                assert b is None or not hits(cb, b), (name, cap["text"], e["tag"], e["a"], e["text"])
    assert n >= 3


def test_right_level_labels_do_not_overlap(charts):
    # V10: the level labels on the right (H, L, 50%, 78.6%, trigger level, STOP, ENTRY) are 17 px boxes whose
    # borders stay at least 2 px apart
    for name, svg, _ in charts:
        els = svg_els(svg)
        right = _frame(els)[2]
        boxes = sorted((bbox(e) for e in els if e["tag"] == "rect" and float(e["a"]["height"]) == 17.0
                        and float(e["a"]["x"]) > right + 5), key=lambda b: b[1])
        assert len(boxes) == 7, name
        for b0, b1 in zip(boxes, boxes[1:]):
            assert b1[1] - b0[3] >= 2.0, (name, b0, b1)


def test_h1_panel_numbers_read_as_in_the_checklist(g0):
    # V9: the 1h panel's EMA30 5 bars ago, EMA30 now and last close are the checklist's numbers, digit for digit
    for no, s in sections(g0["page"]).items():
        h1 = re.search(r'<svg class="h1".*?</svg>', s, re.S).group(0)
        panel = {e["text"].rsplit(" ", 1)[0]: e["text"].rsplit(" ", 1)[1] for e in svg_els(h1)
                 if e["tag"] == "text" and e["text"].startswith(("EMA30 ", "last close "))}
        m = re.search(r"close ([\d.]+) vs EMA30 ([\d.]+) USD/oz; EMA30 5 bars earlier ([\d.]+) USD/oz", s)
        assert panel == {"last close": m.group(1), "EMA30 now": m.group(2), "EMA30 5 bars ago": m.group(3)}, no


# ---------------------------------------------------------------------------------------
# paths and calendars

def test_paths_with_non_ascii_and_html_special_characters(g0, tmp_path):
    # S2: a folder named with Chinese characters, an apostrophe and an ampersand: the console line is ASCII with
    # backslash escapes (cli._say, for a cp936 console) and the page holds the paths as character references
    d = tmp_path / "临时 张三 O'Neil & co"
    sig = d / "sig"
    sig.mkdir(parents=True)
    bid, ask = d / g0["bid"].name, d / g0["ask"].name
    for src, dst in ((g0["bid"], bid), (g0["ask"], ask), (g0["sample"], sig / "g0_sample.csv"),
                     (g0["sig"] / "signals_report.json", sig / "signals_report.json")):
        shutil.copyfile(src, dst)
    code, out, err = call(["zeno-v1", "g0-charts", "--m15-bid", bid, "--m15-ask", ask, "--sample",
                           sig / "g0_sample.csv"])
    assert code == 0, err
    page = sig / g.G0_CHARTS_FILE
    assert out.splitlines()[1] == f"Wrote {console(page)}" and "\\u4e34\\u65f6 \\u5f20\\u4e09 O'Neil & co" in out
    text = page.read_text(encoding="ascii")
    head = text.split("<section ", 1)[0]
    for f in (bid, ask, sig / "g0_sample.csv"):
        assert in_page(f) in head and str(f) not in head
    assert "&#20020;&#26102; &#24352;&#19977; O&#x27;Neil &amp; co" in head
    assert sections(text) == sections(g0["page"])


def test_a_sample_from_another_calendar_names_the_option(g0, engine, tmp_path):
    # S1: stage 1 read another news calendar (the packaged one without its 2015 events, so nothing is blacked
    # out in these 2015 bars) and sampled signals that the packaged calendar blocks. g0-charts reads the packaged
    # calendar by default: it must refuse with the right diagnosis (stage 1's calendar, --news <that file>), not
    # blame the M15 files; with --news <that file> every row is drawn
    base = ["zeno-v1", "g0-charts", "--m15-bid", g0["bid"], "--m15-ask", g0["ask"]]
    cal = pd.read_csv(g.PACKAGED_NEWS_CSV, dtype=str, keep_default_na=False)
    other = tmp_path / "news no 2015.csv"
    cal[~cal["datetime_utc"].str.startswith("2015")].to_csv(other, index=False)
    sig = tmp_path / "sig_news"
    code, _, err = call(["zeno-v1", "signals", "--m15-bid", g0["bid"], "--m15-ask", g0["ask"], "--news", other,
                         "--sample", "1000", "--out", sig])
    assert code == 0, err
    full = pd.read_csv(sig / "g0_sample.csv", dtype=str, keep_default_na=False)
    chk = zr.g0_sample_check(engine["prep"], full, zr.STAGE1_CELL)       # the packaged calendar
    blocked = [int(u["sample_no"]) for u in chk["unmatched"] if "news_blackout" in u["why"]]
    assert blocked and len(blocked) == chk["n_rows"] - chk["n_matched"], chk["unmatched"][:3]
    keep = [blocked[0]] + [k for k in range(1, len(full) + 1) if k not in blocked][:2]
    full[full["sample_no"].astype(int).isin(keep)].to_csv(sig / "g0_sample.csv", index=False)
    code, out, err = call(base + ["--sample", sig / "g0_sample.csv"])
    assert code == 2 and out == "" and not (sig / g.G0_CHARTS_FILE).exists(), err
    assert "Stage 1 read another news calendar" in err and "news_blackout" in err, err
    assert f"Run g0-charts again with --news {console(other)}." in err, err
    assert "does not belong to this data" not in err and "run `zeno-v1 signals`" not in err, err
    code, out, err = call(base + ["--sample", sig / "g0_sample.csv", "--news", other])
    assert code == 0 and "3 of 3 rows" in out, err
    # the same for master_fp's restricted calendar: stage 1 read the packaged one, this run reads one with an
    # extra event at a sampled signal's entry time, so that row is blocked here; the message names --restricted
    sig = tmp_path / "sig_fp"
    code, _, err = call(["zeno-v1", "signals", "--m15-bid", g0["bid"], "--m15-ask", g0["ask"], "--news",
                         g.PACKAGED_NEWS_CSV, "--variant", "master_fp", "--out", sig])
    assert code == 0, err
    df = pd.read_csv(sig / "g0_sample.csv", dtype=str, keep_default_na=False).iloc[:3]
    df.to_csv(sig / "g0_sample.csv", index=False)
    when = pd.Timestamp(t_of(df["signal_time_utc"].iat[1]), unit="s")
    rc = pd.read_csv(z.RESTRICTED_CSV, dtype=str, keep_default_na=False)
    rc.loc[len(rc)] = {**dict.fromkeys(rc.columns, ""), "event": "CPI", "date_et": when.strftime("%Y-%m-%d"),
                       "time_et": when.strftime("%H:%M"), "utc_offset_ny": "+0000", "kind": "scheduled",
                       "datetime_utc": when.strftime("%Y-%m-%dT%H:%MZ")}
    plus = tmp_path / "restricted plus one.csv"
    rc.to_csv(plus, index=False)
    code, out, err = call(base + ["--sample", sig / "g0_sample.csv", "--restricted", plus])
    assert code == 2 and out == "" and "sample row 2" in err and "fp_restricted_window" in err, err
    assert "Stage 1 read another restricted calendar" in err, err
    assert f"Run g0-charts again with --restricted {console(z.RESTRICTED_CSV)}." in err, err
    code, out, err = call(base + ["--sample", sig / "g0_sample.csv"])
    assert code == 0 and "3 of 3 rows" in out, err


# ---------------------------------------------------------------------------------------
# refusals

def _copy_sample(g0, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(g0["sample"], folder / "g0_sample.csv")
    shutil.copyfile(g0["sig"] / "signals_report.json", folder / "signals_report.json")
    return folder / "g0_sample.csv"


def test_refusals_locked_paths_lock_rows_overwrite_and_out(g0, tmp_path):
    sample = _copy_sample(g0, tmp_path / "s")
    page = sample.parent / g.G0_CHARTS_FILE
    state = folder_state(sample.parent)
    base = ["zeno-v1", "g0-charts", "--m15-bid", g0["bid"], "--m15-ask", g0["ask"]]

    def refused(args, *words):
        code, out, err = call(args)
        assert code == 2 and out == "", (args, out, err)
        for w in words:
            assert w in err, (w, err)
        assert folder_state(sample.parent) == state                  # nothing written, the sample unchanged
        return err

    # locked paths, read or written: refused before anything is read or created
    lh = tmp_path / "locked_holdout"
    refused(["zeno-v1", "g0-charts", "--m15-bid", lh / "bid.csv", "--m15-ask", g0["ask"], "--sample", sample],
            "locked")
    refused(["zeno-v1", "g0-charts", "--m15-bid", g0["bid"], "--m15-ask", tmp_path / "ask.csv.locked", "--sample",
             sample], "locked")
    refused(base + ["--sample", lh / "g0_sample.csv"], "locked")
    refused(base + ["--sample", sample, "--news", lh / "news.csv"], "locked")
    refused(base + ["--sample", sample, "--out", lh / "g0.html"], "locked", "write")
    refused(base + ["--sample", sample, "--out", tmp_path / "g0.html.locked"], "locked")
    assert not lh.exists()
    # --out: a folder, not an .html name, an input
    folder = tmp_path / "a_folder"
    folder.mkdir()
    refused(base + ["--sample", sample, "--out", folder], "is a folder")
    refused(base + ["--sample", sample, "--out", sample], ".html")
    refused(base + ["--sample", sample, "--out", tmp_path / "page.txt"], ".html")
    assert list(folder.iterdir()) == [] and not (tmp_path / "page.txt").exists()
    # bars at the lock: a file whose last bar opens at 2025-09-28 00:00 UTC is refused (no override)
    f = z.synthetic_m15_bidask(start=z.LOCK_UTC - 900 * 6000, n_bars=3000, seed=1)
    lb, la = split(f)
    for df in (lb, la):
        df.loc[len(df)] = [z.LOCK_UTC] + df.iloc[-1, 1:].tolist()
    bad_bid, bad_ask = write_pair(tmp_path / "lock", lb, la)
    refused(["zeno-v1", "g0-charts", "--m15-bid", bad_bid, "--m15-ask", bad_ask, "--sample", sample],
            "2025-09-28 00:00:00 UTC", "no override")
    # a row that is not an eligible signal of this data with the same time, side, entry and stop
    df = g0["df"].copy()
    df.loc[2, "stop_level"] = str(float(df.loc[2, "stop_level"]) + 0.5)
    bad = _copy_sample(g0, tmp_path / "bad_stop")
    df.to_csv(bad, index=False)
    for args in ([], ["--out", tmp_path / "x.html"]):
        code, out, err = call(base + ["--sample", bad] + args)
        assert code == 2 and out == "" and "does not belong to this data" in err and "sample row 3" in err
        assert "Nothing was written" in err and not (bad.parent / g.G0_CHARTS_FILE).exists()
    assert not (tmp_path / "x.html").exists()
    df = g0["df"].copy()
    df.loc[4, "signal_time_utc"] = cal.utc_str(t_of(df.loc[4, "signal_time_utc"]) + 420)    # not a bar close
    bad = _copy_sample(g0, tmp_path / "bad_time")
    df.to_csv(bad, index=False)
    code, _, err = call(base + ["--sample", bad])
    assert code == 2 and "sample row 5" in err and "no trigger at that time and side" in err
    empty = _copy_sample(g0, tmp_path / "empty")
    g0["df"].iloc[:0].to_csv(empty, index=False)
    code, _, err = call(base + ["--sample", empty])
    assert code == 2 and "has no rows" in err and not (empty.parent / g.G0_CHARTS_FILE).exists()
    code, _, err = call(base + ["--sample", tmp_path / "missing.csv"])
    assert code == 2 and "file not found" in err
    # an existing page is replaced only with --force
    code, _, err = call(base + ["--sample", sample])
    assert code == 0, err
    page.write_text("an old page", encoding="ascii")
    state = folder_state(sample.parent)
    refused(base + ["--sample", sample], "already exists", "--force")
    assert page.read_text(encoding="ascii") == "an old page"
    code, out, err = call(base + ["--sample", sample, "--force"])
    assert code == 0 and f"Wrote {console(page)}" in out, err
    assert sections(page.read_text(encoding="ascii")) == sections(g0["page"])
    assert sha(sample) == sha(g0["sample"])
    assert sorted(p.name for p in sample.parent.iterdir()) == sorted(["g0_sample.csv", "signals_report.json",
                                                                       g.G0_CHARTS_FILE])


# ---------------------------------------------------------------------------------------
# help and docs

def test_help_and_docs_point_to_g0_charts():
    code, out, _ = call(["zeno-v1", "--help"])
    assert code == 0 and "g0-charts" in out
    code, out, _ = call(["zeno-v1", "g0-charts", "--help"])
    assert code == 0
    for opt in ("--m15-bid", "--m15-ask", "--sample", "--out", "--force", "--news"):
        assert opt in out, opt
    code, _, err = call(["zeno-v1"])
    assert code == 2 and "needs a stage" in err and "g0-charts" in err
    assert "zeno-v1 g0-charts" in zr.G0_INSTRUCTIONS and "g0_charts.html" in zr.G0_INSTRUCTIONS
    root = Path(__file__).resolve().parents[2]
    methods = (root / "propkit" / "METHODS.md").read_text(encoding="ascii")
    assert "### 8.9 The G0 charts" in methods and "zeno-v1 g0-charts" in methods
    sec = methods.split("### 8.9 The G0 charts", 1)[1]
    assert "nothing after" in sec.lower() and "test_zeno_v1_g0_charts.py" in sec
    readme = (root / "propkit" / "README.md").read_text(encoding="ascii")
    assert "zeno-v1 g0-charts" in readme
