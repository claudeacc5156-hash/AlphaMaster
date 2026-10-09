"""zeno_pullback_v1 against the independent oracle: 17 hand-built cases, one test per case.

RESEARCH ONLY: synthetic prices; nothing places, prepares or simulates sending orders.

Where the cases come from
  The oracle (ORACLE.md + cases.json of the zv1 workflow) was built from the spec text alone
  (records/specs/zeno_pullback_v1.md, sha256 d36ad25f74c293166bd82f117cb96a6e6890a2ab2c3dd63dbc41bff52b67bbcb)
  without reading propkit/zeno_v1.py. Its bar files are copied byte for byte to
  tests/unit/fixtures/zeno_v1_oracle/<case>_bid.csv and _ask.csv (dukascopy-node CSV: timestamp = bar OPEN in UTC
  milliseconds, open, high, low, close, volume) and are read here through zeno_v1.load_m15_bidask. The expected
  values are the oracle's, with its hand arithmetic in the comments. Where the oracle was wrong, the value was
  re-derived from the spec and the comment says "ORACLE CORRECTED" with the spec line.

How the oracle's injected indicators are reproduced (the oracle has no price history before a case)
  * ATR14 and EMA30(1h) go through prepare(..., test_indicators=...), the documented test-only parameter:
    ATR14 = 2.00 on every bar except the wide bars of C10a, C10b and C12 (Wilder values, ORACLE section 2);
    EMA30_k = 1950.00 + 0.10 k (long cases) or 2050.00 - 0.10 k (C04a, C04b, C14) for the k-th 1h bar of the case
    (k = 0 is the case's first hour); the oracle's 1h close is the close of the M15 bar that opens at HH:45.
  * The oracle treats the 30-trading-day warm-up (D1) as complete and injects the D18 median as 2.0. prepare() has
    no parameter for either, so each case is preceded by WARMUP_DAYS days of flat bars that repeat the case's first
    bar, with ATR14 = 2.00 and EMA NaN (no trend). The warm-up is then over and the median of the previous 20
    trading days is exactly 2.0 (checked in run_case). Every test compares the COMPLETE event list, so the prefix
    adds no event; the oracle's "diagnostic, NOT expected" re-arm events would also show up there.
  * News (D20): the 22 rows of the real calendar CSV dated 2024-06-01 .. 2024-11-30, verbatim. Only C08's date has
    an event. test_embedded_calendar_rows_are_the_real_files compares them with the real file when it is mounted.
Cost cell (ORACLE section 1): evaluation, commission 10 USD per lot round trip, spread base S1, multiplier 1,
100,000 USD, risk 0.5%, stop slippage 0.05 USD/oz on every stop fill (breakeven included), M1 not run.

Field names: for a short, the impl's l_level is the extreme (the lowest low, the spec's short "L") and h_level the
highest high of the 20 bars before it; the oracle's armed rows put the extreme under "H". The helpers compare
`extreme` / `opposite` so both conventions read the same. Oracle reason and exit labels map through REASONS / EXITS.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from propkit import zeno_v1 as z

FIX = Path(__file__).resolve().parent / "fixtures" / "zeno_v1_oracle"
REAL_CALENDAR = Path("/mnt/project-files/research/news_calendar/us_macro_events_2015-01-01_2025-09-27.csv")
WARMUP_DAYS = 35                 # > 30 trading days of warm-up (D1) and >= 20 days for the median (D18)
M15 = 900
ATR = 2.0
CELL = z.ZenoCell("evaluation", 10.0, "S1", 1.0)
TOL = 1e-6                       # USD/oz and USD; the oracle prints at most 6 decimals

# oracle reason label -> zeno_v1.BLOCK_REASONS key (I6: with several reasons the first one is implementation-defined)
REASONS = {"session": "outside_session", "news": "news_blackout", "spread": "spread_gt_10pct_of_stop",
           "trend": "trend_disagrees", "cooldown_15min": "cooldown_15min",
           "daily_max_2_entries": "max_entries_per_day", "daily_2_losses": "two_losses_today",
           "daily_loss_1pct": "day_loss_1pct"}
# oracle exit label -> (zeno_v1 TRADES exit_reason, leg)
EXITS = {"tp1_+2R": ("target", "tp1"), "tp2_+4R": ("target", "runner"), "breakeven": ("stop", "runner"),
         "stop": ("stop", "full"), "time_exit_1630NY": ("time", "full")}

CALENDAR_HEADER = "event,date_et,time_et,utc_offset_ny,datetime_utc,kind,basis,source_list,cross_check,note\n"
CALENDAR_ROWS = (   # verbatim rows of the real calendar CSV, date_et 2024-06-01 .. 2024-11-30
    "NFP,2024-06-07,08:30,-0400,2024-06-07T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/empsit.htm,"
    "ALFRED rid=50 + BLS yearly schedules,May 2024 Employment Situation; BLS archive empsit_06072024.htm (Fri)\n",
    "CPI,2024-06-12,08:30,-0400,2024-06-12T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/cpi.htm,"
    "ALFRED rid=10 + BLS yearly schedules,May 2024 CPI (cpi_06122024.htm)\n",
    "FOMC,2024-06-12,14:00,-0400,2024-06-12T18:00Z,scheduled,both,https://www.federalreserve.gov/monetarypolicy/"
    "fomchistorical<YEAR>.htm (2015-2020); fomccalendars.htm (2021-2025),Fed press releases + openmarket.htm,"
    "Jun 11-12 meeting\n",
    "PPI,2024-06-13,08:30,-0400,2024-06-13T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/ppi.htm,"
    "ALFRED rid=46 + BLS yearly schedules,May 2024 PPI\n",
    "NFP,2024-07-05,08:30,-0400,2024-07-05T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/empsit.htm,"
    "ALFRED rid=50 + BLS yearly schedules,Jun 2024 Employment Situation; BLS archive empsit_07052024.htm (Fri)\n",
    "CPI,2024-07-11,08:30,-0400,2024-07-11T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/cpi.htm,"
    "ALFRED rid=10 + BLS yearly schedules,Jun 2024 CPI (cpi_07112024.htm)\n",
    "PPI,2024-07-12,08:30,-0400,2024-07-12T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/ppi.htm,"
    "ALFRED rid=46 + BLS yearly schedules,Jun 2024 PPI\n",
    "FOMC,2024-07-31,14:00,-0400,2024-07-31T18:00Z,scheduled,both,https://www.federalreserve.gov/monetarypolicy/"
    "fomchistorical<YEAR>.htm (2015-2020); fomccalendars.htm (2021-2025),Fed press releases + openmarket.htm,"
    "Jul 30-31 meeting\n",
    "NFP,2024-08-02,08:30,-0400,2024-08-02T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/empsit.htm,"
    "ALFRED rid=50 + BLS yearly schedules,Jul 2024 Employment Situation; BLS archive empsit_08022024.htm (Fri)\n",
    "PPI,2024-08-13,08:30,-0400,2024-08-13T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/ppi.htm,"
    "ALFRED rid=46 + BLS yearly schedules,Jul 2024 PPI\n",
    "CPI,2024-08-14,08:30,-0400,2024-08-14T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/cpi.htm,"
    "ALFRED rid=10 + BLS yearly schedules,Jul 2024 CPI (cpi_08142024.htm)\n",
    "NFP,2024-09-06,08:30,-0400,2024-09-06T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/empsit.htm,"
    "ALFRED rid=50 + BLS yearly schedules,Aug 2024 Employment Situation; BLS archive empsit_09062024.htm (Fri)\n",
    "CPI,2024-09-11,08:30,-0400,2024-09-11T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/cpi.htm,"
    "ALFRED rid=10 + BLS yearly schedules,Aug 2024 CPI (cpi_09112024.htm)\n",
    "PPI,2024-09-12,08:30,-0400,2024-09-12T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/ppi.htm,"
    "ALFRED rid=46 + BLS yearly schedules,Aug 2024 PPI\n",
    "FOMC,2024-09-18,14:00,-0400,2024-09-18T18:00Z,scheduled,both,https://www.federalreserve.gov/monetarypolicy/"
    "fomchistorical<YEAR>.htm (2015-2020); fomccalendars.htm (2021-2025),Fed press releases + openmarket.htm,"
    "Sep 17-18 meeting\n",
    "NFP,2024-10-04,08:30,-0400,2024-10-04T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/empsit.htm,"
    "ALFRED rid=50 + BLS yearly schedules,Sep 2024 Employment Situation; BLS archive empsit_10042024.htm (Fri)\n",
    "CPI,2024-10-10,08:30,-0400,2024-10-10T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/cpi.htm,"
    "ALFRED rid=10 + BLS yearly schedules,Sep 2024 CPI (cpi_10102024.htm)\n",
    "PPI,2024-10-11,08:30,-0400,2024-10-11T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/ppi.htm,"
    "ALFRED rid=46 + BLS yearly schedules,Sep 2024 PPI\n",
    "NFP,2024-11-01,08:30,-0400,2024-11-01T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/empsit.htm,"
    "ALFRED rid=50 + BLS yearly schedules,Oct 2024 Employment Situation; BLS archive empsit_11012024.htm (Fri)\n",
    "FOMC,2024-11-07,14:00,-0500,2024-11-07T19:00Z,scheduled,both,https://www.federalreserve.gov/monetarypolicy/"
    "fomchistorical<YEAR>.htm (2015-2020); fomccalendars.htm (2021-2025),Fed press releases + openmarket.htm,"
    "Nov 6-7 meeting (Thursday release)\n",
    "CPI,2024-11-13,08:30,-0500,2024-11-13T13:30Z,scheduled,both,https://www.bls.gov/bls/news-release/cpi.htm,"
    "ALFRED rid=10 + BLS yearly schedules,Oct 2024 CPI (cpi_11132024.htm)\n",
    "PPI,2024-11-14,08:30,-0500,2024-11-14T13:30Z,scheduled,both,https://www.bls.gov/bls/news-release/ppi.htm,"
    "ALFRED rid=46 + BLS yearly schedules,Oct 2024 PPI\n",
)


# ---------------------------------------------------------------------------------------
# harness

def utc(text: str) -> int:
    """UTC epoch seconds of 'YYYY-MM-DD HH:MM' (UTC)."""
    return int(pd.Timestamp(text, tz="UTC").timestamp())


@dataclass
class Run:
    prep: z.Prepared
    res: z.CellResult
    n_pre: int


@pytest.fixture(scope="module")
def news(tmp_path_factory) -> z.NewsCalendar:
    p = tmp_path_factory.mktemp("zeno_oracle_news") / "us_macro_events_rows.csv"
    p.write_text(CALENDAR_HEADER + "".join(CALENDAR_ROWS), encoding="ascii")
    return z.read_news_csv(p)


def _wilder(tr: np.ndarray) -> np.ndarray:
    """Wilder ATR14 with ATR(before the first bar) = 2.00 (the oracle's atr14_rule)."""
    out = np.empty(tr.size)
    prev = ATR
    for i, x in enumerate(tr):
        prev = (13.0 * prev + x) / 14.0
        out[i] = prev
    return out


def run_case(case_id: str, news: z.NewsCalendar, ema: str, wide_tr: dict[int, float] | None = None) -> Run:
    """Load the oracle's bid/ask CSVs, prepend the flat warm-up, inject ATR14 and EMA30, simulate the cell.

    ema: "up" (EMA_k = 1950.00 + 0.10 k) or "down" (EMA_k = 2050.00 - 0.10 k). wide_tr: {case bar: true range}
    of the oracle's wide bars; every other bar has TR = 2.00, so ATR14 = 2.00 there."""
    case = z.load_m15_bidask(FIX / f"{case_id}_bid.csv", FIX / f"{case_id}_ask.csv")
    n = len(case)
    t0 = int(case["time"].iloc[0])
    assert t0 % 3600 == 0
    n_pre = WARMUP_DAYS * 96
    times = np.r_[t0 - M15 * np.arange(n_pre, 0, -1), case["time"].to_numpy(dtype=np.int64)]
    bid = pd.DataFrame({"time": times})
    ask = pd.DataFrame({"time": times})
    for col in ("open", "high", "low", "close"):
        bid[col] = np.r_[np.full(n_pre, case[f"bid_{col}"].iloc[0]), case[f"bid_{col}"].to_numpy()]
        ask[col] = np.r_[np.full(n_pre, case[f"ask_{col}"].iloc[0]), case[f"ask_{col}"].to_numpy()]
    frame = z.bidask_frame(bid, ask)

    # ATR14 (ORACLE section 2): the oracle's TR list must be the bars' true TR, then Wilder from 2.00.
    tr = np.full(n, 2.0)
    for i, v in (wide_tr or {}).items():
        tr[i] = v
    h, lo, c = (case[f"bid_{x}"].to_numpy() for x in ("high", "low", "close"))
    pc = np.r_[case["bid_open"].iloc[0], c[:-1]]
    true_tr = np.maximum(h - lo, np.maximum(np.abs(h - pc), np.abs(lo - pc)))
    np.testing.assert_allclose(true_tr, tr, rtol=0, atol=1e-9)
    atr = np.r_[np.full(n_pre, ATR), _wilder(tr)]

    # EMA30 per 1h bar of h1_from_m15 (D1): k-th hour of the case; NaN over the warm-up prefix.
    h1 = z.h1_from_m15(frame)
    k = (h1["time"].to_numpy(dtype=np.int64) - t0) // 3600
    slope = {"up": (1950.0, 0.10), "down": (2050.0, -0.10)}[ema]
    ema_h1 = np.where(k >= 0, slope[0] + slope[1] * k, np.nan)
    # the oracle's 1h close = the close of the M15 bar opening at HH:45 (complete hours of the case)
    full = (k >= 0) & (h1["n_m15"].to_numpy() == 4)
    hh45 = case.set_index("time")["bid_close"]
    for t, cl in zip(h1["time"].to_numpy()[full], h1["close"].to_numpy()[full]):
        assert cl == hh45[int(t) + 2700]

    prep = z.prepare(frame, news, test_indicators={"atr14": atr, "ema30_h1": ema_h1})
    rank = prep.close_rank[n_pre:]
    assert rank.min() >= z.WARMUP_TRADING_DAYS                       # D1 warm-up behind us
    assert np.all(prep.vol_median[rank] == ATR)                      # D18 median = the oracle's injected 2.0
    res = z.simulate(prep, z.ZenoConfig(CELL, 100_000.0))
    assert res.meta["indicators_injected"] == ["atr14", "ema30_h1"]
    assert res.meta["m1"] == z.M1_NOT_RUN
    return Run(prep, res, n_pre)


def _one(r: Run, time: str, side: str, event: str) -> pd.Series:
    d = r.res.decisions
    rows = d[(d["time"] == utc(time)) & (d["side"] == side) & (d["event"] == event)]
    assert len(rows) == 1, f"expected one {side} {event} at {time}, got {len(rows)}"
    return rows.iloc[0]


def _levels(row: pd.Series, side: str) -> tuple[float, float]:
    return (row["h_level"], row["l_level"]) if side == "long" else (row["l_level"], row["h_level"])


def assert_events(r: Run, expected: list[tuple[str, str, str]]) -> None:
    """The complete decisions table, in order: (bar close UTC, side, event)."""
    d = r.res.decisions
    got = [(int(t), s, e) for t, s, e in zip(d["time"], d["side"], d["event"])]
    assert got == [(utc(t), s, e) for t, s, e in expected]


def assert_armed(r: Run, time: str, side: str, *, extreme, opposite, leg, r50, void, ext_bar, pl_bar, atr) -> None:
    row = _one(r, time, side, "armed")
    assert _levels(row, side) == pytest.approx((extreme, opposite), abs=TOL)
    assert row["leg_usd"] == pytest.approx(leg, abs=TOL)
    assert row["retrace_level"] == pytest.approx(r50, abs=TOL)
    assert row["void_level"] == pytest.approx(void, abs=TOL)
    assert row["atr_arm"] == pytest.approx(atr, abs=TOL)
    assert (row["extreme_bar_time"], row["pullback_bar_time"]) == (utc(ext_bar), utc(pl_bar))


def assert_end(r: Run, time: str, side: str, event: str, *, ext_bar, pl_bar, bars_since_pullback) -> None:
    """A voided or expired setup: its H bar, its last pullback-extreme bar, bars since that bar."""
    row = _one(r, time, side, event)
    assert (row["extreme_bar_time"], row["pullback_bar_time"]) == (utc(ext_bar), utc(pl_bar))
    assert row["bars_since_pullback"] == bars_since_pullback


def assert_trigger(r: Run, time: str, side: str, status: str, reasons: tuple, *, extreme, opposite, pl_bar, atr,
                   spread, entry, stop, R, trend_ok) -> None:
    row = _one(r, time, side, "trigger")
    if status == "entered":
        assert (row["status"], row["reasons"]) == ("entered", "")
        assert row["position_id"] >= 1
    else:
        want = {REASONS[x] for x in reasons}
        assert set(row["reasons"].split(";")) == want
        assert row["status"] in want                                 # I6: which one is first is implementation-defined
        assert row["position_id"] == -1
    assert _levels(row, side) == pytest.approx((extreme, opposite), abs=TOL)
    assert row["pullback_bar_time"] == utc(pl_bar)
    assert row["entry_time"] == utc(time)                            # contiguous bars: the fill bar opens at the close
    assert row["atr_trigger"] == pytest.approx(atr, abs=TOL)
    assert row["spread_entry"] == pytest.approx(spread, abs=TOL)
    assert row["entry_price"] == pytest.approx(entry, abs=TOL)
    assert row["stop_level"] == pytest.approx(stop, abs=TOL)
    sign = 1 if side == "long" else -1
    assert sign * (row["entry_price"] - row["stop_level"]) == pytest.approx(R, abs=TOL)
    assert bool(row["trend_ok"]) is trend_ok


def assert_position(r: Run, pid: int, *, side, entry_time, entry, stop, R, balance, lots, partial, rest, tp1, tp2,
                    be, net, r_mult, outcome) -> None:
    p = r.res.positions
    row = p[p["position_id"] == pid].iloc[0]
    assert row["side"] == side
    assert row["entry_time"] == utc(entry_time)
    got = (row["entry_price"], row["stop_level"], row["R_usd_per_oz"], row["balance_at_entry"], row["lots"],
           row["partial_lots"], row["runner_lots"], row["tp1_level"], row["tp2_level"], row["be_level"],
           row["net_pnl_usd"], row["r_multiple_net"])
    want = (entry, stop, R, balance, lots, partial, rest, tp1, tp2, be, net, r_mult)
    assert got == pytest.approx(want, abs=TOL)
    assert row["outcome"] == outcome


def assert_legs(r: Run, expected: list[tuple]) -> None:
    """Every leg in order: (position_id, lots, exit bar open, exit stamp, price, oracle reason, gross, commission,
    net). Intrabar exits sit in TRADES at the bar open + 899 s and are stamped at its close (D22, [SI-14]); a time
    exit fills at the bar open and is stamped there (D17)."""
    lg = r.res.legs
    assert len(lg) == len(expected)
    stamps: dict[int, int] = {}
    for (_, row), (pid, lots, bar_open, stamp, price, reason, gross, comm, net) in zip(lg.iterrows(), expected):
        kind, leg = EXITS[reason]
        assert (row["position_id"], row["leg"], row["exit_reason"]) == (pid, leg, kind)
        assert row["units"] / 100.0 == pytest.approx(lots, abs=1e-9)
        assert int(row["exit_time"]) // M15 * M15 == utc(bar_open)
        assert int(row["exit_time"]) == (utc(stamp) if kind == "time" else utc(stamp) - 1)
        got = (row["exit_price"], row["pnl_usd"] + row["commission_usd"], row["commission_usd"], row["pnl_usd"])
        assert got == pytest.approx((price, gross, comm, net), abs=TOL)
        stamps[pid] = max(stamps.get(pid, 0), utc(stamp))
    p = r.res.positions
    for pid, stamp in stamps.items():
        assert int(p.loc[p["position_id"] == pid, "final_exit_stamp"].iloc[0]) == stamp


def assert_final(r: Run, balance: float) -> None:
    assert r.res.meta["final_balance_usd"] == pytest.approx(balance, abs=TOL)


def test_embedded_calendar_rows_are_the_real_files(news):
    """The embedded rows are the real calendar's rows for 2024-06-01 .. 2024-11-30, byte for byte."""
    assert news.times.size == len(CALENDAR_ROWS)
    if not REAL_CALENDAR.is_file():
        pytest.skip("the real news calendar is not mounted here")
    lines = REAL_CALENDAR.read_text(encoding="ascii").splitlines(keepends=True)
    assert lines[0] == CALENDAR_HEADER
    window = [x for x in lines[1:] if "2024-06-01" <= x.split(",")[1] <= "2024-11-30"]
    assert window == list(CALENDAR_ROWS)


# ---------------------------------------------------------------------------------------
# the long template (ORACLE section 4), all at ATR14 = 2.00:
#   flat bars 2001/2002/2000/2001; r1..r7 rise 2.00 each; T (H bar) 2015/2016/2014/2014.4; p1..p3 pull back;
#   p4 2009.6/2009.8/2007.8/2009.4 (arms, pullback-low bar); trigger 2009.4/2011.1/2009.1/2011.1.
#   H = 2016, L = 2000 (the flat low of the 20 bars before T), leg 16 >= 1.5 x 2.0 = 3.0;
#   50% = 2016 - 8 = 2008.00: p1-p3 lows 2012.6/2011.0/2009.4 stay above, p4's 2007.8 touches -> ARMED at p4's close;
#   void = 2016 - 0.786 x 16 = 2003.424, no close since T below it;
#   trigger: low 2009.1 > 2007.8 (no new low), close 2011.1 > p4's high 2009.8, 1 bar after p4 -> TRIGGER;
#   entry = next ask open 2011.10 + 0.20 = 2011.30; stop = 2007.80 - 0.25 x 2.0 = 2007.30; R = 4.00 USD/oz;
#   spread 0.20 <= 0.40, R 4.00 <= 6.0, ATR 2.0 <= 2 x 2.0; lots = floor(500 / 400) = 1.25 -> 0.62 + 0.63;
#   +2R 2019.30, +4R 2027.30, breakeven 2011.30 + 0.10 = 2011.40; a full stop fills 2007.25 (slippage 0.05):
#   -4.05 x 125 = -506.25, commission 12.50, net -518.75 = -1.0375 R.

def test_c01_long_winner_2r_then_4r(news):
    """C01: +2R partial on bar 41 (bar 40's high 2019.1 misses 2019.30), +4R on bar 45 (bars 42-44 top at 2027.1)."""
    d = "2024-07-16"
    r = run_case("C01_long_winner_2R_4R", news, "up")
    assert_events(r, [(f"{d} 13:00", "long", "armed"), (f"{d} 13:15", "long", "trigger")])
    assert_armed(r, f"{d} 13:00", "long", extreme=2016.0, opposite=2000.0, leg=16.0, r50=2008.0, void=2003.424,
                 ext_bar=f"{d} 11:45", pl_bar=f"{d} 12:45", atr=2.0)
    # trend (D3/D4): 1h bar 12:00-13:00 closes 2009.4 > EMA_8 1950.80, and EMA_8 > EMA_3 1950.30
    assert_trigger(r, f"{d} 13:15", "long", "entered", (), extreme=2016.0, opposite=2000.0, pl_bar=f"{d} 12:45",
                   atr=2.0, spread=0.20, entry=2011.30, stop=2007.30, R=4.00, trend_ok=True)
    # leg 1: 8.00 x 62 = 496.00 - 6.20 = 489.80; leg 2: 16.00 x 63 = 1008.00 - 6.30 = 1001.70; 1491.50 / 500 = 2.983 R
    assert_position(r, 1, side="long", entry_time=f"{d} 13:15", entry=2011.30, stop=2007.30, R=4.00,
                    balance=100_000.0, lots=1.25, partial=0.62, rest=0.63, tp1=2019.30, tp2=2027.30, be=2011.40,
                    net=1491.50, r_mult=2.983, outcome="+3R")
    assert_legs(r, [(1, 0.62, f"{d} 14:15", f"{d} 14:30", 2019.30, "tp1_+2R", 496.00, 6.20, 489.80),
                    (1, 0.63, f"{d} 15:15", f"{d} 15:30", 2027.30, "tp2_+4R", 1008.00, 6.30, 1001.70)])
    assert_final(r, 101_491.50)


def test_c02_long_2r_then_breakeven_on_a_later_bar(news):
    """C02: bar 41 (high 2019.5, low 2017.5) takes +2R; bar 45's low 2010.4 hits breakeven 2011.40 -> 2011.35."""
    d = "2024-07-17"
    r = run_case("C02_long_2R_then_BE", news, "up")
    # after bar 41 the window's H is bar 41 (2019.5), L 2000, 50% 2009.75; lows 2010.4 and 2010.0 never touch it
    assert_events(r, [(f"{d} 13:00", "long", "armed"), (f"{d} 13:15", "long", "trigger")])
    assert_armed(r, f"{d} 13:00", "long", extreme=2016.0, opposite=2000.0, leg=16.0, r50=2008.0, void=2003.424,
                 ext_bar=f"{d} 11:45", pl_bar=f"{d} 12:45", atr=2.0)
    assert_trigger(r, f"{d} 13:15", "long", "entered", (), extreme=2016.0, opposite=2000.0, pl_bar=f"{d} 12:45",
                   atr=2.0, spread=0.20, entry=2011.30, stop=2007.30, R=4.00, trend_ok=True)
    # leg 2: (2011.35 - 2011.30) x 63 = 3.15 - 6.30 = -3.15; net 489.80 - 3.15 = 486.65 = 0.9733 R (I4: BE slips 0.05)
    assert_position(r, 1, side="long", entry_time=f"{d} 13:15", entry=2011.30, stop=2007.30, R=4.00,
                    balance=100_000.0, lots=1.25, partial=0.62, rest=0.63, tp1=2019.30, tp2=2027.30, be=2011.40,
                    net=486.65, r_mult=0.9733, outcome="+1R(BE)")
    assert_legs(r, [(1, 0.62, f"{d} 14:15", f"{d} 14:30", 2019.30, "tp1_+2R", 496.00, 6.20, 489.80),
                    (1, 0.63, f"{d} 15:15", f"{d} 15:30", 2011.35, "breakeven", 3.15, 6.30, -3.15)])
    assert_final(r, 100_486.65)


def test_c03_long_full_stop(news):
    """C03: bar 38's low 2007.1 <= 2007.30 -> all 1.25 lots at 2007.25, stamped 13:45; no re-arm on the used H (I1)."""
    d = "2024-07-18"
    r = run_case("C03_long_full_stop", news, "up")
    assert_events(r, [(f"{d} 13:00", "long", "armed"), (f"{d} 13:15", "long", "trigger")])
    assert_armed(r, f"{d} 13:00", "long", extreme=2016.0, opposite=2000.0, leg=16.0, r50=2008.0, void=2003.424,
                 ext_bar=f"{d} 11:45", pl_bar=f"{d} 12:45", atr=2.0)
    assert_trigger(r, f"{d} 13:15", "long", "entered", (), extreme=2016.0, opposite=2000.0, pl_bar=f"{d} 12:45",
                   atr=2.0, spread=0.20, entry=2011.30, stop=2007.30, R=4.00, trend_ok=True)
    assert_position(r, 1, side="long", entry_time=f"{d} 13:15", entry=2011.30, stop=2007.30, R=4.00,
                    balance=100_000.0, lots=1.25, partial=0.62, rest=0.63, tp1=2019.30, tp2=2027.30, be=2011.40,
                    net=-518.75, r_mult=-1.0375, outcome="-1R")
    assert_legs(r, [(1, 1.25, f"{d} 13:30", f"{d} 13:45", 2007.25, "stop", -506.25, 12.50, -518.75)])
    assert_final(r, 99_481.25)


def test_c04a_short_stop_includes_spread_and_triggers_on_the_ask(news):
    """C04a: the long template mirrored (p -> 4000 - p). Short L = 1984 (bar 31), H = 2000; 50% = 1992.00 touched by
    q4's bid high 1992.2 -> ARMED 13:00; pullback-high bar 35 (high 1992.2, low 1990.2); bar 36 closes 1988.9 < 1990.2
    -> TRIGGER 13:15. Entry bar 37: bid open 1988.90 (the fill), ask open 1989.20, spread 0.30 (D11);
    stop = 1992.20 + 0.25 x 2.0 + 0.30 = 1993.00 (rule 5), R = 4.10; lots floor(500 / 410) = 1.21 -> 0.60 + 0.61.
    Bar 39 (spread 0.40): ask high 1992.65 + 0.40 = 1993.05 >= 1993.00 -> STOP at 1993.05 (D14: the ask), stamped
    14:00. Neither the bid high 1992.65 nor bid + entry spread 1992.95 reaches the stop."""
    d = "2024-07-18"
    r = run_case("C04a_short_stop_on_ask", news, "down")
    assert_events(r, [(f"{d} 13:00", "short", "armed"), (f"{d} 13:15", "short", "trigger")])
    assert_armed(r, f"{d} 13:00", "short", extreme=1984.0, opposite=2000.0, leg=16.0, r50=1992.0, void=1996.576,
                 ext_bar=f"{d} 11:45", pl_bar=f"{d} 12:45", atr=2.0)
    # ORACLE CORRECTED: cases.json gives this trigger row H = -1984.0, L = -2000.0. Spec rule 3 ("shorts mirror") as
    # the brief spells it: "L = lowest low of last 20, H = highest high of the 20 bars before L", so L = 1984.00 and
    # H = 2000.00 USD/oz, the same values as the armed row (prices cannot be negative). The oracle's build_oracle.py
    # flips the short's H/L back from the negated machine once for all events and a second time for trigger rows.
    # short trend: 1h close 12:00-13:00 = 1990.6 < EMA_8 2049.20, and EMA_8 < EMA_3 2049.70
    assert_trigger(r, f"{d} 13:15", "short", "entered", (), extreme=1984.0, opposite=2000.0, pl_bar=f"{d} 12:45",
                   atr=2.0, spread=0.30, entry=1988.90, stop=1993.00, R=4.10, trend_ok=True)
    # -4.15 x 121 = -502.15, commission 12.10, net -514.25; R x units = 4.10 x 121 = 496.10 -> -1.036585 R
    assert_position(r, 1, side="short", entry_time=f"{d} 13:15", entry=1988.90, stop=1993.00, R=4.10,
                    balance=100_000.0, lots=1.21, partial=0.60, rest=0.61, tp1=1980.70, tp2=1972.50, be=1988.80,
                    net=-514.25, r_mult=-514.25 / 496.10, outcome="-1R")
    assert_legs(r, [(1, 1.21, f"{d} 13:45", f"{d} 14:00", 1993.05, "stop", -502.15, 12.10, -514.25)])
    assert_final(r, 99_485.75)


def test_c04b_short_targets_trigger_on_the_ask(news):
    """C04b: entry as C04a. Bar 41: bid low 1980.60 <= 1980.70 but ask low 1980.80 > 1980.70 -> no fill; bar 42's
    ask low 1979.20 -> 0.60 at 1980.70 (14:45). Bar 46 (spread 0.40): bid low 1972.20 but ask low 1972.60 > 1972.50
    -> no fill; bar 47's ask low 1971.20 -> 0.61 at 1972.50 (16:00). A bid-based check would exit one bar early."""
    d = "2024-07-19"
    r = run_case("C04b_short_winner_targets_on_ask", news, "down")
    assert_events(r, [(f"{d} 13:00", "short", "armed"), (f"{d} 13:15", "short", "trigger")])
    assert_armed(r, f"{d} 13:00", "short", extreme=1984.0, opposite=2000.0, leg=16.0, r50=1992.0, void=1996.576,
                 ext_bar=f"{d} 11:45", pl_bar=f"{d} 12:45", atr=2.0)
    # ORACLE CORRECTED: cases.json gives H = -1984.0, L = -2000.0 here; spec rule 3 short mirror -> L = 1984.00,
    # H = 2000.00 USD/oz (see C04a).
    assert_trigger(r, f"{d} 13:15", "short", "entered", (), extreme=1984.0, opposite=2000.0, pl_bar=f"{d} 12:45",
                   atr=2.0, spread=0.30, entry=1988.90, stop=1993.00, R=4.10, trend_ok=True)
    # leg 1: 8.20 x 60 = 492.00 - 6.00 = 486.00; leg 2: 16.40 x 61 = 1000.40 - 6.10 = 994.30; 1480.30 / 496.10 R
    assert_position(r, 1, side="short", entry_time=f"{d} 13:15", entry=1988.90, stop=1993.00, R=4.10,
                    balance=100_000.0, lots=1.21, partial=0.60, rest=0.61, tp1=1980.70, tp2=1972.50, be=1988.80,
                    net=1480.30, r_mult=1480.30 / 496.10, outcome="+3R")
    assert_legs(r, [(1, 0.60, f"{d} 14:30", f"{d} 14:45", 1980.70, "tp1_+2R", 492.00, 6.00, 486.00),
                    (1, 0.61, f"{d} 15:45", f"{d} 16:00", 1972.50, "tp2_+4R", 1000.40, 6.10, 994.30)])
    assert_final(r, 101_480.30)


def test_c05_void_on_a_close_beyond_78_6(news):
    """C05: armed 13:00; bars 36-38 make new lows (pullback-low bar moves to 38, low 2004.0) with closes above
    2003.424; bar 39 closes 2003.2 < 2003.424 -> VOIDED at 14:00 (D6). Bar 40 would have been a trigger (close
    2005.2 > bar 39's high 2004.2) but the setup is dead; H stays in the window, so nothing re-arms (D8)."""
    d = "2024-07-16"
    r = run_case("C05_void_close_beyond_786", news, "up")
    assert_events(r, [(f"{d} 13:00", "long", "armed"), (f"{d} 14:00", "long", "voided")])
    assert_armed(r, f"{d} 13:00", "long", extreme=2016.0, opposite=2000.0, leg=16.0, r50=2008.0, void=2003.424,
                 ext_bar=f"{d} 11:45", pl_bar=f"{d} 12:45", atr=2.0)
    assert_end(r, f"{d} 14:00", "long", "voided", ext_bar=f"{d} 11:45", pl_bar=f"{d} 13:30", bars_since_pullback=1)
    assert len(r.res.positions) == 0
    assert_final(r, 100_000.0)


def test_c06_new_low_restarts_the_count_then_expiry(news):
    """C06: armed 13:00 (pullback-low bar 35, low 2007.8). Bar 37's low 2007.5 < 2007.8 moves the pullback-low bar
    (high 2009.5) and restarts the count (D9). Bars 38-45 (+1..+8): lows above 2007.5, closes at or below 2009.5 ->
    EXPIRED at bar 45's close 15:30 (I3). Bar 46 closes 2010.6 > 2009.5 but is bar +9: not a trigger. Without the
    restart the count would have expired at 15:00."""
    d = "2024-07-17"
    r = run_case("C06_pl_update_restart_and_expiry", news, "up")
    assert_events(r, [(f"{d} 13:00", "long", "armed"), (f"{d} 15:30", "long", "expired")])
    assert_armed(r, f"{d} 13:00", "long", extreme=2016.0, opposite=2000.0, leg=16.0, r50=2008.0, void=2003.424,
                 ext_bar=f"{d} 11:45", pl_bar=f"{d} 12:45", atr=2.0)
    assert_end(r, f"{d} 15:30", "long", "expired", ext_bar=f"{d} 11:45", pl_bar=f"{d} 13:15", bars_since_pullback=8)
    assert len(r.res.positions) == 0
    assert_final(r, 100_000.0)


def test_c07_session_block_uses_up_the_setup(news):
    """C07: T = bar 34 (08:30), armed 09:45; trigger at bar 39's close 10:00 UTC = 18:00 SGT, outside [07:00, 10:00)
    and [12:30, 16:00) UTC -> BLOCKED: session (D19), the only reason. Trend passes at the D3 edge: the 1h bar
    09:00-10:00 closes exactly at 10:00 (its close = the trigger close 2011.1 > EMA_9 1950.90 > EMA_4 1950.40).
    Bars 40 and 41 close above 2009.8 within 8 bars but are not chased (D10)."""
    d = "2024-07-18"
    r = run_case("C07_session_block_consumed", news, "up")
    assert_events(r, [(f"{d} 09:45", "long", "armed"), (f"{d} 10:00", "long", "trigger")])
    assert_armed(r, f"{d} 09:45", "long", extreme=2016.0, opposite=2000.0, leg=16.0, r50=2008.0, void=2003.424,
                 ext_bar=f"{d} 08:30", pl_bar=f"{d} 09:30", atr=2.0)
    assert_trigger(r, f"{d} 10:00", "long", "blocked", ("session",), extreme=2016.0, opposite=2000.0,
                   pl_bar=f"{d} 09:30", atr=2.0, spread=0.20, entry=2011.30, stop=2007.30, R=4.00, trend_ok=True)
    assert len(r.res.positions) == 0
    assert_final(r, 100_000.0)


def test_c08_news_blackout_from_a_real_calendar_row(news):
    """C08: real row CPI,2024-07-11,08:30,-0400,2024-07-11T12:30Z -> blackout [12:00, 13:30] UTC, both ends in (D20).
    Armed 13:15 (T = bar 32, 12:00); trigger at bar 37's close 13:30 = T + 60 min -> BLOCKED: news (CPI), the only
    reason; session [12:30, 16:00) and trend (2009.6 > EMA_8 1950.80 > EMA_3 1950.30) pass. Bars 38-39 not chased."""
    d = "2024-07-11"
    r = run_case("C08_news_blackout_CPI", news, "up")
    assert_events(r, [(f"{d} 13:15", "long", "armed"), (f"{d} 13:30", "long", "trigger")])
    assert_armed(r, f"{d} 13:15", "long", extreme=2016.0, opposite=2000.0, leg=16.0, r50=2008.0, void=2003.424,
                 ext_bar=f"{d} 12:00", pl_bar=f"{d} 13:00", atr=2.0)
    # the oracle's reason text "news:CPI 2024-07-11T12:30Z" is zeno_v1's news_blackout (the log names the filter)
    assert_trigger(r, f"{d} 13:30", "long", "blocked", ("news",), extreme=2016.0, opposite=2000.0,
                   pl_bar=f"{d} 13:00", atr=2.0, spread=0.20, entry=2011.30, stop=2007.30, R=4.00, trend_ok=True)
    # the edges of that one event: T - 30 and T + 60 blocked, T - 31 and T + 61 not ("T + 61 would be allowed")
    blocked = r.prep.news.blocked
    assert (blocked(utc(f"{d} 12:00")), blocked(utc(f"{d} 13:30"))) == (True, True)
    assert (blocked(utc(f"{d} 11:59")), blocked(utc(f"{d} 13:31"))) == (False, False)
    assert len(r.res.positions) == 0
    assert_final(r, 100_000.0)


def test_c09a_time_exit_1630_new_york_in_us_summer_time(news):
    """C09a: 2024-10-31, the US still on EDT (UTC-4) until 2024-11-03, the EU already changed. Trigger 15:00 UTC,
    entered. Grind: bar 43 + j opens 2016.3 + 0.10 j, high + 0.32 (max 2018.42 < 2019.30), low - 1.68 (> 2007.30).
    16:30 New York = 20:30 UTC: bar 62 opens at 20:30 -> close at its bid open 2016.3 + 1.9 = 2018.20, no slippage
    (D17). 6.90 x 125 = 862.50 - 12.50 = 850.00 = 1.70 R."""
    d = "2024-10-31"
    r = run_case("C09a_time_exit_summer", news, "up")
    assert_events(r, [(f"{d} 14:45", "long", "armed"), (f"{d} 15:00", "long", "trigger")])
    assert_armed(r, f"{d} 14:45", "long", extreme=2016.0, opposite=2000.0, leg=16.0, r50=2008.0, void=2003.424,
                 ext_bar=f"{d} 13:30", pl_bar=f"{d} 14:30", atr=2.0)
    assert_trigger(r, f"{d} 15:00", "long", "entered", (), extreme=2016.0, opposite=2000.0, pl_bar=f"{d} 14:30",
                   atr=2.0, spread=0.20, entry=2011.30, stop=2007.30, R=4.00, trend_ok=True)
    assert_position(r, 1, side="long", entry_time=f"{d} 15:00", entry=2011.30, stop=2007.30, R=4.00,
                    balance=100_000.0, lots=1.25, partial=0.62, rest=0.63, tp1=2019.30, tp2=2027.30, be=2011.40,
                    net=850.00, r_mult=1.70, outcome="time-exit")
    assert_legs(r, [(1, 1.25, f"{d} 20:30", f"{d} 20:30", 2018.20, "time_exit_1630NY", 862.50, 12.50, 850.00)])
    assert_final(r, 100_850.00)


def test_c09b_time_exit_1630_new_york_in_us_winter_time(news):
    """C09b: 2024-11-04 (EST, UTC-5). The 20:30 UTC bar is 15:30 New York and does NOT close the trade; 16:30 New
    York = 21:30 UTC: bar 66 opens 2016.3 + 2.3 = 2018.60 (highest high before it 2018.82 < 2019.30).
    7.30 x 125 = 912.50 - 12.50 = 900.00 = 1.80 R; 17:00 New York (22:00 UTC) is not crossed."""
    d = "2024-11-04"
    r = run_case("C09b_time_exit_winter", news, "up")
    assert_events(r, [(f"{d} 14:45", "long", "armed"), (f"{d} 15:00", "long", "trigger")])
    assert_armed(r, f"{d} 14:45", "long", extreme=2016.0, opposite=2000.0, leg=16.0, r50=2008.0, void=2003.424,
                 ext_bar=f"{d} 13:30", pl_bar=f"{d} 14:30", atr=2.0)
    assert_trigger(r, f"{d} 15:00", "long", "entered", (), extreme=2016.0, opposite=2000.0, pl_bar=f"{d} 14:30",
                   atr=2.0, spread=0.20, entry=2011.30, stop=2007.30, R=4.00, trend_ok=True)
    assert_position(r, 1, side="long", entry_time=f"{d} 15:00", entry=2011.30, stop=2007.30, R=4.00,
                    balance=100_000.0, lots=1.25, partial=0.62, rest=0.63, tp1=2019.30, tp2=2027.30, be=2011.40,
                    net=900.00, r_mult=1.80, outcome="time-exit")
    assert_legs(r, [(1, 1.25, f"{d} 21:30", f"{d} 21:30", 2018.60, "time_exit_1630NY", 912.50, 12.50, 900.00)])
    assert_final(r, 100_900.00)


def test_c10a_one_bar_touches_stop_and_2r_stop_first(news):
    """C10a: bar 38 = 2011.5 / 2019.5 / 2007.0 / 2012.0 touches the stop 2007.30 and +2R 2019.30 -> stop first
    (D15): all 1.25 lots at 2007.25, stamped 13:45, no partial. Its TR 12.5 gives ATR (13 x 2 + 12.5) / 14 = 2.75."""
    d = "2024-07-22"
    r = run_case("C10a_ambiguous_stop_first", news, "up", wide_tr={38: 12.5})
    assert r.prep.atr[r.n_pre + 38] == pytest.approx(2.75, abs=TOL)
    assert r.prep.atr[r.n_pre + 39] == pytest.approx(2.696429, abs=TOL)
    assert_events(r, [(f"{d} 13:00", "long", "armed"), (f"{d} 13:15", "long", "trigger")])
    assert_armed(r, f"{d} 13:00", "long", extreme=2016.0, opposite=2000.0, leg=16.0, r50=2008.0, void=2003.424,
                 ext_bar=f"{d} 11:45", pl_bar=f"{d} 12:45", atr=2.0)
    assert_trigger(r, f"{d} 13:15", "long", "entered", (), extreme=2016.0, opposite=2000.0, pl_bar=f"{d} 12:45",
                   atr=2.0, spread=0.20, entry=2011.30, stop=2007.30, R=4.00, trend_ok=True)
    assert_position(r, 1, side="long", entry_time=f"{d} 13:15", entry=2011.30, stop=2007.30, R=4.00,
                    balance=100_000.0, lots=1.25, partial=0.62, rest=0.63, tp1=2019.30, tp2=2027.30, be=2011.40,
                    net=-518.75, r_mult=-1.0375, outcome="-1R")
    assert bool(r.res.positions["ambiguous_bar"].iloc[0])
    assert_legs(r, [(1, 1.25, f"{d} 13:30", f"{d} 13:45", 2007.25, "stop", -506.25, 12.50, -518.75)])
    assert_final(r, 99_481.25)


def test_c10b_2r_and_breakeven_in_one_bar(news):
    """C10b: bar 38 = 2011.5 / 2019.5 / 2010.0 / 2012.0: the stop 2007.30 is not touched; the high takes +2R (0.62
    at 2019.30) and the same bar's low 2010.0 <= breakeven 2011.40 closes 0.63 at 2011.35 (D15), both stamped 13:45.
    489.80 - 3.15 = 486.65. TR 9.5 -> ATR 35.5 / 14 = 2.535714."""
    d = "2024-07-22"
    r = run_case("C10b_same_bar_2R_then_BE", news, "up", wide_tr={38: 9.5})
    assert r.prep.atr[r.n_pre + 38] == pytest.approx(2.535714, abs=TOL)
    assert r.prep.atr[r.n_pre + 39] == pytest.approx(2.497449, abs=TOL)
    assert_events(r, [(f"{d} 13:00", "long", "armed"), (f"{d} 13:15", "long", "trigger")])
    assert_armed(r, f"{d} 13:00", "long", extreme=2016.0, opposite=2000.0, leg=16.0, r50=2008.0, void=2003.424,
                 ext_bar=f"{d} 11:45", pl_bar=f"{d} 12:45", atr=2.0)
    assert_trigger(r, f"{d} 13:15", "long", "entered", (), extreme=2016.0, opposite=2000.0, pl_bar=f"{d} 12:45",
                   atr=2.0, spread=0.20, entry=2011.30, stop=2007.30, R=4.00, trend_ok=True)
    assert_position(r, 1, side="long", entry_time=f"{d} 13:15", entry=2011.30, stop=2007.30, R=4.00,
                    balance=100_000.0, lots=1.25, partial=0.62, rest=0.63, tp1=2019.30, tp2=2027.30, be=2011.40,
                    net=486.65, r_mult=0.9733, outcome="+1R(BE)")
    assert_legs(r, [(1, 0.62, f"{d} 13:30", f"{d} 13:45", 2019.30, "tp1_+2R", 496.00, 6.20, 489.80),
                    (1, 0.63, f"{d} 13:30", f"{d} 13:45", 2011.35, "breakeven", 3.15, 6.30, -3.15)])
    assert_final(r, 100_486.65)


def test_c11_daily_limits_two_losses_and_minus_1pct_on_the_server_day(news):
    """C11: server day 2024-07-24 = 2024-07-23 21:00 UTC .. 2024-07-24 21:00 UTC (17:00 New York EDT, D21).
    A (template, T bar 33): armed 09:30, trigger 09:45 entered, stop on bar 40 -> -518.75 (balance 99,481.25).
    B: new high T_B bar 45 (2017.3); L = lowest low of bars 25-44 = 2000.0; leg 17.3; 50% 2008.65;
    void 2017.3 - 0.786 x 17.3 = 2003.7022; bar 49's low 2008.1 arms 12:30; bar 50 closes 2011.4 > 2010.1 -> 12:45
    entered (1 entry, 1 loss, -518.75 > -1,000; cooldown from 10:15 + 15 min passed). E = 2011.40 + 0.20 = 2011.60,
    stop 2008.10 - 0.50 = 2007.60, R 4.00, lots floor(497.40625 / 400 = 1.2435) = 1.24; stop bar 52 -> 2007.55:
    -4.05 x 124 = -502.20 - 12.40 = -514.60 (balance 98,966.65; day -1,033.35 = -1.03335%).
    C: T_C bar 57 (2017.6), L = bars 37-56 lowest low 2007.1 (bar 40), leg 10.5, 50% 2012.35, void 2009.347; arms
    15:00; trigger 15:15 -> BLOCKED by all three daily limits (2 entries, 2 losses, -1.0%); everything else passes
    (would-be E 2015.30, stop 2011.30, R 4.00).
    D: next server day 2024-07-25 (from 2024-07-24 21:00 UTC); template from bar 111, armed 06:45, trigger 07:00 UTC
    = 15:00 SGT (inclusive start of the window) -> entered; trend 2011.1 > EMA_30 1953.00 > EMA_25 1952.50;
    lots floor(494.83325 / 400 = 1.237) = 1.23; stop bar 125 -> -4.05 x 123 = -498.15 - 12.30 = -510.45."""
    d1, d2 = "2024-07-24", "2024-07-25"
    r = run_case("C11_daily_limits_two_losses", news, "up")
    assert_events(r, [(f"{d1} 09:30", "long", "armed"), (f"{d1} 09:45", "long", "trigger"),
                      (f"{d1} 12:30", "long", "armed"), (f"{d1} 12:45", "long", "trigger"),
                      (f"{d1} 15:00", "long", "armed"), (f"{d1} 15:15", "long", "trigger"),
                      (f"{d2} 06:45", "long", "armed"), (f"{d2} 07:00", "long", "trigger")])
    assert_armed(r, f"{d1} 09:30", "long", extreme=2016.0, opposite=2000.0, leg=16.0, r50=2008.0, void=2003.424,
                 ext_bar=f"{d1} 08:15", pl_bar=f"{d1} 09:15", atr=2.0)
    assert_trigger(r, f"{d1} 09:45", "long", "entered", (), extreme=2016.0, opposite=2000.0, pl_bar=f"{d1} 09:15",
                   atr=2.0, spread=0.20, entry=2011.30, stop=2007.30, R=4.00, trend_ok=True)
    assert_armed(r, f"{d1} 12:30", "long", extreme=2017.3, opposite=2000.0, leg=17.3, r50=2008.65, void=2003.7022,
                 ext_bar=f"{d1} 11:15", pl_bar=f"{d1} 12:15", atr=2.0)
    assert_trigger(r, f"{d1} 12:45", "long", "entered", (), extreme=2017.3, opposite=2000.0, pl_bar=f"{d1} 12:15",
                   atr=2.0, spread=0.20, entry=2011.60, stop=2007.60, R=4.00, trend_ok=True)
    assert_armed(r, f"{d1} 15:00", "long", extreme=2017.6, opposite=2007.1, leg=10.5, r50=2012.35, void=2009.347,
                 ext_bar=f"{d1} 14:15", pl_bar=f"{d1} 14:45", atr=2.0)
    assert_trigger(r, f"{d1} 15:15", "long", "blocked", ("daily_max_2_entries", "daily_2_losses", "daily_loss_1pct"),
                   extreme=2017.6, opposite=2007.1, pl_bar=f"{d1} 14:45", atr=2.0, spread=0.20, entry=2015.30,
                   stop=2011.30, R=4.00, trend_ok=True)
    assert_armed(r, f"{d2} 06:45", "long", extreme=2016.0, opposite=2000.0, leg=16.0, r50=2008.0, void=2003.424,
                 ext_bar=f"{d2} 05:30", pl_bar=f"{d2} 06:30", atr=2.0)
    assert_trigger(r, f"{d2} 07:00", "long", "entered", (), extreme=2016.0, opposite=2000.0, pl_bar=f"{d2} 06:30",
                   atr=2.0, spread=0.20, entry=2011.30, stop=2007.30, R=4.00, trend_ok=True)
    assert_position(r, 1, side="long", entry_time=f"{d1} 09:45", entry=2011.30, stop=2007.30, R=4.00,
                    balance=100_000.0, lots=1.25, partial=0.62, rest=0.63, tp1=2019.30, tp2=2027.30, be=2011.40,
                    net=-518.75, r_mult=-1.0375, outcome="-1R")
    assert_position(r, 2, side="long", entry_time=f"{d1} 12:45", entry=2011.60, stop=2007.60, R=4.00,
                    balance=99_481.25, lots=1.24, partial=0.62, rest=0.62, tp1=2019.60, tp2=2027.60, be=2011.70,
                    net=-514.60, r_mult=-1.0375, outcome="-1R")
    assert_position(r, 3, side="long", entry_time=f"{d2} 07:00", entry=2011.30, stop=2007.30, R=4.00,
                    balance=98_966.65, lots=1.23, partial=0.61, rest=0.62, tp1=2019.30, tp2=2027.30, be=2011.40,
                    net=-510.45, r_mult=-1.0375, outcome="-1R")
    assert r.res.positions["server_day"].tolist() == [d1, d1, d2]
    assert_legs(r, [(1, 1.25, f"{d1} 10:00", f"{d1} 10:15", 2007.25, "stop", -506.25, 12.50, -518.75),
                    (2, 1.24, f"{d1} 13:00", f"{d1} 13:15", 2007.55, "stop", -502.20, 12.40, -514.60),
                    (3, 1.23, f"{d2} 07:15", f"{d2} 07:30", 2007.25, "stop", -498.15, 12.30, -510.45)])
    assert_final(r, 98_456.20)


def test_c12_cooldown_blocks_a_trigger_on_the_exit_bar(news):
    """C12 (I1: a used H cannot arm again; a different, lower H can).
    S1: H1 = bar 32 (2011.0), L1 = lowest low of bars 12-31 = 1990.3, leg 20.7, 50% 2000.65, void 1994.7298; bar 41's
    low 2000.6 arms 10:30; the pullback-low bar moves to 43, 45, then 49 (2000.0, high 2002.0); bar 51 closes
    2002.3 > 2002.0 -> 13:00 entered. E = 2002.30 + 0.20 = 2002.50, stop 2000.0 - 0.50 = 1999.50, R 3.00,
    lots floor(500 / 300 = 1.667) = 1.66 -> 0.83 + 0.83; +2R 2008.50, +4R 2014.50, BE 2002.60.
    S2: from bar 52 H1 has left the window; H2 = bar 37 (2009.2), L2 = bars 17-36 = 1990.3, leg 18.9, 50% 1999.75,
    void 1994.3446; bar 53's low 1999.65 arms 13:30 (pullback-low bar 53, high 2001.65).
    Bar 54 = 2001.0 / 2008.6 / 2000.6 / 2001.7: the trade takes +2R (0.83 at 2008.50) and its low 2000.6 <= 2002.60
    closes 0.83 at 2002.55, both stamped 13:45 (D15, D22); S2's close 2001.7 > 2001.65 -> trigger 13:45 -> BLOCKED:
    cooldown (next same-direction entry from 14:00), the only reason; ATR(54) = (13 x 2 + 8.0) / 14 = 34 / 14,
    would-be stop = 1999.65 - 0.25 x 34 / 14 = 1999.042857, R = 2001.90 - 1999.042857 = 2.857143.
    Short (I7): L_s = 1999.65 (bar 53), H_s = highest high of bars 33-52 = 2009.2, leg 9.55 >= 1.5 x 34 / 14; 50%
    2004.425 reached by bar 54's high 2008.6 -> armed 13:45; void 1999.65 + 0.786 x 9.55 = 2007.1563; no trigger."""
    d = "2024-07-23"
    atr54 = 34.0 / 14.0
    r = run_case("C12_cooldown_block_same_bar", news, "up", wide_tr={54: 8.0})
    assert r.prep.atr[r.n_pre + 54] == pytest.approx(atr54, abs=1e-12)
    assert (r.prep.atr[r.n_pre + 55], r.prep.atr[r.n_pre + 56]) == pytest.approx((2.397959, 2.369534), abs=TOL)
    assert_events(r, [(f"{d} 10:30", "long", "armed"), (f"{d} 13:00", "long", "trigger"),
                      (f"{d} 13:30", "long", "armed"), (f"{d} 13:45", "long", "trigger"),
                      (f"{d} 13:45", "short", "armed")])
    assert_armed(r, f"{d} 10:30", "long", extreme=2011.0, opposite=1990.3, leg=20.7, r50=2000.65, void=1994.7298,
                 ext_bar=f"{d} 08:00", pl_bar=f"{d} 10:15", atr=2.0)
    assert_trigger(r, f"{d} 13:00", "long", "entered", (), extreme=2011.0, opposite=1990.3, pl_bar=f"{d} 12:15",
                   atr=2.0, spread=0.20, entry=2002.50, stop=1999.50, R=3.00, trend_ok=True)
    assert_armed(r, f"{d} 13:30", "long", extreme=2009.2, opposite=1990.3, leg=18.9, r50=1999.75, void=1994.3446,
                 ext_bar=f"{d} 09:15", pl_bar=f"{d} 13:15", atr=2.0)
    assert_trigger(r, f"{d} 13:45", "long", "blocked", ("cooldown_15min",), extreme=2009.2, opposite=1990.3,
                   pl_bar=f"{d} 13:15", atr=atr54, spread=0.20, entry=2001.90, stop=1999.65 - 0.25 * atr54,
                   R=2001.90 - (1999.65 - 0.25 * atr54), trend_ok=True)
    assert_armed(r, f"{d} 13:45", "short", extreme=1999.65, opposite=2009.2, leg=9.55, r50=2004.425, void=2007.1563,
                 ext_bar=f"{d} 13:15", pl_bar=f"{d} 13:30", atr=atr54)
    # leg 1: 6.00 x 83 = 498.00 - 8.30 = 489.70; leg 2: 0.05 x 83 = 4.15 - 8.30 = -4.15; 485.55 / 498 = 0.975 R
    assert_position(r, 1, side="long", entry_time=f"{d} 13:00", entry=2002.50, stop=1999.50, R=3.00,
                    balance=100_000.0, lots=1.66, partial=0.83, rest=0.83, tp1=2008.50, tp2=2014.50, be=2002.60,
                    net=485.55, r_mult=0.975, outcome="+1R(BE)")
    assert_legs(r, [(1, 0.83, f"{d} 13:30", f"{d} 13:45", 2008.50, "tp1_+2R", 498.00, 8.30, 489.70),
                    (1, 0.83, f"{d} 13:30", f"{d} 13:45", 2002.55, "breakeven", 4.15, 8.30, -4.15)])
    assert_final(r, 100_485.55)


def test_c13_spread_above_10pct_of_r_is_skipped(news):
    """C13: the entry bar's spread is 0.45: E = 2011.10 + 0.45 = 2011.55, stop 2007.30, R = 4.25; 10% of R = 0.425
    < 0.45 -> BLOCKED: spread (rule 10, D11), the only reason (R 4.25 <= 6.0; trend and session pass)."""
    d = "2024-07-25"
    r = run_case("C13_spread_filter", news, "up")
    assert_events(r, [(f"{d} 13:00", "long", "armed"), (f"{d} 13:15", "long", "trigger")])
    assert_armed(r, f"{d} 13:00", "long", extreme=2016.0, opposite=2000.0, leg=16.0, r50=2008.0, void=2003.424,
                 ext_bar=f"{d} 11:45", pl_bar=f"{d} 12:45", atr=2.0)
    assert_trigger(r, f"{d} 13:15", "long", "blocked", ("spread",), extreme=2016.0, opposite=2000.0,
                   pl_bar=f"{d} 12:45", atr=2.0, spread=0.45, entry=2011.55, stop=2007.30, R=4.25, trend_ok=True)
    assert len(r.res.positions) == 0
    assert_final(r, 100_000.0)


def test_c14_trend_disagrees_at_the_trigger_close(news):
    """C14: injected EMA_k = 2050.00 - 0.10 k. At 13:15 the last closed 1h bar (12:00-13:00) closes 2009.4 < EMA_8
    2049.20 and the EMA is falling (2049.20 < EMA_3 2049.70) -> BLOCKED: trend (rule 2, D4), the only reason."""
    d = "2024-07-26"
    r = run_case("C14_trend_block", news, "down")
    assert_events(r, [(f"{d} 13:00", "long", "armed"), (f"{d} 13:15", "long", "trigger")])
    assert_armed(r, f"{d} 13:00", "long", extreme=2016.0, opposite=2000.0, leg=16.0, r50=2008.0, void=2003.424,
                 ext_bar=f"{d} 11:45", pl_bar=f"{d} 12:45", atr=2.0)
    assert_trigger(r, f"{d} 13:15", "long", "blocked", ("trend",), extreme=2016.0, opposite=2000.0,
                   pl_bar=f"{d} 12:45", atr=2.0, spread=0.20, entry=2011.30, stop=2007.30, R=4.00, trend_ok=False)
    assert len(r.res.positions) == 0
    assert_final(r, 100_000.0)
