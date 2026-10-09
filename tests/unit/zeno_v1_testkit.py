"""Builders shared by tests/unit/test_zeno_v1_*.py: hand-built M15 bid/ask bars (NOT market data).

The canonical LONG setup (prices for base B = 2000, bid OHLC; the ask is bid + 0.20 unless a test says so;
ATR14 is injected as 2.00 on every bar, EMA30 is injected so the 1h trend is up):

    bar  open    high    low     close
    b0   2000    2000    1995    1996     L = 1995 (lowest low of the 20 bars before H; the rest are flat 2000)
    b1   1996    2001    1996    2001
    b2   2001    2006    2001    2006
    b3   2006    2010    2005    2009     H = 2010, leg = 15 >= 1.5 x 2 = 3
    b4   2009    2009    2004    2005     low 2004 > 2002.5: the 50% level is not touched yet
    b5   2005    2005.5  2002    2003     low 2002 <= 2002.5 = H - 0.5 x 15: ARMED; pullback-low bar
    b6   2003    2004    2002.5  2004     close 2004 <= 2005.5 (b5's high): no trigger (1 bar after the low)
    b7   2004    2007    2003.5  2006     close 2006 > 2005.5: TRIGGER (2 bars after the low)

    void level = 2010 - 0.786 x 15 = 1998.21; every close since H is above it.
    With the next bar opening at bid 2006.00: long entry = ask open = 2006.20, stop = 2002 - 0.25 x 2 = 2001.50,
    R = 4.70 USD/oz, risk 0.5% of 100,000 = 500 USD -> 500 / 4.70 = 106.38 oz -> 106 oz = 1.06 lots,
    tp1 = 2006.20 + 9.40 = 2015.60, tp2 = 2006.20 + 18.80 = 2025.00, breakeven = 2006.20 + 10 / 100 = 2006.30
    (commission 10 USD per lot round trip, cost multiplier 1).
The SHORT case is the mirror around B (price' = 2B - price, high and low swapped) on bid bars: entry =
bid open 1994.00, stop = 1998.00 + 0.50 + 0.20 (entry spread) = 1998.70 (an ASK level), R = 4.70.

Research only: synthetic data, no orders.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from propkit import zeno_v1 as z

BASE = 2000.0
SPREAD = 0.20
ATR = 2.0
M15 = 900
EVAL_10_X1 = z.ZenoCell("evaluation", 10.0, "S1", 1.0)

# Six rows copied verbatim from research/news_calendar/us_macro_events_2015-01-01_2025-09-27.csv
NEWS_CSV = (
    "event,date_et,time_et,utc_offset_ny,datetime_utc,kind,basis,source_list,cross_check,note\n"
    "FOMC,2020-03-15,17:00,-0400,2020-03-15T21:00Z,unscheduled,both,https://www.federalreserve.gov/monetarypolicy/"
    "fomchistorical<YEAR>.htm (2015-2020); fomccalendars.htm (2021-2025),Fed press releases + openmarket.htm,"
    "Mar 15 (Sunday) unscheduled meeting; 100bp cut to 0-0.25%; released 5:00 p.m. EDT. Scheduled Mar 17-18 "
    "meeting was cancelled\n"
    "NFP,2024-03-08,08:30,-0500,2024-03-08T13:30Z,scheduled,both,https://www.bls.gov/bls/news-release/empsit.htm,"
    "ALFRED rid=50 + BLS yearly schedules,Feb 2024 Employment Situation; BLS archive empsit_03082024.htm (Fri)\n"
    "CPI,2024-03-12,08:30,-0400,2024-03-12T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/cpi.htm,"
    "ALFRED rid=10 + BLS yearly schedules,Feb 2024 CPI (cpi_03122024.htm)\n"
    "FOMC,2024-03-20,14:00,-0400,2024-03-20T18:00Z,scheduled,both,https://www.federalreserve.gov/monetarypolicy/"
    "fomchistorical<YEAR>.htm (2015-2020); fomccalendars.htm (2021-2025),Fed press releases + openmarket.htm,"
    "Mar 19-20 meeting\n"
    "NFP,2024-11-01,08:30,-0400,2024-11-01T12:30Z,scheduled,both,https://www.bls.gov/bls/news-release/empsit.htm,"
    "ALFRED rid=50 + BLS yearly schedules,Oct 2024 Employment Situation; BLS archive empsit_11012024.htm (Fri)\n"
    "FOMC,2024-11-07,14:00,-0500,2024-11-07T19:00Z,scheduled,both,https://www.federalreserve.gov/monetarypolicy/"
    "fomchistorical<YEAR>.htm (2015-2020); fomccalendars.htm (2021-2025),Fed press releases + openmarket.htm,"
    "Nov 6-7 meeting (Thursday release)\n")


def utc(text: str) -> int:
    """UTC epoch seconds of 'YYYY-MM-DD HH:MM' (UTC)."""
    return int(pd.Timestamp(text, tz="UTC").timestamp())


def news_from_rows(tmp_path) -> z.NewsCalendar:
    """The six real calendar rows above, read through zeno_v1.read_news_csv."""
    p = tmp_path / "news_rows.csv"
    p.write_text(NEWS_CSV, encoding="ascii")
    return z.read_news_csv(p)


def long_setup(base: float = BASE) -> list[tuple[float, float, float, float]]:
    """The 8 canonical long bars (o, h, l, c) relative to base; the trigger is the last one."""
    b = base
    return [(b, b, b - 5, b - 4), (b - 4, b + 1, b - 4, b + 1), (b + 1, b + 6, b + 1, b + 6),
            (b + 6, b + 10, b + 5, b + 9), (b + 9, b + 9, b + 4, b + 5), (b + 5, b + 5.5, b + 2, b + 3),
            (b + 3, b + 4, b + 2.5, b + 4), (b + 4, b + 7, b + 3.5, b + 6)]


def mirror(bars: Iterable[tuple[float, float, float, float]], base: float = BASE):
    """Bars reflected around base: price' = 2 base - price (high and low swap)."""
    return [(2 * base - o, 2 * base - lo, 2 * base - h, 2 * base - c) for o, h, lo, c in bars]


class Scenario:
    """A bar script on a continuous 15-minute grid: `filler_days` days of flat bars at `base` (so the 30-day
    warm-up and the 20-day ATR median are behind us), then whatever the test adds."""

    def __init__(self, first_bar_utc: str, filler_days: int = 32, base: float = BASE, spread: float = SPREAD):
        t0 = utc(first_bar_utc)
        assert t0 % M15 == 0
        self.start = t0 - filler_days * 86400
        self.bid: list[tuple[float, float, float, float]] = []
        self.ask: list[tuple[float, float, float, float] | None] = []
        self.spread: list[float] = []
        self.default_spread = spread
        self.flat(filler_days * 96, base)
        self.times: list[int] = [self.start + k * M15 for k in range(len(self.bid))]
        self.cursor = t0

    # ----- building ------------------------------------------------------------------

    def _append(self, bar, spread=None, ask=None) -> int:
        self.bid.append(tuple(float(x) for x in bar))
        self.spread.append(self.default_spread if spread is None else float(spread))
        self.ask.append(None if ask is None else tuple(float(x) for x in ask))
        if hasattr(self, "times"):
            self.times.append(self.cursor)
            self.cursor += M15
        return len(self.bid) - 1

    def add(self, o, h, lo, c, spread=None, ask=None) -> int:
        """Append one bar (bid OHLC; ask = bid + spread, or the explicit ask OHLC); returns its index."""
        return self._append((o, h, lo, c), spread, ask)

    def flat(self, n: int, price: float, spread=None) -> int:
        """Append n flat bars (open = high = low = close = price); returns the last index."""
        for _ in range(n):
            self._append((price, price, price, price), spread)
        return len(self.bid) - 1

    def flat_until(self, when_utc: str, price: float) -> int:
        """Append flat bars until the next bar opens at when_utc."""
        target = utc(when_utc)
        assert target >= self.cursor and (target - self.cursor) % M15 == 0
        while self.cursor < target:
            self.flat(1, price)
        return len(self.bid) - 1

    def skip_to(self, when_utc: str) -> None:
        """A gap: the next bar opens at when_utc."""
        target = utc(when_utc)
        assert target > self.cursor and target % M15 == 0
        self.cursor = target

    def setup(self, side: int = 1, base: float = BASE) -> int:
        """Append the canonical 8-bar setup (mirrored for a short); returns the trigger bar's index."""
        bars = long_setup(base) if side > 0 else mirror(long_setup(base), base)
        for bar in bars:
            idx = self.add(*bar)
        return idx

    def time(self, i: int) -> int:
        """Open time of bar i."""
        return self.times[i]

    # ----- running -------------------------------------------------------------------

    def frame(self) -> pd.DataFrame:
        """The zeno frame (bidask_frame validates everything)."""
        bid = pd.DataFrame(self.bid, columns=["open", "high", "low", "close"])
        bid.insert(0, "time", np.asarray(self.times, dtype=np.int64))
        ask = bid.copy()
        rows = []
        for b, a, s in zip(self.bid, self.ask, self.spread):
            rows.append(a if a is not None else tuple(x + s for x in b))
        ask[["open", "high", "low", "close"]] = np.asarray(rows, dtype=np.float64)
        return z.bidask_frame(bid, ask)

    def prepare(self, news=None, atr: float = ATR, atr_at: dict | None = None, trend: str = "long", restricted=None):
        """prepare() with ATR14 injected (atr everywhere, atr_at {bar index: value}) and EMA30 injected so
        the trend is 'long' (EMA = 1000 + 0.01 j, below price and rising), 'short' (3000 - 0.01 j) or 'none';
        restricted: FundingPips' restricted calendar for the variant master_fp (addendum A1)."""
        f = self.frame()
        h1 = z.h1_from_m15(f)
        j = np.arange(len(h1), dtype=np.float64)
        ema = {"long": 1000.0 + 0.01 * j, "short": 3000.0 - 0.01 * j, "none": np.full(len(h1), np.nan)}[trend]
        a = np.full(len(f), float(atr))
        for i, v in (atr_at or {}).items():
            a[i] = v
        return z.prepare(f, news, restricted=restricted, test_indicators={"atr14": a, "ema30_h1": ema})

    def run(self, cell=EVAL_10_X1, capital: float = 100_000.0, risk_pct=None, news=None, atr: float = ATR,
            atr_at: dict | None = None, trend: str = "long", m1=None, restricted=None):
        """(prep, result) for one cell."""
        prep = self.prepare(news=news, atr=atr, atr_at=atr_at, trend=trend, restricted=restricted)
        res = z.simulate(prep, z.ZenoConfig(cell, capital, risk_pct), m1=m1)
        return prep, res


def triggers(res) -> pd.DataFrame:
    """The trigger rows of a result's decisions table."""
    d = res.decisions
    return d[d["event"] == "trigger"].reset_index(drop=True)


def events_of(res, setup_id: str) -> list[str]:
    """The event names of one setup, in order."""
    d = res.decisions
    return d.loc[d["setup_id"] == setup_id, "event"].tolist()


def trigger_at(res, bar: int) -> pd.Series:
    """The single trigger row at a bar index."""
    t = triggers(res)
    rows = t[t["bar_index"] == bar]
    assert len(rows) == 1, f"expected one trigger at bar {bar}, got {len(rows)}"
    return rows.iloc[0]
