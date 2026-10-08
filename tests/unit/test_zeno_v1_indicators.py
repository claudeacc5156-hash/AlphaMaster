"""zeno_v1 indicators and clocks (D2, D3, D17, D19, D21): EMA30 with an SMA seed, Wilder ATR14, the
closed-H1-bar rule, the 5-bar slope, the broker server day, the 16:30 New York time exit and the entry
windows. Every expected number is computed by hand in the comments. Research only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propkit import calendar as cal
from propkit import indicators as ind
from propkit import zeno_v1 as z
from tests.unit.zeno_v1_testkit import utc


def _flat_frame(start: str, n: int, price: float = 2000.0) -> pd.DataFrame:
    t = utc(start) + 900 * np.arange(n, dtype=np.int64)
    bid = pd.DataFrame({"time": t, "open": price, "high": price + 1.0, "low": price - 1.0, "close": price})
    ask = bid.copy()
    ask[["open", "high", "low", "close"]] += 0.2
    return z.bidask_frame(bid, ask)


# ---------------------------------------------------------------------------------------------------
# D2: EMA30 (SMA seed) and ATR14 (Wilder)

def test_ema_sma_seed_hand_table_period_3():
    # alpha = 2/(3+1) = 0.5; seed = mean(1, 2, 3) = 2 at index 2; then 0.5 x 4 + 0.5 x 2 = 3, 0.5 x 5 + 0.5 x 3 = 4
    out = ind.ema([1, 2, 3, 4, 5], 3, seed="sma")
    assert np.isnan(out[:2]).all()
    assert out[2:].tolist() == [2.0, 3.0, 4.0]


def test_ema30_on_a_ramp():
    # x = 1..40. Seed ema[29] = mean(1..30) = 15.5. alpha = 2/31: ema[30] = (2/31) x 31 + (29/31) x 15.5
    # = 2 + 14.5 = 16.5. On a ramp of slope 1 the EMA lags by (1 - alpha)/alpha = 14.5, and the seed already
    # sits at x[29] - 14.5, so ema[t] = x[t] - 14.5 for every t >= 29: ema[39] = 40 - 14.5 = 25.5.
    x = np.arange(1, 41, dtype=float)
    e = z.ema30_h1(x)
    assert np.isnan(e[:29]).all()
    assert e[29] == pytest.approx(15.5, abs=1e-12)
    assert e[30] == pytest.approx(16.5, abs=1e-12)
    assert e[39] == pytest.approx(25.5, abs=1e-12)
    assert np.allclose(e[29:], x[29:] - 14.5, atol=1e-10)
    assert np.isnan(z.ema30_h1(x[:29])).all()                    # fewer than 30 closes: no value yet


def test_ema_default_seed_is_unchanged():
    # seed "first": ema[0] = x[0], then 0.5 x 2 + 0.5 x 1 = 1.5, 0.5 x 3 + 0.5 x 1.5 = 2.25 (period 3)
    assert ind.ema([1, 2, 3], 3).tolist() == [1.0, 1.5, 2.25]
    assert ind.ema([1, 2, 3], 3, seed="first").tolist() == [1.0, 1.5, 2.25]
    with pytest.raises(ValueError, match="seed"):
        ind.ema([1, 2, 3], 3, seed="zero")


def test_atr14_wilder_with_a_gap():
    # bars 0-13: high 2001, low 1999, close 2000 -> TR = 2 each (bar 0: high - low). ATR[13] = mean = 2.
    # bar 14 opens with a gap: high 2010, low 2009, close 2009.5; previous close 2000 -> TR = max(1, 10, 9) = 10.
    # Wilder: ATR[14] = (2 x 13 + 10) / 14 = 36/14 = 2.5714...
    # bar 15: high 2010, low 2009, close 2009.5 -> TR = max(1, 0.5, 0.5) = 1; ATR[15] = (36/14 x 13 + 1) / 14.
    f = _flat_frame("2024-03-04 00:00", 16)
    f.loc[14, ["bid_open", "bid_high", "bid_low", "bid_close"]] = [2009.5, 2010.0, 2009.0, 2009.5]
    f.loc[15, ["bid_open", "bid_high", "bid_low", "bid_close"]] = [2009.5, 2010.0, 2009.0, 2009.5]
    a = z.atr14_m15(f)
    assert np.isnan(a[:13]).all()
    assert a[13] == pytest.approx(2.0, abs=1e-12)
    assert a[14] == pytest.approx(36 / 14, abs=1e-12)
    assert a[15] == pytest.approx((36 / 14 * 13 + 1) / 14, abs=1e-12)


def test_prepare_default_path_computes_both_indicators():
    f = z.synthetic_m15_bidask(n_bars=600, seed=3)
    prep = z.prepare(f)
    assert prep.injected == ()
    assert np.array_equal(prep.atr, z.atr14_m15(f), equal_nan=True)
    h1 = z.h1_from_m15(f)
    assert np.array_equal(prep.h1["ema"].to_numpy(), z.ema30_h1(h1["close"].to_numpy()), equal_nan=True)


def test_prepare_test_indicators_are_checked_and_recorded():
    f = _flat_frame("2024-03-04 00:00", 40)
    n_h1 = len(z.h1_from_m15(f))
    prep = z.prepare(f, test_indicators={"atr14": np.full(40, 2.0), "ema30_h1": np.full(n_h1, 1.0)})
    assert prep.injected == ("atr14", "ema30_h1")
    res = z.simulate(prep, z.ZenoConfig(z.ZenoCell("evaluation", 10.0, "S1", 1.0)))
    assert res.meta["indicators_injected"] == ["atr14", "ema30_h1"]
    with pytest.raises(ValueError, match="one value per M15 bar"):
        z.prepare(f, test_indicators={"atr14": np.full(39, 2.0)})
    with pytest.raises(ValueError, match="one value per H1 bar"):
        z.prepare(f, test_indicators={"ema30_h1": np.full(n_h1 + 1, 1.0)})
    with pytest.raises(ValueError, match="accepts only"):
        z.prepare(f, test_indicators={"rsi": np.ones(40)})


# ---------------------------------------------------------------------------------------------------
# D3 / D4: closed H1 bars only, slope over 5 H1 bars of the data

def test_h1_bar_is_usable_at_its_close_not_before():
    # H1 bar [05:00, 06:00) closes at 06:00. The M15 bar opening 05:45 closes at 06:00: it may use it.
    # The M15 bar opening 05:30 closes at 05:45: the last closed H1 bar is [04:00, 05:00).
    t = utc("2024-03-04 00:00") + 900 * np.arange(40, dtype=np.int64)
    h1_close = utc("2024-03-04 01:00") + 3600 * np.arange(10, dtype=np.int64)
    j = z.h1_index_at_m15_close(t, h1_close)
    assert j[0:3].tolist() == [-1, -1, -1]                      # 00:00-00:30 opens: no H1 bar closed yet
    assert j[3] == 0                                            # 00:45 bar closes at 01:00
    assert j[22] == 4 and j[23] == 5                            # 05:30 -> [04:00,05:00); 05:45 -> [05:00,06:00)


def test_trend_state_slope_and_close_side():
    # long_ok[j] = close > ema and ema[j] > ema[j-5]; short_ok mirrors it. j < 5 or NaN -> False.
    e = np.array([10, 10, 10, 10, 10, 10.1, 10, 10.05, np.nan, 10.2, 10.3])
    c = np.full(e.size, 11.0)
    lo, sh = z.trend_state(c, e)
    assert lo.tolist() == [False] * 5 + [True, False, True, False, True, True]
    assert not sh.any()
    c[9] = 10.2                                                 # close == ema: not above it
    assert not z.trend_state(c, e)[0][9]
    e2 = 20.0 - e
    lo2, sh2 = z.trend_state(np.full(e.size, 9.0), e2)
    assert sh2.tolist() == [False] * 5 + [True, False, True, False, True, True] and not lo2.any()
    with pytest.raises(ValueError):
        z.trend_state(c[:-1], e)


def test_trend_in_prepare_uses_the_last_closed_h1_bar():
    # 1 day of bars from 00:00; injected EMA = 1000 + 0.01 j (below the price 2000, rising): long_ok from
    # H1 bar j = 5 ([05:00, 06:00), closed at 06:00) on. The first M15 bar with the long trend is the one
    # closing at 06:00 (opening 05:45, index 23); the bar before it still sees H1 bar 4 -> j - 5 < 0 -> False.
    f = _flat_frame("2024-03-04 00:00", 96)
    n_h1 = len(z.h1_from_m15(f))
    prep = z.prepare(f, test_indicators={"atr14": np.full(96, 2.0), "ema30_h1": 1000 + 0.01 * np.arange(n_h1)})
    assert not prep.trend_long[:23].any() and prep.trend_long[23:].all()
    assert not prep.trend_short.any()


def test_five_bars_ago_counts_h1_bars_of_the_data_across_a_gap():
    # H1 bars 00:00-03:00 on Monday, then a gap to 10:00-13:00. "5 bars ago" for the 11:00 bar (index 5) is
    # the 00:00 bar (index 0), 11 clock hours earlier, not a missing 06:00 bar [SI-6].
    times = np.r_[utc("2024-03-04 00:00") + 900 * np.arange(16), utc("2024-03-04 10:00") + 900 * np.arange(16)]
    bid = pd.DataFrame({"time": times, "open": 2000.0, "high": 2001.0, "low": 1999.0, "close": 2000.0})
    ask = bid.copy()
    ask[["open", "high", "low", "close"]] += 0.2
    f = z.bidask_frame(bid, ask)
    h1 = z.h1_from_m15(f)
    assert h1["time"].tolist()[4:6] == [utc("2024-03-04 10:00"), utc("2024-03-04 11:00")]
    ema = np.array([1000, 999, 999, 999, 999, 1000.5, 999, 999.0])   # j=5: 1000.5 > ema[0] = 1000
    prep = z.prepare(f, test_indicators={"atr14": np.full(f.shape[0], 2.0), "ema30_h1": ema})
    # M15 bars opening 11:45 .. 12:30 close at 12:00 .. 12:45: the last closed H1 bar is j = 5 (long_ok);
    # the 11:30 bar (closes 11:45) sees j = 4, the 12:45 bar (closes 13:00) sees j = 6 (999 > 999 fails).
    k = int(np.flatnonzero(f["time"].to_numpy() == utc("2024-03-04 11:45"))[0])
    assert prep.trend_long[k:k + 4].all()
    assert not prep.trend_long[k - 1] and not prep.trend_long[k + 4]


# ---------------------------------------------------------------------------------------------------
# D21 / D17: server day and 16:30 New York

def _day(text: str) -> int:
    return utc(text + " 00:00") // 86400


def test_server_day_boundaries():
    # winter (EST, UTC-5): 17:00 New York = 22:00 UTC starts the next server day
    assert z.server_day(utc("2024-03-07 21:45")) == _day("2024-03-07")
    assert z.server_day(utc("2024-03-07 22:00")) == _day("2024-03-08")
    # summer (EDT, UTC-4): 17:00 New York = 21:00 UTC
    assert z.server_day(utc("2024-03-11 20:45")) == _day("2024-03-11")
    assert z.server_day(utc("2024-03-11 21:00")) == _day("2024-03-12")
    # Sunday evening bars belong to Monday's server day
    assert z.server_day(utc("2024-11-03 23:00")) == _day("2024-11-04")
    arr = z.server_day(np.array([utc("2024-03-07 21:45"), utc("2024-03-07 22:00")]))
    assert arr.tolist() == [_day("2024-03-07"), _day("2024-03-08")]
    assert cal.day_to_str(_day("2024-03-08")) == "2024-03-08"


@pytest.mark.parametrize("day,expected", [
    ("2024-03-08", "2024-03-08 21:30"),    # Friday, EST (DST starts Sunday 2024-03-10)
    ("2024-03-11", "2024-03-11 20:30"),    # Monday, EDT
    ("2024-11-01", "2024-11-01 20:30"),    # Friday, EDT (DST ends Sunday 2024-11-03)
    ("2024-11-04", "2024-11-04 21:30"),    # Monday, EST
    ("2024-03-10", "2024-03-10 20:30"),    # the switch day itself: 16:30 is already EDT
    ("2024-11-03", "2024-11-03 21:30"),    # 16:30 is already EST
])
def test_time_exit_instant_known_answers(day, expected):
    assert z.time_exit_instant(_day(day)) == utc(expected)


def test_time_exit_is_inside_its_own_server_day():
    days = np.arange(_day("2015-01-01"), _day("2025-09-28"))
    tx = z.time_exit_instant(days)
    assert np.array_equal(z.server_day(tx), days)
    assert np.array_equal(z.server_day(tx + 1799), days)       # 16:59:59 New York: still the same day
    assert np.array_equal(z.server_day(tx + 1800), days + 1)   # 17:00 New York starts the next one


# ---------------------------------------------------------------------------------------------------
# D19: entry windows at the trigger close

@pytest.mark.parametrize("hhmmss,ok", [
    ("06:59:59", False), ("07:00:00", True), ("09:59:59", True), ("10:00:00", False),
    ("12:29:59", False), ("12:30:00", True), ("15:59:59", True), ("16:00:00", False), ("00:00:00", False),
])
def test_session_windows_utc(hhmmss, ok):
    t = int(pd.Timestamp("2024-03-05 " + hhmmss, tz="UTC").timestamp())
    assert z.session_ok(t) is ok
    assert bool(z.session_ok(np.array([t]))[0]) is ok


def test_session_windows_equal_the_sgt_text():
    # 15:00-18:00 and 20:30-24:00 SGT = UTC + 8 h
    for (a, b), (sa, sb) in zip(z.SESSION_WINDOWS_UTC, ((15 * 3600, 18 * 3600), (20 * 3600 + 1800, 24 * 3600))):
        assert (a + 8 * 3600, b + 8 * 3600) == (sa, sb)


def test_trading_days_and_ranks():
    # bars on server days 2024-03-04 (Mon) and 2024-03-05, none on 03-06, some on 03-07: three trading days
    times = np.array([utc("2024-03-04 01:00"), utc("2024-03-04 05:00"), utc("2024-03-04 22:00"),
                      utc("2024-03-07 01:00")])
    days, first = z.trading_days(times)
    assert [cal.day_to_str(int(d)) for d in days] == ["2024-03-04", "2024-03-05", "2024-03-07"]
    assert first.tolist() == [0, 2, 3, 4]
    with pytest.raises(ValueError):
        z.trading_days(times[::-1])
