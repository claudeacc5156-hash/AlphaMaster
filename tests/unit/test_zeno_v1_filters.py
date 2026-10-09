"""zeno_v1 filters and limits (rules 8-11, D1 warm-up, D10 one shot, D18-D22): every blocking reason, its
boundary, the one-shot consumption of a blocked setup, the daily limits on the broker server day and the
same-direction cooldown. Canonical long of tests/unit/zeno_v1_testkit.py (entry 2006.20, stop 2001.50,
R 4.70, 106 oz) unless a test says otherwise; cell evaluation / 10 / S1 / x1. Research only."""
from __future__ import annotations

import numpy as np
import pytest

from propkit import zeno_v1 as z
from tests.unit.zeno_v1_testkit import EVAL_10_X1, Scenario, events_of, news_from_rows, triggers, utc

STOP_BAR = (2006.0, 2006.0, 2001.0, 2002.0)
AGAIN = (2006.5, 2009.0, 2006.0, 2008.5)          # would be a second trigger (close > 2005.5) if not one-shot


def case(start="2024-03-05 12:00", after=(STOP_BAR, AGAIN), tail=4, filler_days=32, side=1, **run_kw):
    sc = Scenario(start, filler_days=filler_days)
    ti = sc.setup(side)
    for bar in after:
        sc.add(**bar) if isinstance(bar, dict) else sc.add(*bar)
    if tail:
        sc.flat(tail, sc.bid[-1][3])
    prep, res = sc.run(**run_kw)
    return sc, ti, prep, res


def status_at(res, bar):
    tr = triggers(res)
    row = tr[tr["bar_index"] == bar]
    assert len(row) == 1
    return row.iloc[0]["status"], row.iloc[0]["reasons"].split(";")


# ---------------------------------------------------------------------------------------------------
# every blocker consumes the setup (D10)

def _blocked(kind, tmp_path):
    if kind == "outside_session":            # trigger closes 06:45 UTC; AGAIN closes 07:00 inside the window
        return case(start="2024-03-05 04:45")
    if kind == "news_blackout":              # trigger closes 13:00 UTC = NFP 13:30 - 30 min
        return case(start="2024-03-08 11:00", news=news_from_rows(tmp_path))
    if kind == "spread_gt_10pct_of_stop":    # entry-bar spread 0.51: R = 4.50 + 0.51 = 5.01, 10% = 0.501
        return case(after=(dict(o=2006.0, h=2006.0, lo=2005.0, c=2005.5, spread=0.51), AGAIN))
    if kind == "atr_above_2x_median":        # median of 20 days of ATR 2.00 = 2.00; ATR 4.01 > 4.00
        sc = Scenario("2024-03-05 12:00")
        ti = sc.setup(1)
        sc.add(*STOP_BAR)
        sc.add(*AGAIN)
        sc.flat(4, 2008.5)
        return (sc, ti) + sc.run(atr_at={ti: 4.01})
    if kind == "stop_wider_than_3_atr":      # ATR 1.50: stop 2001.625, R = 4.575 > 4.50
        sc = Scenario("2024-03-05 12:00")
        ti = sc.setup(1)
        sc.add(*STOP_BAR)
        sc.add(*AGAIN)
        sc.flat(4, 2008.5)
        return (sc, ti) + sc.run(atr_at={ti: 1.5})
    if kind == "trend_disagrees":
        return case(trend="short")
    if kind == "warmup":                     # only 10 trading days before the trigger
        return case(filler_days=10)
    if kind == "size_below_lot_step":        # 0.5% of 500 = 2.50 USD < 4.70 x 1 oz
        return case(capital=500.0)
    if kind == "entry_beyond_stop":          # the next bar opens at bid 2001: R = 2001.20 - 2001.50 < 0
        return case(after=((2001.0, 2001.0, 2000.0, 2000.5), (2000.5, 2009.0, 2000.5, 2008.5)))
    if kind == "entry_after_time_exit":      # trigger closes 15:45 UTC; the data resumes at 21:30 UTC =
        sc = Scenario("2024-03-05 13:45")    # 16:30 New York of the same server day (EST)
        ti = sc.setup(1)
        sc.skip_to("2024-03-05 21:30")
        sc.add(*AGAIN)
        sc.flat(4, 2008.5)
        return (sc, ti) + sc.run()
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["outside_session", "news_blackout", "spread_gt_10pct_of_stop",
                                  "atr_above_2x_median", "stop_wider_than_3_atr", "trend_disagrees", "warmup",
                                  "size_below_lot_step", "entry_beyond_stop", "entry_after_time_exit"])
def test_a_blocked_trigger_is_one_shot(kind, tmp_path):
    sc, ti, prep, res = _blocked(kind, tmp_path)
    status, reasons = status_at(res, ti)
    assert status == kind and reasons[0] == kind
    assert len(res.positions) == 0
    assert events_of(res, "L1") == ["armed", "trigger"]          # no second trigger, no re-arm
    assert (triggers(res)["side"] == "long").sum() == 1
    assert all(r in z.BLOCK_REASONS for r in reasons)


@pytest.mark.parametrize("start,resume", [
    ("2024-03-05 13:45", "2024-03-05 21:45"),   # Tuesday (EST): the data resumes 16:45 New York
    ("2024-03-05 13:45", "2024-03-05 23:00"),   # resumes 18:00 New York, already the NEXT server day
    ("2024-03-05 13:45", "2024-03-06 07:00"),   # resumes the next morning
    ("2024-03-01 13:45", "2024-03-03 23:00"),   # Friday trigger, the data resumes at the Sunday reopen
])
def test_a_gap_cannot_carry_an_entry_past_16_30_new_york_of_the_trigger_day(start, resume):
    # The trigger closes 15:45 UTC; the entry time is that close (D19) and it counts on that server day (D21,
    # [SI-25]), whose 16:30 New York is 21:30 UTC (D17). Any fill at or after that instant is blocked,
    # however long the gap: a longer gap must not let the entry through onto a later server day.
    sc = Scenario(start)
    ti = sc.setup(1)
    sc.skip_to(resume)
    sc.add(*AGAIN)
    sc.flat(8, 2008.5)
    prep, res = sc.run()
    assert z.time_exit_instant(int(prep.close_day[ti])) == utc(start[:10] + " 21:30")
    assert status_at(res, ti)[0] == "entry_after_time_exit"
    assert len(res.positions) == 0


def test_warmup_also_reports_the_undefined_median():
    sc, ti, prep, res = case(filler_days=10)
    assert status_at(res, ti) == ("warmup", ["warmup", "atr_above_2x_median"])
    assert np.isnan(triggers(res).iloc[0]["atr_median"])


@pytest.mark.parametrize("filler_days,status", [(29, "warmup"), (30, "entered")])
def test_warmup_is_30_trading_days(filler_days, status):
    # the filler starts 12:00 UTC N days before the trigger day and every day has bars, so the trigger's
    # server day is trading day N (counting from 0): N = 29 is still the 30th day of warm-up, N = 30 is not
    sc, ti, prep, res = case(filler_days=filler_days)
    assert int(prep.close_rank[ti]) == filler_days
    assert status_at(res, ti)[0] == status


def test_the_last_bar_cannot_be_entered():
    sc, ti, prep, res = case(after=(), tail=0)
    assert ti == prep.n - 1
    assert status_at(res, ti)[0] == "no_next_bar"


def test_block_reasons_are_reported_in_order():
    keys = list(z.BLOCK_REASONS)
    assert keys.index("cooldown_15min") < keys.index("position_open") < keys.index("trend_disagrees")
    assert keys[:4] == ["warmup", "no_next_bar", "outside_session", "news_blackout"]
    # several at once: outside the session, against the trend and too small
    sc, ti, prep, res = case(start="2024-03-05 04:45", trend="short", capital=500.0)
    assert status_at(res, ti)[1] == ["outside_session", "trend_disagrees", "size_below_lot_step"]


# ---------------------------------------------------------------------------------------------------
# boundaries

@pytest.mark.parametrize("trigger_close,entered", [
    ("07:00", True), ("09:45", True), ("10:00", False), ("12:15", False), ("12:30", True), ("15:45", True),
    ("16:00", False), ("06:45", False),
])
def test_entry_window_edges(trigger_close, entered):
    start = utc(f"2024-03-05 {trigger_close}") - 8 * 900
    sc, ti, prep, res = case(start=z.calendar.utc_str(start)[:16])
    assert (status_at(res, ti)[0] == "entered") is entered
    if not entered:
        assert status_at(res, ti)[0] == "outside_session"


@pytest.mark.parametrize("trigger_close,status", [
    ("12:45", "entered"),          # NFP 13:30 - 45 min
    ("13:00", "news_blackout"),    # T - 30 min: blocked (inclusive)
    ("14:30", "news_blackout"),    # T + 60 min: blocked (inclusive)
    ("14:45", "entered"),          # T + 75 min
])
def test_news_blackout_edges(trigger_close, status, tmp_path):
    start = utc(f"2024-03-08 {trigger_close}") - 8 * 900
    sc, ti, prep, res = case(start=z.calendar.utc_str(start)[:16], news=news_from_rows(tmp_path))
    assert status_at(res, ti)[0] == status


def test_news_window_to_the_second(tmp_path):
    news = news_from_rows(tmp_path)
    T = utc("2024-03-08 13:30")
    for dt, blocked in ((-31 * 60, False), (-30 * 60 - 1, False), (-30 * 60, True), (0, True), (60 * 60, True),
                        (60 * 60 + 1, False), (61 * 60, False)):
        assert news.blocked(T + dt) is blocked, dt
    # the unscheduled FOMC of Sunday 2020-03-15 21:00 UTC is in the calendar too
    assert news.blocked(utc("2020-03-15 21:30")) and "unscheduled" in news.kinds
    assert news.summary()["per_event"] == {"CPI": 1, "FOMC": 3, "NFP": 2}
    empty = z.news_calendar(np.zeros(0, dtype=np.int64))
    assert not empty.blocked(T)


def test_no_news_calendar_means_no_blackout_and_says_so():
    sc, ti, prep, res = case(start="2024-03-08 11:00")
    assert status_at(res, ti)[0] == "entered"
    assert res.meta["news"].startswith("no news calendar given")


@pytest.mark.parametrize("spread,status", [(0.50, "entered"), (0.51, "spread_gt_10pct_of_stop")])
def test_spread_edge(spread, status):
    # R = (2006 + s) - 2001.50 = 4.50 + s; blocked when s > 0.1 x (4.50 + s), i.e. s > 0.50
    sc, ti, prep, res = case(after=(dict(o=2006.0, h=2006.0, lo=2005.0, c=2005.5, spread=spread), AGAIN))
    assert status_at(res, ti)[0] == status


@pytest.mark.parametrize("atr,status", [
    (4.0, "entered"),                  # = 2 x median 2.00 (stop 2001.00, R 5.20 <= 12)
    (4.01, "atr_above_2x_median"),
    (1.6, "entered"),                  # stop 2001.60, R 4.60 <= 4.80
    (1.5, "stop_wider_than_3_atr"),    # stop 2001.625, R 4.575 > 4.50
])
def test_volatility_and_stop_width_edges(atr, status):
    sc = Scenario("2024-03-05 12:00")
    ti = sc.setup(1)
    sc.flat(4, 2006.0)
    prep, res = sc.run(atr_at={ti: atr})
    assert status_at(res, ti)[0] == status
    d = triggers(res).iloc[0]
    assert d["atr_trigger"] == atr and d["atr_median"] == 2.0
    assert d["stop_level"] == pytest.approx(2002.0 - 0.25 * atr)      # D12: the trigger bar's ATR


def test_vol_median_is_pooled_over_every_bar_of_the_20_days():
    # days 0-9: 10 bars each with ATR 1; days 10-19: 1 bar each with ATR 5. Pooled per bar: 100 ones and
    # 10 fives -> median 1.0 (a median of daily medians would be 3.0). Rank 20 = the day after day 19.
    atr = np.r_[np.ones(100), np.full(10, 5.0), [9.0]]
    first = np.r_[np.arange(0, 100, 10), np.arange(100, 111), 111]
    med = z.vol_medians(atr, first)
    assert med.size == 22
    assert np.isnan(med[:20]).all()
    assert med[20] == 1.0
    # rank 21 (days 1-20): 90 ones, 10 fives and the 9 -> still 1.0; NaN values are skipped
    assert med[21] == 1.0
    atr2 = atr.copy()
    atr2[:95] = np.nan                         # 5 ones, 10 fives left -> median 5.0
    assert z.vol_medians(atr2, first)[20] == 5.0


# ---------------------------------------------------------------------------------------------------
# D21: daily limits on the broker server day

def _three_blocks(risk_pct=None, next_day=False):
    sc = Scenario("2024-03-05 05:30")
    t1 = sc.setup(1)                       # closes 07:15 UTC
    sc.add(*STOP_BAR)
    t2 = sc.setup(1)                       # closes 09:30 UTC
    sc.add(*STOP_BAR)
    sc.flat_until("2024-03-05 11:00", 2000.0)
    t3 = sc.setup(1)                       # closes 12:45 UTC
    sc.add(*STOP_BAR)
    ts = [t1, t2, t3]
    if next_day:
        sc.flat_until("2024-03-06 05:30", 2000.0)
        ts.append(sc.setup(1))             # closes 2024-03-06 07:15 UTC: the next server day
        sc.add(*STOP_BAR)
    sc.flat(4, 2000.0)
    prep, res = sc.run(risk_pct=risk_pct)
    return ts, prep, res


def test_two_entries_two_losses_and_minus_one_percent():
    # trade 1: 106 oz, -514.10; trade 2 on 99,485.90: 105 oz, -509.25; the day: -1,023.35 <= -1,000
    ts, prep, res = _three_blocks()
    assert [status_at(res, t)[0] for t in ts[:2]] == ["entered", "entered"]
    assert res.positions["net_pnl_usd"].tolist() == pytest.approx([-514.10, -509.25])
    assert status_at(res, ts[2]) == ("max_entries_per_day",
                                     ["max_entries_per_day", "two_losses_today", "day_loss_1pct"])


def test_day_loss_alone_blocks_and_the_next_server_day_resets():
    # at 1.1% risk: 1100 / 4.70 = 234.04 -> 234 oz; the stop costs 234 x 4.75 + 23.40 = 1,134.90 >= 1% of
    # 100,000 after ONE trade: the second and third triggers of the day are blocked by day_loss_1pct alone.
    ts, prep, res = _three_blocks(risk_pct=0.011, next_day=True)
    assert status_at(res, ts[0])[0] == "entered"
    assert res.positions["net_pnl_usd"].iloc[0] == pytest.approx(-1134.90)
    assert status_at(res, ts[1]) == ("day_loss_1pct", ["day_loss_1pct"])
    assert status_at(res, ts[2]) == ("day_loss_1pct", ["day_loss_1pct"])
    # the next server day starts at 17:00 New York: its trigger is entered, sized on the closed balance
    assert status_at(res, ts[3])[0] == "entered"
    p = res.positions.iloc[1]
    assert p["server_day"] == "2024-03-06" and p["balance_at_entry"] == pytest.approx(100_000 - 1134.90)
    assert p["units_oz"] == float(int(0.011 * (100_000 - 1134.90) / 4.70))


@pytest.mark.parametrize("day,close_before,close_after", [
    ("2024-03-05", "21:45", "22:00"),      # EST: 17:00 New York = 22:00 UTC
    ("2024-07-09", "20:45", "21:00"),      # EDT: 17:00 New York = 21:00 UTC
])
def test_daily_limits_reset_at_17_00_new_york(day, close_before, close_after):
    # two losing entries in the morning (07:15 and 09:30 UTC); a third trigger closing just before 17:00 New
    # York still counts against that server day, one closing at 17:00 New York belongs to the next one.
    # (Both are outside the entry window, which is reported too: the limits are evaluated for every trigger.)
    for close, same_day in ((close_before, True), (close_after, False)):
        sc = Scenario(f"{day} 05:30")
        sc.setup(1)
        sc.add(*STOP_BAR)
        sc.setup(1)
        sc.add(*STOP_BAR)
        sc.flat_until(z.calendar.utc_str(utc(f"{day} {close}") - 8 * 900)[:16], 2000.0)
        t3 = sc.setup(1)
        sc.flat(4, 2006.0)
        prep, res = sc.run()
        assert len(res.positions) == 2
        assert bool(prep.close_day[t3] == z.server_day(utc(f"{day} 12:00"))) == same_day
        reasons = status_at(res, t3)[1]
        limits = ["max_entries_per_day", "two_losses_today", "day_loss_1pct"]
        assert [r in reasons for r in limits] == [same_day] * 3
        assert reasons[0] == "outside_session"


def test_a_trade_open_blocks_a_new_entry():
    # trade 1 (07:15) stays open (flat 2006, then 2008); a second long setup at base 2008 triggers at 14:45
    # with trade 1's runner still open (tp1 2015.60 was hit by its H bar 2018): position_open, one shot.
    sc = Scenario("2024-03-05 05:30")
    t1 = sc.setup(1)
    sc.flat(2, 2006.0)
    sc.flat(20, 2008.0)
    t2 = sc.setup(1, base=2008.0)
    sc.add(2014.0, 2016.0, 2013.0, 2015.5)
    sc.flat(4, 2015.0)
    prep, res = sc.run()
    assert status_at(res, t1)[0] == "entered"
    assert status_at(res, t2) == ("position_open", ["position_open"])
    assert len(res.positions) == 1 and res.positions.iloc[0]["tp1_reached"]
    assert events_of(res, "L2") == ["armed", "trigger"]


def test_an_exit_at_the_next_open_does_not_free_the_trigger():
    # D21: "a trigger that comes while a position is open is used up". Same as above, but the bar after the
    # second trigger gaps down to 2005 (< breakeven 2006.30): trade 1's runner is stopped at that open
    # (2005 - 0.05 = 2004.95), the very open the second entry would have used. At the trigger close the
    # runner was open, so the trigger is consumed ([SI-16] cannot occur). (The gap also opens below setup
    # 2's own stop 2009.50, which is listed too.)
    sc = Scenario("2024-03-05 05:30")
    sc.setup(1)
    sc.flat(2, 2006.0)
    sc.flat(20, 2008.0)
    t2 = sc.setup(1, base=2008.0)
    sc.add(2005.0, 2006.0, 2004.0, 2005.5)
    sc.flat(4, 2005.5)
    prep, res = sc.run()
    assert status_at(res, t2) == ("position_open", ["position_open", "entry_beyond_stop"])
    p = res.positions.iloc[0]
    assert (p["exit2_reason"], p["exit2_time"], p["exit2_price"]) == ("stop", sc.time(t2 + 1), pytest.approx(2004.95))
    assert len(res.positions) == 1


# ---------------------------------------------------------------------------------------------------
# D22: 15-minute same-direction cooldown from the exit stamp

def _short_cooldown(spike_bar: int):
    """Short trade 1 (mirror canonical: entry 1994.00, stop 1998.70 on the ASK) and short setup 2:
    b8 makes the low 1990 again (a tie: the latest bar is the new L), H' = 2005 -> leg 15, 50% = 1997.5;
    b10 is the pullback-high bar (bid high 1998.00, low 1995); b11 closes 1994.50 < 1995 -> trigger
    (closes 15:00 UTC). An ask spike to 1998.80 in bar `spike_bar` stops trade 1 there (stamp = its close):
    spike in b10 -> stamp 14:45, b11's close is exactly 15 min later -> entered; spike in b11 -> stamp
    15:00 = the trigger close -> cooldown_15min."""
    sc = Scenario("2024-03-05 12:00")
    t1 = sc.setup(-1)
    bars = {8: (1994.0, 1994.5, 1990.0, 1990.5), 9: (1990.5, 1995.0, 1990.5, 1995.0),
            10: (1995.0, 1998.0, 1995.0, 1997.0), 11: (1997.0, 1997.9, 1994.0, 1994.5)}
    for k in (8, 9, 10, 11):
        o, h, lo, c = bars[k]
        if k == spike_bar:
            sc.add(o, h, lo, c, ask=(o + 0.2, 1998.8, lo + 0.2, c + 0.2))
        else:
            sc.add(o, h, lo, c)
    t2 = len(sc.bid) - 1
    sc.flat(4, 1994.5)
    prep, res = sc.run(trend="short")
    return sc, t1, t2, prep, res


def test_cooldown_counts_from_the_intrabar_exit_stamp():
    sc, t1, t2, prep, res = _short_cooldown(spike_bar=10)
    p = res.positions.iloc[0]
    assert (p["exit1_reason"], p["exit1_price"]) == ("stop", pytest.approx(1998.75))
    assert p["final_exit_stamp"] == utc("2024-03-05 14:45")
    tr = triggers(res).iloc[-1]
    assert tr["time"] == utc("2024-03-05 15:00") and tr["setup_id"] == "S2"
    assert status_at(res, t2)[0] == "entered"


def test_cooldown_blocks_a_trigger_at_the_close_of_the_exit_bar():
    sc, t1, t2, prep, res = _short_cooldown(spike_bar=11)
    assert res.positions.iloc[0]["final_exit_stamp"] == utc("2024-03-05 15:00")
    assert status_at(res, t2) == ("cooldown_15min", ["cooldown_15min"])
    assert len(res.positions) == 1


def test_cooldown_is_per_direction():
    # white box: the decision for the canonical long trigger with an exit stamp set 1 s too recently
    sc = Scenario("2024-03-05 12:00")
    ti = sc.setup(1)
    sc.flat(4, 2006.0)
    prep = sc.prepare()
    idx = next(k for k, e in enumerate(prep.events) if e["event"] == "trigger")
    t_c = sc.time(ti) + 900
    for side, stamp, expected in ((1, t_c - 899, "cooldown_15min"), (1, t_c - 900, "entered"),
                                  (-1, t_c - 1, "entered")):
        eng = z._Engine(prep, z.ZenoConfig(EVAL_10_X1), None)
        eng.last_stamp[side] = stamp
        eng._decide(idx, prep.events[idx])
        assert eng.status[idx]["status"] == expected, (side, stamp)
