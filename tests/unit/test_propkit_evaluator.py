"""Tests for propkit/evaluator.py: evaluate_path on hand-built EQUITY frames. Research only.

Every EQUITY frame here is written by hand (the equity engine is tested on its own). Known answers from
the contract: daily floors at B_00:00 = 100,000 and 108,000, the trailing and static max floors, day
resets at 22:00 UTC (summer) / 23:00 UTC (winter) including the EU change days 2024-10-27 (25 h) and
2025-03-30 (23 h) and the US-only DST shift weeks, the best-day rule, minimum trading days, and a
hand-counted 10-day path.

US-only shift weeks (2024-10-27..2024-11-03 and 2025-03-09..2025-03-30): the contract text says the
Sunday reopen "lands in Monday's prop day"; it does not. In those weeks New York is on summer time and
Prague on winter time, so the 18:00 New York reopen is 22:00 UTC = 23:00 CET SUNDAY: that first hour is
a one-hour Sunday prop day and Monday's prop day starts at 23:00 UTC. In every other week the reopen
(22:00 UTC in summer, 23:00 UTC in winter) is 00:00 CE(S)T and opens Monday's day. The tests assert the
correct behaviour (cross-checked with zoneinfo in test_propkit_calendar.py).
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from propkit import bars as B
from propkit import calendar as cal
from propkit import rules as R
from propkit.evaluator import entry_flags, equity_arrays, evaluate_path, r_summary

C0 = 100_000.0
H = 3600
UTC = dt.timezone.utc
COLS = ["time", "balance", "equity_close", "equity_worst", "units_open"]


def ts(text: str) -> int:
    """'YYYY-MM-DD HH:MM' read as UTC -> epoch seconds."""
    return int(dt.datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=UTC).timestamp())


def dnum(text: str) -> int:
    return (dt.date.fromisoformat(text) - dt.date(1970, 1, 1)).days


def frame(rows) -> pd.DataFrame:
    """EQUITY from (time, balance, equity_close, equity_worst, units_open) rows."""
    df = pd.DataFrame(rows, columns=COLS)
    df["time"] = df["time"].astype(np.int64)
    for c in COLS[1:]:
        df[c] = df[c].astype(np.float64)
    return df


def day_rows(date: str, bars) -> list[tuple]:
    """Hourly bars from 00:00 CE(S)T of a prop day; bars = (balance, close, worst, units) tuples."""
    start = cal.day_start_utc(dnum(date))
    return [(start + k * H, *b) for k, b in enumerate(bars)]


def flat_day(date: str, balance: float, n: int = 4) -> list[tuple]:
    return day_rows(date, [(balance, balance, balance, 0.0)] * n)


def loss_frame(times, losses: dict[int, float]) -> pd.DataFrame:
    """Flat account; a realised loss (USD) in the bar opening at each given time."""
    times = np.asarray(times, dtype=np.int64)
    step = np.array([losses.get(int(t), 0.0) for t in times])
    bal = C0 - np.cumsum(step)
    return frame(list(zip(times, bal, bal, bal, np.zeros(times.size))))


def hourly(t0: str, t1: str) -> np.ndarray:
    return np.arange(ts(t0), ts(t1), H, dtype=np.int64)


def metals_hours(t0: str, t1: str) -> np.ndarray:
    t = hourly(t0, t1)
    return t[B.metals_market_open(t)]


# ---------------------------------------------------------------------------------------
# known answers: daily floors through evaluate_path

def one_dip_path(b00: float, worst: float) -> pd.DataFrame:
    """Day 1 (Mon 2024-01-08) moves the balance from C0 to b00 (flat); day 2 dips to `worst` and recovers."""
    rows = day_rows("2024-01-08", [(C0, C0, C0, 0.0), (b00, b00, min(C0, b00), 0.0)])
    rows += day_rows("2024-01-09", [(b00, b00, b00, 0.0), (b00, b00 - 10.0, worst, 1.0), (b00, b00, b00 - 10.0, 0.0)])
    return frame(rows)


@pytest.mark.parametrize("pct, rules, status", [
    (-0.0301, R.ftmo_1step(C0), "breached_daily"),
    (-0.0299, R.ftmo_1step(C0), "running"),
    (-0.0301, R.ftmo_2step(C0), "running"),
])
def test_known_answer_daily_breach_at_b00_100k(pct, rules, status):
    worst = C0 * (1 + pct)
    res = evaluate_path(one_dip_path(C0, worst), None, rules)
    assert res.status == status
    if status == "breached_daily":
        assert res.breach["kind"] == "daily" and res.breach["floor"] == 97_000.0
        assert res.breach["equity_worst"] == pytest.approx(96_990.0)
        assert res.breach["date"] == "2024-01-09"
        assert res.end_time == cal.day_start_utc(dnum("2024-01-09")) + H
    assert res.days["daily_floor"].tolist()[1] == (97_000.0 if rules.daily_loss_pct == 0.03 else 95_000.0)


@pytest.mark.parametrize("worst, status", [(104_999.99, "breached_daily"), (105_000.01, "running"),
                                           (105_000.00, "running")])
def test_known_answer_daily_floor_105k_at_b00_108k(worst, status):
    res = evaluate_path(one_dip_path(108_000.0, worst), None, R.ftmo_1step(C0))
    assert res.status == status
    day2 = res.days.iloc[1]
    assert day2["start_balance"] == 108_000.0 and day2["daily_floor"] == 105_000.0
    assert day2["max_floor"] == 98_000.0           # trailing: 108,000 - 10,000
    if res.breach:
        assert res.breach["floor"] == 105_000.0 and res.breach["kind"] == "daily"


def test_known_answer_trailing_max_floor_102k_never_moves_down():
    # B_00:00 per day: 100,000 -> 112,000 -> 109,500 -> 107,000 -> 104,500; then a dip to 101,999.99
    bals = [112_000.0, 109_500.0, 107_000.0, 104_500.0]
    dates = ["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"]
    rows = day_rows(dates[0], [(C0, C0, C0, 0.0), (bals[0], bals[0], C0, 0.0)])
    prev = bals[0]
    for date, b in zip(dates[1:4], bals[1:]):
        rows += day_rows(date, [(prev, prev, prev, 0.0), (b, b, b, 0.0)])
        prev = b
    rows += day_rows(dates[4], [(104_500.0, 104_500.0, 104_500.0, 0.0), (104_500.0, 102_500.0, 101_999.99, 1.0)])
    res = evaluate_path(frame(rows), None, R.ftmo_1step(C0))
    assert res.days["max_floor"].tolist() == [90_000.0, 102_000.0, 102_000.0, 102_000.0, 102_000.0]
    assert res.days["start_balance"].tolist() == [C0, 112_000.0, 109_500.0, 107_000.0, 104_500.0]
    assert res.status == "breached_max"
    assert res.breach["kind"] == "max" and res.breach["floor"] == 102_000.0
    assert res.breach["daily_floor"] == 101_500.0
    # the balance target (110,000) was met on day 1 but the best-day rule failed (one day = 100%)
    assert res.target_first_time == cal.day_start_utc(dnum(dates[0])) + H
    assert res.best_day_ok_at_target is False
    assert (np.diff(res.per_bar["max_floor"].to_numpy()) >= 0).all()


def test_known_answer_static_max_floor_90k():
    def path(worst):
        rows = day_rows("2024-01-08", [(C0, C0, C0, 0.0), (105_000.0, 105_000.0, C0, 0.0)])
        prev = 105_000.0
        for date, b in (("2024-01-09", 100_500.0), ("2024-01-10", 96_000.0), ("2024-01-11", 94_000.0)):
            rows += day_rows(date, [(prev, prev, prev, 0.0), (b, b, b, 0.0)])      # -4,500 / -4,500 / -2,000
            prev = b
        rows += day_rows("2024-01-12", [(94_000.0, 94_000.0, 94_000.0, 0.0), (94_000.0, 91_000.0, worst, -1.0)])
        return frame(rows)
    res = evaluate_path(path(89_999.99), None, R.ftmo_2step(C0))
    assert res.days["max_floor"].tolist() == [90_000.0] * 5      # static: does not trail the 105,000
    assert res.status == "breached_max"
    assert res.breach["floor"] == 90_000.0 and res.breach["daily_floor"] == 89_000.0
    assert evaluate_path(path(90_000.0), None, R.ftmo_2step(C0)).status == "running"


def test_breach_kind_is_the_higher_floor_and_max_on_a_tie():
    # 2-Step, B_00:00 = 95,000: daily floor 90,000 = static floor 90,000 -> 'max' on the tie
    rows = day_rows("2024-01-08", [(C0, C0, C0, 0.0), (95_000.0, 95_000.0, 95_000.0, 0.0)])
    rows += day_rows("2024-01-09", [(95_000.0, 95_000.0, 95_000.0, 0.0), (95_000.0, 91_000.0, 89_999.0, 1.0)])
    res = evaluate_path(frame(rows), None, R.ftmo_2step(C0))
    assert res.status == "breached_max"
    assert res.breach["daily_floor"] == res.breach["max_floor"] == 90_000.0


# ---------------------------------------------------------------------------------------
# day boundaries: 22:00 UTC in summer, 23:00 UTC in winter, EU change days, US-only shift weeks

def test_reset_at_2200_utc_in_summer_and_2300_utc_in_winter():
    summer = hourly("2024-07-01 18:00", "2024-07-02 03:00")
    res = evaluate_path(loss_frame(summer, {ts("2024-07-01 21:00"): 2000.0, ts("2024-07-01 22:00"): 2000.0}),
                        None, R.ftmo_1step(C0))
    assert res.status == "running"                                  # two different prop days
    pb = res.per_bar.set_index("time")["day"]
    assert pb[ts("2024-07-01 21:00")] == dnum("2024-07-01") and pb[ts("2024-07-01 22:00")] == dnum("2024-07-02")
    assert res.days.set_index("date").loc["2024-07-02", "start_balance"] == 98_000.0

    winter = hourly("2024-01-08 18:00", "2024-01-09 03:00")
    res = evaluate_path(loss_frame(winter, {ts("2024-01-08 21:00"): 2000.0, ts("2024-01-08 22:00"): 2000.0}),
                        None, R.ftmo_1step(C0))
    assert res.status == "breached_daily"                           # same prop day: 4% > 3%
    assert res.breach["time"] == ts("2024-01-08 22:00") and res.breach["date"] == "2024-01-08"
    pb = res.per_bar.set_index("time")["day"]
    assert pb[ts("2024-01-08 22:00")] == dnum("2024-01-08") and pb[ts("2024-01-08 23:00")] == dnum("2024-01-09")


def test_eu_change_day_2024_10_27_is_25_hours():
    t = hourly("2024-10-26 20:00", "2024-10-28 02:00")
    quiet = evaluate_path(loss_frame(t, {}), None, R.ftmo_1step(C0))
    days = quiet.days.set_index("date")
    assert days.loc["2024-10-27", "n_bars"] == 25
    assert quiet.per_bar.set_index("time")["day"][ts("2024-10-26 22:00")] == dnum("2024-10-27")
    assert quiet.per_bar.set_index("time")["day"][ts("2024-10-27 22:00")] == dnum("2024-10-27")  # 23:00 CET
    assert quiet.per_bar.set_index("time")["day"][ts("2024-10-27 23:00")] == dnum("2024-10-28")
    # 2% at 00:00 CEST and 2% at 23:00 CET of the same 25-hour day -> breach
    res = evaluate_path(loss_frame(t, {ts("2024-10-26 22:00"): 2000.0, ts("2024-10-27 22:00"): 2000.0}),
                        None, R.ftmo_1step(C0))
    assert res.status == "breached_daily" and res.breach["time"] == ts("2024-10-27 22:00")
    assert res.breach["date"] == "2024-10-27"
    # the same hours two days earlier are in different (summer) prop days -> no breach
    t2 = hourly("2024-10-23 20:00", "2024-10-26 02:00")
    res2 = evaluate_path(loss_frame(t2, {ts("2024-10-23 22:00"): 2000.0, ts("2024-10-24 22:00"): 2000.0}),
                         None, R.ftmo_1step(C0))
    assert res2.status == "running"


def test_eu_change_day_2025_03_30_is_23_hours():
    t = hourly("2025-03-29 20:00", "2025-03-31 02:00")
    quiet = evaluate_path(loss_frame(t, {}), None, R.ftmo_1step(C0))
    assert quiet.days.set_index("date").loc["2025-03-30", "n_bars"] == 23
    day = quiet.per_bar.set_index("time")["day"]
    assert day[ts("2025-03-29 22:00")] == dnum("2025-03-29")       # 23:00 CET
    assert day[ts("2025-03-29 23:00")] == dnum("2025-03-30")       # 00:00 CET
    assert day[ts("2025-03-30 21:00")] == dnum("2025-03-30")       # 23:00 CEST
    assert day[ts("2025-03-30 22:00")] == dnum("2025-03-31")       # 00:00 CEST
    same = evaluate_path(loss_frame(t, {ts("2025-03-29 23:00"): 2000.0, ts("2025-03-30 21:00"): 2000.0}),
                         None, R.ftmo_1step(C0))
    assert same.status == "breached_daily" and same.breach["date"] == "2025-03-30"
    split = evaluate_path(loss_frame(t, {ts("2025-03-30 21:00"): 2000.0, ts("2025-03-30 22:00"): 2000.0}),
                          None, R.ftmo_1step(C0))
    assert split.status == "running"


def test_us_only_shift_week_autumn_2024_sunday_reopen_is_its_own_prop_day():
    t = metals_hours("2024-10-25 00:00", "2024-11-05 06:00")
    losses = {ts("2024-10-27 22:00"): 2500.0, ts("2024-10-27 23:00"): 2500.0,     # US-only shift week
              ts("2024-11-03 23:00"): 2500.0, ts("2024-11-04 00:00"): 2500.0}     # normal week (both on winter time)
    res = evaluate_path(loss_frame(t, losses), None, R.ftmo_1step(C0))
    days = res.days.set_index("date")
    # Sunday 2024-10-27: the reopen hour 22:00 UTC (18:00 EDT = 23:00 CET) is a one-bar Sunday prop day
    assert days.loc["2024-10-27", "n_bars"] == 1
    assert ts("2024-10-27 22:00") in set(res.per_bar.loc[res.per_bar["day"] == dnum("2024-10-27"), "time"])
    # Monday 2024-10-28 starts at 23:00 UTC (00:00 CET) with the day-start balance after the Sunday loss
    first_monday = res.per_bar.loc[res.per_bar["day"] == dnum("2024-10-28"), "time"].min()
    assert first_monday == ts("2024-10-27 23:00")
    assert days.loc["2024-10-28", "start_balance"] == 97_500.0
    assert days.loc["2024-10-28", "daily_floor"] == 94_500.0
    # so 2.5% + 2.5% across the Sunday/Monday split is NOT one day's loss ...
    assert "2024-11-03" not in days.index                            # ... while on 2024-11-03 (US back on EST)
    first_nov4 = res.per_bar.loc[res.per_bar["day"] == dnum("2024-11-04"), "time"].min()
    assert first_nov4 == ts("2024-11-03 23:00")                      # the reopen opens Monday's day
    assert res.status == "breached_daily"                            # 2.5% + 2.5% inside Monday 2024-11-04
    assert res.breach["date"] == "2024-11-04" and res.breach["time"] == ts("2024-11-04 00:00")
    assert res.breach["floor"] == 92_000.0                           # B_00:00 95,000 - 3,000


def test_us_only_shift_weeks_spring_2025():
    t = metals_hours("2025-03-07 00:00", "2025-04-01 06:00")
    res = evaluate_path(loss_frame(t, {}), None, R.ftmo_1step(C0))
    days = res.days.set_index("date")
    for sunday in ("2025-03-09", "2025-03-16", "2025-03-23"):
        assert days.loc[sunday, "n_bars"] == 1                       # 22:00 UTC = 18:00 EDT = 23:00 CET
        monday = (dt.date.fromisoformat(sunday) + dt.timedelta(days=1)).isoformat()
        start = res.per_bar.loc[res.per_bar["day"] == dnum(monday), "time"].min()
        assert start == ts(f"{sunday} 23:00")
    assert "2025-03-02" not in days.index and "2025-03-30" not in days.index
    start_mar31 = res.per_bar.loc[res.per_bar["day"] == dnum("2025-03-31"), "time"].min()
    assert start_mar31 == ts("2025-03-30 22:00")                     # both on summer time again
    start_mar03 = res.per_bar.loc[res.per_bar["day"] == dnum("2025-03-10"), "time"].min()
    assert start_mar03 == ts("2025-03-09 23:00")
    # a 2.5% loss in the Sunday hour and 2.5% in Monday's first hour do not add up to one day
    losses = {ts("2025-03-16 22:00"): 2500.0, ts("2025-03-16 23:00"): 2500.0}
    assert evaluate_path(loss_frame(t, losses), None, R.ftmo_1step(C0)).status == "running"


# ---------------------------------------------------------------------------------------
# best-day rule, flat target, minimum trading days

def profit_days(profits, dates=None, start="2024-01-08"):
    """One trade per day realising the given profits (USD); TRADES with entries; flat at each close."""
    d0 = dt.date.fromisoformat(start)
    dates = dates or [(d0 + dt.timedelta(days=k)).isoformat() for k in range(len(profits))]
    rows, trades, bal = [], [], C0
    for k, (date, p) in enumerate(zip(dates, profits)):
        new = bal + p
        rows += day_rows(date, [(bal, bal, bal, 0.0), (bal, bal + p / 2, min(bal, bal + p / 2), 1.0),
                                (new, new, min(bal, new), 0.0)])
        s = cal.day_start_utc(dnum(date))
        trades.append({"trade_id": k, "entry_time": s + H, "exit_time": s + 2 * H, "pnl_usd": p, "risk_usd": 1000.0})
        bal = new
    tr = pd.DataFrame(trades)
    tr["entry_time"] = tr["entry_time"].astype(np.int64)
    tr["exit_time"] = tr["exit_time"].astype(np.int64)
    return frame(rows), tr


def test_known_answer_best_day_60_percent_blocks_pass_until_50_percent():
    eq, tr = profit_days([6000.0, 2000.0, 2000.0, 2000.0])
    res = evaluate_path(eq, tr, R.ftmo_1step(C0))
    day3 = cal.day_start_utc(dnum("2024-01-10"))
    assert res.target_first_time == day3 + 2 * H                     # balance 110,000 at the end of day 3
    assert res.best_day_ok_at_target is False                        # 6,000 / 10,000 = 60%
    assert res.status == "passed"
    assert res.pass_time == cal.day_start_utc(dnum("2024-01-11")) + 2 * H   # 6,000 / 12,000 = 50% exactly
    assert res.days_to_target_trading == 4 and res.days_to_target_calendar == 4
    assert res.best_day_share == pytest.approx(0.5)
    assert res.days["day_profit"].tolist() == [6000.0, 2000.0, 2000.0, 2000.0]


def test_known_answer_best_day_exactly_50_percent_passes():
    eq, tr = profit_days([5000.0, 5000.0])
    res = evaluate_path(eq, tr, R.ftmo_1step(C0))
    assert res.status == "passed" and res.best_day_ok_at_target is True
    assert res.days_to_target_trading == 2


def test_best_day_rule_equity_basis():
    # day 1 ends with an open position worth +3,000 (balance unchanged); day 2 closes it: +0 equity
    rows = day_rows("2024-01-08", [(C0, C0, C0, 0.0), (C0, 103_000.0, C0, 1.0)])
    rows += day_rows("2024-01-09", [(103_000.0, 103_000.0, 102_000.0, 0.0)])
    eq = frame(rows)
    by_bal = evaluate_path(eq, None, R.custom(profit_target_pct=None, best_day_max_share=0.5))
    by_eq = evaluate_path(eq, None, R.custom(profit_target_pct=None, best_day_max_share=0.5, best_day_basis="equity"))
    assert by_bal.days["day_profit"].tolist() == [0.0, 3000.0]
    assert by_eq.days["day_profit"].tolist() == [3000.0, 0.0]


def test_target_requires_no_open_position():
    rows = day_rows("2024-01-08", [(C0, C0, C0, 0.0), (110_000.0, 111_000.0, 110_000.0, 1.0),
                                   (110_000.0, 112_000.0, 110_500.0, 1.0), (112_000.0, 112_000.0, 111_500.0, 0.0)])
    eq = frame(rows)
    base = dict(daily_loss_pct=0.05, max_loss_pct=0.10, profit_target_pct=0.10)
    flat = evaluate_path(eq, None, R.custom(**base))
    assert flat.status == "passed" and flat.pass_index == 3
    loose = evaluate_path(eq, None, R.custom(**base, target_requires_flat=False))
    assert loose.status == "passed" and loose.pass_index == 1


def test_float_residue_in_units_open_counts_as_flat():
    rows = day_rows("2024-01-08", [(C0, C0, C0, 0.0), (110_000.0, 110_000.0, 109_000.0, 3e-13)])
    res = evaluate_path(frame(rows), None, R.custom(profit_target_pct=0.10))
    assert res.status == "passed" and res.pass_index == 1
    rows[1] = rows[1][:4] + (1e-6,)
    assert evaluate_path(frame(rows), None, R.custom(profit_target_pct=0.10)).status == "running"


def test_min_trading_days_2step():
    eq, tr = profit_days([6000.0, 4500.0, 100.0, 100.0])
    res = evaluate_path(eq, tr, R.ftmo_2step(C0))
    assert res.target_first_time is not None and res.min_days_ok_at_target is False
    assert res.status == "passed" and res.days_to_target_trading == 4
    assert res.pass_time == cal.day_start_utc(dnum("2024-01-11")) + 2 * H
    # only two trading days: the target is met but the challenge keeps running
    eq2, tr2 = profit_days([6000.0, 4500.0])
    rows = eq2.values.tolist() + flat_day("2024-01-10", 110_500.0) + flat_day("2024-01-11", 110_500.0)
    res2 = evaluate_path(frame(rows), tr2, R.ftmo_2step(C0))
    assert res2.status == "running" and res2.trading_days == 2 and res2.calendar_days == 4
    # without TRADES the trading days are inferred from units_open (same answer here)
    assert evaluate_path(eq, None, R.ftmo_2step(C0)).days_to_target_trading == 4


def test_breach_in_the_passing_bar_wins():
    rows = day_rows("2024-01-08", [(C0, C0, C0, 0.0), (110_000.0, 110_000.0, 96_000.0, 0.0)])
    res = evaluate_path(frame(rows), None, R.custom(daily_loss_pct=0.03, best_day_max_share=None))
    assert res.status == "breached_daily" and res.target_first_time is None


def test_breach_after_pass_is_reported_for_information():
    eq, tr = profit_days([5000.0, 5000.0])
    rows = eq.values.tolist() + day_rows("2024-01-10", [(110_000.0, 110_000.0, 110_000.0, 0.0),
                                                       (110_000.0, 105_000.0, 104_000.0, 1.0)])
    res = evaluate_path(frame(rows), tr, R.ftmo_1step(C0))
    assert res.status == "passed" and res.breach is None
    assert res.breach_after_pass["date"] == "2024-01-10" and res.breach_after_pass["kind"] == "daily"


# ---------------------------------------------------------------------------------------
# hand-counted 10-day path (Mon 2024-01-08 .. Fri 2024-01-19, winter: days start 23:00 UTC)

TEN_DAYS = {  # date: bars of (balance, equity_close, equity_worst, units_open)
    "2024-01-08": [(100000, 100400, 99800, 1), (100000, 100900, 100300, 1), (101500, 101500, 100800, 0),
                   (101500, 101500, 101500, 0)],
    "2024-01-09": [(101500, 100500, 100200, -1), (101500, 99900, 99700, -1), (99500, 99500, 99400, 0),
                   (99500, 99500, 99500, 0)],
    "2024-01-10": [(99500, 100500, 99300, 1), (99500, 101800, 100400, 1), (102500, 102500, 101700, 0),
                   (102500, 102500, 102500, 0)],
    "2024-01-11": [(102500, 102500, 102500, 0), (102500, 102900, 102200, 1), (102500, 103200, 102600, 1),
                   (102500, 103400, 102900, 1)],
    "2024-01-12": [(102500, 101800, 100100, 1), (101600, 101600, 101300, 0), (101600, 101600, 101600, 0),
                   (101600, 101600, 101600, 0)],
    "2024-01-15": [(101600, 100000, 99500, 1), (101600, 99000, 98600.01, 1), (99200, 99200, 98900, 0),
                   (99200, 99200, 99200, 0)],
    "2024-01-16": [(99200, 98000, 97500, -1), (99200, 96800, 96300, -1), (99200, 96500, 96199.99, -1),
                   (96600, 96600, 96400, 0)],
    "2024-01-17": [(96600, 98000, 96500, 1), (96600, 100500, 97900, 1), (101000, 101000, 100400, 0),
                   (101000, 101000, 101000, 0)],
    "2024-01-18": [(101000, 103000, 100800, 1), (101000, 105500, 102900, 1), (106000, 106000, 105400, 0),
                   (106000, 106000, 106000, 0)],
    "2024-01-19": [(106000, 108000, 105800, 1), (106000, 110300, 107900, 1), (110200, 110200, 109900, 0),
                   (110200, 110200, 110200, 0)],
}


def ten_day_path() -> pd.DataFrame:
    rows = []
    for date, bars in TEN_DAYS.items():
        rows += day_rows(date, bars)
    return frame(rows)


def test_ten_day_path_1step_daily_breach_on_day_7():
    res = evaluate_path(ten_day_path(), None, R.ftmo_1step(C0))
    assert res.status == "breached_daily"
    b = res.breach
    assert b["date"] == "2024-01-16" and b["time"] == cal.day_start_utc(dnum("2024-01-16")) + 2 * H
    assert b["time_utc"] == "2024-01-16 01:00:00 UTC"                # day starts 2024-01-15 23:00 UTC
    assert b["floor"] == 96_200.0 and b["equity_worst"] == 96_199.99 and b["max_floor"] == 92_500.0
    d = res.days
    assert d["start_balance"].tolist() == [100000, 101500, 99500, 102500, 102500, 101600, 99200, 96600, 101000, 106000]
    assert d["start_equity"].tolist()[4] == 103_400.0                # floating +900 at midnight on day 5
    assert d["daily_floor"].tolist()[:7] == [97000, 98500, 96500, 99500, 99500, 98600, 96200]
    assert d["max_floor"].tolist()[:7] == [90000, 91500, 91500, 92500, 92500, 92500, 92500]
    assert d["min_equity_worst"].tolist()[5] == 98_600.01            # one cent above the floor: no breach
    assert d["in_challenge"].tolist() == [True] * 7 + [False] * 3
    assert d["day_profit"].tolist()[:7] == [1500, -2000, 3000, 0, -900, -2400, -2600]
    assert res.trading_days == 6                                     # entries on days 1, 2, 3, 4, 6, 7
    assert res.calendar_days == 9 and res.days_with_bars == 7
    assert res.worst_daily_dd_usd == pytest.approx(3000.01) and res.worst_daily_dd_date == "2024-01-16"
    assert res.worst_daily_dd_pct == pytest.approx(3.00001)
    assert res.max_dd_usd == pytest.approx(7200.01)                  # peak close 103,400 -> worst 96,199.99
    assert res.max_dd_time == b["time"]
    assert res.target_first_time is None and res.pass_time is None
    assert res.final_balance == 99_200.0 and res.final_equity == 96_500.0


def test_ten_day_path_2step_passes_on_day_10():
    res = evaluate_path(ten_day_path(), None, R.ftmo_2step(C0))
    assert res.status == "passed" and res.breach is None
    assert res.pass_time == cal.day_start_utc(dnum("2024-01-19")) + 2 * H
    assert res.days_to_target_trading == 9                           # day 5 had no entry
    assert res.days_to_target_calendar == 12 and res.days_to_target_with_bars == 10
    assert res.target_first_time == res.pass_time and res.min_days_ok_at_target is True
    assert res.days["daily_floor"].tolist() == [95000, 96500, 94500, 97500, 97500, 96600, 94200, 91600, 96000, 101000]
    assert res.worst_daily_dd_usd == pytest.approx(3000.01)
    assert res.max_dd_usd == pytest.approx(7200.01)
    assert res.final_balance == 110_200.0


def test_ten_day_path_max_balance_equity_reference_breaches_on_day_5():
    rules = R.custom(**{**R.ftmo_1step(C0).to_dict(), "day_start_reference": "max_balance_equity"})
    res = evaluate_path(ten_day_path(), None, rules)
    assert res.status == "breached_daily"
    assert res.breach["date"] == "2024-01-12" and res.breach["floor"] == 100_400.0   # 103,400 - 3,000
    assert res.days["start_ref"].tolist()[4] == 103_400.0


def test_ten_day_path_trading_days_from_trades_match_inference():
    rows = []
    for k, (date, _) in enumerate(TEN_DAYS.items()):
        if date == "2024-01-12":
            continue
        offset = H if date == "2024-01-11" else 0
        rows.append({"trade_id": k, "entry_time": cal.day_start_utc(dnum(date)) + offset})
    tr = pd.DataFrame(rows)
    tr["entry_time"] = tr["entry_time"].astype(np.int64)
    eq = ten_day_path()
    a = equity_arrays(eq, C0)
    np.testing.assert_array_equal(entry_flags(a, tr), entry_flags(a, None))
    assert evaluate_path(eq, tr, R.ftmo_2step(C0)).days_to_target_trading == 9


# ---------------------------------------------------------------------------------------
# R summary, outputs, validation

def test_r_summary_hand_computed():
    tr = pd.DataFrame({"trade_id": [0, 1, 2, 3, 4], "exit_time": np.arange(5, dtype=np.int64) * H + 10 ** 9,
                       "pnl_usd": [100.0, -50.0, -75.0, 200.0, -100.0], "risk_usd": [50.0] * 4 + [np.nan]})
    s = r_summary(tr)
    r = np.array([2.0, -1.0, -1.5, 4.0])
    assert s["n"] == 4 and s["n_skipped"] == 1
    assert s["mean_r"] == pytest.approx(r.mean()) and s["sd_r"] == pytest.approx(r.std(ddof=1))
    assert s["se_r"] == pytest.approx(r.std(ddof=1) / 2.0)
    assert s["win_rate"] == 0.5 and s["profit_factor"] == pytest.approx(300.0 / 125.0)
    assert s["expectancy_usd"] == pytest.approx(43.75)
    assert s["max_consecutive_losses"] == 2 and s["worst_losing_run_r"] == pytest.approx(-2.5)
    assert r_summary(None) is None and r_summary(tr.drop(columns="risk_usd")) is None
    only_wins = r_summary(tr.iloc[[0, 3]])
    assert only_wins["profit_factor"] is None and only_wins["max_consecutive_losses"] == 0


def test_r_summary_is_attached_to_the_path_result():
    eq, tr = profit_days([5000.0, -1000.0, 1000.0])
    res = evaluate_path(eq, tr, R.ftmo_1step(C0))
    assert res.r_summary["n"] == 3 and res.r_summary["mean_r"] == pytest.approx(5000.0 / 3000.0)


def test_to_dict_is_json_serialisable_and_summary_is_ascii():
    res = evaluate_path(ten_day_path(), None, R.ftmo_1step(C0))
    d = res.to_dict()
    text = json.dumps(d)
    assert d["status"] == "breached_daily" and len(d["days"]) == 10
    assert d["rules"]["name"] == "FTMO 1-Step"
    "\n".join(res.summary_lines()).encode("ascii")
    text.encode("ascii")
    eq, tr = profit_days([5000.0, 5000.0])
    "\n".join(evaluate_path(eq, tr, R.ftmo_1step(C0)).summary_lines()).encode("ascii")


def test_headroom_column():
    res = evaluate_path(ten_day_path(), None, R.ftmo_1step(C0))
    pb = res.per_bar
    assert pb["headroom"].min() == pytest.approx(-0.01)
    assert (pb["headroom"].to_numpy()[: res.end_index] >= 0).all()


def test_single_bar_frame_works():
    res = evaluate_path(frame([(ts("2024-01-08 23:00"), C0, C0, 99_000.0, 0.0)]), None, R.ftmo_1step(C0))
    assert res.status == "running" and res.worst_daily_dd_usd == 1000.0


@pytest.mark.parametrize("mutate, message", [
    (lambda df: df.drop(columns="units_open"), "missing column"),
    (lambda df: df.iloc[::-1].reset_index(drop=True), "sorted"),
    (lambda df: df.assign(equity_worst=df["equity_close"] + 5.0), "above equity_close"),
    (lambda df: df.assign(balance=df["balance"].where(df.index != 3, np.nan)), "missing or infinite"),
    (lambda df: df.assign(time=df["time"] * 1000), "milliseconds"),
    (lambda df: df.assign(time=df["time"].astype(np.float64)), "int64"),
    (lambda df: df.assign(balance=df["balance"] * 2, equity_close=df["equity_close"] * 2,
                          equity_worst=df["equity_worst"] * 2), "initial capital"),
    (lambda df: df.iloc[:0], "no rows"),
])
def test_invalid_equity_raises(mutate, message):
    with pytest.raises(ValueError, match=message):
        evaluate_path(mutate(ten_day_path()), None, R.ftmo_1step(C0))


def test_bars_crossing_the_day_boundary_are_refused():
    t = np.arange(ts("2024-01-08 20:00"), ts("2024-01-10 00:00"), 4 * H, dtype=np.int64)   # H4 from 20:00
    eq = frame([(x, C0, C0, C0, 0.0) for x in t])
    with pytest.raises(ValueError, match="crosses 00:00"):
        evaluate_path(eq, None, R.ftmo_1step(C0))


def test_trades_outside_the_bars_are_refused():
    eq = ten_day_path()
    bad = pd.DataFrame({"entry_time": np.array([ts("2024-01-13 12:00")], dtype=np.int64)})     # a weekend gap
    with pytest.raises(ValueError, match="not inside any bar"):
        evaluate_path(eq, bad, R.ftmo_1step(C0))
    with pytest.raises(ValueError, match="entry_time"):
        evaluate_path(eq, pd.DataFrame({"x": [1]}), R.ftmo_1step(C0))
    with pytest.raises(ValueError, match="PropRules"):
        evaluate_path(eq, None, {"daily_loss_pct": 0.03})


def test_entry_at_a_bar_end_before_a_gap_is_placed_like_the_equity_engine():
    # review finding: equity.bar_of_instants accepts a fill at the close of the last bar before a gap
    # (t == open + bar_seconds when no bar opens there); entry_flags refused it, so a trade list that built
    # an EQUITY path then failed in evaluate_path and build_day_units. Both now use the same rule.
    from propkit import bootstrap as bs
    from propkit.costs import CostModel
    from propkit.equity import bar_of_instants, equity_from_trades
    bars = B.synthetic_bars(1704067200, 500, seed=1)
    t = bars["time"].to_numpy()
    k = int(np.flatnonzero(np.diff(t) > H)[0])             # last bar before the first gap
    entry = int(t[k]) + H                                   # its close, inside no bar
    tr = pd.DataFrame({"side": [1], "units": [10.0], "entry_time": [entry],
                       "entry_price": [float(bars["close"].iloc[k]) + 0.5], "exit_time": [int(t[k + 3])],
                       "exit_price": [float(bars["open"].iloc[k + 3])], "exit_reason": ["signal"]})
    eq, trades = equity_from_trades(bars, tr, C0, CostModel(), price_tolerance=None)
    a = equity_arrays(eq, C0)
    flags = entry_flags(a, trades)
    assert flags.sum() == 1 and flags[k] and int(bar_of_instants(t, np.array([entry]), H)[0]) == k
    res = evaluate_path(eq, trades, R.ftmo_1step(C0))
    assert res.trading_days == 1
    bs.build_day_units(eq, C0, trades)                      # no error either
    # one second later is inside the gap: still refused, by both modules
    late = trades.assign(entry_time=entry + 1)
    with pytest.raises(ValueError, match="not inside any bar"):
        entry_flags(a, late)


def test_timing_63645_bars():
    import time
    n = 63_645
    start = cal.day_start_utc(dnum("2016-01-04"))
    t = B.synthetic_bars(start, n, market_hours="metals", spread=None)["time"].to_numpy()
    rng = np.random.default_rng(3)
    bal = C0 + np.cumsum(rng.normal(0.5, 40.0, n))
    eq = frame(list(zip(t, bal, bal, bal - np.abs(rng.normal(0, 30, n)), np.zeros(n))))
    t0 = time.perf_counter()
    res = evaluate_path(eq, None, R.custom(profit_target_pct=None, daily_loss_pct=0.03, max_loss_pct=0.10,
                                           max_loss_mode="trailing_eod_balance"))
    assert time.perf_counter() - t0 < 5.0
    assert res.status == "running" and res.days_with_bars == len(res.days) > 2500


def test_module_source_is_ascii():
    import pathlib
    import propkit.evaluator as ev
    pathlib.Path(ev.__file__).read_bytes().decode("ascii")


# ---------------------------------------------------------------------------------------
# round-2 review findings

def test_r_summary_longest_losing_run_is_a_run_not_a_count():
    # R = -1, -1, +2, -1, -1, -1, +1: five losses, the longest run is 3 (sum -3), not 5
    pnl = [-100.0, -100.0, 200.0, -100.0, -100.0, -100.0, 100.0]
    tr = pd.DataFrame({"trade_id": np.arange(7), "exit_time": np.arange(7, dtype=np.int64) * H + 10 ** 9,
                       "pnl_usd": pnl, "risk_usd": [100.0] * 7})
    s = r_summary(tr)
    assert s["max_consecutive_losses"] == 3 and s["worst_losing_run_r"] == pytest.approx(-3.0)
    s = r_summary(tr.iloc[::-1].reset_index(drop=True))             # sorted by exit_time first: same answer
    assert s["max_consecutive_losses"] == 3 and s["worst_losing_run_r"] == pytest.approx(-3.0)


def _big_first_bar():
    """1,500 oz long from the first bar's open; gold falls 15 USD in that bar: equity_close of the first bar
    is 77,050 although the account started at exactly 100,000 (a valid, breaching path)."""
    from propkit.costs import CostModel
    from propkit.equity import equity_from_positions
    bars = _big_first_bar_bars()
    pos = pd.DataFrame({"time": bars["time"], "position": [1.0, 1.0, 0.0, 0.0]})
    return equity_from_positions(bars, pos, C0, CostModel(swap_enabled=False), size_mode="units", size=1500.0)


def test_a_first_bar_that_moves_more_than_20_percent_is_not_refused():
    from propkit.bootstrap import build_day_units
    eq, tr = _big_first_bar()
    assert eq["equity_close"].iloc[0] == pytest.approx(77_050.0)
    res = evaluate_path(eq, tr, R.ftmo_1step(C0))
    assert res.status == "breached_daily" and res.end_index == 0
    assert build_day_units(eq, C0, tr).n_units == 1
    # the start is checked exactly from the ledger columns: an equity built with another C0 is refused...
    from propkit.costs import CostModel
    from propkit.equity import equity_from_trades
    other, _ = equity_from_trades(_big_first_bar_bars(), tr, 105_000.0, CostModel(swap_enabled=False))
    with pytest.raises(ValueError, match="balance before the first bar is 105,000.00"):
        evaluate_path(other, tr, R.ftmo_1step(C0))
    # ... and without them the 20% heuristic stays, with a message that names both causes
    with pytest.raises(ValueError, match="moved the account by more than 20%"):
        evaluate_path(eq.drop(columns=["realised_usd", "commission_usd", "swap_usd"]), None, R.ftmo_1step(C0))


def _big_first_bar_bars() -> pd.DataFrame:
    t0 = ts("2024-01-08 08:00")
    return pd.DataFrame({"time": t0 + H * np.arange(4), "open": [2000.0, 1985.0, 1984.0, 1990.0],
                         "high": [2001.0, 1987.0, 1991.0, 1992.0], "low": [1984.0, 1983.0, 1983.5, 1989.0],
                         "close": [1985.0, 1984.0, 1990.0, 1991.0], "spread": 0.3})


def test_a_hedge_with_net_units_zero_is_not_flat_for_the_target():
    # a long of 5,000 oz closed +10.5% at bar 2, while a 10 oz long and a 10 oz short stay open until bar 100:
    # units_open is 0 (net) but two positions are open, so the target counts only once they are closed
    import dataclasses
    from propkit.bars import synthetic_bars
    from propkit.bootstrap import build_day_units
    from propkit.costs import CostModel
    from propkit.equity import equity_from_trades
    b = synthetic_bars(ts("2024-01-08 00:00"), 120, bar_seconds=3600, seed=1, spread=0.34, market_hours="always")
    t, o, sp = b["time"].to_numpy(), b["open"].to_numpy(), b["spread"].to_numpy()
    rows = [dict(side=1, units=5000.0, entry_time=t[1], entry_price=o[1] + sp[1], exit_time=t[2],
                 exit_price=o[1] + sp[1] + 2.1),
            dict(side=1, units=10.0, entry_time=t[1], entry_price=o[1] + sp[1], exit_time=t[100], exit_price=o[100]),
            dict(side=-1, units=10.0, entry_time=t[1], entry_price=o[1], exit_time=t[100],
                 exit_price=o[100] + sp[100])]
    eq, tr = equity_from_trades(b, pd.DataFrame(rows), C0, CostModel(swap_enabled=False), price_tolerance=None)
    assert eq["units_open"].iloc[2] == 0.0 and eq["balance"].iloc[2] >= 110_000.0
    assert eq["equity_close"].iloc[2] != eq["balance"].iloc[2]          # the hedge's floating PnL
    rules = dataclasses.replace(R.ftmo_2step(C0), min_trading_days=0, daily_loss_pct=None, max_loss_pct=None)
    res = evaluate_path(eq, tr, rules)
    assert res.status == "passed" and res.pass_index == 100
    a = equity_arrays(eq, C0)
    assert not a["flat"][2:100].any() and a["flat"][100:].all()
    u = build_day_units(eq, C0, tr)
    assert not (u.flat_start & (np.abs(u.u_start) > 1e-9)).any()       # no block starts with a position open


def test_two_bars_across_a_weekend_are_refused_with_the_real_reason():
    fri, sun = ts("2025-07-25 20:00"), ts("2025-07-27 22:00")
    eq = frame([(fri, C0, C0, C0, 0.0), (sun, C0, C0, C0, 0.0)])
    with pytest.raises(ValueError, match="not a bar size but a gap"):
        evaluate_path(eq, None, R.ftmo_1step(C0))
    # one bar per prop day at 00:00 CE(S)T (daily equity) is a whole-day bar and still accepted
    days = [cal.day_start_utc(dnum(d)) for d in ("2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11")]
    assert evaluate_path(frame([(x, C0, C0, C0, 0.0) for x in days]), None, R.ftmo_1step(C0)).status == "running"
