"""Tests for propkit/bootstrap.py: day units, Monte Carlo challenges, max_size, the clustered-loss gate.

EQUITY frames are built by hand (the equity engine is tested on its own). Research only.

Clustered-loss gate (contract): 250 prop days, trades per day Poisson(3) + 1, a shared day shock
N(0, 0.9% of C0) split equally over the day's trades plus per-trade N(+0.08%, 0.35%) of C0; the quantity
is P(any day <= -3%) over a 60-day challenge (the 3% daily rule alone). One 250-day history holds on
average only 0.46 days at or below -3% (population rate 0.18% per day), so the day-block answer of a single
history is either 0 or about 0.21; the expected values quoted by the contract are the averages over
histories. The gate therefore draws K independent 250-day histories from that process and compares the
averages. Population values (4,000,000 simulated days, outside the test): day-block 1 - (1 - 0.00183)^60
= 0.104 and trade-shuffle 1 - (1 - 0.00046)^60 = 0.027 (the contract quotes about 0.145 and 0.028; the
average over 250-day histories is a little below 0.104 because P is concave in the number of bad days).
Without the shock a -3% day is a one-in-70,000 event, so both answers are about 0.001 or 0; the
"agree within Monte Carlo error" check is made where the event is common enough to measure: P(a day
<= -1%) on one simulated day, with the error taken from the spread over the K histories.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import time

import numpy as np
import pandas as pd
import pytest

from propkit import bars as B
from propkit import bootstrap as bs
from propkit import calendar as cal
from propkit import rules as R
from propkit.evaluator import evaluate_path

C0 = 100_000.0
H = 3600
UTC = dt.timezone.utc
COLS = ["time", "balance", "equity_close", "equity_worst", "units_open"]
STATUS_OF = {"passed": bs.PASSED, "breached_daily": bs.BREACHED_DAILY, "breached_max": bs.BREACHED_MAX,
             "running": bs.RUNNING}


def ts(text: str) -> int:
    return int(dt.datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=UTC).timestamp())


def dnum(text: str) -> int:
    return (dt.date.fromisoformat(text) - dt.date(1970, 1, 1)).days


def frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS)
    df["time"] = df["time"].astype(np.int64)
    for c in COLS[1:]:
        df[c] = df[c].astype(np.float64)
    return df


def weekdays(n: int, start: str = "2024-01-08") -> list[int]:
    out, d = [], dnum(start)
    while len(out) < n:
        if cal.day_weekday(d) < 5:
            out.append(d)
        d += 1
    return out


def day_type_path(types, n_days: int = 200, start: str = "2024-01-08") -> pd.DataFrame:
    """A path whose days cycle through `types`; a type is a list of (d_balance, d_close, d_worst, units)
    bars relative to the day start (balance and equity both start the day at their carried values,
    every day ends flat)."""
    rows, bal = [], C0
    for k, d in enumerate(weekdays(n_days, start)):
        bars = types[k % len(types)]
        s = cal.day_start_utc(d)
        for j, (db, dc, dw, u) in enumerate(bars):
            rows.append((s + j * H, bal + db, bal + dc, bal + dw, u))
        bal = bal + bars[-1][0]
    return frame(rows)


UP_1000 = [(0.0, 400.0, -500.0, 1.0), (1000.0, 1000.0, 300.0, 0.0)]      # +1,000, dips 500 first


# ---------------------------------------------------------------------------------------
# day units

def test_build_day_units_hand_values():
    rows = [(cal.day_start_utc(dnum("2024-01-08")) + k * H, *b) for k, b in enumerate(
        [(C0, C0 + 400, C0 - 200, 1.0), (C0, C0 + 900, C0 + 100, 1.0), (C0, C0 + 900, C0 + 800, 1.0)])]
    rows += [(cal.day_start_utc(dnum("2024-01-09")) + k * H, *b) for k, b in enumerate(
        [(C0 + 500, C0 + 500, C0 + 300, 0.0)])]
    u = bs.build_day_units(frame(rows), C0)
    assert u.n_units == 2 and u.n_bars.tolist() == [3, 1]
    assert u.u_start.tolist() == [0.0, 900.0]                         # floating +900 at midnight
    np.testing.assert_array_equal(u.d_close[0], [400, 900, 900])
    np.testing.assert_array_equal(u.d_worst[0], [-200, 100, 800])
    np.testing.assert_array_equal(u.d_close[1], [-400, -400, -400])   # padded with the last value
    np.testing.assert_array_equal(u.d_worst[1], [-600, -400, -400])   # padding uses the close
    np.testing.assert_array_equal(u.d_balance[1], [500, 500, 500])
    assert u.flat.tolist() == [[False, False, False], [True, True, True]]
    assert u.entered.tolist() == [[True, True, True], [False, False, False]]
    assert u.traded.tolist() == [True, False]
    assert u.d_close_last.tolist() == [900.0, -400.0] and u.d_balance_last.tolist() == [0.0, 500.0]


def test_week_blocks_put_the_sunday_reopen_with_the_following_week():
    assert bs.week_id(dnum("2024-10-27")) == bs.week_id(dnum("2024-10-28"))      # Sunday + Monday
    assert bs.week_id(dnum("2024-11-01")) == bs.week_id(dnum("2024-10-28"))      # Friday
    assert bs.week_id(dnum("2024-11-02")) == bs.week_id(dnum("2024-10-28"))      # Saturday
    assert bs.week_id(dnum("2024-11-03")) != bs.week_id(dnum("2024-11-02"))      # next Sunday starts a block
    t = np.arange(ts("2024-10-21 00:00"), ts("2024-11-09 00:00"), H, dtype=np.int64)
    t = t[B.metals_market_open(t)]
    eq = frame([(x, C0, C0, C0, 0.0) for x in t])
    u = bs.build_day_units(eq, C0)
    first, count = u.week_blocks()
    dates = [list(cal.day_to_str(u.day[f:f + c])) for f, c in zip(first, count)]
    assert dates[1] == ["2024-10-27", "2024-10-28", "2024-10-29", "2024-10-30", "2024-10-31", "2024-11-01"]
    assert dates[2][0] == "2024-11-04" and len(dates[2]) == 5        # 2024-11-03 has no prop-day bar


def test_weeks_mode_draws_whole_week_blocks_in_order():
    t = np.arange(ts("2024-09-02 00:00"), ts("2024-12-07 00:00"), H, dtype=np.int64)
    t = t[B.metals_market_open(t)]
    units = bs.build_day_units(frame([(x, C0, C0, C0, 0.0) for x in t]), C0)
    first, count = units.week_blocks()
    block_of = np.repeat(np.arange(first.size), count)
    drawer = bs._Drawer(units, "weeks", 200, seed=4)
    seq = np.stack([drawer.day_units(d) for d in range(40)], axis=1)      # (sims, days)
    for row in seq:
        k = 0
        while k < row.size:
            b = block_of[row[k]]
            assert row[k] == first[b]                                    # a block is entered at its start
            run = row[k:k + count[b]]
            np.testing.assert_array_equal(run, np.arange(first[b], first[b] + run.size))
            k += count[b]
    # common random numbers: a second drawer with the same seed gives the same sequence
    again = bs._Drawer(units, "weeks", 200, seed=4)
    np.testing.assert_array_equal(np.stack([again.day_units(d) for d in range(40)], axis=1), seq)


# ---------------------------------------------------------------------------------------
# analytic answers

def test_identical_days_pass_deterministically():
    eq = day_type_path([UP_1000])
    for mode in ("days", "weeks"):
        res = bs.bootstrap_challenges(eq, R.ftmo_1step(C0), mode=mode, n_sims=500, seed=1)
        assert res.p_pass == 1.0 and res.se_pass == 0.0
        assert res.days_to_target_q == {f"p{q}": 10.0 for q in bs.QUANTILES}
        assert res.trading_days_to_target_q["p50"] == 10.0
        assert res.p_best_day_unmet_at_target == 0.0 and res.p_timeout == 0.0
    # the best-day rule in 2-Step is off, min trading days 4 are met long before day 10
    assert bs.bootstrap_challenges(eq, R.ftmo_2step(C0), n_sims=200).days_to_target_q["p90"] == 10.0


def test_size_multiplier_scales_every_usd_increment():
    eq = day_type_path([UP_1000])
    rules = R.ftmo_1step(C0)
    assert bs.bootstrap_challenges(eq, rules, n_sims=100, size_multiplier=2.0).days_to_target_q["p50"] == 5.0
    half = bs.bootstrap_challenges(eq, rules, n_sims=100, size_multiplier=0.5)
    assert half.days_to_target_q["p50"] == 20.0
    assert bs.bootstrap_challenges(eq, rules, n_sims=100, size_multiplier=0.5, horizon_days=15).p_timeout == 1.0
    # the intraday dip of 500 x 7 = 3,500 > 3% breaches on day 1
    big = bs.bootstrap_challenges(eq, rules, n_sims=100, size_multiplier=7.0)
    assert big.p_breach_daily == 1.0 and big.days_to_breach_q["p90"] == 1.0
    assert bs.bootstrap_challenges(eq, rules, n_sims=100, size_multiplier=0.0).p_timeout == 1.0


def test_identical_losing_days_breach_static_max_on_day_11():
    eq = day_type_path([[(0.0, -500.0, -500.0, 1.0), (-1000.0, -1000.0, -1000.0, 0.0)]])
    res = bs.bootstrap_challenges(eq, R.ftmo_2step(C0), n_sims=300, horizon_days=None)
    assert res.p_breach_max == 1.0
    assert res.days_to_breach_q == {f"p{q}": 11.0 for q in bs.QUANTILES}   # 90,000 is not below 90,000


def test_identical_losing_days_trailing_floor_1step():
    # +1,000 on day 1 then -1,000 every day: highest B_00:00 = 101,000 -> floor 91,000; equity after
    # day k >= 2 is 101,000 - 1,000 (k - 1); 91,000 is reached on day 11 and broken on day 12
    eq = day_type_path([[(1000.0, 1000.0, 0.0, 0.0)]] + [[(-1000.0, -1000.0, -1000.0, 0.0)]] * 40, n_days=41)
    res = bs.bootstrap_challenges(eq, R.ftmo_1step(C0), mode="replay")
    assert res.p_breach_max == 1.0 and res.days_to_breach_q["p50"] == 12.0


def test_two_day_types_analytic_pass_probability():
    # up: +2,000 closed; down: an intraday dip of 3,500 (> 3%) -> breach. Pass needs 5 ups in a row:
    # P(pass) = 0.5^5 = 1/32, always on day 5; P(daily breach) = 31/32 (the horizon of 60 never binds)
    up = [(0.0, 800.0, 0.0, 1.0), (2000.0, 2000.0, 700.0, 0.0)]
    down = [(0.0, -3500.0, -3500.0, 1.0), (0.0, 0.0, -3500.0, 0.0)]
    eq = day_type_path([up, down], n_days=200)
    res = bs.bootstrap_challenges(eq, R.ftmo_1step(C0), n_sims=40_000, seed=11)
    assert abs(res.p_pass - 1 / 32) < 4 * math.sqrt((1 / 32) * (31 / 32) / 40_000)
    assert res.se_pass == pytest.approx(math.sqrt(res.p_pass * (1 - res.p_pass) / 40_000))
    assert res.p_breach_daily == pytest.approx(1 - res.p_pass)
    assert res.days_to_target_q["p10"] == res.days_to_target_q["p90"] == 5.0
    # days to breach are geometric(1/2): median 1, p90 = 4 (P(T <= 3) = 0.875 < 0.9 <= P(T <= 4))
    assert res.days_to_breach_q["p50"] == 1.0
    assert 3.0 <= res.days_to_breach_q["p90"] <= 4.0
    # best-day: an up day is 2,000 of 10,000 = 20%: never unmet
    assert res.p_best_day_unmet_at_target == 0.0


def test_horizon_none_runs_until_an_event_and_caps():
    eq = day_type_path([UP_1000])
    assert bs.bootstrap_challenges(eq, R.ftmo_1step(C0), n_sims=50, horizon_days="none").p_pass == 1.0
    rules = R.custom(profit_target_pct=None, daily_loss_pct=None, max_loss_pct=None)
    res = bs.bootstrap_challenges(eq, rules, n_sims=20, horizon_days=None)
    assert res.p_timeout == 1.0 and int(res.days.max()) == bs.MAX_DAYS_UNLIMITED


# ---------------------------------------------------------------------------------------
# replay agrees with evaluate_path

TEN_DAYS = {  # the hand-counted path of test_propkit_evaluator.py
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
        s = cal.day_start_utc(dnum(date))
        rows += [(s + k * H, *b) for k, b in enumerate(bars)]
    return frame(rows)


def assert_replay_matches(eq: pd.DataFrame, rules: R.PropRules, trades=None) -> None:
    path = evaluate_path(eq, trades, rules)
    rep = bs.bootstrap_challenges(eq, rules, trades=trades, mode="replay", horizon_days=None)
    assert rep.n_sims == 1
    assert int(rep.status[0]) == STATUS_OF[path.status], (path.status, rep.status)
    if path.status != "running":
        assert int(rep.days[0]) == path.days_with_bars
    if path.passed:
        assert int(rep.trading_days[0]) == path.days_to_target_trading
    assert (rep.p_best_day_unmet_at_target == 1.0) == (path.best_day_ok_at_target is False)
    assert (rep.p_min_days_unmet_at_target == 1.0) == (path.min_days_ok_at_target is False)


@pytest.mark.parametrize("rules", [
    R.ftmo_1step(C0), R.ftmo_2step(C0), R.ftmo_2step(C0, target=0.05),
    R.custom(**{**R.ftmo_1step(C0).to_dict(), "day_start_reference": "max_balance_equity"}),
])
def test_replay_of_the_ten_day_path_matches_evaluate_path(rules):
    assert_replay_matches(ten_day_path(), rules)


def test_replay_ten_day_path_known_outcomes():
    eq = ten_day_path()
    one = bs.bootstrap_challenges(eq, R.ftmo_1step(C0), mode="replay")
    assert one.p_breach_daily == 1.0 and one.days_to_breach_q["p50"] == 7.0
    two = bs.bootstrap_challenges(eq, R.ftmo_2step(C0), mode="replay")
    assert two.p_pass == 1.0 and two.days_to_target_q["p50"] == 10.0 and two.trading_days_to_target_q["p50"] == 9.0
    assert two.horizon_days == 10


def random_path(seed: int, n_bars: int = 6000, drift: float = 1.5, vol: float = 120.0, change_prob: float = 0.15,
                jump_bar: int | None = None, jump_usd: float = 0.0) -> pd.DataFrame:
    """Random hourly path on metals hours with positions held across midnights (USD). A position (side
    -1/0/+1) changes with probability change_prob per bar; an optional one-off realised jump_usd is booked
    at bar jump_bar (a large winning day for the best-day and minimum-days cases)."""
    t = B.synthetic_bars(cal.day_start_utc(dnum("2023-03-01")), n_bars, market_hours="metals", spread=None,
                         seed=seed)["time"].to_numpy()
    rng = np.random.default_rng(seed)
    change = rng.random(n_bars) < change_prob
    side = rng.choice([-1.0, 0.0, 1.0], n_bars)
    bal, floating, units = C0, 0.0, 0.0
    rows = []
    for k in range(n_bars):
        start_eq = bal + floating
        low = start_eq
        if units != 0:
            step = rng.normal(drift, vol)
            low = start_eq + min(0.0, step) - abs(rng.normal(0.0, vol / 3))
            floating += step
        if change[k] and side[k] != units:
            bal += floating                         # close (realise) and maybe open the other way
            floating = 0.0
            units = side[k]
        if k == jump_bar:
            bal += jump_usd
        eq = bal + floating
        rows.append((t[k], bal, eq, min(low, eq), units))
    return frame(rows)


REPLAY_RULES = (
    R.ftmo_1step(C0), R.ftmo_2step(C0), R.ftmo_2step(C0, target=0.05),
    R.custom(daily_loss_pct=0.02, daily_loss_base="day_start", day_start_reference="max_balance_equity",
             max_loss_pct=0.06, max_loss_mode="trailing_eod_balance", best_day_max_share=0.4,
             best_day_basis="equity", min_trading_days=3, profit_target_pct=0.04),
    R.custom(daily_loss_pct=0.03, target_requires_flat=False, profit_target_pct=0.03, breach_inclusive=True),
)
REPLAY_PATHS = [  # (seed, vol, drift, change_prob, jump_bar, jump_usd)
    *[(s, 120.0, 1.5, 0.15, None, 0.0) for s in (1, 2, 3, 6)],
    *[(s, 300.0, 4.0, 0.15, None, 0.0) for s in (1, 3, 4)],
    *[(s, 300.0, -2.0, 0.15, None, 0.0) for s in (2, 3)],
    *[(s, 200.0, 8.0, 0.15, None, 0.0) for s in (1, 6)],
    *[(s, 150.0, 4.0, 0.15, 150, 7000.0) for s in (1, 2, 3)],          # one big day: best-day unmet
    *[(s, 100.0, 1.0, 0.01, 40, 5600.0) for s in (1, 2, 3)],           # rare entries: min days unmet
]


@pytest.mark.parametrize("seed, vol, drift, change_prob, jump_bar, jump_usd", REPLAY_PATHS)
def test_replay_of_random_paths_matches_evaluate_path(seed, vol, drift, change_prob, jump_bar, jump_usd):
    eq = random_path(seed, vol=vol, drift=drift, change_prob=change_prob, jump_bar=jump_bar, jump_usd=jump_usd)
    for rules in REPLAY_RULES:
        assert_replay_matches(eq, rules)


def test_replay_paths_cover_every_outcome():
    seen = set()
    for seed, vol, drift, change_prob, jump_bar, jump_usd in REPLAY_PATHS:
        eq = random_path(seed, vol=vol, drift=drift, change_prob=change_prob, jump_bar=jump_bar, jump_usd=jump_usd)
        for rules in REPLAY_RULES:
            res = evaluate_path(eq, None, rules)
            seen.add(res.status)
            if res.best_day_ok_at_target is False:
                seen.add("best_day_unmet_at_target")
            if res.min_days_ok_at_target is False:
                seen.add("min_days_unmet_at_target")
            if res.status == "passed" and res.target_first_time != res.pass_time:
                seen.add("passed_later_than_target")
    assert seen >= {"passed", "breached_daily", "breached_max", "running", "best_day_unmet_at_target",
                    "min_days_unmet_at_target", "passed_later_than_target"}, seen


# ---------------------------------------------------------------------------------------
# common random numbers, determinism, max_size

def test_same_seed_same_answer_and_different_seed_differs():
    eq = random_path(7)
    a = bs.bootstrap_challenges(eq, R.ftmo_1step(C0), n_sims=2000, seed=3)
    b = bs.bootstrap_challenges(eq, R.ftmo_1step(C0), n_sims=2000, seed=3)
    c = bs.bootstrap_challenges(eq, R.ftmo_1step(C0), n_sims=2000, seed=4)
    np.testing.assert_array_equal(a.status, b.status)
    np.testing.assert_array_equal(a.days, b.days)
    assert not np.array_equal(a.status, c.status) or not np.array_equal(a.days, c.days)


@pytest.mark.parametrize("mode", ["days", "weeks"])
def test_common_random_numbers_make_breaches_monotone_per_simulation(mode):
    # daily rule only, no target: a day breaches iff m x (day's dip) < -3% of C0, so under common random
    # numbers every simulation that breaches at m = 1 also breaches at m = 1.5
    eq = random_path(8, vol=250.0)
    rules = R.custom(profit_target_pct=None, daily_loss_pct=0.03, max_loss_pct=None)
    units = bs.build_day_units(eq, C0)
    lo = bs.bootstrap_challenges(None, rules, mode=mode, n_sims=3000, seed=5, units=units, size_multiplier=1.0)
    hi = bs.bootstrap_challenges(None, rules, mode=mode, n_sims=3000, seed=5, units=units, size_multiplier=1.5)
    assert 0.0 < lo.p_breach_daily < hi.p_breach_daily < 1.0
    was = lo.status == bs.BREACHED_DAILY
    assert (hi.status[was] == bs.BREACHED_DAILY).all()
    assert (hi.days[was] <= lo.days[was]).all()


def test_max_size_analytic_daily_threshold():
    # every day dips 1,000 below its start and ends flat: P(daily breach) = 1 iff m x 1,000 > 3,000
    eq = day_type_path([[(0.0, -600.0, -1000.0, 1.0), (0.0, 0.0, -1000.0, 0.0)],
                        [(0.0, 300.0, 0.0, 1.0), (0.0, 0.0, -200.0, 0.0)]])
    rules = R.custom(profit_target_pct=None, daily_loss_pct=0.03, max_loss_pct=0.10)
    res = bs.max_size(eq, rules, n_sims=500, seed=2, n_bisect=10)
    step = (4.0 - 2.5) / 2 ** 10
    assert 3.0 - step <= res.multiplier <= 3.0 and res.note == "ok"
    assert res.at["p_breach_daily"] <= 0.05
    assert res.multiplier_any == res.multiplier                      # the max floor is never the first hit
    curve = res.curve
    assert list(curve["multiplier"]) == sorted(curve["multiplier"])
    assert set(bs.DEFAULT_GRID) <= set(curve["multiplier"])
    assert curve.loc[curve["multiplier"] == 4.0, "p_breach_daily"].item() == 1.0
    json.dumps(res.to_dict())


def test_max_size_analytic_max_loss_threshold_and_grid_top():
    # every day loses 1,000 at the close: the static 10% floor breaks within 60 days iff 60 m > 10
    eq = day_type_path([[(0.0, -500.0, -500.0, 1.0), (-1000.0, -1000.0, -1000.0, 0.0)]])
    rules = R.custom(profit_target_pct=None, daily_loss_pct=None, max_loss_pct=0.10)
    res = bs.max_size(eq, rules, n_sims=200, seed=2, n_bisect=12)
    assert res.multiplier == max(bs.DEFAULT_GRID) and res.note == "at grid top"   # no daily rule
    assert 1 / 6 - (0.2 - 0.15) / 2 ** 12 <= res.multiplier_any <= 1 / 6
    assert res.at_any["p_breach_max"] == 0.0


def test_max_size_below_the_grid_bisects_from_zero():
    eq = day_type_path([[(0.0, -600.0, -1000.0, 1.0), (0.0, 0.0, -1000.0, 0.0)]])
    rules = R.custom(profit_target_pct=None, daily_loss_pct=0.03, max_loss_pct=None)
    res = bs.max_size(eq, rules, n_sims=100, grid=[4.0, 8.0], n_bisect=12)
    assert 3.0 - 4.0 / 2 ** 12 <= res.multiplier <= 3.0 and res.note == "ok"


# ---------------------------------------------------------------------------------------
# the trades mode

def test_trades_mode_deterministic_trade_pool():
    # 30 days with one +100 trade each: every simulated day is +100 -> 1-Step target on day 100
    rows, trades, bal = [], [], C0
    for k, d in enumerate(weekdays(30)):
        s = cal.day_start_utc(d)
        rows += [(s, bal, bal + 50, bal, 1.0), (s + H, bal + 100, bal + 100, bal + 50, 0.0)]
        trades.append((k, s + 60, s + H + 60, 100.0))
        bal += 100
    tr = pd.DataFrame(trades, columns=["trade_id", "entry_time", "exit_time", "pnl_usd"])
    res = bs.bootstrap_challenges(frame(rows), R.ftmo_1step(C0), trades=tr, mode="trades", n_sims=200,
                                  horizon_days=None)
    assert res.p_pass == 1.0 and res.days_to_target_q["p50"] == 100.0
    assert res.trading_days_to_target_q["p50"] == 100.0


def test_trades_mode_needs_trades():
    with pytest.raises(ValueError, match="needs TRADES"):
        bs.bootstrap_challenges(day_type_path([UP_1000]), R.ftmo_1step(C0), mode="trades", n_sims=10)


def test_trades_exiting_outside_the_equity_days_are_refused():
    eq = day_type_path([UP_1000], n_days=5)
    tr = pd.DataFrame({"trade_id": [0], "entry_time": np.array([ts("2024-02-20 01:00")], dtype=np.int64),
                       "exit_time": np.array([ts("2024-02-20 02:00")], dtype=np.int64), "pnl_usd": [1.0]})
    with pytest.raises(ValueError, match="(no bar in EQUITY|not inside any bar)"):
        bs.build_day_units(eq, C0, tr)


# ---------------------------------------------------------------------------------------
# the clustered-loss gate (contract)

def gate_history(rng: np.random.Generator, shock_sd: float, n_days: int = 250):
    """One 250-day history: trades per day Poisson(3) + 1; each trade earns shock / n + N(0.08%, 0.35%)
    of C0, the day's shock N(0, shock_sd x C0) being shared by its trades. One bar per trade (opened and
    closed inside the bar, so equity_worst = min(previous close, close)). Returns (EQUITY, TRADES)."""
    days = np.array(weekdays(n_days, "2023-01-02"), dtype=np.int64)
    n = rng.poisson(3.0, n_days) + 1
    shock = rng.normal(0.0, shock_sd * C0, n_days)
    which = np.repeat(np.arange(n_days), n)
    pnl = shock[which] / n[which] + rng.normal(0.0008 * C0, 0.0035 * C0, which.size)
    k = np.arange(which.size) - np.repeat(np.cumsum(n) - n, n)
    times = np.asarray(cal.day_start_utc(days), dtype=np.int64)[which] + H * (k + 1)
    bal = C0 + np.cumsum(pnl)
    prev = np.r_[C0, bal[:-1]]
    eq = pd.DataFrame({"time": times, "balance": bal, "equity_close": bal, "equity_worst": np.minimum(prev, bal),
                       "units_open": 0.0, "commission_usd": 0.0, "realised_usd": pnl})
    tr = pd.DataFrame({"trade_id": np.arange(pnl.size), "entry_time": times + 60, "exit_time": times + 1800,
                       "pnl_usd": pnl})
    return eq, tr


def gate_runs(shock_sd: float, daily_loss: float, horizon: int, k_histories: int, n_sims: int, seed: int = 2026):
    rules = R.custom(name="gate", profit_target_pct=None, daily_loss_pct=daily_loss, max_loss_pct=None)
    p_day, p_trade = [], []
    for k in range(k_histories):
        eq, tr = gate_history(np.random.default_rng([seed, k]), shock_sd)
        units = bs.build_day_units(eq, C0, tr)
        for mode, out in (("days", p_day), ("trades", p_trade)):
            out.append(bs.bootstrap_challenges(None, rules, mode=mode, n_sims=n_sims, seed=k, units=units,
                                               horizon_days=horizon).p_breach_daily)
    return np.array(p_day), np.array(p_trade)


K_GATE = 100


def test_gate_clustered_losses_day_blocks_breach_more_than_trade_shuffle():
    t0 = time.perf_counter()
    p_day, p_trade = gate_runs(0.009, 0.03, 60, K_GATE, 1000)
    day, trade = p_day.mean(), p_trade.mean()
    se_day = p_day.std(ddof=1) / math.sqrt(K_GATE)
    assert day > 2.0 * trade, (day, trade)
    assert abs(day - 0.104) < 3.5 * se_day + 0.02, (day, se_day)                 # population 0.104
    assert 0.017 < trade < 0.040, trade                                           # population 0.027
    diff = p_day - p_trade
    assert diff.mean() > 3.0 * diff.std(ddof=1) / math.sqrt(K_GATE)
    assert time.perf_counter() - t0 < 60.0


def test_gate_without_day_shock_the_two_agree():
    # at -3% both are about zero (a one-in-70,000 day without the shock)
    p_day, p_trade = gate_runs(0.0, 0.03, 60, 30, 1000)
    assert p_day.mean() < 0.005 and p_trade.mean() < 0.005
    # where the event is common (a day <= -1%, about 4% of days) they agree within Monte Carlo error,
    # the error being the spread over independent histories (history sampling + resampling)
    p_day, p_trade = gate_runs(0.0, 0.01, 1, K_GATE, 2000)
    diff = p_day - p_trade
    se = diff.std(ddof=1) / math.sqrt(K_GATE)
    assert abs(diff.mean()) <= 3.0 * se, (p_day.mean(), p_trade.mean(), se)
    assert 0.02 < p_day.mean() < 0.06
    # and with the shock, the same one-day check separates them by many standard errors
    s_day, s_trade = gate_runs(0.009, 0.01, 1, K_GATE, 2000)
    sdiff = s_day - s_trade
    assert sdiff.mean() > 8.0 * sdiff.std(ddof=1) / math.sqrt(K_GATE)


# ---------------------------------------------------------------------------------------
# timing (contract: 10,000 challenges x 60 days from about 2,600 prop days of H1 in under 20 s)

def test_timing_10000_challenges_from_2600_prop_days():
    eq = random_path(9, n_bars=59_000, vol=60.0, drift=0.5)
    t0 = time.perf_counter()
    units = bs.build_day_units(eq, C0)
    assert 2500 <= units.n_units <= 2800
    for mode in ("days", "weeks"):
        res = bs.bootstrap_challenges(None, R.ftmo_1step(C0), mode=mode, n_sims=10_000, seed=7, units=units)
        assert res.n_sims == 10_000
        total = res.p_pass + res.p_breach_daily + res.p_breach_max + res.p_timeout
        assert total == pytest.approx(1.0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 20.0, elapsed


# ---------------------------------------------------------------------------------------
# validation and output

@pytest.mark.parametrize("kwargs, message", [
    ({"mode": "blocks"}, "mode"),
    ({"n_sims": 0}, "n_sims"),
    ({"n_sims": 2.5}, "n_sims"),
    ({"seed": -1}, "seed"),
    ({"horizon_days": 0}, "horizon_days"),
    ({"horizon_days": "forever"}, "horizon_days"),
    ({"size_multiplier": -1.0}, "size_multiplier"),
    ({"size_multiplier": float("inf")}, "size_multiplier"),
])
def test_invalid_arguments_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bs.bootstrap_challenges(day_type_path([UP_1000], n_days=5), R.ftmo_1step(C0), **kwargs)


def test_rules_must_match_the_units_capital():
    units = bs.build_day_units(day_type_path([UP_1000], n_days=5), C0)
    with pytest.raises(ValueError, match="initial capital"):
        bs.bootstrap_challenges(None, R.ftmo_1step(200_000), units=units, n_sims=10)
    with pytest.raises(ValueError, match="EQUITY"):
        bs.bootstrap_challenges(None, R.ftmo_1step(C0), n_sims=10)
    with pytest.raises(ValueError, match="alpha"):
        bs.max_size(None, R.ftmo_1step(C0), units=units, alpha=5)
    with pytest.raises(ValueError, match="replay"):
        bs.max_size(None, R.ftmo_1step(C0), units=units, mode="replay")
    with pytest.raises(ValueError, match="grid"):
        bs.max_size(None, R.ftmo_1step(C0), units=units, grid=[0.0, 1.0])


def test_result_dict_is_json_and_summary_is_ascii():
    res = bs.bootstrap_challenges(random_path(10), R.ftmo_1step(C0), n_sims=500, seed=1)
    d = res.to_dict()
    json.dumps(d)
    assert "status" not in d and d["n_sims"] == 500 and d["mode"] == "days"
    for key in ("pass", "breach_daily", "breach_max", "timeout", "best_day_unmet_at_target"):
        p, se = d[f"p_{key}"], d[f"se_{key}"]
        assert 0.0 <= p <= 1.0 and se == pytest.approx(math.sqrt(p * (1 - p) / 500))
    "\n".join(res.summary_lines()).encode("ascii")


def test_module_source_is_ascii_and_imports_nothing_from_alphamaster():
    import pathlib
    import re
    import propkit.bootstrap as mod
    import propkit.evaluator as ev
    import propkit.rules as ru
    for m in (mod, ev, ru):
        text = pathlib.Path(m.__file__).read_bytes().decode("ascii")
        assert not re.search(r"^\s*(from|import)\s+(model_core|data_pipeline|config|web|utils|strategy_manager|"
                             r"execution|scripts|scipy|zoneinfo)\b", text, re.M)


# ---------------------------------------------------------------------------------------
# positions held over midnight: flat-to-flat blocks (review finding: iid days overstated P(daily breach))

def test_flat_start_marks_days_without_an_open_position_and_cuts_blocks():
    rows = []
    plan = {"2024-01-08": 1.0, "2024-01-09": 1.0, "2024-01-10": 0.0, "2024-01-11": 0.0, "2024-01-12": 1.0}
    for date, last_units in plan.items():
        s = cal.day_start_utc(dnum(date))
        rows += [(s, C0, C0, C0, 0.0), (s + H, C0, C0, C0, last_units)]
    t = cal.day_start_utc(dnum("2024-01-15"))
    rows += [(t, C0, C0, C0, 0.0)]
    u = bs.build_day_units(frame(rows), C0)
    # a day starts flat when the previous day's last bar is flat; the first day always does
    assert u.starts_flat.tolist() == [True, False, False, True, True, False]
    first, count = u.day_blocks()
    assert first.tolist() == [0, 3, 4] and count.tolist() == [3, 1, 2]
    wfirst, wcount = u.week_blocks()        # the Friday hold joins the two weeks
    assert wfirst.tolist() == [0] and wcount.tolist() == [6]
    s = u.block_summary()
    assert s["n_blocks_days"] == 3 and s["max_block_days_days"] == 3 and s["n_blocks_weeks"] == 1
    assert u.share_open_at_start == pytest.approx(3 / 6)


def test_day_flat_history_gives_one_day_blocks_and_the_iid_draws():
    eq = day_type_path([UP_1000, [(0.0, -300.0, -800.0, 1.0), (-500.0, -500.0, -500.0, 0.0)]], n_days=40)
    units = bs.build_day_units(eq, C0)
    first, count = units.day_blocks()
    assert first.tolist() == list(range(40)) and (count == 1).all()
    drawer = bs._Drawer(units, "days", 300, seed=9)
    for d in range(5):     # the same draws as a plain uniform day draw (unchanged from the iid version)
        np.testing.assert_array_equal(drawer.day_units(d), np.random.default_rng([9, 1, d]).integers(0, 40, 300))


def test_every_block_and_every_challenge_starts_flat():
    eq = random_path(12, n_bars=4000, change_prob=0.05)
    units = bs.build_day_units(eq, C0)
    assert 0.2 < units.share_open_at_start < 0.95
    for mode in ("days", "weeks"):
        drawer = bs._Drawer(units, mode, 500, seed=3)
        seq = np.stack([drawer.day_units(d) for d in range(30)], axis=1)
        assert units.starts_flat[seq[:, 0]].all()                   # day 1 of every challenge is flat
        # a simulated day either continues the historical sequence or jumps to a flat block start, and a
        # day whose successor in history starts with a position open is always followed by that successor
        cont = seq[:, 1:] == seq[:, :-1] + 1
        assert (cont | units.starts_flat[seq[:, 1:]]).all()
        nxt = seq[:, :-1] + 1
        must = (nxt < units.n_units) & ~units.starts_flat[np.minimum(nxt, units.n_units - 1)]
        assert cont[must].all()


def _carry_history(seed: int, n_days: int = 1300, size: float = 30.0, q_in: float = 0.02, q_out: float = 0.01):
    """Additive-price H1 history (constant USD volatility) and a position that enters +-1 with
    probability q_in per bar when flat and exits with q_out per bar: holds of days, flat spells between."""
    from propkit.costs import CostModel
    from propkit.equity import equity_from_positions
    rng = np.random.default_rng(seed)
    n = n_days * 23
    cand = 1704672000 + 3600 * np.arange(int(n * 1.6) + 400, dtype=np.int64)
    t = cand[B.metals_market_open(cand)][:n]
    c = 5000.0 + np.cumsum(rng.normal(0.0, 4.0, n))
    o = np.r_[5000.0, c[:-1]] + rng.normal(0.0, 0.3, n)
    bars = pd.DataFrame({"time": t, "open": o, "high": np.maximum(o, c) + np.abs(rng.normal(0.0, 2.0, n)),
                         "low": np.minimum(o, c) - np.abs(rng.normal(0.0, 2.0, n)), "close": c})
    u, side = rng.random(n), np.where(rng.random(n) < 0.5, 1.0, -1.0)
    pos, cur = np.zeros(n), 0.0
    for k in range(1, n):
        if cur == 0.0:
            cur = side[k] if u[k] < q_in else 0.0
        elif u[k] < q_out:
            cur = 0.0
        pos[k] = cur
    cm = CostModel(fixed_spread=0.0, spread_source="fixed", swap_enabled=False)
    eq, _ = equity_from_positions(bars, pd.DataFrame({"time": t, "position": pos}), C0, cm, size_mode="units",
                                  size=size, price_tolerance=None)
    return eq


def _fresh_challenge_windows(eq: pd.DataFrame, rules: R.PropRules, horizon: int, stride: int) -> list[str]:
    """Ground truth: every `horizon`-day window of the history that starts FLAT at a prop-day boundary,
    rebased to C0 and run through evaluate_path as a fresh challenge (stride: days between starts)."""
    from propkit.evaluator import equity_arrays
    a = equity_arrays(eq, C0)
    starts, units, bal = a["starts"], a["units"], a["balance"]
    flat_start = np.r_[True, units[starts[1:] - 1] == 0]
    out, d = [], 0
    while d + horizon <= starts.size:
        if not flat_start[d]:
            d += 1
            continue
        i0, i1 = starts[d], (starts[d + horizon] if d + horizon < starts.size else len(eq))
        w = eq.iloc[i0:i1].reset_index(drop=True).copy()
        off = (bal[i0 - 1] if i0 > 0 else C0) - C0
        for col in ("balance", "equity_close", "equity_worst"):
            w[col] = w[col] - off
        out.append(evaluate_path(w, None, rules).status)
        d += stride
    return out


def test_overnight_holds_day_blocks_match_fresh_challenge_windows():
    # Same-history calibration (the reviewer's design): positions held over midnight on most days.
    # Truth = 60-market-day windows that start flat, as fresh challenges; the day-block bootstrap of the
    # same histories must agree. The old iid-day chaining (every day its own unit, B = E - u_start of the
    # drawn day) put P(daily breach) at about 0.58 where the truth is about 0.25.
    import dataclasses
    rules = R.ftmo_1step(C0)
    truth, boot, iid, shares = [], [], [], []
    for h in range(4):
        eq = _carry_history(h)
        units = bs.build_day_units(eq, C0)
        shares.append(units.share_open_at_start)
        truth += _fresh_challenge_windows(eq, rules, 60, 5)
        kw = dict(mode="days", n_sims=2000, seed=h, horizon_days=60, horizon_unit="market")
        boot.append(bs.bootstrap_challenges(None, rules, units=units, **kw).p_breach_daily)
        old = dataclasses.replace(units, flat_start=np.ones(units.n_units, dtype=bool))
        iid.append(bs.bootstrap_challenges(None, rules, units=old, **kw).p_breach_daily)
    p_true = float(np.mean(np.array(truth) == "breached_daily"))
    p_boot, p_iid = float(np.mean(boot)), float(np.mean(iid))
    assert 0.5 < np.mean(shares) < 0.8                  # most days start with a position open
    assert 0.15 < p_true < 0.35, p_true
    assert abs(p_boot - p_true) < 0.06, (p_boot, p_true)
    assert p_iid > p_true + 0.2, (p_iid, p_true)        # the bias the blocks remove


def test_trading_day_horizon_counts_days_with_an_entry():
    idle = [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)]           # no entry: not a trading day
    up = [(0.0, 50.0, -50.0, 1.0), (100.0, 100.0, 50.0, 0.0)]
    eq = day_type_path([up, idle], n_days=100)
    rules = R.custom(profit_target_pct=None, daily_loss_pct=0.03, max_loss_pct=0.10)
    trading = bs.bootstrap_challenges(eq, rules, n_sims=400, seed=1, horizon_days=10)
    market = bs.bootstrap_challenges(eq, rules, n_sims=400, seed=1, horizon_days=10, horizon_unit="market")
    assert trading.horizon_unit == "trading" and market.horizon_unit == "market"
    assert trading.p_timeout == market.p_timeout == 1.0
    assert (market.days == 10).all()
    assert (trading.trading_days == 0).all()                       # trading_days is filled at a pass only
    assert trading.days.min() >= 10 and trading.days.mean() > 15  # about 20 market days for 10 trading days
    with pytest.raises(ValueError, match="horizon_unit"):
        bs.bootstrap_challenges(eq, rules, n_sims=10, horizon_unit="calendar")


def test_trading_day_horizon_ends_right_after_the_last_trading_day():
    up = [(0.0, 50.0, -50.0, 1.0), (100.0, 100.0, 50.0, 0.0)]
    eq = day_type_path([up], n_days=50)
    rules = R.custom(profit_target_pct=None, daily_loss_pct=0.03, max_loss_pct=0.10)
    res = bs.bootstrap_challenges(eq, rules, n_sims=50, seed=1, horizon_days=7)
    assert (res.days == 7).all() and res.p_timeout == 1.0         # every day trades: same as market days


def test_history_uncertainty_is_wider_than_the_monte_carlo_error():
    up = [(0.0, 800.0, 0.0, 1.0), (2000.0, 2000.0, 700.0, 0.0)]
    down = [(0.0, -1500.0, -1600.0, 1.0), (-1500.0, -1500.0, -1600.0, 0.0)]
    eq = day_type_path([up, down, up, up, down], n_days=60)
    rules = R.ftmo_1step(C0)
    one = bs.bootstrap_challenges(eq, rules, n_sims=4000, seed=2)
    hu = bs.history_uncertainty(eq, rules, n_reps=40, n_sims=500, seed=2, alpha=0.05)
    assert hu["n_reps"] == 40 and len(hu["values"]["p_pass"]) == 40 and hu["n_blocks"] == 60
    q = hu["p_pass"]
    assert q["p5"] <= q["p50"] <= q["p95"]
    assert q["p95"] - q["p5"] > 8 * one.se_pass                  # the history part dwarfs the MC error
    m = hu["max_size_multiplier"]
    assert m is not None and 0.0 < m["p5"] <= m["p50"] <= m["p95"] <= 64.0
    json.dumps(hu)
    # identical days: every resampled history is the same history, so there is no spread at all
    same = bs.history_uncertainty(day_type_path([UP_1000], n_days=30), rules, n_reps=5, n_sims=200, alpha=None)
    assert same["p_pass"] == {"p5": 1.0, "p50": 1.0, "p95": 1.0} and same["max_size_multiplier"] is None


def test_days_breaching_alone_counts_days_whose_own_drawdown_breaks_the_daily_limit():
    eq = day_type_path([[(0.0, -600.0, -1000.0, 1.0), (0.0, 0.0, -1000.0, 0.0)],
                        [(0.0, 300.0, -200.0, 1.0), (0.0, 0.0, -200.0, 0.0)]], n_days=10)
    units = bs.build_day_units(eq, C0)
    one = R.ftmo_1step(C0)
    assert bs.days_breaching_alone(units, one, 2.9) == 0
    assert bs.days_breaching_alone(units, one, 3.1) == 5            # the five 1,000-dip days
    assert bs.days_breaching_alone(units, one, 16.0) == 10
    assert bs.days_breaching_alone(units, R.custom(daily_loss_pct=None), 5.0) is None


def test_prob_text_uses_the_rule_of_three_at_zero_and_one():
    assert bs.prob_text(0.0, 0.0, 10_000) == "0.0000 (< 0.0003)"
    assert bs.prob_text(1.0, 0.0, 1000) == "1.0000 (> 0.9970)"
    assert bs.prob_text(0.25, 0.0043, 10_000) == "0.2500 +- 0.0043"
    assert bs.prob_text(None, None) == "n/a"


# ---------------------------------------------------------------------------------------
# round-2 review findings: size scaling of u_start, common random numbers, max_size envelope, the best-day
# boundary and the breach-kind tie inside the bootstrap

def _hold_over_midnight_path(n_days: int = 100) -> pd.DataFrame:
    """Two-day blocks that hold a position over midnight (USD, relative to the block's start balance b):
    day A opens a long that floats +2,000 at midnight (balance b, equity b + 2,000); day B starts with
    u_start = 2,000, dips to equity b - 800 (2,800 below E_00:00, only 800 below B_00:00) and closes flat
    at b. So at size m day B falls m x 800 below its 00:00 balance: a 3% daily breach iff m > 3.75."""
    day_a = [(0.0, 1000.0, -200.0, 1.0), (0.0, 2000.0, 900.0, 1.0)]
    day_b = [(0.0, -800.0, -800.0, 1.0), (0.0, 0.0, -800.0, 0.0)]
    return day_type_path([day_a, day_b], n_days=n_days)


DAILY_ONLY = R.custom(profit_target_pct=None, daily_loss_pct=0.03, max_loss_pct=None)


def test_size_multiplier_scales_the_open_position_at_day_start():
    # u_start must scale with the size: B_00:00 = E_00:00 - m x u_start. Unscaled, day B would fall
    # m x 2,800 - 2,000 below its 00:00 balance and breach from m = 1.79 on instead of 3.75.
    eq = _hold_over_midnight_path()
    units = bs.build_day_units(eq, C0)
    assert units.u_start[1::2].tolist() == [2000.0] * 50 and units.share_open_at_start == 0.5
    for m, want in ((1.0, 0.0), (2.0, 0.0), (3.7, 0.0), (3.8, 1.0), (5.0, 1.0)):
        res = bs.bootstrap_challenges(None, DAILY_ONLY, units=units, n_sims=200, seed=1, size_multiplier=m,
                                      horizon_days=10)
        assert res.p_breach_daily == want, (m, res.p_breach_daily)
    ms = bs.max_size(None, DAILY_ONLY, units=units, n_sims=200, seed=1, horizon_days=10, n_bisect=12)
    assert 3.75 - (4.0 - 3.0) / 2 ** 12 <= ms.multiplier <= 3.75 and ms.note == "ok"


def _scaled(eq: pd.DataFrame, m: float) -> pd.DataFrame:
    """The same path at m x the size: every USD amount measured from C0 multiplied by m."""
    out = eq.copy()
    for col in ("balance", "equity_close", "equity_worst"):
        out[col] = C0 + m * (eq[col] - C0)
    out["units_open"] = eq["units_open"] * m
    return out


@pytest.mark.parametrize("m", [0.5, 2.0, 3.0])
@pytest.mark.parametrize("seed", [1, 3])
def test_replay_at_size_m_equals_evaluate_path_of_the_scaled_path(seed, m):
    # paths that hold over midnight on many days: every USD increment of the day units, the floating PnL
    # carried over midnight (u_start) included, must scale with m
    eq = random_path(seed, vol=200.0, drift=2.0)
    assert (eq["units_open"] != 0).mean() > 0.4
    for rules in REPLAY_RULES:
        path = evaluate_path(_scaled(eq, m), None, rules)
        rep = bs.bootstrap_challenges(eq, rules, mode="replay", horizon_days=None, size_multiplier=m)
        assert int(rep.status[0]) == STATUS_OF[path.status], (rules.name, m, path.status, rep.status)
        if path.status != "running":
            assert int(rep.days[0]) == path.days_with_bars


@pytest.mark.parametrize("mode", ["days", "weeks", "trades"])
def test_stacked_multipliers_equal_separate_runs(mode):
    # the stacked run behind max_size / history_uncertainty must give every multiplier exactly the draws of
    # a separate run with the same seed (common random numbers)
    if mode == "trades":
        eq, tr = gate_history(np.random.default_rng(5), 0.009)
        units = bs.build_day_units(eq, C0, tr)
    else:
        units = bs.build_day_units(random_path(12, n_bars=4000, change_prob=0.05, vol=250.0), C0)
        assert units.day_blocks()[1].max() > 1                     # multi-day blocks are exercised
    rules = R.ftmo_1step(C0)
    mults = [0.5, 1.0, 2.0, 3.5]
    stacked = bs._run_many(units, rules, mode, 1500, 9, 60, mults, "trading")
    outcomes = set()
    for m, st in zip(mults, stacked):
        one = bs.bootstrap_challenges(None, rules, units=units, mode=mode, n_sims=1500, seed=9, size_multiplier=m)
        np.testing.assert_array_equal(st.status, one.status)
        np.testing.assert_array_equal(st.days, one.days)
        np.testing.assert_array_equal(st.trading_days, one.trading_days)
        assert st.to_dict() == one.to_dict()
        outcomes |= set(np.unique(one.status).tolist())
    assert len(outcomes) >= 3                                        # not a degenerate all-same case


def test_max_size_uses_common_random_numbers_at_every_point():
    # daily rule only: a simulation breaches at m iff some day falls m x (its drop below B_00:00) < -3% of
    # C0, so under common random numbers its breach indicator can only switch on as m grows, and every
    # point of max_size's curve (grid and bisection) is the separate run with the same seed
    eq = random_path(8, vol=250.0)
    units = bs.build_day_units(eq, C0)
    kw = dict(units=units, n_sims=1500, seed=5, horizon_days=20)
    res = bs.max_size(None, DAILY_ONLY, alpha=0.05, n_bisect=6, **kw)
    curve = res.curve
    assert len(curve) > len(bs.DEFAULT_GRID) and res.note == "ok"     # bisection points were added
    assert curve["p_breach_daily"].is_monotonic_increasing
    prev = None
    for m, p in zip(curve["multiplier"], curve["p_breach_daily"]):
        one = bs.bootstrap_challenges(None, DAILY_ONLY, size_multiplier=float(m), **kw)
        assert one.p_breach_daily == p, m
        breach = one.status == bs.BREACHED_DAILY
        if prev is not None:
            assert not (prev & ~breach).any(), m                       # breached at a smaller size -> here too
        prev = breach
    assert res.at["p_breach_daily"] <= 0.05
    above = curve.loc[curve["multiplier"] > res.multiplier, "p_breach_daily"]
    assert (above > 0.05).all()
    extra = curve.loc[~curve["multiplier"].isin(bs.DEFAULT_GRID), "p_breach_daily"]
    assert ((extra > 0) & (extra < 1)).any()                           # bisection points where the draws matter


class _FakeRun:
    """Stands in for a BootstrapResult with a chosen breach probability (max_size envelope tests)."""

    def __init__(self, m: float, p: float):
        self.size_multiplier, self.p_breach_daily, self.p_breach_any = m, p, p
        self.p_pass = self.p_breach_max = self.p_timeout = 0.0
        self.se_breach_daily = self.se_breach_any = 0.0

    def to_dict(self):
        return {"size_multiplier": self.size_multiplier, "p_breach_daily": self.p_breach_daily}


def test_max_size_allows_a_size_only_if_every_smaller_size_is_allowed(monkeypatch):
    # a non-monotone Monte Carlo curve p = 0.01, 0.06, 0.04, 0.07 with alpha 0.05: the answer is the first
    # grid point (1.0), never 3.0 where p dipped back under alpha
    curve = {1.0: 0.01, 2.0: 0.06, 3.0: 0.04, 4.0: 0.07}
    monkeypatch.setattr(bs, "_run_many", lambda units, rules, mode, n, seed, h, mults, unit, chunk=8:
                        [_FakeRun(float(x), curve[float(x)]) for x in mults])
    units = bs.build_day_units(day_type_path([UP_1000], n_days=5), C0)
    res = bs.max_size(None, DAILY_ONLY, units=units, grid=[1.0, 2.0, 3.0, 4.0], n_bisect=0, alpha=0.05)
    assert res.multiplier == 1.0 and res.multiplier_any == 1.0 and res.note == "ok"


def test_quick_max_size_uses_the_running_maximum(monkeypatch):
    # p(m) = 0.06 on [0.1, 0.2) and 0.01 elsewhere below 3: the running maximum puts the crossing just below
    # 0.1 (every size from 0.1 on has a smaller size that breaches too often), not near 3
    def p_of(m: float) -> float:
        return 0.06 if 0.1 <= m < 0.2 else (0.5 if m >= 3.0 else 0.01)
    monkeypatch.setattr(bs, "_run_many", lambda units, rules, mode, n, seed, h, mults, unit, chunk=8:
                        [_FakeRun(float(x), p_of(float(x))) for x in mults])
    m, _ = bs._quick_max_size(None, DAILY_ONLY, 0.05, 100, 1, 60, "trading")
    assert 0.09 <= m < 0.1, m


def _profit_days(profits, extra_flat_days: int = 3) -> pd.DataFrame:
    """One prop day per closed profit (USD, flat at every midnight), then flat days without trades."""
    rows, bal = [], C0
    days = weekdays(len(profits) + extra_flat_days)
    for d, x in zip(days, list(profits) + [None] * extra_flat_days):
        s = cal.day_start_utc(d)
        if x is None:
            rows += [(s, bal, bal, bal, 0.0), (s + H, bal, bal, bal, 0.0)]
        else:
            rows += [(s, bal, bal + x / 2, bal, 1.0), (s + H, bal + x, bal + x, bal + x / 2, 0.0)]
            bal = bal + x
    return frame(rows)


@pytest.mark.parametrize("profits", [
    [2500.0, 2500.0, 5000.0],              # best day exactly 50% of 10,000 at the target close
    [1405.29, 3739.01, 5144.30],           # 50% up to float noise: best - 0.5 x total = +7e-12 USD
])
def test_bootstrap_best_day_boundary_matches_the_evaluator(profits):
    eq = _profit_days(profits)
    rules = R.ftmo_1step(C0)
    path = evaluate_path(eq, None, rules)
    assert path.status == "passed" and path.days_to_target_with_bars == 3 and path.best_day_ok_at_target
    rep = bs.bootstrap_challenges(eq, rules, mode="replay", horizon_days=None)
    assert rep.p_pass == 1.0 and rep.days_to_target_q["p50"] == 3.0 and rep.p_best_day_unmet_at_target == 0.0
    assert_replay_matches(eq, rules)
    # one cent more on the best day breaks the rule in both
    worse = _profit_days(profits[:-1] + [profits[-1] + 0.01])
    assert evaluate_path(worse, None, rules).status == "running"
    assert bs.bootstrap_challenges(worse, rules, mode="replay", horizon_days=None).p_pass == 0.0


def test_bootstrap_breach_kind_on_a_floor_tie_is_max_as_in_the_evaluator():
    # daily 3% of C0 below B_00:00 and a static max floor of 90,000: at B_00:00 = 93,000 both floors are
    # 90,000; equity 89,999.99 breaches both in the same bar and the contract's tie rule says 'max'
    rules = R.custom(profit_target_pct=None, daily_loss_pct=0.03, daily_loss_base="initial",
                     day_start_reference="balance", max_loss_pct=0.10, max_loss_mode="static")
    rows, bal = [], C0
    for d, (loss, low) in zip(weekdays(4), [(2900.0, 2900.0), (2900.0, 2900.0), (1200.0, 1200.0),
                                            (0.0, 3000.01)]):
        s = cal.day_start_utc(d)
        rows += [(s, bal, bal - low / 2, bal - low / 2, 1.0), (s + H, bal - loss, bal - loss, bal - low, 0.0)]
        bal -= loss
    eq = frame(rows)
    path = evaluate_path(eq, None, rules)
    assert path.status == "breached_max"
    assert path.breach["daily_floor"] == path.breach["max_floor"] == 90_000.0
    res = bs.bootstrap_challenges(eq, rules, mode="replay", horizon_days=None)
    assert int(res.status[0]) == bs.BREACHED_MAX and res.p_breach_max == 1.0 and res.p_breach_daily == 0.0
    assert_replay_matches(eq, rules)


def test_a_net_zero_hedge_is_not_flat():
    # units_open is the NET size: a long and a short of 10 oz held together give 0 while two positions are
    # open. Equity differs from balance by their floating PnL, so the day boundary is not flat (u_start != 0
    # stays inside its block) and the target is not taken while they are open.
    rows = []
    for d, (eq_end, units_end) in zip(weekdays(3), [(C0 - 6.8, 0.0), (C0 - 4.0, 0.0), (C0, 0.0)]):
        s = cal.day_start_utc(d)
        rows += [(s, C0, C0 - 3.4, C0 - 3.4, 0.0), (s + H, C0, eq_end, C0 - 7.0, units_end)]
    eq = frame(rows)                                                    # the hedge is closed on day 3
    units = bs.build_day_units(eq, C0)
    assert units.starts_flat.tolist() == [True, False, False]
    assert units.u_start.tolist() == pytest.approx([0.0, -6.8, -4.0])
    assert not units.flat[0].any() and units.flat[2, -1]
