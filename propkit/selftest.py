"""propkit/selftest.py - the contract's known-answer gates, runnable on any PC in a few seconds.

`python -m propkit selftest` runs every gate below and prints PASS or FAIL per gate (ASCII). Each gate
rebuilds a small hand-made or synthetic case and compares propkit's answer with the answer worked out by
hand (the arithmetic is in the comments). No market data, no network, no files are needed. A FAIL means
this installation does not reproduce the known answers: do not use its results until it is fixed.

Gates (contract section in brackets):
  rules/evaluator [A]: daily floors at B_00:00 100,000 and 108,000; trailing and static max floors; day
    resets at 22:00 / 23:00 UTC; the EU change days; the US-only DST shift weeks (the Sunday 18:00 New York
    reopen is its own one-hour Sunday prop day, see propkit.calendar; the contract text said Monday);
    the best-day rule; a hand-counted 10-day path.
  bootstrap [A]: identical days give the analytic answer; two day types give P(pass) = 1/32; replay equals
    evaluate_path; clustered losses breach more often with day blocks than with trade shuffling; a
    position held over midnight keeps its days in one flat-to-flat block and every challenge starts flat.
  costs/equity/adapters [B]: the 3-trade hand example to the cent; the flat rate; equity_worst marks;
    swap timing and the US DST rollover shift; leverage re-sizing; the AlphaMaster one-bar shift and log PnL.
  stats [C]: the Bailey and Lopez de Prado (2014) DSR example (SR0 0.1132, DSR 0.9004); PSR and DSR
    checks; sharpe_stats' SR standard error on simulated IID normal returns; hand answers of sharpe_stats,
    expected_shortfall, daily_returns_from_equity and drawdown_stats.
  pullback [D]: the hand path to the cent; gap through the stop; stop first; truncation (no look-ahead);
    mirror.
  integration: synthetic bars -> trades and positions -> equity -> rules -> bootstrap -> report; source scan.
Research only.
"""
from __future__ import annotations

import ast
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from propkit import adapters
from propkit import bars as B
from propkit import bootstrap as bs
from propkit import calendar as cal
from propkit import rules as R
from propkit import stats
from propkit.costs import CostModel
from propkit.equity import equity_from_positions, equity_from_trades
from propkit.evaluator import evaluate_path
from propkit.pullback import PullbackSpec, compute_signals, generate_trades, generate_trades_detailed

C0 = 100_000.0
H = 3600
COLS = ["time", "balance", "equity_close", "equity_worst", "units_open"]
FORBIDDEN_IMPORTS = {"model_core", "data_pipeline", "config", "web", "utils", "strategy_manager", "execution",
                     "scripts", "scipy", "zoneinfo", "torch"}


class GateFailure(AssertionError):
    """A gate's answer differs from the known answer."""


def check(cond, message: str) -> None:
    """Raise GateFailure(message) unless cond is true."""
    if not cond:
        raise GateFailure(message)


def near(a, b, tol: float = 1e-6) -> bool:
    """True when |a - b| <= tol (absolute, in the units of a and b)."""
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------------------
# small builders

def ts(text: str) -> int:
    """'YYYY-MM-DD HH:MM[:SS]' read as UTC -> epoch seconds."""
    return int(np.datetime64(text.replace(" ", "T"), "s").astype(np.int64))


def dnum(text: str) -> int:
    """'YYYY-MM-DD' -> days since 1970-01-01."""
    return int(np.datetime64(text, "D").astype(np.int64))


def frame(rows) -> pd.DataFrame:
    """EQUITY from (time, balance, equity_close, equity_worst, units_open) rows."""
    df = pd.DataFrame(rows, columns=COLS)
    df["time"] = df["time"].astype(np.int64)
    for c in COLS[1:]:
        df[c] = df[c].astype(np.float64)
    return df


def day_rows(date: str, bars) -> list[tuple]:
    """Hourly rows from 00:00 CE(S)T of a prop day; bars = (balance, close, worst, units) tuples."""
    start = cal.day_start_utc(dnum(date))
    return [(start + k * H, *b) for k, b in enumerate(bars)]


def loss_frame(times, losses: dict[int, float]) -> pd.DataFrame:
    """A flat account with a realised loss (USD) in the bar opening at each given time."""
    times = np.asarray(times, dtype=np.int64)
    step = np.array([losses.get(int(t), 0.0) for t in times])
    bal = C0 - np.cumsum(step)
    return frame(list(zip(times, bal, bal, bal, np.zeros(times.size))))


def hourly(t0: str, t1: str, metals: bool = False) -> np.ndarray:
    """Hourly bar opens (UTC epoch s) from t0 up to t1 (excluded), metals hours only if asked."""
    t = np.arange(ts(t0), ts(t1), H, dtype=np.int64)
    return t[B.metals_market_open(t)] if metals else t


def one_dip_path(b00: float, worst: float) -> pd.DataFrame:
    """Mon 2024-01-08 moves the balance from C0 to b00 (flat); Tue 2024-01-09 dips to `worst`."""
    rows = day_rows("2024-01-08", [(C0, C0, C0, 0.0), (b00, b00, min(C0, b00), 0.0)])
    rows += day_rows("2024-01-09", [(b00, b00, b00, 0.0), (b00, b00 - 10.0, worst, 1.0), (b00, b00, b00 - 10.0, 0.0)])
    return frame(rows)


# ---------------------------------------------------------------------------------------
# [A] rules and the evaluator

def gate_daily_floor_b00_100k() -> str:
    """Gate A1: daily floor at B_00:00 100,000 (1-Step / 2-Step).

    Returns a one-line detail; raises GateFailure on a wrong answer."""
    # floor 1-Step = 100,000 - 3% x 100,000 = 97,000; 2-Step = 95,000
    out = []
    for pct, rules, want in ((-0.0301, R.ftmo_1step(C0), "breached_daily"), (-0.0299, R.ftmo_1step(C0), "running"),
                             (-0.0301, R.ftmo_2step(C0), "running")):
        res = evaluate_path(one_dip_path(C0, C0 * (1 + pct)), None, rules)
        check(res.status == want, f"{rules.name} at {pct:+.2%}: {res.status}, expected {want}")
        out.append(res.status)
    return "1-Step -3.01% breaches, -2.99% holds; 2-Step -3.01% holds"


def gate_daily_floor_b00_108k() -> str:
    """Gate A2: daily floor at B_00:00 108,000 = 105,000.

    Returns a one-line detail; raises GateFailure on a wrong answer."""
    # B_00:00 108,000 -> 1-Step floor 108,000 - 3,000 = 105,000 (strictly below breaches)
    check(R.ftmo_1step(C0).daily_floor(108_000.0) == 105_000.0, "floor is not 105,000")
    for worst, want in ((104_999.99, "breached_daily"), (105_000.01, "running"), (105_000.00, "running")):
        res = evaluate_path(one_dip_path(108_000.0, worst), None, R.ftmo_1step(C0))
        check(res.status == want, f"equity {worst:,.2f}: {res.status}, expected {want}")
    return "floor 105,000.00; 104,999.99 breaches, 105,000.01 and 105,000.00 hold"


def gate_trailing_max_floor() -> str:
    """Gate A3: trailing max floor 102,000, never down.

    Returns a one-line detail; raises GateFailure on a wrong answer."""
    # highest B_00:00 112,000 -> floor 112,000 - 10,000 = 102,000, kept while the balance falls
    bals = [112_000.0, 109_500.0, 107_000.0, 104_500.0]
    dates = ["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"]
    rows = day_rows(dates[0], [(C0, C0, C0, 0.0), (bals[0], bals[0], C0, 0.0)])
    prev = bals[0]
    for date, b in zip(dates[1:4], bals[1:]):
        rows += day_rows(date, [(prev, prev, prev, 0.0), (b, b, b, 0.0)])
        prev = b
    rows += day_rows(dates[4], [(104_500.0, 104_500.0, 104_500.0, 0.0), (104_500.0, 102_500.0, 101_999.99, 1.0)])
    res = evaluate_path(frame(rows), None, R.ftmo_1step(C0))
    check(res.days["max_floor"].tolist() == [90_000.0] + [102_000.0] * 4, f"floors {res.days['max_floor'].tolist()}")
    check(res.status == "breached_max" and res.breach["floor"] == 102_000.0, f"status {res.status}")
    check((np.diff(res.per_bar["max_floor"].to_numpy()) >= 0).all(), "the trailing floor moved down")
    return "floor 102,000 after a 112,000 day-start balance, never moves down; 101,999.99 breaches"


def gate_static_max_floor() -> str:
    """Gate A4: 2-Step static max floor 90,000. Returns a one-line detail; raises GateFailure on a wrong answer."""
    two = R.ftmo_2step(C0)
    check(two.max_floor() == 90_000.0 and two.max_floor(130_000.0) == 90_000.0, "2-Step floor is not 90,000")
    check(two.breached(89_999.99, 90_000.0) and not two.breached(90_000.0, 90_000.0), "breach test")
    return "2-Step floor 90,000 whatever the balance"


def gate_day_reset_summer_winter() -> str:
    """Gate A5: prop day reset 22:00 / 23:00 UTC. Returns a one-line detail; raises GateFailure on a wrong answer."""
    # summer: 21:00 and 22:00 UTC are different prop days (2% + 2% does not breach)
    summer = hourly("2024-07-01 18:00", "2024-07-02 03:00")
    res = evaluate_path(loss_frame(summer, {ts("2024-07-01 21:00"): 2000.0, ts("2024-07-01 22:00"): 2000.0}),
                        None, R.ftmo_1step(C0))
    check(res.status == "running", f"summer: {res.status}")
    # winter: 21:00 and 22:00 UTC are the same prop day (4% > 3%: breach); 23:00 UTC starts the next day
    winter = hourly("2024-01-08 18:00", "2024-01-09 03:00")
    res = evaluate_path(loss_frame(winter, {ts("2024-01-08 21:00"): 2000.0, ts("2024-01-08 22:00"): 2000.0}),
                        None, R.ftmo_1step(C0))
    check(res.status == "breached_daily" and res.breach["time"] == ts("2024-01-08 22:00"), f"winter: {res.status}")
    check(cal.prop_day(ts("2024-01-08 23:00")) == dnum("2024-01-09"), "23:00 UTC in winter is not the new day")
    check(cal.prop_day(ts("2024-07-01 22:00")) == dnum("2024-07-02"), "22:00 UTC in summer is not the new day")
    return "00:00 CE(S)T = 22:00 UTC in summer, 23:00 UTC in winter"


def gate_eu_change_days() -> str:
    """Gate A6: EU change days 25 h / 23 h. Returns a one-line detail; raises GateFailure on a wrong answer."""
    t = hourly("2024-10-26 20:00", "2024-10-28 02:00")
    n_oct = evaluate_path(loss_frame(t, {}), None, R.ftmo_1step(C0)).days.set_index("date").loc["2024-10-27", "n_bars"]
    t2 = hourly("2025-03-29 20:00", "2025-03-31 02:00")
    n_mar = evaluate_path(loss_frame(t2, {}), None, R.ftmo_1step(C0)).days.set_index("date").loc["2025-03-30", "n_bars"]
    check(n_oct == 25 and n_mar == 23, f"2024-10-27 has {n_oct} hours, 2025-03-30 has {n_mar}")
    return "2024-10-27 is 25 hours, 2025-03-30 is 23 hours"


def gate_us_only_shift_weeks() -> str:
    """Gate A7: US-only DST shift weeks. Returns a one-line detail; raises GateFailure on a wrong answer."""
    # US on summer time, EU still on winter time: 18:00 New York = 22:00 UTC = 23:00 CET Sunday, so the
    # reopen hour is a one-hour SUNDAY prop day and Monday starts at 23:00 UTC (00:00 CET)
    t = np.concatenate([hourly("2024-10-25 00:00", "2024-11-05 06:00", metals=True),
                        hourly("2025-03-07 00:00", "2025-04-01 06:00", metals=True)])
    res = evaluate_path(loss_frame(t, {}), None, R.ftmo_1step(C0))
    days = res.days.set_index("date")
    first = res.per_bar.groupby("day")["time"].min()
    for sunday in ("2024-10-27", "2025-03-09", "2025-03-16", "2025-03-23"):
        check(days.loc[sunday, "n_bars"] == 1, f"{sunday} is not a one-hour prop day")
        check(first[dnum(sunday) + 1] == ts(f"{sunday} 23:00"), f"Monday after {sunday} does not start at 23:00 UTC")
    check("2024-11-03" not in days.index and "2025-03-30" not in days.index, "a normal-week reopen made a Sunday day")
    check(first[dnum("2024-11-04")] == ts("2024-11-03 23:00") and first[dnum("2025-03-31")] == ts("2025-03-30 22:00"),
          "normal weeks: the reopen does not open Monday's day")
    losses = {ts("2025-03-16 22:00"): 2500.0, ts("2025-03-16 23:00"): 2500.0}
    check(evaluate_path(loss_frame(t, losses), None, R.ftmo_1step(C0)).status == "running",
          "2.5% Sunday + 2.5% Monday counted as one day")
    return "Sunday reopen 22:00 UTC is its own 1-hour prop day (2024-10-27, 2025-03-09/16/23)"


def gate_best_day_rule() -> str:
    """Gate A8: best-day rule 60% fails, 50% passes. Returns a one-line detail; raises GateFailure on a wrong answer."""
    one = R.ftmo_1step(C0)
    check(one.best_day_ok([6000.0, 2000.0, 2000.0]) is False, "60% passed")
    check(one.best_day_ok([5000.0, 5000.0]) is True and one.best_day_ok([6000.0, 2000.0, 2000.0, 2000.0]) is True,
          "exactly 50% failed")
    # through evaluate_path: +6,000 then +2,000 a day; the target 110,000 is met on day 3 (60%) and the
    # pass comes on day 4 (6,000 / 12,000 = 50%)
    rows, bal = [], C0
    for date, p in zip(("2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"), (6000.0, 2000.0, 2000.0, 2000.0)):
        new = bal + p
        rows += day_rows(date, [(bal, bal, bal, 0.0), (bal, bal + p / 2, bal, 1.0), (new, new, bal, 0.0)])
        bal = new
    res = evaluate_path(frame(rows), None, one)
    check(res.best_day_ok_at_target is False and res.status == "passed", f"{res.status}")
    check(res.pass_time == cal.day_start_utc(dnum("2024-01-11")) + 2 * H, "passed on the wrong bar")
    return "60% of positive profit fails, exactly 50% passes"


TEN_DAYS = {  # date: bars of (balance, equity_close, equity_worst, units_open); winter, days start 23:00 UTC
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
    """EQUITY of the hand-counted 10-day path (TEN_DAYS), C0 = 100,000 USD."""
    rows = []
    for date, bars in TEN_DAYS.items():
        rows += day_rows(date, bars)
    return frame(rows)


def gate_ten_day_path() -> str:
    """Gate A9: hand-counted 10-day path. Returns a one-line detail; raises GateFailure on a wrong answer."""
    # 1-Step: day 7 (2024-01-16) starts at B 99,200 -> floor 96,200; equity_worst 96,199.99 at its 3rd bar.
    # Day 6 touched 98,600.01 against its floor 98,600 (one cent above: no breach).
    res = evaluate_path(ten_day_path(), None, R.ftmo_1step(C0))
    b = res.breach or {}
    check(res.status == "breached_daily" and b.get("date") == "2024-01-16" and b.get("floor") == 96_200.0,
          f"1-Step: {res.status} {b}")
    check(res.trading_days == 6 and res.calendar_days == 9, f"days {res.trading_days}/{res.calendar_days}")
    check(near(res.max_dd_usd, 7200.01), f"max dd {res.max_dd_usd}")   # peak close 103,400 -> 96,199.99
    # 2-Step: target 110,000 at the 2nd bar of day 10 (110,200 closed and flat), 9 trading days
    two = evaluate_path(ten_day_path(), None, R.ftmo_2step(C0))
    check(two.status == "passed" and two.days_to_target_trading == 9 and two.days_to_target_calendar == 12,
          f"2-Step: {two.status} {two.days_to_target_trading} {two.days_to_target_calendar}")
    return "1-Step breaches daily on day 7 at floor 96,200; 2-Step passes on day 10 after 9 trading days"


# ---------------------------------------------------------------------------------------
# [A] bootstrap

def _weekdays(n: int, start: str = "2024-01-08") -> list[int]:
    out, d = [], dnum(start)
    while len(out) < n:
        if cal.day_weekday(d) < 5:
            out.append(d)
        d += 1
    return out


def day_type_path(types, n_days: int = 200) -> pd.DataFrame:
    """Days cycling through `types` (lists of (d_balance, d_close, d_worst, units) per bar), each ending flat."""
    rows, bal = [], C0
    for k, d in enumerate(_weekdays(n_days)):
        bars = types[k % len(types)]
        s = cal.day_start_utc(d)
        for j, (db, dc, dw, u) in enumerate(bars):
            rows.append((s + j * H, bal + db, bal + dc, bal + dw, u))
        bal = bal + bars[-1][0]
    return frame(rows)


def gate_bootstrap_analytic() -> str:
    """Gate A10: bootstrap analytic answers. Returns a one-line detail; raises GateFailure on a wrong answer."""
    # identical days of +1,000 (dip 500 first): the 10% target is reached on day 10 in every simulation
    up = [(0.0, 400.0, -500.0, 1.0), (1000.0, 1000.0, 300.0, 0.0)]
    res = bs.bootstrap_challenges(day_type_path([up]), R.ftmo_1step(C0), n_sims=300, seed=1)
    check(res.p_pass == 1.0 and res.days_to_target_q["p50"] == 10.0, f"identical days: {res.p_pass}")
    # up (+2,000) or down (intraday -3,500 > 3%: breach) with equal odds: P(pass) = 0.5^5 = 1/32 on day 5
    up2 = [(0.0, 800.0, 0.0, 1.0), (2000.0, 2000.0, 700.0, 0.0)]
    down = [(0.0, -3500.0, -3500.0, 1.0), (0.0, 0.0, -3500.0, 0.0)]
    res = bs.bootstrap_challenges(day_type_path([up2, down]), R.ftmo_1step(C0), n_sims=20_000, seed=11)
    se = math.sqrt((1 / 32) * (31 / 32) / 20_000)
    check(abs(res.p_pass - 1 / 32) < 4 * se, f"P(pass) {res.p_pass:.4f}, expected 1/32 = 0.03125")
    check(res.days_to_target_q["p10"] == res.days_to_target_q["p90"] == 5.0, "pass not on day 5")
    return f"identical days pass on day 10 with P = 1; two day types P(pass) = {res.p_pass:.4f} (1/32 = 0.0313)"


def gate_bootstrap_replay() -> str:
    """Gate A11: bootstrap replay = evaluate_path. Returns a one-line detail; raises GateFailure on a wrong answer."""
    for rules in (R.ftmo_1step(C0), R.ftmo_2step(C0)):
        eq = ten_day_path()
        path = evaluate_path(eq, None, rules)
        rep = bs.bootstrap_challenges(eq, rules, mode="replay", horizon_days=None)
        want = {"passed": bs.PASSED, "breached_daily": bs.BREACHED_DAILY, "breached_max": bs.BREACHED_MAX,
                "running": bs.RUNNING}[path.status]
        check(int(rep.status[0]) == want and int(rep.days[0]) == path.days_with_bars,
              f"{rules.name}: replay {rep.status[0]} vs {path.status}")
    return "replaying the history day by day gives evaluate_path's outcome (1-Step and 2-Step)"


def _gate_history(rng: np.random.Generator, shock_sd: float, n_days: int = 250):
    """250 prop days; trades per day Poisson(3) + 1; each trade earns shock / n + N(0.08%, 0.35%) of C0."""
    days = np.array(_weekdays(n_days, "2023-01-02"), dtype=np.int64)
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


def gate_clustered_losses() -> str:
    """Gate A12: clustered losses: day blocks > 2 x trade shuffle.

    Returns a one-line detail; raises GateFailure on a wrong answer."""
    # a shared day shock N(0, 0.9% of C0) clusters losses inside days: resampling whole days keeps that,
    # shuffling trades does not. Averaged over 40 independent histories (one history holds about 0.46
    # days at -3% or worse, too few on its own). Population values: 0.104 (days) vs 0.027 (trades).
    rules = R.custom(name="gate", profit_target_pct=None, daily_loss_pct=0.03, max_loss_pct=None)
    p_day, p_trade = [], []
    for k in range(40):
        eq, tr = _gate_history(np.random.default_rng([2026, k]), 0.009)
        units = bs.build_day_units(eq, C0, tr)
        for mode, out in (("days", p_day), ("trades", p_trade)):
            out.append(bs.bootstrap_challenges(None, rules, mode=mode, n_sims=500, seed=k, units=units,
                                               horizon_days=60).p_breach_daily)
    day, trade = float(np.mean(p_day)), float(np.mean(p_trade))
    check(day > 2.0 * trade, f"day blocks {day:.3f} not above 2 x trade shuffle {trade:.3f}")
    return f"P(a -3% day in 60 days): day blocks {day:.3f} > 2 x trade shuffle {trade:.3f}"


# ---------------------------------------------------------------------------------------
# [B] costs, equity, adapters

def make_bars(first: str, last_end: str, overrides: dict, spread: float = 0.30, metals: bool = True) -> pd.DataFrame:
    """H1 bars; bars named in overrides get (open, high, low, close, spread), the rest are flat."""
    times = hourly(first, last_end, metals=metals)
    over = {ts(k): v for k, v in overrides.items()}
    rows, last = [], 2000.0
    for t in times:
        if int(t) in over:
            o, h, lo, c, s = over[int(t)]
            last = c
        else:
            o = h = lo = c = last
            s = spread
        rows.append((int(t), float(o), float(h), float(lo), float(c), float(s)))
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "spread"])


def gate_flat_to_flat_blocks() -> str:
    """Gate A13: flat-to-flat blocks: overnight holds stay together.

    Returns a one-line detail; raises GateFailure on a wrong answer."""
    # Mon 8 and Tue 9 Jan end with a position open, Wed 10 and Thu 11 end flat, Fri 12 holds over the weekend:
    # days starting flat = Mon, Thu, Fri (first day) and not Tue, Wed (carried in), not Mon 15 (weekend hold)
    rows = []
    for date, last_units in (("2024-01-08", 1.0), ("2024-01-09", 1.0), ("2024-01-10", 0.0), ("2024-01-11", 0.0),
                             ("2024-01-12", 1.0)):
        start = cal.day_start_utc(dnum(date))
        rows += [(start, C0, C0, C0, 0.0), (start + H, C0, C0, C0, last_units)]
    rows += [(cal.day_start_utc(dnum("2024-01-15")), C0, C0, C0, 0.0)]
    units = bs.build_day_units(frame(rows), C0)
    check(units.starts_flat.tolist() == [True, False, False, True, True, False], "flat day starts")
    first, count = units.day_blocks()
    check(first.tolist() == [0, 3, 4] and count.tolist() == [3, 1, 2], f"day blocks {first} {count}")
    wfirst, wcount = units.week_blocks()
    check(wfirst.tolist() == [0] and wcount.tolist() == [6], "the weekend hold must join the two weeks")
    drawer = bs._Drawer(units, "days", 2000, seed=4)
    seq = np.stack([drawer.day_units(d) for d in range(8)], axis=1)
    check(units.starts_flat[seq[:, 0]].all(), "a simulated challenge started with a position open")
    nxt = seq[:, 1:]
    must = ~units.starts_flat[np.minimum(seq[:, :-1] + 1, units.n_units - 1)] & (seq[:, :-1] + 1 < units.n_units)
    check((nxt[must] == seq[:, :-1][must] + 1).all(), "a carried day was separated from the day before it")
    return "3 day blocks (3, 1, 2 days), 1 week block; 2,000 challenges all start flat, carried days stay joined"


def gate_three_trade_hand_example() -> str:
    """Gate B1: 3-trade hand example to the cent. Returns a one-line detail; raises GateFailure on a wrong answer."""
    # July 2024 (New York EDT, rollover 17:00 New York = 21:00 UTC). Costs: markup 0.05 + slippage 0.03 per
    # side, commission 7 USD per 100-oz lot round trip, swap 6%/yr long, 2%/yr short, triple on Wednesday.
    cm = CostModel(markup_per_side=0.05, slippage_per_side=0.03, commission_per_lot_round_trip=7.0,
                   swap_long=6.0, swap_short=2.0, triple_swap_weekday=2)
    bars = make_bars("2024-07-09 00:00", "2024-07-12 21:00", {
        "2024-07-09 08:00": (2000.0, 2003.0, 1998.0, 2001.0, 0.30),
        "2024-07-09 14:00": (2008.0, 2011.0, 2007.0, 2010.5, 0.30),
        "2024-07-10 15:00": (2020.0, 2021.0, 2019.0, 2020.0, 0.30),
        "2024-07-10 18:00": (2020.0, 2023.0, 2018.0, 2019.0, 0.30),
        "2024-07-10 20:00": (2019.0, 2019.5, 2015.0, 2016.0, 0.30),
        "2024-07-11 10:00": (2005.0, 2006.0, 2004.0, 2005.5, 0.40),
        "2024-07-11 12:00": (2030.0, 2031.0, 2029.0, 2030.0, 0.30),
        "2024-07-11 20:00": (2030.0, 2030.0, 2024.0, 2025.0, 0.30),
        "2024-07-11 22:00": (2015.0, 2016.0, 2012.0, 2014.0, 0.30)})
    trades = pd.DataFrame({
        "trade_id": [1, 2, 3], "side": [1, -1, 1], "units": [100.0, 50.0, 200.0],
        "entry_time": [ts("2024-07-09 08:00"), ts("2024-07-10 15:00"), ts("2024-07-11 12:00")],
        "entry_price": [2000.38, 2019.92, 2030.38],          # ask + 0.08 / bid - 0.08 / ask + 0.08
        "exit_time": [ts("2024-07-09 14:30"), ts("2024-07-11 10:00"), ts("2024-07-11 22:00")],
        "exit_price": [2009.92, 2005.48, 2014.92],           # target bid - 0.08 / ask 2005.40 + 0.08 / gap open
        "exit_reason": ["target", "signal", "stop"], "stop_price": [np.nan, np.nan, 2020.0]})
    eq, tr = equity_from_trades(bars, trades, C0, cm)
    # 1: 100 x 9.54 - 7.00 = 947.00; 2: 50 x 14.44 - 3.50 - (0.02 x 50 x 2016 / 360 x 3 = 16.80) = 701.70;
    # 3: 200 x -15.46 - 14.00 - (0.06 x 200 x 2025 / 360 = 67.50) = -3173.50; 1R of 3: the stop 2020.00 (bid)
    # fills at 2019.92 after markup + slippage, 200 x 10.46 + 14 = 2106.00
    want = [947.00, 701.70, -3173.50]
    check(all(near(a, b) for a, b in zip(tr["pnl_usd"], want)), f"pnl {tr['pnl_usd'].tolist()}")
    check(near(tr["swap_usd"].iloc[1], -16.80) and near(tr["risk_usd"].iloc[2], 2106.00),
          f"swap {tr['swap_usd'].iloc[1]:.2f} or 1R {tr['risk_usd'].iloc[2]:.2f}")
    check(near(eq["balance"].iloc[-1], 98_475.20), f"final balance {eq['balance'].iloc[-1]:.2f}")
    return "pnl 947.00 / 701.70 / -3,173.50 USD, 1R of trade 3 2,106.00 (stop fill 2019.92), final balance 98,475.20"


def gate_flat_rate() -> str:
    """Gate B2: flat rate 0.0003 per fill. Returns a one-line detail; raises GateFailure on a wrong answer."""
    bars = make_bars("2024-01-08 00:00", "2024-01-08 06:00", {
        "2024-01-08 02:00": (2000.0, 2001.0, 1999.0, 2000.0, 0.30),
        "2024-01-08 04:00": (2010.0, 2011.0, 2009.0, 2010.0, 0.30)}, metals=False)
    cm = CostModel(flat_rate_per_side=0.0003, swap_enabled=False)
    tr = pd.DataFrame({"side": [1], "units": [10.0], "entry_time": [ts("2024-01-08 02:00")], "entry_price": [2000.0],
                       "exit_time": [ts("2024-01-08 04:00")], "exit_price": [2010.0], "exit_reason": ["signal"]})
    eq, out = equity_from_trades(bars, tr, C0, cm)
    # 0.0003 x 10 x 2000 = 6.00 at entry, 0.0003 x 10 x 2010 = 6.03 at exit
    check(near(eq["commission_usd"].iloc[2], 6.00) and near(eq["commission_usd"].iloc[4], 6.03), "fees per fill")
    check(near(out["pnl_usd"].iloc[0], 87.97), f"pnl {out['pnl_usd'].iloc[0]}")
    return "0.0003 x notional at each fill: 6.00 + 6.03 USD"


def gate_equity_marks() -> str:
    """Gate B3: equity_worst and marks. Returns a one-line detail; raises GateFailure on a wrong answer."""
    bars = make_bars("2024-01-08 00:00", "2024-01-08 06:00", {
        "2024-01-08 03:00": (2000.0, 2006.0, 1993.0, 2001.0, 0.70)}, metals=False)
    cm = CostModel(swap_enabled=False)
    for side, entry, worst, close in ((1, 2000.30, -73.00, 7.00), (-1, 2000.00, -67.00, -17.00)):
        tr = pd.DataFrame({"side": [side], "units": [10.0], "entry_time": [ts("2024-01-08 01:00")],
                           "entry_price": [entry], "exit_time": [ts("2024-01-08 05:00")],
                           "exit_price": [2000.0 if side > 0 else 2000.30], "exit_reason": ["signal"]})
        eq, _ = equity_from_trades(bars, tr, C0, cm)
        r = eq.iloc[3]
        # long: bid low 1993 -> 10 x (1993 - 2000.30) = -73; short: ask high 2006.70 -> 10 x (2000 - 2006.70) = -67
        check(near(r["equity_worst"], C0 + worst) and near(r["equity_close"], C0 + close), f"side {side}")
        flat = eq["units_open"] == 0
        check((eq.loc[flat, "equity_close"] == eq.loc[flat, "balance"]).all(), "equity_close != balance when flat")
    # opened at a bar open: marked from its fill 2000.80 (bid 2000 + spread 0.50 + 0.10 + 0.20), not the open
    bars2 = make_bars("2024-01-08 00:00", "2024-01-08 04:00", {
        "2024-01-08 01:00": (2000.0, 2004.0, 1999.0, 2002.0, 0.50)}, metals=False)
    cm2 = CostModel(markup_per_side=0.10, slippage_per_side=0.20, commission_per_lot_round_trip=10.0,
                    swap_enabled=False)
    tr = pd.DataFrame({"side": [1], "units": [100.0], "entry_time": [ts("2024-01-08 01:00")],
                       "entry_price": [cm2.buy_fill(2000.0, 0.50)], "exit_time": [ts("2024-01-08 03:00")],
                       "exit_price": [2001.70], "exit_reason": ["time"]})
    eq, _ = equity_from_trades(bars2, tr, C0, cm2)
    check(near(eq["equity_close"].iloc[1], C0 - 5.00 + 120.00) and near(eq["equity_worst"].iloc[1], C0 - 5.00 - 180.00),
          "position opened at the open is not marked from its fill")
    return "long worst at bid low, short at bid high + spread; marked from the fill; flat close = balance"


def gate_swap_timing() -> str:
    """Gate B4: swap timing and US DST rollover. Returns a one-line detail; raises GateFailure on a wrong answer."""
    # January 2024 (EST): rollover 17:00 New York = 22:00 UTC; 3.6%/yr / 360 of 100 oz x 2000 = 20.00 a night
    bars = make_bars("2024-01-08 00:00", "2024-01-12 22:00", {})
    cm = CostModel(swap_long=3.6, swap_short=3.6, triple_swap_weekday=2)

    def swap(entry: str, exit_: str, side: int = 1) -> float:
        tr = pd.DataFrame({"side": [side], "units": [100.0], "entry_time": [ts(entry)], "entry_price": [2000.3],
                           "exit_time": [ts(exit_)], "exit_price": [2000.0 if side > 0 else 2000.3]})
        return float(equity_from_trades(bars, tr, C0, cm)[1]["swap_usd"].iloc[0])
    check(swap("2024-01-08 10:00", "2024-01-08 21:59:59") == 0.0, "paid swap before 17:00 New York")
    check(near(swap("2024-01-08 10:00", "2024-01-08 22:00"), -20.00), "no swap when closed at 17:00 New York")
    check(swap("2024-01-08 22:00", "2024-01-09 10:00") == 0.0, "paid swap when opened at the rollover")
    check(near(swap("2024-01-08 10:00", "2024-01-12 10:00", side=-1), -120.00), "Mon-Fri is not 6 nights")
    # the US DST switch moves the rollover: 2024-03-08 22:00 UTC (EST), 2024-03-11 21:00 UTC (EDT)
    r = cal.rollover_instants(ts("2024-03-08 00:00"), ts("2024-03-12 00:00"))
    check([cal.utc_str(int(x))[:16] for x in r] == ["2024-03-08 22:00", "2024-03-11 21:00"], f"rollovers {r}")
    return "once per held rollover, none before 17:00 New York, Wednesday x3; 22:00 -> 21:00 UTC across US DST"


def gate_leverage_resizing() -> str:
    """Gate B5: leverage mode re-sizes only on changes.

    Returns a one-line detail; raises GateFailure on a wrong answer."""
    bars = B.synthetic_bars(ts("2024-04-01 00:00"), 300, seed=11)
    cm = CostModel(commission_per_lot_round_trip=7.0, markup_per_side=0.02)
    p = np.zeros(len(bars))
    p[10:60], p[60:90], p[90:140], p[200:300] = 1.0, 0.4, -0.8, 0.6
    eq, tr = equity_from_positions(bars, pd.DataFrame({"time": bars["time"], "position": p}), C0, cm,
                                   size_mode="leverage", size=2.0)
    times = bars["time"].to_numpy()
    fills = set(tr["entry_time"]) | set(tr.loc[tr["exit_reason"] == "signal", "exit_time"])
    check(fills == {int(times[k]) for k in (10, 60, 90, 140, 200)}, "fills away from position changes")
    check(eq["units_open"].iloc[10:60].nunique() == 1 and eq["equity_close"].iloc[10:60].nunique() > 10,
          "units changed while the position was unchanged")
    return "leverage units change only where the position changes"


def gate_alphamaster_alignment() -> str:
    """Gate B6: AlphaMaster one-bar shift and log PnL.

    Returns a one-line detail; raises GateFailure on a wrong answer."""
    times = ts("2024-01-01 00:00") + 3600 * np.arange(5)
    pos = adapters.positions_from_alphamaster(times, [0.5, -1.0, 0.0, 1.0, 0.25])
    check(pos["position"].tolist() == [0.0, 0.5, -1.0, 0.0, 1.0], "held[t+1] = p[t] shift is wrong")
    p, o, r = [1.0, 0.0, -1.0, -1.0, 0.0], [100.0, 101.0, 102.0, 104.0, 103.0], 0.001
    want = [math.log(102 / 101) - r, -r, -math.log(103 / 104) - r, 0.0, -r]
    check(np.allclose(adapters.alphamaster_log_pnl(p, o, r), want, atol=1e-15), "alphamaster_log_pnl")
    return "held[t+1] = p[t]; miner log PnL p[t] x log(open[t+2]/open[t+1]) - |dp| x cost"


# ---------------------------------------------------------------------------------------
# [C] statistics

def gate_dsr_example() -> str:
    """Gate C1: DSR example SR0 0.1132, DSR 0.9004. Returns a one-line detail; raises GateFailure on a wrong answer."""
    # SR 2.5/yr over 250 days -> 0.158113883 per day; V = 0.5/250; N = 100; skew -3; kurtosis 10; T = 1250
    sr, v = 2.5 / math.sqrt(250), 0.5 / 250
    sr0 = stats.expected_max_sr(100, v)
    value = stats.dsr(sr, 1250, -3.0, 10.0, 100, v)
    check(round(sr0, 4) == 0.1132 and round(value, 4) == 0.9004, f"SR0 {sr0:.6f}, DSR {value:.6f}")
    return f"SR0 = {sr0:.4f}, DSR = {value:.4f} (Bailey and Lopez de Prado 2014)"


def gate_psr_dsr_checks() -> str:
    """Gate C2: PSR / DSR / MinTRL checks. Returns a one-line detail; raises GateFailure on a wrong answer."""
    check(near(stats.psr(0.1, 0.1, 250, 0.0, 3.0), 0.5, 1e-12), "PSR(sr, sr) != 0.5")
    vals = [stats.dsr(0.1, 1000, -0.5, 4.0, n, 0.001) for n in (1, 10, 100, 1000)]
    check(all(a > b for a, b in zip(vals, vals[1:])), f"DSR does not fall with more trials: {vals}")
    trl = stats.min_track_record_length(0.1, 0.0, -0.5, 4.0, 0.95)
    check(near(stats.psr(0.1, 0.0, trl, -0.5, 4.0), 0.95, 1e-9), "MinTRL inconsistent with PSR")
    return "PSR(sr, sr) = 0.5; DSR falls with more trials; PSR at MinTRL = 0.95"


def gate_sr_standard_error() -> str:
    """Gate C3: sharpe_stats SR standard error on IID normal returns.

    Returns a one-line detail; raises GateFailure on a wrong answer."""
    # propkit.stats.sharpe_stats on 2,000 simulated IID normal series (n = 250, true SR 0.1 per period):
    # its SR must be mean / sample sd, and its se_sr must match the spread of the SR estimates across the
    # series (within 10%) and Lo's (2002) normal formula sqrt((1 + SR^2 / 2) / (n - 1)) = 0.06353 (within 3%)
    rng = np.random.default_rng(5)
    n, true_sr = 250, 0.1
    x = rng.normal(true_sr, 1.0, (2000, n))
    res = [stats.sharpe_stats(row, None) for row in x]
    est = np.array([r.sr for r in res])
    se = np.array([r.se_sr for r in res])
    check(np.allclose(est, x.mean(axis=1) / x.std(axis=1, ddof=1), rtol=1e-12), "sr is not mean / sample sd")
    ratio = float(est.std(ddof=1) / se.mean())
    lo = math.sqrt((1 + true_sr ** 2 / 2) / (n - 1))
    check(abs(ratio - 1) < 0.10, f"sd of the SR estimates / mean se_sr = {ratio:.3f}")
    check(abs(se.mean() / lo - 1) < 0.03, f"mean se_sr {se.mean():.5f} vs Lo's {lo:.5f}")
    return f"sd of SR estimates / sharpe_stats se_sr = {ratio:.3f}; mean se_sr {se.mean():.5f} vs Lo {lo:.5f}"


def gate_stats_known_answers() -> str:
    """Gate C4: sharpe_stats / ES / daily returns / drawdown known answers.

    Returns a one-line detail; raises GateFailure on a wrong answer."""
    # sharpe_stats([1, 2, 3, 4, 10]): mean 4, sample sd sqrt(50 / 4) = 3.535534, SR = 1.131371;
    # m2 = 10, m3 = 36, m4 = 278.8 -> skew 36 / 10^1.5 = 1.138420, kurt 2.788;
    # SE = sqrt((1 - 1.138420 x 1.131371 + 1.788 / 4 x 1.28) / 4) = sqrt(0.284185 / 4) = 0.266545
    sh = stats.sharpe_stats([1.0, 2.0, 3.0, 4.0, 10.0], None)
    check(near(sh.sr, 1.131371, 1e-6) and near(sh.skew, 1.138420, 1e-6) and near(sh.kurt, 2.788, 1e-9),
          f"sr {sh.sr} skew {sh.skew} kurt {sh.kurt}")
    check(near(sh.se_sr, math.sqrt(0.284185 / 4), 1e-6) and sh.sr_annual is None, f"se_sr {sh.se_sr}")
    # expected shortfall at 15% of 10 returns: n alpha = 1.5 -> (worst + 0.5 x second worst) / 1.5
    es = stats.expected_shortfall([-5.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.15)
    check(near(es, -6.5 / 1.5, 1e-12), f"ES {es}")
    # three prop days closing at 101,000 / 99,990 / 100,989.90: returns +1%, -1%, +1%
    rows = day_rows("2024-01-09", [(C0, 100_500.0, 100_400.0, 1.0), (101_000.0, 101_000.0, 100_900.0, 0.0)])
    rows += day_rows("2024-01-10", [(101_000.0, 100_000.0, 99_800.0, 1.0), (99_990.0, 99_990.0, 99_950.0, 0.0)])
    rows += day_rows("2024-01-11", [(99_990.0, 100_500.0, 99_900.0, 1.0), (100_989.9, 100_989.9, 100_989.9, 0.0)])
    eq = frame(rows)
    d = stats.daily_returns_from_equity(eq, C0)
    check(d["date"].tolist() == ["2024-01-09", "2024-01-10", "2024-01-11"], f"days {d['date'].tolist()}")
    check(np.allclose(d["ret"], [0.01, -0.01, 0.01], atol=1e-12), f"returns {d['ret'].tolist()}")
    # drawdowns: intrabar = peak 101,000 - worst 99,800 = 1,200 (1.2% of C0); close to close = 101,000 -
    # 99,990 = 1,010; days 2 and 3 close below the 101,000 peak: 2 days underwater at the end; ES with
    # n alpha < 1 = the worst day, -1%
    dd = stats.drawdown_stats(eq, C0)
    check(near(dd["max_dd_usd"], 1200.0) and near(dd["max_dd_pct"], 1.2) and near(dd["max_dd_close_usd"], 1010.0),
          f"max dd {dd['max_dd_usd']} / close {dd['max_dd_close_usd']}")
    check(dd["longest_underwater_days"] == 2 and dd["underwater_at_end"] and near(dd["es_ret"], -0.01, 1e-12)
          and dd["worst_day_date"] == "2024-01-10", f"underwater {dd['longest_underwater_days']} es {dd['es_ret']}")
    return ("SR 1.131371 +- 0.266545, skew 1.138420, kurt 2.788; ES(15%) -4.3333; daily returns +1/-1/+1%; "
            "max DD 1,200 intrabar / 1,010 close, 2 days underwater")


# ---------------------------------------------------------------------------------------
# [D] pullback

MONDAY = ts("2024-01-08 00:00")
PB_COSTS = CostModel(markup_per_side=0.10, slippage_per_side=0.05, commission_per_lot_round_trip=7.0,
                     swap_enabled=False)
PB_SPEC = PullbackSpec(name="hand", placeholder=False, direction="both", trend_ema_period=10, trend_slope_bars=3,
                       trend_require_close_side=True, pullback_mode="atr_from_swing", swing_lookback_bars=20,
                       min_depth_atr=1.0, max_depth_atr=3.0, trigger="break_prev_extreme", stop_mode="swing",
                       stop_buffer_atr=0.25, atr_period=14, exit_mode="fixed_r", target_r=2.0, risk_pct=0.01,
                       warmup_bars=20)


def pullback_hand_bars(after) -> pd.DataFrame:
    """Bars 0..29 rise 1 per bar (true range 2), 30..32 pull back, bar 33 breaks bar 32's high, bar 34 is the
    entry bar (open 128.5); `after` = bars 35.. as (open, high, low, close); spread 0.30 everywhere."""
    rows = [(100.0 + i, 101.0 + i + 0.5, 100.0 + i - 0.5, 101.0 + i) for i in range(30)]
    rows += [(130.0 - i, 130.0 - i, 128.0 - i, 129.0 - i) for i in range(3)]
    rows += [(127.0, 129.0, 127.0, 128.5), (128.5, 130.0, 128.0, 129.8)] + list(after)
    o, h, lo, c = (np.array(x, dtype=float) for x in zip(*rows))
    return pd.DataFrame({"time": MONDAY + 3600 * np.arange(len(rows)), "open": o, "high": h, "low": lo,
                         "close": c, "spread": np.full(len(rows), 0.30)})


def mirror(bars: pd.DataFrame, level: float, spread: float) -> pd.DataFrame:
    """Reflect bars so the mirror's ask is the reflection of the bid: bid' = 2 level - spread - bid."""
    k = 2.0 * level - spread
    return pd.DataFrame({"time": bars["time"].to_numpy(), "open": k - bars["open"].to_numpy(),
                         "high": k - bars["low"].to_numpy(), "low": k - bars["high"].to_numpy(),
                         "close": k - bars["close"].to_numpy(), "spread": np.full(len(bars), spread)})


def gate_pullback_hand_path() -> str:
    """Gate D1: pullback hand path to the cent. Returns a one-line detail; raises GateFailure on a wrong answer."""
    up = [(129.8, 131.8, 129.8, 131.5), (131.5, 133.5, 131.5, 133.2), (133.2, 135.2, 133.2, 135.0),
          (135.0, 137.0, 135.0, 136.5), (136.5, 138.5, 136.5, 138.0), (138.0, 140.0, 138.0, 139.5)]
    t = generate_trades(pullback_hand_bars(up), PB_SPEC, C0, PB_COSTS).iloc[0]
    # ATR = 2; swing high 130.5, pullback low 126 (depth 2.25 ATR); entry = ask 128.80 + 0.15 = 128.95;
    # stop = 126 - 0.25 x 2 = 125.50 (fill 125.35); loss/oz = 3.60 + 0.07 commission = 3.67;
    # units = floor(1,000 / 3.67) = 272; 1R = 272 x 3.60 + 19.04 = 998.24; target 128.95 + 2 x 3.45 = 135.85,
    # hit in bar 38 (high 137), filled 135.70; pnl = 272 x 6.75 - 19.04 = 1,816.96
    check(t["units"] == 272.0 and near(t["entry_price"], 128.95) and near(t["stop_price"], 125.50), "entry")
    check(t["exit_reason"] == "target" and near(t["exit_price"], 135.70), f"exit {t['exit_reason']}")
    check(near(t["risk_usd"], 998.24) and near(t["pnl_usd"], 1816.96), f"pnl {t['pnl_usd']}")
    return "entry 128.95, stop 125.50, 272 oz, 1R 998.24, target exit 135.70, pnl +1,816.96"


def gate_pullback_gap_and_stop_first() -> str:
    """Gate D2: pullback gap through stop, stop first.

    Returns a one-line detail; raises GateFailure on a wrong answer."""
    gap = generate_trades(pullback_hand_bars([(125.0, 125.4, 124.5, 125.2), (125.2, 125.5, 124.8, 125.0)]),
                          PB_SPEC, C0, PB_COSTS).iloc[0]
    check(gap["exit_reason"] == "stop" and gap["exit_time"] == MONDAY + 35 * 3600 and near(gap["exit_price"], 124.85),
          "gap through the stop does not fill at the open")
    both = pullback_hand_bars([(129.8, 136.0, 125.0, 130.0), (130.0, 130.5, 129.5, 130.2)])
    t = generate_trades(both, PB_SPEC, C0, PB_COSTS).iloc[0]
    check(t["exit_reason"] == "stop" and near(t["exit_price"], 125.35), "target taken before the stop")
    s = generate_trades(mirror(both, 200.0, 0.30), PB_SPEC, C0, PB_COSTS).iloc[0]
    check(s["side"] == -1 and s["exit_reason"] == "stop" and near(s["exit_price"], 400.0 - 125.35), "short stop first")
    return "gap fills at the open (124.85); stop first when stop and target share a bar (long and short)"


def _synth(n: int, seed: int) -> pd.DataFrame:
    return B.synthetic_bars(MONDAY, n, bar_seconds=3600, seed=seed, vol_per_hour=0.003, spread=0.34)


SYN_SPEC = PullbackSpec(name="selftest", placeholder=False, trend_ema_period=30, trend_slope_bars=3,
                        ema_pullback_period=10, pullback_lookback_bars=4, swing_lookback_bars=12, atr_period=10,
                        min_depth_atr=0.5, max_depth_atr=None, risk_pct=0.005, pullback_mode="ema_touch",
                        trigger="close_back_over_ema", stop_mode="swing", exit_mode="fixed_r", target_r=1.5)
SYN_COSTS = CostModel(markup_per_side=0.05, slippage_per_side=0.02, commission_per_lot_round_trip=7.0)


def gate_pullback_truncation() -> str:
    """Gate D3: pullback no look-ahead (truncation). Returns a one-line detail; raises GateFailure on a wrong answer."""
    bars = _synth(900, seed=21)
    full, _ = generate_trades_detailed(bars, SYN_SPEC, C0, SYN_COSTS)
    sig = compute_signals(bars, SYN_SPEC)
    times = bars["time"].to_numpy()
    cols = ["trade_id", "side", "units", "entry_time", "entry_price", "stop_price", "risk_usd"]
    checked = 0
    for k in (200, 333, 457, 600, 731, 899):
        part = generate_trades(bars.iloc[:k], SYN_SPEC, C0, SYN_COSTS)
        pd.testing.assert_frame_equal(compute_signals(bars.iloc[:k], SYN_SPEC), sig.iloc[:k], check_exact=True)
        entered = full[full["entry_time"] < times[k]].reset_index(drop=True)
        check(len(part) == len(entered), f"cut {k}: {len(part)} vs {len(entered)} entries")
        pd.testing.assert_frame_equal(part[cols], entered[cols], check_exact=True)
        checked += len(part)
    check(checked > 20, "too few trades to check")
    return f"cutting the data never changes earlier signals or entries ({checked} entries compared)"


def gate_pullback_mirror() -> str:
    """Gate D4: pullback long/short mirror. Returns a one-line detail; raises GateFailure on a wrong answer."""
    bars = _synth(1500, seed=12)
    bars["spread"] = 0.30
    cm = CostModel(markup_per_side=0.05, slippage_per_side=0.02, commission_per_lot_round_trip=7.0, swap_enabled=False)
    a = generate_trades(bars, SYN_SPEC, C0, cm)
    b = generate_trades(mirror(bars, 2000.0, 0.30), SYN_SPEC, C0, cm)
    check(len(a) == len(b) > 5 and (b["side"].to_numpy() == -a["side"].to_numpy()).all(), "sides not mirrored")
    check(np.allclose(b["entry_price"], 4000.0 - a["entry_price"], atol=1e-8)
          and np.allclose(b["pnl_usd"], a["pnl_usd"], atol=1e-6), "prices or pnl not mirrored")
    return f"reflected prices give mirrored trades ({len(a)} trades)"


# ---------------------------------------------------------------------------------------
# integration

def gate_end_to_end() -> str:
    """Gate I1: end to end on synthetic bars. Returns a one-line detail; raises GateFailure on a wrong answer."""
    from propkit import report as rep
    bars = B.synthetic_bars(ts("2024-03-04 00:00"), 1500, seed=8, spread=0.34)   # weekends, March DST changes
    cm = CostModel(commission_per_lot_round_trip=7.0)
    trades = generate_trades(bars, SYN_SPEC, C0, cm)
    eq, tr = equity_from_trades(bars, trades, C0, cm)
    check((eq["equity_worst"] <= eq["equity_close"] + 1e-9).all(), "equity_worst above equity_close")
    check(near(eq["balance"].iloc[-1], C0 + tr["pnl_usd"].sum(), 1e-6), "balance != C0 + sum of trade pnl")
    rng = np.random.default_rng(3)
    p = np.repeat(rng.choice([-1.0, 0.0, 1.0], len(bars) // 10 + 1), 10)[:len(bars)]
    pos = adapters.positions_from_alphamaster(bars["time"].to_numpy(), p)
    eq2, tr2 = equity_from_positions(bars, pos, C0, cm, size_mode="units", size=20.0)
    check(near(eq2["balance"].iloc[-1], C0 + tr2["pnl_usd"].sum(), 1e-6), "positions: balance mismatch")
    report = rep.analyse(bars, tr, eq, R.ftmo_1step(C0), cm, n_sims=300, seed=7, dd_sims=100, history_reps=3)
    text = rep.render_markdown(json.loads(json.dumps(report)))
    check(text.startswith(rep.HEADER) and text.isascii(), "report header or ASCII")
    path = evaluate_path(eq, tr, R.ftmo_1step(C0))
    replay = bs.bootstrap_challenges(eq, R.ftmo_1step(C0), trades=tr, mode="replay", horizon_days=None)
    want = {"passed": bs.PASSED, "breached_daily": bs.BREACHED_DAILY, "breached_max": bs.BREACHED_MAX,
            "running": bs.RUNNING}[path.status]
    check(int(replay.status[0]) == want, "bootstrap replay disagrees with evaluate_path")
    return f"bars -> {len(tr)} trades / {len(tr2)} position trades -> equity -> rules -> bootstrap -> report"


def imported_modules(text: str) -> set[str]:
    """Top-level names of every module a Python source imports, from its syntax tree: import x.y,
    from x import y (relative imports excluded), and importlib.import_module("x") / __import__("x") with a
    literal name - also inside functions, try blocks and one-line statements."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name in ("import_module", "__import__"):
                found.add(node.args[0].value.split(".")[0])
    return found


def gate_source_scan() -> str:
    """Gate I2: source scan (no AlphaMaster imports, ASCII).

    Returns a one-line detail; raises GateFailure on a wrong answer."""
    root = Path(__file__).resolve().parent
    bad = []
    for f in sorted(root.glob("*.py")):
        text = f.read_text(encoding="utf-8")
        if not text.isascii():
            bad.append(f"{f.name}: not ASCII")
        hit = FORBIDDEN_IMPORTS & imported_modules(text)
        if hit:
            bad.append(f"{f.name}: imports {sorted(hit)}")
    probe = "import os\ntry:\n    import model_core.backtest as m\nexcept ImportError: pass\nx = 1; from scipy import stats"
    check(imported_modules(probe) == {"os", "model_core", "scipy"}, "the import scan misses nested imports")
    check(not bad, "; ".join(bad))
    return f"{len(list(root.glob('*.py')))} files ASCII only, nothing imported from AlphaMaster, scipy, zoneinfo"


# ---------------------------------------------------------------------------------------
# runner

GATES: list[tuple[str, Callable[[], str]]] = [
    ("A1 daily floor at B_00:00 100,000 (1-Step / 2-Step)", gate_daily_floor_b00_100k),
    ("A2 daily floor at B_00:00 108,000 = 105,000", gate_daily_floor_b00_108k),
    ("A3 trailing max floor 102,000, never down", gate_trailing_max_floor),
    ("A4 2-Step static max floor 90,000", gate_static_max_floor),
    ("A5 prop day reset 22:00 / 23:00 UTC", gate_day_reset_summer_winter),
    ("A6 EU change days 25 h / 23 h", gate_eu_change_days),
    ("A7 US-only DST shift weeks", gate_us_only_shift_weeks),
    ("A8 best-day rule 60% fails, 50% passes", gate_best_day_rule),
    ("A9 hand-counted 10-day path", gate_ten_day_path),
    ("A10 bootstrap analytic answers", gate_bootstrap_analytic),
    ("A11 bootstrap replay = evaluate_path", gate_bootstrap_replay),
    ("A12 clustered losses: day blocks > 2 x trade shuffle", gate_clustered_losses),
    ("A13 flat-to-flat blocks: overnight holds stay together", gate_flat_to_flat_blocks),
    ("B1 3-trade hand example to the cent", gate_three_trade_hand_example),
    ("B2 flat rate 0.0003 per fill", gate_flat_rate),
    ("B3 equity_worst and marks", gate_equity_marks),
    ("B4 swap timing and US DST rollover", gate_swap_timing),
    ("B5 leverage mode re-sizes only on changes", gate_leverage_resizing),
    ("B6 AlphaMaster one-bar shift and log PnL", gate_alphamaster_alignment),
    ("C1 DSR example SR0 0.1132, DSR 0.9004", gate_dsr_example),
    ("C2 PSR / DSR / MinTRL checks", gate_psr_dsr_checks),
    ("C3 sharpe_stats SR standard error on IID normal returns", gate_sr_standard_error),
    ("C4 sharpe_stats / ES / daily returns / drawdown known answers", gate_stats_known_answers),
    ("D1 pullback hand path to the cent", gate_pullback_hand_path),
    ("D2 pullback gap through stop, stop first", gate_pullback_gap_and_stop_first),
    ("D3 pullback no look-ahead (truncation)", gate_pullback_truncation),
    ("D4 pullback long/short mirror", gate_pullback_mirror),
    ("I1 end to end on synthetic bars", gate_end_to_end),
    ("I2 source scan (no AlphaMaster imports, ASCII)", gate_source_scan),
]


def run_selftest(out=print) -> tuple[int, int]:
    """Run every gate; `out` receives one ASCII line per gate ('PASS name: detail' or 'FAIL name: reason')
    and a final summary line. Returns (number passed, number failed)."""
    n_pass = n_fail = 0
    t_all = time.perf_counter()
    for name, fn in GATES:
        t0 = time.perf_counter()
        try:
            detail = fn()
            ok, text = True, detail
        except Exception as e:              # a gate must report, never crash the run
            ok, text = False, f"{type(e).__name__}: {e}"
        dt = time.perf_counter() - t0
        line = f"{'PASS' if ok else 'FAIL'}  {name}: {text} ({dt:.2f} s)"
        out(line.encode("ascii", "backslashreplace").decode("ascii"))
        n_pass += ok
        n_fail += not ok
    total = time.perf_counter() - t_all
    out(f"{n_pass} of {len(GATES)} gates passed, {n_fail} failed, in {total:.1f} s. "
        + ("The installation reproduces every known answer." if n_fail == 0 else
           "Do NOT use results from this installation until every gate passes."))
    return n_pass, n_fail


__all__ = ["GATES", "run_selftest", "GateFailure"]
