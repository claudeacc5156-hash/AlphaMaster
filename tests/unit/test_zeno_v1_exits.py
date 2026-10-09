"""zeno_v1 exits (rule 6, D14-D17, D15's M1 run): gap fills, stop first, breakeven after the partial in the
same bar, +2R and +4R in one bar, the 16:30 New York time exit across the DST switches, an early close,
the short side on the ask, the t + 899 booking and the M1 resolution. Canonical long of
tests/unit/zeno_v1_testkit.py, cell evaluation / 10 / S1 / x1: entry 2006.20, stop 2001.50, R 4.70,
106 oz (53 + 53), tp1 2015.60, tp2 2025.00, breakeven 2006.30. Research only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propkit import zeno_v1 as z
from tests.unit.zeno_v1_testkit import Scenario, triggers, utc

START = "2024-03-05 12:00"            # trigger closes 14:00 UTC; entry bar opens 14:00
QUIET = (2006.0, 2007.0, 2005.0, 2006.5)


def run_long(after, start=START, tail=4, **kw):
    sc = Scenario(start)
    ti = sc.setup(1)
    for bar in after:
        sc.add(**bar) if isinstance(bar, dict) else sc.add(*bar)
    if tail:
        sc.flat(tail, sc.bid[-1][3])
    prep, res = sc.run(**kw)
    return sc, ti, prep, res


def legs_of(res):
    return list(zip(res.legs["leg"], res.legs["exit_reason"], res.legs["units"],
                    np.round(res.legs["exit_price"], 6)))


def test_gap_through_the_stop_fills_at_the_open_with_slippage():
    # bar 2 opens at bid 2000.00 < stop 2001.50: filled at 2000.00 - 0.05 = 1999.95 at the OPEN (D14)
    # gross = 106 x (1999.95 - 2006.20) = -662.50, net = -673.10
    sc, ti, prep, res = run_long([QUIET, (2000.0, 2001.0, 1999.0, 2000.5)])
    assert legs_of(res) == [("full", "stop", 106.0, 1999.95)]
    p = res.positions.iloc[0]
    assert p["exit1_time"] == sc.time(ti + 2) and p["final_exit_stamp"] == sc.time(ti + 2)
    assert p["net_pnl_usd"] == pytest.approx(-673.10) and p["outcome"] == "-1R"


def test_gap_through_a_target_fills_at_the_level_never_better():
    # bar 2 opens at 2017 (> tp1 2015.60): half at 2015.60 at the open, stop to 2006.30.
    # bar 3 opens at 2026 (> tp2 2025.00): the runner at 2025.00 at the open -> +3R.
    sc, ti, prep, res = run_long([QUIET, (2017.0, 2018.0, 2016.0, 2017.5), (2026.0, 2027.0, 2025.5, 2026.5)])
    assert legs_of(res) == [("tp1", "target", 53.0, 2015.6), ("runner", "target", 53.0, 2025.0)]
    assert res.legs["exit_time"].tolist() == [sc.time(ti + 2), sc.time(ti + 3)]
    p = res.positions.iloc[0]
    assert p["outcome"] == "+3R" and p["tp1_reached"] and p["tp2_reached"]
    # 53 x 9.40 - 5.30 + 53 x 18.80 - 5.30 = 492.90 + 991.10
    assert p["net_pnl_usd"] == pytest.approx(1484.00)


def test_gap_through_both_targets_at_once():
    sc, ti, prep, res = run_long([QUIET, (2030.0, 2031.0, 2029.0, 2030.0)])
    assert legs_of(res) == [("tp1", "target", 53.0, 2015.6), ("runner", "target", 53.0, 2025.0)]
    assert res.positions.iloc[0]["outcome"] == "+3R"


def test_stop_first_when_one_bar_touches_the_stop_and_a_target():
    sc, ti, prep, res = run_long([QUIET, (2006.5, 2016.0, 2001.0, 2010.0)])
    assert legs_of(res) == [("full", "stop", 106.0, 2001.45)]
    p = res.positions.iloc[0]
    assert p["outcome"] == "-1R" and p["ambiguous_bar"] and not p["tp1_reached"]
    assert res.meta["m1"] == z.M1_NOT_RUN == "M1 resolution not run"
    assert res.meta["n_ambiguous_positions"] == 1


def test_breakeven_is_checked_after_the_partial_in_the_same_bar():
    # low 2005 is above the stop 2001.50 but below breakeven 2006.30; high 2016 reaches +2R:
    # half at 2015.60, then the breakeven stop in the same bar at 2006.25 (D15)
    sc, ti, prep, res = run_long([QUIET, (2006.5, 2016.0, 2005.0, 2010.0)])
    assert legs_of(res) == [("tp1", "target", 53.0, 2015.6), ("runner", "stop", 53.0, 2006.25)]
    p = res.positions.iloc[0]
    assert p["outcome"] == "+1R(BE)" and p["ambiguous_bar"]
    assert res.legs["exit_time"].tolist() == [sc.time(ti + 2) + 899] * 2


def test_tp1_and_tp2_in_one_bar():
    # low 2006.50 stays above breakeven 2006.30; high 2026 passes 2015.60 and 2025.00
    sc, ti, prep, res = run_long([QUIET, (2006.5, 2026.0, 2006.5, 2025.5)])
    assert legs_of(res) == [("tp1", "target", 53.0, 2015.6), ("runner", "target", 53.0, 2025.0)]
    p = res.positions.iloc[0]
    assert p["outcome"] == "+3R" and not p["ambiguous_bar"]
    assert p["net_pnl_usd"] == pytest.approx(1484.00)


def test_an_exact_touch_of_the_breakeven_level_closes_the_runner():
    # [SI-34] tie rule. Entry bar: bid open 2006.00, ask open 2006.30 (spread 0.30 <= 10% of R): entry 2006.30,
    # R = 2006.30 - 2001.50 = 4.80, 500 / 4.80 = 104.2 -> 104 oz (52 + 52), tp1 = 2015.90, breakeven =
    # 2006.30 + 10 / 100 = 2006.40, which is 2006.3999999999999 in binary floating point. Bar 2 fills +2R (high
    # 2016.50, low 2006.50 above breakeven); bar 3's bid low is EXACTLY 2006.40, a touch (D14, D16), so the
    # runner exits at 2006.40 - 0.05 slippage = 2006.35.
    # net = 52 x 9.60 - 5.20 + 52 x 0.05 - 5.20 = 491.40
    entry_bar = dict(o=2006.0, h=2007.0, lo=2005.0, c=2006.5, ask=(2006.30, 2007.30, 2005.30, 2006.80))
    sc, ti, prep, res = run_long([entry_bar, (2006.5, 2016.5, 2006.5, 2016.0), (2010.0, 2010.0, 2006.40, 2007.0)])
    assert legs_of(res) == [("tp1", "target", 52.0, 2015.9), ("runner", "stop", 52.0, 2006.35)]
    p = res.positions.iloc[0]
    assert p["be_level"] == pytest.approx(2006.40, abs=1e-9)
    assert p["outcome"] == "+1R(BE)" and p["net_pnl_usd"] == pytest.approx(491.40)
    assert res.legs["exit_time"].iloc[1] == sc.time(ti + 3) + 899


def test_an_exact_touch_of_the_tp1_level_fills_the_partial():
    # [SI-34] tie rule. tp1 = 2006.20 + 2 x 4.70 = 2015.60, which is 2015.6000000000001 in floating point; bar 2's
    # bid high is exactly 2015.60: the +2R partial fills at 2015.60 (D14: a target triggers when the closing-side
    # price touches it) and the stop moves to breakeven; the runner ends with the data at 2010.00.
    sc, ti, prep, res = run_long([QUIET, (2006.5, 2015.6, 2006.5, 2010.0)])
    assert legs_of(res) == [("tp1", "target", 53.0, 2015.6), ("runner", "end_of_data", 53.0, 2010.0)]
    assert bool(res.positions.iloc[0]["tp1_reached"])


def test_an_exact_touch_of_a_short_target_on_the_ask():
    # [SI-34] tie rule on the short side. Short entry 1994.00 (bid open), stop 1998.00 + 0.50 + 0.20 = 1998.70
    # (an ASK level), R = 4.70 (4.7000000000000455 in floating point), tp1 = 1984.60, tp2 = 1994 - 18.80 =
    # 1975.20, which is 1975.1999999999998 in floating point; breakeven 1993.90. Bar 2: ask low 1984.20 fills
    # +2R, ask high 1993.70 stays under breakeven. Bar 3's ASK low is exactly 1975.20: +4R fills there (D14).
    sc = Scenario(START)
    ti = sc.setup(-1)
    sc.add(1994.0, 1995.0, 1993.0, 1994.5)
    sc.add(1993.5, 1993.5, 1984.0, 1985.0)
    sc.add(1985.0, 1985.0, 1975.0, 1976.0, ask=(1985.20, 1985.20, 1975.20, 1976.20))
    sc.flat(4, 1976.0)
    prep, res = sc.run(trend="short")
    assert legs_of(res) == [("tp1", "target", 53.0, 1984.6), ("runner", "target", 53.0, 1975.2)]
    assert res.positions.iloc[0]["outcome"] == "+3R"


def test_runner_stop_and_target_in_one_bar_is_stop_first():
    sc, ti, prep, res = run_long([QUIET, (2006.5, 2016.0, 2006.5, 2015.0), (2015.0, 2026.0, 2006.0, 2010.0)])
    assert legs_of(res) == [("tp1", "target", 53.0, 2015.6), ("runner", "stop", 53.0, 2006.25)]
    assert res.positions.iloc[0]["ambiguous_bar"]


# ---------------------------------------------------------------------------------------------------
# D17: 16:30 New York

@pytest.mark.parametrize("day,exit_utc", [
    ("2024-03-08", "21:30"),      # Friday, EST
    ("2024-03-11", "20:30"),      # Monday after the switch to EDT
    ("2024-11-01", "20:30"),      # Friday, EDT
    ("2024-11-04", "21:30"),      # Monday after the switch back to EST
])
def test_time_exit_at_the_open_of_the_16_30_new_york_bar(day, exit_utc):
    sc = Scenario(f"{day} 12:00")
    ti = sc.setup(1)
    sc.flat_until(f"{day} 23:00", 2006.0)
    prep, res = sc.run()
    p = res.positions.iloc[0]
    assert p["entry_time"] == utc(f"{day} 14:00")
    assert (p["exit1_reason"], p["exit1_time"], p["final_exit_stamp"]) == ("time", utc(f"{day} {exit_utc}"),
                                                                            utc(f"{day} {exit_utc}"))
    # at the bid open 2006.00: 106 x (2006.00 - 2006.20) - 10.60 = -31.80
    assert p["exit1_price"] == 2006.0 and p["net_pnl_usd"] == pytest.approx(-31.80)
    assert p["outcome"] == "time-exit"


def test_early_close_exits_at_the_close_of_the_last_bar_before_the_break():
    # Friday 2024-11-29 (EST): bars stop after the one opening 13:15 New York = 18:15 UTC and resume Sunday
    # 18:00 New York = 23:00 UTC. No bar opens at 16:30 New York = 21:30 UTC, so the position is closed at
    # the close of the 18:15 UTC bar (18:30 UTC) at its bid close (D17).
    sc = Scenario("2024-11-29 12:00")
    ti = sc.setup(1)
    last = sc.flat_until("2024-11-29 18:15", 2006.0)
    sc.add(2006.0, 2006.5, 2005.5, 2006.4)
    sc.skip_to("2024-12-01 23:00")
    sc.flat(8, 2006.0)
    prep, res = sc.run()
    p = res.positions.iloc[0]
    assert (p["exit1_reason"], p["exit1_time"], p["exit1_price"]) == ("time", utc("2024-11-29 18:30"), 2006.4)
    assert p["exit1_time_utc"] == "2024-11-29 18:30:00 UTC"
    assert last + 1 == int(np.flatnonzero(prep.time == utc("2024-11-29 18:15"))[0])
    z.cell_equity(prep, res)                               # propkit books it in the 18:15 bar


def test_short_time_exit_at_the_ask_open():
    # short entry 1994.00 (bid open); flat bid 1994 with ask 1994.20: time exit at the ASK open 1994.20,
    # gross = -106 x 0.20 = -21.20, net -31.80
    sc = Scenario("2024-03-05 12:00")
    sc.setup(-1)
    sc.flat_until("2024-03-05 23:00", 1994.0)
    prep, res = sc.run(trend="short")
    p = res.positions.iloc[0]
    assert (p["side"], p["exit1_reason"], p["exit1_time"]) == ("short", "time", utc("2024-03-05 21:30"))
    assert p["exit1_price"] == pytest.approx(1994.20) and p["net_pnl_usd"] == pytest.approx(-31.80)


def test_end_of_data_closes_at_the_last_close():
    sc, ti, prep, res = run_long([QUIET], tail=3)
    p = res.positions.iloc[0]
    assert p["exit1_reason"] == "end_of_data" and p["exit1_time"] == sc.time(ti + 4) + 900
    assert p["exit1_price"] == 2006.5


def test_no_position_may_cross_the_rollover():
    legs = pd.DataFrame({"position_id": [1], "entry_time": [utc("2024-03-05 20:00")],
                         "exit_time": [utc("2024-03-05 21:59")]})
    z._assert_no_rollover(legs)                            # 17:00 New York = 22:00 UTC in March (EST)
    legs["exit_time"] = [utc("2024-03-05 22:00")]
    with pytest.raises(RuntimeError, match="rollover"):
        z._assert_no_rollover(legs)


# ---------------------------------------------------------------------------------------------------
# bookkeeping with propkit.equity

def test_intrabar_exits_are_booked_in_their_own_bar_and_equity_agrees():
    sc, ti, prep, res = run_long([QUIET, (2006.5, 2016.0, 2006.5, 2015.0), (2015.0, 2015.0, 2006.0, 2007.0)])
    equity, trades = z.cell_equity(prep, res)
    assert np.allclose(trades["pnl_usd"], res.legs["pnl_usd"])
    assert equity["balance"].iloc[-1] == pytest.approx(res.meta["final_balance_usd"])
    for xt in res.legs["exit_time"]:
        assert xt % 900 == 899                             # inside the bar, before the next one opens
    # propkit books half the commission per fill: entry bar ti+1: -5.30 (106 oz); bar ti+2 (tp1, 53 oz):
    # +498.20 - 2.65; bar ti+3 (breakeven, 53 oz): +2.65 - 2.65. Each exit lands in its own bar.
    t = equity["time"].to_numpy()
    bal = equity["balance"].to_numpy()
    k = int(np.flatnonzero(t == sc.time(ti + 1))[0])
    assert bal[k - 1] == 100_000.0
    assert bal[k:k + 3].tolist() == pytest.approx([99_994.70, 100_490.25, 100_490.25])
    assert equity["realised_usd"].to_numpy()[k + 1:k + 3].tolist() == pytest.approx([498.20, 2.65])


# ---------------------------------------------------------------------------------------------------
# D15 second run: M1 resolution

def _m1_frames(start: int, rows):
    t = start + 60 * np.arange(len(rows), dtype=np.int64)
    bid = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    bid.insert(0, "time", t)
    ask = bid.copy()
    ask[["open", "high", "low", "close"]] += 0.2
    return bid, ask


def test_m1_resolution_changes_a_stop_first_bar():
    # M15 bar 2 (14:15-14:30) touches 2001 and 2016: stop first -> -1R. Its M1 bars say +2R came first
    # (minute 0 high 2016) and the price then fell through breakeven (minute 1 low 2001): +1R(BE).
    after = [QUIET, (2006.5, 2016.0, 2001.0, 2010.0)]
    sc, ti, prep, res = run_long(after)
    t_amb = sc.time(ti + 2)
    rows = [(2006.5, 2016.0, 2006.5, 2015.5), (2015.5, 2015.5, 2001.0, 2002.0)] + \
        [(2002.0, 2010.0, 2002.0, 2010.0)] * 13
    m1_bid, m1_ask = _m1_frames(t_amb, rows)
    cfg = z.ZenoConfig(z.ZenoCell("evaluation", 10.0, "S1", 1.0))
    res_m1, diff = z.resolve_with_m1(prep, cfg, m1_bid, m1_ask, base=res)
    assert res.positions.iloc[0]["outcome"] == "-1R"
    p = res_m1.positions.iloc[0]
    assert p["outcome"] == "+1R(BE)" and p["m1_bars_resolved"] == 1
    assert res_m1.legs["exit_price"].tolist() == pytest.approx([2015.60, 2006.25])
    assert res_m1.meta["m1"] == {"run": True, "bars_resolved": 1, "bars_unresolved": 0, "bars_unresolved_no_m1": 0,
                                 "bars_unresolved_m1_mismatch": 0}
    assert len(diff) == 1
    d = diff.iloc[0]
    assert (d["status"], d["outcome_m15"], d["outcome_m1"]) == ("changed", "-1R", "+1R(BE)")
    assert d["diff_usd"] == pytest.approx(490.25 - (-514.10))


def test_m1_resolution_without_m1_bars_keeps_the_m15_answer():
    sc, ti, prep, res = run_long([QUIET, (2006.5, 2016.0, 2001.0, 2010.0)])
    m1_bid, m1_ask = _m1_frames(sc.time(ti + 6), [(2006.0, 2006.5, 2005.5, 2006.0)] * 5)   # elsewhere
    res_m1, diff = z.resolve_with_m1(prep, z.ZenoConfig(z.ZenoCell("evaluation", 10.0, "S1", 1.0)), m1_bid, m1_ask)
    assert res_m1.meta["m1"] == {"run": True, "bars_resolved": 0, "bars_unresolved": 1, "bars_unresolved_no_m1": 1,
                                 "bars_unresolved_m1_mismatch": 0}
    assert diff["status"].tolist() == ["same"]
    assert res_m1.positions.iloc[0]["m1_bars_unresolved"] == 1


def test_m1_gap_inside_the_bar_uses_the_gap_rules():
    # minute 1 opens below the stop (bid 2001.00): filled at that M1 open minus slippage = 2000.95
    sc, ti, prep, res = run_long([QUIET, (2006.5, 2016.0, 2001.0, 2010.0)])
    rows = [(2006.5, 2007.0, 2006.0, 2006.5), (2001.0, 2016.0, 2001.0, 2010.0)] + [(2010.0, 2010.0, 2010.0, 2010.0)] * 13
    m1_bid, m1_ask = _m1_frames(sc.time(ti + 2), rows)
    res_m1, _ = z.resolve_with_m1(prep, z.ZenoConfig(z.ZenoCell("evaluation", 10.0, "S1", 1.0)), m1_bid, m1_ask)
    assert res_m1.legs["exit_price"].tolist() == pytest.approx([2000.95])
    assert res_m1.legs["exit_time"].tolist() == [sc.time(ti + 2) + 899]


@pytest.mark.parametrize("rows, old_outcome", [
    # (a) the M1 bars stay inside 2005 .. 2008: they touch neither the stop 2001.50 nor tp1 2015.60 (the old code
    #     counted the bar as resolved and held the position to the end of the data)
    ([(2006.5, 2008.0, 2005.0, 2007.0)] * 15, "other"),
    # (b) they reach the M15 high 2016 but not its low 2001: tp1 and then the breakeven 2006.30 in minute 0
    #     (the old code turned the M15 bar's -1R into +1R(BE))
    ([(2006.5, 2016.0, 2005.0, 2010.0)] + [(2010.0, 2010.0, 2005.0, 2006.0)] * 14, "+1R(BE)"),
])
def test_m1_bars_that_miss_the_m15_high_or_low_leave_the_bar_unresolved(rows, old_outcome):
    # RR-3 [SI-68]: the ambiguous M15 bar 2 (14:30) has bid low 2001.0 <= stop 2001.50 and bid high 2016.0 >=
    # tp1 2015.60. M1 bars whose lowest bid low and highest bid high do not reach that low and high cannot say
    # which level came first (minutes missing, or another feed): the bar keeps the M15 answer (stop first,
    # -1R = -514.10 USD) and is counted as unresolved because the M1 bars disagree.
    sc, ti, prep, res = run_long([QUIET, (2006.5, 2016.0, 2001.0, 2010.0)])
    m1_bid, m1_ask = _m1_frames(sc.time(ti + 2), rows)
    res_m1, diff = z.resolve_with_m1(prep, z.ZenoConfig(z.ZenoCell("evaluation", 10.0, "S1", 1.0)), m1_bid, m1_ask,
                                     base=res)
    p = res_m1.positions.iloc[0]
    assert old_outcome != "-1R"
    assert (p["outcome"], p["m1_bars_resolved"], p["m1_bars_unresolved"]) == ("-1R", 0, 1)
    assert p["net_pnl_usd"] == pytest.approx(-514.10)
    assert res_m1.meta["m1"] == {"run": True, "bars_resolved": 0, "bars_unresolved": 1, "bars_unresolved_no_m1": 0,
                                 "bars_unresolved_m1_mismatch": 1}
    assert diff["status"].tolist() == ["same"]


def test_m1_check_on_the_short_side_uses_the_ask():
    # RR-3 [SI-68], short: the mirror bar (ask low 1984.20 <= tp1 1984.60, ask high 1999.20 >= stop 1998.70) is
    # ambiguous; M1 ask bars that reach both resolve it (tp1 first, then the breakeven 1993.90 -> +1R(BE)); the
    # same minutes without the ask high's spike leave it unresolved (-1R).
    sc = Scenario(START)
    ti = sc.setup(-1)
    sc.add(1994.0, 1995.0, 1993.0, 1993.5)                         # entry bar, ask = bid + 0.20
    sc.add(1993.5, 1999.0, 1984.0, 1990.0)                         # ambiguous on the ask: 1984.20 .. 1999.20
    sc.flat(4, 1990.0)
    prep, res = sc.run(trend="short")
    assert res.positions.iloc[0]["outcome"] == "-1R"
    cfg = z.ZenoConfig(z.ZenoCell("evaluation", 10.0, "S1", 1.0))
    reach = [(1993.5, 1993.5, 1984.0, 1985.0), (1985.0, 1999.0, 1985.0, 1990.0)] + [(1990.0,) * 4] * 13
    miss = [(1993.5, 1993.5, 1984.0, 1985.0), (1985.0, 1995.0, 1985.0, 1990.0)] + [(1990.0,) * 4] * 13
    for rows, outcome, resolved in ((reach, "+1R(BE)", 1), (miss, "-1R", 0)):
        res_m1, _ = z.resolve_with_m1(prep, cfg, *_m1_frames(sc.time(ti + 2), rows), base=res)
        p = res_m1.positions.iloc[0]
        assert (p["outcome"], p["m1_bars_resolved"]) == (outcome, resolved)


def test_the_last_bar_before_a_break_is_labelled_us_holiday_or_data_gap_and_counted():
    # CAUS-3 [SI-64]: D17 closes at the last bar before the break "on a US holiday with an early close". The
    # code does that wherever no bar opens at 16:30 New York (kept: holding through the hole would cross the
    # 17:00 New York rollover); each such exit is now labelled from a declared US holiday / early-close list
    # and counted, because on an ordinary day nobody knows at that close that no bar follows.
    sc = Scenario("2024-03-05 12:00")                      # an ordinary Tuesday, no bars 15:00-23:00 UTC
    sc.setup(1)
    sc.flat_until("2024-03-05 15:00", 2006.0)
    sc.skip_to("2024-03-05 23:00")
    sc.flat(8, 2006.0)
    prep, res = sc.run()
    p = res.positions.iloc[0]
    assert (p["exit1_reason"], p["exit1_time"]) == ("time", utc("2024-03-05 15:00"))
    assert p["time_exit_rule"] == "early_close_other_day"
    te = res.meta["time_exits"]
    assert (te["at_16_30_open"], te["early_close_us_holiday"], te["early_close_other_day"]) == (0, 0, 1)
    assert te["other_day_position_ids"] == [int(p["position_id"])]

    sc = Scenario("2024-11-29 12:00")                      # Black Friday: a declared US early close
    sc.setup(1)
    sc.flat_until("2024-11-29 18:15", 2006.0)
    sc.add(2006.0, 2006.5, 2005.5, 2006.4)
    sc.skip_to("2024-12-01 23:00")
    sc.flat(8, 2006.0)
    prep, res = sc.run()
    assert res.positions.iloc[0]["time_exit_rule"] == "early_close_us_holiday"
    assert res.meta["time_exits"]["early_close_us_holiday"] == 1

    sc = Scenario("2024-03-05 12:00")                      # the normal case: the 16:30 New York open
    sc.setup(1)
    sc.flat_until("2024-03-05 23:00", 2006.0)
    prep, res = sc.run()
    assert res.positions.iloc[0]["time_exit_rule"] == "16:30_open"
    assert res.meta["time_exits"]["at_16_30_open"] == 1


@pytest.mark.parametrize("day,expected", [
    ("2024-11-29", True), ("2024-11-28", True), ("2024-03-05", False), ("2024-07-03", True),
    ("2024-07-04", True), ("2024-12-24", True), ("2024-12-31", True), ("2025-01-01", True),
    ("2025-04-18", True), ("2025-04-17", False), ("2024-01-15", True), ("2024-02-19", True),
    ("2024-05-27", True), ("2024-09-02", True), ("2021-12-24", True), ("2020-07-03", True),
    ("2022-06-20", True), ("2021-06-18", False), ("2015-01-19", True), ("2024-10-14", False)])
def test_declared_us_holiday_and_early_close_days(day, expected):
    # [SI-64] New Year, MLK, Presidents, Good Friday, Memorial, Juneteenth (from 2022), Independence Day and
    # July 3, Labor, Thanksgiving and the day after, Christmas Eve and Day, New Year's Eve; a fixed holiday
    # on a weekend is observed on the Friday before / Monday after. Columbus Day (2024-10-14) is not one.
    assert z.us_early_close_day(z.calendar.firm_day(utc(f"{day} 12:00"), "utc_midnight")) is expected
