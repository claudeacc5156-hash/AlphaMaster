"""zeno_v1 Master variant (D23): 0.4% risk; an open position is closed at the open of the M15 bar that
contains T - 10 min of an NFP/CPI/PPI/FOMC release when it was opened less than 5 h before T - 10 min; the
entry blackout (D20) still applies. Release used: CPI 2024-03-12 08:30 New York = 12:30 UTC (a real
calendar row), so T - 10 min = 12:20 UTC, inside the bar opening 12:15; 5 h earlier = 07:20 UTC.
Research only."""
from __future__ import annotations

import pytest

from propkit import zeno_v1 as z
from tests.unit.zeno_v1_testkit import Scenario, news_from_rows, triggers, utc

MASTER = z.ZenoCell("master", 10.0, "S1", 1.0)
EVAL = z.ZenoCell("evaluation", 10.0, "S1", 1.0)


def run(trigger_close: str, cell, tmp_path, gap_1215=False, side=1, news=True):
    start = z.calendar.utc_str(utc(f"2024-03-12 {trigger_close}") - 8 * 900)[:16]
    sc = Scenario(start)
    ti = sc.setup(side)
    price = 2006.0 if side > 0 else 1994.0
    if gap_1215:
        sc.flat_until("2024-03-12 12:15", price)
        sc.skip_to("2024-03-12 12:30")
    sc.flat_until("2024-03-12 22:00", price)
    prep, res = sc.run(cell=cell, news=news_from_rows(tmp_path) if news else None,
                       trend="long" if side > 0 else "short")
    return sc, ti, prep, res


def test_master_closes_a_young_position_at_the_open_of_the_bar_holding_t_minus_10(tmp_path):
    # entry 07:30 UTC > 07:20: opened under 5 h before T - 10 -> closed at the 12:15 open (bid 2006.00)
    sc, ti, prep, res = run("07:30", MASTER, tmp_path)
    p = res.positions.iloc[0]
    assert p["entry_time"] == utc("2024-03-12 07:30") and p["units_oz"] == 85.0      # 400 / 4.70 -> 85
    assert (p["exit1_reason"], p["exit1_time"], p["exit1_price"]) == ("signal", utc("2024-03-12 12:15"), 2006.0)
    assert p["outcome"] == "other"
    # 85 x (2006.00 - 2006.20) - 8.50 = -25.50
    assert p["net_pnl_usd"] == pytest.approx(-25.50)


def test_master_keeps_a_position_opened_5h_or_more_before(tmp_path):
    # entry 07:15 UTC < 07:20: kept through the release; closed by the 16:30 New York exit (20:30 UTC, EDT)
    sc, ti, prep, res = run("07:15", MASTER, tmp_path)
    p = res.positions.iloc[0]
    assert (p["exit1_reason"], p["exit1_time"]) == ("time", utc("2024-03-12 20:30"))


def test_evaluation_variant_holds_through_news(tmp_path):
    sc, ti, prep, res = run("07:30", EVAL, tmp_path)
    p = res.positions.iloc[0]
    assert (p["exit1_reason"], p["exit1_time"]) == ("time", utc("2024-03-12 20:30"))
    assert p["units_oz"] == 106.0


def test_master_close_when_no_bar_holds_t_minus_10(tmp_path):
    # the 12:15 bar is missing: the close happens at the open of the next bar (12:30) [SI-40]
    sc, ti, prep, res = run("07:30", MASTER, tmp_path, gap_1215=True)
    p = res.positions.iloc[0]
    assert (p["exit1_reason"], p["exit1_time"]) == ("signal", utc("2024-03-12 12:30"))


def test_master_short_closes_at_the_ask_open(tmp_path):
    sc, ti, prep, res = run("07:30", MASTER, tmp_path, side=-1)
    p = res.positions.iloc[0]
    assert (p["side"], p["exit1_reason"], p["exit1_price"]) == ("short", "signal", pytest.approx(1994.20))


@pytest.mark.parametrize("cell", [MASTER, EVAL])
def test_master_keeps_the_entry_blackout(cell, tmp_path):
    # trigger closes 13:15 UTC: in the entry window, inside [12:00, 13:30] of the 12:30 CPI (both variants)
    sc, ti, prep, res = run("13:15", cell, tmp_path)
    tr = triggers(res)
    assert tr[tr["bar_index"] == ti].iloc[0]["status"] == "news_blackout"


def test_master_without_a_calendar_closes_nothing(tmp_path):
    sc, ti, prep, res = run("07:30", MASTER, tmp_path, news=False)
    assert res.positions.iloc[0]["exit1_reason"] == "time"
    assert res.meta["news"].startswith("no news calendar given")


def _gap_to_fill(fill: str, cell=MASTER, tmp_path=None):
    """NFP 2024-03-08 13:30 UTC (a real calendar row, EST): T - 10 min = 13:20 UTC. The canonical long's trigger
    bar opens 12:30 and closes 12:45 UTC (in session; 12:45 < T - 30 = 13:00, so no news block); a data gap
    follows and the next bar, the entry bar, opens at `fill`; then flat 2006.00 to 21:45 UTC."""
    sc = Scenario("2024-03-08 10:45")
    ti = sc.setup(1)
    sc.skip_to(f"2024-03-08 {fill}")
    sc.flat_until("2024-03-08 21:45", 2006.0)
    sc.flat(4, 2006.0)
    prep, res = sc.run(cell=cell, news=news_from_rows(tmp_path))
    return sc, ti, prep, res


def test_master_closes_at_the_entry_open_when_the_fill_bar_holds_t_minus_10(tmp_path):
    # RR-4 / RR1-D23-1 [SI-70]: after a gap the fill bar (13:15-13:30) contains T - 10 min = 13:20. At 13:20 the
    # trade is open and 5 minutes old, so D23 closes it "at the open of the bar containing T - 10 min": the
    # entry bar's own open. Long 85 oz (400 / 4.70 = 85.1 -> 85) bought at the ask open 2006.20 and sold at the
    # bid open 2006.00: gross 85 x -0.20 = -17.00, commission 10 x 0.85 = 8.50, net -25.50 (a D21 loss).
    sc, ti, prep, res = _gap_to_fill("13:15", tmp_path=tmp_path)
    assert triggers(res).iloc[-1]["status"] == "entered"
    p = res.positions.iloc[0]
    assert (p["entry_time"], p["entry_price"], p["units_oz"]) == (utc("2024-03-08 13:15"), pytest.approx(2006.20), 85.0)
    assert (p["exit1_reason"], p["exit1_time"], p["exit1_price"]) == ("signal", utc("2024-03-08 13:15"), 2006.0)
    assert p["final_exit_stamp"] == utc("2024-03-08 13:15") and p["outcome"] == "other"
    assert p["net_pnl_usd"] == pytest.approx(-25.50)
    eq, _ = z.cell_equity(prep, res)                                  # propkit.equity books it in that bar
    assert eq["balance"].iloc[-1] == pytest.approx(100_000.0 - 25.50)
    # the evaluation variant holds it through the release to the 16:30 New York exit (21:30 UTC, EST)
    sc, ti, prep, res = _gap_to_fill("13:15", EVAL, tmp_path)
    assert (res.positions.iloc[0]["exit1_reason"], res.positions.iloc[0]["exit1_time"]) == ("time", utc("2024-03-08 21:30"))


def test_master_keeps_a_fill_after_t_minus_10_when_no_bar_holds_it(tmp_path):
    # [SI-40] kept: bars 12:45-13:15 are missing, so no bar holds 13:20 and the Master close belongs to the first
    # bar after it, 13:30, which is the entry bar. The fill at 13:30 comes after T - 10 min: the trade was not
    # open at T - 10 min, so nothing closes it; it ends at the 16:30 New York exit (21:30 UTC).
    sc, ti, prep, res = _gap_to_fill("13:30", tmp_path=tmp_path)
    p = res.positions.iloc[0]
    assert p["entry_time"] == utc("2024-03-08 13:30")
    assert (p["exit1_reason"], p["exit1_time"]) == ("time", utc("2024-03-08 21:30"))


# ---------------------------------------------------------------------------------------------------
# unscheduled rows (D20, D23) [SI-63]: the rule is kept as written; what it uses before T is counted

def _unscheduled(kind="unscheduled", trigger_close="14:45", cell=EVAL):
    """2020-03-03, the real unscheduled FOMC cut at 10:00 New York = 15:00 UTC (EST): one canonical long whose
    trigger closes at trigger_close UTC, then flat to 22:00 UTC."""
    news = z.news_calendar([utc("2020-03-03 15:00")], ["FOMC"], [kind])
    sc = Scenario(z.calendar.utc_str(utc(f"2020-03-03 {trigger_close}") - 8 * 900)[:16])
    ti = sc.setup(1)
    sc.flat_until("2020-03-03 22:00", 2006.0)
    prep, res = sc.run(cell=cell, news=news)
    return ti, prep, res


def test_a_trigger_blocked_only_before_an_unscheduled_row_is_flagged_and_counted():
    # D20 includes unscheduled FOMC statements, so the trigger closing 14:45 UTC (inside [14:30, 16:00]) is
    # blocked as the spec says; nobody knew at 14:45 that a statement would come at 15:00, so the row is
    # flagged news_pre_unscheduled and counted in meta (the spec wins, the look-ahead is made visible).
    ti, prep, res = _unscheduled()
    row = triggers(res).iloc[-1]
    assert row["bar_index"] == ti and row["status"] == "news_blackout" and bool(row["news_pre_unscheduled"])
    assert res.meta["news_unscheduled"]["n_triggers_blocked_only_before_unscheduled"] == 1
    # the same instant as a scheduled row: no flag; after T (15:15 UTC) the block is causal: no flag either
    for kind, close in (("scheduled", "14:45"), ("unscheduled", "15:15")):
        ti, prep, res = _unscheduled(kind, close)
        row = triggers(res).iloc[-1]
        assert row["status"] == "news_blackout" and not bool(row["news_pre_unscheduled"])
        assert res.meta["news_unscheduled"]["n_triggers_blocked_only_before_unscheduled"] == 0
    # stage 1 carries the flag too (it is not an outcome)
    from propkit import zeno_report as zr
    ti, prep, res = _unscheduled()
    _, t = zr.signals_stage(prep, EVAL)
    d = t["decisions"]
    assert d.loc[d["event"] == "trigger", "news_pre_unscheduled"].tolist() == [True]
    assert t["signals"]["status"].tolist() == ["news_blackout"]


def test_a_master_close_for_an_unscheduled_row_is_counted():
    # entry 13:15 UTC (trigger close 13:15, in session, outside [14:30, 16:00]); Master closes it at the open of
    # the bar holding T - 10 min = 14:50, the 14:45 bar, as D23 says; that close is listed as caused only by an
    # unscheduled row. Under the evaluation variant it is held; with a scheduled row nothing is listed.
    ti, prep, res = _unscheduled(trigger_close="13:15", cell=MASTER)
    p = res.positions.iloc[0]
    assert (p["exit1_reason"], p["exit1_time"]) == ("signal", utc("2020-03-03 14:45"))
    assert res.meta["news_unscheduled"]["master_closes_only_for_unscheduled"] == [int(p["position_id"])]
    ti, prep, res = _unscheduled("scheduled", "13:15", MASTER)
    assert res.positions.iloc[0]["exit1_reason"] == "signal"
    assert res.meta["news_unscheduled"]["master_closes_only_for_unscheduled"] == []
    ti, prep, res = _unscheduled(trigger_close="13:15", cell=EVAL)
    assert res.positions.iloc[0]["exit1_reason"] == "time"
    assert res.meta["news_unscheduled"]["n_rows"] == 1
