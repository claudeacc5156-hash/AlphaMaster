"""Addendum A1 (zeno_pullback_v1): the variant "master_fp" with FundingPips' restricted events.

Rows below are copied verbatim from propkit/data/news_calendar/us_restricted_events_fundingpips_2015-01-01_2025-
09-27.csv (no other restricted event lies within 10 h of the FEDCHAIR and CLAIMS rows used; the 2020-11-12 rows
are that day's full set). Times: 2019-06-04 09:55 EDT = 13:55 UTC (a speech, 60 min), 2023-08-25 10:05 EDT =
14:05 UTC (a speech), 2024-07-09 10:00 EDT = 14:00 UTC (a testimony, 180 min), CLAIMS 2024-02-15 08:30 EST =
13:30 UTC (a release). The canonical long (zeno_v1_testkit) enters at the ask open 2006.20, R 4.70, 85 oz at
0.4% (400 / 4.70 = 85.1 -> 85). Session (D19): trigger closes in [07:00, 10:00) and [12:30, 16:00) UTC.
Research only; synthetic bars.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from propkit import zeno_v1 as z
from tests.unit.zeno_v1_testkit import Scenario, news_from_rows, triggers, utc

MASTER = z.ZenoCell("master", 10.0, "S1", 1.0)
MASTER_FP = z.ZenoCell("master_fp", 10.0, "S1", 1.0)
EVAL = z.ZenoCell("evaluation", 10.0, "S1", 1.0)

RESTRICTED_CSV = (
    "event,date_et,time_et,utc_offset_ny,datetime_utc,kind,basis,source_list,note\n"
    "FEDCHAIR,2019-06-04,09:55,-0400,2019-06-04T13:55Z,scheduled,both,federalreserve.gov speeches/testimony/calendar"
    " + dated wires,\"Speech (Powell): Fed Listens strategy conference, Chicago. Fed calendar: 9:55 a.m.\"\n"
    "CLAIMS,2020-11-12,08:30,-0500,2020-11-12T13:30Z,scheduled,both,DOL oui.doleta.gov press archive + ALFRED "
    "rid=180,\n"
    "CPI,2020-11-12,08:30,-0500,2020-11-12T13:30Z,scheduled,both,https://www.bls.gov/bls/news-release/cpi.htm,Oct 2020"
    " CPI (cpi_11122020.htm)\n"
    "AUCT30,2020-11-12,13:00,-0500,2020-11-12T18:00Z,scheduled,both,Treasury FiscalData auctions_query,\"original "
    "issue, CUSIP 912810SS8\"\n"
    "FEDCHAIR,2020-11-12,unknown,-0500,,scheduled,critic-verified,federalreserve.gov speeches/testimony/calendar + "
    "dated wires,\"ECB Forum policy panel with Lagarde and Bailey; start about 11:45 ET, not pinned (critic)\"\n"
    "FEDCHAIR,2023-08-25,10:05,-0400,2023-08-25T14:05Z,scheduled,both,federalreserve.gov speeches/testimony/calendar"
    " + dated wires,Speech (Powell): Jackson Hole. Fed calendar: 10:05 a.m.\n"
    "CLAIMS,2024-02-15,08:30,-0500,2024-02-15T13:30Z,scheduled,both,DOL oui.doleta.gov press archive + ALFRED "
    "rid=180,\n"
    "FEDCHAIR,2024-07-09,10:00,-0400,2024-07-09T14:00Z,scheduled,both,federalreserve.gov speeches/testimony/calendar"
    " + dated wires,\"Testimony (Powell): Semiannual MPR, Senate Banking. Fed calendar: 10:00 a.m.\"\n")


def secs(text: str) -> int:
    return int(dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp())


@pytest.fixture(scope="module")
def rows(tmp_path_factory) -> z.RestrictedCalendar:
    p = tmp_path_factory.mktemp("restricted") / "restricted_rows.csv"
    p.write_text(RESTRICTED_CSV, encoding="ascii")
    return z.read_restricted_csv(p)


@pytest.fixture(scope="module")
def packaged() -> z.RestrictedCalendar:
    return z.read_restricted_csv()


def run_at(day: str, trigger_close: str, cell, restricted, tmp_path, fill: str | None = None,
           until: str = "22:00", news=True, **kw):
    """The canonical long whose trigger bar closes at `trigger_close` UTC on `day`; the entry bar opens at that
    close, or after a data gap at `fill`; then flat bid 2006.00 to `until` UTC."""
    sc = Scenario(z.calendar.utc_str(utc(f"{day} {trigger_close}") - 8 * 900)[:16])
    ti = sc.setup(1)
    if fill is not None:
        sc.skip_to(f"{day} {fill}")
    sc.flat_until(f"{day} {until}", 2006.0)
    prep, res = sc.run(cell=cell, news=news_from_rows(tmp_path) if news else None, restricted=restricted, **kw)
    return ti, prep, res


def status(res, ti) -> tuple[str, list[str]]:
    row = triggers(res).set_index("bar_index").loc[ti]
    return row["status"], [r for r in str(row["reasons"]).split(";") if r]


# ---------------------------------------------------------------------------------------
# the reader

def test_packaged_restricted_calendar(packaged):
    s = packaged.summary()
    assert packaged.sha256 == z.RESTRICTED_SHA256 and z.RESTRICTED_CSV.is_file()
    assert s["n_events_known_time"] == 2824 and s["n_unknown_time"] == 1           # 2,825 rows
    assert s["unknown_time_rows"] == [{"event": "FEDCHAIR", "date_et": "2020-11-12"}]
    assert s["fedchair_testimony"] == 70 and s["fedchair_other"] == 145             # 216 FEDCHAIR rows, 1 unknown
    assert s["n_unscheduled"] == 13 and s["per_event"]["CLAIMS"] == 560 and s["per_event"]["NFP"] == 129
    dur = packaged.end - packaged.start
    fed = np.array([n == "FEDCHAIR" for n in packaged.names])
    tes = np.array(packaged.testimony, dtype=bool)
    assert (dur[~fed] == 0).all() and (dur[fed & tes] == 180 * 60).all() and (dur[fed & ~tes] == 60 * 60).all()
    assert (np.diff(packaged.start) >= 0).all()
    assert packaged.unknown_days.tolist() == [(dt.date(2020, 11, 12) - dt.date(1970, 1, 1)).days]


def test_reader_rows_and_errors(rows, tmp_path):
    s = rows.summary()
    assert s["n_events_known_time"] == 7 and s["fedchair_testimony"] == 1 and s["fedchair_other"] == 2
    bad = tmp_path / "bad.csv"
    bad.write_text("event,date_et,time_et\nCLAIMS,2024-02-15,08:30\n", encoding="ascii")
    with pytest.raises(ValueError, match="datetime_utc"):
        z.read_restricted_csv(bad)
    with pytest.raises(ValueError):
        z.read_restricted_csv(tmp_path / "locked_holdout" / "x.csv")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        z.restricted_calendar([], unknown=[("FEDCHAIR", "12/11/2020")])


# ---------------------------------------------------------------------------------------
# A1 entry block: window edges to the second, on real rows (the packaged file agrees)

@pytest.mark.parametrize("instant,blocked", [
    # FEDCHAIR testimony 2024-07-09 14:00 UTC: [T - 5 min, T + 180 min + 5 min] = [13:55:00, 17:05:00]
    ("2024-07-09 13:54:59", False), ("2024-07-09 13:55:00", True), ("2024-07-09 15:15:00", True),
    ("2024-07-09 17:05:00", True), ("2024-07-09 17:05:01", False),
    # FEDCHAIR speech 2019-06-04 13:55 UTC: [13:50:00, 13:55 + 65 min = 15:00:00]
    ("2019-06-04 13:49:59", False), ("2019-06-04 13:50:00", True), ("2019-06-04 15:00:00", True),
    ("2019-06-04 15:00:01", False),
    # CLAIMS release 2024-02-15 13:30 UTC: [13:25:00, 13:35:00]
    ("2024-02-15 13:24:59", False), ("2024-02-15 13:25:00", True), ("2024-02-15 13:35:00", True),
    ("2024-02-15 13:35:01", False),
])
def test_window_edges_on_real_rows(rows, packaged, instant, blocked):
    t = secs(instant)
    assert rows.in_window(t) is blocked and rows.blocked(t) is blocked
    assert packaged.blocked(t) is blocked


@pytest.mark.parametrize("instant,blocked", [
    # the unknown-time Fed Chair row of 2020-11-12 blocks the whole New York date (EST: 05:00 UTC to 05:00 UTC)
    ("2020-11-12 04:59:59", False), ("2020-11-12 05:00:00", True), ("2020-11-12 07:30:00", True),
    ("2020-11-13 04:59:59", True), ("2020-11-13 05:00:00", False),
])
def test_unknown_time_blocks_the_new_york_date(rows, packaged, instant, blocked):
    t = secs(instant)
    assert rows.on_unknown_day(t) is blocked and packaged.on_unknown_day(t) is blocked
    assert rows.in_window(t) is False


def test_engine_blocks_at_the_edges_with_real_rows(rows, tmp_path):
    # non-testimony 2019-06-04 13:55: a trigger close at T_end + 5 min = 15:00 is blocked, 15:15 is entered
    ti, prep, res = run_at("2019-06-04", "15:00", MASTER_FP, rows, tmp_path)
    assert status(res, ti) == ("fp_restricted_window", ["fp_restricted_window"])
    ti, prep, res = run_at("2019-06-04", "15:15", MASTER_FP, rows, tmp_path)
    assert status(res, ti)[0] == "entered"
    # 2023-08-25 14:05: a trigger close at T - 5 min = 14:00 is blocked (the left edge on the bar grid)
    ti, prep, res = run_at("2023-08-25", "14:00", MASTER_FP, rows, tmp_path)
    assert status(res, ti)[0] == "fp_restricted_window"
    # testimony 2024-07-09 14:00: 15:15 is 75 min after T, inside the 180-min appearance ...
    ti, prep, res = run_at("2024-07-09", "15:15", MASTER_FP, rows, tmp_path)
    assert status(res, ti)[0] == "fp_restricted_window"
    # ... while the same appearance at the 60-min length of a speech would have ended at 15:05
    speech = z.restricted_calendar([utc("2024-07-09 14:00")], ["FEDCHAIR"], testimony=[False])
    ti, prep, res = run_at("2024-07-09", "15:15", MASTER_FP, speech, tmp_path)
    assert status(res, ti)[0] == "entered"
    # the variant "master" knows no restricted list: 14:00 on 2023-08-25 is entered there
    ti, prep, res = run_at("2023-08-25", "14:00", MASTER, rows, tmp_path)
    assert status(res, ti)[0] == "entered"


def test_the_entry_fill_instant_is_checked_too(rows, tmp_path):
    # testimony 2024-07-09 14:00 UTC: the trigger closes at 13:45 (outside [13:55, 17:05]); after a data gap the
    # fill bar opens at 14:00, inside the window: blocked (D20 checks the trigger close only, A1 both instants)
    ti, prep, res = run_at("2024-07-09", "13:45", MASTER_FP, rows, tmp_path, fill="14:00")
    assert status(res, ti) == ("fp_restricted_window", ["fp_restricted_window"])
    assert bool(prep.fp_open[ti + 1]) and not bool(prep.fp_close[ti])


def test_unknown_time_date_blocks_entries_all_day(rows, tmp_path):
    # 2020-11-12 (Thursday): a trigger close at 07:30 UTC is far from the day's known windows (13:25-13:35 and
    # 17:55-18:05 UTC) and from D20's (none in the news rows): blocked only for the unknown-time Fed Chair row
    ti, prep, res = run_at("2020-11-12", "07:30", MASTER_FP, rows, tmp_path, until="12:00")
    assert status(res, ti) == ("fp_restricted_window", ["fp_restricted_window"])
    assert res.positions.empty
    ti, prep, res = run_at("2020-11-11", "07:30", MASTER_FP, rows, tmp_path, until="12:00")
    assert status(res, ti)[0] == "entered"
    ti, prep, res = run_at("2020-11-12", "07:30", MASTER, rows, tmp_path, until="12:00")
    assert status(res, ti)[0] == "entered"


# ---------------------------------------------------------------------------------------
# A1 close rule (D23's convention, c - 5 h < entry_time <= c, [SI-40], [SI-70])

def test_close_for_4h59_and_not_for_5h00():
    # synthetic restricted releases on Wed 2024-02-14 (no other event): entry 08:15 UTC in both cases.
    #   T = 13:24 -> c = 13:14, inside the 13:00 bar; 08:15 is 4 h 59 min before c -> closed at the 13:00 open.
    #   T = 13:25 -> c = 13:15, inside the 13:15 bar; 08:15 is exactly 5 h 00 min before c -> kept (time exit,
    #   16:30 EST = 21:30 UTC).
    for T, expect in (("13:24", ("signal", utc("2024-02-14 13:00"))), ("13:25", ("time", utc("2024-02-14 21:30")))):
        cal = z.restricted_calendar([utc(f"2024-02-14 {T}")], ["CLAIMS"])
        ti, prep, res = run_at("2024-02-14", "08:15", MASTER_FP, cal, None, news=False)
        p = res.positions.iloc[0]
        assert p["entry_time"] == utc("2024-02-14 08:15")
        assert (p["exit1_reason"], p["exit1_time"]) == expect, T


def test_close_age_on_a_real_row(rows, tmp_path):
    # FEDCHAIR 2019-06-04 13:55 UTC: c = 13:45; an entry at 08:45 is exactly 5 h old (kept), one at 09:00 is
    # 4 h 45 min old and is closed at the 13:45 open (bid 2006.00): 85 x -0.20 - 8.50 = -25.50
    ti, prep, res = run_at("2019-06-04", "08:45", MASTER_FP, rows, tmp_path)
    assert res.positions.iloc[0]["exit1_reason"] == "time"
    ti, prep, res = run_at("2019-06-04", "09:00", MASTER_FP, rows, tmp_path)
    p = res.positions.iloc[0]
    assert (p["exit1_reason"], p["exit1_time"], p["exit1_price"]) == ("signal", utc("2019-06-04 13:45"), 2006.0)
    assert p["units_oz"] == 85.0 and p["net_pnl_usd"] == pytest.approx(-25.50)
    assert res.meta["master_closes"] == {"n_positions_closed": 1, "n_closed_only_for_restricted_list": 1,
                                         "closed_only_for_restricted_list": [int(p["position_id"])]}


def test_close_on_the_fill_bar_after_a_gap(rows, tmp_path):
    # [SI-70] for a restricted event: FEDCHAIR 2019-06-04 13:55 UTC, c = 13:45. The trigger closes 12:45; bars
    # 12:45-13:30 are missing and the fill bar opens 13:45 (outside [13:50, 15:00]), the bar holding c: the
    # trade is open at c, so it is closed at its own entry open (bought at the ask 2006.20, sold at the bid
    # 2006.00: -25.50). The variant master (no restricted list) keeps it.
    ti, prep, res = run_at("2019-06-04", "12:45", MASTER_FP, rows, tmp_path, fill="13:45")
    p = res.positions.iloc[0]
    assert (p["entry_time"], p["exit1_reason"], p["exit1_time"]) == (utc("2019-06-04 13:45"), "signal",
                                                                    utc("2019-06-04 13:45"))
    assert p["net_pnl_usd"] == pytest.approx(-25.50)
    eq, _ = z.cell_equity(prep, res)
    assert eq["balance"].iloc[-1] == pytest.approx(100_000.0 - 25.50)
    ti, prep, res = run_at("2019-06-04", "12:45", MASTER, rows, tmp_path, fill="13:45")
    assert res.positions.iloc[0]["exit1_reason"] == "time"


def test_master_fp_against_master_on_a_claims_release(rows, tmp_path):
    # CLAIMS 2024-02-15 08:30 EST = 13:30 UTC, not one of rule 9's four events. Entry 09:00 UTC: c = 13:20,
    # 4 h 20 min later. master_fp closes it at the open of the 13:15 bar; master and evaluation hold it to the
    # 16:30 New York exit (21:30 UTC). A trigger at 13:30 is blocked in master_fp only.
    ti, prep, res_fp = run_at("2024-02-15", "09:00", MASTER_FP, rows, tmp_path)
    p = res_fp.positions.iloc[0]
    assert (p["exit1_reason"], p["exit1_time"]) == ("signal", utc("2024-02-15 13:15"))
    for cell in (MASTER, EVAL):
        ti, prep, res = run_at("2024-02-15", "09:00", cell, rows, tmp_path)
        assert (res.positions.iloc[0]["exit1_reason"], res.positions.iloc[0]["exit1_time"]) == (
            "time", utc("2024-02-15 21:30"))
    ti, prep, res_fp = run_at("2024-02-15", "13:30", MASTER_FP, rows, tmp_path)
    ti, prep, res_m = run_at("2024-02-15", "13:30", MASTER, rows, tmp_path)
    assert status(res_fp, ti)[0] == "fp_restricted_window" and status(res_m, ti)[0] == "entered"


def test_master_is_unchanged_by_the_restricted_calendar(rows, tmp_path):
    # the same bars with and without the restricted calendar in prepare: identical master and evaluation results
    for cell in (MASTER, EVAL):
        _, _, a = run_at("2024-02-15", "09:00", cell, rows, tmp_path)
        _, _, b = run_at("2024-02-15", "09:00", cell, None, tmp_path)
        pd.testing.assert_frame_equal(a.positions, b.positions)
        pd.testing.assert_frame_equal(a.decisions, b.decisions)
        pd.testing.assert_frame_equal(a.legs, b.legs)
        assert list(a.decisions.columns) == list(z.DECISION_COLUMNS)
    with pytest.raises(ValueError, match="master_fp needs"):
        run_at("2024-02-15", "09:00", MASTER_FP, None, tmp_path)


def test_unscheduled_flags_work_for_master_fp():
    # [SI-63] for A1: a synthetic UNSCHEDULED restricted release at 14:05 UTC (Wed 2024-02-14). A trigger close at
    # 14:00 lies in [T - 5 min, T): blocked as A1 says, flagged fp_pre_unscheduled and counted; as a scheduled
    # row it is blocked without a flag. A position entered at 09:00 is closed at the 13:45 open (c = 13:55) only
    # for that unscheduled row: listed in master_closes_only_for_unscheduled.
    for kind, flag in (("unscheduled", True), ("scheduled", False)):
        cal = z.restricted_calendar([utc("2024-02-14 14:05")], ["DURABLE"], [kind])
        ti, prep, res = run_at("2024-02-14", "14:00", MASTER_FP, cal, None, news=False)
        row = triggers(res).set_index("bar_index").loc[ti]
        assert row["status"] == "fp_restricted_window" and bool(row["fp_pre_unscheduled"]) is flag
        nu = res.meta["news_unscheduled"]
        assert nu["fp_n_triggers_flagged"] == int(flag)
        assert nu["n_triggers_blocked_only_before_unscheduled_any"] == int(flag)
        assert nu["n_triggers_blocked_only_before_unscheduled"] == 0          # the D20 count: no news here
        assert nu["fp_n_rows"] == int(flag)
        assert list(res.decisions.columns) == list(z.DECISION_COLUMNS_FP)
    cal = z.restricted_calendar([utc("2024-02-14 14:05")], ["DURABLE"], ["unscheduled"])
    ti, prep, res = run_at("2024-02-14", "09:00", MASTER_FP, cal, None, news=False)
    p = res.positions.iloc[0]
    assert (p["exit1_reason"], p["exit1_time"]) == ("signal", utc("2024-02-14 13:45"))
    assert res.meta["news_unscheduled"]["master_closes_only_for_unscheduled"] == [int(p["position_id"])]


def test_stage1_unscheduled_line_for_master_fp():
    # [SI-63] in the stage-1 report of --variant master_fp: an unscheduled restricted row (DURABLE, 14:05 UTC) and
    # no news. The 14:00 trigger is blocked only by the restricted window, in the 5 min before that row. The D20
    # count and its "no other reason" subset stay 0 (a subset never exceeds its set); the restricted window's
    # count and the count blocked only before unscheduled rows by either block are a clause of their own.
    from propkit import zeno_report as zr
    cal = z.restricted_calendar([utc("2024-02-14 14:05")], ["DURABLE"], ["unscheduled"])
    sc = Scenario(z.calendar.utc_str(utc("2024-02-14 14:00") - 8 * 900)[:16])
    ti = sc.setup(1)
    sc.flat_until("2024-02-14 22:00", 2006.0)
    prep = sc.prepare(news=z.news_calendar([]), restricted=cal)
    rep, t = zr.signals_stage(prep, MASTER_FP, 100_000.0, 20, 7)
    d = t["decisions"].set_index("bar_index")
    assert d.loc[ti, "status"] == "fp_restricted_window" and bool(d.loc[ti, "fp_pre_unscheduled"])
    nu = rep["news_unscheduled"]
    assert (nu["n_rows"], nu["n_triggers_flagged"], nu["n_triggers_blocked_only_before_unscheduled"]) == (0, 0, 0)
    assert (nu["fp_n_rows"], nu["fp_n_triggers_flagged"], nu["n_triggers_blocked_only_before_unscheduled_any"]) == (
        1, 1, 1)
    md = zr.render_signals_markdown(rep)
    line = next(x for x in md.splitlines() if "unscheduled rows:" in x)
    assert ("0 trigger(s) were blocked by news only in the 30 min before an unscheduled row (0 with no other "
            "reason; decisions column news_pre_unscheduled)") in line
    assert ("the restricted window (variant master_fp) blocked 1 trigger(s) only in the 5 min before one of the 1 "
            "unscheduled restricted rows (decisions column fp_pre_unscheduled), and 1 trigger(s) were blocked only "
            "before unscheduled rows (news and/or the restricted window) with no other reason") in line
    # the evaluation variant's line is unchanged: no restricted-window clause
    rep_e, _ = zr.signals_stage(prep, EVAL, 100_000.0, 20, 7)
    assert "fp_n_rows" not in rep_e["news_unscheduled"]
    assert "restricted window" not in next(x for x in zr.render_signals_markdown(rep_e).splitlines()
                                           if "unscheduled rows:" in x)


def test_d20_applies_in_master_fp(tmp_path):
    # rule 9's own blackout (CPI 2024-03-12 12:30 UTC, [12:00, 13:30]) blocks a 13:15 trigger close in every
    # variant, master_fp included, even with an empty restricted list
    empty = z.restricted_calendar([])
    for cell in (EVAL, MASTER, MASTER_FP):
        ti, prep, res = run_at("2024-03-12", "13:15", cell, empty, tmp_path)
        assert status(res, ti)[0] == "news_blackout"


def test_variants_risk_reasons_and_grid():
    assert z.VARIANTS == ("evaluation", "master", "master_fp") and z.MASTER_VARIANTS == ("master", "master_fp")
    assert z.RISK_PCT["master_fp"] == z.RISK_PCT["master"] == 0.004
    keys = list(z.BLOCK_REASONS)
    assert keys.index("fp_restricted_window") == keys.index("news_blackout") + 1
    assert keys.index("margin_cap_below_lot_step") == keys.index("size_below_lot_step") - 1
    cells = z.grid_cells()
    assert len(cells) == 36 == len({c.label for c in cells})
    assert [c.variant for c in cells[::12]] == ["evaluation", "master", "master_fp"]
