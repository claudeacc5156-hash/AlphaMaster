"""Addendum A (zeno_pullback_v1): the "utc_plus3" firm day (A4) and the verified FundingPips preset (A3).

Known answers are computed by hand in the comments. US DST 2024: summer time from Sun 2024-03-10 07:00 UTC to
Sun 2024-11-03 06:00 UTC, so 17:00 New York is 21:00 UTC in summer and 22:00 UTC in winter; 00:00 UTC+3 is
21:00 UTC all year. Research only; synthetic data only.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from propkit import bootstrap as B
from propkit import calendar as cal
from propkit import cli
from propkit import rules as R
from propkit import stats as S
from propkit import zeno_v1 as z
from propkit.evaluator import evaluate_path

H = 3600
UTC = dt.timezone.utc
C0 = 100_000.0
COLS = ["time", "balance", "equity_close", "equity_worst", "units_open"]


def ts(text: str) -> int:
    return int(dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp())


def dnum(text: str) -> int:
    return (dt.date.fromisoformat(text) - dt.date(1970, 1, 1)).days


def frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS)
    df["time"] = df["time"].astype(np.int64)
    for c in COLS[1:]:
        df[c] = df[c].astype(np.float64)
    return df


# ---------------------------------------------------------------------------------------
# A4: calendar.firm_day "utc_plus3"

@pytest.mark.parametrize("instant,plus3,ny17", [
    # US summer (EDT), Wed 2024-07-10: 00:00 UTC+3 and 17:00 New York are both 21:00 UTC
    ("2024-07-10 20:59:59", "2024-07-10", "2024-07-10"),
    ("2024-07-10 21:00:00", "2024-07-11", "2024-07-11"),
    # US winter (EST), Wed 2024-01-10: 00:00 UTC+3 is 21:00 UTC, 17:00 New York is 22:00 UTC
    ("2024-01-10 20:59:59", "2024-01-10", "2024-01-10"),
    ("2024-01-10 21:00:00", "2024-01-11", "2024-01-10"),     # 16:00 New York: still the old ny_17 day
    ("2024-01-10 22:00:00", "2024-01-11", "2024-01-11"),
])
def test_utc_plus3_known_answers_summer_and_winter(instant, plus3, ny17):
    t = ts(instant)
    assert cal.day_to_str(cal.firm_day(t, "utc_plus3")) == plus3
    assert cal.day_to_str(cal.firm_day(t, "ny_17")) == ny17
    assert cal.firm_day(t, "utc_plus3") == (t + 3 * H) // 86400          # the UTC+3 calendar date


@pytest.mark.parametrize("friday,sunday,fri_plus3,fri_ny17,monday", [
    # US winter: the Friday close is 17:00 EST = 22:00 UTC, so 21:30 UTC is the last trading hour. Under ny_17
    # it is still Friday's day; under utc_plus3 it is 00:30 UTC+3 Saturday: a short SATURDAY day (documented in
    # propkit.calendar). The Sunday 22:05 UTC instant (17:05 EST, before the 23:00 UTC reopen) is Monday's day
    # under both boundaries, as the Sunday reopen is.
    ("2024-01-12 21:30:00", "2024-01-14 22:05:00", "2024-01-13", "2024-01-12", "2024-01-15"),
    # US summer: the Friday close is 21:00 UTC, so 21:30 UTC is after the close; both boundaries call it
    # Saturday (no bar exists there). Sunday 22:05 UTC = 18:05 EDT, just after the reopen: Monday's day.
    ("2024-07-12 21:30:00", "2024-07-14 22:05:00", "2024-07-13", "2024-07-13", "2024-07-15"),
])
def test_utc_plus3_weekend_pair_against_ny_17(friday, sunday, fri_plus3, fri_ny17, monday):
    f, s = ts(friday), ts(sunday)
    assert cal.day_to_str(cal.firm_day(f, "utc_plus3")) == fri_plus3
    assert cal.day_to_str(cal.firm_day(f, "ny_17")) == fri_ny17
    assert cal.day_to_str(cal.firm_day(s, "utc_plus3")) == monday
    assert cal.day_to_str(cal.firm_day(s, "ny_17")) == monday
    # week blocks (bootstrap.week_id, Sunday..Saturday): the Saturday day stays in Friday's week, Monday's day
    # opens the next one
    fri_day = dnum(friday[:10])
    assert B.week_id(cal.firm_day(f, "utc_plus3")) == B.week_id(fri_day)
    assert B.week_id(cal.firm_day(s, "utc_plus3")) == B.week_id(fri_day) + 1


def test_utc_plus3_day_start_label_and_round_trip():
    assert cal.firm_day_start_utc(dnum("2024-07-11"), "utc_plus3") == ts("2024-07-10 21:00:00")
    assert cal.firm_day_start_utc(dnum("2024-01-11"), "utc_plus3") == ts("2024-01-10 21:00:00")
    days = np.arange(dnum("2015-01-01"), dnum("2026-12-31") + 1)
    start = np.asarray(cal.firm_day_start_utc(days, "utc_plus3"))
    assert (np.diff(start) == 86400).all()                                # always 24 h: no daylight saving
    assert (np.asarray(cal.firm_day(start, "utc_plus3")) == days).all()
    assert (np.asarray(cal.firm_day(start - 1, "utc_plus3")) == days - 1).all()
    assert cal.boundary_label("utc_plus3") == "00:00 UTC+3"
    assert cal.boundary_utc_text("utc_plus3") == "21:00 UTC all year, no daylight saving"
    assert R.PropRules(day_boundary="utc_plus3").day_boundary == "utc_plus3"
    lines = R.PropRules(day_boundary="utc_plus3", daily_loss_base="day_start").describe()
    assert any("firm day: starts at 00:00 UTC+3 (21:00 UTC all year, no daylight saving)" in x for x in lines)


def test_existing_boundaries_unchanged_by_the_new_one():
    # the three older boundaries give exactly the values they gave before (independent recomputation)
    t = z.synthetic_m15_bidask(start=ts("2024-01-01 00:00:00"), n_bars=40_000, seed=4)["time"].to_numpy()
    assert (np.asarray(cal.firm_day(t, "utc_midnight")) == t // 86400).all()
    assert (np.asarray(cal.firm_day(t, "cet_midnight")) == np.asarray(cal.prop_day(t))).all()
    ny = np.asarray(cal.firm_day(t, "ny_17"))
    assert (ny == (t + (np.asarray(cal.ny_offset_hours(t)) + 7) * H) // 86400).all()
    assert (np.asarray(cal.firm_day(t, "utc_plus3")) == (t + 3 * H) // 86400).all()


# ---------------------------------------------------------------------------------------
# A4: the evaluator, the bootstrap and the statistics under "utc_plus3"

def fp_rules(boundary: str) -> R.PropRules:
    return dataclasses.replace(R.rules_and_info("fundingpips-1step-flex")[0], day_boundary=boundary)


def test_evaluator_daily_breach_under_utc_plus3_but_not_ny_17():
    # US winter, Wed 2024-01-10, the verified FundingPips rules (2% of max(balance, equity) at the day start,
    # touching the floor breaches). A loss of 1,200 USD closes in the 21:00 UTC bar (balance 98,800); in the
    # 22:00 UTC bar a position dips to 97,500 and closes at 98,700.
    #   utc_plus3: the day began at 21:00 UTC at max(100,000, 100,000) -> floor 98,000 -> 97,500 breaches.
    #   ny_17:     the day began at 22:00 UTC (17:00 EST) at max(98,800, 98,800) -> floor 96,824 -> no breach.
    rows = [(ts("2024-01-10 19:00:00"), C0, C0, C0, 0.0),
            (ts("2024-01-10 20:00:00"), C0, C0, C0, 0.0),
            (ts("2024-01-10 21:00:00"), 98_800.0, 98_800.0, 98_800.0, 0.0),
            (ts("2024-01-10 22:00:00"), 98_800.0, 98_700.0, 97_500.0, 10.0),
            (ts("2024-01-10 23:00:00"), 98_700.0, 98_700.0, 98_700.0, 0.0)]
    eq = frame(rows)
    plus3, ny = evaluate_path(eq, None, fp_rules("utc_plus3")), evaluate_path(eq, None, fp_rules("ny_17"))
    assert plus3.status == "breached_daily" and plus3.breach["time"] == ts("2024-01-10 22:00:00")
    assert ny.status == "running"
    assert plus3.per_bar["daily_floor"].tolist()[2:4] == [98_000.0, 98_000.0]
    assert ny.per_bar["daily_floor"].tolist()[3] == pytest.approx(96_824.0)
    assert plus3.per_bar["day"].tolist() == [dnum("2024-01-10")] * 2 + [dnum("2024-01-11")] * 3
    # US summer (EDT), Wed 2024-07-10: both boundaries are 21:00 UTC. The loss closes in the 20:00 UTC bar and
    # the dip comes in the 21:00 UTC bar: both days began at 21:00 UTC at 98,800 -> floor 96,824 -> no breach.
    summer = frame([(ts("2024-07-10 19:00:00"), C0, C0, C0, 0.0),
                    (ts("2024-07-10 20:00:00"), 98_800.0, 98_800.0, 98_800.0, 0.0),
                    (ts("2024-07-10 21:00:00"), 98_800.0, 98_700.0, 97_500.0, 10.0),
                    (ts("2024-07-10 22:00:00"), 98_700.0, 98_700.0, 98_700.0, 0.0)])
    a, b = evaluate_path(summer, None, fp_rules("utc_plus3")), evaluate_path(summer, None, fp_rules("ny_17"))
    assert a.status == b.status == "running" and a.per_bar["day"].tolist() == b.per_bar["day"].tolist()


def test_bootstrap_and_daily_returns_under_utc_plus3():
    f = z.synthetic_m15_bidask(start=ts("2024-01-01 00:00:00"), n_bars=8_000, seed=5, spread=0.05)
    prep = z.prepare(f)
    eq, tr = z.cell_equity(prep, z.simulate(prep, z.ZenoConfig(z.ZenoCell("evaluation", 10.0, "S1", 1.0))))
    u = B.build_day_units(eq, C0, tr, "utc_plus3")
    t = eq["time"].to_numpy()
    assert u.day_boundary == "utc_plus3" and u.n_units == np.unique((t + 3 * H) // 86400).size
    rules = fp_rules("utc_plus3")
    a = B.bootstrap_challenges(eq, rules, tr, n_sims=200, seed=3)
    b = B.bootstrap_challenges(None, rules, None, n_sims=200, seed=3, units=u)
    assert a.to_dict() == b.to_dict()
    with pytest.raises(ValueError, match="day_boundary"):
        B.bootstrap_challenges(None, fp_rules("ny_17"), None, n_sims=50, units=u)
    d = S.daily_returns_from_equity(eq, C0, by="utc_plus3")
    assert d["day"].tolist() == sorted(set(((t + 3 * H) // 86400).tolist()))
    assert "utc_plus3" in S.DAY_KEYS


# ---------------------------------------------------------------------------------------
# A3: the verified preset

def test_verified_preset_loads_without_fallback():
    rules, info = R.rules_and_info("fundingpips-1step-flex")
    assert (R.PRESETS_DIR / "fundingpips_1step_flex.json").is_file()
    assert info["fallback"] is None and info["verified"] is True and info["firm"] == "FundingPips"
    assert info["preset"] == "fundingpips-1step-flex" and len(info["sha256"]) == 64
    # A3: target 12%; daily loss 2% of the higher of the day-start balance and equity, touching it breaches;
    # max loss 12% static (floor 88,000); no minimum days, no best-day rule; target only when flat [U]
    assert (rules.profit_target_pct, rules.daily_loss_pct, rules.max_loss_pct) == (0.12, 0.02, 0.12)
    assert rules.daily_loss_base == "day_start" and rules.day_start_reference == "max_balance_equity"
    assert rules.max_loss_mode == "static" and rules.breach_inclusive is True and rules.min_trading_days == 0
    assert rules.best_day_max_share is None and rules.target_requires_flat is True
    assert rules.day_boundary == "ny_17" and rules.initial_capital == C0
    assert rules != R.rules_and_info("fundingpips-1step-flex-placeholder")[0]
    # the placeholder keeps its own name
    assert R.FIRM_PRESET_NAMES == ("fundingpips-1step-flex", "fundingpips-1step-flex-placeholder")
    assert "PLACEHOLDER" in R.rules_and_info("fundingpips-1step-flex-placeholder")[0].name


def test_verified_preset_u_tags_and_unmodelled():
    _, info = R.rules_and_info("fundingpips-1step-flex")
    # tags starting with [U]: target_requires_flat and best_day_basis (the one tag that had to be fixed to load)
    assert sorted(info["unverified_fields"]) == ["best_day_basis", "target_requires_flat"]
    # [U] anywhere in the tag: also the 'static' wording and the Platform Time DST behaviour
    assert sorted(info["u_tags"]) == ["best_day_basis", "day_boundary", "max_loss_mode", "target_requires_flat"]
    assert info["tags"]["daily_loss_pct"].startswith("[VP 1SF, CMP]")
    assert len(info["unmodelled"]) == 5 and any("20-lot hard cap" in x for x in info["unmodelled"])
    assert all(x.isascii() for x in info["unmodelled"])
    text = (R.PRESETS_DIR / "fundingpips_1step_flex.json").read_bytes()
    assert text.isascii() and json.loads(text)["_meta"]["verified"] is True


@pytest.mark.parametrize("tag,kind", [("[VP] x", "VP"), ("[VP 1SF] x", "VP"), ("[VP 1SF, CMP] x", "VP"),
                                      ("[U] x", "U"), ("[U RTP] x", "U"), ("[VPX] x", None), ("[V] x", None),
                                      ("verified [VP]", None), ("", None), (None, None)])
def test_tag_kind(tag, kind):
    assert R.tag_kind(tag) == kind


def test_loader_rejects_bad_unmodelled_and_untagged_fields():
    with pytest.raises(ValueError, match="unmodelled"):
        R.rules_from_json({"_meta": {"unmodelled": [1, 2]}, "name": "x"})
    with pytest.raises(ValueError, match=r"must start with \[VP\] or \[U\]"):
        R.rules_from_json({"_meta": {"tags": {"best_day_basis": "unused while best_day_max_share is null"}},
                           "name": "x"})
    _, info = R.rules_from_json({"_meta": {"unmodelled": "one rule"}, "name": "x"})
    assert info["unmodelled"] == ["one rule"] and info["u_tags"] == {}


def test_placeholder_is_used_only_when_the_verified_file_is_missing(tmp_path, monkeypatch):
    real = R.PRESETS_DIR
    shutil.copy(real / "fundingpips_1step_flex_placeholder.json", tmp_path)
    monkeypatch.setattr(R, "PRESETS_DIR", tmp_path)
    _, info = R.rules_and_info("fundingpips-1step-flex")
    assert "not installed yet" in info["fallback"]
    shutil.copy(real / "fundingpips_1step_flex.json", tmp_path)
    _, info2 = R.rules_and_info("fundingpips-1step-flex")
    assert info2["fallback"] is None and info2["verified"] is True


def test_cli_rules_lists_the_verified_preset_and_run_defaults_to_it(capsys):
    assert cli.main(["rules"]) == 0
    out = capsys.readouterr().out
    block = out.split("--rules fundingpips-1step-flex ", 1)[1].split("--rules fundingpips-1step-flex-placeholder")[0]
    assert "verified rule sheet" in block and "NOTE:" not in block and "not installed" not in block
    assert "+12.00%" in block and "88,000.00" in block
    args = cli.make_parser().parse_args(["zeno-v1", "run", "--m15-bid", "b.csv", "--m15-ask", "a.csv", "--news",
                                         "n.csv", "--out", "o"])
    assert args.rules == "fundingpips-1step-flex"
