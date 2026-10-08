"""PropRules.day_boundary (deliverable 3, D24): the firm's day in propkit.calendar, the evaluator, the
bootstrap and the statistics; the rules-from-JSON path and the FundingPips placeholder preset.

Known answers are computed by hand in the comments. US DST 2024: summer time from Sun 2024-03-10 07:00 UTC
(02:00 EST) to Sun 2024-11-03 06:00 UTC (02:00 EDT), so 17:00 New York is 22:00 UTC before 2024-03-10 and
from 2024-11-03 on, and 21:00 UTC in between. Research only; synthetic data only.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

import propkit
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
    return int(dt.datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=UTC).timestamp())


def dnum(text: str) -> int:
    return (dt.date.fromisoformat(text) - dt.date(1970, 1, 1)).days


def frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS)
    df["time"] = df["time"].astype(np.int64)
    for c in COLS[1:]:
        df[c] = df[c].astype(np.float64)
    return df


def daily_rules(boundary: str) -> R.PropRules:
    """2% of the initial capital below the BALANCE at the day start; no target, no max-loss rule."""
    return R.PropRules(name=f"t-{boundary}", initial_capital=C0, profit_target_pct=None, daily_loss_pct=0.02,
                       daily_loss_base="initial", day_start_reference="balance", max_loss_pct=None,
                       best_day_max_share=None, min_trading_days=0, day_boundary=boundary)


# ---------------------------------------------------------------------------------------
# calendar.firm_day

@pytest.mark.parametrize("instant,ny17,utc_day", [
    ("2024-03-08 21:59", "2024-03-08", "2024-03-08"),     # Fri, EST: 17:00 NY = 22:00 UTC, not reached
    ("2024-03-08 22:00", "2024-03-09", "2024-03-08"),     # 17:00 EST: the next server day starts
    ("2024-03-10 20:59", "2024-03-10", "2024-03-10"),     # Sun after the change: EDT, 17:00 NY = 21:00 UTC
    ("2024-03-10 21:00", "2024-03-11", "2024-03-10"),
    ("2024-11-01 20:59", "2024-11-01", "2024-11-01"),     # Fri, still EDT
    ("2024-11-01 21:00", "2024-11-02", "2024-11-01"),
    ("2024-11-03 21:59", "2024-11-03", "2024-11-03"),     # Sun after the change: EST again, 22:00 UTC
    ("2024-11-03 22:00", "2024-11-04", "2024-11-03"),
    ("2024-07-10 23:59", "2024-07-11", "2024-07-10"),
    ("2024-07-11 00:00", "2024-07-11", "2024-07-11"),     # 00:00 UTC starts the UTC day only
])
def test_ny_17_and_utc_midnight_known_answers(instant, ny17, utc_day):
    t = ts(instant)
    assert cal.day_to_str(cal.firm_day(t, "ny_17")) == ny17
    assert cal.day_to_str(cal.firm_day(t, "utc_midnight")) == utc_day
    assert cal.firm_day(t, "cet_midnight") == cal.prop_day(t)
    assert cal.firm_day(t) == cal.prop_day(t)                       # the default is the CE(S)T prop day


def test_firm_day_start_round_trips_2015_2026_and_the_server_day():
    days = np.arange(dnum("2015-01-01"), dnum("2026-12-31") + 1)
    for b in cal.DAY_BOUNDARIES:
        start = np.array([cal.firm_day_start_utc(int(d), b) for d in days], dtype=np.int64)
        assert (np.asarray(cal.firm_day(start, b)) == days).all(), b
        assert (np.asarray(cal.firm_day(start - 1, b)) == days - 1).all(), b
    # 17:00 New York: 22:00 UTC in winter, 21:00 UTC in summer
    assert cal.firm_day_start_utc(dnum("2024-03-09"), "ny_17") == ts("2024-03-08 22:00")
    assert cal.firm_day_start_utc(dnum("2024-03-11"), "ny_17") == ts("2024-03-10 21:00")
    assert cal.firm_day_start_utc(dnum("2024-11-04"), "ny_17") == ts("2024-11-03 22:00")
    assert cal.firm_day_start_utc(dnum("2024-07-11"), "utc_midnight") == ts("2024-07-11 00:00")
    # the zeno_v1 broker server day (D21) is the ny_17 firm day
    t = z.synthetic_m15_bidask(n_bars=3000, seed=2)["time"].to_numpy()
    assert (np.asarray(z.server_day(t)) == np.asarray(cal.firm_day(t, "ny_17"))).all()


def test_day_boundary_names_and_labels():
    assert cal.DAY_BOUNDARIES == ("cet_midnight", "ny_17", "utc_midnight")
    assert cal.boundary_label("ny_17") == "17:00 New York" and cal.boundary_label() == "00:00 CE(S)T"
    with pytest.raises(ValueError, match="day_boundary must be one of"):
        cal.firm_day(0, "ny_18")


# ---------------------------------------------------------------------------------------
# PropRules, the evaluator and the bootstrap

def test_prop_rules_day_boundary_default_and_validation():
    assert R.PropRules().day_boundary == "cet_midnight"
    assert R.ftmo_1step().day_boundary == "cet_midnight" and R.ftmo_2step().day_boundary == "cet_midnight"
    assert R.PropRules(day_boundary="ny_17").day_boundary == "ny_17"
    with pytest.raises(ValueError, match="day_boundary"):
        R.PropRules(day_boundary="midnight")
    lines = daily_rules("ny_17").describe()
    assert any("max(balance" not in x and "balance at 17:00 New York" in x for x in lines)
    assert any(x.strip().startswith("firm day: starts at 17:00 New York") for x in lines)
    assert not any("firm day" in x for x in R.ftmo_1step().describe())       # default text unchanged


def test_evaluator_daily_floor_resets_at_17_00_new_york():
    # 2024-07-10 (EDT): 17:00 New York = 21:00 UTC, 00:00 CEST = 22:00 UTC (the previous UTC day).
    # A loss of 1,200 USD closes in the 19:00 UTC bar (balance 98,800); in the 21:00 UTC bar a position dips
    # to 97,500 and closes at 98,700.
    #   cet_midnight: the day began 2024-07-09 22:00 UTC at balance 100,000 -> floor 98,000 -> 97,500 breaches.
    #   ny_17:        a new day began at 21:00 UTC at balance 98,800 -> floor 96,800 -> no breach.
    #   utc_midnight: the day began 2024-07-10 00:00 UTC at 100,000 -> floor 98,000 -> breaches.
    rows = [(ts("2024-07-10 18:00"), C0, C0, C0, 0.0),
            (ts("2024-07-10 19:00"), 98_800.0, 98_800.0, 98_800.0, 0.0),
            (ts("2024-07-10 20:00"), 98_800.0, 98_800.0, 98_800.0, 0.0),
            (ts("2024-07-10 21:00"), 98_800.0, 98_700.0, 97_500.0, 10.0),
            (ts("2024-07-10 22:00"), 98_700.0, 98_700.0, 98_700.0, 0.0)]
    eq = frame(rows)
    res = {b: evaluate_path(eq, None, daily_rules(b)) for b in cal.DAY_BOUNDARIES}
    assert res["cet_midnight"].status == "breached_daily" and res["cet_midnight"].breach["time"] == ts("2024-07-10 21:00")
    assert res["utc_midnight"].status == "breached_daily"
    assert res["ny_17"].status == "running"
    floors = res["ny_17"].per_bar["daily_floor"].to_numpy()
    assert floors.tolist() == [98_000.0, 98_000.0, 98_000.0, 96_800.0, 96_800.0]
    assert res["ny_17"].per_bar["day"].tolist() == [dnum("2024-07-10")] * 3 + [dnum("2024-07-11")] * 2


def test_evaluator_utc_midnight_known_answer():
    # A loss of 1,200 USD closes in the 22:00 UTC bar; at 00:00 UTC (2024-07-11) a position dips to 97,500.
    #   cet_midnight: the day began at 22:00 UTC at balance 100,000 (before the loss) -> floor 98,000 -> breach.
    #   ny_17:        the day began at 21:00 UTC at 100,000 -> floor 98,000 -> breach.
    #   utc_midnight: the day began at 00:00 UTC at 98,800 -> floor 96,800 -> no breach.
    rows = [(ts("2024-07-10 20:00"), C0, C0, C0, 0.0),
            (ts("2024-07-10 21:00"), C0, C0, C0, 0.0),
            (ts("2024-07-10 22:00"), 98_800.0, 98_800.0, 98_800.0, 0.0),
            (ts("2024-07-10 23:00"), 98_800.0, 98_800.0, 98_800.0, 0.0),
            (ts("2024-07-11 00:00"), 98_800.0, 98_600.0, 97_500.0, 10.0),
            (ts("2024-07-11 01:00"), 98_600.0, 98_600.0, 98_600.0, 0.0)]
    eq = frame(rows)
    st = {b: evaluate_path(eq, None, daily_rules(b)).status for b in cal.DAY_BOUNDARIES}
    assert st == {"cet_midnight": "breached_daily", "ny_17": "breached_daily", "utc_midnight": "running"}


def _synthetic_path():
    f = z.synthetic_m15_bidask(start=ts("2024-01-01 00:00"), n_bars=12_000, seed=5, spread=0.05)
    prep = z.prepare(f)
    res = z.simulate(prep, z.ZenoConfig(z.ZenoCell("evaluation", 10.0, "S1", 1.0)))
    return z.cell_equity(prep, res)


def test_bootstrap_units_follow_the_rules_boundary_and_a_mismatch_is_refused():
    eq, tr = _synthetic_path()
    u_ny = B.build_day_units(eq, C0, tr, "ny_17")
    u_cet = B.build_day_units(eq, C0, tr)
    assert u_ny.day_boundary == "ny_17" and u_cet.day_boundary == "cet_midnight"
    t = eq["time"].to_numpy()
    assert u_ny.n_units == np.unique(cal.firm_day(t, "ny_17")).size
    assert u_cet.n_units == np.unique(cal.prop_day(t)).size
    rules = daily_rules("ny_17")
    a = B.bootstrap_challenges(eq, rules, tr, n_sims=300, seed=3)
    b = B.bootstrap_challenges(None, rules, None, n_sims=300, seed=3, units=u_ny)
    assert a.p_breach_daily == b.p_breach_daily and a.n_units == u_ny.n_units
    with pytest.raises(ValueError, match="day_boundary"):
        B.bootstrap_challenges(None, daily_rules("cet_midnight"), None, n_sims=50, units=u_ny)


def test_ftmo_known_answers_unchanged():
    eq, tr = _synthetic_path()
    for base in (R.ftmo_1step(), R.ftmo_2step()):
        explicit = dataclasses.replace(base, day_boundary="cet_midnight")
        a, b = evaluate_path(eq, tr, base), evaluate_path(eq, tr, explicit)
        assert a.status == b.status and a.final_balance == b.final_balance and a.max_dd_usd == b.max_dd_usd
        assert (a.per_bar["day"].to_numpy() == np.asarray(cal.prop_day(eq["time"].to_numpy()))).all()
        x = B.bootstrap_challenges(eq, base, tr, n_sims=300, seed=11)
        y = B.bootstrap_challenges(eq, explicit, tr, n_sims=300, seed=11)
        assert x.to_dict() == y.to_dict()
    # every FTMO known-answer gate of the selftest still passes (29 of 29)
    n_ok, n_fail = propkit.run_selftest(out=lambda s: None)
    assert (n_ok, n_fail) == (29, 0)


def test_daily_returns_by_boundary():
    # one bar per hour from Fri 2024-03-08 20:00 to 23:00 UTC (EST: 17:00 NY = 22:00 UTC; CET: 00:00 CET =
    # 23:00 UTC); equity closes 100,100 / 100,200 / 100,300 / 100,400.
    rows = [(ts("2024-03-08 20:00") + k * H, C0, C0 + 100.0 * (k + 1), C0 + 100.0 * (k + 1), 1.0) for k in range(4)]
    eq = frame(rows)
    ny = S.daily_returns_from_equity(eq, C0, by="ny_17")
    assert ny["date"].tolist() == ["2024-03-08", "2024-03-09"]           # 20:00, 21:00 | 22:00, 23:00
    assert ny["equity_end"].tolist() == [100_200.0, 100_400.0]
    cet = S.daily_returns_from_equity(eq, C0, by="cet_midnight")
    assert cet["date"].tolist() == ["2024-03-08", "2024-03-09"]          # 20:00..22:00 | 23:00
    assert cet["equity_end"].tolist() == [100_300.0, 100_400.0]
    assert cet.equals(S.daily_returns_from_equity(eq, C0))               # prop_day = cet_midnight
    utc = S.daily_returns_from_equity(eq, C0, by="utc_midnight")
    assert utc["date"].tolist() == ["2024-03-08"]
    with pytest.raises(ValueError, match="by must be"):
        S.daily_returns_from_equity(eq, C0, by="ny_18")


# ---------------------------------------------------------------------------------------
# rules files and the FundingPips placeholder

def test_placeholder_preset_lists_every_unverified_field():
    rules, info = R.rules_and_info("fundingpips-1step-flex-placeholder")
    assert "PLACEHOLDER" in rules.name and "UNVERIFIED" in rules.name
    assert info["firm"] == "FundingPips" and info["verified"] is False and "UNVERIFIED" in info["status"]
    fields = [f.name for f in dataclasses.fields(R.PropRules) if f.name not in ("name", "notes")]
    assert sorted(info["unverified_fields"]) == sorted(fields)          # every rule field is [U]
    assert all(info["tags"][f].startswith("[U]") for f in fields)
    # no FundingPips number is invented: the target and the max-loss rule stay off
    assert rules.profit_target_pct is None and rules.max_loss_pct is None
    assert rules.daily_loss_pct == 0.02 and rules.initial_capital == 100_000.0 and rules.day_boundary == "ny_17"
    assert rules.breach_inclusive is True and rules.day_start_reference == "max_balance_equity"
    assert "G4 is not evaluated" in info["warning"] and info["fallback"] is None
    assert info["sha256"] and len(info["sha256"]) == 64


def test_fundingpips_name_falls_back_to_the_placeholder_and_says_so():
    rules, info = R.rules_and_info("fundingpips-1step-flex")
    assert not (R.PRESETS_DIR / "fundingpips_1step_flex.json").exists()
    assert info["preset"] == "fundingpips-1step-flex" and "not installed yet" in info["fallback"]
    assert rules == R.rules_and_info("fundingpips-1step-flex-placeholder")[0]
    assert R.preset("fundingpips-1step-flex", 50_000).initial_capital == 50_000
    assert R.PRESET_NAMES == ("ftmo-1step", "ftmo-2step")


def test_rules_from_json_with_base_meta_and_tags(tmp_path):
    p = tmp_path / "sheet.json"
    p.write_text(json.dumps({
        "_comment": "TEST FIXTURE - synthetic numbers, not any firm's terms",
        "_meta": {"firm": "TestFirm", "verified": True, "verified_on": "2026-10-08", "sources": ["unit test"],
                  "tags": {"daily_loss_pct": "[VP] test page, 2026-10-08", "day_boundary": "[U] assumed"}},
        "name": "Test sheet", "initial_capital": 50_000, "profit_target_pct": 0.031, "daily_loss_pct": 0.02,
        "max_loss_pct": 0.061, "day_boundary": "utc_midnight"}), encoding="ascii")
    rules, info = R.rules_from_json(p)
    assert rules.day_boundary == "utc_midnight" and rules.initial_capital == 50_000
    assert info["tags"]["daily_loss_pct"].startswith("[VP]") and info["tags"]["max_loss_pct"].startswith("[VP]")
    assert info["unverified_fields"] == ["day_boundary"] and info["verified"] is True
    # a "base" file changes only the fields it lists; a changed field of a tagged preset becomes [U]
    rules2, info2 = R.rules_from_json({"base": "fundingpips-1step-flex-placeholder", "daily_loss_pct": 0.03})
    assert rules2.daily_loss_pct == 0.03 and rules2.day_boundary == "ny_17"
    assert "changed" in info2["tags"]["daily_loss_pct"]
    rules3, _ = R.rules_from_json({"base": "ftmo-1step", "day_boundary": "ny_17"})
    assert rules3.day_boundary == "ny_17" and rules3.daily_loss_pct == R.ftmo_1step().daily_loss_pct
    with pytest.raises(ValueError, match=r"must start with \[VP\] or \[U\]"):
        R.rules_from_json({"_meta": {"tags": {"daily_loss_pct": "verified"}}, "name": "x"})
    with pytest.raises(ValueError, match="unknown _meta key"):
        R.rules_from_json({"_meta": {"firmm": "x"}, "name": "x"})
    with pytest.raises(ValueError):
        R.rules_from_json({"name": "x", "daily_los_pct": 0.02})              # a typo cannot pass silently


def test_cli_accepts_the_firm_presets_and_lists_them(capsys):
    r = cli.build_rules("fundingpips-1step-flex-placeholder", 100_000.0)
    assert r.day_boundary == "ny_17" and r.daily_loss_pct == 0.02
    assert cli.build_rules("ftmo-1step") == R.ftmo_1step()
    with pytest.raises(cli.UsageError, match="unknown --rules"):
        cli.build_rules("fundingpips-2step")
    assert cli.main(["rules"]) == 0
    out = capsys.readouterr().out
    assert "--rules fundingpips-1step-flex " in out and "UNVERIFIED" in out and "unverified [U] fields" in out
    assert "--rules ftmo-1step" in out and "3.00% of initial capital" in out


def test_bootstrap_warning_names_the_firm_day_boundary():
    from propkit import report as rep
    blocks = {"n_blocks_days": 100, "n_days": 200, "share_open_at_start": 0.6}
    ftmo = rep.bootstrap_warnings(blocks)
    assert ftmo == rep.bootstrap_warnings(blocks, "cet_midnight") and "held over midnight" in ftmo[0]
    ny = rep.bootstrap_warnings(blocks, "ny_17")
    assert "held over the day boundary (17:00 New York)" in ny[0] and "midnight" not in ny[0]


def test_generic_report_wording_follows_the_firm_day(tmp_path, capsys):
    # A pullback run (the example PLACEHOLDER spec, synthetic H1 bars) under FTMO's rules with the firm day moved
    # to 17:00 New York: the report must not describe that day as running from midnight / 00:00, while the FTMO
    # (CE(S)T midnight) report keeps its wording byte for byte.
    from pathlib import Path
    from propkit.bars import synthetic_bars
    bars = tmp_path / "XAUUSD_H1.parquet"
    synthetic_bars(1704067200, 3000, seed=9, spread=0.34).to_parquet(bars, index=False)
    rules_ny = tmp_path / "rules_ny17.json"
    rules_ny.write_text(json.dumps({"base": "ftmo-1step", "day_boundary": "ny_17"}), encoding="ascii")
    spec = Path(propkit.__file__).resolve().parent / "examples" / "pullback_spec_example.json"
    md = {}
    for key, rules in (("ny", rules_ny), ("ftmo", "ftmo-1step")):
        out = tmp_path / key
        code = cli.main(["pullback", "--bars", str(bars), "--spec", str(spec), "--rules", str(rules), "--n-sims", "200",
                         "--history-reps", "0", "--no-stress", "--out", str(out)])
        capsys.readouterr()
        assert code == 0
        md[key] = (out / "report.md").read_text(encoding="ascii")
    assert "Prop days run 17:00 New York to 17:00 New York" in md["ny"]
    assert "midnight" not in md["ny"] and "00:00 balance" not in md["ny"]
    assert "a position held over the day boundary (17:00 New York) keeps its days together" in md["ny"]
    assert "a position held over midnight keeps its days together" in md["ftmo"]
    if "would break the daily limit" in md["ny"]:
        assert "drawdown from the 17:00 New York balance" in md["ny"]
        assert "drawdown from the 00:00 balance" in md["ftmo"]
