"""zeno_v1 entries (rules 5-7, D11-D13, D16): fill prices, stop levels, R, lot sizing and rounding, the
breakeven level and the closed balance at entry, hand-computed on the canonical setup of
tests/unit/zeno_v1_testkit.py (cell evaluation / commission 10 / S1 / x1 unless a test says otherwise).

Long: trigger b7 closes 2006; the next bar opens at bid 2006.00, ask 2006.20 -> entry 2006.20 (D11);
stop = pullback low 2002 - 0.25 x ATR 2 = 2001.50 (bid level); R = 4.70; 0.5% of 100,000 = 500 USD;
500 / 4.70 = 106.38 -> 106 oz = 1.06 lots (D13); tp1 = 2015.60, tp2 = 2025.00; breakeven = entry + 10 USD
per lot / 100 oz = 2006.30 (D16). A full stop fills at 2001.50 - 0.05 slippage = 2001.45:
gross = 106 x (2001.45 - 2006.20) = -503.50, commission 10 x 1.06 = 10.60, net = -514.10;
risk_usd = 106 x 4.70 = 498.20; net R = -514.10 / 498.20 = -1.0319.
Research only."""
from __future__ import annotations

import numpy as np
import pytest

from propkit import zeno_v1 as z
from tests.unit.zeno_v1_testkit import EVAL_10_X1, Scenario, triggers, utc

START = "2024-03-05 12:00"            # Tuesday, winter time: the trigger closes 14:00 UTC (22:00 SGT)


def long_case(after, start=START, tail=4, tail_price=None, **kw):
    """Canonical long + `after` bars (dicts for add()); returns (sc, trigger index, prep, res)."""
    sc = Scenario(start)
    ti = sc.setup(1)
    for bar in after:
        sc.add(**bar) if isinstance(bar, dict) else sc.add(*bar)
    if tail:
        sc.flat(tail, tail_price if tail_price is not None else sc.bid[-1][3])
    prep, res = sc.run(**kw)
    return sc, ti, prep, res


def short_case(after, start=START, tail=4, **kw):
    sc = Scenario(start)
    ti = sc.setup(-1)
    for bar in after:
        sc.add(**bar) if isinstance(bar, dict) else sc.add(*bar)
    if tail:
        sc.flat(tail, sc.bid[-1][3])
    prep, res = sc.run(trend="short", **kw)
    return sc, ti, prep, res


STOP_BAR = (2006.0, 2006.0, 2001.0, 2002.0)       # entry bar: low 2001 <= 2001.50 -> stopped inside it


def test_long_fill_stop_r_size_and_full_stop_numbers():
    sc, ti, prep, res = long_case([STOP_BAR])
    tr = triggers(res)
    assert len(tr) == 1
    d = tr.iloc[0]
    assert (d["status"], d["reasons"], d["position_id"]) == ("entered", "", 1)
    assert d["entry_time"] == sc.time(ti + 1) and d["time"] == sc.time(ti) + 900
    assert d["time_utc"] == "2024-03-05 14:00:00 UTC" and d["time_sgt"] == "2024-03-05 22:00:00 SGT"
    assert d["entry_price"] == pytest.approx(2006.20) and d["stop_level"] == pytest.approx(2001.50)
    assert d["spread_entry"] == pytest.approx(0.20) and d["atr_trigger"] == 2.0 and d["atr_median"] == 2.0
    assert d["trend_ok"]

    p = res.positions.iloc[0]
    assert p["side"] == "long" and p["setup_id"] == "L1" and p["server_day"] == "2024-03-05"
    assert p["entry_price"] == pytest.approx(2006.20)
    assert p["R_usd_per_oz"] == pytest.approx(4.70)
    assert (p["units_oz"], p["lots"], p["partial_lots"], p["runner_lots"]) == (106.0, 1.06, 0.53, 0.53)
    assert p["tp1_level"] == pytest.approx(2015.60) and p["tp2_level"] == pytest.approx(2025.00)
    assert p["be_level"] == pytest.approx(2006.30)
    assert p["balance_at_entry"] == 100_000.0 and p["risk_budget_usd"] == pytest.approx(500.0)
    assert (p["h_level"], p["l_level"], p["leg_usd"]) == (2010.0, 1995.0, 15.0)
    assert p["exit1_reason"] == "stop" and p["exit1_price"] == pytest.approx(2001.45)
    assert p["exit1_time"] == sc.time(ti + 1) + 899 and p["final_exit_stamp"] == sc.time(ti + 1) + 900
    assert p["gross_usd"] == pytest.approx(-503.50)
    assert p["commission_usd"] == pytest.approx(10.60)
    assert p["slippage_usd"] == pytest.approx(106 * 0.05)
    assert p["net_pnl_usd"] == pytest.approx(-514.10)
    assert p["r_multiple_net"] == pytest.approx(-514.10 / 498.20)
    assert p["outcome"] == "-1R" and not p["tp1_reached"] and not p["ambiguous_bar"]

    lg = res.legs
    assert len(lg) == 1
    row = lg.iloc[0]
    assert (row["leg"], row["exit_reason"], row["units"], row["side"]) == ("full", "stop", 106.0, 1)
    assert row["risk_usd"] == pytest.approx(498.20)                # units x R, costs on top [SI-26]
    assert row["stop_price"] == pytest.approx(2001.50)
    assert row["pnl_usd"] == pytest.approx(-514.10)
    assert res.meta["final_balance_usd"] == pytest.approx(100_000 - 514.10)


def test_short_fills_at_the_bid_and_its_stop_is_an_ask_level_with_the_entry_spread():
    # Mirror: trigger close 1994; next bar opens at bid 1994.00 (short entry, D11) with ask 1994.20.
    # stop = pullback high 1998 + 0.25 x 2 + entry spread 0.20 = 1998.70, checked on the ASK high (D14).
    # Entry bar: bid high 1998.60 (bid + 0.20 would be 1998.80) but the explicit ASK high is 1998.65 < 1998.70:
    # no stop. Next bar: ask high = 1998.50 + 0.20 = 1998.70 -> stopped at 1998.70 + 0.05 = 1998.75.
    # gross = -106 x (1998.75 - 1994.00) = -503.50, net -514.10 (the mirror of the long).
    entry_bar = dict(o=1994.0, h=1998.6, lo=1993.5, c=1996.0, ask=(1994.2, 1998.65, 1993.7, 1996.2))
    sc, ti, prep, res = short_case([entry_bar, (1996.0, 1998.5, 1995.5, 1998.0)])
    d = triggers(res).iloc[0]
    assert d["side"] == "short" and d["status"] == "entered"
    assert d["entry_price"] == pytest.approx(1994.00) and d["stop_level"] == pytest.approx(1998.70)
    p = res.positions.iloc[0]
    assert p["R_usd_per_oz"] == pytest.approx(4.70) and p["units_oz"] == 106.0
    assert p["tp1_level"] == pytest.approx(1984.60) and p["tp2_level"] == pytest.approx(1975.20)
    assert p["be_level"] == pytest.approx(1993.90)
    assert p["exit1_time"] == sc.time(ti + 2) + 899                 # not in the entry bar
    assert p["exit1_price"] == pytest.approx(1998.75)
    assert p["net_pnl_usd"] == pytest.approx(-514.10)
    assert res.legs.iloc[0]["side"] == -1


QUIET = (2006.0, 2007.0, 2005.0, 2006.5)          # entry bar: nothing happens
TP1_BAR = (2006.5, 2016.0, 2006.5, 2015.0)        # +2R (2015.60) reached, low stays above breakeven 2006.30
BACK_TO_BE = (2015.0, 2015.0, 2006.0, 2007.0)     # low 2006 <= 2006.30: breakeven stop at 2006.25


@pytest.mark.parametrize("capital,units,partial,runner", [
    (100_000.0, 106.0, 53.0, 53.0),     # 500 / 4.70 = 106.38 -> 106; half = 53
    (101_000.0, 107.0, 53.0, 54.0),     # 505 / 4.70 = 107.45 -> 107; floor(53.5) = 53, the runner keeps 54
    (3_000.0, 3.0, 1.0, 2.0),           # 15 / 4.70 = 3.19 -> 3; floor(1.5) = 1
    (1_000.0, 1.0, 0.0, 1.0),           # 5 / 4.70 = 1.06 -> 1 oz = 0.01 lot; floor(0.5) = 0: no partial
])
def test_lot_and_partial_rounding(capital, units, partial, runner):
    sc, ti, prep, res = long_case([QUIET, TP1_BAR, BACK_TO_BE], capital=capital)
    p = res.positions.iloc[0]
    assert (p["units_oz"], p["partial_lots"] * 100, p["runner_lots"] * 100) == (units, partial, runner)
    assert p["tp1_reached"]
    legs = res.legs
    if partial:
        assert legs["leg"].tolist() == ["tp1", "runner"]
        assert legs["units"].tolist() == [partial, runner]
        assert legs["exit_price"].tolist() == pytest.approx([2015.60, 2006.25])
        assert p["outcome"] == "+1R(BE)"
    else:
        # 0.01 lot: nothing to take at +2R, but the stop still moves to breakeven (D16) [SI-22]
        assert legs["leg"].tolist() == ["full"] and legs["units"].tolist() == [1.0]
        assert legs["exit_price"].iloc[0] == pytest.approx(2006.25)
        assert p["outcome"] == "other"


def test_tp1_then_breakeven_pnl_by_hand():
    # tp1 leg 53 oz at 2015.60: gross 53 x 9.40 = 498.20, commission 5.30, net 492.90.
    # runner 53 oz at 2006.30 - 0.05 = 2006.25: gross 53 x 0.05 = 2.65, commission 5.30, net -2.65.
    sc, ti, prep, res = long_case([QUIET, TP1_BAR, BACK_TO_BE])
    lg = res.legs
    assert lg["pnl_usd"].tolist() == pytest.approx([492.90, -2.65])
    assert lg["exit_reason"].tolist() == ["target", "stop"]
    assert lg["exit_time"].tolist() == [sc.time(ti + 2) + 899, sc.time(ti + 3) + 899]
    p = res.positions.iloc[0]
    assert p["net_pnl_usd"] == pytest.approx(490.25)
    assert p["exit1_reason"] == "target" and p["exit2_reason"] == "stop"
    assert p["r_multiple_net"] == pytest.approx(490.25 / 498.20)


@pytest.mark.parametrize("cell,side,be", [
    (z.ZenoCell("evaluation", 10.0, "S1", 1.0), 1, 2006.30),     # 10 / 100 = 0.10 USD/oz above 2006.20
    (z.ZenoCell("evaluation", 5.0, "S1", 1.0), 1, 2006.25),      # 5 / 100
    (z.ZenoCell("evaluation", 10.0, "S1", 1.5), 1, 2006.45),     # entry 2006.30 (spread 0.30) + 15 / 100
    (z.ZenoCell("evaluation", 10.0, "S1", 1.0), -1, 1993.90),    # short: 1994.00 - 0.10
])
def test_breakeven_level(cell, side, be):
    fn = long_case if side > 0 else short_case
    sc, ti, prep, res = fn([], tail=6, cell=cell)
    assert res.positions.iloc[0]["be_level"] == pytest.approx(be)


def test_entry_at_or_beyond_the_stop_is_blocked():
    # the next bar opens at bid 2001.00 (ask 2001.20) under the stop 2001.50: R = -0.30
    sc, ti, prep, res = long_case([(2001.0, 2001.0, 2000.0, 2000.5)])
    d = triggers(res).iloc[0]
    assert d["status"] == "entry_beyond_stop" and len(res.positions) == 0
    assert d["entry_price"] == pytest.approx(2001.20) and d["position_id"] == -1


def test_entry_after_an_opening_gap_uses_the_actual_open():
    # the next bar gaps up to bid 2007.00 -> entry 2007.20, R = 5.70 (<= 3 x 2), 500 / 5.70 = 87.7 -> 87 oz,
    # tp1 = 2007.20 + 11.40 = 2018.60
    sc, ti, prep, res = long_case([(2007.0, 2007.5, 2006.5, 2007.0)])
    p = res.positions.iloc[0]
    assert p["entry_price"] == pytest.approx(2007.20) and p["R_usd_per_oz"] == pytest.approx(5.70)
    assert p["units_oz"] == 87.0 and p["tp1_level"] == pytest.approx(2018.60)


def test_size_below_one_lot_step_is_blocked():
    # 0.5% of 900 = 4.50 USD < 4.70 x 1 oz
    sc, ti, prep, res = long_case([STOP_BAR], capital=900.0)
    d = triggers(res).iloc[0]
    assert d["status"] == "size_below_lot_step" and len(res.positions) == 0


def test_risk_comes_from_the_closed_balance_at_entry():
    # Trade 1 (trigger closes 08:15 UTC) loses 514.10. Trade 2 (15:30 UTC) sizes on 99,485.90:
    # 0.5% = 497.4295 / 4.70 = 105.84 -> 105 oz; its full stop: 105 x (-4.75) - 10.50 = -509.25.
    sc = Scenario("2024-03-05 06:15")
    t1 = sc.setup(1)                                   # trigger closes 08:15 UTC
    sc.add(*STOP_BAR)
    sc.flat(20, 2000.0)
    t2 = sc.setup(1)                                   # second setup, trigger closes 15:30 UTC
    sc.add(*STOP_BAR)
    sc.flat(4, 2002.0)
    prep, res = sc.run()
    tr = triggers(res)
    entered = tr[tr["status"] == "entered"]
    assert entered["bar_index"].tolist() == [t1, t2]
    pos = res.positions
    assert pos["units_oz"].tolist() == [106.0, 105.0]
    assert pos["balance_at_entry"].tolist() == pytest.approx([100_000.0, 99_485.90])
    assert pos["risk_budget_usd"].tolist() == pytest.approx([500.0, 497.4295])
    assert pos["net_pnl_usd"].tolist() == pytest.approx([-514.10, -509.25])
    assert res.meta["final_balance_usd"] == pytest.approx(100_000 - 514.10 - 509.25)


def test_master_variant_risks_0_4_percent():
    # 0.4% of 100,000 = 400 / 4.70 = 85.1 -> 85 oz
    sc, ti, prep, res = long_case([STOP_BAR], cell=z.ZenoCell("master", 10.0, "S1", 1.0))
    assert res.positions.iloc[0]["units_oz"] == 85.0
    assert res.meta["risk_pct"] == 0.004


def test_explicit_risk_pct_and_its_check():
    sc, ti, prep, res = long_case([STOP_BAR], risk_pct=0.01)          # 1000 / 4.70 = 212.8 -> 212
    assert res.positions.iloc[0]["units_oz"] == 212.0
    with pytest.raises(ValueError, match="fraction"):
        z.ZenoConfig(EVAL_10_X1, 100_000.0, 0.5)
