"""Addendum A2 (zeno_pullback_v1): FundingPips' Master margin cap in the variants master and master_fp; the
evaluation variant is not capped and only counted at a flat 1:10 and 1:30.

Tiers per position (100 oz = 1 lot): 0.05 lot at 1:50, the next 0.05 at 1:30, the next 0.05 at 1:25, the next
0.10 at 1:20, the next 0.25 at 1:10, the rest at 1:5. Margin = 100 x P x sum(lots in tier / leverage).
Research only; synthetic bars.
"""
from __future__ import annotations

import pytest

from propkit import zeno_v1 as z
from tests.unit.zeno_v1_testkit import Scenario, triggers

MASTER = z.ZenoCell("master", 10.0, "S1", 1.0)
MASTER_FP = z.ZenoCell("master_fp", 10.0, "S1", 1.0)
EVAL = z.ZenoCell("evaluation", 10.0, "S1", 1.0)
EMPTY = z.restricted_calendar([])


def test_margin_known_answers():
    # 0.50 lot at 4,000: 5 x 4000/50 + 5 x 4000/30 + 5 x 4000/25 + 10 x 4000/20 + 25 x 4000/10
    #                  = 400 + 666.67 + 800 + 2,000 + 10,000 = 13,866.67; every lot above 0.50 adds 4000 x 100/5
    #                  = 80,000.
    assert z.margin_usd(5, 4000.0) == pytest.approx(400.0)
    assert z.margin_usd(50, 4000.0) == pytest.approx(13_866.666667)
    # 1.57 lots at 4,000: 13,866.67 + 1.07 x 80,000 =  99,466.67 <= 100,000: fits
    # 1.58 lots at 4,000: 13,866.67 + 1.08 x 80,000 = 100,266.67 >  100,000: does not fit
    assert z.margin_usd(157, 4000.0) == pytest.approx(99_466.666667)
    assert z.margin_usd(158, 4000.0) == pytest.approx(100_266.666667)
    assert z.max_units_within_margin(100_000.0, 4000.0) == 157.0
    # at 3,700: 0.50 lot = 12,826.67, each further lot 74,000
    # 1.67 lots: 12,826.67 + 1.17 x 74,000 =  99,406.67: fits; 1.68 lots: 12,826.67 + 1.18 x 74,000 = 100,146.67: not
    assert z.margin_usd(167, 3700.0) == pytest.approx(99_406.666667)
    assert z.margin_usd(168, 3700.0) == pytest.approx(100_146.666667)
    assert z.max_units_within_margin(100_000.0, 3700.0) == 167.0
    # NOTE (ADDENDUM_ISSUES [AI-1]): the build brief's "1.58 fits at 4,000" and "1.68 fits at 3,700" contradict
    # A2's own formula (above); 1.58 and 1.68 are A2's rounded continuous break-even sizes (1.5767 and 1.6780
    # lots), not sizes that fit. The tests assert the formula.
    # 1 oz at 4,000 needs 80 USD; nothing fits below that
    assert z.max_units_within_margin(80.0, 4000.0) == 1.0 and z.max_units_within_margin(79.99, 4000.0) == 0.0
    assert z.max_units_within_margin(0.0, 4000.0) == 0.0
    # exactly at the balance fits (to 1e-6 USD)
    assert z.max_units_within_margin(z.margin_usd(157, 4000.0), 4000.0) == 157.0


def _flat_then_tp1(cell, risk_pct=None, capital=100_000.0, base=2000.0):
    """Canonical long, trigger close 08:00 UTC Wed 2024-02-14 (EST); entry bar flat at bid 2006 (ask open
    2006.20); then a bar (2010, 2016, 2010, 2012) takes +2R at 2015.60 (stop 2001.50 and breakeven 2006.30 not
    touched), then flat 2012.00 to the 16:30 New York exit (21:30 UTC)."""
    d = base - 2000.0
    sc = Scenario("2024-02-14 06:00", base=base)
    ti = sc.setup(1, base=base)
    sc.add(2006.0 + d, 2006.0 + d, 2006.0 + d, 2006.0 + d)
    sc.add(2010.0 + d, 2016.0 + d, 2010.0 + d, 2012.0 + d)
    sc.flat_until("2024-02-14 23:00", 2012.0 + d)
    prep, res = sc.run(cell=cell, risk_pct=risk_pct, capital=capital, restricted=EMPTY)
    return ti, prep, res


def test_a_capped_master_position_and_its_pnl():
    # master at 5% risk (risk_pct is a ZenoConfig setting; the spec's 0.4% never binds near 2,000 USD gold):
    # D13: 5,000 / 4.70 = 1,063.8 -> 1,063 oz = 10.63 lots. Margin at the fill 2006.20: 0.50 lot = 6,954.83, each
    # further lot 40,124: 2.81 lots = 6,954.83 + 2.31 x 40,124 = 99,641.27 fits, 2.82 = 100,042.51 does not
    # -> capped to 281 oz.
    for cell in (MASTER, MASTER_FP):
        ti, prep, res = _flat_then_tp1(cell, risk_pct=0.05)
        p = res.positions.iloc[0]
        assert (p["units_oz"], p["lots"], p["lots_uncapped"], bool(p["margin_capped"])) == (281.0, 2.81, 10.63, True)
        assert z.margin_usd(281, 2006.20) == pytest.approx(99_641.2667, abs=1e-3)
        # P&L of the capped size: 140 oz at +2R (2015.60 - 2006.20 = 9.40) = 1,316.00; the runner 141 oz to the
        # time exit at 2012.00: 141 x 5.80 = 817.80; commission 10 x 2.81 = 28.10 -> net 2,105.70
        assert (p["partial_lots"], p["runner_lots"]) == (1.40, 1.41)
        assert p["net_pnl_usd"] == pytest.approx(1316.0 + 817.8 - 28.1)
        assert p["r_multiple_net"] == pytest.approx((1316.0 + 817.8 - 28.1) / (281 * 4.70))
        m = res.meta["margin"]
        assert (m["cap_applies"], m["n_capped"], m["lots_before_cap"], m["lots_after_cap"]) == (True, 1, 10.63, 2.81)
        eq, _ = z.cell_equity(prep, res)
        assert eq["balance"].iloc[-1] == pytest.approx(100_000.0 + 2105.70)
    # the same 281 oz reached through D13 alone (evaluation, risk 1,322 / 4.70 = 281.3 -> 281) gives the same P&L
    ti, prep, ref = _flat_then_tp1(EVAL, risk_pct=0.01322)
    assert ref.positions.iloc[0]["units_oz"] == 281.0
    assert ref.positions.iloc[0]["net_pnl_usd"] == pytest.approx(res.positions.iloc[0]["net_pnl_usd"])


def test_the_spec_risk_is_not_capped_near_2000():
    # 0.4%: 85 oz = 0.85 lot needs 6,954.83 + 0.35 x 40,124 = 20,998.23 USD << 100,000: lots unchanged
    ti, prep, res = _flat_then_tp1(MASTER)
    p = res.positions.iloc[0]
    assert (p["units_oz"], p["lots_uncapped"], bool(p["margin_capped"])) == (85.0, 0.85, False)
    assert res.meta["margin"]["n_capped"] == 0


def test_evaluation_is_not_capped_but_counted_at_flat_leverage():
    # evaluation at 5% risk: 1,063 oz at 2006.20 = 2,132,590.60 USD notional; at 1:10 that needs 213,259.06 >
    # 100,000 (counted), at 1:30 71,086.35 (not counted). The position keeps its 10.63 lots.
    ti, prep, res = _flat_then_tp1(EVAL, risk_pct=0.05)
    p = res.positions.iloc[0]
    assert (p["units_oz"], p["lots_uncapped"], bool(p["margin_capped"])) == (1063.0, 10.63, False)
    m = res.meta["margin"]
    assert m["cap_applies"] is False and m["n_over_flat_1to10"] == 1 and m["n_over_flat_1to30"] == 0
    ti, prep, res = _flat_then_tp1(EVAL)                                          # 106 oz: neither
    assert (res.meta["margin"]["n_over_flat_1to10"], res.meta["margin"]["n_over_flat_1to30"]) == (0, 0)


def test_an_entry_that_cannot_hold_001_lot_is_blocked_by_margin():
    # synthetic bars around 50,000 USD/oz (NOT a market price; only to reach the rule): 990 USD at 10% risk buys
    # 99 / 4.70 = 21 oz by D13, but 1 oz at the fill 50,006.20 needs 50,006.20 / 50 = 1,000.12 USD > 990:
    # blocked with margin_cap_below_lot_step (before size_below_lot_step), in both Master variants only
    for cell, want in ((MASTER, "margin_cap_below_lot_step"), (MASTER_FP, "margin_cap_below_lot_step"),
                       (EVAL, "entered")):
        ti, prep, res = _flat_then_tp1(cell, risk_pct=0.1, capital=990.0, base=50_000.0)
        row = triggers(res).set_index("bar_index").loc[ti]
        assert row["status"] == want and row["reasons"] == ("" if want == "entered" else want)
        if want != "entered":
            assert res.meta["margin"]["n_blocked"] == 1 and res.positions.empty
