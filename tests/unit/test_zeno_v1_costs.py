"""zeno_v1 cost cells (Costs section of the spec): spread bases S1/S2, the multiplier k on spread, commission
and stop slippage, commission 5 vs 10, slippage on stop fills only, and the matching propkit CostModel.
Canonical long of tests/unit/zeno_v1_testkit.py with its entry bar stopping it (low 2001). Research only."""
from __future__ import annotations

import numpy as np
import pytest

from propkit import zeno_v1 as z
from tests.unit.zeno_v1_testkit import Scenario, triggers, utc

STOP_BAR = (2006.0, 2006.0, 2001.0, 2002.0)


def full_stop(cell, side=1):
    sc = Scenario("2024-03-05 12:00")
    ti = sc.setup(side)
    sc.add(*(STOP_BAR if side > 0 else (1994.0, 1999.0, 1994.0, 1998.0)))
    sc.flat(4, sc.bid[-1][3])
    prep, res = sc.run(cell=cell, trend="long" if side > 0 else "short")
    return sc, ti, prep, res


@pytest.mark.parametrize("comm,k,entry,units,exit_price,net", [
    # k = 1: spread 0.20, commission 10/lot, slippage 0.05: R 4.70, 106 oz, 106 x -4.75 - 10.60
    (10.0, 1.0, 2006.20, 106.0, 2001.45, -514.10),
    # k = 1.5: spread 0.30 -> entry 2006.30, R 4.80, 500/4.80 = 104.2 -> 104 oz; slippage 0.075 -> 2001.425;
    # gross 104 x -4.875 = -507.00; commission 15 x 1.04 = 15.60; net -522.60
    (10.0, 1.5, 2006.30, 104.0, 2001.425, -522.60),
    # k = 2: spread 0.40 -> 2006.40, R 4.90, 102.04 -> 102 oz; exit 2001.40; 102 x -5.00 - 20 x 1.02 = -530.40
    (10.0, 2.0, 2006.40, 102.0, 2001.40, -530.40),
    # commission 5: 106 x -4.75 - 5 x 1.06 = -508.80
    (5.0, 1.0, 2006.20, 106.0, 2001.45, -508.80),
])
def test_multiplier_scales_spread_commission_and_slippage(comm, k, entry, units, exit_price, net):
    sc, ti, prep, res = full_stop(z.ZenoCell("evaluation", comm, "S1", k))
    p = res.positions.iloc[0]
    assert p["entry_price"] == pytest.approx(entry) and p["units_oz"] == units
    assert p["spread_entry"] == pytest.approx(0.20 * k)
    assert p["exit1_price"] == pytest.approx(exit_price)
    assert p["commission_usd"] == pytest.approx(comm * k * units / 100)
    assert p["slippage_usd"] == pytest.approx(0.05 * k * units)
    assert p["net_pnl_usd"] == pytest.approx(net)
    assert p["be_level"] == pytest.approx(entry + comm * k / 100)
    z.cell_equity(prep, res)


def test_s2_spread_base_in_the_engine():
    # S2 at 14:00 UTC: 0.18 -> entry 2006.18, R 4.68, 500 / 4.68 = 106.8 -> 106 oz; the data spread is ignored
    sc, ti, prep, res = full_stop(z.ZenoCell("evaluation", 10.0, "S2", 1.0))
    p = res.positions.iloc[0]
    assert p["entry_price"] == pytest.approx(2006.18) and p["spread_entry"] == pytest.approx(0.18)
    assert p["R_usd_per_oz"] == pytest.approx(4.68) and p["units_oz"] == 106.0
    # S2 x1.5: 0.27 -> entry 2006.27
    sc, ti, prep, res = full_stop(z.ZenoCell("evaluation", 10.0, "S2", 1.5))
    assert res.positions.iloc[0]["entry_price"] == pytest.approx(2006.27)


def test_s2_short_stop_includes_the_s2_entry_spread():
    # short under S2: stop = 1998 + 0.50 + 0.18 = 1998.68 (an ask level: bid + 0.18 at 14:15 UTC);
    # the entry bar's bid high 1999 -> ask high 1999.18 >= 1998.68: stopped at 1998.73
    sc, ti, prep, res = full_stop(z.ZenoCell("evaluation", 10.0, "S2", 1.0), side=-1)
    p = res.positions.iloc[0]
    assert p["stop_level"] == pytest.approx(1998.68) and p["R_usd_per_oz"] == pytest.approx(4.68)
    assert p["exit1_price"] == pytest.approx(1998.73)


def test_s2_rollover_spread_on_a_short_time_exit():
    # winter: 16:30 New York = 21:30 UTC, inside the 21:00-24:00 UTC rollover where S2 = 0.20. The short is
    # bought back at the ask open of the 21:30 bar = 1994.00 + 0.20 (S1 would use the data ask 1994.25).
    sc = Scenario("2024-03-05 12:00", spread=0.25)
    sc.setup(-1)
    sc.flat_until("2024-03-05 23:00", 1994.0)
    for base, exit_price in (("S2", 1994.20), ("S1", 1994.25)):
        prep, res = sc.run(cell=z.ZenoCell("evaluation", 10.0, base, 1.0), trend="short")
        p = res.positions.iloc[0]
        assert (p["exit1_reason"], p["exit1_time"]) == ("time", utc("2024-03-05 21:30"))
        assert p["exit1_price"] == pytest.approx(exit_price)


def test_slippage_applies_to_stop_fills_only():
    sc = Scenario("2024-03-05 12:00")
    sc.setup(1)
    sc.add(2006.0, 2007.0, 2005.0, 2006.5)
    sc.add(2006.5, 2026.0, 2006.5, 2025.5)               # tp1 2015.60 and tp2 2025.00 in one bar
    sc.flat(4, 2025.5)
    prep, res = sc.run()
    p = res.positions.iloc[0]
    assert p["outcome"] == "+3R" and p["slippage_usd"] == 0.0
    assert res.legs["exit_price"].tolist() == pytest.approx([2015.60, 2025.00])
    # a time exit has no slippage either (exit at the bid open)
    sc = Scenario("2024-03-05 12:00")
    sc.setup(1)
    sc.flat_until("2024-03-05 23:00", 2006.0)
    prep, res = sc.run()
    p = res.positions.iloc[0]
    assert p["exit1_reason"] == "time" and p["slippage_usd"] == 0.0 and p["exit1_price"] == 2006.0


def test_cost_model_for_cell():
    cm = z.cost_model_for_cell(z.ZenoCell("master", 5.0, "S2", 2.0))
    assert cm.commission_per_lot_round_trip == 10.0
    assert cm.spread_source == "bar" and cm.spread_multiplier == 1.0
    assert cm.slippage_per_side == 0.0 and cm.markup_per_side == 0.0
    assert cm.lot_size_oz == 100.0 and cm.swap_enabled is False


def test_cell_validation_and_labels():
    c = z.ZenoCell("evaluation", 10, "S1", 1.5)
    assert c.label == "evaluation/c10/S1/x1.5" and c.to_dict()["cost_mult"] == 1.5
    for bad in (dict(variant="funded"), dict(spread_base="S3"), dict(cost_mult=0.0), dict(commission_rt_per_lot=-1)):
        with pytest.raises(ValueError):
            z.ZenoCell(**bad)


def test_every_grid_cell_runs_and_agrees_with_propkit_equity():
    f = z.synthetic_m15_bidask(start=utc("2023-01-02 00:00"), n_bars=12_000, seed=11, price=1900.0,
                               vol_per_hour=0.004, spread=0.25)
    prep = z.prepare(f)
    assert len(prep.triggers()) > 50
    finals = {}
    for cell in z.grid_cells():
        res = z.simulate(prep, z.ZenoConfig(cell))
        equity, _ = z.cell_equity(prep, res)
        assert equity["balance"].iloc[-1] == pytest.approx(res.meta["final_balance_usd"], abs=1e-6)
        assert set(res.decisions["status"]) - {""} <= set(z.BLOCK_REASONS) | {"entered"}
        finals[cell.label] = res.meta["final_balance_usd"]
    # identical triggers in every cell; costs only move sizes, fills and the filters that depend on them
    n_trig = {len(triggers(z.simulate(prep, z.ZenoConfig(c)))) for c in z.grid_cells()[:3]}
    assert len(n_trig) == 1
    assert len(set(np.round(list(finals.values()), 6))) > 1


def _short_with_ask_spikes():
    # Canonical short (cell evaluation / 10 / S1 / x1): entry = bid open 1994.00, stop 1998.70 (ask), R 4.70,
    # 106 oz (53 + 53), tp1 1984.60, tp2 1975.20, breakeven 1994.00 - 0.10 = 1993.90 (all ask levels).
    sc = Scenario("2024-03-05 12:00")                      # trigger closes 14:00 UTC; entry bar opens 14:00
    ti = sc.setup(-1)
    sc.add(1994.0, 1994.5, 1993.5, 1994.0)                 # entry bar; ask = bid + 0.20
    # bar k: the data's ask high spikes to 1998.60 (spread 3.60 at the high, 0.20 at the open, 0.90 at the
    # close): below the stop 1998.70, so the position stays open
    k = sc.add(1994.0, 1995.0, 1993.0, 1994.0, ask=(1994.2, 1998.6, 1993.2, 1994.9))
    sc.add(1990.0, 1990.5, 1984.0, 1985.0)                 # ask low 1984.20 <= tp1 1984.60: 53 oz at 1984.60
    # bar r: the runner reaches tp2 1975.20 (ask low) while the ask high spikes to 1993.50 < breakeven 1993.90
    r = sc.add(1985.0, 1986.0, 1975.0, 1976.0, ask=(1985.2, 1993.5, 1975.2, 1976.2))
    sc.flat(4, 1976.0)
    prep, res = sc.run(trend="short")
    return sc, ti, k, r, prep, res


def test_the_prop_evaluator_marks_a_short_on_the_cells_ask_high_and_close():
    # RR-2 [SI-67]: D14 marks and closes a short on the ASK. propkit.equity marks shorts at bid + the bar's
    # OPEN spread unless it is given the cell's ask prices; under S1 the data's spread widens inside a bar.
    sc, ti, k, r, prep, res = _short_with_ask_spikes()
    assert res.legs[["leg", "exit_reason", "units", "exit_price"]].values.tolist() == [
        ["tp1", "target", 53.0, pytest.approx(1984.60)], ["runner", "target", 53.0, pytest.approx(1975.20)]]
    eq, _ = z.cell_equity(prep, res)
    t = eq["time"].to_numpy()
    kk, rr = (int(np.flatnonzero(t == sc.time(i))[0]) for i in (k, r))
    # balance after the entry commission 5 x 1.06 = 5.30: 99,994.70.
    # bar k, open short 106 oz from 1994.00: worst at the ask high 1998.60 -> 99,994.70 - 106 x 4.60 = 99,507.10
    # (bid high + open spread 1995.20 would give 99,867.50); close at the ask close 1994.90 -> 99,994.70 - 106 x
    # 0.90 = 99,899.30 (bid close + open spread 1994.20 would give 99,973.50)
    assert eq["balance"].iloc[kk] == pytest.approx(99_994.70)
    assert eq["equity_worst"].iloc[kk] == pytest.approx(99_507.10)
    assert eq["equity_close"].iloc[kk] == pytest.approx(99_899.30)
    # bar r: the runner (53 oz) closes at tp2 inside the bar it was exposed in; its worst is at the ask high
    # 1993.50: 53 x (1994.00 - 1993.50) = +26.50, less the exit commission 2.65 -> balance before + 23.85
    # (bid high + open spread 1986.20 would give + 413.40 - 2.65)
    assert eq["equity_worst"].iloc[rr] == pytest.approx(eq["balance"].iloc[rr - 1] + 23.85)
    # the cell's ask is what the engine's own stop and target checks use (ask_side)
    a = z.ask_side(prep.frame, "S1", 1.0)
    assert (a["ask_high"][kk], a["ask_close"][kk], a["ask_high"][rr]) == pytest.approx((1998.6, 1994.9, 1993.5))


def test_the_ask_marks_change_nothing_where_the_ask_is_bid_plus_the_open_spread():
    # Under S2 the ask is bid + k x S2 at every price of a bar, so the cell's ask marks equal propkit's own
    # bid + spread marks exactly; propkit.equity without ask_prices keeps its old marks (default unchanged).
    from propkit.equity import equity_from_trades
    sc, ti, k, r, prep, res = _short_with_ask_spikes()
    for cell in (z.ZenoCell("evaluation", 10.0, "S2", 1.0), z.ZenoCell("evaluation", 10.0, "S2", 1.5)):
        res2 = z.simulate(prep, z.ZenoConfig(cell))
        eq, _ = z.cell_equity(prep, res2)
        old, _ = equity_from_trades(z.to_propkit_bars(prep.frame, "S2", cell.cost_mult), res2.legs, 100_000.0,
                                    z.cost_model_for_cell(cell), price_tolerance=None)
        assert np.allclose(eq.to_numpy(dtype=float), old.to_numpy(dtype=float), rtol=0, atol=1e-9)
    old, _ = equity_from_trades(z.to_propkit_bars(prep.frame), res.legs, 100_000.0,
                                z.cost_model_for_cell(z.ZenoCell("evaluation", 10.0, "S1", 1.0)), price_tolerance=None)
    kk = int(np.flatnonzero(old["time"].to_numpy() == sc.time(k))[0])
    assert old["equity_worst"].iloc[kk] == pytest.approx(99_867.50)       # bid high 1995.00 + open spread 0.20
