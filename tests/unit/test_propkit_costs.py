"""Tests for propkit/costs.py: CostModel fills, commission, swap_for_night, multiplied, flat rate. Research only."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from propkit.costs import DUKASCOPY_SPREAD_MEDIAN, DUKASCOPY_SPREAD_P90, CostModel


def _bars(spread=None) -> pd.DataFrame:
    df = pd.DataFrame({"time": [0, 3600, 7200], "open": [2000.0] * 3, "high": [2001.0] * 3,
                       "low": [1999.0] * 3, "close": [2000.5] * 3})
    if spread is not None:
        df["spread"] = spread
    return df


FULL = CostModel(markup_per_side=0.05, slippage_per_side=0.10, commission_per_lot_round_trip=7.0)


def test_documented_defaults():
    c = CostModel()
    assert c.spread_source == "bar" and c.fixed_spread == 0.34 == DUKASCOPY_SPREAD_MEDIAN
    assert DUKASCOPY_SPREAD_P90 == 0.68
    assert (c.markup_per_side, c.slippage_per_side, c.commission_per_lot_round_trip) == (0.0, 0.0, 0.0)
    assert c.lot_size_oz == 100.0 and c.spread_multiplier == 1.0
    assert (c.swap_long, c.swap_short, c.swap_unit, c.swap_day_count) == (6.0, 2.0, "pct_per_year", 360.0)
    assert c.triple_swap_weekday == 2 and c.rollover_hour_ny == 17
    assert c.swap_enabled is True and c.flat_rate_per_side is None and not c.is_flat
    assert "[ASSUMPTION]" in CostModel.__doc__ and "positive = you pay" in CostModel.__doc__


def test_frozen():
    c = CostModel()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.fixed_spread = 1.0
    assert hash(c) == hash(CostModel())


def test_fills_known_answers():
    assert FULL.buy_fill(2000.0, 0.30) == pytest.approx(2000.45, abs=1e-9)    # ask + markup + slippage
    assert FULL.sell_fill(2000.0, 0.30) == pytest.approx(1999.85, abs=1e-9)   # bid - markup - slippage
    assert FULL.buy_fill(2000.0) == pytest.approx(2000.0 + 0.34 + 0.15, abs=1e-9)   # None = fixed spread
    assert isinstance(FULL.buy_fill(2000.0, 0.3), float)
    bids = np.array([2000.0, 2100.0])
    spreads = np.array([0.30, 0.50])
    assert np.allclose(FULL.buy_fill(bids, spreads), [2000.45, 2100.65])
    assert np.allclose(FULL.sell_fill(bids, spreads), [1999.85, 2099.85])
    plain = CostModel()
    assert plain.buy_fill(2000.0, 0.3) == pytest.approx(2000.3) and plain.sell_fill(2000.0, 0.3) == 2000.0


@pytest.mark.parametrize("bid, spread", [(float("nan"), 0.3), (0.0, 0.3), (-1.0, 0.3), (2000.0, -0.1),
                                         (2000.0, float("inf")), ("2000", 0.3), (True, 0.3)])
def test_fill_inputs_checked(bid, spread):
    with pytest.raises(ValueError):
        FULL.buy_fill(bid, spread)
    with pytest.raises(ValueError):
        FULL.sell_fill(bid, spread)


def test_sell_fill_cannot_go_non_positive():
    with pytest.raises(ValueError, match="<= 0"):
        CostModel(markup_per_side=5.0).sell_fill(4.0, 0.0)


def test_commission_per_lot():
    assert FULL.commission(150.0) == pytest.approx(10.5)            # 1.5 lots x 7 USD round trip
    assert FULL.fill_commission(150.0) == pytest.approx(5.25)       # half on each fill
    assert FULL.commission(150.0, 2000.0, 2100.0) == pytest.approx(10.5)   # prices ignored
    assert np.allclose(FULL.commission(np.array([100.0, 50.0])), [7.0, 3.5])
    assert CostModel().commission(100.0) == 0.0
    assert CostModel(commission_per_lot_round_trip=7.0, lot_size_oz=10.0).commission(100.0) == pytest.approx(70.0)
    with pytest.raises(ValueError):
        FULL.commission(-1.0)


def test_flat_rate_replaces_spread_markup_slippage_commission():
    f = CostModel(flat_rate_per_side=0.0003, markup_per_side=0.05, slippage_per_side=0.1,
                  commission_per_lot_round_trip=7.0, fixed_spread=0.5)
    assert f.is_flat and f.effective_fixed_spread == 0.0
    assert f.buy_fill(2000.0, 0.3) == 2000.0 and f.sell_fill(2000.0, 0.3) == 2000.0
    # a long round trip, 100 oz, in at 2000 and out at 2010: 0.0003 x notional at each fill
    assert f.fill_commission(100.0, 2000.0) == pytest.approx(60.0)
    assert f.fill_commission(100.0, 2010.0) == pytest.approx(60.3)
    assert f.commission(100.0, 2000.0, 2010.0) == pytest.approx(120.3)
    assert np.array_equal(f.bar_spreads(_bars([0.3, 0.4, 0.5])), np.zeros(3))
    assert f.spread_source_used(_bars([0.3, 0.4, 0.5])) == "none (flat rate)"
    with pytest.raises(ValueError, match="price"):
        f.fill_commission(100.0)
    with pytest.raises(ValueError, match="entry_price"):
        f.commission(100.0)
    assert f.swap_for_night(1, 100.0, 2000.0, 0) == pytest.approx(-100.0 * 2000.0 * 0.06 / 360)  # swap stays
    assert CostModel(flat_rate_per_side=0.0003, swap_enabled=False).swap_for_night(1, 100.0, 2000.0, 0) == 0.0


def test_swap_pct_per_year_known_answers():
    c = CostModel()                                                # 6 %/yr long, 2 %/yr short, 360 days
    assert c.swap_for_night(1, 100.0, 2000.0, 0) == pytest.approx(-33.333333333333336)   # 0.06 x 200000 / 360
    assert c.swap_for_night(1, 100.0, 2000.0, 2) == pytest.approx(-100.0)                # Wednesday x3
    assert c.swap_for_night(-1, 50.0, 1800.0, 4) == pytest.approx(-5.0)                  # 0.02 x 90000 / 360
    assert c.swap_for_night(-1, 50.0, 1800.0, 2) == pytest.approx(-15.0)
    credit = CostModel(swap_short=-1.0)
    assert credit.swap_for_night(-1, 50.0, 1800.0, 1) == pytest.approx(2.5)                # credit: positive
    assert CostModel(swap_day_count=365).swap_for_night(1, 100.0, 2000.0, 1) == pytest.approx(-12000 / 365)
    out = c.swap_for_night(np.array([1, -1, 1]), np.array([100.0, 50.0, 10.0]), 2000.0, np.array([2, 2, 3]))
    assert np.allclose(out, [-100.0, -2000 * 50 * 0.02 / 360 * 3, -2000 * 10 * 0.06 / 360])
    assert isinstance(c.swap_for_night(1, 1.0, 2000.0, 0), float)


def test_swap_usd_per_lot_and_triple_day_options():
    c = CostModel(swap_unit="usd_per_lot_per_night", swap_long=25.0, swap_short=-4.0)
    assert c.swap_for_night(1, 50.0, 2000.0, 1) == pytest.approx(-12.5)
    assert c.swap_for_night(1, 50.0, 2000.0, 2) == pytest.approx(-37.5)
    assert c.swap_for_night(-1, 50.0, 2000.0, 1) == pytest.approx(2.0)
    friday = CostModel(triple_swap_weekday=4)
    assert friday.swap_nights(2) == 1 and friday.swap_nights(4) == 3
    assert friday.swap_for_night(1, 100.0, 2000.0, 4) == pytest.approx(-100.0)
    never = CostModel(triple_swap_weekday=None)
    assert never.swap_nights(np.arange(7)).tolist() == [1] * 7
    assert CostModel().swap_nights(np.arange(7)).tolist() == [1, 1, 3, 1, 1, 1, 1]
    assert CostModel(swap_enabled=False).swap_for_night(1, 100.0, 2000.0, 2) == 0.0


@pytest.mark.parametrize("args", [(0, 1.0, 2000.0, 0), (2, 1.0, 2000.0, 0), (1, -1.0, 2000.0, 0),
                                  (1, 1.0, 0.0, 0), (1, 1.0, float("nan"), 0), (1, 1.0, 2000.0, 7),
                                  (1, 1.0, 2000.0, 2.5), (1, 1.0, 2000.0, -1)])
def test_swap_inputs_checked(args):
    with pytest.raises(ValueError):
        CostModel().swap_for_night(*args)


def test_bar_spreads_sources_and_multiplier():
    c = CostModel()
    bars = _bars([0.30, 0.40, 0.50])
    assert np.allclose(c.bar_spreads(bars), [0.30, 0.40, 0.50]) and c.spread_source_used(bars) == "bar"
    assert np.allclose(c.bar_spreads(_bars()), [0.34] * 3) and c.spread_source_used(_bars()) == "fixed"
    fixed = CostModel(spread_source="fixed", fixed_spread=0.68)
    assert np.allclose(fixed.bar_spreads(bars), [0.68] * 3) and fixed.spread_source_used(bars) == "fixed"
    wide = CostModel(spread_multiplier=1.5)
    assert np.allclose(wide.bar_spreads(bars), [0.45, 0.60, 0.75])
    assert np.allclose(wide.bar_spreads(_bars()), [0.51] * 3) and wide.effective_fixed_spread == pytest.approx(0.51)
    with pytest.raises(ValueError, match="missing or infinite"):
        c.bar_spreads(_bars([0.3, np.nan, 0.3]))
    with pytest.raises(ValueError, match="negative"):
        c.bar_spreads(_bars([0.3, -0.1, 0.3]))
    with pytest.raises(ValueError):
        c.bar_spreads(np.array([0.3]))


def test_multiplied_scales_every_cost():
    base = CostModel(markup_per_side=0.05, slippage_per_side=0.10, commission_per_lot_round_trip=7.0,
                     swap_long=6.0, swap_short=-1.0)
    k2 = base.multiplied(2.0)
    assert k2.spread_multiplier == 2.0 and k2.fixed_spread == base.fixed_spread
    assert (k2.markup_per_side, k2.slippage_per_side, k2.commission_per_lot_round_trip) == (0.1, 0.2, 14.0)
    assert k2.swap_long == 12.0 and k2.swap_short == -1.0           # a credit is not enlarged
    assert (k2.lot_size_oz, k2.swap_day_count, k2.triple_swap_weekday, k2.rollover_hour_ny) == (100.0, 360.0, 2, 17)
    bars = _bars([0.30, 0.30, 0.30])
    s = k2.bar_spreads(bars)
    assert k2.buy_fill(2000.0, s[0]) == pytest.approx(2000.0 + 0.6 + 0.1 + 0.2)
    assert k2.sell_fill(2000.0, s[0]) == pytest.approx(2000.0 - 0.1 - 0.2)
    assert k2.commission(100.0) == pytest.approx(14.0)
    assert k2.swap_for_night(1, 100.0, 2000.0, 0) == pytest.approx(2 * base.swap_for_night(1, 100.0, 2000.0, 0))
    assert base.multiplied(1.0) == base
    assert CostModel(flat_rate_per_side=0.0003).multiplied(1.5).flat_rate_per_side == pytest.approx(0.00045)
    k15 = CostModel().multiplied(1.5)
    assert k15.effective_fixed_spread == pytest.approx(0.51) and k15.swap_short == pytest.approx(3.0)
    zero = base.multiplied(0.0)
    assert zero.buy_fill(2000.0, zero.bar_spreads(bars)[0]) == 2000.0 and zero.commission(100.0) == 0.0
    for bad in (-1.0, float("nan"), "2", True):
        with pytest.raises(ValueError):
            base.multiplied(bad)


@pytest.mark.parametrize("kwargs", [
    {"spread_source": "tick"}, {"fixed_spread": -0.1}, {"spread_multiplier": -1.0}, {"markup_per_side": -0.01},
    {"slippage_per_side": float("nan")}, {"commission_per_lot_round_trip": -7.0}, {"lot_size_oz": 0.0},
    {"swap_unit": "points"}, {"swap_day_count": 0.0}, {"triple_swap_weekday": 7}, {"triple_swap_weekday": 2.0},
    {"rollover_hour_ny": 2}, {"rollover_hour_ny": 24}, {"rollover_hour_ny": 17.0}, {"swap_enabled": "yes"},
    {"flat_rate_per_side": -0.0003}, {"flat_rate_per_side": 1.0}, {"swap_long": "6"}, {"swap_long": float("inf")},
])
def test_invalid_settings_raise(kwargs):
    with pytest.raises(ValueError):
        CostModel(**kwargs)


def test_numbers_are_normalised_to_float():
    c = CostModel(fixed_spread=1, lot_size_oz=np.int64(100), swap_long=np.float32(6.5), triple_swap_weekday=np.int64(4))
    assert isinstance(c.fixed_spread, float) and isinstance(c.lot_size_oz, float) and isinstance(c.swap_long, float)
    assert isinstance(c.triple_swap_weekday, int) and c.triple_swap_weekday == 4


def test_dict_round_trip_and_unknown_keys():
    c = CostModel(markup_per_side=0.05, triple_swap_weekday=None, flat_rate_per_side=0.0003)
    d = c.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert CostModel.from_dict(json.loads(json.dumps(d))) == c
    assert CostModel.from_dict({}) == CostModel()
    with pytest.raises(ValueError, match="unknown cost setting"):
        CostModel.from_dict({"spread": 0.3})
    with pytest.raises(ValueError):
        CostModel.from_dict([("fixed_spread", 0.3)])


def _utc(text: str) -> int:
    return int(dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp())


def test_held_rollovers_open_strictly_before_closed_at_or_after():
    c = CostModel()
    wed = _utc("2024-07-17 21:00:00")                     # 17:00 New York (EDT), a Wednesday: triple
    r, n = c.held_rollovers(_utc("2024-07-17 10:00:00"), _utc("2024-07-18 10:00:00"))
    assert r.tolist() == [wed] and n.tolist() == [3]
    assert c.held_rollovers(_utc("2024-07-17 10:00:00"), wed - 1)[0].size == 0     # closed before 17:00 NY
    assert c.held_rollovers(_utc("2024-07-17 10:00:00"), wed)[0].tolist() == [wed]  # closed at 17:00 NY
    assert c.held_rollovers(wed, wed + 3600)[0].size == 0                           # opened at 17:00 NY
    r, n = c.held_rollovers(_utc("2024-07-19 10:00:00"), _utc("2024-07-22 10:00:00"))   # over a weekend
    assert r.tolist() == [_utc("2024-07-19 21:00:00")] and n.tolist() == [1]
    r, n = c.held_rollovers(_utc("2024-01-15 00:00:00"), _utc("2024-01-22 00:00:00"))   # winter week
    assert r.tolist() == [_utc(f"2024-01-{d} 22:00:00") for d in (15, 16, 17, 18, 19)]
    assert n.tolist() == [1, 1, 3, 1, 1] and n.sum() == 7                              # 7 nights a week
    fri = CostModel(triple_swap_weekday=4, rollover_hour_ny=16)
    r, n = fri.held_rollovers(_utc("2024-01-15 00:00:00"), _utc("2024-01-22 00:00:00"))
    assert r[0] == _utc("2024-01-15 21:00:00") and n.tolist() == [1, 1, 1, 1, 3]
    assert c.held_rollovers(wed, wed)[0].size == 0
    with pytest.raises(ValueError, match="before entry_time"):
        c.held_rollovers(wed, wed - 1)
    with pytest.raises(ValueError):
        c.held_rollovers(1.5, wed)
