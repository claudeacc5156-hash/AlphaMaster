"""Tests for propkit/pullback.py: the parametric pullback TRADES generator and PullbackSpec.
Synthetic data only (no market data until zeno's spec arrives). Research only."""
from __future__ import annotations

import dataclasses
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from propkit import bars as pb
from propkit import calendar as cal
from propkit import pullback as pl
from propkit.costs import CostModel
from propkit.pullback import PullbackSpec, compute_signals, floor_to_lot_step, generate_trades, \
    generate_trades_detailed

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "propkit" / "pullback.py"
EXAMPLE = ROOT / "propkit" / "examples" / "pullback_spec_example.json"
README = ROOT / "propkit" / "METHODS.md"  # the fields table (from README_pullback_fields.md, then README.md)
C0 = 100_000.0
MONDAY = 1704672000                 # 2024-01-08 00:00 UTC, a Monday (winter: rollover 22:00 UTC)

# hand-example costs: spread 0.30 per bar (in the bars), markup 0.10 + slippage 0.05 per side,
# commission 7 USD per 100-oz lot round trip = 0.07 USD/oz, no swap
HAND_COSTS = CostModel(markup_per_side=0.10, slippage_per_side=0.05, commission_per_lot_round_trip=7.0,
                       swap_enabled=False)
HAND_SPEC = PullbackSpec(name="hand", placeholder=False, direction="both", trend_ema_period=10,
                         trend_slope_bars=3, trend_require_close_side=True, pullback_mode="atr_from_swing",
                         swing_lookback_bars=20, min_depth_atr=1.0, max_depth_atr=3.0,
                         trigger="break_prev_extreme", stop_mode="swing", stop_buffer_atr=0.25, atr_period=14,
                         exit_mode="fixed_r", target_r=2.0, risk_pct=0.01, warmup_bars=20)

# hand numbers (see _hand_bars): ATR = 2 exactly (every true range is 2), pullback low 126 after the
# swing high 130.5, signal at bar 33's close (close 128.5 > previous high 128), entry at bar 34's open
ENTRY = 128.5 + 0.30 + 0.10 + 0.05            # ask + markup + slippage = 128.95
STOP = 126.0 - 0.25 * 2.0                     # 125.5 (bid level)
STOP_FILL = STOP - 0.15                       # 125.35
TARGET = ENTRY + 2.0 * (ENTRY - STOP)         # 135.85 (bid level)
LOSS_PER_OZ = ENTRY - STOP_FILL + 0.07        # 3.67
UNITS = 272.0                                 # floor(1000 / 3.67) = floor(272.48)
RISK_USD = UNITS * (ENTRY - STOP_FILL) + UNITS * 0.07     # 998.24
COMM = UNITS * 0.07                           # 19.04 (9.52 per fill)


def _hand_bars(after: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Bars 0..29 rise 1 per bar (true range 2), bars 30..32 pull back 1 per bar (true range 2), bar 33
    closes above bar 32's high, bar 34 (the entry bar) opens at 128.5; `after` = bars 35.. (o, h, l, c)."""
    rows = []
    for i in range(30):
        o, c = 100.0 + i, 101.0 + i
        rows.append((o, c + 0.5, o - 0.5, c))
    for i in range(3):
        o = 130.0 - i
        rows.append((o, o, o - 2.0, o - 1.0))
    rows.append((127.0, 129.0, 127.0, 128.5))           # bar 33: the trigger
    rows.append((128.5, 130.0, 128.0, 129.8))           # bar 34: entry bar (low 128 > stop)
    rows.extend(after)
    o, h, lo, c = (np.array(x, dtype=float) for x in zip(*rows))
    return pd.DataFrame({"time": MONDAY + 3600 * np.arange(len(rows)), "open": o, "high": h, "low": lo,
                         "close": c, "spread": np.full(len(rows), 0.30)})


UP_AFTER = [(129.8, 131.8, 129.8, 131.5), (131.5, 133.5, 131.5, 133.2), (133.2, 135.2, 133.2, 135.0),
            (135.0, 137.0, 135.0, 136.5), (136.5, 138.5, 136.5, 138.0), (138.0, 140.0, 138.0, 139.5)]


def cents(a, b) -> bool:
    return abs(float(a) - float(b)) < 1e-6


def _synth(n: int = 2500, seed: int = 7, spread: float | None = 0.34, start: int = MONDAY) -> pd.DataFrame:
    return pb.synthetic_bars(start, n, bar_seconds=3600, seed=seed, vol_per_hour=0.003, spread=spread)


BASE = dict(name="test", placeholder=False, trend_ema_period=30, trend_slope_bars=3, ema_pullback_period=10,
            pullback_lookback_bars=4, swing_lookback_bars=12, atr_period=10, min_depth_atr=0.5,
            max_depth_atr=None, risk_pct=0.005)
SPEC_A = PullbackSpec(**BASE, pullback_mode="ema_touch", trigger="close_back_over_ema", stop_mode="swing",
                      exit_mode="fixed_r", target_r=1.5)
SPEC_B = PullbackSpec(**BASE, pullback_mode="atr_from_swing", trigger="break_prev_extreme", stop_mode="atr",
                      stop_atr_mult=1.2, exit_mode="trail_ema", trail_ema_period=8)
SPEC_C = PullbackSpec(**BASE, pullback_mode="ema_touch", trigger="break_prev_extreme", stop_mode="swing",
                      exit_mode="time", time_exit_bars=6, max_open_positions=2, max_trades_per_day=3,
                      daily_stop_losses=2, daily_stop_pct=0.004, sessions=("london", "newyork"),
                      session_hours_utc=((22, 2),), max_spread_usd=0.45,
                      news_blackouts_utc=(("2024-01-12T13:00:00Z", "2024-01-12T15:00:00Z"),
                                          ("2024-02-02T13:00:00Z", "2024-02-02T14:30:00Z")),
                      news_flatten=True, entry_after_gap=False)
COSTS = CostModel(markup_per_side=0.05, slippage_per_side=0.02, commission_per_lot_round_trip=7.0)


# ======================================================================== hand-built path

def test_hand_path_signal_entry_target_known_to_the_cent():
    bars = _hand_bars(UP_AFTER)
    sig = compute_signals(bars, HAND_SPEC)
    assert np.allclose(sig["atr"].to_numpy()[13:], 2.0)
    # bar 33 is the trigger; bar 34 (close 129.8 > high 129, still below the swing high 130.5) signals
    # again, but the position is open and max_open_positions is 1; bars 0..32 never signal
    assert list(np.flatnonzero(sig["long_signal"].to_numpy())) == [33, 34]
    assert not sig["short_signal"].any()
    assert sig["long_extreme"][33] == 126.0 and sig["long_depth_atr"][33] == pytest.approx(2.25)
    trades, dec = generate_trades_detailed(bars, HAND_SPEC, C0, HAND_COSTS, lot_step_oz=1.0)
    assert len(trades) == 1 and list(dec["status"]) == ["entered", "max_open_positions"]
    t = trades.iloc[0]
    assert t["side"] == 1 and t["units"] == UNITS
    assert t["entry_time"] == MONDAY + 34 * 3600                  # bar 34's open
    assert cents(t["entry_price"], ENTRY) and cents(t["stop_price"], STOP)
    assert t["exit_reason"] == "target"
    assert t["exit_time"] == MONDAY + 38 * 3600 + 3599            # bar 38 (high 137 >= 135.85), last second
    assert cents(t["exit_price"], TARGET - 0.15)                  # 135.70
    assert cents(t["risk_usd"], RISK_USD) and cents(RISK_USD, 998.24)
    assert cents(t["commission_usd"], COMM) and t["swap_usd"] == 0.0
    assert cents(t["pnl_usd"], UNITS * (135.70 - 128.95) - 19.04)   # 1816.96
    assert cents(t["pnl_usd"], 1816.96)


def test_hand_path_stop_hit_inside_a_bar_is_minus_one_r():
    bars = _hand_bars([(129.8, 129.8, 127.8, 128.0), (128.0, 128.0, 125.0, 125.6), (125.6, 126.0, 125.2, 125.8)])
    trades = generate_trades(bars, HAND_SPEC, C0, HAND_COSTS)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "stop" and t["exit_time"] == MONDAY + 36 * 3600 + 3599
    assert cents(t["exit_price"], STOP_FILL)
    assert cents(t["pnl_usd"], -RISK_USD) and cents(t["pnl_usd"], -998.24)


def test_gap_through_stop_fills_at_the_open():
    bars = _hand_bars([(125.0, 125.4, 124.5, 125.2), (125.2, 125.5, 124.8, 125.0)])
    t = generate_trades(bars, HAND_SPEC, C0, HAND_COSTS).iloc[0]
    assert t["exit_reason"] == "stop"
    assert t["exit_time"] == MONDAY + 35 * 3600                   # at the open of the gap bar
    assert cents(t["exit_price"], 125.0 - 0.15)                   # bid open - markup - slippage, not the stop
    assert cents(t["pnl_usd"], UNITS * (124.85 - ENTRY) - COMM)   # -1134.24, worse than -1R
    assert cents(t["pnl_usd"], -1134.24)


def test_gap_through_target_fills_at_the_open():
    bars = _hand_bars([(136.5, 137.0, 136.0, 136.8), (136.8, 137.0, 136.5, 136.9)])
    t = generate_trades(bars, HAND_SPEC, C0, HAND_COSTS).iloc[0]
    assert t["exit_reason"] == "target" and t["exit_time"] == MONDAY + 35 * 3600
    assert cents(t["exit_price"], 136.5 - 0.15)


def test_stop_first_when_stop_and_target_inside_one_bar():
    bars = _hand_bars([(129.8, 136.0, 125.0, 130.0), (130.0, 130.5, 129.5, 130.2)])
    t = generate_trades(bars, HAND_SPEC, C0, HAND_COSTS).iloc[0]
    assert 136.0 >= TARGET and 125.0 <= STOP                      # both levels are inside bar 35
    assert t["exit_reason"] == "stop" and cents(t["exit_price"], STOP_FILL)
    assert t["exit_time"] == MONDAY + 35 * 3600 + 3599
    # the same rule for a short (mirrored bars): stop first
    m = _mirror(bars, 200.0, 0.30)
    s = generate_trades(m, HAND_SPEC, C0, HAND_COSTS).iloc[0]
    assert s["side"] == -1 and s["exit_reason"] == "stop" and cents(s["exit_price"], 400.0 - STOP_FILL)


def test_stop_live_in_the_entry_bar():
    rows = _hand_bars(UP_AFTER)
    rows.loc[34, "low"] = 125.0                                   # the entry bar itself trades through the stop
    t = generate_trades(rows, HAND_SPEC, C0, HAND_COSTS).iloc[0]
    assert t["entry_time"] == MONDAY + 34 * 3600 and t["exit_reason"] == "stop"
    assert t["exit_time"] == MONDAY + 34 * 3600 + 3599 and cents(t["exit_price"], STOP_FILL)


def test_short_hand_numbers_with_ask_side_stop_and_gap():
    # mirror the hand path around 200 with the spread kept: bid' = 400 - 0.30 - bid, so ask' = 400 - bid
    bars = _mirror(_hand_bars([(125.0, 125.4, 124.5, 125.2), (125.2, 125.5, 124.8, 125.0)]), 200.0, 0.30)
    t = generate_trades(bars, HAND_SPEC, C0, HAND_COSTS).iloc[0]
    assert t["side"] == -1 and t["units"] == UNITS
    o34 = 400.0 - 0.30 - 128.5                                   # bid open of the entry bar = 271.2
    assert cents(t["entry_price"], o34 - 0.15)                   # short sells at bid - markup - slippage
    swing_high = 400.0 - 0.30 - 126.0                            # bid high of the pullback = 273.7
    assert cents(t["stop_price"], swing_high + 0.25 * 2.0 + 0.30)  # ask level: + buffer + spread = 274.5
    o35 = 400.0 - 0.30 - 125.0                                   # gap bar bid open 274.7; ask 275.0 >= stop
    assert t["exit_reason"] == "stop" and t["exit_time"] == MONDAY + 35 * 3600
    assert cents(t["exit_price"], o35 + 0.30 + 0.15)             # ask open + markup + slippage = 275.15
    assert cents(t["pnl_usd"], -UNITS * (275.15 - 271.05) - COMM)


def test_open_beyond_stop_skips_the_entry():
    rows = _hand_bars(UP_AFTER)
    rows.loc[34, ["open", "low", "close"]] = [125.4, 125.0, 128.0]  # opens below the stop 125.5, no new signal
    trades, dec = generate_trades_detailed(rows, HAND_SPEC, C0, HAND_COSTS)
    assert len(trades) == 0 and list(dec["status"]) == ["open_beyond_stop"]


# ======================================================================== look-ahead

def _cut_check(bars: pd.DataFrame, spec: PullbackSpec, cuts) -> int:
    full, dfull = generate_trades_detailed(bars, spec, C0, COSTS)
    sfull = compute_signals(bars, spec)
    times = bars["time"].to_numpy()
    checked = 0
    for k in cuts:
        pd.testing.assert_frame_equal(compute_signals(bars.iloc[:k], spec), sfull.iloc[:k], check_exact=True)
        part, dpart = generate_trades_detailed(bars.iloc[:k], spec, C0, COSTS)
        cut = times[k]
        entered = full[full["entry_time"] < cut].reset_index(drop=True)
        assert len(part) == len(entered), k
        entry_cols = ["trade_id", "side", "units", "entry_time", "entry_price", "stop_price", "risk_usd"]
        pd.testing.assert_frame_equal(part[entry_cols], entered[entry_cols], check_exact=True)
        done = part["exit_reason"] != "end_of_data"
        pd.testing.assert_frame_equal(part[done].reset_index(drop=True), entered[done.to_numpy()].reset_index(drop=True),
                                      check_exact=True)
        assert (part.loc[~done, "exit_time"] == times[k - 1] + 3600).all()
        # every signal decided at bars <= k-2 is resolved identically; bar k-1's signal has no next bar
        dp = dpart[dpart["signal_bar"] <= k - 2].reset_index(drop=True)
        df_ = dfull[dfull["signal_bar"] <= k - 2].reset_index(drop=True)
        pd.testing.assert_frame_equal(dp, df_, check_exact=True)
        assert (dpart.loc[dpart["signal_bar"] == k - 1, "status"] == "no_next_bar").all()
        checked += len(part)
    return checked


@pytest.mark.parametrize("spec", [SPEC_A, SPEC_B, SPEC_C], ids=["touch-ema-swing-R", "swing-break-atr-trail",
                                                                "filters-time-exit"])
def test_truncation_never_changes_past_signals_or_entries(spec):
    bars = _synth(1600, seed=21)
    full = generate_trades(bars, spec, C0, COSTS)
    assert len(full) >= 15, "the synthetic path should produce trades for this test to mean anything"
    cuts = list(range(150, 1600, 29)) + [1599, 1600 - 2]
    checked = _cut_check(bars, spec, cuts)
    assert checked > 100


def test_changing_any_later_bar_never_changes_earlier_entries():
    bars = _synth(1200, seed=4)
    base = generate_trades(bars, SPEC_A, C0, COSTS)
    rng = np.random.default_rng(9)
    for k in (300, 555, 800, 1100):
        moved = bars.copy()
        f = np.exp(rng.normal(0, 0.03, len(bars) - k))
        for col in ("open", "high", "low", "close"):
            moved.loc[k:, col] = moved.loc[k:, col].to_numpy() * f
        moved.loc[k:, "spread"] = 0.9
        out = generate_trades(moved, SPEC_A, C0, COSTS)
        cut = bars["time"].iloc[k]
        a = base[base["entry_time"] < cut].reset_index(drop=True)
        b = out[out["entry_time"] < cut].reset_index(drop=True)
        cols = ["trade_id", "side", "units", "entry_time", "entry_price", "stop_price", "risk_usd"]
        pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=True)
        closed = a["exit_time"] < cut
        pd.testing.assert_frame_equal(a[closed], b[closed.to_numpy()], check_exact=True)


# ======================================================================== sizing

def test_size_times_stop_distance_is_the_risk_within_one_lot_step():
    bars = _synth(3000, seed=2)
    for step in (1.0, 10.0):
        for spec in (SPEC_A, SPEC_B):
            tr = generate_trades(bars, spec, C0, COSTS, lot_step_oz=step).sort_values("trade_id")
            assert len(tr) > 20
            adj = COSTS.markup_per_side + COSTS.slippage_per_side
            for _, t in tr.iterrows():
                closed = tr[tr["exit_time"] <= t["entry_time"]]
                balance = C0 + closed["pnl_usd"].sum()
                budget = spec.risk_pct * balance
                stop_fill = t["stop_price"] - adj if t["side"] > 0 else t["stop_price"] + adj
                per_oz = abs(t["entry_price"] - stop_fill) + 7.0 / 100.0
                assert t["risk_usd"] == pytest.approx(t["units"] * per_oz, rel=1e-12)
                assert t["units"] * per_oz <= budget + 1e-6
                assert (t["units"] + step) * per_oz > budget
                assert t["units"] / step == pytest.approx(round(t["units"] / step), abs=1e-9)
                if t["exit_reason"] == "stop" and t["exit_time"] % 3600 == 3599:
                    assert t["pnl_usd"] - t["swap_usd"] == pytest.approx(-t["risk_usd"], abs=1e-6)  # -1R


def test_size_below_one_lot_step_is_skipped():
    trades, dec = generate_trades_detailed(_hand_bars(UP_AFTER), HAND_SPEC, 1000.0, HAND_COSTS, lot_step_oz=10.0)
    # budget 10 USD / 3.67 USD per oz = 2.7 oz < one 10-oz step
    assert len(trades) == 0 and list(dec["status"]) == ["size_below_lot_step"] * 2   # bars 33 and 34


def test_floor_to_lot_step():
    assert floor_to_lot_step(272.48, 1.0) == 272.0
    assert floor_to_lot_step(0.3, 0.1) == 0.3                    # 0.3 / 0.1 = 2.9999999999999996
    assert floor_to_lot_step(2.999, 1.0) == 2.0
    assert floor_to_lot_step(0.99, 1.0) == 0.0
    assert floor_to_lot_step(105.0, 10.0) == 100.0
    with pytest.raises(ValueError):
        floor_to_lot_step(1.0, 0.0)
    with pytest.raises(ValueError):
        floor_to_lot_step(float("nan"), 1.0)


# ======================================================================== filters

def _entry_bars(bars, trades):
    return np.searchsorted(bars["time"].to_numpy(), trades["entry_time"].to_numpy(), side="right") - 1


def test_session_filter_blocks_entries_outside():
    bars = _synth(3000, seed=5)
    free = generate_trades(bars, SPEC_A, C0, COSTS)
    assert (~cal.session_mask(free["entry_time"].to_numpy(), "london")).any()
    spec = dataclasses.replace(SPEC_A, sessions=("london",))
    tr, dec = generate_trades_detailed(bars, spec, C0, COSTS)
    assert len(tr) > 5 and cal.session_mask(tr["entry_time"].to_numpy(), "london").all()
    assert (dec["status"] == "outside_session").any()
    spec_h = dataclasses.replace(SPEC_A, session_hours_utc=((22, 2),))
    trh = generate_trades(bars, spec_h, C0, COSTS)
    hours = (trh["entry_time"].to_numpy() % 86400) // 3600
    assert len(trh) > 0 and set(hours.tolist()) <= {22, 23, 0, 1}


def test_spread_filter_blocks_wide_spread_entries():
    bars = _synth(3000, seed=5)
    free = generate_trades(bars, SPEC_A, C0, COSTS)
    sp = bars["spread"].to_numpy()
    assert (sp[_entry_bars(bars, free) - 1] > 0.34).any() and (sp[_entry_bars(bars, free)] > 0.34).any()
    # default spread_filter_bar "signal": the signal bar (the bar before the entry) is tested
    spec = dataclasses.replace(SPEC_A, max_spread_usd=0.34)
    assert spec.spread_filter_bar == "signal"
    tr, dec = generate_trades_detailed(bars, spec, C0, COSTS)
    assert len(tr) > 5 and (sp[_entry_bars(bars, tr) - 1] <= 0.34).all()
    assert (dec["status"] == "spread_cap").any()
    # "entry": the entry bar's spread is tested instead
    tre = generate_trades(bars, dataclasses.replace(spec, spread_filter_bar="entry"), C0, COSTS)
    assert len(tre) > 5 and (sp[_entry_bars(bars, tre)] <= 0.34).all()
    # the filter uses the EFFECTIVE spread: x2 spreads block more
    tr2 = generate_trades(bars, spec, C0, COSTS.multiplied(2.0))
    assert (sp[_entry_bars(bars, tr2) - 1] * 2.0 <= 0.34).all()
    with pytest.raises(ValueError, match="spread_filter_bar"):
        PullbackSpec(spread_filter_bar="next")


def test_default_spread_filter_does_not_read_the_entry_bar_spread():
    """Finding 15: a bar's spread is often an average / minimum / maximum over the bar, known only after it.
    With the default spread_filter_bar "signal", changing only the entry bar's spread value must not change
    whether the signal enters (it still changes the fill price, the one-spread-per-bar cost approximation)."""
    bars = _synth(3000, seed=5)
    spec = dataclasses.replace(SPEC_A, max_spread_usd=0.45)
    tr, dec = generate_trades_detailed(bars, spec, C0, COSTS)
    k = int(_entry_bars(bars, tr.iloc[3:4])[0])
    wide = bars.copy()
    sp = wide["spread"].to_numpy().copy()
    sp[k] = 0.60                                         # only the entry bar's spread value
    wide["spread"] = sp
    _, dec2 = generate_trades_detailed(wide, spec, C0, COSTS)
    t_k = int(bars["time"].iloc[k])
    assert dec.loc[dec["entry_time"] == t_k, "status"].tolist() == ["entered"]
    assert dec2.loc[dec2["entry_time"] == t_k, "status"].tolist() == ["entered"]
    _, dec3 = generate_trades_detailed(wide, dataclasses.replace(spec, spread_filter_bar="entry"), C0, COSTS)
    assert dec3.loc[dec3["entry_time"] == t_k, "status"].tolist() == ["spread_cap"]


def test_max_trades_per_day():
    bars = _synth(3000, seed=8)
    spec = dataclasses.replace(SPEC_B, exit_mode="time", time_exit_bars=1)
    free = generate_trades(bars, spec, C0, COSTS)
    assert pd.Series(cal.prop_day(free["entry_time"].to_numpy())).value_counts().max() >= 2
    capped, dec = generate_trades_detailed(bars, dataclasses.replace(spec, max_trades_per_day=1), C0, COSTS)
    assert pd.Series(cal.prop_day(capped["entry_time"].to_numpy())).value_counts().max() == 1
    assert (dec["status"] == "max_trades_per_day").any()


def test_max_open_positions():
    bars = _synth(3000, seed=8)
    one = generate_trades(bars, SPEC_B, C0, COSTS).sort_values("entry_time")
    assert (one["entry_time"].to_numpy()[1:] >= one["exit_time"].to_numpy()[:-1]).all()
    three = generate_trades(bars, dataclasses.replace(SPEC_B, max_open_positions=3), C0, COSTS)
    et, xt = three["entry_time"].to_numpy(), three["exit_time"].to_numpy()
    open_at_entry = np.array([((et < e) & (xt > e)).sum() for e in et])
    assert open_at_entry.max() in (1, 2) and open_at_entry.max() >= 1      # overlaps happen, never above 3 open


def test_daily_stop_after_losses_and_after_loss_pct():
    bars = _synth(3000, seed=8)
    spec = dataclasses.replace(SPEC_B, exit_mode="time", time_exit_bars=2)
    for extra, status in (({"daily_stop_losses": 1}, "daily_stop_losses"), ({"daily_stop_pct": 0.002}, "daily_stop_pct")):
        tr, dec = generate_trades_detailed(bars, dataclasses.replace(spec, **extra), C0, COSTS)
        assert (dec["status"] == status).any()
        pday_x = cal.prop_day(tr["exit_time"].to_numpy())
        for _, t in tr.iterrows():
            d = cal.prop_day(int(t["entry_time"]))
            start = cal.day_start_utc(d)
            before = tr[(tr["exit_time"] <= t["entry_time"]) & (pday_x == d)]
            start_balance = C0 + tr.loc[tr["exit_time"] < start, "pnl_usd"].sum()
            if "daily_stop_losses" in extra:
                assert (before["pnl_usd"] < 0).sum() < 1
            else:
                assert before["pnl_usd"].sum() > -0.002 * start_balance


def test_news_blackout_blocks_entries_and_flatten_closes():
    bars = _synth(3000, seed=5)
    free = generate_trades(bars, SPEC_A, C0, COSTS)
    # windows around the entries of three free trades, 30 minutes before to 30 minutes after the entry
    picks = free.iloc[[2, 7, 12]]
    windows = tuple((int(e) - 1800, int(e) + 1800) for e in picks["entry_time"])
    spec = dataclasses.replace(SPEC_A, news_blackouts_utc=windows)
    tr, dec = generate_trades_detailed(bars, spec, C0, COSTS)
    et = tr["entry_time"].to_numpy()
    for a, b in windows:
        assert not ((et < b) & (et + 3600 > a)).any()
    assert (dec["status"] == "news_blackout").sum() >= 3
    # flatten: a window that starts inside a held trade closes it at the open of the first overlapping bar
    held = free[(free["exit_time"] - free["entry_time"]) >= 5 * 3600].iloc[0]
    w0 = int(held["entry_time"]) + 2 * 3600 + 900                 # 15 minutes into the third bar held
    flat = generate_trades(bars, dataclasses.replace(SPEC_A, news_blackouts_utc=((w0, w0 + 3600),),
                                                       news_flatten=True), C0, COSTS)
    hit = flat[flat["entry_time"] == held["entry_time"]].iloc[0]
    assert hit["exit_reason"] == "signal" and hit["exit_time"] == int(held["entry_time"]) + 2 * 3600
    k = int(np.searchsorted(bars["time"].to_numpy(), hit["exit_time"]))
    assert cents(hit["exit_price"], COSTS.sell_fill(bars["open"].iloc[k]) if hit["side"] > 0
                 else COSTS.buy_fill(bars["open"].iloc[k], bars["spread"].iloc[k]))


def test_vol_caps():
    bars = _synth(3000, seed=5)
    free = generate_trades(bars, SPEC_A, C0, COSTS)
    dist = (free["entry_price"] - free["stop_price"]).abs()
    cap = float(dist.median())
    tr, dec = generate_trades_detailed(bars, dataclasses.replace(SPEC_A, max_stop_usd=cap), C0, COSTS)
    assert len(tr) > 0 and ((tr["entry_price"] - tr["stop_price"]).abs() <= cap).all()
    assert (dec["status"] == "stop_cap").any()
    sig = compute_signals(bars, SPEC_A)
    atr_cap = float(np.nanmedian(sig["atr"]))
    tr2, dec2 = generate_trades_detailed(bars, dataclasses.replace(SPEC_A, max_atr_usd=atr_cap), C0, COSTS)
    assert (sig["atr"].to_numpy()[dec2.loc[dec2["status"] == "entered", "signal_bar"]] <= atr_cap).all()
    assert (dec2["status"] == "atr_cap").any()


def test_entry_after_gap_false_skips_signals_before_a_gap():
    bars = _synth(3000, seed=5)
    tr, dec = generate_trades_detailed(bars, dataclasses.replace(SPEC_A, entry_after_gap=False), C0, COSTS)
    t = bars["time"].to_numpy()
    k = _entry_bars(bars, tr)
    assert (t[k] - t[k - 1] == 3600).all()
    assert (dec["status"] == "gap_before_entry").any()


# ======================================================================== exits

def test_trail_ema_exit_at_next_open_after_a_close_beyond_the_ema():
    bars = _synth(3000, seed=3)
    tr = generate_trades(bars, SPEC_B, C0, COSTS)
    sig = compute_signals(bars, SPEC_B)
    t, c, o, sp = (bars[x].to_numpy() for x in ("time", "close", "open", "spread"))
    trail = tr[tr["exit_reason"] == "trail"]
    assert len(trail) > 10
    for _, r in trail.iterrows():
        k = int(np.searchsorted(t, r["exit_time"]))
        assert t[k] == r["exit_time"]                              # at a bar open
        assert r["side"] * (c[k - 1] - sig["ema_trail"].iloc[k - 1]) < 0
        fill = COSTS.sell_fill(o[k]) if r["side"] > 0 else COSTS.buy_fill(o[k], sp[k])
        assert cents(r["exit_price"], fill)
    assert set(tr["exit_reason"]) <= {"trail", "stop", "end_of_data"}


def test_time_exit_after_n_bars():
    bars = _synth(3000, seed=3)
    spec = dataclasses.replace(SPEC_B, exit_mode="time", time_exit_bars=5)
    tr = generate_trades(bars, spec, C0, COSTS)
    t = bars["time"].to_numpy()
    timed = tr[tr["exit_reason"] == "time"]
    assert len(timed) > 10
    e = np.searchsorted(t, timed["entry_time"].to_numpy())
    x = np.searchsorted(t, timed["exit_time"].to_numpy())
    assert (t[x] == timed["exit_time"].to_numpy()).all() and (x - e == 5).all()
    assert set(tr["exit_reason"]) <= {"time", "stop", "end_of_data"}


def test_end_of_data_exit_and_no_entry_on_the_last_bar():
    bars = _hand_bars(UP_AFTER[:2])                                # data ends while the trade is open
    t = generate_trades(bars, HAND_SPEC, C0, HAND_COSTS).iloc[0]
    last = len(bars) - 1
    assert t["exit_reason"] == "end_of_data" and t["exit_time"] == MONDAY + last * 3600 + 3600
    assert cents(t["exit_price"], bars["close"].iloc[last] - 0.15)
    sig_only = _hand_bars([])[:34]                                 # the trigger bar is the last bar
    trades, dec = generate_trades_detailed(sig_only, HAND_SPEC, C0, HAND_COSTS)
    assert len(trades) == 0 and list(dec["status"]) == ["no_next_bar"] and dec["entry_time"].iloc[0] == -1


# ======================================================================== mirror

def _mirror(bars: pd.DataFrame, level: float, spread: float) -> pd.DataFrame:
    """Reflect the bars so the ask of the mirror is the reflection of the bid: bid' = 2L - spread - bid."""
    k = 2.0 * level - spread
    return pd.DataFrame({"time": bars["time"].to_numpy(), "open": k - bars["open"].to_numpy(),
                         "high": k - bars["low"].to_numpy(), "low": k - bars["high"].to_numpy(),
                         "close": k - bars["close"].to_numpy(), "spread": np.full(len(bars), spread)})


@pytest.mark.parametrize("spec", [SPEC_A, SPEC_B, dataclasses.replace(SPEC_B, exit_mode="time", time_exit_bars=4),
                                  dataclasses.replace(SPEC_A, pullback_mode="atr_from_swing", stop_mode="atr")],
                         ids=["A", "B", "B-time", "A-swing-atr"])
def test_both_directions_mirror_exactly(spec):
    bars = _synth(3000, seed=12)
    bars["spread"] = 0.30
    costs = CostModel(markup_per_side=0.05, slippage_per_side=0.02, commission_per_lot_round_trip=7.0,
                      swap_enabled=False)
    level = 2000.0
    m = _mirror(bars, level, 0.30)
    a = generate_trades(bars, spec, C0, costs)
    b = generate_trades(m, spec, C0, costs)
    assert len(a) > 10 and (a["side"] == 1).any() and (a["side"] == -1).any()
    assert len(a) == len(b)
    assert (b["side"].to_numpy() == -a["side"].to_numpy()).all()
    for col in ("units", "entry_time", "exit_time", "exit_reason"):
        assert (a[col].to_numpy() == b[col].to_numpy()).all(), col
    for col in ("entry_price", "exit_price", "stop_price"):
        assert np.allclose(b[col].to_numpy(), 2 * level - a[col].to_numpy(), rtol=0, atol=1e-8), col
    for col in ("risk_usd", "commission_usd", "pnl_usd"):
        assert np.allclose(b[col].to_numpy(), a[col].to_numpy(), rtol=0, atol=1e-6), col
    # a long-only spec on the bars equals a short-only spec on the mirror
    la = generate_trades(bars, dataclasses.replace(spec, direction="long"), C0, costs)
    sb = generate_trades(m, dataclasses.replace(spec, direction="short"), C0, costs)
    assert len(la) == len(sb) > 0 and (la["side"] == 1).all() and (sb["side"] == -1).all()
    assert np.allclose(sb["entry_price"], 2 * level - la["entry_price"], atol=1e-8)
    assert np.allclose(sb["pnl_usd"], la["pnl_usd"], atol=1e-6)


# ======================================================================== ledger and costs

def test_ledger_matches_propkit_equity_and_equity_accepts_the_trades():
    equity = pytest.importorskip("propkit.equity")
    adapters = pytest.importorskip("propkit.adapters")
    bars = _synth(3000, seed=6)
    costs = CostModel(markup_per_side=0.05, slippage_per_side=0.02, commission_per_lot_round_trip=7.0)
    tr = generate_trades(bars, SPEC_A, C0, costs)
    assert tuple(tr.columns) == pl.TRADES_COLUMNS == adapters.TRADES_COLUMNS
    assert (tr["swap_usd"] < 0).any()                             # trades held over 17:00 New York pay swap
    eq, out = equity.equity_from_trades(bars, tr, C0, costs)
    for col in ("commission_usd", "swap_usd", "pnl_usd", "risk_usd"):
        assert np.allclose(out[col], tr[col], rtol=0, atol=1e-6), col
    assert eq["balance"].iloc[-1] == pytest.approx(C0 + tr["pnl_usd"].sum(), abs=1e-6)
    assert set(tr["exit_reason"]) <= set(adapters.EXIT_REASONS)


def test_flat_rate_costs_fill_at_the_bid():
    bars = _synth(2000, seed=6)
    flat = CostModel(flat_rate_per_side=0.0003, swap_enabled=False)
    tr = generate_trades(bars, SPEC_A, C0, flat)
    assert len(tr) > 5
    k = _entry_bars(bars, tr)
    assert np.allclose(tr["entry_price"], bars["open"].to_numpy()[k])
    assert np.allclose(tr["commission_usd"], 0.0003 * tr["units"] * (tr["entry_price"] + tr["exit_price"]))


def test_deterministic_and_fast():
    bars = pb.synthetic_bars(MONDAY, 63_645, seed=1)
    t0 = time.perf_counter()
    a = generate_trades(bars, SPEC_A, C0, COSTS)
    elapsed = time.perf_counter() - t0
    b = generate_trades(bars, SPEC_A, C0, COSTS)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert len(a) > 500
    assert elapsed < 30.0, f"63,645 H1 bars took {elapsed:.1f} s"


# ======================================================================== the spec

def test_defaults_are_marked_placeholders():
    s = PullbackSpec()
    assert s.placeholder is True and "PLACEHOLDER" in s.name
    assert "PLACEHOLDER" in PullbackSpec.__doc__ and "PLACEHOLDER" in pl.__doc__
    assert s.summary_lines()[0].startswith("PLACEHOLDER SPEC")
    assert PullbackSpec(placeholder=False, name="zeno v1").summary_lines()[0] == "spec: zeno v1"


def test_json_round_trip_and_files(tmp_path):
    s = dataclasses.replace(SPEC_C, name="round trip")
    text = s.to_json()
    assert text.isascii() and PullbackSpec.from_json(text) == s
    d = json.loads(text)
    assert d["news_blackouts_utc"][0] == ["2024-01-12T13:00:00Z", "2024-01-12T15:00:00Z"]
    assert d["session_hours_utc"] == [[22.0, 2.0]]
    path = tmp_path / "spec.json"
    s.to_json(path)
    assert PullbackSpec.from_json(str(path)) == s and PullbackSpec.from_json(path) == s
    assert PullbackSpec.from_dict({"_comment": "ignored", "direction": "long"}) == PullbackSpec(direction="long")
    epoch = PullbackSpec(news_blackouts_utc=[[1704891600, 1704898800]])
    assert epoch.news_blackouts_utc == ((1704891600, 1704898800),)


def test_example_file_and_readme_document_every_field():
    assert EXAMPLE.read_text(encoding="utf-8").isascii() and README.read_text(encoding="utf-8").isascii()
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    names = [f.name for f in dataclasses.fields(PullbackSpec)]
    assert [k for k in raw if not k.startswith("_")] == names       # every field, in order
    assert PullbackSpec.from_json(EXAMPLE) == PullbackSpec()         # = the placeholder defaults
    assert raw["placeholder"] is True
    readme = README.read_text(encoding="utf-8")
    for name in names:
        assert f"| `{name}` |" in readme, name
    assert "PLACEHOLDER" in readme


@pytest.mark.parametrize("bad", [
    {"direction": "up"}, {"risk_pct": 0.5}, {"risk_pct": 0.0}, {"risk_pct": "0.01"}, {"trend_ema_period": 0},
    {"trend_ema_period": 2.5}, {"pullback_mode": "fib"}, {"trigger": "hammer"}, {"stop_mode": "pct"},
    {"exit_mode": "never"}, {"min_depth_atr": 2.0, "max_depth_atr": 1.0}, {"sessions": ["tokyo"]},
    {"session_hours_utc": [[5, 5]]}, {"session_hours_utc": [[5, 25]]}, {"session_hours_utc": [5, 6]},
    {"news_blackouts_utc": [["2024-01-02T10:00:00Z", "2024-01-02T09:00:00Z"]]},
    {"news_blackouts_utc": [["2024-01-02 10:00", "2024-01-02 11:00"]]}, {"max_open_positions": 0},
    {"daily_stop_pct": 3.0}, {"placeholder": "no"}, {"target_r": -1.0}, {"max_spread_usd": float("nan")},
    {"directon": "long"}, {"bar_minutes": 0},
])
def test_invalid_spec_values_refused(bad):
    with pytest.raises(ValueError):
        PullbackSpec.from_dict(bad)


def test_bad_json_locked_paths_and_missing_files(tmp_path):
    with pytest.raises(ValueError, match="not valid JSON"):
        PullbackSpec.from_json("{\"direction\": \"long\",}")
    with pytest.raises(ValueError, match="not found"):
        PullbackSpec.from_json(str(tmp_path / "nope.json"))
    with pytest.raises(pb.LockedPathError):
        PullbackSpec.from_json(str(tmp_path / "locked_holdout" / "spec.json"))
    with pytest.raises(pb.LockedPathError):
        PullbackSpec().to_json(tmp_path / "spec.json.locked")
    with pytest.raises(ValueError, match="unknown"):
        PullbackSpec.from_json("{\"risk\": 0.01}")


def test_generator_input_checks():
    bars = _hand_bars(UP_AFTER)
    with pytest.raises(ValueError, match="CostModel"):
        generate_trades(bars, HAND_SPEC, C0, None)
    with pytest.raises(ValueError, match="PullbackSpec"):
        generate_trades(bars, {"direction": "long"}, C0, HAND_COSTS)
    with pytest.raises(ValueError, match="C0"):
        generate_trades(bars, HAND_SPEC, -5.0, HAND_COSTS)
    with pytest.raises(ValueError, match="lot_step_oz"):
        generate_trades(bars, HAND_SPEC, C0, HAND_COSTS, lot_step_oz=0.0)
    with pytest.raises(ValueError, match="15-minute"):
        generate_trades(bars, dataclasses.replace(HAND_SPEC, bar_minutes=15), C0, HAND_COSTS)
    assert len(generate_trades(bars, dataclasses.replace(HAND_SPEC, bar_minutes=60), C0, HAND_COSTS)) == 1
    broken = bars.copy()
    broken.loc[5, "high"] = 1.0
    with pytest.raises(ValueError):
        generate_trades(broken, HAND_SPEC, C0, HAND_COSTS)


def test_warmup_rule():
    assert PullbackSpec().warmup() == 3 * 200 + 10
    assert dataclasses.replace(SPEC_B, trend_require_close_side=False, trend_slope_bars=0).warmup() == 3 * 8
    assert HAND_SPEC.warmup() == 20
    bars = _synth(800, seed=1)
    sig = compute_signals(bars, SPEC_A)
    first = np.flatnonzero(sig["long_signal"].to_numpy() | sig["short_signal"].to_numpy())
    assert first.size and first[0] >= SPEC_A.warmup()


def test_decision_log_statuses_are_documented():
    bars = _synth(3000, seed=5)
    _, dec = generate_trades_detailed(bars, SPEC_C, C0, COSTS)
    assert tuple(dec.columns) == pl.DECISION_COLUMNS
    assert set(dec["status"]) <= set(pl.DECISION_STATUS)
    assert len(set(dec["status"])) >= 4


def test_source_is_ascii_and_imports_nothing_from_alphamaster():
    text = SRC.read_text(encoding="utf-8")
    assert text.isascii()
    forbidden = r"^\s*(from|import)\s+(model_core|data_pipeline|config|web|utils|strategy_manager|execution|scripts)\b"
    assert not re.search(forbidden, text, flags=re.M)
    assert "zoneinfo" not in text and "scipy" not in text and "random" not in text.replace("np.random", "")
    for fn in (generate_trades, generate_trades_detailed, compute_signals, floor_to_lot_step,
               PullbackSpec.from_json, PullbackSpec.to_json, PullbackSpec.warmup):
        assert fn.__doc__, fn.__name__
    assert math.isfinite(HAND_SPEC.risk_pct)


def test_no_signal_gives_empty_frames_with_the_standard_columns():
    bars = _synth(80, seed=1)                                      # shorter than the warm-up (3 x 30 + 3)
    trades, dec = generate_trades_detailed(bars, SPEC_A, C0, COSTS)
    assert len(trades) == 0 and tuple(trades.columns) == pl.TRADES_COLUMNS
    assert len(dec) == 0 and tuple(dec.columns) == pl.DECISION_COLUMNS
    assert trades["entry_time"].dtype == np.int64 and trades["pnl_usd"].dtype == np.float64
    sig = compute_signals(bars, SPEC_A)
    assert list(sig.columns) == ["time", "ema_trend", "ema_pullback", "ema_trail", "atr", "long_signal",
                                 "short_signal", "long_extreme", "short_extreme", "long_depth_atr",
                                 "short_depth_atr"]
    assert not sig["long_signal"].any() and not sig["short_signal"].any()


def test_swap_of_a_trade_closed_at_a_rollover_open_does_not_see_that_bar_close():
    """Finding 14: on 24-hour bars a rollover falls on a bar open. A trade closed at that open pays its swap
    on the bid at the rollover (the open); moving only that bar's close must change neither that swap nor
    the trades entered at the same open (whose size comes from the balance)."""
    b = pb.synthetic_bars(MONDAY, 1500, bar_seconds=3600, seed=11, vol_per_hour=0.003, spread=0.34,
                          market_hours="always")
    cm = CostModel(markup_per_side=0.05, slippage_per_side=0.02, commission_per_lot_round_trip=7.0)
    spec = PullbackSpec(name="t", placeholder=False, trend_ema_period=20, trend_slope_bars=0,
                        trend_require_close_side=False, pullback_mode="ema_touch", ema_pullback_period=5,
                        pullback_lookback_bars=3, trigger="break_prev_extreme", stop_mode="atr", stop_atr_mult=3.0,
                        exit_mode="time", time_exit_bars=3, max_open_positions=3, risk_pct=0.01, atr_period=5)
    base, _ = generate_trades_detailed(b, spec, C0, cm, lot_step_oz=0.0001)
    t = b["time"].to_numpy()
    roll = set(cal.rollover_instants(int(t[0]), int(t[-1]) + 1).tolist())
    ks = [k for k in range(1, len(t) - 1) if int(t[k]) in roll and (base["exit_time"] == t[k]).any()]
    assert len(ks) >= 3                                  # the case occurs in this data
    cols = ["trade_id", "units", "entry_price", "risk_usd"]
    for k in ks[:4]:
        m = b.copy()
        c, h, lo = m["close"].to_numpy().copy(), m["high"].to_numpy().copy(), m["low"].to_numpy().copy()
        c[k] = m["open"].iloc[k] * 1.03
        h[k], lo[k] = max(h[k], c[k]), min(lo[k], c[k])
        m["close"], m["high"], m["low"] = c, h, lo
        out, _ = generate_trades_detailed(m, spec, C0, cm, lot_step_oz=0.0001)
        assert np.array_equal(base.loc[base["exit_time"] == t[k], "swap_usd"].to_numpy(),
                              out.loc[out["exit_time"] == t[k], "swap_usd"].to_numpy())
        assert base.loc[base["entry_time"] == t[k], cols].reset_index(drop=True).equals(
            out.loc[out["entry_time"] == t[k], cols].reset_index(drop=True))
