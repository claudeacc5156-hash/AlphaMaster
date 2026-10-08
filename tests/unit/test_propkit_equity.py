"""Tests for propkit/equity.py: the EQUITY path of trades and positions. Synthetic data only. Research only."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from propkit import adapters, calendar
from propkit.bars import metals_market_open, synthetic_bars
from propkit.costs import CostModel
from propkit.equity import (EQUITY_COLUMNS, bar_of_instants, equity_from_positions, equity_from_trades,
                            validate_equity)

C0 = 100_000.0
NO_SWAP = CostModel(swap_enabled=False)


def T(text: str) -> int:
    """'YYYY-MM-DD HH:MM[:SS]' UTC -> epoch seconds."""
    return int(np.datetime64(text.replace(" ", "T"), "s").astype(np.int64))


def make_bars(first: str, last_end: str, overrides: dict, start_price: float = 2000.0, spread: float = 0.30,
              market: str = "metals", step: int = 3600) -> pd.DataFrame:
    """H1 bars from `first` to before `last_end` (metals hours or every hour). Bars named in overrides get
    (open, high, low, close, spread); every other bar is flat at the previous close with `spread`."""
    cand = np.arange(T(first), T(last_end), step, dtype=np.int64)
    times = cand[metals_market_open(cand)] if market == "metals" else cand
    over = {T(k): v for k, v in overrides.items()}
    assert set(over) <= set(times.tolist()), "an override names a bar that does not exist"
    rows, last = [], start_price
    for t in times:
        if t in over:
            o, h, lo, c, s = over[t]
            last = c
        else:
            o = h = lo = c = last
            s = spread
        rows.append((int(t), float(o), float(h), float(lo), float(c), float(s)))
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "spread"])


def row(eq: pd.DataFrame, text: str) -> pd.Series:
    hit = eq.loc[eq["time"] == T(text)]
    assert len(hit) == 1, text
    return hit.iloc[0]


def cents(a, b) -> bool:
    return abs(float(a) - float(b)) < 1e-6


# ---------------------------------------------------------------------------------------
# the 3-trade hand example (July 2024: New York on EDT = UTC-4, Prague on CEST = UTC+2, so the 17:00 New
# York rollover is 21:00 UTC, the metals daily break is 21:00-22:00 UTC and prop days start at 22:00 UTC)

HAND_COSTS = CostModel(markup_per_side=0.05, slippage_per_side=0.03, commission_per_lot_round_trip=7.0,
                       lot_size_oz=100.0, swap_long=6.0, swap_short=2.0, swap_unit="pct_per_year",
                       swap_day_count=360.0, triple_swap_weekday=2, rollover_hour_ny=17)

HAND_BARS = make_bars("2024-07-09 00:00", "2024-07-12 21:00", {
    "2024-07-09 08:00": (2000.0, 2003.0, 1998.0, 2001.0, 0.30),   # trade 1 long entry at the open
    "2024-07-09 14:00": (2008.0, 2011.0, 2007.0, 2010.5, 0.30),   # trade 1 target hit at 14:30, bid 2010.00
    "2024-07-10 15:00": (2020.0, 2021.0, 2019.0, 2020.0, 0.30),   # trade 2 short entry at the open (Wednesday)
    "2024-07-10 18:00": (2020.0, 2023.0, 2018.0, 2019.0, 0.30),   # adverse high for the short
    "2024-07-10 20:00": (2019.0, 2019.5, 2015.0, 2016.0, 0.30),   # ends 21:00 UTC = Wednesday rollover
    "2024-07-11 10:00": (2005.0, 2006.0, 2004.0, 2005.5, 0.40),   # trade 2 exit at the open, spread 0.40
    "2024-07-11 12:00": (2030.0, 2031.0, 2029.0, 2030.0, 0.30),   # trade 3 long entry at the open
    "2024-07-11 20:00": (2030.0, 2030.0, 2024.0, 2025.0, 0.30),   # ends 21:00 UTC = Thursday rollover
    "2024-07-11 22:00": (2015.0, 2016.0, 2012.0, 2014.0, 0.30),   # reopen gaps through the 2020 stop
})

HAND_TRADES = pd.DataFrame({
    "trade_id": [1, 2, 3],
    "side": [1, -1, 1],
    "units": [100.0, 50.0, 200.0],
    "entry_time": [T("2024-07-09 08:00"), T("2024-07-10 15:00"), T("2024-07-11 12:00")],
    # fills: buy = bid + spread + markup 0.05 + slippage 0.03; sell = bid - 0.05 - 0.03
    "entry_price": [2000.00 + 0.30 + 0.05 + 0.03,      # 2000.38 long at the ask
                    2020.00 - 0.05 - 0.03,             # 2019.92 short at the bid
                    2030.00 + 0.30 + 0.05 + 0.03],     # 2030.38 long at the ask
    "exit_time": [T("2024-07-09 14:30"), T("2024-07-11 10:00"), T("2024-07-11 22:00")],
    "exit_price": [2010.00 - 0.05 - 0.03,              # 2009.92 long sold at the bid (target)
                   2005.00 + 0.40 + 0.05 + 0.03,       # 2005.48 short bought back at the ask
                   2015.00 - 0.05 - 0.03],             # 2014.92 stop gapped through: filled at the open bid
    "exit_reason": ["target", "signal", "stop"],
    "stop_price": [np.nan, np.nan, 2020.0],
})


@pytest.fixture(scope="module")
def hand():
    return equity_from_trades(HAND_BARS, HAND_TRADES, C0, HAND_COSTS)


def test_hand_example_trades_to_the_cent(hand):
    _, tr = hand
    # Trade 1, long 100 oz (1 lot), Tue 08:00 -> Tue 14:30 UTC (10:30 New York, before the 17:00 rollover):
    #   gross = 100 x (2009.92 - 2000.38) = 100 x 9.54 = 954.00
    #   commission = 7.00 per lot round trip x 1 lot = 7.00 (3.50 per fill); swap = 0 (no rollover held)
    #   pnl = 954.00 - 7.00 + 0 = 947.00
    # Trade 2, short 50 oz, Wed 15:00 -> Thu 10:00 UTC, holds the Wednesday 17:00 New York rollover (21:00 UTC):
    #   gross = 50 x (2019.92 - 2005.48) = 50 x 14.44 = 722.00
    #   commission = 7.00 x 0.5 lot = 3.50
    #   swap = -(2.0 % / 100) x 50 oz x 2016.00 (bid close of the 20:00 bar) / 360 x 3 nights (Wednesday)
    #        = -(0.02 x 50 x 2016 / 360 x 3) = -(1 x 5.6 x 3) = -16.80
    #   pnl = 722.00 - 3.50 - 16.80 = 701.70
    # Trade 3, long 200 oz, Thu 12:00 -> Thu 22:00 UTC (the reopen), holds the Thursday rollover (1 night):
    #   gross = 200 x (2014.92 - 2030.38) = 200 x -15.46 = -3092.00
    #   commission = 7.00 x 2 lots = 14.00
    #   swap = -(6.0 / 100) x 200 x 2025.00 / 360 x 1 = -(12 x 2025 / 360) = -67.50
    #   pnl = -3092.00 - 14.00 - 67.50 = -3173.50
    #   risk_usd (1R): the stop 2020.00 is a bid level; a stop fill sells at 2020.00 - 0.05 - 0.03 = 2019.92
    #   1R = 200 x |2030.38 - 2019.92| + 14.00 = 2092.00 + 14.00 = 2106.00
    assert list(tr["trade_id"]) == [1, 2, 3]
    for got, want in zip(tr["commission_usd"], [7.00, 3.50, 14.00]):
        assert cents(got, want)
    for got, want in zip(tr["swap_usd"], [0.00, -16.80, -67.50]):
        assert cents(got, want)
    for got, want in zip(tr["pnl_usd"], [947.00, 701.70, -3173.50]):
        assert cents(got, want)
    assert np.isnan(tr["risk_usd"].iloc[0]) and np.isnan(tr["risk_usd"].iloc[1])
    assert cents(tr["risk_usd"].iloc[2], 2106.00)
    assert list(tr["exit_reason"]) == ["target", "signal", "stop"]


def test_hand_example_balance_path_to_the_cent(hand):
    eq, _ = hand
    # bookings (USD): commission per fill = 3.50 per lot; swaps in the bar that ends at the rollover
    steps = [
        ("2024-07-09 07:00", 100_000.00),                     # flat, nothing booked yet
        ("2024-07-09 08:00", 100_000.00 - 3.50),              # trade 1 entry commission -> 99,996.50
        ("2024-07-09 14:00", 99_996.50 + 954.00 - 3.50),      # trade 1 closed -> 100,947.00
        ("2024-07-10 15:00", 100_947.00 - 1.75),              # trade 2 entry commission -> 100,945.25
        ("2024-07-10 20:00", 100_945.25 - 16.80),             # Wednesday triple swap -> 100,928.45
        ("2024-07-11 10:00", 100_928.45 + 722.00 - 1.75),     # trade 2 closed -> 101,648.70
        ("2024-07-11 12:00", 101_648.70 - 7.00),              # trade 3 entry commission -> 101,641.70
        ("2024-07-11 20:00", 101_641.70 - 67.50),             # Thursday swap -> 101,574.20
        ("2024-07-11 22:00", 101_574.20 - 3092.00 - 7.00),    # trade 3 stopped -> 98,475.20
    ]
    for text, want in steps:
        assert cents(row(eq, text)["balance"], want), text
    assert cents(eq["balance"].iloc[-1], 98_475.20)
    assert cents(eq["balance"].iloc[-1], C0 + 947.00 + 701.70 - 3173.50)
    r = row(eq, "2024-07-10 20:00")
    assert cents(r["swap_usd"], -16.80) and cents(r["commission_usd"], 0.0)
    r = row(eq, "2024-07-11 22:00")
    assert cents(r["realised_usd"], -3092.00) and cents(r["commission_usd"], 7.00)
    # exactly two bars carry swap: one per held rollover
    assert int((eq["swap_usd"] != 0).sum()) == 2


def test_hand_example_marks_to_the_cent(hand):
    eq, _ = hand
    # Tue 08:00, long opened AT the open, marked from its fill 2000.38:
    #   equity_close = 99,996.50 + 100 x (2001.00 - 2000.38) = 99,996.50 + 62.00 = 100,058.50
    #   equity_worst = 99,996.50 + 100 x (1998.00 bid low - 2000.38) = 99,996.50 - 238.00 = 99,758.50
    r = row(eq, "2024-07-09 08:00")
    assert cents(r["equity_close"], 100_058.50) and cents(r["equity_worst"], 99_758.50)
    assert r["units_open"] == 100.0
    # Tue 14:00, closed at the 2010.00 target inside the bar: equity_worst is the lower of
    #   the start level 100,058.50 (13:00 close) - 3.50 = 100,055.00, and
    #   99,996.50 + min(954.00, 100 x (2007.00 low - 2000.38) = 662.00) - 3.50 = 100,655.00  -> 100,055.00
    r = row(eq, "2024-07-09 14:00")
    assert cents(r["equity_worst"], 100_055.00) and cents(r["equity_close"], 100_947.00)
    assert r["units_open"] == 0.0
    # Wed 18:00, short 50 held: ask high = 2023.00 + 0.30 -> 50 x (2019.92 - 2023.30) = -169.00
    #   equity_worst = 100,945.25 - 169.00 = 100,776.25; close at ask 2019.30: 50 x 0.62 = +31.00
    r = row(eq, "2024-07-10 18:00")
    assert cents(r["equity_worst"], 100_776.25) and cents(r["equity_close"], 100_976.25)
    assert r["units_open"] == -50.0
    # Thu 20:00: long 200 at close 2025.00: 101,574.20 + 200 x (2025.00 - 2030.38) = 101,574.20 - 1076.00
    r = row(eq, "2024-07-11 20:00")
    assert cents(r["equity_close"], 100_498.20)
    # worst: 101,641.70 + 200 x (2024.00 low - 2030.38) - 67.50 swap = 101,641.70 - 1276.00 - 67.50
    assert cents(r["equity_worst"], 100_298.20)
    # Thu 22:00 (the gap bar; 00:00 CEST Friday): stopped AT the open, so the realised fill counts and the
    # bar's low (2012.00) does not: 101,574.20 - 3092.00 - 7.00 = 98,475.20
    r = row(eq, "2024-07-11 22:00")
    assert cents(r["equity_worst"], 98_475.20) and cents(r["equity_close"], 98_475.20)
    assert calendar.day_to_str(calendar.prop_day(T("2024-07-11 22:00"))) == "2024-07-12"


def test_hand_example_invariants(hand):
    eq, tr = hand
    assert tuple(eq.columns) == EQUITY_COLUMNS and len(eq) == len(HAND_BARS)
    assert eq["time"].tolist() == HAND_BARS["time"].tolist()
    _check_invariants(eq, tr)
    validate_equity(eq)


# ---------------------------------------------------------------------------------------
# general invariants on random trades

def _check_invariants(eq: pd.DataFrame, tr: pd.DataFrame, c0: float = C0) -> None:
    bal = eq["balance"].to_numpy()
    step = eq["realised_usd"].to_numpy() - eq["commission_usd"].to_numpy() + eq["swap_usd"].to_numpy()
    assert np.allclose(np.diff(np.r_[c0, bal]), step, atol=1e-6)
    assert np.isclose(bal[-1], c0 + tr["pnl_usd"].sum(), atol=1e-6)
    assert np.isclose(eq["commission_usd"].sum(), tr["commission_usd"].sum(), atol=1e-6)
    assert np.isclose(eq["swap_usd"].sum(), tr["swap_usd"].sum(), atol=1e-6)
    gross = tr["side"] * tr["units"] * (tr["exit_price"] - tr["entry_price"])
    assert np.allclose(tr["pnl_usd"], gross - tr["commission_usd"] + tr["swap_usd"], atol=1e-9)
    close = eq["equity_close"].to_numpy()
    worst = eq["equity_worst"].to_numpy()
    assert (worst <= close + 1e-9).all()
    prev_adj = np.r_[c0, close[:-1]] - eq["commission_usd"].to_numpy() + np.minimum(eq["swap_usd"].to_numpy(), 0)
    assert (worst <= prev_adj + 1e-9).all()
    flat = eq["units_open"].to_numpy() == 0
    assert (close[flat] == bal[flat]).all()                    # exactly equal when flat


def _random_trades(bars: pd.DataFrame, cm: CostModel, n: int, seed: int, max_hold: int = 40) -> pd.DataFrame:
    """Random round trips entered and exited at bar opens (or inside bars), fills priced by the cost model."""
    rng = np.random.default_rng(seed)
    t = bars["time"].to_numpy()
    o = bars["open"].to_numpy()
    lo, hi = bars["low"].to_numpy(), bars["high"].to_numpy()
    sp = cm.bar_spreads(bars)
    e = rng.integers(0, len(t) - 2, n)
    x = np.minimum(e + rng.integers(0, max_hold, n), len(t) - 1)
    side = rng.choice([-1, 1], n)
    units = np.round(rng.uniform(1, 300, n), 2)
    inside = rng.random(n) < 0.3                        # some exits inside the bar at a price in its range
    off = np.where(inside, rng.integers(1, 3599, n), 0)
    xbid = np.where(inside, lo[x] + rng.random(n) * (hi[x] - lo[x]), o[x])
    ep = np.where(side > 0, cm.buy_fill(o[e], sp[e]), cm.sell_fill(o[e], sp[e]))
    xp = np.where(side > 0, cm.sell_fill(xbid, sp[x]), cm.buy_fill(xbid, sp[x]))
    xt = t[x] + off
    xt = np.maximum(xt, t[e])
    reasons = rng.choice(["stop", "target", "signal", "time", "trail"], n)
    return pd.DataFrame({"side": side, "units": units, "entry_time": t[e], "entry_price": ep,
                         "exit_time": xt, "exit_price": xp, "exit_reason": reasons})


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_invariants_on_random_trades(seed):
    bars = synthetic_bars(T("2024-02-26 00:00"), 1500, seed=seed)       # crosses both 2024 DST changes
    cm = CostModel(markup_per_side=0.02, slippage_per_side=0.05, commission_per_lot_round_trip=6.0,
                   swap_long=5.0, swap_short=-1.0)                     # a short swap CREDIT too
    tr_in = _random_trades(bars, cm, 300, seed)
    eq, tr = equity_from_trades(bars, tr_in, C0, cm)
    _check_invariants(eq, tr)
    assert list(tr["trade_id"]) == list(range(1, 301))
    # units_open at each close = signed units of trades with entry bar <= k < exit bar
    times = bars["time"].to_numpy()
    e = np.searchsorted(times, tr["entry_time"].to_numpy(), side="right") - 1
    x = np.searchsorted(times, tr["exit_time"].to_numpy(), side="right") - 1
    k = 700
    held = (e <= k) & (x > k)
    assert np.isclose(eq["units_open"].iloc[k], (tr["side"] * tr["units"])[held].sum())


def test_swap_per_trade_matches_costmodel_held_rollovers():
    bars = synthetic_bars(T("2024-10-21 00:00"), 800, seed=5)         # crosses the US-only shift weeks
    cm = CostModel(swap_long=7.0, swap_short=3.0, triple_swap_weekday=4)
    tr_in = _random_trades(bars, cm, 200, seed=5, max_hold=120)
    _, tr = equity_from_trades(bars, tr_in, C0, cm)
    times, opens, close = bars["time"].to_numpy(), bars["open"].to_numpy(), bars["close"].to_numpy()
    for i in range(len(tr)):
        r, nights = cm.held_rollovers(int(tr["entry_time"].iloc[i]), int(tr["exit_time"].iloc[i]))
        k = np.searchsorted(times, r, side="right") - 1
        bid = [float(opens[j]) if times[j] == ri else float(close[j]) for j, ri in zip(k, r)]
        want = sum(cm.swap_for_night(int(tr["side"].iloc[i]), float(tr["units"].iloc[i]), b,
                                     int(calendar.ny_weekday(int(ri)))) for b, ri in zip(bid, r))
        assert np.isclose(tr["swap_usd"].iloc[i], want, atol=1e-9)
        if r.size:
            assert set(nights.tolist()) <= {1, 3}


# ---------------------------------------------------------------------------------------
# contract known answers

def test_flat_rate_charges_rate_times_notional_at_each_fill():
    bars = make_bars("2024-01-08 00:00", "2024-01-08 06:00", {
        "2024-01-08 02:00": (2000.0, 2001.0, 1999.0, 2000.0, 0.30),
        "2024-01-08 04:00": (2010.0, 2011.0, 2009.0, 2010.0, 0.30)}, market="always")
    cm = CostModel(flat_rate_per_side=0.0003, swap_enabled=False)
    trades = pd.DataFrame({"side": [1], "units": [10.0], "entry_time": [T("2024-01-08 02:00")],
                           "entry_price": [2000.0], "exit_time": [T("2024-01-08 04:00")],
                           "exit_price": [2010.0], "exit_reason": ["signal"]})
    eq, tr = equity_from_trades(bars, trades, C0, cm)
    # entry fill: 0.0003 x 10 oz x 2000.00 = 6.00; exit fill: 0.0003 x 10 x 2010.00 = 6.03
    assert cents(row(eq, "2024-01-08 02:00")["commission_usd"], 6.00)
    assert cents(row(eq, "2024-01-08 04:00")["commission_usd"], 6.03)
    assert cents(tr["commission_usd"].iloc[0], 12.03)
    assert cents(tr["pnl_usd"].iloc[0], 10 * 10.0 - 12.03)          # 100.00 - 12.03 = 87.97
    # flat rate: no spread, so the long is marked at the bid from its fill: 02:00 close 2000 -> 0 - 6.00
    assert cents(row(eq, "2024-01-08 02:00")["equity_close"], C0 - 6.00)
    assert cents(eq["balance"].iloc[-1], C0 + 87.97)


def test_equity_worst_long_uses_bid_low_short_uses_bid_high_plus_spread():
    bars = make_bars("2024-01-08 00:00", "2024-01-08 06:00", {
        "2024-01-08 03:00": (2000.0, 2006.0, 1993.0, 2001.0, 0.70)}, market="always")
    cm = CostModel(swap_enabled=False)
    for side, entry in ((1, 2000.30), (-1, 2000.00)):
        trades = pd.DataFrame({"side": [side], "units": [10.0], "entry_time": [T("2024-01-08 01:00")],
                               "entry_price": [entry], "exit_time": [T("2024-01-08 05:00")],
                               "exit_price": [2000.0 if side > 0 else 2000.30], "exit_reason": ["signal"]})
        eq, _ = equity_from_trades(bars, trades, C0, cm)
        r = row(eq, "2024-01-08 03:00")
        if side > 0:   # bid low 1993.00: 10 x (1993.00 - 2000.30) = -73.00; close bid 2001.00: +7.00
            assert cents(r["equity_worst"], C0 - 73.00) and cents(r["equity_close"], C0 + 7.00)
        else:          # ask high 2006.00 + 0.70: 10 x (2000.00 - 2006.70) = -67.00; ask close 2001.70: -17.00
            assert cents(r["equity_worst"], C0 - 67.00) and cents(r["equity_close"], C0 - 17.00)


def test_position_opened_at_bar_open_is_marked_from_its_fill():
    bars = make_bars("2024-01-08 00:00", "2024-01-08 04:00", {
        "2024-01-08 01:00": (2000.0, 2004.0, 1999.0, 2002.0, 0.50)}, market="always")
    cm = CostModel(markup_per_side=0.10, slippage_per_side=0.20, commission_per_lot_round_trip=10.0,
                   swap_enabled=False)
    fill = cm.buy_fill(2000.0, 0.50)                         # 2000 + 0.50 + 0.10 + 0.20 = 2000.80
    assert cents(fill, 2000.80)
    trades = pd.DataFrame({"side": [1], "units": [100.0], "entry_time": [T("2024-01-08 01:00")],
                           "entry_price": [fill], "exit_time": [T("2024-01-08 03:00")],
                           "exit_price": [2001.70], "exit_reason": ["time"]})
    eq, _ = equity_from_trades(bars, trades, C0, cm)
    before, r = row(eq, "2024-01-08 00:00"), row(eq, "2024-01-08 01:00")
    assert before["equity_close"] == C0 and before["units_open"] == 0
    # entry commission 5.00 (half of 10 per lot); unrealised from the FILL, not from the previous close:
    assert cents(r["balance"], C0 - 5.00)
    assert cents(r["equity_close"], C0 - 5.00 + 100 * (2002.0 - 2000.80))   # +120.00
    assert cents(r["equity_worst"], C0 - 5.00 + 100 * (1999.0 - 2000.80))   # -180.00
    # the cost of crossing the spread shows at once: at a flat bar after it the mark is bid - fill
    r2 = row(eq, "2024-01-08 02:00")
    assert cents(r2["equity_close"], C0 - 5.00 + 100 * (2002.0 - 2000.80))


def test_equity_close_equals_balance_when_flat():
    bars = synthetic_bars(T("2024-03-04 00:00"), 600, seed=3)
    cm = CostModel(commission_per_lot_round_trip=5.0)
    eq, tr = equity_from_trades(bars, _random_trades(bars, cm, 40, seed=3, max_hold=10), C0, cm)
    flat = eq["units_open"] == 0
    assert flat.sum() > 50 and (~flat).sum() > 50
    assert (eq.loc[flat, "equity_close"] == eq.loc[flat, "balance"]).all()
    eq0, tr0 = equity_from_trades(bars, None, C0, cm)
    assert len(tr0) == 0 and (eq0["balance"] == C0).all() and (eq0["equity_worst"] == C0).all()
    assert (eq0["equity_close"] == C0).all() and (eq0["units_open"] == 0).all()


def test_swap_booked_once_per_held_rollover_and_not_before_17_new_york():
    # January 2024: New York on EST, rollover 17:00 New York = 22:00 UTC; metals bars break 22:00-23:00 UTC,
    # so the 21:00 UTC bar ends exactly at the rollover and carries the swap.
    bars = make_bars("2024-01-08 00:00", "2024-01-12 22:00", {}, market="metals")
    cm = CostModel(swap_long=3.6, swap_short=3.6, triple_swap_weekday=2)
    def one(entry: str, exit_: str, side: int = 1):
        tr = pd.DataFrame({"side": [side], "units": [100.0], "entry_time": [T(entry)], "entry_price": [2000.3],
                           "exit_time": [T(exit_)], "exit_price": [2000.0 if side > 0 else 2000.3],
                           "exit_reason": ["signal"]})
        return equity_from_trades(bars, tr, C0, cm)
    # 3.6 %/year / 360 = 0.01 % per night of 100 oz x 2000 = 20.00 USD per night
    eq, tr = one("2024-01-08 10:00", "2024-01-08 21:59:59")       # closed 1 s before 17:00 New York
    assert tr["swap_usd"].iloc[0] == 0 and (eq["swap_usd"] == 0).all()
    eq, tr = one("2024-01-08 10:00", "2024-01-08 22:00")          # closed AT 17:00 New York: pays it
    assert cents(tr["swap_usd"].iloc[0], -20.00)
    assert cents(row(eq, "2024-01-08 21:00")["swap_usd"], -20.00) and int((eq["swap_usd"] != 0).sum()) == 1
    eq, tr = one("2024-01-08 22:00", "2024-01-09 10:00")          # opened AT the rollover: does not pay it
    assert tr["swap_usd"].iloc[0] == 0
    # Monday 10:00 -> Friday 10:00 UTC holds Mon, Tue, Wed (x3), Thu rollovers: 1 + 1 + 3 + 1 = 6 nights
    eq, tr = one("2024-01-08 10:00", "2024-01-12 10:00", side=-1)
    assert cents(tr["swap_usd"].iloc[0], -6 * 20.00)
    booked = eq.loc[eq["swap_usd"] != 0]
    assert [calendar.utc_str(int(t))[11:16] for t in booked["time"]] == ["21:00"] * 4
    assert np.allclose(booked["swap_usd"], [-20.0, -20.0, -60.0, -20.0])


def test_swap_at_a_rollover_on_a_bar_open_uses_that_open_not_the_later_close():
    """Finding 14: on 24-hour bars 17:00 New York is a bar open. The swap of a trade closed at that open
    is charged on the bid at the rollover (the bar's open); the bar's close, which comes later, must not
    change it (an entry at the same open could otherwise be sized on a future price)."""
    def run(close_of_rollover_bar: float):
        bars = make_bars("2024-01-08 00:00", "2024-01-09 06:00", {
            "2024-01-08 22:00": (2000.0, max(2000.0, close_of_rollover_bar), min(2000.0, close_of_rollover_bar),
                                 close_of_rollover_bar, 0.30)}, market="always")
        cm = CostModel(swap_long=3.6, swap_short=3.6, triple_swap_weekday=2)
        tr = pd.DataFrame({"side": [1], "units": [100.0], "entry_time": [T("2024-01-08 10:00")],
                           "entry_price": [2000.3], "exit_time": [T("2024-01-08 22:00")], "exit_price": [2000.0],
                           "exit_reason": ["signal"]})
        return equity_from_trades(bars, tr, C0, cm)
    eq1, tr1 = run(2000.0)
    eq2, tr2 = run(2100.0)                         # only the rollover bar's close (after its open) moves
    # 3.6 %/year / 360 x 100 oz x 2000.00 (the open at 22:00 UTC = 17:00 New York) = 20.00 USD
    assert cents(tr1["swap_usd"].iloc[0], -20.00) and cents(tr2["swap_usd"].iloc[0], -20.00)
    assert cents(row(eq1, "2024-01-08 22:00")["balance"], row(eq2, "2024-01-08 22:00")["balance"])


def test_rollover_bids_open_at_a_bar_open_else_the_last_close():
    from propkit.bars import rollover_bids
    times = np.array([0, 3600, 7200, 14400], dtype=np.int64)
    o, c = np.array([10.0, 11.0, 12.0, 13.0]), np.array([10.5, 11.5, 12.5, 13.5])
    got = rollover_bids(times, o, c, np.array([3600, 5000, 10800, 14400], dtype=np.int64))
    assert np.allclose(got, [11.0, 11.5, 12.5, 13.0])     # at an open / inside a bar / in a gap / at an open
    assert rollover_bids(times, o, c, np.zeros(0, dtype=np.int64)).size == 0


@pytest.mark.parametrize("market", ["always", "metals"])
def test_us_dst_switch_moves_the_rollover_between_21_and_22_utc(market):
    # 2024-03-10 the US moves to EDT (EU still CET until 03-31): Friday 03-08 rollover at 22:00 UTC, Monday
    # 03-11 at 21:00 UTC. 2024-11-03 the US moves back: Friday 11-01 at 21:00 UTC, Monday 11-04 at 22:00 UTC.
    cm = CostModel(swap_long=3.6, triple_swap_weekday=None)
    for first, entry, exit_, want in (
            ("2024-03-07 00:00", "2024-03-08 12:00", "2024-03-12 12:00",
             ["2024-03-08 22:00", "2024-03-11 21:00"]),
            ("2024-10-31 00:00", "2024-11-01 12:00", "2024-11-05 12:00",
             ["2024-11-01 21:00", "2024-11-04 22:00"])):
        bars = make_bars(first, exit_[:10] + " 20:00", {}, market=market)
        tr = pd.DataFrame({"side": [1], "units": [100.0], "entry_time": [T(entry)], "entry_price": [2000.3],
                           "exit_time": [T(exit_)], "exit_price": [2000.0]})
        eq, out = equity_from_trades(bars, tr, C0, cm)
        booked = eq.loc[eq["swap_usd"] != 0, "time"].to_numpy()
        if market == "always":     # 24-hour bars: the bar opening at the rollover carries it
            assert booked.tolist() == [T(w) for w in want]
        else:                      # metals bars: the bar that ENDS at the rollover carries it
            assert booked.tolist() == [T(w) - 3600 for w in want]
        assert cents(out["swap_usd"].iloc[0], -2 * 20.00)


def test_stop_exit_inside_bar_never_beyond_the_stop_fill_target_exit_takes_the_low():
    bars = make_bars("2024-01-08 00:00", "2024-01-08 05:00", {
        "2024-01-08 02:00": (2000.0, 2001.0, 1980.0, 1985.0, 0.0)}, market="always", spread=0.0)
    cm = CostModel(swap_enabled=False)
    base = {"side": [1], "units": [10.0], "entry_time": [T("2024-01-08 01:00")], "entry_price": [2000.0],
            "exit_time": [T("2024-01-08 02:20")], "exit_price": [1990.0]}
    eq, _ = equity_from_trades(bars, pd.DataFrame({**base, "exit_reason": ["stop"]}), C0, cm)
    assert cents(row(eq, "2024-01-08 02:00")["equity_worst"], C0 - 100.0)   # the stop fill, not the 1980 low
    eq, _ = equity_from_trades(bars, pd.DataFrame({**base, "exit_reason": ["target"]}), C0, cm)
    assert cents(row(eq, "2024-01-08 02:00")["equity_worst"], C0 - 200.0)   # 10 x (1980 - 2000)
    eq, _ = equity_from_trades(bars, pd.DataFrame({**base, "exit_reason": ["unknown"]}), C0, cm)
    assert cents(row(eq, "2024-01-08 02:00")["equity_worst"], C0 - 200.0)


def test_exit_at_the_last_bar_close_is_accepted():
    bars = make_bars("2024-01-08 00:00", "2024-01-08 05:00", {}, market="always")
    tr = pd.DataFrame({"side": [1], "units": [1.0], "entry_time": [T("2024-01-08 01:00")], "entry_price": [2000.3],
                       "exit_time": [T("2024-01-08 05:00")], "exit_price": [2000.0], "exit_reason": ["end_of_data"]})
    eq, out = equity_from_trades(bars, tr, C0, NO_SWAP)
    assert cents(eq["realised_usd"].iloc[-1], -0.30) and eq["units_open"].iloc[-1] == 0


def test_bar_of_instants_and_fills_in_gaps_refused():
    times = np.array([0, 3600, 7200, 18000], dtype=np.int64) + T("2024-01-08 00:00")
    got = bar_of_instants(times, times + np.array([0, 1800, 3600, 3600]), 3600)
    assert got.tolist() == [0, 1, 2, 3]                  # 7200 + 3600 = the end of bar 2 (a gap follows)
    for bad in (times[0] - 1, times[2] + 3601, times[3] + 3601):
        with pytest.raises(ValueError, match="before the first bar|inside a gap|after the end"):
            bar_of_instants(times, np.array([bad]), 3600)
    bars = make_bars("2024-01-12 18:00", "2024-01-15 02:00", {}, market="metals")   # Friday close .. Monday
    tr = pd.DataFrame({"side": [1], "units": [1.0], "entry_time": [T("2024-01-13 12:00")],   # Saturday
                       "entry_price": [2000.3], "exit_time": [T("2024-01-14 23:00")], "exit_price": [2000.0]})
    with pytest.raises(ValueError, match="inside a gap"):
        equity_from_trades(bars, tr, C0, NO_SWAP)


def test_price_tolerance_catches_wrong_scale():
    bars = make_bars("2024-01-08 00:00", "2024-01-08 05:00", {}, market="always")
    tr = pd.DataFrame({"side": [1], "units": [1.0], "entry_time": [T("2024-01-08 01:00")],
                       "entry_price": [200030.0], "exit_time": [T("2024-01-08 02:00")], "exit_price": [200000.0]})
    with pytest.raises(ValueError, match="far outside its bar"):
        equity_from_trades(bars, tr, C0, NO_SWAP)
    eq, _ = equity_from_trades(bars, tr, C0, NO_SWAP, price_tolerance=None)
    assert len(eq) == len(bars)


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), "100000", True])
def test_bad_capital_refused(bad):
    bars = make_bars("2024-01-08 00:00", "2024-01-08 05:00", {}, market="always")
    with pytest.raises(ValueError, match="C0"):
        equity_from_trades(bars, None, bad, NO_SWAP)


def test_risk_usd_flat_rate_and_given_values_kept():
    bars = make_bars("2024-01-08 00:00", "2024-01-08 05:00", {}, market="always")
    cm = CostModel(flat_rate_per_side=0.0005, swap_enabled=False)
    tr = pd.DataFrame({"side": [1, -1], "units": [10.0, 10.0], "entry_time": [T("2024-01-08 01:00")] * 2,
                       "entry_price": [2000.0, 2000.0], "exit_time": [T("2024-01-08 02:00")] * 2,
                       "exit_price": [2000.0, 2000.0], "stop_price": [1990.0, 2010.0], "risk_usd": [np.nan, 55.5]})
    _, out = equity_from_trades(bars, tr, C0, cm)
    # 10 x 10.00 + 0.0005 x 10 x (2000 + 1990) = 100.00 + 19.95
    assert cents(out["risk_usd"].iloc[0], 119.95) and out["risk_usd"].iloc[1] == 55.5


def test_a_trade_stopped_at_its_stop_price_loses_exactly_1r():
    """Finding 7: 1R from a trade list's stop_price includes the exit markup and slippage of the stop fill
    (stop_price is a bid level for a long, an ask level for a short), so a stop exit is -1.000R, not -1.008R."""
    bars = make_bars("2024-01-08 00:00", "2024-01-08 05:00", {}, market="always")
    cm = CostModel(markup_per_side=0.05, slippage_per_side=0.03, commission_per_lot_round_trip=7.0,
                   swap_enabled=False)
    stop_long, stop_short = 1990.0, 2010.30          # long: bid level; short: ask level
    tr = pd.DataFrame({"side": [1, -1], "units": [100.0, 100.0], "entry_time": [T("2024-01-08 01:00")] * 2,
                       "entry_price": [2000.38, 1999.92], "exit_time": [T("2024-01-08 02:00")] * 2,
                       "exit_price": [stop_long - 0.08, stop_short + 0.08], "exit_reason": ["stop", "stop"],
                       "stop_price": [stop_long, stop_short]})
    _, out = equity_from_trades(bars, tr, C0, cm, price_tolerance=None)
    # long: 100 x (2000.38 - 1989.92) + 7 = 1053.00; short: 100 x (2010.38 - 1999.92) + 7 = 1053.00
    assert cents(out["risk_usd"].iloc[0], 1053.00) and cents(out["risk_usd"].iloc[1], 1053.00)
    r = out["pnl_usd"] / out["risk_usd"]
    assert np.allclose(r, -1.0, atol=1e-12)


def test_swap_in_usd_per_lot_per_night_and_inputs_not_modified():
    bars = make_bars("2024-01-08 00:00", "2024-01-12 22:00", {}, market="metals")
    cm = CostModel(swap_unit="usd_per_lot_per_night", swap_long=25.0, swap_short=-4.0, triple_swap_weekday=2)
    tr_in = pd.DataFrame({"side": [1, -1], "units": [200.0, 50.0],
                          "entry_time": [T("2024-01-10 10:00"), T("2024-01-10 10:00")],
                          "entry_price": [2000.3, 2000.0], "exit_time": [T("2024-01-11 10:00")] * 2,
                          "exit_price": [2000.0, 2000.3]})
    bars_before, trades_before = bars.copy(), tr_in.copy()
    eq, tr = equity_from_trades(bars, tr_in, C0, cm)
    # Wednesday rollover, 3 nights: long 2 lots pays 25 x 2 x 3 = 150.00; short 0.5 lot receives 4 x 0.5 x 3
    assert cents(tr["swap_usd"].iloc[0], -150.00) and cents(tr["swap_usd"].iloc[1], 6.00)
    assert cents(row(eq, "2024-01-10 21:00")["swap_usd"], -144.00)
    pd.testing.assert_frame_equal(bars, bars_before)
    pd.testing.assert_frame_equal(tr_in, trades_before)
    _check_invariants(eq, tr)


def test_swap_disabled_and_commission_ledger():
    bars = synthetic_bars(T("2024-05-06 00:00"), 400, seed=9)
    cm = CostModel(swap_enabled=False, commission_per_lot_round_trip=8.0)
    eq, tr = equity_from_trades(bars, _random_trades(bars, cm, 50, seed=9), C0, cm)
    assert (eq["swap_usd"] == 0).all() and (tr["swap_usd"] == 0).all()
    assert np.allclose(tr["commission_usd"], 8.0 * tr["units"] / 100.0)


# ---------------------------------------------------------------------------------------
# positions

def test_equity_from_positions_units_mode_known_answer():
    bars = make_bars("2024-01-08 00:00", "2024-01-08 06:00", {
        "2024-01-08 01:00": (2000.0, 2000.0, 2000.0, 2000.0, 0.20),
        "2024-01-08 02:00": (2010.0, 2010.0, 2010.0, 2010.0, 0.20),
        "2024-01-08 03:00": (2020.0, 2020.0, 2020.0, 2020.0, 0.20),
        "2024-01-08 04:00": (2005.0, 2005.0, 2005.0, 2005.0, 0.20),
        "2024-01-08 05:00": (2001.0, 2001.0, 2001.0, 2001.0, 0.20)}, market="always")
    pos = pd.DataFrame({"time": bars["time"], "position": [0.0, 1.0, 1.0, 0.5, -0.5, -0.5]})
    eq, tr = equity_from_positions(bars, pos, C0, NO_SWAP, size_mode="units", size=10.0)
    # bar 1: buy 10 oz at 2000.20; bar 3: sell 5 at 2020.00 (FIFO part of the lot); bar 4: sell 5 at 2005.00
    # (closes the rest) and sell 5 more (short entry at 2005.00); end of data: buy 5 at 2001 + 0.20 = 2001.20
    assert tr[["side", "units", "entry_price", "exit_price", "exit_reason"]].values.tolist() == [
        [1, 5.0, 2000.2, 2020.0, "signal"], [1, 5.0, 2000.2, 2005.0, "signal"],
        [-1, 5.0, 2005.0, 2001.2, "end_of_data"]]
    assert tr["exit_time"].tolist() == [T("2024-01-08 03:00"), T("2024-01-08 04:00"), T("2024-01-08 06:00")]
    # 5 x 19.80 + 5 x 4.80 + 5 x 3.80 = 99.00 + 24.00 + 19.00 = 142.00
    assert cents(eq["balance"].iloc[-1], C0 + 142.00)
    assert eq["units_open"].tolist() == [0.0, 10.0, 10.0, 5.0, -5.0, 0.0]


def test_leverage_mode_resizes_only_on_position_changes():
    bars = synthetic_bars(T("2024-04-01 00:00"), 300, seed=11)
    cm = CostModel(commission_per_lot_round_trip=7.0, markup_per_side=0.02, swap_long=4.0, swap_short=2.0)
    p = np.zeros(len(bars))
    p[10:60] = 1.0
    p[60:90] = 0.4
    p[90:140] = -0.8
    p[140:150] = -0.3
    p[200:300] = 0.6
    pos = pd.DataFrame({"time": bars["time"], "position": p})
    size = 2.0
    eq, tr = equity_from_positions(bars, pos, C0, cm, size_mode="leverage", size=size)
    times, o = bars["time"].to_numpy(), bars["open"].to_numpy()
    spread = cm.bar_spreads(bars)
    changes = np.flatnonzero(np.diff(np.r_[0.0, p]) != 0)
    assert changes.tolist() == [10, 60, 90, 140, 150, 200]
    # every entry and every non-final exit is at a change bar's open: nothing happens in between
    fills = set(tr["entry_time"]) | set(tr.loc[tr["exit_reason"] == "signal", "exit_time"])
    assert fills == {int(times[k]) for k in changes}
    et, xt = tr["entry_time"].to_numpy(), tr["exit_time"].to_numpy()
    signed = (tr["side"] * tr["units"]).to_numpy()
    for k in changes:
        tk = int(times[k])
        # equity at the open of bar k, before the change, from the returned EQUITY and TRADES
        before = (et < tk) & (xt >= tk)
        mark = np.where(tr["side"].to_numpy() > 0, o[k], o[k] + spread[k])
        unreal = float(np.sum((signed * (mark - tr["entry_price"].to_numpy()))[before]))
        equity_open = float(eq["balance"].iloc[k - 1]) + unreal
        held_after = float(np.sum(signed[(et <= tk) & (xt > tk)]))
        assert np.isclose(held_after, p[k] * size * equity_open / o[k], rtol=1e-12, atol=1e-9), k
    # while the position is unchanged the units stay fixed, although equity moves
    assert (eq["units_open"].iloc[10:60].nunique() == 1) and (eq["units_open"].iloc[200:299].nunique() == 1)
    assert eq["equity_close"].iloc[10:60].nunique() > 10
    _check_invariants(eq, tr)


def test_timing_63645_bars_5000_trades_under_10_seconds():
    bars = synthetic_bars(T("2015-01-05 00:00"), 63_645, seed=1)
    cm = CostModel(markup_per_side=0.02, commission_per_lot_round_trip=7.0)
    trades = _random_trades(bars, cm, 5000, seed=1, max_hold=48)
    rng = np.random.default_rng(1)
    levels = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], len(bars))
    keep = rng.random(len(bars)) < 0.8                       # about one change in five bars
    p = pd.Series(np.where(keep, np.nan, levels)).ffill().fillna(0.0).to_numpy()
    pos = pd.DataFrame({"time": bars["time"], "position": p})
    t0 = time.perf_counter()
    eq, tr = equity_from_trades(bars, trades, C0, cm)
    eq2, tr2 = equity_from_positions(bars, pos, C0, cm, size_mode="leverage", size=1.0)
    elapsed = time.perf_counter() - t0
    if elapsed > 60:
        pytest.skip(f"machine too slow for the timing guard ({elapsed:.0f} s)")
    assert elapsed < 10, f"{elapsed:.1f} s"
    assert len(eq) == 63_645 and len(tr) == 5000 and len(tr2) > 5000
    _check_invariants(eq, tr)
    _check_invariants(eq2, tr2)


def test_validate_equity_refuses_bad_tables(hand):
    eq, _ = hand
    with pytest.raises(ValueError, match="missing column"):
        validate_equity(eq.drop(columns=["equity_worst"]))
    bad = eq.copy()
    bad.loc[5, "equity_worst"] = bad.loc[5, "equity_close"] + 1
    with pytest.raises(ValueError, match="equity_worst above"):
        validate_equity(bad)
    bad = eq.copy()
    bad.loc[3, "balance"] = np.nan
    with pytest.raises(ValueError, match="missing or infinite balance"):
        validate_equity(bad)
