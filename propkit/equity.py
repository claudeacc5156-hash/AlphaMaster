"""propkit/equity.py - the bar-by-bar account path (EQUITY) of a trade list or a held-position series.

EQUITY (pandas DataFrame), one row per bar of BARS, first bar to last (EQUITY_COLUMNS):
  time            int64, the bar OPEN (UTC epoch seconds);
  balance         USD at the END of the bar: C0 + every realised price PnL, minus every commission, plus
                  every swap booked so far (balance[k] = balance[k-1] + realised_usd[k] - commission_usd[k]
                  + swap_usd[k], balance[-1] = C0);
  equity_close    USD: balance + the open positions marked at the bar close (longs at the BID close,
                  shorts at the ASK close = bid close + the bar's spread). The exit commission of an open
                  position is not deducted before it is paid. Equals balance when flat;
  equity_worst    USD, a lower bound of the lowest equity reached in the bar (see below);
  units_open      signed oz open at the bar close (+ long, - short);
  realised_usd    USD, gross price PnL side x units x (exit_price - entry_price) of the trades closed in
                  the bar (commission and swap are NOT in it; they have their own columns);
  commission_usd  USD booked in the bar, POSITIVE = paid: each fill pays CostModel.fill_commission (half the
                  per-lot round trip, or flat_rate_per_side x fill notional) in the bar of the fill;
  swap_usd        USD booked in the bar, signed, NEGATIVE = paid.

Which bar an instant belongs to: the last bar whose open <= t, and t must be before that bar's end
(open + bar_seconds) or exactly at it (a fill at the bar's close before a gap, e.g. the data end).
Instants inside a gap (a weekend, the daily break, a missing bar) are refused.

Swap: charged once for every rollover instant (rollover_hour_ny New York time, Monday..Friday by New York
weekday, US DST rule: 17:00 New York = 21:00 UTC in US summer time, 22:00 UTC in winter) with
entry_time < r <= exit_time (opened strictly before, closed at or after); x3 nights when the New York
weekday of r is triple_swap_weekday (the weekend has no rollover of its own); the notional uses the bid
at r (bars.rollover_bids): the OPEN of a bar that opens exactly at r (24-hour bars), else the close of
the last bar with open <= r (metals bars: the bar ending at 17:00 New York); booked in the last bar with
open <= r. See CostModel.swap_for_night.

equity_worst[k] is the lower of
  (a) the level the bar starts from: equity_close[k-1] (C0 before the first bar) minus the commission
      booked in bar k and minus the swap COSTS booked in bar k (a cost booked in a bar may be paid before
      any price move); and
  (b) balance[k-1] - commission_usd[k] + swap costs of bar k + the worst result of every position that
      is open at any point in the bar:
        - open at the bar close (held over or opened in the bar): longs marked at the BID LOW, shorts at
          the ASK HIGH (bid high + spread), against their own entry fill (a position opened in the bar is
          marked from its fill; the intrabar order of prices is unknown, so it is assumed to see the
          bar's whole adverse extreme after its fill - conservative);
        - closed in the bar: its realised result when it closed at the bar's open (no exposure inside
          the bar) or by a stop ("stop" or "trail": the price cannot have gone beyond the stop before the
          stop filled, so never beyond the stop fill); otherwise the lower of its realised result and
          its mark at the bar's adverse extreme.
  Swap CREDITS (positive swaps) are left out of the worst case. Long and short positions held together
  are each taken at their own worst (conservative). Hence equity_worst[k] <= equity_close[k] and
  equity_worst[k] <= equity_close[k-1] - commission_usd[k] + min(swap_usd[k], 0).

One spread per bar (BARS.spread, or the fixed spread; CostModel.bar_spreads) is used for every fill and
every ask-side mark inside that bar: an approximation, the real spread moves within the bar. A caller with
real ask bars can pass them (equity_from_trades(..., ask_prices=)): shorts are then marked at the ask close
and their worst at the ask high (propkit.zeno_v1 does, for D14).
Research only: nothing here places or prepares orders.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from propkit import adapters
from propkit import bars as bars_mod
from propkit import calendar
from propkit.costs import CostModel

EQUITY_COLUMNS = adapters.EQUITY_COLUMNS
STOP_EXITS = ("stop", "trail")
DEFAULT_PRICE_TOLERANCE = 0.01


def _capital(C0) -> float:
    if isinstance(C0, (bool, np.bool_)) or not isinstance(C0, (int, float, np.integer, np.floating)):
        raise ValueError(f"C0 (initial capital, USD) must be a number, got {C0!r}")
    x = float(C0)
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"C0 (initial capital, USD) must be a finite number > 0, got {C0!r}")
    return x


def planned_stop_fill(cost_model: CostModel, side, stop_price) -> np.ndarray:
    """Fill of a stop exit at the stop LEVEL, USD/oz: a long's stop is a bid level and sells at stop -
    markup - slippage; a short's stop is an ask level and buys at stop + markup + slippage (the spread is
    already in an ask level). Under a flat rate the fill is the level itself (the fee is the commission).
    side: +1 / -1 per trade; stop_price: USD/oz. Returns a float array."""
    side = np.asarray(side, dtype=np.float64)
    stop = np.asarray(stop_price, dtype=np.float64)
    adj = 0.0 if cost_model.is_flat else cost_model.markup_per_side + cost_model.slippage_per_side
    return stop - side * adj


def bar_of_instants(times: np.ndarray, t: np.ndarray, bar_seconds: int, what: str = "fill time") -> np.ndarray:
    """Bar index of each instant: the last bar with open <= t, provided t <= that bar's open + bar_seconds
    (t equal to the end is a fill at the bar's close, allowed only when no bar opens there).

    times: sorted bar opens (UTC epoch seconds); t: int64 UTC epoch seconds. Raises ValueError naming the
    first instant that is before the first bar, after the last bar's end, or inside a gap.
    """
    t = np.asarray(t, dtype=np.int64)
    idx = np.searchsorted(times, t, side="right") - 1
    ok = (idx >= 0) & (t <= times[np.maximum(idx, 0)] + bar_seconds)
    if not ok.all():
        j = int(np.flatnonzero(~ok)[0])
        tj = int(t.ravel()[j])
        if idx.ravel()[j] < 0:
            where = "before the first bar"
        elif tj > int(times[-1]) + bar_seconds:
            where = "after the end of the last bar"
        else:
            where = "inside a gap between bars (weekend, daily break or a missing bar)"
        raise ValueError(f"{what} {calendar.utc_str(tj)} is {where}; every fill must be at a bar open or inside "
                         f"a bar of the data (bars {calendar.utc_str(int(times[0]))} to "
                         f"{calendar.utc_str(int(times[-1]))}, bar size {bar_seconds} s)")
    return idx.astype(np.int64)


def _check_prices(price: np.ndarray, bar: np.ndarray, lo: np.ndarray, hi: np.ndarray, spread: np.ndarray,
                  tol: float | None, what: str, t: np.ndarray) -> None:
    if tol is None or price.size == 0:
        return
    lower = lo[bar] * (1.0 - tol)
    upper = (hi[bar] + spread[bar]) * (1.0 + tol)
    bad = (price < lower) | (price > upper)
    if bad.any():
        j = int(np.flatnonzero(bad)[0])
        raise ValueError(f"trade row {j}: {what} {price[j]:g} at {calendar.utc_str(int(t[j]))} is far outside its "
                         f"bar (bid low {lo[bar[j]]:g}, ask high {hi[bar[j]] + spread[bar[j]]:g}, tolerance "
                         f"{tol:.0%}); check that the trades and the bars are the same instrument, price scale "
                         "and time zone (UTC), or switch the check off with price_tolerance=None (on the command "
                         "line: --price-tolerance none)")


def _held_sums(start: np.ndarray, stop: np.ndarray, weight: np.ndarray, n: int) -> np.ndarray:
    """For each bar k, the sum of weight over items with start <= k < stop (difference array + cumsum,
    with the running sum reset to exactly 0 at bars where nothing is held, so rounding cannot drift)."""
    if start.size == 0:
        return np.zeros(n)
    d = np.bincount(start, weight, minlength=n + 1) - np.bincount(stop, weight, minlength=n + 1)
    cnt = np.cumsum(np.bincount(start, minlength=n + 1) - np.bincount(stop, minlength=n + 1))[:n]
    s = np.cumsum(d)[:n]
    flat = cnt == 0
    last_flat = np.maximum.accumulate(np.where(flat, np.arange(n), -1))
    base = np.where(last_flat >= 0, s[np.maximum(last_flat, 0)], 0.0)
    out = s - base
    out[flat] = 0.0
    return out


def _swap_events(trade_entry: np.ndarray, trade_exit: np.ndarray, times: np.ndarray, bar_seconds: int,
                 hour_ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every (trade, rollover) pair with entry < r <= exit: (trade index, rollover instant, bar index)."""
    r = calendar.rollover_instants(int(times[0]), int(times[-1]) + bar_seconds + 1, hour_ny)
    if r.size == 0 or trade_entry.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    lo = np.searchsorted(r, trade_entry, side="right")
    hi = np.searchsorted(r, trade_exit, side="right")
    counts = np.maximum(hi - lo, 0)
    total = int(counts.sum())
    trade_idx = np.repeat(np.arange(trade_entry.size, dtype=np.int64), counts)
    first = np.repeat(lo, counts)
    offset = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    instants = r[first + offset]
    bar = np.searchsorted(times, instants, side="right") - 1
    return trade_idx, instants, bar.astype(np.int64)


def equity_from_trades(bars: pd.DataFrame, trades: pd.DataFrame | None, C0: float, cost_model: CostModel,
                       price_tolerance: float | None = DEFAULT_PRICE_TOLERANCE,
                       ask_prices: Mapping[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The account path of a trade list: returns (EQUITY, TRADES with commission, swap and PnL filled).

    bars: BARS (BID prices, USD/oz; validated again here); trades: TRADES (see propkit.adapters; entry and
    exit prices are the ACTUAL fills, so spread, markup and slippage are already in them and are not
    charged again); C0: initial capital, USD; cost_model: CostModel (commission per fill, swap, the bar
    spreads used for ask-side marks).
    Filled per trade (USD): commission_usd = fill_commission at entry + at exit (positive = paid);
    swap_usd = the sum of CostModel.swap_for_night over the held rollovers (negative = paid);
    pnl_usd = side x units x (exit_price - entry_price) - commission_usd + swap_usd; risk_usd, when NaN and
    a stop_price is given, = the loss of a stop exit at the planned size: units x |entry_price - stop_fill|
    + CostModel.commission(units, entry_price, stop_fill), where stop_fill = planned_stop_fill (stop_price
    is a BID level for a long, an ASK level for a short; the exit markup and slippage move the fill
    beyond it), so a trade stopped at its stop_price comes out at -1R. Values given in the input for
    commission_usd, swap_usd and pnl_usd are replaced.
    price_tolerance: each fill must lie within [bid low x (1 - tol), (bid high + spread) x (1 + tol)] of
    its bar (catches wrong instruments, price scales or time zones); None switches the check off.
    ask_prices: optional, the ACTUAL ask per bar, {"high": ..., "close": ...} (USD/oz, one finite value per
    bar of `bars`). Given, a short open at a bar's close is marked at that bar's ask close, and its worst
    inside a bar (open at the close, or closed in the bar with exposure) at max(ask high, ask close),
    instead of bid + the bar's one spread; use it when the spread moves inside a bar (data with its own
    ask bars). None (the default) keeps bid + spread, so every existing result is unchanged.
    See the module docstring for the EQUITY columns, the bar of an instant, swap and equity_worst.
    Fully vectorised (no loop over bars or trades).
    """
    if not isinstance(cost_model, CostModel):
        raise ValueError("cost_model must be a propkit.costs.CostModel")
    C0 = _capital(C0)
    if price_tolerance is not None:
        price_tolerance = adapters._finite_number(price_tolerance, "price_tolerance")
        if price_tolerance < 0:
            raise ValueError("price_tolerance must be >= 0 (a fraction such as 0.01), or None")
    b = bars_mod.validate_bars(bars, source="bars")
    tr = adapters.validate_trades(trades, allow_extra=True)
    times = b["time"].to_numpy(dtype=np.int64)
    o, h, lo, c = (b[col].to_numpy(dtype=np.float64) for col in ("open", "high", "low", "close"))
    spread = cost_model.bar_spreads(b)
    n = len(times)
    bar_seconds = bars_mod.infer_bar_seconds(times)
    if ask_prices is None:
        ask_c, ask_h = c + spread, h + spread                  # the ask = bid + the bar's one spread
    else:
        ask_c, ask_h = _ask_marks(ask_prices, n)

    side = tr["side"].to_numpy(dtype=np.int64)
    units = tr["units"].to_numpy(dtype=np.float64)
    et = tr["entry_time"].to_numpy(dtype=np.int64)
    xt = tr["exit_time"].to_numpy(dtype=np.int64)
    ep = tr["entry_price"].to_numpy(dtype=np.float64)
    xp = tr["exit_price"].to_numpy(dtype=np.float64)
    reason = tr["exit_reason"].to_numpy(dtype=object)
    e = bar_of_instants(times, et, bar_seconds, "entry_time")
    x = bar_of_instants(times, xt, bar_seconds, "exit_time")
    _check_prices(ep, e, lo, h, spread, price_tolerance, "entry_price", et)
    _check_prices(xp, x, lo, h, spread, price_tolerance, "exit_price", xt)

    gross = side * units * (xp - ep)
    if len(tr):
        entry_comm = np.asarray(cost_model.fill_commission(units, ep), dtype=np.float64)
        exit_comm = np.asarray(cost_model.fill_commission(units, xp), dtype=np.float64)
    else:
        entry_comm = exit_comm = np.zeros(0)
    ti, r_inst, r_bar = _swap_events(et, xt, times, bar_seconds, cost_model.rollover_hour_ny)
    if ti.size:
        weekday = np.asarray(calendar.ny_weekday(r_inst), dtype=np.int64)
        r_bid = bars_mod.rollover_bids(times, o, c, r_inst)
        sw = np.asarray(cost_model.swap_for_night(side[ti], units[ti], r_bid, weekday), dtype=np.float64)
    else:
        sw = np.zeros(0)
    trade_swap = np.bincount(ti, sw, minlength=len(tr)) if len(tr) else np.zeros(0)

    realised = np.bincount(x, gross, minlength=n)
    commission = np.bincount(e, entry_comm, minlength=n) + np.bincount(x, exit_comm, minlength=n)
    swap = np.bincount(r_bar, sw, minlength=n)
    swap_cost = np.bincount(r_bar, np.minimum(sw, 0.0), minlength=n)
    balance = C0 + np.cumsum(realised - commission + swap)

    # positions open at a bar's close: entry bar <= k < exit bar
    is_long = side > 0
    u_long = _held_sums(e[is_long], x[is_long], units[is_long], n)
    v_long = _held_sums(e[is_long], x[is_long], (units * ep)[is_long], n)
    u_short = _held_sums(e[~is_long], x[~is_long], units[~is_long], n)
    v_short = _held_sums(e[~is_long], x[~is_long], (units * ep)[~is_long], n)
    equity_close = balance + (c * u_long - v_long) + (v_short - ask_c * u_short)
    open_worst = (lo * u_long - v_long) + (v_short - ask_h * u_short)

    # positions closed in a bar
    adverse = np.where(is_long, units * (lo[x] - ep), units * (ep - ask_h[x]))
    no_exposure = (xt == times[x]) | np.isin(reason, STOP_EXITS)
    closed = np.where(no_exposure, gross, np.minimum(gross, adverse))
    closed_worst = np.bincount(x, closed, minlength=n)

    balance_prev = np.concatenate(([C0], balance[:-1]))
    equity_prev = np.concatenate(([C0], equity_close[:-1]))
    intrabar = balance_prev + open_worst + closed_worst - commission + swap_cost
    start = equity_prev - commission + swap_cost
    equity_worst = np.minimum(start, intrabar)

    equity = pd.DataFrame({
        "time": times,
        "balance": balance,
        "equity_close": equity_close,
        "equity_worst": equity_worst,
        "units_open": u_long - u_short,
        "realised_usd": realised,
        "commission_usd": commission,
        "swap_usd": swap,
    })
    out = tr.copy()
    out["commission_usd"] = entry_comm + exit_comm
    out["swap_usd"] = trade_swap
    out["pnl_usd"] = gross - (entry_comm + exit_comm) + trade_swap
    stop = out["stop_price"].to_numpy(dtype=np.float64)
    need = np.isfinite(stop) & ~np.isfinite(out["risk_usd"].to_numpy(dtype=np.float64))
    if need.any():
        risk = out["risk_usd"].to_numpy(dtype=np.float64).copy()
        stop_fill = planned_stop_fill(cost_model, side[need], stop[need])
        planned = cost_model.commission(units[need], ep[need], stop_fill) if cost_model.is_flat \
            else cost_model.commission(units[need])
        risk[need] = units[need] * np.abs(ep[need] - stop_fill) + np.asarray(planned, dtype=np.float64)
        out["risk_usd"] = risk
    return equity, out


def _ask_marks(ask_prices: Mapping[str, Any], n: int) -> tuple[np.ndarray, np.ndarray]:
    """(ask close, worst ask = max(ask high, ask close)) per bar from equity_from_trades' ask_prices."""
    if not isinstance(ask_prices, Mapping) or set(ask_prices) != {"high", "close"}:
        raise ValueError('ask_prices must be a dict {"high": array, "close": array} of the ask per bar, or None')
    out = []
    for key in ("close", "high"):
        a = np.asarray(ask_prices[key], dtype=np.float64)
        if a.shape != (n,):
            raise ValueError(f"ask_prices['{key}'] must have one value per bar ({n}), got shape {a.shape}")
        if not np.isfinite(a).all():
            raise ValueError(f"ask_prices['{key}'] has a missing or infinite value")
        out.append(a)
    return out[0], np.maximum(out[1], out[0])


def equity_from_positions(bars: pd.DataFrame, positions: pd.DataFrame, C0: float, cost_model: CostModel,
                          size_mode: str = "units", size: float = 1.0,
                          price_tolerance: float | None = DEFAULT_PRICE_TOLERANCE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The account path of a held-position series: returns (EQUITY, TRADES).

    positions: POSITIONS (time = bar open, position in [-1, 1] HELD during that bar; for AlphaMaster's p use
    adapters.positions_from_alphamaster first, which shifts by one bar: held[t+1] = p[t]). size_mode
    "units" (position 1.0 = `size` oz) or "leverage" (position 1.0 = a notional of size x current equity,
    re-sized only when the position changes). The series is turned into TRADES by
    adapters.trades_from_positions (changes fill at bar opens, FIFO lots, partial reductions realise PnL
    pro rata; an open position is closed at the end of the data and flagged "end_of_data"), then
    equity_from_trades builds the path. C0 in USD.
    """
    trades = adapters.trades_from_positions(bars, positions, cost_model, size_mode=size_mode, size=size, C0=C0)
    return equity_from_trades(bars, trades, C0, cost_model, price_tolerance=price_tolerance)


def validate_equity(equity: pd.DataFrame, source: str = "equity") -> pd.DataFrame:
    """Check an EQUITY table (columns EQUITY_COLUMNS, finite, time strictly increasing, equity_worst <=
    equity_close) and return a copy with the standard dtypes. Raises ValueError naming the first bad row."""
    if not isinstance(equity, pd.DataFrame):
        raise ValueError(f"{source}: expected an EQUITY DataFrame")
    missing = [col for col in EQUITY_COLUMNS if col not in equity.columns]
    if missing:
        raise ValueError(f"{source} is missing column(s) {missing}; build it with propkit.equity")
    if len(equity) == 0:
        raise ValueError(f"{source} has no rows")
    out = pd.DataFrame({"time": bars_mod.to_epoch_seconds(equity["time"], f"{source} time")})
    for col in EQUITY_COLUMNS[1:]:
        out[col] = adapters._float_col(equity, col, source)
        adapters._first_bad(~np.isfinite(out[col].to_numpy()), source, f"has a missing or infinite {col}")
    t = out["time"].to_numpy()
    adapters._first_bad(np.r_[False, np.diff(t) <= 0], source, "has a time not after the row before")
    adapters._first_bad(out["equity_worst"].to_numpy() > out["equity_close"].to_numpy() + 1e-9, source,
                        "has equity_worst above equity_close")
    return out
