"""propkit/bootstrap.py - Monte Carlo of prop challenges by resampling whole prop days. Research only.

Why whole days: losses cluster inside days (news, trend days, one bad session), and the daily-loss rule
is about exactly that clustering. Shuffling single trades spreads a bad day's losses over many days and
understates P(daily breach); resampling whole prop days keeps each day's intraday path intact.

Day units (build_day_units). From a historical EQUITY path (see propkit.evaluator for the frame), each
prop day with bars (00:00 CE(S)T to 00:00 CE(S)T) becomes one unit:
  u_start     = E_00:00 - B_00:00, the floating PnL at the day boundary (USD);
  d_close[k]  = equity_close of the day's bar k - E_00:00 (USD);
  d_worst[k]  = equity_worst of bar k - E_00:00 (USD);
  d_balance[k]= balance at the end of bar k - B_00:00 (USD);
  flat[k]     = no position open at the close of bar k (a flat bar as in propkit.evaluator: units_open 0
                and equity_close = balance, so a net-zero hedge is not flat); entered[k] = a trade entry at
                or before bar k;
  traded      = the day had a trade entry (a trading day).
(Arrays are padded to the longest day with the day's last values, which never changes a result.)

  flat_start  = no position open at the day boundary (the close of the previous day's last bar; the
                account is flat at C0 before the first bar), so u_start = 0 and B_00:00 = E_00:00.

Flat-to-flat blocks (why a day is not always a unit on its own). A position held over midnight ties a
day to the days before it: its u_start (the floating PnL carried in) only exists because of the earlier
days of the same hold, and the daily floor of that day hangs on the day-start BALANCE (B = E - u_start).
Drawing such a day on its own, after unrelated days, invents a position that never existed and turns one
multi-day losing hold into many separate chances to breach (an iid-day bootstrap of a strategy that holds
over midnight overstates P(daily breach) badly). The history is therefore cut only at prop-day boundaries
where the account is FLAT (flat_start): a block is a run of consecutive prop days from one flat boundary
to the next, and blocks are drawn with replacement and chained. Every block starts flat (B = E), every
simulated challenge starts flat at C0, and inside a block the days follow each other as they did in
history, so the balance chain B = E - u_start is the historical one. The flat boundaries are regeneration
points of the strategy (the regenerative block bootstrap: Athreya and Fuh 1992; Datta and McCormick
1993). For a strategy that is flat at every 00:00 CE(S)T each block is one day and "days" is the plain
iid-day bootstrap, with exactly the same random draws as before. For a strategy that is almost never
flat at midnight there are few, long blocks: the simulations then re-use few historical starting points,
DayUnits.block_summary() and BootstrapResult.n_blocks say how many, and the report warns.

A simulated challenge chains drawn units. Equity carries over from day to day; the drawn day's own
u_start then gives the day-start balance (B = E - u_start; 0 at every block start) and every rule is
applied as in propkit.evaluator.evaluate_path: the day-start reference, the daily floor and the trailing
or static max floor, a breach check at every bar (equity_worst vs the higher floor, kind = the higher
floor), and the pass check at every bar close (closed balance >= target, flat if required, best-day rule
on the day profits so far, trading days >= min_trading_days). A breach in the same bar as a pass wins.
If the history ENDS with a position open, the next block starts flat at the carried equity, as if that
position had been closed at the last bar close (exit costs ignored).

Modes:
  "days"   (default) flat-to-flat day blocks (see above), drawn uniformly with replacement; one day each
           for a strategy that is flat at every day boundary;
  "weeks"  blocks of whole market weeks, cut only at week starts where the account is flat (a hold over
           the weekend joins two weeks), drawn with replacement and chained, keeping day-to-day dependence
           inside a week. A week is Monday..Friday (Saturday too if it has bars); a Sunday prop day, which
           only exists as the reopen hour in the US-only DST shift weeks, belongs to the FOLLOWING
           Monday's week because it is that week's first session (see week_id). Partial weeks at the
           start and end of the data are blocks like any other;
  "trades" for comparison only: each simulated day takes the trade count of a randomly drawn historical
           day (trades counted by the prop day of exit_time) and fills it with trades drawn with
           replacement from all TRADES (pnl_usd), each closed before the next opens; the intraday low is
           taken at trade closes only. This breaks the clustering of losses inside days;
  "replay" the historical days in their order (one simulation; a consistency check against
           evaluate_path).
Horizon (contract: "horizon in trading days"): horizon_unit "trading" (default) ends a challenge after
horizon_days simulated prop days WITH A TRADE ENTRY (a trading day, as min_trading_days counts them), the
day that reaches it included; horizon_unit "market" counts every simulated prop day with bars instead.
Either way a challenge is cut at 2,520 simulated market days (about 10 years); None (or "none") runs until
pass or breach within that cap. Unfinished challenges count as timeouts.

size_multiplier m scales every USD increment (u_start, d_close, d_worst, d_balance): position sizes and
every cost (spread, commission, swap) are linear in units, so the PnL of m x the size is m x the PnL.
Rounding to the broker's lot step is ignored.

Randomness: numpy Generator seeded by (seed, simulated day number); the draws of a simulated day are
the same whatever the size multiplier (common random numbers), so max_size compares sizes on identical
histories. Each probability comes with its Monte Carlo error sqrt(p (1 - p) / n_sims) (field se_*): that
is SIMULATION NOISE ONLY, which more simulations shrink at will. It says nothing about how much the answer
depends on the particular history (a few hundred days); history_uncertainty() measures that by resampling
the history's blocks themselves and re-running the bootstrap (an outer bootstrap). When p = 0 or 1 the MC
error is 0; read it as p < 3 / n_sims (or > 1 - 3 / n_sims) at 95% (rule of three).
Quantiles of days to target use numpy's default linear interpolation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from propkit import calendar
from propkit.evaluator import entry_flags, equity_arrays
from propkit.rules import BEST_DAY_TOL_USD, PropRules

MODES = ("days", "weeks", "trades", "replay")
HORIZON_UNITS = ("trading", "market")
DEFAULT_N_SIMS = 10_000
DEFAULT_SEED = 7
DEFAULT_HORIZON_DAYS = 60
DEFAULT_HORIZON_UNIT = "trading"
MAX_DAYS_UNLIMITED = 2520
FEW_BLOCKS = 30             # fewer flat-to-flat blocks than this: the report warns (few starting points)
DEFAULT_HISTORY_REPS = 30   # outer bootstrap replicates of history_uncertainty
HISTORY_SIMS = 500          # inner simulations per outer replicate
QUANTILES = (10, 25, 50, 75, 90)
DEFAULT_GRID = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5,
                1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0)
RUNNING, PASSED, BREACHED_DAILY, BREACHED_MAX = 0, 1, 2, 3
STATUS_NAMES = {RUNNING: "timeout", PASSED: "passed", BREACHED_DAILY: "breached_daily",
                BREACHED_MAX: "breached_max"}
_NEVER = np.iinfo(np.int64).max
_STREAM_DRAWS = 1
_STREAM_OUTER = 2


# ---------------------------------------------------------------------------------------
# day units

@dataclass(frozen=True)
class DayUnits:
    """Prop-day units built from one EQUITY path (USD; see the module docstring).

    Shapes: n = number of prop days with bars, L = bars in the longest day. trade_pnl / trade_count
    are filled only when TRADES were given (needed by mode "trades"). flat_start (bool per day): no
    position open at the day boundary; None (hand-built units) means u_start == 0.
    """

    initial_capital: float
    day: np.ndarray
    n_bars: np.ndarray
    u_start: np.ndarray
    d_close: np.ndarray
    d_worst: np.ndarray
    d_balance: np.ndarray
    flat: np.ndarray
    entered: np.ndarray
    traded: np.ndarray
    week: np.ndarray
    trade_pnl: np.ndarray | None = None
    trade_count: np.ndarray | None = None
    flat_start: np.ndarray | None = None

    @property
    def n_units(self) -> int:
        """Number of prop days with bars."""
        return int(self.day.size)

    @property
    def starts_flat(self) -> np.ndarray:
        """Bool per day: the account is flat at the day boundary (the first day always is)."""
        out = (self.u_start == 0) if self.flat_start is None else np.asarray(self.flat_start, dtype=bool).copy()
        if out.size:
            out[0] = True
        return out

    @property
    def share_open_at_start(self) -> float:
        """Share of prop days that start with a position open (u_start carried over midnight), 0..1."""
        return float(np.mean(~self.starts_flat)) if self.n_units else 0.0

    def day_blocks(self) -> tuple[np.ndarray, np.ndarray]:
        """(first unit index, number of units) of each flat-to-flat day block, in order: a block starts at
        every day that starts flat and runs until the next one."""
        first = np.flatnonzero(self.starts_flat).astype(np.int64)
        return first, np.diff(np.r_[first, self.n_units]).astype(np.int64)

    def week_blocks(self) -> tuple[np.ndarray, np.ndarray]:
        """(first unit index, number of units) of each week block, in order: a block starts at the first
        day of a market week (week_id) when the account is flat there, so a position held over a weekend
        joins the two weeks into one block."""
        new_week = np.r_[True, self.week[1:] != self.week[:-1]] if self.n_units else np.zeros(0, dtype=bool)
        first = np.flatnonzero(new_week & self.starts_flat).astype(np.int64)
        return first, np.diff(np.r_[first, self.n_units]).astype(np.int64)

    def blocks(self, mode: str) -> tuple[np.ndarray, np.ndarray]:
        """day_blocks() for mode "days", week_blocks() for mode "weeks"."""
        if mode == "days":
            return self.day_blocks()
        if mode == "weeks":
            return self.week_blocks()
        raise ValueError(f"blocks exist for modes 'days' and 'weeks', not {mode!r}")

    def block_summary(self) -> dict[str, Any]:
        """How much the history gives the block bootstrap to resample (JSON-serialisable): n_days,
        share_open_at_start (fraction of days that start with a position open), and per mode ("days",
        "weeks") the number of blocks and their mean and longest length in prop days."""
        out: dict[str, Any] = {"n_days": self.n_units, "share_open_at_start": self.share_open_at_start}
        for mode in ("days", "weeks"):
            _, count = self.blocks(mode)
            out[f"n_blocks_{mode}"] = int(count.size)
            out[f"mean_block_days_{mode}"] = float(count.mean()) if count.size else 0.0
            out[f"max_block_days_{mode}"] = int(count.max()) if count.size else 0
        return out

    def take(self, idx: np.ndarray, block_start: np.ndarray) -> "DayUnits":
        """A new DayUnits made of the days idx (in that order), flat_start = block_start (used by the
        outer bootstrap of history_uncertainty; mode "trades" data is not carried)."""
        idx = np.asarray(idx, dtype=np.int64)
        return DayUnits(
            initial_capital=self.initial_capital, day=self.day[idx], n_bars=self.n_bars[idx],
            u_start=np.where(block_start, 0.0, self.u_start[idx]), d_close=self.d_close[idx],
            d_worst=self.d_worst[idx], d_balance=self.d_balance[idx], flat=self.flat[idx],
            entered=self.entered[idx], traded=self.traded[idx], week=self.week[idx],
            flat_start=np.asarray(block_start, dtype=bool))

    @property
    def d_close_last(self) -> np.ndarray:
        """Equity change over each day (last bar close - E_00:00), USD."""
        return self.d_close[:, -1]

    @property
    def d_balance_last(self) -> np.ndarray:
        """Closed-balance change over each day, USD (the day profit for the best-day rule)."""
        return self.d_balance[:, -1]



def week_id(day) -> np.ndarray:
    """Week block id of prop days (days since 1970-01-01): Sunday starts a block, so a block is
    Sunday..Saturday and holds one market week (Sunday prop days only hold the reopen hour)."""
    return (np.asarray(day, dtype=np.int64) + 4) // 7


def build_day_units(equity: pd.DataFrame, initial_capital: float,
                    trades: pd.DataFrame | None = None) -> DayUnits:
    """Cut a historical EQUITY path into prop-day units (USD); see the module docstring.

    equity: EQUITY frame (validated as in evaluate_path; the account is flat at initial_capital before the
    first bar); initial_capital: C0 in USD; trades: optional TRADES (entry_time gives trading days
    exactly, pnl_usd and exit_time feed mode "trades"). flat_start marks the days whose boundary has no
    position open (the previous day's last bar is flat: units_open 0 and equity_close = balance, see
    propkit.evaluator.equity_arrays); they start the flat-to-flat blocks, so u_start = 0 there.
    """
    c0 = float(initial_capital)
    a = equity_arrays(equity, c0)
    entry = entry_flags(a, trades)
    bal, close, worst, flat = a["balance"], a["close"], a["worst"], a["flat"]
    day, day_id, starts = a["day"], a["day_id"], a["starts"]
    n = bal.size
    n_bars = np.diff(np.r_[starts, n])
    width = int(n_bars.max())
    prev_bal = np.r_[c0, bal[:-1]]
    prev_eq = np.r_[c0, close[:-1]]
    b00, e00 = prev_bal[starts], prev_eq[starts]
    col = np.arange(width)[None, :]
    real = col < n_bars[:, None]
    at = starts[:, None] + np.minimum(col, n_bars[:, None] - 1)
    cum_entry = np.cumsum(entry)
    before = cum_entry[starts] - entry[starts]
    entered_bar = (cum_entry - before[day_id]) > 0
    d_close = close[at] - e00[:, None]
    d_worst = np.where(real, worst[at], close[at]) - e00[:, None]
    unit_days = day[starts]
    trade_pnl = trade_count = None
    if trades is not None and len(trades) and "pnl_usd" in trades.columns and "exit_time" in trades.columns:
        trade_pnl, trade_count = _trade_pool(trades, unit_days, a["time"], day, a["bar_seconds"])
    flat_start = np.r_[True, flat[starts[1:] - 1]]
    return DayUnits(
        initial_capital=c0, day=unit_days, n_bars=n_bars.astype(np.int64), u_start=e00 - b00,
        d_close=d_close, d_worst=d_worst, d_balance=bal[at] - b00[:, None], flat=flat[at],
        entered=entered_bar[at], traded=entered_bar[at[:, -1]], week=week_id(unit_days),
        trade_pnl=trade_pnl, trade_count=trade_count, flat_start=flat_start)


def _trade_pool(trades: pd.DataFrame, unit_days: np.ndarray, times: np.ndarray, bar_day: np.ndarray,
                bar_seconds: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Trade net PnL pool (USD) and the number of trades closed on each unit day.

    A trade's exit day is the prop day of the BAR that books the exit, as in propkit.equity: the last bar
    with open <= exit_time, the exit lying inside it or exactly at its end (a fill at the bar's close, e.g.
    an "end_of_data" exit at the last bar's close). So an exit stamped at the end of a bar that closes at
    00:00 CE(S)T counts on that bar's day, not on the next day (which may have no bar at all).
    """
    pnl = pd.to_numeric(trades["pnl_usd"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(pnl).all():
        raise ValueError("TRADES pnl_usd has missing or non-numeric values")
    et = pd.Series(trades["exit_time"])
    if et.dtype.kind not in "iu":
        raise ValueError("TRADES exit_time must be int64 UTC epoch seconds")
    exit_t = et.to_numpy(dtype=np.int64)
    idx = np.searchsorted(times, exit_t, side="right") - 1
    ok = idx >= 0
    if bar_seconds is not None:
        ok &= exit_t <= times[np.maximum(idx, 0)] + bar_seconds
    if not ok.all():
        i = int(np.flatnonzero(~ok)[0])
        raise ValueError(f"TRADES row {i} exits at {calendar.utc_str(int(exit_t[i]))}, which is not inside a bar "
                         "of EQUITY (before the first bar, after the last bar's end, or in a gap); use the trades "
                         "and the equity of the same run")
    exit_day = bar_day[idx]
    pos = np.searchsorted(unit_days, exit_day)
    return pnl, np.bincount(pos, minlength=unit_days.size).astype(np.int64)


# ---------------------------------------------------------------------------------------
# results

@dataclass(frozen=True)
class BootstrapResult:
    """Monte Carlo outcome (probabilities are fractions; se_* = Monte Carlo errors, simulation noise only).

    p_pass, p_breach_daily, p_breach_max, p_breach_any (= daily + max), p_timeout (no event within the
    horizon): the four outcomes of each simulated challenge add up to 1. p_best_day_unmet_at_target /
    p_min_days_unmet_at_target: share of simulations in which the balance target (flat if required) was
    first met while the best-day rule / the minimum trading days did NOT hold (they may still pass later).
    se_*: sqrt(p (1 - p) / n_sims), the Monte Carlo error of p; it does NOT include the uncertainty from
    the limited history (see history_uncertainty). days_to_target_q: quantiles (keys p10..p90) of
    simulated prop days with bars up to and including the passing day, over passed simulations (None if
    none passed); trading_days_to_target_q: the same in days with a trade entry; days_to_breach_q: prop
    days with bars up to the breach. horizon_days / horizon_unit: the challenge length and what it counts
    ("trading" or "market" days). n_blocks: resampling blocks of the mode (flat-to-flat day or week
    blocks; days for "trades", 1 for "replay"); share_open_at_start: share of historical days that start
    with a position open. status / days / trading_days: per-simulation arrays (status codes: 0 timeout,
    1 passed, 2 breached daily, 3 breached max).
    """

    mode: str
    n_sims: int
    seed: int
    horizon_days: int | None
    horizon_unit: str
    size_multiplier: float
    rules_name: str
    n_units: int
    n_blocks: int
    share_open_at_start: float
    p_pass: float
    se_pass: float
    p_breach_daily: float
    se_breach_daily: float
    p_breach_max: float
    se_breach_max: float
    p_breach_any: float
    se_breach_any: float
    p_timeout: float
    se_timeout: float
    p_best_day_unmet_at_target: float
    se_best_day_unmet_at_target: float
    p_min_days_unmet_at_target: float
    se_min_days_unmet_at_target: float
    days_to_target_q: dict[str, float] | None
    trading_days_to_target_q: dict[str, float] | None
    days_to_breach_q: dict[str, float] | None
    status: np.ndarray = field(repr=False)
    days: np.ndarray = field(repr=False)
    trading_days: np.ndarray = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict (per-simulation arrays left out)."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__
                if k not in ("status", "days", "trading_days")}

    def summary_lines(self) -> list[str]:
        """Plain-English ASCII summary."""
        hz = horizon_text(self.horizon_days, self.horizon_unit)
        lines = [f"Bootstrap ({self.mode}, {self.n_sims} challenges, seed {self.seed}, horizon {hz}, "
                 f"size x{self.size_multiplier:g}, {self.n_blocks} blocks) under {self.rules_name}:"]
        for label, p, se in (("pass", self.p_pass, self.se_pass),
                             ("daily-loss breach", self.p_breach_daily, self.se_breach_daily),
                             ("max-loss breach", self.p_breach_max, self.se_breach_max),
                             ("timeout", self.p_timeout, self.se_timeout),
                             ("best-day rule unmet at target", self.p_best_day_unmet_at_target,
                              self.se_best_day_unmet_at_target)):
            lines.append(f"  P({label}) = {prob_text(p, se, self.n_sims)} (MC error, simulation noise only)")
        if self.days_to_target_q:
            q = self.days_to_target_q
            lines.append("  days to target (market days) p10/p25/p50/p75/p90: "
                         + "/".join(f"{q[k]:.1f}" for k in sorted(q, key=lambda s: int(s[1:]))))
        return lines


def horizon_text(horizon_days: int | None, horizon_unit: str = DEFAULT_HORIZON_UNIT) -> str:
    """'60 trading days', '60 market days' or 'no limit' (ASCII)."""
    if horizon_days is None:
        return f"no limit (cap {MAX_DAYS_UNLIMITED} market days)"
    return f"{int(horizon_days)} {horizon_unit} days"


def prob_text(p, se, n_sims: int | None = None) -> str:
    """'0.3100 +- 0.0046' (the Monte Carlo error); at p = 0 or 1, where that error is 0, the 95% rule of
    three instead: '0.0000 (< 0.0003)' with 3 / n_sims. 'n/a' for None."""
    if p is None:
        return "n/a"
    p = float(p)
    if n_sims and se is not None and float(se) == 0.0 and p in (0.0, 1.0):
        bound = 3.0 / float(n_sims)
        return f"{p:.4f} (< {bound:.4f})" if p == 0.0 else f"{p:.4f} (> {1.0 - bound:.4f})"
    return f"{p:.4f} +- {float(se or 0.0):.4f}"


def _quantiles(x: np.ndarray) -> dict[str, float] | None:
    if x.size == 0:
        return None
    vals = np.quantile(x.astype(np.float64), [q / 100.0 for q in QUANTILES])
    return {f"p{q}": float(v) for q, v in zip(QUANTILES, vals)}


def _prob(mask: np.ndarray) -> tuple[float, float]:
    n = mask.size
    p = float(mask.mean())
    return p, math.sqrt(p * (1.0 - p) / n)


# ---------------------------------------------------------------------------------------
# simulation core

def _check_unit(horizon_unit) -> str:
    if horizon_unit not in HORIZON_UNITS:
        raise ValueError(f"horizon_unit must be one of {', '.join(HORIZON_UNITS)} (what horizon_days counts: "
                         f"days with a trade entry, or every prop day with bars); got {horizon_unit!r}")
    return str(horizon_unit)


def _check_args(mode: str, n_sims, seed, horizon_days, size_multiplier) -> tuple[int, int, int | None, float]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}; got {mode!r}")
    if isinstance(n_sims, (bool, np.bool_)) or not isinstance(n_sims, (int, np.integer)) or int(n_sims) < 1:
        raise ValueError(f"n_sims must be a whole number >= 1, got {n_sims!r}")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError(f"seed must be a whole number >= 0, got {seed!r}")
    if horizon_days is None or (isinstance(horizon_days, str) and horizon_days.strip().lower() == "none"):
        horizon = None
    elif (isinstance(horizon_days, (bool, np.bool_)) or not isinstance(horizon_days, (int, np.integer))
          or int(horizon_days) < 1):
        raise ValueError(f"horizon_days must be a whole number of days >= 1, or None / 'none'; got {horizon_days!r}")
    else:
        horizon = int(horizon_days)
    if (isinstance(size_multiplier, (bool, np.bool_))
            or not isinstance(size_multiplier, (int, float, np.integer, np.floating))
            or not math.isfinite(float(size_multiplier)) or float(size_multiplier) < 0):
        raise ValueError(f"size_multiplier must be a finite number >= 0, got {size_multiplier!r}")
    return int(n_sims), int(seed), horizon, float(size_multiplier)


class _Drawer:
    """Unit draws per simulated day, identical for every size multiplier (common random numbers).

    Modes "days" and "weeks" play blocks (DayUnits.blocks): a simulation draws a block uniformly with
    replacement, plays its days in order, then draws the next block. With one-day blocks every simulated
    day is a fresh uniform draw over all days, the plain iid-day bootstrap."""

    def __init__(self, units: DayUnits, mode: str, n_sims: int, seed: int, tile: int = 1):
        # tile > 1: the same n_sims draws repeated tile times (one copy per size multiplier, so a stacked
        # run gives exactly the draws of tile separate runs - common random numbers)
        self.units, self.mode, self.n_sims, self.seed, self.tile = units, mode, n_sims, seed, int(tile)
        self.n = units.n_units
        if mode in ("days", "weeks"):
            self.w_first, self.w_len = units.blocks(mode)
            self.week = np.full(n_sims * self.tile, -1, dtype=np.int64)
            self.pos = np.zeros(n_sims * self.tile, dtype=np.int64)
        if mode == "trades":
            if units.trade_pnl is None or units.trade_pnl.size == 0:
                raise ValueError("mode 'trades' needs TRADES with pnl_usd and exit_time (pass trades=...)")
            self.max_trades = max(int(units.trade_count.max()), 1)

    def rng(self, d: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, _STREAM_DRAWS, d])

    def day_units(self, d: int) -> np.ndarray:
        """Unit index of simulated day d for every simulation (all n_sims x tile, not only the active ones)."""
        if self.mode == "replay":
            return np.full(self.n_sims, d, dtype=np.int64)
        need = (self.week < 0) | (self.pos >= self.w_len[np.maximum(self.week, 0)])
        if need.any():               # day d's draws come from their own stream: skipping a day changes nothing
            fresh = self.rng(d).integers(0, self.w_first.size, self.n_sims)
            if self.tile > 1:
                fresh = np.tile(fresh, self.tile)
            self.week = np.where(need, fresh, self.week)
            self.pos = np.where(need, 0, self.pos)
        out = self.w_first[self.week] + self.pos
        self.pos = self.pos + 1
        return out

    def trade_draws(self, d: int) -> tuple[np.ndarray, np.ndarray]:
        """(historical day slot, trade indices) of simulated day d for every simulation."""
        rng = self.rng(d)
        slots = rng.integers(0, self.n, self.n_sims)
        picks = rng.integers(0, self.units.trade_pnl.size, (self.n_sims, self.max_trades))
        if self.tile > 1:
            slots, picks = np.tile(slots, self.tile), np.tile(picks, (self.tile, 1))
        return slots, picks


def _simulate(units: DayUnits, rules: PropRules, mode: str, n_sims: int, seed: int,
              horizon: int | None, m, horizon_unit: str = DEFAULT_HORIZON_UNIT) -> dict[str, np.ndarray]:
    """Run the simulations; returns per-simulation arrays (see BootstrapResult).

    m: one size multiplier, or a 1-D array of K multipliers: then K x n_sims simulations run together,
    block k (simulations k x n_sims .. (k + 1) x n_sims - 1) with multiplier m[k] and the same draws as a
    separate run (_split_runs cuts the result into K runs)."""
    if abs(rules.initial_capital - units.initial_capital) > 1e-9 * units.initial_capital:
        raise ValueError(f"the rules' initial capital {rules.initial_capital:,.2f} differs from the one the day "
                         f"units were built with ({units.initial_capital:,.2f})")
    c0 = units.initial_capital
    trading_horizon = horizon is not None and horizon_unit == "trading"
    market_horizon = horizon if (horizon is not None and not trading_horizon) else None
    mults = np.atleast_1d(np.asarray(m, dtype=np.float64))
    if mode == "replay":
        n_sims = 1
        limit = units.n_units if market_horizon is None else min(market_horizon, units.n_units)
        mults = mults[:1]
    else:
        limit = MAX_DAYS_UNLIMITED if market_horizon is None else market_horizon
    drawer = _Drawer(units, mode, n_sims, seed, tile=mults.size)
    n_base = n_sims
    n_sims = n_sims * mults.size
    m_all = np.repeat(mults, n_base)

    equity = np.full(n_sims, c0)
    high_b = np.full(n_sims, c0)
    best = np.zeros(n_sims)
    total = np.zeros(n_sims)
    tdays = np.zeros(n_sims, dtype=np.int64)
    ndays = np.zeros(n_sims, dtype=np.int64)
    status = np.full(n_sims, RUNNING, dtype=np.int8)
    end_days = np.zeros(n_sims, dtype=np.int64)
    end_tdays = np.zeros(n_sims, dtype=np.int64)
    seen = np.zeros(n_sims, dtype=bool)
    bd_unmet = np.zeros(n_sims, dtype=bool)
    md_unmet = np.zeros(n_sims, dtype=bool)
    active = np.arange(n_sims)

    ref_balance = rules.day_start_reference == "balance"
    trailing = rules.max_loss_mode == "trailing_eod_balance" and rules.max_loss_pct is not None
    static_max = rules.max_floor() if not trailing else None
    target = rules.target_balance
    share = rules.best_day_max_share
    basis_balance = rules.best_day_basis == "balance"
    min_days = rules.min_trading_days
    no_floor = rules.daily_loss_pct is None and rules.max_loss_pct is None

    # per-unit day extremes: a row can breach only if its lowest equity_worst is below the floor and can
    # pass only if its highest balance reaches the target (m >= 0 keeps both exact), so the bar-by-bar
    # arrays are gathered for those rows only
    unit_min_worst = units.d_worst.min(axis=1)
    unit_max_bal = units.d_balance.max(axis=1)
    unit_last_close = units.d_close_last
    unit_last_bal = units.d_balance_last
    inclusive = rules.breach_inclusive

    for d in range(limit):
        if active.size == 0:
            break
        n_act = active.size
        m_a = m_all[active]
        if mode == "trades":
            slots_all, picks_all = drawer.trade_draws(d)
            count = units.trade_count[slots_all[active]]
            pnl = units.trade_pnl[picks_all[active]] * (np.arange(drawer.max_trades)[None, :] < count[:, None]) \
                * m_a[:, None]
            t_close = np.cumsum(pnl, axis=1)
            t_worst = np.minimum(np.c_[np.zeros(n_act), t_close[:, :-1]], t_close)
            t_entered = (count > 0)[:, None]
            u = np.zeros(n_act)
            traded = count > 0
            last_close = last_bal = t_close[:, -1]
            row_min_worst, row_max_bal = t_worst.min(axis=1), t_close.max(axis=1)

            def rows_close(r, t_close=t_close):
                return t_close[r]

            def rows_worst(r, t_worst=t_worst):
                return t_worst[r]

            rows_bal = rows_close

            def rows_flat(r):
                return True

            def rows_entered(r, t_entered=t_entered):
                return t_entered[r]
        else:
            idx = drawer.day_units(d)[active]
            u = units.u_start[idx] * m_a
            traded = units.traded[idx]
            last_close, last_bal = unit_last_close[idx] * m_a, unit_last_bal[idx] * m_a
            row_min_worst, row_max_bal = unit_min_worst[idx] * m_a, unit_max_bal[idx] * m_a

            def rows_close(r, idx=idx, m_a=m_a):
                return units.d_close[idx[r]] * m_a[r, None]

            def rows_worst(r, idx=idx, m_a=m_a):
                return units.d_worst[idx[r]] * m_a[r, None]

            def rows_bal(r, idx=idx, m_a=m_a):
                return units.d_balance[idx[r]] * m_a[r, None]

            def rows_flat(r, idx=idx):
                return units.flat[idx[r]]

            def rows_entered(r, idx=idx):
                return units.entered[idx[r]]

        e0 = equity[active]
        b0 = e0 - u
        ref = b0 if ref_balance else np.maximum(b0, e0)
        dfl = np.asarray(rules.daily_floor(ref), dtype=np.float64)
        if trailing:
            hb = np.maximum(high_b[active], b0)
            high_b[active] = hb
            mfl = np.asarray(rules.max_floor(hb), dtype=np.float64)
        else:
            mfl = np.full(n_act, static_max)
        hi = np.maximum(dfl, mfl)

        # first breaching bar (equity_worst vs the higher floor)
        br_k = np.full(n_act, _NEVER)
        if not no_floor:
            lowest = e0 + row_min_worst
            r = np.flatnonzero((lowest <= hi) if inclusive else (lowest < hi))
            if r.size:
                low = e0[r, None] + rows_worst(r)
                br = (low <= hi[r, None]) if inclusive else (low < hi[r, None])
                br_k[r] = np.where(br.any(axis=1), br.argmax(axis=1), _NEVER)

        # first passing bar close (target, flat, best day, minimum trading days)
        ok_k = np.full(n_act, _NEVER)
        pass_td = np.zeros(n_act, dtype=np.int64)
        if target is not None:
            r = np.flatnonzero(b0 + row_max_bal >= target)
            if r.size:
                sims_r = active[r]
                bal_r = rows_bal(r)
                tok = (b0[r, None] + bal_r) >= target
                if rules.target_requires_flat:
                    tok = tok & rows_flat(r)
                ok = tok
                bdok = mdok = None
                if share is not None:
                    cur = bal_r if basis_balance else rows_close(r)
                    bnow = np.maximum(best[sims_r][:, None], cur)
                    tnow = total[sims_r][:, None] + np.maximum(cur, 0.0)
                    bdok = bnow <= share * tnow + BEST_DAY_TOL_USD
                    ok = ok & bdok
                ent = np.broadcast_to(rows_entered(r), tok.shape)
                if min_days > 0:
                    mdok = (tdays[sims_r][:, None] + ent) >= min_days
                    ok = ok & mdok
                ok_any = ok.any(axis=1)
                k_ok = np.where(ok_any, ok.argmax(axis=1), _NEVER)
                ok_k[r] = k_ok
                hit = np.flatnonzero(ok_any)
                pass_td[r[hit]] = tdays[sims_r[hit]] + ent[hit, k_ok[hit]]
                tok_any = tok.any(axis=1)
                k_tok = np.where(tok_any, tok.argmax(axis=1), _NEVER)
                first = np.flatnonzero(~seen[sims_r] & tok_any & (k_tok < br_k[r]))
                if first.size:
                    seen[sims_r[first]] = True
                    if bdok is not None:
                        bd_unmet[sims_r[first]] = ~bdok[first, k_tok[first]]
                    if mdok is not None:
                        md_unmet[sims_r[first]] = ~mdok[first, k_tok[first]]

        breach_now = (br_k != _NEVER) & (br_k <= ok_k)
        pass_now = ~breach_now & (ok_k != _NEVER)
        if breach_now.any():
            sims = active[breach_now]
            status[sims] = np.where(dfl[breach_now] > mfl[breach_now], BREACHED_DAILY, BREACHED_MAX)
            end_days[sims] = ndays[sims] + 1
        if pass_now.any():
            sims = active[pass_now]
            status[sims] = PASSED
            end_days[sims] = ndays[sims] + 1
            end_tdays[sims] = pass_td[pass_now]
        cont = ~(breach_now | pass_now)
        sims = active[cont]
        day_profit = last_bal[cont] if basis_balance else last_close[cont]
        equity[sims] = e0[cont] + last_close[cont]
        best[sims] = np.maximum(best[sims], day_profit)
        total[sims] += np.maximum(day_profit, 0.0)
        tdays[sims] += traded[cont]
        ndays[sims] += 1
        active = sims[tdays[sims] < horizon] if trading_horizon else sims

    end_days[status == RUNNING] = ndays[status == RUNNING]
    return {"status": status, "days": end_days, "trading_days": end_tdays, "seen": seen,
            "bd_unmet": bd_unmet, "md_unmet": md_unmet, "n_sims": n_sims}


def _split_runs(sim: dict[str, np.ndarray], k: int) -> list[dict[str, np.ndarray]]:
    """Cut a stacked _simulate result (k multipliers) into k single runs."""
    n = int(sim["n_sims"]) // k
    keys = [key for key in sim if key != "n_sims"]
    return [{**{key: sim[key][j * n:(j + 1) * n] for key in keys}, "n_sims": n} for j in range(k)]


def _run_many(units: DayUnits, rules: PropRules, mode: str, n_sims: int, seed: int, horizon: int | None,
              mults, unit: str, chunk: int = 8) -> list[BootstrapResult]:
    """BootstrapResult for each multiplier in mults, simulated `chunk` multipliers at a time (stacked runs;
    identical to separate runs)."""
    mults = [float(x) for x in mults]
    out: list[BootstrapResult] = []
    for i in range(0, len(mults), chunk):
        part = mults[i:i + chunk]
        runs = _split_runs(_simulate(units, rules, mode, n_sims, seed, horizon, np.array(part), unit), len(part))
        out += [_result(sim, units, rules, mode, seed, horizon, mm, unit) for sim, mm in zip(runs, part)]
    return out


def _n_blocks(units: DayUnits, mode: str) -> int:
    if mode in ("days", "weeks"):
        return int(units.blocks(mode)[0].size)
    return 1 if mode == "replay" else units.n_units


def _result(sim: dict[str, np.ndarray], units: DayUnits, rules: PropRules, mode: str, seed: int,
            horizon: int | None, m: float, horizon_unit: str = DEFAULT_HORIZON_UNIT) -> BootstrapResult:
    status = sim["status"]
    passed = status == PASSED
    breached = (status == BREACHED_DAILY) | (status == BREACHED_MAX)
    probs = {}
    for name, mask in (("pass", passed), ("breach_daily", status == BREACHED_DAILY),
                       ("breach_max", status == BREACHED_MAX), ("breach_any", breached),
                       ("timeout", status == RUNNING),
                       ("best_day_unmet_at_target", sim["seen"] & sim["bd_unmet"]),
                       ("min_days_unmet_at_target", sim["seen"] & sim["md_unmet"])):
        probs[f"p_{name}"], probs[f"se_{name}"] = _prob(mask)
    return BootstrapResult(
        mode=mode, n_sims=int(sim["n_sims"]), seed=seed,
        horizon_days=(units.n_units if horizon is None else min(horizon, units.n_units))
        if mode == "replay" else horizon, horizon_unit=horizon_unit,
        size_multiplier=m, rules_name=rules.name, n_units=units.n_units, n_blocks=_n_blocks(units, mode),
        share_open_at_start=units.share_open_at_start, **probs,
        days_to_target_q=_quantiles(sim["days"][passed]),
        trading_days_to_target_q=_quantiles(sim["trading_days"][passed]),
        days_to_breach_q=_quantiles(sim["days"][breached]),
        status=status, days=sim["days"], trading_days=sim["trading_days"])


def bootstrap_challenges(equity: pd.DataFrame | None, rules: PropRules, trades: pd.DataFrame | None = None,
                         mode: str = "days", n_sims: int = DEFAULT_N_SIMS, seed: int = DEFAULT_SEED,
                         horizon_days: int | str | None = DEFAULT_HORIZON_DAYS, size_multiplier: float = 1.0,
                         units: DayUnits | None = None,
                         horizon_unit: str = DEFAULT_HORIZON_UNIT) -> BootstrapResult:
    """Monte Carlo of `n_sims` challenges under `rules`, resampling the history of one EQUITY path.

    equity: EQUITY frame (USD) of the historical run; trades: its TRADES (optional; exact trading days,
    and required for mode "trades"); mode: "days" (default, flat-to-flat day blocks), "weeks", "trades"
    (comparison only) or "replay"; seed: Generator seed; horizon_days: days per challenge (default 60) or
    None/'none' (until pass or breach, capped at 2,520 market days); horizon_unit: "trading" (default,
    the contract: days with a trade entry) or "market" (every prop day with bars); size_multiplier: scales
    every USD increment (sizes and costs are linear in units); units: prebuilt DayUnits (then equity and
    trades are not read again). Returns BootstrapResult. See the module docstring for the blocks, the
    chaining and the conventions.
    """
    n_sims, seed, horizon, m = _check_args(mode, n_sims, seed, horizon_days, size_multiplier)
    unit = _check_unit(horizon_unit)
    if not isinstance(rules, PropRules):
        raise ValueError("rules must be a PropRules (see propkit.rules)")
    if units is None:
        if equity is None:
            raise ValueError("pass the EQUITY frame (or prebuilt units=build_day_units(...))")
        units = build_day_units(equity, rules.initial_capital, trades)
    sim = _simulate(units, rules, mode, n_sims, seed, horizon, m, unit)
    return _result(sim, units, rules, mode, seed, horizon, m, unit)


# ---------------------------------------------------------------------------------------
# largest size within a breach budget

@dataclass(frozen=True)
class MaxSizeResult:
    """Largest size multiplier whose breach probability stays within alpha (common random numbers).

    multiplier: largest multiplier with P(daily breach before target) <= alpha (the daily-loss version);
    at: the BootstrapResult.to_dict() of that multiplier; multiplier_any / at_any: the same counting any
    breach (daily or max). note / note_any: 'ok', 'at grid top' (extend the grid) or 'below grid'.
    curve: DataFrame of every multiplier evaluated (grid and bisection points), sorted, with p_pass,
    p_breach_daily, p_breach_max, p_breach_any, p_timeout and the standard errors of the breach ones.
    """

    alpha: float
    mode: str
    n_sims: int
    seed: int
    horizon_days: int | None
    horizon_unit: str
    multiplier: float
    at: dict[str, Any]
    note: str
    multiplier_any: float
    at_any: dict[str, Any]
    note_any: str
    curve: pd.DataFrame = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict (curve as a list of rows)."""
        out = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "curve"}
        out["curve"] = [{k: float(v) for k, v in row.items()} for row in self.curve.to_dict(orient="records")]
        return out


def max_size(equity: pd.DataFrame | None, rules: PropRules, trades: pd.DataFrame | None = None,
             mode: str = "days", alpha: float = 0.05, n_sims: int = DEFAULT_N_SIMS, seed: int = DEFAULT_SEED,
             horizon_days: int | str | None = DEFAULT_HORIZON_DAYS, grid: Sequence[float] | None = None,
             n_bisect: int = 8, units: DayUnits | None = None,
             horizon_unit: str = DEFAULT_HORIZON_UNIT) -> MaxSizeResult:
    """Largest size multiplier m with P(daily breach before target) <= alpha, and the any-breach version.

    Every multiplier uses the same draws (common random numbers: same seed, same unit sequence). The
    probability is evaluated on `grid` (default 0.05 ... 10, multiples of the historical size), made
    monotone by a running maximum (a size is allowed only if every smaller grid size is too), and the
    crossing is refined by `n_bisect` bisection steps between the last allowed grid point (or 0, where
    nothing moves and nothing can breach) and the next one. alpha: a fraction (0.05 = 5%). Other
    arguments as in bootstrap_challenges. Returns MaxSizeResult.
    The multiplier is a point estimate from ONE history: picking the largest m whose estimated P is within
    alpha also favours sizes whose P came out low by chance (Monte Carlo and history noise), so the true
    P at that size can be above alpha; history_uncertainty() shows how much it moves between histories.
    """
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, (int, float, np.integer, np.floating))
            or not 0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be a fraction between 0 and 1 (0.05 = 5%), got {alpha!r}")
    if isinstance(n_bisect, (bool, np.bool_)) or not isinstance(n_bisect, (int, np.integer)) or n_bisect < 0:
        raise ValueError(f"n_bisect must be a whole number >= 0, got {n_bisect!r}")
    if mode == "replay":
        raise ValueError("max_size needs a random mode (days, weeks or trades), not replay")
    n_sims, seed, horizon, _ = _check_args(mode, n_sims, seed, horizon_days, 1.0)
    unit = _check_unit(horizon_unit)
    g = np.unique(np.asarray(DEFAULT_GRID if grid is None else grid, dtype=np.float64))
    if g.size == 0 or not np.isfinite(g).all() or (g <= 0).any():
        raise ValueError("grid must be a list of positive size multipliers")
    if units is None:
        if equity is None:
            raise ValueError("pass the EQUITY frame (or prebuilt units=build_day_units(...))")
        units = build_day_units(equity, rules.initial_capital, trades)
    cache: dict[float, BootstrapResult] = {}

    def run(mult: float) -> BootstrapResult:
        key = float(mult)
        if key not in cache:
            sim = _simulate(units, rules, mode, n_sims, seed, horizon, key, unit)
            cache[key] = _result(sim, units, rules, mode, seed, horizon, key, unit)
        return cache[key]

    for mult, res in zip(g, _run_many(units, rules, mode, n_sims, seed, horizon, g, unit)):
        cache[float(mult)] = res

    def search(metric: str) -> tuple[float, str]:
        p = np.array([getattr(cache[float(x)], metric) for x in g])
        allowed = np.maximum.accumulate(p) <= float(alpha)
        if allowed.all():
            return float(g[-1]), "at grid top"
        k = int(np.argmin(allowed))              # first grid point that is not allowed
        lo = float(g[k - 1]) if k > 0 else 0.0
        hi = float(g[k])
        for _ in range(int(n_bisect)):
            mid = 0.5 * (lo + hi)
            if getattr(run(mid), metric) <= float(alpha):
                lo = mid
            else:
                hi = mid
        return lo, ("ok" if lo > 0 else "below grid")

    m_daily, note = search("p_breach_daily")
    m_any, note_any = search("p_breach_any")
    rows = []
    for mult in sorted(cache):
        r = cache[mult]
        rows.append({"multiplier": mult, "p_pass": r.p_pass, "p_breach_daily": r.p_breach_daily,
                     "se_breach_daily": r.se_breach_daily, "p_breach_max": r.p_breach_max,
                     "p_breach_any": r.p_breach_any, "se_breach_any": r.se_breach_any, "p_timeout": r.p_timeout})
    at = run(m_daily).to_dict() if m_daily > 0 else None
    at_any = run(m_any).to_dict() if m_any > 0 else None
    return MaxSizeResult(alpha=float(alpha), mode=mode, n_sims=n_sims, seed=seed, horizon_days=horizon,
                         horizon_unit=unit, multiplier=m_daily, at=at, note=note, multiplier_any=m_any,
                         at_any=at_any, note_any=note_any, curve=pd.DataFrame(rows))


# ---------------------------------------------------------------------------------------
# uncertainty from the limited history (outer bootstrap)

def days_breaching_alone(units: DayUnits, rules: PropRules, multiplier: float) -> int | None:
    """How many historical prop days would breach the daily rule ON THEIR OWN at size `multiplier`:
    m x (u_start + the day's lowest d_worst) < -daily_loss_pct x C0 (<= with breach_inclusive), i.e. the
    day's own drawdown from its 00:00 balance exceeds the daily limit. Only for daily_loss_base "initial"
    with day_start_reference "balance" (FTMO); None otherwise or without a daily rule."""
    if rules.daily_loss_pct is None or rules.daily_loss_base != "initial" or rules.day_start_reference != "balance":
        return None
    limit = round(rules.daily_loss_pct * rules.initial_capital, 8)
    low = float(multiplier) * (units.u_start + units.d_worst.min(axis=1))
    return int(((low <= -limit) if rules.breach_inclusive else (low < -limit)).sum())


def _resampled_history(units: DayUnits, rng: np.random.Generator) -> DayUnits:
    """One outer-bootstrap history: as many flat-to-flat day blocks as the history has, drawn with
    replacement and laid end to end (block starts stay flat)."""
    first, count = units.day_blocks()
    pick = rng.integers(0, first.size, first.size)
    lens = count[pick]
    total = int(lens.sum())
    offset = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(lens) - lens, lens)
    idx = np.repeat(first[pick], lens) + offset
    return units.take(idx, offset == 0)


def _quick_max_size(units: DayUnits, rules: PropRules, alpha: float, n_sims: int, seed: int,
                    horizon: int | None, unit: str, steps: int = 2, start: float | None = None,
                    extra: Sequence[float] = ()) -> tuple[float, list[BootstrapResult]]:
    """Largest m with P(daily breach) <= alpha (common random numbers), at most 64, 0 if even 1/64 of the
    size breaches too often, inf without a daily rule. One stacked run evaluates start x 2^-3 .. 2^3
    (start: default 1; max_size's answer is a good start; the range is extended if the crossing lies
    outside it), P is made monotone by a running maximum as in max_size, then `steps` stacked runs of 3
    points each narrow the bracket 4-fold (an octave to 1/16 with 2 steps). The multipliers in `extra`
    ride along in the first run; returns (m, their BootstrapResults)."""
    extra = [float(x) for x in extra]
    m0 = 1.0 if start is None or not math.isfinite(float(start)) or float(start) <= 0 else float(start)
    m0 = min(max(m0, 1.0 / 64.0), 64.0)
    grid = np.unique(np.clip(m0 * 2.0 ** np.arange(-3.0, 4.0), 1.0 / 64.0, 64.0))
    if rules.daily_loss_pct is None:
        return math.inf, (_run_many(units, rules, "days", n_sims, seed, horizon, extra, unit) if extra else [])
    res = _run_many(units, rules, "days", n_sims, seed, horizon, list(grid) + extra, unit, chunk=16)
    extra_res = res[grid.size:]
    pts = {float(x): r.p_breach_daily for x, r in zip(grid, res[:grid.size])}
    while True:                                      # extend until the crossing is inside (or a limit is hit)
        xs = np.array(sorted(pts))
        allowed = np.maximum.accumulate(np.array([pts[x] for x in xs])) <= alpha
        if allowed.all() and xs[-1] < 64.0:
            more = np.clip(xs[-1] * 2.0 ** np.arange(1.0, 4.0), None, 64.0)
        elif not allowed.any() and xs[0] > 1.0 / 64.0:
            more = np.clip(xs[0] * 2.0 ** -np.arange(1.0, 4.0), 1.0 / 64.0, None)
        else:
            break
        more = np.unique(more)
        pts.update({float(x): r.p_breach_daily
                    for x, r in zip(more, _run_many(units, rules, "days", n_sims, seed, horizon, more, unit))})
    if allowed.all():
        return float(xs[-1]), extra_res
    if not allowed.any():
        return 0.0, extra_res
    k = int(np.argmin(allowed))                      # first point that is not allowed
    lo, hi = float(xs[k - 1]), float(xs[k])
    for _ in range(int(steps)):
        mids = lo + (hi - lo) * np.array([0.25, 0.5, 0.75])
        ps = [r.p_breach_daily for r in _run_many(units, rules, "days", n_sims, seed, horizon, mids, unit)]
        ok = np.array([all(p <= alpha for p in ps[:i + 1]) for i in range(3)])
        n_ok = int(ok.sum())
        lo, hi = (float(mids[n_ok - 1]) if n_ok else lo), (float(mids[n_ok]) if n_ok < 3 else hi)
    return lo, extra_res


def history_uncertainty(equity: pd.DataFrame | None, rules: PropRules, trades: pd.DataFrame | None = None,
                        n_reps: int = DEFAULT_HISTORY_REPS, n_sims: int = HISTORY_SIMS, seed: int = DEFAULT_SEED,
                        horizon_days: int | str | None = DEFAULT_HORIZON_DAYS, alpha: float | None = 0.05,
                        units: DayUnits | None = None,
                        horizon_unit: str = DEFAULT_HORIZON_UNIT, size_start: float | None = None) -> dict[str, Any]:
    """How much the day-block bootstrap's answers depend on the particular history (an outer bootstrap).

    Each of n_reps replicates draws a new history of the same number of flat-to-flat day blocks WITH
    replacement from the historical blocks, then runs the day-block bootstrap on it (n_sims simulations,
    the same seed for every replicate) and, if alpha is given, the largest size with P(daily breach) <=
    alpha (a stacked search around size_start - pass max_size's answer; default 1 - see
    _quick_max_size: resolution 1/16 of an octave, about 4%). The spread of the replicates is the uncertainty
    that comes from having only this history; the Monte Carlo error se_* of a single run does not contain
    it. Each replicate also carries Monte Carlo noise of n_sims simulations (about sqrt(p (1 - p) /
    n_sims)), so the ranges are slightly wider than the history part alone. With few blocks (see
    DayUnits.block_summary) the replicates re-use few starting points and the ranges are rough.
    Returns a JSON-serialisable dict: n_reps, n_sims, seed, n_blocks, and for p_pass, p_breach_daily,
    p_breach_max, p_breach_any and max_size_multiplier: p5, p50, p95 (fractions; multipliers of the
    run's size), plus 'values' (the replicate values of each).
    """
    if isinstance(n_reps, (bool, np.bool_)) or not isinstance(n_reps, (int, np.integer)) or n_reps < 2:
        raise ValueError(f"n_reps must be a whole number >= 2, got {n_reps!r}")
    n_sims, seed, horizon, _ = _check_args("days", n_sims, seed, horizon_days, 1.0)
    unit = _check_unit(horizon_unit)
    if alpha is not None and (isinstance(alpha, (bool, np.bool_)) or not 0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be a fraction between 0 and 1 (0.05 = 5%) or None, got {alpha!r}")
    if not isinstance(rules, PropRules):
        raise ValueError("rules must be a PropRules (see propkit.rules)")
    if units is None:
        if equity is None:
            raise ValueError("pass the EQUITY frame (or prebuilt units=build_day_units(...))")
        units = build_day_units(equity, rules.initial_capital, trades)
    keys = ("p_pass", "p_breach_daily", "p_breach_max", "p_breach_any")
    vals: dict[str, list[float]] = {k: [] for k in keys}
    vals["max_size_multiplier"] = []
    for b in range(int(n_reps)):
        hist = _resampled_history(units, np.random.default_rng([seed, _STREAM_OUTER, b]))
        if alpha is not None:
            m_b, (res,) = _quick_max_size(hist, rules, float(alpha), n_sims, seed, horizon, unit, start=size_start,
                                          extra=[1.0])
            vals["max_size_multiplier"].append(m_b)
        else:
            res = _run_many(hist, rules, "days", n_sims, seed, horizon, [1.0], unit)[0]
        for k in keys:
            vals[k].append(float(getattr(res, k)))
    out: dict[str, Any] = {"n_reps": int(n_reps), "n_sims": n_sims, "seed": seed, "alpha": alpha,
                           "n_blocks": int(units.day_blocks()[0].size), "horizon_days": horizon,
                           "horizon_unit": unit, "values": vals}
    for k, v in vals.items():
        arr = np.asarray(v, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        out[k] = None if finite.size == 0 else {f"p{q}": float(np.quantile(finite, q / 100.0)) for q in (5, 50, 95)}
    return out
