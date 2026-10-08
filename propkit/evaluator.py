"""propkit/evaluator.py - one equity path through one set of prop rules. Research only.

evaluate_path(equity, trades, rules) scans the EQUITY frame bar by bar (hour by hour for H1) and
answers: was the account breached (first breach: time, kind, equity, floor), did it pass (when, after
how many trading and calendar days), and how close did it come (per-day table, worst daily drawdown,
max drawdown). Money is USD; C0 = rules.initial_capital; "% of C0" figures are percent (3.0 = 3%).

Inputs
  EQUITY (see the propkit contract): time (bar open, UTC epoch s, sorted, unique), balance (closed balance
  at the END of the bar), equity_close (balance + floating PnL at the bar close), equity_worst (lowest
  equity inside the bar), units_open (signed oz at the bar close); optional commission_usd and
  realised_usd (only used to count trading days when no TRADES are given) and swap_usd. The account is
  taken to be flat at C0 before the first bar: with the ledger columns realised_usd, commission_usd and
  swap_usd (as propkit.equity writes them) the balance before the first bar, balance - realised_usd +
  commission_usd - swap_usd of the first row, must equal C0 (to 0.01 USD); without them the first bar's
  equity_close must lie within 20% of C0 (a heuristic: a first bar that moves the account by more than 20%
  is then refused too). A bar is FLAT (no position open) when |units_open| <= 1e-9 oz (float residue) AND
  equity_close equals balance (to 1e-6 USD): units_open is the NET size, so a long and a short of the same
  size held together give units_open 0 while two positions are open, and their floating PnL (equity_close
  - balance) shows them. (A hedge whose floating PnL is exactly 0, e.g. both legs filled at one price with
  no spread, is seen as flat.)
  TRADES (optional): entry_time (fill instant) gives the trading days exactly; pnl_usd and risk_usd give
  the R-multiple summary. Without TRADES a bar counts as an entry when |units_open| grows or changes
  sign, or when commission or realised PnL is booked with units_open unchanged (a round trip inside the
  bar). Pass TRADES for an exact count.

Conventions (each is a choice; they are the same in propkit.bootstrap)
  * Prop days: a bar belongs to the CE(S)T calendar date of its OPEN (propkit.calendar.prop_day); the day
    starts at 00:00 CE(S)T = 22:00 UTC in EU summer time, 23:00 UTC in winter (EU DST rule; the change
    days are 23 h and 25 h long). In the US-only DST shift weeks the Sunday 18:00 New York reopen is
    22:00 UTC = 23:00 CET Sunday, so that first hour is its own (Sunday) prop day. Bars must not cross
    00:00 CE(S)T (true for M15/M30/H1 bars that open on the hour); a frame where they do is refused.
    With rules.day_boundary "ny_17" or "utc_midnight" the day is propkit.calendar.firm_day instead (17:00
    New York or 00:00 UTC) and everything below that says 00:00 means that boundary; the default
    "cet_midnight" is exactly the CE(S)T prop day above.
  * Day-start values: B_00:00 = balance at the end of the last bar before the day (C0 for the first
    day); E_00:00 likewise from equity_close. Floors follow rules.daily_floor / rules.max_floor; the
    trailing max floor uses the highest B_00:00 of the days seen so far (days without bars cannot change
    the balance, so skipping them changes nothing).
  * Breach: equity_worst of a bar < the higher of the two floors (<= with rules.breach_inclusive). The
    kind is the HIGHER floor, the one equity crossed first ('max' on an exact tie). A breach in the same
    bar as a pass wins: equity_worst happens inside the bar, the pass is checked at its close.
  * Target: checked at bar closes. Passed at the first bar close where the closed balance >= C0 x (1 +
    target), no position is open (a flat bar, see Inputs; only when rules.target_requires_flat, the default),
    the best-day rule holds on the day profits so far (the current day counted up to this bar) and the
    trading days so far >= rules.min_trading_days. If the balance target is met but a condition is not,
    the scan goes on until all hold, a breach, or the end of data ("target_first_time" records the first
    bar close at which the balance target, flat, was met).
  * Trading day: a prop day with at least one trade ENTRY (FTMO counts days on which a trade was
    executed; this tool counts the entry fill).
  * Calendar days: prop days from the first bar's day to the event day, inclusive, weekends included.
  * Statistics (trading days, drawdowns, best-day share) run to the end of the challenge: the passing
    or breaching bar, or the last bar ("running"). The per-day table covers every day with bars and flags
    the days inside the challenge.
  * Worst daily drawdown = max over bars of (day-start reference - equity_worst, floored at 0), USD and
    % of C0: the quantity the daily rule limits when daily_loss_base is "initial".
  * Max drawdown = max over bars of (running peak - equity_worst), the peak being the running maximum of
    C0 and the earlier bar closes (equity_close); USD and % of C0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from propkit import calendar
from propkit.bars import bar_size_problem
from propkit.rules import PropRules

EQUITY_REQUIRED = ("time", "balance", "equity_close", "equity_worst", "units_open")
EQUITY_OPTIONAL = ("commission_usd", "realised_usd", "swap_usd")
STATUSES = ("passed", "breached_daily", "breached_max", "running")
WORST_TOL_USD = 1e-6        # equity_worst may exceed equity_close by this much (float noise only)
FLAT_TOL_OZ = 1e-9          # |units_open| at or below this counts as flat (float residue of partial closes)
FLAT_EQUITY_TOL_USD = 1e-6  # a flat bar also has equity_close == balance to this (a net-zero hedge does not)
START_TOLERANCE = 0.2       # no ledger columns: the first bar close must be within 20% of C0 (else: wrong C0)
START_LEDGER_TOL_USD = 0.01 # ledger columns: the balance before the first bar must equal C0 to this
LEDGER_COLUMNS = ("realised_usd", "commission_usd", "swap_usd")


# ---------------------------------------------------------------------------------------
# input checks

def _float_col(df: pd.DataFrame, col: str, what: str) -> np.ndarray:
    try:
        arr = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: column '{col}' must hold numbers (USD or oz)")
    if not np.isfinite(arr).all():
        i = int(np.flatnonzero(~np.isfinite(arr))[0])
        raise ValueError(f"{what}: column '{col}' has a missing or infinite value at row {i}")
    return arr


def _time_col(values, what: str) -> np.ndarray:
    s = pd.Series(values)
    if s.isna().any():
        raise ValueError(f"{what} has missing times")
    if s.dtype.kind not in "iu":
        raise ValueError(f"{what} must be int64 UTC epoch seconds (got dtype {s.dtype}); convert with "
                         "propkit.bars.to_epoch_seconds first")
    arr = s.to_numpy(dtype=np.int64)
    if arr.size and (arr.min() < calendar.MIN_TIME or arr.max() >= calendar.MAX_TIME):
        raise ValueError(f"{what} must be UTC epoch seconds between 1970 and 2200 (milliseconds? divide by 1000)")
    return arr


def equity_arrays(equity: pd.DataFrame, initial_capital: float,
                  day_boundary: str = calendar.DEFAULT_DAY_BOUNDARY) -> dict[str, Any]:
    """Validate an EQUITY frame and return its columns as numpy arrays plus derived day indices.

    day_boundary: where a day starts (propkit.calendar.firm_day; default "cet_midnight" = prop_day).

    Returns a dict with time, balance, close (equity_close), worst (equity_worst), units (units_open, with
    |units| <= 1e-9 oz set to 0), flat (bool per bar: units == 0 and |equity_close - balance| <= 1e-6 USD,
    so a net-zero hedge is not flat), commission, realised (zeros if absent), day (firm day per bar),
    day_id (0-based index of the bar's day among days with bars), starts (first bar index of each day),
    bar_seconds (inferred, or None for one bar). Raises ValueError (with the row and UTC time) for:
    missing columns, unsorted or duplicate times, non-finite values, equity_worst above equity_close,
    an inferred bar size that is a gap (propkit.bars.bar_size_problem), bars crossing the day boundary, or a
    start that does not match initial_capital: with the ledger columns realised_usd, commission_usd and
    swap_usd, the balance before the first bar (balance - realised_usd + commission_usd - swap_usd of row
    0) must equal initial_capital to 0.01 USD; without them the first bar's equity_close must be within
    20% of it (a sign the frame was built with another C0).
    """
    what = "EQUITY"
    if not isinstance(equity, pd.DataFrame):
        raise ValueError("equity must be a pandas DataFrame (the EQUITY frame from propkit.equity)")
    missing = [c for c in EQUITY_REQUIRED if c not in equity.columns]
    if missing:
        raise ValueError(f"{what} is missing column(s) {missing}; expected {', '.join(EQUITY_REQUIRED)}")
    if len(equity) == 0:
        raise ValueError(f"{what} has no rows")
    t = _time_col(equity["time"], f"{what} time")
    if t.size > 1 and (np.diff(t) <= 0).any():
        i = int(np.flatnonzero(np.diff(t) <= 0)[0]) + 1
        raise ValueError(f"{what}: row {i} ({calendar.utc_str(int(t[i]))}) is not after the row before; "
                         "times must be sorted and unique")
    out: dict[str, Any] = {"time": t}
    for key, col in (("balance", "balance"), ("close", "equity_close"), ("worst", "equity_worst"),
                     ("units", "units_open")):
        out[key] = _float_col(equity, col, what)
    out["units"] = np.where(np.abs(out["units"]) <= FLAT_TOL_OZ, 0.0, out["units"])
    for key, col in (("commission", "commission_usd"), ("realised", "realised_usd")):
        out[key] = _float_col(equity, col, what) if col in equity.columns else np.zeros(t.size)
    out["flat"] = (out["units"] == 0) & (np.abs(out["close"] - out["balance"])
                                         <= FLAT_EQUITY_TOL_USD + 1e-12 * np.abs(out["balance"]))
    bad = out["worst"] > out["close"] + WORST_TOL_USD + 1e-12 * np.abs(out["close"])
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(f"{what}: row {i} ({calendar.utc_str(int(t[i]))}) has equity_worst "
                         f"{out['worst'][i]:.2f} above equity_close {out['close'][i]:.2f}; equity_worst is "
                         "the LOWEST equity inside the bar")
    _check_start(equity, out, float(initial_capital), what)
    boundary = calendar.check_day_boundary(day_boundary)
    day = np.asarray(calendar.firm_day(t, boundary), dtype=np.int64)
    bar_seconds = None
    if t.size > 1:
        steps = np.diff(t)
        vals, counts = np.unique(steps, return_counts=True)
        k = int(np.argmax(counts))
        bar_seconds = int(vals[k])
        problem = bar_size_problem(bar_seconds, int(counts[k]), int(t.size))
        if problem:
            raise ValueError(f"{what}: {problem}")
        end_day = np.asarray(calendar.firm_day(t + bar_seconds - 1, boundary), dtype=np.int64)
        cross = end_day != day
        if cross.any():
            i = int(np.flatnonzero(cross)[0])
            raise ValueError(f"{what}: the bar at {calendar.utc_str(int(t[i]))} ({bar_seconds} s) crosses "
                             f"{calendar.boundary_label(boundary)}, the prop-day boundary; use M15, M30 or H1 bars "
                             "that open on the hour")
    new_day = np.r_[True, day[1:] != day[:-1]]
    out.update(day=day, day_id=np.cumsum(new_day) - 1, starts=np.flatnonzero(new_day), bar_seconds=bar_seconds,
               day_boundary=boundary)
    return out


def _check_start(equity: pd.DataFrame, out: dict[str, Any], c0: float, what: str) -> None:
    """Refuse an EQUITY whose start does not match C0 (USD); see equity_arrays."""
    if all(col in equity.columns for col in LEDGER_COLUMNS):
        swap0 = float(_float_col(equity, "swap_usd", what)[0])
        start = float(out["balance"][0] - out["realised"][0] + out["commission"][0] - swap0)
        if abs(start - c0) > START_LEDGER_TOL_USD + 1e-12 * c0:
            raise ValueError(f"{what}: the balance before the first bar is {start:,.2f} (balance - realised_usd + "
                             f"commission_usd - swap_usd of the first row), not the rules' initial capital "
                             f"{c0:,.2f}; build the equity with the same initial capital as the rules (--capital)")
        return
    if abs(out["close"][0] - c0) > START_TOLERANCE * c0:
        raise ValueError(f"{what} starts at equity {out['close'][0]:,.2f} at the first bar close, far from the rules' "
                         f"initial capital {c0:,.2f}: either it was built with another initial capital (build it "
                         "with the same one as the rules) or the first bar moved the account by more than 20%. "
                         "Add the columns realised_usd, commission_usd and swap_usd (propkit.equity writes them) "
                         "so the balance before the first bar can be checked exactly")


def entry_flags(arrays: dict[str, Any], trades: pd.DataFrame | None) -> np.ndarray:
    """Bool per bar: at least one trade ENTRY inside the bar.

    With TRADES: each entry_time (UTC epoch s) is placed in its bar by the same rule as propkit.equity
    (bar_of_instants): the last bar with open <= entry_time, provided entry_time <= open + bar_seconds (an
    entry exactly at a bar's end is a fill at that bar's close, which counts only when no bar opens there,
    e.g. before a weekend, the daily break or the data end); an entry before the first bar, after the last
    bar's end or inside a gap raises ValueError. Without TRADES: inferred from units_open (|units| grows or changes sign) or, when
    units_open is unchanged, from commission_usd or realised_usd booked in the bar (a round trip inside it).
    """
    t = arrays["time"]
    n = t.size
    if trades is None:
        u = arrays["units"]
        prev = np.r_[0.0, u[:-1]]
        grows = (u != 0) & ((np.abs(u) > np.abs(prev)) | (np.sign(u) != np.sign(prev)))
        round_trip = (u == prev) & ((arrays["commission"] != 0) | (arrays["realised"] != 0))
        return grows | round_trip
    if not isinstance(trades, pd.DataFrame):
        raise ValueError("trades must be a pandas DataFrame (TRADES) or None")
    if "entry_time" not in trades.columns:
        raise ValueError("TRADES is missing column 'entry_time' (UTC epoch seconds of the entry fill)")
    flags = np.zeros(n, dtype=bool)
    if len(trades) == 0:
        return flags
    et = _time_col(trades["entry_time"], "TRADES entry_time")
    idx = np.searchsorted(t, et, side="right") - 1
    bs = arrays["bar_seconds"]
    inside = idx >= 0
    if bs is not None:
        inside &= et <= t[np.maximum(idx, 0)] + bs      # same rule as equity.bar_of_instants
    if not inside.all():
        i = int(np.flatnonzero(~inside)[0])
        raise ValueError(f"TRADES row {i}: entry_time {calendar.utc_str(int(et[i]))} is not inside any bar of "
                         "EQUITY (before the first bar, after the last, or in a gap); use the same bars")
    flags[idx] = True
    return flags


# ---------------------------------------------------------------------------------------
# R-multiple summary

def r_summary(trades: pd.DataFrame | None) -> dict[str, Any] | None:
    """R-multiple statistics of TRADES with a usable risk_usd, or None.

    R = pnl_usd / risk_usd for rows with finite risk_usd > 0 (1R = entry-to-stop loss at the planned
    size incl. costs, CLAUDE.md A9). Trades are taken in (exit_time, trade_id) order when those columns
    exist. Returns n, mean_r, se_r (sample sd with ddof=1 / sqrt(n); None for n < 2), sd_r, win_rate
    (share with pnl_usd > 0), profit_factor (sum of winning pnl / -sum of losing pnl; None when there is
    no losing trade), expectancy_r (= mean_r), expectancy_usd (mean pnl_usd), max_consecutive_losses
    (longest run of pnl_usd < 0) and worst_losing_run_r (most negative sum of R over a run of
    consecutive losing trades). n_skipped counts rows without a usable risk_usd.
    """
    if trades is None or len(trades) == 0:
        return None
    if "risk_usd" not in trades.columns or "pnl_usd" not in trades.columns:
        return None
    df = trades
    order = [c for c in ("exit_time", "trade_id") if c in df.columns]
    if order:
        df = df.sort_values(order, kind="mergesort")
    risk = pd.to_numeric(df["risk_usd"], errors="coerce").to_numpy(dtype=np.float64)
    pnl = pd.to_numeric(df["pnl_usd"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(pnl).all():
        raise ValueError("TRADES pnl_usd has missing or non-numeric values")
    ok = np.isfinite(risk) & (risk > 0)
    if not ok.any():
        return None
    r = pnl[ok] / risk[ok]
    p = pnl[ok]
    n = int(r.size)
    sd = float(np.std(r, ddof=1)) if n > 1 else None
    wins, losses = p[p > 0].sum(), p[p < 0].sum()
    lose = p < 0
    best_run, run, worst_sum, cur_sum = 0, 0, 0.0, 0.0
    for is_loss, rv in zip(lose, r):
        if is_loss:
            run += 1
            cur_sum += float(rv)
            best_run = max(best_run, run)
            worst_sum = min(worst_sum, cur_sum)
        else:
            run, cur_sum = 0, 0.0
    return {
        "n": n, "n_skipped": int((~ok).sum()),
        "mean_r": float(r.mean()), "sd_r": sd, "se_r": (sd / math.sqrt(n)) if sd is not None else None,
        "win_rate": float((p > 0).mean()),
        "profit_factor": (float(wins / -losses) if losses < 0 else None),
        "expectancy_r": float(r.mean()), "expectancy_usd": float(p.mean()),
        "max_consecutive_losses": int(best_run), "worst_losing_run_r": float(worst_sum),
    }


# ---------------------------------------------------------------------------------------
# result

@dataclass(frozen=True)
class PathResult:
    """Outcome of one equity path under one rule set (USD; *_pct fields are percent of C0).

    status: 'passed' | 'breached_daily' | 'breached_max' | 'running' (no event by the end of data).
    end_time / end_index: the bar that ended the challenge (pass or breach), else the last bar.
    breach: dict(time, time_utc, index, day, date, kind, equity_worst, floor, daily_floor, max_floor)
            for the first breach inside the challenge, else None. breach_after_pass: the first breach
            after a pass (information only; the challenge was already passed), else None.
    pass_time / pass_index / pass_day: the passing bar close (None unless passed).
    target_first_time: first bar close where the balance target was met (flat if required), within the
            challenge; best_day_ok_at_target / min_days_ok_at_target: whether those conditions held then.
    trading_days, calendar_days, days_with_bars: counts up to end_index (inclusive).
    days_to_target_trading / _calendar / _with_bars: the same counts at the pass (None unless passed).
    days: per-day DataFrame (every prop day with bars): day, date, n_bars, start_balance, start_equity,
          start_ref, daily_floor, max_floor, end_balance, end_equity, min_equity_worst, day_profit (per
          rules.best_day_basis), day_profit_balance, day_profit_equity, traded, breach (a bar of the day
          breached), in_challenge.
    per_bar: DataFrame per bar: time, day, daily_floor, max_floor, headroom (equity_worst minus the
          higher floor, USD; negative = breach), trading_days (count so far).
    worst_daily_dd_usd / _pct / _date; max_dd_usd / _pct / _time; best_day_share (fraction, or None).
    r_summary: see r_summary(), or None.
    """

    status: str
    rules: PropRules
    initial_capital: float
    end_time: int
    end_index: int
    final_balance: float
    final_equity: float
    breach: dict[str, Any] | None
    breach_after_pass: dict[str, Any] | None
    pass_time: int | None
    pass_index: int | None
    pass_day: int | None
    target_first_time: int | None
    best_day_ok_at_target: bool | None
    min_days_ok_at_target: bool | None
    trading_days: int
    calendar_days: int
    days_with_bars: int
    days_to_target_trading: int | None
    days_to_target_calendar: int | None
    days_to_target_with_bars: int | None
    worst_daily_dd_usd: float
    worst_daily_dd_pct: float
    worst_daily_dd_date: str
    max_dd_usd: float
    max_dd_pct: float
    max_dd_time: int
    best_day_share: float | None
    r_summary: dict[str, Any] | None
    days: pd.DataFrame = field(repr=False)
    per_bar: pd.DataFrame = field(repr=False)

    @property
    def passed(self) -> bool:
        """True when the challenge was passed."""
        return self.status == "passed"

    def to_dict(self, include_days: bool = True) -> dict[str, Any]:
        """JSON-serialisable dict of the result (per_bar left out; days as a list of rows if asked)."""
        out: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            if name in ("days", "per_bar"):
                continue
            value = getattr(self, name)
            out[name] = value.to_dict() if isinstance(value, PropRules) else value
        out["end_time_utc"] = calendar.utc_str(self.end_time)
        out["pass_time_utc"] = calendar.utc_str(self.pass_time) if self.pass_time is not None else None
        if include_days:
            rows = self.days.to_dict(orient="records")
            out["days"] = [{k: _plain(v) for k, v in row.items()} for row in rows]
        return out

    def summary_lines(self) -> list[str]:
        """Short plain-English ASCII summary."""
        c0 = self.initial_capital
        lines = [f"Rules: {self.rules.name}; status: {self.status.upper()} at {calendar.utc_str(self.end_time)}"]
        if self.breach is not None:
            b = self.breach
            lines.append(f"  first breach ({b['kind']}) on prop day {b['date']}: equity {b['equity_worst']:,.2f} "
                         f"below the floor {b['floor']:,.2f}")
        if self.passed:
            lines.append(f"  passed after {self.days_to_target_trading} trading days "
                         f"({self.days_to_target_calendar} calendar days)")
        elif self.target_first_time is not None:
            lines.append(f"  balance target met at {calendar.utc_str(self.target_first_time)} but best-day ok = "
                         f"{self.best_day_ok_at_target}, min-days ok = {self.min_days_ok_at_target}")
        lines.append(f"  final balance {self.final_balance:,.2f}, equity {self.final_equity:,.2f} "
                     f"({(self.final_equity / c0 - 1) * 100:+.2f}% of initial capital)")
        lines.append(f"  trading days {self.trading_days}, calendar days {self.calendar_days}")
        lines.append(f"  worst daily drawdown {self.worst_daily_dd_usd:,.2f} USD = {self.worst_daily_dd_pct:.2f}% "
                     f"of initial capital (prop day {self.worst_daily_dd_date})")
        lines.append(f"  max drawdown (worst equity vs running peak) {self.max_dd_usd:,.2f} USD = "
                     f"{self.max_dd_pct:.2f}% of initial capital")
        if self.best_day_share is not None:
            lines.append(f"  best day = {self.best_day_share:.1%} of the total profit of positive days")
        if self.r_summary is not None:
            r = self.r_summary
            se = f" +- {r['se_r']:.3f}" if r["se_r"] is not None else ""
            lines.append(f"  trades with risk: {r['n']}, mean R {r['mean_r']:.3f}{se}, win rate {r['win_rate']:.1%}, "
                         f"max losing streak {r['max_consecutive_losses']}")
        return lines


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# ---------------------------------------------------------------------------------------
# core scan (shared with the bootstrap's replay checks)

def _first(mask: np.ndarray, limit: int | None = None) -> int | None:
    """Index of the first True (optionally only among indices <= limit), else None."""
    hits = np.flatnonzero(mask if limit is None else mask[: limit + 1])
    return int(hits[0]) if hits.size else None


def evaluate_path(equity: pd.DataFrame, trades: pd.DataFrame | None, rules: PropRules) -> PathResult:
    """Run one EQUITY path (and optional TRADES) through `rules`; see the module docstring.

    equity: EQUITY frame (USD; time = bar open, UTC epoch s); trades: TRADES frame or None (entry_time
    gives trading days; pnl_usd and risk_usd give the R summary); rules: PropRules (C0 =
    rules.initial_capital, the balance before the first bar; rules.day_boundary decides the days).
    Returns PathResult. Vectorised: one pass of numpy operations over all bars.
    """
    if not isinstance(rules, PropRules):
        raise ValueError("rules must be a PropRules (see propkit.rules: ftmo_1step, ftmo_2step, custom)")
    c0 = rules.initial_capital
    a = equity_arrays(equity, c0, rules.day_boundary)
    t, bal, close, worst, units = a["time"], a["balance"], a["close"], a["worst"], a["units"]
    day, day_id, starts = a["day"], a["day_id"], a["starts"]
    n, n_days = t.size, starts.size
    entry = entry_flags(a, trades)

    # day-start values and floors
    prev_bal = np.r_[c0, bal[:-1]]
    prev_eq = np.r_[c0, close[:-1]]
    b00, e00 = prev_bal[starts], prev_eq[starts]
    ref = b00 if rules.day_start_reference == "balance" else np.maximum(b00, e00)
    daily_floor_d = np.asarray(rules.daily_floor(ref), dtype=np.float64)
    max_floor_d = np.asarray(rules.max_floor(np.maximum.accumulate(b00)), dtype=np.float64)
    daily_floor, max_floor = daily_floor_d[day_id], max_floor_d[day_id]
    floor_hi = np.maximum(daily_floor, max_floor)
    breach = rules.breached(worst, floor_hi)

    # trading days so far (a day counts from its first entry bar on)
    cum_entry = np.cumsum(entry)
    before_day = cum_entry[starts] - entry[starts]
    traded_so_far = (cum_entry - before_day[day_id]) > 0
    ends = np.r_[starts[1:] - 1, n - 1]
    day_traded = traded_so_far[ends]
    tdays = np.r_[0, np.cumsum(day_traded)[:-1]][day_id] + traded_so_far

    # day profits and the best-day rule (current day counted up to each bar)
    profit_bal = bal - b00[day_id]
    profit_eq = close - e00[day_id]
    cur = profit_bal if rules.best_day_basis == "balance" else profit_eq
    pos_day = np.maximum(cur[ends], 0.0)
    best_before = np.r_[0.0, np.maximum.accumulate(pos_day)[:-1]][day_id]
    total_before = np.r_[0.0, np.cumsum(pos_day)[:-1]][day_id]
    best_now = np.maximum(best_before, cur)
    total_now = total_before + np.maximum(cur, 0.0)
    best_ok = np.asarray(rules.best_day_ok_from(best_now, total_now), dtype=bool)
    min_days_ok = tdays >= rules.min_trading_days

    # target
    if rules.target_balance is None:
        target_ok = np.zeros(n, dtype=bool)
    else:
        target_ok = bal >= rules.target_balance
        if rules.target_requires_flat:
            target_ok &= a["flat"]
    passing = target_ok & best_ok & min_days_ok

    i_breach, i_pass = _first(breach), _first(passing)
    if i_breach is not None and (i_pass is None or i_breach <= i_pass):
        end = i_breach
        status = "breached_daily" if daily_floor[end] > max_floor[end] else "breached_max"
    elif i_pass is not None:
        end, status = i_pass, "passed"
    else:
        end, status = n - 1, "running"

    def breach_info(i: int) -> dict[str, Any]:
        kind = "daily" if daily_floor[i] > max_floor[i] else "max"
        return {"time": int(t[i]), "time_utc": calendar.utc_str(int(t[i])), "index": int(i),
                "day": int(day[i]), "date": calendar.day_to_str(int(day[i])), "kind": kind,
                "equity_worst": float(worst[i]), "floor": float(floor_hi[i]),
                "daily_floor": float(daily_floor[i]), "max_floor": float(max_floor[i])}

    breach_rec = breach_info(end) if status.startswith("breached") else None
    after = None
    if status == "passed":
        later = np.flatnonzero(breach[end + 1:])
        after = breach_info(end + 1 + int(later[0])) if later.size else None

    i_target = _first(target_ok, end)
    if i_target is not None and status.startswith("breached") and i_target >= end:
        i_target = None                     # the breach inside that bar came before its close
    first_day = int(day[0])

    # drawdowns within the challenge
    dd_daily = np.maximum(ref[day_id][: end + 1] - worst[: end + 1], 0.0)
    k_daily = int(np.argmax(dd_daily))
    peak = np.maximum.accumulate(prev_eq)
    dd = np.maximum(peak - worst, 0.0)[: end + 1]
    k_dd = int(np.argmax(dd))

    # best-day share at the end
    share_profits = np.r_[cur[ends][: day_id[end]], cur[end]]
    best_share = PropRules.best_day_share(share_profits)

    days = pd.DataFrame({
        "day": day[starts], "date": np.asarray(calendar.day_to_str(day[starts])),
        "n_bars": np.diff(np.r_[starts, n]),
        "start_balance": b00, "start_equity": e00, "start_ref": ref,
        "daily_floor": daily_floor_d, "max_floor": max_floor_d,
        "end_balance": bal[ends], "end_equity": close[ends],
        "min_equity_worst": np.minimum.reduceat(worst, starts),
        "day_profit": cur[ends], "day_profit_balance": profit_bal[ends], "day_profit_equity": profit_eq[ends],
        "traded": day_traded, "breach": np.logical_or.reduceat(breach, starts),
        "in_challenge": np.arange(n_days) <= day_id[end],
    })
    per_bar = pd.DataFrame({"time": t, "day": day, "daily_floor": daily_floor, "max_floor": max_floor,
                            "headroom": worst - floor_hi, "trading_days": tdays})
    passed = status == "passed"
    return PathResult(
        status=status, rules=rules, initial_capital=c0, end_time=int(t[end]), end_index=int(end),
        final_balance=float(bal[end]), final_equity=float(close[end]),
        breach=breach_rec, breach_after_pass=after,
        pass_time=int(t[end]) if passed else None, pass_index=int(end) if passed else None,
        pass_day=int(day[end]) if passed else None,
        target_first_time=int(t[i_target]) if i_target is not None else None,
        best_day_ok_at_target=bool(best_ok[i_target]) if i_target is not None else None,
        min_days_ok_at_target=bool(min_days_ok[i_target]) if i_target is not None else None,
        trading_days=int(tdays[end]), calendar_days=int(day[end]) - first_day + 1,
        days_with_bars=int(day_id[end]) + 1,
        days_to_target_trading=int(tdays[end]) if passed else None,
        days_to_target_calendar=int(day[end]) - first_day + 1 if passed else None,
        days_to_target_with_bars=int(day_id[end]) + 1 if passed else None,
        worst_daily_dd_usd=float(dd_daily[k_daily]), worst_daily_dd_pct=float(dd_daily[k_daily] / c0 * 100.0),
        worst_daily_dd_date=calendar.day_to_str(int(day[k_daily])),
        max_dd_usd=float(dd[k_dd]), max_dd_pct=float(dd[k_dd] / c0 * 100.0), max_dd_time=int(t[k_dd]),
        best_day_share=best_share, r_summary=r_summary(trades), days=days, per_bar=per_bar,
    )
