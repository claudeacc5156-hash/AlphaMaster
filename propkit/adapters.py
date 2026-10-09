"""propkit/adapters.py - TRADES and POSITIONS tables: checks, CSV files, the AlphaMaster adapter and the
position-series -> trade-list conversion used by propkit.equity.equity_from_positions.

TRADES (pandas DataFrame), one row per round trip (TRADES_COLUMNS, in this order):
  trade_id        int, unique (1..n when the input has none);
  side            int, +1 long / -1 short ("long"/"short"/"buy"/"sell" text is accepted on input);
  units           float > 0, troy ounces (1 lot = CostModel.lot_size_oz oz);
  entry_time      int64 UTC epoch seconds of the entry FILL (a bar open or an instant inside a bar);
  entry_price     float > 0, USD/oz, the actual fill (long at ask, short at bid, markup and slippage in);
  exit_time       int64 UTC epoch seconds of the exit fill (>= entry_time);
  exit_price      float > 0, USD/oz, the actual fill (long at bid, short at ask, costs in);
  exit_reason     str, one of EXIT_REASONS ("unknown" when the input has no exit_reason column);
  stop_price      float or NaN, the INITIAL stop (below the entry for a long, above it for a short);
  risk_usd        float > 0 or NaN, 1R = the loss of a stop exit at the planned size: units x
                  |entry_price - stop fill| + round-trip commission at that size, where the stop fill is
                  stop_price minus (long) or plus (short) the exit markup and slippage; stop_price is a
                  BID level for a long and an ASK level for a short (CLAUDE.md A9: "entry-to-stop loss at
                  planned size incl. costs"; equity.planned_stop_fill);
  commission_usd  float, USD paid, POSITIVE = paid (subtracted from PnL);
  swap_usd        float, USD, signed: NEGATIVE = paid (added to PnL);
  pnl_usd         float, USD, net: side x units x (exit_price - entry_price) - commission_usd + swap_usd.
The last three (and risk_usd when a stop is given) are filled in by propkit.equity.equity_from_trades.

POSITIONS (pandas DataFrame): `time` (int64, bar OPEN, UTC epoch seconds) and `position` (float in
[-1, 1]): the signed fraction of the configured size held DURING that bar, from its open to the next
bar's open. A change happens at the bar's open and fills at that open: buys at bid open + spread,
sells at bid open, plus markup and slippage (CostModel.buy_fill / sell_fill).

AlphaMaster alignment (model_core/backtest.py, data_pipeline target_ret): the miner's p[t] is decided at
the CLOSE of bar t and earns target_ret[t] = log(open[t+2] / open[t+1]), i.e. it is HELD during bar t+1.
positions_from_alphamaster therefore shifts by one bar: held[t+1] = p[t], held[first bar] = 0.

Research only: nothing here places or prepares orders.
"""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from propkit import bars as bars_mod
from propkit import calendar
from propkit.costs import CostModel

TRADES_COLUMNS = ("trade_id", "side", "units", "entry_time", "entry_price", "exit_time", "exit_price",
                  "exit_reason", "stop_price", "risk_usd", "commission_usd", "swap_usd", "pnl_usd")
TRADES_REQUIRED = ("side", "units", "entry_time", "entry_price", "exit_time", "exit_price")
TRADES_TEXT_COLUMNS = ("entry_time_utc", "exit_time_utc")      # written for people, ignored on reading
EQUITY_COLUMNS = ("time", "balance", "equity_close", "equity_worst", "units_open", "realised_usd",
                  "commission_usd", "swap_usd")                # built by propkit.equity
POSITIONS_COLUMNS = ("time", "position")
POSITIONS_EXTRA_COLUMNS = ("p_raw", "time_utc")                # audit columns allowed in a positions file
EXIT_REASONS = ("stop", "target", "trail", "time", "signal", "end_of_data", "unknown")
SIZE_MODES = ("units", "leverage")
_SIDE_WORDS = {"long": 1, "buy": 1, "+1": 1, "1": 1, "short": -1, "sell": -1, "-1": -1}


# ---------------------------------------------------------------------------------------
# small checks

def _finite_number(value, what: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{what} must be a number, got {value!r}")
    x = float(value)
    if not math.isfinite(x) or (positive and x <= 0):
        raise ValueError(f"{what} must be a finite number{' > 0' if positive else ''}, got {value!r}")
    return x


def _float_col(df: pd.DataFrame, col: str, source: str) -> np.ndarray:
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        raise ValueError(f"{source}: column '{col}' must hold numbers, not True/False")
    if not pd.api.types.is_numeric_dtype(s):
        conv = pd.to_numeric(s, errors="coerce")
        bad = conv.isna() & s.notna() & (s.astype(str).str.strip() != "")
        if bad.any():
            i = int(np.flatnonzero(bad.to_numpy())[0])
            raise ValueError(f"{source}: column '{col}' row {i} is not a number: {s.iloc[i]!r}")
        s = conv
    return s.to_numpy(dtype=np.float64, na_value=np.nan)


def _first_bad(bad: np.ndarray, source: str, problem: str) -> None:
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(f"{source}: row {i} {problem} ({int(bad.sum())} row(s) affected)")


def _check_columns(df: pd.DataFrame, required, allowed, source: str, allow_extra: bool) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing column(s) {missing}; required: {list(required)}")
    if not allow_extra:
        extra = [c for c in df.columns if c not in allowed]
        if extra:
            raise ValueError(f"{source} has unknown column(s) {extra}; allowed: {list(allowed)} "
                             "(check the spelling, or remove the column)")


# ---------------------------------------------------------------------------------------
# TRADES

def empty_trades() -> pd.DataFrame:
    """A TRADES DataFrame with no rows and the standard columns and dtypes."""
    return _trades_frame(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64),
                         np.zeros(0), np.zeros(0, np.int64), np.zeros(0), [], np.zeros(0), np.zeros(0),
                         np.zeros(0), np.zeros(0), np.zeros(0))


def _trades_frame(trade_id, side, units, entry_time, entry_price, exit_time, exit_price, exit_reason,
                  stop_price, risk_usd, commission_usd, swap_usd, pnl_usd) -> pd.DataFrame:
    return pd.DataFrame({
        "trade_id": np.asarray(trade_id, dtype=np.int64),
        "side": np.asarray(side, dtype=np.int64),
        "units": np.asarray(units, dtype=np.float64),
        "entry_time": np.asarray(entry_time, dtype=np.int64),
        "entry_price": np.asarray(entry_price, dtype=np.float64),
        "exit_time": np.asarray(exit_time, dtype=np.int64),
        "exit_price": np.asarray(exit_price, dtype=np.float64),
        "exit_reason": pd.Series(list(exit_reason), dtype=object),
        "stop_price": np.asarray(stop_price, dtype=np.float64),
        "risk_usd": np.asarray(risk_usd, dtype=np.float64),
        "commission_usd": np.asarray(commission_usd, dtype=np.float64),
        "swap_usd": np.asarray(swap_usd, dtype=np.float64),
        "pnl_usd": np.asarray(pnl_usd, dtype=np.float64),
    })


def _sides(df: pd.DataFrame, source: str) -> np.ndarray:
    s = df["side"]
    if pd.api.types.is_bool_dtype(s):
        raise ValueError(f"{source}: side must be +1 (long) or -1 (short), not True/False")
    if pd.api.types.is_numeric_dtype(s):
        v = s.to_numpy(dtype=np.float64, na_value=np.nan)
    else:
        words = s.astype(str).str.strip().str.lower()
        v = np.array([_SIDE_WORDS.get(w, np.nan) for w in words], dtype=np.float64)
    _first_bad(~np.isin(v, (1.0, -1.0)), source, "has a side that is not +1 / -1 (or long / short)")
    return v.astype(np.int64)


def validate_trades(trades: pd.DataFrame | None, source: str = "trades",
                    allow_extra: bool = False) -> pd.DataFrame:
    """Check a trade list and return a new TRADES DataFrame (columns TRADES_COLUMNS, in order).

    Required columns: side, units, entry_time, entry_price, exit_time, exit_price. Optional: trade_id
    (default 1..n; must be unique whole numbers), exit_reason (default "unknown"; one of EXIT_REASONS),
    stop_price (NaN = none), risk_usd (NaN = not given), commission_usd / swap_usd / pnl_usd (kept as
    given; equity_from_trades recomputes them). Times: UTC epoch seconds (see bars.to_epoch_seconds for
    the other accepted forms). Each failure raises ValueError naming the first bad row. The row order is
    kept. None or a table with no rows gives an empty TRADES. allow_extra=False refuses unknown columns
    (the derived text columns entry_time_utc / exit_time_utc are dropped).
    """
    if trades is None:
        return empty_trades()
    if not isinstance(trades, pd.DataFrame):
        raise ValueError(f"{source}: expected a pandas DataFrame of trades")
    df = trades.drop(columns=[c for c in TRADES_TEXT_COLUMNS if c in trades.columns])
    if len(df) == 0 and not set(TRADES_REQUIRED) <= set(df.columns):
        return empty_trades()
    _check_columns(df, TRADES_REQUIRED, TRADES_COLUMNS, source, allow_extra)
    n = len(df)
    side = _sides(df, source)
    units = _float_col(df, "units", source)
    _first_bad(~np.isfinite(units) | (units <= 0), source, "has units that are missing, infinite or <= 0 "
               "(units = size in ounces, always positive; the side gives the direction)")
    entry_time = bars_mod.to_epoch_seconds(df["entry_time"], f"{source} entry_time")
    exit_time = bars_mod.to_epoch_seconds(df["exit_time"], f"{source} exit_time")
    if n:
        calendar._as_seconds(entry_time, f"{source} entry_time")
        calendar._as_seconds(exit_time, f"{source} exit_time")
    _first_bad(exit_time < entry_time, source, "exits before it enters (exit_time < entry_time)")
    prices = {}
    for col in ("entry_price", "exit_price"):
        prices[col] = _float_col(df, col, source)
        _first_bad(~np.isfinite(prices[col]) | (prices[col] <= 0), source,
                   f"has a {col} that is missing, infinite or <= 0")
    if "trade_id" in df.columns:
        tid = _float_col(df, "trade_id", source)
        _first_bad(~np.isfinite(tid) | (tid != np.floor(tid)), source, "has a trade_id that is not a whole number")
        tid = tid.astype(np.int64)
        dup = pd.Series(tid).duplicated().to_numpy()
        _first_bad(dup, source, "repeats a trade_id; trade ids must be unique")
    else:
        tid = np.arange(1, n + 1, dtype=np.int64)
    if "exit_reason" in df.columns:
        reasons = [("unknown" if (isinstance(r, float) and math.isnan(r)) or r is None else str(r).strip().lower())
                   for r in df["exit_reason"].tolist()]
        reasons = ["unknown" if r in ("", "nan", "none") else r for r in reasons]
        bad = np.array([r not in EXIT_REASONS for r in reasons], dtype=bool)
        _first_bad(bad, source, f"has an exit_reason that is not one of {list(EXIT_REASONS)}")
    else:
        reasons = ["unknown"] * n
    stop = _float_col(df, "stop_price", source) if "stop_price" in df.columns else np.full(n, np.nan)
    _first_bad(np.isinf(stop) | (stop <= 0), source, "has a stop_price that is infinite or <= 0 (use NaN/empty "
               "for no stop)")
    has_stop = np.isfinite(stop)
    wrong = has_stop & (((side > 0) & (stop >= prices["entry_price"])) | ((side < 0) & (stop <= prices["entry_price"])))
    _first_bad(wrong, source, "has its initial stop on the wrong side of the entry (a long's stop must be "
               "below entry_price, a short's above)")
    risk = _float_col(df, "risk_usd", source) if "risk_usd" in df.columns else np.full(n, np.nan)
    _first_bad(np.isinf(risk) | (risk <= 0), source, "has a risk_usd that is infinite or <= 0 (use NaN/empty "
               "when unknown)")
    ledger = {}
    for col in ("commission_usd", "swap_usd", "pnl_usd"):
        ledger[col] = _float_col(df, col, source) if col in df.columns else np.full(n, np.nan)
        _first_bad(np.isinf(ledger[col]), source, f"has an infinite {col}")
    out = _trades_frame(tid, side, units, entry_time, prices["entry_price"], exit_time, prices["exit_price"],
                        reasons, stop, risk, ledger["commission_usd"], ledger["swap_usd"], ledger["pnl_usd"])
    if allow_extra:
        for col in df.columns:
            if col not in TRADES_COLUMNS:
                out[col] = df[col].to_numpy()
    return out


def read_trades_csv(path, allow_extra: bool = False) -> pd.DataFrame:
    """Read a TRADES CSV (see the module docstring for the columns) and return validated TRADES.

    Column names are matched case-insensitively. Locked-holdout paths are refused before anything is
    opened (bars.LockedPathError); a missing or unreadable file raises ValueError.
    """
    df = _read_csv(path, "trades file")
    return validate_trades(df, source=Path(str(path)).name, allow_extra=allow_extra)


def write_trades_csv(trades: pd.DataFrame, path, utc_text: bool = True) -> Path:
    """Write TRADES to a CSV file (ASCII, no index) and return its path.

    Times are written as UTC epoch seconds; utc_text=True adds entry_time_utc / exit_time_utc text columns
    for people (read_trades_csv ignores them). Columns beyond TRADES_COLUMNS are written after them; read
    such a file back with read_trades_csv(path, allow_extra=True). Refuses locked-holdout paths; the folder
    must exist.
    """
    out = validate_trades(trades, allow_extra=True)
    if utc_text:
        out["entry_time_utc"] = calendar.utc_str(out["entry_time"].to_numpy()) if len(out) else []
        out["exit_time_utc"] = calendar.utc_str(out["exit_time"].to_numpy()) if len(out) else []
    return _write_csv(out, path, "trades file")


# ---------------------------------------------------------------------------------------
# POSITIONS

def validate_positions(positions: pd.DataFrame, bars: pd.DataFrame | None = None, source: str = "positions",
                       allow_extra: bool = False) -> pd.DataFrame:
    """Check a held-position series and return POSITIONS (time int64, position float64, plus any allowed
    audit columns such as p_raw).

    time: bar OPEN times, UTC epoch seconds, strictly increasing; position: finite, in [-1, 1], the signed
    fraction of the configured size held DURING that bar (see the module docstring). With bars given, the
    times must be a run of consecutive bar times of BARS (the same data file); bars before or after the
    run are taken as flat. Extra columns: p_raw (the unshifted miner p, for audit) is carried through and
    time_utc (text) is dropped; any other column is refused unless allow_extra=True (then carried through).
    """
    if not isinstance(positions, pd.DataFrame):
        raise ValueError(f"{source}: expected a pandas DataFrame with columns time, position")
    allowed = POSITIONS_COLUMNS + POSITIONS_EXTRA_COLUMNS
    _check_columns(positions, POSITIONS_COLUMNS, allowed, source, allow_extra)
    if len(positions) == 0:
        raise ValueError(f"{source} has no rows")
    times = bars_mod.to_epoch_seconds(positions["time"], f"{source} time")
    calendar._as_seconds(times, f"{source} time")
    _first_bad(np.r_[False, np.diff(times) <= 0], source, "has a time that is not after the row before; "
               "times must be strictly increasing (sorted, no duplicates)")
    pos = _float_col(positions, "position", source)
    _first_bad(~np.isfinite(pos), source, "has a missing or infinite position")
    _first_bad(np.abs(pos) > 1.0, source, "has a position outside [-1, 1] (position = signed fraction of the "
               "configured size; scale it, or raise --size instead)")
    out = pd.DataFrame({"time": times, "position": pos + 0.0})
    for col in positions.columns:
        if col not in POSITIONS_COLUMNS and col != "time_utc":
            out[col] = positions[col].to_numpy()
    if bars is not None:
        _positions_bar_offset(out["time"].to_numpy(), np.asarray(bars["time"], dtype=np.int64), source)
    return out


def _positions_bar_offset(pos_times: np.ndarray, bar_times: np.ndarray, source: str) -> int:
    """Index of the first positions row in the bars; raises unless the times are consecutive bar times."""
    i0 = int(np.searchsorted(bar_times, pos_times[0]))
    m = len(pos_times)
    seg = bar_times[i0:i0 + m]
    if len(seg) != m or not np.array_equal(seg, pos_times):
        diff = np.flatnonzero(seg != pos_times[:len(seg)])
        j = int(diff[0]) if diff.size else min(len(seg), m - 1)    # first mismatch, else first row past the bars
        raise ValueError(f"{source}: row {j} time {calendar.utc_str(int(pos_times[j]))} does not match the bar "
                         "times; the positions must be one row per bar for a run of consecutive bars of the "
                         "same data file (export them from the bar file you evaluate)")
    return i0


def read_positions_csv(path, allow_extra: bool = False) -> pd.DataFrame:
    """Read a POSITIONS CSV (time, position[, p_raw][, time_utc]) and return validated POSITIONS.

    Column names are matched case-insensitively; other columns are refused unless allow_extra=True.
    Refuses locked-holdout paths before anything is opened; a missing or unreadable file raises ValueError.
    """
    df = _read_csv(path, "positions file")
    return validate_positions(df, source=Path(str(path)).name, allow_extra=allow_extra)


def write_positions_csv(positions: pd.DataFrame, path, utc_text: bool = True) -> Path:
    """Write POSITIONS to a CSV file (ASCII, no index); utc_text=True adds a time_utc text column."""
    out = validate_positions(positions)
    if utc_text:
        out["time_utc"] = calendar.utc_str(out["time"].to_numpy())
    return _write_csv(out, path, "positions file")


def write_equity_csv(equity: pd.DataFrame, path, utc_text: bool = True,
                     day_boundary: str = calendar.DEFAULT_DAY_BOUNDARY) -> Path:
    """Write an EQUITY table to a CSV file (ASCII, no index); utc_text=True adds time_utc and prop_day
    ('YYYY-MM-DD', the CE(S)T date, or the firm day of day_boundary: propkit.calendar.firm_day) text
    columns. Refuses locked-holdout paths."""
    if not isinstance(equity, pd.DataFrame) or not set(EQUITY_COLUMNS) <= set(equity.columns):
        raise ValueError(f"write_equity_csv needs an EQUITY DataFrame with columns {list(EQUITY_COLUMNS)} "
                         "(from propkit.equity)")
    out = equity.copy()
    if utc_text and len(out):
        t = out["time"].to_numpy(dtype=np.int64)
        out["time_utc"] = calendar.utc_str(t)
        out["prop_day"] = calendar.day_to_str(calendar.firm_day(t, day_boundary))
    return _write_csv(out, path, "equity file")


def _read_csv(path, what: str) -> pd.DataFrame:
    bars_mod.check_not_locked(path, what=what)
    p = Path(str(path)).expanduser()
    if not p.is_file():
        raise ValueError(f"{what} not found: {p}")
    try:
        df = pd.read_csv(p, float_precision="round_trip")
    except Exception as e:   # the CSV parser raises several types
        raise ValueError(f"cannot read {p.name}: {type(e).__name__}: {e}")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _write_csv(df: pd.DataFrame, path, what: str) -> Path:
    bars_mod.check_not_locked(path, what=what, verb="write")
    p = Path(str(path)).expanduser()
    if p.suffix.lower() != ".csv":
        raise ValueError(f"the {what} must end in .csv (got {p.name})")
    if not p.parent.is_dir():
        raise ValueError(f"the folder for the {what} does not exist: {p.parent}")
    if p.is_dir():
        raise ValueError(f"the {what} path is a folder: {p}")
    # text columns (a comment, a tag) may hold any character: it is written as a backslash escape
    # (\uXXXX) so the file stays ASCII instead of failing half-way through
    df.to_csv(p, index=False, encoding="ascii", errors="backslashreplace", lineterminator="\n")
    return p


# ---------------------------------------------------------------------------------------
# AlphaMaster

def positions_from_alphamaster(times, p, keep_raw: bool = False) -> pd.DataFrame:
    """POSITIONS from the miner's position series, shifted by ONE bar: held[t+1] = p[t], held[0] = 0.

    AlphaMaster's p[t] (in [-1, 1]) is decided at the close of bar t and earns target_ret[t] =
    log(open[t+2] / open[t+1]) (model_core/backtest.py, data_pipeline target_ret), i.e. it is held during
    bar t+1, entered at bar t+1's open. The last p is never held (there is no bar after it).
    times: bar open times (UTC epoch seconds, strictly increasing), one per p. keep_raw=True adds the
    unshifted p as column p_raw (for audit: row t+1 then shows p_raw = p[t+1], position = p[t]).
    """
    t = np.asarray(times)
    arr, _ = calendar._as_seconds(t, "times")
    q = np.asarray(p, dtype=np.float64)
    if arr.ndim != 1 or q.ndim != 1 or len(arr) != len(q):
        raise ValueError(f"times and p must be 1-D arrays of the same length (got {arr.shape} and {q.shape})")
    if len(q) == 0:
        raise ValueError("p is empty")
    held = np.concatenate(([0.0], q[:-1]))
    df = pd.DataFrame({"time": arr.astype(np.int64), "position": held})
    if keep_raw:
        df["p_raw"] = q
    return validate_positions(df, source="AlphaMaster positions")


def alphamaster_log_pnl(p, open_, cost_rate: float) -> np.ndarray:
    """The miner's own per-bar log PnL, in fractions of equity (log returns), as model_core/backtest.py.

    PnL[t] = p[t] x log(open[t+2] / open[t+1]) - |p[t] - p[t-1]| x cost_rate, with p[-1] = 0 and the
    last two target returns (t = T-2, T-1) set to 0 (as data_pipeline's target_ret). p: positions in
    [-1, 1] decided at bar t's close; open_: BID opens (USD/oz, > 0); cost_rate: fraction of notional per
    unit of position change (AlphaMaster's COST_RATE, e.g. 0.0003; >= 0). Returns a float64 array of
    length T.
    Use it to check a propkit cost model against the miner (flat_rate_per_side = cost_rate, swap off).
    """
    q = np.asarray(p, dtype=np.float64)
    o = np.asarray(open_, dtype=np.float64)
    if q.ndim != 1 or o.shape != q.shape:
        raise ValueError(f"p and open_ must be 1-D arrays of the same length (got {q.shape} and {o.shape})")
    if not np.isfinite(q).all() or not np.isfinite(o).all() or (o <= 0).any():
        raise ValueError("p must be finite and open_ finite and > 0")
    rate = _finite_number(cost_rate, "cost_rate")
    if rate < 0:
        raise ValueError(f"cost_rate is a cost, a fraction of notional >= 0 (AlphaMaster's COST_RATE, e.g. "
                         f"0.0003); got {cost_rate!r}")
    target = np.zeros(q.shape, dtype=np.float64)
    if q.size >= 3:
        target[:-2] = np.log(o[2:] / o[1:-1])
    prev = np.concatenate(([0.0], q[:-1]))
    return q * target - np.abs(q - prev) * rate


# ---------------------------------------------------------------------------------------
# POSITIONS -> TRADES

def trades_from_positions(bars: pd.DataFrame, positions: pd.DataFrame, cost_model: CostModel,
                          size_mode: str = "units", size: float = 1.0, C0: float | None = None) -> pd.DataFrame:
    """Convert a held-position series into TRADES (FIFO lots, one row per closed lot or part of a lot).

    Sizing (target signed units held during bar k, re-computed only at bars where the position CHANGES):
      size_mode "units":    position 1.0 = `size` oz, so target = position x size;
      size_mode "leverage": position 1.0 = a notional of size x the current equity, so target =
                            position x size x E_k / open_k, with E_k = the balance at the end of bar k-1
                            (C0 + realised PnL, commissions and swaps booked so far) + the open lots marked
                            at bar k's open (longs at bid open, shorts at ask open = open + spread), and
                            open_k the BID open (USD/oz). C0 is required. While the position is unchanged
                            the units stay fixed (no re-sizing as equity moves).
    No lot-step rounding is applied (fractional ounces are kept).
    Fills: every change happens at bar k's open: buys at cost_model.buy_fill(open_k, spread_k), sells at
    cost_model.sell_fill(open_k, spread_k) (spread from cost_model.bar_spreads). An increase on the same
    side opens a new lot; a reduction closes the OLDEST lots first (FIFO; a lot partly closed gives a
    trade row for the closed part, pro rata, and keeps the rest with its original entry); a change of side
    closes everything, then opens the new side. Positions after the last row of POSITIONS are flat: an
    open position is closed at the next bar's open, or at the last bar's CLOSE (time = last open +
    bar_seconds, bid close / ask close + costs) when the positions run to the end of the bars; those exits
    are flagged "end_of_data", all others "signal". stop_price and risk_usd are NaN; commission_usd,
    swap_usd and pnl_usd are NaN until propkit.equity.equity_from_trades fills them.
    Returns TRADES sorted by entry_time then exit_time, trade_id 1..n.
    """
    if not isinstance(cost_model, CostModel):
        raise ValueError("cost_model must be a propkit.costs.CostModel")
    if size_mode not in SIZE_MODES:
        raise ValueError(f"size_mode must be one of {SIZE_MODES}, got {size_mode!r}")
    size = _finite_number(size, "size", positive=True)
    if size_mode == "leverage":
        if C0 is None:
            raise ValueError("size_mode 'leverage' sizes from the current equity: pass C0 (initial capital, USD)")
        C0 = _finite_number(C0, "C0", positive=True)
    b = bars_mod.validate_bars(bars, source="bars")
    pos = validate_positions(positions, b, allow_extra=True)
    times = b["time"].to_numpy(dtype=np.int64)
    o = b["open"].to_numpy(dtype=np.float64)
    c = b["close"].to_numpy(dtype=np.float64)
    spread = cost_model.bar_spreads(b)
    n = len(times)
    bar_seconds = bars_mod.infer_bar_seconds(times)
    i0 = _positions_bar_offset(pos["time"].to_numpy(), times, "positions")
    m = len(pos)
    held = np.zeros(n, dtype=np.float64)
    held[i0:i0 + m] = pos["position"].to_numpy(dtype=np.float64)
    prev = np.concatenate(([0.0], held[:-1]))
    changes = np.flatnonzero(held != prev)
    end_bar = i0 + m                       # first bar after the positions (== n when they reach the end)
    builder = _LotBook(cost_model)
    leverage = size_mode == "leverage"
    if leverage:
        ledger = _LeverageLedger(C0, cost_model, times, o, c, bar_seconds)
    for k in changes:
        k = int(k)
        tk = int(times[k])
        if leverage:
            equity_open = ledger.equity_at_open(builder, tk, o[k], spread[k])
            if held[k] != 0 and not equity_open > 0:
                raise ValueError(f"equity at {calendar.utc_str(tk)} is {equity_open:.2f} USD (<= 0): leverage "
                                 "sizing is impossible; use a smaller --size")
            target = held[k] * size * equity_open / o[k]
        else:
            target = held[k] * size
        reason = "end_of_data" if k == end_bar else "signal"
        cash = builder.change_to(target, tk, o[k], spread[k], reason)
        if leverage:
            ledger.book(cash)
    if builder.units > 0:
        last = n - 1
        builder.change_to(0.0, int(times[last]) + bar_seconds, c[last], spread[last], "end_of_data")
    return builder.frame()


class _LotBook:
    """FIFO lots of one net position (all lots on the same side) and the trade rows closed so far.

    Running totals (units, units x entry price) avoid re-summing the lots at every change; they are
    reset to exactly 0 whenever the book is flat.
    """

    def __init__(self, cost_model: CostModel):
        self.cm = cost_model
        self.side = 0
        self.lots: deque[list] = deque()        # [units, entry_time, entry_price]
        self.units = 0.0                        # total open units (oz, >= 0)
        self.cost = 0.0                         # sum of units x entry_price (USD)
        self.rows: list[tuple] = []

    def unrealised(self, bid: float, spread: float) -> float:
        """USD PnL of the open lots marked at bid (longs) or bid + spread (shorts)."""
        if not self.lots:
            return 0.0
        mark = bid if self.side > 0 else bid + spread
        return self.side * (mark * self.units - self.cost)

    def change_to(self, target: float, t: int, bid: float, spread: float, reason: str) -> float:
        """Move the net position to `target` signed units at instant t; returns the cash booked (USD:
        realised PnL - commissions)."""
        current = self.side * self.units
        tol = 1e-12 * max(1.0, abs(target), abs(current))
        if abs(target - current) <= tol:
            return 0.0
        cash = 0.0
        new_side = 1 if target > tol else (-1 if target < -tol else 0)
        if self.lots and new_side != self.side:
            cash += self._close(math.inf, t, bid, spread, reason)
        if new_side == 0:
            return cash
        want, have = abs(target), self.units
        if want > have + tol:
            cash += self._open(new_side, want - have, t, bid, spread)
        elif want < have - tol:
            cash += self._close(have - want, t, bid, spread, reason)
        return cash

    def _open(self, side: int, units: float, t: int, bid: float, spread: float) -> float:
        price = self.cm.buy_fill(bid, spread) if side > 0 else self.cm.sell_fill(bid, spread)
        self.side = side
        self.lots.append([units, t, price])
        self.units += units
        self.cost += units * price
        return -self.cm.fill_commission(units, price)

    def _close(self, amount: float, t: int, bid: float, spread: float, reason: str) -> float:
        price = self.cm.sell_fill(bid, spread) if self.side > 0 else self.cm.buy_fill(bid, spread)
        cash = 0.0
        left = amount
        while self.lots and left > 0:
            lot = self.lots[0]
            whole = lot[0] <= left * (1 + 1e-12) or (len(self.lots) == 1 and left >= lot[0] * (1 - 1e-12))
            q = lot[0] if whole else left
            if whole:
                self.lots.popleft()
            else:
                lot[0] -= q
            left -= q
            self.units -= q
            self.cost -= q * lot[2]
            self.rows.append((self.side, q, lot[1], lot[2], t, price, reason))
            cash += self.side * q * (price - lot[2]) - self.cm.fill_commission(q, price)
        if not self.lots:
            self.side, self.units, self.cost = 0, 0.0, 0.0
        return cash

    def frame(self) -> pd.DataFrame:
        if not self.rows:
            return empty_trades()
        side, units, et, ep, xt, xp, reason = (list(col) for col in zip(*self.rows))
        order = np.lexsort((np.asarray(xt), np.asarray(et)))
        k = len(order)
        nan = np.full(k, np.nan)
        pick = [np.asarray(v, dtype=object)[order] for v in (side, units, et, ep, xt, xp, reason)]
        return _trades_frame(np.arange(1, k + 1), pick[0].astype(np.int64), pick[1].astype(np.float64),
                             pick[2].astype(np.int64), pick[3].astype(np.float64), pick[4].astype(np.int64),
                             pick[5].astype(np.float64), list(pick[6]), nan, nan, nan, nan, nan)


class _LeverageLedger:
    """Running balance for leverage sizing: realised cash, commissions and swaps booked before a bar opens.

    Swaps follow propkit.equity: one charge per rollover r with entry < r <= exit, booked in the bar
    containing r (last bar with open <= r), on the bid at r (bars.rollover_bids); a rollover at exactly a
    bar's open belongs to that bar, so it is not yet in the balance at that open.
    """

    def __init__(self, C0: float, cost_model: CostModel, times: np.ndarray, open_: np.ndarray, close: np.ndarray,
                 bar_seconds: int):
        self.cm = cost_model
        self.balance = C0
        r = calendar.rollover_instants(int(times[0]), int(times[-1]) + bar_seconds + 1, cost_model.rollover_hour_ny)
        self.r = r
        self.r_close = bars_mod.rollover_bids(times, open_, close, r)
        self.r_weekday = np.asarray(calendar.ny_weekday(r), dtype=np.int64) if r.size else np.zeros(0, np.int64)
        self.ptr = 0

    def _swaps_until(self, book: _LotBook, j: int) -> None:
        if j > self.ptr and book.lots:
            sl = slice(self.ptr, j)
            sw = self.cm.swap_for_night(np.full(j - self.ptr, book.side), np.full(j - self.ptr, book.units),
                                        self.r_close[sl], self.r_weekday[sl])
            self.balance += float(np.sum(sw))
        self.ptr = max(self.ptr, j)

    def equity_at_open(self, book: _LotBook, t: int, bid_open: float, spread: float) -> float:
        """Equity at the open instant t (before the change there); then books swaps of rollovers at t."""
        self._swaps_until(book, int(np.searchsorted(self.r, t, side="left")))
        equity = self.balance + book.unrealised(bid_open, spread)
        self._swaps_until(book, int(np.searchsorted(self.r, t, side="right")))
        return equity

    def book(self, cash: float) -> None:
        self.balance += cash


def trades_summary(trades: pd.DataFrame) -> dict[str, Any]:
    """Counts and USD totals of a TRADES table (after equity_from_trades): n, longs, shorts, wins, gross
    price PnL, commission (positive = paid), swap (negative = paid), net PnL. JSON-serialisable."""
    t = validate_trades(trades, allow_extra=True)
    gross = t["side"] * t["units"] * (t["exit_price"] - t["entry_price"])
    return {
        "n_trades": int(len(t)),
        "n_long": int((t["side"] > 0).sum()),
        "n_short": int((t["side"] < 0).sum()),
        "n_wins": int((t["pnl_usd"] > 0).sum()),
        "gross_usd": float(gross.sum()),
        "commission_usd": float(t["commission_usd"].sum()),
        "swap_usd": float(t["swap_usd"].sum()),
        "pnl_usd": float(t["pnl_usd"].sum()),
    }
