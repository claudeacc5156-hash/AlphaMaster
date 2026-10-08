"""propkit/bars.py - load, validate and describe BARS; locked-path guard; synthetic bars for tests.

BARS (pandas DataFrame), one row per bar, sorted by time, times unique:
  time        int64, UTC epoch seconds of the bar OPEN;
  open, high, low, close  float64 > 0, BID prices in USD per oz;
  spread      float64 >= 0, USD per oz (ask - bid), optional (when absent, CostModel.fixed_spread is used);
  tick_volume float64, optional, carried through unchanged and not used by propkit.
The ask at any point inside a bar is bid + the bar's spread: one spread per bar is applied to every fill
and every ask-side mark inside it (an approximation; the real spread moves within the bar).
The bar size (bar_seconds) is the most common positive step of `time` (3600 for H1, 1800 for M30,
900 for M15); gaps (weekends, holidays, the daily break) are normal and kept.

Paths containing 'locked_holdout' (any case, either slash) or ending in '.locked' are the locked
holdout and are refused before anything is opened (LockedPathError, a ValueError).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from propkit import calendar

BARS_COLUMNS = ("time", "open", "high", "low", "close")
OPTIONAL_COLUMNS = ("spread", "tick_volume")
PRICE_COLUMNS = ("open", "high", "low", "close")
PARQUET_SUFFIXES = (".parquet", ".pq")
CSV_SUFFIXES = (".csv", ".txt")
MAX_SPREAD_SHARE = 0.005   # a median spread above 0.5% of the median close is taken to be broker points
MAX_SPREAD_USD = 2.0       # load_bars: a median spread above 2 USD/oz (XAUUSD is about 0.1-0.7) is taken to be
                           # points unless spread_scale is given explicitly
MAX_INTRADAY_STEP = 4 * 3600  # infer_bar_seconds: a longer most-common step must be whole days seen twice
_ISO_WITH_OFFSET = r"\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:Z|[+-]\d{2}:?\d{2})$"  # time + Z/offset


class LockedPathError(ValueError):
    """Raised for the locked holdout: a path containing 'locked_holdout' or ending in '.locked'."""


# ---------------------------------------------------------------------------------------
# locked holdout guard

def is_locked_path(path_text) -> bool:
    """True when a path is the locked holdout: it contains 'locked_holdout' (any case, '/' or '\\\\')
    or ends in '.locked'. Pure text test; nothing is opened."""
    low = str(path_text).strip().replace("\\", "/").rstrip("/").lower()
    return "locked_holdout" in low or low.endswith(".locked")


def check_not_locked(path_text, what: str = "file", verb: str = "read") -> None:
    """Raise LockedPathError if the given path text, or the path it resolves to, is the locked holdout.

    Call it before opening or writing anything. The message tells a first-time user what happened.
    """
    candidates = [str(path_text)]
    try:
        candidates.append(str(Path(str(path_text)).expanduser().resolve()))
    except (OSError, RuntimeError, ValueError):
        pass
    if any(is_locked_path(c) for c in candidates):
        raise LockedPathError(
            f"refusing to {verb} the {what} {path_text}: paths containing 'locked_holdout' or ending in "
            "'.locked' are the locked holdout. The holdout test is a separate pre-registered step and "
            "propkit never touches it.")


# ---------------------------------------------------------------------------------------
# time conversion

def to_epoch_seconds(values, what: str = "time") -> np.ndarray:
    """Convert a time column to int64 UTC epoch seconds.

    Accepted: integer seconds; integer milliseconds / microseconds / nanoseconds (values above 1e11,
    1e14, 1e17 are divided by 1e3, 1e6, 1e9 as AlphaMaster does; they must divide exactly); whole-second
    floats; datetime64 columns (naive ones are taken as UTC, as AlphaMaster does; tz-aware ones are
    converted); ISO text WITH an explicit offset ('2024-01-02T03:00:00Z', '...+00:00'). Text without an
    offset is refused (MT5 exports use the broker's server time, not UTC). Raises ValueError otherwise.
    """
    s = values if isinstance(values, pd.Series) else pd.Series(values)
    if len(s) == 0:
        return np.zeros(0, dtype=np.int64)
    if s.isna().any():
        raise ValueError(f"the {what} column has missing values (row {int(np.flatnonzero(s.isna())[0])})")
    if pd.api.types.is_bool_dtype(s):
        raise ValueError(f"the {what} column is boolean; it must hold UTC epoch seconds or datetimes")
    if pd.api.types.is_datetime64_any_dtype(s):
        return _datetimes_to_seconds(pd.to_datetime(s, utc=True), what)
    if pd.api.types.is_numeric_dtype(s):
        if pd.api.types.is_float_dtype(s):
            arr = s.to_numpy(dtype=np.float64)
            if not np.isfinite(arr).all() or (arr != np.floor(arr)).any():
                raise ValueError(f"the {what} column has fractional or infinite values; it must hold whole "
                                 "UTC epoch seconds")
            if np.abs(arr).max() >= 2.0 ** 62:
                raise ValueError(f"the {what} column has values too large to be epoch times")
            arr = arr.astype(np.int64)
        elif pd.api.types.is_integer_dtype(s):
            if pd.api.types.is_unsigned_integer_dtype(s) and int(s.max()) >= 2 ** 62:
                raise ValueError(f"the {what} column has values too large to be epoch times")
            arr = s.to_numpy(dtype=np.int64)
        else:
            raise ValueError(f"the {what} column must hold UTC epoch seconds (integers), got {s.dtype}")
        mx = int(arr.max())
        if mx > 1e11:
            div = 1_000_000_000 if mx > 1e17 else 1_000_000 if mx > 1e14 else 1000
            if (arr % div != 0).any():
                raise ValueError(f"the {what} column looks like epoch {_unit_name(div)} but some values are "
                                 "not whole seconds; bar open times must be whole seconds")
            arr = arr // div
        return arr
    text = s.astype(str).str.strip()
    bad = ~text.str.contains(_ISO_WITH_OFFSET, case=False, regex=True).to_numpy(dtype=bool)
    if bad.any():
        row = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"the {what} column holds text such as {text.iloc[row]!r} without a UTC offset. Text times are "
            "ambiguous (MT5 exports use the broker's server time); write UTC epoch seconds, or ISO times "
            "ending in 'Z' or '+00:00'.")
    try:
        parsed = pd.to_datetime(text, utc=True, format="ISO8601")
    except (ValueError, TypeError) as e:
        raise ValueError(f"cannot read the {what} column as ISO times: {e}")
    return _datetimes_to_seconds(parsed, what)


def _unit_name(div: int) -> str:
    return {1000: "milliseconds", 1_000_000: "microseconds", 1_000_000_000: "nanoseconds"}[div]


def _datetimes_to_seconds(ts: pd.Series, what: str) -> np.ndarray:
    if ts.isna().any():
        raise ValueError(f"the {what} column has missing times")
    naive = ts.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    secs = naive.astype("datetime64[s]")
    if (secs.astype(naive.dtype) != naive).any():
        raise ValueError(f"the {what} column has times that are not whole seconds; bar opens must be")
    return secs.astype(np.int64)


# ---------------------------------------------------------------------------------------
# validation

def validate_bars(df: pd.DataFrame, spread_scale: float = 1.0, source: str = "bars") -> pd.DataFrame:
    """Check and normalise a bar table; returns a new BARS DataFrame (see the module docstring).

    Checks (each failure raises ValueError naming the first bad row and its UTC time): the columns
    time/open/high/low/close exist; at least 2 bars; times convert to UTC epoch seconds (see
    to_epoch_seconds) between 1970 and 2200; times strictly increasing (sorted, unique: the file is not
    re-sorted silently); prices finite and > 0; high >= max(open, close) and low <= min(open, close);
    spread (if present, after x spread_scale) finite, >= 0 and plausibly USD per oz (a median above 0.5%
    of the median close looks like broker points: pass spread_scale=0.01 for 2-digit quotes or 0.001
    for 3-digit ones, or drop the column). source: a name used in messages.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{source}: expected a pandas DataFrame of bars")
    missing = [c for c in BARS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing column(s) {missing}; expected time, open, high, low, close "
                         "(optional: spread in USD per oz, tick_volume)")
    if len(df) < 2:
        raise ValueError(f"{source} has {len(df)} bar(s); at least 2 are needed")
    scale = _positive_number(spread_scale, "spread_scale")
    times = to_epoch_seconds(df["time"], what=f"{source} time")
    if times.min() < calendar.MIN_TIME or times.max() >= calendar.MAX_TIME:
        raise ValueError(f"{source}: times must be UTC epoch seconds between 1970 and 2200; got "
                         f"{int(times.min())}..{int(times.max())}")
    step = np.diff(times)
    if (step <= 0).any():
        i = int(np.flatnonzero(step <= 0)[0]) + 1
        kind = "a duplicate time" if step[i - 1] == 0 else "a time earlier than the row before"
        raise ValueError(f"{source}: row {i} ({calendar.utc_str(int(times[i]))}) has {kind}; bars must be "
                         "sorted by time with no duplicates. Sort the file and drop duplicate rows first.")
    out = pd.DataFrame({"time": times})
    for col in PRICE_COLUMNS:
        out[col] = _float_column(df[col], f"{source} {col}")
    prices = out[list(PRICE_COLUMNS)].to_numpy()
    bad = ~np.isfinite(prices).all(axis=1) | (prices <= 0).any(axis=1)
    _raise_first(bad, times, source, "has a missing, infinite or non-positive price")
    o, h, lo, c = (out[col].to_numpy() for col in PRICE_COLUMNS)
    _raise_first(h < np.maximum(o, c), times, source, "has high below open or close")
    _raise_first(lo > np.minimum(o, c), times, source, "has low above open or close")
    if "spread" in df.columns:
        spread = _float_column(df["spread"], f"{source} spread") * scale
        _raise_first(~np.isfinite(spread), times, source, "has a missing or infinite spread (fill it, or drop "
                     "the spread column to use the fixed spread)")
        _raise_first(spread < 0, times, source, "has a negative spread (spread = ask - bid must be >= 0)")
        med_spread, med_close = float(np.median(spread)), float(np.median(c))
        if med_spread > MAX_SPREAD_SHARE * med_close:
            raise ValueError(
                f"{source}: the spread column looks like broker points, not USD per oz (median spread "
                f"{med_spread:g} vs median close {med_close:g}). Pass spread_scale=0.01 for 2-digit quotes "
                "(0.001 for 3-digit) - on the command line --spread-scale 0.01 - or drop the column to use "
                "the fixed spread.")
        out["spread"] = spread
    if "tick_volume" in df.columns:
        out["tick_volume"] = _float_column(df["tick_volume"], f"{source} tick_volume", strict=False)
    return out


def _positive_number(value, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{what} must be a number > 0, got {value!r}")
    x = float(value)
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"{what} must be a finite number > 0, got {value!r}")
    return x


def _float_column(col: pd.Series, what: str, strict: bool = True) -> np.ndarray:
    if pd.api.types.is_bool_dtype(col) or not pd.api.types.is_numeric_dtype(col):
        if not strict:
            return pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)
        raise ValueError(f"{what}: the column must be numeric, got {col.dtype}")
    return col.to_numpy(dtype=np.float64, na_value=np.nan)


def _raise_first(bad: np.ndarray, times: np.ndarray, source: str, problem: str) -> None:
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(f"{source}: row {i} ({calendar.utc_str(int(times[i]))}) {problem}; "
                         f"{int(bad.sum())} row(s) affected. Clean the file first.")


# ---------------------------------------------------------------------------------------
# loading

def load_bars(path, spread_scale: float | None = None) -> pd.DataFrame:
    """Read a bar file and return validated BARS (see validate_bars for every check).

    path: AlphaMaster-style Parquet (.parquet/.pq) or CSV (.csv/.txt) with columns time, open, high,
    low, close[, tick_volume][, spread]; time as UTC epoch seconds (ms/us/ns accepted) or datetime64
    (naive = UTC); prices are BID in USD per oz; spread in USD per oz (spread_scale converts broker
    points: 0.01 for 2-digit quotes). spread_scale None (default) means 1.0 plus one more check: a median
    spread above MAX_SPREAD_USD (2 USD/oz; real XAUUSD spreads are about 0.1-0.7) is refused as probable
    broker points (an MT5 export of a raw/ECN account shows 5-9 points = 0.05-0.09 USD on 2-digit quotes,
    which read as USD/oz would overstate costs about 100 times); pass spread_scale explicitly (1.0 when
    the values really are USD/oz) to skip that check. Locked-holdout paths raise LockedPathError before
    anything is opened; a missing or unreadable file raises ValueError.
    """
    check_not_locked(path, what="bar file")
    p = Path(str(path)).expanduser()
    if not p.is_file():
        raise ValueError(f"bar file not found: {p}")
    suffix = p.suffix.lower()
    if suffix not in PARQUET_SUFFIXES + CSV_SUFFIXES:
        raise ValueError(f"bar file {p.name}: use a .parquet or .csv file (got '{suffix or 'no suffix'}')")
    try:
        df = pd.read_parquet(p) if suffix in PARQUET_SUFFIXES else pd.read_csv(p)
    except Exception as e:   # pyarrow and the CSV parser raise several types
        raise ValueError(f"cannot read {p.name}: {type(e).__name__}: {e}")
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"cannot read {p.name} as a table")
    df.columns = [str(c).strip().lower() for c in df.columns]
    out = validate_bars(df, spread_scale=1.0 if spread_scale is None else spread_scale, source=p.name)
    if spread_scale is None and "spread" in out.columns:
        med = float(np.median(out["spread"].to_numpy()))
        if med > MAX_SPREAD_USD:
            raise ValueError(
                f"{p.name}: the spread column's median is {med:g}, which as USD per oz would be far above a "
                f"real XAUUSD spread (about 0.1-0.7; refused above {MAX_SPREAD_USD:g}). It looks like broker "
                "points: for 2-digit quotes pass --spread-scale 0.01 (Python: spread_scale=0.01), for 3-digit "
                "quotes 0.001. If the values really are USD per oz, pass --spread-scale 1 to accept them.")
    return out


# ---------------------------------------------------------------------------------------
# helpers

def infer_bar_seconds(times) -> int:
    """Bar size in seconds: the most common positive difference of consecutive bar times (ties: the
    smaller one). times: sorted UTC epoch seconds (array or BARS['time']). Raises ValueError if there
    is no positive difference, or if the step cannot be a bar size (bar_size_problem): longer than 4 hours
    and not a whole number of days seen at least twice. That happens when the data has so few bars that a
    weekend or another gap was taken for the bar size: with only two bars, Friday 20:00 and Sunday 22:00
    UTC, the "bar" would be 50 hours long and a Saturday fill would look like a fill inside it."""
    t = np.asarray(times, dtype=np.int64)
    d = np.diff(t)
    d = d[d > 0]
    if d.size == 0:
        raise ValueError("cannot infer the bar size: need at least two bars with increasing times")
    values, counts = np.unique(d, return_counts=True)
    k = int(np.argmax(counts))
    step = int(values[k])
    problem = bar_size_problem(step, int(counts[k]), int(t.size))
    if problem:
        raise ValueError(problem)
    return step


def bar_size_problem(step: int, count: int, n_bars: int) -> str | None:
    """Why an inferred bar size cannot be trusted (a plain-English message), or None when it can.

    step: the most common step between bar opens (seconds), seen `count` times among n_bars bars. Up to
    MAX_INTRADAY_STEP (4 hours) any step is accepted; a longer one only when it is a whole number of days
    (D1 bars) seen at least twice. Anything else is a gap (a weekend, a holiday) taken for the bar size."""
    if step <= MAX_INTRADAY_STEP or (step % calendar.SECONDS_PER_DAY == 0 and count >= 2):
        return None
    return (f"cannot infer the bar size: the most common step between the {n_bars} bar times is {step} s "
            f"({step / 3600:.1f} hours), which is not a bar size but a gap such as a weekend. propkit is "
            "built for M15, M30 or H1 bars; pass at least a few hours of consecutive bars")


def bar_index_at(times, t, bar_seconds: int | None = None):
    """Index of the bar an instant falls in: the last bar whose open time <= t (-1 if t is before the
    first bar). With bar_seconds, an instant at or after that bar's end (t >= open + bar_seconds, i.e.
    in a gap such as a weekend or the daily break) gives -1 too.

    times: sorted UTC epoch seconds of bar opens; t: UTC epoch seconds, scalar or array. Returns int or
    int64 array.
    """
    tt = np.asarray(times, dtype=np.int64)
    q = np.asarray(t)
    if q.dtype.kind not in "iu":
        raise ValueError("t must be UTC epoch seconds as integers")
    q = q.astype(np.int64)
    idx = np.searchsorted(tt, q, side="right") - 1
    if bar_seconds is not None:
        if int(bar_seconds) <= 0:
            raise ValueError("bar_seconds must be > 0")
        inside = (idx >= 0) & (q < tt[np.maximum(idx, 0)] + int(bar_seconds))
        idx = np.where(inside, idx, -1)
    return int(idx) if q.ndim == 0 else idx.astype(np.int64)


def rollover_bids(times, open_, close, instants) -> np.ndarray:
    """Bid price (USD/oz) on which the swap of each rollover instant is charged, known at that instant.

    times: sorted bar opens (UTC epoch seconds); open_ / close: the bars' bid open and close; instants:
    rollover instants (UTC epoch seconds, at or after the first bar open). A bar that opens exactly at
    the instant (24-hour bars, where 17:00 New York is a bar open) gives its OPEN, the bid at that
    instant; otherwise the CLOSE of the last bar with open <= instant (metals bars: the bar that ends at
    17:00 New York when the daily break starts). Using the close of the bar opening at the instant would
    let a later price set a cash amount booked at its open (an entry at that open could be sized on it).
    Returns a float64 array.
    """
    tt = np.asarray(times, dtype=np.int64)
    r = np.asarray(instants, dtype=np.int64)
    if r.size == 0:
        return np.zeros(0, dtype=np.float64)
    j = np.maximum(np.searchsorted(tt, r, side="right") - 1, 0)
    o = np.asarray(open_, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    return np.where(tt[j] == r, o[j], c[j]).astype(np.float64)


def bars_summary(bars: pd.DataFrame) -> dict[str, Any]:
    """A small JSON-serialisable description of BARS: n_bars, first/last bar open (epoch s and UTC text),
    bar_seconds, number of gaps (steps longer than one bar), and spread median / p90 in USD per oz (None
    when there is no spread column)."""
    t = bars["time"].to_numpy(dtype=np.int64)
    bar_seconds = infer_bar_seconds(t)
    has_spread = "spread" in bars.columns
    spread = bars["spread"].to_numpy(dtype=np.float64) if has_spread else None
    return {
        "n_bars": int(len(t)),
        "first_time": int(t[0]), "last_time": int(t[-1]),
        "first_time_utc": calendar.utc_str(int(t[0])), "last_time_utc": calendar.utc_str(int(t[-1])),
        "bar_seconds": bar_seconds,
        "n_gaps": int((np.diff(t) > bar_seconds).sum()),
        "spread_median": float(np.median(spread)) if has_spread else None,
        "spread_p90": float(np.quantile(spread, 0.9)) if has_spread else None,
    }


def metals_market_open(times) -> np.ndarray:
    """True where a bar opening at that instant is inside the usual XAUUSD CFD trading week: Sunday
    18:00 to Friday 17:00 New York time, with a daily break 17:00-18:00 New York (Mon-Thu). New York
    time follows the US DST rule. Holidays are not modelled. times: UTC epoch seconds (array)."""
    arr, _ = calendar._as_seconds(times, "times")
    local = arr + calendar._ny_raw(arr) * calendar.SECONDS_PER_HOUR
    wd = (local // calendar.SECONDS_PER_DAY + 3) % 7
    tod = local % calendar.SECONDS_PER_DAY
    h17, h18 = 17 * 3600, 18 * 3600
    closed = ((wd == 5) | ((wd == 4) & (tod >= h17)) | ((wd == 6) & (tod < h18))
              | ((wd <= 3) & (tod >= h17) & (tod < h18)))
    return ~closed


def synthetic_bars(start: int, n_bars: int, bar_seconds: int = 3600, seed: int = 0, price: float = 2000.0,
                   vol_per_hour: float = 0.002, spread: float | None = 0.34,
                   market_hours: str = "metals") -> pd.DataFrame:
    """Deterministic synthetic BARS for tests and the self-test (NOT market data).

    start: UTC epoch seconds of the first candidate bar (a multiple of bar_seconds); n_bars bars are
    kept. market_hours "metals" keeps only bars inside metals_market_open (weekend gap, daily break,
    DST-correct reopen at 22:00/23:00 UTC); "always" keeps every step. Prices: a log random walk with
    sd vol_per_hour x sqrt(bar hours) per bar (numpy Generator(seed)), small opening gaps, wicks
    beyond open/close. spread: base USD per oz, varied per bar by a lognormal factor and rounded to
    0.01 (None = no spread column). Includes tick_volume.
    """
    if int(bar_seconds) <= 0 or start % int(bar_seconds) != 0:
        raise ValueError("bar_seconds must be > 0 and start a multiple of it")
    if int(n_bars) < 2:
        raise ValueError("n_bars must be >= 2")
    if market_hours not in ("metals", "always"):
        raise ValueError("market_hours must be 'metals' or 'always'")
    n, bs = int(n_bars), int(bar_seconds)
    times = np.zeros(0, dtype=np.int64)
    first = int(start)
    while times.size < n:
        cand = first + bs * np.arange(int(n * 1.6) + 400, dtype=np.int64)
        keep = cand if market_hours == "always" else cand[metals_market_open(cand)]
        times = np.concatenate([times, keep])
        first = int(cand[-1]) + bs
    times = times[:n]
    rng = np.random.default_rng(seed)
    sd = vol_per_hour * math.sqrt(bs / 3600.0)
    body = rng.normal(0.0, sd, n)
    gap = rng.normal(0.0, sd * 0.1, n)
    gap[1:][np.diff(times) > bs] = rng.normal(0.0, sd * 3.0, int((np.diff(times) > bs).sum()))
    log_open = math.log(price) + np.cumsum(gap + np.r_[0.0, body[:-1]])
    o = np.exp(log_open)
    c = o * np.exp(body)
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0.0, sd * 0.5, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0.0, sd * 0.5, n)))
    df = pd.DataFrame({"time": times, "open": o, "high": h, "low": lo, "close": c,
                       "tick_volume": rng.integers(100, 5000, n).astype(np.float64)})
    if spread is not None:
        sp = np.round(float(spread) * np.exp(rng.normal(0.0, 0.25, n)), 2)
        df.insert(5, "spread", np.maximum(sp, 0.0))
    return validate_bars(df, source="synthetic bars")
