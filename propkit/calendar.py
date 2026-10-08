"""propkit/calendar.py - prop days, DST offsets, swap rollover instants and trading sessions.

No time-zone database is used at run time (zeno's Windows venv may lack tzdata): the daylight-saving
rules are written out below and the tests cross-check them against zoneinfo for every hour of
2015-2026 (and 1990-2014) wherever zoneinfo is available.

Units and conventions used by every function here:
  * an instant `ts` is UTC epoch seconds (int64); arrays of int64 are handled in one vectorised pass,
    a scalar in gives a Python int / bool out, an array in gives a numpy array of the same shape out;
  * a "day" is an int64 count of days since 1970-01-01 (day 0 = 1970-01-01), always a LOCAL calendar
    date of the zone named by the function; `np.asarray(day).astype("datetime64[D]")` turns it into
    numpy dates and `day_to_str` into 'YYYY-MM-DD' text;
  * weekdays are numbered Monday = 0 ... Sunday = 6 (as datetime.date.weekday);
  * offsets are whole hours to ADD to UTC to get local time (CE(S)T +1/+2, New York -5/-4).
Valid instants: 1970-01-01 00:00 UTC <= ts < 2200-01-01 00:00 UTC; anything else (for example
milliseconds, which are 1000 times too large) raises ValueError.

DST rules (the hour of change is the same UTC instant everywhere in the EU):
  * EU, Europe/Prague (CET = UTC+1, CEST = UTC+2): UTC+2 from the last Sunday of March 01:00 UTC to the
    last Sunday of October 01:00 UTC (rule in force since 1996), else UTC+1. 1981-1995: summer time
    ended on the last Sunday of SEPTEMBER 01:00 UTC; this is implemented too. Before 1981 the 1981 rule
    is extrapolated (real history differed; no market data used with propkit is that old).
  * UK (London, for the London session): the same instants as the EU rule, one hour behind CE(S)T
    (UTC+0 / UTC+1). Before 1996 the UK rule differed from the EU one; the EU rule is used anyway.
  * US, America/New_York (EST = UTC-5, EDT = UTC-4): UTC-4 from the second Sunday of March 07:00 UTC
    (02:00 EST) to the first Sunday of November 06:00 UTC (02:00 EDT) (rule since 2007), else UTC-5.
    1987-2006: the first Sunday of April 07:00 UTC to the last Sunday of October 06:00 UTC; this is
    implemented too, and extrapolated to earlier years.
  * Years after the latest rule change use the current rule; recheck if the EU or US abolishes DST.

The prop day (FTMO, CLAUDE.md C6) is the CE(S)T calendar date: it starts at 00:00 CE(S)T, i.e. at
22:00 UTC the evening before in summer and 23:00 UTC in winter. The day of the EU change to summer
time (last Sunday of March) is 23 hours long, the day of the change back (last Sunday of October) is
25 hours long. In the US-only shift weeks (US on summer time, EU still/already on winter time: from
the second Sunday of March to the last Sunday of March, and from the last Sunday of October to the
first Sunday of November) the Sunday metals reopen at 18:00 New York = 22:00 UTC is 23:00 CET, so the
first hour of the week belongs to SUNDAY's prop day, not Monday's; in every other week the reopen
(22:00 UTC in summer, 23:00 UTC in winter) is 00:00 CE(S)T and opens Monday's prop day.
"""
from __future__ import annotations

import datetime as _dt
from functools import lru_cache
from typing import Iterable

import numpy as np

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
MIN_TIME = 0                 # 1970-01-01 00:00 UTC (inclusive)
MAX_TIME = 7258118400        # 2200-01-01 00:00 UTC (exclusive)
SGT_OFFSET_HOURS = 8         # Singapore: UTC+8 all year, no DST
WEEKDAYS_MON_FRI = (0, 1, 2, 3, 4)
WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

_EPOCH = _dt.date(1970, 1, 1)
_EPOCH_WEEKDAY = 3           # 1970-01-01 was a Thursday (Monday = 0)
_TABLE_YEARS = (1969, 2200)  # internal lookups may step an hour or a day outside the valid range

# Session windows in LOCAL time: (zone, start seconds after local midnight, end seconds), [start, end).
SESSIONS: dict[str, tuple[str, int, int]] = {
    "asia": ("Singapore (UTC+8, no DST)", 7 * 3600, 15 * 3600),                 # = 23:00-07:00 UTC
    "london": ("London (UTC+0, UTC+1 in summer, EU rule)", 8 * 3600, 16 * 3600 + 1800),
    "newyork": ("New York (UTC-5, UTC-4 in summer, US rule)", 8 * 3600, 17 * 3600),
}
SESSION_NAMES = ("asia", "london", "newyork", "overlap")


# ---------------------------------------------------------------------------------------
# input checks

def _as_seconds(ts, what: str = "ts") -> tuple[np.ndarray, bool]:
    """Instants as an int64 array plus a 'was scalar' flag; raises ValueError for anything else."""
    if isinstance(ts, (bool, np.bool_)):
        raise ValueError(f"{what} must be UTC epoch seconds (integers), not a boolean")
    arr = np.asarray(ts)
    kind = arr.dtype.kind
    if kind == "M":                                   # numpy datetime64 (naive = UTC)
        if np.isnat(arr).any():
            raise ValueError(f"{what} contains missing times (NaT)")
        secs = arr.astype("datetime64[s]")
        if (secs.astype(arr.dtype) != arr).any():
            raise ValueError(f"{what} contains times that are not whole seconds")
        arr = secs.astype(np.int64)
    elif kind in "iu":
        if arr.size and kind == "u" and arr.max() >= MAX_TIME:
            raise ValueError(_range_message(what, int(arr.max())))
        arr = arr.astype(np.int64)
    elif kind == "f":
        if not np.isfinite(arr).all():
            raise ValueError(f"{what} contains NaN or infinite values; times must be UTC epoch seconds")
        if (arr != np.floor(arr)).any():
            raise ValueError(f"{what} contains fractional seconds; times must be whole UTC epoch seconds")
        if arr.size and (arr.min() < MIN_TIME or arr.max() >= MAX_TIME):
            bad = arr.max() if arr.max() >= MAX_TIME else arr.min()
            raise ValueError(_range_message(what, float(bad)))
        arr = arr.astype(np.int64)
    else:
        raise ValueError(f"{what} must be UTC epoch seconds as integers (or numpy datetime64); "
                         f"got values of type {arr.dtype}")
    if arr.size and (arr.min() < MIN_TIME or arr.max() >= MAX_TIME):
        bad = int(arr.max()) if arr.max() >= MAX_TIME else int(arr.min())
        raise ValueError(_range_message(what, bad))
    return arr, arr.ndim == 0


def _range_message(what: str, bad) -> str:
    hint = (" They look about 1000 times too large: milliseconds? Divide by 1000."
            if bad >= MAX_TIME else "")
    return (f"{what} must be UTC epoch seconds between 1970-01-01 and 2200-01-01 "
            f"(0 <= ts < {MAX_TIME}); got {bad}.{hint}")


def _as_days(day, what: str = "day") -> tuple[np.ndarray, bool]:
    """Days since 1970-01-01 as an int64 array plus a 'was scalar' flag."""
    if isinstance(day, _dt.datetime):
        raise ValueError(f"{what} must be a date (days since 1970-01-01, datetime64[D] or "
                         "datetime.date), not a datetime with a time of day")
    if isinstance(day, _dt.date):
        day = (day - _EPOCH).days
    if isinstance(day, (bool, np.bool_)):
        raise ValueError(f"{what} must be days since 1970-01-01, not a boolean")
    arr = np.asarray(day)
    kind = arr.dtype.kind
    if kind == "M":
        if np.isnat(arr).any():
            raise ValueError(f"{what} contains missing dates (NaT)")
        days = arr.astype("datetime64[D]")
        if (days.astype(arr.dtype) != arr).any():
            raise ValueError(f"{what} contains datetimes that are not whole dates")
        arr = days.astype(np.int64)
    elif kind in "iu":
        arr = arr.astype(np.int64)
    elif kind == "f":
        if not np.isfinite(arr).all() or (arr != np.floor(arr)).any():
            raise ValueError(f"{what} must be whole days since 1970-01-01 (no NaN, no fractions)")
        arr = arr.astype(np.int64)
    else:
        raise ValueError(f"{what} must be days since 1970-01-01 (integers), datetime64[D] or "
                         f"datetime.date; got values of type {arr.dtype}")
    if arr.size and (arr.min() < 0 or arr.max() >= MAX_TIME // SECONDS_PER_DAY):
        raise ValueError(f"{what} must be between 0 (1970-01-01) and {MAX_TIME // SECONDS_PER_DAY - 1} "
                         f"(2199-12-31) days since 1970-01-01")
    return arr, arr.ndim == 0


def _out(arr: np.ndarray, scalar: bool, kind=int):
    return kind(arr) if scalar else arr


def _check_year(year: int) -> int:
    if isinstance(year, (bool, np.bool_)) or not isinstance(year, (int, np.integer)):
        raise ValueError(f"year must be an integer, got {year!r}")
    year = int(year)
    if not _TABLE_YEARS[0] <= year <= _TABLE_YEARS[1]:
        raise ValueError(f"year must be between {_TABLE_YEARS[0]} and {_TABLE_YEARS[1]}, got {year}")
    return year


def _check_hour(hour: int, what: str) -> int:
    if isinstance(hour, (bool, np.bool_)) or not isinstance(hour, (int, np.integer)):
        raise ValueError(f"{what} must be a whole hour 0..23, got {hour!r}")
    hour = int(hour)
    if not 0 <= hour <= 23:
        raise ValueError(f"{what} must be a whole hour 0..23, got {hour}")
    if hour in (1, 2):
        raise ValueError(f"{what} must be 0 or 3..23: 01:00-02:59 New York time is skipped or repeated "
                         f"on the DST change days, so it has no single UTC instant (got {hour})")
    return hour


def _check_weekdays(weekdays: Iterable[int]) -> np.ndarray:
    try:
        given = list(weekdays)
    except TypeError:
        raise ValueError(f"weekdays must be a list of integers 0..6 (Monday = 0), got {weekdays!r}")
    if any(isinstance(w, (bool, np.bool_)) or not isinstance(w, (int, np.integer)) or not 0 <= w <= 6
           for w in given):
        raise ValueError(f"weekdays must be integers 0..6 (Monday = 0), got {given!r}")
    return np.array(sorted({int(w) for w in given}), dtype=np.int64)


# ---------------------------------------------------------------------------------------
# DST rules

def _days(d: _dt.date) -> int:
    return (d - _EPOCH).days


def _last_sunday(year: int, month: int) -> int:
    """Days since 1970-01-01 of the last Sunday of a month."""
    first_next = _dt.date(year + 1, 1, 1) if month == 12 else _dt.date(year, month + 1, 1)
    z = _days(first_next) - 1
    return z - ((z - 3) % 7)            # (z + 3) % 7 == 6 (Sunday)  <=>  z % 7 == 3


def _first_sunday_from(year: int, month: int, day: int) -> int:
    """Days since 1970-01-01 of the first Sunday on or after year-month-day."""
    z = _days(_dt.date(year, month, day))
    return z + ((3 - z) % 7)


@lru_cache(maxsize=None)
def eu_dst_bounds(year: int) -> tuple[int, int]:
    """EU summer time of `year` as UTC epoch seconds [start, end): CEST (UTC+2) inside, CET (UTC+1) outside.

    start = last Sunday of March 01:00 UTC; end = last Sunday of October 01:00 UTC (1996 onward) or of
    September (1981-1995; the 1981 rule is extrapolated to earlier years). The UK changes at the same
    instants (GMT/BST).
    """
    year = _check_year(year)
    end_month = 10 if year >= 1996 else 9
    return (_last_sunday(year, 3) * SECONDS_PER_DAY + SECONDS_PER_HOUR,
            _last_sunday(year, end_month) * SECONDS_PER_DAY + SECONDS_PER_HOUR)


@lru_cache(maxsize=None)
def us_dst_bounds(year: int) -> tuple[int, int]:
    """US daylight time of `year` as UTC epoch seconds [start, end): EDT (UTC-4) inside, EST (UTC-5) outside.

    2007 onward: second Sunday of March 07:00 UTC (02:00 EST) to first Sunday of November 06:00 UTC
    (02:00 EDT). 1987-2006 (extrapolated to earlier years): first Sunday of April 07:00 UTC to last
    Sunday of October 06:00 UTC.
    """
    year = _check_year(year)
    if year >= 2007:
        start, end = _first_sunday_from(year, 3, 8), _first_sunday_from(year, 11, 1)
    else:
        start, end = _first_sunday_from(year, 4, 1), _last_sunday(year, 10)
    return start * SECONDS_PER_DAY + 7 * SECONDS_PER_HOUR, end * SECONDS_PER_DAY + 6 * SECONDS_PER_HOUR


def _years(arr: np.ndarray) -> np.ndarray:
    return arr.astype("datetime64[s]").astype("datetime64[Y]").astype(np.int64) + 1970


def _offsets(arr: np.ndarray, bounds, dst_hours: int, std_hours: int) -> np.ndarray:
    """Offset in hours for each instant of an int64 array (no range check: internal use)."""
    if arr.size == 0:
        return np.zeros(arr.shape, dtype=np.int64)
    years = _years(arr)
    y0, y1 = int(years.min()), int(years.max())
    table = np.array([bounds(y) for y in range(y0, y1 + 1)], dtype=np.int64).reshape(-1, 2)
    idx = years - y0
    inside = (arr >= table[idx, 0]) & (arr < table[idx, 1])
    return np.where(inside, dst_hours, std_hours).astype(np.int64)


def _cet_raw(arr: np.ndarray) -> np.ndarray:
    return _offsets(arr, eu_dst_bounds, 2, 1)


def _ny_raw(arr: np.ndarray) -> np.ndarray:
    return _offsets(arr, us_dst_bounds, -4, -5)


# ---------------------------------------------------------------------------------------
# offsets, prop days, weekdays

def cet_offset_hours(ts):
    """Hours to add to UTC for CE(S)T (Europe/Prague, EU rule): 2 in summer time, else 1.

    ts: UTC epoch seconds, a scalar or an int64 array (vectorised). Returns int or int64 array.
    """
    arr, scalar = _as_seconds(ts)
    return _out(_cet_raw(arr), scalar)


def london_offset_hours(ts):
    """Hours to add to UTC for London (UK rule = EU instants): 1 in summer time (BST), else 0.

    ts: UTC epoch seconds, scalar or int64 array. Returns int or int64 array.
    """
    arr, scalar = _as_seconds(ts)
    return _out(_cet_raw(arr) - 1, scalar)


def ny_offset_hours(ts):
    """Hours to add to UTC for New York (America/New_York, US rule): -4 in daylight time, else -5.

    ts: UTC epoch seconds, scalar or int64 array (vectorised). Returns int or int64 array (negative).
    """
    arr, scalar = _as_seconds(ts)
    return _out(_ny_raw(arr), scalar)


def prop_day(ts):
    """The prop day of each instant: its CE(S)T calendar date, as int64 days since 1970-01-01.

    The day runs [00:00 CE(S)T, next 00:00 CE(S)T): from 22:00 UTC (summer) or 23:00 UTC (winter) the
    evening before. ts: UTC epoch seconds, scalar or int64 array. Returns int or int64 array; use
    day_to_str or .astype("datetime64[D]") for dates.
    """
    arr, scalar = _as_seconds(ts)
    return _out((arr + _cet_raw(arr) * SECONDS_PER_HOUR) // SECONDS_PER_DAY, scalar)


def day_start_utc(day):
    """UTC epoch seconds of 00:00 CE(S)T on the given prop day(s), the start of the prop day.

    day: int days since 1970-01-01 (as prop_day returns), numpy datetime64[D] or datetime.date;
    scalar or array. Returns int or int64 array. The day ends at day_start_utc(day + 1) (23 h on the
    last Sunday of March, 25 h on the last Sunday of October). Local midnight is never inside a DST
    change (changes happen at 01:00 UTC = 02:00/03:00 local), so the instant is always unique.
    """
    arr, scalar = _as_days(day)
    midnight_as_utc = arr * SECONDS_PER_DAY
    offset = _cet_raw(midnight_as_utc - SECONDS_PER_HOUR)   # the offset in force at local midnight
    return _out(midnight_as_utc - offset * SECONDS_PER_HOUR, scalar)


def ny_day(ts):
    """New York local calendar date of each instant, as int64 days since 1970-01-01 (scalar or array)."""
    arr, scalar = _as_seconds(ts)
    return _out((arr + _ny_raw(arr) * SECONDS_PER_HOUR) // SECONDS_PER_DAY, scalar)


def ny_weekday(ts):
    """Weekday (Monday = 0 ... Sunday = 6) of each instant in New York local time.

    This is the weekday a swap rollover belongs to (17:00 New York); ts: UTC epoch seconds, scalar
    or int64 array. Returns int or int64 array.
    """
    arr, scalar = _as_seconds(ts)
    local_day = (arr + _ny_raw(arr) * SECONDS_PER_HOUR) // SECONDS_PER_DAY
    return _out((local_day + _EPOCH_WEEKDAY) % 7, scalar)


def day_weekday(day):
    """Weekday (Monday = 0 ... Sunday = 6) of a day given as days since 1970-01-01 (any zone's date).

    Use it on prop_day(...) output (e.g. Monday-start week blocks). Scalar or array in, int or int64
    array out.
    """
    arr, scalar = _as_days(day)
    return _out((arr + _EPOCH_WEEKDAY) % 7, scalar)


def day_to_str(day):
    """'YYYY-MM-DD' text of a day given as days since 1970-01-01; scalar -> str, array -> array of str."""
    arr, scalar = _as_days(day)
    text = arr.astype("datetime64[D]").astype(str)
    return str(text) if scalar else text


def utc_str(ts):
    """'YYYY-MM-DD HH:MM:SS UTC' text of UTC epoch seconds; scalar -> str, array -> array of str."""
    arr, scalar = _as_seconds(ts)
    raw = arr.astype("datetime64[s]").astype(str)
    if scalar:
        return str(raw).replace("T", " ") + " UTC"
    return np.array([t.replace("T", " ") + " UTC" for t in raw.ravel()], dtype=object).reshape(arr.shape)


# ---------------------------------------------------------------------------------------
# swap rollovers

def rollover_instants(t0, t1, hour_ny: int = 17, weekdays: Iterable[int] = WEEKDAYS_MON_FRI) -> np.ndarray:
    """UTC instants of hour_ny:00 New York local time inside [t0, t1), sorted, as an int64 array.

    t0, t1: UTC epoch seconds (scalars). hour_ny: the broker's rollover hour in New York time (17 =
    the 17:00 New York close used by most FX/metals brokers; 0 or 3..23 allowed). weekdays: New York
    local weekdays that have a rollover (default Monday..Friday; Saturday and Sunday have none, the
    weekend is paid by the triple swap, see CostModel.triple_swap_weekday). The UTC hour moves with
    the US DST rule only: 17:00 New York = 21:00 UTC in US summer time, 22:00 UTC in winter. Use
    ny_weekday(instant) for the weekday of each returned rollover.
    """
    a, a_scalar = _as_seconds(t0, "t0")
    b, b_scalar = _as_seconds(t1, "t1")
    if not (a_scalar and b_scalar):
        raise ValueError("t0 and t1 must be single instants (UTC epoch seconds)")
    hour = _check_hour(hour_ny, "hour_ny")
    wd = _check_weekdays(weekdays)
    t0, t1 = int(a), int(b)
    if t1 <= t0:
        return np.zeros(0, dtype=np.int64)
    first = (t0 - 5 * SECONDS_PER_HOUR) // SECONDS_PER_DAY - 1
    last = (t1 - 4 * SECONDS_PER_HOUR) // SECONDS_PER_DAY + 1
    days = np.arange(first, last + 1, dtype=np.int64)
    local = days * SECONDS_PER_DAY + hour * SECONDS_PER_HOUR      # local wall clock read as if UTC
    utc = local - _ny_raw(local + 5 * SECONDS_PER_HOUR) * SECONDS_PER_HOUR
    keep = np.isin((days + _EPOCH_WEEKDAY) % 7, wd) & (utc >= t0) & (utc < t1)
    return utc[keep]


# ---------------------------------------------------------------------------------------
# sessions

def session_mask(times, name: str):
    """True where an instant lies inside a trading session, [start, end) in the session's local time.

    name (case-insensitive):
      'asia'    07:00-15:00 Singapore time (UTC+8 fixed) = 23:00-07:00 UTC;
      'london'  08:00-16:30 London time (UTC+0, UTC+1 in summer; EU change instants);
      'newyork' 08:00-17:00 New York time (UTC-5, UTC-4 in summer; US rule);
      'overlap' inside both 'london' and 'newyork'.
    Only the time of day is tested: weekends and holidays are not removed (there are no bars then).
    times: UTC epoch seconds, scalar or int64 array (typically the entry fill instants). Returns bool
    or a bool array of the same shape.
    """
    key = str(name).strip().lower()
    if key not in SESSION_NAMES:
        raise ValueError(f"unknown session {name!r}; use one of {', '.join(SESSION_NAMES)}")
    arr, scalar = _as_seconds(times, "times")
    if key == "overlap":
        mask = _session_raw(arr, "london") & _session_raw(arr, "newyork")
    else:
        mask = _session_raw(arr, key)
    return bool(mask) if scalar else mask


def _session_raw(arr: np.ndarray, key: str) -> np.ndarray:
    _, start, end = SESSIONS[key]
    if key == "asia":
        offset = SGT_OFFSET_HOURS
    elif key == "london":
        offset = _cet_raw(arr) - 1
    else:
        offset = _ny_raw(arr)
    tod = (arr + offset * SECONDS_PER_HOUR) % SECONDS_PER_DAY
    return (tod >= start) & (tod < end)
