"""Tests for propkit/calendar.py: explicit DST rules, prop days, rollovers, sessions. Research only.

The DST rules are cross-checked against zoneinfo (Europe/Prague, America/New_York, Europe/London) for
every hour of 2015-01-01..2026-12-31 when zoneinfo and its tz data are available (skipped otherwise).
"""
from __future__ import annotations

import datetime as dt
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from propkit import calendar as cal

ROOT = Path(__file__).resolve().parents[2]
UTC = dt.timezone.utc
H = 3600
DAY = 86400
T_2015 = 1420070400          # 2015-01-01 00:00 UTC
T_2027 = 1798761600          # 2027-01-01 00:00 UTC


def ts(text: str) -> int:
    """'YYYY-MM-DD HH:MM[:SS]' read as UTC -> epoch seconds."""
    fmt = "%Y-%m-%d %H:%M:%S" if text.count(":") == 2 else "%Y-%m-%d %H:%M"
    return int(dt.datetime.strptime(text, fmt).replace(tzinfo=UTC).timestamp())


def dnum(text: str) -> int:
    """'YYYY-MM-DD' -> days since 1970-01-01."""
    return (dt.date.fromisoformat(text) - dt.date(1970, 1, 1)).days


def _zone(name: str):
    zoneinfo = pytest.importorskip("zoneinfo")
    try:
        return zoneinfo.ZoneInfo(name)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError, OSError):
        pytest.skip(f"tz data for {name} is not installed")


def _truth(zone, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(offset hours, local day number) from zoneinfo for each instant."""
    epoch = dt.date(1970, 1, 1)
    off = np.empty(times.size, dtype=np.int64)
    day = np.empty(times.size, dtype=np.int64)
    for i, t in enumerate(times.tolist()):
        local = dt.datetime.fromtimestamp(t, tz=zone)
        off[i] = int(local.utcoffset().total_seconds()) // 3600
        day[i] = (local.date() - epoch).days
    return off, day


@pytest.fixture(scope="module")
def hours_2015_2026() -> np.ndarray:
    return np.arange(T_2015, T_2027, H, dtype=np.int64)


# ---------------------------------------------------------------------------------------
# cross-checks against zoneinfo

def test_cet_offset_and_prop_day_match_zoneinfo_every_hour_2015_2026(hours_2015_2026):
    off, day = _truth(_zone("Europe/Prague"), hours_2015_2026)
    assert np.array_equal(cal.cet_offset_hours(hours_2015_2026), off)
    assert np.array_equal(cal.prop_day(hours_2015_2026), day)


def test_ny_offset_and_weekday_match_zoneinfo_every_hour_2015_2026(hours_2015_2026):
    off, day = _truth(_zone("America/New_York"), hours_2015_2026)
    assert np.array_equal(cal.ny_offset_hours(hours_2015_2026), off)
    assert np.array_equal(cal.ny_day(hours_2015_2026), day)
    assert np.array_equal(cal.ny_weekday(hours_2015_2026), (day + 3) % 7)


def test_london_offset_matches_zoneinfo_every_hour_2015_2026(hours_2015_2026):
    off, _ = _truth(_zone("Europe/London"), hours_2015_2026)
    assert np.array_equal(cal.london_offset_hours(hours_2015_2026), off)


def test_historical_rules_match_zoneinfo_1990_2014():
    """EU end in September until 1995, US April..October until 2006 (both implemented explicitly)."""
    times = np.arange(ts("1990-01-01 00:00"), T_2015, H, dtype=np.int64)
    off, day = _truth(_zone("Europe/Prague"), times)
    assert np.array_equal(cal.cet_offset_hours(times), off)
    assert np.array_equal(cal.prop_day(times), day)
    off, _ = _truth(_zone("America/New_York"), times)
    assert np.array_equal(cal.ny_offset_hours(times), off)


def test_dst_change_seconds_match_zoneinfo_2015_2026():
    """One second before / at each change instant (the hourly grid hits them too; this pins the second)."""
    prague, ny = _zone("Europe/Prague"), _zone("America/New_York")
    for year in range(2015, 2027):
        for bounds, zone, fn in ((cal.eu_dst_bounds, prague, cal.cet_offset_hours),
                                 (cal.us_dst_bounds, ny, cal.ny_offset_hours)):
            for edge in bounds(year):
                for t in (edge - 1, edge):
                    want = int(dt.datetime.fromtimestamp(t, tz=zone).utcoffset().total_seconds()) // 3600
                    assert fn(t) == want, (year, t)


def test_rollover_instants_match_zoneinfo_2015_2026():
    ny = _zone("America/New_York")
    want = []
    d = dt.date(2015, 1, 1)
    while d.year < 2027:
        if d.weekday() < 5:
            want.append(int(dt.datetime(d.year, d.month, d.day, 17, tzinfo=ny).timestamp()))
        d += dt.timedelta(days=1)
    got = cal.rollover_instants(T_2015, T_2027)
    assert got.dtype == np.int64
    assert np.array_equal(got, np.array(want, dtype=np.int64))
    want8 = [int(dt.datetime(2024, 7, x, 8, tzinfo=ny).timestamp()) for x in (15, 16, 17, 18, 19)]
    assert cal.rollover_instants(ts("2024-07-14 00:00"), ts("2024-07-21 00:00"), hour_ny=8).tolist() == want8


def test_sessions_match_zoneinfo_every_30_minutes_2015_2026():
    times = np.arange(T_2015, T_2027, 1800, dtype=np.int64)
    for name, zone, start, end in (("london", "Europe/London", 8 * H, 16 * H + 1800),
                                   ("newyork", "America/New_York", 8 * H, 17 * H)):
        z = _zone(zone)
        tod = np.array([(lambda x: x.hour * H + x.minute * 60)(dt.datetime.fromtimestamp(t, tz=z))
                        for t in times.tolist()])
        assert np.array_equal(cal.session_mask(times, name), (tod >= start) & (tod < end)), name


# ---------------------------------------------------------------------------------------
# known answers (hand-checked dates)

def test_dst_bounds_known_answers():
    assert cal.eu_dst_bounds(2024) == (ts("2024-03-31 01:00"), ts("2024-10-27 01:00"))
    assert cal.eu_dst_bounds(2025) == (ts("2025-03-30 01:00"), ts("2025-10-26 01:00"))
    assert cal.us_dst_bounds(2024) == (ts("2024-03-10 07:00"), ts("2024-11-03 06:00"))
    assert cal.us_dst_bounds(2025) == (ts("2025-03-09 07:00"), ts("2025-11-02 06:00"))
    assert cal.eu_dst_bounds(1995)[1] == ts("1995-09-24 01:00")          # pre-1996: end of September
    assert cal.us_dst_bounds(2006) == (ts("2006-04-02 07:00"), ts("2006-10-29 06:00"))  # pre-2007 rule
    for bad in (1969.5, "2024", True, 2300):
        with pytest.raises(ValueError):
            cal.eu_dst_bounds(bad)


def test_offsets_at_change_seconds():
    s, e = cal.eu_dst_bounds(2024)
    assert [cal.cet_offset_hours(x) for x in (s - 1, s, e - 1, e)] == [1, 2, 2, 1]
    assert [cal.london_offset_hours(x) for x in (s - 1, s, e - 1, e)] == [0, 1, 1, 0]
    s, e = cal.us_dst_bounds(2025)
    assert [cal.ny_offset_hours(x) for x in (s - 1, s, e - 1, e)] == [-5, -4, -4, -5]


def test_prop_day_resets_at_22_utc_in_summer_and_23_utc_in_winter():
    assert cal.day_start_utc(dnum("2024-07-15")) == ts("2024-07-14 22:00")
    assert cal.day_start_utc(dnum("2024-01-15")) == ts("2024-01-14 23:00")
    assert cal.prop_day(ts("2024-07-14 21:59:59")) == dnum("2024-07-14")
    assert cal.prop_day(ts("2024-07-14 22:00")) == dnum("2024-07-15")
    assert cal.prop_day(ts("2024-01-14 22:59:59")) == dnum("2024-01-14")
    assert cal.prop_day(ts("2024-01-14 23:00")) == dnum("2024-01-15")


def test_eu_change_days_are_25h_and_23h():
    d = dnum("2024-10-27")                       # back to CET: 25 hours
    assert cal.day_start_utc(d) == ts("2024-10-26 22:00")
    assert cal.day_start_utc(d + 1) == ts("2024-10-27 23:00")
    assert cal.day_start_utc(d + 1) - cal.day_start_utc(d) == 25 * H
    d = dnum("2025-03-30")                       # to CEST: 23 hours
    assert cal.day_start_utc(d) == ts("2025-03-29 23:00")
    assert cal.day_start_utc(d + 1) == ts("2025-03-30 22:00")
    assert cal.day_start_utc(d + 1) - cal.day_start_utc(d) == 23 * H
    hours = np.arange(ts("2024-10-26 22:00"), ts("2024-10-27 23:00"), H)
    assert (cal.prop_day(hours) == dnum("2024-10-27")).all() and hours.size == 25
    hours = np.arange(ts("2025-03-29 23:00"), ts("2025-03-30 22:00"), H)
    assert (cal.prop_day(hours) == dnum("2025-03-30")).all() and hours.size == 23


def test_day_start_round_trip_every_day_2015_2026():
    days = np.arange(dnum("2015-01-01"), dnum("2027-01-01"), dtype=np.int64)
    starts = cal.day_start_utc(days)
    assert np.array_equal(cal.prop_day(starts), days)
    assert np.array_equal(cal.prop_day(starts - 1), days - 1)
    lengths = np.diff(starts) // H
    assert set(np.unique(lengths).tolist()) == {23, 24, 25}
    assert (lengths == 23).sum() == 12 and (lengths == 25).sum() == 12      # one of each per year


@pytest.mark.parametrize("reopen, prop_date, why", [
    ("2024-07-14 22:00", "2024-07-15", "summer: 18:00 EDT = 00:00 CEST Monday"),
    ("2024-01-14 23:00", "2024-01-15", "winter: 18:00 EST = 00:00 CET Monday"),
    ("2024-10-27 22:00", "2024-10-27", "US-only shift (EU CET, US EDT): 23:00 CET Sunday"),
    ("2024-11-03 23:00", "2024-11-04", "US back on EST that morning: 00:00 CET Monday"),
    ("2025-03-09 22:00", "2025-03-09", "US-only shift (US EDT from 07:00 UTC, EU CET): Sunday"),
    ("2025-03-16 22:00", "2025-03-16", "US-only shift week: Sunday"),
    ("2025-03-23 22:00", "2025-03-23", "US-only shift week: Sunday"),
    ("2025-03-30 22:00", "2025-03-31", "EU on CEST since 01:00 UTC: 00:00 CEST Monday"),
])
def test_sunday_reopen_prop_day(reopen, prop_date, why):
    """The metals reopen is 18:00 New York (22:00 UTC under EDT, 23:00 UTC under EST)."""
    t = ts(reopen)
    ny_local = t + cal.ny_offset_hours(t) * H
    assert ny_local % DAY == 18 * H and cal.ny_weekday(t) == 6, why
    assert cal.prop_day(t) == dnum(prop_date), why
    assert cal.day_weekday(dnum(prop_date)) == (6 if prop_date == reopen[:10] else 0)


def test_rollover_utc_hour_moves_only_with_us_dst():
    assert cal.rollover_instants(ts("2024-07-17 00:00"), ts("2024-07-18 00:00")).tolist() == [ts("2024-07-17 21:00")]
    assert cal.rollover_instants(ts("2024-01-17 00:00"), ts("2024-01-18 00:00")).tolist() == [ts("2024-01-17 22:00")]
    spring = cal.rollover_instants(ts("2025-03-07 00:00"), ts("2025-03-11 00:00"))
    assert spring.tolist() == [ts("2025-03-07 22:00"), ts("2025-03-10 21:00")]     # Fri EST, Mon EDT
    autumn = cal.rollover_instants(ts("2024-11-01 00:00"), ts("2024-11-05 00:00"))
    assert autumn.tolist() == [ts("2024-11-01 21:00"), ts("2024-11-04 22:00")]     # Fri EDT, Mon EST
    eu_change = cal.rollover_instants(ts("2025-03-28 00:00"), ts("2025-04-01 00:00"))
    assert eu_change.tolist() == [ts("2025-03-28 21:00"), ts("2025-03-31 21:00")]  # EU change: no move


def test_rollover_weekdays_window_and_validation():
    t0, t1 = ts("2024-06-03 00:00"), ts("2024-07-01 00:00")          # four Monday-start weeks
    r = cal.rollover_instants(t0, t1)
    assert r.size == 20 and np.all(np.diff(r) > 0)
    assert cal.ny_weekday(r).tolist() == [0, 1, 2, 3, 4] * 4
    wed = cal.rollover_instants(t0, t1, weekdays=[2])
    assert wed.size == 4 and (cal.ny_weekday(wed) == 2).all()
    every = cal.rollover_instants(t0, t1, weekdays=range(7))
    assert every.size == 28
    first = int(r[0])
    assert cal.rollover_instants(first, first + 1).tolist() == [first]     # [t0, t1): t0 included
    assert cal.rollover_instants(first - 10, first).size == 0               # t1 excluded
    assert cal.rollover_instants(t1, t0).size == 0
    for bad in (1, 2, 24, -1, 17.5, True):
        with pytest.raises(ValueError):
            cal.rollover_instants(t0, t1, hour_ny=bad)
    for bad in ([7], [-1], ["Mon"], [True], 5):
        with pytest.raises(ValueError):
            cal.rollover_instants(t0, t1, weekdays=bad)
    with pytest.raises(ValueError):
        cal.rollover_instants(np.array([t0, t0]), t1)


@pytest.mark.parametrize("name, day, start, end", [
    ("asia", "2024-07-16", "2024-07-15 23:00", "2024-07-16 07:00"),
    ("asia", "2024-01-16", "2024-01-15 23:00", "2024-01-16 07:00"),
    ("london", "2024-07-16", "2024-07-16 07:00", "2024-07-16 15:30"),
    ("london", "2024-01-16", "2024-01-16 08:00", "2024-01-16 16:30"),
    ("newyork", "2024-07-16", "2024-07-16 12:00", "2024-07-16 21:00"),
    ("newyork", "2024-01-16", "2024-01-16 13:00", "2024-01-16 22:00"),
    ("overlap", "2024-07-16", "2024-07-16 12:00", "2024-07-16 15:30"),
    ("overlap", "2024-01-16", "2024-01-16 13:00", "2024-01-16 16:30"),
    ("overlap", "2024-10-29", "2024-10-29 12:00", "2024-10-29 16:30"),   # London GMT, New York EDT
    ("overlap", "2025-03-11", "2025-03-11 12:00", "2025-03-11 16:30"),   # London GMT, New York EDT
])
def test_session_windows_known_answers(name, day, start, end):
    grid = np.arange(ts(day + " 00:00") - DAY, ts(day + " 00:00") + DAY, 60, dtype=np.int64)
    mask = cal.session_mask(grid, name)
    inside = grid[mask]
    a, b = ts(start), ts(end)
    window = inside[(inside >= a - 12 * H) & (inside < b + 12 * H)]
    assert window[0] == a and window[-1] == b - 60, (name, day)
    assert window.size == (b - a) // 60
    assert cal.session_mask(a, name) is True and cal.session_mask(b, name) is False
    assert cal.session_mask(a - 1, name) is False and cal.session_mask(b - 1, name) is True


def test_session_names_and_errors():
    t = ts("2024-07-16 12:00")
    assert cal.session_mask(t, " London ") is True and cal.session_mask(t, "NEWYORK") is True
    with pytest.raises(ValueError, match="unknown session"):
        cal.session_mask(t, "tokyo")
    assert set(cal.SESSION_NAMES) == {"asia", "london", "newyork", "overlap"}


# ---------------------------------------------------------------------------------------
# shapes, types and input checks

def test_scalar_and_array_shapes():
    t = ts("2024-07-16 12:00")
    assert isinstance(cal.cet_offset_hours(t), int) and cal.cet_offset_hours(t) == 2
    assert isinstance(cal.prop_day(np.int64(t)), int)
    assert isinstance(cal.ny_offset_hours(t), int) and cal.ny_offset_hours(t) == -4
    grid = np.array([[t, t + 200 * DAY], [t + 1, t + 2]], dtype=np.int64)
    for fn in (cal.cet_offset_hours, cal.ny_offset_hours, cal.prop_day, cal.ny_weekday, cal.london_offset_hours):
        out = fn(grid)
        assert isinstance(out, np.ndarray) and out.shape == (2, 2) and out.dtype == np.int64
    assert cal.session_mask(grid, "london").shape == (2, 2)
    assert cal.cet_offset_hours(np.zeros(0, dtype=np.int64)).shape == (0,)
    series = pd.Series([t, t + 200 * DAY])
    assert cal.cet_offset_hours(series).tolist() == [2, 1]
    assert cal.prop_day([t, t]).tolist() == [cal.prop_day(t)] * 2
    assert cal.prop_day(float(t)) == cal.prop_day(t)
    as_dt64 = np.array([t, t + 200 * DAY], dtype="datetime64[s]").astype("datetime64[ns]")
    assert np.array_equal(cal.prop_day(as_dt64), cal.prop_day(np.array([t, t + 200 * DAY])))


def test_day_inputs_and_text_helpers():
    d = dnum("2024-10-27")
    assert cal.day_start_utc(dt.date(2024, 10, 27)) == cal.day_start_utc(d)
    assert cal.day_start_utc(np.datetime64("2024-10-27")) == cal.day_start_utc(d)
    assert cal.day_start_utc(np.array([d, d + 1])).tolist() == [ts("2024-10-26 22:00"), ts("2024-10-27 23:00")]
    assert cal.day_to_str(d) == "2024-10-27"
    assert cal.day_to_str(np.array([d, d + 1])).tolist() == ["2024-10-27", "2024-10-28"]
    assert cal.day_weekday(d) == 6 and cal.day_weekday(np.array([d + 1])).tolist() == [0]
    assert cal.utc_str(ts("2024-10-27 01:00")) == "2024-10-27 01:00:00 UTC"
    assert cal.utc_str(np.array([0])).tolist() == ["1970-01-01 00:00:00 UTC"]
    for bad in (dt.datetime(2024, 1, 1), 1.5, "2024-01-01", -1, np.datetime64("2024-01-01T05")):
        with pytest.raises(ValueError):
            cal.day_start_utc(bad)


@pytest.mark.parametrize("bad, pattern", [
    (float("nan"), "NaN"),
    (1.5, "fractional"),
    (1_720_000_000_000, "milliseconds"),
    (-1, "between 1970"),
    ("1720000000", "integers"),
    (True, "boolean"),
    (np.array(["NaT"], dtype="datetime64[s]"), "NaT"),
    (np.array([1_720_000_000_500], dtype="datetime64[ms]"), "whole seconds"),
    (np.array([1, None], dtype=object), "integers"),
])
def test_invalid_instants_raise(bad, pattern):
    with pytest.raises(ValueError, match=pattern):
        cal.prop_day(bad)


def test_vectorised_speed():
    times = np.arange(T_2015, T_2015 + 1_000_000 * H, H, dtype=np.int64)
    t0 = time.perf_counter()
    cal.prop_day(times)
    cal.ny_weekday(times)
    cal.session_mask(times, "overlap")
    assert time.perf_counter() - t0 < 5.0


def test_no_tz_database_or_alphamaster_imports():
    src = (ROOT / "propkit" / "calendar.py").read_text(encoding="utf-8")
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", src, flags=re.M)
    assert not {"zoneinfo", "pytz", "dateutil", "tzdata"} & {m.split(".")[0] for m in imports}
    forbidden = {"model_core", "data_pipeline", "config", "web", "utils", "strategy_manager", "execution",
                 "scripts"}
    for name in ("calendar", "costs", "bars", "__init__"):
        text = (ROOT / "propkit" / f"{name}.py").read_text(encoding="utf-8")
        mods = {m.split(".")[0] for m in re.findall(r"^\s*(?:from|import)\s+([\w.]+)", text, flags=re.M)}
        assert not forbidden & mods, name
        assert text.isascii(), name
