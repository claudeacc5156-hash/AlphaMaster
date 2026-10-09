"""Tests for propkit/bars.py: loading, validation, locked-path guard, bar size, synthetic bars. Research only.

Synthetic data only (files written to pytest's temp folder).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from propkit import bars as pb
from propkit import calendar as cal

UTC = dt.timezone.utc
T0 = 1717977600                       # 2024-06-10 00:00 UTC, a Monday


def _frame(n: int = 50, step: int = 3600, spread: bool = True, start: int = T0) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    t = start + step * np.arange(n, dtype=np.int64)
    t[n // 2:] += 2 * 86400                                  # one weekend-like gap
    c = 2300.0 + np.cumsum(rng.normal(0, 2.0, n))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + 0.5
    lo = np.minimum(o, c) - 0.5
    df = pd.DataFrame({"time": t, "open": o, "high": h, "low": lo, "close": c,
                       "tick_volume": rng.integers(10, 900, n)})
    if spread:
        df["spread"] = np.round(rng.uniform(0.2, 0.6, n), 2)
    return df


# ---------------------------------------------------------------------------------------
# loading

def test_load_parquet_int_seconds(tmp_path):
    src = _frame()
    p = tmp_path / "XAUUSD_H1.parquet"
    src.to_parquet(p, index=False)
    b = pb.load_bars(p)
    assert list(b.columns) == ["time", "open", "high", "low", "close", "spread", "tick_volume"]
    assert b["time"].dtype == np.int64 and all(b[c].dtype == np.float64 for c in ("open", "high", "low",
                                                                                  "close", "spread"))
    assert np.array_equal(b["time"].to_numpy(), src["time"].to_numpy())
    assert np.allclose(b["close"], src["close"]) and np.allclose(b["spread"], src["spread"])
    assert isinstance(b.index, pd.RangeIndex) and b.index[0] == 0
    assert pb.load_bars(str(p)).equals(b)


@pytest.mark.parametrize("unit", ["ns", "us", "s"])
@pytest.mark.parametrize("tz", [None, "UTC", "Asia/Singapore"])
def test_load_parquet_datetime_columns(tmp_path, unit, tz):
    src = _frame()
    times = pd.to_datetime(src["time"], unit="s", utc=True)
    times = times.dt.tz_convert(tz) if tz else times.dt.tz_localize(None)
    src2 = src.assign(time=times.astype(f"datetime64[{unit}, {tz}]" if tz else f"datetime64[{unit}]"))
    p = tmp_path / "XAUUSD_H1.parquet"
    src2.to_parquet(p, index=False)
    assert np.array_equal(pb.load_bars(p)["time"].to_numpy(), src["time"].to_numpy())   # naive = UTC


@pytest.mark.parametrize("mult", [1000, 1_000_000, 1_000_000_000])
def test_epoch_milli_micro_nano_seconds_are_converted(tmp_path, mult):
    src = _frame()
    p = tmp_path / "x.parquet"
    src.assign(time=src["time"] * mult).to_parquet(p, index=False)
    assert np.array_equal(pb.load_bars(p)["time"].to_numpy(), src["time"].to_numpy())


def test_load_csv_and_iso_text_times(tmp_path):
    src = _frame(spread=False)
    p = tmp_path / "bars.csv"
    src.to_csv(p, index=False)
    b = pb.load_bars(p)
    assert "spread" not in b.columns and np.array_equal(b["time"].to_numpy(), src["time"].to_numpy())
    iso = pd.to_datetime(src["time"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    src.assign(time=iso).to_csv(p, index=False)
    assert np.array_equal(pb.load_bars(p)["time"].to_numpy(), src["time"].to_numpy())
    plus8 = pd.to_datetime(src["time"], unit="s", utc=True).dt.tz_convert("Asia/Singapore")
    src.assign(time=plus8.dt.strftime("%Y-%m-%d %H:%M:%S%z")).to_csv(p, index=False)
    assert np.array_equal(pb.load_bars(p)["time"].to_numpy(), src["time"].to_numpy())
    src.assign(time=iso.str.rstrip("Z")).to_csv(p, index=False)
    with pytest.raises(ValueError, match="without a UTC offset"):
        pb.load_bars(p)
    src.assign(time=pd.to_datetime(src["time"], unit="s").dt.strftime("%Y-%m-%d")).to_csv(p, index=False)
    with pytest.raises(ValueError, match="without a UTC offset"):
        pb.load_bars(p)


def test_csv_column_names_are_case_insensitive(tmp_path):
    p = tmp_path / "bars.csv"
    _frame().rename(columns=str.upper).to_csv(p, index=False)
    assert len(pb.load_bars(p)) == 50


def test_missing_file_and_unknown_suffix(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        pb.load_bars(tmp_path / "nope.parquet")
    p = tmp_path / "bars.xlsx"
    p.write_text("x")
    with pytest.raises(ValueError, match=".parquet or .csv"):
        pb.load_bars(p)
    q = tmp_path / "broken.parquet"
    q.write_text("not parquet")
    with pytest.raises(ValueError, match="cannot read"):
        pb.load_bars(q)


# ---------------------------------------------------------------------------------------
# locked holdout

@pytest.mark.parametrize("path", [
    "data/locked_holdout/XAUUSD_H1.parquet", "data\\LOCKED_HOLDOUT\\x.csv", "D:/research/Locked_Holdout",
    "XAUUSD_H1.parquet.locked", "x.LOCKED", "data/x.locked/", "C:\\data\\x.Locked",
])
def test_locked_paths_are_refused_before_opening(path):
    assert pb.is_locked_path(path)
    with pytest.raises(pb.LockedPathError, match="locked holdout"):
        pb.load_bars(path)                     # the file does not exist: the refusal comes first
    with pytest.raises(ValueError):
        pb.check_not_locked(path, "trade file")


def test_locked_check_follows_resolved_paths(tmp_path):
    real = tmp_path / "locked_holdout"
    real.mkdir()
    link = tmp_path / "innocent"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available")
    assert not pb.is_locked_path(str(link / "bars.parquet"))
    with pytest.raises(pb.LockedPathError):
        pb.check_not_locked(str(link / "bars.parquet"))


@pytest.mark.parametrize("path", ["data/XAUUSD_H1.parquet", "locked.parquet", "holdout/x.csv", "x.locked.csv"])
def test_ordinary_paths_pass(path):
    assert not pb.is_locked_path(path)
    pb.check_not_locked(path)


# ---------------------------------------------------------------------------------------
# validation

def _break(df: pd.DataFrame, col: str, i: int, value) -> pd.DataFrame:
    df = df.copy()
    if col in df.columns and df[col].dtype.kind in "iu" and isinstance(value, float):
        df[col] = df[col].astype(np.float64)
    df.loc[i, col] = value
    return df


@pytest.mark.parametrize("col, value, pattern", [
    ("close", np.nan, "missing, infinite or non-positive"),
    ("open", np.inf, "missing, infinite or non-positive"),
    ("low", -1.0, "non-positive"),
    ("high", 1.0, "high below open or close"),
    ("low", 99999.0, "low above open or close"),
    ("spread", np.nan, "missing or infinite spread"),
    ("spread", -0.1, "negative spread"),
    ("time", T0 + 6 * 3600, "duplicate time"),
    ("time", T0 - 3600, "earlier than the row before"),
    ("time", np.nan, "missing values"),
])
def test_validation_errors_name_the_row(col, value, pattern):
    df = _break(_frame(), col, 7, value)
    with pytest.raises(ValueError, match=pattern) as err:
        pb.validate_bars(df, source="T.parquet")
    assert "T.parquet" in str(err.value)
    if col != "time":
        assert "row 7" in str(err.value) and cal.utc_str(int(df.loc[7, "time"])) in str(err.value)


def test_validation_structure_errors():
    with pytest.raises(ValueError, match="missing column"):
        pb.validate_bars(_frame().drop(columns="low"))
    with pytest.raises(ValueError, match="at least 2"):
        pb.validate_bars(_frame().iloc[:1])
    with pytest.raises(ValueError, match="DataFrame"):
        pb.validate_bars(_frame().to_numpy())
    with pytest.raises(ValueError, match="numeric"):
        pb.validate_bars(_frame().assign(close="x"))
    with pytest.raises(ValueError, match="fractional"):
        pb.validate_bars(_frame().assign(time=_frame()["time"] + 0.5))
    with pytest.raises(ValueError, match="1970"):
        pb.validate_bars(_frame().assign(time=_frame()["time"] - T0 - 3600 * 10))
    with pytest.raises(ValueError, match="spread_scale"):
        pb.validate_bars(_frame(), spread_scale=0)


def test_spread_in_points_is_refused_unless_scaled():
    df = _frame().assign(spread=np.full(50, 34))             # MT5 points on a 2-digit quote
    with pytest.raises(ValueError, match="broker points"):
        pb.validate_bars(df)
    b = pb.validate_bars(df, spread_scale=0.01)
    assert np.allclose(b["spread"], 0.34)
    assert np.allclose(pb.validate_bars(_frame().assign(spread=5.0))["spread"], 5.0)   # a wide but plausible USD spread


def test_validate_does_not_modify_input_and_keeps_values():
    df = _frame()
    before = df.copy()
    b = pb.validate_bars(df)
    pd.testing.assert_frame_equal(df, before)
    assert np.allclose(b["tick_volume"], df["tick_volume"])
    assert len(b) == len(df)


# ---------------------------------------------------------------------------------------
# helpers

@pytest.mark.parametrize("step", [3600, 1800, 900])
def test_infer_bar_seconds_with_gaps(step):
    df = _frame(n=200, step=step)
    assert pb.infer_bar_seconds(df["time"]) == step
    assert pb.infer_bar_seconds(pb.validate_bars(df)["time"].to_numpy()) == step


def test_infer_bar_seconds_ties_and_errors():
    assert pb.infer_bar_seconds([0, 900, 2700, 3600, 5400]) == 900          # 900 x2, 1800 x2 -> smaller
    with pytest.raises(ValueError):
        pb.infer_bar_seconds([5])


def test_bar_index_at():
    times = np.array([0, 3600, 7200, 86400], dtype=np.int64)
    assert pb.bar_index_at(times, -1) == -1
    assert pb.bar_index_at(times, 0) == 0 and pb.bar_index_at(times, 3599) == 0
    assert pb.bar_index_at(times, 50000) == 2
    assert pb.bar_index_at(times, 50000, bar_seconds=3600) == -1              # inside a gap
    assert pb.bar_index_at(times, 7300, bar_seconds=3600) == 2
    assert pb.bar_index_at(times, np.array([0, 89999, 90000]), bar_seconds=3600).tolist() == [0, 3, -1]
    with pytest.raises(ValueError):
        pb.bar_index_at(times, 1.5)


def test_bars_summary():
    s = pb.bars_summary(pb.validate_bars(_frame()))
    assert s["n_bars"] == 50 and s["bar_seconds"] == 3600 and s["n_gaps"] == 1
    assert s["first_time"] == T0 and s["first_time_utc"] == "2024-06-10 00:00:00 UTC"
    assert 0.2 <= s["spread_median"] <= s["spread_p90"] <= 0.6
    assert pb.bars_summary(pb.validate_bars(_frame(spread=False)))["spread_median"] is None


def test_metals_market_hours_follow_us_dst():
    sun_summer = int(dt.datetime(2024, 7, 14, 22, tzinfo=UTC).timestamp())    # 18:00 EDT
    sun_winter = int(dt.datetime(2024, 1, 14, 23, tzinfo=UTC).timestamp())    # 18:00 EST
    fri_summer = int(dt.datetime(2024, 7, 19, 20, tzinfo=UTC).timestamp())    # 16:00 EDT
    probe = np.array([sun_summer - 3600, sun_summer, sun_winter - 3600, sun_winter, fri_summer,
                      fri_summer + 3600, sun_summer + 86400 - 3600, sun_summer + 86400 - 7200])
    assert pb.metals_market_open(probe).tolist() == [False, True, False, True, True, False, False, True]


def test_synthetic_bars_deterministic_valid_and_dst_aware():
    a = pb.synthetic_bars(T0, 2000, seed=5)
    b = pb.synthetic_bars(T0, 2000, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(pb.synthetic_bars(T0, 2000, seed=6))
    pd.testing.assert_frame_equal(pb.validate_bars(a), a)
    assert len(a) == 2000 and pb.infer_bar_seconds(a["time"]) == 3600
    t = a["time"].to_numpy()
    assert pb.metals_market_open(t).all()
    reopen = t[1:][np.diff(t) > 86400]                      # first bar after each weekend
    assert (cal.ny_weekday(reopen) == 6).all()
    assert set(((reopen + cal.ny_offset_hours(reopen) * 3600) % 86400).tolist()) == {18 * 3600}
    m15 = pb.synthetic_bars(T0, 500, bar_seconds=900, spread=None, market_hours="always")
    assert "spread" not in m15.columns and np.all(np.diff(m15["time"]) == 900)
    with pytest.raises(ValueError):
        pb.synthetic_bars(T0 + 1, 10)


def test_infer_bar_seconds_refuses_a_gap_taken_for_the_bar_size():
    """Review finding: two bars, Friday 20:00 and Sunday 22:00 UTC, gave a 50-hour 'bar' and a Saturday
    fill was accepted as inside it."""
    fri = 1753473600                                                     # 2025-07-25 20:00 UTC, a Friday
    with pytest.raises(ValueError, match="not a bar size but a gap"):
        pb.infer_bar_seconds([fri, fri + 180_000])
    with pytest.raises(ValueError, match="not a bar size but a gap"):
        pb.infer_bar_seconds([0, 86_400])                                # one whole-day step: ambiguous
    assert pb.infer_bar_seconds([0, 86_400, 172_800, 432_000]) == 86_400  # D1 bars seen twice: accepted
    assert pb.infer_bar_seconds([fri, fri + 3600, fri + 183_600]) == 3600 # tie 3600 / gap -> the smaller
    assert pb.infer_bar_seconds([0, 14_400, 28_800]) == 14_400           # H4: still a bar size
