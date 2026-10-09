"""zeno_v1 data layer (D1, Costs): bid + ask M15 files, the holdout lock, the locked-path refusal, the
cost cell's ask side and propkit BARS, H1 bars from M15. Hand-made bars only (NOT market data).
Research only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propkit import bars as pb
from propkit import zeno_v1 as z
from tests.unit.zeno_v1_testkit import utc

LOCK = 1759017600                         # 2025-09-28 00:00 UTC


def _pair(start: int, n: int, step: int = 900, spread=0.25, price=2000.0):
    """bid and ask tables: bid o/h/l/c = p, p+1, p-1, p+0.5 with p rising 0.1 per bar; ask = bid + spread."""
    t = start + step * np.arange(n, dtype=np.int64)
    p = price + 0.1 * np.arange(n)
    bid = pd.DataFrame({"time": t, "open": p, "high": p + 1.0, "low": p - 1.0, "close": p + 0.5})
    ask = bid.copy()
    sp = np.broadcast_to(np.asarray(spread, dtype=np.float64), (n,))
    for c in ("open", "high", "low", "close"):
        ask[c] = bid[c] + sp
    return bid, ask


# ---------------------------------------------------------------------------------------------------
# bidask_frame

def test_frame_columns_and_spread_open():
    bid, ask = _pair(utc("2024-03-04 00:00"), 40, spread=0.25)
    f = z.bidask_frame(bid, ask)
    assert tuple(f.columns) == z.FRAME_COLUMNS
    assert f["time"].dtype == np.int64
    assert np.allclose(f["spread_open"], 0.25)
    assert np.allclose(f["ask_high"] - f["bid_high"], 0.25)
    s = f.attrs["zeno_v1"]
    assert s["n_bars"] == 40 and s["bar_seconds"] == 900 and s["n_gaps"] == 0
    assert s["first_time_utc"] == "2024-03-04 00:00:00 UTC" and s["lock_utc"] == z.LOCK_TEXT
    assert s["ask_below_bid"] == {"open": 0, "high": 0, "low": 0, "close": 0}


def test_holdout_lock_last_allowed_bar_and_first_refused_bar():
    # 2025-09-27 23:45 UTC is the last M15 bar before the lock (it closes exactly at the lock): accepted.
    bid, ask = _pair(LOCK - 900 * 8, 8)
    f = z.bidask_frame(bid, ask)
    assert int(f["time"].iloc[-1]) == utc("2025-09-27 23:45")
    # one more bar opening at 2025-09-28 00:00 UTC: refused, with no override
    bid, ask = _pair(LOCK - 900 * 8, 9)
    with pytest.raises(z.HoldoutLockError, match="2025-09-28 00:00:00 UTC"):
        z.bidask_frame(bid, ask)
    # the lock is checked on each side on its own (here only the ask file crosses it)
    bid8, _ = _pair(LOCK - 900 * 8, 8)
    with pytest.raises(z.HoldoutLockError):
        z.bidask_frame(bid8, ask)
    z.check_before_lock([LOCK - 1])
    with pytest.raises(z.HoldoutLockError):
        z.check_before_lock([LOCK])
    assert issubclass(z.HoldoutLockError, ValueError)


def test_bars_before_the_range_start_are_cut_and_counted():
    # D1: the range starts 2015-01-01 00:00 UTC. 100 bars from 2014-12-31 23:00 UTC: the 4 before it are cut
    # (and counted), the first bar kept opens at 2015-01-01 00:00 UTC.
    bid, ask = _pair(utc("2014-12-31 23:00"), 100)
    f = z.bidask_frame(bid, ask)
    assert int(f["time"].iloc[0]) == utc("2015-01-01 00:00") and len(f) == 96
    assert f["bid_open"].iloc[0] == pytest.approx(2000.4)                  # the 5th input row
    s = f.attrs["zeno_v1"]
    assert s["cut_before_range_start"] == 4 and s["range_start_utc"] == "2015-01-01 00:00:00 UTC"
    assert s["first_time_utc"] == "2015-01-01 00:00:00 UTC" and s["n_bars"] == 96
    assert s["starts_after_range_start"] is False and "4 bar(s) before 2015-01-01" in s["range_note"]
    # every bar before the start: refused (nothing of D1's range is left)
    bid, ask = _pair(utc("2014-12-01 00:00"), 10)
    with pytest.raises(ValueError, match="2015-01-01 00:00:00 UTC"):
        z.bidask_frame(bid, ask)
    # M1 frames (the D15 second run) are not cut
    bid1, ask1 = _pair(utc("2014-12-31 23:58"), 4, step=60)
    assert len(z.bidask_frame(bid1, ask1, bar_seconds=60, source="M1")) == 4


def test_the_period_all_starts_in_2015_and_a_late_start_is_flagged():
    from propkit import zeno_report as zr
    f = z.synthetic_m15_bidask(start=utc("2014-06-01 00:00"), n_bars=40_000, seed=3)
    assert f.attrs["zeno_v1"]["cut_before_range_start"] > 0
    allp = [p for p in zr.period_table(z.prepare(f)) if p["period"] == "all"][0]
    assert allp["first_date"] == "2015-01-01"
    ok = z.synthetic_m15_bidask(start=utc("2015-01-01 00:00"), n_bars=200).attrs["zeno_v1"]
    assert ok["cut_before_range_start"] == 0 and ok["starts_after_range_start"] is False and ok["range_note"] == ""
    late = z.synthetic_m15_bidask(start=utc("2015-03-02 00:00"), n_bars=200).attrs["zeno_v1"]
    assert late["starts_after_range_start"] is True and "2015-03-02" in late["range_note"]


def test_synthetic_frame_respects_the_lock():
    # synthetic bars follow metals hours (no weekend bars): 150 bars from Thursday 2025-09-25 00:00 end
    # before the lock; 400 bars run past the weekend into Sunday 2025-09-28 22:00 UTC, after the lock.
    f = z.synthetic_m15_bidask(start=utc("2025-09-25 00:00"), n_bars=150)
    assert int(f["time"].iloc[-1]) < LOCK
    with pytest.raises(z.HoldoutLockError):
        z.synthetic_m15_bidask(start=utc("2025-09-25 00:00"), n_bars=400)


def test_bid_and_ask_must_hold_the_same_bars():
    bid, ask = _pair(utc("2024-03-04 00:00"), 40)
    with pytest.raises(ValueError, match="do not hold the same bars"):
        z.bidask_frame(bid, ask.iloc[:-1])
    ask2 = ask.drop(index=5).reset_index(drop=True)
    bid2 = bid.drop(index=6).reset_index(drop=True)
    with pytest.raises(ValueError, match="first difference at row 5"):
        z.bidask_frame(bid2, ask2)


def test_ask_open_below_bid_open_is_refused_other_crossings_are_counted():
    bid, ask = _pair(utc("2024-03-04 00:00"), 40)
    bad = ask.copy()
    bad.loc[7, "open"] = bid.loc[7, "open"] - 0.01
    with pytest.raises(ValueError, match="ask_open below bid_open"):
        z.bidask_frame(bid, bad)
    odd = ask.copy()
    odd.loc[3, "high"] = bid.loc[3, "high"] - 0.05          # still >= its own open/close: a valid ask bar
    odd.loc[4, "close"] = bid.loc[4, "close"] - 0.01
    f = z.bidask_frame(bid, odd)
    assert f.attrs["zeno_v1"]["ask_below_bid"] == {"open": 0, "high": 1, "low": 0, "close": 1}


def test_bar_size_and_alignment():
    bid, ask = _pair(utc("2024-03-04 00:00"), 40, step=300)                     # M5 bars
    with pytest.raises(ValueError, match="300 s apart, expected 900 s"):
        z.bidask_frame(bid, ask)
    bid, ask = _pair(utc("2024-03-04 00:00") + 420, 40)                         # M15 bars opening at :07
    with pytest.raises(ValueError, match="900 s boundary"):
        z.bidask_frame(bid, ask)
    bid, ask = _pair(utc("2024-03-04 00:00"), 40, step=60)                      # M1 is fine with 60 s
    f = z.bidask_frame(bid, ask, bar_seconds=60, source="M1")
    assert f.attrs["zeno_v1"]["bar_seconds"] == 60


def test_invalid_bars_are_refused_like_propkit_bars():
    bid, ask = _pair(utc("2024-03-04 00:00"), 40)
    bad = bid.copy()
    bad.loc[2, "high"] = bad.loc[2, "low"] - 1
    with pytest.raises(ValueError):
        z.bidask_frame(bad, ask)
    with pytest.raises(ValueError, match="missing column"):
        z.bidask_frame(bid.drop(columns="close"), ask)


# ---------------------------------------------------------------------------------------------------
# files

def test_load_csv_pair_records_files_and_hashes(tmp_path):
    bid, ask = _pair(utc("2024-03-04 00:00"), 40)
    bp, ap = tmp_path / "XAUUSD_M15_bid.csv", tmp_path / "XAUUSD_M15_ask.csv"
    bid.to_csv(bp, index=False)
    ask.to_csv(ap, index=False)
    f = z.load_m15_bidask(bp, ap)
    assert len(f) == 40 and np.allclose(f["spread_open"], 0.25)
    s = f.attrs["zeno_v1"]
    assert s["bid_file"] == str(bp) and len(s["bid_sha256"]) == 64 and s["bid_sha256"] != s["ask_sha256"]


def test_load_dukascopy_node_csv_with_ms_timestamps(tmp_path):
    bid, ask = _pair(utc("2024-03-04 00:00"), 40)
    for df in (bid, ask):
        df.insert(0, "timestamp", df.pop("time") * 1000)
        df["volume"] = 1.5
    bp, ap = tmp_path / "bid.csv", tmp_path / "ask.csv"
    bid.to_csv(bp, index=False)
    ask.to_csv(ap, index=False)
    f = z.load_m15_bidask(bp, ap)
    assert int(f["time"].iloc[0]) == utc("2024-03-04 00:00")


def test_load_parquet_pair(tmp_path):
    pytest.importorskip("pyarrow")
    bid, ask = _pair(utc("2024-03-04 00:00"), 40)
    bp, ap = tmp_path / "bid.parquet", tmp_path / "ask.parquet"
    bid.to_parquet(bp, index=False)
    ask.to_parquet(ap, index=False)
    assert len(z.load_m15_bidask(bp, ap)) == 40


def test_load_m1_pair(tmp_path):
    bid, ask = _pair(utc("2024-03-04 00:00"), 30, step=60)
    bp, ap = tmp_path / "m1_bid.csv", tmp_path / "m1_ask.csv"
    bid.to_csv(bp, index=False)
    ask.to_csv(ap, index=False)
    assert z.load_m1_bidask(bp, ap).attrs["zeno_v1"]["bar_seconds"] == 60
    with pytest.raises(ValueError, match="expected 900 s"):
        z.load_m15_bidask(bp, ap)


def test_file_crossing_the_lock_is_refused(tmp_path):
    bid, ask = _pair(LOCK - 900 * 8, 9)
    bp, ap = tmp_path / "bid.csv", tmp_path / "ask.csv"
    bid.to_csv(bp, index=False)
    ask.to_csv(ap, index=False)
    with pytest.raises(z.HoldoutLockError):
        z.load_m15_bidask(bp, ap)


def test_locked_paths_are_refused_before_anything_is_read(tmp_path):
    # The bid path is a real file that cannot be parsed; the ask path names the locked holdout (and does
    # not exist). The refusal must come first: no read of either file is attempted.
    garbage = tmp_path / "bid.csv"
    garbage.write_bytes(b"\x00\x01 not a csv")
    for locked in (tmp_path / "locked_holdout" / "ask.csv", tmp_path / "ask.csv.locked"):
        with pytest.raises(pb.LockedPathError):
            z.load_m15_bidask(garbage, locked)
        with pytest.raises(pb.LockedPathError):
            z.load_m1_bidask(locked, garbage)
        with pytest.raises(pb.LockedPathError):
            z.read_news_csv(locked)
    assert not (tmp_path / "locked_holdout").exists()


def test_missing_file_and_wrong_suffix(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        z.load_m15_bidask(tmp_path / "nope.csv", tmp_path / "nope2.csv")
    p = tmp_path / "bid.json"
    p.write_text("{}", encoding="ascii")
    with pytest.raises(ValueError, match="parquet or .csv"):
        z.load_m15_bidask(p, p)


# ---------------------------------------------------------------------------------------------------
# cost cells: the ask side and propkit BARS

def test_ask_side_s1_and_multiplier():
    bid, ask = _pair(utc("2024-03-04 00:00"), 8, spread=[0.2, 0.3, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2])
    f = z.bidask_frame(bid, ask)
    a1 = z.ask_side(f, "S1", 1.0)
    assert np.array_equal(a1["ask_high"], f["ask_high"].to_numpy())          # k = 1: exactly the ask file
    assert np.allclose(a1["spread_entry"], [0.2, 0.3, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2])
    a15 = z.ask_side(f, "S1", 1.5)
    # ask = bid + 1.5 x (ask - bid): bar 1 open 2000.1 + 1.5 x 0.3 = 2000.55
    assert a15["ask_open"][1] == pytest.approx(2000.55)
    assert np.allclose(a15["spread_entry"], 1.5 * np.array([0.2, 0.3, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2]))
    with pytest.raises(ValueError):
        z.ask_side(f, "S3")
    with pytest.raises(ValueError):
        z.ask_side(f, "S1", 0.0)


def test_s2_spread_by_hour():
    # 0.20 for bars opening 21:00-23:45 UTC (05:00-08:00 SGT), 0.18 otherwise
    times = [utc("2024-03-04 20:45"), utc("2024-03-04 21:00"), utc("2024-03-04 23:45"), utc("2024-03-05 00:00"),
             utc("2024-07-01 21:30"), utc("2024-07-01 13:00")]
    assert z.s2_spread(times).tolist() == [0.18, 0.20, 0.20, 0.18, 0.20, 0.18]
    bid, ask = _pair(utc("2024-03-04 20:30"), 4, spread=0.9)
    f = z.bidask_frame(bid, ask)
    a = z.ask_side(f, "S2", 2.0)
    assert np.allclose(a["spread_entry"], [0.36, 0.36, 0.40, 0.40])
    assert np.allclose(a["ask_low"] - f["bid_low"], [0.36, 0.36, 0.40, 0.40])   # the data ask is not used


def test_to_propkit_bars_carries_the_cell_spread():
    bid, ask = _pair(utc("2024-03-04 00:00"), 12, spread=0.25)
    f = z.bidask_frame(bid, ask)
    b = z.to_propkit_bars(f)
    assert list(b.columns[:6]) == ["time", "open", "high", "low", "close", "spread"]
    assert np.array_equal(b["close"].to_numpy(), f["bid_close"].to_numpy())
    assert np.allclose(b["spread"], 0.25)
    assert np.allclose(z.to_propkit_bars(f, "S1", 2.0)["spread"], 0.50)
    assert np.allclose(z.to_propkit_bars(f, "S2", 1.0)["spread"], 0.18)


# ---------------------------------------------------------------------------------------------------
# H1 bars (D1)

def test_h1_from_m15_with_a_partial_hour():
    t = [utc("2024-03-04 10:00"), utc("2024-03-04 10:15"), utc("2024-03-04 10:30"), utc("2024-03-04 10:45"),
         utc("2024-03-04 11:15"), utc("2024-03-04 11:30")]                     # 11:00 is missing
    bid = pd.DataFrame({"time": t, "open": [10, 11, 12, 13, 20, 21.0], "high": [11, 15, 13, 14, 22, 23.0],
                        "low": [9, 10, 8, 12, 19, 18.0], "close": [11, 12, 13, 14, 21, 22.0]})
    ask = bid.copy()
    ask[["open", "high", "low", "close"]] += 0.3
    h1 = z.h1_from_m15(z.bidask_frame(bid, ask))
    assert h1["time"].tolist() == [utc("2024-03-04 10:00"), utc("2024-03-04 11:00")]
    assert h1[["open", "high", "low", "close"]].values.tolist() == [[10, 15, 8, 14], [20, 23, 18, 22]]
    assert h1["n_m15"].tolist() == [4, 2]
    assert h1["close_time"].tolist() == [utc("2024-03-04 11:00"), utc("2024-03-04 12:00")]


def test_news_csv_keeps_the_four_events(tmp_path):
    p = tmp_path / "cal.csv"
    p.write_text("event,datetime_utc,kind\nNFP,2024-03-08T13:30Z,scheduled\nGDP,2024-03-28T12:30Z,scheduled\n"
                 "fomc,2020-03-15T21:00Z,unscheduled\n", encoding="ascii")
    cal = z.read_news_csv(p)
    assert cal.times.tolist() == [utc("2020-03-15 21:00"), utc("2024-03-08 13:30")]
    assert cal.names == ("FOMC", "NFP") and cal.kinds == ("unscheduled", "scheduled")
    assert cal.summary()["per_event"] == {"FOMC": 1, "NFP": 1} and len(cal.sha256) == 64
    q = tmp_path / "none.csv"
    q.write_text("event,datetime_utc\nGDP,2024-03-28T12:30Z\n", encoding="ascii")
    with pytest.raises(ValueError, match="no NFP"):
        z.read_news_csv(q)
