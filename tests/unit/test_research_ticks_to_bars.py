"""Tests for scripts/research/ticks_to_bars.py (own tick files -> AlphaMaster bars). Research only.

All data is synthetic. The tests never open a *.locked file: the locked part is checked through
the train file, the manifest's bar count and the sha256 the tool computed in memory.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "research" / "ticks_to_bars.py"

_spec = importlib.util.spec_from_file_location("research_ticks_to_bars", SCRIPT)
tb = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tb              # dataclasses look their module up in sys.modules
_spec.loader.exec_module(tb)

NS = 1_000_000_000
CUTOFF_S = 1759017600                      # 2025-09-28 00:00:00 UTC: the default cutoff (owner decision)
CUTOFF = "2025-09-28T00:00:00Z"
NAME = "XAUUSD_H1.parquet"
MANIFEST = "XAUUSD_H1_manifest.json"


# ---------------------------------------------------------------------------------------
# synthetic ticks, writers and an independent pandas reference

def _ticks(start: str = "2024-01-08 00:00", hours: int = 420, seed: int = 0, per_hour=(3, 12),
           repeat: float = 0.0, whole_seconds: bool = False, level: float = 2000.0) -> pd.DataFrame:
    """UTC ticks (t = int64 ns, bid, ask in whole cents), sorted by time. With repeat > 0 a share
    of the rows keeps the previous bid or ask (so MT5 files get blank cells)."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start, tz="UTC").value
    counts = rng.integers(per_hour[0], per_hour[1] + 1, hours)
    unit = 1000 if whole_seconds else 1
    t = np.concatenate([t0 + h * 3600 * NS + np.sort(rng.integers(0, 3600 * 1000 // unit, c)) * unit * 1_000_000
                        for h, c in enumerate(counts)])
    n = len(t)
    bid = np.round(level + np.cumsum(rng.normal(0, 0.3, n)), 2)
    ask = np.round(bid + rng.integers(10, 50, n) / 100, 2)
    for i in range(1, n):
        if rng.random() < repeat:
            bid[i] = bid[i - 1]
        if rng.random() < repeat and ask[i - 1] > bid[i]:
            ask[i] = ask[i - 1]
        if ask[i] <= bid[i]:
            ask[i] = round(bid[i] + 0.2, 2)
    return pd.DataFrame({"t": t.astype("int64"), "bid": bid, "ask": ask})


def _ref_bars(ticks: pd.DataFrame, tf_s: int = 3600, price: str = "bid", exact_mean: bool = False) -> pd.DataFrame:
    """Bars from ticks held in memory: first/last by (time, input order); spread = last ask - bid."""
    df = ticks.reset_index(drop=True).copy()
    df["seq"] = np.arange(len(df))
    df = df.sort_values(["t", "seq"], kind="stable")
    df["p"] = df[price] if price != "mid" else (df["bid"] + df["ask"]) / 2.0
    df["spr"] = df["ask"] - df["bid"]
    df["bar"] = (df["t"] // (tf_s * NS)) * tf_s
    g = df.groupby("bar", sort=True)
    n = g.size()
    if exact_mean:                          # the tool's documented definition (sums in units of 1e-8)
        q = np.rint(df["spr"] / 1e-8).astype("int64")
        mean = q.groupby(df["bar"]).sum().astype("float64") / n.astype("float64") * 1e-8
    else:
        mean = g["spr"].mean()
    out = pd.DataFrame({"time": n.index.to_numpy(dtype="int64"), "open": g["p"].first().to_numpy(),
                        "high": g["p"].max().to_numpy(), "low": g["p"].min().to_numpy(),
                        "close": g["p"].last().to_numpy(), "tick_volume": n.to_numpy(dtype="float64"),
                        "spread": g["spr"].last().to_numpy(), "spread_mean": mean.to_numpy(dtype="float64")})
    return out


def _write_duka(path: Path, ticks: pd.DataFrame) -> Path:
    pd.DataFrame({"timestamp": ticks["t"] // 1_000_000, "askPrice": ticks["ask"], "bidPrice": ticks["bid"],
                  "askVolume": 1.5, "bidVolume": 2.5}).to_csv(path, index=False)
    return path


def _server_time(t_ns, tz: str = "ny+7") -> pd.DatetimeIndex:
    utc = pd.DatetimeIndex(pd.to_datetime(np.asarray(t_ns, dtype="int64"), utc=True))
    if tz == "ny+7":
        return utc.tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)
    return utc.tz_convert(tz).tz_localize(None)


def _write_mt5(path: Path, ticks: pd.DataFrame, tz: str = "ny+7", extra_rows: bool = True) -> Path:
    """MT5 'Export ticks' layout: UTF-16 LE with BOM, tab separated, blank BID/ASK when unchanged."""
    srv = _server_time(ticks["t"], tz)
    lines = ["<DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>"]
    if extra_rows:                          # a row before the ask is known (dropped and counted)
        first = srv[0] - pd.Timedelta(seconds=1)
        lines.append(f"{first:%Y.%m.%d}\t{first:%H:%M:%S}.000\t{ticks['bid'].iloc[0]:.2f}\t\t\t\t2")
    pb = pa = None
    for k, (s, b, a) in enumerate(zip(srv, ticks["bid"], ticks["ask"])):
        bs = "" if (b == pb and k > 0) else f"{b:.2f}"
        as_ = "" if (a == pa and k > 0) else f"{a:.2f}"
        if bs == "" and as_ == "":
            bs = f"{b:.2f}"
        flags = (2 if bs else 0) + (4 if as_ else 0)
        lines.append(f"{s:%Y.%m.%d}\t{s.strftime('%H:%M:%S.%f')[:12]}\t{bs}\t{as_}\t\t\t{flags}")
        if extra_rows and k == 10:          # a LAST/VOLUME-only row: both BID and ASK blank
            lines.append(f"{s:%Y.%m.%d}\t{s.strftime('%H:%M:%S.%f')[:12]}\t\t\t2001.00\t1\t24")
        pb, pa = b, a
    with open(path, "w", encoding="utf-16", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")
    return path


def _run(capsys, *args, priority: bool = False) -> tuple[int, str, str]:
    argv = [str(x) for x in args] + ([] if priority else ["--normal-priority"])
    code = tb.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def _train(out: Path, name: str = NAME) -> pd.DataFrame:
    return pd.read_parquet(out / "train" / name)


def _manifest(out: Path, name: str = MANIFEST) -> dict:
    return json.loads((out / name).read_text(encoding="utf-8"))


def _assert_bars_equal(got: pd.DataFrame, ref: pd.DataFrame) -> None:
    assert list(got.columns[:8]) == ["time", "open", "high", "low", "close", "tick_volume", "spread", "spread_mean"]
    assert str(got["time"].dtype) == "int64"
    assert all(str(got[c].dtype) == "float64" for c in got.columns[1:])
    assert len(got) == len(ref)
    for c in ("time", "open", "high", "low", "close", "tick_volume", "spread"):
        assert np.array_equal(got[c].to_numpy(), ref[c].to_numpy()), c
    assert np.allclose(got["spread_mean"], ref["spread_mean"], rtol=0, atol=1e-8)


# ---------------------------------------------------------------------------------------
# formats

def test_mt5_utf16_blank_cells_ny7_matches_reference(tmp_path, capsys):
    ticks = pd.concat([_ticks("2024-01-08", 220, seed=1, repeat=0.4),
                       _ticks("2024-07-08", 220, seed=2, repeat=0.4)], ignore_index=True)
    src = _write_mt5(tmp_path / "ticks.csv", ticks)
    raw = src.read_bytes()
    assert raw[:2] == b"\xff\xfe"
    text = raw.decode("utf-16")
    assert re.search(r"\t\d+\.\d{2}\t\t\t\t2\r\n", text)          # rows with a blank ASK cell
    assert re.search(r"\t\t\d+\.\d{2}\t\t\t4\r\n", text)          # rows with a blank BID cell
    code, out, err = _run(capsys, "--input", src, "--out", tmp_path / "o", "--tz", "ny+7")
    assert code == 0, err
    assert "mt5" in out
    _assert_bars_equal(_train(tmp_path / "o"), _ref_bars(ticks))
    m = _manifest(tmp_path / "o")
    assert m["inputs"][0]["format"] == "mt5" and m["inputs"][0]["encoding"] == "utf-16"
    assert m["inputs"][0]["separator"] == "tab"
    d = m["rows"]["dropped"]
    assert d["mt5_before_first_quote"] == 1 and d["mt5_no_bid_or_ask_change"] == 1
    assert m["rows"]["kept"] == len(ticks) and m["rows"]["read"] == len(ticks) + 2
    assert m["rows"]["dst_nat_rows"] == 0


def test_mt5_forward_fill_carries_across_files(tmp_path, capsys):
    ticks = _ticks(hours=60, seed=3, repeat=0.5)
    half = len(ticks) // 2
    # make the second file start with blank cells that need the last values of the first file
    a = _write_mt5(tmp_path / "a.csv", ticks.iloc[:half], extra_rows=False)
    second = ticks.iloc[half:].copy()
    second.iloc[0, second.columns.get_loc("bid")] = ticks["bid"].iloc[half - 1]
    second.iloc[0, second.columns.get_loc("ask")] = round(ticks["ask"].iloc[half - 1] + 0.05, 2)
    b = _write_mt5(tmp_path / "b.csv", second, extra_rows=False)
    lines = b.read_bytes().decode("utf-16").split("\r\n")
    cells = lines[1].split("\t")
    assert cells[2] and cells[3]
    cells[2] = ""                          # the bid is the last bid of file a: blank, carried over
    lines[1] = "\t".join(cells)
    b.write_bytes("\r\n".join(lines).encode("utf-16"))
    allt = pd.concat([ticks.iloc[:half], second], ignore_index=True)
    code, _, err = _run(capsys, "--input", a, "--input", b, "--out", tmp_path / "o", "--tz", "ny+7",
                        "--chunksize", 9)
    assert code == 0, err
    _assert_bars_equal(_train(tmp_path / "o"), _ref_bars(allt))
    assert _manifest(tmp_path / "o")["rows"]["dropped"]["mt5_before_first_quote"] == 0


def test_dukascopy_node_ms_and_jforex_gmt(tmp_path, capsys):
    ticks = _ticks(seed=4)
    src = _write_duka(tmp_path / "xauusd-ticks.csv", ticks)
    code, out, err = _run(capsys, "--input", src, "--out", tmp_path / "d")
    assert code == 0, err
    assert "dukascopy-node" in out
    ref = _ref_bars(ticks)
    _assert_bars_equal(_train(tmp_path / "d"), ref)
    m = _manifest(tmp_path / "d")
    assert m["inputs"][0]["format"] == "dukascopy-node" and m["inputs"][0]["epoch_unit"] == "ms"
    # JForex / Dukascopy historical data export, 'Gmt time' header, DD.MM.YYYY HH:MM:SS.fff
    tt = pd.to_datetime(ticks["t"], utc=True)
    jf = tmp_path / "XAUUSD_Ticks.csv"
    pd.DataFrame({"Gmt time": tt.dt.strftime("%d.%m.%Y %H:%M:%S.%f").str[:23], "Ask": ticks["ask"],
                  "Bid": ticks["bid"], "AskVolume": 0.01, "BidVolume": 0.02}).to_csv(jf, index=False)
    code, _, err = _run(capsys, "--input", jf, "--out", tmp_path / "j")
    assert code == 0, err
    _assert_bars_equal(_train(tmp_path / "j"), ref)
    assert _manifest(tmp_path / "j")["inputs"][0]["format"] == "jforex"
    assert (tmp_path / "d" / "train" / NAME).read_bytes() == (tmp_path / "j" / "train" / NAME).read_bytes()


def test_jforex_local_time_with_gmt_suffix(tmp_path, capsys):
    ticks = _ticks("2024-06-03", 60, seed=5)
    local = _server_time(ticks["t"], "Europe/Athens")
    off = np.where(pd.DatetimeIndex(pd.to_datetime(ticks["t"], utc=True)).tz_convert("Europe/Athens")
                   .strftime("%z") == "+0300", "+0300", "+0200")
    times = [f"{s.strftime('%d.%m.%Y %H:%M:%S.%f')[:23]} GMT{o}" for s, o in zip(local, off)]
    jf = tmp_path / "local.csv"
    pd.DataFrame({"Local time": times, "Ask": ticks["ask"], "Bid": ticks["bid"], "AskVolume": 1,
                  "BidVolume": 1}).to_csv(jf, index=False)
    code, _, err = _run(capsys, "--input", jf, "--out", tmp_path / "o")
    assert code == 0, err
    _assert_bars_equal(_train(tmp_path / "o"), _ref_bars(ticks))
    assert "own UTC offset" in _manifest(tmp_path / "o")["inputs"][0]["time_zone_used"]
    # without the suffix a 'Local time' export needs --tz
    pd.DataFrame({"Local time": [x[:23] for x in times], "Ask": ticks["ask"], "Bid": ticks["bid"]}).to_csv(
        tmp_path / "plain.csv", index=False)
    code, _, err = _run(capsys, "--input", tmp_path / "plain.csv", "--out", tmp_path / "p")
    assert code == 2 and "--tz" in err and "Local time" in err
    code, _, err = _run(capsys, "--input", tmp_path / "plain.csv", "--out", tmp_path / "p", "--tz", "Europe/Athens")
    assert code == 0, err
    _assert_bars_equal(_train(tmp_path / "p"), _ref_bars(ticks))


def test_generic_csv_epoch_seconds_and_volume(tmp_path, capsys):
    ticks = _ticks(seed=6, whole_seconds=True)
    vol = np.random.default_rng(6).integers(1, 9, len(ticks)) / 4
    src = tmp_path / "gold.csv"
    pd.DataFrame({"When": ticks["t"] // NS, "BidPx": ticks["bid"], "AskPx": ticks["ask"], "Size": vol}).to_csv(
        src, sep=";", index=False)
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "o", "--time-col", "When", "--bid-col",
                          "BidPx", "--ask-col", "AskPx", "--vol-col", "Size", "--tz", "UTC")
    assert code == 0, err
    got = _train(tmp_path / "o")
    _assert_bars_equal(got, _ref_bars(ticks))
    ref_vol = pd.Series(vol).groupby((ticks["t"] // (3600 * NS)) * 3600).sum().to_numpy()
    assert np.allclose(got["volume_sum"], ref_vol, atol=1e-9)
    m = _manifest(tmp_path / "o")
    assert m["inputs"][0]["format"] == "generic" and m["inputs"][0]["epoch_unit"] == "s"
    assert m["inputs"][0]["separator"] == "semicolon"
    # epoch numbers carry no offset: --tz is required
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "x", "--time-col", "When", "--bid-col",
                        "BidPx", "--ask-col", "AskPx")
    assert code == 2 and "--tz" in err
    # a column name that is not in the file
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "x", "--time-col", "Nope", "--bid-col",
                        "BidPx", "--ask-col", "AskPx", "--tz", "UTC")
    assert code == 2 and "Nope" in err
    assert not (tmp_path / "x").exists()


def test_generic_time_format_and_parquet_input(tmp_path, capsys):
    ticks = _ticks(seed=7)
    tt = pd.to_datetime(ticks["t"], utc=True)
    src = tmp_path / "custom.txt"
    pd.DataFrame({"stamp": tt.dt.tz_convert("Asia/Tokyo").dt.strftime("%d/%m/%Y %H:%M:%S.%f"),
                  "b": ticks["bid"], "a": ticks["ask"]}).to_csv(src, sep="\t", index=False)
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "f", "--format", "generic", "--time-col", "stamp",
                        "--bid-col", "b", "--ask-col", "a", "--time-format", "%d/%m/%Y %H:%M:%S.%f",
                        "--tz", "Asia/Tokyo")
    assert code == 0, err
    ref = _ref_bars(ticks)
    _assert_bars_equal(_train(tmp_path / "f"), ref)
    # Parquet with a time-zone-aware datetime column: no --tz needed
    pq = tmp_path / "ticks.parquet"
    pd.DataFrame({"ts": tt, "bid": ticks["bid"], "ask": ticks["ask"]}).to_parquet(pq, index=False)
    code, _, err = _run(capsys, "--input", pq, "--out", tmp_path / "p", "--time-col", "ts", "--bid-col", "bid",
                        "--ask-col", "ask", "--chunksize", 50)
    assert code == 0, err
    _assert_bars_equal(_train(tmp_path / "p"), ref)
    assert _manifest(tmp_path / "p")["inputs"][0]["encoding"] == "parquet"
    # ISO strings with an offset are converted without --tz
    iso = tmp_path / "iso.csv"
    pd.DataFrame({"time": tt.dt.tz_convert("America/New_York").map(lambda x: x.isoformat()), "bid": ticks["bid"],
                  "ask": ticks["ask"]}).to_csv(iso, index=False)
    code, _, err = _run(capsys, "--input", iso, "--out", tmp_path / "i", "--time-col", "time", "--bid-col", "bid",
                        "--ask-col", "ask")
    assert code == 0, err
    _assert_bars_equal(_train(tmp_path / "i"), ref)


def test_format_detection_refusals(tmp_path, capsys):
    ticks = _ticks(hours=10, seed=8)
    odd = tmp_path / "odd.csv"
    pd.DataFrame({"x": ticks["t"], "y": ticks["bid"], "z": ticks["ask"]}).to_csv(odd, index=False)
    code, _, err = _run(capsys, "--input", odd, "--out", tmp_path / "o")
    assert code == 2 and "--format" in err and "--time-col" in err
    bars = tmp_path / "bars.csv"
    pd.DataFrame({"<DATE>": ["2024.01.02"], "<TIME>": ["01:00"], "<OPEN>": [1.0], "<HIGH>": [1.0], "<LOW>": [1.0],
                  "<CLOSE>": [1.0], "<TICKVOL>": [5]}).to_csv(bars, sep="\t", index=False)
    code, _, err = _run(capsys, "--input", bars, "--out", tmp_path / "o")
    assert code == 2 and "BARS" in err and "Export ticks" in err
    amb = tmp_path / "amb.csv"
    pd.DataFrame({"Date": ["2024.01.02"], "Time": ["01:00:00"], "Bid": [1.0], "Ask": [1.1]}).to_csv(amb, index=False)
    code, _, err = _run(capsys, "--input", amb, "--out", tmp_path / "o")
    assert code == 2 and "ambiguous" in err and "--format mt5" in err
    mt5 = _write_mt5(tmp_path / "m.csv", ticks)
    code, _, err = _run(capsys, "--input", mt5, "--out", tmp_path / "o")
    assert code == 2 and "ny+7" in err and "server time" in err.lower()
    duka = _write_duka(tmp_path / "d.csv", ticks)
    code, _, err = _run(capsys, "--input", mt5, "--input", duka, "--out", tmp_path / "o", "--tz", "ny+7")
    assert code == 2 and "different formats" in err
    code, _, err = _run(capsys, "--input", duka, "--out", tmp_path / "o", "--format", "mt5", "--tz", "UTC")
    assert code == 2 and "not a mt5 file" in err
    code, _, err = _run(capsys, "--input", duka, "--out", tmp_path / "o", "--tz", "Mars/Olympus")
    assert code == 2 and "time zone" in err
    assert not (tmp_path / "o").exists()


# ---------------------------------------------------------------------------------------
# time zones

def test_ny7_fixed_offset_and_iana_zones():
    local = pd.to_datetime(["2024-01-15 12:00", "2024-07-15 12:00"]).to_numpy(dtype="datetime64[ns]").view("int64")

    def hours(tz):
        utc = tb.to_utc_ns(local, tb.parse_tz(tz))
        return [str(pd.Timestamp(int(x), unit="ns").strftime("%H:%M")) for x in utc]

    assert hours("ny+7") == ["10:00", "09:00"]          # UTC+2 in January, UTC+3 in July
    assert hours("+02:00") == ["10:00", "10:00"]
    assert hours("-05:00") == ["17:00", "17:00"]
    assert hours("Europe/Athens") == ["10:00", "09:00"]
    assert hours("Asia/Tokyo") == ["03:00", "03:00"]
    assert hours("UTC") == ["12:00", "12:00"]
    # New York 02:30 on 2024-03-10 does not exist and 01:30 on 2024-11-03 happens twice: NaT
    gaps = pd.to_datetime(["2024-03-10 09:30", "2024-11-03 08:30", "2024-11-04 08:30"]).to_numpy(
        dtype="datetime64[ns]").view("int64")
    out = tb.to_utc_ns(gaps, tb.parse_tz("ny+7"))
    assert out[0] == tb.NAT and out[1] == tb.NAT and out[2] != tb.NAT
    assert tb.parse_tz("ny+7").describe().startswith("ny+7")
    for bad in ("+15:00", "ny+30", "Nowhere/City"):
        with pytest.raises(tb.UsageError):
            tb.parse_tz(bad)


def test_dst_nat_rows_are_counted_and_dropped(tmp_path, capsys):
    ticks = _ticks("2024-03-08 00:00", 30, seed=9)
    src = _write_mt5(tmp_path / "m.csv", ticks, extra_rows=False)
    text = src.read_bytes().decode("utf-16")
    text += "2024.03.10\t09:30:00.000\t2000.00\t2000.30\t\t\t6\r\n"      # New York 02:30: does not exist
    src.write_bytes(text.encode("utf-16"))
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "o", "--tz", "ny+7")
    assert code == 0, err
    m = _manifest(tmp_path / "o")
    assert m["rows"]["dst_nat_rows"] == 1
    assert m["rows"]["dropped"]["dst_nonexistent_or_ambiguous_time"] == 1
    _assert_bars_equal(_train(tmp_path / "o"), _ref_bars(ticks))


# ---------------------------------------------------------------------------------------
# bars

def test_bars_match_reference_for_each_price_and_timeframe(tmp_path, capsys):
    ticks = _ticks(seed=10)
    src = _write_duka(tmp_path / "t.csv", ticks)
    for price, tf, tf_s in (("bid", "M15", 900), ("ask", "H4", 14400), ("mid", "D1", 86400)):
        out = tmp_path / f"o_{price}"
        code, text, err = _run(capsys, "--input", src, "--out", out, "--price", price, "--tf", tf)
        assert code == 0, err
        _assert_bars_equal(_train(out, f"XAUUSD_{tf}.parquet"), _ref_bars(ticks, tf_s, price))
        assert ("day boundary" in text) == (tf == "D1")


def test_chunk_size_does_not_change_the_output(tmp_path, capsys):
    ticks = _ticks("2025-09-20", 400, seed=11, repeat=0.3)
    src = _write_mt5(tmp_path / "m.csv", ticks)
    shas, trains = [], []
    for cs in (7, 333, 10_000_000):
        out = tmp_path / f"o{cs}"
        code, _, err = _run(capsys, "--input", src, "--out", out, "--tz", "ny+7", "--chunksize", cs)
        assert code == 0, err
        trains.append((out / "train" / NAME).read_bytes())
        m = _manifest(out)
        shas.append((m["train"]["sha256"], m["locked"]["sha256"], m["locked"]["bars"]))
    assert trains[0] == trains[1] == trains[2]
    assert shas[0] == shas[1] == shas[2] and shas[0][2] > 0


def test_out_of_order_ticks_across_files_and_ties(tmp_path, capsys):
    ticks = _ticks(seed=12)
    rng = np.random.default_rng(12)
    late, early = ticks.iloc[len(ticks) // 2:], ticks.iloc[: len(ticks) // 2]
    mixed = early.iloc[rng.permutation(len(early))]                       # out of order inside a file too
    a = _write_duka(tmp_path / "a.csv", late)
    b = _write_duka(tmp_path / "b.csv", mixed)
    # a bar holding two ticks at the same instant, one per file: input order decides open and close
    t_tie = pd.Timestamp("2024-03-01 05:00:00.250", tz="UTC").value
    tie_a = pd.DataFrame({"t": [t_tie], "bid": [2100.00], "ask": [2100.30]})
    tie_b = pd.DataFrame({"t": [t_tie], "bid": [2100.50], "ask": [2100.70]})
    _write_duka(tmp_path / "c.csv", tie_a)
    _write_duka(tmp_path / "d.csv", tie_b)
    order = pd.concat([late, mixed, tie_a, tie_b], ignore_index=True)
    ref = _ref_bars(order)
    files = ("--input", a, "--input", b, "--input", tmp_path / "c.csv", "--input", tmp_path / "d.csv")
    code, _, err = _run(capsys, *files, "--out", tmp_path / "refused")       # c and d overlap (same instant)
    assert code == 3 and "overlap" in err and not (tmp_path / "refused").exists()
    outs = []
    for cs in (5, 1_000_000):
        out = tmp_path / f"o{cs}"
        code, _, err = _run(capsys, *files, "--out", out, "--chunksize", cs, "--overlap", "keep")
        assert code == 0, err
        got = _train(out)
        _assert_bars_equal(got, ref)
        outs.append((out / "train" / NAME).read_bytes())
    assert outs[0] == outs[1]
    row = got[got["time"] == t_tie // NS // 3600 * 3600].iloc[0]
    assert row["open"] == 2100.00 and row["close"] == 2100.50 and row["tick_volume"] == 2
    assert row["spread"] == pytest.approx(0.20)


def test_cleaning_counts_and_spikes_flagged(tmp_path, capsys):
    ticks = _ticks(seed=13)
    spike_at = 1000
    b0 = ticks.loc[spike_at, "bid"]
    ticks.loc[spike_at, ["bid", "ask"]] = np.round([b0 + 80.0, b0 + 80.3], 2)     # short decimals, as in real files
    good = ticks.copy()
    rows = _write_duka(tmp_path / "t.csv", ticks).read_text().splitlines()
    t0 = int(ticks["t"].iloc[50] // 1_000_000)
    rows[60:60] = [f"{t0},2000.00,2000.40,1,1",          # crossed: ask < bid
                   f"{t0},2000.40,2000.00,1,1",          # an exact duplicate of the next row: kept
                   f"{t0},2000.40,2000.00,1,1",
                   f"{t0},,2000.00,1,1",                 # missing ask
                   f"{t0},2000.40,0,1,1",                # bid 0
                   "abc,2000.40,2000.00,1,1"]           # unreadable time
    (tmp_path / "t.csv").write_text("\n".join(rows) + "\n")
    good = pd.concat([good.iloc[:59], pd.DataFrame({"t": [t0 * 1_000_000] * 2, "bid": [2000.0] * 2,
                                                    "ask": [2000.4] * 2}), good.iloc[59:]], ignore_index=True)
    code, out, err = _run(capsys, "--input", tmp_path / "t.csv", "--out", tmp_path / "o")
    assert code == 0, err
    got = _train(tmp_path / "o")
    _assert_bars_equal(got, _ref_bars(good))
    m = _manifest(tmp_path / "o")
    d = m["rows"]["dropped"]
    assert d["crossed_quote"] == 1 and d["missing_price"] == 1 and d["nonpositive_price"] == 1
    assert d["unparsable_time"] == 1 and m["rows"]["dropped_total"] == 4
    assert m["rows"]["exact_duplicate_rows_within_chunk"] == 1
    assert "crossed_quote" in out
    top = m["train"]["largest_ranges"][0]
    spike_bar = int(ticks["t"].iloc[spike_at] // NS // 3600 * 3600)
    assert top["time_utc"] == tb._iso_s(spike_bar) and top["ratio_to_median_range"] > 10
    assert len(m["train"]["largest_ranges"]) == 10
    assert got["high"].max() > 2050                       # flagged, not removed


def test_gaps_hours_and_years(tmp_path, capsys):
    ticks = pd.concat([_ticks("2023-12-25", 200, seed=14), _ticks("2024-01-06", 300, seed=15)], ignore_index=True)
    code, _, err = _run(capsys, "--input", _write_duka(tmp_path / "t.csv", ticks), "--out", tmp_path / "o")
    assert code == 0, err
    tr = _manifest(tmp_path / "o")["train"]
    assert tr["bars_per_year"] == {"2023": 168, "2024": 332}
    assert tr["gaps"]["over_1_bar"] == 1 and tr["gaps"]["over_72h"] == 1
    gap = tr["gaps"]["over_72h_longest"][0]
    assert gap["start_utc"] == "2024-01-02T07:00:00Z" and gap["end_utc"] == "2024-01-06T00:00:00Z"
    assert gap["hours_between_bar_opens"] == 89
    assert len(tr["median_spread_by_utc_hour"]["spread"]) == 24
    assert tr["ticks_per_bar"]["bars_with_fewer_than_5_ticks"] > 0
    assert tr["first_bar_utc"] == "2023-12-25T00:00:00Z"


# ---------------------------------------------------------------------------------------
# holdout split

def test_holdout_split_at_cutoff_and_no_locked_statistics(tmp_path, capsys):
    before = _ticks("2025-09-22", 6 * 24, seed=16)
    after = _ticks("2025-09-28", 3 * 24, seed=17, level=7777.0)            # a price level easy to spot
    exact = pd.DataFrame({"t": [CUTOFF_S * NS], "bid": [7777.11], "ask": [7777.44]})
    ticks = pd.concat([before, exact, after], ignore_index=True)
    src = _write_duka(tmp_path / "t.csv", ticks)
    code, out, err = _run(capsys, "--input", src, "--out", tmp_path / "o")
    assert code == 0, err
    train = _train(tmp_path / "o")
    assert train["time"].max() < CUTOFF_S and train["time"].max() == CUTOFF_S - 3600
    ref = _ref_bars(ticks, exact_mean=True)
    ref_locked = ref[ref["time"] >= CUTOFF_S].reset_index(drop=True)
    assert ref_locked["time"].iloc[0] == CUTOFF_S and ref_locked["open"].iloc[0] == 7777.11
    _assert_bars_equal(train, ref[ref["time"] < CUTOFF_S].reset_index(drop=True))
    m = _manifest(tmp_path / "o")
    assert set(m["locked"]) == {"file", "bars", "sha256"}
    assert m["locked"]["bars"] == len(ref_locked)
    assert m["locked"]["sha256"] == hashlib.sha256(tb._parquet_bytes(ref_locked)).hexdigest()
    assert m["locked"]["file"].endswith("locked_holdout/XAUUSD_H1.parquet.locked".replace("/", tb.os.sep))
    assert (tmp_path / "o" / "locked_holdout" / "XAUUSD_H1.parquet.locked").is_file()
    assert m["rows"]["read"] == len(before) and m["rows"]["kept"] == len(before)
    assert m["train"]["last_bar_utc"] == "2025-09-27T23:00:00Z" and m["settings"]["cutoff_utc"] == CUTOFF

    def walk(x, key=""):
        if isinstance(x, dict):
            for k, v in x.items():
                yield from walk(v, k)
        elif isinstance(x, list):
            for v in x:
                yield from walk(v, key)
        else:
            yield key, x

    for key, v in walk({k: v for k, v in m.items() if k != "created_utc"}):
        if isinstance(v, float):
            assert not 7000 < v < 8500, (key, v)
        if isinstance(v, str) and key != "cutoff_utc" and re.match(r"\d{4}-\d{2}-\d{2}", v):
            assert v[:10] < "2025-09-28", (key, v)
    assert "7777." not in out and not re.search(r"2025-09-(28|29|30)|2025-10", out.replace(CUTOFF, ""))


def test_cutoff_rules_and_no_train_bars(tmp_path, capsys):
    ticks = _ticks("2025-09-25", 6 * 24, seed=18)
    src = _write_duka(tmp_path / "t.csv", ticks)
    assert tb.DEFAULT_CUTOFF == CUTOFF and tb.DEFAULT_CUTOFF_S == CUTOFF_S == tb.parse_cutoff(CUTOFF)
    # any cutoff after the default is refused without the flag, the old 2025-10-01 one included
    for late in ("2025-09-28T01:00:00Z", "2025-10-01T00:00:00Z", "2025-11-01"):
        code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "late", "--cutoff", late)
        assert code == 2 and "--allow-later-cutoff" in err and "locked holdout" in err and "2025-09-28" in err
        assert "2025-10-01 on" not in err
    assert not (tmp_path / "late").exists()
    code, out, err = _run(capsys, "--input", src, "--out", tmp_path / "late", "--cutoff", "2025-09-29",
                          "--allow-later-cutoff")
    assert code == 0, err
    assert _train(tmp_path / "late")["time"].max() == CUTOFF_S + 86400 - 3600
    # the train file now overlaps the locked holdout period: said loudly and stored in the manifest
    m = _manifest(tmp_path / "late")
    assert m["settings"]["train_overlaps_default_locked_period"] is True
    assert any("INCLUDES data from the period of your locked holdout" in w for w in m["warnings"])
    assert "WARNING: !!! --cutoff 2025-09-29T00:00:00Z is later than 2025-09-28T00:00:00Z" in out
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "default")
    assert code == 0, err
    assert _manifest(tmp_path / "default")["settings"]["train_overlaps_default_locked_period"] is False
    assert not any("locked holdout" in w for w in _manifest(tmp_path / "default")["warnings"])
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "early", "--cutoff", "2025-09-27T00:00:00Z")
    assert code == 0, err
    assert _train(tmp_path / "early")["time"].max() == CUTOFF_S - 86400 - 3600
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "odd", "--cutoff", "2025-09-27T00:30:00Z")
    assert code == 2 and "bar boundary" in err
    after = _write_duka(tmp_path / "after.csv", _ticks("2025-09-29", 48, seed=19))
    code, _, err = _run(capsys, "--input", after, "--out", tmp_path / "none")
    assert code == 3 and "no bars fall before the cutoff" in err
    assert not (tmp_path / "none").exists()


# ---------------------------------------------------------------------------------------
# safety

def test_refuses_locked_paths_without_reading(tmp_path, capsys, monkeypatch):
    src = _write_duka(tmp_path / "t.csv", _ticks(hours=20, seed=20))
    hold = tmp_path / "locked_holdout"
    hold.mkdir()
    _write_duka(hold / "t.csv", _ticks(hours=20, seed=21))
    (tmp_path / "XAUUSD_H1.parquet.locked").write_bytes(b"not to be read")
    ref = tmp_path / "ref"
    ref.mkdir()

    def no_read(*_a, **_k):
        raise AssertionError("a locked path was read")

    monkeypatch.setattr(tb, "sniff", no_read)
    monkeypatch.setattr(tb.pd, "read_csv", no_read)
    monkeypatch.setattr(tb.pd, "read_parquet", no_read)
    cases = [
        ("--input", hold / "t.csv", "--out", tmp_path / "o"),
        ("--input", hold, "--out", tmp_path / "o"),
        ("--input", str(hold / "*.csv"), "--out", tmp_path / "o"),
        ("--input", tmp_path / "XAUUSD_H1.parquet.locked", "--out", tmp_path / "o"),
        ("--input", tmp_path / "LOCKED_HOLDOUT" / "t.csv", "--out", tmp_path / "o"),
        ("--input", src, "--out", tmp_path / "o", "--compare-with", tmp_path / "XAUUSD_H1.parquet.locked"),
        ("--input", src, "--out", tmp_path / "o", "--compare-with", hold / "XAUUSD_H1.parquet"),
        ("--input", src, "--out", hold / "out"),
        ("--input", src, "--out", hold),
    ]
    for args in cases:
        code, _, err = _run(capsys, *args)
        assert code == 2, args
        assert "locked" in err, err
    assert not (tmp_path / "o").exists() and not (hold / "out").exists()
    assert sorted(p.name for p in hold.iterdir()) == ["t.csv"]


def test_overwrite_only_replaces_own_outputs(tmp_path, capsys):
    a = _write_duka(tmp_path / "a.csv", _ticks("2025-09-20", 300, seed=22))
    b = _write_duka(tmp_path / "b.csv", _ticks("2025-09-20", 300, seed=23))
    out = tmp_path / "o"
    assert _run(capsys, "--input", a, "--out", out)[0] == 0
    first = (out / "train" / NAME).read_bytes()
    first_manifest = (out / MANIFEST).read_bytes()
    code, _, err = _run(capsys, "--input", b, "--out", out)
    assert code == 2 and "--overwrite" in err
    assert (out / "train" / NAME).read_bytes() == first and (out / MANIFEST).read_bytes() == first_manifest
    code, _, err = _run(capsys, "--input", b, "--out", out, "--overwrite")
    assert code == 0, err
    assert (out / "train" / NAME).read_bytes() != first
    m = _manifest(out)
    assert m["train"]["sha256"] == hashlib.sha256((out / "train" / NAME).read_bytes()).hexdigest()
    assert sorted(p.name for p in (out / "train").iterdir()) == [NAME]
    assert sorted(p.name for p in (out / "locked_holdout").iterdir()) == [NAME + ".locked"]
    # files made by something else (e.g. the existing Dukascopy converter) are never replaced
    other = tmp_path / "data"
    (other / "train").mkdir(parents=True)
    (other / "train" / NAME).write_bytes(first)
    (other / MANIFEST).write_text(json.dumps({"source": "Dukascopy via dukascopy-node"}), encoding="utf-8")
    code, _, err = _run(capsys, "--input", b, "--out", other, "--overwrite")
    assert code == 2 and "not made by this tool" in err
    assert (other / "train" / NAME).read_bytes() == first
    assert sorted(p.name for p in other.iterdir()) == [MANIFEST, "train"]
    # a train file edited by hand no longer matches the manifest: refused too
    (out / "train" / NAME).write_bytes(first)
    code, _, err = _run(capsys, "--input", a, "--out", out, "--overwrite")
    assert code == 2 and "not made by this tool" in err


def test_failed_write_leaves_old_outputs(tmp_path, capsys, monkeypatch):
    a = _write_duka(tmp_path / "a.csv", _ticks("2025-09-20", 300, seed=24))
    b = _write_duka(tmp_path / "b.csv", _ticks("2025-09-20", 300, seed=25))
    out = tmp_path / "o"
    assert _run(capsys, "--input", a, "--out", out)[0] == 0
    before = {p.name: p.read_bytes() for p in [out / "train" / NAME, out / MANIFEST]}
    real = tb.os.replace

    def flaky(src, dst):                    # the new manifest cannot be put in place
        if ".tmp-" in str(src) and str(dst).endswith(MANIFEST):
            raise PermissionError(13, "the file is open in another program", str(dst))
        return real(src, dst)

    monkeypatch.setattr(tb.os, "replace", flaky)
    code, _, err = _run(capsys, "--input", b, "--out", out, "--overwrite")
    monkeypatch.setattr(tb.os, "replace", real)
    assert code == 3 and "No output files were written" in err and "earlier run" in err
    assert {p.name: p.read_bytes() for p in [out / "train" / NAME, out / MANIFEST]} == before
    assert sorted(p.name for p in (out / "train").iterdir()) == [NAME]
    assert sorted(p.name for p in (out / "locked_holdout").iterdir()) == [NAME + ".locked"]
    assert sorted(p.name for p in out.iterdir()) == [MANIFEST, "locked_holdout", "train"]


def test_dry_run_writes_nothing(tmp_path, capsys):
    ticks = _ticks("2025-09-26", 72, seed=26)
    src = _write_mt5(tmp_path / "m.csv", ticks, extra_rows=False)
    out = tmp_path / "o"
    code, text, err = _run(capsys, "--input", src, "--out", out, "--dry-run")
    assert code == 0, err
    assert "DRY RUN" in text and "mt5" in text and "utf-16" in text and "needs --tz" in text.lower()
    code, text, err = _run(capsys, "--input", src, "--out", out, "--dry-run", "--tz", "ny+7")
    assert code == 0, err
    assert "UTC time" in text and "would make" in text and "nothing was written" in text
    assert "at or after the cutoff" in text                 # rows from 2025-09-28 00:00 UTC on are hidden
    srv = _server_time(ticks["t"].iloc[:1])[0]
    assert f"{srv:%Y.%m.%d %H:%M}" in text                   # raw broker time and its UTC side by side
    assert f"{pd.Timestamp(int(ticks['t'].iloc[0]), unit='ns'):%Y-%m-%d %H:%M}" in text
    assert not out.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.csv"]


# ---------------------------------------------------------------------------------------
# comparison with a reference file

def test_compare_detects_time_shift(tmp_path, capsys):
    ticks = _ticks("2024-02-05", 450, seed=27)
    ref_dir = tmp_path / "ref"
    code, _, err = _run(capsys, "--input", _write_duka(tmp_path / "t.csv", ticks), "--out", ref_dir)
    assert code == 0, err
    ref = ref_dir / "train" / NAME
    # aligned data: best lag 0, no warning
    code, out, err = _run(capsys, "--input", tmp_path / "t.csv", "--out", tmp_path / "same", "--compare-with", ref)
    assert code == 0, err
    c = _manifest(tmp_path / "same")["compare"]
    assert c["best_lag_bars"] == 0 and c["overlapping_bars"] == len(_train(ref_dir))
    assert c["return_corr_by_lag"]["+0"] > 0.99 and c["abs_close_diff_median"] == 0
    assert not c["warnings"] and "TIME ZONE CHECK FAILED" not in out
    # the same ticks written in UTC+2 but read as UTC: the bars are 2 h late -> best lag +2
    shifted = ticks.assign(t=ticks["t"] + 2 * 3600 * NS)
    src = _write_duka(tmp_path / "s.csv", shifted)
    code, out, err = _run(capsys, "--input", src, "--out", tmp_path / "shift", "--compare-with", ref)
    assert code == 0, err
    c = _manifest(tmp_path / "shift")["compare"]
    assert c["best_lag_bars"] == 2 and c["best_lag_hours"] == 2
    assert "TIME ZONE CHECK FAILED" in out and "+02:00" in out and "ny+7" in out
    # and 2 h early -> best lag -2
    src = _write_duka(tmp_path / "e.csv", ticks.assign(t=ticks["t"] - 2 * 3600 * NS))
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "early", "--compare-with", ref)
    assert code == 0, err
    assert _manifest(tmp_path / "early")["compare"]["best_lag_bars"] == -2
    # refusals: other timeframe, missing columns
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "x", "--tf", "M15", "--compare-with", ref)
    assert code == 2 and "timeframe" in err
    pd.DataFrame({"time": [1, 2]}).to_parquet(tmp_path / "BAD_H1.parquet")
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "x", "--compare-with", tmp_path / "BAD_H1.parquet")
    assert code == 2 and "close" in err


def test_compare_flags_wrong_daylight_saving_rule(tmp_path, capsys):
    ticks = pd.concat([_ticks("2024-01-08", 260, seed=28), _ticks("2024-07-08", 260, seed=29)], ignore_index=True)
    ref_dir = tmp_path / "ref"
    assert _run(capsys, "--input", _write_duka(tmp_path / "t.csv", ticks), "--out", ref_dir)[0] == 0
    src = _write_mt5(tmp_path / "m.csv", ticks)              # broker on New York close time (ny+7)
    code, out, err = _run(capsys, "--input", src, "--out", tmp_path / "fixed", "--tz", "+02:00",
                          "--compare-with", ref_dir / "train" / NAME)
    assert code == 0, err
    reg = _manifest(tmp_path / "fixed")["compare"]["best_lag_by_dst_regime"]
    assert reg["both_standard_time"]["best_lag_bars"] == 0 and reg["both_daylight_saving"]["best_lag_bars"] == 1
    assert reg["us_dst_only"]["best_lag_bars"] is None                       # no March / late-October data here
    assert "daylight-saving rule" in out and "Try --tz ny+7" in out
    code, out, err = _run(capsys, "--input", src, "--out", tmp_path / "ny", "--tz", "ny+7",
                          "--compare-with", ref_dir / "train" / NAME)
    assert code == 0, err
    c = _manifest(tmp_path / "ny")["compare"]
    assert c["best_lag_bars"] == 0 and not c["warnings"]


# ---------------------------------------------------------------------------------------
# AlphaMaster compatibility, naming, inputs, priority

def test_train_file_loads_in_alphamaster(tmp_path, capsys):
    pytest.importorskip("torch")
    from data_pipeline.parquet_manager import ParquetDataManager, inspect_parquet_file, parse_parquet_filename

    src = _write_duka(tmp_path / "t.csv", _ticks(hours=430, seed=30))
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "o")
    assert code == 0, err
    path = tmp_path / "o" / "train" / NAME
    assert parse_parquet_filename(path) == ("XAUUSD", "H1")
    info = inspect_parquet_file(path)
    assert info["bars"] == 430 and info["symbol"] == "XAUUSD" and info["timeframe"] == "H1"
    dm = ParquetDataManager(path)
    dm.load()
    assert dm.symbols == ["XAUUSD"]
    raw = dm._raw_dict
    assert raw["open"].shape == (1, 430) and int(raw["time"][0, 0]) == int(pd.read_parquet(path)["time"].iloc[0])


def test_file_names_follow_symbol_and_timeframe(tmp_path, capsys):
    src = _write_duka(tmp_path / "t.csv", _ticks(hours=40, seed=31))
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "o", "--symbol", "GOLD", "--tf", "m15")
    assert code == 0, err
    assert (tmp_path / "o" / "train" / "GOLD_M15.parquet").is_file()
    assert (tmp_path / "o" / "GOLD_M15_manifest.json").is_file()
    assert not (tmp_path / "o" / "locked_holdout").exists()      # no bars at or after the cutoff
    m = _manifest(tmp_path / "o", "GOLD_M15_manifest.json")
    assert m["locked"] == {"file": None, "bars": 0, "sha256": None}
    assert m["tool"] == tb.TOOL and m["settings"]["timeframe"] == "M15"
    assert "ticks_to_bars.py" in m["command_line"] and "--symbol GOLD" in m["command_line"]
    assert m["inputs"][0]["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    for bad in ("../x", "A B", "locked_holdout"):
        code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "x", "--symbol", bad)
        assert code == 2 and "--symbol" in err
    assert _run(capsys, "--input", src, "--out", tmp_path / "x", "--tf", "W1")[0] == 2


def test_folder_glob_and_gzip_inputs(tmp_path, capsys):
    ticks = _ticks(hours=200, seed=32)
    folder = tmp_path / "ticks"
    folder.mkdir()
    parts = np.array_split(np.arange(len(ticks)), 3)
    _write_duka(folder / "xau-1.csv", ticks.iloc[parts[0]])
    with gzip.open(folder / "xau-2.csv.gz", "wt") as f:
        ticks_part = ticks.iloc[parts[1]]
        pd.DataFrame({"timestamp": ticks_part["t"] // 1_000_000, "askPrice": ticks_part["ask"],
                      "bidPrice": ticks_part["bid"]}).to_csv(f, index=False)
    _write_duka(folder / "xau-3.txt", ticks.iloc[parts[2]])
    (folder / "README.md").write_text("not ticks")
    ref = _ref_bars(ticks)
    code, out, err = _run(capsys, "--input", folder, "--out", tmp_path / "f")
    assert code == 0, err
    _assert_bars_equal(_train(tmp_path / "f"), ref)
    assert [Path(i["path"]).name for i in _manifest(tmp_path / "f")["inputs"]] == ["xau-1.csv", "xau-2.csv.gz",
                                                                                  "xau-3.txt"]
    code, out, err = _run(capsys, "--input", str(folder / "xau-*"), "--input", folder / "xau-1.csv",
                          "--out", tmp_path / "g")
    assert code == 0, err
    assert "given more than once" in out and "skipped" not in out
    _assert_bars_equal(_train(tmp_path / "g"), ref)
    code, _, err = _run(capsys, "--input", str(tmp_path / "nothing-*.csv"), "--out", tmp_path / "h")
    assert code == 2 and "no tick files" in err


def test_priority_is_lowered_by_default(tmp_path, capsys, monkeypatch):
    src = _write_duka(tmp_path / "t.csv", _ticks(hours=10, seed=33))
    if os.name == "nt":                     # the real SetPriorityClass call (harmless for the test run)
        code, out, _ = _run(capsys, "--input", src, "--out", tmp_path / "a", "--dry-run", priority=True)
        assert code == 0 and "BELOW_NORMAL" in out
    else:
        calls = []
        monkeypatch.setattr(tb.os, "nice", lambda inc: calls.append(inc) or 10)
        code, out, _ = _run(capsys, "--input", src, "--out", tmp_path / "a", "--dry-run", priority=True)
        assert code == 0 and calls == [10] and "lower priority" in out

        def refuse(_inc):
            raise PermissionError("not allowed")

        monkeypatch.setattr(tb.os, "nice", refuse)
        code, out, _ = _run(capsys, "--input", src, "--out", tmp_path / "b", "--dry-run", priority=True)
        assert code == 0 and "WARNING: could not lower the process priority" in out
        monkeypatch.setattr(tb.os, "nice", lambda inc: calls.append(inc) or 10)
        calls.clear()
    code, out, _ = _run(capsys, "--input", src, "--out", tmp_path / "c", "--dry-run")
    assert code == 0 and not any(line.startswith(("priority", "WARNING: could not lower")) for line in out.splitlines())
    if os.name != "nt":
        assert calls == []


def test_help_and_usage_errors(tmp_path, capsys):
    code, out, _ = _run(capsys, "--help")
    assert code == 0 and "ny+7" in out and "PowerShell" in out and "--compare-with" in out
    assert _run(capsys, "--out", tmp_path / "o")[0] == 2                       # no --input
    code, _, err = _run(capsys, "--input", tmp_path / "missing.csv", "--out", tmp_path / "o")
    assert code == 2 and "not found" in err
    src = _write_duka(tmp_path / "t.csv", _ticks(hours=10, seed=34))
    assert _run(capsys, "--input", src, "--out", tmp_path / "o", "--chunksize", 0)[0] == 2
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "o", "--time-format", "%Y", "--format", "mt5")
    assert code == 2 and "generic" in err
    assert not (tmp_path / "o").exists()


# ---------------------------------------------------------------------------------------
# the default cutoff (owner decision: the locked holdout starts 2025-09-28 00:00 UTC)

def _boundary_case(tmp_path, fmt: str, t_ns: int, price: float):
    """Train-period ticks up to the last train bar plus one boundary tick; returns (ticks, run args)."""
    ticks = _ticks("2025-09-15", 13 * 24, seed=60)                          # bars up to 2025-09-27 23:00
    edge = pd.DataFrame({"t": [t_ns], "bid": [price], "ask": [round(price + 0.33, 2)]})
    allt = pd.concat([ticks, edge], ignore_index=True)
    if fmt == "mt5":
        return allt, ("--input", _write_mt5(tmp_path / "m.csv", allt, extra_rows=False), "--tz", "ny+7")
    return allt, ("--input", _write_duka(tmp_path / "d.csv", allt))


@pytest.mark.parametrize("fmt", ["dukascopy-node", "mt5"])
def test_tick_just_before_the_default_cutoff_is_train(tmp_path, capsys, fmt):
    t_ns = pd.Timestamp("2025-09-27T23:59:59.999Z").value                    # 1 ms before the cutoff
    ticks, args = _boundary_case(tmp_path, fmt, t_ns, 2222.22)
    code, out, err = _run(capsys, *args, "--out", tmp_path / "o")              # no --cutoff: the default
    assert code == 0, err
    train = _train(tmp_path / "o")
    _assert_bars_equal(train, _ref_bars(ticks))
    last = train.iloc[-1]
    assert last["time"] == CUTOFF_S - 3600 and last["close"] == 2222.22 and last["high"] >= 2222.22
    assert ticks["t"].iloc[:-1].max() < ticks["t"].iloc[-1] and ticks["t"].iloc[:-1].max() >= (CUTOFF_S - 3600) * NS
    m = _manifest(tmp_path / "o")
    assert m["settings"]["cutoff_utc"] == CUTOFF and m["train"]["last_bar_utc"] == "2025-09-27T23:00:00Z"
    assert m["locked"] == {"file": None, "bars": 0, "sha256": None}
    assert not (tmp_path / "o" / "locked_holdout").exists()
    assert m["rows"]["kept"] == len(ticks)


@pytest.mark.parametrize("fmt", ["dukascopy-node", "mt5"])
def test_tick_at_the_default_cutoff_is_locked(tmp_path, capsys, fmt):
    ticks, args = _boundary_case(tmp_path, fmt, CUTOFF_S * NS, 7777.77)       # exactly 2025-09-28 00:00:00.000Z
    code, out, err = _run(capsys, *args, "--out", tmp_path / "o")
    assert code == 0, err
    ref = _ref_bars(ticks, exact_mean=True)
    train = _train(tmp_path / "o")
    _assert_bars_equal(train, ref[ref["time"] < CUTOFF_S].reset_index(drop=True))
    assert train["time"].max() == CUTOFF_S - 3600 and train["high"].max() < 7000   # nothing of it in train
    ref_locked = ref[ref["time"] >= CUTOFF_S].reset_index(drop=True)
    assert len(ref_locked) == 1 and ref_locked["time"].iloc[0] == CUTOFF_S and ref_locked["open"].iloc[0] == 7777.77
    m = _manifest(tmp_path / "o")
    assert m["locked"]["bars"] == 1
    assert m["locked"]["sha256"] == hashlib.sha256(tb._parquet_bytes(ref_locked)).hexdigest()
    assert m["rows"]["read"] == len(ticks) - 1                                 # the locked tick is not counted
    assert "7777" not in out.replace(str(tmp_path), "") and "7777" not in json.dumps(m).replace(str(tmp_path), "")


# ---------------------------------------------------------------------------------------
# review findings (one test per finding unless an earlier test already covers it)

def test_git_commit_lookup_never_writes_the_repository(tmp_path, monkeypatch):
    import shutil
    import subprocess
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(tmp_path)}

    def git(*args):
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=repo,
                              check=True, capture_output=True, text=True, env=env)

    git("init", "-q")
    (repo / "f.txt").write_text("x")
    git("add", "f.txt")
    git("commit", "-q", "-m", "x")
    index = repo / ".git" / "index"

    def stale():                             # same content, other mtime: a plain 'git status' rewrites the index
        os.utime(repo / "f.txt", ns=(1_000_000_000, 1_000_000_000 + len(stale.__name__)))
        stale.__name__ += "x"
        return index.stat().st_mtime_ns, index.read_bytes()

    before = stale()
    monkeypatch.setattr(tb, "ROOT", repo)
    calls = []
    real = tb.subprocess.run
    monkeypatch.setattr(tb.subprocess, "run", lambda cmd, *a, **k: calls.append(list(cmd)) or real(cmd, *a, **k))
    commit = tb._git_commit()
    assert commit and commit.startswith(git("rev-parse", "HEAD").stdout.strip())
    assert (index.stat().st_mtime_ns, index.read_bytes()) == before and not (repo / ".git" / "index.lock").exists()
    assert any("--no-optional-locks" in c and "status" in c for c in calls)
    # the check is meaningful: the plain command does write the index in this state
    before = stale()
    real(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True, env=env)
    assert (index.stat().st_mtime_ns, index.read_bytes()) != before


def test_locked_period_rows_are_never_shown_or_counted(tmp_path, capsys):
    train = _ticks("2025-08-01", 400, seed=61)
    locked = _ticks("2026-03-02", 200, seed=62, level=4400.0)
    gap = "2026.03.08\t09:30:00.000\t4400.10\t4400.40\t\t\t6\r\n"          # New York 02:30: a DST gap (NaT)
    only_locked = _write_mt5(tmp_path / "march.csv", locked, extra_rows=False)
    only_locked.write_bytes((only_locked.read_bytes().decode("utf-16") + gap).encode("utf-16"))
    code, out, err = _run(capsys, "--input", only_locked, "--out", tmp_path / "d", "--tz", "ny+7", "--dry-run")
    assert code == 0, err
    out = out.replace(str(tmp_path), "<tmp>")                                 # the temp folder name may hold digits
    assert "2026" not in out and "4400" not in out and "they are not shown and not counted" in out
    assert not re.search(r"\d+ of them|first [\d,]+ rows", out)          # no count of locked-period rows
    both = _write_mt5(tmp_path / "both.csv", pd.concat([train, locked], ignore_index=True), extra_rows=False)
    both.write_bytes((both.read_bytes().decode("utf-16") + gap).encode("utf-16"))
    code, out, err = _run(capsys, "--input", both, "--out", tmp_path / "o", "--tz", "ny+7")
    assert code == 0, err
    out = out.replace(str(tmp_path), "<tmp>")
    m = _manifest(tmp_path / "o")
    assert m["rows"]["read"] == len(train) and m["rows"]["dst_nat_rows"] == 0
    assert "4400" not in out and "2026-03" not in out and "2026-03" not in json.dumps(m)
    # error messages do not echo a value that may be from the locked period
    g = tmp_path / "nov.csv"
    g.write_text("time,bid,ask\n" + "".join(f"2025/11/03 01:00:{i:02d}.123,2000.1,2000.4\n" for i in range(50)))
    code, _, err = _run(capsys, "--input", g, "--out", tmp_path / "g", "--time-col", "time", "--bid-col", "bid",
                        "--ask-col", "ask", "--tz", "UTC")
    assert code == 2 and "2025/11" not in err and "not shown" in err


def test_negative_tz_offset_works_on_the_command_line(tmp_path, capsys):
    ticks = _ticks("2024-01-08", 60, seed=63)
    shifted = ticks.assign(t=ticks["t"] - 5 * 3600 * NS)                     # New York winter wall clock
    src = tmp_path / "ny.csv"
    pd.DataFrame({"t": shifted["t"] // 1_000_000, "b": ticks["bid"], "a": ticks["ask"]}).to_csv(src, index=False)
    cols = ("--time-col", "t", "--bid-col", "b", "--ask-col", "a")
    outs = []
    for k, form in enumerate((["--tz", "-05:00"], ["--tz=-05:00"], ["--tz", "UTC-5"])):
        code, _, err = _run(capsys, "--input", src, "--out", tmp_path / f"o{k}", *cols, *form)
        assert code == 0, (form, err)
        _assert_bars_equal(_train(tmp_path / f"o{k}"), _ref_bars(ticks))
        outs.append((tmp_path / f"o{k}" / "train" / NAME).read_bytes())
    assert outs[0] == outs[1] == outs[2]
    assert "--tz=-05:00" in _manifest(tmp_path / "o0")["command_line"]
    # the tool's own advice is in a form the command line accepts
    assert tb.suggest_tz(tb.parse_tz("UTC"), -2) == "--tz=-02:00"


def test_archives_are_reported_not_skipped_silently(tmp_path, capsys):
    import zipfile
    folder = tmp_path / "monthly"
    folder.mkdir()
    _write_duka(folder / "XAUUSD_2025-05.csv", _ticks("2025-05-05", 30, seed=64))
    inner = _write_duka(tmp_path / "XAUUSD_2025-06.csv", _ticks("2025-06-02", 30, seed=65))
    with zipfile.ZipFile(folder / "XAUUSD_2025-06.zip", "w") as zf:
        zf.write(inner, inner.name)
    _write_duka(folder / "XAUUSD_2025-07.csv", _ticks("2025-07-07", 30, seed=66))
    code, out, err = _run(capsys, "--input", folder, "--out", tmp_path / "o")
    assert code == 0, err
    assert "skipped 1 file(s)" in out and "XAUUSD_2025-06.zip" in out and "Extract" in out
    assert any("NOT converted" in w for w in _manifest(tmp_path / "o")["warnings"])
    code, _, err = _run(capsys, "--input", folder / "XAUUSD_2025-06.zip", "--out", tmp_path / "z")
    assert code == 2 and ".zip archive" in err and "Extract All" in err
    (tmp_path / "renamed.csv.gz").write_bytes((folder / "XAUUSD_2025-06.zip").read_bytes())
    code, _, err = _run(capsys, "--input", tmp_path / "renamed.csv.gz", "--out", tmp_path / "z")
    assert code == 2 and ".zip archive" in err
    assert not (tmp_path / "z").exists()


def test_bar_accumulator_merges_only_what_new_ticks_can_touch(monkeypatch):
    rng = np.random.default_rng(67)
    n = 6000
    t = np.sort(rng.integers(0, 3000 * 60 * NS, n)).astype("int64")
    p = np.round(2000 + rng.normal(0, 1, n), 2)
    spr = np.round(rng.integers(10, 50, n) / 100, 2)
    sq, vq = np.rint(spr / 1e-8).astype("int64"), np.zeros(n, dtype="int64")
    bar = (t // (60 * NS)) * 60                                               # M1: many bars
    seq = np.arange(n, dtype="int64")
    whole = tb.ticks_to_partials(bar, t, seq, p, spr, sq, vq)
    chunks = np.array_split(np.arange(n), 113)
    real = tb.merge_partials
    merged_rows = []
    monkeypatch.setattr(tb, "merge_partials",
                        lambda parts: merged_rows.append(sum(len(x["bar"]) for x in parts)) or real(parts))
    monkeypatch.setattr(tb.BarAccumulator, "FLUSH_ROWS", 40)
    for order in (np.arange(len(chunks)), rng.permutation(len(chunks))):     # in time order, then shuffled
        merged_rows.clear()
        acc = tb.BarAccumulator()
        new_rows = 0
        for i in order:
            c = chunks[i]
            part = tb.ticks_to_partials(bar[c], t[c], seq[c], p[c], spr[c], sq[c], vq[c])
            new_rows += len(part["bar"])
            acc.add(part)
        got = acc.result()
        for k in tb.PART_FIELDS:
            assert np.array_equal(got[k], whole[k]), k
        if order[0] == 0 and order[-1] == len(chunks) - 1:
            # time-ordered input: a flush re-merges at most the one stored bar it continues
            assert sum(merged_rows) <= new_rows + len(merged_rows)
    assert "M1" in tb.CHUNKSIZE_HELP and "150 bytes" in tb.CHUNKSIZE_HELP


def test_refuses_to_write_into_a_foreign_research_folder(tmp_path, capsys):
    src = _write_duka(tmp_path / "t.csv", _ticks("2025-09-01", 300, seed=68))
    rd = tmp_path / "research_data"                       # the existing Dukascopy layout
    (rd / "train").mkdir(parents=True)
    (rd / "locked_holdout").mkdir()
    (rd / "train" / NAME).write_bytes(b"PAR1")
    (rd / "locked_holdout" / (NAME + ".locked")).write_bytes(b"never read")
    (rd / MANIFEST).write_text(json.dumps({"source": "dukascopy"}), encoding="utf-8")
    listing = sorted(str(p.relative_to(rd)) for p in rd.rglob("*"))
    for extra in ((), ("--tf", "H4"), ("--tf", "H4", "--symbol", "GOLD", "--overwrite")):
        code, _, err = _run(capsys, "--input", src, "--out", rd, *extra)
        assert code == 2 and "not made by this tool" in err and "new, empty folder" in err, extra
    assert sorted(str(p.relative_to(rd)) for p in rd.rglob("*")) == listing
    # this tool's own folder takes another timeframe next to its earlier output
    assert _run(capsys, "--input", src, "--out", tmp_path / "mine")[0] == 0
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "mine", "--tf", "H4")
    assert code == 0, err


def test_ctrl_c_and_windows_friendly_messages(tmp_path, capsys, monkeypatch):
    code, _, err = _run(capsys, "--input", "D:\\ticks\\gold.csv", "--out", tmp_path / "o")
    assert code == 2 and "'D:\\ticks\\gold.csv'" in err and "\\\\" not in err
    # PowerShell: "D:\My Ticks\" swallows the closing quote and the rest of the line
    code, _, err = _run(capsys, "--input", 'D:\\My Ticks" --out D:\\out --tz ny+7')
    assert code == 2 and "Remove the backslash before the closing quote" in err
    src = _write_duka(tmp_path / "t.csv", _ticks("2025-09-15", 300, seed=69))
    out = tmp_path / "o"

    def interrupt(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(tb, "process_inputs", interrupt)
    code, _, err = _run(capsys, "--input", src, "--out", out)
    assert code == 3 and "Ctrl+C" in err and "no output files were written" in err and "Traceback" not in err
    assert not out.exists()
    monkeypatch.undo()
    # Ctrl+C while the new files are put in place: the old ones come back
    assert _run(capsys, "--input", src, "--out", out)[0] == 0
    before = {p.name: p.read_bytes() for p in (out / "train" / NAME, out / MANIFEST)}
    real = tb.os.replace

    def stop(a, b):
        if ".tmp-" in str(a) and str(b).endswith(MANIFEST):
            raise KeyboardInterrupt
        return real(a, b)

    monkeypatch.setattr(tb.os, "replace", stop)
    other = _write_duka(tmp_path / "u.csv", _ticks("2025-09-15", 300, seed=70))
    code, _, err = _run(capsys, "--input", other, "--out", out, "--overwrite")
    monkeypatch.setattr(tb.os, "replace", real)
    assert code == 3 and "no output files were written" in err
    assert {p.name: p.read_bytes() for p in (out / "train" / NAME, out / MANIFEST)} == before
    assert sorted(p.name for p in out.iterdir()) == [MANIFEST, "train"]           # no locked bars here
    assert sorted(p.name for p in (out / "train").iterdir()) == [NAME]


def test_encoding_flag_and_locale_fallback_read_a_gbk_file(tmp_path, capsys, monkeypatch):
    ticks = _ticks(hours=40, seed=71)
    names = ("\u65f6\u95f4", "\u4e70\u4ef7", "\u5356\u4ef7")                 # time / bid / ask in Chinese
    src = tmp_path / "gbk.csv"
    pd.DataFrame({names[0]: ticks["t"] // 1_000_000, names[1]: ticks["bid"], names[2]: ticks["ask"]}).to_csv(
        src, index=False, encoding="gbk")
    cols = ("--time-col", names[0], "--bid-col", names[1], "--ask-col", names[2], "--tz", "UTC")
    monkeypatch.setattr(tb.locale, "getpreferredencoding", lambda *_a: "UTF-8")
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "a", *cols)
    assert code == 2 and "--encoding" in err
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "b", *cols, "--encoding", "gbk")
    assert code == 0, err
    _assert_bars_equal(_train(tmp_path / "b"), _ref_bars(ticks))
    assert _manifest(tmp_path / "b")["inputs"][0]["encoding"] == "gbk"
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "c", *cols, "--encoding", "nonsense-enc")
    assert code == 2 and "--encoding" in err
    # without the flag, a file that is not UTF-8 is read with the computer's own encoding (cp936 in China)
    monkeypatch.setattr(tb.locale, "getpreferredencoding", lambda *_a: "cp936")
    code, out, err = _run(capsys, "--input", src, "--out", tmp_path / "d", *cols)
    assert code == 0, err
    assert "cp936" in out
    _assert_bars_equal(_train(tmp_path / "d"), _ref_bars(ticks))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="per-thread nice values are a Linux matter")
def test_lower_priority_reaches_every_thread_on_linux():
    import subprocess
    import textwrap
    code = textwrap.dedent(f"""
        import importlib.util, os, sys, threading
        spec = importlib.util.spec_from_file_location("ttb", {str(SCRIPT)!r})
        tb = importlib.util.module_from_spec(spec); sys.modules["ttb"] = tb; spec.loader.exec_module(tb)
        ev = threading.Event()
        threading.Thread(target=ev.wait, daemon=True).start()     # a thread that exists before the call

        def nices():
            res = []
            for tid in os.listdir("/proc/self/task"):
                st = open("/proc/self/task/" + tid + "/stat").read()
                res.append(int(st[st.rindex(")") + 2:].split()[16]))
            return res

        before = nices()
        msg = tb.lower_priority()
        after = nices()
        print(max(before), min(after), max(after), len(after))
        print(msg)
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    hi_before, lo_after, hi_after, n = map(int, r.stdout.split("\n")[0].split())
    if hi_before >= 10:
        pytest.skip("the test already runs at nice >= 10")
    assert n >= 2 and lo_after == hi_after > hi_before and "nice +10" in r.stdout


def _bad_times(out: Path) -> int:
    """Rows dropped for a time outside 1990..2100: pandas 3 parses such years and they count as
    time_out_of_range; pandas 2 cannot hold some of them and counts them as unparsable_time."""
    d = _manifest(out)["rows"]["dropped"]
    return d["time_out_of_range"] + d["unparsable_time"]


def test_out_of_range_years_are_counted_not_fatal(tmp_path, capsys):
    ticks = _ticks(hours=300, seed=72)
    src = _write_mt5(tmp_path / "m.csv", ticks, extra_rows=False)
    lines = src.read_bytes().decode("utf-16").split("\r\n")
    for at, day in ((900, "2300.01.01"), (1500, "1700.01.01")):              # after the 200-row sniff window
        lines.insert(at, f"{day}\t10:00:00.000\t2000.10\t2000.40\t\t\t6")
    src.write_bytes("\r\n".join(lines).encode("utf-16"))
    code, _, err = _run(capsys, "--input", src, "--out", tmp_path / "m", "--tz", "ny+7")
    assert code == 0, err
    assert _bad_times(tmp_path / "m") == 2
    _assert_bars_equal(_train(tmp_path / "m"), _ref_bars(ticks))
    tt = pd.to_datetime(ticks["t"], utc=True)
    iso = tt.dt.strftime("%Y-%m-%d %H:%M:%S.%f").tolist()
    iso[900] = "2500-01-01 10:00:00.000000"
    pd.DataFrame({"time": iso, "bid": ticks["bid"], "ask": ticks["ask"]}).to_csv(tmp_path / "iso.csv", index=False)
    code, _, err = _run(capsys, "--input", tmp_path / "iso.csv", "--out", tmp_path / "i", "--time-col", "time",
                        "--bid-col", "bid", "--ask-col", "ask", "--tz", "UTC", "--chunksize", 300)
    assert code == 0, err
    assert _bad_times(tmp_path / "i") == 1
    jf = tt.dt.strftime("%d.%m.%Y %H:%M:%S.%f").str[:23].tolist()
    jf[900] = "01.01.2500 10:00:00.000"
    pd.DataFrame({"Gmt time": jf, "Ask": ticks["ask"], "Bid": ticks["bid"]}).to_csv(tmp_path / "jf.csv", index=False)
    code, _, err = _run(capsys, "--input", tmp_path / "jf.csv", "--out", tmp_path / "j")
    assert code == 0, err
    assert _bad_times(tmp_path / "j") == 1
    # the conversion helper itself: microsecond datetimes far outside the nanosecond range
    arr = np.array(["2024-01-01T00:00:00", "2300-01-01T00:00:00", "1700-01-01T00:00:00", "NaT"],
                   dtype="datetime64[us]")
    ns = tb._dt64_to_ns(arr)
    assert ns[0] == pd.Timestamp("2024-01-01").value and ns[3] == tb.NAT
    assert ns[1] >= tb.TIME_MAX_NS and 0 > ns[2] != tb.NAT


def test_compare_at_h4_gives_no_hour_verdict(tmp_path, capsys):
    ticks = _ticks("2024-01-08", 24 * 60, seed=73, per_hour=(8, 12))
    assert _run(capsys, "--input", _write_duka(tmp_path / "r.csv", ticks), "--out", tmp_path / "ref",
                "--tf", "H4")[0] == 0
    ref = tmp_path / "ref" / "train" / "XAUUSD_H4.parquet"
    for hours in (2, 3):
        src = _write_duka(tmp_path / f"s{hours}.csv", ticks.assign(t=ticks["t"] + hours * 3600 * NS))
        code, out, err = _run(capsys, "--input", src, "--out", tmp_path / f"o{hours}", "--tf", "H4",
                              "--compare-with", ref)
        assert code == 0, err
        c = _manifest(tmp_path / f"o{hours}", "XAUUSD_H4_manifest.json")["compare"]
        assert "TIME ZONE CHECK FAILED" not in out and "Try --tz" not in out
        assert "--tf H1" in out and c["best_lag_hours"] is None
        assert not any(" h)" in w for w in c["warnings"])                      # no made-up hour count


def test_compare_suggests_the_right_daylight_saving_rule(tmp_path, capsys):
    # January (both on standard time), the US-only daylight-saving weeks of March, and July
    ticks = pd.concat([_ticks("2024-01-15", 120, seed=74, per_hour=(10, 16)),
                       _ticks("2024-03-18", 120, seed=75, per_hour=(10, 16)),
                       _ticks("2024-07-15", 120, seed=76, per_hour=(10, 16))], ignore_index=True)
    assert _run(capsys, "--input", _write_duka(tmp_path / "r.csv", ticks), "--out", tmp_path / "ref")[0] == 0
    ref = tmp_path / "ref" / "train" / NAME
    cases = (("Etc/GMT-2", "ny+7", "Try --tz +02:00"),                     # fixed UTC+2 broker read as ny+7
             ("Europe/Athens", "ny+7", "Try --tz Europe/Athens"),          # EU daylight-saving broker read as ny+7
             ("ny+7", "+02:00", "Try --tz ny+7"),
             ("ny+7", "ny+7", None))
    for k, (true_tz, used, tip) in enumerate(cases):
        src = _write_mt5(tmp_path / f"m{k}.csv", ticks, tz=true_tz, extra_rows=False)
        code, out, err = _run(capsys, "--input", src, "--out", tmp_path / f"o{k}", "--tz", used, "--compare-with", ref)
        assert code == 0, err
        c = _manifest(tmp_path / f"o{k}")["compare"]
        if tip is None:
            assert not c["warnings"] and "TIME ZONE CHECK FAILED" not in out
        else:
            assert "TIME ZONE CHECK FAILED" in out and tip in out, (true_tz, used, out)
            assert "Try --tz ny+6" not in out and "Brokers on New York close time need" not in out
    reg = _manifest(tmp_path / "o1")["compare"]["best_lag_by_dst_regime"]
    assert [reg[r]["best_lag_bars"] for r in ("both_standard_time", "us_dst_only", "both_daylight_saving")] == [0, -1, 0]


def test_nanosecond_epochs_with_a_blank_cell_stay_exact(tmp_path, capsys):
    base = _ticks("2025-08-04", 120, seed=77)
    opens = (np.arange(120) * 3600 + pd.Timestamp("2025-08-04", tz="UTC").value // NS) * NS
    edge = pd.DataFrame({"t": np.concatenate([opens[1:] - 1, opens + 1]), "bid": 2001.0, "ask": 2001.3})
    ticks = pd.concat([base, edge], ignore_index=True)
    rows = ["time,bid,ask"] + [f"{t},{b},{a}" for t, b, a in zip(ticks["t"], ticks["bid"], ticks["ask"])]
    (tmp_path / "ns.csv").write_text("\n".join(rows + [",2000.0,2000.3"]) + "\n")      # one blank time cell
    cols = ("--time-col", "time", "--bid-col", "bid", "--ask-col", "ask", "--tz", "UTC", "--chunksize", 97)
    code, _, err = _run(capsys, "--input", tmp_path / "ns.csv", "--out", tmp_path / "c", *cols)
    assert code == 0, err
    ref = _ref_bars(ticks)
    _assert_bars_equal(_train(tmp_path / "c"), ref)
    pd.DataFrame({"time": pd.array(list(ticks["t"]) + [None], dtype="Int64"), "bid": list(ticks["bid"]) + [2000.0],
                  "ask": list(ticks["ask"]) + [2000.3]}).to_parquet(tmp_path / "ns.parquet", index=False)
    code, _, err = _run(capsys, "--input", tmp_path / "ns.parquet", "--out", tmp_path / "p", *cols)
    assert code == 0, err
    _assert_bars_equal(_train(tmp_path / "p"), ref)
    # float64 epochs (a column with a blank cell): whole numbers are converted exactly
    got = tb._epoch_to_ns(pd.Series([1758499199999.0, np.nan, 1.5]), "ms")
    assert got[0] == 1758499199999 * 1_000_000 and got[1] == tb.NAT and got[2] == 1_500_000


def test_no_usable_ticks_error_prints_the_reasons(tmp_path, capsys):
    t = _ticks(hours=30, seed=78)
    t["ask"], t["bid"] = t["bid"].copy(), t["ask"].copy()                    # every quote crossed
    code, out, err = _run(capsys, "--input", _write_duka(tmp_path / "x.csv", t), "--out", tmp_path / "o")
    assert code == 3 and "reasons above" in err
    assert "rows dropped, by reason:" in out and re.search(r"crossed_quote\s+" + f"{len(t):,}", out)
    hdr = tmp_path / "hdr.csv"
    hdr.write_text("<DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>\r\n", encoding="utf-16")
    code, _, err = _run(capsys, "--input", hdr, "--out", tmp_path / "h", "--tz", "ny+7")
    assert code == 3 and "no data rows" in err and "see the counts" not in err
    assert not (tmp_path / "o").exists() and not (tmp_path / "h").exists()


def test_overlapping_files_are_refused_or_dropped(tmp_path, capsys):
    ticks = _ticks("2024-01-08", 24 * 6, seed=79)
    day = pd.Timestamp("2024-01-11", tz="UTC").value
    folder = tmp_path / "ov"
    folder.mkdir()
    a = ticks[ticks["t"] < day + 86400 * NS]                                  # 2024-01-11 is in both exports
    b = ticks[ticks["t"] >= day]
    _write_mt5(folder / "XAU_1.csv", a, extra_rows=False)
    _write_mt5(folder / "XAU_2.csv", b, extra_rows=False)
    code, _, err = _run(capsys, "--input", folder, "--out", tmp_path / "r", "--tz", "ny+7")
    assert code == 3 and "input files overlap" in err and "XAU_1.csv and XAU_2.csv" in err and "--overlap drop" in err
    assert not (tmp_path / "r").exists()
    code, _, err = _run(capsys, "--input", folder, "--out", tmp_path / "d", "--tz", "ny+7", "--overlap", "drop")
    assert code == 0, err
    _assert_bars_equal(_train(tmp_path / "d"), _ref_bars(ticks))              # as from one export
    m = _manifest(tmp_path / "d")
    assert m["rows"]["dropped"]["overlap_with_earlier_file"] == len(a) + len(b) - len(ticks)
    assert m["rows"]["overlap"]["overlapping_file_pairs"][0]["earlier_file"] == "XAU_1.csv"
    code, out, err = _run(capsys, "--input", folder, "--out", tmp_path / "k", "--tz", "ny+7", "--overlap", "keep")
    assert code == 0, err
    got = _train(tmp_path / "k").set_index("time")["tick_volume"]
    ref = _ref_bars(ticks).set_index("time")["tick_volume"]
    assert (got == 2 * ref).sum() == 24 and "WARNING: !!! input files overlap" in out


def test_many_unreadable_rows_are_refused_and_day_month_swaps_flagged(tmp_path, capsys):
    ticks = _ticks("2023-01-02", 24 * 40, seed=80, per_hour=(2, 4))
    tt = pd.to_datetime(ticks["t"], utc=True)
    us = tmp_path / "us.csv"
    pd.DataFrame({"Time": tt.dt.strftime("%m/%d/%Y %H:%M:%S"), "Bid": ticks["bid"], "Ask": ticks["ask"]}).to_csv(
        us, index=False)
    args = ("--time-col", "Time", "--bid-col", "Bid", "--ask-col", "Ask", "--time-format", "%d/%m/%Y %H:%M:%S",
            "--tz", "UTC")                                                   # day and month swapped
    code, out, err = _run(capsys, "--input", us, "--out", tmp_path / "a", *args)
    assert code == 3 and "unparsable_time" in err and "--allow-drops" in err and "--dry-run" in err
    assert re.search(r"read 1/1: us\.csv: [\d,]+ rows .*; kept [\d,]+, dropped [\d,]+", out)
    assert not (tmp_path / "a").exists()
    code, out, err = _run(capsys, "--input", us, "--out", tmp_path / "b", *args, "--allow-drops")
    assert code == 0, err
    assert "WARNING: !!!" in out and "jump back by more than a day" in out
    # days 1-12 only: every row parses, but the dates jump back -> a warning names the likely swap
    few = ticks[tt.dt.day <= 12]
    pd.DataFrame({"Time": pd.to_datetime(few["t"], utc=True).dt.strftime("%m/%d/%Y %H:%M:%S"), "Bid": few["bid"],
                  "Ask": few["ask"]}).to_csv(tmp_path / "us12.csv", index=False)
    code, out, err = _run(capsys, "--input", tmp_path / "us12.csv", "--out", tmp_path / "c", *args)
    assert code == 0, err
    assert "jump back by more than a day" in out and "%d/%m or %m/%d" in out
    assert _manifest(tmp_path / "c")["inputs"][0]["backward_time_jumps_over_1_day"] > 0


def test_utc_by_definition_files_refuse_another_tz(tmp_path, capsys):
    ticks = _ticks(hours=60, seed=81)
    tt = pd.to_datetime(ticks["t"], utc=True)
    gmt = tmp_path / "gmt.csv"
    pd.DataFrame({"Gmt time": tt.dt.strftime("%d.%m.%Y %H:%M:%S.%f").str[:23], "Ask": ticks["ask"],
                  "Bid": ticks["bid"]}).to_csv(gmt, index=False)
    for extra in ((), ("--force-tz",)):
        code, _, err = _run(capsys, "--input", gmt, "--out", tmp_path / "a", "--tz", "ny+7", *extra)
        assert code == 2 and "Gmt time" in err and "shift every bar" in err
    assert _run(capsys, "--input", gmt, "--out", tmp_path / "u", "--tz", "UTC")[0] == 0
    duka = _write_duka(tmp_path / "d.csv", ticks)
    code, _, err = _run(capsys, "--input", duka, "--out", tmp_path / "b", "--tz", "ny+7")
    assert code == 2 and "--force-tz" in err and "--utc-offset" in err
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
    code, out, err = _run(capsys, "--input", duka, "--out", tmp_path / "c", "--tz", "+02:00", "--force-tz")
    assert code == 0, err
    assert "because of --force-tz" in out
    _assert_bars_equal(_train(tmp_path / "c"), _ref_bars(ticks.assign(t=ticks["t"] - 2 * 3600 * NS)))


def test_dukascopy_node_files_with_formatted_dates(tmp_path, capsys):
    ticks = _ticks(hours=60, seed=82, whole_seconds=True)
    tt = pd.to_datetime(ticks["t"], utc=True)
    ref = _ref_bars(ticks)

    def write(name, times):
        pd.DataFrame({"timestamp": times, "askPrice": ticks["ask"], "bidPrice": ticks["bid"]}).to_csv(
            tmp_path / name, index=False)
        return tmp_path / name

    iso = write("iso.csv", tt.dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"))             # -df iso: UTC with 'Z'
    code, out, err = _run(capsys, "--input", iso, "--out", tmp_path / "a")
    assert code == 0, err
    assert "--date-format" in out
    _assert_bars_equal(_train(tmp_path / "a"), ref)
    plain = write("plain.csv", tt.dt.strftime("%Y-%m-%d %H:%M:%S"))              # no offset: --tz needed
    code, _, err = _run(capsys, "--input", plain, "--out", tmp_path / "b")
    assert code == 2 and "--tz is required" in err
    assert _run(capsys, "--input", plain, "--out", tmp_path / "b", "--tz", "UTC")[0] == 0
    _assert_bars_equal(_train(tmp_path / "b"), ref)
    other = write("other.csv", tt.dt.strftime("%d.%m.%Y %H:%M:%S"))
    code, _, err = _run(capsys, "--input", other, "--out", tmp_path / "c")
    assert code == 2 and "--format generic --time-col timestamp --bid-col bidPrice --ask-col askPrice" in err
    code, _, err = _run(capsys, "--input", other, "--out", tmp_path / "c", "--format", "generic", "--time-col",
                        "timestamp", "--bid-col", "bidPrice", "--ask-col", "askPrice", "--time-format",
                        "%d.%m.%Y %H:%M:%S", "--tz", "UTC")
    assert code == 0, err
    _assert_bars_equal(_train(tmp_path / "c"), ref)


def test_split_date_time_headerless_and_date_only_files(tmp_path, capsys):
    ticks = _ticks("2024-01-07", 120, seed=83)
    tt = pd.to_datetime(ticks["t"], utc=True)
    split = tmp_path / "split.csv"
    pd.DataFrame({"Date": tt.dt.strftime("%d/%m/%Y"), "Time": tt.dt.strftime("%H:%M:%S.%f").str[:12],
                  "Bid": ticks["bid"], "Ask": ticks["ask"]}).to_csv(split, index=False)
    code, _, err = _run(capsys, "--input", split, "--out", tmp_path / "x")
    assert code == 2 and "--date-col" in err
    code, _, err = _run(capsys, "--input", split, "--out", tmp_path / "a", "--date-col", "Date", "--time-col", "Time",
                        "--bid-col", "Bid", "--ask-col", "Ask", "--time-format", "%d/%m/%Y %H:%M:%S.%f", "--tz", "UTC")
    assert code == 0, err
    _assert_bars_equal(_train(tmp_path / "a"), _ref_bars(ticks))
    # the date column alone: every tick would land at 00:00 -> refused
    code, _, err = _run(capsys, "--input", split, "--out", tmp_path / "x", "--time-col", "Date", "--bid-col", "Bid",
                        "--ask-col", "Ask", "--time-format", "%d/%m/%Y", "--tz", "UTC")
    assert code == 3 and "00:00:00" in err and "--date-col" in err
    # HistData-style: no header row
    hist = tmp_path / "hist.csv"
    hist.write_text("\n".join(f"{s:%Y%m%d %H%M%S}{s.microsecond // 1000:03d},{b},{a},0"
                              for s, b, a in zip(tt, ticks["bid"], ticks["ask"])) + "\n")
    code, _, err = _run(capsys, "--input", hist, "--out", tmp_path / "x")
    assert code == 2 and "--no-header" in err and "looks like data" in err
    code, _, err = _run(capsys, "--input", hist, "--out", tmp_path / "h", "--no-header", "--time-col", "col0",
                        "--bid-col", "col1", "--ask-col", "col2", "--time-format", "%Y%m%d %H%M%S%f", "--tz", "UTC")
    assert code == 0, err
    _assert_bars_equal(_train(tmp_path / "h"), _ref_bars(ticks))
    assert not (tmp_path / "x").exists()


def test_mt5_carry_is_not_taken_from_an_unrelated_file(tmp_path, capsys):
    jan = _ticks("2024-01-08", 48, seed=84, level=2050.0)
    jul = _ticks("2024-07-08", 48, seed=85, level=2030.0)
    _write_mt5(tmp_path / "XAUUSD July.csv", jul, extra_rows=False)
    src = _write_mt5(tmp_path / "XAUUSD January.csv", jan, extra_rows=False)
    lines = src.read_bytes().decode("utf-16").split("\r\n")
    cells = lines[1].split("\t")
    cells[3], cells[6] = "", "2"                                               # first January row: ASK blank
    lines[1] = "\t".join(cells)
    src.write_bytes("\r\n".join(lines).encode("utf-16"))
    code, _, err = _run(capsys, "--input", tmp_path / "XAUUSD July.csv", "--input", src, "--out", tmp_path / "o",
                        "--tz", "ny+7")
    assert code == 0, err
    d = _manifest(tmp_path / "o")["rows"]["dropped"]
    assert d["mt5_before_first_quote"] == 1 and d["crossed_quote"] == 0       # July's ask was not carried
    _assert_bars_equal(_train(tmp_path / "o"), _ref_bars(pd.concat([jul, jan.iloc[1:]], ignore_index=True)))


def test_tick_volume_meaning_is_stated(tmp_path, capsys):
    src = _write_duka(tmp_path / "t.csv", _ticks("2024-01-08", 120, seed=86))
    assert _run(capsys, "--input", src, "--out", tmp_path / "ref")[0] == 0
    code, out, err = _run(capsys, "--input", src, "--out", tmp_path / "o", "--compare-with",
                          tmp_path / "ref" / "train" / NAME)
    assert code == 0, err
    m = _manifest(tmp_path / "o")
    assert "NOT the traded volume" in m["note"] and "vol_ratio" in m["note"]
    assert "NOT traded volume" in m["settings"]["column_notes"]["tick_volume"]
    vol = m["compare"]["volume"]
    assert "NUMBER OF TICKS" in vol["note"] and vol["reference_volume_column"] == "tick_volume"
    assert vol["log_volume_corr"] == pytest.approx(1.0)
    assert "tick_volume = number of ticks per bar, not the Dukascopy file's traded volume" in out


def test_dry_run_shows_weekday_offset_and_reopen_hint(tmp_path, capsys):
    ticks = _ticks("2024-01-07 23:00", 30, seed=87)                           # Sunday 23:00 UTC: the weekly reopen
    src = _write_mt5(tmp_path / "m.csv", ticks, extra_rows=False)
    code, out, err = _run(capsys, "--input", src, "--out", tmp_path / "o", "--tz", "ny+7", "--dry-run")
    assert code == 0, err
    assert re.search(r"Mon 2024-01-08 01:\d\d:\d\d\.\d{3}\s+Sun 2024-01-07 23:\d\d:\d\d\.\d{3}\s+UTC\+02:00", out)
    assert "Sunday 22:00 UTC (US summer) or 23:00 UTC (US winter)" in out
    assert not (tmp_path / "o").exists()


# ---------------------------------------------------------------------------------------
# round-2 review findings

def test_overlap_is_decided_by_ticks_not_by_time_spans(tmp_path, capsys):
    ticks = _ticks("2024-01-02", 24 * 20, seed=90, per_hour=(20, 40))
    ref = _ref_bars(ticks)
    lo, hi = pd.Timestamp("2024-01-10", tz="UTC").value, pd.Timestamp("2024-01-14", tz="UTC").value
    hole = (ticks["t"] >= lo) & (ticks["t"] < hi)
    # a re-downloaded missing week inside the first file's first-to-last span: no tick is in both files
    main = _write_duka(tmp_path / "jan.csv", ticks[~hole])
    fill = _write_duka(tmp_path / "jan_missing_week.csv", ticks[hole])
    for extra in ((), ("--overlap", "drop"), ("--overlap", "keep")):
        out = tmp_path / ("o" + "".join(extra))
        code, text, err = _run(capsys, "--input", main, "--input", fill, "--out", out, *extra)
        assert code == 0, (extra, err)
        _assert_bars_equal(_train(out), ref)
        m = _manifest(out)
        assert m["rows"]["overlap"]["overlapping_file_pairs"] == [] and "input files overlap" not in text
        assert m["rows"]["dropped"]["overlap_with_earlier_file"] == 0 and not m["train"]["gaps"]["over_72h_longest"]
    # one stray tick with a wrong date in the first file does not swallow the second file
    jan, rest = ticks[ticks["t"] < lo], ticks[ticks["t"] >= lo]
    stray = pd.DataFrame({"t": [hi + 5 * 3600 * NS + 123], "bid": [2000.01], "ask": [2000.31]})
    a = _write_duka(tmp_path / "a.csv", pd.concat([jan, stray], ignore_index=True))
    b = _write_duka(tmp_path / "b.csv", rest)
    code, _, err = _run(capsys, "--input", a, "--input", b, "--out", tmp_path / "s", "--overlap", "drop")
    assert code == 0, err
    m = _manifest(tmp_path / "s")
    assert m["rows"]["dropped"]["overlap_with_earlier_file"] == 0 and m["rows"]["kept"] == len(ticks) + 1
    # a real overlap that ends in the middle of a minute: only the repeated ticks are dropped
    cut_a = int(ticks["t"].iloc[len(ticks) // 2])
    start_b = cut_a - 90 * 60 * NS
    a = _write_duka(tmp_path / "x1.csv", ticks[ticks["t"] <= cut_a])
    b = _write_duka(tmp_path / "x2.csv", ticks[ticks["t"] >= start_b])
    code, _, err = _run(capsys, "--input", a, "--input", b, "--out", tmp_path / "r")
    assert code == 3 and "input files overlap" in err and "probably the same ticks" in err
    assert "x1.csv and x2.csv" in err
    code, text, err = _run(capsys, "--input", a, "--input", b, "--out", tmp_path / "d", "--overlap", "drop")
    assert code == 0, err
    _assert_bars_equal(_train(tmp_path / "d"), ref)
    both = int(((ticks["t"] >= start_b) & (ticks["t"] <= cut_a)).sum())
    m = _manifest(tmp_path / "d")
    assert m["rows"]["dropped"]["overlap_with_earlier_file"] == both
    assert m["rows"]["overlap"]["overlapping_file_pairs"][0]["later_file_rows_in_overlap_train_period"] == both
    assert "WARNING: input files overlap" in text and "!!! --overlap drop" not in text      # a small share
    # a copy of a file: all of it is dropped (correctly), with a loud warning
    copy = tmp_path / "copy_of_x1.csv"
    copy.write_bytes(a.read_bytes())
    code, text, err = _run(capsys, "--input", a, "--input", copy, "--out", tmp_path / "c", "--overlap", "drop")
    assert code == 0, err
    _assert_bars_equal(_train(tmp_path / "c"), _ref_bars(ticks[ticks["t"] <= cut_a]))
    assert "WARNING: !!! --overlap drop dropped a large share" in text and "copy_of_x1.csv: 100.0%" in text


def test_mt5_carry_at_a_file_boundary_does_not_depend_on_chunksize(tmp_path, capsys):
    jan = _ticks("2024-01-08", 60, seed=91, repeat=0.3)
    jul = _ticks("2024-07-08", 60, seed=92, repeat=0.3, level=2300.0)
    a = _write_mt5(tmp_path / "a_jan.csv", jan, extra_rows=False)
    b = _write_mt5(tmp_path / "b_jul.csv", jul, extra_rows=False)
    lines = b.read_bytes().decode("utf-16").split("\r\n")
    c1, c2 = lines[1].split("\t"), lines[2].split("\t")
    c1[1] = "xx:yy:zz"                                    # first July row: unreadable time, full prices
    c1[2], c1[3] = f"{jul['bid'].iloc[0]:.2f}", f"{jul['ask'].iloc[0]:.2f}"
    c2[2], c2[3], c2[6] = "", f"{jul['ask'].iloc[1]:.2f}", "4"      # second row: BID blank = the row above
    lines[1], lines[2] = "\t".join(c1), "\t".join(c2)
    b.write_bytes("\r\n".join(lines).encode("utf-16"))
    expect = pd.concat([jan, jul.iloc[1:].assign(bid=[jul["bid"].iloc[0]] + list(jul["bid"].iloc[2:]))],
                       ignore_index=True)
    results = set()
    for cs in (1, 2, 1000, 10_000_000):
        out = tmp_path / f"o{cs}"
        code, _, err = _run(capsys, "--input", a, "--input", b, "--out", out, "--tz", "ny+7", "--chunksize", cs)
        assert code == 0, (cs, err)
        d = _manifest(out)["rows"]["dropped"]
        results.add(((out / "train" / NAME).read_bytes(), tuple(sorted((k, v) for k, v in d.items() if v))))
    assert len(results) == 1
    _, drops = results.pop()
    assert drops == (("unparsable_time", 1),)
    _assert_bars_equal(_train(tmp_path / "o1"), _ref_bars(expect))
    # a file that continues the previous one takes its last bid/ask for leading blank cells, at any chunk size
    first, second = jan.iloc[:400], jan.iloc[400:].copy()
    second.iloc[0, second.columns.get_loc("bid")] = first["bid"].iloc[-1]
    a2 = _write_mt5(tmp_path / "c1.csv", first, extra_rows=False)
    b2 = _write_mt5(tmp_path / "c2.csv", second, extra_rows=False)
    lines = b2.read_bytes().decode("utf-16").split("\r\n")
    junk = lines[1].split("\t")
    junk[1], junk[2], junk[3] = "??:??", "", ""                        # unreadable time, no prices
    c = lines[1].split("\t")
    c[2] = ""                                                         # bid unchanged from file c1
    lines[1] = "\t".join(c)
    lines.insert(1, "\t".join(junk))
    b2.write_bytes("\r\n".join(lines).encode("utf-16"))
    outs = set()
    for cs in (1, 3, 10_000_000):
        out = tmp_path / f"k{cs}"
        code, _, err = _run(capsys, "--input", a2, "--input", b2, "--out", out, "--tz", "ny+7", "--chunksize", cs)
        assert code == 0, (cs, err)
        _assert_bars_equal(_train(out), _ref_bars(pd.concat([first, second], ignore_index=True)))
        outs.add((out / "train" / NAME).read_bytes())
    assert len(outs) == 1


def test_messages_never_echo_values_that_may_be_from_the_locked_period(tmp_path, capsys):
    ticks = _ticks("2025-11-03", 30, seed=93, level=4400.0)              # all after the cutoff
    tt = pd.to_datetime(ticks["t"], utc=True)
    hist = tmp_path / "hist.csv"
    hist.write_text("\n".join(f"{s:%Y%m%d %H%M%S}{s.microsecond // 1000:03d},{b},{a},0"
                              for s, b, a in zip(tt, ticks["bid"], ticks["ask"])) + "\n")
    hist2 = tmp_path / "hist2.csv"
    hist2.write_text("\n".join(f"{s:%Y%m%d},{s:%H%M%S},{b},{a}" for s, b, a in zip(tt, ticks["bid"], ticks["ask"])))
    yy = tmp_path / "yy.csv"
    pd.DataFrame({"time": tt.dt.strftime("%d/%m/%y %H:%M:%S"), "bid": ticks["bid"], "ask": ticks["ask"]}).to_csv(
        yy, index=False)
    g = ("--time-col", "time", "--bid-col", "bid", "--ask-col", "ask", "--tz", "UTC")
    cases = [(hist, ()), (hist, ("--format", "generic", "--time-col", "Time", "--bid-col", "b", "--ask-col", "a")),
             (hist2, ()), (hist, ("--format", "mt5", "--tz", "UTC")), (yy, g), (yy, g + ("--time-format", "%Y-%m-%d"))]
    first_bid, first_ask = f"{ticks['bid'].iloc[0]}", f"{ticks['ask'].iloc[0]}"
    for path, extra in cases:
        code, out, err = _run(capsys, "--input", path, "--out", tmp_path / "o", *extra)
        assert code == 2, (path.name, extra, err)
        text = (out + err).replace(str(tmp_path), "<tmp>")                    # the temp folder may hold digits
        for leak in ("2025", "251103", "03/11/25", "4399", "4400", first_bid, first_ask):
            assert leak not in text, (path.name, extra, leak, err)
    code, _, err = _run(capsys, "--input", hist, "--out", tmp_path / "o")
    assert "4 cells and looks like data" in err and "not shown" in err and "--no-header" in err
    # the rule itself: a value is printed only when it surely names a time before the cutoff - 14 h
    show = lambda v: tb._value_shown(v, CUTOFF_S)                              # noqa: E731
    assert show("2024-01-03 10:00:00") and show("2024.01.03 10:00:00.123") and show("22:01:39.212")
    assert show("11/03/24 10:00") and show("2025-09-27 09:59:59")
    for v in ("03/11/25 00:07:27", "03.11.2025 00:07:27.000", "20251103", "20240107 170139212", "4399.9",
              "1762128447000", "2025-11-03", "2025-09-27 10:00:00", "Nov 3 2025", "xx:yy:zz", ""):
        assert not show(v), v
    assert tb._cols_text(["4399.9", "20251103 000727", "0", "Bid", "col0", "<DATE>"], CUTOFF_S) == \
        "(hidden), (hidden), (hidden), Bid, col0, <DATE>"


def test_failed_or_emptied_outputs_leave_no_folder_that_blocks_the_next_run(tmp_path, capsys, monkeypatch):
    src = _write_duka(tmp_path / "t.csv", _ticks("2025-09-20", 300, seed=94))      # has locked bars
    real = tb.os.replace
    for kind in (PermissionError(13, "the file is open in another program"), KeyboardInterrupt()):
        out = tmp_path / f"o_{type(kind).__name__}"

        def flaky(a, b, kind=kind):
            if ".tmp-" in str(a) and str(b).endswith(MANIFEST):
                raise kind
            return real(a, b)

        monkeypatch.setattr(tb.os, "replace", flaky)
        code, _, err = _run(capsys, "--input", src, "--out", out)
        monkeypatch.setattr(tb.os, "replace", real)
        assert code == 3 and "no output files were written" in err.lower() and "nothing was written or" not in err
        assert not out.exists()                          # the folders it made are removed again
        code, _, err = _run(capsys, "--input", src, "--out", out)
        assert code == 0, err
    # a run with locked bars, then --overwrite with data that ends before the cutoff, then again, then H4
    early = _write_duka(tmp_path / "early.csv", _ticks("2025-09-01", 400, seed=95))
    out = tmp_path / "o"
    assert _run(capsys, "--input", src, "--out", out)[0] == 0
    assert (out / "locked_holdout").is_dir()
    code, _, err = _run(capsys, "--input", early, "--out", out, "--overwrite")
    assert code == 0, err
    assert not (out / "locked_holdout").exists()         # emptied, so removed
    for extra in (("--overwrite",), ("--tf", "H4")):
        code, _, err = _run(capsys, "--input", early, "--out", out, *extra)
        assert code == 0, (extra, err)
    # if the empty folder cannot be removed, this tool's manifest still claims it
    assert _run(capsys, "--input", src, "--out", out, "--overwrite")[0] == 0
    monkeypatch.setattr(tb.os, "rmdir", lambda *_a: (_ for _ in ()).throw(OSError(16, "in use")))
    assert _run(capsys, "--input", early, "--out", out, "--overwrite")[0] == 0
    monkeypatch.undo()
    assert (out / "locked_holdout").is_dir()
    assert "locked_holdout" in _manifest(out)["folders_made_by_this_tool"]
    code, _, err = _run(capsys, "--input", early, "--out", out, "--overwrite")
    assert code == 0, err


def test_folder_and_pattern_inputs_never_list_a_locked_folder(tmp_path, capsys, monkeypatch):
    data = tmp_path / "data"
    (data / "ticks").mkdir(parents=True)
    (data / "locked_holdout").mkdir()
    _write_duka(data / "ticks" / "a.csv", _ticks("2025-08-01", 400, seed=96))
    (data / "locked_holdout" / "XAUUSD_H1.parquet.locked").write_bytes(b"synthetic, never read")
    stuff = tmp_path / "stuff"
    stuff.mkdir()
    _write_duka(stuff / "b.csv", _ticks("2025-08-01", 400, seed=97))
    (stuff / "XAUUSD_H1.parquet.locked").write_bytes(b"synthetic, never read")
    listed = []
    real_scandir, real_iterdir = tb.os.scandir, Path.iterdir
    monkeypatch.setattr(tb.os, "scandir", lambda p=".": listed.append(str(p)) or real_scandir(p))
    monkeypatch.setattr(Path, "iterdir", lambda self: listed.append(str(self)) or real_iterdir(self))
    monkeypatch.chdir(tmp_path)
    for inp in ("data/*/*", "data/*/a.csv", str(data / "*" / "*.csv"), "stuff", str(stuff)):
        code, _, err = _run(capsys, "--input", inp, "--out", tmp_path / "o")
        assert code == 2 and "refused" in err and "locked" in err, (inp, err)
    assert listed and not any(tb.is_forbidden(p) for p in listed), listed
    assert not (tmp_path / "o").exists()
    # patterns and folders without locked entries work as before
    code, _, err = _run(capsys, "--input", "data/ticks/*.csv", "--out", tmp_path / "o")
    assert code == 0, err


def test_bad_flags_and_damaged_files_give_plain_errors_not_tracebacks(tmp_path, capsys, monkeypatch):
    ticks = _ticks("2024-01-08", 60, seed=98)
    tt = pd.to_datetime(ticks["t"], utc=True)
    g = tmp_path / "g.csv"
    pd.DataFrame({"time": tt.dt.strftime("%Y-%m-%d %H:%M:%S.%f"), "bid": ticks["bid"], "ask": ticks["ask"]}).to_csv(
        g, index=False)
    cols = ("--time-col", "time", "--bid-col", "bid", "--ask-col", "ask", "--tz", "UTC")
    for fmt in ("%Y-%m-%d %H:%M:%S.%L", "%Y-%m-%d %H:%M:%S.%", "%f%"):
        code, _, err = _run(capsys, "--input", g, "--out", tmp_path / "o", *cols, "--time-format", fmt)
        assert code == 2 and "not a valid time format" in err and "%f" in err, (fmt, err)
    lines = g.read_text().splitlines()
    lines[300] = lines[300].replace(",", ',"', 1)                          # an unclosed quote
    (tmp_path / "q.csv").write_text("\n".join(lines) + "\n")
    for extra in (("--dry-run",), ()):
        code, _, err = _run(capsys, "--input", tmp_path / "q.csv", "--out", tmp_path / "o", *cols, *extra)
        assert code == 3 and "could not read" in err and "Traceback" not in err, (extra, err)
    for enc in ("rot13", "base64", "hex", "no-such-codec"):
        code, _, err = _run(capsys, "--input", g, "--out", tmp_path / "o", *cols, "--encoding", enc)
        assert code == 2 and "--encoding" in err and "not a text encoding" in err, (enc, err)
    # anything unforeseen: one plain line, exit 3
    monkeypatch.setattr(tb, "train_stats", lambda *_a, **_k: {}["boom"])
    code, _, err = _run(capsys, "--input", g, "--out", tmp_path / "o", *cols)
    assert code == 3 and err.startswith("unexpected error: KeyError") and "No output files were written" in err
    assert "Traceback" not in err and len(err.strip().splitlines()) == 1
    assert not (tmp_path / "o").exists()


def test_non_ascii_separator_prints_plain_ascii_and_reads_without_a_warning(tmp_path, capsys, recwarn):
    ticks = _ticks("2024-01-08", 60, seed=99)
    tt = pd.to_datetime(ticks["t"], utc=True)
    src = tmp_path / "s.csv"
    pd.DataFrame({"time": tt.dt.strftime("%Y-%m-%d %H:%M:%S.%f"), "bid": ticks["bid"], "ask": ticks["ask"]}).to_csv(
        src, index=False, sep="¦", encoding="utf-8")
    cols = ("--time-col", "time", "--bid-col", "bid", "--ask-col", "ask", "--tz", "UTC", "--sep", "¦")
    code, out, err = _run(capsys, "--input", src, "--out", tmp_path / "o", *cols, "--dry-run")
    assert code == 0, err
    assert out.isascii() and "separator : '\\xa6'" in out
    code, out, err = _run(capsys, "--input", src, "--out", tmp_path / "o", *cols, "--chunksize", 97)
    assert code == 0, err
    assert out.isascii() and err.isascii()
    _assert_bars_equal(_train(tmp_path / "o"), _ref_bars(ticks))
    assert not [w for w in recwarn if "python" in str(w.message).lower() or "ParserWarning" in w.category.__name__]


def test_dry_run_does_not_count_locked_period_rows(tmp_path, capsys):
    oct_ = _write_mt5(tmp_path / "oct.csv", _ticks("2025-10-06", 40, seed=100, level=3900.0), extra_rows=False)
    mixed = _write_mt5(tmp_path / "mixed.csv", pd.concat([_ticks("2025-09-26", 20, seed=101),
                                                          _ticks("2025-09-29", 20, seed=102, level=3900.0)],
                                                         ignore_index=True), extra_rows=False)
    for src in (oct_, mixed):
        n_rows = len(src.read_bytes().decode("utf-16").strip().split("\r\n")) - 1
        for extra in (("--tz", "ny+7"), ()):
            code, out, err = _run(capsys, "--input", src, "--out", tmp_path / "o", "--dry-run", *extra)
            assert code == 0, err
            body = out.replace(str(tmp_path), "<tmp>")              # the temp folder name may hold digits
            assert "some of these rows are (or may be) at or after the cutoff" in body
            assert f"{n_rows:,}" not in body and not re.search(r"first [\d,]+ rows|[\d,]+ of them", body)
            assert "3900" not in body and "2025-10" not in body and "2025.10" not in body
    code, out, _ = _run(capsys, "--input", mixed, "--out", tmp_path / "o", "--dry-run", "--tz", "ny+7")
    assert "rows before the cutoff" in out and "2025-09-26" in out              # the train rows are still shown


def test_zip_only_folders_and_excel_files_get_advice_that_works(tmp_path, capsys):
    import zipfile
    zips = tmp_path / "zips"
    zips.mkdir()
    inner = _write_duka(tmp_path / "XAUUSD_2025-06.csv", _ticks("2025-06-02", 30, seed=103))
    for name in ("XAUUSD_2025-06.zip", "XAUUSD_2025-07.zip"):
        with zipfile.ZipFile(zips / name, "w") as zf:
            zf.write(inner, inner.name)
    for inp in (zips, str(zips / "*"), str(zips / "*.csv")):
        code, _, err = _run(capsys, "--input", inp, "--out", tmp_path / "o")
        assert code == 2 and "Extract" in err, (inp, err)
    code, _, err = _run(capsys, "--input", zips, "--out", tmp_path / "o")
    assert "XAUUSD_2025-06.zip" in err and "2 archive file(s)" in err
    xlsx = tmp_path / "ticks.xlsx"
    with zipfile.ZipFile(xlsx, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("xl/workbook.xml", "<workbook/>")
    (tmp_path / "renamed.csv").write_bytes(xlsx.read_bytes())
    for path in (xlsx, tmp_path / "renamed.csv"):
        code, _, err = _run(capsys, "--input", path, "--out", tmp_path / "o")
        assert code == 2 and "Excel workbook" in err and "Save As" in err and "Extract" not in err, err
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    _write_duka(mixed / "a.csv", _ticks("2025-06-02", 400, seed=104))
    (mixed / "b.xlsx").write_bytes(xlsx.read_bytes())
    code, out, err = _run(capsys, "--input", mixed, "--out", tmp_path / "m")
    assert code == 0, err
    assert "Excel workbook(s) were NOT converted" in out and "Save As" in out
    assert not (tmp_path / "o").exists()
