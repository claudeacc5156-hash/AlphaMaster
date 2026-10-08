"""scripts/research/ticks_to_bars.py - turn your own tick files into AlphaMaster bar files.

Usage (from the AlphaMaster folder; Windows PowerShell shown):
    $env:PYTHONUTF8 = "1"
    # 1) look first (writes nothing): detected format, time zone, 5 sample rows
    python scripts\\research\\ticks_to_bars.py --input D:\\ticks --out D:\\research\\mydata --tz ny+7 --dry-run
    # 2) convert (one file or a folder of monthly files) to H1 bars
    python scripts\\research\\ticks_to_bars.py --input D:\\ticks --out D:\\research\\mydata --tz ny+7
    # 3) the same, plus a time-zone / price check against the Dukascopy research file
    python scripts\\research\\ticks_to_bars.py --input D:\\ticks --out D:\\research\\mydata --tz ny+7 `
        --compare-with D:\\research\\data\\train\\XAUUSD_H1.parquet --overwrite

What it does:
  * reads tick files (MetaTrader 5 tick export, dukascopy-node tick CSV, Dukascopy JForex /
    historical-data export, or any CSV / Parquet with named columns) in chunks, so the raw
    ticks are never all in memory;
  * converts the times to UTC (--tz: UTC, a fixed offset, an IANA zone, or 'ny+7' for brokers
    on New York close time: UTC+2 in US winter, UTC+3 in US summer);
  * builds bars (M1 .. D1): time = bar open in UTC epoch seconds; open/high/low/close of the
    chosen price (bid by default, as the Dukascopy research file); tick_volume = number of
    ticks; spread = ask - bid of the bar's last tick; spread_mean = mean ask - bid of the bar;
  * splits at a fixed cutoff (default 2025-09-28 00:00 UTC, the start of your locked holdout
    period): <out>/train/<SYMBOL>_<TF>.parquet gets the bars before it and
    <out>/locked_holdout/<SYMBOL>_<TF>.parquet.locked the rest. Of the locked part only the
    bar count and sha256 are printed and stored; every statistic comes from the train part;
  * writes <out>/<SYMBOL>_<TF>_manifest.json (inputs with sha256, cleaning counts, gaps,
    ticks per bar, spreads, largest ranges, optional comparison with a reference file).

It refuses any input, --compare-with or --out path that contains 'locked_holdout' or ends in
'.locked' (nothing is read), never writes outside --out, and never replaces existing output
files unless --overwrite is given; --overwrite replaces only files this tool made (its
manifest must match them). By default it lowers its own process priority so a running
research grid keeps the CPU. Exit code 0 = written (or dry run done), 2 = usage error,
3 = data problem. Research only.
"""
from __future__ import annotations

import argparse
import codecs
import fnmatch
import gzip
import hashlib
import io
import json
import locale
import math
import os
import platform
import re
import subprocess
import sys
import warnings
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BANNER = "RESEARCH ONLY - not trading advice"
TOOL = "scripts/research/ticks_to_bars.py"
VERSION = "1.0.0"
EXIT_OK, EXIT_USAGE, EXIT_DATA = 0, 2, 3

DEFAULT_CUTOFF = "2025-09-28T00:00:00Z"            # the start of the locked holdout period (owner decision)
DEFAULT_CUTOFF_S = 1759017600                      # 2025-09-28 00:00:00 UTC
DEFAULT_CUTOFF_DAY = DEFAULT_CUTOFF[:10]
TF_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}
FORMATS = ("auto", "mt5", "dukascopy-node", "jforex", "generic")
INPUT_SUFFIXES = (".csv", ".txt", ".csv.gz", ".parquet")
ARCHIVE_SUFFIXES = (".zip", ".7z", ".rar", ".gz", ".tar", ".tgz", ".bz2", ".xz")
ARCHIVE_MAGIC = {b"PK\x03\x04": "zip", b"PK\x05\x06": "zip", b"7z\xbc\xaf\x27\x1c": "7z", b"Rar!": "rar"}
EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xlsb", ".xls")
OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"    # old binary Office files (.xls)
OVERLAP_MODES = ("refuse", "drop", "keep")
COVER_NS = 60 * 1_000_000_000                      # overlap coverage: a whole clock minute without ticks splits it
MIN_BARS_ALPHAMASTER = 300                         # config.Config.MIN_BARS
SPREAD_QUANTUM = 1e-8                              # spread sums are exact integers in these units
VOLUME_QUANTUM = 1e-6
NS = 1_000_000_000
NAT = np.iinfo(np.int64).min                       # numpy's NaT as int64
TIME_MIN_NS = 631152000 * NS                       # 1990-01-01 UTC
TIME_MAX_NS = 4102444800 * NS                      # 2100-01-01 UTC
CLIP_LO = np.datetime64("1900-01-01T00:00:00")     # parsed times outside 1900..2200 are clipped to these
CLIP_HI = np.datetime64("2200-01-01T00:00:00")     # (then dropped as time_out_of_range), never overflow
LOCAL_SLACK_NS = 14 * 3600 * NS                    # local time can differ from UTC by up to 14 h
MT5_CARRY_GAP_NS = 96 * 3600 * NS                  # MT5 blank-cell carry across files only over gaps <= 96 h
DROP_WARN_SHARE = 0.001                            # problem drops above 0.1 % of rows: a warning
DROP_FAIL_SHARE = 0.05                             # above 5 % (overall or in one file): refused without --allow-drops
DRY_RUN_ROWS = 10_000
SNIFF_ROWS = 200
GAP_LONG_HOURS = 72
COMPARE_LAGS = range(-3, 4)
MIN_PAIRS = 30
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
FORBIDDEN_PART = "locked_holdout"
FORBIDDEN_SUFFIX = ".locked"
NY_ZONE = "America/New_York"
EU_ZONE = "Europe/Berlin"                         # EU daylight-saving dates (all EU zones switch together)

MT5_TIME_FORMATS = ("%Y.%m.%d %H:%M:%S.%f", "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M")
JFOREX_TIME_FORMATS = ("%d.%m.%Y %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M")
JFOREX_OFFSET_FORMATS = ("%d.%m.%Y %H:%M:%S.%f GMT%z", "%d.%m.%Y %H:%M:%S GMT%z")
_JFOREX_OFFSET_RE = re.compile(r"\sGMT[+-]\d{2}:?\d{2}\s*$")
_ISO_RE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}[T ]\d{1,2}:\d{2}")
_ISO_OFFSET_RE = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})\s*$")
_MT5STR_RE = re.compile(r"^\s*\d{4}\.\d{2}\.\d{2}[ T]\d{1,2}:\d{2}")
_NUMBER_RE = re.compile(r"^\s*-?\d+(\.\d*)?([eE][+-]?\d+)?\s*$")
_SYMBOL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-_]{0,39}")

# Why a row was dropped; the first reason that applies wins. 0 = kept.
REASONS = ("kept", "unparsable_time", "mt5_no_bid_or_ask_change", "unparsable_price",
           "mt5_before_first_quote", "dst_nonexistent_or_ambiguous_time", "time_out_of_range",
           "missing_price", "nonpositive_price", "crossed_quote", "overlap_with_earlier_file")
R = {name: i for i, name in enumerate(REASONS)}
REASON_HELP = {
    "unparsable_time": "the time could not be read",
    "mt5_no_bid_or_ask_change": "MT5 rows with both BID and ASK blank (only LAST/VOLUME changed)",
    "unparsable_price": "a bid or ask cell held text that is not a number",
    "mt5_before_first_quote": "MT5 rows before both bid and ask were known (start of the stream)",
    "dst_nonexistent_or_ambiguous_time": "local time that does not exist or exists twice at a daylight-saving "
                                         "change (NaT); markets are normally shut then",
    "time_out_of_range": "UTC time before 1990 or from 2100 on (a wrong epoch unit or garbage)",
    "missing_price": "bid or ask missing / not finite",
    "nonpositive_price": "bid or ask <= 0",
    "crossed_quote": "ask below bid",
    "overlap_with_earlier_file": "--overlap drop: an earlier input file already has ticks right around that time "
                                 "(in that clock minute), so it is probably the same tick twice",
}
# drops that point at a wrong format, column, time zone or broken data (not the harmless MT5 kinds)
PROBLEM_REASONS = ("unparsable_time", "unparsable_price", "dst_nonexistent_or_ambiguous_time",
                   "time_out_of_range", "missing_price", "nonpositive_price", "crossed_quote")

MT5_TZ_HELP = """\
MetaTrader 5 tick exports are in your broker's SERVER time, not UTC, so --tz is required.
How to find it: the MT5 Market Watch window shows the server time; compare it with the current
UTC time (search the web for 'UTC time now'). Most CFD brokers run on 'New York close' time:
UTC+2 in winter and UTC+3 in summer, switching on the US daylight-saving dates; for those pass
--tz ny+7. A broker on a fixed offset: --tz +02:00. A broker on UTC: --tz UTC. A broker that
switches on the EUROPEAN dates (last Sunday of March / October): --tz Europe/Athens (UTC+2/+3).
A negative offset: --tz -05:00 (or --tz=-05:00).
Check the result with --dry-run (sample rows in local and UTC time, with the weekday) and with
--compare-with <the Dukascopy train file> (at --tf H1)."""

HELP_EPILOG = r"""
examples (Windows PowerShell, from the AlphaMaster folder):
  # look first: format, encoding, time zone, 5 sample rows, nothing is written
  python scripts\research\ticks_to_bars.py --input D:\ticks --out D:\research\mydata --tz ny+7 --dry-run

  # MetaTrader 5 tick exports (a folder of monthly files) -> H1 bars, broker on New York close time
  python scripts\research\ticks_to_bars.py --input D:\ticks --out D:\research\mydata --tz ny+7

  # the same, checked against the Dukascopy research file (time zone and price level)
  python scripts\research\ticks_to_bars.py --input D:\ticks --out D:\research\mydata --tz ny+7 `
      --compare-with D:\research\data\train\XAUUSD_H1.parquet --overwrite

  # dukascopy-node tick CSVs (UTC), several files by pattern (quote the pattern)
  python scripts\research\ticks_to_bars.py --input "D:\download\xauusd-tick-*.csv" --out D:\research\duka_ticks

  # any other CSV: name the columns, the time format and the time zone
  python scripts\research\ticks_to_bars.py --input D:\ticks\gold.csv --out D:\research\other --format generic `
      --time-col Time --bid-col Bid --ask-col Ask --time-format "%Y-%m-%d %H:%M:%S.%f" --tz UTC

inputs: --input takes a file, a folder (every *.csv, *.txt, *.csv.gz and *.parquet in it,
  sorted by name) or a pattern; give --input several times for several files or folders.
  Extract .zip files first; save Excel workbooks as CSV first. Files that hold ticks at the same
  times (e.g. two exports that both contain one day) are refused (see --overlap); a file that
  only fills a gap in another file is fine.
formats: mt5 (header <DATE> <TIME> <BID> <ASK> ...; needs --tz), dukascopy-node
  (timestamp,askPrice,bidPrice in epoch ms UTC), jforex ('Gmt time,Ask,Bid,...' in UTC, or
  'Local time,...' which needs --tz), generic (--time-col --bid-col --ask-col; --tz unless the
  times carry an offset such as +02:00 or Z). auto (default) reads the header and says what it found.
time zones: UTC, +02:00 or -05:00, an IANA name such as Europe/Athens (brokers that switch on
  the European dates), or ny+7 (broker time = New York time + 7 h: UTC+2 in US winter, UTC+3 in
  US summer; most MT5 CFD brokers).
outputs: <out>\train\<SYMBOL>_<TF>.parquet (what AlphaMaster reads), <out>\locked_holdout\
  <SYMBOL>_<TF>.parquet.locked (bars from the cutoff on; never open it) and
  <out>\<SYMBOL>_<TF>_manifest.json. Columns: time (int64 UTC epoch seconds, bar open), open,
  high, low, close, tick_volume (NUMBER of ticks per bar, not traded volume), spread (ask - bid
  of the last tick), spread_mean (mean ask - bid), and volume_sum when --vol-col is given.
cutoff: bars opening before 2025-09-28 00:00 UTC go to train, the rest to the locked file (a
  tick at 23:59:59.999 on 2025-09-27 is train, one at 00:00:00 on 2025-09-28 is locked).
exit codes: 0 = done, 2 = usage error (nothing written), 3 = data problem (nothing written).
"""

CHUNKSIZE_HELP = ("rows read at a time (default 2,000,000: roughly 0.4-0.7 GB of memory at peak; 500000 needs "
                  "about 0.2-0.3 GB). The finished bars add about 150 bytes each: little at H1, about 0.3 GB for "
                  "5 years of M1 (use H1, or convert in yearly pieces, if memory is short). The bars do not depend "
                  "on it")


class UsageError(Exception):
    """Bad flags or paths: nothing is written and the exit code is 2."""


class DataError(Exception):
    """A problem with the data itself: nothing is written and the exit code is 3."""


# what reading a damaged or misread file can raise (turned into a DataError naming the file)
READ_ERRORS = (ValueError, TypeError, KeyError, OverflowError, UnicodeError, pd.errors.ParserError)


# ---------------------------------------------------------------------------------------
# small helpers

def _ascii(value) -> str:
    """Plain-ASCII text for the console (non-ASCII becomes a \\u escape)."""
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def _qp(value) -> str:
    """A path or value in single quotes for messages (Windows backslashes are not doubled)."""
    return "'" + _ascii(value) + "'"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _plain(value):
    """numpy scalars/arrays -> JSON-safe python values (non-finite floats -> None)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    return value


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        # --no-optional-locks: 'git status' must not refresh (write) the repository's index
        dirty = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, capture_output=True, text=True, timeout=10,
                               env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
        return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def _iso_s(t_s: int) -> str:
    """UTC epoch seconds -> 'YYYY-MM-DDTHH:MM:SSZ'."""
    return datetime.fromtimestamp(int(t_s), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_ns(t_ns: int) -> str:
    if t_ns == NAT:
        return "NaT"
    return pd.Timestamp(int(t_ns), unit="ns").strftime("%Y-%m-%d %H:%M:%S.%f")[:23]


def _q(x, q: float) -> float | None:
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else None


def _fmt(x, spec: str = ".4f") -> str:
    return "n/a" if x is None else format(x, spec)


def is_forbidden(path_text) -> bool:
    """True for a path the research protocol must never read or write (the locked holdout)."""
    candidates = [str(path_text)]
    try:
        candidates.append(str(Path(path_text).resolve()))
    except (OSError, RuntimeError):
        pass
    for text in candidates:
        low = text.replace("\\", "/").lower()
        if FORBIDDEN_PART in low or low.rstrip("/").endswith(FORBIDDEN_SUFFIX):
            return True
    return False


def _norm(name) -> str:
    return str(name).strip().lstrip("\ufeff").strip().strip("\"'").strip().lower()


def lower_priority() -> str:
    """Lower this process's own CPU priority; returns the line to print (a failure is a warning)."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.SetPriorityClass.restype = wintypes.BOOL
            if not k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS):
                raise OSError(ctypes.get_last_error(), "SetPriorityClass failed")
            return ("priority: this process runs at BELOW_NORMAL priority so other work (e.g. a research "
                    "grid) keeps the CPU; --normal-priority skips this")
        os.nice(10)
        _lower_other_threads()
        return "priority: this process runs at nice +10 (lower priority); --normal-priority skips this"
    except Exception as e:  # noqa: BLE001 - any failure here is only a warning
        return f"WARNING: could not lower the process priority ({_ascii(e)}); continuing at normal priority"


def _lower_other_threads() -> None:
    """Linux: nice() changes only the calling thread, so give the threads numpy / pyarrow started
    at import time the same (lower) priority. Best effort; other systems need nothing."""
    task_dir = "/proc/self/task"
    if not sys.platform.startswith("linux") or not os.path.isdir(task_dir):
        return
    try:
        target = os.getpriority(os.PRIO_PROCESS, 0)       # this (main) thread, after nice()
        for tid in os.listdir(task_dir):
            try:
                if os.getpriority(os.PRIO_PROCESS, int(tid)) < target:
                    os.setpriority(os.PRIO_PROCESS, int(tid), target)
            except (OSError, ValueError):
                pass
    except OSError:
        pass


# ---------------------------------------------------------------------------------------
# time zones

@dataclass(frozen=True)
class TzSpec:
    text: str
    kind: str                   # 'utc' | 'fixed' | 'iana' | 'ny'
    offset_s: int = 0           # fixed: local = UTC + offset
    zone: str | None = None     # iana
    ny_hours: int = 0           # ny: local = New York wall clock + ny_hours

    def describe(self) -> str:
        if self.kind == "utc":
            return "UTC"
        if self.kind == "fixed":
            return f"fixed offset UTC{_offset_text(self.offset_s)} (no daylight saving)"
        if self.kind == "ny":
            w, s = self.ny_hours - 5, self.ny_hours - 4
            return (f"ny{self.ny_hours:+d}: broker time = New York wall-clock time {self.ny_hours:+d} h "
                    f"(UTC{w:+d} while the US is on standard time, UTC{s:+d} during US daylight saving)")
        return f"IANA zone {self.zone} (with its daylight-saving rules)"


def _offset_text(offset_s: int) -> str:
    sign = "+" if offset_s >= 0 else "-"
    m = abs(offset_s) // 60
    return f"{sign}{m // 60:02d}:{m % 60:02d}"


def parse_tz(text: str) -> TzSpec:
    t = str(text).strip()
    if t.upper() in ("UTC", "GMT", "Z", "ETC/UTC", "UTC0"):
        return TzSpec(t, "utc")
    m = re.fullmatch(r"(?:UTC|GMT)?\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?", t, re.IGNORECASE)
    if m:
        hours, minutes = int(m.group(2)), int(m.group(3) or 0)
        off = (hours * 3600 + minutes * 60) * (1 if m.group(1) == "+" else -1)
        if minutes >= 60 or abs(off) > 14 * 3600:
            raise UsageError(f"--tz {_ascii(t)}: the offset must be between -14:00 and +14:00")
        return TzSpec(t, "utc") if off == 0 else TzSpec(t, "fixed", offset_s=off)
    m = re.fullmatch(r"ny\s*([+-])\s*(\d{1,2})", t, re.IGNORECASE)
    if m:
        hours = int(m.group(2)) * (1 if m.group(1) == "+" else -1)
        if abs(hours) > 19:
            raise UsageError(f"--tz {_ascii(t)}: ny+N needs N between -19 and +19 (most brokers: ny+7)")
        return TzSpec(t, "ny", ny_hours=hours)
    try:
        pd.Timestamp("2020-01-01").tz_localize(t)
    except Exception:  # noqa: BLE001 - unknown zone names raise several types
        raise UsageError(f"--tz {_qp(t)} is not a time zone this tool knows. Use UTC, a fixed offset such as "
                         "+02:00, an IANA name such as Europe/Athens, or ny+7 (broker time = New York + 7 h, "
                         "UTC+2 winter / UTC+3 summer)")
    return TzSpec(t, "iana", zone=t)


def _dt64_to_ns(arr) -> np.ndarray:
    """numpy datetime64 of any unit -> int64 ns (NaT = NAT). Times outside 1900..2200 are clipped to
    those bounds, so they are dropped later as time_out_of_range instead of overflowing the ns range
    (pandas 3 parses e.g. a garbage year 2300 into datetime64[us]; casting that to ns overflows)."""
    arr = np.asarray(arr)
    if arr.dtype.kind != "M":
        raise TypeError(f"expected datetime64 values, got {arr.dtype}")
    if arr.dtype != np.dtype("datetime64[ns]"):
        lo, hi = CLIP_LO.astype(arr.dtype), CLIP_HI.astype(arr.dtype)
        arr = arr.copy()
        arr[arr < lo] = lo                     # NaT compares False: it stays NaT
        arr[arr > hi] = hi
        arr = arr.astype("datetime64[ns]")
    return arr.view("int64").copy()


def _dt_to_ns(values) -> np.ndarray:
    """datetime-like Series/Index (naive, or aware -> UTC) -> int64 ns, NaT = NAT."""
    if isinstance(values, pd.DatetimeIndex):
        idx = values
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        return _dt64_to_ns(idx.to_numpy())
    s = pd.Series(values)
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return _dt64_to_ns(s.to_numpy())


def to_utc_ns(local_ns: np.ndarray, tz: TzSpec) -> np.ndarray:
    """Naive local wall-clock times (int64 ns) -> UTC int64 ns; NAT where the local time does not
    exist or is ambiguous (daylight-saving changes) or was NAT already."""
    local_ns = np.asarray(local_ns, dtype="int64")
    valid = local_ns != NAT
    if tz.kind == "utc":
        return local_ns.copy()
    out = np.full(len(local_ns), NAT, dtype="int64")
    if tz.kind == "fixed":
        out[valid] = local_ns[valid] - tz.offset_s * NS
        return out
    if tz.kind == "ny":
        zone, shift = NY_ZONE, tz.ny_hours * 3600 * NS
    else:
        zone, shift = tz.zone, 0
    src = np.where(valid, local_ns - shift, NAT)
    idx = pd.DatetimeIndex(src.view("datetime64[ns]"))
    loc = idx.tz_localize(zone, ambiguous="NaT", nonexistent="NaT")
    return _dt_to_ns(loc)


def _parse_strings(s: pd.Series, formats, utc: bool = False) -> np.ndarray:
    """Parse time strings with the first matching format of `formats` (per row) -> int64 ns."""
    n = len(s)
    out = np.full(n, NAT, dtype="int64")
    todo = np.array(s.notna().to_numpy(), dtype=bool)          # a writable copy (pandas 3 returns read-only)
    for fmt in formats:
        if not todo.any():
            break
        pos = np.flatnonzero(todo)
        sub = s if len(pos) == n else s.iloc[pos]          # no copy on the first (usual) pass
        if fmt == "ISO8601":
            parsed = pd.to_datetime(sub, format="ISO8601", errors="coerce", utc=utc)
        else:
            parsed = pd.to_datetime(sub, format=fmt, errors="coerce", utc=utc)
        ns = _dt_to_ns(parsed)
        ok = ns != NAT
        out[pos[ok]] = ns[ok]
        todo[pos[ok]] = False
    return out


_EPOCH_FACTOR = {"s": NS, "ms": 1_000_000, "us": 1_000, "ns": 1}


def _epoch_unit(values: np.ndarray) -> str | None:
    v = np.abs(np.asarray(values, dtype="float64"))
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    med = float(np.median(v))
    return "ns" if med > 1e17 else "us" if med > 1e14 else "ms" if med > 1e11 else "s"


_INT_TEXT_RE = r"[+-]?(?:\d{1,18}|[1-8]\d{18})"           # whole numbers that fit int64 exactly


def _epoch_to_ns(col: pd.Series, unit: str) -> np.ndarray:
    """Epoch numbers -> int64 ns. Whole numbers (int columns, nullable or not, and whole-number text)
    are converted exactly; only decimals go through float64."""
    factor = _EPOCH_FACTOR[unit]
    limit = 9.0e18 / factor
    out = np.full(len(col), NAT, dtype="int64")
    if pd.api.types.is_integer_dtype(col):
        na = np.asarray(col.isna().to_numpy(), dtype=bool)
        v = col.to_numpy(dtype="int64", na_value=0) if na.any() else col.to_numpy(dtype="int64")
        ok = ~na & (v > -limit) & (v < limit)
        out[ok] = v[ok] * factor
        return out
    if pd.api.types.is_float_dtype(col):
        _float_epoch(col.to_numpy(dtype="float64", na_value=np.nan), factor, limit, out, np.arange(len(col)))
        return out
    txt = col.astype(object).where(col.notna(), "").astype(str).str.strip()
    whole = np.asarray(txt.str.fullmatch(_INT_TEXT_RE).to_numpy(), dtype=bool)
    if whole.any():
        v = pd.to_numeric(txt[whole]).to_numpy(dtype="int64")
        ok = (v > -limit) & (v < limit)
        pos = np.flatnonzero(whole)
        out[pos[ok]] = v[ok] * factor
    rest = ~whole
    if rest.any():
        f = pd.to_numeric(txt[rest], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
        _float_epoch(f, factor, limit, out, np.flatnonzero(rest))
    return out


def _float_epoch(f: np.ndarray, factor: int, limit: float, out: np.ndarray, pos: np.ndarray) -> None:
    """Epoch numbers held as float64 -> int64 ns into out[pos]; whole numbers are converted exactly (a
    column turns float64 when it has one blank cell), only fractions are rounded."""
    ok = np.isfinite(f) & (np.abs(f) < limit)
    whole = ok & (np.abs(f) < 2.0 ** 53) & (f == np.floor(np.where(ok, f, 0.0)))
    out[pos[whole]] = f[whole].astype("int64") * factor
    frac = ok & ~whole
    out[pos[frac]] = np.rint(f[frac] * factor).astype("int64")


# ---------------------------------------------------------------------------------------
# inputs: discovery and sniffing

@dataclass
class InputInfo:
    path: Path
    size: int
    kind: str = "text"                # 'text' | 'parquet'
    compression: str | None = None
    encoding: str | None = None
    encoding_note: str = ""
    sep: str | None = None
    columns: list = field(default_factory=list)
    fmt: str = ""
    fmt_note: str = ""
    roles: dict = field(default_factory=dict)
    time_kind: str = ""
    time_format: str | None = None
    aware: bool = False               # the times carry their own offset -> UTC directly
    jforex_header: str | None = None  # 'gmt' | 'local'
    default_utc: bool = False         # the times are UTC by definition (Gmt time, dukascopy-node epochs)
    names: list | None = None         # --no-header: the column names given to the file (col0, col1, ...)
    tz: TzSpec | None = None
    tz_note: str = ""
    epoch_unit: str | None = None
    sha256: str | None = None

    def sep_name(self) -> str:
        return {"\t": "tab", ",": "comma", ";": "semicolon", "|": "pipe", " ": "space"}.get(self.sep or "",
                                                                                          _ascii(repr(self.sep)))


def _suffix_ok(p: Path) -> bool:
    name = p.name.lower()
    return any(name.endswith(s) for s in INPUT_SUFFIXES)


def _archive_kind(name: str) -> str | None:
    low = name.lower()
    if low.endswith(INPUT_SUFFIXES):
        return None
    return next((suf for suf in ARCHIVE_SUFFIXES if low.endswith(suf)), None)


def _is_excel_name(name: str) -> bool:
    return name.lower().endswith(EXCEL_SUFFIXES)


EXCEL_ADVICE = ("open it in Excel, File > Save As > 'CSV (Comma delimited) (*.csv)', then pass the .csv file to "
                "--input")


def _skipped_advice(skipped: list[Path]) -> str:
    """What to do about skipped archives and Excel workbooks ('' when there are none)."""
    arch = [m for m in skipped if _archive_kind(m.name)]
    xls = [m for m in skipped if _is_excel_name(m.name)]
    parts = []
    if arch:
        parts.append(f"{len(arch)} archive file(s) were NOT converted (" + ", ".join(_ascii(m.name) for m in arch[:10])
                     + "). Extract them first (Windows: right-click > Extract All) and pass the extracted files: "
                       "the ticks inside an archive are not read")
    if xls:
        parts.append(f"{len(xls)} Excel workbook(s) were NOT converted (" + ", ".join(_ascii(m.name) for m in xls[:10])
                     + "): " + EXCEL_ADVICE.replace("open it", "open each").replace("the .csv file", "the .csv files"))
    return ". ".join(parts)


_MAGIC_RE = re.compile(r"[*?[]")


def _refuse_locked_entry(flag_value: str, path) -> None:
    raise UsageError(f"refused: --input {_qp(flag_value)} reaches {_qp(path)}, which is (in) the locked holdout "
                     f"('{FORBIDDEN_PART}' or '*{FORBIDDEN_SUFFIX}'); this tool never reads or lists it. Point --input "
                     "at a folder or files that hold only your tick files")


def _glob_no_locked(pattern: str) -> list[Path]:
    """glob.glob() without ever listing a locked_holdout folder: the pattern is expanded one path part at a
    time, and anything it would enter or match that is in the locked holdout is refused (UsageError)."""
    p = Path(pattern)
    parts = list(p.parts)
    if p.anchor:
        current, parts = [Path(p.anchor)], parts[1:]
    else:
        current = [Path(".")]
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        nxt: list[Path] = []
        for folder in current:
            if not _MAGIC_RE.search(part):
                cand = folder / part
                if is_forbidden(cand):
                    _refuse_locked_entry(pattern, cand)
                if cand.is_dir() or (last and cand.exists()):
                    nxt.append(cand)
                continue
            if is_forbidden(folder):
                _refuse_locked_entry(pattern, folder)
            try:
                names = sorted(e.name for e in os.scandir(folder))
            except OSError:
                continue
            for name in names:
                if name.startswith(".") and not part.startswith("."):
                    continue
                if not fnmatch.fnmatch(name, part):
                    continue
                cand = folder / name
                if is_forbidden(cand):                 # checked before the folder could be entered or listed
                    _refuse_locked_entry(pattern, cand)
                if last or cand.is_dir():
                    nxt.append(cand)
        current = nxt
    return sorted(current)


def expand_inputs(texts: list[str]) -> tuple[list[Path], list[str], list[str]]:
    """Files from --input values (files, folders, patterns); refuses locked-holdout paths, including any
    locked-holdout file or folder a folder or pattern reaches (such folders are never listed).
    Returns (files, notes, warnings)."""
    files: list[Path] = []
    notes: list[str] = []
    warns: list[str] = []

    def skipped_note(where: str, skipped: list[Path]) -> str:
        if not skipped:
            return ""
        names = ", ".join(_ascii(m.name) for m in skipped[:10]) + (" ..." if len(skipped) > 10 else "")
        notes.append(f"{where}: skipped {len(skipped)} file(s) that are not {', '.join(INPUT_SUFFIXES)}: {names}")
        advice = _skipped_advice(skipped)
        if advice:
            warns.append(f"{where}: {advice}")
        return advice

    for text in texts:
        if is_forbidden(text):
            raise UsageError(f"refused: --input {_qp(text)} points at the locked holdout ('{FORBIDDEN_PART}' or "
                             f"'*{FORBIDDEN_SUFFIX}'); this tool never reads it")
        if _MAGIC_RE.search(text) and not Path(text).exists():
            matches = _glob_no_locked(text)
            found = [m for m in matches if m.is_file() and _suffix_ok(m)]
            advice = skipped_note(f"--input {_ascii(text)}", [m for m in matches if m.is_file() and not _suffix_ok(m)])
            if not found:
                raise UsageError(f"--input {_qp(text)}: no tick files ({', '.join(INPUT_SUFFIXES)}) match this "
                                 "pattern" + (f". {advice}" if advice else
                                              " (tick files inside a .zip must be extracted first: right-click > "
                                              "Extract All)"))
        else:
            p = Path(text)
            if p.is_dir():
                entries = sorted(p.iterdir(), key=lambda c: c.name)
                for c in entries:                     # files and folders, before any is filtered by its name
                    if is_forbidden(c):
                        raise UsageError(f"refused: the --input folder {_qp(text)} holds {_qp(c.name)}, which is "
                                         f"(in) the locked holdout ('{FORBIDDEN_PART}' or '*{FORBIDDEN_SUFFIX}'); this "
                                         "tool never reads it or anything next to it. Put your tick files in a "
                                         "folder of their own and pass that folder")
                kids = [c for c in entries if c.is_file()]
                found = [c for c in kids if _suffix_ok(c)]
                advice = skipped_note(f"--input {_ascii(text)}", [c for c in kids if not _suffix_ok(c)
                                                                  and not c.name.startswith(".")])
                if not found:
                    raise UsageError(f"--input {_qp(text)}: the folder has no {', '.join(INPUT_SUFFIXES)} files"
                                     + (f". {advice}" if advice else ""))
            elif p.is_file():
                found = [p]
            else:
                raise UsageError(f"--input {_qp(text)}: file or folder not found")
        for f in found:
            if is_forbidden(str(f)):
                raise UsageError(f"refused: input file {_qp(f)} is in the locked holdout ('{FORBIDDEN_PART}' "
                                 f"or '*{FORBIDDEN_SUFFIX}'); this tool never reads it")
            files.append(f.resolve())
    unique: list[Path] = []
    seen: set[str] = set()
    for f in files:
        key = os.path.normcase(str(f))
        if key in seen:
            notes.append(f"{_ascii(f)} was given more than once; it is read once")
            continue
        seen.add(key)
        unique.append(f)
    return unique, notes, warns


def _head_bytes(path: Path, gz: bool, n: int = 65536) -> bytes:
    if gz:
        with gzip.open(path, "rb") as f:
            return f.read(n)
    with open(path, "rb") as f:
        return f.read(n)


def _valid_text(head: bytes, encoding: str) -> bool:
    """True when `head` decodes with `encoding` (a character cut at the end of the head is allowed)."""
    for cut in range(4):
        try:
            head[: len(head) - cut].decode(encoding)
            return True
        except UnicodeDecodeError:
            continue
        except LookupError:
            return False
    return False


def detect_encoding(head: bytes, forced: str | None = None) -> tuple[str, str]:
    if forced:
        return forced, f"{forced} (from --encoding)"
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", "UTF-8 with byte-order mark"
    if head.startswith(b"\xff\xfe"):
        return "utf-16", "UTF-16 LE with byte-order mark (usual for MT5 exports)"
    if head.startswith(b"\xfe\xff"):
        return "utf-16", "UTF-16 BE with byte-order mark"
    sample = head[:4096]
    if len(sample) >= 8:
        even, odd = sample[0::2], sample[1::2]
        if odd.count(0) > 0.4 * len(odd) and even.count(0) < 0.05 * len(even):
            return "utf-16-le", "UTF-16 LE without byte-order mark"
        if even.count(0) > 0.4 * len(even) and odd.count(0) < 0.05 * len(odd):
            return "utf-16-be", "UTF-16 BE without byte-order mark"
    if _valid_text(head, "utf-8"):
        return "utf-8", "UTF-8 (no byte-order mark)"
    pref = locale.getpreferredencoding(False) or ""
    if pref and pref.replace("-", "").lower() not in ("utf8", "ascii", "usascii", "ansix3.41968") \
            and _valid_text(head, pref):
        return pref, (f"{pref} (not valid UTF-8, so this computer's own text encoding was used; pass "
                      "--encoding if the names look wrong)")
    return "utf-8", ("UTF-8 with unreadable bytes replaced (the file is not valid UTF-8: pass --encoding, "
                     "e.g. --encoding gbk or --encoding cp1252)")


def _sep_arg(text: str | None) -> str | None:
    if text is None:
        return None
    if text in ("\t", " "):
        return text
    t = text.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        t = t[1:-1]
    named = {"tab": "\t", "\\t": "\t", "comma": ",", "semicolon": ";", "pipe": "|", "space": " "}
    if t.lower() in named:
        return named[t.lower()]
    if len(t) == 1:
        return t
    raise UsageError(f"--sep {_qp(text)}: give one character, or tab / comma / semicolon / pipe / space")


def _guess_sep(header: str) -> str:
    counts = [(header.count(c), -i, c) for i, c in enumerate(("\t", ",", ";", "|"))]
    best = max(counts)
    return best[2] if best[0] > 0 else ","


def _csv_kwargs(info: InputInfo) -> dict:
    kw = {"sep": info.sep, "encoding": info.encoding, "compression": info.compression,
          "encoding_errors": "replace"}
    if info.sep and not info.sep.isascii():
        kw["engine"] = "python"       # pandas' fast reader needs a one-byte separator (it would warn and switch)
    if info.names is not None:
        kw.update(header=None, names=info.names)
    return kw


def _role(columns: list[str], *names: str, strip_brackets: bool = False) -> str | None:
    for c in columns:
        n = _norm(c)
        if strip_brackets:
            n = n.strip("<>")
        if n in names:
            return c
    return None


def _zip_is_office(path: Path) -> bool:
    """True for a zip that is an Excel workbook (e.g. an .xlsx under another name); reads only its file list."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except (OSError, zipfile.BadZipFile, ValueError):
        return False
    return "[Content_Types].xml" in names and any(n.startswith("xl/") for n in names)


def _archive_message(path: Path, kind: str) -> str:
    return (f"{_ascii(path.name)} is a .{kind} archive, not a tick file. Extract it first (Windows: right-click > "
            "Extract All), then pass the extracted .csv file or folder to --input")


def sniff(path: Path, a) -> InputInfo:
    """Detect encoding, separator and format of one input; reads only its first rows."""
    info = InputInfo(path=path, size=path.stat().st_size)
    low = path.name.lower()
    try:
        with open(path, "rb") as f:
            magic = f.read(8)
    except OSError as e:
        raise DataError(f"cannot read {_ascii(path)}: {_ascii(e)}")
    if _is_excel_name(low) or magic.startswith(OLE_MAGIC) or (magic.startswith(b"PK") and _zip_is_office(path)):
        raise UsageError(f"{_ascii(path.name)} is an Excel workbook, not a tick file: {EXCEL_ADVICE}")
    for sig, kind in ARCHIVE_MAGIC.items():
        if magic.startswith(sig):
            raise UsageError(_archive_message(path, kind))
    if low.endswith(".parquet"):
        info.kind = "parquet"
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            info.columns = list(pf.schema_arrow.names)
            types = {n: pf.schema_arrow.field(n).type for n in info.columns}
            sample = next(pf.iter_batches(batch_size=SNIFF_ROWS), None)
            sample = _batch_to_pandas(sample) if sample is not None else pd.DataFrame(columns=info.columns)
        except Exception as e:  # noqa: BLE001 - pyarrow raises several types
            raise DataError(f"cannot read {_ascii(path)} as Parquet: {type(e).__name__}: {_ascii(e)}")
        info.encoding_note = "Parquet"
    else:
        if low.endswith(".gz") and not magic.startswith(b"\x1f\x8b"):
            raise UsageError(f"{_ascii(path.name)} ends in .gz but is not gzip-compressed; rename it or extract it")
        info.compression = "gzip" if low.endswith(".gz") else None
        try:
            head = _head_bytes(path, info.compression == "gzip")
        except OSError as e:
            raise DataError(f"cannot read {_ascii(path)}: {_ascii(e)}")
        info.encoding, info.encoding_note = detect_encoding(head, getattr(a, "encoding", None))
        text = head.decode(info.encoding, errors="replace").lstrip("\ufeff")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise DataError(f"{_ascii(path)} is empty")
        info.sep = _sep_arg(a.sep) or _guess_sep(lines[0])
        if not info.sep.isascii():
            info.encoding_note += "; the separator is not an ASCII character, so the slower Python CSV reader is used"
        if getattr(a, "no_header", False):
            ncols = len(lines[0].split(info.sep))
            info.names = [f"col{i}" for i in range(ncols)]
        try:
            sample = pd.read_csv(path, nrows=SNIFF_ROWS, dtype=str, keep_default_na=False,
                                 **_csv_kwargs(info))
        except Exception as e:  # noqa: BLE001 - pandas raises several types
            raise DataError(f"cannot read {_ascii(path)} as text with {info.sep_name()} separators: "
                            f"{type(e).__name__}: {_ascii(e)}")
        info.columns = [str(c) for c in sample.columns]
        types = {}
    _detect_format(info, a)
    _detect_time_kind(info, a, sample, types)
    return info


def _batch_to_pandas(batch) -> pd.DataFrame:
    """A pyarrow RecordBatch -> DataFrame; int64 columns become nullable Int64, so an epoch column with
    a few nulls is not turned into float64 (which would round nanosecond times)."""
    import pyarrow as pa
    return batch.to_pandas(types_mapper=lambda t: pd.Int64Dtype() if t == pa.int64() else None)


_YEAR_RE = re.compile(r"(?:19|20|21)\d\d")
_DATA_CELL_RE = re.compile(r"^\s*(?:\d{4}[.\-/]?\d{2}[.\-/]?\d{2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})")


def _cutoff_of(a) -> int:
    try:
        return parse_cutoff(getattr(a, "cutoff", None) or DEFAULT_CUTOFF)
    except UsageError:
        return DEFAULT_CUTOFF_S


_TIME_OF_DAY_RE = re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?\s*$")
_VALUE_LIKE_RE = re.compile(r"\d{3,}|\d\s*[./:,\-]\s*\d")
_DIGITS_ONLY_RE = re.compile(r"^[\d\s]+$")
_NOT_SHOWN = "(not shown: it may be from the locked holdout period)"


def _readings(text: str) -> list:
    """Every time a text can be read as (day or month first, year first, MT5 and JForex layouts), as
    naive UTC pd.Timestamps; [] when it reads as none."""
    t = _JFOREX_OFFSET_RE.sub("", text).strip()
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tries = [{"dayfirst": False}, {"dayfirst": True}, {"yearfirst": True}, {"yearfirst": True, "dayfirst": True}]
        tries += [{"format": f} for f in MT5_TIME_FORMATS + JFOREX_TIME_FORMATS]
        for kw in tries:
            try:
                v = pd.to_datetime(t, errors="coerce", **kw)
            except (ValueError, TypeError, OverflowError):
                continue
            if v is pd.NaT or pd.isna(v):
                continue
            if v.tzinfo is not None:
                v = v.tz_convert("UTC").tz_localize(None)
            out.append(v)
    return out


def _value_shown(text, cutoff_s: int) -> bool:
    """True only when a raw value from a file may be printed: it is a bare time of day, or it reads as a
    date/time before the cutoff - 14 h in every way it can be read (day or month first, 2- or 4-digit year).
    Bare numbers (prices, epoch times, compact dates such as 20251103) are never printed."""
    t = str(text).strip()
    if not t or len(t) > 80 or _NUMBER_RE.match(t) or _DIGITS_ONLY_RE.match(t):
        return False
    if _TIME_OF_DAY_RE.match(t):
        return True
    lim_s = min(cutoff_s, DEFAULT_CUTOFF_S) - LOCAL_SLACK_NS // NS
    cut_year = datetime.fromtimestamp(lim_s, timezone.utc).year
    if any(int(y) > cut_year for y in _YEAR_RE.findall(t)):
        return False
    readings = _readings(t)
    return bool(readings) and all(v.value // NS < lim_s for v in readings)


def _show(text, cutoff_s: int) -> str:
    """A raw time value for a message, or a note when it is not certainly from before the cutoff."""
    return _qp(text) if _value_shown(text, cutoff_s) else _NOT_SHOWN


def _cols_text(cols, cutoff_s: int, n: int = 20) -> str:
    """Header names for a message; a cell that looks like a value (a number, a date or a time, as in the first
    line of a file without a header) is shown as (hidden)."""
    def one(c) -> str:
        s = str(c).strip()
        if _NUMBER_RE.match(s) or _VALUE_LIKE_RE.search(s):
            return "(hidden)"
        return s
    return _ascii(", ".join(one(c) for c in list(cols)[:n]))


def _looks_like_data(cols) -> bool:
    """True when the 'header' cells look like values (numbers, dates): the file probably has no header."""
    cells = [str(c).strip() for c in cols if str(c).strip()]
    if len(cells) < 2:
        return False
    hits = sum(bool(_NUMBER_RE.match(c) or _DATA_CELL_RE.match(c)) for c in cells)
    return hits >= max(2, len(cells) - 1)


def _detect_format(info: InputInfo, a) -> None:
    cols = info.columns
    cut = _cutoff_of(a)
    norms = {_norm(c) for c in cols}
    generic_flags = any(getattr(a, k) for k in ("time_col", "bid_col", "ask_col"))
    detected = None
    if {"<date>", "<time>", "<bid>", "<ask>"} <= norms:
        detected = "mt5"
    elif {"timestamp", "askprice", "bidprice"} <= norms:
        detected = "dukascopy-node"
    elif cols and _norm(cols[0]) in ("gmt time", "local time") and {"ask", "bid"} <= norms:
        detected = "jforex"
    fmt = a.format
    if fmt == "auto" and not generic_flags and detected is None \
            and {"open", "high", "low", "close"} <= {n.strip("<>") for n in norms}:
        raise UsageError(f"{_ascii(info.path)} holds BARS (open/high/low/close), not ticks. This tool needs tick "
                         "data. In MT5: View > Symbols (Ctrl+U) > Ticks tab > pick the dates > Request > Export "
                         "ticks. Bar files from Dukascopy are converted by the existing convert_dukascopy.py")
    if fmt == "auto":
        if generic_flags:
            fmt, info.fmt_note = "generic", "generic (because --time-col/--bid-col/--ask-col were given)"
        elif detected:
            fmt, info.fmt_note = detected, f"{detected} (detected from the header)"
        elif {"date", "time", "bid", "ask"} <= {n.strip("<>") for n in norms}:
            dc = _role(cols, "date", strip_brackets=True)
            tc = _role(cols, "time", strip_brackets=True)
            bc = _role(cols, "bid", strip_brackets=True)
            ac = _role(cols, "ask", strip_brackets=True)
            raise UsageError(f"{_ascii(info.path)}: the header ({_cols_text(cols, cut)}) has separate date and time "
                             "columns but not the MT5 <DATE> <TIME> <BID> <ASK> header, which is ambiguous. If it is "
                             "an MT5 export (dates like 2024.01.31), pass --format mt5 with --tz. Otherwise name the "
                             f"columns: --date-col \"{_ascii(dc)}\" --time-col \"{_ascii(tc)}\" --bid-col "
                             f"\"{_ascii(bc)}\" --ask-col \"{_ascii(ac)}\" --time-format \"%d/%m/%Y %H:%M:%S.%f\" "
                             "(adjust it to the dates in the file) --tz ...")
        elif _looks_like_data(cols) and info.names is None:
            raise UsageError(f"{_ascii(info.path)}: the first line has {len(cols)} cells and looks like data, not a "
                             "header row (the values are not shown). Pass --no-header: the columns are then called "
                             "col0, col1, col2, ... e.g. --no-header --time-col col0 --bid-col col1 --ask-col col2 "
                             "--time-format \"%Y%m%d %H%M%S%f\" --tz ...")
        else:
            raise UsageError(f"{_ascii(info.path)}: could not recognise the format (header: "
                             f"{_cols_text(cols, cut, 12)}). Pass --format mt5, dukascopy-node or jforex, or "
                             "--format generic with --time-col, --bid-col and --ask-col (and --time-format / "
                             "--tz as needed)")
    else:
        info.fmt_note = f"{fmt} (from --format)" + ("" if detected in (None, fmt) else
                                                     f"; NOTE: the header looks like {detected}")
    info.fmt = fmt
    need: dict[str, tuple] = {}
    if fmt == "mt5":
        info.roles = {"date": _role(cols, "date", strip_brackets=True),
                      "time": _role(cols, "time", strip_brackets=True),
                      "bid": _role(cols, "bid", strip_brackets=True),
                      "ask": _role(cols, "ask", strip_brackets=True)}
        need = {"date": ("<DATE>",), "time": ("<TIME>",), "bid": ("<BID>",), "ask": ("<ASK>",)}
    elif fmt == "dukascopy-node":
        info.roles = {"ts": _role(cols, "timestamp"), "bid": _role(cols, "bidprice"), "ask": _role(cols, "askprice")}
        need = {"ts": ("timestamp",), "bid": ("bidPrice",), "ask": ("askPrice",)}
    elif fmt == "jforex":
        tcol = _role(cols, "gmt time", "local time")
        info.roles = {"ts": tcol, "bid": _role(cols, "bid"), "ask": _role(cols, "ask")}
        need = {"ts": ("Gmt time or Local time",), "bid": ("Bid",), "ask": ("Ask",)}
        if tcol is not None:
            info.jforex_header = "gmt" if _norm(tcol) == "gmt time" else "local"
    else:
        missing_flags = [f"--{k.replace('_', '-')}" for k in ("time_col", "bid_col", "ask_col") if not getattr(a, k)]
        if missing_flags:
            raise UsageError(f"--format generic needs {', '.join(missing_flags)} (the column names; "
                             f"{_ascii(info.path.name)} has: {_cols_text(cols, cut)})")
        info.roles = {}
        for role, flag in (("date", "date_col"), ("ts", "time_col"), ("bid", "bid_col"), ("ask", "ask_col"),
                           ("vol", "vol_col")):
            want = getattr(a, flag, None)
            if not want:
                continue
            exact = [c for c in cols if c == want]
            loose = [c for c in cols if _norm(c) == _norm(want)]
            match = exact or loose
            if len(match) != 1:
                what = "is not" if not match else "matches more than one column"
                raise UsageError(f"--{flag.replace('_', '-')} {_qp(want)} {what} in {_ascii(info.path.name)} "
                                 f"(columns: {_cols_text(cols, cut)})"
                                 + ("; if the file has no header row, add --no-header (columns col0, col1, ...)"
                                    if info.names is None and _looks_like_data(cols) else "")
                                 + ("; the column names look garbled, so the file is probably not UTF-8: pass "
                                    "--encoding (e.g. --encoding gbk or --encoding cp1252)"
                                    if any("\ufffd" in str(c) for c in cols) else ""))
            info.roles[role] = match[0]
        named = [info.roles[r] for r in ("date", "ts", "bid", "ask") if r in info.roles]
        if len(set(named)) < len(named):
            raise UsageError("--date-col, --time-col, --bid-col and --ask-col must name different columns")
    missing = [need[r][0] for r, c in info.roles.items() if c is None and r in need]
    if missing:
        raise UsageError(f"{_ascii(info.path.name)} is not a {fmt} file: column(s) {', '.join(missing)} not found "
                         f"(columns: {_cols_text(cols, cut)})")


def _clean_text(col: pd.Series) -> pd.Series:
    s = col.astype(str).str.strip()
    return s[(s != "") & (s.str.lower() != "nan") & (s.str.lower() != "none") & (s.str.lower() != "<na>")]


def _bad_time_format(fmt: str, err) -> str:
    return (f"--time-format {_qp(fmt)} is not a valid time format ({_ascii(err)}). Use the strftime codes: %Y year, "
            "%m month, %d day, %H hour, %M minute, %S second, %f fractions of a second (.123 or .123456; there is no "
            "%L), e.g. \"%d.%m.%Y %H:%M:%S.%f\"; a literal % is written %%")


def _detect_text_time(info: InputInfo, a, s: pd.Series, what: str) -> None:
    """Time kind of text times (generic, or dukascopy-node written with --date-format)."""
    cut = _cutoff_of(a)
    if a.time_format:
        info.time_kind, info.time_format = "format", a.time_format
        info.aware = "%z" in a.time_format or "%Z" in a.time_format
        try:
            parsed = pd.to_datetime(s, format=a.time_format, errors="coerce", utc=info.aware)
        except (ValueError, TypeError) as e:
            raise UsageError(_bad_time_format(a.time_format, e))
        if parsed.isna().mean() > 0.5:
            raise UsageError(f"--time-format {_qp(a.time_format)} does not match the times in "
                             f"{_ascii(info.path.name)} (first value: {_show(s.iloc[0], cut)})")
        return
    # the kind most of the sampled values have (a stray garbage row is later counted as unparsable_time)
    num = np.asarray(s.str.match(_NUMBER_RE).to_numpy(), dtype=bool)
    if num.mean() > 0.5:
        info.time_kind = "epoch"
        info.epoch_unit = _epoch_unit(pd.to_numeric(s[num], errors="coerce"))
        return
    iso = np.asarray(s.str.match(_ISO_RE).to_numpy(), dtype=bool)
    if iso.mean() > 0.5:
        s = s[iso]
        has_off = s.str.contains(_ISO_OFFSET_RE)
        if has_off.all():
            info.time_kind, info.aware = "iso_offset", True
            _check_sample(info, a, s, ("ISO8601",), "ISO 8601 with an offset", utc=True)
        elif not has_off.any():
            info.time_kind = "iso"
            _check_sample(info, a, s, ("ISO8601",), "ISO 8601")
        else:
            raise UsageError(f"{_ascii(info.path.name)}: some times carry a UTC offset and some do not; "
                             "pass --time-format")
        return
    if s.str.match(_MT5STR_RE).mean() > 0.5:
        info.time_kind = "mt5str"
        _check_sample(info, a, s, MT5_TIME_FORMATS, "YYYY.MM.DD HH:MM:SS.fff")
        return
    raise UsageError(f"{_ascii(info.path.name)}: cannot tell the time format of {what} (first value: "
                     f"{_show(s.iloc[0], cut)}); pass --time-format, e.g. \"%d.%m.%Y %H:%M:%S.%f\" (day first) or "
                     "\"%m/%d/%Y %H:%M:%S\"")


def _detect_time_kind(info: InputInfo, a, sample: pd.DataFrame, types: dict) -> None:
    r = info.roles
    if info.fmt == "mt5":
        info.time_kind = "mt5"
        raw = sample[r["date"]].astype(str).str.strip() + " " + sample[r["time"]].astype(str).str.strip()
        _check_sample(info, a, raw, MT5_TIME_FORMATS, "YYYY.MM.DD HH:MM:SS.fff")
        return
    col = sample[r["ts"]]
    typ = types.get(r["ts"]) if info.kind == "parquet" else None
    if typ is not None:
        import pyarrow as pa
        if pa.types.is_timestamp(typ):
            info.time_kind, info.aware = ("datetime_aware", True) if typ.tz else ("datetime", False)
            return
        if pa.types.is_integer(typ) or pa.types.is_floating(typ):
            info.time_kind = "epoch"
            info.epoch_unit = _epoch_unit(pd.to_numeric(col, errors="coerce"))
            info.default_utc = info.fmt == "dukascopy-node"
            return
    if info.fmt == "dukascopy-node":
        s = _clean_text(col)
        if not len(s) or s.str.match(_NUMBER_RE).mean() > 0.5:
            info.time_kind, info.default_utc = "epoch", True
            info.epoch_unit = _epoch_unit(pd.to_numeric(s, errors="coerce")) if len(s) else None
            return
        if s.str.match(_ISO_RE).mean() > 0.5 and not a.time_format:
            _detect_text_time(info, a, s, "the timestamp column")
            info.fmt_note += "; times written as text (dukascopy-node --date-format)"
            return
        raise UsageError(f"{_ascii(info.path.name)}: the dukascopy-node 'timestamp' column holds text, not epoch "
                         "milliseconds (a download made with dukascopy-node's --date-format option). Convert it as "
                         "a generic file: --format generic --time-col timestamp --bid-col bidPrice --ask-col askPrice "
                         "--time-format \"<the format of the times>\" --tz UTC (or the zone given to dukascopy-node's "
                         "--time-zone option)")
    if info.fmt == "jforex":
        s = _clean_text(col)
        if len(s) and s.str.contains(_JFOREX_OFFSET_RE).all():
            info.time_kind, info.aware = "jforex_offset", True
            _check_sample(info, a, s, JFOREX_OFFSET_FORMATS, "DD.MM.YYYY HH:MM:SS.fff GMT+hhmm", utc=True)
        else:
            info.time_kind = "jforex"
            info.default_utc = info.jforex_header == "gmt"
            _check_sample(info, a, s, JFOREX_TIME_FORMATS, "DD.MM.YYYY HH:MM:SS.fff")
        return
    # generic
    if "date" in r:
        s = (sample[r["date"]].astype(str).str.strip() + " " + col.astype(str).str.strip()).str.strip()
        s = s[s != ""]
        what = f"the date and time columns ({_ascii(r['date'])} + {_ascii(r['ts'])})"
    else:
        s = _clean_text(col)
        what = f"the time column {_qp(r['ts'])}"
    if not len(s):
        raise DataError(f"{_ascii(info.path.name)}: {what} is empty in the first rows")
    _detect_text_time(info, a, s, what)
    if "date" in r and info.time_kind == "epoch":
        raise UsageError("--date-col is for a date column next to a time column; the joined values look like "
                         "numbers. Leave out --date-col")


def _check_sample(info: InputInfo, a, s: pd.Series, formats, expected: str, utc: bool = False) -> None:
    s = s[s.astype(str).str.strip() != ""]
    if not len(s):
        return
    ns = _parse_strings(s.reset_index(drop=True), formats, utc=utc)
    if np.mean(ns == NAT) > 0.5:
        raise UsageError(f"{_ascii(info.path.name)}: the times do not look like {expected} (first value: "
                         f"{_show(s.iloc[0], _cutoff_of(a))}); check --format, or use --format generic with "
                         "--time-format")


def resolve_tz(info: InputInfo, tz: TzSpec | None, dry_run: bool, force_tz: bool = False) -> list[str]:
    """Set the time zone used for one input; returns warnings."""
    warn: list[str] = []
    name = _ascii(info.path.name)
    if info.aware:
        info.tz = None
        info.tz_note = "each time carries its own UTC offset (converted exactly)"
        if tz is not None:
            info.tz_note += f"; --tz {tz.text} is not used for this file"
        return warn
    if info.default_utc and tz is not None and tz.kind != "utc":
        if info.fmt == "jforex":
            raise UsageError(f"{name}: the header says 'Gmt time', so the times are UTC; --tz {_ascii(tz.text)} would "
                             "shift every bar. Leave out --tz (or pass --tz UTC)")
        if not force_tz:
            raise UsageError(f"{name}: dukascopy-node epoch times are UTC; --tz {_ascii(tz.text)} would shift every "
                             "bar. Leave out --tz (or pass --tz UTC). Only if this file was downloaded with "
                             "dukascopy-node's --utc-offset option (times shifted on purpose) add --force-tz")
        warn.append(f"{name}: dukascopy-node times are normally UTC; --tz {tz.text} is applied because of --force-tz")
    if tz is None:
        if info.default_utc:
            info.tz = parse_tz("UTC")
            info.tz_note = "UTC (" + ("dukascopy-node epoch times are UTC" if info.fmt == "dukascopy-node"
                                      else "the header says 'Gmt time'") + ")"
            return warn
        if dry_run:
            info.tz = None
            info.tz_note = "NEEDS --tz (only local times are shown in this dry run)"
            return warn
        if info.fmt == "mt5":
            raise UsageError(f"{name}: --tz is required for MT5 files.\n" + MT5_TZ_HELP)
        if info.fmt == "jforex":
            raise UsageError(f"{name}: the header says 'Local time', so --tz is required (the time zone of the "
                             "computer that made the export, e.g. --tz Europe/Athens). A 'Gmt time' export avoids this")
        if info.fmt == "dukascopy-node":
            raise UsageError(f"{name}: these dukascopy-node times are text without a UTC offset, so --tz is required: "
                             "--tz UTC, unless the download used dukascopy-node's --time-zone option (then that zone)")
        raise UsageError(f"{name}: the times carry no UTC offset, so --tz is required (if they are already UTC, "
                         "pass --tz UTC)")
    info.tz = tz
    info.tz_note = tz.describe() + " (from --tz)"
    return warn


# ---------------------------------------------------------------------------------------
# reading and parsing chunks

@dataclass
class StreamState:
    carry_bid: float = float("nan")
    carry_ask: float = float("nan")
    seq: int = 0
    last_local: int = NAT       # last readable local time read so far (in row order)
    new_file: bool = False      # the next chunk is the first of a new file
    pending: bool = False       # a new file whose first readable time has not been seen yet
    saved_bid: float = float("nan")    # the previous file's last bid/ask, kept aside while pending
    saved_ask: float = float("nan")


def iter_raw(info: InputInfo, chunksize: int, nrows: int | None = None):
    """Yield DataFrames with only the needed columns (the generator keeps no reference to a chunk,
    so the caller can free it while it is being parsed)."""
    cols = [c for c in info.roles.values() if c is not None]
    if info.kind == "parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(info.path)
        left = nrows
        for batch in pf.iter_batches(batch_size=chunksize, columns=cols, use_threads=False):
            if left is not None:
                batch = batch.slice(0, left)
                left -= batch.num_rows
            yield _batch_to_pandas(batch)
            if left is not None and left <= 0:
                return
        return
    dtype = {}
    if info.fmt == "mt5":
        dtype = {info.roles["date"]: str, info.roles["time"]: str}
    elif info.time_kind != "epoch" or info.epoch_unit in ("ns", None):
        # text; nanosecond epochs too: as float64 (after one blank cell) they would lose ~256 ns
        dtype = {info.roles["ts"]: str}
        if "date" in info.roles:
            dtype[info.roles["date"]] = str
    with pd.read_csv(info.path, usecols=cols, dtype=dtype, chunksize=chunksize, nrows=nrows,
                     **_csv_kwargs(info)) as reader:
        yield from reader


def _prices(col: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (values float64 with NaN, blank mask, unparsable-text mask)."""
    if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
        v = col.to_numpy(dtype="float64", na_value=np.nan)
        return v, np.isnan(v), np.zeros(len(v), dtype=bool)
    s = col.astype(object)
    blank = s.isna().to_numpy() | (s.astype(str).str.strip() == "").to_numpy()
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    bad = np.isnan(v) & ~blank
    return v, blank, bad


def _ffill(v: np.ndarray, carry: float) -> tuple[np.ndarray, float]:
    out = np.array(pd.Series(v).ffill().to_numpy(dtype="float64"), dtype="float64")
    head = np.isnan(out)
    if head.any() and not math.isnan(carry):
        out[head] = carry
    last = out[-1] if len(out) else float("nan")
    return out, (carry if math.isnan(last) else float(last))


@dataclass
class Parsed:
    raw_time: pd.Series | None   # only kept for the dry run
    local: np.ndarray        # int64 ns (naive local) or NAT; for aware inputs = utc
    utc: np.ndarray | None   # int64 ns or NAT; None when no tz is known (dry run)
    bid: np.ndarray
    ask: np.ndarray
    vol: np.ndarray | None
    seq: np.ndarray
    reason: np.ndarray       # int8 index into REASONS


def parse_chunk(info: InputInfo, df: pd.DataFrame, state: StreamState, keep_raw: bool = False) -> Parsed:
    """Parse one raw chunk. The caller should hold no other reference to `df`: it is released as
    soon as the needed columns are taken, which keeps the memory peak low."""
    n = len(df)
    r = info.roles
    reason = np.zeros(n, dtype=np.int8)
    seq = np.arange(state.seq, state.seq + n, dtype=np.int64)
    state.seq += n
    bid, bid_blank, bid_bad = _prices(df[r["bid"]])
    ask, ask_blank, ask_bad = _prices(df[r["ask"]])
    vol = None
    if "vol" in r:
        vol = pd.to_numeric(df[r["vol"]], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    if info.fmt == "mt5":
        d, tm = df[r["date"]], df[r["time"]]
        del df
        raw = d + " " + tm
        del d, tm
        local = _parse_strings(raw, MT5_TIME_FORMATS)
    else:
        if "date" in r:                       # generic --date-col: 'date time'
            raw = (df[r["date"]].astype(str).str.strip() + " " + df[r["ts"]].astype(str).str.strip()).where(
                df[r["date"]].notna() & df[r["ts"]].notna())
        else:
            raw = df[r["ts"]]
        del df
        if info.time_kind == "epoch":
            if info.epoch_unit is None:
                info.epoch_unit = _epoch_unit(pd.to_numeric(raw, errors="coerce"))
            local = _epoch_to_ns(raw, info.epoch_unit) if info.epoch_unit else np.full(n, NAT, dtype="int64")
        elif info.time_kind in ("datetime", "datetime_aware"):
            local = _dt_to_ns(raw)
        else:
            fmts = {"jforex": JFOREX_TIME_FORMATS, "jforex_offset": JFOREX_OFFSET_FORMATS,
                    "iso": ("ISO8601",), "iso_offset": ("ISO8601",), "mt5str": MT5_TIME_FORMATS,
                    "format": (info.time_format,)}[info.time_kind]
            local = _parse_strings(raw.astype(str).str.strip().where(raw.notna()), fmts, utc=info.aware)
    if not keep_raw:
        del raw
        raw = None
    reason[local == NAT] = R["unparsable_time"]
    valid_t = np.flatnonzero(local != NAT)
    # MT5 blank cells: a new file starts with no known bid/ask; the previous file's last values are kept aside
    # and used for the new file's leading blank cells only when its first readable time shows that it simply
    # continues the previous file. The decision waits for that first readable time, wherever the chunk
    # boundaries fall, so the result does not depend on --chunksize.
    if state.new_file:
        state.new_file = False
        if not state.pending:                 # (after a file with no readable time, the older values stay aside)
            state.saved_bid, state.saved_ask = state.carry_bid, state.carry_ask
        state.carry_bid = state.carry_ask = float("nan")
        state.pending = True
    if state.pending and len(valid_t):
        state.pending = False
        first = int(local[valid_t[0]])
        if state.last_local != NAT and 0 <= first - state.last_local <= MT5_CARRY_GAP_NS:
            if math.isnan(state.carry_bid):
                state.carry_bid = state.saved_bid
            if math.isnan(state.carry_ask):
                state.carry_ask = state.saved_ask
    if len(valid_t):
        state.last_local = int(local[valid_t[-1]])
    if info.fmt == "mt5":
        reason[(reason == 0) & bid_blank & ask_blank] = R["mt5_no_bid_or_ask_change"]
        reason[(reason == 0) & (bid_bad | ask_bad)] = R["unparsable_price"]
        bid, state.carry_bid = _ffill(bid, state.carry_bid)
        ask, state.carry_ask = _ffill(ask, state.carry_ask)
        reason[(reason == 0) & (np.isnan(bid) | np.isnan(ask))] = R["mt5_before_first_quote"]
    else:
        reason[(reason == 0) & (bid_bad | ask_bad)] = R["unparsable_price"]
    # time zone
    if info.aware:
        utc = local
    elif info.tz is None:
        utc = None
    else:
        utc = to_utc_ns(local, info.tz)
        reason[(reason == 0) & (local != NAT) & (utc == NAT)] = R["dst_nonexistent_or_ambiguous_time"]
    if utc is not None:
        oor = (utc != NAT) & ((utc < TIME_MIN_NS) | (utc >= TIME_MAX_NS))
        reason[(reason == 0) & oor] = R["time_out_of_range"]
    with np.errstate(invalid="ignore"):
        reason[(reason == 0) & ~(np.isfinite(bid) & np.isfinite(ask))] = R["missing_price"]
        reason[(reason == 0) & ((bid <= 0) | (ask <= 0))] = R["nonpositive_price"]
        reason[(reason == 0) & (ask < bid)] = R["crossed_quote"]
    return Parsed(raw, local, utc, bid, ask, vol, seq, reason)


# ---------------------------------------------------------------------------------------
# bars: partial aggregates that merge exactly, whatever the chunking

PART_FIELDS = ("bar", "ft", "fs", "fp", "lt", "ls", "lp", "lsp", "hi", "lo", "n", "ssum", "vsum")


def ticks_to_partials(bar, t, seq, p, spr, sq, vq) -> dict:
    """Ticks (bar key s, time ns, sequence, price, spread, spread and volume in integer units) ->
    one partial bar per bar key: first/last tick by (time, sequence), high, low, count, sums."""
    order = None
    if len(t) > 1 and (np.any(t[1:] < t[:-1]) or np.any(seq[1:] <= seq[:-1])):
        order = np.lexsort((seq, t))
    if order is not None:
        bar, t, seq, p, spr, sq, vq = (x[order] for x in (bar, t, seq, p, spr, sq, vq))
    starts = np.r_[0, np.flatnonzero(np.diff(bar)) + 1]
    ends = np.r_[starts[1:] - 1, len(bar) - 1]
    return {"bar": bar[starts], "ft": t[starts], "fs": seq[starts], "fp": p[starts],
            "lt": t[ends], "ls": seq[ends], "lp": p[ends], "lsp": spr[ends],
            "hi": np.maximum.reduceat(p, starts), "lo": np.minimum.reduceat(p, starts),
            "n": (ends - starts + 1).astype("int64"),
            "ssum": np.add.reduceat(sq, starts), "vsum": np.add.reduceat(vq, starts)}


def merge_partials(parts: list[dict]) -> dict:
    parts = [p for p in parts if p is not None and len(p["bar"])]
    if not parts:
        return {k: np.array([], dtype="float64" if k in ("fp", "lp", "lsp", "hi", "lo") else "int64")
                for k in PART_FIELDS}
    if len(parts) == 1:
        return parts[0]
    c = {k: np.concatenate([p[k] for p in parts]) for k in PART_FIELDS}
    of = np.lexsort((c["fs"], c["ft"], c["bar"]))
    ol = np.lexsort((c["ls"], c["lt"], c["bar"]))
    bar = c["bar"][of]
    starts = np.r_[0, np.flatnonzero(np.diff(bar)) + 1]
    ends = np.r_[starts[1:] - 1, len(bar) - 1]
    f, last = of[starts], ol[ends]
    return {"bar": bar[starts], "ft": c["ft"][f], "fs": c["fs"][f], "fp": c["fp"][f],
            "lt": c["lt"][last], "ls": c["ls"][last], "lp": c["lp"][last], "lsp": c["lsp"][last],
            "hi": np.maximum.reduceat(c["hi"][of], starts), "lo": np.minimum.reduceat(c["lo"][of], starts),
            "n": np.add.reduceat(c["n"][of], starts), "ssum": np.add.reduceat(c["ssum"][of], starts),
            "vsum": np.add.reduceat(c["vsum"][of], starts)}


class BarAccumulator:
    """Merged partial bars, sorted by bar key, kept in one growing array per field, plus a short list of new
    partials. New partials are merged in batches only with the stored bars they can touch (with time-ordered
    input: the last bar or nothing), and the stored arrays are returned without a final copy, so memory stays
    close to the size of the bars themselves (about 100 bytes per bar) whatever --chunksize is."""

    FLUSH_ROWS = 200_000

    def __init__(self) -> None:
        self.buf: dict | None = None             # field -> array; rows [0, n) are used, the rest is spare room
        self.n = 0
        self.pending: list[dict] = []
        self.pending_rows = 0

    def add(self, part: dict) -> None:
        if not len(part["bar"]):
            return
        self.pending.append(part)
        self.pending_rows += len(part["bar"])
        if self.pending_rows >= self.FLUSH_ROWS:
            self._flush()

    def _flush(self) -> None:
        if not self.pending:
            return
        kmin = min(int(p["bar"][0]) for p in self.pending)        # partials are sorted by bar key
        touched: list[dict] = []
        if self.n and int(self.buf["bar"][self.n - 1]) >= kmin:
            i = int(np.searchsorted(self.buf["bar"][: self.n], kmin, side="left"))
            touched.append({k: v[i: self.n].copy() for k, v in self.buf.items()})
            self.n = i
        merged = merge_partials(touched + self.pending)
        self.pending, self.pending_rows = [], 0
        del touched
        self._append(merged)

    def _append(self, m: dict) -> None:
        add = len(m["bar"])
        if not add:
            return
        need = self.n + add
        if self.buf is None:
            # np.empty: spare room that is never written takes no physical memory
            self.buf = {k: np.empty(max(need, 4096), dtype=m[k].dtype) for k in PART_FIELDS}
        elif need > len(self.buf["bar"]):
            cap = max(need, len(self.buf["bar"]) * 3 // 2 + 4096)
            for k in PART_FIELDS:                 # one field at a time keeps the peak low
                grown = np.empty(cap, dtype=self.buf[k].dtype)
                grown[: self.n] = self.buf[k][: self.n]
                self.buf[k] = grown
        for k in PART_FIELDS:
            self.buf[k][self.n: need] = m[k]
        self.n = need

    def result(self) -> dict:
        self._flush()
        if self.buf is None:
            return merge_partials([])
        out = {k: v[: self.n] for k, v in self.buf.items()}
        self.buf, self.n = None, 0
        return out


def bars_frame(P: dict, with_volume: bool) -> pd.DataFrame:
    """The bar table from merged partials. It empties P as it goes and does not copy the columns again, so
    the bars are held about once (this matters for M1 over many years)."""
    for k in ("ft", "fs", "lt", "ls"):                 # only needed for merging
        P.pop(k, None)
    n = P.pop("n").astype("float64")
    cols = {"time": P.pop("bar").astype("int64", copy=False),
            "open": P.pop("fp").astype("float64", copy=False), "high": P.pop("hi").astype("float64", copy=False),
            "low": P.pop("lo").astype("float64", copy=False), "close": P.pop("lp").astype("float64", copy=False),
            "tick_volume": n, "spread": P.pop("lsp").astype("float64", copy=False)}
    cols["spread_mean"] = P.pop("ssum").astype("float64") / n * SPREAD_QUANTUM
    vsum = P.pop("vsum")
    if with_volume:
        cols["volume_sum"] = vsum.astype("float64") * VOLUME_QUANTUM
    del vsum
    return pd.DataFrame(cols, copy=False)


# ---------------------------------------------------------------------------------------
# settings

@dataclass
class Settings:
    tz: TzSpec | None
    tf: str
    tf_s: int
    price: str
    cutoff_s: int
    symbol: str
    chunksize: int
    out: Path
    train_path: Path
    locked_path: Path
    manifest_path: Path
    compare: Path | None
    overlap: str = "refuse"
    allow_drops: bool = False
    later_cutoff: bool = False
    compare_vol_col: str | None = None
    warnings: list = field(default_factory=list)


def parse_cutoff(text: str) -> int:
    try:
        ts = pd.Timestamp(text)
    except Exception:  # noqa: BLE001
        raise UsageError(f"--cutoff {_qp(text)} is not a date/time; use e.g. {DEFAULT_CUTOFF}")
    if ts is pd.NaT or pd.isna(ts):
        raise UsageError(f"--cutoff {_qp(text)} is not a date/time")
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.value // NS)


def make_settings(a) -> Settings:
    if a.chunksize < 1:
        raise UsageError("--chunksize must be at least 1")
    tf = a.tf.upper()
    if tf not in TF_SECONDS:
        raise UsageError(f"--tf {a.tf}: choose one of {', '.join(TF_SECONDS)}")
    symbol = a.symbol.strip()
    if not _SYMBOL_RE.fullmatch(symbol) or is_forbidden(symbol) or "locked" in symbol.lower():
        raise UsageError(f"--symbol {_qp(a.symbol)}: use letters, digits, '.', '-' or '_' (e.g. XAUUSD)")
    if a.encoding:
        try:
            text_codec = getattr(codecs.lookup(a.encoding), "_is_text_encoding", True)
        except LookupError:
            text_codec = None
        if not text_codec:                     # unknown, or a bytes codec such as rot13 / base64 / hex
            raise UsageError(f"--encoding {_qp(a.encoding)} is not a text encoding Python knows (e.g. utf-8, "
                             "utf-16, gbk, cp936, cp1252)")
    if a.time_format:
        try:
            pd.to_datetime(pd.Series(["2024-01-02 03:04:05.678"]), format=a.time_format, errors="coerce")
        except (ValueError, TypeError) as e:
            raise UsageError(_bad_time_format(a.time_format, e))
    tz = parse_tz(a.tz) if a.tz else None
    cutoff_s = parse_cutoff(a.cutoff)
    if cutoff_s > DEFAULT_CUTOFF_S and not a.allow_later_cutoff:
        raise UsageError(f"--cutoff {_ascii(a.cutoff)} is later than {DEFAULT_CUTOFF}. Data from "
                         f"{DEFAULT_CUTOFF_DAY} on is the period of your locked holdout, so it must stay out of the "
                         "train file. Use the default cutoff (or an earlier one); add --allow-later-cutoff only if "
                         "you really mean to train on that period")
    if cutoff_s < TIME_MIN_NS // NS:
        raise UsageError(f"--cutoff {a.cutoff} is before 1990")
    tf_s = TF_SECONDS[tf]
    if cutoff_s % tf_s:
        raise UsageError(f"--cutoff {a.cutoff} is not on a {tf} bar boundary (UTC); a bar would hold ticks from "
                         "both sides of the cutoff. Use a time such as YYYY-MM-DDT00:00:00Z")
    if a.time_format and a.format not in ("auto", "generic"):
        raise UsageError("--time-format is only used with --format generic")
    col_flags = [f for f in ("time_col", "bid_col", "ask_col", "vol_col") if getattr(a, f)]
    if col_flags and a.format not in ("auto", "generic"):
        raise UsageError("--time-col, --bid-col, --ask-col and --vol-col are only used with --format generic")
    if a.vol_col and not (a.time_col and a.bid_col and a.ask_col):
        raise UsageError("--vol-col needs --time-col, --bid-col and --ask-col (--format generic)")
    if a.date_col and not (a.time_col and a.bid_col and a.ask_col):
        raise UsageError("--date-col needs --time-col, --bid-col and --ask-col (--format generic): the date column "
                         "and the time column are joined with a space, and --time-format describes both")
    _sep_arg(a.sep)
    out_text = a.out
    if is_forbidden(out_text):
        raise UsageError(f"refused: --out {_qp(out_text)} is inside a locked holdout folder; choose another folder "
                         "(the tool makes its own locked_holdout subfolder inside --out)")
    out = Path(out_text).resolve()
    if out.exists() and not out.is_dir():
        raise UsageError(f"--out {_qp(out_text)} exists and is not a folder")
    name = f"{symbol}_{tf}.parquet"
    st = Settings(tz=tz, tf=tf, tf_s=tf_s, price=a.price, cutoff_s=cutoff_s, symbol=symbol,
                  chunksize=a.chunksize, out=out, train_path=out / "train" / name,
                  locked_path=out / "locked_holdout" / (name + FORBIDDEN_SUFFIX),
                  manifest_path=out / f"{symbol}_{tf}_manifest.json", compare=None,
                  overlap=a.overlap, allow_drops=a.allow_drops, later_cutoff=cutoff_s > DEFAULT_CUTOFF_S)
    if st.later_cutoff:
        st.warnings.append(f"!!! --cutoff {_iso_s(cutoff_s)} is later than {DEFAULT_CUTOFF}: this train file "
                           f"INCLUDES data from the period of your locked holdout ({DEFAULT_CUTOFF_DAY} on). Do not "
                           "use it for holdout-clean research or to check results on that holdout")
    if a.compare_with:
        st.compare, st.compare_vol_col = check_compare_path(a.compare_with, tf)
    if tf == "D1":
        st.warnings.append("D1 bars use UTC days (00:00-24:00 UTC). Daily bars depend on the chosen day "
                           "boundary: a broker's daily candles (often New York 17:00) will differ")
    return st


_TF_ALIAS = {"M1": ("m1", "1m", "1min", "min1"), "M5": ("m5", "5m", "5min", "min5"),
             "M15": ("m15", "15m", "15min", "min15"), "M30": ("m30", "30m", "30min", "min30"),
             "H1": ("h1", "1h", "60m", "60min", "min60", "60"), "H4": ("h4", "4h", "240m", "240min", "min240", "240"),
             "D1": ("d1", "1d", "day", "daily", "1440m", "1440min")}


def _tf_from_name(path: Path) -> str | None:
    stem = path.name[: -len(".parquet")] if path.name.lower().endswith(".parquet") else path.stem
    if "_" not in stem:
        return None
    tok = stem.rsplit("_", 1)[1].strip().lower().replace("-", "").replace("_", "")
    for tf, names in _TF_ALIAS.items():
        if tok in names:
            return tf
    return None


def check_compare_path(text: str, tf: str) -> tuple[Path, str | None]:
    """-> (resolved path, its volume column or None)."""
    if is_forbidden(text):
        raise UsageError(f"refused: --compare-with {_qp(text)} points at the locked holdout ('{FORBIDDEN_PART}' or "
                         f"'*{FORBIDDEN_SUFFIX}'); compare with the TRAIN file only")
    p = Path(text)
    if not p.is_file():
        raise UsageError(f"--compare-with {_qp(text)}: file not found")
    p = p.resolve()
    if is_forbidden(str(p)):
        raise UsageError(f"refused: --compare-with {_qp(text)} resolves into the locked holdout")
    try:
        import pyarrow.parquet as pq
        names = pq.read_schema(p).names
    except Exception as e:  # noqa: BLE001
        raise UsageError(f"--compare-with {_qp(text)} is not a readable Parquet file: {type(e).__name__}")
    if "time" not in names or "close" not in names:
        raise UsageError(f"--compare-with {_qp(text)} must have 'time' and 'close' columns (AlphaMaster layout)")
    ref_tf = _tf_from_name(p)
    if ref_tf is None:
        raise UsageError(f"--compare-with {_qp(text)}: the name must look like SYMBOL_TF.parquet (e.g. "
                         "XAUUSD_H1.parquet) so its timeframe is known")
    if ref_tf != tf:
        raise UsageError(f"--compare-with is a {ref_tf} file but --tf is {tf}; the comparison needs the same "
                         "timeframe")
    vol = next((c for c in ("tick_volume", "volume") if c in names), None)
    return p, vol


# ---------------------------------------------------------------------------------------
# outputs: overwrite rule and publishing

def _read_own_manifest(path: Path) -> dict | None:
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) and rec.get("tool") == TOOL else None


def foreign_out_problem(st: Settings) -> str | None:
    """Why --out must not be written into (it holds research data this tool did not make), or None.
    Only manifests and the train folder are looked at; the locked_holdout folder is never listed."""
    out = st.out
    if not out.is_dir():
        return None
    own_train: set[str] = set()
    own_locked = False
    for m in sorted(out.glob("*_manifest.json")):
        rec = _read_own_manifest(m)
        if rec is None:
            return f"{m.name} was not made by ticks_to_bars.py (for example the Dukascopy converter's manifest)"
        train_file = (rec.get("train") or {}).get("file")
        if train_file:
            own_train.add(re.split(r"[\\/]", str(train_file))[-1])
        own_locked = (own_locked or bool((rec.get("locked") or {}).get("file"))
                      or "locked_holdout" in (rec.get("folders_made_by_this_tool") or []))
    train_dir = out / "train"
    if train_dir.is_dir():
        for f in sorted(train_dir.iterdir()):
            if not f.name.startswith(".") and f.name not in own_train:
                return f"train{os.sep}{_ascii(f.name)} is not listed in a manifest of ticks_to_bars.py"
    if (out / "locked_holdout").exists() and not own_locked:
        return "its locked_holdout folder was not made by ticks_to_bars.py"
    return None


def _foreign_out_message(st: Settings, why: str) -> str:
    return (f"refused: the --out folder {_qp(st.out)} holds research data not made by this tool ({why}). Writing "
            "your broker's bars next to it would mix data sources under similar names. Choose a new, empty folder, "
            "e.g. --out D:\\research\\mt5_ticks")


def check_outputs(st: Settings, overwrite: bool) -> None:
    """Refuse to replace files unless --overwrite, and then only files this tool made.

    The locked file is never read: it is replaced only when this tool's manifest says it wrote it."""
    outs = (st.train_path, st.locked_path, st.manifest_path)
    existing = [p for p in outs if p.exists()]
    if not existing:
        return
    if not overwrite:
        raise UsageError("output already exists: " + ", ".join(_ascii(p) for p in existing)
                         + ". Choose another --out folder, or add --overwrite to replace files this tool made "
                         "(other files are never replaced)")
    rec = _read_own_manifest(st.manifest_path)
    ok = rec is not None
    if ok and st.train_path.exists():
        ok = st.train_path.is_file() and rec.get("train", {}).get("sha256") == _sha256_file(st.train_path)
    if ok and st.locked_path.exists():
        ok = st.locked_path.is_file() and bool(rec.get("locked", {}).get("sha256"))
    if not ok:
        raise UsageError(f"refused: the files in {_ascii(st.out)} were not made by this tool (there is no "
                         f"{st.manifest_path.name} from ticks_to_bars.py that matches them), so --overwrite will not "
                         "replace them: they may be your existing research data. Choose another --out folder")


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _make_dirs(folder: Path, made: list[Path]) -> None:
    """mkdir -p that records each folder it creates (in creation order), so a failed run can remove them."""
    missing = []
    p = folder
    while not p.exists():
        missing.append(p)
        if p.parent == p:
            break
        p = p.parent
    for q in reversed(missing):
        try:
            q.mkdir()
        except FileExistsError:
            continue
        made.append(q)


def publish(st: Settings, files: dict[Path, bytes | None]) -> None:
    """Write all outputs or none: temp files first, old files moved aside, restored on failure.

    `files` maps each output path to its new bytes (None = the file must not exist afterwards).
    Old files are renamed, never read. Folders this call creates are removed again on failure (they are
    empty then), and an emptied locked_holdout folder is removed after a run without locked bars, so a
    failed or earlier run never leaves a folder that a later run would take for someone else's data."""
    temps: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    done: list[Path] = []
    made: list[Path] = []
    try:
        for path, data in files.items():
            if data is None:
                continue
            _make_dirs(path.parent, made)
            tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}" + (FORBIDDEN_SUFFIX if path == st.locked_path
                                                                       else ""))
            tmp.write_bytes(data)
            temps[path] = tmp
        for path in files:
            if path.exists():
                bak = path.with_name(f".{path.name}.bak-{os.getpid()}" + (FORBIDDEN_SUFFIX if path == st.locked_path
                                                                           else ""))
                os.replace(path, bak)
                backups[path] = bak
            if path in temps:
                os.replace(temps[path], path)
                del temps[path]
                done.append(path)
    except BaseException:
        for path in done:
            try:
                path.unlink()
            except OSError:
                pass
        for path, bak in backups.items():
            try:
                os.replace(bak, path)
            except OSError:
                pass
        for tmp in temps.values():
            try:
                tmp.unlink()
            except OSError:
                pass
        for folder in reversed(made):          # only empty folders can be removed: nothing else is touched
            try:
                os.rmdir(folder)
            except OSError:
                pass
        raise
    for bak in backups.values():
        try:
            bak.unlink()
        except OSError:
            pass
    if files.get(st.locked_path, b"") is None:
        try:
            os.rmdir(st.locked_path.parent)    # removed only when empty (another timeframe's file keeps it)
        except OSError:
            pass


# ---------------------------------------------------------------------------------------
# statistics (train part only)

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _wd_hour(t_s: int) -> str:
    return f"{_WEEKDAYS[(int(t_s) // 86400 + 3) % 7]} {(int(t_s) % 86400) // 3600:02d}:{(int(t_s) % 3600) // 60:02d}"


def train_stats(train: pd.DataFrame, tf_s: int) -> dict:
    t = train["time"].to_numpy(dtype="int64")
    o, h, lo, c = (train[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    n = len(t)
    years = pd.Series(pd.to_datetime(t, unit="s", utc=True).year).value_counts().sort_index()
    dt = np.diff(t)
    long_idx = np.flatnonzero(dt > GAP_LONG_HOURS * 3600)
    top = long_idx[np.argsort(-dt[long_idx], kind="stable")][:10]
    long_list = [{"start_utc": _iso_s(t[i]), "end_utc": _iso_s(t[i + 1]),
                  "hours_between_bar_opens": float(dt[i] / 3600)} for i in sorted(top)]
    tv = train["tick_volume"].to_numpy(dtype="float64")
    med_close = float(np.median(c))

    def spread_block(x):
        res = {"median": _q(x, 50), "p90": _q(x, 90), "p99": _q(x, 99)}
        res.update({f"{k}_pct_of_median_close": (v / med_close * 100 if v is not None and med_close else None)
                    for k, v in list(res.items())})
        return res

    hours = (t % 86400) // 3600
    by_hour = {}
    for col in ("spread", "spread_mean"):
        x = train[col].to_numpy(dtype="float64")
        by_hour[col] = {f"{hh:02d}": (float(np.median(x[hours == hh])) if np.any(hours == hh) else None)
                        for hh in range(24)}
    rng = h - lo
    med_rng = float(np.median(rng))
    order = np.argsort(-rng, kind="stable")[:10]
    largest = [{"time_utc": _iso_s(t[i]), "high": float(h[i]), "low": float(lo[i]), "range": float(rng[i]),
                "ratio_to_median_range": (float(rng[i] / med_rng) if med_rng > 0 else None)} for i in order]
    weekend = None
    if tf_s <= 3600:
        wk = np.flatnonzero(dt >= 24 * 3600)
        reopen = pd.Series([_wd_hour(t[i + 1]) for i in wk], dtype=object).value_counts()
        close = pd.Series([_wd_hour(t[i]) for i in wk], dtype=object).value_counts()
        weekend = {"breaks_of_24h_or_more": int(len(wk)),
                   "reopen_bar_utc_most_common": {k: int(v) for k, v in reopen.head(3).items()},
                   "last_bar_before_break_utc_most_common": {k: int(v) for k, v in close.head(3).items()},
                   "note": "XAUUSD usually reopens Sunday 22:00 UTC (US summer) or 23:00 UTC (US winter) and closes "
                           "Friday about 21:00/22:00 UTC; whole-hour differences suggest a wrong --tz"}
    return {
        "bars": n,
        "first_bar_utc": _iso_s(t[0]), "last_bar_utc": _iso_s(t[-1]),
        "bars_per_year": {str(k): int(v) for k, v in years.items()},
        "gaps": {"over_1_bar": int((dt > tf_s).sum()), f"over_{GAP_LONG_HOURS}h": int(len(long_idx)),
                 f"over_{GAP_LONG_HOURS}h_longest": long_list,
                 "note": "gaps are measured between consecutive train bar opens: 'over_1_bar' counts bar opens more "
                         "than one bar length apart (weekends, holidays and quiet periods included)"},
        "ticks_per_bar": {"median": _q(tv, 50), "p10": _q(tv, 10), "bars_with_fewer_than_5_ticks": int((tv < 5).sum())},
        "median_close": med_close,
        "spread_last_tick": spread_block(train["spread"].to_numpy(dtype="float64")),
        "spread_mean": spread_block(train["spread_mean"].to_numpy(dtype="float64")),
        "median_spread_by_utc_hour": by_hour,
        "median_spread_by_utc_hour_note": "median over train bars whose open falls in that UTC hour (null when "
                                          "no bar opens in that hour, e.g. H4/D1)",
        "median_bar_range": med_rng,
        "largest_ranges": largest,
        "largest_ranges_note": "flagged, not removed: eyeball these for bad ticks (spikes)",
        "weekend": weekend,
        "ohlc_valid": bool(np.all(h >= np.maximum(o, c)) and np.all(lo <= np.minimum(o, c)) and np.all(lo > 0)),
    }


# ---------------------------------------------------------------------------------------
# comparison with a reference bar file (e.g. the Dukascopy research train file)

def _consecutive_returns(t: np.ndarray, c: np.ndarray, tf_s: int) -> tuple[np.ndarray, np.ndarray]:
    """Close-to-close log returns of bars whose previous bar is exactly one bar earlier."""
    ok = np.flatnonzero(np.diff(t) == tf_s) + 1
    return t[ok], np.log(c[ok] / c[ok - 1])


def _corr(x, y) -> float | None:
    if len(x) < MIN_PAIRS or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _lag_corrs(tu, ru, tr, rr, tf_s, mask_ref=None) -> dict:
    res = {}
    for lag in COMPARE_LAGS:
        _, iu, ir = np.intersect1d(tu - lag * tf_s, tr, assume_unique=True, return_indices=True)
        if mask_ref is not None:
            keep = mask_ref[ir]
            iu, ir = iu[keep], ir[keep]
        res[lag] = (_corr(ru[iu], rr[ir]), int(len(iu)))
    return res


def _best(corrs: dict) -> int | None:
    vals = [(v[0], -abs(k), k) for k, v in corrs.items() if v[0] is not None]
    return max(vals)[2] if vals else None


def _offset_at(t_s: np.ndarray, zone: str) -> np.ndarray:
    """UTC offset in seconds of `zone` at UTC epoch seconds t_s."""
    idx = pd.DatetimeIndex(pd.to_datetime(t_s, unit="s", utc=True)).tz_convert(zone)
    return np.asarray((idx.tz_localize(None) - pd.to_datetime(t_s, unit="s")).total_seconds())


def _us_dst(t_s: np.ndarray) -> np.ndarray:
    return _offset_at(t_s, NY_ZONE) == -4 * 3600


def _eu_dst(t_s: np.ndarray) -> np.ndarray:
    return _offset_at(t_s, EU_ZONE) == 2 * 3600


# daylight-saving regimes (by the reference bar's UTC time) and a UTC moment inside each one
DST_REGIMES = {"both_standard_time": "2024-01-15T12:00:00Z",
               "us_dst_only": "2024-03-20T12:00:00Z",          # 2nd Sunday of March .. last Sunday of March,
               "both_daylight_saving": "2024-07-15T12:00:00Z"}  # last Sunday of October .. 1st Sunday of November
EU_DST_ZONES = {0: "Europe/London", 1: "Europe/Berlin", 2: "Europe/Athens"}


def _regime_masks(t_s: np.ndarray) -> dict:
    us, eu = _us_dst(t_s), _eu_dst(t_s)
    return {"both_standard_time": ~us & ~eu, "us_dst_only": us & ~eu, "both_daylight_saving": us & eu}


def _tz_offset_hours(tz: TzSpec, when_utc: str) -> float:
    """local = UTC + this many hours, for --tz at a given UTC moment."""
    if tz.kind == "utc":
        return 0.0
    if tz.kind == "fixed":
        return tz.offset_s / 3600
    t = np.array([pd.Timestamp(when_utc).value // NS])
    if tz.kind == "ny":
        return float(_offset_at(t, NY_ZONE)[0]) / 3600 + tz.ny_hours
    return float(_offset_at(t, tz.zone)[0]) / 3600


def _tz_flag(value: str) -> str:
    """'--tz X' (with '=' for a negative offset, which the command line would read as a flag)."""
    return f"--tz={value}" if value.startswith("-") else f"--tz {value}"


def _zone_from_offsets(w, m, s) -> str | None:
    """A --tz suggestion from the true UTC offsets (hours) in the three regimes (None = unknown)."""
    known = [x for x in (w, m, s) if x is not None]
    if not known or any(x != int(x) for x in known):
        return None
    if len(set(known)) == 1:
        off = int(known[0])
        fixed = "UTC" if off == 0 else _offset_text(off * 3600)
        return _tz_flag(fixed) + (" (or --tz ny+7, if your broker is on New York close time)" if off in (2, 3)
                                  and len(known) == 1 else "")
    if w is None:
        return None
    w = int(w)
    ny, eu = _tz_flag(f"ny{w + 5:+d}"), EU_DST_ZONES.get(w)
    eu_txt = _tz_flag(eu) if eu else f"an IANA zone with EU daylight saving and UTC{w:+d} in winter"
    if s is not None and s == w + 1 and m is None:
        return f"{ny} (broker on New York close time) or {eu_txt} (broker on the European dates)"
    if s is not None and m is not None and s == w + 1 and m == w + 1:
        return f"{ny} (New York close time: UTC{w:+d} in US winter, UTC{w + 1:+d} in US summer)"
    if s is not None and m is not None and s == w + 1 and m == w:
        return f"{eu_txt} (daylight saving on the European dates)"
    return None


def suggest_tz(tz: TzSpec | None, hours: float) -> str:
    """--tz to try when the whole series is `hours` off (positive: your bars are stamped later)."""
    if tz is None:
        return "check the offsets written in your file's time stamps"
    if hours != int(hours):
        return "check --tz (the shift is not a whole number of hours)"
    true = [_tz_offset_hours(tz, when) + hours for when in DST_REGIMES.values()]
    if tz.kind in ("utc", "fixed"):
        return _zone_from_offsets(true[0], None, None) or "check --tz"
    if tz.kind == "ny":
        return _tz_flag(f"ny{tz.ny_hours + int(hours):+d}")
    return f"a zone {int(hours):+d} h from {tz.zone}"


def _volume_check(train: pd.DataFrame, ref_path: Path, vol_col: str | None, iu, ir, ref_index) -> dict:
    res = {"note": "tick_volume in your file is the NUMBER OF TICKS per bar; in the Dukascopy research file it is "
                   "Dukascopy's traded volume. AlphaMaster reads tick_volume as volume, so formulas that use volume "
                   "features (vol_ratio, vol_z, pv_corr, vwap_dev, obv_slope, mfi14) are not comparable between the "
                   "two files"}
    if vol_col is None:
        res["log_volume_corr"] = None
        return res
    rv = pd.read_parquet(ref_path, columns=[vol_col])[vol_col].to_numpy(dtype="float64")[ref_index]
    uv = train["tick_volume"].to_numpy(dtype="float64")
    a, b = uv[iu], rv[ir]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    res["reference_volume_column"] = vol_col
    res["log_volume_corr"] = _corr(np.log(a[ok]), np.log(b[ok]))
    res["bars_used"] = int(ok.sum())
    return res


def compare_with_reference(train: pd.DataFrame, ref_path: Path, st: Settings, tz_used: TzSpec | None) -> dict:
    ref = pd.read_parquet(ref_path, columns=["time", "close"])
    rt = ref["time"]
    if pd.api.types.is_datetime64_any_dtype(rt):
        rt_s = _dt_to_ns(rt) // NS
    else:
        rt_s = pd.to_numeric(rt, errors="coerce").to_numpy(dtype="float64")
        unit = _epoch_unit(rt_s) or "s"
        rt_s = np.floor(rt_s / {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9}[unit])
        rt_s = np.where(np.isfinite(rt_s), rt_s, -1).astype("int64")
    rc = ref["close"].to_numpy(dtype="float64")
    ok = (rt_s > 0) & np.isfinite(rc) & (rc > 0) & (rt_s < st.cutoff_s)
    r = pd.DataFrame({"t": rt_s[ok], "c": rc[ok], "i": np.flatnonzero(ok)}).drop_duplicates("t", keep="last")
    r = r.sort_values("t")
    tr_all, cr_all = r["t"].to_numpy(dtype="int64"), r["c"].to_numpy(dtype="float64")
    ri_all = r["i"].to_numpy(dtype="int64")
    del ref, r
    tu_all = train["time"].to_numpy(dtype="int64")
    cu_all = train["close"].to_numpy(dtype="float64")
    res: dict = {"reference_file": str(ref_path), "reference_sha256": _sha256_file(ref_path),
                 "scope": "bars before the cutoff, over the time span both files cover",
                 "lag_convention": "lag L pairs YOUR return at time t + L bars with the REFERENCE return at time t; "
                                   "best lag +2 means your bars are stamped 2 bars LATER than the reference (your "
                                   "--tz is 2 bars' worth of hours behind the data's real zone)"}
    if not len(tr_all):
        res["error"] = "the reference has no bars before the cutoff"
        return res
    lo_t, hi_t = max(tu_all[0], tr_all[0]), min(tu_all[-1], tr_all[-1])
    if lo_t > hi_t:
        res["error"] = "the two files do not overlap in time"
        return res
    mu = (tu_all >= lo_t) & (tu_all <= hi_t)
    mr = (tr_all >= lo_t) & (tr_all <= hi_t)
    tu, cu, tr, cr = tu_all[mu], cu_all[mu], tr_all[mr], cr_all[mr]
    common, iu, ir = np.intersect1d(tu, tr, assume_unique=True, return_indices=True)
    diff = cu[iu] - cr[ir]
    res.update({
        "common_span_utc": [_iso_s(lo_t), _iso_s(hi_t)],
        "overlapping_bars": int(len(common)),
        "share_of_your_bars_matched_in_reference": float(len(common) / len(tu)) if len(tu) else None,
        "share_of_reference_bars_matched_in_yours": float(len(common) / len(tr)) if len(tr) else None,
        "abs_close_diff_median": _q(np.abs(diff), 50), "abs_close_diff_p90": _q(np.abs(diff), 90),
        "close_diff_median_signed": _q(diff, 50),
    })
    res["volume"] = _volume_check(train, ref_path, st.compare_vol_col, np.flatnonzero(mu)[iu], ir,
                                  ri_all[mr])
    tur, rur = _consecutive_returns(tu, cu, st.tf_s)
    trr, rrr = _consecutive_returns(tr, cr, st.tf_s)
    corrs = _lag_corrs(tur, rur, trr, rrr, st.tf_s)
    res["return_corr_by_lag"] = {f"{k:+d}": v[0] for k, v in corrs.items()}
    res["return_pairs_by_lag"] = {f"{k:+d}": v[1] for k, v in corrs.items()}
    best = _best(corrs)
    res["best_lag_bars"] = best
    # at H4/D1 a whole-bar lag does not say how many hours the time stamps are off: no hours then
    res["best_lag_hours"] = (best * st.tf_s / 3600) if best is not None and st.tf_s <= 3600 else None
    by_regime = {}
    for name, mask in _regime_masks(trr).items():
        cc = _lag_corrs(tur, rur, trr, rrr, st.tf_s, mask_ref=mask)
        b = _best(cc)
        by_regime[name] = {"best_lag_bars": b, "corr_at_best": cc[b][0] if b is not None else None,
                           "pairs_at_lag0": cc[0][1]}
    res["best_lag_by_dst_regime"] = by_regime
    res["dst_regime_note"] = ("regimes by the reference bar's UTC time: both_standard_time (US and EU on standard "
                              "time), us_dst_only (2nd Sunday of March to the last Sunday of March, and the last "
                              "Sunday of October to the 1st Sunday of November), both_daylight_saving")
    warns = []
    if best is None:
        warns.append("compare: too few matching bars to estimate the time shift")
    elif st.tf_s == 3600:
        warns.extend(_h1_tz_verdict(best, by_regime, tz_used))
    else:
        if best != 0:
            size = f" ({best * st.tf_s // 60:+d} minutes)" if st.tf_s < 3600 else ""
            warns.append(f"!!! compare: your {st.tf} bars match the reference best at lag {best:+d} bar(s){size}; "
                         "the time stamps look shifted.")
        span = f"{3 * st.tf_s // 60} minutes" if st.tf_s < 3600 else f"{st.tf_s // 7200} h"
        warns.append(f"compare: with {st.tf} bars this check cannot measure an hour-sized time-zone error "
                     + (f"(the lags cover only {span})" if st.tf_s < 3600 else
                        f"(an error of up to {span} still matches best at lag 0)")
                     + "; run the time-zone check with --tf H1 (the comparison file must be H1 too)")
    if best is not None:
        cb = corrs[best][0]
        if cb is not None and cb < 0.5:
            warns.append(f"compare: the best return correlation is only {cb:.2f}; the two series do not track each "
                         "other well (check --symbol, --price, the time zone and the data)")
    res["warnings"] = warns
    return res


def _h1_tz_verdict(best: int, by_regime: dict, tz_used: TzSpec | None) -> list[str]:
    """Warnings for H1 bars, where one bar of lag is one hour of time-zone error."""
    lags = {k: v["best_lag_bars"] for k, v in by_regime.items()}
    known = {k: v for k, v in lags.items() if v is not None}
    if best == 0 and all(v == 0 for v in known.values()):
        return []
    if len(set(known.values())) > 1:
        tip = None
        if tz_used is not None:
            true = {k: (_tz_offset_hours(tz_used, DST_REGIMES[k]) + lags[k] if lags[k] is not None else None)
                    for k in DST_REGIMES}
            tip = _zone_from_offsets(true["both_standard_time"], true["us_dst_only"], true["both_daylight_saving"])
        parts = ", ".join(f"{k.replace('_', ' ')}: {v:+d} h" for k, v in known.items())
        return [f"!!! TIME ZONE CHECK FAILED: the time shift is not the same all year ({parts}), so your --tz uses "
                "the wrong daylight-saving rule. " + (f"Try {tip}, " if tip else "Check --tz, ")
                + "re-run with --overwrite and check again."]
    hours = best if not known else next(iter(known.values()))
    if hours == 0:
        return []
    return [f"!!! TIME ZONE CHECK FAILED: your bars match the reference best at lag {hours:+d} bars ({hours:+d} h). "
            f"Your --tz looks off by {hours:+d} hours: your bar times are {abs(hours)} h "
            f"{'later' if hours > 0 else 'earlier'} than the reference. Try {suggest_tz(tz_used, hours)}, re-run "
            "with --overwrite and check again."]


# ---------------------------------------------------------------------------------------
# the run

def _reason_counts(counts: np.ndarray) -> dict:
    return {REASONS[i]: int(counts[i]) for i in range(1, len(REASONS))}


def _print_counts(counts: np.ndarray, indent: str = "     ") -> None:
    for k, v in _reason_counts(counts).items():
        if v:
            print(f"{indent}{k:<36} {v:>12,}  ({REASON_HELP[k]})")


def _runs_of(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tick times (ns, any order) -> coverage runs [lo, hi]: ticks in the same or in consecutive clock
    minutes form one run; a whole clock minute without ticks ends it."""
    if not len(t):
        return np.array([], dtype="int64"), np.array([], dtype="int64")
    if len(t) > 1 and np.any(t[1:] < t[:-1]):
        t = np.sort(t)
    m = t // COVER_NS
    starts = np.r_[0, np.flatnonzero(np.diff(m) > 1) + 1]
    ends = np.r_[starts[1:] - 1, len(t) - 1]
    return t[starts], t[ends]


def _merge_runs(lo: np.ndarray, hi: np.ndarray, minute_gap: bool) -> tuple[np.ndarray, np.ndarray]:
    """Union of runs. With minute_gap (one file's coverage) runs in the same or consecutive clock minutes
    are joined too, so the result does not depend on how the file was chunked; without it (the union of
    several files) only runs that overlap are joined."""
    if len(lo) < 2:
        return lo, hi
    order = np.argsort(lo, kind="stable")
    lo, hi = lo[order], hi[order]
    reach = np.maximum.accumulate(hi)[:-1]
    new = (lo[1:] // COVER_NS > reach // COVER_NS + 1) if minute_gap else (lo[1:] > reach)
    starts = np.r_[0, np.flatnonzero(new) + 1]
    return lo[starts], np.maximum.reduceat(hi, starts)


def _in_runs(lo: np.ndarray, hi: np.ndarray, t: np.ndarray) -> np.ndarray:
    if not len(lo):
        return np.zeros(len(t), dtype=bool)
    i = np.searchsorted(lo, t, side="right") - 1
    res = i >= 0
    res[res] = t[res] <= hi[i[res]]
    return res


_EMPTY_I64 = np.array([], dtype="int64")


@dataclass
class FileStats:
    """Per-input counts. cov_lo/cov_hi are the stretches of time in which the file has ticks (also after the
    cutoff); they are used only to find ticks that an earlier file holds too, and are never printed or stored
    past the cutoff."""
    name: str
    index: int = 0
    counts: np.ndarray = field(default_factory=lambda: np.zeros(len(REASONS), dtype=np.int64))
    rows: int = 0
    cov_lo: np.ndarray = field(default_factory=lambda: _EMPTY_I64)
    cov_hi: np.ndarray = field(default_factory=lambda: _EMPTY_I64)
    overlap_rows_train: int = 0
    back_jumps: int = 0
    last_local: int = NAT

    def problem_drops(self) -> int:
        return int(sum(self.counts[R[r]] for r in PROBLEM_REASONS))

    def add_cover(self, t: np.ndarray) -> None:
        lo, hi = _runs_of(t)
        self.cov_lo, self.cov_hi = _merge_runs(np.r_[self.cov_lo, lo], np.r_[self.cov_hi, hi], minute_gap=True)


@dataclass
class RunState:
    acc: BarAccumulator = field(default_factory=BarAccumulator)
    stream: StreamState = field(default_factory=StreamState)
    dups: int = 0
    vol_missing: int = 0
    any_time_of_day: bool = False
    covers: list = field(default_factory=list)       # coverage (lo, hi) of each file read so far
    u_lo: np.ndarray = field(default_factory=lambda: _EMPTY_I64)       # their union
    u_hi: np.ndarray = field(default_factory=lambda: _EMPTY_I64)
    pairs: dict = field(default_factory=dict)        # (earlier file, later file) -> overlap record

    def file_done(self, fs: FileStats) -> None:
        self.covers.append((fs.cov_lo, fs.cov_hi))
        self.u_lo, self.u_hi = _merge_runs(np.r_[self.u_lo, fs.cov_lo], np.r_[self.u_hi, fs.cov_hi], minute_gap=False)

    def note_overlap(self, later: int, t: np.ndarray, cutoff_ns: int) -> None:
        """Record which earlier files hold ticks at the times t (ticks of file `later`)."""
        train = t < cutoff_ns
        for j, (lo, hi) in enumerate(self.covers):
            hit = _in_runs(lo, hi, t)
            if not hit.any():
                continue
            rec = self.pairs.setdefault((j, later), {"rows_train": 0, "lo": NAT, "hi": NAT, "after_cutoff": False})
            ht = t[hit & train]
            if len(ht):
                rec["rows_train"] += int(len(ht))
                lo_t, hi_t = int(ht.min()), int(ht.max())
                rec["lo"] = lo_t if rec["lo"] == NAT else min(rec["lo"], lo_t)
                rec["hi"] = max(rec["hi"], hi_t)
            if (hit & ~train).any():
                rec["after_cutoff"] = True


def _consume(p: Parsed, st: Settings, rs: RunState, fs: FileStats, with_volume: bool) -> None:
    """Count one parsed chunk and add its kept ticks to the bars."""
    cutoff_ns = st.cutoff_s * NS
    utc, local = p.utc, p.local
    readable = local[(local != NAT) & (local < cutoff_ns - LOCAL_SLACK_NS)]     # train-period times only
    if len(readable):                          # time jumps back by more than a day, in file order
        seq = readable if fs.last_local == NAT else np.r_[fs.last_local, readable]
        fs.back_jumps += int((np.diff(seq) < -86400 * NS).sum())
        fs.last_local = int(readable[-1])
    valid = p.reason == 0
    if valid.any():
        if not rs.any_time_of_day:
            rs.any_time_of_day = bool(np.any(local[valid] % (86400 * NS) != 0))
        vidx = np.flatnonzero(valid)
        tv = utc[vidx]
        fs.add_cover(tv)
        # a tick overlaps when an earlier file has ticks right around it (not merely somewhere between that
        # file's first and last tick): a file that fills a gap in another file does not overlap it
        inside = _in_runs(rs.u_lo, rs.u_hi, tv)
        if inside.any():
            idx = vidx[inside]
            rs.note_overlap(fs.index, utc[idx], cutoff_ns)
            fs.overlap_rows_train += int((utc[idx] < cutoff_ns).sum())
            if st.overlap == "drop":
                p.reason[idx] = R["overlap_with_earlier_file"]
    # rows counted: train period, unreadable times, and NaT/garbage times that cannot be in the locked period
    countable = ((utc != NAT) & ((utc < cutoff_ns) | (utc >= TIME_MAX_NS))) | \
                ((utc == NAT) & ((local == NAT) | (local < cutoff_ns - LOCAL_SLACK_NS)))
    fs.counts += np.bincount(p.reason[countable], minlength=len(REASONS))
    fs.rows += int(countable.sum())
    keep = p.reason == 0
    train_keep = keep & (utc < cutoff_ns)
    if train_keep.sum() > 1:
        rs.dups += int(pd.DataFrame({"t": utc[train_keep], "b": p.bid[train_keep],
                                     "a": p.ask[train_keep]}).duplicated().sum())
    if keep.any():
        t = utc[keep]
        bid, ask = p.bid[keep], p.ask[keep]
        price = bid if st.price == "bid" else ask if st.price == "ask" else (bid + ask) / 2.0
        spr = ask - bid
        sq = np.rint(spr / SPREAD_QUANTUM).astype("int64")
        if with_volume:
            v = p.vol[keep]
            rs.vol_missing += int((~np.isfinite(v) & (t < cutoff_ns)).sum())
            vq = np.rint(np.where(np.isfinite(v), v, 0.0) / VOLUME_QUANTUM).astype("int64")
        else:
            vq = np.zeros(len(t), dtype="int64")
        bar = (t // (st.tf_s * NS)) * st.tf_s
        rs.acc.add(ticks_to_partials(bar, t, p.seq[keep], price, spr, sq, vq))


def process_inputs(infos: list[InputInfo], st: Settings, with_volume: bool) -> tuple[dict, RunState, list]:
    """Stream every input, chunk by chunk; returns (bars dict, run state, per-file stats)."""
    rs = RunState()
    stats: list[FileStats] = []
    for k, info in enumerate(infos, 1):
        fs = FileStats(name=info.path.name, index=k - 1)
        stats.append(fs)
        rs.stream.new_file = True
        chunks = iter_raw(info, st.chunksize)
        try:
            while True:
                try:
                    p = parse_chunk(info, next(chunks), rs.stream)  # no reference to the raw chunk is kept here
                except StopIteration:
                    break
                _consume(p, st, rs, fs, with_volume)
                del p
        except (UsageError, DataError):
            raise
        except READ_ERRORS as e:
            raise DataError(f"could not read {_ascii(info.path)} (after {fs.rows:,} rows): "
                            f"{type(e).__name__}: {_ascii(e)}")
        rs.file_done(fs)
        kept = int(fs.counts[0])
        print(f"  read {k}/{len(infos)}: {_ascii(info.path.name)}: {fs.rows:,} rows in the train period or without "
              f"a readable time; kept {kept:,}, dropped {fs.rows - kept:,}", flush=True)
    return rs.acc.result(), rs, stats


def _overlap_records(rs: RunState, stats: list[FileStats]) -> list[dict]:
    """Pairs of files that hold ticks at the same times; times are shown only up to the cutoff, and only
    train-period rows are counted."""
    out = []
    for (j, k), rec in sorted(rs.pairs.items()):
        r = {"earlier_file": stats[j].name, "later_file": stats[k].name,
             "later_file_rows_in_overlap_train_period": rec["rows_train"]}
        if rec["rows_train"]:
            r["span_utc"] = [_iso_ns(rec["lo"]) + "Z",
                             _iso_ns(rec["hi"]) + "Z" + (" (and after the cutoff)" if rec["after_cutoff"] else "")]
        else:
            r["span_utc"] = "after the cutoff (not shown)"
        out.append(r)
    return out


def check_reads(stats: list[FileStats], rs: RunState, st: Settings) -> tuple[list[str], dict]:
    """Refuse (DataError) or warn about what the read showed; returns (warnings, overlap record)."""
    warns: list[str] = []
    total = np.sum([fs.counts for fs in stats], axis=0)
    rows = int(sum(fs.rows for fs in stats))
    pairs = _overlap_records(rs, stats)
    overlap_rows = int(sum(fs.overlap_rows_train for fs in stats))
    overlap = {"mode": st.overlap,
               "rule": "a tick of a later file overlaps when an earlier file has ticks right around it (no whole "
                       "clock minute without ticks in between); a file that only fills a gap in another file does "
                       "not overlap it",
               "overlapping_file_pairs": pairs,
               "later_file_rows_in_overlap_train_period": overlap_rows}
    if pairs:
        listing = "; ".join(
            f"{_ascii(x['earlier_file'])} and {_ascii(x['later_file'])}: "
            + (x["span_utc"] if isinstance(x["span_utc"], str) else
               f"{x['later_file_rows_in_overlap_train_period']:,} ticks of {_ascii(x['later_file'])} from "
               + " to ".join(x["span_utc"]))
            for x in pairs[:6]) + (f"; and {len(pairs) - 6} more pair(s)" if len(pairs) > 6 else "")
        if st.overlap == "refuse":
            raise DataError(f"input files overlap: a later file has ticks at times where an earlier file already has "
                            f"ticks ({listing}). They are probably the same ticks twice, which would double "
                            "tick_volume on those bars (two exports that both contain the same day do this). Fix: "
                            "leave out the duplicate file or days, or add --overlap drop (drops the later file's "
                            "ticks only where an earlier file already has ticks), or --overlap keep (keeps both; only "
                            "if the files really hold different ticks, e.g. two different feeds)")
        if st.overlap == "drop":
            warns.append(f"input files overlap ({listing}); --overlap drop dropped "
                         f"{int(total[R['overlap_with_earlier_file']]):,} train-period ticks of the later files there")
            heavy = [(fs, int(fs.counts[R["overlap_with_earlier_file"]])) for fs in stats]
            heavy = [(fs, n) for fs, n in heavy if fs.rows >= 20 and n > DROP_FAIL_SHARE * fs.rows]
            if heavy:                          # correct when the files repeat each other, but worth a look
                what = "; ".join(f"{_ascii(fs.name)}: {n / fs.rows:.1%} of its train-period rows ({n:,} of "
                                 f"{fs.rows:,})" for fs, n in heavy[:6])
                warns.append(f"!!! --overlap drop dropped a large share of a file because an earlier file already has "
                             f"ticks at those times ({what}). That is right when the files repeat the same ticks (a "
                             "copy of a file, or exports that share days); if the files come from different feeds, "
                             "convert them in separate runs instead")
        else:
            warns.append(f"!!! input files overlap ({listing}); --overlap keep kept both, so "
                         f"{overlap_rows:,} train-period ticks may be counted twice (tick_volume too high there)")
    problem = int(sum(total[R[r]] for r in PROBLEM_REASONS))
    bad_files = [fs for fs in stats if fs.rows >= 20 and fs.problem_drops() > DROP_FAIL_SHARE * fs.rows]
    share = problem / rows if rows else 0.0
    if rows and (share > DROP_FAIL_SHARE or bad_files):
        worst = max(stats, key=lambda fs: fs.problem_drops() / max(fs.rows, 1))
        top = sorted(((int(worst.counts[R[r]]), r) for r in PROBLEM_REASONS), reverse=True)[0]
        msg = (f"{share:.1%} of the rows were dropped as unreadable or invalid"
               + (f"; in {_ascii(worst.name)} {worst.problem_drops() / max(worst.rows, 1):.1%}, mostly {top[1]}"
                  if worst.rows else "")
               + ". A wrong --format, --time-format (day/month order), column, --sep or --tz usually causes this: "
                 "check with --dry-run.")
        if not st.allow_drops:
            print("rows dropped, by reason:")
            _print_counts(total)
            raise DataError(msg + " If these drops are expected, add --allow-drops")
        warns.append("!!! " + msg + " (--allow-drops: converted anyway)")
    elif rows and share > DROP_WARN_SHARE:
        warns.append(f"{share:.2%} of the rows were dropped as unreadable or invalid (see the counts by reason); "
                     "check them if that is more than you expect")
    if st.tf_s < 86400 and int(total[0]) > 0 and not rs.any_time_of_day:
        raise DataError("every tick time is exactly 00:00:00: the time column seems to hold dates only, so each "
                        "day's ticks would land in one bar. If the time of day is in another column, pass "
                        "--date-col <date column> --time-col <time column> (--format generic)")
    for fs in stats:
        if fs.back_jumps:
            warns.append(f"{_ascii(fs.name)}: the times jump back by more than a day {fs.back_jumps:,} time(s) inside "
                         "the file. If it should be in time order, the day and month may be swapped "
                         "(check --time-format: %d/%m or %m/%d) or the file mixes periods")
    return warns, overlap


def run(a, argv: list[str], script: str) -> int:
    st = make_settings(a)
    files, notes, input_warns = expand_inputs(a.input)
    for p in files:
        for out in (st.train_path, st.locked_path, st.manifest_path):
            if os.path.normcase(str(p)) == os.path.normcase(str(out)):
                raise UsageError(f"--input {_qp(p)} is one of this run's output files")
        if st.compare is not None and os.path.normcase(str(p)) == os.path.normcase(str(st.compare)):
            raise UsageError("--compare-with must not also be an --input")
    if st.compare is not None and os.path.normcase(str(st.compare)) == os.path.normcase(str(st.train_path)):
        raise UsageError("--compare-with must not be this run's own train output; compare with the Dukascopy file")
    for note in notes:
        print(f"note: {note}")
    for w in input_warns:
        print(f"WARNING: {w}")
    st.warnings.extend(input_warns)
    infos = [sniff(p, a) for p in files]
    fmts = sorted({i.fmt for i in infos})
    if len(fmts) > 1:
        raise UsageError("the inputs have different formats (" + ", ".join(f"{_ascii(i.path.name)}: {i.fmt}"
                                                                          for i in infos) + "); convert one format "
                         "per run")
    for info in infos:
        st.warnings.extend(resolve_tz(info, st.tz, a.dry_run, a.force_tz))
    if a.dry_run:
        return dry_run(infos, st)
    why = foreign_out_problem(st)
    if why:
        raise UsageError(_foreign_out_message(st, why))
    check_outputs(st, a.overwrite)
    for k, i in enumerate(infos, 1):
        print(f"input {k}/{len(infos)}: {_ascii(i.path.name)}: {_ascii(i.fmt_note)}; {i.encoding or 'parquet'}"
              + (f", {i.sep_name()} separated" if i.kind == "text" else ""), flush=True)
    with_volume = "vol" in infos[0].roles
    print(f"converting {len(infos)} file(s) to {st.symbol} {st.tf} bars (price: {st.price}; time zone: "
          f"{_ascii(infos[0].tz_note)})", flush=True)
    print("  hashing the inputs (sha256) ...", flush=True)
    for info in infos:
        info.sha256 = _sha256_file(info.path)
    P, rs, fstats = process_inputs(infos, st, with_volume)
    counts = np.sum([fs.counts for fs in fstats], axis=0)
    bars = bars_frame(P, with_volume)
    del P
    if not len(bars):
        read = int(counts.sum())
        if read:
            print("rows dropped, by reason:")
            _print_counts(counts)
            raise DataError(f"no usable ticks: all {read:,} rows read were dropped (reasons above); check --format, "
                            "--tz, --sep and the column flags with --dry-run")
        raise DataError("no usable ticks: the inputs hold no data rows (only a header?), or none before the cutoff")
    read_warns, overlap = check_reads(fstats, rs, st)
    st.warnings.extend(read_warns)
    n_train = int(np.searchsorted(bars["time"].to_numpy(), st.cutoff_s, side="left"))   # bars are sorted by time
    train = bars.iloc[:n_train]                 # row slices, not copies (the index is not written)
    locked = bars.iloc[n_train:]
    del bars
    if not len(train):
        raise DataError(f"no bars fall before the cutoff ({_iso_s(st.cutoff_s)}): there is nothing to train on, "
                        "so no train file was written. Your data starts at or after the cutoff; the cutoff "
                        "keeps that period as the locked holdout")
    train_bytes = _parquet_bytes(train)
    locked_bytes = _parquet_bytes(locked) if len(locked) else None
    n_locked = len(locked)
    del locked
    stats = train_stats(train, st.tf_s)
    compare = None
    if st.compare is not None:
        compare = compare_with_reference(train, st.compare, st, infos[0].tz if not infos[0].aware else None)
        st.warnings.extend(compare.get("warnings", []))
    if len(train) < MIN_BARS_ALPHAMASTER:
        st.warnings.append(f"the train file has only {len(train)} bars; AlphaMaster needs at least "
                           f"{MIN_BARS_ALPHAMASTER} (Config.MIN_BARS) and will refuse it")
    if not stats["ohlc_valid"]:
        st.warnings.append("some bars have high/low inconsistent with open/close (should not happen)")
    cnt = _reason_counts(counts)
    read = int(counts.sum())
    record = {
        "label": "research only",
        "tool": TOOL, "tool_version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command_line": subprocess.list2cmdline(["python", script] + list(argv)),
        "argv": list(argv),
        "git_commit": _git_commit(),
        "versions": _versions(),
        "note": "Bars built from the user's own tick data (their broker's feed or Dukascopy), for research only. "
                "Broker feeds differ from Dukascopy in prices, spreads, session hours and time stamps. tick_volume "
                "here is the number of ticks per bar, NOT the traded volume in the Dukascopy research file, so "
                "results of volume features (vol_ratio, vol_z, pv_corr, vwap_dev, obv_slope, mfi14) are not "
                "comparable between the two files.",
        "settings": {"symbol": st.symbol, "timeframe": st.tf, "bar_seconds": st.tf_s, "price": st.price,
                     "time_zone": (st.tz.text if st.tz else None),
                     "time_zone_used": sorted({i.tz_note for i in infos}),
                     "cutoff_utc": _iso_s(st.cutoff_s), "chunksize": st.chunksize,
                     "train_overlaps_default_locked_period": st.later_cutoff,
                     "overlap": st.overlap, "allow_drops": st.allow_drops,
                     "columns": list(train.columns),
                     "column_notes": {"time": "int64 UTC epoch seconds, bar open",
                                      "open/high/low/close": f"{st.price} price ticks (first, max, min, last)",
                                      "tick_volume": "number of ticks kept in the bar (float64); NOT traded volume "
                                                     "(the Dukascopy research file's tick_volume is Dukascopy volume)",
                                      "spread": "ask - bid of the bar's last tick, price units",
                                      "spread_mean": f"mean ask - bid over the bar's ticks (sums kept in units "
                                                     f"of {SPREAD_QUANTUM:g})",
                                      **({"volume_sum": f"sum of --vol-col over the bar's ticks (units of "
                                                        f"{VOLUME_QUANTUM:g})"} if with_volume else {})}},
        "inputs": [{"path": str(i.path), "size_bytes": i.size, "sha256": i.sha256, "format": i.fmt,
                    "format_note": i.fmt_note, "encoding": i.encoding or "parquet",
                    "encoding_note": i.encoding_note, "separator": i.sep_name() if i.kind == "text" else None,
                    "columns_used": i.roles, "time_kind": i.time_kind, "epoch_unit": i.epoch_unit,
                    "time_zone_used": i.tz_note,
                    "rows_counted": fs.rows, "rows_kept": int(fs.counts[0]),
                    "rows_dropped": {k: v for k, v in _reason_counts(fs.counts).items() if v},
                    "backward_time_jumps_over_1_day": fs.back_jumps} for i, fs in zip(infos, fstats)],
        "rows": {"scope": "rows whose UTC time is before the cutoff, plus rows whose time could not be read; rows "
                          "in the locked period are not counted",
                 "read": read, "kept": int(counts[0]), "dropped_total": read - int(counts[0]),
                 "dropped": cnt, "dropped_reason_help": REASON_HELP,
                 "problem_drop_share": (sum(cnt[r] for r in PROBLEM_REASONS) / read) if read else None,
                 "dst_nat_rows": cnt["dst_nonexistent_or_ambiguous_time"],
                 "exact_duplicate_rows_within_chunk": rs.dups,
                 "exact_duplicate_note": "same UTC time, bid and ask within one chunk; counted, NOT dropped",
                 "overlap": overlap,
                 **({"volume_missing_rows_counted_as_0": rs.vol_missing} if with_volume else {})},
        "train": {"file": str(st.train_path), "sha256": hashlib.sha256(train_bytes).hexdigest(), **stats},
        "locked": {"file": str(st.locked_path) if locked_bytes is not None else None, "bars": n_locked,
                   "sha256": hashlib.sha256(locked_bytes).hexdigest() if locked_bytes is not None else None},
        # this tool owns these subfolders of --out (a later run may write into them; see foreign_out_problem)
        "folders_made_by_this_tool": ["train", "locked_holdout"],
        "compare": compare,
        "warnings": list(st.warnings),
    }
    record = _plain(record)
    manifest_bytes = (json.dumps(record, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
    why = foreign_out_problem(st)              # again, right before writing
    if why:
        raise UsageError(_foreign_out_message(st, why))
    check_outputs(st, a.overwrite)
    publish(st, {st.train_path: train_bytes, st.locked_path: locked_bytes, st.manifest_path: manifest_bytes})
    a.published = True
    print_summary(record, st)
    return EXIT_OK


def _versions() -> dict:
    v = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
         "platform": platform.platform()}
    try:
        import pyarrow
        v["pyarrow"] = pyarrow.__version__
    except ImportError:
        pass
    return v


def _wd(t_ns: int) -> str:
    return _WEEKDAYS[(int(t_ns) // (86400 * NS) + 3) % 7]


def dry_run(infos: list[InputInfo], st: Settings) -> int:
    print("DRY RUN - nothing is written")
    for k, i in enumerate(infos, 1):
        print(f"input {k}/{len(infos)}: {_ascii(i.path)} ({i.size:,} bytes)")
        print(f"  format    : {_ascii(i.fmt_note)}")
        print(f"  encoding  : {i.encoding or 'parquet'} ({_ascii(i.encoding_note)})")
        if i.kind == "text":
            print(f"  separator : {i.sep_name()}")
        print(f"  columns   : {_cols_text(i.columns, st.cutoff_s, 12)}")
        print(f"  time zone : {_ascii(i.tz_note)}")
    first = infos[0]
    state = StreamState()
    try:
        parts = [parse_chunk(first, df, state, keep_raw=True)
                 for df in iter_raw(first, DRY_RUN_ROWS, nrows=DRY_RUN_ROWS)]
    except READ_ERRORS as e:
        raise DataError(f"could not read {_ascii(first.path)} (in its first {DRY_RUN_ROWS:,} rows): "
                        f"{type(e).__name__}: {_ascii(e)}")
    if not parts:
        print("the first file has no data rows")
        return EXIT_OK
    raw = pd.concat([p.raw_time for p in parts], ignore_index=True)
    local = np.concatenate([p.local for p in parts])
    bid = np.concatenate([p.bid for p in parts])
    ask = np.concatenate([p.ask for p in parts])
    reason = np.concatenate([p.reason for p in parts])
    utc = None if parts[0].utc is None else np.concatenate([p.utc for p in parts])
    cutoff_ns = st.cutoff_s * NS
    # hide anything that is, or could be, in the locked period: UTC at or after the cutoff; with no UTC
    # (no --tz, or a daylight-saving gap) a local time from cutoff - 14 h on; an unreadable time whose
    # text could name such a date
    near = (local != NAT) & (local >= cutoff_ns - LOCAL_SLACK_NS)
    hide = ((utc != NAT) & (utc >= cutoff_ns)) | ((utc == NAT) & near) if utc is not None else near
    unreadable = np.flatnonzero(local == NAT)
    if len(unreadable):                         # shown only when the text surely names a time before the cutoff
        texts = raw.iloc[unreadable].astype(str).to_numpy()
        verdict = {x: not _value_shown(x, st.cutoff_s) for x in pd.unique(texts)[:200]}
        hide[unreadable] = [verdict.get(x, True) for x in texts]
    shown = np.flatnonzero(~hide)
    if hide.any():                              # no row counts here: they would count locked-period ticks
        print(f"sample rows from the start of {_ascii(first.path.name)} (up to {DRY_RUN_ROWS:,} rows are read):")
        print(f"  some of these rows are (or may be) at or after the cutoff ({_iso_s(st.cutoff_s)}, locked holdout "
              "period): they are not shown and not counted")
    else:
        print(f"first {len(raw):,} rows of {_ascii(first.path.name)}:")
    if len(shown):
        pick = shown[np.unique(np.linspace(0, len(shown) - 1, num=min(5, len(shown))).round().astype(int))]
        print(f"  {'raw time':<28} {'parsed local time':<28} {'UTC time':<28} {'UTC offset':<10} {'bid':>11} "
              f"{'ask':>11}  status")
        for j in pick:
            if first.aware:
                loc = "(offset in the text)"
            elif local[j] == NAT:
                loc = "NaT"
            else:
                loc = f"{_wd(local[j])} {_iso_ns(local[j])}"
            u = "(needs --tz)" if utc is None else ("NaT" if utc[j] == NAT else f"{_wd(utc[j])} {_iso_ns(utc[j])}")
            off = ""
            if utc is not None and not first.aware and NAT not in (int(local[j]), int(utc[j])):
                off = "UTC" + _offset_text(int(local[j] - utc[j]) // NS)
            print(f"  {_ascii(raw.iloc[j])[:28]:<28} {loc:<28} {u:<28} {off:<10} {bid[j]:>11.5f} {ask[j]:>11.5f}  "
                  f"{REASONS[reason[j]]}")
        cnt = np.bincount(reason[shown], minlength=len(REASONS))
        dropped = {REASONS[i]: int(cnt[i]) for i in range(1, len(REASONS)) if cnt[i]}
        print(f"  rows before the cutoff in this sample: {int(cnt[0]):,} kept, {int(cnt[1:].sum()):,} dropped"
              + (f" ({', '.join(f'{k} {v:,}' for k, v in dropped.items())})" if dropped else ""))
        if utc is not None:
            ok = shown[reason[shown] == 0]
            nb = len(np.unique((utc[ok] // (st.tf_s * NS)))) if len(ok) else 0
            print(f"  these rows would make {nb:,} {st.tf} bar(s)")
        else:
            print("  the bar count needs --tz")
        print("  check the time zone: your MT5 Market Watch clock shows the broker's server time (the 'parsed local "
              "time' column). XAUUSD reopens after the weekend on Sunday 22:00 UTC (US summer) or 23:00 UTC (US "
              "winter), and the first bars after a weekend should show that in the UTC column")
    for path in (st.train_path, st.locked_path, st.manifest_path):
        print(f"would write: {_ascii(path)}" + ("  (EXISTS: needs --overwrite)" if path.exists() else ""))
    why = foreign_out_problem(st)
    if why:
        st.warnings.append(_foreign_out_message(st, why) + " (a real run would stop here)")
    if st.compare is not None:
        print(f"the comparison with {_ascii(st.compare.name)} runs only in a real run (not in a dry run)")
    for w in st.warnings:
        print(f"WARNING: {_ascii(w)}")
    print("dry run done: nothing was written")
    return EXIT_OK


def print_summary(rec: dict, st: Settings) -> None:
    s, tr, rows = rec["settings"], rec["train"], rec["rows"]
    print(f"ticks_to_bars: {s['symbol']} {s['timeframe']} bars, price = {s['price']}, cutoff {s['cutoff_utc']}")
    for k, i in enumerate(rec["inputs"], 1):
        print(f"  input {k}: {_ascii(i['path'])} ({i['format']}, {i['encoding']}"
              + (f", {i['separator']}" if i["separator"] else "") + f", {i['size_bytes']:,} bytes)")
    print(f"  time zone : {_ascii('; '.join(s['time_zone_used']))}")
    print(f"  rows (train period + unreadable times): read {rows['read']:,}, kept {rows['kept']:,}, "
          f"dropped {rows['dropped_total']:,}")
    for k, v in rows["dropped"].items():
        if v:
            print(f"     {k:<36} {v:>12,}  ({REASON_HELP[k]})")
    print(f"  exact duplicate rows (within a chunk; kept): {rows['exact_duplicate_rows_within_chunk']:,}; "
          f"DST NaT rows: {rows['dst_nat_rows']:,}")
    if rows["overlap"]["overlapping_file_pairs"]:
        print(f"  overlapping input files: {len(rows['overlap']['overlapping_file_pairs'])} pair(s), "
              f"--overlap {rows['overlap']['mode']}")
    print(f"  train : {tr['bars']:,} bars, {tr['first_bar_utc']} .. {tr['last_bar_utc']}")
    print(f"          {_ascii(tr['file'])}")
    print(f"          sha256 {tr['sha256']}")
    lk = rec["locked"]
    print(f"  locked: {lk['bars']:,} bars, sha256 {lk['sha256'] or '(no locked file: no bars at or after the cutoff)'}")
    print("          (only the bar count and sha256 of the locked part are shown; never open that file)")
    print("  bars per year: " + ", ".join(f"{y} {n:,}" for y, n in tr["bars_per_year"].items()))
    g = tr["gaps"]
    print(f"  gaps: {g['over_1_bar']:,} longer than one bar; {g[f'over_{GAP_LONG_HOURS}h']:,} longer than "
          f"{GAP_LONG_HOURS} h")
    for x in g[f"over_{GAP_LONG_HOURS}h_longest"]:
        print(f"     {x['start_utc']} -> {x['end_utc']}  ({x['hours_between_bar_opens']:.0f} h)")
    tp = tr["ticks_per_bar"]
    print(f"  ticks per bar: median {_fmt(tp['median'], ',.0f')}, p10 {_fmt(tp['p10'], ',.0f')}; bars with fewer "
          f"than 5 ticks: {tp['bars_with_fewer_than_5_ticks']:,}")
    for key, label in (("spread_last_tick", "spread (last tick)"), ("spread_mean", "spread_mean (bar mean)")):
        sp = tr[key]
        print(f"  {label:<23}: median {_fmt(sp['median'])} ({_fmt(sp['median_pct_of_median_close'], '.4f')}% of "
              f"median close), p90 {_fmt(sp['p90'])}, p99 {_fmt(sp['p99'])}")
    hrs = tr["median_spread_by_utc_hour"]["spread_mean"]
    print("  median spread_mean by UTC hour: " + " ".join(f"{h}h {_fmt(v, '.3f')}" for h, v in hrs.items()
                                                          if v is not None))
    if tr.get("weekend"):
        wk = tr["weekend"]
        print("  weekly reopen (first bar after a 24 h+ break, UTC): "
              + (", ".join(f"{k} x{v}" for k, v in wk["reopen_bar_utc_most_common"].items()) or "no such breaks"))
        if "XAU" in st.symbol.upper() or "GOLD" in st.symbol.upper():
            print("     (XAUUSD usually reopens Sun 22:00 UTC in US summer, Sun 23:00 UTC in US winter)")
    print(f"  largest bar ranges (flagged, not removed; median range {_fmt(tr['median_bar_range'])}):")
    for x in tr["largest_ranges"]:
        print(f"     {x['time_utc']}  range {x['range']:.4f}  ({_fmt(x['ratio_to_median_range'], '.1f')}x median)")
    c = rec.get("compare")
    if c:
        print(f"  compare with {_ascii(c['reference_file'])}:")
        if c.get("error"):
            print(f"     {c['error']}")
        else:
            print(f"     overlapping bars {c['overlapping_bars']:,}; your bars matched "
                  f"{_fmt(c['share_of_your_bars_matched_in_reference'], '.1%')}, reference bars matched "
                  f"{_fmt(c['share_of_reference_bars_matched_in_yours'], '.1%')}")
            print(f"     |close difference|: median {_fmt(c['abs_close_diff_median'])}, p90 "
                  f"{_fmt(c['abs_close_diff_p90'])} (signed median {_fmt(c['close_diff_median_signed'])})")
            print("     return correlation by lag (bars): " + "  ".join(
                f"{k} {_fmt(v, '.3f')}" for k, v in c["return_corr_by_lag"].items()))
            print(f"     best lag: {c['best_lag_bars']} bar(s) (positive = your bars are stamped later than the "
                  "reference; 0 = the time zones agree)")
            reg = c.get("best_lag_by_dst_regime") or {}
            print("     best lag by daylight-saving period: " + ", ".join(
                f"{k.replace('_', ' ')} {v['best_lag_bars'] if v['best_lag_bars'] is not None else 'n/a'}"
                for k, v in reg.items()))
            vol = c.get("volume") or {}
            if vol.get("log_volume_corr") is not None:
                print(f"     log volume correlation with the reference's {vol.get('reference_volume_column')}: "
                      f"{vol['log_volume_corr']:.3f} (tick count vs traded volume)")
    print("  note: tick_volume = number of ticks per bar, not the Dukascopy file's traded volume; volume-feature "
          "results are not comparable between the two files")
    print(f"  manifest: {_ascii(st.manifest_path)}")
    for w in rec["warnings"]:
        print(f"WARNING: {_ascii(w)}")


# ---------------------------------------------------------------------------------------
# command line

def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ticks_to_bars.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Convert your own tick files (MT5, Dukascopy or any CSV/Parquet) into AlphaMaster bar files "
                    "with a fixed locked-holdout split and a validation manifest. Research only.",
        epilog=HELP_EPILOG)
    ap.add_argument("--input", action="append", required=True, metavar="PATH",
                    help="tick file, folder or pattern; give it several times for several (read in that order)")
    ap.add_argument("--out", required=True, metavar="DIR",
                    help="output folder (train\\, locked_holdout\\, manifest); use a new folder, not the Dukascopy "
                         "research folder")
    ap.add_argument("--format", choices=FORMATS, default="auto", help="input format (default: auto, from the header)")
    ap.add_argument("--tz", metavar="TZ", help="time zone of the times in the files: UTC, +02:00, -05:00, "
                                               "Europe/Athens or ny+7 (required for MT5)")
    ap.add_argument("--force-tz", action="store_true",
                    help="apply --tz to dukascopy-node epoch times, which are UTC; only for downloads made with "
                         "dukascopy-node's --utc-offset option")
    ap.add_argument("--symbol", default="XAUUSD", help="symbol for the file name (default XAUUSD)")
    ap.add_argument("--tf", default="H1", type=str.upper, metavar="{" + ",".join(TF_SECONDS) + "}",
                    help="bar timeframe (default H1)")
    ap.add_argument("--price", choices=("bid", "mid", "ask"), default="bid",
                    help="price series for OHLC (default bid, like the Dukascopy research file)")
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF,
                    help=f"start of the locked holdout (default {DEFAULT_CUTOFF}; later dates need "
                         "--allow-later-cutoff)")
    ap.add_argument("--allow-later-cutoff", action="store_true",
                    help=f"allow a cutoff after {DEFAULT_CUTOFF} (that period is your locked holdout)")
    ap.add_argument("--compare-with", metavar="PARQUET",
                    help="AlphaMaster-layout train file to check time zone and prices against (same timeframe; "
                         "the time-zone check needs --tf H1)")
    ap.add_argument("--overlap", choices=OVERLAP_MODES, default="refuse",
                    help="input files that hold ticks at the same times (e.g. two exports that both contain one "
                         "day): refuse (default), drop (drop the later file's ticks where an earlier file already "
                         "has ticks) or keep (count both)")
    ap.add_argument("--allow-drops", action="store_true",
                    help="convert even when more than 5%% of the rows (or of one file) cannot be read or are invalid")
    ap.add_argument("--chunksize", type=int, default=2_000_000,
                    help=CHUNKSIZE_HELP)
    ap.add_argument("--overwrite", action="store_true", help="replace outputs this tool made earlier (never others)")
    ap.add_argument("--normal-priority", action="store_true", help="do not lower the process priority")
    ap.add_argument("--dry-run", action="store_true", help="sniff the inputs and show sample rows; write nothing")
    g = ap.add_argument_group("generic format (any CSV or Parquet)")
    g.add_argument("--time-col", help="name of the time column")
    g.add_argument("--date-col", help="name of a separate date column; joined to --time-col with a space, and "
                                      "--time-format then describes both, e.g. \"%%d/%%m/%%Y %%H:%%M:%%S.%%f\"")
    g.add_argument("--bid-col", help="name of the bid column")
    g.add_argument("--ask-col", help="name of the ask column")
    g.add_argument("--vol-col", help="optional volume column (summed into an extra volume_sum column)")
    g.add_argument("--time-format", metavar="FMT",
                   help="strftime format of the time strings, e.g. \"%%d.%%m.%%Y %%H:%%M:%%S.%%f\" (epoch "
                        "numbers in s/ms/us/ns are detected without it)")
    g.add_argument("--no-header", action="store_true",
                   help="the file has no header row: its columns are called col0, col1, col2, ...")
    g.add_argument("--sep", metavar="SEP", help="CSV separator: tab, comma, semicolon, pipe or one character "
                                                "(default: detected)")
    g.add_argument("--encoding", metavar="ENC",
                   help="text encoding of CSV files when it is not detected right, e.g. gbk, cp936, cp1252, utf-8")
    return ap


_PATH_FLAGS = ("--input", "--out", "--compare-with")


def _fix_negative_tz(argv: list[str]) -> list[str]:
    """'--tz -05:00' -> '--tz=-05:00': argparse would read a value starting with '-' as a new flag."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--tz" and i + 1 < len(argv) and re.match(r"-\d", argv[i + 1]):
            out.append("--tz=" + argv[i + 1])
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def _quote_problem(argv: list[str]) -> str | None:
    """A message when a path value holds a double quote, the trace of PowerShell's trailing-backslash trap
    ("D:\\My Ticks\\" reaches Python as 'D:\\My Ticks" --out ...'); None when all is well."""
    for i, tok in enumerate(argv):
        for flag in _PATH_FLAGS:
            if tok == flag and i + 1 < len(argv):
                val = argv[i + 1]
            elif tok.startswith(flag + "="):
                val = tok[len(flag) + 1:]
            else:
                continue
            if '"' in val and not Path(val).exists():
                return (f"the value of {flag} contains a double quote: {_qp(val)}. In PowerShell a quoted path that "
                        "ends in a backslash, like \"D:\\My Ticks\\\", swallows its closing quote and the rest of the "
                        "line. Remove the backslash before the closing quote: \"D:\\My Ticks\"")
    return None


def main(argv: list[str] | None = None) -> int:
    print(BANNER, flush=True)
    script = TOOL
    if argv is None:
        argv = sys.argv[1:]
        script = sys.argv[0] or TOOL
    argv = _fix_negative_tz([str(x) for x in argv])
    problem = _quote_problem(argv)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return EXIT_USAGE
    ap = _parser()
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    a.published = False
    try:
        if not a.normal_priority:
            print(lower_priority(), flush=True)
        return run(a, argv, script)
    except UsageError as e:
        print(f"error: {_ascii(e)}", file=sys.stderr)
        return EXIT_USAGE
    except DataError as e:
        print(f"data error: {_ascii(e)}", file=sys.stderr)
        return EXIT_DATA
    except MemoryError:
        print("data error: out of memory; try a smaller --chunksize, a longer --tf, or convert in yearly pieces",
              file=sys.stderr)
        return EXIT_DATA
    except OSError as e:
        print(f"error: could not read or write a file: {_ascii(e)}. "
              + ("The output files had been written before this happened." if a.published else
                 "No output files were written (outputs of an earlier run, if any, are unchanged)."),
              file=sys.stderr)
        return EXIT_DATA
    except KeyboardInterrupt:
        if a.published:
            print("stopped by Ctrl+C after the files were written (they are complete; only the summary was cut)",
                  file=sys.stderr)
        else:
            print("stopped by Ctrl+C: no output files were written (outputs of an earlier run, if any, are "
                  "unchanged)", file=sys.stderr)
        return EXIT_DATA
    except Exception as e:  # noqa: BLE001 - last resort: one plain line instead of a traceback
        import traceback
        where = traceback.extract_tb(e.__traceback__)[-1]
        print(f"unexpected error: {type(e).__name__}: {_ascii(e)} (at {_ascii(Path(where.filename).name)} line "
              f"{where.lineno}, in {where.name}). "
              + ("The output files had been written before this happened." if a.published else
                 "No output files were written."), file=sys.stderr)
        return EXIT_DATA


if __name__ == "__main__":
    sys.exit(main())
