#!/usr/bin/env python3
r"""scripts/research/dukascopy_web_to_node.py - Dukascopy website candles -> dukascopy-node CSV layout.

Convert Dukascopy website candle exports (XAU/USD, 15-minute bars, BID and ASK) into the CSV layout
that dukascopy-node 1.50.0 writes and propkit reads.

zeno downloads XAU/USD 15-minute candles in a browser from Dukascopy's "Historical Data Export" page,
one or several files per side. This script turns them into what

    npx dukascopy-node@1.50.0 -i xauusd -t m15 -p bid|ask -v -f csv

would have written (one file per side):

    timestamp,open,high,low,close,volume
    1420156800000,1186.585,1187.315,1186.1,1186.835,1234.56

  * timestamp: the bar OPEN as UTC epoch milliseconds (dukascopy-node's default: no date format);
  * open/high/low/close: the website's decimal values, unchanged. They are written the way JavaScript
    prints a number (no trailing zeros: "1186.100" -> "1186.1"); the decimal value itself never changes;
  * volume: the website's own value, unchanged (in whatever unit the page was set to; propkit ignores it);
  * flats: dukascopy-node's default (ignoreFlats true) removes every candle whose volume is exactly 0
    (dist/cli/index.js, getOHLC: input.filter((data) => data[5] !== 0)), so a 15-minute bar is missing
    exactly when all its minutes had volume 0, i.e. when the bar's volume is 0 (assuming, as is inferred
    and not verified, that the page's bar volume is the sum of its minutes' volumes). The same rule is
    applied here, to each side on its own, and the number dropped is printed;
  * one row per bar, sorted by time. Rows opening at or after 2025-09-28 00:00 UTC (the holdout lock) are
    cut as soon as their time is read, before any other cell of the row is looked at; only their number is
    printed.

Accepted website layouts (each one can be placed in UTC without guessing):
  * time column 'Gmt time' (or 'GMT time', 'Time (UTC)', 'UTC time' ...): UTC;
  * time column 'Local time' (or any other name) ONLY when every time ends in an explicit offset such as
    ' GMT+0800' or ' GMT+08:00'; each row is converted with its own offset;
  * times DD.MM.YYYY HH:MM:SS.fff (also without .fff or without seconds) or YYYY-MM-DD / YYYY.MM.DD;
  * comma, semicolon, tab or pipe separated; a decimal comma only in semicolon/tab/pipe files;
  * UTF-8 with or without a byte-order mark, UTF-16 with a byte-order mark; LF or CRLF line ends;
  * one or several files per side (parts of the range). Overlapping parts must agree: exact duplicate
    rows are dropped and counted, rows that disagree stop the conversion.
Refused (exit 3, nothing written): local times without an offset, times with slashes (Excel), prices in
scientific notation (Excel), bars that are not 15 minutes (each file is checked on its own, and a file name
that says another candle size is refused) or not on 15-minute UTC boundaries, prices that are not
XAU/USD-like, tick files, a bid and an ask file that look swapped or are the same side twice (checked over
all bars and again per calendar month), bid and ask files that do not hold exactly the same bar times, and
anything propkit itself would refuse. A file whose week opens look shifted (off the hour, outside
21:00-23:00 UTC, or at other times than the other files' in the same months) gets a WARNING line.

The side of a file comes from its name (BID or ASK as a separate word, as in Dukascopy's
XAUUSD_Candlestick_15_M_BID_01.01.2015-31.12.2015.csv), or from --bid / --ask.

Usage (Windows PowerShell, from C:\Users\PC\Documents\GitHub\AlphaMaster-propkit):
    $env:PYTHONUTF8 = "1"
    ..\AlphaMaster\.venv\Scripts\python.exe scripts\research\dukascopy_web_to_node.py `
        C:\MarketData\dukascopy\web_m15 --out C:\MarketData\dukascopy\xauusd_m15_web
    (add --check-only to check the files without writing anything)

Output: <out>\xauusd_m15_bid.csv and <out>\xauusd_m15_ask.csv. The output folder must be new or empty
and must not be the input folder; nothing is ever overwritten; paths containing 'locked_holdout' or
ending in '.locked' are refused before anything is opened.

Exit codes: 0 ok; 2 usage error (arguments, paths, folders); 3 data problem (nothing written).
Standard library only; Python 3.11 or later.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import os
import re
import statistics
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

VERSION = "1.1"

LOCK_S = 1759017600                 # 2025-09-28 00:00:00 UTC, the holdout lock (zeno_pullback_v1, D1)
LOCK_TEXT = "2025-09-28 00:00 UTC"
RANGE_START_S = 1420070400          # 2015-01-01 00:00:00 UTC, D1's range start (propkit cuts earlier bars)
WANT_FIRST_BY_S = 1420416000        # 2015-01-05 00:00 UTC: warn when the data starts later than this
WANT_LAST_FROM_S = 1758844800       # 2025-09-26 00:00 UTC: warn when the data ends before this
BAR_S = 900                         # 15 minutes
GAP_REPORT_S = 3 * 86400            # gaps longer than 3 days are listed (as step E of the PC plan does)
GAP_WARN_S = 4 * 86400              # and a WARNING only above 4 days (a holiday weekend is about 73 to 80 h)
WEEK_GAP_S = 36 * 3600              # a bar after a gap this long is a week (or holiday) open
OUT_HEADER = "timestamp,open,high,low,close,volume"
OUT_NAMES = {"bid": "xauusd_m15_bid.csv", "ask": "xauusd_m15_ask.csv"}
SIDES = ("bid", "ask")
PRICE_MIN, PRICE_MAX = 100.0, 20000.0       # every XAU/USD price must lie in here (USD per ounce)
MEDIAN_MIN, MEDIAN_MAX = 250.0, 10000.0     # and the median close in here (2015-2025: about 1,050-3,800)
SWAP_SHARE = 0.5                    # ask close below bid close on more than half the bars: swapped files
UNFINISHED = (".crdownload", ".part", ".partial", ".download", ".opdownload", ".tmp")
_VOLUME_NAME = re.compile(r"(?:tick[\s_]*)?vol(?:ume)?s?(?![a-z0-9])")   # Volume, Volume (units), Tick volume
EXIT_OK, EXIT_USAGE, EXIT_DATA = 0, 2, 3
MONTH_MIN_BARS = 20                 # per-month bid/ask checks need at least this many common bars
WEEK_OPEN_LO, WEEK_OPEN_HI = 21, 23 # week opens are expected on the hour, 21:00 to 23:00 UTC [inferred]

# row tuple fields
R_TS, R_O, R_H, R_L, R_C, R_V, R_VZERO, R_PART, R_LINE = range(9)


class UsageError(Exception):
    """Arguments, paths or folders are wrong (exit 2)."""


class DataError(Exception):
    """The files cannot be converted faithfully (exit 3, nothing written)."""


def say(text: str = "") -> None:
    print(text, flush=True)


def utc_text(ts: int) -> str:
    return _dt.datetime.fromtimestamp(int(ts), _dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


_FOUR_DIGITS = re.compile(r"(?<![0-9])[0-9]{4}(?![0-9])")


def shown(text: str) -> str:
    """A time cell for an error message: shown only when it holds a 4-digit year from 1900 to 2024 and no
    4-digit number of 2025 or later; anything else may be dated on or after the lock, so only the line
    number (printed by the caller) identifies it."""
    t = str(text).strip()
    years = [int(y) for y in _FOUR_DIGITS.findall(t)]
    if years and all(y < 2025 for y in years) and any(1900 <= y for y in years):
        return repr(t[:48])
    return "(value not shown: it may be dated on or after the holdout lock)"


# ---------------------------------------------------------------------------------------------------
# paths

def is_locked_path(text) -> bool:
    low = str(text).strip().replace("\\", "/").rstrip("/").lower()
    return "locked_holdout" in low or low.endswith(".locked")


def check_not_locked(path, what: str) -> None:
    candidates = [str(path)]
    try:
        candidates.append(str(Path(str(path)).expanduser().resolve()))
    except (OSError, RuntimeError, ValueError):
        pass
    if any(is_locked_path(c) for c in candidates):
        raise UsageError(f"refusing the {what} {path}: paths containing 'locked_holdout' or ending in '.locked' "
                         "are the locked holdout, which this script never opens or writes")


def _clean_arg(text: str) -> str:
    # PowerShell turns a trailing backslash before a closing quote into a stray quote: drop it
    return str(text).strip().strip('"').strip("'").strip()


def _norm_path(p: Path) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


_SIDE_TOKEN = re.compile(r"(?<![A-Za-z])(bid|ask)(?![A-Za-z])", re.IGNORECASE)


def side_from_name(name: str) -> str | None:
    found = {m.group(1).lower() for m in _SIDE_TOKEN.finditer(Path(name).stem)}
    return found.pop() if len(found) == 1 else None


def _folder_files(folder: Path) -> list[Path]:
    files = []
    unfinished = []
    try:
        children = sorted(folder.iterdir(), key=lambda p: p.name.lower())
    except OSError as e:
        raise UsageError(f"cannot list the folder {folder}: {e}")
    for child in children:
        if is_locked_path(child.name):
            raise UsageError(f"the folder {folder} holds {child.name}, a locked-holdout name; move it out of the "
                             "folder first (this script never opens it)")
        check_not_locked(child, "input file")      # also a link or junction that leads into a locked path
        if not child.is_file():
            continue
        low = child.name.lower()
        if low.endswith(UNFINISHED):
            unfinished.append(child.name)
        elif low.endswith(".csv"):
            files.append(child)
    if unfinished:
        raise UsageError(f"the folder {folder} holds an unfinished download ({unfinished[0]}): wait until the "
                         "browser has finished saving, or delete that file if the download was cancelled")
    if not files:
        raise UsageError(f"no .csv files in the folder {folder}")
    return files


def collect_inputs(args) -> tuple[dict[str, list[Path]], list[Path]]:
    """Input files per side, and the input folders (files' folders and folders given)."""
    sides: dict[str, list[Path]] = {"bid": [], "ask": []}
    folders: list[Path] = []
    seen: set[str] = set()

    def expand(text: str) -> list[Path]:
        text = _clean_arg(text)
        if not text:
            raise UsageError("an input path is empty")
        check_not_locked(text, "input")
        p = Path(text).expanduser()
        if p.is_dir():
            folders.append(p)
            return _folder_files(p)
        if p.is_file():
            if p.suffix.lower() not in (".csv", ".txt"):
                raise UsageError(f"{p.name}: give the .csv files the export page saved (got '{p.suffix or 'no suffix'}')")
            folders.append(p.parent)
            return [p]
        raise UsageError(f"input not found: {p}")

    def add(side: str, p: Path) -> None:
        key = _norm_path(p)
        if key in seen:
            return
        seen.add(key)
        sides[side].append(p)

    for side in SIDES:
        for text in getattr(args, side) or []:
            for p in expand(text):
                named = side_from_name(p.name)
                if named is not None and named != side:
                    raise UsageError(f"{p.name} was given with --{side} but its name says {named.upper()}; "
                                     "check which file is which")
                add(side, p)
    for text in args.inputs or []:
        for p in expand(text):
            side = side_from_name(p.name)
            if side is None:
                raise UsageError(f"{p.name}: cannot tell from the name whether it is BID or ASK. Rename it so "
                                 "the name contains _BID_ or _ASK_ (as the export page names its files), or "
                                 "pass it with --bid or --ask")
            add(side, p)
    if not sides["bid"] and not sides["ask"]:
        raise UsageError("no input files: give the folder with the downloaded .csv files (or --bid/--ask files)")
    return sides, folders


def check_out_folder(text: str | None, folders: list[Path]) -> Path:
    if not text:
        raise UsageError("--out is required: a new folder for the converted files, e.g. "
                         "--out C:\\MarketData\\dukascopy\\xauusd_m15_web (or use --check-only)")
    text = _clean_arg(text)
    check_not_locked(text, "output folder")
    out = Path(text).expanduser()
    for name in OUT_NAMES.values():
        check_not_locked(out / name, "output file")
    o = _norm_path(out)
    for f in folders:
        d = _norm_path(f)
        if o == d or o.startswith(d.rstrip("\\/") + os.sep):
            raise UsageError(f"--out {out} is the input folder or inside it ({f}); write the converted files to "
                             "a separate folder, e.g. C:\\MarketData\\dukascopy\\xauusd_m15_web")
    if out.exists() and not out.is_dir():
        raise UsageError(f"--out {out} is a file; give a new folder")
    if not out.parent.is_dir():
        raise UsageError(f"the folder above --out does not exist ({out.parent}): check the path; this script "
                         "creates only the output folder itself")
    if out.is_dir():
        try:
            busy = any(out.iterdir())
        except OSError as e:
            raise UsageError(f"cannot list --out {out}: {e}")
        if busy:
            raise UsageError(f"--out {out} already holds files; this script never overwrites anything: give a "
                             "new (or empty) folder")
    return out


# ---------------------------------------------------------------------------------------------------
# reading one file

def read_text(path: Path) -> tuple[str, str]:
    try:
        data = path.read_bytes()
    except OSError as e:
        raise UsageError(f"cannot read {path}: {e} (is it open in another program?)")
    if not data.strip():
        raise DataError(f"{path.name}: the file is empty (0 bytes or only blanks); download this part again")
    if data.startswith(b"\xef\xbb\xbf"):
        enc, raw = "UTF-8 with byte-order mark", data[3:]
        codec = "utf-8"
    elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
        enc, raw, codec = "UTF-16 with byte-order mark", data, "utf-16"
    else:
        enc, raw, codec = "UTF-8", data, "utf-8"
        head = data[:4096]
        if head.count(0) > len(head) // 4:
            raise DataError(f"{path.name}: the file looks like UTF-16 text without a byte-order mark (was it "
                            "saved by Excel?); download this part again and do not open it in Excel")
    try:
        text = raw.decode(codec)
    except UnicodeDecodeError as e:
        raise DataError(f"{path.name}: not readable as text ({codec}: {e}); download this part again")
    return text, enc


def sniff_delimiter(header: str) -> str:
    counts = [(header.count(c), -i, c) for i, c in enumerate((",", ";", "\t", "|"))]
    best = max(counts)
    return best[2] if best[0] > 0 else ","


_DELIM_NAMES = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}
_ZONE_WORD = re.compile(r"\b(eet|eest|cet|cest|est|edt|cst|cdt|pst|pdt|sgt|hkt|jst|bst|msk|ist|aest|wet)\b")


def time_column_kind(norm: str) -> str | None:
    """'utc', 'local', 'zone' (a named zone) or 'unknown' (no zone) for a time column name, else None."""
    has_time = "time" in norm or "date" in norm
    if "local" in norm and has_time:
        return "local"
    if norm in ("gmttime", "utctime", "gmt_time", "utc_time"):
        return "utc"
    if re.search(r"\b(gmt|utc)\b", norm) and (has_time or norm in ("gmt", "utc")):
        z = re.search(r"(gmt|utc)\s*[+-]\s*(\d{1,2})(?::?(\d{2}))?", norm)
        if z and (int(z.group(2)) or int(z.group(3) or 0)):
            return "zone"           # e.g. 'Time (GMT+2)': a fixed label, not a per-row offset
        return "utc"                # 'Gmt time', 'Time (UTC)', 'Time (GMT+0)'
    if has_time and _ZONE_WORD.search(norm):
        return "zone"
    if norm in ("time", "date", "datetime", "date time", "date/time", "date_time", "time/date"):
        return "unknown"
    return None


class Header:
    def __init__(self, cells: list[str], name: str):
        norms = [re.sub(r"\s+", " ", c.strip().strip('"').strip("'").strip().lower()) for c in cells]
        self.ncols = len(cells)
        self.cells = [c.strip() for c in cells]
        shown_header = ",".join(self.cells)[:120]
        names = set(norms)
        if {"ask", "bid"} <= names or {"askprice", "bidprice"} <= names:
            raise DataError(f"{name}: this is a tick file (Ask and Bid columns), not candles. On the export page "
                            "choose 15-minute candles, not ticks")
        if norms and norms[0] in ("timestamp", "time stamp") and "open" in names:
            raise DataError(f"{name}: this already is a dukascopy-node style file (first column 'timestamp'); "
                            "propkit reads it directly, so it needs no converting. Give only the files the "
                            "export page saved")
        self.t = next((i for i, n in enumerate(norms) if time_column_kind(n)), None)
        if self.t is None:
            raise DataError(f"{name}: no time column in the first line ({shown_header}); a Dukascopy candle export "
                            "starts 'Gmt time' or 'Local time'. Is this the file the export page saved?")
        self.time_name = self.cells[self.t]
        self.time_kind = time_column_kind(norms[self.t])
        idx = {}
        for key in ("open", "high", "low", "close"):
            if key in norms:
                idx[key] = norms.index(key)
        vol = next((i for i, n in enumerate(norms) if i not in (self.t, *idx.values()) and _VOLUME_NAME.match(n)),
                   None)
        missing = [k.capitalize() for k in ("open", "high", "low", "close") if k not in idx]
        if vol is None:
            missing.append("Volume")
        if missing:
            raise DataError(f"{name}: column(s) {', '.join(missing)} not found in the first line ({shown_header}); "
                            "a Dukascopy candle export has Open, High, Low, Close and Volume (the flat rule needs "
                            "the volume)")
        self.o, self.h, self.l, self.c = idx["open"], idx["high"], idx["low"], idx["close"]
        self.v = vol
        used = {self.t, self.o, self.h, self.l, self.c, self.v}
        self.extra = [self.cells[i] for i in range(self.ncols) if i not in used and self.cells[i]]


_DT_RE = re.compile(
    r"(?:(?P<d>\d{2})\.(?P<m>\d{2})\.(?P<y>\d{4})|(?P<Y>\d{4})(?P<sep>[-.])(?P<M>\d{2})(?P=sep)(?P<D>\d{2}))"
    r"(?:T|\s+)(?P<H>\d{2}):(?P<I>\d{2})(?::(?P<S>\d{2})(?:[.,](?P<F>\d{1,9}))?)?"
    r"(?:\s*(?P<tz>Z|(?:GMT|UTC)(?:\s*(?P<zs>[+-])\s*(?P<zh>\d{1,2})(?::?(?P<zm>\d{2}))?)?"
    r"|(?P<os>[+-])(?P<oh>\d{2}):?(?P<om>\d{2})))?\s*\Z",
    re.IGNORECASE | re.ASCII)


def _layout(m: re.Match) -> str:
    if m.group("d"):
        date = "DD.MM.YYYY"
    else:
        s = m.group("sep")
        date = f"YYYY{s}MM{s}DD"
    t = "HH:MM"
    if m.group("S") is not None:
        t += ":SS"
        if m.group("F") is not None:
            t += "." + "f" * len(m.group("F"))
    tz = ""
    if m.group("tz"):
        if m.group("zs") or m.group("os"):
            tz = " GMT+hhmm (offset on each row)"
        else:
            tz = " " + m.group("tz").upper()
    return f"{date} {t}{tz}"


def time_problem(name: str, line: int, text: str, column: str) -> str:
    t = str(text).strip()
    if re.search(r"\d{1,4}/\d{1,2}/\d{1,4}", t):
        return (f"{name} line {line}: the time {shown(t)} uses slashes, which Dukascopy does not write: the file "
                "was probably opened and saved in Excel. Download this part again and do not open it in Excel")
    if re.fullmatch(r"\d{9,13}(\.0+)?", t):
        return (f"{name} line {line}: the time {shown(t)} is a plain number; this is not a Dukascopy export page "
                "file (those write day-first times such as 01.01.2015 00:00:00.000)")
    return (f"{name} line {line}: cannot read the time {shown(t)} in column '{column}'; expected a day-first "
            "time such as 01.01.2015 00:00:00.000 (or one ending in GMT+0800). If the file was opened in Excel, "
            "download it again")


_PLAIN_NUM = re.compile(r"\d+(?:\.\d+)?\Z", re.ASCII)              # ASCII digits only (VD-4)
_EXP_NUM = re.compile(r"\d+(?:\.\d+)?[eE][+-]?\d+\Z", re.ASCII)
_EPOCH_DAY0 = _dt.date(1970, 1, 1).toordinal()


def canon_plain(txt: str) -> str:
    """'0123.4500' -> '123.45', '7.000' -> '7': the same decimal value, printed as JavaScript prints it."""
    ip, _, fp = txt.partition(".")
    ip = ip.lstrip("0") or "0"
    fp = fp.rstrip("0")
    return ip + "." + fp if fp else ip


class Part:
    """One input file, parsed."""

    def __init__(self, path: Path, side: str, index: int):
        self.path, self.side, self.index = path, side, index
        self.name = path.name
        self.rows: list[tuple] = []
        self.n_rows = 0
        self.n_cut = 0
        self.decimal_comma = False
        self.layout = ""
        self.encoding = ""
        self.delimiter = ","
        self.header: Header | None = None
        self.line_ends = "LF"

    def describe(self) -> str:
        h = self.header
        zone = {"utc": "UTC by its name", "local": "local, placed in UTC by the offset on each row",
                "zone": "a named zone, placed in UTC by the offset on each row",
                "unknown": "no zone in its name, placed in UTC by the offset on each row"}[h.time_kind]
        extra = f"; other columns ignored: {', '.join(h.extra)}" if h.extra else ""
        comma = "; decimal comma" if self.decimal_comma else ""
        return (f"time column '{h.time_name}' ({zone}); times {self.layout}; "
                f"{_DELIM_NAMES.get(self.delimiter, repr(self.delimiter))}-separated{comma}; {self.encoding}, "
                f"{self.line_ends}; {self.n_rows} rows{extra}")


def _check_symbol_name(name: str) -> None:
    m = re.match(r"([A-Za-z0-9.]+)_Candlestick", name)
    if m and m.group(1).replace(".", "").upper() != "XAUUSD":
        raise DataError(f"{name}: the name says {m.group(1)}, not XAU/USD; on the export page choose the "
                        "instrument XAU/USD (Gold vs US Dollar)")
    m = re.search(r"_Candlestick_(\d+)_([A-Za-z]+)_", name, re.ASCII)
    if m and not (int(m.group(1)) == 15 and m.group(2).lower() in ("m", "min", "mins", "minute", "minutes")):
        raise DataError(f"{name}: the name says {m.group(1)} {m.group(2)} candles, not 15 minutes; on the export "
                        "page choose 15-minute candles and download this part again")


def read_part(path: Path, side: str, index: int) -> Part:
    part = Part(path, side, index)
    name = part.name
    _check_symbol_name(name)
    text, part.encoding = read_text(path)
    part.line_ends = "CRLF" if "\r\n" in text[:20000] else ("CR" if "\r" in text[:20000] else "LF")
    lines = text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i == len(lines):
        raise DataError(f"{name}: the file is empty; download this part again")
    header_line = lines[i].lstrip("\ufeff")
    part.delimiter = delim = sniff_delimiter(header_line)
    h = part.header = Header(next(csv.reader([header_line], delimiter=delim)), name)
    body = [(n, ln) for n, ln in enumerate(lines[i + 1:], start=i + 2) if ln.strip()]
    if not body:
        raise DataError(f"{name}: the file has a first line but no rows; download this part again")
    part.n_rows = len(body)
    utc_header = h.time_kind == "utc"
    comma_ok = delim != ","
    days: dict[tuple[str, str, str], int] = {}
    rows = part.rows
    first = True

    def price(txt: str, col: str, line: int) -> str:
        t = txt.strip()
        if comma_ok and "," in t:
            if "." in t:
                raise DataError(f"{name} line {line}: the {col} value {t!r} has both a comma and a point")
            t = t.replace(",", ".")
            part.decimal_comma = True
        if not _PLAIN_NUM.match(t):
            if _EXP_NUM.match(t):
                raise DataError(f"{name} line {line}: the {col} value {t!r} is in scientific notation, which "
                                "Dukascopy does not write for prices: the file was probably opened and saved in "
                                "Excel. Download this part again and do not open it in Excel")
            raise DataError(f"{name} line {line}: the {col} value {t!r} is not a plain positive number")
        c = canon_plain(t)
        if c == "0":
            raise DataError(f"{name} line {line}: the {col} is 0; prices must be above 0")
        return c

    def volume(txt: str, line: int) -> str:
        t = txt.strip()
        if comma_ok and "," in t and "." not in t:
            t = t.replace(",", ".")
            part.decimal_comma = True
        if _PLAIN_NUM.match(t):
            return canon_plain(t)
        if _EXP_NUM.match(t):
            if "E+" in t:
                raise DataError(f"{name} line {line}: the Volume value {t!r} is written the way Excel writes "
                                "numbers (E+): the file was probably opened and saved in Excel. Download this "
                                "part again and do not open it in Excel")
            try:
                return canon_plain(format(Decimal(t), "f"))
            except (InvalidOperation, ValueError):
                pass
        raise DataError(f"{name} line {line}: the Volume value {t!r} is not a plain number >= 0")

    reader = csv.reader((ln for _, ln in body), delimiter=delim)
    tcol = h.t

    def cut_short(line: int, nc: int) -> DataError:
        return DataError(f"{name} line {line}: {nc} fields but the first line has {h.ncols}; the file looks cut or "
                         "damaged: download this part again")

    for (line, _raw), cells in zip(body, reader):
        nc = len(cells)
        if nc <= tcol:
            raise cut_short(line, nc)
        ttxt = cells[tcol].strip()
        m = _DT_RE.match(ttxt)
        if m is None:
            raise DataError(time_problem(name, line, ttxt, h.time_name))
        if m.group("d"):
            key = (m.group("y"), m.group("m"), m.group("d"))
        else:
            key = (m.group("Y"), m.group("M"), m.group("D"))
        day = days.get(key)
        if day is None:
            try:
                day = _dt.date(int(key[0]), int(key[1]), int(key[2])).toordinal() - _EPOCH_DAY0
            except ValueError:
                raise DataError(f"{name} line {line}: the date in {shown(ttxt)} does not exist (expected day-first "
                                "DD.MM.YYYY)")
            days[key] = day
        hh, mi = int(m.group("H")), int(m.group("I"))
        ss = int(m.group("S") or 0)
        if hh > 23 or mi > 59 or ss > 59:
            raise DataError(f"{name} line {line}: the time {shown(ttxt)} is not a valid clock time")
        if m.group("zs"):
            off = int(m.group("zh")) * 3600 + int(m.group("zm") or 0) * 60
            off = -off if m.group("zs") == "-" else off
            has_off = True
        elif m.group("os"):
            off = int(m.group("oh")) * 3600 + int(m.group("om")) * 60
            off = -off if m.group("os") == "-" else off
            has_off = True
        elif m.group("tz"):
            off, has_off = 0, True          # a bare GMT, UTC or Z
        else:
            off, has_off = 0, False
        if has_off and abs(off) > 14 * 3600:
            raise DataError(f"{name} line {line}: the UTC offset in {shown(ttxt)} is not a real one")
        if not has_off and not utc_header:
            raise DataError(f"{name} line {line}: the time column is '{h.time_name}' and the time has no GMT "
                            "offset, so it cannot be placed in UTC. On the export page set the time zone to GMT "
                            "(UTC) and download this part again")
        ts = day * 86400 + hh * 3600 + mi * 60 + ss - off
        if first:
            part.layout = _layout(m)
            first = False
        # the holdout lock: cut as soon as the time is read, before any other cell or check of the row (a UTC
        # header with an offset is read both ways, and cut if either reading is at or after the lock)
        if ts >= LOCK_S or (utc_header and ts + off >= LOCK_S):
            part.n_cut += 1
            continue
        if has_off and utc_header and off != 0:
            raise DataError(f"{name} line {line}: the column says '{h.time_name}' (UTC) but the time carries a "
                            f"GMT offset of {off / 3600:+g} h; the file contradicts itself. Download this part "
                            "again with the time zone set to GMT (UTC)")
        if nc != h.ncols and (nc < h.ncols or any(c.strip() for c in cells[h.ncols:])):
            raise cut_short(line, nc)
        frac = m.group("F")
        if ss or (frac and frac.strip("0")):
            raise DataError(f"{name} line {line}: the time {shown(ttxt)} is not on a whole minute; 15-minute bars "
                            "open on whole minutes")
        o = price(cells[h.o], "Open", line)
        hi = price(cells[h.h], "High", line)
        lo = price(cells[h.l], "Low", line)
        cl = price(cells[h.c], "Close", line)
        v = volume(cells[h.v], line)
        rows.append((ts, o, hi, lo, cl, v, v == "0", index, line))
    if not part.layout:
        part.layout = "(no rows)"
    return part


# ---------------------------------------------------------------------------------------------------
# one side

def _bar_words(step: int) -> str:
    return {60: "1 minute", 300: "5 minutes", 600: "10 minutes", 1800: "30 minutes", 3600: "1 hour",
            14400: "4 hours", 86400: "1 day", 604800: "1 week"}.get(step, f"{step} seconds")


def _main_step(times: list[int]) -> int:
    """The most common step between consecutive sorted bar times (the smallest one on a tie)."""
    steps = Counter(b - a for a, b in zip(times, times[1:]))
    top = max(steps.values())
    return min(k for k, c in steps.items() if c == top)


def _hhmm(ts: int) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%H:%M")


def _month(ts: int) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y-%m")


def week_open_warnings(lab: str, opens: list[tuple[int, int]], names: dict[int, str]) -> list[str]:
    """WARNING lines for input parts whose week opens look shifted. opens: (bar time, part index) of each
    first bar after a gap of 36 h or more. Checked per part: (1) off the hour or outside 21:00-23:00 UTC on
    more than a quarter of the part's week opens; (2) for the other parts, a different time of day than the
    one the parts not flagged by (1) open at in the same calendar month (held by at least 2/3 of their week
    opens in that month), on more than half of the part's week opens: one part shifted by whole hours."""
    out = []
    by_part: dict[int, list[int]] = {}
    for ts, pi in opens:
        by_part.setdefault(pi, []).append(ts)
    flagged = set()
    for pi in sorted(by_part):
        mine = by_part[pi]
        bad = [ts for ts in mine if ts % 3600 or not WEEK_OPEN_LO <= ts % 86400 // 3600 <= WEEK_OPEN_HI]
        if len(bad) >= 2 and len(bad) * 4 > len(mine):
            flagged.add(pi)
            seen = Counter(_hhmm(ts) for ts in bad)
            out.append(f"WARNING: {lab}: {names[pi]}: {len(bad)} of its {len(mine)} week opens are off the hour or "
                       f"outside 21:00-23:00 UTC ({', '.join(f'{k} x{n}' for k, n in seen.most_common(3))}): its "
                       "times may be bar END times or not UTC. Check the time zone setting of that download")
    for pi in sorted(set(by_part) - flagged):
        ref: dict[str, Counter] = {}                # calendar month -> the other good parts' week-open times
        for ts, pj in opens:
            if pj != pi and pj not in flagged:
                ref.setdefault(_month(ts)[5:], Counter())[_hhmm(ts)] += 1
        pairs = []
        for ts in by_part[pi]:
            c = ref.get(_month(ts)[5:])
            if c:
                top, n = c.most_common(1)[0]
                if n >= 2 and n * 3 >= sum(c.values()) * 2:     # a clear (2/3) majority, else not compared
                    pairs.append((_hhmm(ts), top))
        differ = [(a, b) for a, b in pairs if a != b]
        if len(pairs) >= 4 and len(differ) * 2 > len(pairs):
            a, b = Counter(differ).most_common(1)[0][0]
            out.append(f"WARNING: {lab}: {names[pi]}: {len(differ)} of {len(pairs)} week opens are at a different "
                       f"time of day than the other files' in the same months (e.g. {a} against {b} UTC): this part "
                       "may be shifted. Check the time zone setting of that download")
    return out


class Side:
    def __init__(self, side: str, parts: list[Part]):
        self.side, self.parts = side, parts
        self.label = side.upper()
        self.kept: list[tuple] = []
        self.times: list[int] = []


def build_side(side: str, parts: list[Part]) -> Side:
    s = Side(side, parts)
    lab = s.label
    names = {p.index: p.name for p in parts}
    n_read = sum(p.n_rows for p in parts)
    n_cut = sum(p.n_cut for p in parts)
    say(f"{lab}: {len(parts)} file(s), {n_read} rows read")
    say(f"{lab}: holdout lock: {n_cut} row(s) opening at or after {LOCK_TEXT} were cut (not read further)")

    # each file on its own must be 15-minute bars (a 1-hour or daily part would hide among 15-minute ones)
    for p in parts:
        pt = sorted({r[R_TS] for r in p.rows})
        if len(pt) >= 2 and _main_step(pt) != BAR_S:
            raise DataError(f"{lab}: {p.name}: the bars in this file are mostly {_bar_words(_main_step(pt))} apart, "
                            "not 15 minutes. On the export page choose 15-minute candles and download this part again")

    # overlapping parts: exact duplicates are dropped, disagreements stop the conversion; two volume-0 rows
    # (flats, dropped below anyway) count as duplicates even when their prices differ
    by_ts: dict[int, tuple] = {}
    dup = 0
    vol_only = 0
    flat_pairs = 0
    conflicts: list[tuple[tuple, tuple]] = []
    for p in parts:
        for r in p.rows:
            prev = by_ts.get(r[R_TS])
            if prev is None:
                by_ts[r[R_TS]] = r
            elif prev[R_O:R_C + 1] == r[R_O:R_C + 1] and prev[R_VZERO] == r[R_VZERO]:
                dup += 1
                if prev[R_V] != r[R_V]:
                    vol_only += 1
            elif prev[R_VZERO] and r[R_VZERO]:
                dup += 1
                flat_pairs += 1
            else:
                conflicts.append((prev, r))
    if conflicts:
        a, b = conflicts[0]
        where_a = f"{names[a[R_PART]]} line {a[R_LINE]}"
        where_b = f"{names[b[R_PART]]} line {b[R_LINE]}"
        raise DataError(f"{lab}: {len(conflicts)} bar time(s) appear twice with different values (first at "
                        f"{utc_text(a[R_TS])}: {where_a} has open/high/low/close/volume "
                        f"{'/'.join(a[R_O:R_V + 1])}, {where_b} has {'/'.join(b[R_O:R_V + 1])}). Overlapping parts "
                        "must agree: download the parts again with the same settings (same side, 15 minutes, GMT)")
    rows = [by_ts[t] for t in sorted(by_ts)]
    msg = f"{lab}: {dup} exact duplicate row(s) dropped (overlapping parts or files)"
    if vol_only:
        msg += f"; {vol_only} of them differ only in a non-zero volume (the first file's volume is kept)"
    if flat_pairs:
        msg += (f"; {flat_pairs} of them are volume-0 rows (flats) whose prices differ between the files: both "
                "copies are flats, dropped below either way")
    say(msg)
    if len(rows) < 2:
        raise DataError(f"{lab}: {len(rows)} bar(s) before the lock; at least 2 are needed")

    # 15-minute bars on UTC 15-minute boundaries
    times = [r[R_TS] for r in rows]
    mode = _main_step(times)
    if mode != BAR_S:
        raise DataError(f"{lab}: the bars are mostly {_bar_words(mode)} apart, not 15 minutes. On the export page "
                        "choose 15-minute candles and download again")
    off = [t for t in times if t % BAR_S]
    if off:
        raise DataError(f"{lab}: {len(off)} bar(s) do not open on a 15-minute boundary in UTC (the first at "
                        f"{utc_text(off[0] - off[0] % 60)} + {off[0] % 60} s); these are not Dukascopy 15-minute "
                        "bars in a whole-hour time zone. Download again with the time zone set to GMT (UTC)")

    # flats: dukascopy-node 1.50.0 ignoreFlats drops volume == 0 exactly (getOHLC filter data[5] !== 0)
    kept = [r for r in rows if not r[R_VZERO]]
    flats = [r for r in rows if r[R_VZERO]]
    flat_equal = sum(1 for r in flats if r[R_O] == r[R_H] == r[R_L] == r[R_C])
    flat_moving = len(flats) - flat_equal
    say(f"{lab}: flats dropped (volume exactly 0, as dukascopy-node's default ignoreFlats): {len(flats)} "
        f"(open=high=low=close: {flat_equal}; prices moving: {flat_moving})")
    if flat_moving:
        first = next(r for r in flats if not (r[R_O] == r[R_H] == r[R_L] == r[R_C]))
        say(f"WARNING: {lab}: {flat_moving} bar(s) with volume 0 have moving prices (first {utc_text(first[R_TS])}); "
            "they were dropped as dukascopy-node drops them, but if the page rounds small volumes to 0 they were "
            "real bars: download again with the finest volume unit (units) if the page offers one")
    if len(kept) < 2:
        raise DataError(f"{lab}: only {len(kept)} bar(s) with volume above 0; the files hold no usable bars "
                        "(every bar has volume 0)")

    # XAU/USD-like prices
    vals = [float(x) for r in kept for x in r[R_O:R_C + 1]]
    lo, hi = min(vals), max(vals)
    med = statistics.median(float(r[R_C]) for r in kept)
    if lo < PRICE_MIN or hi > PRICE_MAX or not MEDIAN_MIN <= med <= MEDIAN_MAX:
        raise DataError(f"{lab}: prices from {lo:g} to {hi:g} (median close {med:g}) do not look like XAU/USD in "
                        "USD per ounce (about 1,050 to 3,800 in 2015-2025). On the export page choose the "
                        "instrument XAU/USD")

    # what propkit's validate_bars refuses (float64, as propkit reads the file)
    bad_h = [r for r in kept if float(r[R_H]) < max(float(r[R_O]), float(r[R_C]))]
    bad_l = [r for r in kept if float(r[R_L]) > min(float(r[R_O]), float(r[R_C]))]
    for bad, what in ((bad_h, "high below open or close"), (bad_l, "low above open or close")):
        if bad:
            raise DataError(f"{lab}: {len(bad)} bar(s) have {what} (the first at {utc_text(bad[0][R_TS])}, "
                            f"{names[bad[0][R_PART]]} line {bad[0][R_LINE]}); propkit refuses such bars. Download "
                            "that part again")

    s.kept = kept
    s.times = [r[R_TS] for r in kept]
    t = s.times
    early = sum(1 for x in t if x < RANGE_START_S)
    say(f"{lab}: kept {len(t)} bars, {utc_text(t[0])} to {utc_text(t[-1])}; opening before 2015-01-01 00:00 UTC: "
        f"{early} (kept: propkit cuts and counts them itself)")
    years = Counter(_dt.datetime.fromtimestamp(x, _dt.timezone.utc).year for x in t)
    say(f"{lab}: bars per year: " + ", ".join(f"{y} {years[y]}" for y in sorted(years)))
    gaps = [(a, b) for a, b in zip(t, t[1:]) if b - a > GAP_REPORT_S]

    def listed(gs):
        return "; ".join(f"{utc_text(a)} to {utc_text(b)}" for a, b in gs[:5]) + (
            f" (and {len(gs) - 5} more)" if len(gs) > 5 else "")
    say(f"{lab}: gaps longer than 3 days (holiday weekends are about 3 days): {listed(gaps) if gaps else 'none'}")
    long_gaps = [(a, b) for a, b in gaps if b - a > GAP_WARN_S]
    if long_gaps:
        say(f"WARNING: {lab}: {len(long_gaps)} gap(s) longer than 4 days (a missing part?): {listed(long_gaps)}")
    week_opens = [(b, r[R_PART]) for (a, b), r in zip(zip(t, t[1:]), kept[1:]) if b - a >= WEEK_GAP_S]
    opens = Counter(_hhmm(b) for b, _ in week_opens)
    if opens:
        say(f"{lab}: first bar after each weekend or holiday (UTC time, count): "
            + ", ".join(f"{k} x{n}" for k, n in opens.most_common(6)))
    for line in week_open_warnings(lab, week_opens, names):
        say(line)
    nz = [Decimal(r[R_V]) for r in kept]
    equal_kept = sum(1 for r in kept if r[R_O] == r[R_H] == r[R_L] == r[R_C])
    say(f"{lab}: volume (the page's unit) smallest above 0: {min(nz)}, largest: {max(nz)}; bars with "
        f"open=high=low=close and volume above 0 (kept, as dukascopy-node keeps them): {equal_kept}")
    if t[0] > WANT_FIRST_BY_S:
        say(f"WARNING: {lab}: the data starts {utc_text(t[0])}; the test needs bars from 2015-01-02 (the first "
            "trading day of 2015). Check that the 2015 part was downloaded, or whether the page offers 15-minute "
            "bars that far back")
    if t[-1] < WANT_LAST_FROM_S:
        say(f"WARNING: {lab}: the data ends {utc_text(t[-1])}; the test needs bars up to 2025-09-26 (the last "
            "trading day before the lock). Check that the last part was downloaded")
    return s


# ---------------------------------------------------------------------------------------------------
# both sides

def pair_check(bid: Side, ask: Side) -> None:
    bmap = {r[R_TS]: r for r in bid.kept}
    amap = {r[R_TS]: r for r in ask.kept}
    common = [t for t in bid.times if t in amap]
    if common:
        same = sum(1 for t in common if bmap[t][R_O:R_C + 1] == amap[t][R_O:R_C + 1])
        if same == len(common):
            raise DataError(f"PAIR: the bid and ask files hold identical prices on all {len(common)} common bars: the "
                            "same side was downloaded twice. Download the other side (BID or ASK) again")
        below = sum(1 for t in common if float(amap[t][R_C]) < float(bmap[t][R_C]))
        if below > SWAP_SHARE * len(common):
            raise DataError(f"PAIR: the ask close is below the bid close on {below} of {len(common)} common bars: "
                            "the BID and ASK files look swapped (their names do not match their contents). Check "
                            "which side each download was")
        # the same two checks per UTC calendar month: one part of the wrong side hides among the right ones
        months: dict[str, list[int]] = {}
        for t in common:
            months.setdefault(_month(t), []).append(t)
        for mon, ts in sorted(months.items()):
            if len(ts) < MONTH_MIN_BARS:
                continue
            same = sum(1 for t in ts if bmap[t][R_O:R_C + 1] == amap[t][R_O:R_C + 1])
            below = sum(1 for t in ts if float(amap[t][R_C]) < float(bmap[t][R_C]))
            if same * 2 > len(ts) or below * 2 > len(ts):
                files = (f"{bid.parts[Counter(bmap[t][R_PART] for t in ts).most_common(1)[0][0]].name} (BID) and "
                         f"{ask.parts[Counter(amap[t][R_PART] for t in ts).most_common(1)[0][0]].name} (ASK)")
                what = (f"bid and ask prices are identical on {same} of {len(ts)} bars: one of that month's files is "
                        "the other side (e.g. an ASK file that holds BID prices)" if same * 2 > len(ts) else
                        f"ask close is below the bid close on {below} of {len(ts)} bars: that month's BID and ASK "
                        "files look swapped")
                raise DataError(f"PAIR: in {mon} the {what}. Files: {files}. Download that part again for both "
                                "sides and check the side (BID or ASK) on the page before each download")
    only_b = [t for t in bid.times if t not in amap]
    only_a = [t for t in ask.times if t not in bmap]
    if only_b or only_a:
        def first(xs):
            return ", ".join(utc_text(x) for x in xs[:5]) + (" ..." if len(xs) > 5 else "") if xs else "none"
        raise DataError(f"PAIR: bid and ask do NOT hold the same bar times (bid {len(bid.times)} bars, ask "
                        f"{len(ask.times)}; {len(only_b)} only in bid: {first(only_b)}; {len(only_a)} only in ask: "
                        f"{first(only_a)}). Nothing was written; nothing is filled or dropped to make them match. "
                        "Report this; do not edit or merge the files")
    say(f"PAIR: bid and ask hold exactly the same {len(common)} bar times")
    neg = [t for t in common if float(amap[t][R_O]) < float(bmap[t][R_O])]
    if neg:
        raise DataError(f"PAIR: {len(neg)} bar(s) have the ask open below the bid open (the first at "
                        f"{utc_text(neg[0])}); propkit refuses any such bar. Report this; do not edit the files")
    counts = {k: sum(1 for t in common if float(amap[t][i]) < float(bmap[t][i]))
              for k, i in (("high", R_H), ("low", R_L), ("close", R_C))}
    spreads = sorted(float(amap[t][R_O]) - float(bmap[t][R_O]) for t in common)   # as propkit: float64
    pos = 0.9 * (len(spreads) - 1)                                                  # numpy's linear quantile
    k = int(pos)
    p90 = spreads[k] + (spreads[min(k + 1, len(spreads) - 1)] - spreads[k]) * (pos - k)
    say(f"PAIR: ask below bid: open 0, high {counts['high']}, low {counts['low']}, close {counts['close']} "
        f"(counted, as propkit counts them); spread at the open median {statistics.median(spreads):.3f}, p90 "
        f"{p90:.3f} USD/oz")


# ---------------------------------------------------------------------------------------------------
# writing

def render(rows: list[tuple]) -> bytes:
    out = [OUT_HEADER]
    out.extend(f"{r[R_TS] * 1000},{r[R_O]},{r[R_H]},{r[R_L]},{r[R_C]},{r[R_V]}" for r in rows)
    return ("\n".join(out) + "\n").encode("ascii")


def write_outputs(out: Path, sides: dict[str, Side]) -> None:
    datas = {side: render(sides[side].kept) for side in SIDES if side in sides}    # both sides before writing
    try:
        out.mkdir(exist_ok=True)          # never its parents: a mistyped --out creates nothing above it
        busy = any(out.iterdir())
    except OSError as e:
        raise UsageError(f"cannot create --out {out}: {e}")
    if busy:
        raise UsageError(f"--out {out} is no longer empty; nothing was written")
    done: list[Path] = []
    temps: list[Path] = []
    try:
        for side in SIDES:
            if side not in sides:
                continue
            data = datas[side]
            target = out / OUT_NAMES[side]
            tmp = out / (OUT_NAMES[side] + ".writing")
            if target.exists():
                raise UsageError(f"{target} already exists; nothing is overwritten")
            temps.append(tmp)
            with open(tmp, "xb") as f:
                f.write(data)
            os.rename(tmp, target)
            temps.remove(tmp)
            done.append(target)
            say(f"WROTE {target} ({len(sides[side].kept)} bars, {len(data)} bytes, sha256 "
                f"{hashlib.sha256(data).hexdigest()})")
    except BaseException as e:            # any failure (or Ctrl+C) removes what this run wrote
        for p in temps + done:
            try:
                p.unlink()
            except OSError:
                pass
        if isinstance(e, OSError):
            raise UsageError(f"cannot write in {out}: {e}; nothing was kept")
        raise


# ---------------------------------------------------------------------------------------------------
# command line

class _Parser(argparse.ArgumentParser):
    def error(self, message):  # one line, exit 2
        sys.stderr.write(f"ERROR: {message} (see --help)\n")
        raise SystemExit(EXIT_USAGE)


def parse_args(argv):
    p = _Parser(prog="dukascopy_web_to_node.py",
                description="Convert Dukascopy website XAU/USD 15-minute candle CSVs (BID and ASK, one or several "
                            "files each) into the dukascopy-node 1.50.0 CSV layout propkit reads.")
    p.add_argument("inputs", nargs="*", metavar="INPUT",
                   help="a folder with the downloaded .csv files, or the files; the side comes from BID or ASK in "
                        "each file name")
    p.add_argument("--bid", nargs="+", metavar="FILE", help="BID files (or a folder) whatever their names")
    p.add_argument("--ask", nargs="+", metavar="FILE", help="ASK files (or a folder) whatever their names")
    p.add_argument("--out", metavar="FOLDER", help="a new (or empty) folder for xauusd_m15_bid.csv and "
                                                   "xauusd_m15_ask.csv; not the input folder")
    p.add_argument("--check-only", action="store_true", help="check and report, write nothing")
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return p.parse_args(argv)


def run(argv) -> int:
    args = parse_args(argv)
    say(f"dukascopy_web_to_node {VERSION}: Dukascopy website candles -> dukascopy-node 1.50.0 layout "
        f"({OUT_HEADER}); flats rule: volume exactly 0 is dropped; holdout lock {LOCK_TEXT}")
    inputs, folders = collect_inputs(args)
    out = None if args.check_only else check_out_folder(args.out, folders)
    sides: dict[str, Side] = {}
    for side in SIDES:
        if not inputs[side]:
            continue
        parts = []
        for i, path in enumerate(inputs[side]):
            part = read_part(path, side, i)
            say(f"  {side.upper()} file {i + 1} of {len(inputs[side])}: {part.name}: {part.describe()}")
            parts.append(part)
        sides[side] = build_side(side, parts)
    if len(sides) == 2:
        pair_check(sides["bid"], sides["ask"])
    else:
        only = next(iter(sides))
        say(f"NOTE: only {only.upper()} files were given; the bid/ask checks need both sides")
    if args.check_only:
        say("CHECK ONLY: all checks passed; nothing was written")
        return EXIT_OK
    write_outputs(out, sides)
    say("DONE: exit code 0")
    return EXIT_OK


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    try:
        return run(sys.argv[1:] if argv is None else argv)
    except UsageError as e:
        sys.stdout.flush()
        sys.stderr.write(f"ERROR: {e}\n")
        return EXIT_USAGE
    except DataError as e:
        sys.stdout.flush()
        sys.stderr.write(f"ERROR: {e}\n")
        sys.stderr.write("Nothing was written (exit code 3).\n")
        return EXIT_DATA


if __name__ == "__main__":
    sys.exit(main())
