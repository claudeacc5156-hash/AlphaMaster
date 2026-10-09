"""propkit/zeno_v1.py - zeno_pullback_v1, zeno's frozen pullback rule, as code.

RESEARCH ONLY - not trading advice. Nothing here places, prepares or simulates sending orders, nothing
talks to a broker or the network, and nothing reads price data by itself: the caller passes the files.

Authority: propkit/specs/zeno_pullback_v1.md (records sha256 d36ad25f...bbcb; the readable copy folds the
star character to '(*)', the byte-exact copy is zeno_pullback_v1.md.utf8) and its JSON copy. The
D-numbers below are the spec's defaults D1-D24. Where the spec leaves a detail open, the literal reading
chosen here is marked [SI-n] at the place it is used.

Pipeline (one data set, one cost cell):

    frame = load_m15_bidask(bid_path, ask_path)      # D1: bid + ask M15 in one frame; the lock is enforced
    news = read_news_csv(calendar_csv)                # D20: NFP, CPI, PPI, FOMC instants (UTC)
    prep = prepare(frame, news)                       # cost-independent: H1 bars, EMA30, ATR14, server
                                                      # days, filters, the two setup state machines
    res = simulate(prep, ZenoConfig(ZenoCell("evaluation", 10.0, "S1", 1.5)))     # D11-D23 for one cell
    res.legs, res.positions, res.decisions            # the three output tables
    equity, trades = cell_equity(prep, res)           # propkit EQUITY/TRADES for the prop evaluator (shorts
                                                      # marked on the cell's ask high/close [SI-67])
    dec = screen(prep, ZenoConfig(cell))              # stage 1 (G0): every check that needs no earlier
                                                      # trade; status "eligible" or a reason [SI-54]
    chart = chart_prices(prep)                        # the entry and stop a chart of the data shows [SI-66]

Conventions
  * time: int64 UTC epoch seconds; a bar's `time` is its OPEN; the bar closes at time + 900 (M15).
  * prices: USD per troy ounce. The signal (rules 2-4) uses BID bars only (rule 1) for both sides [SI-15].
    A long fills at the ask and exits on the bid; a short fills at the bid and exits on the ask (D11,
    D14): a short's stop and targets are ASK levels, checked against the ask bars' high and low.
  * the ask side of one cost cell (AskSide): S1 = the data's ask, S2 = bid + 0.18 (0.20 from 21:00 to
    24:00 UTC = 05:00-08:00 SGT); a cost multiplier k scales the spread: ask = bid + k x (ask - bid).
  * size: oz (units); 1 lot = 100 oz; lot step 0.01 lot = 1 oz. Money: USD.
  * the broker server day (D21) runs 17:00 New York to 17:00 New York (server clock = New York + 7 h);
    a "trading day" (D1 warm-up, D18 median) is a server day with at least one bar [SI-8].

Per-bar order inside simulate (one position at most, D21):
  at the OPEN of bar j (positions entered before j): gap through the stop (fill at the open minus/plus
  the stop slippage), gap through a target (fill at the target level, never better), the 16:30 New York
  time exit (fill at the open), the Master news close (variants "master" and "master_fp"); then the entry
  accepted at the previous close fills at this open (and the Master close takes it at that same open when,
  after a data gap, this bar holds T - 10 min [SI-70]). INSIDE bar j: stop first when one bar touches the
  stop and a target (D15); after the +2R partial the breakeven stop is checked in the same bar (D15) and
  wins over +4R.
  At the CLOSE of bar j: the setup machines move and may trigger; every trigger is decided at once
  (entered for the next open, or blocked with every failing reason; one shot either way, D10).
An intrabar exit is written to TRADES at time + 899 (so propkit.equity books it in its own bar) and
stamped at the bar's close (time + 900) for the cooldown (D22).

Addendum A (propkit/specs/zeno_pullback_v1_addendum_A.md, pre-registered 2026-10-08; it changes only how the
prop firm is simulated, never the 12 rules, D1-D24 or the gates):
  * A1, variant "master_fp": everything as "master" plus FundingPips' restricted list (read_restricted_csv):
    no entry when the trigger close OR the entry fill lies in [T - 5 min, T_end + 5 min] (T_end = T for a
    release, T + 180 min for a Fed Chair testimony, T + 60 min for any other Fed Chair appearance) or on the
    New York date of an event with no known time (reason "fp_restricted_window", right after D20's); the D23
    close (10 min before T, positions opened under 5 h before it, [SI-40], [SI-70]) for every restricted event
    with a known time, on top of rule 9's four events.
  * A2, both Master variants: the tiered metals margin (margin_usd) at the entry fill price; D13 lots whose
    margin exceeds the closed balance are cut to the largest 0.01-lot size that fits (max_units_within_margin),
    and an entry that cannot hold 0.01 lot is blocked ("margin_cap_below_lot_step", before the size reason).
    The evaluation variant is not capped; meta["margin"] counts its entries over the margin at a flat 1:10 and
    a flat 1:30.

Test-only parameter: prepare(..., test_indicators={"atr14": per-M15-bar array, "ema30_h1": per-H1-bar
array}) replaces the computed indicators so hand-computed cases are possible. Never use it for a real
run: the result's meta records "indicators_injected".
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from propkit import adapters
from propkit import bars as bars_mod
from propkit import calendar
from propkit import indicators as ind
from propkit.costs import CostModel
from propkit.pullback import floor_to_lot_step

# ---------------------------------------------------------------------------------------
# the spec's numbers (zeno_pullback_v1.md; the JSON copy agrees)

SPEC_ID = "zeno_pullback_v1"
SPEC_VERSION = "1.0"
SPEC_SHA256_MD = "d36ad25f74c293166bd82f117cb96a6e6890a2ab2c3dd63dbc41bff52b67bbcb"
SPEC_SHA256_JSON = "ebd8017a271229786c5a79f08baddd0e6b4940be52f8e28d40706dddb1f7b31a"
SPECS_DIR = Path(__file__).resolve().parent / "specs"
ADDENDUM_A_FILE = "zeno_pullback_v1_addendum_A.md"   # addendum A (FundingPips firm rules), a byte-exact copy
ADDENDUM_A_SHA256 = "0f64bf584e325cc665e1afee0f9dccb9e53f83645f078abad2d455f5982af7b0"
RESTRICTED_CSV = (Path(__file__).resolve().parent / "data" / "news_calendar"
                  / "us_restricted_events_fundingpips_2015-01-01_2025-09-27.csv")    # addendum A1, packaged
RESTRICTED_SHA256 = "6685ee94fa4d0b2c860e1d3fd3a780dc49ad65e3885a2de9f70fc4c218e53cba"

M15_SECONDS = 900
M1_SECONDS = 60
H1_SECONDS = 3600
LOCK_UTC = 1759017600                    # 2025-09-28 00:00:00 UTC, the holdout lock (D1)
LOCK_TEXT = "2025-09-28 00:00:00 UTC"
RANGE_START_UTC = 1420070400             # 2015-01-01 00:00:00 UTC, D1's range start (JSON data.start_utc)
RANGE_START_TEXT = "2015-01-01 00:00:00 UTC"
FIRST_SERVER_DAY_2015 = 16437            # 2015-01-02 (Friday), the first server day of 2015 with gold trading
CONTRACT_OZ = 100.0                      # oz per lot
LOT_STEP_OZ = 1.0                        # 0.01 lot

EMA_PERIOD = 30                          # D2, rule 2
EMA_SLOPE_BARS = 5                       # D3
ATR_PERIOD = 14                          # D2
H_LOOKBACK = 20                          # D5
L_LOOKBACK = 20                          # D5
MIN_LEG_ATR = 1.5                        # rule 3, D7
RETRACE_VALID = 0.5                      # D6
RETRACE_VOID = 0.786                     # D6
TRIGGER_MAX_BARS = 8                     # D9
LEVEL_TOL = 1e-9                         # [SI-34] tie rule, USD/oz: a price within 1e-9 of a COMPUTED level
                                         # (50%, 78.6%, stop, breakeven, targets) is AT that level, so binary
                                         # float noise (2006.30 + 0.10 = 2006.3999999999999) cannot decide an
                                         # exact touch; real prices have 2-3 decimals, far above 1e-9
STOP_BUFFER_ATR = 0.25                   # rule 5
TP1_R = 2.0                              # rule 6
TP2_R = 4.0
RISK_PCT = {"evaluation": 0.005, "master": 0.004, "master_fp": 0.004}     # rule 7, D23, addendum A1
VOL_CAP_X = 2.0                          # rule 8, D18
VOL_MEDIAN_DAYS = 20
MAX_STOP_ATR = 3.0
MAX_SPREAD_FRAC_OF_R = 0.10              # rule 10
WARMUP_TRADING_DAYS = 30                 # D1
SESSION_WINDOWS_UTC = ((7 * 3600, 10 * 3600), (12 * 3600 + 1800, 16 * 3600))   # D19 = 15-18, 20:30-24 SGT
NEWS_EVENTS = ("NFP", "CPI", "PPI", "FOMC")
NEWS_BEFORE_S = 1800                     # D20: [T - 30 min, T + 60 min], both ends blocked
NEWS_AFTER_S = 3600
MASTER_CLOSE_BEFORE_S = 600              # D23
MASTER_MAX_AGE_S = 5 * 3600
FP_BEFORE_S = 300                        # addendum A1: no entry from T - 5 min ...
FP_AFTER_S = 300                         # ... to T_end + 5 min, both ends blocked
FEDCHAIR_EVENT = "FEDCHAIR"
FEDCHAIR_TESTIMONY_S = 180 * 60          # A1 [ASSUMPTION]: a testimony lasts 180 min ("testimony" in the note)
FEDCHAIR_OTHER_S = 60 * 60               # every other Fed Chair appearance 60 min
# A2: FundingPips Master dynamic metals leverage, per position: (oz inside the tier, leverage); 100 oz = 1 lot
MARGIN_TIERS = ((5.0, 50.0), (5.0, 30.0), (5.0, 25.0), (10.0, 20.0), (25.0, 10.0), (math.inf, 5.0))
MARGIN_TOL_USD = 1e-6                    # a margin "fits" when it is <= the closed balance + 1e-6 USD
EVAL_FLAT_LEVERAGES = (10.0, 30.0)       # A2: the evaluation run is not capped; counted at 1:10 and 1:30
MAX_ENTRIES_PER_DAY = 2                  # rule 11, D21
MAX_LOSSES_PER_DAY = 2
DAY_LOSS_FRAC = 0.01
COOLDOWN_S = 900                         # D22
SERVER_HOURS_AHEAD_OF_NY = 7             # "Server clock: New York + 7 h"
TIME_EXIT_NY_SECONDS = 16 * 3600 + 1800  # D17: 16:30 New York
S2_USD = 0.18                            # Costs: spread base S2
S2_ROLLOVER_USD = 0.20
S2_ROLLOVER_UTC = (21 * 3600, 24 * 3600)  # 05:00-08:00 SGT
STOP_SLIPPAGE_USD = 0.05                 # Costs [ASSUMPTION], on stop fills only [SI-21]
VARIANTS = ("evaluation", "master", "master_fp")     # D23 and addendum A1
MASTER_VARIANTS = ("master", "master_fp")             # the D23 close and the A2 margin cap apply
COMMISSIONS = (5.0, 10.0)
SPREAD_BASES = ("S1", "S2")
COST_MULTS = (1.0, 1.5, 2.0)
M1_NOT_RUN = "M1 resolution not run"

SIDE_NAMES = {1: "long", -1: "short"}
SETUP_EVENTS = ("armed", "first_close_before_arming", "cancelled_new_extreme", "voided", "expired", "trigger")
OUTCOMES = ("-1R", "+1R(BE)", "+3R", "time-exit", "other")
LEG_NAMES = ("tp1", "runner", "full")

# Every reason a trigger can be blocked, in the order they are reported (the first one is the decision's
# status). The order is D10's own list (session, news, spread, volatility, daily limits, cooldown, an open
# position, the trend) after the two structural checks; the checks D10 does not name (no valid trade at
# the next open, or too small a size) come last.
BLOCK_REASONS: dict[str, str] = {
    "warmup": "the entry falls in the first 30 trading days (D1 warm-up)",
    "no_next_bar": "the trigger is the last bar of the data: there is no next open to fill at",
    "outside_session": "the entry time (the trigger bar's close) is outside 15:00-18:00 and 20:30-24:00 SGT "
                       "(rule 9, D19)",
    "news_blackout": "the entry time is from 30 min before to 60 min after NFP, CPI, PPI or FOMC (rule 9, D20)",
    "fp_restricted_window": "variant master_fp: the trigger close or the entry fill is from 5 min before a "
                            "FundingPips restricted event to 5 min after its end, or on the New York date of one "
                            "whose time is unknown (addendum A1)",
    "spread_gt_10pct_of_stop": "the entry bar's spread is above 10% of the stop distance R (rule 10, D11)",
    "atr_above_2x_median": "ATR14 at the trigger close is above 2 x its median over the previous 20 trading "
                           "days, or that median is undefined (rule 8, D18)",
    "stop_wider_than_3_atr": "the stop distance R is above 3 x ATR14 at the trigger close (rule 8, D18)",
    "max_entries_per_day": "2 entries were already made this server day (rule 11, D21)",
    "two_losses_today": "2 losing positions already closed this server day (rule 11, D21)",
    "day_loss_1pct": "the server day's realised net P&L is at or below -1.0% of its start balance (rule 11, "
                     "D21)",
    "cooldown_15min": "less than 15 minutes after the last same-direction exit stamp (rule 11, D22)",
    "position_open": "a position is open, or an entry was already accepted at this close (rule 11, D21)",
    "trend_disagrees": "the 1h trend does not agree with this side at the trigger close (rule 2, D4)",
    "entry_after_time_exit": "the next bar opens at or after 16:30 New York of the trigger close's server day "
                             "(only after a data gap; D17, D21)",
    "entry_beyond_stop": "the entry fill is at or beyond the stop, so R <= 0: no valid trade",
    "margin_cap_below_lot_step": "Master variants: the closed balance does not cover the margin of 0.01 lot at "
                                 "the entry price (FundingPips' tiered metals leverage, addendum A2)",
    "size_below_lot_step": "the risk budget buys less than 0.01 lot (rule 7, D13)",
}
# The checks that need how and when earlier trades ended (their P&L, exit times, the entries they let
# through). screen() - stage 1, the G0 signal check - leaves them unevaluated [SI-54].
STATE_REASONS = ("max_entries_per_day", "two_losses_today", "day_loss_1pct", "cooldown_15min", "position_open")
ELIGIBLE = "eligible"                    # screen(): every check that does not need earlier trades passes

FRAME_PRICE_COLUMNS = ("bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high", "ask_low",
                       "ask_close")
FRAME_COLUMNS = ("time",) + FRAME_PRICE_COLUMNS + ("spread_open",)
LEG_COLUMNS = adapters.TRADES_COLUMNS + ("position_id", "leg")
POSITION_COLUMNS = (
    "position_id", "side", "variant", "commission_rt_per_lot", "spread_base", "cost_mult", "setup_id",
    "server_day", "trigger_bar", "trigger_time", "trigger_time_utc", "entry_bar", "entry_time", "entry_time_utc",
    "entry_time_sgt", "entry_price", "spread_entry", "stop_level", "be_level", "tp1_level", "tp2_level",
    "R_usd_per_oz", "lots", "units_oz", "partial_lots", "runner_lots", "balance_at_entry", "risk_budget_usd",
    "atr_trigger", "h_level", "l_level", "leg_usd", "extreme_bar_time", "pullback_bar_time", "tp1_reached",
    "tp2_reached", "exit1_time", "exit1_time_utc", "exit1_price", "exit1_reason", "exit2_time", "exit2_time_utc",
    "exit2_price", "exit2_reason", "final_exit_stamp", "gross_usd", "commission_usd", "slippage_usd", "swap_usd",
    "net_pnl_usd", "r_multiple_net", "outcome", "ambiguous_bar", "m1_bars_resolved", "m1_bars_unresolved",
    "time_exit_rule", "lots_uncapped", "margin_capped")      # the last two: addendum A2 (lots = the lots held)
TIME_EXIT_RULES = ("16:30_open", "early_close_us_holiday", "early_close_other_day")   # [SI-64]
DECISION_COLUMNS = (
    "time", "time_utc", "time_sgt", "event", "side", "setup_id", "bar_index", "bar_time", "status", "reasons",
    "position_id", "h_level", "l_level", "leg_usd", "atr_arm", "retrace_level", "void_level", "extreme_bar_time",
    "arm_bar_time", "pullback_bar_time", "pullback_level", "trigger_level", "bars_since_pullback",
    "entry_time", "entry_price", "stop_level", "spread_entry", "atr_trigger", "atr_median", "trend_ok",
    "news_pre_unscheduled")
# variant master_fp only: its restricted-window block comes only from the 5 min before unscheduled rows [SI-63]
DECISION_COLUMNS_FP = DECISION_COLUMNS + ("fp_pre_unscheduled",)


# ---------------------------------------------------------------------------------------
# small helpers

def sgt_str(ts):
    """'YYYY-MM-DD HH:MM:SS SGT' text (Singapore, UTC+8, no DST) of UTC epoch seconds; scalar or array."""
    arr = np.asarray(ts, dtype=np.int64)
    raw = (arr + calendar.SGT_OFFSET_HOURS * 3600).astype("datetime64[s]").astype(str)
    if arr.ndim == 0:
        return str(raw).replace("T", " ") + " SGT"
    return np.array([x.replace("T", " ") + " SGT" for x in raw.ravel()], dtype=object).reshape(arr.shape)


def _utc_text(ts):
    arr = np.asarray(ts, dtype=np.int64)
    if arr.ndim == 0:
        return calendar.utc_str(int(arr))
    return calendar.utc_str(arr) if arr.size else np.array([], dtype=object)


def _num(value, what: str, lo: float | None = None, lo_open: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{what} must be a number, got {value!r}")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    if lo is not None and (x < lo or (lo_open and x <= lo)):
        raise ValueError(f"{what} must be {'>' if lo_open else '>='} {lo:g}, got {value!r}")
    return x


# ---------------------------------------------------------------------------------------
# data (D1): bid + ask M15 bars, the holdout lock

class HoldoutLockError(ValueError):
    """A bar at or after the holdout lock (2025-09-28 00:00 UTC) was given; there is no override."""


def check_before_lock(times, what: str = "bars") -> None:
    """Raise HoldoutLockError if any time is at or after 2025-09-28 00:00 UTC (D1's holdout lock).

    times: UTC epoch seconds (bar opens). There is no flag, environment variable or argument that skips
    this check: the holdout is a separate pre-registered step."""
    t = np.asarray(times, dtype=np.int64)
    bad = t >= LOCK_UTC
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise HoldoutLockError(
            f"refusing the {what}: {int(bad.sum())} bar(s) open at or after {LOCK_TEXT} (the first is row {i}, "
            f"{calendar.utc_str(int(t[i]))}). That instant is the locked holdout boundary of zeno_pullback_v1 "
            "(spec D1). Cut the file so its last bar opens before it; there is no override.")


def _read_table(path, what: str) -> tuple[pd.DataFrame, str]:
    """Read a Parquet or CSV table after the locked-path check; column names lower-cased and stripped."""
    bars_mod.check_not_locked(path, what=what)
    p = Path(str(path)).expanduser()
    if not p.is_file():
        raise ValueError(f"{what} not found: {p}")
    suffix = p.suffix.lower()
    if suffix not in bars_mod.PARQUET_SUFFIXES + bars_mod.CSV_SUFFIXES:
        raise ValueError(f"{what} {p.name}: use a .parquet or .csv file (got '{suffix or 'no suffix'}')")
    try:
        df = pd.read_parquet(p) if suffix in bars_mod.PARQUET_SUFFIXES else pd.read_csv(p)
    except Exception as e:   # pyarrow and the CSV parser raise several types
        raise ValueError(f"cannot read {what} {p.name}: {type(e).__name__}: {e}")
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"cannot read {what} {p.name} as a table")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df, p.name


def _ohlc(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """One side's OHLC (propkit BARS layout, no spread): the lock is checked on the raw times first, then
    every propkit.bars.validate_bars check runs. Accepts `time` (epoch s/ms/us/ns, datetime64, ISO with an
    offset) or the dukascopy-node `timestamp` column (epoch ms); other columns are ignored."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{source}: expected a pandas DataFrame with time, open, high, low, close")
    d = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    if "time" not in d.columns and "timestamp" in d.columns:
        d = d.rename(columns={"timestamp": "time"})
    missing = [c for c in bars_mod.BARS_COLUMNS if c not in d.columns]
    if missing:
        raise ValueError(f"{source} is missing column(s) {missing}; expected time (or the dukascopy-node "
                         "timestamp in ms), open, high, low, close")
    times = bars_mod.to_epoch_seconds(d["time"], f"{source} time")
    check_before_lock(times, source)
    core = pd.DataFrame({"time": times})
    for col in bars_mod.PRICE_COLUMNS:
        core[col] = d[col].to_numpy()
    return bars_mod.validate_bars(core, source=source)


def bidask_frame(bid: pd.DataFrame, ask: pd.DataFrame, bar_seconds: int = M15_SECONDS,
                 source: str = "M15") -> pd.DataFrame:
    """Join validated bid and ask bars into the one zeno frame (D1) and return it.

    bid, ask: tables with time, open, high, low, close (see _ohlc for the accepted time forms). Both are
    validated like propkit BARS (sorted, unique, finite, > 0, high >= max(open, close), low <= min(open,
    close)); the holdout lock is enforced on each (HoldoutLockError); the bar size must be bar_seconds
    (900 for M15, 60 for M1) and every bar must open on a multiple of it; the two time columns must be
    identical (no silent join, [SI-4]); ask_open >= bid_open on every bar. ask < bid at the high, low or
    close is counted (frame.attrs["zeno_v1"]["ask_below_bid"]), not refused [SI-3]. D1's range starts
    2015-01-01 00:00 UTC: M15 bars opening before it are cut and counted (attrs "cut_before_range_start",
    "range_note"; nothing left is refused) [SI-62]; M1 frames are not cut.
    Columns: time (int64 bar open, UTC epoch s), bid_open/high/low/close, ask_open/high/low/close (USD/oz),
    spread_open = ask_open - bid_open (USD/oz).
    """
    b = _ohlc(bid, f"{source} bid")
    a = _ohlc(ask, f"{source} ask")
    tb = b["time"].to_numpy(dtype=np.int64)
    ta = a["time"].to_numpy(dtype=np.int64)
    cut = 0
    if int(bar_seconds) == M15_SECONDS and tb.size == ta.size and np.array_equal(tb, ta) \
            and (tb < RANGE_START_UTC).any():                       # D1: the range starts 2015-01-01 [SI-62]
        keep = tb >= RANGE_START_UTC
        if not keep.any():
            raise ValueError(f"{source}: every bar opens before {RANGE_START_TEXT}, the start of zeno_pullback_v1's "
                             f"data range (spec D1); nothing is left to test (last bar {calendar.utc_str(int(tb[-1]))})")
        cut = int((~keep).sum())
        b = b.loc[keep].reset_index(drop=True)
        a = a.loc[keep].reset_index(drop=True)
        tb, ta = tb[keep], ta[keep]
    if tb.size != ta.size or not np.array_equal(tb, ta):
        m = min(tb.size, ta.size)
        diff = np.flatnonzero(tb[:m] != ta[:m])
        j = int(diff[0]) if diff.size else m
        tb_txt = calendar.utc_str(int(tb[j])) if j < tb.size else "no row"
        ta_txt = calendar.utc_str(int(ta[j])) if j < ta.size else "no row"
        raise ValueError(f"{source}: the bid and ask files do not hold the same bars (bid {tb.size} rows, ask "
                         f"{ta.size} rows; first difference at row {j}: bid {tb_txt}, ask {ta_txt}). Export both "
                         "sides for the same range; nothing is joined or filled silently.")
    step = bars_mod.infer_bar_seconds(tb)
    if step != int(bar_seconds):
        raise ValueError(f"{source}: the bars are {step} s apart, expected {int(bar_seconds)} s "
                         f"({'M15' if bar_seconds == M15_SECONDS else 'M1'} bars)")
    off = tb % int(bar_seconds) != 0
    if off.any():
        i = int(np.flatnonzero(off)[0])
        raise ValueError(f"{source}: row {i} ({calendar.utc_str(int(tb[i]))}) does not open on a "
                         f"{int(bar_seconds)} s boundary; bars must open on UTC boundaries (D1)")
    spread_open = a["open"].to_numpy() - b["open"].to_numpy()
    if (spread_open < 0).any():
        i = int(np.flatnonzero(spread_open < 0)[0])
        raise ValueError(f"{source}: row {i} ({calendar.utc_str(int(tb[i]))}) has ask_open below bid_open "
                         f"({a['open'].iloc[i]:g} < {b['open'].iloc[i]:g}); {int((spread_open < 0).sum())} row(s) "
                         "affected. The ask must be >= the bid at the open (the entry spread of rules 5 and 10).")
    frame = pd.DataFrame({"time": tb})
    for side, df in (("bid", b), ("ask", a)):
        for col in bars_mod.PRICE_COLUMNS:
            frame[f"{side}_{col}"] = df[col].to_numpy(dtype=np.float64)
    frame["spread_open"] = spread_open.astype(np.float64)
    frame.attrs["zeno_v1"] = bidask_summary(frame, cut_before_range_start=cut)
    return frame


def bidask_summary(frame: pd.DataFrame, cut_before_range_start: int = 0) -> dict[str, Any]:
    """A JSON-serialisable description of a zeno frame: bars, range, bar size, gaps, spread_open median
    and p90 (USD/oz), counts of ask < bid per price column, and the lock (recomputed from the frame, so it
    survives pandas operations that drop attrs). For M15 frames also D1's range start [SI-62]:
    cut_before_range_start (bars before 2015-01-01 00:00 UTC that bidask_frame cut), starts_after_range_start
    (the first bar's server day is after 2015-01-02, the first server day of 2015 with gold trading, so the
    warm-up and the 2015-2017 period are shorter than the spec's) and range_note (plain text, "" when
    neither)."""
    t = frame["time"].to_numpy(dtype=np.int64)
    step = bars_mod.infer_bar_seconds(t)
    sp = frame["spread_open"].to_numpy(dtype=np.float64)
    below = {col: int((frame[f"ask_{col}"].to_numpy() < frame[f"bid_{col}"].to_numpy()).sum())
             for col in bars_mod.PRICE_COLUMNS}
    return {
        "n_bars": int(t.size), "bar_seconds": int(step),
        "first_time": int(t[0]), "last_time": int(t[-1]),
        "first_time_utc": calendar.utc_str(int(t[0])), "last_time_utc": calendar.utc_str(int(t[-1])),
        "n_gaps": int((np.diff(t) > step).sum()),
        "spread_open_median": float(np.median(sp)), "spread_open_p90": float(np.quantile(sp, 0.9)),
        "ask_below_bid": below,
        "lock_utc": LOCK_TEXT,
    } | (_range_fields(t, int(cut_before_range_start)) if step == M15_SECONDS else {})


def _range_fields(t: np.ndarray, cut: int) -> dict[str, Any]:
    late = bool(server_day(int(t[0])) > FIRST_SERVER_DAY_2015)
    notes = []
    if cut:
        notes.append(f"{cut} bar(s) before 2015-01-01 00:00 UTC were cut: D1's range starts there")
    if late:
        notes.append(f"the data starts {calendar.utc_str(int(t[0]))}, after the first trading day of 2015 "
                     "(2015-01-02): D1's range starts 2015-01-01, so the warm-up and the 2015-2017 period differ "
                     "from the spec's")
    return {"range_start_utc": RANGE_START_TEXT, "cut_before_range_start": cut, "starts_after_range_start": late,
            "range_note": "; ".join(notes)}


def _load_pair(bid_path, ask_path, bar_seconds: int, label: str) -> pd.DataFrame:
    for path, side in ((bid_path, "bid"), (ask_path, "ask")):        # refuse locked paths before any read
        bars_mod.check_not_locked(path, what=f"{label} {side} file")
    bid, bid_name = _read_table(bid_path, f"{label} bid file")
    ask, ask_name = _read_table(ask_path, f"{label} ask file")
    frame = bidask_frame(bid, ask, bar_seconds=bar_seconds, source=label)
    from propkit.report import file_sha256
    frame.attrs["zeno_v1"].update({
        "bid_file": str(bid_path), "ask_file": str(ask_path),
        "bid_sha256": file_sha256(bid_path), "ask_sha256": file_sha256(ask_path)})
    return frame


def load_m15_bidask(bid_path, ask_path) -> pd.DataFrame:
    """Read Dukascopy-style XAUUSD M15 BID and ASK bar files (D1) and return one validated zeno frame.

    Files: Parquet (.parquet/.pq) or CSV (.csv/.txt) in the formats propkit.bars.load_bars accepts (time,
    open, high, low, close[, tick_volume]) or a dukascopy-node CSV (timestamp in ms, open, high, low,
    close[, volume]); UTC. Locked-holdout paths are refused before anything is opened (LockedPathError);
    any bar opening at or after 2025-09-28 00:00 UTC is refused (HoldoutLockError, no override). See
    bidask_frame for every check and the columns; frame.attrs["zeno_v1"] also records both files and their
    sha256."""
    return _load_pair(bid_path, ask_path, M15_SECONDS, "M15")


def load_m1_bidask(bid_path, ask_path) -> pd.DataFrame:
    """Read M1 BID and ASK bar files for the D15 second run (resolve_with_m1); same formats, checks and
    lock as load_m15_bidask, with 60 s bars."""
    return _load_pair(bid_path, ask_path, M1_SECONDS, "M1")


def _check_frame(frame: pd.DataFrame, what: str = "frame") -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{what} must be the DataFrame from load_m15_bidask / bidask_frame")
    missing = [c for c in FRAME_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} is missing column(s) {missing}; build it with load_m15_bidask or bidask_frame")
    check_before_lock(frame["time"].to_numpy(dtype=np.int64), what)


# ---------------------------------------------------------------------------------------
# the ask side of one cost cell (Costs: S1, S2, multipliers)

def s2_spread(times) -> np.ndarray:
    """Spread base S2 per bar, USD/oz: 0.20 when the bar opens in 21:00-24:00 UTC (05:00-08:00 SGT, the
    rollover), else 0.18 (zeno's broker numbers; Costs). times: UTC epoch seconds (array)."""
    t = np.asarray(times, dtype=np.int64)
    tod = t % calendar.SECONDS_PER_DAY
    roll = (tod >= S2_ROLLOVER_UTC[0]) & (tod < S2_ROLLOVER_UTC[1])
    return np.where(roll, S2_ROLLOVER_USD, S2_USD).astype(np.float64)


def ask_side(frame: pd.DataFrame, spread_base: str = "S1", cost_mult: float = 1.0) -> dict[str, np.ndarray]:
    """Effective ask prices of one cost cell, USD/oz: ask_open, ask_high, ask_low, ask_close, and
    spread_entry = the spread used by rules 5 and 10 (ask_open - bid_open of the cell, D11).

    S1: ask_x = bid_x + k x (data ask_x - bid_x) per bar and price (at k = 1 exactly the ask file's prices,
    so short stops and targets trigger on the ask bars' high/low). S2: ask_x = bid_x + k x S2(bar)
    (s2_spread). k = cost_mult > 0. Works on a zeno frame of any bar size (M15 or M1)."""
    if spread_base not in SPREAD_BASES:
        raise ValueError(f"spread_base must be one of {SPREAD_BASES}, got {spread_base!r}")
    k = _num(cost_mult, "cost_mult", 0.0, lo_open=True)
    out: dict[str, np.ndarray] = {}
    if spread_base == "S1":
        for col in bars_mod.PRICE_COLUMNS:
            a = frame[f"ask_{col}"].to_numpy(dtype=np.float64)
            if k == 1.0:
                out[f"ask_{col}"] = a.copy()
            else:
                b = frame[f"bid_{col}"].to_numpy(dtype=np.float64)
                out[f"ask_{col}"] = b + k * (a - b)
        d_open = frame["ask_open"].to_numpy(dtype=np.float64) - frame["bid_open"].to_numpy(dtype=np.float64)
    else:
        d = s2_spread(frame["time"].to_numpy(dtype=np.int64))
        for col in bars_mod.PRICE_COLUMNS:
            out[f"ask_{col}"] = frame[f"bid_{col}"].to_numpy(dtype=np.float64) + k * d
        d_open = d
    out["spread_entry"] = d_open if k == 1.0 else k * d_open
    return out


def to_propkit_bars(frame: pd.DataFrame, spread_base: str = "S1", cost_mult: float = 1.0) -> pd.DataFrame:
    """propkit BARS for the equity engine: time, open/high/low/close = the BID bars, spread = the cell's
    entry spread (spread_open under S1 at k = 1, the default; ask_side(...)["spread_entry"] otherwise),
    so propkit's ask-side marks match the cell. Validated with propkit.bars.validate_bars."""
    _check_frame(frame)
    sp = ask_side(frame, spread_base, cost_mult)["spread_entry"]
    b = pd.DataFrame({"time": frame["time"].to_numpy(dtype=np.int64),
                      "open": frame["bid_open"].to_numpy(dtype=np.float64),
                      "high": frame["bid_high"].to_numpy(dtype=np.float64),
                      "low": frame["bid_low"].to_numpy(dtype=np.float64),
                      "close": frame["bid_close"].to_numpy(dtype=np.float64),
                      "spread": sp})
    return bars_mod.validate_bars(b, source="zeno_v1 bars")


# ---------------------------------------------------------------------------------------
# indicators and clocks (D1-D3, D17-D19, D21)

def h1_from_m15(frame: pd.DataFrame) -> pd.DataFrame:
    """1h bars built from the M15 BID bars on UTC hour boundaries (D1).

    One H1 bar per UTC hour holding at least one M15 bar [SI-5]: time = the hour's start, open = the first
    M15 bid open, high/low = max/min of the hour's bid highs/lows, close = the LAST M15 bid close,
    close_time = time + 3600 (D3: the bar is usable from that instant), n_m15 = M15 bars in the hour."""
    t = frame["time"].to_numpy(dtype=np.int64)
    hour = t // H1_SECONDS
    starts = np.flatnonzero(np.r_[True, hour[1:] != hour[:-1]])
    ends = np.r_[starts[1:], t.size]
    bo = frame["bid_open"].to_numpy(dtype=np.float64)
    bh = frame["bid_high"].to_numpy(dtype=np.float64)
    bl = frame["bid_low"].to_numpy(dtype=np.float64)
    bc = frame["bid_close"].to_numpy(dtype=np.float64)
    h1_time = hour[starts] * H1_SECONDS
    return pd.DataFrame({
        "time": h1_time,
        "open": bo[starts],
        "high": np.maximum.reduceat(bh, starts) if t.size else bh,
        "low": np.minimum.reduceat(bl, starts) if t.size else bl,
        "close": bc[ends - 1],
        "n_m15": (ends - starts).astype(np.int64),
        "close_time": h1_time + H1_SECONDS,
    })


def ema30_h1(h1_close) -> np.ndarray:
    """EMA30 of the 1h bid closes (D2): alpha = 2/31, seeded with the simple average of the first 30
    closes (NaN before the 30th H1 bar). propkit.indicators.ema(..., seed="sma")."""
    return ind.ema(h1_close, EMA_PERIOD, seed="sma")


def atr14_m15(frame: pd.DataFrame) -> np.ndarray:
    """ATR14 on the M15 bid bars (D2): Wilder smoothing of the true range (previous close, so a gap
    counts, [SI-7]); NaN for the first 13 bars. propkit.indicators.atr."""
    return ind.atr(frame["bid_high"].to_numpy(), frame["bid_low"].to_numpy(), frame["bid_close"].to_numpy(),
                   ATR_PERIOD)


def h1_index_at_m15_close(m15_time, h1_close_time, bar_seconds: int = M15_SECONDS) -> np.ndarray:
    """For each M15 bar, the index of the last CLOSED 1h bar at that bar's close t = time + 900 (D3): the
    H1 bar whose close_time <= t (a bar closing exactly at t is usable at t, one closing at t + 15 min is
    not). -1 when none. int64 array."""
    t_close = np.asarray(m15_time, dtype=np.int64) + int(bar_seconds)
    return (np.searchsorted(np.asarray(h1_close_time, dtype=np.int64), t_close, side="right") - 1).astype(np.int64)


def trend_state(h1_close, ema) -> tuple[np.ndarray, np.ndarray]:
    """Rule 2 per H1 bar j: long_ok = close[j] > ema[j] and ema[j] > ema[j-5]; short_ok = close[j] < ema[j]
    and ema[j] < ema[j-5] ("5 bars ago" = 5 H1 bars of the data earlier, [SI-6]). False while any input is
    NaN or j < 5. Returns two bool arrays."""
    c = np.asarray(h1_close, dtype=np.float64)
    e = np.asarray(ema, dtype=np.float64)
    if c.shape != e.shape:
        raise ValueError(f"h1_close and ema must have the same length, got {c.size} and {e.size}")
    prev = np.full(e.size, np.nan)
    if e.size > EMA_SLOPE_BARS:
        prev[EMA_SLOPE_BARS:] = e[:-EMA_SLOPE_BARS]
    ok = np.isfinite(c) & np.isfinite(e) & np.isfinite(prev)
    with np.errstate(invalid="ignore"):
        return ok & (c > e) & (e > prev), ok & (c < e) & (e < prev)


def server_day(ts):
    """The broker server day (D21) of each instant: the calendar date of New York time + 7 h, so the day
    runs 17:00 New York -> 17:00 New York (21:00 UTC in US summer time, 22:00 UTC otherwise; US DST rule
    from propkit.calendar). It is propkit.calendar.firm_day(ts, "ny_17") (server clock = New York + 7 h).
    int64 days since 1970-01-01; scalar or array."""
    arr = np.asarray(ts, dtype=np.int64)
    out = calendar.firm_day(arr, "ny_17")
    return int(out) if arr.ndim == 0 else np.asarray(out, dtype=np.int64)


def time_exit_instant(day):
    """UTC instant of 16:30 New York on the given server day(s) (D17): 20:30 UTC in US summer time, 21:30
    UTC otherwise. day: int days since 1970-01-01 (server_day output); scalar or array."""
    d = np.asarray(day, dtype=np.int64)
    local = d * calendar.SECONDS_PER_DAY + TIME_EXIT_NY_SECONDS        # New York wall clock read as UTC
    off = np.asarray(calendar.ny_offset_hours(local + 5 * calendar.SECONDS_PER_HOUR), dtype=np.int64)
    out = local - off * calendar.SECONDS_PER_HOUR
    return int(out) if d.ndim == 0 else out.astype(np.int64)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> _dt.date:
    """The n-th (1-based; n = -1: the last) given weekday (Monday 0) of a month."""
    if n > 0:
        first = _dt.date(year, month, 1)
        return first + _dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    nxt = _dt.date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - _dt.timedelta(days=1)
    return last - _dt.timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> _dt.date:
    """Gregorian Easter Sunday (the anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    return _dt.date(year, month, (h + l_ - 7 * m + 114) % 31 + 1)


def _observed(d: _dt.date) -> _dt.date:
    """A fixed-date holiday on a Saturday is observed the Friday before, on a Sunday the Monday after."""
    return d - _dt.timedelta(days=1) if d.weekday() == 5 else d + _dt.timedelta(days=1) if d.weekday() == 6 else d


def us_early_close_dates(year: int) -> frozenset:
    """The declared US holiday and early-close dates of a year [SI-64], fixed before any result: New Year's
    Day, Martin Luther King Jr. Day, Presidents' Day, Good Friday, Memorial Day, Juneteenth (from 2022),
    July 3 and Independence Day, Labor Day, Thanksgiving and the day after, Christmas Eve and Day, New
    Year's Eve; a fixed-date holiday on a weekend also counts on its observed weekday."""
    D = _dt.date
    fixed = [D(year, 1, 1), D(year, 7, 4), D(year, 12, 25)] + ([D(year, 6, 19)] if year >= 2022 else [])
    out = set(fixed) | {_observed(d) for d in fixed}
    out |= {D(year, 7, 3), D(year, 12, 24), D(year, 12, 31), _observed(D(year + 1, 1, 1))}
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    out |= {_nth_weekday(year, 1, 0, 3), _nth_weekday(year, 2, 0, 3), _easter(year) - _dt.timedelta(days=2),
            _nth_weekday(year, 5, 0, -1), _nth_weekday(year, 9, 0, 1), thanksgiving,
            thanksgiving + _dt.timedelta(days=1)}
    return frozenset(d for d in out if d.year == year)


def us_early_close_day(day: int) -> bool:
    """True when the date `day` (int days since 1970-01-01; for D17 the server day, whose 16:30 New York
    falls on that date) is a declared US holiday or early-close day (us_early_close_dates) [SI-64]."""
    d = _dt.date(1970, 1, 1) + _dt.timedelta(days=int(day))
    return d in us_early_close_dates(d.year)


def session_ok(t):
    """D19: True when an entry time (the trigger bar's close, UTC epoch s) is in [07:00, 10:00) or
    [12:30, 16:00) UTC = [15:00, 18:00) or [20:30, 24:00) SGT (Singapore has no clock change)."""
    arr = np.asarray(t, dtype=np.int64)
    tod = arr % calendar.SECONDS_PER_DAY
    ok = np.zeros(arr.shape, dtype=bool)
    for a, b in SESSION_WINDOWS_UTC:
        ok |= (tod >= a) & (tod < b)
    return bool(ok) if arr.ndim == 0 else ok


def trading_days(times) -> tuple[np.ndarray, np.ndarray]:
    """The trading days of a bar series [SI-8]: (days, first_bar) where days are the sorted unique server
    days with at least one bar and first_bar[r] the index of the first bar of day r, with one extra entry
    = the number of bars (so day r's bars are first_bar[r]:first_bar[r+1])."""
    sd = np.asarray(server_day(np.asarray(times, dtype=np.int64)), dtype=np.int64)
    if sd.size and (np.diff(sd) < 0).any():
        raise ValueError("bar times must be sorted")
    starts = np.flatnonzero(np.r_[True, sd[1:] != sd[:-1]]) if sd.size else np.zeros(0, dtype=np.int64)
    return sd[starts], np.r_[starts, sd.size].astype(np.int64)


def vol_medians(atr, first_bar) -> np.ndarray:
    """D18 default: for each trading-day rank r (0 .. n_days, the last one = a day after the data), the
    median of every finite M15 ATR14 value of the previous 20 trading days (r-20 .. r-1, today excluded),
    pooled per bar [SI-9]; NaN when fewer than 20 earlier trading days exist. USD/oz."""
    a = np.asarray(atr, dtype=np.float64)
    fb = np.asarray(first_bar, dtype=np.int64)
    n_days = fb.size - 1
    out = np.full(n_days + 1, np.nan)
    for r in range(VOL_MEDIAN_DAYS, n_days + 1):
        seg = a[fb[r - VOL_MEDIAN_DAYS]:fb[r]]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[r] = float(np.median(seg))
    return out


# ---------------------------------------------------------------------------------------
# news (D20, D23)

def _in_windows(T: np.ndarray, arr: np.ndarray, before: int, after: int) -> np.ndarray:
    """True where arr lies in [T - before, T + after] of some T (T sorted, both ends included)."""
    if T.size == 0:
        return np.zeros(arr.shape, dtype=bool)
    k = np.searchsorted(T, arr - after, side="left")                  # first event with T >= t - after
    kk = np.minimum(k, T.size - 1)
    return (k < T.size) & (T[kk] - before <= arr)


@dataclass(frozen=True, eq=False)
class NewsCalendar:
    """Scheduled release instants of NFP, CPI, PPI and FOMC statements (D20), UTC epoch seconds, sorted.

    times: int64 array T; names / kinds: per event ("NFP", ..., "scheduled"/"unscheduled"); source: the
    file read (or "in-memory"); sha256 of that file when read from disk."""

    times: np.ndarray
    names: tuple = ()
    kinds: tuple = ()
    source: str = "in-memory"
    sha256: str | None = None

    def blocked(self, t):
        """True where an entry time t (UTC epoch s) lies in [T - 30 min, T + 60 min] of some event T
        (both ends blocked, D20). Scalar or array."""
        arr = np.asarray(t, dtype=np.int64)
        out = _in_windows(self.times, arr, NEWS_BEFORE_S, NEWS_AFTER_S)
        return bool(out) if arr.ndim == 0 else out

    def unscheduled(self) -> np.ndarray:
        """True per event whose kind is "unscheduled" (an unscheduled FOMC statement, D20)."""
        if len(self.kinds) != self.times.size:
            return np.zeros(self.times.size, dtype=bool)
        return np.array([str(k).strip().lower() == "unscheduled" for k in self.kinds], dtype=bool)

    def blocked_before_unscheduled_only(self, t):
        """True where an entry time t is blocked (blocked) ONLY by the [T - 30 min, T) part of an unscheduled
        row: nobody could know at t that the statement was coming. blocked() still blocks it, as D20 says;
        this only marks it [SI-63]. Scalar or array."""
        arr = np.asarray(t, dtype=np.int64)
        u = self.unscheduled()
        T = self.times
        known = _in_windows(T[~u], arr, NEWS_BEFORE_S, NEWS_AFTER_S) | _in_windows(T[u], arr, 0, NEWS_AFTER_S)
        out = _in_windows(T, arr, NEWS_BEFORE_S, NEWS_AFTER_S) & ~known
        return bool(out) if arr.ndim == 0 else out

    def summary(self) -> dict[str, Any]:
        """Counts per event name, first and last event (UTC text), the source file and its sha256."""
        names, counts = np.unique(np.asarray(self.names, dtype=object).astype(str), return_counts=True) \
            if len(self.names) else (np.array([]), np.array([]))
        return {"source": self.source, "sha256": self.sha256, "n_events": int(self.times.size),
                "n_unscheduled": int(self.unscheduled().sum()),
                "per_event": {str(a): int(b) for a, b in zip(names, counts)},
                "first_utc": calendar.utc_str(int(self.times[0])) if self.times.size else None,
                "last_utc": calendar.utc_str(int(self.times[-1])) if self.times.size else None}


def news_calendar(times, names: Sequence[str] | None = None, kinds: Sequence[str] | None = None,
                  source: str = "in-memory") -> NewsCalendar:
    """A NewsCalendar from instants (UTC epoch s, or anything propkit.bars.to_epoch_seconds reads); names
    and kinds default to "event" / "scheduled". Sorted by time."""
    if isinstance(times, pd.Series):
        raw = times.reset_index(drop=True)
    elif isinstance(times, np.ndarray):
        raw = pd.Series(np.atleast_1d(times))
    else:
        raw = pd.Series(list(np.atleast_1d(times)) if np.ndim(times) == 0 else list(times))
    t = bars_mod.to_epoch_seconds(raw, "news time") if len(raw) else np.zeros(0, dtype=np.int64)
    n = t.size
    nm = list(names) if names is not None else ["event"] * n
    kd = list(kinds) if kinds is not None else ["scheduled"] * n
    if len(nm) != n or len(kd) != n:
        raise ValueError("names and kinds must have one entry per time")
    order = np.argsort(t, kind="stable")
    return NewsCalendar(times=t[order].astype(np.int64), names=tuple(nm[i] for i in order),
                        kinds=tuple(kd[i] for i in order), source=source)


def read_news_csv(path) -> NewsCalendar:
    """Read the US macro calendar CSV (columns event, date_et, time_et, utc_offset_ny, datetime_utc, kind,
    ...; research/news_calendar/README.md) and keep NFP, CPI, PPI and FOMC rows, scheduled and
    unscheduled (D20). Only datetime_utc is used (ISO text ending in Z). Refuses locked paths, a missing
    file, missing columns and a file with no such event."""
    df, name = _read_table(path, "news calendar")
    for col in ("event", "datetime_utc"):
        if col not in df.columns:
            raise ValueError(f"news calendar {name} is missing the column '{col}' (expected event, date_et, "
                             "time_et, utc_offset_ny, datetime_utc, kind, ...)")
    ev = df["event"].astype(str).str.strip().str.upper()
    keep = ev.isin(NEWS_EVENTS).to_numpy()
    if not keep.any():
        raise ValueError(f"news calendar {name} has no {', '.join(NEWS_EVENTS)} rows")
    sub = df.loc[keep]
    t = bars_mod.to_epoch_seconds(sub["datetime_utc"].reset_index(drop=True), "news calendar datetime_utc")
    kinds = sub["kind"].astype(str).str.strip().tolist() if "kind" in sub.columns else ["scheduled"] * len(sub)
    cal = news_calendar(t, ev[keep].tolist(), kinds, source=str(path))
    from propkit.report import file_sha256
    return dataclasses.replace(cal, sha256=file_sha256(path))


# ---------------------------------------------------------------------------------------
# FundingPips' restricted events (addendum A1, variant "master_fp")

def _in_spans(lo: np.ndarray, hi: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """True where arr lies in [lo_i, hi_i] of some span i (both ends included; spans of any length that may
    overlap): sort by lo, take the running maximum of hi, and compare it with arr."""
    if lo.size == 0:
        return np.zeros(arr.shape, dtype=bool)
    order = np.argsort(lo, kind="stable")
    lo_s, hi_max = lo[order], np.maximum.accumulate(hi[order])
    k = np.searchsorted(lo_s, arr, side="right")                     # spans with lo <= t
    return (k > 0) & (hi_max[np.maximum(k - 1, 0)] >= arr)


def fp_duration_s(name: str, testimony: bool) -> int:
    """How long a restricted event lasts after T, seconds (addendum A1): 180 min for a Fed Chair testimony,
    60 min for every other Fed Chair appearance, 0 for a release."""
    if str(name).strip().upper() != FEDCHAIR_EVENT:
        return 0
    return FEDCHAIR_TESTIMONY_S if testimony else FEDCHAIR_OTHER_S


@dataclass(frozen=True, eq=False)
class RestrictedCalendar:
    """FundingPips' Master restricted USD events (addendum A1) for the variant "master_fp".

    start: int64 T (UTC epoch s) of each event with a known time, sorted; end: T_end per event (T for a
    release, T + 180 min for a Fed Chair testimony, T + 60 min for any other Fed Chair appearance, fp_duration_s);
    names, kinds ("scheduled" / "unscheduled") and testimony (bool) per event; unknown_days: the New York
    calendar dates (int64 days since 1970-01-01, sorted, unique) of the events whose time is unknown, and
    unknown_rows their (event, 'YYYY-MM-DD') pairs; source and sha256 of the file read."""

    start: np.ndarray
    end: np.ndarray
    names: tuple = ()
    kinds: tuple = ()
    testimony: tuple = ()
    unknown_days: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    unknown_rows: tuple = ()
    source: str = "in-memory"
    sha256: str | None = None

    def unscheduled(self) -> np.ndarray:
        """True per known-time event whose kind is "unscheduled"."""
        if len(self.kinds) != self.start.size:
            return np.zeros(self.start.size, dtype=bool)
        return np.array([str(k).strip().lower() == "unscheduled" for k in self.kinds], dtype=bool)

    def in_window(self, t):
        """True where t (UTC epoch s) lies in [T - 5 min, T_end + 5 min] of some event (both ends blocked)."""
        arr = np.asarray(t, dtype=np.int64)
        out = _in_spans(self.start - FP_BEFORE_S, self.end + FP_AFTER_S, arr)
        return bool(out) if arr.ndim == 0 else out

    def on_unknown_day(self, t):
        """True where the New York calendar date of t is the date of an event whose time is unknown."""
        arr = np.asarray(t, dtype=np.int64)
        out = np.isin(np.asarray(calendar.ny_day(arr), dtype=np.int64), self.unknown_days)
        return bool(out) if arr.ndim == 0 else out

    def blocked(self, t):
        """The A1 entry block for one instant: in_window(t) or on_unknown_day(t). Scalar or array."""
        arr = np.asarray(t, dtype=np.int64)
        out = np.asarray(self.in_window(arr)) | np.asarray(self.on_unknown_day(arr))
        return bool(out) if arr.ndim == 0 else out

    def blocked_before_unscheduled_only(self, t):
        """True where t is blocked ONLY by the [T - 5 min, T) part of unscheduled rows: nobody could know at t
        that the event was coming. blocked() still blocks it, as A1 says; this only marks it [SI-63]."""
        arr = np.asarray(t, dtype=np.int64)
        u = self.unscheduled()
        lo, hi = self.start - FP_BEFORE_S, self.end + FP_AFTER_S
        known = (_in_spans(lo[~u], hi[~u], arr) | _in_spans(self.start[u], hi[u], arr)
                 | np.asarray(self.on_unknown_day(arr)))
        out = np.asarray(self.blocked(arr)) & ~known
        return bool(out) if arr.ndim == 0 else out

    def summary(self) -> dict[str, Any]:
        """Counts per event name, the Fed Chair testimony / other split, the unknown-time rows, the first and
        last event (UTC text), the source file and its sha256."""
        names, counts = np.unique(np.asarray(self.names, dtype=object).astype(str), return_counts=True) \
            if len(self.names) else (np.array([]), np.array([]))
        fed = np.array([str(x).upper() == FEDCHAIR_EVENT for x in self.names], dtype=bool)
        tes = np.asarray(self.testimony, dtype=bool) if len(self.testimony) == self.start.size \
            else np.zeros(self.start.size, dtype=bool)
        return {"source": self.source, "sha256": self.sha256, "n_events_known_time": int(self.start.size),
                "n_unknown_time": len(self.unknown_rows),
                "unknown_time_rows": [{"event": a, "date_et": b} for a, b in self.unknown_rows],
                "n_unscheduled": int(self.unscheduled().sum()),
                "fedchair_testimony": int((fed & tes).sum()), "fedchair_other": int((fed & ~tes).sum()),
                "per_event": {str(a): int(b) for a, b in zip(names, counts)},
                "first_utc": calendar.utc_str(int(self.start[0])) if self.start.size else None,
                "last_utc": calendar.utc_str(int(self.start[-1])) if self.start.size else None,
                "window": f"[T - {FP_BEFORE_S // 60} min, T_end + {FP_AFTER_S // 60} min]; T_end = T (release), "
                          f"T + {FEDCHAIR_TESTIMONY_S // 60} min (Fed Chair testimony), T + {FEDCHAIR_OTHER_S // 60} "
                          "min (other Fed Chair appearance)"}


def restricted_calendar(times, names: Sequence[str] | None = None, kinds: Sequence[str] | None = None,
                        testimony: Sequence[bool] | None = None, unknown: Sequence[tuple[str, str]] = (),
                        source: str = "in-memory") -> RestrictedCalendar:
    """A RestrictedCalendar from known-time instants (UTC epoch s, or anything propkit.bars.to_epoch_seconds
    reads) with their event names (default "event", a release), kinds (default "scheduled") and testimony
    flags (default False), plus unknown = (event, New York date 'YYYY-MM-DD') pairs for the events whose
    time is unknown. T_end follows fp_duration_s. Sorted by time."""
    raw = pd.Series(list(np.atleast_1d(times))) if not isinstance(times, pd.Series) else times.reset_index(drop=True)
    t = bars_mod.to_epoch_seconds(raw, "restricted event time") if len(raw) else np.zeros(0, dtype=np.int64)
    n = t.size
    nm = [str(x).strip().upper() for x in names] if names is not None else ["event"] * n
    kd = [str(x).strip() for x in kinds] if kinds is not None else ["scheduled"] * n
    ts_ = [bool(x) for x in testimony] if testimony is not None else [False] * n
    if len(nm) != n or len(kd) != n or len(ts_) != n:
        raise ValueError("names, kinds and testimony must have one entry per time")
    dur = np.array([fp_duration_s(a, b) for a, b in zip(nm, ts_)], dtype=np.int64)
    order = np.argsort(t, kind="stable")
    unk_rows, unk_days = [], []
    for ev, d in unknown:
        try:
            day = (_dt.date.fromisoformat(str(d).strip()) - _dt.date(1970, 1, 1)).days
        except ValueError:
            raise ValueError(f"restricted event {ev} with an unknown time needs its New York date as YYYY-MM-DD, "
                             f"got {d!r}")
        unk_rows.append((str(ev).strip().upper(), str(d).strip()))
        unk_days.append(day)
    return RestrictedCalendar(start=t[order].astype(np.int64), end=(t + dur)[order].astype(np.int64),
                              names=tuple(nm[i] for i in order), kinds=tuple(kd[i] for i in order),
                              testimony=tuple(ts_[i] for i in order),
                              unknown_days=np.unique(np.asarray(unk_days, dtype=np.int64)),
                              unknown_rows=tuple(unk_rows), source=source)


def read_restricted_csv(path=RESTRICTED_CSV) -> RestrictedCalendar:
    """Read FundingPips' restricted USD events (addendum A1; default the packaged file RESTRICTED_CSV, columns
    event, date_et, time_et, utc_offset_ny, datetime_utc, kind, basis, source_list, note). Every row is kept.
    T = datetime_utc (ISO text ending in Z); a row whose time_et is "unknown" or whose datetime_utc is blank is
    an unknown-time event kept as its New York date (date_et). A FEDCHAIR row is a testimony when its note
    holds "testimony" (any case). Refuses locked paths, a missing file and missing columns."""
    df, name = _read_table(path, "restricted calendar")
    for col in ("event", "date_et", "time_et", "datetime_utc"):
        if col not in df.columns:
            raise ValueError(f"restricted calendar {name} is missing the column '{col}' (expected event, date_et, "
                             "time_et, utc_offset_ny, datetime_utc, kind, basis, source_list, note)")
    if not len(df):
        raise ValueError(f"restricted calendar {name} has no rows")
    ev = df["event"].fillna("").astype(str).str.strip().str.upper()
    when = df["datetime_utc"].fillna("").astype(str).str.strip()
    unknown = ((when == "") | (df["time_et"].fillna("").astype(str).str.strip().str.lower() == "unknown")).to_numpy()
    note = df["note"].fillna("").astype(str) if "note" in df.columns else pd.Series([""] * len(df))
    tes = ((ev == FEDCHAIR_EVENT) & note.str.contains("testimony", case=False, regex=False)).to_numpy()
    kinds = df["kind"].fillna("scheduled").astype(str).str.strip() if "kind" in df.columns \
        else pd.Series(["scheduled"] * len(df))
    known = ~unknown
    t = bars_mod.to_epoch_seconds(when[known].reset_index(drop=True), "restricted calendar datetime_utc") \
        if known.any() else np.zeros(0, dtype=np.int64)
    dates = df["date_et"].fillna("").astype(str).str.strip()
    cal = restricted_calendar(t, ev[known].tolist(), kinds[known].tolist(), tes[known].tolist(),
                              unknown=list(zip(ev[unknown].tolist(), dates[unknown].tolist())), source=str(path))
    from propkit.report import file_sha256
    return dataclasses.replace(cal, sha256=file_sha256(path))


# ---------------------------------------------------------------------------------------
# the setup state machines (D4-D10)

def _after_min(values: np.ndarray, idx: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """For each t: (min of values[idx[t]+1 .. t], the LATEST position of that min), with idx[t] in
    (t - w, t]; (NaN, -1) when idx[t] < 0 or idx[t] == t. Vectorised over windows of w bars."""
    n = values.size
    ext = np.full(n, np.nan)
    pos = np.full(n, -1, dtype=np.int64)
    if n < w:
        return ext, pos
    chunk = 65536
    for lo in range(w - 1, n, chunk):
        hi = min(n, lo + chunk)
        t = np.arange(lo, hi)
        win = sliding_window_view(values[lo - w + 1:hi], w)              # row r <-> bar t = lo + r
        rel = idx[lo:hi] - (t - w + 1)                                    # column of the H bar
        masked = np.where(np.arange(w)[None, :] > rel[:, None], win, np.inf)[:, ::-1]
        k = np.argmin(masked, axis=1)                                     # newest first: latest on a tie
        m = masked[np.arange(masked.shape[0]), k]
        ok = (idx[lo:hi] >= 0) & np.isfinite(m)
        ext[lo:hi][ok] = m[ok]
        pos[lo:hi][ok] = (t - k)[ok]
    return ext, pos


def _prescan(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray) -> dict[str, np.ndarray]:
    """The long machine's per-bar arming inputs (D5-D8), vectorised; the short machine runs on negated
    prices (high' = -low, low' = -high, close' = -close), which mirrors every comparison exactly."""
    n = high.size
    hi_idx = ind.swing_high_index(high, H_LOOKBACK)                 # latest bar of the 20-bar max (D5)
    lwin = ind.swing_low(low, L_LOOKBACK)                           # min of the 20 lows ending at t
    ok = hi_idx >= L_LOOKBACK                                       # 20 bars exist before H [SI-43]
    H = np.full(n, np.nan)
    L = np.full(n, np.nan)
    H[ok] = high[hi_idx[ok]]
    L[ok] = lwin[hi_idx[ok] - 1]                                    # the 20 bars ENDING just before H
    leg = H - L
    min_low, pl_bar = _after_min(low, hi_idx, H_LOOKBACK)           # bars after H only [SI-35]
    min_close, _ = _after_min(close, hi_idx, H_LOOKBACK)
    with np.errstate(invalid="ignore"):
        can = (ok & np.isfinite(atr) & np.isfinite(min_low)
               & (leg >= MIN_LEG_ATR * atr)                          # D7: ATR at the arming bar
               & (min_low <= H - RETRACE_VALID * leg + LEVEL_TOL)    # D6: a wick touches 50%
               & (min_close >= H - RETRACE_VOID * leg - LEVEL_TOL))  # D8: no close beyond 78.6% since H
    return {"hi_idx": hi_idx, "H": H, "L": L, "leg": leg, "pl_bar": pl_bar, "can": can}


def _machine(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray) -> list[dict[str, Any]]:
    """The LONG setup machine on (possibly negated) bid bars; events in machine coordinates."""
    pre = _prescan(high, low, close, atr)
    can = pre["can"]
    hi_idx = pre["hi_idx"].tolist()
    Hs, Ls, legs, plb = pre["H"].tolist(), pre["L"].tolist(), pre["leg"].tolist(), pre["pl_bar"].tolist()
    hl, ll, cl, al = high.tolist(), low.tolist(), close.tolist(), np.asarray(atr, dtype=np.float64).tolist()
    can_l = can.tolist()
    events: list[dict[str, Any]] = []
    s: dict[str, Any] | None = None
    # A used-up setup (D10) never comes back: neither its H bar nor its last pullback-low bar can define a
    # new setup, so a later close of the same pullback is never chased [SI-46].
    used_hi = -1
    used_pl = -1
    seq = 0

    def emit(name: str, i: int, st: dict[str, Any]) -> None:
        nonlocal used_pl
        ev = dict(st)
        ev["event"] = name
        ev["bar"] = i
        events.append(ev)
        used_pl = max(used_pl, st["pl_bar"])

    def trigger_or_expire(i: int, st: dict[str, Any]) -> bool:
        k = i - st["pl_bar"]
        if 1 <= k <= TRIGGER_MAX_BARS and not st["passed"] and cl[i] > st["trig"]:
            emit("trigger", i, st)                              # D9 + D10: one shot
            return True
        if k >= TRIGGER_MAX_BARS:
            emit("expired", i, st)                              # no trigger within 8 bars of the low
            return True
        return False

    for i in range(high.size):
        if s is not None:
            if cl[i] < s["void"] - LEVEL_TOL:
                emit("voided", i, s)                            # D6: a close beyond 78.6% (wins, [SI-11])
                s = None
            elif hl[i] > s["h"]:
                emit("cancelled_new_extreme", i, s)             # D8: a new high above the frozen H
                s = None
            elif ll[i] <= s["pl_level"]:                        # D9: the latest lowest low moves the bar
                s["pl_bar"], s["pl_level"], s["trig"] = i, ll[i], hl[i]     # and restarts the count [SI-12]
                s["passed"] = False                             # a new bar: its first close is still to come
            elif trigger_or_expire(i, s):
                s = None
        if s is None and can_l[i] and hi_idx[i] > used_hi and plb[i] > used_pl:    # D8: arm on the first
            # closed bar where D5-D7 hold
            seq += 1
            h, leg, p = Hs[i], legs[i], plb[i]
            trig = hl[p]
            # [SI-61] rule 4 / D9 / D10: the trigger is the FIRST close above the pullback-low bar's high. When
            # the setup arms late (D7's ATR or an older H in the window held it back), that first close may
            # already lie between the pullback-low bar and the arming bar: then it has passed, no later close
            # of this pullback-low bar is chased, and only a new pullback low (which restarts the count) can
            # still trigger. Uses closes p+1 .. i-1 only (causal).
            passed = any(cl[q] > trig for q in range(p + 1, i))
            s = {"seq": seq, "ext_bar": hi_idx[i], "arm_bar": i, "h": h, "l": Ls[i], "leg": leg,
                 "atr_arm": al[i], "retrace": h - RETRACE_VALID * leg, "void": h - RETRACE_VOID * leg,
                 "pl_bar": p, "pl_level": ll[p], "trig": trig, "passed": passed}
            used_hi = hi_idx[i]
            emit("armed", i, s)
            if passed:
                emit("first_close_before_arming", i, s)
            if trigger_or_expire(i, s):                          # the arming bar may trigger [SI-13]
                s = None
    return events


def setup_machines(high, low, close, atr) -> list[dict[str, Any]]:
    """Run the long and the short setup state machines (rules 3-4, D5-D10) on BID bars and return every
    event in time order (bar index, then long before short).

    high, low, close: the M15 bid bars (USD/oz); atr: ATR14 per bar (NaN = not available; no setup arms
    on a NaN). Each event is a dict: event (armed, first_close_before_arming [SI-61], cancelled_new_extreme,
    voided, expired, trigger), side (+1 long / -1 short), bar (index of the bar at whose close it happened),
    setup_id ("L7", "S3"), ext_bar (the H bar of a long, the L bar of a short), arm_bar, pl_bar (the
    pullback-low bar of a long,
    the pullback-high bar of a short), pl_level (that bar's low / high), trigger_level (that bar's high /
    low, which the close must cross strictly), h_level, l_level (H and L, frozen at arming), leg (H - L),
    atr_arm, retrace_level (50%), void_level (78.6%), bars_since_pullback. Everything at bar i uses bars
    0..i only."""
    h = np.asarray(high, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    a = np.asarray(atr, dtype=np.float64)
    if not (h.size == lo.size == c.size == a.size):
        raise ValueError("high, low, close and atr must have the same length")
    out: list[tuple] = []
    for side, (hh, ll, cc) in ((1, (h, lo, c)), (-1, (-lo, -h, -c))):
        for ev in _machine(hh, ll, cc, a):
            if side > 0:
                rec = {"h_level": ev["h"], "l_level": ev["l"], "pl_level": ev["pl_level"],
                       "trigger_level": ev["trig"], "retrace_level": ev["retrace"], "void_level": ev["void"]}
            else:                                               # back from the negated coordinates
                rec = {"h_level": -ev["l"], "l_level": -ev["h"], "pl_level": -ev["pl_level"],
                       "trigger_level": -ev["trig"], "retrace_level": -ev["retrace"], "void_level": -ev["void"]}
            rec.update({"event": ev["event"], "side": side, "bar": ev["bar"],
                        "setup_id": f"{'L' if side > 0 else 'S'}{ev['seq']}", "ext_bar": ev["ext_bar"],
                        "arm_bar": ev["arm_bar"], "pl_bar": ev["pl_bar"], "leg": ev["leg"], "atr_arm": ev["atr_arm"],
                        "bars_since_pullback": ev["bar"] - ev["pl_bar"]})
            out.append((ev["bar"], 0 if side > 0 else 1, len(out), rec))
    out.sort(key=lambda x: x[:3])
    return [x[3] for x in out]


# ---------------------------------------------------------------------------------------
# the cost-independent layer

@dataclass(eq=False)
class Prepared:
    """Everything about one data set that does not depend on the cost cell (build it with prepare).

    Per M15 bar (arrays of the frame's length): time, atr (ATR14), trend_long / trend_short (rule 2 at the
    bar's close, D3/D4), server_day (of the bar's open), close_rank (trading-day rank of the server day of
    the bar's close), session_ok and news_blocked (of the bar's close, D19/D20), news_pre_unscheduled (that
    block comes only from the 30 minutes before an unscheduled row [SI-63]). Per H1 bar: h1 (with the ema
    column). Per trading-day rank: days, first_bar, vol_median (D18). events: setup_machines output.
    injected: names of indicators replaced through test_indicators (empty for a real run).
    Addendum A1 (only when a restricted calendar is given, else None): restricted; fp_close / fp_open: the
    bar's CLOSE (a trigger close) / OPEN (an entry fill) is blocked by RestrictedCalendar.blocked; and
    fp_close_pre_unscheduled / fp_open_pre_unscheduled: that block comes only from the 5 minutes before an
    unscheduled row [SI-63]."""

    frame: pd.DataFrame
    news: NewsCalendar | None
    time: np.ndarray
    atr: np.ndarray
    h1: pd.DataFrame
    trend_long: np.ndarray
    trend_short: np.ndarray
    server_day: np.ndarray
    close_day: np.ndarray
    close_rank: np.ndarray
    days: np.ndarray
    first_bar: np.ndarray
    vol_median: np.ndarray
    session_ok: np.ndarray
    news_blocked: np.ndarray
    events: list
    injected: tuple = ()
    news_pre_unscheduled: np.ndarray | None = None
    restricted: RestrictedCalendar | None = None
    fp_close: np.ndarray | None = None
    fp_open: np.ndarray | None = None
    fp_close_pre_unscheduled: np.ndarray | None = None
    fp_open_pre_unscheduled: np.ndarray | None = None

    @property
    def n(self) -> int:
        """Number of M15 bars."""
        return int(self.time.size)

    def triggers(self) -> list[dict[str, Any]]:
        """The trigger events, in time order."""
        return [e for e in self.events if e["event"] == "trigger"]


def prepare(frame: pd.DataFrame, news: NewsCalendar | None = None, *, restricted: RestrictedCalendar | None = None,
            test_indicators: Mapping[str, Any] | None = None) -> Prepared:
    """Build the cost-independent layer of zeno_pullback_v1 from a zeno frame (load_m15_bidask).

    news: the D20 calendar (read_news_csv); None means NO news blackout at all and is meant for tests (the
    result's meta says so). restricted: FundingPips' restricted events (read_restricted_csv), needed only by
    the variant "master_fp" (addendum A1); the other variants never read it. test_indicators (TEST ONLY,
    documented here and nowhere else): a dict with "atr14" (one value per M15 bar, USD/oz) and/or "ema30_h1" (one value per H1 bar of h1_from_m15(frame),
    USD/oz) that replaces the computed indicator, so a hand-computed case can fix them; NaN is allowed and
    means "not available". The default path computes both from the bars (atr14_m15, ema30_h1)."""
    _check_frame(frame)
    t = frame["time"].to_numpy(dtype=np.int64)
    if t.size < 2 or (np.diff(t) <= 0).any():
        raise ValueError("the frame must hold at least 2 bars, sorted by time with no duplicates")
    injected: list[str] = []
    extra = dict(test_indicators or {})
    unknown = sorted(set(extra) - {"atr14", "ema30_h1"})
    if unknown:
        raise ValueError(f"test_indicators accepts only 'atr14' and 'ema30_h1', got {unknown}")
    h1 = h1_from_m15(frame)
    if "atr14" in extra:
        atr = np.asarray(extra["atr14"], dtype=np.float64).copy()
        if atr.shape != (t.size,):
            raise ValueError(f"test_indicators['atr14'] must have one value per M15 bar ({t.size}), got {atr.shape}")
        injected.append("atr14")
    else:
        atr = atr14_m15(frame)
    if "ema30_h1" in extra:
        ema = np.asarray(extra["ema30_h1"], dtype=np.float64).copy()
        if ema.shape != (len(h1),):
            raise ValueError(f"test_indicators['ema30_h1'] must have one value per H1 bar ({len(h1)}), "
                             f"got {ema.shape}")
        injected.append("ema30_h1")
    else:
        ema = ema30_h1(h1["close"].to_numpy())
    h1 = h1.assign(ema=ema)
    long_h1, short_h1 = trend_state(h1["close"].to_numpy(), ema)
    j = h1_index_at_m15_close(t, h1["close_time"].to_numpy())
    jj = np.maximum(j, 0)
    trend_long = (j >= 0) & long_h1[jj]
    trend_short = (j >= 0) & short_h1[jj]
    sday = np.asarray(server_day(t), dtype=np.int64)
    t_close = t + M15_SECONDS
    cday = np.asarray(server_day(t_close), dtype=np.int64)
    days, first_bar = trading_days(t)
    rank = np.searchsorted(days, cday, side="left").astype(np.int64)
    vol_med = vol_medians(atr, first_bar)
    sess = np.asarray(session_ok(t_close), dtype=bool)
    nb = np.asarray(news.blocked(t_close), dtype=bool) if news is not None else np.zeros(t.size, dtype=bool)
    npu = np.asarray(news.blocked_before_unscheduled_only(t_close), dtype=bool) if news is not None \
        else np.zeros(t.size, dtype=bool)
    events = setup_machines(frame["bid_high"].to_numpy(), frame["bid_low"].to_numpy(),
                            frame["bid_close"].to_numpy(), atr)
    fp: dict[str, Any] = {}
    if restricted is not None:
        if not isinstance(restricted, RestrictedCalendar):
            raise ValueError("restricted must come from zeno_v1.read_restricted_csv (a RestrictedCalendar)")
        fp = {"restricted": restricted,
              "fp_close": np.asarray(restricted.blocked(t_close), dtype=bool),
              "fp_open": np.asarray(restricted.blocked(t), dtype=bool),
              "fp_close_pre_unscheduled": np.asarray(restricted.blocked_before_unscheduled_only(t_close), dtype=bool),
              "fp_open_pre_unscheduled": np.asarray(restricted.blocked_before_unscheduled_only(t), dtype=bool)}
    return Prepared(frame=frame, news=news, time=t, atr=atr, h1=h1, trend_long=trend_long, trend_short=trend_short,
                    server_day=sday, close_day=cday, close_rank=rank, days=days, first_bar=first_bar,
                    vol_median=vol_med, session_ok=sess, news_blocked=nb, events=events, injected=tuple(injected),
                    news_pre_unscheduled=npu, **fp)


# ---------------------------------------------------------------------------------------
# cost cells and configuration

@dataclass(frozen=True)
class ZenoCell:
    """One cost cell of the pre-registered grid: variant ("evaluation" | "master", D23 | "master_fp", addendum
    A1), commission in USD
    per lot round trip (5 or 10), spread base ("S1" data | "S2" zeno's broker numbers) and cost multiplier
    k (1, 1.5, 2: scales spread, commission and stop slippage). The defaults are the stage-1 cell
    (evaluation, 10, S1, x1.5)."""

    variant: str = "evaluation"
    commission_rt_per_lot: float = 10.0
    spread_base: str = "S1"
    cost_mult: float = 1.5

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        if self.spread_base not in SPREAD_BASES:
            raise ValueError(f"spread_base must be one of {SPREAD_BASES}, got {self.spread_base!r}")
        object.__setattr__(self, "commission_rt_per_lot",
                           _num(self.commission_rt_per_lot, "commission_rt_per_lot", 0.0))
        object.__setattr__(self, "cost_mult", _num(self.cost_mult, "cost_mult", 0.0, lo_open=True))

    @property
    def label(self) -> str:
        """Short text such as 'evaluation/c10/S1/x1.5'."""
        return f"{self.variant}/c{self.commission_rt_per_lot:g}/{self.spread_base}/x{self.cost_mult:g}"

    def to_dict(self) -> dict[str, Any]:
        """The four fields as a dict."""
        return dataclasses.asdict(self)


def grid_cells() -> list[ZenoCell]:
    """The 36 cells of the pre-registered grid (3 variants x 2 commissions x 2 spread bases x 3 multipliers;
    addendum A1 added "master_fp"), in a fixed order: every evaluation cell, then master, then master_fp."""
    return [ZenoCell(v, c, s, k) for v in VARIANTS for c in COMMISSIONS for s in SPREAD_BASES for k in COST_MULTS]


@dataclass(frozen=True)
class ZenoConfig:
    """A simulation's settings: the cost cell, the starting balance (USD, 100,000 = zeno's account) and the
    risk per trade as a fraction (None = the spec's 0.5%, or 0.4% in the Master variants, rule 7 / D23)."""

    cell: ZenoCell = field(default_factory=ZenoCell)
    capital_usd: float = 100_000.0
    risk_pct: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cell, ZenoCell):
            raise ValueError("cell must be a ZenoCell")
        object.__setattr__(self, "capital_usd", _num(self.capital_usd, "capital_usd", 0.0, lo_open=True))
        if self.risk_pct is not None:
            r = _num(self.risk_pct, "risk_pct", 0.0, lo_open=True)
            if r > 0.1:
                raise ValueError(f"risk_pct is a fraction (0.005 = 0.5%); {r} looks like a percent")
            object.__setattr__(self, "risk_pct", r)

    @property
    def risk_fraction(self) -> float:
        """Risk per trade as a fraction of the closed balance (0.005 evaluation, 0.004 master and master_fp by
        default)."""
        return RISK_PCT[self.cell.variant] if self.risk_pct is None else float(self.risk_pct)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable settings."""
        return {"cell": self.cell.to_dict(), "capital_usd": self.capital_usd, "risk_pct": self.risk_fraction}


def cost_model_for_cell(cell: ZenoCell) -> CostModel:
    """The propkit CostModel matching a cell for propkit.equity: commission = cell commission x k per lot
    round trip (half per fill), 100 oz per lot, spreads from the BARS of to_propkit_bars(frame, base, k)
    (spread_multiplier 1, the cell's k is already in them), no markup, no per-fill slippage (the stop
    slippage is inside the stop fill prices), swap off (D17: no position crosses 17:00 New York)."""
    return CostModel(spread_source="bar", spread_multiplier=1.0, markup_per_side=0.0, slippage_per_side=0.0,
                     commission_per_lot_round_trip=cell.commission_rt_per_lot * cell.cost_mult,
                     lot_size_oz=CONTRACT_OZ, swap_enabled=False)


# ---------------------------------------------------------------------------------------
# the Master margin cap (addendum A2)

def margin_usd(units: float, price: float) -> float:
    """FundingPips' Master margin for `units` oz (100 oz = 1 lot) at `price` USD/oz, one position (addendum A2):
    each tier of MARGIN_TIERS (0.05 lot at 1:50, the next 0.05 at 1:30, the next 0.05 at 1:25, the next 0.10
    at 1:20, the next 0.25 at 1:10, the rest at 1:5) takes only the volume inside it: sum of oz x price / lev.
    E.g. 0.50 lot at 4,000 = 400 + 666.67 + 800 + 2,000 + 10,000 = 13,866.67 USD; each lot above 0.50 adds
    80,000 USD."""
    rest = max(float(units), 0.0)
    total = 0.0
    for size, lev in MARGIN_TIERS:
        q = min(rest, size)
        if q <= 0:
            break
        total += q * float(price) / lev
        rest -= q
    return total


def max_units_within_margin(balance: float, price: float) -> float:
    """The largest whole-oz (0.01-lot) size whose margin_usd at `price` is at most `balance` (+ MARGIN_TOL_USD);
    0.0 when not even 1 oz fits (addendum A2)."""
    balance, price = float(balance), float(price)
    if not (balance > 0 and price > 0 and math.isfinite(balance) and math.isfinite(price)):
        return 0.0
    used, units = 0.0, 0.0
    for size, lev in MARGIN_TIERS:                       # whole tiers that fit, then whole oz of the next one
        cost = size * price / lev
        if used + cost <= balance + MARGIN_TOL_USD:
            used += cost
            units += size
            continue
        units += math.floor((balance + MARGIN_TOL_USD - used) * lev / price)
        break
    while units > 0 and margin_usd(units, price) > balance + MARGIN_TOL_USD:   # guard float rounding both ways
        units -= LOT_STEP_OZ
    while margin_usd(units + LOT_STEP_OZ, price) <= balance + MARGIN_TOL_USD:
        units += LOT_STEP_OZ
    return float(max(units, 0.0))


def margin_counts(positions: pd.DataFrame, decisions: pd.DataFrame, variant: str) -> dict[str, Any]:
    """What the margin rule did in one cell (addendum A2). Master variants: entries capped (n_capped) with
    their lots before (lots_uncapped) and after the cap, and the triggers blocked because not even 0.01 lot
    fit (n_blocked). Evaluation (not capped): the entries whose margin at a flat 1:10 and at a flat 1:30
    (lots x 100 x entry price / leverage) would exceed the closed balance at entry."""
    trig = decisions[decisions["event"] == "trigger"] if len(decisions) else decisions
    n = int(len(positions))
    if variant in MASTER_VARIANTS:
        cap = positions["margin_capped"].astype(bool).to_numpy() if n else np.zeros(0, dtype=bool)
        blocked = trig["reasons"].astype(str).str.split(";").apply(lambda r: "margin_cap_below_lot_step" in r) \
            if len(trig) else pd.Series([], dtype=bool)
        return {"cap_applies": True, "n_entries": n, "n_capped": int(cap.sum()),
                "lots_before_cap": round(float(positions["lots_uncapped"].to_numpy()[cap].sum()), 6) if n else 0.0,
                "lots_after_cap": round(float(positions["lots"].to_numpy()[cap].sum()), 6) if n else 0.0,
                "n_blocked": int(blocked.sum()),
                "rule": "tiered margin at the entry fill price <= the closed balance at entry (addendum A2)"}
    out: dict[str, Any] = {"cap_applies": False, "n_entries": n}
    for lev in EVAL_FLAT_LEVERAGES:
        if n:
            need = positions["units_oz"].to_numpy() * positions["entry_price"].to_numpy() / lev
            over = need > positions["balance_at_entry"].to_numpy() + MARGIN_TOL_USD
        else:
            over = np.zeros(0, dtype=bool)
        out[f"n_over_flat_1to{lev:g}"] = int(over.sum())
    out["rule"] = ("not capped (zeno's evaluation account type is not known; Standard 1:30 assumed): entries whose "
                   "margin at a flat 1:10 or 1:30 would exceed the closed balance at entry (addendum A2)")
    return out


# ---------------------------------------------------------------------------------------
# the simulation (D11-D23)

class _Pos:
    """One open position (mutable engine state)."""

    __slots__ = ("pid", "side", "ev", "trigger_bar", "entry_bar", "entry_time", "entry", "spread_entry", "stop0",
                 "stop", "be", "tp1", "tp2", "R", "units", "partial_units", "remaining", "tp1_done", "tp2_hit",
                 "balance_at_entry", "risk_budget", "atr_trigger", "atr_median", "x_bar", "x_mode", "next_bar",
                 "legs", "ambiguous", "m1_ok", "m1_missing", "slippage", "closed", "final_stamp", "day",
                 "units_uncapped", "margin_capped")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


class _Day:
    __slots__ = ("start_balance", "entries", "losses", "pnl")

    def __init__(self, start_balance: float):
        self.start_balance = start_balance
        self.entries = 0
        self.losses = 0
        self.pnl = 0.0


@dataclass(eq=False)
class CellResult:
    """The output of simulate for one cost cell.

    legs: TRADES (propkit columns, one row PER LEG: the +2R half and the rest are separate rows) plus
    position_id and leg ("tp1", "runner", "full"); risk_usd = units x the spec's R (D13: commission and
    slippage are on top, so a full stop is slightly worse than -1R) [SI-26]. positions: one row per entry
    (POSITION_COLUMNS). decisions: every setup event and every trigger with its status (DECISION_COLUMNS).
    meta: counts and settings (cell, capital, risk, M1 status, injected indicators)."""

    cell: ZenoCell
    config: ZenoConfig
    legs: pd.DataFrame
    positions: pd.DataFrame
    decisions: pd.DataFrame
    meta: dict


class _Engine:
    """One pass over the triggers of a Prepared layer for one cost cell (see simulate)."""

    def __init__(self, prep: Prepared, cfg: ZenoConfig, m1: pd.DataFrame | None, stateful: bool = True):
        self.p = prep
        self.cfg = cfg
        self.stateful = stateful                 # False: screen() - no position, no STATE_REASONS check
        cell = cfg.cell
        self.k = cell.cost_mult
        self.comm_rt = cell.commission_rt_per_lot * self.k            # USD per lot round trip
        self.comm_per_oz = self.comm_rt / CONTRACT_OZ                 # D16 breakeven offset, USD/oz
        self.slip = STOP_SLIPPAGE_USD * self.k
        self.risk = cfg.risk_fraction
        f = prep.frame
        ask = ask_side(f, cell.spread_base, self.k)
        self.t_np = prep.time
        self.t = prep.time.tolist()
        self.n = prep.n
        self.bo = f["bid_open"].to_numpy(dtype=np.float64).tolist()
        self.bh = f["bid_high"].to_numpy(dtype=np.float64).tolist()
        self.bl = f["bid_low"].to_numpy(dtype=np.float64).tolist()
        self.bc = f["bid_close"].to_numpy(dtype=np.float64).tolist()
        self.ao = ask["ask_open"].tolist()
        self.ah = ask["ask_high"].tolist()
        self.al = ask["ask_low"].tolist()
        self.ac = ask["ask_close"].tolist()
        self.sp = ask["spread_entry"].tolist()
        self.sday = prep.server_day.tolist()
        self.balance = cfg.capital_usd
        self.days: dict[int, _Day] = {}
        self.last_stamp = {1: -10 ** 12, -1: -10 ** 12}
        self.pos: _Pos | None = None
        self.closed: list[_Pos] = []
        self.status: dict[int, dict[str, Any]] = {}
        self.next_pid = 1
        self.fp = cell.variant == "master_fp"             # addendum A1
        self.margin_cap = cell.variant in MASTER_VARIANTS  # addendum A2
        if self.fp and (prep.restricted is None or prep.fp_close is None):
            raise ValueError("variant master_fp needs FundingPips' restricted calendar: prepare(frame, news, "
                             "restricted=read_restricted_csv(path)) (addendum A1)")
        self.master: dict[int, list[tuple[int, bool, bool]]] = {}
        self.master_unscheduled: list[int] = []         # positions closed only for unscheduled rows [SI-63]
        self.master_fp_only: list[int] = []             # master_fp: closed only for events outside rule 9's four
        if cell.variant in MASTER_VARIANTS and prep.news is not None:
            self._add_master_closes(prep.news.times, prep.news.unscheduled(), False)
        if self.fp:                                      # A1: every restricted event with a known time, too
            self._add_master_closes(prep.restricted.start, prep.restricted.unscheduled(), True)
        self.m1 = None
        if m1 is not None:
            a1 = ask_side(m1, cell.spread_base, self.k)
            self.m1 = {"t": m1["time"].to_numpy(dtype=np.int64),
                       "bo": m1["bid_open"].to_numpy(dtype=np.float64), "bh": m1["bid_high"].to_numpy(dtype=np.float64),
                       "bl": m1["bid_low"].to_numpy(dtype=np.float64), "ao": a1["ask_open"], "ah": a1["ask_high"],
                       "al": a1["ask_low"]}
        self.m1_counts = {"bars_resolved": 0, "bars_unresolved": 0, "bars_unresolved_no_m1": 0,
                          "bars_unresolved_m1_mismatch": 0}

    def _add_master_closes(self, times: np.ndarray, unscheduled: np.ndarray, fp_list: bool) -> None:
        """The D23 close map: for each event T, the bar holding c = T - 10 min (or the next bar when none does,
        [SI-40]) gets (c, the row is unscheduled, the event comes from the A1 restricted list)."""
        for T, unsched in zip(np.asarray(times).tolist(), np.asarray(unscheduled).tolist()):
            c = T - MASTER_CLOSE_BEFORE_S
            b = int(np.searchsorted(self.t_np, c, side="right")) - 1
            if not (b >= 0 and self.t[b] + M15_SECONDS > c):
                b += 1                                  # no bar holds T - 10 min: the next bar [SI-40]
            if 0 <= b < self.n:
                self.master.setdefault(b, []).append((c, bool(unsched), bool(fp_list)))

    # ----- bookkeeping ---------------------------------------------------------------

    def _day(self, d: int) -> _Day:
        day = self.days.get(d)
        if day is None:
            day = self.days[d] = _Day(self.balance)    # every earlier exit is booked: the day-start balance
        return day

    def _leg(self, pos: _Pos, units: float, price: float, reason: str, j: int, exit_time: int, leg: str,
             slipped: bool) -> None:
        s = pos.side
        gross = s * units * (price - pos.entry)
        comm = self.comm_rt * units / CONTRACT_OZ
        net = gross - comm
        day = self._day(self.sday[j])                    # created BEFORE this exit is booked
        self.balance += net
        day.pnl += net
        pos.legs.append({"leg": leg, "units": units, "exit_time": exit_time, "exit_price": price, "reason": reason,
                         "gross": gross, "commission": comm, "net": net, "bar": j})
        pos.remaining = round(pos.remaining - units, 9)
        if slipped:
            pos.slippage += units * self.slip

    def _close(self, pos: _Pos, price: float, reason: str, j: int, exit_time: int, stamp: int,
               slipped: bool = False) -> None:
        leg = "runner" if (pos.tp1_done and pos.partial_units > 0) else "full"
        self._leg(pos, pos.remaining, price, reason, j, exit_time, leg, slipped)
        pos.closed = True
        pos.final_stamp = stamp
        self.last_stamp[pos.side] = stamp
        if sum(x["net"] for x in pos.legs) < 0:
            self._day(self.sday[j]).losses += 1          # D21: a position with net P&L < 0 [SI-24]
        self.closed.append(pos)

    def _take_tp1(self, pos: _Pos, j: int, exit_time: int) -> None:
        pos.tp1_done = True                              # rule 6 / D16: half at +2R, stop to breakeven
        if pos.partial_units > 0:
            self._leg(pos, pos.partial_units, pos.tp1, "target", j, exit_time, "tp1", False)
        pos.stop = pos.be

    # ----- one bar of an open position ------------------------------------------------

    # Touch tests (D14) with the [SI-34] tie rule: a stop is hit when the adverse price is at or beyond it, a
    # target when the favourable price is at or beyond it, both to within LEVEL_TOL.

    @staticmethod
    def _hits_stop(s: int, price: float, level: float) -> bool:
        return s * (price - level) <= LEVEL_TOL

    @staticmethod
    def _hits_target(s: int, price: float, level: float) -> bool:
        return s * (price - level) >= -LEVEL_TOL

    def _ambiguous(self, pos: _Pos, adv: float, fav: float) -> bool:
        s = pos.side
        if not pos.tp1_done:
            if not self._hits_target(s, fav, pos.tp1):
                return False
            return self._hits_stop(s, adv, pos.stop) or self._hits_stop(s, adv, pos.be)
        return self._hits_stop(s, adv, pos.stop) and self._hits_target(s, fav, pos.tp2)

    def _intrabar(self, pos: _Pos, adv: float, fav: float, j: int) -> None:
        """D14/D15 inside one bar: adv/fav = the adverse/favourable extreme on the closing side."""
        s = pos.side
        xt, stamp = self.t[j] + M15_SECONDS - 1, self.t[j] + M15_SECONDS
        if not pos.tp1_done:
            if self._hits_stop(s, adv, pos.stop):                           # stop first (D15)
                self._close(pos, pos.stop - s * self.slip, "stop", j, xt, stamp, True)
                return
            if not self._hits_target(s, fav, pos.tp1):
                return
            self._take_tp1(pos, j, xt)
            if self._hits_stop(s, adv, pos.stop):                           # breakeven in the same bar (D15)
                self._close(pos, pos.stop - s * self.slip, "stop", j, xt, stamp, True)
            elif self._hits_target(s, fav, pos.tp2):                        # +2R and +4R in one bar [SI-17]
                pos.tp2_hit = True
                self._close(pos, pos.tp2, "target", j, xt, stamp)
            return
        if self._hits_stop(s, adv, pos.stop):
            self._close(pos, pos.stop - s * self.slip, "stop", j, xt, stamp, True)
        elif self._hits_target(s, fav, pos.tp2):
            pos.tp2_hit = True
            self._close(pos, pos.tp2, "target", j, xt, stamp)

    def _open_gaps(self, pos: _Pos, o: float, j: int, exit_time: int, stamp: int) -> bool:
        """Gap rules at an open price o (D14): stop at the open (+ slippage), targets at their level.
        Returns True when the position closed."""
        s = pos.side
        if self._hits_stop(s, o, pos.stop):
            self._close(pos, o - s * self.slip, "stop", j, exit_time, stamp, True)
            return True
        if not pos.tp1_done and self._hits_target(s, o, pos.tp1):
            self._take_tp1(pos, j, exit_time)
        if pos.tp1_done and self._hits_target(s, o, pos.tp2):
            pos.tp2_hit = True
            self._close(pos, pos.tp2, "target", j, exit_time, stamp)
            return True
        return False

    def _m1_bar(self, pos: _Pos, j: int, adv: float, fav: float) -> str:
        """Replay an ambiguous M15 bar on its M1 bars (D15 second run) and return "resolved"; "no_m1" when no M1
        bar lies inside it, "mismatch" when the M1 bars on the closing side (bid for a long, the cell's ask for
        a short) do not reach the M15 bar's adverse extreme `adv` or favourable extreme `fav` to LEVEL_TOL
        (minutes missing, or another feed): they cannot say which level the M15 bar touched first, so nothing
        is replayed and the caller keeps the M15 answer [SI-68]."""
        m = self.m1
        lo = int(np.searchsorted(m["t"], self.t[j], side="left"))
        hi = int(np.searchsorted(m["t"], self.t[j] + M15_SECONDS, side="left"))
        if lo >= hi:
            return "no_m1"
        s = pos.side
        adv1 = float(m["bl"][lo:hi].min()) if s > 0 else float(m["ah"][lo:hi].max())
        fav1 = float(m["bh"][lo:hi].max()) if s > 0 else float(m["al"][lo:hi].min())
        if s * (adv1 - adv) > LEVEL_TOL or s * (fav1 - fav) < -LEVEL_TOL:
            return "mismatch"
        xt, stamp = self.t[j] + M15_SECONDS - 1, self.t[j] + M15_SECONDS
        for q in range(lo, hi):
            if pos.closed:
                break
            if int(m["t"][q]) > self.t[j]:
                o = float(m["bo"][q]) if s > 0 else float(m["ao"][q])
                if self._open_gaps(pos, o, j, xt, stamp):
                    break
            a1 = float(m["bl"][q]) if s > 0 else float(m["ah"][q])
            f1 = float(m["bh"][q]) if s > 0 else float(m["al"][q])
            self._intrabar(pos, a1, f1, j)                       # stop first again inside an M1 bar
        return "resolved"

    def _bar(self, pos: _Pos, j: int) -> None:
        s = pos.side
        t = self.t
        o = self.bo[j] if s > 0 else self.ao[j]
        if j > pos.entry_bar:
            if self._open_gaps(pos, o, j, t[j], t[j]):
                return
            if pos.x_mode == "open" and j == pos.x_bar:                     # D17: 16:30 New York, at the open
                self._close(pos, o, "time", j, t[j], t[j])
                return
        closes = self.master.get(j)
        # D23: open at T - 10 min (c) and opened under 5 h before it. On the entry bar (only after a data gap,
        # D20 keeps the trigger close out of [T - 30 min, T + 60 min]) the fill at this open is open at c when
        # c lies in this bar, so the close is at this same open; a fill after c (the bar after a hole, [SI-40])
        # was not open at c and is kept [SI-70]. For a later bar, entry_time <= c always holds.
        live = [(u, f) for c, u, f in closes if c - MASTER_MAX_AGE_S < pos.entry_time <= c] if closes else []
        if live:
            if all(u for u, _ in live):
                self.master_unscheduled.append(pos.pid)
            if all(f for _, f in live):                 # master_fp: no rule-9 event (D23) asked for this close
                self.master_fp_only.append(pos.pid)
            self._close(pos, o, "signal", j, t[j], t[j])
            return
        adv = self.bl[j] if s > 0 else self.ah[j]
        fav = self.bh[j] if s > 0 else self.al[j]
        if self._ambiguous(pos, adv, fav):
            pos.ambiguous = True
            if self.m1 is not None:
                how = self._m1_bar(pos, j, adv, fav)
                if how == "resolved":
                    pos.m1_ok += 1
                    self.m1_counts["bars_resolved"] += 1
                else:                                                       # the M15 answer stands [SI-68]
                    pos.m1_missing += 1
                    self.m1_counts["bars_unresolved"] += 1
                    self.m1_counts["bars_unresolved_no_m1" if how == "no_m1" else "bars_unresolved_m1_mismatch"] += 1
                    self._intrabar(pos, adv, fav, j)
            else:
                self._intrabar(pos, adv, fav, j)
        else:
            self._intrabar(pos, adv, fav, j)
        if pos.closed:
            return
        c = self.bc[j] if s > 0 else self.ac[j]
        if pos.x_mode == "close" and j == pos.x_bar:                        # early close: the last bar before
            self._close(pos, c, "time", j, t[j] + M15_SECONDS, t[j] + M15_SECONDS)     # the break, at its close
        elif j == self.n - 1:
            self._close(pos, c, "end_of_data", j, t[j] + M15_SECONDS, t[j] + M15_SECONDS)

    def _advance(self, upto: int) -> None:
        pos = self.pos
        if pos is None:
            return
        j = pos.next_bar
        while not pos.closed and j <= upto:
            self._bar(pos, j)
            j += 1
        pos.next_bar = j
        if pos.closed:
            self.pos = None

    # ----- one trigger ---------------------------------------------------------------

    def _decide(self, idx: int, ev: dict[str, Any]) -> None:
        p = self.p
        i, s = ev["bar"], ev["side"]
        t_c = self.t[i] + M15_SECONDS                     # the entry time (D19)
        fail: set[str] = set()
        rank = int(p.close_rank[i])
        if rank < WARMUP_TRADING_DAYS:
            fail.add("warmup")
        has_next = i + 1 < self.n
        if not has_next:
            fail.add("no_next_bar")
        if not p.session_ok[i]:
            fail.add("outside_session")
        if p.news_blocked[i]:
            fail.add("news_blackout")
        fp_pre = None
        if self.fp:                                       # A1: the trigger close OR the entry fill
            fc = bool(p.fp_close[i])
            fo = bool(p.fp_open[i + 1]) if has_next else False
            fp_pre = False
            if fc or fo:
                fail.add("fp_restricted_window")
                fp_pre = not ((fc and not p.fp_close_pre_unscheduled[i])
                              or (fo and not p.fp_open_pre_unscheduled[i + 1]))
        atr_t = float(p.atr[i])                            # D12: ATR at the trigger bar's close
        med = float(p.vol_median[rank]) if rank < p.vol_median.size else float("nan")
        if not math.isfinite(med) or not atr_t <= VOL_CAP_X * med:
            fail.add("atr_above_2x_median")
        entry = stop = R = spread_e = float("nan")
        x_bar, x_mode = -1, None
        if has_next:
            e = i + 1
            entry = self.ao[e] if s > 0 else self.bo[e]   # D11: long at the ask open, short at the bid open
            spread_e = self.sp[e]
            if s > 0:
                stop = ev["pl_level"] - STOP_BUFFER_ATR * atr_t                     # rule 5, a BID level
            else:
                stop = ev["pl_level"] + STOP_BUFFER_ATR * atr_t + spread_e          # rule 5, an ASK level
            R = s * (entry - stop)                                                  # D13, USD/oz
            # D17 on the server day the entry counts against: the trigger close's (D19, D21, [SI-25]). Any
            # fill at or after 16:30 New York of that day is refused, however long a data gap before it.
            t_exit = time_exit_instant(int(p.close_day[i]))
            if self.t[e] >= t_exit:
                fail.add("entry_after_time_exit")
            else:
                xb = int(np.searchsorted(self.t_np, t_exit, side="left"))
                if xb < self.n and self.t[xb] == t_exit:
                    x_bar, x_mode = xb, "open"
                elif xb < self.n:
                    x_bar, x_mode = xb - 1, "close"
            if not R > 0:
                fail.add("entry_beyond_stop")
            else:
                if spread_e > MAX_SPREAD_FRAC_OF_R * R:
                    fail.add("spread_gt_10pct_of_stop")
                if R > MAX_STOP_ATR * atr_t:
                    fail.add("stop_wider_than_3_atr")
        if self.stateful:
            day = self._day(int(p.close_day[i]))          # the server day of the entry time [SI-25]
            if day.entries >= MAX_ENTRIES_PER_DAY:
                fail.add("max_entries_per_day")
            if day.losses >= MAX_LOSSES_PER_DAY:
                fail.add("two_losses_today")
            if day.pnl <= -DAY_LOSS_FRAC * day.start_balance:
                fail.add("day_loss_1pct")
            if t_c < self.last_stamp[s] + COOLDOWN_S:     # D22 measured to the entry time (D19) [SI-14]
                fail.add("cooldown_15min")
            if self.pos is not None:
                fail.add("position_open")
        trend = bool(p.trend_long[i] if s > 0 else p.trend_short[i])
        if not trend:
            fail.add("trend_disagrees")
        units = units_uncapped = 0.0
        capped = False
        budget = self.risk * self.balance
        if R > 0:
            units = units_uncapped = floor_to_lot_step(max(budget, 0.0) / R, LOT_STEP_OZ)   # D13: floor to 0.01 lot
            if units < LOT_STEP_OZ:
                fail.add("size_below_lot_step")
            elif self.margin_cap:                         # A2: the tiered margin at the entry fill price must
                fit = max_units_within_margin(self.balance, entry)      # fit in the closed balance
                if units > fit:
                    units, capped = fit, True
                    if units < LOT_STEP_OZ:
                        fail.add("margin_cap_below_lot_step")
        reasons = [r for r in BLOCK_REASONS if r in fail]
        passed = "entered" if self.stateful else ELIGIBLE
        rec = {"status": reasons[0] if reasons else passed, "reasons": ";".join(reasons), "position_id": -1,
               "entry_time": self.t[i + 1] if has_next else -1, "entry_price": entry, "stop_level": stop,
               "spread_entry": spread_e, "atr_trigger": atr_t, "atr_median": med, "trend_ok": trend,
               "news_pre_unscheduled": bool(p.news_pre_unscheduled[i]) if p.news_pre_unscheduled is not None
               else False, "fp_pre_unscheduled": fp_pre}
        if not reasons and self.stateful:
            e = i + 1
            pid = self.next_pid
            self.next_pid += 1
            pos = _Pos(pid=pid, side=s, ev=ev, trigger_bar=i, entry_bar=e, entry_time=self.t[e], entry=entry,
                       spread_entry=spread_e, stop0=stop, stop=stop, R=R, units=units,
                       partial_units=floor_to_lot_step(units * 0.5, LOT_STEP_OZ), remaining=units,
                       be=entry + s * self.comm_per_oz, tp1=entry + s * TP1_R * R, tp2=entry + s * TP2_R * R,
                       tp1_done=False, tp2_hit=False, balance_at_entry=self.balance, risk_budget=budget,
                       atr_trigger=atr_t, atr_median=med, x_bar=x_bar, x_mode=x_mode, next_bar=e, legs=[],
                       ambiguous=False, m1_ok=0, m1_missing=0, slippage=0.0, closed=False, final_stamp=None,
                       day=int(p.close_day[i]), units_uncapped=units_uncapped, margin_capped=capped)
            day.entries += 1
            self.pos = pos
            rec["position_id"] = pid
        self.status[idx] = rec

    def run(self) -> None:
        for idx, ev in enumerate(self.p.events):
            if ev["event"] != "trigger":
                continue
            self._advance(ev["bar"])                       # exits up to and including the trigger bar
            self._decide(idx, ev)
        self._advance(self.n - 1)


def _outcome(pos: _Pos) -> str:
    last = pos.legs[-1]["reason"]
    if last == "time":
        return "time-exit"
    if last == "stop" and not pos.tp1_done:
        return "-1R"
    if pos.tp1_done and pos.partial_units > 0 and last == "stop":
        return "+1R(BE)"
    if pos.tp1_done and pos.partial_units > 0 and last == "target":
        return "+3R"
    return "other"


def _time_exit_rule(pos: _Pos) -> str:
    """How a D17 time exit happened [SI-64]: "16:30_open" (a bar opens at 16:30 New York), or the last bar
    before the break because none does: "early_close_us_holiday" on a declared US holiday / early-close day
    (us_early_close_day), "early_close_other_day" on any other day (a data gap the bars only show later).
    Empty when the position did not end with a time exit."""
    if pos.legs[-1]["reason"] != "time":
        return ""
    if pos.x_mode == "open":
        return TIME_EXIT_RULES[0]
    return TIME_EXIT_RULES[1] if us_early_close_day(pos.day) else TIME_EXIT_RULES[2]


def time_exit_counts(positions: pd.DataFrame) -> dict[str, Any]:
    """Counts of the time exits per rule (TIME_EXIT_RULES) and the ids of the positions closed at the last bar
    before a break on a day that is not a declared US early close [SI-64]."""
    rule = positions["time_exit_rule"] if len(positions) else pd.Series([], dtype=object)
    other = positions.loc[rule == TIME_EXIT_RULES[2], "position_id"] if len(positions) else []
    return {"at_16_30_open": int((rule == TIME_EXIT_RULES[0]).sum()),
            "early_close_us_holiday": int((rule == TIME_EXIT_RULES[1]).sum()),
            "early_close_other_day": int((rule == TIME_EXIT_RULES[2]).sum()),
            "other_day_position_ids": [int(x) for x in other], "reading": "SI-64"}


def _assert_no_rollover(legs: pd.DataFrame) -> None:
    """D17 / Costs: swap is zero only if no position crosses 17:00 New York; raise if one would."""
    if not len(legs):
        return
    et = legs["entry_time"].to_numpy(dtype=np.int64)
    xt = legs["exit_time"].to_numpy(dtype=np.int64)
    r = calendar.rollover_instants(int(et.min()), int(xt.max()) + 2, 17, weekdays=range(7))
    if r.size == 0:
        return
    lo = np.searchsorted(r, et, side="right")       # first rollover strictly after the entry
    hi = np.searchsorted(r, xt, side="right")       # rollovers at or before the exit
    bad = hi > lo
    if bad.any():
        k = int(np.flatnonzero(bad)[0])
        raise RuntimeError(f"position {int(legs['position_id'].iloc[k])} would cross the 17:00 New York rollover "
                           f"{calendar.utc_str(int(r[lo[k]]))}; swap zero (D17) no longer holds")


def _tables(eng: _Engine) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = eng.p
    cell = eng.cfg.cell
    t = eng.t
    positions = sorted(eng.closed, key=lambda x: x.pid)
    leg_rows, pos_rows = [], []
    order = {"tp1": 0, "runner": 1, "full": 1}
    for pos in positions:
        ev = pos.ev
        legs = sorted(pos.legs, key=lambda x: order[x["leg"]])
        for lg in legs:
            leg_rows.append({
                "side": pos.side, "units": lg["units"], "entry_time": pos.entry_time, "entry_price": pos.entry,
                "exit_time": lg["exit_time"], "exit_price": lg["exit_price"], "exit_reason": lg["reason"],
                "stop_price": pos.stop0, "risk_usd": lg["units"] * pos.R, "commission_usd": lg["commission"],
                "swap_usd": 0.0, "pnl_usd": lg["net"], "position_id": pos.pid, "leg": lg["leg"]})
        first, last = legs[0], legs[-1] if len(legs) > 1 else None
        gross = sum(x["gross"] for x in legs)
        comm = sum(x["commission"] for x in legs)
        net = sum(x["net"] for x in legs)
        pos_rows.append({
            "position_id": pos.pid, "side": SIDE_NAMES[pos.side], "variant": cell.variant,
            "commission_rt_per_lot": cell.commission_rt_per_lot, "spread_base": cell.spread_base,
            "cost_mult": cell.cost_mult, "setup_id": ev["setup_id"], "server_day": calendar.day_to_str(pos.day),
            "trigger_bar": pos.trigger_bar, "trigger_time": t[pos.trigger_bar] + M15_SECONDS,
            "trigger_time_utc": calendar.utc_str(t[pos.trigger_bar] + M15_SECONDS),
            "entry_bar": pos.entry_bar, "entry_time": pos.entry_time, "entry_time_utc": calendar.utc_str(pos.entry_time),
            "entry_time_sgt": sgt_str(pos.entry_time), "entry_price": pos.entry, "spread_entry": pos.spread_entry,
            "stop_level": pos.stop0, "be_level": pos.be, "tp1_level": pos.tp1, "tp2_level": pos.tp2,
            "R_usd_per_oz": pos.R, "lots": round(pos.units / CONTRACT_OZ, 6), "units_oz": pos.units,
            "partial_lots": round(pos.partial_units / CONTRACT_OZ, 6),
            "runner_lots": round((pos.units - pos.partial_units) / CONTRACT_OZ, 6),
            "balance_at_entry": pos.balance_at_entry, "risk_budget_usd": pos.risk_budget,
            "atr_trigger": pos.atr_trigger, "h_level": ev["h_level"], "l_level": ev["l_level"], "leg_usd": ev["leg"],
            "extreme_bar_time": t[ev["ext_bar"]], "pullback_bar_time": t[ev["pl_bar"]],
            "tp1_reached": bool(pos.tp1_done), "tp2_reached": bool(pos.tp2_hit),
            "exit1_time": first["exit_time"], "exit1_time_utc": calendar.utc_str(first["exit_time"]),
            "exit1_price": first["exit_price"], "exit1_reason": first["reason"],
            "exit2_time": last["exit_time"] if last else -1,
            "exit2_time_utc": calendar.utc_str(last["exit_time"]) if last else "",
            "exit2_price": last["exit_price"] if last else float("nan"), "exit2_reason": last["reason"] if last else "",
            "final_exit_stamp": pos.final_stamp, "gross_usd": gross, "commission_usd": comm,
            "slippage_usd": pos.slippage, "swap_usd": 0.0, "net_pnl_usd": net,
            "r_multiple_net": net / (pos.units * pos.R), "outcome": _outcome(pos), "ambiguous_bar": bool(pos.ambiguous),
            "m1_bars_resolved": pos.m1_ok, "m1_bars_unresolved": pos.m1_missing, "time_exit_rule": _time_exit_rule(pos),
            "lots_uncapped": round(pos.units_uncapped / CONTRACT_OZ, 6), "margin_capped": bool(pos.margin_capped)})
    legs_df = pd.DataFrame(leg_rows, columns=[c for c in LEG_COLUMNS if c != "trade_id"])
    legs_df.insert(0, "trade_id", np.arange(1, len(legs_df) + 1, dtype=np.int64))
    if len(legs_df):
        legs_df = adapters.validate_trades(legs_df, source="zeno_v1 legs", allow_extra=True)
    else:
        legs_df = adapters.empty_trades().assign(position_id=np.zeros(0, dtype=np.int64), leg=[])
    legs_df["position_id"] = legs_df["position_id"].astype(np.int64)
    pos_df = pd.DataFrame(pos_rows, columns=list(POSITION_COLUMNS))

    dec_rows = []
    for idx, ev in enumerate(p.events):
        i = ev["bar"]
        row = {"time": t[i] + M15_SECONDS, "event": ev["event"], "side": SIDE_NAMES[ev["side"]],
               "setup_id": ev["setup_id"], "bar_index": i, "bar_time": t[i], "status": "", "reasons": "",
               "position_id": -1, "h_level": ev["h_level"], "l_level": ev["l_level"], "leg_usd": ev["leg"],
               "atr_arm": ev["atr_arm"], "retrace_level": ev["retrace_level"], "void_level": ev["void_level"],
               "extreme_bar_time": t[ev["ext_bar"]], "arm_bar_time": t[ev["arm_bar"]],
               "pullback_bar_time": t[ev["pl_bar"]], "pullback_level": ev["pl_level"],
               "trigger_level": ev["trigger_level"], "bars_since_pullback": ev["bars_since_pullback"],
               "entry_time": -1, "entry_price": float("nan"), "stop_level": float("nan"),
               "spread_entry": float("nan"), "atr_trigger": float("nan"), "atr_median": float("nan"),
               "trend_ok": None, "news_pre_unscheduled": None, "fp_pre_unscheduled": None}
        st = eng.status.get(idx)
        if st is not None:
            row.update(st)
        dec_rows.append(row)
    cols = DECISION_COLUMNS_FP if eng.fp else DECISION_COLUMNS
    dec = pd.DataFrame(dec_rows, columns=[c for c in cols if c not in ("time_utc", "time_sgt")])
    tt = dec["time"].to_numpy(dtype=np.int64)
    dec.insert(1, "time_utc", _utc_text(tt) if len(dec) else [])
    dec.insert(2, "time_sgt", sgt_str(tt) if len(dec) else [])
    return legs_df, pos_df, dec


def simulate(prep: Prepared, config: ZenoConfig | None = None, *, m1: pd.DataFrame | None = None) -> CellResult:
    """Run zeno_pullback_v1's entries and exits (rules 4-11, D10-D23) for one cost cell.

    prep: prepare(frame, news); config: ZenoConfig (cell, capital, risk; default the stage-1 cell
    evaluation/10/S1/x1.5 on 100,000 USD); m1: an M1 zeno frame (load_m1_bidask) to resolve the ambiguous
    M15 bars (D15 second run; use resolve_with_m1). Returns a CellResult with the legs (TRADES + position_id,
    leg), positions and decisions tables. Raises RuntimeError if any position would cross 17:00 New York
    (then swap would not be zero, D17)."""
    if not isinstance(prep, Prepared):
        raise ValueError("prep must come from zeno_v1.prepare(frame, news)")
    cfg = config if config is not None else ZenoConfig()
    if not isinstance(cfg, ZenoConfig):
        raise ValueError("config must be a ZenoConfig")
    if m1 is not None:
        _check_frame(m1, "M1 frame")
    eng = _Engine(prep, cfg, m1)
    eng.run()
    legs, positions, decisions = _tables(eng)
    _assert_no_rollover(legs)
    trig = decisions[decisions["event"] == "trigger"]
    meta = {
        "spec_id": SPEC_ID, "spec_version": SPEC_VERSION, "cell": cfg.cell.to_dict(), "label": cfg.cell.label,
        "capital_usd": cfg.capital_usd, "risk_pct": cfg.risk_fraction, "final_balance_usd": eng.balance,
        "n_triggers": int(len(trig)), "n_entered": int((trig["status"] == "entered").sum()),
        "status_counts": {str(k): int(v) for k, v in trig["status"].value_counts().items()},
        "n_positions": int(len(positions)), "n_legs": int(len(legs)),
        "n_ambiguous_positions": int(positions["ambiguous_bar"].sum()) if len(positions) else 0,
        "m1": M1_NOT_RUN if m1 is None else {"run": True, **eng.m1_counts},
        "news": prep.news.summary() if prep.news is not None else "no news calendar given: NO news blackout",
        "news_unscheduled": unscheduled_counts(prep, decisions, eng.master_unscheduled),
        "time_exits": time_exit_counts(positions),
        "indicators_injected": list(prep.injected),
        "margin": margin_counts(positions, decisions, cfg.cell.variant),
    }
    if cfg.cell.variant in MASTER_VARIANTS:
        meta["master_closes"] = master_close_counts(positions, eng.master_fp_only if eng.fp else None)
    if eng.fp:
        meta["restricted"] = restricted_counts(prep, decisions)
    return CellResult(cell=cfg.cell, config=cfg, legs=legs, positions=positions, decisions=decisions, meta=meta)


def unscheduled_counts(prep: Prepared, decisions: pd.DataFrame, master_closes: Sequence[int] = ()) -> dict[str, Any]:
    """What the unscheduled rows of the calendar do before their instant T [SI-63]: n_rows; the triggers whose
    news blackout comes only from the 30 minutes before an unscheduled row (n_triggers_flagged), and those
    of them with no other blocking reason (n_triggers_blocked_only_before_unscheduled); the Master positions
    closed only for an unscheduled row (master_closes_only_for_unscheduled, position ids). D20 and D23 are
    applied as written; these counts only show what a trader could not have known at that time.
    Variant master_fp (decisions with fp_pre_unscheduled): also fp_n_rows (unscheduled restricted rows),
    fp_n_triggers_flagged (the restricted-window block comes only from the 5 minutes before an unscheduled
    row) and n_triggers_blocked_only_before_unscheduled_any (the reasons are news_blackout and/or
    fp_restricted_window and each of them is flagged). n_triggers_blocked_only_before_unscheduled keeps its
    D20 meaning in every variant, so it never exceeds n_triggers_flagged."""
    trig = decisions[decisions["event"] == "trigger"]
    flag = trig["news_pre_unscheduled"].fillna(False).astype(bool).to_numpy() if len(trig) else np.zeros(0, bool)
    only = flag & (trig["reasons"].to_numpy(dtype=object) == "news_blackout") if len(trig) else flag
    out = {"n_rows": int(prep.news.unscheduled().sum()) if prep.news is not None else 0,
           "n_triggers_flagged": int(flag.sum()), "n_triggers_blocked_only_before_unscheduled": int(only.sum()),
           "master_closes_only_for_unscheduled": [int(x) for x in master_closes], "reading": "SI-63"}
    if "fp_pre_unscheduled" in decisions.columns:
        fflag = trig["fp_pre_unscheduled"].fillna(False).astype(bool).to_numpy() if len(trig) else np.zeros(0, bool)
        sets = [set(str(r).split(";")) - {""} for r in trig["reasons"].tolist()]
        either = np.array([bool(r) and r <= {"news_blackout", "fp_restricted_window"}
                           and ("news_blackout" not in r or a) and ("fp_restricted_window" not in r or b)
                           for r, a, b in zip(sets, flag, fflag)], dtype=bool)
        out["fp_n_rows"] = int(prep.restricted.unscheduled().sum()) if prep.restricted is not None else 0
        out["fp_n_triggers_flagged"] = int(fflag.sum())
        out["n_triggers_blocked_only_before_unscheduled_any"] = int(either.sum())
    return out


def master_close_counts(positions: pd.DataFrame, fp_only: Sequence[int] | None = None) -> dict[str, Any]:
    """The D23 Master closes of one cell: positions closed 10 min before an event (exit reason "signal") and,
    for master_fp, those closed only for restricted events outside rule 9's four (fp_only, position ids)."""
    n = 0
    if len(positions):
        e2 = positions["exit2_reason"].astype(str).to_numpy()
        last = np.where(e2 != "", e2, positions["exit1_reason"].astype(str).to_numpy())
        n = int((last == "signal").sum())
    out: dict[str, Any] = {"n_positions_closed": n}
    if fp_only is not None:
        out["n_closed_only_for_restricted_list"] = len(fp_only)
        out["closed_only_for_restricted_list"] = [int(x) for x in fp_only]
    return out


def restricted_counts(prep: Prepared, decisions: pd.DataFrame) -> dict[str, Any]:
    """master_fp: the restricted calendar's summary and the triggers its window blocked (n_triggers_blocked)
    and blocked with no other reason (n_triggers_blocked_only_by_it), addendum A1."""
    trig = decisions[decisions["event"] == "trigger"]
    sets = [set(str(r).split(";")) - {""} for r in trig["reasons"].tolist()] if len(trig) else []
    return {"calendar": prep.restricted.summary() if prep.restricted is not None else None,
            "n_triggers_blocked": int(sum("fp_restricted_window" in r for r in sets)),
            "n_triggers_blocked_only_by_it": int(sum(r == {"fp_restricted_window"} for r in sets))}


def screen(prep: Prepared, config: ZenoConfig | None = None) -> pd.DataFrame:
    """Stage 1's decisions table (the G0 signal check, [SI-54]): every setup event and trigger, each trigger
    checked like simulate does EXCEPT the checks that need how and when earlier trades ended (STATE_REASONS:
    the D21 daily limits, the D22 cooldown, the one open position). No position is opened, so the size uses
    the initial capital. A trigger's status is ELIGIBLE ("eligible") when no other check fails, else its
    first failing reason; reasons lists every failing one. The entered triggers of a run are a subset of
    the eligible ones of the same cell. Same columns as simulate's decisions (position_id is always -1).
    Variant master_fp also checks the restricted window (addendum A1); both Master variants size with the
    margin cap at the initial capital (A2)."""
    if not isinstance(prep, Prepared):
        raise ValueError("prep must come from zeno_v1.prepare(frame, news)")
    cfg = config if config is not None else ZenoConfig()
    if not isinstance(cfg, ZenoConfig):
        raise ValueError("config must be a ZenoConfig")
    eng = _Engine(prep, cfg, None, stateful=False)
    eng.run()
    return _tables(eng)[2]


CHART_CELL = ZenoCell("evaluation", 10.0, "S1", 1.0)     # the data's own prices: S1 at costs x1 [SI-66]
CHART_PRICE_COLUMNS = ("entry_price", "stop_level", "spread_entry")


def chart_prices(prep: Prepared) -> pd.DataFrame:
    """Each decisions row's entry fill, stop level and entry spread as a chart of the DATA shows them [SI-66]:
    a long's entry = the ask file's open of the next bar, a short's = the bid file's open; the long stop =
    pullback low - 0.25 x ATR14, the short stop = pullback high + 0.25 x ATR14 + the data's entry spread (ask
    open - bid open). These are screen()'s prices in CHART_CELL (S1 at costs x1, rule 5 and D11 as the engine
    computes them). Another spread base or multiplier moves the long's entry and the short's stop (ask = bid
    + k x spread), which no chart shows. Columns CHART_PRICE_COLUMNS (USD/oz), one row per row of screen() /
    simulate()'s decisions in the same order; NaN for setup events and a trigger without a next bar."""
    return screen(prep, ZenoConfig(CHART_CELL))[list(CHART_PRICE_COLUMNS)].reset_index(drop=True)


def cell_ask_marks(frame: pd.DataFrame, cell: ZenoCell) -> dict[str, np.ndarray]:
    """The cell's ask high and close per bar (ask_side), USD/oz, for propkit.equity's ask_prices: D14 marks and
    closes a short on the ASK, and under S1 the data's spread moves inside a bar, so bid + the open spread
    (propkit's default mark) can understate a short's adverse extreme [SI-67]. Under S2 they equal bid + the
    bar's spread exactly."""
    a = ask_side(frame, cell.spread_base, cell.cost_mult)
    return {"high": a["ask_high"], "close": a["ask_close"]}


def cell_equity(prep: Prepared, result: CellResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """propkit EQUITY and TRADES of a cell (propkit.equity.equity_from_trades on to_propkit_bars with the
    cell's spread, cost_model_for_cell, the result's capital, and the cell's ask high and close as the short
    marks (cell_ask_marks, [SI-67]); price_tolerance off because stop fills carry the slippage). The
    recomputed pnl_usd must equal the engine's to 1e-6 USD, else ValueError."""
    from propkit.equity import equity_from_trades
    cell = result.cell
    bars = to_propkit_bars(prep.frame, cell.spread_base, cell.cost_mult)
    equity, trades = equity_from_trades(bars, result.legs, result.config.capital_usd, cost_model_for_cell(cell),
                                        price_tolerance=None, ask_prices=cell_ask_marks(prep.frame, cell))
    if len(trades) and not np.allclose(trades["pnl_usd"].to_numpy(), result.legs["pnl_usd"].to_numpy(),
                                       rtol=0, atol=1e-6):
        raise ValueError("propkit.equity and zeno_v1 disagree on a leg's net P&L; this is a bug")
    return equity, trades


# ---------------------------------------------------------------------------------------
# D15 second run: M1 resolution of ambiguous bars

def resolve_with_m1(prep: Prepared, config: ZenoConfig | None, m1_bid, m1_ask,
                    base: CellResult | None = None) -> tuple[CellResult, pd.DataFrame]:
    """The D15 second run: replay every ambiguous M15 bar (one that touches the live stop and a target,
    or +2R and the breakeven stop) on its M1 bars, stop first again inside an ambiguous M1 bar.

    m1_bid, m1_ask: M1 bar files (paths, read by load_m1_bidask) or bid/ask DataFrames (bidask_frame with
    60 s bars). An ambiguous bar keeps the M15 assumption (stop first) and is counted as unresolved when no
    M1 bar lies inside it (meta "bars_unresolved_no_m1") or when its M1 bars on the closing side do not reach
    its M15 low and high (meta "bars_unresolved_m1_mismatch", [SI-68]).
    Returns (the M1-resolved CellResult, a diff table matching positions on entry time and side:
    position_id_m15, position_id_m1, outcome_m15, outcome_m1, net_pnl_m15, net_pnl_m1, diff_usd, r_m15,
    r_m1, diff_r, status = same | changed | only_m15 | only_m1). base: the M15-only result (computed when
    None)."""
    if isinstance(m1_bid, pd.DataFrame) and isinstance(m1_ask, pd.DataFrame):
        m1 = bidask_frame(m1_bid, m1_ask, bar_seconds=M1_SECONDS, source="M1")
    else:
        m1 = load_m1_bidask(m1_bid, m1_ask)
    base = base if base is not None else simulate(prep, config)
    res = simulate(prep, config, m1=m1)
    key = ["entry_time", "side"]
    cols = key + ["position_id", "outcome", "net_pnl_usd", "r_multiple_net", "ambiguous_bar"]
    a = base.positions[cols].rename(columns={c: c + "_m15" for c in cols if c not in key})
    b = res.positions[cols].rename(columns={c: c + "_m1" for c in cols if c not in key})
    d = a.merge(b, on=key, how="outer", indicator=True)
    d["diff_usd"] = d["net_pnl_usd_m1"] - d["net_pnl_usd_m15"]
    d["diff_r"] = d["r_multiple_net_m1"] - d["r_multiple_net_m15"]
    same = (d["_merge"] == "both") & (d["outcome_m15"] == d["outcome_m1"]) & (d["diff_usd"].abs() <= 1e-6)
    d["status"] = np.where(d["_merge"] == "left_only", "only_m15",
                           np.where(d["_merge"] == "right_only", "only_m1", np.where(same, "same", "changed")))
    diff = d.drop(columns="_merge").rename(columns={"net_pnl_usd_m15": "net_pnl_m15", "net_pnl_usd_m1": "net_pnl_m1",
                                                    "r_multiple_net_m15": "r_m15", "r_multiple_net_m1": "r_m1"})
    return res, diff.sort_values(key, kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# synthetic data for tests (NOT market data)

def synthetic_m15_bidask(start: int = 1420070400, n_bars: int = 2000, seed: int = 0, price: float = 1200.0,
                         vol_per_hour: float = 0.002, spread: float = 0.30) -> pd.DataFrame:
    """A deterministic synthetic zeno frame for tests and timing (NOT market data): propkit's synthetic
    metals-hours bid walk (bars.synthetic_bars, M15) with ask = bid + a per-bar spread (lognormal around
    `spread`, USD/oz, rounded to 0.01, at least 0.01). start: UTC epoch s (default 2015-01-01 00:00); the
    last bar must open before the holdout lock (HoldoutLockError otherwise)."""
    b = bars_mod.synthetic_bars(int(start), int(n_bars), bar_seconds=M15_SECONDS, seed=seed, price=price,
                                vol_per_hour=vol_per_hour, spread=spread)
    sp = np.maximum(b["spread"].to_numpy(dtype=np.float64), 0.01)
    bid = b[["time", "open", "high", "low", "close"]].copy()
    ask = bid.copy()
    for col in bars_mod.PRICE_COLUMNS:
        ask[col] = bid[col].to_numpy() + sp
    return bidask_frame(bid, ask, bar_seconds=M15_SECONDS, source="synthetic M15")
