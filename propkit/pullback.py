"""propkit/pullback.py - zeno's trend-continuation pullback rule as a parametric TRADES generator.

PLACEHOLDER WARNING: every default of PullbackSpec is a placeholder chosen only so the code runs. None
of them is zeno's rule. Fill in the real values (a JSON file, see propkit/METHODS.md section 6 and
propkit/examples/pullback_spec_example.json) and set "placeholder": false before reading any result.
Synthetic data only until zeno's spec arrives.

What one bar looks like to the generator (BARS are BID prices in USD/oz, ask = bid + the bar's spread):
  1. at the bar's OPEN: exits that are due at the open, in this order of priority: the open gaps
     through the stop (fill at the open, reason "stop"), the open gaps through the target (fill at the
     open, reason "target"), an exit decided at the previous close (trail / time, reason "trail" /
     "time"), a news flatten (reason "signal"). Then entries decided at the previous close fill at
     this open (longs: ask + markup + slippage = CostModel.buy_fill; shorts: bid - markup - slippage =
     CostModel.sell_fill), after the entry-time filters below.
  2. INSIDE the bar: the stop and the target of every open position (including one opened at this
     open) are checked against the bar's extremes: longs with the bid low / bid high, shorts with the
     ask (bid + spread) high / low. If both the stop and the target lie inside one bar the STOP is
     assumed to come first (the order of prices inside a bar is unknown; this is the conservative
     choice). Stop fills: at the stop level minus (long) / plus (short) markup and slippage; target
     fills likewise at the target level. The exact instant is unknown, so an intrabar exit is stamped
     at the bar's last second (open + bar_seconds - 1): it is booked in this bar and pays any swap
     rollover inside the bar (conservative).
  3. at the bar's CLOSE: exits for the next open are decided (trail_ema: the close is beyond the trail
     EMA; time: the position has been held time_exit_bars bars); then the pullback signal is evaluated
     from bars <= this bar only; a signal fills at the NEXT bar's open (step 1). The last bar closes
     every open position at its close (bid close for longs, ask close for shorts, plus costs), reason
     "end_of_data", stamped at the bar's end (open + bar_seconds). A signal at the last bar's close
     cannot enter (there is no next bar to fill at).

Stops are price LEVELS on the side that closes the position: a long's stop and target are BID levels
(a long exits by selling at the bid), a short's are ASK levels (a short exits by buying at the ask).
"Spread included in the stop distance": the stop is placed by market structure on the BID chart and the
spread is added on top, so the distance from the entry fill to the stop fill is the structural distance
plus the spread (plus markup and slippage on both fills):
  stop_mode "swing": long stop = pullback low - stop_buffer_atr x ATR (bid level); short stop = pullback
      high + stop_buffer_atr x ATR + the entry bar's spread (ask level);
  stop_mode "atr":   long stop = bid at the entry open - stop_atr_mult x ATR; short stop = ask at the
      entry open + stop_atr_mult x ATR.
ATR is the Wilder ATR at the SIGNAL bar. The fixed-R target is entry_fill +/- target_r x |entry_fill -
stop| (R measured from the entry fill to the stop LEVEL).

Sizing: units = floor_to_lot_step(risk_pct x balance / loss_per_oz, lot_step_oz) where balance = C0 + the
net PnL of every trade closed at or before the entry instant (open positions are not marked) and
loss_per_oz = |entry_fill - stop_fill| + round-trip commission per oz (CostModel.commission(1, entry_fill,
stop_fill)); stop_fill includes the exit markup and slippage. So units x loss_per_oz <= the risk budget
and one more lot step would exceed it. TRADES.risk_usd = units x loss_per_oz, the full 1R at that size
including costs (CLAUDE.md A9); a stop filled at its level is exactly -1R before swap.

Ledger columns are filled the same way as propkit.equity.equity_from_trades: commission_usd =
CostModel.fill_commission at the entry + at the exit (positive = paid); swap_usd = the sum of
CostModel.swap_for_night over the rollovers r with entry_time < r <= exit_time, the notional at the bid
at r (bars.rollover_bids: the open of a bar that opens at r, else the close of the last bar with open
<= r; negative = paid); pnl_usd = side x units x (exit_price - entry_price) - commission_usd + swap_usd.

Look-ahead: a signal at bar t uses bars 0..t only; the entry at bar t+1's open additionally uses bar
t+1's open price and time, and that bar's spread for the fill (the one-spread-per-bar approximation of
the cost model). The max_spread_usd filter tests the signal bar's spread by default (spread_filter_bar
"signal"): a bar's spread value is often an average, minimum or maximum over the bar, known only after
it. Cutting the data after any bar never changes a signal or an entry at or before that cut (tests
check this over many cut points).

Research only: nothing here places, simulates sending, or prepares orders.
"""
from __future__ import annotations

import bisect
import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from propkit import bars as bars_mod
from propkit import calendar
from propkit import indicators as ind
from propkit.costs import CostModel

DIRECTIONS = ("long", "short", "both")
PULLBACK_MODES = ("ema_touch", "atr_from_swing")
TRIGGERS = ("close_back_over_ema", "break_prev_extreme")
STOP_MODES = ("swing", "atr")
EXIT_MODES = ("fixed_r", "trail_ema", "time")
SPREAD_FILTER_BARS = ("signal", "entry")
SESSION_NAMES = calendar.SESSION_NAMES
MAX_RISK_PCT = 0.10          # risk_pct is a FRACTION; above 10% per trade is refused as a likely mix-up

TRADES_COLUMNS = ("trade_id", "side", "units", "entry_time", "entry_price", "exit_time", "exit_price",
                  "exit_reason", "stop_price", "risk_usd", "commission_usd", "swap_usd", "pnl_usd")
DECISION_COLUMNS = ("signal_time", "signal_bar", "side", "entry_time", "status", "trade_id")

# status of every signal in the decision log (generate_trades_detailed)
DECISION_STATUS = {
    "entered": "the signal was filled at the next bar's open",
    "no_next_bar": "signal at the close of the last bar: there is no next open to fill at",
    "conflict": "a long and a short signal on the same bar: both skipped",
    "atr_cap": "ATR at the signal bar above max_atr_usd",
    "gap_before_entry": "the next bar opens after a gap and entry_after_gap is false",
    "outside_session": "the entry instant is outside every allowed session / UTC hour range",
    "news_blackout": "the entry bar overlaps a news blackout window",
    "spread_cap": "the effective spread of the bar named by spread_filter_bar (the signal bar by default, "
                  "or the entry bar) is above max_spread_usd",
    "max_open_positions": "max_open_positions positions are already open",
    "max_trades_per_day": "max_trades_per_day entries already made this prop day",
    "daily_stop_losses": "daily_stop_losses losing trades already closed this prop day",
    "daily_stop_pct": "realised loss this prop day reached daily_stop_pct of the day-start balance",
    "open_beyond_stop": "the entry bar opens at or beyond the stop: no valid trade",
    "stop_cap": "entry-to-stop distance above max_stop_usd",
    "balance_not_positive": "the closed balance is <= 0: nothing to risk",
    "size_below_lot_step": "the risk budget buys less than one lot step",
}

PLACEHOLDER_NAME = "PLACEHOLDER - not zeno's rule; replace every value"


# ---------------------------------------------------------------------------------------
# field checks

def _is_int(v) -> bool:
    return not isinstance(v, (bool, np.bool_)) and isinstance(v, (int, np.integer))


def _int(v, name: str, lo: int) -> int:
    if isinstance(v, (float, np.floating)) and math.isfinite(v) and float(v).is_integer():
        v = int(v)
    if not _is_int(v) or int(v) < lo:
        raise ValueError(f"{name} must be a whole number >= {lo}, got {v!r}")
    return int(v)


def _opt_int(v, name: str, lo: int) -> int | None:
    return None if v is None else _int(v, name, lo)


def _float(v, name: str, lo: float | None = None, lo_open: bool = False, hi: float | None = None) -> float:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number, got {v!r}")
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    if lo is not None and (x <= lo if lo_open else x < lo):
        raise ValueError(f"{name} must be {'>' if lo_open else '>='} {lo:g}, got {x:g}")
    if hi is not None and x > hi:
        raise ValueError(f"{name} must be <= {hi:g}, got {x:g}")
    return x


def _opt_float(v, name: str, lo: float | None = None, lo_open: bool = False) -> float | None:
    return None if v is None else _float(v, name, lo, lo_open)


def _bool(v, name: str) -> bool:
    if not isinstance(v, (bool, np.bool_)):
        raise ValueError(f"{name} must be true or false, got {v!r}")
    return bool(v)


def _choice(v, name: str, options: tuple[str, ...]) -> str:
    key = str(v).strip().lower() if isinstance(v, str) else v
    if key not in options:
        raise ValueError(f"{name} must be one of {list(options)}, got {v!r}")
    return key


def _pairs(v, name: str) -> list:
    if isinstance(v, (str, bytes)) or not isinstance(v, (list, tuple)):
        raise ValueError(f"{name} must be a list of [start, end] pairs, got {v!r}")
    out = []
    for item in v:
        if isinstance(item, (str, bytes)) or not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{name}: every entry must be a [start, end] pair, got {item!r}")
        out.append(tuple(item))
    return out


def _instant(v, name: str) -> int:
    """UTC epoch seconds from an integer or an ISO text with an explicit offset ('2024-06-07T12:30:00Z')."""
    if isinstance(v, str):
        secs = int(bars_mod.to_epoch_seconds(pd.Series([v]), what=name)[0])
    elif _is_int(v) or (isinstance(v, (float, np.floating)) and math.isfinite(v) and float(v).is_integer()):
        secs = int(v)
    else:
        raise ValueError(f"{name} must be UTC epoch seconds or ISO text such as '2024-06-07T12:30:00Z', "
                         f"got {v!r}")
    calendar._as_seconds(secs, name)
    return secs


def _iso(ts: int) -> str:
    return str(np.datetime64(int(ts), "s")) + "Z"


# ---------------------------------------------------------------------------------------
# the spec

@dataclass(frozen=True)
class PullbackSpec:
    """zeno's pullback rule (CLAUDE.md C3 spec list). EVERY DEFAULT IS A PLACEHOLDER [PLACEHOLDER] chosen so
    the code runs; none is zeno's rule. Periods and lookbacks are in BARS of the data's own bar size
    (bar_minutes can pin it); prices and distances in USD per oz; fractions are fractions (0.005 = 0.5%).
    propkit/METHODS.md (section 6) documents every field; from_json / to_json read and write it.

      name, placeholder        a label, and True until the values are zeno's real rule (reports flag it).
      bar_minutes              None = any bar size, or the bar size the spec is written for (15, 30, 60);
                               a mismatch with the data raises ValueError.
      direction                "long", "short" or "both".
      trend_ema_period         trend EMA period (bars) on BID closes.
      trend_slope_bars         slope test: EMA[t] > EMA[t - n] for longs (< for shorts); 0 = no slope test.
      trend_require_close_side True: the close must be above (long) / below (short) the trend EMA.
      pullback_mode            "ema_touch": some bar of the last pullback_lookback_bars bars touched the
                               pullback EMA (low <= EMA for longs, high >= EMA for shorts); the pullback
                               extreme is the lowest low (long) / highest high (short) of those bars.
                               "atr_from_swing": the swing high (long) is the latest highest high of the
                               last swing_lookback_bars bars; the pullback extreme is the lowest low AFTER
                               it (none if the swing high is the signal bar itself); depth = (swing high -
                               extreme) / ATR must lie in [min_depth_atr, max_depth_atr] (mirrored for
                               shorts).
      ema_pullback_period      the pullback EMA (ema_touch) and the trigger EMA (close_back_over_ema).
      pullback_lookback_bars   window of the ema_touch test and of its pullback extreme.
      swing_lookback_bars      window of the swing high / low (atr_from_swing).
      min_depth_atr, max_depth_atr  depth bounds in ATR (max None = no upper bound).
      trigger                  at the signal bar's close: "close_back_over_ema" (close > EMA and the low
                               of this bar or the previous close was at or below the EMA; mirrored for
                               shorts) or "break_prev_extreme" (close > previous bar's high; short: close <
                               previous bar's low). Filled at the next bar's open.
      stop_mode                "swing" (pullback extreme -/+ stop_buffer_atr x ATR) or "atr" (entry quote
                               -/+ stop_atr_mult x ATR); see the module docstring for the spread.
      stop_buffer_atr, stop_atr_mult, atr_period   in ATR units / bars (Wilder ATR).
      exit_mode                one exit, the stop always live: "fixed_r" (target at target_r x R),
                               "trail_ema" (a close beyond the trail EMA exits at the next open), "time"
                               (exit at the open after time_exit_bars bars held).
      target_r, trail_ema_period, time_exit_bars   parameters of the chosen exit.
      risk_pct                 fraction of the current closed balance risked per trade (0.005 = 0.5%);
                               values above 0.10 are refused as a percent/fraction mix-up.
      max_stop_usd             skip if |entry fill - stop| > this many USD per oz (None = no cap).
      max_atr_usd              skip if ATR at the signal bar > this many USD per oz (None = no cap).
      sessions                 entries only inside one of these propkit.calendar sessions ('asia',
                               'london', 'newyork', 'overlap'), tested at the entry instant;
      session_hours_utc        ... or inside one of these UTC hour ranges [start, end) (e.g. [[7, 16]];
                               [22, 2] wraps midnight). Both empty = any time.
      news_blackouts_utc       [start, end) UTC windows (epoch seconds or ISO text with 'Z'): no entry
                               whose entry bar overlaps a window;
      news_flatten             True: also close open positions at the open of the first bar that overlaps
                               a window (reason "signal").
      max_spread_usd           skip if the effective spread of the spread_filter_bar > this (None = no
                               filter).
      spread_filter_bar        which bar's spread max_spread_usd tests: "signal" (default; the signal
                               bar, fully known when the decision is made) or "entry" (the entry bar;
                               causal only if the data's spread is the spread at the bar OPEN - many
                               files hold a bar average, minimum or maximum, which is known only later).
      max_trades_per_day       entries per prop day (CE(S)T date of the entry), None = no limit.
      max_open_positions       positions open at once (default 1).
      daily_stop_losses        no new entry this prop day after this many losing closed trades (None = off).
      daily_stop_pct           no new entry this prop day once the realised PnL of the day is <= -this
                               fraction of the day-start closed balance (None = off).
      entry_after_gap          False: skip a signal whose next bar opens after a gap (weekend, daily break).
      warmup_bars              no signal before this bar index; None = automatic (see warmup()).
    """

    name: str = PLACEHOLDER_NAME
    placeholder: bool = True
    bar_minutes: int | None = None
    direction: str = "both"
    trend_ema_period: int = 200
    trend_slope_bars: int = 10
    trend_require_close_side: bool = True
    pullback_mode: str = "ema_touch"
    ema_pullback_period: int = 20
    pullback_lookback_bars: int = 5
    swing_lookback_bars: int = 20
    min_depth_atr: float = 1.0
    max_depth_atr: float | None = 3.0
    trigger: str = "close_back_over_ema"
    stop_mode: str = "swing"
    stop_buffer_atr: float = 0.25
    stop_atr_mult: float = 1.5
    atr_period: int = 14
    exit_mode: str = "fixed_r"
    target_r: float = 2.0
    trail_ema_period: int = 20
    time_exit_bars: int = 24
    risk_pct: float = 0.005
    max_stop_usd: float | None = None
    max_atr_usd: float | None = None
    sessions: tuple = ()
    session_hours_utc: tuple = ()
    news_blackouts_utc: tuple = ()
    news_flatten: bool = False
    max_spread_usd: float | None = None
    spread_filter_bar: str = "signal"
    max_trades_per_day: int | None = None
    max_open_positions: int = 1
    daily_stop_losses: int | None = None
    daily_stop_pct: float | None = None
    entry_after_gap: bool = True
    warmup_bars: int | None = None

    def __post_init__(self) -> None:
        s = object.__setattr__
        if not isinstance(self.name, str):
            raise ValueError(f"name must be text, got {self.name!r}")
        s(self, "placeholder", _bool(self.placeholder, "placeholder"))
        bm = _opt_int(self.bar_minutes, "bar_minutes", 1)
        s(self, "bar_minutes", bm)
        s(self, "direction", _choice(self.direction, "direction", DIRECTIONS))
        s(self, "trend_ema_period", _int(self.trend_ema_period, "trend_ema_period", 1))
        s(self, "trend_slope_bars", _int(self.trend_slope_bars, "trend_slope_bars", 0))
        s(self, "trend_require_close_side", _bool(self.trend_require_close_side, "trend_require_close_side"))
        s(self, "pullback_mode", _choice(self.pullback_mode, "pullback_mode", PULLBACK_MODES))
        s(self, "ema_pullback_period", _int(self.ema_pullback_period, "ema_pullback_period", 1))
        s(self, "pullback_lookback_bars", _int(self.pullback_lookback_bars, "pullback_lookback_bars", 1))
        s(self, "swing_lookback_bars", _int(self.swing_lookback_bars, "swing_lookback_bars", 2))
        s(self, "min_depth_atr", _float(self.min_depth_atr, "min_depth_atr", 0.0))
        s(self, "max_depth_atr", _opt_float(self.max_depth_atr, "max_depth_atr", 0.0, lo_open=True))
        if self.max_depth_atr is not None and self.max_depth_atr < self.min_depth_atr:
            raise ValueError(f"max_depth_atr ({self.max_depth_atr:g}) is below min_depth_atr "
                             f"({self.min_depth_atr:g})")
        s(self, "trigger", _choice(self.trigger, "trigger", TRIGGERS))
        s(self, "stop_mode", _choice(self.stop_mode, "stop_mode", STOP_MODES))
        s(self, "stop_buffer_atr", _float(self.stop_buffer_atr, "stop_buffer_atr", 0.0))
        s(self, "stop_atr_mult", _float(self.stop_atr_mult, "stop_atr_mult", 0.0, lo_open=True))
        s(self, "atr_period", _int(self.atr_period, "atr_period", 1))
        s(self, "exit_mode", _choice(self.exit_mode, "exit_mode", EXIT_MODES))
        s(self, "target_r", _float(self.target_r, "target_r", 0.0, lo_open=True))
        s(self, "trail_ema_period", _int(self.trail_ema_period, "trail_ema_period", 1))
        s(self, "time_exit_bars", _int(self.time_exit_bars, "time_exit_bars", 1))
        risk = _float(self.risk_pct, "risk_pct", 0.0, lo_open=True)
        if risk > MAX_RISK_PCT:
            raise ValueError(f"risk_pct is a FRACTION of the balance (0.005 = 0.5% per trade); {risk:g} would "
                             f"risk {risk:.0%} per trade. Values above {MAX_RISK_PCT:g} are refused.")
        s(self, "risk_pct", risk)
        s(self, "max_stop_usd", _opt_float(self.max_stop_usd, "max_stop_usd", 0.0, lo_open=True))
        s(self, "max_atr_usd", _opt_float(self.max_atr_usd, "max_atr_usd", 0.0, lo_open=True))
        s(self, "sessions", self._sessions(self.sessions))
        s(self, "session_hours_utc", self._hours(self.session_hours_utc))
        s(self, "news_blackouts_utc", self._news(self.news_blackouts_utc))
        s(self, "news_flatten", _bool(self.news_flatten, "news_flatten"))
        s(self, "max_spread_usd", _opt_float(self.max_spread_usd, "max_spread_usd", 0.0))
        s(self, "spread_filter_bar", _choice(self.spread_filter_bar, "spread_filter_bar", SPREAD_FILTER_BARS))
        s(self, "max_trades_per_day", _opt_int(self.max_trades_per_day, "max_trades_per_day", 1))
        s(self, "max_open_positions", _int(self.max_open_positions, "max_open_positions", 1))
        s(self, "daily_stop_losses", _opt_int(self.daily_stop_losses, "daily_stop_losses", 1))
        dsp = _opt_float(self.daily_stop_pct, "daily_stop_pct", 0.0, lo_open=True)
        if dsp is not None and dsp >= 1.0:
            raise ValueError(f"daily_stop_pct is a FRACTION of the day-start balance (0.02 = 2%), got {dsp:g}")
        s(self, "daily_stop_pct", dsp)
        s(self, "entry_after_gap", _bool(self.entry_after_gap, "entry_after_gap"))
        s(self, "warmup_bars", _opt_int(self.warmup_bars, "warmup_bars", 0))

    @staticmethod
    def _sessions(v) -> tuple:
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"sessions must be a list of session names {list(SESSION_NAMES)}, got {v!r}")
        out = []
        for name in v:
            key = _choice(name, "sessions entry", SESSION_NAMES)
            if key not in out:
                out.append(key)
        return tuple(out)

    @staticmethod
    def _hours(v) -> tuple:
        out = []
        for a, b in _pairs(v, "session_hours_utc"):
            a = _float(a, "session_hours_utc start", 0.0, hi=24.0)
            b = _float(b, "session_hours_utc end", 0.0, hi=24.0)
            if a == b or a == 24.0 or b == 0.0:
                raise ValueError(f"session_hours_utc range [{a:g}, {b:g}] is ambiguous: start must be in "
                                 "[0, 24), end in (0, 24] and different from the start ([22, 2] wraps "
                                 "midnight; [0, 24] is the whole day)")
            out.append((a, b))
        return tuple(out)

    @staticmethod
    def _news(v) -> tuple:
        out = []
        for a, b in _pairs(v, "news_blackouts_utc"):
            a = _instant(a, "news_blackouts_utc start")
            b = _instant(b, "news_blackouts_utc end")
            if b <= a:
                raise ValueError(f"news_blackouts_utc window {_iso(a)} .. {_iso(b)}: the end must be after "
                                 "the start")
            out.append((a, b))
        return tuple(sorted(out))

    # ----- derived values ---------------------------------------------------------------

    @property
    def uses_trend_ema(self) -> bool:
        """True when the trend EMA is used (close-side test or slope test)."""
        return self.trend_require_close_side or self.trend_slope_bars > 0

    @property
    def uses_pullback_ema(self) -> bool:
        """True when the pullback EMA is used (ema_touch pullback or close_back_over_ema trigger)."""
        return self.pullback_mode == "ema_touch" or self.trigger == "close_back_over_ema"

    def warmup(self) -> int:
        """First bar index at which a signal may fire (bars, counted from the first bar of the data).

        warmup_bars when given; otherwise the largest of: atr_period; 3 x trend_ema_period +
        trend_slope_bars (if the trend EMA is used); 3 x ema_pullback_period (if the pullback EMA is used);
        swing_lookback_bars (atr_from_swing); pullback_lookback_bars (ema_touch); 3 x trail_ema_period
        (trail_ema exit). After 3 x period bars an EMA's seed weighs < 0.25%.
        """
        if self.warmup_bars is not None:
            return self.warmup_bars
        parts = [self.atr_period]
        if self.uses_trend_ema:
            parts.append(3 * self.trend_ema_period + self.trend_slope_bars)
        if self.uses_pullback_ema:
            parts.append(3 * self.ema_pullback_period)
        if self.pullback_mode == "atr_from_swing":
            parts.append(self.swing_lookback_bars)
        else:
            parts.append(self.pullback_lookback_bars)
        if self.exit_mode == "trail_ema":
            parts.append(3 * self.trail_ema_period)
        return int(max(parts))

    def sides(self) -> tuple[int, ...]:
        """The sides the spec trades: (+1,), (-1,) or (+1, -1)."""
        return {"long": (1,), "short": (-1,), "both": (1, -1)}[self.direction]

    # ----- serialisation ----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """All fields as a JSON-serialisable dict (lists for the pairs, ISO 'Z' text for news windows);
        from_dict(to_dict()) gives an equal spec."""
        d = dataclasses.asdict(self)
        d["sessions"] = list(self.sessions)
        d["session_hours_utc"] = [list(p) for p in self.session_hours_utc]
        d["news_blackouts_utc"] = [[_iso(a), _iso(b)] for a, b in self.news_blackouts_utc]
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PullbackSpec":
        """Build a spec from a dict. Keys starting with '_' are comments and are ignored; unknown keys raise
        ValueError (a typo must not silently fall back to a placeholder); missing keys take the
        placeholder defaults."""
        if not isinstance(data, Mapping):
            raise ValueError("a pullback spec must be a JSON object ({\"field\": value, ...})")
        names = [f.name for f in dataclasses.fields(cls)]
        clean = {str(k): v for k, v in data.items() if not str(k).startswith("_")}
        unknown = sorted(set(clean) - set(names))
        if unknown:
            raise ValueError(f"unknown pullback spec field(s) {unknown}; valid names: {names}")
        return cls(**clean)

    def to_json(self, path=None, indent: int = 2) -> str:
        """The spec as JSON text (ASCII). With a path, also writes the file (locked paths are refused)."""
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=True) + "\n"
        if path is not None:
            bars_mod.check_not_locked(path, what="spec file", verb="write")
            Path(str(path)).expanduser().write_text(text, encoding="ascii")
        return text

    @classmethod
    def from_json(cls, source) -> "PullbackSpec":
        """Read a spec from JSON text (starting with '{') or from a .json file path. Locked paths are
        refused before opening; a missing file or bad JSON raises ValueError naming the problem."""
        if isinstance(source, Path) or (isinstance(source, str) and not source.lstrip().startswith("{")):
            bars_mod.check_not_locked(source, what="spec file")
            p = Path(str(source)).expanduser()
            if not p.is_file():
                raise ValueError(f"spec file not found: {p}")
            try:
                text = p.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError) as e:
                raise ValueError(f"cannot read spec file {p.name}: {e}")
            where = p.name
        elif isinstance(source, str):
            text, where = source, "spec text"
        else:
            raise ValueError("from_json takes JSON text or a path to a .json file")
        try:
            data = json.loads(text)
        except ValueError as e:
            raise ValueError(f"{where} is not valid JSON: {e}. Check commas and quotes (JSON has no comments; "
                             "use \"_comment\" keys)")
        return cls.from_dict(data)

    def summary_lines(self) -> list[str]:
        """A short ASCII description for reports (one 'field: value' per line, placeholder flagged)."""
        head = ["PLACEHOLDER SPEC - not zeno's rule" if self.placeholder else f"spec: {self.name}"]
        return head + [f"  {k}: {v}" for k, v in self.to_dict().items() if k not in ("name",)]


# ---------------------------------------------------------------------------------------
# helpers

def floor_to_lot_step(units, lot_step_oz: float) -> float:
    """Round a size DOWN to a whole number of lot steps, oz. units, lot_step_oz in oz (e.g. 1.0 oz =
    0.01 lot of 100 oz). A relative 1e-9 tolerance absorbs float noise (3 x 0.1 counts as 0.3).
    Returns 0.0 when the size is below one step."""
    step = _float(lot_step_oz, "lot_step_oz", 0.0, lo_open=True)
    u = _float(units, "units", 0.0)
    k = math.floor(u / step * (1.0 + 1e-12) + 1e-9)
    return round(k * step, 10)


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(x.size, np.nan)
    if k < x.size:
        out[k:] = x[:x.size - k]
    return out


def _rolling_any(flag: np.ndarray, m: int) -> np.ndarray:
    out = np.zeros(flag.size, dtype=bool)
    if flag.size >= m:
        out[m - 1:] = ind._windows(flag.astype(np.int64), m).max(axis=1) > 0
    return out


def _after_index_extreme(values: np.ndarray, idx: np.ndarray, w: int, kind: str) -> np.ndarray:
    """For each t: min (kind 'min') or max of values[idx[t]+1 .. t], NaN when idx[t] < 0 or idx[t] == t."""
    n = values.size
    out = np.full(n, np.nan)
    if n < w:
        return out
    win = ind._windows(values, w)                                # row r <-> t = r + w - 1
    t = np.arange(w - 1, n)
    rel = idx[w - 1:] - (t - w + 1)                             # column of the swing bar
    cols = np.arange(w)[None, :]
    fill = np.inf if kind == "min" else -np.inf
    masked = np.where(cols > rel[:, None], win, fill)
    ext = masked.min(axis=1) if kind == "min" else masked.max(axis=1)
    ok = (idx[w - 1:] >= 0) & np.isfinite(ext)
    out[w - 1:][ok] = ext[ok]
    return out


def _check_cost_model(cost_model) -> CostModel:
    if not isinstance(cost_model, CostModel):
        raise ValueError("cost_model must be a propkit.costs.CostModel")
    return cost_model


# ---------------------------------------------------------------------------------------
# signals

def compute_signals(bars: pd.DataFrame, spec: PullbackSpec) -> pd.DataFrame:
    """Per-bar indicators and the raw pullback signals at each bar's CLOSE (before the entry-time filters).

    bars: BARS (validated here; BID prices, USD/oz). Returns a DataFrame, one row per bar:
      time; ema_trend, ema_pullback, ema_trail, atr (USD/oz, NaN when unused or in warm-up);
      long_signal, short_signal (bool: trend + pullback + trigger + warm-up + direction all hold);
      long_extreme, short_extreme (the pullback low / high used by the swing stop, USD/oz, NaN if none);
      long_depth_atr, short_depth_atr (atr_from_swing depth in ATR, NaN otherwise).
    Every value in row t uses bars 0..t only.
    """
    if not isinstance(spec, PullbackSpec):
        raise ValueError("spec must be a propkit.pullback.PullbackSpec (PullbackSpec.from_json reads a file)")
    b = bars_mod.validate_bars(bars, source="bars")
    return _signals(b, spec)


def _signals(b: pd.DataFrame, spec: PullbackSpec) -> pd.DataFrame:
    o, h, lo, c = (b[col].to_numpy(dtype=np.float64) for col in ("open", "high", "low", "close"))
    n = c.size
    a = ind.atr(h, lo, c, spec.atr_period)
    nan = np.full(n, np.nan)
    e_tr = ind.ema(c, spec.trend_ema_period) if spec.uses_trend_ema else nan
    e_pb = ind.ema(c, spec.ema_pullback_period) if spec.uses_pullback_ema else nan
    e_trail = ind.ema(c, spec.trail_ema_period) if spec.exit_mode == "trail_ema" else nan
    idx = np.arange(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = (idx >= spec.warmup()) & np.isfinite(a) & (a > 0)
        up = np.ones(n, dtype=bool)
        dn = np.ones(n, dtype=bool)
        if spec.trend_require_close_side:
            up &= c > e_tr
            dn &= c < e_tr
        if spec.trend_slope_bars > 0:
            prev = _shift(e_tr, spec.trend_slope_bars)
            up &= e_tr > prev
            dn &= e_tr < prev
        depth_l = nan.copy()
        depth_s = nan.copy()
        if spec.pullback_mode == "ema_touch":
            m = spec.pullback_lookback_bars
            pb_l = _rolling_any(lo <= e_pb, m)
            pb_s = _rolling_any(h >= e_pb, m)
            ext_l = ind.swing_low(lo, m)
            ext_s = ind.swing_high(h, m)
        else:
            w = spec.swing_lookback_bars
            jh = ind.swing_high_index(h, w)
            jl = ind.swing_low_index(lo, w)
            ext_l = _after_index_extreme(lo, jh, w, "min")
            ext_s = _after_index_extreme(h, jl, w, "max")
            depth_l = (h[np.maximum(jh, 0)] - ext_l) / a
            depth_s = (ext_s - lo[np.maximum(jl, 0)]) / a
            hi_l = depth_l <= spec.max_depth_atr if spec.max_depth_atr is not None else np.isfinite(depth_l)
            hi_s = depth_s <= spec.max_depth_atr if spec.max_depth_atr is not None else np.isfinite(depth_s)
            pb_l = np.isfinite(depth_l) & (depth_l >= spec.min_depth_atr) & hi_l
            pb_s = np.isfinite(depth_s) & (depth_s >= spec.min_depth_atr) & hi_s
        if spec.trigger == "close_back_over_ema":
            pc, pe = _shift(c, 1), _shift(e_pb, 1)
            tr_l = (c > e_pb) & ((lo <= e_pb) | (pc <= pe))
            tr_s = (c < e_pb) & ((h >= e_pb) | (pc >= pe))
        else:
            tr_l = c > _shift(h, 1)
            tr_s = c < _shift(lo, 1)
        long_sig = valid & up & pb_l & tr_l & np.isfinite(ext_l) & (spec.direction != "short")
        short_sig = valid & dn & pb_s & tr_s & np.isfinite(ext_s) & (spec.direction != "long")
    return pd.DataFrame({
        "time": b["time"].to_numpy(dtype=np.int64),
        "ema_trend": e_tr, "ema_pullback": e_pb, "ema_trail": e_trail, "atr": a,
        "long_signal": long_sig, "short_signal": short_sig,
        "long_extreme": ext_l, "short_extreme": ext_s,
        "long_depth_atr": depth_l, "short_depth_atr": depth_s,
    })


def _session_ok(times: np.ndarray, spec: PullbackSpec) -> np.ndarray:
    if not spec.sessions and not spec.session_hours_utc:
        return np.ones(times.size, dtype=bool)
    ok = np.zeros(times.size, dtype=bool)
    for name in spec.sessions:
        ok |= np.asarray(calendar.session_mask(times, name), dtype=bool)
    hours = (times % calendar.SECONDS_PER_DAY) / 3600.0
    for a, b in spec.session_hours_utc:
        ok |= ((hours >= a) & (hours < b)) if a < b else ((hours >= a) | (hours < b))
    return ok


def _news_overlap(times: np.ndarray, bar_seconds: int, windows: tuple) -> np.ndarray:
    """True where the bar [open, open + bar_seconds) overlaps a [start, end) window."""
    out = np.zeros(times.size, dtype=bool)
    for a, b in windows:
        out |= (times < b) & (times + bar_seconds > a)
    return out


# ---------------------------------------------------------------------------------------
# the generator

class _Position:
    __slots__ = ("trade_id", "side", "units", "entry_bar", "entry_time", "entry_price", "stop", "target",
                 "risk_usd", "scheduled")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Ledger:
    """Closed-trade bookkeeping: balance, and per prop day the day-start balance, realised PnL and losses."""

    def __init__(self, C0: float):
        self.C0 = C0
        self.balance = C0
        self.exit_times: list[int] = []
        self.cum_pnl: list[float] = []
        self.cum_losses: list[int] = []

    def book(self, exit_time: int, pnl: float) -> None:
        self.balance += pnl
        self.exit_times.append(exit_time)
        self.cum_pnl.append((self.cum_pnl[-1] if self.cum_pnl else 0.0) + pnl)
        self.cum_losses.append((self.cum_losses[-1] if self.cum_losses else 0) + (1 if pnl < 0 else 0))

    def day_state(self, day_start: int) -> tuple[float, float, int]:
        """(day-start closed balance, realised PnL since day_start, losing trades closed since day_start)."""
        i = bisect.bisect_left(self.exit_times, day_start)
        pnl_before = self.cum_pnl[i - 1] if i > 0 else 0.0
        losses_before = self.cum_losses[i - 1] if i > 0 else 0
        total_losses = self.cum_losses[-1] if self.cum_losses else 0
        start_balance = self.C0 + pnl_before
        return start_balance, self.balance - start_balance, total_losses - losses_before


def generate_trades(bars: pd.DataFrame, spec: PullbackSpec, C0: float, cost_model: CostModel,
                    lot_step_oz: float = 1.0) -> pd.DataFrame:
    """Run the pullback rule over BARS and return TRADES (see the module docstring for every convention).

    bars: BARS (BID prices, USD/oz; optional spread in USD/oz); spec: PullbackSpec; C0: starting closed
    balance, USD (risk_pct is taken of the closed balance); cost_model: CostModel (fills, commission,
    swap, effective spreads); lot_step_oz: the broker's size step in oz ([ASSUMPTION] 1.0 oz = 0.01 lot of
    100 oz). TRADES columns (in order): trade_id (1.. in entry order), side (+1/-1), units (oz), entry_time,
    entry_price (fill incl. spread/markup/slippage), exit_time, exit_price, exit_reason ("stop", "target",
    "trail", "time", "signal" = news flatten, "end_of_data"), stop_price (the initial stop LEVEL: bid level
    for longs, ask level for shorts), risk_usd (1R incl. exit costs and commission), commission_usd
    (positive = paid), swap_usd (negative = paid), pnl_usd (net). Raises ValueError on invalid input.
    """
    trades, _ = generate_trades_detailed(bars, spec, C0, cost_model, lot_step_oz)
    return trades


def generate_trades_detailed(bars: pd.DataFrame, spec: PullbackSpec, C0: float, cost_model: CostModel,
                             lot_step_oz: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """generate_trades plus a decision log: (TRADES, DECISIONS).

    DECISIONS has one row per raw signal (columns DECISION_COLUMNS): signal_time (bar open of the signal
    bar, UTC epoch s), signal_bar (index), side, entry_time (the planned fill instant, the next bar's open;
    -1 if none), status ("entered" or a skip reason, see DECISION_STATUS), trade_id (-1 if not entered).
    """
    if not isinstance(spec, PullbackSpec):
        raise ValueError("spec must be a propkit.pullback.PullbackSpec (PullbackSpec.from_json reads a file)")
    cm = _check_cost_model(cost_model)
    c0 = _float(C0, "C0 (starting balance, USD)", 0.0, lo_open=True)
    step = _float(lot_step_oz, "lot_step_oz", 0.0, lo_open=True)
    b = bars_mod.validate_bars(bars, source="bars")
    times = b["time"].to_numpy(dtype=np.int64)
    bs = bars_mod.infer_bar_seconds(times)
    if spec.bar_minutes is not None and bs != spec.bar_minutes * 60:
        raise ValueError(f"the spec is written for {spec.bar_minutes}-minute bars but the data has "
                         f"{bs // 60}-minute bars ({bs} s); use the matching bar file or change bar_minutes")
    sig = _signals(b, spec)
    gen = _Generator(b, times, bs, spec, cm, c0, step, sig)
    return gen.run()


class _Generator:
    """One pass over the bars (see the module docstring for the order of events inside a bar)."""

    def __init__(self, b, times, bs, spec, cm, c0, step, sig):
        self.spec, self.cm, self.step, self.bs = spec, cm, step, bs
        self.t = times
        self.o, self.h, self.lo, self.c = (b[col].to_numpy(dtype=np.float64)
                                           for col in ("open", "high", "low", "close"))
        self.sp = cm.bar_spreads(b)
        self.n = times.size
        self.atr = sig["atr"].to_numpy()
        self.ema_trail = sig["ema_trail"].to_numpy()
        self.long_sig = sig["long_signal"].to_numpy(dtype=bool)
        self.short_sig = sig["short_signal"].to_numpy(dtype=bool)
        self.ext = {1: sig["long_extreme"].to_numpy(), -1: sig["short_extreme"].to_numpy()}
        nxt = np.concatenate((times[1:], [times[-1] + bs]))
        self.stamp = np.minimum(times + bs, nxt) - 1                    # intrabar exit instant
        self.pday = np.asarray(calendar.prop_day(times), dtype=np.int64)
        self.day_start = np.asarray(calendar.day_start_utc(self.pday), dtype=np.int64)
        self.session_ok = _session_ok(times, spec)
        self.news = _news_overlap(times, bs, spec.news_blackouts_utc)
        self.adj = 0.0 if cm.is_flat else cm.markup_per_side + cm.slippage_per_side
        self.ledger = _Ledger(c0)
        self.open: list[_Position] = []
        self.rows: dict[int, dict] = {}
        self.decisions: list[tuple] = []
        self.entries_per_day: dict[int, int] = {}
        self.next_id = 1

    # ----- fills ------------------------------------------------------------------------

    def _market_exit_fill(self, side: int, k: int, bid: float) -> float:
        return float(self.cm.sell_fill(bid, self.sp[k]) if side > 0 else self.cm.buy_fill(bid, self.sp[k]))

    def _level_fill(self, side: int, level: float) -> float:
        """Fill of an exit at a price LEVEL (bid level for longs, ask level for shorts), USD/oz."""
        fill = level - self.adj if side > 0 else level + self.adj
        if fill <= 0:
            raise ValueError(f"an exit fill at level {level:g} would be <= 0 after markup and slippage")
        return fill

    def _close(self, p: _Position, time: int, price: float, reason: str) -> None:
        cm = self.cm
        gross = p.side * p.units * (price - p.entry_price)
        comm = float(cm.fill_commission(p.units, p.entry_price)) + float(cm.fill_commission(p.units, price))
        inst, _ = cm.held_rollovers(p.entry_time, time)
        swap = 0.0
        if inst.size:
            wd = np.asarray(calendar.ny_weekday(inst), dtype=np.int64)
            bid = bars_mod.rollover_bids(self.t, self.o, self.c, inst)    # known at each rollover
            swap = float(np.sum(cm.swap_for_night(np.full(inst.size, p.side), np.full(inst.size, p.units),
                                                  bid, wd)))
        pnl = gross - comm + swap
        self.ledger.book(int(time), pnl)
        self.rows[p.trade_id] = {
            "trade_id": p.trade_id, "side": p.side, "units": p.units, "entry_time": p.entry_time,
            "entry_price": p.entry_price, "exit_time": int(time), "exit_price": float(price),
            "exit_reason": reason, "stop_price": p.stop, "risk_usd": p.risk_usd, "commission_usd": comm,
            "swap_usd": swap, "pnl_usd": pnl}

    # ----- the pass ---------------------------------------------------------------------

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        pending: list[tuple[int, int]] = []
        for k in range(self.n):
            self._exits_at_open(k)
            for side, s in pending:
                self._try_enter(side, s, k)
            pending = []
            self._intrabar(k)
            self._at_close(k)
            pending = self._new_signals(k)
        return self._trades_frame(), self._decisions_frame()

    def _exits_at_open(self, k: int) -> None:
        if not self.open:
            return
        keep = []
        o, sp = self.o[k], self.sp[k]
        for p in self.open:
            reason = None
            quote = o if p.side > 0 else o + sp                         # the side the exit trades at
            if p.side * (quote - p.stop) <= 0:
                reason = "stop"
            elif math.isfinite(p.target) and p.side * (quote - p.target) >= 0:
                reason = "target"
            elif p.scheduled is not None:
                reason = p.scheduled
            elif self.spec.news_flatten and self.news[k]:
                reason = "signal"
            if reason is None:
                keep.append(p)
            else:
                self._close(p, int(self.t[k]), self._market_exit_fill(p.side, k, o), reason)
        self.open = keep

    def _intrabar(self, k: int) -> None:
        if not self.open:
            return
        keep = []
        for p in self.open:
            if p.side > 0:
                stop_hit = self.lo[k] <= p.stop
                target_hit = math.isfinite(p.target) and self.h[k] >= p.target
            else:
                stop_hit = self.h[k] + self.sp[k] >= p.stop
                target_hit = math.isfinite(p.target) and self.lo[k] + self.sp[k] <= p.target
            if stop_hit:                                               # stop first when both are inside
                self._close(p, int(self.stamp[k]), self._level_fill(p.side, p.stop), "stop")
            elif target_hit:
                self._close(p, int(self.stamp[k]), self._level_fill(p.side, p.target), "target")
            else:
                keep.append(p)
        self.open = keep

    def _at_close(self, k: int) -> None:
        if not self.open:
            return
        if k == self.n - 1:
            for p in self.open:
                bid = self.c[k]
                self._close(p, int(self.t[k]) + self.bs, self._market_exit_fill(p.side, k, bid), "end_of_data")
            self.open = []
            return
        spec = self.spec
        for p in self.open:
            if spec.exit_mode == "trail_ema" and p.side * (self.c[k] - self.ema_trail[k]) < 0:
                p.scheduled = "trail"
            elif spec.exit_mode == "time" and k - p.entry_bar + 1 >= spec.time_exit_bars:
                p.scheduled = "time"

    def _new_signals(self, k: int) -> list[tuple[int, int]]:
        sides = [s for s, flag in ((1, self.long_sig[k]), (-1, self.short_sig[k])) if flag]
        if not sides:
            return []
        planned = int(self.t[k + 1]) if k + 1 < self.n else -1
        if len(sides) == 2:
            for s in sides:
                self._log(k, s, planned, "conflict")
            return []
        side = sides[0]
        if k == self.n - 1:
            self._log(k, side, -1, "no_next_bar")
            return []
        cap = self.spec.max_atr_usd
        if cap is not None and self.atr[k] > cap:
            self._log(k, side, planned, "atr_cap")
            return []
        return [(side, k)]

    def _log(self, s: int, side: int, entry_time: int, status: str, trade_id: int = -1) -> None:
        self.decisions.append((int(self.t[s]), int(s), int(side), int(entry_time), status, int(trade_id)))

    def _try_enter(self, side: int, s: int, k: int) -> None:
        spec, cm = self.spec, self.cm
        tk = int(self.t[k])
        if not spec.entry_after_gap and self.t[k] > self.t[s] + self.bs:
            return self._log(s, side, tk, "gap_before_entry")
        if not self.session_ok[k]:
            return self._log(s, side, tk, "outside_session")
        if self.news[k]:
            return self._log(s, side, tk, "news_blackout")
        if spec.max_spread_usd is not None and \
                self.sp[s if spec.spread_filter_bar == "signal" else k] > spec.max_spread_usd:
            return self._log(s, side, tk, "spread_cap")
        if len(self.open) >= spec.max_open_positions:
            return self._log(s, side, tk, "max_open_positions")
        day = int(self.pday[k])
        if spec.max_trades_per_day is not None and self.entries_per_day.get(day, 0) >= spec.max_trades_per_day:
            return self._log(s, side, tk, "max_trades_per_day")
        start_bal, realised, losses = self.ledger.day_state(int(self.day_start[k]))
        if spec.daily_stop_losses is not None and losses >= spec.daily_stop_losses:
            return self._log(s, side, tk, "daily_stop_losses")
        if spec.daily_stop_pct is not None and realised <= -spec.daily_stop_pct * start_bal:
            return self._log(s, side, tk, "daily_stop_pct")
        o, sp, a = self.o[k], self.sp[k], self.atr[s]
        if side > 0:
            entry = float(cm.buy_fill(o, sp))
            stop = self.ext[1][s] - spec.stop_buffer_atr * a if spec.stop_mode == "swing" \
                else o - spec.stop_atr_mult * a
            beyond = stop <= 0 or o <= stop
        else:
            entry = float(cm.sell_fill(o, sp))
            stop = self.ext[-1][s] + spec.stop_buffer_atr * a + sp if spec.stop_mode == "swing" \
                else o + sp + spec.stop_atr_mult * a
            beyond = o + sp >= stop
        if beyond:
            return self._log(s, side, tk, "open_beyond_stop")
        dist = abs(entry - stop)
        if spec.max_stop_usd is not None and dist > spec.max_stop_usd:
            return self._log(s, side, tk, "stop_cap")
        balance = self.ledger.balance
        if balance <= 0:
            return self._log(s, side, tk, "balance_not_positive")
        stop_fill = self._level_fill(side, stop)
        loss_per_oz = abs(entry - stop_fill) + float(cm.commission(1.0, entry, stop_fill))
        units = floor_to_lot_step(spec.risk_pct * balance / loss_per_oz, self.step)
        if units <= 0:
            return self._log(s, side, tk, "size_below_lot_step")
        target = entry + side * spec.target_r * dist if spec.exit_mode == "fixed_r" else math.nan
        risk_usd = units * abs(entry - stop_fill) + float(cm.commission(units, entry, stop_fill))
        tid = self.next_id
        self.next_id += 1
        self.open.append(_Position(trade_id=tid, side=side, units=units, entry_bar=k, entry_time=tk,
                                   entry_price=entry, stop=float(stop), target=float(target),
                                   risk_usd=float(risk_usd), scheduled=None))
        self.entries_per_day[day] = self.entries_per_day.get(day, 0) + 1
        self._log(s, side, tk, "entered", tid)

    # ----- output -----------------------------------------------------------------------

    def _trades_frame(self) -> pd.DataFrame:
        rows = [self.rows[i] for i in sorted(self.rows)]
        col = {name: [r[name] for r in rows] for name in TRADES_COLUMNS}
        return pd.DataFrame({
            "trade_id": np.asarray(col["trade_id"], dtype=np.int64),
            "side": np.asarray(col["side"], dtype=np.int64),
            "units": np.asarray(col["units"], dtype=np.float64),
            "entry_time": np.asarray(col["entry_time"], dtype=np.int64),
            "entry_price": np.asarray(col["entry_price"], dtype=np.float64),
            "exit_time": np.asarray(col["exit_time"], dtype=np.int64),
            "exit_price": np.asarray(col["exit_price"], dtype=np.float64),
            "exit_reason": pd.Series(col["exit_reason"], dtype=object),
            "stop_price": np.asarray(col["stop_price"], dtype=np.float64),
            "risk_usd": np.asarray(col["risk_usd"], dtype=np.float64),
            "commission_usd": np.asarray(col["commission_usd"], dtype=np.float64),
            "swap_usd": np.asarray(col["swap_usd"], dtype=np.float64),
            "pnl_usd": np.asarray(col["pnl_usd"], dtype=np.float64),
        })

    def _decisions_frame(self) -> pd.DataFrame:
        cols = list(zip(*self.decisions)) if self.decisions else [[] for _ in DECISION_COLUMNS]
        return pd.DataFrame({
            "signal_time": np.asarray(cols[0], dtype=np.int64),
            "signal_bar": np.asarray(cols[1], dtype=np.int64),
            "side": np.asarray(cols[2], dtype=np.int64),
            "entry_time": np.asarray(cols[3], dtype=np.int64),
            "status": pd.Series(list(cols[4]), dtype=object),
            "trade_id": np.asarray(cols[5], dtype=np.int64),
        })
