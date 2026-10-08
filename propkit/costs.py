"""propkit/costs.py - the cost model: spread, markup, slippage, commission, swap, or one flat rate.

Units: prices and spreads in USD per troy ounce (oz); sizes in oz ("units"); money in USD.
Sign conventions:
  * every cost PARAMETER is "positive = you pay" (spread, markup, slippage, commission, swap_long,
    swap_short, flat_rate_per_side); a negative swap parameter is a credit (the broker pays you);
  * every LEDGER amount a method returns follows TRADES / EQUITY: commission(...) and fill_commission(...)
    return the USD you pay as a positive number (TRADES.commission_usd, subtracted from PnL), while
    swap_for_night(...) returns the signed cash flow, negative = paid (TRADES.swap_usd, added to PnL).
Prices: bars carry BID prices. A buy fills at the ask plus costs, a sell at the bid minus costs:
    buy_fill  = bid + spread + markup_per_side + slippage_per_side
    sell_fill = bid          - markup_per_side - slippage_per_side
The spread to pass is the EFFECTIVE spread of the bar, from bar_spreads(bars) (it applies the spread
source, the spread multiplier and the flat-rate switch). A bar's spread is used for every fill and every
ask-side mark inside that bar (an approximation: real spreads move within the bar).

flat_rate_per_side (e.g. 0.0003, AlphaMaster's own cost): when set, it REPLACES spread, markup, slippage
and commission: fills happen at the bid itself (bar_spreads returns 0, so ask marks equal the bid) and
each fill pays flat_rate_per_side x fill notional (units x fill price), booked as commission. Swap still
applies unless swap_enabled is False (AlphaMaster has no swap; disable it to compare with the miner).

[ASSUMPTION] defaults, to be checked against the broker's contract specification before relying on any
result: fixed_spread 0.34 USD/oz (Dukascopy XAUUSD median; the p90 is 0.68 - use it for a stress run);
markup 0 and slippage 0 (the spread is taken to be the whole fill cost); commission 0 per lot round trip;
lot 100 oz; swaps 6.0 %/year long and 2.0 %/year short as costs on a 360-day year (placeholders); triple
swap on Wednesday (some brokers use Friday); rollover at 17:00 New York.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from propkit import calendar

SPREAD_SOURCES = ("bar", "fixed")
SWAP_UNITS = ("pct_per_year", "usd_per_lot_per_night")
DUKASCOPY_SPREAD_MEDIAN = 0.34   # USD/oz, XAUUSD
DUKASCOPY_SPREAD_P90 = 0.68      # USD/oz, XAUUSD

_FLOAT_FIELDS = ("fixed_spread", "spread_multiplier", "markup_per_side", "slippage_per_side",
                 "commission_per_lot_round_trip", "lot_size_oz", "swap_long", "swap_short",
                 "swap_day_count")


def _num(value, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{what} must be a number, got {value!r}")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    return x


def _arr(value, what: str, positive: bool = False, nonneg: bool = False) -> np.ndarray:
    raw = np.asarray(value) if not isinstance(value, (bool, np.bool_)) else np.asarray([True])
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{what} must be a number or an array of numbers, got {value!r}")
    a = raw.astype(np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f"{what} contains NaN or infinite values")
    if positive and (a <= 0).any():
        raise ValueError(f"{what} must be > 0")
    if nonneg and (a < 0).any():
        raise ValueError(f"{what} must be >= 0")
    return a


def _ret(a: np.ndarray, scalar: bool):
    return float(a) if scalar else a


def _is_scalar(*values) -> bool:
    return all(np.ndim(v) == 0 for v in values)


@dataclass(frozen=True)
class CostModel:
    """Trading costs for one instrument (XAUUSD CFD by default). Frozen: use dataclasses.replace or
    multiplied(k) for variants.

    Fields (units; positive = you pay unless noted):
      spread_source        "bar": use BARS.spread when the column exists (else fixed_spread);
                           "fixed": always fixed_spread.
      fixed_spread         USD/oz, ask - bid. [ASSUMPTION] 0.34 = Dukascopy XAUUSD median (p90 0.68).
      spread_multiplier    scales whichever spread is used (bar or fixed); 1.0 = as given.
      markup_per_side      USD/oz added against you on every fill (broker markup). [ASSUMPTION] 0.
      slippage_per_side    USD/oz added against you on every fill. [ASSUMPTION] 0.
      commission_per_lot_round_trip  USD per lot per round trip (half booked on each fill).
                           [ASSUMPTION] 0.
      lot_size_oz          oz per lot. 100 (check the broker spec).
      swap_long, swap_short  swap COST per night for a long / short position (negative = a credit),
                           in swap_unit. [ASSUMPTION placeholders] 6.0 and 2.0 %/year.
      swap_unit            "pct_per_year": percent per year of the notional (units x bid close at the
                           rollover), divided by swap_day_count per night; "usd_per_lot_per_night": USD
                           per lot per night (convert broker "points" to USD per lot yourself:
                           points x point size x lot size).
      swap_day_count       days per year for "pct_per_year" (360 default; some brokers use 365).
      triple_swap_weekday  New York weekday (Monday = 0) whose rollover is charged 3 nights for the
                           weekend; None = never. [ASSUMPTION] 2 = Wednesday (some brokers use Friday).
      rollover_hour_ny     rollover hour in New York local time (17 = 17:00 New York).
      swap_enabled         False switches swap off (e.g. to compare with AlphaMaster, which has none).
      flat_rate_per_side   None, or a fraction of the fill notional charged on every fill (0.0003 =
                           AlphaMaster); REPLACES spread, markup, slippage and commission when set.
    """

    spread_source: str = "bar"
    fixed_spread: float = DUKASCOPY_SPREAD_MEDIAN
    spread_multiplier: float = 1.0
    markup_per_side: float = 0.0
    slippage_per_side: float = 0.0
    commission_per_lot_round_trip: float = 0.0
    lot_size_oz: float = 100.0
    swap_long: float = 6.0
    swap_short: float = 2.0
    swap_unit: str = "pct_per_year"
    swap_day_count: float = 360.0
    triple_swap_weekday: int | None = 2
    rollover_hour_ny: int = 17
    swap_enabled: bool = True
    flat_rate_per_side: float | None = None

    def __post_init__(self) -> None:
        for name in _FLOAT_FIELDS:
            object.__setattr__(self, name, _num(getattr(self, name), name))
        if self.spread_source not in SPREAD_SOURCES:
            raise ValueError(f"spread_source must be one of {SPREAD_SOURCES}, got {self.spread_source!r}")
        if self.swap_unit not in SWAP_UNITS:
            raise ValueError(f"swap_unit must be one of {SWAP_UNITS}, got {self.swap_unit!r}")
        for name in ("fixed_spread", "spread_multiplier", "markup_per_side", "slippage_per_side",
                     "commission_per_lot_round_trip"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0 (it is a cost per side or per round trip), "
                                 f"got {getattr(self, name)}")
        if self.lot_size_oz <= 0:
            raise ValueError(f"lot_size_oz must be > 0, got {self.lot_size_oz}")
        if self.swap_day_count <= 0:
            raise ValueError(f"swap_day_count must be > 0, got {self.swap_day_count}")
        wd = self.triple_swap_weekday
        if wd is not None:
            if isinstance(wd, (bool, np.bool_)) or not isinstance(wd, (int, np.integer)) or not 0 <= wd <= 6:
                raise ValueError(f"triple_swap_weekday must be None or an integer 0..6 (Monday = 0), got {wd!r}")
            object.__setattr__(self, "triple_swap_weekday", int(wd))
        hour = self.rollover_hour_ny
        if (isinstance(hour, (bool, np.bool_)) or not isinstance(hour, (int, np.integer))
                or not (hour == 0 or 3 <= hour <= 23)):
            raise ValueError(f"rollover_hour_ny must be a whole hour 0 or 3..23 New York time, got {hour!r}")
        object.__setattr__(self, "rollover_hour_ny", int(hour))
        if not isinstance(self.swap_enabled, (bool, np.bool_)):
            raise ValueError(f"swap_enabled must be True or False, got {self.swap_enabled!r}")
        object.__setattr__(self, "swap_enabled", bool(self.swap_enabled))
        if self.flat_rate_per_side is not None:
            rate = _num(self.flat_rate_per_side, "flat_rate_per_side")
            if not 0 <= rate < 1:
                raise ValueError(f"flat_rate_per_side must be a fraction in [0, 1) such as 0.0003, got {rate}")
            object.__setattr__(self, "flat_rate_per_side", rate)

    # ----- spreads -------------------------------------------------------------------

    @property
    def is_flat(self) -> bool:
        """True when flat_rate_per_side is set (it replaces spread, markup, slippage and commission)."""
        return self.flat_rate_per_side is not None

    @property
    def effective_fixed_spread(self) -> float:
        """The fixed spread actually charged, USD/oz: fixed_spread x spread_multiplier (0 under a flat rate)."""
        return 0.0 if self.is_flat else self.fixed_spread * self.spread_multiplier

    def spread_source_used(self, bars) -> str:
        """Which spread bar_spreads(bars) uses: "bar", "fixed" or "none (flat rate)"."""
        if self.is_flat:
            return "none (flat rate)"
        has_col = hasattr(bars, "columns") and "spread" in bars.columns
        return "bar" if self.spread_source == "bar" and has_col else "fixed"

    def bar_spreads(self, bars) -> np.ndarray:
        """Effective spread of every bar, USD/oz (float64 array, one value per row of BARS).

        BARS.spread x spread_multiplier when spread_source is "bar" and the column exists; otherwise
        fixed_spread x spread_multiplier; all zeros under a flat rate. NaN or negative bar spreads raise
        ValueError. Use these values for fills (buy_fill/sell_fill) and for ask marks (ask = bid + spread).
        """
        if not hasattr(bars, "columns"):
            raise ValueError("bar_spreads needs a BARS DataFrame (columns time, open, high, low, close[, spread])")
        n = len(bars)
        source = self.spread_source_used(bars)
        if source != "bar":
            return np.full(n, self.effective_fixed_spread, dtype=np.float64)
        spread = np.asarray(bars["spread"], dtype=np.float64)
        if not np.isfinite(spread).all():
            raise ValueError("the bars' spread column has missing or infinite values; fill them, drop the "
                             "column (the fixed spread is then used) or use spread_source='fixed'")
        if (spread < 0).any():
            raise ValueError("the bars' spread column has negative values; spread = ask - bid must be >= 0")
        return spread * self.spread_multiplier

    # ----- fills and commission --------------------------------------------------------

    def buy_fill(self, bid, spread=None):
        """Fill price of a buy (long entry, short exit), USD/oz: bid + spread + markup + slippage.

        bid: bid price(s) at the fill instant (> 0); spread: the EFFECTIVE spread from bar_spreads (None
        = effective_fixed_spread). Under a flat rate the fill is the bid itself (the fee is charged by
        fill_commission). Scalars or numpy arrays (broadcast); returns float or array.
        """
        scalar = _is_scalar(bid, spread)
        b = _arr(bid, "bid", positive=True)
        if self.is_flat:
            return _ret(b + 0.0, scalar)
        s = _arr(self.effective_fixed_spread if spread is None else spread, "spread", nonneg=True)
        return _ret(b + s + self.markup_per_side + self.slippage_per_side, scalar)

    def sell_fill(self, bid, spread=None):
        """Fill price of a sell (short entry, long exit), USD/oz: bid - markup - slippage.

        The spread does not enter a sell (sells fill at the bid); the argument is accepted for symmetry
        with buy_fill and checked. Under a flat rate the fill is the bid itself. Raises ValueError if the
        fill would be <= 0. Scalars or numpy arrays; returns float or array.
        """
        scalar = _is_scalar(bid, spread)
        b = _arr(bid, "bid", positive=True)
        if spread is not None:
            _arr(spread, "spread", nonneg=True)
        if self.is_flat:
            return _ret(b + 0.0, scalar)
        fill = b - self.markup_per_side - self.slippage_per_side
        if (fill <= 0).any():
            raise ValueError("sell fill price would be <= 0 (markup + slippage >= bid); check the cost model")
        return _ret(fill, scalar)

    def fill_commission(self, units, price=None):
        """Commission for ONE fill, USD, positive = you pay.

        Per-lot model: commission_per_lot_round_trip / 2 x units / lot_size_oz (price not needed).
        Flat rate: flat_rate_per_side x units x price (price = the fill price, required).
        units: oz (>= 0); scalars or arrays; returns float or array.
        """
        scalar = _is_scalar(units, price)
        u = _arr(units, "units", nonneg=True)
        if self.is_flat:
            if price is None:
                raise ValueError("under a flat rate the commission is a fraction of the fill notional: pass price")
            p = _arr(price, "price", positive=True)
            return _ret(self.flat_rate_per_side * u * p, scalar)
        return _ret(0.5 * self.commission_per_lot_round_trip * u / self.lot_size_oz, scalar)

    def commission(self, units, entry_price=None, exit_price=None):
        """Round-trip commission, USD, positive = you pay (TRADES.commission_usd for a full round trip).

        Per-lot model: commission_per_lot_round_trip x units / lot_size_oz (prices not needed).
        Flat rate: flat_rate_per_side x units x (entry_price + exit_price) (both prices required).
        units: oz (>= 0); scalars or arrays; returns float or array.
        """
        if self.is_flat:
            if entry_price is None or exit_price is None:
                raise ValueError("under a flat rate the commission depends on the fill prices: pass "
                                 "entry_price and exit_price")
            return self.fill_commission(units, entry_price) + self.fill_commission(units, exit_price)
        scalar = _is_scalar(units)
        u = _arr(units, "units", nonneg=True)
        return _ret(self.commission_per_lot_round_trip * u / self.lot_size_oz, scalar)

    # ----- swap ------------------------------------------------------------------------

    def swap_nights(self, weekday):
        """Nights charged at a rollover on the given New York weekday (Monday = 0): 3 on
        triple_swap_weekday, else 1. Scalar or array; returns int or int64 array."""
        scalar = _is_scalar(weekday)
        w = np.asarray(weekday)
        if w.dtype.kind not in "iu" or (w < 0).any() or (w > 6).any():
            raise ValueError(f"weekday must be an integer 0..6 (Monday = 0), got {weekday!r}")
        nights = np.where(w == self.triple_swap_weekday, 3, 1) if self.triple_swap_weekday is not None \
            else np.ones_like(w)
        nights = nights.astype(np.int64)
        return int(nights) if scalar else nights

    def held_rollovers(self, entry_time, exit_time) -> tuple[np.ndarray, np.ndarray]:
        """Rollovers a position held from entry_time to exit_time pays swap for, and the nights each counts.

        A rollover instant r counts when entry_time < r <= exit_time (opened strictly before, closed at or
        after). Rollovers are at rollover_hour_ny New York time, Monday..Friday by New York weekday (no
        weekend rollovers; the weekend is in the triple day). entry_time, exit_time: UTC epoch seconds.
        Returns (instants: int64 UTC epoch seconds, nights: int64, 3 on triple_swap_weekday else 1).
        Ignores swap_enabled (swap_for_night returns 0 when swap is off).
        """
        a, a_scalar = calendar._as_seconds(entry_time, "entry_time")
        b, b_scalar = calendar._as_seconds(exit_time, "exit_time")
        if not (a_scalar and b_scalar):
            raise ValueError("entry_time and exit_time must be single instants (UTC epoch seconds)")
        if int(b) < int(a):
            raise ValueError(f"exit_time ({int(b)}) is before entry_time ({int(a)})")
        instants = calendar.rollover_instants(int(a) + 1, int(b) + 1, self.rollover_hour_ny)
        return instants, np.asarray(self.swap_nights(calendar.ny_weekday(instants)), dtype=np.int64)

    def swap_for_night(self, side, units, bid_close, weekday):
        """Swap cash flow for holding a position across ONE rollover, USD, NEGATIVE = paid.

        side: +1 long / -1 short; units: oz (>= 0); bid_close: bid close used for the notional (USD/oz,
        the close of the bar containing the rollover); weekday: New York weekday of the rollover
        (calendar.ny_weekday(instant)), x3 on triple_swap_weekday.
          pct_per_year:          -rate/100 x units x bid_close / swap_day_count x nights
          usd_per_lot_per_night: -rate x units / lot_size_oz x nights
        with rate = swap_long or swap_short (positive = cost, so the result is negative; a negative rate
        is a credit and gives a positive result). 0 when swap_enabled is False. Scalars or arrays
        (broadcast); returns float or array.
        """
        scalar = _is_scalar(side, units, bid_close, weekday)
        s = np.asarray(side)
        if s.dtype.kind not in "iuf" or not np.isin(s, (1, -1)).all():
            raise ValueError(f"side must be +1 (long) or -1 (short), got {side!r}")
        u = _arr(units, "units", nonneg=True)
        b = _arr(bid_close, "bid_close", positive=True)
        nights = np.asarray(self.swap_nights(weekday))
        if not self.swap_enabled:
            return _ret(np.zeros(np.broadcast(s, u, b, nights).shape), scalar)
        rate = np.where(s > 0, self.swap_long, self.swap_short)
        if self.swap_unit == "pct_per_year":
            cost = rate / 100.0 * u * b / self.swap_day_count * nights
        else:
            cost = rate * u / self.lot_size_oz * nights + 0.0 * b
        return _ret(-cost, scalar)

    # ----- variants and serialisation ---------------------------------------------------

    def multiplied(self, k: float) -> "CostModel":
        """A copy with every cost x k (k >= 0; 1.5 and 2 are the stress runs).

        Scaled: spread_multiplier (so bar and fixed spreads), markup, slippage, commission, the flat rate,
        and swap rates that are COSTS (> 0). Swap CREDITS (< 0) are left unchanged, so a stress never
        makes a credit larger. Lot size, day count, triple day and rollover hour are unchanged.
        """
        k = _num(k, "k")
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        return dataclasses.replace(
            self,
            spread_multiplier=self.spread_multiplier * k,
            markup_per_side=self.markup_per_side * k,
            slippage_per_side=self.slippage_per_side * k,
            commission_per_lot_round_trip=self.commission_per_lot_round_trip * k,
            swap_long=self.swap_long * k if self.swap_long > 0 else self.swap_long,
            swap_short=self.swap_short * k if self.swap_short > 0 else self.swap_short,
            flat_rate_per_side=None if self.flat_rate_per_side is None else self.flat_rate_per_side * k,
        )

    def to_dict(self) -> dict[str, Any]:
        """All fields as a JSON-serialisable dict (from_dict(to_dict()) gives an equal model)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CostModel":
        """Build a model from a dict (e.g. a costs JSON file); unknown keys raise ValueError, missing
        keys take the documented defaults."""
        if not isinstance(data, Mapping):
            raise ValueError("cost settings must be a JSON object (a dict of field: value)")
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ValueError(f"unknown cost setting(s) {unknown}; valid names: {sorted(names)}")
        return cls(**dict(data))
