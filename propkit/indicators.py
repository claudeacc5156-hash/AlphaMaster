"""propkit/indicators.py - causal indicators for the pullback generator: EMA, true range, Wilder ATR,
swing highs/lows, rolling standard deviation and the z-score of the close against its EMA.

Causality: every value at bar t is computed from bars 0..t only (the bar t itself is closed when the
value is read, at bar t's close). Nothing looks at bar t+1 or later, so cutting the series at any bar k
gives exactly the first k values of the full series (the tests check this over many cut points).

Units: the outputs are in the units of the inputs (USD per oz for BID prices), except the z-score
(dimensionless) and the swing indices (bar positions). Every function takes 1-D array-likes of finite
numbers (a pandas Series or a numpy array) and returns a float64 numpy array of the same length (int64
for the swing indices). Warm-up values that cannot be computed yet are NaN (or -1 for an index); the
warm-up length is stated in each docstring. Invalid input (NaN, infinite values, mismatched lengths,
a period < 1) raises ValueError.

Research only: nothing here places or prepares orders.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------------------
# input checks

def _series(values, what: str) -> np.ndarray:
    """A 1-D float64 copy of the input, refusing NaN, infinite, non-numeric and boolean values."""
    if isinstance(values, (pd.Series, pd.Index)):
        values = values.to_numpy()
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{what} must be a 1-D series of numbers, got an array of shape {arr.shape}")
    if arr.dtype.kind == "b" or arr.dtype.kind not in "iuf":
        raise ValueError(f"{what} must hold numbers, got values of type {arr.dtype}")
    out = arr.astype(np.float64)
    if not np.isfinite(out).all():
        i = int(np.flatnonzero(~np.isfinite(out))[0])
        raise ValueError(f"{what} has a missing or infinite value at position {i}; clean the bars first")
    return out


def _period(n, what: str = "period") -> int:
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError(f"{what} must be a whole number of bars >= 1, got {n!r}")
    if int(n) < 1:
        raise ValueError(f"{what} must be a whole number of bars >= 1, got {n}")
    return int(n)


def _same_length(*arrays: np.ndarray) -> None:
    lengths = {a.size for a in arrays}
    if len(lengths) > 1:
        raise ValueError(f"high, low and close must have the same length, got lengths {sorted(lengths)}")


# ---------------------------------------------------------------------------------------
# moving averages and ranges

def ema(values, period: int) -> np.ndarray:
    """Exponential moving average, causal, in the units of `values` (e.g. USD/oz of BID closes).

    alpha = 2 / (period + 1); seeded with the FIRST value: ema[0] = x[0], then
    ema[t] = alpha * x[t] + (1 - alpha) * ema[t-1]. There is no NaN warm-up, but the seed's weight
    (1 - alpha)^t fades slowly: after 3 x period bars it is below 0.25%, which is why the pullback
    generator ignores signals before its warm-up (PullbackSpec.warmup). period 1 returns the input.
    (Same recursion as pandas ewm(span=period, adjust=False).mean(), which computes it.)
    """
    x = _series(values, "values")
    n = _period(period)
    if x.size == 0:
        return x
    alpha = 2.0 / (n + 1.0)
    return pd.Series(x).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=np.float64)


def true_range(high, low, close) -> np.ndarray:
    """True range per bar, USD/oz: max(high - low, |high - close[t-1]|, |low - close[t-1]|).

    Uses the PREVIOUS bar's close (so an opening gap counts); the first bar has no previous close and
    its true range is high[0] - low[0]. Prices are BID; high >= low is expected (validated BARS).
    """
    h = _series(high, "high")
    lo = _series(low, "low")
    c = _series(close, "close")
    _same_length(h, lo, c)
    if h.size == 0:
        return h
    if (h < lo).any():
        i = int(np.flatnonzero(h < lo)[0])
        raise ValueError(f"high is below low at position {i}; validate the bars first")
    prev = np.concatenate(([np.nan], c[:-1]))
    tr = h - lo
    tr[1:] = np.maximum(tr[1:], np.maximum(np.abs(h[1:] - prev[1:]), np.abs(lo[1:] - prev[1:])))
    return tr


def atr(high, low, close, period: int = 14) -> np.ndarray:
    """Average true range, Wilder's smoothing, USD/oz.

    TR = true_range(high, low, close) (previous close, see there). Seed: atr[period-1] = mean of the
    first `period` true ranges; then atr[t] = (atr[t-1] * (period - 1) + TR[t]) / period, i.e. an EMA
    with alpha = 1/period. Warm-up: atr[0 .. period-2] = NaN. Causal: atr[t] uses bars 0..t.
    """
    n = _period(period)
    tr = true_range(high, low, close)
    out = np.full(tr.size, np.nan)
    if tr.size < n:
        return out
    seeded = tr[n - 1:].copy()
    seeded[0] = tr[:n].mean()
    out[n - 1:] = pd.Series(seeded).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy(dtype=np.float64)
    return out


# ---------------------------------------------------------------------------------------
# swings

def _windows(x: np.ndarray, k: int) -> np.ndarray:
    return sliding_window_view(x, k)


def swing_high(high, k: int) -> np.ndarray:
    """Highest high of the last k bars ENDING at t (bars t-k+1 .. t, all closed at bar t's close), USD/oz.

    Warm-up: the first k-1 values are NaN.
    """
    h = _series(high, "high")
    k = _period(k, "k")
    out = np.full(h.size, np.nan)
    if h.size >= k:
        out[k - 1:] = _windows(h, k).max(axis=1)
    return out


def swing_low(low, k: int) -> np.ndarray:
    """Lowest low of the last k bars ENDING at t (bars t-k+1 .. t), USD/oz. Warm-up: first k-1 NaN."""
    lo = _series(low, "low")
    k = _period(k, "k")
    out = np.full(lo.size, np.nan)
    if lo.size >= k:
        out[k - 1:] = _windows(lo, k).min(axis=1)
    return out


def swing_high_index(high, k: int) -> np.ndarray:
    """Bar index of the swing_high(high, k) at each t: the MOST RECENT bar among t-k+1 .. t with the
    highest high (ties go to the later bar). int64; -1 during the warm-up (t < k-1)."""
    h = _series(high, "high")
    k = _period(k, "k")
    out = np.full(h.size, -1, dtype=np.int64)
    if h.size >= k:
        w = _windows(h, k)[:, ::-1]                       # newest first, so argmax finds the latest tie
        out[k - 1:] = np.arange(k - 1, h.size) - np.argmax(w, axis=1)
    return out


def swing_low_index(low, k: int) -> np.ndarray:
    """Bar index of the swing_low(low, k) at each t: the most recent bar among t-k+1 .. t with the lowest
    low (ties go to the later bar). int64; -1 during the warm-up (t < k-1)."""
    lo = _series(low, "low")
    k = _period(k, "k")
    out = np.full(lo.size, -1, dtype=np.int64)
    if lo.size >= k:
        w = _windows(lo, k)[:, ::-1]
        out[k - 1:] = np.arange(k - 1, lo.size) - np.argmin(w, axis=1)
    return out


# ---------------------------------------------------------------------------------------
# dispersion

def rolling_sd(values, k: int) -> np.ndarray:
    """Rolling SAMPLE standard deviation (ddof = 1) of the last k values ending at t, in the units of the
    input. k must be >= 2. Warm-up: the first k-1 values are NaN. Computed per window (two-pass), so
    there is no running-sum drift and a cut series gives exactly the same values."""
    x = _series(values, "values")
    k = _period(k, "k")
    if k < 2:
        raise ValueError("k must be >= 2 for a sample standard deviation (ddof = 1)")
    out = np.full(x.size, np.nan)
    if x.size >= k:
        out[k - 1:] = _windows(x, k).std(axis=1, ddof=1)
    return out


def zscore_vs_ema(close, ema_period: int, sd_period: int) -> np.ndarray:
    """z-score of the close against its EMA, dimensionless:

        z[t] = (close[t] - ema(close, ema_period)[t]) / rolling_sd(close, sd_period)[t]

    rolling_sd is the sample sd (ddof = 1) of the last sd_period CLOSES ending at t (the Bollinger-band
    convention: dispersion of price, not of the deviation). Warm-up: NaN for t < sd_period - 1. Where
    the rolling sd is exactly 0 (a flat window) z is undefined and NaN is returned (stated, not hidden).
    """
    c = _series(close, "close")
    dev = c - ema(c, ema_period)
    sd = rolling_sd(c, sd_period)
    out = np.full(c.size, np.nan)
    ok = np.isfinite(sd) & (sd > 0)
    out[ok] = dev[ok] / sd[ok]
    return out


def wilder_alpha(period: int) -> float:
    """Wilder's smoothing constant 1/period (the ATR recursion's alpha), for documentation and tests."""
    return 1.0 / _period(period)


def ema_alpha(period: int) -> float:
    """The EMA smoothing constant 2/(period + 1) used by ema()."""
    n = _period(period)
    return 2.0 / (n + 1.0)


def ema_seed_weight(period: int, bars: int) -> float:
    """Weight left on the EMA's seed (the first value) after `bars` further bars: (1 - alpha)^bars."""
    n = _period(period)
    if isinstance(bars, (bool, np.bool_)) or not isinstance(bars, (int, np.integer)) or bars < 0:
        raise ValueError(f"bars must be a whole number >= 0, got {bars!r}")
    return math.pow(1.0 - ema_alpha(n), int(bars))
