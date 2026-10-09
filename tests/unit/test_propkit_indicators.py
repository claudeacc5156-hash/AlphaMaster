"""Tests for propkit/indicators.py: causal EMA, true range, Wilder ATR, swings, rolling sd, z-score.
Synthetic data only. Research only."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from propkit import indicators as ind

SRC = Path(__file__).resolve().parents[2] / "propkit" / "indicators.py"


def _walk(n: int = 600, seed: int = 3):
    rng = np.random.default_rng(seed)
    c = 2000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.003, n)))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0.0, 0.0005, n))
    h = np.maximum(o, c) + np.abs(rng.normal(0.0, 1.5, n))
    lo = np.minimum(o, c) - np.abs(rng.normal(0.0, 1.5, n))
    return o, h, lo, c


def _ema_loop(x, n):
    a = 2.0 / (n + 1.0)
    out = np.empty(len(x))
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = a * x[t] + (1.0 - a) * out[t - 1]
    return out


def _atr_loop(h, lo, c, n):
    tr = np.empty(len(c))
    tr[0] = h[0] - lo[0]
    for t in range(1, len(c)):
        tr[t] = max(h[t] - lo[t], abs(h[t] - c[t - 1]), abs(lo[t] - c[t - 1]))
    out = np.full(len(c), np.nan)
    out[n - 1] = tr[:n].mean()
    for t in range(n, len(c)):
        out[t] = (out[t - 1] * (n - 1) + tr[t]) / n
    return out


# ---------------------------------------------------------------- EMA

def test_ema_hand_example_seeded_with_first_value():
    # period 3 -> alpha = 0.5: 1, 0.5*2+0.5*1 = 1.5, 0.5*3+0.5*1.5 = 2.25, 0.5*3+0.5*2.25 = 2.625
    assert np.allclose(ind.ema([1.0, 2.0, 3.0, 3.0], 3), [1.0, 1.5, 2.25, 2.625], atol=1e-15)
    assert ind.ema_alpha(3) == 0.5 and ind.ema_alpha(19) == pytest.approx(0.1)


def test_ema_matches_plain_recursion_and_edge_cases():
    _, _, _, c = _walk()
    for n in (1, 2, 5, 20, 200):
        assert np.allclose(ind.ema(c, n), _ema_loop(c, n), rtol=0, atol=1e-9)
    assert np.array_equal(ind.ema(c, 1), c)
    assert np.allclose(ind.ema(np.full(50, 7.5), 10), 7.5)
    assert ind.ema(np.zeros(0), 5).size == 0
    out = ind.ema(pd.Series(c), 10)
    assert isinstance(out, np.ndarray) and out.dtype == np.float64


def test_ema_seed_weight_documented():
    # after 3 x period bars the seed weighs below 0.25%
    for n in (10, 20, 50, 200):
        assert ind.ema_seed_weight(n, 3 * n) < 0.0025
    assert ind.ema_seed_weight(3, 0) == 1.0


# ---------------------------------------------------------------- true range / ATR

def test_true_range_uses_previous_close():
    h = [10.0, 12.0, 11.0]
    lo = [9.0, 11.5, 10.5]
    c = [9.5, 11.8, 10.6]
    # bar 0: 10 - 9 = 1; bar 1: gap up: max(0.5, |12 - 9.5| = 2.5, |11.5 - 9.5| = 2) = 2.5;
    # bar 2: max(0.5, |11 - 11.8| = 0.8, |10.5 - 11.8| = 1.3) = 1.3
    assert np.allclose(ind.true_range(h, lo, c), [1.0, 2.5, 1.3])


def test_atr_wilder_hand_example():
    # constant true range 2 for 3 bars then a bar with TR 5: period 3
    h = [12.0, 12.0, 12.0, 15.0, 12.0]
    lo = [10.0, 10.0, 10.0, 10.0, 10.0]
    c = [11.0, 11.0, 11.0, 11.0, 11.0]
    out = ind.atr(h, lo, c, 3)
    assert np.isnan(out[:2]).all()
    assert out[2] == pytest.approx(2.0)                       # seed = mean(2, 2, 2)
    assert out[3] == pytest.approx((2.0 * 2 + 5.0) / 3)      # (atr * (n-1) + TR) / n = 3.0
    assert out[4] == pytest.approx((3.0 * 2 + 2.0) / 3)
    assert ind.wilder_alpha(14) == pytest.approx(1 / 14)


def test_atr_matches_plain_recursion():
    _, h, lo, c = _walk()
    for n in (1, 2, 14, 50):
        ref = _atr_loop(h, lo, c, n)
        out = ind.atr(h, lo, c, n)
        assert np.array_equal(np.isnan(out), np.isnan(ref))
        assert np.allclose(out[n - 1:], ref[n - 1:], rtol=0, atol=1e-9)
    assert np.isnan(ind.atr(h[:5], lo[:5], c[:5], 14)).all()


def test_atr_constant_range_is_that_range():
    n = 40
    o = 100.0 + np.arange(n)
    c = o + 1.0
    h, lo = c + 0.5, o - 0.5                 # high - low = 2, |high - prev close| = 1.5, |low - prev close| = 0.5
    assert np.allclose(ind.atr(h, lo, c, 14)[13:], 2.0)


# ---------------------------------------------------------------- swings

def test_swing_values_and_indices():
    h = np.array([1.0, 3.0, 2.0, 3.0, 1.0, 0.5])
    lo = np.array([0.5, 2.0, 1.0, 1.0, 0.2, 0.3])
    sh = ind.swing_high(h, 3)
    sl = ind.swing_low(lo, 3)
    assert np.isnan(sh[:2]).all() and np.isnan(sl[:2]).all()
    assert np.allclose(sh[2:], [3.0, 3.0, 3.0, 3.0])
    assert np.allclose(sl[2:], [0.5, 1.0, 0.2, 0.2])
    # ties go to the most recent bar: at t = 3 the window is bars 1..3, highs 3, 2, 3 -> bar 3
    assert list(ind.swing_high_index(h, 3)) == [-1, -1, 1, 3, 3, 3]
    # lows at t = 3: bars 1..3 = 2, 1, 1 -> the later 1 is bar 3
    assert list(ind.swing_low_index(lo, 3)) == [-1, -1, 0, 3, 4, 4]
    assert np.allclose(ind.swing_high(h, 1), h) and list(ind.swing_high_index(h, 1)) == list(range(6))


# ---------------------------------------------------------------- dispersion

def test_rolling_sd_is_sample_sd():
    x = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    out = ind.rolling_sd(x, 3)
    assert np.isnan(out[:2]).all()
    for t in range(2, 5):
        assert out[t] == pytest.approx(np.std(x[t - 2:t + 1], ddof=1))
    with pytest.raises(ValueError, match=">= 2"):
        ind.rolling_sd(x, 1)


def test_zscore_hand_example_and_flat_window():
    c = np.array([10.0, 10.0, 13.0])
    # ema period 1 = the close itself, so z = 0 wherever defined; sd of [10, 10] is 0 -> NaN
    z1 = ind.zscore_vs_ema(c, 1, 2)
    assert np.isnan(z1[:2]).all() and z1[2] == 0.0
    z = ind.zscore_vs_ema(c, 3, 3)
    e = _ema_loop(c, 3)                       # 10, 10, 11.5
    assert np.isnan(z[:2]).all()
    assert z[2] == pytest.approx((13.0 - e[2]) / np.std(c, ddof=1))
    flat = ind.zscore_vs_ema(np.full(10, 5.0), 3, 4)
    assert np.isnan(flat).all()               # sd = 0: undefined, NaN (documented)


# ---------------------------------------------------------------- causality

def _all_indicators(o, h, lo, c):
    return {
        "ema": ind.ema(c, 20), "ema1": ind.ema(c, 1), "tr": ind.true_range(h, lo, c), "atr": ind.atr(h, lo, c, 14),
        "sh": ind.swing_high(h, 10), "sl": ind.swing_low(lo, 10),
        "shi": ind.swing_high_index(h, 10).astype(float), "sli": ind.swing_low_index(lo, 10).astype(float),
        "sd": ind.rolling_sd(c, 20), "z": ind.zscore_vs_ema(c, 20, 30),
    }


def test_truncation_never_changes_past_values_many_cut_points():
    o, h, lo, c = _walk(700, seed=11)
    full = _all_indicators(o, h, lo, c)
    cuts = list(range(1, 40)) + list(range(40, 700, 7)) + [699, 700]
    for k in cuts:
        part = _all_indicators(o[:k], h[:k], lo[:k], c[:k])
        for name, values in part.items():
            assert np.array_equal(values, full[name][:k], equal_nan=True), (name, k)


def test_changing_future_bars_never_changes_past_values():
    o, h, lo, c = _walk(400, seed=5)
    base = _all_indicators(o, h, lo, c)
    rng = np.random.default_rng(1)
    for k in (30, 100, 250, 399):
        c2 = c.copy()
        h2, lo2 = h.copy(), lo.copy()
        c2[k:] *= np.exp(rng.normal(0, 0.05, c.size - k))
        h2[k:] = np.maximum(h2[k:], c2[k:]) + 3.0
        lo2[k:] = np.minimum(lo2[k:], c2[k:]) - 3.0
        moved = _all_indicators(o, h2, lo2, c2)
        for name in base:
            assert np.array_equal(moved[name][:k], base[name][:k], equal_nan=True), (name, k)


# ---------------------------------------------------------------- input checks

@pytest.mark.parametrize("bad", [[1.0, float("nan"), 2.0], [1.0, float("inf")], ["a", "b"], [True, False],
                                 np.ones((2, 2))])
def test_bad_values_refused(bad):
    with pytest.raises(ValueError):
        ind.ema(bad, 3)


@pytest.mark.parametrize("period", [0, -1, 2.5, True, "3", None])
def test_bad_periods_refused(period):
    with pytest.raises(ValueError):
        ind.ema([1.0, 2.0], period)
    with pytest.raises(ValueError):
        ind.atr([2.0, 2.0], [1.0, 1.0], [1.5, 1.5], period)


def test_length_mismatch_and_high_below_low_refused():
    with pytest.raises(ValueError, match="same length"):
        ind.atr([2.0, 2.0], [1.0], [1.5, 1.5], 2)
    with pytest.raises(ValueError, match="below low"):
        ind.true_range([1.0, 2.0], [1.5, 1.0], [1.2, 1.5])


def test_source_is_ascii_and_imports_nothing_from_alphamaster():
    text = SRC.read_text(encoding="utf-8")
    assert text.isascii()
    forbidden = r"^\s*(from|import)\s+(model_core|data_pipeline|config|web|utils|strategy_manager|execution|scripts)\b"
    assert not re.search(forbidden, text, flags=re.M)
    assert "zoneinfo" not in text and "scipy" not in text
    for name in ("ema", "true_range", "atr", "swing_high", "swing_low", "swing_high_index", "swing_low_index",
                 "rolling_sd", "zscore_vs_ema"):
        assert getattr(ind, name).__doc__, name
    assert math.isclose(ind.ema_seed_weight(1, 5), 0.0)
