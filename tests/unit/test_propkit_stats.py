"""Tests for propkit/stats.py: Sharpe SE (Lo/Mertens), PSR, expected max SR, DSR (Bailey and Lopez de Prado
2014 gate), MinTRL, effective trials, prop-day returns, expected shortfall and drawdowns. Research only."""
from __future__ import annotations

import datetime as dt
import json
import math
import pathlib
import re
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from propkit import calendar
from propkit import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
PHI = NormalDist().cdf
PHI_INV = NormalDist().inv_cdf
C0 = 100_000.0


def _utc(*args) -> int:
    return int(dt.datetime(*args, tzinfo=dt.timezone.utc).timestamp())


def _day(y, m, d) -> int:
    return (dt.date(y, m, d) - dt.date(1970, 1, 1)).days


# ---------------------------------------------------------------------------------------
# gate: Bailey and Lopez de Prado (2014), "The Deflated Sharpe Ratio", numerical example

def test_gate_bailey_lopez_de_prado_2014_dsr_example():
    # Inputs: SR = 2.5 annualised, T = 1250 daily returns, N = 100 trials, V = 0.5 (annualised variance of the
    # trials' SRs), skew = -3, kurtosis = 10 (Pearson), 250 days per year. Everything goes per period first:
    #   SR per day       = 2.5 / sqrt(250)                       = 0.158113883
    #   V per day        = 0.5 / 250 = 0.002, sqrt(V)            = 0.044721360
    #   Phi^-1(1 - 1/N)          = Phi^-1(0.99)                  = 2.326347874
    #   Phi^-1(1 - 1/(N e))      = Phi^-1(0.996321206)           = 2.680210445
    #   (1 - g) * 2.326347874 + g * 2.680210445 (g = 0.5772156649) = 2.530602893
    #   SR0 = 0.044721360 * 2.530602893                          = 0.113172002  -> 0.1132
    #   variance term 1 - (-3)(0.158113883) + (10 - 1)/4 * 0.158113883^2 = 1 + 0.474342 + 0.05625 = 1.530591649
    #   z = (0.158113883 - 0.113172002) * sqrt(1249) / sqrt(1.530591649) = 0.044941881 * 35.341194 / 1.237171
    #     = 1.283816037
    #   DSR = Phi(1.283816037)                                   = 0.900396834  -> 0.9004
    q = 250
    sr = 2.5 / math.sqrt(q)
    v = 0.5 / q
    sr0 = stats.expected_max_sr(100, v)
    assert round(sr0, 4) == 0.1132
    value = stats.dsr(sr, 1250, -3.0, 10.0, 100, v)
    assert round(value, 4) == 0.9004
    d = stats.dsr_details(sr, 1250, -3.0, 10.0, 100, v)
    assert d["sr"] == pytest.approx(0.158113883, abs=1e-9)
    assert d["sr0"] == pytest.approx(0.113172002, abs=1e-9)
    assert d["variance_term"] == pytest.approx(1.530591649, abs=1e-9)
    assert d["z"] == pytest.approx(1.283816037, abs=1e-9)
    assert d["dsr"] == pytest.approx(0.900396834, abs=1e-9) == value
    assert d["n_trials"] == 100 and d["n"] == 1250
    # the DSR is the PSR against SR0
    assert stats.psr(sr, sr0, 1250, -3.0, 10.0) == pytest.approx(value, abs=1e-15)


# ---------------------------------------------------------------------------------------
# sharpe_stats

def test_sharpe_stats_hand_example():
    # x = 1, 2, 3, 4, 10: mean 4, deviations -3, -2, -1, 0, 6, sum of squares 50.
    # sd (ddof 1) = sqrt(50/4) = sqrt(12.5); SR = 4 / sqrt(12.5) = 0.8 sqrt(2) = 1.131371
    # m2 = 50/5 = 10, m3 = (-27 - 8 - 1 + 0 + 216)/5 = 36, m4 = (81 + 16 + 1 + 0 + 1296)/5 = 278.8
    # skew = 36 / 10^1.5 = 3.6 / sqrt(10) = 1.138420; kurt = 278.8 / 100 = 2.788 (Pearson)
    # SE = sqrt((1 - skew SR + (kurt - 1)/4 SR^2) / (n - 1)); SR^2 = 1.28
    s = stats.sharpe_stats([1.0, 2.0, 3.0, 4.0, 10.0], 252)
    sr = 0.8 * math.sqrt(2.0)
    skew = 3.6 / math.sqrt(10.0)
    term = 1.0 - skew * sr + (2.788 - 1.0) / 4.0 * 1.28
    assert s.n == 5 and s.mean == 4.0
    assert s.sd == pytest.approx(math.sqrt(12.5), rel=1e-14)
    assert s.sr == pytest.approx(sr, rel=1e-14)
    assert s.skew == pytest.approx(skew, rel=1e-13)
    assert s.kurt == pytest.approx(2.788, rel=1e-13)
    assert term == pytest.approx(0.284185, abs=1e-6)          # 1.57216 - 2.88 sqrt(0.2)
    assert s.se_sr == pytest.approx(math.sqrt(term / 4.0), rel=1e-12)
    assert s.periods_per_year == 252.0
    assert s.sr_annual == pytest.approx(sr * math.sqrt(252), rel=1e-14)
    assert s.se_sr_annual == pytest.approx(s.se_sr * math.sqrt(252), rel=1e-14)
    assert s.psr_0 == pytest.approx(stats.psr(s.sr, 0.0, 5, s.skew, s.kurt), abs=1e-15)


def test_sharpe_stats_normal_case_reduces_to_lo():
    # for skew 0 and kurt 3 the Mertens variance is Lo's IID-normal 1 + SR^2/2
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])          # symmetric around 0.5: skew exactly 0
    s = stats.sharpe_stats(x, None)
    assert abs(s.skew) < 1e-15
    lo = math.sqrt((1 - s.skew * s.sr + (s.kurt - 1) / 4 * s.sr ** 2) / 5)
    assert s.se_sr == pytest.approx(lo, rel=1e-14)
    assert s.sr_annual is None and s.se_sr_annual is None and s.periods_per_year is None


def test_sharpe_stats_accepts_series_and_is_json():
    rng = np.random.default_rng(11)
    x = rng.normal(0.001, 0.01, 300)
    a = stats.sharpe_stats(x, 252)
    b = stats.sharpe_stats(pd.Series(x), 252.0)
    c = stats.sharpe_stats(list(x), 252)
    assert a == b == c
    d = json.loads(json.dumps(a.to_dict()))
    assert set(d) == {"n", "mean", "sd", "sr", "skew", "kurt", "se_sr", "psr_0", "periods_per_year",
                      "sr_annual", "se_sr_annual"}
    assert isinstance(d["n"], int)
    a.summary_line("day").encode("ascii")
    assert "assumes IID" in a.summary_line("day")
    assert stats.sharpe_stats.__doc__ and "ddof = 1" in stats.sharpe_stats.__doc__
    assert "Pearson" in stats.sharpe_stats.__doc__ and "IID" in stats.sharpe_stats.__doc__


@pytest.mark.parametrize("bad", [
    [0.01],                                  # n < 2
    [0.01, 0.01, 0.01],                      # constant: sd 0
    [0.01, float("nan"), 0.02],              # NaN
    [0.01, float("inf"), 0.02],
    [[0.01, 0.02], [0.03, 0.04]],            # 2-D
    [True, False, True],                     # booleans
    ["a", "b"],                              # text
    "0.01",
])
def test_sharpe_stats_rejects_bad_returns(bad):
    with pytest.raises(ValueError):
        stats.sharpe_stats(bad, 252)


@pytest.mark.parametrize("q", [0, -252, float("nan"), float("inf"), True, "252"])
def test_sharpe_stats_rejects_bad_periods_per_year(q):
    with pytest.raises(ValueError):
        stats.sharpe_stats([0.01, -0.02, 0.03], q)


def test_sharpe_se_coverage_iid_normal():
    # IID normal returns, true SR 0.1 per period, n = 250: the formula SE must match the spread of the
    # SR estimates within 10%, and SR +- 1.96 SE must cover the true SR about 95% of the time.
    rng = np.random.default_rng(20261008)
    true_sr, n, sims = 0.1, 250, 3000
    x = rng.normal(true_sr, 1.0, (sims, n))
    res = [stats.sharpe_stats(row, None) for row in x]
    sr = np.array([r.sr for r in res])
    se = np.array([r.se_sr for r in res])
    ratio = sr.std(ddof=1) / se.mean()
    assert 0.9 <= ratio <= 1.1, ratio
    cover = np.mean(np.abs(sr - true_sr) <= 1.959963985 * se)
    assert 0.93 <= cover <= 0.97, cover


def test_sharpe_se_non_normal_needs_skew_and_kurtosis():
    # r = mu + (1 - E), E ~ Exp(1): sd 1, skew -2, Pearson kurtosis 9, true SR = mu = 0.3, n = 500.
    # Mertens: 1 - (-2)(0.3) + (9 - 1)/4 * 0.09 = 1.78 vs Lo's normal 1 + 0.09/2 = 1.045 (30% smaller SE).
    rng = np.random.default_rng(7)
    true_sr, n, sims = 0.3, 500, 3000
    x = true_sr + (1.0 - rng.exponential(1.0, (sims, n)))
    res = [stats.sharpe_stats(row, None) for row in x]
    sr = np.array([r.sr for r in res])
    se = np.array([r.se_sr for r in res])
    emp = sr.std(ddof=1)
    assert 0.9 <= emp / se.mean() <= 1.1
    se_true = math.sqrt((1 + 2 * true_sr + 2 * true_sr ** 2) / (n - 1))
    assert 0.9 <= emp / se_true <= 1.1
    se_normal = math.sqrt((1 + true_sr ** 2 / 2) / (n - 1))
    assert emp / se_normal > 1.15                       # the IID-normal SE would be far too small
    cover = np.mean(np.abs(sr - true_sr) <= 1.959963985 * se)
    assert 0.92 <= cover <= 0.97, cover


# ---------------------------------------------------------------------------------------
# PSR

def test_psr_known_value_and_self_benchmark():
    # normal moments, SR 0.1, n 101: z = 0.1 * sqrt(100) / sqrt(1 + 0.01/2) = 0.997509; Phi(z) = 0.840741
    assert stats.psr(0.1, 0.0, 101, 0.0, 3.0) == pytest.approx(0.8407413278, abs=1e-10)
    assert stats.psr(0.1, 0.0, 101, 0.0, 3.0) == pytest.approx(PHI(1 / math.sqrt(1.005)), abs=1e-15)
    # a series against its own SR: exactly one half, whatever the moments
    for sr, skew, kurt in [(0.1, 0.0, 3.0), (0.05, -1.5, 8.0), (-0.2, 0.7, 4.0), (0.3, -3.0, 10.0)]:
        assert stats.psr(sr, sr, 500, skew, kurt) == 0.5
    rng = np.random.default_rng(3)
    s = stats.sharpe_stats(rng.normal(0.05, 1.0, 400), 252)
    assert stats.psr(s.sr, s.sr, s.n, s.skew, s.kurt) == 0.5
    assert stats.psr(s.sr, 0.0, s.n, s.skew, s.kurt) == s.psr_0


def test_psr_monotone_and_skew_kurtosis_penalty():
    base = stats.psr(0.1, 0.0, 250, 0.0, 3.0)
    assert stats.psr(0.12, 0.0, 250, 0.0, 3.0) > base           # higher SR
    assert stats.psr(0.1, 0.02, 250, 0.0, 3.0) < base           # higher benchmark
    assert stats.psr(0.1, 0.0, 500, 0.0, 3.0) > base            # longer record
    assert stats.psr(0.1, 0.0, 250, -1.0, 3.0) < base           # negative skew hurts
    assert stats.psr(0.1, 0.0, 250, 0.0, 9.0) < base            # fat tails hurt
    assert stats.psr(-0.1, 0.0, 250, 0.0, 3.0) == pytest.approx(1 - stats.psr(0.1, 0.0, 250, 0.0, 3.0), abs=1e-15)


def test_psr_size_under_the_null():
    # true SR = benchmark: PSR is close to uniform, so it exceeds 0.95 about 5% of the time
    rng = np.random.default_rng(99)
    true_sr, n, sims = 0.1, 500, 3000
    p = np.array([stats.psr(s.sr, true_sr, s.n, s.skew, s.kurt)
                  for s in (stats.sharpe_stats(row, None) for row in rng.normal(true_sr, 1.0, (sims, n)))])
    assert 0.035 <= np.mean(p > 0.95) <= 0.065
    assert 0.47 <= np.mean(p > 0.5) <= 0.53


def test_psr_rejects_bad_inputs():
    with pytest.raises(ValueError, match="add 3"):
        stats.psr(0.1, 0.0, 250, 0.0, 0.0)                      # excess kurtosis by mistake
    with pytest.raises(ValueError, match="add 3"):
        stats.psr(0.1, 0.0, 250, -3.0, 9.5)                     # below Pearson's bound 1 + skew^2 = 10
    with pytest.raises(ValueError):
        stats.psr(0.1, 0.0, 1, 0.0, 3.0)                        # n must be > 1
    for bad in (float("nan"), "0.1", None, True):
        with pytest.raises(ValueError):
            stats.psr(bad, 0.0, 250, 0.0, 3.0)
        with pytest.raises(ValueError):
            stats.psr(0.1, bad, 250, 0.0, 3.0)
    # exactly on Pearson's bound (a two-point distribution, as in the 2014 example) is allowed
    assert 0 < stats.psr(0.1, 0.0, 250, -3.0, 10.0) < 1


# ---------------------------------------------------------------------------------------
# expected maximum SR and DSR

def _expected_max_normal(n: int) -> float:
    """E[max of n IID standard normals] = integral of x n phi(x) Phi(x)^(n-1) dx (trapezoid rule)."""
    x = np.linspace(-12.0, 12.0, 48001)
    cdf = np.array([0.5 * math.erfc(-v / math.sqrt(2.0)) for v in x])
    f = x * n * np.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi) * cdf ** (n - 1)
    return float(np.sum((f[1:] + f[:-1]) / 2.0 * np.diff(x)))


def test_expected_max_sr_against_numerical_integration():
    # documented accuracy: -7.9% at N = 2, +2.3% at N = 10, +0.9% at N = 100, +0.4% at N = 1000
    assert _expected_max_normal(2) == pytest.approx(1 / math.sqrt(math.pi), abs=1e-9)
    for n, err in ((2, -0.0788), (10, 0.0233), (100, 0.0092), (1000, 0.0042)):
        rel = stats.expected_max_sr(n, 1.0) / _expected_max_normal(n) - 1.0
        assert rel == pytest.approx(err, abs=5e-4), (n, rel)


def test_expected_max_sr_formula_scaling_and_one_trial():
    g = 0.5772156649
    n, v = 37, 0.0123
    expect = math.sqrt(v) * ((1 - g) * PHI_INV(1 - 1 / n) + g * PHI_INV(1 - 1 / (n * math.e)))
    assert stats.expected_max_sr(n, v) == pytest.approx(expect, rel=1e-10)
    assert stats.expected_max_sr(n, 4 * v) == pytest.approx(2 * stats.expected_max_sr(n, v), rel=1e-14)
    assert stats.expected_max_sr(1, v) == 0.0
    assert stats.expected_max_sr(n, 0.0) == 0.0
    assert stats.expected_max_sr(100.0, v) == stats.expected_max_sr(100, v)
    vals = [stats.expected_max_sr(k, v) for k in (2, 3, 5, 10, 100, 1000, 10 ** 6)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("n_trials", [0, -3, 1.5, float("nan"), True, "10"])
def test_expected_max_sr_rejects_bad_trials(n_trials):
    with pytest.raises(ValueError):
        stats.expected_max_sr(n_trials, 0.01)


@pytest.mark.parametrize("v", [-0.01, float("nan"), float("inf"), None])
def test_expected_max_sr_rejects_bad_variance(v):
    with pytest.raises(ValueError):
        stats.expected_max_sr(10, v)


def test_dsr_monotone_in_trials_and_variance():
    sr, n, skew, kurt, v = 0.08, 1000, -0.5, 5.0, 0.001
    # one trial: nothing was selected, DSR = PSR against 0
    assert stats.dsr(sr, n, skew, kurt, 1, v) == stats.psr(sr, 0.0, n, skew, kurt)
    by_trials = [stats.dsr(sr, n, skew, kurt, k, v) for k in (1, 2, 5, 10, 100, 1000, 10 ** 6)]
    assert all(b < a for a, b in zip(by_trials, by_trials[1:])), by_trials
    by_var = [stats.dsr(sr, n, skew, kurt, 100, w) for w in (0.0, 1e-4, 1e-3, 1e-2)]
    assert all(b < a for a, b in zip(by_var, by_var[1:])), by_var
    assert by_var[0] == stats.psr(sr, 0.0, n, skew, kurt)          # no dispersion of trials: no deflation
    # with V = 1e-4, SR0 = 0.01 x 2.5306 = 0.0253 < SR: more data raises the DSR
    by_n = [stats.dsr(sr, m, skew, kurt, 100, 1e-4) for m in (300, 1000, 3000)]
    assert all(b > a for a, b in zip(by_n, by_n[1:])), by_n
    # with V = 1e-3, SR0 = 0.0800 > SR = 0.08: more data LOWERS it (the SR is no better than luck)
    assert stats.dsr(sr, 3000, skew, kurt, 100, v) < stats.dsr(sr, 300, skew, kurt, 100, v) < 0.5
    for x in by_trials + by_var + by_n:
        assert 0.0 <= x <= 1.0


# ---------------------------------------------------------------------------------------
# MinTRL

def test_min_track_record_length_known_value_and_consistency():
    # normal, SR 0.1, SR* 0: 1 + (1 + 0.01/2) * (1.644854 / 0.1)^2 = 272.9071
    assert stats.min_track_record_length(0.1, 0.0, 0.0, 3.0) == pytest.approx(272.9071171366, abs=1e-8)
    for sr, b, skew, kurt, prob in [(0.1, 0.0, 0.0, 3.0, 0.95), (0.158113883, 0.113172002, -3.0, 10.0, 0.95),
                                    (0.05, -0.02, 0.4, 6.0, 0.99), (0.2, 0.1, -1.0, 4.0, 0.9)]:
        m = stats.min_track_record_length(sr, b, skew, kurt, prob)
        assert stats.psr(sr, b, m, skew, kurt) == pytest.approx(prob, abs=1e-12)
    assert (stats.min_track_record_length(0.1, 0.0, 0.0, 3.0, 0.99)
            > stats.min_track_record_length(0.1, 0.0, 0.0, 3.0, 0.95))
    assert (stats.min_track_record_length(0.1, 0.0, -1.0, 6.0)
            > stats.min_track_record_length(0.1, 0.0, 0.0, 3.0))
    assert stats.min_track_record_length(0.1, 0.1, 0.0, 3.0) == math.inf
    assert stats.min_track_record_length(0.05, 0.1, 0.0, 3.0) == math.inf


@pytest.mark.parametrize("prob", [0.0, 1.0, 1.5, -0.1, float("nan")])
def test_min_track_record_length_rejects_bad_prob(prob):
    with pytest.raises(ValueError):
        stats.min_track_record_length(0.1, 0.0, 0.0, 3.0, prob)


# ---------------------------------------------------------------------------------------
# number of trials

def test_effective_trials_and_note():
    assert stats.effective_trials(100) == 100
    assert stats.effective_trials(100, 0.0) == 100
    assert stats.effective_trials(100, 1.0) == 1
    assert stats.effective_trials(100, 0.5) == 51                   # 0.5 + 0.5 * 100 = 50.5 -> rounded up
    assert stats.effective_trials(3, 0.25) == 3                     # 0.25 + 0.75 * 3 = 2.5 -> 3
    assert stats.effective_trials(1, 0.3) == 1
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            stats.effective_trials(10, bad)
    with pytest.raises(ValueError):
        stats.effective_trials(0)
    note = stats.TRIALS_NOTE
    assert "null maxima are the correction" in note and "spec variants" in note
    assert "optional and conservative" in note
    note.encode("ascii")


def test_trials_sr_variance():
    srs = [0.05, 0.10, -0.02, 0.07]
    assert stats.trials_sr_variance(srs) == pytest.approx(np.var(srs, ddof=1), rel=1e-15)
    # annualised SRs -> per-period variance = annual variance / q
    annual = np.array([1.0, 2.0, 0.5, -0.3])
    assert stats.trials_sr_variance(annual / math.sqrt(250)) == pytest.approx(
        stats.trials_sr_variance(annual) / 250, rel=1e-12)
    with pytest.raises(ValueError):
        stats.trials_sr_variance([0.1])
    with pytest.raises(ValueError):
        stats.trials_sr_variance([0.1, float("nan")])


# ---------------------------------------------------------------------------------------
# prop-day returns

def _summer_equity() -> pd.DataFrame:
    # EU summer time (CEST = UTC+2): the prop day starts at 22:00 UTC the evening before.
    times = [_utc(2024, 7, 1, 20), _utc(2024, 7, 1, 21),       # prop day Mon 2024-07-01
             _utc(2024, 7, 1, 22), _utc(2024, 7, 2, 10),       # prop day Tue 2024-07-02 (22:00 UTC = 00:00 CEST)
             _utc(2024, 7, 5, 20),                             # prop day Fri 2024-07-05 (Wed/Thu: no bars)
             _utc(2024, 7, 7, 22)]                             # Sunday 22:00 UTC = Monday 00:00 CEST: 2024-07-08
    close = [101_000.0, 102_000.0, 100_980.0, 99_960.0, 101_959.2, 99_920.016]
    return pd.DataFrame({"time": np.array(times, dtype=np.int64), "equity_close": close})


def test_daily_returns_use_prop_days():
    eq = _summer_equity()
    d = stats.daily_returns_from_equity(eq, C0)
    assert list(d.columns) == ["day", "date", "n_bars", "equity_start", "equity_end", "ret"]
    assert list(d["date"]) == ["2024-07-01", "2024-07-02", "2024-07-05", "2024-07-08"]
    assert list(d["day"]) == [_day(2024, 7, 1), _day(2024, 7, 2), _day(2024, 7, 5), _day(2024, 7, 8)]
    assert d["day"].dtype == np.int64
    assert list(d["n_bars"]) == [2, 2, 1, 1]
    assert list(d["equity_start"]) == [C0, 102_000.0, 99_960.0, 101_959.2]
    assert list(d["equity_end"]) == [102_000.0, 99_960.0, 101_959.2, 99_920.016]
    np.testing.assert_allclose(d["ret"], [0.02, -0.02, 0.02, -0.02], rtol=0, atol=1e-12)
    # the day key is exactly propkit.calendar.prop_day of the bar open
    np.testing.assert_array_equal(d["day"], np.unique(calendar.prop_day(eq["time"].to_numpy())))


def test_daily_returns_utc_day_option_differs():
    d = stats.daily_returns_from_equity(_summer_equity(), C0, by="utc_day")
    assert list(d["date"]) == ["2024-07-01", "2024-07-02", "2024-07-05", "2024-07-07"]
    assert list(d["n_bars"]) == [3, 1, 1, 1]
    assert d["ret"].iloc[0] == pytest.approx(100_980.0 / C0 - 1, abs=1e-15)


def test_daily_returns_winter_and_us_only_shift_week():
    # Winter (CET = UTC+1): Sunday 23:00 UTC is Monday 00:00 CET. In the US-only shift week of
    # 2025-03-09..03-30 the 18:00 New York reopen is 22:00 UTC = 23:00 CET SUNDAY: its own prop day.
    times = [_utc(2025, 1, 5, 23), _utc(2025, 1, 6, 10),        # Monday 2025-01-06
             _utc(2025, 3, 9, 22),                              # Sunday 2025-03-09 (stub day)
             _utc(2025, 3, 9, 23), _utc(2025, 3, 10, 12)]       # Monday 2025-03-10
    eq = pd.DataFrame({"time": times, "equity_close": [100_100.0, 100_200.0, 100_300.0, 100_250.0, 100_500.0]})
    d = stats.daily_returns_from_equity(eq, C0)
    assert list(d["date"]) == ["2025-01-06", "2025-03-09", "2025-03-10"]
    assert list(d["n_bars"]) == [2, 1, 2]
    np.testing.assert_allclose(d["ret"], [100_200 / C0 - 1, 100_300 / 100_200 - 1, 100_500 / 100_300 - 1],
                               rtol=0, atol=1e-15)


def test_daily_returns_match_prop_day_on_synthetic_year():
    from propkit.bars import synthetic_bars
    b = synthetic_bars(_utc(2024, 1, 2, 0), 24 * 300, seed=5)
    rng = np.random.default_rng(5)
    close = C0 + np.cumsum(rng.normal(0.0, 50.0, len(b)))
    eq = pd.DataFrame({"time": b["time"], "equity_close": close})
    d = stats.daily_returns_from_equity(eq, C0)
    day = np.asarray(calendar.prop_day(b["time"].to_numpy()))
    np.testing.assert_array_equal(d["day"], np.unique(day))
    last = np.r_[np.flatnonzero(np.diff(day)), len(day) - 1]
    np.testing.assert_array_equal(d["equity_end"], close[last])
    assert d["n_bars"].sum() == len(b)
    growth = np.prod(1.0 + d["ret"].to_numpy())
    assert growth == pytest.approx(close[-1] / C0, rel=1e-12)
    assert 250 < stats.observed_periods_per_year(d["day"]) < 275


def test_daily_returns_rejects_bad_input():
    eq = _summer_equity()
    with pytest.raises(ValueError):
        stats.daily_returns_from_equity(eq["equity_close"], C0)                  # not a DataFrame
    with pytest.raises(ValueError, match="equity_close"):
        stats.daily_returns_from_equity(eq[["time"]], C0)
    with pytest.raises(ValueError):
        stats.daily_returns_from_equity(eq.iloc[0:0], C0)
    with pytest.raises(ValueError, match="sorted"):
        stats.daily_returns_from_equity(eq.iloc[[1, 0, 2]].reset_index(drop=True), C0)
    with pytest.raises(ValueError, match="sorted"):
        stats.daily_returns_from_equity(eq.iloc[[0, 0, 2]].reset_index(drop=True), C0)
    nan = eq.copy()
    nan.loc[2, "equity_close"] = float("nan")
    with pytest.raises(ValueError, match="row 2"):
        stats.daily_returns_from_equity(nan, C0)
    for c0 in (0.0, -1.0, float("nan"), "100000", True):
        with pytest.raises(ValueError):
            stats.daily_returns_from_equity(eq, c0)
    with pytest.raises(ValueError, match="by must be"):
        stats.daily_returns_from_equity(eq, C0, by="week")
    blown = eq.copy()
    blown.loc[1, "equity_close"] = 0.0
    with pytest.raises(ValueError, match="blown"):
        stats.daily_returns_from_equity(blown, C0)


def test_observed_periods_per_year():
    # 5 weekdays out of 7 calendar days: 5 / (7 / 365.25)
    days = [_day(2024, 7, 1) + k for k in range(5)]
    assert stats.observed_periods_per_year(days) == pytest.approx(5 / (5 / 365.25))
    days = [_day(2024, 7, 1) + k for k in (0, 1, 2, 3, 4, 7, 8, 9, 10, 11)]
    assert stats.observed_periods_per_year(days) == pytest.approx(10 / (12 / 365.25))
    with pytest.raises(ValueError):
        stats.observed_periods_per_year([_day(2024, 7, 1)])
    with pytest.raises(ValueError):
        stats.observed_periods_per_year([5, 3])


# ---------------------------------------------------------------------------------------
# expected shortfall and drawdowns

def test_expected_shortfall_acerbi_tasche():
    x = [3.0, -5.0, 1.0, 6.0, -1.0, 0.0, 2.0, -3.0, 4.0, 5.0]    # sorted: -5, -3, -1, 0, 1, ...
    assert stats.expected_shortfall(x, 0.2) == pytest.approx(-4.0)            # n alpha = 2: (-5 - 3) / 2
    assert stats.expected_shortfall(x, 0.15) == pytest.approx(-6.5 / 1.5)    # n alpha = 1.5: (-5 + 0.5 x -3) / 1.5
    assert stats.expected_shortfall(x, 0.05) == pytest.approx(-5.0)           # n alpha < 1: the worst value
    assert stats.expected_shortfall(x, 0.3) == pytest.approx(-3.0)            # (-5 - 3 - 1) / 3
    rng = np.random.default_rng(1)
    y = rng.normal(0, 0.01, 1000)
    assert stats.expected_shortfall(y, 0.05) == pytest.approx(np.sort(y)[:50].mean(), rel=1e-12)
    assert stats.expected_shortfall(rng.permutation(y), 0.05) == stats.expected_shortfall(y, 0.05)
    for bad in (0.0, 1.0, -0.05, float("nan")):
        with pytest.raises(ValueError):
            stats.expected_shortfall(x, bad)
    with pytest.raises(ValueError):
        stats.expected_shortfall([], 0.05)


def _dd_equity() -> pd.DataFrame:
    # one bar per prop day at 12:00 UTC; C0 = 100,000
    days = [(2024, 7, 1), (2024, 7, 2), (2024, 7, 3), (2024, 7, 4), (2024, 7, 5),
            (2024, 7, 8), (2024, 7, 9), (2024, 7, 10)]
    close = [101_000.0, 100_500.0, 101_200.0, 99_000.0, 98_000.0, 100_000.0, 101_300.0, 101_000.0]
    worst = [100_700.0, 100_200.0, 100_900.0, 98_700.0, 97_000.0, 99_700.0, 101_000.0, 100_700.0]
    return pd.DataFrame({"time": [_utc(y, m, d, 12) for y, m, d in days], "equity_close": close,
                         "equity_worst": worst})


def test_drawdown_stats_hand_example():
    eq = _dd_equity()
    s = stats.drawdown_stats(eq, C0)
    # intrabar: peak_k = max(C0, closes BEFORE bar k) = 100000, 101000, 101000, 101200, 101200, 101200, 101200,
    # 101300; peak - worst = -700, 800, 100, 2500, 4200, 1500, 200, 600 -> 4200 on Fri 2024-07-05
    assert s["max_dd_usd"] == pytest.approx(4200.0)
    assert s["max_dd_pct"] == pytest.approx(4.2)
    assert s["max_dd_pct_of_peak"] == pytest.approx(4200 / 101_200 * 100)
    assert s["max_dd_peak_usd"] == 101_200.0
    assert s["max_dd_time"] == _utc(2024, 7, 5, 12) and s["max_dd_time_utc"] == "2024-07-05 12:00:00 UTC"
    # close to close: running peak incl. the bar: 101000, 101000, 101200, ..., 101300; deepest 101200 - 98000
    assert s["max_dd_close_usd"] == pytest.approx(3200.0)
    assert s["max_dd_close_pct"] == pytest.approx(3.2)
    assert s["max_dd_close_pct_of_peak"] == pytest.approx(3200 / 101_200 * 100)
    assert s["max_dd_close_time"] == _utc(2024, 7, 5, 12)
    # underwater days: Tue 07-02 (1 day), Thu 07-04 .. Mon 07-08 (3 days with bars, 5 calendar days), Wed 07-10
    assert s["longest_underwater_days"] == 3
    assert s["longest_underwater_calendar_days"] == 5
    assert (s["longest_underwater_start"], s["longest_underwater_end"]) == ("2024-07-04", "2024-07-08")
    assert s["underwater_at_end"] is True
    # daily returns: worst is 99000 / 101200 - 1 on 07-04; 8 days, n alpha = 0.4 < 1: ES = worst return
    assert s["n_days"] == 8 and s["n_bars"] == 8 and s["initial_capital"] == C0
    assert s["worst_day_ret"] == pytest.approx(99_000 / 101_200 - 1, abs=1e-15)
    assert s["worst_day_date"] == "2024-07-04"
    assert s["es_alpha"] == 0.05 and s["es_ret"] == pytest.approx(99_000 / 101_200 - 1, abs=1e-15)
    s25 = stats.drawdown_stats(eq, C0, alpha=0.25)            # n alpha = 2: the two worst days
    assert s25["es_ret"] == pytest.approx(((99_000 / 101_200 - 1) + (98_000 / 99_000 - 1)) / 2, abs=1e-15)
    json.dumps(s)


def test_drawdown_stats_without_worst_column_and_flat_path():
    eq = _dd_equity().drop(columns="equity_worst")
    s = stats.drawdown_stats(eq, C0)
    assert s["max_dd_usd"] is None and s["max_dd_pct"] is None and s["max_dd_time"] is None
    assert s["max_dd_close_usd"] == pytest.approx(3200.0)
    up = pd.DataFrame({"time": [_utc(2024, 7, 1, 12), _utc(2024, 7, 2, 12)], "equity_close": [100_500.0, 101_000.0],
                       "equity_worst": [100_000.0, 100_500.0]})
    s = stats.drawdown_stats(up, C0)
    assert s["max_dd_usd"] == 0.0 and s["max_dd_time"] is None and s["max_dd_peak_usd"] is None
    assert s["max_dd_close_usd"] == 0.0 and s["max_dd_close_time"] is None
    assert s["longest_underwater_days"] == 0 and s["longest_underwater_start"] is None
    assert s["underwater_at_end"] is False
    json.dumps(s)


def test_drawdown_stats_matches_evaluator_definition_on_hourly_path():
    # the intrabar max drawdown uses the evaluator's definition: running max of C0 and EARLIER bar closes,
    # minus equity_worst; recomputed here independently on a random hourly path
    from propkit.bars import synthetic_bars
    b = synthetic_bars(_utc(2024, 3, 1, 0), 24 * 60, seed=9)
    rng = np.random.default_rng(9)
    close = C0 + np.cumsum(rng.normal(0.0, 150.0, len(b)))
    worst = close - rng.uniform(0.0, 120.0, len(b))
    eq = pd.DataFrame({"time": b["time"], "equity_close": close, "equity_worst": worst})
    s = stats.drawdown_stats(eq, C0)
    expect = 0.0
    peak = C0
    for c, w in zip(close, worst):
        expect = max(expect, peak - w)
        peak = max(peak, c)
    assert s["max_dd_usd"] == pytest.approx(expect, abs=1e-9)
    assert s["max_dd_close_usd"] <= s["max_dd_usd"]


def test_drawdown_stats_rejects_worst_above_close():
    eq = _dd_equity()
    eq.loc[3, "equity_worst"] = eq.loc[3, "equity_close"] + 1.0
    with pytest.raises(ValueError, match="equity_worst"):
        stats.drawdown_stats(eq, C0)
    with pytest.raises(ValueError):
        stats.drawdown_stats(_dd_equity(), C0, alpha=1.5)


# ---------------------------------------------------------------------------------------
# hygiene

def test_source_is_ascii_imports_nothing_from_alphamaster_and_no_scipy():
    text = (ROOT / "propkit" / "stats.py").read_text(encoding="utf-8")
    assert text.isascii()
    mods = {m.split(".")[0] for m in re.findall(r"^\s*(?:from|import)\s+([\w.]+)", text, flags=re.M)}
    forbidden = {"model_core", "data_pipeline", "config", "web", "utils", "strategy_manager", "execution",
                 "scripts", "scipy", "zoneinfo"}
    assert not forbidden & mods, mods
    for name in ("sharpe_stats", "psr", "expected_max_sr", "dsr", "dsr_details", "min_track_record_length",
                 "effective_trials", "trials_sr_variance", "daily_returns_from_equity", "observed_periods_per_year",
                 "expected_shortfall", "drawdown_stats"):
        doc = getattr(stats, name).__doc__
        assert doc and len(doc) > 80, name
    for name in ("psr", "expected_max_sr", "dsr", "dsr_details", "min_track_record_length", "sharpe_stats",
                 "expected_shortfall"):
        assert re.search(r"(Bailey|Lo \(2002\)|Mertens|Acerbi)", getattr(stats, name).__doc__), name


def test_report_sr_var_note_does_not_call_the_default_a_lower_bound():
    """Finding 13: 1/(n-1) is the null sampling variance of one SR estimate, not the smallest plausible
    variance across trials (correlated variants spread less)."""
    from propkit import report
    note = report.SR_VAR_NOTE
    assert "smallest plausible" not in note and "NOT a bound" in note
    assert "trials_sr_variance" in note and "effective_trials" in note and note.isascii()


def test_longest_underwater_does_not_count_days_at_the_peak():
    """Review finding: a day that closes exactly at the running peak is not underwater (strict <)."""
    days = [(2024, 7, 1), (2024, 7, 2), (2024, 7, 3), (2024, 7, 4), (2024, 7, 5), (2024, 7, 8)]
    close = [100_000.0, 101_000.0, 101_000.0, 101_000.0, 100_000.0, 102_000.0]
    eq = pd.DataFrame({"time": [_utc(y, m, d, 12) for y, m, d in days], "equity_close": close})
    s = stats.drawdown_stats(eq, C0)
    assert s["longest_underwater_days"] == 1                              # only 07-05, not 07-03 .. 07-05
    assert (s["longest_underwater_start"], s["longest_underwater_end"]) == ("2024-07-05", "2024-07-05")
    assert s["underwater_at_end"] is False
