"""propkit/stats.py - Sharpe-ratio statistics (standard error, PSR, DSR, MinTRL), prop-day returns and
drawdowns. Research only.

Conventions used by every function here:
  * a "return" is a simple return per period as a FRACTION (0.01 = +1%), or any per-period quantity whose
    mean / standard deviation is wanted (R multiples per trade work too);
  * Sharpe ratios (SR) are PER PERIOD unless a name ends in _annual: SR = mean / sd with no risk-free rate
    subtracted (a prop account earns no interest on its notional capital); annualised SR = SR x sqrt(q),
    q = periods per year, which assumes IID returns (CLAUDE.md A9: autocorrelated returns make sqrt(q) wrong);
  * sd is the SAMPLE standard deviation (ddof = 1);
  * skew and kurtosis are the moment estimators g1 = m3 / m2^1.5 and b2 = m4 / m2^2 (central moments
    m_k = mean((x - mean)^k), ddof = 0), kurtosis is PEARSON kurtosis (normal = 3, NOT excess kurtosis);
  * n is the number of returns (observations); in the PSR / DSR formulas "sr" is the estimated SR and
    every SR, benchmark and variance of SRs must be in the SAME per-period units as n (daily SRs with a
    count of days; to use annual figures divide SR by sqrt(q) and a variance of SRs by q);
  * money in USD; *_pct fields are PERCENT (3.0 = 3%); *_ret fields are return fractions (negative = loss);
  * days are prop days (propkit.calendar.prop_day: the CE(S)T calendar date) as int64 days since
    1970-01-01, unless by="utc_day" is asked for, or a day boundary of propkit.calendar.DAY_BOUNDARIES
    ("cet_midnight" = the prop day, "ny_17" = 17:00 New York, "utc_midnight" = the UTC date).

References (cited again at each formula):
  Lo, A. W. (2002). The Statistics of Sharpe Ratios. Financial Analysts Journal 58(4), 36-52.
  Mertens, E. (2002). Comments on Variance of the IID Estimator in Lo (2002). Working paper, Univ. Basel.
  Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. Journal of Risk 15(2),
    3-44. (Probabilistic Sharpe Ratio PSR, minimum track record length MinTRL.)
  Bailey, D. H. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias,
    Backtest Overfitting and Non-Normality. Journal of Portfolio Management 40(5), 94-107. (Expected
    maximum SR SR0, DSR, and the numerical example SR0 = 0.1132, DSR = 0.9004 used as the test gate.)
  Bailey, D. H., Borwein, J., Lopez de Prado, M. and Zhu, Q. J. (2014). Pseudo-Mathematics and Financial
    Charlatanism. Notices of the AMS 61(5), 458-471. (The expected maximum of N normal trials.)
  Acerbi, C. and Tasche, D. (2002). On the Coherence of Expected Shortfall. Journal of Banking and
    Finance 26(7), 1487-1503. (The empirical expected shortfall estimator.)
No scipy: Phi and its inverse come from statistics.NormalDist.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from propkit import bars as bars_mod
from propkit import calendar

EULER_GAMMA = 0.5772156649015329       # Euler-Mascheroni constant (the contract's 0.5772156649)
DAY_KEYS = ("prop_day", "utc_day") + calendar.DAY_BOUNDARIES
KURTOSIS_TOL = 1e-9                     # relative slack on Pearson's bound kurt >= 1 + skew^2
WORST_TOL_USD = 1e-6                    # equity_worst may exceed equity_close by this much (float noise)
_STD_NORMAL = NormalDist()              # immutable standard normal: Phi = cdf, Phi^-1 = inv_cdf

TRIALS_NOTE = (
    "Number of trials N (an input; propkit cannot count it for you). N is how many strategy variants "
    "were tried before this one was picked, INCLUDING the discarded ones. For zeno's rule, N = the number "
    "of spec variants evaluated (every parameter set, session filter or exit tried counts once). For "
    "AlphaMaster's miner the null maxima are the correction: the best scores of the noise-arm runs (the "
    "same search on data with no signal) show directly how high selection alone pushes a score, so compare "
    "against them first; DSR with N = the number of unique formulas scored is optional and conservative "
    "(mined formulas are strongly correlated, so the effective N is far smaller). Correlated trials: "
    "effective_trials(M, rho) gives the rough proxy N = rho + (1 - rho) x M. var_sr is the variance of the "
    "trials' SR estimates in the same per-period units as sr.")


# ---------------------------------------------------------------------------------------
# input checks

def _real(value, what: str, positive: bool = False, nonneg: bool = False) -> float:
    """A finite real number as float; raises ValueError for booleans, text, NaN and infinities."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{what} must be a number, got {value!r}")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    if positive and x <= 0:
        raise ValueError(f"{what} must be > 0, got {value!r}")
    if nonneg and x < 0:
        raise ValueError(f"{what} must be >= 0, got {value!r}")
    return x


def _returns_array(returns, what: str = "returns") -> np.ndarray:
    """A 1-D float64 array of finite values; raises ValueError naming the first bad position."""
    if isinstance(returns, (pd.DataFrame, str, bytes)):
        raise ValueError(f"{what} must be a 1-D list, array or Series of numbers")
    raw = np.asarray(returns)
    if raw.ndim != 1:
        raise ValueError(f"{what} must be 1-D (one return per period), got shape {raw.shape}")
    if raw.dtype.kind == "b":
        raise ValueError(f"{what} must hold numbers, not booleans")
    try:
        arr = raw.astype(np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must hold numbers (simple returns as fractions), got dtype {raw.dtype}")
    bad = ~np.isfinite(arr)
    if bad.any():
        raise ValueError(f"{what} has a missing or infinite value at position {int(np.flatnonzero(bad)[0])}; "
                         "drop or fix it first (propkit never fills gaps silently)")
    return arr


def _check_n(n, what: str = "n") -> float:
    x = _real(n, what)
    if x <= 1:
        raise ValueError(f"{what} (the number of returns) must be > 1, got {n!r}")
    return x


def _check_moments(skew, kurt) -> tuple[float, float]:
    """Skew and PEARSON kurtosis, checked against Pearson's bound kurt >= 1 + skew^2 (every distribution)."""
    s = _real(skew, "skew")
    k = _real(kurt, "kurt")
    bound = 1.0 + s * s
    if k < bound - KURTOSIS_TOL * bound:
        raise ValueError(
            f"kurtosis {k} is below 1 + skew^2 = {bound}, which no distribution has. propkit uses PEARSON "
            "kurtosis (normal = 3); if your figure is EXCESS kurtosis (normal = 0), add 3.")
    return s, k


def _sr_variance_term(sr: float, skew: float, kurt: float) -> float:
    """1 - skew x SR + (kurt - 1) / 4 x SR^2: n - 1 times the asymptotic variance of the SR estimate
    (Mertens 2002; Bailey and Lopez de Prado 2012). Raises ValueError when it is not positive."""
    term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if not term > 0:
        raise ValueError(f"the SR variance term 1 - skew*SR + (kurt-1)/4*SR^2 is {term} (not positive) for "
                         f"SR={sr}, skew={skew}, kurt={kurt}: the moments describe a degenerate two-point "
                         "distribution; check the inputs")
    return term


# ---------------------------------------------------------------------------------------
# Sharpe ratio and its standard error

@dataclass(frozen=True)
class SharpeStats:
    """Sharpe-ratio statistics of one return series (see sharpe_stats for every definition).

    n: number of returns; mean: arithmetic mean per period; sd: sample sd (ddof = 1); sr: mean / sd per
    period; skew: g1 = m3 / m2^1.5; kurt: Pearson b2 = m4 / m2^2 (normal = 3); se_sr: standard error of
    sr (Lo 2002 / Mertens 2002, IID, non-normal); psr_0: PSR against a zero benchmark, P(true SR > 0);
    periods_per_year: q, or None; sr_annual = sr x sqrt(q) and se_sr_annual = se_sr x sqrt(q) (IID
    assumption, CLAUDE.md A9), None when q is None.
    """

    n: int
    mean: float
    sd: float
    sr: float
    skew: float
    kurt: float
    se_sr: float
    psr_0: float
    periods_per_year: float | None
    sr_annual: float | None
    se_sr_annual: float | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict of the fields."""
        return asdict(self)

    def summary_line(self, period: str = "period") -> str:
        """One plain ASCII line, e.g. 'SR 0.0800 +- 0.0640 per day (n = 250, skew -0.10, kurt 3.20) ...'."""
        line = (f"SR {self.sr:.4f} +- {self.se_sr:.4f} per {period} (n = {self.n}, skew {self.skew:.2f}, "
                f"kurt {self.kurt:.2f}, PSR vs 0 = {self.psr_0:.3f})")
        if self.sr_annual is not None:
            line += (f"; annualised {self.sr_annual:.2f} +- {self.se_sr_annual:.2f} at "
                     f"{self.periods_per_year:g} {period}s per year (assumes IID returns)")
        return line


def sharpe_stats(returns, periods_per_year: float | None) -> SharpeStats:
    """Sharpe ratio of a return series with its standard error, skew, kurtosis and PSR against 0.

    returns: 1-D numbers, one per period (simple returns as fractions, or R multiples per trade); NaN or
    infinite values raise ValueError (drop them first). At least 2 values, not all equal.
    periods_per_year: q > 0 to annualise (252 or 260 trading days, or observed_periods_per_year), or
    None (e.g. per-trade statistics): the annual fields are then None.

    Definitions (no risk-free rate is subtracted):
      mean = sum(x) / n;  sd = sqrt(sum((x - mean)^2) / (n - 1))  (SAMPLE sd, ddof = 1);
      SR = mean / sd  (per period);
      skew = m3 / m2^1.5 and kurt = m4 / m2^2, m_k = sum((x - mean)^k) / n  (moment estimators, ddof = 0;
        kurt is Pearson kurtosis: 3 for normal returns);
      SE(SR) = sqrt((1 - skew x SR + (kurt - 1) / 4 x SR^2) / (n - 1))  - the asymptotic standard error of
        the SR estimate for IID but non-normal returns: Lo (2002) for normal returns (where it reduces to
        sqrt((1 + SR^2 / 2) / (n - 1))), extended to skew and kurtosis by Mertens (2002), with n - 1 as in
        Bailey and Lopez de Prado (2012);
      psr_0 = psr(SR, 0, n, skew, kurt);
      SR_annual = SR x sqrt(q), SE_annual = SE x sqrt(q): valid only for IID returns (CLAUDE.md A9).
    """
    x = _returns_array(returns)
    n = int(x.size)
    if n < 2:
        raise ValueError(f"returns needs at least 2 values to estimate a standard deviation, got {n}")
    if np.ptp(x) == 0:
        raise ValueError("returns are all equal (standard deviation 0): the Sharpe ratio is undefined")
    q = None if periods_per_year is None else _real(periods_per_year, "periods_per_year", positive=True)
    mean = float(x.mean())
    d = x - mean
    m2 = float(np.mean(d * d))
    skew = float(np.mean(d ** 3)) / m2 ** 1.5
    kurt = float(np.mean(d ** 4)) / (m2 * m2)
    sd = float(np.std(x, ddof=1))
    sr = mean / sd
    se = math.sqrt(_sr_variance_term(sr, skew, kurt) / (n - 1))
    return SharpeStats(
        n=n, mean=mean, sd=sd, sr=sr, skew=skew, kurt=kurt, se_sr=se,
        psr_0=psr(sr, 0.0, n, skew, kurt), periods_per_year=q,
        sr_annual=None if q is None else sr * math.sqrt(q),
        se_sr_annual=None if q is None else se * math.sqrt(q),
    )


# ---------------------------------------------------------------------------------------
# PSR, expected maximum SR, DSR, MinTRL

def _psr_z(sr: float, sr_benchmark: float, n: float, skew: float, kurt: float) -> float:
    return (sr - sr_benchmark) * math.sqrt(n - 1.0) / math.sqrt(_sr_variance_term(sr, skew, kurt))


def psr(sr: float, sr_benchmark: float, n: float, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe Ratio: the probability that the TRUE per-period SR exceeds sr_benchmark.

    PSR(SR*) = Phi((SR - SR*) x sqrt(n - 1) / sqrt(1 - skew x SR + (kurt - 1) / 4 x SR^2))
    (Bailey and Lopez de Prado 2012, "The Sharpe Ratio Efficient Frontier"; the denominator is the
    Mertens (2002) variance term evaluated at the ESTIMATED SR). Phi is the standard normal CDF.

    sr: estimated SR per period; sr_benchmark: SR* in the same per-period units; n: number of returns
    behind sr (> 1; normally an integer, any real is accepted); skew: g1; kurt: PEARSON kurtosis (normal
    = 3; a value below 1 + skew^2, e.g. excess kurtosis, raises ValueError). Returns a probability in
    [0, 1]; psr(sr, sr, ...) = 0.5.
    """
    s = _real(sr, "sr")
    b = _real(sr_benchmark, "sr_benchmark")
    m = _check_n(n)
    sk, ku = _check_moments(skew, kurt)
    return _STD_NORMAL.cdf(_psr_z(s, b, m, sk, ku))


def _check_trials(n_trials) -> int:
    x = _real(n_trials, "n_trials")
    if x < 1 or x != math.floor(x):
        raise ValueError(f"n_trials must be a whole number >= 1 (the count of strategy variants tried), got "
                         f"{n_trials!r}; round an effective count UP (effective_trials does)")
    return int(x)


def expected_max_sr(n_trials: int, var_sr: float) -> float:
    """Expected maximum of the estimated SRs of n_trials independent trials whose TRUE SR is 0: SR0.

    SR0 = sqrt(V) x ((1 - gamma) x Phi^-1(1 - 1/N) + gamma x Phi^-1(1 - 1/(N e)))
    (Bailey and Lopez de Prado 2014, "The Deflated Sharpe Ratio"; the extreme-value approximation of the
    expected maximum of N standard normals is from Bailey, Borwein, Lopez de Prado and Zhu 2014),
    gamma = Euler-Mascheroni constant 0.5772156649, e = Euler's number, N = n_trials, V = var_sr.

    n_trials: whole number >= 1; N = 1 returns 0.0 (one trial, nothing selected: the formula itself
    tends to -infinity there). var_sr: the variance of the trials' SR estimates (>= 0), in the SAME
    per-period units as the SR it will be compared with (an annual variance / periods per year).
    Returns SR0 in those per-period units. Accuracy of the approximation for the expected maximum of N
    standard normals (checked by numerical integration in the tests): -7.9% at N = 2 (SR0 too low), then
    +2.3% at N = 10, +0.9% at N = 100, +0.4% at N = 1000 (slightly conservative: SR0 a little too high).
    """
    n = _check_trials(n_trials)
    v = _real(var_sr, "var_sr", nonneg=True)
    if n == 1:
        return 0.0
    z1 = _STD_NORMAL.inv_cdf(1.0 - 1.0 / n)
    z2 = _STD_NORMAL.inv_cdf(1.0 - 1.0 / (n * math.e))
    return math.sqrt(v) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def dsr_details(sr: float, n: float, skew: float, kurt: float, n_trials: int, var_sr: float) -> dict[str, Any]:
    """Deflated Sharpe Ratio with its intermediate numbers (all per period).

    DSR = PSR(SR0) = Phi((SR - SR0) x sqrt(n - 1) / sqrt(1 - skew x SR + (kurt - 1) / 4 x SR^2)),
    SR0 = expected_max_sr(n_trials, var_sr) (Bailey and Lopez de Prado 2014, "The Deflated Sharpe
    Ratio"): the probability that the true SR is above the SR the best of n_trials zero-skill trials
    would show by luck. Arguments as in psr and expected_max_sr; see TRIALS_NOTE for choosing n_trials.

    Returns a dict: sr, n, skew, kurt, n_trials, var_sr, sr0 (the expected maximum SR), variance_term
    (1 - skew x SR + (kurt - 1) / 4 x SR^2), z (the argument of Phi) and dsr (a probability).
    """
    s = _real(sr, "sr")
    m = _check_n(n)
    sk, ku = _check_moments(skew, kurt)
    sr0 = expected_max_sr(n_trials, var_sr)
    term = _sr_variance_term(s, sk, ku)
    z = _psr_z(s, sr0, m, sk, ku)
    return {"sr": s, "n": m, "skew": sk, "kurt": ku, "n_trials": _check_trials(n_trials),
            "var_sr": float(var_sr), "sr0": sr0, "variance_term": term, "z": z, "dsr": _STD_NORMAL.cdf(z)}


def dsr(sr: float, n: float, skew: float, kurt: float, n_trials: int, var_sr: float) -> float:
    """Deflated Sharpe Ratio: psr(sr, expected_max_sr(n_trials, var_sr), n, skew, kurt).

    DSR = Phi((SR - SR0) x sqrt(n - 1) / sqrt(1 - skew x SR + (kurt - 1) / 4 x SR^2)) with SR0 the
    expected maximum SR of n_trials zero-skill trials (Bailey and Lopez de Prado 2014). sr, SR0 and
    var_sr are per period; n is the number of returns; kurt is Pearson kurtosis (normal = 3). With
    n_trials = 1, DSR = PSR against 0. More trials or a larger var_sr give a LOWER DSR. Returns a
    probability in [0, 1]. See dsr_details for the intermediate numbers and TRIALS_NOTE for n_trials.
    """
    return float(dsr_details(sr, n, skew, kurt, n_trials, var_sr)["dsr"])


def min_track_record_length(sr: float, sr_benchmark: float, skew: float, kurt: float,
                            prob: float = 0.95) -> float:
    """Minimum track record length: how many returns are needed before PSR(sr_benchmark) reaches prob.

    MinTRL = 1 + (1 - skew x SR + (kurt - 1) / 4 x SR^2) x (Phi^-1(prob) / (SR - SR*))^2
    (Bailey and Lopez de Prado 2012, "The Sharpe Ratio Efficient Frontier"); by construction
    psr(sr, sr_benchmark, MinTRL, skew, kurt) = prob.

    sr: estimated SR per period; sr_benchmark: SR* per period; kurt: Pearson kurtosis (normal = 3);
    prob: confidence in (0, 1), default 0.95. Returns a number of PERIODS (a real number; round up),
    or math.inf when sr <= sr_benchmark (no track record is long enough).
    """
    s = _real(sr, "sr")
    b = _real(sr_benchmark, "sr_benchmark")
    sk, ku = _check_moments(skew, kurt)
    p = _real(prob, "prob")
    if not 0.0 < p < 1.0:
        raise ValueError(f"prob must be strictly between 0 and 1 (e.g. 0.95), got {prob!r}")
    term = _sr_variance_term(s, sk, ku)
    if s <= b:
        return math.inf
    return 1.0 + term * (_STD_NORMAL.inv_cdf(p) / (s - b)) ** 2


def effective_trials(n_trials: int, avg_correlation: float = 0.0) -> int:
    """Rough effective number of independent trials among n_trials correlated ones (rounded UP).

    N_eff = rho + (1 - rho) x M, M = n_trials, rho = the average pairwise correlation of the trials'
    return series (0 <= rho <= 1): M when the trials are independent, 1 when they are identical, linear
    in between. It is a heuristic with no known error bound, used in the multiple-testing discussion
    around Bailey and Lopez de Prado (2014, "The Deflated Sharpe Ratio") [ASSUMPTION: attribution not
    re-checked against the paper]; clustering the trials' return series is better when they are
    available. The result is rounded up (more trials = lower, more conservative DSR). Use it as n_trials
    in dsr / expected_max_sr; see TRIALS_NOTE.
    """
    m = _check_trials(n_trials)
    rho = _real(avg_correlation, "avg_correlation")
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"avg_correlation must be between 0 and 1, got {avg_correlation!r} (for a negative "
                         "average correlation use 0: it treats the trials as independent, the conservative "
                         "choice)")
    return max(1, int(math.ceil(rho + (1.0 - rho) * m - 1e-9)))


def trials_sr_variance(sr_values) -> float:
    """Variance of the SR estimates of the trials (sample variance, ddof = 1): the V of expected_max_sr.

    sr_values: one SR per trial, all per period in the same units (to convert an annualised SR divide it
    by sqrt(q); the variance of annualised SRs divided by q gives the per-period variance). At least 2
    values. Returns V in (per-period SR)^2.
    """
    x = _returns_array(sr_values, "sr_values")
    if x.size < 2:
        raise ValueError("sr_values needs at least 2 trials to estimate a variance")
    return float(np.var(x, ddof=1))


# ---------------------------------------------------------------------------------------
# prop-day returns, expected shortfall, drawdowns

def _equity_inputs(equity: pd.DataFrame, C0) -> tuple[np.ndarray, np.ndarray, float]:
    """Validated (time, equity_close, C0) of an EQUITY frame (only time and equity_close are needed)."""
    if not isinstance(equity, pd.DataFrame):
        raise ValueError("equity must be a pandas DataFrame (the EQUITY frame from propkit.equity)")
    missing = [c for c in ("time", "equity_close") if c not in equity.columns]
    if missing:
        raise ValueError(f"EQUITY is missing column(s) {missing}; build it with propkit.equity")
    if len(equity) == 0:
        raise ValueError("EQUITY has no rows")
    c0 = _real(C0, "C0 (the initial capital, USD)", positive=True)
    t = bars_mod.to_epoch_seconds(equity["time"], "EQUITY time")
    bad = np.diff(t) <= 0
    if bad.any():
        i = int(np.flatnonzero(bad)[0]) + 1
        raise ValueError(f"EQUITY row {i} ({calendar.utc_str(int(t[i]))}) is not after the row before; times "
                         "must be sorted and unique")
    close = _float_column(equity, "equity_close")
    return t, close, c0


def _float_column(df: pd.DataFrame, col: str) -> np.ndarray:
    try:
        arr = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"EQUITY column '{col}' must hold numbers (USD)")
    bad = ~np.isfinite(arr)
    if bad.any():
        raise ValueError(f"EQUITY column '{col}' has a missing or infinite value at row {int(np.flatnonzero(bad)[0])}")
    return arr


def _day_keys(t: np.ndarray, by: str) -> np.ndarray:
    if by == "prop_day":
        return np.asarray(calendar.prop_day(t), dtype=np.int64)
    if by == "utc_day":
        calendar.prop_day(t)                          # same range check as prop_day
        return t // calendar.SECONDS_PER_DAY
    if by in calendar.DAY_BOUNDARIES:
        return np.asarray(calendar.firm_day(t, by), dtype=np.int64)
    raise ValueError(f"by must be one of {DAY_KEYS}, got {by!r}")


def daily_returns_from_equity(equity: pd.DataFrame, C0: float, by: str = "prop_day") -> pd.DataFrame:
    """Simple daily returns of an EQUITY path, one row per day WITH bars.

    ret_d = E_d / E_(d-1) - 1, E_d = equity_close of the last bar of day d (the day's closing equity,
    open positions marked at the bar close), E before the first day = C0 (EQUITY starts flat at C0).
    Days without bars (weekends, holidays) are skipped, so a Monday's return runs from the previous
    day with bars (normally Friday). Days: by="prop_day" (default) keys each bar by
    propkit.calendar.prop_day(time), the CE(S)T calendar date of its open (00:00 CE(S)T = 22:00 UTC in
    summer, 23:00 UTC in winter); by="utc_day" uses the UTC date; by= a day boundary ("cet_midnight" = the
    prop day, "ny_17" = 17:00 New York to 17:00 New York, "utc_midnight") uses propkit.calendar.firm_day.
    In the US-only DST shift weeks the Sunday reopen hour is its own prop day (see propkit.calendar): it is
    kept as a (short) day.

    equity: EQUITY frame (only time, int64 UTC epoch seconds of the bar open, sorted and unique, and
    equity_close, USD, are used); C0: initial capital in USD (> 0). A closing equity <= 0 before the
    last day raises ValueError (the next return is undefined).
    Returns a DataFrame: day (int64 days since 1970-01-01), date ('YYYY-MM-DD'), n_bars, equity_start
    (USD, the previous day's close or C0), equity_end (USD), ret (fraction, 0.01 = +1%).
    """
    t, close, c0 = _equity_inputs(equity, C0)
    day = _day_keys(t, by)
    last = np.r_[np.flatnonzero(day[1:] != day[:-1]), t.size - 1]
    end = close[last]
    start = np.r_[c0, end[:-1]]
    if (start <= 0).any():
        i = int(np.flatnonzero(start <= 0)[0])
        raise ValueError(f"the closing equity before {calendar.day_to_str(int(day[last[i]]))} is "
                         f"{start[i]:,.2f} USD (<= 0): the account is blown and later returns are undefined")
    keys = day[last]
    return pd.DataFrame({
        "day": keys.astype(np.int64),
        "date": np.asarray(calendar.day_to_str(keys), dtype=object),
        "n_bars": np.diff(np.r_[-1, last]).astype(np.int64),
        "equity_start": start, "equity_end": end, "ret": end / start - 1.0,
    })


def observed_periods_per_year(days) -> float:
    """Days with data per year actually observed: len(days) / ((last - first + 1) / 365.25).

    days: the day keys of daily_returns_from_equity (int days since 1970-01-01, sorted), at least 2.
    For Monday-Friday data this is about 261 (a little more with the Sunday stub days of the US-only
    DST shift weeks). Use it as periods_per_year in sharpe_stats when no fixed day count is wanted;
    short samples give a noisy figure.
    """
    d = _returns_array(days, "days")
    if d.size < 2:
        raise ValueError("days needs at least 2 values")
    if (np.diff(d) <= 0).any():
        raise ValueError("days must be sorted and unique")
    return float(d.size / ((d[-1] - d[0] + 1.0) / 365.25))


def expected_shortfall(returns, alpha: float = 0.05) -> float:
    """Empirical expected shortfall: the mean return of the worst alpha share of the returns.

    ES_alpha = (sum of the k = floor(n alpha) smallest returns + (n alpha - k) x the (k+1)-th smallest)
               / (n alpha)
    (Acerbi and Tasche 2002, the estimator that weights the boundary observation fractionally). Returned
    as a RETURN: negative = loss (e.g. -0.021 = the worst 5% of days lose 2.1% on average). With n alpha <
    1 it is the single worst return. returns: 1-D finite numbers; alpha in (0, 1), default 0.05.
    """
    x = np.sort(_returns_array(returns))
    if x.size == 0:
        raise ValueError("returns is empty")
    a = _real(alpha, "alpha")
    if not 0.0 < a < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1 (e.g. 0.05), got {alpha!r}")
    na = x.size * a
    k = int(math.floor(na + 1e-9))
    frac = na - k if na - k > 1e-9 else 0.0
    tail = float(x[:k].sum()) + (frac * float(x[k]) if frac > 0 else 0.0)
    return tail / na


def _longest_underwater(daily: pd.DataFrame, c0: float) -> dict[str, Any]:
    """Longest run of consecutive days (with bars) whose closing equity is below the running peak."""
    end = daily["equity_end"].to_numpy(dtype=np.float64)
    keys = daily["day"].to_numpy(dtype=np.int64)
    peak = np.maximum.accumulate(np.r_[c0, end])[:-1]          # C0 and the EARLIER day closes
    under = end < peak
    best = (0, -1, -1)                                          # (length, first index, last index)
    i = 0
    while i < under.size:
        if under[i]:
            j = i
            while j + 1 < under.size and under[j + 1]:
                j += 1
            if j - i + 1 > best[0]:
                best = (j - i + 1, i, j)
            i = j + 1
        else:
            i += 1
    length, a, b = best
    if length == 0:
        return {"longest_underwater_days": 0, "longest_underwater_calendar_days": 0,
                "longest_underwater_start": None, "longest_underwater_end": None,
                "underwater_at_end": False}
    return {"longest_underwater_days": int(length),
            "longest_underwater_calendar_days": int(keys[b] - keys[a] + 1),
            "longest_underwater_start": calendar.day_to_str(int(keys[a])),
            "longest_underwater_end": calendar.day_to_str(int(keys[b])),
            "underwater_at_end": bool(under[-1])}


def _max_drawdown(t: np.ndarray, peak: np.ndarray, low: np.ndarray, c0: float) -> dict[str, Any]:
    """Largest peak - low (USD, % of C0, % of that peak) and the bar where it happened (None if 0)."""
    dd = np.maximum(peak - low, 0.0)
    k = int(np.argmax(dd))
    hit = bool(dd[k] > 0)
    return {"usd": float(dd[k]), "pct": float(dd[k] / c0 * 100.0),
            "pct_of_peak": float(dd[k] / peak[k] * 100.0), "peak_usd": float(peak[k]) if hit else None,
            "time": int(t[k]) if hit else None, "time_utc": calendar.utc_str(int(t[k])) if hit else None}


def drawdown_stats(equity: pd.DataFrame, C0: float, alpha: float = 0.05, by: str = "prop_day") -> dict[str, Any]:
    """Drawdowns, underwater time and expected shortfall of an EQUITY path (USD; *_pct are PERCENT).

    Intrabar max drawdown (needs the equity_worst column; None fields without it):
      max_dd_usd = max over bars k of (peak_k - equity_worst_k), peak_k = max(C0, equity_close of the bars
      BEFORE k) - the same definition as propkit.evaluator (over the whole path here); max_dd_pct = that /
      C0 x 100; max_dd_pct_of_peak = that / peak_k x 100; max_dd_time = the bar open of bar k.
    Close-to-close max drawdown: max over k of (max(C0, equity_close up to k) - equity_close_k), as USD,
    % of C0 and % of the peak (max_dd_close_*).
    Underwater: a day (with bars) is underwater when its closing equity is below max(C0, the closing
      equity of every EARLIER day); longest_underwater_days = the longest run of consecutive underwater
      days counted in prop days with bars, longest_underwater_calendar_days = its first to last day
      inclusive (weekends included), with its start and end dates; underwater_at_end = the last day is
      underwater (the run has not recovered).
    Daily returns (daily_returns_from_equity, prop days): n_days, es_ret = expected_shortfall at alpha
      (default 0.05; a return, negative = loss), worst_day_ret and worst_day_date.

    equity: EQUITY frame (time and equity_close required, equity_worst optional; equity_worst above
    equity_close raises ValueError); C0: initial capital in USD (> 0); by: the day key of the daily
    returns (daily_returns_from_equity). Returns a JSON-serialisable dict.
    """
    t, close, c0 = _equity_inputs(equity, C0)
    daily = daily_returns_from_equity(equity, c0, by)
    out: dict[str, Any] = {"initial_capital": c0, "n_bars": int(t.size), "n_days": int(len(daily))}
    if "equity_worst" in equity.columns:
        worst = _float_column(equity, "equity_worst")
        bad = worst > close + WORST_TOL_USD + 1e-12 * np.abs(close)
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise ValueError(f"EQUITY row {i} ({calendar.utc_str(int(t[i]))}) has equity_worst {worst[i]:,.2f} "
                             f"above equity_close {close[i]:,.2f}; equity_worst is the LOWEST equity in the bar")
        intrabar = _max_drawdown(t, np.maximum.accumulate(np.r_[c0, close[:-1]]), worst, c0)
        out.update({f"max_dd_{k}": v for k, v in intrabar.items()})
    else:
        out.update({key: None for key in ("max_dd_usd", "max_dd_pct", "max_dd_pct_of_peak", "max_dd_peak_usd",
                                          "max_dd_time", "max_dd_time_utc")})
    c2c = _max_drawdown(t, np.maximum(np.maximum.accumulate(close), c0), close, c0)
    out.update({f"max_dd_close_{k}": v for k, v in c2c.items()})
    out.update(_longest_underwater(daily, c0))
    rets = daily["ret"].to_numpy(dtype=np.float64)
    k = int(np.argmin(rets))
    out.update({"es_alpha": _real(alpha, "alpha"), "es_ret": expected_shortfall(rets, alpha),
                "worst_day_ret": float(rets[k]), "worst_day_date": str(daily["date"].iloc[k])})
    return out
