"""scripts/research/make_null.py - build placebo, sign-flip null and positive-control data files.

Usage (from the repo root; Windows PowerShell shown; $D is the folder that holds the data):
    $env:PYTHONUTF8 = "1"
    $D = "..\\AlphaMaster\\research\\data"
    python scripts\\research\\make_null.py shuffle  $D\\XAUUSD_H1.parquet  $D\\PLACEBO_H1.parquet --seed 7
    python scripts\\research\\make_null.py signflip $D\\XAUUSD_H1.parquet  $D\\FLIP1_H1.parquet --seed 1
    python scripts\\research\\make_null.py plant    $D\\PLACEBO_H1.parquet $D\\PLANT1_H1.parquet --target-sharpe 1.0
    python scripts\\research\\make_null.py plant --help      (each mode has its own --help)

What it does:
  * reads a source Parquet file in AlphaMaster's schema (time, open, high, low, close,
    tick_volume) and writes a research data file in the same schema: time int64 unix seconds,
    open/high/low/close/tick_volume float64. The output name must look like SYMBOL_TF.parquet
    (e.g. FLIP1_H1.parquet), because scripts/run_trial.py requires it;
  * splits every bar into its gap g = log(open/previous close), body b = log(close/open),
    upper wick uw = log(high/max(open,close)) and lower wick lw = log(min(open,close)/low),
    changes those numbers as the mode says, and re-chains the prices from the first open
    (the same decomposition as make_placebo.py);
  * shuffle  : shuffles the order of the bars (same output as make_placebo.py shuffle mode);
    signflip : random-direction null that keeps the volatility structure;
    plant    : positive control, adds a known causal 24-bar momentum signal to a base file;
  * prints a summary and writes a sidecar JSON next to the output (same name + '.json') with
    the mode, seed, parameters, source and output sha256 and the diagnostics.

It refuses any path that contains 'locked_holdout' or ends in '.locked' (nothing is read),
an output name that is not SYMBOL_TF.parquet, that equals the source or that carries the
symbol of the real data it descends from, and an existing output unless --overwrite is
given; --overwrite replaces only a file this tool made (its sidecar must match it), never
real data. The sidecar is written first and the output last, and a failure restores the
old sidecar. Exit code 0 when the file was written, 2 for usage or data errors (nothing is
written then). Research only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BANNER = "RESEARCH ONLY - not trading advice"
TOOL = "scripts/research/make_null.py"
N_BLOCKS = 5                 # AlphaMaster's walk-forward uses T // 5 sized folds (engine.n_folds = 5)
MOMENTUM_BARS = 24           # plant: s[t] = sign of the sum of g+b over bars t-24 .. t-1
MIN_BARS = 50
MIN_SIGNFLIP_BARS = 2000     # signflip: shorter files cannot cancel the block drift to float precision
DEV_TOL = 1e-12              # signflip: max change of |g - block mean| and |b - block mean|
DRIFT_REL_TOL = 1e-9         # signflip: max relative change of the block sum of g+b
SHARPE_TOL = 0.05            # plant: the calibrated oracle Sharpe must be this close to the target
HYSTERESIS_BANDS = (0.1, 0.2)  # plant: low-turnover reference rules (band in rolling sd of the window)
HYSTERESIS_SD_BARS = 500     # plant: rolling window for the sd of the 24-bar log return
FORBIDDEN_PART = "locked_holdout"
FORBIDDEN_SUFFIX = ".locked"

PLANT_HELP = f"""\
Positive control: adds a known, causal directional signal to a BASE file (intended: the
seed-7 placebo, which has no time structure of its own).

  s[t] = sign of the sum of (g+b) over bars t-{MOMENTUM_BARS} .. t-1 of the NEW series, computed bar by
         bar, so the planted series is its own {MOMENTUM_BARS}-bar momentum (s = 0 for the first
         {MOMENTUM_BARS} bars). In prices: s[t+1] = sign(log(close[t] / close[t-{MOMENTUM_BARS}])), known at the
         close of bar t.
  b_new[t] = b[t] + c * s[t] * sigma, sigma = std of (g+b) of the base file.

c is found by bisection so that the ORACLE strategy reaches the target annualised Sharpe
(net of costs by default; --calibrate-on gross uses the Sharpe before costs). The oracle
holds position[t] = s[t+1] (known at the close of bar t), which is AlphaMaster's timing:
the position at bar t earns target_ret[t] = log(open[t+2]/open[t+1]). Costs are
|change in position| x Config.COST_RATE; the annualisation uses estimate_periods_per_year
on the file's timestamps, as the engine does. c is one constant fitted on the whole file:
it sets the strength, never the direction, which uses past bars only. The oracle Sharpe
jumps a little as c changes (the series feeds its own signal), so on short files the
target is hit to within a few hundredths; the achieved value is printed and must be within
{SHARPE_TOL} of the target, or nothing is written.

How strong is the plant? The oracle is a simple reference, NOT an upper bound. It trades
the exact rule at full size (+1/-1) and changes position often (typically about 0.15 per
bar), so costs take a large share of its gross Sharpe: a 'net 1.0' plant is a strong
signal (its gross oracle Sharpe, printed, was about 2 on a gold-like H1 placebo). Rules
that trade less can net MORE from the same signal; the sidecar reports two such references
(keep the previous sign until the {MOMENTUM_BARS}-bar log return moves more than {HYSTERESIS_BANDS[0]} or {HYSTERESIS_BANDS[1]} of
its rolling {HYSTERESIS_SD_BARS}-bar sd), so a miner result above the oracle is possible and is not by
itself a sign of leakage. Measures that do not depend on
turnover are also reported: c (in units of sigma), the oracle's gross Sharpe and its IC
(correlation of position[t] with target_ret[t]), plus the oracle per AlphaMaster
validation fold. The planted rule is close to features the miner already has (a 20-bar
return, a 20-bar sum), so detecting it shows power only for signals of that kind.

One plant is one point. To see where detection stops, build a ladder of weaker plants on
the gross scale, e.g. --calibrate-on gross --target-sharpe 0.5, 1 and 2 (a ladder of net
targets is squeezed together, because costs alone push the oracle's net Sharpe far below 0
at c = 0; the sidecar records that value as oracle_net_sharpe_at_c0).

gaps, wicks, tick_volume and timestamps are kept; the planted file has no randomness, so
--seed is only recorded in the sidecar (e.g. the seed of the base placebo)."""

SIGNFLIP_HELP = f"""\
Random-direction null that keeps the volatility structure (needs at least {MIN_SIGNFLIP_BARS:,} bars).

  * the bars are split into 5 consecutive blocks of T // 5 bars (the remainder goes to the
    last block). These are AlphaMaster's walk-forward fold boundaries (fold size T // 5,
    engine.n_folds = 5) when the walk-forward gap collapses to 0, as it does for the
    63,645-bar train file; AlphaMaster leaves the few remainder bars out of its folds;
  * within each block the block means of g and of b are subtracted; each bar is flipped
    with probability 0.5: its demeaned g and b are both negated and its upper and lower
    wicks are swapped; the block means are added back; tick_volume and timestamps stay;
  * flipping coins alone would change each block's total log return (the sum of g+b) by
    about sqrt(block size) x sigma. To keep that drift exactly, a few extra bars per block
    are flipped to cancel the change (random bars that move the sum back, then the four bars
    that best cancel the rest; about 0.3 percent of the bars of a 63,645-bar file, more on
    short files), and the tiny remainder (typically below 1e-12) is spread over the block's
    bodies. The share of flipped bars stays near 0.5, and |g - block mean| and
    |b - block mean| are unchanged to float precision. Below about 2,000 bars the blocks are
    too small to cancel the drift that precisely, so shorter files are refused;
  * the first bar is never flipped: its gap is not observable (it has no previous close).

Diagnostics: share of bars flipped; max change of |g - block mean| and |b - block mean|
(must be 0 up to float error); block drift kept (sum of g+b per block); lag-1
autocorrelation of b and |b| before and after; OHLC validity."""

SHUFFLE_HELP = """\
Placebo: shuffles the order of the bars (gap, body, wicks and tick_volume move together)
and re-chains the prices from the first open, keeping the timestamps. Any time-series
structure is destroyed. The output is byte-for-byte the same as make_placebo.py's shuffle
mode for the same seed (for sources whose time column is already unix seconds)."""


class DataError(Exception):
    """A usage or data problem: nothing is written and the exit code is 2."""


# ---------------------------------------------------------------------------------------
# helpers

def _ascii(value) -> str:
    """Plain-ASCII text for the console (non-ASCII becomes a \\u escape)."""
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


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


def _fmt(x, spec: str = ".4f") -> str:
    if x is None:
        return "n/a"
    return format(x, spec)


def is_forbidden(path_text: str) -> bool:
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


# ---------------------------------------------------------------------------------------
# bar decomposition (same formulas as make_placebo.py)

def decompose(o, h, lo, c):
    """Per-bar gap, body, upper wick and lower wick (log units)."""
    prev_c = np.r_[o[0], c[:-1]]
    g = np.log(o / prev_c)
    b = np.log(c / o)
    uw = np.log(h / np.maximum(o, c))
    lw = np.log(np.minimum(o, c) / lo)
    return g, b, uw, lw


def rebuild(first_open, g, b, uw, lw):
    """Re-chain prices from the first open (the exact arithmetic of make_placebo.py)."""
    n = len(g)
    out_o = np.empty(n)
    out_c = np.empty(n)
    last = first_open
    for i in range(n):
        out_o[i] = last * np.exp(g[i])
        out_c[i] = out_o[i] * np.exp(b[i])
        last = out_c[i]
    out_h = np.maximum(out_o, out_c) * np.exp(uw)
    out_l = np.minimum(out_o, out_c) * np.exp(-lw)
    return out_o, out_h, out_l, out_c


def ohlc_check(o, h, lo, c) -> dict:
    """Counts of OHLC problems; valid = high >= max(open,close), low <= min(open,close), all > 0."""
    arr = np.vstack([o, h, lo, c])
    finite = np.isfinite(arr).all(axis=0)
    with np.errstate(invalid="ignore"):
        res = {
            "bars": int(len(o)),
            "nonfinite_bars": int((~finite).sum()),
            "nonpositive_bars": int((arr <= 0).any(axis=0).sum()),
            "high_below_body": int((h < np.maximum(o, c)).sum()),
            "low_above_body": int((lo > np.minimum(o, c)).sum()),
        }
    res["valid"] = all(res[k] == 0 for k in ("nonfinite_bars", "nonpositive_bars",
                                              "high_below_body", "low_above_body"))
    return res


def lag1_autocorr(x) -> float | None:
    x = np.asarray(x, dtype="float64")
    if len(x) < 3:
        return None
    a, b = x[1:], x[:-1]
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def block_bounds(n: int, n_blocks: int = N_BLOCKS) -> list[tuple[int, int]]:
    """n_blocks consecutive blocks of n // n_blocks bars; the remainder goes to the last block."""
    size = n // n_blocks
    if size < 2:
        raise DataError(f"too few bars ({n}) for {n_blocks} blocks")
    edges = [k * size for k in range(n_blocks)] + [n]
    return list(zip(edges[:-1], edges[1:]))


# ---------------------------------------------------------------------------------------
# modes

def shuffle_bars(g, b, uw, lw, vol, seed: int):
    """make_placebo.py shuffle mode: one permutation of whole bars."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(g))
    return g[idx], b[idx], uw[idx], lw[idx], vol[idx]


def _best_quad(w: np.ndarray, target: float, rng, max_bars: int = 1500) -> tuple[list[int], float]:
    """Four distinct indices into w whose values sum closest to target (two disjoint pairs).

    Meet in the middle: all pair sums of (a random subset of at most max_bars) entries are
    sorted, and for each pair the best partner pair is found by binary search.
    """
    m = len(w)
    sub = np.sort(rng.choice(m, size=max_bars, replace=False)) if m > max_bars else np.arange(m)
    iu, ju = np.triu_indices(len(sub), k=1)
    ps = w[sub[iu]] + w[sub[ju]]
    order = np.argsort(ps, kind="stable")
    ps, iu, ju = ps[order], sub[iu[order]], sub[ju[order]]
    at = np.searchsorted(ps, target - ps)
    best: tuple[list[int], float] = ([], math.inf)
    for off in (-1, 0):
        k = np.clip(at + off, 0, len(ps) - 1)
        err = np.abs(ps + ps[k] - target)
        err[(iu == iu[k]) | (iu == ju[k]) | (ju == iu[k]) | (ju == ju[k])] = np.inf
        a = int(np.argmin(err))
        if err[a] < best[1]:
            best = ([int(iu[a]), int(ju[a]), int(iu[k[a]]), int(ju[k[a]])], float(err[a]))
    return best


def _balance_block(y: np.ndarray, flip: np.ndarray, rng, fixed: np.ndarray) -> dict:
    """Flip a few extra bars so that sum(sign * y) returns to 0 (y = demeaned g+b of a block).

    flip is changed in place. Phase 1 flips random bars that move the sum toward 0 without
    overshooting; phase 2 flips the four bars that best cancel what is left. Bars in `fixed`
    (and bars already used here) are never touched.
    """
    sign = np.where(flip, -1.0, 1.0)
    v = sign * y
    resid = float(v.sum())
    start = resid
    used = fixed.copy()
    n_repair = 0
    while resid != 0.0:
        ok = ~used & (np.sign(v) == np.sign(resid)) & (2.0 * np.abs(v) <= abs(resid))
        cand = np.flatnonzero(ok)
        if cand.size == 0:
            break
        j = int(cand[rng.integers(cand.size)])
        resid -= 2.0 * v[j]
        v[j] = -v[j]
        flip[j] = not flip[j]
        used[j] = True
        n_repair += 1
    free = np.flatnonzero(~used)
    if resid != 0.0 and free.size >= 4:
        picks, err = _best_quad(2.0 * v[free], resid, rng)
        if err < abs(resid):
            for k in free[picks]:
                resid -= 2.0 * v[k]
                v[k] = -v[k]
                flip[k] = not flip[k]
                n_repair += 1
    return {"drift_change_before_repair": start, "repair_flips": n_repair, "residual_after_flips": resid}


def signflip_bars(g, b, uw, lw, seed: int, n_blocks: int = N_BLOCKS):
    """Random-direction null: flip demeaned g and b (and swap wicks) per bar, keep block drift."""
    n = len(g)
    rng = np.random.default_rng(seed)
    flip = rng.random(n) < 0.5
    flip[0] = False                        # bar 0 has no previous close: its gap is not observable
    random_flips = int(flip.sum())
    fixed = np.zeros(n, dtype=bool)
    fixed[0] = True
    g_new = np.empty(n)
    b_new = np.empty(n)
    blocks = []
    for k, (s, e) in enumerate(block_bounds(n, n_blocks)):
        mg = float(g[s:e].mean())
        mb = float(b[s:e].mean())
        gd = g[s:e] - mg
        bd = b[s:e] - mb
        fl = flip[s:e]                     # a view: _balance_block updates `flip` in place
        info = _balance_block(gd + bd, fl, rng, fixed[s:e])
        sign = np.where(fl, -1.0, 1.0)
        spread = info["residual_after_flips"] / (e - s)
        g_new[s:e] = sign * gd + mg
        b_new[s:e] = sign * bd + mb - spread
        info.update({"block": k + 1, "start": s, "end": e, "mean_g": mg, "mean_b": mb,
                     "residual_spread_per_bar": spread})
        blocks.append(info)
    uw_new = np.where(flip, lw, uw)
    lw_new = np.where(flip, uw, lw)
    info = {"flipped": flip, "random_flips": random_flips, "blocks": blocks}
    return g_new, b_new, uw_new, lw_new, info


def plant_series(g, b, k: float, window: int = MOMENTUM_BARS):
    """Planted bodies and the oracle position, computed bar by bar (causal).

    s[t] = sign(sum of g+b_new over bars t-window .. t-1), 0 for t < window;
    b_new[t] = b[t] + k * s[t]   (k = c * sigma).
    Returns (b_new, s, position) with position[t] = s[t+1], i.e. the sign of the window that
    ends at bar t (known at the close of bar t); position[t] = 0 for t < window - 1.
    """
    n = len(g)
    gl = np.asarray(g, dtype="float64").tolist()
    bl = np.asarray(b, dtype="float64").tolist()
    k = float(k)
    r: list[float] = []
    b_out: list[float] = []
    s_out: list[int] = []
    for t in range(n):
        if t >= window:
            w = sum(r[t - window:t])
            st = 1 if w > 0 else (-1 if w < 0 else 0)
        else:
            st = 0
        bn = bl[t] + k * st
        b_out.append(bn)
        s_out.append(st)
        r.append(gl[t] + bn)
    if n >= window:
        w = sum(r[n - window:n])
        s_next = 1 if w > 0 else (-1 if w < 0 else 0)
    else:
        s_next = 0
    s = np.asarray(s_out, dtype="float64")
    position = np.r_[s[1:], s_next]
    return np.asarray(b_out, dtype="float64"), s, position


def target_returns_from_parts(g, b):
    """AlphaMaster's target_ret[t] = log(open[t+2]/open[t+1]) = g[t+2] + b[t+1]; last two are 0."""
    n = len(g)
    tr = np.zeros(n)
    if n >= 3:
        tr[:n - 2] = g[2:] + b[1:n - 1]
    return tr


def _pnl(position, target_ret, cost_rate: float):
    """Per-bar gross and net returns and |change in position| (AlphaMaster's cost model)."""
    position = np.asarray(position, dtype="float64")
    prev = np.r_[0.0, position[:-1]]
    turnover = np.abs(position - prev)
    gross = position * target_ret
    net = gross - turnover * cost_rate
    return gross, net, turnover


def _sharpe(x, ppy: int) -> float | None:
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / sd * math.sqrt(ppy)) if sd > 0 else None


def strategy_stats(position, target_ret, ppy: int, cost_rate: float) -> dict:
    """Gross/net annualised Sharpe and turnover with AlphaMaster's cost model."""
    position = np.asarray(position, dtype="float64")
    gross, net, turnover = _pnl(position, target_ret, cost_rate)
    return {
        "gross_sharpe": _sharpe(gross, ppy),
        "net_sharpe": _sharpe(net, ppy),
        "mean_abs_dpos_per_bar": float(turnover.mean()),
        "turnover_per_year": float(turnover.mean() * ppy),
        "share_bars_in_market": float((position != 0).mean()),
        "share_long": float((position > 0).mean()),
    }


def position_ic(position, target_ret, start: int = MOMENTUM_BARS - 1, end: int | None = None) -> float | None:
    """Pearson correlation of position[t] with target_ret[t] over bars [start, end) (end <= n - 2)."""
    n = len(position)
    end = n - 2 if end is None else min(end, n - 2)
    p = np.asarray(position, dtype="float64")[start:end]
    r = np.asarray(target_ret, dtype="float64")[start:end]
    if len(p) < 3 or p.std() == 0 or r.std() == 0:
        return None
    return float(np.corrcoef(p, r)[0, 1])


def hysteresis_position(r, band: float, window: int = MOMENTUM_BARS, sd_bars: int = HYSTERESIS_SD_BARS):
    """Low-turnover reference rule on the same signal (causal).

    w[t] = sum of r over bars t-window+1 .. t (= log(close[t]/close[t-window]), known at the
    close of t). position[t] = sign(w[t]) when |w[t]| > band x rolling sd of w over the last
    sd_bars bars (or while that sd is not yet available), else position[t-1].
    """
    n = len(r)
    cs = np.r_[0.0, np.cumsum(r)]
    w = np.full(n, np.nan)
    w[window - 1:] = cs[window:] - cs[:-window]
    sd = pd.Series(w).rolling(sd_bars, min_periods=50).std().to_numpy()
    pos = np.zeros(n)
    cur = 0.0
    for t in range(window - 1, n):
        if not np.isfinite(sd[t]) or abs(w[t]) > band * sd[t]:
            cur = float(np.sign(w[t]))
        pos[t] = cur
    return pos


def fold_bounds(n: int) -> list[dict]:
    """AlphaMaster's walk-forward folds for n bars (engine defaults: 5 folds, ModelConfig.WF_GAP)."""
    from model_core.config import ModelConfig
    from model_core.engine import _build_walk_forward_folds
    return _build_walk_forward_folds(n, N_BLOCKS, ModelConfig.WF_GAP)


def fold_stats(position, target_ret, ppy: int, cost_rate: float, folds: list[dict]) -> dict:
    """The strategy per validation fold and over all validation bars (one continuous run)."""
    gross, net, turnover = _pnl(position, target_ret, cost_rate)
    rows = []
    for k, f in enumerate(folds):
        s, e = f["val_start"], f["val_end"]
        rows.append({"fold": k + 1, "val_start": s, "val_end": e,
                     "gross_sharpe": _sharpe(gross[s:e], ppy), "net_sharpe": _sharpe(net[s:e], ppy),
                     "mean_abs_dpos_per_bar": float(turnover[s:e].mean()),
                     "ic": position_ic(position, target_ret, s, e)})
    idx = np.concatenate([np.arange(f["val_start"], f["val_end"]) for f in folds])
    return {"folds": rows,
            "all_val_gross_sharpe": _sharpe(gross[idx], ppy), "all_val_net_sharpe": _sharpe(net[idx], ppy)}


def calibrate_plant(g, b, sigma: float, ppy: int, cost_rate: float, target: float,
                    measure: str = "net_sharpe", aim: float = 1e-3, max_iter: int = 40,
                    max_brackets: int = 12):
    """Find c >= 0 so that the oracle's annualised Sharpe (`measure`: net_sharpe or
    gross_sharpe) equals target (within aim).

    The Sharpe rises with c but jumps wherever a window sum crosses zero (the planted series
    feeds its own signal), so one bisection can end on a jump. Then a grid of c values around
    it is searched for other crossings, each is bisected, and the closest c is kept.
    """
    cache: dict[float, dict] = {}

    def f(c: float) -> float:
        c = float(c)
        if c not in cache:
            b_new, _, pos = plant_series(g, b, c * sigma)
            cache[c] = strategy_stats(pos, target_returns_from_parts(g, b_new), ppy, cost_rate)
        v = cache[c][measure]
        return -math.inf if v is None else v

    def best() -> float:
        return min(cache, key=lambda c: abs(f(c) - target))

    def bisect(lo: float, hi: float) -> None:   # f(lo) < target <= f(hi)
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if not lo < mid < hi:
                return
            v = f(mid)
            if abs(v - target) <= aim:
                return
            if v < target:
                lo = mid
            else:
                hi = mid

    base = f(0.0)
    if base >= target:
        raise DataError(f"the base file already gives the oracle a {measure.split('_')[0]} Sharpe of "
                        f"{base:.3f} >= target {target}; use a placebo base or a higher --target-sharpe")
    lo, hi = 0.0, 0.05
    while f(hi) < target:
        lo, hi = hi, hi * 2.0
        if hi > 100.0:
            raise DataError("could not reach the target Sharpe with c <= 100; check the base file")
    bisect(lo, hi)
    brackets = 1
    if abs(f(best()) - target) > aim:
        c0 = best()
        grid = [float(x) for x in np.linspace(0.8 * c0, 1.2 * c0, 41)]
        vals = [f(x) for x in grid]
        pairs = [(grid[i], grid[i + 1]) for i in range(len(grid) - 1) if vals[i] < target <= vals[i + 1]]
        pairs.sort(key=lambda p: abs(p[0] - c0))
        for a, z in pairs[:max_brackets]:
            bisect(a, z)
            brackets += 1
            if abs(f(best()) - target) <= aim:
                break
    c = best()
    b_new, s, pos = plant_series(g, b, c * sigma)
    return c, cache[c], (b_new, s, pos), {"oracle_net_sharpe_at_c0": cache[0.0]["net_sharpe"],
                                          "oracle_gross_sharpe_at_c0": cache[0.0]["gross_sharpe"],
                                          "brackets_searched": brackets, "evaluations": len(cache)}


# ---------------------------------------------------------------------------------------
# I/O and checks

def load_source(path: Path):
    """Read and validate a source file; returns the time-sorted frame (make_placebo.py order)."""
    try:
        df = pd.read_parquet(path)
    except Exception as e:  # pyarrow raises several types
        raise DataError(f"cannot read {path.name} as Parquet: {type(e).__name__}: {e}")
    vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
    missing = [col for col in ("time", "open", "high", "low", "close", vol_col) if col not in df.columns]
    if missing:
        raise DataError(f"{path.name} is missing column(s) {missing}; expected time, open, high, low, "
                        "close, tick_volume")
    if len(df) < MIN_BARS:
        raise DataError(f"{path.name} has {len(df)} bars; at least {MIN_BARS} are needed")
    from data_pipeline.parquet_manager import _time_to_unix_seconds
    try:
        df = df.sort_values("time").reset_index(drop=True)
        ts = _time_to_unix_seconds(df["time"])
        if pd.api.types.is_numeric_dtype(ts) and ts.max() < 10_000_000:
            ts = ts * 1000                 # AlphaMaster's fix for 'seconds / 1000' exports
        if ts.isna().any():
            raise DataError(f"{path.name} has missing timestamps")
        ts = ts.astype("int64")
    except (ValueError, TypeError, OverflowError) as e:   # e.g. text such as '2021-01-01 00:00:00'
        raise DataError(f"{path.name}: the time column is not unix seconds or a datetime column "
                        f"(AlphaMaster cannot read it either): {type(e).__name__}: {e}")
    if (np.diff(ts.to_numpy()) <= 0).any():
        raise DataError(f"{path.name} has duplicate timestamps; AlphaMaster would drop rows, so clean "
                        "the file first")
    prices = df[["open", "high", "low", "close"]].to_numpy(dtype="float64")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise DataError(f"{path.name} has missing, infinite or non-positive prices; clean the file first")
    return df, ts, vol_col


def _read_own_sidecar(path: Path) -> dict | None:
    """The record in a sidecar JSON written by this tool, or None (missing, unreadable, not ours)."""
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):          # missing, a folder, locked, not UTF-8 / not JSON
        return None
    return rec if isinstance(rec, dict) and rec.get("tool") == TOOL else None


def check_paths(src_text: str, out_text: str, overwrite: bool) -> tuple[Path, Path, Path]:
    """Apply the path rules; returns (source, output, sidecar).

    Reads nothing, except an existing output and its sidecar when --overwrite is given: an
    existing output is replaced only when its sidecar was written by this tool and its
    output_sha256 matches the file, so real data can never be overwritten.
    """
    for label, text in (("source", src_text), ("output", out_text)):
        if is_forbidden(text):
            raise DataError(f"refused: the {label} path points at the locked holdout "
                            f"('{FORBIDDEN_PART}' or '*{FORBIDDEN_SUFFIX}'); this tool never reads or "
                            "writes it")
    src = Path(src_text).resolve()
    out = Path(out_text).resolve()
    if not src.is_file():
        raise DataError(f"source file not found: {src}")
    from data_pipeline.parquet_manager import parse_parquet_filename
    try:
        _, out_tf = parse_parquet_filename(out)
    except ValueError:
        raise DataError(f"the output name must look like SYMBOL_TF.parquet (e.g. FLIP1_H1.parquet), "
                        f"because run_trial.py requires it: {out.name}")
    if os.path.normcase(str(src)) == os.path.normcase(str(out)) or src.name.lower() == out.name.lower():
        raise DataError(f"the output name must differ from the source name ({src.name}), so a null file "
                        "is never confused with the data it came from")
    try:
        _, src_tf = parse_parquet_filename(src)
    except ValueError:
        src_tf = None
    if src_tf is not None and src_tf != out_tf:
        raise DataError(f"the output timeframe ({out_tf}) must match the source timeframe ({src_tf})")
    if not out.parent.is_dir():
        raise DataError(f"output folder does not exist: {out.parent}")
    sidecar = out.with_name(out.name + ".json")
    existing = [p for p in (out, sidecar) if p.exists()]
    if existing and not overwrite:
        raise DataError("output already exists: " + ", ".join(str(p) for p in existing)
                        + ". Choose another output name, or add --overwrite to replace a file this tool "
                        "made (real data is never replaced)")
    if existing:
        rec = _read_own_sidecar(sidecar)
        ok = rec is not None
        if ok and out.exists():
            ok = out.is_file() and rec.get("output_sha256") == _sha256(out)
        if not ok:
            raise DataError(f"refused: {out} was not made by this tool (there is no sidecar {sidecar.name} "
                            "from make_null.py whose output_sha256 matches it), so --overwrite will not "
                            "replace it: it may be real data. Choose another output name")
    return src, out, sidecar


def root_symbol(src: Path, src_sha: str) -> str | None:
    """The symbol of the real data a source descends from.

    Follows the source's own make_null sidecar when it matches the file (a null built from a
    null keeps the root), otherwise it is the source's own symbol (None if the name has none).
    """
    rec = _read_own_sidecar(src.with_name(src.name + ".json"))
    if rec is not None and rec.get("output_sha256") == src_sha and rec.get("root_symbol"):
        return str(rec["root_symbol"])
    from data_pipeline.parquet_manager import parse_parquet_filename
    try:
        return parse_parquet_filename(src)[0]
    except ValueError:
        return None


def publish(out: Path, sidecar: Path, out_df: pd.DataFrame, record: dict) -> dict:
    """Write the output and its sidecar so that a failure leaves the old pair as it was.

    Both go to temp files first; then the sidecar is replaced, then the output. If replacing
    the output fails (e.g. the file is open in another program), the old sidecar is put back
    (or the new one removed), so a sidecar never describes a file that was not written.
    """
    tmp_out = out.with_name(out.name + ".tmp")
    tmp_side = sidecar.with_name(sidecar.name + ".tmp")
    old_side = sidecar.read_bytes() if sidecar.is_file() else None
    try:
        out_df.to_parquet(tmp_out, index=False)
        record["output_sha256"] = _sha256(tmp_out)
        record = _plain(record)
        tmp_side.write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        os.replace(tmp_side, sidecar)
        try:
            os.replace(tmp_out, out)
        except OSError:
            try:
                if old_side is None:
                    sidecar.unlink()
                else:
                    tmp_side.write_bytes(old_side)
                    os.replace(tmp_side, sidecar)
            except OSError as e2:
                raise DataError(f"could not replace {out}, and could not restore its old sidecar ({e2}): "
                                f"{sidecar.name} now describes a file that was NOT written; delete it by "
                                "hand") from e2
            raise
    finally:
        for tmp in (tmp_out, tmp_side):
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
    return record


# ---------------------------------------------------------------------------------------
# main

def build(mode: str, src_text: str, out_text: str, seed: int | None, target_sharpe: float | None,
          overwrite: bool, calibrate_on: str = "net") -> dict:
    """Build one file; returns the sidecar record. Raises DataError for usage/data problems."""
    src, out, sidecar = check_paths(src_text, out_text, overwrite)
    df, ts, vol_col = load_source(src)
    src_sha = _sha256(src)
    from data_pipeline.parquet_manager import parse_parquet_filename
    out_symbol = parse_parquet_filename(out)[0]
    root = root_symbol(src, src_sha)
    if root is not None and out_symbol.lower() == root.lower():
        raise DataError(f"the output symbol ({out_symbol}) is the symbol of the real data this file comes "
                        f"from; give the null its own name (e.g. FLIP1_H1.parquet), so run_trial.py never "
                        "logs it as real data")
    if mode == "signflip" and len(df) < MIN_SIGNFLIP_BARS:
        raise DataError(f"file too short for signflip ({len(df)} bars; need at least {MIN_SIGNFLIP_BARS}): "
                        "smaller blocks cannot keep the block drift and |g|, |b| to float precision")
    o = df["open"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    lo = df["low"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    vol = df[vol_col].to_numpy()
    n = len(df)
    g, b, uw, lw = decompose(o, h, lo, c)
    src_ohlc = ohlc_check(o, h, lo, c)
    params: dict = {}
    diag: dict = {}
    problems: list[str] = []

    if mode == "shuffle":
        g2, b2, uw2, lw2, vol2 = shuffle_bars(g, b, uw, lw, vol, seed)
        params = {"bar_order": "numpy default_rng(seed).permutation of whole bars",
                  "same_as": f"make_placebo.py <source> <output> --mode shuffle --seed {seed}"}
    elif mode == "signflip":
        g2, b2, uw2, lw2, info = signflip_bars(g, b, uw, lw, seed)
        vol2 = vol
        params = {"blocks": N_BLOCKS, "block_rule": "T // 5 bars per block, remainder to the last block",
                  "flip_probability": 0.5}
    else:
        from config import Config
        from model_core.backtest import estimate_periods_per_year
        cost_rate = float(Config.COST_RATE)
        ppy = int(estimate_periods_per_year(ts.to_numpy()))
        sigma = float(np.std(g + b))
        if not sigma > 0:
            raise DataError("the base file has no variation in g+b")
        measure = f"{calibrate_on}_sharpe"
        print(f"plant: calibrating c for {n} bars (one pass per try; this can take a minute) ...", flush=True)
        c_val, stats, (b2, s, pos), cal = calibrate_plant(g, b, sigma, ppy, cost_rate, target_sharpe, measure)
        g2, uw2, lw2, vol2 = g, uw, lw, vol
        params = {"target_sharpe": target_sharpe, "calibrate_on": calibrate_on, "momentum_bars": MOMENTUM_BARS,
                  "cost_rate": cost_rate, "periods_per_year": ppy, "sigma_g_plus_b": sigma,
                  "seed_note": "plant has no randomness; the seed is recorded only"}
        if stats[measure] is None or abs(stats[measure] - target_sharpe) > SHARPE_TOL:
            problems.append(f"oracle {calibrate_on} Sharpe {_fmt(stats[measure])} is not within {SHARPE_TOL} "
                            f"of the target {target_sharpe}")
        tr_plan = target_returns_from_parts(g, b2)
        folds = fold_bounds(n)
        hyst = {}
        for band in HYSTERESIS_BANDS:
            hp = hysteresis_position(g + b2, band)
            hs = strategy_stats(hp, tr_plan, ppy, cost_rate)
            hf = fold_stats(hp, tr_plan, ppy, cost_rate, folds)
            hyst[f"band_{band:g}_sd"] = {"net_sharpe": hs["net_sharpe"], "gross_sharpe": hs["gross_sharpe"],
                                         "mean_abs_dpos_per_bar": hs["mean_abs_dpos_per_bar"],
                                         "all_val_net_sharpe": hf["all_val_net_sharpe"],
                                         "val_fold_net_sharpe": [r["net_sharpe"] for r in hf["folds"]]}
        diag.update({"c": c_val, "c_note": "planted drift per bar in units of sigma (std of g+b of the base)",
                     "planted_drift_per_bar": c_val * sigma,
                     "oracle_gross_sharpe": stats["gross_sharpe"], "oracle_net_sharpe": stats["net_sharpe"],
                     "oracle_ic": position_ic(pos, tr_plan),
                     "oracle_mean_abs_dpos_per_bar": stats["mean_abs_dpos_per_bar"],
                     "oracle_turnover_per_year": stats["turnover_per_year"],
                     "oracle_share_in_market": stats["share_bars_in_market"],
                     "oracle_share_long": stats["share_long"], **cal,
                     "oracle_by_val_fold": fold_stats(pos, tr_plan, ppy, cost_rate, folds),
                     "val_folds_note": "AlphaMaster walk-forward folds (5 folds, ModelConfig.WF_GAP); stats on "
                                       "one continuous run sliced to each validation fold",
                     "low_turnover_reference": hyst,
                     "low_turnover_reference_note": f"position = previous sign until |{MOMENTUM_BARS}-bar log "
                                                    f"return| > band x its rolling {HYSTERESIS_SD_BARS}-bar sd; "
                                                    "the oracle is not an upper bound"})

    out_o, out_h, out_l, out_c = rebuild(o[0], g2, b2, uw2, lw2)
    out_df = pd.DataFrame({
        "time": ts.astype("int64"),
        "open": out_o, "high": out_h, "low": out_l, "close": out_c,
        "tick_volume": vol2.astype("float64"),  # Dukascopy volumes are fractional; int64 would zero them
    })

    # diagnostics on the prices exactly as they will be saved
    out_ohlc = ohlc_check(out_o, out_h, out_l, out_c)
    if out_ohlc["nonfinite_bars"] or out_ohlc["nonpositive_bars"]:
        problems.append("the output has non-finite or non-positive prices")
    if (out_ohlc["high_below_body"] + out_ohlc["low_above_body"]
            > src_ohlc["high_below_body"] + src_ohlc["low_above_body"]):
        problems.append("the output has more high/low violations than the source")
    g3, b3, _, _ = decompose(out_o, out_h, out_l, out_c)
    diag.update({
        "bars": n,
        "first_time_utc": datetime.fromtimestamp(int(ts.iloc[0]), timezone.utc).isoformat(),
        "last_time_utc": datetime.fromtimestamp(int(ts.iloc[-1]), timezone.utc).isoformat(),
        "lag1_autocorr_b_before": lag1_autocorr(b), "lag1_autocorr_b_after": lag1_autocorr(b3),
        "lag1_autocorr_abs_b_before": lag1_autocorr(np.abs(b)), "lag1_autocorr_abs_b_after": lag1_autocorr(np.abs(b3)),
        "total_log_return_before": float(np.log(c[-1] / o[0])),
        "total_log_return_after": float(np.log(out_c[-1] / out_o[0])),
        "ohlc_source": src_ohlc, "ohlc_output": out_ohlc,
    })

    if mode == "signflip":
        flip = info["flipped"]
        dev_g = dev_b = 0.0
        drift_rows = []
        for blk in info["blocks"]:
            s0, e0 = blk["start"], blk["end"]
            dev_g = max(dev_g, float(np.max(np.abs(np.abs(g3[s0:e0] - blk["mean_g"])
                                                    - np.abs(g[s0:e0] - blk["mean_g"])))))
            dev_b = max(dev_b, float(np.max(np.abs(np.abs(b3[s0:e0] - blk["mean_b"])
                                                    - np.abs(b[s0:e0] - blk["mean_b"])))))
            before = float(np.sum(g[s0:e0] + b[s0:e0]))
            after = float(np.sum(g3[s0:e0] + b3[s0:e0]))
            scale = max(abs(before), float(np.mean(np.abs(g[s0:e0] + b[s0:e0]))))
            drift_rows.append({"block": blk["block"], "start": s0, "end": e0, "sum_g_plus_b_before": before,
                               "sum_g_plus_b_after": after, "rel_diff": abs(after - before) / scale,
                               "share_flipped": float(flip[s0:e0].mean()),
                               "repair_flips": blk["repair_flips"],
                               "drift_change_before_repair": blk["drift_change_before_repair"],
                               "residual_spread_per_bar": blk["residual_spread_per_bar"]})
        max_rel = max(r["rel_diff"] for r in drift_rows)
        diag.update({"share_flipped": float(flip.mean()), "share_flipped_by_coin": info["random_flips"] / n,
                     "repair_flips_total": int(sum(r["repair_flips"] for r in drift_rows)),
                     "max_abs_change_dev_g": dev_g, "max_abs_change_dev_b": dev_b,
                     "block_drift_max_rel_diff": max_rel, "blocks": drift_rows,
                     "dev_tolerance": DEV_TOL, "drift_rel_tolerance": DRIFT_REL_TOL,
                     "drift_rel_diff_note": "|after - before| / max(|block sum|, mean |g+b| per bar)"})
        if dev_g > DEV_TOL or dev_b > DEV_TOL:
            problems.append(f"|g - block mean| or |b - block mean| changed by {max(dev_g, dev_b):.3g} "
                            f"(limit {DEV_TOL}); the blocks ({n // N_BLOCKS} bars) were too small to cancel "
                            "the block drift precisely, so use a longer file")
        if max_rel > DRIFT_REL_TOL:
            problems.append(f"a block's sum of g+b changed by {max_rel:.3g} relative (limit {DRIFT_REL_TOL})")
    elif mode == "plant":
        # end-to-end: recompute the signal and the oracle from the prices as saved
        r3 = g3 + b3
        cs = np.r_[0.0, np.cumsum(r3)]
        win = cs[MOMENTUM_BARS:] - cs[:-MOMENTUM_BARS]          # window sums ending at bar t (t >= 23)
        pos3 = np.zeros(n)
        pos3[MOMENTUM_BARS - 1:] = np.sign(win)
        tr3 = np.zeros(n)
        tr3[:n - 2] = np.log(out_o[2:] / out_o[1:-1])
        o32 = out_o.astype(np.float32)
        tr32 = np.zeros(n)
        tr32[:n - 2] = np.log(o32[2:] / o32[1:-1]).astype("float64")
        diag.update({
            "oracle_position_agreement_from_file": float((pos3 == pos).mean()),
            "oracle_net_sharpe_from_file": strategy_stats(pos, tr3, ppy, cost_rate)["net_sharpe"],
            "oracle_net_sharpe_float32_open": strategy_stats(pos, tr32, ppy, cost_rate)["net_sharpe"],
            "max_abs_change_g": float(np.max(np.abs(g3 - g))),
            "max_abs_change_wicks": float(max(np.max(np.abs(np.log(out_h / np.maximum(out_o, out_c)) - uw)),
                                              np.max(np.abs(np.log(np.minimum(out_o, out_c) / out_l) - lw)))),
        })

    if problems:
        raise DataError("checks failed, nothing written: " + "; ".join(problems))

    record = {
        "label": "research only",
        "tool": TOOL,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "mode": mode,
        "seed": seed,
        "parameters": params,
        "source_file": str(src),
        "source_sha256": src_sha,
        "root_symbol": root,
        "output_file": str(out),
        "output_sha256": None,             # filled in by publish() from the bytes written
        "sidecar_file": str(sidecar),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "platform": platform.platform()},
        "diagnostics": diag,
    }
    try:
        import pyarrow
        record["versions"]["pyarrow"] = pyarrow.__version__
    except ImportError:
        pass
    return publish(out, sidecar, out_df, record)


def print_summary(rec: dict) -> None:
    d = rec["diagnostics"]
    seed = "none" if rec["seed"] is None else rec["seed"]
    print(f"make_null {rec['mode']} (seed {seed}): {d['bars']} bars, {d['first_time_utc'][:10]} .. "
          f"{d['last_time_utc'][:10]}")
    print(f"  source : {_ascii(rec['source_file'])}")
    print(f"           sha256 {rec['source_sha256']}")
    print(f"  output : {_ascii(rec['output_file'])}")
    print(f"           sha256 {rec['output_sha256']}")
    if rec["mode"] == "signflip":
        print(f"  bars flipped           : {d['share_flipped']:.1%} (coin flips {d['share_flipped_by_coin']:.1%}, "
              f"{d['repair_flips_total']} extra flips to keep the block drift)")
        print(f"  |g - block mean| change: max {d['max_abs_change_dev_g']:.2e}   (must be ~0)")
        print(f"  |b - block mean| change: max {d['max_abs_change_dev_b']:.2e}   (must be ~0)")
        print(f"  block drift (sum g+b)  : max relative change {d['block_drift_max_rel_diff']:.2e} "
              f"(limit {DRIFT_REL_TOL:g})")
        for r in d["blocks"]:
            print(f"    block {r['block']} bars [{r['start']}, {r['end']}): sum before {r['sum_g_plus_b_before']:+.6f} "
                  f"after {r['sum_g_plus_b_after']:+.6f}, flipped {r['share_flipped']:.1%}")
    if rec["mode"] == "plant":
        p = rec["parameters"]
        print(f"  planted signal         : c = {d['c']:.6f} sigma (drift {d['planted_drift_per_bar']:.3e} per bar, "
              f"sigma {p['sigma_g_plus_b']:.3e})")
        print(f"  oracle Sharpe (annual) : net {_fmt(d['oracle_net_sharpe'])}, gross {_fmt(d['oracle_gross_sharpe'])} "
              f"(calibrated on {p['calibrate_on']}, target {p['target_sharpe']}); IC {_fmt(d['oracle_ic'])}; "
              f"{p['periods_per_year']} bars/year, cost {p['cost_rate']}")
        print(f"  oracle turnover        : {d['oracle_mean_abs_dpos_per_bar']:.4f} |dpos| per bar "
              f"({d['oracle_turnover_per_year']:.0f} per year), in market {d['oracle_share_in_market']:.1%}")
        vf = d["oracle_by_val_fold"]
        print("  oracle net by val fold : " + ", ".join(_fmt(r["net_sharpe"], "+.2f") for r in vf["folds"])
              + f" (all validation bars {_fmt(vf['all_val_net_sharpe'], '+.3f')})")
        for name, r in d["low_turnover_reference"].items():
            band = name.replace("band_", "").replace("_sd", " sd")
            print(f"  low-turnover ref {band:>6}: net {_fmt(r['net_sharpe'])} at {r['mean_abs_dpos_per_bar']:.4f} "
                  f"|dpos| per bar (all validation bars {_fmt(r['all_val_net_sharpe'], '+.3f')})")
        print(f"  check from saved file  : net Sharpe {_fmt(d['oracle_net_sharpe_from_file'])} "
              f"(float32 opens {_fmt(d['oracle_net_sharpe_float32_open'])}), positions agree "
              f"{d['oracle_position_agreement_from_file']:.4%}")
        print("  The oracle is a reference, not an upper bound: rules that trade less can net more from")
        print("  this signal, so a miner result above the oracle is possible. For a power curve, build a")
        print("  ladder of weaker plants with --calibrate-on gross (see 'make_null.py plant --help').")
    print(f"  lag-1 autocorr b       : before {_fmt(d['lag1_autocorr_b_before'])}, after {_fmt(d['lag1_autocorr_b_after'])}")
    print(f"  lag-1 autocorr |b|     : before {_fmt(d['lag1_autocorr_abs_b_before'])}, "
          f"after {_fmt(d['lag1_autocorr_abs_b_after'])}")
    oc = d["ohlc_output"]
    print(f"  OHLC valid             : {'yes' if oc['valid'] else 'NO'} (high below body {oc['high_below_body']}, "
          f"low above body {oc['low_above_body']}, non-positive {oc['nonpositive_bars']}; source valid: "
          f"{'yes' if d['ohlc_source']['valid'] else 'NO'})")
    if not d["ohlc_source"]["valid"]:
        print("  WARNING: the source already has OHLC problems (see ohlc_source in the sidecar); the output "
              "keeps them.")
    print(f"  sidecar: {_ascii(rec['sidecar_file'])}")


def _parser() -> argparse.ArgumentParser:
    fmt = argparse.RawDescriptionHelpFormatter
    ap = argparse.ArgumentParser(
        prog="make_null.py", formatter_class=fmt,
        description="Build research data files (placebo, sign-flip null, positive control) in "
                    "AlphaMaster's Parquet schema. Research only.",
        epilog="examples (from the repo root; $D is the folder that holds the data):\n"
               "  python scripts\\research\\make_null.py shuffle  $D\\XAUUSD_H1.parquet  $D\\PLACEBO_H1.parquet --seed 7\n"
               "  python scripts\\research\\make_null.py signflip $D\\XAUUSD_H1.parquet  $D\\FLIP1_H1.parquet --seed 1\n"
               "  python scripts\\research\\make_null.py plant    $D\\PLACEBO_H1.parquet $D\\PLANT1_H1.parquet "
               "--target-sharpe 1.0\n\n"
               "Run 'make_null.py <mode> --help' for the details of a mode. The output name must look like\n"
               "SYMBOL_TF.parquet (e.g. FLIP1_H1.parquet). A sidecar <output>.json records the mode, seed,\n"
               "sha256 of both files and the diagnostics. --overwrite replaces only a file this tool made\n"
               "(its sidecar must match it), never real data. Exit code 0 = file written, 2 = usage or data\n"
               "error (nothing written). Paths with 'locked_holdout' or ending in '.locked' are refused.")
    sub = ap.add_subparsers(dest="mode", metavar="{shuffle,signflip,plant}")
    sub.required = True

    def add_common(p, seed_required: bool,
                   source_help: str = "source Parquet file (time, open, high, low, close, tick_volume)"):
        p.add_argument("source", help=source_help)
        p.add_argument("output", help="output file, named SYMBOL_TF.parquet (e.g. FLIP1_H1.parquet)")
        if seed_required:
            p.add_argument("--seed", type=int, required=True, help="random seed (the same seed gives the same file)")
        else:
            p.add_argument("--seed", type=int, default=None, help="recorded in the sidecar only")
        p.add_argument("--overwrite", action="store_true",
                       help="replace an existing output and sidecar made by this tool (never other files)")

    add_common(sub.add_parser("shuffle", formatter_class=fmt, description=SHUFFLE_HELP,
                              help="placebo: shuffle the bar order (as make_placebo.py)"), True)
    add_common(sub.add_parser("signflip", formatter_class=fmt, description=SIGNFLIP_HELP,
                              help="random-direction null that keeps the volatility structure"), True)
    p = sub.add_parser("plant", formatter_class=fmt, description=PLANT_HELP,
                       help="positive control: plant a known momentum signal into a base file")
    add_common(p, False, "base Parquet file (intended: the seed-7 placebo, e.g. PLACEBO_H1.parquet)")
    p.add_argument("--target-sharpe", type=float, default=1.0,
                   help="annualised Sharpe of the oracle (default 1.0; net of costs unless --calibrate-on gross)")
    p.add_argument("--calibrate-on", choices=("net", "gross"), default="net",
                   help="calibrate c on the oracle's Sharpe after costs (net, default) or before costs "
                        "(gross; use gross for a ladder of weaker plants)")
    return ap


def main(argv: list[str] | None = None) -> int:
    print(BANNER, flush=True)
    ap = _parser()
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    if a.mode == "plant" and not (math.isfinite(a.target_sharpe) and a.target_sharpe > 0):
        print("error: --target-sharpe must be a positive number", file=sys.stderr)
        return 2
    if a.seed is not None and a.seed < 0:
        print("error: --seed must be 0 or a positive whole number", file=sys.stderr)
        return 2
    try:
        rec = build(a.mode, a.source, a.output, a.seed,
                    a.target_sharpe if a.mode == "plant" else None, a.overwrite,
                    a.calibrate_on if a.mode == "plant" else "net")
    except DataError as e:
        print(f"error: {_ascii(e)}", file=sys.stderr)
        return 2
    except OSError as e:  # e.g. the output file is open in another program
        print(f"error: could not read or write a file (the output and its sidecar were left as they were): "
              f"{_ascii(e)}", file=sys.stderr)
        return 2
    print_summary(rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
