"""Tests for scripts/research/make_null.py (placebo, sign-flip null and positive control). Research only."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "research" / "make_null.py"

_spec = importlib.util.spec_from_file_location("research_make_null", SCRIPT)
mn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mn)

# The real make_placebo.py, when it can be found (it is not part of this repo).
_PLACEBO_CANDIDATES = [os.environ.get("MAKE_PLACEBO_PY", ""),
                       "/mnt/project-files/alphamaster/tools/make_placebo.py"]
MAKE_PLACEBO = next((Path(p) for p in _PLACEBO_CANDIDATES if p and Path(p).is_file()), None)


def _source(path: Path, n: int = 3000, seed: int = 0, start: int = 1609459200) -> Path:
    """Synthetic H1 bars in AlphaMaster's schema, with weekend gaps and small intraday gaps."""
    rng = np.random.default_rng(seed)
    times, t = [], start
    while len(times) < n:
        if (t // 86400 + 4) % 7 < 5:        # Monday..Friday
            times.append(t)
        t += 3600
    times = np.array(times, dtype=np.int64)
    gap = rng.normal(0.0, 0.0002, n)
    weekend = np.r_[False, np.diff(times) > 3600]
    gap[weekend] = rng.normal(0.0, 0.004, weekend.sum())
    body = rng.standard_t(5, n) * 0.0015 + 0.00003
    o, c = np.empty(n), np.empty(n)
    last = 1800.0
    for i in range(n):
        o[i] = last * math.exp(gap[i])
        c[i] = o[i] * math.exp(body[i])
        last = c[i]
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0.0, 0.001, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0.0, 0.001, n)))
    pd.DataFrame({"time": times, "open": o, "high": h, "low": lo, "close": c,
                  "tick_volume": rng.integers(100, 5000, n), "spread": 25}).to_parquet(path, index=False)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(capsys, *args) -> tuple[int, str, str]:
    code = mn.main([str(a) for a in args])
    out = capsys.readouterr()
    return code, out.out, out.err


def _parts(path: Path):
    df = pd.read_parquet(path).sort_values("time").reset_index(drop=True)
    o, h, lo, c = (df[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    prev_c = np.r_[o[0], c[:-1]]
    return df, np.log(o / prev_c), np.log(c / o), np.log(h / np.maximum(o, c)), np.log(np.minimum(o, c) / lo)


def _reference_make_placebo_shuffle(src: Path, dst: Path, seed: int) -> None:
    """Frozen copy of make_placebo.py's shuffle mode (the tool's reference)."""
    df = pd.read_parquet(src).sort_values("time").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    c = df["close"].to_numpy(dtype="float64")
    o = df["open"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    lo = df["low"].to_numpy(dtype="float64")
    prev_c = np.r_[o[0], c[:-1]]
    r_open = np.log(o / prev_c)
    r_close = np.log(c / o)
    r_high = np.log(h / np.maximum(o, c))
    r_low = np.log(np.minimum(o, c) / lo)
    n = len(df)
    idx = rng.permutation(n)
    r_open, r_close, r_high, r_low = r_open[idx], r_close[idx], r_high[idx], r_low[idx]
    vol = df["tick_volume"].to_numpy()[idx]
    out_o = np.empty(n); out_c = np.empty(n)  # noqa: E702
    last = o[0]
    for i in range(n):
        out_o[i] = last * np.exp(r_open[i])
        out_c[i] = out_o[i] * np.exp(r_close[i])
        last = out_c[i]
    out_h = np.maximum(out_o, out_c) * np.exp(r_high)
    out_l = np.minimum(out_o, out_c) * np.exp(-r_low)
    out = pd.DataFrame({
        "time": df["time"].astype("int64"),
        "open": out_o, "high": out_h, "low": out_l, "close": out_c,
        "tick_volume": vol.astype("float64"),
    })
    out.to_parquet(dst, index=False)


# ---------------------------------------------------------------------------------------
# shuffle

def test_shuffle_matches_frozen_make_placebo(tmp_path, capsys):
    src = _source(tmp_path / "XAUUSD_H1.parquet")
    ref = tmp_path / "REF_H1.parquet"
    _reference_make_placebo_shuffle(src, ref, 7)
    code, out, _ = _run(capsys, "shuffle", src, tmp_path / "PLACEBO_H1.parquet", "--seed", 7)
    assert code == 0
    assert out.splitlines()[0] == mn.BANNER
    assert (tmp_path / "PLACEBO_H1.parquet").read_bytes() == ref.read_bytes()
    side = json.loads((tmp_path / "PLACEBO_H1.parquet.json").read_text(encoding="utf-8"))
    assert side["mode"] == "shuffle" and side["seed"] == 7
    assert side["source_sha256"] == _sha(src)
    assert side["output_sha256"] == _sha(tmp_path / "PLACEBO_H1.parquet")


@pytest.mark.skipif(MAKE_PLACEBO is None, reason="make_placebo.py not found (set MAKE_PLACEBO_PY to its path)")
def test_shuffle_matches_make_placebo_script(tmp_path, capsys):
    src = _source(tmp_path / "XAUUSD_H1.parquet", n=2500, seed=5)
    for seed in (7, 11):
        ref = tmp_path / f"REF{seed}_H1.parquet"
        subprocess.run([sys.executable, str(MAKE_PLACEBO), str(src), str(ref), "--seed", str(seed)],
                       check=True, capture_output=True)
        out = tmp_path / f"PLAC{seed}_H1.parquet"
        assert _run(capsys, "shuffle", src, out, "--seed", seed)[0] == 0
        assert out.read_bytes() == ref.read_bytes()


def test_output_schema(tmp_path, capsys):
    src = _source(tmp_path / "XAUUSD_H1.parquet")
    out = tmp_path / "FLIP1_H1.parquet"
    assert _run(capsys, "signflip", src, out, "--seed", 1)[0] == 0
    df = pd.read_parquet(out)
    assert list(df.columns) == ["time", "open", "high", "low", "close", "tick_volume"]
    assert str(df["time"].dtype) == "int64"
    assert all(str(df[k].dtype) == "float64" for k in ("open", "high", "low", "close", "tick_volume"))


# ---------------------------------------------------------------------------------------
# signflip

def test_signflip_invariants(tmp_path, capsys):
    src = _source(tmp_path / "XAUUSD_H1.parquet", n=5003, seed=3)
    out = tmp_path / "FLIP1_H1.parquet"
    code, text, _ = _run(capsys, "signflip", src, out, "--seed", 1)
    assert code == 0, text
    s_df, g, b, uw, lw = _parts(src)
    o_df, g2, b2, uw2, lw2 = _parts(out)
    n = len(g)
    assert (o_df["time"].to_numpy() == s_df["time"].to_numpy()).all()
    assert (o_df["tick_volume"].to_numpy() == s_df["tick_volume"].to_numpy().astype("float64")).all()
    size = n // 5
    edges = [0, size, 2 * size, 3 * size, 4 * size, n]                  # remainder in the last block
    flipped = np.zeros(n, dtype=bool)
    flipped_g = np.zeros(n, dtype=bool)
    clear = np.zeros(n, dtype=bool)               # bars whose demeaned g and b are both clearly non-zero
    for s, e in zip(edges[:-1], edges[1:]):
        mg, mb = g[s:e].mean(), b[s:e].mean()
        assert np.max(np.abs(np.abs(g2[s:e] - mg) - np.abs(g[s:e] - mg))) < 1e-12
        assert np.max(np.abs(np.abs(b2[s:e] - mb) - np.abs(b[s:e] - mb))) < 1e-12
        before, after = np.sum(g[s:e] + b[s:e]), np.sum(g2[s:e] + b2[s:e])
        assert abs(after - before) <= 1e-9 * abs(before)
        flipped[s:e] = np.sign(b2[s:e] - mb) == -np.sign(b[s:e] - mb)
        flipped_g[s:e] = np.sign(g2[s:e] - mg) == -np.sign(g[s:e] - mg)
        clear[s:e] = (np.abs(g[s:e] - mg) > 1e-9) & (np.abs(b[s:e] - mb) > 1e-9)
    # the gap is negated together with the body on every flipped bar (spec: negate BOTH g and b)
    assert clear.mean() > 0.99
    assert (flipped_g[clear] == flipped[clear]).all()
    assert 0.45 < flipped_g[clear].mean() < 0.55
    assert not flipped[0] and not flipped_g[0]                         # bar 0 is never flipped
    # flipped bars have swapped wicks, the others keep theirs
    assert np.allclose(uw2[flipped], lw[flipped], atol=1e-12, rtol=0)
    assert np.allclose(lw2[flipped], uw[flipped], atol=1e-12, rtol=0)
    assert np.allclose(uw2[~flipped], uw[~flipped], atol=1e-12, rtol=0)
    assert 0.45 < flipped.mean() < 0.55
    # the direction is randomised: bodies now disagree with the source about half the time
    agree = np.mean(np.sign(b2) == np.sign(b))
    assert 0.4 < agree < 0.6
    side = json.loads((tmp_path / "FLIP1_H1.parquet.json").read_text(encoding="utf-8"))
    d = side["diagnostics"]
    assert abs(d["share_flipped"] - flipped.mean()) < 0.01
    assert d["max_abs_change_dev_g"] < 1e-12 and d["max_abs_change_dev_b"] < 1e-12
    assert d["block_drift_max_rel_diff"] < 1e-9
    assert d["ohlc_output"]["valid"] and d["ohlc_output"]["bars"] == n
    for key in ("lag1_autocorr_b_before", "lag1_autocorr_b_after",
                "lag1_autocorr_abs_b_before", "lag1_autocorr_abs_b_after"):
        assert isinstance(d[key], float)
    assert [r["end"] - r["start"] for r in d["blocks"]] == [1000, 1000, 1000, 1000, 1003]
    o, h, lo, c = (o_df[k].to_numpy() for k in ("open", "high", "low", "close"))
    assert (h >= np.maximum(o, c)).all() and (lo <= np.minimum(o, c)).all() and (lo > 0).all()


def test_signflip_seeds_differ_and_repeat(tmp_path, capsys):
    src = _source(tmp_path / "XAUUSD_H1.parquet")
    (tmp_path / "again").mkdir()
    a, b_, a2 = tmp_path / "FLIP1_H1.parquet", tmp_path / "FLIP2_H1.parquet", tmp_path / "again" / "FLIP1_H1.parquet"
    assert _run(capsys, "signflip", src, a, "--seed", 1)[0] == 0
    assert _run(capsys, "signflip", src, b_, "--seed", 2)[0] == 0
    assert _run(capsys, "signflip", src, a2, "--seed", 1)[0] == 0
    assert a.read_bytes() == a2.read_bytes()
    assert a.read_bytes() != b_.read_bytes()
    c1, c2 = pd.read_parquet(a)["close"].to_numpy(), pd.read_parquet(b_)["close"].to_numpy()
    assert np.mean(np.sign(np.diff(c1)) == np.sign(np.diff(c2))) < 0.75


# ---------------------------------------------------------------------------------------
# plant

def test_plant_oracle_sharpe_long_series(tmp_path, capsys):
    from model_core.backtest import estimate_periods_per_year

    base = _source(tmp_path / "BASE_H1.parquet", n=40000, seed=11)
    out = tmp_path / "PLANT1_H1.parquet"
    code, text, _ = _run(capsys, "plant", base, out, "--target-sharpe", 1.0)
    assert code == 0, text
    b_df, g, b, uw, lw = _parts(base)
    p_df, g2, b2, uw2, lw2 = _parts(out)
    n = len(g)
    # only the bodies change, by exactly +/- the planted drift
    assert np.max(np.abs(g2 - g)) < 1e-12
    assert np.max(np.abs(uw2 - uw)) < 1e-12 and np.max(np.abs(lw2 - lw)) < 1e-12
    side = json.loads((tmp_path / "PLANT1_H1.parquet.json").read_text(encoding="utf-8"))
    d = side["diagnostics"]
    k = d["planted_drift_per_bar"]
    assert k > 0
    assert np.all(np.isclose(np.abs(b2 - b), 0, atol=1e-12) | np.isclose(np.abs(b2 - b), k, atol=1e-12))
    # independent oracle: position[t] = sign of the 24-bar sum of g+b ending at t (= s[t+1])
    r = g2 + b2
    cs = np.r_[0.0, np.cumsum(r)]
    pos = np.zeros(n)
    pos[23:] = np.sign(cs[24:] - cs[:-24])
    # the drift on bar t follows the sign of the 24 bars before t, read from the saved file
    s_file = np.r_[0.0, pos[:-1]]
    assert np.allclose(b2 - b, k * s_file, atol=1e-12, rtol=0)
    o = p_df["open"].to_numpy()
    tr = np.zeros(n)
    tr[:n - 2] = np.log(o[2:] / o[1:-1])
    prev = np.r_[0.0, pos[:-1]]
    pnl = pos * tr - np.abs(pos - prev) * 0.0003
    ppy = estimate_periods_per_year(p_df["time"].to_numpy())
    sharpe = pnl.mean() / pnl.std(ddof=1) * math.sqrt(ppy)
    assert abs(sharpe - 1.0) < 0.05
    assert abs(d["oracle_net_sharpe"] - 1.0) < 0.05
    assert d["oracle_gross_sharpe"] > d["oracle_net_sharpe"]
    assert d["oracle_position_agreement_from_file"] > 0.999
    assert side["parameters"]["periods_per_year"] == ppy
    assert d["ohlc_output"]["valid"]
    # turnover-free strength and per-fold figures (the oracle is a reference, not an upper bound)
    from model_core.config import ModelConfig
    from model_core.engine import _build_walk_forward_folds

    assert abs(d["c"] * side["parameters"]["sigma_g_plus_b"] - k) < 1e-15
    ok = np.arange(n) >= 23
    ok[n - 2:] = False
    assert abs(d["oracle_ic"] - np.corrcoef(pos[ok], tr[ok])[0, 1]) < 1e-6 and d["oracle_ic"] > 0
    folds = _build_walk_forward_folds(n, 5, ModelConfig.WF_GAP)
    rows = d["oracle_by_val_fold"]["folds"]
    assert [(r["val_start"], r["val_end"]) for r in rows] == [(f["val_start"], f["val_end"]) for f in folds]
    for r in rows:
        x = pnl[r["val_start"]:r["val_end"]]
        assert abs(r["net_sharpe"] - x.mean() / x.std(ddof=1) * math.sqrt(ppy)) < 0.01
    for name in ("band_0.1_sd", "band_0.2_sd"):
        ref = d["low_turnover_reference"][name]
        assert ref["mean_abs_dpos_per_bar"] < d["oracle_mean_abs_dpos_per_bar"]
        assert len(ref["val_fold_net_sharpe"]) == len(folds)
    assert "best possible" not in text and "not an upper bound" in text
    # gross calibration (for a ladder of weaker plants) hits the gross target instead
    out_g = tmp_path / "PLANTG1_H1.parquet"
    code, text, _ = _run(capsys, "plant", base, out_g, "--calibrate-on", "gross", "--target-sharpe", 0.5)
    assert code == 0, text
    dg = json.loads((tmp_path / "PLANTG1_H1.parquet.json").read_text(encoding="utf-8"))
    assert dg["parameters"]["calibrate_on"] == "gross"
    assert abs(dg["diagnostics"]["oracle_gross_sharpe"] - 0.5) < 0.05
    assert dg["diagnostics"]["c"] < d["c"]


def test_plant_is_causal():
    rng = np.random.default_rng(4)
    n, m = 3000, 1500
    g = rng.normal(0, 0.0002, n)
    b = rng.normal(0, 0.0015, n)
    k = 0.0002
    b1, s1, pos1 = mn.plant_series(g, b, k)
    # s[t] is the sign of the sum of the NEW g+b over the 24 bars before t, and 0 before bar 24
    r1 = g + b1
    assert (s1[:24] == 0).all()
    for t in range(24, n):
        assert s1[t] == np.sign(np.sum(r1[t - 24:t]))
    assert (pos1[:-1] == s1[1:]).all()
    # the drift lands on the same bar as its signal: b_new[t] = b[t] + k * s[t] (not s[t-1])
    assert np.allclose(b1 - b, k * s1, atol=1e-15, rtol=0)
    # change a future bar: nothing before it may move
    b_mod = b.copy()
    b_mod[m] += 0.05
    b2, s2, pos2 = mn.plant_series(g, b_mod, k)
    assert (s2[:m + 1] == s1[:m + 1]).all()
    assert (pos2[:m] == pos1[:m]).all()
    assert (b2[:m] == b1[:m]).all()
    assert (s2[m + 1:] != s1[m + 1:]).any()          # the change does reach later bars


def test_plant_repeat_is_identical(tmp_path, capsys):
    base = _source(tmp_path / "BASE_H1.parquet", n=4000, seed=2)
    (tmp_path / "again").mkdir()
    a, a2 = tmp_path / "PLANT1_H1.parquet", tmp_path / "again" / "PLANT1_H1.parquet"
    assert _run(capsys, "plant", base, a, "--seed", 7)[0] == 0
    assert _run(capsys, "plant", base, a2, "--seed", 7)[0] == 0
    assert a.read_bytes() == a2.read_bytes()


def test_shuffle_deterministic_by_seed(tmp_path, capsys):
    src = _source(tmp_path / "XAUUSD_H1.parquet")
    (tmp_path / "again").mkdir()
    a, a2, b_ = (tmp_path / "PLAC_H1.parquet", tmp_path / "again" / "PLAC_H1.parquet",
                 tmp_path / "PLAC8_H1.parquet")
    assert _run(capsys, "shuffle", src, a, "--seed", 7)[0] == 0
    assert _run(capsys, "shuffle", src, a2, "--seed", 7)[0] == 0
    assert _run(capsys, "shuffle", src, b_, "--seed", 8)[0] == 0
    assert a.read_bytes() == a2.read_bytes() != b_.read_bytes()


# ---------------------------------------------------------------------------------------
# refusals and errors

def test_refuses_holdout_paths(tmp_path, capsys, monkeypatch):
    def no_read(*_a, **_k):
        raise AssertionError("a holdout path was read")

    monkeypatch.setattr(mn.pd, "read_parquet", no_read)
    monkeypatch.setattr(mn, "_sha256", no_read)
    good = tmp_path / "FLIP1_H1.parquet"
    cases = [
        (tmp_path / "locked_holdout" / "XAUUSD_H1.parquet", good),
        (tmp_path / "data" / "XAUUSD_H1.parquet.locked", good),
        (tmp_path / "LOCKED_HOLDOUT" / "XAUUSD_H1.parquet", good),
        (tmp_path / "XAUUSD_H1.parquet", tmp_path / "locked_holdout" / "FLIP1_H1.parquet"),
    ]
    for src, out in cases:
        code, _, err = _run(capsys, "signflip", src, out, "--seed", 1)
        assert code == 2
        assert "locked holdout" in err
    assert not good.exists()


def test_refuses_bad_output_names(tmp_path, capsys):
    src = _source(tmp_path / "XAUUSD_H1.parquet")
    (tmp_path / "other").mkdir()
    bad = [tmp_path / "flip.parquet", tmp_path / "FLIP1_H1.csv", tmp_path / "FLIP1_XX.parquet",
           src, tmp_path / "other" / "XAUUSD_H1.parquet", tmp_path / "FLIP1_D1.parquet",
           tmp_path / "missing_folder" / "FLIP1_H1.parquet"]
    for out in bad:
        code, _, err = _run(capsys, "signflip", src, out, "--seed", 1)
        assert code == 2, out
        assert err.startswith("error:")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["XAUUSD_H1.parquet", "other"]
    assert list((tmp_path / "other").iterdir()) == []


def test_never_overwrites_without_flag(tmp_path, capsys):
    src = _source(tmp_path / "XAUUSD_H1.parquet")
    out = tmp_path / "FLIP1_H1.parquet"
    assert _run(capsys, "signflip", src, out, "--seed", 1)[0] == 0
    first = out.read_bytes()
    code, _, err = _run(capsys, "signflip", src, out, "--seed", 2)
    assert code == 2 and "--overwrite" in err
    assert out.read_bytes() == first
    assert _run(capsys, "signflip", src, out, "--seed", 2, "--overwrite")[0] == 0
    assert out.read_bytes() != first
    assert json.loads((tmp_path / "FLIP1_H1.parquet.json").read_text(encoding="utf-8"))["seed"] == 2


def test_usage_and_data_errors(tmp_path, capsys):
    assert _run(capsys, "--help")[0] == 0
    code, out, _ = _run(capsys, "plant", "--help")
    assert code == 0 and out.splitlines()[0] == mn.BANNER and "not an upper bound" in out.lower()
    assert "best possible" not in out and "--calibrate-on" in out
    assert _run(capsys, "signflip", tmp_path / "XAUUSD_H1.parquet", tmp_path / "FLIP1_H1.parquet")[0] == 2  # no seed
    assert _run(capsys, "signflip", tmp_path / "NOPE_H1.parquet", tmp_path / "FLIP1_H1.parquet",
                "--seed", 1)[0] == 2
    src = _source(tmp_path / "XAUUSD_H1.parquet", n=200)
    df = pd.read_parquet(src)
    df.drop(columns=["high"]).to_parquet(tmp_path / "NOHIGH_H1.parquet", index=False)
    pd.concat([df, df.iloc[[5]]]).to_parquet(tmp_path / "DUP_H1.parquet", index=False)
    for name in ("NOHIGH_H1.parquet", "DUP_H1.parquet"):
        code, _, err = _run(capsys, "shuffle", tmp_path / name, tmp_path / "PLAC_H1.parquet", "--seed", 1)
        assert code == 2 and err.startswith("error:")
    assert _run(capsys, "plant", src, tmp_path / "PLANT1_H1.parquet", "--target-sharpe", -1)[0] == 2
    # a time column of text is a data error (exit 2), not a traceback
    txt = df.copy()
    txt["time"] = pd.to_datetime(txt["time"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    txt.to_parquet(tmp_path / "TXT_H1.parquet", index=False)
    code, _, err = _run(capsys, "shuffle", tmp_path / "TXT_H1.parquet", tmp_path / "PLAC_H1.parquet", "--seed", 1)
    assert code == 2 and "time column" in err, err
    assert not (tmp_path / "PLAC_H1.parquet").exists()
    assert not (tmp_path / "PLANT1_H1.parquet").exists()


def test_signflip_refuses_short_files(tmp_path, capsys):
    short = _source(tmp_path / "XAUUSD_H1.parquet", n=mn.MIN_SIGNFLIP_BARS - 1, seed=3)
    code, _, err = _run(capsys, "signflip", short, tmp_path / "FLIP1_H1.parquet", "--seed", 1)
    assert code == 2 and "too short for signflip" in err
    assert not (tmp_path / "FLIP1_H1.parquet").exists()
    (tmp_path / "ok").mkdir()
    enough = _source(tmp_path / "ok" / "XAUUSD_H1.parquet", n=mn.MIN_SIGNFLIP_BARS, seed=3)
    assert _run(capsys, "signflip", enough, tmp_path / "ok" / "FLIP1_H1.parquet", "--seed", 1)[0] == 0


def test_overwrite_only_replaces_own_files(tmp_path, capsys):
    real = _source(tmp_path / "XAUUSD_H1.parquet")
    real_bytes = real.read_bytes()
    flip = tmp_path / "FLIP1_H1.parquet"
    assert _run(capsys, "signflip", real, flip, "--seed", 1)[0] == 0
    # swapped arguments: the real file has no sidecar from this tool, so it is never replaced
    for extra in ((), ("--overwrite",)):
        code, _, err = _run(capsys, "signflip", flip, real, "--seed", 2, *extra)
        assert code == 2 and err.startswith("error:"), err
    assert "not made by this tool" in err
    assert real.read_bytes() == real_bytes
    assert not (tmp_path / "XAUUSD_H1.parquet.json").exists()
    # a sidecar from this tool that no longer matches the file (the file was replaced by hand)
    flip_side = tmp_path / "FLIP1_H1.parquet.json"
    flip.write_bytes(real_bytes)
    code, _, err = _run(capsys, "signflip", real, flip, "--seed", 2, "--overwrite")
    assert code == 2 and "not made by this tool" in err
    assert flip.read_bytes() == real_bytes
    # a null may not carry the real symbol, even in another folder or through a chain of nulls
    (tmp_path / "other").mkdir()
    for src, out in ((real, tmp_path / "other" / "XAUUSD_1h.parquet"),
                     (tmp_path / "PLAC_H1.parquet", tmp_path / "other" / "XAUUSD_H1.parquet")):
        if src.name == "PLAC_H1.parquet":
            assert _run(capsys, "shuffle", real, src, "--seed", 7)[0] == 0
            assert json.loads(Path(str(src) + ".json").read_text(encoding="utf-8"))["root_symbol"] == "XAUUSD"
        code, _, err = _run(capsys, "plant" if src.name == "PLAC_H1.parquet" else "signflip", src, out,
                            "--seed", 1)
        assert code == 2 and "symbol of the real data" in err, err
    assert list((tmp_path / "other").iterdir()) == []
    assert json.loads(flip_side.read_text(encoding="utf-8"))["tool"] == mn.TOOL


def test_failed_write_leaves_old_pair(tmp_path, capsys, monkeypatch):
    src = _source(tmp_path / "XAUUSD_H1.parquet")
    out, side = tmp_path / "FLIP1_H1.parquet", tmp_path / "FLIP1_H1.parquet.json"
    assert _run(capsys, "signflip", src, out, "--seed", 1)[0] == 0
    old_out, old_side = out.read_bytes(), side.read_bytes()
    real_replace = os.replace

    def locked(suffix):
        def replace(a, b):
            if str(b).endswith(suffix):
                raise PermissionError(13, "the file is open in another program", str(b))
            return real_replace(a, b)
        return replace

    for suffix in (".parquet", ".json"):          # the output, then the sidecar, is locked
        monkeypatch.setattr(mn.os, "replace", locked(suffix))
        code, _, err = _run(capsys, "signflip", src, out, "--seed", 2, "--overwrite")
        monkeypatch.setattr(mn.os, "replace", real_replace)
        assert code == 2 and "left as they were" in err, err
        assert out.read_bytes() == old_out and side.read_bytes() == old_side
        assert sorted(p.name for p in tmp_path.iterdir()) == ["FLIP1_H1.parquet", "FLIP1_H1.parquet.json",
                                                             "XAUUSD_H1.parquet"]
    # a fresh output whose write fails leaves no sidecar behind
    monkeypatch.setattr(mn.os, "replace", locked(".parquet"))
    code, _, _ = _run(capsys, "signflip", src, tmp_path / "FLIP2_H1.parquet", "--seed", 2)
    monkeypatch.setattr(mn.os, "replace", real_replace)
    assert code == 2
    assert not (tmp_path / "FLIP2_H1.parquet").exists() and not (tmp_path / "FLIP2_H1.parquet.json").exists()
    # after a successful overwrite the sidecar describes the new file
    assert _run(capsys, "signflip", src, out, "--seed", 2, "--overwrite")[0] == 0
    rec = json.loads(side.read_text(encoding="utf-8"))
    assert rec["seed"] == 2 and rec["output_sha256"] == _sha(out)
