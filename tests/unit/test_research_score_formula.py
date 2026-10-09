"""Tests for scripts/research/score_formula.py (one-formula scorer). Research only.

Synthetic data only: a seeded random-walk SYN_H1.parquet written to a temp folder.
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
import re
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "research" / "score_formula.py"

_spec = importlib.util.spec_from_file_location("research_score_formula", SCRIPT)
sf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sf)

T_BARS = 3000
HANDBUILT = ([0], [3, 120], [0, 1, 66], [11, 69, 69])  # RET; TS_ZSCORE_10(MA_DIFF); RET-RET5; NEG(NEG(RSI14))


def _write_synth(path: Path, n: int = T_BARS, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.002, n)
    close = 2000 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, n)))
    t = 1609459200 + 3600 * np.arange(n)
    pd.DataFrame({"time": t.astype("int64"), "open": open_, "high": high, "low": low, "close": close,
                  "tick_volume": rng.integers(100, 2000, n)}).to_parquet(path)
    return path


@pytest.fixture(scope="module")
def synth_file(tmp_path_factory) -> Path:
    return _write_synth(tmp_path_factory.mktemp("score_formula") / "SYN_H1.parquet")


@pytest.fixture(scope="module")
def setup(synth_file):
    return sf.prepare(synth_file)


def _independent_engine(path: Path, capsys):
    """Engine built the way train_file does; folds and bars/year read back from engine.train's own output."""
    from data_pipeline.parquet_manager import ParquetDataManager
    from model_core.engine import AlphaEngine

    mgr = ParquetDataManager(str(path))
    mgr.load()
    engine = AlphaEngine(data_manager=mgr, target_symbol="SYN")
    capsys.readouterr()
    engine.train(start_step=0, end_step=0)  # prints the folds, sets bt.periods_per_year, trains nothing
    out = capsys.readouterr().out
    folds = [{"train_start": int(a), "train_end": int(b), "gap": int(g), "val_start": int(c), "val_end": int(d)}
             for a, b, g, c, d in re.findall(r"\[(\d+),(\d+)\) \S+=(\d+) \S+\[(\d+),(\d+)\)", out)]
    assert len(folds) == 4, out
    return mgr, engine, folds


# -- item 1 + 2: folds, annualisation and engine score are the engine's --------

def test_folds_and_bars_per_year_match_engine_train(setup, synth_file, capsys):
    _, engine, folds = _independent_engine(synth_file, capsys)
    assert setup.use_wf
    assert [{k: f[k] for k in ("train_start", "train_end", "gap", "val_start", "val_end")} for f in setup.folds] == folds
    assert setup.periods_per_year == engine.bt.periods_per_year
    assert setup.cost_rate == engine.bt.cost_rate


@pytest.mark.parametrize("formula", HANDBUILT)
def test_engine_score_bit_equal_to_engine(setup, synth_file, capsys, formula):
    mgr, engine, folds = _independent_engine(synth_file, capsys)
    r = engine._eval_formula_task(0, list(formula), mgr.feat_tensor, mgr.target_ret, folds, True, [])
    rep, res = sf.engine_score(setup, formula)
    assert r["status"] == rep["status"] == "ok"
    assert rep["val_score"] == r["val_score"]
    assert rep["train_reward"] == r["reward"]
    assert rep["reproduced_matches_engine"] is True
    assert torch.equal(res, r["res"])
    assert rep["repetition_penalty"] == (0.3 if formula == [11, 69, 69] else 0.0)


def test_engine_score_bit_equal_to_a_real_training_step(setup, synth_file, tmp_path, monkeypatch, capsys):
    """Run one real engine.train step, record every formula it scored, re-score them with the tool."""
    from data_pipeline.parquet_manager import ParquetDataManager
    from model_core.engine import AlphaEngine

    monkeypatch.chdir(tmp_path)  # the engine writes checkpoints/ and strategies/ relative to cwd
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    seen = []
    original = AlphaEngine._eval_formula_task

    def recording(self, idx, fml, *args, **kwargs):
        r = original(self, idx, fml, *args, **kwargs)
        seen.append((list(fml), r["status"], r["val_score"], r["reward"]))
        return r

    monkeypatch.setattr(AlphaEngine, "_eval_formula_task", recording)
    mgr = ParquetDataManager(str(synth_file))
    mgr.load()
    engine = AlphaEngine(data_manager=mgr, target_symbol="SYN")
    engine.timeframe, engine.data_file, engine.mode = "H1", str(synth_file), "parquet_file"
    engine.train(start_step=0, end_step=1)
    monkeypatch.setattr(AlphaEngine, "_eval_formula_task", original)

    ok = []
    for fml, status, val, reward in seen:
        if status == "ok" and fml not in [o[0] for o in ok]:
            ok.append((fml, val, reward))
    assert len(ok) >= 3
    for fml, val, reward in ok[:6]:
        rep, _ = sf.engine_score(setup, fml)
        assert rep["val_score"] == val, fml
        assert rep["train_reward"] == reward, fml
        assert rep["reproduced_matches_engine"] is True
    assert engine.best_formula is not None
    best, _ = sf.engine_score(setup, engine.best_formula)
    assert best["val_score"] == engine.best_score
    assert best["checks"]["would_be_accepted_as_best"] is True

    # the same run, logged like scripts/run_trial.py, re-scored through the command line
    log = tmp_path / "trials.jsonl"
    log.write_text(json.dumps({"run_id": "r1", "tag": "real", "seed": 1, "status": "completed",
                               "git_commit": sf._git_commit(), "data_file": str(synth_file),
                               "data_sha256": sf._sha256(synth_file),
                               "best_validation_score": float(engine.best_score), "unique_formulas": 123,
                               "best_formula": engine.best_formula}) + "\n", encoding="utf-8")
    out = tmp_path / "r1.json"
    capsys.readouterr()
    assert sf.main(["--from-trial", "r1", "--trials-log", str(log), "--json", str(out)]) == 0
    flat = " ".join(capsys.readouterr().out.split())
    assert "best of the 123 distinct formulas that run r1 tried" in flat and "IN-SAMPLE WARNING" in flat
    trial = json.loads(out.read_text(encoding="utf-8"))["inputs"]["trial"]
    assert trial["logged_vs_recomputed"].startswith("equal, bit for bit")
    assert trial["data_sha256_matches"] is True


def test_constant_factor_scores_minus_two(setup):
    rep, _ = sf.engine_score(setup, [], factor=sf.constant_factor(setup.t_ret))
    assert rep["status"] == "const" and rep["val_score"] == -2.0


# -- item 5: one-bar-delay check ----------------------------------------------

def test_delay_shifts_one_bar_and_uses_only_past_values():
    g = torch.Generator().manual_seed(0)
    f = torch.randn(1, 50, generator=g)
    d = sf.delay_one_bar(f)
    assert d[0, 0].item() == 0.0
    assert torch.equal(d[0, 1:], f[0, :-1])
    for k in (1, 10, 49):
        g2 = f.clone()
        g2[:, k:] += 5.0  # change the present and the future only
        assert torch.equal(sf.delay_one_bar(g2)[:, :k + 1], d[:, :k + 1])


def test_delay_changes_the_engine_score(setup):
    rep, res = sf.engine_score(setup, [0])
    rep_d, res_d = sf.engine_score(setup, [0], factor=sf.delay_one_bar(res))
    assert rep_d["status"] == "ok"
    assert rep_d["val_score"] != rep["val_score"]
    assert rep_d["repetition_penalty"] == rep["repetition_penalty"]
    assert torch.equal(res_d[:, 1:], res[:, :-1])


# -- holdout refusal ----------------------------------------------------------

def test_is_locked_path():
    assert sf.is_locked_path(r"C:\data\locked_holdout\XAUUSD_H1.parquet")
    assert sf.is_locked_path("/x/LOCKED_HOLDOUT_2025/XAUUSD_H1.parquet")
    assert sf.is_locked_path("XAUUSD_H1.parquet.locked")
    assert sf.is_locked_path("XAUUSD_H1.parquet.LOCKED")
    assert not sf.is_locked_path(r"..\AlphaMaster\research\data\XAUUSD_H1.parquet")
    assert not sf.is_locked_path("locked/XAUUSD_H1.parquet")


def test_holdout_paths_refused_before_reading(tmp_path, monkeypatch, capsys):
    def must_not_run(*a, **k):
        raise AssertionError("a locked file was opened")

    monkeypatch.setattr(sf, "_sha256", must_not_run)
    monkeypatch.setattr(sf, "prepare", must_not_run)
    monkeypatch.chdir(tmp_path)
    locked_dir = tmp_path / "locked_holdout"
    locked_dir.mkdir()
    paths = [locked_dir / "XAUUSD_H1.parquet", tmp_path / "XAUUSD_H1.parquet.locked"]
    for p in paths:
        p.write_bytes(b"not a parquet file")
        assert sf.main(["--data-file", str(p), "--formula", "[0]"]) == 2
        assert "locked holdout" in capsys.readouterr().err
    log = tmp_path / "trials.jsonl"
    log.write_text(json.dumps({"run_id": "h1", "best_formula": [0], "data_file": str(paths[0])}) + "\n",
                   encoding="utf-8")
    assert sf.main(["--from-trial", "h1", "--trials-log", str(log)]) == 2
    assert "locked holdout" in capsys.readouterr().err
    # --data-file overriding a trial's (unlocked) file is checked too; the override's name parses as
    # SYMBOL_TF.parquet, so only the lock check can refuse it (a .locked name would also fail the parser)
    log.write_text(json.dumps({"run_id": "h2", "best_formula": [0], "data_file": str(tmp_path / "SYN_H1.parquet")})
                   + "\n", encoding="utf-8")
    assert sf.main(["--from-trial", "h2", "--trials-log", str(log), "--data-file", str(paths[0])]) == 2
    assert "locked holdout" in capsys.readouterr().err


# -- item 4 and 6b: plain statistics by hand ----------------------------------

POS = [0.0, 0.5, 0.5, -0.2, 0.0, 1.0]
RET = [0.01, -0.02, 0.03, 0.01, -0.01, 0.02]


def _hand_stats(pnl, ppy):
    mean = sum(pnl) / len(pnl)
    std = statistics.pstdev(pnl)
    eq, peak, mdd = 0.0, 0.0, 0.0
    for x in pnl:
        eq += x
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return mean * ppy, mean / std * math.sqrt(ppy), mdd, sum(pnl)


def test_plain_stats_match_hand_computation():
    c = 0.001 * 2  # cost rate 0.001, multiplier 2
    # PnL[t] = p[t]*r[t] - |p[t]-p[t-1]|*c with p[-1] = 0
    hand = [0.0, 0.5 * -0.02 - 0.5 * c, 0.5 * 0.03, -0.2 * 0.01 - 0.7 * c, -0.2 * c, 1.0 * 0.02 - 1.0 * c]
    got = sf.pnl_series(np.array(POS), np.array(RET), 0.001, 2.0)
    assert np.allclose(got, hand, rtol=0, atol=1e-15)

    folds = [{"val_start": 0, "val_end": 2}, {"val_start": 2, "val_end": 6}]
    out = sf.plain_val_stats(np.array(POS), np.array(RET), folds, [2.0], 0.001, 100)["x2"]
    f2 = out["folds"][1]
    ann, sharpe, mdd, total = _hand_stats(hand[2:6], 100)
    assert f2["bars"] == 4
    assert f2["ann_return"] == pytest.approx(ann, abs=1e-12) and ann == pytest.approx(0.73)
    assert f2["sharpe"] == pytest.approx(sharpe, rel=1e-12)
    assert f2["max_drawdown"] == pytest.approx(mdd, abs=1e-15) and mdd == pytest.approx(0.0038)
    assert f2["total_log_return"] == pytest.approx(total, abs=1e-15)
    assert f2["direction_changes"] == 3  # long -> short -> flat -> long (bar 2 stays long like bar 1)
    assert out["folds"][0]["direction_changes"] == 1  # flat start, flat -> long
    a_ann, a_sharpe, a_mdd, _ = _hand_stats(hand, 100)
    assert out["all_val"]["ann_return"] == pytest.approx(a_ann, abs=1e-12)
    assert out["all_val"]["sharpe"] == pytest.approx(a_sharpe, rel=1e-12)
    assert out["all_val"]["max_drawdown"] == pytest.approx(a_mdd, abs=1e-15)
    assert out["all_val"]["direction_changes"] == 4


def test_direction_changes_ignore_size_changes_on_the_same_side():
    # continuous tanh positions change size almost every bar; only side changes count
    pos = np.array([0.3, 0.4, 0.5, 0.0, -0.2, -0.6, 0.7])
    assert sf.direction_changes(pos, 0, 7) == 4  # entry long, exit, entry short, reversal to long
    assert sf.direction_changes(pos, 1, 3) == 0  # 0.3 -> 0.4 -> 0.5 stays long
    assert sf.direction_changes(pos, 4, 6) == 1  # flat (bar 3) -> short, then short stays short
    st = sf.plain_val_stats(pos, np.zeros(7), [{"val_start": 0, "val_end": 7}], [1.0], 0.001, 100)["x1"]
    assert st["folds"][0]["direction_changes"] == 4 and "position_changes" not in st["folds"][0]


def test_buy_and_hold_matches_hand_computation():
    c = 0.001 * 1.5
    folds = [{"val_start": 0, "val_end": 3}, {"val_start": 3, "val_end": 6}]
    out = sf.buy_and_hold_stats(np.array(RET), folds, [1.5], 0.001, 252)["x1.5"]
    hand1 = [0.01 - c, -0.02, 0.03]
    hand2 = [0.01 - c, -0.01, 0.02]
    for got, hand in ((out["folds"][0], hand1), (out["folds"][1], hand2), (out["all_val"], hand1 + hand2)):
        ann, sharpe, mdd, total = _hand_stats(hand, 252)
        assert got["ann_return"] == pytest.approx(ann, abs=1e-12)
        assert got["sharpe"] == pytest.approx(sharpe, rel=1e-12)
        assert got["max_drawdown"] == pytest.approx(mdd, abs=1e-15)
        assert got["total_log_return"] == pytest.approx(total, abs=1e-15)
    assert out["folds"][0]["max_drawdown"] == pytest.approx(0.02)
    assert out["folds"][1]["max_drawdown"] == pytest.approx(0.01)
    assert [f["direction_changes"] for f in out["folds"]] == [1, 1]


# -- item 3 and 6a helpers ----------------------------------------------------

def test_trailing_vol_is_causal_and_correct():
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    v = sf.trailing_vol(close, 24)
    r = np.diff(np.log(close))
    assert np.isnan(v[:24]).all()
    assert v[24] == pytest.approx(statistics.stdev(r[0:24]), rel=1e-12)
    assert v[150] == pytest.approx(statistics.stdev(r[150 - 24:150]), rel=1e-12)
    c2 = close.copy()
    c2[100:] *= 1.5
    assert np.array_equal(sf.trailing_vol(c2, 24)[:100], v[:100], equal_nan=True)


def test_position_window_stats_shares():
    pos = np.array([0.0, 0.5, -0.5, 0.0, 0.2, 0.2])
    vol = np.full(6, np.nan)
    d = sf.position_window_stats(pos, 2, 6, vol)
    assert (d["share_long"], d["share_short"], d["share_flat"]) == (0.5, 0.25, 0.25)
    assert d["mean_abs_position"] == pytest.approx(0.225)
    assert d["turnover_per_bar"] == pytest.approx((1.0 + 0.5 + 0.2 + 0.0) / 4)
    assert d["corr_abs_position_trailing_vol"] is None


def test_momentum_factor_is_normalised_log_ratio(setup):
    from model_core.vm import StackVM
    close = setup.close
    raw = torch.zeros_like(close)
    c = close[0].double().numpy()
    raw[0, 24:] = torch.from_numpy(np.log(c[24:] / c[:-24])).float()
    mom = sf.momentum_factor(close, 24)
    assert torch.allclose(mom, StackVM._normalize_output(raw), atol=1e-5)
    rep, _ = sf.engine_score(setup, [], factor=mom)
    assert rep["status"] == "ok" and rep["reproduced_matches_engine"] is True


# -- command line -------------------------------------------------------------

def test_cli_writes_report(synth_file, setup, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out" / "report.json"
    code = sf.main(["--data-file", str(synth_file), "--formula", "[3,", "120]", "--benchmarks",
                    "--costs", "1", "2", "--json", str(out)])
    text = capsys.readouterr().out
    assert code == 0
    assert text.splitlines()[0] == "RESEARCH ONLY - not trading advice"
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["inputs"]["formula"] == [3, 120]
    assert rep["inputs"]["cost_multipliers"] == [1.0, 2.0]
    assert rep["engine"]["val_score"] == sf.engine_score(setup, [3, 120])[0]["val_score"]
    assert set(rep["plain_val_stats"]) == {"x1", "x2"}
    assert len(rep["folds"]) == 4 and len(rep["positions"]["val_folds"]) == 4
    assert rep["delay_check"]["engine"]["status"] == "ok"
    assert rep["benchmarks"]["buy_and_hold"]["engine_constant_factor"]["val_score"] == -2.0
    assert rep["benchmarks"]["momentum_24"]["engine"]["status"] == "ok"
    flat = " ".join(text.split())
    # a mined formula's validation numbers are in-sample for its selection: the report must say so
    assert "IN-SAMPLE WARNING" in flat and "only the locked holdout test is out of sample" in flat
    assert "out-of-sample Sortino gate" not in flat
    assert rep["notes"][0] == sf.SELECTION_NOTE
    # the delay note judges the drop against selected noise-arm formulas, with an unselected reference
    assert "noise-arm best formulas" in sf.DELAY_NOTE and "best of many" in sf.DELAY_NOTE
    mom_delay = rep["benchmarks"]["momentum_24"]["delay_check"]
    assert mom_delay["status"] == "ok" and isinstance(mom_delay["change"], float)
    assert "one-bar-delay check on this unselected factor" in flat
    assert "dir.changes" in text and "pos.changes" not in text


@pytest.mark.parametrize("args, needle", [
    (["--formula", "abc"], "whole numbers"),
    (["--formula", "[999]"], "outside the vocabulary"),
    (["--formula", "[66]"], "does not execute"),
    (["--formula", "[0]", "--costs", "-1"], "0 or more"),
])
def test_usage_errors_exit_2(synth_file, tmp_path, monkeypatch, capsys, args, needle):
    monkeypatch.chdir(tmp_path)
    assert sf.main(["--data-file", str(synth_file)] + args) == 2
    assert needle in capsys.readouterr().err


def test_missing_inputs_exit_2(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert sf.main(["--data-file", str(tmp_path / "NOPE_H1.parquet"), "--formula", "[0]"]) == 2
    assert "not found" in capsys.readouterr().err
    log = tmp_path / "trials.jsonl"
    log.write_text(json.dumps({"run_id": "a", "best_formula": [0], "data_file": "x"}) + "\n", encoding="utf-8")
    assert sf.main(["--from-trial", "zzz", "--trials-log", str(log)]) == 2
    assert "not in" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        sf.main(["--formula", "[0]"])  # no --data-file and no --from-trial
    assert e.value.code == 2


def test_compare_logged():
    assert sf.compare_logged(1.5, 1.5, 0.8).startswith("equal, bit for bit")
    assert sf.compare_logged(1.5 + 1e-7, 1.5, 0.8).startswith("equal within float rounding")
    assert sf.compare_logged(1.5 - 0.3, 1.5, 0.8).startswith("lower by one correlation penalty")
    assert sf.compare_logged(-1.0 - 0.2, -1.0, 0.8).startswith("lower by one correlation penalty")
    assert sf.compare_logged(1.0, 1.5, 0.8).startswith("lower, but")
    assert sf.compare_logged(2.0, 1.5, 0.8).startswith("HIGHER")
    assert "no best" in sf.compare_logged(None, 1.5, 0.8)


# -- best-update checks under the run's correlation penalty (review finding 1) -----

@pytest.mark.parametrize("formula", [[3, 120], [11, 69, 69]])
def test_corr_penalised_score_is_the_engines_own(setup, formula):
    """With a correlated factor in the pool the engine's _eval_formula_task gives exactly these values."""
    from model_core.config import ModelConfig

    rep, res = sf.engine_score(setup, formula)
    pen = sf.corr_penalised_score(setup, formula, res, rep["checks"]["exposure"])
    assert setup.engine.factor_pool == []  # restored for the next scoring
    eng = setup.engine
    eng.factor_pool = [(1.0, 0, res.clone())]  # an earlier best that correlates 1.0 with this factor
    try:
        r = eng._eval_formula_task(0, formula, setup.feat, setup.t_ret, setup.folds, setup.use_wf,
                                   list(eng.factor_pool))
    finally:
        eng.factor_pool = []
    assert pen["status"] == "ok"
    assert pen["val_score"] == r["val_score"] and pen["train_reward"] == r["reward"]
    k = 1.0 - ModelConfig.CORR_PENALTY
    assert pen["val_score"] == pytest.approx(rep["val_score"] - abs(rep["val_score"]) * k, rel=1e-6)
    assert pen["train_reward"] == pytest.approx(rep["train_reward"] - abs(rep["train_reward"]) * k, rel=1e-6)
    assert pen["checks"]["train_reward"] == pen["train_reward"] and pen["checks"]["val_score"] == pen["val_score"]
    assert rep["checks"]["factor_pool"] == "empty"


def test_overfit_verdict_can_flip_under_the_penalty():
    # the reviewer's case: formula [26,118,5,98,110,6,97,72] on the 6000-bar synthetic file
    empty = sf.best_checks(0.5510, 0.0130, 0.5, pool="empty")
    run = sf.best_checks(0.5510 * 0.8, 0.0130 * 0.8, 0.5, pool="one correlated factor")
    assert empty["overfit_check_passed"] is False and run["overfit_check_passed"] is True


def test_from_trial_with_a_penalised_logged_score_prints_the_runs_checks(setup, synth_file, tmp_path,
                                                                         monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rep, res = sf.engine_score(setup, [3, 120])
    pen = sf.corr_penalised_score(setup, [3, 120], res, rep["checks"]["exposure"])
    log = tmp_path / "trials.jsonl"
    log.write_text(json.dumps({"run_id": "p1", "tag": "real", "seed": 1, "status": "completed",
                               "data_file": str(synth_file), "best_validation_score": pen["val_score"],
                               "best_formula": [3, 120]}) + "\n", encoding="utf-8")
    out = tmp_path / "p1.json"
    capsys.readouterr()
    assert sf.main(["--from-trial", "p1", "--trials-log", str(log), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "includes one correlation penalty, so engine.train made its checks on the penalised values" in text
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["inputs"]["trial"]["logged_vs_recomputed"].startswith("lower by one correlation penalty, bit for bit")
    assert data["engine"]["with_one_corr_penalty"]["checks"]["val_score"] == pen["val_score"]


# -- constant formulas: no penalty shown as applied, no delay rescoring (findings 3 and 4) --

def test_constant_formula_report(synth_file, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "const.json"
    assert sf.main(["--data-file", str(synth_file), "--formula", "[0,0,66]", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["engine"]["status"] == "const" and data["engine"]["val_score"] == -2.0
    assert data["engine"]["repetition_penalty"] == pytest.approx(0.3)
    assert data["engine"]["repetition_penalty_applied"] is False
    assert "not applied (status const" in text
    assert "skipped" in data["delay_check"] and "skipped: the formula's output is constant" in text


def test_nonzero_constant_factor_is_not_turned_into_a_scored_one_by_the_delay(synth_file, tmp_path,
                                                                              monkeypatch):
    """A non-zero constant delayed with bar 0 = 0 is no longer constant and would get an ordinary score."""
    real_prepare = sf.prepare

    def constant_vm(path):
        s = real_prepare(path)
        s.engine.vm = sf._FixedFactorVM(torch.full_like(s.t_ret, 0.7))
        return s

    monkeypatch.setattr(sf, "prepare", constant_vm)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "c07.json"
    assert sf.main(["--data-file", str(synth_file), "--formula", "[0]", "--json", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["engine"]["status"] == "const"
    assert "engine" not in data["delay_check"] and "skipped" in data["delay_check"]


# -- --json output path (finding 7) --------------------------------------------

def test_json_path_refused_before_any_work(synth_file, tmp_path, monkeypatch, capsys):
    def must_not_run(*a, **k):
        raise AssertionError("work started although --json was refused")

    monkeypatch.setattr(sf, "prepare", must_not_run)
    monkeypatch.setattr(sf, "_sha256", must_not_run)
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "trials.json"  # a trial log whose name ends in .json
    log_text = json.dumps({"run_id": "j1", "best_formula": [0], "data_file": str(synth_file)}) + "\n"
    log.write_text(log_text, encoding="utf-8")
    (tmp_path / "dir.json").mkdir()
    (tmp_path / "locked_holdout").mkdir()
    cases = [
        (["--data-file", str(synth_file), "--formula", "[0]", "--json", str(tmp_path / "report.txt")], "ending in .json"),
        (["--from-trial", "j1", "--trials-log", str(log), "--json", str(log)], "overwrite the trial log"),
        (["--data-file", str(synth_file), "--formula", "[0]", "--json", str(tmp_path / "locked_holdout" / "r.json")],
         "locked holdout"),
        (["--data-file", str(synth_file), "--formula", "[0]", "--json", str(tmp_path / "r.json.locked")], "locked holdout"),
        (["--data-file", str(synth_file), "--formula", "[0]", "--json", str(sf.ROOT / "strategies" / "sf_test.json")],
         "inside the repo but not under logs"),
        (["--data-file", str(synth_file), "--formula", "[0]", "--json", str(tmp_path / "dir.json")], "folder"),
    ]
    for args, needle in cases:
        assert sf.main(args) == 2, args
        assert needle in capsys.readouterr().err, args
    assert log.read_text(encoding="utf-8") == log_text
    assert not (sf.ROOT / "strategies" / "sf_test.json").exists()


def test_check_json_path_rules(tmp_path):
    data, log = tmp_path / "XAUUSD_H1.parquet", tmp_path / "t.jsonl"
    ok = sf.check_json_path(str(sf.ROOT / "logs" / "score.json"), data, log)
    assert ok == (sf.ROOT / "logs" / "score.json").resolve()
    assert sf.check_json_path(str(tmp_path / "a" / "b.JSON"), data, log).name == "b.JSON"
    with pytest.raises(sf.UsageError, match="data file"):
        sf.check_json_path(str(tmp_path / "d.json"), tmp_path / "d.json", log)
    with pytest.raises(sf.UsageError, match="inside the repo"):
        sf.check_json_path(str(sf.ROOT / "score.json"), data, log)
