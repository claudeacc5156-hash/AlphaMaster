"""Tests for the propkit command line (python -m propkit ...) and scripts/research/export_positions.py.
Research only; synthetic data only."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from propkit import adapters, cli, rules
from propkit.bars import synthetic_bars
from propkit.costs import CostModel

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_SPEC = ROOT / "propkit" / "examples" / "pullback_spec_example.json"
EXPORT_SCRIPT = ROOT / "scripts" / "research" / "export_positions.py"
HEADER = "RESEARCH ONLY - not trading advice"
FAST = ["--n-sims", "200", "--dd-sims", "100", "--history-reps", "4"]
FOUR = ("report.json", "report.md", "equity.csv", "trades.csv")


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """A synthetic H1 bar file with spread, a trades CSV and a positions CSV made from it."""
    d = tmp_path_factory.mktemp("cli_inputs")
    bars = synthetic_bars(1704067200, 3000, seed=9, spread=0.34)      # 2024-01-01, across the March DST changes
    bars_file = d / "XAUUSD_H1.parquet"
    bars.to_parquet(bars_file, index=False)
    cm = CostModel()
    t, o, s = bars["time"].to_numpy(), bars["open"].to_numpy(), cm.bar_spreads(bars)
    rows = []
    for k, i in enumerate(range(30, len(bars) - 20, 41)):
        side, j = (1 if k % 2 == 0 else -1), i + 13
        rows.append({"side": "long" if side > 0 else "short", "units": 25.0, "entry_time": int(t[i]),
                     "entry_price": cm.buy_fill(o[i], s[i]) if side > 0 else cm.sell_fill(o[i], s[i]),
                     "exit_time": int(t[j]),
                     "exit_price": cm.sell_fill(o[j], s[j]) if side > 0 else cm.buy_fill(o[j], s[j])})
    trades_file = d / "my_trades.csv"
    pd.DataFrame(rows).to_csv(trades_file, index=False)
    p = np.sign(np.sin(np.arange(len(bars)) / 50.0))
    pos_file = d / "positions.csv"
    adapters.write_positions_csv(adapters.positions_from_alphamaster(t, p, keep_raw=True), pos_file)
    return {"dir": d, "bars": bars_file, "trades": trades_file, "positions": pos_file, "frame": bars}


def run(args, capsys) -> tuple[int, str, str]:
    code = cli.main([str(a) for a in args])
    out, err = capsys.readouterr()
    assert out.isascii() and err.isascii()
    return code, out, err


# ---------------------------------------------------------------------------------------
# commands

def test_rules_prints_both_presets(capsys):
    code, out, _ = run(["rules"], capsys)
    assert code == 0 and out.startswith(HEADER)
    assert "--rules ftmo-1step" in out and "--rules ftmo-2step" in out
    assert "3.00% of initial capital" in out and "minimum trading days: 4" in out


def test_help_and_version_exit_0(capsys):
    assert run(["--help"], capsys)[0] == 0
    code, out, _ = run(["evaluate", "--help"], capsys)
    assert code == 0 and "--size-mode" in out and "--n-trials" in out
    assert run(["--version"], capsys)[0] == 0


def test_evaluate_trades_writes_the_report_files(files, tmp_path, capsys):
    out = tmp_path / "run1"
    code, stdout, err = run(["evaluate", "--bars", files["bars"], "--trades", files["trades"], "--rules", "ftmo-1step",
                             "--n-trials", "12", "--out", out] + FAST, capsys)
    assert code == 0, err
    for name in FOUR + ("days.csv",):
        assert (out / name).is_file() and (out / name).stat().st_size > 0, name
    assert "Files written:" in stdout and "P(pass)" in stdout
    rep = json.loads((out / "report.json").read_text(encoding="ascii"))
    md = (out / "report.md").read_text(encoding="ascii")
    assert md.splitlines()[0] == HEADER and rep["header"] == HEADER
    assert rep["input"]["kind"] == "trades" and rep["input"]["n_rows"] == len(pd.read_csv(files["trades"]))
    assert rep["stats"]["per_day"]["dsr"]["n_trials"] == 12
    assert rep["settings"] == {"n_sims": 200, "seed": 7, "alpha": 0.05, "horizon_days": 60, "horizon_unit": "trading",
                               "history_reps": 4, "n_trials": 12, "sr_var": None, "dd_sims": 100}
    hu = rep["history_uncertainty"]
    assert hu["n_reps"] == 4 and hu["p_pass"]["p5"] <= hu["p_pass"]["p50"] <= hu["p_pass"]["p95"]
    assert rep["blocks"]["n_blocks_days"] >= 1 and isinstance(rep["warnings"], list)
    assert "history uncertainty (outer bootstrap, 5-95%)" in stdout and "Monte Carlo error" in stdout
    assert "60 trading days" in md and "### History uncertainty" in md and "simulation noise only" in md
    eq = pd.read_csv(out / "equity.csv")
    assert len(eq) == len(files["frame"]) and {"time_utc", "prop_day"} <= set(eq.columns)
    # trades.csv written by propkit reads back as an input and gives the same account path
    out2 = tmp_path / "run2"
    code, _, err = run(["evaluate", "--bars", files["bars"], "--trades", out / "trades.csv", "--rules", "ftmo-1step",
                        "--out", out2] + FAST, capsys)
    assert code == 0, err
    rep2 = json.loads((out2 / "report.json").read_text(encoding="ascii"))
    assert rep2["path"]["final_balance"] == pytest.approx(rep["path"]["final_balance"], abs=1e-6)


@pytest.mark.parametrize("mode,size,horizon", [("units", "20", "40"), ("leverage", "0.5", "none")])
def test_evaluate_positions(files, tmp_path, capsys, mode, size, horizon):
    out = tmp_path / mode
    extra = ["--no-stress", "--n-sims", "50"] if horizon == "none" else []   # no horizon is slower
    code, _, err = run(["evaluate", "--bars", files["bars"], "--positions", files["positions"], "--size-mode", mode,
                        "--size", size, "--rules", "ftmo-2step", "--target", "0.05", "--capital", "50000",
                        "--costs", "flat0.0003", "--cost-mult", "1.5", "--horizon-days", horizon, "--out", out]
                       + FAST + extra, capsys)
    assert code == 0, err
    rep = json.loads((out / "report.json").read_text(encoding="ascii"))
    assert rep["input"]["size_mode"] == mode and rep["input"]["size"] == float(size)
    assert rep["rules"]["initial_capital"] == 50000 and rep["rules"]["profit_target_pct"] == 0.05
    assert rep["costs"]["model"]["flat_rate_per_side"] == pytest.approx(0.00045)
    assert rep["settings"]["horizon_days"] == (None if horizon == "none" else 40)
    assert (rep["stress"] is None) == (horizon == "none")
    assert all((out / name).is_file() for name in FOUR)


def test_pullback_with_the_example_spec(files, tmp_path, capsys):
    out = tmp_path / "pb"
    code, stdout, err = run(["pullback", "--bars", files["bars"], "--spec", EXAMPLE_SPEC, "--rules", "ftmo-1step",
                             "--out", out] + FAST, capsys)
    assert code == 0, err
    assert "PLACEHOLDER" in stdout
    for name in FOUR + ("days.csv", "decisions.csv"):
        assert (out / name).is_file(), name
    md = (out / "report.md").read_text(encoding="ascii")
    assert "WARNING: the pullback spec is a PLACEHOLDER" in md
    dec = pd.read_csv(out / "decisions.csv")
    assert {"signal_time", "status", "signal_time_utc"} <= set(dec.columns)
    rep = json.loads((out / "report.json").read_text(encoding="ascii"))
    assert rep["input"]["kind"] == "pullback" and rep["input"]["spec"]["placeholder"] is True


def test_selftest_command_passes(capsys):
    code, out, _ = run(["selftest"], capsys)
    lines = out.strip().splitlines()
    assert code == 0
    assert sum(line.startswith("PASS  ") for line in lines) == 29 and not any(line.startswith("FAIL") for line in lines)
    assert lines[-1].startswith("29 of 29 gates passed")


def test_python_dash_m_propkit_selftest_subprocess():
    proc = subprocess.run([sys.executable, "-m", "propkit", "selftest"], cwd=ROOT, capture_output=True, timeout=300)
    assert proc.returncode == 0, proc.stdout.decode("ascii", "replace") + proc.stderr.decode("ascii", "replace")
    out = proc.stdout.decode("ascii")                                  # ASCII only, or this raises
    assert "29 of 29 gates passed" in out


# ---------------------------------------------------------------------------------------
# refusals and usage errors (exit code 2)

def _locked_cases(files, tmp_path):
    lock = tmp_path / "locked_holdout"
    ok_out = tmp_path / "out_ok"
    base = ["--rules", "ftmo-1step"] + FAST
    return {
        "bars": ["evaluate", "--bars", lock / "XAUUSD_H1.parquet", "--trades", files["trades"], "--out", ok_out] + base,
        "bars windows text": ["evaluate", "--bars", "C:\\data\\Locked_Holdout\\XAUUSD_H1.parquet", "--trades",
                              files["trades"], "--out", ok_out] + base,
        "trades .locked": ["evaluate", "--bars", files["bars"], "--trades", tmp_path / "t.csv.locked", "--out",
                           ok_out] + base,
        "positions": ["evaluate", "--bars", files["bars"], "--positions", lock / "p.csv", "--size", "1", "--out",
                      ok_out] + base,
        "out": ["evaluate", "--bars", files["bars"], "--trades", files["trades"], "--out", lock / "run"] + base,
        "rules file": ["evaluate", "--bars", files["bars"], "--trades", files["trades"], "--rules", "custom",
                       "--rules-file", lock / "r.json", "--out", ok_out] + FAST,
        "costs file": ["evaluate", "--bars", files["bars"], "--trades", files["trades"], "--costs", lock / "c.json",
                       "--out", ok_out] + base,
        "spec": ["pullback", "--bars", files["bars"], "--spec", lock / "spec.json", "--out", ok_out] + base,
    }


@pytest.mark.parametrize("case", ["bars", "bars windows text", "trades .locked", "positions", "out", "rules file",
                                  "costs file", "spec"])
def test_locked_holdout_paths_are_refused(files, tmp_path, capsys, case):
    code, out, err = run(_locked_cases(files, tmp_path)[case], capsys)
    assert code == 2
    assert "locked holdout" in err and err.startswith("ERROR: ")
    assert not (tmp_path / "locked_holdout").exists()
    assert not (tmp_path / "out_ok" / "report.json").exists()


def test_outputs_never_overwrite_an_input(files, tmp_path, capsys):
    out = tmp_path / "shared"
    out.mkdir()
    trades_in = out / "trades.csv"
    trades_in.write_bytes(files["trades"].read_bytes())
    before = trades_in.read_bytes()
    code, _, err = run(["evaluate", "--bars", files["bars"], "--trades", trades_in, "--rules", "ftmo-1step",
                        "--out", out] + FAST, capsys)
    assert code == 2 and "would be overwritten" in err
    assert trades_in.read_bytes() == before and not (out / "report.json").exists()
    a_file = tmp_path / "not_a_folder.txt"
    a_file.write_text("x", encoding="ascii")
    code, _, err = run(["evaluate", "--bars", files["bars"], "--trades", files["trades"], "--rules", "ftmo-1step",
                        "--out", a_file] + FAST, capsys)
    assert code == 2 and "is a file, not a folder" in err


def test_write_outputs_stays_inside_the_folder(tmp_path):
    with pytest.raises(cli.UsageError, match="not inside the output folder"):
        cli._inside(tmp_path / "a", tmp_path / "a" / ".." / "escape.json")


@pytest.mark.parametrize("args,needle", [
    ([], "usage"),
    (["evaluate", "--bars", "x.parquet", "--rules", "ftmo-1step", "--out", "o"], "one of the arguments"),
    (["frobnicate"], "invalid choice"),
    (["evaluate", "--bars", "{bars}", "--positions", "{positions}", "--rules", "ftmo-1step", "--out", "{tmp}/o"],
     "--positions needs --size"),
    (["evaluate", "--bars", "{bars}", "--trades", "{trades}", "--size", "2", "--rules", "ftmo-1step", "--out",
      "{tmp}/o"], "--size and --size-mode are for --positions"),
    (["evaluate", "--bars", "{bars}", "--trades", "{trades}", "--positions", "{positions}", "--rules", "ftmo-1step",
      "--out", "{tmp}/o"], "not allowed with"),
    (["evaluate", "--bars", "{bars}", "--trades", "{trades}", "--rules", "ftmo-3step", "--out", "{tmp}/o"],
     "unknown --rules"),
    (["evaluate", "--bars", "{bars}", "--trades", "{trades}", "--rules", "custom", "--out", "{tmp}/o"],
     "needs --rules-file"),
    (["evaluate", "--bars", "{bars}", "--trades", "{trades}", "--rules", "ftmo-1step", "--costs", "flatx",
      "--out", "{tmp}/o"], "write flat followed by a fraction"),
    (["evaluate", "--bars", "{bars}", "--trades", "{trades}", "--rules", "ftmo-1step", "--costs", "ecn",
      "--out", "{tmp}/o"], "unknown --costs"),
    (["evaluate", "--bars", "{bars}", "--trades", "{trades}", "--rules", "ftmo-1step", "--alpha", "5",
      "--out", "{tmp}/o"], "fraction"),
    (["evaluate", "--bars", "{bars}", "--trades", "{trades}", "--rules", "ftmo-1step", "--n-sims", "0",
      "--out", "{tmp}/o"], "above 0"),
    (["evaluate", "--bars", "{tmp}/missing.parquet", "--trades", "{trades}", "--rules", "ftmo-1step",
      "--out", "{tmp}/o"], "not found"),
    (["evaluate", "--bars", "{bars}", "--trades", "{tmp}/missing.csv", "--rules", "ftmo-1step",
      "--out", "{tmp}/o"], "not found"),
])
def test_usage_and_data_errors_exit_2(files, tmp_path, capsys, args, needle):
    subst = {"{bars}": str(files["bars"]), "{trades}": str(files["trades"]), "{positions}": str(files["positions"]),
             "{tmp}": str(tmp_path)}
    argv = []
    for a in args:
        for k, v in subst.items():
            a = a.replace(k, v)
        argv.append(a)
    code = cli.main(argv)
    out, err = capsys.readouterr()
    assert code == 2
    assert needle in (out + err), out + err
    assert (out + err).isascii()
    assert not (tmp_path / "o" / "report.json").exists()


def test_a_bad_trade_file_is_a_data_error(files, tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"side": [1], "units": [-3.0], "entry_time": [1704186000], "entry_price": [2000.0],
                  "exit_time": [1704189600], "exit_price": [2001.0]}).to_csv(bad, index=False)
    code, _, err = run(["evaluate", "--bars", files["bars"], "--trades", bad, "--rules", "ftmo-1step",
                        "--out", tmp_path / "o"] + FAST, capsys)
    assert code == 2 and err.startswith("ERROR: ") and "units" in err


# ---------------------------------------------------------------------------------------
# option helpers

def test_build_costs(tmp_path):
    assert cli.build_costs("dukascopy") == CostModel()
    flat = cli.build_costs("flat0.0003")
    assert flat.flat_rate_per_side == 0.0003 and flat.is_flat and flat.swap_enabled
    # finding 9: the miner's cost exactly is flat0.0003 WITHOUT swap
    miner = cli.build_costs("flat0.0003-noswap")
    assert miner.flat_rate_per_side == 0.0003 and not miner.swap_enabled
    assert not cli.build_costs("dukascopy-noswap").swap_enabled
    assert cli.build_costs("dukascopy", 2.0) == CostModel().multiplied(2.0)
    f = tmp_path / "costs.json"
    f.write_text(json.dumps({"_comment": "broker X, checked 2026-10-01", "markup_per_side": 0.1,
                             "commission_per_lot_round_trip": 7.0}), encoding="ascii")
    cm = cli.build_costs(str(f))
    assert cm.markup_per_side == 0.1 and cm.commission_per_lot_round_trip == 7.0
    f.write_text(json.dumps({"markup": 0.1}), encoding="ascii")
    with pytest.raises(ValueError, match="unknown cost setting"):
        cli.build_costs(str(f))


def test_build_rules(tmp_path):
    assert cli.build_rules("ftmo-1step") == rules.ftmo_1step(100_000.0)
    assert cli.build_rules("FTMO_1step", 50_000).initial_capital == 50_000
    r = cli.build_rules("ftmo-2step", target=0.05)
    assert r.profit_target_pct == 0.05 and r.min_trading_days == 4
    f = tmp_path / "rules.json"
    f.write_text(json.dumps({"_note": "my firm", "name": "mine", "daily_loss_pct": 0.04, "max_loss_pct": 0.08}),
                 encoding="ascii")
    mine = cli.build_rules("custom", 25_000, str(f))
    assert (mine.name, mine.initial_capital, mine.daily_loss_pct, mine.max_loss_pct) == ("mine", 25_000, 0.04, 0.08)
    assert cli.build_rules(str(f)) == cli.build_rules("custom", None, str(f))
    f.write_text(json.dumps({"base": "ftmo-1step", "best_day_max_share": 0.4}), encoding="ascii")
    based = cli.build_rules(str(f), 200_000)
    assert based.best_day_max_share == 0.4 and based.daily_loss_pct == 0.03 and based.initial_capital == 200_000
    f.write_text(json.dumps({"daily_loss": 0.04}), encoding="ascii")
    with pytest.raises(ValueError, match="unknown rule field"):
        cli.build_rules("custom", None, str(f))


# ---------------------------------------------------------------------------------------
# scripts/research/export_positions.py (imports AlphaMaster: this test may)

def _load_export():
    spec = importlib.util.spec_from_file_location("research_export_positions", EXPORT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_export_positions_refuses_unsafe_paths(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)                                        # the script changes the working folder
    ex = _load_export()
    data = tmp_path / "SYN_H1.parquet"
    synthetic_bars(1704067200, 400, seed=1).to_parquet(data, index=False)
    cases = [
        (["--data-file", str(tmp_path / "locked_holdout" / "SYN_H1.parquet"), "--formula", "[0]"], "locked holdout"),
        (["--data-file", str(data), "--formula", "[0]", "--out", str(tmp_path / "p.csv.locked")], "locked holdout"),
        (["--data-file", str(data), "--formula", "[0]", "--out", str(ROOT / "propkit" / "p.csv")], "under logs"),
        (["--data-file", str(data), "--formula", "[0]", "--out", str(tmp_path / "p.json")], "ending in .csv"),
        (["--data-file", str(tmp_path / "missing_H1.parquet"), "--formula", "[0]"], "not found"),
        (["--data-file", str(data), "--formula", "[x]"], "whole numbers"),
    ]
    for argv, needle in cases:
        assert ex.main(argv) == 2, argv
        _, err = capsys.readouterr()
        assert needle in err, (argv, err)
    assert not (ROOT / "propkit" / "p.csv").exists()


def test_export_positions_writes_held_positions_that_propkit_evaluates(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    ex = _load_export()
    bars = synthetic_bars(1704067200, 3000, seed=3, spread=0.34)
    data = tmp_path / "SYN_H1.parquet"
    bars.to_parquet(data, index=False)
    out = tmp_path / "exports" / "pos.csv"
    assert ex.main(["--data-file", str(data), "--formula", "[3,120]", "--out", str(out)]) == 0, capsys.readouterr().err
    stdout = capsys.readouterr().out
    assert stdout.startswith(HEADER) and "position[t+1] = p[t]" in stdout
    df = pd.read_csv(out)
    assert list(df.columns) == ["time", "position", "p_raw", "time_utc"]
    assert df["time"].tolist() == bars["time"].tolist()
    assert df["position"].iloc[0] == 0.0
    assert np.array_equal(df["position"].to_numpy()[1:], df["p_raw"].to_numpy()[:-1])   # held[t+1] = p[t]
    assert df["position"].abs().max() <= 1.0 and (df["position"] != 0).mean() > 0.5
    run_out = tmp_path / "eval"
    code = cli.main(["evaluate", "--bars", str(data), "--positions", str(out), "--size", "10", "--rules",
                     "ftmo-1step", "--out", str(run_out)] + FAST)
    assert code == 0, capsys.readouterr().err
    assert all((run_out / name).is_file() for name in FOUR)
    # the default output goes under the repo's logs folder
    assert ex.default_out(data, [3, 120], None) == ROOT / "logs" / "positions_SYN_H1_3-120.csv"
    assert ex.default_out(data, list(range(30)), None).name.startswith("positions_SYN_H1_f")


def test_export_positions_console_is_ascii_even_with_alphamaster_log_lines(tmp_path):
    """Finding 27: AlphaMaster's data loader logs Chinese text through loguru; the script's console must stay
    ASCII (the characters become backslash escapes)."""
    pytest.importorskip("torch")
    data = tmp_path / "SYN_H1.parquet"
    synthetic_bars(1704067200, 400, seed=3, spread=0.34).to_parquet(data, index=False)
    env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    res = subprocess.run([sys.executable, str(EXPORT_SCRIPT), "--data-file", str(data), "--formula", "[3,120]",
                          "--out", str(tmp_path / "pos.csv")], capture_output=True, cwd=tmp_path, env=env,
                         timeout=300)
    assert res.returncode == 0, res.stderr.decode("utf-8", "replace")
    both = res.stdout + res.stderr
    assert both.isascii(), [ln for ln in both.decode("utf-8", "replace").splitlines() if not ln.isascii()][:3]
    assert b"\\u" in both                                   # the loader's log line is there, escaped



# ---------------------------------------------------------------------------------------
# review findings

def _one_run(files, tmp_path, capsys, name, extra, rules="ftmo-1step", src=("--trades", None)):
    out = tmp_path / name
    flag, path = src
    code, stdout, err = run(["evaluate", "--bars", files["bars"], flag, path or files["trades"], "--rules", rules,
                             "--no-stress", "--out", out] + list(extra) + FAST, capsys)
    return code, stdout, err, out


def test_cost_mult_with_trades_reprices_the_fills(files, tmp_path, capsys):
    """Finding 6: --cost-mult with --trades multiplied only commission and swap; the fills kept the base
    spread. Now every buy fill moves up by (k - 1) x the bar spread (dukascopy: no markup / slippage)."""
    reps = {}
    for k in ("1", "2"):
        code, _, err, out = _one_run(files, tmp_path, capsys, f"k{k}", ["--cost-mult", k])
        assert code == 0, err
        reps[k] = json.loads((out / "report.json").read_text(encoding="ascii"))
    tr = pd.read_csv(files["trades"])
    t = files["frame"]["time"].to_numpy()
    sp = CostModel().bar_spreads(files["frame"])
    buy_time = np.where(tr["side"] == "long", tr["entry_time"], tr["exit_time"])
    extra = float((sp[np.searchsorted(t, buy_time)] * tr["units"]).sum())
    g1, g2 = reps["1"]["trades_summary"]["gross_usd"], reps["2"]["trades_summary"]["gross_usd"]
    assert extra > 0 and g2 == pytest.approx(g1 - extra, abs=1e-6)
    assert reps["2"]["input"]["fills_repriced_for_cost_mult"] == 2.0 and reps["2"]["input"]["cost_mult"] == 2.0
    assert "fills of the trade list repriced" in (tmp_path / "k2" / "report.md").read_text(encoding="ascii")
    code, _, err, out = _one_run(files, tmp_path, capsys, "k05", ["--cost-mult", "0.5"])
    assert code == 2 and "--cost-mult below 1" in err and not out.exists()


@pytest.mark.parametrize("value", [None, [100000], True, "100k", -5])
def test_a_bad_initial_capital_in_a_rules_file_is_a_usage_error(files, tmp_path, capsys, value):
    """Finding 18: null / a list / true gave a traceback or a 1-USD account."""
    for body in ({"base": "ftmo-1step", "initial_capital": value},
                 {"name": "mine", "daily_loss_pct": 0.04, "initial_capital": value}):
        f = tmp_path / "r.json"
        f.write_text(json.dumps(body), encoding="ascii")
        code, stdout, err, out = _one_run(files, tmp_path, capsys, "o", [], rules=str(f))
        assert code == 2 and "initial_capital" in err, err
        assert "Traceback" not in stdout + err and "internal error" not in err and not (out / "report.json").exists()


def test_a_rules_file_inside_out_is_never_overwritten(files, tmp_path, capsys):
    """Finding 19: --rules FILE.json was missing from the overwrite check."""
    out = tmp_path / "shared"
    out.mkdir()
    f = out / "report.json"
    f.write_text(json.dumps({"base": "ftmo-1step", "best_day_max_share": 0.5}), encoding="ascii")
    before = f.read_bytes()
    code, _, err, _ = _one_run(files, tmp_path, capsys, "shared", [], rules=str(f))
    assert code == 2 and "would be overwritten" in err and f.read_bytes() == before


def test_a_rules_file_base_that_differs_from_rules_is_refused(files, tmp_path, capsys):
    """Finding 20: --rules ftmo-1step --rules-file {base: ftmo-2step} silently used the 1-Step."""
    f = tmp_path / "phase2.json"
    f.write_text(json.dumps({"base": "ftmo-2step", "profit_target_pct": 0.05}), encoding="ascii")
    code, _, err, _ = _one_run(files, tmp_path, capsys, "o", ["--rules-file", str(f)])
    assert code == 2 and "says base" in err and "ftmo-2step" in err
    assert cli.build_rules("ftmo-2step", None, str(f)).name == "FTMO 2-Step (target 5%)"
    assert cli.build_rules("ftmo-2step", target=0.05).name == "FTMO 2-Step (target 5%)"
    assert "(modified: daily_loss_pct=0.04)" in cli.build_rules("ftmo-1step", rules_file=None).name + \
        rules.preset("ftmo-1step", daily_loss_pct=0.04).name


def test_errors_name_the_command_line_options(files, tmp_path, capsys):
    """Finding 21: the price check named the Python argument price_tolerance; the README lacked the options."""
    tr = pd.read_csv(files["trades"])
    tr["entry_price"] = tr["entry_price"] * 1.5
    bad = tmp_path / "scaled.csv"
    tr.to_csv(bad, index=False)
    code, _, err, _ = _one_run(files, tmp_path, capsys, "o", [], src=("--trades", bad))
    assert code == 2 and "--price-tolerance none" in err
    methods = (ROOT / "propkit" / "METHODS.md").read_text(encoding="ascii")    # the options table (README: 1 page)
    for option in ("--spread-scale", "--price-tolerance", "--horizon-unit", "--history-reps", "-noswap"):
        assert option in methods, option


def test_a_spread_in_broker_points_is_refused_unless_scaled(files, tmp_path, capsys):
    """Finding 22: a spread of 5 broker points (0.05 USD) was accepted as 5 USD/oz."""
    pts = tmp_path / "XAUUSD_points_H1.parquet"
    files["frame"].assign(spread=5.0).to_parquet(pts, index=False)
    args = ["evaluate", "--bars", pts, "--positions", files["positions"], "--size", "10", "--rules", "ftmo-1step",
            "--no-stress"]
    code, _, err = run(args + ["--out", tmp_path / "o1"] + FAST, capsys)
    assert code == 2 and "--spread-scale 0.01" in err
    code, stdout, err = run(args + ["--spread-scale", "0.01", "--out", tmp_path / "o2"] + FAST, capsys)
    assert code == 0, err
    assert "spread column median 0.050 USD/oz" in stdout


def test_a_stale_decision_log_is_removed_by_a_later_evaluate(files, tmp_path, capsys):
    """Finding 24: decisions.csv of an earlier pullback run survived an evaluate run in the same folder."""
    out = tmp_path / "same"
    code, _, err = run(["pullback", "--bars", files["bars"], "--spec", EXAMPLE_SPEC, "--rules", "ftmo-1step",
                        "--no-stress", "--out", out] + FAST, capsys)
    assert code == 0 and (out / "decisions.csv").is_file(), err
    code, stdout, err, _ = _one_run(files, tmp_path, capsys, "same", [])
    assert code == 0, err
    assert not (out / "decisions.csv").exists() and "removed decisions.csv" in stdout
    (out / "decisions.csv").write_text("my,own,notes\n1,2,3\n", encoding="ascii")      # not propkit's: kept
    code, stdout, err, _ = _one_run(files, tmp_path, capsys, "same", [])
    assert code == 0 and (out / "decisions.csv").is_file() and "is not from this run" in stdout


def test_one_trade_prints_na_for_its_standard_error(files, tmp_path, capsys):
    """Finding 25: the console printed '+- 0.000' when se_r is undefined (one trade)."""
    tr = pd.read_csv(files["trades"]).iloc[:1].copy()
    tr["stop_price"] = tr["entry_price"] - 5.0
    one = tmp_path / "one.csv"
    tr.to_csv(one, index=False)
    code, stdout, err, out = _one_run(files, tmp_path, capsys, "o", [], src=("--trades", one))
    assert code == 0, err
    assert "+- n/a (one trade)" in stdout and "+- 0.000" not in stdout
    assert "n/a (one trade)" in (out / "report.md").read_text(encoding="ascii")


def test_any_text_column_is_written_and_a_failed_write_leaves_no_half_folder(files, tmp_path, capsys, monkeypatch):
    """Finding 16: a non-ASCII text column crashed write_trades_csv and left report.json of the new run next
    to the old run's trades.csv."""
    tr = pd.read_csv(files["trades"])
    tr["comment"] = "\u9ec4\u91d1 setup"
    noted = tmp_path / "noted.csv"
    tr.to_csv(noted, index=False, encoding="utf-8")
    code, _, err, out = _one_run(files, tmp_path, capsys, "o", [], src=("--trades", noted))
    assert code == 0, err
    text = (out / "trades.csv").read_text(encoding="ascii")
    assert "\\u9ec4\\u91d1 setup" in text
    before = {n: (out / n).read_bytes() for n in FOUR}

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(cli.adapters, "write_trades_csv", boom)
    code, _, err, _ = _one_run(files, tmp_path, capsys, "o", ["--seed", "8"])
    assert code == 2 and "disk full" in err
    assert {n: (out / n).read_bytes() for n in FOUR} == before                 # the old run, untouched
    assert not [p.name for p in out.iterdir() if p.name.startswith(cli.TMP_PREFIX)]


def test_horizon_unit_market_is_accepted_and_reported(files, tmp_path, capsys):
    """Finding 4: the horizon counts trading days by default; --horizon-unit market counts market days."""
    code, stdout, err, out = _one_run(files, tmp_path, capsys, "m", ["--horizon-unit", "market"])
    assert code == 0, err
    rep = json.loads((out / "report.json").read_text(encoding="ascii"))
    assert rep["settings"]["horizon_unit"] == "market" and rep["bootstrap_days"]["horizon_unit"] == "market"
    assert "60 market days" in stdout
    code, _, err = run(["evaluate", "--bars", files["bars"], "--trades", files["trades"], "--rules", "ftmo-1step",
                        "--horizon-unit", "calendar", "--out", tmp_path / "x"], capsys)
    assert code == 2 and "invalid choice" in err
