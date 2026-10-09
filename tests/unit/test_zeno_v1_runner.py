"""The zeno_pullback_v1 runner: `python -m propkit zeno-v1 signals|run` (propkit.cli, propkit.zeno_report).

Stage 1 writes no result; stage 2 refuses without --g0-confirmed; the lock and the locked-path refusals;
nothing is written outside --out; gates.json known answers on hand-built grids; the metric definitions
[SI-41], the judging cell [SI-28], G1 counting positions [SI-29], the period assignment [SI-42], G4 under the
placeholder and under a verified-sheet stand-in [SI-31, SI-32]; timing guards on ~260,000 M15 bars.
Research only; synthetic data only (NOT market data); every rules file here is a labelled test fixture.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from propkit import calendar as cal
from propkit import cli
from propkit import rules as R
from propkit import zeno_report as zr
from propkit import zeno_v1 as z
from propkit.evaluator import evaluate_path
from tests.unit.zeno_v1_testkit import NEWS_CSV, utc

HEADER = "RESEARCH ONLY - not trading advice"
FAST = ["--n-sims", "200", "--history-reps", "0"]


def write_pair(folder: Path, frame: pd.DataFrame, stem: str = "SYNTH") -> tuple[Path, Path]:
    """Bid and ask CSV files (time, open, high, low, close) of a zeno frame."""
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for side in ("bid", "ask"):
        df = frame[["time", f"{side}_open", f"{side}_high", f"{side}_low", f"{side}_close"]].copy()
        df.columns = ["time", "open", "high", "low", "close"]
        p = folder / f"{stem}_M15_{side}.csv"
        df.to_csv(p, index=False)
        out.append(p)
    return out[0], out[1]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """~6 months of synthetic M15 bid/ask bars from 2015-01-01 (spread 0.05 USD/oz) and the six real
    calendar rows of the testkit."""
    d = tmp_path_factory.mktemp("zeno_runner")
    frame = z.synthetic_m15_bidask(n_bars=16_000, seed=3, spread=0.05)
    bid, ask = write_pair(d / "in", frame)
    news = d / "in" / "news.csv"
    news.write_text(NEWS_CSV, encoding="ascii")
    return {"dir": d, "bid": bid, "ask": ask, "news": news, "frame": frame}


def run(args, capsys) -> tuple[int, str, str]:
    code = cli.main([str(a) for a in args])
    out, err = capsys.readouterr()
    assert out.isascii() and err.isascii()
    return code, out, err


def inputs(f) -> list:
    return ["--m15-bid", f["bid"], "--m15-ask", f["ask"], "--news", f["news"]]


# ---------------------------------------------------------------------------------------
# stage 1

def test_signals_stage_writes_no_results(files, tmp_path, capsys):
    out = tmp_path / "g0"
    code, stdout, err = run(["zeno-v1", "signals"] + inputs(files) + ["--out", out], capsys)
    assert code == 0, err
    assert stdout.startswith(HEADER) and "NEXT: G0 signal check" in stdout
    assert sorted(p.name for p in out.iterdir()) == sorted(zr.SIGNALS_FILES)
    for name in ("signals.csv", "decisions.csv", "g0_sample.csv"):
        cols = pd.read_csv(out / name, nrows=0).columns
        bad = [c for c in cols if any(w.lower() in c.lower() for w in zr.FORBIDDEN_STAGE1_WORDS)]
        assert not bad, (name, bad)
        assert "position_id" not in cols
    sig = pd.read_csv(out / "signals.csv")
    assert list(sig.columns) == list(zr.SIGNAL_COLUMNS) and len(sig) > 100
    g0 = pd.read_csv(out / "g0_sample.csv", keep_default_na=False)
    assert list(g0.columns) == list(zr.G0_COLUMNS) and len(g0) == 20
    assert (g0["agree_y_n"] == "").all()
    assert set(sig.set_index("signal_no").loc[g0["signal_no"], "status"]) == {"eligible"}
    assert g0["signal_time_utc"].is_monotonic_increasing
    rep = json.loads((out / "signals_report.json").read_text(encoding="ascii"))
    text = json.dumps(rep).lower()
    for word in ("expectancy", "pnl", "r_multiple", "sharpe", "hit_rate", "outcome\"", "net_usd", "p_pass"):
        assert word not in text, word
    assert rep["declared_cell"]["label"] == "evaluation/c10/S1/x1.5"
    assert rep["g0"]["sample_size"] == 20 and len(rep["g0"]["sample_sha256"]) == 64
    md = (out / "signals_report.md").read_text(encoding="ascii")
    assert md.startswith(HEADER) and "agree with at least 18 of 20" in md
    # RR-1 [SI-66]: the G0 entry is the chart's: the ask FILE's open of the entry bar for a long, the bid file's
    # open for a short (the declared cell is x1.5, whose long entry is on no chart)
    bid_f, ask_f = (pd.read_csv(files[s]).set_index("time") for s in ("bid", "ask"))
    g0n = pd.read_csv(out / "g0_sample.csv")
    et = sig.set_index("signal_no").loc[g0n["signal_no"], "entry_time"].to_numpy()
    want = np.where(g0n["side"] == "long", ask_f.loc[et, "open"].to_numpy(), bid_f.loc[et, "open"].to_numpy())
    assert (g0n["side"] == "long").any() and np.allclose(g0n["entry_price"].to_numpy(), want, rtol=0, atol=1e-9)


def test_signals_cut_bars_before_2015_and_say_so(tmp_path, capsys):
    # D1's range starts 2015-01-01 00:00 UTC [SI-62]: earlier bars in the files are cut, counted and named
    from propkit.bars import synthetic_bars
    raw = synthetic_bars(utc("2014-12-15 00:00"), 5_000, bar_seconds=900, seed=5, spread=0.05)
    (tmp_path / "in").mkdir()
    bid, ask, news = tmp_path / "in" / "X_M15_bid.csv", tmp_path / "in" / "X_M15_ask.csv", tmp_path / "in" / "news.csv"
    raw_bid = raw[["time", "open", "high", "low", "close"]]
    raw_bid.to_csv(bid, index=False)
    raw_bid.assign(**{c: raw_bid[c] + 0.05 for c in ("open", "high", "low", "close")}).to_csv(ask, index=False)
    news.write_text(NEWS_CSV, encoding="ascii")
    out = tmp_path / "g0"
    code, stdout, err = run(["zeno-v1", "signals", "--m15-bid", bid, "--m15-ask", ask, "--news", news, "--out", out],
                            capsys)
    assert code == 0, err
    n_pre = int((raw_bid["time"] < utc("2015-01-01 00:00")).sum())
    assert n_pre > 0 and f"NOTE: {n_pre} bar(s) before 2015-01-01 00:00 UTC were cut" in stdout
    rep = json.loads((out / "signals_report.json").read_text(encoding="ascii"))
    assert rep["data"]["cut_before_range_start"] == n_pre and rep["data"]["n_bars"] == 5_000 - n_pre
    assert rep["data"]["first_time_utc"] == "2015-01-01 00:00:00 UTC"
    assert "were cut: D1's range starts there" in (out / "signals_report.md").read_text(encoding="ascii")


def test_g0_sample_is_deterministic_and_draws_eligible_signals_only(files):
    prep = z.prepare(files["frame"])
    _, t = zr.signals_stage(prep, sample=20, seed=7)
    sig = t["signals"]
    a, b = zr.g0_sample(sig, 20, 7), zr.g0_sample(sig, 20, 7)
    assert a.equals(b) and not a.equals(zr.g0_sample(sig, 20, 8))
    assert t["g0_sample"].equals(a)
    eligible = sig[sig["status"] == "eligible"]
    few = zr.g0_sample(eligible.head(5), 20, 7)
    assert len(few) == 5                                    # fewer eligible signals than asked: all of them
    # the run of the same cell enters a subset of the eligible signals [SI-54]
    res = z.simulate(prep, z.ZenoConfig(zr.STAGE1_CELL))
    tr = res.decisions[res.decisions["event"] == "trigger"].reset_index(drop=True)
    assert set(tr.index[tr["status"] == "entered"]) <= set(eligible.index)
    assert (tr["status"] == "entered").sum() < len(eligible)
    with pytest.raises(ValueError, match="sample size"):
        zr.g0_sample(sig, 0, 7)


STATE_CHECKS = ("max_entries_per_day", "two_losses_today", "day_loss_1pct", "cooldown_15min", "position_open")


def test_stage_1_files_never_name_a_check_that_needs_earlier_trades(files, tmp_path, capsys):
    # F2 / RUN-1 [SI-54]: the daily limits, the cooldown and the one-position check need how and when earlier
    # trades ended (their P&L and exit times). Stage 1 leaves them unevaluated, so no stage-1 file, no count
    # and no screen line names them (a row listing two_losses_today told which trades lost).
    out = tmp_path / "g0"
    code, stdout, err = run(["zeno-v1", "signals"] + inputs(files) + ["--out", out], capsys)
    assert code == 0, err
    for name in zr.SIGNALS_FILES:
        text = (out / name).read_text(encoding="ascii")
        assert [w for w in STATE_CHECKS if w in text] == [], name
    assert [w for w in STATE_CHECKS if w in stdout] == []
    sig = pd.read_csv(out / "signals.csv", keep_default_na=False)
    assert "eligible" in set(sig["status"]) and "entered" not in set(sig["status"])


def test_stage_1_status_does_not_depend_on_how_earlier_trades_ended():
    # F2 / RUN-1: two losing entries, then a third trigger on the same server day (test_zeno_v1_filters'
    # _three_blocks), and a trigger while a runner is still open. The run blocks them with reasons that need
    # the earlier trades' outcome and exit time; stage 1 does not know those before any result is read, so
    # every one of these triggers passes the checks it can evaluate: status eligible, no reason listed.
    from tests.unit.zeno_v1_testkit import EVAL_10_X1, Scenario
    stop_bar = (2006.0, 2006.0, 2001.0, 2002.0)
    sc = Scenario("2024-03-05 05:30")
    sc.setup(1)
    sc.add(*stop_bar)
    sc.setup(1)
    sc.add(*stop_bar)
    sc.flat_until("2024-03-05 11:00", 2000.0)
    sc.setup(1)
    sc.add(*stop_bar)
    sc.flat(4, 2000.0)
    prep, res = sc.run()
    tr = res.decisions[res.decisions["event"] == "trigger"]
    assert tr["status"].tolist() == ["entered", "entered", "max_entries_per_day"]
    assert tr["reasons"].iloc[2] == "max_entries_per_day;two_losses_today;day_loss_1pct"
    _, t = zr.signals_stage(prep, EVAL_10_X1)
    assert t["signals"]["status"].tolist() == ["eligible"] * 3
    assert t["signals"]["reasons"].tolist() == [""] * 3
    assert len(t["g0_sample"]) == 3

    sc = Scenario("2024-03-05 05:30")                     # trade 1's runner is open at the second trigger
    sc.setup(1)
    sc.flat(2, 2006.0)
    sc.flat(20, 2008.0)
    sc.setup(1, base=2008.0)
    sc.add(2014.0, 2016.0, 2013.0, 2015.5)
    sc.flat(4, 2015.0)
    prep, res = sc.run()
    tr = res.decisions[res.decisions["event"] == "trigger"]
    assert tr["status"].tolist() == ["entered", "position_open"]
    _, t = zr.signals_stage(prep, EVAL_10_X1)
    assert t["signals"]["status"].tolist() == ["eligible", "eligible"]


@pytest.mark.parametrize("side", [1, -1])
def test_g0_sample_shows_the_chart_prices_not_the_cost_cell_prices(side):
    # RR-1 [SI-66]: stage 1 screens in the declared cell (default evaluation / 10 / S1 / x1.5), but zeno checks
    # the entry and the stop on a chart of the DATA, so g0_sample.csv and signals.csv show those prices; the
    # cell's prices (what rule 10 compares) are in signals.csv's *_at_costs columns only.
    # Canonical long, data spread 0.20 USD/oz, the next bar opens at bid 2006.00:
    #   chart: entry = the ask file's open = 2006.00 + 0.20 = 2006.20; stop = 2002 - 0.25 x 2 = 2001.50
    #   cell x1.5: entry = 2006.00 + 1.5 x 0.20 = 2006.30 (on no chart); stop 2001.50; spread 0.30
    # Canonical short (mirror), the next bar opens at bid 1994.00 (ask 1994.20):
    #   chart: entry = the bid file's open = 1994.00; stop = 1998 + 0.50 + 0.20 (data entry spread) = 1998.70
    #   cell x1.5: entry 1994.00; stop = 1998 + 0.50 + 0.30 = 1998.80 (on no chart)
    from tests.unit.zeno_v1_testkit import Scenario
    sc = Scenario("2024-03-05 05:30")
    ti = sc.setup(side)
    sc.flat(4, 2006.0 if side > 0 else 1994.0)
    prep = sc.prepare(trend="long" if side > 0 else "short")
    rep, t = zr.signals_stage(prep)
    assert rep["declared_cell"]["label"] == "evaluation/c10/S1/x1.5"
    g0, sig = t["g0_sample"], t["signals"]
    assert len(g0) == 1 and len(sig) == 1 and sig["status"].iloc[0] == "eligible"
    f = prep.frame
    e = ti + 1
    chart_entry = f["ask_open"].iloc[e] if side > 0 else f["bid_open"].iloc[e]
    assert chart_entry == pytest.approx(2006.20 if side > 0 else 1994.00)
    chart = (chart_entry, 2001.50 if side > 0 else 1998.70)
    cell = (2006.30, 2001.50) if side > 0 else (1994.00, 1998.80)
    for table in (g0, sig):
        assert (table["entry_price"].iloc[0], table["stop_level"].iloc[0]) == pytest.approx(chart)
    assert sig["spread_entry"].iloc[0] == pytest.approx(0.20)
    assert (sig["entry_price_at_costs"].iloc[0], sig["stop_level_at_costs"].iloc[0]) == pytest.approx(cell)
    assert sig["spread_entry_at_costs"].iloc[0] == pytest.approx(0.30)
    why = rep["declared_cell"]["why"]
    assert "chart" in why and "x1" in why and "does not depend on the cell except" not in why


def test_signals_refuses_to_overwrite_an_answered_g0_sample(files, tmp_path, capsys):
    # RR1-RUN-1: zeno writes the answers into DIR/g0_sample.csv; running stage 1 again into the same folder
    # must not replace that file. An unanswered sample may be redrawn (the earlier behaviour).
    out = tmp_path / "g0"
    for _ in range(2):
        code, _, err = run(["zeno-v1", "signals"] + inputs(files) + ["--out", out], capsys)
        assert code == 0, err
    g0 = pd.read_csv(out / "g0_sample.csv", dtype=str, keep_default_na=False)
    g0.loc[0, "agree_y_n"] = "y"
    g0.to_csv(out / "g0_sample.csv", index=False)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    code, stdout, err = run(["zeno-v1", "signals"] + inputs(files) + ["--out", out], capsys)
    assert code == 2 and "g0_sample.csv" in err and "1 answer" in err and "another --out" in err
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before          # nothing replaced or added


def test_server_time_text():
    # server clock = New York + 7 h: 13:00 UTC is 09:00 EDT (summer) -> 16:00, 08:00 EST (winter) -> 15:00
    assert zr.server_time_str(utc("2024-07-10 13:00")) == "2024-07-10 16:00:00 server"
    assert zr.server_time_str(utc("2024-01-10 13:00")) == "2024-01-10 15:00:00 server"
    assert zr.server_time_str(np.array([-1, utc("2024-01-10 22:00")])).tolist() == ["", "2024-01-11 00:00:00 server"]


# ---------------------------------------------------------------------------------------
# refusals

def test_run_refuses_without_g0_confirmed(files, tmp_path, capsys):
    out = tmp_path / "never"
    code, stdout, err = run(["zeno-v1", "run"] + inputs(files) + ["--out", out], capsys)
    assert code == 2 and "G0 first" in err and "zeno-v1 signals" in err and "at least 18 of 20" in err
    assert not out.exists() and stdout == ""


def test_zeno_help_and_missing_stage(capsys):
    code, out, _ = run(["zeno-v1", "--help"], capsys)
    assert code == 0 and "signals" in out and "run" in out
    code, out, _ = run(["zeno-v1", "run", "--help"], capsys)
    assert code == 0 and "--g0-confirmed" in out and "--reference-rules" in out
    code, _, err = run(["zeno-v1"], capsys)
    assert code == 2 and "needs a stage" in err


def test_lock_and_locked_path_refusals(files, tmp_path, capsys):
    # a file whose last bar opens at the lock instant (2025-09-28 00:00 UTC) is refused, whatever the stage
    start = z.LOCK_UTC - 900 * 6000                     # 3000 metals-hours bars end well before the lock
    f = z.synthetic_m15_bidask(start=start, n_bars=3000, seed=1)
    ok_bid, ok_ask = write_pair(tmp_path / "ok", f)
    bad = pd.read_csv(ok_bid)
    bad.loc[len(bad)] = [z.LOCK_UTC] + bad.iloc[-1, 1:].tolist()
    bad_bid, bad_ask = tmp_path / "bad_bid.csv", tmp_path / "bad_ask.csv"
    bad.to_csv(bad_bid, index=False)
    a = pd.read_csv(ok_ask)
    a.loc[len(a)] = [z.LOCK_UTC] + a.iloc[-1, 1:].tolist()
    a.to_csv(bad_ask, index=False)
    for stage in (["signals"], ["run", "--g0-confirmed"] + FAST):
        out = tmp_path / ("o_" + stage[0])
        code, _, err = run(["zeno-v1"] + stage + ["--m15-bid", bad_bid, "--m15-ask", bad_ask, "--news", files["news"],
                                                  "--out", out], capsys)
        assert code == 2 and "2025-09-28 00:00:00 UTC" in err and "no override" in err
        assert not out.exists() or not any(out.iterdir())
    locked = tmp_path / "locked_holdout" / "bid.csv"
    for args in (["--m15-bid", locked, "--m15-ask", ok_ask], ["--m15-bid", ok_bid, "--m15-ask", tmp_path / "x.csv.locked"]):
        code, _, err = run(["zeno-v1", "signals"] + args + ["--news", files["news"], "--out", tmp_path / "o3"], capsys)
        assert code == 2 and "locked" in err.lower()
    code, _, err = run(["zeno-v1", "signals", "--m15-bid", ok_bid, "--m15-ask", ok_ask, "--news", files["news"],
                        "--out", tmp_path / "locked_holdout" / "out"], capsys)
    assert code == 2 and not (tmp_path / "locked_holdout").exists()


def test_refuses_writing_outside_out(files, tmp_path, capsys):
    out = tmp_path / "o"
    out.mkdir()
    with pytest.raises(cli.UsageError, match="not inside the output folder"):
        cli.write_staged(out, [("../escape.csv", lambda p: p.write_text("x"))])
    assert not (tmp_path / "escape.csv").exists()
    assert not [p for p in out.iterdir()]
    # an input inside --out under an output name would be overwritten: refused before anything is read
    victim = out / "signals.csv"
    victim.write_text("keep", encoding="ascii")
    code, _, err = run(["zeno-v1", "signals", "--m15-bid", victim, "--m15-ask", files["ask"], "--news", files["news"],
                        "--out", out], capsys)
    assert code == 2 and "would be overwritten" in err and victim.read_text(encoding="ascii") == "keep"
    code, _, err = run(["zeno-v1", "signals"] + inputs(files) + ["--out", files["bid"]], capsys)
    assert code == 2 and "is a file, not a folder" in err


def _old_run(out: Path) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    old = {"a.csv": "old a", "b.csv": "old b", "c.csv": "old c"}
    for name, text in old.items():
        (out / name).write_text(text, encoding="ascii")
    return old


def _new_jobs():
    return [(name, (lambda t: (lambda p: p.write_text(t, encoding="ascii")))(f"new {name[0]}"))
            for name in ("a.csv", "b.csv", "c.csv")]


def test_write_staged_replaces_nothing_when_a_final_file_cannot_be_replaced(tmp_path):
    # an output name taken by a folder (as a file held open by Excel on Windows): refused before any rename
    out = tmp_path / "out"
    old = _old_run(out)
    (out / "c.csv").unlink()
    (out / "c.csv").mkdir()
    (out / "c.csv" / "keep.txt").write_text("x", encoding="ascii")
    with pytest.raises(OSError, match="c.csv"):
        cli.write_staged(out, _new_jobs())
    assert (out / "a.csv").read_text(encoding="ascii") == old["a.csv"]
    assert (out / "b.csv").read_text(encoding="ascii") == old["b.csv"]
    assert sorted(p.name for p in out.iterdir()) == ["a.csv", "b.csv", "c.csv"]      # no _partial_ file left


def test_write_staged_rolls_back_when_a_rename_fails(tmp_path, monkeypatch):
    # the second rename fails (a file locked between the check and the rename): the files already replaced are
    # put back, so --out holds the whole earlier run, and no _partial_ or backup file is left
    out = tmp_path / "out"
    old = _old_run(out)
    real = cli.os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        if str(src).endswith(f"{cli.TMP_PREFIX}b.csv"):
            calls["n"] += 1
            raise PermissionError(13, "The process cannot access the file", str(dst))
        return real(src, dst)
    monkeypatch.setattr(cli.os, "replace", flaky)
    with pytest.raises(OSError, match="b.csv"):
        cli.write_staged(out, _new_jobs())
    assert calls["n"] == 1
    for name, text in old.items():
        assert (out / name).read_text(encoding="ascii") == text, name
    assert sorted(p.name for p in out.iterdir()) == ["a.csv", "b.csv", "c.csv"]
    monkeypatch.setattr(cli.os, "replace", real)
    written = cli.write_staged(out, _new_jobs())                       # and a normal run replaces all three
    assert [p.read_text(encoding="ascii") for p in written] == ["new a", "new b", "new c"]
    assert sorted(p.name for p in out.iterdir()) == ["a.csv", "b.csv", "c.csv"]


def test_g0_sample_answers_are_recorded_and_a_failed_g0_is_refused(tmp_path):
    def sample(answers):
        p = tmp_path / f"g0_{len(answers)}_{answers.count('y')}.csv"
        pd.DataFrame({"sample_no": range(1, len(answers) + 1), "agree_y_n": answers}).to_csv(p, index=False)
        return str(p)
    rec = cli._g0_record(sample(["y"] * 19 + ["n"]))
    assert rec["agree_count"] == 19 and rec["answered"] == 20 and len(rec["sample_sha256"]) == 64
    assert rec["status"] == "confirmed_by_operator" and rec["confirmed_by_operator"] is True
    with pytest.raises(cli.UsageError, match="G0 failed: .* 17 of 20"):
        cli._g0_record(sample(["y"] * 17 + ["n"] * 3))
    assert cli._g0_record(sample([""] * 20))["answered"] == 0              # not filled in: recorded only
    assert cli._g0_record(None)["sample_sha256"] is None


def test_a_g0_sample_that_can_no_longer_reach_18_of_20_is_refused(tmp_path):
    # G0: zeno agrees with at least 18 of 20, so at most 2 of 20 may be "n" whatever the blanks hold.
    def sample(name, answers):
        p = tmp_path / f"{name}.csv"
        pd.DataFrame({"sample_no": range(1, len(answers) + 1), "agree_y_n": answers}).to_csv(p, index=False)
        return str(p)
    with pytest.raises(cli.UsageError, match=r"G0 failed: .*\(3 n"):
        cli._g0_record(sample("a", ["y"] * 15 + ["n"] * 3 + [""] * 2))       # 15 + 2 blanks = 17 < 18
    with pytest.raises(cli.UsageError, match=r"G0 failed: .*\(10 n"):
        cli._g0_record(sample("b", ["n"] * 10))                              # 10 rows, every one n
    with pytest.raises(cli.UsageError, match="20 rows"):
        cli._g0_record(sample("c", ["y"] * 10))                              # 10 of 20 cannot be 18 of 20
    with pytest.raises(cli.UsageError, match=r"G0 not met: .*2 of 20 rows"):  # RR1-G0-1: 16 y is not 18 of 20
        cli._g0_record(sample("d", ["y"] * 16 + ["n"] * 2 + [""] * 2))


def test_an_answered_g0_sample_needs_every_row_answered_and_18_y(tmp_path):
    # RR1-G0-1 [SI-60]: once any row holds an answer, the sample is zeno's check: every row must be answered y or
    # n and at least ceil(18 x rows / 20) must be y. Blanks and anything else ("?") are not agreements.
    def sample(name, answers):
        p = tmp_path / f"{name}.csv"
        pd.DataFrame({"sample_no": range(1, len(answers) + 1), "agree_y_n": answers}).to_csv(p, index=False)
        return str(p)
    for name, answers, text in (("16y_4blank", ["y"] * 16 + [""] * 4, "4 of 20 rows"),
                                ("17y_3q", ["y"] * 17 + ["?"] * 3, "3 of 20 rows"),
                                ("19_rows_all_y", ["y"] * 19, "19 rows")):
        with pytest.raises(cli.UsageError, match="G0 not met") as e:
            cli._g0_record(sample(name, answers))
        assert text in str(e.value), name
    rec = cli._g0_record(sample("18y_2n", ["y"] * 18 + ["n"] * 2))
    assert rec["status"] == "confirmed_by_operator" and (rec["agree_count"], rec["answered"], rec["n_rows"]) == (18, 20, 20)
    rec = cli._g0_record(sample("yes_no_case", ["Yes"] * 19 + [" N "]))
    assert (rec["agree_count"], rec["n_no"], rec["unanswered"]) == (19, 1, 0)
    rec = cli._g0_record(sample("blank", [""] * 20))                         # not filled in: recorded only
    assert (rec["answered"], rec["unanswered"]) == (0, 20)


# ---------------------------------------------------------------------------------------
# stage 2 end to end (tiny synthetic run)

def test_run_writes_every_file_and_gates_json(files, tmp_path, capsys):
    out = tmp_path / "run"
    code, stdout, err = run(["zeno-v1", "run", "--g0-confirmed"] + inputs(files) + ["--out", out] + FAST, capsys)
    assert code == 0, err
    # updated for addendum A3: the default rules are the verified preset, whose warning names its [U] fields
    assert "WARNING" in stdout and "[U]" in stdout and "Verdict:" in stdout
    assert sorted(p.name for p in out.iterdir()) == sorted(zr.RUN_FILES)
    g = json.loads((out / "gates.json").read_text(encoding="ascii"))
    assert set(g["gates"]) == {"G0", "G1", "G2", "G3", "G4", "G5"}
    assert g["gates"]["G0"]["confirmed_by_operator"] is True
    for name in ("G1", "G2", "G3", "G5"):
        assert g["gates"][name]["status"] in ("pass", "fail", "flagged")
    assert g["judging_cell"]["variant"] == "evaluation" and g["judging_cell"]["commission_rt_per_lot"] == 10.0
    assert g["judging_cell"]["cost_mult"] == 1.5 and g["x1_cell"]["cost_mult"] == 1.0
    assert isinstance(g["kill"]["fired"], bool) and g["verdict"] and g["n_trials"] == 1
    rep = json.loads((out / "report.json").read_text(encoding="ascii"))
    assert rep["gates"] == g
    grid = pd.read_csv(out / "grid.csv")
    n_periods = len(rep["periods"])
    assert len(grid) == 36 * 3 * n_periods and grid["cell"].nunique() == 36  # updated for addendum A1: 36 cells
    assert set(grid["side"]) == {"long", "short", "combined"}
    assert (grid.loc[grid["cost_mult"] == 1.0, "positions_vs_x1"] == 0).all()          # ARITH-2 [SI-65]
    assert (grid["n_spread_blocked"] <= grid["n_triggers"]).all() and (grid["n_positions"] <= grid["n_triggers"]).all()
    pos = pd.read_csv(out / "positions.csv")
    assert list(pos.columns) == list(z.POSITION_COLUMNS)
    assert set(pos["commission_rt_per_lot"]) <= {10.0} and set(pos["cost_mult"]) <= {1.5}
    tr = pd.read_csv(out / "trades.csv")
    assert {"position_id", "leg", "entry_time_utc"} <= set(tr.columns)
    assert len(pd.read_csv(out / "positions_all_cells.csv")) == int(grid.query("side == 'combined' and period == "
                                                                                "'all'")["n_positions"].sum())
    dec = pd.read_csv(out / "decisions.csv")
    assert list(dec.columns) == list(z.DECISION_COLUMNS)
    md = (out / "report.md").read_text(encoding="ascii")
    # updated for addendum A3: the verified preset's [U] tags replace the placeholder's UNVERIFIED banner
    assert md.startswith(HEADER) and "Rules fields with an unverified [U] part" in md and "DSR with N = 1 understates the bar" in md
    assert "M1 resolution not run" in md and zr.S2_DISCLOSURE in md and "## Verdict and gates" in md
    nu = rep["news_unscheduled"]                         # CAUS-2 [SI-63]: the 2020-03-15 row is unscheduled
    assert nu["judging"]["n_rows"] == 1 and nu["master_twin"]["master_closes_only_for_unscheduled"] == []
    assert "- unscheduled rows: 1. D20 blocks entries from 30 min before" in md
    te = rep["decision_log"]["time_exits"]                # CAUS-3 [SI-64]
    assert te["at_16_30_open"] + te["early_close_us_holiday"] + te["early_close_other_day"] == int(
        (pos["exit2_reason"].fillna(pos["exit1_reason"]).eq("time") | pos["exit1_reason"].eq("time")).sum())
    assert set(pos["time_exit_rule"].fillna("")) <= {""} | set(z.TIME_EXIT_RULES)
    assert "D17 time exits: " in md and "[SI-64]" in md
    assert "| cell | triggers | spread blocks | positions | vs x1 |" in md and "[SI-65]" in md
    label = g["judging_cell"]["label"]
    combined = grid[(grid["cell"] == label) & (grid["side"] == "combined") & (grid["period"] == "all")].iloc[0]
    assert g["gates"]["G1"]["value"] == combined["n_positions"]


def test_g4_not_evaluated_with_the_placeholder_rules(files, tmp_path, capsys):
    out = tmp_path / "run"
    # updated for addendum A3: the default is now the verified preset, so the placeholder is named explicitly
    code, _, err = run(["zeno-v1", "run", "--g0-confirmed"] + inputs(files) + ["--out", out,
                        "--rules", "fundingpips-1step-flex-placeholder"] + FAST, capsys)
    assert code == 0, err
    g = json.loads((out / "gates.json").read_text(encoding="ascii"))
    g4 = g["gates"]["G4"]
    assert g4["status"] == "not_evaluated" and "FundingPips rule sheet unverified" in g4["reason"]
    assert "G1-G4 pass" not in g["verdict"]                                  # never an overall pass without G4
    rep = json.loads((out / "report.json").read_text(encoding="ascii"))
    assert rep["prop"]["g4_applicable"] is False and rep["prop"]["judging"]["horizon_days"] == 60
    assert rep["rules_section"]["unverified"] is True
    assert not rep["rules_section"]["info"].get("fallback")  # updated for addendum A3: named, not a fallback
    assert "master_twin" in rep["prop"] and "reference" not in rep["prop"]


def test_run_checks_that_the_g0_sample_belongs_to_the_data_it_judges(files, tmp_path, capsys):
    # RR1-G0-2 [SI-69]: stage 1 on data A, zeno answers y to all 20 rows. Stage 2 on another data set B with A's
    # sample must refuse (none of A's sampled signals is an eligible signal of B), before anything is written;
    # stage 2 on A accepts it and records the match in gates.json.
    sig = tmp_path / "sigA"
    code, _, err = run(["zeno-v1", "signals"] + inputs(files) + ["--out", sig], capsys)
    assert code == 0, err
    g0_path = sig / "g0_sample.csv"
    g0 = pd.read_csv(g0_path, dtype=str, keep_default_na=False)
    g0["agree_y_n"] = "y"
    g0.to_csv(g0_path, index=False)
    bid_b, ask_b = write_pair(tmp_path / "B", z.synthetic_m15_bidask(n_bars=16_000, seed=4, spread=0.05))
    out_b = tmp_path / "runB"
    code, stdout, err = run(["zeno-v1", "run", "--g0-confirmed", "--g0-sample", g0_path, "--m15-bid", bid_b,
                             "--m15-ask", ask_b, "--news", files["news"], "--out", out_b] + FAST, capsys)
    assert code == 2 and "does not belong to this data" in err and "20 of 20" in err, err
    assert not out_b.exists() or not any(out_b.iterdir())
    out_a = tmp_path / "runA"
    code, stdout, err = run(["zeno-v1", "run", "--g0-confirmed", "--g0-sample", g0_path] + inputs(files)
                            + ["--out", out_a] + FAST, capsys)
    assert code == 0, err
    chk = json.loads((out_a / "gates.json").read_text(encoding="ascii"))["gates"]["G0"]["sample_check"]
    assert (chk["ok"], chk["n_rows"], chk["n_matched"], chk["unmatched"]) == (True, 20, 20, [])
    assert chk["declared_cell"] == "evaluation/c10/S1/x1.5" and chk["signals_report"].endswith("signals_report.json")
    assert chk["same_data_files"] is True
    assert "every sampled signal is an eligible signal of this data" in (out_a / "report.md").read_text(encoding="ascii")
    # row by row on data A: a moved stop, a blocked trigger and an unknown time are each named
    prep = z.prepare(files["frame"], z.read_news_csv(files["news"]))
    sigs = pd.read_csv(sig / "signals.csv", dtype=str, keep_default_na=False)
    blocked = sigs[sigs["status"] == "outside_session"].iloc[0]
    bad = g0.copy()
    bad.loc[0, "stop_level"] = str(float(bad.loc[0, "stop_level"]) + 0.01)
    for col in ("signal_time_utc", "side", "entry_price", "stop_level"):
        bad.loc[1, col] = blocked[col]
    bad.loc[2, "signal_time_utc"] = "2015-01-01 00:00:00 UTC"
    chk = zr.g0_sample_check(prep, bad, zr.STAGE1_CELL, 100_000.0)
    assert (chk["ok"], chk["n_matched"]) == (False, 17)
    why = {u["sample_no"]: u["why"] for u in chk["unmatched"]}
    assert "stop" in why[1] and "outside_session" in why[2] and "no trigger" in why[3]


# ---------------------------------------------------------------------------------------
# metric definitions and gates on hand-built tables

def test_metric_definitions_on_a_hand_built_cell():
    # six positions; by FINAL EXIT the order is 1, 3, 2, 4, 5, 6 with net USD -10, +5, -20, -30, -40, +50, so
    # the longest losing streak is 3 (positions 2, 4, 5); by position id it would be 2.
    pos = pd.DataFrame({
        "position_id": [1, 2, 3, 4, 5, 6], "final_exit_stamp": [100, 300, 200, 400, 500, 600],
        "net_pnl_usd": [-10.0, -20.0, 5.0, -30.0, -40.0, 50.0],
        "r_multiple_net": [-1.02, -1.03, 1.0, -1.01, -1.05, 3.0],
        "tp1_reached": [False, False, True, False, False, True], "tp2_reached": [False, False, False, False, False, True],
        "partial_lots": [0.53] * 6, "outcome": ["-1R", "-1R", "+1R(BE)", "-1R", "-1R", "+3R"]})
    m = zr.position_metrics(pos, n_legs=8, span_years=0.5)
    assert m["n_positions"] == 6 and m["n_legs"] == 8 and m["trades_per_year"] == 12.0
    assert m["longest_losing_streak"] == 3
    assert m["hit_rate_tp1"] == pytest.approx(2 / 6) and m["hit_rate_tp2_after_tp1"] == pytest.approx(1 / 2)
    assert m["win_rate"] == pytest.approx(2 / 6)
    assert m["expectancy_r"] == pytest.approx(-0.11 / 6)                     # (-1.02-1.03+1-1.01-1.05+3) / 6
    r = pos["r_multiple_net"].to_numpy()
    sd = math.sqrt(sum((x - r.mean()) ** 2 for x in r) / 5)
    assert m["se_expectancy_r"] == pytest.approx(sd / math.sqrt(6))
    assert m["expectancy_usd"] == pytest.approx(-45.0 / 6) and m["net_usd"] == -45.0
    assert (m["n_outcome_minus_1r"], m["n_outcome_be_plus_1r"], m["n_outcome_plus_3r"]) == (4, 1, 1)
    empty = zr.position_metrics(pos.iloc[:0], 0, None)
    assert empty["expectancy_r"] is None and empty["hit_rate_tp2_after_tp1"] is None
    assert empty["longest_losing_streak"] == 0 and empty["trades_per_year"] is None
    one = zr.position_metrics(pos.iloc[:1], 1, 1.0)
    assert one["se_expectancy_r"] is None and one["hit_rate_tp2_after_tp1"] is None


def test_a_0_01_lot_position_that_reaches_2r_did_not_fill_the_partial():
    # [SI-41]: the +2R hit rate counts positions that FILLED the +2R partial. Capital 1,000 USD: 0.5% = 5 USD,
    # 5 / R 4.70 = 1.06 oz -> 1 oz = 0.01 lot, so half rounds down to 0 (D16): at +2R nothing is closed and only
    # the stop moves to breakeven 2006.30 ([SI-22] corrected); the next bar falls through it (stop at
    # 2006.30 - 0.05). The position reached +2R but filled no partial: not a +2R hit, counted apart.
    from tests.unit.zeno_v1_testkit import Scenario
    sc = Scenario("2024-03-05 12:00")
    sc.setup(1)
    sc.add(2006.0, 2007.0, 2005.0, 2006.5)
    sc.add(2006.5, 2016.0, 2006.5, 2015.0)
    sc.add(2015.0, 2015.0, 2006.0, 2006.5)
    sc.flat(4, 2006.5)
    _, res = sc.run(capital=1_000.0)
    p = res.positions.iloc[0]
    assert (p["units_oz"], p["partial_lots"], bool(p["tp1_reached"]), p["outcome"]) == (1.0, 0.0, True, "other")
    m = zr.position_metrics(res.positions)
    assert m["hit_rate_tp1"] == 0.0 and m["n_tp1"] == 0 and m["hit_rate_tp2_after_tp1"] is None
    assert m["n_tp1_reached_without_partial"] == 1


def _row(base, mult, side, period, n=150, e=0.2, psr=0.97, variant="evaluation", comm=10.0):
    return {"variant": variant, "commission_rt_per_lot": comm, "spread_base": base, "cost_mult": mult, "side": side,
            "period": period, "n_positions": n, "n_legs": n + n // 3, "expectancy_r": e, "se_expectancy_r": 0.05,
            "psr_0": psr, "n_days": 500, "sharpe_daily": 0.05, "dsr_n1": psr}


def _grid(j=None, x1=None, g3=(0.1, -0.05, 0.2, 0.3), sides=(0.3, -0.01), n=150) -> pd.DataFrame:
    """A hand-built grid: (S1, S2) combined E[R] and PSR at x1.5 (j) and x1 (x1); the G3 periods and the
    sides are written for the base that should be judged (the worse one)."""
    j = j or {"S1": (0.20, 0.97), "S2": (0.15, 0.96)}
    x1 = x1 or {"S1": (0.30, 0.99), "S2": (0.25, 0.98)}
    rows = []
    for base, (e, p) in j.items():
        rows.append(_row(base, 1.5, "combined", "all", n, e, p))
        for (label, _, _), v in zip(zr.G3_PERIODS, g3):
            rows.append(_row(base, 1.5, "combined", label, n // 4, v, 0.5))
    for base, (e, p) in x1.items():
        rows.append(_row(base, 1.0, "combined", "all", n, e, p))
        for side, v in zip(("long", "short"), sides):
            rows.append(_row(base, 1.0, side, "all", n // 2, v, 0.5))
    rows.append(_row("S1", 1.5, "combined", "all", n, 9.0, 1.0, comm=5.0))       # other cells are ignored
    rows.append(_row("S1", 1.5, "combined", "all", n, -9.0, 0.0, variant="master"))
    return pd.DataFrame(rows)


def test_gates_known_answers_on_a_hand_built_grid():
    g = zr.gates_from(_grid())
    # worse base at x1.5: S2 (0.15 < 0.20); at x1: S2 (0.25 < 0.30)
    assert g["judging_cell"]["label"] == "evaluation/c10/S2/x1.5" and g["x1_cell"]["label"] == "evaluation/c10/S2/x1"
    G = g["gates"]
    assert G["G1"]["status"] == "pass" and G["G1"]["value"] == 150
    assert G["G2"]["status"] == "pass" and G["G2"]["value"]["expectancy_r"] == 0.15
    assert G["G3"]["status"] == "pass" and G["G3"]["value"]["periods_above_0"] == 3
    assert G["G4"]["status"] == "not_evaluated"
    assert G["G5"]["status"] == "flagged" and G["G5"]["flagged_sides"] == ["short"]   # -0.01 <= 0
    assert g["kill"]["fired"] is False
    assert g["verdict"].startswith("G1-G3 pass; G4 not evaluated")
    # G2 edges: PSR exactly 0.95 passes, 0.9499 fails; E[R] exactly 0 fails
    assert zr.gates_from(_grid(j={"S1": (0.2, 0.97), "S2": (0.1, 0.95)}))["gates"]["G2"]["status"] == "pass"
    assert zr.gates_from(_grid(j={"S1": (0.2, 0.97), "S2": (0.1, 0.9499)}))["gates"]["G2"]["status"] == "fail"
    assert zr.gates_from(_grid(j={"S1": (0.2, 0.97), "S2": (0.0, 0.99)}))["gates"]["G2"]["status"] == "fail"
    # G3: 2 of 4 periods above 0 (0 is not above 0) fails
    assert zr.gates_from(_grid(g3=(0.1, 0.0, 0.2, -0.3)))["gates"]["G3"]["status"] == "fail"
    # kill: G2 at x1 fails on PSR -> fired, and the verdict says so with the spec's sentence
    k = zr.gates_from(_grid(x1={"S1": (0.3, 0.99), "S2": (0.25, 0.90)}))
    assert k["kill"]["fired"] is True and zr.KILL_TEXT in k["verdict"]
    # G1: 99 positions -> insufficient sample, whatever else holds
    assert zr.gates_from(_grid(n=99))["verdict"].startswith("insufficient sample")
    # G4 given and passing -> the after-a-pass note
    g4 = zr.g4_gate(True, "", {"p_breach_daily": 0.04, "p_breach_max": 0.10, "n_sims": 100, "horizon_days": None},
                    "x")
    assert g4["status"] == "pass"
    ok = zr.gates_from(_grid(sides=(0.3, 0.2)), g4)
    assert ok["verdict"].startswith("G1-G4 pass") and zr.AFTER_A_PASS in ok["verdict"]
    assert ok["gates"]["G5"]["status"] == "pass"
    assert zr.g4_gate(True, "", {"p_breach_daily": 0.0501, "p_breach_max": 0.0}, "x")["status"] == "fail"
    assert zr.g4_gate(True, "", {"p_breach_daily": 0.0, "p_breach_max": 0.1001}, "x")["status"] == "fail"


def test_the_cost_multiplier_also_tightens_the_spread_filter_and_the_grid_shows_it():
    # ARITH-2 [SI-65]: rule 10 sees the scaled spread (ask = bid + k x spread, D11), so a canonical long whose
    # entry bar has a 0.40 spread is entered at x1 (0.40 <= 10% of R 4.90) and blocked at x1.5 (0.60 > 10% of
    # R 5.10). The literal reading is kept; each grid row now shows its triggers, the triggers the spread
    # filter blocks and the positions it has against the same cell at x1.
    from tests.unit.zeno_v1_testkit import Scenario
    sc = Scenario("2024-03-05 12:00")
    sc.setup(1)
    sc.add(2006.0, 2007.0, 2005.0, 2006.5, spread=0.40)
    sc.flat(4, 2006.5)
    prep = sc.prepare()
    periods = [p for p in zr.period_table(prep) if p["period"] in ("all", "2024")]
    rows = []
    for k in (1.0, 1.5, 2.0):
        res = z.simulate(prep, z.ZenoConfig(z.ZenoCell("evaluation", 10.0, "S1", k)))
        rows += zr.cell_rows(prep, res, periods)
    grid = zr.grid_frame(rows)
    cols = ["n_triggers", "n_spread_blocked", "n_positions", "positions_vs_x1"]
    for period in ("all", "2024"):
        comb = grid[(grid["side"] == "combined") & (grid["period"] == period)].set_index("cost_mult")
        assert comb.loc[1.0, cols].tolist() == [1, 0, 1, 0]
        assert comb.loc[1.5, cols].tolist() == [1, 1, 0, -1]
        assert comb.loc[2.0, cols].tolist() == [1, 1, 0, -1]
    short = grid[(grid["side"] == "short") & (grid["period"] == "all")]
    assert short["n_triggers"].tolist() == [0, 0, 0] and short["positions_vs_x1"].tolist() == [0, 0, 0]


def test_judging_cell_picks_the_worse_base():
    def pick(s1, s2, mult=1.5):
        return zr.worse_base(_grid(j={"S1": (s1, 0.9), "S2": (s2, 0.9)}), cost_mult=mult)["spread_base"]
    assert pick(0.10, 0.05) == "S2" and pick(0.05, 0.10) == "S1"
    assert pick(0.07, 0.07) == "S2"                                        # a tie goes to S2
    assert pick(None, 0.10) == "S1" and pick(0.10, None) == "S2"           # no position counts as worse
    w = zr.worse_base(_grid())
    assert w["expectancy_r"] == {"S1": 0.20, "S2": 0.15} and w["cost_mult"] == 1.5


def test_g1_counts_positions_not_legs():
    g = zr.gates_from(_grid(n=99))
    assert g["gates"]["G1"]["status"] == "fail" and g["gates"]["G1"]["value"] == 99
    assert g["gates"]["G1"]["verdict_text"] == "insufficient sample"
    assert "132 legs" in g["gates"]["G1"]["reading"]                        # 99 + 99 // 3 legs, not counted
    pos = pd.DataFrame({"position_id": [1, 2], "final_exit_stamp": [1, 2], "net_pnl_usd": [1.0, -1.0],
                        "r_multiple_net": [0.5, -1.0], "tp1_reached": [True, False], "tp2_reached": [False, False],
                        "partial_lots": [0.53, 0.53], "outcome": ["+1R(BE)", "-1R"]})
    assert zr.position_metrics(pos, n_legs=3)["n_positions"] == 2


def test_period_assignment_by_server_day():
    f = z.synthetic_m15_bidask(start=utc("2017-09-01 00:00"), n_bars=14_000, seed=4, spread=0.05)
    prep = z.prepare(f)
    res = z.simulate(prep, z.ZenoConfig(z.ZenoCell("evaluation", 10.0, "S1", 1.0)))
    periods = zr.period_table(prep)
    labels = [p["period"] for p in periods]
    assert labels[:3] == ["all", "2017", "2018"] and labels[-4:] == [p[0] for p in zr.G3_PERIODS]
    g3 = {p["period"]: (p["first_date"], p["last_date"]) for p in periods if p["kind"] == "g3"}
    assert g3["2024-2025-09-27"] == ("2024-01-01", "2025-09-27") and g3["2015-2017"] == ("2015-01-01", "2017-12-31")
    rows = {(r["side"], r["period"]): r for r in zr.cell_rows(prep, res, periods)}
    pos = res.positions
    # the server day of each entry is the ny_17 firm day of its fill [SI-42]
    assert (pd.to_datetime(pos["server_day"]).to_numpy().astype("datetime64[D]").astype(np.int64)
            == np.asarray(cal.firm_day(pos["entry_time"].to_numpy(), "ny_17"))).all()
    n17 = int(pos["server_day"].str.startswith("2017").sum())
    n18 = int(pos["server_day"].str.startswith("2018").sum())
    assert n17 > 0 and n18 > 0 and n17 + n18 == len(pos)
    assert rows[("combined", "2017")]["n_positions"] == n17 == rows[("combined", "2015-2017")]["n_positions"]
    assert rows[("combined", "2018")]["n_positions"] == n18 == rows[("combined", "2018-2020")]["n_positions"]
    assert rows[("combined", "all")]["n_positions"] == len(pos)
    assert rows[("long", "all")]["n_positions"] + rows[("short", "all")]["n_positions"] == len(pos)
    assert rows[("combined", "2021-2023")]["n_positions"] == 0 and rows[("combined", "2021-2023")]["expectancy_r"] is None
    # daily returns split by server day too: the two years' days add up to all days
    assert rows[("combined", "2017")]["n_days"] + rows[("combined", "2018")]["n_days"] == rows[("combined", "all")]["n_days"]
    assert rows[("combined", "all")]["dsr_n1"] == pytest.approx(rows[("combined", "all")]["psr_0"])   # N = 1


def _verified_stand_in(tmp_path: Path) -> Path:
    """A TEST FIXTURE shaped like a verified FundingPips sheet so the G4 code path can run. Its numbers are
    synthetic (0.031 target, 0.061 max loss) and are NOT FundingPips' terms."""
    p = tmp_path / "verified_stand_in.json"
    p.write_text(json.dumps({
        "_comment": "TEST FIXTURE - synthetic numbers, NOT FundingPips' rules",
        "_meta": {"firm": "FundingPips", "plan": "test fixture", "status": "TEST FIXTURE (synthetic numbers)",
                  "verified": True, "sources": ["unit test fixture, not a rule sheet"]},
        "name": "TEST FIXTURE verified-sheet stand-in", "initial_capital": 100_000, "profit_target_pct": 0.031,
        "daily_loss_pct": 0.02, "max_loss_pct": 0.061, "day_boundary": "ny_17"}), encoding="ascii")
    return p


def test_g4_uses_an_unlimited_horizon(files, tmp_path):
    prep = z.prepare(files["frame"])
    cells = [z.ZenoCell("evaluation", 10.0, b, k) for b in z.SPREAD_BASES for k in (1.0, 1.5)]
    rules, info = R.rules_and_info(str(_verified_stand_in(tmp_path)))
    assert zr.g4_applicability(rules, info) == (True, "")
    rep, tables = zr.run_stage(prep, rules, info, n_sims=200, history_reps=0, cells=cells)
    pj = rep["prop"]["judging"]
    assert rep["prop"]["g4_applicable"] is True and pj["horizon_days"] is None
    assert pj["bootstrap"]["horizon_days"] is None and pj["bootstrap_extra"]["horizon_days"] == 60
    g4 = rep["gates"]["gates"]["G4"]
    assert g4["status"] in ("pass", "fail") and g4["value"]["horizon_days"] is None and g4["value"]["n_sims"] == 200
    assert g4["value"]["p_breach_daily"] == pj["bootstrap"]["p_breach_daily"]
    assert "master_twin" not in rep["prop"]                               # not among the cells given
    # the same run under the placeholder: not evaluated, 60 trading days
    prules, pinfo = R.rules_and_info("fundingpips-1step-flex-placeholder")
    ok, why = zr.g4_applicability(prules, pinfo)
    assert not ok and "unverified" in why
    ftmo, finfo = R.rules_and_info("ftmo-1step")
    assert zr.g4_applicability(ftmo, finfo)[0] is False and "not FundingPips" in zr.g4_applicability(ftmo, finfo)[1]
    assert set(tables) >= {"trades", "positions", "decisions", "grid", "positions_all_cells"}


def test_m1_second_run_is_reported_beside_the_first(files, tmp_path):
    prep = z.prepare(files["frame"].iloc[:6000].reset_index(drop=True))
    cells = [z.ZenoCell("evaluation", 10.0, b, k) for b in z.SPREAD_BASES for k in (1.0, 1.5)]
    rules, info = R.rules_and_info("fundingpips-1step-flex-placeholder")
    # M1 bars: each M15 bar split into 15 flat-path M1 bars that keep its open, high, low and close
    fr = files["frame"].iloc[:6000]
    rows = {"bid": [], "ask": []}
    for side in ("bid", "ask"):
        o, h, lo, c = (fr[f"{side}_{k}"].to_numpy() for k in ("open", "high", "low", "close"))
        for i, t in enumerate(fr["time"].to_numpy()):
            path = [o[i]] + [h[i]] * 7 + [lo[i]] * 6 + [c[i]]
            for m in range(15):
                p = path[m]
                rows[side].append((t + 60 * m, p, max(p, path[m + 1] if m < 14 else p), min(p, path[m + 1] if m < 14 else p),
                                   path[m + 1] if m < 14 else c[i]))
    m1 = {s: pd.DataFrame(v, columns=["time", "open", "high", "low", "close"]) for s, v in rows.items()}
    rep, tables = zr.run_stage(prep, rules, info, n_sims=100, history_reps=0, cells=cells, m1=(m1["bid"], m1["ask"]))
    assert isinstance(rep["m1"], dict) and rep["m1"]["run"] is True and "m1_diff" in tables
    assert "D15" in rep["m1"]["note"] and rep["m1"]["combined_all_m15"]["cell"] == rep["judging"]["label"]
    assert "## M1 resolution" in zr.render_run_markdown(rep)


# ---------------------------------------------------------------------------------------
# timing guards (deliverable 5): ~10.7 years of synthetic M15 bars (257,000, the most that fit before the lock)

@pytest.fixture(scope="module")
def big(tmp_path_factory):
    d = tmp_path_factory.mktemp("zeno_big")
    frame = z.synthetic_m15_bidask(n_bars=257_000, seed=1)       # 2015-01-01 .. 2025-09, just before the lock
    bid, ask = write_pair(d, frame)
    news = d / "news.csv"
    news.write_text(NEWS_CSV, encoding="ascii")
    return {"dir": d, "bid": bid, "ask": ask, "news": news, "frame": frame}


def test_timing_guard_signals_on_257k_bars(big, tmp_path, capsys):
    t0 = time.perf_counter()
    code, _, err = run(["zeno-v1", "signals", "--m15-bid", big["bid"], "--m15-ask", big["ask"], "--news", big["news"],
                        "--out", tmp_path / "sig"], capsys)
    dt = time.perf_counter() - t0
    assert code == 0, err
    assert dt < 60, f"zeno-v1 signals on 257,000 bars took {dt:.1f} s"
    assert len(pd.read_csv(tmp_path / "sig" / "signals.csv")) > 1000


def test_timing_guard_full_single_cell_on_257k_bars(big):
    t0 = time.perf_counter()
    prep = z.prepare(big["frame"])
    res = z.simulate(prep, z.ZenoConfig(z.ZenoCell("evaluation", 10.0, "S1", 1.5)))
    eq, tr = z.cell_equity(prep, res)
    path = evaluate_path(eq, tr, R.rules_and_info("fundingpips-1step-flex-placeholder")[0])
    rows = zr.cell_rows(prep, res, zr.period_table(prep))
    dt = time.perf_counter() - t0
    assert dt < 120, f"one full cell (prepare, simulate, equity, evaluator, metrics) took {dt:.1f} s"
    assert path.status in ("running", "breached_daily", "breached_max", "passed") and len(rows) == 3 * 16


# ---------------------------------------------------------------------------------------
# docs (deliverable 6)

def test_methods_maps_every_default_to_a_function_and_an_existing_test():
    root = Path(__file__).resolve().parents[2]
    methods = (root / "propkit" / "METHODS.md").read_text(encoding="ascii")
    section = methods.split("### 8.7 D1-D24 -> function -> test", 1)[1]
    rows = [line for line in section.splitlines() if line.startswith("| D") and not line.startswith("| D |")]
    assert [r.split("|")[1].strip() for r in rows] == [f"D{k}" for k in range(1, 25)]
    refs = []
    for line in section.splitlines():
        if line.startswith("| "):
            cells = [c.strip() for c in line.split("|")]
            refs += [x.strip() for x in cells[-2].split(";") if "::" in x]
    assert len(refs) > 60
    for ref in refs:
        name, test = ref.split("::")
        text = (root / "tests" / "unit" / name).read_text(encoding="utf-8")
        assert f"def {test}(" in text, ref
    readme = (root / "propkit" / "README.md").read_text(encoding="ascii")
    assert "zeno-v1 signals" in readme and "--g0-confirmed" in readme and "g0_sample.csv" in readme
