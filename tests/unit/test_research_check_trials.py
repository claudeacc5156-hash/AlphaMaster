"""Tests for scripts/research/check_trials.py (the amended 3+3 rule checker). Research only."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "research" / "check_trials.py"

_spec = importlib.util.spec_from_file_location("research_check_trials", SCRIPT)
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)

COMMIT = "29a3834" + "f" * 33
NOISE_SHA = "a" * 64
REAL_SHA = "b" * 64


def _run(tag: str, seed: int, score, **over) -> dict:
    rec = {
        "label": "research only", "run_id": f"20261008_0{seed}0000_XAUUSD_H1_seed{seed}_{tag}", "tag": tag,
        "started_utc": f"2026-10-08T0{seed}:00:00+00:00", "git_commit": COMMIT,
        "data_file": "C:\\data\\PLACEBO_H1.parquet" if tag == "noise" else "C:\\data\\XAUUSD_H1.parquet",
        "data_sha256": NOISE_SHA if tag == "noise" else REAL_SHA, "timeframe": "H1",
        "symbol": "PLACEBO" if tag == "noise" else "XAUUSD",
        "steps": 150, "seed": seed, "torch_threads": 4, "batch_size": 192, "score_version": 2,
        "status": "completed", "wall_seconds": 3600.0, "unique_formulas": 20000, "formulas_evaluated": 28800,
        "best_validation_score": score, "best_formula": [3, 7, seed], "best_formula_decoded": f"CLOSE -> TS_MEAN -> T{seed}",
    }
    rec.update(over)
    return rec


def _batch(noise=(0.10, 0.20, 0.30), real=(0.40, 0.50, 0.60)) -> list[dict]:
    return ([_run("noise", s, v) for s, v in zip((1, 2, 3), noise)]
            + [_run("real", s, v) for s, v in zip((1, 2, 3), real)])


def _check(recs: list[dict], **kw) -> dict:
    return ct.check(list(enumerate(recs, start=1)), **kw)


def _precheck(summary: dict, name: str) -> dict:
    return next(c for c in summary["prechecks"] if c["name"] == name)


def _write(path: Path, recs: list) -> Path:
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in recs), encoding="utf-8")
    return path


def test_clean_pass():
    s = _check(_batch())
    assert s["prechecks_passed"]
    assert all(c["passed"] for c in s["prechecks"])
    assert s["rule"]["result"] == "PASS"
    assert s["verdict"] == "PASS"
    assert s["rule"]["completed_pairs"] == [1, 2, 3]
    assert s["rule"]["min_real"] == 0.40 and s["rule"]["max_noise"] == 0.30
    st = s["stats"]
    assert [(r["arm"], r["seed"]) for r in st["ranking"]] == [
        ("real", 3), ("real", 2), ("real", 1), ("noise", 3), ("noise", 2), ("noise", 1)]
    assert st["arms"]["noise"]["n"] == 3
    assert st["arms"]["noise"]["range"] == pytest.approx(0.20)
    assert st["arms"]["real"]["mean"] == pytest.approx(0.50)
    assert st["arms"]["real"]["sd"] == pytest.approx(0.10)
    assert [d["real_minus_noise"] for d in st["seed_differences"]] == pytest.approx([0.30, 0.30, 0.30])


def test_strict_tie_fails():
    s = _check(_batch(noise=(0.10, 0.20, 0.40), real=(0.40, 0.50, 0.60)))
    assert s["prechecks_passed"]
    assert s["verdict"] == "FAIL"
    assert "tie" in s["verdict_reason"]
    assert sum(r["tie"] for r in s["stats"]["ranking"]) == 2


def test_one_real_below_one_noise_fails():
    s = _check(_batch(noise=(0.10, 0.20, 0.45), real=(0.40, 0.50, 0.60)))
    assert s["prechecks_passed"]
    assert s["verdict"] == "FAIL"
    assert "below" in s["verdict_reason"]
    assert s["stats"]["seed_differences"][2]["real_minus_noise"] == pytest.approx(0.15)


def test_missing_pair_is_inconclusive():
    recs = [r for r in _batch() if not (r["tag"] == "real" and r["seed"] == 3)]
    s = _check(recs)
    assert s["rule"]["result"] == "INCONCLUSIVE"
    assert s["rule"]["completed_pairs"] == [1, 2]
    assert s["verdict"] == "INCONCLUSIVE"
    assert not _precheck(s, "three seeds per arm")["passed"]


def test_interrupted_run_is_inconclusive():
    recs = _batch()
    recs[4] = _run("real", 2, None, status="interrupted")
    del recs[4]["best_formula"]
    s = _check(recs)
    assert s["rule"]["result"] == "INCONCLUSIVE"
    assert s["verdict"] == "INCONCLUSIVE"
    chk = _precheck(s, "all runs completed")
    assert not chk["passed"] and "interrupted" in chk["evidence"]
    assert _precheck(s, "three seeds per arm")["passed"]


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_missing_or_nan_score_counts_as_not_completed(bad):
    recs = _batch()
    recs[0]["best_validation_score"] = bad
    s = _check(recs)
    assert s["rule"]["completed_pairs"] == [2, 3]
    assert s["verdict"] == "INCONCLUSIVE"
    assert not _precheck(s, "all runs completed")["passed"]


def test_dirty_commit_is_flagged_and_blocks_pass():
    recs = _batch()
    recs[1]["git_commit"] = COMMIT + "+dirty"
    s = _check(recs)
    assert s["lines"]["selected"] == 6          # selected by prefix, then failed by the pre-check
    chk = _precheck(s, "expected commit")
    assert not chk["passed"] and "+dirty" in chk["evidence"]
    assert s["rule"]["result"] == "PASS"
    assert s["verdict"] == "INCONCLUSIVE"
    assert "expected commit" in s["verdict_reason"]


def test_mismatched_data_hashes_are_flagged():
    recs = _batch()
    recs[2]["data_sha256"] = "c" * 64              # a second file inside the noise arm
    s = _check(recs)
    assert not _precheck(s, "one data file per arm, different between arms")["passed"]
    assert s["verdict"] == "INCONCLUSIVE"

    same = _batch()
    for r in same:
        r["data_sha256"] = REAL_SHA                 # both arms on one file
    s = _check(same)
    chk = _precheck(s, "one data file per arm, different between arms")
    assert not chk["passed"] and "SAME" in chk["evidence"]


def test_settings_mismatch_is_flagged():
    recs = _batch()
    recs[5]["torch_threads"] = 8
    s = _check(recs)
    chk = _precheck(s, "identical settings")
    assert not chk["passed"] and "torch_threads" in chk["evidence"]
    assert s["verdict"] == "INCONCLUSIVE"


def test_duplicate_tag_seed_uses_latest_started_utc():
    # an interrupted run re-run: the latest attempt is the one that completed
    recs = _batch()
    newer = _run("noise", 2, 0.25, run_id="newer", started_utc="2026-10-08T09:00:00+00:00")
    older = _run("noise", 2, None, run_id="older", status="interrupted", started_utc="2026-10-07T09:00:00+00:00")
    recs[1] = newer
    recs.append(older)                              # older run written later in the file
    s = _check(recs)
    chosen = next(r for r in s["selected_runs"] if r["arm"] == "noise" and r["seed"] == 2)
    assert chosen["run_id"] == "newer"
    assert [d["run_id"] for d in s["dropped_duplicates"]] == ["older"]
    assert any("2 runs for noise s2" in w for w in s["warnings"])
    assert _precheck(s, "no completed run re-run")["passed"]
    assert s["verdict"] == "PASS"

    # no completed run yet: the latest attempt is shown
    recs = _batch()
    recs[1] = _run("noise", 2, None, run_id="first", status="error: X", started_utc="2026-10-08T01:00:00+00:00")
    recs.append(_run("noise", 2, None, run_id="second", status="interrupted", started_utc="2026-10-08T05:00:00+00:00"))
    s = _check(recs)
    assert next(r for r in s["selected_runs"] if r["seed"] == 2 and r["arm"] == "noise")["run_id"] == "second"
    assert s["verdict"] == "INCONCLUSIVE"


def test_completed_run_rerun_cannot_turn_fail_into_pass():
    # registered batch fails (real s1 0.30 < noise s2 0.52); a later re-run of real s1 scores 0.60
    recs = _batch(noise=(0.41, 0.52, 0.38), real=(0.30, 0.70, 0.55))
    recs.append(_run("real", 1, 0.60, run_id="rerun", started_utc="2026-10-08T09:00:00+00:00"))
    s = _check(recs)
    chosen = next(r for r in s["selected_runs"] if r["arm"] == "real" and r["seed"] == 1)
    assert chosen["best_validation_score"] == 0.30            # the earliest completed run is the registered look
    chk = _precheck(s, "no completed run re-run")
    assert not chk["passed"] and "0.3" in chk["evidence"] and "0.6" in chk["evidence"]
    assert s["rule"]["result"] == "FAIL" and s["verdict"] == "FAIL"

    # the same re-run on a passing batch blocks the PASS
    recs = _batch()
    recs.append(_run("real", 1, 0.45, run_id="rerun", started_utc="2026-10-08T09:00:00+00:00"))
    s = _check(recs)
    assert s["rule"]["result"] == "PASS" and s["verdict"] == "INCONCLUSIVE"
    assert "no completed run re-run" in s["verdict_reason"]

    # an exact reproduction is harmless
    recs = _batch()
    recs.append(_run("real", 1, 0.40, run_id="repro", started_utc="2026-10-08T09:00:00+00:00"))
    s = _check(recs)
    assert _precheck(s, "no completed run re-run")["passed"] and s["verdict"] == "PASS"

    # a later interrupted attempt does not hide the completed run
    recs = _batch()
    recs.append(_run("real", 1, None, run_id="late", status="interrupted", started_utc="2026-10-08T09:00:00+00:00"))
    s = _check(recs)
    assert s["verdict"] == "PASS"


def test_unrelated_lines_are_ignored_and_counted():
    recs = [
        _run("timing-8t", 1, 0.9, steps=5),
        _run("timing-4t", 1, 0.9, steps=5),
        _run("real", 1, 0.01, steps=300),           # right tag, other step count
        _run("noise", 1, 0.99, git_commit="d38bf72" + "0" * 33),   # right tag, other commit
        {"note": "not a trial"},
    ] + _batch()
    s = _check(recs)
    assert s["lines"]["selected"] == 6
    assert s["lines"]["ignored"] == 5
    assert s["lines"]["ignored_by_reason"]["tag 'timing-8t'"] == 1
    assert s["verdict"] == "PASS"
    s = _check(recs, tags=("timing-8t", "timing-4t"), steps=5)
    assert s["lines"]["selected"] == 2


def test_unique_formula_gap_warns_but_does_not_change_verdict():
    recs = _batch()
    for r in recs:
        if r["tag"] == "real":
            r["unique_formulas"] = 24100            # 20.5% above the noise mean of 20000
    s = _check(recs)
    assert s["unique_formulas"]["warning"]
    assert any("unique_formulas differ" in w for w in s["warnings"])
    assert _precheck(s, "unique_formulas present")["passed"]
    assert s["verdict"] == "PASS"

    for r in recs:
        if r["tag"] == "real":
            r["unique_formulas"] = 24000            # exactly 20%: no warning
    s = _check(recs)
    assert not s["unique_formulas"]["warning"]

    del recs[0]["unique_formulas"]
    s = _check(recs)
    assert not _precheck(s, "unique_formulas present")["passed"]


def test_main_prints_ascii_report_and_writes_json(tmp_path, capsys):
    recs = _batch()
    recs[0]["best_formula_decoded"] = "\u65e0"     # the engine's Chinese "none"
    log = _write(tmp_path / "trials.jsonl", recs + ["{not json"])
    out = tmp_path / "sub" / "summary.json"
    assert ct.main([str(log), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.splitlines()[0] == "RESEARCH ONLY - not trading advice"
    assert text.isascii()
    assert "VERDICT: PASS" in text
    assert "[PASS] expected commit" in text
    flat = " ".join(text.split())
    for line in ct.INTERPRETATION:
        assert " ".join(line.split()) in flat
    assert "tokens [3, 7, 1]" in text
    assert "line 7: not valid JSON" in text
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["verdict"] == "PASS" and data["label"] == "research only"
    assert len(data["selected_runs"]) == 6 and len(data["prechecks"]) == 10
    assert data["rule_result"] == "PASS" and data["batch_matches_registration"] is True


def test_main_inconclusive_still_exits_zero_and_json_is_strict(tmp_path, capsys):
    recs = _batch()
    recs[3]["best_validation_score"] = float("nan")
    log = _write(tmp_path / "trials.jsonl", recs)
    out = tmp_path / "s.json"
    assert ct.main([str(log), "--json", str(out)]) == 0
    assert "VERDICT: INCONCLUSIVE" in capsys.readouterr().out
    data = json.loads(out.read_text(encoding="utf-8"),
                      parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    assert data["selected_runs"][3]["best_validation_score"] is None


def test_usage_and_data_errors_exit_2(tmp_path, capsys):
    assert ct.main([str(tmp_path / "missing.jsonl")]) == 2
    assert "log file not found" in capsys.readouterr().out
    (tmp_path / "empty.jsonl").write_text("\n", encoding="utf-8")
    assert ct.main([str(tmp_path / "empty.jsonl")]) == 2
    log = _write(tmp_path / "t.jsonl", _batch())
    for bad in (["--tags", "noise"], ["--tags", "real", "real"], ["--commit", "xyz"], ["--steps", "0"]):
        with pytest.raises(SystemExit) as e:
            ct.main([str(log), *bad])
        assert e.value.code == 2


def test_tags_accept_comma_or_space(tmp_path, capsys):
    recs = ([_run("placebo2", s, 0.1 * s, data_sha256=NOISE_SHA, data_file="C:\\data\\PLACEBO_H1.parquet",
                  symbol="PLACEBO") for s in (1, 2, 3)]
            + [_run("real2", s, 1.0 + s) for s in (1, 2, 3)])
    log = _write(tmp_path / "t.jsonl", recs)
    assert ct.main([str(log), "--tags", "placebo2,real2"]) == 0
    assert "VERDICT: PASS" in capsys.readouterr().out
    assert ct.main([str(log), "--tags", "placebo2", "real2"]) == 0
    assert "VERDICT: PASS" in capsys.readouterr().out


def test_utf16_log_is_read(tmp_path, capsys):
    log = tmp_path / "t.jsonl"
    log.write_bytes("".join(json.dumps(r) + "\r\n" for r in _batch()).encode("utf-16"))
    assert ct.main([str(log)]) == 0
    assert "VERDICT: PASS" in capsys.readouterr().out


def test_imports_only_the_standard_library():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    assert names <= {"__future__", "argparse", "json", "math", "re", "statistics", "sys", "textwrap",
                     "collections", "datetime", "pathlib"}


# ---------------------------------------------------------------- review fixes
def test_interpretation_is_the_ledger_amendment_text():
    # records/LEDGER.md, 2026-10-08 02:26Z amendment, items (2) to (5), word for word
    text = " ".join(ct.INTERPRETATION)
    for exact in (
        "PASS means only 'real H1 gold beats a placebo with no time structure at 150 steps "
        "(1.67% of the 9,000-step default)'. It does not unlock the locked holdout and does not "
        "justify more seeds of this design.",
        "FAIL means 'no evidence at 150 steps; the search's power is unknown until a positive control "
        "has run'. It does not retire the tool; any deeper rerun is a new test with a new written rule.",
        "The false-positive rate is unknown: about 0.05 only if seed effects are independent between arms; "
        "up to about 0.5 if the data file dominates (each arm reuses one file: one placebo permutation, one "
        "real file); below 0.05 if the shared seed numbers correlate the arms' search paths.",
        "For the question 'directional edge' it can exceed 0.5: the shuffle also destroys volatility "
        "clustering and time-of-day volatility, and the score's Calmar term, half-sample bonus and gates "
        "reward that; in a GARCH simulation with zero directional edge, real beat placebo in 65% of 400 "
        "replications (fixed set of 46 direction-free formulas, no RL search).",
        "Status: exploratory.",
    ):
        assert exact in text
    assert "between below 0.05 and about 0.5" not in text
    assert ct.RULE_TEXT == ("PASS only if all three real best_validation_scores are strictly above all three "
                            "noise scores; a tie is a fail; fewer than three completed pairs is inconclusive.")
    assert text.isascii()


@pytest.mark.parametrize("deviation", ["dirty", "threads", "hash", "locked"])
def test_failed_precheck_does_not_turn_fail_into_inconclusive(deviation):
    recs = _batch(noise=(0.60, 0.70, 0.65), real=(0.10, 0.20, 0.15))
    if deviation == "dirty":
        for r in recs:
            r["git_commit"] = COMMIT + "+dirty"
    elif deviation == "threads":
        recs[5]["torch_threads"] = 8
    elif deviation == "hash":
        recs[0]["data_sha256"] = "c" * 64
    else:
        recs[3]["data_file"] = "C:\\data\\locked_holdout\\XAUUSD_H1.parquet"
    s = _check(recs)
    assert not s["prechecks_passed"]
    assert s["rule"]["result"] == "FAIL"
    assert s["verdict"] == "FAIL"
    assert "Recorded deviations" in s["verdict_reason"]
    assert s["batch_matches_registration"] is False


def test_extra_seed_is_not_used_and_cannot_move_the_verdict():
    fail = _batch(noise=(0.60, 0.70, 0.65), real=(0.10, 0.20, 0.15)) + [_run("real", 4, 0.99)]
    s = _check(fail)
    assert s["verdict"] == "FAIL" and s["lines"]["selected"] == 6
    assert [r["seed"] for r in s["not_used_runs"]] == [4]
    assert any("seed outside 1, 2, 3" in w and "real s4" in w for w in s["warnings"])
    assert _precheck(s, "three seeds per arm")["passed"]
    s = _check(_batch() + [_run("noise", 4, 0.99)])
    assert s["verdict"] == "PASS"


def test_null_commit_runs_are_shown_and_flagged():
    recs = _batch()
    for r in recs:
        r["git_commit"] = None                       # run_trial.py could not run git
    recs.append(_run("noise", 1, 0.95, git_commit="d38bf72" + "0" * 33, run_id="old"))
    s = _check(recs)
    assert s["lines"]["selected"] == 6
    assert [r["best_validation_score"] for r in s["not_used_runs"]] == [0.95]
    chk = _precheck(s, "expected commit")
    assert not chk["passed"] and "git_commit missing" in chk["evidence"]
    assert s["rule"]["result"] == "PASS" and s["verdict"] == "INCONCLUSIVE"
    recs = _batch(noise=(0.60, 0.70, 0.65), real=(0.10, 0.20, 0.15))
    for r in recs:
        r["git_commit"] = None
    assert _check(recs)["verdict"] == "FAIL"
    # a run with a known commit counts before an earlier one without
    recs = _batch()
    recs.append(_run("real", 1, 0.05, git_commit=None, started_utc="2026-10-07T00:00:00+00:00"))
    s = _check(recs)
    assert next(r for r in s["selected_runs"] if r["arm"] == "real" and r["seed"] == 1)["git_commit"] == COMMIT


def test_main_prints_not_used_runs_with_scores(tmp_path, capsys):
    recs = _batch() + [_run("real", 1, 0.777, steps=300)]
    log = _write(tmp_path / "t.jsonl", recs)
    assert ct.main([str(log)]) == 0
    text = capsys.readouterr().out
    assert "NOT used" in text and "0.777" in text


def test_swapped_tags_or_non_placebo_arm_are_not_a_verdict(tmp_path, capsys):
    log = _write(tmp_path / "t.jsonl", _batch())
    assert ct.main([str(log), "--tags", "real", "noise"]) == 0
    text = capsys.readouterr().out
    assert "[FAIL] placebo arm on the placebo file" in text
    assert "RULE RESULT: FAIL" in text and "VERDICT: INCONCLUSIVE" in text
    assert "--tags" in " ".join(text.split("VERDICT:")[1].split())
    assert "Arms: noise (placebo) = XAUUSD_H1.parquet; real = PLACEBO_H1.parquet" in text

    recs = _batch()
    for r in recs:
        if r["tag"] == "noise":                       # 'placebo' arm trained on a second real file
            r.update(data_file="C:\\data\\XAUUSD_H1_full.parquet", symbol="XAUUSD")
    s = _check(recs)
    assert not _precheck(s, ct.ARM_CHECK)["passed"] and s["verdict"] == "INCONCLUSIVE"
    s = _check(recs, placebo_marker="FULL")
    assert _precheck(s, ct.ARM_CHECK)["passed"] and s["verdict"] == "PASS"


def test_mixed_timeframes_are_flagged():
    recs = _batch()
    for r in recs:
        if r["tag"] == "real":
            r.update(timeframe="M15", data_file="C:\\data\\XAUUSD_M15.parquet")
    s = _check(recs)
    chk = _precheck(s, "one timeframe")
    assert not chk["passed"] and "M15" in chk["evidence"]
    assert s["verdict"] == "INCONCLUSIVE"


@pytest.mark.parametrize("target", ["log", "other.jsonl", "notes.txt"])
def test_json_never_overwrites_the_log_or_a_foreign_file(tmp_path, capsys, target):
    log = _write(tmp_path / "trials.jsonl", _batch())
    other = tmp_path / target
    if target == "notes.txt":
        other.write_text("keep me\n", encoding="utf-8")
    before = log.read_bytes()
    dest = log if target == "log" else other
    assert ct.main([str(log), "--json", str(dest)]) == 2
    assert "ERROR" in capsys.readouterr().out
    assert log.read_bytes() == before
    if target == "notes.txt":
        assert other.read_text(encoding="utf-8") == "keep me\n"
    else:
        assert target == "log" or not other.exists()


def test_json_may_replace_an_earlier_summary(tmp_path, capsys):
    log = _write(tmp_path / "trials.jsonl", _batch())
    out = tmp_path / "check.json"
    assert ct.main([str(log), "--json", str(out)]) == 0
    assert ct.main([str(log), "--json", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["tool"] == ct.TOOL_ID


@pytest.mark.parametrize("extra", [b"", b"\n"])
def test_utf16_log_with_utf8_lines_appended(tmp_path, capsys, extra):
    recs = _batch()
    recs[0]["best_formula_decoded"] = "\u65e0"
    log = tmp_path / "t.jsonl"
    head = "".join(json.dumps(r, ensure_ascii=False) + "\r\n" for r in recs[:4]).encode("utf-16")
    tail = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs[4:]).encode("utf-8") + extra
    for size_fix in (b"", b" "):                     # odd and even total sizes
        log.write_bytes(head + tail + size_fix)
        assert ct.main([str(log)]) == 0
        text = capsys.readouterr().out
        assert "Records read: 6" in text and "VERDICT: PASS" in text
        assert "continues in UTF-8" in text


def test_broken_utf16_log_is_a_data_error(tmp_path, capsys):
    log = tmp_path / "t.jsonl"
    log.write_bytes(b"\xff\xfe" + "{}".encode("utf-16-le") + b"\x00\xd8\x00\xd8\x00")   # lone surrogates
    assert ct.main([str(log)]) == 2
    assert "ERROR" in capsys.readouterr().out


def test_log_path_after_tags_gets_a_clear_error(tmp_path, capsys):
    log = _write(tmp_path / "t.jsonl", _batch())
    for args in (["--tags", "noise", "real", str(log)], ["--tags", "noise,real", str(log)]):
        with pytest.raises(SystemExit) as e:
            ct.main(args)
        assert e.value.code == 2
        assert "put the log path before --tags" in capsys.readouterr().err
    assert ct.main([str(log), "--tags", "noise", "real"]) == 0


def test_locked_holdout_paths_are_never_read_or_written(tmp_path, capsys, monkeypatch):
    calls = []
    real_open = open
    harmless = {"read_bytes": b"", "read_text": "", "is_file": False, "exists": False}

    def spy(name):
        def wrapper(self, *a, **k):
            calls.append((name, str(self)))
            return harmless[name]
        return wrapper

    for name in harmless:
        monkeypatch.setattr(Path, name, spy(name))
    monkeypatch.setattr("builtins.open", lambda f, *a, **k: (calls.append(("open", str(f))), real_open(f, *a, **k))[1])
    results = []
    try:
        for bad in ("..\\AlphaMaster\\research\\data\\locked_holdout\\XAUUSD_H1.parquet",
                    "data/XAUUSD_H1_holdout.parquet.locked", "data/LOCKED_HOLDOUT/x.jsonl"):
            results.append((ct.main([bad]), capsys.readouterr().out))
    finally:
        monkeypatch.undo()
    assert calls == []
    for code, out in results:
        assert code == 2 and "locked holdout" in out

    log = _write(tmp_path / "t.jsonl", _batch())
    for bad in (tmp_path / "locked_holdout" / "s.json", tmp_path / "s.json.locked"):
        assert ct.main([str(log), "--json", str(bad)]) == 2
        assert not bad.exists() and not bad.parent.joinpath("s.json").exists()
    assert "locked holdout" in capsys.readouterr().out
