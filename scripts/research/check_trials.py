"""scripts/research/check_trials.py - apply the amended 3+3 placebo rule to logs/trials.jsonl.

Usage (from the repo root; Windows PowerShell shown):
    $env:PYTHONUTF8 = "1"
    python scripts\\research\\check_trials.py
    python scripts\\research\\check_trials.py logs\\trials.jsonl --json logs\\check_trials.json

What it does:
  * reads the trial log that scripts/run_trial.py appends to and selects the batch: by default
    the lines tagged 'noise' (placebo arm) or 'real' (real arm) with steps == 150, seed 1, 2 or
    3 and a git_commit starting with 29a3834. A run whose git_commit is missing (run_trial.py
    could not run git) is kept and fails the commit pre-check. Every other line (e.g. the
    timing runs) is ignored and counted; lines with a batch tag that are not used are listed
    with their scores. If a (tag, seed) appears more than once, the earliest completed run
    counts (it is the registered look); when none completed, the latest attempt is used. A
    warning names the others;
  * runs the pre-checks and prints PASS/FAIL with the evidence: seeds 1, 2, 3 in each arm,
    every run completed with a score, no completed run re-run with a different score, one
    clean commit (a '+dirty' or missing commit fails), one data file per arm and a different
    one for each arm, the placebo arm on the placebo file (its file name or symbol contains
    PLACEBO) and the real arm not, one timeframe, identical steps/batch_size/torch_threads/
    score_version, unique_formulas present, and no run on a locked holdout file;
  * applies the amended rule mechanically: PASS only if all three real best_validation_scores
    are strictly above all three noise scores; a tie fails; fewer than three completed seed
    pairs is INCONCLUSIVE (a missing or non-finite score counts as not completed);
  * VERDICT is the rule result, except: a PASS with any failed pre-check is INCONCLUSIVE, and
    if the placebo-arm pre-check fails (the arms may be swapped) any result is INCONCLUSIVE.
    A FAIL stays FAIL when other pre-checks fail; they are listed as recorded deviations;
  * prints the score ranking, per-arm statistics, the seed-matched differences, the best
    formulas and the fixed interpretation text (items 2 to 5 of the 2026-10-08 02:26Z
    amendment in records/LEDGER.md). --json PATH also writes a summary file; it refuses to
    write over the trial log or any file that is not an earlier summary.

The verdict is in the printed report, not in the exit code: the exit code is 0 for PASS, FAIL
and INCONCLUSIVE alike, and 2 only for usage or data errors (e.g. the log file is missing).
Nothing here imports torch or the engine, and no path containing 'locked_holdout' or ending
in '.locked' is ever read or written. Research only.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BANNER = "RESEARCH ONLY - not trading advice"
TOOL_ID = "scripts/research/check_trials.py"
DEFAULT_LOG = Path("logs") / "trials.jsonl"
DEFAULT_TAGS = ("noise", "real")
DEFAULT_STEPS = 150
DEFAULT_COMMIT = "29a3834"
DEFAULT_PLACEBO_MARKER = "PLACEBO"
EXPECTED_SEEDS = (1, 2, 3)
PAIRS_NEEDED = 3
ARMS = ("noise", "real")
SETTINGS_FIELDS = ("steps", "batch_size", "torch_threads", "score_version")
UNIQUE_WARN_FRACTION = 0.20
ARM_CHECK = "placebo arm on the placebo file"
_HEX_COMMIT = re.compile(r"[0-9a-f]{7,40}")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Item (1) of the amendment, word for word.
RULE_TEXT = (
    "PASS only if all three real best_validation_scores are strictly above all three noise "
    "scores; a tie is a fail; fewer than three completed pairs is inconclusive."
)
# Items (2) to (5) of the amendment, word for word. Printed as is, whatever the verdict.
INTERPRETATION_SOURCE = ("records/LEDGER.md, 2026-10-08 02:26Z amendment to the 3 + 3 rule "
                         "(alphamaster-xau-h1), items (2) to (5)")
INTERPRETATION = (
    "(2) The 'p = 1/20' statement is withdrawn. The false-positive rate is unknown: about 0.05 "
    "only if seed effects are independent between arms; up to about 0.5 if the data file "
    "dominates (each arm reuses one file: one placebo permutation, one real file); below 0.05 "
    "if the shared seed numbers correlate the arms' search paths. For the question 'directional "
    "edge' it can exceed 0.5: the shuffle also destroys volatility clustering and time-of-day "
    "volatility, and the score's Calmar term, half-sample bonus and gates reward that; in a "
    "GARCH simulation with zero directional edge, real beat placebo in 65% of 400 replications "
    "(fixed set of 46 direction-free formulas, no RL search).",
    "(3) Status: exploratory.",
    "(4) PASS means only 'real H1 gold beats a placebo with no time structure at 150 steps "
    "(1.67% of the 9,000-step default)'. It does not unlock the locked holdout and does not "
    "justify more seeds of this design.",
    "(5) FAIL means 'no evidence at 150 steps; the search's power is unknown until a positive "
    "control has run'. It does not retire the tool; any deeper rerun is a new test with a new "
    "written rule.",
)


class DataError(Exception):
    """The log cannot be used at all (missing, unreadable or without any trial record)."""


def _ascii(value) -> str:
    """Plain-ASCII text for the console (non-ASCII becomes a \\u escape)."""
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def _is_locked(path: Path) -> bool:
    """True for a path into the locked holdout. Only the path text is looked at, never the file."""
    def bad(text: str) -> bool:
        text = text.lower().rstrip("\\/")
        return "locked_holdout" in text or text.endswith(".locked")
    if bad(str(path)):
        return True
    try:
        return bad(str(path.resolve()))
    except (OSError, RuntimeError):
        return False


def _as_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _score(rec: dict) -> float | None:
    """The run's best_validation_score, or None when it is missing, None, NaN or infinite."""
    value = rec.get("best_validation_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _finished(rec: dict) -> bool:
    return rec.get("status") == "completed"


def _scored(rec: dict) -> bool:
    """Completed for the rule: status 'completed' and a finite score."""
    return _finished(rec) and _score(rec) is not None


def _started(rec: dict) -> datetime | None:
    text = rec.get("started_utc")
    if not isinstance(text, str):
        return None
    try:
        t = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


def _label(arm: str, seed) -> str:
    return f"{arm} s{seed}"


def _file_name(path) -> str:
    return re.split(r"[\\/]", path)[-1] if isinstance(path, str) else "?"


def _short_commit(value) -> str:
    if not isinstance(value, str):
        return str(value)
    base, sep, suffix = value.partition("+")
    return base[:10] + sep + suffix


def _fmt(value, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _run_desc(no: int, rec: dict) -> str:
    return (f"line {no} (started {rec.get('started_utc')}, {rec.get('status')}, "
            f"score {rec.get('best_validation_score')!r})")


def _json_safe(value):
    """Strict-JSON copy: NaN and infinities become None."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


# --------------------------------------------------------------------------- reading
def _decode_utf16(raw: bytes, path: Path) -> tuple[str, list[str]]:
    """Decode a log that starts with a UTF-16 byte-order mark (e.g. re-saved by PowerShell 5).

    run_trial.py appends UTF-8, so such a file can continue in UTF-8 after its last UTF-16
    line. In UTF-16 every ASCII character carries a zero byte and UTF-8 JSON has none, so the
    UTF-8 tail starts right after the last zero byte.
    """
    mixed = (f"{path} starts as UTF-16 (byte-order mark) but is not valid UTF-16; it may mix "
             f"UTF-16 and UTF-8 text. Open it in a text editor and save it as UTF-8")
    last_nul = raw.rfind(b"\x00")
    cut = len(raw) if last_nul < 0 else min(len(raw), last_nul + (2 if raw[:2] == b"\xfe\xff" else 1))
    head, tail = raw[:cut], raw[cut:]
    try:
        text = head.decode("utf-16")
        tail_text = tail.decode("utf-8") if tail else ""
    except UnicodeDecodeError:
        try:
            return raw.decode("utf-16"), []
        except UnicodeDecodeError:
            raise DataError(mixed) from None
    if not tail_text.strip():
        return text + tail_text, []
    if text and not text.endswith(("\n", "\r")):
        text += "\n"
    return text + tail_text, [f"the log starts in UTF-16 (e.g. re-saved by Windows PowerShell 5) and "
                              f"continues in UTF-8 (lines appended later by run_trial.py); both parts "
                              f"were read"]


def read_log(path: Path) -> tuple[list[tuple[int, dict]], list[str]]:
    """Return ((line number, record) for every JSON object line, warnings about bad lines)."""
    if _is_locked(path):
        raise DataError(f"refusing to read {path}: it points into the locked holdout ('locked_holdout' "
                        f"or '.locked'). Pass the trial log, e.g. logs\\trials.jsonl")
    if not path.is_file():
        raise DataError(f"log file not found: {path.resolve()}. Run this from the repo root (where "
                        f"logs\\trials.jsonl is), or pass the log path.")
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise DataError(f"cannot read {path.resolve()}: {e}") from e
    warnings: list[str] = []
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text, warnings = _decode_utf16(raw, path)
        lines: list = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    else:
        lines = raw.splitlines()
    records: list[tuple[int, dict]] = []
    for no, line in enumerate(lines, start=1):
        if isinstance(line, bytes):
            if b"\x00" in line:
                warnings.append(f"line {no}: contains zero bytes (UTF-16 text inside a UTF-8 log?); skipped")
                continue
            try:
                line = line.decode("utf-8-sig" if no == 1 else "utf-8")
            except UnicodeDecodeError:
                warnings.append(f"line {no}: not UTF-8 text; skipped")
                continue
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            warnings.append(f"line {no}: not valid JSON (a run may still be writing it); skipped")
            continue
        if not isinstance(rec, dict):
            warnings.append(f"line {no}: not a trial record; skipped")
            continue
        records.append((no, rec))
    if not records:
        raise DataError(f"no trial records in {path.resolve()}")
    return records, warnings


# --------------------------------------------------------------------------- selection
def select_runs(records: list[tuple[int, dict]], tags: tuple[str, str], steps: int, commit: str):
    """Pick the batch's runs. Returns (runs, ignored reasons, not used, dropped, re-runs, warnings).

    runs maps (arm, seed) to (line number, record); arm is 'noise' or 'real'. 'not used' lists
    the lines with a batch tag that the selection left out (other steps, commit or seed). For a
    (tag, seed) with several lines the earliest completed run is kept (the registered look; a
    run with a known commit before one without); with none completed, the latest attempt.
    're-runs' maps (arm, seed) to [(line, score)] when it has more than one completed run.
    """
    arm_of = {tags[0]: "noise", tags[1]: "real"}
    ignored: Counter = Counter()
    not_used: list[dict] = []
    candidates: dict[tuple[str, int], list[tuple[int, dict]]] = {}
    for no, rec in records:
        tag = rec.get("tag")
        if not isinstance(tag, str) or tag not in arm_of:
            ignored[f"tag {tag!r}"] += 1
            continue
        seed = _as_int(rec.get("seed"))
        git_commit = rec.get("git_commit")
        if _as_int(rec.get("steps")) != steps:
            reason = f"steps {rec.get('steps')!r}"
        elif isinstance(git_commit, str) and not git_commit.startswith(commit):
            reason = f"commit {_short_commit(git_commit)}"
        elif seed is None:
            reason = "no usable seed"
        elif seed not in EXPECTED_SEEDS:
            reason = f"seed {seed}, not one of the registered seeds 1, 2, 3"
        else:
            candidates.setdefault((arm_of[tag], seed), []).append((no, rec))
            continue
        ignored[f"tag {tag!r} with {reason}"] += 1
        not_used.append({"line": no, "arm": arm_of[tag], "tag": tag, "seed": rec.get("seed"),
                         "reason": reason, "steps": rec.get("steps"), "git_commit": git_commit,
                         "status": rec.get("status"), "best_validation_score": rec.get("best_validation_score")})

    warnings: list[str] = []
    stray = [r for r in not_used if r["reason"].startswith("seed ")]
    if stray:
        warnings.append(f"{len(stray)} run(s) with a batch tag, {steps} steps and the expected commit have a "
                        f"seed outside 1, 2, 3 and are not used: "
                        + "; ".join(f"{_label(r['arm'], r['seed'])} line {r['line']} score "
                                    f"{r['best_validation_score']!r}" for r in stray)
                        + ". The rule uses seeds 1, 2, 3 only")

    runs: dict[tuple[str, int], tuple[int, dict]] = {}
    dropped: list[dict] = []
    reruns: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for key in sorted(candidates):
        items = sorted(candidates[key], key=lambda it: (_started(it[1]) or _EPOCH, it[0]))
        scored = [it for it in items if _scored(it[1])]
        if scored:
            known = [it for it in scored if isinstance(it[1].get("git_commit"), str)]
            keep = (known or scored)[0]
            why = "the earliest completed run, which is the registered look"
        else:
            keep = items[-1]
            why = "the latest attempt, as none completed"
        runs[key] = keep
        if len(scored) > 1:
            reruns[key] = [(no, _score(rec)) for no, rec in scored]
        if len(items) > 1:
            others = [it for it in items if it is not keep]
            for no, rec in others:
                dropped.append({"arm": key[0], "seed": key[1], "line": no, "run_id": rec.get("run_id"),
                                "started_utc": rec.get("started_utc"), "status": rec.get("status"),
                                "best_validation_score": rec.get("best_validation_score")})
            warnings.append(f"{len(items)} runs for {_label(*key)}: using {_run_desc(*keep)}, {why}; ignoring "
                            + ", ".join(_run_desc(no, rec) for no, rec in others))
    return runs, ignored, not_used, dropped, reruns, warnings


# --------------------------------------------------------------------------- checks
def _check(name: str, passed: bool, evidence: str) -> dict:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _group(runs: dict, field: str) -> dict:
    """{value: [labels]} for one field over the selected runs."""
    out: dict = {}
    for (arm, seed), (_, rec) in sorted(runs.items()):
        value = rec.get(field)
        key = json.dumps(value, sort_keys=True)      # hashable for lists/dicts, keeps None distinct
        out.setdefault(key, []).append(_label(arm, seed))
    return out


def _placebo_like(rec: dict, marker: str) -> bool:
    symbol = rec.get("symbol")
    return (marker.lower() in _file_name(rec.get("data_file")).lower()
            or (isinstance(symbol, str) and marker.lower() in symbol.lower()))


def run_prechecks(runs: dict, commit: str, placebo_marker: str = DEFAULT_PLACEBO_MARKER,
                  reruns: dict | None = None) -> list[dict]:
    checks: list[dict] = []
    n = len(runs)

    seeds = {arm: sorted(s for (a, s) in runs if a == arm) for arm in ARMS}
    expected = list(EXPECTED_SEEDS)
    checks.append(_check(
        "three seeds per arm", all(seeds[arm] == expected for arm in ARMS),
        f"noise seeds {seeds['noise']}, real seeds {seeds['real']} (expected {expected} in each arm)"))

    if not runs:
        checks.append(_check("all runs completed", False, "no runs selected"))
    else:
        bad = []
        for (arm, seed), (_, rec) in sorted(runs.items()):
            if not _finished(rec):
                bad.append(f"{_label(arm, seed)} status {rec.get('status')!r}")
            elif _score(rec) is None:
                bad.append(f"{_label(arm, seed)} completed but best_validation_score is "
                           f"{rec.get('best_validation_score')!r}")
        done = sum(_scored(rec) for _, rec in runs.values())
        checks.append(_check(
            "all runs completed", not bad,
            f"{done} of {n} selected runs have status 'completed' and a finite score"
            + (f"; not completed: {'; '.join(bad)}" if bad else "")))

    reruns = reruns or {}
    desc = "; ".join(f"{_label(*k)}: " + ", ".join(f"line {no} score {s!r}" for no, s in v)
                     for k, v in sorted(reruns.items()))
    differ = sorted(_label(*k) for k, v in reruns.items() if len({s for _, s in v}) > 1)
    if differ:
        checks.append(_check("no completed run re-run", False,
                             f"completed run(s) re-run with a different score: {desc}. The earliest completed "
                             f"run is used; re-running a finished seed is a second look"))
    else:
        checks.append(_check("no completed run re-run", True,
                             f"re-run(s) reproduced the score exactly: {desc}" if reruns
                             else "no (arm, seed) has more than one completed run"))

    commits = _group(runs, "git_commit")
    problems = []
    for key, labels in commits.items():
        value = json.loads(key)
        if value is None:
            problems.append(f"git_commit missing for {', '.join(labels)} (run_trial.py could not run git, "
                            f"e.g. git not on PATH), so the commit cannot be confirmed")
            continue
        clean = (isinstance(value, str) and _HEX_COMMIT.fullmatch(value) is not None
                 and value.startswith(commit) and (len(commit) < 40 or value == commit))
        if not clean:
            problems.append(f"{value!r} is not the clean commit {commit} ({', '.join(labels)})")
    if not runs:
        checks.append(_check("expected commit", False, "no runs selected"))
    elif problems or len(commits) > 1:
        found = "; ".join(f"{json.loads(k)} ({', '.join(v)})" for k, v in commits.items())
        checks.append(_check("expected commit", False,
                             ("; ".join(problems) + ". " if problems else "")
                             + f"commits found: {found}"))
    else:
        checks.append(_check("expected commit", True,
                             f"all {n} runs at {json.loads(next(iter(commits)))} (no '+dirty')"))

    arm_sha: dict[str, list[str]] = {}
    parts = []
    for arm in ARMS:
        arm_runs = {k: v for k, v in runs.items() if k[0] == arm}
        groups = _group(arm_runs, "data_sha256")
        arm_sha[arm] = [json.loads(k) for k in groups]
        if not groups:
            parts.append(f"{arm} arm: no runs")
            continue
        files = sorted({_file_name(rec.get("data_file")) for _, rec in arm_runs.values()})
        desc = "; ".join(f"sha256 {str(json.loads(k))[:12]} ({', '.join(v)})" for k, v in groups.items())
        parts.append(f"{arm} arm: {len(groups)} data hash(es): {desc}, file(s) {', '.join(files)}")
    one_each = all(len(arm_sha[arm]) == 1 and isinstance(arm_sha[arm][0], str) and arm_sha[arm][0]
                   for arm in ARMS)
    differ = one_each and arm_sha["noise"][0] != arm_sha["real"][0]
    if one_each and not differ:
        parts.append("the two arms used the SAME data file")
    elif differ:
        parts.append("the arms use different files")
    checks.append(_check("one data file per arm, different between arms", differ, "; ".join(parts)))

    wrong = []
    parts = []
    for arm in ARMS:
        arm_runs = sorted((k, v) for k, v in runs.items() if k[0] == arm)
        names = sorted({f"{_file_name(rec.get('data_file'))} (symbol {rec.get('symbol')})"
                        for _, (_, rec) in arm_runs})
        parts.append(f"{arm} arm: {', '.join(names) if names else 'no runs'}")
        wrong += [_label(*k) for k, (_, rec) in arm_runs if _placebo_like(rec, placebo_marker) != (arm == "noise")]
    evidence = "; ".join(parts) + f"; placebo marker {placebo_marker!r}"
    if wrong:
        evidence += (f". Wrong arm for {', '.join(wrong)}: the placebo (noise) arm's file name or symbol must "
                     f"contain {placebo_marker!r} and the real arm's must not. Are the --tags in the wrong "
                     f"order (placebo tag first)? If the placebo file has another name, use --placebo-marker")
    checks.append(_check(ARM_CHECK, all(seeds[arm] for arm in ARMS) and not wrong, evidence))

    groups = _group(runs, "timeframe")
    values = [json.loads(k) for k in groups]
    checks.append(_check(
        "one timeframe", len(groups) == 1 and isinstance(values[0], str) and bool(values[0]),
        "; ".join(f"{json.loads(k)} ({', '.join(v)})" for k, v in groups.items()) if runs else "no runs selected"))

    settings_ok = bool(runs)
    parts = []
    for field in SETTINGS_FIELDS:
        groups = _group(runs, field)
        values = [json.loads(k) for k in groups]
        if len(groups) == 1 and values[0] is not None:
            parts.append(f"{field} {values[0]}")
        else:
            settings_ok = False
            parts.append(f"{field} " + ("not identical: " if len(groups) > 1 else "missing: ")
                         + "; ".join(f"{json.loads(k)} ({', '.join(v)})" for k, v in groups.items()))
    checks.append(_check("identical settings", settings_ok, ", ".join(parts) if runs else "no runs selected"))

    missing = []
    per_arm = []
    for arm in ARMS:
        items = []
        for (a, seed), (_, rec) in sorted(runs.items()):
            if a != arm:
                continue
            value = _as_int(rec.get("unique_formulas"))
            if value is None or value < 0:
                missing.append(_label(arm, seed))
            items.append(f"s{seed}={_fmt(value)}")
        per_arm.append(f"{arm} " + (" ".join(items) if items else "none"))
    checks.append(_check("unique_formulas present", bool(runs) and not missing,
                         "; ".join(per_arm) + (f"; missing or invalid: {', '.join(missing)}" if missing else "")))

    locked = [_label(arm, seed) for (arm, seed), (_, rec) in sorted(runs.items())
              if isinstance(rec.get("data_file"), str)
              and ("locked_holdout" in rec["data_file"].lower() or rec["data_file"].lower().endswith(".locked"))]
    checks.append(_check("locked holdout untouched", not locked,
                         f"data_file points into the locked holdout for {', '.join(locked)}" if locked
                         else "no selected run's data_file contains 'locked_holdout' or ends in '.locked'"))
    return checks


def unique_formula_summary(runs: dict) -> tuple[dict, str | None]:
    """Arm means of unique_formulas over finished runs, and a warning when they differ by > 20%."""
    means = {}
    for arm in ARMS:
        values = [_as_int(rec.get("unique_formulas")) for (a, _), (_, rec) in runs.items()
                  if a == arm and _finished(rec)]
        values = [v for v in values if v is not None and v >= 0]
        means[arm] = statistics.fmean(values) if values else None
    info = {"noise_mean": means["noise"], "real_mean": means["real"], "difference_fraction": None,
            "warn_above_fraction": UNIQUE_WARN_FRACTION, "warning": False}
    if means["noise"] is None or means["real"] is None:
        return info, None
    diff = means["real"] - means["noise"]
    if means["noise"] > 0:
        frac = abs(diff) / means["noise"]
        info["difference_fraction"] = frac
        info["warning"] = frac > UNIQUE_WARN_FRACTION
    else:
        info["warning"] = diff != 0
    if not info["warning"]:
        return info, None
    pct = "n/a" if info["difference_fraction"] is None else f"{100 * info['difference_fraction']:.1f}%"
    return info, (f"unique_formulas differ between arms by {pct} of the noise-arm mean (real mean "
                  f"{means['real']:.1f} vs noise mean {means['noise']:.1f}; warning above "
                  f"{100 * UNIQUE_WARN_FRACTION:.0f}%). The arms searched different numbers of formulas, "
                  f"so their best scores had different numbers of tries.")


# --------------------------------------------------------------------------- rule
def apply_rule(runs: dict) -> dict:
    scores = {key: _score(rec) for key, (_, rec) in runs.items() if _scored(rec) and key[1] in EXPECTED_SEEDS}
    pairs = sorted(seed for (arm, seed) in scores if arm == "real" and ("noise", seed) in scores)
    out = {"completed_pairs": pairs, "pairs_needed": PAIRS_NEEDED, "min_real": None, "max_noise": None}
    if len(pairs) < PAIRS_NEEDED:
        out.update(result="INCONCLUSIVE",
                   reason=f"only {len(pairs)} completed seed pair(s) {pairs}; the rule needs {PAIRS_NEEDED}")
        return out
    real = {s: scores[("real", s)] for s in pairs}
    noise = {s: scores[("noise", s)] for s in pairs}
    low_seed = min(real, key=real.get)
    high_seed = max(noise, key=noise.get)
    min_real, max_noise = real[low_seed], noise[high_seed]
    out.update(min_real=min_real, max_noise=max_noise)
    if min_real > max_noise:
        out.update(result="PASS", reason=f"the lowest real score {min_real!r} (real s{low_seed}) is strictly "
                                         f"above the highest noise score {max_noise!r} (noise s{high_seed})")
    elif min_real == max_noise:
        out.update(result="FAIL", reason=f"tie: the lowest real score (real s{low_seed}) equals the highest "
                                         f"noise score (noise s{high_seed}), both {min_real!r}; a tie fails")
    else:
        out.update(result="FAIL", reason=f"the lowest real score {min_real!r} (real s{low_seed}) is below the "
                                         f"highest noise score {max_noise!r} (noise s{high_seed})")
    return out


def decide(rule: dict, prechecks: list[dict]) -> tuple[str, str]:
    """The headline verdict from the rule result and the pre-checks.

    A failed pre-check can only block a PASS (to INCONCLUSIVE). A FAIL stays FAIL, with the
    failed pre-checks recorded, because FAIL already means 'no evidence' and turning it into
    INCONCLUSIVE would invite a second look. The one exception is the placebo-arm check: if it
    fails, the arms may be swapped, so neither a PASS nor a FAIL on them counts.
    """
    failed = [c["name"] for c in prechecks if not c["passed"]]
    result = rule["result"]
    if result == "INCONCLUSIVE":
        return "INCONCLUSIVE", rule["reason"] + (f"; pre-check(s) also failed: {', '.join(failed)}" if failed else "")
    if ARM_CHECK in failed:
        return "INCONCLUSIVE", (f"the rule gives {result} on the arms as tagged, but the pre-check '{ARM_CHECK}' "
                                f"failed, so the arms may be swapped or mislabelled and the result does not count "
                                f"either way. Check the --tags order (placebo tag first) and the data files. "
                                f"Failed pre-check(s): {', '.join(failed)}")
    if not failed:
        return result, rule["reason"]
    if result == "PASS":
        return "INCONCLUSIVE", (f"the rule alone gives PASS ({rule['reason']}), but pre-check(s) failed: "
                                f"{', '.join(failed)}. A PASS counts only when the batch matches the registration")
    return "FAIL", (f"{rule['reason']}. Recorded deviations (failed pre-checks): {', '.join(failed)}. They do not "
                    f"change a FAIL, which means no evidence; any rerun would be a new test with a new written rule")


def _arm_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "min": None, "max": None, "range": None, "mean": None, "sd": None}
    return {"n": len(values), "min": min(values), "max": max(values), "range": max(values) - min(values),
            "mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else None}


def score_stats(runs: dict, pairs: list[int]) -> dict:
    scored = {key: _score(rec) for key, (_, rec) in runs.items() if _scored(rec)}
    ranking = []
    rank = 0
    previous = None
    for i, ((arm, seed), score) in enumerate(sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))):
        if score != previous:
            rank = i + 1
        tied = sum(1 for v in scored.values() if v == score) > 1
        ranking.append({"rank": rank, "arm": arm, "seed": seed, "score": score, "tie": tied})
        previous = score
    unscored = [{"arm": arm, "seed": seed, "status": rec.get("status"),
                 "best_validation_score": rec.get("best_validation_score")}
                for (arm, seed), (_, rec) in sorted(runs.items()) if not _scored(rec)]
    diffs = [{"seed": s, "real": scored[("real", s)], "noise": scored[("noise", s)],
              "real_minus_noise": scored[("real", s)] - scored[("noise", s)]} for s in pairs]
    return {
        "ranking": ranking,
        "not_scored": unscored,
        "arms": {arm: _arm_stats([v for (a, _), v in sorted(scored.items()) if a == arm]) for arm in ARMS},
        "seed_differences": diffs,
        "mean_difference": statistics.fmean(d["real_minus_noise"] for d in diffs) if diffs else None,
    }


# --------------------------------------------------------------------------- whole check
def check(records: list[tuple[int, dict]], tags: tuple[str, str] = DEFAULT_TAGS,
          steps: int = DEFAULT_STEPS, commit: str = DEFAULT_COMMIT,
          placebo_marker: str = DEFAULT_PLACEBO_MARKER) -> dict:
    """Run the selection, pre-checks and rule on parsed log records; returns the summary dict."""
    runs, ignored, not_used, dropped, reruns, warnings = select_runs(records, tags, steps, commit)
    prechecks = run_prechecks(runs, commit, placebo_marker, reruns)
    unique_info, unique_warning = unique_formula_summary(runs)
    if unique_warning:
        warnings.append(unique_warning)
    rule = apply_rule(runs)
    verdict, reason = decide(rule, prechecks)
    failed = [c["name"] for c in prechecks if not c["passed"]]
    selected = []
    for (arm, seed), (no, rec) in sorted(runs.items()):
        selected.append({
            "arm": arm, "seed": seed, "line": no,
            **{k: rec.get(k) for k in ("tag", "run_id", "started_utc", "status", "git_commit", "data_file",
                                       "data_sha256", "symbol", "timeframe", "steps", "batch_size",
                                       "torch_threads", "score_version", "unique_formulas", "formulas_evaluated",
                                       "best_validation_score", "best_formula", "best_formula_decoded")},
        })
    return {
        "label": "research only",
        "tool": TOOL_ID,
        "criteria": {"noise_tag": tags[0], "real_tag": tags[1], "steps": steps, "commit": commit,
                     "expected_seeds": list(EXPECTED_SEEDS), "placebo_marker": placebo_marker, "rule": RULE_TEXT},
        "lines": {"records": len(records), "selected": len(runs), "duplicates_dropped": len(dropped),
                  "ignored": sum(ignored.values()), "ignored_by_reason": dict(sorted(ignored.items()))},
        "selected_runs": selected,
        "arm_files": {arm: sorted({_file_name(r["data_file"]) for r in selected if r["arm"] == arm}) for arm in ARMS},
        "not_used_runs": not_used,
        "dropped_duplicates": dropped,
        "warnings": warnings,
        "prechecks": prechecks,
        "prechecks_passed": not failed,
        "failed_prechecks": failed,
        "unique_formulas": unique_info,
        "rule": rule,
        "stats": score_stats(runs, rule["completed_pairs"]),
        "rule_result": rule["result"],
        "batch_matches_registration": not failed,
        "verdict": verdict,
        "verdict_reason": reason,
        "interpretation_source": INTERPRETATION_SOURCE,
        "interpretation": list(INTERPRETATION),
    }


# --------------------------------------------------------------------------- report
def _wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(_ascii(text), width=100, subsequent_indent=indent, break_long_words=False,
                         break_on_hyphens=False)


def print_report(summary: dict, log_path: Path) -> None:
    c = summary["criteria"]
    lines = summary["lines"]
    print(f"Trial log: {_ascii(log_path)}")
    print(f"Selection: tag {_ascii(c['noise_tag'])!r} = noise (placebo) arm, tag {_ascii(c['real_tag'])!r} = "
          f"real arm, steps == {c['steps']}, seeds 1, 2, 3, git_commit starting with {_ascii(c['commit'])} "
          f"(or missing)")
    print(f"Records read: {lines['records']}. Selected runs: {lines['selected']}. "
          f"Ignored lines: {lines['ignored']}. Older duplicates dropped: {lines['duplicates_dropped']}.")
    for reason, count in lines["ignored_by_reason"].items():
        print(f"  ignored {count} x {_ascii(reason)}")
    for w in summary["warnings"]:
        print(_wrap(f"WARNING: {w}"))

    print("\n== Selected runs ==")
    if not summary["selected_runs"]:
        print("  (none)")
    else:
        print(f"  {'arm':<6}{'seed':>5}  {'status':<12}{'best score':>12}{'unique':>9}  "
              f"{'commit':<18}{'data sha256':<14}started_utc")
        for r in summary["selected_runs"]:
            score = r["best_validation_score"]
            score_txt = _fmt(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else str(score)
            print(f"  {r['arm']:<6}{r['seed']:>5}  {_ascii(r['status'])[:11]:<12}{_ascii(score_txt):>12}"
                  f"{_ascii(_fmt(r['unique_formulas'])):>9}  {_ascii(_short_commit(r['git_commit'])):<18}"
                  f"{_ascii(str(r['data_sha256'])[:12]):<14}{_ascii(r['started_utc'])}")
    if summary["not_used_runs"]:
        print("Lines with a batch tag that were NOT used (not part of the registered batch):")
        for r in summary["not_used_runs"]:
            print(f"  line {r['line']}: {_ascii(r['tag'])} s{_ascii(r['seed'])}, {_ascii(r['reason'])}, status "
                  f"{_ascii(r['status'])}, score {_ascii(repr(r['best_validation_score']))}")

    print("\n== Pre-checks ==")
    for chk in summary["prechecks"]:
        print(_wrap(f"[{'PASS' if chk['passed'] else 'FAIL'}] {chk['name']}: {chk['evidence']}"))
    u = summary["unique_formulas"]
    if u["noise_mean"] is None or u["real_mean"] is None:
        print("unique_formulas arm means (completed runs): not computed (an arm has no completed run)")
    else:
        frac = "n/a" if u["difference_fraction"] is None else f"{100 * u['difference_fraction']:.1f}%"
        print(f"unique_formulas arm means (completed runs): noise {u['noise_mean']:.1f}, real "
              f"{u['real_mean']:.1f}, difference {frac} of the noise mean -> "
              + ("WARNING (above 20%)" if u["warning"] else "ok (20% or less)"))

    rule, stats = summary["rule"], summary["stats"]
    print("\n== Rule (amended 3+3) ==")
    print(_wrap(f"Rule: {RULE_TEXT}"))
    print(f"Completed seed pairs: {rule['completed_pairs']} ({len(rule['completed_pairs'])} of "
          f"{rule['pairs_needed']} needed)")
    if rule["min_real"] is not None:
        print(f"min(real) = {rule['min_real']!r}   max(noise) = {rule['max_noise']!r}")
    print(_wrap(f"Rule result: {rule['result']} - {rule['reason']}"))
    print("Ranking of scores (highest first):")
    for row in stats["ranking"]:
        print(f"  {row['rank']:>2}. {row['arm']:<5} s{row['seed']:<3} {row['score']!r}"
              + ("   (tie)" if row["tie"] else ""))
    for row in stats["not_scored"]:
        print(f"   -  {row['arm']:<5} s{row['seed']:<3} not scored (status {_ascii(row['status'])!r}, "
              f"score {_ascii(repr(row['best_validation_score']))})")
    if not stats["ranking"] and not stats["not_scored"]:
        print("  (no runs)")
    print("Arm statistics over completed runs (sd = sample standard deviation):")
    for arm in ARMS:
        s = stats["arms"][arm]
        print(f"  {arm:<5}: n={s['n']}  min {_fmt(s['min'])}  max {_fmt(s['max'])}  range {_fmt(s['range'])}"
              f"  mean {_fmt(s['mean'])}  sd {_fmt(s['sd'])}")
    print("Seed-matched differences (real - noise):")
    for d in stats["seed_differences"]:
        print(f"  seed {d['seed']}: {_fmt(d['real'])} - {_fmt(d['noise'])} = {d['real_minus_noise']:+.6f}")
    if stats["seed_differences"]:
        print(f"  mean difference {stats['mean_difference']:+.6f} over {len(stats['seed_differences'])} pair(s)")
    else:
        print("  (no completed pairs)")

    print("\n== Best formulas ==")
    if not summary["selected_runs"]:
        print("  (none)")
    for r in summary["selected_runs"]:
        print(f"  {r['arm']} s{r['seed']}: {_ascii(r['best_formula_decoded'])}")
        print(f"  {' ' * len(r['arm'])}  {' ' * len(str(r['seed']))}  tokens {_ascii(r['best_formula'])}")

    print("\n== Verdict ==")
    print(_wrap(f"RULE RESULT: {summary['rule_result']}"))
    failed = summary["failed_prechecks"]
    print(_wrap("BATCH MATCHES REGISTRATION: " + ("YES (all pre-checks passed)" if not failed
                                                   else f"NO (failed pre-checks: {', '.join(failed)})")))
    files = summary["arm_files"]
    print(_wrap(f"Arms: noise (placebo) = {', '.join(files['noise']) or 'no runs'}; "
                f"real = {', '.join(files['real']) or 'no runs'}"))
    print(f"VERDICT: {summary['verdict']}")
    print(_wrap(f"Why: {summary['verdict_reason']}"))
    if c["steps"] != DEFAULT_STEPS:
        print(f"Note: the fixed text below is about the registered {DEFAULT_STEPS}-step design; "
              f"this check used --steps {c['steps']}.")
    print(f"\n== Interpretation (fixed text, word for word from {INTERPRETATION_SOURCE}) ==")
    for line in INTERPRETATION:
        print(_wrap(line))
    print("The exit code is 0 whatever the verdict.")


# --------------------------------------------------------------------------- main
def _parse_tags(values: list[str], ap: argparse.ArgumentParser) -> tuple[str, str]:
    tags = [t.strip() for v in values for t in v.split(",") if t.strip()]
    if len(tags) > 2:
        ap.error(f"--tags took {len(tags)} values ({', '.join(tags)}) but needs exactly two. If one of them is "
                 f"the log path, put the log path before --tags, e.g. "
                 f"check_trials.py logs\\trials.jsonl --tags noise real (or write --tags noise,real)")
    if len(tags) != 2 or tags[0] == tags[1]:
        ap.error("--tags needs two different tags: the placebo (noise) arm first, then the real arm, "
                 "e.g. --tags noise real")
    return tags[0], tags[1]


def _is_previous_summary(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("tool") == TOOL_ID


def _json_target_problem(out: Path, log_path: Path) -> str | None:
    """Why --json must not write to out (None when it may). Nothing is written here."""
    if _is_locked(out):
        return f"refusing to write to {out}: it points into the locked holdout"
    if out.suffix.lower() == ".jsonl":
        return (f"refusing to write the summary to {out}: .jsonl is the trial-log format, and writing there "
                f"could replace the trial log. Use a new .json name, e.g. --json logs\\check_trials.json")
    try:
        same = out.resolve() == log_path.resolve() or (out.exists() and log_path.exists()
                                                       and out.samefile(log_path))
    except OSError:
        same = False
    if same:
        return (f"--json {out} is the trial log itself; writing there would destroy the log. "
                f"Use a new name, e.g. --json logs\\check_trials.json")
    if out.exists():
        if not out.is_file():
            return f"--json {out} is a folder; give a file name, e.g. --json logs\\check_trials.json"
        if not _is_previous_summary(out):
            return (f"{out} already exists and is not an earlier check_trials summary; nothing was written. "
                    f"Choose a new name, e.g. --json logs\\check_trials.json")
    return None


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    print(BANNER)
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Example: python scripts\\research\\check_trials.py logs\\trials.jsonl --json "
               "logs\\check_trials.json. Put the log path before --tags. Exit code: 0 for any verdict, "
               "2 for usage or data errors.")
    ap.add_argument("log", nargs="?", default=str(DEFAULT_LOG),
                    help="trial log written by scripts/run_trial.py (default: logs/trials.jsonl, "
                         "relative to the current folder)")
    ap.add_argument("--tags", nargs="+", default=list(DEFAULT_TAGS), metavar="TAG",
                    help="the placebo-arm tag then the real-arm tag (default: noise real; "
                         "noise,real also works)")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                    help=f"only runs with this many steps (default: {DEFAULT_STEPS})")
    ap.add_argument("--commit", default=DEFAULT_COMMIT,
                    help=f"only runs whose git_commit starts with this (default: {DEFAULT_COMMIT}); "
                         "the pre-check then requires one clean commit (no '+dirty')")
    ap.add_argument("--placebo-marker", default=DEFAULT_PLACEBO_MARKER, metavar="TEXT",
                    help="text that the placebo arm's data file name or symbol contains and the real arm's "
                         f"does not (default: {DEFAULT_PLACEBO_MARKER}, as in PLACEBO_H1.parquet)")
    ap.add_argument("--json", metavar="PATH",
                    help="also write a machine-readable summary to this new .json file (never the trial log; "
                         "an existing file is replaced only if it is an earlier summary)")
    a = ap.parse_args(argv)
    tags = _parse_tags(a.tags, ap)
    commit = a.commit.strip().lower()
    if not _HEX_COMMIT.fullmatch(commit):
        ap.error(f"--commit must be 7 to 40 hex characters, got {a.commit!r}")
    if a.steps < 1:
        ap.error("--steps must be at least 1")
    marker = a.placebo_marker.strip()
    if not marker:
        ap.error("--placebo-marker must not be empty")

    log_path = Path(a.log)
    out = Path(a.json) if a.json else None
    if out is not None and not _is_locked(log_path):
        problem = _json_target_problem(out, log_path)
        if problem:
            print(f"ERROR: {_ascii(problem)}")
            return 2
    try:
        records, read_warnings = read_log(log_path)
    except DataError as e:
        print(f"ERROR: {_ascii(e)}")
        return 2
    summary = check(records, tags, a.steps, commit, marker)
    summary["log_file"] = str(log_path.resolve())
    summary["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary["warnings"] = read_warnings + summary["warnings"]
    print_report(summary, log_path)

    if out is not None:
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(_json_safe(summary), f, indent=2, allow_nan=False)
                f.write("\n")
        except (OSError, ValueError) as e:
            print(f"ERROR: could not write the JSON summary to {_ascii(out)}: {_ascii(e)}")
            return 2
        print(f"JSON summary written to {_ascii(out.resolve())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
