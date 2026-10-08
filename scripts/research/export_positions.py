"""scripts/research/export_positions.py - write the held-position series of ONE AlphaMaster formula on ONE
data file as a propkit POSITIONS CSV, so `python -m propkit evaluate --positions` can test it against
prop-firm rules. Research only: nothing here places or prepares orders.

Usage (from the repo root; Windows PowerShell shown):
    python scripts\\research\\export_positions.py --data-file research\\data\\xauusd\\train\\XAUUSD_H1.parquet --formula "[3,120]"
    python scripts\\research\\export_positions.py --from-trial 20261008_010203_XAUUSD_H1_seed1_real --data-file research\\data\\xauusd\\train\\XAUUSD_H1.parquet
    python -m propkit evaluate --bars research\\data\\xauusd\\train\\XAUUSD_H1.parquet --positions logs\\positions_XAUUSD_H1_3-120.csv --size-mode units --size 10 --rules ftmo-1step --out logs\\propkit_formula

How the positions are made: the formula runs through the engine exactly as scripts/research/score_formula.py
scores it (AlphaEngine._eval_formula_task, empty factor pool), and p[t] = compute_target_positions_stateless
(tanh of the factor, |p| < 0.05 -> flat) is the miner's position for bar t, decided at bar t's close.
The miner earns target_ret[t] = log(open[t+2] / open[t+1]) with it, so it is HELD during bar t+1. The CSV is
already in that held convention (propkit.adapters.positions_from_alphamaster: position[t+1] = p[t],
position[0] = 0); the column p_raw keeps the unshifted p for audit (row t+1 shows p_raw = p[t+1]).

Columns: time (bar open, UTC epoch seconds), position (in [-1, 1], held during that bar), p_raw, time_utc.
position 1.0 = the full size given to propkit (--size oz in units mode, or --size x equity in leverage mode).

Paths containing 'locked_holdout' or ending in '.locked' are refused. --out must end in .csv; by default it
goes under logs\\ (ignored by git); inside the repo it may only be under logs\\, and it may not be the data
file or the trial log. Exit code 0 when the file was written, 2 for usage or data errors.
In-sample warning: a mined formula was chosen on this same data, so any result from it is in-sample.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_SF_PATH = Path(__file__).resolve().parent / "score_formula.py"
_spec = importlib.util.spec_from_file_location("research_score_formula", _SF_PATH)
sf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sf)

HEADER = sf.HEADER
UsageError = sf.UsageError
DEFAULT_TRIALS_LOG = sf.DEFAULT_TRIALS_LOG
LOGS = ROOT / "logs"
IN_SAMPLE_NOTE = ("IN-SAMPLE WARNING: if this formula was mined on this file (e.g. a run's best), every propkit "
                  "result from these positions is in-sample and optimistic; only the locked holdout test is out "
                  "of sample.")


def default_out(data_file: Path, formula: list[int], run_id: str | None) -> Path:
    """logs/positions_<data file stem>_<run id, or the token ids (shortened with a hash when long)>.csv."""
    if run_id:
        tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_id)
    else:
        tag = "-".join(str(t) for t in formula)
        if len(tag) > 40:
            tag = "f" + hashlib.sha1(tag.encode("ascii")).hexdigest()[:10]
    return LOGS / f"positions_{data_file.stem}_{tag}.csv"


def check_out_path(out_text: str, data_file: Path, trials_log: Path) -> Path:
    """Where the CSV may be written: a .csv file that is not an input and not the holdout, and inside the
    repo only under logs\\ (so a mistyped path cannot overwrite a tracked file). Returns the resolved path."""
    if not str(out_text).strip():
        raise UsageError("--out needs a file name ending in .csv")
    sf.check_not_locked(out_text, "--out output path", verb="write")
    path = Path(out_text).expanduser().resolve()
    sf.check_not_locked(str(path), "--out output path", verb="write")
    if path.suffix.lower() != ".csv":
        raise UsageError(f"--out must name a file ending in .csv (got {out_text})")
    if path.is_dir():
        raise UsageError(f"--out names a folder, not a file: {path}")
    for other, what in ((data_file, "the data file"), (trials_log, "the trial log")):
        if sf._same_path(path, other):
            raise UsageError(f"--out would overwrite {what}: {path}")
    if sf._inside(path, ROOT) and not sf._inside(path, LOGS):
        raise UsageError(f"--out {path} is inside the repo but not under logs\\ ; write it under logs\\ "
                         "(ignored by git) or outside the repo, so no tracked file is overwritten")
    return path


def ascii_console() -> None:
    """Make every console line ASCII: AlphaMaster's own log lines (loguru, e.g. the data loader's Chinese
    messages) are written through the same sys.stdout / sys.stderr objects, so re-configuring them in place
    turns any other character into a backslash escape (\\uXXXX) instead of printing it."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="ascii", errors="backslashreplace")
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-file", help="{SYMBOL}_{TF}.parquet file (required unless --from-trial)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--formula", nargs="+", help="token ids as JSON, e.g. \"[3,120]\" (also 3,120 or 3 120)")
    src.add_argument("--from-trial", metavar="RUN_ID", help="take best_formula (and data_file) from this run's "
                                                           "line in the trial log")
    ap.add_argument("--trials-log", default=str(DEFAULT_TRIALS_LOG),
                    help="trial log for --from-trial (default: logs/trials.jsonl in the repo)")
    ap.add_argument("--out", default=None, help="output .csv (default: logs\\positions_<file>_<formula>.csv)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: AlphaMaster's)")
    return ap


def main(argv: list[str] | None = None) -> int:
    ascii_console()
    print(HEADER, flush=True)
    ap = build_parser()
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code in (0, None) else 2
    if not a.from_trial and not a.data_file:
        print("\nERROR: --data-file is required unless --from-trial is used", file=sys.stderr, flush=True)
        return 2
    try:
        return run(a)
    except (UsageError, ValueError) as e:
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        return 2


def run(a: argparse.Namespace) -> int:
    trials_log = Path(a.trials_log).expanduser().resolve()
    trial = None
    if a.from_trial:
        trial = sf.load_trial(trials_log, a.from_trial)
        try:
            formula = [int(t) for t in trial["best_formula"]]
        except (TypeError, ValueError):
            raise UsageError(f"trial {a.from_trial} has an unreadable best_formula: {trial['best_formula']!r}") from None
        data_text = a.data_file or str(trial["data_file"])
    else:
        formula = sf.parse_formula(",".join(a.formula) if len(a.formula) > 1 else a.formula[0])
        data_text = a.data_file
    sf.check_not_locked(data_text, "data file")
    data_file = Path(data_text).expanduser().resolve()
    sf.check_not_locked(str(data_file), "data file")
    if not data_file.is_file():
        hint = " (the trial's path may be from another machine: pass --data-file)" if trial and not a.data_file else ""
        raise UsageError(f"data file not found: {data_file}{hint}")
    out = check_out_path(a.out if a.out is not None else str(default_out(data_file, formula, a.from_trial)),
                         data_file, trials_log)

    os.chdir(ROOT)  # AlphaMaster resolves its config relative to the repo root (as score_formula.py does)
    from data_pipeline.parquet_manager import parse_parquet_filename
    try:
        parse_parquet_filename(data_file)
    except ValueError:
        raise UsageError(f"the file name must look like SYMBOL_TF.parquet (e.g. XAUUSD_H1.parquet): "
                         f"{data_file.name}") from None

    import numpy as np
    import torch
    from model_core.config import ModelConfig
    from model_core.vocab import FORMULA_VOCAB

    import model_core.engine  # noqa: F401  (sets AlphaMaster's default torch threads on import)
    from propkit import adapters
    from propkit import bars as pk_bars

    ModelConfig.REWARD_MODE = "ftmo"  # as score_formula.py and scripts/run_trial.py
    if a.threads and a.threads > 0:
        torch.set_num_threads(a.threads)
    bad = [t for t in formula if not 0 <= t < FORMULA_VOCAB.size]
    if bad:
        raise UsageError(f"token ids {bad} are outside the vocabulary (0..{FORMULA_VOCAB.size - 1})")

    try:
        setup = sf.prepare(data_file)
    except Exception as e:
        raise UsageError(f"could not load {data_file}: {type(e).__name__}: {e}") from None
    rep, res = sf.engine_score(setup, formula)
    if res is None:
        raise UsageError(f"the formula does not execute on this data (engine status {rep['status']!r}); "
                         "check the token list")
    p = sf.positions_of(res)
    times = np.asarray(setup.times, dtype=np.int64)
    if not np.all(np.isfinite(p)):
        raise UsageError(f"the formula gives {int((~np.isfinite(p)).sum())} missing positions on this data")
    positions = adapters.positions_from_alphamaster(times, p, keep_raw=True)

    # the CSV is for propkit: check now that propkit accepts it with this same bar file
    try:
        bars = pk_bars.load_bars(data_file)
    except ValueError as e:
        print(f"  WARNING: propkit cannot read this file as bars ({e}); the positions are written, but "
              "`python -m propkit evaluate` will refuse the same file", flush=True)
    else:
        adapters.validate_positions(positions, bars, source="exported positions")

    out.parent.mkdir(parents=True, exist_ok=True)
    adapters.write_positions_csv(positions, out)
    held = positions["position"].to_numpy()
    print(f"  data file      : {data_file}")
    print(f"  data sha256    : {sf._sha256(data_file)}")
    print(f"  formula        : {formula}" + (f"  (trial {a.from_trial})" if a.from_trial else ""))
    print(f"  decoded        : {setup.engine._decode_formula(formula)}")
    print(f"  engine status  : {rep['status']}, engine score {sf._fmt(rep['val_score'], 6)}")
    print(f"  bars           : {len(held)}, {sf._utc(times[0])} .. {sf._utc(times[-1])} UTC")
    print(f"  held position  : long {np.mean(held > 0):.1%}, short {np.mean(held < 0):.1%}, flat "
          f"{np.mean(held == 0):.1%} of bars; mean |position| {np.mean(np.abs(held)):.3f}; "
          f"{int(np.count_nonzero(np.diff(held)))} changes")
    print("  convention     : position[t+1] = p[t] (held during the bar after the decision); p_raw = unshifted p")
    print(f"  written        : {out}")
    print("  next step      : python -m propkit evaluate --bars <the same data file> --positions "
          f"{out} --size-mode units --size <oz per 1.0> --rules ftmo-1step --out logs\\propkit_run")
    print("  " + IN_SAMPLE_NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
