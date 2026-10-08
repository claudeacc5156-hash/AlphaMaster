"""propkit/cli.py - the command line: python -m propkit <command> [options]. Research only.

Commands
  evaluate  bars + a trade list (--trades CSV) or a held-position series (--positions CSV) -> the account
            path under the prop rules, the bootstrap of challenges, the largest size, statistics and
            stress tests, written to --out DIR as report.json, report.md, equity.csv, trades.csv, days.csv.
  pullback  bars + a pullback spec (JSON) -> the rule's trades (propkit.pullback) -> the same report, plus
            decisions.csv (one row per raw signal, entered or the reason it was skipped).
  rules     print the rule presets.
  selftest  run the known-answer gates (a few seconds) and print PASS/FAIL per gate.
  zeno-v1   zeno_pullback_v1 (propkit.zeno_v1, propkit.zeno_report), in two stages:
            signals  M15 bid/ask bars + the news calendar -> signals.csv, decisions.csv, g0_sample.csv and a
                     counts-only report (no P&L, R or outcome) for the G0 chart check;
            run      after the G0 check (--g0-confirmed): the 24-cell pre-registered grid, the prop evaluator
                     and gates.json (G0-G5, kill) -> report.md, report.json, gates.json, trades.csv,
                     positions.csv, decisions.csv, grid.csv, positions_all_cells.csv (m1_diff.csv with M1).

Exit codes: 0 success; 1 a selftest gate failed; 2 a usage or data error (a bad option, a missing or
invalid file, a locked-holdout path, an output that would overwrite an input or leave --out), and also
any unexpected internal error (one line; PROPKIT_DEBUG=1 prints the traceback).

Units on the command line: --capital and money in USD; --size in oz ("units" mode, position 1.0 = size
oz) or a multiple of equity ("leverage" mode, position 1.0 = a notional of size x equity); --alpha and
--target are fractions (0.05 = 5%); --costs flat<rate> is a fraction of the fill notional per fill.
Nothing here places, sends or prepares orders.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

import propkit
from propkit import adapters
from propkit import bars as bars_mod
from propkit import bootstrap as boot_mod
from propkit import calendar
from propkit import stress as stress_mod
from propkit import report as report_mod
from propkit import rules as rules_mod
from propkit import zeno_report
from propkit import zeno_v1 as zv
from propkit.costs import CostModel
from propkit.equity import equity_from_positions, equity_from_trades
from propkit.pullback import PullbackSpec, generate_trades_detailed
from propkit.rules import PropRules

PROG = "python -m propkit"
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
OUTPUT_FILES = ("report.json", "report.md", "equity.csv", "trades.csv", "days.csv")
PULLBACK_OUTPUT_FILES = OUTPUT_FILES + ("decisions.csv",)


class UsageError(ValueError):
    """A command-line problem the user can fix (bad option, missing file, unsafe output path)."""


class _Parser(argparse.ArgumentParser):
    """argparse that raises UsageError instead of exiting, so main() returns 2 with one clear message."""

    def error(self, message: str):  # noqa: D401 - argparse hook
        command = self.prog[len(PROG):].strip() if self.prog.startswith(PROG) else ""
        raise UsageError(f"{message}. Run 'python -m propkit {command + ' ' if command else ''}--help' for the "
                         "options.")


# ---------------------------------------------------------------------------------------
# console output (ASCII only: zeno's console is cp936)

def _say(text: str = "", stream=None) -> None:
    """Print one line as ASCII (anything else becomes a backslash escape), flushed at once."""
    out = stream or sys.stdout
    out.write(str(text).encode("ascii", "backslashreplace").decode("ascii") + "\n")
    out.flush()


def _err(text: str) -> None:
    _say(f"ERROR: {text}", sys.stderr)


# ---------------------------------------------------------------------------------------
# option parsing helpers

def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}")
    if v <= 0:
        raise argparse.ArgumentTypeError(f"expected a whole number above 0, got {v}")
    return v


def _nonneg_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}")
    if v < 0 or v == 1:
        raise argparse.ArgumentTypeError(f"expected 0 (skip) or a whole number of at least 2, got {v}")
    return v


def _positive_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}")
    if not v > 0 or v == float("inf"):
        raise argparse.ArgumentTypeError(f"expected a finite number above 0, got {text}")
    return v


def _fraction(text: str) -> float:
    v = _positive_float(text)
    if v >= 1:
        raise argparse.ArgumentTypeError(f"expected a fraction between 0 and 1 (0.05 = 5%), got {text}")
    return v


def _horizon(text: str) -> int | None:
    if str(text).strip().lower() in ("none", "no", "unlimited"):
        return None
    return _positive_int(text)


def _tolerance(text: str) -> float | None:
    if str(text).strip().lower() in ("none", "off"):
        return None
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a price tolerance as a fraction (0.01 = 1%) or 'none', "
                                         f"got {text!r}")
    if not v >= 0:
        raise argparse.ArgumentTypeError(f"the price tolerance must be >= 0, got {text}")
    return v


def _read_json_object(path_text: str, what: str) -> dict[str, Any]:
    """A JSON file holding one object; keys starting with '_' are comments and are dropped."""
    bars_mod.check_not_locked(path_text, what=what)
    p = Path(path_text).expanduser()
    if not p.is_file():
        raise UsageError(f"{what} not found: {p}. Check the path (in PowerShell, Tab completes file names).")
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError) as e:
        raise UsageError(f"cannot read {what} {p.name}: {e}")
    except ValueError as e:
        raise UsageError(f"{what} {p.name} is not valid JSON: {e}. Check commas and quotes "
                         "(JSON has no comments; use \"_comment\" keys)")
    if not isinstance(data, dict):
        raise UsageError(f"{what} {p.name} must hold one JSON object: {{\"field\": value, ...}}")
    return {str(k): v for k, v in data.items() if not str(k).startswith("_")}


def build_rules(name: str, capital: float | None = None, rules_file: str | None = None,
                target: float | None = None) -> PropRules:
    """PropRules from the --rules / --capital / --rules-file / --target options.

    name: "ftmo-1step", "ftmo-2step", a firm preset (rules.FIRM_PRESET_NAMES, read from propkit/presets),
    "custom" (needs rules_file) or a path to a .json file. A rules JSON holds PropRules fields (fractions:
    0.03 = 3%); with a "base" key naming a preset it holds only the fields to change. capital (USD) replaces
    initial_capital; target (a fraction) replaces profit_target_pct.
    """
    key = str(name).strip()
    low = key.lower().replace("_", "-")
    if low in rules_mod.ALL_PRESET_NAMES and rules_file is None:
        data: dict[str, Any] = {"base": low}
    elif low == "custom" or rules_file is not None:
        if low not in rules_mod.ALL_PRESET_NAMES and low != "custom":
            raise UsageError("give either --rules FILE.json or --rules custom --rules-file FILE.json, not both")
        if rules_file is None:
            raise UsageError("--rules custom needs --rules-file FILE.json (the PropRules fields; see "
                             "'python -m propkit rules')")
        data = _read_json_object(rules_file, "rules file")
        if low in rules_mod.ALL_PRESET_NAMES:
            file_base = data.get("base")
            if file_base is not None and str(file_base).strip().lower().replace("_", "-") != low:
                raise UsageError(f"--rules says {low} but the rules file {Path(rules_file).name} says base "
                                 f"{file_base!r}; make them the same (or use --rules custom with that file)")
            data["base"] = low
    elif low.endswith(".json"):
        data = _read_json_object(key, "rules file")
    else:
        raise UsageError(f"unknown --rules {name!r}; use {', '.join(rules_mod.ALL_PRESET_NAMES)}, custom (with "
                         "--rules-file) or a .json file")
    overrides = {k: v for k, v in data.items() if k != "base"}
    if capital is not None:
        overrides["initial_capital"] = capital
    if target is not None:
        overrides["profit_target_pct"] = target
    if "base" in data:
        base = str(data["base"])
        c0 = overrides.pop("initial_capital", 100_000.0)     # checked by PropRules (no float() cast here)
        return rules_mod.preset(base, c0, **overrides)
    return rules_mod.custom(**overrides)


def build_costs(spec: str, cost_mult: float = 1.0) -> CostModel:
    """CostModel from --costs and --cost-mult.

    spec: "dukascopy" (the defaults: the bar's own spread, no markup, slippage or commission, placeholder
    swap; see propkit.costs), "flat<rate>" such as flat0.0003 (that fraction of the fill notional on every
    fill replaces spread, markup, slippage and commission; the [ASSUMPTION] placeholder swap STILL
    applies), either of them with the suffix "-noswap" (swap off: "flat0.0003-noswap" is AlphaMaster's
    miner cost exactly, the miner has no swap), or a .json file of CostModel fields. cost_mult scales
    spread, markup, slippage, commission, the flat rate and swap costs (CostModel.multiplied; swap credits
    are not scaled). A trade list's fills already contain the base spread, markup and slippage: `evaluate
    --trades` re-prices them (propkit.stress.reprice_costs) before using the multiplied model.
    """
    text = str(spec).strip()
    low = text.lower()
    no_swap = low.endswith("-noswap")
    if no_swap:
        low = low[: -len("-noswap")].strip()
    if low in ("dukascopy", "default", "bar"):
        cm = CostModel()
    elif low.startswith("flat"):
        rate_text = low[4:].strip()
        try:
            rate = float(rate_text)
        except ValueError:
            raise UsageError(f"--costs {spec!r}: write flat followed by a fraction, such as flat0.0003 (= 0.03% "
                             "of the notional per fill), optionally with -noswap")
        cm = CostModel(flat_rate_per_side=rate)
    elif low.endswith(".json") and not no_swap:
        cm = CostModel.from_dict(_read_json_object(text, "costs file"))
    else:
        raise UsageError(f"unknown --costs {spec!r}; use dukascopy, flat0.0003 (any fraction), either with "
                         "-noswap, or a .json file of CostModel fields")
    if no_swap:
        cm = dataclasses.replace(cm, swap_enabled=False)
    if cost_mult != 1.0:
        cm = cm.multiplied(cost_mult)
    return cm


# ---------------------------------------------------------------------------------------
# output folder safety

def _resolved(path_text) -> Path:
    return Path(str(path_text)).expanduser().resolve()


def prepare_out_dir(out: str, inputs: Sequence[str | None], names: Sequence[str]) -> Path:
    """Create (if needed) and return the output folder; refuse unsafe outputs.

    Refused: a locked-holdout path; an existing file in place of the folder; any output file that is one
    of the input files (it would be overwritten). Outputs are written only as the fixed names inside it.
    """
    bars_mod.check_not_locked(out, what="output folder", verb="write")
    d = Path(str(out)).expanduser()
    if d.exists() and not d.is_dir():
        raise UsageError(f"--out {out} is a file, not a folder; give a folder name such as logs\\run1")
    targets = {_resolved(d / n) for n in names}
    for src in inputs:
        if src is not None and _resolved(src) in targets:
            raise UsageError(f"the input file {src} is inside --out and would be overwritten; choose another "
                             "--out folder")
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise UsageError(f"cannot create the output folder {d}: {e}")
    return d


def _inside(folder: Path, file: Path) -> Path:
    """The file path, after checking that it resolves inside folder (no writing outside --out)."""
    f, root = file.resolve(), folder.resolve()
    if f.parent != root:
        raise UsageError(f"refusing to write {file}: it is not inside the output folder {folder}")
    bars_mod.check_not_locked(f, what="output file", verb="write")
    return f


TMP_PREFIX = "_partial_"
BACKUP_PREFIX = "_previous_"
DECISIONS_HEADER = "signal_time,signal_bar,side,entry_time,status,trade_id"


def write_outputs(out_dir: Path, report: dict[str, Any], equity, trades, days=None,
                  decisions=None) -> list[Path]:
    """Write report.json, report.md, equity.csv, trades.csv, days.csv (and decisions.csv) inside out_dir.

    All-or-nothing (write_staged): every file is first written as _partial_<name>, every earlier output
    must be replaceable, and the set is swapped in only then; a failed swap is rolled back, so an error (a
    full disk, a text column that cannot be written, a file held open by another program) cannot leave a
    half-written mix of this run and an earlier one. Text is ASCII; any other character in a text column
    becomes a backslash escape (\\uXXXX).
    """
    boundary = (report.get("rules") or {}).get("day_boundary") or calendar.DEFAULT_DAY_BOUNDARY
    jobs = [("report.json", lambda p: p.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False)
                                                   + "\n", encoding="ascii")),
            ("report.md", lambda p: p.write_text(report_mod.render_markdown(report), encoding="ascii")),
            ("equity.csv", lambda p: adapters.write_equity_csv(equity, p, day_boundary=boundary)),
            ("trades.csv", lambda p: adapters.write_trades_csv(trades, p))]
    if days is not None:
        jobs.append(("days.csv", lambda p: days.to_csv(p, index=False, encoding="ascii", errors="backslashreplace",
                                                       lineterminator="\n")))
    if decisions is not None:
        def write_decisions(p: Path) -> None:
            dec = decisions.copy()
            if len(dec):
                dec["signal_time_utc"] = calendar.utc_str(dec["signal_time"].to_numpy(dtype="int64"))
                et = dec["entry_time"].to_numpy(dtype="int64")
                dec["entry_time_utc"] = [calendar.utc_str(int(t)) if t >= 0 else "" for t in et]
            dec.to_csv(p, index=False, encoding="ascii", errors="backslashreplace", lineterminator="\n")
        jobs.append(("decisions.csv", write_decisions))
    return write_staged(out_dir, jobs)


def _check_replaceable(final: Path) -> None:
    """Raise OSError when an earlier output at `final` cannot be replaced: a folder (or anything but a regular
    file) under that name, or a file another program holds open (on Windows, opening it for writing fails
    with a sharing violation, e.g. a CSV open in Excel). Nothing is changed by the check."""
    if not os.path.lexists(final):
        return
    if final.is_symlink() or not final.is_file():
        raise OSError(f"cannot replace {final}: it is a folder or not a regular file; move it out of the output "
                      "folder and run again (nothing was changed)")
    try:
        with open(final, "ab"):
            pass
    except OSError as e:
        raise OSError(f"cannot replace {final} ({type(e).__name__}: {e}); close it in every program that has it "
                      "open (Excel, an editor) and run again (nothing was changed)") from e


def write_staged(out_dir: Path, jobs: Sequence[tuple[str, Any]]) -> list[Path]:
    """Write each (name, writer(path)) job inside out_dir, all or nothing (see write_outputs).

    1. every file is written as _partial_<name>; 2. every earlier output under a final name must be
    replaceable (_check_replaceable: not a folder, not held open by another program); 3. the set is swapped
    in: an earlier output is moved to _previous_<name>, the partial file to <name>, and the backups are
    deleted once all names are swapped. On an error in 1 or 2 the partial files are removed and nothing of the
    earlier run is touched; on an error in 3 the swap is rolled back (new files removed, backups moved back)
    and the OSError names the file that failed (and any file that could not be restored, with the name of
    its backup). A name that would leave out_dir is refused."""
    staged: list[tuple[Path, Path]] = []
    try:
        for name, write in jobs:
            final = _inside(out_dir, out_dir / name)
            tmp = _inside(out_dir, out_dir / f"{TMP_PREFIX}{name}")
            write(tmp)
            staged.append((tmp, final))
        for _, final in staged:
            _check_replaceable(final)
    except BaseException:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)
        for name, _ in jobs:
            (out_dir / f"{TMP_PREFIX}{name}").unlink(missing_ok=True)
        raise
    backups: list[tuple[Path, Path]] = []
    done: list[Path] = []
    current = None
    try:
        for tmp, final in staged:
            current = final
            if final.exists():
                bak = _inside(out_dir, out_dir / f"{BACKUP_PREFIX}{final.name}")
                os.replace(final, bak)
                backups.append((bak, final))
            os.replace(tmp, final)
            done.append(final)
    except BaseException as e:
        not_restored = _roll_back(staged, backups, done)
        if not isinstance(e, Exception):
            raise
        msg = (f"could not replace {current} ({type(e).__name__}: {e}); every output was put back as it was "
               "before this run" if not not_restored else
               f"could not replace {current} ({type(e).__name__}: {e}); the rollback also failed for "
               + ", ".join(not_restored) + ": rename each backup to its name by hand")
        raise OSError(msg + ". Close the file in every program that has it open (Excel, an editor) and run "
                      "again.") from e
    for bak, _ in backups:
        bak.unlink(missing_ok=True)
    return [final for _, final in staged]


def _roll_back(staged: Sequence[tuple[Path, Path]], backups: Sequence[tuple[Path, Path]],
               done: Sequence[Path]) -> list[str]:
    """Undo a partial swap of write_staged: remove the new files that had no earlier version, move every
    backup back to its name and remove the partial files left. Returns '<name> (backup <backup name>)' for
    each earlier output that could not be put back."""
    had_backup = {final for _, final in backups}
    for final in done:
        if final not in had_backup:
            try:
                final.unlink(missing_ok=True)
            except OSError:
                pass
    failed = []
    for bak, final in backups:
        try:
            os.replace(bak, final)
        except OSError:
            failed.append(f"{final.name} (backup {bak.name})")
    for tmp, _ in staged:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return failed


def stale_decisions(out_dir: Path) -> tuple[Path | None, bool]:
    """(decisions.csv in out_dir or None, whether its header is propkit's own decision log)."""
    p = out_dir / "decisions.csv"
    if not p.is_file():
        return None, False
    try:
        with open(p, "r", encoding="ascii", errors="replace") as f:
            head = f.readline().strip()
    except OSError:
        return p, False
    return p, head.startswith(DECISIONS_HEADER)


# ---------------------------------------------------------------------------------------
# commands

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bars", required=True, help="bar file: AlphaMaster Parquet or CSV (time, open, high, low, "
                                                  "close, tick_volume[, spread]); BID prices, USD/oz")
    p.add_argument("--rules", required=True, help="ftmo-1step, ftmo-2step, custom (with --rules-file) or FILE.json")
    p.add_argument("--rules-file", default=None, help="JSON of PropRules fields (with --rules custom, or to "
                                                     "change fields of a preset)")
    p.add_argument("--capital", type=_positive_float, default=None, help="initial capital C0, USD (default "
                                                                         "100000 for the presets)")
    p.add_argument("--target", type=_fraction, default=None, help="profit target as a fraction (0.05 = 5%%, "
                                                                  "FTMO 2-Step phase 2)")
    p.add_argument("--costs", default="dukascopy", help="dukascopy (default), flat0.0003 or FILE.json; add "
                                                        "-noswap to switch the placeholder swap off "
                                                        "(flat0.0003-noswap = AlphaMaster's miner cost)")
    p.add_argument("--cost-mult", type=_positive_float, default=1.0,
                   help="multiply every cost (default 1); with --trades the fills are re-priced, k >= 1 only")
    p.add_argument("--spread-scale", type=_positive_float, default=None,
                   help="multiply the bar file's spread column to get USD/oz, e.g. 0.01 for MT5 points of a "
                        "2-digit quote (default: as given, and a median above 2 USD/oz is refused)")
    p.add_argument("--n-sims", type=_positive_int, default=10_000, help="bootstrap simulations (default 10000)")
    p.add_argument("--seed", type=int, default=7, help="random seed (default 7)")
    p.add_argument("--alpha", type=_fraction, default=0.05, help="breach budget for the largest size (default 0.05)")
    p.add_argument("--horizon-days", type=_horizon, default=60,
                   help="challenge length in days, or none (default 60; see --horizon-unit)")
    p.add_argument("--horizon-unit", choices=boot_mod.HORIZON_UNITS, default=boot_mod.DEFAULT_HORIZON_UNIT,
                   help="what --horizon-days counts: trading (days with a trade entry, default) or market "
                        "(every prop day with bars)")
    p.add_argument("--history-reps", type=_nonneg_int, default=boot_mod.DEFAULT_HISTORY_REPS,
                   help="outer bootstrap replicates for the uncertainty from the limited history "
                        f"(default {boot_mod.DEFAULT_HISTORY_REPS}; 0 = skip)")
    p.add_argument("--n-trials", type=_positive_int, default=None,
                   help="how many variants were tried in all, this one included (for the deflated Sharpe ratio)")
    p.add_argument("--sr-var", type=_positive_float, default=None,
                   help="variance of the tried variants' daily SR estimates (for the DSR; default 1/(n-1))")
    p.add_argument("--dd-sims", type=_positive_int, default=report_mod.DEFAULT_DD_SIMS,
                   help="trade orders in the drawdown stress test (default 2000)")
    p.add_argument("--no-stress", action="store_true", help="skip the stress tests (faster)")
    p.add_argument("--out", required=True, help="output folder (created if missing), such as logs\\run1")


def make_parser() -> argparse.ArgumentParser:
    """The argparse parser of `python -m propkit`."""
    top = _Parser(prog=PROG, description="propkit: what does this trade list or position series "
                  "do to a prop-firm account? RESEARCH ONLY - not trading advice.")
    top.add_argument("--version", action="version", version=f"propkit {propkit.__version__}")
    sub = top.add_subparsers(dest="command", metavar="{evaluate,pullback,rules,selftest,zeno-v1}",
                             parser_class=_Parser)

    ev = sub.add_parser("evaluate", help="evaluate a trade list or a position series")
    src = ev.add_mutually_exclusive_group(required=True)
    src.add_argument("--trades", default=None, help="TRADES CSV (fills already include spread and markup)")
    src.add_argument("--positions", default=None, help="POSITIONS CSV (time, position held during the bar)")
    ev.add_argument("--size-mode", choices=("units", "leverage"), default="units",
                    help="with --positions: units (1.0 = --size oz) or leverage (1.0 = --size x equity)")
    ev.add_argument("--size", type=_positive_float, default=None,
                    help="with --positions: oz per 1.0 of position, or the leverage multiple")
    ev.add_argument("--price-tolerance", type=_tolerance, default=0.01,
                    help="with --trades: how far a fill may sit outside its bar's bid low .. ask high range, "
                         "as a fraction of the price (0.01 = 1%%), or none to skip the check (default 0.01)")
    _add_common(ev)

    pb = sub.add_parser("pullback", help="generate the pullback rule's trades from a spec, then evaluate them")
    pb.add_argument("--spec", required=True, help="pullback spec JSON (see propkit/examples/pullback_spec_example.json)")
    pb.add_argument("--lot-step", type=_positive_float, default=1.0,
                    help="size step in oz; sizes are rounded down to it (default 1 oz = 0.01 lot)")
    _add_common(pb)

    sub.add_parser("rules", help="print the rule presets")
    sub.add_parser("selftest", help="run the known-answer gates and print PASS/FAIL per gate")
    _add_zeno(sub)
    return top


def _zeno_inputs(p: argparse.ArgumentParser) -> None:
    p.add_argument("--m15-bid", required=True, help="XAUUSD M15 BID bars: Parquet or CSV (time, open, high, low, "
                                                     "close) or a dukascopy-node CSV (timestamp in ms); UTC; every "
                                                     "bar must open before 2025-09-28 00:00 UTC (the holdout lock)")
    p.add_argument("--m15-ask", required=True, help="the matching M15 ASK bars (the same bars as --m15-bid)")
    p.add_argument("--news", required=True, help="US macro calendar CSV (event, ..., datetime_utc, kind); its NFP, "
                                                 "CPI, PPI and FOMC rows are the news blackout (D20)")
    p.add_argument("--out", required=True, help="output folder (created if missing), such as logs\\zeno_g0")


def _add_zeno(sub) -> None:
    """The zeno-v1 command with its two stages, signals and run."""
    zs = sub.add_parser("zeno-v1", help="zeno_pullback_v1: signals for the G0 check, then the pre-registered run")
    zsub = zs.add_subparsers(dest="zeno_command", metavar="{signals,run}", parser_class=_Parser)
    sg = zsub.add_parser("signals", help="stage 1: signals, decisions and the G0 sample (no P&L, R or outcome)")
    _zeno_inputs(sg)
    sg.add_argument("--sample", type=_positive_int, default=zeno_report.G0_SAMPLE_SIZE,
                    help="eligible signals to sample for the chart check (default 20)")
    sg.add_argument("--seed", type=int, default=zeno_report.DEFAULT_SEED, help="sample seed (default 7)")
    sg.add_argument("--variant", choices=zv.VARIANTS, default=zeno_report.STAGE1_CELL.variant,
                    help="the declared cell's variant (default evaluation)")
    sg.add_argument("--commission", type=float, choices=zv.COMMISSIONS,
                    default=zeno_report.STAGE1_CELL.commission_rt_per_lot,
                    help="the declared cell's commission, USD per lot round trip (default 10)")
    sg.add_argument("--spread-base", choices=zv.SPREAD_BASES, default=zeno_report.STAGE1_CELL.spread_base,
                    help="the declared cell's spread base (default S1, the data's spread)")
    sg.add_argument("--cost-mult", type=float, choices=zv.COST_MULTS, default=zeno_report.STAGE1_CELL.cost_mult,
                    help="the declared cell's cost multiplier (default 1.5)")
    sg.add_argument("--capital", type=_positive_float, default=100_000.0, help="account size, USD (default 100000)")
    rn = zsub.add_parser("run", help="stage 2: the 24-cell grid, the prop evaluator and the gates (only after the "
                                     "G0 check: --g0-confirmed)")
    _zeno_inputs(rn)
    rn.add_argument("--g0-confirmed", action="store_true",
                    help="you checked the signals of g0_sample.csv on a chart and agree with at least 18 of 20")
    rn.add_argument("--g0-sample", default=None, help="the g0_sample.csv you checked (its sha256 and your y/n "
                                                      "answers in agree_y_n are recorded in gates.json)")
    rn.add_argument("--rules", default="fundingpips-1step-flex",
                    help="firm rules: fundingpips-1step-flex (default; the placeholder until the verified sheet is "
                         "installed), another preset, or a rules .json file")
    rn.add_argument("--reference-rules", default=None, help="also run the judging cell under these rules for "
                                                            "comparison (e.g. ftmo-1step); off by default")
    rn.add_argument("--capital", type=_positive_float, default=None,
                    help="account size, USD (default: the rules' initial capital, 100000)")
    rn.add_argument("--m1-bid", default=None, help="M1 BID bars for the D15 second run (with --m1-ask)")
    rn.add_argument("--m1-ask", default=None, help="M1 ASK bars for the D15 second run (with --m1-bid)")
    rn.add_argument("--n-sims", type=_positive_int, default=boot_mod.DEFAULT_N_SIMS,
                    help="bootstrap simulations (default 10000)")
    rn.add_argument("--seed", type=int, default=zeno_report.DEFAULT_SEED, help="random seed (default 7)")
    rn.add_argument("--history-reps", type=_nonneg_int, default=boot_mod.DEFAULT_HISTORY_REPS,
                    help=f"outer bootstrap replicates for the history uncertainty (default "
                         f"{boot_mod.DEFAULT_HISTORY_REPS}; 0 = skip)")


def _rules_path(rules_arg: str) -> str | None:
    """The --rules value when it names a .json file (an input that must not be overwritten), else None."""
    text = str(rules_arg).strip()
    return text if text.lower().endswith(".json") else None


def _load_bars(args):
    """Load --bars (with --spread-scale) and print what was read, including the spread actually found."""
    _say(f"Loading bars from {args.bars} ...")
    bars = bars_mod.load_bars(args.bars, spread_scale=args.spread_scale)
    if "spread" in bars.columns:
        sp = bars["spread"].to_numpy()
        _say(f"  {len(bars)} bars; spread column median {float(pd.Series(sp).median()):.3f} USD/oz, p90 "
             f"{float(pd.Series(sp).quantile(0.9)):.3f} (XAUUSD is usually 0.1 .. 0.7; if not, see --spread-scale)")
    else:
        _say(f"  {len(bars)} bars; no spread column (the cost model's fixed spread is used)")
    return bars


def _input_entry(kind: str, path_text: str, **extra) -> dict[str, Any]:
    return {"kind": kind, "file": str(path_text), "sha256": report_mod.file_sha256(path_text), **extra}


def _run_analysis(args, bars, trades, equity, rules, cm, out_dir, input_info, decisions=None) -> int:
    t0 = time.perf_counter()
    if len(trades) == 0:
        _say("WARNING: the input gives no trades; every number below describes an account that never trades.")
    input_info = {**input_info, "cost_mult": float(args.cost_mult)}
    report = report_mod.analyse(bars, trades, equity, rules, cm, bars_path=args.bars, input_info=input_info,
                                n_sims=args.n_sims, seed=args.seed, alpha=args.alpha,
                                horizon_days=args.horizon_days, horizon_unit=args.horizon_unit,
                                n_trials=args.n_trials, sr_var=args.sr_var,
                                run_stress=not args.no_stress, dd_sims=args.dd_sims,
                                history_reps=args.history_reps, progress=lambda s: _say(f"  {s}"))
    days = pd.DataFrame(report["path"].get("days") or [])
    stale, ours = stale_decisions(out_dir) if decisions is None else (None, False)
    written = write_outputs(out_dir, report, equity, trades, days=days, decisions=decisions)
    _say(f"  analysis took {time.perf_counter() - t0:.1f} s")
    if stale is not None:
        if ours:
            stale.unlink(missing_ok=True)
            _say(f"NOTE: removed {stale.name} left in {out_dir} by an earlier pullback run (it described that run).")
        else:
            _say(f"WARNING: {stale} is not from this run; it does not describe these results.")
    _say("")
    _summary(report)
    _say("")
    _say("Files written:")
    for p in written:
        _say(f"  {p}")
    return EXIT_OK


def _summary(r: dict[str, Any]) -> None:
    d, path, ms = r["bootstrap_days"], r["path"], r["max_size"]
    rs = (r["stats"] or {}).get("r_summary") or {}
    n = r["settings"]["n_sims"]

    def pct(p, se):
        if p is None:
            return "n/a"
        if se == 0 and p in (0, 0.0, 1, 1.0):
            return f"{100 * p:.1f}% ({'<' if p == 0 else '>'} {100 * (3 / n if p == 0 else 1 - 3 / n):.2f}%)"
        return f"{100 * p:.1f}% (+- {100 * (se or 0):.1f})"

    _say(report_mod.HEADER)
    _say(f"Rules: {r['rules']['name']}, capital {r['rules']['initial_capital']:,.0f} USD")
    _say(f"Historical path: {path['status']}, final balance {path['final_balance']:,.2f} USD, "
         f"max drawdown {path['max_dd_pct']:.2f}% of capital")
    if rs:
        se_r = rs.get("se_r")
        _say(f"Trades: {rs.get('n')}, mean R {rs.get('mean_r', float('nan')):.3f} +- "
             + ("n/a (one trade)" if se_r is None else f"{se_r:.3f}"))
    hz = boot_mod.horizon_text(r["settings"]["horizon_days"], r["settings"].get("horizon_unit", "trading"))
    _say(f"Bootstrap ({n} challenges of {hz}, flat-to-flat day blocks): P(pass) {pct(d['p_pass'], d['se_pass'])}, "
         f"P(daily breach) {pct(d['p_breach_daily'], d['se_breach_daily'])}, "
         f"P(max breach) {pct(d['p_breach_max'], d['se_breach_max'])}")
    _say("  (+- is the Monte Carlo error: simulation noise only, not the uncertainty from the limited history)")
    hu = r.get("history_uncertainty") or {}
    if hu.get("p_pass") and hu.get("p_breach_daily"):
        _say(f"  history uncertainty (outer bootstrap, 5-95%): P(pass) {100 * hu['p_pass']['p5']:.1f}% .. "
             f"{100 * hu['p_pass']['p95']:.1f}%, P(daily breach) {100 * hu['p_breach_daily']['p5']:.1f}% .. "
             f"{100 * hu['p_breach_daily']['p95']:.1f}%")
    for w in r.get("warnings") or []:
        _say(f"WARNING: {w}")
    mult = ms.get("multiplier")
    mrange = (hu.get("max_size_multiplier") or {}) if hu else {}
    extra = f"; 5-95% over histories {mrange['p5']:.2f} .. {mrange['p95']:.2f}" if mrange else ""
    _say(f"Largest size with P(daily breach) <= {r['settings']['alpha']:g}: "
         f"{'n/a' if mult is None else f'{mult:.3f}'} x this run's size ({ms.get('note')}{extra})")


def cmd_evaluate(args) -> int:
    """`evaluate`: bars + trades or positions -> report files in --out."""
    if args.positions is not None and args.size is None:
        raise UsageError("--positions needs --size (oz per 1.0 of position with --size-mode units, or the "
                         "leverage multiple with --size-mode leverage)")
    if args.trades is not None and args.size is not None:
        raise UsageError("--size and --size-mode are for --positions; a trade list already has its sizes")
    src = args.trades if args.trades is not None else args.positions
    for path_text, what in ((args.bars, "bar file"), (src, "trades file" if args.trades else "positions file")):
        bars_mod.check_not_locked(path_text, what=what)
    if args.trades is not None and args.cost_mult < 1.0:
        raise UsageError("--cost-mult below 1 cannot be used with --trades: the fills in the trade list already "
                         "contain the full spread, markup and slippage, and propkit does not take costs out of "
                         "them. Use --cost-mult 1 or more (or --positions / pullback, which build their own fills).")
    rules = build_rules(args.rules, args.capital, args.rules_file, args.target)
    base_cm = build_costs(args.costs, 1.0)
    cm = base_cm.multiplied(args.cost_mult) if args.cost_mult != 1.0 else base_cm
    out_dir = prepare_out_dir(args.out, [args.bars, src, args.rules_file, args.costs, _rules_path(args.rules)],
                              OUTPUT_FILES)
    _say(report_mod.HEADER)
    bars = _load_bars(args)
    c0 = rules.initial_capital
    if args.trades is not None:
        _say(f"Loading trades from {args.trades} ...")
        raw = adapters.read_trades_csv(args.trades, allow_extra=True)
        extra = {}
        if args.cost_mult != 1.0:
            # the fills carry the base spread, markup and slippage: move them by the extra (k - 1) x those
            # costs, exactly what the multiplied model would have filled at (propkit.stress.reprice_costs)
            raw = stress_mod.reprice_costs(bars, raw, base_cm, args.cost_mult)
            extra["fills_repriced_for_cost_mult"] = float(args.cost_mult)
            _say(f"  fills re-priced for --cost-mult {args.cost_mult:g} (spread, markup and slippage x "
                 f"{args.cost_mult:g}; commission and swap from the multiplied cost model)")
        equity, trades = equity_from_trades(bars, raw, c0, cm, price_tolerance=args.price_tolerance)
        info = _input_entry("trades", args.trades, n_rows=int(len(raw)), **extra)
    else:
        _say(f"Loading positions from {args.positions} ...")
        pos = adapters.read_positions_csv(args.positions)
        equity, trades = equity_from_positions(bars, pos, c0, cm, size_mode=args.size_mode, size=args.size)
        info = _input_entry("positions", args.positions, n_rows=int(len(pos)), size_mode=args.size_mode,
                            size=float(args.size), trades_made=int(len(trades)))
    _say(f"  {len(trades)} trades; running the analysis ...")
    return _run_analysis(args, bars, trades, equity, rules, cm, out_dir, info)


def cmd_pullback(args) -> int:
    """`pullback`: bars + spec -> the rule's trades -> report files (plus decisions.csv) in --out."""
    for path_text, what in ((args.bars, "bar file"), (args.spec, "spec file")):
        bars_mod.check_not_locked(path_text, what=what)
    spec = PullbackSpec.from_json(Path(args.spec))
    rules = build_rules(args.rules, args.capital, args.rules_file, args.target)
    cm = build_costs(args.costs, args.cost_mult)
    out_dir = prepare_out_dir(args.out, [args.bars, args.spec, args.rules_file, args.costs, _rules_path(args.rules)],
                              PULLBACK_OUTPUT_FILES)
    _say(report_mod.HEADER)
    if spec.placeholder:
        _say("WARNING: this spec is a PLACEHOLDER (\"placeholder\": true), not zeno's rule. The results say "
             "nothing about the real rule.")
    bars = _load_bars(args)
    _say("  generating trades ...")
    c0 = rules.initial_capital
    raw, decisions = generate_trades_detailed(bars, spec, c0, cm, lot_step_oz=args.lot_step)
    equity, trades = equity_from_trades(bars, raw, c0, cm, price_tolerance=None)
    info = _input_entry("pullback", args.spec, signals=int(len(decisions)), trades_made=int(len(trades)),
                        lot_step_oz=float(args.lot_step), placeholder=bool(spec.placeholder))
    info["spec"] = spec.to_dict()
    _say(f"  {len(decisions)} signals, {len(trades)} trades; running the analysis ...")
    return _run_analysis(args, bars, trades, equity, rules, cm, out_dir, info, decisions=decisions)


def cmd_rules(args) -> int:
    """`rules`: print each preset in plain English."""
    _say(report_mod.HEADER)
    _say(f"Rule presets (FTMO as of {rules_mod.FTMO_AS_OF}):")
    for name in rules_mod.PRESET_NAMES:
        _say("")
        _say(f"--rules {name}")
        for line in rules_mod.preset(name).describe():
            _say(f"    {line}")
    for name in rules_mod.FIRM_PRESET_NAMES:
        r, info = rules_mod.rules_and_info(name)
        _say("")
        _say(f"--rules {name}   [{info.get('status')}]")
        if info.get("fallback"):
            _say(f"    NOTE: {info['fallback']}")
        for line in r.describe():
            _say(f"    {line}")
        if info.get("unverified_fields"):
            _say(f"    unverified [U] fields: {', '.join(info['unverified_fields'])}")
    _say("")
    _say("--rules custom --rules-file FILE.json takes these fields (fractions: 0.03 = 3%):")
    _say("    " + ", ".join(PropRules.__dataclass_fields__))
    _say("A rules JSON with \"base\": \"ftmo-2step\" changes only the fields it lists, e.g.")
    _say("    {\"base\": \"ftmo-2step\", \"profit_target_pct\": 0.05}")
    return EXIT_OK


def cmd_selftest(args) -> int:
    """`selftest`: run every known-answer gate; exit 0 if all pass, 1 otherwise."""
    from propkit.selftest import run_selftest
    _, n_fail = run_selftest(out=_say)
    return EXIT_OK if n_fail == 0 else EXIT_FAILED


# ---------------------------------------------------------------------------------------
# zeno-v1

def _json_job(obj: Any):
    return lambda p: p.write_text(json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
                                  encoding="ascii")


def _bytes_job(data: bytes):
    return lambda p: p.write_bytes(data)


def _zeno_data(args):
    """Load the M15 bid/ask pair (the lock is enforced there) and the news calendar; prepare once."""
    _say(f"Loading M15 bid/ask bars from {args.m15_bid} and {args.m15_ask} ...")
    frame = zv.load_m15_bidask(args.m15_bid, args.m15_ask)
    s = frame.attrs["zeno_v1"]
    _say(f"  {s['n_bars']} bars, {s['first_time_utc']} .. {s['last_time_utc']}; spread at the open median "
         f"{s['spread_open_median']:.3f} USD/oz, p90 {s['spread_open_p90']:.3f} (XAUUSD is usually 0.1 .. 0.7)")
    if s.get("range_note"):
        _say(f"  {'WARNING' if s.get('starts_after_range_start') else 'NOTE'}: {s['range_note']}.")
    _say(f"Loading the news calendar from {args.news} ...")
    news = zv.read_news_csv(args.news)
    _say(f"  {news.times.size} NFP/CPI/PPI/FOMC events")
    _say("  preparing (1h trend, ATR14, server days, filters, setup state machines) ...")
    return zv.prepare(frame, news)


def _zeno_input_info(args, extra: Sequence[tuple[str, str | None]] = ()) -> dict[str, Any]:
    out = {}
    for kind, path_text in (("m15_bid", args.m15_bid), ("m15_ask", args.m15_ask), ("news", args.news)) + tuple(extra):
        if path_text is not None:
            out[kind] = _input_entry(kind, path_text)
    return out


def cmd_zeno_signals(args) -> int:
    """`zeno-v1 signals`: stage 1 - signals, decisions and the G0 sample in --out (no P&L, R or outcome)."""
    for path_text, what in ((args.m15_bid, "M15 bid file"), (args.m15_ask, "M15 ask file"),
                            (args.news, "news calendar")):
        bars_mod.check_not_locked(path_text, what=what)
    cell = zv.ZenoCell(args.variant, args.commission, args.spread_base, args.cost_mult)
    out_dir = prepare_out_dir(args.out, [args.m15_bid, args.m15_ask, args.news], zeno_report.SIGNALS_FILES)
    answered = _g0_answers_in(out_dir / "g0_sample.csv")
    if answered:
        raise UsageError(f"refusing to replace {out_dir / 'g0_sample.csv'}: it holds {answered} answer(s) in agree_y_n "
                         "(your G0 chart check), and a new run of `zeno-v1 signals` would draw a fresh, unanswered "
                         "sample over it. Choose another --out folder, or move that file and the signals_report.json "
                         "beside it out of this folder first. Nothing was read or written.")
    _say(report_mod.HEADER)
    prep = _zeno_data(args)
    _say(f"  screening the triggers in the declared cell {cell.label} ...")
    report, tables = zeno_report.signals_stage(prep, cell, args.capital, args.sample, args.seed,
                                               inputs=_zeno_input_info(args))
    g0_bytes = zeno_report.csv_bytes(tables["g0_sample"])
    report["g0"]["sample_sha256"] = hashlib.sha256(g0_bytes).hexdigest()
    written = write_staged(out_dir, [
        ("signals.csv", _bytes_job(zeno_report.csv_bytes(tables["signals"]))),
        ("decisions.csv", _bytes_job(zeno_report.csv_bytes(tables["decisions"]))),
        ("g0_sample.csv", _bytes_job(g0_bytes)),
        ("signals_report.json", _json_job(report)),
        ("signals_report.md", lambda p: p.write_text(zeno_report.render_signals_markdown(report), encoding="ascii"))])
    _say("")
    _say(f"Triggers: {report['n_triggers']}, eligible {report['n_eligible']} (cell {cell.label}); G0 sample: "
         f"{report['g0']['sample_size']} eligible signals (seed {args.seed}).")
    _say("No P&L, R multiple or outcome was computed for display; nothing in these files is a result.")
    _say("")
    _say("NEXT: " + zeno_report.G0_INSTRUCTIONS)
    _say("")
    _say("Files written:")
    for p in written:
        _say(f"  {p}")
    return EXIT_OK


def _g0_answers_in(path: Path) -> int:
    """The number of non-empty agree_y_n cells of an existing g0_sample.csv (0 when there is no such file, it is
    empty or it has no agree_y_n column). A file that cannot be read is refused: it may hold zeno's answers."""
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, ValueError) as e:
        raise UsageError(f"{path} exists but cannot be read ({e}); it may hold your G0 answers, so it is not "
                         "replaced. Choose another --out folder, or move that file first.")
    cols = {str(c).strip().lower(): c for c in df.columns}
    if "agree_y_n" not in cols:
        return 0
    return int((df[cols["agree_y_n"]].astype(str).str.strip() != "").sum())


def _g0_declared(sample: Path, capital: float, prep: zv.Prepared) -> tuple[zv.ZenoCell, float, dict[str, Any]]:
    """The cost cell and capital stage 1 declared for a G0 sample, from the signals_report.json beside it (the
    stage-1 cell and `capital` when there is none), and what that report says about the data (stage-1 seed;
    same_data_files: whether its bid and ask sha256 equal this run's, None when unknown) [SI-69]."""
    rp = sample.parent / "signals_report.json"
    info: dict[str, Any] = {"signals_report": None, "stage1_seed": None, "same_data_files": None}
    if not rp.is_file():
        return zeno_report.STAGE1_CELL, float(capital), info
    bars_mod.check_not_locked(rp, what="signals report")
    try:
        rep = json.loads(rp.read_text(encoding="utf-8"))
        dc = rep["declared_cell"]
        cell = zv.ZenoCell(dc["variant"], dc["commission_rt_per_lot"], dc["spread_base"], dc["cost_mult"])
        cap = float(dc.get("capital_usd", capital))
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise UsageError(f"cannot read the declared cost cell from {rp} ({type(e).__name__}: {e}); it belongs to the "
                         "G0 sample beside it (zeno-v1 signals writes both)")
    data, now = rep.get("data") or {}, prep.frame.attrs.get("zeno_v1") or {}
    pairs = [(data.get(k), now.get(k)) for k in ("bid_sha256", "ask_sha256")]
    same = None if any(a is None or b is None for a, b in pairs) else all(a == b for a, b in pairs)
    info.update(signals_report=str(rp), stage1_seed=(rep.get("g0") or {}).get("seed"), same_data_files=same)
    return cell, cap, info


def _g0_sample_match(sample_path: str, prep: zv.Prepared, capital: float) -> dict[str, Any]:
    """Check that the G0 sample belongs to the data this run judges (zeno_report.g0_sample_check, [SI-69]) and
    return the record for gates.json; UsageError (before anything is written) when a row does not match."""
    p = Path(sample_path).expanduser()
    try:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
    except (OSError, ValueError) as e:
        raise UsageError(f"--g0-sample {p.name} cannot be read as CSV: {e}")
    df.columns = [str(c).strip() for c in df.columns]
    cell, cap, info = _g0_declared(p, capital, prep)
    try:
        chk = zeno_report.g0_sample_check(prep, df, cell, cap)
    except ValueError as e:
        raise UsageError(f"--g0-sample {p.name}: {e}")
    chk.update(info)
    if not chk["ok"]:
        n_bad = chk["n_rows"] - chk["n_matched"]
        u = chk["unmatched"][0]
        raise UsageError(f"the G0 sample {p.name} does not belong to this data: {n_bad} of {chk['n_rows']} rows are not "
                         f"eligible signals of these M15 files in the declared cell {chk['declared_cell']} with the "
                         f"same time, side, entry and stop (first: sample row {u['sample_no']}, {u['signal_time_utc']} "
                         f"{u['side']}: {u['why']}). G0 checks THIS data's signals: run `zeno-v1 signals` on these "
                         "files and check that sample on a chart (spec: Gates, G0). Nothing was written.")
    if chk["same_data_files"] is False:
        _say(f"NOTE: {p.name} was drawn from other files (their sha256 differ from these), but each of its rows is an "
             "eligible signal of these files with the same entry and stop.")
    return chk


def _g0_record(sample_path: str | None) -> dict[str, Any]:
    """The G0 entry of gates.json: the operator's confirmation, and with --g0-sample the file's sha256 and the
    y/n answers in its agree_y_n column. G0 needs zeno to agree with at least 18 of 20 signals, so a sample is
    refused (G0 failed or impossible) when it has more "n" answers than it can afford (more than 2 of 20;
    ceil(18 x rows / 20) y are needed from larger samples) whatever its blanks hold. Once any row holds an
    answer the sample is zeno's check [SI-60, SI-69]: it is refused ("G0 not met") when it has fewer than 20
    rows, when any row is not answered y/yes or n/no (blank, "?", ...), or when fewer than ceil(18 x rows / 20)
    rows are y. A sample with no answer at all is only recorded. cmd_zeno_run then checks that the sample's
    rows are signals of the data it judges (_g0_sample_match)."""
    rec: dict[str, Any] = {"status": "confirmed_by_operator", "confirmed_by_operator": True,
                           "threshold": f"zeno agrees with >= {zeno_report.G0_MIN_AGREE} of "
                                        f"{zeno_report.G0_SAMPLE_SIZE} sampled signals",
                           "reading": "G0 is zeno's chart check; propkit records zeno's confirmation and never computes it",
                           "sample_file": None, "sample_sha256": None, "answered": None, "agree_count": None}
    if sample_path is None:
        return rec
    bars_mod.check_not_locked(sample_path, what="G0 sample file")
    p = Path(sample_path).expanduser()
    if not p.is_file():
        raise UsageError(f"--g0-sample {sample_path}: file not found")
    data = p.read_bytes()
    try:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
    except (OSError, ValueError) as e:
        raise UsageError(f"--g0-sample {p.name} cannot be read as CSV: {e}")
    rec.update(sample_file=str(p), sample_sha256=hashlib.sha256(data).hexdigest())
    if "agree_y_n" in df.columns:
        ans = df["agree_y_n"].astype(str).str.strip().str.lower()
        yes = int(ans.isin(("y", "yes")).sum())
        no = int(ans.isin(("n", "no")).sum())
        answered = yes + no
        rows = int(len(df))
        need = math.ceil(zeno_report.G0_MIN_AGREE * max(rows, zeno_report.G0_SAMPLE_SIZE) / zeno_report.G0_SAMPLE_SIZE)
        rec.update(answered=answered, agree_count=yes, n_no=no, unanswered=rows - answered, n_rows=rows,
                   min_agree_needed=need)
        if no > max(rows - need, 0):
            raise UsageError(f"G0 failed: {p.name} says you agree with {yes} of {answered} answered signals ({no} n, "
                             f"{rows - answered} unanswered, {rows} rows); the spec needs at least "
                             f"{zeno_report.G0_MIN_AGREE} of {zeno_report.G0_SAMPLE_SIZE}, so at most "
                             f"{max(rows - need, 0)} n. Stage 2 is not run: find out why the code and your reading of "
                             "the rule differ first (spec: Gates, G0).")
        holds = bool((ans != "").any())                     # any row holds an answer: the sample is the check
        if holds and rows < zeno_report.G0_SAMPLE_SIZE:
            raise UsageError(f"G0 not met: {p.name} has {rows} rows, but G0 needs at least "
                             f"{zeno_report.G0_SAMPLE_SIZE} rows ({zeno_report.G0_MIN_AGREE} of "
                             f"{zeno_report.G0_SAMPLE_SIZE} checked signals). Draw a sample of "
                             f"{zeno_report.G0_SAMPLE_SIZE} with `zeno-v1 signals --sample {zeno_report.G0_SAMPLE_SIZE}` "
                             "and check it on a chart (spec: Gates, G0).")
        if holds and answered < rows:
            odd = sorted({a for a in df["agree_y_n"].astype(str).str.strip() if a.lower() not in ("y", "yes", "n", "no")})
            raise UsageError(f"G0 not met: {p.name}: {rows - answered} of {rows} rows are not answered y or n "
                             f"({', '.join(repr(a) for a in odd[:5])}); a blank or another answer is not an "
                             f"agreement. G0 needs at least {need} y of {rows} checked signals: answer every row "
                             "(spec: Gates, G0).")
        if holds and yes < need:
            raise UsageError(f"G0 not met: {p.name} says you agree with {yes} of {rows} signals; the spec needs at least "
                             f"{need} of {rows} ({zeno_report.G0_MIN_AGREE} of {zeno_report.G0_SAMPLE_SIZE}). Stage 2 "
                             "is not run (spec: Gates, G0).")
    return rec


def cmd_zeno_run(args) -> int:
    """`zeno-v1 run`: stage 2 - refuses without --g0-confirmed; otherwise the 24-cell grid, the prop evaluator
    and the gates, written to --out."""
    if not args.g0_confirmed:
        raise UsageError(zeno_report.G0_REFUSAL)
    if (args.m1_bid is None) != (args.m1_ask is None):
        raise UsageError("give both --m1-bid and --m1-ask (the D15 second run), or neither")
    inputs = [args.m15_bid, args.m15_ask, args.news, _rules_path(args.rules),
              _rules_path(args.reference_rules) if args.reference_rules else None, args.m1_bid, args.m1_ask,
              args.g0_sample]
    for path_text in inputs:
        if path_text is not None:
            bars_mod.check_not_locked(path_text, what="input file")
    rules, info = rules_mod.rules_and_info(args.rules, args.capital)
    reference = (rules_mod.rules_and_info(args.reference_rules, rules.initial_capital)
                 if args.reference_rules else None)
    g0 = _g0_record(args.g0_sample)
    names = zeno_report.RUN_FILES + (zeno_report.M1_DIFF_FILE,)
    out_dir = prepare_out_dir(args.out, inputs, names)
    _say(report_mod.HEADER)
    if info.get("fallback"):
        _say(f"WARNING: {info['fallback']}.")
    if info.get("verified") is False or info.get("unverified_fields"):
        _say(f"WARNING: {info.get('warning') or 'firm rules unverified: ' + ', '.join(info['unverified_fields'])}")
    prep = _zeno_data(args)
    if args.g0_sample is not None:
        _say(f"Checking that the G0 sample {Path(args.g0_sample).name} belongs to this data ...")
        g0["sample_check"] = _g0_sample_match(args.g0_sample, prep, rules.initial_capital)
        _say(f"  {g0['sample_check']['n_matched']} of {g0['sample_check']['n_rows']} sampled signals are eligible "
             f"signals of this data ({g0['sample_check']['declared_cell']}) with the same entry and stop.")
    _say(f"Running the 24 cells under {rules.name} ({rules.initial_capital:,.0f} USD) ...")
    extra = [("m1_bid", args.m1_bid), ("m1_ask", args.m1_ask), ("g0_sample", args.g0_sample),
             ("rules", _rules_path(args.rules))]
    report, tables = zeno_report.run_stage(
        prep, rules, info, n_sims=args.n_sims, seed=args.seed, history_reps=args.history_reps, reference=reference,
        m1=(args.m1_bid, args.m1_ask) if args.m1_bid else None, g0=g0, inputs=_zeno_input_info(args, extra),
        progress=lambda s: _say(f"  {s}"))
    jobs = [("report.json", _json_job(report)),
            ("report.md", lambda p: p.write_text(zeno_report.render_run_markdown(report), encoding="ascii")),
            ("gates.json", _json_job(report["gates"])),
            ("trades.csv", lambda p: adapters.write_trades_csv(tables["trades"], p)),
            ("positions.csv", _bytes_job(zeno_report.csv_bytes(tables["positions"]))),
            ("decisions.csv", _bytes_job(zeno_report.csv_bytes(tables["decisions"]))),
            ("grid.csv", _bytes_job(zeno_report.csv_bytes(tables["grid"]))),
            ("positions_all_cells.csv", _bytes_job(zeno_report.csv_bytes(tables["positions_all_cells"])))]
    if "m1_diff" in tables:
        jobs.append((zeno_report.M1_DIFF_FILE, _bytes_job(zeno_report.csv_bytes(tables["m1_diff"]))))
    written = write_staged(out_dir, jobs)
    stale = out_dir / zeno_report.M1_DIFF_FILE
    if "m1_diff" not in tables and stale.is_file():
        stale.unlink()
        _say(f"NOTE: removed {stale.name} left by an earlier run (it described that run's M1 resolution).")
    g = report["gates"]
    _say("")
    _say(f"Judging cell: {g['judging_cell']['label']}")
    for name in ("G0", "G1", "G2", "G3", "G4", "G5"):
        gate = g["gates"][name]
        _say(f"  {name}: {gate.get('status')}" + (f" - {gate.get('reading')}" if gate.get("reading") else ""))
    _say(f"  kill: {'FIRED' if g['kill']['fired'] else 'not fired'} ({g['kill']['judged_at']})")
    _say(f"Verdict: {g['verdict']}")
    _say("")
    _say("Files written:")
    for p in written:
        _say(f"  {p}")
    return EXIT_OK


def cmd_zeno_v1(args) -> int:
    """`zeno-v1 signals|run`."""
    if args.zeno_command is None:
        raise UsageError("zeno-v1 needs a stage: signals (first, for the G0 check) or run (after it)")
    return {"signals": cmd_zeno_signals, "run": cmd_zeno_run}[args.zeno_command](args)


COMMANDS = {"evaluate": cmd_evaluate, "pullback": cmd_pullback, "rules": cmd_rules, "selftest": cmd_selftest,
            "zeno-v1": cmd_zeno_v1}


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; returns the exit code (0 ok, 1 selftest failure, 2 usage or data error)."""
    parser = make_parser()
    try:
        args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
        if args.command is None:
            parser.print_help()
            return EXIT_USAGE
        return COMMANDS[args.command](args)
    except SystemExit as e:  # --help / --version
        return int(e.code or 0) if isinstance(e.code, int) or e.code is None else EXIT_USAGE
    except (UsageError, ValueError) as e:  # includes LockedPathError and every input check
        _err(str(e))
        return EXIT_USAGE
    except (OSError, MemoryError) as e:
        _err(f"{type(e).__name__}: {e}")
        return EXIT_USAGE
    except KeyboardInterrupt:
        _err("interrupted")
        return EXIT_USAGE
    except Exception as e:  # noqa: BLE001 - a safety net: one line, not a traceback, and never exit code 1
        _err(f"internal error ({type(e).__name__}: {e}). This is a propkit problem, not your input's fault as far "
             "as propkit can tell; set the environment variable PROPKIT_DEBUG=1 and run again to see the details, "
             "and send them with the command you ran.")
        if os.environ.get("PROPKIT_DEBUG"):
            _say(traceback.format_exc(), sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
