"""scripts/research/score_formula.py - score one fixed formula the way AlphaMaster's engine does, plus diagnostics.

Usage (from the repo root; Windows PowerShell shown):
    $env:PYTHONUTF8 = "1"
    python scripts\\research\\score_formula.py --data-file ..\\AlphaMaster\\research\\data\\XAUUSD_H1.parquet --formula "[58,49,27,75,94,72,77,69]"
    python scripts\\research\\score_formula.py --from-trial 20261008_010203_XAUUSD_H1_seed1_real --benchmarks
    python scripts\\research\\score_formula.py --from-trial <run id> --costs "1,2,3" --json logs\\score_seed1_real.json

The formula is a list of token ids (as in best_formula in logs/trials.jsonl). With --from-trial the
formula and the data file are taken from that run's line in the trial log (--data-file overrides
the file, e.g. when the run was made on another machine).

What it prints (and writes to --json):
  1. the walk-forward folds AlphaMaster builds for this file (bar ranges and UTC dates) and the
     bars-per-year figure the engine uses to annualise;
  2. the engine score: the formula goes through AlphaEngine._eval_formula_task itself, with an
     empty factor pool, plus the per-fold train/validation composites, IC, the repetition penalty
     and the two checks engine.train makes before it accepts a new best;
  3. position diagnostics: share of bars long/short/flat, mean |position|, turnover, and how
     |position| moves with trailing volatility;
  4. plain per-validation-fold statistics outside the composite score (annualised return, Sharpe,
     max drawdown, direction changes) for each cost multiplier;
  5. a one-bar-delay leakage check: the same factor one bar later, scored again;
  6. with --benchmarks: a 24-bar momentum factor scored the same way, and buy-and-hold.

In-sample warning: a mined formula (such as a run's best) was chosen because it scored highest on
these same validation windows, so for it every "validation" number here is in-sample and optimistic.
Compare it only with the noise-arm best formulas scored the same way; only the locked holdout test
is out of sample.

Paths containing 'locked_holdout' or ending in '.locked' are refused: the holdout test is a separate
pre-registered step. --json must end in .json and, inside the repo, go under logs\\ ; it may not be
the data file or the trial log. Exit code 0 when the report was produced (whatever the numbers
say), 2 for usage or data errors. The numbers describe this file only and say nothing about future
data. Research only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HEADER = "RESEARCH ONLY - not trading advice"
DEFAULT_TRIALS_LOG = ROOT / "logs" / "trials.jsonl"
MOMENTUM_BARS = 24
CORR_NOTE = ("A run's logged best_validation_score can be LOWER than this val_score: inside a run the "
             "factor pool (earlier bests) may apply a correlation penalty (score - 0.2 x |score| when the "
             "correlation is above 0.85). This tool scores with an empty pool, as at the start of a run.")
DELAY_NOTE = ("How to read this: a score that holds up after the delay does not depend on the newest bar. "
              "For a formula picked as the best of many, the score usually falls after any small change, "
              "even with no leak and no edge, because the selection fitted noise at this exact alignment. "
              "Judge the drop against the same check on the noise-arm best formulas (and, with "
              "--benchmarks, the unselected momentum factor). Only a drop much larger than theirs, or a "
              "score that is extreme before the delay and ordinary after it, points to the newest bar - "
              "where a look-ahead leak would sit; then check the features the formula uses. A drop alone "
              "does not prove a leak.")
SELECTION_NOTE = ("IN-SAMPLE WARNING: if this formula was mined (e.g. a run's best), it was picked because it "
                  "scored highest on these same validation windows, out of many formulas tried, and each "
                  "validation window except the last is also (most of) the next fold's training window. So "
                  "every validation number in this report - val composites, the validation Sortino, the "
                  "plain validation statistics - is in-sample for that choice and optimistic. Compare it "
                  "only with the noise-arm best formulas scored the same way; only the locked holdout test "
                  "is out of sample. The validation numbers are out of sample only for a formula fixed "
                  "before anyone looked at this file.")
DIR_CHANGES_NOTE = ("dir.changes = bars where the position changes side (flat->long/short, long/short->flat, "
                    "long<->short), with the bar before the window taken as it was (flat before bar 0); size "
                    "changes on the same side show up in turnover/bar, not here.")


class UsageError(Exception):
    """Bad input or unusable data: printed as one message, exit code 2."""


# -- small helpers (same ideas as scripts/run_trial.py) ------------------------

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


def is_locked_path(path_text: str) -> bool:
    """True for the locked holdout: the path contains 'locked_holdout' or ends in '.locked'."""
    low = str(path_text).strip().replace("\\", "/").rstrip("/").lower()
    return "locked_holdout" in low or low.endswith(".locked")


def check_not_locked(path_text: str, what: str, verb: str = "read") -> None:
    """Refuse before anything opens the file (the given text and the resolved path are both checked)."""
    candidates = [str(path_text)]
    try:
        candidates.append(str(Path(path_text).expanduser().resolve()))
    except (OSError, RuntimeError):
        pass
    if any(is_locked_path(c) for c in candidates):
        raise UsageError(
            f"refusing to {verb} the {what} {path_text}: paths containing 'locked_holdout' or ending in "
            f"'.locked' are the locked holdout. The holdout test is a separate pre-registered step and "
            f"this tool never touches it.")


def _same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(str(a)) == os.path.normcase(str(b))


def _inside(path: Path, folder: Path) -> bool:
    p, f = os.path.normcase(str(path)), os.path.normcase(str(folder)).rstrip("\\/")
    return p == f or p.startswith(f + os.sep)


def check_json_path(json_text: str, data_file: Path, trials_log: Path) -> Path:
    """Where --json may write: a .json file that is not an input, not the holdout, and inside the repo
    only under logs/ (ignored by git), so a mistyped path cannot overwrite the trial log, the data or
    a tracked file. Returns the resolved path."""
    if not str(json_text).strip():
        raise UsageError("--json needs a file name ending in .json")
    check_not_locked(json_text, "--json output path", verb="write")
    path = Path(json_text).expanduser().resolve()
    check_not_locked(str(path), "--json output path", verb="write")
    if path.suffix.lower() != ".json":
        raise UsageError(f"--json must name a file ending in .json (got {json_text})")
    if path.is_dir():
        raise UsageError(f"--json names a folder, not a file: {path}")
    for other, what in ((data_file, "the data file"), (trials_log, "the trial log")):
        if _same_path(path, other):
            raise UsageError(f"--json would overwrite {what}: {path}")
    if _inside(path, ROOT) and not _inside(path, ROOT / "logs"):
        raise UsageError(f"--json {path} is inside the repo but not under logs\\ ; write reports under "
                         f"logs\\ (ignored by git) or outside the repo, so no tracked file is overwritten")
    return path


def parse_formula(text: str) -> list[int]:
    """'[1,2,3]' (JSON), or '1,2,3' / '1 2 3' when the shell ate the brackets."""
    s = (text or "").strip()
    try:
        val = json.loads(s)
    except json.JSONDecodeError:
        val = [p for p in re.split(r"[\s,\[\]]+", s) if p]
    if isinstance(val, int) and not isinstance(val, bool):
        val = [val]
    if not isinstance(val, list) or not val:
        raise UsageError(f"--formula must be a list of token ids such as [3,120] (got {text!r})")
    tokens = []
    for v in val:
        if isinstance(v, bool) or not (isinstance(v, int) or (isinstance(v, str) and re.fullmatch(r"-?\d+", v))):
            raise UsageError(f"--formula must contain whole numbers only (got {v!r} in {text!r})")
        tokens.append(int(v))
    return tokens


def parse_costs(text: str) -> list[float]:
    out: list[float] = []
    for part in re.split(r"[\s,]+", (text or "").strip()):
        if not part:
            continue
        try:
            m = float(part)
        except ValueError:
            raise UsageError(f"--costs must be numbers separated by commas, e.g. 1,1.5,2 (got {text!r})") from None
        if not math.isfinite(m) or m < 0:
            raise UsageError(f"--costs multipliers must be 0 or more (got {part})")
        if m not in out:
            out.append(m)
    if not out:
        raise UsageError("--costs needs at least one multiplier, e.g. 1,1.5,2")
    return out


def load_trial(log_path: Path, run_id: str) -> dict:
    """The last line of the trial log whose run_id matches."""
    check_not_locked(str(log_path), "trials log")
    if not log_path.is_file():
        raise UsageError(f"trials log not found: {log_path} (use --trials-log to point at it)")
    found, ids = None, []
    with open(log_path, encoding="utf-8-sig") as f:  # -sig: tolerate a BOM added by an editor
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            ids.append(str(rec.get("run_id")))
            if rec.get("run_id") == run_id:
                found = rec
    if found is None:
        tail = ", ".join(ids[-6:]) if ids else "none"
        raise UsageError(f"run id {run_id!r} is not in {log_path}. Latest run ids there: {tail}")
    if not found.get("best_formula"):
        raise UsageError(f"trial {run_id} has no best_formula (status: {found.get('status')})")
    if not found.get("data_file"):
        raise UsageError(f"trial {run_id} has no data_file; pass --data-file")
    return found


def _fmt(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _pct(x) -> str:
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


def _utc(ts) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _mult_key(m: float) -> str:
    return f"x{m:g}"


def _clean(obj):
    """JSON-safe copy: numpy/torch scalars to Python, NaN/inf to None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if type(obj).__name__ == "bool_":
        return bool(obj)
    if isinstance(obj, int):
        return obj
    try:
        f = float(obj)
    except (TypeError, ValueError):
        return str(obj)
    if float(f).is_integer() and type(obj).__name__.startswith("int"):
        return int(f)
    return f if math.isfinite(f) else None


# -- data and engine set-up (train_file.train_from_file + engine.train) --------

class Setup:
    """Everything one scoring needs (plain class, so the module also loads via importlib).

    data_file, symbol, timeframe; engine (AlphaEngine, factor pool empty); feat [1, F, T];
    t_ret [1, T] = log(open[t+2]/open[t+1]) with the last 2 bars 0; close [1, T] float32 as the
    engine holds it; times numpy int64 [T] unix seconds; folds, use_wf and gap_config as engine.train
    builds them; periods_per_year and cost_rate as the engine's backtester uses them.
    """

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


def prepare(data_file: Path) -> Setup:
    """Load the file and build engine, folds and annualisation exactly as train_file + engine.train do."""
    from data_pipeline.parquet_manager import ParquetDataManager, inspect_parquet_file
    from model_core.backtest import estimate_periods_per_year
    from model_core.config import ModelConfig
    from model_core.engine import AlphaEngine, _build_walk_forward_folds

    data_file = Path(data_file)
    # train_file.train_from_file
    info = inspect_parquet_file(str(data_file))
    mgr = ParquetDataManager(str(data_file))
    mgr.load()
    engine = AlphaEngine(data_manager=mgr, target_symbol=info["symbol"])
    engine.timeframe = info["timeframe"]
    engine.data_file = str(data_file.resolve())
    engine.mode = "parquet_file"

    # engine.train: folds, walk-forward switch, tensors, data-driven annualisation
    T = mgr.target_ret.shape[1]
    gap_config = getattr(ModelConfig, "WF_GAP", 20)
    folds = _build_walk_forward_folds(T, engine.n_folds, gap=gap_config)
    use_wf = len(folds) > 1 and not (folds[0]["train_start"] == 0 and folds[0]["train_end"] == T)
    feat = mgr.feat_tensor.to(ModelConfig.DEVICE)
    t_ret = mgr.target_ret.to(ModelConfig.DEVICE)
    dm_raw = getattr(mgr, "raw_dict", None) or {}
    dm_time = dm_raw.get("time", None)
    if dm_time is not None:
        try:
            ppy = estimate_periods_per_year(dm_time)
            if ppy != engine.bt.periods_per_year:
                engine.bt.periods_per_year = ppy
        except Exception:
            pass  # the engine keeps its default 6240 in this case too

    return Setup(
        data_file=data_file, symbol=info["symbol"], timeframe=info["timeframe"], engine=engine,
        feat=feat, t_ret=t_ret, close=mgr.raw_dict["close"],
        times=mgr.raw_dict["time"][0].cpu().numpy(), folds=folds, use_wf=use_wf,
        gap_config=int(gap_config), periods_per_year=int(engine.bt.periods_per_year),
        cost_rate=float(engine.bt.cost_rate),
    )


# -- engine score --------------------------------------------------------------

class _FixedFactorVM:
    """Stands in for engine.vm so a ready-made factor runs through the engine's own _eval_formula_task."""

    def __init__(self, factor):
        self.factor = factor

    def execute(self, formula_tokens, feat_tensor):
        return self.factor


def engine_score(setup: Setup, formula: list[int], factor=None) -> tuple[dict, Any]:
    """Score through AlphaEngine._eval_formula_task with an empty factor pool.

    factor=None runs the formula through the engine's StackVM; otherwise the given (already
    normalised) factor replaces the VM output and everything after it is the engine's code.
    Returns (report, factor tensor or None).
    """
    import torch
    from model_core.config import ModelConfig
    from model_core.engine import AlphaEngine, _repetition_penalty, compute_target_positions_stateless

    eng, bt = setup.engine, setup.engine.bt
    if eng.factor_pool:
        raise RuntimeError("score_formula needs an empty factor pool")
    fml = [int(t) for t in formula]
    args = (setup.feat, setup.t_ret, setup.folds, setup.use_wf, [])
    if factor is None:
        r = eng._eval_formula_task(0, fml, *args)
    else:
        saved = eng.vm
        eng.vm = _FixedFactorVM(factor)
        try:
            r = eng._eval_formula_task(0, fml, *args)
        finally:
            eng.vm = saved

    status = r.get("status", "error")
    rp = _repetition_penalty(fml)
    rep: dict = {"status": status, "val_score": float(r["val_score"]), "train_reward": float(r["reward"]),
                 "repetition_penalty": rp,
                 # -2.0 (constant) and -5.0 (no output / error) are returned before any penalty
                 "repetition_penalty_applied": status == "ok",
                 "error": r.get("error"), "folds": [],
                 "mean_gated_val": None, "reproduced_val_score": None, "reproduced_train_reward": None,
                 "reproduced_matches_engine": None, "ic_full": r.get("ic_full"), "checks": None}
    if status == "ok":
        res = r["res"]
    elif status == "const":
        if factor is not None:
            res = factor
        else:
            with torch.no_grad():
                res = eng.vm.execute(fml, setup.feat)
    else:
        return rep, None
    if status != "ok":
        return rep, res

    t_ret = setup.t_ret
    with torch.no_grad():
        position = compute_target_positions_stateless(res)
        prev = torch.roll(position, 1, dims=1)
        prev[:, 0] = 0.0
        pnl = position * t_ret - torch.abs(position - prev) * bt.cost_rate
        if setup.use_wf:
            # the same calls, in the same order, as _eval_formula_task (engine.py, use_wf branch)
            fold_tr, fold_vl = [], []
            for k, fold in enumerate(setup.folds, 1):
                ts, te, vs, ve = fold["train_start"], fold["train_end"], fold["val_start"], fold["val_end"]
                tr_sc, vl_sc = bt.evaluate_fold(res, t_ret, ts, te, vs, ve)
                ic_m, _ = AlphaEngine._compute_ic(res[:, ts:te], t_ret[:, ts:te])
                tr_adj = AlphaEngine._apply_ic_gate(tr_sc, ic_m)
                fold_tr.append(ModelConfig.REWARD_ALPHA * tr_adj)
                ic_v, _ = AlphaEngine._compute_ic(res[:, vs:ve], t_ret[:, vs:ve])
                vl_adj = AlphaEngine._apply_ic_gate(vl_sc, ic_v)
                fold_vl.append(vl_adj)
                rep["folds"].append({
                    "fold": k, "train_composite": float(tr_sc.item()), "train_ic": float(ic_m.item()),
                    "train_after_ic_gate": float(tr_adj.item()),
                    "val_composite": float(vl_sc.item()), "val_oos_sortino": float(bt._sortino(pnl[:, vs:ve]).item()),
                    "val_ic": float(ic_v.item()), "val_after_ic_gate": float(vl_adj.item()),
                })
            train_score = torch.stack(fold_tr).mean()
            val_score = torch.stack(fold_vl).mean()
            rep["mean_gated_val"] = float(val_score.item())
            if rp > 0:
                train_score = train_score - rp
                val_score = val_score - rp
            # an empty factor pool leaves both unchanged (_apply_corr_penalty returns early)
            rep["reproduced_val_score"] = float(val_score.item())
            rep["reproduced_train_reward"] = float(train_score.item())
            rep["reproduced_matches_engine"] = (rep["reproduced_val_score"] == rep["val_score"]
                                                and rep["reproduced_train_reward"] == rep["train_reward"])
        exposure = float(position.abs().mean().item())

    rep["checks"] = best_checks(rep["train_reward"], rep["val_score"], exposure, pool="empty")
    return rep, res


def best_checks(train_reward: float, val_score: float, exposure: float, pool: str) -> dict:
    """The two checks engine.train makes before accepting a new best (engine.py, 'if final_val > self.best_score')."""
    overfit_skip = train_reward > 0.5 and val_score < train_reward * 0.5
    sparse_skip = exposure < 0.05
    return {
        "factor_pool": pool,
        "train_reward": train_reward,
        "val_score": val_score,
        "overfit_check_passed": not overfit_skip,
        "exposure": exposure,
        "exposure_check_passed": not sparse_skip,
        "would_be_accepted_as_best": not overfit_skip and not sparse_skip,
    }


def corr_penalised_score(setup: Setup, formula: list[int], res, exposure: float) -> dict:
    """The engine's scores when its factor pool holds a factor correlated with this one (|corr| above
    ModelConfig.CORR_THRESHOLD), as for a run whose earlier best looks like this formula: the factor
    itself is put in the pool and _eval_formula_task runs again, so _apply_corr_penalty fires once.
    Returns the penalised train reward and val_score and the best-update checks on those values."""
    eng = setup.engine
    saved_pool, saved_vm = eng.factor_pool, eng.vm
    eng.factor_pool = [(0.0, 0, res)]
    eng.vm = _FixedFactorVM(res)
    try:
        r = eng._eval_formula_task(0, [int(t) for t in formula], setup.feat, setup.t_ret, setup.folds,
                                   setup.use_wf, list(eng.factor_pool))
    finally:
        eng.factor_pool, eng.vm = saved_pool, saved_vm
    if r.get("status") != "ok":
        return {"status": r.get("status"), "checks": None}
    return {"status": "ok", "train_reward": float(r["reward"]), "val_score": float(r["val_score"]),
            "checks": best_checks(float(r["reward"]), float(r["val_score"]), exposure,
                                  pool="one correlated factor (one correlation penalty)")}


# -- factors derived from the scored one --------------------------------------

def delay_one_bar(factor):
    """factor_delayed[t] = factor[t-1]; bar 0 gets 0, the value the VM gives its warm-up bars."""
    import torch
    out = torch.zeros_like(factor)
    out[:, 1:] = factor[:, :-1]
    return out


def momentum_factor(close, bars: int = MOMENTUM_BARS):
    """log(close[t]/close[t-bars]) (0 for the first bars), normalised with the VM's own _normalize_output."""
    import torch
    from model_core.vm import StackVM
    raw = torch.zeros_like(close)
    raw[:, bars:] = torch.log(close[:, bars:] / close[:, :-bars])
    raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return StackVM._normalize_output(raw)


def constant_factor(t_ret):
    import torch
    return torch.ones_like(t_ret)


def positions_of(factor):
    """compute_target_positions_stateless (the engine's own binding) as float64 numpy [T]."""
    import torch
    from model_core.engine import compute_target_positions_stateless
    with torch.no_grad():
        return compute_target_positions_stateless(factor)[0].double().cpu().numpy()


# -- plain statistics (numpy, float64) -----------------------------------------

def pnl_series(position, target_ret, cost_rate: float, mult: float):
    """PnL[t] = p[t]*target_ret[t] - |p[t]-p[t-1]|*cost_rate*mult, p[-1] = 0 (model_core/backtest.py)."""
    import numpy as np
    p = np.asarray(position, dtype=np.float64)
    prev = np.concatenate(([0.0], p[:-1]))
    return p * np.asarray(target_ret, dtype=np.float64) - np.abs(p - prev) * cost_rate * mult


def pnl_stats(pnl, periods_per_year: float) -> dict:
    """Annualised mean log return, Sharpe (population std), max drawdown of cumulative log PnL from 0."""
    import numpy as np
    x = np.asarray(pnl, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return {"bars": 0, "total_log_return": None, "ann_return": None, "sharpe": None, "max_drawdown": None}
    mean = float(x.mean())
    std = float(x.std())
    equity = np.concatenate(([0.0], np.cumsum(x)))
    mdd = float((np.maximum.accumulate(equity) - equity).max())
    return {
        "bars": n,
        "total_log_return": float(x.sum()),
        "ann_return": mean * periods_per_year,
        "sharpe": mean / std * math.sqrt(periods_per_year) if std > 0 else None,
        "max_drawdown": mdd,
    }


def direction_changes(position, start: int, end: int) -> int:
    """Bars in [start, end) where the position changes side: entries (flat -> long/short), exits
    (long/short -> flat) and reversals (long <-> short); bar -1 counts as flat. Positions are
    continuous (tanh), so a plain 'differs from the bar before' count would include almost every bar."""
    import numpy as np
    s = np.sign(np.asarray(position, dtype=np.float64))
    prev = s[start - 1] if start > 0 else 0.0
    seg = s[start:end]
    prevs = np.concatenate(([prev], seg[:-1]))
    return int((seg != prevs).sum())


def plain_val_stats(position, target_ret, folds, costs, cost_rate: float, periods_per_year: float) -> dict:
    import numpy as np
    out = {}
    for m in costs:
        pnl = pnl_series(position, target_ret, cost_rate, m)
        rows, parts, changes = [], [], 0
        for k, f in enumerate(folds, 1):
            s, e = f["val_start"], f["val_end"]
            st = pnl_stats(pnl[s:e], periods_per_year)
            st["direction_changes"] = direction_changes(position, s, e)
            st["fold"] = k
            rows.append(st)
            parts.append(pnl[s:e])
            changes += st["direction_changes"]
        all_val = pnl_stats(np.concatenate(parts), periods_per_year)
        all_val["direction_changes"] = changes
        out[_mult_key(m)] = {"cost_multiplier": m, "folds": rows, "all_val": all_val}
    return out


def buy_and_hold_stats(target_ret, folds, costs, cost_rate: float, periods_per_year: float) -> dict:
    """Position +1 on every validation bar; one entry cost (cost_rate*mult) on each fold's first bar."""
    import numpy as np
    r = np.asarray(target_ret, dtype=np.float64)
    out = {}
    for m in costs:
        rows, parts = [], []
        for k, f in enumerate(folds, 1):
            seg = r[f["val_start"]:f["val_end"]].copy()
            if seg.size:
                seg[0] -= cost_rate * m
            st = pnl_stats(seg, periods_per_year)
            st["direction_changes"] = 1  # the entry from flat at the fold's first bar
            st["fold"] = k
            rows.append(st)
            parts.append(seg)
        all_val = pnl_stats(np.concatenate(parts), periods_per_year)
        all_val["direction_changes"] = len(rows)
        out[_mult_key(m)] = {"cost_multiplier": m, "folds": rows, "all_val": all_val}
    return out


def trailing_vol(close, window: int):
    """Std of the last `window` close-to-close log returns, ending at bar t (causal); NaN before that."""
    import numpy as np
    import pandas as pd
    c = np.asarray(close, dtype=np.float64)
    r = np.full(c.shape, np.nan)
    r[1:] = np.diff(np.log(c))
    return pd.Series(r).rolling(window, min_periods=window).std().to_numpy()


def _pearson(a, b):
    import numpy as np
    if a.size < 3:
        return None
    sa, sb = float(a.std()), float(b.std())
    if sa <= 0 or sb <= 0:
        return None
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def position_window_stats(position, start: int, end: int, vol) -> dict:
    import numpy as np
    p = np.asarray(position, dtype=np.float64)
    seg = p[start:end]
    prev = p[start - 1] if start > 0 else 0.0
    turnover = np.abs(seg - np.concatenate(([prev], seg[:-1])))
    v = np.asarray(vol, dtype=np.float64)[start:end]
    ok = np.isfinite(v)
    return {
        "bars": int(seg.size),
        "share_long": float((seg > 0).mean()),
        "share_short": float((seg < 0).mean()),
        "share_flat": float((seg == 0).mean()),
        "mean_abs_position": float(np.abs(seg).mean()),
        "mean_position": float(seg.mean()),
        "turnover_per_bar": float(turnover.mean()),
        "corr_abs_position_trailing_vol": _pearson(np.abs(seg[ok]), v[ok]),
    }


def position_diagnostics(position, folds, vol) -> dict:
    return {
        "full": position_window_stats(position, 0, len(position), vol),
        "val_folds": [dict(position_window_stats(position, f["val_start"], f["val_end"], vol), fold=k)
                      for k, f in enumerate(folds, 1)],
    }


CMP_PENALISED = "lower by one correlation penalty"


def compare_logged(logged, ours: float, corr_penalty: float, penalised_engine: float | None = None) -> str:
    """How a run's logged best_validation_score relates to the empty-pool score computed here.
    penalised_engine: the engine's own val_score with one correlation penalty (corr_penalised_score)."""
    if logged is None:
        return "the trial logged no best_validation_score"
    logged = float(logged)
    if logged == ours:
        return "equal, bit for bit: the run logged this score (no correlation penalty applied)"
    if penalised_engine is not None and logged == penalised_engine and logged != ours:
        return f"{CMP_PENALISED}, bit for bit: consistent with the run's factor pool"
    # float32 sums can differ in the last digits with another torch thread count
    tol = 1e-5 * max(1.0, abs(ours))
    if abs(logged - ours) <= tol:
        return "equal within float rounding: the run logged this score (no correlation penalty applied)"
    penalised = ours - abs(ours) * (1.0 - corr_penalty)
    if abs(logged - penalised) <= tol:
        return f"{CMP_PENALISED}: consistent with the run's factor pool"
    if logged < ours:
        return ("lower, but not by one correlation penalty: check the data file, commit and config "
                "the run used")
    return ("HIGHER than the score computed here: unexpected on the same data and code; check the data "
            "file, commit and config the run used")


# -- printing ------------------------------------------------------------------

def _note(text: str) -> None:
    print(textwrap.fill(text, width=100, initial_indent="  ", subsequent_indent="  ", break_on_hyphens=False))


def _table(header: list[str], rows: list[list[str]], indent: str = "  ") -> None:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    print(indent + "  ".join(str(h).rjust(w) for h, w in zip(header, widths)))
    for r in rows:
        print(indent + "  ".join(str(c).rjust(w) for c, w in zip(r, widths)))


def _print_engine(rep: dict) -> None:
    print(f"  status: {rep['status']}" + (f" ({rep['error']})" if rep.get("error") else ""))
    if rep["folds"]:
        _table(["fold", "train comp", "train IC", "train gated", "val comp", "val Sortino", "val IC", "val gated"],
               [[f["fold"], _fmt(f["train_composite"]), _fmt(f["train_ic"]), _fmt(f["train_after_ic_gate"]),
                 _fmt(f["val_composite"]), _fmt(f["val_oos_sortino"], 3), _fmt(f["val_ic"]),
                 _fmt(f["val_after_ic_gate"])] for f in rep["folds"]])
        from model_core.config import ModelConfig
        _note(f"train/val are AlphaMaster's walk-forward windows (nothing is fitted when a fixed formula is "
              f"scored). comp = backtester composite; val comp already includes the engine's Sortino gate "
              f"on the validation window (val Sortino column; the engine calls it out-of-sample, which holds "
              f"only for a formula fixed in advance). gated = after the IC gate: IC above "
              f"+{ModelConfig.IC_GATE_THRESH:g} x{ModelConfig.IC_GATE_MULT:g}, below "
              f"-{ModelConfig.IC_GATE_THRESH:g} x{ModelConfig.IC_NEG_MULT:g} "
              f"(applied to |score|, so a negative score gets more negative under the penalty).")
        print(f"  mean of gated validation composites : {_fmt(rep['mean_gated_val'])}")
    if rep.get("repetition_penalty_applied", True):
        print(f"  repetition penalty                  : {_fmt(rep['repetition_penalty'])} (0.3 per extra repeat of the same adjacent token)")
    else:
        print(f"  repetition penalty                  : not applied (status {rep['status']}: the engine returns "
              f"-2.0/-5.0 before any penalty)")
    print(f"  engine val_score                    : {_fmt(rep['val_score'], 6)}")
    print(f"  engine train-side reward            : {_fmt(rep['train_reward'], 6)}")
    if rep["reproduced_matches_engine"] is not None:
        print(f"  fold breakdown reproduces the engine: {'yes (bit-equal)' if rep['reproduced_matches_engine'] else 'NO - report this'}")
    c = rep.get("checks")
    if c:
        print("  checks engine.train makes before accepting a new best (on these empty-pool values):")
        _print_checks(c, indent="    ")


def _print_checks(c: dict, indent: str) -> None:
    print(f"{indent}overfit check : {'PASS' if c['overfit_check_passed'] else 'FAIL'} "
          f"(fails if train reward > 0.5 and val_score < 0.5 x train reward; "
          f"train {_fmt(c['train_reward'])}, val {_fmt(c['val_score'])})")
    print(f"{indent}exposure check: {'PASS' if c['exposure_check_passed'] else 'FAIL'} "
          f"mean |position| = {_fmt(c['exposure'])} (fails if < 0.05)")


def _print_positions(diag: dict, vol_window: int) -> None:
    rows = []
    for name, d in [("full", diag["full"])] + [(f"val {d['fold']}", d) for d in diag["val_folds"]]:
        rows.append([name, d["bars"], _pct(d["share_long"]), _pct(d["share_short"]), _pct(d["share_flat"]),
                     _fmt(d["mean_abs_position"]), _fmt(d["mean_position"]), _fmt(d["turnover_per_bar"]),
                     _fmt(d["corr_abs_position_trailing_vol"], 3)])
    _table(["window", "bars", "long", "short", "flat", "mean|p|", "mean p", "turnover/bar",
            f"corr(|p|,vol{vol_window})"], rows)


def _stat_row(name: str, st: dict) -> list[str]:
    return [name, st["bars"], _fmt(st["ann_return"]), _fmt(st["sharpe"], 3), _fmt(st["max_drawdown"]),
            _fmt(st["total_log_return"]), st["direction_changes"]]


def _print_plain(stats: dict) -> None:
    for key, block in stats.items():
        print(f"  cost {key}:")
        rows = [_stat_row(f"val {st['fold']}", st) for st in block["folds"]]
        rows.append(_stat_row("all val", block["all_val"]))
        _table(["window", "bars", "ann.return", "Sharpe", "max DD", "total", "dir.changes"], rows, indent="    ")


def _print_side_by_side(scored: dict, delayed: dict) -> None:
    for key in scored:
        print(f"  cost {key}:")
        rows = []
        pairs = list(zip(scored[key]["folds"], delayed[key]["folds"]))
        pairs.append((scored[key]["all_val"], delayed[key]["all_val"]))
        for i, (a, b) in enumerate(pairs, 1):
            name = f"val {i}" if i <= len(scored[key]["folds"]) else "all val"
            rows.append([name, _fmt(a["ann_return"]), _fmt(b["ann_return"]), _fmt(a["sharpe"], 3),
                         _fmt(b["sharpe"], 3), a["direction_changes"], b["direction_changes"]])
        _table(["window", "ann.ret scored", "ann.ret delayed", "Sharpe scored", "Sharpe delayed",
                "dir.ch scored", "dir.ch delayed"], rows, indent="    ")


# -- main ----------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-file", help="{SYMBOL}_{TF}.parquet file to score on (required unless --from-trial)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--formula", nargs="+",
                     help="token ids as JSON, e.g. \"[3,120]\" (also accepts 3,120 or 3 120)")
    src.add_argument("--from-trial", metavar="RUN_ID",
                     help="take best_formula and data_file from this run's line in the trial log")
    ap.add_argument("--trials-log", default=str(DEFAULT_TRIALS_LOG),
                    help="trial log for --from-trial (default: logs/trials.jsonl in the repo)")
    ap.add_argument("--costs", nargs="+", default=["1,1.5,2"],
                    help="cost multipliers of Config.COST_RATE for the plain statistics, e.g. \"1,1.5,2\" "
                         "(the default) or 1 1.5 2")
    ap.add_argument("--vol-window", type=int, default=24,
                    help="bars of close-to-close log returns in the trailing volatility (default: 24)")
    ap.add_argument("--benchmarks", action="store_true",
                    help="also score a 24-bar momentum factor and buy-and-hold")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads (0 = AlphaMaster default). With --from-trial the default is the "
                         "trial's torch_threads: float32 sums can differ in the last digits with another count")
    ap.add_argument("--json", metavar="PATH",
                    help="also write the full report to this .json file (inside the repo only under logs\\; "
                         "never the data file or the trial log)")
    return ap


def main(argv: list[str] | None = None) -> int:
    print(HEADER, flush=True)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass
    ap = _build_parser()
    a = ap.parse_args(argv)
    if not a.from_trial and not a.data_file:
        ap.error("--data-file is required unless --from-trial is used")
    if a.vol_window < 2:
        ap.error("--vol-window must be at least 2")
    if a.threads is not None and a.threads < 0:
        ap.error("--threads must be 0 or more")
    try:
        return _run(a)
    except UsageError as e:
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        return 2


def _run(a: argparse.Namespace) -> int:
    t0 = time.time()
    costs = parse_costs(",".join(a.costs))
    trial = None
    trials_log = Path(a.trials_log).expanduser().resolve()
    if a.from_trial:
        trial = load_trial(trials_log, a.from_trial)
        try:
            formula = [int(t) for t in trial["best_formula"]]
        except (TypeError, ValueError):
            raise UsageError(f"trial {a.from_trial} has an unreadable best_formula: {trial['best_formula']!r}") from None
        data_text = a.data_file or str(trial["data_file"])
    else:
        formula = parse_formula(",".join(a.formula) if len(a.formula) > 1 else a.formula[0])
        data_text = a.data_file
    check_not_locked(data_text, "data file")

    data_file = Path(data_text).expanduser().resolve()
    check_not_locked(str(data_file), "data file")
    if not data_file.is_file():
        hint = " (the trial's path may be from another machine: pass --data-file)" if trial and not a.data_file else ""
        raise UsageError(f"data file not found: {data_file}{hint}")
    json_path = check_json_path(a.json, data_file, trials_log) if a.json is not None else None

    os.chdir(ROOT)  # AlphaMaster resolves its config relative to the repo root (as scripts/run_trial.py)
    from data_pipeline.parquet_manager import parse_parquet_filename
    try:
        parse_parquet_filename(data_file)
    except ValueError:
        raise UsageError(f"the file name must look like SYMBOL_TF.parquet (e.g. XAUUSD_H1.parquet): {data_file.name}") from None

    import torch
    from config import Config
    from model_core.backtest import SCORE_VERSION
    from model_core.config import ModelConfig
    from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION

    import model_core.engine  # noqa: F401  (sets AlphaMaster's default torch threads on import)

    ModelConfig.REWARD_MODE = "ftmo"  # same as train_file.py's command line and scripts/run_trial.py
    threads = a.threads
    if threads is None and trial and isinstance(trial.get("torch_threads"), int):
        threads = trial["torch_threads"]
    if threads and threads > 0:
        torch.set_num_threads(threads)
    bad = [t for t in formula if not 0 <= t < FORMULA_VOCAB.size]
    if bad:
        raise UsageError(f"token ids {bad} are outside the vocabulary (0..{FORMULA_VOCAB.size - 1}, "
                         f"vocab version {VOCAB_VERSION})")

    data_sha = _sha256(data_file)
    commit = _git_commit()
    try:
        setup = prepare(data_file)
    except Exception as e:
        raise UsageError(f"could not load {data_file}: {type(e).__name__}: {e}") from None
    eng = setup.engine
    decoded = eng._decode_formula(formula)
    T = int(setup.t_ret.shape[1])
    ppy = setup.periods_per_year

    print("\n== Inputs ==")
    print(f"  data file      : {data_file}")
    print(f"  data sha256    : {data_sha}")
    print(f"  symbol / tf    : {setup.symbol} {setup.timeframe}, {T} bars, {_utc(setup.times[0])} .. {_utc(setup.times[-1])} UTC")
    print(f"  formula        : {formula}")
    print(f"  decoded        : {decoded}")
    if trial:
        print(f"  source         : trial {trial.get('run_id')} (tag {trial.get('tag')!r}, seed {trial.get('seed')}, "
              f"status {trial.get('status')})")
        print(f"  trial commit   : {trial.get('git_commit')}")
        if trial.get("data_sha256") and trial.get("data_sha256") != data_sha:
            print("  WARNING        : this file's sha256 differs from the trial's data_sha256 - not the same data")
    else:
        print("  source         : --formula")
    print(f"  git commit now : {commit}")
    if trial and trial.get("git_commit") and commit and trial["git_commit"] != commit:
        print("  WARNING        : the trial ran on another commit; scores match only if the scoring code is the same")
    print(f"  score version  : {SCORE_VERSION}   vocab version: {VOCAB_VERSION}   reward mode: {ModelConfig.REWARD_MODE}")
    print(f"  cost rate      : {setup.cost_rate:g} per unit of position change (Config.COST_RATE); "
          f"multipliers for plain stats: {', '.join(f'{m:g}' for m in costs)}")
    print(f"  bars per year  : {ppy} (estimated from the timestamps, as engine.train does)")
    print(f"  torch threads  : {torch.get_num_threads()}")

    print("\n== Walk-forward folds (as engine.train builds them) ==")
    gap_used = setup.folds[0]["gap"] if setup.folds else 0
    print(f"  n_folds={eng.n_folds} (AlphaEngine default), WF_GAP={setup.gap_config} in config, gap used={gap_used}"
          + ("  (the gap collapses when the data is too short for it)" if gap_used != setup.gap_config else ""))
    if not setup.use_wf:
        print("  NOTE: the engine does not use walk-forward folds for data this short (80/20 split); "
              "per-fold breakdowns below are not available.")
    fold_rows = []
    folds_json = []
    for k, f in enumerate(setup.folds, 1):
        tr = f"{_utc(setup.times[f['train_start']])} .. {_utc(setup.times[f['train_end'] - 1])}"
        vl = f"{_utc(setup.times[f['val_start']])} .. {_utc(setup.times[f['val_end'] - 1])}"
        fold_rows.append([k, f"[{f['train_start']}, {f['train_end']})", tr, f"[{f['val_start']}, {f['val_end']})", vl])
        folds_json.append(dict(f, fold=k, train_utc=tr, val_utc=vl))
    _table(["fold", "train bars", "train dates (UTC)", "val bars", "val dates (UTC)"], fold_rows)

    print("\n== Engine score (AlphaEngine._eval_formula_task, empty factor pool) ==")
    rep, res = engine_score(setup, formula)
    _print_engine(rep)
    if res is None:
        raise UsageError(f"the formula does not execute on this data (engine status {rep['status']!r}, "
                         f"engine score {rep['val_score']}); check the token list")
    if rep["status"] == "const":
        print("  the formula's output is constant, so the engine scores it -2.0 by construction")
    _note(CORR_NOTE)
    pen = None
    if rep["status"] == "ok":
        pen = corr_penalised_score(setup, formula, res, rep["checks"]["exposure"])
        rep["with_one_corr_penalty"] = pen
    trial_cmp = None
    if trial:
        trial_cmp = compare_logged(trial.get("best_validation_score"), rep["val_score"], ModelConfig.CORR_PENALTY,
                                   penalised_engine=pen["val_score"] if pen and pen["status"] == "ok" else None)
        print(f"  trial's logged best_validation_score: {_fmt(trial.get('best_validation_score'), 6)} -> {trial_cmp}")
    pc = pen.get("checks") if pen else None
    if pc:
        if trial_cmp and trial_cmp.startswith(CMP_PENALISED):
            print("  the run's logged score includes one correlation penalty, so engine.train made its checks on "
                  "the penalised values:")
            _print_checks(pc, indent="    ")
        elif pc["overfit_check_passed"] != rep["checks"]["overfit_check_passed"]:
            print("  NOTE: inside a run whose factor pool holds a factor correlated with this one "
                  f"(|corr| > {ModelConfig.CORR_THRESHOLD:g}), engine.train checks the penalised values and the "
                  "overfit verdict changes:")
            _print_checks(pc, indent="    ")
    print()
    if trial:
        if trial.get("unique_formulas"):
            how_many = f"the {trial['unique_formulas']} distinct formulas"
        elif trial.get("formulas_evaluated"):
            how_many = f"{trial['formulas_evaluated']} formula evaluations"
        else:
            how_many = "all the formulas"
        _note(f"This formula is the best of {how_many} that run {trial.get('run_id')} tried, picked by this "
              f"same validation score. " + SELECTION_NOTE)
    else:
        _note(SELECTION_NOTE)

    close = setup.close[0].double().cpu().numpy()
    t_ret = setup.t_ret[0].double().cpu().numpy()
    vol = trailing_vol(close, a.vol_window)
    pos = positions_of(res)

    print("\n== Positions (compute_target_positions_stateless: tanh(factor), |p| < 0.05 -> flat) ==")
    diag = position_diagnostics(pos, setup.folds, vol)
    _print_positions(diag, a.vol_window)
    print(f"  turnover/bar = mean |p[t] - p[t-1]|; vol{a.vol_window} = std of the last {a.vol_window} "
          f"close-to-close log returns up to bar t (causal)")

    print("\n== Plain validation-fold statistics (outside the composite score; in-sample for a mined formula) ==")
    print(f"  PnL[t] = p[t] x target_ret[t] - |p[t] - p[t-1]| x {setup.cost_rate:g} x cost multiplier, "
          f"target_ret[t] = log(open[t+2]/open[t+1]) (the engine's alignment; the last 2 bars have no return)")
    print(f"  ann.return = mean log PnL per bar x {ppy}; Sharpe = mean / std (population) x sqrt({ppy}); "
          f"max DD = largest fall of cumulative log PnL from its running peak (start = 0)")
    plain = plain_val_stats(pos, t_ret, setup.folds, costs, setup.cost_rate, ppy)
    _print_plain(plain)
    _note(DIR_CHANGES_NOTE)

    print("\n== One-bar-delay leakage check ==")
    if rep["status"] == "const":
        delay_check = {"skipped": "the formula's output is constant (engine score -2.0), so there is no "
                                  "timing to test; shifting it would only add an artificial step at bar 0"}
        print(f"  skipped: {delay_check['skipped']}")
    else:
        print("  factor_delayed[t] = factor[t-1] (bar 0 = 0, like the engine's warm-up bars); same scoring path")
        delayed = delay_one_bar(res)
        rep_d, res_d = engine_score(setup, formula, factor=delayed)
        plain_d = plain_val_stats(positions_of(res_d), t_ret, setup.folds, costs, setup.cost_rate, ppy) \
            if res_d is not None else None
        _table(["", "as scored", "delayed 1 bar"],
               [["engine val_score", _fmt(rep["val_score"], 6), _fmt(rep_d["val_score"], 6)],
                ["engine status", rep["status"], rep_d["status"]]]
               + [[f"val {i + 1} gated", _fmt(x["val_after_ic_gate"]), _fmt(y["val_after_ic_gate"])]
                  for i, (x, y) in enumerate(zip(rep["folds"], rep_d["folds"]))])
        if plain_d is not None:
            _print_side_by_side(plain, plain_d)
        _note(DELAY_NOTE)
        delay_check = {"engine": rep_d, "plain_val_stats": plain_d}

    bench = None
    if a.benchmarks:
        print("\n== Benchmarks ==")
        print(f"  (a) {MOMENTUM_BARS}-bar momentum: factor = log(close[t]/close[t-{MOMENTUM_BARS}]), "
              f"normalised by StackVM._normalize_output, scored through the same engine path")
        mom = momentum_factor(setup.close, MOMENTUM_BARS)
        rep_m, res_m = engine_score(setup, [], factor=mom)
        _print_engine(rep_m)
        mom_pos = positions_of(res_m) if res_m is not None else None
        mom_diag = position_diagnostics(mom_pos, setup.folds, vol) if mom_pos is not None else None
        mom_plain = plain_val_stats(mom_pos, t_ret, setup.folds, costs, setup.cost_rate, ppy) \
            if mom_pos is not None else None
        if mom_diag is not None:
            print("  positions:")
            _print_positions(mom_diag, a.vol_window)
            print("  plain validation-fold statistics:")
            _print_plain(mom_plain)
        mom_delay = None
        if rep_m["status"] == "ok":
            rep_md, _ = engine_score(setup, [], factor=delay_one_bar(res_m))
            mom_delay = {"val_score": rep_md["val_score"], "status": rep_md["status"],
                         "change": rep_md["val_score"] - rep_m["val_score"]}
            print(f"  one-bar-delay check on this unselected factor (a reference for the drop above): "
                  f"val_score {_fmt(rep_m['val_score'])} -> {_fmt(rep_md['val_score'])} "
                  f"(change {mom_delay['change']:+.4f})")
        print("  (b) buy-and-hold: position +1 on every bar, one entry cost at the first bar of each validation fold")
        rep_c, _ = engine_score(setup, [], factor=constant_factor(setup.t_ret))
        print(f"  the engine scores any constant factor -2.0 by construction (res.std() < 1e-4); "
              f"checked here: status {rep_c['status']}, score {_fmt(rep_c['val_score'])}")
        bh = buy_and_hold_stats(t_ret, setup.folds, costs, setup.cost_rate, ppy)
        print("  plain validation-fold statistics:")
        _print_plain(bh)
        bench = {
            "momentum_24": {"definition": f"log(close[t]/close[t-{MOMENTUM_BARS}]) -> StackVM._normalize_output",
                            "engine": rep_m, "positions": mom_diag, "plain_val_stats": mom_plain,
                            "delay_check": mom_delay},
            "buy_and_hold": {"definition": "position +1 every bar, one entry cost at each validation fold's first bar",
                             "engine_constant_factor": {"status": rep_c["status"], "val_score": rep_c["val_score"]},
                             "plain_val_stats": bh},
        }

    report = {
        "label": "research only - not trading advice",
        "tool": "scripts/research/score_formula.py",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "inputs": {
            "data_file": str(data_file), "data_sha256": data_sha, "symbol": setup.symbol,
            "timeframe": setup.timeframe, "bars": T, "first_bar_utc": _utc(setup.times[0]),
            "last_bar_utc": _utc(setup.times[-1]), "formula": formula, "formula_decoded": decoded,
            "source": "trial" if trial else "formula",
            "trial": None if not trial else {
                "run_id": trial.get("run_id"), "tag": trial.get("tag"), "seed": trial.get("seed"),
                "git_commit": trial.get("git_commit"), "data_file": trial.get("data_file"),
                "data_sha256": trial.get("data_sha256"),
                "data_sha256_matches": trial.get("data_sha256") == data_sha if trial.get("data_sha256") else None,
                "best_validation_score": trial.get("best_validation_score"),
                "logged_vs_recomputed": trial_cmp,
                "unique_formulas": trial.get("unique_formulas"),
                "formulas_evaluated": trial.get("formulas_evaluated"),
            },
            "cost_rate": setup.cost_rate, "cost_multipliers": costs, "vol_window": a.vol_window,
            "periods_per_year": ppy, "torch_threads": torch.get_num_threads(), "reward_mode": ModelConfig.REWARD_MODE, "score_version": SCORE_VERSION,
            "vocab_version": VOCAB_VERSION, "n_folds": eng.n_folds, "wf_gap_config": setup.gap_config,
            "use_walk_forward": setup.use_wf, "min_trade_exposure": float(getattr(Config, "MIN_TRADE_EXPOSURE", 0.05)),
        },
        "folds": folds_json,
        "engine": rep,
        "positions": diag,
        "plain_val_stats": plain,
        "delay_check": delay_check,
        "benchmarks": bench,
        "notes": [SELECTION_NOTE, CORR_NOTE, DELAY_NOTE, DIR_CHANGES_NOTE,
                  "engine.checks are engine.train's best-update checks on the empty-pool values; "
                  "engine.with_one_corr_penalty holds the same checks on the values a run would see if its "
                  "factor pool held a correlated factor.",
                  "Plain statistics use float64; the engine's composite uses float32 tensors."],
        "wall_seconds": round(time.time() - t0, 1),
    }
    if json_path is not None:
        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(_clean(report), f, indent=2, ensure_ascii=True)
        except OSError as e:
            raise UsageError(f"could not write the --json report to {json_path}: {e}") from None
        print(f"\nJSON report written to {json_path}")
    print(f"\nDone in {time.time() - t0:.1f} s. {HEADER}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
