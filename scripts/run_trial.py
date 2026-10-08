"""scripts/run_trial.py - one fixed-size, seeded training run, logged as a research trial.

Usage (from the repo root; Windows PowerShell shown):
    $env:PYTHONUTF8 = "1"
    python scripts\\run_trial.py --data-file D:\\data\\XAUUSD_H1.parquet --steps 300 --seed 1 [--threads 4]

What it does differently from `python train_file.py --data-file ... --from-scratch`:
  * trains for exactly --steps steps (ModelConfig.TRAIN_STEPS is overridden for this run only);
  * seeds python/numpy/torch so a run can be repeated;
  * starts truly from scratch: an existing strategies/best_<SYMBOL>.json is NOT used as a
    score floor, so the reported best belongs to this run alone (the file itself is still
    only overwritten when this run beats it, as in train_file.py);
  * counts every formula evaluation (the trial count needed for multiple-testing checks)
    and appends one JSON line to logs/trials.jsonl.

The best score is AlphaMaster's composite validation score on its walk-forward folds of the
data file. It is chosen from thousands of formulas, so it is optimistic: compare it against a
placebo run of the same size before reading anything into it. Research only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-file", required=True, help="{SYMBOL}_{TF}.parquet file to train on")
    ap.add_argument("--steps", type=int, required=True, help="training steps for this run")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = AlphaMaster default)")
    ap.add_argument("--tag", default="", help="free-text label stored with the trial")
    a = ap.parse_args()
    if a.steps < 1:
        ap.error("--steps must be at least 1")

    data_file = Path(a.data_file).resolve()
    if not data_file.is_file():
        ap.error(f"data file not found: {data_file}")
    os.chdir(ROOT)  # checkpoints/, strategies/ and logs/ are relative to the repo root

    import numpy as np
    import torch

    import train_file
    from model_core.backtest import SCORE_VERSION
    from model_core.config import ModelConfig
    from model_core.engine import AlphaEngine

    ModelConfig.REWARD_MODE = "ftmo"          # same as train_file.py's command line
    ModelConfig.TRAIN_STEPS = a.steps
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    # Truly from scratch: no score floor from an earlier run's strategy file.
    train_file._seed_best_from_strategy = lambda engine, symbol: None

    seen: set[tuple[int, ...]] = set()
    counts = {"evaluated": 0}
    original_task = AlphaEngine._eval_formula_task

    def counting_task(self, idx, fml, *args, **kwargs):
        counts["evaluated"] += 1
        seen.add(tuple(int(t) for t in fml))
        return original_task(self, idx, fml, *args, **kwargs)

    AlphaEngine._eval_formula_task = counting_task

    started = datetime.now(timezone.utc)
    t0 = time.time()
    engine = train_file.train_from_file(str(data_file), from_scratch=True)
    wall = time.time() - t0

    record = {
        "label": "research only",
        "tag": a.tag,
        "started_utc": started.isoformat(timespec="seconds"),
        "wall_seconds": round(wall, 1),
        "seconds_per_step": round(wall / a.steps, 1),
        "git_commit": _git_commit(),
        "data_file": str(data_file),
        "data_sha256": _sha256(data_file),
        "steps": a.steps,
        "seed": a.seed,
        "torch_threads": torch.get_num_threads(),
        "batch_size": ModelConfig.BATCH_SIZE,
        "formulas_evaluated": counts["evaluated"],
        "unique_formulas": len(seen),
        "score_version": SCORE_VERSION,
        "completed": engine is not None,
    }
    if engine is not None:
        record.update({
            "symbol": engine.target_symbol,
            "best_validation_score": None if engine.best_formula is None else float(engine.best_score),
            "best_formula": engine.best_formula,
            "best_formula_decoded": (engine._decode_formula(engine.best_formula)
                                     if engine.best_formula else None),
        })

    log = ROOT / "logs" / "trials.jsonl"
    log.parent.mkdir(exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n=== Trial record (research only) ===")
    print(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"Appended to {log}")
    return 0 if engine is not None else 1


if __name__ == "__main__":
    sys.exit(main())
