"""propkit - a research toolkit that answers "what does this trade list or position series do to a
prop-firm account under these rules, at these costs?".

RESEARCH ONLY - not trading advice. Nothing in propkit places, simulates sending, or prepares
orders, and nothing in it talks to a broker or the network.

Conventions shared by every module (see the module docstrings for details):
  * time: int64 UTC epoch seconds; a bar's `time` is its OPEN instant;
  * prices: USD per troy ounce, BID side unless a name says otherwise (ask = bid + spread);
  * size: ounces ("units"); 1 lot = CostModel.lot_size_oz ounces (default 100, check the broker spec);
  * money: USD; ledger amounts are signed (negative = paid), cost parameters are positive = you pay;
  * the prop day is the CE(S)T calendar date (Europe/Prague, EU DST rule), see propkit.calendar.

The usual pipeline (see propkit/README.md and `python -m propkit --help`):

    bars = load_bars("XAUUSD_H1.parquet")                           # BARS
    equity, trades = equity_from_trades(bars, raw_trades, 100_000, CostModel())
    rules = preset("ftmo-1step", 100_000)
    report = analyse(bars, trades, equity, rules, CostModel())       # dict; render_markdown(report) -> text

Modules: calendar (prop day, DST, sessions), costs (CostModel), bars (load and validate BARS), rules
(PropRules, presets), evaluator (one path under the rules), bootstrap (Monte Carlo of challenges, largest
size), equity (EQUITY from TRADES or POSITIONS), adapters (CSV files, AlphaMaster positions), stats (Sharpe,
PSR, DSR, drawdown), indicators, pullback (zeno's rule as a spec), stress, report, cli, selftest.
"""
from __future__ import annotations

__version__ = "0.1.0"  # defined before the submodule imports: propkit.report reads it

from propkit import calendar, costs, bars, rules, evaluator, bootstrap, equity, adapters  # noqa: E402
from propkit import stats, indicators, pullback, stress, report  # noqa: E402
from propkit.adapters import (alphamaster_log_pnl, positions_from_alphamaster, read_positions_csv,  # noqa: E402
                              read_trades_csv, trades_from_positions, trades_summary, validate_positions,
                              validate_trades, write_equity_csv, write_positions_csv, write_trades_csv)
from propkit.bars import (LockedPathError, bars_summary, check_not_locked, is_locked_path, load_bars,  # noqa: E402
                          synthetic_bars, validate_bars)
from propkit.bootstrap import (BootstrapResult, MaxSizeResult, bootstrap_challenges, build_day_units,  # noqa: E402
                               history_uncertainty, max_size)
from propkit.calendar import day_start_utc, prop_day, session_mask  # noqa: E402
from propkit.costs import CostModel  # noqa: E402
from propkit.equity import equity_from_positions, equity_from_trades  # noqa: E402
from propkit.evaluator import PathResult, evaluate_path, r_summary  # noqa: E402
from propkit.pullback import PullbackSpec, generate_trades, generate_trades_detailed  # noqa: E402
from propkit.report import analyse, render_markdown  # noqa: E402
from propkit.rules import PRESET_NAMES, PropRules, ftmo_1step, ftmo_2step, preset  # noqa: E402
from propkit.stats import (daily_returns_from_equity, drawdown_stats, dsr, expected_shortfall,  # noqa: E402
                           min_track_record_length, psr, sharpe_stats)
from propkit.stress import run_stress  # noqa: E402


def run_selftest(out=print) -> tuple[int, int]:
    """Run every known-answer gate (propkit.selftest); prints one PASS/FAIL line per gate and returns
    (n_passed, n_failed)."""
    from propkit.selftest import run_selftest as _run
    return _run(out=out)


__all__ = [
    "__version__",
    # modules
    "calendar", "costs", "bars", "rules", "evaluator", "bootstrap", "equity", "adapters", "stats", "indicators",
    "pullback", "stress", "report",
    # data in
    "load_bars", "validate_bars", "bars_summary", "synthetic_bars", "LockedPathError", "is_locked_path",
    "check_not_locked", "read_trades_csv", "read_positions_csv", "validate_trades", "validate_positions",
    "positions_from_alphamaster", "alphamaster_log_pnl", "trades_from_positions", "trades_summary",
    # costs, rules, calendar
    "CostModel", "PropRules", "preset", "ftmo_1step", "ftmo_2step", "PRESET_NAMES", "prop_day", "day_start_utc",
    "session_mask",
    # account path and evaluation
    "equity_from_trades", "equity_from_positions", "evaluate_path", "PathResult", "r_summary",
    "build_day_units", "bootstrap_challenges", "max_size", "history_uncertainty", "BootstrapResult", "MaxSizeResult",
    # statistics and stress
    "sharpe_stats", "psr", "dsr", "min_track_record_length", "daily_returns_from_equity", "drawdown_stats",
    "expected_shortfall", "run_stress",
    # pullback rule
    "PullbackSpec", "generate_trades", "generate_trades_detailed",
    # output
    "analyse", "render_markdown", "write_trades_csv", "write_positions_csv", "write_equity_csv", "run_selftest",
]
