"""propkit/report.py - one run's full analysis as a JSON-serialisable dict and as Markdown. Research only.

analyse(...) runs every step on one EQUITY path (and its TRADES): the historical path under the rules
(propkit.evaluator), the day-block and week-block bootstrap (propkit.bootstrap), the largest size within
the breach budget (max_size), the statistics (propkit.stats) and the stress tests (propkit.stress). It
returns a plain dict (numbers, text, lists; NaN and infinity become None) that json.dumps can write;
render_markdown(report) turns that dict into the Markdown report. Every report starts with the line
"RESEARCH ONLY - not trading advice".

Units: money in USD; probabilities are fractions (0.05 = 5%) with their Monte Carlo standard error;
fields ending in _pct are PERCENT of the initial capital (3.0 = 3%); SR is per period unless it says
annualised. Statistics from prop-day returns state the day count used to annualise (the observed number
of prop days with bars per year).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import propkit
from propkit import adapters
from propkit import bars as bars_mod
from propkit import bootstrap as boot
from propkit import calendar
from propkit import stats as st
from propkit import stress as stress_mod
from propkit.costs import CostModel
from propkit.evaluator import evaluate_path, r_summary
from propkit.rules import PropRules

HEADER = "RESEARCH ONLY - not trading advice"
DEFAULT_DD_SIMS = stress_mod.DEFAULT_DD_SIMS
UNKNOWN = "[U]"
ASSUMED = "[ASSUMPTION]"
SR_VAR_NOTE = ("var_sr (the variance of the trials' SR estimates) was not given; the default is 1/(n-1) per prop "
               "day [ASSUMPTION], the sampling variance of one SR estimate when the true SR is 0 and returns are "
               "normal. It is NOT a bound in either direction: variants that differ in true skill, or fat-tailed "
               "returns, make the real variance larger (DSR lower); variants that are small changes of one rule "
               "have strongly correlated estimates, so their spread is smaller (DSR higher). Measure it from the "
               "trials' own SRs (propkit.stats.trials_sr_variance) and pass it with --sr-var; for correlated "
               "trials also reduce N (propkit.stats.effective_trials).")


# ---------------------------------------------------------------------------------------
# small helpers

def file_sha256(path) -> str | None:
    """SHA-256 hex digest of a file (None when path is None). Locked-holdout paths are refused."""
    if path is None:
        return None
    bars_mod.check_not_locked(path, what="file")
    h = hashlib.sha256()
    with open(Path(str(path)).expanduser(), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(obj):
    """A JSON-safe copy: numpy scalars and arrays to Python, NaN and infinity to None, DataFrames to lists of
    row dicts, tuples to lists, dict keys to text."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return [clean(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return str(obj)


def _ascii(text: str) -> str:
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def _fmt(x, nd: int = 4, na: str = "n/a") -> str:
    if x is None:
        return na
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{f:.{nd}f}" if math.isfinite(f) else na


def _usd(x, na: str = "n/a") -> str:
    return na if x is None else f"{float(x):,.2f}"


def _pse(p, se, n_sims=None) -> str:
    """'p +- MC error', or the rule-of-three bound at p = 0 or 1 when n_sims is given (boot.prob_text)."""
    return boot.prob_text(p, se, n_sims)


# ---------------------------------------------------------------------------------------
# sections

def data_section(bars: pd.DataFrame, path=None, cost_model: CostModel | None = None) -> dict[str, Any]:
    """Data description: file, sha256, number of bars, first/last bar open (UTC), bar size, gaps, spread
    median / p90 (USD/oz, None without a spread column) and the spread actually used by the cost model."""
    s = bars_mod.bars_summary(bars)
    s.update(file=None if path is None else str(path), sha256=file_sha256(path),
             bar_minutes=s["bar_seconds"] / 60.0,
             spread_used=None if cost_model is None else cost_model.spread_source_used(bars))
    return s


def _sharpe_or_reason(values, periods_per_year) -> tuple[dict | None, str | None]:
    try:
        return st.sharpe_stats(values, periods_per_year).to_dict(), None
    except ValueError as e:
        return None, str(e)


def stats_section(equity: pd.DataFrame, trades: pd.DataFrame | None, C0: float, n_trials: int | None = None,
                  sr_var: float | None = None, es_alpha: float = 0.05) -> dict[str, Any]:
    """Statistics of one run (propkit.stats; see that module for every formula).

    per_trade: E[R] +- SE and the other R-multiple figures (evaluator.r_summary), and the Sharpe statistics
    of the per-trade R multiples (or of pnl_usd / C0 when no trade has a risk_usd), not annualised.
    per_day: Sharpe statistics of the simple prop-day returns (daily_returns_from_equity), annualised with
    q = the observed number of prop days with bars per year (stated as periods_per_year; sqrt(q) assumes
    IID returns), PSR against 0, the minimum track record length at 95% (in prop days), and the DSR when
    n_trials is given (sr_var default: 1/(n - 1), see SR_VAR_NOTE). drawdown: drawdown_stats (max
    drawdown intrabar and close-to-close, longest underwater spell, expected shortfall of the daily
    returns at es_alpha, worst day).
    """
    out: dict[str, Any] = {"n_trades": 0 if trades is None else int(len(trades))}
    rs = r_summary(trades) if trades is not None else None
    out["r_summary"] = rs
    per_trade, why = None, "no trades"
    basis = None
    if trades is not None and len(trades):
        risk = trades["risk_usd"].to_numpy(dtype=np.float64) if "risk_usd" in trades.columns else np.zeros(0)
        pnl = trades["pnl_usd"].to_numpy(dtype=np.float64)
        ok = np.isfinite(risk) & (risk > 0) if risk.size else np.zeros(pnl.size, dtype=bool)
        if ok.sum() >= 2:
            per_trade, why = _sharpe_or_reason(pnl[ok] / risk[ok], None)
            basis = "R multiples (pnl_usd / risk_usd)"
        else:
            per_trade, why = _sharpe_or_reason(pnl / float(C0), None)
            basis = "pnl_usd / C0 per trade"
    out["per_trade"] = {"basis": basis, "sharpe": per_trade, "unavailable": None if per_trade else why}

    try:
        daily = st.daily_returns_from_equity(equity, C0)
    except ValueError as e:                     # e.g. the account went to zero: returns are undefined after it
        out["per_day"] = {"n_days": None, "periods_per_year": None, "sharpe": None, "unavailable": str(e),
                          "min_track_record_days_95": None, "dsr": None}
        out["drawdown"] = {"unavailable": str(e)}
        out["trials_note"] = st.TRIALS_NOTE
        return out
    n_days = int(len(daily))
    q = st.observed_periods_per_year(daily["day"].to_numpy()) if n_days >= 2 else None
    day_sharpe, why = _sharpe_or_reason(daily["ret"].to_numpy(), q) if n_days >= 2 else (None, "fewer than 2 days")
    per_day: dict[str, Any] = {"n_days": n_days, "periods_per_year": q,
                               "periods_per_year_note": "observed prop days with bars per year (IID assumed for "
                                                        "annualising, CLAUDE.md A9)",
                               "sharpe": day_sharpe, "unavailable": None if day_sharpe else why,
                               "min_track_record_days_95": None, "dsr": None}
    if day_sharpe:
        sr, skew, kurt = day_sharpe["sr"], day_sharpe["skew"], day_sharpe["kurt"]
        try:
            per_day["min_track_record_days_95"] = st.min_track_record_length(sr, 0.0, skew, kurt, 0.95)
        except ValueError:
            per_day["min_track_record_days_95"] = None
        if n_trials is not None:
            var = float(sr_var) if sr_var is not None else 1.0 / (n_days - 1)
            details = st.dsr_details(sr, n_days, skew, kurt, int(n_trials), var)
            details["var_sr_source"] = "given (--sr-var)" if sr_var is not None else "default 1/(n-1) [ASSUMPTION]"
            details["note"] = st.TRIALS_NOTE if sr_var is not None else SR_VAR_NOTE + " " + st.TRIALS_NOTE
            per_day["dsr"] = details
    out["per_day"] = per_day
    out["drawdown"] = st.drawdown_stats(equity, C0, alpha=es_alpha)
    out["trials_note"] = st.TRIALS_NOTE
    return out


def strategy_card(report: dict[str, Any]) -> list[dict[str, str]]:
    """STRATEGY CARD skeleton (CLAUDE.md B fields as far as propkit can fill them): a list of rows
    {field, value}. Unknowns are marked [U] (fill by hand) and assumed values [ASSUMPTION]."""
    data, inp, rules = report["data"], report["input"], report["rules"]
    st_ = report["stats"]
    spec = inp.get("spec")
    rs = st_.get("r_summary") or {}
    day = (st_.get("per_day") or {}).get("sharpe") or {}
    q = (st_.get("per_day") or {}).get("periods_per_year")
    days = report.get("bootstrap_days") or {}
    n_days_sims = days.get("n_sims")
    ms = report.get("max_size") or {}
    dsr = (st_.get("per_day") or {}).get("dsr")
    dd = st_.get("drawdown") or {}

    def from_spec(keys: tuple[str, ...]) -> str:
        if not spec:
            return UNKNOWN
        text = ", ".join(f"{k}={spec.get(k)}" for k in keys)
        return text + (" [ASSUMPTION: PLACEHOLDER spec]" if spec.get("placeholder") else "")

    name = (spec or {}).get("name") or UNKNOWN
    rows = [
        ("name", name if not (spec or {}).get("placeholder") else f"{name} [ASSUMPTION: PLACEHOLDER]"),
        ("status", "research (not traded) " + UNKNOWN),
        ("instrument", f"{Path(data['file']).stem if data.get('file') else UNKNOWN} (XAUUSD CFD assumed) {ASSUMED}"),
        ("timeframe", f"{data['bar_minutes']:g}-minute bars"),
        ("hypothesis (one trader sentence)", UNKNOWN),
        ("entry rule", from_spec(("direction", "trend_ema_period", "pullback_mode", "trigger"))
         if spec else f"from {inp.get('kind')} {UNKNOWN}"),
        ("stop and exit", from_spec(("stop_mode", "exit_mode", "target_r"))),
        ("sizing", from_spec(("risk_pct",)) if spec else
         f"{inp.get('size_mode') or 'as in the trade list'} {inp.get('size') or ''}".strip()),
        ("sessions and filters", from_spec(("sessions", "session_hours_utc", "max_spread_usd",
                                            "max_trades_per_day"))),
        ("costs", f"{report['costs']['summary']} {ASSUMED}"),
        ("data", f"{data.get('file')} {data['first_time_utc']} .. {data['last_time_utc']}, {data['n_bars']} bars, "
                 f"sha256 {str(data.get('sha256'))[:12]}"),
        ("sample", f"{st_['n_trades']} trades, {(st_.get('per_day') or {}).get('n_days')} prop days"),
        ("E[R] +- SE", f"{_fmt(rs.get('mean_r'), 3)} +- {_fmt(rs.get('se_r'), 3)}" if rs else UNKNOWN),
        ("SR per prop day +- SE", f"{_fmt(day.get('sr'))} +- {_fmt(day.get('se_sr'))}; annualised "
                                  f"{_fmt(day.get('sr_annual'), 2)} at q = {_fmt(q, 1)}" if day else UNKNOWN),
        ("DSR", f"{_fmt(dsr['dsr'])} (N = {dsr['n_trials']})" if dsr else f"{UNKNOWN} (number of trials not given)"),
        ("max drawdown", f"{_usd(dd.get('max_dd_usd'))} USD = {_fmt(dd.get('max_dd_pct'), 2)}% of C0 (intrabar)"),
        ("prop rules", f"{rules['name']} ({rules.get('notes') or 'recheck the firm rules'}) {ASSUMED}"),
        ("historical path", report["path"]["status"]),
        ("P(pass) / P(daily breach)", f"{_pse(days.get('p_pass'), days.get('se_pass'), n_days_sims)} / "
                                      f"{_pse(days.get('p_breach_daily'), days.get('se_breach_daily'), n_days_sims)}"),
        ("max size multiplier (daily breach <= alpha)", f"{_fmt(ms.get('multiplier'), 3)} ({ms.get('note')})"
         if ms else UNKNOWN),
        ("number of variants tried (N)", str(dsr["n_trials"]) if dsr else UNKNOWN),
        ("locked holdout", "not touched by propkit; one pre-registered look only " + UNKNOWN),
        ("kill criteria", UNKNOWN),
        ("broker checks", f"lot {report['costs']['model']['lot_size_oz']:g} oz, swap, triple-swap day, commission "
                          f"{ASSUMED} - verify on the broker's contract specification"),
    ]
    return [{"field": f, "value": _ascii(v)} for f, v in rows]


def cost_summary(cost_model: CostModel) -> str:
    """One plain line describing a cost model (units in the text)."""
    cm = cost_model
    if cm.is_flat:
        fill = f"flat {cm.flat_rate_per_side:g} of notional per fill (no spread)"
    else:
        sp = "bar spread" if cm.spread_source == "bar" else f"fixed spread {cm.fixed_spread:g}"
        fill = (f"{sp} x {cm.spread_multiplier:g}, markup {cm.markup_per_side:g} + slippage "
                f"{cm.slippage_per_side:g} USD/oz per side, commission {cm.commission_per_lot_round_trip:g} USD "
                f"per {cm.lot_size_oz:g}-oz lot round trip")
    if cm.swap_enabled:
        unit = "%/yr" if cm.swap_unit == "pct_per_year" else "USD/lot/night"
        swap = (f"swap {cm.swap_long:g} long / {cm.swap_short:g} short {unit}, triple on "
                f"{'none' if cm.triple_swap_weekday is None else calendar.WEEKDAY_NAMES[cm.triple_swap_weekday]}")
    else:
        swap = "no swap"
    return f"{fill}; {swap}"


# ---------------------------------------------------------------------------------------
# the whole analysis

def bootstrap_warnings(blocks: dict[str, Any]) -> list[str]:
    """Plain-text reasons to distrust the block bootstrap, from DayUnits.block_summary() (empty list when
    there are none): few flat-to-flat day blocks (fewer than bootstrap.FEW_BLOCKS), or many prop days that
    start with a position open (they cannot start a block, so long holds are resampled as one piece)."""
    out = []
    n_blocks, n_days = blocks.get("n_blocks_days"), blocks.get("n_days")
    share = blocks.get("share_open_at_start") or 0.0
    if n_blocks is not None and n_blocks < boot.FEW_BLOCKS:
        out.append(f"only {n_blocks} flat-to-flat day blocks in {n_days} prop days: the bootstrap re-uses very few "
                   "starting points and its probabilities are rough (see the history-uncertainty range)")
    if share >= 0.5:
        out.append(f"{share * 100:.0f}% of the prop days start with a position open: positions held over midnight "
                   "join days into long blocks, so the bootstrap has little to resample")
    return out


def analyse(bars: pd.DataFrame, trades: pd.DataFrame | None, equity: pd.DataFrame, rules: PropRules,
            cost_model: CostModel, *, bars_path=None, input_info: dict[str, Any] | None = None,
            n_sims: int = boot.DEFAULT_N_SIMS, seed: int = boot.DEFAULT_SEED, alpha: float = 0.05,
            horizon_days=boot.DEFAULT_HORIZON_DAYS, n_trials: int | None = None, sr_var: float | None = None,
            run_stress: bool = True, dd_sims: int = DEFAULT_DD_SIMS, progress=None,
            horizon_unit: str = boot.DEFAULT_HORIZON_UNIT,
            history_reps: int = boot.DEFAULT_HISTORY_REPS) -> dict[str, Any]:
    """Every analysis step for one run; returns the JSON-serialisable report dict.

    bars: BARS; trades: TRADES after equity_from_trades (or None); equity: its EQUITY (USD; C0 =
    rules.initial_capital); rules: PropRules; cost_model: the cost model the path was built with;
    bars_path: the bar file (for the sha256); input_info: what the trades came from (kind, path, ...);
    n_sims / seed / horizon_days / horizon_unit: the bootstraps (days and weeks), max_size and the stress
    scenarios (horizon_unit "trading": the horizon counts days with a trade entry, "market": every prop
    day with bars); alpha: the breach budget of max_size (a fraction); n_trials / sr_var: for the DSR;
    run_stress: run propkit.stress; dd_sims: orders in the trade-order drawdown bootstrap; history_reps:
    replicates of the outer bootstrap of the history (bootstrap.history_uncertainty; 0 = skip it);
    progress: optional callable(str) for stage messages.

    The bootstrap's se_* fields are Monte Carlo error only (simulation noise). The uncertainty from having
    only this history is in "history_uncertainty" (5-95% range over the replicates); "warnings" lists
    reasons to distrust the bootstrap (few flat-to-flat blocks, many days starting with a position open).
    """
    if isinstance(history_reps, (bool, np.bool_)) or not isinstance(history_reps, (int, np.integer)) \
            or history_reps < 0 or history_reps == 1:
        raise ValueError(f"history_reps must be 0 (skip) or a whole number >= 2, got {history_reps!r}")
    horizon_unit = boot._check_unit(horizon_unit)
    say = progress or (lambda text: None)
    c0 = rules.initial_capital
    timing: dict[str, float] = {}

    def timed(name, fn):
        t0 = time.perf_counter()
        say(f"{name} ...")
        value = fn()
        timing[name] = round(time.perf_counter() - t0, 3)
        return value

    path = timed("historical path", lambda: evaluate_path(equity, trades, rules))
    units = timed("day units", lambda: boot.build_day_units(equity, c0, trades))
    kw = dict(n_sims=n_sims, seed=seed, horizon_days=horizon_days, units=units, horizon_unit=horizon_unit)
    days = timed("bootstrap days", lambda: boot.bootstrap_challenges(None, rules, mode="days", **kw))
    weeks = timed("bootstrap weeks", lambda: boot.bootstrap_challenges(None, rules, mode="weeks", **kw))
    ms = timed("max size", lambda: boot.max_size(None, rules, mode="days", alpha=alpha, **kw))
    hist = None
    if history_reps:
        hist = timed("history uncertainty", lambda: boot.history_uncertainty(
            None, rules, n_reps=int(history_reps), seed=seed, horizon_days=horizon_days, alpha=alpha, units=units,
            horizon_unit=horizon_unit, size_start=ms.multiplier))
    blocks = units.block_summary()
    m_daily = ms.multiplier if ms.multiplier is not None and math.isfinite(ms.multiplier) else None
    blocks["days_breaching_alone_at_max_size"] = (None if m_daily is None
                                                  else boot.days_breaching_alone(units, rules, m_daily))
    warnings = bootstrap_warnings(blocks)
    stats = timed("statistics", lambda: stats_section(equity, trades, c0, n_trials, sr_var))
    stress = None
    if run_stress and trades is not None:
        stress = timed("stress tests", lambda: stress_mod.run_stress(bars, trades, c0, cost_model, rules,
                                                                       n_sims=n_sims, seed=seed,
                                                                       horizon_days=horizon_days, dd_sims=dd_sims,
                                                                       horizon_unit=horizon_unit))
    report = {
        "header": HEADER,
        "propkit_version": propkit.__version__,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "data": data_section(bars, bars_path, cost_model),
        "input": input_info or {"kind": "trades"},
        "rules": {**rules.to_dict(), "describe": rules.describe()},
        "costs": {"model": cost_model.to_dict(), "summary": cost_summary(cost_model),
                  "spread_used": cost_model.spread_source_used(bars)},
        "trades_summary": adapters.trades_summary(trades) if trades is not None else None,
        "path": path.to_dict(include_days=True),
        "bootstrap_days": days.to_dict(),
        "bootstrap_weeks": weeks.to_dict(),
        "max_size": ms.to_dict(),
        "blocks": blocks,
        "history_uncertainty": None if hist is None else {k: v for k, v in hist.items() if k != "values"},
        "warnings": warnings,
        "stats": stats,
        "stress": None if stress is None else {
            "scenarios": stress["scenarios"], "sessions": stress["sessions"], "volatility": stress["volatility"],
            "drawdown_order": stress["drawdown_order"], "notes": stress["notes"]},
        "settings": {"n_sims": int(n_sims), "seed": int(seed), "alpha": float(alpha),
                     "horizon_days": None if days.horizon_days is None else int(days.horizon_days),
                     "horizon_unit": horizon_unit, "history_reps": int(history_reps),
                     "n_trials": n_trials, "sr_var": sr_var, "dd_sims": int(dd_sims)},
        "timing_seconds": timing,
    }
    report = clean(report)
    report["strategy_card"] = strategy_card(report)
    return report


# ---------------------------------------------------------------------------------------
# Markdown

def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_ascii(str(c)) for c in r) + " |")
    return out


def _boot_rows(b: dict[str, Any]) -> list[list[str]]:
    q = b.get("days_to_target_q") or {}
    n = b.get("n_sims")
    return [["P(pass)", _pse(b["p_pass"], b["se_pass"], n)],
            ["P(daily-loss breach)", _pse(b["p_breach_daily"], b["se_breach_daily"], n)],
            ["P(max-loss breach)", _pse(b["p_breach_max"], b["se_breach_max"], n)],
            ["P(timeout, no event in the horizon)", _pse(b["p_timeout"], b["se_timeout"], n)],
            ["P(best-day rule unmet when the target was first met)",
             _pse(b["p_best_day_unmet_at_target"], b["se_best_day_unmet_at_target"], n)],
            ["P(min trading days unmet when the target was first met)",
             _pse(b["p_min_days_unmet_at_target"], b["se_min_days_unmet_at_target"], n)],
            ["flat-to-flat blocks resampled", b.get("n_blocks", "n/a")],
            ["market days to target p10/p25/p50/p75/p90",
             "/".join(_fmt(q.get(f"p{k}"), 1) for k in boot.QUANTILES) if q else "n/a (no pass)"]]


def render_markdown(report: dict[str, Any]) -> str:
    """The report dict (from analyse, or read back from report.json) as Markdown text (ASCII only)."""
    r = report
    data, inp, path, rules = r["data"], r["input"], r["path"], r["rules"]
    days, weeks, ms, stats = r["bootstrap_days"], r["bootstrap_weeks"], r["max_size"], r["stats"]
    s = r["settings"]
    hz = boot.horizon_text(s["horizon_days"], s.get("horizon_unit", "market"))
    n_sims = s["n_sims"]
    hu = r.get("history_uncertainty")
    blocks = r.get("blocks") or {}

    def hrange(key: str, nd: int = 4) -> str:
        q = (hu or {}).get(key)
        return f"{_fmt(q['p5'], nd)} .. {_fmt(q['p95'], nd)}" if q else "n/a"
    rs = stats.get("r_summary") or {}
    pdx = stats.get("per_day") or {}
    dsh = pdx.get("sharpe") or {}
    dd = stats.get("drawdown") or {}
    L: list[str] = [HEADER, "", "# propkit report", ""]
    L.append(f"Generated {r['generated_utc']} by propkit {r['propkit_version']}. The numbers describe this data "
             "file and this trade list only; they say nothing certain about future data.")
    if (inp.get("spec") or {}).get("placeholder"):
        L += ["", "**WARNING: the pullback spec is a PLACEHOLDER, not zeno's rule. Do not read these results as "
                  "evidence about the real rule.**"]
    L += ["", "## Summary", ""]
    end = path.get("end_time_utc")
    L += _table(["item", "value"], [
        ["rules", f"{rules['name']}, initial capital {_usd(rules['initial_capital'])} USD"],
        ["historical path", f"{path['status'].upper()} (end {end})"],
        [f"P(pass), day blocks, {hz} (+- MC error; 5-95% over resampled histories)",
         f"{_pse(days['p_pass'], days['se_pass'], n_sims)}; {hrange('p_pass')}"],
        ["P(daily-loss breach)", f"{_pse(days['p_breach_daily'], days['se_breach_daily'], n_sims)}; "
                                 f"{hrange('p_breach_daily')}"],
        ["P(max-loss breach)", f"{_pse(days['p_breach_max'], days['se_breach_max'], n_sims)}; "
                               f"{hrange('p_breach_max')}"],
        ["P(timeout)", _pse(days["p_timeout"], days["se_timeout"], n_sims)],
        [f"largest size multiplier with P(daily breach) <= {s['alpha']:g}",
         f"{_fmt(ms.get('multiplier'), 3)} ({ms.get('note')}); {hrange('max_size_multiplier', 3)}"],
        ["trades / E[R] +- SE", f"{stats['n_trades']} / " + (f"{_fmt(rs.get('mean_r'), 3)} +- "
                                                         f"{_fmt(rs.get('se_r'), 3, 'n/a (one trade)')}"
                                                         if rs else "n/a (no risk_usd)")],
        ["SR per prop day +- SE (annualised)",
         f"{_fmt(dsh.get('sr'))} +- {_fmt(dsh.get('se_sr'))} ({_fmt(dsh.get('sr_annual'), 2)} at "
         f"{_fmt(pdx.get('periods_per_year'), 1)} days/yr)" if dsh else "n/a"],
        ["max drawdown over the whole file (intrabar, from the running peak)",
         f"{_usd(dd.get('max_dd_usd'))} USD = {_fmt(dd.get('max_dd_pct'), 2)}% of C0"],
    ])
    L += ["", "+- is the Monte Carlo error (simulation noise only). The range after it is the 5-95% spread over "
              "resampled histories (history uncertainty)" + ("" if hu else ": not computed (--history-reps 0)")
          + ". At 0 or 1 the bound is the 95% rule of three (3 / simulations)."]
    for w in r.get("warnings") or []:
        L += ["", f"**WARNING: {w}.**"]
    L += ["", "## Data", ""]
    L += _table(["item", "value"], [
        ["file", data.get("file")], ["sha256", data.get("sha256")], ["bars", data["n_bars"]],
        ["first / last bar open", f"{data['first_time_utc']} / {data['last_time_utc']}"],
        ["bar size", f"{data['bar_minutes']:g} minutes ({data['n_gaps']} gaps longer than one bar)"],
        ["spread median / p90 (USD/oz)", f"{_fmt(data.get('spread_median'), 3)} / {_fmt(data.get('spread_p90'), 3)}"],
        ["spread used for fills and ask marks", data.get("spread_used")],
    ])
    L += ["", "## Input", ""]
    L += _table(["item", "value"], [[k, v] for k, v in inp.items() if k not in ("spec",)])
    ts_ = r.get("trades_summary") or {}
    if ts_:
        L += ["", f"Trades: {ts_['n_trades']} ({ts_['n_long']} long, {ts_['n_short']} short, {ts_['n_wins']} winners); "
                  f"gross {_usd(ts_['gross_usd'])}, commission {_usd(ts_['commission_usd'])} (paid), swap "
                  f"{_usd(ts_['swap_usd'])} (negative = paid), net {_usd(ts_['pnl_usd'])} USD."]
    L += ["", "## Rules", ""] + [f"    {line}" for line in rules["describe"]]
    cmult = inp.get("cost_mult")
    L += ["", "## Costs", "", r["costs"]["summary"] + "."
          + ("" if cmult in (None, 1, 1.0) else f" Every cost x {cmult:g} (--cost-mult)"
             + ("; fills of the trade list repriced." if inp.get("fills_repriced_for_cost_mult") else ".")),
          "Defaults marked [ASSUMPTION] in propkit.costs (swap rates, triple-swap day, commission, lot size) must "
          "be checked on the broker's contract specification."]
    L += ["", "## Historical path", ""]
    L += _table(["item", "value"], [
        ["status", path["status"]],
        ["first breach", "none" if not path.get("breach") else
         f"{path['breach']['kind']} on {path['breach']['date']} at {path['breach']['time_utc']}: equity "
         f"{_usd(path['breach']['equity_worst'])} below the floor {_usd(path['breach']['floor'])}"],
        ["passed at", path.get("pass_time_utc") or "not passed"],
        ["trading days / calendar days", f"{path['trading_days']} / {path['calendar_days']}"],
        ["worst daily drawdown", f"{_usd(path['worst_daily_dd_usd'])} USD = {_fmt(path['worst_daily_dd_pct'], 2)}% "
                                 f"of C0 on {path['worst_daily_dd_date']}"],
        ["max drawdown until the path ended", f"{_usd(path['max_dd_usd'])} USD = {_fmt(path['max_dd_pct'], 2)}% of C0"],
        ["final balance / equity", f"{_usd(path['final_balance'])} / {_usd(path['final_equity'])}"],
    ])
    L += ["", f"## Bootstrap of challenges ({s['n_sims']} simulations, seed {s['seed']}, horizon {hz})", ""]
    L += ["Flat-to-flat blocks of the history are drawn with replacement and chained: a day block runs from a "
          "prop day that starts with no position open to the next such day (a position held over midnight "
          "keeps its days together), a week block from a flat market-week start to the next. Every challenge "
          "starts flat, and every rule is applied bar by bar as on the historical path. The horizon counts "
          + ("days with a trade entry (FTMO trading days); days without one still run and can breach."
             if s.get("horizon_unit") == "trading" else "every prop day with bars (market days).")
          + " +- is the Monte Carlo error (simulation noise only), not the uncertainty from the history.", ""]
    rows_d, rows_w = _boot_rows(days), _boot_rows(weeks)
    L += _table(["", "day blocks", "week blocks"], [[a[0], a[1], b[1]] for a, b in zip(rows_d, rows_w)])
    if blocks:
        L += ["", f"History: {blocks.get('n_days')} prop days, {_fmt((blocks.get('share_open_at_start') or 0) * 100, 1)}% "
                  f"of them start with a position open; {blocks.get('n_blocks_days')} day blocks (mean "
                  f"{_fmt(blocks.get('mean_block_days_days'), 2)}, longest {blocks.get('max_block_days_days')} days), "
                  f"{blocks.get('n_blocks_weeks')} week blocks (mean {_fmt(blocks.get('mean_block_days_weeks'), 2)}, "
                  f"longest {blocks.get('max_block_days_weeks')} days)."]
    L += ["", "### History uncertainty (outer bootstrap of the day blocks)", ""]
    if hu:
        L += [f"{hu['n_reps']} histories of {hu['n_blocks']} day blocks each, drawn with replacement from the "
              f"historical blocks; each run with {hu['n_sims']} simulations (seed {hu['seed']}). The 5-95% range "
              "shows how much the answers depend on having only this history (it also holds a little Monte "
              "Carlo noise). The size multiplier here comes from a quicker search around the size above (to "
              "about 4%, at most 64).", ""]
        L += _table(["quantity", "this history", "p5", "p50", "p95"], [
            [label, value] + ([_fmt(q[f"p{k}"], nd) for k in (5, 50, 95)] if q else ["n/a"] * 3)
            for label, value, q, nd in (
                ("P(pass)", _fmt(days["p_pass"]), hu.get("p_pass"), 4),
                ("P(daily-loss breach)", _fmt(days["p_breach_daily"]), hu.get("p_breach_daily"), 4),
                ("P(max-loss breach)", _fmt(days["p_breach_max"]), hu.get("p_breach_max"), 4),
                ("P(any breach)", _fmt(days["p_breach_any"]), hu.get("p_breach_any"), 4),
                (f"size multiplier, P(daily breach) <= {s['alpha']:g}", _fmt(ms.get("multiplier"), 3),
                 hu.get("max_size_multiplier"), 3))])
    else:
        L += ["Not computed (--history-reps 0)."]
    L += ["", f"## Largest size (common random numbers, alpha = {s['alpha']:g})", ""]
    at, at_any = ms.get("at") or {}, ms.get("at_any") or {}
    L += _table(["budget", "size multiplier", "note", "P(pass) there", "P(breach) there"], [
        ["P(daily breach) <= alpha", _fmt(ms.get("multiplier"), 3), ms.get("note"),
         _pse(at.get("p_pass"), at.get("se_pass"), at.get("n_sims")) if at else "n/a",
         _pse(at.get("p_breach_daily"), at.get("se_breach_daily"), at.get("n_sims")) if at else "n/a"],
        ["P(any breach) <= alpha", _fmt(ms.get("multiplier_any"), 3), ms.get("note_any"),
         _pse(at_any.get("p_pass"), at_any.get("se_pass"), at_any.get("n_sims")) if at_any else "n/a",
         _pse(at_any.get("p_breach_any"), at_any.get("se_breach_any"), at_any.get("n_sims")) if at_any else "n/a"],
    ])
    L += ["", "A multiplier scales every USD amount of the history (size and costs are linear in units; lot "
              "rounding ignored). 1.0 = the size of this run. It is the largest size whose ESTIMATED P stays "
              "within alpha on this one history; choosing the largest such size favours sizes whose P came out "
              "low by chance, so the true P there can exceed alpha (see the history-uncertainty range)."]
    alone = blocks.get("days_breaching_alone_at_max_size")
    if alone is not None:
        L += ["", f"Historical prop days whose own drawdown from the 00:00 balance would break the daily limit at "
                  f"that size: {alone}."]
    L += ["", "## Statistics", ""]
    pt = stats.get("per_trade") or {}
    psh = pt.get("sharpe") or {}
    dsr = pdx.get("dsr")
    mtrl = pdx.get("min_track_record_days_95")
    L += _table(["item", "value"], [
        ["trades", stats["n_trades"]],
        ["E[R] +- SE (R = pnl / risk_usd)", f"{_fmt(rs.get('mean_r'), 3)} +- {_fmt(rs.get('se_r'), 3, 'n/a (one trade)')} "
                                            f"(n = {rs.get('n')})"
         if rs else "n/a (no risk_usd)"],
        ["win rate / profit factor", f"{_fmt(rs.get('win_rate'), 3)} / {_fmt(rs.get('profit_factor'), 2)}" if rs else "n/a"],
        ["SR per trade +- SE", f"{_fmt(psh.get('sr'))} +- {_fmt(psh.get('se_sr'))} on {pt.get('basis')}" if psh
         else f"n/a ({pt.get('unavailable')})"],
        ["prop days with bars", pdx.get("n_days")],
        ["SR per prop day +- SE", f"{_fmt(dsh.get('sr'))} +- {_fmt(dsh.get('se_sr'))}" if dsh
         else f"n/a ({pdx.get('unavailable')})"],
        ["SR annualised +- SE", f"{_fmt(dsh.get('sr_annual'), 2)} +- {_fmt(dsh.get('se_sr_annual'), 2)} with q = "
                                f"{_fmt(pdx.get('periods_per_year'), 1)} prop days per year (observed; IID assumed)"
         if dsh else "n/a"],
        ["skew / kurtosis of daily returns (Pearson, normal = 3)", f"{_fmt(dsh.get('skew'), 2)} / {_fmt(dsh.get('kurt'), 2)}"
         if dsh else "n/a"],
        ["PSR vs 0", _fmt(dsh.get("psr_0"), 3) if dsh else "n/a"],
        ["min track record for PSR 95% (prop days)", "never (SR <= 0)" if dsh and mtrl is None else _fmt(mtrl, 0)],
        ["DSR", f"{_fmt(dsr['dsr'], 3)} (N = {dsr['n_trials']}, SR0 = {_fmt(dsr['sr0'])} per day, var_sr "
                f"{dsr['var_sr_source']})" if dsr else "not computed (pass --n-trials)"],
        ["max drawdown intrabar / close to close", f"{_fmt(dd.get('max_dd_pct'), 2)}% / {_fmt(dd.get('max_dd_close_pct'), 2)}% of C0"],
        ["longest underwater", "none (never below a previous peak)" if not dd.get("longest_underwater_days") else
         f"{dd.get('longest_underwater_days')} prop days ({dd.get('longest_underwater_start')} .. "
         f"{dd.get('longest_underwater_end')})"],
        [f"expected shortfall {_fmt(dd.get('es_alpha'), 2)} of daily returns", f"{_fmt(dd.get('es_ret'), 5)} "
                                                                               "(a return, negative = loss)"],
        ["worst prop day", f"{_fmt(dd.get('worst_day_ret'), 5)} on {dd.get('worst_day_date')}"],
    ])
    stress = r.get("stress")
    if stress:
        L += ["", "## Stress tests", ""]
        sc = stress["scenarios"]
        L += _table(["scenario", "trades", "net USD", "mean R", "PF", "path", "path max DD %", "P(pass)",
                     "P(daily breach)", "P(any breach)", "note"],
                    [[x["scenario"], x["n_trades"], _usd(x["net_pnl_usd"]), _fmt(x["mean_r"], 3),
                      _fmt(x["profit_factor"], 2), x["path_status"], _fmt(x["max_dd_pct"], 2), _fmt(x["p_pass"]),
                      _fmt(x["p_breach_daily"]), _fmt(x["p_breach_any"]), x["note"]] for x in sc])
        for title, key, extra in (("Sessions (entry time)", "sessions", []),
                                  ("Volatility terciles (ATR 14 of the last closed bar before the entry, USD/oz)",
                                   "volatility", ["atr_from", "atr_to"])):
            L += ["", f"### {title}", ""]
            L += _table(["group", "trades", "share", "net USD", "win rate", "mean R +- SE", "PF"] + extra,
                        [[g["group"], g["n_trades"], _fmt(g["share"], 3), _usd(g["net_pnl_usd"]),
                          _fmt(g["win_rate"], 3), f"{_fmt(g['mean_r'], 3)} +- {_fmt(g['se_r'], 3)}",
                          _fmt(g["profit_factor"], 2)] + [_fmt(g.get(c), 3) for c in extra]
                         for g in stress[key]])
        do = stress["drawdown_order"]
        L += ["", f"### Trade-order bootstrap of the closed-balance drawdown ({do['n_sims']} orders, seed {do['seed']})",
              "", f"Historical order: {_usd(do['hist_max_dd_usd'])} USD = {_fmt(do['hist_max_dd_pct'], 2)}% of C0; "
                  f"share of random orders at least as deep: {_fmt(do.get('share_sims_at_least_hist'), 3)}.", ""]
        L += _table(["quantile", "max DD USD", "max DD % of C0"],
                    [[x["quantile"], _usd(x["max_dd_usd"]), _fmt(x["max_dd_pct"], 2)] for x in do["table"]])
        L += [""] + [f"- {n}" for n in stress["notes"]]
    L += ["", "## STRATEGY CARD (skeleton)", "", "[U] = unknown, fill in by hand; [ASSUMPTION] = an assumed value "
          "to verify.", ""]
    L += _table(["field", "value"], [[x["field"], x["value"]] for x in r["strategy_card"]])
    L += ["", "## Notes", "",
          "- Prop days run 00:00-00:00 CE(S)T (Europe/Prague): 22:00 UTC in summer, 23:00 UTC in winter.",
          "- One spread per bar is used for every fill and ask-side mark in that bar (an approximation).",
          "- Bootstrap probabilities assume the future days look like these historical days. They do not cover "
          "regime changes, rule changes or execution problems. Their +- is simulation noise only.",
          f"- {HEADER}. Nothing here places or prepares orders.", ""]
    return _ascii("\n".join(L))
