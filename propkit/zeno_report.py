"""propkit/zeno_report.py - the zeno_pullback_v1 runner behind `python -m propkit zeno-v1 signals|run`.

RESEARCH ONLY - not trading advice. Nothing here places, prepares or simulates sending orders, and nothing
reads price data by itself: the command line passes the files (propkit.zeno_v1 loads them and enforces the
holdout lock and the locked-path refusal).

Two stages, in this order (the spec's change policy and gate G0):

  stage 1  signals_stage(prep, cell, ...) -> signals.csv (one row per trigger), decisions.csv (every setup
           event and trigger), g0_sample.csv (20 random eligible signals for zeno's chart check) and a
           counts-only signals report. No price after the entry, no P&L, no R multiple, no outcome: the G0
           signal check is not a "result" (spec: Change policy). A trigger is "eligible" when every check
           that does not need earlier trades passes (zeno_v1.screen): the daily limits, the cooldown and
           the one-open-position check need how and when earlier trades ended, so stage 1 leaves them out
           [SI-54]. The spread filter needs one declared cost cell (default evaluation, 10 USD/lot, S1,
           x1.5) [SI-27]; the entry and stop shown are the chart's (the data at costs x1, [SI-66]) and the
           cell's prices are in the *_at_costs columns. run_stage's caller checks that a G0 sample belongs
           to the data it judges (g0_sample_check, [SI-69]).
  stage 2  run_stage(prep, rules, rules_info, ...) -> the pre-registered grid (36 cells: variant {evaluation,
           master, master_fp} x commission x spread base x cost multiplier), metrics per side (long, short,
           combined) and period (all, each year, the four G3 periods), the prop evaluator (path, day-block
           bootstrap, largest size, history uncertainty) on the judging cell, its master twin and its master_fp
           twin, the judging cell again with the firm day at 00:00 UTC+3 (addendum A4), and the gates G0-G5
           plus the kill rule (G4 per addendum A5).

Addendum A (propkit/specs/zeno_pullback_v1_addendum_A.md, its sha256 in the report header) adds the Master run
"master_fp" (A1), the Master margin cap (A2), the verified FundingPips preset (A3), the "utc_plus3" day-boundary
sensitivity (A4) and G4 on the higher of the two P(daily-loss breach) values (A5). Neither Master run feeds
G1-G5; --master-primary records which one zeno chose as the primary Master result (A1).

The judging cell is (evaluation, commission 10 USD/lot round trip, the WORSE spread base, costs x1.5); the
worse base is the one with the lower combined net expectancy in R per position at those settings, a tie
goes to S2 [SI-28]. G5 and the kill rule are judged at costs x1 with the worse base at x1 [SI-52].

Units: money in USD; R multiples per position, net of every cost; probabilities are fractions with their
Monte Carlo error; Sharpe ratios per server day (17:00 New York to 17:00 New York, [SI-30]) unless they say
annualised; *_pct fields are percent of the initial capital. Readings of the spec are marked [SI-n]
(SPEC_ISSUES.md; propkit/METHODS.md section 7 lists them).
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

import propkit
from propkit import bootstrap as boot
from propkit import calendar
from propkit import report as report_mod
from propkit import stats as st
from propkit import zeno_v1 as zv
from propkit.equity import equity_from_trades
from propkit.evaluator import evaluate_path
from propkit.rules import PropRules

HEADER = report_mod.HEADER
SIGNALS_FILES = ("signals.csv", "decisions.csv", "g0_sample.csv", "signals_report.md", "signals_report.json")
RUN_FILES = ("report.md", "report.json", "gates.json", "trades.csv", "positions.csv", "decisions.csv",
             "grid.csv", "positions_all_cells.csv")
M1_DIFF_FILE = "m1_diff.csv"
SIDES = ("long", "short", "combined")
G3_PERIODS = (("2015-2017", "2015-01-01", "2017-12-31"), ("2018-2020", "2018-01-01", "2020-12-31"),
              ("2021-2023", "2021-01-01", "2023-12-31"), ("2024-2025-09-27", "2024-01-01", "2025-09-27"))
JUDGING_VARIANT = "evaluation"
JUDGING_COMMISSION = 10.0
JUDGING_COST_MULT = 1.5
KILL_COST_MULT = 1.0
G0_SAMPLE_SIZE = 20
G0_MIN_AGREE = 18
G1_MIN_POSITIONS = 100
G2_MIN_PSR = 0.95
G3_MIN_PERIODS = 3
G4_MAX_P_DAILY = 0.05
G4_MAX_P_MAX = 0.10
DEFAULT_SEED = 7
INFO_HORIZON_DAYS = 60
DAY_KEY = "ny_17"                      # the broker server day (D21) for daily returns [SI-30]
SENSITIVITY_DAY_BOUNDARY = "utc_plus3"  # addendum A4: the judging cell again with the firm day at 00:00 UTC+3
MASTER_PRIMARY_CHOICES = ("master_fp", "master")
DEFAULT_MASTER_PRIMARY = "master_fp"   # addendum A1: also when zeno has not answered before the first result
MASTER_ROLES = {   # addendum A1's wording, by the --master-primary choice
    "master_fp": {"master_fp": "the primary Master result", "master": "the literal-rule comparison",
                  "why": "zeno answered \"Widen it\" to the card in zeno's thread, or did not answer before the first "
                         "result was shown (addendum A1)"},
    "master": {"master": "the primary Master result", "master_fp": "a sensitivity run",
               "why": "zeno answered \"Keep 4 events\" to the card in zeno's thread (addendum A1)"}}
MASTER_NOT_GATED = "Neither Master run feeds G1-G5 (addendum A1)."
# addendum A3: FundingPips' Master rules that no run simulates; the report lists them as not modelled (values from
# research/FUNDINGPIPS_RULES_2026-10-08.md, section 3)
A3_NOT_MODELLED = (
    ("Master minimum reward", "1% of the account size (1,000 USD on 100,000 USD) [VP 1SF, RWD]"),
    ("Master Monthly 100% consistency rule", "no single trading day may account for more than 35% of the total "
     "profit; the Monthly cycle also needs 7 profitable days of 0.5% or more of the account size [VP 1SF]"),
    ("Striking System ([U] applicability)", "a closed and floating loss of 1% of the account (1,000 USD) on one "
     "trade idea gives a warning, and the 4th warning breaches the account [VP 1SF]; whether it applies to this "
     "account is [U]"))
# the FundingPips preset's last "unmodelled" item says its Master rules "are handled by the strategy's Master
# variant"; the preset stays as given and the report adds what the code does with each of them
MASTER_CLAIM = "handled by the strategy's Master variant"
UNMODELLED_MASTER_NOTE = (
    "[report note, not in the rules file: of these, the code models the news window (master_fp: FundingPips' "
    "restricted list, addendum A1; master: rule 9's four events, D20 and D23) and dynamic metals leverage (the "
    "margin cap, addendum A2), and never reaches the daily auto-close (rule 6 closes every trade at 16:30 New York, "
    "D17); it does not simulate the profit deduction or the Striking System (addendum A3)]")
STAGE1_CELL = zv.ZenoCell()            # evaluation / 10 / S1 / x1.5 [SI-27]
OUTCOME_KEYS = {"-1R": "n_outcome_minus_1r", "+1R(BE)": "n_outcome_be_plus_1r", "+3R": "n_outcome_plus_3r",
                "time-exit": "n_outcome_time_exit", "other": "n_outcome_other"}
STAGE1_NOT_CHECKED = ("Not checked in stage 1: the rule-11 daily limits (2 entries, 2 losing positions, -1.0% "
                      "of the day's start balance), the 15-minute same-direction cooldown and the one-open-"
                      "position rule. They need how and when earlier trades ended, which is a result, so stage 1 "
                      "leaves them out and sizes every signal on the initial capital. The run applies them: the "
                      "signals it enters are some of these eligible ones [SI-54]")

UNSCHEDULED_NOTE = ("D20 blocks entries from 30 min before an unscheduled FOMC row and D23 closes Master positions "
                    "10 min before it, as the spec says, although nobody knew of the statement at that time [SI-63]")

G0_REFUSAL = ("G0 first: run `python -m propkit zeno-v1 signals --m15-bid BID --m15-ask ASK --news CSV --out DIR`, "
              "check the 20 random signals in DIR/g0_sample.csv on a chart, and run `zeno-v1 run ... "
              "--g0-confirmed` only if you agree with at least 18 of 20. Reading results before the G0 check "
              "breaks the pre-registration (spec: Gates, G0). Nothing was read or written.")
G0_INSTRUCTIONS = (
    "G0 signal check (spec: Gates, G0). Open g0_sample.csv. For each of the 20 rows, open an XAUUSD M15 BID "
    "chart (UTC, or SGT with the _sgt columns) and check by eye that the setup is your rule: H and L, the "
    "pullback-low bar (pullback-high bar for a short), the trigger bar, the entry at the next bar's open and "
    "the stop (entry_price and stop_level are the chart's: the ask open for a long, the bid open for a short, "
    "the short's stop with the data's spread at that open). `zeno-v1 g0-charts --m15-bid BID --m15-ask ASK "
    "--sample DIR/g0_sample.csv` (add --news NEWS if this stage read a calendar other than the packaged one) "
    "draws every row from these files with each part of the rule marked, in DIR/g0_charts.html (open it in a "
    "browser). Write y or n in agree_y_n on EVERY row. Go on to "
    "`zeno-v1 run ... --g0-confirmed --g0-sample DIR/g0_sample.csv` only if you agree with at least 18 of 20; "
    "leave the file beside signals_report.json, because the run checks that every sampled signal is a signal "
    "of the data it judges. This file shows signals without P&L, so it is not a result (spec: Change policy).")
S2_DISCLOSURE = ("S2 was measured after the holdout lock. It carries cost information only, no returns. "
                 "A constant dollar spread overstates costs in years when gold was cheaper (2015-2019).")
AFTER_A_PASS = ("A forward demo of at least 50 trades or 3 months comes before the rule counts. The backtest is "
                "in-sample by construction, because zeno designed the rule while watching this market over these "
                "years.")
KILL_TEXT = ("If G2 fails at costs x1, the rule as written has no edge on this data. No tuning on the same data "
             "follows; any change is v2.")
CHANGE_POLICY = ("After the first result is shown, any change creates v2. N (the number of variants tried) goes to "
                 "2, and the DSR bar rises.")
DSR_CAVEAT = ("DSR with N = 1 understates the bar (spec, Reports): with one trial nothing is deflated, so the DSR "
              "equals PSR(SR > 0). zeno designed the rule while watching this market over these years, so the "
              "backtest is in-sample and the real number of trials is unknown and larger than 1.")
SLIPPAGE_NOTE = (f"stop slippage {zv.STOP_SLIPPAGE_USD:.2f} USD/oz on stop fills only [ASSUMPTION] (spec, Costs), "
                 "scaled by the cost multiplier; swap 0 (D17, asserted per trade)")
SPEC_READINGS = (
    ("SI-27", "stage 1 runs one declared cost cell (default evaluation / 10 / S1 / x1.5) for the spread filter "
              "behind eligible vs blocked"),
    ("SI-28", "the worse spread base = the lower combined net E[R] at (evaluation, 10, x1.5); a tie goes to S2"),
    ("SI-29", "G1 and D21 count positions (entries), not legs; legs are reported as n_legs"),
    ("SI-30", "daily Sharpe and PSR use the broker server day (17:00 New York); the CE(S)T and UTC days are in "
              "report.json"),
    ("SI-31", "G4 'before target' = no horizon (until pass or breach, cap 2,520 market days); 60 trading days "
              "printed beside it"),
    ("SI-32", "G4 needs the verified FundingPips rules with a profit target and a max-loss rule: the preset "
              "fundingpips-1step-flex (addendum A3, A5); under the placeholder preset (zeno's account facts only, "
              "every field [U]) or another firm's rules G4 is not evaluated"),
    ("SI-41", "+2R hit rate = positions that filled the partial / all; +4R after +2R = runners at +4R / partials; "
              "a 0.01-lot position that reaches +2R fills no partial (half rounds to 0), so it is not a +2R hit and "
              "is counted as n_tp1_reached_without_partial; losing streak = consecutive positions with net P&L < 0, "
              "ordered by final exit"),
    ("SI-42", "a position belongs to the server day of its trigger close, which is also its entry fill's (no fill "
              "at or after 16:30 New York of that day is accepted); that date decides its year and G3 period"),
    ("SI-52", "G5 and the kill rule use the worse spread base re-chosen at costs x1 by the same measure"),
    ("SI-53", "a G3 period without positions has no expectancy and counts as not above 0"),
    ("SI-54", "stage 1 leaves out the checks that need earlier trades (daily limits, cooldown, one open "
              "position): its status is eligible or a reason that needs no outcome; the G0 sample draws "
              "eligible signals"),
    ("SI-55", "under rules without a profit target the bootstrap runs 60 trading days and reports breach "
              "probabilities as an illustration; P(pass) and days to target are not defined"),
    ("SI-56", "trades per year = positions / (first to last trading day of the period, calendar days / 365.25)"),
    ("SI-57", "per-side daily returns come from the equity of that side's legs alone, from the same capital"),
    ("SI-61", "the trigger is the first close above the pullback-low bar's high; if it came before the setup armed, "
              "no later close of that pullback-low bar triggers (decisions: first_close_before_arming)"),
    ("SI-62", "D1's range starts 2015-01-01 00:00 UTC: earlier M15 bars are cut and counted, a later start is "
              "flagged"),
    ("SI-63", "unscheduled FOMC rows block from T - 30 min and close Master positions at T - 10 min as D20/D23 say; "
              "what that uses before T is counted (decisions: news_pre_unscheduled)"),
    ("SI-64", "D17's last bar before the break applies wherever no bar opens at 16:30 New York; each time exit is "
              "labelled 16:30_open, early_close_us_holiday or early_close_other_day (positions: time_exit_rule)"),
    ("SI-65", "the cost multiplier also scales the spread rule 10 compares with 10% of R, so x1.5 and x2 trade fewer "
              "triggers (grid: n_spread_blocked, positions_vs_x1)"),
    ("SI-66", "stage 1 shows the chart's entry and stop (the data at costs x1); the declared cell only decides "
              "eligible vs blocked (signals.csv: *_at_costs)"),
    ("SI-67", "the prop evaluator marks an open short at the cell's ask close and its worst at the cell's ask high "
              "(under S1 the data's spread moves inside a bar)"),
    ("SI-68", "the M1 second run replays an ambiguous bar only when its M1 bars reach the M15 low and high on the "
              "closing side; otherwise the M15 answer stands (meta: bars_unresolved_m1_mismatch)"),
    ("SI-69", "run --g0-sample refuses a sample whose rows are not eligible signals of the data it judges, with the "
              "same chart entry and stop; every row must be answered once one is, and >= 18 of 20 must be y"),
    ("SI-70", "Master: a fill at the open of the bar holding T - 10 min (only after a data gap) is closed at that "
              "same open; a fill after T - 10 min is kept"),
    ("A1", "master_fp = master plus FundingPips' restricted list: no entry when the trigger close or the entry fill "
           "lies in [T - 5 min, T_end + 5 min] or on the New York date of an event without a time; the D23 close "
           "for every restricted event with a time (a fill in the bar holding T - 10 min is closed at its own open, "
           "as SI-70)"),
    ("A2", "both Master runs: the tiered margin at the entry fill price must fit the closed balance; D13 lots are cut "
           "to the largest 0.01-lot size that fits, an entry that cannot hold 0.01 lot is blocked "
           "(margin_cap_below_lot_step); the evaluation run is counted at a flat 1:10 and 1:30, not capped"),
    ("A4-A5", "the judging cell's prop evaluator also runs with the firm day at 00:00 UTC+3 (21:00 UTC); G4 uses the "
              "higher P(daily-loss breach) of the two firm days (the default's on a tie) and the default firm "
              "day's P(max-loss breach)"),
)


# ---------------------------------------------------------------------------------------
# small helpers

def _day_of(text: str) -> int:
    """'YYYY-MM-DD' -> days since 1970-01-01."""
    return (_dt.date.fromisoformat(text) - _dt.date(1970, 1, 1)).days


def server_time_str(ts):
    """'YYYY-MM-DD HH:MM:SS server' text of the broker server clock (New York + 7 h) for UTC epoch seconds;
    scalar or array; a negative time (no value) gives ''."""
    arr = np.asarray(ts, dtype=np.int64)
    flat = arr.ravel()
    out = np.full(flat.shape, "", dtype=object)
    ok = flat >= 0
    if ok.any():
        t = flat[ok]
        local = t + (np.asarray(calendar.ny_offset_hours(t), dtype=np.int64) + zv.SERVER_HOURS_AHEAD_OF_NY) * 3600
        out[ok] = [x.replace("T", " ") + " server" for x in local.astype("datetime64[s]").astype(str)]
    return str(out[0]) if arr.ndim == 0 else out.reshape(arr.shape)


def _texts(ts, fn) -> np.ndarray:
    """fn applied to the non-negative times, '' elsewhere (an object array)."""
    arr = np.asarray(ts, dtype=np.int64)
    out = np.full(arr.shape, "", dtype=object)
    ok = arr >= 0
    if ok.any():
        out[ok] = np.asarray(fn(arr[ok]), dtype=object)
    return out


def _num(x) -> float | None:
    """A finite float or None."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_bytes(df: pd.DataFrame) -> bytes:
    """The ASCII CSV bytes of a table (no index, '\\n' line ends; other characters as backslash escapes)."""
    return df.to_csv(index=False, lineterminator="\n").encode("ascii", "backslashreplace")


def spec_identity() -> dict[str, Any]:
    """The spec's id, version, the recorded sha256 values and those of the packaged copies (they must match)."""
    out: dict[str, Any] = {"spec_id": zv.SPEC_ID, "spec_version": zv.SPEC_VERSION,
                           "sha256_md_recorded": zv.SPEC_SHA256_MD, "sha256_json_recorded": zv.SPEC_SHA256_JSON}
    for key, name, want in (("md", "zeno_pullback_v1.md.utf8", zv.SPEC_SHA256_MD),
                            ("json", "zeno_pullback_v1.json", zv.SPEC_SHA256_JSON)):
        p = zv.SPECS_DIR / name
        got = report_mod.file_sha256(p) if p.is_file() else None
        out[f"sha256_{key}_packaged"] = got
        out[f"{key}_matches"] = got == want
    out["all_match"] = bool(out["md_matches"] and out["json_matches"])
    return out


def addendum_identity() -> dict[str, Any]:
    """Addendum A's packaged copy (propkit/specs/zeno_pullback_v1_addendum_A.md): file, recorded and packaged
    sha256, whether they match."""
    p = zv.SPECS_DIR / zv.ADDENDUM_A_FILE
    got = report_mod.file_sha256(p) if p.is_file() else None
    return {"file": f"propkit/specs/{zv.ADDENDUM_A_FILE}", "sha256_recorded": zv.ADDENDUM_A_SHA256,
            "sha256_packaged": got, "matches": got == zv.ADDENDUM_A_SHA256,
            "changes": "only how the prop firm is simulated (A1-A5); zeno's 12 rules, D1-D24, the costs and the gates "
                       "are unchanged"}


def restricted_info(prep: zv.Prepared) -> dict[str, Any] | None:
    """The restricted calendar's summary (addendum A1) with whether it is the packaged file (sha256), or None."""
    if prep.restricted is None:
        return None
    s = prep.restricted.summary()
    s["packaged_sha256"] = zv.RESTRICTED_SHA256
    s["is_packaged_file"] = s.get("sha256") == zv.RESTRICTED_SHA256
    return s


def data_info(prep: zv.Prepared) -> dict[str, Any]:
    """The data description of a prepared frame: bars, range, gaps, spreads (USD/oz), ask < bid counts, the
    files and their sha256 when loaded from disk, and the number of trading (server) days."""
    info = dict(prep.frame.attrs.get("zeno_v1") or zv.bidask_summary(prep.frame))
    info["n_trading_days"] = int(prep.days.size)
    info["first_server_day"] = calendar.day_to_str(int(prep.days[0])) if prep.days.size else None
    info["last_server_day"] = calendar.day_to_str(int(prep.days[-1])) if prep.days.size else None
    info["indicators_injected"] = list(prep.injected)
    return info


def news_info(prep: zv.Prepared) -> Any:
    """The news calendar summary, or the plain statement that there is none (no blackout)."""
    return prep.news.summary() if prep.news is not None else "no news calendar given: NO news blackout"


# ---------------------------------------------------------------------------------------
# stage 1: signals and the G0 sample

SIGNAL_COLUMNS = (
    "signal_no", "setup_id", "side", "status", "reasons",
    "signal_time", "signal_time_utc", "signal_time_sgt", "signal_time_server",
    "trigger_bar_time", "trigger_bar_time_utc", "trigger_bar_time_sgt",
    "pullback_bar_time", "pullback_bar_time_utc", "pullback_bar_time_sgt",
    "extreme_bar_time_utc", "arm_bar_time_utc",
    "h_level", "l_level", "leg_usd", "atr_arm", "retrace_level", "void_level", "pullback_level", "trigger_level",
    "bars_since_pullback", "trend_ok", "atr_trigger", "atr_median",
    "entry_time", "entry_time_utc", "entry_time_sgt", "entry_price", "stop_level", "spread_entry",
    "entry_price_at_costs", "stop_level_at_costs", "spread_entry_at_costs")
AT_COSTS = "_at_costs"                 # suffix of the declared cell's prices in stage 1 [SI-66]
G0_COLUMNS = (
    "sample_no", "signal_no", "setup_id", "side", "signal_time_utc", "signal_time_sgt", "signal_time_server",
    "trigger_bar_time_utc", "trigger_bar_time_sgt", "pullback_bar_time_utc", "pullback_bar_time_sgt",
    "extreme_bar_time_utc", "h_level", "l_level", "entry_time_utc", "entry_time_sgt", "entry_price", "stop_level",
    "agree_y_n")
FORBIDDEN_STAGE1_WORDS = ("pnl", "r_multiple", "outcome", "exit", "tp1", "tp2", "gross", "net_", "R_usd")


def signals_table(decisions: pd.DataFrame) -> pd.DataFrame:
    """One row per trigger of a decisions table (SIGNAL_COLUMNS): times in UTC, SGT and server text, the
    setup's levels, the status and every blocking reason, and the entry fill and stop (known at entry; in
    stage 1 the chart's, with the declared cell's in the *_at_costs columns, [SI-66]; a table without those
    columns repeats its own prices there). No exit, P&L, R multiple or outcome. signal_time is the trigger
    bar's close (when the decision is made), trigger_bar_time its open (what a chart shows)."""
    trig = decisions[decisions["event"] == "trigger"].reset_index(drop=True)
    out = pd.DataFrame({"signal_no": np.arange(1, len(trig) + 1, dtype=np.int64)})
    for col in ("setup_id", "side", "status", "reasons"):
        out[col] = trig[col].to_numpy()
    st_ = trig["time"].to_numpy(dtype=np.int64)
    out["signal_time"] = st_
    out["signal_time_utc"] = _texts(st_, calendar.utc_str)
    out["signal_time_sgt"] = _texts(st_, zv.sgt_str)
    out["signal_time_server"] = server_time_str(st_)
    for key, col in (("trigger_bar_time", "bar_time"), ("pullback_bar_time", "pullback_bar_time")):
        t = trig[col].to_numpy(dtype=np.int64)
        out[key] = t
        out[f"{key}_utc"] = _texts(t, calendar.utc_str)
        out[f"{key}_sgt"] = _texts(t, zv.sgt_str)
    for key in ("extreme_bar_time", "arm_bar_time"):
        out[f"{key}_utc"] = _texts(trig[key].to_numpy(dtype=np.int64), calendar.utc_str)
    for col in ("h_level", "l_level", "leg_usd", "atr_arm", "retrace_level", "void_level", "pullback_level",
                "trigger_level", "bars_since_pullback", "trend_ok", "atr_trigger", "atr_median"):
        out[col] = trig[col].to_numpy()
    et = trig["entry_time"].to_numpy(dtype=np.int64)
    out["entry_time"] = et
    out["entry_time_utc"] = _texts(et, calendar.utc_str)
    out["entry_time_sgt"] = _texts(et, zv.sgt_str)
    for col in zv.CHART_PRICE_COLUMNS:
        out[col] = trig[col].to_numpy(dtype=np.float64)
    for col in zv.CHART_PRICE_COLUMNS:
        out[col + AT_COSTS] = trig[col + AT_COSTS if col + AT_COSTS in trig.columns else col].to_numpy(dtype=np.float64)
    return out[list(SIGNAL_COLUMNS)]


def g0_sample(signals: pd.DataFrame, n: int = G0_SAMPLE_SIZE, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """n eligible signals (status zeno_v1.ELIGIBLE, [SI-54]) drawn without replacement with
    numpy.random.default_rng(seed), in time order (G0_COLUMNS, with an empty agree_y_n column for zeno's
    answers). Fewer eligible signals than n: all of them."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"the G0 sample size must be a whole number >= 1, got {n!r}")
    eligible = signals[signals["status"] == zv.ELIGIBLE].reset_index(drop=True)
    k = min(int(n), len(eligible))
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(len(eligible), size=k, replace=False)) if k else np.zeros(0, dtype=np.int64)
    s = eligible.iloc[idx].reset_index(drop=True)
    out = pd.DataFrame({"sample_no": np.arange(1, k + 1, dtype=np.int64)})
    for col in G0_COLUMNS[1:-1]:
        out[col] = s[col].to_numpy() if k else []
    out["agree_y_n"] = ""
    return out[list(G0_COLUMNS)]


G0_CHECK_COLUMNS = ("signal_time_utc", "side", "entry_price", "stop_level")
G0_PRICE_TOL = 1e-6                    # USD/oz: a CSV round trip, or a spreadsheet's 15 digits, keeps far more


def _sample_no(value: Any, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def g0_row_key(row: Any) -> tuple[int | None, str, float, float]:
    """One G0 sample row read as (signal time, side, entry_price, stop_level): the time in UTC epoch seconds from
    signal_time_utc (stage 1's 'YYYY-MM-DD HH:MM:SS UTC' text; None when it, the entry or the stop cannot be
    read, with NaN prices), the side lower-cased, the prices in USD/oz. g0_sample_check and the G0 charts
    (zeno_g0_charts) read a row this way."""
    t_txt, side = str(row["signal_time_utc"]).strip(), str(row["side"]).strip().lower()
    try:
        t = int(pd.Timestamp(t_txt.removesuffix("UTC").strip(), tz="UTC").timestamp())
        entry, stop = float(row["entry_price"]), float(row["stop_level"])
    except (TypeError, ValueError):
        return None, side, math.nan, math.nan
    return t, side, entry, stop


def g0_sample_check(prep: zv.Prepared, sample: pd.DataFrame, cell: zv.ZenoCell | None = None,
                    capital: float = 100_000.0) -> dict[str, Any]:
    """Whether a G0 sample belongs to the data of `prep` [SI-69]: each row must be a trigger of this data at the
    same signal_time_utc (the trigger bar's close, UTC text as stage 1 writes it) and side, eligible in the
    declared cell (zeno_v1.screen at `cell`, default the stage-1 cell, with `capital` in USD), with the same
    chart entry_price and stop_level (zeno_v1.chart_prices) to 1e-6 USD/oz.
    sample: the g0_sample.csv table (columns G0_CHECK_COLUMNS at least; read as text is fine). Returns ok (every
    row matched), n_rows, n_matched, unmatched (per failing row: sample_no, signal_time_utc, side, why; the
    first 20), declared_cell (label) and capital_usd. ValueError when a column is missing."""
    cell = cell if cell is not None else STAGE1_CELL
    missing = [c for c in G0_CHECK_COLUMNS if c not in sample.columns]
    if missing:
        raise ValueError(f"the G0 sample lacks the column(s) {missing}; use the g0_sample.csv that `zeno-v1 signals` "
                         "wrote")
    dec = zv.screen(prep, zv.ZenoConfig(cell, capital))
    chart = zv.chart_prices(prep)
    is_trig = (dec["event"] == "trigger").to_numpy()
    where = {(int(t), str(sd)): i for i, t, sd in zip(np.flatnonzero(is_trig), dec["time"].to_numpy()[is_trig],
                                                       dec["side"].to_numpy()[is_trig])}
    unmatched = []
    for r in range(len(sample)):
        row = sample.iloc[r]
        t_txt = str(row["signal_time_utc"]).strip()
        t, side, entry, stop = g0_row_key(row)
        why = "" if t is not None else "its time, entry or stop cannot be read"
        i = where.get((t, side)) if t is not None else None
        if not why and i is None:
            why = "no trigger at that time and side in this data"
        elif not why and dec.at[i, "status"] != zv.ELIGIBLE:
            why = f"not eligible in {cell.label}: {dec.at[i, 'status']}"
        elif not why:
            ce, cs = float(chart.at[i, "entry_price"]), float(chart.at[i, "stop_level"])
            if not (abs(entry - ce) <= G0_PRICE_TOL and abs(stop - cs) <= G0_PRICE_TOL):
                why = (f"the entry or stop differs (sample {entry:.6g} / {stop:.6g} USD/oz, this data's chart "
                       f"{ce:.6g} / {cs:.6g} USD/oz)")
        if why:
            unmatched.append({"sample_no": _sample_no(row.get("sample_no"), r + 1), "signal_time_utc": t_txt,
                              "side": side, "why": why})
    n = int(len(sample))
    return {"ok": not unmatched, "n_rows": n, "n_matched": n - len(unmatched), "unmatched": unmatched[:20],
            "declared_cell": cell.label, "capital_usd": float(capital),
            "rule": "each row = an eligible trigger of this data at the same time and side, with the same chart "
                    "entry and stop [SI-69]"}


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def stage1_decisions(prep: zv.Prepared, cell: zv.ZenoCell | None = None,
                     capital: float = 100_000.0) -> pd.DataFrame:
    """Stage 1's decisions table: zeno_v1.screen in the declared cell (status and reasons, [SI-27], [SI-54]),
    without position_id, whose entry_price, stop_level and spread_entry are the CHART's (zeno_v1.chart_prices:
    the data at costs x1, what zeno sees on a chart, [SI-66]); the cell's own prices, which the rule-10 spread
    filter compared, are kept as entry_price_at_costs, stop_level_at_costs and spread_entry_at_costs."""
    cell = cell if cell is not None else STAGE1_CELL
    dec = zv.screen(prep, zv.ZenoConfig(cell, capital)).drop(columns=["position_id"]).reset_index(drop=True)
    chart = zv.chart_prices(prep)
    for col in zv.CHART_PRICE_COLUMNS:
        dec[col + AT_COSTS] = dec[col].to_numpy(dtype=np.float64)
        dec[col] = chart[col].to_numpy(dtype=np.float64)
    return dec


def signals_stage(prep: zv.Prepared, cell: zv.ZenoCell | None = None, capital: float = 100_000.0,
                  sample: int = G0_SAMPLE_SIZE, seed: int = DEFAULT_SEED,
                  inputs: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Stage 1 (G0): screen the triggers in one declared cost cell [SI-27] (zeno_v1.screen: every check
    except the ones that need earlier trades [SI-54]) and return (report, tables) with tables signals,
    decisions (without position ids) and g0_sample. The report holds counts only (triggers per status, side
    and year of the signal's server day; setup events), the declared cell, the data and news descriptions,
    the spec identity and the G0 instructions: no P&L, R, hit rate, outcome, exit or price after an entry,
    and no check whose answer depends on them."""
    cell = cell if cell is not None else STAGE1_CELL
    dec = stage1_decisions(prep, cell, capital)
    sig = signals_table(dec)
    g0 = g0_sample(sig, sample, seed)
    year = np.asarray(calendar.day_to_str(np.asarray(zv.server_day(sig["signal_time"].to_numpy()), dtype=np.int64)),
                      dtype=object).astype(str) if len(sig) else np.array([], dtype=str)
    years = pd.Series([y[:4] for y in year], dtype=object)
    ev = dec[dec["event"] != "trigger"]
    status_by_side = {side: _counts(sig.loc[sig["side"] == side, "status"]) for side in ("long", "short")}
    by_year = {}
    for y in sorted(set(years)):
        m = (years == y).to_numpy()
        by_year[y] = {"triggers": int(m.sum()), "eligible": int((sig["status"].to_numpy()[m] == zv.ELIGIBLE).sum())}
    report = {
        "header": HEADER, "stage": "signals (stage 1, for the G0 chart check)",
        "generated_utc": calendar.utc_str(int(time.time())), "propkit_version": propkit.__version__,
        "spec": spec_identity(), "inputs": dict(inputs or {}), "data": data_info(prep), "news": news_info(prep),
        "declared_cell": {**cell.to_dict(), "label": cell.label, "capital_usd": float(capital),
                          "risk_pct": zv.ZenoConfig(cell, capital).risk_fraction,
                          "why": "eligible vs blocked depends on the spread filter (rule 10), which needs costs "
                                 "[SI-27]: it compares this cell's entry spread (k x the data's) with 10% of the stop "
                                 "distance at this cell's prices (signals.csv: entry_price_at_costs, "
                                 "stop_level_at_costs, spread_entry_at_costs). entry_price, stop_level and "
                                 "spread_entry in signals.csv and g0_sample.csv are the chart's, the data at costs "
                                 "x1: the ask open (long) or bid open (short) of the next bar, the short stop with "
                                 "the data's spread at that open [SI-66]. H, L, the pullback bar and the trigger do "
                                 "not depend on the cell"},
        "n_triggers": int(len(sig)), "n_eligible": int((sig["status"] == zv.ELIGIBLE).sum()),
        "status_counts": _counts(sig["status"]), "status_counts_by_side": status_by_side,
        "triggers_by_year": by_year, "setup_events": _counts(ev["event"]) if len(ev) else {},
        "setup_events_by_side": {side: _counts(ev.loc[ev["side"] == side, "event"]) for side in ("long", "short")},
        "not_checked": STAGE1_NOT_CHECKED,
        "news_unscheduled": zv.unscheduled_counts(prep, dec),
        "block_reasons": {k: v for k, v in zv.BLOCK_REASONS.items() if k not in zv.STATE_REASONS
                          and _reason_applies(k, cell.variant)},
        "g0": {"sample_size_asked": int(sample), "sample_size": int(len(g0)), "seed": int(seed),
               "min_agree": G0_MIN_AGREE, "instructions": G0_INSTRUCTIONS,
               "short_sample_note": None if len(g0) >= sample else
               f"only {len(g0)} eligible signals exist, so the sample holds all of them"},
        "no_results": "This stage writes no P&L, R multiple, hit rate, outcome or price after an entry.",
    }
    if cell.variant == "master_fp":                       # addendum A1: the restricted list this cell uses
        report["addendum_a"] = addendum_identity()
        report["restricted"] = restricted_info(prep)
    return report_mod.clean(report), {"signals": sig, "decisions": dec, "g0_sample": g0}


def _reason_applies(reason: str, variant: str) -> bool:
    """Whether a blocking reason can occur in a variant: the restricted window only in master_fp (addendum A1),
    the margin cap only in the Master variants (A2), every other reason everywhere."""
    if reason == "fp_restricted_window":
        return variant == "master_fp"
    if reason == "margin_cap_below_lot_step":
        return variant in zv.MASTER_VARIANTS
    return True


def render_signals_markdown(report: Mapping[str, Any]) -> str:
    """The stage-1 report (signals_stage) as Markdown text (ASCII)."""
    r = report
    c = r["declared_cell"]
    d = r["data"]
    L = [HEADER, "", "# zeno_pullback_v1 - stage 1: signals for the G0 check", "",
         f"Generated {r['generated_utc']} by propkit {r['propkit_version']}. {r['no_results']}", "",
         "## What to do now", "", r["g0"]["instructions"], ""]
    if r["g0"].get("short_sample_note"):
        L += [f"NOTE: {r['g0']['short_sample_note']}.", ""]
    L += [f"Sample: {r['g0']['sample_size']} eligible signals, drawn with seed {r['g0']['seed']} (numpy "
          "default_rng, without replacement), in time order.", "",
          "## Counts", "",
          f"- triggers: {r['n_triggers']} (eligible {r['n_eligible']}); setup events: "
          + ", ".join(f"{k} {v}" for k, v in (r["setup_events"] or {}).items()), "",
          "| status | triggers | long | short |", "|---|---|---|---|"]
    for s, n in (r["status_counts"] or {}).items():
        L.append(f"| {s} | {n} | {r['status_counts_by_side']['long'].get(s, 0)} | "
                 f"{r['status_counts_by_side']['short'].get(s, 0)} |")
    L += ["", "| year (server day of the signal) | triggers | eligible |", "|---|---|---|"]
    for y, v in (r["triggers_by_year"] or {}).items():
        L.append(f"| {y} | {v['triggers']} | {v['eligible']} |")
    L += ["", f"{r['not_checked']}.", "",
          "## Declared cost cell", "",
          f"{c['label']} (variant {c['variant']}, commission {c['commission_rt_per_lot']:g} USD per lot round trip, "
          f"spread base {c['spread_base']}, cost multiplier x{c['cost_mult']:g}), capital {c['capital_usd']:,.2f} USD, "
          f"risk {100 * c['risk_pct']:.2f}% per trade. Why: {c['why']}.", "",
          "## Data", "",
          f"- M15 bid/ask bars: {d.get('n_bars')} bars, {d.get('first_time_utc')} .. {d.get('last_time_utc')} (bar "
          f"opens), {d.get('n_trading_days')} server days with bars; spread at the open median "
          f"{report_mod._fmt(d.get('spread_open_median'), 3)} USD/oz, p90 {report_mod._fmt(d.get('spread_open_p90'), 3)} "
          "USD/oz.",
          f"- bid file {d.get('bid_file', 'in memory')} (sha256 {d.get('bid_sha256', 'n/a')}); ask file "
          f"{d.get('ask_file', 'in memory')} (sha256 {d.get('ask_sha256', 'n/a')}).",
          f"- D1 range: from {d.get('range_start_utc', zv.RANGE_START_TEXT)} to the lock [SI-62]"
          + (f"; {d['range_note']}." if d.get("range_note") else "; the data starts inside it."),
          f"- news: {_news_line(r['news'])}",
          f"- {_unscheduled_line(r.get('news_unscheduled'))}."]
    if r.get("restricted"):
        L.append(f"- restricted events (variant master_fp, addendum A1): {_restricted_line(r['restricted'])}.")
    if r.get("addendum_a"):
        a = r["addendum_a"]
        L.append(f"- addendum A: {a['file']} (sha256 {a['sha256_packaged']}; matches the recorded value: "
                 f"{'yes' if a['matches'] else 'NO'}).")
    L += [f"- spec {r['spec']['spec_id']} v{r['spec']['spec_version']}: packaged copies match the recorded sha256: "
          f"{'yes' if r['spec']['all_match'] else 'NO'}.", "",
          "## Files", "",
          "- signals.csv: one row per trigger (status eligible or the first blocking reason, every blocking reason, "
          "times in UTC, SGT and server time, H, L, the pullback bar, the trigger bar, the entry and stop a chart "
          "of the data shows, and the declared cell's entry, stop and spread in the *_at_costs columns [SI-66]). "
          "No exit, P&L, R or outcome.",
          "- decisions.csv: every setup event (armed, first_close_before_arming, cancelled_new_extreme, voided, "
          "expired) and every trigger.",
          "- g0_sample.csv: the sample to check on a chart, with an empty agree_y_n column.", ""]
    return report_mod._ascii("\n".join(L))


def _unscheduled_line(nu: Mapping[str, Any] | None, twin: Mapping[str, Any] | None = None,
                      twin_fp: Mapping[str, Any] | None = None) -> str:
    """One report line on what the unscheduled rows do before their instant [SI-63]. nu: the counts of the cell
    itself (zeno_v1.unscheduled_counts); when it is a master_fp cell's (stage 1 with --variant master_fp), the
    restricted window's counts follow in a clause of their own. twin, twin_fp: the run's Master twins."""
    if not isinstance(nu, Mapping):
        return "unscheduled rows: n/a"
    txt = (f"unscheduled rows: {nu.get('n_rows', 0)}. {UNSCHEDULED_NOTE}: {nu.get('n_triggers_flagged', 0)} "
           f"trigger(s) were blocked by news only in the 30 min before an unscheduled row "
           f"({nu.get('n_triggers_blocked_only_before_unscheduled', 0)} with no other reason; decisions column "
           "news_pre_unscheduled)")
    if "fp_n_rows" in nu:
        txt += (f"; the restricted window (variant master_fp) blocked {nu.get('fp_n_triggers_flagged', 0)} trigger(s) "
                f"only in the 5 min before one of the {nu.get('fp_n_rows', 0)} unscheduled restricted rows (decisions "
                f"column fp_pre_unscheduled), and {nu.get('n_triggers_blocked_only_before_unscheduled_any', 0)} "
                "trigger(s) were blocked only before unscheduled rows (news and/or the restricted window) with no "
                "other reason")
    if isinstance(twin, Mapping):
        txt += (f"; the master twin closed {len(twin.get('master_closes_only_for_unscheduled') or [])} position(s) "
                "only for an unscheduled row")
    if isinstance(twin_fp, Mapping):
        txt += (f"; the master_fp twin closed {len(twin_fp.get('master_closes_only_for_unscheduled') or [])} "
                f"position(s) only for an unscheduled row, and its restricted window blocked "
                f"{twin_fp.get('fp_n_triggers_flagged', 0)} trigger(s) only in the 5 min before one of the "
                f"{twin_fp.get('fp_n_rows', 0)} unscheduled restricted rows")
    return txt


def _restricted_line(rs: Mapping[str, Any] | None) -> str:
    """One report line on FundingPips' restricted calendar (addendum A1)."""
    if not isinstance(rs, Mapping):
        return "none"
    unk = ", ".join(f"{u['event']} {u['date_et']}" for u in rs.get("unknown_time_rows") or []) or "none"
    return (f"{rs.get('n_events_known_time')} events with a time ({rs.get('fedchair_testimony')} Fed Chair testimonies "
            f"at 180 min, {rs.get('fedchair_other')} other Fed Chair appearances at 60 min, the rest releases), "
            f"{rs.get('n_unknown_time')} without a time (whole New York day blocked: {unk}), "
            f"{rs.get('n_unscheduled')} unscheduled; window {rs.get('window')}; {rs.get('first_utc')} .. "
            f"{rs.get('last_utc')}; file {rs.get('source')} (sha256 {rs.get('sha256')}"
            + ("" if rs.get("is_packaged_file") else "; NOT the packaged file") + ")")


def _news_line(news: Any) -> str:
    if isinstance(news, Mapping):
        per = ", ".join(f"{k} {v}" for k, v in (news.get("per_event") or {}).items())
        return (f"{news.get('n_events')} events ({per}), {news.get('first_utc')} .. {news.get('last_utc')}, file "
                f"{news.get('source')} (sha256 {news.get('sha256')})")
    return str(news)


# ---------------------------------------------------------------------------------------
# stage 2: periods and metrics

def period_table(prep: zv.Prepared) -> list[dict[str, Any]]:
    """The report periods: "all", every calendar year of the data's server days, and the four G3 periods
    (2015-2017, 2018-2020, 2021-2023, 2024 to 2025-09-27). Each: label, kind (all | year | g3), first_day and
    last_day (days since 1970-01-01, inclusive; server days, [SI-42]) and span_years (first to last trading
    day of the data inside the period, calendar days / 365.25; None without data) [SI-56]."""
    days = np.asarray(prep.days, dtype=np.int64)
    rows: list[tuple[str, str, int, int]] = []
    if days.size:
        rows.append(("all", "all", int(days[0]), int(days[-1])))
        years = sorted({int(y) for y in np.asarray(calendar.day_to_str(days), dtype=object).astype(str).astype("U4")})
        for y in years:
            rows.append((str(y), "year", _day_of(f"{y}-01-01"), _day_of(f"{y}-12-31")))
    else:
        rows.append(("all", "all", 0, -1))
    for label, a, b in G3_PERIODS:
        rows.append((label, "g3", _day_of(a), _day_of(b)))
    out = []
    for label, kind, lo, hi in rows:
        inside = days[(days >= lo) & (days <= hi)]
        span = (int(inside[-1]) - int(inside[0]) + 1) / 365.25 if inside.size else None
        out.append({"period": label, "kind": kind, "first_day": lo, "last_day": hi, "span_years": span,
                    "first_date": calendar.day_to_str(lo) if hi >= lo else None,
                    "last_date": calendar.day_to_str(hi) if hi >= lo else None,
                    "n_trading_days": int(inside.size)})
    return out


def longest_losing_streak(positions: pd.DataFrame) -> int:
    """The longest run of consecutive positions with net P&L < 0, positions ordered by their final exit
    stamp (ties by position id) [SI-41]. 0 for no position or no loss."""
    if len(positions) == 0:
        return 0
    p = positions.sort_values(["final_exit_stamp", "position_id"], kind="stable")
    best = run = 0
    for loss in (p["net_pnl_usd"].to_numpy(dtype=np.float64) < 0):
        run = run + 1 if loss else 0
        best = max(best, run)
    return int(best)


def position_metrics(positions: pd.DataFrame, n_legs: int | None = None,
                     span_years: float | None = None) -> dict[str, Any]:
    """Trade-level metrics of a POSITIONS table (zeno_v1 columns), net of every cost [SI-41]:
    n_positions, n_legs, trades_per_year (positions / span_years), hit_rate_tp1 (share of positions that
    FILLED the +2R partial: tp1_reached and partial_lots > 0) with its count n_tp1,
    hit_rate_tp2_after_tp1 (share of those whose runner reached +4R), n_tp1_reached_without_partial (0.01-lot
    positions that reached +2R, where half rounds to 0 lots (D16): no partial filled, so not a +2R hit; their
    stop still moved to breakeven, [SI-22]), win_rate (net P&L > 0), expectancy_r (mean net R per position)
    and se_expectancy_r (sample sd / sqrt(n)), expectancy_usd (mean net USD per position), net_usd,
    longest_losing_streak (positions) and the count of each outcome class. Rates are fractions; None where
    undefined (no position, n < 2 for the SE)."""
    n = int(len(positions))
    r = positions["r_multiple_net"].to_numpy(dtype=np.float64) if n else np.zeros(0)
    usd = positions["net_pnl_usd"].to_numpy(dtype=np.float64) if n else np.zeros(0)
    reached = positions["tp1_reached"].to_numpy(dtype=bool) if n else np.zeros(0, dtype=bool)
    has_partial = positions["partial_lots"].to_numpy(dtype=np.float64) > 0 if n else np.zeros(0, dtype=bool)
    tp1 = reached & has_partial                  # filled the +2R partial (a tp1 leg exists)
    tp2 = positions["tp2_reached"].to_numpy(dtype=bool) if n else np.zeros(0, dtype=bool)
    out: dict[str, Any] = {
        "n_positions": n, "n_legs": None if n_legs is None else int(n_legs),
        "trades_per_year": (n / span_years) if span_years else None,
        "hit_rate_tp1": float(tp1.mean()) if n else None, "n_tp1": int(tp1.sum()),
        "hit_rate_tp2_after_tp1": float(tp2[tp1].mean()) if tp1.any() else None,
        "n_tp1_reached_without_partial": int((reached & ~has_partial).sum()),
        "win_rate": float((usd > 0).mean()) if n else None,
        "expectancy_r": float(r.mean()) if n else None,
        "se_expectancy_r": float(r.std(ddof=1) / math.sqrt(n)) if n >= 2 else None,
        "expectancy_usd": float(usd.mean()) if n else None, "net_usd": float(usd.sum()),
        "longest_losing_streak": longest_losing_streak(positions),
    }
    oc = positions["outcome"].astype(str).to_numpy() if n else np.array([], dtype=str)
    for name, key in OUTCOME_KEYS.items():
        out[key] = int((oc == name).sum())
    return out


def daily_metrics(daily: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    """Daily (server-day) statistics of a slice of daily_returns_from_equity: n_days, sharpe_daily +-
    se_sharpe_daily (per server day), the annualised versions (at the observed days per year, IID), psr_0 =
    P(true SR > 0), dsr_n1 (the DSR with N = 1, which equals psr_0; see DSR_CAVEAT), skew, kurt, and
    max_dd_close_pct (largest fall of the server-day closing equity from its running peak, % of the initial
    capital). sharpe_note says why the Sharpe fields are None (fewer than 2 days, no variation)."""
    rets = daily["ret"].to_numpy(dtype=np.float64)
    out: dict[str, Any] = {"n_days": int(rets.size), "sharpe_daily": None, "se_sharpe_daily": None,
                           "sharpe_annual": None, "se_sharpe_annual": None, "days_per_year": None, "psr_0": None,
                           "dsr_n1": None, "skew": None, "kurt": None, "sharpe_note": "", "max_dd_close_pct": None}
    if rets.size:
        eq = np.r_[float(daily["equity_start"].iloc[0]), daily["equity_end"].to_numpy(dtype=np.float64)]
        out["max_dd_close_pct"] = float((np.maximum.accumulate(eq) - eq).max() / float(initial_capital) * 100.0)
    if rets.size < 2:
        out["sharpe_note"] = "fewer than 2 server days"
        return out
    if np.ptp(rets) == 0:
        out["sharpe_note"] = "no variation in the daily returns (no trade closed or open)"
        return out
    q = st.observed_periods_per_year(daily["day"].to_numpy(dtype=np.int64))
    s = st.sharpe_stats(rets, q)
    d = st.dsr_details(s.sr, s.n, s.skew, s.kurt, 1, 0.0)
    out.update(sharpe_daily=s.sr, se_sharpe_daily=s.se_sr, sharpe_annual=s.sr_annual, se_sharpe_annual=s.se_sr_annual,
               days_per_year=q, psr_0=s.psr_0, dsr_n1=float(d["dsr"]), skew=s.skew, kurt=s.kurt)
    return out


def _bars_and_costs(prep: zv.Prepared, cell: zv.ZenoCell, cache: dict) -> tuple[pd.DataFrame, Any, dict]:
    key = (cell.spread_base, cell.cost_mult, cell.commission_rt_per_lot)
    if key not in cache:
        cache[key] = (zv.to_propkit_bars(prep.frame, cell.spread_base, cell.cost_mult), zv.cost_model_for_cell(cell),
                      zv.cell_ask_marks(prep.frame, cell))
    return cache[key]


def side_equity(prep: zv.Prepared, res: zv.CellResult, side: str, cache: dict | None = None) -> pd.DataFrame:
    """EQUITY of one side's legs alone ("long" / "short"), from the cell's capital [SI-57], shorts marked on the
    cell's ask [SI-67]; "combined" is zeno_v1.cell_equity. cache: an optional dict reused across calls (bars,
    cost model and ask marks per cell)."""
    if side == "combined":
        return zv.cell_equity(prep, res)[0]
    if side not in ("long", "short"):
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    bars, cm, marks = _bars_and_costs(prep, res.cell, {} if cache is None else cache)
    ids = res.positions.loc[res.positions["side"] == side, "position_id"].to_numpy()
    legs = res.legs[res.legs["position_id"].isin(ids)].drop(columns=["position_id", "leg"])
    return equity_from_trades(bars, legs, res.config.capital_usd, cm, price_tolerance=None, ask_prices=marks)[0]


def cell_rows(prep: zv.Prepared, res: zv.CellResult, periods: Sequence[Mapping[str, Any]],
              cache: dict | None = None) -> list[dict[str, Any]]:
    """The grid rows of one cell: for each side (long, short, combined) and period, the cell's fields,
    n_triggers and n_spread_blocked (triggers of the side whose trigger close falls in the period, and those
    of them the rule-10 spread filter blocks, alone or with other reasons; the filter sees the cell's scaled
    spread [SI-65]), position_metrics (positions assigned by the server day of their entry [SI-42]) and
    daily_metrics (the server-day returns of that side's equity inside the period [SI-30]). grid_frame adds
    positions_vs_x1."""
    cache = {} if cache is None else cache
    cell = res.cell
    c0 = res.config.capital_usd
    pos = res.positions
    pday = (pd.to_datetime(pos["server_day"]).to_numpy().astype("datetime64[D]").astype(np.int64)
            if len(pos) else np.zeros(0, dtype=np.int64))
    legs_per_pos = res.legs.groupby("position_id").size() if len(res.legs) else pd.Series(dtype=np.int64)
    trig = res.decisions[res.decisions["event"] == "trigger"]
    tday = prep.close_day[trig["bar_index"].to_numpy(dtype=np.int64)] if len(trig) else np.zeros(0, dtype=np.int64)
    tside = trig["side"].to_numpy(dtype=object)
    tspread = np.array(["spread_gt_10pct_of_stop" in str(r).split(";") for r in trig["reasons"]], dtype=bool)
    rows = []
    for side in SIDES:
        eq = side_equity(prep, res, side, cache)
        daily = st.daily_returns_from_equity(eq, c0, by=DAY_KEY)
        dday = daily["day"].to_numpy(dtype=np.int64)
        sm = np.ones(len(pos), dtype=bool) if side == "combined" else (pos["side"].to_numpy() == side)
        tm = np.ones(len(trig), dtype=bool) if side == "combined" else (tside == side)
        for p in periods:
            m = sm & (pday >= p["first_day"]) & (pday <= p["last_day"])
            mt = tm & (tday >= p["first_day"]) & (tday <= p["last_day"])
            sub = pos[m]
            n_legs = int(legs_per_pos.reindex(sub["position_id"]).fillna(0).sum()) if len(sub) else 0
            row = {"variant": cell.variant, "commission_rt_per_lot": cell.commission_rt_per_lot,
                   "spread_base": cell.spread_base, "cost_mult": cell.cost_mult, "cell": cell.label,
                   "side": side, "period": p["period"], "period_kind": p["kind"],
                   "first_date": p["first_date"], "last_date": p["last_date"],
                   "n_triggers": int(mt.sum()), "n_spread_blocked": int((mt & tspread).sum())}
            row.update(position_metrics(sub, n_legs, p["span_years"]))
            dm = (dday >= p["first_day"]) & (dday <= p["last_day"])
            row.update(daily_metrics(daily[dm], c0))
            rows.append(row)
    return rows


def grid_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """The grid table (one row per cell x side x period) from cell_rows output, with positions_vs_x1: the
    row's positions minus those of the same variant, commission, spread base, side and period at costs x1
    (the multiplier also tightens the spread filter [SI-65]; empty when that x1 row is not in the grid)."""
    grid = pd.DataFrame(list(rows))
    key = ["variant", "commission_rt_per_lot", "spread_base", "side", "period"]
    if len(grid) and set(key) | {"cost_mult", "n_positions"} <= set(grid.columns):
        x1 = grid[np.isclose(grid["cost_mult"].astype(float), 1.0)][key + ["n_positions"]]
        x1 = x1.drop_duplicates(key).rename(columns={"n_positions": "_n_x1"})
        merged = grid[key].merge(x1, on=key, how="left")
        grid["positions_vs_x1"] = (grid["n_positions"].to_numpy(dtype=np.float64)
                                   - merged["_n_x1"].to_numpy(dtype=np.float64))
        if not grid["positions_vs_x1"].isna().any():
            grid["positions_vs_x1"] = grid["positions_vs_x1"].astype(np.int64)
    return grid


def _grid_row(grid: pd.DataFrame, variant: str, commission: float, base: str, mult: float, side: str,
              period: str) -> dict[str, Any] | None:
    m = ((grid["variant"] == variant) & np.isclose(grid["commission_rt_per_lot"].astype(float), commission)
         & (grid["spread_base"] == base) & np.isclose(grid["cost_mult"].astype(float), mult)
         & (grid["side"] == side) & (grid["period"] == period))
    sel = grid[m]
    if len(sel) == 0:
        return None
    return {k: (None if (isinstance(v, float) and not math.isfinite(v)) else v) for k, v in sel.iloc[0].items()}


def worse_base(grid: pd.DataFrame, variant: str = JUDGING_VARIANT, commission: float = JUDGING_COMMISSION,
               cost_mult: float = JUDGING_COST_MULT) -> dict[str, Any]:
    """The worse spread base at (variant, commission, cost_mult): the one with the LOWER combined net
    expectancy in R per position over all data [SI-28]. No position (expectancy undefined) counts as worse;
    a tie goes to S2. Returns spread_base, the expectancy of each base and the rule text."""
    vals = {}
    for b in zv.SPREAD_BASES:
        row = _grid_row(grid, variant, commission, b, cost_mult, "combined", "all")
        vals[b] = None if row is None else _num(row.get("expectancy_r"))
    key = {b: (-math.inf if v is None else v) for b, v in vals.items()}
    base = "S2" if key["S2"] <= key["S1"] else "S1"
    return {"spread_base": base, "expectancy_r": vals, "variant": variant, "commission_rt_per_lot": float(commission),
            "cost_mult": float(cost_mult),
            "rule": "the base with the lower combined net expectancy (R per position, all data); no position counts "
                    "as worse; a tie goes to S2 [SI-28]"}


# ---------------------------------------------------------------------------------------
# gates

def _gate(status: str, judged_at: str, value: Any, threshold: str, reading: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "judged_at": judged_at, "value": value, "threshold": threshold, "reading": reading,
            **extra}


def g4_applicability(rules: PropRules | None, rules_info: Mapping[str, Any] | None) -> tuple[bool, str]:
    """(whether G4 can be evaluated, the reason if not). G4 needs the VERIFIED FundingPips rules with a profit
    target and a max-loss rule ("before target" and P(max-loss breach) are undefined otherwise) [SI-32]."""
    info = dict(rules_info or {})
    if rules is None:
        return False, "no rules given"
    if str(info.get("firm") or "").strip().lower() != "fundingpips":
        return False, (f"the rules are {info.get('firm') or rules.name}'s, not FundingPips' (G4 is defined under the "
                       "verified FundingPips rules)")
    if info.get("verified") is not True:
        return False, "FundingPips rule sheet unverified (no verified profit target or max-loss rule) [SI-32]"
    missing = [f for f in ("profit_target_pct", "max_loss_pct") if getattr(rules, f) is None]
    if missing:
        return False, f"the verified FundingPips rules give no {' and no '.join(missing)}, so G4 is undefined"
    return True, ""


def g4_gate(applicable: bool, reason: str, boot_result: Mapping[str, Any] | None, judged_at: str,
            sensitivity: tuple[str, Mapping[str, Any]] | None = None,
            day_boundary: str | None = None) -> dict[str, Any]:
    """G4 from the judging cell's bootstrap (horizon: until pass or breach, [SI-31]); not_evaluated with the
    reason when g4_applicability says so or no bootstrap ran.

    Addendum A5: with sensitivity = (its firm-day boundary, the judging cell's bootstrap under it) and
    day_boundary = the rules' own boundary, G4's P(daily-loss breach) is the HIGHER of the two (the rules' own
    on a tie), and the value names the boundary that set it (p_breach_daily_set_by) and lists both
    (p_breach_daily_by_boundary). P(max-loss breach) is the rules' own boundary's (A5 names one value);
    both are listed (p_breach_max_by_boundary)."""
    if sensitivity is None:
        thr = f"P(daily-loss breach before target) <= {G4_MAX_P_DAILY:g} and P(max-loss breach before target) <= " \
              f"{G4_MAX_P_MAX:g}"
    else:
        thr = f"the higher P(daily-loss breach before target) of the two firm days <= {G4_MAX_P_DAILY:g} and " \
              f"P(max-loss breach before target) <= {G4_MAX_P_MAX:g} (addendum A5)"
    if not applicable or boot_result is None:
        return _gate("not_evaluated", judged_at, None, thr, f"G4 is not evaluated: {reason or 'no bootstrap ran'}.",
                     reason=reason or "no bootstrap ran")
    pd_, pm = float(boot_result["p_breach_daily"]), float(boot_result["p_breach_max"])
    se_d = boot_result.get("se_breach_daily")
    value: dict[str, Any] = {"p_breach_daily": pd_, "se_breach_daily": se_d, "p_breach_max": pm,
                             "se_breach_max": boot_result.get("se_breach_max"), "n_sims": boot_result.get("n_sims"),
                             "horizon_days": boot_result.get("horizon_days")}
    extra = ""
    if sensitivity is not None:
        own = day_boundary or "rules"
        alt, alt_boot = sensitivity
        pd_alt = float(alt_boot["p_breach_daily"])
        set_by = alt if pd_alt > pd_ else own
        value["p_breach_daily_by_boundary"] = {own: pd_, alt: pd_alt}
        value["p_breach_max_by_boundary"] = {own: pm, alt: float(alt_boot["p_breach_max"])}
        value["p_breach_daily_set_by"] = set_by
        value["p_breach_daily_tie"] = pd_alt == pd_
        if pd_alt > pd_:
            pd_, se_d = pd_alt, alt_boot.get("se_breach_daily")
            value["p_breach_daily"], value["se_breach_daily"] = pd_, se_d
        extra = (f" (the higher of {own} {value['p_breach_daily_by_boundary'][own]:.4f} and {alt} {pd_alt:.4f}: set by "
                 f"{set_by}{', a tie' if value['p_breach_daily_tie'] else ''}; P(max-loss breach) under {own})")
    ok = pd_ <= G4_MAX_P_DAILY and pm <= G4_MAX_P_MAX
    return _gate("pass" if ok else "fail", judged_at, value, thr,
                 f"P(daily-loss breach before target) {pd_:.4f}{extra}, P(max-loss breach before target) {pm:.4f} "
                 f"({'within' if ok else 'outside'} the limits).")


def gates_from(grid: pd.DataFrame, g4: Mapping[str, Any] | None = None,
               g0: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """gates.json content from the grid table (grid_frame), the G4 gate (g4_gate; None = not evaluated) and
    the G0 record (from the command line; never computed by code).

    G1: positions (combined, all data) at the judging cell >= 100, else "insufficient sample" [SI-29].
    G2: expectancy_r > 0 and psr_0 >= 0.95 (server-day returns [SI-30]) at the judging cell.
    G3: expectancy_r > 0 in at least 3 of the 4 G3 periods at the judging cell (no position: not above 0,
        [SI-53]). G5: per side expectancy_r at commission 10, the worse base at x1 [SI-52], evaluation; a side
    at or below 0 is flagged not tradeable on its own. Kill: G2's two tests at that x1 cell; fired when
    either fails. The judging cell: evaluation, commission 10, the worse base at x1.5 [SI-28]."""
    j = worse_base(grid)
    j1 = worse_base(grid, cost_mult=KILL_COST_MULT)
    jcell = zv.ZenoCell(JUDGING_VARIANT, JUDGING_COMMISSION, j["spread_base"], JUDGING_COST_MULT)
    kcell = zv.ZenoCell(JUDGING_VARIANT, JUDGING_COMMISSION, j1["spread_base"], KILL_COST_MULT)

    def row(cell: zv.ZenoCell, side: str = "combined", period: str = "all") -> dict[str, Any]:
        return _grid_row(grid, cell.variant, cell.commission_rt_per_lot, cell.spread_base, cell.cost_mult, side,
                         period) or {}

    at, at1 = jcell.label, kcell.label
    jr = row(jcell)
    n = int(jr.get("n_positions") or 0)
    g1 = _gate("pass" if n >= G1_MIN_POSITIONS else "fail", at, n, f">= {G1_MIN_POSITIONS} positions",
               f"{n} positions (entries; {jr.get('n_legs')} legs)" +
               ("" if n >= G1_MIN_POSITIONS else ": insufficient sample"),
               verdict_text=None if n >= G1_MIN_POSITIONS else "insufficient sample")

    def edge(r: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
        e, p = _num(r.get("expectancy_r")), _num(r.get("psr_0"))
        ok = e is not None and e > 0 and p is not None and p >= G2_MIN_PSR
        return ok, {"expectancy_r": e, "se_expectancy_r": _num(r.get("se_expectancy_r")), "psr_0": p,
                    "n_positions": int(r.get("n_positions") or 0), "n_days": int(r.get("n_days") or 0),
                    "sharpe_daily": _num(r.get("sharpe_daily")), "dsr_n1": _num(r.get("dsr_n1"))}

    ok2, v2 = edge(jr)
    g2 = _gate("pass" if ok2 else "fail", at, v2, f"expectancy_r > 0 R per position and PSR(SR > 0) >= {G2_MIN_PSR:g}",
               f"E[R] {_txt(v2['expectancy_r'], 4)} R per position, PSR {_txt(v2['psr_0'], 4)} on server-day returns.")
    per = {}
    for label, _, _ in G3_PERIODS:
        r = row(jcell, "combined", label)
        e = _num(r.get("expectancy_r"))
        per[label] = {"expectancy_r": e, "n_positions": int(r.get("n_positions") or 0), "above_0": bool(e is not None
                                                                                                     and e > 0)}
    k3 = sum(1 for v in per.values() if v["above_0"])
    g3 = _gate("pass" if k3 >= G3_MIN_PERIODS else "fail", at, {"periods_above_0": k3, "per_period": per},
               f"expectancy_r > 0 in >= {G3_MIN_PERIODS} of 4 periods", f"{k3} of 4 periods above 0 R.")
    sides = {}
    for side in ("long", "short"):
        r = row(kcell, side)
        e = _num(r.get("expectancy_r"))
        sides[side] = {"expectancy_r": e, "n_positions": int(r.get("n_positions") or 0),
                       "flag": None if (e is not None and e > 0) else "not_tradeable_alone"}
    flagged = [s for s, v in sides.items() if v["flag"]]
    g5 = _gate("flagged" if flagged else "pass", at1, sides, "per side expectancy_r > 0 at costs x1",
               ("flagged as not tradeable on its own: " + ", ".join(flagged)) if flagged else
               "both sides above 0 R at costs x1.", flagged_sides=flagged)
    okk, vk = edge(row(kcell))
    kill = {"fired": not okk, "judged_at": at1, "value": vk,
            "test": f"G2 at costs x1: expectancy_r > 0 and PSR >= {G2_MIN_PSR:g}",
            "reading": KILL_TEXT if not okk else "G2 holds at costs x1: the kill rule does not fire."}
    g4d = dict(g4) if g4 is not None else g4_gate(False, "not run", None, at)
    g0d = dict(g0 or {"status": "not_confirmed", "confirmed_by_operator": False})
    gates = {"G0": g0d, "G1": g1, "G2": g2, "G3": g3, "G4": g4d, "G5": g5}
    if g1["status"] == "fail":
        verdict = "insufficient sample (G1 failed: fewer than 100 positions at the judging cell)"
    else:
        failed = [k for k in ("G2", "G3", "G4") if gates[k]["status"] == "fail"]
        if failed:
            verdict = "FAIL: " + ", ".join(failed) + " failed at the judging cell"
        elif g4d["status"] == "not_evaluated":
            verdict = "G1-G3 pass; G4 not evaluated (" + str(g4d.get("reason")) + "): no overall pass"
        else:
            verdict = "G1-G4 pass at the judging cell. " + AFTER_A_PASS
    if kill["fired"]:
        verdict += ". KILL: " + KILL_TEXT
    return report_mod.clean({
        "header": HEADER, "spec_id": zv.SPEC_ID, "spec_version": zv.SPEC_VERSION,
        "judging_cell": {**jcell.to_dict(), "label": at, "worse_base": j},
        "x1_cell": {**kcell.to_dict(), "label": at1, "worse_base": j1,
                    "same_base_as_judging": j1["spread_base"] == j["spread_base"]},
        "gates": gates, "kill": kill, "verdict": verdict, "after_a_pass": AFTER_A_PASS,
        "n_trials": 1, "dsr_caveat": DSR_CAVEAT})


def _txt(x, nd: int = 4) -> str:
    return report_mod._fmt(x, nd)


# ---------------------------------------------------------------------------------------
# the prop evaluator stack

def prop_stack(equity: pd.DataFrame, trades: pd.DataFrame, rules: PropRules, n_sims: int = boot.DEFAULT_N_SIMS,
               seed: int = DEFAULT_SEED, alpha: float = 0.05, history_reps: int = boot.DEFAULT_HISTORY_REPS,
               horizon_days: int | None = INFO_HORIZON_DAYS, extra_horizon: int | None = None,
               progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """The prop evaluator on one cell: the historical path (evaluate_path), the day-block bootstrap
    (flat-to-flat day blocks, n_sims, seed) at horizon_days (trading days; None = until pass or breach),
    the largest size multiplier with P(daily breach) <= alpha, the history uncertainty (history_reps outer
    replicates; < 2 skips it) and, with extra_horizon, a second bootstrap at that horizon for comparison.
    Everything uses rules.day_boundary (the firm's day, D24). Returns a JSON-serialisable dict."""
    say = progress or (lambda s: None)
    out: dict[str, Any] = {"rules_name": rules.name, "day_boundary": rules.day_boundary, "n_sims": int(n_sims),
                           "seed": int(seed), "alpha": float(alpha), "horizon_days": horizon_days,
                           "horizon_text": boot.horizon_text(horizon_days, "trading")}
    path = evaluate_path(equity, trades, rules)
    out["path"] = path.to_dict(include_days=False)
    units = boot.build_day_units(equity, rules.initial_capital, trades, rules.day_boundary)
    out["blocks"] = units.block_summary()
    say(f"bootstrap ({n_sims} challenges, {out['horizon_text']}) under {rules.name}")
    b = boot.bootstrap_challenges(None, rules, None, mode="days", n_sims=n_sims, seed=seed, horizon_days=horizon_days,
                                  units=units)
    out["bootstrap"] = b.to_dict()
    if extra_horizon is not None and extra_horizon != horizon_days:
        b2 = boot.bootstrap_challenges(None, rules, None, mode="days", n_sims=n_sims, seed=seed,
                                       horizon_days=extra_horizon, units=units)
        out["bootstrap_extra"] = b2.to_dict()
        out["bootstrap_extra_horizon_text"] = boot.horizon_text(extra_horizon, "trading")
    say(f"largest size within P(daily breach) <= {alpha:g}")
    ms = boot.max_size(None, rules, None, mode="days", alpha=alpha, n_sims=n_sims, seed=seed, horizon_days=horizon_days,
                       units=units)
    out["max_size"] = ms.to_dict()
    if history_reps and history_reps >= 2:
        say(f"history uncertainty ({history_reps} outer replicates)")
        hu = boot.history_uncertainty(None, rules, None, n_reps=int(history_reps), seed=seed,
                                      horizon_days=horizon_days, alpha=alpha, units=units, size_start=ms.multiplier)
        hu.pop("values", None)
        out["history_uncertainty"] = hu
    else:
        out["history_uncertainty"] = None
    return report_mod.clean(out)


# ---------------------------------------------------------------------------------------
# stage 2: the run

def a3_not_modelled() -> list[str]:
    """Addendum A3's not-modelled Master rules as report lines ("name: what it is")."""
    return [f"{name}: {text}" for name, text in A3_NOT_MODELLED]


def _rules_section(rules: PropRules, info: Mapping[str, Any] | None) -> dict[str, Any]:
    info = dict(info or {})
    unverified = list(info.get("unverified_fields") or [])
    firm_unverified = info.get("verified") is False
    return {"rules": rules.to_dict(), "describe": rules.describe(), "info": info,
            "unverified": bool(firm_unverified or unverified),
            "warning": info.get("warning") or (
                f"firm rules UNVERIFIED: {', '.join(unverified)} are [U] assumptions" if unverified else None),
            "u_tags": dict(info.get("u_tags") or {}), "unmodelled": list(info.get("unmodelled") or [])}


def run_stage(prep: zv.Prepared, rules: PropRules, rules_info: Mapping[str, Any] | None = None, *,
              n_sims: int = boot.DEFAULT_N_SIMS, seed: int = DEFAULT_SEED, alpha: float = 0.05,
              history_reps: int = boot.DEFAULT_HISTORY_REPS,
              reference: tuple[PropRules, Mapping[str, Any]] | None = None,
              m1: tuple[Any, Any] | None = None, g0: Mapping[str, Any] | None = None,
              inputs: Mapping[str, Any] | None = None, cells: Sequence[zv.ZenoCell] | None = None,
              master_primary: str = DEFAULT_MASTER_PRIMARY,
              progress: Callable[[str], None] | None = None) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Stage 2: every cell of the pre-registered grid (zeno_v1.grid_cells; cells= only for tests), the
    metrics, the judging cell, the prop evaluator (the judging cell, its master twin and its master_fp twin
    under `rules`; the judging cell again with the firm day moved to 00:00 UTC+3, addendum A4; the judging
    cell under reference rules if given), the gates (G4 per addendum A5) and, with m1 = (bid, ask) M1 files or
    frames, the D15 second run of the judging cell beside the first. The capital is rules.initial_capital.
    The master_fp cells need prep built with the restricted calendar (zeno_v1.read_restricted_csv).
    master_primary: "master_fp" (default) or "master", which Master run is the primary Master result (A1).

    Returns (report, tables): report holds every section of report.md and report.json, with report["gates"]
    the gates.json content; tables: trades (the judging cell's legs), positions, decisions (judging cell),
    grid, positions_all_cells and, with m1, m1_diff."""
    say = progress or (lambda s: None)
    t_start = time.perf_counter()
    if master_primary not in MASTER_PRIMARY_CHOICES:
        raise ValueError(f"master_primary must be one of {MASTER_PRIMARY_CHOICES}, got {master_primary!r}")
    c0 = float(rules.initial_capital)
    periods = period_table(prep)
    cells = list(cells) if cells is not None else zv.grid_cells()
    if prep.restricted is None and any(c.variant == "master_fp" for c in cells):
        raise ValueError("the master_fp cells need FundingPips' restricted calendar: prepare(frame, news, "
                         "restricted=zeno_v1.read_restricted_csv()) (addendum A1)")
    rows: list[dict[str, Any]] = []
    results: dict[str, zv.CellResult] = {}
    all_pos = []
    cache: dict = {}
    for i, cell in enumerate(cells, 1):
        res = zv.simulate(prep, zv.ZenoConfig(cell, c0))
        rows += cell_rows(prep, res, periods, cache)
        results[cell.label] = res
        all_pos.append(res.positions)
        say(f"cell {i}/{len(cells)} {cell.label}: {res.meta['n_positions']} positions")
    grid = grid_frame(rows)
    j = worse_base(grid)
    jcell = zv.ZenoCell(JUDGING_VARIANT, JUDGING_COMMISSION, j["spread_base"], JUDGING_COST_MULT)
    twin = zv.ZenoCell("master", JUDGING_COMMISSION, j["spread_base"], JUDGING_COST_MULT)
    twin_fp = zv.ZenoCell("master_fp", JUDGING_COMMISSION, j["spread_base"], JUDGING_COST_MULT)
    if jcell.label not in results:
        raise ValueError(f"the judging cell {jcell.label} is not among the cells run")
    jres = results[jcell.label]
    applicable, reason = g4_applicability(rules, rules_info)
    horizon = None if applicable else INFO_HORIZON_DAYS
    prop: dict[str, Any] = {"g4_applicable": applicable, "g4_reason": reason,
                            "horizon_reading": ("G4: until pass or breach (no horizon), 60 trading days beside it "
                                                "[SI-31]" if applicable else
                                                "60 trading days, an illustration only: G4 is not evaluated [SI-55]")}
    jeq, jtr = zv.cell_equity(prep, jres)
    say(f"prop evaluator: judging cell {jcell.label}")
    prop["judging"] = {"cell": jcell.label, **prop_stack(jeq, jtr, rules, n_sims, seed, alpha, history_reps, horizon,
                                                          INFO_HORIZON_DAYS if applicable else None, say)}
    # addendum A4: the same judging cell under the same rules with the other firm day (00:00 UTC+3 = 21:00 UTC)
    alt = "ny_17" if rules.day_boundary == SENSITIVITY_DAY_BOUNDARY else SENSITIVITY_DAY_BOUNDARY
    rules_alt = dataclasses.replace(rules, day_boundary=alt,
                                    name=f"{rules.name} [firm day {calendar.boundary_label(alt)}: addendum A4 "
                                         "sensitivity]")
    say(f"prop evaluator: judging cell {jcell.label} with the firm day at {calendar.boundary_label(alt)} (A4)")
    prop["judging_day_sensitivity"] = {"cell": jcell.label, "day_boundary_default": rules.day_boundary,
                                       **prop_stack(jeq, jtr, rules_alt, n_sims, seed, alpha, history_reps, horizon,
                                                    INFO_HORIZON_DAYS if applicable else None, say)}
    for key, cell in (("master_twin", twin), ("master_fp_twin", twin_fp)):
        if cell.label in results:
            say(f"prop evaluator: {key.replace('_', ' ')} {cell.label}")
            teq, ttr = zv.cell_equity(prep, results[cell.label])
            prop[key] = {"cell": cell.label, **prop_stack(teq, ttr, rules, n_sims, seed, alpha, history_reps,
                                                          horizon, None, say)}
    if reference is not None:
        rrules, rinfo = reference
        say(f"prop evaluator: judging cell under the reference rules {rrules.name}")
        prop["reference"] = {"cell": jcell.label, "rules_section": _rules_section(rrules, rinfo),
                             **prop_stack(jeq, jtr, rrules, n_sims, seed, alpha, history_reps, INFO_HORIZON_DAYS,
                                          None, say)}
    g4 = g4_gate(applicable, reason, prop["judging"]["bootstrap"] if applicable else None, jcell.label,
                 sensitivity=(alt, prop["judging_day_sensitivity"]["bootstrap"]), day_boundary=rules.day_boundary)
    gates = gates_from(grid, g4, g0)
    restricted = restricted_info(prep)
    gates["addendum_a"] = addendum_identity()
    gates["restricted_calendar"] = ({"file": restricted.get("source"), "sha256": restricted.get("sha256"),
                                     "is_packaged_file": restricted.get("is_packaged_file")} if restricted else None)
    roles = MASTER_ROLES[master_primary]
    gates["master_primary"] = {"primary": master_primary, "roles": {k: v for k, v in roles.items() if k != "why"},
                               "why": roles["why"], "gated": MASTER_NOT_GATED}
    gates["day_boundaries"] = day_boundary_summary(prop, rules.day_boundary, alt, g4)
    tables: dict[str, pd.DataFrame] = {
        "trades": jres.legs, "positions": jres.positions, "decisions": jres.decisions, "grid": grid,
        "positions_all_cells": pd.concat(all_pos, ignore_index=True) if all_pos else pd.DataFrame()}
    m1_section: Any = zv.M1_NOT_RUN
    if m1 is not None:
        say("M1 resolution of the judging cell's ambiguous bars (D15 second run)")
        mres, diff = zv.resolve_with_m1(prep, zv.ZenoConfig(jcell, c0), m1[0], m1[1], base=jres)
        tables["m1_diff"] = diff
        m1_rows = cell_rows(prep, mres, [p for p in periods if p["kind"] in ("all", "g3")], cache)
        m1_section = {"run": True, "meta": mres.meta.get("m1"),
                      "status_counts": _counts(diff["status"]) if len(diff) else {},
                      "combined_all_m15": _grid_row(grid, jcell.variant, jcell.commission_rt_per_lot,
                                                    jcell.spread_base, jcell.cost_mult, "combined", "all"),
                      "combined_all_m1": next(r for r in m1_rows if r["side"] == "combined" and r["period"] == "all"),
                      "rows": m1_rows,
                      "note": "the gates are judged on the first (M15) run; this second run is reported beside it "
                              "(D15)"}
    daily_alt = {}
    for key in ("cet_midnight", "utc_midnight", DAY_KEY, SENSITIVITY_DAY_BOUNDARY):
        d = st.daily_returns_from_equity(jeq, c0, by=key)
        daily_alt[key] = daily_metrics(d, c0)
    trig = jres.decisions[jres.decisions["event"] == "trigger"]
    rules_sec = _rules_section(rules, rules_info)
    rules_sec["not_modelled_addendum_a3"] = a3_not_modelled()           # addendum A3
    rules_sec["unmodelled_master_note"] = (UNMODELLED_MASTER_NOTE if any(MASTER_CLAIM in str(x)
                                                                         for x in rules_sec["unmodelled"]) else None)
    report = {
        "header": HEADER, "stage": "run (stage 2, the pre-registered grid)",
        "generated_utc": calendar.utc_str(int(time.time())), "propkit_version": propkit.__version__,
        "spec": spec_identity(), "addendum_a": addendum_identity(), "inputs": dict(inputs or {}),
        "data": data_info(prep), "news": news_info(prep), "restricted": restricted,
        "news_unscheduled": {"judging": jres.meta["news_unscheduled"],
                             "master_twin": results[twin.label].meta["news_unscheduled"]
                             if twin.label in results else None,
                             "master_fp_twin": results[twin_fp.label].meta["news_unscheduled"]
                             if twin_fp.label in results else None, "note": UNSCHEDULED_NOTE},
        "rules_section": rules_sec,
        "settings": {"capital_usd": c0, "n_sims": int(n_sims), "seed": int(seed), "alpha": float(alpha),
                     "history_reps": int(history_reps), "n_cells": len(cells), "day_key_daily_returns": DAY_KEY,
                     "risk_pct": {v: zv.RISK_PCT[v] for v in zv.VARIANTS}, "n_trials": 1,
                     "master_primary": master_primary, "day_boundary_sensitivity": alt},
        "costs": {"grid": {"variants": list(zv.VARIANTS), "commissions_rt_per_lot": list(zv.COMMISSIONS),
                           "spread_bases": list(zv.SPREAD_BASES), "cost_mults": list(zv.COST_MULTS)},
                  "s2_disclosure": S2_DISCLOSURE, "slippage": SLIPPAGE_NOTE,
                  "s2_text": "S2 = bid + 0.18 USD/oz, 0.20 USD/oz from 05:00 to 08:00 SGT (21:00-24:00 UTC)"},
        "periods": periods, "judging": gates["judging_cell"], "x1_cell": gates["x1_cell"],
        "judging_rows": [r for r in rows if r["cell"] == jcell.label],
        "risk_adjusted": {"cell": jcell.label, "by_day_key": daily_alt, "dsr_caveat": DSR_CAVEAT, "n_trials": 1},
        "prop": prop, "gates": gates,
        "master": master_section(grid, results, (twin, twin_fp), prop, master_primary),
        "margin": margin_section(results, cells, (jcell, twin, twin_fp)),
        "decision_log": {"cell": jcell.label, "n_triggers": int(len(trig)),
                         "n_entered": int((trig["status"] == "entered").sum()),
                         "status_counts": _counts(trig["status"]), "block_reasons": dict(zv.BLOCK_REASONS),
                         "time_exits": jres.meta["time_exits"]},
        "grid_summary": grid[(grid["side"] == "combined") & (grid["period"] == "all")].to_dict("records"),
        "m1": m1_section, "spec_readings": [{"id": a, "text": b} for a, b in SPEC_READINGS],
        "notes": {"after_a_pass": AFTER_A_PASS, "change_policy": CHANGE_POLICY, "kill": KILL_TEXT},
        "seconds": round(time.perf_counter() - t_start, 1),
    }
    return report_mod.clean(report), tables


def day_boundary_summary(prop: Mapping[str, Any], own: str, alt: str, g4: Mapping[str, Any]) -> dict[str, Any]:
    """Addendum A4-A5 in one place: per firm-day boundary, the judging cell's bootstrap P(daily-loss breach),
    P(max-loss breach) and P(pass), and which boundary set G4's P(daily-loss breach) (None when G4 is not
    evaluated)."""
    out: dict[str, Any] = {"default": own, "sensitivity": alt, "by_boundary": {}}
    for b, key in ((own, "judging"), (alt, "judging_day_sensitivity")):
        bs = (prop.get(key) or {}).get("bootstrap") or {}
        out["by_boundary"][b] = {"label": calendar.boundary_label(b), "utc": calendar.boundary_utc_text(b),
                                 "p_breach_daily": bs.get("p_breach_daily"),
                                 "se_breach_daily": bs.get("se_breach_daily"),
                                 "p_breach_max": bs.get("p_breach_max"), "p_pass": bs.get("p_pass"),
                                 "n_sims": bs.get("n_sims"), "horizon_days": bs.get("horizon_days")}
    v = g4.get("value") if isinstance(g4, Mapping) else None
    out["g4_p_breach_daily_set_by"] = v.get("p_breach_daily_set_by") if isinstance(v, Mapping) else None
    out["g4_status"] = g4.get("status") if isinstance(g4, Mapping) else None
    return out


def master_section(grid: pd.DataFrame, results: Mapping[str, zv.CellResult], twins: Sequence[zv.ZenoCell],
                   prop: Mapping[str, Any], primary: str) -> dict[str, Any]:
    """The two Master runs at the judging settings (addendum A1): role (per --master-primary), combined results
    (all data), the Master closes, the restricted-window blocks (master_fp), the margin cap (A2) and the
    bootstrap breach and pass probabilities under the rules."""
    roles = MASTER_ROLES[primary]
    out: dict[str, Any] = {"primary": primary, "why": roles["why"], "gated": MASTER_NOT_GATED, "runs": {},
                           "not_modelled_addendum_a3": a3_not_modelled()}
    for cell in twins:
        res = results.get(cell.label)
        if res is None:
            continue
        r = _grid_row(grid, cell.variant, cell.commission_rt_per_lot, cell.spread_base, cell.cost_mult, "combined",
                      "all") or {}
        key = "master_twin" if cell.variant == "master" else "master_fp_twin"
        bs = (prop.get(key) or {}).get("bootstrap") or {}
        out["runs"][cell.variant] = {
            "cell": cell.label, "role": roles[cell.variant], "n_triggers": r.get("n_triggers"),
            "n_positions": r.get("n_positions"), "expectancy_r": r.get("expectancy_r"),
            "se_expectancy_r": r.get("se_expectancy_r"), "net_usd": r.get("net_usd"), "psr_0": r.get("psr_0"),
            "master_closes": res.meta.get("master_closes"), "restricted": res.meta.get("restricted"),
            "margin": res.meta.get("margin"), "p_breach_daily": bs.get("p_breach_daily"),
            "p_breach_max": bs.get("p_breach_max"), "p_pass": bs.get("p_pass")}
    return out


def margin_section(results: Mapping[str, zv.CellResult], cells: Sequence[zv.ZenoCell],
                   named: Sequence[zv.ZenoCell]) -> dict[str, Any]:
    """Addendum A2 counts: per named cell (the judging cell and its two Master twins) its meta["margin"], and per
    variant the totals over the grid's cells (Master: capped entries, lots before and after, blocked triggers;
    evaluation: entries over the margin at a flat 1:10 and 1:30)."""
    per_cell = {c.label: results[c.label].meta.get("margin") for c in named if c.label in results}
    totals: dict[str, dict[str, Any]] = {}
    for c in cells:
        m = results[c.label].meta.get("margin") or {}
        t = totals.setdefault(c.variant, {"n_cells": 0, "n_entries": 0})
        t["n_cells"] += 1
        t["n_entries"] += int(m.get("n_entries") or 0)
        for k in ("n_capped", "n_blocked", "n_over_flat_1to10", "n_over_flat_1to30"):
            if k in m:
                t[k] = t.get(k, 0) + int(m[k])
        for k in ("lots_before_cap", "lots_after_cap"):
            if k in m:
                t[k] = round(t.get(k, 0.0) + float(m[k]), 6)
    return {"per_cell": per_cell, "grid_totals": totals,
            "tiers": "0.05 lot at 1:50, the next 0.05 at 1:30, the next 0.05 at 1:25, the next 0.10 at 1:20, the "
                     "next 0.25 at 1:10, the rest at 1:5; per position, at the entry fill price, 100 oz per lot "
                     "(addendum A2)"}


# ---------------------------------------------------------------------------------------
# report.md

def _pct(x, nd: int = 1, na: str = "n/a") -> str:
    f = _num(x)
    return na if f is None else f"{100.0 * f:.{nd}f}%"


def _r(x, se=None, nd: int = 3) -> str:
    f = _num(x)
    if f is None:
        return "n/a"
    s = _num(se)
    return f"{f:+.{nd}f}" + ("" if s is None else f" +- {s:.{nd}f}")


def _usd0(x) -> str:
    f = _num(x)
    return "n/a" if f is None else f"{f:,.0f}"


def _row_cells(r: Mapping[str, Any]) -> list[str]:
    return [str(r.get("n_positions")), report_mod._fmt(r.get("trades_per_year"), 1), _pct(r.get("hit_rate_tp1")),
            _pct(r.get("hit_rate_tp2_after_tp1")), _pct(r.get("win_rate")),
            _r(r.get("expectancy_r"), r.get("se_expectancy_r")), _usd0(r.get("expectancy_usd")), _usd0(r.get("net_usd")),
            str(r.get("longest_losing_streak")), _r(r.get("sharpe_daily"), r.get("se_sharpe_daily"), 4),
            report_mod._fmt(r.get("psr_0"), 3)]


RESULT_HEADER = ["side", "period", "positions", "positions per year", "+2R hit rate", "+4R rate after +2R",
                 "win rate", "E[R] +- SE (R)",
                 "E[USD]/position", "net USD", "longest losing streak (positions)", "daily SR +- SE", "PSR(SR>0)"]


def _boot_line(b: Mapping[str, Any] | None, label: str, has_target: bool) -> list[str]:
    if not b:
        return []
    n = b.get("n_sims")
    q = b.get("days_to_target_q") or {}
    out = [f"| {label}: P(daily-loss breach) | {report_mod._pse(b['p_breach_daily'], b['se_breach_daily'], n)} |",
           f"| {label}: P(max-loss breach) | {report_mod._pse(b['p_breach_max'], b['se_breach_max'], n)}"
           + ("" if has_target or _num(b.get('p_breach_max')) else " (no max-loss rule: 0 by construction, not a result)")
           + " |"]
    if has_target:
        out += [f"| {label}: P(pass) | {report_mod._pse(b['p_pass'], b['se_pass'], n)} |",
                f"| {label}: market days to target p10/p50/p90 | "
                + ("/".join(report_mod._fmt(q.get(k), 1) for k in ("p10", "p50", "p90")) if q else "n/a (no pass)")
                + " |"]
    else:
        out.append(f"| {label}: P(pass), days to target | not defined: these rules have no profit target |")
    out.append(f"| {label}: P(timeout, no event in the horizon) | {report_mod._pse(b['p_timeout'], b['se_timeout'], n)} |")
    return out


def _prop_block(title: str, p: Mapping[str, Any] | None, rules: Mapping[str, Any]) -> list[str]:
    if not p:
        return []
    has_target = rules.get("profit_target_pct") is not None
    path = p.get("path") or {}
    ms = p.get("max_size") or {}
    hu = p.get("history_uncertainty") or {}
    L = [f"### {title} ({p.get('cell')}) under {p.get('rules_name')}", "",
         f"Firm day: {calendar.boundary_label(p.get('day_boundary') or 'cet_midnight')}; {p.get('n_sims')} "
         f"challenges per bootstrap, seed {p.get('seed')}, horizon {p.get('horizon_text')}, flat-to-flat day blocks "
         f"({(p.get('blocks') or {}).get('n_blocks_days', 'n/a')} blocks).", "",
         "| item | value |", "|---|---|",
         f"| historical path | {str(path.get('status', 'n/a')).upper()}, final balance {_usd0(path.get('final_balance'))} "
         f"USD, max drawdown {report_mod._fmt(path.get('max_dd_pct'), 2)}% of initial capital, worst firm-day loss "
         f"{report_mod._fmt(path.get('worst_daily_dd_pct'), 2)}% of initial capital |"]
    L += _boot_line(p.get("bootstrap"), "bootstrap", has_target)
    if p.get("bootstrap_extra"):
        L += _boot_line(p["bootstrap_extra"], f"same, {p.get('bootstrap_extra_horizon_text')}", has_target)
    L.append(f"| largest size multiplier with P(daily breach) <= {p.get('alpha')} | "
             f"{report_mod._fmt(ms.get('multiplier'), 3)} x this run's size ({ms.get('note')})"
             + (f"; 5-95% over resampled histories {report_mod._fmt((hu.get('max_size_multiplier') or {}).get('p5'), 2)}"
                f" .. {report_mod._fmt((hu.get('max_size_multiplier') or {}).get('p95'), 2)}"
                if hu.get("max_size_multiplier") else "")
             + ("; the addendum A2 margin cap is not applied to this scaling, so above about 1.6 lots at "
                "USD 4,000 gold the multiplier overstates what the Master account can hold"
                if str(p.get("cell") or "").split("/")[0] in zv.MASTER_VARIANTS else "") + " |")
    if hu.get("p_breach_daily"):
        q = hu["p_breach_daily"]
        L.append(f"| P(daily-loss breach), 5-95% over {hu.get('n_reps')} resampled histories | "
                 f"{report_mod._fmt(q.get('p5'), 4)} .. {report_mod._fmt(q.get('p95'), 4)} |")
    return L + [""]


def _rules_header(rs: Mapping[str, Any], info: Mapping[str, Any]) -> list[str]:
    """The rules header of report.md: every [U] tag of the rules file and its "unmodelled" list (addendum A3)."""
    L: list[str] = []
    u = rs.get("u_tags") or info.get("u_tags") or {}
    um = rs.get("unmodelled") or info.get("unmodelled") or []
    if u:
        L += [f"Rules fields with an unverified [U] part ({len(u)}; that part is an assumption):", ""]
        L += [f"- {f}: {t}" for f, t in u.items()]
        L.append("")
    if um:
        L += [f"Not modelled by the prop evaluator ({len(um)}, from the rules file):", ""]
        L += _unmodelled_lines(um, rs.get("unmodelled_master_note"))
        L.append("")
    if rs.get("not_modelled_addendum_a3"):
        L += _a3_lines(rs) + [""]
    return L


def _unmodelled_lines(um: Sequence[Any], note: str | None) -> list[str]:
    """The rules file's "unmodelled" items as bullets; the item that says its Master rules are handled by the
    strategy's Master variant carries the report's note on what the code does with each of them."""
    return [f"- {x}" + (f" {note}" if note and MASTER_CLAIM in str(x) else "") for x in um]


def _a3_lines(rs: Mapping[str, Any]) -> list[str]:
    """Addendum A3's Master rules that no run simulates."""
    return ["Not simulated in any run (addendum A3):", ""] + [f"- {x}" for x in rs.get("not_modelled_addendum_a3") or []]


def _master_lines(m: Mapping[str, Any] | None) -> list[str]:
    """report.md section on the two Master runs (addendum A1, A2)."""
    if not isinstance(m, Mapping) or not m.get("runs"):
        return []
    L = ["## Master runs (addendum A1)", "",
         f"Primary Master result: {m['primary']} ({m['why']}). {m['gated']} Both runs: risk 0.40% per trade, rule 9's "
         "close 10 min before NFP, CPI, PPI and FOMC for positions opened under 5 h before it (D23), the D20 blackout, "
         "and the margin cap (A2); master_fp also blocks entries in FundingPips' restricted windows and closes "
         "before every restricted event with a time.", ""]
    a3 = [str(x).split(": ", 1)[0] for x in m.get("not_modelled_addendum_a3") or []]
    if a3:
        L += [f"Neither Master run simulates (addendum A3): {', '.join(a3)}. The rules header lists them.", ""]
    L += ["| run | role | cell | triggers | positions | E[R] +- SE (R) | net USD | Master closes (restricted list "
          "only) | triggers blocked by the restricted window (only by it) | margin-capped entries (lots before -> "
          "after) | P(daily) | P(max) | P(pass) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in ("master_fp", "master"):
        x = (m.get("runs") or {}).get(v)
        if not x:
            continue
        mc = x.get("master_closes") or {}
        rs = x.get("restricted") or {}
        mg = x.get("margin") or {}
        closes = f"{mc.get('n_positions_closed', 'n/a')}" + (
            f" ({mc.get('n_closed_only_for_restricted_list')})" if "n_closed_only_for_restricted_list" in mc else "")
        blocks = (f"{rs.get('n_triggers_blocked')} ({rs.get('n_triggers_blocked_only_by_it')})" if rs else "n/a")
        cap = (f"{mg.get('n_capped', 0)} ({report_mod._fmt(mg.get('lots_before_cap'), 2)} -> "
               f"{report_mod._fmt(mg.get('lots_after_cap'), 2)}); {mg.get('n_blocked', 0)} blocked")
        fmt = report_mod._fmt
        L.append(f"| {v} | {x['role']} | {x['cell']} | {x.get('n_triggers')} | {x.get('n_positions')} | "
                 f"{_r(x.get('expectancy_r'), x.get('se_expectancy_r'))} | {_usd0(x.get('net_usd'))} | {closes} | "
                 f"{blocks} | {cap} | {fmt(x.get('p_breach_daily'), 4)} | {fmt(x.get('p_breach_max'), 4)} | "
                 f"{fmt(x.get('p_pass'), 4)} |")
    return L + ["", "P(daily), P(max), P(pass): the bootstrap under the rules above (the Prop evaluator section has "
                "the full blocks).", ""]


def _margin_lines(mg: Mapping[str, Any] | None) -> list[str]:
    """report.md section on the margin cap (addendum A2)."""
    if not isinstance(mg, Mapping):
        return []
    L = ["## Margin (addendum A2)", "", f"Master dynamic leverage: {mg.get('tiers')}. Capped: the D13 lots needed more "
         "margin than the closed balance at entry and were cut to the largest 0.01-lot size that fits; blocked: not "
         "even 0.01 lot fits (margin_cap_below_lot_step). The evaluation run is not capped (Standard 1:30 assumed); "
         "it counts the entries whose margin at a flat 1:10 or 1:30 would exceed the closed balance.", "",
         "| cell | entries | margin-capped entries | lots before -> after the cap | blocked by margin | over margin "
         "at flat 1:10 | over margin at flat 1:30 |", "|---|---|---|---|---|---|---|"]

    def line(label: str, m: Mapping[str, Any]) -> str:
        if m.get("cap_applies", "n_capped" in m):
            fmt = report_mod._fmt
            return (f"| {label} | {m.get('n_entries')} | {m.get('n_capped', 0)} | "
                    f"{fmt(m.get('lots_before_cap'), 2)} -> {fmt(m.get('lots_after_cap'), 2)} | "
                    f"{m.get('n_blocked', 0)} | not counted (capped) | not counted (capped) |")
        return (f"| {label} | {m.get('n_entries')} | not capped | n/a | n/a | {m.get('n_over_flat_1to10', 0)} | "
                f"{m.get('n_over_flat_1to30', 0)} |")
    for label, m in (mg.get("per_cell") or {}).items():
        L.append(line(label, m or {}))
    for v, t in (mg.get("grid_totals") or {}).items():
        L.append(line(f"all {t.get('n_cells')} {v} cells", {**t, "cap_applies": v in zv.MASTER_VARIANTS}))
    return L + [""]


def _day_boundary_lines(db: Mapping[str, Any] | None) -> list[str]:
    """report.md section on the two firm-day boundaries (addendum A4, A5)."""
    if not isinstance(db, Mapping) or not db.get("by_boundary"):
        return []
    L = ["## Firm day boundary (addendum A4, A5)", "",
         "The judging cell's prop evaluator runs under the rules' own firm day and again with the other one "
         "(00:00 UTC+3 = 21:00 UTC all year, FundingPips' literal \"00:00 Platform Time (UTC+3)\"; or 17:00 New York "
         "when the rules already use UTC+3). G4 uses the higher P(daily-loss breach) of the two.", "",
         "| firm day | role | starts (UTC) | P(daily-loss breach) | P(max-loss breach) | P(pass) |",
         "|---|---|---|---|---|---|"]
    for b, x in db["by_boundary"].items():
        role = "default (the rules')" if b == db.get("default") else "sensitivity"
        pd_ = (report_mod._pse(x["p_breach_daily"], x["se_breach_daily"], x.get("n_sims"))
               if x.get("p_breach_daily") is not None else "n/a")
        L.append(f"| {b} ({x.get('label')}) | {role} | {x.get('utc')} | {pd_} | "
                 f"{report_mod._fmt(x.get('p_breach_max'), 4)} | {report_mod._fmt(x.get('p_pass'), 4)} |")
    sb = db.get("g4_p_breach_daily_set_by")
    L += ["", (f"G4's P(daily-loss breach) is set by {sb}." if sb else
               f"G4 is {db.get('g4_status') or 'not evaluated'}, so no boundary set it; both values are shown."), ""]
    return L


def render_run_markdown(report: Mapping[str, Any]) -> str:
    """The stage-2 report (run_stage) as Markdown text (ASCII): verdict and gates first, then the firm-day
    boundaries (A4-A5), the judging cell's results, the risk-adjusted block (with the DSR caveat), the prop
    evaluator, the Master runs (A1), the margin cap (A2), the grid, the decision log, M1, data, rules ([U]
    fields, unmodelled rules), the spec readings and the notes. Every number carries its unit."""
    r = report
    g = r["gates"]
    gg = g["gates"]
    rs = r["rules_section"]
    rules = rs["rules"]
    info = rs.get("info") or {}
    d = r["data"]
    j = r["judging"]
    L = [HEADER, "", "# zeno_pullback_v1 - stage 2 report (pre-registered grid)", "",
         f"Generated {r['generated_utc']} by propkit {r['propkit_version']} in {r.get('seconds')} s. Spec "
         f"{r['spec']['spec_id']} v{r['spec']['spec_version']} (packaged copies match the recorded sha256: "
         f"{'yes' if r['spec']['all_match'] else 'NO'})"
         + (f" with addendum A ({r['addendum_a']['file']}, sha256 {r['addendum_a']['sha256_packaged']}, matches the "
            f"recorded value: {'yes' if r['addendum_a']['matches'] else 'NO'})" if r.get("addendum_a") else "")
         + f". Data: {d.get('n_bars')} M15 bid/ask bars, "
         f"{d.get('first_time_utc')} .. {d.get('last_time_utc')}, {d.get('n_trading_days')} server days. The numbers "
         "describe this data only; the backtest is in-sample by construction (spec, After a pass).", ""]
    L += [f"Rules: {rules.get('name')} ({info.get('status') or 'status not recorded'}"
          + (f"; file sha256 {info.get('sha256')}" if info.get("sha256") else "") + ").", ""]
    if rs.get("unverified"):
        L += [f"**WARNING: {info.get('warning') or rs.get('warning')}** Unverified [U] fields: "
              f"{', '.join(info.get('unverified_fields') or []) or 'see Rules'}.", ""]
        if info.get("fallback"):
            L += [f"NOTE: {info['fallback']}.", ""]
    L += _rules_header(rs, info)
    L += ["## Verdict and gates", "", f"**Verdict: {g['verdict']}**", "",
          "| gate | judged at | value | threshold | status |", "|---|---|---|---|---|"]
    g0 = gg["G0"]
    L.append(f"| G0 signal check | operator | {'confirmed with --g0-confirmed' if g0.get('confirmed_by_operator') else 'no'}"
             + (f"; sample sha256 {str(g0.get('sample_sha256'))[:12]}..." if g0.get("sample_sha256") else "")
             + (f"; {g0.get('agree_count')} of {g0.get('answered')} answered y" if g0.get("answered") else "")
             + ("; every sampled signal is an eligible signal of this data with the same entry and stop [SI-69]"
                if (g0.get("sample_check") or {}).get("ok") and (g0.get("sample_check") or {}).get("n_rows") else "")
             + f" | zeno agrees with >= {G0_MIN_AGREE} of {G0_SAMPLE_SIZE} | {g0.get('status')} |")
    v1, v2, v3 = gg["G1"], gg["G2"], gg["G3"]
    L.append(f"| G1 sample | {v1['judged_at']} | {v1['value']} positions | {v1['threshold']} | {v1['status']} |")
    L.append(f"| G2 edge | {v2['judged_at']} | E[R] {_r(v2['value']['expectancy_r'], v2['value']['se_expectancy_r'])} R "
             f"per position; PSR {report_mod._fmt(v2['value']['psr_0'], 3)} | {v2['threshold']} | {v2['status']} |")
    per = v3["value"]["per_period"]
    L.append(f"| G3 stability | {v3['judged_at']} | "
             + "; ".join(f"{k}: {_r(v['expectancy_r'])} R ({v['n_positions']} pos.)" for k, v in per.items())
             + f" | {v3['threshold']} | {v3['status']} |")
    v4 = gg["G4"]
    if v4["status"] == "not_evaluated":
        L.append(f"| G4 prop survival | {v4['judged_at']} | not evaluated: {v4.get('reason')} | {v4['threshold']} | "
                 "not_evaluated |")
    else:
        vv = v4["value"]
        both = vv.get("p_breach_daily_by_boundary") or {}
        L.append(f"| G4 prop survival | {v4['judged_at']} | P(daily) {report_mod._fmt(vv['p_breach_daily'], 4)}"
                 + (" (" + ", ".join(f"{b} {report_mod._fmt(x, 4)}" for b, x in both.items())
                    + f"; set by {vv.get('p_breach_daily_set_by')})" if both else "")
                 + f", P(max) {report_mod._fmt(vv['p_breach_max'], 4)} ({vv.get('n_sims')} sims, no horizon) | "
                 f"{v4['threshold']} | {v4['status']} |")
    v5 = gg["G5"]
    L.append(f"| G5 each side | {v5['judged_at']} | "
             + "; ".join(f"{s}: {_r(v['expectancy_r'])} R ({v['n_positions']} pos.)" for s, v in v5["value"].items())
             + f" | {v5['threshold']} | {v5['status']} |")
    k = g["kill"]
    L.append(f"| Kill (G2 at x1) | {k['judged_at']} | E[R] {_r(k['value']['expectancy_r'])} R, PSR "
             f"{report_mod._fmt(k['value']['psr_0'], 3)} | {k['test']} | {'FIRED' if k['fired'] else 'not fired'} |")
    L += ["", f"Kill rule (spec): {KILL_TEXT} {'It FIRED.' if k['fired'] else 'It did not fire.'}", ""]
    L += _day_boundary_lines(g.get("day_boundaries"))
    wb, wb1 = j["worse_base"], r["x1_cell"]["worse_base"]
    L += ["## Judging cell", "",
          f"{j['label']}: variant evaluation (risk 0.50% per trade), commission 10 USD per lot round trip, costs "
          f"x1.5, spread base {j['spread_base']} = the worse base by combined net expectancy (S1 "
          f"{_r(wb['expectancy_r'].get('S1'))} R, S2 {_r(wb['expectancy_r'].get('S2'))} R per position; a tie goes to "
          f"S2) [SI-28]. At costs x1 the worse base is {wb1['spread_base']} (S1 {_r(wb1['expectancy_r'].get('S1'))} R, "
          f"S2 {_r(wb1['expectancy_r'].get('S2'))} R); G5 and the kill rule use it [SI-52].", "",
          "## Results at the judging cell", "",
          "Net of every cost. E[R] in R per position (R = the spec's stop distance x size); USD per position and net "
          "USD on a " + f"{_usd0(r['settings']['capital_usd'])} USD account; rates are shares of positions; daily SR "
          "per server day (17:00 New York); positions belong to the server day of their entry [SI-42].", ""]
    L += ["| " + " | ".join(RESULT_HEADER) + " |", "|" + "|".join("---" for _ in RESULT_HEADER) + "|"]
    order = {"all": 0, "g3": 1, "year": 2}
    jr = sorted(r["judging_rows"], key=lambda x: (SIDES.index(x["side"]) if x["side"] in SIDES else 9,
                                                  order.get(x["period_kind"], 3), x["period"]))
    for x in jr:
        if x["side"] != "combined" and x["period_kind"] == "year":
            continue
        L.append("| " + " | ".join([x["side"], x["period"]] + _row_cells(x)) + " |")
    L += ["", "Per-year rows for longs and shorts alone are in grid.csv.", ""]
    ra = r["risk_adjusted"]
    L += ["## Risk-adjusted (judging cell, combined)", "", "| day used for daily returns | days | SR per day +- SE | "
          "annualised SR +- SE | PSR(SR>0) | DSR (N = 1) |", "|---|---|---|---|---|---|"]
    for key, label in ((DAY_KEY, "server day, 17:00 New York (judged) [SI-30]"), ("cet_midnight", "CE(S)T day"),
                       ("utc_midnight", "UTC day"), (SENSITIVITY_DAY_BOUNDARY, "UTC+3 day, 21:00 UTC (addendum A4)")):
        m = ra["by_day_key"].get(key)
        if m is None:
            continue
        L.append(f"| {label} | {m['n_days']} | {_r(m['sharpe_daily'], m['se_sharpe_daily'], 4)} | "
                 f"{_r(m['sharpe_annual'], m['se_sharpe_annual'], 2)} at {report_mod._fmt(m['days_per_year'], 1)} "
                 f"days/yr | {report_mod._fmt(m['psr_0'], 3)} | {report_mod._fmt(m['dsr_n1'], 3)} |")
    L += ["", f"Caveat: {DSR_CAVEAT}", ""]
    pr = r["prop"]
    L += ["## Prop evaluator", "", f"Horizon: {pr['horizon_reading']}. +- is the Monte Carlo error (simulation noise "
          "only); at 0 or 1 the bound is the 95% rule of three.", ""]
    L += _prop_block("Judging cell", pr.get("judging"), rules)
    L += _prop_block("Judging cell, the other firm day (addendum A4)", pr.get("judging_day_sensitivity"), rules)
    roles = MASTER_ROLES.get((r.get("master") or {}).get("primary") or DEFAULT_MASTER_PRIMARY, {})
    L += _prop_block(f"Master twin (risk 0.40%, rule 9's Master news rule; {roles.get('master', 'master')})",
                     pr.get("master_twin"), rules)
    L += _prop_block(f"master_fp twin (risk 0.40%, FundingPips' restricted list, addendum A1; "
                     f"{roles.get('master_fp', 'master_fp')})", pr.get("master_fp_twin"), rules)
    if pr.get("reference"):
        rr = pr["reference"]["rules_section"]
        L += [f"Reference rules: {rr['rules'].get('name')} - another firm's rules, shown for comparison only.", ""]
        L += _prop_block("Judging cell, reference rules", pr["reference"], rr["rules"])
    L += _master_lines(r.get("master"))
    L += _margin_lines(r.get("margin"))
    L += [f"## The {len(r['grid_summary'])} cells (combined, all data)", "",
          f"Costs: {SLIPPAGE_NOTE}. S2 disclosure (spec): {S2_DISCLOSURE}",
          "", "The cost multiplier also scales the spread that rule 10 compares with 10% of R, so x1.5 and x2 cells "
          "trade fewer triggers than x1 [SI-65]: 'spread blocks' counts the triggers the spread filter blocks, "
          "'vs x1' the positions against the same cell at x1.", "",
          "| cell | triggers | spread blocks | positions | vs x1 | E[R] +- SE (R) | net USD | daily SR | PSR | "
          "max DD (% of capital, server-day closes) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for x in r["grid_summary"]:
        vs = _num(x.get("positions_vs_x1"))
        L.append(f"| {x['cell']} | {x.get('n_triggers', 'n/a')} | {x.get('n_spread_blocked', 'n/a')} | "
                 f"{x['n_positions']} | {'n/a' if vs is None else f'{int(vs):+d}'} | "
                 f"{_r(x['expectancy_r'], x['se_expectancy_r'])} | "
                 f"{_usd0(x['net_usd'])} | {_r(x['sharpe_daily'], None, 4)} | {report_mod._fmt(x['psr_0'], 3)} | "
                 f"{report_mod._fmt(x['max_dd_close_pct'], 2)} |")
    dl = r["decision_log"]
    L += ["", "## Decision log (judging cell)", "", f"{dl['n_triggers']} triggers, {dl['n_entered']} entered. First "
          "blocking reason per trigger (every reason is in decisions.csv):", "", "| status | triggers | meaning |",
          "|---|---|---|"]
    for s, n in dl["status_counts"].items():
        L.append(f"| {s} | {n} | {dl['block_reasons'].get(s, 'entered at the next bar open')} |")
    te = dl.get("time_exits") or {}
    L += ["", f"D17 time exits: {te.get('at_16_30_open', 0)} at the 16:30 New York open, "
          f"{te.get('early_close_us_holiday', 0)} at the last bar before the break on a declared US holiday or "
          f"early-close day, {te.get('early_close_other_day', 0)} at the last bar before a data gap on another day "
          "(the bars after it show the gap; a trader would have held on) [SI-64]. positions.csv: time_exit_rule."]
    L += ["", "## M1 resolution (D15 second run)", ""]
    if isinstance(r["m1"], Mapping):
        a, b = r["m1"]["combined_all_m15"] or {}, r["m1"]["combined_all_m1"] or {}
        mc = r["m1"].get("meta") or {}
        L += [f"Ambiguous M15 bars: {mc.get('bars_resolved', 0)} resolved on M1 bars; {mc.get('bars_unresolved', 0)} "
              f"kept the M15 answer (stop first): {mc.get('bars_unresolved_no_m1', 0)} without M1 bars, "
              f"{mc.get('bars_unresolved_m1_mismatch', 0)} whose M1 bars do not reach the M15 low and high on the "
              "closing side [SI-68].", ""]
        L += [f"Positions compared: {r['m1']['status_counts']}. Judging cell, combined: M15 E[R] "
              f"{_r(a.get('expectancy_r'))} R ({a.get('n_positions')} positions), M1-resolved E[R] "
              f"{_r(b.get('expectancy_r'))} R ({b.get('n_positions')} positions). {r['m1']['note']}.", ""]
    else:
        L += [str(r["m1"]) + " (no --m1-bid / --m1-ask given): ambiguous M15 bars keep the stop-first assumption "
              "(D15).", ""]
    nu = r.get("news_unscheduled") or {}
    L += ["## Data and inputs", "",
          f"- bid file {d.get('bid_file', 'in memory')} (sha256 {d.get('bid_sha256', 'n/a')})",
          f"- ask file {d.get('ask_file', 'in memory')} (sha256 {d.get('ask_sha256', 'n/a')})",
          f"- spread at the open: median {report_mod._fmt(d.get('spread_open_median'), 3)} USD/oz, p90 "
          f"{report_mod._fmt(d.get('spread_open_p90'), 3)} USD/oz; gaps {d.get('n_gaps')}; bars with the ask below "
          "the bid (counted, not refused): "
          + ", ".join(f"{k} {v}" for k, v in (d.get("ask_below_bid") or {}).items()),
          f"- news: {_news_line(r['news'])}",
          f"- restricted events (addendum A1, master_fp): {_restricted_line(r.get('restricted'))}",
          f"- {_unscheduled_line(nu.get('judging'), nu.get('master_twin'), nu.get('master_fp_twin'))}",
          f"- D1 range: from {d.get('range_start_utc', zv.RANGE_START_TEXT)} to the lock [SI-62]"
          + (f"; {d['range_note']}" if d.get("range_note") else "; the data starts inside it"),
          f"- holdout lock: no bar at or after {zv.LOCK_TEXT} was accepted (D1)", "",
          "## Rules", ""]
    L += [f"    {line}" for line in rs["describe"]]
    if info.get("status"):
        L += ["", f"Status: {info.get('status')}; sources: {'; '.join(info.get('sources') or []) or 'n/a'}."]
    tags = info.get("tags") or {}
    if any(tags.values()):
        L += ["", "| field | value | tag |", "|---|---|---|"]
        for f, t in tags.items():
            v = rules.get(f)
            L.append(f"| {f} | {'null (off)' if v is None else v} | {t} |")
    if rs.get("unmodelled"):
        L += ["", "Not modelled (the rules file's _meta.unmodelled):", ""]
        L += _unmodelled_lines(rs["unmodelled"], rs.get("unmodelled_master_note"))
    if rs.get("not_modelled_addendum_a3"):
        L += [""] + _a3_lines(rs)
    L += ["", "## Spec readings used here", ""]
    L += [f"- [{x['id']}] {x['text']}" for x in r["spec_readings"]]
    L += ["", "## Notes", "", f"- After a pass (spec): {AFTER_A_PASS}", f"- Change policy (spec): {CHANGE_POLICY}",
          "- Files: report.json (everything here), gates.json, grid.csv (every cell x side x period), trades.csv "
          "(judging cell, one row per leg), positions.csv and decisions.csv (judging cell), positions_all_cells.csv"
          + (", m1_diff.csv" if isinstance(r["m1"], Mapping) else "") + ".", ""]
    return report_mod._ascii("\n".join(L))
