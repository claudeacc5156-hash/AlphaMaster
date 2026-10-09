# propkit - methods, options and conventions

RESEARCH ONLY - not trading advice. The one-page quick start is `propkit\README.md`; this file holds the
detail behind it: every option, the input and output files, how to read the numbers, the conventions
and assumptions, and the pullback spec fields.

## 1. More examples

```powershell
# FTMO 2-Step phase 2 (5% target, reported as "FTMO 2-Step (target 5%)"), 50,000 USD, costs doubled (the
# trade list's fills are re-priced with twice the spread, markup and slippage), 25 variants tried in all
python -m propkit evaluate --bars research\data\xauusd\train\XAUUSD_H1.parquet --trades logs\my_trades.csv --rules ftmo-2step --target 0.05 --capital 50000 --cost-mult 2 --n-trials 25 --out logs\run_phase2

# the same formula with exactly AlphaMaster's miner cost (0.03% per fill, no swap)
python -m propkit evaluate --bars research\data\xauusd\train\XAUUSD_H1.parquet --positions logs\positions_f1.csv --size 10 --rules ftmo-1step --costs flat0.0003-noswap --out logs\run_f1_minercost
```

A full run on about 64,000 H1 bars with 10,000 simulations takes under a minute: a few seconds for a
strategy that trades most days, about 30 s for one that is always in the market and enters rarely (its
challenges need many market days to reach 60 trading days); `--history-reps 0` skips the slowest part. Open
`logs\run_...\report.md` in any Markdown viewer (VS Code: Ctrl+Shift+V).

## 2. Options of `evaluate` and `pullback`

| option | meaning (default) |
|---|---|
| `--bars FILE` | AlphaMaster Parquet or CSV: time (UTC), open, high, low, close (BID, USD/oz), optional spread (USD/oz) |
| `--trades CSV` or `--positions CSV` | the input (evaluate); `--spec JSON` for pullback |
| `--size-mode units` / `leverage`, `--size X` | positions only: 1.0 = X oz, or 1.0 = X times equity |
| `--rules` | `ftmo-1step`, `ftmo-2step`, `custom` with `--rules-file FILE.json`, or a .json file |
| `--capital`, `--target` | initial capital in USD (100000); profit target as a fraction (0.05 = 5%) |
| `--costs` | `dukascopy` (the bar's own spread, default), `flat0.0003` (0.03% of notional per fill; the [ASSUMPTION] placeholder swap is STILL charged), either with the suffix `-noswap` (swap off: `flat0.0003-noswap` is AlphaMaster's miner cost exactly), or a .json file of CostModel fields (`{"flat_rate_per_side": 0.0003, "swap_enabled": false}` is the same as `flat0.0003-noswap`) |
| `--cost-mult` | multiply every cost (1). With `--trades` the fills are re-priced: buys move up by (k - 1) x (spread + markup + slippage), sells down by (k - 1) x (markup + slippage); commission and swap come from the multiplied model. Below 1 is refused with `--trades` (the fills already hold the full costs) |
| `--spread-scale X` | multiply the bar file's spread column by X to get USD/oz, e.g. `0.01` for MT5 points of a 2-digit quote. Without it a median spread above 2 USD/oz is refused as probably being in points (XAUUSD is about 0.1 .. 0.7) |
| `--price-tolerance F` | `--trades` only: how far a fill may lie outside its bar's bid low .. ask high, as a fraction of the price (0.01 = 1%, default); `none` switches the check off |
| `--n-sims`, `--seed` | simulations (10000), seed (7, same seed = same answer) |
| `--horizon-days`, `--horizon-unit` | challenge length (60; `none` = until pass or breach, which takes minutes instead of seconds) and what it counts: `trading` = days with a trade entry, as FTMO counts trading days (default), or `market` = every prop day with bars. Days without a trade still run and can breach |
| `--history-reps` | outer bootstrap replicates for the history-uncertainty range (30; 0 = skip, 2 or more otherwise) |
| `--alpha` | breach budget for the largest size (0.05) |
| `--n-trials`, `--sr-var` | variants tried in all (this one included), and the variance of their daily Sharpe ratios, for the DSR |
| `--no-stress`, `--dd-sims` | skip the stress tests; orders in the drawdown shuffle (2000) |
| `--lot-step` | pullback only: sizes are rounded down to this many oz (1 = 0.01 lot) |
| `--out DIR` | output folder; created if missing; an input file inside it is never overwritten |

Exit code 0 = done, 2 = a problem with the command or the files (the message says what to fix), 1 = a
selftest gate failed.

## 3. Output files and how to read the key numbers

| file | what it holds |
|---|---|
| report.md | the readable report: data (file, sha256, bars, dates, spread), rules, costs, the historical path, the bootstrap (flat-to-flat day blocks and week blocks) with the history-uncertainty range, the largest safe size, statistics, stress tests, a STRATEGY CARD skeleton ([U] = fill in, [ASSUMPTION] = verify) |
| report.json | the same numbers for scripts (fractions are fractions; fields ending `_pct` are percent of capital) |
| equity.csv | one row per bar: balance, equity at the close, worst equity inside the bar, units open, costs; time_utc and prop_day text |
| trades.csv | the trades with commission (positive = paid), swap (negative = paid) and net PnL in USD |
| days.csv | one row per prop day: start balance, daily and max floors, profit, whether it traded or breached |
| decisions.csv | pullback only: every raw signal and whether it entered or why it was skipped |

How to read the key numbers: P(pass) is the share of simulated challenges that reached the target first.
The `+-` after it (`0.31 +- 0.005`) is the Monte Carlo error only: the noise of using 10,000 simulations
instead of infinitely many. It says nothing about how much the answer depends on THIS history. That is the
"history uncertainty" range next to it (5-95% over `--history-reps` resampled histories): a P(pass) of
0.31 with a range of 0.12 .. 0.55 is a rough guess, not a measurement. When a probability is exactly 0 (or
1) its Monte Carlo error is 0; the report then shows the 95% "rule of three" bound instead (`0.0000 (<
0.0003)` = below 3 / simulations). The largest size is the multiplier of your size at which P(daily-loss
breach) stays at or below `--alpha` (5%) on this history; because it picks the largest size that looks safe,
the true breach probability there can be higher (read its history range). Sharpe ratios come with their
standard error (sampling error of the history, IID assumed); PSR is the probability that the true Sharpe is
above 0; DSR (with `--n-trials`) corrects for the number of variants you tried. The report warns when the
history gives the bootstrap few independent starting points (see section 5).

## 4. Input files

- Trades CSV: `side` (1 / -1 or long / short), `units` (oz), `entry_time`, `entry_price`, `exit_time`,
  `exit_price`; optional `trade_id`, `exit_reason`, `stop_price` (gives R multiples). Times are UTC epoch
  seconds or ISO text with an offset (`2024-03-05T14:00:00Z`); MT5 server times must be converted first.
  Prices are the ACTUAL fills (a long buys at the ask), so spread and markup are already inside them.
  `stop_price` is a price LEVEL: a bid level for a long, an ask level for a short. 1R = the loss of a stop
  exit at that size: units x |entry - stop fill| + commission, where the stop fill is the level minus (long)
  or plus (short) the exit markup and slippage, so a trade stopped at its level is -1R. Other columns (a
  comment, a tag) are kept; any character in them is written as a `\uXXXX` escape.
- Positions CSV: `time` (bar open, UTC) and `position` (-1 .. 1, held DURING that bar). export_positions.py
  writes it already shifted (AlphaMaster's p[t] is decided at bar t's close and held during bar t+1) and
  keeps the unshifted value as `p_raw`.
- Rules JSON: PropRules fields, e.g. `{"base": "ftmo-2step", "profit_target_pct": 0.05}`. A preset with any
  other field changed is renamed "FTMO 1-Step (modified: daily_loss_pct=0.04)" and its notes say it is NOT
  the firm's published terms; a 2-Step with another target is "FTMO 2-Step (target 5%)". With
  `--rules ftmo-1step --rules-file FILE.json` the file's `base` (if any) must be the same preset. Costs JSON:
  CostModel fields, e.g. `{"markup_per_side": 0.10, "commission_per_lot_round_trip": 7.0, "swap_long": 4.5}`.
  Keys starting with `_` are comments; unknown keys are refused, so a typo cannot pass silently.

## 5. Conventions and assumptions

- The prop day is the CE(S)T calendar date: it starts at 22:00 UTC in summer and 23:00 UTC in winter. DST is
  computed from the EU and US rules; no time-zone database is needed.
- FTMO rules as of 24 Sep 2026 (`python -m propkit rules`); recheck them before every challenge.
- One spread per bar is used for every fill and ask-side mark in that bar (an approximation). A caller with
  real ask bars may pass them (`equity_from_trades(..., ask_prices={"high": ..., "close": ...})`): shorts are
  then marked at the ask close and their worst at the ask high; without it nothing changes.
- The bar size is the most common step between bar times (M15, M30 or H1). A file so short that a weekend
  step is as common as the bar step (two bars, Friday and Sunday) is refused: pass a few hours of bars.
- "No position open" (the target needs it; the bootstrap cuts its blocks there) means units open = 0 AND
  equity = balance at the bar close: a long and a short of the same size held together are NOT flat.
- [ASSUMPTION] defaults to verify on the broker's contract specification: 1 lot = 100 oz, swap 6% long / 2%
  short per year with a triple swap on Wednesday at 17:00 New York, no commission.
- The bootstrap assumes future days look like these days; it says nothing about regime changes. It resamples
  flat-to-flat blocks: a day block runs from a prop day that starts with no position open to the next such
  day, so a position held over midnight keeps its days together and every simulated challenge starts flat
  (as a real one does). A strategy that is almost always in the market gives few blocks; the report then
  warns, and the history-uncertainty range is the number to read.
- Any path containing `locked_holdout` or ending in `.locked` is refused (exit code 2): the holdout test is a
  separate, pre-registered step.

## 6. Pullback spec fields (propkit.pullback.PullbackSpec)

**Every default below is a PLACEHOLDER.** None of them is zeno's rule; they exist only so the code runs.
Copy `propkit\examples\pullback_spec_example.json`, replace the values with the real rule, set
`"placeholder": false`, and keep that file as the pre-registered spec. Reports flag any spec that still has
`"placeholder": true`. Missing keys take the placeholder default; `null` means "off" / "no limit" where the
table says so. Periods and lookbacks are in BARS of the data file's bar size (pin it with `bar_minutes`);
prices and distances are USD per ounce; fractions are fractions (0.005 = 0.5%).

| Field | Default (PLACEHOLDER) | Meaning |
|---|---|---|
| `name` | "PLACEHOLDER - ..." | A label printed in reports. |
| `placeholder` | true | true until the values are zeno's real rule. Set it to false when they are. |
| `bar_minutes` | null | The bar size the spec is written for (15, 30 or 60). null = any. If the data has another bar size, propkit stops with a message. |
| `direction` | "both" | "long", "short" or "both". |
| `trend_ema_period` | 200 | Trend EMA period, in bars, on bid closes. |
| `trend_slope_bars` | 10 | Slope test: for a long, the trend EMA now must be above its value this many bars ago (below for a short). 0 = no slope test. |
| `trend_require_close_side` | true | true: a long needs the close above the trend EMA, a short below it. With this false and `trend_slope_bars` 0 there is no trend filter. |
| `pullback_mode` | "ema_touch" | "ema_touch": in the last `pullback_lookback_bars` bars, some bar's low touched the pullback EMA (high for a short). "atr_from_swing": the pullback from the swing high (long) of the last `swing_lookback_bars` bars down to the lowest low after it must lie between `min_depth_atr` and `max_depth_atr` ATR. If the signal bar itself is the swing high, there is no pullback. |
| `ema_pullback_period` | 20 | Pullback EMA period (bars). Used by "ema_touch" and by the "close_back_over_ema" trigger. |
| `pullback_lookback_bars` | 5 | "ema_touch" window. The lowest low (long) or highest high (short) of these bars is the pullback extreme used by the swing stop. |
| `swing_lookback_bars` | 20 | "atr_from_swing" window for the swing high (long) or swing low (short). Ties go to the most recent bar. |
| `min_depth_atr` | 1.0 | "atr_from_swing": smallest pullback depth, in ATR. |
| `max_depth_atr` | 3.0 | "atr_from_swing": largest pullback depth, in ATR. null = no upper bound. |
| `trigger` | "close_back_over_ema" | Decided at the signal bar's CLOSE, filled at the NEXT bar's open. "close_back_over_ema": the close is above the pullback EMA and this bar's low, or the previous close, was at or below it (mirrored for shorts). "break_prev_extreme": the close is above the previous bar's high (short: below the previous bar's low). |
| `stop_mode` | "swing" | "swing": long stop = pullback low - `stop_buffer_atr` x ATR; short stop = pullback high + `stop_buffer_atr` x ATR + spread. "atr": long stop = bid at the entry open - `stop_atr_mult` x ATR; short stop = ask at the entry open + `stop_atr_mult` x ATR. The spread is always inside the stop distance. |
| `stop_buffer_atr` | 0.25 | Extra room beyond the pullback extreme, in ATR ("swing" stop). |
| `stop_atr_mult` | 1.5 | Stop distance in ATR ("atr" stop). |
| `atr_period` | 14 | Wilder ATR period (bars). ATR is taken at the signal bar. |
| `exit_mode` | "fixed_r" | One exit; the stop is always live as well. "fixed_r": target at `target_r` x R from the entry. "trail_ema": exit at the next open after a bar closes beyond the trail EMA (below it for a long). "time": exit at the open after `time_exit_bars` bars held. |
| `target_r` | 2.0 | "fixed_r": target distance in R. R = distance from the entry fill to the stop level. |
| `trail_ema_period` | 20 | "trail_ema": trail EMA period (bars). |
| `time_exit_bars` | 24 | "time": bars held before the exit (the entry bar counts as bar 1). |
| `risk_pct` | 0.005 | Fraction of the current closed balance risked per trade (0.005 = 0.5%). Values above 0.10 are refused as a likely percent/fraction mix-up. |
| `max_stop_usd` | null | Skip the trade if the entry-to-stop distance is more than this many USD per oz. null = no cap. |
| `max_atr_usd` | null | Skip the signal if ATR at the signal bar is more than this many USD per oz. null = no cap. |
| `sessions` | [] | Entries only inside one of these sessions, tested at the entry instant: "asia" (07:00-15:00 Singapore), "london" (08:00-16:30 London), "newyork" (08:00-17:00 New York), "overlap" (London and New York both open). DST is handled. |
| `session_hours_utc` | [] | Or inside one of these UTC hour ranges, start included, end excluded, e.g. [[7, 16]]. [22, 2] wraps midnight. If both lists are empty, any time is allowed. |
| `news_blackouts_utc` | [] | News windows in UTC, e.g. [["2026-11-06T13:15:00Z", "2026-11-06T14:00:00Z"]] (epoch seconds also accepted). No entry whose entry bar overlaps a window. |
| `news_flatten` | false | true: also close open positions at the open of the first bar that overlaps a window (exit reason "signal"). |
| `max_spread_usd` | null | Skip the entry if the spread of the bar named by `spread_filter_bar` (after any cost multiplier) is above this, USD per oz. null = no filter. |
| `spread_filter_bar` | "signal" | Which bar's spread `max_spread_usd` tests: "signal" = the signal bar, fully known when the decision is made (default); "entry" = the entry bar, causal only if your data's spread is the spread at the bar's OPEN (many files hold a bar average, minimum or maximum, which is known only after the bar). |
| `max_trades_per_day` | null | Most entries per prop day (the CE(S)T date of the entry, as FTMO counts days). null = no limit. |
| `max_open_positions` | 1 | Most positions open at the same time. A signal while full is skipped, not queued. |
| `daily_stop_losses` | null | No new entry for the rest of the prop day after this many losing trades closed that day. null = off. |
| `daily_stop_pct` | null | No new entry for the rest of the prop day once that day's closed PnL is at or below minus this fraction of the day's starting closed balance (0.02 = 2%). null = off. Open positions are not closed by it. |
| `entry_after_gap` | true | false: skip a signal when the next bar opens after a gap (weekend, daily break, missing data). |
| `warmup_bars` | null | No signal before this bar number. null = automatic: the largest of ATR period, 3 x each EMA period in use (+ slope bars), and the lookbacks. |

How fills and exits work: a signal is decided at a bar's close from that bar and earlier bars only and fills
at the next bar's open (long at ask + markup + slippage, short at bid - markup - slippage). A long's stop
and target are bid levels, a short's are ask levels, so the loss at the stop includes the spread, markup,
slippage and commission. A bar that opens beyond the stop (or target) fills at that open; when one bar's
range holds both the stop and the target, the STOP is assumed first. An exit inside a bar is stamped at the
bar's last second. Size = (risk_pct x closed balance) / (loss per ounce at the stop), rounded DOWN to
`--lot-step`; below one step the trade is skipped. `risk_usd` is the full 1R at that size, costs included,
so a stop filled at its level is -1R before swap. Positions open at the last bar close at its close
("end_of_data").

## 7. From Python (same pipeline as the command line)

```python
from propkit import CostModel, PullbackSpec, analyse, equity_from_trades, generate_trades, load_bars, preset
bars = load_bars(r"research\data\xauusd\train\XAUUSD_H1.parquet")
raw = generate_trades(bars, PullbackSpec.from_json(r"logs\my_pullback.json"), 100000.0, CostModel())
equity, trades = equity_from_trades(bars, raw, 100000.0, CostModel())
report = analyse(bars, trades, equity, preset("ftmo-1step", 100000.0), CostModel())
```

## 8. zeno_pullback_v1 (propkit.zeno_v1, propkit.zeno_report, `python -m propkit zeno-v1`)

RESEARCH ONLY - not trading advice. zeno's pullback rule exactly as frozen in
`propkit\specs\zeno_pullback_v1.md` (D1-D24, costs, reports, gates G0-G5), with addendum A
(`propkit\specs\zeno_pullback_v1_addendum_A.md`: how the FundingPips account is simulated; section 8.8).
The generic `pullback` command above is NOT this rule. Nothing here places or prepares orders.

### 8.1 Running it on the PC

The project has no 15m data yet: export Dukascopy XAUUSD M15 BID and ASK bars (UTC) for 2015-01-01 to
2025-09-27, as CSV (time, open, high, low, close) or as dukascopy-node CSV (timestamp in ms). Every bar must
open before 2025-09-28 00:00 UTC; a later bar is refused and there is no override (D1, the holdout lock). The
file names below are examples; the news calendar is the project's `research\news_calendar` file.

```powershell
# stage 1 (about 10 s): signals for the G0 chart check - no P&L, no R, no outcome
python -m propkit zeno-v1 signals --m15-bid research\data\xauusd\m15\XAUUSD_M15_bid.csv --m15-ask research\data\xauusd\m15\XAUUSD_M15_ask.csv --news research\news_calendar\us_macro_events_2015-01-01_2025-09-27.csv --out logs\zeno_g0

# G0: open logs\zeno_g0\g0_sample.csv, check every row on an M15 BID chart, write y or n in agree_y_n on every row

# stage 2 (a few minutes for 10.7 years): only if you agree with at least 18 of the 20
python -m propkit zeno-v1 run --g0-confirmed --g0-sample logs\zeno_g0\g0_sample.csv --m15-bid research\data\xauusd\m15\XAUUSD_M15_bid.csv --m15-ask research\data\xauusd\m15\XAUUSD_M15_ask.csv --news research\news_calendar\us_macro_events_2015-01-01_2025-09-27.csv --rules fundingpips-1step-flex --out logs\zeno_run
```

`signals` options: `--sample 20` and `--seed 7` (the G0 draw), and the declared cost cell `--variant
evaluation --commission 10 --spread-base S1 --cost-mult 1.5 --capital 100000` (whether a trigger is
eligible depends on the spread filter, which needs costs [SI-27]; the entry and stop shown are the chart's,
the data at costs x1 [SI-66]). `--variant` is evaluation, master or master_fp; master_fp also reads
`--restricted` (default the packaged FundingPips list, below). `signals` refuses to replace a g0_sample.csv in
--out that already holds answers. `run` options:
`--g0-confirmed` (required), `--g0-sample FILE` (its sha256 and your y/n answers go into gates.json; a sample
with more than 2 n of 20 is refused, and once any row is answered every row must be y or n, there must be
20 rows and at least 18 y [SI-60]; every row must be an eligible signal of the data being judged, in the cell
declared in the signals_report.json beside the sample, with the same entry and stop [SI-69]), `--rules` (default `fundingpips-1step-flex`, the
verified preset of addendum A3; a preset or a rules .json file), `--restricted FILE` (FundingPips' restricted
USD events for the master_fp cells; default
`propkit\data\news_calendar\us_restricted_events_fundingpips_2015-01-01_2025-09-27.csv`; its sha256 goes into
report.json and gates.json; git tracks `propkit\data` despite the root .gitignore's `data/`, and
`propkit\data\.gitattributes` keeps its CRLF bytes), `--master-primary master_fp|master` (default master_fp;
which Master run is the primary Master result, addendum A1), `--reference-rules ftmo-1step` (the judging cell under another firm's
rules, for comparison only), `--m1-bid/--m1-ask` (the D15 second run), `--n-sims 10000`, `--seed 7`, `--history-reps 30`,
`--capital` (default: the rules' 100,000 USD). Without `--g0-confirmed`, `run` stops before reading anything
and says to do the G0 check first. Locked paths and outputs outside `--out` are refused (exit code 2).

### 8.2 Output files

| stage | file | what it holds |
|---|---|---|
| signals | signals.csv | one row per trigger: status (eligible or the first blocking reason; the daily limits, the cooldown and the one-position rule need earlier trades and are not applied [SI-54]) and every reason, times in UTC, SGT and server time (New York + 7 h), H, L, leg, ATR, the pullback bar, the trigger bar, the entry and stop a chart of the data shows (entry_price, stop_level, spread_entry: costs x1 [SI-66]) and the declared cell's (entry_price_at_costs, stop_level_at_costs, spread_entry_at_costs, which the spread filter used). No exit, P&L, R or outcome |
| signals | decisions.csv | every setup event (armed, first_close_before_arming [SI-61], cancelled_new_extreme, voided, expired) and every trigger (news_pre_unscheduled marks a news block that comes only from the 30 min before an unscheduled row [SI-63]) |
| signals | g0_sample.csv | 20 random eligible signals (seed 7) in time order with the chart's entry and stop [SI-66], and an empty agree_y_n column |
| signals | signals_report.md / .json | counts only (triggers per status, side and year; setup events; unscheduled news rows), the declared cell, the data (D1 range, bars cut before 2015 [SI-62]) and the G0 instructions |
| run | report.md / report.json | the addendum's name and sha256 and every [U] tag and unmodelled rule of the rules file (header), the verdict and gates, the firm-day boundaries (A4-A5), the judging cell, results per side, year and G3 period, daily Sharpe / PSR / DSR (N = 1), the prop evaluator (judging cell, other firm day, master and master_fp twins), the Master runs (A1), the margin counts (A2), the 36 cells, the decision log, M1, data (with the restricted calendar), rules with [U] fields, spec readings |
| run | gates.json | G0-G5 and the kill rule: value, threshold, status (pass, fail, flagged, not_evaluated), the cell judged; G0 holds the sample's answers and sample_check (every row matched to this data [SI-69]); G4 holds P(daily-loss breach) per firm day and the one that set it; addendum_a, restricted_calendar (file, sha256), master_primary, day_boundaries |
| run | grid.csv | every cell x side (long, short, combined) x period (all, each year, the G3 periods): triggers, spread-filter blocks and positions vs the same cell at x1 [SI-65], positions, legs, trades per year, +2R hit rate, +4R after +2R, win rate, E[R] +- SE, E[USD], net USD, losing streak, outcome counts, daily SR +- SE, PSR, DSR, max drawdown |
| run | trades.csv | the judging cell's TRADES, one row per leg (+2R half and runner separately), plus position_id and leg |
| run | positions.csv / decisions.csv | the judging cell's positions (one row per entry, outcome class, time_exit_rule [SI-64], lots_uncapped and margin_capped [A2]) and decision log |
| run | positions_all_cells.csv | the positions of all 36 cells |
| run | m1_diff.csv | with --m1-bid/--m1-ask: every judging-cell position, M15 vs M1-resolved (an ambiguous bar whose M1 bars do not reach its M15 low and high keeps the M15 answer [SI-68]) |

### 8.3 Where G0 fits

The spec's change policy: before any result (P&L, R, Sharpe, hit rate, pass or breach odds) is shown, zeno may
still change a default; after it, any change is v2 and N rises. The G0 chart check shows signals without
P&L, so stage 1 writes no result and stage 2 refuses to run until `--g0-confirmed` says the check was done
and agreed (at least 18 of 20). Stage 1 leaves out the checks that need how and when earlier trades ended
(the daily limits, the cooldown, one open position), so its statuses carry no outcome; a trigger that passes
every other check is "eligible", and the run's entered signals are some of the eligible ones [SI-54].

### 8.4 The grid, the judging cell and the gates

36 cells: variant {evaluation 0.5% risk, master 0.4% risk with the news close, master_fp = master plus
FundingPips' restricted list (addendum A1)} x commission {5, 10} USD per
lot round trip x spread base {S1 = the data, S2 = bid + 0.18, 0.20 USD/oz from 05:00 to 08:00 SGT} x cost
multiplier {1, 1.5, 2} (scales spread, commission and the 0.05 USD/oz stop slippage; the scaled spread also
meets rule 10's 10%-of-R filter, so x1.5 and x2 trade fewer triggers: grid.csv shows how many [SI-65]). The
judging cell is
(evaluation, 10, the worse base, x1.5); the worse base is the one with the lower combined net E[R], a tie
goes to S2 [SI-28]. G1: >= 100 positions (not legs) [SI-29]. G2: E[R] > 0 and PSR(SR > 0) >= 0.95 on
server-day returns [SI-30]. G3: E[R] > 0 in >= 3 of 2015-2017, 2018-2020, 2021-2023, 2024-2025-09-27 (a
position belongs to the server day of its trigger close and entry [SI-42]; an empty period is not above 0
[SI-53]). G4:
P(daily-loss breach before target) <= 5% and P(max-loss breach before target) <= 10% under the VERIFIED
FundingPips rules, bootstrap without a horizon [SI-31]; with unverified rules it is `not_evaluated` [SI-32].
Addendum A5: P(daily-loss breach) is the higher of the rules' own firm day and the other one (A4).
G5: per side E[R] at costs x1 (the worse base re-chosen at x1 [SI-52]); a side at or below 0 is flagged. Kill:
G2 at that x1 cell; if it fails, "the rule as written has no edge on this data" and any change is v2. The DSR
uses N = 1 and equals PSR there; the report prints the spec's caveat next to it. The prop evaluator (path,
10,000 day-block challenges, largest size, history uncertainty) runs for the judging cell, the judging cell
under the other firm day (A4), and its master and master_fp twins. Neither Master run feeds G1-G5.

### 8.5 Firm rules, day_boundary and the FundingPips presets

`PropRules.day_boundary` sets the firm's day for the evaluator, the bootstrap and the statistics:
`cet_midnight` (00:00 CE(S)T, FTMO, the default, unchanged results), `ny_17` (17:00 New York, the broker
server day of D21), `utc_midnight` or `utc_plus3` (00:00 UTC+3 = 21:00 UTC all year; in US winter it differs
from ny_17 between 21:00 and 22:00 UTC, and the hour after the Friday close becomes a short Saturday day). A
rules .json holds PropRules fields plus an optional `_meta` block (firm, plan, status, verified, verified_on,
sources, tags per field "[VP] source, date" = verified from a primary source or "[U] why" = unverified, also
qualified as "[VP 1SF, CMP] ...", conservative_choices, warning, unmodelled = the rules the evaluator does not
model); `"base": "ftmo-1step"` keeps a preset's other fields. A field is unverified when its tag starts with
[U]; every tag holding [U] anywhere and every unmodelled item are printed at the top of report.md.
`fundingpips-1step-flex` reads `propkit\presets\fundingpips_1step_flex.json`, the verified sheet of
2026-10-08 (addendum A3: target 12%, daily loss 2% of the higher of the day-start balance and equity, max
loss 12% static, no minimum days, target counted only when flat [U], day_boundary ny_17). Only when that file
is missing does it fall back to `fundingpips-1step-flex-placeholder`, and every report says so. The
placeholder (still available by name) holds no
FundingPips number: 100,000 USD and the 2% daily-loss option are zeno's own account facts, every field is
[U]: initial_capital, profit_target_pct (null), daily_loss_pct (0.02), daily_loss_base (initial),
day_start_reference (max_balance_equity), max_loss_pct (null), max_loss_mode (trailing_eod_balance),
best_day_max_share (null), best_day_basis, min_trading_days (0), target_requires_flat (true),
breach_inclusive (true), day_boundary (ny_17). The structural choices are the conservative ones. With no
target, P(pass) and days to target are not defined; the bootstrap then runs 60 trading days as an
illustration [SI-55]. Under the placeholder G4 is not evaluated; under the verified preset it is.

### 8.6 Deviations from propkit conventions and the spec readings

- `risk_usd` in trades.csv is the spec's R x size (D13: commission and slippage on top), not propkit's
  "1R with costs", so a full stop is slightly worse than -1R [SI-26].
- The prop evaluator marks an open short at the cell's ask close and its worst inside a bar at the cell's ask
  high (`propkit.equity.equity_from_trades(..., ask_prices=)`; propkit's default, bid + the bar's open
  spread, understates a short's drawdown when the data's spread widens inside a bar) [SI-67].
- `propkit\specs\zeno_pullback_v1.md` folds its one non-ASCII character to `(*)` (every .md in propkit must be
  ASCII); `zeno_pullback_v1.md.utf8` is the byte-exact copy with the recorded sha256 [SI-1].
- The README stays one page; this section holds the detail.
- Readings of the spec where it is silent (the full list with the rejected options is SPEC_ISSUES.md): SI-5
  partial hours in H1; SI-6/SI-10 windows count bars of the data across gaps; SI-8 a trading day is a server
  day with bars; SI-12 an equal low moves the pullback bar to the later one and restarts the count; SI-13 the
  arming bar may trigger, unless the first close above the pullback bar came before arming (SI-61); SI-14/
  SI-25 an entry is stamped and counted at the trigger close, and no fill at or after 16:30 New York of that
  day is accepted; SI-34 levels unrounded, a price within 1e-9 USD/oz of a level is at it; SI-15 shorts
  signal on bid bars; SI-17 +2R and +4R in one bar; SI-19 an entry at or beyond the stop is blocked; SI-20 a
  short's breakeven is entry - commission per oz; SI-21 slippage on stop fills only; SI-24 a loss is a
  position with net P&L < 0; SI-27 to SI-32, SI-41, SI-42 and SI-52 to SI-57 as in the report's "Spec
  readings" list; SI-46 one setup per H bar and pullback bar; SI-47 expiry at the close of bar 8; SI-61 to
  SI-65 (first close before arming, the 2015 range start, unscheduled news rows, D17's last bar before a
  break, the multiplier in the spread filter) as in the report's list; SI-66 stage 1 shows the chart's entry
  and stop; SI-67 short marks on the cell's ask; SI-68 the M1 run replays a bar only when its M1 bars reach
  the M15 low and high; SI-69 the G0 sample must be answered on every row and belong to the data judged;
  SI-70 a Master fill at the open of the bar holding T - 10 min (after a data gap) is closed at that open.

### 8.7 D1-D24 -> function -> test

| D | rule | function(s) | test(s) (tests\unit\) |
|---|---|---|---|
| D1 | bid+ask M15 data, holdout lock, H1 from M15, 30-day warm-up | `zeno_v1.load_m15_bidask`, `bidask_frame`, `check_before_lock`, `h1_from_m15`, `trading_days` | test_zeno_v1_data.py::test_holdout_lock_last_allowed_bar_and_first_refused_bar; test_zeno_v1_data.py::test_locked_paths_are_refused_before_anything_is_read; test_zeno_v1_data.py::test_h1_from_m15_with_a_partial_hour; test_zeno_v1_filters.py::test_warmup_is_30_trading_days; test_zeno_v1_runner.py::test_lock_and_locked_path_refusals; test_zeno_v1_data.py::test_bars_before_the_range_start_are_cut_and_counted; test_zeno_v1_data.py::test_the_period_all_starts_in_2015_and_a_late_start_is_flagged |
| D2 | EMA30 with SMA seed, Wilder ATR14 | `indicators.ema(seed="sma")`, `zeno_v1.ema30_h1`, `atr14_m15` | test_zeno_v1_indicators.py::test_ema_sma_seed_hand_table_period_3; test_zeno_v1_indicators.py::test_ema_default_seed_is_unchanged; test_zeno_v1_indicators.py::test_atr14_wilder_with_a_gap |
| D3 | the last closed 1h bar at a 15m close; "5 bars ago" | `h1_index_at_m15_close`, `trend_state` | test_zeno_v1_indicators.py::test_h1_bar_is_usable_at_its_close_not_before; test_zeno_v1_indicators.py::test_trend_state_slope_and_close_side; test_zeno_v1_indicators.py::test_five_bars_ago_counts_h1_bars_of_the_data_across_a_gap |
| D4 | trend at the trigger close, sides separate | `prepare`, `_Engine._decide` | test_zeno_v1_indicators.py::test_trend_in_prepare_uses_the_last_closed_h1_bar; test_zeno_v1_filters.py::test_block_reasons_are_reported_in_order; test_zeno_v1_invariants.py::test_the_trend_is_read_from_the_h1_bar_closed_by_the_trigger_close |
| D5 | H and L windows, ties | `setup_machines` | test_zeno_v1_setup.py::test_h_tie_goes_to_the_latest_bar; test_zeno_v1_setup.py::test_l_is_the_lowest_low_of_the_20_bars_before_h; test_zeno_v1_setup.py::test_no_setup_without_20_bars_before_h |
| D6 | 50% wick retrace, 78.6% void on a close | `setup_machines` | test_zeno_v1_setup.py::test_wick_touch_at_exactly_50_percent_arms; test_zeno_v1_setup.py::test_no_touch_one_cent_above_50_percent; test_zeno_v1_setup.py::test_void_on_the_close_strictly_beyond_78_6_percent; test_zeno_v1_setup.py::test_void_before_arming_prevents_the_setup; test_zeno_v1_setup.py::test_an_exact_50_percent_touch_counts_despite_float_noise; test_zeno_v1_setup.py::test_a_close_exactly_at_the_void_level_does_not_void_despite_float_noise |
| D7 | leg >= 1.5 ATR at the arming bar | `setup_machines` | test_zeno_v1_setup.py::test_leg_uses_atr_at_the_arming_bar |
| D8 | arming, frozen H/L, a new extreme cancels | `setup_machines` | test_zeno_v1_setup.py::test_h_and_l_stay_frozen_after_h_leaves_the_window; test_zeno_v1_setup.py::test_a_new_high_cancels_an_equal_high_does_not; test_zeno_v1_setup.py::test_void_wins_over_a_new_extreme_in_the_same_bar |
| D9 | pullback bar, 8-bar count, expiry | `setup_machines` | test_zeno_v1_setup.py::test_trigger_one_bar_after_the_low; test_zeno_v1_setup.py::test_a_new_low_restarts_the_count_and_bar_8_may_trigger; test_zeno_v1_setup.py::test_expiry_at_the_close_of_bar_8_and_bar_9_cannot_trigger_or_revive; test_zeno_v1_setup.py::test_a_close_above_the_pullback_bar_before_arming_is_the_first_close_and_is_not_chased; test_zeno_v1_setup.py::test_after_a_passed_first_close_a_new_pullback_low_restarts_the_count |
| D10 | one shot on every blocker | `_Engine._decide`, `BLOCK_REASONS` | test_zeno_v1_filters.py::test_a_blocked_trigger_is_one_shot; test_zeno_v1_filters.py::test_block_reasons_are_reported_in_order |
| D11 | entry fill side, entry spread, gap entry | `_Engine._decide`, `ask_side` | test_zeno_v1_entry.py::test_long_fill_stop_r_size_and_full_stop_numbers; test_zeno_v1_entry.py::test_short_fills_at_the_bid_and_its_stop_is_an_ask_level_with_the_entry_spread; test_zeno_v1_entry.py::test_entry_after_an_opening_gap_uses_the_actual_open; test_zeno_v1_invariants.py::test_rule_10_reads_the_entry_bar_open_spread_not_a_later_one |
| D12 | ATR at the trigger close in rules 5 and 8 | `_Engine._decide` | test_zeno_v1_entry.py::test_long_fill_stop_r_size_and_full_stop_numbers; test_zeno_v1_filters.py::test_volatility_and_stop_width_edges |
| D13 | R, lots rounded down, commission on top | `_Engine._decide`, `pullback.floor_to_lot_step` | test_zeno_v1_entry.py::test_lot_and_partial_rounding; test_zeno_v1_entry.py::test_size_below_one_lot_step_is_blocked; test_zeno_v1_entry.py::test_risk_comes_from_the_closed_balance_at_entry |
| D14 | stop and target sides, gaps | `_Engine._open_gaps`, `_Engine._intrabar` | test_zeno_v1_exits.py::test_gap_through_the_stop_fills_at_the_open_with_slippage; test_zeno_v1_exits.py::test_gap_through_a_target_fills_at_the_level_never_better; test_zeno_v1_exits.py::test_short_time_exit_at_the_ask_open; test_zeno_v1_exits.py::test_an_exact_touch_of_the_tp1_level_fills_the_partial; test_zeno_v1_exits.py::test_an_exact_touch_of_a_short_target_on_the_ask |
| D15 | stop first; breakeven after the partial in the same bar; M1 second run | `_Engine._intrabar`, `_Engine._ambiguous`, `resolve_with_m1` | test_zeno_v1_exits.py::test_stop_first_when_one_bar_touches_the_stop_and_a_target; test_zeno_v1_exits.py::test_breakeven_is_checked_after_the_partial_in_the_same_bar; test_zeno_v1_exits.py::test_m1_resolution_changes_a_stop_first_bar; test_zeno_v1_runner.py::test_m1_second_run_is_reported_beside_the_first; test_zeno_v1_exits.py::test_m1_bars_that_miss_the_m15_high_or_low_leave_the_bar_unresolved; test_zeno_v1_exits.py::test_m1_check_on_the_short_side_uses_the_ask |
| D16 | partial rounding, breakeven = entry +- commission per oz | `_Engine._take_tp1`, `_Engine._decide` | test_zeno_v1_entry.py::test_lot_and_partial_rounding; test_zeno_v1_entry.py::test_breakeven_level; test_zeno_v1_entry.py::test_tp1_then_breakeven_pnl_by_hand; test_zeno_v1_exits.py::test_an_exact_touch_of_the_breakeven_level_closes_the_runner |
| D17 | 16:30 New York exit (DST, early close), no rollover | `time_exit_instant`, `_Engine._bar`, `_assert_no_rollover` | test_zeno_v1_indicators.py::test_time_exit_instant_known_answers; test_zeno_v1_exits.py::test_time_exit_at_the_open_of_the_16_30_new_york_bar; test_zeno_v1_exits.py::test_early_close_exits_at_the_close_of_the_last_bar_before_the_break; test_zeno_v1_exits.py::test_no_position_may_cross_the_rollover; test_zeno_v1_filters.py::test_a_gap_cannot_carry_an_entry_past_16_30_new_york_of_the_trigger_day; test_zeno_v1_exits.py::test_the_last_bar_before_a_break_is_labelled_us_holiday_or_data_gap_and_counted |
| D18 | ATR <= 2 x 20-day median, stop <= 3 ATR | `vol_medians`, `_Engine._decide` | test_zeno_v1_filters.py::test_volatility_and_stop_width_edges; test_zeno_v1_filters.py::test_vol_median_is_pooled_over_every_bar_of_the_20_days |
| D19 | entry windows 15:00-18:00 and 20:30-24:00 SGT | `session_ok` | test_zeno_v1_filters.py::test_entry_window_edges; test_zeno_v1_indicators.py::test_session_windows_equal_the_sgt_text |
| D20 | news blackout T-30 min .. T+60 min | `read_news_csv`, `NewsCalendar.blocked` | test_zeno_v1_filters.py::test_news_blackout_edges; test_zeno_v1_filters.py::test_news_window_to_the_second; test_zeno_v1_data.py::test_news_csv_keeps_the_four_events; test_zeno_v1_master.py::test_a_trigger_blocked_only_before_an_unscheduled_row_is_flagged_and_counted |
| D21 | server day; 2 entries, 2 losses, -1.0%, 1 position | `server_day`, `_Engine._day`, `_Engine._decide` | test_zeno_v1_filters.py::test_two_entries_two_losses_and_minus_one_percent; test_zeno_v1_filters.py::test_daily_limits_reset_at_17_00_new_york; test_zeno_v1_filters.py::test_a_trade_open_blocks_a_new_entry |
| D22 | cooldown 15 min from the exit stamp | `_Engine._decide` | test_zeno_v1_filters.py::test_cooldown_counts_from_the_intrabar_exit_stamp; test_zeno_v1_filters.py::test_cooldown_blocks_a_trigger_at_the_close_of_the_exit_bar; test_zeno_v1_filters.py::test_cooldown_is_per_direction |
| D23 | Master variant: 0.4% risk, close 10 min before news | `ZenoConfig.risk_fraction`, `_Engine` | test_zeno_v1_entry.py::test_master_variant_risks_0_4_percent; test_zeno_v1_master.py::test_master_closes_a_young_position_at_the_open_of_the_bar_holding_t_minus_10; test_zeno_v1_master.py::test_master_keeps_a_position_opened_5h_or_more_before; test_zeno_v1_master.py::test_a_master_close_for_an_unscheduled_row_is_counted; test_zeno_v1_master.py::test_master_closes_at_the_entry_open_when_the_fill_bar_holds_t_minus_10; test_zeno_v1_master.py::test_master_keeps_a_fill_after_t_minus_10_when_no_bar_holds_it |
| D24 | the firm's day in the prop evaluator | `PropRules.day_boundary`, `calendar.firm_day`, `evaluator.equity_arrays`, `bootstrap.build_day_units` | test_zeno_v1_rules_day_boundary.py::test_ny_17_and_utc_midnight_known_answers; test_zeno_v1_rules_day_boundary.py::test_evaluator_daily_floor_resets_at_17_00_new_york; test_zeno_v1_rules_day_boundary.py::test_ftmo_known_answers_unchanged; test_zeno_v1_costs.py::test_the_prop_evaluator_marks_a_short_on_the_cells_ask_high_and_close; test_zeno_v1_costs.py::test_the_ask_marks_change_nothing_where_the_ask_is_bid_plus_the_open_spread |
| Costs | commission 5/10, S1/S2, multipliers, stop slippage | `ask_side`, `s2_spread`, `cost_model_for_cell` | test_zeno_v1_costs.py::test_multiplier_scales_spread_commission_and_slippage; test_zeno_v1_costs.py::test_s2_spread_base_in_the_engine; test_zeno_v1_costs.py::test_slippage_applies_to_stop_fills_only; test_zeno_v1_runner.py::test_the_cost_multiplier_also_tightens_the_spread_filter_and_the_grid_shows_it |
| Reports, gates | metrics, judging cell, G0-G5, kill | `zeno_report.position_metrics`, `daily_metrics`, `worse_base`, `gates_from`, `g4_gate` | test_zeno_v1_runner.py::test_metric_definitions_on_a_hand_built_cell; test_zeno_v1_runner.py::test_gates_known_answers_on_a_hand_built_grid; test_zeno_v1_runner.py::test_judging_cell_picks_the_worse_base; test_zeno_v1_runner.py::test_period_assignment_by_server_day; test_zeno_v1_runner.py::test_g4_uses_an_unlimited_horizon; test_zeno_v1_runner.py::test_a_0_01_lot_position_that_reaches_2r_did_not_fill_the_partial |
| Runner | G0 order, outputs, refusals | `cli.cmd_zeno_signals`, `cli.cmd_zeno_run`, `zeno_report.signals_stage`, `run_stage` | test_zeno_v1_runner.py::test_signals_stage_writes_no_results; test_zeno_v1_runner.py::test_run_refuses_without_g0_confirmed; test_zeno_v1_runner.py::test_refuses_writing_outside_out; test_zeno_v1_runner.py::test_run_writes_every_file_and_gates_json; test_zeno_v1_runner.py::test_stage_1_files_never_name_a_check_that_needs_earlier_trades; test_zeno_v1_runner.py::test_a_g0_sample_that_can_no_longer_reach_18_of_20_is_refused; test_zeno_v1_runner.py::test_write_staged_rolls_back_when_a_rename_fails; test_zeno_v1_runner.py::test_g0_sample_shows_the_chart_prices_not_the_cost_cell_prices; test_zeno_v1_runner.py::test_an_answered_g0_sample_needs_every_row_answered_and_18_y; test_zeno_v1_runner.py::test_run_checks_that_the_g0_sample_belongs_to_the_data_it_judges; test_zeno_v1_runner.py::test_signals_refuses_to_overwrite_an_answered_g0_sample |
| Invariants | mirror, causality, timing | engine | test_zeno_v1_invariants.py::test_long_short_mirror_is_exact; test_zeno_v1_invariants.py::test_decisions_are_causal_under_truncation; test_zeno_v1_runner.py::test_timing_guard_signals_on_257k_bars; test_zeno_v1_runner.py::test_timing_guard_full_single_cell_on_257k_bars; test_zeno_v1_invariants.py::test_everything_known_by_an_instant_survives_replacing_all_later_prices |

### 8.8 Addendum A (FundingPips in the simulation) -> function -> test

`propkit\specs\zeno_pullback_v1_addendum_A.md` (sha256
0f64bf584e325cc665e1afee0f9dccb9e53f83645f078abad2d455f5982af7b0, `zeno_v1.ADDENDUM_A_SHA256`) changes only how
the prop firm is simulated; zeno's 12 rules, D1-D24, the costs and the gates stay as frozen. The evaluation
and master cells give the same trades as before the addendum, except where A2's margin cap binds in master.

- A1 master_fp: everything of master (0.4% risk, D20, D23 for NFP, CPI, PPI, FOMC) plus FundingPips'
  restricted list: no entry when the trigger close or the entry fill lies in [T - 5 min, T_end + 5 min]
  (T_end = T for a release, T + 180 min for a Fed Chair testimony, T + 60 min for another Fed Chair
  appearance) or on the New York date of an event without a time (blocking reason fp_restricted_window,
  after news_blackout); the D23 close 10 min before every restricted event with a time, for positions opened
  under 5 h before it (a fill in the bar holding T - 10 min is closed at its own open, as SI-70). The
  unscheduled rows (kind) are flagged as in SI-63 (decisions: fp_pre_unscheduled). SI-63's D20 counts keep
  their meaning in master_fp; the restricted window's own (fp_n_triggers_flagged, and
  n_triggers_blocked_only_before_unscheduled_any for news and/or the window) are a separate clause of the
  "unscheduled rows" line in report.md and signals_report.md.
- A2 margin: per position, margin = 100 x P x sum(lots in tier / leverage) over 0.05 lot at 1:50, 0.05 at
  1:30, 0.05 at 1:25, 0.10 at 1:20, 0.25 at 1:10, the rest at 1:5, at the entry fill price. In both Master
  variants D13's lots are cut to the largest 0.01-lot size whose margin fits the closed balance
  (positions: lots_uncapped, margin_capped; blocked when not even 0.01 lot fits:
  margin_cap_below_lot_step). Example: at 4,000 USD/oz on 100,000 USD, 1.57 lots fit (99,466.67 USD) and
  1.58 do not (100,266.67 USD). The evaluation run is not capped; it counts the entries over margin at a
  flat 1:10 and 1:30.
- A3 rules: the preset of 8.5. report.md and report.json also list A3's Master rules that no run simulates
  (minimum reward, the Monthly 100% consistency rule, the Striking System), and the preset's last unmodelled
  item (kept as given) carries a report note on what the code does with each Master rule it names.
- A4: the judging cell's prop evaluator runs again with the other firm day (utc_plus3, or ny_17 when the
  rules already use utc_plus3). A5: G4 takes the higher P(daily-loss breach) of the two (the rules' own on a
  tie) and the rules' own P(max-loss breach); gates.json and report.md show both and which firm day set it.

| A | what | function(s) | test(s) (tests\unit\) |
|---|---|---|---|
| A1 | master_fp: restricted calendar, entry block, Master close, unknown-time dates | `zeno_v1.read_restricted_csv`, `restricted_calendar`, `RestrictedCalendar.blocked`, `fp_duration_s`, `prepare(restricted=)`, `_Engine._decide`, `_Engine._add_master_closes`, `master_close_counts`, `restricted_counts`, `unscheduled_counts`, `zeno_report._unscheduled_line` | test_zeno_v1_addendum_a_master_fp.py::test_packaged_restricted_calendar; test_zeno_v1_addendum_a_master_fp.py::test_reader_rows_and_errors; test_zeno_v1_addendum_a_master_fp.py::test_window_edges_on_real_rows; test_zeno_v1_addendum_a_master_fp.py::test_engine_blocks_at_the_edges_with_real_rows; test_zeno_v1_addendum_a_master_fp.py::test_the_entry_fill_instant_is_checked_too; test_zeno_v1_addendum_a_master_fp.py::test_unknown_time_blocks_the_new_york_date; test_zeno_v1_addendum_a_master_fp.py::test_unknown_time_date_blocks_entries_all_day; test_zeno_v1_addendum_a_master_fp.py::test_close_for_4h59_and_not_for_5h00; test_zeno_v1_addendum_a_master_fp.py::test_close_age_on_a_real_row; test_zeno_v1_addendum_a_master_fp.py::test_close_on_the_fill_bar_after_a_gap; test_zeno_v1_addendum_a_master_fp.py::test_master_fp_against_master_on_a_claims_release; test_zeno_v1_addendum_a_master_fp.py::test_master_is_unchanged_by_the_restricted_calendar; test_zeno_v1_addendum_a_master_fp.py::test_unscheduled_flags_work_for_master_fp; test_zeno_v1_addendum_a_master_fp.py::test_stage1_unscheduled_line_for_master_fp; test_zeno_v1_addendum_a_master_fp.py::test_d20_applies_in_master_fp; test_zeno_v1_addendum_a_master_fp.py::test_variants_risk_reasons_and_grid |
| A2 | Master margin cap; evaluation counted at flat 1:10 and 1:30 | `zeno_v1.margin_usd`, `max_units_within_margin`, `margin_counts`, `_Engine._decide` | test_zeno_v1_addendum_a_margin.py::test_margin_known_answers; test_zeno_v1_addendum_a_margin.py::test_a_capped_master_position_and_its_pnl; test_zeno_v1_addendum_a_margin.py::test_the_spec_risk_is_not_capped_near_2000; test_zeno_v1_addendum_a_margin.py::test_evaluation_is_not_capped_but_counted_at_flat_leverage; test_zeno_v1_addendum_a_margin.py::test_an_entry_that_cannot_hold_001_lot_is_blocked_by_margin |
| A3 | the verified preset, [U] tags and unmodelled rules in the report; the Master rules no run simulates | `rules.load_firm_preset`, `rules_from_json`, `tag_kind`, `has_u_tag`, `zeno_report._rules_header`, `A3_NOT_MODELLED`, `a3_not_modelled`, `_unmodelled_lines`, `_a3_lines`, `_master_lines` | test_zeno_v1_addendum_a_days_rules.py::test_verified_preset_loads_without_fallback; test_zeno_v1_addendum_a_days_rules.py::test_verified_preset_u_tags_and_unmodelled; test_zeno_v1_addendum_a_days_rules.py::test_tag_kind; test_zeno_v1_addendum_a_days_rules.py::test_loader_rejects_bad_unmodelled_and_untagged_fields; test_zeno_v1_addendum_a_days_rules.py::test_placeholder_is_used_only_when_the_verified_file_is_missing; test_zeno_v1_addendum_a_days_rules.py::test_cli_rules_lists_the_verified_preset_and_run_defaults_to_it; test_zeno_v1_addendum_a_runner.py::test_run_defaults_to_the_verified_preset_and_36_cells; test_zeno_v1_addendum_a_runner.py::test_run_report_lists_a3_master_rules_as_not_modelled |
| A4 | the utc_plus3 firm day and the judging cell under it | `calendar.firm_day`, `calendar.firm_day_start_utc`, `zeno_report.run_stage`, `day_boundary_summary` | test_zeno_v1_addendum_a_days_rules.py::test_utc_plus3_known_answers_summer_and_winter; test_zeno_v1_addendum_a_days_rules.py::test_utc_plus3_weekend_pair_against_ny_17; test_zeno_v1_addendum_a_days_rules.py::test_utc_plus3_day_start_label_and_round_trip; test_zeno_v1_addendum_a_days_rules.py::test_existing_boundaries_unchanged_by_the_new_one; test_zeno_v1_addendum_a_days_rules.py::test_evaluator_daily_breach_under_utc_plus3_but_not_ny_17; test_zeno_v1_addendum_a_days_rules.py::test_bootstrap_and_daily_returns_under_utc_plus3; test_zeno_v1_addendum_a_runner.py::test_the_sensitivity_is_ny_17_when_the_rules_already_use_utc_plus3 |
| A5 | G4 from the higher P(daily-loss breach) of the two firm days | `zeno_report.g4_gate` | test_zeno_v1_addendum_a_runner.py::test_g4_gate_known_answers_with_two_firm_days; test_zeno_v1_addendum_a_runner.py::test_run_g4_uses_the_higher_daily_breach_of_both_firm_days |
| Runner | 36 cells, --restricted, --master-primary, twins, report sections, signals --variant master_fp, the addendum copy, the packaged calendars in git | `cli.cmd_zeno_run`, `cli.cmd_zeno_signals`, `zeno_report.run_stage`, `master_section`, `margin_section`, `addendum_identity`, `restricted_info` | test_zeno_v1_addendum_a_runner.py::test_addendum_copy_is_byte_exact_ascii_and_hashed; test_zeno_v1_addendum_a_runner.py::test_the_packaged_calendars_reach_a_commit_byte_exact; test_zeno_v1_addendum_a_runner.py::test_run_records_the_restricted_calendar_and_its_sha256; test_zeno_v1_addendum_a_runner.py::test_run_has_the_prop_stack_for_the_judging_cell_and_both_master_twins; test_zeno_v1_addendum_a_runner.py::test_run_report_shows_the_master_and_margin_sections; test_zeno_v1_addendum_a_runner.py::test_run_with_master_primary_master_and_another_restricted_file; test_zeno_v1_addendum_a_runner.py::test_run_refuses_a_bad_master_primary; test_zeno_v1_addendum_a_runner.py::test_run_stage_needs_the_restricted_calendar_for_master_fp; test_zeno_v1_addendum_a_runner.py::test_signals_variant_master_fp_reads_the_packaged_restricted_calendar; test_zeno_v1_addendum_a_runner.py::test_signals_evaluation_ignores_restricted_and_keeps_its_output |

Readings where the addendum is silent or does not fit the code (the literal reading is kept):
- A2's "1.58 lots at 4,000" and "1.68 at 3,700" are the continuous break-even sizes (1.577 and 1.678 lots);
  by A2's formula the largest 0.01-lot sizes that fit are 1.57 and 1.67.
- A1 keeps master "exactly as written", A2 caps both Master runs: master is unchanged except where the cap
  binds (report.md: margin-capped entries).
- A5 names one P(max-loss breach): G4 uses the rules' own firm day's; both are reported.
- A1's 5-min entry block does not reach the bar holding T - 10 min, so an entry filled at that bar's open
  is closed at the same open (a zero-length trade that pays spread and commission), as SI-70 does after a
  gap.
- utc_plus3 puts the hour after the US-winter Friday close in a Saturday firm day; propkit.bootstrap's
  week blocks (Sunday to Saturday) keep it in Friday's week.
