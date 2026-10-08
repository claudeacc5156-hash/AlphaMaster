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
- One spread per bar is used for every fill and ask-side mark in that bar (an approximation).
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
