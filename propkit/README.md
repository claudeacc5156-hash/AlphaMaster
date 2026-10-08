# propkit - what does this strategy do to a prop-firm account?

RESEARCH ONLY - not trading advice. propkit reads a bar file plus a trade list or a position series and
answers: under these prop rules and these costs, does the account pass, breach the daily loss or breach the
max loss - on the history and in 10,000 simulated challenges? It never places, sends or prepares orders and
never talks to a broker or the internet. Every option and method is explained in `propkit\METHODS.md`.

All commands run from the AlphaMaster folder in Windows PowerShell, with the project's Python active.
`research\data\xauusd\train\XAUUSD_H1.parquet` is the bar file and `logs\` the output folder below.

## 1. Check the install (once, a few seconds)

```powershell
python -m propkit selftest
```

Every line must start with `PASS` and the last line must say `29 of 29 gates passed`. If a line says
`FAIL`, stop and send that line: the numbers would not be trustworthy on this PC.

## 2. The commands

```powershell
python -m propkit rules                     # the FTMO presets in plain English

# a trade list (CSV: side, units, entry_time, entry_price, exit_time, exit_price[, stop_price]) under 1-Step
python -m propkit evaluate --bars research\data\xauusd\train\XAUUSD_H1.parquet --trades logs\my_trades.csv --rules ftmo-1step --out logs\run_trades

# one AlphaMaster formula: export its held positions, then evaluate 10 oz per 1.0 of position
python scripts\research\export_positions.py --data-file research\data\xauusd\train\XAUUSD_H1.parquet --formula "[3,120]" --out logs\positions_f1.csv
python -m propkit evaluate --bars research\data\xauusd\train\XAUUSD_H1.parquet --positions logs\positions_f1.csv --size-mode units --size 10 --rules ftmo-1step --out logs\run_f1

# the pullback rule from a spec (copy propkit\examples\pullback_spec_example.json and fill it in first)
python -m propkit pullback --bars research\data\xauusd\train\XAUUSD_H1.parquet --spec logs\my_pullback.json --rules ftmo-2step --out logs\run_pullback
```

`python -m propkit evaluate --help` lists every option (capital, target, costs, cost multiplier, number of
simulations, seed, horizon, alpha, number of variants tried); METHODS.md section 2 explains each one. A run
on about 64,000 H1 bars takes under a minute. Exit code 0 = done, 2 = a problem with the command or a file
(the message says what to fix), 1 = a selftest gate failed.

## 3. What you get in --out

| file | what it holds |
|---|---|
| report.md | the readable report: data, rules, costs, the historical path, the simulated challenges, the largest safe size, statistics, stress tests and a STRATEGY CARD skeleton ([U] = fill in, [ASSUMPTION] = verify) |
| report.json | the same numbers for scripts |
| equity.csv, trades.csv, days.csv | the account bar by bar, the trades with costs and net PnL, one row per prop day |
| decisions.csv | pullback only: every signal and whether it entered or why not |

Open report.md in VS Code with Ctrl+Shift+V. The `+-` after a probability is simulation noise only; the
"history uncertainty" range next to it shows how much the answer depends on this one history, and is the
number to read. A probability of exactly 0 is shown as `0.0000 (< 0.0003)`: below 3 / simulations.

## 4. What is assumed (verify before trusting a number)

- FTMO rules as of 24 Sep 2026 (recheck before every challenge). 1-Step: +10% target; daily floor = the
  00:00 CE(S)T balance - 3% of the initial capital; max floor = highest 00:00 balance - 10%, trailing; best
  day <= 50% of all positive days' profit. 2-Step: +10% target (`--target 0.05` for phase 2); daily 5%;
  static max floor 90%; at least 4 trading days. The prop day starts at 00:00 CE(S)T (22:00 / 23:00 UTC).
- [ASSUMPTION] broker costs: 1 lot = 100 oz, swap 6% long / 2% short per year with a triple swap on
  Wednesday at 17:00 New York, no commission, the bar file's spread (else 0.34 USD/oz). Check the broker's
  contract specification and pass a costs file (METHODS.md section 4).
- Every pullback spec default is a PLACEHOLDER, not zeno's rule (METHODS.md section 6).
- Paths containing `locked_holdout` or ending in `.locked` are refused (exit code 2).
