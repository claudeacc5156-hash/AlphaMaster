"""propkit/stress.py - stress tests of one trade list (CLAUDE.md B "STRESS"). Research only.

Every stress keeps the TRADES of the run and changes one thing, then rebuilds the account path with
propkit.equity.equity_from_trades and runs it through the same rules (propkit.evaluator) and the same
day-block bootstrap (propkit.bootstrap, mode "days"):
  * costs x1.5 and x2 (cost_model.multiplied(k)): every BUY fill is moved up by (k - 1) x (the bar's
    effective spread + markup + slippage) and every SELL fill down by (k - 1) x (markup + slippage), so
    the fill costs grow exactly as they would with the multiplied cost model (buy = bid + spread + markup
    + slippage, sell = bid - markup - slippage); commission and swap come from the multiplied model.
    Under a flat rate the fills stay at the bid and only the fee grows. The trades themselves (times,
    sizes, exits) are kept: a stop that a wider spread would have hit earlier is NOT re-simulated;
  * entry +1 bar: each entry is re-priced at the open of the bar after its entry bar (the fill a
    one-bar-late order would get: ask + markup + slippage for longs, bid - markup - slippage for
    shorts), same size, same exit; a trade whose exit is at or before that open is dropped (counted);
  * drop the top 5% of winning trades (the ceil(5% x winners) largest pnl_usd);
  * drop the best month and, separately, the best year (by the summed pnl_usd of the trades that EXIT
    in it; months and years of the prop day, i.e. the CE(S)T calendar date);
  * session splits (asia / london / newyork, propkit.calendar.session_mask at the ENTRY instant; london
    and newyork overlap, so a trade can count in both; "none" = outside all three) and volatility
    splits (terciles of the Wilder ATR(14) of the last bar CLOSED before the entry, so no look-ahead);
  * trade-order bootstrap of the drawdown: the trades' net PnL in random order (permutations, numpy
    Generator(seed)), the closed-balance max drawdown of each order (USD and % of C0) against the
    historical order. This only shows how much the drawdown depends on the order of the same trades.
R multiples in the stress tables keep the BASE risk_usd (1R as planned at the base costs), so they are
comparable across scenarios. Money is USD; *_pct columns are PERCENT of C0 (3.0 = 3%).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from propkit import adapters
from propkit import bars as bars_mod
from propkit import bootstrap as boot
from propkit import calendar
from propkit import indicators as ind
from propkit.costs import CostModel
from propkit.equity import bar_of_instants, equity_from_trades
from propkit.evaluator import evaluate_path, r_summary
from propkit.rules import PropRules

COST_MULTIPLIERS = (1.5, 2.0)
TOP_WINNER_SHARE = 0.05
STRESS_SESSIONS = ("asia", "london", "newyork")
ATR_PERIOD = 14
DD_QUANTILES = (50, 75, 90, 95, 99)
DEFAULT_DD_SIMS = 2000
GROUP_COLUMNS = ("group", "n_trades", "share", "net_pnl_usd", "mean_pnl_usd", "win_rate", "mean_r", "se_r",
                 "profit_factor")
SCENARIO_COLUMNS = ("scenario", "n_trades", "net_pnl_usd", "final_balance", "mean_r", "profit_factor",
                    "path_status", "max_dd_pct", "p_pass", "p_breach_daily", "p_breach_any", "note")


# ---------------------------------------------------------------------------------------
# helpers

def _bars_arrays(bars: pd.DataFrame, cost_model: CostModel) -> dict[str, Any]:
    b = bars_mod.validate_bars(bars, source="bars")
    times = b["time"].to_numpy(dtype=np.int64)
    return {"bars": b, "time": times, "open": b["open"].to_numpy(dtype=np.float64),
            "high": b["high"].to_numpy(dtype=np.float64), "low": b["low"].to_numpy(dtype=np.float64),
            "close": b["close"].to_numpy(dtype=np.float64), "spread": cost_model.bar_spreads(b),
            "bar_seconds": bars_mod.infer_bar_seconds(times)}


def _fill_adjustments(cost_model: CostModel, spread: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-bar extra cost of one buy fill (USD/oz) and of one sell fill, at the BASE model."""
    if cost_model.is_flat:
        return np.zeros(spread.size), 0.0
    side_cost = cost_model.markup_per_side + cost_model.slippage_per_side
    return spread + side_cost, side_cost


def _with_ledger(tr: pd.DataFrame) -> pd.DataFrame:
    """TRADES with commission/swap/pnl cleared (equity_from_trades fills them again)."""
    out = adapters.validate_trades(tr, allow_extra=True)
    for col in ("commission_usd", "swap_usd", "pnl_usd"):
        out[col] = np.nan
    return out


def _r_values(trades: pd.DataFrame) -> np.ndarray:
    risk = trades["risk_usd"].to_numpy(dtype=np.float64)
    pnl = trades["pnl_usd"].to_numpy(dtype=np.float64)
    ok = np.isfinite(risk) & (risk > 0)
    return pnl[ok] / risk[ok]


# ---------------------------------------------------------------------------------------
# trade-list transformations

def reprice_costs(bars: pd.DataFrame, trades: pd.DataFrame, cost_model: CostModel, k: float) -> pd.DataFrame:
    """The same trades with every fill cost x k (k >= 1); returns TRADES (ledger columns cleared).

    Buy fills (long entries, short exits) move UP by (k - 1) x (effective bar spread + markup + slippage);
    sell fills (short entries, long exits) move DOWN by (k - 1) x (markup + slippage) - the extra cost the
    multiplied model charges on each fill (CostModel.buy_fill / sell_fill). Under a flat rate the fills do
    not move (the fee is commission, scaled by cost_model.multiplied(k) in equity_from_trades). Pass the
    result with cost_model.multiplied(k) to equity_from_trades. stop_price and risk_usd are kept (the
    planned 1R at the base costs). bars: BARS; trades: TRADES; cost_model: the BASE model.
    """
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, float, np.integer, np.floating)) \
            or not math.isfinite(float(k)) or float(k) < 1.0:
        raise ValueError(f"the cost multiplier k must be a number >= 1 (1.5 and 2 are the stress runs), got {k!r}")
    tr = _with_ledger(trades)
    if len(tr) == 0:
        return tr
    a = _bars_arrays(bars, cost_model)
    buy_adj, sell_adj = _fill_adjustments(cost_model, a["spread"])
    e = bar_of_instants(a["time"], tr["entry_time"].to_numpy(dtype=np.int64), a["bar_seconds"], "entry_time")
    x = bar_of_instants(a["time"], tr["exit_time"].to_numpy(dtype=np.int64), a["bar_seconds"], "exit_time")
    long_ = tr["side"].to_numpy() > 0
    extra = float(k) - 1.0
    entry = tr["entry_price"].to_numpy(dtype=np.float64)
    exit_ = tr["exit_price"].to_numpy(dtype=np.float64)
    tr["entry_price"] = np.where(long_, entry + extra * buy_adj[e], entry - extra * sell_adj)
    tr["exit_price"] = np.where(long_, exit_ - extra * sell_adj, exit_ + extra * buy_adj[x])
    if (tr["entry_price"] <= 0).any() or (tr["exit_price"] <= 0).any():
        raise ValueError("a stressed fill price would be <= 0; check the cost model")
    return tr


def delay_entries(bars: pd.DataFrame, trades: pd.DataFrame, cost_model: CostModel,
                  delay_bars: int = 1) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Each entry re-priced `delay_bars` bars later: at the OPEN of the bar delay_bars after the entry bar.

    Longs fill at cost_model.buy_fill(open, spread), shorts at sell_fill(open, spread) (spread = the
    bar's effective spread); size and exit are unchanged. A trade whose exit_time is at or before the
    new entry instant, or that has no bar there, is dropped. A stop that is no longer on the right side
    of the new entry is cleared (stop_price NaN); risk_usd keeps the planned base 1R. Returns (TRADES
    with ledger columns cleared, info dict: n_in, n_dropped, n_stop_cleared).
    """
    if isinstance(delay_bars, (bool, np.bool_)) or not isinstance(delay_bars, (int, np.integer)) or delay_bars < 1:
        raise ValueError(f"delay_bars must be a whole number >= 1, got {delay_bars!r}")
    tr = _with_ledger(trades)
    info = {"n_in": int(len(tr)), "n_dropped": 0, "n_stop_cleared": 0, "delay_bars": int(delay_bars)}
    if len(tr) == 0:
        return tr, info
    a = _bars_arrays(bars, cost_model)
    n = a["time"].size
    e = bar_of_instants(a["time"], tr["entry_time"].to_numpy(dtype=np.int64), a["bar_seconds"], "entry_time")
    k = e + int(delay_bars)
    has_bar = k < n
    kk = np.minimum(k, n - 1)
    new_t = a["time"][kk]
    keep = has_bar & (tr["exit_time"].to_numpy(dtype=np.int64) > new_t)
    info["n_dropped"] = int((~keep).sum())
    tr = tr.loc[keep].reset_index(drop=True)
    kk = kk[keep]
    if len(tr) == 0:
        return tr, info
    long_ = tr["side"].to_numpy() > 0
    o, s = a["open"][kk], a["spread"][kk]
    buy = np.asarray(cost_model.buy_fill(o, s), dtype=np.float64)
    sell = np.asarray(cost_model.sell_fill(o, s), dtype=np.float64)
    tr["entry_time"] = a["time"][kk].astype(np.int64)
    tr["entry_price"] = np.where(long_, buy, sell)
    stop = tr["stop_price"].to_numpy(dtype=np.float64)
    wrong = np.isfinite(stop) & ((long_ & (stop >= tr["entry_price"].to_numpy()))
                                 | (~long_ & (stop <= tr["entry_price"].to_numpy())))
    info["n_stop_cleared"] = int(wrong.sum())
    tr.loc[wrong, "stop_price"] = np.nan
    return tr, info


def drop_top_winners(trades: pd.DataFrame, share: float = TOP_WINNER_SHARE) -> tuple[pd.DataFrame, dict[str, Any]]:
    """TRADES without the ceil(share x number of winners) largest winning trades (by pnl_usd > 0).

    share: a fraction (0.05 = the top 5% of winners). Returns (TRADES with ledger columns cleared, info:
    n_winners, n_dropped, dropped_pnl_usd). trades must have pnl_usd (from equity_from_trades)."""
    if not 0.0 < float(share) < 1.0:
        raise ValueError(f"share must be a fraction between 0 and 1 (0.05 = 5%), got {share!r}")
    pnl = trades["pnl_usd"].to_numpy(dtype=np.float64)
    if not np.isfinite(pnl).all():
        raise ValueError("drop_top_winners needs pnl_usd on every trade (run equity_from_trades first)")
    winners = np.flatnonzero(pnl > 0)
    k = int(math.ceil(float(share) * winners.size)) if winners.size else 0
    order = winners[np.argsort(-pnl[winners], kind="mergesort")][:k]
    keep = np.ones(len(trades), dtype=bool)
    keep[order] = False
    info = {"n_winners": int(winners.size), "n_dropped": int(k), "dropped_pnl_usd": float(pnl[order].sum())}
    return _with_ledger(trades.loc[keep].reset_index(drop=True)), info


def drop_best_period(trades: pd.DataFrame, period: str = "month",
                     day_boundary: str = calendar.DEFAULT_DAY_BOUNDARY) -> tuple[pd.DataFrame, dict[str, Any]]:
    """TRADES without those exiting in the best calendar month or year (highest summed pnl_usd).

    period: "month" or "year", of the prop day (CE(S)T date, or the firm day of day_boundary,
    propkit.calendar.firm_day) of exit_time. Returns (TRADES with ledger columns cleared, info: period,
    label e.g. '2024-03', its pnl_usd and n_dropped; label None when there are no trades)."""
    if period not in ("month", "year"):
        raise ValueError(f"period must be 'month' or 'year', got {period!r}")
    info: dict[str, Any] = {"period": period, "label": None, "pnl_usd": 0.0, "n_dropped": 0}
    if len(trades) == 0:
        return _with_ledger(trades), info
    pnl = trades["pnl_usd"].to_numpy(dtype=np.float64)
    if not np.isfinite(pnl).all():
        raise ValueError("drop_best_period needs pnl_usd on every trade (run equity_from_trades first)")
    days = np.asarray(calendar.firm_day(trades["exit_time"].to_numpy(dtype=np.int64), day_boundary), dtype=np.int64)
    text = np.asarray(calendar.day_to_str(days), dtype=object)
    labels = np.array([t[:7] if period == "month" else t[:4] for t in text], dtype=object)
    sums = pd.Series(pnl).groupby(labels).sum()
    best = str(sums.idxmax())
    keep = labels != best
    info.update(label=best, pnl_usd=float(sums.max()), n_dropped=int((~keep).sum()))
    return _with_ledger(trades.loc[keep].reset_index(drop=True)), info


# ---------------------------------------------------------------------------------------
# split tables

def group_table(trades: pd.DataFrame, groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per named group of trades (a bool mask each): n_trades, share of all trades, net and mean
    pnl_usd (USD), win_rate (pnl > 0), mean_r / se_r (pnl_usd / risk_usd where risk_usd > 0; se = sample
    sd / sqrt(n), None for n < 2) and profit_factor (gross wins / gross losses; None without a loss)."""
    pnl_all = trades["pnl_usd"].to_numpy(dtype=np.float64)
    n_all = max(len(trades), 1)
    rows = []
    for name, mask in groups.items():
        m = np.asarray(mask, dtype=bool)
        pnl = pnl_all[m]
        sub = trades.loc[m]
        r = _r_values(sub) if len(sub) else np.zeros(0)
        losses = -pnl[pnl < 0].sum()
        rows.append({
            "group": name, "n_trades": int(m.sum()), "share": float(m.sum() / n_all),
            "net_pnl_usd": float(pnl.sum()), "mean_pnl_usd": float(pnl.mean()) if pnl.size else None,
            "win_rate": float((pnl > 0).mean()) if pnl.size else None,
            "mean_r": float(r.mean()) if r.size else None,
            "se_r": float(r.std(ddof=1) / math.sqrt(r.size)) if r.size > 1 else None,
            "profit_factor": float(pnl[pnl > 0].sum() / losses) if losses > 0 else None,
        })
    return pd.DataFrame(rows, columns=list(GROUP_COLUMNS))


def session_split(trades: pd.DataFrame) -> pd.DataFrame:
    """group_table by session at the ENTRY instant: 'asia', 'london', 'newyork' (propkit.calendar
    session_mask; london and newyork overlap, a trade can count in both) and 'none' (outside all three)."""
    et = trades["entry_time"].to_numpy(dtype=np.int64)
    groups: dict[str, np.ndarray] = {}
    any_ = np.zeros(et.size, dtype=bool)
    for name in STRESS_SESSIONS:
        m = np.asarray(calendar.session_mask(et, name), dtype=bool) if et.size else np.zeros(0, dtype=bool)
        groups[name] = m
        any_ |= m
    groups["none"] = ~any_
    return group_table(trades, groups)


def entry_atr(bars: pd.DataFrame, trades: pd.DataFrame, atr_period: int = ATR_PERIOD) -> np.ndarray:
    """Wilder ATR (USD/oz, propkit.indicators.atr) of the last bar CLOSED before each entry: bar e - 1 for an
    entry in (or at the open of) bar e. NaN in the warm-up or for an entry in the first bar."""
    b = bars_mod.validate_bars(bars, source="bars")
    times = b["time"].to_numpy(dtype=np.int64)
    a = ind.atr(b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy(), atr_period)
    if len(trades) == 0:
        return np.zeros(0)
    e = bar_of_instants(times, trades["entry_time"].to_numpy(dtype=np.int64), bars_mod.infer_bar_seconds(times),
                        "entry_time")
    out = np.full(e.size, np.nan)
    ok = e >= 1
    out[ok] = a[e[ok] - 1]
    return out


def volatility_split(bars: pd.DataFrame, trades: pd.DataFrame, atr_period: int = ATR_PERIOD) -> pd.DataFrame:
    """group_table by tercile of entry_atr (low / mid / high, cut at the 1/3 and 2/3 quantiles of the
    trades' ATR values; 'warm-up' = no ATR yet). Adds atr_from / atr_to columns (USD/oz)."""
    atr = entry_atr(bars, trades, atr_period)
    ok = np.isfinite(atr)
    if ok.sum() >= 3:
        q1, q2 = (float(v) for v in np.quantile(atr[ok], [1 / 3, 2 / 3]))
    else:
        q1 = q2 = float(np.nanmax(atr)) if ok.any() else 0.0
    groups = {"low": ok & (atr <= q1), "mid": ok & (atr > q1) & (atr <= q2), "high": ok & (atr > q2),
              "warm-up": ~ok}
    table = group_table(trades, groups)
    lo = float(atr[ok].min()) if ok.any() else None
    hi = float(atr[ok].max()) if ok.any() else None
    table["atr_from"] = [lo, q1 if ok.any() else None, q2 if ok.any() else None, None]
    table["atr_to"] = [q1 if ok.any() else None, q2 if ok.any() else None, hi, None]
    return table


# ---------------------------------------------------------------------------------------
# trade-order bootstrap of the drawdown

def _max_dd(pnl_rows: np.ndarray, c0: float) -> np.ndarray:
    bal = c0 + np.cumsum(pnl_rows, axis=1)
    peak = np.maximum(np.maximum.accumulate(bal, axis=1), c0)
    return np.max(peak - bal, axis=1)


def trade_order_drawdowns(trades: pd.DataFrame, C0: float, n_sims: int = DEFAULT_DD_SIMS,
                          seed: int = 7) -> dict[str, Any]:
    """Max drawdown of the CLOSED balance when the same trades come in random order.

    The historical order is exit_time order (then trade_id). Each simulation is a random permutation
    (numpy Generator(seed)) of the trades' net pnl_usd; the closed balance starts at C0 and its max
    drawdown is max(running peak - balance), peak = max(C0, balances so far). Returns a JSON-serialisable
    dict: n_trades, n_sims, seed, hist_max_dd_usd / _pct (pct of C0), share_sims_at_least_hist, and a
    'table' list of rows (quantile, max_dd_usd, max_dd_pct) for p50/p75/p90/p95/p99. Intrabar and open-
    position drawdowns are not in this measure (see the path result for those).
    """
    c0 = float(C0)
    if isinstance(n_sims, (bool, np.bool_)) or not isinstance(n_sims, (int, np.integer)) or n_sims < 1:
        raise ValueError(f"n_sims must be a whole number >= 1, got {n_sims!r}")
    order = [c for c in ("exit_time", "trade_id") if c in trades.columns]
    tr = trades.sort_values(order, kind="mergesort") if order else trades
    pnl = tr["pnl_usd"].to_numpy(dtype=np.float64)
    out: dict[str, Any] = {"n_trades": int(pnl.size), "n_sims": int(n_sims), "seed": int(seed),
                           "method": "permutations of the trades' net pnl_usd; closed balance only"}
    if pnl.size == 0:
        out.update(hist_max_dd_usd=0.0, hist_max_dd_pct=0.0, share_sims_at_least_hist=None, table=[])
        return out
    hist = float(_max_dd(pnl[None, :], c0)[0])
    rng = np.random.default_rng(seed)
    sims = np.empty(int(n_sims))
    chunk = max(1, min(int(n_sims), 2_000_000 // max(pnl.size, 1)))
    for start in range(0, int(n_sims), chunk):
        m = min(chunk, int(n_sims) - start)
        rows = rng.permuted(np.broadcast_to(pnl, (m, pnl.size)), axis=1)
        sims[start:start + m] = _max_dd(rows, c0)
    qs = np.quantile(sims, [q / 100.0 for q in DD_QUANTILES])
    out.update(hist_max_dd_usd=hist, hist_max_dd_pct=hist / c0 * 100.0,
               share_sims_at_least_hist=float((sims >= hist - 1e-9).mean()),
               table=[{"quantile": f"p{q}", "max_dd_usd": float(v), "max_dd_pct": float(v / c0 * 100.0)}
                      for q, v in zip(DD_QUANTILES, qs)])
    return out


# ---------------------------------------------------------------------------------------
# scenarios

def scenario_row(name: str, bars: pd.DataFrame, trades: pd.DataFrame, C0: float, cost_model: CostModel,
                 rules: PropRules, n_sims: int, seed: int, horizon_days, note: str = "",
                 horizon_unit: str = boot.DEFAULT_HORIZON_UNIT) -> dict[str, Any]:
    """Rebuild the path of `trades` (equity_from_trades with cost_model), evaluate it under `rules` and
    bootstrap it (mode "days", n_sims, seed, horizon_days, horizon_unit). Returns one row of
    SCENARIO_COLUMNS."""
    eq, tr = equity_from_trades(bars, trades, C0, cost_model, price_tolerance=None)
    res = evaluate_path(eq, tr, rules)
    bs = boot.bootstrap_challenges(eq, rules, trades=tr, mode="days", n_sims=n_sims, seed=seed,
                                   horizon_days=horizon_days, horizon_unit=horizon_unit)
    rs = r_summary(tr)
    return {"scenario": name, "n_trades": int(len(tr)), "net_pnl_usd": float(tr["pnl_usd"].sum()) if len(tr) else 0.0,
            "final_balance": float(eq["balance"].iloc[-1]), "mean_r": rs["mean_r"] if rs else None,
            "profit_factor": rs["profit_factor"] if rs else _profit_factor(tr),
            "path_status": res.status, "max_dd_pct": res.max_dd_pct, "p_pass": bs.p_pass,
            "p_breach_daily": bs.p_breach_daily, "p_breach_any": bs.p_breach_any, "note": note}


def _profit_factor(tr: pd.DataFrame) -> float | None:
    if len(tr) == 0:
        return None
    pnl = tr["pnl_usd"].to_numpy(dtype=np.float64)
    losses = -pnl[pnl < 0].sum()
    return float(pnl[pnl > 0].sum() / losses) if losses > 0 else None


def run_stress(bars: pd.DataFrame, trades: pd.DataFrame, C0: float, cost_model: CostModel, rules: PropRules,
               n_sims: int = boot.DEFAULT_N_SIMS, seed: int = boot.DEFAULT_SEED,
               horizon_days=boot.DEFAULT_HORIZON_DAYS, dd_sims: int = DEFAULT_DD_SIMS,
               horizon_unit: str = boot.DEFAULT_HORIZON_UNIT) -> dict[str, Any]:
    """All stresses of the module docstring for one run.

    bars: BARS; trades: the run's TRADES (after equity_from_trades, so pnl_usd is filled); C0: USD;
    cost_model: the BASE cost model; rules: PropRules; n_sims / seed / horizon_days / horizon_unit: the
    day-block bootstrap of every scenario (same seed: common random numbers across scenarios); dd_sims: orders in
    the trade-order bootstrap. Returns a dict of DataFrames ('scenarios', 'sessions', 'volatility') and
    'drawdown_order' (dict), plus 'notes' (list of str).
    """
    base = _with_ledger(trades)
    kw = dict(C0=C0, rules=rules, n_sims=n_sims, seed=seed, horizon_days=horizon_days, horizon_unit=horizon_unit)
    rows = [scenario_row("base", bars, base, cost_model=cost_model, note="the run as given", **kw)]
    for k in COST_MULTIPLIERS:
        rows.append(scenario_row(f"costs x{k:g}", bars, reprice_costs(bars, trades, cost_model, k),
                                 cost_model=cost_model.multiplied(k), note="same trades, every cost x k", **kw))
    delayed, dinfo = delay_entries(bars, trades, cost_model, 1)
    rows.append(scenario_row("entry +1 bar", bars, delayed, cost_model=cost_model,
                             note=f"{dinfo['n_dropped']} trade(s) dropped, {dinfo['n_stop_cleared']} stop(s) cleared",
                             **kw))
    top, tinfo = drop_top_winners(trades)
    rows.append(scenario_row("drop top 5% winners", bars, top, cost_model=cost_model,
                             note=f"{tinfo['n_dropped']} of {tinfo['n_winners']} winners dropped "
                                  f"({tinfo['dropped_pnl_usd']:,.2f} USD)", **kw))
    for period in ("month", "year"):
        dropped, pinfo = drop_best_period(trades, period, rules.day_boundary)
        label = pinfo["label"] or "none"
        rows.append(scenario_row(f"drop best {period}", bars, dropped, cost_model=cost_model,
                                 note=f"{label}: {pinfo['n_dropped']} trade(s), {pinfo['pnl_usd']:,.2f} USD", **kw))
    notes = [
        "Costs xk keep the same trades and move every fill by the extra cost; stops are not re-simulated.",
        "Entry +1 bar re-prices each entry at the next bar's open with the same exit.",
        "R multiples use the base 1R (risk_usd as planned) in every scenario.",
        "P(pass) and P(breach) come from the day-block bootstrap with the same seed in every scenario.",
        "Sessions use the entry time; London and New York overlap, so a trade entered in the overlap counts in "
        "both groups and the shares can add up to more than 1.",
    ]
    return {"scenarios": pd.DataFrame(rows, columns=list(SCENARIO_COLUMNS)),
            "sessions": session_split(trades), "volatility": volatility_split(bars, trades),
            "drawdown_order": trade_order_drawdowns(trades, C0, n_sims=dd_sims, seed=seed), "notes": notes}
