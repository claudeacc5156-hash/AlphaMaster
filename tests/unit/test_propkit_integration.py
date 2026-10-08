"""Integration tests for propkit: the whole pipeline on synthetic bars, the stress tests, the report, the
public API, the source scan and the timing budget. Research only; synthetic data only."""
from __future__ import annotations

import ast
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import propkit
from propkit import adapters, bootstrap, calendar, cli, report, rules, stress
from propkit.bars import synthetic_bars
from propkit.costs import CostModel
from propkit.equity import equity_from_positions, equity_from_trades

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "propkit"
C0 = 100_000.0
H = 3600
START = 1708905600            # 2024-02-26 00:00 UTC, a Monday; the data runs across both 2024 DST changes
COSTS = CostModel(markup_per_side=0.05, slippage_per_side=0.02, commission_per_lot_round_trip=7.0)
WIDE = rules.custom(name="wide", profit_target_pct=None, daily_loss_pct=0.5, max_loss_pct=0.9)
FORBIDDEN = {"model_core", "data_pipeline", "config", "web", "utils", "strategy_manager", "execution", "scripts",
             "scipy", "zoneinfo", "torch"}


def utc(text: str) -> int:
    return int(pd.Timestamp(text, tz="UTC").timestamp())


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    """~10 months of metals-hours H1 bars with a spread column, weekends and four DST changes."""
    return synthetic_bars(START, 5200, seed=5, spread=0.34)


def hand_trades(bars: pd.DataFrame, cm: CostModel, every: int = 37, hold: int = 11, size: float = 30.0):
    """Round trips entered at bar opens every `every` bars, held `hold` bars (some across weekends),
    alternately long and short, filled at the cost model's ask/bid fills; stops 6 USD away."""
    t = bars["time"].to_numpy()
    o = bars["open"].to_numpy()
    s = cm.bar_spreads(bars)
    rows = []
    for k, i in enumerate(range(20, len(bars) - hold - 1, every)):
        side = 1 if k % 2 == 0 else -1
        j = i + hold + (k % 3) * 40          # every third trade is held about two days longer
        if j >= len(bars):
            break
        ep = cm.buy_fill(o[i], s[i]) if side > 0 else cm.sell_fill(o[i], s[i])
        xp = cm.sell_fill(o[j], s[j]) if side > 0 else cm.buy_fill(o[j], s[j])
        rows.append({"side": side, "units": size, "entry_time": int(t[i]), "entry_price": float(ep),
                     "exit_time": int(t[j]), "exit_price": float(xp), "exit_reason": "time",
                     "stop_price": float(ep - 6.0 if side > 0 else ep + 6.0)})
    return adapters.validate_trades(pd.DataFrame(rows))


def ema_positions(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"].to_numpy()
    fast = propkit.indicators.ema(close, 20)
    slow = propkit.indicators.ema(close, 80)
    p = np.where(fast > slow, 1.0, -1.0)
    p[:80] = 0.0
    return adapters.positions_from_alphamaster(bars["time"].to_numpy(), p, keep_raw=True)


# ---------------------------------------------------------------------------------------
# public API and source hygiene

def test_public_api_names_exist():
    assert isinstance(propkit.__version__, str) and propkit.__version__.count(".") == 2
    missing = [name for name in propkit.__all__ if not hasattr(propkit, name)]
    assert missing == []
    for name in ("load_bars", "CostModel", "preset", "equity_from_trades", "equity_from_positions", "evaluate_path",
                 "bootstrap_challenges", "max_size", "analyse", "render_markdown", "run_stress", "PullbackSpec",
                 "run_selftest"):
        assert name in propkit.__all__


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="ascii"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in (
                "import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant):
            roots.add(str(node.args[0].value).split(".")[0])
    return roots


def test_propkit_imports_nothing_from_alphamaster_and_is_ascii():
    files = sorted(PKG.rglob("*.py"))
    assert len(files) >= 17
    for f in files:
        text = f.read_bytes()
        assert text.isascii(), f"{f.name} has non-ASCII bytes"
        bad = _imported_roots(f) & FORBIDDEN
        assert not bad, f"{f.name} imports {sorted(bad)}"
    for f in sorted(PKG.rglob("*.md")) + sorted(PKG.rglob("*.json")):
        assert f.read_bytes().isascii(), f.name


def test_every_public_function_has_a_docstring():
    """Contract: every public function (module level, and methods of public classes) has a docstring."""
    missing = []
    for f in sorted(PKG.glob("*.py")):
        tree = ast.parse(f.read_text(encoding="ascii"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                if ast.get_docstring(node) is None:
                    missing.append(f"{f.name}:{node.name}")
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                if ast.get_docstring(node) is None:
                    missing.append(f"{f.name}:class {node.name}")
                for m in node.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith("_") \
                            and ast.get_docstring(m) is None:
                        missing.append(f"{f.name}:{node.name}.{m.name}")
    assert missing == []


def test_old_pullback_readme_was_merged():
    assert not (PKG / "README_pullback_fields.md").exists()
    readme = (PKG / "README.md").read_text(encoding="ascii")
    methods = (PKG / "METHODS.md").read_text(encoding="ascii")
    assert "RESEARCH ONLY - not trading advice" in readme and "RESEARCH ONLY - not trading advice" in methods
    assert "research\\data\\xauusd\\train\\XAUUSD_H1.parquet" in readme and "--out logs\\" in readme
    assert "python -m propkit selftest" in readme and "PLACEHOLDER" in readme and "METHODS.md" in readme
    for field in propkit.PullbackSpec.__dataclass_fields__:   # the spec fields table lives in METHODS.md
        assert f"| `{field}` |" in methods


def test_readme_is_one_page():
    """Contract: propkit/README.md is a one-page quick start; the detail is in METHODS.md."""
    readme = (PKG / "README.md").read_text(encoding="ascii")
    assert len(readme.splitlines()) <= 70 and len(readme.split()) <= 800
    for command in ("propkit selftest", "propkit rules", "propkit evaluate", "propkit pullback",
                    "export_positions.py"):
        assert command in readme, command
    for name in ("report.md", "report.json", "equity.csv", "trades.csv", "days.csv"):
        assert name in readme, name


# ---------------------------------------------------------------------------------------
# end to end

def _check_report(rep: dict, n_trades: int) -> str:
    text = json.dumps(rep, allow_nan=False)                        # JSON-serialisable, no NaN
    assert json.loads(text) == rep
    assert rep["header"] == "RESEARCH ONLY - not trading advice"
    for key in ("data", "input", "rules", "costs", "path", "bootstrap_days", "bootstrap_weeks", "max_size", "stats",
                "stress", "settings", "timing_seconds", "strategy_card"):
        assert key in rep, key
    for b in (rep["bootstrap_days"], rep["bootstrap_weeks"]):
        total = b["p_pass"] + b["p_breach_daily"] + b["p_breach_max"] + b["p_timeout"]
        assert total == pytest.approx(1.0, abs=1e-12)
    assert rep["stats"]["n_trades"] == n_trades
    md = report.render_markdown(rep)
    assert md.isascii() and md.splitlines()[0] == "RESEARCH ONLY - not trading advice"
    for title in ("## Summary", "## Data", "## Historical path", "## Bootstrap of challenges", "## Largest size",
                  "## Statistics", "## Stress tests", "## STRATEGY CARD (skeleton)", "[U]", "[ASSUMPTION]"):
        assert title in md, title
    assert report.render_markdown(json.loads(text)) == md         # report.json renders the same report
    assert "+- 0.0000" not in md                                    # p = 0 or 1: the rule-of-three bound instead
    return md


def test_end_to_end_trades_input(bars, tmp_path):
    path = tmp_path / "XAUUSD_H1.parquet"
    bars.to_parquet(path, index=False)
    raw = hand_trades(bars, COSTS)
    equity, trades = equity_from_trades(bars, raw, C0, COSTS)
    assert len(trades) == len(raw) > 100
    # the historical path under wide rules runs to the end: final balance = C0 + the trades' net PnL
    rep = report.analyse(bars, trades, equity, WIDE, COSTS, bars_path=path, n_sims=300, seed=3, dd_sims=200,
                         input_info={"kind": "trades"})
    md = _check_report(rep, len(trades))
    assert rep["path"]["status"] == "running"
    # no target and wide floors: every probability is 0, shown with its bound in the card and the size table
    assert "| P(pass) / P(daily breach) | 0.0000 (< 0.0100) / 0.0000 (< 0.0100) |" in md
    size_rows = [ln for ln in md.splitlines() if ln.startswith("| P(daily breach) <= alpha |")
                 or ln.startswith("| P(any breach) <= alpha |")]
    assert len(size_rows) == 2 and all("0.0000 (< 0.0100)" in ln for ln in size_rows), size_rows
    assert rep["path"]["final_balance"] == pytest.approx(C0 + trades["pnl_usd"].sum(), abs=1e-6)
    assert rep["data"]["n_bars"] == len(bars) and rep["data"]["bar_minutes"] == 60
    assert rep["data"]["sha256"] == report.file_sha256(path)
    assert rep["data"]["spread_median"] == pytest.approx(float(np.median(bars["spread"])))
    # prop days with bars: the CE(S)T dates of the bar opens (DST-correct, weekends absent)
    days = np.unique(calendar.prop_day(bars["time"].to_numpy()))
    assert rep["stats"]["per_day"]["n_days"] == days.size == len(rep["path"]["days"])
    # the same run under FTMO 1-Step; the stress table holds every scenario
    rep2 = report.analyse(bars, trades, equity, rules.preset("ftmo-1step"), COSTS, n_sims=300, dd_sims=200)
    names = [row["scenario"] for row in rep2["stress"]["scenarios"]]
    assert names == ["base", "costs x1.5", "costs x2", "entry +1 bar", "drop top 5% winners", "drop best month",
                     "drop best year"]
    base = rep2["stress"]["scenarios"][0]
    assert base["path_status"] == rep2["path"]["status"]
    assert base["p_pass"] == rep2["bootstrap_days"]["p_pass"]     # same seed, same units: same answer
    nets = [row["net_pnl_usd"] for row in rep2["stress"]["scenarios"][:3]]
    assert nets[0] > nets[1] > nets[2]                              # costs only ever cost money


def test_end_to_end_positions_input_units_and_leverage(bars):
    pos = ema_positions(bars)
    eq_u, tr_u = equity_from_positions(bars, pos, C0, COSTS, size_mode="units", size=20.0)
    eq_l, tr_l = equity_from_positions(bars, pos, C0, COSTS, size_mode="leverage", size=1.0)
    assert len(tr_u) == len(tr_l) > 10
    assert np.allclose(tr_u["units"], 20.0)
    assert not np.allclose(tr_l["units"], tr_l["units"].iloc[0])  # leverage re-sizes with equity
    for eq, tr in ((eq_u, tr_u), (eq_l, tr_l)):
        rep = report.analyse(bars, tr, eq, rules.preset("ftmo-2step"), COSTS, n_sims=200, dd_sims=100,
                             input_info={"kind": "positions"})
        _check_report(rep, len(tr))
        assert rep["stats"]["per_trade"]["basis"].startswith("pnl_usd / C0")    # no stops: no R multiples


def test_prop_days_follow_dst(bars):
    """Prop days start at 22:00 UTC in summer and 23:00 UTC in winter; in the US-only DST weeks the Sunday
    reopen at 22:00 UTC is a prop day of its own with one H1 bar."""
    t = bars["time"].to_numpy()
    for first, last, hour in (("2024-06-02", "2024-06-08", 22), ("2024-12-01", "2024-12-07", 23)):
        x = t[(t >= utc(first)) & (t < utc(last))]
        starts = x[np.r_[True, np.diff(calendar.prop_day(x)) != 0]]
        assert len(starts) == 5 and set(((starts % 86400) // 3600).tolist()) == {hour}
    days, counts = np.unique(calendar.prop_day(t), return_counts=True)
    one_bar = [str(x) for x in calendar.day_to_str(days[counts == 1])]
    assert one_bar == ["2024-03-10", "2024-03-17", "2024-03-24", "2024-10-27"]
    assert set(counts[1:-1].tolist()) == {1, 22, 23}                   # metals hours: 22 or 23 bars a day


# ---------------------------------------------------------------------------------------
# stress unit checks

def test_reprice_costs_k1_is_identity_and_flat_rate_keeps_fills(bars):
    raw = hand_trades(bars, COSTS)
    same = stress.reprice_costs(bars, raw, COSTS, 1.0)
    assert np.array_equal(same["entry_price"], raw["entry_price"])
    assert np.array_equal(same["exit_price"], raw["exit_price"])
    flat = CostModel(flat_rate_per_side=0.0003)
    raw_f = hand_trades(bars, flat)
    rep = stress.reprice_costs(bars, raw_f, flat, 2.0)
    assert np.array_equal(rep["entry_price"], raw_f["entry_price"])
    _, tr1 = equity_from_trades(bars, raw_f, C0, flat)
    _, tr2 = equity_from_trades(bars, rep, C0, flat.multiplied(2.0))
    assert np.allclose(tr2["commission_usd"], 2 * tr1["commission_usd"])
    with pytest.raises(ValueError, match="multiplier"):
        stress.reprice_costs(bars, raw, COSTS, 0.5)


@pytest.mark.parametrize("k", [1.5, 2.0])
def test_reprice_costs_equals_the_multiplied_cost_model(bars, k):
    """A position series run with costs x k gives exactly the repriced trades of the base run."""
    pos = ema_positions(bars)
    _, base = equity_from_positions(bars, pos, C0, COSTS, size_mode="units", size=10.0)
    _, direct = equity_from_positions(bars, pos, C0, COSTS.multiplied(k), size_mode="units", size=10.0)
    repriced = stress.reprice_costs(bars, base, COSTS, k)
    _, rebuilt = equity_from_trades(bars, repriced, C0, COSTS.multiplied(k), price_tolerance=None)
    assert np.allclose(rebuilt["entry_price"], direct["entry_price"], rtol=0, atol=1e-9)
    assert np.allclose(rebuilt["exit_price"], direct["exit_price"], rtol=0, atol=1e-9)
    assert np.allclose(rebuilt["pnl_usd"], direct["pnl_usd"], rtol=0, atol=1e-6)


def test_delay_entries_known_answer():
    t0 = utc("2024-01-09 10:00")                                      # a Tuesday, all bars open
    n = 8
    o = 2000.0 + np.arange(n, dtype=float)
    b = pd.DataFrame({"time": t0 + H * np.arange(n), "open": o, "high": o + 2, "low": o - 2, "close": o + 1,
                      "spread": 0.30})
    cm = CostModel(markup_per_side=0.10, slippage_per_side=0.05, swap_enabled=False)
    tr = adapters.validate_trades(pd.DataFrame([
        {"side": 1, "units": 10.0, "entry_time": t0 + H, "entry_price": 2001.45, "exit_time": t0 + 5 * H,
         "exit_price": 2004.85, "stop_price": 2001.0},                 # still below the delayed entry: kept
        {"side": -1, "units": 10.0, "entry_time": t0 + 2 * H, "entry_price": 2001.85, "exit_time": t0 + 3 * H,
         "exit_price": 2003.45},                                        # exits at the new entry instant: dropped
        {"side": -1, "units": 5.0, "entry_time": t0 + 3 * H + 600, "entry_price": 2002.80,
         "exit_time": t0 + 7 * H, "exit_price": 2007.45, "stop_price": 2003.5},   # below the new entry
    ]), allow_extra=False)
    out, info = stress.delay_entries(b, tr, cm, 1)
    assert info == {"n_in": 3, "n_dropped": 1, "n_stop_cleared": 1, "delay_bars": 1}
    assert out["entry_time"].tolist() == [t0 + 2 * H, t0 + 4 * H]
    assert out["entry_price"].tolist() == pytest.approx([2002.0 + 0.30 + 0.15, 2004.0 - 0.15])
    assert out["exit_time"].tolist() == [t0 + 5 * H, t0 + 7 * H]         # exits unchanged
    assert out["stop_price"].iloc[0] == 2001.0                          # still below the long entry 2002.45
    assert math.isnan(out["stop_price"].iloc[1])                        # 2003.5 is below the short entry 2003.85


def _pnl_trades(pnls, months):
    rows = []
    for i, (p, m) in enumerate(zip(pnls, months)):
        t = utc(f"2024-{m:02d}-10 10:00") + i * H
        rows.append({"trade_id": i + 1, "side": 1, "units": 1.0, "entry_time": t, "entry_price": 2000.0,
                     "exit_time": t + 1800, "exit_price": 2000.0 + p, "pnl_usd": p})
    return adapters.validate_trades(pd.DataFrame(rows))


def test_drop_top_winners_and_best_period():
    pnls = [float(v) for v in range(1, 21)] + [-5.0, -7.0]           # 20 winners, 2 losers
    tr = _pnl_trades(pnls, [1] * 10 + [2] * 12)
    out, info = stress.drop_top_winners(tr, 0.05)
    assert info == {"n_winners": 20, "n_dropped": 1, "dropped_pnl_usd": 20.0}
    kept = (out["exit_price"] - out["entry_price"]).round(9).tolist()
    assert len(out) == 21 and 20.0 not in kept and 19.0 in kept
    out, info = stress.drop_top_winners(tr, 0.20)
    assert info["n_dropped"] == 4 and info["dropped_pnl_usd"] == 20 + 19 + 18 + 17
    # 7 winners x 5% = 0.35: ceil drops one (the largest), so a short list still loses its best trade
    few = _pnl_trades([5.0, 1.0, 7.0, 3.0, 2.0, 6.0, 4.0, -1.0, -2.0], [1] * 9)
    out, info = stress.drop_top_winners(few, 0.05)
    assert info == {"n_winners": 7, "n_dropped": 1, "dropped_pnl_usd": 7.0}
    assert sorted((out["exit_price"] - out["entry_price"]).round(9).tolist()) == [-2, -1, 1, 2, 3, 4, 5, 6]
    out, info = stress.drop_best_period(tr, "month")                  # Jan 1..10 = 55; Feb 11..20 - 12 = 143
    assert info["label"] == "2024-02" and info["n_dropped"] == 12 and info["pnl_usd"] == pytest.approx(143.0)
    assert len(out) == 10
    out, info = stress.drop_best_period(tr, "year")
    assert info["label"] == "2024" and len(out) == 0
    with pytest.raises(ValueError):
        stress.drop_best_period(tr, "week")


def test_session_and_volatility_splits(bars):
    raw = hand_trades(bars, COSTS)
    _, tr = equity_from_trades(bars, raw, C0, COSTS)
    s = stress.session_split(tr).set_index("group")
    et = tr["entry_time"].to_numpy()
    for name in ("asia", "london", "newyork"):
        assert s.loc[name, "n_trades"] == int(calendar.session_mask(et, name).sum())
    assert s.loc["none", "n_trades"] == int((~(calendar.session_mask(et, "asia") | calendar.session_mask(et, "london")
                                               | calendar.session_mask(et, "newyork"))).sum())
    v = stress.volatility_split(bars, tr).set_index("group")
    assert v["n_trades"].sum() == len(tr)                             # terciles partition the trades
    assert v.loc[["low", "mid", "high"], "net_pnl_usd"].sum() + v.loc["warm-up", "net_pnl_usd"] == \
        pytest.approx(tr["pnl_usd"].sum())
    assert v.loc["low", "atr_to"] <= v.loc["mid", "atr_to"] <= v.loc["high", "atr_to"]


def _atr_bars(spike_at: int | None = None, n: int = 80) -> pd.DataFrame:
    """H1 bars: calm (range 1 USD) before bar 40, wider (range 6 USD) from bar 40; optionally one bar with
    a 200 USD range."""
    t0 = utc("2024-01-08 00:00")
    o = np.full(n, 2000.0)
    half = np.where(np.arange(n) < 40, 0.5, 3.0)
    if spike_at is not None:
        half[spike_at] = 100.0
    return pd.DataFrame({"time": t0 + H * np.arange(n, dtype=np.int64), "open": o, "high": o + half,
                         "low": o - half, "close": o, "spread": 0.3})


def test_volatility_split_uses_the_atr_known_before_entry():
    # trade A enters at the open of a 200 USD spike bar after calm bars: the ATR known before its entry is
    # the calm one, so it is the LOW-volatility trade; the spike bar's own ATR (look-ahead) would make it high
    b = _atr_bars(spike_at=20)
    t = b["time"].to_numpy()
    tr = _pnl_trades([10.0, -5.0, 3.0], [1, 1, 1])
    tr["entry_time"] = [int(t[20]), int(t[60]) + 600, int(t[70])]
    tr["exit_time"] = tr["entry_time"] + 1800
    a = stress.entry_atr(b, tr)
    full = propkit.indicators.atr(b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy(), 14)
    assert a.tolist() == pytest.approx([full[19], full[59], full[69]])
    assert a[0] < 1.5 < a[1]                                            # calm before the spike, wide later
    v = stress.volatility_split(b, tr).set_index("group")
    assert v.loc["low", "n_trades"] == 1 and v.loc["low", "net_pnl_usd"] == pytest.approx(10.0)
    # truncation: changing the entry bar and everything after it never changes the entry's ATR
    calm = _atr_bars(spike_at=None)
    assert stress.entry_atr(calm, tr)[0] == a[0]
    assert stress.entry_atr(calm.iloc[:21], tr.iloc[:1])[0] == a[0]


def test_trade_order_drawdowns_are_deterministic_and_exact():
    tr = _pnl_trades([100.0, -300.0, 50.0, -100.0, 400.0], [3] * 5)
    a = stress.trade_order_drawdowns(tr, 1000.0, n_sims=500, seed=11)
    b = stress.trade_order_drawdowns(tr, 1000.0, n_sims=500, seed=11)
    assert a == b
    assert a["hist_max_dd_usd"] == 350.0 and a["hist_max_dd_pct"] == 35.0   # peak 1100 -> 750
    worst = max(r["max_dd_usd"] for r in a["table"])
    assert worst <= 400.0 + 1e-9                                        # all losses in a row: 400
    assert 0.0 <= a["share_sims_at_least_hist"] <= 1.0


def test_trade_order_drawdowns_follow_the_permutation_distribution():
    # the 120 orders of these 5 trades have closed-balance drawdowns 300 (60 orders), 350 (12) and 400 (48):
    # P(DD >= the historical 350) = 0.5 exactly. Keeping the historical order would give 1.0 and a single value.
    import itertools
    pnl = [100.0, -300.0, 50.0, -100.0, 400.0]
    exact = [float(stress._max_dd(np.array([o]), 1000.0)[0]) for o in itertools.permutations(pnl)]
    assert sorted(set(exact)) == [300.0, 350.0, 400.0] and np.mean(np.array(exact) >= 350.0) == 0.5
    n = 20_000
    d = stress.trade_order_drawdowns(_pnl_trades(pnl, [3] * 5), 1000.0, n_sims=n, seed=11)
    assert abs(d["share_sims_at_least_hist"] - 0.5) < 5 * math.sqrt(0.25 / n)
    table = {r["quantile"]: r["max_dd_usd"] for r in d["table"]}
    assert table["p75"] == table["p99"] == 400.0                        # P(DD <= 350) = 0.6 < 0.75
    # numbers (not only the stored seed) change with the seed on a longer list of mixed-sign trades
    rng = np.random.default_rng(3)
    many = _pnl_trades(list(np.round(rng.normal(5.0, 60.0, 40), 2)), [3] * 40)
    x = stress.trade_order_drawdowns(many, 10_000.0, n_sims=300, seed=1)
    y = stress.trade_order_drawdowns(many, 10_000.0, n_sims=300, seed=2)
    assert x["table"] != y["table"] and x["table"][0]["max_dd_usd"] < x["table"][-1]["max_dd_usd"]


# ---------------------------------------------------------------------------------------
# regression: an exit at the last bar's close that lands on 00:00 CE(S)T

def test_bootstrap_books_an_exit_at_the_bar_end_on_the_bar_day():
    start = utc("2024-07-08 00:00")
    n = (utc("2024-07-10 22:00") - start) // H                          # last bar 21:00 UTC; closes at 00:00 CEST
    b = synthetic_bars(start, n, seed=3, market_hours="always")
    pos = adapters.positions_from_alphamaster(b["time"].to_numpy(), np.ones(len(b)))
    eq, tr = equity_from_positions(b, pos, C0, CostModel(), size=10.0)
    assert tr["exit_reason"].iloc[-1] == "end_of_data"
    assert int(tr["exit_time"].iloc[-1]) == int(b["time"].iloc[-1]) + H
    units = bootstrap.build_day_units(eq, C0, tr)                       # raised ValueError before the fix
    assert units.trade_count.sum() == len(tr)
    assert units.trade_count[-1] == 1 and units.day[-1] == calendar.prop_day(int(b["time"].iloc[-1]))
    res = bootstrap.bootstrap_challenges(eq, rules.preset("ftmo-1step"), trades=tr, mode="trades", n_sims=50, seed=1)
    assert 0.0 <= res.p_pass <= 1.0


# ---------------------------------------------------------------------------------------
# selftest and timing

def test_selftest_passes_every_gate():
    lines: list[str] = []
    n_pass, n_fail = propkit.run_selftest(out=lines.append)
    assert n_fail == 0 and n_pass == 29, "\n".join(lines)
    assert all(line.isascii() for line in lines)
    assert sum(line.startswith("PASS  ") for line in lines) == 29
    assert lines[-1].startswith("29 of 29 gates passed")


def test_selftest_stats_gates_exercise_propkit_stats(monkeypatch):
    """Finding 12: gate C3 used its own formula, so a broken stats.sharpe_stats still passed; C3 and C4 must
    call propkit (sharpe_stats, expected_shortfall, daily_returns_from_equity, drawdown_stats)."""
    from propkit import selftest, stats
    real = stats.sharpe_stats

    def wrong_se(returns, periods_per_year):
        return dataclasses.replace(real(returns, periods_per_year), se_sr=0.5)
    monkeypatch.setattr(stats, "sharpe_stats", wrong_se)
    for gate in (selftest.gate_sr_standard_error, selftest.gate_stats_known_answers):
        with pytest.raises(selftest.GateFailure):
            gate()
    monkeypatch.setattr(stats, "sharpe_stats", real)
    monkeypatch.setattr(stats, "expected_shortfall", lambda returns, alpha=0.05: -1.0)
    with pytest.raises(selftest.GateFailure):
        selftest.gate_stats_known_answers()


def test_selftest_import_scan_reads_the_syntax_tree():
    """Finding 26: the regex scan missed imports after a semicolon or a colon on one line and import_module."""
    from propkit.selftest import imported_modules
    src = ("import os, json\nif True: import scipy.stats\nx = 1; from model_core import backtest\n"
           "try:\n    import zoneinfo\nexcept ImportError:\n    pass\n"
           "import importlib\nm = importlib.import_module('data_pipeline.loader')\nfrom . import bars\n"
           "text = 'import torch'\n")
    assert imported_modules(src) == {"os", "json", "scipy", "model_core", "zoneinfo", "importlib", "data_pipeline"}


def test_evaluate_63645_h1_bars_with_10000_sims_under_60_seconds(tmp_path, capsys):
    b = synthetic_bars(utc("2015-01-01"), 63_645, seed=11, price=1200.0, vol_per_hour=0.0025, spread=0.34)
    bars_file = tmp_path / "XAUUSD_H1.parquet"
    b.to_parquet(bars_file, index=False)
    close = b["close"].to_numpy()
    p = np.where(propkit.indicators.ema(close, 50) > propkit.indicators.ema(close, 200), 1.0, -1.0)
    p[:200] = 0.0
    pos_file = tmp_path / "positions.csv"
    adapters.write_positions_csv(adapters.positions_from_alphamaster(b["time"].to_numpy(), p, keep_raw=True), pos_file)
    out = tmp_path / "out"
    t0 = time.perf_counter()
    code = cli.main(["evaluate", "--bars", str(bars_file), "--positions", str(pos_file), "--size-mode", "units",
                     "--size", "30", "--rules", "ftmo-1step", "--n-sims", "10000", "--seed", "7", "--out", str(out)])
    elapsed = time.perf_counter() - t0
    assert code == 0, capsys.readouterr().err
    assert elapsed < 60.0, f"evaluate took {elapsed:.1f} s"
    rep = json.loads((out / "report.json").read_text(encoding="ascii"))
    assert rep["data"]["n_bars"] == 63_645 and rep["settings"]["n_sims"] == 10_000
    for name in ("report.json", "report.md", "equity.csv", "trades.csv"):
        assert (out / name).stat().st_size > 0
