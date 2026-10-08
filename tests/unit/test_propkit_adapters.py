"""Tests for propkit/adapters.py: TRADES / POSITIONS checks and CSV files, the AlphaMaster adapter (one-bar
held shift, the miner's log PnL checked against scripts/research/score_formula.py) and positions -> trades.
Synthetic data only. Research only."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from propkit import adapters
from propkit.adapters import (EXIT_REASONS, TRADES_COLUMNS, alphamaster_log_pnl, positions_from_alphamaster,
                              read_positions_csv, read_trades_csv, trades_from_positions, validate_positions,
                              validate_trades, write_equity_csv, write_positions_csv, write_trades_csv)
from propkit.bars import LockedPathError, synthetic_bars
from propkit.costs import CostModel
from propkit.equity import equity_from_positions, equity_from_trades

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "research" / "score_formula.py"
C0 = 100_000.0
NO_SWAP = CostModel(swap_enabled=False)


def T(text: str) -> int:
    return int(np.datetime64(text.replace(" ", "T"), "s").astype(np.int64))


@pytest.fixture(scope="module")
def sf():
    """scripts/research/score_formula.py loaded from its path (as tests/unit/test_research_score_formula.py)."""
    spec = importlib.util.spec_from_file_location("research_score_formula_for_propkit", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def continuous_bars(n: int, seed: int, start: str = "2024-01-01 00:00") -> pd.DataFrame:
    """24-hour H1 bars whose close equals the next open (no gaps), random-walk opens, no spread column."""
    rng = np.random.default_rng(seed)
    o = 2000.0 * np.exp(np.cumsum(rng.normal(0, 0.003, n + 1)))
    c = o[1:]
    o = o[:-1]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.001, n))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.001, n))
    return pd.DataFrame({"time": T(start) + 3600 * np.arange(n, dtype=np.int64), "open": o, "high": h,
                         "low": lo, "close": c})


# ---------------------------------------------------------------------------------------
# AlphaMaster: the miner's log PnL

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("cost_rate", [0.0003, 0.0, 0.0011])
def test_alphamaster_log_pnl_equals_score_formula_pnl_series(sf, seed, cost_rate):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 2000))
    p = rng.choice([-1.0, 0.0, 1.0], n)
    open_ = 2000.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    # target_ret as data_pipeline builds it: log(open[t+2] / open[t+1]) for t <= T-3, the last two 0
    target = np.zeros(n)
    target[: n - 2] = np.log(open_[2:] / open_[1:-1])
    want = sf.pnl_series(p, target, cost_rate, 1.0)
    got = alphamaster_log_pnl(p, open_, cost_rate)
    assert got.shape == (n,)
    assert np.max(np.abs(got - want)) <= 1e-12


def test_alphamaster_log_pnl_hand_example(sf):
    p = [1.0, 0.0, -1.0, -1.0, 0.0]
    o = [100.0, 101.0, 102.0, 104.0, 103.0]
    r = 0.001
    want = [1 * math.log(102 / 101) - 1 * r,      # enter long: |1 - 0| x r
            0.0 - 1 * r,                          # exit: |0 - 1| x r
            -1 * math.log(103 / 104) - 1 * r,     # enter short
            -1 * 0.0 - 0 * r,                     # t = T-2: no target return
            0.0 - 1 * r]                          # t = T-1: exit cost still charged
    assert np.allclose(alphamaster_log_pnl(p, o, r), want, atol=1e-15)
    assert np.allclose(sf.pnl_series(p, [math.log(102 / 101), math.log(104 / 102), math.log(103 / 104), 0, 0],
                                     r, 1.0), want, atol=1e-15)


@pytest.mark.parametrize("p, o", [([1.0, 0.0], [1.0]), ([np.nan, 0.0], [1.0, 1.0]), ([0.0, 0.0], [1.0, 0.0])])
def test_alphamaster_log_pnl_refuses_bad_input(p, o):
    with pytest.raises(ValueError):
        alphamaster_log_pnl(p, o, 0.0003)


# ---------------------------------------------------------------------------------------
# AlphaMaster: the one-bar held shift

def test_positions_from_alphamaster_shifts_by_one_bar():
    times = T("2024-01-01 00:00") + 3600 * np.arange(5)
    p = [0.5, -1.0, 0.0, 1.0, 0.25]
    pos = positions_from_alphamaster(times, p, keep_raw=True)
    assert list(pos.columns) == ["time", "position", "p_raw"]
    assert pos["time"].tolist() == times.tolist()
    assert pos["position"].tolist() == [0.0, 0.5, -1.0, 0.0, 1.0]     # held[t+1] = p[t], held[0] = 0
    assert pos["p_raw"].tolist() == p
    assert "p_raw" not in positions_from_alphamaster(times, p).columns
    for bad_p in ([0.5, 1.5, 0, 0, 0], [0.5, np.nan, 0, 0, 0], [0.5, 0.5]):
        with pytest.raises(ValueError):
            positions_from_alphamaster(times, bad_p)


def test_one_bar_held_shift_known_answer():
    bars = continuous_bars(6, seed=0)
    o = bars["open"].to_numpy()
    pos = positions_from_alphamaster(bars["time"], [1.0, 0.0, 0.0, -1.0, -1.0, 0.0])
    _, tr = equity_from_positions(bars, pos, C0, CostModel(flat_rate_per_side=0.0, swap_enabled=False),
                                  size_mode="units", size=1.0)
    # p[0] = 1 decided at bar 0's close -> long held during bar 1: in at open[1], out at open[2];
    # p[3] = p[4] = -1 -> short held during bars 4 and 5: in at open[4], out at the last bar's close
    t = bars["time"].to_numpy()
    assert tr[["side", "entry_time", "exit_time", "exit_reason"]].values.tolist() == [
        [1, t[1], t[2], "signal"], [-1, t[4], t[5] + 3600, "end_of_data"]]
    assert tr["entry_price"].tolist() == [o[1], o[4]] and tr["exit_price"].iloc[0] == o[2]


@pytest.mark.parametrize("seed", [0, 1])
def test_one_bar_held_shift_matches_the_miner_bar_by_bar(sf, seed):
    """Units mode, flat rate = AlphaMaster's cost_rate, no swap, bars with close[k] = open[k+1]. For every
    t <= T-3, the bar t+1 of propkit carries exactly the miner's PnL[t] in USD form:
      price move:  equity_close[t+1] - equity_close[t] + commission[t+1] = S x p[t] x (open[t+2] - open[t+1])
                   = S x open[t+1] x p[t] x (exp(target_ret[t]) - 1)   (miner's gross term p[t] x target_ret[t])
      cost:        commission[t+1] = S x open[t+1] x |p[t] - p[t-1]| x cost_rate   (the miner's cost term)."""
    n, size, rate = 500, 7.0, 0.0003
    bars = continuous_bars(n, seed)
    rng = np.random.default_rng(seed + 10)
    p = rng.choice([-1.0, 0.0, 1.0], n)
    o = bars["open"].to_numpy()
    cm = CostModel(flat_rate_per_side=rate, swap_enabled=False)
    eq, tr = equity_from_positions(bars, positions_from_alphamaster(bars["time"], p), C0, cm, "units", size)
    close = eq["equity_close"].to_numpy()
    comm = eq["commission_usd"].to_numpy()
    k = np.arange(n - 2)                                      # t = 0 .. T-3
    move = close[k + 1] - close[k] + comm[k + 1]
    assert np.allclose(move, size * p[k] * (o[k + 2] - o[k + 1]), rtol=0, atol=1e-8)
    target = np.zeros(n)
    target[:-2] = np.log(o[2:] / o[1:-1])
    gross_miner = p * target                                  # = sf.pnl_series(p, target, 0, 1)
    assert np.allclose(gross_miner, sf.pnl_series(p, target, 0.0, 1.0), atol=1e-15)
    assert np.allclose(move, size * o[k + 1] * p[k] * np.expm1(target[k]), rtol=0, atol=1e-8)
    cost_miner = alphamaster_log_pnl(p, o, 0.0) - alphamaster_log_pnl(p, o, rate)     # |p[t] - p[t-1]| x rate
    assert np.allclose(comm[k + 1], size * o[k + 1] * cost_miner[k], rtol=0, atol=1e-9)
    assert comm[0] == 0.0 and close[0] == C0                  # held[0] = 0: nothing happens in the first bar
    # without the shift (p held during bar t itself) the price moves would NOT match
    wrong, _ = equity_from_positions(bars, pd.DataFrame({"time": bars["time"], "position": p}), C0, cm,
                                     "units", size)
    wmove = np.diff(wrong["equity_close"].to_numpy())[k] + wrong["commission_usd"].to_numpy()[k + 1]
    assert not np.allclose(wmove, size * p[k] * (o[k + 2] - o[k + 1]), atol=1e-6)


def test_leverage_one_tracks_the_miner_log_pnl_to_first_order(sf):
    """Leverage size 1 re-sizes only on changes while the miner rebalances in log terms every bar, so the
    two differ at second order only: over 2,000 bars of 0.3 % moves the total log growth agrees within 0.5 %
    of the miner's total absolute PnL (about 0.1 % in practice)."""
    n, rate = 2000, 0.0003
    bars = continuous_bars(n, seed=4)
    rng = np.random.default_rng(4)
    p = pd.Series(np.where(rng.random(n) < 0.1, rng.choice([-1.0, 0.0, 1.0], n), np.nan)).ffill().fillna(0.0)
    p = p.to_numpy()
    o = bars["open"].to_numpy()
    eq, _ = equity_from_positions(bars, positions_from_alphamaster(bars["time"], p), C0,
                                  CostModel(flat_rate_per_side=rate, swap_enabled=False), "leverage", 1.0)
    miner = alphamaster_log_pnl(p, o, rate)[: n - 2]          # through bar T-1's open
    got = math.log(eq["equity_close"].iloc[n - 2] / C0)       # equity at open[T-1] = close[T-2]
    assert abs(got - miner.sum()) <= 0.005 * np.abs(miner).sum()
    assert np.sign(got) == np.sign(miner.sum())


# ---------------------------------------------------------------------------------------
# positions -> trades

def test_fifo_partial_reductions_across_lots():
    bars = continuous_bars(6, seed=1)
    t, o = bars["time"].to_numpy(), bars["open"].to_numpy()
    pos = pd.DataFrame({"time": t, "position": [0.2, 0.5, 1.0, 0.3, 0.0, 0.0]})
    tr = trades_from_positions(bars, pos, CostModel(fixed_spread=0.0, spread_source="fixed"), "units", 10.0)
    # lots: 2 oz at bar 0, 3 oz at bar 1, 5 oz at bar 2; bar 3 cuts 7 oz FIFO: 2 + 3 + 2 of the 5;
    # bar 4 closes the remaining 3 oz of the bar-2 lot
    assert [(r.units, r.entry_time, r.exit_time) for r in tr.itertuples()] == pytest.approx([
        (2.0, t[0], t[3]), (3.0, t[1], t[3]), (2.0, t[2], t[3]), (3.0, t[2], t[4])])
    assert tr["entry_price"].tolist() == pytest.approx([o[0], o[1], o[2], o[2]])
    assert tr["exit_price"].tolist() == pytest.approx([o[3], o[3], o[3], o[4]])
    assert tr["trade_id"].tolist() == [1, 2, 3, 4] and set(tr["exit_reason"]) == {"signal"}
    assert tr["stop_price"].isna().all() and tr["pnl_usd"].isna().all()


def test_positions_on_part_of_the_bars_and_mismatches():
    bars = continuous_bars(10, seed=2)
    t, o = bars["time"].to_numpy(), bars["open"].to_numpy()
    pos = pd.DataFrame({"time": t[3:6], "position": [1.0, 1.0, -0.5]})
    tr = trades_from_positions(bars, pos, NO_SWAP.multiplied(0.0), "units", 2.0)
    # flat before bar 3; long 2 oz bars 3-4; short 1 oz in bar 5; flat after: closed at bar 6's open
    assert tr[["side", "units", "entry_time", "exit_time", "exit_reason"]].values.tolist() == [
        [1, 2.0, t[3], t[5], "signal"], [-1, 1.0, t[5], t[6], "end_of_data"]]
    assert tr["exit_price"].tolist() == pytest.approx([o[5], o[6]])
    with pytest.raises(ValueError, match="does not match the bar times"):
        trades_from_positions(bars, pd.DataFrame({"time": t[[3, 5]], "position": [1.0, 0.0]}), NO_SWAP)
    with pytest.raises(ValueError, match="does not match the bar times"):
        trades_from_positions(bars, pd.DataFrame({"time": t[3:6] + 60, "position": [1.0, 0.0, 0.0]}), NO_SWAP)
    with pytest.raises(ValueError, match="pass C0"):
        trades_from_positions(bars, pos, NO_SWAP, "leverage", 1.0)
    with pytest.raises(ValueError, match="size_mode"):
        trades_from_positions(bars, pos, NO_SWAP, "lots", 1.0)
    with pytest.raises(ValueError, match="size"):
        trades_from_positions(bars, pos, NO_SWAP, "units", 0.0)
    flat = trades_from_positions(bars, pd.DataFrame({"time": t, "position": np.zeros(10)}), NO_SWAP)
    assert len(flat) == 0 and tuple(flat.columns) == TRADES_COLUMNS


def test_leverage_refuses_when_equity_is_gone():
    bars = continuous_bars(30, seed=3)
    bars.loc[10:, ["open", "high", "low", "close"]] *= 0.01             # a 99 % gap down while long 50x
    p = np.zeros(30)
    p[1:12] = 1.0                                                        # held during bars 1..11
    p[12] = -1.0                                                         # re-sizing at bar 12 needs equity
    with pytest.raises(ValueError, match="leverage sizing is impossible"):
        trades_from_positions(bars, pd.DataFrame({"time": bars["time"], "position": p}), NO_SWAP,
                              "leverage", 50.0, C0=C0)


def test_validate_positions():
    t = T("2024-01-01 00:00") + 3600 * np.arange(4)
    ok = validate_positions(pd.DataFrame({"time": t, "position": [0, 1, -1, 0.5], "p_raw": [1, -1, 0.5, 0],
                                          "time_utc": ["a"] * 4}))
    assert list(ok.columns) == ["time", "position", "p_raw"] and ok["position"].dtype == np.float64
    for df, msg in ((pd.DataFrame({"time": t, "position": [0, 1.2, 0, 0]}), "outside"),
                    (pd.DataFrame({"time": t[::-1], "position": [0, 0, 0, 0]}), "strictly increasing"),
                    (pd.DataFrame({"time": t, "position": [0, np.nan, 0, 0]}), "missing"),
                    (pd.DataFrame({"time": t}), "missing column"),
                    (pd.DataFrame({"time": t, "position": [0] * 4, "size": [1] * 4}), "unknown column")):
        with pytest.raises(ValueError, match=msg):
            validate_positions(df)
    assert "size" in validate_positions(pd.DataFrame({"time": t, "position": [0] * 4, "size": [1] * 4}),
                                        allow_extra=True).columns


# ---------------------------------------------------------------------------------------
# TRADES checks

def _trade(**kw) -> pd.DataFrame:
    base = {"side": [1], "units": [10.0], "entry_time": [T("2024-01-02 10:00")], "entry_price": [2000.3],
            "exit_time": [T("2024-01-02 12:00")], "exit_price": [2005.0]}
    base.update(kw)
    return pd.DataFrame(base)


def test_validate_trades_defaults_and_text_sides():
    out = validate_trades(_trade(side=["Short"], exit_price=[1995.0]))
    assert tuple(out.columns) == TRADES_COLUMNS
    assert out["side"].tolist() == [-1] and out["trade_id"].tolist() == [1]
    assert out["exit_reason"].tolist() == ["unknown"] and out["stop_price"].isna().all()
    assert out["entry_time"].dtype == np.int64 and out["units"].dtype == np.float64
    two = validate_trades(pd.concat([_trade(side=["long"]), _trade(side=["SELL"])], ignore_index=True))
    assert two["side"].tolist() == [1, -1] and two["trade_id"].tolist() == [1, 2]
    iso = validate_trades(_trade(entry_time=["2024-01-02T10:00:00Z"], exit_time=["2024-01-02T12:00:00+00:00"]))
    assert iso["entry_time"].iloc[0] == T("2024-01-02 10:00") and iso["exit_time"].iloc[0] == T("2024-01-02 12:00")
    assert len(validate_trades(None)) == 0 and len(validate_trades(pd.DataFrame())) == 0
    assert set(EXIT_REASONS) == {"stop", "target", "trail", "time", "signal", "end_of_data", "unknown"}


@pytest.mark.parametrize("kw, msg", [
    ({"side": [0]}, "side"),
    ({"side": ["up"]}, "side"),
    ({"units": [0.0]}, "units"),
    ({"units": [np.nan]}, "units"),
    ({"exit_time": [T("2024-01-02 09:00")]}, "exits before it enters"),
    ({"entry_price": [-1.0]}, "entry_price"),
    ({"exit_price": [np.inf]}, "exit_price"),
    ({"stop_price": [2001.0]}, "wrong side"),
    ({"side": [-1], "stop_price": [1999.0]}, "wrong side"),
    ({"risk_usd": [-5.0]}, "risk_usd"),
    ({"exit_reason": ["panic"]}, "exit_reason"),
    ({"trade_id": [1.5]}, "trade_id"),
    ({"notes": ["x"]}, "unknown column"),
    ({"entry_time": [T("2024-01-02 10:00") * 1000 * 1000 * 1000 + 1]}, "nanoseconds"),
])
def test_validate_trades_refuses(kw, msg):
    with pytest.raises(ValueError, match=msg):
        validate_trades(_trade(**kw))


def test_validate_trades_duplicate_ids_and_missing_columns():
    with pytest.raises(ValueError, match="repeats a trade_id"):
        validate_trades(pd.concat([_trade(trade_id=[4]), _trade(trade_id=[4])], ignore_index=True))
    with pytest.raises(ValueError, match="missing column"):
        validate_trades(_trade().drop(columns=["exit_price"]))
    kept = validate_trades(_trade(notes=["x"]), allow_extra=True)
    assert kept["notes"].tolist() == ["x"]


# ---------------------------------------------------------------------------------------
# CSV files

def test_trades_csv_round_trip(tmp_path):
    bars = synthetic_bars(T("2024-03-04 00:00"), 300, seed=4)
    cm = CostModel(commission_per_lot_round_trip=7.0)
    t, o = bars["time"].to_numpy(), bars["open"].to_numpy()
    sp = cm.bar_spreads(bars)
    trades = pd.DataFrame({"side": [1, -1], "units": [12.5, 3.0], "entry_time": t[[10, 50]],
                           "entry_price": [cm.buy_fill(o[10], sp[10]), cm.sell_fill(o[50], sp[50])],
                           "exit_time": t[[40, 80]],
                           "exit_price": [cm.sell_fill(o[40], sp[40]), cm.buy_fill(o[80], sp[80])],
                           "exit_reason": ["target", "time"], "stop_price": [o[10] - 5.0, np.nan]})
    eq, tr = equity_from_trades(bars, trades, C0, cm)
    path = write_trades_csv(tr, tmp_path / "trades.csv")
    text = path.read_text(encoding="ascii")
    assert text.splitlines()[0].split(",") == list(TRADES_COLUMNS) + ["entry_time_utc", "exit_time_utc"]
    back = read_trades_csv(path)
    pd.testing.assert_frame_equal(back, tr, check_exact=True)          # floats round-trip exactly
    eq_path = write_equity_csv(eq, tmp_path / "equity.csv")
    eq_back = pd.read_csv(eq_path, float_precision="round_trip")
    assert list(eq_back.columns)[-2:] == ["time_utc", "prop_day"]
    assert np.array_equal(eq_back["balance"].to_numpy(), eq["balance"].to_numpy())
    empty = write_trades_csv(adapters.empty_trades(), tmp_path / "none.csv")
    assert len(read_trades_csv(empty)) == 0


def test_positions_csv_round_trip(tmp_path):
    t = T("2024-01-01 00:00") + 3600 * np.arange(5)
    pos = positions_from_alphamaster(t, [0.1, -0.2, 0.3, 0.0, 1.0], keep_raw=True)
    path = write_positions_csv(pos, tmp_path / "positions.csv")
    back = read_positions_csv(path)
    pd.testing.assert_frame_equal(back, pos, check_exact=True)
    (tmp_path / "extra.csv").write_text("time,position,leverage\n0,0.5,2\n", encoding="ascii")
    with pytest.raises(ValueError, match="unknown column"):
        read_positions_csv(tmp_path / "extra.csv")
    (tmp_path / "upper.csv").write_text("Time , POSITION\n3600,0.5\n7200,-0.5\n", encoding="ascii")
    assert read_positions_csv(tmp_path / "upper.csv")["position"].tolist() == [0.5, -0.5]


@pytest.mark.parametrize("name", ["locked_holdout/trades.csv", "LOCKED_HOLDOUT\\x.csv", "trades.csv.locked"])
def test_locked_paths_refused(tmp_path, name):
    target = tmp_path / name
    with pytest.raises(LockedPathError):
        read_trades_csv(target)
    with pytest.raises(LockedPathError):
        read_positions_csv(target)
    with pytest.raises(LockedPathError):
        write_trades_csv(adapters.empty_trades(), target)
    with pytest.raises(LockedPathError):
        write_positions_csv(pd.DataFrame({"time": [0], "position": [0.0]}), target)


def test_csv_path_errors(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        read_trades_csv(tmp_path / "nope.csv")
    with pytest.raises(ValueError, match="must end in .csv"):
        write_trades_csv(adapters.empty_trades(), tmp_path / "trades.txt")
    with pytest.raises(ValueError, match="does not exist"):
        write_trades_csv(adapters.empty_trades(), tmp_path / "no_such_dir" / "t.csv")


def test_trades_summary():
    bars = synthetic_bars(T("2024-03-04 00:00"), 100, seed=6)
    t, o = bars["time"].to_numpy(), bars["open"].to_numpy()
    tr = pd.DataFrame({"side": [1, -1], "units": [1.0, 2.0], "entry_time": t[[1, 2]], "entry_price": o[[1, 2]] + 0.34,
                       "exit_time": t[[5, 6]], "exit_price": [o[5], o[6] + 0.34]})
    _, out = equity_from_trades(bars, tr, C0, CostModel(spread_source="fixed", swap_enabled=False))
    s = adapters.trades_summary(out)
    assert (s["n_trades"], s["n_long"], s["n_short"]) == (2, 1, 1)
    assert s["pnl_usd"] == pytest.approx(out["pnl_usd"].sum()) and s["commission_usd"] == 0.0


def test_alphamaster_log_pnl_refuses_a_negative_cost_rate():
    """Finding 10: a negative cost_rate would turn every position change into a gain."""
    with pytest.raises(ValueError, match="cost_rate"):
        alphamaster_log_pnl([0.0, 1.0, 1.0, 0.0], [2000.0, 2001.0, 2002.0, 2003.0], -0.0003)
    assert alphamaster_log_pnl([0.0, 1.0, 1.0], [2000.0, 2001.0, 2002.0], 0.0).shape == (3,)


def test_misaligned_positions_name_the_first_bad_row():
    """Finding 23: positions offset from the bar times AND running past the last bar were reported at the
    last row; the first row that does not match is the useful one."""
    bars = continuous_bars(20, seed=3)
    t = bars["time"].to_numpy()
    shifted = pd.DataFrame({"time": t[10:] + 1800, "position": 0.0})        # 30 minutes off, 10 rows
    with pytest.raises(ValueError, match=r"row 0 time"):
        trades_from_positions(bars, shifted, NO_SWAP)
    past_end = pd.DataFrame({"time": np.r_[t[15:], t[-1] + 3600 * np.arange(1, 4)], "position": 0.0})
    with pytest.raises(ValueError, match=r"row 5 time"):                       # the first row after the bars
        trades_from_positions(bars, past_end, NO_SWAP)


def test_text_columns_with_any_character_are_written_as_ascii_escapes(tmp_path):
    """Finding 16: a non-ASCII text column (a comment) made write_trades_csv fail half-way."""
    bars = continuous_bars(30, seed=4)
    cm = CostModel(swap_enabled=False)
    t, o = bars["time"].to_numpy(), bars["open"].to_numpy()
    sp = cm.bar_spreads(bars)
    tr = pd.DataFrame({"side": [1], "units": [10.0], "entry_time": t[[3]], "entry_price": [cm.buy_fill(o[3], sp[3])],
                       "exit_time": t[[8]], "exit_price": [cm.sell_fill(o[8], sp[8])],
                       "comment": ["黄金 breakout – test"]})
    _, out = equity_from_trades(bars, tr, C0, cm)
    path = write_trades_csv(out, tmp_path / "trades.csv")
    text = path.read_bytes().decode("ascii")
    assert "\\u9ec4\\u91d1 breakout \\u2013 test" in text


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_positions_still_open_at_the_data_end_close_at_the_last_close(side):
    """Review finding: an open position at the end of the data is closed at the LAST bar's close (time = its
    open + bar_seconds), priced with that bar's spread and the cost model, and flagged end_of_data."""
    bars = continuous_bars(6, seed=4).assign(spread=[0.30, 0.31, 0.32, 0.33, 0.34, 0.45])
    bars.loc[5, "close"] = bars.loc[5, "open"] + 3.0                   # the last bar moves: open != close
    bars.loc[5, "high"] = max(bars.loc[5, "high"], bars.loc[5, "close"])
    cm = CostModel(markup_per_side=0.10, slippage_per_side=0.05, swap_enabled=False)
    t, o, c = bars["time"].to_numpy(), bars["open"].to_numpy(), bars["close"].to_numpy()
    pos = pd.DataFrame({"time": t, "position": [0.0, 0.0, side, side, side, side]})
    tr = trades_from_positions(bars, pos, cm, "units", 5.0)
    assert len(tr) == 1 and tr["exit_reason"].iloc[0] == "end_of_data"
    assert tr["exit_time"].iloc[0] == t[5] + 3600
    want = cm.sell_fill(c[5], 0.45) if side > 0 else cm.buy_fill(c[5], 0.45)
    assert tr["exit_price"].iloc[0] == pytest.approx(want, abs=1e-12)
    assert tr["exit_price"].iloc[0] != pytest.approx(cm.sell_fill(o[5], 0.45) if side > 0 else cm.buy_fill(o[5], 0.45))
    eq, full = equity_from_trades(bars, tr, C0, cm)
    assert full["pnl_usd"].iloc[0] == pytest.approx(side * 5.0 * (want - tr["entry_price"].iloc[0]))
    assert eq["units_open"].iloc[-1] == 0.0 and eq["balance"].iloc[-1] == eq["equity_close"].iloc[-1]
