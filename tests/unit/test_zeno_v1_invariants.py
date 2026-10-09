"""zeno_v1 whole-pipeline invariants on synthetic bars (NOT market data): the long/short mirror, causality
under truncation and under perturbation of everything after an instant (M15 and M1, news, both variants,
several cost cells), two targeted look-ahead cases (D4 trend, D11 spread), determinism, table schemas and a
timing guard on 256,000 M15 bars. Research only."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from propkit import adapters
from propkit import zeno_v1 as z
from tests.unit.zeno_v1_testkit import utc

Q = 1.0 / 64.0                      # price grid: every price, spread and level below is exact in binary
SPREAD = 0.25
M = 6000.0                          # mirror: bid' = M - ask, ask' = M - bid (prices stay in [2048, 4096))
CELL = z.ZenoCell("evaluation", 10.0, "S1", 1.0)


def _walk(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    step = np.round(rng.normal(0.0, 0.9, n) / Q) * Q
    c = 3000.0 + np.cumsum(step)
    o = np.r_[3000.0, c[:-1]]
    h = np.maximum(o, c) + np.round(np.abs(rng.normal(0, 0.5, n)) / Q) * Q
    lo = np.minimum(o, c) - np.round(np.abs(rng.normal(0, 0.5, n)) / Q) * Q
    t = utc("2024-01-01 00:00") + 900 * np.arange(n, dtype=np.int64)
    return pd.DataFrame({"time": t, "open": o, "high": h, "low": lo, "close": c})


def _frames(bid: pd.DataFrame):
    ask = bid.copy()
    ask[["open", "high", "low", "close"]] += SPREAD
    mb = pd.DataFrame({"time": bid["time"], "open": M - ask["open"], "high": M - ask["low"],
                       "low": M - ask["high"], "close": M - ask["close"]})
    ma = pd.DataFrame({"time": bid["time"], "open": M - bid["open"], "high": M - bid["low"],
                       "low": M - bid["high"], "close": M - bid["close"]})
    return z.bidask_frame(bid, ask), z.bidask_frame(mb, ma)


def test_long_short_mirror_is_exact():
    bid = _walk(12_000, seed=5)
    assert bid[["low"]].min().iloc[0] > 2048 + 8 and bid[["high"]].max().iloc[0] < 4096 - 8
    f, fm = _frames(bid)
    atr = np.round(z.atr14_m15(f) * 256) / 256                       # the same dyadic ATR on both sides
    h1, h1m = z.h1_from_m15(f), z.h1_from_m15(fm)
    ema = np.round(z.ema30_h1(h1["close"].to_numpy()) / Q) * Q
    ema_m = (M - SPREAD) - ema                                        # mirrors c > e and the slope exactly
    assert np.array_equal(h1m["close"].to_numpy(), (M - SPREAD) - h1["close"].to_numpy())
    prep = z.prepare(f, test_indicators={"atr14": atr, "ema30_h1": ema})
    prep_m = z.prepare(fm, test_indicators={"atr14": atr, "ema30_h1": ema_m})
    cfg = z.ZenoConfig(CELL)
    a, b = z.simulate(prep, cfg), z.simulate(prep_m, cfg)

    ta = a.decisions[a.decisions["event"] == "trigger"]
    tb = b.decisions[b.decisions["event"] == "trigger"]
    assert len(ta) > 40 and (ta["status"] == "entered").sum() > 10
    assert (ta["side"] == "long").sum() > 5 and (ta["side"] == "short").sum() > 5
    flip = {"long": "short", "short": "long"}
    ka = sorted(zip(ta["bar_index"], ta["side"].map(flip), ta["status"], ta["reasons"]))
    kb = sorted(zip(tb["bar_index"], tb["side"], tb["status"], tb["reasons"]))
    assert ka == kb
    pa = a.positions.sort_values("entry_time").reset_index(drop=True)
    pb = b.positions.sort_values("entry_time").reset_index(drop=True)
    assert pa["side"].map(flip).tolist() == pb["side"].tolist()
    for col in ("entry_time", "units_oz", "R_usd_per_oz", "outcome", "exit1_reason", "exit1_time", "exit2_time",
                "tp1_reached"):
        assert pa[col].tolist() == pb[col].tolist(), col
    assert np.allclose(pa["net_pnl_usd"], pb["net_pnl_usd"], atol=1e-6)
    assert a.meta["final_balance_usd"] == pytest.approx(b.meta["final_balance_usd"], abs=1e-5)


def test_decisions_are_causal_under_truncation():
    f = z.synthetic_m15_bidask(start=utc("2022-03-01 00:00"), n_bars=9_000, seed=21, price=1900.0,
                               vol_per_hour=0.004, spread=0.25)
    cfg = z.ZenoConfig(CELL)
    full = z.simulate(z.prepare(f), cfg).decisions
    assert (full["status"] == "entered").sum() > 5
    cols = ["event", "side", "setup_id", "bar_index", "status", "reasons", "position_id", "entry_time",
            "entry_price", "stop_level", "atr_trigger", "atr_median"]
    for k in (2_000, 3_100, 3_101, 4_250, 5_555, 6_001, 7_333, 8_200, 8_998, 8_999):
        cut = z.simulate(z.prepare(f.iloc[:k].reset_index(drop=True)), cfg).decisions
        keep_full = full[(full["bar_index"] <= k - 2)][cols].reset_index(drop=True)
        keep_cut = cut[(cut["bar_index"] <= k - 2)][cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(keep_full, keep_cut, check_dtype=False)


# ---------------------------------------------------------------------------------------------------
# causality by perturbation (brief deliverable 5; CAUS-1)
#
# Truncating the data at bar k leaves bar k-1 whole, so a decision at bar k-2 that peeks at bar k-1's high,
# low or close looks causal; it also says nothing about exits, costs, news or M1. Here everything from the
# instant T = open of bar k on is replaced by wild values (except bar k's bid and ask OPEN, which a decision
# at bar k-1's close may use), on M1 and M15 bars that agree, with a different spread at a bar's open, high,
# low and close, a news calendar, both variants, both spread bases, three multipliers and the M1 second run.
# Everything known by T must not change.

# The 2020 calendar rows of NFP, CPI, PPI and FOMC (UTC; three unscheduled FOMC statements) and ten synthetic
# 09:00 UTC instants, so that Master closes also hit positions of the morning entry window.
PERTURB_NEWS = (
    ("NFP", "2020-02-07 13:30", "scheduled"), ("CPI", "2020-02-13 13:30", "scheduled"),
    ("PPI", "2020-02-19 13:30", "scheduled"), ("FOMC", "2020-03-03 15:00", "unscheduled"),
    ("NFP", "2020-03-06 13:30", "scheduled"), ("CPI", "2020-03-11 12:30", "scheduled"),
    ("PPI", "2020-03-12 12:30", "scheduled"), ("FOMC", "2020-03-15 21:00", "unscheduled"),
    ("FOMC", "2020-03-23 12:00", "unscheduled"), ("NFP", "2020-04-03 12:30", "scheduled"),
    ("PPI", "2020-04-09 12:30", "scheduled"), ("CPI", "2020-04-10 12:30", "scheduled"),
    ("FOMC", "2020-04-29 18:00", "scheduled"), ("NFP", "2020-05-08 12:30", "scheduled"),
    ("CPI", "2020-05-12 12:30", "scheduled")) + tuple(
    ("CPI", f"2020-{d} 09:00", "scheduled") for d in ("03-17", "03-19", "03-25", "03-31", "04-07", "04-15",
                                                      "04-21", "04-28", "05-05", "05-14"))
PERTURB_CELLS = (z.ZenoCell("evaluation", 10.0, "S1", 1.5), z.ZenoCell("master", 10.0, "S2", 2.0),
                 z.ZenoCell("evaluation", 5.0, "S2", 1.0), z.ZenoCell("master", 5.0, "S1", 1.0))
ENTRY_COLS = ["position_id", "side", "setup_id", "trigger_bar", "entry_bar", "entry_time", "entry_price",
              "spread_entry", "stop_level", "be_level", "tp1_level", "tp2_level", "R_usd_per_oz", "units_oz",
              "partial_lots", "balance_at_entry", "risk_budget_usd", "atr_trigger", "h_level", "l_level"]


def _m1_walk(n_min: int, seed: int, price: float = 1600.0, vol: float = 0.00035):
    """An M1 bid walk on metals hours from 2020-02-03 (weekend gaps, the daily break, the US DST switch)."""
    from propkit import bars as bars_mod
    rng = np.random.default_rng(seed)
    cand = utc("2020-02-03 00:00") + 60 * np.arange(int(n_min * 1.7), dtype=np.int64)
    t = cand[bars_mod.metals_market_open(cand)][:n_min]
    step = rng.normal(0, vol, t.size)
    o = price * np.exp(np.cumsum(np.r_[0.0, step[:-1]]))
    c = o * np.exp(step)
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, vol * 0.6, t.size)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, vol * 0.6, t.size)))
    spike = rng.random(t.size) < 0.004            # two-sided wicks: ambiguous M15 bars for the M1 run
    h = np.where(spike, h * np.exp(np.abs(rng.normal(0, 0.006, t.size))), h)
    lo = np.where(spike, lo * np.exp(-np.abs(rng.normal(0, 0.006, t.size))), lo)
    return t, [o, h, lo, c]


def _ask_of(o, h, lo, c, rng):
    """An ask with its own spread at the open, high, low and close (0.01 .. ~1.5 USD/oz)."""
    s = lambda: np.round(0.25 * np.exp(rng.normal(0, 0.5, o.size)), 2) + 0.01  # noqa: E731
    ao, ac = o + s(), c + s()
    return [ao, np.maximum.reduce([h + s(), ao, ac]), np.minimum.reduce([lo + s(), ao, ac]), ac]


def _frames_from_m1(t, cols):
    names = ("open", "high", "low", "close")
    bid1 = pd.DataFrame({"time": t, **dict(zip(names, cols[:4]))})
    ask1 = pd.DataFrame({"time": t, **dict(zip(names, cols[4:]))})
    m1 = z.bidask_frame(bid1, ask1, bar_seconds=60, source="M1")
    key = t // 900
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], t.size]

    def agg(o, h, lo, c):
        return pd.DataFrame({"time": key[starts] * 900, "open": o[starts], "high": np.maximum.reduceat(h, starts),
                             "low": np.minimum.reduceat(lo, starts), "close": c[ends - 1]})
    return z.bidask_frame(agg(*cols[:4]), agg(*cols[4:])), m1, starts


def _perturbed(t, cols, starts, k: int, seed: int):
    """Copies of the M1 columns where every minute from M15 bar k's open on is a wild walk (10x the
    volatility, new spreads), except the bid and ask OPEN of bar k's first minute."""
    q = int(starts[k])
    rng = np.random.default_rng(seed)
    n = t.size - q
    out = [x.copy() for x in cols]
    step = rng.normal(0, 0.0035, n)
    step[:15] += (1 if seed % 2 else -1) * 0.002            # bar k itself jumps 3% up or down
    o = out[0][q] * np.exp(np.cumsum(np.r_[0.0, step[:-1]]))
    c = o * np.exp(step)
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.004, n)))
    new = [o, h, lo, c] + _ask_of(o, h, lo, c, rng)
    keep_ao = out[4][q]
    for x, y in zip(out, new):
        x[q:] = y
    out[4][q] = keep_ao
    out[5][q] = max(out[5][q], out[4][q], out[7][q])
    out[6][q] = min(out[6][q], out[4][q], out[7][q])
    return out


def _same(a: pd.DataFrame, b: pd.DataFrame, what: str) -> list[str]:
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        return []
    except AssertionError as e:
        return [f"{what}: {' '.join(str(e).split())[:300]}"]


def _known_by(clean, pert, k: int, T: int) -> list[str]:
    """What must not change when everything from T = open of bar k on is replaced: decisions of bars up to
    k - 1 (all columns), positions closed by T (all columns), the entry fields of positions entered at or
    before bar k, legs exited by T."""
    dc, dp, pc, pp, lc, lp = clean.decisions, pert.decisions, clean.positions, pert.positions, clean.legs, pert.legs
    closed = pc["final_exit_stamp"] <= T
    return (_same(dc[dc["bar_index"] <= k - 1], dp[dp["bar_index"] <= k - 1], "decisions")
            + _same(pc[closed], pp[pp["position_id"].isin(pc.loc[closed, "position_id"])], "closed positions")
            + _same(pc.loc[pc["entry_bar"] <= k, ENTRY_COLS], pp.loc[pp["entry_bar"] <= k, ENTRY_COLS], "entries")
            + _same(lc[lc["exit_time"] <= T].drop(columns="trade_id"),
                    lp[lp["exit_time"] <= T].drop(columns="trade_id"), "legs"))


def test_everything_known_by_an_instant_survives_replacing_all_later_prices():
    # About 100,000 M1 bars (2020-02-03 .. mid-May, the US DST switch included) and 60 cuts: every entry
    # bar of an in-session trigger, every exit bar and the bar after it are candidates, plus 10 random bars.
    # This test fails on each of seven deliberately leaky versions of the engine (trend, ATR, the D18 median,
    # the entry spread, an exit reading one bar ahead); the truncation test above catches only one of them.
    n_cuts = 60
    t, bid = _m1_walk(100_000, seed=3)
    cols = bid + _ask_of(*bid, np.random.default_rng(1003))
    m15, m1, starts = _frames_from_m1(t, cols)
    news = z.news_calendar([utc(x[1]) for x in PERTURB_NEWS], [x[0] for x in PERTURB_NEWS],
                           [x[2] for x in PERTURB_NEWS])
    prep = z.prepare(m15, news)
    clean = {c: z.simulate(prep, z.ZenoConfig(c)) for c in PERTURB_CELLS}
    clean_m1 = {c: z.simulate(prep, z.ZenoConfig(c), m1=m1) for c in PERTURB_CELLS[:2]}
    # the data exercise what the comparison is meant to see
    first = clean[PERTURB_CELLS[0]]
    assert len(first.positions) >= 15 and first.positions["ambiguous_bar"].any()
    assert (clean[PERTURB_CELLS[1]].legs["exit_reason"] == "signal").any()                 # a Master close
    assert any(r.meta["m1"]["bars_resolved"] > 0 for r in clean_m1.values())
    assert (m15["ask_close"] - m15["bid_close"] != m15["ask_open"] - m15["bid_open"]).mean() > 0.9
    # cut points: every entry bar (trigger + 1), every exit bar and the bar after it, some random bars
    tt = m15["time"].to_numpy()
    ks: set[int] = set()
    for res in list(clean.values()) + list(clean_m1.values()):
        d = res.decisions
        tr = d[(d["event"] == "trigger") & ~d["status"].isin(["warmup", "outside_session"])]
        ks |= set((tr["bar_index"] + 1).tolist())
        for col in ("exit1_time", "exit2_time"):
            x = res.positions[col].to_numpy()
            j = np.searchsorted(tt, x[x > 0], side="right") - 1
            ks |= set(j.tolist()) | set((j + 1).tolist())
    rng = np.random.default_rng(0)
    ks = sorted(k for k in ks if 50 <= k < len(m15) - 1)
    ks = sorted(rng.choice(ks, min(n_cuts - 10, len(ks)), replace=False).tolist()
                + rng.integers(2_000, len(m15) - 1, 10).tolist())
    failures = []
    for n, k in enumerate(ks):
        T = int(tt[k])
        m15p, m1p, _ = _frames_from_m1(t, _perturbed(t, cols, starts, k, seed=10_000 + n))
        assert m15p.iloc[:k].equals(m15.iloc[:k])
        assert (m15p[["bid_open", "ask_open"]].iloc[k] == m15[["bid_open", "ask_open"]].iloc[k]).all()
        assert m15p["bid_close"].iloc[k] != m15["bid_close"].iloc[k]
        prep_p = z.prepare(m15p, news)
        for c in PERTURB_CELLS:
            failures += [(k, c.label, e) for e in _known_by(clean[c], z.simulate(prep_p, z.ZenoConfig(c)), k, T)]
        for c in PERTURB_CELLS[:2]:
            failures += [(k, c.label + " M1", e)
                         for e in _known_by(clean_m1[c], z.simulate(prep_p, z.ZenoConfig(c), m1=m1p), k, T)]
    assert not failures, f"{len(failures)} look-ahead failure(s) over {len(ks)} cuts, first: {failures[:3]}"


def test_the_trend_is_read_from_the_h1_bar_closed_by_the_trigger_close():
    # D3/D4 (CAUS-1): the trigger closes 13:45 UTC and reads the H1 bar that closed at 13:00. The H1 bar that
    # closes at 14:00 - 15 min later, at the entry bar's close - turns the trend (EMA far above price); it
    # must not reach the decision.
    from tests.unit.zeno_v1_testkit import ATR, EVAL_10_X1, Scenario
    sc = Scenario("2024-03-05 11:45")
    ti = sc.setup(1)
    sc.flat(8, 2006.0)
    f = sc.frame()
    h1 = z.h1_from_m15(f)
    ema = 1000.0 + 0.01 * np.arange(len(h1), dtype=np.float64)
    turn = int(np.flatnonzero(h1["close_time"].to_numpy() == utc("2024-03-05 14:00"))[0])
    ema[turn:] = 5000.0 + 0.01 * np.arange(len(h1) - turn)
    prep = z.prepare(f, test_indicators={"atr14": np.full(len(f), ATR), "ema30_h1": ema})
    assert f["time"].iloc[ti] + 900 == utc("2024-03-05 13:45")
    assert bool(prep.trend_long[ti]) and not bool(prep.trend_long[ti + 1])
    res = z.simulate(prep, z.ZenoConfig(EVAL_10_X1))
    row = res.decisions[res.decisions["event"] == "trigger"].iloc[0]
    assert (row["bar_index"], row["status"], bool(row["trend_ok"])) == (ti, "entered", True)


def test_rule_10_reads_the_entry_bar_open_spread_not_a_later_one():
    # D11 (CAUS-1): the spread is the entry bar's ask open - bid open, 0.20 here; the same bar closes with a
    # 3.00 spread (above 10% of R 4.70), which is not known at the decision.
    from tests.unit.zeno_v1_testkit import Scenario
    sc = Scenario("2024-03-05 12:00")
    ti = sc.setup(1)
    sc.add(2006.0, 2007.0, 2005.5, 2006.5, ask=(2006.2, 2010.0, 2006.2, 2009.5))
    sc.flat(4, 2006.5)
    prep, res = sc.run()
    row = res.decisions[res.decisions["event"] == "trigger"].iloc[0]
    assert (row["bar_index"], row["status"]) == (ti, "entered")
    assert row["spread_entry"] == pytest.approx(0.20) and row["entry_price"] == pytest.approx(2006.20)


def test_results_are_deterministic_and_tables_have_their_columns():
    f = z.synthetic_m15_bidask(start=utc("2022-03-01 00:00"), n_bars=6_000, seed=4, price=1900.0,
                               vol_per_hour=0.004, spread=0.25)
    prep = z.prepare(f)
    r1 = z.simulate(prep, z.ZenoConfig(z.ZenoCell()))
    r2 = z.simulate(z.prepare(f), z.ZenoConfig(z.ZenoCell()))
    pd.testing.assert_frame_equal(r1.positions, r2.positions)
    pd.testing.assert_frame_equal(r1.decisions, r2.decisions)
    assert tuple(r1.legs.columns) == z.LEG_COLUMNS
    assert tuple(r1.legs.columns[:len(adapters.TRADES_COLUMNS)]) == adapters.TRADES_COLUMNS
    assert tuple(r1.positions.columns) == z.POSITION_COLUMNS
    assert tuple(r1.decisions.columns) == z.DECISION_COLUMNS
    assert set(r1.positions["outcome"]) <= set(z.OUTCOMES)
    assert set(r1.legs["exit_reason"]) <= set(adapters.EXIT_REASONS)
    assert set(r1.legs["leg"]) <= set(z.LEG_NAMES)
    assert set(r1.decisions["event"]) <= set(z.SETUP_EVENTS)
    assert r1.meta["n_positions"] == len(r1.positions) and r1.meta["n_legs"] == len(r1.legs)
    assert r1.meta["m1"] == z.M1_NOT_RUN and r1.meta["indicators_injected"] == []
    # every leg's net P&L sums to its position's and every position is a closed one-shot entry
    by_pos = r1.legs.groupby("position_id")["pnl_usd"].sum()
    assert np.allclose(by_pos.to_numpy(), r1.positions.set_index("position_id")["net_pnl_usd"].loc[by_pos.index])
    tr = r1.decisions[r1.decisions["event"] == "trigger"]
    assert (tr["status"] == "entered").sum() == len(r1.positions)
    # never two positions at once (D21): each entry is after the previous position's last exit
    p = r1.positions.sort_values("entry_time")
    last_exit = np.maximum(p["exit1_time"], p["exit2_time"]).to_numpy()
    assert (p["entry_time"].to_numpy()[1:] >= last_exit[:-1]).all()


def test_empty_result_tables():
    f = z.synthetic_m15_bidask(n_bars=100)
    res = z.simulate(z.prepare(f))
    assert len(res.positions) == 0 and len(res.legs) == 0
    assert tuple(res.legs.columns) == z.LEG_COLUMNS
    eq, tr = z.cell_equity(z.prepare(f), res)
    assert len(tr) == 0 and eq["balance"].iloc[-1] == 100_000.0


def test_timing_guard_on_256k_bars():
    f = z.synthetic_m15_bidask(n_bars=256_000, seed=1)
    t0 = time.perf_counter()
    prep = z.prepare(f)
    t1 = time.perf_counter()
    res = z.simulate(prep, z.ZenoConfig(z.ZenoCell()))
    z.cell_equity(prep, res)
    t2 = time.perf_counter()
    assert t1 - t0 < 60, f"prepare took {t1 - t0:.1f} s"
    assert t2 - t1 < 120, f"simulate + equity took {t2 - t1:.1f} s"
    assert len(prep.triggers()) > 1000
