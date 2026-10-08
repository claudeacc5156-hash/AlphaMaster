"""zeno_v1 setup state machines (rules 3-4, D5-D10, [SI-11], [SI-12], [SI-13], [SI-43], [SI-46]) on hand-made
BID bars. ATR14 is a constant 2.00 unless a test says otherwise. Bar numbers in the comments are relative to
the canonical setup b0..b7 of tests/unit/zeno_v1_testkit.py (H = 2010 at b3, L = 1995 at b0, leg 15,
50% level 2002.5, 78.6% void level 2010 - 11.79 = 1998.21, armed at b5, trigger at b7), which follows 20
flat bars at 2000. Research only."""
from __future__ import annotations

import numpy as np
import pytest

from propkit import zeno_v1 as z
from tests.unit.zeno_v1_testkit import long_setup, mirror

PREFIX = 20
B = 2000.0


def run(bars, prefix: int = PREFIX, atr: float = 2.0, atr_at: dict | None = None, pre_rows=None):
    """Events of setup_machines on `prefix` flat bars at 2000 (or pre_rows) + bars; 'rel' = bar - prefix."""
    rows = list(pre_rows) if pre_rows is not None else [(B, B, B, B)] * prefix
    rows += list(bars)
    o, h, lo, c = (np.array(x, dtype=float) for x in zip(*rows))
    a = np.full(len(rows), float(atr))
    for k, v in (atr_at or {}).items():
        a[prefix + k] = v
    events = z.setup_machines(h, lo, c, a)
    for e in events:
        e["rel"] = e["bar"] - prefix
    return events


def seq(events, side: int = 1):
    return [(e["event"], e["rel"]) for e in events if e["side"] == side]


def canon(**changes):
    """The canonical bars with some replaced: canon(b5=(o, h, l, c))."""
    bars = long_setup(B)
    for k, v in changes.items():
        bars[int(k[1:])] = v
    return bars


def test_canonical_long_events_and_levels():
    ev = run(long_setup())
    assert seq(ev) == [("armed", 5), ("trigger", 7)]
    assert seq(ev, -1) == []
    arm, trig = ev
    for e in (arm, trig):
        assert e["setup_id"] == "L1" and e["side"] == 1
        assert e["ext_bar"] - PREFIX == 3 and e["arm_bar"] - PREFIX == 5
        assert (e["h_level"], e["l_level"], e["leg"]) == (2010.0, 1995.0, 15.0)
        assert e["retrace_level"] == 2002.5 and e["void_level"] == pytest.approx(1998.21)
        assert e["atr_arm"] == 2.0
    assert (arm["pl_bar"] - PREFIX, arm["pl_level"], arm["trigger_level"], arm["bars_since_pullback"]) == \
        (5, 2002.0, 2005.5, 0)
    assert (trig["pl_bar"] - PREFIX, trig["pl_level"], trig["trigger_level"], trig["bars_since_pullback"]) == \
        (5, 2002.0, 2005.5, 2)


def test_short_is_the_exact_mirror():
    # mirrored around 2000: L' = 1990 (b3), H' = 2005 (b0), pullback-high bar b5 (high 1998, low 1994.5)
    ev = run(mirror(long_setup()))
    assert seq(ev, -1) == [("armed", 5), ("trigger", 7)] and seq(ev) == []
    t = ev[-1]
    assert t["setup_id"] == "S1"
    assert (t["h_level"], t["l_level"], t["leg"]) == (2005.0, 1990.0, 15.0)
    assert (t["pl_level"], t["trigger_level"]) == (1998.0, 1994.5)
    assert t["retrace_level"] == 1997.5 and t["void_level"] == pytest.approx(2001.79)


def test_h_tie_goes_to_the_latest_bar():
    # b2 high raised to 2010 = b3 high: H is b3 (the later bar), so the pullback low is searched after b3
    ev = run(canon(b2=(B + 1, B + 10, B + 1, B + 6)))
    assert seq(ev) == [("armed", 5), ("trigger", 7)]
    assert ev[0]["ext_bar"] - PREFIX == 3


@pytest.mark.parametrize("dip_at,expected_l", [(3, 1994.0), (2, 1995.0)])
def test_l_is_the_lowest_low_of_the_20_bars_before_h(dip_at, expected_l):
    # H = b3 at index 23. The 20 bars before it are indices 3..22. A 1994 wick at index 3 is inside the window
    # (L = 1994, leg 16, 50% = 2002.0, still touched by b5's low 2002); at index 2 it is outside (L = 1995).
    pre = [(B, B, B, B)] * PREFIX
    pre[dip_at] = (B, B, 1994.0, B)
    ev = run(long_setup(), pre_rows=pre)
    assert seq(ev)[0] == ("armed", 5)
    assert ev[0]["l_level"] == expected_l and ev[0]["leg"] == 2010.0 - expected_l


def test_no_setup_without_20_bars_before_h():
    # [SI-43]: with 16 flat bars H (b3) is bar 19 and only 19 bars precede it: nothing arms. With 17 it does.
    assert run(long_setup(), prefix=16) == []
    assert seq(run(long_setup(), prefix=17)) == [("armed", 5), ("trigger", 7)]


def test_wick_touch_at_exactly_50_percent_arms():
    # b5 low 2002.5 = H - 0.5 x 15: armed. b6's low 2002.5 ties it, so the pullback-low bar moves to b6
    # ([SI-12]) and the trigger level becomes b6's high 2004; b7 closes 2006 > 2004 one bar later.
    ev = run(canon(b5=(B + 5, B + 5.5, B + 2.5, B + 3)))
    assert seq(ev) == [("armed", 5), ("trigger", 7)]
    assert ev[0]["pl_level"] == 2002.5
    assert (ev[1]["pl_bar"] - PREFIX, ev[1]["trigger_level"], ev[1]["bars_since_pullback"]) == (6, 2004.0, 1)


def test_no_touch_one_cent_above_50_percent():
    # also [SI-35]: b0-b2 have lows far below 2002.5 but come before H (b3); only bars after H count
    ev = run(canon(b5=(B + 5, B + 5.5, B + 2.51, B + 3), b6=(B + 3, B + 4, B + 2.6, B + 4)))
    assert seq(ev) == []


def test_void_on_the_close_strictly_beyond_78_6_percent():
    void = 2010.0 - 0.786 * 15.0                           # 1998.21
    at = run(canon(b6=(B + 3, B + 4, 1998.0, void)))       # a close AT the level does not void
    assert seq(at) == [("armed", 5), ("trigger", 7)]
    assert at[1]["pl_level"] == 1998.0 and at[1]["trigger_level"] == 2004.0   # the new low moved the bar
    below = run(canon(b6=(B + 3, B + 4, 1998.0, 1998.209)))   # a tenth of a cent below: void
    assert seq(below) == [("armed", 5), ("voided", 6)]
    # [SI-34] tie rule: a close within 1e-9 USD/oz of the computed level is AT it (float noise is not a price)
    ulp = run(canon(b6=(B + 3, B + 4, 1998.0, float(np.nextafter(void, -np.inf)))))
    assert seq(ulp) == [("armed", 5), ("trigger", 7)]


def test_an_exact_50_percent_touch_counts_despite_float_noise():
    # [SI-34] tie rule. L = 1995.28 (b0), H = 2010 (b3): leg 14.72, so the 50% level is 2002.64 exactly, which
    # is 2002.6399999999999 in binary floating point. b5's bid low is exactly 2002.64: a touch (D6), armed at
    # b5. (b6's low 2002.50 then moves the pullback-low bar to b6, trigger level 2004; b7 closes 2006: trigger.)
    ev = run(canon(b0=(B, B, 1995.28, B - 4), b5=(B + 5, B + 5.5, 2002.64, B + 3)))
    assert seq(ev) == [("armed", 5), ("trigger", 7)]
    assert ev[0]["retrace_level"] == pytest.approx(2002.64, abs=1e-9) and ev[0]["l_level"] == 1995.28


def test_a_close_exactly_at_the_void_level_does_not_void_despite_float_noise():
    # [SI-34] tie rule. L = 1995.13 (b0), H = 2010.13 (b3): leg 15.00, so the void level is 2010.13 - 11.79 =
    # 1998.34 exactly, which is 1998.3400000000001 in floating point. b6 closes exactly at 1998.34: not BELOW
    # the level (D6), so no void; its low 1998.00 is a new pullback low (trigger level 2004) and b7 closes
    # 2006 > 2004: trigger.
    ev = run(canon(b0=(B, B, 1995.13, B - 4), b3=(B + 6, 2010.13, B + 5, B + 9), b6=(B + 3, B + 4, 1998.0, 1998.34)))
    assert seq(ev) == [("armed", 5), ("trigger", 7)]
    assert ev[0]["void_level"] == pytest.approx(1998.34, abs=1e-9)


def test_void_before_arming_prevents_the_setup():
    # b4 closes at 1998 < 1998.21 before any arming: this H can never arm (its closes since H include b4)
    ev = run(canon(b4=(B + 9, B + 9, 1997.9, 1998.0)))
    assert seq(ev) == []


def test_void_wins_over_a_new_extreme_in_the_same_bar():
    ev = run(canon(b6=(B + 3, 2010.5, 1997.0, 1998.0)))
    assert seq(ev) == [("armed", 5), ("voided", 6)]


def test_a_new_high_cancels_an_equal_high_does_not():
    eq = run(canon(b6=(B + 3, 2010.0, B + 2.5, B + 4)))
    assert seq(eq) == [("armed", 5), ("trigger", 7)]
    above = run(canon(b6=(B + 3, 2010.01, B + 2.5, B + 4)))
    assert seq(above) == [("armed", 5), ("cancelled_new_extreme", 6)]


def test_trigger_needs_a_close_strictly_above_the_pullback_bar_high():
    ev = run(canon(b7=(B + 4, B + 7, B + 3.5, B + 5.5)))        # close 2005.5 = b5's high: no trigger
    assert ("trigger", 7) not in seq(ev)


def test_trigger_one_bar_after_the_low():
    ev = run(canon(b6=(B + 3, B + 6, B + 2.5, B + 5.6)))        # 2005.6 > 2005.5
    assert seq(ev) == [("armed", 5), ("trigger", 6)] and ev[1]["bars_since_pullback"] == 1


def _quiet(n, low=B + 2.5, high=B + 4.0, close=B + 3.5):
    return [(close, high, low, close)] * n


def test_a_new_low_restarts_the_count_and_bar_8_may_trigger():
    # armed at b5 (low 2002); b6..b12 = 7 quiet bars (no new low, closes 2003.5 <= 2005.5); b13 makes a new
    # low 2001.9 (high 2003): the count restarts at b13 and the trigger level becomes 2003; b14..b20 = 7
    # quiet bars closing 2002.9; b21 = 8th bar after the low closes 2003.1 > 2003: trigger.
    bars = long_setup()[:6] + _quiet(7) + [(B + 3, B + 3, 2001.9, B + 2.5)] + \
        _quiet(7, low=2002.0, high=2002.95, close=2002.9) + [(2002.9, 2003.2, 2002.0, 2003.1)]
    ev = run(bars)
    assert seq(ev) == [("armed", 5), ("trigger", 21)]
    t = ev[-1]
    assert (t["pl_bar"] - PREFIX, t["pl_level"], t["trigger_level"], t["bars_since_pullback"]) == \
        (13, 2001.9, 2003.0, 8)


def test_expiry_at_the_close_of_bar_8_and_bar_9_cannot_trigger_or_revive():
    # armed at b5; b6..b13 (8 bars) never close above 2005.5: expired at b13's close. b14 closes 2008 (would
    # have been a trigger) and b15 makes a new low 2001: the used-up setup never comes back [SI-46].
    bars = long_setup()[:6] + _quiet(8) + [(B + 4, B + 8.5, B + 3, B + 8), (B + 8, B + 8, 2001.0, B + 2)]
    ev = run(bars)
    assert seq(ev) == [("armed", 5), ("expired", 13)]
    assert ev[1]["bars_since_pullback"] == 8


def test_leg_uses_atr_at_the_arming_bar():
    # 1.5 x ATR <= 15 is needed at the arming bar. ATR 10.01 everywhere: 15.015 > 15, never arms.
    assert seq(run(long_setup(), atr=10.01)) == []
    # the same, but ATR 10.0 at b5 (exactly 1.5 x 10 = 15 = the leg): arms at b5, ATR elsewhere is irrelevant
    ev = run(long_setup(), atr=10.01, atr_at={5: 10.0})
    assert seq(ev) == [("armed", 5), ("trigger", 7)] and ev[0]["atr_arm"] == 10.0
    # a huge ATR at the H bar b3 does not matter either
    assert seq(run(long_setup(), atr_at={3: 100.0})) == [("armed", 5), ("trigger", 7)]
    # NaN ATR at b5: arms one bar later (b6) with the pullback-low bar still b5
    late = run(long_setup(), atr_at={5: float("nan")})
    assert seq(late) == [("armed", 6), ("trigger", 7)]
    assert late[0]["pl_bar"] - PREFIX == 5 and late[0]["bars_since_pullback"] == 1


def test_the_arming_bar_may_trigger():
    # [SI-13]: no ATR at b5, so b6 arms (pullback-low bar b5, trigger level 2005.5) and b6 closes 2005.6
    ev = run(canon(b6=(B + 3, B + 6, B + 2.5, B + 5.6)), atr_at={5: float("nan")})
    assert seq(ev) == [("armed", 6), ("trigger", 6)]


LATE_ARM = long_setup()[:6] + [
    (B + 3, B + 6.5, B + 2.5, B + 6.0),     # b6: close 2006 > 2005.5 (b5's high): the FIRST close above it, but
                                            #     ATR 11 here (leg 15 < 1.5 x 11 = 16.5): no setup is armed yet
    (B + 6, B + 6.2, B + 4.5, B + 5.0),     # b7: ATR 2 again, D5-D7 hold: ARMED (pullback-low bar still b5)
    (B + 5, B + 7.0, B + 4.6, B + 6.4)]     # b8: close 2006.4 > 2005.5, a LATER close above b5's high


def test_a_close_above_the_pullback_bar_before_arming_is_the_first_close_and_is_not_chased():
    # [SI-61], rule 4 / D9 / D10: the trigger is the first close above the pullback-low bar's high; b6 was that
    # close (before the setup armed at b7), so b8's close is a later close and never triggers. b9..b13 stay
    # inside b5's range (no new low): the setup expires at the close of b13 = b5 + 8 ([SI-47]).
    bars = LATE_ARM + [(B + 6, B + 7.0, B + 5.0, B + 6.5)] * 5
    ev = run(bars, atr_at={5: 11.0, 6: 11.0})
    assert seq(ev) == [("armed", 7), ("first_close_before_arming", 7), ("expired", 13)]
    note = ev[1]
    assert (note["pl_bar"] - PREFIX, note["trigger_level"], note["bars_since_pullback"]) == (5, 2005.5, 2)


def test_after_a_passed_first_close_a_new_pullback_low_restarts_the_count():
    # the same late arming at b7; b9 makes a new low 2001.5 (close 2002 > the void level 1998.21): the
    # pullback-low bar moves to b9 and its count restarts (D9); b10 closes 2003.5 > 2003 (b9's high), the first
    # close above it: trigger, 1 bar after the new pullback-low bar.
    bars = LATE_ARM + [(B + 6, B + 3.0, 2001.5, B + 2.0), (B + 2, B + 3.6, B + 1.8, B + 3.5)]
    ev = run(bars, atr_at={5: 11.0, 6: 11.0})
    assert seq(ev) == [("armed", 7), ("first_close_before_arming", 7), ("trigger", 10)]
    t = ev[-1]
    assert (t["pl_bar"] - PREFIX, t["pl_level"], t["trigger_level"], t["bars_since_pullback"]) == \
        (9, 2001.5, 2003.0, 1)


def test_h_and_l_stay_frozen_after_h_leaves_the_window():
    # armed at b5; 22 bars each make a new low 0.1 lower (the count restarts every bar, closes stay above
    # 1998.21), so H (b3) leaves the 20-bar window at b23. A bar with high 2009 (above every high still in
    # the window, below the frozen H) cancels nothing; the trigger still reports H = 2010, L = 1995.
    bars = long_setup()[:6]
    for k in range(1, 23):
        low = 2002.0 - 0.1 * k
        bars.append((low + 0.5, low + 1.5, low, low + 0.5))
    bars.append((2000.3, 2009.0, 2000.0, 2000.5))           # b28: no new low, close 2000.5 <= 2001.3
    bars.append((2000.5, 2003.0, 1999.9, 2002.0))           # b29: close 2002 > 2001.3 (b27's high)
    ev = run(bars)
    # L1 triggers at b29. At the same close a NEW setup L2 arms: with b3 gone, b28's 2009 is the 20-bar high
    # (L = 1999.8, leg 9.2, 50% = 2004.4 touched by b29's low 1999.9, no close below 2001.77): a new H bar
    # and a new pullback-low bar, so [SI-46] does not stop it.
    # (The short machine also arms on the falling lows at b28; it is not part of this test.)
    assert seq(ev) == [("armed", 5), ("trigger", 29), ("armed", 29)]
    longs = [e for e in ev if e["side"] == 1]
    assert (longs[2]["setup_id"], longs[2]["h_level"], longs[2]["ext_bar"] - PREFIX) == ("L2", 2009.0, 28)
    t = longs[1]
    assert (t["h_level"], t["l_level"], t["ext_bar"] - PREFIX) == (2010.0, 1995.0, 3)
    assert (t["pl_bar"] - PREFIX, t["pl_level"], t["trigger_level"]) == (27, pytest.approx(1999.8),
                                                                       pytest.approx(2001.3))
    assert t["void_level"] == pytest.approx(1998.21)


def test_one_shot_no_rearm_on_the_same_h_but_a_new_h_arms():
    # after the trigger at b7: b8 closes higher again and b9 dips to a new low 2001.5 -> nothing (same H b3,
    # [SI-46]). b10 makes a new high 2011 (L = 1995, leg 16, 50% = 2003) and b11 dips to 2003: setup L2.
    bars = long_setup() + [(B + 6, B + 8, B + 5, B + 8), (B + 8, B + 8, 2001.5, B + 3),
                           (B + 3, 2011.0, B + 3, B + 10), (B + 10, B + 10, 2003.0, B + 4)]
    ev = run(bars)
    assert seq(ev) == [("armed", 5), ("trigger", 7), ("armed", 11)]
    assert ev[-1]["setup_id"] == "L2" and ev[-1]["h_level"] == 2011.0 and ev[-1]["ext_bar"] - PREFIX == 10


def test_events_are_causal_under_truncation():
    # every event at bar <= k is the same when the bars stop at k (nothing looks ahead)
    rng = np.random.default_rng(7)
    n = 3000
    c = 2000 + np.cumsum(rng.normal(0, 1.0, n))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.uniform(0, 1.5, n)
    lo = np.minimum(o, c) - rng.uniform(0, 1.5, n)
    a = np.full(n, 1.2)
    full = z.setup_machines(h, lo, c, a)
    assert sum(e["event"] == "trigger" for e in full) > 20
    for k in (500, 1234, 2001, 2999):
        cut = z.setup_machines(h[:k], lo[:k], c[:k], a[:k])
        assert cut == [e for e in full if e["bar"] < k]


def test_input_lengths_must_match():
    with pytest.raises(ValueError):
        z.setup_machines([1, 2], [1, 2], [1, 2], [1.0])


# ---------------------------------------------------------------------------------------------------
# the machines inside prepare (time, gaps, the ask side)

def test_windows_count_bars_of_the_data_across_a_gap():
    # [SI-10]: a 1994 wick, then 2 flat bars, then a 2-day gap, then 12 flat bars and the canonical setup.
    # Counting bars of the data, the wick is the 18th bar before H (b3): inside the 20-bar L window, so
    # L = 1994 (leg 16, 50% = 2002.0, touched by b5's low 2002). A clock-time window would have missed it.
    from tests.unit.zeno_v1_testkit import Scenario
    sc = Scenario("2024-03-06 12:00")
    sc.add(B, B, 1994.0, B)
    sc.flat(2, B)
    sc.skip_to("2024-03-08 12:00")
    sc.flat(12, B)
    sc.setup(1)
    sc.flat(2, 2006.0)
    ev = [e for e in sc.prepare().events if e["side"] == 1]
    assert [e["event"] for e in ev] == ["armed", "trigger"]
    assert ev[0]["l_level"] == 1994.0 and ev[0]["leg"] == 16.0 and ev[0]["retrace_level"] == 2002.0


def test_short_signal_uses_bid_bars_only():
    # [SI-15]: an ask spike (ask high = bid high + 5) on b4 and b6 changes nothing in the short machine
    from tests.unit.zeno_v1_testkit import Scenario

    def events(spike: bool):
        sc = Scenario("2024-03-05 12:00")
        for k, (o, h, lo, c) in enumerate(mirror(long_setup())):
            ask = (o + 0.2, h + 5.0, lo + 0.2, c + 0.2) if spike and k in (4, 6) else None
            sc.add(o, h, lo, c, ask=ask)
        sc.flat(2, 1994.0)
        return [(e["event"], e["side"], e["bar"], e["pl_level"], e["trigger_level"]) for e in sc.prepare().events]

    assert events(False) == events(True)
    assert [x[0] for x in events(True) if x[1] == -1] == ["armed", "trigger"]
