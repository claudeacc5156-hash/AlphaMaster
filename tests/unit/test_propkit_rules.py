"""Tests for propkit/rules.py: PropRules floors, breach test, best-day rule, FTMO presets. Research only.

Known answers (contract, CLAUDE.md C6): C0 = 100,000 and B_00:00 = 100,000: equity -3.01% breaches the
1-Step daily floor, -2.99% does not, -3.01% does not breach 2-Step (5%); B_00:00 = 108,000: the 1-Step
floor is 105,000; the trailing max floor after the highest 00:00 balance reaches 112,000 is 102,000 and
never moves down; the 2-Step static floor is 90,000; best day 60% of positive profit fails, 50% passes.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from propkit import rules as R
from propkit.rules import PropRules

C0 = 100_000.0


# ---------------------------------------------------------------------------------------
# presets

def test_ftmo_1step_preset_fields():
    r = R.ftmo_1step(C0)
    assert r.initial_capital == C0
    assert r.profit_target_pct == 0.10
    assert r.daily_loss_pct == 0.03 and r.daily_loss_base == "initial" and r.day_start_reference == "balance"
    assert r.max_loss_pct == 0.10 and r.max_loss_mode == "trailing_eod_balance"
    assert r.best_day_max_share == 0.5 and r.best_day_basis == "balance"
    assert r.min_trading_days == 0
    assert r.target_requires_flat is True and r.breach_inclusive is False
    assert r.target_balance == 110_000.0
    assert "24 Sep 2026" in r.notes


def test_ftmo_2step_preset_fields_and_phase_2_target():
    r = R.ftmo_2step(C0)
    assert r.profit_target_pct == 0.10 and r.target_balance == 110_000.0
    assert r.daily_loss_pct == 0.05 and r.daily_loss_base == "initial"
    assert r.max_loss_pct == 0.10 and r.max_loss_mode == "static"
    assert r.best_day_max_share is None
    assert r.min_trading_days == 4
    p2 = R.ftmo_2step(C0, target=0.05)
    assert p2.target_balance == 105_000.0
    assert "5%" in p2.name


def test_preset_by_cli_name_and_overrides():
    assert R.preset("ftmo-1step", 50_000) == R.ftmo_1step(50_000)
    assert R.preset("FTMO_2step") == R.ftmo_2step()
    p = R.preset("ftmo-2step", 200_000, profit_target_pct=0.05)
    assert p.initial_capital == 200_000 and p.profit_target_pct == 0.05 and p.min_trading_days == 4
    with pytest.raises(ValueError, match="unknown rules preset"):
        R.preset("ftmo-3step")
    with pytest.raises(ValueError, match="unknown rule field"):
        R.preset("ftmo-1step", foo=1)
    assert R.PRESET_NAMES == ("ftmo-1step", "ftmo-2step")


def test_custom_defaults_name_and_rejects_unknown_fields():
    c = R.custom(daily_loss_pct=0.04)
    assert c.name == "custom" and c.daily_loss_pct == 0.04
    with pytest.raises(ValueError, match="unknown rule field"):
        R.custom(daily_loss=0.04)


# ---------------------------------------------------------------------------------------
# known answers: daily floors

def test_known_answer_daily_floor_b00_100k():
    one, two = R.ftmo_1step(C0), R.ftmo_2step(C0)
    assert one.daily_floor(100_000.0) == 97_000.0
    assert two.daily_floor(100_000.0) == 95_000.0
    down_301 = 100_000.0 * (1 - 0.0301)      # 96,990
    down_299 = 100_000.0 * (1 - 0.0299)      # 97,010
    assert one.breached(down_301, one.daily_floor(100_000.0)) is True
    assert one.breached(down_299, one.daily_floor(100_000.0)) is False
    assert two.breached(down_301, two.daily_floor(100_000.0)) is False


def test_known_answer_daily_floor_b00_108k():
    one = R.ftmo_1step(C0)
    floor = one.daily_floor(108_000.0)
    assert floor == 105_000.0
    assert one.breached(104_999.99, floor) is True
    assert one.breached(105_000.01, floor) is False
    assert one.breached(105_000.00, floor) is False           # strictly below only
    inclusive = R.custom(**{**one.to_dict(), "breach_inclusive": True})
    assert inclusive.breached(105_000.00, floor) is True


def test_daily_floor_is_free_of_float_noise():
    # in float64 0.07 * 100,000 = 7000.000000000001 and 100,000 * 1.1 = 110000.00000000001;
    # floors and the target are rounded to 1e-8 USD
    assert 0.07 * 100_000 != 7000.0 and 100_000 * 1.1 != 110_000.0
    assert R.custom(daily_loss_pct=0.07).daily_floor(100_000.0) == 93_000.0
    assert R.ftmo_1step(C0).target_balance == 110_000.0
    one = R.ftmo_1step(C0)
    refs = np.array([100_000.0, 108_000.0, 99_123.45, 112_000.0])
    np.testing.assert_array_equal(one.daily_floor(refs), np.round(refs - 3000.0, 8))
    assert one.daily_floor(108_000.0) == 105_000.0


def test_daily_floor_day_start_base():
    r = R.custom(daily_loss_pct=0.05, daily_loss_base="day_start")
    assert r.daily_floor(100_000.0) == 95_000.0
    assert r.daily_floor(120_000.0) == 114_000.0
    initial = R.custom(daily_loss_pct=0.05, daily_loss_base="initial")
    assert initial.daily_floor(120_000.0) == 115_000.0


# ---------------------------------------------------------------------------------------
# known answers: max floors

def test_known_answer_trailing_max_floor():
    one = R.ftmo_1step(C0)
    assert one.max_floor() == 90_000.0
    assert one.max_floor(95_000.0) == 90_000.0                # never below C0 - 10% of C0
    assert one.max_floor(112_000.0) == 102_000.0
    # the running maximum of B_00:00 makes the floor monotone: it never moves down
    b00 = np.array([100_000, 104_000, 112_000, 109_500, 106_000, 104_000, 113_500, 101_000.0])
    floors = one.max_floor(np.maximum.accumulate(b00))
    np.testing.assert_array_equal(floors, [90_000, 94_000, 102_000, 102_000, 102_000, 102_000, 103_500, 103_500])
    assert (np.diff(floors) >= 0).all()


def test_known_answer_static_max_floor():
    two = R.ftmo_2step(C0)
    assert two.max_floor() == 90_000.0
    assert two.max_floor(112_000.0) == 90_000.0
    np.testing.assert_array_equal(two.max_floor(np.array([100_000.0, 130_000.0])), [90_000.0, 90_000.0])
    assert two.breached(89_999.99, two.max_floor()) is True
    assert two.breached(90_000.00, two.max_floor()) is False


def test_rules_switched_off_have_minus_infinite_floors():
    r = R.custom(daily_loss_pct=None, max_loss_pct=None, profit_target_pct=None)
    assert r.daily_floor(100_000.0) == -np.inf
    assert r.max_floor(100_000.0) == -np.inf
    assert r.breached(1.0, r.daily_floor(100_000.0)) is False
    assert r.target_balance is None and r.max_loss_usd is None


# ---------------------------------------------------------------------------------------
# best-day rule

def test_known_answer_best_day_60_percent_fails_50_percent_passes():
    one = R.ftmo_1step(C0)
    assert one.best_day_ok([6000.0, 2000.0, 2000.0]) is False              # 60%
    assert one.best_day_share([6000.0, 2000.0, 2000.0]) == pytest.approx(0.6)
    assert one.best_day_ok([5000.0, 5000.0]) is True                        # exactly 50%
    assert one.best_day_ok([6000.0, 2000.0, 2000.0, 2000.0]) is True        # 6000 / 12000 = 50%
    # negative days do not count in the total of positive days
    assert one.best_day_ok([5000.0, -3000.0, 3000.0, 2000.0]) is True       # 5000 / 10000
    assert one.best_day_ok([5000.0, -3000.0, 3000.0, 1999.0]) is False      # 5000 / 9999
    assert one.best_day_share([-100.0, -5.0]) is None
    assert one.best_day_ok([]) is True
    assert R.ftmo_2step().best_day_ok([9000.0, 1.0]) is True                # no best-day rule
    with pytest.raises(ValueError, match="finite"):
        one.best_day_ok([1.0, np.nan])


def test_best_day_ok_from_vectorised():
    one = R.ftmo_1step(C0)
    out = one.best_day_ok_from(np.array([5000.0, 6000.0, 0.0]), np.array([10_000.0, 10_000.0, 0.0]))
    np.testing.assert_array_equal(out, [True, False, True])


# ---------------------------------------------------------------------------------------
# validation and presentation

@pytest.mark.parametrize("field, value, message", [
    ("daily_loss_pct", 3, "looks like a percent"),
    ("daily_loss_pct", 0.0, "between 0"),
    ("max_loss_pct", -0.1, "between 0"),
    ("profit_target_pct", float("nan"), "fraction"),
    ("best_day_max_share", 1.5, "between 0"),
    ("daily_loss_base", "start", "daily_loss_base"),
    ("day_start_reference", "equity", "day_start_reference"),
    ("max_loss_mode", "trailing", "max_loss_mode"),
    ("best_day_basis", "closed", "best_day_basis"),
    ("min_trading_days", -1, "min_trading_days"),
    ("min_trading_days", 2.5, "min_trading_days"),
    ("initial_capital", 0, "initial_capital"),
    ("initial_capital", True, "initial_capital"),
    ("breach_inclusive", "yes", "breach_inclusive"),
    ("name", "", "name"),
])
def test_invalid_fields_raise_actionable_errors(field, value, message):
    with pytest.raises(ValueError, match=message):
        R.custom(**{field: value})


def test_best_day_share_of_one_is_allowed():
    assert R.custom(best_day_max_share=1.0).best_day_max_share == 1.0


def test_to_dict_from_dict_round_trip_and_json():
    for r in (R.ftmo_1step(C0), R.ftmo_2step(50_000, target=0.05), R.custom(daily_loss_pct=None)):
        d = r.to_dict()
        json.dumps(d)
        assert PropRules.from_dict(d) == r
    with pytest.raises(ValueError, match="unknown rule field"):
        PropRules.from_dict({**R.ftmo_1step().to_dict(), "bogus": 1})
    with pytest.raises(ValueError, match="dict"):
        PropRules.from_dict([1, 2])


def test_frozen():
    r = R.ftmo_1step()
    with pytest.raises(Exception):
        r.daily_loss_pct = 0.05


def test_describe_is_ascii_plain_english():
    bare = R.custom(daily_loss_pct=None, max_loss_pct=None, profit_target_pct=None,
                    day_start_reference="max_balance_equity")
    for r in (R.ftmo_1step(), R.ftmo_2step(), bare):
        text = "\n".join(r.describe())
        text.encode("ascii")
        assert r.name in text
    one = "\n".join(R.ftmo_1step().describe())
    assert "trailing" in one and "best day" in one and "110,000.00" in one
    assert "minimum trading days: 4" in "\n".join(R.ftmo_2step().describe())


def test_module_source_is_ascii():
    import pathlib
    src = pathlib.Path(R.__file__).read_bytes()
    src.decode("ascii")


def test_describe_wording_follows_breach_inclusive():
    """Finding 5: with breach_inclusive=True equity AT the floor breaches, so it must stay strictly above it."""
    loose = "\n".join(R.ftmo_1step().describe())
    assert "must stay at or above" in loose and "strictly above" not in loose
    strict = R.custom(name="strict", daily_loss_pct=0.05, max_loss_pct=0.10, breach_inclusive=True)
    text = "\n".join(strict.describe())
    assert text.count("must stay strictly above") == 2 and "at or above" not in text
    assert strict.breached(strict.max_floor(), strict.max_floor()) and not R.ftmo_1step().breached(90_000.0, 90_000.0)


def test_preset_overrides_change_the_name_unless_they_are_the_phase_2_target():
    """Findings 2 / 17: a preset with changed terms kept the firm's name."""
    assert R.preset("ftmo-1step", 50_000).name == R.ftmo_1step(50_000).name          # capital is not a term
    assert R.preset("ftmo-1step", initial_capital=50_000) == R.ftmo_1step(50_000)     # keyword still works
    p2 = R.preset("ftmo-2step", 200_000, profit_target_pct=0.05)
    assert p2 == R.ftmo_2step(200_000, target=0.05) and p2.name == "FTMO 2-Step (target 5%)"
    m = R.preset("ftmo-1step", daily_loss_pct=0.04)
    assert m.name == "FTMO 1-Step (modified: daily_loss_pct=0.04)"
    assert "NOT the firm's published terms" in m.notes and m.daily_loss_pct == 0.04
    m2 = R.preset("ftmo-2step", profit_target_pct=0.05, best_day_max_share=0.5)
    assert m2.name == "FTMO 2-Step (target 5%) (modified: best_day_max_share=0.5)"
    assert R.preset("ftmo-1step", daily_loss_pct=0.03).name == R.ftmo_1step().name      # same value: no change
    own = R.preset("ftmo-1step", name="mine", daily_loss_pct=0.04)
    assert own.name == "mine" and own.daily_loss_pct == 0.04
