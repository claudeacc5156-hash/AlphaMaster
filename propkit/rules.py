"""propkit/rules.py - prop-firm challenge rules (PropRules) and the FTMO presets. Research only.

Money is USD. C0 = initial_capital. Percentages are FRACTIONS (0.03 = 3%). "B_00:00" is the closed
balance at 00:00 CE(S)T, the start of a prop day (see propkit.calendar.prop_day); "E_00:00" is the
equity (balance + floating PnL) at that instant.

Floors (the account is breached when equity falls below one; equity_worst, the lowest equity inside a
bar, is what the evaluator compares):
  * daily floor, daily_loss_base "initial" (FTMO, CLAUDE.md C6):  L_daily = ref - daily_loss_pct x C0;
                 daily_loss_base "day_start":                     L_daily = ref x (1 - daily_loss_pct);
    ref = B_00:00 (day_start_reference "balance", FTMO per C6) or max(B_00:00, E_00:00)
    (day_start_reference "max_balance_equity", used by some other firms).
  * max floor, max_loss_mode "trailing_eod_balance" (FTMO 1-Step per C6):
                 L_max = max(C0, highest B_00:00 so far) - max_loss_pct x C0  (never moves down);
    max_loss_mode "static" (FTMO 2-Step): L_max = C0 - max_loss_pct x C0.
    The trailing floor always trails the day-start BALANCE, whatever day_start_reference says.
  * breach: equity < floor (breach_inclusive False, FTMO wording "must not fall below") or equity <=
    floor (breach_inclusive True). Floors are rounded to 1e-8 USD to remove binary floating-point noise
    from pct x C0 (in float64 0.07 x 100,000 is 7000.000000000001 and 100,000 x 1.1 is
    110000.00000000001).
  * a rule set to None is switched off (its floor is -infinity).

Passing (see propkit.evaluator for the scan): the closed balance reaches C0 x (1 + profit_target_pct),
with no position open at that bar close when target_requires_flat is True (default; FTMO counts closed
results), the best-day rule holds and at least min_trading_days prop days with a trade have passed.
Best-day rule (FTMO 1-Step): the best day's profit <= best_day_max_share x the sum of the profits of all
positive days. Day profit = change of the closed balance over the prop day (best_day_basis "balance",
default) or change of equity from 00:00 to the day's last bar close ("equity"). A tolerance of 1e-6 USD
is allowed, so a best day of exactly 50% passes.

FTMO figures are as stated in CLAUDE.md C6 (FTMO as of 24 Sep 2026). Prop-firm rules change: recheck
them on the firm's site before every challenge.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

DAILY_LOSS_BASES = ("initial", "day_start")
DAY_START_REFERENCES = ("balance", "max_balance_equity")
MAX_LOSS_MODES = ("trailing_eod_balance", "static")
BEST_DAY_BASES = ("balance", "equity")
PRESET_NAMES = ("ftmo-1step", "ftmo-2step")
FLOOR_DECIMALS = 8          # floors are rounded to 1e-8 USD
BEST_DAY_TOL_USD = 1e-6     # best day may exceed share x total by this much (float noise only)
FTMO_AS_OF = "24 Sep 2026 (CLAUDE.md C6); recheck before every challenge"


def _is_number(value) -> bool:
    return (not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating)))


def _fraction(value, what: str, allow_one: bool = False) -> float | None:
    if value is None:
        return None
    if not _is_number(value) or not math.isfinite(float(value)):
        raise ValueError(f"{what} must be a fraction such as 0.05 (= 5%) or None, got {value!r}")
    v = float(value)
    upper_ok = v <= 1.0 if allow_one else v < 1.0
    if not (v > 0.0 and upper_ok):
        hint = " It looks like a percent: write 0.05 for 5%." if v >= 1.0 else ""
        raise ValueError(f"{what} must be between 0 and {'1' if allow_one else 'below 1'} "
                         f"(a fraction, 0.05 = 5%), got {v!r}.{hint}")
    return v


def _choice(value, what: str, options: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"{what} must be one of {', '.join(options)}; got {value!r}")
    return value


@dataclass(frozen=True)
class PropRules:
    """The rules of one prop-firm challenge phase (all money in USD, all pct fields are fractions).

    Fields:
      name                  label used in reports;
      initial_capital       C0, USD (> 0);
      profit_target_pct     target as a fraction of C0 (0.10 = +10%), or None (no target: a funded
                            account, or a pure breach study);
      daily_loss_pct        daily loss limit as a fraction (0.03 = 3%), or None (no daily rule);
      daily_loss_base       "initial" (limit = pct x C0, FTMO) or "day_start" (pct x day-start ref);
      day_start_reference   "balance" (B_00:00, FTMO) or "max_balance_equity" (max(B_00:00, E_00:00));
      max_loss_pct          maximum loss as a fraction of C0, or None (no max rule);
      max_loss_mode         "trailing_eod_balance" (floor trails the highest B_00:00, FTMO 1-Step) or
                            "static" (C0 - pct x C0, FTMO 2-Step);
      best_day_max_share    best day profit <= share x total profit of positive days (0.5 FTMO 1-Step),
                            or None (no best-day rule);
      best_day_basis        "balance" (day profit = closed-balance change, default) or "equity";
      min_trading_days      prop days with at least one trade entry needed to pass (FTMO 2-Step: 4);
      target_requires_flat  True: the target counts only at a bar close with no position open;
                            False: the closed balance alone counts, positions may be open;
      breach_inclusive      False: breach when equity < floor; True: when equity <= floor;
      notes                 free text (source and date of the rules).
    """

    name: str = "custom"
    initial_capital: float = 100_000.0
    profit_target_pct: float | None = 0.10
    daily_loss_pct: float | None = 0.05
    daily_loss_base: str = "initial"
    day_start_reference: str = "balance"
    max_loss_pct: float | None = 0.10
    max_loss_mode: str = "static"
    best_day_max_share: float | None = None
    best_day_basis: str = "balance"
    min_trading_days: int = 0
    target_requires_flat: bool = True
    breach_inclusive: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty text label")
        c0 = self.initial_capital
        if not _is_number(c0) or not math.isfinite(float(c0)) or float(c0) <= 0:
            raise ValueError(f"initial_capital must be a positive amount in USD, got {c0!r}")
        object.__setattr__(self, "initial_capital", float(c0))
        object.__setattr__(self, "profit_target_pct", _fraction(self.profit_target_pct, "profit_target_pct"))
        object.__setattr__(self, "daily_loss_pct", _fraction(self.daily_loss_pct, "daily_loss_pct"))
        object.__setattr__(self, "max_loss_pct", _fraction(self.max_loss_pct, "max_loss_pct"))
        object.__setattr__(self, "best_day_max_share",
                           _fraction(self.best_day_max_share, "best_day_max_share", allow_one=True))
        _choice(self.daily_loss_base, "daily_loss_base", DAILY_LOSS_BASES)
        _choice(self.day_start_reference, "day_start_reference", DAY_START_REFERENCES)
        _choice(self.max_loss_mode, "max_loss_mode", MAX_LOSS_MODES)
        _choice(self.best_day_basis, "best_day_basis", BEST_DAY_BASES)
        mtd = self.min_trading_days
        if isinstance(mtd, (bool, np.bool_)) or not isinstance(mtd, (int, np.integer)) or int(mtd) < 0:
            raise ValueError(f"min_trading_days must be a whole number >= 0, got {mtd!r}")
        object.__setattr__(self, "min_trading_days", int(mtd))
        for flag in ("target_requires_flat", "breach_inclusive"):
            if not isinstance(getattr(self, flag), (bool, np.bool_)):
                raise ValueError(f"{flag} must be True or False, got {getattr(self, flag)!r}")
            object.__setattr__(self, flag, bool(getattr(self, flag)))
        if not isinstance(self.notes, str):
            raise ValueError("notes must be text")

    # ---- derived amounts -------------------------------------------------------------

    @property
    def target_balance(self) -> float | None:
        """Closed balance that meets the profit target, USD: C0 x (1 + profit_target_pct); None if no target."""
        if self.profit_target_pct is None:
            return None
        return round(self.initial_capital * (1.0 + self.profit_target_pct), FLOOR_DECIMALS)

    @property
    def max_loss_usd(self) -> float | None:
        """Maximum loss amount, USD: max_loss_pct x C0 (None if no max rule)."""
        if self.max_loss_pct is None:
            return None
        return round(self.max_loss_pct * self.initial_capital, FLOOR_DECIMALS)

    def daily_floor(self, ref):
        """Daily-loss floor(s), USD, for day-start reference value(s) `ref` (USD, scalar or array).

        "initial": ref - daily_loss_pct x C0; "day_start": ref x (1 - daily_loss_pct). Returns -inf when
        there is no daily rule. Scalar in -> float out; array in -> float64 array.
        """
        r = np.asarray(ref, dtype=np.float64)
        if self.daily_loss_pct is None:
            out = np.full(r.shape, -np.inf)
        elif self.daily_loss_base == "initial":
            out = np.round(r - round(self.daily_loss_pct * self.initial_capital, FLOOR_DECIMALS), FLOOR_DECIMALS)
        else:
            out = np.round(r * (1.0 - self.daily_loss_pct), FLOOR_DECIMALS)
        return float(out) if out.ndim == 0 else out

    def max_floor(self, highest_day_start_balance=None):
        """Max-loss floor(s), USD.

        trailing_eod_balance: max(C0, highest B_00:00 so far) - max_loss_pct x C0, where
        `highest_day_start_balance` (USD, scalar or array) is the running maximum of B_00:00 (None = C0);
        static: C0 - max_loss_pct x C0. Returns -inf when there is no max rule. Scalar/None in -> float.
        """
        h = np.asarray(self.initial_capital if highest_day_start_balance is None else highest_day_start_balance,
                       dtype=np.float64)
        if self.max_loss_pct is None:
            out = np.full(h.shape, -np.inf)
        elif self.max_loss_mode == "trailing_eod_balance":
            out = np.round(np.maximum(h, self.initial_capital) - self.max_loss_usd, FLOOR_DECIMALS)
        else:
            out = np.full(h.shape, round(self.initial_capital - self.max_loss_usd, FLOOR_DECIMALS))
        return float(out) if out.ndim == 0 else out

    def breached(self, equity, floor):
        """True where equity (USD) breaches floor (USD): equity < floor, or <= with breach_inclusive.

        Scalars -> bool; arrays -> bool array (broadcast)."""
        e = np.asarray(equity, dtype=np.float64)
        f = np.asarray(floor, dtype=np.float64)
        out = (e <= f) if self.breach_inclusive else (e < f)
        return bool(out) if out.ndim == 0 else out

    def best_day_ok_from(self, best_positive, total_positive):
        """Vectorised best-day test: best_positive <= share x total_positive + 1e-6 USD.

        best_positive: the largest day profit so far (USD, >= 0); total_positive: the sum of all positive
        day profits so far (USD). Always True when there is no best-day rule. Scalars -> bool."""
        b = np.asarray(best_positive, dtype=np.float64)
        t = np.asarray(total_positive, dtype=np.float64)
        if self.best_day_max_share is None:
            out = np.ones(np.broadcast(b, t).shape, dtype=bool)
        else:
            out = b <= self.best_day_max_share * t + BEST_DAY_TOL_USD
        return bool(out) if out.ndim == 0 else out

    def best_day_ok(self, day_profits) -> bool:
        """Best-day rule on a list of day profits (USD, one per prop day, any sign).

        True when max(positive day profits) <= best_day_max_share x sum(positive day profits) (+1e-6 USD),
        or when there is no rule. With no positive day at all the rule holds trivially (0 <= 0)."""
        p = np.asarray(day_profits, dtype=np.float64).ravel()
        if not np.isfinite(p).all():
            raise ValueError("day_profits must be finite USD amounts")
        pos = np.maximum(p, 0.0)
        best = float(pos.max()) if pos.size else 0.0
        return bool(self.best_day_ok_from(best, float(pos.sum())))

    @staticmethod
    def best_day_share(day_profits) -> float | None:
        """Best positive day profit / sum of positive day profits (a fraction), None if no positive day."""
        p = np.asarray(day_profits, dtype=np.float64).ravel()
        pos = np.maximum(p, 0.0)
        total = float(pos.sum())
        return None if total <= 0 else float(pos.max()) / total

    # ---- presentation -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """All fields as a JSON-serialisable dict (fractions stay fractions)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PropRules":
        """Build from a dict such as to_dict() returns; unknown keys raise ValueError."""
        if not isinstance(data, Mapping):
            raise ValueError("rules must be given as a dict of field names to values")
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown rule field(s) {unknown}; known fields: {sorted(known)}")
        return cls(**dict(data))

    def describe(self) -> list[str]:
        """Plain-English ASCII lines describing the rules (for reports and `propkit rules`)."""
        c0 = self.initial_capital
        stay = "strictly above" if self.breach_inclusive else "at or above"   # matches breached()
        lines = [f"{self.name}: initial capital {c0:,.2f} USD"]
        if self.profit_target_pct is None:
            lines.append("  profit target: none")
        else:
            flat = " with no position open" if self.target_requires_flat else " (positions may be open)"
            lines.append(f"  profit target: +{self.profit_target_pct:.2%} = closed balance >= "
                         f"{self.target_balance:,.2f}{flat}")
        if self.daily_loss_pct is None:
            lines.append("  daily loss: no rule")
        else:
            ref = "balance at 00:00 CE(S)T" if self.day_start_reference == "balance" else \
                "max(balance, equity) at 00:00 CE(S)T"
            amount = (f"{self.daily_loss_pct:.2%} of initial capital = {self.daily_loss_pct * c0:,.2f}"
                      if self.daily_loss_base == "initial" else f"{self.daily_loss_pct:.2%} of the reference")
            lines.append(f"  daily loss: equity must stay {stay} ({ref}) - {amount}")
        if self.max_loss_pct is None:
            lines.append("  max loss: no rule")
        elif self.max_loss_mode == "static":
            lines.append(f"  max loss: equity must stay {stay} {self.max_floor():,.2f} (static, "
                         f"{self.max_loss_pct:.2%} of initial capital)")
        else:
            lines.append(f"  max loss: equity must stay {stay} max(initial capital, highest balance at "
                         f"00:00 CE(S)T so far) - {self.max_loss_usd:,.2f} (trailing, never moves down)")
        if self.best_day_max_share is not None:
            lines.append(f"  best day: best day profit <= {self.best_day_max_share:.0%} of the total profit of "
                         f"all positive days (day profit from {self.best_day_basis})")
        if self.min_trading_days:
            lines.append(f"  minimum trading days: {self.min_trading_days} prop days with a trade")
        lines.append("  breach when equity " + ("<=" if self.breach_inclusive else "<") + " a floor")
        if self.notes:
            lines.append(f"  notes: {self.notes}")
        return lines


# ---------------------------------------------------------------------------------------
# presets

def ftmo_1step(initial_capital: float = 100_000.0) -> PropRules:
    """FTMO 1-Step per CLAUDE.md C6 (as of 24 Sep 2026; recheck before every challenge).

    Target +10%; daily loss 3% of C0 below the 00:00 CE(S)T balance (L_daily = B_00:00 - 0.03 x C0);
    max loss 10% trailing the end-of-day balance (L_max = max(C0, highest B_00:00) - 0.10 x C0); best
    day <= 50% of the total profit of positive days; no minimum trading days.
    """
    return PropRules(name="FTMO 1-Step", initial_capital=initial_capital, profit_target_pct=0.10,
                     daily_loss_pct=0.03, daily_loss_base="initial", day_start_reference="balance",
                     max_loss_pct=0.10, max_loss_mode="trailing_eod_balance", best_day_max_share=0.5,
                     best_day_basis="balance", min_trading_days=0, notes=f"FTMO as of {FTMO_AS_OF}")


def ftmo_2step(initial_capital: float = 100_000.0, target: float = 0.10) -> PropRules:
    """FTMO 2-Step per CLAUDE.md C6 (as of 24 Sep 2026; recheck before every challenge).

    Target `target` (0.10 for phase 1, 0.05 for phase 2); daily loss 5% of C0 below the 00:00 CE(S)T
    balance; static max loss 10% of C0 (floor 0.90 x C0); at least 4 trading days; no best-day rule.
    """
    return PropRules(name=f"FTMO 2-Step (target {float(target) * 100:g}%)" if _is_number(target) else "FTMO 2-Step",
                     initial_capital=initial_capital, profit_target_pct=target, daily_loss_pct=0.05,
                     daily_loss_base="initial", day_start_reference="balance", max_loss_pct=0.10,
                     max_loss_mode="static", best_day_max_share=None, min_trading_days=4,
                     notes=f"FTMO as of {FTMO_AS_OF}")


def custom(**fields: Any) -> PropRules:
    """Any rule set: PropRules(**fields) with name defaulting to "custom" (see PropRules for the fields)."""
    fields.setdefault("name", "custom")
    return PropRules.from_dict(fields)


def preset(name: str, initial_capital: float = 100_000.0, /, **overrides: Any) -> PropRules:
    """A preset by CLI name ("ftmo-1step" or "ftmo-2step"), optionally with fields replaced.

    Example: preset("ftmo-2step", 50_000, profit_target_pct=0.05) for phase 2 on a 50k account; it is
    named "FTMO 2-Step (target 5%)" like ftmo_2step(50_000, 0.05). Any other change of a preset field
    renames the rules "<preset name> (modified: field=value, ...)" and says in the notes that they are
    not the firm's published terms, so a changed rule set is never shown under the preset's own name.
    A "name" (or "notes") given in the overrides is used as given. Fields whose value equals the preset's
    do not count as changes. initial_capital may also be given by keyword (it is not a change of terms).
    """
    if "initial_capital" in overrides:
        initial_capital = overrides.pop("initial_capital")
    key = str(name).strip().lower().replace("_", "-")
    if key == "ftmo-1step":
        base = ftmo_1step(initial_capital)
    elif key == "ftmo-2step":
        base = ftmo_2step(initial_capital)
    else:
        raise ValueError(f"unknown rules preset {name!r}; use one of {', '.join(PRESET_NAMES)} or custom(...)")
    if not overrides:
        return base
    data = base.to_dict()
    unknown = sorted(set(overrides) - set(data))
    if unknown:
        raise ValueError(f"unknown rule field(s) {unknown}")
    changed = {k: v for k, v in overrides.items() if k not in ("name", "notes") and not _same(v, data[k])}
    if key == "ftmo-2step" and _is_number(changed.get("profit_target_pct")):
        base = ftmo_2step(initial_capital, target=changed.pop("profit_target_pct"))   # phase 2: named by its target
        data = base.to_dict()
    if changed:
        text = ", ".join(f"{k}={v!r}" for k, v in changed.items())
        data["name"] = f"{base.name} (modified: {text})"
        data["notes"] = (f"{base.name} preset with {text} changed by the user: NOT the firm's published terms. "
                         f"Preset source: {base.notes}")
    data.update(overrides)
    return PropRules.from_dict(data)


def _same(a, b) -> bool:
    """Equal values (numbers compared as floats; bools only to bools)."""
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return isinstance(a, (bool, np.bool_)) and isinstance(b, (bool, np.bool_)) and bool(a) == bool(b)
    if _is_number(a) and _is_number(b):
        return float(a) == float(b)
    return a == b
