"""Deliverable 1: propkit/specs holds byte-exact copies of the frozen zeno_pullback_v1 records (sha256 as
in records/LEDGER.md), an ASCII-folded readable .md, and zeno_v1's constants agree with the JSON copy.
Research only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from propkit import zeno_v1 as z

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "propkit" / "specs"
MD_SHA = "d36ad25f74c293166bd82f117cb96a6e6890a2ab2c3dd63dbc41bff52b67bbcb"
JSON_SHA = "ebd8017a271229786c5a79f08baddd0e6b4940be52f8e28d40706dddb1f7b31a"
STAR = "\u2605"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_byte_exact_copies_match_the_recorded_hashes():
    assert _sha(SPECS / "zeno_pullback_v1.json") == JSON_SHA
    assert _sha(SPECS / "zeno_pullback_v1.md.utf8") == MD_SHA
    assert z.SPEC_SHA256_MD == MD_SHA and z.SPEC_SHA256_JSON == JSON_SHA
    assert z.SPECS_DIR == SPECS


def test_source_hashes_file_records_the_same_hashes_and_the_fold():
    d = json.loads((SPECS / "SOURCE_HASHES.json").read_text(encoding="ascii"))
    assert d["zeno_pullback_v1.json"] == JSON_SHA
    assert d["zeno_pullback_v1.md"] == MD_SHA
    assert d["md_ascii_fold"] == {"from": "U+2605", "to": "(*)", "occurrences": 5}


def test_ascii_md_is_the_utf8_text_with_only_the_star_folded():
    raw = (SPECS / "zeno_pullback_v1.md.utf8").read_bytes().decode("utf-8")
    folded = (SPECS / "zeno_pullback_v1.md").read_bytes()
    folded.decode("ascii")                                    # every byte is ASCII
    assert raw.count(STAR) == 5
    assert {c for c in raw if ord(c) > 127} == {STAR}         # nothing else needed folding
    assert raw.replace(STAR, "(*)").encode("ascii") == folded


def test_gitattributes_keeps_the_copies_byte_exact():
    text = (SPECS / ".gitattributes").read_text(encoding="ascii")
    for name in ("zeno_pullback_v1.json", "zeno_pullback_v1.md.utf8", "zeno_pullback_v1.md"):
        assert any(line.split()[:2] == [name, "-text"] for line in text.splitlines() if line.strip())


def test_md_names_the_spec_and_every_default():
    text = (SPECS / "zeno_pullback_v1.md").read_text(encoding="ascii")
    assert "zeno_pullback_v1" in text
    for d in range(1, 25):
        assert f"D{d}" in text


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads((SPECS / "zeno_pullback_v1.json").read_text(encoding="utf-8"))


def test_json_numbers_equal_the_module_constants(spec):
    assert spec["spec_id"] == z.SPEC_ID and spec["version"] == z.SPEC_VERSION and spec["placeholder"] is False
    assert spec["instrument"]["contract_oz"] == z.CONTRACT_OZ
    assert spec["instrument"]["lot_step"] * z.CONTRACT_OZ == z.LOT_STEP_OZ
    assert spec["data"]["end_exclusive_utc"] == "2025-09-28T00:00:00Z" and z.LOCK_UTC == 1759017600
    assert spec["data"]["warmup_trading_days"] == z.WARMUP_TRADING_DAYS
    tr, pb, tg = spec["trend"], spec["pullback"], spec["trigger"]
    assert (tr["ema_period"], tr["slope_bars"], tr["ema_seed"]) == (z.EMA_PERIOD, z.EMA_SLOPE_BARS, "sma_first_30")
    assert (pb["h_lookback_bars"], pb["l_lookback_bars_before_h"]) == (z.H_LOOKBACK, z.L_LOOKBACK)
    assert (pb["min_leg_atr"], pb["valid_retrace"], pb["void_retrace"]) == \
        (z.MIN_LEG_ATR, z.RETRACE_VALID, z.RETRACE_VOID)
    assert (tg["max_bars_after_pullback_low"], tg["min_bars_after_pullback_low"]) == (z.TRIGGER_MAX_BARS, 1)
    assert spec["atr"]["period"] == z.ATR_PERIOD and spec["atr"]["smoothing"] == "wilder"
    assert spec["stop"]["buffer_atr"] == z.STOP_BUFFER_ATR
    assert (spec["exit"]["tp1_r"], spec["exit"]["tp2_r"]) == (z.TP1_R, z.TP2_R)
    assert spec["exit"]["time_exit"]["local"] == "16:30" and z.TIME_EXIT_NY_SECONDS == 16 * 3600 + 1800
    assert spec["risk"]["risk_pct"] == z.RISK_PCT["evaluation"]
    assert spec["risk"]["master_variant_risk_pct"] == z.RISK_PCT["master"]
    f = spec["filters"]
    assert (f["vol_cap_atr_over_median"], f["vol_median_window_trading_days"], f["max_stop_atr"],
            f["max_spread_frac_of_stop"]) == (z.VOL_CAP_X, z.VOL_MEDIAN_DAYS, z.MAX_STOP_ATR, z.MAX_SPREAD_FRAC_OF_R)
    utc = [[int(a[:2]) * 3600 + int(a[3:]) * 60 for a in w] for w in f["entry_windows_utc"]]
    assert [tuple(w) for w in utc] == list(z.SESSION_WINDOWS_UTC)
    n = f["news"]
    assert (n["block_before_min"] * 60, n["block_after_min"] * 60) == (z.NEWS_BEFORE_S, z.NEWS_AFTER_S)
    assert n["master_variant_close_before_min"] * 60 == z.MASTER_CLOSE_BEFORE_S
    assert n["master_variant_close_if_opened_within_h"] * 3600 == z.MASTER_MAX_AGE_S
    assert [e.split()[0] for e in n["events"]] == list(z.NEWS_EVENTS)
    lim = spec["limits"]
    assert (lim["max_entries_per_day"], lim["stop_after_losses"]) == (z.MAX_ENTRIES_PER_DAY, z.MAX_LOSSES_PER_DAY)
    assert lim["stop_after_day_pnl_pct"] / -100 == z.DAY_LOSS_FRAC
    assert lim["same_direction_cooldown_min"] * 60 == z.COOLDOWN_S
    c = spec["costs"]
    assert tuple(c["commission_rt_per_lot_reported"]) == z.COMMISSIONS
    assert (c["spread_bases"]["S2"]["usd"], c["spread_bases"]["S2"]["rollover_usd"]) == (z.S2_USD, z.S2_ROLLOVER_USD)
    assert c["spread_bases"]["S2"]["rollover_sgt"] == ["05:00", "08:00"]      # = 21:00-24:00 UTC
    assert z.S2_ROLLOVER_UTC == (21 * 3600, 24 * 3600)
    assert c["stop_slippage_usd_per_oz"] == z.STOP_SLIPPAGE_USD
    assert tuple(c["multipliers"]) == z.COST_MULTS and c["swap"] == 0.0
    assert spec["gates"]["judged_at"] == {"commission": 10.0, "spread": "worse", "cost_multiplier": 1.5}
    assert z.ZenoCell() == z.ZenoCell("evaluation", 10.0, "S1", 1.5)


def test_grid_has_36_distinct_cells():
    cells = z.grid_cells()
    # updated for addendum A1: the variant master_fp adds 12 cells to the 24 (3 variants x 2 x 2 x 3)
    assert len(cells) == 36 == len({c.label for c in cells})
    assert {c.variant for c in cells} == set(z.VARIANTS)
    assert cells[0].label == "evaluation/c5/S1/x1"


def test_package_exports_the_module():
    import propkit
    assert propkit.zeno_v1 is z and "zeno_v1" in propkit.__all__
