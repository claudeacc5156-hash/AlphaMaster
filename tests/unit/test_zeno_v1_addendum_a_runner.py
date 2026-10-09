"""Addendum A (zeno_pullback_v1) in the runner: the packaged addendum copy and its sha256, `zeno-v1 run` with
the verified FundingPips preset by default, 36 cells, --restricted (its sha256 in report.json and gates.json),
--master-primary with A1's wording, the prop stack for the judging cell and both Master twins, the judging
cell again under the other firm day (A4) and G4 from the higher P(daily-loss breach) (A5); `zeno-v1 signals
--variant master_fp`. Research only; synthetic bars (NOT market data).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from propkit import cli
from propkit import rules as R
from propkit import zeno_report as zr
from propkit import zeno_v1 as z
from tests.unit.test_zeno_v1_runner import FAST, inputs, run, write_pair
from tests.unit.zeno_v1_testkit import NEWS_CSV

ROOT = Path(__file__).resolve().parents[2]
ADDENDUM = ROOT / "propkit" / "specs" / "zeno_pullback_v1_addendum_A.md"
ADDENDUM_SHA = "0f64bf584e325cc665e1afee0f9dccb9e53f83645f078abad2d455f5982af7b0"


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """The runner tests' synthetic data (~6 months of M15 bid/ask bars from 2015-01-01, spread 0.05 USD/oz)."""
    d = tmp_path_factory.mktemp("zeno_addendum_a")
    frame = z.synthetic_m15_bidask(n_bars=16_000, seed=3, spread=0.05)
    bid, ask = write_pair(d / "in", frame)
    news = d / "in" / "news.csv"
    news.write_text(NEWS_CSV, encoding="ascii")
    return {"dir": d, "bid": bid, "ask": ask, "news": news, "frame": frame}


@pytest.fixture(scope="module")
def custom_restricted(tmp_path_factory):
    """A restricted calendar that is NOT the packaged file: its first 200 rows plus the unknown-time row."""
    df = pd.read_csv(z.RESTRICTED_CSV, dtype=str, keep_default_na=False)
    keep = df.iloc[:200]
    unknown = df[df["time_et"].str.lower() == "unknown"]
    p = tmp_path_factory.mktemp("restricted") / "restricted_subset.csv"
    pd.concat([keep, unknown]).to_csv(p, index=False)
    return p


@pytest.fixture(scope="module")
def default_run(files, tmp_path_factory):
    """One `zeno-v1 run` with every default (rules fundingpips-1step-flex, the packaged restricted calendar,
    --master-primary master_fp)."""
    out = tmp_path_factory.mktemp("run_default") / "run"
    buf, err = io.StringIO(), io.StringIO()                  # capsys is function-scoped: capture by hand
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        code = cli.main([str(a) for a in ["zeno-v1", "run", "--g0-confirmed"] + inputs(files) + ["--out", out] + FAST])
    assert code == 0, err.getvalue()
    return {"out": out, "stdout": buf.getvalue(),
            "report": json.loads((out / "report.json").read_text(encoding="ascii")),
            "gates": json.loads((out / "gates.json").read_text(encoding="ascii")),
            "md": (out / "report.md").read_text(encoding="ascii")}


# ---------------------------------------------------------------------------------------
# the addendum copy

def test_addendum_copy_is_byte_exact_ascii_and_hashed():
    assert _sha(ADDENDUM) == ADDENDUM_SHA == z.ADDENDUM_A_SHA256
    ADDENDUM.read_bytes().decode("ascii")
    text = (ROOT / "propkit" / "specs" / ".gitattributes").read_text(encoding="ascii")
    assert any(line.split()[:2] == [ADDENDUM.name, "-text"] for line in text.splitlines() if line.strip())
    a = zr.addendum_identity()
    assert (a["sha256_packaged"], a["matches"]) == (ADDENDUM_SHA, True)
    assert a["file"] == "propkit/specs/zeno_pullback_v1_addendum_A.md"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def test_the_packaged_calendars_reach_a_commit_byte_exact():
    # --restricted defaults to the packaged restricted calendar (zeno-v1 run reads it for every master_fp cell)
    # and the news tests read the packaged macro calendar, so a commit must hold both files with the bytes
    # whose sha256 is recorded. The root .gitignore's "data/" matched propkit/data/, and the root
    # .gitattributes' "* text=auto" would store these CRLF files with LF endings (another sha256 after checkout).
    if shutil.which("git") is None or _git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        pytest.skip("not inside a git work tree")
    folder = z.RESTRICTED_CSV.parent
    names = {p.name for p in folder.iterdir()}
    assert {z.RESTRICTED_CSV.name, "us_macro_events_2015-01-01_2025-09-27.csv", "README.md"} <= names
    for p in sorted(folder.iterdir()):
        rel = p.relative_to(ROOT).as_posix()
        assert _git("check-ignore", "-q", rel).returncode == 1, f"{rel} is ignored by git"
        stored = _git("hash-object", "--path", rel, rel).stdout.strip()     # the blob `git add` would store
        raw = _git("hash-object", "--no-filters", rel).stdout.strip()       # the bytes on disk
        assert stored and stored == raw, f"git would convert the line endings of {rel}"


# ---------------------------------------------------------------------------------------
# zeno-v1 run with every default

def test_run_defaults_to_the_verified_preset_and_36_cells(default_run):
    rep, g, md, stdout = default_run["report"], default_run["gates"], default_run["md"], default_run["stdout"]
    rs = rep["rules_section"]
    assert rs["rules"]["name"].startswith("FundingPips 1 Step Flex 100K") and not rs["info"].get("fallback")
    assert rs["info"]["verified"] is True and rep["prop"]["g4_applicable"] is True
    assert g["gates"]["G4"]["status"] in ("pass", "fail")                    # evaluated under the verified preset
    assert "Running the 36 cells" in stdout and "Loading FundingPips' restricted calendar" in stdout
    grid = pd.read_csv(default_run["out"] / "grid.csv")
    assert grid["cell"].nunique() == 36 and set(grid["variant"]) == set(z.VARIANTS)
    assert rep["settings"]["n_cells"] == 36
    # the report header names the addendum with its hash, then every [U] tag and the "unmodelled" list
    assert f"with addendum A (propkit/specs/zeno_pullback_v1_addendum_A.md, sha256 {ADDENDUM_SHA}" in md
    assert g["addendum_a"]["sha256_packaged"] == ADDENDUM_SHA and g["addendum_a"]["matches"] is True
    head = md.split("## Verdict and gates", 1)[0]
    info = R.rules_and_info("fundingpips-1step-flex")[1]
    assert info["u_tags"] and info["unmodelled"]
    for field, tag in info["u_tags"].items():
        assert f"- {field}: {tag}" in head
    for item in info["unmodelled"]:
        assert f"- {item}" in head


def test_run_records_the_restricted_calendar_and_its_sha256(default_run):
    rep, g = default_run["report"], default_run["gates"]
    assert rep["restricted"]["sha256"] == z.RESTRICTED_SHA256 == g["restricted_calendar"]["sha256"]
    assert g["restricted_calendar"]["is_packaged_file"] is True and rep["restricted"]["is_packaged_file"] is True
    assert rep["inputs"]["restricted"]["sha256"] == z.RESTRICTED_SHA256
    assert Path(rep["inputs"]["restricted"]["file"]).resolve() == z.RESTRICTED_CSV.resolve()
    assert rep["restricted"]["n_events_known_time"] == 2824 and rep["restricted"]["n_unknown_time"] == 1
    assert "restricted events (addendum A1, master_fp): 2824 events with a time" in default_run["md"]


def test_run_has_the_prop_stack_for_the_judging_cell_and_both_master_twins(default_run):
    rep = default_run["report"]
    j = rep["judging"]["label"]
    prop = rep["prop"]
    for key, variant in (("judging", "evaluation"), ("judging_day_sensitivity", "evaluation"),
                         ("master_twin", "master"), ("master_fp_twin", "master_fp")):
        p = prop[key]
        assert p["cell"] == j.replace("evaluation/", f"{variant}/", 1), key
        assert {"path", "bootstrap", "max_size"} <= set(p), key
        assert p["bootstrap"]["n_sims"] == 200 and p["bootstrap"]["horizon_days"] is None
    assert (prop["judging"]["day_boundary"], prop["judging_day_sensitivity"]["day_boundary"]) == ("ny_17", "utc_plus3")
    m = rep["master"]
    assert set(m["runs"]) == {"master", "master_fp"}
    for v in ("master", "master_fp"):
        assert m["runs"][v]["p_breach_daily"] == prop[f"{v}_twin"]["bootstrap"]["p_breach_daily"]
        assert m["runs"][v]["margin"]["cap_applies"] is True
    assert "n_triggers_blocked" in m["runs"]["master_fp"]["restricted"]
    assert m["runs"]["master"]["restricted"] is None


def test_run_g4_uses_the_higher_daily_breach_of_both_firm_days(default_run):
    rep, g, md = default_run["report"], default_run["gates"], default_run["md"]
    v = g["gates"]["G4"]["value"]
    by = v["p_breach_daily_by_boundary"]
    pd_ny = rep["prop"]["judging"]["bootstrap"]["p_breach_daily"]
    pd_u3 = rep["prop"]["judging_day_sensitivity"]["bootstrap"]["p_breach_daily"]
    assert by == {"ny_17": pd_ny, "utc_plus3": pd_u3} and v["p_breach_daily"] == max(pd_ny, pd_u3)
    assert v["p_breach_daily_set_by"] == ("utc_plus3" if pd_u3 > pd_ny else "ny_17")
    assert v["p_breach_max"] == rep["prop"]["judging"]["bootstrap"]["p_breach_max"]
    db = g["day_boundaries"]
    assert (db["default"], db["sensitivity"], db["g4_p_breach_daily_set_by"]) == ("ny_17", "utc_plus3",
                                                                                  v["p_breach_daily_set_by"])
    assert set(db["by_boundary"]) == {"ny_17", "utc_plus3"}
    assert "## Firm day boundary (addendum A4, A5)" in md
    assert f"G4's P(daily-loss breach) is set by {v['p_breach_daily_set_by']}." in md
    assert "| utc_plus3 (00:00 UTC+3) | sensitivity | 21:00 UTC all year, no daylight saving |" in md
    assert "(ny_17 " in md.split("| G4 prop survival |", 1)[1].split("\n", 1)[0]


def test_run_report_shows_the_master_and_margin_sections(default_run):
    md, g = default_run["md"], default_run["gates"]
    mp = g["master_primary"]
    assert mp["primary"] == "master_fp" and mp["roles"] == {"master_fp": "the primary Master result",
                                                           "master": "the literal-rule comparison"}
    assert "Widen it" in mp["why"] and mp["gated"] == "Neither Master run feeds G1-G5 (addendum A1)."
    assert "## Master runs (addendum A1)" in md and "Primary Master result: master_fp" in md
    assert "| master_fp | the primary Master result | master_fp/" in md
    assert "| master | the literal-rule comparison | master/" in md
    assert "## Margin (addendum A2)" in md and "| all 12 master_fp cells |" in md and "| all 12 evaluation cells |" in md
    assert md.index("## Master runs") < md.index("## Margin") < md.index("## The 36 cells")
    mg = default_run["report"]["margin"]
    assert set(mg["grid_totals"]) == set(z.VARIANTS)
    assert {"n_over_flat_1to10", "n_over_flat_1to30"} <= set(mg["grid_totals"]["evaluation"])
    assert {"n_capped", "lots_before_cap", "lots_after_cap", "n_blocked"} <= set(mg["grid_totals"]["master_fp"])


def test_run_report_lists_a3_master_rules_as_not_modelled(default_run):
    # A3: "Master payout rules (minimum reward, the Monthly 100% consistency rule) and the Striking System ([U]
    # applicability) are not simulated; the report lists them as not modelled". The rules file's last
    # unmodelled item says the Striking System is "handled by the strategy's Master variant": no code models it,
    # so every report line that repeats the rules file's claim carries the report's correction.
    rep, md = default_run["report"], default_run["md"]
    names = ["Master minimum reward", "Master Monthly 100% consistency rule", "Striking System ([U] applicability)"]
    head = md.split("## Verdict and gates", 1)[0]
    rules = md.split("\n## Rules\n", 1)[1].split("\n## ", 1)[0]
    master = md.split("## Master runs (addendum A1)", 1)[1].split("\n## ", 1)[0]
    for part in (head, rules):
        assert "Not simulated in any run (addendum A3):" in part
        for name in names:
            assert f"- {name}: " in part
    assert f"Neither Master run simulates (addendum A3): {', '.join(names)}." in master
    claims = [x for x in md.splitlines() if "handled by the strategy's Master variant" in x]
    assert len(claims) == 2                                       # the rules header and the Rules section
    for x in claims:
        assert "it does not simulate the profit deduction or the Striking System" in x
    # report.json: the same list beside the rules file's (kept as given) and in the Master section
    a3 = [f"{name}: {text}" for name, text in zr.A3_NOT_MODELLED]
    assert [n for n, _ in zr.A3_NOT_MODELLED] == names
    rs = rep["rules_section"]
    assert rs["not_modelled_addendum_a3"] == a3 == rep["master"]["not_modelled_addendum_a3"]
    assert rs["unmodelled_master_note"] == zr.UNMODELLED_MASTER_NOTE and all(x.endswith(rs["unmodelled_master_note"])
                                                                             for x in claims)
    assert rs["unmodelled"] == R.rules_and_info("fundingpips-1step-flex")[1]["unmodelled"]
    for item in a3:
        assert f"- {item}" in head and f"- {item}" in rules


def test_run_with_master_primary_master_and_another_restricted_file(files, custom_restricted, tmp_path, capsys):
    out = tmp_path / "run"
    code, stdout, err = run(["zeno-v1", "run", "--g0-confirmed"] + inputs(files)
                            + ["--out", out, "--restricted", custom_restricted, "--master-primary", "master"] + FAST,
                            capsys)
    assert code == 0, err
    g = json.loads((out / "gates.json").read_text(encoding="ascii"))
    rep = json.loads((out / "report.json").read_text(encoding="ascii"))
    sha = _sha(custom_restricted)
    assert g["restricted_calendar"]["sha256"] == sha == rep["restricted"]["sha256"] != z.RESTRICTED_SHA256
    assert g["restricted_calendar"]["is_packaged_file"] is False and "NOT the packaged file" in stdout
    assert rep["restricted"]["n_events_known_time"] == 200 and rep["restricted"]["n_unknown_time"] == 1
    assert g["master_primary"]["primary"] == "master"
    assert g["master_primary"]["roles"] == {"master": "the primary Master result", "master_fp": "a sensitivity run"}
    assert "Keep 4 events" in g["master_primary"]["why"]
    md = (out / "report.md").read_text(encoding="ascii")
    assert "Primary Master result: master" in md and "| master_fp | a sensitivity run |" in md
    assert "NOT the packaged file" in md


def test_run_refuses_a_bad_master_primary(files, tmp_path, capsys):
    code, _, err = run(["zeno-v1", "run", "--g0-confirmed"] + inputs(files)
                       + ["--out", tmp_path / "x", "--master-primary", "both"] + FAST, capsys)
    assert code == 2 and "--master-primary" in err


def test_run_stage_needs_the_restricted_calendar_for_master_fp(files):
    prep = z.prepare(files["frame"].iloc[:3000].reset_index(drop=True))
    rules, info = R.rules_and_info("fundingpips-1step-flex")
    cells = [z.ZenoCell("master_fp", 10.0, "S1", 1.5)]
    with pytest.raises(ValueError, match="restricted"):
        zr.run_stage(prep, rules, info, n_sims=50, history_reps=0, cells=cells)
    with pytest.raises(ValueError, match="master_primary"):
        zr.run_stage(prep, rules, info, n_sims=50, history_reps=0, cells=cells, master_primary="both")


def test_the_sensitivity_is_ny_17_when_the_rules_already_use_utc_plus3(files, tmp_path):
    # A4's other firm day: a rules file whose own day is utc_plus3 gets the ny_17 run beside it
    data = json.loads((ROOT / "propkit" / "presets" / "fundingpips_1step_flex.json").read_text(encoding="ascii"))
    data["day_boundary"] = "utc_plus3"
    data["name"] = "TEST FIXTURE: the FundingPips preset with day_boundary utc_plus3"
    p = tmp_path / "fp_utc_plus3.json"
    p.write_text(json.dumps(data), encoding="ascii")
    rules, info = R.rules_and_info(str(p))
    assert rules.day_boundary == "utc_plus3" and zr.g4_applicability(rules, info) == (True, "")
    prep = z.prepare(files["frame"])
    cells = [z.ZenoCell("evaluation", 10.0, b, k) for b in z.SPREAD_BASES for k in (1.0, 1.5)]
    rep, _ = zr.run_stage(prep, rules, info, n_sims=100, history_reps=0, cells=cells)
    assert rep["settings"]["day_boundary_sensitivity"] == "ny_17"
    assert rep["prop"]["judging_day_sensitivity"]["day_boundary"] == "ny_17"
    v = rep["gates"]["gates"]["G4"]["value"]
    assert set(v["p_breach_daily_by_boundary"]) == {"utc_plus3", "ny_17"}
    assert rep["gates"]["day_boundaries"]["default"] == "utc_plus3"
    assert "master" not in rep["prop"].get("master_twin", {}) and rep["master"]["runs"] == {}


@pytest.mark.parametrize("own, alt, status, set_by, p_daily", [
    ((0.03, 0.05), (0.06, 0.02), "fail", "utc_plus3", 0.06),     # the sensitivity's higher P(daily) fails G4
    ((0.04, 0.05), (0.02, 0.20), "pass", "ny_17", 0.04),         # P(max) is the default's (0.05), not 0.20
    ((0.03, 0.05), (0.03, 0.05), "pass", "ny_17", 0.03),         # a tie goes to the default
    ((0.01, 0.11), (0.01, 0.05), "fail", "ny_17", 0.01),         # the default's P(max) above 10% fails
    ((0.05, 0.10), (0.05, 0.10), "pass", "ny_17", 0.05),         # at both limits: pass (<=)
])
def test_g4_gate_known_answers_with_two_firm_days(own, alt, status, set_by, p_daily):
    boot = {"p_breach_daily": own[0], "p_breach_max": own[1], "se_breach_daily": 0.001, "n_sims": 10_000,
            "horizon_days": None}
    sens = {"p_breach_daily": alt[0], "p_breach_max": alt[1], "se_breach_daily": 0.002}
    g = zr.g4_gate(True, "", boot, "evaluation/c10/S1/x1.5", sensitivity=("utc_plus3", sens), day_boundary="ny_17")
    v = g["value"]
    assert (g["status"], v["p_breach_daily_set_by"], v["p_breach_daily"], v["p_breach_max"]) == (
        status, set_by, p_daily, own[1])
    assert v["p_breach_daily_by_boundary"] == {"ny_17": own[0], "utc_plus3": alt[0]}
    assert v["p_breach_max_by_boundary"] == {"ny_17": own[1], "utc_plus3": alt[1]}
    assert v["p_breach_daily_tie"] is (own[0] == alt[0])
    assert v["se_breach_daily"] == (0.002 if set_by == "utc_plus3" else 0.001)
    assert "addendum A5" in g["threshold"]
    # without a sensitivity G4 is the pre-addendum gate
    g1 = zr.g4_gate(True, "", boot, "evaluation/c10/S1/x1.5")
    assert "p_breach_daily_set_by" not in g1["value"] and g1["value"]["p_breach_daily"] == own[0]


# ---------------------------------------------------------------------------------------
# zeno-v1 signals

def test_signals_variant_master_fp_reads_the_packaged_restricted_calendar(files, tmp_path, capsys):
    out = tmp_path / "sig"
    code, stdout, err = run(["zeno-v1", "signals"] + inputs(files) + ["--out", out, "--variant", "master_fp"], capsys)
    assert code == 0, err
    rep = json.loads((out / "signals_report.json").read_text(encoding="ascii"))
    assert rep["declared_cell"]["variant"] == "master_fp" and rep["declared_cell"]["risk_pct"] == 0.004
    assert rep["restricted"]["sha256"] == z.RESTRICTED_SHA256 and rep["restricted"]["is_packaged_file"] is True
    assert rep["inputs"]["restricted"]["sha256"] == z.RESTRICTED_SHA256
    assert rep["addendum_a"]["sha256_packaged"] == ADDENDUM_SHA
    assert {"fp_restricted_window", "margin_cap_below_lot_step"} <= set(rep["block_reasons"])
    md = (out / "signals_report.md").read_text(encoding="ascii")
    assert "restricted events (variant master_fp, addendum A1)" in md and ADDENDUM_SHA in md


def test_signals_evaluation_ignores_restricted_and_keeps_its_output(files, custom_restricted, tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    code, _, err = run(["zeno-v1", "signals"] + inputs(files) + ["--out", a], capsys)
    assert code == 0, err
    code, stdout, err = run(["zeno-v1", "signals"] + inputs(files) + ["--out", b, "--restricted", custom_restricted],
                            capsys)
    assert code == 0, err
    assert "NOTE: --restricted" in stdout and "only the master_fp variant uses it" in stdout
    for name in ("signals.csv", "decisions.csv", "g0_sample.csv"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    rep = json.loads((b / "signals_report.json").read_text(encoding="ascii"))
    assert "restricted" not in rep and "restricted" not in rep["inputs"]
    assert not {"fp_restricted_window", "margin_cap_below_lot_step"} & set(rep["block_reasons"])


def test_report_wording_for_u_tags_and_the_master_max_size():
    # [VP] fields with a [U] part must not read as wholly unverified; a Master twin's size multiplier
    # ignores the A2 margin cap, and the report must say so (evaluation blocks carry no such note)
    from propkit import zeno_report as zr
    head = "\n".join(zr._rules_header({"u_tags": {"max_loss_mode": "[VP 1SF] static; the word is [U]"}}, {}))
    assert "unverified [U] part" in head and "not verified" not in head
    p = {"cell": "master_fp/c10/S2/x1.5", "max_size": {"multiplier": 2.0, "note": "ok"}, "rules_name": "r"}
    master = "\n".join(zr._prop_block("t", p, {"profit_target_pct": 0.12}))
    assert "margin cap is not applied" in master
    evaluation = "\n".join(zr._prop_block("t", {**p, "cell": "evaluation/c10/S2/x1.5"}, {"profit_target_pct": 0.12}))
    assert "margin cap is not applied" not in evaluation
