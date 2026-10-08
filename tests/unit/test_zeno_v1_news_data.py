"""The news calendars shipped in propkit/data/news_calendar (D20, D23): byte-exact copies of the research files."""

import hashlib
from pathlib import Path

from propkit import zeno_v1 as zv

DATA = Path(zv.__file__).resolve().parent / "data" / "news_calendar"
MACRO = DATA / "us_macro_events_2015-01-01_2025-09-27.csv"
RESTRICTED = DATA / "us_restricted_events_fundingpips_2015-01-01_2025-09-27.csv"

# sha256 recorded in records/LEDGER.md (2026-10-08 15:20Z and 19:30Z) and research/news_calendar/README.md
MACRO_SHA = "31a6735adc03818e24f788507e44bd26eb89ca0ca3dd857464d55eacec0e3eab"
RESTRICTED_SHA = "6685ee94fa4d0b2c860e1d3fd3a780dc49ad65e3885a2de9f70fc4c218e53cba"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_the_shipped_calendars_match_the_recorded_hashes():
    assert _sha(MACRO) == MACRO_SHA
    assert _sha(RESTRICTED) == RESTRICTED_SHA


def test_the_macro_calendar_reads_as_the_d20_events():
    cal = zv.read_news_csv(MACRO)
    s = cal.summary()
    assert s["n_events"] == 478
    assert s["per_event"] == {"CPI": 129, "FOMC": 91, "NFP": 129, "PPI": 129}
    assert s["sha256"] == MACRO_SHA
    assert s["last_utc"] < "2025-09-28"


def test_the_restricted_calendar_keeps_only_the_d20_events_when_read_for_rule_9():
    # rule 9's own blackout uses NFP, CPI, PPI and FOMC only, whichever file it is given
    s = zv.read_news_csv(RESTRICTED).summary()
    assert s["per_event"] == {"CPI": 129, "FOMC": 91, "NFP": 129, "PPI": 129}
