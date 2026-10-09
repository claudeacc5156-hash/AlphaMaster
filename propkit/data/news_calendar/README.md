# US macro event calendar, 2015-01-01 to 2025-09-27

- **File:** us_macro_events_2015-01-01_2025-09-27.csv
  - 478 rows; sha256 31a6735adc03818e24f788507e44bd26eb89ca0ca3dd857464d55eacec0e3eab.
  - Built 2026-10-08 for the news blackout in zeno_pullback_v1 (rule 9, default D20).
- **Range:** ends at the holdout lock, 2025-09-28 00:00 UTC. Events after the lock need a separate file for any forward test.

## Contents

| Event | Rows | Time (New York) |
|---|---|---|
| NFP (Employment Situation) | 129 | 08:30 |
| CPI | 129 | 08:30 |
| PPI | 129 | 08:30 |
| FOMC statements | 91 | 14:00 for scheduled meetings; actual release time for the rest |

- Counts: 12 a year for 2015-2024 and 9 for 2025, except FOMC.
- FOMC counts by year: 2015-2018 have 8 each; 2019 has 9; 2020 has 11; 2021-2024 have 8 each; 2025 has 7.

### Columns

- `event`, `date_et`, `time_et`: the release in New York local time.
- `utc_offset_ny`, `datetime_utc`: computed with the IANA America/New_York rules, never by hand.
- `kind`: scheduled or unscheduled.
- `basis`:
  - `both`: two independent extractions agree;
  - `critic-verified`: added after a third check.
- `source_list`, `cross_check`, `note`: where each date was read.

## Method

1. **Two independent extractions per series**, by different official routes:
   - BLS archived release lists (the release date sits in each archive file name) versus ALFRED release dates plus the BLS yearly schedules;
   - for FOMC, the Fed's per-year historical pages and fomccalendars.htm versus its press releases and openmarket.htm.
2. **Agreement:**
   - CPI and FOMC agreed on every date.
   - NFP: route B had 3 extra dates (ALFRED data-revision vintages, e.g. the 2020-05-11 payroll correction). They were checked and excluded, because they were not releases.
   - PPI: route B had 7 extra dates, the February seasonal-factor postings. They were excluded, because they are not releases and have no official time.
3. **A third check** confirmed the following:
   - **Counts:** they are right for every year.
   - **Non-Friday NFP:** the three non-Friday NFP dates are the July 4 moves (2015-07-02, 2020-07-02, 2025-07-03).
   - **Thursday FOMC decisions:** 2015-09-17, 2018-11-08, 2020-11-05 and 2024-11-07.
   - **Cancelled meeting:** the cancelled March 17-18, 2020 meeting is absent.
   - **Unscheduled statement times:**
     - 2019-10-11 11:00;
     - 2020-03-03 10:00;
     - 2020-03-15 (Sunday) 17:00;
     - 2020-03-23 08:00.
   - **Weekends and holidays:** no date falls on a weekend or a US federal holiday, apart from the official Sunday 2020-03-15 statement.
4. **Inclusion rule** (D20, "plus unscheduled FOMC statements"):
   - Two FOMC strategy-framework statements released by notation vote are included: 2020-08-27 at 09:10 and 2025-08-22 at 10:00.
   - Notation votes with no FOMC statement are excluded: the 2020-03-19 and 2020-03-31 swap-line and FIMA releases.
   - FOMC minutes are excluded.

## Limits

- Dates were read through a web-reading tool that summarises pages. Each date is therefore backed by two independent official routes plus a third check, not by a single read.
- Release times are the official scheduled times. The 08:30 BLS time is uniform across the range.
- The calendar holds scheduled times only. If a release slipped by minutes, that is not captured.

---

# FundingPips restricted USD events, 2015-01-01 to 2025-09-27

- **File:** us_restricted_events_fundingpips_2015-01-01_2025-09-27.csv
  - 2,825 rows; sha256 6685ee94fa4d0b2c860e1d3fd3a780dc49ad65e3885a2de9f70fc4c218e53cba.
  - Built 2026-10-08 for the Master-account variant of zeno_pullback_v1 (D23), on the recommended answer to the "widen the Master news clause" question. FundingPips' Master news rule names about 15 USD event types, while rule 9 named only NFP, CPI, PPI and FOMC (see research/FUNDINGPIPS_RULES_2026-10-08.md, section 8).
- **Range:** first row 2015-01-02, last row 2025-09-25; ends at the holdout lock (2025-09-28 00:00 UTC).
- **Columns:** the same as the first file, without `cross_check`: event, date_et, time_et, utc_offset_ny, datetime_utc, kind, basis, source_list, note.
- **The first file is unchanged.** Its 478 NFP/CPI/PPI/FOMC rows are copied in here as they are.

## Contents

| Event | Rows | Time (New York) | Sources |
|---|---|---|---|
| CLAIMS (initial jobless claims) | 560 | 08:30 | DOL press archive + ALFRED release 180 |
| GDP (advance, second, third) | 128 | 08:30 | BEA news archive + ALFRED release 53 |
| TRADE (goods and services) | 129 | 08:30 | BEA/Census archive + ALFRED release 51 |
| DURABLE (advance durable goods) | 129 | 08:30 | Census advance-report PDFs + ALFRED release 95 |
| NEWHOME (new home sales) | 129 | 10:00 | Census NRS PDFs + ALFRED release 97 |
| EXISTHOME (existing home sales) | 129 | 10:00 | PR Newswire NAR releases + Calculated Risk (not first-party) |
| JOLTS | 129 | 10:00 | BLS archive + ALFRED release 192 |
| ISM_MFG | 129 | 10:00 | ISM / PR Newswire |
| ISM_SVC | 129 | 10:00 | ISM / PR Newswire |
| SPGPMI (S&P Global / Markit US manufacturing flash PMI) | 154 | 09:45 | Bigdata.com calendar + news wires (not first-party) |
| CONF (Conference Board consumer confidence) | 128 | 10:00 | Conference Board / PR Newswire |
| AUCT10 (10-year note auctions, incl. reopenings) | 129 | 13:00 | Treasury FiscalData auctions query |
| AUCT30 (30-year bond auctions, incl. reopenings) | 129 | 13:00 | Treasury FiscalData auctions query |
| FEDCHAIR (Fed Chair speeches, testimony, discussions) | 216 | as scheduled | federalreserve.gov speeches, testimony and calendars; Senate Banking hearing list; dated wires |
| NFP, CPI, PPI | 129 each | 08:30 | from the first file |
| FOMC | 91 | 14:00 scheduled; actual time otherwise | from the first file |

- CLAIMS has 52 rows a year (53 in 2020) and 39 in 2025.
- SPGPMI has 24 rows a year in 2015 and 2016 and 13 in 2017, because the old calendar listed two releases a month until January 2017; after that it has 12 a year (9 in 2025).
- FEDCHAIR rows by year: 2015 17, 2016 14, 2017 23, 2018 17, 2019 28, 2020 22, 2021 28, 2022 18, 2023 17, 2024 18, 2025 14.

## Method

- **Two routes per series**, as for the first file. Each row's `basis` is `both` (the routes agree) except as follows.
- **FEDCHAIR `basis`:** 142 rows are `both`, 27 are `A` (the Fed's speech PDF only), 39 are `B` (the Fed's monthly calendar or the Senate Banking hearing list only) and 8 are `critic-verified` (added by a third check).
  - A third check flagged 11 Fed Chair events as missing. Eight were new and are the `critic-verified` rows: 2016-04-07 17:30, 2016-05-27 13:15, 2017-06-27 13:00, 2018-11-14 17:00, 2019-09-06 12:30, 2020-11-12 (time unknown), 2022-09-23 14:00 and 2024-03-22 09:00.
  - The other three (2019-02-12 11:00, 2022-04-21 11:00 and 2023-12-01 11:00) were second appearances on a day that already had one. The routes had found them, but a first build keyed rows by date only and dropped them. FEDCHAIR was rebuilt from the raw journal keyed by date and time, so two appearances on one day are now two rows.
  - Where the Fed and the Senate gave different start times for one testimony, both rows are kept, so the blackout covers both: 2017-07-13 (09:30 and 10:00), 2019-02-26 (09:30 and 09:45), 2020-02-12 (09:30 and 10:00) and 2024-03-07 (09:40 and 10:00).
- **Removed:** AUCT10 on 2019-06-21. It was a $25 million contingency test auction, not a market event.
- **FOMC:** two rows are `critic-verified`, as in the first file.

## Limits

- **One event has no time:** a Fed Chair appearance on 2020-11-12, at about 11:45 ET per the third check. Its `time_et` is `unknown` and its `datetime_utc` is blank. A consumer must handle it on purpose; the conservative choice is to treat the whole New York day as restricted.
- **Not first-party:** SPGPMI rests mostly on the Bigdata.com calendar plus wires, and EXISTHOME on PR Newswire and Calculated Risk (PR Newswire's NAR listing starts on 2015-02-10, so the January 2015 date comes from same-day reports).
- **FEDCHAIR misses some appearances:** unscripted or unlisted ones from 2015 to March 2017, when the Fed's online calendar did not list them. FEDCHAIR therefore under-counts restricted minutes early in the sample, so a Master backtest takes some trades that the live account would have had to skip or close. The direction of that bias is unknown, because a skipped trade can be a winner or a loser.
- **FundingPips' list is approximate.** The firm's help pages name event types, not a dated list; its own economic calendar decides in live trading. This file is a historical reconstruction of those types, not FundingPips' record.
- Times are scheduled times, as in the first file.
