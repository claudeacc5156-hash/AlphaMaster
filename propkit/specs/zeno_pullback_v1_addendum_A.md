# zeno_pullback_v1, addendum A: FundingPips firm rules in the simulation

- **Status:** pre-registered on 2026-10-08, before any result of zeno_pullback_v1 exists (no real-data run has happened; the cloud has no price data). Frozen with its own sha256 in records/LEDGER.md.
- **What it changes:** only how the prop firm is simulated. zeno's 12 rules, defaults D1-D24, the costs and gates G0-G5 in zeno_pullback_v1.md (sha256 d36ad25f74c293166bd82f117cb96a6e6890a2ab2c3dd63dbc41bff52b67bbcb) are unchanged. The gates are judged in the evaluation run, which this addendum does not touch except for the reporting in A4.
- **Why:** the verified FundingPips rule sheet (research/FUNDINGPIPS_RULES_2026-10-08.md, sha256 23129e6b9df15d618199fcdeea03a7f3f5aed0dd18a9aa91cbf40f1412522e88) arrived after the freeze. It shows three things the frozen spec could not know:
  1. FundingPips' Master news rule covers about 15 USD event types, not just NFP, CPI, PPI and FOMC.
  2. Master accounts use dynamic metals leverage, which can cap position size.
  3. The daily-loss reset is "00:00 Platform Time (UTC+3)" with no statement about daylight saving.

## A1. A second Master run: "master_fp"

- **Kept as written:** the D23 run (variant "master") stays exactly as zeno wrote rule 9: 0.4% risk, and 10 minutes before NFP, CPI, PPI or FOMC, close any trade opened under 5 h earlier.
- **Added:** a run "master_fp" with everything as in "master", plus FundingPips' full restricted list:
  - **Calendar:** research/news_calendar/us_restricted_events_fundingpips_2015-01-01_2025-09-27.csv (2,825 events; sha256 6685ee94fa4d0b2c860e1d3fd3a780dc49ad65e3885a2de9f70fc4c218e53cba).
  - **Close rule:** 10 minutes before every restricted event T, any trade opened less than 5 h before T is closed (at the open of the bar containing T - 10 min, as D23).
  - **Entry block:** no entry from T - 5 min to T + 5 min for a release. For a Fed Chair appearance (FEDCHAIR), no entry from T - 5 min to T + D + 5 min, where D is the appearance's length: 180 min for testimony and 60 min for every other appearance [ASSUMPTION: the calendar has start times only].
  - **Unknown time:** a restricted event whose time is unknown (one row: Fed Chair, 2020-11-12) blocks entries for the whole New York calendar day. No position can then be open that day, so no close is needed.
  - Rule 9's own blackout (30 min before to 60 min after NFP, CPI, PPI and FOMC, D20) applies in both Master runs, as in the evaluation run.
- **Which Master run is primary:** the card in zeno's thread asks whether to widen rule 9's Master clause.
  - If zeno answers "Widen it", or does not answer before the first result is shown, "master_fp" is labelled the primary Master result and "master" the literal-rule comparison.
  - If zeno answers "Keep 4 events", "master" is primary and "master_fp" is a sensitivity run.
  - Neither Master run feeds G1-G5.

## A2. Master margin cap (both Master runs)

- FundingPips' dynamic leverage for metals on Master accounts [VP 1SF]: 0.00-0.05 lots 1:50; 0.05-0.10 1:30; 0.10-0.15 1:25; 0.15-0.25 1:20; 0.25-0.50 1:10; 0.50 and above 1:5; each tier applies only to the volume inside it.
- **Margin** for L lots at price P (USD/oz) = 100 x P x sum over tiers of (lots in the tier / tier leverage). Tiers are applied per position [ASSUMPTION; per position or per account is [U]]; the rule holds one position at a time, so the two readings agree.
- **Cap:** if the D13 lot size needs more margin than the closed balance at entry, the lots are cut to the largest 0.01-lot size whose margin fits [ASSUMPTION: zeno would downsize rather than skip]. The report counts capped entries and their lots before and after.
- **Where it binds (derived):** at 0.4% risk on USD 100,000, lots = 4 / R (R in USD per oz). At USD 3,700 gold the cap binds for R below about USD 2.38 (more than about 1.68 lots); at USD 4,000, for R below about USD 2.54 (1.58 lots). Before 2025 gold was below USD 2,800 and the cap rarely binds.
- **Evaluation run:** fixed metals leverage of 1:30 (Standard) or 1:10 (Swing/Swap-Free) [VP 1SF]. At 0.5% risk, lots = 5 / R, so margin exceeds USD 100,000 only when R < P / 6,000 at 1:30 (USD 0.73 at USD 4,400 gold, which rule 10 already rules out, since it needs R of at least 10 spreads, USD 1.80 at a USD 0.18 spread) or R < P / 2,000 at 1:10 (USD 2.20 at USD 4,400). The evaluation run is not capped (zeno's account type is not known; Standard is assumed); the report counts entries that would exceed margin at 1:10.

## A3. FundingPips rules in the prop evaluator

- The evaluator uses the preset "fundingpips-1step-flex" built from the sheet:
  - target 12%;
  - daily loss 2% of the higher of the day-start balance and equity, floating included, touching the floor is a breach;
  - max loss 12% static (floor USD 88,000);
  - no minimum days and no best-day rule for the Evaluation;
  - target counted only when flat [U].
- Master payout rules (minimum reward, the Monthly 100% consistency rule) and the Striking System ([U] applicability) are not simulated; the report lists them as not modelled.

## A4. Day-boundary sensitivity

- **Default:** the firm's day runs 17:00 New York to 17:00 New York ("ny_17"; zeno's MT5 clock reads New York + 7 h, and FundingPips' summer and winter auto-close windows fit a server that shifts with US daylight saving).
- **Sensitivity:** the same judging cell is also run with the firm's day fixed at 00:00 UTC+3 = 21:00 UTC all year ("utc_plus3"), the literal reading of "00:00 Platform Time (UTC+3)". Both P(daily-loss breach) values are reported; G4 uses the higher.

## A5. Gate G4

- G4 now uses the verified preset of A3, as the spec already required ("under the verified FundingPips rules").
- If the higher of the two P(daily-loss breach) values in A4 is above 5%, or P(max-loss breach) is above 10%, G4 fails.
