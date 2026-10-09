"""propkit/zeno_g0_charts.py - the G0 charts behind `python -m propkit zeno-v1 g0-charts`.

RESEARCH ONLY - not trading advice. A viewer: it draws each row of a G0 sample (g0_sample.csv, stage 1) from
the same M15 bid/ask bars the rule ran on, with every element of zeno_pullback_v1's setup marked, so zeno can
judge row by row whether the code's setup is zeno's rule (spec: Gates, G0). It changes no result, reads no
price by itself (the caller passes the prepared data) and writes nothing (the command line writes the page).

One self-contained HTML page (inline CSS and inline SVG; no script, no font, no link, nothing fetched):
a header (data files with sha256 and bar count, the sample's sha256, the cell of the chart prices, how to
answer) and one section per sample row: an M15 bid candlestick chart from before the 20-bar windows to the
TRIGGER bar, then the ENTRY marker at the next bar's open and NOTHING after it (no high, low or close of the
entry bar or of any later bar, no exit, no P&L, no R multiple, no outcome); a 1h panel with EMA30; and a
checklist of the numbers each rule needs.

Every number comes from the engine's own code path (propkit.zeno_v1): the setup events of prepare()
(setup_machines: H, L, leg, ATR14 at arming, the 50% and 78.6% levels, the arming, pullback and trigger bars),
screen() in the declared cell (eligible or not) and chart_prices() (the entry and stop at the data's own
prices, [SI-66]), the 1h bars and EMA30 of prepare() (h1_from_m15, ema30_h1) read through
h1_index_at_m15_close, the trend, session and news flags of prepare(), and the volatility median of
vol_medians. Two bars the engine uses but does not record are located here with the engine's functions and
constants: the bar of the far end of the leg (the latest bar of the 20 before the extreme holding L, or H for
a short: propkit.indicators.swing_low_index / swing_high_index, the latest on a tie as D5 does for H) and the
first bar after the extreme whose wick reached the 50% level (to LEVEL_TOL, [SI-34], [SI-35]).

No-leak guarantee: a row's section reads the bars up to its trigger bar, the trigger bar's close instant and
the entry bar's OPEN prices (bid and ask) and time, and nothing later; tests/unit/test_zeno_v1_g0_charts.py
replaces every later price with garbage and gets the same section byte for byte.
"""
from __future__ import annotations

import html
import itertools
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import propkit
from propkit import calendar
from propkit import indicators as ind
from propkit import zeno_report as zr
from propkit import zeno_v1 as zv

TITLE = "G0 signal check - RESEARCH ONLY - no results"
G0_CHARTS_FILE = "g0_charts.html"
PACKAGED_NEWS_CSV = (Path(zv.__file__).resolve().parent / "data" / "news_calendar"
                     / "us_macro_events_2015-01-01_2025-09-27.csv")      # the packaged US macro calendar (D20)
PAD_BARS = 8                       # bars drawn before the L window (the 20 bars before the H bar of a long)
H1_PANEL_BARS = 40                 # closed 1h bars in the trend panel

# Okabe-Ito colours (colour-blind safe); every element is also labelled in text
SIDE_COLOR = {1: "#0072B2", -1: "#D55E00"}          # long blue, short vermillion
SIDE_INK = {1: "#004f7c", -1: "#9c3a00"}            # darker shades for text on white
SIDE_WORD = {1: "LONG", -1: "SHORT"}
C_RETRACE = "#009E73"
C_VOID = "#CC79A7"
C_THRESHOLD = "#E69F00"
C_HL = "#3a3a3a"
C_STOP = "#000000"
C_EMA = "#5B2A86"
C_GRID = "#e4e4e4"
FONT_PX = 11.5
CHAR_PX = 6.6                      # rough width of one character at FONT_PX (labels are sized with it)
LANE_H = 21.0                      # px between two lanes of the event labels above the M15 chart
LABEL_END = 9.0                    # px: an event label's connector leaves its box at least this far from an end
PAD_PX = 22.0                      # px kept free above the highest and below the lowest price drawn: the marks
                                   # hang up to 19 px beyond a price (the arming tick), so they stay in the frame
GAP_ROW_H = 13.0                   # px per row of gap captions at the bottom of the M15 plot
RIGHT_LABEL_GAP = 22.0             # px between the level labels on the right (17 px boxes, 1.6 px borders)
AXIS_MIN_GAP = 10.0                # px between two time-axis labels on one line
# Advance widths (em) of Arial; Liberation Sans, its usual stand-in, has the same metrics. They size the
# time-axis labels and the gap captions, which must not touch each other or a mark.
_EM = {**dict.fromkeys("0123456789$#_?", 0.556), " ": 0.278, ".": 0.278, ",": 0.278, ":": 0.278, "/": 0.278,
       "-": 0.333, "(": 0.333, ")": 0.333, "%": 0.889, ">": 0.584, "<": 0.584, "=": 0.584, "+": 0.584,
       **dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", (0.667, 0.667, 0.722, 0.722, 0.667, 0.611, 0.778, 0.722, 0.278,
                                                 0.5, 0.667, 0.556, 0.833, 0.722, 0.778, 0.667, 0.778, 0.722,
                                                 0.667, 0.611, 0.722, 0.667, 0.944, 0.667, 0.667, 0.611))),
       **dict(zip("abcdefghijklmnopqrstuvwxyz", (0.556, 0.556, 0.5, 0.556, 0.556, 0.278, 0.556, 0.556, 0.222,
                                                 0.222, 0.5, 0.222, 0.833, 0.556, 0.556, 0.556, 0.556, 0.333,
                                                 0.5, 0.278, 0.556, 0.5, 0.722, 0.5, 0.5, 0.5)))}


# ---------------------------------------------------------------------------------------
# small helpers

def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _full(x: float) -> str:
    """A price or level for a data-* attribute: the shortest text that reads back as the same float."""
    return repr(float(x))


def _p(x: float, nd: int = 3) -> str:
    """A price, USD/oz, nd decimals ('n/a' when not finite)."""
    x = float(x)
    return f"{x:.{nd}f}" if math.isfinite(x) else "n/a"


def _utc(ts: int) -> str:
    return calendar.utc_str(int(ts))


def _sgt(ts: int) -> str:
    return str(zv.sgt_str(int(ts)))


def _short_utc(ts: int) -> str:
    """'YYYY-MM-DD HH:MM UTC'."""
    return _utc(ts)[:16] + " UTC"


def _short_sgt(ts: int) -> str:
    """'YYYY-MM-DD HH:MM SGT'."""
    return _sgt(ts)[:16] + " SGT"


def _both(ts: int) -> str:
    """'YYYY-MM-DD HH:MM UTC (HH:MM SGT)' of a bar open or an instant; the SGT date is added when it differs."""
    u, s = _utc(int(ts)), _sgt(int(ts))
    return f"{u[:16]} UTC ({s[11:16] if s[:10] == u[:10] else s[:16]} SGT)"


def _mmdd_hhmm(ts: int, hours: int) -> str:
    return str((np.int64(ts) + hours * 3600).astype("datetime64[s]")).replace("T", " ")[5:16]


def _minutes(seconds: int) -> str:
    """'2 d 3 h 15 min' text of a duration in seconds (sign dropped)."""
    m = abs(int(seconds)) // 60
    d, rest = divmod(m, 1440)
    h, mm = divmod(rest, 60)
    parts = ([f"{d} d"] if d else []) + ([f"{h} h"] if h or d else []) + [f"{mm} min"]
    return " ".join(parts)


def _yes(ok: bool) -> str:
    return '<span class="chip ok">yes</span>' if ok else '<span class="chip no">NO</span>'


def _verdict(ok: bool, text: str) -> str:
    """The code's verdict as HTML: a yes / NO chip and the words (escaped)."""
    return _yes(ok) + " " + _esc(text)


def _nice_step(span: float, n: int = 6) -> float:
    raw = max(span, 1e-9) / n
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if m * mag >= raw:
            return m * mag
    return 10.0 * mag


def _decimals(step: float) -> int:
    """Decimals that show every multiple of a tick step exactly (0 to 4)."""
    for d in range(5):
        if abs(round(step * 10 ** d) - step * 10 ** d) < 1e-6:
            return d
    return 4


def _text_w(s: str, size: float) -> float:
    """Width in px of s in regular Arial at `size` px (an unknown character counts as a wide one)."""
    return sum(_EM.get(ch, 0.667) for ch in s) * size


# ---------------------------------------------------------------------------------------
# the numbers of one sampled signal (all from the engine)

def trigger_rows(decisions: pd.DataFrame, sample: pd.DataFrame) -> list[int | None]:
    """For each row of a G0 sample, the index of its trigger in a decisions table of the same data (screen or
    simulate output, one row per prepare() event): the trigger at the row's signal_time_utc and side
    (zeno_report.g0_row_key, the reading g0_sample_check uses); None when there is none."""
    is_trig = (decisions["event"] == "trigger").to_numpy()
    where = {(int(t), str(sd)): int(i) for i, t, sd in zip(np.flatnonzero(is_trig),
                                                            decisions["time"].to_numpy()[is_trig],
                                                            decisions["side"].to_numpy()[is_trig])}
    out: list[int | None] = []
    for r in range(len(sample)):
        t, side, _, _ = zr.g0_row_key(sample.iloc[r])
        out.append(where.get((t, side)) if t is not None else None)
    return out


def annotate(prep: zv.Prepared, idx: int, chart: pd.DataFrame, decisions: pd.DataFrame,
             cell_label: str = "") -> dict[str, Any]:
    """Every number the chart and the checklist of one trigger show, from the engine's code path.

    prep: zeno_v1.prepare(frame, news); idx: the trigger's row in prep.events (= its row in screen() /
    simulate() decisions); chart: zeno_v1.chart_prices(prep) (the data's own prices, [SI-66]); decisions:
    zeno_v1.screen(prep, the declared cell) (eligible or blocked in that cell); cell_label: that cell's label.
    Reads the bars up to the trigger bar, the trigger close instant and the entry bar's open (bid, ask, time)
    only. Prices USD/oz, times UTC epoch seconds (bar opens unless a name says close)."""
    ev = prep.events[int(idx)]
    if ev["event"] != "trigger":
        raise ValueError(f"decisions row {idx} is a {ev['event']!r} event, not a trigger")
    i = int(ev["bar"])
    if i + 1 >= prep.n:
        raise ValueError(f"the trigger at bar {i} has no next bar: there is no entry to draw")
    s = int(ev["side"])
    f = prep.frame
    t = prep.time
    bo = f["bid_open"].to_numpy(dtype=np.float64)
    bh = f["bid_high"].to_numpy(dtype=np.float64)
    bl = f["bid_low"].to_numpy(dtype=np.float64)
    bc = f["bid_close"].to_numpy(dtype=np.float64)
    ext, arm, pl = int(ev["ext_bar"]), int(ev["arm_bar"]), int(ev["pl_bar"])
    w0 = ext - zv.L_LOOKBACK                                    # the 20 bars ending just before the extreme (D5)
    if s > 0:                                                   # L: the latest bar holding the window's lowest low
        far = w0 + int(ind.swing_low_index(bl[w0:ext], zv.L_LOOKBACK)[-1])
        h_bar, l_bar = ext, far
    else:                                                       # a short's H: the latest bar with the highest high
        far = w0 + int(ind.swing_high_index(bh[w0:ext], zv.H_LOOKBACK)[-1])
        h_bar, l_bar = far, ext
    retrace, void = float(ev["retrace_level"]), float(ev["void_level"])
    after = np.arange(ext + 1, arm + 1)                         # [SI-35]: the touch comes after the extreme
    touched = (bl[after] <= retrace + zv.LEVEL_TOL) if s > 0 else (bh[after] >= retrace - zv.LEVEL_TOL)
    touch = int(after[np.flatnonzero(touched)[0]]) if touched.any() else -1
    since = np.arange(ext + 1, i + 1)                           # closes since the extreme, up to the trigger
    k_worst = int(np.flatnonzero(s * bc[since] == np.min(s * bc[since]))[-1])
    worst_bar = int(since[k_worst])
    before = np.arange(pl + 1, i)                               # closes between the pullback bar and the trigger
    prior_bar = int(before[np.flatnonzero(s * bc[before] == np.max(s * bc[before]))[-1]]) if before.size else -1
    e = i + 1
    entry, stop, spread = (float(chart.at[idx, c]) for c in zv.CHART_PRICE_COLUMNS)
    dist = s * (entry - stop)                                    # D13's stop distance, USD/oz
    row = decisions.loc[idx]
    atr_t, med = float(row["atr_trigger"]), float(row["atr_median"])
    c_entry, c_stop, c_spread = float(row["entry_price"]), float(row["stop_level"]), float(row["spread_entry"])
    t_c = int(t[i]) + zv.M15_SECONDS
    # the trend panel: the last CLOSED 1h bar at the trigger close (D3) and the bars before it
    h1 = prep.h1
    j = int(zv.h1_index_at_m15_close(np.array([t[i]], dtype=np.int64), h1["close_time"].to_numpy())[0])
    lo1 = max(0, j - H1_PANEL_BARS + 1)
    h1_rows = []
    if j >= 0:
        for q in range(lo1, j + 1):
            h1_rows.append({"t": int(h1["time"].iat[q]), "close_t": int(h1["close_time"].iat[q]),
                            "o": float(h1["open"].iat[q]), "h": float(h1["high"].iat[q]),
                            "l": float(h1["low"].iat[q]), "c": float(h1["close"].iat[q]),
                            "ema": float(h1["ema"].iat[q])})
    prev = j - zv.EMA_SLOPE_BARS
    nearest = None
    if prep.news is not None and prep.news.times.size:
        T = prep.news.times
        k = int(np.searchsorted(T, t_c))
        cand = [q for q in (k - 1, k) if 0 <= q < T.size]
        q = min(cand, key=lambda x: (abs(int(T[x]) - t_c), x))
        nearest = {"name": str(prep.news.names[q]) if len(prep.news.names) == T.size else "event",
                   "kind": str(prep.news.kinds[q]) if len(prep.news.kinds) == T.size else "scheduled",
                   "T": int(T[q])}
    start = max(0, w0 - PAD_BARS)
    bars = [{"t": int(t[q]), "o": float(bo[q]), "h": float(bh[q]), "l": float(bl[q]), "c": float(bc[q])}
            for q in range(start, i + 1)]
    return {
        "side": s, "side_name": zv.SIDE_NAMES[s], "setup_id": str(ev["setup_id"]), "idx": int(idx),
        "start": start, "trigger_bar": i, "bars": bars,
        "ext_bar": ext, "h_bar": h_bar, "l_bar": l_bar, "arm_bar": arm, "touch_bar": touch, "pl_bar": pl,
        "h_window": (arm - zv.H_LOOKBACK + 1, arm), "l_window": (w0, ext - 1),
        "h": float(ev["h_level"]), "l": float(ev["l_level"]), "leg": float(ev["leg"]),
        "atr_arm": float(ev["atr_arm"]), "retrace": retrace, "void": void,
        "pl_level": float(ev["pl_level"]), "threshold": float(ev["trigger_level"]),
        "k": int(ev["bars_since_pullback"]),
        "worst_close": float(bc[worst_bar]), "worst_bar": worst_bar,
        "prior_close": float(bc[prior_bar]) if prior_bar >= 0 else float("nan"), "prior_bar": prior_bar,
        "trigger_close": float(bc[i]),
        "signal_time": t_c, "entry_time": int(t[e]), "entry": entry, "stop": stop, "spread": spread,
        "entry_bid_open": float(bo[e]), "entry_ask_open": float(f["ask_open"].iat[e]), "dist": dist,
        "atr": atr_t, "atr_median": med,
        "trend_ok": bool(prep.trend_long[i] if s > 0 else prep.trend_short[i]),
        "h1": h1_rows, "h1_last": j, "h1_prev": prev,
        "ema_now": float(h1["ema"].iat[j]) if j >= 0 else float("nan"),
        "ema_prev": float(h1["ema"].iat[prev]) if prev >= 0 else float("nan"),
        "h1_close": float(h1["close"].iat[j]) if j >= 0 else float("nan"),
        "h1_close_time": int(h1["close_time"].iat[j]) if j >= 0 else -1,
        "session_ok": bool(prep.session_ok[i]), "news_blocked": bool(prep.news_blocked[i]),
        "news_used": prep.news is not None, "news_nearest": nearest,
        "trading_day": int(prep.close_rank[i]) + 1,
        "cell": {"label": cell_label, "status": str(row["status"]), "reasons": str(row["reasons"]),
                 "entry": c_entry, "stop": c_stop, "spread": c_spread, "dist": s * (c_entry - c_stop)},
    }


# ---------------------------------------------------------------------------------------
# SVG pieces

def _text(x: float, y: float, s: str, anchor: str = "start", cls: str = "", size: float | None = None,
          weight: str = "", fill: str = "", extra: str = "") -> str:
    attrs = [f'x="{x:.1f}"', f'y="{y:.1f}"']
    if anchor != "start":
        attrs.append(f'text-anchor="{anchor}"')
    if cls:
        attrs.append(f'class="{cls}"')
    if size:
        attrs.append(f'font-size="{size:g}"')
    if weight:
        attrs.append(f'font-weight="{weight}"')
    if fill:
        attrs.append(f'fill="{fill}"')
    if extra:
        attrs.append(extra)
    return f"<text {' '.join(attrs)}>{_esc(s)}</text>"


def _label_box(x: float, y: float, s: str, color: str, anchor: str = "middle", ink: str = "#111111",
               weight: str = "600", group: str = "") -> str:
    """A text label on a white box with a coloured border; (x, y) = the anchor point of the text baseline
    (anchor "start": x is the box's left edge). group: attributes of a <g> around the box and its text."""
    w = _box_w(s)
    x0 = x - w / 2 if anchor == "middle" else (x if anchor == "start" else x - w)
    body = (f'<rect x="{x0:.1f}" y="{y - 12.5:.1f}" width="{w:.1f}" height="17" rx="3" fill="#ffffff" '
            f'stroke="{color}" stroke-width="1.6"/>'
            + _text(x0 + w / 2, y, s, "middle", size=FONT_PX, weight=weight, fill=ink))
    return f"<g {group}>{body}</g>" if group else body


def _box_w(s: str) -> float:
    """Width in px of the box _label_box draws around s."""
    return len(s) * CHAR_PX + 10


def _event_lanes(items: list[dict], x_lo: float, x_hi: float, combos_max: int = 4096) -> int:
    """Lanes (0 = top) and box positions of the event labels above the M15 chart. Each label (item["xm"] = the
    x of its mark, item["text"]) gets a box in one lane and a connector straight down from the box at xm to
    its mark. Placed so that no connector passes through another label's box (a label whose box spans another
    label's xm sits in a lane above it) and no two boxes of a lane overlap; among such placements, the fewest
    lanes, then the boxes nearest their marks' centres. Sets item["lane"] and item["x0"] (the box's left
    edge, inside [x_lo, x_hi]); returns the lanes used."""
    n = len(items)
    for d in items:
        d["w"] = _box_w(d["text"])
    if not n:
        return 1

    def candidates(d: dict) -> list[float]:
        w, xm = d["w"], d["xm"]
        out: list[float] = []
        for x0 in (xm - w / 2, xm - LABEL_END, xm + LABEL_END - w):       # centred, to the right, to the left
            x0 = min(max(x0, x_lo), x_hi - w)
            if x0 + 4.0 <= xm <= x0 + w - 4.0 and all(abs(x0 - y) > 0.05 for y in out):
                out.append(x0)
        return out or [min(max(xm - w / 2, x_lo), x_hi - w)]

    cand = [candidates(d) for d in items]

    def layers(idx: list[int], x0s: Mapping[int, float]) -> dict[int, int] | None:
        covers = {i: [j for j in idx if j != i and x0s[i] - 3.0 < items[j]["xm"] < x0s[i] + items[i]["w"] + 3.0]
                  for i in idx}
        preds: dict[int, list[int]] = {i: [] for i in idx}
        for i in idx:
            for j in covers[i]:
                preds[j].append(i)
        left = {i: len(preds[i]) for i in idx}
        ready = sorted((i for i in idx if not left[i]), key=lambda i: items[i]["xm"])
        lane: dict[int, int] = {}
        boxes: dict[int, list[tuple[float, float]]] = {}
        while ready:
            i = ready.pop(0)
            a, b = x0s[i], x0s[i] + items[i]["w"]
            q = max((lane[p] + 1 for p in preds[i]), default=0)
            while any(a < e + 6.0 and s < b + 6.0 for s, e in boxes.get(q, [])):
                q += 1
            lane[i] = q
            boxes.setdefault(q, []).append((a, b))
            for j in covers[i]:
                left[j] -= 1
                if not left[j]:
                    ready.append(j)
                    ready.sort(key=lambda k: items[k]["xm"])
        return lane if len(lane) == len(idx) else None            # None: two boxes span each other's mark

    # labels that can never meet (boxes and marks apart whatever the placement) are laid out separately
    reach = [(min(c), max(c) + d["w"]) for c, d in zip(cand, items)]
    groups: list[list[int]] = []
    for i in sorted(range(n), key=lambda i: reach[i][0]):
        if groups and reach[i][0] < max(reach[j][1] for j in groups[-1]) + 6.0:
            groups[-1].append(i)
        else:
            groups.append([i])
    n_lanes = 1
    for idx in groups:
        best = None
        n_combos = math.prod(len(cand[i]) for i in idx)
        options = itertools.product(*(cand[i] for i in idx)) if n_combos <= combos_max else \
            [tuple(cand[i][0] for i in idx)]
        for combo in options:
            x0s = dict(zip(idx, combo))
            lane = layers(idx, x0s)
            if lane is None:
                continue
            score = (max(lane.values()) + 1, sum(abs(x0s[i] + items[i]["w"] / 2 - items[i]["xm"]) for i in idx))
            if best is None or score < best[0]:
                best = (score, lane, x0s)
        if best is None:                     # no such placement (never seen): centred boxes, stacked lanes
            for q, i in enumerate(sorted(idx, key=lambda i: items[i]["xm"])):
                items[i]["lane"], items[i]["x0"] = q, cand[i][0]
            n_lanes = max(n_lanes, len(idx))
            continue
        _, lane, x0s = best
        for i in idx:
            items[i]["lane"], items[i]["x0"] = lane[i], x0s[i]
        n_lanes = max(n_lanes, max(lane.values()) + 1)
    return n_lanes


def _gap_rows(gaps: list[dict], x_lo: float, x_hi: float, max_rows: int = 3) -> int:
    """Rows (0 = the bottom one) and x of the captions of the data gaps at the bottom of the M15 plot. Each
    caption (gap["text"], gap["w"] px wide) sits just right of its gap's dashed line (gap["x"]), or just left
    of it, within [x_lo, x_hi], crossing no other gap line and overlapping no caption of its row. Sets
    gap["row"] (None when no row has room: the caption is written under the chart) and gap["x0"] (left edge);
    returns the rows used."""
    placed: list[dict] = []
    for g in sorted(gaps, key=lambda g: g["x"]):
        g["row"], g["x0"] = None, g["x"] + 3.0
        for row in range(max_rows):
            for x0 in (g["x"] + 3.0, g["x"] - 3.0 - g["w"]):
                x1 = x0 + g["w"]
                if x0 < x_lo or x1 > x_hi:
                    continue
                if any(o is not g and x0 - 3.0 < o["x"] < x1 + 3.0 for o in gaps):
                    continue
                if any(p["row"] == row and x0 < p["x0"] + p["w"] + 6.0 and p["x0"] < x1 + 6.0 for p in placed):
                    continue
                g["row"], g["x0"] = row, x0
                break
            if g["row"] is not None:
                placed.append(g)
                break
    return max((g["row"] + 1 for g in placed), default=0)


def _axis_labels(ticks: Sequence[tuple[float, int]], x_lo: float, x_hi: float, skip: bool = True,
                 sizes: tuple[float, float] = (11.0, 10.0)) -> list[tuple[float, int, str, str]] | None:
    """The labels of a time axis with two lines, SGT above and UTC below, at the ticks (x, bar open, UTC
    epoch s) in time order: (x, t, SGT text, UTC text) of each tick drawn. A date is printed on the SGT line
    when it changes; on the UTC line when it differs from the date a reader takes from above it (the SGT
    date when the SGT label shows one, else the UTC date of the label before), so no time is read on the
    wrong day; the first tick shows both dates. A tick whose label would leave [x_lo, x_hi] or come within
    AXIS_MIN_GAP px of the label before it on either line is left out (skip=False: None instead)."""
    out: list[tuple[float, int, str, str]] = []
    s_day = u_day = None
    end_s = end_u = -math.inf
    for x, t in ticks:
        sg, ug = _mmdd_hhmm(t, calendar.SGT_OFFSET_HOURS), _mmdd_hhmm(t, 0)
        s_dated = sg[:5] != s_day
        s_txt = sg if s_dated else sg[6:]
        u_txt = (ug if (not out or ug[:5] != (sg[:5] if s_dated else u_day)) else ug[6:]) + " UTC"
        hs, hu = _text_w(s_txt, sizes[0]) / 2, _text_w(u_txt, sizes[1]) / 2
        fits = x - max(hs, hu) >= x_lo and x + max(hs, hu) <= x_hi
        if not fits or x - hs < end_s + AXIS_MIN_GAP or x - hu < end_u + AXIS_MIN_GAP:
            if not skip:
                return None
            continue
        out.append((x, int(t), s_txt, u_txt))
        s_day, u_day, end_s, end_u = sg[:5], ug[:5], x + hs, x + hu
    return out


def _spread_out(items: list[dict], lo: float, hi: float, gap: float = RIGHT_LABEL_GAP) -> None:
    """Move the label positions items[k]["y"] apart (at least `gap` px, inside [lo, hi]), keeping their order."""
    items.sort(key=lambda d: d["y0"])
    for d in items:
        d["y"] = d["y0"]
    for k in range(1, len(items)):
        items[k]["y"] = max(items[k]["y"], items[k - 1]["y"] + gap)
    if items and items[-1]["y"] > hi:
        items[-1]["y"] = hi
        for k in range(len(items) - 2, -1, -1):
            items[k]["y"] = min(items[k]["y"], items[k + 1]["y"] - gap)
    if items and items[0]["y"] < lo:
        items[0]["y"] = lo
        for k in range(1, len(items)):
            items[k]["y"] = max(items[k]["y"], items[k - 1]["y"] + gap)


def m15_svg(a: Mapping[str, Any]) -> str:
    """The M15 bid candlestick chart of one annotated signal (annotate): from PAD_BARS bars before the L window
    to the trigger bar, then the entry marker at the next bar's open; nothing after it. Every element carries
    data-role (and data-t = a bar open, UTC epoch s; data-price = its level, USD/oz) for checks.
    Layout: the event labels sit in lanes above the plot, each with a dotted connector down to its mark, and
    no connector passes through another label (_event_lanes); the price scale leaves PAD_PX px above and
    below the prices so every mark stays inside the frame, and below that a strip holds the gap captions
    clear of every mark (_gap_rows); the level labels on the right are RIGHT_LABEL_GAP px apart; the time-axis
    labels never touch (_axis_labels). Connectors are drawn before every label box, so a box hides a line."""
    s = a["side"]
    bars = a["bars"]
    n = len(bars)
    start = a["start"]
    sc = SIDE_COLOR[s]
    W = 1180.0
    left, right_lab = 66.0, 176.0
    plot_l, plot_r = left, W - right_lab
    n_slots = n + 1                                              # the bars and the entry slot
    sw = (plot_r - plot_l) / (n_slots + 0.3)
    body = max(2.0, min(14.0, sw * 0.62))

    def xs(k: int) -> float:                                     # centre of slot k (k = bar index - start)
        return plot_l + (k + 0.5) * sw

    # event labels (top strip): one per slot, the texts of a slot joined
    ev: dict[int, list[tuple[str, str, float]]] = {}

    def add(bar: int, text: str, color: str, price: float) -> None:
        ev.setdefault(bar - start, []).append((text, color, price))

    if s > 0:
        add(a["h_bar"], "H", C_HL, a["h"])
        add(a["l_bar"], "L", C_HL, a["l"])
    else:
        add(a["l_bar"], "L", C_HL, a["l"])
        add(a["h_bar"], "H", C_HL, a["h"])
    wick = (lambda q: bars[q - start]["l"]) if s > 0 else (lambda q: bars[q - start]["h"])
    if a["touch_bar"] >= 0:
        add(a["touch_bar"], "50% touch", C_RETRACE, wick(a["touch_bar"]))
    add(a["arm_bar"], "armed", C_RETRACE, wick(a["arm_bar"]))
    add(a["pl_bar"], "pullback low" if s > 0 else "pullback high", C_THRESHOLD, a["pl_level"])
    add(a["trigger_bar"], f"TRIGGER close {'>' if s > 0 else '<'} {_p(a['threshold'])}", sc, a["trigger_close"])
    add(a["trigger_bar"] + 1, f"ENTRY {_p(a['entry'])} ({'ask' if s > 0 else 'bid'} open)", sc, a["entry"])
    items = []
    for k, lst in sorted(ev.items()):
        items.append({"xm": xs(k), "text": " / ".join(x[0] for x in lst), "color": lst[-1][1],
                      "price": lst[-1][2], "slot": k})
    n_lanes = _event_lanes(items, 4.0, W - 4.0)
    top = 10.0 + n_lanes * LANE_H + 8.0
    plot_h = 340.0
    plot_b = top + plot_h
    # gaps in the data (a weekend, the daily break, a hole): a dashed line between the two bars, captioned in
    # a strip at the bottom of the plot that no price, line or mark reaches
    times = [b["t"] for b in bars] + [a["entry_time"]]
    gaps = []
    for k in range(1, len(times)):
        if times[k] - times[k - 1] > zv.M15_SECONDS:
            text = f"gap {_minutes(times[k] - times[k - 1] - zv.M15_SECONDS)}"
            gaps.append({"x": plot_l + k * sw, "t": times[k], "t_before": times[k - 1], "text": text,
                         "w": _text_w(text, 11.0)})
    n_rows = _gap_rows(gaps, plot_l + 2.0, plot_r - 2.0)
    strip_h = n_rows * GAP_ROW_H + 2.0 if n_rows else 0.0
    y_hi, y_lo = top + PAD_PX, plot_b - strip_h - PAD_PX         # the highest and the lowest price drawn
    levels = [a["h"], a["l"], a["retrace"], a["void"], a["threshold"], a["entry"], a["stop"], a["pl_level"]]
    lo_p = min([b["l"] for b in bars] + levels)
    hi_p = max([b["h"] for b in bars] + levels)
    span = max(hi_p - lo_p, 1e-6)

    def ys(p: float) -> float:
        return y_lo - (p - lo_p) / span * (y_lo - y_hi)

    def price_at(y: float) -> float:
        return lo_p + (y_lo - y) / (y_lo - y_hi) * span

    out: list[str] = []
    out.append(f'<rect class="frame" x="{plot_l:.1f}" y="{top:.1f}" width="{plot_r - plot_l:.1f}" '
               f'height="{plot_h:.1f}" fill="#ffffff" stroke="#9a9a9a"/>')
    y_grid_lo = plot_b - strip_h - 1.0                           # no grid line in the caption strip
    p_bot, p_top = price_at(y_grid_lo), price_at(top + 1.0)
    step = _nice_step(p_top - p_bot, 7)
    nd = _decimals(step)
    for q in range(math.ceil(p_bot / step), math.floor(p_top / step) + 1):
        v = q * step
        y = ys(v)
        out.append(f'<line x1="{plot_l:.1f}" y1="{y:.1f}" x2="{plot_r:.1f}" y2="{y:.1f}" stroke="{C_GRID}"/>')
        out.append(_text(plot_l - 6, y + 4, f"{v:.{nd}f}", "end", "axis"))
    # the trigger window (bars 1..k after the pullback bar, D9) shaded
    k0, k1 = a["pl_bar"] + 1 - start, a["trigger_bar"] - start
    out.append(f'<rect data-role="window-trigger-shade" x="{xs(k0) - sw / 2:.1f}" y="{top:.1f}" '
               f'width="{(k1 - k0 + 1) * sw:.1f}" height="{plot_h:.1f}" fill="{sc}" fill-opacity="0.09"/>')
    for g in gaps:
        out.append(f'<line data-role="gap" data-t="{g["t"]}" x1="{g["x"]:.1f}" y1="{top:.1f}" x2="{g["x"]:.1f}" '
                   f'y2="{plot_b:.1f}" stroke="#7a7a7a" stroke-dasharray="3 3"/>')
        if g["row"] is not None:
            out.append(_text(g["x0"], plot_b - 4.0 - g["row"] * GAP_ROW_H, g["text"], cls="small gap-label"))
    # level lines (from the bar that defines them to the label column)
    xe = xs(n)                                                    # the entry slot
    right_items: list[dict] = []

    def level(role: str, price: float, x_from: float, x_to: float, color: str, dash: str, width: float,
              label: str, t_attr: int | None = None) -> None:
        y = ys(price)
        ta = f' data-t="{t_attr}"' if t_attr is not None else ""
        out.append(f'<line data-role="{role}" data-price="{_full(price)}"{ta} x1="{x_from:.1f}" y1="{y:.1f}" '
                   f'x2="{x_to:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="{width:g}"'
                   + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
        right_items.append({"y0": y, "text": label, "color": color, "yl": y})

    xh, xl = xs(a["h_bar"] - start), xs(a["l_bar"] - start)
    level("h", a["h"], xh, plot_r, C_HL, "", 1.2, f"H {_p(a['h'])}", int(times[a["h_bar"] - start]))
    level("l", a["l"], xl, plot_r, C_HL, "", 1.2, f"L {_p(a['l'])}", int(times[a["l_bar"] - start]))
    xx = xs(a["ext_bar"] - start)
    level("retrace", a["retrace"], xx, plot_r, C_RETRACE, "7 4", 2.0, f"50% {_p(a['retrace'])}")
    level("void", a["void"], xx, plot_r, C_VOID, "2 3", 2.2, f"78.6% {_p(a['void'])}")
    xp = xs(a["pl_bar"] - start)
    level("threshold", a["threshold"], xp, xs(k1) + sw / 2, C_THRESHOLD, "", 2.2,
          f"trigger level {_p(a['threshold'])}", int(times[a["pl_bar"] - start]))
    out.append(f'<line x1="{xs(k1) + sw / 2:.1f}" y1="{ys(a["threshold"]):.1f}" x2="{plot_r:.1f}" '
               f'y2="{ys(a["threshold"]):.1f}" stroke="{C_THRESHOLD}" stroke-width="1" stroke-dasharray="1 3"/>')
    level("stop", a["stop"], xe - sw * 0.5, plot_r, C_STOP, "8 4", 2.0, f"STOP {_p(a['stop'])}",
          a["entry_time"])
    level("entry-level", a["entry"], xe - sw * 0.5, plot_r, sc, "", 2.0, f"ENTRY {_p(a['entry'])}",
          a["entry_time"])
    # candles (bid OHLC): up = white body, down = dark body
    for k, b in enumerate(bars):
        x = xs(k)
        up = b["c"] >= b["o"]
        yo, yc = ys(b["o"]), ys(b["c"])
        y_top, y_bot = min(yo, yc), max(yo, yc)
        out.append(f'<g data-role="candle" data-t="{b["t"]}" data-o="{_full(b["o"])}" data-h="{_full(b["h"])}" '
                   f'data-l="{_full(b["l"])}" data-c="{_full(b["c"])}">'
                   f'<line x1="{x:.1f}" y1="{ys(b["h"]):.1f}" x2="{x:.1f}" y2="{ys(b["l"]):.1f}" stroke="#222222" '
                   'stroke-width="1.2"/>'
                   f'<rect x="{x - body / 2:.1f}" y="{y_top:.1f}" width="{body:.1f}" '
                   f'height="{max(y_bot - y_top, 1.2):.1f}" fill="{"#ffffff" if up else "#2b2b2b"}" '
                   'stroke="#222222" stroke-width="1.1"/></g>')
    # highlights: the pullback bar (orange box), the trigger bar (side-coloured box)
    for bar, role, color, wdt in ((a["pl_bar"], "pullback", C_THRESHOLD, 2.4),
                                  (a["trigger_bar"], "trigger", sc, 2.8)):
        b = bars[bar - start]
        x = xs(bar - start)
        y1, y2 = ys(b["h"]) - 5, ys(b["l"]) + 5
        price = a["pl_level"] if role == "pullback" else b["c"]
        out.append(f'<rect data-role="{role}" data-t="{b["t"]}" data-price="{_full(price)}" '
                   f'x="{x - body / 2 - 4:.1f}" y="{y1:.1f}" width="{body + 8:.1f}" height="{y2 - y1:.1f}" rx="3" '
                   f'fill="none" stroke="{color}" stroke-width="{wdt:g}"/>')
    # markers: H and L (triangles), the 50% touch (circle on the wick), the arming bar (tick), the entry (diamond)
    for role, bar, price, up in (("h-mark", a["h_bar"], a["h"], True), ("l-mark", a["l_bar"], a["l"], False)):
        x, y = xs(bar - start), ys(price)
        pts = (f"{x - 5:.1f},{y - 11:.1f} {x + 5:.1f},{y - 11:.1f} {x:.1f},{y - 3:.1f}" if up else
               f"{x - 5:.1f},{y + 11:.1f} {x + 5:.1f},{y + 11:.1f} {x:.1f},{y + 3:.1f}")
        out.append(f'<polygon data-role="{role}" data-t="{times[bar - start]}" data-price="{_full(price)}" '
                   f'points="{pts}" fill="{C_HL}"/>')
    if a["touch_bar"] >= 0:
        x, y = xs(a["touch_bar"] - start), ys(wick(a["touch_bar"]))
        out.append(f'<circle data-role="touch" data-t="{times[a["touch_bar"] - start]}" '
                   f'data-price="{_full(wick(a["touch_bar"]))}" cx="{x:.1f}" cy="{y:.1f}" r="5" fill="none" '
                   f'stroke="{C_RETRACE}" stroke-width="2.4"/>')
    xa = xs(a["arm_bar"] - start)
    ya = ys(wick(a["arm_bar"])) + (14 if s > 0 else -14)
    out.append(f'<line data-role="armed" data-t="{times[a["arm_bar"] - start]}" x1="{xa:.1f}" '
               f'y1="{ya - 5:.1f}" x2="{xa:.1f}" y2="{ya + 5:.1f}" stroke="{C_RETRACE}" stroke-width="3"/>')
    ye = ys(a["entry"])
    out.append(f'<polygon data-role="entry" data-t="{a["entry_time"]}" data-price="{_full(a["entry"])}" '
               f'points="{xe - 7:.1f},{ye:.1f} {xe:.1f},{ye - 7:.1f} {xe + 7:.1f},{ye:.1f} {xe:.1f},{ye + 7:.1f}" '
               f'fill="{sc}" stroke="#ffffff" stroke-width="1"/>')
    # level labels on the right, moved apart so that their boxes do not overlap
    _spread_out(right_items, top + 8, plot_b - 2, RIGHT_LABEL_GAP)
    for d in right_items:
        x0 = plot_r + 14
        out.append(f'<line x1="{plot_r:.1f}" y1="{d["yl"]:.1f}" x2="{x0:.1f}" y2="{d["y"] - 4:.1f}" '
                   f'stroke="{d["color"]}" stroke-width="1"/>')
        out.append(_label_box(x0, d["y"], d["text"], d["color"], "start", group='class="lvl-label"'))
    # event labels on top: every dotted connector first, then the boxes (a box hides any line behind it)
    for q, d in enumerate(items):
        yl = 10.0 + d["lane"] * LANE_H + 14.0
        out.append(f'<line class="ev-link" data-n="{q}" x1="{d["xm"]:.1f}" y1="{yl + 4.5:.1f}" x2="{d["xm"]:.1f}" '
                   f'y2="{ys(d["price"]):.1f}" stroke="{d["color"]}" stroke-width="1" stroke-dasharray="1 2"/>')
    for q, d in enumerate(items):
        yl = 10.0 + d["lane"] * LANE_H + 14.0
        out.append(_label_box(d["x0"], yl, d["text"], d["color"], "start", group=f'class="ev-label" data-n="{q}"'))
    # x axis: bar opens in SGT (UTC below), whole hours, labels apart (_axis_labels)
    y_ax = plot_b + 15
    every = max(1, int(math.ceil(78.0 / (4 * sw))))              # whole hours between labels
    ticks = [(xs(k), tt) for k, tt in enumerate(times) if not (tt % 3600 or (tt // 3600) % every)]
    for x, tt, s_txt, u_txt in _axis_labels(ticks, plot_l, W - 2.0) or []:
        out.append(f'<line x1="{x:.1f}" y1="{plot_b:.1f}" x2="{x:.1f}" y2="{plot_b + 4:.1f}" stroke="#666666"/>')
        out.append(_text(x, y_ax, s_txt, "middle", "axis xt", extra=f'data-tick="{tt}"'))
        out.append(_text(x, y_ax + 13, u_txt, "middle", "axis2 xt", extra=f'data-tick="{tt}"'))
    out.append(_text(plot_l - 6, y_ax, "SGT", "end", "axis"))
    out.append(_text(plot_l - 6, y_ax + 13, "UTC", "end", "axis2"))
    # the 20-bar windows and the trigger window as brackets
    yb = y_ax + 30
    hw, lw = a["h_window"], a["l_window"]
    if s > 0:
        br = [("window-h", hw, "H window: 20 bars ending at the arming bar, H = their highest high", C_HL),
              ("window-l", lw, "L window: the 20 bars before the H bar, L = their lowest low", C_HL)]
    else:
        br = [("window-l", hw, "L window: 20 bars ending at the arming bar, L = their lowest low", C_HL),
              ("window-h", lw, "H window: the 20 bars before the L bar, H = their highest high", C_HL)]
    br.append(("window-trigger", (a["pl_bar"] + 1, a["trigger_bar"]),
               f"trigger window: bars 1-8 after the pullback bar; the trigger is bar {a['k']} of 8", sc))
    for row_k, (role, (b0, b1), label, color) in enumerate(br):
        y = yb + row_k * 22
        x0, x1 = xs(b0 - start) - sw * 0.45, xs(b1 - start) + sw * 0.45
        out.append(f'<g data-role="{role}" data-t0="{times[b0 - start]}" data-t1="{times[b1 - start]}">'
                   f'<path d="M{x0:.1f},{y - 6:.1f} L{x0:.1f},{y:.1f} L{x1:.1f},{y:.1f} L{x1:.1f},{y - 6:.1f}" '
                   f'fill="none" stroke="{color}" stroke-width="2"/></g>')
        lw_px = len(label) * 6.1
        if x0 + lw_px <= W - 4:
            out.append(_text(x0, y + 14, label, cls="small", weight="600"))
        else:                                                     # too long to start here: end it at the bracket
            out.append(_text(max(x1, lw_px + 4), y + 14, label, "end", cls="small", weight="600"))
    height = yb + len(br) * 22 + 8
    # a gap whose caption found no room in the strip (dense holes in the data) is written under the chart
    for g in (g for g in gaps if g["row"] is None):
        between = (f"{_mmdd_hhmm(g['t_before'], calendar.SGT_OFFSET_HOURS)} and "
                   f"{_mmdd_hhmm(g['t'], calendar.SGT_OFFSET_HOURS)} SGT")
        out.append(_text(plot_l, height + 6, f"{g['text']} between the bars of {between} (the dashed line)",
                         cls="small gap-label"))
        height += 16
    return (f'<svg class="m15" viewBox="0 0 {W:.0f} {height:.0f}" width="{W:.0f}" height="{height:.0f}" '
            f'role="img" aria-label="M15 bid chart up to the entry">' + "".join(out) + "</svg>")


def h1_svg(a: Mapping[str, Any]) -> str:
    """The 1h trend panel of one annotated signal: the last H1_PANEL_BARS closed 1h bid bars at the trigger
    close (D3: a bar closing exactly then is usable, the next one is not), EMA30 (SMA seed, D2), the EMA value
    5 bars earlier and the last close marked."""
    rows = a["h1"]
    W, Hh = 540.0, 270.0
    if not rows:
        return (f'<svg class="h1" viewBox="0 0 {W:.0f} 60" width="{W:.0f}" height="60">'
                + _text(10, 30, "no closed 1h bar at the trigger close") + "</svg>")
    left, right, top, bot = 54.0, 178.0, 14.0, 222.0
    n = len(rows)
    sw = (W - left - right) / n
    vals = [r["h"] for r in rows] + [r["l"] for r in rows] + [r["ema"] for r in rows if math.isfinite(r["ema"])]
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-6)
    lo, hi = lo - 0.06 * span, hi + 0.06 * span

    def ys(p: float) -> float:
        return bot - (p - lo) / (hi - lo) * (bot - top)

    def xs(k: int) -> float:
        return left + (k + 0.5) * sw

    out = [f'<rect x="{left:.1f}" y="{top:.1f}" width="{W - left - right:.1f}" height="{bot - top:.1f}" '
           'fill="#ffffff" stroke="#9a9a9a"/>']
    step = _nice_step(hi - lo, 5)
    nd = _decimals(step)
    v = math.ceil(lo / step) * step
    while v <= hi:
        out.append(f'<line x1="{left:.1f}" y1="{ys(v):.1f}" x2="{W - right:.1f}" y2="{ys(v):.1f}" stroke="{C_GRID}"/>')
        out.append(_text(left - 5, ys(v) + 4, f"{v:.{nd}f}", "end", "axis"))
        v += step
    body = max(1.5, min(8.0, sw * 0.6))
    for k, r in enumerate(rows):
        x = xs(k)
        up = r["c"] >= r["o"]
        y1, y2 = sorted((ys(r["o"]), ys(r["c"])))
        out.append(f'<g data-role="h1-candle" data-t="{r["t"]}" data-close-t="{r["close_t"]}" '
                   f'data-c="{_full(r["c"])}" data-ema="{_full(r["ema"])}">'
                   f'<line x1="{x:.1f}" y1="{ys(r["h"]):.1f}" x2="{x:.1f}" y2="{ys(r["l"]):.1f}" stroke="#555555"/>'
                   f'<rect x="{x - body / 2:.1f}" y="{y1:.1f}" width="{body:.1f}" height="{max(y2 - y1, 1.0):.1f}" '
                   f'fill="{"#ffffff" if up else "#666666"}" stroke="#555555"/></g>')
    pts = [f"{xs(k):.1f},{ys(r['ema']):.1f}" for k, r in enumerate(rows) if math.isfinite(r["ema"])]
    if len(pts) >= 2:
        out.append(f'<polyline data-role="ema" points="{" ".join(pts)}" fill="none" stroke="{C_EMA}" '
                   'stroke-width="2.4"/>')
    last = n - 1
    kp = a["h1_prev"] - (a["h1_last"] - last)
    labels = []
    if 0 <= kp < n and math.isfinite(a["ema_prev"]):
        x, y = xs(kp), ys(a["ema_prev"])
        out.append(f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bot:.1f}" stroke="{C_EMA}" '
                   'stroke-dasharray="2 3"/>')
        out.append(f'<circle data-role="ema-5-earlier" data-t="{rows[kp]["t"]}" data-price="{_full(a["ema_prev"])}" '
                   f'cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#ffffff" stroke="{C_EMA}" stroke-width="2.4"/>')
        labels.append({"y0": y, "text": f"EMA30 5 bars ago {_p(a['ema_prev'])}", "color": C_EMA, "yl": y,
                       "x": x})
    if math.isfinite(a["ema_now"]):
        x, y = xs(last), ys(a["ema_now"])
        out.append(f'<circle data-role="ema-now" data-t="{rows[last]["t"]}" data-price="{_full(a["ema_now"])}" '
                   f'cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{C_EMA}"/>')
        labels.append({"y0": y, "text": f"EMA30 now {_p(a['ema_now'])}", "color": C_EMA, "yl": y, "x": x})
    x, y = xs(last), ys(a["h1_close"])
    out.append(f'<line data-role="h1-close" data-t="{rows[last]["t"]}" data-price="{_full(a["h1_close"])}" '
               f'x1="{x - 8:.1f}" y1="{y:.1f}" x2="{x + 8:.1f}" y2="{y:.1f}" stroke="#000000" stroke-width="2.5"/>')
    labels.append({"y0": y, "text": f"last close {_p(a['h1_close'])}", "color": "#000000", "yl": y, "x": x})
    _spread_out(labels, top + 6, bot, 15.0)
    for d in labels:
        out.append(f'<line x1="{d["x"]:.1f}" y1="{d["yl"]:.1f}" x2="{W - right + 8:.1f}" y2="{d["y"] - 4:.1f}" '
                   f'stroke="{d["color"]}" stroke-width="1" stroke-dasharray="1 2"/>')
        out.append(_text(W - right + 10, d["y"], d["text"], size=10.5, weight="600"))
    # x axis: every 8th bar back from the last one (wider apart when dated labels would come too close); a
    # date only where it changes or differs from the line above (_axis_labels, as the M15 axis)
    for every in (8, 10, 12, 16, 20, max(n, 1)):
        ticks = [(xs(k), rows[k]["t"]) for k in range(last, -1, -every)][::-1]
        axis = _axis_labels(ticks, 2.0, W - 2.0, skip=False)
        if axis is not None:
            break
    for x, tt, s_txt, u_txt in axis or []:
        out.append(f'<line x1="{x:.1f}" y1="{bot:.1f}" x2="{x:.1f}" y2="{bot + 4:.1f}" stroke="#666666"/>')
        out.append(_text(x, bot + 15, s_txt, "middle", "axis xt", extra=f'data-tick="{tt}"'))
        out.append(_text(x, bot + 28, u_txt, "middle", "axis2 xt", extra=f'data-tick="{tt}"'))
    out.append(_text(left, Hh - 8, "1h bid bars (open time; SGT, UTC below); EMA30 of the 1h closes, SMA seed",
                     cls="small"))
    return (f'<svg class="h1" viewBox="0 0 {W:.0f} {Hh:.0f}" width="{W:.0f}" height="{Hh:.0f}" role="img" '
            'aria-label="1h trend panel">' + "".join(out) + "</svg>")


# ---------------------------------------------------------------------------------------
# HTML

def trend_words(a: Mapping[str, Any]) -> str:
    """Rule 2 in words for one annotated signal: the last closed 1h close vs EMA30, EMA30 vs 5 bars earlier,
    and whether that agrees with the signal's side (the engine's trend flag at the trigger close)."""
    s = a["side"]
    c, e, ep = a["h1_close"], a["ema_now"], a["ema_prev"]
    if not (math.isfinite(c) and math.isfinite(e) and math.isfinite(ep)):
        return "no trend: EMA30 or its value 5 bars earlier is not defined yet (D2: the first 30 1h closes seed it)"
    above, rising = c > e, e > ep
    if above and rising:
        state = "UP: the last closed 1h bar closes ABOVE EMA30 and EMA30 is ABOVE its value 5 bars earlier"
    elif c < e and e < ep:
        state = "DOWN: the last closed 1h bar closes BELOW EMA30 and EMA30 is BELOW its value 5 bars earlier"
    else:
        state = ("MIXED: the close is " + ("above" if above else "below" if c < e else "at") + " EMA30 and EMA30 is "
                 + ("above" if rising else "below" if e < ep else "at") + " its value 5 bars earlier")
    agree = "agrees with" if a["trend_ok"] else "does NOT agree with"
    return f"{state}; {agree} a {a['side_name']} ({'needs UP' if s > 0 else 'needs DOWN'})."


def checklist_rows(a: Mapping[str, Any]) -> list[tuple[str, str, str, str]]:
    """The checklist of one annotated signal (annotate), every number with its unit: (rule, what the rule
    needs, the value as HTML, the code's verdict as HTML). The verdicts are the engine's flags (trend, session,
    news), the engine's own comparison on the engine's numbers (leg, levels, volatility), or, for the stop
    width and the spread (rules 8 and 10), the engine's reasons in the declared cell, whose R and spread are
    shown first (they decided eligibility), then the chart's."""
    s = a["side"]
    lt, gt = ("below", "above") if s > 0 else ("above", "below")
    ext_name, far_name = ("H", "L") if s > 0 else ("L", "H")
    ext_lvl, far_lvl = (a["h"], a["l"]) if s > 0 else (a["l"], a["h"])
    far_bar = a["l_bar"] if s > 0 else a["h_bar"]
    pl_word = "pullback-low" if s > 0 else "pullback-high"

    def bt(q: int) -> int:
        return a["bars"][q - a["start"]]["t"]

    rows: list[tuple[str, str, str, str]] = []

    def add(rule: str, what: str, value: str, verdict: str, value_is_html: bool = False) -> None:
        rows.append((rule, what, value if value_is_html else _esc(value), verdict))

    last_h1 = (f"last closed 1h bar {_both(a['h1'][-1]['t'])}, closed {_short_utc(a['h1_close_time'])}"
               if a["h1"] else "no closed 1h bar")
    add("2 (D3, D4)", "1h trend at the trigger close: the last closed 1h close vs EMA30, and EMA30 vs its value "
        "5 bars earlier",
        f"{last_h1}: close {_p(a['h1_close'])} vs EMA30 {_p(a['ema_now'])} USD/oz; EMA30 5 bars earlier "
        f"{_p(a['ema_prev'])} USD/oz", _verdict(a["trend_ok"], trend_words(a)))
    add("3 (D5)", f"{ext_name}: the {'highest high' if s > 0 else 'lowest low'} of the last 20 closed bars at "
        "the arming bar", f"{ext_name} = {_p(ext_lvl)} USD/oz, bar {_both(bt(a['ext_bar']))}",
        _verdict(True, "the engine's level and bar"))
    add("3 (D5)", f"{far_name}: the {'lowest low' if s > 0 else 'highest high'} of the 20 bars before the "
        f"{ext_name} bar", f"{far_name} = {_p(far_lvl)} USD/oz, bar {_both(bt(far_bar))} (the latest bar of the "
        "window at that price)", _verdict(True, "the engine's level"))
    ok_leg = a["leg"] >= zv.MIN_LEG_ATR * a["atr_arm"]
    add("3 (D7)", "leg H - L vs 1.5 x ATR14 at the arming bar",
        f"leg {_p(a['leg'])} USD/oz vs 1.5 x {_p(a['atr_arm'])} = {_p(zv.MIN_LEG_ATR * a['atr_arm'])} USD/oz",
        _verdict(ok_leg, f"leg {'>=' if ok_leg else '<'} 1.5 x ATR14"))
    tb = a["touch_bar"]
    wick = "low" if s > 0 else "high"
    touch_txt = (f"first reached after {ext_name} by the {wick} of the bar {_both(bt(tb))} "
                 f"({_p(a['bars'][tb - a['start']]['l' if s > 0 else 'h'])})" if tb >= 0 else "not reached")
    add("3 (D6, D8)", f"50% level = {ext_name} {'-' if s > 0 else '+'} 0.5 x leg, reached by a wick; the setup "
        "arms at the first close where D5-D8 hold",
        f"{_p(a['retrace'])} USD/oz; {touch_txt}; armed at the close of the bar {_both(bt(a['arm_bar']))}",
        _verdict(tb >= 0, "reached, armed"))
    void_ok = s * (a["worst_close"] - a["void"]) >= -zv.LEVEL_TOL
    add("3 (D6, D8)", f"78.6% level = {ext_name} {'-' if s > 0 else '+'} 0.786 x leg: no close {lt} it",
        f"{_p(a['void'])} USD/oz; the {'lowest' if s > 0 else 'highest'} close since {ext_name} is "
        f"{_p(a['worst_close'])} USD/oz (bar {_both(bt(a['worst_bar']))})", _verdict(void_ok, f"no close {lt} it"))
    add("4 (D9)", f"{pl_word} bar: the {'lowest low' if s > 0 else 'highest high'} since {ext_name} (the latest "
        "on a tie); its " + ("high" if s > 0 else "low") + " is the trigger level",
        f"bar {_both(bt(a['pl_bar']))}: {'low' if s > 0 else 'high'} {_p(a['pl_level'])} USD/oz, "
        f"{'high' if s > 0 else 'low'} {_p(a['threshold'])} USD/oz", _verdict(True, "the engine's bar"))
    first_ok = a["prior_bar"] < 0 or s * (a["prior_close"] - a["threshold"]) <= 0
    prior_txt = ("it is the first bar after the pullback bar" if a["prior_bar"] < 0 else
                 f"the {a['k'] - 1} close(s) before it stayed at or {lt} the level (the "
                 f"{'highest' if s > 0 else 'lowest'} {_p(a['prior_close'])} USD/oz)")
    add("4 (D9, D10)", f"trigger: the first close {gt} the trigger level, 1 to 8 bars after the {pl_word} bar",
        f"bar {_both(bt(a['trigger_bar']))}: close {_p(a['trigger_close'])} {'>' if s > 0 else '<'} "
        f"{_p(a['threshold'])} USD/oz, {a['k']} bar(s) after the {pl_word} bar; {prior_txt}",
        _verdict(1 <= a["k"] <= zv.TRIGGER_MAX_BARS and first_ok, f"bar {a['k']} of 8, the first close {gt}"))
    add("4 (D11)", f"entry: the next bar's open, at the {'ask' if s > 0 else 'bid'}",
        f"{_both(a['entry_time'])}: {_p(a['entry'])} USD/oz (bid open {_p(a['entry_bid_open'])}, ask open "
        f"{_p(a['entry_ask_open'])})", _verdict(True, "= entry_price in g0_sample.csv"))
    add("5, 8 (D12)", "ATR14 (Wilder, M15 bid) at the trigger close", f"{_p(a['atr'])} USD/oz",
        _verdict(math.isfinite(a["atr"]), "defined"))
    formula = (f"pullback low {_p(a['pl_level'])} - 0.25 x ATR14 {_p(a['atr'])}" if s > 0 else
               f"pullback high {_p(a['pl_level'])} + 0.25 x ATR14 {_p(a['atr'])} + entry spread {_p(a['spread'])}")
    add("5", "stop: " + ("pullback low - 0.25 x ATR14" if s > 0 else "pullback high + 0.25 x ATR14 + spread"),
        f"{formula} = {_p(a['stop'])} USD/oz ({'a bid' if s > 0 else 'an ask'} level)",
        _verdict(a["dist"] > 0, f"{'below' if s > 0 else 'above'} the entry; = stop_level in g0_sample.csv"))
    # Rules 8 (stop width) and 10 (spread) compare the stop distance R and the entry spread of the DECLARED
    # cell: screen() decided eligibility there, and its prices differ from the chart's (S1 x1.5 moves a long's
    # entry and a short's stop by 0.5 x the spread; S2 replaces the spread). So the cell's numbers come first,
    # the verdict is the engine's own (the cell's reasons), and the chart's numbers follow.
    c = a["cell"]
    reasons = {r for r in c["reasons"].split(";") if r}
    cell_name = (f"in the declared cell {c['label']}, which decided eligibility" if c["label"] else
                 "in the cell that decided eligibility")
    same = c["dist"] == a["dist"] and c["spread"] == a["spread"]
    r_ok = c["dist"] > 0                                        # R <= 0: the engine compares neither

    def vs(x: float, limit_txt: str, limit: float) -> str:
        return f"{_p(x)} USD/oz vs {limit_txt} = {_p(limit)} USD/oz"

    def cell_verdict(ok: bool, yes: str, no: str) -> str:
        if not r_ok:
            return _verdict(False, "not checked: the entry is at or beyond the stop in that cell")
        return _verdict(ok, f"{yes if ok else no} in that cell")

    atr3, lim_w = f"3 x {_p(a['atr'])}", zv.MAX_STOP_ATR * a["atr"]
    chart_w = ("; the chart's prices are the same" if same else
               f"; at the chart's prices (g0_sample.csv): {vs(a['dist'], atr3, lim_w)}")
    add("8 (D18)", "stop distance R = |entry - stop| vs 3 x ATR14",
        f"{vs(c['dist'], atr3, lim_w)} {cell_name}{chart_w}",
        cell_verdict("stop_wider_than_3_atr" not in reasons, "not wider", "wider"))
    med = a["atr_median"]
    v_ok = math.isfinite(med) and a["atr"] <= zv.VOL_CAP_X * med
    add("8 (D18)", "ATR14 vs 2 x its median over every M15 bar of the previous 20 trading days",
        f"{_p(a['atr'])} USD/oz vs 2 x {_p(med)} = {_p(zv.VOL_CAP_X * med)} USD/oz",
        _verdict(v_ok, "not above" if v_ok else "above, or no median"))
    frac = zv.MAX_SPREAD_FRAC_OF_R
    cell_s = vs(c["spread"], f"10% x {_p(c['dist'])}", frac * c["dist"])
    chart_s = ("; the chart's prices are the same" if same else "; at the chart's prices (g0_sample.csv): "
               + vs(a["spread"], f"10% x {_p(a['dist'])}", frac * a["dist"]))
    add("10 (D11)", "entry spread (the entry bar's ask open - bid open, at the cell's costs) vs 10% of the stop "
        "distance R", f"{cell_s} {cell_name}{chart_s}",
        cell_verdict("spread_gt_10pct_of_stop" not in reasons, "not above", "above"))
    add("9 (D19)", "entry time (the trigger close) in 15:00-18:00 or 20:30-24:00 SGT",
        f"{_short_sgt(a['signal_time'])} = {_short_utc(a['signal_time'])}",
        _verdict(a["session_ok"], "inside" if a["session_ok"] else "outside"))
    nn = a["news_nearest"]
    news_ok = not a["news_blocked"]
    what = "no entry from 30 min before to 60 min after NFP, CPI, PPI or FOMC (the nearest event)"
    if not a["news_used"]:
        add("9 (D20)", what, "no news calendar was read: no blackout", _verdict(news_ok, "not checked"))
    elif nn is None:
        add("9 (D20)", what, "the calendar has no NFP, CPI, PPI or FOMC event", _verdict(news_ok, "outside"))
    else:
        d = nn["T"] - a["signal_time"]
        rel = "at" if d == 0 else (f"{_minutes(d)} after" if d > 0 else f"{_minutes(d)} before")
        value = (f'<span class="news-event" data-news-t="{nn["T"]}">{_esc(nn["name"])} {_esc(_both(nn["T"]))}'
                 f'{" (unscheduled)" if nn["kind"].strip().lower() == "unscheduled" else ""}</span>: '
                 f"{_esc(rel)} the entry time; its window runs from 30 min before to 60 min after it")
        add("9 (D20)", what, value, _verdict(news_ok, "outside every window" if news_ok else "inside a window"),
            value_is_html=True)
    wd = a["trading_day"]
    add("D1", "warm-up: no entry in the first 30 trading days of the data",
        f"trading day {wd} of the data (server days with bars)",
        _verdict(wd > zv.WARMUP_TRADING_DAYS, "after the warm-up" if wd > zv.WARMUP_TRADING_DAYS else "warm-up"))
    return rows


def _sample_agreement(a: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    """Whether the g0_sample.csv row's H, L and bar times equal what is drawn (they come from the same stage)."""
    diffs = []

    def num(col: str, want: float) -> None:
        try:
            ok = abs(float(str(row.get(col, "")).strip()) - want) <= zr.G0_PRICE_TOL
        except ValueError:
            ok = False
        if col in row and not ok:
            diffs.append(col)

    def tim(col: str, want: int) -> None:
        if col in row and str(row.get(col, "")).strip() != _utc(want):
            diffs.append(col)

    num("h_level", a["h"])
    num("l_level", a["l"])
    tim("extreme_bar_time_utc", a["bars"][a["ext_bar"] - a["start"]]["t"])
    tim("pullback_bar_time_utc", a["bars"][a["pl_bar"] - a["start"]]["t"])
    tim("trigger_bar_time_utc", a["bars"][a["trigger_bar"] - a["start"]]["t"])
    tim("entry_time_utc", a["entry_time"])
    if not diffs:
        return ("g0_sample.csv agrees: its h_level, l_level, extreme, pullback, trigger and entry times are the "
                "ones drawn here.")
    return ("NOTE: this row of g0_sample.csv differs from the chart in " + ", ".join(diffs)
            + " (edited by hand?); the chart shows the engine's values.")


def section_html(a: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    """One sample row's section: title with the times and the TradingView hint, the M15 chart, the checklist,
    the 1h panel and an empty answer box (not a form; the answer goes in g0_sample.csv). Uses nothing but
    the annotation `a` (annotate) and the sample row's own text (sample_no, signal_no)."""
    s = a["side"]
    sample_no = str(row.get("sample_no", "")).strip()
    signal_no = str(row.get("signal_no", "")).strip()
    t_c = a["signal_time"]
    trig_t = a["bars"][-1]["t"]
    tv_go = _sgt(trig_t)[:16]
    head = (f'<div class="sec-head"><span class="badge" style="background:{SIDE_COLOR[s]}">{SIDE_WORD[s]}</span>'
            f'<h2>Row {_esc(sample_no)} of g0_sample.csv</h2>'
            f'<span class="meta">signal_no {_esc(signal_no)}, setup {_esc(a["setup_id"])}</span></div>'
            f'<p class="times">Signal (the trigger bar closes, entry time): <b>{_esc(_utc(t_c))}</b> | '
            f'<b>{_esc(_sgt(t_c))}</b> | <b>{_esc(str(zr.server_time_str(t_c)))}</b> (server = New York + 7 h). '
            f'Entry at the next bar\'s open: {_esc(_both(a["entry_time"]))}.</p>'
            f'<p class="tv">TradingView: symbol XAUUSD, 15 m, chart time zone (UTC+8) Singapore; press Alt+G and go '
            f'to <b>{_esc(tv_go)}</b> SGT, the trigger bar\'s open (the entry is the next bar). Another feed\'s '
            'prices differ a little from these Dukascopy bid bars: judge the shape and the order of events.</p>')
    c = a["cell"]
    status = (f'In the declared cell {_esc(c["label"])} the code says <b>{_esc(c["status"])}</b> '
              "(stage 1 leaves out the daily limits, the cooldown and the one-open-position rule, which need "
              "earlier trades [SI-54]).") if c["label"] else ""
    rows = checklist_rows(a)
    table = ['<table class="check"><thead><tr><th>rule</th><th>what the rule needs</th><th>value</th>'
             "<th>the code says</th></tr></thead><tbody>"]
    for rule, what, value, verdict in rows:
        table.append(f"<tr><td class=\"rule\">{_esc(rule)}</td><td>{_esc(what)}</td><td>{value}</td>"
                     f"<td>{verdict}</td></tr>")
    table.append("</tbody></table>")
    trend = (f'<p class="trend"><b>Trend (rule 2):</b> {_esc(trend_words(a))}</p>')
    answer = (f'<div class="answer"><span class="q">Your answer for row {_esc(sample_no)}:</span>'
              '<span class="box">y</span><span class="slash">/</span><span class="box">n</span>'
              f'<span class="hint">Write y or n in agree_y_n of row {_esc(sample_no)} in g0_sample.csv. '
              "This page is not a form and saves nothing.</span></div>")
    return (f'<section class="row {a["side_name"]}" id="row-{_esc(sample_no)}" data-sample-no="{_esc(sample_no)}" '
            f'data-side="{a["side_name"]}" data-signal-time="{t_c}" data-trigger-time="{trig_t}" '
            f'data-entry-time="{a["entry_time"]}">' + head
            + '<div class="chart">' + m15_svg(a) + "</div>"
            + '<p class="note">M15 bid candles up to the trigger bar (the last candle); the diamond is the entry at '
            "the next bar's open. Nothing after that open is drawn or written: no high, low or close of the entry "
            "bar or of any later bar.</p>"
            + '<div class="lower"><div class="left">' + "".join(table)
            + f'<p class="note">{_esc(_sample_agreement(a, row))} {status}</p></div>'
            + '<div class="right"><h3>1h trend (rule 2, D3)</h3>' + h1_svg(a) + trend + "</div></div>"
            + answer + "</section>")


CSS = """
:root { --ink:#111111; --muted:#4a4a4a; --line:#c9c9c9; --card:#ffffff; --page:#f3f3f1; --long:#0072B2;
        --short:#D55E00; --ok:#e3f1e8; --okink:#0b5a2a; --no:#fde0dc; --noink:#8a1c0b; }
* { box-sizing: border-box; }
html { background: var(--page); }
body { margin: 0; min-width: 1240px; background: var(--page); color: var(--ink);
       font-family: "Segoe UI", Arial, Helvetica, sans-serif; font-size: 15px; line-height: 1.45; }
.wrap { max-width: 1560px; margin: 0 auto; padding: 18px 24px 40px; }
h1 { font-size: 26px; margin: 4px 0 6px; }
h2 { font-size: 20px; margin: 0; }
h3 { font-size: 15px; margin: 0 0 4px; }
.card, section.row { background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 14px 18px;
       margin: 14px 0; }
section.row { border-left: 10px solid var(--long); }
section.row.short { border-left-color: var(--short); }
.sec-head { display: flex; align-items: center; gap: 12px; }
.badge { color: #ffffff; font-weight: 700; padding: 2px 10px; border-radius: 4px; letter-spacing: 0.5px; }
.meta { color: var(--muted); }
p { margin: 6px 0; }
.times b { white-space: nowrap; }
.tv { color: var(--muted); }
.chart { overflow-x: auto; margin: 6px 0 4px; }
svg { display: block; }
svg.m15 { width: 100%; height: auto; }
svg.h1 { width: 100%; height: auto; }
svg text { font-family: Arial, Helvetica, sans-serif; font-size: 11.5px; fill: #111111; }
svg text.axis { font-size: 11px; fill: #222222; }
svg text.axis2 { font-size: 10px; fill: #5a5a5a; }
svg text.small { font-size: 11px; fill: #333333; }
.lower { display: grid; grid-template-columns: minmax(0, 1fr) 540px; gap: 18px; align-items: start; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid var(--line); padding: 4px 7px; vertical-align: top; text-align: left; font-size: 13.5px; }
th { background: #ececea; }
td.rule { white-space: nowrap; font-weight: 600; }
.chip { display: inline-block; font-weight: 700; padding: 0 6px; border-radius: 3px; margin-right: 3px; }
.chip.ok { background: var(--ok); color: var(--okink); }
.chip.no { background: var(--no); color: var(--noink); }
.note { color: var(--muted); font-size: 13.5px; }
.trend { font-size: 14px; }
.answer { display: flex; align-items: center; gap: 12px; margin-top: 10px; padding: 10px 12px;
          border: 2px dashed #7a7a7a; border-radius: 6px; background: #fafafa; }
.answer .q { font-weight: 700; }
.answer .box { display: inline-block; width: 46px; height: 34px; border: 2px solid #111111; border-radius: 4px;
          text-align: center; line-height: 30px; font-weight: 700; color: #777777; background: #ffffff; }
.answer .hint { color: var(--muted); }
.files td, .files th { font-size: 13px; }
.files td.mono { word-break: break-all; }
.mono { font-family: Consolas, "Courier New", monospace; font-size: 12.5px; }
.legend { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 2px 24px; font-size: 14px; }
.sw { display: inline-block; width: 30px; height: 0; vertical-align: middle; margin-right: 8px; }
.toc td, .toc th { font-size: 13.5px; }
.warn { background: #fff4d6; border: 1px solid #c99a00; padding: 6px 10px; border-radius: 4px; }
"""


def _legend() -> str:
    items = [
        (f"border-top:2px solid {C_HL}", "H and L: the leg (a triangle marks each bar); H window and L window: the "
                                         "20-bar windows (brackets under the chart)"),
        (f"border-top:2px dashed {C_RETRACE}", "50% level: armed once a wick touches it after the extreme (circle "
                                                "= the first touch, tick = the arming bar)"),
        (f"border-top:3px dotted {C_VOID}", "78.6% level: a close beyond it would void the setup"),
        (f"border-top:3px solid {C_THRESHOLD}", "pullback bar (orange box) and its high (low for a short) = the "
                                                 "trigger level"),
        (f"border-top:3px solid {SIDE_COLOR[1]}", "LONG: trigger bar box, trigger window shade, entry diamond "
                                                   "(ask open of the next bar)"),
        (f"border-top:3px solid {SIDE_COLOR[-1]}", "SHORT: the same marks (entry = bid open of the next bar)"),
        (f"border-top:2px dashed {C_STOP}", "STOP: set with the entry (rule 5)"),
        ("border-top:2px dashed #7a7a7a", "gap: bars missing between two drawn bars (weekend, daily break, hole)"),
    ]
    cells = "".join(f'<div><span class="sw" style="{st}"></span>{_esc(txt)}</div>' for st, txt in items)
    return ('<div class="legend">' + cells + "</div><p class=\"note\">Candles are M15 BID bars (rule 1; shorts "
            "signal on bid bars too [SI-15]): white body = close above open, dark body = close below open. Every "
            "mark is also named in text on the chart and in the checklist.</p>")


def page_html(header: Mapping[str, Any], sections: Sequence[str], toc: Sequence[Mapping[str, Any]]) -> str:
    """The whole page: header (files, sha256, bar counts, the chart-price cell, how to answer, legend, a table
    of contents) and the sections. header keys: bid_file, ask_file, bid_sha256, ask_sha256, n_bars, sample_file,
    sample_sha256, n_rows, news_file, news_sha256, news_events, declared_cell, capital_usd, notes (list)."""
    h = header
    chart_cell = zv.CHART_CELL.label
    files = ['<table class="files"><thead><tr><th>file</th><th>path</th><th>sha256</th><th>rows</th></tr></thead>'
             "<tbody>"]
    for label, path, sha, nrow in (
            ("M15 bid bars", h["bid_file"], h["bid_sha256"], f"{h['n_bars']} bars"),
            ("M15 ask bars", h["ask_file"], h["ask_sha256"], f"{h['n_bars']} bars"),
            ("G0 sample", h["sample_file"], h["sample_sha256"], f"{h['n_rows']} rows"),
            ("news calendar", h["news_file"], h["news_sha256"], f"{h['news_events']} NFP/CPI/PPI/FOMC events")):
        files.append(f"<tr><td>{_esc(label)}</td><td class=\"mono\">{_esc(path)}</td><td class=\"mono\">"
                     f"{_esc(sha)}</td><td>{_esc(nrow)}</td></tr>")
    files.append("</tbody></table>")
    notes = "".join(f'<p class="warn">{_esc(x)}</p>' for x in h.get("notes") or [])
    toc_rows = "".join(
        f'<tr><td><a href="#row-{_esc(r["sample_no"])}">row {_esc(r["sample_no"])}</a></td><td>'
        f'{SIDE_WORD[r["side"]]}</td><td>{_esc(_short_sgt(r["signal_time"]))}</td>'
        f'<td>{_esc(_short_utc(r["signal_time"]))}</td><td>{_esc(_short_sgt(r["trigger_time"]))}</td></tr>'
        for r in toc)
    head = (
        f"<h1>{_esc(TITLE)}</h1>"
        '<div class="card"><p><b>What this is.</b> Each section draws one row of g0_sample.csv from the M15 bid and '
        "ask bars below, with every element of zeno_pullback_v1 marked, so you can judge whether the code's setup "
        "is your rule (spec: Gates, G0). It shows signals only: <b>each chart ends at the entry</b>. It draws the "
        "bars up to the trigger bar and the entry price at the next bar's open, and nothing after that open (no "
        "high, low or close of the entry bar or of any later bar, no exit and no profit or loss), so it shows no "
        "outcome and is not a result (spec: Change policy).</p>"
        "<p><b>How to answer.</b> For each row decide: is this setup your rule, element by element? Then write "
        "<b>y</b> or <b>n</b> in the <b>agree_y_n</b> column of g0_sample.csv on <b>every</b> row. This page is not "
        f"a form and saves nothing. G0 passes with at least {zr.G0_MIN_AGREE} y of {zr.G0_SAMPLE_SIZE}; only then "
        "run <span class=\"mono\">zeno-v1 run ... --g0-confirmed --g0-sample DIR/g0_sample.csv</span> (leave the "
        "file beside signals_report.json).</p>"
        f"<p><b>Chart prices.</b> The entry and stop drawn are the chart's, cell {_esc(chart_cell)} = the data's own "
        "prices at costs x1 [SI-66]: a long fills at the ask file's open of the next bar, a short at the bid file's "
        "open, and a short's stop includes the data's spread at that open; these are entry_price and stop_level "
        f"in g0_sample.csv. Which signals are eligible was decided in the declared cell {_esc(h['declared_cell'])} "
        f"({_esc(h['declared_from'])}); every row below is an eligible signal of these files in that cell with the "
        "same time, side, entry and stop [SI-69].</p>"
        f"<p><b>Times.</b> UTC, SGT (UTC+8) and server time (New York + 7 h). A bar is named by its open; the "
        "signal time is the trigger bar's close, which is also the entry time.</p>"
        + notes + "</div>"
        '<div class="card"><h3>Files</h3>' + "".join(files) + "</div>"
        '<div class="card"><h3>Marks</h3>' + _legend() + "</div>"
        '<div class="card"><h3>Rows</h3><table class="toc"><thead><tr><th>row</th><th>side</th>'
        "<th>signal (SGT)</th><th>signal (UTC)</th><th>trigger bar open (SGT)</th></tr></thead><tbody>"
        + toc_rows + "</tbody></table></div>")
    foot = (f'<p class="note">propkit {_esc(propkit.__version__)}, zeno-v1 g0-charts. RESEARCH ONLY - not trading '
            "advice. Nothing here places or prepares orders.</p>")
    return ("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<title>{_esc(TITLE)}</title><meta name=\"viewport\" content=\"width=1240\">"
            f"<style>{CSS}</style></head><body><div class=\"wrap\">" + head + "\n"
            + "\n".join(sections) + "\n" + foot + "</div></body></html>\n")


def g0_charts_html(prep: zv.Prepared, sample: pd.DataFrame, cell: zv.ZenoCell, capital: float,
                   header: Mapping[str, Any]) -> tuple[str, int]:
    """The G0 charts page of a sample that belongs to prep's data (check it first with
    zeno_report.g0_sample_check in the same cell): (ASCII HTML text, rows drawn). cell / capital: the
    declared stage-1 cell and account size (they decide eligibility; the chart prices are CHART_CELL's).
    header: see page_html. ValueError when a row has no trigger in this data."""
    dec = zv.screen(prep, zv.ZenoConfig(cell, capital))
    chart = zv.chart_prices(prep)
    idx = trigger_rows(dec, sample)
    sections, toc = [], []
    for r, i in enumerate(idx):
        row = {str(k): v for k, v in sample.iloc[r].items()}
        if i is None:
            raise ValueError(f"sample row {r + 1} is not a trigger of this data; run zeno_report.g0_sample_check first")
        a = annotate(prep, i, chart, dec, cell.label)
        sections.append(section_html(a, row))
        toc.append({"sample_no": str(row.get("sample_no", r + 1)).strip(), "side": a["side"],
                    "signal_time": a["signal_time"], "trigger_time": a["bars"][-1]["t"]})
    text = page_html(header, sections, toc)
    return text.encode("ascii", "xmlcharrefreplace").decode("ascii"), len(sections)
