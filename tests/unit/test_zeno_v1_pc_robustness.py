"""zeno-v1 on zeno's PC: the ways a G0 sample edited in Excel or Notepad, or a dukascopy-node download, could be
misread silently or refused with a message a first-time user cannot act on. Each test is a regression test of
one finding of the robustness audit (EXCEL-01..09, DATA-1/6/8/11, WINDOWS-3). Valid input reads as before.
Hand-made and synthetic bars only (NOT market data). Research only."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from propkit import cli
from propkit import zeno_g0_charts as g
from propkit import zeno_report as zr
from propkit import zeno_v1 as z
from tests.unit.zeno_v1_testkit import utc


def call(args) -> tuple[int, str, str]:
    """cli.main with stdout and stderr captured (both must be ASCII: zeno's console is cp936)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main([str(a) for a in args])
    assert out.getvalue().isascii() and err.getvalue().isascii()
    return code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------------------------------
# G0 samples as stage 1 writes them, and as Excel or Notepad may leave them

def stage1_lines(n: int = 20) -> list[str]:
    """The lines of a stage-1 g0_sample.csv (zeno_report.csv_bytes, G0_COLUMNS, agree_y_n empty: every row ends
    with ','), with placeholder cells: the header, then one line per row."""
    df = pd.DataFrame({c: [f"{c[:4]}{i}" for i in range(n)] for c in zr.G0_COLUMNS})
    df["sample_no"] = range(1, n + 1)
    df["agree_y_n"] = ""
    return zr.csv_bytes(df).decode("ascii").split("\n")[:-1]


def write(path: Path, lines: list[str], eol: str = "\n", encoding: str = "ascii", bom: bytes = b"") -> Path:
    path.write_bytes(bom + (eol.join(lines) + eol).encode(encoding))
    return path


def answered(answers: list[str]) -> list[str]:
    """answers typed in agree_y_n."""
    lines = stage1_lines(len(answers))
    return [lines[0]] + [r + a for r, a in zip(lines[1:], answers)]


def one_row_up(answers: list[str]) -> list[str]:
    """answers typed from the header row down: the first answer overwrites the header cell agree_y_n."""
    lines = stage1_lines(len(answers))
    return [lines[0].rsplit(",", 1)[0] + "," + answers[0]] + [r + a for r, a in zip(lines[1:], answers[1:] + [""])]


def next_column(answers: list[str]) -> list[str]:
    """answers typed in the column right of agree_y_n (no header there): Excel saves its used rectangle."""
    lines = stage1_lines(len(answers))
    return [lines[0] + ","] + [r + "," + a for r, a in zip(lines[1:], answers)]


Y17_N3 = ["y"] * 17 + ["n"] * 3


def test_excel01_answers_over_the_agree_y_n_header_or_beside_it_are_refused(tmp_path):
    # EXCEL-01: 17 y + 3 n typed one row up (the header cell agree_y_n becomes 'y') or one column right was
    # accepted with no answer counted, so stage 2 ran. The same answers in agree_y_n fail G0.
    with pytest.raises(cli.UsageError, match="G0 failed: .* 17 of 20"):
        cli._g0_record(str(write(tmp_path / "in_place.csv", answered(Y17_N3))))
    for name, lines, text in (
            ("one_row_up", one_row_up(Y17_N3), "the header cell agree_y_n is missing (the header row ends 'y')"),
            ("one_row_up_19y", one_row_up(["y"] * 19 + [""]), "the header cell agree_y_n is missing"),
            ("next_column", next_column(Y17_N3), "column T (no header), beside the sample's columns, holds 20 "
                                                 "filled cell(s) (the first on line 2: 'y')"),
            ("no_agree_column", [ln.rsplit(",", 1)[0] for ln in stage1_lines()], "header cell agree_y_n is missing"),
            ("named_note_column", [stage1_lines()[0] + ",note"] + [r + "y," + ("ok" if i == 4 else "")
                                                                   for i, r in enumerate(stage1_lines()[1:])],
             "column T (note), beside the sample's columns, holds 1 filled cell(s) (the first on line 6: 'ok')")):
        with pytest.raises(cli.UsageError) as e:
            cli._g0_record(str(write(tmp_path / f"{name}.csv", lines)))
        assert text in str(e.value), name
    # still accepted: Excel's trailing empty column (a ',' after agree_y_n on every line) and the stage-1 file
    rec = cli._g0_record(str(write(tmp_path / "empty_col.csv", [ln + "," for ln in answered(["y"] * 20)], "\r\n")))
    assert (rec["answered"], rec["agree_count"], rec["n_rows"]) == (20, 20, 20)
    rec = cli._g0_record(str(write(tmp_path / "stage1.csv", stage1_lines())))
    assert (rec["answered"], rec["unanswered"], rec["n_rows"]) == (0, 20, 20)


def test_excel02_a_rerun_of_stage_1_keeps_answers_typed_beside_agree_y_n(tmp_path):
    # EXCEL-02: stage 1 replaced a sample whose answers sat outside agree_y_n with a fresh, unanswered one.
    for name, lines, n in (("one_row_up", one_row_up(Y17_N3), 20), ("next_column", next_column(Y17_N3), 20),
                           ("in_place", answered(Y17_N3), 20), ("stage1", stage1_lines(), 0),
                           ("empty_col", [ln + "," for ln in stage1_lines()], 0)):
        d = tmp_path / name
        d.mkdir()
        sample = write(d / "g0_sample.csv", lines)
        assert cli._g0_answers_in(sample) == n, name
        if n:
            raw = sample.read_bytes()
            code, _, err = call(["zeno-v1", "signals", "--m15-bid", tmp_path / "bid.csv", "--m15-ask",
                                 tmp_path / "ask.csv", "--news", tmp_path / "news.csv", "--out", d])
            assert code == 2 and "refusing to replace" in err and f"it holds {n} answer(s)" in err, name
            assert sample.read_bytes() == raw


def test_excel03_windows3_a_sample_with_no_answer_on_disk_is_named(tmp_path, monkeypatch):
    # EXCEL-03 / WINDOWS-3: answers typed in Excel but not saved (or saved to .xlsx): stage 2 read the unanswered
    # file, ran, and said nothing. The run still goes on (an unanswered sample is only recorded), with a warning.
    def stop(*_a, **_k):
        raise cli.UsageError("stopped before the data (test)")
    monkeypatch.setattr(cli, "_zeno_data", stop)

    def run(sample: Path) -> tuple[int, str, str]:
        return call(["zeno-v1", "run", "--m15-bid", tmp_path / "bid.csv", "--m15-ask", tmp_path / "ask.csv",
                     "--news", tmp_path / "news.csv", "--out", tmp_path / "out", "--g0-confirmed", "--g0-sample",
                     sample])
    code, out, err = run(write(tmp_path / "g0_sample.csv", stage1_lines(), "\r\n", bom=b"\xef\xbb\xbf"))
    assert code == 2 and "stopped before the data" in err
    assert ("WARNING: g0_sample.csv holds no y/n answers in agree_y_n. propkit reads the file as last saved: if you "
            "answered it in Excel, save it as CSV (Ctrl+S, keep the CSV format), close it and run again.") in out
    code, out, _ = run(write(tmp_path / "answered.csv", answered(["y"] * 20)))
    assert code == 2 and "WARNING: answered.csv holds no" not in out
    # a partly saved sample (10 of 20 answered) is refused, and the refusal says the file is read as last saved
    with pytest.raises(cli.UsageError, match=r"10 of 20 rows are not answered y or n \(''\).*reads the file as last "
                                             r"saved: if you answered in Excel, save it as CSV \(Ctrl\+S"):
        cli._g0_record(str(write(tmp_path / "half.csv", answered(["y"] * 10 + [""] * 10))))


def test_excel05_a_workbook_utf16_or_code_page_save_is_refused_with_save_as_advice(tmp_path):
    # EXCEL-05: these were refused with a bare codec error ("'utf-8' codec can't decode byte 0xca ...").
    lines = answered(["y"] * 20)
    cp936 = answered(["\u662f"] * 20)                                  # 'shi' (yes) typed in Chinese, ANSI save
    note = lines[:-1] + [lines[-1] + " \u5bf9"]                        # 'y dui' in the last row only
    cases = (
        ("book.xlsx", b"PK\x03\x04\x14\x00\x08\x08" + b"\x00" * 64, "is an Excel workbook, not a CSV file"),
        ("book.xls", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64, "is an Excel workbook, not a CSV file"),
        ("utf16.csv", b"\xff\xfe" + ("\r\n".join(lines) + "\r\n").encode("utf-16-le"),
         "is saved as UTF-16 ('Unicode' / 'Unicode Text'), not as a CSV file"),
        ("cp936.csv", ("\r\n".join(cp936) + "\r\n").encode("cp936"),
         "line 2, column S (agree_y_n) holds text that is not UTF-8"),
        ("note.csv", ("\r\n".join(note) + "\r\n").encode("cp936"),
         "line 21, column S (agree_y_n) holds text that is not UTF-8"))
    for name, raw, text in cases:
        p = tmp_path / name
        p.write_bytes(raw)
        with pytest.raises(cli.UsageError) as e:
            cli._g0_record(str(p))
        msg = str(e.value)
        assert text in msg and "File > Save As > 'CSV UTF-8 (Comma delimited) (*.csv)'" in msg, name
        assert "codec" not in msg, name
    # g0-charts reads through the same reader (the refusal comes before any data is read)
    code, _, err = call(["zeno-v1", "g0-charts", "--m15-bid", tmp_path / "bid.csv", "--m15-ask", tmp_path / "ask.csv",
                         "--sample", tmp_path / "cp936.csv", "--out", tmp_path / "c.html"])
    assert code == 2 and "--sample cp936.csv: line 2, column S (agree_y_n) holds text that is not UTF-8" in err
    # an all-ASCII ANSI save is byte-identical to UTF-8 and still read
    p = write(tmp_path / "ascii_cp936.csv", lines, "\r\n", encoding="cp936")
    assert cli._g0_record(str(p))["agree_count"] == 20


def test_excel06_a_full_width_answer_is_named_as_such(tmp_path):
    # EXCEL-06: a full-width y (Chinese input in full-width mode) showed only as '\uff59'.
    for bom in (b"\xef\xbb\xbf", b""):
        p = write(tmp_path / f"fw{len(bom)}.csv", answered(["\uff59"] * 20), "\r\n", encoding="utf-8", bom=bom)
        with pytest.raises(cli.UsageError, match=r"20 of 20 rows are not answered y or n") as e:
            cli._g0_record(str(p))
        assert "is a full-width letter (Chinese input in full-width mode): switch to English input" in str(e.value)
    with pytest.raises(cli.UsageError) as e:                            # a plain odd answer gets no such hint
        cli._g0_record(str(write(tmp_path / "q.csv", answered(["y"] * 17 + ["?"] * 3))))
    assert "full-width" not in str(e.value) and "last saved" not in str(e.value)


def test_excel07_trailing_all_empty_rows_are_not_sample_rows(tmp_path):
    # EXCEL-07: Excel writes ',,,' rows when its used range reaches below the table; stage 2 counted 23 rows.
    lines = answered(["y"] * 20)
    p = write(tmp_path / "trail.csv", lines + ["," * 18] * 3, "\r\n")
    rec = cli._g0_record(str(p))
    assert (rec["n_rows"], rec["answered"], rec["agree_count"]) == (20, 20, 20)
    df = cli._g0_read_csv(p.read_bytes(), "x")
    assert len(df) == 20 and isinstance(df.index, pd.RangeIndex)
    p2 = write(tmp_path / "trail_col.csv", [ln + "," for ln in lines] + ["," * 19] * 2)  # and an empty column
    assert cli._g0_record(str(p2))["n_rows"] == 20
    # only trailing rows are dropped: an emptied row inside the table is still a row that is not answered
    mid = lines[:5] + ["," * 18] + lines[6:]
    with pytest.raises(cli.UsageError, match="1 of 20 rows are not answered"):
        cli._g0_record(str(write(tmp_path / "mid.csv", mid)))


def test_excel09_notepad_lines_with_one_field_more_are_refused_plainly(tmp_path):
    # EXCEL-09: ',y' appended in Notepad to lines that already end with ',' (the empty agree_y_n): on every
    # line pandas shifted the columns (sample_no became the index), on some lines it gave a tokenizer error.
    rows = stage1_lines()
    every = [rows[0]] + [r + ",y" for r in rows[1:]]
    some = [rows[0]] + [r + (",y" if i % 2 else "y") for i, r in enumerate(rows[1:])]
    for name, lines, text in (("every", every, "the lines below the header have 20 fields, more than the "
                                               "header's 19"),
                              ("some", some, "line 3 has 20 fields, more than the header's 19")):
        p = write(tmp_path / f"{name}.csv", lines, "\r\n")
        with pytest.raises(cli.UsageError) as e:
            cli._g0_record(str(p))
        assert text in str(e.value) and "(the line ends ',y', not ',,y')" in str(e.value), name
        assert "Error tokenizing" not in str(e.value)
    with pytest.raises(cli.UsageError, match="exists but cannot be read .* it may hold your G0 answers"):
        cli._g0_answers_in(tmp_path / "every.csv")


def test_a_valid_sample_reads_exactly_as_before(tmp_path):
    # the shared reader changes nothing for a stage-1 file, its Excel CSV UTF-8 save (BOM, CRLF) or blank lines
    for name, raw in (("lf", ("\n".join(answered(["y"] * 19 + ["n"])) + "\n").encode("ascii")),
                      ("bom_crlf", b"\xef\xbb\xbf" + ("\r\n".join(answered(["Yes"] * 20)) + "\r\n").encode("ascii")),
                      ("blank_lines", ("\n".join(answered(["y"] * 20)) + "\n\n\n").encode("ascii")),
                      ("unanswered", ("\n".join(stage1_lines()) + "\n").encode("ascii"))):
        got = cli._g0_read_csv(raw, name)
        want = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(got, want)


# ---------------------------------------------------------------------------------------------------
# end to end: a real stage-1 sample with Excel's trailing rows, through stage 2's check and g0-charts

@pytest.fixture(scope="module")
def stage1(tmp_path_factory):
    """~6 months of synthetic M15 bid/ask bars (NOT market data) and `zeno-v1 signals` on them."""
    d = tmp_path_factory.mktemp("zeno_pc")
    frame = z.synthetic_m15_bidask(n_bars=16_000, seed=3, spread=0.05)
    paths = []
    for side in ("bid", "ask"):
        df = frame[["time"] + [f"{side}_{c}" for c in ("open", "high", "low", "close")]].copy()
        df.columns = ["time", "open", "high", "low", "close"]
        paths.append(d / f"SYNTH_M15_{side}.csv")
        df.to_csv(paths[-1], index=False)
    sig = d / "sig"
    code, _, err = call(["zeno-v1", "signals", "--m15-bid", paths[0], "--m15-ask", paths[1], "--news",
                         g.PACKAGED_NEWS_CSV, "--out", sig])
    assert code == 0, err
    prep = z.prepare(z.load_m15_bidask(*paths), z.read_news_csv(g.PACKAGED_NEWS_CSV))
    return {"dir": d, "bid": paths[0], "ask": paths[1], "sig": sig, "prep": prep}


def test_excel07_trailing_rows_end_to_end(stage1, tmp_path):
    lines = (stage1["sig"] / "g0_sample.csv").read_text(encoding="ascii").split("\n")[:-1]
    width = lines[0].count(",")
    d = tmp_path / "s"
    d.mkdir()
    (d / "signals_report.json").write_bytes((stage1["sig"] / "signals_report.json").read_bytes())
    sample = write(d / "g0_sample.csv", [lines[0]] + [r + "y" for r in lines[1:]] + ["," * width] * 3, "\r\n")
    rec = cli._g0_record(str(sample))
    assert (rec["n_rows"], rec["agree_count"]) == (len(lines) - 1, len(lines) - 1)
    chk = cli._g0_sample_match(str(sample), stage1["prep"], 100_000.0)
    assert chk["ok"] and chk["n_rows"] == chk["n_matched"] == len(lines) - 1
    code, out, err = call(["zeno-v1", "g0-charts", "--m15-bid", stage1["bid"], "--m15-ask", stage1["ask"],
                           "--sample", sample])
    assert code == 0, err
    assert f"{len(lines) - 1} of {len(lines) - 1} rows of g0_sample.csv drawn" in out


def test_excel09_notepad_shift_end_to_end(stage1, tmp_path):
    lines = (stage1["sig"] / "g0_sample.csv").read_text(encoding="ascii").split("\n")[:-1]
    sample = write(tmp_path / "g0_sample.csv", [lines[0]] + [r + ",y" for r in lines[1:]], "\r\n")
    code, _, err = call(["zeno-v1", "g0-charts", "--m15-bid", stage1["bid"], "--m15-ask", stage1["ask"],
                         "--sample", sample, "--out", tmp_path / "c.html"])
    assert code == 2 and "the lines below the header have 20 fields, more than the header's 19" in err
    assert "does not belong to this data" not in err


# ---------------------------------------------------------------------------------------------------
# dukascopy-node downloads

def bars(start: int, n: int, step: int = 900, price: float = 2000.0) -> pd.DataFrame:
    """A dukascopy-node style table (timestamp in ms, open, high, low, close); price rises 0.1 per bar."""
    t = start + step * np.arange(n, dtype=np.int64)
    p = price + 0.1 * np.arange(n)
    return pd.DataFrame({"timestamp": t * 1000, "open": p, "high": p + 1.0, "low": p - 1.0, "close": p + 0.5})


def pair(folder: Path, bid: pd.DataFrame, ask: pd.DataFrame) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    bp, ap = folder / "bid.csv", folder / "ask.csv"
    bid.to_csv(bp, index=False)
    ask.to_csv(ap, index=False)
    return bp, ap


def plus(df: pd.DataFrame, spread) -> pd.DataFrame:
    out = df.copy()
    for c in ("open", "high", "low", "close"):
        out[c] = df[c] + spread
    return out


def test_data1_an_ask_file_holding_the_bid_prices_is_refused(tmp_path):
    # DATA-1: dukascopy-node -p defaults to bid, so an ask download without -p ask (or the bid path given twice)
    # was accepted with a zero spread, and stage 1 drew the G0 sample from the wrong eligible set.
    b = bars(utc("2024-03-04 00:00"), 40)
    bp, ap = pair(tmp_path / "copy", b, b)
    for args in ((bp, ap), (bp, bp)):
        with pytest.raises(ValueError, match=r"M15: the ask file .* holds the same prices as the bid file .* on every "
                                             r"bar \(spread 0\).*download the ask side with -p ask and give that "
                                             r"file to --m15-ask"):
            z.load_m15_bidask(*args)
    m1 = bars(utc("2024-03-04 00:00"), 30, step=60)
    m1p = pair(tmp_path / "m1", m1, m1)
    with pytest.raises(ValueError, match="--m1-ask"):
        z.load_m1_bidask(*m1p)
    code, _, err = call(["zeno-v1", "signals", "--m15-bid", bp, "--m15-ask", bp, "--news", g.PACKAGED_NEWS_CSV,
                         "--out", tmp_path / "out"])
    assert code == 2 and "holds the same prices as the bid file" in err
    # a zero spread on some bars is data, not a mistake
    sp = np.full(40, 0.2)
    sp[:39] = 0.0
    f = z.load_m15_bidask(*pair(tmp_path / "zero_some", b, plus(b, sp)))
    assert len(f) == 40 and float(f["spread_open"].max()) == pytest.approx(0.2)


def test_data6_a_zero_byte_download_is_named(tmp_path):
    # DATA-6: dukascopy-node creates its output file before the first batch; a stop then leaves 0 bytes, which
    # gave "EmptyDataError: No columns to parse from file".
    b = bars(utc("2024-03-04 00:00"), 40)
    bp, ap = pair(tmp_path, b, plus(b, 0.2))
    bp.write_bytes(b"")
    with pytest.raises(ValueError, match=r"M15 bid file bid.csv is empty \(0 bytes\): its download stopped before "
                                         r"any data was written. Run the dukascopy-node command for that side again"):
        z.load_m15_bidask(bp, ap)
    code, _, err = call(["zeno-v1", "signals", "--m15-bid", bp, "--m15-ask", ap, "--news", g.PACKAGED_NEWS_CSV,
                         "--out", tmp_path / "out"])
    assert code == 2 and "is empty (0 bytes)" in err and "EmptyDataError" not in err
    news = tmp_path / "news.csv"
    news.write_bytes(b"")
    with pytest.raises(ValueError, match=r"news calendar news.csv is empty \(0 bytes\)$"):
        z.read_news_csv(news)


def test_data8_swapped_files_are_named(tmp_path):
    # DATA-8: the bid file given to --m15-ask and the ask file to --m15-bid
    b = bars(utc("2024-03-04 00:00"), 40)
    a = plus(b, 0.3)
    with pytest.raises(ValueError, match=r"ask_open below bid_open .* 40 row\(s\) affected.*The files look swapped: "
                                         r"give the bid file to --m15-bid and the ask file to --m15-ask"):
        z.load_m15_bidask(*pair(tmp_path / "s", a, b))
    few = a.copy()
    few.loc[:2, "open"] = b.loc[:2, "open"] - 0.1                     # 3 bad bars: not a swap
    with pytest.raises(ValueError, match="ask_open below bid_open") as e:
        z.load_m15_bidask(*pair(tmp_path / "f", b, few))
    assert "swapped" not in str(e.value)


def test_data11_weekend_flat_filler_bars_are_refused(tmp_path):
    # DATA-11: dukascopy-node -fl keeps 0-volume candles, so every slot of the weekend holds a flat bar; they were
    # accepted and changed the triggers and the eligible set.
    fri = utc("2024-03-08 00:00")                                       # a Friday
    b = bars(fri, 96 * 3)                                               # Friday, Saturday, Sunday: every slot
    with pytest.raises(ValueError, match=r"M15: 96 bars open on a Saturday \(UTC\), and 1 Saturday\(s\) have a bar in "
                                         r"every 15-minute slot \(the first: 2024-03-09\).*dukascopy-node -fl"):
        z.load_m15_bidask(*pair(tmp_path / "flats", b, plus(b, 0.2)))
    # a stray Saturday bar does not fill the day: accepted
    t = np.concatenate([fri + 900 * np.arange(88), [utc("2024-03-09 00:00")],
                        utc("2024-03-10 22:00") + 900 * np.arange(20)])
    s = bars(0, len(t))
    s["timestamp"] = t * 1000
    assert len(z.load_m15_bidask(*pair(tmp_path / "stray", s, plus(s, 0.2)))) == len(t)
    # in-memory frames (the testkit's 24/7 scenarios) are not files: bidask_frame still takes them
    tb = b.rename(columns={"timestamp": "time"})
    tb["time"] //= 1000
    assert len(z.bidask_frame(tb, plus(tb, 0.2))) == 96 * 3
