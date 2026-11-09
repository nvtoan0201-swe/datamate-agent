"""Nạp gói dữ liệu một shop thành các DataFrame sạch.

Các bước cho mỗi sheet (hoặc file CSV, bảng SQLite):
1. Đọc lưới ô thô, không đoán header.
2. Dò dòng header trong 12 dòng đầu (bỏ dòng tiêu đề báo cáo, chọn tầng dưới của header 2 tầng).
3. Bỏ dòng trống, dòng lặp header, dòng tổng ("Tổng ngày 12/09"), ô lỗi Excel.
4. Điền xuống (fill-down) ô trống ở dòng tiếp theo của cùng đơn khi cột mã đơn để trống.
5. Ép kiểu từng cột: ngày, số (tiền "25.000đ", "25k", "1tr440"), số lượng kèm đơn vị, chuỗi.
6. Gộp các sheet / file cùng bộ cột (sheet theo tháng, file `_1`, `_2`, `_T1_2025`) thành một bảng.
7. Bỏ dòng trùng hoàn toàn trong cùng đơn nếu bảng ghi theo đơn (§8 rulebook), hai lần: trước và sau khi điền
   ô mức đơn / gộp cách viết (sau fill-down hai dòng trước đó khác nhau chỉ ở ô trống có thể thành giống hệt).
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from datamate.loaders.parse import (
    HHMM,
    date_order,
    looks_like_date,
    norm_hhmm,
    parse_date,
    parse_number,
    parse_qty_unit,
)
from datamate.text import clean, key, strip_accents

_ERR_CELLS = {"#VALUE!", "#N/A", "#REF!", "#DIV/0!", "#NAME?", "#NUM!", "#NULL!"}
_TOTAL_ROW = re.compile(r"^\s*(tổng|tong|TỔNG|TONG|cộng|cong)\b", re.IGNORECASE)
# Ô ghi dòng tổng ở bất kỳ cột nào: "Tổng ngày 01/01", "tổng 01/05", "Cộng tháng 3", "TỔNG CỘNG".
_TOTAL_CELL = re.compile(
    r"^\s*(tổng|tong|cộng|cong)\s*(cộng|cong|ngày|ngay|tháng|thang)?\s*(\d{1,2}\s*[/.-]\s*\d{1,2}([/.-]\d{2,4})?|\d{1,2})?\s*:?\s*$",
    re.IGNORECASE,
)
_ORDER_COL = re.compile(
    r"(mã|ma|số|so)[ _]*(đơn|don|hóa đơn|hoá đơn|hoa don|hđ|hd|bill|phiếu|phieu|chứng từ)"
    r"|^(hđ|hd|stt)$|order|invoice|^ma_?don|^so_?hd|madon|mahd|^mã bill$",
    re.IGNORECASE,
)
_DATE_NAME = re.compile(
    r"ngày|ngay|date|thời gian|thoi gian|giờ|gio|time|hạn|han|thời điểm", re.IGNORECASE
)
_YEAR_IN = re.compile(r"(20\d\d)")
UNIT_SUFFIX = " (đơn vị ghi kèm)"
# Cột bộ nạp thêm khi sổ không có mã nhóm dòng mà dòng sau của nhóm để trống ngày (§8): số thứ tự nhóm dòng.
GROUP_COL = "Nhóm dòng (bộ nạp)"


@dataclass
class Table:
    name: str
    df: pd.DataFrame
    sources: list[str]
    header_rows: int = 1
    deduped: int = 0  # tổng số dòng trùng đã bỏ (trước + sau fill-down)
    qty_unit_cols: list[str] = field(default_factory=list)
    deduped_before_fill: int = 0
    deduped_after_fill: int = 0
    dedupe_key: str | None = None  # cột mã đơn: chỉ bỏ dòng giống hệt nhau (mọi cột) có cùng mã đơn
    source_rows: int = 0  # tổng số dòng của các sheet / file trước khi gộp (kiểm tra mất bản ghi)
    notes: list[str] = field(
        default_factory=list
    )  # bộ nạp đã biến đổi gì (ghi vào thẻ schema cho LLM)


@dataclass
class Shop:
    biz_id: str
    path: Path
    readme: str
    tables: dict[str, Table]
    data_start: pd.Timestamp | None
    data_end: pd.Timestamp | None

    def frames(self) -> dict[str, pd.DataFrame]:
        return {name: t.df for name, t in self.tables.items()}

    @property
    def snapshot_columns(self) -> list[tuple[str, str]]:
        """(bảng, cột) mà README nói chỉ đúng tại ngày cuối dữ liệu (còn nợ, tồn cuối…)."""
        return snapshot_columns(self.readme, self.tables)


# ---------------------------------------------------------------- đọc lưới ô


def _read_grids(path: Path) -> list[tuple[str, pd.DataFrame]]:
    """Trả về [(nhãn nguồn, lưới ô thô)] cho một file."""
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xlsm", ".xls"):
        try:
            sheets = pd.read_excel(
                path, sheet_name=None, header=None, engine="calamine", dtype=object
            )
            return [(f"{path.name}:{sn}", g) for sn, g in sheets.items()]
        except Exception:  # noqa: BLE001, S110 — calamine lỗi thì đọc chậm bằng openpyxl
            pass
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        out = []
        for ws in wb.worksheets:
            rows = list(ws.iter_rows(values_only=True))
            out.append((f"{path.name}:{ws.title}", pd.DataFrame(rows, dtype=object)))
        wb.close()
        return out
    if suf in (".csv", ".tsv", ".txt"):
        raw = path.read_bytes()
        encs = ("utf-8-sig", "cp1258", "latin-1")
        if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
            encs = ("utf-16",)
        elif len(raw) > 4 and raw[:2000].count(b"\x00") > 0.3 * min(len(raw), 2000):
            encs = ("utf-16-le" if raw[1:2] == b"\x00" else "utf-16-be",) + encs
        for enc in encs:
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        sample = text[:20000]
        delim = max([",", ";", "\t", "|"], key=sample.count)
        import io

        g = pd.read_csv(
            io.StringIO(text),
            header=None,
            dtype=str,
            sep=delim,
            keep_default_na=False,
            skip_blank_lines=False,
            engine="python",
        )
        return [(path.name, g)]
    if suf == ".json":
        return _json_grids(path)
    if suf in (".sqlite", ".db", ".sqlite3"):
        out = []
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
            names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            for n in names:
                df = pd.read_sql_query(f'SELECT * FROM "{n}"', con)
                g = pd.concat(
                    [pd.DataFrame([list(df.columns)]), pd.DataFrame(df.values)], ignore_index=True
                )
                out.append((f"{path.name}:{n}", g.astype(object)))
        return out
    return []


def _json_grids(path: Path) -> list[tuple[str, pd.DataFrame]]:
    """File JSON: mảng bản ghi, hoặc object {tên bảng: mảng bản ghi}, hoặc {cột: mảng giá trị}. Ô lồng (object,
    mảng) ghi lại thành chuỗi JSON."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeDecodeError):
        return []

    def grid(records) -> pd.DataFrame | None:
        if (
            isinstance(records, dict)
            and records
            and all(isinstance(v, list) for v in records.values())
            and len({len(v) for v in records.values()}) == 1
            and not any(isinstance(x, dict) for v in records.values() for x in v)
        ):
            records = pd.DataFrame(dict(records)).to_dict("records")
        if (
            not isinstance(records, list)
            or not records
            or not all(isinstance(r, dict) for r in records)
        ):
            return None
        df = pd.DataFrame(records)
        df = df.map(
            lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
        )
        return pd.concat(
            [pd.DataFrame([list(df.columns)]), pd.DataFrame(df.values)], ignore_index=True
        ).astype(object)

    g = grid(data)
    if g is not None:
        return [(path.name, g)]
    out = []
    if isinstance(data, dict):
        for k, v in data.items():
            g = grid(v)
            if g is not None:
                out.append((f"{path.name}:{k}", g))
    return out


def _cell_empty(v: object) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str) and (not v.strip() or v.strip() in _ERR_CELLS):
        return True
    return v is pd.NaT


def _is_texty(v: object) -> bool:
    if _cell_empty(v) or not isinstance(v, str):
        return False
    s = v.strip()
    return parse_number(s) is None and not looks_like_date(s)


def _detect_header(g: pd.DataFrame) -> int:
    """Chỉ số dòng header: dòng nhiều ô chữ nhất trong 12 dòng đầu, dòng sau đó phải có dữ liệu."""
    best, best_score = 0, -1.0
    ncols = g.shape[1]
    for i in range(min(12, len(g))):
        row = g.iloc[i].tolist()
        texty = sum(_is_texty(v) for v in row)
        filled = sum(not _cell_empty(v) for v in row)
        if filled == 0 or texty < max(1, 0.5 * filled):
            continue
        # Phạt dòng tiêu đề báo cáo (1–2 ô) so với độ rộng bảng.
        score = texty + 0.5 * filled / max(ncols, 1)
        if score > best_score + 0.5:
            best, best_score = i, score
    return best


def _make_columns(g: pd.DataFrame, h: int) -> list[str]:
    cols, seen = [], Counter()
    for j, v in enumerate(g.iloc[h].tolist()):
        name = clean(str(v)) if not _cell_empty(v) else ""
        if not name and h > 0 and not _cell_empty(g.iat[h - 1, j]):
            name = clean(str(g.iat[h - 1, j]))
        if not name:
            name = f"col_{j}"
        seen[name] += 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        cols.append(name)
    return cols


# ---------------------------------------------------------------- ép kiểu cột


def _coerce_column(
    name: str, s: pd.Series, default_year: int | None
) -> tuple[pd.Series, pd.Series | None]:
    """Ép kiểu một cột. Trả về (cột đã ép, cột đơn vị tách từ số lượng hoặc None)."""
    vals = [v for v in s.tolist() if not _cell_empty(v)]
    if not vals:
        return s.map(lambda v: None if _cell_empty(v) else v), None
    sample = (
        vals
        if len(vals) <= 3000
        else [vals[i] for i in np.linspace(0, len(vals) - 1, 3000).astype(int)]
    )

    # Chuỗi số có số 0 đứng đầu (SĐT, mã) hoặc quá dài (IMEI) giữ là chuỗi.
    def codey(v):
        return isinstance(v, str) and re.fullmatch(r"0\d{3,}|\d{13,}", v.strip()) is not None

    if sum(codey(v) for v in sample) > 0.2 * len(sample):
        return s.map(lambda v: None if _cell_empty(v) else clean(str(v))), None

    # Giờ trong ngày ghi dạng chuỗi "09:15" / "14:05:30" / "8h30": giữ chuỗi "HH:MM" (không tách thành số lượng 9 +
    # đơn vị ":15").
    if sum(isinstance(v, str) and HHMM.match(v) is not None for v in sample) >= 0.95 * len(sample):
        return s.map(lambda v: None if _cell_empty(v) else (norm_hhmm(v) or clean(str(v)))), None

    order = date_order(vals)
    n_date = sum(looks_like_date(v) for v in sample)
    num_parsed = [parse_number(v) for v in sample]
    n_num = sum(x is not None for x in num_parsed)
    date_named = bool(_DATE_NAME.search(name))
    serial_like = n_num and all(
        x is not None and 36526 <= x <= 54789 for x in num_parsed if x is not None
    )
    n_parse = (
        sum(parse_date(v, default_year, order) is not None for v in sample)
        if date_named and n_date < 0.8 * len(sample)
        else 0
    )
    if (
        n_date >= 0.5 * len(sample)
        or (date_named and (n_date + (n_num if serial_like else 0)) >= 0.8 * len(sample))
        or (date_named and n_parse >= 0.95 * len(sample))
    ):
        parsed = s.map(lambda v: None if _cell_empty(v) else parse_date(v, default_year, order))
        if parsed.notna().sum() >= 0.8 * len(vals):
            return pd.to_datetime(parsed, errors="coerce"), None

    if n_num >= 0.95 * len(sample):
        return pd.to_numeric(
            s.map(lambda v: None if _cell_empty(v) else parse_number(v)), errors="coerce"
        ), None

    qty = [parse_qty_unit(v) for v in sample]
    n_qty = sum(q[0] is not None for q in qty)
    if n_qty >= 0.95 * len(sample) and any(q[1] for q in qty):
        pairs = s.map(lambda v: (None, None) if _cell_empty(v) else parse_qty_unit(v))
        num = pd.to_numeric(pairs.map(lambda p: p[0]), errors="coerce")
        unit = pairs.map(lambda p: p[1])
        return num, unit

    if all(isinstance(v, bool) for v in vals):
        return s.map(lambda v: None if _cell_empty(v) else bool(v)), None
    return s.map(
        lambda v: None if _cell_empty(v) else (clean(v) if isinstance(v, str) else v)
    ), None


def _date_cell(v: object) -> bool:
    return looks_like_date(v) or (
        isinstance(v, (int, float)) and not isinstance(v, bool) and 36526 <= v <= 54789
    )


def _group_anchor(body: pd.DataFrame, empty: pd.DataFrame) -> tuple[str | None, bool]:
    """Cột neo của nhóm dòng (§8) và cờ "neo là cột ngày".

    1. Cột mã đơn (nhận theo tên) để trống ở 2–95 % số dòng.
    2. Không có: cột ngày của sổ ghi theo thời gian, ghi ở dòng đầu nhóm, để trống ở dòng sau. Nhận theo hình dạng:
       ≥ 80 % ô có giá trị trông như ngày; trống 2–95 % số dòng; dòng đầu có ngày; các ngày đã ghi tăng dần theo
       dòng (≥ 95 % cặp liền nhau: sổ ghi theo thời gian; danh mục hạn dùng, ngày sinh thì không); và không cột ngày
       nào khác vẫn ghi ở các dòng để trống (có thì cột thưa là ngày của một sự kiện, như ngày trả, không phải neo).
       Ô ngày trống ở sổ như vậy nghĩa là "như dòng trên".
    Cột khác (khách, ghi chú) trống không đủ để nhận nhóm: trống có thể là khách lẻ, không ghi chú.
    """
    if not len(body):
        return None, False
    for c in body.columns:
        if _ORDER_COL.search(c) and 0.02 < empty[c].mean() < 0.95:
            return c, False
    if any(_ORDER_COL.search(c) for c in body.columns):
        return None, False
    datey = {}
    for c in body.columns:
        vals = body.loc[~empty[c].to_numpy(), c]
        if len(vals) and vals.map(_date_cell).mean() >= 0.8:
            datey[c] = vals
    best, best_n = None, -1
    for c, vals in datey.items():
        e = empty[c].to_numpy()
        if not 0.02 < e.mean() < 0.95 or e[0]:
            continue
        if any((~empty.loc[e, d]).mean() >= 0.5 for d in datey if d != c):
            continue
        ts = pd.to_datetime(vals.map(parse_date), errors="coerce").dropna()
        if len(ts) < 3 or (ts.diff().dropna() >= pd.Timedelta(0)).mean() < 0.95:
            continue
        line = sum(1 for d in body.columns if d != c and (~empty.loc[e, d]).mean() >= 0.9)
        if not line:
            continue
        co_blank = sum(
            1
            for d in body.columns
            if d != c and empty.loc[e, d].mean() >= 0.98 and not empty.loc[~e, d].all()
        )
        if co_blank > best_n:
            best, best_n = c, co_blank
    return best, best is not None


def _grid_to_frame(
    g: pd.DataFrame, default_year: int | None, label_tag: str = ""
) -> tuple[pd.DataFrame, int, list[str]]:
    h = _detect_header(g)
    cols = _make_columns(g, h)
    body = g.iloc[h + 1 :].copy()
    body.columns = cols
    # Bỏ cột không tên và hoàn toàn trống.
    keep = [c for c in cols if not (c.startswith("col_") and body[c].map(_cell_empty).all())]
    body = body[keep]
    empty = body.map(_cell_empty)
    body = body[~empty.all(axis=1)]
    empty = empty.loc[body.index]
    # Dòng lặp lại header.
    hdr = [clean(str(c)) for c in body.columns]
    is_hdr = body.apply(
        lambda r: (
            sum(isinstance(v, str) and clean(v) == hdr[i] for i, v in enumerate(r))
            >= max(2, len(hdr) // 2)
        ),
        axis=1,
    )
    body = body[~is_hdr]
    empty = empty.loc[body.index]

    # Dòng tổng: ô chữ đầu tiên bắt đầu bằng "Tổng"/"Cộng" và có nhiều ô trống.
    def total_row(r):
        texts = [v for v in r if isinstance(v, str) and v.strip()]
        if not texts:
            return False
        if _TOTAL_ROW.match(texts[0]) and len(texts) <= max(2, len(r) // 3):
            return True
        # Ô tổng nằm ở cột khác (khách, mã đơn): ô chỉ gồm "Tổng [ngày] dd/mm", không phải tên hàng.
        return any(
            _TOTAL_CELL.match(v) and re.search(r"\d|cộng|cong", v, re.IGNORECASE) for v in texts
        )

    body = body[~body.apply(total_row, axis=1)]
    empty = empty.loc[body.index]

    # Fill-down nhóm dòng (§8): ô neo của nhóm (cột mã đơn, hoặc cột ngày) trống ở dòng sau = dòng tiếp của cùng
    # nhóm, lấy ô trống từ dòng đầu nhóm — không lấy từ nhóm trước (ô đầu nhóm trống nghĩa là nhóm không có khoản
    # đó), và chỉ ở cột mức nhóm: cột hầu như không ghi ở dòng tiếp, hoặc ghi lại đúng giá trị dòng đầu. Cột mức
    # dòng (IMEI, số lượng) giữ trống.
    anchor, by_date = _group_anchor(body, empty)
    if anchor is not None:
        cont = empty[anchor].to_numpy()
        block = pd.Series((~cont).cumsum(), index=body.index)
        started = block.to_numpy() > 0
        heads = body[~cont]
        heads.index = block[~cont].to_numpy()
        head_vals = heads.reindex(block.to_numpy())
        head_vals.index = body.index
        head_empty = head_vals.map(_cell_empty)
        rows = cont & started
        filled = []
        for c in body.columns:
            written = rows & ~empty[c].to_numpy()
            if written.any():
                hv = head_vals.loc[written, c].map(lambda v: None if _cell_empty(v) else str(v))
                same = (hv == body.loc[written, c].map(str)).mean()
                if written.sum() > 0.02 * rows.sum() and same < 0.98:
                    continue  # cột mức dòng
            fill = rows & empty[c].to_numpy() & ~head_empty[c].to_numpy()
            if fill.any():
                body.loc[fill, c] = head_vals.loc[fill, c]
                filled.append(c)
        if by_date and set(filled) - {anchor}:
            # Sổ không có mã nhóm: ghi số thứ tự nhóm dòng để các cơ chế giải gom dòng về nhóm (đơn, phiếu).
            body[GROUP_COL] = np.where(started, block.to_numpy(), None)
            body[GROUP_COL] = body[GROUP_COL].map(
                lambda b: None if b is None else f"{label_tag}#{b}"
            )
        empty = body.map(_cell_empty)

    out = {}
    unit_cols = []
    for c in body.columns:
        col, unit = _coerce_column(c, body[c], default_year)
        out[c] = col.to_numpy() if not isinstance(col.dtype, pd.DatetimeTZDtype) else col
        if unit is not None:
            out[f"{c}{UNIT_SUFFIX}"] = unit.to_numpy()
            unit_cols.append(c)
    df = pd.DataFrame(out)
    for c in df.columns:
        if df[c].dtype == object:
            vals = df[c].dropna()
            if len(vals) and vals.map(lambda v: isinstance(v, dt.datetime)).all():
                df[c] = pd.to_datetime(df[c])
    return df.reset_index(drop=True), h + 1, unit_cols


# ---------------------------------------------------------------- gộp bảng


def _label_year(sheet: str, stem: str) -> int | None:
    """Năm ghi trong tên sheet / file (cho ngày không ghi năm, §8): "2025", hoặc tháng-năm 2 chữ số "T3-26",
    "03.26", "Thang 3_26"."""
    for t in (sheet, stem):
        m = _YEAR_IN.search(t)
        if m:
            return int(m.group(1))
    for t in (sheet, stem):
        m = re.search(r"(?:^|[^\d])(0?[1-9]|1[0-2])\s*[-/._ ]\s*(\d{2})$", t)
        if m:
            return 2000 + int(m.group(2))
    return None


# ---------------------------------------------------------------- nhiều khối trong một sheet, đơn vị ở tên cột, pivot


def _header_like(row: list) -> bool:
    """Mọi ô có giá trị là chữ (không số, không ngày), ít nhất 2 ô."""
    filled = [v for v in row if not _cell_empty(v)]
    return len(filled) >= 2 and all(_is_texty(v) for v in filled)


def _starts_table(g: pd.DataFrame, i: int) -> bool:
    """Dòng i là header của một bảng: giống header, và dưới nó có cột mà header ghi chữ còn dữ liệu là số / ngày."""
    row = g.iloc[i].tolist()
    if not _header_like(row):
        return False
    below = g.iloc[i + 1 : i + 4]
    return any(
        not _cell_empty(row[j])
        and below.iloc[:, j].map(lambda v: not _cell_empty(v) and not _is_texty(v)).any()
        for j in range(len(row))
    )


def _split_blocks(label: str, g: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Một sheet chứa nhiều bảng xếp chồng (bảng chi tiết, dòng trống, rồi bảng tổng hợp có header riêng) → tách khối.

    Tách tại dòng trống mà dòng ngay sau là header của một bảng mới (`_starts_table`) khác header khối đầu. Header lặp lại y hệt (sổ ghi nhiều tháng trong một
    sheet) không tách: bộ nạp bỏ dòng lặp header như cũ."""
    if len(g) < 6:
        return [(label, g)]
    empty = g.apply(lambda r: all(_cell_empty(v) for v in r), axis=1).to_numpy()
    h0 = _detect_header(g)
    head0 = [clean(str(v)).lower() for v in g.iloc[h0].tolist() if not _cell_empty(v)]
    cuts = []
    for i in range(h0 + 2, len(g) - 2):
        if not empty[i - 1] or empty[i]:
            continue
        row = g.iloc[i].tolist()
        if not _starts_table(g, i):
            continue
        if [clean(str(v)).lower() for v in row if not _cell_empty(v)] == head0:
            continue
        cuts.append(i)
    if not cuts:
        return [(label, g)]
    out, start = [], 0
    for n, c in enumerate([*cuts, len(g)]):
        block = g.iloc[start:c].reset_index(drop=True)
        out.append((label if n == 0 else f"{label}#{n + 1}", block))
        start = c
    return out


# Tên cột ghi đơn vị tiền: "Tiền (nghìn đ)", "Doanh thu (k)", "Giá (x1000)", "Tổng (triệu đồng)".
_UNIT_IN_NAME = [
    (
        re.compile(
            r"\((?:nghìn|ngàn|nghin|ngan)(?: ?(?:đồng|dong|đ|vnd))?\)|\((?:k|x ?1\.?000|1\.?000 ?đ?|000 ?đ)\)",
            re.IGNORECASE,
        ),
        1000,
    ),
    (re.compile(r"\((?:triệu|trieu|tr)(?: ?(?:đồng|dong|đ|vnd))?\)", re.IGNORECASE), 1_000_000),
]


def _scale_unit_columns(t: Table) -> None:
    """Cột tiền mà tên cột ghi rõ đơn vị nghìn / triệu đồng, giá trị cỡ đơn vị đó (trung vị < 100 000 với nghìn,
    < 10 000 với triệu) → quy ra đồng (đáp án tiền là đồng, §9). Ghi chú vào `t.notes` để thẻ schema nói cho LLM biết
    giá trị đã quy đổi, không nhân lần nữa."""
    for c in t.df.columns:
        s = t.df[c]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        for pat, f in _UNIT_IN_NAME:
            if pat.search(str(c)):
                nz = s[s != 0].abs().dropna()
                if len(nz) and nz.median() < (1e5 if f == 1000 else 1e4):
                    t.df[c] = s * f
                    t.notes.append(
                        f"cột `{c}`: tên cột ghi đơn vị, giá trị đã nhân {f} ra đồng; KHÔNG nhân lại"
                    )
                break


_MONTH_HEAD = re.compile(
    r"^(?:t|th|tháng|thang)?\s*(\d{1,2})\s*[/.-]\s*((?:19|20)\d\d)$", re.IGNORECASE
)


def _head_period(name: str) -> tuple[pd.Timestamp, str] | None:
    s = clean(str(name))
    m = _MONTH_HEAD.match(s)
    if m and 1 <= int(m.group(1)) <= 12:
        return pd.Timestamp(int(m.group(2)), int(m.group(1)), 1), "month"
    d = parse_date(s)
    return (pd.Timestamp(d), "day") if d is not None else None


def _unpivot(t: Table) -> Table | None:
    """Bảng dạng pivot (≥ 3 cột mà tên cột là ngày / tháng, giá trị là số) → thêm bảng dọc `<tên> (dọc)`: các cột còn
    lại giữ nguyên, thêm cột `Ngày` (hoặc `Tháng`) và `Giá trị`. Bảng gốc giữ nguyên."""
    heads = {c: _head_period(c) for c in t.df.columns}
    per = [c for c, h in heads.items() if h is not None and pd.api.types.is_numeric_dtype(t.df[c])]
    if len(per) < 3:
        return None
    kinds = {heads[c][1] for c in per}
    col = "Tháng" if kinds == {"month"} else "Ngày"
    ids = [c for c in t.df.columns if c not in per]
    long = t.df.melt(id_vars=ids, value_vars=per, var_name=col, value_name="Giá trị")
    long[col] = long[col].map(lambda c: heads[c][0])
    long = long[long["Giá trị"].notna()].reset_index(drop=True)
    note = (
        f"bảng dọc do bộ nạp tạo từ bảng pivot `{t.name}` ({len(per)} cột {col.lower()}): mỗi dòng một "
        f"({', '.join(ids) or 'dòng'}, {col.lower()}, giá trị); ô trống của bảng gốc đã bỏ"
    )
    return Table(name=f"{t.name} (dọc)", df=long, sources=list(t.sources), notes=[note])


def _family(file_name: str) -> str:
    """'DanhSachHoaDonChiTiet_1.xlsx' -> 'DanhSachHoaDonChiTiet'; 'BanHang_T1_2025' -> 'BanHang'."""
    stem = Path(file_name).stem
    stem = re.sub(r"([_ -]?(T|Thang|Tháng|thang)?\d{1,2})?[_ -]\d{4}$", "", stem)
    stem = re.sub(r"[_ -]\d{1,2}$", "", stem)
    return stem


def _less_marked(v: str, best: str) -> bool:
    """v là cách viết thiếu dấu / khác hoa thường / khác khoảng trắng của best.

    Mỗi ký tự của v phải trùng ký tự của best (không phân biệt hoa thường) hoặc là chữ không dấu của nó.
    'Thach ca phe' → 'Thạch cà phê' được gộp; 'Doãn Dũng Thanh' và 'Đoàn Dũng Thanh' là hai tên khác nhau.
    """
    a, b = clean(v).lower(), clean(best).lower()
    if len(a) != len(b):
        return False
    return all(x == y or (x.isascii() and strip_accents(y).lower() == x) for x, y in zip(a, b))


def _same_but_number_format(v: str, best: str) -> bool:
    """Cùng tên, cùng dãy số dài (SĐT), chỉ khác cách ghi số / dấu câu: 'Lan - 0909 123 456' ↔ 'Lan (0909123456)'."""
    da, db = re.sub(r"\D", "", v), re.sub(r"\D", "", best)
    if da != db or len(da) < 8:
        return False
    na, nb = (clean(re.sub(r"[\d\W_]+", " ", x)) for x in (v, best))
    return na.lower() == nb.lower() or _less_marked(na, nb)


def _spelling_key(v: str) -> str:
    """Khoá gộp cách viết: `key` + dãy số dài (SĐT, mã) ghi tách nhóm ("0327 988 481", "0908.995.064") nối liền."""
    k = key(v)
    return re.sub(
        r"\d+(?: \d+)+",
        lambda m: (
            m.group(0).replace(" ", "")
            if len(m.group(0)) - m.group(0).count(" ") >= 8
            else m.group(0)
        ),
        k,
    )


def _unify_spellings(df: pd.DataFrame) -> None:
    """Gộp cách viết thiếu dấu / khác hoa thường / khác khoảng trắng của cùng một tên về cách viết đủ dấu."""
    for c in df.columns:
        if not (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])):
            continue
        vals = df[c].dropna()
        if not len(vals) or not vals.map(lambda v: isinstance(v, str)).all():
            continue
        counts = vals.value_counts()
        if len(counts) > 20000:
            continue
        groups = defaultdict(list)
        for v, n in counts.items():
            groups[_spelling_key(v)].append(
                (_accent_score(v), not v.isupper(), n, v)
            )  # ưu tiên có dấu, không viết hoa hết, phổ biến
        mapping = {}
        for variants in groups.values():
            if len(variants) > 1:
                best = max(variants)[-1]
                for *_, v in variants:
                    if v != best and (_less_marked(v, best) or _same_but_number_format(v, best)):
                        mapping[v] = best
        if mapping:
            df[c] = df[c].replace(mapping)


def _fill_order_level(df: pd.DataFrame, key_col: str | None = None) -> None:
    """§8: ô trống ở dòng sau của cùng một đơn giống ô đó ở dòng đầu đơn.

    Chỉ áp cho cột mức đơn, nhận ra bằng dữ liệu: trong các đơn nhiều dòng mà cột không trống, giá trị cột
    không đổi trong ≥ 98 % số đơn (ngày, khách, trạng thái, kênh). Cột mức dòng (số lượng, đơn giá) không điền.
    """
    ocs = [c for c in df.columns if _ORDER_COL.search(c) and not re.fullmatch(r"(?i)stt", c)]
    if key_col is not None and (key_col not in df.columns or re.fullmatch(r"(?i)stt", key_col)):
        key_col = None  # STT thường là số thứ tự dòng trong phiếu, không phải mã nhóm
    oc = key_col if key_col is not None else (ocs[0] if ocs else None)
    if oc is None or not len(df):
        return
    key_ = df[oc]
    if key_.isna().mean() > 0.05:
        return
    sizes = key_.map(key_.value_counts())
    multi = sizes > 1
    if not multi.any():
        return
    is_first = ~key_.duplicated()
    for c in df.columns:
        if c == oc or not df[c].isna().any():
            continue
        sub = df.loc[multi & df[c].notna(), [oc, c]]
        if sub.empty:
            continue
        g = sub.groupby(oc)[c]
        full = g.size()
        full = full[full > 1]
        if len(full) < 20:
            continue
        nun = g.nunique().reindex(full.index)
        if (nun <= 1).mean() < 0.98:
            continue
        # Ô mức đơn ghi ở dòng đầu đơn (§8): nhóm có dòng đầu trống mà dòng sau lại ghi thì cột không phải mức đơn
        # (hoặc khoá không phải mã đơn, như mã ca gom nhiều đơn).
        head_filled = df.loc[multi & is_first, [oc, c]].set_index(oc)[c].notna()
        later_filled = df.loc[multi & ~is_first & df[c].notna(), oc].unique()
        if len(head_filled) and (~head_filled & head_filled.index.isin(later_filled)).mean() > 0.02:
            continue
        holes = df[c].isna() & multi & ~is_first
        if holes.any():
            first = df.loc[is_first & df[c].notna()].set_index(oc)[c]
            df.loc[holes, c] = df.loc[holes, oc].map(first)


def _order_key(df: pd.DataFrame) -> str | None:
    """Cột mã đơn nhận theo tên: khoá bỏ dòng trùng §8 ở bộ nạp (giữ hành vi cũ). None nếu không có."""
    ocs = [c for c in df.columns if _ORDER_COL.search(c)]
    named = [c for c in ocs if not re.fullmatch(r"(?i)stt", c)]
    return (named or ocs or [None])[0]


def _group_key(df: pd.DataFrame) -> str | None:
    """Khoá nhóm dòng để điền ô mức nhóm (§8): theo tên, không có thì theo hình dạng dữ liệu.

    Chỉ dùng để fill-down, KHÔNG để bỏ dòng trùng: bộ nạp không biết bảng là dòng hàng hay sổ sự kiện (các lần trả nợ,
    lần góp của cùng một đơn có thể giống hệt nhau mà đều là thật). Bỏ trùng theo hình dạng để graph làm theo vai trò
    (`semantic_reconcile.apply_grain_dedupe`, chỉ bảng dòng hàng / bảng đơn)."""
    return (
        _order_key(df) or _shape_order_key(df) or (GROUP_COL if GROUP_COL in df.columns else None)
    )


def _shape_order_key(df: pd.DataFrame, min_groups: int = 20) -> str | None:
    """Cột mã nhóm dòng nhận theo hình dạng, không theo tên: gần như không trống, không phải ngày / số đo thực;
    mỗi giá trị nằm liền một khối dòng; có ≥ `min_groups` nhóm nhiều dòng; một cột ngày không đổi trong ≥ 98 % các
    nhóm nhiều dòng (dòng của một đơn cùng ngày; cột khách, nhân viên trải nhiều ngày nên không qua). Nhiều cột qua
    thì lấy cột chia nhỏ nhất (nhiều nhóm nhất: mã đơn chứ không phải mã ca)."""
    if len(df) < 2 * min_groups:
        return None
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not dates:
        return None
    for c in df.columns:
        # Bảng đã có định danh duy nhất từng dòng (mã đơn ở bảng đơn): mỗi dòng một bản ghi, không gom theo khoá thô
        # hơn (mã ca, mã khách).
        s = df[c]
        if c not in dates and s.notna().mean() >= 0.99 and not pd.api.types.is_float_dtype(s):
            try:
                if s.nunique() >= 0.99 * len(df):
                    return None
            except TypeError:
                continue
    best, best_groups = None, 0
    for c in df.columns:
        s = df[c]
        if c in dates or s.isna().mean() > 0.05 or pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_float_dtype(s) and not (s.dropna() % 1 == 0).all():
            continue
        try:
            k = s.map(lambda v: None if _cell_empty(v) else str(v).strip())
        except TypeError:
            continue
        vc = k.value_counts()
        multi = vc[vc > 1]
        if len(multi) < min_groups or vc.mean() > 50 or len(vc) <= best_groups:
            continue
        runs = int((k.fillna("") != k.fillna("").shift()).sum())
        if runs > 1.02 * len(vc) + 1:
            continue
        in_multi = k.isin(multi.index)
        ok = False
        for d in dates:
            sub = pd.DataFrame({"k": k[in_multi], "d": df.loc[in_multi, d].dt.normalize()}).dropna()
            if len(sub) and (sub.groupby("k")["d"].nunique() <= 1).mean() >= 0.98:
                ok = True
                break
        if ok:
            best, best_groups = c, len(vc)
    return best


def _drop_order_duplicates(df: pd.DataFrame, key_col: str | None) -> tuple[pd.DataFrame, int]:
    """§8: hai dòng giống hệt nhau (mọi cột) trong cùng một đơn chỉ tính một lần.

    Bảng không có cột mã đơn (sổ sự kiện, danh mục) giữ nguyên: hai dòng giống nhau có thể là hai quan sát thật.
    Ở bảng ghi theo đơn, dòng không có mã đơn là dòng ghi chú / chân trang lặp lại theo sheet, bỏ trùng như cũ.
    """
    if key_col is None or not len(df):
        return df, 0
    try:
        dup = df.duplicated()
    except TypeError:  # ô kiểu không băm được (list…): so theo chuỗi
        dup = df.astype(str).duplicated()
    n = int(dup.sum())
    return (df[~dup].reset_index(drop=True), n) if n else (df, 0)


def _accent_score(s: str) -> int:
    return sum(1 for ch in s if ord(ch) > 127)


def load_shop(path: str | Path) -> Shop:
    path = Path(path)
    readme = (
        (path / "README.md").read_text(encoding="utf-8") if (path / "README.md").exists() else ""
    )
    pieces: dict[tuple, list] = defaultdict(list)
    for f in sorted(path.iterdir()):
        if f.name == "README.md" or f.name.startswith("."):
            continue
        for label, g in (blk for lg in _read_grids(f) for blk in _split_blocks(*lg)):
            if g.empty:
                continue
            sheet = label.split(":", 1)[1] if ":" in label else ""
            df, hrows, units = _grid_to_frame(g, _label_year(sheet, f.stem), label)
            if df.empty and not len(df.columns):
                continue
            fam = _family(f.name)
            base_cols = tuple(c for c in df.columns if not c.endswith(UNIT_SUFFIX))
            pieces[(fam, base_cols)].append((label, f.name, sheet, df, hrows, units))

    tables: dict[str, Table] = {}
    by_family = defaultdict(list)
    for (fam, cols), items in pieces.items():
        by_family[fam].append((cols, items))
    for fam, groups in by_family.items():
        for cols, items in groups:
            files = {it[1] for it in items}
            if len(groups) == 1:
                name = fam
            elif len(files) == 1 and len(items) == 1:
                sheet = items[0][2]
                name = f"{fam}.{sheet}" if sheet else Path(items[0][1]).stem
            elif max(len(g[1]) for g in groups) == len(items):
                name = fam  # bảng chính của họ file (gộp nhiều sheet theo tháng)
            else:
                name = f"{fam}.{items[0][2]}" if items[0][2] else Path(items[0][1]).stem
            frames = [it[3] for it in items]
            df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            df = df[
                [c for c in frames[0].columns]
                + [c for c in df.columns if c not in frames[0].columns]
            ]
            key_col = _order_key(df)
            df, before = _drop_order_duplicates(df, key_col)
            _fill_order_level(df, _group_key(df))
            _unify_spellings(df)
            df, after = _drop_order_duplicates(df, key_col)
            base = name
            k = 2
            while name in tables:
                name = f"{base}_{k}"
                k += 1
            tables[name] = Table(
                name=name,
                df=df,
                sources=[it[0] for it in items],
                header_rows=items[0][4],
                deduped=before + after,
                deduped_before_fill=before,
                deduped_after_fill=after,
                dedupe_key=key_col,
                source_rows=sum(len(f) for f in frames),
                qty_unit_cols=sorted({u for it in items for u in it[5]}),
            )

    for t in list(tables.values()):
        _scale_unit_columns(t)
        long = _unpivot(t)
        if long is not None and long.name not in tables:
            tables[long.name] = long

    conv = readme_conversions(readme)
    if conv is not None:
        tables["README_quy_doi"] = Table(name="README_quy_doi", df=conv, sources=["README.md"])

    data_end = None
    m = re.search(r"có đến hết (\d{1,2}/\d{1,2}/\d{4})", readme)
    if m:
        data_end = pd.Timestamp(dt.datetime.strptime(m.group(1), "%d/%m/%Y"))
    data_start = _guess_start(tables)
    return Shop(path.name, path, readme, tables, data_start, data_end)


_SNAPSHOT = re.compile(
    r"(đến hết|cuối|tại) ngày cuối (có )?dữ liệu|ngày cuối có dữ liệu|tại thời điểm xuất",
    re.IGNORECASE,
)
_BOLD = re.compile(r"\*\*(.+?)\*\*")


def snapshot_columns(readme: str, tables: dict[str, Table]) -> list[tuple[str, str]]:
    """Tìm câu README dạng "**X** là … tính đến hết ngày cuối dữ liệu" / "tồn cuối ngày cuối có dữ liệu".

    Cột X (tên in đậm trong đúng mệnh đề có cụm đó, khớp tên cột thật) là cột chụp tại ngày cuối dữ liệu.
    """
    out: list[tuple[str, str]] = []
    for line in readme.splitlines():
        body = re.sub(r"^\s*[-*]\s*\*\*[^*]+\*\*\s*:", "", line)  # bỏ nhãn đầu dòng "- **Nhãn**:"
        for clause in re.split(r";|\.\s", body):
            if not _SNAPSHOT.search(clause):
                continue
            for name in _BOLD.findall(clause):
                for tn, t in tables.items():
                    for c in t.df.columns:
                        if key(c) == key(name) and (tn, c) not in out:
                            out.append((tn, c))
    return out


def _guess_start(tables: dict[str, Table]) -> pd.Timestamp | None:
    """Ngày đầu dữ liệu: ngày nhỏ nhất của cột ngày trong bảng lớn nhất có cột ngày."""
    best = None
    for t in sorted(tables.values(), key=lambda t: -len(t.df)):
        dcols = [c for c in t.df.columns if pd.api.types.is_datetime64_any_dtype(t.df[c])]
        if dcols:
            best = min(t.df[c].min() for c in dcols)
            break
    return None if best is None or pd.isna(best) else best.normalize()


_CONV_LINE = re.compile(r"^\s*-\s*(.+?):\s*1\s+(.+?)\s*=\s*([\d.,]+)\s+(.+?)\s*$")


def readme_conversions(readme: str) -> pd.DataFrame | None:
    """Dòng 'Coca Cola 330ml: 1 thùng = 24 lon' trong README thành bảng quy đổi."""
    rows = []
    for line in readme.splitlines():
        m = _CONV_LINE.match(line)
        if m:
            f = parse_number(m.group(3))
            if f is not None:
                rows.append((clean(m.group(1)), clean(m.group(2)), f, clean(m.group(4))))
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["ten_hang", "don_vi_lon", "he_so", "don_vi_co_ban"])


# ---------------------------------------------------------------- cache


# Mã nguồn bộ nạp đọc lúc import (khoá cache): đọc lúc gọi thì một tiến trình đang chạy code cũ ghi kết quả cũ dưới khoá
# của code mới trên đĩa, mọi lần chạy sau đọc nhầm.
_LOADER_SOURCE = Path(__file__).read_bytes() + Path(__file__).with_name("parse.py").read_bytes()


def load_shop_cached(path: str | Path, cache_dir: str | Path | None = None) -> Shop:
    """Nạp shop, lưu cache pickle theo mtime các file để lần sau đọc nhanh."""
    import hashlib
    import pickle

    path = Path(path)
    if cache_dir is None:
        return load_shop(path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = json.dumps(
        sorted((f.name, f.stat().st_size, int(f.stat().st_mtime)) for f in path.iterdir()),
    )
    h = hashlib.sha1((sig + str(path.resolve())).encode() + _LOADER_SOURCE).hexdigest()[:16]
    cp = cache_dir / f"{path.name}_{h}.pkl"
    if cp.exists():
        with cp.open("rb") as fh:
            return pickle.load(fh)
    shop = load_shop(path)
    with cp.open("wb") as fh:
        pickle.dump(shop, fh)
    return shop
