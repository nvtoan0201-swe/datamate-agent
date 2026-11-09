"""Giả lập shop mới từ shop train bằng đột biến giữ nguyên nghĩa (generalization-plan §1.3).

Đáp án chuẩn của shop gốc không đổi sau đột biến, nên vẫn chấm được bằng `answers.jsonl`.

| Đột biến | Làm gì |
|---|---|
| rename | đổi tên cột sang từ đồng nghĩa, sửa README tương ứng |
| status | đổi cách ghi từng giá trị trạng thái sang cách ghi khác cùng nghĩa |
| layout | gộp sheet theo tháng thành một sheet, hoặc tách bảng một sheet thành file theo tháng; thêm dòng tiêu đề |
| format | tiền "25000" → "25.000đ" / "25k"; ngày → "1 thg 6 2025" |
| walkin | "Khách lẻ" → "KVL" hoặc ô trống |

`mutation.json` trong thư mục đột biến ghi lại ánh xạ tên cột, giá trị trạng thái, khách lẻ (cũ → mới).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import random
import re
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from datamate.biz import Biz
from datamate.loaders.parse import looks_like_date, parse_date, parse_number
from datamate.loaders.shop import _ORDER_COL, _cell_empty, _detect_header, _read_grids
from datamate.text import clean, key

KINDS = ("rename", "status", "layout", "format", "walkin")

# Từ đồng nghĩa cho tên cột (khoá đã bỏ dấu → các tên mới). Chỉ đổi cột có trong bảng này.
SYNONYMS: dict[str, list[str]] = {
    "thanh tien": ["Tiền hàng", "Thành tiền dòng", "Tiền"],
    "ma hoa don": ["Số HĐ", "Mã HĐ", "Số hoá đơn"],
    "ma don hang": ["Số đơn", "Mã ĐH", "Order no"],
    "ma don": ["Số đơn", "Mã ĐH"],
    "so luong": ["SL", "Số lg", "Qty"],
    "ten hang": ["Mặt hàng", "Tên SP", "Hàng hoá"],
    "ten san pham": ["Sản phẩm", "Tên mặt hàng"],
    "ten hang hoa": ["Mặt hàng", "Tên hàng"],
    "khach hang": ["Người mua", "Tên khách", "KH"],
    "ten khach hang": ["Người mua", "Khách"],
    "trang thai": ["Tình trạng", "TT đơn", "Trạng thái đơn hàng"],
    "trang thai don hang": ["Tình trạng đơn", "Trạng thái"],
    "status": ["Tình trạng", "Trạng thái"],
    "ket qua": ["Tình trạng", "KQ"],
    "ngay": ["Ngày bán", "Ngày lập"],
    "ngay thang": ["Ngày bán", "Ngày"],
    "ngay dat hang": ["Ngày đặt", "Ngày tạo đơn"],
    "ngay giao dich": ["Ngày bán", "Ngày"],
    "thoi gian": ["Ngày giờ", "Thời điểm"],
    "don gia": ["Giá", "Giá đơn vị"],
    "tri gia": ["Tiền hàng", "Thành tiền"],
    "doi tac": ["Khách", "Người mua"],
    "khach": ["Người mua", "Tên khách"],
    "noi dung": ["Mặt hàng", "Hàng bán"],
    "tong tien": ["Tổng cộng", "Tổng"],
    "tong tien hang": ["Tiền hàng", "Cộng tiền hàng"],
    "don vi tinh": ["ĐVT", "Đơn vị"],
    "dvt": ["Đơn vị tính", "Đơn vị"],
}

# Cách ghi khác của 4 trạng thái chuẩn. Chỉ dùng cách ghi không đổi nghĩa.
STATUS_SYNONYMS: dict[str, list[str]] = {
    "hoàn thành": ["Đã giao xong", "Thành công", "OK", "Xong rồi", "Đã nhận hàng"],
    "huỷ": ["Khách bỏ", "Hủy bỏ", "Đã hủy đơn", "Cancelled", "Bom hàng"],
    "trả hàng": ["Hoàn trả", "Khách trả lại", "Returned", "Trả lại hàng"],
    "đang giao": ["Đang vận chuyển", "Chờ giao", "Shipper đang giao", "Đang ship"],
}
WALKIN_NEW = ["KVL", ""]
_WALKIN_OLD = {"khach le", "kl", "le", "khach vang lai", "vang lai"}

_SALES_AMOUNT = re.compile(
    r"thanh tien|tong tien|tri gia|so tien|amount|tien hang|tong cong|doanh thu"
)
_IMPORT = re.compile(r"\bnhap\b|\bncc\b|nha cung cap")
_STATUS_COL = re.compile(r"trang thai|tinh trang|status|ket qua|xu ly")
_CUST_COL = re.compile(r"khach|ben mua|doi tac|customer|\bkh\b|nguoi mua")
_MONEY_COL = re.compile(r"tien|gia|tong|amount|thanh toan|tri gia|\btt\b")
_SKIP_MONEY = re.compile(r"\bma\b|so luong|\bsl\b|ty le|ti le|he so|quy doi|diem|stt|thu tu")


def _header_names(g: pd.DataFrame, h: int) -> list[str]:
    return [clean(str(v)) if not _cell_empty(v) else "" for v in g.iloc[h].tolist()]


class Mutator:
    def __init__(self, src: Path, kinds: list[str], seed: int, status_map: dict | None = None):
        self.src = Path(src)
        # Nhãn chuẩn của từng giá trị trạng thái gốc: theo ánh xạ đã kiểm tra nếu có, không thì từ khoá.
        self.status_map = {key(k): v for k, v in (status_map or {}).items()}
        self.kinds = set(kinds)
        self.rng = random.Random(seed)
        self.rename: dict[str, str] = {}
        self.status: dict[str, str] = {}
        self.walkin: dict[str, str] = {}
        self.layout: list[str] = []
        self._fmt_style: dict[str, str] = {}

    # ---------------------------------------------------------------- quyết định ánh xạ (ổn định theo tên)

    def new_name(self, name: str, taken: set[str]) -> str:
        if "rename" not in self.kinds or not name:
            return name
        if name in self.rename:
            return self.rename[name]
        opts = [o for o in SYNONYMS.get(key(name), []) if o not in taken and key(o) != key(name)]
        if not opts:
            return name
        new = self.rng.choice(opts)
        self.rename[name] = new
        return new

    def new_status(self, v: str) -> str:
        if v in self.status:
            return self.status[v]
        st = self.status_map.get(key(v)) or Biz.status_by_keyword(v)
        used = set(self.status.values())
        opts = [o for o in STATUS_SYNONYMS[st] if o not in used] or STATUS_SYNONYMS[st]
        new = self.rng.choice(opts)
        self.status[v] = new
        return new

    def new_walkin(self, v: str) -> str:
        if v not in self.walkin:
            self.walkin[v] = self.walkin.get("_choice") or self.rng.choice(WALKIN_NEW)
            self.walkin.setdefault("_choice", self.walkin[v] or "")
        return self.walkin[v]

    # ---------------------------------------------------------------- đột biến một lưới ô

    def mutate_grid(self, g: pd.DataFrame, default_year: int | None) -> pd.DataFrame:
        g = g.copy()
        h = _detect_header(g)
        names = _header_names(g, h)
        # Chỉ đổi trạng thái ở bảng bán hàng (có cột mã đơn), không đổi trạng thái phiếu nhập, khách hàng.
        keys = [key(n) for n in names if n]
        sales = any(_ORDER_COL.search(n) for n in names if n) or (
            any(_SALES_AMOUNT.search(k) for k in keys) and not any(_IMPORT.search(k) for k in keys)
        )
        for j, name in enumerate(names):
            k = key(name)
            col = g.columns[j]
            vals = [g.at[i, col] for i in g.index[h + 1 :]]
            if "status" in self.kinds and sales and _STATUS_COL.search(k):
                distinct = {clean(str(v)) for v in vals if not _cell_empty(v)}
                if 0 < len(distinct) <= 20:
                    for i in g.index[h + 1 :]:
                        v = g.at[i, col]
                        if not _cell_empty(v):
                            g.at[i, col] = self.new_status(clean(str(v)))
            if "walkin" in self.kinds and _CUST_COL.search(k):
                for i in g.index[h + 1 :]:
                    v = g.at[i, col]
                    if not _cell_empty(v) and key(v) in _WALKIN_OLD:
                        g.at[i, col] = self.new_walkin(clean(str(v)))
            if "format" in self.kinds:
                self._format_column(g, col, name, list(g.index[h + 1 :]), default_year)
        if "rename" in self.kinds:
            taken = set(names)
            for j, name in enumerate(names):
                new = self.new_name(name, taken)
                if new != name:
                    g.iat[h, j] = new
                    taken.add(new)
        return g

    def _format_column(self, g, col, name, rows, default_year) -> None:
        k = key(name)
        vals = [g.at[i, col] for i in rows if not _cell_empty(g.at[i, col])]
        if not vals:
            return
        # Ngày: mọi ô là ngày (datetime hoặc chuỗi ngày đầy đủ) → "1 thg 6 2025 [HH:MM]".
        if all(isinstance(v, (dt.datetime, pd.Timestamp)) or looks_like_date(v) for v in vals):
            for i in rows:
                v = g.at[i, col]
                if _cell_empty(v):
                    continue
                d = parse_date(v, default_year)
                if d is None:
                    continue
                s = f"{d.day} thg {d.month} {d.year}"
                if d.hour or d.minute or d.second:
                    s += f" {d.hour:02d}:{d.minute:02d}:{d.second:02d}"
                g.at[i, col] = s
            return
        if not _MONEY_COL.search(k) or _SKIP_MONEY.search(k):
            return
        nums = [parse_number(v) for v in vals]
        if any(x is None for x in nums) or not any(abs(x) >= 1000 for x in nums):
            return
        if not all(float(x).is_integer() for x in nums):
            return
        style = self._fmt_style.setdefault(name, self.rng.choice(["dot_d", "k", "space"]))
        if style == "k" and not all(x % 1000 == 0 for x in nums):
            style = "dot_d"
        for i in rows:
            v = g.at[i, col]
            x = parse_number(v) if not _cell_empty(v) else None
            if x is None:
                continue
            n = int(x)
            if style == "k":
                g.at[i, col] = f"{n // 1000}k" if n >= 0 else f"-{abs(n) // 1000}k"
            elif style == "space":
                g.at[i, col] = f"{n:,}".replace(",", " ")
            else:
                g.at[i, col] = f"{n:,}đ".replace(",", ".")

    # ---------------------------------------------------------------- ghi ra

    def run(self, out: Path) -> dict:
        out = Path(out)
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        for f in sorted(self.src.iterdir()):
            if f.name == "README.md" or f.name.startswith("."):
                continue
            suf = f.suffix.lower()
            if suf in (".sqlite", ".db", ".sqlite3"):
                self._mutate_sqlite(f, out / f.name)
                continue
            grids = _read_grids(f)
            if not grids:
                shutil.copy(f, out / f.name)
                continue
            mutated = []
            for label, g in grids:
                sheet = label.split(":", 1)[1] if ":" in label else ""
                ym = re.search(r"(20\d\d)", sheet) or re.search(r"(20\d\d)", f.stem)
                mg = self.mutate_grid(g, int(ym.group(1)) if ym else None)
                mutated.append((sheet, mg, ym is not None and bool(re.search(r"20\d\d", sheet))))
            self._write_with_layout(f, out, mutated)
        readme = (self.src / "README.md").read_text(encoding="utf-8")
        (out / "README.md").write_text(self._rewrite_readme(readme), encoding="utf-8")
        info = {
            "source": str(self.src),
            "kinds": sorted(self.kinds),
            "rename": self.rename,
            "status": self.status,
            "walkin": {k: v for k, v in self.walkin.items() if k != "_choice"},
            "layout": self.layout,
        }
        (out / "mutation.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return info

    def _write_with_layout(self, f: Path, out: Path, sheets: list) -> None:
        suf = f.suffix.lower()
        if "layout" in self.kinds:
            sheets = [(n, self._add_title(g, f.stem), y) for n, g, y in sheets]
            if suf in (".xlsx", ".xlsm") and len(sheets) > 1 and not any(y for *_, y in sheets):
                merged = self._merge(sheets)
                if merged is not None:
                    self.layout.append(f"gộp {len(sheets)} sheet của {f.name} thành một sheet")
                    sheets = [("Data", merged, False)]
            elif len(sheets) == 1:
                parts = self._split_by_month(sheets[0][1])
                if parts:
                    self.layout.append(f"tách {f.name} thành {len(parts)} file theo tháng")
                    for (y, m), g in parts:
                        self._write_file(out / f"{f.stem}_T{m}_{y}{f.suffix}", [("Data", g, False)])
                    return
        self._write_file(out / f.name, sheets)

    @staticmethod
    def _add_title(g: pd.DataFrame, stem: str) -> pd.DataFrame:
        top = pd.DataFrame([[f"BÁO CÁO {stem.upper()}"] + [None] * (g.shape[1] - 1),
                            [None] * g.shape[1]], columns=g.columns)  # fmt: skip
        return pd.concat([top, g], ignore_index=True)

    @staticmethod
    def _merge(sheets: list) -> pd.DataFrame | None:
        heads = []
        for _, g, _ in sheets:
            h = _detect_header(g)
            heads.append((h, tuple(_header_names(g, h))))
        if len({hh for _, hh in heads}) != 1:
            return None
        parts = [sheets[0][1]] + [
            g.iloc[h + 1 :] for (h, _), (_, g, _) in zip(heads[1:], sheets[1:])
        ]
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def _split_by_month(g: pd.DataFrame):
        h = _detect_header(g)
        names = _header_names(g, h)
        if not any(_ORDER_COL.search(n) for n in names if n):
            return None  # chỉ tách bảng bán hàng, không tách danh mục
        body = g.iloc[h + 1 :]
        for j, n in enumerate(names):
            if not re.search(r"ngay|date|thoi gian", key(n)):
                continue
            d = body.iloc[:, j].map(lambda v: parse_date(v) if not _cell_empty(v) else None)
            if d.notna().mean() < 0.98:
                continue
            ym = d.map(lambda x: (x.year, x.month))
            if ym.nunique() < 2:
                return None
            head = g.iloc[: h + 1]
            return [((y, m), pd.concat([head, body[ym == (y, m)]])) for y, m in sorted(ym.unique())]
        return None

    @staticmethod
    def _cell(v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
            return None
        if isinstance(v, pd.Timestamp):
            return v.to_pydatetime()
        if isinstance(v, np.generic):
            return v.item()
        return v

    def _write_file(self, path: Path, sheets: list) -> None:
        suf = path.suffix.lower()
        if suf in (".csv", ".tsv", ".txt"):
            g = sheets[0][1]
            with path.open("w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                for row in g.itertuples(index=False):
                    w.writerow(["" if self._cell(v) is None else self._cell(v) for v in row])
            return
        import openpyxl

        wb = openpyxl.Workbook()
        wb.remove(wb.active)
        for name, g, _ in sheets:
            ws = wb.create_sheet(title=(name or "Sheet1")[:31])
            for row in g.itertuples(index=False):
                ws.append([self._cell(v) for v in row])
        wb.save(path)

    def _mutate_sqlite(self, src: Path, dst: Path) -> None:
        shutil.copy(src, dst)
        con = sqlite3.connect(dst)
        try:
            tables = [
                r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
            ]
            for t in tables:
                df = pd.read_sql_query(f'SELECT * FROM "{t}"', con)
                grid = pd.concat(
                    [pd.DataFrame([list(df.columns)]), pd.DataFrame(df.values)], ignore_index=True
                ).astype(object)
                kinds = self.kinds
                self.kinds = kinds - {"layout"}
                mg = self.mutate_grid(grid, None)
                self.kinds = kinds
                cols = [str(c) for c in mg.iloc[0].tolist()]
                new = pd.DataFrame(mg.iloc[1:].values, columns=cols)
                new = new.map(self._cell)
                new.to_sql(t, con, if_exists="replace", index=False)
            con.commit()
        finally:
            con.close()

    def _rewrite_readme(self, readme: str) -> str:
        repl = {k: v for k, v in self.rename.items()}
        repl.update({k: v for k, v in self.status.items()})
        repl.update({k: v for k, v in self.walkin.items() if k != "_choice" and v})
        if not repl:
            return readme
        names = sorted(repl, key=len, reverse=True)
        pat = re.compile(r"(?<![\wÀ-ỹ])(" + "|".join(re.escape(n) for n in names) + r")(?![\wÀ-ỹ])")
        return pat.sub(lambda m: repl[m.group(1)], readme)


def mutate_shop(
    src: str | Path, out: str | Path, kinds=KINDS, seed: int = 0, status_map: dict | None = None
) -> dict:
    """`status_map`: ánh xạ trạng thái gốc → nhãn chuẩn (từ ánh xạ ngữ nghĩa đã kiểm tra) để đột biến giữ nghĩa."""
    return Mutator(Path(src), list(kinds), seed, status_map).run(Path(out))


def translate_mapping(mapping: dict, info: dict) -> dict:
    """Ánh xạ ngữ nghĩa của shop gốc → của shop đột biến (đổi tên cột, giá trị trạng thái, khách lẻ)."""
    m = json.loads(json.dumps(mapping))
    ren = info.get("rename") or {}
    for role in ("sales_lines", "orders", "payments", "conversion", "returns", "products"):
        r = m.get(role)
        if not r:
            continue
        for f, c in r.items():
            if f not in ("table", "by") and c in ren:
                r[f] = ren[c]
    st = info.get("status") or {}
    m["status_map"] = {st.get(k, k): v for k, v in (m.get("status_map") or {}).items()}
    wk = info.get("walkin") or {}
    m["walkin_values"] = [wk.get(v, v) for v in m.get("walkin_values") or [] if wk.get(v, v)]
    return m
