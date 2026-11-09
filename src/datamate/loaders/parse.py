"""Parse giá trị ô bẩn: tiền ("25.000đ", "25k", "1tr440"), số lượng kèm đơn vị, ngày nhiều kiểu."""

from __future__ import annotations

import datetime as dt
import math
import re

import pandas as pd

from datamate.text import strip_accents

# Số serial Excel: ngày 0 là 1899-12-30.
_EXCEL_EPOCH = dt.datetime(1899, 12, 30)
_SERIAL_MIN, _SERIAL_MAX = 36526, 54789  # 2000-01-01 .. 2049-12-31

_CURRENCY = re.compile(r"(vnđ|vnd|đồng|dong|đ|₫)\s*$", re.IGNORECASE)
_TR = re.compile(r"^(-?\d+(?:[.,]\d+)?)\s*(?:tr|triệu|trieu)\s*(\d*)$", re.IGNORECASE)
_K = re.compile(r"^(-?\d+(?:[.,]\d+)?)\s*(?:k|nghìn|ngàn|ngan|nghin)$", re.IGNORECASE)
_PLAIN = re.compile(r"^[+-]?\d[\d.,]*$")
_SPACED_THOUSANDS = re.compile(r"^-?\d{1,3}(?: \d{3})+(?:[.,]\d+)?$")
_QTY_UNIT = re.compile(r"^(-?\d+(?:[.,]\d+)?)\s*([^\d\s].*)$")


def _plain_number(s: str) -> float | None:
    """'1.200.000' -> 1200000, '25,000.0000' -> 25000, '16,56' -> 16.56, '18.6' -> 18.6."""
    if not _PLAIN.match(s):
        return None
    neg = s.startswith("-")
    s = s.lstrip("+-")
    if "." in s and "," in s:
        dec = "." if s.rfind(".") > s.rfind(",") else ","
        thou = "," if dec == "." else "."
        s = s.replace(thou, "").replace(dec, ".")
    elif "." in s:
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
            s = s.replace(".", "")
    elif "," in s:
        if re.fullmatch(r"\d{1,3}(,\d{3})+", s):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def parse_number(v: object) -> float | None:
    """Parse một ô thành số. Trả None nếu không phải số."""
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return None if isinstance(v, float) and math.isnan(v) else float(v)
    s = str(v).strip()
    if not s:
        return None
    s = _CURRENCY.sub("", s).strip()
    m = re.fullmatch(r"\((.+)\)", s)  # số âm kiểu kế toán: (25.000)
    if m:
        inner = parse_number(m.group(1))
        return None if inner is None or inner < 0 else -inner
    if _SPACED_THOUSANDS.match(s):
        s = s.replace(" ", "")
    m = _TR.match(s)
    if m:
        b = m.group(1)
        # "2,584tr" = 2,584 triệu: với đơn vị triệu, một dấu phân cách duy nhất là dấu thập phân.
        base = float(b.replace(",", ".")) if len(re.findall(r"[.,]", b)) == 1 else _plain_number(b)
        if base is None:
            return None
        tail = m.group(2)
        extra = int(tail) * 10 ** (6 - len(tail)) if tail else 0
        sign = -1 if base < 0 else 1
        return round(base * 1e6 + sign * extra, 6)
    m = _K.match(s)
    if m:
        base = _plain_number(m.group(1))
        return None if base is None else round(base * 1000, 6)
    return _plain_number(s)


def parse_qty_unit(v: object) -> tuple[float | None, str | None]:
    """'3 chai' -> (3, 'chai'); '2 xe 4m3' -> (2, 'xe 4m3'); '5' -> (5, None)."""
    n = parse_number(v)
    if n is not None:
        return n, None
    if v is None:
        return None, None
    s = str(v).strip()
    m = _QTY_UNIT.match(s)
    if not m:
        return None, None
    n = _plain_number(m.group(1))
    if n is None:
        return None, None
    return n, m.group(2).strip()


_MONTHS_THG = re.compile(
    r"^(\d{1,2})\s*(?:thg|tháng|thang|th)\s*(\d{1,2})[,\s]+(\d{4})(.*)$", re.IGNORECASE
)
# Phần giờ sau ngày: "14:30", "14:30:05", "14h30", "14g30", "2:05 PM", "14h".
_TIME_TAIL = r"(?:[ T,]+(\d{1,2})(?:[:hHgG](\d{2})?)(?::(\d{2}))?(?:\.\d+)?\s*([AaPp][Mm]|SA|CH|sáng|chiều|tối)?)?"
_DMY = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4}|\d{2})" + _TIME_TAIL + "$")
_YMD = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})" + _TIME_TAIL + "$")
_DM = re.compile(r"^(\d{1,2})[/.-](\d{1,2})$")
_TIME = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?")
# Giờ trong ngày ghi riêng một ô: "09:15", "14:05:30", "8h30", "14g05".
HHMM = re.compile(r"^\s*([01]?\d|2[0-3])\s*[:hHgG]\s*([0-5]\d)(?::([0-5]\d))?\s*$")


def norm_hhmm(v: object) -> str | None:
    """'8h30' -> '08:30', '14:05:30' -> '14:05:30'; None nếu không phải giờ trong ngày."""
    m = HHMM.match(str(v)) if v is not None else None
    if not m:
        return None
    out = f"{int(m.group(1)):02d}:{m.group(2)}"
    return out + (f":{m.group(3)}" if m.group(3) else "")


def _year(y: str) -> int:
    return int(y) + 2000 if len(y) == 2 else int(y)


def _hour(hh, ampm) -> int | None:
    if hh is None:
        return None
    h = int(hh)
    a = (ampm or "").lower()
    if a in ("pm", "ch", "chiều", "tối") and h < 12:
        h += 12
    elif a in ("am", "sa") and h == 12:
        h = 0
    return h


def date_order(values) -> str:
    """'dmy' hoặc 'mdy' cho một cột, theo bằng chứng của cả cột: ô có phần đầu > 12 chỉ đọc được ngày trước tháng,
    ô có phần thứ hai > 12 chỉ đọc được tháng trước ngày. Không có bằng chứng thì theo cách ghi ở Việt Nam (dmy)."""
    first = second = 0
    for v in values:
        if not isinstance(v, str):
            continue
        m = _DMY.match(v.strip())
        if m:
            first += int(m.group(1)) > 12
            second += int(m.group(2)) > 12
    return "mdy" if second > 0 and first == 0 else "dmy"


def _mk(y: int, m: int, d: int, hh=None, mm=None, ss=None) -> dt.datetime | None:
    try:
        return dt.datetime(y, m, d, int(hh or 0), int(mm or 0), int(ss or 0))
    except ValueError:
        return None


def parse_date(
    v: object, default_year: int | None = None, order: str = "dmy"
) -> dt.datetime | None:
    """Parse một ô thành datetime (ngày đặt ở Việt Nam: ngày trước tháng; `order='mdy'` khi cả cột ghi tháng
    trước ngày, xem `date_order`)."""
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.to_pydatetime()
    if isinstance(v, dt.datetime):
        return v
    if isinstance(v, dt.date):
        return dt.datetime(v.year, v.month, v.day)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if isinstance(v, float) and math.isnan(v):
            return None
        if _SERIAL_MIN <= v <= _SERIAL_MAX:
            return _EXCEL_EPOCH + dt.timedelta(days=float(v))
        return None
    s = str(v).strip()
    if not s:
        return None
    if re.fullmatch(r"\d{5}(\.\d+)?", s):
        return parse_date(float(s))
    m = _YMD.match(s)
    if m:
        return _mk(int(m[1]), int(m[2]), int(m[3]), _hour(m[4], m[7]), m[5], m[6])
    m = _DMY.match(s)
    if m:
        d, mo = (int(m[1]), int(m[2])) if order == "dmy" else (int(m[2]), int(m[1]))
        return _mk(_year(m[3]), mo, d, _hour(m[4], m[7]), m[5], m[6])
    m = _MONTHS_THG.match(strip_accents(s))
    if m:
        t = _TIME.search(m[4] or "")
        return _mk(int(m[3]), int(m[2]), int(m[1]), *(t.groups() if t else ()))
    m = _DM.match(s)
    if m and default_year:
        d, mo = (int(m[1]), int(m[2])) if order == "dmy" else (int(m[2]), int(m[1]))
        return _mk(default_year, mo, d)
    return None


def looks_like_date(v: object) -> bool:
    if isinstance(v, (dt.date, dt.datetime, pd.Timestamp)):
        return True
    if isinstance(v, str):
        s = v.strip()
        return bool(_YMD.match(s) or _DMY.match(s) or _MONTHS_THG.match(strip_accents(s)))
    return False
