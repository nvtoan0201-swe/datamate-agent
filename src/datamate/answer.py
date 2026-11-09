"""Ép đáp án do code sinh ra về đúng format §9 rulebook, và khoá chuẩn hoá để so các lời giải."""

from __future__ import annotations

import datetime as dt
import json
import math
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from datamate.scoring import normalize_string
from datamate.submission import check_row

ANSWER_TYPES = ("number", "percent", "string", "date", "boolean", "list", "table")


class FormatError(ValueError):
    pass


def _py(v):
    """numpy/pandas scalar → kiểu Python."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def _num(v, decimals: int | None = None):
    v = _py(v)
    if isinstance(v, bool) or v is None:
        raise FormatError(f"không phải số: {v!r}")
    if isinstance(v, str):
        from datamate.loaders.parse import parse_number

        p = parse_number(v.replace("%", "").strip())
        if p is None:
            raise FormatError(f"không phải số: {v!r}")
        v = p
    try:
        v = float(v)
    except (
        TypeError,
        ValueError,
    ) as e:  # dict, list, object… do code LLM gán: lỗi format, không làm sập câu
        raise FormatError(f"không phải số: {v!r}"[:200]) from e
    if not math.isfinite(v):
        raise FormatError(f"số không hữu hạn: {v!r}")
    if decimals is not None:
        v = float(Decimal(repr(v)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))
    if abs(v - round(v)) < 1e-9:
        return round(v)
    return v


def _date(v) -> str:
    v = _py(v)
    if isinstance(v, pd.Period):
        return str(v)[:7] if v.freqstr.startswith("M") else str(v.start_time.date())
    if isinstance(v, (dt.datetime, dt.date)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, str):
        s = v.strip()
        for fmt, out in (
            ("%Y-%m-%d", "%Y-%m-%d"),
            ("%Y-%m", "%Y-%m"),
            ("%d/%m/%Y", "%Y-%m-%d"),
            ("%m/%Y", "%Y-%m"),
        ):
            try:
                return dt.datetime.strptime(s[:10] if fmt == "%Y-%m-%d" else s, fmt).strftime(out)
            except ValueError:
                continue
    raise FormatError(f"không phải ngày: {v!r}")


def _scalar_list_item(v):
    v = _py(v)
    if isinstance(v, (dt.datetime, dt.date)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return _num(v, 2)
    if isinstance(v, str):
        return v
    raise FormatError(f"phần tử list không hợp lệ: {v!r}")


def coerce(value, typ: str):
    """Trả về đáp án đúng format của `typ`, hoặc raise FormatError."""
    if typ not in ANSWER_TYPES:
        raise FormatError(f"type lạ: {typ!r}")
    if isinstance(value, pd.Series):
        value = value.tolist()
    if isinstance(value, (np.ndarray, tuple, set)):
        value = list(value)
    if typ in ("number", "percent", "string", "date", "boolean") and isinstance(value, list):
        if len(value) != 1:
            raise FormatError(f"{typ} nhưng nhận list {len(value)} phần tử")
        value = value[0]

    match typ:
        case "number":
            ans = _num(value)
            if isinstance(ans, float):
                ans = _num(ans, 2)
        case "percent":
            ans = _num(value, 2)
        case "date":
            ans = _date(value)
        case "boolean":
            v = _py(value)
            if isinstance(v, str) and v.strip().lower() in (
                "true",
                "false",
                "có",
                "không",
                "co",
                "khong",
            ):
                v = v.strip().lower() in ("true", "có", "co")
            if not isinstance(v, (bool, int)):
                raise FormatError(f"không phải boolean: {v!r}")
            ans = bool(v)
        case "string":
            v = _py(value)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                raise FormatError("chuỗi rỗng")
            ans = str(v).strip()
        case "list":
            if isinstance(value, dict):
                value = list(value.keys())
            if not isinstance(value, list):
                value = [value]
            ans = [_scalar_list_item(v) for v in value]
        case "table":
            ans = _table(value)
    row = {"id": "x", "type": typ, "answer": ans}
    errs = check_row(row)
    if errs:
        raise FormatError("; ".join(errs))
    return ans


def _table(value) -> dict:
    if isinstance(value, pd.DataFrame):
        df = value.reset_index() if not isinstance(value.index, pd.RangeIndex) else value
        cols = [str(c) for c in df.columns]
        rows = df.values.tolist()
    elif isinstance(value, dict) and "columns" in value and "rows" in value:
        cols, rows = list(value["columns"]), list(value["rows"])
    elif isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        cols, rows = ["khach_hang", "so_tien_no"][: len(value[0])], [list(r) for r in value]
    else:
        raise FormatError(f"không dựng được bảng từ {type(value).__name__}")
    out_rows = []
    for r in rows:
        out = []
        for v in r:
            v = _py(v)
            out.append(
                _num(v, 2) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
            )
        out_rows.append(out)
    return {"columns": [str(c) for c in cols], "rows": out_rows}


def answer_key(typ: str, ans) -> str:
    """Khoá so sánh hai đáp án khi bỏ phiếu: cùng khoá thì coi là cùng đáp án theo luật chấm."""
    if typ == "unanswerable":
        return "UNANSWERABLE"
    if typ in ("number", "percent"):
        return f"{typ}:{round(float(ans), 2):.2f}"
    if typ == "string":
        return f"string:{normalize_string(ans)}"
    if typ == "list":
        return "list:" + json.dumps(
            [normalize_string(x) if isinstance(x, str) else round(float(x), 2) for x in ans],
            ensure_ascii=False,
        )
    if typ == "table":
        rows = sorted(
            json.dumps(
                [normalize_string(x) if isinstance(x, str) else round(float(x), 2) for x in r],
                ensure_ascii=False,
            )
            for r in ans["rows"]
        )
        return "table:" + json.dumps(rows, ensure_ascii=False)
    return f"{typ}:{json.dumps(ans, ensure_ascii=False)}"
