"""Kiểm tra format một dòng submission theo §9 rulebook."""

import datetime as dt
import re

from datamate.scoring import is_number

TYPES = {
    "number",
    "percent",
    "string",
    "date",
    "boolean",
    "list",
    "table",
    "unanswerable",
    "abstain",
}

_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_MONTH = re.compile(r"\d{4}-\d{2}")


def _valid_date(s: str) -> bool:
    try:
        if _DAY.fullmatch(s):
            dt.date.fromisoformat(s)
            return True
        if _MONTH.fullmatch(s):
            dt.date.fromisoformat(s + "-01")
            return True
    except ValueError:
        pass
    return False


def check_row(row: dict) -> list[str]:
    """Trả về danh sách lỗi của một dòng. Lỗi làm câu đó chắc chắn sai hoặc bị bỏ."""
    errors = []
    t = row.get("type")
    if t not in TYPES:
        return [f"type không hợp lệ: {t!r}"]
    if t == "abstain":
        return []
    if t == "unanswerable":
        if "reason" in row and not isinstance(row["reason"], str):
            errors.append("reason không phải chuỗi (cả dòng tính như bỏ)")
        return errors

    if "answer" not in row:
        return ["thiếu trường answer"]
    a = row["answer"]
    match t:
        case "number" | "percent":
            if not is_number(a):
                errors.append(f"{t} phải là JSON number hữu hạn, nhận {a!r}")
        case "string":
            if not isinstance(a, str) or not a.strip():
                errors.append(f"string phải là chuỗi không rỗng, nhận {a!r}")
        case "date":
            if not isinstance(a, str) or not _valid_date(a):
                errors.append(f"date phải là YYYY-MM-DD hoặc YYYY-MM có thật, nhận {a!r}")
        case "boolean":
            if not isinstance(a, bool):
                errors.append(f"boolean phải là true/false, nhận {a!r}")
        case "list":
            if not isinstance(a, list):
                errors.append(f"list phải là mảng JSON, nhận {type(a).__name__}")
            elif not all(isinstance(e, str) or is_number(e) for e in a):
                errors.append("phần tử list phải là chuỗi hoặc số")
        case "table":
            errors += _check_table(a)
    return errors


def percent_ratio_warning(rows: list[dict], threshold: float = 0.2) -> str | None:
    """Cảnh báo nếu quá nhiều câu percent có |x| <= 1: có thể đã ghi tỉ lệ thay vì phần trăm.

    Xét tổng thể vì từng câu |x| <= 1 vẫn có thể đúng (train có khoảng 7%).
    """
    vals = [r["answer"] for r in rows if r.get("type") == "percent" and is_number(r.get("answer"))]
    small = sum(1 for v in vals if abs(v) <= 1)
    if vals and small / len(vals) > threshold:
        return (
            f"{small}/{len(vals)} câu percent có |x| <= 1: có thể ghi dạng tỉ lệ (0.124) "
            "thay vì số phần trăm (12.4)"
        )
    return None


def _check_table(a) -> list[str]:
    if not isinstance(a, dict):
        return ["table phải là object {columns, rows}"]
    cols, rows = a.get("columns"), a.get("rows")
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        return ["table.columns phải là mảng chuỗi"]
    if not isinstance(rows, list):
        return ["table.rows phải là mảng"]
    errors = []
    for i, r in enumerate(rows):
        if not isinstance(r, list) or len(r) != len(cols):
            errors.append(f"dòng {i} không phải mảng dài {len(cols)}")
        elif not all(isinstance(c, str) or is_number(c) for c in r):
            errors.append(f"dòng {i} có ô không phải chuỗi hoặc số")
        elif any(
            isinstance(c, str) and re.fullmatch(r"[\d.,\s]+(đ|vnd|%)?", c.lower()) for c in r[1:]
        ):
            errors.append(f"dòng {i} có ô số ghi dạng chuỗi")
    return errors
