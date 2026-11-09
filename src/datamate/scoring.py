"""Chấm điểm submission theo §9 rulebook (Penalized Accuracy).

Đúng +1, sai -1, bỏ (abstain hoặc thiếu dòng) 0. PA = (đúng - sai) / N.
"""

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

CORRECT, WRONG, ABSTAIN = 1, -1, 0

# Sai số dấu phẩy động cho phép khi so số.
_ABS_TOL = 1e-6
_REL_TOL = 1e-9


def read_jsonl(path: str | Path) -> list[dict]:
    """Đọc file JSONL, bỏ qua dòng trống. Dòng JSON hỏng bị bỏ qua (tính như thiếu)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


# ---------- So từng kiểu giá trị ----------


def is_number(x) -> bool:
    """JSON number thật: int/float, không phải bool, không NaN/Infinity."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def number_equal(pred, gold) -> bool:
    """Khớp tuyệt đối, chỉ bỏ qua sai số dấu phẩy động."""
    if not is_number(pred):
        return False
    return math.isclose(pred, gold, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def percent_equal(pred, gold) -> bool:
    """Phần trăm đúng 2 chữ số thập phân."""
    if not is_number(pred):
        return False
    return math.isclose(round(pred, 2), round(gold, 2), abs_tol=_ABS_TOL)


def normalize_string(s: str) -> str:
    """Bỏ phân biệt hoa thường, khoảng trắng, dấu câu; giữ dấu tiếng Việt."""
    s = unicodedata.normalize("NFC", s).casefold()
    s = re.sub(r"[\W_]+", " ", s)  # mọi ký tự không phải chữ/số thành khoảng trắng
    return " ".join(s.split())


def string_equal(pred, gold: str, aliases=()) -> bool:
    """Khớp đáp án hoặc một trong các alias (cách viết khác, mã)."""
    if not isinstance(pred, str):
        return False
    p = normalize_string(pred)
    return any(p == normalize_string(g) for g in [gold, *aliases] if isinstance(g, str))


def element_equal(pred, gold, aliases=()) -> bool:
    """Một phần tử list hoặc một ô bảng: số so như number, chuỗi so như string."""
    if is_number(gold):
        return number_equal(pred, gold)
    return string_equal(pred, gold, aliases)


def list_equal(pred, gold: list, aliases: list) -> bool:
    """Đúng cả tập và thứ tự. aliases[i] là các cách viết khác của gold[i]."""
    if not isinstance(pred, list) or len(pred) != len(gold):
        return False
    return all(
        element_equal(p, g, aliases[i] if i < len(aliases) else ())
        for i, (p, g) in enumerate(zip(pred, gold))
    )


def _align_columns(pred_cols: list, gold_cols: list) -> list[int] | None:
    """Trả về chỉ số cột của pred ứng với từng cột gold.

    Cột so theo tên đã chuẩn hoá; tên không khớp thì so theo vị trí.
    """
    if len(pred_cols) != len(gold_cols):
        return None
    pred_norm = [normalize_string(c) if isinstance(c, str) else None for c in pred_cols]
    gold_norm = [normalize_string(c) for c in gold_cols]
    if sorted(pred_norm, key=str) == sorted(gold_norm) and len(set(gold_norm)) == len(gold_norm):
        return [pred_norm.index(g) for g in gold_norm]
    return list(range(len(gold_cols)))


def _row_equal(pred_row, gold_row: list, name_aliases) -> bool:
    if not isinstance(pred_row, list) or len(pred_row) != len(gold_row):
        return False
    # Cột đầu là tên đối tượng, nhận alias; các cột sau là số đo.
    return all(
        element_equal(p, g, name_aliases if j == 0 else ())
        for j, (p, g) in enumerate(zip(pred_row, gold_row))
    )


def table_equal(pred, gold: dict, aliases: list) -> bool:
    """Bảng: cột theo tên (hoặc vị trí), dòng so như tập hợp. aliases[i] cho tên ở dòng i."""
    if not isinstance(pred, dict):
        return False
    cols, rows = pred.get("columns"), pred.get("rows")
    if not isinstance(cols, list) or not isinstance(rows, list):
        return False
    gold_rows = gold["rows"]
    if len(rows) != len(gold_rows):
        return False
    order = _align_columns(cols, gold["columns"])
    if order is None:
        return False
    if not all(isinstance(r, list) and len(r) == len(cols) for r in rows):
        return False
    pred_rows = [[r[k] for k in order] for r in rows]

    # Ghép mỗi dòng gold với một dòng pred chưa dùng (thứ tự dòng không quan trọng).
    used = [False] * len(pred_rows)
    for i, g in enumerate(gold_rows):
        name_aliases = aliases[i] if i < len(aliases) else ()
        for k, p in enumerate(pred_rows):
            if not used[k] and _row_equal(p, g, name_aliases):
                used[k] = True
                break
        else:
            return False
    return True


# ---------- Chấm một câu ----------


def score_one(pred: dict | None, gold: dict) -> int:
    """Chấm một câu: trả về CORRECT, WRONG hoặc ABSTAIN.

    pred: một dòng submission ({"id", "type", "answer"}), None nếu thiếu.
    gold: một dòng answers.jsonl ({"id", "type", "answer", "alias"}).
    """
    if pred is None or pred.get("type") == "abstain":
        return ABSTAIN
    ptype, gtype = pred.get("type"), gold["type"]

    if ptype == "unanswerable":
        # reason không phải chuỗi: dòng không đọc được, tính như bỏ.
        if "reason" in pred and not isinstance(pred["reason"], str):
            return ABSTAIN
        return CORRECT if gtype == "unanswerable" else WRONG
    if ptype != gtype:
        return WRONG

    if "answer" not in pred:
        return WRONG
    p, g, aliases = pred["answer"], gold["answer"], gold.get("alias") or []
    match gtype:
        case "number":
            ok = number_equal(p, g)
        case "percent":
            ok = percent_equal(p, g)
        case "date":
            ok = isinstance(p, str) and p == g
        case "boolean":
            ok = isinstance(p, bool) and p == g
        case "string":
            ok = string_equal(p, g, aliases)
        case "list":
            ok = list_equal(p, g, aliases)
        case "table":
            ok = table_equal(p, g, aliases)
        case _:
            ok = False
    return CORRECT if ok else WRONG


# ---------- Chấm cả file ----------


@dataclass
class Report:
    n: int
    correct: int
    wrong: int
    abstain: int
    per_question: dict[str, int]  # id -> CORRECT / WRONG / ABSTAIN

    @property
    def pa(self) -> float:
        return (self.correct - self.wrong) / self.n if self.n else 0.0


def score(preds: list[dict], golds: list[dict]) -> Report:
    """Chấm toàn bộ. N là số câu trong golds; id thừa trong preds bị bỏ qua.

    Một id xuất hiện nhiều lần trong preds thì lấy dòng đầu tiên.
    """
    pred_by_id: dict[str, dict] = {}
    for p in preds:
        pid = p.get("id")
        if isinstance(pid, str) and pid not in pred_by_id:
            pred_by_id[pid] = p

    per_q = {g["id"]: score_one(pred_by_id.get(g["id"]), g) for g in golds}
    vals = list(per_q.values())
    return Report(
        n=len(vals),
        correct=vals.count(CORRECT),
        wrong=vals.count(WRONG),
        abstain=vals.count(ABSTAIN),
        per_question=per_q,
    )
