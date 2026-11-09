"""Kiểm tra tĩnh code LLM viết, bắt lỗi ngữ nghĩa mà chạy vẫn ra số (IMP-02).

Mốc hết ngày: cột ngày giờ có giờ thật (thời điểm thanh toán, giờ tạo đơn) so `<= ngày` hay `> ngày` với một ngày
không giờ thì bỏ sót / tính thừa mọi dòng trong ngày đó sau 00:00. "Đến hết ngày D" phải là `< D + 1 ngày`,
`.dt.normalize() <= D` hoặc `p.mask(cột)`. Không dựa vào tên cột: cột có giờ nhận bằng dữ liệu.
"""

from __future__ import annotations

import re

import pandas as pd

# Vế phải đã xử lý mốc hết ngày (cộng ngày, giờ cuối ngày, normalize…).
_RHS_OK = re.compile(
    r"Timedelta|DateOffset|timedelta|23:59|normalize|\.date\(\)|\bdt\.date\b|day_end|end_of_day|\+\s*1\b"
)


def time_columns(tables: dict) -> set[str]:
    """Tên các cột ngày giờ có giờ thật (≥ 1 % ô không phải 00:00:00)."""
    out = set()
    for df in tables.values():
        for c in df.columns:
            s = df[c]
            if not pd.api.types.is_datetime64_any_dtype(s):
                continue
            s = s.dropna()
            if len(s) and (s != s.dt.normalize()).mean() >= 0.01:
                out.add(str(c))
    return out


def _var_ok(rhs: str, code: str) -> bool:
    """Vế phải là biến đã gán giá trị hết ngày ở chỗ khác trong code."""
    m = re.match(r"\s*([A-Za-z_][A-Za-z_0-9]*)", rhs)
    if not m:
        return False
    return any(
        _RHS_OK.search(a.group(1))
        for a in re.finditer(rf"^\s*{re.escape(m.group(1))}\s*=(?!=)(.+)$", code, re.MULTILINE)
    )


def temporal_issues(code: str | None, time_cols: set[str]) -> list[str]:
    """Các chỗ code so cột có giờ với một ngày bằng `<=` / `>` (pandas) hoặc `<=` / `>` / BETWEEN (SQL)."""
    if not code or not time_cols:
        return []
    out = []
    sql_day_end = "23:59" in code or "days=1" in code
    for line in code.splitlines():
        s = line.split("#", 1)[0]
        for m in re.finditer(r"""\[\s*(['"])(.+?)\1\s*\]\s*(<=|>(?!=))\s*(.+)""", s):
            col, rhs = m.group(2), m.group(4)
            if col in time_cols and not _RHS_OK.search(rhs) and not _var_ok(rhs, code):
                out.append(f"cột {col!r} có giờ, so `{m.group(3)} {rhs.strip()[:40]}`")
        for m in re.finditer(r""""([^"]+)"\s*(<=|>(?!=)|between)\s*(.+)""", s, re.IGNORECASE):
            col, rhs = m.group(1), m.group(3)
            if col in time_cols and not _RHS_OK.search(rhs) and not sql_day_end:
                out.append(f"cột {col!r} có giờ, SQL so `{m.group(2)} {rhs.strip()[:40]}`")
    return out


def temporal_repair_message(issues: list[str]) -> str:
    return (
        "Code so cột ngày giờ có giờ thật với một ngày không giờ: "
        + "; ".join(issues[:3])
        + ". Các dòng trong ngày đó sau 00:00 bị bỏ sót (hoặc tính thừa). 'Đến hết ngày D' phải là "
        "`cột < D + pd.Timedelta(days=1)` hoặc `cột.dt.normalize() <= D` hoặc `p.mask(cột)` (SQL: "
        "`date(\"cột\") <= 'YYYY-MM-DD'`); 'sau ngày D' là `cột >= D + pd.Timedelta(days=1)`. Sửa và viết lại toàn bộ code."
    )


# ---------------------------------------------------------------- ngưỡng "từ N trở lên" / "quá N" (§2 rulebook)
# Từ vựng tiếng Việt chung của câu hỏi (đã bỏ dấu). Mỗi mẫu: (regex có nhóm số, phép so sánh đúng).
_NUM = r"(\d[\d.,]*(?:\s*(?:k|tr|trieu|nghin|ngan)\b)?)"
_THRESHOLD_PATTERNS = [
    (rf"\btu {_NUM}\s*\S*\s*(?:\S+\s+)?tro len", ">="),
    (rf"\b(?:it nhat|toi thieu|khong duoi|khong it hon) {_NUM}", ">="),
    (rf"(?:>=|≥)\s*{_NUM}", ">="),
    (rf"\b(?:toi da|khong qua|khong vuot qua|khong hon) {_NUM}", "<="),
    (rf"\b(?:qua|hon|tren|vuot) {_NUM}", ">"),
    (rf"\b(?:duoi|it hon) {_NUM}", "<"),
]
_FLIP = {">=": "<=", "<=": ">=", ">": "<", "<": ">"}


def question_thresholds(question: str) -> list[tuple[float, str]]:
    """Các ngưỡng câu hỏi nêu: [(N, phép so sánh đúng theo §2)]. 'từ 4 đơn trở lên' → (4, '>=')."""
    from datamate.loaders.parse import parse_number
    from datamate.text import strip_accents

    t = strip_accents(question).lower()
    t = re.sub(r"(\d)\s+(tr|trieu)\b", r"\1\2", t)  # "15 triệu" → "15trieu"
    out, seen = [], set()
    for pat, op in _THRESHOLD_PATTERNS:
        for m in re.finditer(pat, t):
            if m.start(1) in seen:
                continue
            tail = t[m.end(1) : m.end(1) + 30]
            if re.match(r"[/-]\d", tail) or (m.group(1).strip() == "1" and op == ">"):
                continue  # số là một phần của ngày tháng ("6/2025 qua 7/2025"); "trên 1 đơn" là "mỗi đơn"
            # "qua 3 tháng", "hơn 7 ngày liền trước": khoảng / kỳ thời gian, không phải ngưỡng. Riêng "quá N ngày"
            # (có dấu, như tuổi nợ quá 10 ngày) vẫn là ngưỡng.
            if re.match(r"\s*(thang|tuan|nam|quy|ngay)\b", tail):
                accented = re.search(rf"\bquá\s*{re.escape(m.group(1).strip())}", question.lower())
                if (
                    re.match(r"\s*(thang|tuan|nam|quy)\b", tail)
                    or re.search(r"truoc|ke tu|lien", tail)
                    or (op == ">" and not accented and t[m.start() : m.start(1)].strip() == "qua")
                ):
                    continue
            raw = (
                m.group(1).strip().replace("trieu", "tr").replace("nghin", "k").replace("ngan", "k")
            )
            n = parse_number(raw)
            if n is not None:
                seen.add(m.start(1))
                out.append((float(n), op))
    return out


_CMP = re.compile(
    r"(>=|<=|(?<![-=<>])>(?!=)|(?<![<=])<(?![=<]))\s*\(?\s*(-?\d[\d_]*(?:\.\d+)?(?:e\d+)?)\b"
)
_CMP_REV = re.compile(
    r"\b(-?\d[\d_]*(?:\.\d+)?(?:e\d+)?)\s*\)?\s*(>=|<=|(?<![-=<>])>(?!=)|(?<![<=])<(?![=<]))"
)


def _ok(op: str, lit: float, n: float, want: str) -> bool:
    """`x op lit` có cùng nghĩa với `x want n` không (với số nguyên: `> n-1` ≡ `>= n`, `< n+1` ≡ `<= n`)."""
    if lit == n:
        return op == want
    pairs = {(">", n - 1): ">=", (">=", n + 1): ">", ("<", n + 1): "<=", ("<=", n - 1): "<"}
    return pairs.get((op, lit)) == want


def threshold_issues(code: str | None, question: str) -> list[str]:
    """Code so với ngưỡng N của câu hỏi bằng phép so sánh khác §2 ('từ N trở lên' mà viết `> N`)."""
    if not code:
        return []
    th = question_thresholds(question)
    if not th:
        return []
    found: list[tuple[str, float]] = []
    for line in code.splitlines():
        s = line.split("#", 1)[0]
        found += [(m.group(1), float(m.group(2).replace("_", ""))) for m in _CMP.finditer(s)]
        found += [
            (_FLIP[m.group(2)], float(m.group(1).replace("_", ""))) for m in _CMP_REV.finditer(s)
        ]
    out = []
    for n, want in th:
        rel = [(op, lit) for op, lit in found if lit in (n, n - 1, n + 1)]
        if (
            rel
            and not any(_ok(op, lit, n, want) for op, lit in rel)
            and any(lit == n for _, lit in rel)
        ):
            ops = ", ".join(sorted({f"{op} {lit:g}" for op, lit in rel}))
            out.append(f"ngưỡng: câu hỏi cần `{want} {n:g}`, code so `{ops}`")
    return out


def threshold_repair_message(issues: list[str]) -> str:
    return (
        "Phép so sánh với ngưỡng trong code khác câu hỏi: "
        + "; ".join(issues[:3])
        + ". Theo rulebook §2: 'quá N, hơn N, trên N, dưới N' KHÔNG gồm N; 'từ N trở lên', 'ít nhất N' gồm N. "
        "Sửa phép so sánh và viết lại toàn bộ code."
    )
