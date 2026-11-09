"""Phát hiện câu không trả lời được trước khi gọi LLM (§7), bằng luật tổng quát, không theo shop.

Hai nhóm:
- Câu chủ quan / ý kiến / tư vấn: cụm từ đánh giá cảm tính mà rulebook của split KHÔNG định nghĩa.
- Hỏi thuộc tính shop không ghi (giới tính, địa chỉ): không cột nào và README không nhắc tới.
"""

from __future__ import annotations

import re

from datamate.text import key

# Từ vựng chung của câu hỏi kinh doanh (không lấy từ dữ liệu shop): tiêu chí so sánh / đánh giá có một trong các từ
# này là số đo tính được (tiền, đơn, số lượng, lần, ngày, nợ, lãi…). Tiêu chí không có từ nào ở đây và không phải khái
# niệm rulebook định nghĩa là đánh giá cảm tính (§7).
# Từ lượng ("nhiều", "cao") chỉ là số đo khi đi cùng một thứ sổ sách ghi được: "khen nhiều", "tay nghề cao" thì không.
MEASURE = re.compile(
    r"\b(dong|gan|xa|som|muon|tre|moi|cu|lau|nhanh|dat|re|tien|dthu|doanh thu|doanh so|dt|don|dh|so luong|sl|lan|"
    r"ngay|luot|phieu|no|lai|loi|gia|von|chay|hot|ban|mua|nhap|ton|con|het|giao|tra|huy|tang|giam|chi|thu|ty le|"
    r"ti le|ty trong|ti trong|phan tram|top|tong|trung binh|may|bao nhieu|bnhiu|bn|\d+)\b"
)
_QUANT = frozenset(["nhieu", "nhiu", "it", "cao", "thap", "lon", "nho", "hon", "kem", "bang"])
# Chủ ngữ là một chỉ số ("doanh thu … có tăng không"): câu có / không đó là so sánh số đo.
_METRIC = re.compile(
    r"\b(doanh thu|doanh so|dthu|tien|so don|don hang|so luong|lai gop|loi nhuan|cong no|ton kho|gia ban|gia nhap)\b"
)
_NUM_WORDS = "(?:mot|hai|ba|bon|nam|sau|bay|tam|chin|muoi)"
# Đối tượng của câu so sánh / đánh giá ("khách", "thợ"…): bỏ khỏi cửa sổ tiêu chí.
_ENTITY = re.compile(
    r"\b(khach hang|khach|nhan vien|nv|tho|nguoi|ai|mat hang|hang|san pham|sp|mon|ngay|thang|tuan|quy|nam|"
    r"nha cung cap|ncc|xe|may|dong may|lo|nhom|cua hang|tiem|quan|shop|nao|mo|la|cua|trong|cac|so|ban)\b"
)


def _rule_concepts(rulebook: str) -> list[tuple[str, set[str]]]:
    """Khái niệm "… nhất" rulebook định nghĩa: (tên đã bỏ dấu, các từ tiêu chí sau khi bỏ từ chỉ đối tượng)."""
    out = []
    for line in rulebook.splitlines():
        if not line.startswith("|"):
            continue
        for name in re.split(r",", key(line.strip("|").split("|")[0])):
            if re.search(r"\bnhat\b", name):
                words = set(_ENTITY.sub(" ", name.replace("nhat", " ")).split())
                out.append((name.strip(), words))
    return out


def _subject_of(window: str) -> set[str]:
    return set(re.findall(_ENTITY, window)) - {
        "nao",
        "mo",
        "la",
        "cua",
        "trong",
        "cac",
        "so",
        "ban",
        "ai",
    }


def evaluative(question: str, rulebook: str, known: str = "") -> tuple[str, str] | None:
    """Câu so sánh "… nhất" / câu "… có <đánh giá> không" mà tiêu chí không phải số đo, không phải khái niệm rulebook.

    ('unanswerable', lý do) khi tiêu chí là đánh giá cảm tính; ('abstain', lý do) khi rulebook có khái niệm "… nhất"
    cùng đối tượng và chung từ tiêu chí (có thể là cách nói lệch của khái niệm có định nghĩa). Ngược lại None.

    `known`: README + tên cột của shop. Tiêu chí có trong đó (khái niệm shop ghi, như hoa hồng, điểm) là dữ liệu."""
    q = key(question)
    rb = key(rulebook)
    # "… nhất" theo sau là số ("ít nhất 5 đơn") là ngưỡng, không phải so sánh.
    windows = [
        m.group(1) for m in re.finditer(rf"((?:\S+ ){{1,5}})nhat\b(?! (?:\d|{_NUM_WORDS}\b))", q)
    ]
    # Câu có / không: "X có <đánh giá> (với khách) không" — "không" là tiểu từ cuối câu, câu không hỏi số lượng.
    m = re.search(r"\bco ((?:\S+ ){1,8})(?:khong|hong|ko|k|khum)$", q)
    if (
        m
        and not windows
        and not re.search(r"\b(may|bao nhieu|bnhiu|bn|nhieu)\b", q)
        and not _METRIC.search(q[: m.start()])
    ):
        windows = [m.group(1)]
    for w in windows:
        if MEASURE.search(w):
            continue
        crit = _ENTITY.sub(" ", w).split()
        kn = f" {key(known)} "
        if (
            not crit
            or re.search(rf"\b{re.escape(' '.join(crit))}\b", rb)
            or f" {' '.join(w for w in crit if w not in _QUANT)} " in kn
        ):
            continue  # không có tiêu chí, hoặc tiêu chí có trong rulebook / dữ liệu của shop
        subj = _subject_of(w) or _subject_of(q)
        for name, cwords in _rule_concepts(rulebook):
            if cwords & set(crit) and (_subject_of(name) & subj or not subj):
                return (
                    "abstain",
                    f"tiêu chí '{' '.join(crit)}' gần khái niệm '{name}' của rulebook, không chắc",
                )
        return (
            "unanswerable",
            f"tiêu chí '{' '.join(crit)}' là đánh giá cảm tính, rulebook không định nghĩa",
        )
    return None


ATTRIBUTES = [
    # (mẫu trong câu hỏi đã bỏ dấu, các từ khoá phải có trong cột/README để trả lời được, lý do)
    (
        r"\bkhach (nu|nam)\b|gioi tinh",
        ["gioi tinh", "gender", "nam nu"],
        "shop không ghi giới tính khách",
    ),
    (
        r"\b(o|khach) duong\b|dia chi|\bo (quan|phuong)\b",
        ["dia chi", "address", "addr"],
        "shop không ghi địa chỉ khách",
    ),
]


def classify(
    question: str, rulebook: str, readme: str, columns: list[str]
) -> tuple[str, str] | None:
    """('unanswerable', lý do) nếu chắc chắn không trả lời được; ('abstain', lý do) nếu câu hỏi dùng từ cảm
    tính cho một đối tượng mà rulebook có định nghĩa khái niệm "… nhất" gần giống (diễn đạt lệch so với luật,
    ví dụ "nhà cung cấp uy tín" khi rulebook có "nhà cung cấp chính"): không chắc nên bỏ. Ngược lại None."""
    q = key(question)
    ev = evaluative(question, rulebook, readme + " " + " ".join(columns))
    if ev:
        return ev
    known = key(readme) + " " + " ".join(key(c) for c in columns)
    for pat, needs, reason in ATTRIBUTES:
        if re.search(pat, q) and not any(n in known for n in needs):
            return "unanswerable", reason
    return None


def precheck(question: str, rulebook: str, readme: str, columns: list[str]) -> str | None:
    """Trả về lý do nếu chắc chắn không trả lời được, ngược lại None."""
    r = classify(question, rulebook, readme, columns)
    return r[1] if r and r[0] == "unanswerable" else None
