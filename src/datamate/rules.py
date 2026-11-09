"""Gắn hàm nghiệp vụ với câu luật trong rulebook của split đang chạy.

- `check_rulebook(text)`: với mỗi hàm cài đặt một luật, kiểm tra đoạn văn bản đặc trưng (neo) của luật
  còn trong rulebook không. Neo không còn thì tắt hàm: gọi hàm sẽ báo lỗi hướng dẫn LLM tự viết code.
- Đọc thông số có dạng cố định (giờ ca, mốc Tết, ngày đầu tuần) thay vì viết cứng. Không đọc được thì
  dùng mặc định và ghi cảnh báo.
- `split_rules` / `select_rules`: tách rulebook thành từng dòng luật và chọn các luật gần câu hỏi nhất
  bằng từ khoá (đưa lên cạnh câu hỏi, không thay rulebook đầy đủ).
"""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass, field

from datamate.text import clean, key

log = logging.getLogger(__name__)

# Neo của mỗi luật: các đoạn phải cùng có trong rulebook (đã chuẩn hoá). `{n}` khớp một số bất kỳ,
# để luật đổi thông số (giờ ca, ngày Tết) không làm tắt hàm; thông số đọc riêng ở `_parse_params`.
RULE_ANCHORS: dict[str, list[str]] = {
    "tet": ["{n} tháng chạp", "mùng {n} tháng giêng"],
    "before_tet": ["so với trước tết", "cùng số ngày ngay trước"],
    "orders.partial_return": ["kể cả chỉ trả một phần", "bỏ cả đơn"],
    "orders.shipping_counts": ["đơn đang giao vẫn tính"],
    "cancel_rate": ["tỉ lệ đơn huỷ", "mọi đơn đặt trong kỳ"],
    "shift": ["ca sáng {n}:{n}", "ca chiều {n}:{n}"],
    "this_week": ["tuần này", "từ thứ hai của tuần chứa hôm nay"],
    "last_week": ["tuần trước, tuần rồi", "thứ hai đến chủ nhật"],
    "this_month": ["tháng này", "từ ngày 1 đến hết hôm nay"],
    "this_month_compare": ["cùng số ngày", "ngày trùng số với hôm nay"],
    "month_no_year": ["không ghi năm", "sau hôm nay thì của năm trước"],
    "last_n_days": ["n ngày gần nhất", "kể cả hôm nay"],
    "ytd": ["từ đầu năm", "đến hết hôm nay"],
    "until": ["đến hết ngày x", "từ đầu dữ liệu"],
    "debt_age": ["tuổi nợ", "từ ngày đặt"],
    "money_round": ["làm tròn về đồng", "nửa đồng làm tròn lên"],
    "tiebreak": ["bằng nhau", "theo mã tăng dần"],
    "change_pct": ["so với kỳ trước", "giảm thì ra số âm"],
    "walkin": ["khách lẻ"],
    # Xếp hạng §3 của rulebook public (rulebook train không có: các hàm này tắt khi chạy train).
    "loyal_customers": ["khách trung thành nhất", "nhiều ngày có mua nhất"],
    "big_spenders": ["khách sộp nhất", "từ {n} đơn trở lên"],
    "fickle_customers": ["khách hay đổi ý nhất", "nhiều đơn huỷ nhất"],
    "popular_items": ["được ưa chuộng nhất", "nhiều đơn nhất"],
    "busiest_days": ["ngày đắt khách nhất", "nhiều đơn nhất"],
    "peak_hour": ["giờ cao điểm", "giờ bắt đầu khung"],
    "hardest_workers": ["chăm chỉ nhất", "nhiều ngày làm việc nhất"],
    "phone_model": ["dòng máy", "dung lượng"],
    # Nhóm hàng lấy theo cột nhóm hàng (§3): shop không có cột nhóm thì câu hỏi nhóm hàng không trả lời được.
    "product_group": ["nhóm hàng", "theo cột nhóm hàng"],
    "phone_brand": ["hãng", "chữ đầu tên hàng"],
    "repeat_vehicles": ["xe khách quen", "phiếu có thu tiền"],
    # Luật chỉ có ở rulebook train (public đã xoá): hàm tương ứng tắt khi chạy public.
    "split_units": ["số đơn vị lớn nguyên", "số đơn vị cơ bản lẻ"],
    "weighted_import_price": ["bình quân theo số lượng nhập"],
}

# Hàm `dm` → các luật nó cài đặt. Một luật bị tắt thì mọi hàm dựa trên nó bị tắt.
FUNCTION_RULES: dict[str, list[str]] = {
    "tet": ["tet"],
    "before_tet": ["tet", "before_tet"],
    "orders": ["orders.partial_return", "orders.shipping_counts"],
    "sold_lines": ["orders.partial_return", "orders.shipping_counts"],
    "revenue": ["orders.partial_return", "orders.shipping_counts"],
    "count_orders": ["orders.partial_return", "orders.shipping_counts"],
    "aov": ["orders.partial_return", "orders.shipping_counts", "money_round"],
    "daily_revenue": ["orders.partial_return", "orders.shipping_counts"],
    "cancel_rate": ["cancel_rate"],
    "shift": ["shift"],
    "this_week": ["this_week"],
    "last_week": ["last_week"],
    "this_month": ["this_month"],
    "last_month_same_days": ["this_month_compare"],
    "last_n_days": ["last_n_days"],
    "ytd": ["ytd"],
    "until": ["until"],
    "age_days": ["debt_age"],
    "change_pct": ["change_pct"],
    "split_units": ["split_units"],
    "weighted_import_price": ["weighted_import_price"],
    "loyal_customers": ["loyal_customers", "orders.partial_return"],
    "big_spenders": ["big_spenders", "orders.partial_return", "money_round"],
    "fickle_customers": ["fickle_customers"],
    "popular_items": ["popular_items", "orders.partial_return"],
    "busiest_days": ["busiest_days", "orders.partial_return"],
    "peak_hour": ["peak_hour", "orders.partial_return"],
    "hardest_workers": ["hardest_workers"],
    "phone_model": ["phone_model"],
    "phone_brand": ["phone_brand"],
    "repeat_vehicles": ["repeat_vehicles"],
}

_WEEKDAYS = ["thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật"]


def normalize(text: str) -> str:
    """Bỏ markdown, gộp khoảng trắng, chữ thường; thống nhất 'hủy' / 'huỷ' và các dấu gạch."""
    t = clean(text)
    t = re.sub(r"[*_`#>|]", " ", t)
    t = t.replace("–", "-").replace("—", "-").replace("−", "-")
    t = t.lower().replace("hủy", "huỷ").replace("thủy", "thuỷ")
    return re.sub(r"\s+", " ", t).strip()


def _anchor_regex(anchor: str) -> re.Pattern:
    parts = anchor.split("{n}")
    norm = [re.escape(normalize(p)) for p in parts]
    # Giữ khoảng trắng quanh {n} mà normalize đã cắt ở hai đầu từng đoạn.
    pat = ""
    for i, (raw, n) in enumerate(zip(parts, norm)):
        if i:
            pat += r"\d{1,2}" + (" " if raw.startswith(" ") else "")
        pat += n + (" " if raw.endswith(" ") and i < len(parts) - 1 else "")
    return re.compile(pat)


@dataclass
class RuleConfig:
    """Kết quả đối chiếu rulebook của split với các hàm dựng sẵn."""

    disabled: dict[str, str] = field(default_factory=dict)  # tên luật → neo không còn thấy
    shifts: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"sáng": (6 * 60, 13 * 60 + 59), "chiều": (14 * 60, 21 * 60 + 59)}
    )  # phút trong ngày, gồm cả hai đầu
    tet_start: tuple[int, int] = (12, 23)  # (tháng âm, ngày âm), năm âm Y − 1
    tet_end: tuple[int, int] = (1, 10)  # năm âm Y
    week_start: int = 0  # 0 = thứ Hai
    warnings: list[str] = field(default_factory=list)
    # Loại dòng điều chỉnh (nằm trong bảng dòng hàng) KHÔNG tính vào doanh thu, đọc từ câu luật doanh thu một đơn.
    revenue_excludes: frozenset = frozenset({"discount", "shipping", "vat"})

    def function_disabled(self, fn: str) -> str | None:
        """Lý do nếu hàm `fn` bị tắt, ngược lại None."""
        for r in FUNCTION_RULES.get(fn, []):
            if r in self.disabled:
                return r
        return None

    @property
    def disabled_functions(self) -> list[str]:
        return sorted(f for f in FUNCTION_RULES if self.function_disabled(f))


DEFAULT_RULES = RuleConfig()


def check_rulebook(text: str) -> RuleConfig:
    """Đối chiếu neo của từng luật với rulebook, đọc thông số. Ghi cảnh báo cho mọi hàm bị tắt."""
    norm = normalize(text)
    cfg = RuleConfig()
    for rule, anchors in RULE_ANCHORS.items():
        missing = [a for a in anchors if not _anchor_regex(a).search(norm)]
        if missing:
            cfg.disabled[rule] = missing[0]
    _parse_params(norm, cfg)
    for fn in cfg.disabled_functions:
        r = cfg.function_disabled(fn)
        cfg.warnings.append(
            f"tắt dm.{fn}: luật '{r}' không còn trong rulebook (thiếu neo {cfg.disabled[r]!r})"
        )
    for w in cfg.warnings:
        log.warning("rulebook: %s", w)
    return cfg


def _parse_params(norm: str, cfg: RuleConfig) -> None:
    for name in ("sáng", "chiều"):
        m = re.search(rf"ca {name} (\d{{1,2}}):(\d{{2}}) ?- ?(\d{{1,2}}):(\d{{2}})", norm)
        if m:
            a, b, c, d = (int(x) for x in m.groups())
            cfg.shifts[name] = (a * 60 + b, c * 60 + d)
        elif "shift" not in cfg.disabled:
            cfg.warnings.append(f"không đọc được giờ ca {name}, dùng mặc định {cfg.shifts[name]}")
    m1 = re.search(r"(\d{1,2}) tháng chạp", norm)
    m2 = re.search(r"mùng (\d{1,2}) tháng giêng", norm)
    if m1 and m2:
        cfg.tet_start, cfg.tet_end = (12, int(m1.group(1))), (1, int(m2.group(1)))
    elif "tet" not in cfg.disabled:
        cfg.warnings.append("không đọc được mốc Tết, dùng mặc định 23 tháng Chạp → mùng 10 Giêng")
    # Doanh thu một đơn: "không trừ giảm giá trên cả đơn", "không cộng thuế vat", "không gồm phí giao hàng".
    # Dòng điều chỉnh loại khác (phí phục vụ, phụ thu) tính vào doanh thu như dòng thường.
    excl = set()
    for kind, pat in (
        ("discount", r"không trừ giảm giá (?:trên )?cả đơn"),
        ("vat", r"không cộng (?:thuế )?vat"),
        ("shipping", r"không gồm phí (?:giao|vận chuyển|ship)"),
    ):
        if re.search(pat, norm):
            excl.add(kind)
    if excl:
        cfg.revenue_excludes = frozenset(excl)
    else:
        cfg.warnings.append(
            "không đọc được luật doanh thu một đơn (giảm giá / VAT / phí giao), dùng mặc định"
        )
    m = re.search(r"tuần (?:bắt đầu|tính từ) (" + "|".join(_WEEKDAYS) + ")", norm)
    if m:
        cfg.week_start = _WEEKDAYS.index(m.group(1))
    else:
        cfg.warnings.append("không đọc được ngày đầu tuần, dùng mặc định thứ Hai")


# ---------------------------------------------------------------- nhãn trạng thái đơn

DEFAULT_STATUS_LABELS = ("hoàn thành", "huỷ", "trả hàng", "đang giao")


def status_labels(text: str) -> list[str]:
    """Tên các nhãn trạng thái đơn đọc từ rulebook: các cụm in đậm trên dòng luật nói đơn có "một trong … trạng
    thái". Không đọc được thì dùng 4 nhãn mặc định."""
    for line in text.splitlines():
        n = normalize(line)
        if "trạng thái" in n and "một trong" in n:
            tail = line.split("một trong", 1)[1]
            labels = [normalize(x) for x in re.findall(r"\*\*(.+?)\*\*", tail)]
            if len(labels) >= 2:
                return labels
    return list(DEFAULT_STATUS_LABELS)


# ---------------------------------------------------------------- chọn luật liên quan


@dataclass
class Rule:
    section: str  # "§2 Thời gian" hoặc "§6 Theo ngành / Cà phê"
    name: str  # cột trái của dòng bảng
    text: str  # nguyên văn dòng luật (markdown)
    industry: str | None = None  # tên ngành nếu thuộc §6

    @property
    def keys(self) -> set[str]:
        return set(key(self.name).split())


def split_rules(rulebook: str) -> list[Rule]:
    """Mỗi dòng bảng của rulebook là một luật, kèm tên mục (và tên ngành nếu thuộc mục theo ngành)."""
    rules, section, sub = [], "", None
    for line in rulebook.splitlines():
        s = line.strip()
        if s.startswith("## "):
            section, sub = s[3:].strip(), None
            continue
        if s.startswith("### "):
            sub = s[4:].strip()
            continue
        if not s.startswith("|") or re.fullmatch(r"\|[-| :]+\|?", s):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 2 or normalize(cells[0]) in ("luật", "cách nói", "cách ghi", "type", "từ"):
            continue
        sec = f"{section} / {sub}" if sub else section
        rules.append(Rule(sec, cells[0], s, sub))
    return rules


_STOP = frozenset(
    {"la", "cua", "va", "cho", "cac", "mot", "nhung", "trong", "theo", "khi", "thi", "co", "khong",
     "duoc", "bao", "nhieu", "nao", "gi", "hay", "hoac", "ve", "den", "tu", "nhat", "cung", "so",
     "ngay", "thang", "nam", "don", "hang", "ky"}
)  # fmt: skip


def _ngrams(words: list[str]) -> set[str]:
    return {w for w in words if w not in _STOP and len(w) > 1} | {
        f"{a} {b}" for a, b in itertools.pairwise(words)
    }


def _idf(rules: list[Rule]) -> dict[str, float]:
    import math

    df: dict[str, int] = {}
    for r in rules:
        for g in _ngrams(key(r.text).split()):
            df[g] = df.get(g, 0) + 1
    n = len(rules)
    return {g: math.log((n + 1) / (c + 0.5)) for g, c in df.items()}


_IDF_CACHE: dict[int, tuple[list[Rule], dict[str, float]]] = {}


def select_rules(
    question: str, rules: list[Rule], industry: str | None = None, k: int = 6
) -> list[Rule]:
    """Chọn k luật trùng từ khoá (bỏ dấu, có cụm 2 từ) với câu hỏi nhiều nhất.

    Mỗi từ / cụm trùng được cộng theo IDF trên tập luật (từ hiếm như "tết", "ca chiều" nặng hơn
    "doanh thu"); trùng ở tên luật nặng gấp 3 trùng ở nội dung; luật thuộc ngành của shop được cộng điểm,
    luật của ngành khác bị trừ điểm.
    """
    cached = _IDF_CACHE.get(id(rules))
    if cached is None or cached[0] is not rules:
        cached = (rules, _idf(rules))
        _IDF_CACHE[id(rules)] = cached
    idf = cached[1]
    q = _ngrams(key(question).split())
    scored = []
    for i, r in enumerate(rules):
        name = _ngrams(key(r.name).split())
        body = _ngrams(key(r.text).split())
        s = sum(3.0 * idf.get(g, 0) for g in q & name) + sum(idf.get(g, 0) for g in q & body - name)
        if r.industry and s > 0:
            s = s * 1.5 if industry and key(r.industry) == key(industry) else s * 0.5
        if s > 0:
            scored.append((s, -i, r))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [r for *_, r in scored[:k]]


_INDUSTRY_HINTS = {
    "Cà phê": ["ca phe", "tra sua", "topping", "bac xiu", "tra dao", "sinh to"],
    "Nhà thuốc": ["nha thuoc", "hoat chat", "ke don", "vien nang", "siro", "han su dung"],
    "Vật liệu xây dựng": ["xi mang", "thep", "gach", "dot giao", "cat xay", "vat lieu"],
    "Mẹ và bé": ["sua bot", "bim", "ta dan", "me va be", "so sua"],
    "Sửa xe": ["sua xe", "thay nhot", "bien so", "va ruot", "bugi", "luot xe"],
    "Cửa hàng điện thoại": ["dien thoai", "iphone", "samsung", "tra gop", "imei", "may cu"],
}


def guess_industry(
    readme: str, rules: list[Rule], sample_text: str = "", min_hits: int = 3
) -> str | None:
    """Đoán ngành của shop trong các ngành rulebook nêu ở mục theo ngành (README + tên hàng).

    Đếm cụm từ đặc trưng (khớp nguyên từ, đã bỏ dấu); ngành có tên trong rulebook nhưng không có gợi ý
    sẵn thì dùng chính tên ngành. Dưới `min_hits` lần khớp thì không đoán (shop thuộc ngành chung).
    """
    names = {r.industry for r in rules if r.industry}
    text = " " + key(readme + " " + sample_text) + " "
    best, best_s = None, 0
    for ind in sorted(names):
        hints = _INDUSTRY_HINTS.get(ind, []) + [key(ind)]
        s = sum(len(re.findall(rf"(?<![a-z0-9]){re.escape(h)}(?![a-z0-9])", text)) for h in hints)
        if s > best_s:
            best, best_s = ind, s
    return best if best_s >= min_hits else None
