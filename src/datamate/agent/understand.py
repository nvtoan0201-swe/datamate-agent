"""Hiểu câu hỏi trước khi sinh code: kỳ hỏi, thông tin cần, cách đọc chuẩn, kiểu đáp án.

Hai phần:
- Tất định: tách kỳ ghi rõ trong câu ("tháng 2/2025", "quý 3 năm 2025") bằng regex, so với khoảng ngày dữ liệu.
  Kỳ nằm hoàn toàn trước ngày đầu dữ liệu (hoặc sau ngày cuối nhưng không sau hôm nay) → unanswerable (§7).
- Một lần gọi LLM (có README, thẻ schema, tên các luật): viết lại câu hỏi bằng tiếng Việt chuẩn, kiểu đáp án,
  các kỳ (biểu thức `dm.*`), các thông tin cần và cột chứa thông tin đó, loại câu hỏi. Mọi thứ LLM trả về đều
  được code kiểm tra lại: số trong câu viết lại phải khớp câu gốc, tham số của kỳ phải có trong câu hỏi, cột phải
  tồn tại. Phần không qua kiểm tra bị bỏ, không dùng.

Kết quả LLM chỉ là tín hiệu: solver chỉ trả unanswerable vì "thiếu thông tin" khi bước sinh code cũng không ra
đáp án (xem `solver.decide`).
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field

import pandas as pd

from datamate.agent.prompts import normalize_time
from datamate.agent.semantic import parse_json
from datamate.answer import ANSWER_TYPES
from datamate.biz import Biz, RuleDisabled, Unanswerable
from datamate.text import key

# ---------------------------------------------------------------- kỳ ghi rõ trong câu (tất định)

_MONTH = re.compile(r"\btháng\s+(\d{1,2})\s*(?:/|năm\s+|nam\s+)\s*((?:19|20)\d\d)\b", re.IGNORECASE)
_QUARTER = re.compile(r"\bquý\s+([1-4])\s*(?:/|năm\s+|nam\s+)\s*((?:19|20)\d\d)\b", re.IGNORECASE)
_PREV_MONTH = re.compile(
    r"tháng\s+(liền\s+)?trước|thang\s+(lien\s+)?truoc|so với tháng|so vs thang"
)


def explicit_periods(question: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Các tháng / quý ghi rõ năm trong câu hỏi: (nhãn, ngày đầu, ngày cuối)."""
    q = normalize_time(question)
    q = re.sub(r"\bthang\b", "tháng", q, flags=re.IGNORECASE)
    q = re.sub(r"\bquy\b", "quý", q, flags=re.IGNORECASE)
    out = []
    for m in _MONTH.finditer(q):
        mo, y = int(m[1]), int(m[2])
        if 1 <= mo <= 12:
            s = pd.Timestamp(y, mo, 1)
            out.append((f"tháng {mo}/{y}", s, s + pd.offsets.MonthEnd(0)))
    for m in _QUARTER.finditer(q):
        qq, y = int(m[1]), int(m[2])
        s = pd.Timestamp(y, 3 * qq - 2, 1)
        out.append((f"quý {qq}/{y}", s, s + pd.offsets.QuarterEnd(0)))
    # "tháng M/Y … so với tháng liền trước": cần cả tháng trước đó.
    months = [p for p in out if p[0].startswith("tháng")]
    if len(months) == 1 and _PREV_MONTH.search(key(q) + " " + q.lower()):
        s = months[0][1] - pd.offsets.MonthBegin(1)
        out.append((f"tháng {s.month}/{s.year} (tháng liền trước)", s, s + pd.offsets.MonthEnd(0)))
    return out


def out_of_range(start: pd.Timestamp, end: pd.Timestamp, data_start, data_end, today) -> bool:
    """Kỳ nằm hoàn toàn ngoài khoảng có dữ liệu. Kỳ sau hôm nay (hạn dùng, kế hoạch) không tính là ngoài."""
    today = pd.Timestamp(today)
    if start > today:
        return False
    lo = pd.Timestamp(data_start) if data_start is not None else None
    hi = pd.Timestamp(data_end) if data_end is not None else None
    if lo is not None and end < lo:
        return True
    return bool(hi is not None and start > hi)


def period_unanswerable(question: str, data_start, data_end, today) -> str | None:
    """Lý do nếu một kỳ ghi rõ trong câu hỏi nằm hoàn toàn ngoài dữ liệu, ngược lại None."""
    for label, s, e in explicit_periods(question):
        if out_of_range(s, e, data_start, data_end, today):
            return (
                f"kỳ {label} nằm ngoài khoảng dữ liệu "
                f"{pd.Timestamp(data_start):%Y-%m-%d} → {pd.Timestamp(data_end):%Y-%m-%d}"
            )
    return None


# ---------------------------------------------------------------- LLM đọc câu hỏi

SYSTEM = """\
Bạn đọc câu hỏi của chủ một cửa hàng nhỏ ở Việt Nam về dữ liệu bán hàng của shop. Bạn KHÔNG trả lời câu hỏi; \
bạn chỉ phân tích câu hỏi cần gì và dữ liệu của shop có những thông tin đó không.

Trả về DUY NHẤT một object JSON:
{{
  "rewritten": "câu hỏi viết lại bằng tiếng Việt chuẩn, đủ dấu, không viết tắt, giữ nguyên mọi số, ngày, tên \
mặt hàng, tên khách; KHÔNG đổi 'tháng này', 'hôm nay', 'tuần rồi' thành ngày cụ thể",
  "answer_type": "number | percent | string | date | boolean | list | table",
  "periods": ["các kỳ thời gian của giao dịch cần lọc, viết bằng hàm dm (xem dưới); [] nếu không có"],
  "needs": [{{"info": "một loại thông tin cần có trong dữ liệu", "column": "Tên bảng.Tên cột" hoặc null}}],
  "kind": "data | subjective | outside"
}}

- periods: dm.month(m, y), dm.month(m) (không ghi năm), dm.quarter(q, y), dm.year(y), dm.day('YYYY-MM-DD'), \
dm.between('YYYY-MM-DD', 'YYYY-MM-DD'), dm.this_month(), dm.last_month(), dm.this_week(), dm.last_week(), \
dm.today_p(), dm.yesterday(), dm.last_n_days(n), dm.ytd(), dm.tet(y), dm.until('YYYY-MM-DD') (số dư, tồn tính \
đến hết ngày). Câu so sánh thì ghi cả kỳ được so (ví dụ tháng liền trước). Không ghi khoảng hạn dùng hay ngày \
trong tương lai.
- needs: các thông tin THÔ câu hỏi cần (ngày bán, tiền, tên mặt hàng, trạng thái đơn, tên khách, phương thức \
thanh toán, kênh bán, lý do trả hàng, tồn kho, giá nhập, công nợ, hạn dùng…). Khái niệm do luật định nghĩa thì \
tách thành thông tin thô nó dùng. Mỗi thông tin: ghi đúng tên bảng và tên cột chứa nó trong thẻ schema, hoặc \
null nếu README và các bảng của shop KHÔNG có thông tin đó. Thông tin suy ra được thì KHÔNG phải null: tháng, \
tuần, thứ, ca sáng / chiều, giờ cao điểm tính từ cột ngày giờ; size, dung tích, quy cách nằm trong tên mặt hàng; \
giá cũ / giá mới, mốc đổi giá ghi ngay trong câu hỏi. Chỉ ghi null khi không cột nào chứa và không suy ra được \
(ví dụ địa chỉ, giới tính, kênh bán, lý do trả hàng khi shop không ghi).
- kind: "data" nếu tính được từ dữ liệu bán hàng (dù shop thiếu cột); "subjective" nếu hỏi ý kiến, đánh giá \
cảm tính, tư vấn mà luật không định nghĩa; "outside" nếu hỏi điều nằm ngoài dữ liệu của shop (thời tiết, đối \
thủ, dự đoán tương lai).

# Các khái niệm luật (rulebook) có định nghĩa
{rule_names}

# README của shop
{readme}

# Thẻ schema của shop
{card}"""

# Khối thêm vào prompt khi shop có semantic graph v2: câu hỏi cần những khả năng dữ liệu nào.
CAPABILITY_BLOCK = """

# Khả năng dữ liệu của shop (đã kiểm tra trên dữ liệu: verified = có chắc, uncertain = chưa chắc, absent = không có)
{caps}
Thêm vào JSON khoá "capabilities": [tên các khả năng ở trên mà câu hỏi CẦN để trả lời]; chỉ dùng tên trong danh sách."""

CAPABILITY_DESC = {
    "order_revenue": "đơn hàng: ngày đặt, trạng thái, doanh thu, số đơn",
    "time_of_day": "giờ tạo đơn trong ngày (ca sáng / chiều, giờ cao điểm)",
    "customer_identity": "khách hàng (tên / mã khách)",
    "product_identity": "mặt hàng trên từng dòng bán",
    "staff": "nhân viên bán / người làm",
    "debt_payments": "các lần thu tiền, công nợ",
    "returns": "đơn / phiếu trả hàng",
    "cost": "giá vốn",
    "units": "số lượng, đơn vị tính, quy đổi",
}

USER = 'Hôm nay (as_of_date): {as_of}\nCâu hỏi: "{question}"\n\nTrả về JSON.'

_PERIOD_FUNCS = {
    "month",
    "quarter",
    "year",
    "day",
    "between",
    "this_month",
    "last_month",
    "this_week",
    "last_week",
    "today_p",
    "yesterday",
    "last_n_days",
    "ytd",
    "tet",
    "until",
}
_CALL = re.compile(r"^\s*dm\.(\w+)\((.*)\)\s*$")


@dataclass
class Understanding:
    rewritten: str | None = None  # câu viết lại (None nếu LLM lỗi hoặc không qua kiểm tra)
    answer_type: str | None = None
    periods: list[str] = field(default_factory=list)  # biểu thức kỳ đã qua kiểm tra
    period_out: str | None = (
        None  # kỳ ghi rõ trong câu (regex) nằm hoàn toàn ngoài dữ liệu: chắc chắn
    )
    period_out_llm: str | None = None  # kỳ do LLM tách ra nằm ngoài dữ liệu: chỉ là tín hiệu
    needs: list[dict] = field(
        default_factory=list
    )  # {"info", "column"} — column None nếu shop không có
    capabilities: list[str] = field(default_factory=list)  # khả năng dữ liệu câu hỏi cần (graph v2)
    missing: list[str] = field(default_factory=list)  # thông tin shop không có
    kind: str | None = None
    warnings: list[str] = field(default_factory=list)
    raw: str | None = None

    @property
    def unanswerable_signal(self) -> str | None:
        """Lý do LLM cho rằng câu không trả lời được (chưa phải kết luận).

        Chỉ dùng "thiếu thông tin". Kỳ do LLM tách và `kind` chỉ ghi vào trace: đo trên dev: kỳ của LLM không bắt thêm câu nào so với regex mà đọc nhầm "quý 4" thành tháng 4; `kind` bắt
        thêm 1 câu nhưng nhầm 4/200 câu trả lời được.
        """
        if self.missing:
            return f"shop không ghi: {', '.join(self.missing)}"
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["unanswerable_signal"] = self.unanswerable_signal
        return d


def _numbers(s: str) -> set[int]:
    return {int(x) for x in re.findall(r"\d+", s)}


def check_rewrite(question: str, rewritten: str) -> str | None:
    """Lỗi nếu câu viết lại làm mất / thêm số hoặc năm so với câu gốc, ngược lại None."""
    if not rewritten or not rewritten.strip():
        return "rỗng"
    orig = _numbers(normalize_time(question))
    new = _numbers(rewritten)
    if orig - new:
        return f"mất số {sorted(orig - new)}"
    extra = new - orig
    years = {n for n in extra if 1900 <= n <= 2100}
    if years:
        return f"thêm năm {sorted(years)}"
    if any(n > 31 for n in extra):
        return f"thêm số {sorted(extra)}"
    return None


def eval_period(expr: str, dm: Biz, question: str):
    """Tính biểu thức kỳ `dm.f(args)` nếu f trong danh sách cho phép và mọi số trong args có trong câu hỏi."""
    m = _CALL.match(expr or "")
    if not m or m[1] not in _PERIOD_FUNCS:
        raise ValueError(f"không phải hàm kỳ: {expr!r}")
    raw = m[2].strip()
    args = ast.literal_eval(f"({raw},)") if raw else ()
    qnums = _numbers(normalize_time(question)) | _numbers(str(dm.today.date()))
    for a in args:
        if not isinstance(a, (int, str)):
            raise TypeError(f"tham số lạ: {a!r}")
        if not _numbers(str(a)) <= qnums:
            raise ValueError(f"tham số {a!r} không có trong câu hỏi")
    return getattr(dm, m[1])(*args)


def _resolve_column(col: str | None, tables: dict) -> str | None:
    """'Bảng.Cột' → tên chuẩn nếu cột có thật (so bằng khoá bỏ dấu), ngược lại None.

    Nhận cả khi LLM bỏ tiền tố tên bảng ('DuLieuXuat.X' ghi thành 'X') hoặc ghi nhầm bảng nhưng tên cột chỉ có
    ở đúng một bảng: điều cần biết là shop có thông tin đó.
    """
    if not col or not isinstance(col, str):
        return None
    kc = key(col)
    full, by_col = [], []
    for t, df in tables.items():
        for c in df.columns:
            kcol = key(c)
            if not kc.endswith(kcol):
                continue
            prefix = kc[: len(kc) - len(kcol)].strip()
            if prefix in (key(t), key(str(t).split(".")[-1])):
                full.append(f"{t}.{c}")
            elif kc == kcol or prefix:
                by_col.append(f"{t}.{c}")
    if full:
        return full[0]
    return by_col[0] if len({b.split(".")[-1] for b in by_col}) == 1 and by_col else None


_TIME_INFO = re.compile(r"^(?:ngay|thang|tuan|nam|thu|quy|thoi gian|ngay ban|ngay gio)$")
_SHIFT_INFO = re.compile(r"\b(?:ca|gio|khung gio|cao diem)\b")


def derivable_from_time(info: str, tables: dict) -> bool:
    """Thông tin "thiếu" mà thật ra lấy được từ cột ngày giờ: chỉ là mốc thời gian, hoặc ca / giờ khi shop có cột
    ngày kèm giờ trong ngày."""
    k = key(info)
    if _TIME_INFO.match(k):
        return True
    if not _SHIFT_INFO.search(k):
        return False
    for df in tables.values():
        for c in df.columns:
            s = df[c]
            if pd.api.types.is_datetime64_any_dtype(s):
                v = s.dropna()
                if len(v) and (v.dt.hour != 0).any():
                    return True
    return False


def capability_lines(graph) -> str:
    out = []
    for name, c in graph.capabilities.items():
        desc = CAPABILITY_DESC.get(name, c.reason or "")
        out.append(f"- {name}: {c.status}" + (f" ({desc})" if desc else ""))
    return "\n".join(out)


def build_messages(
    question: str, as_of: str, readme: str, card: str, rule_names: list[str], graph=None
) -> list[dict]:
    system = SYSTEM.format(
        rule_names=", ".join(rule_names) or "(không có)",
        readme=readme.strip(),
        card=card.strip(),
    )
    if graph is not None and graph.capabilities:
        system += CAPABILITY_BLOCK.format(caps=capability_lines(graph))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": USER.format(as_of=as_of, question=question)},
    ]


def understand(llm, ctx, q: dict, use_llm: bool = True, use_caps: bool = True) -> Understanding:
    """Phân tích câu hỏi `q` của shop `ctx` (ShopContext). Không bao giờ raise. `use_caps`: hỏi thêm khả năng dữ liệu
    câu hỏi cần (khi shop có graph v2)."""
    u = Understanding()
    u.period_out = period_unanswerable(q["question"], ctx.data_start, ctx.data_end, q["as_of_date"])
    if u.period_out or not use_llm or llm is None:
        return u
    names = sorted({r.name for r in ctx.rule_index}) if ctx.rule_index else []
    graph = getattr(ctx, "graph", None) if use_caps else None
    msgs = build_messages(q["question"], q["as_of_date"], ctx.readme, ctx.card, names, graph=graph)
    try:
        text, _ = llm.chat(msgs, max_tokens=600, json_mode=True)
    except Exception as e:  # noqa: BLE001 — lỗi server: chỉ dùng phần tất định
        u.warnings.append(f"LLM lỗi: {e!r}")
        return u
    u.raw = text
    d = parse_json(text)
    if d is None:
        u.warnings.append("không đọc được JSON")
        return u

    rw = d.get("rewritten")
    err = check_rewrite(q["question"], rw) if isinstance(rw, str) else "không phải chuỗi"
    if err:
        u.warnings.append(f"bỏ câu viết lại: {err}")
    else:
        u.rewritten = rw.strip()

    t = d.get("answer_type")
    if isinstance(t, str) and t.strip() in ANSWER_TYPES and t.strip() != "unanswerable":
        u.answer_type = t.strip()

    kind = d.get("kind")
    if kind in ("data", "subjective", "outside"):
        u.kind = kind
    if graph is not None and isinstance(d.get("capabilities"), list):
        known = set(graph.capabilities)
        u.capabilities = [c for c in d["capabilities"] if isinstance(c, str) and c in known]

    dm = Biz(
        q["as_of_date"],
        ctx.data_start,
        ctx.data_end,
        ctx.calendar_csv,
        rules=ctx.rules,
        tables=ctx.tables,
    )
    for expr in d.get("periods") or []:
        if not isinstance(expr, str):
            continue
        try:
            p = eval_period(expr, dm, q["question"])
        except (ValueError, SyntaxError, TypeError, RuleDisabled, Unanswerable, RuntimeError) as e:
            u.warnings.append(f"bỏ kỳ {expr!r}: {e}")
            continue
        u.periods.append(expr.strip())
        if u.period_out_llm is None and out_of_range(
            p.start, p.end, ctx.data_start, ctx.data_end, q["as_of_date"]
        ):
            u.period_out_llm = f"kỳ {expr.strip()} ({p.start:%Y-%m-%d} → {p.end:%Y-%m-%d}) nằm ngoài khoảng dữ liệu"

    for n in d.get("needs") or []:
        if not isinstance(n, dict) or not isinstance(n.get("info"), str):
            continue
        info, col = n["info"].strip(), n.get("column")
        if col is None or (isinstance(col, str) and col.strip().lower() in ("", "null", "none")):
            u.needs.append({"info": info, "column": None})
            if derivable_from_time(info, ctx.tables):
                u.warnings.append(f"bỏ thiếu {info!r}: suy ra được từ cột ngày giờ")
            else:
                u.missing.append(info)
            continue
        real = _resolve_column(col, ctx.tables)
        if real is None:
            # LLM nêu một cột không có thật: không đủ căn cứ nói là thiếu hay có.
            u.warnings.append(f"cột không có thật: {col!r} ({info})")
            u.needs.append({"info": info, "column": col, "verified": False})
            continue
        u.needs.append({"info": info, "column": real})
    return u
