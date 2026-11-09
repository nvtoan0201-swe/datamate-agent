"""Giải một câu: K biến thể prompt → code → chạy → sửa lỗi → ép kiểu → bỏ phiếu → kiểm tra hợp lý → trả lời
hoặc abstain.

Ngưỡng đồng thuận tách theo `path` của câu (generalization-plan §5.2):
- `helper`: code gọi hàm nghiệp vụ đơn hàng đã có test → đáp án nhiều phiếu nhất (1/3, chọn bằng CV theo shop);
- `free`: code tự viết pandas (khái niệm mới, ngành mới) → cần mọi biến thể đồng ý;
- shop có ánh xạ không dùng được (`mapping_ok = False`) → cần mọi biến thể và phải qua `sanity`.
Unanswerable do code báo "không có cột / bảng" chỉ nhận khi mọi biến thể đồng ý (§5.3).
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from datamate.agent.lint import (
    temporal_issues,
    temporal_repair_message,
    threshold_issues,
    threshold_repair_message,
    time_columns,
)
from datamate.agent.llm import LLM
from datamate.agent.precheck import classify
from datamate.agent.prompts import build_messages, extract_code, repair_message
from datamate.agent.sandbox import run_code
from datamate.agent.understand import Understanding, understand
from datamate.agent.verifier import VerifierConfig, verify
from datamate.answer import ANSWER_TYPES, FormatError, answer_key, coerce
from datamate.biz import Biz
from datamate.rules import Rule, RuleConfig
from datamate.text import key


@dataclass
class ShopContext:
    biz_id: str
    tables: dict
    readme: str
    card: str
    data_start: object
    data_end: object
    calendar_csv: str
    rulebook: str
    rules: RuleConfig = field(default_factory=RuleConfig)
    mapping: dict | None = None  # ánh xạ đã qua kiểm tra (None nếu không có hoặc không dùng được)
    mapping_summary: str = ""
    mapping_ok: bool = False
    rule_index: list[Rule] = field(default_factory=list)
    industry: str | None = None
    snapshot_cols: list[str] = field(default_factory=list)
    total_revenue: float | None = None  # doanh thu cả dữ liệu theo ánh xạ (để chặn số tiền vô lý)
    graph: object = None  # SemanticGraph v2 (None khi chạy --semantic v1)
    raw_summary: str = ""  # ánh xạ thô cho cơ chế giải độc lập (pipeline v2)
    _texts: set | None = None
    _time_cols: set | None = None

    def time_cols(self) -> set[str]:
        """Cột ngày giờ có giờ thật (cho kiểm tra mốc hết ngày, agent/lint.py)."""
        if self._time_cols is None:
            self._time_cols = time_columns(self.tables)
        return self._time_cols

    def text_values(self) -> set[str]:
        """Khoá của mọi giá trị chữ trong dữ liệu shop (để kiểm tra đáp án tên có lấy từ dữ liệu)."""
        if self._texts is None:
            vals = set()
            for df in self.tables.values():
                for c in df.columns:
                    s = df[c]
                    if s.dtype == object or pd.api.types.is_string_dtype(s):
                        vals |= {key(v) for v in s.dropna().astype(str).unique()[:50000]}
            self._texts = vals
        return self._texts


@dataclass
class SolverConfig:
    k: int = 1  # số biến thể prompt
    min_agree: int | None = None  # ghi đè: số lời giải phải trùng đáp án cho mọi nhóm
    # Ngưỡng theo nhóm, viết theo K = 3 và co giãn theo K thật: need = ceil(n / 3 * K).
    # Chọn bằng CV theo shop (scripts/tune_thresholds.py):
    # helper 1/3 (đa số tương đối sau khi loại lời giải không qua sanity), free 3/3. Shop không có ánh xạ
    # chưa có số đo, giữ 3/3 theo kế hoạch.
    agree_helper: int = 1
    agree_free: int = 3
    agree_nomap: int = 3
    use_sanity: bool = True
    # Unanswerable do code tự báo (không có cột / bảng, không phải kỳ ngoài dữ liệu): 'abstain' (mặc định:
    # tỉ lệ nền unanswerable ở train chỉ 2,3 %, phần lớn đã do precheck / kỳ bắt; dev 29/10: 0/1 đúng) hoặc
    # 'all_agree' (nhận khi mọi biến thể đồng ý, §5.3 kế hoạch).
    code_unanswerable: str = "abstain"
    near_miss: str = "abstain"  # câu diễn đạt lệch luật: 'abstain' | 'unanswerable'
    max_repairs: int = 2
    exec_timeout: float = 60
    # Bước hiểu câu hỏi (agent/understand.py): một lần gọi LLM mỗi câu. Kỳ ghi rõ nằm ngoài dữ liệu (regex) luôn
    # bật, không cần LLM.
    understand: bool = True
    # LLM báo câu thiếu thông tin nhưng code vẫn ra đáp án đủ phiếu: 'abstain' hoặc 'answer' (tin code).
    # Mặc định tin code: tín hiệu nhầm 3–5 % câu trả lời được (dev 03/11/2026), abstain những câu đó mất nhiều hơn
    # số câu unanswerable bắt thêm; tín hiệu chỉ đổi câu code không ra đáp án (vốn 0 điểm) thành unanswerable.
    signal_vs_answer: str = "answer"
    # Pipeline: 'v1' (biến thể prompt + bỏ phiếu `decide`) | 'v2' (cơ chế giải khác nhau + verifier).
    pipeline: str = "v1"
    mechanisms: tuple = ("helper", "pandas")  # chạy mỗi câu
    third: str | None = "sql"  # chạy thêm khi hai cơ chế đầu bất đồng / verifier nghi
    understand_caps: bool = True  # bước hiểu câu hỏi nêu khả năng dữ liệu câu hỏi cần (graph v2)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)


@dataclass
class Candidate:
    variant: int
    typ: str | None = None
    answer: object = None
    key: str | None = None
    reason: str | None = None
    error: str | None = None
    code: str | None = None
    attempts: list = field(default_factory=list)
    mechanism: str = "helper"  # cơ chế giải (pipeline v2): helper | pandas | sql
    rows: int | None = None  # kết quả trung gian ROWS
    period: tuple | None = None  # kết quả trung gian PERIOD
    seconds: float = 0.0
    lint: list = field(default_factory=list)  # lỗi tĩnh còn lại trong code cuối (mốc hết ngày…)

    @property
    def path(self) -> str:
        return classify_path(self.code)


# Hàm nghiệp vụ đơn hàng: code cuối gọi một trong các hàm này thì câu thuộc nhóm `helper`.
_HELPER_CALL = re.compile(
    r"\bdm\.(orders|sold_lines|revenue|count_orders|count_cancelled|cancel_rate|aov|daily_revenue"
    r"|payments|customers|base_qty|loyal_customers|big_spenders|fickle_customers|popular_items"
    r"|busiest_days|peak_hour|hardest_workers)\s*\("
)


def classify_path(code: str | None) -> str:
    """'helper' nếu code gọi hàm nghiệp vụ đơn hàng của `dm`, ngược lại 'free'."""
    return "helper" if code and _HELPER_CALL.search(code) else "free"


_TYPE_HINTS = [
    (r"%|phần trăm|phan tram|tỉ lệ|ti le|tỷ lệ|ty le|tỉ trọng|ti trong", "percent"),
    (r"tháng nào|thang nao", "date"),
    (r"ngày nào|ngay nao", "date"),
]


def guess_type(question: str) -> str | None:
    q = question.lower()
    for pat, t in _TYPE_HINTS:
        if re.search(pat, q):
            return t
    return None


def snapshot_safe_tables(tables: dict, snapshot_cols: list[str], as_of, data_end) -> dict:
    """Hôm nay sớm hơn ngày cuối dữ liệu: bỏ các cột chỉ đúng tại ngày cuối (còn nợ, tồn cuối…) khỏi bảng đưa cho
    code, để code không dùng nhầm (snapshot safety). Ngược lại trả nguyên `tables`."""
    if (
        not snapshot_cols
        or data_end is None
        or pd.Timestamp(as_of) >= pd.Timestamp(data_end).normalize()
    ):
        return tables
    drop: dict[str, list[str]] = {}
    for sc in snapshot_cols:
        for t in sorted(tables, key=len, reverse=True):
            if sc.startswith(t + ".") and sc[len(t) + 1 :] in tables[t].columns:
                drop.setdefault(t, []).append(sc[len(t) + 1 :])
                break
    if not drop:
        return tables
    return {n: (df.drop(columns=drop[n]) if n in drop else df) for n, df in tables.items()}


def _period_unanswerable(reason: str | None) -> bool:
    """Unanswerable do kỳ hỏi nằm ngoài dữ liệu (Biz.check) — chắc chắn, khác với "không có cột"."""
    return bool(reason) and "nằm ngoài khoảng dữ liệu" in reason


def _run_candidate(
    llm: LLM,
    ctx: ShopContext,
    q: dict,
    variant: int,
    cfg: SolverConfig,
    u: Understanding | None = None,
    mechanism: str | None = None,
) -> Candidate:
    t_start = time.time()
    cand = Candidate(variant, mechanism=mechanism or "helper")
    dm = Biz(
        q["as_of_date"],
        ctx.data_start,
        ctx.data_end,
        ctx.calendar_csv,
        rules=ctx.rules,
        mapping=ctx.mapping if ctx.mapping_ok else None,
        tables=ctx.tables,
        graph=ctx.graph if ctx.mapping_ok else None,
    )
    tables = snapshot_safe_tables(ctx.tables, ctx.snapshot_cols, q["as_of_date"], ctx.data_end)
    msgs = build_messages(
        ctx.rulebook,
        ctx.readme,
        ctx.card,
        q["question"],
        q["as_of_date"],
        variant,
        rules=ctx.rules,
        mapping_summary=ctx.mapping_summary if ctx.mapping_ok else "",
        rule_index=ctx.rule_index,
        industry=ctx.industry,
        data_end=ctx.data_end,
        snapshot_cols=ctx.snapshot_cols,
        reading=u.rewritten if u else None,
        mechanism=mechanism,
        raw_summary=ctx.raw_summary,
    )
    try:
        return _attempts(llm, ctx, q, cfg, u, cand, dm, tables, msgs)
    finally:
        cand.seconds = round(time.time() - t_start, 2)


def _attempts(llm, ctx, q, cfg, u, cand, dm, tables, msgs) -> Candidate:
    for attempt in range(cfg.max_repairs + 1):
        try:
            text, usage = llm.chat(msgs)
        except Exception as e:  # noqa: BLE001 — lỗi server LLM: dừng biến thể này
            cand.error = f"LLM lỗi: {e!r}"
            return cand
        code = extract_code(text)
        rec = {"usage": usage, "output": text}
        cand.attempts.append(rec)
        if not code:
            err = "Không thấy khối code ```python```."
        else:
            cand.code = code
            res = run_code(code, tables, dm, timeout=cfg.exec_timeout, mechanism=cand.mechanism)
            cand.rows, cand.period = res.rows, res.period
            rec["exec"] = {
                "ok": res.ok,
                "error": res.error,
                "stdout": res.stdout[-1000:],
                "typ": res.typ,
                "unanswerable": res.unanswerable,
                "value": repr(res.value)[:500],
                "rows": res.rows,
                "period": res.period,
            }
            if res.ok and res.unanswerable is not None:
                cand.typ, cand.reason, cand.key = "unanswerable", res.unanswerable, "UNANSWERABLE"
                return cand
            temporal = temporal_issues(code, ctx.time_cols()) if res.ok else []
            thresholds = threshold_issues(code, q["question"]) if res.ok else []
            cand.lint = temporal + thresholds
            rec["lint"] = cand.lint
            if res.ok and cand.lint and attempt < cfg.max_repairs:
                err = "\n".join(
                    ([temporal_repair_message(temporal)] if temporal else [])
                    + ([threshold_repair_message(thresholds)] if thresholds else [])
                )
            elif res.ok:
                typ = (
                    res.typ
                    if res.typ in ANSWER_TYPES
                    else (u.answer_type if u and u.answer_type else guess_type(q["question"]))
                )
                if typ is None:
                    err = f"TYPE không hợp lệ ({res.typ!r}). Gán TYPE là một trong {list(ANSWER_TYPES)}."
                else:
                    try:
                        ans = coerce(res.value, typ)
                        cand.typ, cand.answer, cand.key = typ, ans, answer_key(typ, ans)
                        return cand
                    except FormatError as e:
                        err = f"ANSWER = {repr(res.value)[:300]} không đúng format của type {typ}: {e}"
            else:
                err = res.error or "lỗi không rõ"
                if "KeyError" in err or "AttributeError" in err:
                    cols = "\n".join(
                        f"- tables[{n!r}]: {list(df.columns)}" for n, df in tables.items()
                    )
                    err += f"\nCác bảng và cột hiện có:\n{cols}"
                    if tables is not ctx.tables:
                        err += (
                            "\nCác cột chỉ đúng tại ngày cuối dữ liệu đã bị bỏ vì hôm nay sớm hơn ngày đó: "
                            "tính lại từ phát sinh đến hết hôm nay (dm.balance_at)."
                        )
        cand.error = err
        msgs = msgs + [{"role": "assistant", "content": text}, repair_message(err)]
    return cand


# ---------------------------------------------------------------- kiểm tra hợp lý (§5.1)

_Q_AMOUNT = re.compile(
    r"doanh thu|dthu|\bban\b|thu ve|\bdon\b|so luong|tien hang|t[iy] le|t[iy] trong|chiem"
)
_Q_MONEY = re.compile(r"doanh thu|dthu|tien|\bno\b|thu ve|ban duoc|gia tri")
_Q_CHANGE = re.compile(r"tang|giam|so voi|thay doi|chenh|hon|kem|lech")
_Q_SHARE = re.compile(r"t[iy] le|t[iy] trong|chiem|phan tram cua|bao nhieu phan tram")
_Q_NEG_OK = re.compile(r"lai|lo|chenh|tang|giam|so voi|thay doi|hon|kem|lech|con no|ton|nhap tru")


def sanity(q: dict, typ: str | None, answer, ctx: ShopContext) -> list[str]:
    """Các dấu hiệu đáp án nhiều khả năng sai (lọc nhầm tên / trạng thái, sai đơn vị, tên không có thật)."""
    if typ in (None, "unanswerable"):
        return []
    flags = []
    kq = key(q["question"])
    if typ == "number" and isinstance(answer, (int, float)):
        if answer == 0 and _Q_AMOUNT.search(kq) and not _Q_CHANGE.search(kq):
            flags.append(
                "số bằng 0 cho câu hỏi doanh thu / số lượng (có thể lọc sai tên, trạng thái)"
            )
        if answer < 0 and not _Q_NEG_OK.search(kq):
            flags.append("số âm cho câu hỏi không phải chênh lệch / lãi lỗ")
    if (
        typ == "number"
        and isinstance(answer, (int, float))
        and ctx.total_revenue
        and _Q_MONEY.search(kq)
        and abs(answer) > 1.05 * ctx.total_revenue
    ):
        flags.append("số tiền lớn hơn doanh thu của cả dữ liệu shop")
    if typ == "percent" and isinstance(answer, (int, float)):
        if answer == 100 and _Q_SHARE.search(kq) and not _Q_CHANGE.search(kq):
            flags.append("tỉ trọng / tỉ lệ đúng 100 % (có thể lọc sai mẫu số)")
        if answer == 0 and not _Q_CHANGE.search(kq):
            flags.append("tỉ lệ bằng 0 (có thể so sai chuỗi trạng thái, lọc sai tên)")
        if _Q_SHARE.search(kq) and not _Q_CHANGE.search(kq) and not (0 <= answer <= 100):
            flags.append("tỉ lệ / tỉ trọng ngoài [0, 100]")
        elif not (-100 <= answer <= 10000):
            flags.append("phần trăm thay đổi ngoài [-100, 10000]")
    if typ == "list" and isinstance(answer, list) and not answer:
        flags.append("danh sách rỗng")
    names = []
    if typ == "string" and isinstance(answer, str):
        names = [answer]
    elif typ == "list" and isinstance(answer, list):
        names = [a for a in answer if isinstance(a, str)]
    if names:
        texts = ctx.text_values()
        missing = [
            n for n in names if key(n) not in texts and not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", n)
        ]
        if missing:
            flags.append(f"tên không có trong dữ liệu: {missing[:3]}")
    if typ == "date" and isinstance(answer, str):
        try:
            d = pd.Timestamp(answer)
        except ValueError:
            d = None
        lo = pd.Timestamp(ctx.data_start) if ctx.data_start is not None else None
        hi = pd.Timestamp(q["as_of_date"])
        out = d is not None and len(answer) == 10 and ((lo is not None and d < lo) or d > hi)
        # Hạn dùng, hạn trả nợ có thể sau hôm nay.
        if out and not re.search(r"\bhan\b|het han|den han|qua han|\blo\b", kq):
            flags.append("ngày nằm ngoài khoảng dữ liệu tới hôm nay")
    return flags


# ---------------------------------------------------------------- bỏ phiếu


def need_for(path: str, cfg: SolverConfig, ctx: ShopContext) -> int:
    if cfg.min_agree is not None:
        return cfg.min_agree
    if not ctx.mapping_ok:
        n3 = cfg.agree_nomap
    else:
        n3 = cfg.agree_helper if path == "helper" else cfg.agree_free
    return max(1, min(cfg.k, math.ceil(n3 / 3 * cfg.k - 1e-9)))


def decide(
    q: dict,
    cands: list[Candidate],
    cfg: SolverConfig,
    ctx: ShopContext,
    signal: str | None = None,
) -> tuple[dict, str, str, list[str]]:
    """(dòng submission, quyết định, path, cờ sanity).

    `signal`: lý do bước hiểu câu hỏi cho rằng câu không trả lời được (thiếu thông tin, chủ quan…). Chỉ thành
    unanswerable khi code cũng không ra đáp án đủ phiếu; code ra đáp án thì theo `cfg.signal_vs_answer`.
    """
    row, decision, path, flags = _decide(q, cands, cfg, ctx)
    if not signal or row["type"] == "unanswerable":
        return row, decision, path, flags
    unans = {"id": q["id"], "type": "unanswerable", "reason": signal}
    if row["type"] == "abstain":
        return unans, f"tín hiệu + code không ra đáp án ({decision}) → unanswerable", path, flags
    if cfg.signal_vs_answer == "abstain":
        return (
            {"id": q["id"], "type": "abstain"},
            f"tín hiệu ngược đáp án ({decision}) → abstain",
            path,
            flags,
        )
    return row, decision + ", bỏ qua tín hiệu", path, flags


def _decide(
    q: dict, cands: list[Candidate], cfg: SolverConfig, ctx: ShopContext
) -> tuple[dict, str, str, list[str]]:
    # Lời giải không qua sanity bị loại khỏi bỏ phiếu: hai lời giải cùng sai một kiểu (cùng so nhầm
    # chuỗi trạng thái ra 0) không được thắng lời giải đúng.
    bad = {}
    if cfg.use_sanity:
        for c in cands:
            if c.key and c.typ != "unanswerable":
                f = sanity(q, c.typ, c.answer, ctx)
                if f:
                    bad[id(c)] = f
    flagged = [f for fs in bad.values() for f in fs]
    cands = [c for c in cands if id(c) not in bad]
    keys = Counter(c.key for c in cands if c.key)
    row = {"id": q["id"], "type": "abstain"}
    if not keys and flagged:
        return row, f"sanity: {'; '.join(sorted(set(flagged)))}", "free", flagged
    if not keys:
        paths = Counter(c.path for c in cands if c.code)
        return row, "no_valid_candidate", (paths.most_common(1)[0][0] if paths else "free"), []
    best_key, n = keys.most_common(1)[0]
    winners = [c for c in cands if c.key == best_key]
    best = winners[0]
    path = Counter(c.path for c in winners).most_common(1)[0][0]
    if best.typ == "unanswerable":
        period = _period_unanswerable(best.reason)
        if not period and cfg.code_unanswerable == "abstain":
            return row, f"unanswerable do code {n}/{cfg.k} → abstain", path, []
        need = cfg.k if not period else need_for(path, cfg, ctx)
        if n >= need:
            return (
                {"id": q["id"], "type": "unanswerable", "reason": best.reason or ""},
                f"agree {n}/{cfg.k} unanswerable",
                path,
                [],
            )
        return row, f"disagree {n}/{cfg.k} unanswerable (cần {need})", path, []
    need = need_for(path, cfg, ctx)
    if n < need:
        return row, f"disagree {n}/{cfg.k} (cần {need}, {path})", path, flagged
    return (
        {"id": q["id"], "type": best.typ, "answer": best.answer},
        f"agree {n}/{cfg.k} ({path})" + (f", loại {len(bad)} lời giải sanity" if bad else ""),
        path,
        flagged,
    )


def solve(llm: LLM, ctx: ShopContext, q: dict, cfg: SolverConfig) -> tuple[dict, dict]:
    """Trả về (dòng submission, trace)."""
    t0 = time.time()
    pre = classify(
        q["question"],
        ctx.rulebook,
        ctx.readme,
        [c for df in ctx.tables.values() for c in df.columns],
    )
    if pre and pre[0] == "abstain" and cfg.near_miss == "unanswerable":
        pre = ("unanswerable", pre[1])
    if pre:
        kind, reason = pre
        row = (
            {"id": q["id"], "type": "unanswerable", "reason": reason}
            if kind == "unanswerable"
            else {"id": q["id"], "type": "abstain"}
        )
        return row, {
            "id": q["id"],
            "business_id": q["business_id"],
            "question": q["question"],
            "as_of_date": q["as_of_date"],
            "decision": f"precheck {kind}: {reason}",
            "reason_code": "PRECHECK_UNANSWERABLE"
            if kind == "unanswerable"
            else "PRECHECK_ABSTAIN",
            "path": "precheck",
            "row": row,
            "seconds": 0,
        }
    u = understand(llm, ctx, q, use_llm=cfg.understand, use_caps=cfg.understand_caps)
    if u.period_out:
        row = {"id": q["id"], "type": "unanswerable", "reason": u.period_out}
        return row, {
            "id": q["id"],
            "business_id": q["business_id"],
            "question": q["question"],
            "as_of_date": q["as_of_date"],
            "decision": f"precheck unanswerable: {u.period_out}",
            "reason_code": "PERIOD_OUT_OF_DATA",
            "path": "precheck",
            "understand": u.to_dict(),
            "row": row,
            "seconds": round(time.time() - t0, 2),
        }
    if cfg.pipeline == "v2":
        return _solve_v2(llm, ctx, q, cfg, u, t0)
    cands = [_run_candidate(llm, ctx, q, v, cfg, u) for v in range(cfg.k)]
    row, decision, path, flags = decide(q, cands, cfg, ctx, signal=u.unanswerable_signal)
    trace = {
        "id": q["id"],
        "business_id": q["business_id"],
        "question": q["question"],
        "as_of_date": q["as_of_date"],
        "decision": decision,
        "path": path,
        "mapping_ok": ctx.mapping_ok,
        "understand": u.to_dict(),
        "sanity": flags,
        "row": row,
        "seconds": round(time.time() - t0, 2),
        "candidates": [
            {
                "variant": c.variant,
                "type": c.typ,
                "answer": c.answer,
                "key": c.key,
                "reason": c.reason,
                "path": c.path,
                "sanity": sanity(q, c.typ, c.answer, ctx) if c.key else [],
                "error": c.error if not c.key else None,
                "code": c.code,
                "attempts": c.attempts,
            }
            for c in cands
        ],
    }
    return row, trace


# ---------------------------------------------------------------- pipeline v2: cơ chế giải + verifier


def reason_code(row: dict, decision: str, checks: dict | None = None) -> str:
    """Mã lý do cố định cho mỗi quyết định (IMP-09), để phân loại thay đổi giữa hai lần chạy mà không đọc log."""
    t = row.get("type")
    d = decision or ""
    names = {c.get("name") for cs in (checks or {}).values() for c in cs}
    if d.startswith("precheck"):
        return "PRECHECK_UNANSWERABLE" if t == "unanswerable" else "PRECHECK_ABSTAIN"
    if t == "unanswerable":
        if "kỳ ngoài dữ liệu" in d or "nằm ngoài khoảng dữ liệu" in d:
            return "PERIOD_OUT_OF_DATA"
        if "capability vắng" in d:
            return "CAPABILITY_ABSENT"
        if "tín hiệu" in d:
            return "SIGNAL_UNANSWERABLE"
        return "CODE_UNANSWERABLE"
    if t == "abstain":
        if "capability chưa chắc" in d:
            caps = re.findall(r"'([a-z_]+)'", d)
            return "CAP_UNCERTAIN:" + ",".join(caps) if caps else "CAP_UNCERTAIN"
        if "graph không reconcile" in d:
            return "GRAPH_UNRECONCILED"
        if "bất đồng" in d:
            return "MECHANISM_DISAGREE"
        if "temporal_boundary" in names:
            return "TEMPORAL_BOUNDARY"
        if "không lời giải hợp lệ" in d:
            return (
                "CHECKS_FAILED:" + ",".join(sorted(n for n in names if n))
                if names
                else "NO_VALID_CANDIDATE"
            )
        if "unanswerable do code" in d:
            return "CODE_UNANSWERABLE_ABSTAIN"
        if "một cơ chế" in d:
            return "SINGLE_MECHANISM"
        return "ABSTAIN_OTHER"
    if "2/3" in d:
        return "ANSWER_MAJORITY"
    if "một cơ chế" in d:
        return "ANSWER_SINGLE"
    return "ANSWER_AGREE"


def _cand_trace(c: Candidate) -> dict:
    return {
        "variant": c.variant,
        "mechanism": c.mechanism,
        "type": c.typ,
        "answer": c.answer,
        "key": c.key,
        "reason": c.reason,
        "path": c.path,
        "rows": c.rows,
        "period": c.period,
        "seconds": c.seconds,
        "lint": c.lint,
        "error": c.error if not c.key else None,
        "code": c.code,
        "attempts": c.attempts,
    }


def _solve_v2(llm: LLM, ctx: ShopContext, q: dict, cfg: SolverConfig, u: Understanding, t0: float):
    """Chạy các cơ chế giải khác nhau (mặc định helper + pandas), verifier quyết định; cơ chế thứ ba chỉ khi hai cái
    đầu bất đồng hoặc verifier nghi. Ghi thời gian từng giai đoạn vào trace."""
    timings = {"understand": round(time.time() - t0, 2)}
    needs = list(getattr(u, "capabilities", []) or [])
    cands = [
        _run_candidate(llm, ctx, q, i, cfg, u, mechanism=m) for i, m in enumerate(cfg.mechanisms)
    ]
    for c in cands:
        timings[c.mechanism] = c.seconds
    v = verify(q, cands, ctx, cfg.verifier, needs)
    if v.need_more and cfg.third and cfg.third not in cfg.mechanisms:
        c3 = _run_candidate(llm, ctx, q, len(cands), cfg, u, mechanism=cfg.third)
        timings[c3.mechanism] = c3.seconds
        cands.append(c3)
        v = verify(q, cands, ctx, cfg.verifier, needs)
    row, decision = v.row, v.decision
    signal = u.unanswerable_signal
    from datamate.agent.verifier import has_data_answer

    if signal and row["type"] not in ("unanswerable",):
        if row["type"] == "abstain" and has_data_answer(cands):
            decision += ", có tín hiệu thiếu thông tin nhưng có cơ chế ra đáp án → giữ abstain"
        elif row["type"] == "abstain":
            row, decision = (
                {"id": q["id"], "type": "unanswerable", "reason": signal},
                f"tín hiệu + code không ra đáp án ({decision}) → unanswerable",
            )
        elif cfg.signal_vs_answer == "abstain":
            row, decision = (
                {"id": q["id"], "type": "abstain"},
                f"tín hiệu ngược đáp án ({decision}) → abstain",
            )
        else:
            decision += ", bỏ qua tín hiệu"
    timings["total"] = round(time.time() - t0, 2)
    trace = {
        "id": q["id"],
        "business_id": q["business_id"],
        "question": q["question"],
        "as_of_date": q["as_of_date"],
        "decision": decision,
        "reason_code": reason_code(row, decision, v.checks),
        "path": v.path,
        "pipeline": "v2",
        "mapping_ok": ctx.mapping_ok,
        "capabilities_needed": needs,
        "understand": u.to_dict(),
        "verifier": v.checks,
        "row": row,
        "seconds": timings["total"],
        "timings": timings,
        "candidates": [_cand_trace(c) for c in cands],
    }
    return row, trace
