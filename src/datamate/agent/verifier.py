"""Verifier: quyết định answer / unanswerable / abstain từ lời giải của các cơ chế giải khác nhau
(generalization-v2 §9). Thay cho bỏ phiếu theo biến thể prompt (`solver._decide`) ở pipeline v2.

Kiểm mỗi lời giải:
- type / format §9 (đã ép kiểu ở solver; lời giải lỗi format không có khoá đáp án);
- kết quả trung gian: `ROWS` = 0 (lọc kỳ / đối tượng ra rỗng) mà đáp án là số → nghi;
- `PERIOD` bắt đầu sau hôm nay → nghi;
- snapshot safety: code nhắc cột chỉ đúng tại ngày cuối dữ liệu khi hôm nay sớm hơn → nghi;
- hợp lý: số âm cho câu không hỏi chênh lệch, phần trăm ngoài khoảng, list rỗng, tên không có trong dữ liệu (so khoá
  chuẩn, nhận cả phần của ô — tên dẫn xuất như dòng xe tách từ ô "dòng xe + biển số"), ngày ngoài khoảng dữ liệu.
Quyết định:
- capability câu hỏi cần mà graph chắc chắn không có (không planner nào đề xuất role) → unanswerable (kèm reason);
- capability uncertain, graph không reconcile được, mọi lời giải bị nghi, các cơ chế cho đáp án khác nhau → abstain;
- còn lại → answer.
Không abstain chỉ vì đáp án bằng 0 hay chuỗi không trùng nguyên ô thô (cờ "= 0" cũ thay bằng kiểm tra ROWS).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

import pandas as pd

from datamate.text import key

# Ngưỡng chọn bằng CV theo shop (scripts/tune_thresholds.py --verifier).


@dataclass
class VerifierConfig:
    # Chỉ một cơ chế ra đáp án hợp lệ (cơ chế kia lỗi / không ra): nhận nếu path của lời giải nằm trong tập này và
    # capability câu hỏi cần đều verified.
    single_ok_paths: tuple = ("helper",)
    # Ba cơ chế (cơ chế thứ ba chạy khi hai cái đầu bất đồng): 2/3 đồng ý thì nhận.
    majority_of_three: bool = True
    # Capability cần dùng uncertain / graph không reconcile: 'abstain' (mặc định, §9: các cơ chế dùng chung một graph
    # có thể cùng sai, đồng thuận không đủ) | 'strict' (cần mọi cơ chế hợp lệ đồng ý và ≥ 2 cơ chế).
    uncertain_policy: str = "abstain"
    # Cờ cũ "đáp án = 0 cho câu doanh thu / tỉ lệ" (solver.sanity). Mặc định tắt: thay bằng kiểm tra ROWS.
    zero_flags: bool = False
    # Unanswerable do code tự báo (không phải kỳ ngoài dữ liệu): 'abstain' | 'all_agree'.
    code_unanswerable: str = "abstain"
    # Capability câu hỏi cần mà graph chắc chắn không có: 'if_no_answer' (unanswerable khi không cơ chế nào ra đáp
    # án hợp lệ; tín hiệu của LLM ở bước hiểu câu hỏi có thể nhầm) | 'always' | 'off'.
    capability_unanswerable: str = "if_no_answer"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Verdict:
    row: dict
    decision: str
    path: str
    checks: dict = field(default_factory=dict)  # biến thể → các kiểm tra trượt
    need_more: bool = False  # nên chạy thêm cơ chế thứ ba

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


_Q_NEG_OK = re.compile(
    r"lai|lo|chenh|tang|giam|so voi|thay doi|hon|kem|lech|con no|ton|nhap tru|am"
)
_Q_CHANGE = re.compile(r"tang|giam|so voi|thay doi|chenh|hon|kem|lech")
# Câu hỏi về hạn (hạn trả, hạn góp, hạn dùng): kỳ / ngày sau hôm nay là hợp lệ.
_Q_DUE = re.compile(r"\bhan\b|het han|den han|qua han|phai tra|phai gop|\blo\b")
_Q_SHARE = re.compile(r"t[iy] le|t[iy] trong|chiem|phan tram cua|bao nhieu phan tram")
_Q_AMOUNT = re.compile(
    r"doanh thu|dthu|\bban\b|thu ve|\bdon\b|so luong|tien hang|t[iy] le|t[iy] trong|chiem"
)


def _name_known(name: str, ctx) -> bool:
    """Tên có trong dữ liệu: trùng khoá chuẩn của một ô chữ, hoặc là một cụm từ nguyên vẹn bên trong ô (thực thể
    dẫn xuất: dòng máy tách từ tên hàng, dòng xe tách từ ô "dòng xe + biển số"), hoặc là mã / ngày."""
    k = key(name)
    if not k or re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", name):
        return True
    texts = ctx.text_values()
    if k in texts:
        return True
    pat = re.compile(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])")
    return any(pat.search(t) for t in texts if len(t) > len(k))


# Câu hỏi số lượng / đối tượng ("có mấy khách", "bao nhiêu đơn", "ai", "ngày nào"): đáp án không thể là đúng / sai.
_Q_COUNT = re.compile(r"\b(may|bao nhieu|bnhiu|bn|ai|ngay nao|thang nao|mat hang nao|khach nao)\b")


def candidate_checks(q: dict, c, ctx, cfg: VerifierConfig) -> list[Check]:
    """Các kiểm tra trượt của một lời giải có đáp án."""
    out: list[Check] = []
    typ, ans = c.typ, c.answer
    if typ in (None, "unanswerable"):
        return out
    kq = key(q["question"])
    rows = getattr(c, "rows", None)
    if rows == 0 and typ in ("number", "percent", "table", "list"):
        out.append(
            Check("rows_empty", False, "lọc kỳ / đối tượng ra 0 dòng mà đáp án là số / danh sách")
        )
    period = getattr(c, "period", None)
    if period and pd.Timestamp(period[0]) > pd.Timestamp(q["as_of_date"]) and not _Q_DUE.search(kq):
        out.append(Check("period_future", False, f"kỳ {period} bắt đầu sau hôm nay"))
    if typ == "boolean" and _Q_COUNT.search(kq):
        out.append(
            Check("type_question", False, "câu hỏi số lượng / đối tượng mà đáp án là đúng / sai")
        )
    lint = getattr(c, "lint", None) or []
    th = [x for x in lint if x.startswith("ngưỡng:")]
    tb = [x for x in lint if not x.startswith("ngưỡng:")]
    if tb:
        out.append(Check("temporal_boundary", False, "; ".join(tb[:2])))
    if th:
        out.append(Check("threshold", False, "; ".join(th[:2])))
    snaps = getattr(ctx, "snapshot_cols", None) or []
    if (
        snaps
        and ctx.data_end is not None
        and pd.Timestamp(q["as_of_date"]) < pd.Timestamp(ctx.data_end).normalize()
    ):
        used = [sc for sc in snaps if c.code and sc.rpartition(".")[2] in c.code]
        if used:
            out.append(
                Check("snapshot", False, f"dùng cột {used} khi hôm nay sớm hơn ngày cuối dữ liệu")
            )
    if typ == "number" and isinstance(ans, (int, float)):
        if ans < 0 and not _Q_NEG_OK.search(kq):
            out.append(Check("negative", False, "số âm cho câu không hỏi chênh lệch / lãi lỗ"))
        if cfg.zero_flags and ans == 0 and _Q_AMOUNT.search(kq) and not _Q_CHANGE.search(kq):
            out.append(Check("zero", False, "số bằng 0 cho câu doanh thu / số lượng"))
        tot = getattr(ctx, "total_revenue", None)
        if (
            tot
            and re.search(r"doanh thu|dthu|tien|thu ve|ban duoc|gia tri", kq)
            and abs(ans) > 1.05 * tot
        ):
            out.append(Check("too_large", False, "số tiền lớn hơn doanh thu của cả dữ liệu shop"))
    if typ == "percent" and isinstance(ans, (int, float)):
        if _Q_SHARE.search(kq) and not _Q_CHANGE.search(kq) and not (0 <= ans <= 100):
            out.append(Check("share_range", False, "tỉ lệ / tỉ trọng ngoài [0, 100]"))
        elif not (-100 <= ans <= 10000):
            out.append(Check("change_range", False, "phần trăm thay đổi ngoài [-100, 10000]"))
        if cfg.zero_flags and ans == 0 and not _Q_CHANGE.search(kq):
            out.append(Check("zero", False, "tỉ lệ bằng 0"))
    if typ == "list" and isinstance(ans, list) and not ans:
        out.append(Check("empty_list", False, "danh sách rỗng"))
    names = [ans] if typ == "string" and isinstance(ans, str) else []
    if typ == "list" and isinstance(ans, list):
        names = [a for a in ans if isinstance(a, str)]
    missing = [n for n in names if not _name_known(n, ctx)]
    if missing:
        out.append(Check("unknown_name", False, f"tên không có trong dữ liệu: {missing[:3]}"))
    if typ == "date" and isinstance(ans, str):
        try:
            d = pd.Timestamp(ans)
        except ValueError:
            d = None
        lo = pd.Timestamp(ctx.data_start) if ctx.data_start is not None else None
        hi = pd.Timestamp(q["as_of_date"])
        out_ = d is not None and len(ans) == 10 and ((lo is not None and d < lo) or d > hi)
        if out_ and not _Q_DUE.search(kq):
            out.append(Check("date_range", False, "ngày nằm ngoài khoảng dữ liệu tới hôm nay"))
    return out


_ORDER_FUNCS = re.compile(
    r"\bdm\.(orders|sold_lines|revenue|count_orders|count_cancelled|cancel_rate|aov|daily_revenue|customers"
    r"|loyal_customers|big_spenders|fickle_customers|popular_items|busiest_days|peak_hour)\s*\("
)


def implied_capabilities(code: str | None, graph) -> set[str]:
    """Capability lời giải thực sự dùng, suy từ code: hàm đơn hàng → order_revenue; ca / giờ → time_of_day;
    sổ thu tiền → debt_payments; nhắc tên bảng / cột của role → capability của role đó."""
    if not code:
        return set()
    out = set()
    if _ORDER_FUNCS.search(code):
        out.add("order_revenue")
    if re.search(r"\bdm\.(shift|peak_hour)\s*\(|\.dt\.hour|\bgio\b", code):
        out.add("time_of_day")
    if re.search(r"\bdm\.payments\s*\(", code):
        out.add("debt_payments")
    # Code nhắc nguyên văn cột của role bắt buộc (cơ chế pandas / sql tự đọc bảng). Cố ý kéo cả capability, kể cả khi
    # role được nhắc đã verified: graph của shop chưa chắc thì "hai cơ chế đồng ý" không còn là bằng chứng độc lập
    # (đã thử chỉ chặn theo role được nhắc: lọt câu sai do các cơ chế cùng giả định sai, IMP-12 mục 5).
    for cap in ("order_revenue", "time_of_day"):
        req = _caps_roles()[cap][0]
        roles = [
            r for spec in req for r in spec.split("|") if r.split(".")[0] in ("orders", "lines")
        ]
        cols = [graph.role(r).ref.column for r in roles if graph.role(r).ref is not None]
        if any(c and (repr(c) in code or f'"{c}"' in code) for c in cols):
            out.add(cap)
    return out


def _caps_roles():
    from datamate.agent.semantic_graph import CAPABILITIES

    return CAPABILITIES


def certain_absent(graph, cap: str) -> bool:
    """Capability absent và mọi role cần của nó không được planner nào đề xuất (không phải absent do probe loại)."""
    c = graph.capabilities.get(cap)
    if c is None or c.status != "absent":
        return False
    for spec in c.roles:
        for r in spec.split("|"):
            try:
                if graph.role(r).votes:
                    return False
            except KeyError:
                continue
    return True


_Q_GROUP = re.compile(r"(?<!khach )(?<!khach hang )\bnhom\b(?!\s+(khach|tuoi|\d))")


def product_group_question(question: str) -> bool:
    """Câu hỏi về một nhóm hàng ('nhóm cà phê', 'nhóm hàng bình sữa'), không phải nhóm khách / nhóm tuổi nợ."""
    k = key(question)
    return bool(_Q_GROUP.search(k)) and "tuoi no" not in k


def _group_absent(graph) -> bool:
    try:
        rb = graph.entity("product").group
    except Exception:  # noqa: BLE001
        return False
    return rb.status == "absent" and not rb.votes


def _period_unanswerable(reason: str | None) -> bool:
    return bool(reason) and "nằm ngoài khoảng dữ liệu" in reason


def verify(
    q: dict, cands: list, ctx, cfg: VerifierConfig, needs: list[str] | None = None
) -> Verdict:
    """`cands`: lời giải của các cơ chế (solver.Candidate, có `mechanism`, `rows`, `period`). `needs`: capability
    câu hỏi cần (từ bước hiểu câu hỏi)."""

    graph = getattr(ctx, "graph", None)
    needs = needs or []
    # 0. Câu hỏi theo nhóm hàng mà shop chắc chắn không ghi cột nhóm hàng (không planner nào đề xuất): rulebook §3
    # "nhóm hàng theo cột nhóm hàng", §7 không suy từ cột khác (tên hàng) → unanswerable dù code ra số.
    rules = getattr(ctx, "rules", None)
    group_rule = rules is None or "product_group" not in getattr(rules, "disabled", {})
    if (
        graph is not None
        and group_rule
        and product_group_question(q["question"])
        and _group_absent(graph)
    ):
        return Verdict(
            {"id": q["id"], "type": "unanswerable", "reason": "shop không ghi nhóm hàng"},
            "nhóm hàng: shop không có cột nhóm hàng",
            "capability",
        )
    # 1. Capability chắc chắn vắng → unanswerable (ngay, hoặc khi không cơ chế nào ra đáp án hợp lệ).
    gone = []
    if graph is not None and cfg.capability_unanswerable != "off":
        gone = [c for c in needs if certain_absent(graph, c)]
    if gone:
        unans = {
            "id": q["id"],
            "type": "unanswerable",
            "reason": f"shop không ghi dữ liệu cho: {', '.join(gone)}",
        }
        if cfg.capability_unanswerable == "always":
            return Verdict(unans, f"capability vắng: {gone}", "capability")
        v = _verify(q, cands, ctx, cfg, needs, graph)
        if v.row["type"] == "abstain":
            # "Vắng" là kết luận của planner (LLM), không phải bằng chứng chắc chắn: có cơ chế tính ra đáp án từ dữ liệu
            # thì mâu thuẫn với "shop không ghi" → bỏ (abstain), không trả unanswerable.
            if has_data_answer(cands):
                return Verdict(v.row, f"capability vắng {gone} nhưng có cơ chế ra đáp án → {v.decision}", v.path,
                               v.checks)  # fmt: skip
            return Verdict(unans, f"capability vắng {gone} + {v.decision}", v.path, v.checks)
        return v
    return _verify(q, cands, ctx, cfg, needs, graph)


def has_data_answer(cands: list) -> bool:
    """Có lời giải chạy xong và ra đáp án (không phải unanswerable) từ dữ liệu."""
    return any(
        getattr(c, "key", None) and getattr(c, "typ", None) not in (None, "unanswerable")
        for c in cands
    )


def _verify(q: dict, cands: list, ctx, cfg: VerifierConfig, needs: list[str], graph) -> Verdict:
    from datamate.agent.solver import classify_path

    row_abstain = {"id": q["id"], "type": "abstain"}
    uncertain = []
    if graph is not None:
        implied = set(needs)
        for c in cands:
            if c.key and c.typ != "unanswerable":
                implied |= implied_capabilities(c.code, graph)
        # Chỉ capability gắn với role (có probe) mới chặn; capability planner tự đề xuất là thông tin.
        uncertain = sorted(
            c for c in implied if graph.capability(c) == "uncertain" and graph.capabilities[c].roles
        )
    shaky = bool(uncertain) or graph is None or not graph.reconciled
    if shaky and cfg.uncertain_policy == "abstain" and any(c.key for c in cands):
        why = f"capability chưa chắc {uncertain}" if uncertain else "graph không reconcile được"
        path = Counter(classify_path(c.code) for c in cands if c.code).most_common(1)
        return Verdict(row_abstain, why, path[0][0] if path else "free")
    strict = cfg.uncertain_policy == "strict" and shaky
    # 2. Kiểm từng lời giải.
    checks: dict[str, list[dict]] = {}
    valid = []
    for c in cands:
        if not c.key:
            continue
        if c.typ == "unanswerable":
            valid.append(c)
            continue
        bad = candidate_checks(q, c, ctx, cfg)
        if bad:
            checks[c.mechanism] = [asdict(b) for b in bad]
        else:
            valid.append(c)
    n_mech = len({c.mechanism for c in cands})
    if not valid:
        path = Counter(classify_path(c.code) for c in cands if c.code).most_common(1)
        return Verdict(row_abstain, "không lời giải hợp lệ" + (f": {checks}" if checks else ""),
                       path[0][0] if path else "free", checks, need_more=n_mech < 3)  # fmt: skip
    keys = Counter(c.key for c in valid)
    best_key, n = keys.most_common(1)[0]
    winners = [c for c in valid if c.key == best_key]
    best = winners[0]
    path = classify_path(next((c.code for c in winners if c.mechanism == "helper"), best.code))
    # 3. Unanswerable do code.
    if best.typ == "unanswerable":
        if _period_unanswerable(best.reason):
            return Verdict({"id": q["id"], "type": "unanswerable", "reason": best.reason or ""},
                           f"kỳ ngoài dữ liệu ({n}/{len(valid)})", path, checks)  # fmt: skip
        if cfg.code_unanswerable == "all_agree" and n == len(valid) and n >= 2:
            return Verdict({"id": q["id"], "type": "unanswerable", "reason": best.reason or ""},
                           f"mọi cơ chế báo unanswerable ({n})", path, checks)  # fmt: skip
        return Verdict(
            row_abstain, f"unanswerable do code {n}/{len(valid)} → abstain", path, checks
        )
    answer = {"id": q["id"], "type": best.typ, "answer": best.answer}
    others = [c for c in valid if c.key != best_key]
    # 4. Đồng thuận giữa cơ chế.
    if others:
        if n_mech >= 3 and cfg.majority_of_three and n >= 2 and not strict:
            return Verdict(answer, f"2/3 cơ chế đồng ý ({path})", path, checks)
        return Verdict(
            row_abstain, f"cơ chế bất đồng: {dict(keys)}", path, checks, need_more=n_mech < 3
        )
    if n >= 2:
        return Verdict(answer, f"{n} cơ chế đồng ý ({path})", path, checks)
    # Chỉ một lời giải hợp lệ.
    if strict:
        why = f"capability chưa chắc {uncertain}" if uncertain else "graph không reconcile được"
        return Verdict(
            row_abstain, f"một cơ chế, {why} → cần đồng thuận", path, checks, need_more=n_mech < 3
        )
    if path in cfg.single_ok_paths:
        return Verdict(answer, f"một cơ chế hợp lệ ({path})", path, checks, need_more=False)
    return Verdict(
        row_abstain,
        f"một cơ chế hợp lệ ({path}), cần đồng thuận",
        path,
        checks,
        need_more=n_mech < 3,
    )
