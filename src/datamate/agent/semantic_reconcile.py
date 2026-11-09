"""Hợp nhất K đề xuất của planner thành semantic graph v2 bằng probe trên dữ liệu.

1. Chuẩn hoá đầu ra mỗi planner (tên gần đúng → tên thật, `semantic.resolve_names`; sửa hình thức bảng phẳng).
2. Mỗi role: gom ứng viên (bảng, cột) từ K planner.
3. Probe **từng ứng viên** (không chỉ ứng viên thắng phiếu): kiểu dữ liệu, parse ngày / giờ, khoá duy nhất, độ phủ nối,
   nối có nhân dòng, tổng tiền dòng ↔ tiền đơn, overlap mã / tên, số loại trạng thái.
4. Loại ứng viên trượt probe trọng yếu; trong số còn lại ưu tiên probe mạnh rồi tới phiếu. Một ứng viên (hoặc mọi
   ứng viên cho cùng kết quả kiểm tra) → verified nếu ≥ min(2, K) phiếu hoặc có probe mạnh; nhiều ứng viên cùng qua mà
   kết quả kiểm tra khác nhau → uncertain (trừ khi chỉ ứng viên thắng có probe mạnh); không ứng viên nào qua →
   uncertain không cột (có ứng viên nhưng đều sai) hoặc absent (không planner nào đề xuất).
5. Không majority-vote khi planner cùng sai: K planner chọn cùng một cột trượt probe thì role vẫn không dùng được.
6. Phát hiện bỏ sót: profile thấy cột có giờ thật mà mọi planner để `time = null` → role `time` uncertain; trạng
   thái nằm trong cột ghi chú tương tự (giá trị trạng thái planner nêu nằm ở một cột mà role `status` trống).
7. Capability = hàm của role (`semantic_graph.compute_capabilities`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from datamate.agent import invariants as inv
from datamate.agent.profile import ShopProfile, norm_series, profile_shop
from datamate.agent.semantic_graph import (
    SCHEMA_VERSION,
    Capability,
    ColumnRef,
    EntitySpec,
    JoinSpec,
    ObservationSpec,
    RoleBinding,
    SemanticGraph,
    compute_capabilities,
)
from datamate.agent.semantic_planner import (
    build_planner_messages,
    classify_statuses,
    planner_roles,
    prompt_hash,
    run_planner,
    status_definitions,
)
from datamate.loaders.shop import Shop
from datamate.rules import DEFAULT_STATUS_LABELS, status_labels
from datamate.text import key

log = logging.getLogger(__name__)

K_PLAN = 3
# Số phiếu tối thiểu để một ứng viên (không có probe mạnh) là verified: min(VERIFY_MIN_VOTES, K).
VERIFY_MIN_VOTES = 2
# Trạng thái: tỉ lệ dòng có giá trị bị planner xếp lẫn giữa nhóm tính / không tính doanh thu, tối đa.
STATUS_CONTEST_MAX = 0.01
# Tỉ lệ đơn được tính doanh thu tối thiểu theo phân vùng trạng thái (v1: 50 %).
MIN_REVENUE_ORDERS = 0.5
# Dòng điều chỉnh: tỉ lệ dòng của một giá trị có hình dạng "không phải hàng" tối thiểu.
ADJ_SHAPE_MIN = 0.8
# Bảng phiếu trả: tỉ lệ đơn xuất hiện trong bảng tối đa (bảng chứa > 50 % đơn là bảng bán hàng, không phải phiếu trả).
RETURN_TABLE_MAX_COVER = 0.5
# Sổ thu tiền / phiếu trả: tỉ lệ mã đơn có trong bảng đơn tối thiểu.
REF_MIN_COVER = 0.5
# Overlap mã hàng dòng ↔ danh mục, tên hàng ↔ bảng quy đổi.
MASTER_MIN_OVERLAP = 0.5
DATE_RANGE_MIN = 0.95
MAX_STATUS_VALUES = 60
# Cột mã đơn: số giá trị khác nhau / số dòng tối thiểu (bảng phẳng hiếm khi > 50 dòng mỗi đơn).
MIN_KEY_CARDINALITY = 0.02

RECONCILE_CONFIG = {
    "LOGIC_REVISION": 10,
    "VERIFY_MIN_VOTES": VERIFY_MIN_VOTES,
    "STATUS_CONTEST_MAX": STATUS_CONTEST_MAX,
    "MIN_REVENUE_ORDERS": MIN_REVENUE_ORDERS,
    "ADJ_SHAPE_MIN": ADJ_SHAPE_MIN,
    "RETURN_TABLE_MAX_COVER": RETURN_TABLE_MAX_COVER,
    "REF_MIN_COVER": REF_MIN_COVER,
    "MASTER_MIN_OVERLAP": MASTER_MIN_OVERLAP,
    "DATE_RANGE_MIN": DATE_RANGE_MIN,
    "MIN_KEY_CARDINALITY": MIN_KEY_CARDINALITY,
}

DATE_ROLES = {"orders.date", "payments.date"}
NUM_ROLES = {
    "lines.qty", "lines.qty_base", "lines.line_amount", "lines.line_discount", "lines.unit_cost",
    "products.unit_cost", "orders.goods_amount", "orders.order_discount", "orders.shipping", "orders.vat",
    "orders.order_total", "orders.paid", "payments.amount", "conversion.factor", "returns.qty",
}  # fmt: skip
TEXT_ROLES = {
    "orders.status", "orders.customer", "orders.payment_method", "orders.staff", "orders.channel",
    "lines.product", "lines.unit", "payments.method", "products.name", "products.group",
    "conversion.product", "conversion.unit",
}  # fmt: skip
GROUPS = ("orders", "lines", "payments", "returns", "conversion", "products")
# Thứ tự xử lý role cột: khoá trước (probe của role sau dùng khoá đã chọn), tiền dòng trước tiền hàng của đơn.
COLUMN_ORDER = [
    "orders.order", "lines.order", "orders.date", "orders.time", "lines.line_amount", "orders.goods_amount",
    "orders.status", "lines.product", "lines.product_code", "lines.qty", "lines.qty_base", "lines.unit",
    "lines.line_discount", "lines.unit_cost", "orders.customer", "orders.customer_code",
    "orders.payment_method", "orders.order_discount", "orders.shipping", "orders.vat", "orders.order_total",
    "orders.paid", "orders.staff", "orders.channel", "payments.order", "payments.date", "payments.amount",
    "payments.method", "returns.qty", "returns.order", "products.code", "products.name", "products.group",
    "products.unit_cost", "conversion.product", "conversion.unit", "conversion.factor",
]  # fmt: skip
_INCLUDED = {key("hoàn thành"), key("đang giao")}


def _canon(label: str) -> str | None:
    """Nhãn rulebook → nhãn nội bộ (4 nhãn mặc định) nếu trùng tên, không thì None."""
    for d in DEFAULT_STATUS_LABELS:
        if key(d) == key(label):
            return d
    return None


# ---------------------------------------------------------------- chuẩn hoá hình thức mỗi planner


def _value_column(df: pd.DataFrame, values: list, min_hits: int) -> str | None:
    """Cột chữ chứa nhiều nhất các giá trị (so theo khoá bỏ dấu) — ít nhất `min_hits` giá trị; hoà → không chọn."""
    want = {key(v) for v in values if key(v)}
    if not want:
        return None
    hits = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        try:
            have = {key(v) for v in s.dropna().astype(str).unique()[:5000]}
        except TypeError:
            continue
        hits[c] = len(want & have)
    if not hits:
        return None
    ranked = sorted(hits.items(), key=lambda x: -x[1])
    best, n = ranked[0]
    if n < min_hits or (len(ranked) > 1 and ranked[1][1] == n):
        return None
    return best


def _locate_order_columns(p: dict, T: dict) -> list[str]:
    """Planner nêu GIÁ TRỊ (nhóm trạng thái, cách ghi khách lẻ) nhưng bỏ trống CỘT của bảng đơn (hay gặp ở bảng phẳng với
    model nhỏ): tìm cột chứa các giá trị đó trong bảng đơn. Ngày đơn: chỉ khi bảng có đúng một cột ngày phủ ≥ 95 % số
    dòng, hoặc đúng một cột ngày thuần (không mang giờ) cạnh các cột thời điểm; còn lại là mơ hồ, để trống. Chỉ điền role
    còn trống; probe kiểm như mọi đề xuất khác."""
    r, notes = p["roles"], []
    t = r.get("orders.table")
    if t not in T:
        return notes
    df = T[t]
    # Kể cả trạng thái ghi trong cột ghi chú (`status_from_note`): cột ghi chú chứa đúng các cách ghi planner nêu.
    if not r.get("orders.status") and p.get("status"):
        col = _value_column(df, list(p["status"]), min_hits=min(2, len(p["status"])))
        if col:
            r["orders.status"] = col
            kind = "ghi chú" if p.get("status_from_note") else "trạng thái"
            notes.append(f"orders.status trống: cột {kind} {col!r} chứa các giá trị planner nêu")
    if not r.get("orders.customer") and p.get("walkin_values"):
        col = _value_column(df, p["walkin_values"], min_hits=1)
        if col:
            r["orders.customer"] = col
            notes.append(f"orders.customer trống: cột {col!r} chứa cách ghi khách lẻ planner nêu")
    if not r.get("orders.date"):
        dates = [
            c
            for c in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[c]) and df[c].notna().mean() >= 0.95
        ]
        if len(dates) > 1:
            # Cột ngày thuần (mọi giá trị đúng 00:00) cạnh cột thời điểm mang giờ: cột ngày của đơn là cột không mang
            # giờ; cột kia là ứng viên giờ (`_detect_missing_time`). Nhiều cột ngày thuần vẫn mơ hồ → để trống.
            dates = [c for c in dates if (df[c].dropna() == df[c].dropna().dt.normalize()).all()]
        if len(dates) == 1:
            r["orders.date"] = dates[0]
            notes.append(f"orders.date trống: {dates[0]!r} là cột ngày duy nhất của bảng đơn")
    return notes


def normalize_plan(p: dict, T: dict) -> list[str]:
    """Sửa hình thức đề xuất của một planner (không đổi lựa chọn cột): bảng phẳng thiếu `orders`, bảng "đơn" có
    nhiều dòng mỗi mã mà không có bảng dòng hàng, phiếu trả trỏ vào bảng bán hàng. Trả về ghi chú."""
    r = p["roles"]
    notes = []
    if "orders.table" not in r and "lines.table" in r:
        r["orders.table"] = r["lines.table"]
        if "lines.order" in r:
            r.setdefault("orders.order", r["lines.order"])
        notes.append("orders = null: dùng bảng dòng hàng (file phẳng)")
    if (
        "lines.table" in r
        and "lines.line_amount" not in r
        and (r.get("orders.goods_amount") or r.get("orders.order_total"))
    ):
        for k in [k for k in r if k.startswith("lines.")]:
            del r[k]
        notes.append("bảng dòng hàng không có thành tiền: bỏ, dùng tiền của bảng đơn")
    ot, oo = r.get("orders.table"), r.get("orders.order")
    if "lines.table" not in r and ot in T and oo in T[ot].columns:
        ids = T[ot][oo].dropna()
        amt = r.get("orders.goods_amount") or r.get("orders.order_total")
        if len(ids) and ids.duplicated().mean() > inv.MAX_KEY_DUP and amt:
            r.update({"lines.table": ot, "lines.order": oo, "lines.line_amount": amt})
            r.pop("orders.goods_amount", None)
            r.pop("orders.order_total", None)
            notes.append("bảng đơn nhiều dòng mỗi mã: coi là bảng phẳng")
    notes += _locate_order_columns(p, T)
    if r.get("returns.table") in {r.get("lines.table"), r.get("orders.table")}:
        for k in [k for k in r if k.startswith("returns.")]:
            del r[k]
        notes.append("returns trỏ vào bảng bán hàng: bỏ")
    if r.get("orders.table") == r.get("lines.table"):
        for f in ("goods_amount", "order_total"):
            if r.get(f"orders.{f}") and r.get(f"orders.{f}") == r.get("lines.line_amount"):
                del r[f"orders.{f}"]
    return notes


# ---------------------------------------------------------------- probe


class _Ctx:
    def __init__(self, shop: Shop, prof: ShopProfile, k: int):
        self.shop = shop
        self.T = shop.frames()
        self.prof = prof
        self.k = k
        self.chosen: dict[str, str | None] = {}  # role → cột / bảng đã chọn

    def table(self, group: str) -> str | None:
        return self.chosen.get(f"{group}.table")

    @property
    def flat(self) -> bool:
        return self.table("lines") is not None and self.table("lines") == self.table("orders")


def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _overlap(a: pd.Series, b: pd.Series) -> float:
    av = set(a.dropna().map(key))
    bv = set(b.dropna().map(key))
    return len(av & bv) / len(av) if av else 0.0


def probe(role: str, ref: ColumnRef, c: _Ctx) -> tuple[dict, bool, bool, object]:
    """(kết quả probe, qua probe trọng yếu, có probe mạnh, giá trị kiểm tra để so các ứng viên)."""
    T = c.T
    df = T[ref.table]
    s = df[ref.column]
    pr: dict[str, bool | float] = {}
    crit, strong = True, False
    check: object = None
    if role in DATE_ROLES:
        is_dt = pd.api.types.is_datetime64_any_dtype(s)
        pr["is_date"] = is_dt
        crit &= is_dt
        if is_dt:
            r = inv.date_coverage(df, ref.column)
            pr["date_parse"] = r.value
            crit &= r.passed
            lo, hi = c.shop.data_start, c.shop.data_end
            if role == "orders.date" and lo is not None and hi is not None and s.notna().any():
                inside = float(
                    s.dropna().dt.normalize().between(lo, hi + pd.Timedelta(days=366)).mean()
                )
                pr["in_range"] = inside
                crit &= inside >= DATE_RANGE_MIN
            strong = bool(crit and s.notna().mean() >= 0.99)
            d = s.dropna().dt.normalize()
            check = hashlib.sha1(
                d.value_counts().sort_index().to_json(date_format="iso").encode()
            ).hexdigest()[:10]
    elif role == "orders.time":
        rate = inv.time_of_day_rate(s)
        pr["time_of_day"] = rate
        crit = strong = rate >= inv.TIME_MIN_FRAC
        check = round(rate, 3)
    elif role in NUM_ROLES:
        pr["is_number"] = _is_num(s)
        crit &= pr["is_number"]
        if crit:
            check = round(float(pd.to_numeric(s, errors="coerce").sum()), 2)
    elif role in TEXT_ROLES:
        pr["is_text"] = not _is_num(s)
        crit &= pr["is_text"]
        check = int(s.nunique())
        if role == "orders.status":
            n = int(s.dropna().nunique())
            pr["n_values"] = n
            from_note = c.chosen.get("_status_from_note")
            crit &= bool(from_note) or n <= MAX_STATUS_VALUES
    else:
        check = int(s.nunique()) if not s.empty else 0

    ot, lt = c.table("orders"), c.table("lines")
    oo = c.chosen.get("orders.order")
    if role in ("orders.order", "lines.order") and len(df):
        card = float(s.nunique() / len(df))
        pr["key_cardinality"] = card
        crit &= card >= MIN_KEY_CARDINALITY
    if role == "orders.order" and not c.flat and ref.table == ot:
        r = inv.key_unique(df, [ref.column])
        pr["unique"] = r.value
        crit &= r.passed
    if role == "lines.order" and ot and oo and not c.flat:
        cov, fan, _ = inv.join_check(df, ref.column, T[ot], oo)
        pr["join_coverage"], pr["join_fanout"] = cov.value, fan.value
        crit &= cov.passed and fan.passed
        strong = crit
    if role in ("payments.order", "returns.order") and ot and oo:
        R = df
        if role == "returns.order":
            q = c.chosen.get("returns.qty")
            if q and q in R.columns:
                R = R[pd.to_numeric(R[q], errors="coerce").fillna(0) > 0]
        cov, _, _ = inv.join_check(R, ref.column, T[ot], oo)
        pr["ref_coverage"] = cov.value
        crit &= (cov.value or 0) >= REF_MIN_COVER
        strong = crit and (cov.value or 0) >= 0.9
        if role == "returns.order":
            ids = set(norm_series(T[ot][oo]).dropna())
            share = len(ids & set(norm_series(R[ref.column]).dropna())) / max(len(ids), 1)
            pr["orders_in_returns"] = share
            crit &= share <= RETURN_TABLE_MAX_COVER
    if role == "lines.line_amount" and crit and ot and oo and lt and not c.flat:
        lo_ = c.chosen.get("lines.order")
        if lo_:
            best = 0.0
            for oc in T[ot].columns:
                if _is_num(T[ot][oc]) and oc != oo:
                    r = inv.line_order_sum(df, lo_, ref.column, T[ot], oo, oc)
                    best = max(best, r.value or 0.0)
            pr["sum_matches_order_col"] = best
            strong = best >= inv.MIN_SUM_MATCH
    if role == "orders.goods_amount" and crit and lt and not c.flat:
        lo_, la = c.chosen.get("lines.order"), c.chosen.get("lines.line_amount")
        if lo_ and la and oo:
            r = inv.line_order_sum(T[lt], lo_, la, df, oo, ref.column)
            pr["line_sum_match"] = r.value
            crit &= r.passed
            strong = r.passed
    if role == "products.code" and lt and c.chosen.get("lines.product_code"):
        ov = _overlap(T[lt][c.chosen["lines.product_code"]], s)
        pr["overlap_lines"] = ov
        crit &= ov >= MASTER_MIN_OVERLAP
        strong = crit
    if role == "conversion.product" and lt:
        cols = [c.chosen.get("lines.product"), c.chosen.get("lines.product_code")]
        ov = max([_overlap(s, T[lt][x]) for x in cols if x] + [0.0])
        pn = c.chosen.get("products.name")
        if pn and c.table("products"):
            ov = max(ov, _overlap(s, T[c.table("products")][pn]))
        pr["overlap_products"] = ov
        crit &= ov > 0
        strong = ov >= MASTER_MIN_OVERLAP
    return pr, bool(crit), bool(strong), check


def probe_table(group: str, table: str, c: _Ctx) -> tuple[dict, bool]:
    df = c.T[table]
    pr: dict[str, bool | float] = {}
    ok = True
    if group == "orders":
        dates = [x for x in df.columns if pd.api.types.is_datetime64_any_dtype(df[x])]
        pr["has_date"] = bool(dates)
        ok &= bool(dates)
    if group == "lines":
        pr["has_number"] = any(_is_num(df[x]) for x in df.columns)
        ok &= pr["has_number"]
    if group == "returns":
        pr["not_sales_table"] = table not in {c.table("lines"), c.table("orders")}
        ok &= pr["not_sales_table"]
    return pr, bool(ok)


# ---------------------------------------------------------------- chọn ứng viên


def _need(k: int) -> int:
    return min(VERIFY_MIN_VOTES, k)


def choose(role: str, votes: Counter, c: _Ctx, table: str | None) -> RoleBinding:
    """Chọn cột cho một role từ phiếu planner + probe. `table`: bảng đã chọn của nhóm role."""
    rb = RoleBinding(votes={f"{table}.{col}": n for col, n in votes.items()})
    if not votes:
        return rb  # absent: không planner nào đề xuất
    if table is None:
        rb.status = "uncertain"
        rb.evidence.append("bảng của nhóm chưa xác định")
        return rb
    cands = []
    for col, n in votes.most_common():
        ref = ColumnRef(table, col)
        pr, crit, strong, check = probe(role, ref, c)
        cands.append(
            {"ref": ref, "votes": n, "probes": pr, "crit": crit, "strong": strong, "check": check}
        )
    passing = [x for x in cands if x["crit"]]
    failed = [x for x in cands if not x["crit"]]
    for x in failed:
        bad = [
            k for k, v in x["probes"].items() if v is False or (isinstance(v, float) and v < 0.5)
        ]
        rb.evidence.append(
            f"loại {x['ref']} ({x['votes']} phiếu): trượt probe {bad or list(x['probes'])}"
        )
    if not passing:
        rb.status = "uncertain"
        rb.probes = cands[0]["probes"]
        rb.evidence.append("mọi ứng viên trượt probe trọng yếu: không dùng cột nào")
        return rb
    passing.sort(key=lambda x: (x["strong"], x["votes"], sum(1 for v in x["probes"].values() if v is True)),
                 reverse=True)  # fmt: skip
    win = passing[0]
    same = [x for x in passing if x["check"] == win["check"]]
    others = [x for x in passing if x["check"] != win["check"]]
    v = sum(x["votes"] for x in same)
    rb.ref, rb.probes = win["ref"], win["probes"]
    if not others:
        ok = v >= _need(c.k) or win["strong"]
        why = f"{v}/{c.k} phiếu" + (", probe mạnh" if win["strong"] else "")
    else:
        ok = win["strong"] and not any(x["strong"] for x in others)
        why = (
            f"{len(others)} ứng viên khác cũng qua probe nhưng cho kết quả kiểm tra khác: "
            + ", ".join(f"{x['ref']} ({x['votes']} phiếu)" for x in others)
        )
    rb.status = "verified" if ok else "uncertain"
    rb.evidence.append(why)
    pf = [vv for vv in win["probes"].values() if isinstance(vv, bool)]
    probe_score = sum(pf) / len(pf) if pf else 1.0
    rb.confidence = min(1.0, 0.6 * v / c.k + 0.4 * probe_score + (0.2 if win["strong"] else 0.0))
    if not ok:
        rb.confidence = min(rb.confidence, 0.5)
    return rb


# ---------------------------------------------------------------- reconcile


def reconcile(
    plans: list[dict],
    shop: Shop,
    prof: ShopProfile,
    labels: list[str] | None = None,
    k: int | None = None,
    status_votes: dict[str, str] | None = None,
) -> SemanticGraph:
    """`plans`: đầu ra đã chuẩn hoá của các planner (`semantic_planner.planner_roles`), None = planner lỗi.
    `status_votes`: kết quả lần gọi hẹp xếp nhóm trạng thái (một phiếu, phá hoà)."""
    labels = labels or list(DEFAULT_STATUS_LABELS)
    k = k or len(plans) or 1
    good = [p for p in plans if p]
    T = shop.frames()
    g = SemanticGraph(biz_id=shop.biz_id)
    g.status.labels = list(labels)
    g.log = {
        "k_plan": k,
        "planners_ok": len(good),
        "notes": [],
        "overridden": {},
        "probe_failures": {},
    }
    if not good:
        g.reconciled = False
        g.log["notes"].append("không planner nào trả JSON dùng được")
        compute_capabilities(g)
        return g
    for i, p in enumerate(good):
        for n in normalize_plan(p, T):
            g.log["notes"].append(f"planner {i}: {n}")
    c = _Ctx(shop, prof, k)
    c.chosen["_status_from_note"] = sum(p["status_from_note"] for p in good) * 2 > len(good)

    # 1. Bảng của từng nhóm.
    for grp in GROUPS:
        votes = Counter(p["roles"][f"{grp}.table"] for p in good if p["roles"].get(f"{grp}.table"))
        rb = RoleBinding(votes=dict(votes))
        cands = []
        for t, n in votes.most_common():
            pr, ok = probe_table(grp, t, c)
            col_votes = sum(1 for p in good for r, v in p["roles"].items()
                            if r.startswith(grp + ".") and p["roles"].get(f"{grp}.table") == t)  # fmt: skip
            cands.append((ok, n, col_votes, t, pr))
            if not ok:
                rb.evidence.append(f"loại bảng {t!r}: trượt probe {pr}")
        passing = sorted([x for x in cands if x[0]], key=lambda x: (x[1], x[2]), reverse=True)
        if passing:
            _, n, _, t, pr = passing[0]
            rb.ref, rb.probes = ColumnRef(t), pr
            split = len(passing) > 1
            # Bảng giao dịch một dòng như sổ sửa xe có thể không có mã phiếu. Nếu một planner
            # nhận đúng bảng, bảng có ngày + cột tiền và không có bảng dòng hàng, hãy dùng nó
            # làm bảng đơn thay vì để phiếu "event" từ các planner khác hạ capability.
            single_transaction = (
                grp == "orders"
                and len(votes) == 1
                and n < _need(k)
                and col_votes >= 2
                and not any(p["roles"].get("lines.table") for p in good)
                and any(
                    p["roles"].get(role) in T[t].columns and _is_num(T[t][p["roles"][role]])
                    for p in good
                    for role in ("orders.order_total", "orders.goods_amount", "orders.paid")
                )
            )
            rb.status = (
                "verified"
                if (n >= _need(k) and not (split and passing[1][1] == n)) or single_transaction
                else "uncertain"
            )
            rb.confidence = max(n / k, 0.75) if single_transaction else n / k
            rb.evidence.append(
                f"{n}/{k} planner chọn"
                + (f"; khác: {[x[3] for x in passing[1:]]}" if split else "")
                + (
                    "; bảng một dòng có ngày và tiền, dùng làm bảng đơn"
                    if single_transaction
                    else ""
                )
            )
            c.chosen[f"{grp}.table"] = t
        elif cands:
            rb.status = "uncertain"
        g.set_role(f"{grp}.table", rb)
    if c.table("orders") is None and c.table("lines") is not None:
        lrb = g.role("lines.table")
        g.set_role(
            "orders.table",
            RoleBinding(
                lrb.ref, lrb.status, lrb.confidence, {}, ["dùng bảng dòng hàng (file phẳng)"]
            ),
        )
        c.chosen["orders.table"] = c.table("lines")

    # 2. Cột của từng role (planner chọn đúng bảng của nhóm mới được tính phiếu).
    for role in COLUMN_ORDER:
        grp = role.split(".")[0]
        t = c.table(grp)
        votes = Counter(
            p["roles"][role]
            for p in good
            if p["roles"].get(role) and p["roles"].get(f"{grp}.table") == t
        )
        other = [
            p["roles"][role]
            for p in good
            if p["roles"].get(role) and p["roles"].get(f"{grp}.table") != t
        ]
        if t and other:
            # Planner chọn bảng khác cho nhóm nhưng cột cùng tên có trong bảng đã chọn: vẫn tính phiếu.
            for col in other:
                if col in T[t].columns:
                    votes[col] += 1
        rb = choose(role, votes, c, t)
        if votes and rb.ref is not None:
            maj = votes.most_common(1)[0][0]
            if maj != rb.ref.column:
                g.log["overridden"][role] = {"majority": maj, "chosen": rb.ref.column}
        if rb.evidence and rb.status != "verified":
            g.log["probe_failures"][role] = rb.evidence
        if rb.ref is not None:
            c.chosen[role] = rb.ref.column
        g.set_role(role, rb)

    # Nhóm phụ có cột khoá trượt probe (bảng không phải phiếu trả / sổ thu tiền / danh mục): bỏ cả nhóm (như v1),
    # vẫn giữ phiếu bầu và lý do trong evidence.
    for grp, keycol in (("returns", "returns.order"), ("payments", "payments.order"),
                        ("products", "products.code"), ("conversion", "conversion.product")):  # fmt: skip
        kr = g.role(keycol)
        if c.table(grp) and kr.votes and not kr.usable:
            for r in [x for x in COLUMN_ORDER if x.startswith(grp + ".")] + [f"{grp}.table"]:
                old_rb = g.role(r)
                g.set_role(r, RoleBinding(None, "absent", 0.0, old_rb.votes,
                                          old_rb.evidence + [f"bỏ nhóm {grp}: cột khoá {keycol} trượt probe"]))  # fmt: skip
                c.chosen.pop(r, None)
            g.log["notes"].append(f"bỏ nhóm {grp}: cột khoá trượt probe")

    # Bảng phẳng: mã đơn của bảng đơn = mã đơn của dòng hàng.
    if c.flat and not g.role("orders.order").usable and g.role("lines.order").usable:
        g.set_role("orders.order", g.role("lines.order"))
        c.chosen["orders.order"] = c.chosen.get("lines.order")
    if c.flat and not g.role("lines.order").usable and g.role("orders.order").usable:
        g.set_role("lines.order", g.role("orders.order"))
        c.chosen["lines.order"] = c.chosen.get("orders.order")

    _detect_missing_time(g, c, good)
    _reconcile_status(g, c, good, labels, status_votes)
    _reconcile_adjustments(g, c, good)
    _reconcile_misc(g, c, good)
    _observations_and_joins(g, c, good)
    g.money.formula = money_formula(g)
    extra = {}
    for name in sorted({n for p in good for n in p["capabilities"]}):
        n = sum(1 for p in good if p["capabilities"].get(name))
        if n:
            extra[name] = Capability(
                "verified" if n >= _need(k) else "uncertain", [], f"{n}/{k} planner nêu"
            )
    compute_capabilities(g, extra)
    g.log["capabilities"] = {n: cap.status for n, cap in g.capabilities.items()}
    g.log["votes"] = {r: g.role(r).votes for r in COLUMN_ORDER if g.role(r).votes}
    return g


def _detect_missing_time(g: SemanticGraph, c: _Ctx, good: list[dict]) -> None:
    """Mọi planner để `time = null` mà bảng đơn (hoặc bảng dòng của file phẳng) có cột giờ thật → uncertain."""
    if g.role("orders.time").votes:
        return
    for t in [c.table("orders"), c.table("lines")]:
        if not t:
            continue
        cand = []
        for col, p in c.prof.tables[t].columns.items():
            if p.has_time:
                cand.append((col == c.chosen.get("orders.date"), p.time_frac + p.hhmm_frac, col))
        if cand:
            cand.sort(reverse=True)
            col = cand[0][2]
            g.set_role(
                "orders.time",
                RoleBinding(ColumnRef(t, col), "uncertain", 0.4, {},
                            [f"mọi planner để time = null nhưng {t}.{col} có giờ thật (profile)"],
                            {"time_of_day": round(inv.time_of_day_rate(c.T[t][col]), 4)}),
            )  # fmt: skip
            c.chosen["orders.time"] = col
            g.log["notes"].append(f"bỏ sót giờ: {t}.{col}")
            return


def _reconcile_status(g, c: _Ctx, good, labels, status_votes) -> None:
    sp = g.status
    sp.from_note = bool(c.chosen.get("_status_from_note"))
    planned_vals = {v for p in good for v in p["status"]}
    if not g.role("orders.status").usable and planned_vals:
        # Phát hiện bỏ sót: giá trị trạng thái planner nêu nằm trong một cột chữ của bảng đơn / bảng dòng.
        best = None
        for t in {c.table("orders"), c.table("lines")} - {None}:
            for col in c.T[t].columns:
                s = c.T[t][col]
                if _is_num(s) or pd.api.types.is_datetime64_any_dtype(s):
                    continue
                vals = {key(v) for v in s.dropna().astype(str).unique()[:5000]}
                hit = len(vals & {key(v) for v in planned_vals}) / max(len(planned_vals), 1)
                if hit > 0.5 and (best is None or hit > best[0]):
                    best = (hit, t, col)
        if best:
            _, t, col = best
            g.set_role("orders.status", RoleBinding(ColumnRef(t, col), "uncertain", 0.4, {},
                                                    [f"giá trị trạng thái planner nêu nằm ở {t}.{col}"]))  # fmt: skip
            c.chosen["orders.status"] = col
            g.log["notes"].append(f"bỏ sót cột trạng thái: {t}.{col}")
    rb = g.role("orders.status")
    if not rb.usable:
        return
    s = c.T[rb.ref.table][rb.ref.column].dropna().astype(str)
    counts = s.value_counts()
    values = list(counts.index[:MAX_STATUS_VALUES])
    by_key = defaultdict(Counter)
    for p in good:
        for v, lab in p["status"].items():
            by_key[key(v)][lab] += 1
    sv = {key(v): lab for v, lab in (status_votes or {}).items()}
    part, contested_rows, contested_values, unmapped = {}, 0, set(), []
    for v in values:
        kv = key(v)
        votes = Counter(by_key.get(kv, Counter()))
        if kv in sv:
            votes[sv[kv]] += 1
        if not votes:
            unmapped.append(v)
            continue
        top = votes.most_common()
        lab = top[0][0]
        if len(top) > 1 and top[0][1] == top[1][1] and kv in sv:
            lab = sv[kv]
        part[v] = lab
        inc = {key(_canon(x) or x) in _INCLUDED for x in votes}
        if len(inc) > 1:
            contested_rows += int(counts[v])
            contested_values.add(v)
        sp.evidence.setdefault(lab, []).append(f"{v!r}: {dict(votes)}")
    sp.partition = part
    sp.unmapped = unmapped
    r = inv.status_coverage(c.T[rb.ref.table][rb.ref.column], part, labels, sp.from_note)
    sp.coverage = r.value or 0.0
    contest = contested_rows / max(len(s), 1)
    rb.probes = {**rb.probes, "status_coverage": sp.coverage, "contested_rows": round(contest, 4)}
    bad = []
    if not r.passed and not sp.from_note:
        bad.append(r.detail)
    if contest > STATUS_CONTEST_MAX:
        amount_col = next(
            (
                col
                for role in (
                    "orders.order_total",
                    "orders.goods_amount",
                    "lines.line_amount",
                )
                if (col := c.chosen.get(role)) in c.T[rb.ref.table].columns
                and _is_num(c.T[rb.ref.table][col])
            ),
            None,
        )
        zero_value_contest = False
        if amount_col:
            amounts = pd.to_numeric(
                c.T[rb.ref.table].loc[s.index[s.isin(contested_values)], amount_col],
                errors="coerce",
            ).fillna(0)
            zero_value_contest = len(amounts) > 0 and bool((amounts == 0).all())
        if zero_value_contest:
            rb.evidence.append(
                f"{contest:.1%} dòng tranh chấp nhưng tiền bằng 0; không ảnh hưởng doanh thu"
            )
        else:
            bad.append(f"{contest:.1%} dòng có giá trị planner xếp lẫn tính / không tính doanh thu")
    # Tỉ lệ đơn được tính doanh thu.
    oo = c.chosen.get("orders.order")
    if oo and oo in c.T[rb.ref.table].columns:
        src = c.T[rb.ref.table]
        oid = norm_series(src[oo])
        lab = src[rb.ref.column].map(lambda v: part.get(str(v)) if pd.notna(v) else None)
        excl = lab.map(lambda x: pd.notna(x) and key(_canon(x) or x) not in _INCLUDED)
        has = oid.notna()
        share = 1 - float(excl[has].groupby(oid[has]).any().mean()) if has.any() else 1.0
        rb.probes["revenue_orders"] = round(share, 4)
        if share < MIN_REVENUE_ORDERS:
            bad.append(f"chỉ {share:.0%} đơn được tính doanh thu")
    if bad:
        rb.status = "uncertain"
        rb.evidence += bad
        g.log["probe_failures"]["orders.status"] = rb.evidence


def _adj_shape(L: pd.DataFrame, rows: pd.Series, c: _Ctx) -> float:
    """Tỉ lệ dòng (mask `rows`) có hình dạng không phải hàng: không số lượng / số lượng 0, tiền âm, không mã hàng."""
    sub = L[rows]
    if not len(sub):
        return 0.0
    sig = pd.Series(False, index=sub.index)
    q = c.chosen.get("lines.qty")
    if q and q in sub.columns:
        qq = pd.to_numeric(sub[q], errors="coerce")
        sig |= qq.isna() | (qq == 0)
    a = c.chosen.get("lines.line_amount")
    if a and a in sub.columns:
        sig |= pd.to_numeric(sub[a], errors="coerce") < 0
    pc = c.chosen.get("lines.product_code")
    if pc and pc in sub.columns and pc != c.chosen.get("lines.product"):
        sig |= sub[pc].isna()
    return float(sig.mean())


_PLACEHOLDER = re.compile(r"(<[^>]*>|\{[^}]*\}|\[[^\]]*\]|\.\.\.|…|\b[nNxX]\b|\d+)")


def _template_values(v: str, present: dict) -> list[str]:
    """Planner ghi giá trị dạng mẫu ('Dùng <n> điểm', 'Giảm {x}%'): các giá trị thật của cột khớp mẫu (chỗ giữ là
    một đoạn bất kỳ). Mẫu không có chỗ giữ hoặc không có phần chữ thì không khớp gì."""
    parts = _PLACEHOLDER.split(v)
    if len(parts) < 2:
        return []
    lits = [key(x) for i, x in enumerate(parts) if i % 2 == 0]
    if not any(lits):
        return []
    pat = re.compile("^" + r"\s*(.+?)\s*".join(re.escape(x) for x in lits) + "$")
    return sorted(real for k, real in present.items() if pat.match(k))


def _reconcile_adjustments(g, c: _Ctx, good) -> None:
    lt = c.table("lines")
    if not lt:
        return
    L = c.T[lt]
    props = [
        p["adjustments"]
        for p in good
        if p.get("adjustments") and p["adjustments"].get("column") in L.columns
    ]
    if not props:
        # Bỏ sót: dòng không số lượng nhưng có tiền trong bảng dòng hàng mà không planner nào nêu.
        q, a = c.chosen.get("lines.qty"), c.chosen.get("lines.line_amount")
        if q and a:
            odd = (
                pd.to_numeric(L[q], errors="coerce").isna()
                & pd.to_numeric(L[a], errors="coerce").notna()
            )
            if odd.mean() > 0.001:
                g.money.adjustment = RoleBinding(None, "uncertain", 0.3, {},
                                                 [f"{int(odd.sum())} dòng không số lượng mà có tiền, chưa planner nào nêu"])  # fmt: skip
                g.log["notes"].append("có thể bỏ sót dòng điều chỉnh")
        return
    col_votes = Counter(p["column"] for p in props)
    col = col_votes.most_common(1)[0][0]
    kinds = defaultdict(Counter)
    for p in props:
        if p["column"] == col:
            for v, kd in p["values"].items():
                kinds[v][kd] += 1
    present = {key(x): x for x in L[col].dropna().astype(str).unique()}
    keep, ev, kmap = [], [], {}
    for v, kc in kinds.items():
        real = present.get(key(v))
        reals = [real] if real is not None else _template_values(v, present)
        if not reals:
            ev.append(f"{v!r} không có trong cột {col!r}: bỏ")
            continue
        for real in reals:
            if real in kmap:
                continue
            shape = _adj_shape(L, L[col].astype(str) == real, c)
            if shape < ADJ_SHAPE_MIN:
                ev.append(f"{real!r}: chỉ {shape:.0%} dòng có hình dạng không phải hàng: bỏ")
                continue
            keep.append(real)
            kmap[real] = kc.most_common(1)[0][0]
            amt = c.chosen.get("lines.line_amount")
            if kmap[real] in ("other", "fee") and amt in L.columns:
                # Dòng chỉ ghi tiền âm (trừ vào hoá đơn: điểm, voucher) là giảm giá cấp đơn.
                a = pd.to_numeric(L.loc[L[col].astype(str) == real, amt], errors="coerce").dropna()
                if len(a) and bool((a <= 0).all()) and bool((a < 0).any()):
                    kmap[real] = "discount"
            ev.append(
                f"{real!r}: {sum(kc.values())} planner"
                + (f" (mẫu {v!r})" if real != present.get(key(v)) else "")
                + f", {shape:.0%} dòng không phải hàng, loại {kmap[real]}"
            )
    n = col_votes[col]
    st = "verified" if keep and n >= _need(c.k) else ("uncertain" if keep else "absent")
    g.money.adjustment = RoleBinding(
        ColumnRef(lt, col) if keep else None, st, n / c.k, {f"{lt}.{col}": n}, ev
    )
    g.money.adjustment_values = keep
    mask = L[col].astype(str).isin(keep)
    g.money.adjustment_rows = int(mask.sum())
    g.log["adjustment_kinds"] = kmap


def money_formula(g) -> str:
    """Cách dựng doanh thu một đơn từ role (cho người đọc và cho cơ chế giải độc lập)."""
    ga, la = g.role("orders.goods_amount"), g.role("lines.line_amount")
    if ga.ok and la.usable and ga.ref.table != la.ref.table:
        return (
            f"doanh thu một đơn = {ga.ref} (tổng tiền hàng của đơn, đã kiểm khớp tổng {la.ref.column!r} theo đơn; "
            "dùng cột này, không cộng lại tiền dòng)"
        )
    if la.usable:
        adj = " (bỏ dòng điều chỉnh không tính doanh thu)" if g.money.adjustment_values else ""
        return f"doanh thu một đơn = tổng {la.ref} của các dòng thuộc đơn{adj}"
    for r in ("orders.goods_amount", "orders.order_total"):
        if g.role(r).usable:
            return f"doanh thu một đơn = {g.role(r).ref}"
    return ""


def _reconcile_misc(g, c: _Ctx, good) -> None:
    T = c.T
    # Khách lẻ: giá trị planner nêu có thật trong cột khách.
    cust = set()
    for r in ("orders.customer", "orders.customer_code"):
        rb = g.role(r)
        if rb.usable:
            cust |= {key(v) for v in T[rb.ref.table][rb.ref.column].dropna().astype(str).unique()}
    wv = Counter(v for p in good for v in p["walkin_values"] if key(v) in cust)
    if g.role("orders.customer").usable or g.role("orders.customer_code").usable:
        ent = g.entities.setdefault("customer", EntitySpec(kind="customer"))
        ent.walkin_values = sorted(wv)
        ent.canonical = "code" if g.role("orders.customer_code").usable else "name"
    if "product" in g.entities:
        g.entities["product"].canonical = "code" if g.role("lines.product_code").usable else "name"
    # Snapshot: README (tất định) + planner (≥ min(2, K) phiếu).
    snaps = [ColumnRef(t, col) for t, col in c.shop.snapshot_columns]
    sv = Counter(x for p in good for x in p["snapshot_columns"])
    for x, n in sv.items():
        ref = ColumnRef.parse(x, T)
        if ref and n >= _need(c.k) and ref not in snaps:
            snaps.append(ref)
    g.time.snapshot_columns = snaps
    # Quy đổi theo tên hay mã: đo overlap (như v1).
    cv = g.role("conversion.product")
    lt = c.table("lines")
    if cv.usable and lt:
        vals = T[cv.ref.table][cv.ref.column]

        def ov(col):
            return _overlap(vals, T[lt][col]) if col else 0.0

        g.units.by = (
            "code"
            if ov(c.chosen.get("lines.product_code")) > ov(c.chosen.get("lines.product"))
            else "name"
        )
    # Giờ, granularity.
    tr = g.role("orders.time")
    if tr.usable:
        rate = inv.time_of_day_rate(T[tr.ref.table][tr.ref.column])
        g.time.time_parse_rate = rate
        g.time.time_of_day = rate >= inv.TIME_MIN_FRAC
    dr = g.role("orders.date")
    if dr.usable:
        g.time.date_parse_rate = float(T[dr.ref.table][dr.ref.column].notna().mean())
        g.time.granularity = "datetime" if g.time.time_of_day else "date"
    # Thực thể planner thêm (thợ, xe, nhà cung cấp…): một EntitySpec mỗi loại.
    ents = defaultdict(list)
    for p in good:
        for e in p["entities"]:
            ents[key(e["kind"])].append(e)
    for kk, es in ents.items():
        if kk in g.entities:
            continue
        best = Counter((e["table"], e.get("code"), e.get("name")) for e in es).most_common(1)[0]
        (t, code, name), n = best
        st = "verified" if n >= _need(c.k) else "uncertain"
        spec = EntitySpec(kind=es[0]["kind"])
        if code:
            spec.code = RoleBinding(ColumnRef(t, code), st, n / c.k, evidence=[f"{n} planner"])
        if name:
            spec.name = RoleBinding(ColumnRef(t, name), st, n / c.k, evidence=[f"{n} planner"])
        spec.canonical = "code" if code else "name"
        g.entities[es[0]["kind"]] = spec


def _observations_and_joins(g, c: _Ctx, good) -> None:
    T = c.T
    grains = defaultdict(Counter)
    for p in good:
        for t, gr in p["tables"].items():
            grains[t][gr] += 1
    used = {g.role(r).ref.table for r in ("lines.table", "orders.table", "payments.table", "returns.table",
                                          "conversion.table", "products.table") if g.role(r).usable}  # fmt: skip
    used |= set(grains)
    for t in sorted(used):
        if t not in T:
            continue
        tab = c.shop.tables.get(t)
        vc = grains.get(t, Counter())
        if t == c.table("lines"):
            grain = "line"
        elif t == c.table("orders"):
            grain = "order"
        elif t == c.table("payments"):
            grain = "payment"
        else:
            grain = vc.most_common(1)[0][0] if vc else "event"
        keys = c.prof.tables[t].candidate_keys
        ent_keys = [g.role(r).ref.column for r in ("lines.order", "orders.order", "payments.order", "returns.order")
                    if g.role(r).usable and g.role(r).ref.table == t]  # fmt: skip
        g.observations[t] = ObservationSpec(
            table=t,
            grain=grain,
            row_identity=list(keys[0]) if keys else [],
            entity_keys=ent_keys,
            dedupe="full_row_within_order" if tab is not None and tab.dedupe_key else "none",
            duplicates_dropped=int(tab.deduped) if tab is not None else 0,
            votes=dict(vc),
        )
    ot, oo = c.table("orders"), c.chosen.get("orders.order")
    pairs = [("lines.order", "n:1"), ("payments.order", "n:1"), ("returns.order", "n:1")]
    if ot and oo:
        for r, exp in pairs:
            rb = g.role(r)
            if not rb.usable or (r == "lines.order" and c.flat):
                continue
            cov, fan, info = inv.join_check(T[rb.ref.table], rb.ref.column, T[ot], oo)
            g.joins.append(JoinSpec(str(rb.ref), f"{ot}.{oo}", exp, info["measured"], round(cov.value or 0, 4),
                                    info["missing_keys"], None, "verified" if cov.passed and fan.passed else "uncertain"))  # fmt: skip
    # Sản phẩm: dòng hàng ↔ danh mục theo mã; khoá mã thiếu thì thử nối theo tên (fallback).
    pt = c.table("products")
    lt = c.table("lines")
    if pt and lt:
        pc, pn = g.role("products.code"), g.role("products.name")
        lc, ln = g.role("lines.product_code"), g.role("lines.product")
        if lc.usable and pc.usable:
            cov, fan, info = inv.join_check(T[lt], lc.ref.column, T[pt], pc.ref.column)
            g.joins.append(JoinSpec(str(lc.ref), str(pc.ref), "n:1", info["measured"], round(cov.value or 0, 4),
                                    info["missing_keys"], None, "verified" if cov.passed else "uncertain"))  # fmt: skip
        if ln.usable and pn.usable and (not lc.usable or lc.ref.column == ln.ref.column):
            ov = _overlap(T[lt][ln.ref.column], T[pt][pn.ref.column])
            if ov >= MASTER_MIN_OVERLAP:
                g.joins.append(JoinSpec(str(ln.ref), str(pn.ref), "n:1", "", round(ov, 4), 0, "name",
                                        "verified" if ov >= inv.MIN_JOIN_COVERAGE else "uncertain"))  # fmt: skip


# ---------------------------------------------------------------- chạy planner + cache


def cache_key(shop: Shop, model: str, rulebook: str, k: int, digest: str) -> str:
    sig = json.dumps(
        {
            "model": model,
            "schema": SCHEMA_VERSION,
            "prompts": prompt_hash(),
            "rulebook": hashlib.sha1(rulebook.encode()).hexdigest(),
            "data": digest,
            "k": k,
            "config": RECONCILE_CONFIG,
        },
        sort_keys=True,
    )
    return hashlib.sha1(sig.encode()).hexdigest()[:16]


def infer_graph(
    llm,
    shop: Shop,
    rulebook: str = "",
    rule_index=None,
    cache_dir: str | Path | None = None,
    k: int = K_PLAN,
    workers: int = 1,
    prof: ShopProfile | None = None,
) -> SemanticGraph:
    """Graph v2 của shop: đọc cache, không có thì chạy K planner (+ một lần xếp trạng thái), reconcile."""
    import time

    t0 = time.time()
    prof = prof or profile_shop(shop)
    labels = status_labels(rulebook) if rulebook else list(DEFAULT_STATUS_LABELS)
    model = getattr(getattr(llm, "cfg", None), "model", "?")
    cp = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = (
            cache_dir
            / f"graph_v{SCHEMA_VERSION}_{shop.biz_id}_{cache_key(shop, model, rulebook, k, prof.digest)}.json"
        )
        if cp.exists():
            try:
                return SemanticGraph.from_dict(json.loads(cp.read_text(encoding="utf-8")))
            except (ValueError, KeyError):
                pass
    t_prof = time.time() - t0
    msgs = [build_planner_messages(i, shop, prof, labels, rule_index) for i in range(k)]

    def one(m):
        return run_planner(llm, m)

    if workers > 1:
        with ThreadPoolExecutor(min(workers, k)) as ex:
            outs = list(ex.map(one, msgs))
    else:
        outs = [one(m) for m in msgs]
    plans, raw_log = [], []
    for i, (d, text, errs) in enumerate(outs):
        plans.append(planner_roles(d, shop, labels) if d is not None else None)
        raw_log.append({"planner": i, "errors": errs, "raw": text[:6000]})
    # Xếp trạng thái bằng một lần gọi hẹp trên các giá trị của cột trạng thái đa số. Chọn cột trên bản đã chuẩn hoá của đề
    # xuất (bảng phẳng, cột tìm bằng giá trị planner nêu), như reconcile sẽ thấy.
    T = shop.frames()
    normed = []
    for p in plans:
        if p:
            q = {**p, "roles": dict(p["roles"])}
            normalize_plan(q, T)
            normed.append(q)
    st_votes = Counter(
        (p["roles"].get("orders.table"), p["roles"].get("orders.status"))
        for p in normed
        if p["roles"].get("orders.status")
    )
    status_votes = {}
    if st_votes:
        (t, col), _ = st_votes.most_common(1)[0]
        if t in T and col in T[t].columns:
            vals = list(T[t][col].dropna().astype(str).value_counts().index[:MAX_STATUS_VALUES])
            status_votes = classify_statuses(
                llm, vals, labels, status_definitions(rule_index, labels)
            )
    t_llm = time.time() - t0 - t_prof
    g = reconcile(plans, shop, prof, labels, k, status_votes)
    g.log["planner_raw"] = raw_log
    g.log["seconds"] = {
        "profile": round(t_prof, 1),
        "llm": round(t_llm, 1),
        "total": round(time.time() - t0, 1),
    }
    if cp is not None:
        cp.write_text(
            json.dumps(g.to_dict(), ensure_ascii=False, indent=1, default=str), encoding="utf-8"
        )
    return g


def apply_grain_dedupe(graph: SemanticGraph, shop: Shop) -> list[str]:
    """Bỏ dòng trùng theo grain của graph (§8): bảng dòng hàng / bảng đơn có cột mã đơn (role verified hoặc uncertain có
    cột) mà bộ nạp không nhận ra là bảng ghi theo đơn (bộ nạp dò cột mã đơn bằng tên) → dòng giống hệt nhau trong cùng
    đơn tính một lần. Sửa `shop.tables` tại chỗ (dùng chung cho mọi cơ chế giải); ghi vào ObservationSpec và log."""
    notes = []
    for rn in ("lines.order", "orders.order"):
        rb = graph.role(rn)
        if not rb.usable:
            continue
        t = shop.tables.get(rb.ref.table)
        if t is None or t.dedupe_key is not None:
            continue
        from datamate.loaders.shop import _fill_order_level

        _fill_order_level(
            t.df, rb.ref.column
        )  # ô mức đơn trống ở dòng sau (§8), theo mã đơn của graph
        df = t.df
        try:
            dup = df.duplicated() & df[rb.ref.column].notna()
        except TypeError:
            dup = df.astype(str).duplicated() & df[rb.ref.column].notna()
        n = int(dup.sum())
        t.dedupe_key = rb.ref.column
        if n:
            t.df = df[~dup].reset_index(drop=True)
            t.deduped += n
            t.deduped_after_fill += n
        obs = graph.observations.get(t.name)
        if obs is not None:
            obs.dedupe = "full_row_within_order"
            obs.duplicates_dropped = int(t.deduped)
        notes.append(
            f"{t.name}: mã đơn {rb.ref.column!r} theo graph, bỏ {n} dòng trùng trong cùng đơn"
        )
    graph.log.setdefault("notes", []).extend(notes)
    return notes
