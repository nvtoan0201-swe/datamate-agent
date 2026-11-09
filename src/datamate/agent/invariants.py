"""Invariant dữ liệu: định luật cấu trúc đúng với mọi shop, kiểm trên role của semantic graph.

Mỗi invariant nhận DataFrame + tên cột lấy từ role (không nhận tên cột cứng), trả `InvariantResult`.
`check_graph(graph, shop)` chạy mọi invariant áp dụng được cho một graph; dùng ở reconcile (probe), ở
`build_context` (ghi log, hạ capability) và ở verifier.

Ngưỡng là hằng số có tên. Chọn trên các shop có nhãn: mỗi ngưỡng để mọi shop dev
có ánh xạ đúng đều qua, và các ánh xạ sai có chủ đích trong test (nối nhầm cột, cột tiền sai) đều trượt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from datamate.agent.profile import TIME_MIN_FRAC, norm_series

# Khoá quan sát / mã đơn ở bảng đơn: tỉ lệ trùng tối đa (v1 dùng 1 %, giữ nguyên).
MAX_KEY_DUP = 0.01
# Dòng trùng theo row identity còn lại sau nạp: tối đa 0 dòng.
MAX_IDENTITY_DUP = 0
# Nối n:1: tỉ lệ khoá bên trái có bên phải tối thiểu (v1: mã đơn dòng hàng thiếu ở bảng đơn ≤ 1 %).
MIN_JOIN_COVERAGE = 0.99
# Nối n:1 không được nhân dòng: số dòng sau nối / trước nối tối đa.
MAX_JOIN_FANOUT = 1.0
# Tổng tiền dòng ↔ tiền đơn: tỉ lệ đơn khớp tối thiểu, sai số cho phép (đồng). v1: 95 %, 1 đồng.
MIN_SUM_MATCH = 0.95
SUM_TOL = 1.0
# Cột ngày: tỉ lệ ô parse được tối thiểu (v1: 90 %).
MIN_DATE_PARSE = 0.9
# Trạng thái: tỉ lệ dòng (có giá trị) được xếp nhãn tối thiểu.
MIN_STATUS_COVERAGE = 0.99
# Quy đổi đơn vị: tỉ lệ mặt hàng mà mọi đơn vị trừ đơn vị cơ bản đều có hệ số.
MIN_UNIT_COVERAGE = 0.99


@dataclass
class InvariantResult:
    name: str
    passed: bool
    value: float | int | None
    threshold: float | int | None
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _r(name, passed, value, threshold, detail="") -> InvariantResult:
    v = round(float(value), 4) if isinstance(value, float) else value
    return InvariantResult(name, bool(passed), v, threshold, detail)


# ---------------------------------------------------------------- invariant đơn lẻ


def key_unique(df: pd.DataFrame, cols: list[str], name: str = "key_unique") -> InvariantResult:
    """Observation key duy nhất: tỉ lệ dòng trùng khoá (trên dòng khoá không trống) ≤ MAX_KEY_DUP."""
    sub = df[cols].dropna()
    if not len(sub):
        return _r(name, False, None, MAX_KEY_DUP, f"khoá {cols} trống hết")
    dup = (
        float(norm_series(sub[cols[0]]).duplicated().mean())
        if len(cols) == 1
        else float(sub.astype(str).duplicated().mean())
    )
    return _r(name, dup <= MAX_KEY_DUP, dup, MAX_KEY_DUP, f"khoá {cols}: trùng {dup:.2%}")


def identity_duplicates(
    df: pd.DataFrame, identity: list[str] | None = None, name: str = "identity_duplicates"
) -> InvariantResult:
    """Dòng trùng theo row identity (mặc định mọi cột) còn lại: ≤ MAX_IDENTITY_DUP."""
    cols = identity or list(df.columns)
    try:
        n = int(df.duplicated(subset=cols).sum())
    except TypeError:
        n = int(df[cols].astype(str).duplicated().sum())
    return _r(
        name, n <= MAX_IDENTITY_DUP, n, MAX_IDENTITY_DUP, f"{n} dòng trùng theo {len(cols)} cột"
    )


def join_check(
    left: pd.DataFrame,
    left_col: str,
    right: pd.DataFrame,
    right_col: str,
    expected: str = "n:1",
    name: str = "join",
) -> tuple[InvariantResult, InvariantResult, dict]:
    """(độ phủ, nhân dòng, thông tin đo). So theo giá trị chuẩn (mã số ↔ chuỗi vẫn nối)."""
    lk = norm_series(left[left_col]).dropna()
    rk = norm_series(right[right_col]).dropna()
    lset = set(lk)
    rset = set(rk)
    cov = len(lset & rset) / len(lset) if lset else 0.0
    counts = rk.value_counts()
    fanout = float(lk.map(counts).fillna(1).sum() / len(lk)) if len(lk) else 1.0
    measured = (
        f"{'n' if lk.duplicated().any() else '1'}:{'n' if counts.max() > 1 else '1'}"
        if len(counts)
        else "?"
    )
    info = {
        "coverage": cov,
        "missing_keys": len(lset - rset),
        "fanout": fanout,
        "measured": measured,
    }
    r1 = _r(f"{name}_coverage", cov >= MIN_JOIN_COVERAGE, cov, MIN_JOIN_COVERAGE,
            f"{len(lset - rset)}/{len(lset)} khoá bên trái không có bên phải")  # fmt: skip
    ok_fan = fanout <= MAX_JOIN_FANOUT + 1e-9 if expected.endswith(":1") else True
    r2 = _r(f"{name}_fanout", ok_fan, fanout, MAX_JOIN_FANOUT,
            f"nối {expected} đo được {measured}, số dòng × {fanout:.3f}")  # fmt: skip
    return r1, r2, info


def line_order_sum(
    lines: pd.DataFrame,
    line_order: str,
    line_amount: str,
    orders: pd.DataFrame,
    order_order: str,
    order_amount: str,
    keep: pd.Series | None = None,
    name: str = "line_order_sum",
) -> InvariantResult:
    """Tổng tiền dòng (sau loại dòng điều chỉnh: `keep`) theo đơn ↔ cột tiền của bảng đơn: tỉ lệ đơn khớp
    (sai ≤ SUM_TOL đồng) ≥ MIN_SUM_MATCH."""
    L = lines if keep is None else lines[keep.reindex(lines.index).fillna(False).astype(bool)]
    if not pd.api.types.is_numeric_dtype(L[line_amount]) or not pd.api.types.is_numeric_dtype(
        orders[order_amount]
    ):
        return _r(name, False, None, MIN_SUM_MATCH, "cột tiền không phải số")
    s = (
        pd.to_numeric(L[line_amount], errors="coerce")
        .fillna(0)
        .groupby(norm_series(L[line_order]))
        .sum()
    )
    g = orders.groupby(norm_series(orders[order_order]))[order_amount].first()
    j = pd.concat([s, pd.to_numeric(g, errors="coerce")], axis=1, join="inner").dropna()
    if not len(j):
        return _r(name, False, None, MIN_SUM_MATCH, "không đơn nào nối được")
    ok = float(((j.iloc[:, 0] - j.iloc[:, 1]).abs() <= SUM_TOL).mean())
    return _r(name, ok >= MIN_SUM_MATCH, ok, MIN_SUM_MATCH,
              f"tổng {line_amount!r} theo đơn khớp {order_amount!r} ở {ok:.1%} số đơn")  # fmt: skip


def date_coverage(df: pd.DataFrame, col: str, name: str = "date_parse") -> InvariantResult:
    s = df[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        rate = float(s.notna().mean()) if len(s) else 0.0
    else:
        rate = (
            float(pd.to_datetime(s, errors="coerce", format="mixed").notna().mean())
            if len(s)
            else 0.0
        )
    return _r(name, rate >= MIN_DATE_PARSE, rate, MIN_DATE_PARSE, f"{col!r} parse được {rate:.1%}")


def time_of_day_rate(s: pd.Series) -> float:
    """Tỉ lệ ô có giờ thật (khác 00:00) — cột ngày giờ, cột giờ riêng, hoặc chuỗi HH:MM."""
    v = s.dropna()
    if not len(v):
        return 0.0
    if pd.api.types.is_datetime64_any_dtype(v):
        return float(((v.dt.hour != 0) | (v.dt.minute != 0) | (v.dt.second != 0)).mean())
    t = pd.to_datetime(v.astype(str), errors="coerce", format="mixed")
    if t.notna().mean() < 0.5:
        return 0.0
    t = t.dropna()
    return float(((t.dt.hour != 0) | (t.dt.minute != 0)).mean())


def time_coverage(df: pd.DataFrame, col: str, name: str = "time_of_day") -> InvariantResult:
    rate = time_of_day_rate(df[col])
    return _r(
        name, rate >= TIME_MIN_FRAC, rate, TIME_MIN_FRAC, f"{col!r} có giờ thật ở {rate:.1%} ô"
    )


def status_coverage(values: pd.Series, partition: dict, labels: list[str] | None = None,
                    from_note: bool = False, name: str = "status_coverage") -> InvariantResult:  # fmt: skip
    """Tỉ lệ dòng có giá trị trạng thái được xếp vào một nhãn hợp lệ. Cột ghi chú tự do: ô không có trong phân
    vùng coi là không nhắc trạng thái (hoàn thành), invariant chỉ kiểm nhãn hợp lệ."""
    from datamate.text import key

    v = values.dropna().astype(str)
    if not len(v):
        return _r(name, True, 1.0, MIN_STATUS_COVERAGE, "cột trạng thái trống")
    kp = {key(k): lab for k, lab in partition.items()}
    bad_labels = sorted({lab for lab in partition.values() if labels and lab not in labels})
    if from_note:
        rate = 1.0
    else:
        rate = float(v.map(lambda x: key(x) in kp).mean())
    miss = sorted({x for x in v.unique() if key(x) not in kp})[:10] if not from_note else []
    passed = rate >= MIN_STATUS_COVERAGE and not bad_labels
    detail = f"xếp được {rate:.1%} dòng" + (f"; thiếu {miss}" if miss else "")
    if bad_labels:
        detail += f"; nhãn không có trong rulebook {bad_labels}"
    return _r(name, passed, rate, MIN_STATUS_COVERAGE, detail)


def unit_coverage(
    lines: pd.DataFrame, product: str, unit: str, factors: dict, name: str = "unit_conversion"
) -> InvariantResult:
    """Mỗi mặt hàng: mọi đơn vị trừ tối đa một (đơn vị cơ bản) phải có hệ số quy đổi. `factors`: khoá
    (key(tên hàng), key(đơn vị)) → hệ số."""
    from datamate.text import key

    sub = lines[[product, unit]].dropna()
    if not len(sub):
        return _r(name, True, 1.0, MIN_UNIT_COVERAGE, "không có đơn vị")
    units = sub.drop_duplicates().groupby(product)[unit].apply(list)
    ok = 0
    bad = []
    for p, us in units.items():
        no_fac = [u for u in us if (key(p), key(u)) not in factors]
        if len(no_fac) <= 1:
            ok += 1
        else:
            bad.append((p, no_fac))
    rate = ok / len(units) if len(units) else 1.0
    return _r(name, rate >= MIN_UNIT_COVERAGE, rate, MIN_UNIT_COVERAGE,
              f"{len(bad)} mặt hàng có ≥ 2 đơn vị không hệ số" + (f", ví dụ {bad[:2]}" if bad else ""))  # fmt: skip


def snapshot_safety(as_of, data_end, used: list[str], snapshot: list[str],
                    name: str = "snapshot_safety") -> InvariantResult:  # fmt: skip
    """Cột chỉ đúng tại ngày cuối dữ liệu không được dùng khi as_of_date sớm hơn ngày cuối."""
    early = (
        data_end is not None
        and pd.Timestamp(as_of).normalize() < pd.Timestamp(data_end).normalize()
    )
    bad = sorted(set(used) & set(snapshot)) if early else []
    return _r(
        name,
        not bad,
        len(bad),
        0,
        f"dùng cột snapshot {bad} khi hôm nay < ngày cuối" if bad else "",
    )


def merge_loss(table, name: str = "merge_loss") -> InvariantResult:
    """Gộp sheet / file / bảng không mất bản ghi: số dòng sau gộp + dòng trùng đã bỏ = tổng dòng nguồn."""
    if not getattr(table, "source_rows", 0):
        return _r(name, True, 0, 0, "không có số dòng nguồn")
    lost = int(table.source_rows - len(table.df) - table.deduped)
    return _r(name, lost == 0, lost, 0, f"{table.name}: nguồn {table.source_rows}, còn {len(table.df)}, "
              f"bỏ trùng {table.deduped}")  # fmt: skip


# ---------------------------------------------------------------- chạy trên graph


def conversion_factors(graph, tables: dict) -> dict:
    """(key(sản phẩm), key(đơn vị)) → hệ số, từ UnitSpec."""
    from datamate.text import key

    u = graph.units
    if not (
        u.conv_table.usable
        and u.conv_product.usable
        and u.conv_unit.usable
        and u.conv_factor.usable
    ):
        return {}
    C = tables[u.conv_table.ref.table]
    out = {}
    for p, un, f in zip(C[u.conv_product.ref.column], C[u.conv_unit.ref.column],
                        pd.to_numeric(C[u.conv_factor.ref.column], errors="coerce"), strict=False):  # fmt: skip
        if pd.notna(f):
            out[(key(p), key(un))] = float(f)
    return out


def adjustment_mask(graph, L: pd.DataFrame) -> pd.Series:
    """True ở dòng hàng thật (không phải dòng điều chỉnh) của bảng dòng hàng."""
    from datamate.text import key

    adj = graph.money.adjustment
    if not adj.usable or adj.ref.column not in L.columns or not graph.money.adjustment_values:
        return pd.Series(True, index=L.index)
    vals = {key(v) for v in graph.money.adjustment_values}
    return ~L[adj.ref.column].map(lambda v: key(v) in vals if isinstance(v, str) else False)


def check_graph(graph, shop) -> list[InvariantResult]:
    """Mọi invariant áp dụng được cho graph trên dữ liệu shop."""
    T = shop.frames()
    out: list[InvariantResult] = []
    lt, ot = graph.table_of("lines.table"), graph.table_of("orders.table")
    flat = lt is not None and lt == ot
    for t in shop.tables.values():
        out.append(merge_loss(t))
        if t.dedupe_key:
            out.append(identity_duplicates(t.df, name=f"identity_duplicates[{t.name}]"))
    oo = graph.col("orders.order")
    if ot and oo and not flat:
        out.append(key_unique(T[ot], [oo], name="order_key_unique"))
    lo = graph.col("lines.order")
    if lt and ot and lo and oo and not flat:
        cov, fan, _ = join_check(T[lt], lo, T[ot], oo, "n:1", name="lines_orders")
        out += [cov, fan]
        la = graph.col("lines.line_amount")
        ga = graph.col("orders.goods_amount")
        if la and ga:
            out.append(
                line_order_sum(T[lt], lo, la, T[ot], oo, ga, keep=adjustment_mask(graph, T[lt]))
            )
    for rn, nm in (("orders.date", "order_date_parse"), ("payments.date", "payment_date_parse")):
        t, c = graph.table_of(rn), graph.col(rn)
        if t and c:
            out.append(date_coverage(T[t], c, name=nm))
    t, c = graph.table_of("orders.time"), graph.col("orders.time")
    if t and c:
        out.append(time_coverage(T[t], c))
    t, c = graph.table_of("orders.status"), graph.col("orders.status")
    if t and c:
        out.append(status_coverage(T[t][c], graph.status.partition, graph.status.labels or None,
                                   graph.status.from_note))  # fmt: skip
    for rn in ("payments.order", "returns.order"):
        t, c = graph.table_of(rn), graph.col(rn)
        if t and c and ot and oo:
            cov, _, _ = join_check(T[t], c, T[ot], oo, "n:1", name=rn.split(".")[0] + "_orders")
            out.append(cov)
    un = graph.col("lines.unit")
    if lt and un and graph.units.conv_table.usable and not graph.role("lines.qty_base").usable:
        L = product_frame(graph, T, by=graph.units.by)
        if L is not None:
            out.append(unit_coverage(L, "_sp", un, conversion_factors(graph, T)))
    return out


def product_frame(graph, T: dict, by: str = "name") -> pd.DataFrame | None:
    """Bảng dòng hàng kèm cột `_sp` = khoá sản phẩm theo kiểu `by` ('name' / 'code'); dòng hàng chỉ ghi mã mà
    cần tên thì tra danh mục sản phẩm (mã → tên)."""
    lt = graph.table_of("lines.table")
    if not lt:
        return None
    L = T[lt]
    name, code = graph.col("lines.product"), graph.col("lines.product_code")
    ent = graph.entity("product")
    if by == "code" and code:
        return L.assign(_sp=L[code])
    if by == "name" and name and name != code:
        return L.assign(_sp=L[name])
    src = code or name
    if src and ent.master.usable and ent.master_code.usable and ent.master_name.usable:
        P = T[ent.master.ref.table].dropna(subset=[ent.master_code.ref.column])
        m = dict(
            zip(
                norm_series(P[ent.master_code.ref.column]),
                P[ent.master_name.ref.column],
                strict=False,
            )
        )
        return L.assign(_sp=norm_series(L[src]).map(m))
    return L.assign(_sp=L[src]) if src else None


def failed(results: list[InvariantResult]) -> list[InvariantResult]:
    return [r for r in results if not r.passed]


# Invariant trượt → capability bị hạ xuống uncertain (dữ liệu / graph không đảm bảo bất biến cấu trúc).
INVARIANT_CAPABILITY = {
    "order_key_unique": "order_revenue",
    "lines_orders_coverage": "order_revenue",
    "lines_orders_fanout": "order_revenue",
    "line_order_sum": "order_revenue",
    "order_date_parse": "order_revenue",
    "status_coverage": "order_revenue",
    "identity_duplicates": "order_revenue",
    "merge_loss": "order_revenue",
    "time_of_day": "time_of_day",
    "payment_date_parse": "debt_payments",
    "payments_orders_coverage": "debt_payments",
    "returns_orders_coverage": "returns",
    # unit_conversion không hạ capability: rulebook (§10) quy định đơn vị không có dòng quy đổi là đơn vị cơ bản,
    # hệ số 1 (Biz.base_qty làm đúng như vậy); mặt hàng có hai đơn vị không hệ số vẫn tính được theo luật.
}


def downgrade_capabilities(graph, results: list[InvariantResult]) -> list[str]:
    """Hạ capability verified → uncertain theo invariant trượt; trả về ghi chú."""
    notes = []
    for r in failed(results):
        cap = INVARIANT_CAPABILITY.get(r.name.split("[")[0])
        c = graph.capabilities.get(cap) if cap else None
        if c is not None and c.status == "verified":
            c.status = "uncertain"
            c.reason = (
                c.reason + "; " if c.reason else ""
            ) + f"invariant {r.name} trượt: {r.detail}"
            notes.append(f"{cap}: {r.name} ({r.detail})")
    return notes
