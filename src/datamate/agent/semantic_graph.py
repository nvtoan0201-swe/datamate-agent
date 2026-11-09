"""Semantic graph v2 của một shop: mỗi vai trò (role) có trạng thái verified / uncertain / absent.

Graph do K planner LLM đề xuất và `semantic_reconcile` hợp nhất bằng probe trên dữ liệu (không tin phiếu bầu
mù quáng). Mỗi role là một trường của một spec:

- `tables` (TableRoles): bảng dòng hàng, bảng đơn, sổ thu tiền, phiếu trả + cột mã đơn của từng bảng.
- `observations` (ObservationSpec mỗi bảng dùng tới): grain, row identity, khoá thực thể, luật bỏ trùng.
- `joins` (JoinSpec): cặp nối, cardinality dự kiến / đo được, độ phủ.
- `time` (TimeSpec): cột ngày, cột giờ, granularity, cột snapshot, tỉ lệ parse.
- `money` (MoneyComponents): tiền dòng, giảm giá dòng / đơn, phí vận chuyển, VAT, phải trả, đã thu, dòng điều chỉnh.
- `status` (StatusSpec): cột trạng thái / ghi chú, phân vùng giá trị → nhãn của rulebook, coverage.
- `entities` (EntitySpec theo tên): khách, sản phẩm, nhân viên, kênh, phương thức thanh toán, thực thể planner thêm.
- `units` (UnitSpec): số lượng, đơn vị, bảng quy đổi.
- `capabilities`: hàm của role (verified khi mọi role cần thiết verified).

Tên role phẳng (`orders.date`, `lines.line_amount`…) giữ tương thích ánh xạ v1: `to_v1_mapping()` chiếu graph về
dict ánh xạ v1 để `biz.py` và code cũ chạy được trong lúc chuyển.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Literal

SCHEMA_VERSION = 2

Status = Literal["verified", "uncertain", "absent"]
STATES = ("verified", "uncertain", "absent")
GRAINS = ("line", "order", "payment", "snapshot", "event", "master")


@dataclass(frozen=True)
class ColumnRef:
    """Tham chiếu cột thật `table.column`; `column = None` là tham chiếu cả bảng."""

    table: str
    column: str | None = None

    def __str__(self) -> str:
        return self.table if self.column is None else f"{self.table}.{self.column}"

    def to_dict(self) -> dict:
        return {"table": self.table, "column": self.column}

    @classmethod
    def from_dict(cls, d: dict | None) -> ColumnRef | None:
        return None if not d else cls(d["table"], d.get("column"))

    def exists(self, tables: dict) -> bool:
        if self.table not in tables:
            return False
        return self.column is None or self.column in tables[self.table].columns

    @classmethod
    def parse(cls, text: str, tables: dict) -> ColumnRef | None:
        """'Bảng.Cột' → ColumnRef có thật (tên bảng có thể chứa dấu chấm: thử tiền tố dài nhất)."""
        if not text:
            return None
        for t in sorted(tables, key=len, reverse=True):
            if text.startswith(t + "."):
                c = text[len(t) + 1 :]
                if c in tables[t].columns:
                    return cls(t, c)
        return None


@dataclass
class RoleBinding:
    ref: ColumnRef | None = None  # None = shop không có
    status: Status = "absent"
    confidence: float = 0.0  # 0–1, tính từ phiếu planner + kết quả probe, không do LLM tự khai
    votes: dict[str, int] = field(default_factory=dict)  # "table.column" → số planner chọn
    evidence: list[str] = field(default_factory=list)
    probes: dict[str, bool | float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "verified" and self.ref is not None

    @property
    def usable(self) -> bool:
        """Có cột để dùng (verified hoặc uncertain)."""
        return self.ref is not None and self.status != "absent"

    def to_dict(self) -> dict:
        return {
            "ref": self.ref.to_dict() if self.ref else None,
            "status": self.status,
            "confidence": round(float(self.confidence), 4),
            "votes": dict(self.votes),
            "evidence": list(self.evidence),
            "probes": {
                k: (v if isinstance(v, bool) else round(float(v), 4))
                for k, v in self.probes.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> RoleBinding:
        if not d:
            return cls()
        return cls(
            ColumnRef.from_dict(d.get("ref")),
            d.get("status", "absent"),
            float(d.get("confidence", 0.0)),
            dict(d.get("votes") or {}),
            list(d.get("evidence") or []),
            dict(d.get("probes") or {}),
        )


def _rb() -> RoleBinding:
    return RoleBinding()


# ---------------------------------------------------------------- spec


class _Spec:
    """to_dict / from_dict chung: trường RoleBinding, list, dict, giá trị đơn."""

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, (RoleBinding, ColumnRef)):
                out[f.name] = v.to_dict()
            elif isinstance(v, list):
                out[f.name] = [x.to_dict() if isinstance(x, ColumnRef) else x for x in v]
            else:
                out[f.name] = v
        return out

    @classmethod
    def from_dict(cls, d: dict | None):
        obj = cls()
        for f in fields(cls):
            if not d or f.name not in d:
                continue
            cur = getattr(obj, f.name)
            v = d[f.name]
            if isinstance(cur, RoleBinding):
                setattr(obj, f.name, RoleBinding.from_dict(v))
            elif f.name == "snapshot_columns":
                setattr(obj, f.name, [ColumnRef.from_dict(x) for x in v or []])
            else:
                setattr(obj, f.name, v)
        return obj


@dataclass
class TableRoles(_Spec):
    lines: RoleBinding = field(default_factory=_rb)  # bảng dòng hàng (ref.column = None)
    orders: RoleBinding = field(
        default_factory=_rb
    )  # bảng đơn (có thể trùng bảng dòng hàng: file phẳng)
    payments: RoleBinding = field(default_factory=_rb)
    returns: RoleBinding = field(default_factory=_rb)
    line_order: RoleBinding = field(default_factory=_rb)  # cột mã đơn trên bảng dòng hàng
    order_id: RoleBinding = field(default_factory=_rb)  # cột mã đơn trên bảng đơn
    payment_order: RoleBinding = field(default_factory=_rb)
    return_order: RoleBinding = field(default_factory=_rb)


@dataclass
class ObservationSpec(_Spec):
    table: str = ""
    grain: str = "event"  # line | order | payment | snapshot | event | master
    row_identity: list = field(default_factory=list)  # tập cột xác định một quan sát
    entity_keys: list = field(default_factory=list)
    dedupe: str = "none"  # 'full_row_within_order' | 'identity' | 'none'
    duplicates_dropped: int = 0
    votes: dict = field(default_factory=dict)  # grain → số planner


@dataclass
class JoinSpec(_Spec):
    left: str = ""  # "table.column"
    right: str = ""
    expected: str = "n:1"
    measured: str = ""
    coverage: float = 0.0
    missing_keys: int = 0
    fallback: str | None = None  # nối theo 'name' / 'code' / 'alias' khi khoá chính thiếu
    status: str = "absent"


@dataclass
class TimeSpec(_Spec):
    date: RoleBinding = field(default_factory=_rb)  # ngày đặt
    time: RoleBinding = field(
        default_factory=_rb
    )  # giờ tạo đơn (cột riêng hoặc chính cột ngày giờ)
    payment_date: RoleBinding = field(default_factory=_rb)
    granularity: str = "date"  # datetime | date | month
    date_parse_rate: float = 0.0
    time_parse_rate: float = 0.0
    time_of_day: bool = False  # chỉ bật khi có giờ thật
    snapshot_columns: list = field(default_factory=list)  # ColumnRef chỉ đúng tại ngày cuối dữ liệu


@dataclass
class MoneyComponents(_Spec):
    line_amount: RoleBinding = field(default_factory=_rb)  # thành tiền dòng (đã trừ giảm giá dòng)
    line_discount: RoleBinding = field(default_factory=_rb)
    unit_cost: RoleBinding = field(default_factory=_rb)
    master_unit_cost: RoleBinding = field(default_factory=_rb)
    goods_amount: RoleBinding = field(
        default_factory=_rb
    )  # tổng tiền hàng của đơn trước giảm giá cả đơn
    order_discount: RoleBinding = field(default_factory=_rb)
    shipping: RoleBinding = field(default_factory=_rb)
    vat: RoleBinding = field(default_factory=_rb)
    order_total: RoleBinding = field(default_factory=_rb)  # khách phải trả
    paid: RoleBinding = field(default_factory=_rb)  # đã thu
    payment_amount: RoleBinding = field(default_factory=_rb)
    # Dòng điều chỉnh: dòng không phải hàng nằm trong bảng dòng hàng (giảm giá cả đơn, phí, điểm…). Nhận bằng hình
    # dạng dữ liệu (không số lượng, tiền âm, lặp theo đơn) + xác nhận của planner, không bằng tên.
    adjustment: RoleBinding = field(default_factory=_rb)  # cột chứa nhãn dòng điều chỉnh
    adjustment_values: list = field(default_factory=list)
    adjustment_rows: int = 0
    formula: str = ""  # cách dựng tiền đơn từ dòng, ghi cho người đọc / prompt


@dataclass
class StatusSpec(_Spec):
    column: RoleBinding = field(default_factory=_rb)
    from_note: bool = False
    labels: list = field(default_factory=list)  # 4 nhãn đọc từ rulebook đang chạy
    partition: dict = field(default_factory=dict)  # giá trị thô → nhãn
    return_qty: RoleBinding = field(default_factory=_rb)
    coverage: float = 0.0  # tỉ lệ dòng có giá trị trạng thái được xếp nhãn
    unmapped: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)  # nhãn → bằng chứng


@dataclass
class EntitySpec(_Spec):
    kind: str = ""  # customer | product | staff | channel | payment_method | <planner đề xuất>
    name: RoleBinding = field(default_factory=_rb)
    code: RoleBinding = field(default_factory=_rb)
    group: RoleBinding = field(default_factory=_rb)
    master: RoleBinding = field(default_factory=_rb)  # bảng danh mục
    master_code: RoleBinding = field(default_factory=_rb)
    master_name: RoleBinding = field(default_factory=_rb)
    alt: RoleBinding = field(default_factory=_rb)  # cột cùng nghĩa ở bảng khác (sổ thu tiền…)
    walkin_values: list = field(default_factory=list)
    aliases: dict = field(default_factory=dict)  # cách ghi khác → canonical
    canonical: str = "code"  # khoá canonical: 'code' nếu có mã, không thì 'name'
    display: str = "name"


@dataclass
class UnitSpec(_Spec):
    qty: RoleBinding = field(default_factory=_rb)
    qty_base: RoleBinding = field(default_factory=_rb)
    unit: RoleBinding = field(default_factory=_rb)
    conv_table: RoleBinding = field(default_factory=_rb)
    conv_product: RoleBinding = field(default_factory=_rb)
    conv_unit: RoleBinding = field(default_factory=_rb)
    conv_factor: RoleBinding = field(default_factory=_rb)
    by: str = "name"  # cột sản phẩm của bảng quy đổi là tên ('name') hay mã ('code')
    coverage: float = 0.0  # tỉ lệ dòng hàng có đơn vị cơ bản hoặc có hệ số


@dataclass
class Capability(_Spec):
    status: str = "absent"
    roles: list = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------- role phẳng ↔ trường của spec

# Tên role phẳng (tương thích ánh xạ v1) → (spec, [khoá entity], trường).
ROLE_PATHS: dict[str, tuple] = {
    "lines.table": ("tables", "lines"),
    "orders.table": ("tables", "orders"),
    "payments.table": ("tables", "payments"),
    "returns.table": ("tables", "returns"),
    "lines.order": ("tables", "line_order"),
    "orders.order": ("tables", "order_id"),
    "payments.order": ("tables", "payment_order"),
    "returns.order": ("tables", "return_order"),
    "orders.date": ("time", "date"),
    "orders.time": ("time", "time"),
    "payments.date": ("time", "payment_date"),
    "lines.line_amount": ("money", "line_amount"),
    "lines.line_discount": ("money", "line_discount"),
    "lines.unit_cost": ("money", "unit_cost"),
    "products.unit_cost": ("money", "master_unit_cost"),
    "orders.goods_amount": ("money", "goods_amount"),
    "orders.order_discount": ("money", "order_discount"),
    "orders.shipping": ("money", "shipping"),
    "orders.vat": ("money", "vat"),
    "orders.order_total": ("money", "order_total"),
    "orders.paid": ("money", "paid"),
    "payments.amount": ("money", "payment_amount"),
    "lines.adjustment": ("money", "adjustment"),
    "orders.status": ("status", "column"),
    "returns.qty": ("status", "return_qty"),
    "orders.customer": ("entities", "customer", "name"),
    "orders.customer_code": ("entities", "customer", "code"),
    "orders.staff": ("entities", "staff", "name"),
    "orders.channel": ("entities", "channel", "name"),
    "orders.payment_method": ("entities", "payment_method", "name"),
    "payments.method": ("entities", "payment_method", "alt"),
    "lines.product": ("entities", "product", "name"),
    "lines.product_code": ("entities", "product", "code"),
    "products.table": ("entities", "product", "master"),
    "products.code": ("entities", "product", "master_code"),
    "products.name": ("entities", "product", "master_name"),
    "products.group": ("entities", "product", "group"),
    "lines.qty": ("units", "qty"),
    "lines.qty_base": ("units", "qty_base"),
    "lines.unit": ("units", "unit"),
    "conversion.table": ("units", "conv_table"),
    "conversion.product": ("units", "conv_product"),
    "conversion.unit": ("units", "conv_unit"),
    "conversion.factor": ("units", "conv_factor"),
}
ROLE_NAMES = list(ROLE_PATHS)
TABLE_ROLES = [r for r in ROLE_NAMES if r.endswith(".table")]

# Capability → role bắt buộc (mọi role phải có) và role tuỳ chọn (uncertain thì capability uncertain).
CAPABILITIES: dict[str, tuple[list[str], list[str]]] = {
    "order_revenue": (
        ["orders.table", "orders.date"],
        ["orders.order", "lines.order", "orders.status", "lines.adjustment"],
    ),
    "time_of_day": (["orders.time"], []),
    "customer_identity": (["orders.customer|orders.customer_code"], []),
    "product_identity": (["lines.table", "lines.product|lines.product_code"], []),
    "staff": (["orders.staff"], []),
    "debt_payments": (["payments.table", "payments.order", "payments.amount", "payments.date"], []),
    "returns": (["orders.status|returns.table|lines.qty"], []),
    "inventory_lots": ([], []),
    "cost": (["lines.unit_cost|products.unit_cost"], []),
    "units": (["lines.qty"], ["lines.unit", "conversion.table", "lines.qty_base"]),
    "purchase_import": ([], []),
    "warranty": ([], []),
    "installments": ([], []),
}

V1_ROLES: dict[str, list[str]] = {
    "sales_lines": [
        "table", "order", "product", "product_code", "qty", "qty_base", "unit", "line_amount",
        "unit_cost",
    ],
    "orders": [
        "table", "order", "date", "time", "status", "customer", "customer_code", "payment_method",
        "order_total", "goods_amount", "staff", "channel",
    ],
    "payments": ["table", "order", "date", "amount", "method"],
    "conversion": ["table", "product", "unit", "factor", "by"],
    "returns": ["table", "order", "qty"],
    "products": ["table", "code", "name", "group", "unit_cost"],
}  # fmt: skip
_V1_PREFIX = {"sales_lines": "lines"}


@dataclass
class SemanticGraph:
    biz_id: str = ""
    schema_version: int = SCHEMA_VERSION
    tables: TableRoles = field(default_factory=TableRoles)
    observations: dict = field(default_factory=dict)  # tên bảng → ObservationSpec
    joins: list = field(default_factory=list)  # JoinSpec
    time: TimeSpec = field(default_factory=TimeSpec)
    money: MoneyComponents = field(default_factory=MoneyComponents)
    status: StatusSpec = field(default_factory=StatusSpec)
    entities: dict = field(default_factory=dict)  # kind → EntitySpec
    units: UnitSpec = field(default_factory=UnitSpec)
    capabilities: dict = field(default_factory=dict)  # tên → Capability
    reconciled: bool = True  # False nếu planner không reconcile được (lỗi LLM, JSON hỏng hết)
    log: dict = field(default_factory=dict)  # phiếu, probe fail, role đổi so với đa số

    # ------------------------------------------------------------ truy cập role phẳng

    def _spec(self, path: tuple, create: bool = False):
        spec = getattr(self, path[0])
        if path[0] == "entities":
            if path[1] not in spec:
                if not create:
                    return None, path[-1]
                spec[path[1]] = EntitySpec(kind=path[1])
            return spec[path[1]], path[2]
        return spec, path[1]

    def role(self, name: str) -> RoleBinding:
        if name not in ROLE_PATHS:
            raise KeyError(f"role không có: {name!r}")
        spec, f = self._spec(ROLE_PATHS[name])
        return getattr(spec, f) if spec is not None else RoleBinding()

    def set_role(self, name: str, rb: RoleBinding) -> None:
        spec, f = self._spec(ROLE_PATHS[name], create=True)
        setattr(spec, f, rb)

    def col(self, name: str, usable: bool = True) -> str | None:
        """Tên cột của role (None nếu role vắng; `usable=False` chỉ lấy role verified)."""
        rb = self.role(name)
        ok = rb.usable if usable else rb.ok
        return rb.ref.column if ok and rb.ref else None

    def table_of(self, name: str) -> str | None:
        rb = self.role(name)
        return rb.ref.table if rb.usable and rb.ref else None

    def entity(self, kind: str) -> EntitySpec:
        return self.entities.get(kind) or EntitySpec(kind=kind)

    def capability(self, name: str) -> str:
        c = self.capabilities.get(name)
        return c.status if c else "absent"

    @property
    def mapping_ok(self) -> bool:
        """Tương thích v1: ánh xạ dùng được khi tính được doanh thu theo đơn."""
        return self.capability("order_revenue") in ("verified", "uncertain") and all(
            self.role(r).usable for r in ("orders.table", "orders.date")
        )

    def uncertain_roles(self) -> list[str]:
        return [r for r in ROLE_NAMES if self.role(r).status == "uncertain"]

    # ------------------------------------------------------------ chiếu về ánh xạ v1

    def to_v1_mapping(self) -> dict:
        """Dict ánh xạ v1 (semantic.ROLES) dùng role verified / uncertain, để biz.py cũ chạy được."""
        out: dict = {}
        for v1, fs in V1_ROLES.items():
            pre = _V1_PREFIX.get(v1, v1)
            tbl = self.role(f"{pre}.table")
            if not tbl.usable:
                out[v1] = None
                continue
            r = {"table": tbl.ref.table}
            for f in fs[1:]:
                if f == "by":
                    r["by"] = self.units.by
                    continue
                rn = f"{pre}.{f}"
                if rn not in ROLE_PATHS:
                    r[f] = None
                    continue
                rb = self.role(rn)
                r[f] = (
                    rb.ref.column
                    if rb.usable and rb.ref and rb.ref.table == tbl.ref.table
                    else None
                )
            out[v1] = r
        out["status_map"] = dict(self.status.partition)
        out["status_from_note"] = bool(self.status.from_note)
        out["walkin_values"] = list(self.entity("customer").walkin_values)
        out["snapshot_columns"] = [str(c) for c in self.time.snapshot_columns]
        out["adjustment"] = (
            {
                "column": self.money.adjustment.ref.column,
                "values": list(self.money.adjustment_values),
                "kinds": dict(self.log.get("adjustment_kinds") or {}),
            }
            if self.money.adjustment.usable and self.money.adjustment_values
            else None
        )
        return out

    # ------------------------------------------------------------ tóm tắt cho prompt

    def summary(self) -> str:
        """Tóm tắt theo role: `o.ngay ← Bảng.Cột (verified)`; nói rõ role chưa chắc."""
        v1 = self.to_v1_mapping()
        from datamate.agent.semantic import _LINE_OUT, _ORDER_OUT  # tránh vòng import

        lines = []

        def st(rn):
            rb = self.role(rn)
            return "" if rb.status == "verified" else f" ({rb.status}, chưa chắc)"

        od, sl = v1.get("orders"), v1.get("sales_lines")
        if od:
            cols = ", ".join(
                f"o.{out} ← {od['table']}.{od[f]}{st('orders.' + f)}"
                for f, out in _ORDER_OUT.items()
                if od.get(f)
            )
            lines.append(f"- o = dm.orders(): bảng đơn {od['table']!r}; {cols}")
        if sl:
            cols = ", ".join(
                f"L.{out} ← {sl['table']}.{sl[f]}{st('lines.' + f)}"
                for f, out in _LINE_OUT.items()
                if sl.get(f)
            )
            lines.append(f"- L = dm.sold_lines(): bảng dòng hàng {sl['table']!r}; {cols}")
        for role in ("payments", "conversion", "returns", "products"):
            r = v1.get(role)
            if r:
                cols = ", ".join(f"{k}={v!r}" for k, v in r.items() if v and k != "table")
                lines.append(f"- {role}: bảng {r['table']!r} ({cols})")
        if v1.get("walkin_values"):
            lines.append(f"- khách lẻ (o.khach = None): {v1['walkin_values']}")
        adj = v1.get("adjustment")
        if adj:
            lines.append(
                f"- dòng điều chỉnh (không phải hàng bán, đã loại khỏi L và doanh thu): cột {adj['column']!r} "
                f"giá trị {adj['values'][:8]}"
            )
        caps = [f"{n}: {c.status}" for n, c in self.capabilities.items() if c.status != "absent"]
        absent = [n for n, c in self.capabilities.items() if c.status == "absent"]
        if caps:
            lines.append("- khả năng của dữ liệu: " + ", ".join(caps))
        if absent:
            lines.append("- shop KHÔNG có dữ liệu cho: " + ", ".join(absent))
        unc = self.uncertain_roles()
        if unc:
            lines.append(
                "- role chưa chắc (kiểm tra kỹ trước khi dùng): "
                + "; ".join(f"{r} = {self.role(r).ref}" for r in unc)
            )
        return "\n".join(lines)

    def raw_summary(self, revenue_excludes=("discount", "shipping", "vat")) -> str:
        """Ánh xạ ở mức bảng / cột thô cho cơ chế giải độc lập (không dùng dm.orders): bảng, cột theo role, phân vùng
        trạng thái, dòng điều chỉnh, khách lẻ, quy đổi, danh mục; role chưa chắc."""
        out = []
        for grp, label in (("orders", "bảng đơn"), ("lines", "bảng dòng hàng"), ("payments", "sổ thu tiền"),
                           ("returns", "phiếu trả"), ("products", "danh mục hàng"), ("conversion", "bảng quy đổi")):  # fmt: skip
            t = self.role(f"{grp}.table")
            if not t.usable:
                continue
            cols = []
            for rn in ROLE_NAMES:
                if rn.startswith(grp + ".") and rn != f"{grp}.table":
                    rb = self.role(rn)
                    if rb.usable and rb.ref.table == t.ref.table:
                        flag = "" if rb.status == "verified" else " (chưa chắc)"
                        cols.append(f"{rn.split('.', 1)[1]} = {rb.ref.column!r}{flag}")
            out.append(f"- {label} {t.ref.table!r}: " + ", ".join(cols))
        lo, oo = self.role("lines.order"), self.role("orders.order")
        if lo.usable and oo.usable and lo.ref.table != oo.ref.table:
            out.append(
                f"- nối dòng hàng → đơn: {lo.ref} = {oo.ref} (mã chuẩn hoá: số 1001.0 và chuỗi '1001' là một)"
            )
        if self.status.partition:
            out.append(
                f"- phân vùng trạng thái (cột {self.role('orders.status').ref}): "
                + str(dict(self.status.partition))
                + (
                    " — cột ghi chú tự do: ô không nhắc trạng thái là hoàn thành"
                    if self.status.from_note
                    else ""
                )
            )
        if self.money.formula:
            out.append(f"- {self.money.formula}")
        if self.money.adjustment.usable and self.money.adjustment_values:
            kinds = self.log.get("adjustment_kinds") or {}
            parts = [
                f"{v!r} ({kinds.get(v, 'other')}, {'không ' if kinds.get(v, 'other') in revenue_excludes else ''}"
                "tính doanh thu)"
                for v in self.money.adjustment_values
            ]
            out.append(
                f"- dòng điều chỉnh (KHÔNG phải mặt hàng) ở cột {self.money.adjustment.ref}: "
                + ", ".join(parts)
            )
        wv = self.entity("customer").walkin_values
        if wv:
            out.append(f"- khách lẻ (không phải khách): {wv}")
        if self.time.snapshot_columns:
            out.append(
                f"- cột chỉ đúng tại ngày cuối dữ liệu: {[str(c) for c in self.time.snapshot_columns]}"
            )
        if not self.time.time_of_day:
            out.append("- shop KHÔNG ghi giờ tạo đơn (chỉ có ngày)")
        return "\n".join(out)

    # ------------------------------------------------------------ serialize

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "biz_id": self.biz_id,
            "tables": self.tables.to_dict(),
            "observations": {k: v.to_dict() for k, v in self.observations.items()},
            "joins": [j.to_dict() for j in self.joins],
            "time": self.time.to_dict(),
            "money": self.money.to_dict(),
            "status": self.status.to_dict(),
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "units": self.units.to_dict(),
            "capabilities": {k: v.to_dict() for k, v in self.capabilities.items()},
            "reconciled": self.reconciled,
            "log": self.log,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SemanticGraph:
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema_version {d.get('schema_version')!r} ≠ {SCHEMA_VERSION}")
        return cls(
            biz_id=d.get("biz_id", ""),
            tables=TableRoles.from_dict(d.get("tables")),
            observations={
                k: ObservationSpec.from_dict(v) for k, v in (d.get("observations") or {}).items()
            },
            joins=[JoinSpec.from_dict(j) for j in d.get("joins") or []],
            time=TimeSpec.from_dict(d.get("time")),
            money=MoneyComponents.from_dict(d.get("money")),
            status=StatusSpec.from_dict(d.get("status")),
            entities={k: EntitySpec.from_dict(v) for k, v in (d.get("entities") or {}).items()},
            units=UnitSpec.from_dict(d.get("units")),
            capabilities={
                k: Capability.from_dict(v) for k, v in (d.get("capabilities") or {}).items()
            },
            reconciled=bool(d.get("reconciled", True)),
            log=d.get("log") or {},
        )


def capability_status(
    graph: SemanticGraph, required: list[str], optional: list[str]
) -> tuple[str, str]:
    """Capability = hàm của role: thiếu role bắt buộc → absent; role (bắt buộc hoặc tuỳ chọn đang có) uncertain →
    uncertain; còn lại verified. Role 'a|b' nghĩa là có một trong các role."""
    worst = "verified"
    reasons = []
    for spec in required:
        alts = [graph.role(r) for r in spec.split("|")]
        have = [rb for rb in alts if rb.usable]
        if not have:
            # Có ứng viên nhưng đều trượt probe: không biết shop có hay không → uncertain, không phải absent.
            if any(rb.status == "uncertain" for rb in alts):
                worst = "uncertain"
                reasons.append(f"{spec}: ứng viên trượt probe")
                continue
            return "absent", f"thiếu {spec}"
        if all(rb.status == "uncertain" for rb in have):
            worst = "uncertain"
            reasons.append(f"{spec} chưa chắc")
    for r in optional:
        rb = graph.role(r)
        if rb.status == "uncertain":
            worst = "uncertain"
            reasons.append(f"{r} chưa chắc")
    return worst, "; ".join(reasons)


def compute_capabilities(graph: SemanticGraph, extra: dict | None = None) -> None:
    """Gán `graph.capabilities` từ role. `extra`: capability planner đề xuất (tên → trạng thái đã probe)."""
    caps = {}
    for name, (req, opt) in CAPABILITIES.items():
        if not req:
            continue  # chưa có role mô tả (lô hàng, nhập hàng…): chỉ có khi planner đề xuất
        st, why = capability_status(graph, req, opt)
        caps[name] = Capability(st, [r for r in req + opt], why)
    if (
        "time_of_day" in caps
        and caps["time_of_day"].status != "absent"
        and not graph.time.time_of_day
    ):
        caps["time_of_day"] = Capability("uncertain", ["orders.time"], "cột giờ không có giờ thật")
    for name, c in (extra or {}).items():
        caps.setdefault(name, c)
    graph.capabilities = caps


def from_v1_mapping(mapping: dict, tables: dict, biz_id: str = "") -> SemanticGraph:
    """Ánh xạ v1 (đã kiểm tra) → graph v2: role có cột thật là verified, không phiếu bầu. Dùng để so v1 / v2 và
    dịch fixture ánh xạ train; không dùng lúc chạy thật."""
    g = SemanticGraph(biz_id=biz_id)
    for v1, fs in V1_ROLES.items():
        r = mapping.get(v1)
        pre = _V1_PREFIX.get(v1, v1)
        if not r or r.get("table") not in tables:
            continue
        t = r["table"]
        g.set_role(
            f"{pre}.table", RoleBinding(ColumnRef(t), "verified", 1.0, evidence=["ánh xạ v1"])
        )
        for f in fs[1:]:
            rn = f"{pre}.{f}"
            c = r.get(f)
            if rn in ROLE_PATHS and c and c in tables[t].columns:
                g.set_role(
                    rn, RoleBinding(ColumnRef(t, c), "verified", 1.0, evidence=["ánh xạ v1"])
                )
        if v1 == "conversion" and r.get("by") in ("name", "code"):
            g.units.by = r["by"]
    g.status.partition = dict(mapping.get("status_map") or {})
    g.status.from_note = bool(mapping.get("status_from_note"))
    if g.role("orders.customer").usable or g.role("orders.customer_code").usable:
        g.entities.setdefault("customer", EntitySpec(kind="customer"))
        g.entities["customer"].walkin_values = list(mapping.get("walkin_values") or [])
    g.time.snapshot_columns = [
        ref
        for ref in (ColumnRef.parse(x, tables) for x in mapping.get("snapshot_columns") or [])
        if ref
    ]
    adj = mapping.get("adjustment")
    if (
        adj
        and g.table_of("lines.table")
        and adj.get("column") in tables[g.table_of("lines.table")].columns
    ):
        g.money.adjustment = RoleBinding(
            ColumnRef(g.table_of("lines.table"), adj["column"]),
            "verified",
            1.0,
            evidence=["ánh xạ v1"],
        )
        g.money.adjustment_values = list(adj.get("values") or [])
    od = mapping.get("orders") or {}
    tcol = od.get("time") or od.get("date")
    if tcol and od.get("table") in tables and tcol in tables[od["table"]].columns:
        s = tables[od["table"]][tcol]
        if hasattr(s, "dt"):
            v = s.dropna()
            g.time.time_of_day = bool(len(v)) and bool(
                ((v.dt.hour != 0) | (v.dt.minute != 0)).mean() >= 0.5
            )
        else:
            g.time.time_of_day = bool(od.get("time"))
    compute_capabilities(g)
    return g
