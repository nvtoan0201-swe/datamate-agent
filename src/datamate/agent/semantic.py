"""Ánh xạ ngữ nghĩa mỗi shop một lần: vai trò bảng, cột, trạng thái, khách lẻ (generalization-plan §2).

Trước câu hỏi đầu tiên của một shop, gọi LLM một lần với README + mô tả bảng + giá trị của các cột chữ.
LLM trả JSON mô tả bảng dòng hàng, bảng đơn, ánh xạ trạng thái… `validate` kiểm tra bằng luật; không qua
thì gọi lại một lần kèm lỗi cụ thể; vẫn không qua thì `ok = False` và agent quay về cách cũ (LLM tự chọn
cột ở mỗi câu, ngưỡng abstain chặt hơn). Kết quả cache theo shop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from datamate.biz import Biz
from datamate.loaders.shop import Shop
from datamate.text import key

log = logging.getLogger(__name__)

STATUSES = ("hoàn thành", "huỷ", "trả hàng", "đang giao")

# Vai trò → các trường (cột) của vai trò. `table` là tên bảng trong `tables`.
ROLES: dict[str, list[str]] = {
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

PROMPT = """\
Bạn đọc dữ liệu bán hàng của một cửa hàng và xác định VAI TRÒ của các bảng, cột. Trả về MỘT đối tượng JSON
duy nhất (không giải thích), đúng các khoá sau; vai trò nào shop không có thì để null, trường nào không có thì null.
Tên bảng, tên cột phải chép NGUYÊN VĂN từ mô tả bên dưới.

{{
  "sales_lines": {{  // bảng DÒNG HÀNG: mỗi dòng một mặt hàng trong một đơn. null nếu shop chỉ có bảng hoá đơn
    "table": tên bảng, "order": cột mã đơn (null nếu mỗi dòng là một đơn riêng),
    "product": cột tên hàng, "product_code": cột mã hàng, "qty": cột số lượng bán,
    "qty_base": cột số lượng ĐÃ quy về đơn vị cơ bản / đơn vị lẻ (nếu file có sẵn), "unit": cột đơn vị tính,
    "line_amount": cột thành tiền của dòng (đã trừ giảm giá dòng), "unit_cost": cột giá vốn ghi trên dòng bán
  }},
  "orders": {{  // bảng ĐƠN / HOÁ ĐƠN (mỗi đơn một dòng). Nếu shop chỉ có một bảng phẳng thì dùng chính bảng dòng hàng
    "table": tên bảng, "order": cột mã đơn, "date": cột ngày đặt, "time": cột giờ tạo đơn nếu tách riêng,
    "status": cột trạng thái đơn (không có thì cột ghi chú có nhắc huỷ / trả), "customer": cột tên khách,
    "customer_code": cột mã khách, "payment_method": cột hình thức thanh toán,
    "order_total": cột tổng khách phải trả cả đơn (sau giảm giá cả đơn), "goods_amount": cột tổng tiền hàng
    của đơn TRƯỚC giảm giá cả đơn, "staff": cột nhân viên bán, "channel": cột kênh bán
  }},
  "status_groups": {{"hoàn thành": [các giá trị cột trạng thái nghĩa là hoàn thành], "huỷ": [...], "trả hàng": [...],
                    "đang giao": [...]}},  // chép NGUYÊN VĂN từng giá trị của cột trạng thái vào đúng nhóm
  "status_from_note": true nếu trạng thái nằm trong cột ghi chú tự do, false nếu là cột trạng thái,
  "walkin_values": [các giá trị cột khách nghĩa là khách lẻ / khách vãng lai / không ghi khách],
  "payments": {{"table", "order", "date", "amount", "method"}},  // bảng các lần thu tiền (sổ quỹ, thanh toán)
  "conversion": {{"table", "product", "unit", "factor", "by"}},  // bảng quy đổi đơn vị: 1 <unit> = factor đơn vị cơ bản; "by" = "name" nếu cột product là tên hàng, "code" nếu là mã hàng
  "returns": {{"table", "order", "qty"}},  // bảng phiếu trả hàng riêng (nếu có): mã đơn bị trả; nếu là bảng chi tiết có cột số lượng trả thì "qty" = cột đó (dòng có số lượng trả > 0 mới là trả)
  "products": {{"table", "code", "name", "group", "unit_cost"}},  // danh mục hàng: mã hàng, tên hàng, nhóm hàng, giá vốn
  "snapshot_columns": ["Bảng.Cột", ...]  // cột số chỉ đúng tại ngày cuối dữ liệu (còn nợ, tồn cuối…)
}}

Luật: trạng thái đơn có đúng 4 loại: hoàn thành, huỷ, trả hàng, đang giao. Mỗi giá trị của cột trạng thái
được liệt kê phải nằm trong đúng một nhóm của status_groups. Ô trống, "xong", "đã giao", "thành công" là hoàn thành; khách không lấy, bom hàng là huỷ.

# README của shop
{readme}

# Các bảng (đã nạp; cột chữ ít giá trị liệt kê đủ giá trị, cột chữ nhiều giá trị chỉ ghi các giá trị hay gặp)
{tables}
"""


@dataclass
class Mapping:
    biz_id: str
    data: dict | None
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attempts: int = 0

    def summary(self) -> str:
        """Tóm tắt ánh xạ để đưa vào prompt sinh code: cột chuẩn của dm.orders() / dm.sold_lines() lấy từ
        cột gốc nào (để LLM dùng đúng tên cột chuẩn, không dùng tên trường của ánh xạ)."""
        if not self.ok or not self.data:
            return ""
        d = self.data
        lines = []
        od, sl = d.get("orders"), d.get("sales_lines")
        if od:
            cols = ", ".join(f"o.{out} ← {od[f]!r}" for f, out in _ORDER_OUT.items() if od.get(f))
            lines.append(f"- o = dm.orders(): bảng đơn {od['table']!r}; {cols}")
        if sl:
            cols = ", ".join(f"L.{out} ← {sl[f]!r}" for f, out in _LINE_OUT.items() if sl.get(f))
            lines.append(f"- L = dm.sold_lines(): bảng dòng hàng {sl['table']!r}; {cols}")
        for role in ("payments", "conversion", "returns", "products"):
            r = d.get(role)
            if r:
                cols = ", ".join(f"{k}={v!r}" for k, v in r.items() if v and k != "table")
                lines.append(f"- {role}: bảng {r['table']!r} ({cols})")
        if d.get("walkin_values"):
            lines.append(f"- khách lẻ (o.khach = None): {d['walkin_values']}")
        return "\n".join(lines)


_ORDER_OUT = {"order": "ma_don", "date": "ngay", "time": "gio", "status": "trang_thai",
              "customer": "ten_khach", "customer_code": "ma_khach", "payment_method": "thanh_toan",
              "order_total": "tong_don", "goods_amount": "doanh_thu", "staff": "nhan_vien",
              "channel": "kenh"}  # fmt: skip
_LINE_OUT = {"product": "ten_hang", "product_code": "ma_hang", "qty": "sl", "qty_base": "sl_co_ban",
             "unit": "don_vi", "line_amount": "thanh_tien"}  # fmt: skip


# ---------------------------------------------------------------- mô tả bảng cho prompt


def describe_tables(shop: Shop, max_cat: int = 30, top: int = 8) -> str:
    out = []
    for name, t in shop.tables.items():
        df = t.df
        out.append(f"## Bảng {name!r}: {len(df)} dòng")
        for c in df.columns:
            s = df[c]
            if pd.api.types.is_datetime64_any_dtype(s):
                v = s.dropna()
                rng = f"{v.min():%Y-%m-%d} → {v.max():%Y-%m-%d}" if len(v) else "trống"
                out.append(f"- {c!r}: ngày giờ, {rng}")
            elif pd.api.types.is_numeric_dtype(s):
                v = s.dropna()
                rng = f"{v.min():g} → {v.max():g}" if len(v) else "trống"
                out.append(f"- {c!r}: số, {rng}")
            else:
                v = s.dropna().astype(str)
                vc = v.value_counts()
                blank = int(s.isna().sum())
                bl = f", trống {blank}" if blank else ""
                if not len(vc):
                    out.append(f"- {c!r}: toàn trống")
                elif vc.iloc[0] == 1 and len(vc) > max_cat:
                    ex = ", ".join(f"{k!r}" for k in vc.index[:3])
                    out.append(f"- {c!r}: chữ, mỗi giá trị một dòng ({len(vc)}), ví dụ {ex}{bl}")
                elif len(vc) <= max_cat:
                    vals = ", ".join(f"{k!r}" for k in vc.index)
                    out.append(f"- {c!r}: chữ, {len(vc)} giá trị: {vals}{bl}")
                else:
                    vals = ", ".join(f"{k!r} ({n})" for k, n in vc.head(top).items())
                    out.append(f"- {c!r}: chữ, {len(vc)} giá trị, hay gặp: {vals}{bl}")
    return "\n".join(out)


def build_prompt(shop: Shop) -> str:
    return PROMPT.format(readme=shop.readme.strip(), tables=describe_tables(shop))


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?", "", text)
    m = _JSON.search(text)
    if not m:
        return None
    raw = re.sub(r"//[^\n]*", "", m.group(0))  # bỏ comment kiểu // nếu model chép lại
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


# ---------------------------------------------------------------- kiểm tra bằng luật


def _norm_status(v) -> str | None:
    k = key(v)
    for s in STATUSES:
        if key(s) == k:
            return s
    return {"hoan tat": "hoàn thành", "huy bo": "huỷ", "tra": "trả hàng"}.get(k)


def clean_mapping(d: dict) -> dict:
    """Chuẩn hoá hình thức: vai trò rỗng → None, chỉ giữ trường đã biết, trạng thái về 4 nhãn chuẩn."""
    out: dict = {}
    for role, fields in ROLES.items():
        r = d.get(role)
        if isinstance(r, dict) and r.get("table"):
            out[role] = {f: (r.get(f) if r.get(f) not in ("", "null") else None) for f in fields}
        else:
            out[role] = None
    sm = d.get("status_map") or {}
    groups = d.get("status_groups") or {}
    if isinstance(groups, dict) and groups:
        sm = {}
        for lab, vals in groups.items():
            st = _norm_status(lab)
            for v in [vals] if isinstance(vals, str) else vals or []:
                if st and v is not None:
                    sm[str(v)] = st
    if (
        isinstance(sm, dict)
        and sm
        and all(_norm_status(k) for k in sm)
        and not any(_norm_status(v) for v in sm.values())
    ):
        sm = {v: k for k, v in sm.items()}  # model ghi ngược chiều {nhãn chuẩn: giá trị trong file}
    out["status_map"] = (
        {str(k): (_norm_status(v) or v) for k, v in sm.items()} if isinstance(sm, dict) else {}
    )
    out["status_from_note"] = bool(d.get("status_from_note"))
    wv = d.get("walkin_values") or []
    out["walkin_values"] = [str(x) for x in wv] if isinstance(wv, list) else []
    sc = d.get("snapshot_columns") or []
    out["snapshot_columns"] = [str(x) for x in sc] if isinstance(sc, list) else []
    if out["conversion"] and out["conversion"].get("by") not in ("name", "code"):
        out["conversion"]["by"] = "name"
    return out


_FILE_EXT = re.compile(r"\.(csv|tsv|txt|xlsx|xlsm|xls|sqlite3?|db)$", re.IGNORECASE)


def _table_candidates(t: str, T: dict) -> list[str]:
    """Bảng có thật mà tên model ghi gần đúng chỉ tới, lần lượt: khác hoa thường / dấu; kèm đuôi file; thiếu tiền tố
    'file.'; thừa tên sheet đã gộp vào bảng chính ('File.Sheet' khi chỉ có bảng 'File' và không bảng nào có sheet đó)."""
    t2 = _FILE_EXT.sub("", t.strip())
    for name in (t, t2):
        cand = [n for n in T if key(n) == key(name)] or [
            n for n in T if key(n.rpartition(".")[2]) == key(name)
        ]
        if cand:
            return cand
    head, _, tail = t2.rpartition(".")
    if head and not any(key(n.rpartition(".")[2]) == key(tail) for n in T):
        return [n for n in T if key(n) == key(head)]
    return []


def resolve_names(mapping: dict, shop: Shop, roles: dict | None = None) -> None:
    """Sửa tên bảng, cột model chép gần đúng: khác hoa thường / dấu, hoặc thiếu tiền tố 'file.' của bảng.
    `roles`: bộ vai trò → trường (mặc định ROLES của ánh xạ v1; planner v2 dùng bộ của nó)."""
    T = shop.frames()
    for role in roles or ROLES:
        r = mapping.get(role)
        if not r:
            continue
        t = r.get("table")
        if t not in T:
            cand = _table_candidates(str(t), T) if t else []
            if len(cand) == 1:
                r["table"] = cand[0]
        if r.get("table") not in T:
            continue
        cols = list(T[r["table"]].columns)
        for f, c in list(r.items()):
            if f in ("table", "by") or not c or c in cols:
                continue
            cand = [x for x in cols if key(x) == key(c)] or [
                x for x in cols if key(x).replace(" ", "") == key(c).replace(" ", "")
            ]
            if len(cand) == 1:
                r[f] = cand[0]


# Trường bắt buộc của từng vai trò; trường khác chép sai tên thì bỏ trường đó (cảnh báo), không hỏng cả ánh xạ.
_ESSENTIAL = {"sales_lines": {"order", "line_amount"}, "orders": {"order", "date"}}


def _orders_from_flat(mapping: dict, T: dict, warns: list[str]) -> None:
    """Model để orders = null với shop một bảng phẳng: dựng vai trò orders từ chính bảng dòng hàng,
    chọn cột bằng dữ liệu (cột ngày, cột có giá trị trùng status_groups, cột có giá trị khách lẻ)."""
    sl = mapping.get("sales_lines")
    if mapping.get("orders") or not sl or sl.get("table") not in T:
        return
    L = T[sl["table"]]
    dates = [c for c in L.columns if pd.api.types.is_datetime64_any_dtype(L[c])]
    if not dates:
        return
    date = max(dates, key=lambda c: (L[c].notna().mean(), bool(re.search(r"ngay|date", key(c)))))
    texts = [c for c in L.columns if not pd.api.types.is_numeric_dtype(L[c]) and c not in dates]
    smap = {key(k) for k in mapping.get("status_map") or {}}
    status = None
    if smap:
        best = 0.0
        for c in texts:
            vals = {key(v) for v in L[c].dropna().astype(str).unique()[:200]}
            if vals and len(vals) <= 50:
                hit = len(vals & smap) / len(vals)
                if hit > best:
                    status, best = c, hit
        if best < 0.5:
            status = None
    walk = {key(v) for v in mapping.get("walkin_values") or []}
    customer = None
    for c in texts:
        if walk and walk & {key(v) for v in L[c].dropna().astype(str).unique()[:5000]}:
            customer = c
            break
    od = dict.fromkeys(ROLES["orders"])
    od.update(table=sl["table"], order=sl.get("order"), date=date, status=status, customer=customer)
    mapping["orders"] = od
    warns.append(
        f"orders = null: dựng từ bảng phẳng {sl['table']!r} (ngày {date!r}, trạng thái {status!r})"
    )


def _fixups(mapping: dict, T: dict, warns: list[str]) -> None:
    """Sửa các lựa chọn sai hiển nhiên kiểm được bằng dữ liệu."""
    sl0, od0 = mapping.get("sales_lines"), mapping.get("orders")
    if (
        sl0
        and not sl0.get("line_amount")
        and od0
        and (od0.get("goods_amount") or od0.get("order_total"))
    ):
        warns.append(
            f"sales_lines {sl0['table']!r} không có cột thành tiền: bỏ, dùng tiền của bảng đơn"
        )
        mapping["sales_lines"] = None
    sl0 = mapping.get("sales_lines")
    if od0 and not sl0 and od0.get("order") and od0["table"] in T:
        ids = T[od0["table"]][od0["order"]].dropna()
        amt = od0.get("goods_amount") or od0.get("order_total")
        if len(ids) and ids.duplicated().mean() > 0.01 and amt:
            # "Bảng đơn" có nhiều dòng mỗi mã: thực ra là bảng phẳng mỗi dòng một dòng hàng.
            mapping["sales_lines"] = dict.fromkeys(ROLES["sales_lines"]) | {
                "table": od0["table"], "order": od0["order"], "line_amount": amt,
            }  # fmt: skip
            od0["goods_amount"] = od0["order_total"] = None
            warns.append(f"bảng đơn {od0['table']!r} có nhiều dòng mỗi mã đơn: coi là bảng phẳng")
    sl, od, cv, rt = (mapping.get(k) for k in ("sales_lines", "orders", "conversion", "returns"))
    if rt and (
        rt["table"] in {(sl or {}).get("table"), (od or {}).get("table")} or not rt.get("order")
    ):
        warns.append(
            f"returns trỏ vào bảng bán hàng {rt['table']!r}, không phải bảng phiếu trả: bỏ"
        )
        mapping["returns"] = None
    rt = mapping.get("returns")
    if (
        rt
        and od
        and od.get("order")
        and rt["table"] in T
        and rt.get("order") in T[rt["table"]].columns
    ):
        R = T[rt["table"]]
        if rt.get("qty") in R.columns:
            R = R[pd.to_numeric(R[rt["qty"]], errors="coerce").fillna(0) > 0]
        ids = set(T[od["table"]][od["order"]].dropna().astype(str))
        cover = len(ids & set(R[rt["order"]].dropna().astype(str))) / max(len(ids), 1)
        if cover > 0.5:
            warns.append(
                f"returns: {cover:.0%} số đơn có trong bảng {rt['table']!r}, không phải bảng phiếu trả: bỏ"
            )
            mapping["returns"] = None
    text_fields = {"orders": ["status", "customer", "payment_method", "staff", "channel"],
                   "sales_lines": ["product", "unit"], "payments": ["method"]}  # fmt: skip
    for role, fs in text_fields.items():
        r = mapping.get(role)
        if not r or r.get("table") not in T:
            continue
        for f in fs:
            c = r.get(f)
            if c in T[r["table"]].columns and pd.api.types.is_numeric_dtype(T[r["table"]][c]):
                warns.append(f"{role}.{f} = {c!r} là cột số, không phải cột chữ: bỏ")
                r[f] = None
    if od and sl and od["table"] == sl["table"]:
        for f in ("order_total", "goods_amount"):
            if od.get(f) and od[f] == sl.get("line_amount"):
                warns.append(f"orders.{f} trùng cột thành tiền dòng của file phẳng: bỏ")
                od[f] = None
    if cv and sl and cv.get("product"):
        vals = {key(v) for v in T[cv["table"]][cv["product"]].dropna()}
        L = T[sl["table"]]

        def overlap(col):
            return len(vals & {key(v) for v in L[col].dropna().unique()}) if col else 0

        by = "code" if overlap(sl.get("product_code")) > overlap(sl.get("product")) else "name"
        if overlap(sl.get("product_code")) == overlap(sl.get("product")) == 0:
            warns.append("tên hàng trong bảng quy đổi không khớp bảng dòng hàng")
        cv["by"] = by
    if od and mapping.get("walkin_values"):
        cust = set()
        for f in ("customer", "customer_code"):
            if od.get(f):
                cust |= {key(v) for v in T[od["table"]][od[f]].dropna().unique()}
        keep = [v for v in mapping["walkin_values"] if key(v) in cust]
        if len(keep) < len(mapping["walkin_values"]):
            warns.append(
                f"walkin_values bỏ các giá trị không có trong cột khách: "
                f"{[v for v in mapping['walkin_values'] if v not in keep]}"
            )
        mapping["walkin_values"] = keep


def validate(mapping: dict, shop: Shop) -> tuple[list[str], list[str]]:
    """Trả về (lỗi, cảnh báo). Lỗi làm ánh xạ không dùng được; cảnh báo chỉ ghi log.

    Có thể sửa trực tiếp `mapping` ở các lựa chọn sai hiển nhiên (tên gần đúng, bảng trả hàng trỏ nhầm).
    """
    errs: list[str] = []
    warns: list[str] = []
    T = shop.frames()
    resolve_names(mapping, shop)
    for role, fields in ROLES.items():
        r = mapping.get(role)
        if not r:
            continue
        if r["table"] not in T:
            errs.append(f"{role}.table = {r['table']!r} không phải tên bảng; các bảng: {list(T)}")
            continue
        cols = T[r["table"]].columns
        for f in fields:
            if f in ("table", "by") or not r.get(f):
                continue
            if r[f] not in cols:
                msg = f"{role}.{f} = {r[f]!r} không có trong bảng {r['table']!r}; các cột: {list(cols)}"
                if f in _ESSENTIAL.get(role, set()) or role not in _ESSENTIAL:
                    errs.append(msg)
                else:
                    warns.append(msg + " → bỏ trường này")
                    r[f] = None
    if errs:
        return errs, warns
    _orders_from_flat(mapping, T, warns)
    _fixups(mapping, T, warns)
    sl, od = mapping.get("sales_lines"), mapping.get("orders")
    if not od:
        errs.append(
            "orders không được null: shop chỉ có một bảng phẳng thì orders.table = sales_lines.table, "
            "các cột ngày đặt, trạng thái, khách lấy ngay trong bảng đó"
        )
        return errs, warns
    if od:
        O = T[od["table"]]
        if not od.get("date"):
            errs.append("orders.date (cột ngày đặt) không được để trống")
        elif not pd.api.types.is_datetime64_any_dtype(O[od["date"]]):
            errs.append(f"orders.date = {od['date']!r} không phải cột ngày")
        else:
            d = O[od["date"]]
            if d.notna().mean() < 0.9:
                errs.append(
                    f"orders.date = {od['date']!r} trống {1 - d.notna().mean():.0%} số dòng"
                )
            lo, hi = shop.data_start, shop.data_end
            if lo is not None and hi is not None and d.notna().any():
                inside = d.dropna().dt.normalize().between(lo, hi + pd.Timedelta(days=366))
                if inside.mean() < 0.95:
                    errs.append(f"orders.date = {od['date']!r} nằm ngoài khoảng ngày dữ liệu")
        flat = sl is not None and od["table"] == sl["table"]
        if od.get("order") and not flat and O[od["order"]].dropna().duplicated().mean() > 0.01:
            errs.append(
                f"orders.order = {od['order']!r} không duy nhất trong bảng đơn {od['table']!r}: "
                "bảng đơn phải mỗi đơn một dòng (nếu đây là bảng dòng hàng thì để orders.table = sales_lines.table)"
            )
        if sl and not flat:
            if not sl.get("order") or not od.get("order"):
                errs.append("bảng dòng hàng và bảng đơn tách riêng thì phải có cột mã đơn ở cả hai")
            else:
                a = set(T[sl["table"]][sl["order"]].dropna().astype(str))
                b = set(O[od["order"]].dropna().astype(str))
                miss = len(a - b) / max(len(a), 1)
                if miss > 0.01:
                    errs.append(
                        f"{miss:.0%} mã đơn của sales_lines.order = {sl['order']!r} không có trong "
                        f"orders.order = {od['order']!r}: hai cột không cùng là mã đơn"
                    )
                elif (
                    sl.get("line_amount")
                    and od.get("goods_amount")
                    and pd.api.types.is_numeric_dtype(T[sl["table"]][sl["line_amount"]])
                    and pd.api.types.is_numeric_dtype(O[od["goods_amount"]])
                ):
                    L = T[sl["table"]]
                    s = L.groupby(L[sl["order"]].astype(str))[sl["line_amount"]].sum()
                    g = O.groupby(O[od["order"]].astype(str))[od["goods_amount"]].first()
                    j = pd.concat([s, g], axis=1, join="inner").dropna()
                    if len(j):
                        okr = ((j.iloc[:, 0] - j.iloc[:, 1]).abs() <= 1).mean()
                        if okr < 0.95:
                            errs.append(
                                f"tổng {sl['line_amount']!r} theo đơn chỉ khớp {od['goods_amount']!r} ở "
                                f"{okr:.0%} số đơn: goods_amount phải là tổng tiền hàng trước giảm giá cả đơn "
                                "(để null nếu không có cột đó)"
                            )
        st = od.get("status")
        if st:
            sm = mapping.get("status_map") or {}
            bad = {k: v for k, v in sm.items() if v not in STATUSES}
            if bad:
                errs.append(f"status_map có nhãn không hợp lệ {bad}; chỉ dùng {list(STATUSES)}")
            vals = O[st].dropna().astype(str).unique()
            if not mapping.get("status_from_note") or len(vals) <= 30:
                miss = [v for v in vals if v not in sm and key(v) not in {key(k) for k in sm}]
                if miss:
                    errs.append(f"status_map thiếu các giá trị của cột {st!r}: {miss[:20]}")
            # Đối chiếu với từ khoá: chỉ khi từ khoá có tín hiệu rõ (huỷ / trả / giao) mà LLM xếp khác.
            conflict = {}
            for v, lab in sm.items():
                kw = Biz.status_by_keyword(v)
                if kw != "hoàn thành" and lab in STATUSES and kw != lab:
                    conflict[v] = (lab, kw)
            if conflict:
                errs.append(
                    "status_groups có thể xếp nhầm nhóm, xem lại: "
                    + "; ".join(
                        f"{v!r} đang ở nhóm {a!r}, có vẻ là {b!r}" for v, (a, b) in conflict.items()
                    )
                )
    if sl:
        L = T[sl["table"]]
        for f in ("qty", "qty_base", "line_amount"):
            c = sl.get(f)
            if c and not pd.api.types.is_numeric_dtype(L[c]):
                errs.append(f"sales_lines.{f} = {c!r} không phải cột số")
        if not sl.get("line_amount"):
            errs.append("sales_lines.line_amount (thành tiền dòng) không được để trống")
    elif od and not (od.get("goods_amount") or od.get("order_total")):
        errs.append("không có bảng dòng hàng thì orders phải có goods_amount hoặc order_total")
    if not errs:
        try:
            tot = mapping_totals(mapping, shop)
        except Exception as e:  # noqa: BLE001 — ánh xạ không tính được đơn
            errs.append(f"tính đơn theo ánh xạ lỗi: {e!r}")
        else:
            if tot["all_orders"] == 0:
                errs.append("ánh xạ cho 0 đơn")
            elif tot["orders"] < 0.5 * tot["all_orders"]:
                errs.append(
                    f"chỉ {tot['orders']}/{tot['all_orders']} đơn được tính doanh thu: quá nửa số đơn bị coi "
                    "là huỷ / trả, kiểm tra status_groups và returns"
                )
    pm = mapping.get("payments")
    if pm and (not pm.get("order") or not pm.get("amount") or not pm.get("date")):
        warns.append("payments thiếu order / amount / date, bỏ vai trò payments")
        mapping["payments"] = None
    for sc in mapping.get("snapshot_columns") or []:
        t, _, c = sc.rpartition(".")
        if sc != "Bảng.Cột" and (t not in T or c not in T[t].columns):
            warns.append(f"snapshot_columns: {sc!r} không có, bỏ")
    mapping["snapshot_columns"] = [
        sc
        for sc in mapping.get("snapshot_columns") or []
        if sc.rpartition(".")[0] in T and sc.rpartition(".")[2] in T[sc.rpartition(".")[0]].columns
    ]
    return errs, warns


_OPTIONAL_ROLES = ("payments", "conversion", "returns", "products")


def downgrade_errors(mapping: dict, errs: list[str], warns: list[str]) -> list[str]:
    """Sau lần thử cuối, lỗi không làm hỏng phần chính (bảng đơn, dòng hàng) chỉ là cảnh báo:
    - thiếu giá trị trạng thái: dùng từ khoá dự phòng cho các giá trị đó;
    - LLM xếp trạng thái khác tín hiệu rõ của từ khoá: theo từ khoá;
    - vai trò phụ (sổ thu tiền, quy đổi, phiếu trả, danh mục) sai: bỏ vai trò đó."""
    keep = []
    for e in errs:
        role = e.split(".", 1)[0]
        if e.startswith("status_map thiếu"):
            warns.append(e + " → dùng từ khoá dự phòng cho các giá trị này")
        elif e.startswith("status_groups có thể xếp nhầm"):
            sm = mapping.get("status_map") or {}
            for v in list(sm):
                kw = Biz.status_by_keyword(v)
                if kw != "hoàn thành" and kw != sm[v]:
                    sm[v] = kw
            warns.append(e + " → theo từ khoá")
        elif role in _OPTIONAL_ROLES and mapping.get(role) is not None:
            mapping[role] = None
            warns.append(e + f" → bỏ vai trò {role}")
        else:
            keep.append(e)
    return keep


# ---------------------------------------------------------------- gọi LLM + cache


# Xếp trạng thái: dùng chung prompt của planner v2 (nhãn + câu luật định nghĩa đọc từ rulebook, không ví dụ cách ghi
# đã thấy). Đo trên 20 shop dev: 103/103 giá trị đúng, so với 102/103 của prompt cũ có ví dụ.


def _fill_missing_statuses(
    llm, mapping: dict, shop: Shop, warns: list[str], override: bool = False, rulebook: str = ""
) -> None:
    """Một lần gọi hẹp: chỉ xếp nhóm các giá trị trạng thái (còn thiếu, hoặc mọi giá trị nếu `override`)."""
    from datamate.agent.semantic_planner import classify_statuses, status_definitions
    from datamate.rules import DEFAULT_STATUS_LABELS, split_rules, status_labels

    od = mapping.get("orders") or {}
    st = od.get("status")
    T = shop.frames()
    if not st or od.get("table") not in T or st not in T[od["table"]].columns:
        return
    sm = mapping.setdefault("status_map", {})
    known = set() if override else {key(k) for k in sm}
    vals = [v for v in T[od["table"]][st].dropna().astype(str).unique() if key(v) not in known]
    if not vals or len(vals) > 40:
        return
    labels = status_labels(rulebook) if rulebook else list(DEFAULT_STATUS_LABELS)
    defs = status_definitions(split_rules(rulebook), labels) if rulebook else ""
    got = classify_statuses(llm, vals, labels, defs)
    for v in vals:
        lab = _norm_status(got.get(v, "")) if isinstance(got.get(v), str) else None
        if lab:
            old = next((sm[k] for k in sm if key(k) == key(v)), None)
            for k in [k for k in sm if key(k) == key(v)]:
                del sm[k]
            sm[v] = lab
            if old != lab:
                warns.append(
                    f"trạng thái {v!r}: ánh xạ chính {old!r}, lần xếp riêng {lab!r} → {lab!r}"
                )


def _cache_path(shop: Shop, cache_dir: Path, model: str) -> Path:
    from datamate.agent.semantic_planner import STATUS_PROMPT

    sig = shop.readme + "|" + model + "|" + PROMPT + STATUS_PROMPT
    for n, t in shop.tables.items():
        sig += f"|{n}:{[(c, str(d)) for c, d in t.df.dtypes.items()]}:{len(t.df)}"
    h = hashlib.sha1(sig.encode()).hexdigest()[:12]
    return cache_dir / f"mapping_{shop.biz_id}_{h}.json"


def infer_mapping(
    llm, shop: Shop, cache_dir: str | Path | None = None, retries: int = 1, rulebook: str = ""
) -> Mapping:
    """Ánh xạ của shop: đọc cache, nếu không có thì gọi LLM, kiểm tra, sửa lại tối đa `retries` lần."""
    cp = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = _cache_path(shop, cache_dir, getattr(getattr(llm, "cfg", None), "model", "?"))
        if cp.exists():
            return Mapping(**json.loads(cp.read_text(encoding="utf-8")))
    msgs = [{"role": "user", "content": build_prompt(shop)}]
    data, errs, warns, n = None, ["chưa gọi"], [], 0
    for attempt in range(retries + 1):
        n += 1
        try:
            text, _ = llm.chat(msgs, max_tokens=1500, json_mode=True)
        except Exception as e:  # noqa: BLE001 — lỗi server: không có ánh xạ
            errs = [f"LLM lỗi: {e!r}"]
            break
        raw = parse_json(text)
        if raw is None:
            errs = ["không đọc được JSON"]
        else:
            data = clean_mapping(raw)
            try:
                errs, warns = validate(data, shop)
            except Exception as e:  # noqa: BLE001 — ánh xạ làm hỏng phép kiểm tra: coi là lỗi
                errs, warns = [f"kiểm tra ánh xạ lỗi: {e!r}"], []
        if not errs:
            break
        if attempt < retries:
            msgs = msgs + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": "JSON trên sai:\n- "
                    + "\n- ".join(errs)
                    + "\nSửa và trả lại TOÀN BỘ JSON.",
                },
            ]
    if data is not None and (data.get("orders") or {}).get("status"):
        # Xếp nhóm trạng thái bằng một lần gọi hẹp (việc đơn giản, model nhỏ làm đúng hơn khi tách riêng);
        # nhãn của ánh xạ chính chỉ dùng cho giá trị lần gọi này không xếp được.
        n += 1
        _fill_missing_statuses(llm, data, shop, warns, override=True, rulebook=rulebook)
        try:
            errs, w2 = validate(data, shop)
            warns += [w for w in w2 if w not in warns]
        except Exception as e:  # noqa: BLE001
            errs = [f"kiểm tra ánh xạ lỗi: {e!r}"]
    if data is not None and errs:
        errs = downgrade_errors(data, errs, warns)
        if not errs:
            try:
                errs, w2 = validate(data, shop)
            except Exception as e:  # noqa: BLE001
                errs, w2 = [f"kiểm tra ánh xạ lỗi: {e!r}"], []
            errs = downgrade_errors(data, errs, warns)
            warns += [w for w in w2 if w not in warns]
    m = Mapping(shop.biz_id, data, ok=data is not None and not errs, errors=errs, warnings=warns,
                attempts=n)  # fmt: skip
    for w in warns:
        log.warning("[%s] ánh xạ: %s", shop.biz_id, w)
    if not m.ok:
        log.warning("[%s] ánh xạ không dùng được: %s", shop.biz_id, errs)
    if cp is not None:
        cp.write_text(json.dumps(asdict(m), ensure_ascii=False, indent=1), encoding="utf-8")
    return m


def mapping_totals(mapping: dict, shop: Shop, calendar_csv: str | None = None) -> dict:
    """Tổng doanh thu và số đơn tính doanh thu trên cả dữ liệu theo ánh xạ (để so shop gốc / đột biến)."""
    today = shop.data_end or max(
        t.df[c].max()
        for t in shop.tables.values()
        for c in t.df.columns
        if pd.api.types.is_datetime64_any_dtype(t.df[c])
    )
    dm = Biz(
        today, shop.data_start, shop.data_end, calendar_csv, mapping=mapping, tables=shop.frames()
    )
    o = dm.orders()
    p = dm.until(today)
    return {
        "revenue": dm.revenue(o, p),
        "orders": dm.count_orders(o, p),
        "cancelled": dm.count_cancelled(o, p),
        "all_orders": len(o),
    }
