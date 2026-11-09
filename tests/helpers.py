"""Helper cho test: shop tổng hợp (sinh ngẫu nhiên, không dùng dữ liệu thật) và LLM giả."""

import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- shop synthetic + fake LLM


# Tên cột mặc định (ghi đè qua `names`): ngành, tên hàng, nhãn trạng thái đều do test tự đặt.
DEFAULT_NAMES = {
    "order": "Số HĐ", "date": "Ngày", "time": "Giờ", "customer": "Người mua", "status": "Tình hình",
    "goods": "Tiền hàng", "total": "Phải thu", "product": "Mặt hàng", "code": "Mã hàng", "qty": "SL",
    "price": "Đơn giá", "amount": "Thành tiền", "unit": "ĐVT",
}  # fmt: skip
DEFAULT_STATUSES = {
    "hoàn thành": "xong",
    "huỷ": "khách bỏ",
    "trả hàng": "đã hoàn lại",
    "đang giao": "đang chuyển",
}
DEFAULT_ITEMS = [("Trà gừng", "SP01", 25000), ("Bánh quy", "SP02", 18000), ("Nước cam", "SP03", 30000),
                 ("Kẹo lạc", "SP04", 12000), ("Sữa chua", "SP05", 15000)]  # fmt: skip


def make_orders(n_orders: int = 160, seed: int = 0, items=None, statuses=None, walkin: str = "Khách lẻ",
                adjustment: str | None = None, start: str = "2026-01-01", days: int = 90):  # fmt: skip
    """Đơn + dòng hàng giả, tất định theo seed. Trả về (orders, lines, đáp án).

    `statuses`: nhãn chuẩn → giá trị ghi trong file (chuỗi, hoặc list cách ghi cùng nghĩa: chọn theo seed riêng nên
    dữ liệu đơn không đổi khi đổi cách ghi)."""
    rng = random.Random(seed)
    items = items or DEFAULT_ITEMS
    statuses = statuses or DEFAULT_STATUSES
    labs = list(DEFAULT_STATUSES)
    weights = [0.8, 0.08, 0.05, 0.07]
    t0 = pd.Timestamp(start)
    orders, lines = [], []
    for i in range(n_orders):
        oid = f"H{i + 1:04d}"
        ts = t0 + pd.Timedelta(days=rng.randrange(days), minutes=rng.randrange(6 * 60, 22 * 60))
        lab = rng.choices(labs, weights)[0]
        cust = walkin if rng.random() < 0.3 else f"Khách {rng.randrange(1, 25):02d}"
        n_lines = rng.randint(1, 3)
        picked = rng.sample(items, n_lines)
        goods = 0
        for name, code, price in picked:
            q = rng.randint(1, 4)
            lines.append({"order": oid, "product": name, "code": code, "qty": q, "price": price,
                          "amount": q * price, "unit": "cái", "adj": False})  # fmt: skip
            goods += q * price
        ship = 15000 if rng.random() < 0.3 else 0
        disc = 10000 if (adjustment and rng.random() < 0.25) else 0
        if disc:
            lines.append({"order": oid, "product": adjustment, "code": None, "qty": None, "price": None,
                          "amount": -disc, "unit": None, "adj": True})  # fmt: skip
        orders.append({"order": oid, "ts": ts, "customer": cust, "status": lab, "goods": goods,
                       "total": goods - disc + ship})  # fmt: skip
    O = pd.DataFrame(orders)
    L = pd.DataFrame(lines)
    ok = O[O.status.isin(["hoàn thành", "đang giao"])]
    Lok = L[L.order.isin(set(ok.order)) & ~L.adj]
    truth = {
        "revenue": int(ok.goods.sum()),
        "orders": len(ok),
        "cancelled": int((O.status == "huỷ").sum()),
        "all_orders": len(O),
        "revenue_by_month": {
            str(k): int(v) for k, v in ok.groupby(ok.ts.dt.to_period("M")).goods.sum().items()
        },
        "top_qty": Lok.groupby("product").qty.sum().sort_values(ascending=False).index[0],
        "qty_by_product": {k: int(v) for k, v in Lok.groupby("product").qty.sum().items()},
        "morning_orders": int((ok.ts.dt.hour < 14).sum()),
    }
    rs = random.Random(seed + 1000)
    O["status"] = [
        rs.choice(statuses[lab]) if isinstance(statuses[lab], list) else statuses[lab]
        for lab in O.status
    ]
    return O, L, truth


def _excel_serial(ts: pd.Series, with_time: bool) -> pd.Series:
    d = (ts - pd.Timestamp("1899-12-30")) / pd.Timedelta(days=1)
    return d.round(6) if with_time else d.floordiv(1)


def write_shop(path, O, L, names=None, layout: str = "separate", time_mode: str = "column",
               readme: str = "Cửa hàng thử nghiệm.", fmt: str = "csv", *, row_seed: int | None = None,
               col_seed: int | None = None, split_months: bool = False, date_style: str = "text",
               dup_before: int = 0, dup_after: int = 0, product_mode: str = "both",
               pair_unit: str | None = None, numeric_ids: bool = False, alias_rate: float = 0.0,
               snapshot: bool = False):  # fmt: skip
    """Ghi shop ra thư mục, mỗi tuỳ chọn là một phép biến đổi giữ nghĩa (test metamorphic).

    layout: 'separate' (bảng đơn + chi tiết) | 'flat' (một bảng, ô mức đơn ghi ở mọi dòng).
    time_mode: 'column' (cột giờ HH:MM riêng) | 'datetime' (cột ngày có giờ) | 'none' (chỉ ngày) |
        'stamp' (cột ngày thuần + cột thời điểm đầy đủ ngày giờ riêng).
    date_style: 'text' (dd/mm/yyyy) | 'iso' (timestamp ISO) | 'serial' (số serial Excel, cần fmt='xlsx').
    dup_before: số dòng hàng nhân đôi nguyên dạng; dup_after (bảng phẳng): dòng nhân đôi để trống ô mức đơn (§8).
    product_mode: 'both' | 'code_only' (dòng chỉ ghi mã, danh mục mã → tên) | 'name_only' (dòng chỉ ghi tên).
    pair_unit: tên một mặt hàng: dòng số lượng chẵn ghi theo "cặp" (README: 1 cặp = 2 cái).
    numeric_ids: mã đơn là số (1001) thay cho chuỗi ('H0001'). alias_rate: tỉ lệ dòng ghi tên hàng không dấu.
    snapshot: bảng đơn có cột "Còn nợ" chỉ đúng tại ngày cuối dữ liệu (README ghi)."""
    n = {**DEFAULT_NAMES, **(names or {})}
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    O, L = O.copy(), L.copy()
    rng = random.Random(row_seed if row_seed is not None else 7)
    if numeric_ids:
        conv = {o: 1000 + int(o[1:]) for o in O.order}
        O["order"], L["order"] = O.order.map(conv), L.order.map(conv)
    if alias_rate:
        from datamate.text import strip_accents

        flip = [rng.random() < alias_rate and not a for a in L.adj]
        L.loc[flip, "product"] = L.loc[flip, "product"].map(strip_accents)
    readme_lines = [readme]
    if pair_unit:
        m = (L["product"] == pair_unit) & (L.qty.fillna(1) % 2 == 0)
        L.loc[m, "qty"] = L.loc[m, "qty"] / 2
        L.loc[m, "unit"] = "cặp"
        readme_lines.append(f"- {pair_unit}: 1 cặp = 2 cái")

    def dates(ts, with_time):
        if date_style == "serial":
            return _excel_serial(ts, with_time)
        if date_style == "iso":
            return ts.dt.strftime("%Y-%m-%dT%H:%M:%S" if with_time else "%Y-%m-%d")
        return ts.dt.strftime("%d/%m/%Y %H:%M" if with_time else "%d/%m/%Y")

    od = pd.DataFrame({n["order"]: O.order})
    od[n["date"]] = dates(O.ts, time_mode == "datetime")
    if time_mode == "column":
        od[n["time"]] = O.ts.dt.strftime("%H:%M")
    elif time_mode == "stamp":
        od[n["time"]] = dates(O.ts, True)
    od[n["customer"]] = O.customer
    od[n["status"]] = O.status
    ld = pd.DataFrame({n["order"]: L.order})
    if product_mode in ("both", "name_only"):
        ld[n["product"]] = L["product"]
    if product_mode in ("both", "code_only"):
        ld[n["code"]] = L.code.where(
            L.code.notna(), L["product"] if product_mode == "code_only" else None
        )
    for k in ("qty", "price", "amount", "unit"):
        ld[n[k]] = L[k]
    if product_mode != "both":
        cat = L[~L.adj].drop_duplicates("code")[["code", "product"]]
        _write(
            path / f"DanhMuc.{fmt}",
            pd.DataFrame({n["code"]: cat.code, n["product"]: cat["product"]}),
            fmt,
        )
    if dup_before:
        ld = pd.concat([ld, ld.iloc[rng.sample(range(len(ld)), dup_before)]], ignore_index=True)
    order_cols = [c for c in od.columns if c != n["order"]]
    if layout == "flat":
        main = ld.merge(od, on=n["order"], how="left")
        if dup_after:
            extra = main.iloc[rng.sample(range(len(main)), dup_after)].copy()
            extra[order_cols] = None
            main = pd.concat([main, extra], ignore_index=True)
        tables = {"SoBan": main}
    else:
        od[n["goods"]] = O.goods
        od[n["total"]] = O.total
        if snapshot:
            od["Còn nợ"] = [rng.choice([0, 0, 0, 50000]) for _ in range(len(od))]
            readme_lines.append(
                "- **Còn nợ** là số khách còn thiếu, tính đến hết ngày cuối dữ liệu."
            )
        tables = {"DonHang": od, "ChiTiet": ld}
    month = dict(zip(O.order, O.ts.dt.to_period("M"), strict=True))
    for name, df in tables.items():
        if row_seed is not None:
            df = df.sample(frac=1, random_state=row_seed).reset_index(drop=True)
        if col_seed is not None:
            cols = list(df.columns)
            random.Random(col_seed).shuffle(cols)
            df = df[cols]
        if split_months and name != "DonHang":
            per = df[n["order"]].map(month)
            for pr in sorted(per.dropna().unique()):
                _write(path / f"{name}_T{pr.month}_{pr.year}.{fmt}", df[per == pr], fmt)
        else:
            _write(path / f"{name}.{fmt}", df, fmt)
    (path / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return n


def _write(p, df, fmt):
    if fmt == "xlsx":
        df.to_excel(p, index=False)
    else:
        df.to_csv(p, index=False)


def correct_plan(n: dict, statuses=None, layout="separate", time_mode="column", walkin="Khách lẻ",
                 adjustment: str | None = None, product_mode: str = "both", pair_unit: str | None = None,
                 snapshot: bool = False) -> dict:  # fmt: skip
    """JSON một planner đúng trả về cho shop của `write_shop`."""
    statuses = statuses or DEFAULT_STATUSES
    lt = "SoBan" if layout == "flat" else "ChiTiet"
    ot = "SoBan" if layout == "flat" else "DonHang"
    time = (
        n["time"]
        if time_mode in ("column", "stamp")
        else (n["date"] if time_mode == "datetime" else None)
    )
    groups = {lab: (v if isinstance(v, list) else [v]) for lab, v in statuses.items()}
    plan = {
        "tables": {lt: "line"} if layout == "flat" else {lt: "line", ot: "order"},
        "lines": {"table": lt, "order": n["order"],
                  "product": n["product"] if product_mode != "code_only" else None,
                  "product_code": n["code"] if product_mode != "name_only" else None,
                  "qty": n["qty"], "unit": n["unit"], "line_amount": n["amount"]},
        "orders": {"table": ot, "order": n["order"], "date": n["date"], "time": time, "status": n["status"],
                   "customer": n["customer"]},
        "status_groups": groups,
        "status_from_note": False,
        "walkin_values": [walkin],
        "adjustments": {"column": n["code"] if product_mode == "code_only" else n["product"],
                        "values": {adjustment: "discount"}} if adjustment else None,
    }  # fmt: skip
    if layout != "flat":
        plan["orders"].update(goods_amount=n["goods"], order_total=n["total"])
    if product_mode != "both":
        plan["products"] = {"table": "DanhMuc", "code": n["code"], "name": n["product"]}
    if pair_unit:
        plan["conversion"] = {"table": "README_quy_doi", "product": "ten_hang", "unit": "don_vi_lon",
                              "factor": "he_so", "by": "name"}  # fmt: skip
    if snapshot:
        plan["snapshot_columns"] = ["DonHang.Còn nợ"]
    return plan


class FakeLLM:
    """LLM giả: trả lần lượt các JSON planner; lời gọi xếp trạng thái trả `status` (giá trị → nhãn)."""

    def __init__(self, plans: list, status: dict | None = None):
        self.plans = plans
        self.status = status or {}
        self.calls = Counter()

        class _C:
            model = "fake"

        self.cfg = _C()

    def chat(self, messages, max_tokens=None, json_mode=False):
        if messages[0]["content"].startswith("Cửa hàng ghi trạng thái"):
            self.calls["status"] += 1
            return json.dumps(self.status, ensure_ascii=False), {}
        i = self.calls["plan"]
        self.calls["plan"] += 1
        p = self.plans[i % len(self.plans)]
        if isinstance(p, Exception):
            raise p
        return (p if isinstance(p, str) else json.dumps(p, ensure_ascii=False)), {}
