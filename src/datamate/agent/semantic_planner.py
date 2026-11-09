"""K semantic planner LLM độc lập: mỗi planner đọc bằng chứng của shop theo một cách trình bày khác và trả về
một đề xuất semantic graph (JSON), không trả đáp án.

Temperature 0, seed cố định: khác nhau giữa các planner đến từ **đầu vào**, không từ sampling.
1. README trước, bảng sau; bảng là thẻ cột + giá trị hay gặp.
2. Dữ liệu trước (profile: kiểu, khoá ứng viên, cặp nối, dòng mẫu thô), README sau.
3. Quan hệ: các bảng như một đồ thị nối + vài đơn mẫu đầy đủ (một đơn và mọi dòng của nó), kèm luật rút gọn về
   tiền / trạng thái.
Dòng mẫu chọn bằng seed từ hash nội dung bảng (không từ `business_id`): chạy lại ra cùng prompt.

Planner có thể bỏ trống role. Đầu ra được chuẩn hoá thành dict role phẳng (`orders.date` → tên cột) cho
`semantic_reconcile`.
"""

from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd

from datamate.agent.profile import ShopProfile, describe_profile, norm_series, sample_rows
from datamate.agent.semantic import describe_tables, parse_json, resolve_names
from datamate.loaders.shop import Shop
from datamate.text import key

log = logging.getLogger(__name__)

# Nhóm role của planner → các trường (cột). `table` là tên bảng.
PLANNER_ROLES: dict[str, list[str]] = {
    "lines": ["table", "order", "product", "product_code", "qty", "qty_base", "unit", "line_amount",
              "line_discount", "unit_cost"],
    "orders": ["table", "order", "date", "time", "status", "customer", "customer_code", "payment_method",
               "goods_amount", "order_discount", "shipping", "vat", "order_total", "paid", "staff", "channel"],
    "payments": ["table", "order", "date", "amount", "method"],
    "returns": ["table", "order", "qty"],
    "conversion": ["table", "product", "unit", "factor"],
    "products": ["table", "code", "name", "group", "unit_cost"],
}  # fmt: skip
ADJUSTMENT_KINDS = ("discount", "shipping", "vat", "fee", "other")
# Số dòng tối đa in cho mỗi mẫu của presentation quan hệ (mỗi bảng). Đủ thấy ô mức đơn lặp / trống giữa các dòng;
# prompt không được lớn theo số dòng dữ liệu khi khoá mẫu không phải mã đơn (mã hàng, khách, bàn).
MAX_SAMPLE_ROWS = 20

SCHEMA = """\
Trả về MỘT đối tượng JSON duy nhất (không giải thích), đúng các khoá sau. Vai trò nào shop không có thì để null,
trường nào không có thì null; KHÔNG đoán khi không chắc. Tên bảng, tên cột chép NGUYÊN VĂN từ bằng chứng.

{{
  "tables": {{"<tên bảng>": "line | order | payment | snapshot | event | master"}},  // hạt của mỗi bảng dùng tới
  "lines": {{  // bảng DÒNG HÀNG: mỗi dòng một mặt hàng trong một đơn; null nếu shop chỉ có bảng hoá đơn
    "table", "order": cột mã đơn, "product": tên hàng, "product_code": mã hàng, "qty": số lượng bán,
    "qty_base": số lượng ĐÃ quy về đơn vị cơ bản (nếu file có sẵn), "unit": đơn vị tính,
    "line_amount": thành tiền của dòng (đã trừ giảm giá dòng), "line_discount": giảm giá trên dòng,
    "unit_cost": giá vốn ghi trên dòng bán
  }},
  "orders": {{  // bảng ĐƠN (mỗi đơn một dòng). Shop chỉ có một bảng phẳng thì table = bảng dòng hàng
    "table", "order": mã đơn, "date": ngày đặt, "time": cột có GIỜ tạo đơn (cột giờ riêng, hoặc chính cột ngày
    nếu cột đó có giờ), "status": cột trạng thái đơn (không có thì cột ghi chú có nhắc trạng thái),
    "customer": tên khách, "customer_code": mã khách, "payment_method": hình thức thanh toán,
    "goods_amount": tổng tiền hàng của đơn TRƯỚC giảm giá cả đơn, "order_discount": giảm giá cả đơn,
    "shipping": phí giao, "vat": thuế, "order_total": tổng khách phải trả cả đơn, "paid": khách đã trả,
    "staff": nhân viên bán, "channel": kênh bán
  }},
  "payments": {{"table", "order", "date", "amount", "method"}},  // các lần thu tiền (sổ quỹ, thanh toán)
  "returns": {{"table", "order", "qty"}},  // bảng phiếu trả hàng RIÊNG (không phải bảng bán hàng)
  "conversion": {{"table", "product", "unit", "factor", "by"}},  // quy đổi: 1 <unit> = factor đơn vị cơ bản;
                                                             // by = "name" nếu product là tên hàng, "code" nếu mã
  "products": {{"table", "code", "name", "group", "unit_cost"}},  // danh mục hàng
  "status_groups": {{{label_keys}}},  // chép NGUYÊN VĂN từng giá trị của cột trạng thái vào đúng một nhóm
  "status_from_note": true nếu trạng thái nằm trong cột ghi chú tự do, false nếu là cột trạng thái,
  "adjustments": {{"column": cột của bảng dòng hàng chứa nhãn dòng, "values": {{"<giá trị>": "discount | shipping | vat | fee | other"}}}},
     // dòng KHÔNG phải mặt hàng nằm lẫn trong bảng dòng hàng (giảm giá cả đơn, dùng điểm, phí giao, thuế,
     // phí khác), nhận ra bằng hình dạng: không số lượng, tiền âm, không mã hàng. null nếu không có
  "walkin_values": [giá trị cột khách nghĩa là khách lẻ / vãng lai / không ghi khách],
  "entities": [{{"kind": "loại thực thể khác (ví dụ thợ, xe, nhà cung cấp…)", "table", "code", "name"}}],
  "snapshot_columns": ["Bảng.Cột"],  // cột số chỉ đúng tại ngày cuối dữ liệu (còn nợ, tồn cuối…)
  "capabilities": {{"<khả năng khác shop có dữ liệu: inventory_lots | purchase_import | warranty | installments | …>": true}}
}}

Luật: trạng thái đơn có đúng các nhãn {labels}; mỗi giá trị của cột trạng thái nằm trong đúng một nhóm.
Không suy luận chỉ từ tên cột: đối chiếu kiểu dữ liệu, giá trị mẫu, README, quan hệ giữa các bảng."""

STATUS_PROMPT = """\
Cửa hàng ghi trạng thái đơn hàng bằng các giá trị dưới đây. Luật của cửa hàng: mỗi đơn có đúng một trong các
trạng thái {labels}.
{definitions}
Xếp MỖI giá trị vào đúng một trạng thái. Trả về một đối tượng JSON: {{"giá trị": "trạng thái", ...}} gồm đủ mọi
giá trị.

Các giá trị: {values}
"""

INTRO = "Bạn đọc dữ liệu bán hàng của một cửa hàng và xác định VAI TRÒ của các bảng, cột (semantic graph)."


# ---------------------------------------------------------------- bằng chứng theo 3 cách trình bày


def _rows_text(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d %H:%M").str.replace(" 00:00", "")
    return d.to_csv(index=False).strip()


def _rows_capped(df: pd.DataFrame) -> str:
    """Như `_rows_text` nhưng tối đa MAX_SAMPLE_ROWS dòng (giữ thứ tự gốc), ghi số dòng còn lại."""
    if len(df) <= MAX_SAMPLE_ROWS:
        return _rows_text(df)
    return (
        _rows_text(df.head(MAX_SAMPLE_ROWS)) + f"\n… còn {len(df) - MAX_SAMPLE_ROWS} dòng cùng khoá"
    )


def evidence_readme_first(shop: Shop, prof: ShopProfile) -> str:
    return (
        f"# README của shop\n{shop.readme.strip()}\n\n"
        "# Các bảng (cột chữ ít giá trị liệt kê đủ, nhiều giá trị chỉ ghi các giá trị hay gặp)\n"
        + describe_tables(shop)
    )


def evidence_data_first(shop: Shop, prof: ShopProfile, k_rows: int = 4) -> str:
    parts = [
        "# Profile dữ liệu (kiểu, khoá ứng viên, cặp nối)",
        describe_profile(prof),
        "# Dòng mẫu thô",
    ]
    for n, t in shop.tables.items():
        tp = prof.tables[n]
        parts.append(f"## {n!r}\n{_rows_text(sample_rows(t.df, k_rows, tp.seed))}")
    parts.append(f"# README của shop\n{shop.readme.strip()}")
    return "\n\n".join(parts)


def _main_join(prof: ShopProfile):
    """Cặp nối n:1 có bên trái lặp nhiều nhất (bảng dòng → bảng đơn), chọn bằng dữ liệu."""
    best = None
    for j in prof.joins:
        lp = prof.tables[j.left_table].columns[j.left_column]
        if not j.right_unique or lp.unique_frac > 0.9 or j.coverage < 0.95:
            continue
        rp = prof.tables[j.right_table].columns[j.right_column]
        score = (rp.n_unique, j.coverage)
        if best is None or score > best[0]:
            best = (score, j)
    return best[1] if best else None


def evidence_relational(shop: Shop, prof: ShopProfile, rule_excerpt: str, n_orders: int = 2) -> str:
    T = shop.frames()
    parts = ["# Quan hệ giữa các bảng (cột trái → cột phải, độ phủ giá trị)"]
    for j in prof.joins[:15]:
        parts.append(
            f"- {j.left_table}.{j.left_column} → {j.right_table}.{j.right_column} "
            f"({'n:1' if j.right_unique else 'n:n?'}, phủ {j.coverage:.0%})"
        )
    if len(prof.joins) == 0:
        parts.append("- (không thấy cặp cột nào nối được)")
    parts.append("# Các bảng")
    for n, tp in prof.tables.items():
        cols = ", ".join(f"{c!r} ({p.kind})" for c, p in tp.columns.items())
        parts.append(f"- {n!r}: {tp.n_rows} dòng; {cols}")
    j = _main_join(prof)
    if j is not None:
        L, O = T[j.left_table], T[j.right_table]
        lk = norm_series(L[j.left_column])
        sizes = lk.value_counts()
        # Ưu tiên khoá có cỡ nhóm như một đơn (2..MAX_SAMPLE_ROWS dòng). Khoá của cặp nối có thể không phải mã đơn
        # (mã hàng → danh mục, khách, bàn): mỗi nhóm hàng trăm dòng, nên mọi mẫu đều cắt còn MAX_SAMPLE_ROWS dòng.
        fit = sizes[(sizes >= 2) & (sizes <= MAX_SAMPLE_ROWS)]
        pool = (
            fit if len(fit) else sizes[sizes >= 2] if (sizes >= 2).any() else sizes
        ).index.tolist()
        seed = prof.tables[j.left_table].seed
        ids = sorted(pool)[:: max(1, len(pool) // 50)][: max(n_orders, 1) * 3]
        import random

        pick = random.Random(seed).sample(ids, min(n_orders, len(ids))) if ids else []
        parts.append(
            f"# {len(pick)} mẫu theo khoá {j.left_table}.{j.left_column} (dòng của {j.left_table!r} và dòng "
            f"tương ứng của {j.right_table!r}; tối đa {MAX_SAMPLE_ROWS} dòng mỗi bảng)"
        )
        for oid in pick:
            parts.append(f"## Mã {oid}")
            parts.append(_rows_capped(O[norm_series(O[j.right_column]) == oid]))
            parts.append(_rows_capped(L[lk == oid]))
    else:
        # Không có bảng đơn riêng: mẫu theo nhóm dòng của bảng lớn nhất.
        n, tp = max(prof.tables.items(), key=lambda kv: kv[1].n_rows)
        parts.append(f"# Dòng mẫu của bảng lớn nhất {n!r}")
        parts.append(_rows_text(sample_rows(T[n], 8, tp.seed)))
    parts.append(f"# README của shop\n{shop.readme.strip()}")
    if rule_excerpt:
        parts.append(f"# Luật về tiền và trạng thái (trích rulebook)\n{rule_excerpt}")
    return "\n\n".join(parts)


def rule_excerpt(rule_index) -> str:
    """Luật mục đầu (đơn hàng, doanh thu) + luật nhắc trạng thái / dòng không phải mặt hàng."""
    if not rule_index:
        return ""
    first = rule_index[0].section
    keep = [
        r for r in rule_index
        if r.section == first or any(w in key(r.text) for w in ("trang thai", "khong phai mat hang"))
    ]  # fmt: skip
    return "\n".join(r.text for r in keep[:20])


PRESENTATIONS = ("readme_first", "data_first", "relational")


def build_planner_messages(
    i: int, shop: Shop, prof: ShopProfile, labels: list[str], rule_index=None
) -> list[dict]:
    kind = PRESENTATIONS[i % len(PRESENTATIONS)]
    if kind == "readme_first":
        ev = evidence_readme_first(shop, prof)
    elif kind == "data_first":
        ev = evidence_data_first(shop, prof)
    else:
        ev = evidence_relational(shop, prof, rule_excerpt(rule_index))
    schema = SCHEMA.format(
        label_keys=", ".join(f'"{lab}": [...]' for lab in labels),
        labels=", ".join(labels),
    )
    return [
        {"role": "system", "content": f"{INTRO}\n\n{schema}"},
        {"role": "user", "content": ev + "\n\nTrả về JSON."},
    ]


def prompt_hash() -> str:
    """Hash các template prompt (khoá cache v2)."""
    import inspect

    src = SCHEMA + STATUS_PROMPT + INTRO + "|".join(PRESENTATIONS)
    for f in (evidence_readme_first, evidence_data_first, evidence_relational, rule_excerpt):
        src += inspect.getsource(f)
    return hashlib.sha1(src.encode()).hexdigest()[:12]


# ---------------------------------------------------------------- chuẩn hoá đầu ra của một planner


def _scalar(v) -> str | None:
    """Ô của JSON planner phải là một tên (chuỗi). Model khác nhau trả khác: list một phần tử, số, object → lấy chuỗi đầu
    tiên trong list, ép số thành chuỗi, bỏ object. Output LLM là dữ liệu không tin cậy, không được làm sập reconcile."""
    if isinstance(v, (list, tuple)):
        v = next(
            (x for x in v if isinstance(x, (str, int, float)) and not isinstance(x, bool)), None
        )
    if v is None or isinstance(v, (dict, bool)):
        return None
    return str(v)


def _blank(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none"))


def normalize_label(v, labels: list[str]) -> str | None:
    k = key(v)
    for lab in labels:
        if key(lab) == k:
            return lab
    for lab in labels:  # "đã hoàn thành" ↔ "hoàn thành", "huỷ bỏ" ↔ "huỷ"
        kl = key(lab)
        if kl and (k.startswith(kl) or kl.startswith(k)) and len(k) >= 2:
            return lab
    return None


def planner_roles(raw: dict, shop: Shop, labels: list[str]) -> dict:
    """JSON của planner → {'roles': {role phẳng: tên cột / tên bảng}, 'status': {giá trị: nhãn}, …}.

    Chỉ chuẩn hoá hình thức (tên gần đúng, nhãn trạng thái); không sửa lựa chọn của planner (việc của reconcile)."""
    groups = {}
    for g, fs in PLANNER_ROLES.items():
        r = raw.get(g)
        if isinstance(r, dict) and not _blank(_scalar(r.get("table"))):
            groups[g] = {f: (None if _blank(_scalar(r.get(f))) else _scalar(r.get(f))) for f in fs}
            if g == "conversion":
                groups[g]["by"] = r.get("by") if r.get("by") in ("name", "code") else None
    resolve_names(groups, shop, roles=PLANNER_ROLES)
    T = shop.frames()
    roles: dict[str, str | None] = {}
    for g, r in groups.items():
        if r.get("table") not in T:
            continue
        roles[f"{g}.table"] = r["table"]
        for f in PLANNER_ROLES[g][1:]:
            c = r.get(f)
            if c and c in T[r["table"]].columns:
                roles[f"{g}.{f}"] = c
    status: dict[str, str] = {}
    sg = raw.get("status_groups")
    if isinstance(sg, dict):
        for lab, vals in sg.items():
            nl = normalize_label(lab, labels)
            vs = [vals] if isinstance(vals, (str, int, float)) else vals or []
            for v in vs if isinstance(vs, (list, tuple, dict)) else []:
                if nl and v is not None and not isinstance(v, (dict, list)):
                    status[str(v)] = nl
    adj = raw.get("adjustments")
    adjustments = None
    if isinstance(adj, dict) and isinstance(adj.get("values"), (dict, list)):
        vals = adj["values"]
        if isinstance(vals, list):
            vals = {str(v): "other" for v in vals}
        adjustments = {
            "column": _scalar(adj.get("column")),
            "values": {
                str(v): (_scalar(k) if _scalar(k) in ADJUSTMENT_KINDS else "other")
                for v, k in vals.items()
            },
        }
    tables = {}
    if isinstance(raw.get("tables"), dict):
        for t, gr in raw["tables"].items():
            name = t if t in T else next((n for n in T if key(n) == key(t)), None)
            if name and isinstance(gr, str) and gr.strip().lower() in (
                "line", "order", "payment", "snapshot", "event", "master",
            ):  # fmt: skip
                tables[name] = gr.strip().lower()
    entities = []
    for e in raw.get("entities") or []:
        tb = _scalar(e.get("table")) if isinstance(e, dict) else None
        if isinstance(e, dict) and _scalar(e.get("kind")) and tb in T:
            cols = T[tb].columns
            ent = {"kind": _scalar(e["kind"]), "table": tb}
            for f in ("code", "name"):
                ent[f] = _scalar(e.get(f)) if _scalar(e.get(f)) in cols else None
            if ent["code"] or ent["name"]:
                entities.append(ent)
    caps = {}
    if isinstance(raw.get("capabilities"), dict):
        caps = {str(k): bool(v) for k, v in raw["capabilities"].items()}
    wv = raw.get("walkin_values")
    sc = raw.get("snapshot_columns")
    by = (groups.get("conversion") or {}).get("by")
    return {
        "roles": roles,
        "status": status,
        "status_from_note": bool(raw.get("status_from_note")),
        "adjustments": adjustments,
        "walkin_values": [str(x) for x in wv] if isinstance(wv, list) else [],
        "tables": tables,
        "entities": entities,
        "snapshot_columns": [str(x) for x in sc] if isinstance(sc, list) else [],
        "capabilities": caps,
        "conversion_by": by,
    }


# ---------------------------------------------------------------- gọi LLM


def run_planner(llm, msgs: list[dict], retries: int = 1) -> tuple[dict | None, str, list[str]]:
    """(JSON, văn bản thô, lỗi). Sửa lại một lần nếu không đọc được JSON."""
    errs: list[str] = []
    text = ""
    for attempt in range(retries + 1):
        try:
            text, _ = llm.chat(msgs, max_tokens=2000, json_mode=True)
        except Exception as e:  # noqa: BLE001 — lỗi server: planner này không có đề xuất
            errs.append(f"LLM lỗi: {e!r}")
            return None, text, errs
        d = parse_json(text)
        if d is not None:
            return d, text, errs
        errs.append("không đọc được JSON")
        if attempt < retries:
            msgs = msgs + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": "Không đọc được JSON. Trả lại TOÀN BỘ một đối tượng JSON hợp lệ.",
                },
            ]
    return None, text, errs


def classify_statuses(
    llm, values: list[str], labels: list[str], definitions: str = ""
) -> dict[str, str]:
    """Một lần gọi hẹp chỉ xếp nhóm các giá trị trạng thái (nhãn và định nghĩa đọc từ rulebook)."""
    if not values or len(values) > 60:
        return {}
    msg = STATUS_PROMPT.format(
        labels=", ".join(f'"{lab}"' for lab in labels),
        definitions=definitions,
        values=json.dumps(values, ensure_ascii=False),
    )
    try:
        text, _ = llm.chat([{"role": "user", "content": msg}], max_tokens=800, json_mode=True)
    except Exception:  # noqa: BLE001
        return {}
    got = parse_json(text) or {}
    out = {}
    for v in values:
        lab = got.get(v)
        nl = normalize_label(lab, labels) if isinstance(lab, str) else None
        if nl:
            out[v] = nl
    return out


def status_definitions(rule_index, labels: list[str]) -> str:
    """Các dòng luật nhắc tới trạng thái đơn hoặc tên một nhãn (định nghĩa nhãn theo rulebook đang chạy)."""
    if not rule_index:
        return ""
    labs = [key(lab) for lab in labels]
    rows = [
        r.text
        for r in rule_index
        if "trang thai" in key(r.text) or any(lab and lab in key(r.name).split() for lab in labs)
    ]
    return "Định nghĩa trong luật:\n" + "\n".join(rows[:8]) if rows else ""


def plan_from_v1(mapping: dict) -> dict:
    """Ánh xạ v1 → JSON như planner trả về (cho fake planner trong test và script đối chiếu, không dùng lúc chạy)."""
    rename = {"sales_lines": "lines"}
    out: dict = {}
    for v1, g in [("sales_lines", "lines"), ("orders", "orders"), ("payments", "payments"),
                  ("returns", "returns"), ("conversion", "conversion"), ("products", "products")]:  # fmt: skip
        r = mapping.get(v1)
        out[rename.get(v1, g)] = dict(r) if r else None
    groups: dict[str, list] = {}
    for v, lab in (mapping.get("status_map") or {}).items():
        groups.setdefault(lab, []).append(v)
    out["status_groups"] = groups
    out["status_from_note"] = bool(mapping.get("status_from_note"))
    out["walkin_values"] = list(mapping.get("walkin_values") or [])
    out["snapshot_columns"] = list(mapping.get("snapshot_columns") or [])
    adj = mapping.get("adjustment")
    out["adjustments"] = (
        {"column": adj["column"], "values": dict.fromkeys(adj["values"], "other")} if adj else None
    )
    return out
