"""Split synthetic cho eval không cần dữ liệu thật: shop sinh ngẫu nhiên (tests/helpers.py), rulebook tự viết,
câu hỏi có đáp án tính độc lập bằng pandas, và kịch bản cho LLM giả (tests/fake_llm.py).

Kịch bản LLM giả:
- planner trả `correct_plan` của shop, xếp trạng thái trả đúng nhãn;
- bước hiểu câu hỏi trả JSON đúng (kiểu đáp án, kỳ, capability);
- cơ chế `helper` viết code dm thật (kiểm hàm nghiệp vụ + ánh xạ so với đáp án tính độc lập);
- cơ chế `pandas` / `sql` trả đáp án viết sẵn (đúng, hoặc cố ý sai để kiểm verifier).

Mỗi câu có `expect` (correct / wrong / abstain): kết quả pipeline PHẢI cho với kịch bản đó. Đổi logic verifier,
reconcile, ép kiểu, ghi file nộp mà làm lệch `expect` là lỗi logic. Câu `wrong` là giới hạn đã biết (hai cơ chế cùng
sai một kiểu thì cross-check không bắt được), giữ lại để đo, không phải mục tiêu.

    uv run python tests/synth_split.py /tmp/synth     # ghi split + fake_llm.json ra thư mục
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from helpers import DEFAULT_STATUSES, correct_plan, make_orders, write_shop

from datamate.service import DEFAULT_RULEBOOK

AS_OF = "2026-04-15"

# Rulebook tự viết (không phải rulebook cuộc thi), dùng chung với chế độ service. Chứa neo của các luật mà hàm dm
# cần (datamate.rules.RULE_ANCHORS), để hàm không bị tắt.
RULEBOOK = DEFAULT_RULEBOOK.read_text(encoding="utf-8")

# Shop B: cùng loại dữ liệu, tên cột / nhãn trạng thái / định dạng khác (loader và planner phải chịu được).
NAMES_B = {"order": "Mã phiếu", "date": "Ngày bán", "time": "Giờ bán", "customer": "Khách hàng",
           "status": "Trạng thái", "product": "Tên SP", "code": "Mã SP", "qty": "Số lượng",
           "price": "Giá", "amount": "Tiền", "unit": "Đơn vị"}  # fmt: skip
STATUSES_B = {"hoàn thành": "Đã giao xong", "huỷ": "Huỷ đơn", "trả hàng": "Khách trả",
              "đang giao": "Đang ship"}  # fmt: skip

# Shop A ghi ngày cuối dữ liệu trong README (như dữ liệu cuộc thi), shop B không ghi (như file người dùng tải lên).
README = {
    "synth_a": "Cửa hàng tạp hoá thử nghiệm A. Dữ liệu có đến hết 31/03/2026.",
    "synth_b": "Cửa hàng tạp hoá thử nghiệm B.",
}

SHOPS = {
    # biz_id: (seed, write_shop kwargs, statuses, marker = cột chỉ shop đó có, để LLM giả nhận ra shop)
    "synth_a": (0, {}, DEFAULT_STATUSES, "Số HĐ"),
    "synth_b": (1, {"names": NAMES_B, "layout": "flat", "fmt": "xlsx"}, STATUSES_B, "Mã phiếu"),
}


# ---------------------------------------------------------------- đáp án tính độc lập (không dùng dm)


def _canon(O: pd.DataFrame, statuses: dict) -> pd.DataFrame:
    inv = {v: k for k, v in statuses.items()}
    return O.assign(lab=O.status.map(inv), day=O.ts.dt.normalize())


def _month(O, m, y=2026):
    return O[(O.ts.dt.month == m) & (O.ts.dt.year == y)]


def _counted(O):
    return O[O.lab.isin(["hoàn thành", "đang giao"])]


def _sold(O, L, months):
    ok = _counted(O[O.ts.dt.month.isin(months)])
    return L[L.order.isin(set(ok.order)) & ~L.adj]


def _rank(s: pd.Series, k: int) -> list:
    df = pd.DataFrame({"v": s.astype(float), "t": s.index.astype(str)})
    return list(df.sort_values(["v", "t"], ascending=[False, True], kind="mergesort").index[:k])


def _pct(a, b) -> float:
    return round(100 * a / b, 2)


def _hp(period: str, body: str, rows: str = "int(p.mask(o.ngay).sum())") -> str:
    return f"o = dm.orders()\np = {period}\nROWS, PERIOD = {rows}, p\n{body}"


def _hl(period: str, body: str) -> str:
    return f"L = dm.sold_lines()\np = {period}\nL = L[p.mask(L.ngay)]\nROWS, PERIOD = len(L), p\n{body}"


def _literal(typ: str, ans) -> str:
    if typ == "table":
        cols = {c: [r[i] for r in ans["rows"]] for i, c in enumerate(ans["columns"])}
        val = f"pd.DataFrame({json.dumps(cols, ensure_ascii=False)})"
    elif typ == "boolean":
        val = repr(bool(ans))
    else:
        val = json.dumps(ans, ensure_ascii=False)
    return f"ROWS = 1\nTYPE = {typ!r}\nANSWER = {val}"


def _m(m: int) -> str:
    return f"dm.month({m}, 2026)"


def _q(**kw) -> dict:
    """Mô tả một câu hỏi: question, type, answer, helper, periods, caps; tuỳ chọn pandas_answer, sql_answer
    (đáp án cơ chế độc lập trả, mặc định đúng; sql_answer=None: code SQL lỗi), expect, category."""
    return kw


def _understand(q: str, typ: str, periods: list[str], caps: list[str]) -> dict:
    return {"rewritten": q, "answer_type": typ, "kind": "data", "periods": periods, "needs": [],
            "capabilities": caps}  # fmt: skip


def shop_questions(biz: str, O: pd.DataFrame, L: pd.DataFrame) -> list[dict]:
    """Câu hỏi của một shop: mỗi câu {question, type, answer, expect, category, helper, pandas, sql, understand}."""
    m1, m2, m3 = (_month(O, m) for m in (1, 2, 3))
    rev = {m: int(_counted(x).goods.sum()) for m, x in ((1, m1), (2, m2), (3, m3))}
    daily2 = _counted(m2).groupby("day").goods.sum()
    best_day = _rank(daily2, 1)[0].strftime("%Y-%m-%d")
    sold_q1 = _sold(O, L, [1, 2, 3])
    qty_q1 = sold_q1.groupby("product").qty.sum()
    amt_q1 = sold_q1.groupby("product").amount.sum()
    cancel1 = _pct(int((m1.lab == "huỷ").sum()), len(m1))
    cancelled2 = int((m2.lab == "huỷ").sum())
    table = {"columns": ["mat_hang", "doanh_thu"],
             "rows": [[k, int(v)] for k, v in sorted(amt_q1.items())]}  # fmt: skip
    Q1 = "dm.quarter(1, 2026)"
    rev_cap, prod_cap = ["order_revenue"], ["order_revenue", "product_identity"]

    qs = [
        _q(question="Doanh thu tháng 2/2026 là bao nhiêu?", type="number", answer=rev[2],
             helper=_hp(_m(2), "TYPE = 'number'\nANSWER = dm.revenue(o, p)"), periods=[_m(2)], caps=rev_cap),
        _q(question="Tháng 3/2026 shop có bao nhiêu đơn?", type="number", answer=len(_counted(m3)),
             helper=_hp(_m(3), "TYPE = 'number'\nANSWER = dm.count_orders(o, p)"), periods=[_m(3)], caps=rev_cap),
        _q(question="Tỉ lệ đơn huỷ tháng 1/2026 là bao nhiêu %?", type="percent", answer=cancel1,
             helper=_hp(_m(1), "TYPE = 'percent'\nANSWER = dm.cancel_rate(o, p)"), periods=[_m(1)], caps=rev_cap),
        _q(question="Ngày nào trong tháng 2/2026 có doanh thu cao nhất?", type="date", answer=best_day,
             helper=_hp(_m(2), "TYPE = 'date'\nANSWER = dm.top(dm.daily_revenue(o, p), 1)[0].strftime('%Y-%m-%d')"),
             periods=[_m(2)], caps=rev_cap),
        _q(question="Mặt hàng nào bán được nhiều nhất theo số lượng trong quý 1/2026?", type="string",
             answer=_rank(qty_q1, 1)[0],
             helper=_hl(Q1, "TYPE = 'string'\nANSWER = dm.top(L.groupby('ten_hang')['sl_co_ban'].sum(), 1)[0]"),
             periods=[Q1], caps=prod_cap),
        _q(question="Top 3 mặt hàng bán chạy nhất theo số lượng quý 1/2026 là gì?", type="list",
             answer=_rank(qty_q1, 3),
             helper=_hl(Q1, "TYPE = 'list'\nANSWER = dm.top(L.groupby('ten_hang')['sl_co_ban'].sum(), 3)"),
             periods=[Q1], caps=prod_cap),
        _q(question="Doanh thu từng mặt hàng trong quý 1/2026?", type="table", answer=table,
             helper=_hl(Q1, "s = L.groupby('ten_hang')['thanh_tien'].sum()\nTYPE = 'table'\n"
                            "ANSWER = pd.DataFrame({'mat_hang': s.index, 'doanh_thu': s.values.astype(int)})"),
             periods=[Q1], caps=prod_cap),
        _q(question="Doanh thu tháng 3/2026 có cao hơn tháng 2/2026 không?", type="boolean",
             answer=rev[3] > rev[2],
             helper=_hp(_m(3), "TYPE = 'boolean'\nANSWER = bool(dm.revenue(o, p) > dm.revenue(o, dm.month(2, 2026)))"),
             periods=[_m(3), _m(2)], caps=rev_cap),
        # Kỳ trước dữ liệu: regex của bước hiểu câu hỏi bắt, không cần code.
        _q(question="Doanh thu tháng 6/2025 là bao nhiêu?", type="unanswerable", answer=None,
             category="period_out"),
        # Tiêu chí cảm tính, không có trong dữ liệu / rulebook: precheck bắt.
        _q(question="Nhân viên nào lễ phép nhất?", type="unanswerable", answer=None, category="subjective"),
        # Cơ chế pandas sai (quên bỏ đơn huỷ), SQL đúng: 2/3 cơ chế đồng ý → trả lời đúng.
        _q(question="Tổng doanh thu tháng 1/2026 bao nhiêu?", type="number", answer=rev[1],
             helper=_hp(_m(1), "TYPE = 'number'\nANSWER = dm.revenue(o, p)"), periods=[_m(1)], caps=rev_cap,
             pandas_answer=int(m1.goods.sum()), category="majority"),
        # Ba cơ chế ra ba đáp án khác nhau → abstain (bỏ câu thay vì đoán).
        _q(question="Có bao nhiêu đơn bị huỷ trong tháng 2/2026?", type="number", answer=cancelled2,
             helper=_hp(_m(2), "TYPE = 'number'\nANSWER = dm.count_cancelled(o, p)"), periods=[_m(2)], caps=rev_cap,
             pandas_answer=cancelled2 + 1, sql_answer=cancelled2 + 2, expect="abstain", category="disagree"),
        # Hai cơ chế cùng sai một kiểu (đếm cả đơn huỷ) → khớp nhau, verifier để lọt. Giới hạn đã biết.
        _q(question="Số đơn của tháng 1/2026 là bao nhiêu?", type="number", answer=len(_counted(m1)),
             helper=_hp(_m(1), "TYPE = 'number'\nANSWER = int(p.mask(o.ngay).sum())"), periods=[_m(1)],
             caps=rev_cap, pandas_answer=len(m1), expect="wrong", category="correlated_error"),
        # Code helper lỗi mọi lần sửa, chỉ cơ chế độc lập ra đáp án → không đủ chắc, abstain.
        _q(question="Doanh thu tháng 3/2026 đạt bao nhiêu?", type="number", answer=rev[3],
             helper="o = dm.orders()\nTYPE = 'number'\nANSWER = dm.revenue(o, thang_ba)", periods=[_m(3)],
             caps=rev_cap, sql_answer=None, expect="abstain", category="codegen_error"),
        # Mọi cơ chế lỗi sau mọi lần sửa → không có đáp án hợp lệ → abstain (nhóm "codegen lỗi" của báo cáo lỗi).
        _q(question="Doanh thu tháng 2/2026 của shop là bao nhiêu tiền?", type="number", answer=rev[2],
           helper="TYPE = 'number'\nANSWER = dm.revenue(o, p)", periods=[_m(2)], caps=rev_cap,
           pandas_code="ANSWER = tables['KhongCo']", sql_answer=None, expect="abstain", category="all_fail"),
        # Hai cơ chế cùng trả tỉ lệ dạng số thập phân, type number thay vì percent → sai type (nhóm "hiểu sai câu hỏi").
        _q(question="Tháng 1/2026 tỉ lệ huỷ đơn là bao nhiêu?", type="percent", answer=cancel1,
           helper=_hp(_m(1), "TYPE = 'number'\nANSWER = round(dm.cancel_rate(o, p) / 100, 4)"), periods=[_m(1)],
           caps=rev_cap, pandas_code=_literal("number", round(cancel1 / 100, 4)), expect="wrong",
           category="wrong_type"),
    ]  # fmt: skip
    out = []
    for i, q in enumerate(qs, 1):
        typ, ans = q["type"], q["answer"]
        item = {
            "id": f"{biz}_q{i:02d}",
            "business_id": biz,
            "question": q["question"],
            "type": typ,
            "answer": ans,
            "expect": q.get("expect", "correct"),
            "category": q.get("category", "basic"),
        }
        if typ != "unanswerable":
            item["understand"] = _understand(q["question"], typ, q["periods"], q["caps"])
            item["helper"] = q["helper"]
            pa = q.get("pandas_answer", ans)
            item["pandas"] = q.get("pandas_code") or _literal(typ, pa)
            sa = q.get("sql_answer", ans)
            item["sql"] = _literal(typ, sa) if sa is not None else "ANSWER = khong_co_bien"
        out.append(item)
    return out


def build_split(root: Path) -> dict:
    """Ghi split synthetic vào `root` (questions.jsonl, answers.jsonl, rulebook.md, calendar.csv, data/) và trả về
    kịch bản LLM giả (cũng ghi ra `root/fake_llm.json`)."""
    root = Path(root)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "rulebook.md").write_text(RULEBOOK, encoding="utf-8")
    # Lịch tối thiểu (dương lịch); câu hỏi synthetic không hỏi Tết.
    days = pd.date_range("2025-01-01", "2026-12-31")
    pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "lunar_year": days.year, "lunar_month": days.month,
                  "lunar_day": days.day, "lunar_leap": False, "holiday": ""}).to_csv(
        root / "calendar.csv", index=False)  # fmt: skip
    script = {"plans": [], "statuses": {}, "questions": {}}
    items = []
    for biz, (seed, kw, statuses, marker) in SHOPS.items():
        O, L, _ = make_orders(seed=seed, statuses=statuses)
        n = write_shop(root / "data" / biz, O, L, readme=README[biz], **kw)
        script["plans"].append(
            {
                "marker": marker,
                "plan": correct_plan(n, statuses, layout=kw.get("layout", "separate")),
            }
        )
        script["statuses"].update({v: k for k, v in statuses.items()})
        for it in shop_questions(biz, _canon(O, statuses), L):
            items.append(it)
            if "helper" in it:
                script["questions"][f"{marker}||{it['question']}"] = {
                    k: it[k] for k in ("understand", "helper", "pandas", "sql")
                }  # fmt: skip
    with (root / "questions.jsonl").open("w", encoding="utf-8") as fq, \
         (root / "answers.jsonl").open("w", encoding="utf-8") as fa:  # fmt: skip
        for it in items:
            fq.write(json.dumps({"id": it["id"], "business_id": it["business_id"], "question": it["question"],
                                 "as_of_date": AS_OF}, ensure_ascii=False) + "\n")  # fmt: skip
            gold = {"id": it["id"], "type": it["type"], "alias": []}
            if it["type"] != "unanswerable":
                gold["answer"] = it["answer"]
            fa.write(json.dumps(gold, ensure_ascii=False) + "\n")
    script["expect"] = {it["id"]: it["expect"] for it in items}
    script["category"] = {it["id"]: it["category"] for it in items}
    (root / "fake_llm.json").write_text(
        json.dumps(script, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return script


if __name__ == "__main__":
    s = build_split(Path(sys.argv[1]))
    print(f"{len(s['expect'])} câu → {sys.argv[1]}")
