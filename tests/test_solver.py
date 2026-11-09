"""Bỏ phiếu theo nhóm, sanity, unanswerable chắc / nghi ngờ (generalization-plan §5). Không cần LLM."""

import pandas as pd

from datamate.agent.precheck import classify
from datamate.agent.solver import Candidate, ShopContext, SolverConfig, decide, need_for, sanity

Q = {
    "id": "q1",
    "question": "tỉ lệ huỷ đơn tháng 6/2025 là bao nhiêu %",
    "as_of_date": "2026-06-24",
}


def _ctx(mapping_ok=True):
    t = {"Ban": pd.DataFrame({"Tên hàng": ["Sting dâu 330ml", "Cà phê sữa (M)"]})}
    return ShopContext("b", t, "", "", pd.Timestamp("2025-01-01"), pd.Timestamp("2026-06-24"), "", "",
                       mapping_ok=mapping_ok)  # fmt: skip


def _c(v, typ, ans, code="o = dm.orders()\nANSWER = 1", key=None, reason=None):
    c = Candidate(v, typ=typ, answer=ans, code=code, reason=reason)
    c.key = key or ("UNANSWERABLE" if typ == "unanswerable" else f"{typ}:{ans}")
    return c


def test_need_scales_with_k_and_group():
    cfg = SolverConfig(k=3)
    assert need_for("helper", cfg, _ctx()) == 1
    assert need_for("free", cfg, _ctx()) == 3
    assert need_for("helper", cfg, _ctx(mapping_ok=False)) == 3
    assert need_for("free", SolverConfig(k=5), _ctx()) == 5
    assert need_for("helper", SolverConfig(k=1), _ctx()) == 1


def test_sanity_candidates_excluded_from_vote():
    # Hai biến thể so nhầm chuỗi trạng thái cùng ra 0 %, một biến thể đúng 2.08.
    cands = [_c(0, "percent", 2.08), _c(1, "percent", 0), _c(2, "percent", 0)]
    row, _, path, flags = decide(Q, cands, SolverConfig(k=3), _ctx())
    assert row == {"id": "q1", "type": "percent", "answer": 2.08}
    assert path == "helper" and flags
    row, *_ = decide(Q, cands, SolverConfig(k=3, use_sanity=False), _ctx())
    assert row["answer"] == 0


def test_free_path_needs_all_variants():
    code = "df = tables['Ban']\nANSWER = 3"
    q = {**Q, "question": "bao nhiêu lượt xe"}
    cands = [_c(0, "number", 3, code), _c(1, "number", 3, code), _c(2, "number", 4, code)]
    row, _, path, _ = decide(q, cands, SolverConfig(k=3), _ctx())
    assert path == "free" and row["type"] == "abstain"
    cands[2] = _c(2, "number", 3, code)
    row, *_ = decide(q, cands, SolverConfig(k=3), _ctx())
    assert row["answer"] == 3


def test_unanswerable_from_code_needs_all_but_period_does_not():
    no_col = [_c(i, "unanswerable", None, reason="shop không ghi giới tính") for i in range(2)]
    no_col.append(_c(2, "number", 5, key="number:5"))
    row, *_ = decide(Q, no_col, SolverConfig(k=3, code_unanswerable="all_agree"), _ctx())
    assert row["type"] == "abstain"
    all3 = [_c(i, "unanswerable", None, reason="shop không ghi giới tính") for i in range(3)]
    row, *_ = decide(Q, all3, SolverConfig(k=3, code_unanswerable="all_agree"), _ctx())
    assert row["type"] == "unanswerable"
    row, *_ = decide(Q, all3, SolverConfig(k=3), _ctx())
    assert row["type"] == "abstain"
    period = [
        _c(i, "unanswerable", None, reason="kỳ Period(...) nằm ngoài khoảng dữ liệu x")
        for i in range(2)
    ]
    period.append(_c(2, "number", 5, key="number:5"))
    row, *_ = decide(Q, period, SolverConfig(k=3), _ctx())
    assert row["type"] == "unanswerable"


def test_sanity_flags():
    ctx = _ctx()
    assert sanity({**Q, "question": "tháng 8/2025 quán bán đc bn ly size m"}, "number", 0, ctx)
    assert (
        sanity(
            {**Q, "question": "doanh thu tháng 5 giảm bao nhiêu so với tháng 4"}, "number", -5, ctx
        )
        == []
    )
    assert sanity({**Q, "question": "doanh thu tháng 5"}, "number", -5, ctx)
    assert sanity(Q, "percent", 140, ctx)
    assert sanity({**Q, "question": "doanh thu tăng bao nhiêu %"}, "percent", 140, ctx) == []
    assert sanity(Q, "string", "Sting dau 330ml", ctx) == []  # khác dấu vẫn là tên có trong dữ liệu
    assert sanity(Q, "string", "Pepsi lon", ctx)
    assert sanity({**Q, "question": "ngày nào doanh thu cao nhất"}, "date", "2027-01-01", ctx)


def test_near_miss_subjective_abstains():
    rb = "| Nhà cung cấp chính | nhà cung cấp có nhiều phiếu nhập nhất |"
    # "khen nhiều": từ lượng không đi cùng thứ sổ sách ghi được, rulebook không có khái niệm "… nhất" chung tiêu chí.
    assert (
        classify("nhà cung cấp được khen nhiều nhất tháng 11 là ai", rb, "", [])[0]
        == "unanswerable"
    )
    rb2 = rb + "\n| Nhà cung cấp uy tín nhất | nhà cung cấp có ít phiếu huỷ nhất |"
    assert classify("nhà cung cấp uy tín cao nhất là ai", rb2, "", [])[0] == "abstain"
    assert classify("nhân viên nào lễ phép nhất", rb, "", [])[0] == "unanswerable"
    assert classify("doanh thu tháng 3", rb, "", []) is None
