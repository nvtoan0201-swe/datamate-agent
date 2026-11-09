"""Bước hiểu câu hỏi (agent/understand.py): phần tất định và phần kiểm tra lại kết quả LLM. Không cần LLM."""

import json

import pandas as pd

from datamate.agent.prompts import normalize_time
from datamate.agent.solver import Candidate, ShopContext, SolverConfig, decide
from datamate.agent.understand import (
    _resolve_column,
    check_rewrite,
    explicit_periods,
    period_unanswerable,
    understand,
)

START, END = pd.Timestamp("2025-02-01"), pd.Timestamp("2026-09-14")


def test_quarter_not_read_as_month():
    assert "tháng" not in normalize_time("Quý 4/2025 cửa hàng có bao nhiêu đơn")
    assert "tháng" not in normalize_time("e so don hang quy 4/2025 la bao nhiu")
    assert normalize_time("4-2025 doanh thu") == "tháng 4/2025 doanh thu"


def test_explicit_periods():
    labels = [p[0] for p in explicit_periods("dthu thang 1 nam 2025 bnhiu")]
    assert labels == ["tháng 1/2025"]
    assert [p[0] for p in explicit_periods("Quý 4/2025 có bao nhiêu đơn")] == ["quý 4/2025"]
    labels = [
        p[0]
        for p in explicit_periods(
            "Tháng 2/2025 doanh thu thay đổi bao nhiêu % so với tháng liền trước?"
        )
    ]
    assert labels == ["tháng 2/2025", "tháng 1/2025 (tháng liền trước)"]


def test_period_unanswerable():
    today = "2026-09-14"
    assert period_unanswerable("tổng dthu tháng 1 năm 2025 là bao nhiu", START, END, today)
    assert period_unanswerable(
        "Tháng 2/2025 doanh thu thay đổi bao nhiêu % so với tháng liền trước?", START, END, today
    )
    assert period_unanswerable("doanh thu tháng 2/2025", START, END, today) is None
    # Quý chỉ một phần trong dữ liệu: tính trên phần có dữ liệu.
    assert period_unanswerable("số đơn quý 1/2025", START, END, today) is None
    # Khoảng sau hôm nay (hạn dùng) không phải ngoài dữ liệu.
    assert period_unanswerable("hàng hết hạn tháng 12/2026", START, END, today) is None


def test_check_rewrite():
    q = "dthu T4/2026 cua Sting bnhiu"
    assert check_rewrite(q, "Doanh thu tháng 4/2026 của Sting là bao nhiêu?") is None
    assert check_rewrite(q, "Doanh thu tháng 5/2026 của Sting là bao nhiêu?")  # đổi số
    assert check_rewrite("doanh thu tháng này", "Doanh thu tháng 9/2026")  # tự quy ra ngày
    assert check_rewrite(q, "")


def test_resolve_column():
    t = {
        "DuLieuXuat.DonNhapHang_ChiTiet": pd.DataFrame(columns=["Số lượng nhập", "Mã lô"]),
        "DanhSachHoaDon": pd.DataFrame(columns=["Tổng tiền hàng"]),
    }
    assert (
        _resolve_column("DonNhapHang_ChiTiet.Số lượng nhập", t)
        == "DuLieuXuat.DonNhapHang_ChiTiet.Số lượng nhập"
    )
    assert (
        _resolve_column("DanhSachHoaDonChiTiet.Tổng tiền hàng", t)
        == "DanhSachHoaDon.Tổng tiền hàng"
    )
    assert _resolve_column("KhachHang.Nhóm khách", t) is None


class FakeLLM:
    def __init__(self, reply: dict):
        self.reply = reply

    def chat(self, messages, max_tokens=None, json_mode=False):
        return json.dumps(self.reply, ensure_ascii=False), {}


def _ctx():
    t = {"HoaDon": pd.DataFrame({"Thời gian": [START, END], "Tổng tiền": [1, 2]})}
    return ShopContext("b", t, "", "", START, END, "", "")


def test_understand_validates_llm_output():
    q = {"question": "dthu thang 3/2026 qua dien thoai bnhiu", "as_of_date": "2026-09-14"}
    reply = {
        "rewritten": "Doanh thu tháng 3/2026 qua điện thoại là bao nhiêu?",
        "answer_type": "number",
        "periods": ["dm.month(3, 2026)", "dm.month(7, 2024)", "os.system('x')"],
        "needs": [
            {"info": "ngày bán", "column": "HoaDon.Thời gian"},
            {"info": "kênh bán", "column": None},
            {"info": "tiền", "column": "HoaDon.Cột bịa"},
        ],
        "kind": "data",
    }
    u = understand(FakeLLM(reply), _ctx(), q)
    assert u.rewritten.startswith("Doanh thu")
    assert u.answer_type == "number"
    assert u.periods == [
        "dm.month(3, 2026)"
    ]  # 7/2024 không có trong câu, os.system không phải hàm kỳ
    assert u.missing == ["kênh bán"]
    assert u.period_out is None and u.period_out_llm is None
    assert "kênh bán" in u.unanswerable_signal


def test_understand_without_llm_still_checks_period():
    q = {"question": "tổng dthu tháng 1 năm 2025", "as_of_date": "2026-09-14"}
    u = understand(None, _ctx(), q)
    assert u.period_out and u.rewritten is None


def _c(v, typ, ans, key=None):
    c = Candidate(v, typ=typ, answer=ans, code="o = dm.orders()\nANSWER = 1")
    c.key = key or ("UNANSWERABLE" if typ == "unanswerable" else f"{typ}:{ans}")
    return c


def test_decide_with_signal():
    q = {
        "id": "q1",
        "question": "doanh thu qua điện thoại tháng 3/2026",
        "as_of_date": "2026-09-14",
    }
    cfg = SolverConfig(k=3)
    sig = "shop không ghi: kênh bán"
    # code không ra đáp án + tín hiệu → unanswerable
    bad = Candidate(0, error="KeyError")
    row, *_ = decide(q, [bad, bad, bad], cfg, _ctx(), signal=sig)
    assert row["type"] == "unanswerable"
    # code báo unanswerable (mặc định chỉ abstain) + tín hiệu → unanswerable
    row, *_ = decide(q, [_c(i, "unanswerable", None) for i in range(3)], cfg, _ctx(), signal=sig)
    assert row["type"] == "unanswerable"
    # code ra đáp án + tín hiệu → abstain hoặc trả lời (mặc định tin code)
    cands = [_c(i, "number", 5) for i in range(3)]
    assert (
        decide(q, cands, SolverConfig(k=3, signal_vs_answer="abstain"), _ctx(), signal=sig)[0][
            "type"
        ]
        == "abstain"
    )
    cfg2 = SolverConfig(k=3, signal_vs_answer="answer")
    assert decide(q, cands, cfg2, _ctx(), signal=sig)[0] == {
        "id": "q1",
        "type": "number",
        "answer": 5,
    }
    # không tín hiệu: như cũ
    assert decide(q, cands, cfg, _ctx())[0]["type"] == "number"


def test_derivable_from_time():
    from datamate.agent.understand import derivable_from_time

    with_hours = {
        "HD": pd.DataFrame({"Thời gian": pd.to_datetime(["2025-05-01 08:30", "2025-05-01 15:00"])})
    }
    dates_only = {"HD": pd.DataFrame({"Ngày": pd.to_datetime(["2025-05-01", "2025-05-02"])})}
    assert derivable_from_time("tháng", dates_only)
    assert derivable_from_time("ca bán của đơn", with_hours)
    assert not derivable_from_time("ca bán của đơn", dates_only)
    assert not derivable_from_time("kênh bán", with_hours)
