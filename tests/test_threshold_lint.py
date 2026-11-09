"""Lint ngưỡng (§2 rulebook): 'từ N trở lên' gồm N, 'quá / trên / hơn N' không gồm N. Câu hỏi và code là giả định."""

import pytest

from datamate.agent.lint import question_thresholds, threshold_issues


@pytest.mark.parametrize(
    "q,want",
    [
        ("Tháng 3 có mấy khách mua từ 5 đơn trở lên?", [(5, ">=")]),
        ("so khach mua it nhat 7 don thang nay", [(7, ">=")]),
        ("Có mấy ngày bán được từ 12.000.000 đồng trở lên?", [(12_000_000, ">=")]),
        ("đơn có tiền hàng từ 250k trở lên", [(250_000, ">=")]),
        ("ngày nào doanh thu từ 15 triệu trở lên", [(15_000_000, ">=")]),
        ("mặt hàng có mặt trong ≥ 40 đơn", [(40, ">=")]),
        ("Nợ của các đơn đã đặt quá 10 ngày", [(10, ">")]),
        ("khách mua dưới 3 đơn", [(3, "<")]),
        ("đơn tối đa 2 món", [(2, "<=")]),
        # Không phải ngưỡng: khoảng / kỳ thời gian, ngày tháng, "mỗi".
        ("Doanh thu có tăng liên tục qua 3 tháng 4/2026, 5/2026, 6/2026 không?", []),
        ("7 ngày kể từ ngày tăng giá bán nhiều hay ít hơn 7 ngày liền trước?", []),
        ("Trong 3 tháng 6/2025 qua 7/2025 rồi 8/2025 doanh thu có tăng không?", []),
        ("Tuần trước doanh thu bình quân trên 1 đơn là bao nhiêu?", []),
    ],
)
def test_question_thresholds(q, want):
    assert [(n, op) for n, op in question_thresholds(q)] == [(float(n), op) for n, op in want]


Q = "Tháng 3 có mấy khách mua từ 5 đơn trở lên?"


@pytest.mark.parametrize(
    "code",
    [
        "n = (s >= 5).sum()",
        "n = (s > 4).sum()",  # số nguyên: > 4 ≡ >= 5
        "n = (5 <= s).sum()",
        "rows = sql('SELECT COUNT(*) FROM t WHERE so_don >= 5')",
        "x = s[s >= MIN]",  # ngưỡng là biến: không kết luận
    ],
)
def test_threshold_code_ok(code):
    assert threshold_issues(code, Q) == []


@pytest.mark.parametrize(
    "code", ["n = (s > 5).sum()", "n = (5 < s).sum()", "sql('SELECT * FROM t WHERE so_don > 5')"]
)
def test_threshold_code_wrong_operator(code):
    issues = threshold_issues(code, Q)
    assert len(issues) == 1 and issues[0].startswith("ngưỡng:")


def test_strict_threshold():
    q = "Tổng nợ của các đơn đã đặt quá 10 ngày"
    assert threshold_issues("m = age > 10", q) == []
    assert threshold_issues("m = age >= 11", q) == []
    assert threshold_issues("m = age >= 10", q)


class _ScriptedLLM:
    """LLM giả: trả lần lượt các câu trả lời đã viết sẵn, ghi lại tin nhắn nhận được."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages, **kw):
        self.seen.append(messages)
        return self.replies.pop(0), {}


def test_solver_repairs_wrong_threshold_and_verifier_flags_unrepaired():
    import pandas as pd

    from datamate.agent.solver import ShopContext, SolverConfig, _run_candidate
    from datamate.agent.verifier import VerifierConfig, candidate_checks

    t = {"Khach": pd.DataFrame({"so_don": [1, 2, 2, 3, 5]})}
    ctx = ShopContext("s", t, "", "", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31"),
                      "calendar.csv", "")  # fmt: skip
    q = {"id": "q", "question": "Có mấy khách mua từ 2 đơn trở lên?", "as_of_date": "2026-03-31"}
    wrong = "```python\nTYPE = 'number'\nANSWER = int((tables['Khach'].so_don > 2).sum())\n```"
    right = "```python\nTYPE = 'number'\nANSWER = int((tables['Khach'].so_don >= 2).sum())\n```"
    llm = _ScriptedLLM([wrong, right])
    c = _run_candidate(llm, ctx, q, 0, SolverConfig(understand=False))
    assert c.answer == 4 and c.lint == []
    assert "rulebook §2" in llm.seen[1][-1]["content"]  # thông báo sửa gửi lại cho LLM
    # LLM không sửa: lời giải còn lỗi lint, verifier đánh trượt (không được trả lời bằng lời giải này).
    c2 = _run_candidate(_ScriptedLLM([wrong] * 3), ctx, q, 0, SolverConfig(understand=False))
    assert c2.answer == 2 and c2.lint
    assert "threshold" in {x.name for x in candidate_checks(q, c2, ctx, VerifierConfig())}
