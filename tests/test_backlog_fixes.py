"""Test cho các kiểu lỗi đã sửa (fill-down trong đơn, mốc hết ngày, SQL ngày, resolver tên hàng,
dòng hàng nhiều biến thể, dòng điều chỉnh dạng mẫu, kỳ tương lai cho câu hỏi về hạn). Dữ liệu tự sinh."""

from types import SimpleNamespace

import pandas as pd

from datamate.agent.lint import temporal_issues, time_columns
from datamate.agent.sandbox import _sql_factory
from datamate.agent.semantic_reconcile import _template_values
from datamate.agent.verifier import VerifierConfig, candidate_checks
from datamate.biz import Biz
from datamate.loaders.shop import load_shop
from datamate.text import key


def test_fill_down_stays_inside_order_and_skips_line_columns(tmp_path):
    rows = [
        # STT, Hàng, IMEI, Bớt, Ghi chú
        (1, "Máy A", "111", 10000, None),
        (None, "Ốp", None, None, None),
        (2, "Máy B", "222", None, "khách trả"),
        (None, "Sạc", None, None, None),
        (None, "Máy C", "333", None, None),
        (3, "Cáp", None, None, None),
        (None, "Kính", None, None, None),
    ] * 3
    df = pd.DataFrame(rows, columns=["STT", "Hàng", "IMEI", "Bớt", "Ghi chú"])
    df.insert(0, "Ngày", "2026-01-01")
    d = tmp_path / "shop"
    d.mkdir()
    df.to_excel(d / "SoBan.xlsx", index=False)
    out = load_shop(d).tables["SoBan"].df
    first = out.iloc[:7]
    assert first["STT"].tolist() == [1, 1, 2, 2, 2, 3, 3]
    # Bớt của đơn 1 không tràn sang đơn 2, 3 (ô đầu đơn trống = không có khoản đó).
    assert first["Bớt"].isna().tolist() == [False, False, True, True, True, True, True]
    # Ghi chú mức đơn điền trong đơn 2, không sang đơn 3.
    assert first["Ghi chú"].tolist()[2:5] == ["khách trả"] * 3
    assert first["Ghi chú"].iloc[5:].isna().all()
    # IMEI mức dòng: phụ kiện giữ trống.
    assert pd.isna(first["IMEI"].iloc[1]) and pd.isna(first["IMEI"].iloc[3])


def test_temporal_lint_flags_day_end_comparisons():
    P = pd.DataFrame({"Ngày TT": pd.to_datetime(["2025-07-02 09:30", "2025-07-02 00:00"])})
    O = pd.DataFrame({"Ngày": pd.to_datetime(["2025-07-02", "2025-07-03"])})
    tc = time_columns({"P": P, "O": O})
    assert tc == {"Ngày TT"}
    bad = "d = pd.Timestamp('2025-07-02')\npaid = P[P['Ngày TT'] <= d]"
    assert temporal_issues(bad, tc)
    assert temporal_issues("x = O[O['Ngày'] <= d]", tc) == []  # cột chỉ có ngày
    ok = [
        "x = P[P['Ngày TT'] < d + pd.Timedelta(days=1)]",
        "x = P[P['Ngày TT'].dt.normalize() <= d]",
        "e = d + pd.Timedelta(days=1)\nx = P[P['Ngày TT'] <= e]",
        "x = P[P['Ngày TT'] >= d]",
        'x = sql(\'SELECT * FROM "P" WHERE date("Ngày TT") <= ?\', [s])',
    ]
    for code in ok:
        assert temporal_issues(code, tc) == [], code
    assert temporal_issues('x = sql(\'SELECT * FROM "P" WHERE "Ngày TT" <= ?\', [s])', tc)


def test_sql_date_only_columns_include_last_day():
    O = pd.DataFrame(
        {"Ngày": pd.to_datetime(["2025-12-01", "2025-12-31", "2026-01-01"]), "Tiền": [1, 2, 4]}
    )
    sql = _sql_factory({"O": O})
    q = 'SELECT SUM("Tiền") AS s FROM "O" WHERE "Ngày" >= ? AND "Ngày" <= ?'
    assert sql(q, ("2025-12-01", "2025-12-31"))["s"].iloc[0] == 3
    p_start, p_end = pd.Timestamp("2025-12-01"), pd.Timestamp("2025-12-31")
    assert sql(q, (p_start, p_end))["s"].iloc[0] == 3


def test_match_product_line_takes_all_variants():
    b = Biz("2026-06-30")
    names = pd.Series(
        [
            "Samsung Galaxy A56 5G 128GB Đen",
            "Samsung Galaxy A56 5G 256GB Xám",
            "Samsung Galaxy A36 5G 128GB Tím",
            "Samsung Galaxy A56 5G 256GB Xám",
        ]
    )
    assert b.match(names, "Samsung Galaxy A56 5G").tolist() == [True, True, False, True]
    # Tên trùng nguyên văn: chỉ đúng tên đó.
    assert b.match(names, "Samsung Galaxy A36 5G 128GB Tím").tolist() == [False, False, True, False]
    # Token kế tiếp không có số ('Pro') không phải biến thể của 'iPhone 16'.
    ip = pd.Series(["iPhone 16 128GB Đen", "iPhone 16 Pro 128GB Titan"])
    assert b.match(ip, "iPhone 16").tolist() == [True, False]


def test_product_resolver_expands_abbreviations():
    P = pd.DataFrame(
        {
            "Ghi tắt": ["A56 128 đen", "CL 9D"],
            "Tên hàng": ["Samsung Galaxy A56 5G 128GB Đen", "Kính cường lực 9D"],
            "Loại": ["máy mới", "phụ kiện"],
        }
    )
    b = Biz("2026-06-30", tables={"Danh": P})
    L = pd.DataFrame({"ten_hang": ["A56 128 đen", "Kính cường lực 9D", "CL 9D", "Hàng lạ"]})
    L["ma_hang"] = None
    mp = {
        "products": {"table": "Danh", "code": "Ghi tắt", "name": "Tên hàng", "group": "Loại"},
        "sales_lines": {"product": "Hàng"},
    }
    b._fill_product_identity(L, mp)
    assert L["ten_hang"].tolist() == [
        "Samsung Galaxy A56 5G 128GB Đen",
        "Kính cường lực 9D",
        "Kính cường lực 9D",
        "Hàng lạ",
    ]
    assert L["ten_hang_goc"].tolist()[0] == "A56 128 đen"
    assert L["nhom_hang"].tolist()[:3] == ["máy mới", "phụ kiện", "phụ kiện"]
    assert L["ma_hang"].tolist()[:3] == ["A56 128 đen", "CL 9D", "CL 9D"]


def test_adjustment_template_values():
    vals = ["Voucher", "Dùng 200 điểm", "Dùng 300 điểm", "Phí ship", "Sữa bột số 2"]
    present = {key(v): v for v in vals}
    assert _template_values("Dùng <n> điểm", present) == ["Dùng 200 điểm", "Dùng 300 điểm"]
    assert _template_values("Voucher", present) == []
    assert _template_values("<x>", present) == []


def test_period_future_allowed_for_due_questions():
    ctx = SimpleNamespace(snapshot_cols=[], data_end=None, data_start=None, total_revenue=None)
    c = SimpleNamespace(typ="number", answer=21, rows=21, period=("2025-11-01", "2025-11-30"),
                        code="", lint=[])  # fmt: skip
    due = {
        "question": "Có bao nhiêu đơn có ngày phải trả rơi vào tháng 11/2025?",
        "as_of_date": "2025-10-02",
    }
    sales = {"question": "Doanh thu tháng 11/2025 bao nhiêu?", "as_of_date": "2025-10-02"}
    names = lambda q: {x.name for x in candidate_checks(q, c, ctx, VerifierConfig())}
    assert "period_future" not in names(due)
    assert "period_future" in names(sales)


def test_lint_marks_candidate():
    ctx = SimpleNamespace(snapshot_cols=[], data_end=None, data_start=None, total_revenue=None)
    c = SimpleNamespace(
        typ="number", answer=3, rows=3, period=None, code="", lint=["cột 'x' có giờ"]
    )
    q = {"question": "Đến hết ngày 02/07/2025 còn bao nhiêu khách nợ?", "as_of_date": "2025-07-02"}
    assert "temporal_boundary" in {x.name for x in candidate_checks(q, c, ctx, VerifierConfig())}


def test_period_prev_is_calendar_aware():
    b = Biz("2026-08-21")
    pr = b.month(4, 2026).prev()
    assert (str(pr.start.date()), str(pr.end.date())) == ("2026-03-01", "2026-03-31")
    pr = b.quarter(2, 2026).prev()
    assert (str(pr.start.date()), str(pr.end.date())) == ("2026-01-01", "2026-03-31")
    pr = b.between("2026-03-01", "2026-03-31").prev()
    assert (str(pr.start.date()), str(pr.end.date())) == ("2026-02-01", "2026-02-28")
    pr = b.between("2026-06-01", "2026-06-18").prev()  # "tháng này" → cùng số ngày tháng trước
    assert (str(pr.start.date()), str(pr.end.date())) == ("2026-05-01", "2026-05-18")
    pr = b.last_week().prev()
    assert pr.days == 7 and pr.end == b.last_week().start - pd.Timedelta(days=1)


def test_expression_conversion_factors():
    from datamate.biz import _expr_factors

    C = pd.DataFrame(
        {
            "Tên hàng": ["Tã size L", "Khăn ướt", "Bình sữa"],
            "Quy đổi": [1.0, 1.0, None],
            "Quy đổi (đơn vị ghi kèm)": ["thùng = 4 bịch", "lốc = 4 gói; 1 thùng = 24 gói", None],
        }
    )
    f = _expr_factors(C, "Tên hàng", "Quy đổi")
    assert f[(key("Tã size L"), "thung")] == 4
    assert f[(key("Khăn ướt"), "loc")] == 4 and f[(key("Khăn ướt"), "thung")] == 24
    assert not any(p == key("Bình sữa") for p, _ in f)


def test_product_group_question_without_group_column_is_unanswerable():
    from datamate.agent.verifier import product_group_question, verify

    assert product_group_question("Tháng 9/2025 nhóm trà sữa bán được bao nhiêu ly?")
    assert not product_group_question("Khách nhóm thân thiết tháng 12 mua bao nhiêu?")
    assert not product_group_question("Nợ chia theo tuổi nợ 0–7, 8–15 ngày, mỗi nhóm bao nhiêu?")
    absent = SimpleNamespace(status="absent", votes={})
    graph = SimpleNamespace(entity=lambda k: SimpleNamespace(group=absent))
    ctx = SimpleNamespace(graph=graph)
    q = {
        "id": "x",
        "question": "Tháng 9/2025 nhóm trà sữa bán được bao nhiêu ly?",
        "as_of_date": "2026-01-01",
    }
    c = SimpleNamespace(key="number:793", typ="number", answer=793, mechanism="pandas", code="")
    assert verify(q, [c], ctx, VerifierConfig()).row["type"] == "unanswerable"


def test_absent_capability_is_not_unanswerable_when_a_mechanism_answers():
    """'Vắng' là kết luận của planner (LLM): có cơ chế tính ra đáp án từ dữ liệu thì không trả unanswerable."""
    from datamate.agent.verifier import has_data_answer

    ok = SimpleNamespace(key="number:5", typ="number")
    un = SimpleNamespace(key="UNANSWERABLE", typ="unanswerable")
    err = SimpleNamespace(key=None, typ=None)
    assert has_data_answer([err, ok]) and not has_data_answer([err, un])


def test_boolean_answer_for_count_question_fails_check():
    ctx = SimpleNamespace(snapshot_cols=[], data_end=None, data_start=None, total_revenue=None)
    c = SimpleNamespace(typ="boolean", answer=False, rows=3, period=None, code="", lint=[])
    q = {
        "question": "Đến ngày 13/03/2026 có mấy khách trả góp còn nợ kỳ quá hạn?",
        "as_of_date": "2026-03-13",
    }
    assert "type_question" in {x.name for x in candidate_checks(q, c, ctx, VerifierConfig())}
    q2 = {"question": "Doanh thu 3 tháng có tăng liên tục không?", "as_of_date": "2026-03-13"}
    assert "type_question" not in {x.name for x in candidate_checks(q2, c, ctx, VerifierConfig())}
